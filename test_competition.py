import argparse
import yaml
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import datasets
import models
import utils
import os
import numpy as np

# Import the Viterbi Decoder from your training utilities!
from train_funcs.train_utils import viterbi_plate_decoder

# ==============================================================================
# 1. CONVERTERS (Standardized)
# ==============================================================================
class strLabelConverter(object):
    """Decodes Logits to Strings"""
    def __init__(self, alphabet="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        self.alphabet = ['-'] + list(alphabet) # 0 is blank/pad
        self.dict = {char: i for i, char in enumerate(self.alphabet)}
    
    def decode(self, t):
        """
        Decodes a batch of indices [B, 7] to strings.
        """
        if t.dim() == 1: t = t.unsqueeze(0)
        texts = []
        for i in range(t.shape[0]):
            char_list = []
            for j in range(t.shape[1]):
                idx = t[i, j].item()
                # Skip blank/pad index 0, map rest
                if 0 < idx < len(self.alphabet):
                    char_list.append(self.alphabet[idx])
            texts.append(''.join(char_list))
        return texts

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--mode", required=True, choices=['val', 'test'])
    parser.add_argument("--swa", action="store_true") 
    parser.add_argument("--output", default="submission.txt")
    args = parser.parse_args()

    utils.setup_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config, "r") as f: config = yaml.load(f, Loader=yaml.FullLoader)
    
    print("Building Architecture...")
    model = models.make(config['model_g']).to(device)
    model.eval()

    # --- SWA / CHECKPOINT LOADING ---
    ckpt_dir = Path(args.checkpoints)
    if args.swa:
        print("\n⚖️   SWA ENABLED: Averaging Top Models...")
        pth_files = list(ckpt_dir.glob("model_acc_*.pth"))
        if not pth_files: 
            print("⚠️  No top models found for SWA. Falling back to last.pth")
            pth_files = [ckpt_dir / 'last.pth']
        
        pth_files.sort(key=lambda x: float(x.stem.split('_')[2]) if '_' in x.stem else 0.0, reverse=True)
        
        valid_state_dicts = []
        
        # Load the first model to establish the "Gold Standard" for keys AND shapes
        first_checkpoint = torch.load(pth_files[0], map_location=device)['model_g_sd']
        ref_sd = {k.replace('module.', ''): v for k, v in first_checkpoint.items()}
        valid_state_dicts.append(ref_sd)
        print(f"  ✅ [REF]     {pth_files[0].name}")

        for pth in pth_files[1:5]: # Top 5 max
            try:
                raw_sd = torch.load(pth, map_location=device)['model_g_sd']
                clean_sd = {k.replace('module.', ''): v for k, v in raw_sd.items()}
                
                # 🛑 SAFETY CHECK: Keys AND Shapes must match
                is_compatible = True
                if set(clean_sd.keys()) != set(ref_sd.keys()):
                    is_compatible = False
                else:
                    for k in ref_sd.keys():
                        if clean_sd[k].shape != ref_sd[k].shape:
                            is_compatible = False
                            break
                
                if is_compatible:
                    print(f"  ✅ [INCLUDE] {pth.name}")
                    valid_state_dicts.append(clean_sd)
                else:
                    print(f"  ⚠️ [SKIP]    {pth.name} (Shape/Arch Mismatch)")
            except Exception as e:
                print(f"  ❌ [ERROR]   {pth.name}: {e}")

        if not valid_state_dicts: raise RuntimeError("No compatible checkpoints found!")
        
        # Average weights
        swa_dict = {k: v.clone().float() for k, v in valid_state_dicts[0].items()}
        for i in range(1, len(valid_state_dicts)):
            for k, v in valid_state_dicts[i].items():
                swa_dict[k] += v.float()
        for k in swa_dict.keys(): swa_dict[k] /= len(valid_state_dicts)
        
        model.load_state_dict(swa_dict, strict=False)
        print("✅ SWA Weights Loaded.")

    else:
        # 1. Try to find the best accuracy model first
        pth_files = list(ckpt_dir.glob("model_acc_*.pth"))
        if pth_files:
            # Sort by the accuracy float in the filename
            pth_files.sort(key=lambda x: float(x.stem.split('_')[2]), reverse=True)
            best_ckpt = pth_files[0]
            print(f"\nLoading Best Checkpoint: {best_ckpt}")
        else:
            # 2. Fallback to last.pth if no best models are found
            best_ckpt = ckpt_dir / 'last.pth'
            
        print(f"\nLoading Single Checkpoint: {best_ckpt}")
        if not best_ckpt.exists(): raise FileNotFoundError(f"Checkpoint not found: {best_ckpt}")
        
        raw_sd = torch.load(best_ckpt, map_location=device)['model_g_sd']
        state_dict = {k.replace('module.', ''): v for k, v in raw_sd.items()}
        model.load_state_dict(state_dict, strict=False)

    # --- DATASET ---
    print(f"\nPreparing Data from {args.split}...")
    dataset_spec = config['val_dataset']['dataset']
    dataset_spec['args']['path_split'] = args.split
    dataset_spec['args']['phase'] = args.mode 
    base_dataset = datasets.make(dataset_spec)
    
    wrapper_spec = config['val_dataset']['wrapper']
    wrapper_spec['args']['dataset'] = base_dataset
    wrapper_spec['args']['test'] = True 
    val_dataset = datasets.make(wrapper_spec)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, 
        num_workers=4, pin_memory=True, collate_fn=val_dataset.collate_fn
    )

    true_converter = strLabelConverter(config.get('alphabet', "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    
    correct_plates = 0
    total_plates = 0
    failures = []
    submission_lines = []

    # --- INFERENCE LOOP (ALIGNED WITH SROCR_VAL) ---
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Evaluating ({args.mode.upper()} Mode)")
        for batch in pbar:
            # 1. Fetch the sequence tensor (NOT 'lr')
            lr_seqs = batch['lr_seq'].to(device)
            
            # Ground truth for val mode
            gt_text = batch['gt'][0] if 'gt' in batch and batch['gt'][0] else ""
            
            # Safely extract track name 
            # Sequence collate_fn usually returns a list of lists for names: [['track1_f1', 'track1_f2', ...]]
            try:
                raw_name = batch['names'][0][0] if 'names' in batch else batch['name'][0][0]
            except:
                # Fallback if structure is flat
                raw_name = batch['names'][0] if 'names' in batch else batch['name'][0]
                
            track_name = raw_name.rsplit('_f', 1)[0]
            
            B, Seq_Len, C, H, W = lr_seqs.shape
            
            # 2. Flatten for the model forward pass
            flat_imgs = lr_seqs.view(B * Seq_Len, C, H, W)

            with torch.amp.autocast('cuda', enabled=config.get('use_fp16', True)):
                output = model(flat_imgs, temporal_pool=True) 
                if isinstance(output, tuple): output = output[0]
                logits = output['logits']

            # 3. Use the Viterbi Decoder (Enforcing Plate Rules)
            all_decoded_preds, all_scores = viterbi_plate_decoder(logits, true_converter, return_scores=True)

            # 4. SMART MAJORITY VOTE LOGIC (Exactly like validation)
            pred_tracker = {}
            for pred_str, conf_score in zip(all_decoded_preds, all_scores):
                if pred_str not in pred_tracker:
                    pred_tracker[pred_str] = {'votes': 0, 'confidence': 0.0}
                pred_tracker[pred_str]['votes'] += 1
                pred_tracker[pred_str]['confidence'] += conf_score.item()
            
            sorted_preds = sorted(
                pred_tracker.items(), 
                key=lambda item: (item[1]['votes'], item[1]['confidence']), 
                reverse=True
            )
            
            # The winner!
            final_pred_str = sorted_preds[0][0]
            
            # Approximate sequence confidence (average confidence of the winning votes)
            avg_conf = sorted_preds[0][1]['confidence'] / sorted_preds[0][1]['votes'] 

            # --- METRICS & LOGGING ---
            total_plates += 1
            if args.mode == 'val':
                if final_pred_str == gt_text:
                    correct_plates += 1
                else:
                    failures.append(f"{track_name} | Pred: {final_pred_str} | GT: {gt_text}")
                pbar.set_postfix({'SeqAcc': f"{correct_plates/total_plates:.1%}"})
            else:
                submission_lines.append(f"{track_name},{final_pred_str};{avg_conf:.4f}")

    # 5. FINAL RESULTS
    if args.mode == 'val':
        acc = (correct_plates / total_plates) * 100.0 if total_plates > 0 else 0
        print(f"\n🏆 FINAL SEQUENCE ACCURACY: {acc:.2f}% ({correct_plates}/{total_plates})")
        
        if failures:
            fail_path = "validation_failures_sequence.txt"
            with open(fail_path, "w") as f: 
                f.write("\n".join(failures))
            print(f"❌ Failures saved to {fail_path}")
            
    elif args.mode == 'test':
        with open(args.output, 'w') as f: 
            f.write("\n".join(submission_lines))
        print(f"\n✅ Submission generated: {args.output}")