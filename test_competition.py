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
import re
import pickle
import numpy as np
import torchvision.transforms.functional as TF
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
        if t.dim() == 1: t = t.unsqueeze(0)
        texts = []
        for i in range(t.shape[0]):
            char_list = []
            for j in range(t.shape[1]):
                idx = t[i, j].item()
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
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation")
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
        first_checkpoint = torch.load(pth_files[0], map_location=device)['model_g_sd']
        ref_sd = {k.replace('module.', ''): v for k, v in first_checkpoint.items()}
        valid_state_dicts.append(ref_sd)
        print(f"  ✅ [REF]     {pth_files[0].name}")

        for pth in pth_files[1:5]: 
            try:
                raw_sd = torch.load(pth, map_location=device)['model_g_sd']
                clean_sd = {k.replace('module.', ''): v for k, v in raw_sd.items()}
                
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
        
        swa_dict = {k: v.clone().float() for k, v in valid_state_dicts[0].items()}
        for i in range(1, len(valid_state_dicts)):
            for k, v in valid_state_dicts[i].items():
                swa_dict[k] += v.float()
        for k in swa_dict.keys(): swa_dict[k] /= len(valid_state_dicts)
        
        model.load_state_dict(swa_dict, strict=False)
        print("✅ SWA Weights Loaded.")

    else:
        pth_files = list(ckpt_dir.glob("model_acc_*.pth"))
        if pth_files:
            pth_files.sort(key=lambda x: float(x.stem.split('_')[2]), reverse=True)
            best_ckpt = pth_files[0]
            print(f"\nLoading Best Checkpoint: {best_ckpt}")
        else:
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
    
    # --- METADATA TRACK ALIGNMENT (The Fix) ---
    track_names_ordered = []
    pkl_path = Path(args.split) / "metadata.pkl"
    if pkl_path.exists():
        with open(pkl_path, 'rb') as f:
            meta = pickle.load(f)
        for item in meta:
            match = re.search(r'(track_\d+)', str(item))
            if match:
                track_names_ordered.append(match.group(1))

    # BULLETPROOF SQUEEZE: Compress 15,000 names down to exactly 3,000 unique tracks
    track_names_ordered = list(dict.fromkeys(track_names_ordered))

    correct_plates = 0
    total_plates = 0
    
    # --- NEW: Layout Specific Trackers ---
    correct_mercosur, total_mercosur = 0, 0
    correct_brazil, total_brazil = 0, 0
    
    failures = []
    submission_lines = []

    # --- INFERENCE LOOP ---
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Evaluating ({args.mode.upper()} Mode)")
        for batch in pbar:
            lr_seqs = batch['lr_seq'].to(device)
            gt_text = batch['gt'][0] if 'gt' in batch and batch['gt'][0] else ""
            
            # --- BULLETPROOF TRACK NAME EXTRACTION ---
            track_name = None
            
            # 1. Hunt safely in the batch metadata strings (ignores heavy tensors)
            for k, v in batch.items():
                if isinstance(v, (list, tuple, str)):
                    match = re.search(r'(track_\d+)', str(v))
                    if match:
                        track_name = match.group(1)
                        break
            
            # 2. If completely stripped by dataloader, pull from exact metadata.pkl order
            if not track_name and track_names_ordered and total_plates < len(track_names_ordered):
                track_name = track_names_ordered[total_plates]
                
            # 3. Ultimate Fallback
            if not track_name:
                track_name = f"track_{total_plates:05d}"
            
            B, Seq_Len, C, H, W = lr_seqs.shape
            flat_imgs = lr_seqs.view(B * Seq_Len, C, H, W)

            with torch.amp.autocast('cuda', enabled=config.get('use_fp16', True)):
                # ==========================================
                # PASS 1: Base Resolution (0 degrees)
                # ==========================================
                output_base = model(flat_imgs, temporal_pool=True) 
                if isinstance(output_base, tuple): output_base = output_base[0]
                logits_base = output_base['logits'].view(B, Seq_Len, 7, 37).mean(dim=1)
                
                if args.tta:
                    # PASS 2 & 3: Positive Sweep (+5, +10)
                    flat_p5 = TF.rotate(flat_imgs, angle=2.5, interpolation=TF.InterpolationMode.BILINEAR)
                    out_p5 = model(flat_p5, temporal_pool=True)
                    if isinstance(out_p5, tuple): out_p5 = out_p5[0]
                    logits_p5 = out_p5['logits'].view(B, Seq_Len, 7, 37).mean(dim=1)

                    flat_p10 = TF.rotate(flat_imgs, angle=5.0, interpolation=TF.InterpolationMode.BILINEAR)
                    out_p10 = model(flat_p10, temporal_pool=True)
                    if isinstance(out_p10, tuple): out_p10 = out_p10[0]
                    logits_p10 = out_p10['logits'].view(B, Seq_Len, 7, 37).mean(dim=1)

                    # PASS 4 & 5: Negative Sweep (-5, -10)
                    flat_m5 = TF.rotate(flat_imgs, angle=-2.5, interpolation=TF.InterpolationMode.BILINEAR)
                    out_m5 = model(flat_m5, temporal_pool=True)
                    if isinstance(out_m5, tuple): out_m5 = out_m5[0]
                    logits_m5 = out_m5['logits'].view(B, Seq_Len, 7, 37).mean(dim=1)

                    flat_m10 = TF.rotate(flat_imgs, angle=-5.0, interpolation=TF.InterpolationMode.BILINEAR)
                    out_m10 = model(flat_m10, temporal_pool=True)
                    if isinstance(out_m10, tuple): out_m10 = out_m10[0]
                    logits_m10 = out_m10['logits'].view(B, Seq_Len, 7, 37).mean(dim=1)
                    
                    # --- SOTA 5-WAY LOGIT ENSEMBLING ---
                    final_logits = (logits_base + logits_p5 + logits_p10 + logits_m5 + logits_m10) / 5.0
                else:
                    final_logits = logits_base

                # Decode the consensus logits
                all_decoded_preds, all_scores = viterbi_plate_decoder(final_logits, true_converter, return_scores=True)
                
                final_pred_str = all_decoded_preds[0]
                avg_conf = all_scores[0].item()

            # --- METRICS & LOGGING ---
            if args.mode == 'val':
                is_correct = (final_pred_str == gt_text)
                
                if is_correct:
                    correct_plates += 1
                else:
                    failures.append(f"{track_name} | Pred: {final_pred_str} | GT: {gt_text}")
                
                # --- NEW: Route by Layout type ---
                if len(gt_text) >= 5:
                    if gt_text[4].isalpha(): # Mercosur uses a letter at the 5th position
                        total_mercosur += 1
                        if is_correct: correct_mercosur += 1
                    else: # Old Brazilian uses a number
                        total_brazil += 1
                        if is_correct: correct_brazil += 1
                        
                pbar.set_postfix({'SeqAcc': f"{correct_plates/(total_plates+1):.1%}"})
            else:
                # Direct write: The model has already averaged the 5 sequence frames internally!
                submission_lines.append(f"{track_name},{final_pred_str};{avg_conf:.4f}")
            
            total_plates += 1

    # 5. FINAL RESULTS
    if args.mode == 'val':
        acc = (correct_plates / total_plates) * 100.0 if total_plates > 0 else 0
        acc_merc = (correct_mercosur / total_mercosur) * 100.0 if total_mercosur > 0 else 0
        acc_braz = (correct_brazil / total_brazil) * 100.0 if total_brazil > 0 else 0
        
        print(f"\n🏆 FINAL SEQUENCE ACCURACY: {acc:.2f}% ({correct_plates}/{total_plates})")
        print(f"   🇧🇷 Old Brazilian (LLL-NNNN): {acc_braz:.2f}% ({correct_brazil}/{total_brazil})")
        print(f"   🌎 Mercosur Layout (LLL-NLNN): {acc_merc:.2f}% ({correct_mercosur}/{total_mercosur})")
        
        if failures:
            fail_path = "validation_failures_sequence.txt"
            with open(fail_path, "w") as f: 
                f.write("\n".join(failures))
            print(f"❌ Failures saved to {fail_path}")
            
    elif args.mode == 'test':
        submission_lines.sort() # Ensure numerical track order
        
        with open(args.output, 'w') as f: 
            f.write("\n".join(submission_lines))
        print(f"\n✅ Submission generated: {args.output} (Total tracks: {len(submission_lines)})")