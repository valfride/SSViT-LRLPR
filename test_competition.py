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

# ==============================================================================
# 1. THE IMPORT FIX (Single Source of Truth)
# ==============================================================================
from train_funcs.train_utils import ctc_greedy_decoder, decode_batch_logits, strLabelConverter

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--mode", required=True, choices=['val', 'test'])
    parser.add_argument("--swa", action="store_true", help="Enable Stochastic Weight Averaging") 
    parser.add_argument("--num_swa", type=int, default=5, help="Number of top checkpoints to average")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation")
    parser.add_argument("--output", default="submission.txt")
    args = parser.parse_args()

    utils.setup_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config, "r") as f: config = yaml.load(f, Loader=yaml.FullLoader)
    
    cls_loss_type = config.get('cls_loss', 'SmoothPoly1') 
    
    print("Building Architecture...")
    # ---> SINGLE INFERENCE MODEL (Will become either the Ghost, the SWA, or the Student)
    model = models.make(config['model_g']).to(device)
    model.eval()

    # --- SWA / CHECKPOINT LOADING ---
    ckpt_dir = Path(args.checkpoints)
    
    # Helper to safely extract accuracy from filenames like 'model_acc_0.95.pth'
    def extract_acc(path):
        match = re.search(r'([\d\.]+)', path.stem)
        return float(match.group(1)) if match else 0.0

    if args.swa:
        print(f"\n⚖️   SWA ENABLED: Targeting Top {args.num_swa} Models...")
        pth_files = list(ckpt_dir.glob("model_acc_*.pth"))
        if not pth_files: 
            print("⚠️  No top models found for SWA. Falling back to last.pth")
            pth_files = [ckpt_dir / 'last.pth']
        
        pth_files.sort(key=extract_acc, reverse=True)
        
        # Calculate how many models we actually have available to safely display in the UI
        actual_models_used = min(args.num_swa, len(pth_files))
        print(f"📊 Found {len(pth_files)} checkpoints. Averaging the best {actual_models_used}.")
        
        valid_state_dicts = []
        
        # Load the reference checkpoint
        first_checkpoint = torch.load(pth_files[0], map_location=device, weights_only=False)
        
        # HONESTY UPGRADE: Always prioritize the Ghost EMA weights if they exist
        target_key = 'model_ghost_sd' if 'model_ghost_sd' in first_checkpoint else 'model_g_sd'
        print(f"🧠 SWA Target Source: '{target_key}'")
        
        ref_sd = {k.replace('module.', ''): v for k, v in first_checkpoint[target_key].items()}
        valid_state_dicts.append(ref_sd)
            
        print(f"  ✅ [REF]     {pth_files[0].name}")

        # ---> THE FIX: Dynamically slice based on the user's argument
        for pth in pth_files[1:args.num_swa]: 
            try:
                full_ckpt = torch.load(pth, map_location=device, weights_only=False)
                # Ensure the subsequent checkpoints also use the same target key
                if target_key not in full_ckpt:
                    print(f"  ⚠️ [SKIP]    {pth.name} (Missing {target_key})")
                    continue
                    
                clean_sd = {k.replace('module.', ''): v for k, v in full_ckpt[target_key].items()}
                
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
        
        # Calculate the SWA
        swa_dict = {k: v.clone().float() for k, v in valid_state_dicts[0].items()}
        for i in range(1, len(valid_state_dicts)):
            for k, v in valid_state_dicts[i].items():
                swa_dict[k] += v.float()
        for k in swa_dict.keys(): swa_dict[k] /= len(valid_state_dicts)
        
        model.load_state_dict(swa_dict, strict=False)
        print("✅ SWA Weights Successfully Loaded into the Inference Model.")

    else:
        # SINGLE CHECKPOINT LOADING
        pth_files = list(ckpt_dir.glob("model_acc_*.pth"))
        if pth_files:
            pth_files.sort(key=extract_acc, reverse=True)
            best_ckpt = pth_files[0]
            print(f"\nLoading Best Checkpoint: {best_ckpt}")
        else:
            best_ckpt = ckpt_dir / 'last.pth'
            print(f"\nLoading Fallback Checkpoint: {best_ckpt}")
            
        if not best_ckpt.exists(): raise FileNotFoundError(f"Checkpoint not found: {best_ckpt}")
        
        full_ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        
        # HONESTY UPGRADE: Prioritize the Ghost
        if 'model_ghost_sd' in full_ckpt:
            print("👻 Found Ghost EMA weights! Promoting Ghost to Primary Inference Model.")
            raw_sd = full_ckpt['model_ghost_sd']
        else:
            print("👤 No Ghost weights found. Loading standard Student model.")
            raw_sd = full_ckpt['model_g_sd']
            
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

    # --- ALPHABET SYNC ---
    true_converter = strLabelConverter(config.get('alphabet', "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"))
    
    # --- METADATA TRACK ALIGNMENT ---
    track_names_ordered = []
    pkl_path = Path(args.split) / "metadata.pkl"
    if pkl_path.exists():
        with open(pkl_path, 'rb') as f:
            meta = pickle.load(f)
        for item in meta:
            match = re.search(r'(track_\d+)', str(item))
            if match:
                track_names_ordered.append(match.group(1))

    # BULLETPROOF SQUEEZE: Compress names down to unique tracks
    track_names_ordered = list(dict.fromkeys(track_names_ordered))

    correct_plates = 0
    total_plates = 0
    
    # Layout Specific Trackers
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
            for k, v in batch.items():
                if isinstance(v, (list, tuple, str)):
                    match = re.search(r'(track_\d+)', str(v))
                    if match:
                        track_name = match.group(1)
                        break
            
            if not track_name and track_names_ordered and total_plates < len(track_names_ordered):
                track_name = track_names_ordered[total_plates]
            if not track_name:
                track_name = f"track_{total_plates:05d}"
            
            B, Seq_Len, C, H, W = lr_seqs.shape
            flat_imgs = lr_seqs.view(B * Seq_Len, C, H, W)
            
            # Match the training script's memory format for exact reproducibility
            flat_imgs = flat_imgs.contiguous().to(memory_format=torch.channels_last)

            pred_tracker = {}

            # Helper function to decode and accumulate votes
            def accumulate_votes(logits_tensor):
                if cls_loss_type == 'CTC':
                    preds, scores = ctc_greedy_decoder(logits_tensor, true_converter, return_scores=True)
                else:
                    preds = decode_batch_logits(logits_tensor, true_converter)
                    scores = F.log_softmax(logits_tensor, dim=-1).max(dim=-1)[0].sum(dim=1)
                    
                for pred_str, conf_score in zip(preds, scores):
                    if pred_str not in pred_tracker:
                        pred_tracker[pred_str] = {'votes': 0, 'confidence': 0.0}
                    pred_tracker[pred_str]['votes'] += 1
                    pred_tracker[pred_str]['confidence'] += conf_score.item()

            with torch.amp.autocast('cuda', enabled=config.get('use_fp16', True)):
                # ==========================================
                # PASS 1: Base Resolution
                # ==========================================
                output_base = model(flat_imgs, temporal_pool=True) 
                if isinstance(output_base, tuple): output_base = output_base[0]
                accumulate_votes(output_base['logits'])
                
                # ==========================================
                # TTA: HONEST AUGMENTATION ENSEMBLE
                # ==========================================
                if args.tta:
                    # Positive Sweep (+2.5)
                    flat_p2_5 = TF.rotate(flat_imgs, angle=2.5, interpolation=TF.InterpolationMode.BILINEAR)
                    out_p2_5 = model(flat_p2_5, temporal_pool=True)
                    if isinstance(out_p2_5, tuple): out_p2_5 = out_p2_5[0]
                    accumulate_votes(out_p2_5['logits'])

                    # Positive Sweep (+5.0)
                    flat_p5_0 = TF.rotate(flat_imgs, angle=5.0, interpolation=TF.InterpolationMode.BILINEAR)
                    out_p5_0 = model(flat_p5_0, temporal_pool=True)
                    if isinstance(out_p5_0, tuple): out_p5_0 = out_p5_0[0]
                    accumulate_votes(out_p5_0['logits'])

                    # Negative Sweep (-2.5)
                    flat_m2_5 = TF.rotate(flat_imgs, angle=-2.5, interpolation=TF.InterpolationMode.BILINEAR)
                    out_m2_5 = model(flat_m2_5, temporal_pool=True)
                    if isinstance(out_m2_5, tuple): out_m2_5 = out_m2_5[0]
                    accumulate_votes(out_m2_5['logits'])

                    # Negative Sweep (-5.0)
                    flat_m5_0 = TF.rotate(flat_imgs, angle=-5.0, interpolation=TF.InterpolationMode.BILINEAR)
                    out_m5_0 = model(flat_m5_0, temporal_pool=True)
                    if isinstance(out_m5_0, tuple): out_m5_0 = out_m5_0[0]
                    accumulate_votes(out_m5_0['logits'])

            # ==========================================================
            # STRING-LEVEL ENSEMBLING
            # ==========================================================
            sorted_preds = sorted(
                pred_tracker.items(), 
                key=lambda item: (item[1]['votes'], item[1]['confidence']), 
                reverse=True
            )
            
            final_pred_str = sorted_preds[0][0]
            # Average the confidence based on how many votes the winning string got
            avg_conf = sorted_preds[0][1]['confidence'] / sorted_preds[0][1]['votes']

            # --- METRICS & LOGGING ---
            if args.mode == 'val':
                is_correct = (final_pred_str == gt_text)
                
                if is_correct:
                    correct_plates += 1
                else:
                    failures.append(f"{track_name} | Pred: {final_pred_str} | GT: {gt_text} | Votes: {sorted_preds[0][1]['votes']}")
                
                # --- Layout Specific Tracking (Hyphen-safe) ---
                clean_gt = gt_text.replace('-', '')
                if len(clean_gt) >= 5:
                    if clean_gt[4].isalpha(): 
                        total_mercosur += 1
                        if is_correct: correct_mercosur += 1
                    else: 
                        total_brazil += 1
                        if is_correct: correct_brazil += 1
                        
                pbar.set_postfix({'SeqAcc': f"{correct_plates/(total_plates+1):.1%}"})
            else:
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
        # Sort numerically by extracting the track ID
        submission_lines.sort(key=lambda x: int(re.search(r'track_(\d+)', x).group(1)) if re.search(r'track_(\d+)', x) else 0)
        
        with open(args.output, 'w') as f: 
            f.write("\n".join(submission_lines))
        print(f"\n✅ Submission generated: {args.output} (Total tracks: {len(submission_lines)})")