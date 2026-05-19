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
import math
from collections import defaultdict

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
    parser.add_argument("--in_images", type=int, default=None, help="Override the number of temporal frames")
    parser.add_argument("--fusion", type=str, default="logit_average", 
                        choices=["bayes", "average", "majority", "logit_average"], 
                        help="Temporal fusion strategy")
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
        pth_files = list(ckpt_dir.glob("*acc_*.pth"))
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
        pth_files = list(ckpt_dir.glob("*acc_*.pth"))
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
    
    if args.in_images is not None:
        config['val_dataset']['wrapper']['args']['in_images'] = args.in_images
        print(f"🔄 TEMPORAL OVERRIDE: Forced val_dataset in_images to {args.in_images}")

    dataset_spec = config['val_dataset']['dataset']
    dataset_spec['args']['path_split'] = args.split
    dataset_spec['args']['phase'] = 'test'
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
    correct_plates = 0 
    correct_6plus = 0  
    correct_5plus = 0  
    total_plates = 0
    confidence_tracking = []
    
    # --- INFERENCE LOOP ---
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Evaluating ({args.mode.upper()} Mode | Fusion: {args.fusion.upper()})")
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
            flat_imgs = flat_imgs.contiguous().to(memory_format=torch.channels_last)

            # Raw logits from all views (base + TTA)
            view_logits = []

            with torch.amp.autocast('cuda', enabled=config.get('use_fp16', True)):
                # PASS 1: Base Resolution
                output_base = model(flat_imgs, epoch=100)
                if isinstance(output_base, tuple): output_base = output_base[0]
                view_logits.append(output_base['logits'])

                # TTA: HONEST AUGMENTATION ENSEMBLE
                if args.tta:
                    # Positive Sweep (+2.5)
                    flat_p2_5 = TF.rotate(flat_imgs, angle=2.5, interpolation=TF.InterpolationMode.BILINEAR)
                    out_p2_5 = model(flat_p2_5, epoch=100)
                    if isinstance(out_p2_5, tuple): out_p2_5 = out_p2_5[0]
                    view_logits.append(out_p2_5['logits'])

                    # Negative Sweep (-2.5)
                    flat_m2_5 = TF.rotate(flat_imgs, angle=-2.5, interpolation=TF.InterpolationMode.BILINEAR)
                    out_m2_5 = model(flat_m2_5, epoch=100)
                    if isinstance(out_m2_5, tuple): out_m2_5 = out_m2_5[0]
                    view_logits.append(out_m2_5['logits'])

            # ==========================================================
            # TEMPORAL & TTA FUSION (BASED ON FLAG)
            # ==========================================================
            if args.fusion in ['bayes', 'average', 'logit_average']:
                accumulated_log_probs = 0.0 
                accumulated_raw_probs = 0.0 
                accumulated_logits = 0.0

                for logits_tensor in view_logits:
                    _, num_chars, num_classes = logits_tensor.shape
                    logits_seq = logits_tensor.view(B, Seq_Len, num_chars, num_classes)
                    
                    if torch.allclose(logits_seq.sum(dim=-1), torch.ones_like(logits_seq.sum(dim=-1)), atol=1e-2):
                        probs_seq = logits_seq
                        log_probs = torch.log(probs_seq + 1e-8) 
                    else:
                        probs_seq = torch.softmax(logits_seq, dim=-1)
                        log_probs = F.log_softmax(logits_seq, dim=-1)

                    if args.fusion == 'bayes':
                        # Summing Log-Probs (Sharpens confidence, ruins gap)
                        temporal_fused = log_probs.sum(dim=1) 
                        accumulated_log_probs = accumulated_log_probs + temporal_fused
                    elif args.fusion == 'average':
                        # Averaging Probs (Changes argmax, drops accuracy)
                        temporal_fused = probs_seq.mean(dim=1)
                        accumulated_raw_probs = accumulated_raw_probs + temporal_fused
                    elif args.fusion == 'logit_average':
                        # THE ORIGINAL MAGIC: Averaging raw logits (Log-Linear Pooling)
                        # Retains Bayes accuracy, retains original confidence gap!
                        temporal_fused = logits_seq.mean(dim=1)
                        accumulated_logits = accumulated_logits + temporal_fused

                # TTA FUSION
                if args.fusion == 'bayes':
                    fused_pseudo_logits = accumulated_log_probs / len(view_logits)
                elif args.fusion == 'average':
                    fused_pseudo_logits = torch.log((accumulated_raw_probs / len(view_logits)) + 1e-8)
                elif args.fusion == 'logit_average':
                    fused_pseudo_logits = accumulated_logits / len(view_logits)

                # DECODE (LATE FUSION)
                if cls_loss_type == 'CTC':
                    preds, scores = ctc_greedy_decoder(fused_pseudo_logits, true_converter, return_scores=True)
                else:
                    preds = decode_batch_logits(fused_pseudo_logits, true_converter)
                    scores = F.log_softmax(fused_pseudo_logits, dim=-1).max(dim=-1)[0].sum(dim=1)
                
                final_pred_str = preds[0]
                avg_conf = scores[0].item() if isinstance(scores[0], torch.Tensor) else scores[0]

            elif args.fusion == 'majority':
                # ==========================================================
                # DISCRETE EARLY DECODING (HARD VOTING)
                # ==========================================================
                all_decoded_strings = []
                all_decoded_confs = []
                
                for logits_tensor in view_logits:
                    if cls_loss_type == 'CTC':
                        frame_preds, frame_scores = ctc_greedy_decoder(logits_tensor, true_converter, return_scores=True)
                    else:
                        frame_preds = decode_batch_logits(logits_tensor, true_converter)
                        frame_scores = F.log_softmax(logits_tensor, dim=-1).max(dim=-1)[0].sum(dim=1)
                    
                    all_decoded_strings.extend(frame_preds)
                    
                    if isinstance(frame_scores, torch.Tensor):
                        all_decoded_confs.extend(frame_scores.tolist())
                    else:
                        all_decoded_confs.extend(frame_scores)
                        
                vote_counts = defaultdict(int)
                vote_confs = defaultdict(list)
                
                for s, c in zip(all_decoded_strings, all_decoded_confs):
                    vote_counts[s] += 1
                    vote_confs[s].append(c)
                    
                max_votes = max(vote_counts.values())
                tied_candidates = [s for s, count in vote_counts.items() if count == max_votes]
                
                best_candidate = None
                best_conf = -float('inf')
                
                for s in tied_candidates:
                    avg_c = sum(vote_confs[s]) / len(vote_confs[s])
                    if avg_c > best_conf:
                        best_conf = avg_c
                        best_candidate = s
                        
                final_pred_str = best_candidate
                avg_conf = best_conf

            # --- METRICS & LOGGING ---
            if args.mode == 'val':
                match_count = sum(1 for p, g in zip(final_pred_str, gt_text) if p == g)
                
                if match_count >= 5: correct_5plus += 1
                if match_count >= 6: correct_6plus += 1
                
                is_correct = (final_pred_str == gt_text)
                if is_correct:
                    correct_plates += 1
                else:
                    failures.append(f"{track_name} | Pred: {final_pred_str} | GT: {gt_text} | Conf: {avg_conf:.4f} | Matches: {match_count}")
                
                if cls_loss_type == 'CTC':
                    normalized_conf = avg_conf
                else:
                    normalized_conf = math.exp(avg_conf / max(1, len(final_pred_str)))
                    
                confidence_tracking.append({'correct': is_correct, 'conf': normalized_conf})

                pbar.set_postfix({'SeqAcc': f"{correct_plates/(total_plates+1):.1%}"})
            else:
                submission_lines.append(f"{track_name},{final_pred_str};{avg_conf:.4f}")
            
            total_plates += 1
            
    # ==========================================================
    # FINAL RESULTS
    # ==========================================================
    if args.mode == 'val':
        acc = (correct_plates / total_plates) * 100.0 if total_plates > 0 else 0
        acc_6plus = (correct_6plus / total_plates) * 100.0 if total_plates > 0 else 0
        acc_5plus = (correct_5plus / total_plates) * 100.0 if total_plates > 0 else 0
        
        acc_merc = (correct_mercosur / total_mercosur) * 100.0 if total_mercosur > 0 else 0
        acc_braz = (correct_brazil / total_brazil) * 100.0 if total_brazil > 0 else 0
        
        print(f"\n🏆 FINAL EVALUATION METRICS ({args.fusion.upper()} FUSION)")
        print(f"   Full Sequence (7/7):  {acc:.2f}% ({correct_plates}/{total_plates})")
        print(f"   Partial Match (≥6/7): {acc_6plus:.2f}% ({correct_6plus}/{total_plates})")
        print(f"   Partial Match (≥5/7): {acc_5plus:.2f}% ({correct_5plus}/{total_plates})")
        print(f"{'-'*40}")
        print(f"   🇧🇷 Old Brazilian (LLL-NNNN): {acc_braz:.2f}% ({correct_brazil}/{total_brazil})")
        print(f"   🌎 Mercosur Layout (LLL-NLNN): {acc_merc:.2f}% ({correct_mercosur}/{total_mercosur})")
        
        print(f"\n📈 RECOGNITION RATE VS. CONFIDENCE THRESHOLD")
        print(f"{'-'*65}")
        print(f"| {'Minimum Confidence':<20} | {'Retained (Coverage)':<22} | {'Accuracy':<12} |")
        print(f"|{'-'*22}|{'-'*24}|{'-'*14}|")
        
        thresholds = [0.0, 0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99]
        for thresh in thresholds:
            retained = [x for x in confidence_tracking if x['conf'] >= thresh]
            if not retained:
                continue
                
            coverage = (len(retained) / len(confidence_tracking)) * 100.0
            thresh_acc = (sum(1 for x in retained if x['correct']) / len(retained)) * 100.0
            
            print(f"| ≥ {thresh:<18.2f} | {len(retained):<6} ({coverage:>6.2f}%)         | {thresh_acc:>7.2f}%    |")
        print(f"{'-'*65}")

        correct_confs = [x['conf'] for x in confidence_tracking if x['correct']]
        incorrect_confs = [x['conf'] for x in confidence_tracking if not x['correct']]

        mean_correct = sum(correct_confs) / len(correct_confs) if correct_confs else 0.0
        mean_incorrect = sum(incorrect_confs) / len(incorrect_confs) if incorrect_confs else 0.0
        
        confidence_gap = mean_correct - mean_incorrect
        
        print(f"\n🧠 MODEL CALIBRATION (CONFIDENCE GAP)")
        print(f"{'-'*45}")
        print(f"   Mean Conf (Correct):   {mean_correct:.4f}")
        print(f"   Mean Conf (Incorrect): {mean_incorrect:.4f}")
        print(f"   Confidence Gap:        {confidence_gap:.4f}") 
        print(f"{'-'*45}")

        if failures:
            fail_path = "validation_failures_sequence.txt"
            with open(fail_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(failures))
            print(f"\nSaved {len(failures)} failures to {fail_path}")

    elif args.mode == 'test':
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write("\n".join(submission_lines))
        print(f"✅ Submission saved to {args.output}")