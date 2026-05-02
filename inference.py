import argparse
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import models
from train_funcs.train_utils import decode_batch_logits, ctc_greedy_decoder, strLabelConverter

def main():
    parser = argparse.ArgumentParser(description="Single-Image Inference with TTA Bayesian Fusion")
    parser.add_argument("--config", required=True, help="Path to the config_snapshot.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to the .pth weights file")
    parser.add_argument("--folder", default="samples2infer", help="Folder containing images to infer")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation (TTA as Temporal Sequence)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. CONFIG & ARCHITECTURE ---
    with open(args.config, "r") as f: 
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    cls_loss_type = config.get('cls_loss', 'SmoothPoly1')
    true_converter = strLabelConverter(config.get('alphabet', "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"))
    
    print("\nBuilding Architecture...")
    model = models.make(config['model_g']).to(device)
    model.eval()

    # --- 2. LOAD WEIGHTS ---
    print(f"Loading Checkpoint: {args.checkpoint}")
    full_ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Priority: Ghost EMA weights > Standard Student weights
    if 'model_ghost_sd' in full_ckpt:
        print("👻 Found Ghost EMA weights! Promoting Ghost to Primary Inference Model.")
        raw_sd = full_ckpt['model_ghost_sd']
    else:
        print("👤 No Ghost weights found. Loading standard model.")
        raw_sd = full_ckpt.get('model_g_sd', full_ckpt)
        
    state_dict = {k.replace('module.', ''): v for k, v in raw_sd.items()}
    model.load_state_dict(state_dict, strict=False)

    # --- 3. IMAGE PREPROCESSING ---
    # Assuming standard 32x96 resize and simple normalization. 
    # Adjust Normalize mean/std if your dataset uses a specific RGB mean!
    transform = transforms.Compose([
        transforms.Resize((32, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    ])

    folder_path = Path(args.folder)
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = [p for p in folder_path.iterdir() if p.suffix.lower() in valid_exts]
    
    if not images:
        print(f"⚠️ No images found in {folder_path.resolve()}")
        return

    print(f"\n🚀 Starting Inference on {len(images)} images (TTA: {'Enabled' if args.tta else 'Disabled'})...\n")
    print(f"{'Filename':<30} | {'Prediction'}")
    print("-" * 50)

    # --- 4. INFERENCE LOOP ---
    with torch.no_grad():
        for img_path in images:
            img = Image.open(img_path).convert('RGB')
            base_t = transform(img).unsqueeze(0).to(device) # Shape: [1, C, H, W]
            
            # Build Sequence: If TTA is enabled, we use TTA sweeps as the Temporal (T) dimension!
            views = [base_t]
            if args.tta:
                views.append(TF.rotate(base_t, angle=2.5, interpolation=TF.InterpolationMode.BILINEAR))
                views.append(TF.rotate(base_t, angle=-2.5, interpolation=TF.InterpolationMode.BILINEAR))
                views.append(TF.rotate(base_t, angle=5.0, interpolation=TF.InterpolationMode.BILINEAR))
                views.append(TF.rotate(base_t, angle=-5.0, interpolation=TF.InterpolationMode.BILINEAR))
            
            # Stack into [B, T, C, H, W] where B=1, and T = 1 (no TTA) or 5 (with TTA)
            seqs = torch.stack(views, dim=1)
            B, T, C, H, W = seqs.shape
            
            # Flatten to [B*T, C, H, W] for the network forward pass
            flat_imgs = seqs.view(B * T, C, H, W).contiguous().to(memory_format=torch.channels_last)
            
            with torch.amp.autocast('cuda', enabled=config.get('use_fp16', True)):
                out = model(flat_imgs, epoch=100)
                
                # Handle model output variations (tuple/dict/tensor)
                if isinstance(out, tuple): out = out[0]
                logits = out['logits'] if isinstance(out, dict) else out
                
                # Reshape back to sequence [B, T, Text_Len, Num_Classes]
                _, text_len, num_classes = logits.shape
                logits_seq = logits.view(B, T, text_len, num_classes)
                
                # ========================================================
                # BAYESIAN SEQUENCE FUSION
                # ========================================================
                # 1. Safely handle models that output raw probabilities vs logits
                if torch.allclose(logits_seq.sum(dim=-1), torch.ones_like(logits_seq.sum(dim=-1)), atol=1e-2):
                    probs_seq = logits_seq # CPPD, IGTR, etc.
                else:
                    probs_seq = torch.softmax(logits_seq, dim=-1) # VSR_CURVATURE
                
                # 2. Global Bayesian Fusion (Sum of log-probabilities across the T dimension)
                log_probs = torch.log(probs_seq + 1e-8)
                fused_pseudo_logits = log_probs.sum(dim=1) # Result Shape: [B, Text_Len, Num_Classes]
                
                # ========================================================
                # PURE VISUAL DECODING
                # ========================================================
                if cls_loss_type == 'CTC':
                    preds = ctc_greedy_decoder(fused_pseudo_logits, true_converter, return_scores=False)
                else:
                    preds = decode_batch_logits(fused_pseudo_logits, true_converter)
                
                # Clean up prediction output
                pred_str = preds[0] if isinstance(preds, list) else preds
                
                print(f"📄 {img_path.name:<27} | 🎯 {pred_str}")

if __name__ == "__main__":
    main()