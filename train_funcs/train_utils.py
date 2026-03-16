import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
import random
import json
import time
from pathlib import Path
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from train_funcs import register
import matplotlib.pyplot as plt
import os
import math
from collections import Counter
# ==============================================================================
# 1. HELPERS
# ==============================================================================
class strLabelConverter(object):
    def __init__(self, alphabet="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        self.alphabet = ['-'] + list(alphabet) 
        self.dict = {char: i for i, char in enumerate(self.alphabet)}
    
    def encode_list(self, text_list):
        all_result = []
        for text in text_list:
            result = [self.dict.get(char, 0) for char in text[:7]]
            while len(result) < 7: result.append(0)
            all_result.append(result)
        return torch.LongTensor(all_result)

    def decode_list(self, t):
        texts = []
        for i in range(t.shape[0]):
            char_list = [self.alphabet[idx] for idx in t[i] if idx > 0]
            texts.append(''.join(char_list))
        return texts

def viterbi_plate_decoder(batch_logits, converter, return_scores=False):
    """
    Enforces Brazilian Plate Formats (Old: LLL-NNNN, Mercosur: LLL-NLNN)
    """
    B, T, C = batch_logits.shape
    if T != 7: 
        if return_scores:
            scores = F.log_softmax(batch_logits, dim=-1).max(dim=-1)[0].sum(dim=1)
            return decode_batch_logits(batch_logits, converter), scores
        return decode_batch_logits(batch_logits, converter)

    # Masks: 0=Allowed, -inf=Forbidden
    L_mask = torch.full((C,), float('-inf'), device=batch_logits.device)
    L_mask[11:37] = 0.0 # A-Z
    
    N_mask = torch.full((C,), float('-inf'), device=batch_logits.device)
    N_mask[1:11] = 0.0 # 0-9

    path_old = torch.stack([L_mask, L_mask, L_mask, N_mask, N_mask, N_mask, N_mask])
    path_mercosur = torch.stack([L_mask, L_mask, L_mask, N_mask, L_mask, N_mask, N_mask])

    logits_old = batch_logits + path_old.unsqueeze(0)
    logits_mercosur = batch_logits + path_mercosur.unsqueeze(0)

    score_old = F.log_softmax(logits_old, dim=-1).max(dim=-1)[0].sum(dim=1)
    score_mercosur = F.log_softmax(logits_mercosur, dim=-1).max(dim=-1)[0].sum(dim=1)

    is_mercosur = (score_mercosur > score_old).unsqueeze(1).unsqueeze(2)
    final_logits = torch.where(is_mercosur, logits_mercosur, logits_old)
    
    decoded_preds = decode_batch_logits(final_logits, converter)
    
    if return_scores:
        # Get the actual log probabilities of the final chosen path
        final_scores = torch.where(is_mercosur.squeeze(-1).squeeze(-1), score_mercosur, score_old)
        return decoded_preds, final_scores
        
    return decoded_preds

def get_layout_label(text):
    if len(text) < 5: return 0
    return 1 if text[4].isalpha() else 0

def decode_batch_logits(logits, converter):
    indices = logits.argmax(dim=-1)
    decoded_preds = []
    for b in range(len(indices)):
        chars = []
        for t in range(7):
            idx = indices[b, t].item()
            char = converter.alphabet[idx]
            if char != '-': chars.append(char)
        decoded_preds.append("".join(chars))
    return decoded_preds


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, ignore_index=0): 
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        if logits.dim() == 3: logits = logits.reshape(-1, logits.shape[-1])
        if targets.dim() == 2: targets = targets.reshape(-1)
        
        # --- NUMERICAL STABILITY FIX: Force Float32 ---
        # This prevents FP16 underflow/overflow during exponentiation
        logits = logits.float()
        
        # Calculate log probabilities and probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        
        # Gather the probabilities of the true targets
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # --- SAFETY CLAMP ---
        # Prevent 'pt' from being exactly 1.0, which makes (1 - pt) exactly 0.0
        # 0.0 ** gamma can cause gradient spikes
        pt = torch.clamp(pt, min=1e-7, max=1.0 - 1e-7)

        focal_weight = (1 - pt) ** self.gamma
        loss = self.alpha * focal_weight * (-log_pt)
        
        if self.ignore_index >= 0:
            mask = targets != self.ignore_index
            if mask.sum() > 0: 
                loss = loss[mask]
            else: 
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
                
        return loss.mean()

class AdvancedAttentionLoss(nn.Module):
    def __init__(self, grid_h=16, grid_w=48, margin=0.035, monotonic_weight=2.0, ortho_weight=0.2, vertical_stretch=2.0):
        super().__init__()
        x_coords = torch.linspace(0, 1, grid_w)
        y_coords = torch.linspace(0, 1, grid_h)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        self.register_buffer('x_flat', x_grid.flatten())
        self.register_buffer('y_flat', y_grid.flatten())
        
        # Calculate physical aspect ratio once during initialization
        self.aspect_ratio = grid_w / grid_h # e.g., 48 / 16 = 3.0
        self.margin_sq = margin ** 2
        self.monotonic_weight = monotonic_weight
        self.ortho_weight = ortho_weight
        self.vertical_stretch = vertical_stretch

    def forward(self, multi_head_attn, current_epoch=0, max_decay_epoch=2):
        # multi_head_attn shape: (B, Num_Heads=16, Num_Queries=7, Num_Tokens=768)
        B, H, Q, T = multi_head_attn.shape
        
        # --- 1. Average across heads for Macro Rules ---
        avg_attn = multi_head_attn.mean(dim=1)
        
        # Calculate Center of Mass (cx, cy)
        cx = torch.sum(avg_attn * self.x_flat, dim=-1, keepdim=True)
        cy = torch.sum(avg_attn * self.y_flat, dim=-1, keepdim=True)

        # ================================================================
        # DECAY MULTIPLIER (The "Closing Iris" mechanism)
        # ================================================================
        if current_epoch < max_decay_epoch:
            progress = current_epoch / max_decay_epoch
            decay_multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            decay_multiplier = 0.0

        # ----------------------------------------------------------------
        # A. Spread Penalty (Closing Iris Ellipses + Aspect Ratio)
        # ----------------------------------------------------------------
        target_focal_offset = 0.15
        target_major_axis_length = 0.50
        expansion_factor = 0.20 # Starts 20% larger in Epoch 1
        
        current_focal_offset = target_focal_offset + (target_focal_offset * expansion_factor * decay_multiplier)
        current_major_axis_length = target_major_axis_length + (target_major_axis_length * expansion_factor * decay_multiplier)

        # Define the exact coordinates of the Top and Bottom Foci
        f1_x = cx
        f1_y = cy - current_focal_offset # Top Focus
        f2_x = cx
        f2_y = cy + current_focal_offset # Bottom Focus
        
        # Distance to Top Focus (F1)
        dx_f1 = (self.x_flat - f1_x) * self.aspect_ratio
        dy_f1 = self.y_flat - f1_y
        dist_to_f1 = torch.sqrt((dx_f1)**2 + (dy_f1)**2 + 1e-6)
        
        # Distance to Bottom Focus (F2)
        dx_f2 = (self.x_flat - f2_x) * self.aspect_ratio
        dy_f2 = self.y_flat - f2_y
        dist_to_f2 = torch.sqrt((dx_f2)**2 + (dy_f2)**2 + 1e-6)
        
        # The Elliptical Rule: The sum of the distances defines the boundary
        sum_of_distances = dist_to_f1 + dist_to_f2
        
        # Penalty applies only if the sum of distances exceeds the major axis length
        penalized_dist = F.relu(sum_of_distances - current_major_axis_length)
        
        # We square the penalty to heavily punish outliers
        spread_loss = torch.sum(avg_attn * (penalized_dist ** 2), dim=-1).mean()

        # ----------------------------------------------------------------
        # B. Monotonic Penalty (Dynamic Slack Band with Overlap Allowance)
        # ----------------------------------------------------------------
        cx_flat = cx.squeeze(-1)
        dx = cx_flat[:, 1:] - cx_flat[:, :-1]
        
        # Calculate exactly what 1 pixel is on your 48-width grid
        pixel_w = 1.0 / 48.0
        
        # --- Controlled Overlap ---
        # We allow a 1.5 pixel overlap to handle merged characters.
        # So, centers can get as close as 4.5 pixels before triggering a penalty.
        min_center_dist = 4.5 * pixel_w
        
        # We still prevent them from drifting too far apart (max 2 pixel gap)
        # 6 pixel width + 2 pixel gap = 8 pixels max center distance
        max_center_dist = 8.0 * pixel_w
        
        # Penalty 1: Triggers heavily if they overlap TOO much (dx < 4.5 pixels)
        overlap_penalty = F.relu(min_center_dist - dx)
        
        # Penalty 2: Triggers heavily if they drift too far apart (dx > 8.0 pixels)
        drift_penalty = F.relu(dx - max_center_dist)
        
        monotonic_penalty = (overlap_penalty ** 2) + (drift_penalty ** 2)
        
        # Multiply by 10.0 to ensure the squared penalty has enough tension
        monotonic_loss = monotonic_penalty.mean() * (self.monotonic_weight * 5.0)

        # ----------------------------------------------------------------
        # C. Aggressive Orthogonality Penalty (Cosine Similarity)
        # ----------------------------------------------------------------
        # 1. Normalize each head so the penalty is about % of overlap, not brightness
        norm_attn = F.normalize(multi_head_attn, p=2, dim=-1)
        
        # 2. Calculate Similarity (Overlap %)
        attn_by_query = norm_attn.transpose(1, 2) # (B, Q, H, T)
        overlap_matrix = torch.matmul(attn_by_query, attn_by_query.transpose(-2, -1))
        
        # 3. Mask out self-overlap (the diagonal)
        device = multi_head_attn.device
        identity_mask = torch.eye(H, device=device).view(1, 1, H, H)
        off_diagonal_overlap = overlap_matrix * (1.0 - identity_mask)
        
        # 4. THE MARGIN: Allow up to 50% overlap
        max_overlap = 0.50
        clumping_penalty = F.relu(off_diagonal_overlap - max_overlap)
        
        # 5. Square the penalty to create a "Wall"
        ortho_loss = (clumping_penalty ** 2).mean() * (self.ortho_weight * 20.0)

        # Return the final loss sum
        return spread_loss + ortho_loss + monotonic_loss

class ConfusionTracker:
    def __init__(self, alphabet="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        self.counts = {} 
        self.alphabet = set(alphabet)
    def update(self, preds, gts):
        for p_str, g_str in zip(preds, gts):
            if len(p_str) == len(g_str):
                for p_char, g_char in zip(p_str, g_str):
                    if p_char != g_char and p_char in self.alphabet and g_char in self.alphabet:
                        self.counts[(g_char, p_char)] = self.counts.get((g_char, p_char), 0) + 1
    def get_worst_pairs_dict(self, top_k=50):
        sorted_pairs = sorted(self.counts.items(), key=lambda item: item[1], reverse=True)
        simple_map = {}
        for (gt, pred), count in sorted_pairs[:top_k]:
            if gt not in simple_map: simple_map[gt] = pred
        return simple_map

def write_live_monitor(path, gts, preds_s, preds_t, epoch, batch_idx):
    header = f"{'GROUND TRUTH':<12} | {'STUDENT':<12} | S"
    with open(path, 'w') as f:
        f.write(f"\n{'='*20} EPOCH {epoch} | BATCH {batch_idx} {'='*20}\n")
        f.write(header + "\n" + "-" * len(header) + "\n")
        for b in range(min(len(gts), 32)):
            gt, ps = gts[b], preds_s[b]
            stat_s = "✅" if (ps == gt) else "❌"
            f.write(f"{gt:<12} | {ps:<12} | {stat_s}\n")

def robust_unpack_loss(loss):
    if isinstance(loss, (tuple, list)):
        return robust_unpack_loss(loss[0])
    return loss

# ==============================================================================
# 2. VISUALIZATION MODULES
# ==============================================================================

def visualize_feature_maps(latent_tensor, original_images, batch_idx, epoch, save_dir='./train_features', num_channels=16):
    """
    Saves a visualization of the early fusion feature maps using pure torchvision.
    """
    import os
    import torch
    import torchvision.utils as vutils
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Get the Raw LR Image (32x96)
    if original_images.dim() == 5:
        center_frame_idx = original_images.shape[1] // 2
        orig_img = original_images[0, center_frame_idx].detach().cpu()
    else:
        orig_img = original_images[0].detach().cpu()

    # Un-normalize the original image to [0, 1]
    if orig_img.min() < 0:
        orig_img = (orig_img * 0.5) + 0.5
    orig_img = torch.clamp(orig_img, 0, 1)

    # 2. Get the Latent Channels (64x192)
    feat_map = latent_tensor[0].detach().cpu() # Shape: (Channels, 64, 192)
    
    # We will grab the mean activation + the first 15 individual channels
    mean_activation = feat_map.mean(dim=0, keepdim=True) # Shape: (1, 64, 192)
    channels_to_plot = feat_map[:15] # Shape: (15, 64, 192)
    
    # Combine them into a single batch of 16 grayscale images
    # Shape becomes: (16, 1, 64, 192)
    grid_items = torch.cat([mean_activation.unsqueeze(0), channels_to_plot.unsqueeze(1)], dim=0)

    # Normalize each channel independently so features pop out
    for i in range(grid_items.shape[0]):
        c_min = grid_items[i].min()
        c_max = grid_items[i].max()
        grid_items[i] = (grid_items[i] - c_min) / (c_max - c_min + 1e-6)

    # 3. Create the Grids
    # Save the original image as its own tiny file (32x96)
    vutils.save_image(orig_img, os.path.join(save_dir, f'raw_input_ep.png'))
    
    # Save the 16 latent channels as a 4x4 grid. 
    # Because each is 64x192, the final image will be exactly 256x768 pixels.
    vutils.save_image(
        grid_items, 
        os.path.join(save_dir, f'latent_ep.png'), 
        nrow=4, 
        padding=1, 
        normalize=False # We already normalized manually
    )

def visualize_vit_attention(image_tensor, latent_tensor, attn_weights, query_texts, epoch, batch_idx, save_dir="attn_maps"):
    import os
    import cv2
    import torch
    import numpy as np
    import torchvision.utils as vutils
    import math
    
    os.makedirs(save_dir, exist_ok=True)

    # 1. PROCESS ATTENTION WEIGHTS (Native 16x48)
    if attn_weights.dim() == 4:
        attn_weights = attn_weights.mean(dim=1)
    attn = attn_weights[0].detach().cpu() # Shape: (7, 768)
    num_queries = attn.shape[0] # 7
    grid_h, grid_w = 16, 48

    # 2. PROCESS ORIGINAL LR IMAGE (Native 32x96)
    if image_tensor.dim() == 5:
        img = image_tensor[0, image_tensor.shape[1] // 2].detach().cpu() 
    else:
        img = image_tensor[0].detach().cpu()

    # Un-normalize LR image to [0, 1]
    if img.min() < 0: 
        img = (img * 0.5) + 0.5 
    img_lr_tensor = torch.clamp(img, 0, 1) # (3, 32, 96)
    
    # Convert to numpy for OpenCV drawing (H, W, C) in [0, 255]
    img_lr_np = (img_lr_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    lr_h, lr_w = img_lr_np.shape[:2] # 32x96

    # --- CALCULATE THE CLOSING IRIS MATH ---
    max_decay_epoch = 50
    if epoch < max_decay_epoch:
        progress = epoch / max_decay_epoch
        decay_multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        decay_multiplier = 0.0

    target_focal_offset = 0.15
    target_major_axis_length = 0.50 
    expansion_factor = 0.2 
    
    current_focal_offset = target_focal_offset + (target_focal_offset * expansion_factor * decay_multiplier)
    current_major_axis_length = target_major_axis_length + (target_major_axis_length * expansion_factor * decay_multiplier)

    a = current_major_axis_length / 2.0
    c = current_focal_offset
    b_sq = max(0, a**2 - c**2)
    minor_axis_length = math.sqrt(b_sq) * 2.0

    aspect_ratio = grid_w / grid_h  # 3.0
    logical_height = current_major_axis_length
    logical_width = minor_axis_length / aspect_ratio 

    # Map to LR Image Coordinates (32x96)
    major_px_y = int((logical_height / 2.0) * lr_h)
    minor_px_x = int((logical_width / 2.0) * lr_w)
    axes_length = (minor_px_x, major_px_y)

    x_coords = np.linspace(0, 1, grid_w)
    y_coords = np.linspace(0, 1, grid_h)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    # 3. BUILD THE GRID ITEMS
    # We will collect tensors of shape (3, 32, 96)
    grid_items = [img_lr_tensor]

    for i in range(num_queries):
        attn_map = attn[i].view(grid_h, grid_w).numpy()
        
        # Calculate Center of Mass [0 to 1]
        a_norm = attn_map / (attn_map.sum() + 1e-6)
        cx = np.sum(a_norm * x_grid)
        cy = np.sum(a_norm * y_grid)
        center_px = (int(cx * lr_w), int(cy * lr_h))
        
        # Resize attention map to LR size using NEAREST
        attn_map_resized = cv2.resize(attn_map, (lr_w, lr_h), interpolation=cv2.INTER_NEAREST)
        
        # Normalize and apply colormap
        attn_map_norm = (attn_map_resized - attn_map_resized.min()) / (attn_map_resized.max() - attn_map_resized.min() + 1e-6)
        attn_heatmap = (attn_map_norm * 255).astype(np.uint8)
        attn_color = cv2.applyColorMap(attn_heatmap, cv2.COLORMAP_HOT)
        
        # Convert BGR to RGB
        attn_color = cv2.cvtColor(attn_color, cv2.COLOR_BGR2RGB)
        
        # Blend original image and heatmap
        blended = cv2.addWeighted(img_lr_np, 0.4, attn_color, 0.6, 0)
        
        # Draw the red ellipse (1 pixel thick on a 32x96 image)
        cv2.ellipse(blended, center_px, axes_length, 0, 0, 360, (255, 0, 0), 1)
        
        # Convert back to tensor [0, 1]
        blended_tensor = torch.from_numpy(blended).permute(2, 0, 1).float() / 255.0
        grid_items.append(blended_tensor)

    # Stack all items: 1 original + 7 queries = 8 images
    grid_tensor = torch.stack(grid_items) # Shape: (8, 3, 32, 96)
    
    # Save as a single image grid (2 rows of 4)
    vutils.save_image(
        grid_tensor, 
        os.path.join(save_dir, f'attn_ep.png'), 
        nrow=4, 
        padding=1, 
        normalize=False
    )

# ==============================================================================
# 3. TRAINING LOOP
# ==============================================================================
@register('SROCR_TRAIN')
def SROCR_TRAIN(train_loader, model_g, model_d, optimizer_g, optimizer_d, loss_fn, config, **kwargs):
    device = next(model_g.parameters()).device
    true_converter = strLabelConverter(config.get('alphabet', "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    use_fp16 = config.get('use_fp16', False)
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16) 
    pbar = tqdm(train_loader, leave=False)
    
    save_root = kwargs.get('save_path', Path('.'))
    
    # Removed 'layout_acc' from tracking
    loss_stats = {'total': [], 'cls': [], 'spread': []}
    acc_seq_accum = []
    acc_char_accum = [] 
    
    current_epoch = kwargs.get('epoch', 0)
    
    loss_fn_spatial = FocalLoss(gamma=3.0, ignore_index=0).to(device)
    loss_fn_spread = AdvancedAttentionLoss(grid_h=16,
                                        grid_w=48,
                                        margin=0.035,
                                        monotonic_weight=2.0,
                                        ortho_weight=0.2).to(device)
    epoch_tracker = ConfusionTracker()

    for batch_idx, batch in enumerate(pbar):
        if batch is None: continue

        lr_batch = batch['lr'].to(device, non_blocking=True)
        text_label = batch['gt'] 
        true_targets = true_converter.encode_list(text_label).to(device)
        
        # REMOVED: layout_targets = ...
        
        optimizer_g.zero_grad()
            
        with torch.amp.autocast('cuda', enabled=use_fp16):

            # 1. Forward Pass
            preds_lr = model_g(
                lr_batch, 
                temporal_pool=True, 
                tgt=true_targets, 
                epoch=current_epoch,
                return_attn=True 
            )

            if isinstance(preds_lr, (tuple, list)): preds_lr = preds_lr[0]

            # 2. Character Classification Loss 
            loss_cls_lr = loss_fn_spatial(preds_lr['logits'], true_targets)
            if loss_cls_lr.dim() > 0: loss_cls_lr = loss_cls_lr.mean()
            
            # REMOVED: loss_layout = ...

            # 3. Attention Penalties (Spread + Sequence + Orthogonality)
            loss_spread = 0.0
            if preds_lr['attn_maps'] is not None:
                loss_spread = loss_fn_spread(
                    preds_lr['attn_maps'], 
                    current_epoch=current_epoch, 
                    max_decay_epoch=2 
                )

            # 4. Total Loss (Layout removed)
            total_loss = loss_cls_lr + loss_spread

        if not torch.isfinite(total_loss):
            print(f"⚠️ Warning: Non-finite loss at batch {batch_idx}. Skipping.")
            continue

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer_g)
        scaler.step(optimizer_g)
        scaler.update()

        with torch.no_grad():
            loss_stats['total'].append(total_loss.item())
            loss_stats['cls'].append(loss_cls_lr.item()) 
            loss_stats['spread'].append(loss_spread.item())
            
            # REMOVED: layout_preds, acc_l, and loss_stats['layout_acc']
            
            decoded_s = decode_batch_logits(preds_lr['logits'], true_converter)
            
            if batch_idx % 10 == 0:
                if 'latent_lr' in preds_lr:
                    visualize_feature_maps(
                        latent_tensor=preds_lr['latent_lr'], 
                        original_images=lr_batch, 
                        batch_idx=batch_idx,
                        epoch=current_epoch,
                        save_dir=save_root / 'train_features'
                    )
                
                if 'attn_maps' in preds_lr and preds_lr['attn_maps'] is not None:
                    sample_pr = decoded_s[0]
                    visualize_vit_attention(
                        image_tensor=lr_batch,             
                        latent_tensor=preds_lr['latent_lr'], # <--- ADD THIS LINE
                        attn_weights=preds_lr['attn_maps'],
                        query_texts=sample_pr,             
                        epoch=current_epoch,
                        batch_idx=batch_idx,
                        save_dir=save_root / 'train_features'
                    )

            acc_s = sum([1 for p, t in zip(decoded_s, text_label) if p == t]) / len(text_label)
            acc_seq_accum.append(acc_s)
            
            total_chars = sum(len(t) for t in text_label)
            correct_chars = sum(sum(1 for pc, tc in zip(p, t) if pc == tc) for p, t in zip(decoded_s, text_label))
            acc_c = correct_chars / max(total_chars, 1)
            acc_char_accum.append(acc_c)
            
            epoch_tracker.update(decoded_s, text_label) 

            if batch_idx % 5 == 0:
                # REMOVED: 'Lyt%' from progress bar
                pbar.set_postfix({
                    'Loss': f"{np.mean(loss_stats['total']):.4f}",
                    'Spd': f"{np.mean(loss_stats['spread']):.4f}",
                    'Cls': f"{np.mean(loss_stats['cls']):.4f}",
                    'Seq%': f"{np.mean(acc_seq_accum):.1%}",
                    'Chr%': f"{np.mean(acc_char_accum):.1%}", 
                })
            
            if batch_idx % 50 == 0:
                sample_gt = text_label[0]
                sample_pr = decoded_s[0]
                tqdm.write(f"🔎 [B{batch_idx}] GT: {sample_gt} | Pr: {sample_pr}")

            if batch_idx % 10 == 0:
                write_live_monitor(save_root / 'live_monitor.txt', text_label, decoded_s, decoded_s, current_epoch, batch_idx)
        
    total_failures = epoch_tracker.get_worst_pairs_dict(top_k=50) 
    with open(save_root / 'confusion_stats.json', 'w') as f: json.dump(total_failures, f, indent=4)
    
    return np.mean(loss_stats['total']) if loss_stats['total'] else 0.0

@register('SROCR_VAL')
def SROCR_VAL(val_loader, model_g, model_d, loss_fn, config, **kwargs):
    model_g.eval() 
    device = next(model_g.parameters()).device
    true_converter = strLabelConverter(config.get('alphabet', "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    
    correct_sequences = 0
    total_sequences = 0
    
    correct_chars = 0
    total_chars = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            lr_seqs = batch['lr_seq'].to(device, non_blocking=True)
            text_labels = batch['gt']     

            B, Seq_Len, C, H, W = lr_seqs.shape
            flat_imgs = lr_seqs.view(B * Seq_Len, C, H, W)

            with torch.amp.autocast('cuda', enabled=config.get('use_fp16', False)):
                output = model_g(flat_imgs, temporal_pool=True) 
                if isinstance(output, (tuple, list)): output = output[0]
                logits = output['logits']

            # --- Extract both text AND confidence scores ---
            all_decoded_preds, all_scores = viterbi_plate_decoder(logits, true_converter, return_scores=True)

            # --- SMART MAJORITY VOTE LOGIC ---
            for b in range(B):
                seq_preds = all_decoded_preds[b * Seq_Len : (b + 1) * Seq_Len]
                seq_scores = all_scores[b * Seq_Len : (b + 1) * Seq_Len]
                
                # Dictionary to track votes and cumulative confidence
                pred_tracker = {}
                
                for pred_str, conf_score in zip(seq_preds, seq_scores):
                    if pred_str not in pred_tracker:
                        pred_tracker[pred_str] = {'votes': 0, 'confidence': 0.0}
                    
                    pred_tracker[pred_str]['votes'] += 1
                    # Sum the log-probabilities (closer to 0 is better)
                    pred_tracker[pred_str]['confidence'] += conf_score.item()
                
                # Sort first by 'votes' (Descending), then by 'confidence' (Descending)
                sorted_preds = sorted(
                    pred_tracker.items(), 
                    key=lambda item: (item[1]['votes'], item[1]['confidence']), 
                    reverse=True
                )
                
                # The winner is the first item in the sorted list
                winning_prediction = sorted_preds[0][0]
                
                gt = text_labels[b]
                
                total_sequences += 1
                if winning_prediction == gt: 
                    correct_sequences += 1
                
                total_chars += len(gt)
                correct_chars += sum(1 for pc, gc in zip(winning_prediction, gt) if pc == gc)
            
    acc_seq = correct_sequences / total_sequences if total_sequences else 0.0
    acc_char = correct_chars / total_chars if total_chars else 0.0
    
    print(f"\n{'='*30} SEQUENCE EVALUATION {'='*30}")
    print(f"Total Sequences: {total_sequences}")
    print(f"Sequence Accuracy:  {acc_seq:.2%}")
    print(f"Character Accuracy: {acc_char:.2%}")
    print(f"{'='*81}\n")
    
    return 0.0, acc_seq, 0.0
