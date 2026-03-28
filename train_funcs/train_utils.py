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
import higher # Add this at the top of train_utils.py
from torch.nn.attention import SDPBackend


import kornia.augmentation as K
import torch.nn.functional as F
import random
import torch.nn as nn

# 1. We build a custom module that safely downscales and upscales the whole batch
class GPULanczosSimulator(nn.Module):
    def __init__(self, scale_min=0.16, scale_max=0.18):
        super().__init__()
        self.scale_min = scale_min
        self.scale_max = scale_max

    def forward(self, x):
        # x is (B, C, H, W)
        scale = random.uniform(self.scale_min, self.scale_max)
        down_h, down_w = max(1, int(x.shape[2] * scale)), max(1, int(x.shape[3] * scale))
        
        # Downscale
        x_down = F.interpolate(x, size=(down_h, down_w), mode='bicubic', align_corners=False)
        # Upscale back to original
        x_up = F.interpolate(x_down, size=(x.shape[2], x.shape[3]), mode='bicubic', align_corners=False)
        return x_up

# 2. Your new, safe pipeline
gpu_degrader = K.AugmentationSequential(
    K.RandomGaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0), p=1.0),
    GPULanczosSimulator(scale_min=0.20, scale_max=0.30), # Replaces the dangerous crop!
    K.ColorJitter(brightness=0.1, contrast=0.1, p=0.6),
    K.RandomJPEG(jpeg_quality=(95, 100), p=0.8),
    data_keys=["input"]
)
# ==============================================================================
# 1. HELPERS
# ==============================================================================
def is_main_process():
    """Independent helper to check DDP status without importing from train_gan."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0

class strLabelConverter(object):
    def __init__(self, alphabet="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"):
        # Index 0 is EOS ($). Index 38 is now PAD (#).
        self.alphabet = ['$'] + list(alphabet) + ['#'] 
        self.dict = {char: i for i, char in enumerate(self.alphabet)}
    
    def encode_cppd(self, text_list, max_len=7):
        char_tgts, node_tgts = [], []
        for text in text_list:
            # 1. Standard Targets (with EOS and ignore_index padding)
            chars = [self.dict.get(c, 0) for c in text[:max_len]]
            chars.append(0) 
            char_tgt = chars + [100] * (max_len + 1 - len(chars))
            char_tgts.append(char_tgt)
            
            # 2. Node Graph (Character Counting + Position Mask)
            text_char_node = [0] * len(self.alphabet)
            text_char_node[0] = 1 
            for c in chars[:-1]: text_char_node[c] += 1
            text_pos_node = [1] * len(chars) + [0] * (max_len + 1 - len(chars))
            node_tgts.append(text_char_node + text_pos_node)
            
        return torch.LongTensor(char_tgts), torch.LongTensor(node_tgts)

    def encode_poly(self, text_list, max_len=7):
        """Creates exactly 7 targets for the Polygonal Spotter (No EOS token)"""
        all_result = []
        for text in text_list:
            # Get up to 7 characters
            result = [self.dict.get(char, 0) for char in text[:max_len]]
            # Pad with 0s if shorter than 7
            while len(result) < max_len: 
                result.append(0)
            
            # -> REMOVED THE 8th TOKEN APPEND HERE <-
            
            all_result.append(result)
        return torch.LongTensor(all_result)

    def encode_ote(self, text_list, max_len=7):
        BOS = 37
        EOS = 0
        PAD = 38
        result = []
        for text in text_list:
            seq = [BOS]
            for char in text[:max_len]:
                seq.append(self.dict.get(char, 0))
            seq.append(EOS)
            while len(seq) < max_len + 2:
                seq.append(PAD)
            result.append(seq)
        return torch.LongTensor(result)

    def encode_list(self, text_list):
        all_result = []
        for text in text_list:
            result = [self.dict.get(char, 0) for char in text[:7]]
            # while len(result) < 7: result.append(0)
            all_result.append(result)
        return torch.LongTensor(all_result)

    def decode_list(self, t):
        texts = []
        for i in range(t.shape[0]):
            char_list = [self.alphabet[idx] for idx in t[i] if idx > 0]
            texts.append(''.join(char_list))
        return texts
    
    def encode_variable(self, text_list, max_len=12):
        # FIX: Use 0 ('-') as EOS so it doesn't overwrite the letter 'Z'!
        EOS_INDEX = 0   
        PAD_INDEX = 38  # The index PyTorch will ignore
        
        all_targets = []
        for text in text_list:
            # 1. The String
            target = [self.dict.get(char, 0) for char in text[:max_len]]
            
            # 2. The End (EOS)
            if len(target) < max_len:
                target.append(EOS_INDEX)
                
            # 3. The Padding (PAD)
            while len(target) < max_len:
                target.append(PAD_INDEX)
                
            all_targets.append(target)
            
        return torch.LongTensor(all_targets)

def decode_batch_logits(logits, converter):
    indices = logits.argmax(dim=-1).cpu().numpy() 
    
    decoded_preds = []
    for b in range(len(indices)):
        chars = []
        for t in range(indices.shape[1]):
            idx = indices[b, t] 
            
            if idx >= len(converter.alphabet):
                continue
                
            char = converter.alphabet[idx]
            
            # --- THE FIX: Ignore Padding, Break on EOS ---
            if char == '#': 
                continue
            if char == '$': 
                break 
                
            chars.append(char)
        decoded_preds.append("".join(chars))
    return decoded_preds

def differentiable_iou_repulsion(corners, margin=0.10):
    """
    corners: The output from your Spotter, shape (B, 7, 4, 2) 
            where 7 is the number of lassos, 4 is the corners, 2 is (X,Y)
    margin: The maximum allowed IoU overlap (e.g., 0.10 means 10% overlap is tolerated)
    """
    B, L, _, _ = corners.shape
    
    # 1. Convert the 4-point polygons into strict Bounding Boxes (xmin, ymin, xmax, ymax)
    # This operation is fully differentiable!
    x_coords = corners[..., 0] # (B, 7, 4)
    y_coords = corners[..., 1] # (B, 7, 4)
    
    xmin = x_coords.min(dim=-1)[0] # (B, 7)
    xmax = x_coords.max(dim=-1)[0] # (B, 7)
    ymin = y_coords.min(dim=-1)[0] # (B, 7)
    ymax = y_coords.max(dim=-1)[0] # (B, 7)
    
    # Calculate the area of each box
    areas = (xmax - xmin) * (ymax - ymin) # (B, 7)
    
    # 2. Prepare for Pairwise Comparison (Every box vs Every box)
    # Add dimensions to broadcast: (B, 7, 1) and (B, 1, 7)
    xmin1, xmin2 = xmin.unsqueeze(2), xmin.unsqueeze(1)
    xmax1, xmax2 = xmax.unsqueeze(2), xmax.unsqueeze(1)
    ymin1, ymin2 = ymin.unsqueeze(2), ymin.unsqueeze(1)
    ymax1, ymax2 = ymax.unsqueeze(2), ymax.unsqueeze(1)
    areas1, areas2 = areas.unsqueeze(2), areas.unsqueeze(1)
    
    # 3. Calculate the Intersection Box
    inter_xmin = torch.max(xmin1, xmin2)
    inter_xmax = torch.min(xmax1, xmax2)
    inter_ymin = torch.max(ymin1, ymin2)
    inter_ymax = torch.min(ymax1, ymax2)
    
    # Width and Height of intersection (must be >= 0)
    inter_w = torch.clamp(inter_xmax - inter_xmin, min=0)
    inter_h = torch.clamp(inter_ymax - inter_ymin, min=0)
    inter_area = inter_w * inter_h
    
    # 4. Calculate IoU
    union_area = areas1 + areas2 - inter_area
    # Add epsilon to prevent division by zero
    iou = inter_area / (union_area + 1e-6) 
    
    # 5. Apply the Margin and Mask
    # We only penalize if IoU is GREATER than the margin
    penalty = F.relu(iou - margin)
    
    # We must ignore self-comparisons (Lasso 1 vs Lasso 1 will always have IoU = 1.0!)
    identity_mask = torch.eye(L, device=corners.device).unsqueeze(0).bool()
    penalty = penalty.masked_fill(identity_mask, 0.0)
    
    # Average the penalty across the batch
    return penalty.mean()


def viterbi_plate_decoder(batch_logits, converter, return_scores=False):
    B, T, C = batch_logits.shape
    
    # 1. Slice out the 8th EOS node for CPPD so Viterbi can cleanly align the 7 characters
    plate_logits = batch_logits[:, :7, :] if T >= 7 else batch_logits
    
    if plate_logits.shape[1] != 7: 
        if return_scores:
            scores = F.log_softmax(batch_logits, dim=-1).max(dim=-1)[0].sum(dim=1)
            return decode_batch_logits(batch_logits, converter), scores
        return decode_batch_logits(batch_logits, converter)

    L_mask = torch.full((C,), float('-inf'), device=batch_logits.device)
    L_mask[11:37] = 0.0 
    
    N_mask = torch.full((C,), float('-inf'), device=batch_logits.device)
    N_mask[1:11] = 0.0 

    path_old = torch.stack([L_mask, L_mask, L_mask, N_mask, N_mask, N_mask, N_mask])
    path_mercosur = torch.stack([L_mask, L_mask, L_mask, N_mask, L_mask, N_mask, N_mask])

    logits_old = plate_logits + path_old.unsqueeze(0)
    logits_mercosur = plate_logits + path_mercosur.unsqueeze(0)

    score_old = F.log_softmax(logits_old, dim=-1).max(dim=-1)[0].sum(dim=1)
    score_mercosur = F.log_softmax(logits_mercosur, dim=-1).max(dim=-1)[0].sum(dim=1)

    is_mercosur = (score_mercosur > score_old).unsqueeze(1).unsqueeze(2)
    final_plate_logits = torch.where(is_mercosur, logits_mercosur, logits_old)
    
    # 2. Stitch the EOS node back on so decode_batch_logits can stop gracefully
    if T > 7:
        final_logits = torch.cat([final_plate_logits, batch_logits[:, 7:, :]], dim=1)
    else:
        final_logits = final_plate_logits
    
    decoded_preds = decode_batch_logits(final_logits, converter)
    
    if return_scores:
        eos_score = 0.0
        # Give CPPD credit for predicting the EOS token
        if T > 7:
            eos_score = F.log_softmax(batch_logits[:, 7:, :], dim=-1).max(dim=-1)[0].sum(dim=1)
        final_scores = torch.where(is_mercosur.squeeze(-1).squeeze(-1), score_mercosur, score_old) + eos_score
        return decoded_preds, final_scores
        
    return decoded_preds

def get_layout_label(text):
    if len(text) < 5: return 0
    return 1 if text[4].isalpha() else 0

class SVTR_CTCLoss(nn.Module):
    def __init__(self, blank_idx=0):
        super().__init__()
        # zero_infinity=True prevents explosions on weird crops
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, zero_infinity=True)

    def forward(self, logits, targets):
        # logits: (B, T, C) -> permute to (T, B, C) for CTCLoss
        logits_t = logits.permute(1, 0, 2)
        log_probs = F.log_softmax(logits_t, dim=2)
        
        B, T, _ = logits.shape
        
        # Filter out the padding (0s) from the targets
        valid_targets = []
        target_lengths = []
        for i in range(B):
            valid = targets[i][targets[i] != 0]
            valid_targets.append(valid)
            target_lengths.append(len(valid))
            
        flat_targets = torch.cat(valid_targets) if valid_targets else torch.empty(0, dtype=torch.long, device=logits.device)
        input_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.long, device=logits.device)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=logits.device)
        
        return self.ctc_loss(log_probs, flat_targets, input_lengths, target_lengths)

def ctc_greedy_decoder(batch_logits, converter, return_scores=False):
    """Decodes CTC sequence probabilities into text strings."""
    probs = F.softmax(batch_logits, dim=-1)
    scores_max, indices = probs.max(dim=-1)
    
    indices = indices.cpu().numpy()
    scores_max = scores_max.cpu().numpy()
    
    decoded_preds = []
    final_scores = []
    
    for b in range(len(indices)):
        chars = []
        conf_sum = 0.0
        char_count = 0
        prev_idx = -1
        
        for t in range(indices.shape[1]):
            idx = indices[b, t]
            # Standard CTC rule: ignore blanks (0) and consecutive duplicates
            if idx != 0 and idx != prev_idx:
                chars.append(converter.alphabet[idx])
                conf_sum += scores_max[b, t]
                char_count += 1
            prev_idx = idx
            
        decoded_preds.append("".join(chars))
        final_scores.append(conf_sum / max(char_count, 1))
        
    if return_scores:
        return decoded_preds, torch.tensor(final_scores, device=batch_logits.device)
    return decoded_preds

class SmoothPoly1Loss(nn.Module):
    def __init__(self, epsilon=2.0, smoothing=0.1, ignore_index=38):
        super().__init__()
        self.epsilon = epsilon
        self.smoothing = smoothing
        self.ignore_index = ignore_index # ADD THIS

    def forward(self, logits, targets):
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)

        # ADD ignore_index here!
        ce_loss = F.cross_entropy(
            logits_flat, targets_flat, 
            label_smoothing=self.smoothing, reduction='none', ignore_index=self.ignore_index
        )

        with torch.no_grad():
            # AND ADD ignore_index here!
            clean_ce = F.cross_entropy(
                logits_flat, targets_flat, 
                reduction='none', ignore_index=self.ignore_index
            )
            pt = torch.exp(-clean_ce)

        poly1_loss = ce_loss + self.epsilon * (1.0 - pt)
        
        # Mask out the padding tokens before taking the mean!
        valid_mask = (targets_flat != self.ignore_index).float()
        return (poly1_loss * valid_mask).sum() / valid_mask.sum()

class HyperAttentionLoss(nn.Module):
    def __init__(self, grid_h=16, grid_w=48):
        super().__init__()
        x_coords = torch.linspace(0, 1, grid_w)
        y_coords = torch.linspace(0, 1, grid_h)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        self.register_buffer('x_flat', x_grid.flatten())
        self.register_buffer('y_flat', y_grid.flatten())
        
        self.aspect_ratio = grid_w / grid_h 
        
        # --- NEW: Fully Learnable Iris Scale ---
        # Starts at 0.5 (medium pressure). The Meta-Optimizer will pull 
        # this down as it mathematically proves that a tighter focus helps validation.
        self.iris_scale = nn.Parameter(torch.tensor(2.0))
        
        # Hypergradient Parameters
        self.v_stretch = nn.Parameter(torch.tensor(1.2)) 
        self.monotonic_scale = nn.Parameter(torch.tensor(15.0))
        self.boundary_scale = nn.Parameter(torch.tensor(20.0))
        self.ortho_scale = nn.Parameter(torch.tensor(5.0))

    def forward(self, multi_head_attn, current_epoch=0):
        # Note: current_epoch is kept in the signature so it doesn't break 
        # your train_utils.py calls, but the math is now 100% time-free!
        
        B, H, Q, T = multi_head_attn.shape
        avg_attn = multi_head_attn.mean(dim=1) 
        
        cx = torch.sum(avg_attn * self.x_flat, dim=-1, keepdim=True) 
        cy = torch.sum(avg_attn * self.y_flat, dim=-1, keepdim=True) 
        cx_flat = cx.squeeze(-1) 

        # ==========================================
        # A. Spread (Now Autonomous)
        # ==========================================
        # We add 0.05 so the scale never mathematically hits absolute zero
        base_iris = torch.abs(self.iris_scale) + 0.05
        v_s = torch.abs(self.v_stretch)
        
        dist_sq = ((self.x_flat - cx) * self.aspect_ratio)**2 + ((self.y_flat - cy) * v_s)**2
        
        # Dividing by base_iris allows the Meta-Optimizer to directly control 
        # the strictness of the spatial penalty without relying on a schedule.
        spread_loss = (torch.sum(avg_attn * dist_sq, dim=-1) / base_iris).mean()

        # ==========================================
        # B. Monotonic
        # ==========================================
        dx = cx_flat[:, 1:] - cx_flat[:, :-1]
        pixel_w = 1.0 / 48.0
        min_center_dist = 4.5 * pixel_w 
        max_center_dist = 8.5 * pixel_w 
        overlap_penalty = F.relu(min_center_dist - dx)
        drift_penalty = F.relu(dx - max_center_dist)
        monotonic_loss = ((overlap_penalty ** 2) + (drift_penalty ** 2)).mean() * torch.abs(self.monotonic_scale)

        # ==========================================
        # C. Orthogonality
        # ==========================================
        norm_attn = F.normalize(multi_head_attn, p=2, dim=-1)
        attn_by_query = norm_attn.transpose(1, 2)
        overlap_matrix = torch.matmul(attn_by_query, attn_by_query.transpose(-2, -1))
        device = multi_head_attn.device
        identity_mask = torch.eye(H, device=device).view(1, 1, H, H)
        off_diagonal_overlap = overlap_matrix * (1.0 - identity_mask)
        clumping_penalty = F.relu(off_diagonal_overlap - 0.50) 
        ortho_loss = (clumping_penalty ** 2).mean() * torch.abs(self.ortho_scale)

        # ==========================================
        # D. Boundary
        # ==========================================
        left_bound = F.relu(0.15 - cx_flat[:, 0])
        right_bound = F.relu(cx_flat[:, -1] - 0.88)
        
        # THE FIX: Evaluate every character's Y-coordinate independently!
        cy_flat = cy.squeeze(-1) 
        top_bound = F.relu(0.01 - cy_flat).mean()
        bottom_bound = F.relu(cy_flat - 0.99).mean()
        
        boundary_loss = ((left_bound**2).mean() + (right_bound**2).mean() + top_bound**2 + bottom_bound**2) * torch.abs(self.boundary_scale)

        return spread_loss + ortho_loss + monotonic_loss + boundary_loss

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
def visualize_feature_maps(latent_tensor, original_images, stages, batch_idx, epoch, save_dir='./train_features'):
    os.makedirs(save_dir, exist_ok=True)
    
    # # 1. Process Stages (Coarse to Fine)
    # vis_strip = []
    # for s in stages:
    #     # Take mean of channels and normalize to 0-1 for visualization
    #     s_mean = s[0].mean(dim=0, keepdim=True).detach().cpu()
    #     s_mean = (s_mean - s_mean.min()) / (s_mean.max() - s_mean.min() + 1e-6)
    #     vis_strip.append(s_mean)
        
    # # 2. Add the final refined latent map at the end
    # final_mean = latent_tensor[0].mean(dim=0, keepdim=True).detach().cpu()
    # final_mean = (final_mean - final_mean.min()) / (final_mean.max() - final_mean.min() + 1e-6)
    # vis_strip.append(final_mean)
    
    # # 3. Concatenate horizontally (Dim 2 is Width)
    # hierarchy_strip = torch.cat(vis_strip, dim=2)
    
    # # 4. Save the new hierarchy strip
    # vutils.save_image(hierarchy_strip, os.path.join(save_dir, 'latent_hierarchy.png'))

    if original_images.dim() == 5:
        center_frame_idx = original_images.shape[1] // 2
        orig_img = original_images[0, center_frame_idx].detach().cpu()
    else:
        orig_img = original_images[0].detach().cpu()

    if orig_img.min() < 0:
        orig_img = (orig_img * 0.5) + 0.5
    orig_img = torch.clamp(orig_img, 0, 1)

    feat_map = latent_tensor[0].detach().cpu() 
    mean_activation = feat_map.mean(dim=0, keepdim=True) 
    channels_to_plot = feat_map[:15] 
    
    grid_items = torch.cat([mean_activation.unsqueeze(0), channels_to_plot.unsqueeze(1)], dim=0)

    for i in range(grid_items.shape[0]):
        c_min = grid_items[i].min()
        c_max = grid_items[i].max()
        grid_items[i] = (grid_items[i] - c_min) / (c_max - c_min + 1e-6)

    vutils.save_image(orig_img, os.path.join(save_dir, f'raw_input_ep.png'))
    vutils.save_image(
        grid_items, 
        os.path.join(save_dir, f'latent_ep.png'), 
        nrow=4, 
        padding=1, 
        normalize=False 
    )

def visualize_vit_attention(image_tensor, latent_tensor, attn_weights, query_texts, epoch, batch_idx, save_dir="attn_maps"):
    import cv2
    import numpy as np
    os.makedirs(save_dir, exist_ok=True)

    if image_tensor.dim() == 5:
        img = image_tensor[0, image_tensor.shape[1] // 2].detach().cpu()
    else:
        img = image_tensor[0].detach().cpu()

    if img.min() < 0:
        img = (img * 0.5) + 0.5
    img_lr_tensor = torch.clamp(img, 0, 1)
    img_lr_np = (img_lr_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    lr_h, lr_w = img_lr_np.shape[:2]

    # --- NEW: Unpack the Heatmap Tuple ---
    # sample_locs: (B, 7, 9, 2)
    # weights: (B, 7, Heads, 9)
    sample_locs, weights = attn_weights 
    
    # Take Batch 0
    locs_b0 = sample_locs[0].detach().cpu().numpy() # (7, 9, 2)
    
    # Average the heat across all Attention Heads
    weights_b0 = weights[0].mean(dim=1).detach().cpu().numpy() # (7, 9)

    num_queries = locs_b0.shape[0]
    grid_items = [img_lr_tensor]

    colors = [
        (255, 50, 50),   (50, 255, 50),   (50, 50, 255),   (255, 255, 50),  
        (255, 50, 255),  (50, 255, 255),  (255, 128, 0),   (128, 128, 128) 
    ]

    for i in range(num_queries):
        vis_img = img_lr_np.copy()
        color = colors[i % len(colors)]
        
        # Get the 9 points and 9 weights for this specific character
        char_pts = locs_b0[i] # (9, 2)
        char_weights = weights_b0[i] # (9,)
        
        # Normalize weights so the hottest point is 1.0
        if char_weights.max() > 0:
            char_weights = char_weights / char_weights.max()

        for pt_idx in range(len(char_pts)):
            x = int(char_pts[pt_idx, 0] * lr_w)
            y = int(char_pts[pt_idx, 1] * lr_h)
            heat = char_weights[pt_idx]
            
            # The radius and brightness scale with the "heat"
            radius = max(1, int(4 * heat))
            pt_color = (
                int(color[0] * heat),
                int(color[1] * heat),
                int(color[2] * heat)
            )
            cv2.circle(vis_img, (x, y), radius=radius, color=pt_color, thickness=-1)

        if i < len(query_texts):
            char = query_texts[i]
            # Put text near the very first sampled point (usually the center)
            base_x = int(char_pts[0, 0] * lr_w)
            base_y = int(char_pts[0, 1] * lr_h)
            cv2.putText(vis_img, char, (base_x, base_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        blended_tensor = torch.from_numpy(vis_img).permute(2, 0, 1).float() / 255.0
        grid_items.append(blended_tensor)

    grid_tensor = torch.stack(grid_items)
    vutils.save_image(
        grid_tensor, os.path.join(save_dir, f'attn_ep.png'),
        nrow=4, padding=1, normalize=False
    )

def point_spread_loss(heatmap_data, max_variance=0.01):
    """
    Penalizes the 9 free-floating points if they scatter too far from their center.
    """
    sample_locs, _ = heatmap_data # sample_locs is (B, 7, 9, 2)
    
    # 1. Find the center of mass for each character's 9 points
    centers = sample_locs.mean(dim=2, keepdim=True) 
    
    # 2. Calculate how far the points are straying from their center (Variance)
    variance = torch.mean((sample_locs - centers) ** 2, dim=-1) # (B, 7, 9)
    spread = variance.mean(dim=-1) # Average spread per character: (B, 7)
    
    # 3. Penalize only if they blast wider than the allowed variance
    violation = F.relu(spread - max_variance)
    return (violation ** 2).mean()
# ==============================================================================
# 3. TRAINING LOOP
# ==============================================================================
import higher  # MUST BE AT THE TOP OF train_utils.py

def box_size_penalties(corners, max_area=0.20, std_tolerance=2.0):
    """
    1. Flexible Consensus: Allows boxes to vary within X standard deviations of the mean.
    2. Hard Cap: No box can exceed max_area of the total image.
    corners: (B, 7, 4, 2)
    """
    # 1. Get X and Y coordinates: (B, 7, 4)
    x_coords = corners[..., 0]
    y_coords = corners[..., 1]
    
    # 2. Find the bounding box limits
    xmin = x_coords.min(dim=-1)[0]
    xmax = x_coords.max(dim=-1)[0]
    ymin = y_coords.min(dim=-1)[0]
    ymax = y_coords.max(dim=-1)[0]
    
    # 3. Calculate actual Widths, Heights, and Areas: (B, 7)
    widths = torch.clamp(xmax - xmin, min=0)
    heights = torch.clamp(ymax - ymin, min=0)
    areas = widths * heights
    
    # ==========================================
    # PENALTY A: The Flexible Consensus (Std Dev)
    # ==========================================
    # Calculate Mean and Standard Deviation (detached so they act as fixed targets)
    mean_w = widths.mean(dim=-1, keepdim=True).detach()
    std_w = widths.std(dim=-1, keepdim=True).detach() + 1e-4  # Add epsilon to prevent 0 std
    
    mean_h = heights.mean(dim=-1, keepdim=True).detach()
    std_h = heights.std(dim=-1, keepdim=True).detach() + 1e-4
    
    # Calculate absolute deviation from the mean
    dev_w = torch.abs(widths - mean_w)
    dev_h = torch.abs(heights - mean_h)
    
    # The ReLU creates the "Safe Zone". 
    # If deviation is LESS than (std * tolerance), it becomes 0 (no penalty).
    # If deviation is GREATER, only the excess is penalized.
    penalty_w = F.relu(dev_w - (std_tolerance * std_w))
    penalty_h = F.relu(dev_h - (std_tolerance * std_h))
    
    consensus_loss = (penalty_w ** 2).mean() + (penalty_h ** 2).mean()
    
    # ==========================================
    # PENALTY B: The 20% Hard Cap
    # ==========================================
    area_violation = F.relu(areas - max_area)
    cap_penalty = (area_violation ** 2).mean()
    
    return consensus_loss + cap_penalty

@register('SROCR_TRAIN')
def SROCR_TRAIN(train_loader, val_loader, model_g, model_d, optimizer_g, optimizer_d, optimizer_hyper, loss_fn_spread, config, **kwargs):
    device = next(model_g.parameters()).device
    true_converter = strLabelConverter(config.get('alphabet', "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    use_fp16 = config.get('use_fp16', False)
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16) 
    pbar = tqdm(train_loader, leave=False)
    
    save_root = kwargs.get('save_path', Path('.'))
    loss_stats = {'total': [], 'cls': [], 'spread': [], 'loss_size': []}
    acc_seq_accum = []
    acc_char_accum = [] 
    current_epoch = kwargs.get('epoch', 0)
    
    cls_loss_type = config.get('cls_loss', 'SmoothPoly1')
    if cls_loss_type == 'CTC':
        loss_fn_spatial = SVTR_CTCLoss().to(device)
    elif cls_loss_type == 'CPPD':
        from models.cppd.cppd_bridge import CPPDLossWrapper
        loss_fn_spatial = CPPDLossWrapper(max_len=7).to(device)
    elif cls_loss_type == 'OTE':
        from models.ote.ote_bridge import OTELossWrapper
        loss_fn_spatial = OTELossWrapper(ignore_index=38).to(device)
    # --- NEW: Route POLY to the standard Smooth Poly 1 Loss ---
    elif cls_loss_type == 'POLY':
        loss_fn_spatial = SmoothPoly1Loss(epsilon=1.5, smoothing=0.1).to(device)
    else:
        loss_fn_spatial = SmoothPoly1Loss(epsilon=1.5, smoothing=0.1).to(device)

    epoch_tracker = ConfusionTracker()

    # --- NEW: Validation Iterator for Meta-Learning ---
    val_iter = iter(val_loader)

    for batch_idx, batch in enumerate(pbar):
        if batch is None: continue

        # lr_batch = batch['lr'].to(device, non_blocking=True)
        
        lr_batch = batch['lr'].to(device, non_blocking=True, memory_format=torch.channels_last)
        
        is_hr_mask = batch['is_hr'].to(device, non_blocking=True)
        text_label = batch['gt'] 
        if cls_loss_type == 'CPPD':
            true_targets = true_converter.encode_cppd(text_label, max_len=7)
            true_targets = (true_targets[0].to(device), true_targets[1].to(device))
        elif cls_loss_type == 'OTE':
            true_targets = true_converter.encode_ote(text_label, max_len=7).to(device)
        elif cls_loss_type == 'POLY': 
            # Use the variable encoder for 12 slots!
            true_targets = true_converter.encode_variable(text_label, max_len=12).to(device)
        else:
            true_targets = true_converter.encode_list(text_label).to(device)

        # ====================================================================
        # THE "TRUE" HYPERGRADIENT META-STEP (Every 10 Batches)
        # ====================================================================
        if batch_idx % 10 == 0 and batch_idx > 0 and config.get('cls_loss', 'SmoothPoly1') not in ['CTC', 'CPPD', 'LISTER_INTERNAL', 'OTE', 'POLY']:
            # 1. Grab a fresh validation batch
            try:
                val_batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                val_batch = next(val_iter)
            
            if 'lr' in val_batch:
                # Add it directly here if it's already 4D
                val_lr = val_batch['lr'].to(device, non_blocking=True, memory_format=torch.channels_last)
            elif 'lr_seq' in val_batch:
                seq = val_batch['lr_seq'].to(device, non_blocking=True)
                val_lr = seq[:, seq.shape[1] // 2, :, :, :] if seq.dim() == 5 else seq
                # Add it here after slicing the 5D down to 4D
                val_lr = val_lr.contiguous().to(memory_format=torch.channels_last)
            else:
                raise KeyError("Validation batch contains neither 'lr' nor 'lr_seq'.")

            val_targets = true_converter.encode_list(val_batch['gt']).to(device)

            # Micro-Batch Slicing (Memory Protection)
            meta_bs = 4 
            m_lr = lr_batch[:meta_bs]
            m_tgt = true_targets[:meta_bs]
            m_val_lr = val_lr[:meta_bs]
            m_val_tgt = val_targets[:meta_bs]

            # --- THE FIREWALL FIX ---
            # Save the original gradient states and freeze the entire CNN backbone.
            # By only leaving the ViT/Transformer unfrozen, the double-backward 
            # will never reach the Deformable Convolutions.
            # --- THE FIREWALL FIX ---
            orig_grad_states = {}
            active_vit_params = [] # NEW: We will collect exactly what is left active
            
            for name, param in model_g.named_parameters():
                orig_grad_states[name] = param.requires_grad
                if 'vit' not in name.lower() and 'transformer' not in name.lower():
                    param.requires_grad = False
                else:
                    active_vit_params.append(param) # Collect active ViT params

            optimizer_hyper.zero_grad()

            # --- THE VRAM EXPLOSION FIX ---
            # Create a lightweight SGD optimizer just for the lookahead step!
            # This completely stops 'higher' from building a 35GB Adam momentum graph.
            current_lr = optimizer_g.param_groups[0]['lr']
            meta_opt = torch.optim.SGD(active_vit_params, lr=current_lr)

            # --- THE ATTENTION FIX ---
            with torch.nn.attention.sdpa_kernel(SDPBackend.MATH):
                
                # Use our new 'meta_opt' instead of 'optimizer_g'
                with higher.innerloop_ctx(model_g, meta_opt, copy_initial_weights=False) as (fmodel, diffopt):
                    
                    # --- INNER LOOP (Using Micro-Batch) ---
                    # with torch.amp.autocast('cuda', enabled=use_fp16):
                    train_preds = fmodel(
                        m_lr, temporal_pool=True, tgt=m_tgt, 
                        epoch=current_epoch, return_attn=True
                    )
                    if isinstance(train_preds, (tuple, list)): train_preds = train_preds[0]
                    
                    loss_cls_train = loss_fn_spatial(train_preds['logits'], m_tgt)
                    if loss_cls_train.dim() > 0: loss_cls_train = loss_cls_train.mean()
                    
                    loss_spread_train = 0.0
                    if train_preds['attn_maps'] is not None:
                        loss_spread_train = loss_fn_spread(train_preds['attn_maps'], current_epoch=current_epoch)
                        
                    simulated_train_loss = loss_cls_train + loss_spread_train
                    
                    diffopt.step(simulated_train_loss)

                    # --- OUTER LOOP (Using Validation Micro-Batch) ---
                    # with torch.amp.autocast('cuda', enabled=use_fp16):
                    val_preds = fmodel(
                        m_val_lr, temporal_pool=True, tgt=m_val_tgt, 
                        epoch=current_epoch, return_attn=False
                    )
                    if isinstance(val_preds, (tuple, list)): val_preds = val_preds[0]
                    
                    loss_cls_val = loss_fn_spatial(val_preds['logits'], m_val_tgt)
                    if loss_cls_val.dim() > 0: loss_cls_val = loss_cls_val.mean()

                    # 3. BACKPROP THROUGH TIME
                    scaler.scale(loss_cls_val).backward()

            # 4. Update the Spatial Penalties
            scaler.unscale_(optimizer_hyper)
            scaler.step(optimizer_hyper)
            scaler.update() 
            
            # --- THE RESTORE ---
            # Flawlessly restore the exact original states for the main training step
            for name, param in model_g.named_parameters():
                param.requires_grad = orig_grad_states[name]
                    
            if is_main_process():
                # Extract all 5 values
                i_val = loss_fn_spread.iris_scale.item()
                v_val = loss_fn_spread.v_stretch.item()
                m_val = loss_fn_spread.monotonic_scale.item()
                b_val = loss_fn_spread.boundary_scale.item()
                o_val = loss_fn_spread.ortho_scale.item()
                
                # Add "I: {i_val:.2f} |" to the front of the string!
                hyper_str = f"I: {i_val:.2f} | V: {v_val:.2f} | M: {m_val:.1f} | B: {b_val:.1f} | O: {o_val:.1f}"
                
                # The JS dashboard reads exactly what is written here
                hyper_path = save_root / 'train_features' / 'hyper.txt'
                with open(hyper_path, 'w') as f:
                    f.write(hyper_str)

        # ====================================================================
        # STANDARD TRAINING STEP (Using the real model)
        # ====================================================================
        optimizer_g.zero_grad()
            
        with torch.amp.autocast('cuda', enabled=use_fp16):
            preds_lr = model_g(
                lr_batch, temporal_pool=True, tgt=true_targets, 
                epoch=current_epoch, return_attn=True 
            )

            if isinstance(preds_lr, (tuple, list)): preds_lr = preds_lr[0]

            # If the model provides its own internal loss (like LISTER)
            if 'loss_internal' in preds_lr and preds_lr['loss_internal'] is not None:
                loss_cls_lr = preds_lr['loss_internal']
            elif cls_loss_type in ['CPPD', 'OTE']:
                # Both CPPD and OTE wrappers expect the FULL dictionary to calculate loss!
                loss_cls_lr = loss_fn_spatial(preds_lr, true_targets)
            else:
                loss_cls_lr = loss_fn_spatial(preds_lr['logits'], true_targets)
                    
            loss_spread = 0.0
            
            # preds_lr['attn_maps'] now contains the tuple (locations, weights)
            if preds_lr['attn_maps'] is not None and cls_loss_type == 'POLY':
                # Apply our new minimalist point spread penalty!
                loss_spread = point_spread_loss(preds_lr['attn_maps'], max_variance=0.01) * 2.0

            total_loss = loss_cls_lr + loss_spread

        if not torch.isfinite(total_loss):
            print(f"⚠️ Warning: Non-finite loss at batch {batch_idx}. Skipping.")
            continue

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer_g)
        scaler.step(optimizer_g)
        scaler.update()

        # ====================================================================
        # METRICS & VISUALIZATION
        # ====================================================================
        with torch.no_grad():
            loss_stats['total'].append(total_loss.item())
            loss_stats['cls'].append(loss_cls_lr.item()) 
            # loss_stats['loss_size'].append(loss_size.item() if isinstance(loss_size, torch.Tensor) else loss_size)
            
            if cls_loss_type == 'CTC':
                decoded_s = ctc_greedy_decoder(preds_lr['logits'], true_converter)
            else:
                decoded_s = decode_batch_logits(preds_lr['logits'], true_converter)
            
            if batch_idx % 10 == 0:
                if 'latent_lr' in preds_lr and preds_lr['latent_lr'] is not None:
                    visualize_feature_maps(
                        latent_tensor=preds_lr['latent_lr'], 
                        original_images=lr_batch, 
                        stages=preds_lr.get('latent_stages'), # <--- PASS STAGES
                        batch_idx=batch_idx, 
                        epoch=current_epoch, 
                        save_dir=save_root / 'train_features'
                    )
                
                if 'attn_maps' in preds_lr and preds_lr['attn_maps'] is not None:
                    sample_pr = decoded_s[0]
                    visualize_vit_attention(
                        image_tensor=lr_batch, latent_tensor=preds_lr['latent_lr'], 
                        attn_weights=preds_lr['attn_maps'], query_texts=sample_pr,             
                        epoch=current_epoch, batch_idx=batch_idx, save_dir=save_root / 'train_features'
                    )

            acc_s = sum([1 for p, t in zip(decoded_s, text_label) if p == t]) / len(text_label)
            acc_seq_accum.append(acc_s)
            
            total_chars = sum(len(t) for t in text_label)
            correct_chars = sum(sum(1 for pc, tc in zip(p, t) if pc == tc) for p, t in zip(decoded_s, text_label))
            acc_c = correct_chars / max(total_chars, 1)
            acc_char_accum.append(acc_c)
            
            epoch_tracker.update(decoded_s, text_label) 

            if batch_idx % 5 == 0:
                # --- NEW: Safe extraction of the Grid Gain (Proof of Life) ---
                try:
                    # Safely bypass DDP wrapper if it exists
                    base_model = model_g.module if hasattr(model_g, 'module') else model_g
                    g_gain = base_model.cgnet.student_extractor.latent_sr.grid_gain.item()
                except Exception:
                    g_gain = 0.0
                
                pbar.set_postfix({
                    'Loss': f"{np.mean(loss_stats['total']):.4f}",
                    # 'Olp': f"{np.mean(loss_stats['loss_size']):.4f}",
                    'Cls': f"{np.mean(loss_stats['cls']):.4f}",
                    # 'Gain': f"{g_gain:.4f}",  # <--- YOUR PEACE OF MIND
                    'Seq%': f"{np.mean(acc_seq_accum):.1%}", # The weighted history
                    'B_Seq': f"{acc_s:.1%}",                 
                    'Chr%': f"{np.mean(acc_char_accum):.1%}",
                })

                # if batch_idx % 5 == 0:
                # pbar.set_postfix({
                #     'Loss': f"{np.mean(loss_stats['total']):.4f}",
                #     'Spd': f"{np.mean(loss_stats['spread']):.4f}",
                #     'Cls': f"{np.mean(loss_stats['cls']):.4f}",
                #     'Seq%': f"{np.mean(acc_seq_accum):.1%}", # The weighted history
                #     'B_Seq': f"{acc_s:.1%}",                 # <--- NEW: Instant Batch Accuracy
                #     'Chr%': f"{np.mean(acc_char_accum):.1%}",
                # })
            
            if batch_idx % 50 == 0:
                sample_gt = text_label[0]
                sample_pr = decoded_s[0]
                # tqdm.write(f"🔎 [B{batch_idx}] GT: {sample_gt} | Pr: {sample_pr}")

            if batch_idx % 10 == 0:
                write_live_monitor(save_root / 'live_monitor.txt', text_label, decoded_s, decoded_s, current_epoch, batch_idx)
            
    return np.mean(loss_stats['total']) if loss_stats['total'] else 0.0

@register('SROCR_VAL')
def SROCR_VAL(val_loader, model_g, model_d, loss_fn, config, **kwargs):
    model_g.eval() 
    device = next(model_g.parameters()).device
    true_converter = strLabelConverter(config.get('alphabet', "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    
    # --- NEW: Validation Error Tracker ---
    val_tracker = ConfusionTracker()
    save_root = kwargs.get('save_path', Path('.'))
    
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
            flat_imgs = flat_imgs.contiguous().to(memory_format=torch.channels_last)
            with torch.amp.autocast('cuda', enabled=config.get('use_fp16', False)):
                output = model_g(flat_imgs, temporal_pool=True) 
                if isinstance(output, (tuple, list)): output = output[0]
                logits = output['logits']

            cls_loss_type = config.get('cls_loss', 'SmoothPoly1')
            # --- THE NEW FIX: Trust the Transformer ---
            if cls_loss_type == 'CTC':
                all_decoded_preds, all_scores = ctc_greedy_decoder(logits, true_converter, return_scores=True)
            else:
                # 1. Decode the raw strings without rigid layout masks
                all_decoded_preds = decode_batch_logits(logits, true_converter)
                # 2. Calculate the raw confidence score for the ensembling logic
                all_scores = F.log_softmax(logits, dim=-1).max(dim=-1)[0].sum(dim=1)

            for b in range(B):
                seq_preds = all_decoded_preds[b * Seq_Len : (b + 1) * Seq_Len]
                seq_scores = all_scores[b * Seq_Len : (b + 1) * Seq_Len]
                
                pred_tracker = {}
                for pred_str, conf_score in zip(seq_preds, seq_scores):
                    if pred_str not in pred_tracker:
                        pred_tracker[pred_str] = {'votes': 0, 'confidence': 0.0}
                    
                    pred_tracker[pred_str]['votes'] += 1
                    pred_tracker[pred_str]['confidence'] += conf_score.item()
                
                sorted_preds = sorted(
                    pred_tracker.items(), 
                    key=lambda item: (item[1]['votes'], item[1]['confidence']), 
                    reverse=True
                )
                
                winning_prediction = sorted_preds[0][0]
                gt = text_labels[b]
                
                # --- NEW: Track the post-ensembled errors! ---
                val_tracker.update([winning_prediction], [gt])
                
                total_sequences += 1
                if winning_prediction == gt: correct_sequences += 1
                
                total_chars += len(gt)
                correct_chars += sum(1 for pc, gc in zip(winning_prediction, gt) if pc == gc)
            
    acc_seq = correct_sequences / total_sequences if total_sequences else 0.0
    acc_char = correct_chars / total_chars if total_chars else 0.0
    
    print(f"\n{'='*30} SEQUENCE EVALUATION {'='*30}")
    print(f"Total Sequences: {total_sequences}")
    print(f"Sequence Accuracy:  {acc_seq:.2%}")
    print(f"Character Accuracy: {acc_char:.2%}")
    print(f"{'='*81}\n")
    
    # --- NEW: Save the true validation blind spots for the next epoch ---
    total_failures = val_tracker.get_worst_pairs_dict(top_k=50) 
    with open(save_root / 'confusion_stats.json', 'w') as f: 
        json.dump(total_failures, f, indent=4)
    
    return 0.0, acc_seq, 0.0