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
    # REMOVED THE HYPHEN:
    def __init__(self, alphabet="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        # Index 0 is EOS ($). The last index is PAD (#).
        self.alphabet = ['$'] + list(alphabet) + ['#'] 
        self.dict = {char: i for i, char in enumerate(self.alphabet)}
        self.pad_idx = len(self.alphabet) - 1 # Dynamically handles the padding index!
    
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
    
    def encode_variable(self, text_list, max_len=7):
        EOS_INDEX = 0   
        PAD_INDEX = self.pad_idx  # <--- THE FIX: Use the dynamic property!

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
    def __init__(self, blank_idx=0, pad_idx=37): # <--- THE FIX: Accept pad_idx
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
        self.pad_idx = pad_idx

    def forward(self, logits, targets):
        logits_t = logits.permute(1, 0, 2)
        log_probs = F.log_softmax(logits_t, dim=2)
        
        B, T, _ = logits.shape
        
        valid_targets = []
        target_lengths = []
        for i in range(B):
            # ---> THE FIX: Safely strip out both Blank/EOS (0) AND Padding (38)
            valid = targets[i][(targets[i] != 0) & (targets[i] != self.pad_idx)]
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
    
    # ---> NEW: Dynamically get the blank index (37)
    blank_idx = getattr(converter, 'pad_idx', 37) 
    
    for b in range(len(indices)):
        chars = []
        conf_sum = 0.0
        char_count = 0
        prev_idx = -1
        
        for t in range(indices.shape[1]):
            idx = indices[b, t]
            # ---> THE FIX: Ignore the official blank_idx (37), EOS (0), and consecutive duplicates
            if idx != blank_idx and idx != 0 and idx != prev_idx:
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
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class ShiftInvariantL1Loss(nn.Module):
    """
    A spatially-relaxed L1 loss that forgives YOLO bounding box misalignment.
    It allows the SR prediction to search a local neighborhood in the GT for the best match.
    """
    def __init__(self, max_shift=3, patch_size=3):
        super().__init__()
        self.max_shift = max_shift
        self.patch_size = patch_size

    def forward(self, sr_img, gt_img):
        B, C, H, W = sr_img.shape
        
        # 1. Pad the Ground Truth so we can shift it without losing edge pixels
        gt_padded = F.pad(gt_img, (self.max_shift, self.max_shift, self.max_shift, self.max_shift), mode='replicate')
        
        shifted_errors = []
        
        # 2. Vectorized Grid Search (No Python 'for' loops across the image!)
        # We shift the GT image in all possible (x,y) directions within the max_shift limit
        for dy in range(2 * self.max_shift + 1):
            for dx in range(2 * self.max_shift + 1):
                # Crop out the shifted version of the GT
                gt_shifted = gt_padded[:, :, dy:dy+H, dx:dx+W]
                
                # Calculate the raw pixel-wise absolute error for this shift
                raw_l1 = torch.abs(sr_img - gt_shifted)
                
                # 3. Patch-wise smoothing (Emulating your "window" idea)
                # We pool the errors locally so an entire stroke has to match, not just stray pixels
                patch_error = F.avg_pool2d(raw_l1, kernel_size=self.patch_size, stride=1, padding=self.patch_size//2)
                
                shifted_errors.append(patch_error)
                
        # Stack all shifted error maps: Shape (B, Number_of_Shifts, C, H, W)
        all_errors = torch.stack(shifted_errors, dim=1)
        
        # 4. The Magic Step: Take the Minimum Error across all spatial shifts!
        # This mathematically forgives spatial jitter. If the "B" is 2 pixels to the left, 
        # one of the shifts will perfectly align it, resulting in near-zero error!
        min_error, _ = torch.min(all_errors, dim=1)
        
        # Return the mean of the lowest possible errors
        return min_error.mean()

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
def visualize_feature_maps(student_sr, teacher_sr, lr_images, hr_images, batch_idx, epoch, save_dir='./train_features'):
    os.makedirs(save_dir, exist_ok=True)

    # 1. Process and Save the Student Input (LR)
    student_img = lr_images[0].detach().cpu()
    if student_img.min() < 0: student_img = (student_img * 0.5) + 0.5
    vutils.save_image(torch.clamp(student_img, 0, 1), os.path.join(save_dir, 'student_in.png'))

    # 2. Process and Save the Teacher Input (HR)
    teacher_img = hr_images[0].detach().cpu() if hr_images is not None else student_img
    if teacher_img.min() < 0: teacher_img = (teacher_img * 0.5) + 0.5
    vutils.save_image(torch.clamp(teacher_img, 0, 1), os.path.join(save_dir, 'teacher_in.png'))

    # 3. Process and Save the Super-Resolved Images INDIVIDUALLY
    if student_sr is not None:
        s_sr_img = student_sr[0].detach().cpu()
        s_sr_img = (s_sr_img * 0.5) + 0.5 
        vutils.save_image(torch.clamp(s_sr_img, 0, 1), os.path.join(save_dir, 'student_sr.png'))
        
    if teacher_sr is not None:
        t_sr_img = teacher_sr[0].detach().cpu()
        t_sr_img = (t_sr_img * 0.5) + 0.5
        vutils.save_image(torch.clamp(t_sr_img, 0, 1), os.path.join(save_dir, 'teacher_sr.png'))

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

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, ignore_index=38):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # Flatten the sequences so we evaluate character-by-character
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)

        # 1. Calculate standard Cross Entropy (Unreduced)
        ce_loss = F.cross_entropy(
            logits_flat, targets_flat, 
            reduction='none', ignore_index=self.ignore_index
        )

        with torch.no_grad():
            # 2. Get the probability of the true class (p_t)
            pt = torch.exp(-ce_loss)
            
            # ---> THE FP16 FIX: Clamp to prevent exact 0.0 or 1.0 (underflow)
            # 1e-4 is extremely safe for fp16 precision
            pt = torch.clamp(pt, min=1e-4, max=1.0 - 1e-4)

        # 3. Apply the Focal Loss formula: alpha * (1 - p_t)^gamma * CE
        focal_loss = self.alpha * ((1.0 - pt) ** self.gamma) * ce_loss
        
        # 4. Safely mask out the padding tokens before averaging
        valid_mask = (targets_flat != self.ignore_index).float()
        
        # ---> ANOTHER FP16 FIX: Clamp the denominator to prevent division by absolute zero
        return (focal_loss * valid_mask).sum() / torch.clamp(valid_mask.sum(), min=1e-4)

@register('SROCR_TRAIN')
def SROCR_TRAIN(train_loader, val_loader, model_g, model_d, optimizer_g, optimizer_d, optimizer_hyper, loss_fn_spread, config, **kwargs):
    device = next(model_g.parameters()).device
    true_converter = strLabelConverter(config.get('alphabet', "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    use_fp16 = config.get('use_fp16', False)
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16) 
    pbar = tqdm(train_loader, leave=False)
    
    save_root = kwargs.get('save_path', Path('.'))
    loss_stats = {'total': [], 'cls_s': [], 'cls_t': [], 'distill': []}
    
    running_acc_seq_s, running_acc_char_s = 0.0, 0.0
    running_acc_seq_t, running_acc_char_t = 0.0, 0.0
    
    current_epoch = kwargs.get('epoch', 0)
    cls_loss_type = config.get('cls_loss', 'SmoothPoly1')
    
    if cls_loss_type == 'CTC':
        loss_fn_spatial = SVTR_CTCLoss(blank_idx=true_converter.pad_idx, pad_idx=true_converter.pad_idx).to(device)
    elif cls_loss_type == 'CPPD':
        from models.cppd.cppd_bridge import CPPDLossWrapper
        loss_fn_spatial = CPPDLossWrapper(max_len=7).to(device)
    elif cls_loss_type == 'OTE':
        from models.ote.ote_bridge import OTELossWrapper
        loss_fn_spatial = OTELossWrapper(ignore_index=38).to(device)
    elif cls_loss_type == 'POLY':
        loss_fn_spatial = SmoothPoly1Loss(epsilon=1.5, smoothing=0.1).to(device)
    elif cls_loss_type == 'FL':
        loss_fn_spatial = FocalLoss(gamma=2.0, alpha=1.0, ignore_index=38).to(device)

    epoch_tracker = ConfusionTracker()
    shift_loss_fn = ShiftInvariantL1Loss(max_shift=3, patch_size=3).to(device)
    
    for batch_idx, batch in enumerate(pbar):
        if batch is None: continue        
        
        lr_batch = batch['lr'].to(device, non_blocking=True, memory_format=torch.channels_last)
        text_label = batch['gt'] 
        
        # --- Safely detect if Distillation is active for this run ---
        use_distillation = (model_d is not None) and ('hr' in batch)
        if use_distillation:
            hr_batch = batch['hr'].to(device, non_blocking=True, memory_format=torch.channels_last)
            is_hr_mask = batch['is_hr'].to(device)
            if 'hr_gt' in batch:
                hr_gt_batch = batch['hr_gt'].to(device, non_blocking=True, memory_format=torch.channels_last)
        
        if cls_loss_type == 'CPPD':
            true_targets = true_converter.encode_cppd(text_label, max_len=7)
            true_targets = (true_targets[0].to(device), true_targets[1].to(device))
        elif cls_loss_type in ['OTE', 'POLY', 'FL', 'CTC']:
            true_targets = getattr(true_converter, f'encode_{"variable" if cls_loss_type in ["POLY", "CTC", "FL"] else "ote"}')(text_label, max_len=7).to(device)
        else:
            true_targets = true_converter.encode_list(text_label).to(device)

        optimizer_g.zero_grad()
        if use_distillation and optimizer_d is not None:  
            optimizer_d.zero_grad()
            
        with torch.amp.autocast('cuda', enabled=use_fp16):
            # ==========================================
            # 1. STUDENT FORWARD PASS
            # ==========================================
            preds_lr = model_g(
                lr_batch, temporal_pool=True, tgt=true_targets, 
                epoch=current_epoch, return_attn=True 
            )
            if isinstance(preds_lr, (tuple, list)): preds_lr = preds_lr[0]

            # Primary Text Loss
            if 'loss_internal' in preds_lr and preds_lr['loss_internal'] is not None:
                loss_cls_lr = preds_lr['loss_internal']
            elif cls_loss_type in ['CPPD', 'OTE']:
                loss_cls_lr = loss_fn_spatial(preds_lr, true_targets)
            else:
                loss_cls_lr = loss_fn_spatial(preds_lr['logits'], true_targets)
            
            total_loss_g = loss_cls_lr

            # ---> NEW: FORMAT ROUTER SUPERVISION
            # Trains the Dynamic Geometry Router to guess Mercosur vs Old Brazilian
            loss_router = torch.tensor(0.0, device=device)
            if 'format_logits' in preds_lr:
                # 0 = Old Brazilian, 1 = Mercosur
                true_format_labels = torch.tensor([get_layout_label(txt) for txt in text_label], device=device)
                loss_router = F.cross_entropy(preds_lr['format_logits'], true_format_labels)
                total_loss_g = total_loss_g + (2.0 * loss_router)

            total_loss = total_loss_g

            # ==========================================
            # 2. TEACHER FORWARD PASS & DISTILLATION
            # ==========================================
            if use_distillation:
                context_manager = torch.no_grad() if optimizer_d is None else torch.enable_grad()
                with context_manager:
                    preds_hr = model_d(
                        hr_batch, temporal_pool=True, tgt=true_targets, 
                        epoch=current_epoch, return_attn=False 
                    )
                    if isinstance(preds_hr, (tuple, list)): preds_hr = preds_hr[0]
                    
                    if 'loss_internal' in preds_hr and preds_hr['loss_internal'] is not None:
                        loss_cls_hr = preds_hr['loss_internal']
                    elif cls_loss_type in ['CPPD', 'OTE']:
                        loss_cls_hr = loss_fn_spatial(preds_hr, true_targets)
                    else:
                        loss_cls_hr = loss_fn_spatial(preds_hr['logits'], true_targets)

                    # Teacher SR Supervision (Teacher strictly learns perfect HR -> HR mapping)
                    if optimizer_d is not None and 'sr_image' in preds_hr and 'hr_gt_batch' in locals():
                        teacher_pixel_loss = F.l1_loss(preds_hr['sr_image'], hr_gt_batch)
                        loss_cls_hr = loss_cls_hr + (10.0 * teacher_pixel_loss)

                # ==========================================
                # DISTILLATION LOSSES (GRADIENT ISOLATION)
                # ==========================================
                teacher_logits = preds_hr['logits'][is_hr_mask].detach()
                
                # 1. PIXEL: Dual-Routed Spatial Supervision against GROUND TRUTH
                loss_distill_pixel = torch.tensor(0.0, device=device)
                if 'sr_image' in preds_lr and 'hr_gt_batch' in locals():
                    # Route A: Strict L1 for Perfectly Aligned Images (RODOSOL/Syn)
                    if is_hr_mask.any():
                        loss_aligned = F.l1_loss(preds_lr['sr_image'][is_hr_mask], hr_gt_batch[is_hr_mask])
                        loss_distill_pixel = loss_distill_pixel + loss_aligned
                        
                    # Route B: Shift-Invariant L1 for Misaligned CCTV
                    is_lr_mask = ~is_hr_mask
                    if is_lr_mask.any():
                        loss_jittered = shift_loss_fn(preds_lr['sr_image'][is_lr_mask], hr_gt_batch[is_lr_mask])
                        loss_distill_pixel = loss_distill_pixel + (0.5 * loss_jittered)

                # 2. LOGITS: KL Divergence (Student mimics Teacher's text probabilities)
                temperature = config.get('distill_temp', 2.0)
                student_log_probs = F.log_softmax(preds_lr['logits'][is_hr_mask] / temperature, dim=-1)
                teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
                loss_distill_kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
                
                # Apply Final Distillation Weights
                kl_weight = config.get('distill_weight_kl', 2.0)
                gt_pixel_weight = 10.0  
                
                total_loss_g = total_loss_g + (kl_weight * loss_distill_kl) + (gt_pixel_weight * loss_distill_pixel)
                                
                if optimizer_d is not None:
                    total_loss = total_loss_g + loss_cls_hr
                else:
                    total_loss = total_loss_g

        if not torch.isfinite(total_loss):
            print(f"⚠️ Warning: Non-finite loss at batch {batch_idx}. Skipping.")
            continue

        scaler.scale(total_loss).backward()
        
        scaler.unscale_(optimizer_g)
        torch.nn.utils.clip_grad_norm_(model_g.parameters(), max_norm=5.0)
        
        if use_distillation and optimizer_d is not None:  
            scaler.unscale_(optimizer_d)
            torch.nn.utils.clip_grad_norm_(model_d.parameters(), max_norm=5.0)

        scaler.step(optimizer_g)
        if use_distillation and optimizer_d is not None:  
            scaler.step(optimizer_d)
            
        scaler.update()
        # warmup_epochs = 1000
        # if use_distillation and optimizer_d is None and current_epoch >= warmup_epochs:
            
        #     # ---> THE FIX: The Brain Transplant! 
        #     # On the exact first batch of the warmup epoch, clone the Student!
        #     if current_epoch == warmup_epochs and batch_idx == 0:
        #         print("\n🧠 Executing Brain Transplant: Student -> Teacher...")
        #         model_d.load_state_dict(model_g.state_dict())
            
        #     # Now proceed with the normal 0.9999 EMA drip...
        #     decay = 0.9999 
        #     with torch.no_grad():
        #         for (name_s, param_s), (name_t, param_t) in zip(model_g.named_parameters(), model_d.named_parameters()):
        #             is_eye_or_hand = ('patch_embed' in name_s) or ('latent_sr' in name_s) or ('hr_refine' in name_s)
        #             if not is_eye_or_hand:
        #                 param_t.data.mul_(decay).add_(param_s.data, alpha=1.0 - decay)
                        
        # ====================================================================
        # METRICS & VISUALIZATION
        # ====================================================================
        with torch.no_grad():
            loss_stats['total'].append(total_loss.item())
            loss_stats['cls_s'].append(loss_cls_lr.item()) 
            
            if batch_idx % 10 == 0:
                if cls_loss_type == 'CTC':
                    decoded_s = ctc_greedy_decoder(preds_lr['logits'], true_converter)
                else:
                    decoded_s = decode_batch_logits(preds_lr['logits'], true_converter)
                
                acc_s_seq = sum([1 for p, t in zip(decoded_s, text_label) if p == t]) / len(text_label)
                running_acc_seq_s = (running_acc_seq_s * 0.9) + (acc_s_seq * 0.1) if running_acc_seq_s > 0 else acc_s_seq
                
                total_chars = sum(len(t) for t in text_label)
                correct_chars_s = sum(sum(1 for pc, tc in zip(p, t) if pc == tc) for p, t in zip(decoded_s, text_label))
                acc_s_chr = correct_chars_s / max(total_chars, 1)
                running_acc_char_s = (running_acc_char_s * 0.9) + (acc_s_chr * 0.1) if running_acc_char_s > 0 else acc_s_chr
                
                epoch_tracker.update(decoded_s, text_label) 
                
                decoded_t = decoded_s 
                if use_distillation:
                    if cls_loss_type == 'CTC':
                        decoded_t = ctc_greedy_decoder(preds_hr['logits'], true_converter)
                    else:
                        decoded_t = decode_batch_logits(preds_hr['logits'], true_converter)
                    
                    acc_t_seq = sum([1 for p, t in zip(decoded_t, text_label) if p == t]) / len(text_label)
                    running_acc_seq_t = (running_acc_seq_t * 0.9) + (acc_t_seq * 0.1) if running_acc_seq_t > 0 else acc_t_seq
                    
                    correct_chars_t = sum(sum(1 for pc, tc in zip(p, t) if pc == tc) for p, t in zip(decoded_t, text_label))
                    acc_t_chr = correct_chars_t / max(total_chars, 1)
                    running_acc_char_t = (running_acc_char_t * 0.9) + (acc_t_chr * 0.1) if running_acc_char_t > 0 else acc_t_chr
                
                postfix_dict = {
                    'Loss': f"{np.mean(loss_stats['total'][-50:]):.4f}",
                    'S-Seq': f"{running_acc_seq_s:.1%}", 
                    'S-Chr': f"{running_acc_char_s:.1%}",
                }
                
                if use_distillation:
                    postfix_dict['T-Seq'] = f"{running_acc_seq_t:.1%}"
                    postfix_dict['T-Chr'] = f"{running_acc_char_t:.1%}"
                
                pbar.set_postfix(postfix_dict)
                write_live_monitor(save_root / 'live_monitor.txt', text_label, decoded_s, decoded_t, current_epoch, batch_idx)

            if batch_idx % 20 == 0:
                if 'sr_image' in preds_lr and preds_lr['sr_image'] is not None:
                    visualize_feature_maps(
                        student_sr=preds_lr['sr_image'], 
                        teacher_sr=preds_hr['sr_image'] if use_distillation else None, 
                        lr_images=lr_batch, 
                        hr_images=hr_batch if use_distillation else None, 
                        batch_idx=batch_idx, 
                        epoch=current_epoch, 
                        save_dir=save_root / 'train_features'
                    )
                
                if 'attn_maps' in preds_lr and preds_lr['attn_maps'] is not None:
                    if 'decoded_s' not in locals():
                        decoded_s = ctc_greedy_decoder(preds_lr['logits'], true_converter) if cls_loss_type == 'CTC' else decode_batch_logits(preds_lr['logits'], true_converter)
                        
                    visualize_vit_attention(
                        image_tensor=lr_batch, latent_tensor=preds_lr.get('latent_lr', None), 
                        attn_weights=preds_lr['attn_maps'], query_texts=decoded_s[0],             
                        epoch=current_epoch, batch_idx=batch_idx, save_dir=save_root / 'train_features'
                    )
                
    final_loss = np.mean(loss_stats['total']) if loss_stats['total'] else 0.0
    
    if use_distillation:
        return final_loss, running_acc_seq_t
        
    return final_loss

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
    
    # ==========================================
    # NEW: EVALUATE THE TEACHER ON HR VALIDATION
    # ==========================================
    # ==========================================
    # NEW: EVALUATE THE TEACHER ON HR VALIDATION
    # ==========================================
    acc_seq_t = 0.0
    if model_d is not None and 'hr_seq' in next(iter(val_loader)):
        model_d.eval()
        correct_sequences_t = 0
        total_sequences_t = 0
        
        # ---> THE FIX: Add this line to prevent the VRAM explosion!
        with torch.no_grad(): 
            for batch in val_loader:
                hr_seqs = batch['hr_seq'].to(device, non_blocking=True)
                text_labels = batch['gt']
                B, Seq_Len, C, H, W = hr_seqs.shape
                
                flat_hr = hr_seqs.view(B * Seq_Len, C, H, W).contiguous().to(memory_format=torch.channels_last)
                
                with torch.amp.autocast('cuda', enabled=config.get('use_fp16', False)):
                    out_t = model_d(flat_hr, temporal_pool=True)
                    logits_t = out_t['logits'] if isinstance(out_t, dict) else out_t[0]['logits']
                
                if cls_loss_type == 'CTC':
                    preds_t = ctc_greedy_decoder(logits_t, true_converter)
                else:
                    preds_t = decode_batch_logits(logits_t, true_converter)
                    
                # Since the Teacher sees the exact same HR image 5 times, we just take the first frame's answer
                for b in range(B):
                    pred_str = preds_t[b * Seq_Len]
                    if pred_str == text_labels[b]: 
                        correct_sequences_t += 1
                    total_sequences_t += 1
                
        acc_seq_t = correct_sequences_t / total_sequences_t if total_sequences_t else 0.0

    print(f"\n{'='*30} SEQUENCE EVALUATION {'='*30}")
    print(f"Total Sequences: {total_sequences}")
    print(f"Student Acc: {acc_seq:.2%} | Teacher HR Acc: {acc_seq_t:.2%}") # <--- UPDATED PRINT
    print(f"{'='*81}\n")
    
    # --- NEW: Save the true validation blind spots for the next epoch ---
    total_failures = val_tracker.get_worst_pairs_dict(top_k=50) 
    with open(save_root / 'confusion_stats.json', 'w') as f: 
        json.dump(total_failures, f, indent=4)
    
    return 0.0, acc_seq, acc_seq_t