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
import kornia.augmentation as K
import torch.nn.functional as F
import random
import torch.nn as nn
from models.igtr.igtr_label_encode import IGTRLabelEncode

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
    
    def encode_mdiff(self, text_list, max_len=7):
        """Creates the Masked Language targets (Noisy Batch) for MDiff."""
        B = len(text_list)
        eos_id = 0
        mask_id = 37 # Because out_channels=39, mask_id is 37
        pad_id = 38  # ignore_index is 38
        
        labels, lengths = [], []
        
        # 1. Base Encoding
        for text in text_list:
            chars = [self.dict.get(c, 0) for c in text[:max_len]]
            lengths.append(len(chars))
            chars.append(eos_id)
            chars += [pad_id] * (max_len + 1 - len(chars))
            labels.append(chars)
            
        labels = torch.tensor(labels, dtype=torch.long)
        lengths = torch.tensor(lengths, dtype=torch.long)
        
        # 2. Diffusion Corruption (The Masking)
        noisy_batch = labels.clone()
        masked_indices = torch.zeros_like(labels, dtype=torch.bool)
        p_mask = torch.ones(B, dtype=torch.float32) * 0.5 # 50% mask probability
        
        for i in range(B):
            valid_len = lengths[i] + 1 
            for j in range(valid_len):
                # We randomly mask tokens with a 50% chance
                if random.random() < 0.5: 
                    noisy_batch[i, j] = mask_id
                    masked_indices[i, j] = True
                    
            # Ensure at least ONE token is masked so the network has something to learn!
            if not masked_indices[i].any():
                idx_to_mask = random.randint(0, valid_len - 1)
                noisy_batch[i, idx_to_mask] = mask_id
                masked_indices[i, idx_to_mask] = True
        
        # 3. Reflect Targets (Same as labels)
        reflect_ids = labels.clone()
        
        return (labels, reflect_ids, noisy_batch, masked_indices, p_mask, lengths)

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
    def __init__(self, epsilon=2.0, smoothing=0.1, ignore_index=38, lambda_sep=2.0):
        super().__init__()
        self.epsilon = epsilon
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.lambda_sep = lambda_sep
        
        # Calculate the mathematical ceiling for the gap
        # If smoothing=0.0, target_gap = 1.0. If smoothing=0.1, target_gap = 0.90.
        self.target_gap = 1.0 - smoothing

    def forward(self, logits, targets):
        logits_flat = logits.view(-1, logits.size(-1)) # (B*T, C)
        targets_flat = targets.view(-1)                # (B*T,)

        # ==========================================
        # 1. Base PolyLoss (The Foundation)
        # ==========================================
        ce_loss = F.cross_entropy(
            logits_flat, targets_flat, 
            label_smoothing=self.smoothing, reduction='none', ignore_index=self.ignore_index
        )

        with torch.no_grad():
            clean_ce = F.cross_entropy(
                logits_flat, targets_flat, 
                reduction='none', ignore_index=self.ignore_index
            )
            pt = torch.exp(-clean_ce)

        poly1_seq = ce_loss + self.epsilon * (1.0 - pt)

        # ==========================================
        # 2. Max Separation Penalty (The Aggressor)
        # ==========================================
        probs = F.softmax(logits_flat, dim=-1)
        
        # A. Get probability of the true class
        p_true = probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)

        # B. Find the highest probability among all WRONG classes
        mask = torch.zeros_like(probs).scatter_(1, targets_flat.unsqueeze(1), 1.0)
        probs_wrong = probs.masked_fill(mask.bool(), -1.0) # Hide the true class
        p_max_wrong, _ = probs_wrong.max(dim=1)

        # C. Calculate the actual gap and penalize distance from the perfect gap
        current_gap = p_true - p_max_wrong
        
        # We use ReLU so we don't accidentally reward the network for exceeding the target gap
        # (which shouldn't happen under CE, but acts as a mathematical safety net)
        gap_deficit = F.relu(self.target_gap - current_gap)
        
        # Square it to heavily punish small gaps, but smooth out as it nears perfection
        separation_penalty = gap_deficit ** 2

        # ==========================================
        # 3. Fusion & Masking
        # ==========================================
        total_loss = poly1_seq + (self.lambda_sep * separation_penalty)

        # Spatial weighting (Index 2 and 3 are penalized heavily)
        total_loss = total_loss.view(-1, 7)
        spatial_weights = torch.tensor([1.0, 1.0, 1.5, 1.5, 1.0, 1.0, 1.0], device=logits.device)
        total_weighted = total_loss * spatial_weights
        
        # Apply padding mask and average
        total_flat = total_weighted.view(-1)
        valid_mask = (targets_flat != self.ignore_index).float()
        
        return (total_flat * valid_mask).sum() / torch.clamp(valid_mask.sum(), min=1e-4)

# class SmoothPoly1Loss(nn.Module):
#     def __init__(self, epsilon=2.0, smoothing=0.1, ignore_index=38):
#         super().__init__()
#         self.epsilon = epsilon
#         self.smoothing = smoothing
#         self.ignore_index = ignore_index # ADD THIS

#     def forward(self, logits, targets):
#         logits_flat = logits.view(-1, logits.size(-1))
#         targets_flat = targets.view(-1)

#         # ADD ignore_index here!
#         ce_loss = F.cross_entropy(
#             logits_flat, targets_flat, 
#             label_smoothing=self.smoothing, reduction='none', ignore_index=self.ignore_index
#         )

#         with torch.no_grad():
#             # AND ADD ignore_index here!
#             clean_ce = F.cross_entropy(
#                 logits_flat, targets_flat, 
#                 reduction='none', ignore_index=self.ignore_index
#             )
#             pt = torch.exp(-clean_ce)

#         poly1_loss = ce_loss + self.epsilon * (1.0 - pt)
        
#         # Reshape to apply spatial weights: (B, 7)
#         poly1_seq = poly1_loss.view(-1, 7)
        
#         # Create a weight mask that penalizes Index 2 and Index 3 heavily
#         # Normal weights = 1.0, Gap weights = 1.5
#         spatial_weights = torch.tensor([1.0, 1.0, 1.5, 1.5, 1.0, 1.0, 1.0], device=logits.device)
#         poly1_weighted = poly1_seq * spatial_weights
        
#         # Flatten back and mask out the padding tokens
#         poly1_flat = poly1_weighted.view(-1)
#         valid_mask = (targets_flat != self.ignore_index).float()
        
#         return (poly1_flat * valid_mask).sum() / valid_mask.sum()
    
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

def build_loss_function(cls_loss_type, converter, device, current_epoch=None):
    """Factory to instantiate the correct spatial OCR loss based on the baseline."""
    if cls_loss_type == 'CTC':
        return SVTR_CTCLoss(blank_idx=converter.pad_idx, pad_idx=converter.pad_idx).to(device)
    elif cls_loss_type == 'CPPD':
        from models.cppd.cppd_bridge import CPPDLossWrapper
        return CPPDLossWrapper(max_len=7).to(device)
    elif cls_loss_type == 'OTE':
        from models.ote.ote_bridge import OTELossWrapper
        return OTELossWrapper(ignore_index=38).to(device)
    elif cls_loss_type == 'POLY':
        
        # ==========================================
        # ---> UPGRADE: Dynamic PolyLoss Annealing
        # ==========================================
        max_epsilon = 2.0
        warmup_epochs = 30
        
        if current_epoch is not None: # We are Training
            if current_epoch <= warmup_epochs:
                # Smooth cosine ramp from 0.0 to max_epsilon
                progress = (current_epoch - 1) / max(1, warmup_epochs - 1)
                current_eps = max_epsilon * 0.5 * (1.0 - math.cos(math.pi * progress))
            else:
                current_eps = max_epsilon
                
            if is_main_process():
                print(f"📉 PolyLoss Schedule | Epoch {current_epoch} | Epsilon: {current_eps:.3f}")
        else:
            # We are Validating. Lock epsilon to max_epsilon so the validation 
            # loss metric remains consistent and comparable across all epochs.
            current_eps = max_epsilon 

        return SmoothPoly1Loss(epsilon=current_eps, smoothing=0.1).to(device)
    
    elif cls_loss_type == 'FL':
        return FocalLoss(gamma=2.0, alpha=1.0, ignore_index=38).to(device)
    elif cls_loss_type in ['LISTER_INTERNAL', 'MDIFF_INTERNAL', 'IGTR_INTERNAL']:
        return None
    else:
        raise ValueError(f"Unknown baseline loss type: {cls_loss_type}")

_CACHED_IGTR_ENCODER = None
def prepare_targets(cls_loss_type, text_label, converter, device, is_training=True):
    """Routes the text strings to the correct tensor encoding format."""
    global _CACHED_IGTR_ENCODER
    if cls_loss_type == 'CPPD':
        t1, t2 = converter.encode_cppd(text_label, max_len=7)
        return (t1.to(device), t2.to(device))
    # ---> THE FIX: Route LISTER_INTERNAL to the variable encoder! <---
    elif cls_loss_type in ['OTE', 'POLY', 'FL', 'CTC', 'LISTER_INTERNAL']:
        encode_func = getattr(converter, f'encode_{"variable" if cls_loss_type in ["POLY", "CTC", "FL", "LISTER_INTERNAL"] else "ote"}')
        return encode_func(text_label, max_len=7).to(device)
    elif cls_loss_type == 'MDIFF_INTERNAL':
        targets = converter.encode_mdiff(text_label, max_len=7)
        return tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in targets)
    elif cls_loss_type == 'IGTR_INTERNAL':
        if is_training:
            # 1. Initialize the authentic OpenOCR encoder ONLY ONCE
            if _CACHED_IGTR_ENCODER is None:
                encoder = IGTRLabelEncode(max_text_length=25, k=8)
                
                # Inject your vocabulary
                custom_dict = converter.dict.copy()
                custom_dict['</s>'] = 0  
                custom_dict['<s>'] = 37   
                custom_dict['<pad>'] = 38 
                
                encoder.dict = custom_dict
                encoder.ignore_index = 38
                encoder.lower = False
                
                _CACHED_IGTR_ENCODER = encoder
            
            # Load the cached encoder for the batch!
            encoder = _CACHED_IGTR_ENCODER
            
            # 2. Process the batch
            batch_targets = [[] for _ in range(12)]
            
            def force_1d(val):
                """Ensures scalars/empty values are cast to 1D lists so PyTorch can stack them."""
                if val is None: return [0]
                if isinstance(val, (int, float)): return [val]
                if isinstance(val, np.ndarray) and val.ndim == 0: return [val.item()]
                return val
            
            for txt in text_label:
                res = encoder({'label': txt})
                if res is None or not isinstance(res, dict):
                    res = {} # Safe fallback
                
                # Dynamic key finder
                def get_val(hints):
                    for hint in hints:
                        for k, v in res.items():
                            if hint in k and k != 'label':
                                return v
                    return [0]
                
                lbl = res.get('label', [0])
                batch_targets[0].append(lbl)
                
                batch_targets[1].append(force_1d(get_val(['prompt_pos'])))
                batch_targets[2].append(force_1d(get_val(['prompt_char'])))
                batch_targets[3].append(force_1d(get_val(['ques_pos'])))
                batch_targets[4].append(force_1d(get_val(['ques1'])))
                batch_targets[5].append(force_1d(get_val(['ques2_char'])))
                batch_targets[6].append(force_1d(get_val(['ques2_ans'])))
                
                # Index 7: MANUALLY BUILD 'label_ace' (Exactly length 37)
                label_ace = [0] * 37
                for char_idx in lbl:
                    if isinstance(char_idx, int) and char_idx < 37:
                        label_ace[char_idx] += 1
                batch_targets[7].append(label_ace)
                
                batch_targets[8].append(force_1d(get_val(['ques4'])))
                batch_targets[9].append(force_1d(get_val(['ques_len'])))
                batch_targets[10].append(force_1d(get_val(['ques2_len'])))
                batch_targets[11].append(force_1d(get_val(['prompt_len'])))
            
            # 3. Stack into Tensors
            tensor_targets = []
            for i in range(12):
                if i == 0:
                    tensor_targets.append(batch_targets[0])
                else:
                    tensor_targets.append(torch.tensor(np.array(batch_targets[i]), dtype=torch.long, device=device))
            
            return tensor_targets
        else:
            return converter.encode_list(text_label).to(device)
        
def compute_task_loss(preds, true_targets, loss_fn_spatial, cls_loss_type):
    """Intelligently calculates the primary OCR loss regardless of architecture."""
    if 'loss_internal' in preds and preds['loss_internal'] is not None:
        return preds['loss_internal']
    elif cls_loss_type in ['CPPD', 'OTE']:
        return loss_fn_spatial(preds, true_targets)
    else:
        return loss_fn_spatial(preds['logits'], true_targets)

def decode_predictions(logits, cls_loss_type, converter, return_scores=False):
    """Unified API for text decoding."""
    if cls_loss_type == 'CTC':
        return ctc_greedy_decoder(logits, converter, return_scores=return_scores)
    else:
        return viterbi_plate_decoder(logits, converter, return_scores=return_scores)

# In train_utils.py
def update_ema_ghost(model_g, model_ghost, epochs_since_reset, cycle_length, warmup_epochs=0, current_epoch=0):
    """Cyclic EMA Schedule with Warmup. Pulses when Student hits a new peak."""
    if current_epoch < warmup_epochs:
        current_decay = 0.0 # Force a 100% hard copy during the unstable warmup phase
    else:
        base_decay = 0.990   
        max_decay  = 0.9999  
        
        # Smoothly transition from base_decay to max_decay over 'cycle_length' epochs
        effective_epoch = min(epochs_since_reset, cycle_length) 
        
        cosine_val = math.cos(math.pi * effective_epoch / cycle_length)
        schedule_multiplier = (cosine_val + 1.0) / 2.0 
        
        # At peak (effective_epoch=0), multiplier is 1.0 -> current_decay = 0.990
        # At stagnation (effective_epoch=cycle_length), multiplier is 0.0 -> current_decay = 0.9999
        current_decay = max_decay - (max_decay - base_decay) * schedule_multiplier
    
    with torch.no_grad():
        for param_s, param_t in zip(model_g.parameters(), model_ghost.parameters()):
            param_t.data.mul_(current_decay).add_(param_s.data, alpha=1.0 - current_decay)
            
    return current_decay

def compute_distillation_losses(preds_lr, preds_hr, is_hr_mask, config, device):
    """Encapsulates all KD, FDD, CRD, and Latent math."""
    losses = {
        'kl': torch.tensor(0.0, device=device),
        'latent': torch.tensor(0.0, device=device),
        'feature': torch.tensor(0.0, device=device),
        'ocr': torch.tensor(0.0, device=device)
    }
    
    # If there are no valid HR samples in this batch, return zero losses
    if not is_hr_mask.any():
        return losses

    # 1. KL Divergence (Logits)
    teacher_logits = preds_hr['logits'][is_hr_mask].detach()
    temperature = config.get('distill_temp', 2.0)
    student_log_probs = F.log_softmax(preds_lr['logits'][is_hr_mask] / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    losses['kl'] = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

    # 2. High-Resolution Latent Distillation (Cosine + L1)
    if 'latent_lr' in preds_lr and 'latent_lr' in preds_hr:
        s_latent = preds_lr['latent_lr'][is_hr_mask]
        t_latent = preds_hr['latent_lr'][is_hr_mask].detach()
        
        if s_latent.shape[0] > 0:
            loss_l1_latent = F.l1_loss(s_latent, t_latent)
            
            s_flat = s_latent.permute(0, 2, 3, 1).reshape(-1, s_latent.size(1))
            t_flat = t_latent.permute(0, 2, 3, 1).reshape(-1, t_latent.size(1))
            cosine_sim = F.cosine_similarity(s_flat, t_flat, dim=-1)
            loss_cos_latent = 1.0 - cosine_sim.mean()
            
            losses['latent'] = loss_l1_latent + loss_cos_latent

    # 3. Directional Feature Distillation (FDD Trajectory)
    if 'trajectory' in preds_lr and 'trajectory' in preds_hr:
        s_traj = preds_lr['trajectory']
        t_traj = preds_hr['trajectory']
        
        fdd_loss_accum = 0.0
        num_steps = len(s_traj) - 1
        for i in range(num_steps):
            t_prev, t_next = t_traj[i][is_hr_mask].detach(), t_traj[i+1][is_hr_mask].detach()
            s_prev, s_next = s_traj[i][is_hr_mask], s_traj[i+1][is_hr_mask]
            
            delta_t = t_next - t_prev
            delta_s = s_next - s_prev
            
            dir_loss = 1.0 - F.cosine_similarity(delta_s, delta_t, dim=-1).mean()
            mag_s = torch.norm(delta_s, p=2, dim=-1)
            mag_t = torch.norm(delta_t, p=2, dim=-1)
            mag_loss = F.mse_loss(mag_s, mag_t)
            
            fdd_loss_accum += (dir_loss + 0.1 * mag_loss)
            
        losses['feature'] = fdd_loss_accum / max(1, num_steps)

    # 4. Contrastive Token Distillation (InfoNCE)
    if 'query_tokens' in preds_lr and 'query_tokens' in preds_hr:
        feature_dim = preds_lr['query_tokens'].shape[-1]
        teacher_tokens = preds_hr['query_tokens'][is_hr_mask].detach().reshape(-1, feature_dim)
        student_tokens = preds_lr['query_tokens'][is_hr_mask].reshape(-1, feature_dim)
        
        if student_tokens.shape[0] > 0:
            t_norm = F.normalize(teacher_tokens, p=2, dim=-1)
            s_norm = F.normalize(student_tokens, p=2, dim=-1)
            
            sim_matrix = torch.matmul(s_norm, t_norm.T) / 0.1 # 0.1 is Temperature
            
            N = student_tokens.shape[0]
            labels = torch.arange(N, device=device)
            losses['ocr'] = F.cross_entropy(sim_matrix, labels)

    return losses

@register('SROCR_TRAIN')
def SROCR_TRAIN(train_loader, val_loader, model_g, model_ghost, optimizer_g, config, **kwargs):
    device = next(model_g.parameters()).device
    true_converter = strLabelConverter(config.get('alphabet', "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    use_fp16 = config.get('use_fp16', False)
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16) 
    pbar = tqdm(train_loader, leave=False)
    
    save_root = kwargs.get('save_path', Path('.'))
    loss_stats = {'total': [], 'task_s': []}
    
    running_acc_seq_s, running_acc_char_s = 0.0, 0.0
    current_epoch = kwargs.get('epoch', 0)
    epochs_without_improvement = kwargs.get('epochs_without_improvement', 0) # <--- GET THE STATE
    epoch_max_ema = config.get('epoch_max_ema', 50) # Repurpose as the Cycle Length for the EMA schedule
    cls_loss_type = config.get('cls_loss', 'SmoothPoly1')
    use_ema_ghost = config.get('use_ema_ghost', False)
    
    # ---> UPGRADE: Pass the current_epoch into the factory
    loss_fn_spatial = build_loss_function(cls_loss_type, true_converter, device, current_epoch=current_epoch)
    epoch_tracker = ConfusionTracker()
    
    for batch_idx, batch in enumerate(pbar):
        if batch is None: continue        
        
        lr_batch = batch['lr'].to(device, non_blocking=True, memory_format=torch.channels_last)
        text_label = batch['gt'] 
        
        true_targets = prepare_targets(cls_loss_type, text_label, true_converter, device)

        optimizer_g.zero_grad()
            
        with torch.amp.autocast('cuda', enabled=use_fp16):
            # ==========================================
            # 1. STUDENT FORWARD PASS
            # ==========================================
            preds_lr = model_g(lr_batch, temporal_pool=True, tgt=true_targets, epoch=current_epoch)
            if isinstance(preds_lr, (tuple, list)): preds_lr = preds_lr[0]

            # Primary Text Loss
            total_loss = compute_task_loss(preds_lr, true_targets, loss_fn_spatial, cls_loss_type) + 0.0

        if not torch.isfinite(total_loss):
            print(f"⚠️ Warning: Non-finite loss at batch {batch_idx}. Skipping.")
            continue

        # ==========================================
        # 2. BACKWARD PASS & GHOST TRACKING
        # ==========================================
        scaler.scale(total_loss).backward()
        
        scaler.unscale_(optimizer_g)
        torch.nn.utils.clip_grad_norm_(model_g.parameters(), max_norm=5.0)
        
        scaler.step(optimizer_g)
        scaler.update()
        
        # SILENT GHOST UPDATE
        if use_ema_ghost and model_ghost is not None:
            warmup_epochs = config.get('ema_warmup_epochs', 5)
            
            # ---> THE UPGRADE: Cyclic EMA Reset
            current_decay = update_ema_ghost(
                model_g, model_ghost, 
                epochs_since_reset=epochs_without_improvement, 
                cycle_length=epoch_max_ema, 
                warmup_epochs=warmup_epochs, 
                current_epoch=current_epoch
            )
        # ====================================================================
        # METRICS & LOGGING
        # ====================================================================
        with torch.no_grad():
            loss_stats['total'].append(total_loss.item())
            loss_stats['task_s'].append(total_loss.item()) 
                
            if batch_idx % 5 == 0:
                decoded_s = decode_predictions(preds_lr['logits'], cls_loss_type, true_converter)
                
                acc_s_seq = sum([1 for p, t in zip(decoded_s, text_label) if p == t]) / len(text_label)
                running_acc_seq_s = (running_acc_seq_s * 0.9) + (acc_s_seq * 0.1) if running_acc_seq_s > 0 else acc_s_seq
                
                total_chars = sum(len(t) for t in text_label)
                correct_chars_s = sum(sum(1 for pc, tc in zip(p, t) if pc == tc) for p, t in zip(decoded_s, text_label))
                acc_s_chr = correct_chars_s / max(total_chars, 1)
                running_acc_char_s = (running_acc_char_s * 0.9) + (acc_s_chr * 0.1) if running_acc_char_s > 0 else acc_s_chr
                
                epoch_tracker.update(decoded_s, text_label) 
                
                
                postfix_dict = {
                    'Loss': f"{np.mean(loss_stats['total'][-50:]):.4f}",
                    'S-Seq': f"{running_acc_seq_s:.1%}", 
                    'S-Chr': f"{running_acc_char_s:.1%}",
                }

                if use_ema_ghost:
                    postfix_dict['EMA'] = f"{current_decay:.4f}"

                
                pbar.set_postfix(postfix_dict)

    final_loss = np.mean(loss_stats['total']) if loss_stats['total'] else 0.0
    return final_loss

@register('SROCR_VAL')
def SROCR_VAL(val_loader, model_g, model_ghost, config, **kwargs):
    model_g.eval() 
    if model_ghost is not None:
        model_ghost.eval()
        
    device = next(model_g.parameters()).device
    true_converter = strLabelConverter(config.get('alphabet', "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    cls_loss_type = config.get('cls_loss', 'SmoothPoly1')
    
    # ---> UPGRADE: Pass current_epoch=None to lock validation to max_epsilon
    loss_fn_spatial = build_loss_function(cls_loss_type, true_converter, device, current_epoch=None)
    
    val_tracker = ConfusionTracker()
    save_root = kwargs.get('save_path', Path('.'))
    
    correct_sequences_s = 0
    correct_sequences_t = 0
    total_sequences = 0
    
    # Track the running validation loss
    total_val_loss = 0.0
    val_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            lr_seqs = batch['lr_seq'].to(device, non_blocking=True)
            text_labels = batch['gt']     

            B, Seq_Len, C, H, W = lr_seqs.shape
            
            # Flatten the images (B * Seq_Len, C, H, W)
            flat_imgs = lr_seqs.view(B * Seq_Len, C, H, W).contiguous().to(memory_format=torch.channels_last)
            
            # 2. Expand the labels to match the flattened sequence length
            flat_labels = []
            for label in text_labels:
                flat_labels.extend([label] * Seq_Len)
                
            # 3. Prepare the mathematically correct targets for the loss function
            true_targets = prepare_targets(cls_loss_type, flat_labels, true_converter, device, is_training=False)
            
            # ==========================================
            # 1. STUDENT INFERENCE & LOSS COMPUTATION
            # ==========================================
            with torch.amp.autocast('cuda', enabled=config.get('use_fp16', False)):
                output_s = model_g(flat_imgs, temporal_pool=True) 
                if isinstance(output_s, (tuple, list)): output_s = output_s[0]
                logits_s = output_s['logits']
                
                # Calculate Validation Loss
                if loss_fn_spatial is not None:
                    loss_s = compute_task_loss(output_s, true_targets, loss_fn_spatial, cls_loss_type)
                    if torch.isfinite(loss_s):
                        total_val_loss += loss_s.item()
                        val_batches += 1

            # ==========================================
            # 2. GHOST INFERENCE (Fixed Variable Name)
            # ==========================================
            logits_t = None
            if model_ghost is not None:
                with torch.amp.autocast('cuda', enabled=config.get('use_fp16', False)):
                    output_t = model_ghost(flat_imgs, temporal_pool=True)
                    if isinstance(output_t, (tuple, list)): output_t = output_t[0]
                    logits_t = output_t['logits']

            # ==========================================
            # 3. TEMPORAL SOFT ENSEMBLING (Zero-Cost Upgrade)
            # ==========================================
            # logits_s shape: (B * Seq_Len, T, C)
            _, T_len, C_classes = logits_s.shape
            
            # Reshape to group by sequence: (B, Seq_Len, T, C)
            logits_seq_s = logits_s.view(B, Seq_Len, T_len, C_classes)
            
            # Convert to probabilities and Mean-Pool across the 5 frames
            probs_seq_s = torch.softmax(logits_seq_s, dim=-1)
            fused_probs_s = probs_seq_s.mean(dim=1) # Shape: (B, T, C)
            
            # Convert back to pseudo-logits for the decoder
            fused_logits_s = torch.log(fused_probs_s + 1e-8)

            # Do the same for the Ghost if it exists
            if logits_t is not None:
                logits_seq_t = logits_t.reshape(B, Seq_Len, T_len, C_classes)
                fused_probs_t = torch.softmax(logits_seq_t, dim=-1).mean(dim=1)
                fused_logits_t = torch.log(fused_probs_t + 1e-8)

            # ==========================================
            # 4. DECODING (Once per sequence!)
            # ==========================================
            if cls_loss_type == 'CTC':
                seq_preds_s, seq_scores_s = ctc_greedy_decoder(fused_logits_s, true_converter, return_scores=True)
                if logits_t is not None:
                    seq_preds_t, seq_scores_t = ctc_greedy_decoder(fused_logits_t, true_converter, return_scores=True)
            else:
                seq_preds_s = decode_batch_logits(fused_logits_s, true_converter)
                seq_scores_s = F.log_softmax(fused_logits_s, dim=-1).max(dim=-1)[0].sum(dim=1)
                
                if logits_t is not None:
                    seq_preds_t = decode_batch_logits(fused_logits_t, true_converter)
                    seq_scores_t = F.log_softmax(fused_logits_t, dim=-1).max(dim=-1)[0].sum(dim=1)

            # ==========================================
            # 5. SCORING
            # ==========================================
            for b in range(B):
                gt = text_labels[b]  # Only need 1 GT per sequence now
                total_sequences += 1
                
                if seq_preds_s[b] == gt: correct_sequences_s += 1
                val_tracker.update([seq_preds_s[b]], [gt])
                
                if logits_t is not None and seq_preds_t[b] == gt: 
                    correct_sequences_t += 1
            
    acc_seq_s = correct_sequences_s / total_sequences if total_sequences else 0.0
    acc_seq_t = correct_sequences_t / total_sequences if total_sequences else 0.0
    avg_val_loss = total_val_loss / val_batches if val_batches > 0 else 0.0

    print(f"\n{'='*30} SEQUENCE EVALUATION {'='*30}")
    print(f"Total Sequences: {total_sequences}")
    print(f"Student Acc: {acc_seq_s:.2%} | Ghost (EMA) Acc: {acc_seq_t:.2%}")
    print(f"{'='*81}\n")
    
    total_failures = val_tracker.get_worst_pairs_dict(top_k=50) 
    with open(save_root / 'confusion_stats.json', 'w+') as f: 
        json.dump(total_failures, f, indent=4)
    
    # Returns the mathematically valid loss instead of 0.0
    return avg_val_loss, acc_seq_s, acc_seq_t