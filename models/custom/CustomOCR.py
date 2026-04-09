import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import ones_, trunc_normal_, zeros_
from models.ctc_decoder import CTCDecoder
# 1. Import your Custom Backbone Components
from models.custom.VSR_curvature_att import FReLU, HighContrastGate, DeformableProj

# 2. Import common Transformer helpers from the OpenOCR codebase
from models.common import DropPath, Identity, Mlp, Embeddings

# ==============================================================================
# 1. TRUE DEFORMABLE ATTENTION (No K projection, No Dot Product)
# ==============================================================================
class PureDeformableAttention(nn.Module):
    def __init__(self, dim, num_heads=8, num_points=4): 
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = dim // num_heads

        # REMOVED ref_predictor! We will pass explicit, ordered anchors.

        self.offset_predictor = nn.Linear(dim, num_heads * num_points * 2)
        self.weight_predictor = nn.Linear(dim, num_heads * num_points)

        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1) 
        self.out_proj = nn.Linear(dim, dim)

        # ---> IMPROVEMENT: Zero-Init Offsets (Prevents blind lassos)
        nn.init.constant_(self.offset_predictor.weight, 0.0)
        nn.init.constant_(self.offset_predictor.bias, 0.0)

    # ---> NEW: Accept explicit ref_points
    def forward(self, query, feature_map, ref_points):
        B, L, C = query.shape
        _, _, H, W = feature_map.shape

        # 1. Format Fixed Reference Points (B, L, 1, 1, 2)
        ref_points = ref_points.unsqueeze(2).unsqueeze(2).expand(B, -1, 1, 1, -1)

        # 2. Predict Offsets & Weights (Starting from exactly 0.0!)
        offsets = self.offset_predictor(query).reshape(B, L, self.num_heads, self.num_points, 2)
        weights = F.softmax(self.weight_predictor(query).reshape(B, L, self.num_heads, self.num_points), dim=-1)

        # 3. Calculate Sampling Locations safely
        sample_locs = torch.clamp(ref_points + offsets, 0.0, 1.0)
        sample_grids = (sample_locs * 2.0 - 1.0).transpose(1, 2) 
        sample_grids = sample_grids.reshape(B * self.num_heads, L * self.num_points, 1, 2)

        v_map = self.v_proj(feature_map).reshape(B * self.num_heads, self.head_dim, H, W)

        sampled_v = F.grid_sample(v_map, sample_grids, mode='bilinear', padding_mode='zeros', align_corners=False) 
        sampled_v = sampled_v.reshape(B, self.num_heads, self.head_dim, L, self.num_points).permute(0, 3, 1, 4, 2)

        out = torch.einsum('blhk,blhkd->blhd', weights, sampled_v).reshape(B, L, C)
        return self.out_proj(out)

class PureDeformableLayer(nn.Module):
    def __init__(self, dim, num_heads=4, num_points=8, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = PureDeformableAttention(dim, num_heads, num_points)
        
        self.norm3 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim), nn.GELU(), nn.Dropout(0.1), 
            nn.Linear(mlp_hidden_dim, dim), nn.Dropout(0.1)
        )

    # ---> NEW: Pass ref_points down the chain
    def forward(self, query, pos_embed, feature_map, ref_points):
        q_pos = self.norm1(query) + pos_embed
        q_self, _ = self.self_attn(q_pos, q_pos, q_pos)
        query = query + q_self
        
        q2_pos = self.norm2(query) + pos_embed
        q_cross = self.cross_attn(q2_pos, feature_map, ref_points)
        query = query + q_cross
        
        return query + self.mlp(self.norm3(query))

class PureDeformableSpotter(nn.Module):
    def __init__(self, in_channels, dim=128, num_classes=39, num_chars=7, num_layers=2):
        super().__init__()
        self.query_embed = nn.Parameter(torch.randn(1, num_chars, dim) * 0.02)
        self.memory_proj = nn.Conv2d(in_channels, dim, kernel_size=1)

        # ==========================================
        # 1. THE FORMAT-AWARE GEOMETRY
        # ==========================================
        # Layout A: Mercosur (Equidistant spacing)
        x_merc = torch.linspace(0.05, 0.95, num_chars)
        y_merc = torch.full((num_chars,), 0.5) 
        self.ref_points_mercosur = nn.Parameter(torch.stack([x_merc, y_merc], dim=-1).unsqueeze(0))

        # Layout B: Old Brazilian (Gap between char 3 and 4)
        # Spatial positions roughly mimicking: LLL (gap) NNNN
        x_old = torch.tensor([0.05, 0.18, 0.31,   0.55, 0.68, 0.81, 0.94])
        y_old = torch.full((num_chars,), 0.5)
        self.ref_points_old = nn.Parameter(torch.stack([x_old, y_old], dim=-1).unsqueeze(0))

        # 2. The Format Router (Global Image Context -> 2 Classes)
        self.format_router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, 2) # Outputs: [Logit_Mercosur, Logit_Old]
        )
        # ==========================================

        self.layers = nn.ModuleList([PureDeformableLayer(dim=dim) for _ in range(num_layers)])
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        pos_embed = self.query_embed.expand(B, -1, -1)
        query = torch.zeros_like(pos_embed)
        
        feature_map = self.memory_proj(x)
        
        # ==========================================
        # 2. DYNAMIC GEOMETRY ROUTING
        # ==========================================
        # Guess the layout based on the semantic edge map
        format_logits = self.format_router(feature_map) # Shape: (B, 2)
        format_weights = F.softmax(format_logits, dim=-1) # Shape: (B, 2)
        
        # Isolate the weights and reshape them for broadcasting
        w_merc = format_weights[:, 0].view(B, 1, 1)
        w_old = format_weights[:, 1].view(B, 1, 1)
        
        # Expand the static parameters to match the batch size
        ref_merc_b = self.ref_points_mercosur.expand(B, -1, -1)
        ref_old_b = self.ref_points_old.expand(B, -1, -1)
        
        # The Differentiable Blend: The winning layout pulls the points to its physical location!
        dynamic_ref_points = (w_merc * ref_merc_b) + (w_old * ref_old_b)
        # ==========================================
        
        for layer in self.layers:
            # Pass the dynamically generated points into the Attention mechanism!
            query = layer(query, pos_embed, feature_map, dynamic_ref_points)
            
        logits = self.classifier(query)
        
        # We must return the format_logits so the loss function can train the Router!
        return logits, query, feature_map, format_logits

class QueryRefinementNet(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.local_mixer = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GroupNorm(1, dim),
            nn.GELU()
        )
        self.seq_mixer = nn.GRU(dim, dim // 2, bidirectional=True, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query):
        residual = query
        q_local = self.local_mixer(query.transpose(1, 2)).transpose(1, 2)
        q_seq, _ = self.seq_mixer(q_local)
        return self.norm(q_seq + residual)

# ==============================================================================
# 3. SURGEON HELPER BLOCKS & CUSTOM OCR
# ==============================================================================
class SurgicalFocusBlock(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.GroupNorm(4, mip) 
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x) 
        x_w = self.pool_w(x).permute(0, 1, 3, 2) 
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_w * a_h

class CustomOCR(nn.Module):
    def __init__(self, input_shape=(128, 32, 96), num_classes=37, num_chars=7, **kwargs):
        super().__init__()
        in_channels = input_shape[0] 
        
        self.spotter = PureDeformableSpotter(
            in_channels=in_channels,
            dim=128, 
            num_classes=num_classes, 
            num_chars=num_chars, 
            num_layers=2
        )

    def forward(self, x, tgt=None, epoch=0, **kwargs):
        # ---> THE FIX: Catch the 4th output (format_logits)
        logits, query_tokens, feature_map, format_logits = self.spotter(x)
        
        preds = {
            'logits': logits,        
            'query_tokens': query_tokens,  
            'ocr_features': feature_map,  
            'format_logits': format_logits, # <--- Expose to train_utils!
            'edge_feats': None,    
            'attn_maps': None           
        }
        return preds