import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import ones_, trunc_normal_, zeros_

# 1. Import your Custom Backbone Components
from models.custom.VSR_curvature_att import FReLU, HighContrastGate, DeformableProj

# 2. Import common Transformer helpers from the OpenOCR codebase
from models.common import DropPath, Identity, Mlp, Embeddings

# ==============================================================================
# 1. TRUE DEFORMABLE ATTENTION (No K projection, No Dot Product)
# ==============================================================================
class PureDeformableAttention(nn.Module):
    def __init__(self, dim, num_heads=8, num_points=4): # Standard DETR uses 4 points per head
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = dim // num_heads

        # 1. Predicts the Center Reference Point (cx, cy)
        self.ref_predictor = nn.Linear(dim, 2)

        # 2. Predicts Offsets: (Heads * Points * 2) 
        # Every head gets to look at its own unique set of points!
        self.offset_predictor = nn.Linear(dim, num_heads * num_points * 2)

        # 3. Predicts Attention Weights directly from Q (Heads * Points)
        self.weight_predictor = nn.Linear(dim, num_heads * num_points)

        # 4. Only Value Projection (No Q or K projection needed for attention matching!)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1) 
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query, feature_map):
        B, L, C = query.shape
        _, _, H, W = feature_map.shape

        # 1. Reference Points [0, 1]
        ref_points = torch.sigmoid(self.ref_predictor(query)) # (B, L, 2)
        ref_points = ref_points.unsqueeze(2).unsqueeze(2)     # (B, L, 1, 1, 2) for broadcasting

        # 2. Predict Offsets & Weights strictly from the Query
        offsets = self.offset_predictor(query).reshape(B, L, self.num_heads, self.num_points, 2)
        
        # Predict weights and normalize across the K sampled points
        weights = self.weight_predictor(query).reshape(B, L, self.num_heads, self.num_points)
        weights = F.softmax(weights, dim=-1) # (B, L, Heads, Points)

        # 3. Calculate Sampling Locations
        sample_locs = ref_points + offsets # (B, L, Heads, Points, 2)
        sample_locs = torch.clamp(sample_locs, 0.0, 1.0)
        
        # Convert [0, 1] to [-1, 1] for grid_sample
        sample_grids = sample_locs * 2.0 - 1.0 
        
        # --- THE SHAPE FIX STARTS HERE ---
        # Route each Head to its specific 32-channel slice by folding Batch and Heads!
        sample_grids = sample_grids.transpose(1, 2) # (B, Heads, L, Points, 2)
        sample_grids = sample_grids.reshape(B * self.num_heads, L * self.num_points, 1, 2)

        # 4. Value Projection
        v_map = self.v_proj(feature_map) # (B, C, H, W)
        
        # Reshape the feature map to match the folded batch
        v_map = v_map.reshape(B * self.num_heads, self.head_dim, H, W)

        # 5. Sample the Values (Now each head ONLY extracts its own 32 channels!)
        sampled_v = F.grid_sample(
            v_map, sample_grids, mode='bilinear', padding_mode='zeros', align_corners=False
        ) # Output: (B * Heads, Head_Dim, L * Points, 1)

        # 6. Unpack the shapes back to normal
        sampled_v = sampled_v.reshape(B, self.num_heads, self.head_dim, L, self.num_points)
        
        # Rearrange to: (B, L, Heads, Points, Head_Dim)
        sampled_v = sampled_v.permute(0, 3, 1, 4, 2)

        # 7. Apply Predicted Attention Weights
        # Multiply the sampled values directly by the query's predicted weights and sum them up
        out = torch.einsum('blhk,blhkd->blhd', weights, sampled_v)
        
        # 8. Final Projection
        out = out.reshape(B, L, C)
        return self.out_proj(out)

# ==============================================================================
# 2. THE TRANSFORMER LAYERS
# ==============================================================================
class PureDeformableLayer(nn.Module):
    def __init__(self, dim, num_heads=8, num_points=4, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        
        # Retained the Neighbor Linker because sequence ordering is still critical for OCR
        self.neighbor_linker = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.GroupNorm(1, dim)
        )
        
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        self.cross_attn = PureDeformableAttention(dim, num_heads, num_points)
        
        self.norm3 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim), 
            nn.GELU(), 
            # Added Dropout to fight the overfitting you saw!
            nn.Dropout(0.1), 
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(0.1)
        )

    def forward(self, query, feature_map):
        q1 = self.norm1(query)
        
        q_linked = self.neighbor_linker(q1.transpose(1, 2)).transpose(1, 2)
        q1 = q1 + q_linked 
        
        q_self, _ = self.self_attn(q1, q1, q1)
        query = query + q_self
        
        q2 = self.norm2(query)
        # No more tuple unpacking. Pure Deformable Attention just returns the tensor.
        q_cross = self.cross_attn(q2, feature_map)
        query = query + q_cross
        
        q3 = self.norm3(query)
        q_mlp = self.mlp(q3)
        return query + q_mlp

class HeavyQueryGenerator(nn.Module):
    def __init__(self, in_channels, dim=256, num_chars=12):
        super().__init__()
        self.num_chars = num_chars
        
        # This is your "Localization Engine"
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            FReLU(64), 
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            HighContrastGate(128), 
            DeformableProj(128, 256, kernel_size=3, stride=2, offset_groups=4),
            nn.GroupNorm(8, 256),
            FReLU(256), 
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), 
            nn.GroupNorm(8, 256),
            FReLU(256)
        )
        
        self.surgical_focus = SurgicalFocusBlock(in_channels=256, reduction=8)

        # The MLP that turns visual context into 12 unique character queries
        self.net = nn.Sequential(
            nn.Linear(256, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, num_chars * dim)
        )

    def forward(self, x):
        # 1. Deep Feature Extraction for localization
        feat = self.stem(x) 
        feat = self.surgical_focus(feat) 
        
        # 2. Global "Glance" at the image
        context = feat.mean(dim=(2, 3)) 
        
        # 3. Generate the 12 Character Queries
        queries = self.net(context).view(-1, self.num_chars, 256)
        return queries

class PureDeformableSpotter(nn.Module):
    def __init__(self, in_channels, dim=256, num_classes=39, num_chars=12, num_layers=4):
        super().__init__()
        self.query_generator = HeavyQueryGenerator(in_channels, dim, num_chars)
        
        # ---------------------------------------------------------
        # ---> 1. ADD IT HERE: The Structural DNA (Identity) <---
        # ---------------------------------------------------------
        self.slot_embed = nn.Parameter(torch.randn(1, num_chars, dim))
        
        # Project raw input features to Transformer dim (256) for the "Memory" path
        self.memory_proj = nn.Conv2d(in_channels, dim, kernel_size=1)
        
        self.layers = nn.ModuleList([PureDeformableLayer(dim=dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        # ---------------------------------------------------------
        # ---> 2. ADD IT HERE: Generator Output + Static Identity <---
        # ---------------------------------------------------------
        query = self.query_generator(x) + self.slot_embed
        
        # Path B: Project the memory features the transformer will attend to
        feature_map = self.memory_proj(x)
        
        for layer in self.layers:
            query = layer(query, feature_map)
            
        query = self.norm(query)
        return self.classifier(query)

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
    def __init__(self, input_shape=(128, 48, 144), num_classes=37, num_chars=7, **kwargs):
        super().__init__()
        in_channels = input_shape[0] 
        
        # self.stem = nn.Sequential(
        #     nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
        #     nn.GroupNorm(8, 64),
        #     FReLU(64), 
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        #     nn.GroupNorm(8, 128),
        #     HighContrastGate(128), 
        #     DeformableProj(128, 256, kernel_size=3, stride=2, offset_groups=4),
        #     nn.GroupNorm(8, 256),
        #     FReLU(256), 
        #     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), 
        #     nn.GroupNorm(8, 256),
        #     FReLU(256)
        # )
        
        
        
        # The new Pure Deformable Spotter
        self.spotter = PureDeformableSpotter(
            in_channels=in_channels,
            dim=256, 
            num_classes=num_classes, 
            num_chars=num_chars, 
            num_layers=4
        )

    def forward(self, x, tgt=None, epoch=0, **kwargs):
        # feat = self.stem(x) 
        # feat = self.surgical_focus(feat) 
        
        logits = self.spotter(x)
        
        preds = {
            'logits': logits,        
            'node_feats': None,      
            'edge_feats': logits,    
            # We no longer care about visualization, so return None
            'attn_maps': None           
        }
        
        return preds