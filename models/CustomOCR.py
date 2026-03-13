import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from models.VSR_curvature_att import FReLU
from models.VSR_curvature_att import HighContrastGate, DeformableProj
# ==============================================================================
# 1. HELPER BLOCKS 
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

class LayoutScout(nn.Module):
    def __init__(self, in_channels=128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1) 
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, 2) 
        )
    def forward(self, x):
        return self.head(self.pool(x))

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(d_model, height, width)
        d_y = d_model // 2; d_x = d_model - d_y
        div_term_y = torch.exp(torch.arange(0, d_y, 2).float() * (-math.log(10000.0) / d_y))
        div_term_x = torch.exp(torch.arange(0, d_x, 2).float() * (-math.log(10000.0) / d_x))
        pos_y = torch.arange(0, height).unsqueeze(1)
        pe[0:d_y:2, :, :] = torch.sin(pos_y * div_term_y).transpose(0, 1).unsqueeze(-1).repeat(1, 1, width)
        pe[1:d_y:2, :, :] = torch.cos(pos_y * div_term_y).transpose(0, 1).unsqueeze(-1).repeat(1, 1, width)
        pos_x = torch.arange(0, width).unsqueeze(1)
        pe[d_y::2, :, :] = torch.sin(pos_x * div_term_x).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_y+1::2, :, :] = torch.cos(pos_x * div_term_x).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        self.register_buffer('pe', pe)
    def forward(self, x): return self.dropout(x + self.pe)

# ==============================================================================
# 2. ViT EXPERT (With Teacher Forcing)
# ==============================================================================
class ViT_CrossAttn_OCR(nn.Module):
    def __init__(self, in_channels=256, d_model=256, num_chars=8, num_classes=37, num_layers=4, num_heads=16):
        super().__init__()
        self.d_model = d_model
        
        # Inside ViT_CrossAttn_OCR.__init__
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, d_model // 2, kernel_size=3, stride=1, padding=1, padding_mode='replicate'), # <--- FIXED
            nn.BatchNorm2d(d_model // 2), nn.ReLU(True),
            
            nn.Conv2d(d_model // 2, d_model, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),     # <--- FIXED
            nn.BatchNorm2d(d_model), nn.ReLU(True),
            
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1, padding_mode='replicate')           # <--- FIXED
        )
        self.input_norm = nn.LayerNorm(d_model)

        # Inside ViT_CrossAttn_OCR.__init__
        self.num_patches = (64 // 4) * (192 // 4) # Now 768 tokens
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=1.0)

        # Base Queries
        self.char_queries = nn.Parameter(torch.randn(1, num_chars, d_model) * 1.0)
        
        # ðTEACHER FORCING: Embedding to convert GT class to d_model vector
        self.char_embed = nn.Embedding(num_classes, d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=1024,
            activation='gelu', dropout=0.1, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))

    def random_masking(self, x, mask_ratio):
        B, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        mask = mask.unsqueeze(-1)
        
        x_masked = x * (1 - mask) 
        mask_token_expanded = self.mask_token.expand(B, L, D)
        x_masked = x_masked + (mask_token_expanded * mask)
        
        return x_masked

    def generate_par_mask(self, batch_size, device):
        # --- FIX 1: Change back to 7x7 mask (No longer 8x8) ---
        mask = torch.zeros((batch_size, 7, 7), device=device)
        probs = torch.rand(batch_size, device=device)
        neg_inf = -10000.0 
        
        ltr_indices = (probs < 0.33)
        if ltr_indices.any():
            char_mask = torch.triu(torch.ones(7, 7, device=device), diagonal=1) * neg_inf
            # --- FIX 2: Apply to the whole mask, no longer skipping index 0 ---
            mask[ltr_indices, :, :] = char_mask 

        rtl_indices = (probs >= 0.33) & (probs < 0.66)
        if rtl_indices.any():
            char_mask = torch.tril(torch.ones(7, 7, device=device), diagonal=-1) * neg_inf
            # --- FIX 3: Apply to the whole mask ---
            mask[rtl_indices, :, :] = char_mask 
            
        nhead = self.decoder.layers[0].self_attn.num_heads
        return mask.repeat_interleave(nhead, dim=0)

    def forward(self, x, refine_iters=1, tgt=None, forcing_prob=0.0, return_attn=False):
        B = x.size(0)

        # 1. Visual Tokens
        features = self.patch_embed(x)
        vis_tokens = features.flatten(2).transpose(1, 2)
        vis_tokens = self.input_norm(vis_tokens) 

        # 2. Positional Encoding (Fixing the hardcoded 8, 24)
        if vis_tokens.shape[1] != self.pos_embed.shape[1]:
            # The learned pos_embed was initialized for 64x192 (16x48 patches)
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2).reshape(1, self.d_model, 16, 48), # <--- FIXED
                size=(features.shape[2], features.shape[3]), mode='bilinear'
            ).flatten(2).transpose(1, 2)
            vis_tokens = vis_tokens + pos_embed
        else:
            vis_tokens = vis_tokens + self.pos_embed

        # 3. Apply Masking AFTER Positional Encoding
        if self.training:
            vis_tokens = self.random_masking(vis_tokens, mask_ratio=0.1)

        # 4. Queries & Teacher Forcing
        current_queries = self.char_queries.expand(B, -1, -1)
        
        # --- 4. APPLY TEACHER FORCING ONLY TO CHARACTERS (Indices 1-7) ---
        # --- FIX 4: Apply Teacher Forcing to ALL queries ---
        if self.training and tgt is not None and forcing_prob > 0.0:
            tgt_emb = self.char_embed(tgt) 
            mask = (torch.rand(B, 7, 1, device=x.device) < forcing_prob).float()
            
            # No longer skipping index 0
            current_queries = (current_queries * (1.0 - mask)) + (tgt_emb * mask)

        tgt_mask = self.generate_par_mask(B, x.device) if self.training else None
        
        final_tokens = None
        loops = refine_iters + 1
        attention_maps = None 
        
        # 5. Native Decoder Loop
        for i in range(loops):
            out_tokens = self.decoder(tgt=current_queries, memory=vis_tokens, tgt_mask=tgt_mask)
            if i < loops - 1:
                current_queries = out_tokens
            final_tokens = out_tokens

        attention_maps = None
        # Extract maps if explicitly requested (regardless of training mode)
        if return_attn:
            _, attention_maps = self.decoder.layers[-1].multihead_attn(
                query=final_tokens, 
                key=vis_tokens, 
                value=vis_tokens, 
                need_weights=True, 
                average_attn_weights=False
            )

        logits = self.head(final_tokens)
        return logits, final_tokens, attention_maps
# ==============================================================================
# 3. WRAPPER CLASS
# ==============================================================================
# ==============================================================================
# 3. WRAPPER CLASS
# ==============================================================================
class CustomOCR(nn.Module):
    def __init__(self, input_shape=(128, 64, 192), num_classes=37, num_chars=7, d_model=384, num_heads=16):
        super().__init__()
        in_channels = input_shape[0]
        
        # --- NEW: ADVANCED SEMANTIC STEM ---
        self.stem = nn.Sequential(
            # 1. Initial Projection & Noise Cleanup 
            nn.Conv2d(in_channels, 128, 3, 1, 1, padding_mode='replicate'),
            nn.GroupNorm(8, 128),
            HighContrastGate(128), 
            
            # 2. Deformable Structural Alignment 
            DeformableProj(128, 256, kernel_size=3, offset_groups=4),
            nn.GroupNorm(8, 256),
            FReLU(256), 
            
            # 3. Final Semantic Refinement 
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, padding_mode='replicate'), 
            nn.GroupNorm(8, 256),
            FReLU(256)
        )
        
        # Update Positional Encoding for the 64x192 resolution
        self.pos_encoder = PositionalEncoding2D(256, 64, 192)
        self.surgical_focus = SurgicalFocusBlock(in_channels=256, reduction=8)
        
        # ViT Expert must now handle 768 tokens (16x48 grid)
        self.vit_expert = ViT_CrossAttn_OCR(
            in_channels=256, 
            d_model=d_model, 
            num_chars=num_chars, # EXACTLY 7 QUERIES
            num_classes=num_classes,
            num_heads=num_heads  
        )
        self.projector = nn.Sequential(nn.Linear(d_model, 128), nn.Mish(), nn.Linear(128, 128))

        # REMOVED: self.layout_head

    def forward(self, x, tgt=None, epoch=0, **kwargs):
        feat = self.stem(x) 
        feat = self.pos_encoder(feat)
        feat = self.surgical_focus(feat)
        
        # SCHEDULED SAMPLING
        forcing_prob = max(0.0, 0.5 - (epoch * 0.02)) if self.training else 0.0
        
        iters = 1 if self.training else 2
        
        # logits: (B, 7, 37) | all_tokens: (B, 7, 384) | attn_maps: (B, H, 7, 768)
        logits, all_tokens, attn_maps = self.vit_expert(feat, refine_iters=iters, tgt=tgt, forcing_prob=forcing_prob, return_attn=True)
        
        # All tokens belong to characters!
        char_tokens = all_tokens  # (B, 7, 384)
        char_logits = logits      # (B, 7, 37)
        
        # REMOVED: global_token and layout_logits calculation
        
        z_vector = F.normalize(self.projector(char_tokens), p=2, dim=-1)
        
        # --- RETURN THE CLEANED DICTIONARY ---
        return {
            'logits': char_logits,            # (B, 7, 37) - Ready for FocalLoss
            'features': char_tokens,          # (B, 7, 384)
            'z_vector': z_vector,             # (B, 7, 128)
            'attn_maps': attn_maps,           # (B, H, 7, 768) - Sent to AdvancedAttentionLoss!
            'theta': None,
            'crops': None 
        }

