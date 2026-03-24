import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from models.custom.VSR_curvature_att import FReLU
from models.custom.VSR_curvature_att import HighContrastGate, DeformableProj
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

class RotaryEmbedding2D(nn.Module):
    def __init__(self, dim, max_h=32, max_w=96): # Plentiful headroom
        super().__init__()
        # 1. We allocate exactly half the dimension space to Y, and half to X
        half_dim = dim // 2
        
        # 2. Step by 2 (because each frequency covers a sin/cos pair)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, 2).float() / half_dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # 3. Pre-calculate the absolute maximum grids
        seq_y = torch.arange(max_h).float()
        seq_x = torch.arange(max_w).float()
        
        freqs_y = torch.einsum("i,j->ij", seq_y, self.inv_freq)
        freqs_x = torch.einsum("i,j->ij", seq_x, self.inv_freq)
        
        # 4. Repeat for sin/cos 
        freqs_y = torch.cat((freqs_y, freqs_y), dim=-1)
        freqs_x = torch.cat((freqs_x, freqs_x), dim=-1)
        
        # 5. Expand to 2D matrices
        freqs_y = freqs_y.unsqueeze(1).expand(-1, max_w, -1) # (max_h, max_w, half_dim)
        freqs_x = freqs_x.unsqueeze(0).expand(max_h, -1, -1) # (max_h, max_w, half_dim)
        
        self.register_buffer("freqs_y", freqs_y)
        self.register_buffer("freqs_x", freqs_x)

    def forward(self, h, w):
        # Dynamically slice the grid to match the exact patch dimensions.
        # This prevents X and Y coordinates from corrupting if the aspect ratio changes!
        fy = self.freqs_y[:h, :w, :]
        fx = self.freqs_x[:h, :w, :]
        
        # Combine X and Y, then safely flatten
        freqs = torch.cat((fy, fx), dim=-1) # (h, w, dim)
        return freqs.reshape(h * w, -1)     # (h*w, dim)

def apply_rotary_emb(x, freqs):
    # x: (Batch, Heads, SeqLen, HeadDim)
    # freqs: (SeqLen, HeadDim)
    
    # Split the last dimension into pairs to apply the rotation matrix
    x1, x2 = x[..., ::2], x[..., 1::2]
    x_rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
    
    # Apply standard trigonometric rotation
    return (x * freqs.cos()) + (x_rotated * freqs.sin())

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
    def forward(self, x): 
        # If the input shape matches the default 64x192 PE, do the standard addition
        if x.size(2) == self.pe.size(1) and x.size(3) == self.pe.size(2):
            return self.dropout(x + self.pe)
        
        # --- THE TTA FIX: Interpolate the PE grid to match the new upscale ---
        pe_resized = F.interpolate(
            self.pe.unsqueeze(0),         # Make it 4D for the interpolator: (1, C, H, W)
            size=(x.size(2), x.size(3)),  # Stretch to the new TTA dimensions (e.g., 70x210)
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)                      # Back to 3D
        
        return self.dropout(x + pe_resized)

class CosineClassifierHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        # Learnable temperature scalar, initialized to 20.0 (common for cosine margins)
        self.tau = nn.Parameter(torch.tensor(20.0))

    def forward(self, x):
        # Normalize features (X) and weights (W) to magnitude of 1
        x_norm = F.normalize(x, p=2, dim=-1, eps=1e-6)
        w_norm = F.normalize(self.weight, p=2, dim=-1, eps=1e-6)
        
        # Calculate cosine similarity and scale by temperature
        # x_norm: (B, 7, 384) | w_norm: (37, 384) -> logits: (B, 7, 37)
        logits = F.linear(x_norm, w_norm) * self.tau
        return logits

# ==============================================================================
# 2. ViT EXPERT (With Teacher Forcing)
# ==============================================================================
class ViT_CrossAttn_OCR1(nn.Module):
    def __init__(self, in_channels=256, d_model=256, num_chars=8, num_classes=37, num_layers=4, num_heads=16):
        super().__init__()
        self.d_model = d_model
        
        # Inside ViT_CrossAttn_OCR.__init__
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, d_model // 2, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, d_model // 2), 
            FReLU(d_model // 2),
            
            nn.Conv2d(d_model // 2, d_model, kernel_size=3, stride=2, padding=1), 
            nn.GroupNorm(8, d_model), 
            FReLU(d_model),
            
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        )
        self.input_norm = nn.LayerNorm(d_model)

        # Inside ViT_CrossAttn_OCR.__init__
        self.num_patches = (64 // 4) * (192 // 4) # Now 768 tokens
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=1.0)

        # Base Queries (Initialized with 1D Positional Awareness)
        self.char_queries = nn.Parameter(torch.zeros(1, num_chars, d_model))
        position = torch.arange(0, num_chars, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, num_chars, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        # Seed the parameters with the spatial math, but leave them learnable
        self.char_queries.data.copy_(pe)
        
        # ðTEACHER FORCING: Embedding to convert GT class to d_model vector
        self.char_embed = nn.Embedding(num_classes, d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=1024,
            activation='gelu', dropout=0.1, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.head = CosineClassifierHead(d_model, num_classes)

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

        tgt_mask = None
        
        final_tokens = None
        loops = refine_iters + 1
        attention_maps = None 
        
        # 5. Native Decoder Loop
        for i in range(loops):
            out_tokens = self.decoder(tgt=current_queries, memory=vis_tokens, tgt_mask=tgt_mask)
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

class ViT_CrossAttn_OCR(nn.Module):
    def __init__(self, in_channels=256, d_model=256, num_chars=7, num_classes=37, num_layers=4, num_heads=16):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # The Patch Embedder (Unchanged)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, d_model // 2, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, d_model // 2), FReLU(d_model // 2),
            nn.Conv2d(d_model // 2, d_model, kernel_size=3, stride=2, padding=1), 
            nn.GroupNorm(8, d_model), FReLU(d_model),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        )
        self.input_norm = nn.LayerNorm(d_model)

        # 1. NEW: The 2D RoPE Generator
        # d_model // num_heads gives us the dimension per head (e.g., 256 / 16 = 16)
        self.rope = RotaryEmbedding2D(dim=d_model // num_heads, max_h=16, max_w=48)

        # The Masking Token (Unchanged)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=1.0)

        # 2. UPDATED: Purely Learned Character Queries (No rigid 1D Positional injection)
        self.char_queries = nn.Parameter(torch.randn(1, num_chars, d_model) * 0.02)
        
        # Teacher Forcing Embedding (Unchanged)
        self.char_embed = nn.Embedding(num_classes, d_model)
        
        # 3. NEW: Manual Cross-Attention Layers to support RoPE
        self.layers = nn.ModuleList([
            CustomRoPELayer(d_model, num_heads) for _ in range(num_layers)
        ])

        self.head = CosineClassifierHead(d_model, num_classes)

    def random_masking(self, x, mask_ratio):
        # (Unchanged from your current code)
        B, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore).unsqueeze(-1)
        x_masked = x * (1 - mask) 
        mask_token_expanded = self.mask_token.expand(B, L, D)
        return x_masked + (mask_token_expanded * mask)

    def forward(self, x, refine_iters=1, tgt=None, forcing_prob=0.0, return_attn=False):
        B = x.size(0)

        # 1. Visual Tokens
        features = self.patch_embed(x)
        vis_tokens = features.flatten(2).transpose(1, 2) # (B, 768, d_model)
        vis_tokens = self.input_norm(vis_tokens) 

        # 2. NO POSITIONAL ENCODING HERE! 
        # Apply Masking directly
        if self.training:
            vis_tokens = self.random_masking(vis_tokens, mask_ratio=0.1)

        # 3. Queries & Teacher Forcing (Unchanged)
        current_queries = self.char_queries.expand(B, -1, -1)
        if self.training and tgt is not None and forcing_prob > 0.0:
            tgt_emb = self.char_embed(tgt) 
            mask = (torch.rand(B, 7, 1, device=x.device) < forcing_prob).float()
            current_queries = (current_queries * (1.0 - mask)) + (tgt_emb * mask)
        
        # 4. Generate the RoPE Frequencies for the image tokens
        H_patches, W_patches = features.shape[2], features.shape[3]
        freqs = self.rope(H_patches, W_patches) # (H*W, head_dim)

        final_tokens = current_queries
        attention_maps = None 
        loops = refine_iters + 1
        
        # 5. Decoder Loop with RoPE
        for layer in self.layers:
            # We pass the image tokens (vis_tokens) as the Keys/Values, and the queries as Queries
            final_tokens, attn = layer(query=final_tokens, key_value=vis_tokens, freqs=freqs)
            attention_maps = attn # Store the last layer's attention for your HyperLoss!

        logits = self.head(final_tokens)
        
        if not return_attn: attention_maps = None
        return logits, final_tokens, attention_maps

# ==============================================================================
# 4. THE CUSTOM RoPE ATTENTION LAYER
# ==============================================================================
class CustomRoPELayer(nn.Module):
    """A minimal Transformer Decoder layer that applies RoPE to the Keys."""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, query, key_value, freqs):
        B, Q_len, _ = query.shape
        B, KV_len, _ = key_value.shape
        
        # 1. Pre-norm
        q = self.norm1(query)
        kv = key_value # Image tokens are already normed in the main body
        
        # 2. Linear Projections
        q = self.q_proj(q).view(B, Q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(B, KV_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, KV_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. APPLY RoPE ONLY TO THE KEYS!
        # This gives the image tokens relative spatial awareness without forcing absolute coordinates
        k = apply_rotary_emb(k, freqs)
        
        # 4. Dot-Product Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1) # (B, Heads, Q_len, KV_len) -> Perfect for HyperLoss!
        
        out = (attn @ v).transpose(1, 2).reshape(B, Q_len, -1)
        out = self.out_proj(out)
        
        # 5. Residual + FFN
        query = query + out
        query = query + self.mlp(self.norm2(query))
        
        return query, attn

# ==============================================================================
# 3. WRAPPER CLASS
# ==============================================================================

# class CustomOCR(nn.Module):
#     def __init__(self, input_shape=(128, 64, 192), num_classes=37, num_chars=7, d_model=384, num_heads=16):
#         super().__init__()
#         in_channels = input_shape[0]
        
#         # --- NEW: ADVANCED SEMANTIC STEM ---
#         self.stem = nn.Sequential(
#             # 1. Initial Projection & Noise Cleanup 
#             nn.Conv2d(in_channels, 128, 3, 1, 1),
#             nn.GroupNorm(8, 128),
#             HighContrastGate(128), 
            
#             # 2. Deformable Structural Alignment 
#             DeformableProj(128, 256, kernel_size=3, offset_groups=4),
#             nn.GroupNorm(8, 256),
#             FReLU(256), 
            
#             # 3. Final Semantic Refinement 
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), 
#             nn.GroupNorm(8, 256),
#             FReLU(256)
#         )
        
#         # Update Positional Encoding for the 64x192 resolution
#         self.pos_encoder = PositionalEncoding2D(256, 64, 192)
#         self.surgical_focus = SurgicalFocusBlock(in_channels=256, reduction=8)
        
#         # ViT Expert must now handle 768 tokens (16x48 grid)
#         self.vit_expert = ViT_CrossAttn_OCR(
#             in_channels=256, 
#             d_model=d_model, 
#             num_chars=num_chars, # EXACTLY 7 QUERIES
#             num_classes=num_classes,
#             num_heads=num_heads  
#         )
#         self.projector = nn.Sequential(nn.Linear(d_model, 128), nn.Mish(), nn.Linear(128, 128))

#         # REMOVED: self.layout_head

#     def forward(self, x, tgt=None, epoch=0, **kwargs):
#         feat = self.stem(x) 
#         feat = self.pos_encoder(feat)
#         feat = self.surgical_focus(feat)
        
#         # SCHEDULED SAMPLING
#         forcing_prob = max(0.05, 0.5 - (epoch * 0.01)) if self.training else 0.0
        
#         iters = 0
        
#         # logits: (B, 7, 37) | all_tokens: (B, 7, 384) | attn_maps: (B, H, 7, 768)
#         logits, all_tokens, attn_maps = self.vit_expert(feat, refine_iters=iters, tgt=tgt, forcing_prob=forcing_prob, return_attn=True)
        
#         # All tokens belong to characters!
#         char_tokens = all_tokens  # (B, 7, 384)
#         char_logits = logits      # (B, 7, 37)
        
#         # REMOVED: global_token and layout_logits calculation
        
#         z_vector = F.normalize(self.projector(char_tokens), p=2, dim=-1, eps=1e-6)
        
#         # --- RETURN THE CLEANED DICTIONARY ---
#         return {
#             'logits': char_logits,            # (B, 7, 37) - Ready for FocalLoss
#             'features': char_tokens,          # (B, 7, 384)
#             'z_vector': z_vector,             # (B, 7, 128)
#             'attn_maps': attn_maps,           # (B, H, 7, 768) - Sent to AdvancedAttentionLoss!
#             'theta': None,
#             'crops': None 
#         }

# ==============================================================================
# 3. WRAPPER CLASS (Slimmed & Optimized)
# ==============================================================================
class CustomOCR(nn.Module):
    # Reduced input_shape to 128, d_model to 256, and num_heads to 8
    def __init__(self, input_shape=(128, 64, 192), num_classes=37, num_chars=7, d_model=64, num_heads=8):
        super().__init__()
        in_channels = input_shape[0] # Now 128
        
        # --- SLIMMED SEMANTIC STEM ---
        self.stem = nn.Sequential(
            # 1. Initial Projection & Noise Cleanup (128 -> 128)
            nn.Conv2d(in_channels, 128, 3, 1, 1),
            nn.GroupNorm(8, 128),
            HighContrastGate(128), 
            
            # 2. Deformable Structural Alignment (128 -> 128)
            DeformableProj(128, 128, kernel_size=3, offset_groups=4),
            nn.GroupNorm(8, 128),
            FReLU(128), 
            
            # 3. Final Semantic Refinement (Expands to 256 right before the ViT)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), 
            nn.GroupNorm(8, 256),
            FReLU(256)
        )
        
        # REMOVED: self.pos_encoder (Absolute PE is dead, long live RoPE!)
        
        self.surgical_focus = SurgicalFocusBlock(in_channels=256, reduction=8)
        
        # ViT Expert (Now strictly 256 dims and 8 heads)
        self.vit_expert = ViT_CrossAttn_OCR(
            in_channels=256, 
            d_model=d_model, 
            num_chars=num_chars,
            num_layers=1,
            num_classes=num_classes,
            num_heads=num_heads,
            
        )
        
        self.projector = nn.Sequential(nn.Linear(d_model, 128), nn.Mish(), nn.Linear(128, 128))

    def forward(self, x, tgt=None, epoch=0, **kwargs):
        feat = self.stem(x) 
        # REMOVED: feat = self.pos_encoder(feat)
        feat = self.surgical_focus(feat)
        
        forcing_prob = max(0.05, 0.5 - (epoch * 0.01)) if self.training else 0.0
        iters = 0
        
        # logits: (B, 7, 37) | all_tokens: (B, 7, 256) | attn_maps: (B, 8, 7, 768)
        logits, all_tokens, attn_maps = self.vit_expert(feat, refine_iters=iters, tgt=tgt, forcing_prob=forcing_prob, return_attn=True)
        
        char_tokens = all_tokens  
        char_logits = logits      
        
        z_vector = F.normalize(self.projector(char_tokens), p=2, dim=-1, eps=1e-6)
        
        return {
            'logits': char_logits,            
            'features': char_tokens,          
            'z_vector': z_vector,             
            'attn_maps': attn_maps,           
            'theta': None,
            'crops': None 
        }

