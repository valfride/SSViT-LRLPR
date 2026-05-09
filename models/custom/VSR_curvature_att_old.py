import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from einops import rearrange
from models import register
from timm.models.layers import DropPath

# ==============================================================================
# 1. CORE UTILITY BLOCKS
# ==============================================================================
class FReLU(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.spatial_condition = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1, 
            groups=in_channels,
            padding_mode='replicate' # <--- ADDED
        )
        self.norm = nn.GroupNorm(8, in_channels)

    def forward(self, x):
        spatial_context = self.norm(self.spatial_condition(x))
        return torch.max(x, spatial_context)

class HighContrastGate(nn.Module):
    def __init__(self, num_channels, reduction=16):
        super().__init__()
        # Tiny bottleneck MLP to predict tau and beta on the fly
        mip = max(8, num_channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.meta_network = nn.Sequential(
            nn.Conv2d(num_channels, mip, kernel_size=1),
            nn.GroupNorm(4, mip), # Optional, stabilizes the prediction
            nn.ReLU(inplace=True),
            # Outputs exactly 2 values per channel (one for tau, one for beta)
            nn.Conv2d(mip, num_channels * 2, kernel_size=1) 
        )
        
        # We initialize the final layer to zero so it starts with neutral behavior
        nn.init.constant_(self.meta_network[3].weight, 0)
        nn.init.constant_(self.meta_network[3].bias, 0)

        # Base fallback values (so it doesn't crash on epoch 1)
        self.base_tau = 0.1
        self.base_beta = 10.0

    def forward(self, x):
        # 1. 'Look' at the current image and predict modifiers
        # stats shape: (B, 2*num_channels, 1, 1)
        stats = self.meta_network(self.pool(x))
        
        # 2. Split the predictions into tau and beta modifiers
        tau_mod, beta_mod = stats.chunk(2, dim=1)
        
        # 3. Apply the dynamic modifiers safely
        # Softplus ensures tau and beta strictly remain positive and smooth
        tau = nn.functional.softplus(tau_mod + self.base_tau) + 1e-4
        beta = nn.functional.softplus(beta_mod + self.base_beta)
        
        # 4. Standard Gating Logic
        magnitude = torch.abs(x)
        gate = torch.sigmoid(beta * (magnitude - tau))
        return x * gate

class DeformableProj(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, offset_groups=2):
        super().__init__()
        self.padding = kernel_size // 2
        
        self.offset_conv = nn.Conv2d(
            in_channels, 2 * kernel_size * kernel_size * offset_groups, 
            kernel_size=kernel_size, padding=self.padding, padding_mode='replicate'
        )
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        
        self.deform_conv = ops.DeformConv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=self.padding
        )

    def forward(self, x):
        offsets = self.offset_conv(x)
        return self.deform_conv(x, offsets)

class SurgicalFocusBlock(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        raw_mip = max(8, in_channels // reduction)
        mip = math.ceil(raw_mip / 4) * 4 
        
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

# ==============================================================================
# 2. RESTORMER BACKBONE
# ==============================================================================
class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_dwconv_h = nn.Conv2d(channels * 3, channels * 3, kernel_size=(1, 3), padding=(0, 1), groups=channels * 3, bias=False)
        self.qkv_dwconv_v = nn.Conv2d(channels * 3, channels * 3, kernel_size=(3, 1), padding=(1, 0), groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        qkv = self.qkv_dwconv_h(qkv) + self.qkv_dwconv_v(qkv)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.project_out(out)

class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor=1.2):
        super(GDFN, self).__init__()
        raw_hidden = channels * expansion_factor
        hidden_channels = int(round(raw_hidden / 8) * 8)
        hidden_channels = max(8, hidden_channels)
        
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1, groups=hidden_channels * 2, bias=False)
        self.act = FReLU(hidden_channels) 
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = x1 * self.act(x2)
        return self.project_out(x)
    
class RestormerBlock(nn.Module):
    def __init__(self, channels, num_heads=8, drop_path=0.0): # <--- 1. Add the argument
        super(RestormerBlock, self).__init__()
        self.norm1 = nn.GroupNorm(8, channels) 
        self.attn = MDTA(channels, num_heads) 
        self.norm2 = nn.GroupNorm(8, channels)
        self.ffn = GDFN(channels, expansion_factor=1.2)
        
        # <--- 2. Initialize the DropPath layer (defaults to Identity if 0.0)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        # <--- 3. Wrap the block outputs in the drop_path function
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x

class LatentUpsampler(nn.Module):
    def __init__(self, dim, upscale_factor=2):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.upsample = nn.PixelShuffle(upscale_factor)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, dim)
        self.act = FReLU(dim)

    def forward(self, x):
        x = self.upsample(self.conv1(x))
        x = self.conv2(x) 
        return self.act(self.norm(x))

class SpatialFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, feature_dim=128, cnn_heads=4):
        super().__init__()
        self.stem = nn.Sequential(
            # 1. The Front Door: Safely extend the raw image border!
            nn.Conv2d(
                in_channels, feature_dim, 3, stride=1, padding=1, 
                padding_mode='replicate'  # <--- ADDED
            ),
            nn.GroupNorm(8, feature_dim),
            HighContrastGate(feature_dim),
            
            # 2. The Squeeze Layer: Safely compress without edge artifacts
            nn.Conv2d(
                feature_dim, feature_dim, 3, stride=2, padding=1, 
                padding_mode='replicate'  # <--- ADDED
            ),
            nn.GroupNorm(8, feature_dim),
            FReLU(feature_dim)
        )
        
        # Operates purely at 16x48 (Ultra-fast, deep semantics)
        self.body = nn.Sequential(*[RestormerBlock(feature_dim, num_heads=cnn_heads, drop_path=0.2) for _ in range(2)])
        self.conv_after_body = nn.Conv2d(feature_dim, feature_dim, 3, 1, 1)
        
        # Restores the feature map back to 32x96
        self.latent_sr = LatentUpsampler(feature_dim, upscale_factor=2)
        
        self.refine_conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 1, 1, 0), 
            nn.GroupNorm(8, feature_dim), 
            FReLU(feature_dim)
        )

    def forward(self, x):
        feat_shallow = self.stem(x) 
        feat_deep = self.body(feat_shallow)
        texture = self.conv_after_body(feat_deep) + feat_shallow 
        feat_sr = self.latent_sr(texture)
        
        # ---> THE UPGRADE: Local Skip Connection
        return self.refine_conv(feat_sr) + feat_sr
# ==============================================================================
# 3. 2D RoPE ViT DECODER
# ==============================================================================
class RotaryEmbedding2D(nn.Module):
    def __init__(self, dim, max_h=32, max_w=96): 
        super().__init__()
        half_dim = dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, 2).float() / half_dim))
        self.register_buffer("inv_freq", inv_freq)
        
        seq_y = torch.arange(max_h).float()
        seq_x = torch.arange(max_w).float()
        
        freqs_y = torch.einsum("i,j->ij", seq_y, self.inv_freq)
        freqs_x = torch.einsum("i,j->ij", seq_x, self.inv_freq)
        
        freqs_y = torch.cat((freqs_y, freqs_y), dim=-1)
        freqs_x = torch.cat((freqs_x, freqs_x), dim=-1)
        
        freqs_y = freqs_y.unsqueeze(1).expand(-1, max_w, -1) 
        freqs_x = freqs_x.unsqueeze(0).expand(max_h, -1, -1) 
        
        self.register_buffer("freqs_y", freqs_y)
        self.register_buffer("freqs_x", freqs_x)

    def forward(self, h, w):
        fy = self.freqs_y[:h, :w, :]
        fx = self.freqs_x[:h, :w, :]
        freqs = torch.cat((fy, fx), dim=-1) 
        return freqs.reshape(h * w, -1)     

def apply_rotary_emb(x, freqs):
    x1, x2 = x[..., ::2], x[..., 1::2]
    x_rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
    return (x * freqs.cos()) + (x_rotated * freqs.sin())

class CosineClassifierHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.tau = nn.Parameter(torch.tensor(20.0))

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=-1, eps=1e-6)
        w_norm = F.normalize(self.weight, p=2, dim=-1, eps=1e-6)
        logits = F.linear(x_norm, w_norm) * self.tau
        return logits

class CustomRoPELayer(nn.Module):
    # ---> ADDED: dropout and drop_path arguments
    def __init__(self, d_model, num_heads, dropout=0.1, drop_path=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        # ---> ADDED: MultiheadAttention native dropout
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm_self = nn.LayerNorm(d_model)
        
        # Existing Cross-Attention Projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # ---> ADDED: Explicit regularizers
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # ---> UPGRADE: Dropout injected into the FFN Bottleneck
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout), # Prevents co-adaptation of the expanded neurons
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, query, key_value, freqs):
        B, Q_len, _ = query.shape
        B, KV_len, _ = key_value.shape
        
        # ==========================================
        # STEP 1: AUTOCORRECTION (Self-Attention)
        # ==========================================
        q_norm = self.norm_self(query)
        q_self, _ = self.self_attn(q_norm, q_norm, q_norm)
        
        # ---> UPGRADE: Wrap the self-attention residual in DropPath and Dropout
        query = query + self.drop_path(self.proj_drop(q_self))
        
        # ==========================================
        # STEP 2: VISION (Cross-Attention with RoPE)
        # ==========================================
        q = self.norm1(query)
        kv = key_value 
        
        q = self.q_proj(q).view(B, Q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(B, KV_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, KV_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        k = apply_rotary_emb(k, freqs)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1) 
        
        # ---> UPGRADE: Softmax Dropout
        attn = self.attn_drop(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, Q_len, -1)
        out = self.out_proj(out)
        
        # ---> UPGRADE: Wrap Cross-Attention residual
        query = query + self.drop_path(self.proj_drop(out))
        
        # ---> UPGRADE: Wrap MLP residual
        query = query + self.drop_path(self.mlp(self.norm2(query)))
        
        return query

class ViT_CrossAttn_OCR(nn.Module):
    def __init__(self, in_channels=256, d_model=256, num_chars=7, num_classes=37, num_layers=3, num_heads=8, dropout=0.1, drop_path_rate=0.2):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # ---> UPGRADE: Unrestricted feature mapping (No d_model // 2 compression)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, d_model), FReLU(d_model),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1), 
            nn.GroupNorm(8, d_model), FReLU(d_model),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=1) 
        )

        # self.patch_embed = nn.Sequential(
        #     nn.Conv2d(in_channels, d_model // 2, kernel_size=3, stride=1, padding=1),
        #     nn.GroupNorm(8, d_model // 2), FReLU(d_model // 2),
            
        #     # 32x96 -> 16x48
        #     nn.Conv2d(d_model // 2, d_model, kernel_size=3, stride=2, padding=1), 
        #     nn.GroupNorm(8, d_model), FReLU(d_model),
            
        #     # ---> THE FIX: Changed to stride=1 to preserve the 16x48 grid!
        #     nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=1) 
        # )

        self.input_norm = nn.LayerNorm(d_model)

        self.rope = RotaryEmbedding2D(dim=d_model // num_heads, max_h=32, max_w=96)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=1.0)

        self.char_queries = nn.Parameter(torch.randn(1, num_chars, d_model) * 0.02)
        self.char_embed = nn.Embedding(num_classes, d_model)
        
        # ---> UPGRADE: Linear DropPath Schedule
        # Scales smoothly from 0.0 up to your target drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        
        self.layers = nn.ModuleList([
            CustomRoPELayer(
                d_model=d_model, 
                num_heads=num_heads, 
                dropout=dropout,
                drop_path=dpr[i] # Pass the specific scheduled rate to each layer
            ) 
            for i in range(num_layers)
        ])
        
        self.head = CosineClassifierHead(d_model, num_classes)

    def random_masking(self, x, mask_ratio):
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

    def forward(self, x, tgt=None, forcing_prob=0.0):
        B = x.size(0)

        features = self.patch_embed(x)
        vis_tokens = features.flatten(2).transpose(1, 2) 
        vis_tokens = self.input_norm(vis_tokens) 

        if self.training:
            vis_tokens = self.random_masking(vis_tokens, mask_ratio=0.1)

        current_queries = self.char_queries.expand(B, -1, -1)
        if self.training and tgt is not None and forcing_prob > 0.0:
            tgt_emb = self.char_embed(tgt) 
            mask = (torch.rand(B, 7, 1, device=x.device) < forcing_prob).float()
            current_queries = (current_queries * (1.0 - mask)) + (tgt_emb * mask)
        
        H_patches, W_patches = features.shape[2], features.shape[3]
        freqs = self.rope(H_patches, W_patches) 

        final_tokens = current_queries
        
        for layer in self.layers:
            final_tokens = layer(query=final_tokens, key_value=vis_tokens, freqs=freqs)

        logits = self.head(final_tokens)
        return logits, final_tokens

# ==============================================================================
# 4. WRAPPER CLASSES
# ==============================================================================
class CustomOCR(nn.Module):
    def __init__(self, input_shape=(128, 32, 96), num_classes=37, num_chars=7, d_model=256, num_heads=8):
        super().__init__()
        in_channels = input_shape[0] 
        
        # ---> DYNAMIC SCALING: Smooth hourglass bottleneck
        mid_channels = in_channels // 2 
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1),
            nn.GroupNorm(8, mid_channels),
            HighContrastGate(mid_channels), 
            # Scale the Deformable groups safely to match the width
            DeformableProj(mid_channels, mid_channels, kernel_size=3, offset_groups=max(4, mid_channels // 16)),
            nn.GroupNorm(8, mid_channels),
            FReLU(mid_channels), 
            nn.Conv2d(mid_channels, in_channels, kernel_size=3, stride=1, padding=1), 
            nn.GroupNorm(8, in_channels),
            FReLU(in_channels)
        )
        
        self.surgical_focus = SurgicalFocusBlock(in_channels=in_channels, reduction=8)
        
        self.vit_expert = ViT_CrossAttn_OCR(
            in_channels=in_channels, d_model=d_model, num_chars=num_chars,
            num_layers=3, num_classes=num_classes, num_heads=num_heads,
            dropout=0.1,           
            drop_path_rate=0.2     
        )

    def forward(self, x, tgt=None, epoch=0, **kwargs):
        feat = self.stem(x) 
        feat = self.surgical_focus(feat)
        
        forcing_prob = max(0.05, 0.5 - (epoch * 0.01)) if self.training else 0.0
        
        logits, char_tokens = self.vit_expert(feat, tgt=tgt, forcing_prob=forcing_prob)
        
        return {
            'logits': logits,            
            'features': char_tokens,                    
        }

class Cgnet(nn.Module):
    def __init__(self, in_channels=3, mode='hybrid', feature_dim=128, cnn_heads=4, d_model=256, vit_heads=8, **kwargs): 
        super(Cgnet, self).__init__()
        self.mode = mode.lower()
        self.feature_dim = feature_dim
        
        self.student_extractor = SpatialFeatureExtractor(in_channels, feature_dim, cnn_heads=cnn_heads)
        
        if self.mode in ['ocr', 'hybrid']:
            self.student_ocr = CustomOCR(
                input_shape=(feature_dim, 64, 192),
                num_classes=38, # 36 chars + EOS + PAD
                num_chars=7, 
                d_model=d_model,     
                num_heads=vit_heads  
            )

    def forward(self, x, temporal_pool=False, return_latent=True, **kwargs): 
        if x.dim() == 5:
            x = x.squeeze(1) 
            
        latent_lr = self.student_extractor(x)
        epoch = kwargs.get('epoch', 0)
        tgt = kwargs.get('tgt', None) # Pass targets down for scheduled sampling
        
        preds_lr = self.student_ocr(latent_lr, epoch=epoch, tgt=tgt)
        
        if return_latent:
            preds_lr['latent_lr'] = latent_lr
            
        return preds_lr

class SR_LPR_NET(nn.Module):
    def __init__(self, in_channels=3, mode='hybrid', feature_dim=128, **kwargs): 
        super().__init__()
        self.cgnet = Cgnet(in_channels=in_channels, mode=mode, feature_dim=feature_dim, **kwargs)

    def forward(self, x, **kwargs):
        return self.cgnet(x, **kwargs)

@register('VSR_CURVATURE') 
def make(**kwargs): return SR_LPR_NET(**kwargs)