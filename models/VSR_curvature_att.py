import torch
import torch.nn as nn
import torch.nn.functional as F
from models import register
from . import CustomOCR 
import torch.nn.utils as nn_utils
import numpy as np
import torchvision.ops as ops
from einops import rearrange

# ==============================================================================
# 1. BACKBONE COMPONENTS (Standard)
# ==============================================================================
class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        
        self.qkv_dwconv_h = nn.Conv2d(
            channels * 3, channels * 3, kernel_size=(1, 3), padding=(0, 1), 
            groups=channels * 3, bias=False
        )
        self.qkv_dwconv_v = nn.Conv2d(
            channels * 3, channels * 3, kernel_size=(3, 1), padding=(1, 0), 
            groups=channels * 3, bias=False
        )
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
    def __init__(self, channels, expansion_factor=2.0):
        super(GDFN, self).__init__()
        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(
            hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1, 
            groups=hidden_channels * 2, bias=False
        )
        self.act = FReLU(hidden_channels) 
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = x1 * self.act(x2)
        return self.project_out(x)
    
class RestormerBlock(nn.Module):
    def __init__(self, channels, num_heads=8): 
        super(RestormerBlock, self).__init__()
        self.norm1 = nn.GroupNorm(8, channels) 
        self.attn = MDTA(channels, num_heads) 
        self.norm2 = nn.GroupNorm(8, channels)
        self.ffn = GDFN(channels)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

import math

class LatentUpsampler(nn.Module):
    def __init__(self, dim, upscale_factor=2, num_levels=4, features_per_level=8):
        super().__init__()
        self.scale = upscale_factor
        self.grid_levels = nn.ParameterList()
        
        # --- THE FIX: Starting at your requested 24x72 ---
        base_res = 24  
        max_res = 64   
        growth_factor = math.exp((math.log(max_res) - math.log(base_res)) / (num_levels - 1))
        
        for i in range(num_levels):
            res_h = int(base_res * (growth_factor ** i))
            res_w = res_h * 3 
            
            # INITIALIZATION: Increased to 0.05 to give the MLP a stronger starting signal
            grid = nn.Parameter(torch.randn(1, features_per_level, res_h, res_w) * 0.05)
            self.grid_levels.append(grid)

        # Learnable gain for the grids - starts at 0.1 so they aren't ignored
        self.grid_gain = nn.Parameter(torch.tensor(0.1))

        mlp_in_dim = dim + (num_levels * features_per_level)
        self.mlp = nn.Sequential(
            nn.Conv2d(mlp_in_dim, dim, kernel_size=1),
            nn.GroupNorm(8, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1)
        )
        
        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, dim),
            FReLU(dim)
        )

    def forward(self, x, return_stages=True):
        B, C, H, W = x.shape
        target_H, target_W = H * self.scale, W * self.scale
        
        with torch.amp.autocast('cuda', enabled=False):
            x_fp32 = x.float()
            # FIX 1 (Already in your code): Explicit size alignment
            x_nearest = F.interpolate(x_fp32, size=(target_H, target_W), mode='nearest')
            
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1, 1, target_H, device=x.device, dtype=torch.float32),
                torch.linspace(-1, 1, target_W, device=x.device, dtype=torch.float32),
                indexing='ij'
            )
            rel_coords = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
            
            sampled_features = []
            for grid in self.grid_levels:
                batched_grid = grid.expand(B, -1, -1, -1).float()
                # FIX 2 (Already in your code): align_corners=True
                sampled = F.grid_sample(batched_grid, rel_coords, mode='bilinear', align_corners=True)
                # Apply the gain to ensure the signal reaches the MLP
                sampled_features.append(sampled * self.grid_gain)
            
            encoded_coords = torch.cat(sampled_features, dim=1)
            mlp_input = torch.cat([x_nearest, encoded_coords], dim=1)
            
        mlp_input = mlp_input.to(x.dtype)
        hr_features = self.mlp(mlp_input)
        out = self.refine(hr_features)

        if return_stages:
            return out, sampled_features
        return out

# class LatentUpsampler(nn.Module):
#     def __init__(self, dim, upscale_factor=2):
#         super().__init__()
#         # 1. Project to higher channel depth
#         self.conv1 = nn.Conv2d(
#             dim, dim * (upscale_factor ** 2), kernel_size=3, padding=1
#         )
#         # 2. Shuffle channels into spatial resolution
#         self.upsample = nn.PixelShuffle(upscale_factor)
        
#         # --- THE FIX: Smoothing Convolution ---
#         # 3. Blend the shuffled pixels together to destroy the checkerboard artifact
#         self.conv2 = nn.Conv2d(
#             dim, dim, kernel_size=3, padding=1
#         )
        
#         self.norm = nn.GroupNorm(8, dim)
#         self.act = nn.Mish()

#     def forward(self, x):
#         x = self.upsample(self.conv1(x))
#         x = self.conv2(x) # Apply smoothing
#         return self.act(self.norm(x))

class HighContrastGate(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.tau_raw = nn.Parameter(torch.full((1, num_channels, 1, 1), 0.1))
        self.beta_raw = nn.Parameter(torch.full((1, num_channels, 1, 1), 10.0))

    def forward(self, x):
        tau = torch.abs(self.tau_raw) + 1e-4 
        beta = torch.abs(self.beta_raw)
        magnitude = torch.abs(x)
        gate = torch.sigmoid(beta * (magnitude - tau))
        return x * gate

class DeformableProj(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, offset_groups=4):
        super().__init__()
        self.padding = kernel_size // 2
        
        # 1. The Offset Calculator gets 'replicate' to protect it from the black edge cliff!
        self.offset_conv = nn.Conv2d(
            in_channels, 2 * kernel_size * kernel_size * offset_groups, 
            kernel_size=kernel_size, padding=self.padding, padding_mode='replicate'
        )
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        
        # 2. The native C++ operator gets standard zero padding (it will use the safe offsets)
        self.deform_conv = ops.DeformConv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=self.padding
        )

    def forward(self, x):
        offsets = self.offset_conv(x)
        return self.deform_conv(x, offsets)

class FReLU(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.spatial_condition = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1, 
            groups=in_channels
        )
        self.norm = nn.GroupNorm(8, in_channels)

    def forward(self, x):
        spatial_context = self.norm(self.spatial_condition(x))
        return torch.max(x, spatial_context)

# ==============================================================================
# 2. SINGLE-IMAGE SPATIAL FEATURE EXTRACTOR
# ==============================================================================
class SpatialFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, feature_dim=128):
        super().__init__()
        
        # 1. STEM (3 channels -> feature_dim)
        # Replaces the temporal fusion with a robust spatial stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, 3, 1, 1),
            nn.GroupNorm(8, feature_dim),
            HighContrastGate(feature_dim)
        )
        
        # 2. COORDINATE INJECTION PROJECTION
        # Expects feature_dim (from stem) + 2 (X, Y grids)
        self.coord_proj = nn.Sequential(
            nn.Conv2d(feature_dim + 2, feature_dim, 3, 1, 1),
            nn.GroupNorm(8, feature_dim),
            FReLU(feature_dim)
        )
        
        # 3. DEEP RESTORATION (Optimized Restormer)
        self.body = nn.Sequential(*[RestormerBlock(feature_dim, num_heads=8) for _ in range(2)])
        self.conv_after_body = nn.Conv2d(feature_dim, feature_dim, 3, 1, 1)
        
        # 4. Feature Hallucination Layer (Super-Resolution)
        self.latent_sr = LatentUpsampler(feature_dim, upscale_factor=2)
        
        # 5. FINAL REFINE (FReLU)
        self.refine_conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 1, 1, 0), 
            nn.GroupNorm(8, feature_dim), 
            FReLU(feature_dim)
        )

    def forward(self, x, x_grid, y_grid, return_stages=False): # Add return_stages flag
        feat_stem = self.stem(x) 
        
        feat_with_coords = torch.cat([feat_stem, x_grid, y_grid], dim=1)
        feat_shallow = self.coord_proj(feat_with_coords)
        
        feat_deep = self.body(feat_shallow)
        texture = self.conv_after_body(feat_deep) + feat_shallow 

        # --- THE MODIFICATION ---
        # Capture the tuple if return_stages is True
        if return_stages:
            feat_sr, stages = self.latent_sr(texture, return_stages=True)
            out = self.refine_conv(feat_sr)
            return out, stages # Return both
        else:
            feat_sr = self.latent_sr(texture, return_stages=False)
            return self.refine_conv(feat_sr)

# ==============================================================================
# 3. THE CGNET WRAPPER
# ==============================================================================
class Cgnet(nn.Module):
    def __init__(self, in_channels=3, mode='hybrid', feature_dim=128): # Removed in_images
        super(Cgnet, self).__init__()
        self.mode = mode.lower()
        self.feature_dim = feature_dim
        
        # Instantiating the new Single-Image Extractor
        self.student_extractor = SpatialFeatureExtractor(in_channels, feature_dim)
        
        if self.mode in ['ocr', 'hybrid']:
            # CustomOCR remains the same, expecting (B, feature_dim, H, W)
            self.student_ocr = CustomOCR.CustomOCR(
                input_shape=(feature_dim, 64, 192),
                num_classes=37, 
                num_chars=7, 
                d_model=384,     
                num_heads=16     
            )

    def forward(self, x, temporal_pool=False, return_latent=True, **kwargs): 
        # --- A. Setup Input Dimensions ---
        # We enforce strict 4D tensors (B, C, H, W) since this is a single-image model now
        if x.dim() == 5:
            # If the dataloader accidentally sends a 5D tensor of length 1, squeeze it
            x = x.squeeze(1) 
            
        b, c, h, w = x.shape 

        # --- B. Grid Generation ---
        y_grid = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).expand(b, 1, h, w)
        x_grid = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).expand(b, 1, h, w)

        # --- C. Feature Extraction ---
        latent_lr, stages = self.student_extractor(x, x_grid, y_grid, return_stages=True)
        # --- D. Cross-Attention OCR ---
        # Note: We pass epoch down if it is in kwargs to enable Scheduled Sampling
        epoch = kwargs.get('epoch', 0)
        preds_lr = self.student_ocr(latent_lr, epoch=epoch)
        
        preds_lr['loss_vq'] = torch.tensor(0.0, device=x.device)
        preds_lr['kld_raw'] = torch.tensor(0.0, device=x.device)

        if return_latent:
            preds_lr['latent_lr'] = latent_lr
            # Pack the stages into the dictionary for the dashboard
            preds_lr['latent_stages'] = stages
        return preds_lr

class SR_LPR_NET(nn.Module):
    def __init__(self, mode='hybrid', feature_dim=128, **kwargs): 
        super().__init__()
        self.cgnet = Cgnet(in_channels=3, mode=mode, feature_dim=feature_dim)

    def forward(self, x, **kwargs):
        return self.cgnet(x, **kwargs)

@register('VSR_CURVATURE') 
def make(**kwargs): return SR_LPR_NET(**kwargs)