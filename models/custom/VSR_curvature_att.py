import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from models import register
from . import CustomOCR 

# ==============================================================================
# 1. IMPORT THE INDUSTRY STANDARD: SVTRv2
# ==============================================================================
from models.svtrv2.svtrv2 import Block

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# ==============================================================================
# 2. VSR & SUPER-RESOLUTION UTILITIES
# ==============================================================================
class FReLU(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.spatial_condition = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.norm = nn.GroupNorm(8, in_channels)

    def forward(self, x):
        spatial_context = self.norm(self.spatial_condition(x))
        return torch.max(x, spatial_context)

class DeformableProj(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, offset_groups=2): 
        super().__init__()
        self.padding = kernel_size // 2
        self.offset_conv = nn.Conv2d(
            in_channels, 2 * kernel_size * kernel_size * offset_groups, 
            kernel_size=kernel_size, stride=stride, padding=self.padding, padding_mode='replicate'
        )
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        
        self.deform_conv = ops.DeformConv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=self.padding
        )

    def forward(self, x):
        offsets = self.offset_conv(x)
        return self.deform_conv(x, offsets)

class MultiScaleContext(nn.Module):
    def __init__(self, dim):
        super().__init__()
        branch_dim = dim // 4
        
        # Keep Mish here to maintain the non-linear math!
        self.branch1 = nn.Sequential(nn.Conv2d(dim, branch_dim, kernel_size=1), nn.GroupNorm(2, branch_dim), nn.Mish())
        self.branch2 = nn.Sequential(nn.Conv2d(dim, branch_dim, kernel_size=3, padding=2, dilation=2), nn.GroupNorm(2, branch_dim), nn.Mish())
        self.branch3 = nn.Sequential(nn.Conv2d(dim, branch_dim, kernel_size=3, padding=4, dilation=4), nn.GroupNorm(2, branch_dim), nn.Mish())
        self.branch4 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, branch_dim, kernel_size=1), nn.Mish())
        
        # The Fusion Layer
        self.fuse = nn.Sequential(nn.Conv2d(branch_dim * 4, dim, kernel_size=1), nn.GroupNorm(8, dim))
        
        # ---> THE UPGRADE: The Gate becomes a Spatial Attention module!
        self.contrast_attention = HighContrastGate(num_channels=dim, window_size=5)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x).expand_as(b1) # Expands the 1x1 back to HxW
        
        # 1. Fuse the multi-scale features
        fused_features = self.fuse(torch.cat([b1, b2, b3, b4], dim=1))
        
        # 2. Apply the Windowed Contrast Gate as Spatial Attention!
        attended_features = self.contrast_attention(fused_features)
        
        return attended_features

class LatentUpsampler(nn.Module):
    def __init__(self, dim, upscale_factor=2):
        super().__init__()
        self.multi_scale_analyzer = MultiScaleContext(dim)
        self.up_proj = nn.Conv2d(dim, dim * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.upsample = nn.PixelShuffle(upscale_factor)
        
        # ---> THE FIX: Collapse the 64 channels down to 3 RGB channels
        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1), 
            nn.GroupNorm(8, dim // 2), 
            nn.Mish(),
            nn.Conv2d(dim // 2, 3, kernel_size=3, padding=1),
            nn.Tanh() # Bounds the output to [-1, 1] RGB space
        )

    def forward(self, x):
        enriched_x = x + self.multi_scale_analyzer(x)
        hr_grid = self.upsample(self.up_proj(enriched_x))
        return self.refine(hr_grid)
    
class HighContrastGate(nn.Module):
    def __init__(self, num_channels, window_size=5):
        super().__init__()
        self.window_size = window_size
        self.padding = window_size // 2
        
        # tau is now evaluating "Standard Deviations" instead of raw pixels.
        # A tau of 1.0 means: "Only pass pixels that are 1 Std Dev away from their neighbors"
        self.tau_raw = nn.Parameter(torch.full((1, num_channels, 1, 1), 1.0)) 
        self.beta_raw = nn.Parameter(torch.full((1, num_channels, 1, 1), 10.0))

    def forward(self, x):
        tau = torch.abs(self.tau_raw) + 1e-4 
        beta = torch.abs(self.beta_raw)
        
        # ==========================================
        # 1. WINDOWED MEAN (The Baseline)
        # ==========================================
        local_mean = F.avg_pool2d(x, kernel_size=self.window_size, stride=1, padding=self.padding)
        
        # ==========================================
        # 2. WINDOWED VARIANCE (The Local Contrast)
        # ==========================================
        # Var(X) = E[X^2] - E[X]^2
        local_sq_mean = F.avg_pool2d(x**2, kernel_size=self.window_size, stride=1, padding=self.padding)
        local_var = torch.relu(local_sq_mean - local_mean**2)
        local_std = torch.sqrt(local_var + 1e-5) # Add epsilon to prevent division by zero
        
        # ==========================================
        # 3. RELATIVE CONTRAST SCORE
        # ==========================================
        # How significantly does the central pixel pop out from its specific window?
        # If std is small (flat area), even a tiny difference creates a HUGE contrast score!
        local_contrast = torch.abs(x - local_mean) / local_std
        
        # ==========================================
        # 4. THE GATE
        # ==========================================
        gate = torch.sigmoid(beta * (local_contrast - tau))
        
        # Multiply the most relevant pixels, diminish the irrelevant!
        return x * gate

# ==============================================================================
# 3. THE COGNITIVE SVTR BACKBONE (The Hallucination Engine)
# ==============================================================================
# ==============================================================================
# 3. THE COGNITIVE SVTR BACKBONE (The Hallucination Engine)
# ==============================================================================
class CognitiveSVTRBackbone(nn.Module):
    def __init__(self, in_channels=3, feature_dim=128, depths=[2, 4], cnn_heads=4):
        super().__init__()
        
        # 1. Convolutional Stem (Extract raw strokes & edges quickly)
        self.patch_embed = nn.Sequential(
            # ---> THE FIX: Change stride=2 to stride=1 
            nn.Conv2d(in_channels, feature_dim // 2, kernel_size=3, stride=1, padding=1),
            HighContrastGate(feature_dim // 2),
            nn.Conv2d(feature_dim // 2, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, feature_dim)
        )
        
        # 2. SVTR Cognitive Stages
        total_depth = sum(depths)
        
        # ---> THE FIX: Dynamically handle any number of stages
        mixers = []
        for i, d in enumerate(depths):
            # Stage 0 is always Local. All deeper stages are Global.
            mixer_type = 'Local' if i == 0 else 'Global'
            mixers.extend([mixer_type] * d)
        
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        
        self.svtr_blocks = nn.ModuleList([
            Block(
                dim=feature_dim,
                num_heads=cnn_heads,
                mixer=mixers[i], 
                mlp_ratio=4.0,
                qkv_bias=True,
                drop=0.0,
                act_layer=Swish,
                attn_drop=0.0,
                drop_path=dpr[i],
                norm_layer=nn.LayerNorm,
                eps=1e-05
            ) for i in range(total_depth)
        ])
        self.norm = nn.LayerNorm(feature_dim)
        
        # 3. Super-Resolution Latent Upsampler
        # ---> THE FIX 2: Change upscale_factor from 2 to 4 to restore the 96x192 resolution!
        self.latent_sr = LatentUpsampler(feature_dim, upscale_factor=2)
        
        # 4. Final HR Polish 
        self.hr_refine = nn.Sequential(
            DeformableProj(feature_dim, feature_dim, kernel_size=3, offset_groups=4),
            nn.GroupNorm(8, feature_dim),
            HighContrastGate(feature_dim)
        )

    def forward(self, x): 
        B, C_in, H, W = x.shape
        
        feat_stem = self.patch_embed(x) 
        feat_seq = feat_stem.flatten(2).transpose(1, 2).contiguous() 
        
        for blk in self.svtr_blocks:
            feat_seq = blk(feat_seq)
        feat_seq = self.norm(feat_seq)
        
        # ---> THE FIX: Remove the "H // 2, W // 2" division!
        # Because stride is 1, the spatial dims are exactly H and W.
        feat_2d = feat_seq.transpose(1, 2).contiguous().view(B, -1, H, W)
        
        # 5. GLOBAL SKIP CONNECTION 
        feat = feat_2d + feat_stem
            
        # 6. Upscale and Refine (Generates the 3-channel RGB image)
        sr_image = self.latent_sr(feat)
        
        # ---> THE FIX: Return BOTH the deep features and the physical image
        return feat, sr_image
    
# ==============================================================================
# 4. THE CGNET WRAPPER & OCR INTEGRATION
# ==============================================================================
class Cgnet(nn.Module):
    def __init__(self, in_channels=3, mode='hybrid', feature_dim=128, cnn_heads=4, **kwargs): 
        super(Cgnet, self).__init__()
        self.mode = mode.lower()
        self.feature_dim = feature_dim
        
        self.student_extractor = CognitiveSVTRBackbone(
            in_channels, feature_dim, depths=[2, 4], cnn_heads=cnn_heads
        )
        
        if self.mode in ['ocr', 'hybrid']:
            # ---> THE FIX: Dynamically link the backbone's dimension to the OCR!
            self.student_ocr = CustomOCR.CustomOCR(
                input_shape=(self.feature_dim, 32, 96), num_classes=38, num_chars=7,  
            )

    def forward(self, x, temporal_pool=False, return_latent=True, **kwargs): 
        if x.dim() == 5: x = x.squeeze(1) 
            
        # Unpack the new dual outputs
        deep_features, sr_image = self.student_extractor(x)

        epoch = kwargs.get('epoch', 0)
        
        preds_lr = self.student_ocr(deep_features, epoch=epoch)
        
        # ---> NEW: Expose the generated physical image for the training loop
        preds_lr['sr_image'] = sr_image
        
        if return_latent:
            # We will grab the true latent features from inside the OCR head in the next step
            preds_lr['latent_lr'] = preds_lr.get('ocr_features', sr_image) 
            
        return preds_lr

class SR_LPR_NET(nn.Module):
    def __init__(self, in_channels=3, mode='hybrid', feature_dim=128, **kwargs): 
        super().__init__()
        self.cgnet = Cgnet(in_channels=in_channels, mode=mode, feature_dim=feature_dim, **kwargs)

    def forward(self, x, **kwargs):
        return self.cgnet(x, **kwargs)

@register('VSR_CURVATURE') 
def make(**kwargs): return SR_LPR_NET(**kwargs)