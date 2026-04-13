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
        self.contrast_attention = CustomOCR.SurgicalFocusBlock(in_channels=dim, reduction=16)

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

# ==============================================================================
# 3. THE COGNITIVE SVTR BACKBONE (The Hallucination Engine)
# ==============================================================================
class CognitiveSVTRBackbone(nn.Module):
    def __init__(self, in_channels=3, feature_dim=128, depths=[2, 4], cnn_heads=4):
        super().__init__()
        
        # 1. Convolutional Stem (The Squeeze!)
        self.patch_embed = nn.Sequential(
            # Stride=2 cuts the 32x96 input down to a 16x48 latent space
            nn.Conv2d(in_channels, feature_dim // 2, kernel_size=3, stride=2, padding=1),
            CustomOCR.SurgicalFocusBlock(in_channels=feature_dim // 2, reduction=16),
            nn.Conv2d(feature_dim // 2, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, feature_dim)
        )
        
        # 2. SVTR Cognitive Stages
        total_depth = sum(depths)
        mixers = []
        for i, d in enumerate(depths):
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

    def forward(self, x): 
        B, C_in, H, W = x.shape
        feat_stem = self.patch_embed(x) 
        feat_seq = feat_stem.flatten(2).transpose(1, 2).contiguous() 
        trajectory = [feat_seq] 
        
        for blk in self.svtr_blocks:
            feat_seq = blk(feat_seq)
            trajectory.append(feat_seq) 
            
        feat_seq = self.norm(feat_seq)
        H_latent = H // 2
        W_latent = W // 2 
        
        feat_2d = feat_seq.transpose(1, 2).contiguous().view(B, -1, H_latent, W_latent)
        feat = feat_2d + feat_stem
        
        # ---> THE UPGRADE: Return the High-Res Latent!
        return feat, None, trajectory
    
# ==============================================================================
# 4. THE CGNET WRAPPER & OCR INTEGRATION
# ==============================================================================
class Cgnet(nn.Module):
    def __init__(self, in_channels=3, mode='hybrid', feature_dim=128, cnn_heads=4, **kwargs): 
        super(Cgnet, self).__init__()
        self.mode = mode.lower()
        self.feature_dim = feature_dim
        
        self.student_extractor = CognitiveSVTRBackbone(
            in_channels, feature_dim, depths=[3, 6, 3], cnn_heads=cnn_heads
        )
        
        if self.mode in ['ocr', 'hybrid']:
            # ---> THE FIX: The OCR now expects the glorious 32x96 High-Res Latent Space!
            self.student_ocr = CustomOCR.CustomOCR(
                input_shape=(self.feature_dim, 16, 48), num_classes=38, num_chars=7,  
            )
            
    def forward(self, x, temporal_pool=False, return_latent=True, **kwargs): 
        if x.dim() == 5: x = x.squeeze(1) 
            
        sr_latent, sr_image, trajectory = self.student_extractor(x)
        epoch = kwargs.get('epoch', 0)
        
        preds_lr = self.student_ocr(sr_latent, epoch=epoch)
        
        # ---> THE FIX: Only pass the image to the framework IF it exists!
        if sr_image is not None:
            preds_lr['sr_image'] = sr_image
            
        preds_lr['trajectory'] = trajectory 
        
        if return_latent:
            preds_lr['latent_lr'] = sr_latent
            
        return preds_lr

class SR_LPR_NET(nn.Module):
    def __init__(self, in_channels=3, mode='hybrid', feature_dim=128, **kwargs): 
        super().__init__()
        self.cgnet = Cgnet(in_channels=in_channels, mode=mode, feature_dim=feature_dim, **kwargs)

    def forward(self, x, **kwargs):
        return self.cgnet(x, **kwargs)

@register('VSR_CURVATURE') 
def make(**kwargs): return SR_LPR_NET(**kwargs)