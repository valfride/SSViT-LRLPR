import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from models import register
from . import CustomOCR 

# ==============================================================================
# 1. HYBRID SOTA BLOCKS (Global Sequence + Dense Deformable Attention)
# ==============================================================================
class GlobalSweepMixer(nn.Module):
    """Pure PyTorch approximation of Vision Mamba's 1D spatial sweep."""
    def __init__(self, dim):
        super().__init__()
        # A Bidirectional GRU acts as our linear State Space Model (SSM)
        self.sweep = nn.GRU(dim, dim // 2, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        # Flatten the image into a 1D sequence: (B, C, H, W) -> (B, H*W, C)
        flat_x = x.view(b, c, -1).permute(0, 2, 1)
        
        # Sweep across the entire sequence linearly
        swept, _ = self.sweep(flat_x)
        swept = self.norm(swept)
        
        # Fold it back into a 2D image
        return swept.permute(0, 2, 1).view(b, c, h, w)

class DenseDeformableAttention(nn.Module):
    """Image-to-Image Deformable Attention. Every pixel throws a lasso to K neighbors."""
    def __init__(self, dim, num_heads=4, k_points=9):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.k_points = k_points
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Predicts (X, Y) offsets for the K sampling points for EVERY pixel
        self.offset_net = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, 2 * k_points, kernel_size=1)
        )
        
        # Initialize offsets to 0 so it starts as a standard centered grid
        nn.init.constant_(self.offset_net[-1].weight, 0)
        nn.init.constant_(self.offset_net[-1].bias, 0)
        
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        
        # 1. Generate Normalized Base Grid [-1, 1] for the whole image
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).view(1, H, W, 1, 2)
        
        # 2. Predict the dynamic offsets
        offsets = self.offset_net(x).view(B, self.k_points, 2, H, W).permute(0, 3, 4, 1, 2)
        offsets = torch.tanh(offsets) * 0.2 # Scale down to prevent flying off screen
        
        # 3. Create the sampling grids
        sample_grid = base_grid + offsets 
        sample_grid = sample_grid.reshape(B, H, W * self.k_points, 2)
        
        # 4. Extract the dynamic Keys and Values
        k_map = self.k_proj(x)
        v_map = self.v_proj(x)
        
        k_sampled = F.grid_sample(k_map, sample_grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        v_sampled = F.grid_sample(v_map, sample_grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        
        k_sampled = k_sampled.view(B, C, N, self.k_points).permute(0, 2, 3, 1)
        v_sampled = v_sampled.view(B, C, N, self.k_points).permute(0, 2, 3, 1)
        
        # 5. Prepare the Queries (Standard local pixels)
        q = self.q_proj(x).view(B, C, N).permute(0, 2, 1).unsqueeze(2) 
        
        # 6. Multi-Head Attention
        q = q.view(B, N, 1, self.num_heads, self.head_dim)
        k = k_sampled.view(B, N, self.k_points, self.num_heads, self.head_dim)
        v = v_sampled.view(B, N, self.k_points, self.num_heads, self.head_dim)
        
        attn = torch.einsum('bnqhd,bnkhd->bnhqk', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.einsum('bnhqk,bnkhd->bnqhd', attn, v).reshape(B, N, C)
        
        # 7. Norm, Reshape, and Project
        out = self.norm(out).permute(0, 2, 1).view(B, C, H, W)
        return self.out_proj(out)

class DeformableHybridBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.global_sweep = GlobalSweepMixer(dim)
        self.local_deform = DenseDeformableAttention(dim, num_heads=num_heads, k_points=9)
        
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.GroupNorm(8, dim),
            FReLU(dim)
        )

    def forward(self, x):
        global_feat = self.global_sweep(x)
        local_feat = self.local_deform(x) 
        combined = torch.cat([global_feat, local_feat], dim=1)
        return x + self.fusion(combined)

# ==============================================================================
# 2. VSR & CNN UTILITIES
# ==============================================================================
class MultiScaleContext(nn.Module):
    def __init__(self, dim):
        super().__init__()
        branch_dim = dim // 4
        self.branch1 = nn.Sequential(nn.Conv2d(dim, branch_dim, kernel_size=1), nn.GroupNorm(2, branch_dim), nn.Mish())
        self.branch2 = nn.Sequential(nn.Conv2d(dim, branch_dim, kernel_size=3, padding=2, dilation=2), nn.GroupNorm(2, branch_dim), nn.Mish())
        self.branch3 = nn.Sequential(nn.Conv2d(dim, branch_dim, kernel_size=3, padding=4, dilation=4), nn.GroupNorm(2, branch_dim), nn.Mish())
        self.branch4 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, branch_dim, kernel_size=1), nn.Mish())
        
        self.fuse = nn.Sequential(nn.Conv2d(branch_dim * 4, dim, kernel_size=1), nn.GroupNorm(8, dim))

    def forward(self, x):
        b1, b2, b3 = self.branch1(x), self.branch2(x), self.branch3(x)
        b4 = self.branch4(x).expand_as(b1) 
        return self.fuse(torch.cat([b1, b2, b3, b4], dim=1))

class LatentUpsampler(nn.Module):
    def __init__(self, dim, upscale_factor=2):
        super().__init__()
        self.multi_scale_analyzer = MultiScaleContext(dim)
        self.up_proj = nn.Conv2d(dim, dim * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.upsample = nn.PixelShuffle(upscale_factor)
        self.refine = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, padding=1), nn.GroupNorm(8, dim), nn.Mish())

    def forward(self, x):
        enriched_x = x + self.multi_scale_analyzer(x)
        hr_grid = self.upsample(self.up_proj(enriched_x))
        return self.refine(hr_grid)

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

class FReLU(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.spatial_condition = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels
        )
        self.norm = nn.GroupNorm(8, in_channels)

    def forward(self, x):
        spatial_context = self.norm(self.spatial_condition(x))
        return torch.max(x, spatial_context)

# ==============================================================================
# 3. SINGLE-IMAGE SPATIAL FEATURE EXTRACTOR
# ==============================================================================
class SpatialFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, feature_dim=128, cnn_heads=4):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, 3, 1, 1),
            nn.GroupNorm(8, feature_dim),
            HighContrastGate(feature_dim),
            nn.Conv2d(feature_dim, feature_dim, 3, 1, 1),
            nn.GroupNorm(8, feature_dim),
            FReLU(feature_dim)
        )
        
        # Deep Restoration with the new Deformable Hybrid Blocks
        self.body = nn.Sequential(*[DeformableHybridBlock(feature_dim, num_heads=cnn_heads) for _ in range(2)])
        self.conv_after_body = nn.Conv2d(feature_dim, feature_dim, 3, 1, 1)
        
        self.latent_sr = LatentUpsampler(feature_dim, upscale_factor=2)
        
        self.hr_refine = nn.Sequential(
            DeformableProj(feature_dim, feature_dim, kernel_size=3, offset_groups=4),
            nn.GroupNorm(8, feature_dim),
            FReLU(feature_dim),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, feature_dim),
            FReLU(feature_dim)
        )

    def forward(self, x): 
        feat_shallow = self.stem(x) 
        feat_deep = self.body(feat_shallow)
        texture = self.conv_after_body(feat_deep) + feat_shallow 
        feat_sr = self.latent_sr(texture)
        return self.hr_refine(feat_sr)
        
# ==============================================================================
# 4. THE CGNET WRAPPER & OCR INTEGRATION
# ==============================================================================
class Cgnet(nn.Module):
    def __init__(self, in_channels=3, mode='hybrid', feature_dim=128, cnn_heads=4, **kwargs): 
        super(Cgnet, self).__init__()
        self.mode = mode.lower()
        self.feature_dim = feature_dim
        
        self.student_extractor = SpatialFeatureExtractor(in_channels, feature_dim, cnn_heads=cnn_heads)
        
        if self.mode in ['ocr', 'hybrid']:
            self.student_ocr = CustomOCR.CustomOCR(
                # Expecting 128 latent channels + 3 RGB channels = 131
                input_shape=(feature_dim + in_channels, 48, 144),
                num_classes=39, 
                num_chars=12, 
            )

    def forward(self, x, temporal_pool=False, return_latent=True, **kwargs): 
        if x.dim() == 5:
            x = x.squeeze(1) 
            
        # 1. Extract Super-Resolved Features (48x144)
        latent_lr = self.student_extractor(x)
        
        # 2. Stretch the raw input to match the upsampled latent map
        x_up = F.interpolate(x, size=latent_lr.shape[2:], mode='bilinear', align_corners=False)
        
        # 3. Global Skip Connection (Fusing the real image with the VSR hallucination)
        conditioned_latent = torch.cat([latent_lr, x_up], dim=1)

        # 4. Pass the combined 131-channel tensor to the OCR Spotter
        epoch = kwargs.get('epoch', 0)
        preds_lr = self.student_ocr(conditioned_latent, epoch=epoch)
        
        # Compatibility placeholders
        preds_lr['loss_vq'] = torch.tensor(0.0, device=x.device)
        preds_lr['kld_raw'] = torch.tensor(0.0, device=x.device)

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