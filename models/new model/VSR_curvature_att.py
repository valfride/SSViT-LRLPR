import torch
import torch.nn as nn
import torch.nn.functional as F
from models import register
from . import CustomOCR 
from torch.cuda.amp import autocast

import torch.nn.utils as nn_utils

class SotaPatchGAN(nn.Module):
    """
    SOTA Spectral-Normalized PatchGAN Discriminator.
    Extremely stable, no BatchNorm required.
    """
    def __init__(self, in_channels=3, ndf=64):
        super().__init__()
        
        # Helper to apply Spectral Norm to convolutions (FIXED ARGUMENTS)
        def sn_conv(in_c, out_c, kernel_size, stride, padding, bias=False):
            return nn_utils.spectral_norm(nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=bias))
            
        # Input: [B, 3, 64, 192]
        self.net = nn.Sequential(
            # Layer 1: 32x96
            sn_conv(in_channels, ndf, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: 16x48
            sn_conv(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: 8x24
            sn_conv(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4: 8x24 (Stride 1 to retain feature map size)
            sn_conv(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output: A localized 1-channel patch map
            sn_conv(ndf * 8, 1, kernel_size=4, stride=1, padding=1, bias=True)
        )

    def forward(self, x):
        return self.net(x)

# ==============================================================================
# 1. RESTORMER BLOCKS (Topological Sharpeners in Latent Space)
# ==============================================================================
class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention: Computes attention across channels."""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Sequential(
            nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False),
            nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=False) 
        )
        self.attn_drop = nn.Dropout(dropout)
        
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        b, c, h, w = x.shape
        
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.view(b, self.num_heads, c // self.num_heads, h * w)
        k = k.view(b, self.num_heads, c // self.num_heads, h * w)
        v = v.view(b, self.num_heads, c // self.num_heads, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = out.view(b, c, h, w)

        out = self.project_out(out)
        return self.proj_drop(out)

class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network: Cleans up feature artifacts."""
    def __init__(self, dim, expansion_factor=2.66, dropout=0.1):
        super(GDFN, self).__init__()
        hidden_features = int(dim * expansion_factor)
        
        self.project_in = nn.Sequential(
            nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=False),
            nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, padding=1, groups=hidden_features * 2, bias=False)
        )
        
        # SOTA FIX: Feature Dropout forces distributed feature learning
        self.drop = nn.Dropout(dropout)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.project_in(x).chunk(2, dim=1)
        x = x1 * F.gelu(x2)
        x = self.drop(x)
        return self.project_out(x)

class RestormerBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super(RestormerBlock, self).__init__()
        self.norm1 = nn.GroupNorm(1, dim) 
        self.attn = MDTA(dim, dropout=dropout)
        self.norm2 = nn.GroupNorm(1, dim)
        self.ffn = GDFN(dim, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# ==============================================================================
# 2. MULTI-SCALE SOBEL EDGE EXTRACTOR 
# ==============================================================================
class SobelLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_x = nn.Parameter(torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3), requires_grad=False)
        self.kernel_y = nn.Parameter(torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3), requires_grad=False)
        
    def forward(self, x):
        if x.size(1) == 3:
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x
            
        # Dynamically cast kernels to match the current precision of the input (FP16 or FP32)
        kx = self.kernel_x.to(x.dtype)
        ky = self.kernel_y.to(x.dtype)
        
        grad_x_fine = F.conv2d(gray, kx, padding=1, dilation=1)
        grad_y_fine = F.conv2d(gray, ky, padding=1, dilation=1)
        grad_x_med = F.conv2d(gray, kx, padding=2, dilation=2)
        grad_y_med = F.conv2d(gray, ky, padding=2, dilation=2)
        grad_x_ext = F.conv2d(gray, kx, padding=(1, 3), dilation=(1, 3))
        
        return torch.cat([grad_x_fine, grad_y_fine, grad_x_med, grad_y_med, grad_x_ext], dim=1)

class SymmetricalFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, feature_dim=128):
        super().__init__()
        self.sobel = SobelLayer()
        # Edge encoder expects 7 channels (5 Sobel edges + X + Y)
        self.edge_encoder = nn.Sequential(nn.Conv2d(7, 32, 3, 1, 1), nn.GroupNorm(8, 32), nn.ReLU(True))
        
        # Shallow feat expects 5 channels (3 RGB + X + Y)
        self.shallow_feat = nn.Sequential(nn.Conv2d(in_channels + 2, feature_dim, 3, 1, 1), nn.GroupNorm(8, feature_dim), nn.LeakyReLU(0.1, True))
        
        self.body = nn.Sequential(*[RestormerBlock(feature_dim, dropout=0.1) for _ in range(2)])
        self.conv_after_body = nn.Conv2d(feature_dim, feature_dim, 3, 1, 1)
        self.fusion_conv = nn.Sequential(nn.Conv2d(feature_dim + 32, feature_dim, 1, 1, 0), nn.GroupNorm(8, feature_dim), nn.ReLU(True))

    def forward(self, img, x_grid, y_grid):
        img_with_coords = torch.cat([img, x_grid, y_grid], dim=1)
        
        edges = self.sobel(img)            
        edges_with_coords = torch.cat([edges, x_grid, y_grid], dim=1)
        encoded_edges = self.edge_encoder(edges_with_coords) 
        
        feat_shallow = self.shallow_feat(img_with_coords) 
        feat_deep = self.body(feat_shallow)
        texture = self.conv_after_body(feat_deep) + feat_shallow 
        
        latent_features = self.fusion_conv(torch.cat([texture, encoded_edges], dim=1))
        return latent_features

# ==============================================================================
# SOTA NAFNet BLOCKS (Robust Neural Rendering)
# ==============================================================================
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1)
        )

        # SimpleGate replaces ReLU/GELU
        self.sg = SimpleGate()
        
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1)

        self.norm1 = nn.LayerNorm(c, eps=1e-6)
        self.norm2 = nn.LayerNorm(c, eps=1e-6)
        
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # Weight scaling for stability
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x):
        input = x
        x = input.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x) # Channel Attention
        x = self.conv3(x)
        
        x = self.dropout1(x)
        y = input + x * self.beta

        input = y
        x = input.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2)
        
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        return y + x * self.gamma

class Cgnet(nn.Module):
    def __init__(self, in_channels=3, mode='hybrid', feature_dim=128):
        super(Cgnet, self).__init__()
        self.mode = mode.lower()
        self.feature_dim = feature_dim
        
        # 1. BACKBONES
        self.student_extractor = SymmetricalFeatureExtractor(in_channels, feature_dim)
        self.teacher_extractor = SymmetricalFeatureExtractor(in_channels, feature_dim)

        # 2. OCR HEADS (Now using the new ViT CustomOCR)
        if self.mode in ['ocr', 'hybrid']:
            self.teacher_ocr = CustomOCR.CustomOCR(input_shape=(feature_dim, 32, 96), num_classes=37, num_chars=7, d_model=256)
            self.student_ocr = CustomOCR.CustomOCR(input_shape=(feature_dim, 32, 96), num_classes=37, num_chars=7, d_model=256)
        
        # 3. GATES
        self.student_temporal_gate = self._build_gate(feature_dim)
        self.teacher_temporal_gate = self._build_gate(feature_dim)

        # 4. DECODERS
        # Input Channels = Feature(128) + X_Grid(1) + Y_Grid(1) + RGB_Image(3) = 133
        self.student_decoder = self._build_decoder(feature_dim + 2 + 3)
        self.teacher_decoder = self._build_decoder(feature_dim + 2 + 3)
        
    def _build_gate(self, dim):
        return nn.Sequential(
            nn.Conv3d(dim, dim // 4, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.GroupNorm(8, dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim // 4, dim, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.Sigmoid() 
        )

    def _build_decoder(self, dim):
        internal_dim = dim - 5 
        return nn.Sequential(
            nn.Conv2d(dim, internal_dim, kernel_size=3, padding=1),
            NAFBlock(internal_dim), NAFBlock(internal_dim),
            NAFBlock(internal_dim), NAFBlock(internal_dim),
            nn.Conv2d(internal_dim, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh() 
        )
        
    def forward(self, x, hr_img=None, temporal_pool=False, **kwargs): 
        # ... (Input Prep & Skip Connections same as before) ...
        b_orig, t_orig = None, None
        x_center = x
        
        if x.dim() == 5:
            b_orig, t_orig, c, h, w = x.shape
            x_center = x[:, t_orig // 2].clone() 
            x = x.reshape(b_orig * t_orig, c, h, w)
            
        if hr_img is not None and hr_img.dim() == 5:
            b_hr, t_hr, c_hr, h_hr, w_hr = hr_img.shape
            hr_center = hr_img[:, t_hr // 2].clone()
            hr_img = hr_img.reshape(b_hr * t_hr, c_hr, h_hr, w_hr)
        else:
            hr_center = hr_img
            
        # 1. COORDCONV GENERATOR
        b_curr, _, h_curr, w_curr = x.shape
        y_grid = torch.linspace(-1, 1, h_curr, device=x.device).view(1, 1, h_curr, 1).expand(b_curr, 1, h_curr, w_curr)
        x_grid = torch.linspace(-1, 1, w_curr, device=x.device).view(1, 1, 1, w_curr).expand(b_curr, 1, h_curr, w_curr)

        # 2. ENCODE
        latent_lr = self.student_extractor(x, x_grid, y_grid)
        latent_hr = None
        if self.training and hr_img is not None:
            latent_hr = self.teacher_extractor(hr_img, x_grid, y_grid)

        # 3. TEMPORAL POOLING
        if b_orig is not None and (temporal_pool or not self.training):
            latent_lr_seq = latent_lr.view(b_orig, t_orig, latent_lr.shape[1], latent_lr.shape[2], latent_lr.shape[3])
            latent_3d = latent_lr_seq.permute(0, 2, 1, 3, 4) 
            attn_mask_lr = self.student_temporal_gate(latent_3d) 
            latent_lr = (latent_3d * attn_mask_lr).permute(0, 2, 1, 3, 4).max(dim=1)[0]
            
            if latent_hr is not None:
                latent_hr_seq = latent_hr.view(b_orig, t_orig, latent_hr.shape[1], latent_hr.shape[2], latent_hr.shape[3])
                latent_hr_3d = latent_hr_seq.permute(0, 2, 1, 3, 4)
                attn_mask_hr = self.teacher_temporal_gate(latent_hr_3d) 
                latent_hr = (latent_hr_3d * attn_mask_hr).permute(0, 2, 1, 3, 4).max(dim=1)[0]
                
        # 4. DECODE
        if self.mode in ['ocr', 'hybrid']:
            curr_b = latent_lr.size(0)
            grid_x_dec = x_grid[:curr_b]
            grid_y_dec = y_grid[:curr_b]

            if self.training and latent_hr is not None:
                preds_hr = self.teacher_ocr(latent_hr)
                preds_lr = self.student_ocr(latent_lr)
                
                # ==============================================================
                # 🆕 DISTILLATION WITH MARGIN (The "Deadzone" Loss)
                # ==============================================================
                
                # We align the extracted ViT features (Character Tokens)
                student_feats = preds_lr['features']
                teacher_feats = preds_hr['features'].detach()

                # 1. Directional Alignment (Cosine Similarity)
                # We normalize the vectors and check if they point in the same direction.
                # Flatten the [B, 7, D] tensor to [B*7, D] for simpler cosine calc
                s_flat = student_feats.reshape(-1, student_feats.size(-1))
                t_flat = teacher_feats.reshape(-1, teacher_feats.size(-1))
                
                # Cosine Similarity returns 1.0 for perfect alignment. We want to minimize (1 - cos).
                loss_cosine = 1.0 - F.cosine_similarity(s_flat, t_flat, dim=-1).mean()
                
                # 2. Magnitude Alignment with Deadzone (Structure)
                # We calculate the absolute difference per feature.
                diff = (student_feats - teacher_feats).abs()
                
                # Define the Margin (Epsilon)
                # 0.1 allows the Teacher to "wobble" due to input noise without punishing the Student.
                margin = 0.1
                
                # Hinge Loss: Only penalize if difference > margin
                loss_magnitude = F.relu(diff - margin).mean()
                
                # Combine: Direction must be exact, but Magnitude allows freedom.
                loss_topology = loss_cosine + loss_magnitude
                
                # Decoder Skip Connections
                safe_latent_hr = latent_hr.detach()
                cat_hr = torch.cat([safe_latent_hr, grid_x_dec, grid_y_dec, hr_center], dim=1)
                pred_hr_img = self.teacher_decoder(cat_hr) 

                safe_latent_lr = latent_lr.detach() 
                cat_lr = torch.cat([safe_latent_lr, grid_x_dec, grid_y_dec, x_center], dim=1)
                pred_lr_img = self.student_decoder(cat_lr)
                
                return preds_lr, preds_hr, loss_topology, pred_hr_img, pred_lr_img
                
            # Inference
            preds_lr = self.student_ocr(latent_lr, **kwargs)
            safe_latent_lr = latent_lr.detach()
            cat_lr = torch.cat([safe_latent_lr, grid_x_dec, grid_y_dec, x_center], dim=1)
            pred_lr_img = self.student_decoder(cat_lr) 
            
            return preds_lr, pred_lr_img
            
        return None

class SR_LPR_NET(nn.Module):
    def __init__(self, mode='hybrid', feature_dim=128, **kwargs): # Changed dim to 128 to match config
        super().__init__()
        self.cgnet = Cgnet(in_channels=3, mode=mode, feature_dim=feature_dim)

    def forward(self, x, **kwargs):
        return self.cgnet(x, **kwargs)

@register('VSR_CURVATURE') 
def make(**kwargs): return SR_LPR_NET(**kwargs)