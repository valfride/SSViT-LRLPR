import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops 
from models import register
from . import CustomOCR 
from torch.cuda.amp import GradScaler, autocast

# ==============================================================================
# 1. SOBEL EDGE EXTRACTOR (The "Structure" Branch)
# ==============================================================================
class SobelLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # Hardcoded Sobel Kernels (No learning needed, pure math)
        self.kernel_x = nn.Parameter(torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3), requires_grad=False)
        self.kernel_y = nn.Parameter(torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3), requires_grad=False)
    @autocast(enabled=False)
    def forward(self, x):
        # x: [B, 3, H, W] -> Convert to Grayscale
        if x.size(1) == 3:
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x
            
        # 1. Compute Gradients
        grad_x = F.conv2d(gray, self.kernel_x, padding=1)
        grad_y = F.conv2d(gray, self.kernel_y, padding=1)
        
        # 2. Force Magnitude Calculation to FP32 to avoid Overflow
        # The squaring operation (x^2) is the dangerous part in FP16
        #with torch.cuda.amp.autocast(enabled=False):
        #    grad_x_f32 = grad_x.float()
        #    grad_y_f32 = grad_y.float()
        #    
        #   # Safe calculation in FP32 space
        #    magnitude = torch.sqrt(grad_x_f32**2 + grad_y_f32**2 + 1e-6)
            
        # Cast back to original type (FP16) for the next layers
        return torch.cat([grad_x, grad_y], dim=1)

# ==============================================================================
# 2. DEFORMABLE CONVOLUTION
# ==============================================================================
class DeformableConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, bias=False):
        super(DeformableConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = 1
        
        self.conv_offset = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=1)
        self.conv_mask = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=1)
        
        self.weight = nn.Parameter(torch.Tensor(outc, inc, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(outc)) if bias else None
        
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=0.1)
        nn.init.constant_(self.conv_offset.weight, 0)
        nn.init.constant_(self.conv_offset.bias, 0)
        nn.init.constant_(self.conv_mask.weight, 0)
        nn.init.constant_(self.conv_mask.bias, 0)

    def forward(self, x):
        offset = self.conv_offset(x)
        mask = torch.sigmoid(self.conv_mask(x)) 
        return ops.deform_conv2d(input=x, offset=offset, weight=self.weight, bias=self.bias, 
                                 stride=(1, 1), padding=(self.padding, self.padding), mask=mask)

# ==============================================================================
# 3. COORDINATE ATTENTION
# ==============================================================================
class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1)) 
        self.pool_w = nn.AdaptiveAvgPool2d((1, None)) 

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.InstanceNorm2d(mip, affine=True) 
        self.act = nn.Hardswish()
        
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

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

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        return identity * a_h * a_w

# ==============================================================================
# 4. RESIDUAL BLOCK
# ==============================================================================
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(dim, affine=True)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        
        self.conv2 = DeformableConv2d(dim, dim, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(dim, affine=True)
        
        self.att = CoordAtt(dim) 

    def forward(self, x):
        res = self.conv1(x)
        res = self.norm1(res)
        res = self.act(res)
        
        res = self.conv2(res)
        res = self.norm2(res)
        
        res = self.att(res) 
        
        return self.act(x + res) 

# ==============================================================================
# 5. MAIN BACKBONE (Cgnet with Global Edge Skip)
# ==============================================================================
class Cgnet(nn.Module):
    def __init__(self, in_channels=3, mode='hybrid', feature_dim=128):
        super(Cgnet, self).__init__()
        self.mode = mode.lower()
        
        # 1. Edge Branch (New)
        self.sobel = SobelLayer()
        self.edge_encoder = nn.Sequential(
            nn.Conv2d(2, 32, 3, 1, 1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(True)
        )
        
        # 2. Texture Branch (Existing)
        self.shallow_feat = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(feature_dim, affine=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.body = nn.Sequential(*[ResBlock(feature_dim) for _ in range(8)])
        self.conv_after_body = nn.Conv2d(feature_dim, feature_dim, 3, 1, 1)

        # 3. Upsampling (Common)
        self.upsample_block = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.InstanceNorm2d(feature_dim, affine=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # 4. Edge Upsampling (Independent Highway)
        self.edge_upsample = nn.Sequential(
            nn.Conv2d(32, 32 * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.InstanceNorm2d(32, affine=True),
            nn.Sigmoid() # Edges are 0 to 1
        )
        
        # 5. Fusion (Merge Texture + Edges)
        # Input to OCR will be feature_dim (from texture) + 32 (from edges)
        self.fusion_conv = nn.Conv2d(feature_dim + 32, feature_dim, 1, 1, 0)

        # D. Head Integration
        if self.mode in ['ocr', 'hybrid']:
            self.ocr_head = CustomOCR.CustomOCR(
                input_shape=(feature_dim, 64, 192), 
                num_classes=37, 
                num_chars=7,
                d_model=128
            )
        
    def forward(self, x, **kwargs): 
        text_labels = kwargs.pop('text_labels', None)
        
        if x.dim() == 5:
            b, c, t, h, w = x.shape
            x = x.reshape(b*t, c, h, w)
        raw_lr = x
        # A. Edge Branch (Structure)
        edges = self.sobel(x)            # [B, 1, H, W]
        edge_feat = self.edge_encoder(edges) # [B, 32, H, W]
        hr_edges = self.edge_upsample(edge_feat) # [B, 32, 2H, 2W]
        
        # B. Texture Branch (Content)
        feat_shallow = self.shallow_feat(x) 
        feat_deep = self.body(feat_shallow)
        feat_residual = self.conv_after_body(feat_deep) + feat_shallow
        hr_texture = self.upsample_block(feat_residual) # [B, Dim, 2H, 2W]
        
        # C. Global Fusion (Skip Connection)
        # We Concatenate instead of Add, to preserve the distinct edge info
        combined = torch.cat([hr_texture, hr_edges], dim=1) # [B, Dim+32, 2H, 2W]
        final_hr = self.fusion_conv(combined) # [B, Dim, 2H, 2W]
        
        if self.mode in ['ocr', 'hybrid']:
            return self.ocr_head(final_hr, raw_lr=raw_lr, text_labels=text_labels, **kwargs)            
        return None

class SR_LPR_NET(nn.Module):
    def __init__(self, mode='hybrid', feature_dim=96, **kwargs):
        super().__init__()
        self.cgnet = Cgnet(in_channels=3, mode=mode, feature_dim=feature_dim)

    def forward(self, x, **kwargs):
        return self.cgnet(x, **kwargs)

@register('VSR_CURVATURE') 
def make(**kwargs): return SR_LPR_NET(**kwargs)
