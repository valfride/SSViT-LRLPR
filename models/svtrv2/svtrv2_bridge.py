import torch
import torch.nn as nn
from models import register

# try:
#     from .svtrv2 import SVTRv2
#     from .ctc_decoder import CTCDecoder
# except ImportError:
#     print("⚠️ Please place svtrv2.py and ctc_decoder.py in your models/ folder.")
from .svtrv2 import SVTRv2
from models.ctc_decoder import CTCDecoder

class True_SVTRv2_Baseline(nn.Module):
    def __init__(self, num_classes=37, **kwargs):
        super().__init__()
        
        # 1. The Official Vision Backbone
        self.backbone = SVTRv2(
            max_sz=[32, 96],       
            in_channels=3,
            out_channels=192, 
            dims=[64, 128, 256],   
            depths=[3, 6, 3],      
            num_heads=[2, 4, 8],
            mixer=[
                ['Local', 'Local', 'Local'], 
                ['Local', 'Local', 'Local', 'Global', 'Global', 'Global'], 
                ['Global', 'Global', 'Global']
            ],
            last_stage=True # <--- THE FIX: Forces the backbone to output (B, W, 192)
        )
        
        # 2. The Official CTC Head 
        self.classifier = CTCDecoder(
            in_channels=192, 
            out_channels=num_classes, 
            svtr_encoder=None 
        )

    def forward(self, x, **kwargs):
        # 1. Extract Visual Features (Output is already B, Time, Channels)
        features = self.backbone(x)     # (B, W, 192)
        
        # 2. Official CTC Decode
        logits_or_probs = self.classifier(features)   # (B, W, 37)
        
        return {
            'logits': logits_or_probs, 
            'attn_maps': None,
            'latent_lr': None,
            'z_vector': None
        }

@register('SVTRV2_BASELINE')
def make(**kwargs):
    return True_SVTRv2_Baseline(**kwargs)