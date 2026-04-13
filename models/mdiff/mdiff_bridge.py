import torch
import torch.nn as nn
from models import register

from models.svtrv2.svtrv2_lnconv_two33 import SVTRv2LNConvTwo33
from .mdiff_decoder import MDiffDecoder

class MDiff_Baseline(nn.Module):
    def __init__(self, in_channels=3, max_len=7, num_classes=38, **kwargs):
        super().__init__()
        
        # 1. The 1D Sequence Encoder
        self.encoder = SVTRv2LNConvTwo33(
            max_sz=[32, 96],
            in_channels=in_channels,
            dims=[128, 256, 384],
            depths=[6, 6, 6],
            num_heads=[4, 8, 12],
            mixer=[['Conv']*6, ['Conv']*2 + ['FGlobal'] + ['Global']*3, ['Global']*6],
            sub_k=[[1, 1], [2, 1], [-1, -1]], 
            feat2d=False, # MDiff wants flattened 1D sequences
            last_stage=False
        )
        
        # 2. The Diffusion Decoder
        # The Math: Our vocab is 0-36 (37 total). 
        # MDiff explicitly reserves the LAST TWO channels for MASK and IGNORE.
        # So we pass out_channels = 39. 
        # Logits 0-36 = Chars, 37 = MASK, 38 = IGNORE/PAD
        self.decoder = MDiffDecoder(
            in_channels=384,
            out_channels=39, 
            num_decoder_layers=6,
            nhead=6,
            max_len=max_len,
            parallel_decoding=True, # Ultra-fast 1-step inference for evaluation
            sample_k=0 # Standard single-pass masking for training
        )

    def forward(self, x, tgt=None, **kwargs):
        vis_feats = self.encoder(x) 
        
        if self.training and tgt is not None:
            # 1. MDiff calculates its official loss based on the masked tokens
            loss = self.decoder(vis_feats, data=tgt)
            
            # ---> THE FIX: The "Sneak Peek" Inference Pass <---
            # We temporarily disable gradients and dropout to get a clean 
            # prediction for your live monitor, without polluting the training graph!
            with torch.no_grad():
                was_training = self.decoder.training
                self.decoder.eval()
                logits = self.decoder(vis_feats)
                self.decoder.train(was_training)
                
            return {
                'logits': logits[:, :7, :], 
                'loss_internal': loss, 
                'attn_maps': None
            }
        else:
            # Standard Evaluation
            logits = self.decoder(vis_feats)
            return {
                'logits': logits[:, :7, :], 
                'loss_internal': None,
                'attn_maps': None
            }

@register('MDIFF_BASELINE')
def make_mdiff(**kwargs):
    return MDiff_Baseline(**kwargs)