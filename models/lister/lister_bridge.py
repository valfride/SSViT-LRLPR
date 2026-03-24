import torch
import torch.nn as nn
from models import register
from models.svtrv2.svtrv2_lnconv_two33 import SVTRv2LNConvTwo33
from .lister_decoder import LISTERDecoder

class LISTER_Baseline(nn.Module):
    def __init__(self, in_channels=3, max_len=7, num_classes=37, **kwargs):
        super().__init__()
        # 1. SVTRv2 Backbone (configured for 24x72 crops)
        self.encoder = SVTRv2LNConvTwo33(
            max_sz=[24, 72],
            in_channels=in_channels,
            dims=[64, 128, 256],
            depths=[3, 6, 3],
            num_heads=[2, 4, 8],
            mixer=[['Conv']*3, ['Conv']*3 + ['FGlobal'] + ['Global']*2, ['Global']*3],
            sub_k=[[1, 1], [2, 1], [1, 1]], 
            feat2d=True # Required to pass 2D maps to the NRM decoder
        )
        
        # 2. LISTER Decoder (The Detective)
        self.decoder = LISTERDecoder(
            in_channels=256,
            out_channels=num_classes + 1, # LISTER adds an EOS token
            max_len=max_len,
            iters=1,
            use_fem=True,
            attn_scaling=True
        )

    def forward(self, x, tgt=None, **kwargs):
        # x: [B, 3, 24, 72]
        vis_feats = self.encoder(x) 
        
        data = None
        if self.training and tgt is not None:
            # 1. Pad the 7-slot target to 8 slots using LISTER's ignore_index
            # This gives LISTER the extra slot it needs for its End-Of-Sequence math
            ignore_idx = self.decoder.ignore_index 
            tgt_padded = torch.nn.functional.pad(tgt, (0, 1), value=ignore_idx)
            
            # 2. Hardcode lengths to 7 (since your license plates are fixed length)
            lens = torch.full((x.size(0),), 7, device=x.device)
            data = (tgt_padded, lens)
            
        # Run Decoder
        loss_dict, res = self.decoder(vis_feats, data=data)
        
        # Slice the max-len logits back to your 7 character slots
        return {
            'logits': res['logits'][:, :7, :], 
            'loss_internal': loss_dict['loss'] if loss_dict else None,
            'attn_maps': None
        }

@register('LISTER_BASELINE')
def make_lister(**kwargs):
    return LISTER_Baseline(**kwargs)