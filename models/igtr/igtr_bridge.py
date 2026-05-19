import torch
import torch.nn as nn
from models import register
from .svtrnet2dpos import SVTRNet2DPos
from .igtr_decoder import IGTRDecoder 
from .igtr_loss import IGTRLoss

class IGTR_Baseline(nn.Module):
    def __init__(self, in_channels=3, max_len=25, num_classes=37, **kwargs):
        super().__init__()
        
        # 1. Visual Encoder (Exact OpenOCR Base Parameters)
        self.encoder = SVTRNet2DPos(
            in_channels=in_channels,
            out_channels=384,
            out_char_num=max_len, 
            embed_dim=[128, 256, 384],
            depth=[3, 6, 9],
            num_heads=[4, 8, 12],
            mixer=['Local']*5 + ['Global']*13,
            local_mixer=[[7, 11], [7, 11], [7, 11]],
            last_stage=False,
            feat2d=True
        )
        
        # 2. Instruction Decoder (3 Layers!)
        # 2. Instruction Decoder (3 Layers!)
        self.decoder = IGTRDecoder(
            in_channels=384,          
            dim=384,                  
            out_channels=num_classes, 
            num_decoder_layers=3, 
            max_len=max_len,
            
            # ---> THE FIX: Tell the decoder to process a 4D spatial map!
            ds=True,
            pos2d=True
        )
        
        # 3. Internal Task Loss 
        self.criterion = IGTRLoss()

    def forward(self, x, tgt=None, epoch=0, **kwargs):
        vis_feats = self.encoder(x)
        
        if self.training and tgt is not None:
            # 1. The decoder outputs a tuple: (loss_dict, logits)
            preds = self.decoder(vis_feats, tgt)
            
            # 2. Extract the loss object
            loss_obj = self.criterion(preds, tgt) 
            
            # ---> THE FIX: Unpack OpenOCR's nested outputs into a pure scalar tensor <---
            if isinstance(loss_obj, (tuple, list)):
                loss_obj = loss_obj[0]
            if isinstance(loss_obj, dict):
                loss_obj = loss_obj['loss'] # Grab the final combined scalar loss!
            
            return {
                'logits': preds[1],
                'loss_internal': loss_obj,  # <--- Now a pure float tensor!
                'attn_maps': None
            }
        else:
            # ---> THE FIX: Use the native forward pass for inference! <---
            # IGTR automatically routes to its inference logic when data/tgt is None.
            preds = self.decoder(vis_feats, None) 
            
            # OpenOCR inference functions sometimes return tuples (logits, hidden_states). 
            # We safely unpack just the logits for your validation metric functions.
            if isinstance(preds, (tuple, list)):
                logits = preds[0]
            elif isinstance(preds, dict):
                logits = preds.get('logits', preds)
            else:
                logits = preds

            return {
                'logits': logits, 
                'loss_internal': None,
                'attn_maps': None,
                'query_tokens': None, 
                'latent_lr': vis_feats
            }

@register('IGTR_BASELINE')
def make_igtr(**kwargs):
    return IGTR_Baseline(**kwargs)