import torch
import torch.nn as nn
from models import register
from models.cppd.svtrnet import SVTRNet
from .ote_decoder import OTEDecoder

class OTELossWrapper(nn.Module):
    def __init__(self, ignore_index=38):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=0.1)
        
    def forward(self, preds_dict, targets):
        logits = preds_dict['logits'] # [B, 8, 37]
        # targets: [B, 9] -> [BOS, c1..c7, EOS, PAD]
        # OTE is Autoregressive: It predicts the NEXT token.
        tgt = targets[:, 1 : logits.shape[1] + 1].contiguous()
        return self.ce(logits.flatten(0, 1), tgt.flatten(0))

class OTEBaseline(nn.Module):
    def __init__(self, in_channels=3, max_len=7, num_classes=37, **kwargs):
        super().__init__()
        self.encoder = SVTRNet(
            img_size=[32, 96], in_channels=in_channels, out_char_num=max_len, out_channels=256,
            patch_merging='Conv', embed_dim=[128, 256, 384], depth=[6, 6, 6],
            num_heads=[4, 8, 12], mixer=['Conv']*8 + ['Global']*10,
            local_mixer=[[5, 5], [5, 5], [5, 5]], last_stage=False, prenorm=True
        )
        
        # OTE internal math: 37 standard classes + BOS + PAD = 39
        self.decoder = OTEDecoder(
            in_channels=384, out_channels=39, max_len=max_len,
            num_heads=12, ar=True, num_decoder_layers=1
        )

    def forward(self, x, tgt=None, **kwargs):
        vis_feats = self.encoder(x)
        if self.training and tgt is not None:
            lens = torch.full((x.size(0),), 7, device=x.device)
            logits = self.decoder(vis_feats, data=(tgt, lens))
            return {'logits': logits, 'attn_maps': None}
        else:
            logits = self.decoder(vis_feats)
            # Slice off special tokens to keep evaluation safe
            return {'logits': logits[:, :7, :37], 'attn_maps': None}

@register('OTE_BASELINE')
def make_ote(**kwargs): return OTEBaseline(**kwargs)