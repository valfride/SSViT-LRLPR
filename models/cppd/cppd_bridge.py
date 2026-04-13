import torch
import torch.nn as nn
from models import register
from .svtrnet import SVTRNet
from .cppd_decoder import CPPDDecoder

class CPPDLossWrapper(nn.Module):
    """Encapsulates the OpenOCR CPPD Loss for seamless integration."""
    def __init__(self, ignore_index=100, max_len=7):
        super().__init__()
        self.edge_ce = nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index)
        self.char_node_ce = nn.CrossEntropyLoss(reduction='mean')
        self.pos_node_ce = nn.BCEWithLogitsLoss(reduction='mean')
        self.max_len = max_len + 1
        
    def forward(self, preds_dict, targets):
        char_tgt, node_tgt = targets
        node_feats = preds_dict['node_feats']
        edge_feats = preds_dict['edge_feats']
        
        char_num_label = torch.clip(node_tgt[:, :-self.max_len].flatten(0, 1), 0, node_feats[0].shape[-1] - 1)
        loss_char_node = self.char_node_ce(node_feats[0].flatten(0, 1), char_num_label)
        loss_pos_node = self.pos_node_ce(node_feats[1].flatten(0, 1), node_tgt[:, -self.max_len:].flatten(0, 1).float())
        loss_node = loss_char_node + loss_pos_node
        
        edge_feats = edge_feats.flatten(0, 1)
        char_tgt = char_tgt.flatten(0, 1)
        loss_edge = self.edge_ce(edge_feats, char_tgt)
        
        return loss_node + loss_edge

class CPPDBaseline(nn.Module):
    def __init__(self, in_channels=3, max_len=7, num_classes=37, **kwargs):
        super().__init__()
        self.encoder = SVTRNet(
            img_size=[32, 96], # <--- THE FIX: Explicitly tell SVTR the new shape
            in_channels=in_channels, out_char_num=max_len, out_channels=256,
            patch_merging='Conv', embed_dim=[128, 256, 384], depth=[6, 6, 6],
            num_heads=[4, 8, 12], mixer=['Conv']*8 + ['Global']*10,
            local_mixer=[[5, 5], [5, 5], [5, 5]], last_stage=False, prenorm=True
        )
        self.decoder = CPPDDecoder(
            in_channels=384, out_channels=num_classes, num_layer=2,
            vis_seq=48, # <--- THE FIX: 32x96 yields exactly 48 flattened patches (2x24)
            pos_len=False, rec_layer=1, max_len=max_len
        )
        
    def forward(self, x, tgt=None, epoch=0, **kwargs):
        vis_feats = self.encoder(x)
        if self.training:
            node_feats, edge_feats = self.decoder(vis_feats)
            return {'logits': edge_feats, 'node_feats': node_feats, 'edge_feats': edge_feats, 'attn_maps': None}
        else:
            edge_logits = self.decoder(vis_feats)
            return {'logits': edge_logits, 'attn_maps': None}

@register('CPPD_BASELINE')
def make_cppd(**kwargs): return CPPDBaseline(**kwargs)