import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import register 

# ==============================================================================
# 1. UTILS
# ==============================================================================
class strLabelConverter(object):
    """
    Converts between Text Strings and Index Tensors.
    """
    def __init__(self, alphabet="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        self.alphabet = '-' + alphabet 
        self.dict = {char: i for i, char in enumerate(self.alphabet)}

    def encode_list(self, text, K=7):
        all_result = []
        for item in text:
            result = []
            if isinstance(item, bytes): item = item.decode('utf-8')
            for i in range(K):
                if i < len(item): result.append(self.dict.get(item[i], 0))
                else: result.append(0) # Padding is 0
            all_result.append(result)
        return torch.LongTensor(all_result)

# ==============================================================================
# 2. CORE LOSSES
# ==============================================================================
class FocalLoss(nn.Module):
    """
    SOTA Classification Loss.
    Penalizes hard-to-classify examples more than easy ones.
    """
    def __init__(self, gamma=2.0, alpha=0.25, ignore_index=0): 
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=-1)
        probs = torch.clamp(probs, min=1e-5, max=1.0 - 1e-5)
        
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[-1]).float()
        pt = (probs * targets_one_hot).sum(dim=1)
        
        focal_weight = (1 - pt) ** self.gamma
        ce_loss = -torch.log(pt)
        loss = self.alpha * focal_weight * ce_loss
        
        if self.ignore_index >= 0:
            mask = targets != self.ignore_index
            if mask.sum() > 0:
                loss = loss[mask]
            else:
                return torch.tensor(0.0, device=logits.device)
            
        return loss.mean()

# ==============================================================================
# 2.5 SOTA FIX: TOPOLOGICAL REPULSION LOSS
# ==============================================================================
class TopologicalConfusionLoss(nn.Module):
    """
    Explicitly penalizes the network for guessing visually similar characters.
    Forces the latent space to build massive "valleys" between O/Q, M/N, 8/B.
    """
    def __init__(self, alphabet, ignore_index=0):
        super().__init__()
        self.dict = {char: i for i, char in enumerate(alphabet)}
        self.ignore_index = ignore_index
        num_classes = len(alphabet)
        
        # Build the (37, 37) Repulsion Matrix
        self.penalty_matrix = torch.zeros((num_classes, num_classes), dtype=torch.float32)
        
        # Map out the exact micro-topological failures from the logs
        confusions = {
            # The Loop Cluster
            'O': ['Q', 'D', '0', 'C'], 'Q': ['O', 'D', '0'], '0': ['O', 'Q', 'D'], 'C': ['O', 'G', 'E'], 'G': ['C', '6'],
            # The Numeric Curve Cluster
            'B': ['8'], '8': ['B', '9', '3', 'S', '0'], '3': ['8', '9'], '9': ['8', '3', '0'],
            '5': ['6', 'S'], '6': ['5', 'G'], 'S': ['5', '8'],
            # The Dense Stroke Cluster
            'M': ['N', 'H', 'W'], 'N': ['M', 'H'], 'H': ['M', 'N'],
            # The Diagonal Cluster
            'V': ['W', 'Y', 'U', 'X'], 'W': ['V', 'M'], 'Y': ['V', 'X'], 'X': ['Y', 'V'],
            # The Vertical Bar Cluster
            'I': ['1', 'T', 'L', 'J'], '1': ['I', 'T', 'L'], 'T': ['I', '1'], 'J': ['I', '1'],
            # Structural bleed
            'A': ['R'], 'R': ['A', 'P', 'B'], 'P': ['R', 'F', 'B'], 'F': ['P', 'E'], 'E': ['F', 'C']
        }
        
        for true_char, traps in confusions.items():
            if true_char in self.dict:
                t_idx = self.dict[true_char]
                for trap_char in traps:
                    if trap_char in self.dict:
                        trap_idx = self.dict[trap_char]
                        # Set a 1.0 penalty multiplier for this specific trap
                        self.penalty_matrix[t_idx, trap_idx] = 1.0 
                        
    def forward(self, logits, targets):
        device = logits.device
        probs = F.softmax(logits, dim=-1)
        
        # Fetch the active penalties for the true ground truth characters
        # Shape: (Batch*T, 37)
        active_mask = self.penalty_matrix.to(device)[targets]
        
        # Multiply predicted probabilities by the trap mask.
        # If the GT is 'Q', and the network predicts 'O' with 80% confidence, trap_probs = 0.8
        trap_probs = (probs * active_mask).sum(dim=1)
        
        # Mask out the padding index so we don't penalize sequence length variations
        mask = (targets != self.ignore_index).float()
        
        # Square the penalty to aggressively punish high-confidence wrong guesses
        return ((trap_probs ** 2) * mask).mean()

# ==============================================================================
# 3. COMPOSITE MODULE
# ==============================================================================
@register('SROCR_loss')
class SROCR_loss(nn.Module):
    def __init__(self, args=None, **kwargs):
        super().__init__()
        self.args = args or {}
        self.converter = strLabelConverter()
        
        # -- Core Losses --
        self.ocr_loss_fn = FocalLoss(gamma=2.0, ignore_index=0)
        self.confusion_loss = TopologicalConfusionLoss(self.converter.alphabet, ignore_index=0)
        
        # -- Weights --
        self.weights = {
            'ocr_weight': 1.0,
            'confusion_weight': 2.0, # Heavily weighted to force feature separation
        }

    def forward(self, logits=None, gt_strings=None, sr_img=None, hr_img=None, **kwargs):
        losses = {}
        total_loss = torch.tensor(0.0, device=logits.device if logits is not None else sr_img.device)
        
        # 1. OCR Loss (Focal + Topological Repulsion)
        if logits is not None and gt_strings is not None:
            if logits.dim() == 3: logits = logits.view(-1, logits.shape[-1])
            gt_ids = self.converter.encode_list(gt_strings).to(logits.device).view(-1)
            
            # Standard Focal Loss
            val_focal = self.ocr_loss_fn(logits, gt_ids)
            losses['L_Focal'] = val_focal
            total_loss += self.weights['ocr_weight'] * val_focal
            
            # Active Trap Repulsion Loss
            val_conf = self.confusion_loss(logits, gt_ids)
            losses['L_Trap'] = val_conf
            total_loss += self.weights['confusion_weight'] * val_conf

        return total_loss, {k: v.item() for k, v in losses.items()}, None, self.weights