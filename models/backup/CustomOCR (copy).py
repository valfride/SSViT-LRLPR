import torch
import torch.nn as nn
from torch.cuda.amp import autocast

class CustomOCR(nn.Module):
    def __init__(self, input_shape=(128, 64, 192), num_classes=37, num_chars=7, d_model=None):
        super().__init__()
        in_channels = input_shape[0]
        
        # Block 1: Feature Refinement (Based on your shared logic)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) # 64x192 -> 32x96
        )

        # Block 2: Deep Character Awareness
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) # 32x96 -> 16x48
        )

        # Block 3: Final Distillation
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 16)) # Collapse height, keep 16 horizontal steps
        )

        # Multi-Task Heads (Replacing the Transformer queries)
        # Instead of 1 head, we have 7 independent heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512 * 16, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_classes)
            ) for _ in range(num_chars)
        ])

    @autocast(enabled=False)
    def forward(self, x, **kwargs):
        B = x.shape[0]
        x = x.float()

        # Visual Pathway
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x) # Shape: [B, 512, 1, 16]
        
        x = x.view(B, -1) # Flatten to [B, 512 * 16]

        # Multi-Head Classification
        logits = [head(x) for head in self.heads] # List of 7 tensors of shape [B, 37]
        
        # Stack them to match your existing (B, 7, 37) loss format
        return torch.stack(logits, dim=1)