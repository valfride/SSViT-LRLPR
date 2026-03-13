import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

# ==============================================================================
# 1. INDEPENDENT RECOGNITION HEAD (The "Expert")
# ==============================================================================
class RecognitionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_chars=7):
        super().__init__()
        
        # A. Learned Queries (The "Questions")
        # Instead of reading a Draft, the head asks: "What is in slot 1? Slot 2?..."
        # We learn 7 vectors, one for each character position.
        self.query_embed = nn.Parameter(torch.randn(1, num_chars, hidden_dim))
        
        # B. Transformer Decoder (The "Brain")
        # It matches the Queries (Positions) to the Keys/Values (Visual Features)
        # norm_first=True is critical for stability in FP16
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, 
            nhead=4, 
            dim_feedforward=1024, 
            dropout=0.1, 
            batch_first=True, 
            norm_first=True 
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        
        # C. Output Projector (The "Speaker")
        self.cls = nn.Linear(hidden_dim, num_classes)
        
        # Project input features to hidden dim if needed
        self.proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

    def forward(self, visual_feats):
        # visual_feats: [Batch, 7, Input_Dim]
        
        # 1. Expand Queries for Batch
        # [1, 7, Dim] -> [Batch, 7, Dim]
        b = visual_feats.size(0)
        queries = self.query_embed.repeat(b, 1, 1)
        
        # 2. Project Visuals
        # [Batch, 7, Dim]
        keys = self.proj(visual_feats)
        
        # 3. Attend (Decoder)
        # Query = "What is at Position X?"
        # Key/Value = Visual Features from BiLSTM
        # Output = "Here is the feature for Position X."
        decoded = self.decoder(tgt=queries, memory=keys)
        
        # 4. Classify
        logits = self.cls(decoded) # [Batch, 7, Num_Classes]
        return logits

# ==============================================================================
# 2. HELPER BLOCKS (Unchanged)
# ==============================================================================
class SELayer(nn.Module): # ... (Keep your MaxPool version) ...
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveMaxPool2d(1) 
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class RDB_Block(nn.Module): # ... (Keep Unchanged) ...
    def __init__(self, in_channels, growth_rate=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_rate, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels + growth_rate, growth_rate, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channels + 2*growth_rate, in_channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(in_channels)
    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        return self.se(x3) + x 

class VisualMixerBlock(nn.Module): # ... (Keep Unchanged) ...
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(dim, dim, kernel_size, padding=padding)
        self.norm1 = nn.InstanceNorm1d(dim, affine=True)
        self.act   = nn.ReLU(True)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size, padding=padding)
        self.norm2 = nn.InstanceNorm1d(dim, affine=True)
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return self.act(out + residual)

# ==============================================================================
# 3. MAIN CUSTOM OCR (Pure Ensemble)
# ==============================================================================
class CustomOCR(nn.Module):
    def __init__(self, input_shape=(128, 64, 192), num_classes=37, num_chars=7, d_model=256):
        super().__init__()
        self.num_classes = num_classes
        in_channels = input_shape[0]
        
        # --- PATH A: Shallow Features ---
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, 1, 1), 
            nn.InstanceNorm2d(128, affine=True), 
            nn.ReLU(True)
        )
        
        # --- PATH B: Deep Features ---
        self.deep_extractor = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1), nn.InstanceNorm2d(256, affine=True), nn.ReLU(True),
            RDB_Block(256), 
            nn.Conv2d(256, 512, 3, 1, 1), nn.InstanceNorm2d(512, affine=True), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)), 
            nn.Conv2d(512, 512, 3, 1, 1), nn.InstanceNorm2d(512, affine=True), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)), 
            nn.Conv2d(512, 512, kernel_size=3, stride=(2, 1), padding=1), 
            nn.InstanceNorm2d(512, affine=True), nn.ReLU(True) 
        )
        
        # Compression & Mixer
        self.neck_compress = nn.Linear(640 * 8, d_model) 
        self.visual_mixer = VisualMixerBlock(d_model, kernel_size=3)
        self.lstm = nn.LSTM(d_model, d_model // 2, num_layers=2, bidirectional=True, batch_first=True, dropout=0.5)
        
        # ======================================================
        # THE ENSEMBLE (N Identical Heads)
        # ======================================================
        self.num_heads = 4  # 4 Parallel Experts
        
        self.heads = nn.ModuleList([
            RecognitionHead(input_dim=d_model, hidden_dim=d_model, num_classes=num_classes)
            for _ in range(self.num_heads)
        ])

    def forward(self, x, text_labels=None, **kwargs):
        # 1. Feature Extraction (VSR Backbone)
        x_stem = self.stem(x) 
        x_deep = self.deep_extractor(x_stem) 
        x_stem_pooled = F.adaptive_avg_pool2d(x_stem, (8, 192))
        feat = torch.cat([x_deep, x_stem_pooled], dim=1) 
        
        # 2. Sequence Formation (Force FP32)
        with torch.cuda.amp.autocast(enabled=False):
            feat = feat.float() 
            b, c, h, w = feat.size()
            feat = feat.permute(0, 3, 1, 2).contiguous().view(b, w, c * h) 
            
            feat = self.neck_compress(feat) 
            feat = feat.permute(0, 2, 1)
            feat = self.visual_mixer(feat)
            feat = feat.permute(0, 2, 1) 
            
            self.lstm.flatten_parameters()
            context_feats, _ = self.lstm(feat) 
            
            # Pool to 7 chars [Batch, 7, Hidden]
            # This is the SHARED truth that all heads look at.
            char_feats = F.adaptive_avg_pool1d(context_feats.permute(0, 2, 1), 7).permute(0, 2, 1)
            
            # ==================================================
            # 3. PARALLEL PREDICTION
            # ==================================================
            all_stage_logits = []
            
            for head in self.heads:
                # A. Diversity Mechanism (Dropout)
                # Randomly mask 20% of the features so Head 1 sees something different than Head 2.
                # This prevents "Mode Collapse" (where all heads learn the exact same thing).
                diverse_input = F.dropout(char_feats, p=0.2, training=self.training)
                
                # B. Independent Prediction
                logits = head(diverse_input)
                all_stage_logits.append(logits)
            
            # 4. Consensus
            stacked_logits = torch.stack(all_stage_logits, dim=0)
            ensemble_logits = torch.mean(stacked_logits, dim=0)
    
            loss = torch.tensor(0.0, device=x.device)
            return {'logits': ensemble_logits, 'all_logits': all_stage_logits, 'loss': loss}