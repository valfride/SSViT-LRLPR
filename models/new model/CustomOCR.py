import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# ==============================================================================
# 1. HELPER BLOCKS (Unchanged)
# ==============================================================================
class SurgicalFocusBlock(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.GroupNorm(4, mip) 
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)

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
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_w * a_h

class LayoutScout(nn.Module):
    def __init__(self, in_channels=128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1) 
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, 2) 
        )
    def forward(self, x):
        return self.head(self.pool(x.detach()))

# ==============================================================================
# 2. THE NEW ViT EXPERT (Direct Latent -> Characters)
import torch
import torch.nn as nn
import torch.nn.functional as F

class ViT_Direct_OCR(nn.Module):
    def __init__(self, in_channels=256, d_model=256, num_chars=7, num_classes=37, num_layers=4):
        super().__init__()
        self.d_model = d_model

        # ==============================================================================
        # 1. [span_0](start_span)HYBRID STEM (The "Anti-Blur" Fix)[span_0](end_span)
        # ==============================================================================
        # Old: Naive 4x4 Patch Embedding (Destroys Edges)
        # New: 3-Layer CNN (Preserves Topology & Corners)
        self.patch_embed = nn.Sequential(
            # Layer 1: Detect Edges (Input: 256 -> 128)
            nn.Conv2d(in_channels, d_model // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(d_model // 2),
            nn.ReLU(True),
            
            # Layer 2: Downsample /2 (32x96 -> 16x48)
            nn.Conv2d(d_model // 2, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(True),
            
            # Layer 3: Overlapping Projection to Token Space (Stride 2 = 50% Overlap)
            # Result: 16x48 -> 8x24 Feature Map
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        )
        
        # Calculate new sequence length based on 2 downsampling steps (/2, /2 = /4 total)
        # Input: 32x96 -> Output: 8x24 = 192 tokens
        self.num_patches = (32 // 4) * (96 // 4) 

        # 2. Positional Embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)

        # 3. Learnable Character Queries (The "Questions")
        self.char_queries = nn.Parameter(torch.randn(1, num_chars, d_model) * 0.02)

        # 4. Transformer (The "Brain")
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=1024,
            activation='gelu',
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5. Output Head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

        # --- Mercosur Masking Buffers (Unchanged) ---
        self.register_buffer('mask_old', torch.zeros(1, 7, num_classes))
        self.register_buffer('mask_mercosur', torch.zeros(1, 7, num_classes))
        self.mask_old.fill_(-30.0); self.mask_mercosur.fill_(-30.0)
        
        nums = list(range(1, 11)); lets = list(range(11, 37))
        # Old Format: LLL-NNNN
        for i in [0, 1, 2]: self.mask_old[:, i, lets] = 0.0
        for i in [3, 4, 5, 6]: self.mask_old[:, i, nums] = 0.0
        
        # Mercosur: LLL-NLNN
        for i in [0, 1, 2, 4]: self.mask_mercosur[:, i, lets] = 0.0
        for i in [3, 5, 6]: self.mask_mercosur[:, i, nums] = 0.0

    def forward(self, x, layout_logits=None, refine_iters=1):
        B = x.size(0)

        # 1. Tokenize Image via Hybrid Stem
        # Input: [B, C, 32, 96] -> Output: [B, D, 8, 24]
        features = self.patch_embed(x)
        
        # Flatten: [B, D, 8, 24] -> [B, 192, D]
        vis_tokens = features.flatten(2).transpose(1, 2)

        # 2. Add Positional Embeddings
        # (Make sure shapes match! 192 tokens + 192 pos embeddings)
        if vis_tokens.shape[1] != self.pos_embed.shape[1]:
            # Safety interpolation if input size changes slightly
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2).reshape(1, self.d_model, 8, 24),
                size=(features.shape[2], features.shape[3]), mode='bilinear'
            ).flatten(2).transpose(1, 2)
            vis_tokens = vis_tokens + pos_embed
        else:
            vis_tokens = vis_tokens + self.pos_embed
            
        # ==========================================================
        # ⚡ NEW: ITERATIVE REFINEMENT LOOP
        # ==========================================================
        current_queries = self.char_queries.expand(B, -1, -1)
        final_tokens = None

        # Loop: [Initial Pass] -> [Refinement Pass 1] -> ...
        # (Reduced refinement passes for speed, usually 1 is enough)
        loops = refine_iters + 1
        
        for i in range(loops):
            # Combine: [Queries (7), Visual Tokens (192)]
            tokens = torch.cat([current_queries, vis_tokens], dim=1)
            
            # Run Transformer
            out_tokens = self.transformer(tokens)
            
            # Extract answers (first 7 tokens)
            char_tokens = out_tokens[:, :7]
            
            # FEEDBACK: The answers become the questions for the next round
            if i < loops - 1:
                current_queries = char_tokens
            
            final_tokens = char_tokens

        # 5. Output Head
        logits = self.head(final_tokens)

        # 6. Apply Layout Masks (Explicit Layout Gating)
        if layout_logits is not None:
            is_mercosur = layout_logits.argmax(dim=-1).bool()
            batch_mask = self.mask_old.repeat(B, 1, 1)
            
            # If any in batch are Mercosur, swap their masks
            if is_mercosur.any():
                # We need to ensure we only select the rows where is_mercosur is True
                # Mask shape is [B, 7, 37]. We copy from mask_mercosur [1, 7, 37]
                batch_mask[is_mercosur] = self.mask_mercosur.repeat(is_mercosur.sum(), 1, 1)
            
            logits = logits + batch_mask

        return logits, final_tokens


# ==============================================================================
# 3. TPS MODULE (Kept for rectification)
# ==============================================================================
class TPS_SpatialTransformerNetwork(nn.Module):
    def __init__(self, F_I=20, F_O=(32, 96), F_N=20, I_channel_num=3):
        super(TPS_SpatialTransformerNetwork, self).__init__()
        self.F_N = F_N
        self.F_O = F_O
        self.LocalizationNetwork = LocalizationNetwork(self.F_N, I_channel_num)
        self.GridGenerator = TPS_GridGen(self.F_O, self.F_N)

    def forward(self, x):
        theta = self.LocalizationNetwork(x)
        inv_index = self.GridGenerator(theta)
        return F.grid_sample(x, inv_index, padding_mode='border', align_corners=True), theta

class LocalizationNetwork(nn.Module):
    def __init__(self, F_N, I_channel_num):
        super(LocalizationNetwork, self).__init__()
        self.F_N = F_N
        self.conv = nn.Sequential(
            nn.Conv2d(I_channel_num, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.localization_fc2 = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(True),
            nn.Linear(256, F_N * 2)
        )
        self.localization_fc2[2].weight.data.fill_(0)
        
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F_N / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F_N / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F_N / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        self.localization_fc2[2].bias.data = torch.from_numpy(initial_bias).float().view(-1)

    def forward(self, x):
        features = self.conv(x).view(-1, 512)
        return self.localization_fc2(features).view(-1, self.F_N, 2)

class TPS_GridGen(nn.Module):
    def __init__(self, target_size, target_control_points):
        super(TPS_GridGen, self).__init__()
        self.N = target_control_points
        self.target_height, self.target_width = target_size
        
        target_control_partial_repr = self.compute_partial_repr(
            torch.tensor(np.concatenate([
                np.stack([np.linspace(-1.0, 1.0, int(self.N / 2)), np.linspace(-1.0, -1.0, int(self.N / 2))], axis=1),
                np.stack([np.linspace(-1.0, 1.0, int(self.N / 2)), np.linspace(1.0, 1.0, int(self.N / 2))], axis=1)
            ], axis=0), dtype=torch.float32)
        )
        self.register_buffer('target_control_partial_repr', target_control_partial_repr)

    def compute_partial_repr(self, input_points, control_points=None):
        N = input_points.size(0)
        M = control_points.size(0) if control_points is not None else N
        pairwise_diff = input_points.view(N, 1, 2) - (control_points if control_points is not None else input_points).view(1, M, 2)
        pairwise_diff_square = pairwise_diff * pairwise_diff
        pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
        repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist + 1e-6)
        
        if N == M:
            mask = torch.eye(N, dtype=torch.bool, device=input_points.device)
            repr_matrix.masked_fill_(mask, 0)
        return repr_matrix

    def forward(self, source_control_points):
        B = source_control_points.size(0)
        if not hasattr(self, 'inv_delta_c'):
             N = self.N
             C = torch.tensor(np.concatenate([
                np.stack([np.linspace(-1.0, 1.0, int(N / 2)), np.linspace(-1.0, -1.0, int(N / 2))], axis=1),
                np.stack([np.linspace(-1.0, 1.0, int(N / 2)), np.linspace(1.0, 1.0, int(N / 2))], axis=1)
            ], axis=0), dtype=torch.float32, device=source_control_points.device)
             
             L = torch.zeros(N + 3, N + 3, device=source_control_points.device)
             L[:N, :N] = self.compute_partial_repr(C, C)
             L[:N, N] = 1
             L[:N, N+1:] = C
             L[N, :N] = 1
             L[N+1:, :N] = C.t()
             self.register_buffer('inv_delta_c', torch.inverse(L))

        right_mat = torch.cat([source_control_points, torch.zeros(B, 3, 2, device=source_control_points.device)], dim=1) 
        W = torch.matmul(self.inv_delta_c, right_mat) 
        
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, self.target_height), 
            torch.linspace(-1, 1, self.target_width),
            indexing='ij'
        )
        target_grid = torch.stack([grid_x, grid_y], dim=2).view(-1, 2).to(source_control_points.device)
        
        batch_dist = self.compute_partial_repr(target_grid, self.target_control_partial_repr.new_tensor(np.concatenate([
                np.stack([np.linspace(-1.0, 1.0, int(self.N / 2)), np.linspace(-1.0, -1.0, int(self.N / 2))], axis=1),
                np.stack([np.linspace(-1.0, 1.0, int(self.N / 2)), np.linspace(1.0, 1.0, int(self.N / 2))], axis=1)
            ], axis=0)))
        
        target_grid_aug = torch.cat([torch.ones(target_grid.size(0), 1, device=source_control_points.device), target_grid], dim=1) 
        batch_grid_aug = torch.cat([batch_dist, target_grid_aug], dim=1)
        output_grid = torch.matmul(batch_grid_aug, W) 
        return output_grid.view(B, self.target_height, self.target_width, 2)

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(d_model, height, width)
        d_y = d_model // 2
        d_x = d_model - d_y
        div_term_y = torch.exp(torch.arange(0, d_y, 2).float() * (-math.log(10000.0) / d_y))
        div_term_x = torch.exp(torch.arange(0, d_x, 2).float() * (-math.log(10000.0) / d_x))
        pos_y = torch.arange(0, height).unsqueeze(1)
        pe[0:d_y:2, :, :] = torch.sin(pos_y * div_term_y).transpose(0, 1).unsqueeze(-1).repeat(1, 1, width)
        pe[1:d_y:2, :, :] = torch.cos(pos_y * div_term_y).transpose(0, 1).unsqueeze(-1).repeat(1, 1, width)
        pos_x = torch.arange(0, width).unsqueeze(1)
        pe[d_y::2, :, :] = torch.sin(pos_x * div_term_x).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_y+1::2, :, :] = torch.cos(pos_x * div_term_x).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe)

# ==============================================================================
# 4. CUSTOM OCR (Final Assembly)
# ==============================================================================
class CustomOCR(nn.Module):
    def __init__(self, input_shape=(128, 32, 96), num_classes=37, num_chars=7, d_model=256):
        super().__init__()
        # ... (Init logic remains the same) ...
        self.num_classes = num_classes
        in_channels = input_shape[0]
        
        self.stn = TPS_SpatialTransformerNetwork(F_I=20, F_O=(32, 96), F_N=20, I_channel_num=in_channels)
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, 1, 1), nn.GroupNorm(8, 128), nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)), 
            nn.GroupNorm(8, 256), nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)), 
            nn.GroupNorm(8, 256), nn.ReLU(True)
        )
        
        self.layout_scout = LayoutScout(in_channels=256)
        self.pos_encoder = PositionalEncoding2D(256, 32, 96)
        self.surgical_focus = SurgicalFocusBlock(in_channels=256, reduction=8)
        
        self.vit_expert = ViT_Direct_OCR(in_channels=256, d_model=d_model, num_chars=num_chars, num_classes=num_classes)
        
        self.projector = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(True), nn.Linear(128, 128) 
        )

    def forward(self, x, **kwargs):
        # 1. Rectify
        x_rectified, theta = self.stn(x) 
        
        # 2. Extract Features
        feat = self.stem(x_rectified) 
        layout_logits = self.layout_scout(feat)
        feat = self.pos_encoder(feat)
        feat = self.surgical_focus(feat)
        
        # 3. ViT Expert (Randomized Refinement)
        if self.training:
            # Randomly choose 1, 2, or 3 loops. 
            # Weighted towards 1 to stabilize the "First Glance".
            iters = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
        else:
            # Validation: Always use the best capacity
            iters = 2
        
        logits, char_tokens = self.vit_expert(feat, layout_logits, refine_iters=iters)
        
        # 4. Project
        z_vector = F.normalize(self.projector(char_tokens), p=2, dim=-1)
        
        return {
            'logits': logits,         
            'layout_logits': layout_logits,
            'features': char_tokens, 
            'z_vector': z_vector,
            'theta': theta,
            'crops': None 
        }
