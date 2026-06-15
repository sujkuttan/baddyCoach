"""BST (Badminton Stroke-type Transformer) model architecture.

Based on the paper: "BST: Badminton Stroke-type Transformer for Skeleton-based
Action Recognition in Racket Sports" (CVPRW 2026)

Architecture: BST-CG (Clean Gate variant)
- Input: Pose sequences + Shuttle trajectory + Player positions
- TCN for feature extraction
- Transformer Encoder for temporal modeling
- Cross Transformer for pose-shuttle interaction
- Clean Gate for noise reduction
"""

import torch
import torch.nn as nn
import math


class TCN(nn.Module):
    """Temporal Convolutional Network."""
    
    def __init__(self, in_channels, channels, kernel_size, drop_p=0.3):
        super().__init__()
        layers = []
        prev_ch = in_channels
        for ch in channels:
            layers.append(nn.Conv1d(prev_ch, ch, kernel_size, padding=kernel_size // 2))
            layers.append(nn.BatchNorm1d(ch))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(drop_p))
            prev_ch = ch
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class FeedForward(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, in_dim, out_dim, hd_dim, drop_p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hd_dim),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(hd_dim, out_dim),
            nn.Dropout(drop_p),
        )
    
    def forward(self, x):
        return self.net(x)


class MLP(nn.Module):
    """Simple MLP."""
    
    def __init__(self, in_dim, out_dim=None, hd_dim=None, drop_p=0.3):
        super().__init__()
        if out_dim is None:
            out_dim = in_dim
        if hd_dim is None:
            hd_dim = in_dim * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hd_dim),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(hd_dim, out_dim),
            nn.Dropout(drop_p),
        )
    
    def forward(self, x):
        return self.net(x)


class MLP_Head(nn.Module):
    """MLP classification head."""
    
    def __init__(self, in_dim, out_dim, hd_dim, drop_p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hd_dim),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(hd_dim, out_dim),
        )
    
    def forward(self, x):
        return self.net(x)


class PositionalEncoding1D(nn.Module):
    """1D positional encoding."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer."""
    
    def __init__(self, d_model, d_head, n_head, hd_ff, drop_p=0.3):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=drop_p, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_model, hd_ff, drop_p)
    
    def forward(self, x, mask=None):
        residual = x
        x = self.layer_norm1(x)
        x, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = residual + x
        x = x + self.ff(self.layer_norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Stack of Transformer encoder layers."""
    
    def __init__(self, d_model, d_head, n_head, depth, hd_ff, drop_p=0.3):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, d_head, n_head, hd_ff, drop_p)
            for _ in range(depth)
        ])
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross attention."""
    
    def __init__(self, d_model, d_head, n_head, drop_p=0.3):
        super().__init__()
        d_cat = d_head * n_head
        self.h = n_head
        self.to_q = nn.Linear(d_model, d_cat, bias=False)
        self.to_kv = nn.Linear(d_model, d_cat * 2, bias=False)
        self.scale = d_head ** -0.5
        self.attend = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(drop_p))
        self.tail = nn.Sequential(
            nn.Linear(d_cat, d_model),
            nn.Dropout(drop_p)
        ) if n_head != 1 or d_cat != d_model else nn.Identity()
    
    def forward(self, x1, x2, mask=None):
        q = self.to_q(x1)
        kv = self.to_kv(x2)
        b, t, _ = q.shape
        
        q = q.view(b, t, self.h, -1).transpose(1, 2)
        k, v = kv.view(b, t, self.h, -1).chunk(2, dim=-1)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        
        dots = (q @ k.transpose(-1, -2)) * self.scale
        if mask is not None:
            dots = dots.masked_fill(mask.view(b, 1, 1, t) == 0, -torch.inf)
        
        att = self.attend(dots) @ v
        out = att.transpose(1, 2).reshape(b, t, -1)
        return self.tail(out)


class CrossTransformerLayer(nn.Module):
    """Cross Transformer layer for pose-shuttle interaction."""
    
    def __init__(self, d_model, d_head, n_head, hd_ff, drop_p=0.3):
        super().__init__()
        self.layer_norm1_x1 = nn.LayerNorm(d_model)
        self.layer_norm1_x2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadCrossAttention(d_model, d_head, n_head, drop_p)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_model, hd_ff, drop_p)
    
    def forward(self, x1, x2, mask=None):
        x1 = self.layer_norm1_x1(x1)
        x2 = self.layer_norm1_x2(x2)
        x = self.cross_attn(x1, x2, mask)
        x = x + self.ff(self.layer_norm2(x))
        return x


class BST_CG(nn.Module):
    """BST-CG: Badminton Stroke-type Transformer with Clean Gate.
    
    Args:
        in_dim: Input dimension per person (n_joints * 2 + n_bones * 2 for JnB_bone)
        seq_len: Sequence length (number of frames per clip)
        n_classes: Number of stroke classes
        d_model: Transformer hidden dimension
        d_head: Attention head dimension
        n_head: Number of attention heads
        depth_tem: Depth of temporal encoder
        depth_inter: Depth of interactional encoder
        drop_p: Dropout probability
        mlp_d_scale: MLP hidden dim scale factor
        tcn_kernel_size: TCN kernel size
    """
    
    def __init__(
        self,
        in_dim=72,       # (17 joints + 19 bones) * 2 coords
        seq_len=30,
        n_classes=25,
        d_model=100,
        d_head=128,
        n_head=6,
        depth_tem=2,
        depth_inter=1,
        drop_p=0.3,
        mlp_d_scale=4,
        tcn_kernel_size=5,
    ):
        super().__init__()
        
        # Pose TCN
        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)
        
        # Shuttle TCN
        self.tcn_shuttle = TCN(2, [d_model // 2, d_model], tcn_kernel_size, drop_p)
        
        # Temporal Transformer
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)
        
        # Cross Transformer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)
        
        # Interactional Transformer
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + seq_len, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)
        
        # Clean Gate
        self.mlp_clean = MLP(d_model, d_model, d_model, drop_p)
        
        # MLP Head
        self.mlp_head = MLP_Head(d_model * 3, n_classes, d_model * mlp_d_scale, drop_p)
        
        self.d_model = d_model
        self.seq_len = seq_len
        self._init_weights()
    
    def _init_weights(self):
        """Initialize positional encodings and weights."""
        # Positional encodings
        pe = torch.zeros(self.seq_len + 1, self.d_model)
        position = torch.arange(0, self.seq_len + 1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.embedding_tem.data.copy_(pe)
        
        pe_cross = torch.zeros(self.seq_len, self.d_model)
        position_cross = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1)
        pe_cross[:, 0::2] = torch.sin(position_cross * div_term[:self.d_model // 2])
        pe_cross[:, 1::2] = torch.cos(position_cross * div_term[:self.d_model // 2])
        self.embedding_cross.data.copy_(pe_cross)
        
        pe_inter = torch.zeros(self.seq_len + 1, self.d_model)
        pe_inter[:, 0::2] = torch.sin(position * div_term)
        pe_inter[:, 1::2] = torch.cos(position * div_term)
        self.embedding_inter.data.copy_(pe_inter)
        
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)
        
        self.apply(self._init_weights_recursive)
    
    def _init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)
    
    def forward(self, JnB, shuttle, video_len):
        """
        Args:
            JnB: (batch, seq_len, n_people, in_dim) - pose + bones
            shuttle: (batch, seq_len, 2) - shuttle positions
            video_len: (batch,) - actual sequence length
        Returns:
            logits: (batch, n_classes)
        """
        b, t, n, in_dim = JnB.shape
        
        # Process pose through TCN
        JnB = JnB.permute(0, 2, 3, 1).reshape(b * n, in_dim, t)
        JnB = self.tcn_pose(JnB)
        JnB = JnB.view(b, n, -1, t).transpose(-2, -1)
        
        # Process shuttle through TCN
        shuttle = shuttle.transpose(1, 2).contiguous()
        shuttle = self.tcn_shuttle(shuttle)
        shuttle = shuttle.unsqueeze(1).transpose(-2, -1)
        
        # Concatenate pose and shuttle
        x = torch.cat((JnB, shuttle), dim=1)  # (b, n+1, t, d_model)
        
        # Add class token and positional encoding
        class_token_tem = self.learned_token_tem.view(1, 1, -1).expand(b * n, -1, -1)
        x = x.view(b * n, t, -1)
        x = torch.cat((class_token_tem, x), dim=1) + self.embedding_tem
        
        # Create mask
        range_t = torch.arange(0, 1 + t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len_expanded = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len_expanded)
        mask_n = mask.repeat_interleave(n, dim=0)
        
        # Temporal encoding
        x = self.pre_dropout(x)
        x = self.encoder_tem(x, mask_n)
        x = x.view(b, n, 1 + t, -1)
        
        # Split into player 1, player 2, shuttle
        p1, p2, shuttle = x[:, 0], x[:, 1], x[:, 2]
        p1_cls, p2_cls, shuttle_cls = p1[:, 0], p2[:, 0], shuttle[:, 0]
        
        # Cross attention (pose-shuttle interaction)
        p1 = p1[:, 1:] + self.embedding_cross
        p2 = p2[:, 1:] + self.embedding_cross
        shuttle = shuttle[:, 1:] + self.embedding_cross
        cross_mask = mask[:, 1:]
        
        p1_shuttle = self.cross_trans(p1, shuttle, cross_mask)
        p2_shuttle = self.cross_trans(p2, shuttle, cross_mask)
        
        # Interactional encoding
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        p1_shuttle = torch.cat((class_token_inter, p1_shuttle), dim=1) + self.embedding_inter
        p2_shuttle = torch.cat((class_token_inter, p2_shuttle), dim=1) + self.embedding_inter
        
        p1_shuttle = self.encoder_inter(p1_shuttle, mask)
        p2_shuttle = self.encoder_inter(p2_shuttle, mask)
        
        p1_shuttle_cls = p1_shuttle[:, 0]
        p2_shuttle_cls = p2_shuttle[:, 0]
        
        # Clean Gate: remove noise from shuttle representation
        info_need_clean = torch.minimum(p1_shuttle_cls, p2_shuttle_cls)
        dirt = self.mlp_clean(info_need_clean)
        shuttle_cls = shuttle_cls - dirt
        
        # Combine representations
        p1_conclusion = p1_cls + p1_shuttle_cls
        p2_conclusion = p2_cls + p2_shuttle_cls
        
        x = torch.cat((p1_conclusion, p2_conclusion, shuttle_cls), dim=1)
        x = self.mlp_head(x)
        
        return x


# Class names for ShuttleSet merged 25 classes
SHUTTLESET_MERGED_CLASSES = [
    'unknown',           # 0
    'Top_net_shot',      # 1  放小球
    'Top_block',         # 2  擋小球
    'Top_smash',         # 3  殺球
    'Top_lift',          # 4  挑球
    'Top_clear',         # 5  長球
    'Top_drive',         # 6  平球
    'Top_drop',          # 7  切球
    'Top_push',          # 8  推球
    'Top_rush',          # 9  撲球
    'Top_cross_court',   # 10 勾球
    'Top_short_serve',   # 11 發短球
    'Top_long_serve',    # 12 發長球
    'Bottom_net_shot',   # 13
    'Bottom_block',      # 14
    'Bottom_smash',      # 15
    'Bottom_lift',       # 16
    'Bottom_clear',      # 17
    'Bottom_drive',      # 18
    'Bottom_drop',       # 19
    'Bottom_push',       # 20
    'Bottom_rush',       # 21
    'Bottom_cross_court',# 22
    'Bottom_short_serve',# 23
    'Bottom_long_serve', # 24
]

# Simplified class mapping for coaching (merge Top/Bottom)
COACH_STROKE_CLASSES = [
    'unknown',     # 0
    'net_shot',    # 1, 13
    'block',       # 2, 14
    'smash',       # 3, 15
    'lift',        # 4, 16
    'clear',       # 5, 17
    'drive',       # 6, 18
    'drop',        # 7, 19
    'push',        # 8, 20
    'rush',        # 9, 21
    'cross_court', # 10, 22
    'short_serve', # 11, 23
    'long_serve',  # 12, 24
]


def get_coach_class(pred_class_id):
    """Map ShuttleSet merged class ID to simplified coaching class."""
    if pred_class_id == 0:
        return 'unknown'
    elif 1 <= pred_class_id <= 12:
        return COACH_STROKE_CLASSES[pred_class_id]
    elif 13 <= pred_class_id <= 24:
        return COACH_STROKE_CLASSES[pred_class_id - 12]
    return 'unknown'
