"""
Step 2 – Transformer Importance Scorer
========================================
Architecture:
    Linear projection  (D → d_model)
    Positional encoding
    N × TransformerEncoderLayer   (self-attention + FFN)
    Linear head        (d_model → 1)  → sigmoid → importance score per frame

This is essentially the VASNet architecture (Fajtl et al., 2018) with the
addition of sinusoidal positional encoding and configurable depth.
"""

import math
import torch
import torch.nn as nn
import config as C


class SinusoidalPE(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 10_000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                        * (-math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerScorer(nn.Module):
    """
    Maps a sequence of frame features to per-frame importance scores in [0, 1].

    Input  : (B, T, feat_dim)
    Output : (B, T)   — importance scores
    """

    def __init__(self,
                 feat_dim:  int = C.FEATURE_DIM,
                 d_model:   int = C.D_MODEL,
                 n_heads:   int = C.N_HEADS,
                 n_layers:  int = C.N_LAYERS,
                 dropout:   float = C.DROPOUT):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_enc = SinusoidalPE(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,         # pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers,
                                                  enable_nested_tensor=False)

        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, feat_dim)
        Returns:
            scores: (B, T)
        """
        x = self.input_proj(x)          # (B, T, d_model)
        x = self.pos_enc(x)             # add positional encoding
        x = self.transformer(x)         # (B, T, d_model)
        scores = self.score_head(x)     # (B, T, 1)
        return scores.squeeze(-1)       # (B, T)


# ── Quick sanity check ───────────────────────────────────────────────────────
if __name__ == '__main__':
    model = TransformerScorer()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'TransformerScorer  parameters: {n_params:,}')

    dummy = torch.randn(1, 300, C.FEATURE_DIM)   # 1 video, 300 frames
    out   = model(dummy)
    print(f'Input shape:  {dummy.shape}')
    print(f'Output shape: {out.shape}')
    print(f'Score range:  [{out.min().item():.4f}, {out.max().item():.4f}]')
