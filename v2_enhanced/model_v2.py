"""
model_v2.py — Enhanced Transformer Scorer for video summarization
=================================================================
Improvements over model.py (v1):
  1. Works with any feature dimension (512-d CLIP or 1024-d GoogLeNet)
  2. 4 Transformer layers (was 2) — stronger representation capacity
  3. Pre-LayerNorm (more stable training with deeper networks)
  4. Learned positional encoding (replaces fixed sinusoidal)
  5. Residual quality head (two-branch scorer for robustness)
  6. Attention dropout separate from residual dropout

Architecture:
    Input  (T, D)              ← CLIP 512-d or GoogLeNet 1024-d
      ↓
    Linear(D → 512) + LayerNorm
      ↓
    Learned Positional Encoding (max 10,000 frames)
      ↓
    TransformerEncoder × N_LAYERS (pre-LN, 8 heads, FFN=2048)
      ↓
    Frame-level scorer:
      Linear(512→256) → GELU → Dropout → Linear(256→1) → Sigmoid
      ↓
    Output (T,)  ← importance score in [0, 1]
"""

import math
import torch
import torch.nn as nn
import config_v2 as C


class LearnedPositionalEncoding(nn.Module):
    """Learned absolute positional embedding (supports up to max_len frames)."""

    def __init__(self, d_model: int, max_len: int = 10_000, dropout: float = 0.1):
        super().__init__()
        self.dropout  = nn.Dropout(dropout)
        self.embedding = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (T, B, D)"""
        T = x.size(0)
        positions = torch.arange(T, device=x.device)           # (T,)
        pos_emb   = self.embedding(positions).unsqueeze(1)      # (T, 1, D)
        return self.dropout(x + pos_emb)


class PreLNTransformerLayer(nn.Module):
    """
    Transformer encoder layer with Pre-LayerNorm (more stable than Post-LN
    for deeper networks; see 'On Layer Normalization in the Transformer', 2020).
    """

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int,
                 dropout: float = 0.1, attn_dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, n_heads,
            dropout=attn_dropout, batch_first=False)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (T, B, D)"""
        # Self-attention with pre-LN
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        # FFN with pre-LN
        h = self.norm2(x)
        h = self.ffn(h)
        return x + h


class TransformerScorerV2(nn.Module):
    """
    Enhanced Transformer scorer.
    Compatible with CLIP (512-d) and GoogLeNet (1024-d) features.
    """

    def __init__(
        self,
        feat_dim:    int   = C.FEATURE_DIM,
        d_model:     int   = C.D_MODEL,
        n_heads:     int   = C.N_HEADS,
        n_layers:    int   = C.N_LAYERS,
        ffn_dim:     int   = C.FFN_DIM,
        dropout:     float = C.DROPOUT,
        max_len:     int   = 10_000,
    ):
        super().__init__()

        # ── Input projection ─────────────────────────────────────────────────
        # If feat_dim == d_model (e.g. CLIP 512→512), still project for
        # parameter parity and normalisation control.
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # ── Positional encoding ───────────────────────────────────────────────
        self.pos_enc = LearnedPositionalEncoding(d_model, max_len, dropout)

        # ── Transformer layers ────────────────────────────────────────────────
        self.layers = nn.ModuleList([
            PreLNTransformerLayer(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # ── Scoring head ─────────────────────────────────────────────────────
        self.scorer = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (T, D) feature tensor for one video

        Returns:
            scores : (T,) importance scores in [0, 1]
        """
        # Project and add batch dim for Transformer: (T, D) → (T, 1, d_model)
        x = self.input_proj(x).unsqueeze(1)   # (T, 1, d_model)
        x = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)                 # (T, 1, d_model)
        x = x.squeeze(1)                       # (T, d_model)
        scores = self.scorer(x).squeeze(-1)    # (T,)
        return scores


# ── Parameter count helper ───────────────────────────────────────────────────
def count_params(model: nn.Module) -> str:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f'{n/1e6:.2f}M'


if __name__ == '__main__':
    import config_v2 as C
    print(f'Mode: {C.MODE}  |  feat_dim: {C.FEATURE_DIM}')
    model = TransformerScorerV2()
    print(f'TransformerScorerV2  params: {count_params(model)}')

    # Forward pass test
    T, D = 300, C.FEATURE_DIM
    x = torch.randn(T, D)
    scores = model(x)
    print(f'Input: ({T}, {D})  →  scores: {scores.shape}  '
          f'[{scores.min():.3f}, {scores.max():.3f}]')
    assert scores.shape == (T,), 'Shape mismatch!'
    print('model_v2.py: forward pass OK')
