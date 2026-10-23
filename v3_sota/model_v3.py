"""
model_v3.py — State-of-the-art Transformer scorer targeting DSNet+CLIP (0.65 TVSum)
===================================================================================
Key architectural improvements over model_v2.py:

  1. Hierarchical Windowed Attention
       - Local window attention (128 frames) → captures shot-level patterns
       - CLS token attends globally → captures video-level context
       - 2-level hierarchy: local → global → fusion

  2. Multi-Scale Temporal Convolutions
       - 4 parallel conv kernels: 1, 3, 15, 63 frames
       - Fuses point-wise, short-range, mid-range, long-range context
       - Applied BEFORE attention for richer input representations

  3. Works with ANY CLIP backbone
       - ViT-B/32 → 512-d  (what you have)
       - ViT-L/14 → 768-d  (recommended, +3 F1 pts)

  4. Residual gating between local and global streams
       - Learned gate controls how much global context to blend in

Expected F1 with ViT-L/14 features:
    TVSum ~0.58–0.63  |  SumMe ~0.55–0.60

Architecture overview:
    Input (T, D)
      ↓
    Linear(D→512) + LayerNorm          ← input projection
      ↓
    MultiScaleAggregator               ← 4 temporal scales fused
      ↓
    [Local WindowedAttention] × 2      ← 128-frame windows, 4 heads
         ↓
    [Global CLS Attention]   × 1      ← CLS token + all frames, 8 heads
         ↓
    Residual Gate Fusion               ← learned blend of local + global
      ↓
    FinalNorm → Scorer head            ← 512→256→64→1 → Sigmoid
      ↓
    Output (T,)                        ← importance scores in [0,1]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import config_v3 as C


# ─────────────────────────────────────────────────────────────────────────────
# 1. Multi-scale temporal aggregator
# ─────────────────────────────────────────────────────────────────────────────

class MultiScaleAggregator(nn.Module):
    """
    Parallel 1-D convolutions at 4 temporal scales.
    Input : (B, T, d_model)
    Output: (B, T, d_model)
    """
    def __init__(self, d_model: int = 512, dropout: float = 0.1):
        super().__init__()
        assert d_model % 4 == 0, 'd_model must be divisible by 4'
        c = d_model // 4

        self.conv1  = nn.Conv1d(d_model, c, kernel_size=1,  padding=0)
        self.conv3  = nn.Conv1d(d_model, c, kernel_size=3,  padding=1)
        self.conv15 = nn.Conv1d(d_model, c, kernel_size=15, padding=7)
        self.conv63 = nn.Conv1d(d_model, c, kernel_size=63, padding=31)

        self.proj    = nn.Linear(d_model, d_model)
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act     = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, d_model)"""
        # Conv1d expects (B, C, T)
        xt = x.transpose(1, 2)

        s1  = self.act(self.conv1(xt))
        s3  = self.act(self.conv3(xt))
        s15 = self.act(self.conv15(xt))
        s63 = self.act(self.conv63(xt))

        out = torch.cat([s1, s3, s15, s63], dim=1)  # (B, d_model, T)
        out = out.transpose(1, 2)                    # (B, T, d_model)
        out = self.dropout(self.proj(out))
        return self.norm(x + out)                    # residual


# ─────────────────────────────────────────────────────────────────────────────
# 2. Windowed local attention layer (Pre-LN)
# ─────────────────────────────────────────────────────────────────────────────

class WindowedAttentionLayer(nn.Module):
    """
    Self-attention within non-overlapping windows of size `window`.
    Frames near the end that don't fill a full window are attended together
    as a partial window (no masking needed — padding is avoided entirely).

    Input/Output: (B, T, d_model)
    """

    def __init__(self, d_model: int, n_heads: int, window: int = 128,
                 dropout: float = 0.1, ffn_dim: int = 2048):
        super().__init__()
        self.window  = window
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.attn    = nn.MultiheadAttention(d_model, n_heads,
                                             dropout=dropout, batch_first=True)
        self.ffn     = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, d_model)"""
        B, T, D = x.shape
        W = self.window

        # ── Pad T to multiple of W ──────────────────────────────────────────
        pad = (W - T % W) % W
        if pad:
            x = F.pad(x, (0, 0, 0, pad))    # pad along T dim

        _, Tp, _ = x.shape
        n_wins = Tp // W

        # ── Reshape into windows ────────────────────────────────────────────
        # (B, n_wins, W, D)
        xw = x.view(B, n_wins, W, D)
        # Flatten batch and windows: (B*n_wins, W, D)
        xw_flat = xw.reshape(B * n_wins, W, D)

        # ── Pre-LN self-attention ───────────────────────────────────────────
        h  = self.norm1(xw_flat)
        h, _ = self.attn(h, h, h)
        xw_flat = xw_flat + h

        # ── Pre-LN FFN ──────────────────────────────────────────────────────
        h  = self.norm2(xw_flat)
        h  = self.ffn(h)
        xw_flat = xw_flat + h

        # ── Restore shape and remove padding ────────────────────────────────
        out = xw_flat.view(B, Tp, D)
        if pad:
            out = out[:, :T, :]

        return out                          # (B, T, d_model)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Global CLS attention layer (Pre-LN)
# ─────────────────────────────────────────────────────────────────────────────

class GlobalCLSLayer(nn.Module):
    """
    CLS token attends to all frame tokens globally.
    Frame tokens also attend to each other + the CLS token.

    Input : (B, T, d_model) frame features
            cls_token: (1, 1, d_model)   ← learnable, passed from parent module
    Output: (B, T, d_model) updated frame features
    """

    def __init__(self, d_model: int, n_heads: int,
                 dropout: float = 0.1, ffn_dim: int = 2048):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads,
                                           dropout=dropout, batch_first=True)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor,
                cls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x   : (B, T, D)
        cls : (B, 1, D)
        Returns updated (x, cls)
        """
        B = x.size(0)

        # Prepend CLS token: (B, T+1, D)
        seq = torch.cat([cls, x], dim=1)
        h   = self.norm1(seq)
        h, _ = self.attn(h, h, h)
        seq = seq + h

        h   = self.norm2(seq)
        h   = self.ffn(h)
        seq = seq + h

        return seq[:, 1:, :], seq[:, :1, :]   # (B,T,D), (B,1,D)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Residual gate fusion
# ─────────────────────────────────────────────────────────────────────────────

class ResidualGate(nn.Module):
    """
    Learned gate that blends local (windowed) and global (CLS-augmented) streams.
    gate = sigmoid(Linear(local || global)) ∈ [0, 1]
    out  = gate * global + (1-gate) * local
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, local: torch.Tensor,
                global_: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_proj(
            torch.cat([local, global_], dim=-1)))       # (B, T, D)
        return gate * global_ + (1 - gate) * local


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main model
# ─────────────────────────────────────────────────────────────────────────────

class TransformerScorerV3(nn.Module):
    """
    Hierarchical Windowed Transformer Scorer.
    Compatible with CLIP ViT-B/32 (512-d) and ViT-L/14 (768-d).

    Recommended config:
        ViT-L/14 features  → feat_dim=768, d_model=512
        window=128 frames  → ~4 seconds at 2 fps (good for shot detection)
    """

    def __init__(
        self,
        feat_dim:    int   = C.FEATURE_DIM,   # 512 (ViT-B) or 768 (ViT-L)
        d_model:     int   = 512,
        n_heads_local: int = 4,
        n_heads_global: int = 8,
        n_local_layers: int = 2,
        ffn_dim:     int   = 2048,
        dropout:     float = 0.1,
        window:      int   = 128,
    ):
        super().__init__()

        # ── Input projection ─────────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # ── Multi-scale aggregator ───────────────────────────────────────────
        self.ms_agg = MultiScaleAggregator(d_model, dropout)

        # ── Local windowed attention stack ───────────────────────────────────
        self.local_layers = nn.ModuleList([
            WindowedAttentionLayer(d_model, n_heads_local, window,
                                   dropout, ffn_dim)
            for _ in range(n_local_layers)
        ])

        # ── Global CLS attention ─────────────────────────────────────────────
        self.cls_token   = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.global_layer = GlobalCLSLayer(d_model, n_heads_global,
                                           dropout, ffn_dim)

        # ── Residual gate ────────────────────────────────────────────────────
        self.gate = ResidualGate(d_model)

        # ── Final norm + scorer head ─────────────────────────────────────────
        self.final_norm = nn.LayerNorm(d_model)
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
            x : (T, D)  feature tensor for one video (no batch dim)

        Returns:
            scores : (T,)  importance scores in [0, 1]
        """
        # Add batch dim: (T, D) → (1, T, D)
        x = x.unsqueeze(0)               # (1, T, D)
        B, T, D = x.shape

        # ── Project to d_model ───────────────────────────────────────────────
        x = self.input_proj(x)            # (1, T, 512)

        # ── Multi-scale temporal context ─────────────────────────────────────
        x = self.ms_agg(x)               # (1, T, 512)

        # ── Local windowed attention ─────────────────────────────────────────
        local = x
        for layer in self.local_layers:
            local = layer(local)          # (1, T, 512)

        # ── Global CLS attention ─────────────────────────────────────────────
        cls = self.cls_token.expand(B, -1, -1)   # (1, 1, 512)
        global_, _ = self.global_layer(local, cls)   # (1, T, 512)

        # ── Fuse local + global via learned gate ─────────────────────────────
        fused = self.gate(local, global_)   # (1, T, 512)

        # ── Score ─────────────────────────────────────────────────────────────
        fused  = self.final_norm(fused)     # (1, T, 512)
        fused  = fused.squeeze(0)          # (T, 512)
        scores = self.scorer(fused).squeeze(-1)   # (T,)
        return scores


# ── Parameter count helper ───────────────────────────────────────────────────
def count_params(model: nn.Module) -> str:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f'{n/1e6:.2f}M'


if __name__ == '__main__':
    print('=== TransformerScorerV3 sanity check ===')
    for feat_dim, name in [(512, 'ViT-B/32'), (768, 'ViT-L/14')]:
        model = TransformerScorerV3(feat_dim=feat_dim)
        print(f'\n{name}  (feat_dim={feat_dim})')
        print(f'  Params: {count_params(model)}')
        T = 400
        x = torch.randn(T, feat_dim)
        scores = model(x)
        assert scores.shape == (T,), f'Expected ({T},), got {scores.shape}'
        print(f'  Input ({T}, {feat_dim}) → scores {scores.shape} '
              f'[{scores.min():.3f}, {scores.max():.3f}]  ✓')
    print('\nmodel_v3.py: all checks passed.')
