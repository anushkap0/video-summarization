"""
Central configuration for the improved video summarization pipeline.
Edit the paths here to match your environment before running.
"""

# ── Dataset paths ────────────────────────────────────────────────────────────
# Each dataset should be an HDF5 file in the standard format used by
# DR-DSN / VASNet papers (keys: 'video_name', 'features', 'gtscore',
# 'gtsummary', 'n_frames', 'picks', 'n_steps', 'change_points',
# 'n_frame_per_seg', 'user_summary').
# Public download: https://github.com/KaiyangZhou/pytorch-vsumm-reinforce
DATASET_PATHS = {
    'summe':  'data/summe.h5',
    'tvsum':  'data/tvsum.h5',
    'vsumm':  'data/vsumm.h5',
}

# ── Feature extraction ───────────────────────────────────────────────────────
BACKBONE         = 'googlenet'      # 'googlenet' (matches trained checkpoints)
                                    # 'resnet50' | 'efficientnet_b3' only if retraining
FRAME_RATE       = 2                # fps used during feature extraction

# Feature dimensionality — MUST match your HDF5 files AND trained checkpoints:
#   eccv16_dataset_*_google_pool5.h5  →  1024  ← standard download, trained checkpoints
#   Self-extracted with ResNet50      →  2048  ← only if you retrain
#   Self-extracted with EfficientNetB3→  1536  ← only if you retrain
FEATURE_DIM      = 1024             # GoogLeNet pool5 — do not change unless retraining

# ── Transformer scorer ───────────────────────────────────────────────────────
D_MODEL          = 512              # projection dimension
N_HEADS          = 8                # attention heads
N_LAYERS         = 2               # transformer encoder layers
DROPOUT          = 0.1

# ── Training ─────────────────────────────────────────────────────────────────
N_FOLDS          = 5               # cross-validation folds
EPOCHS           = 50
LR               = 1e-4
WEIGHT_DECAY     = 1e-4
SUMMARY_RATIO    = 0.15            # target summary length (fraction of video)

# ── DPP diversity sampling ───────────────────────────────────────────────────
DPP_LAMBDA       = 1.0             # quality-diversity trade-off (higher → more quality)

# ── Misc ─────────────────────────────────────────────────────────────────────
SEED             = 42
DEVICE           = 'cuda'          # 'cuda' | 'cpu'
