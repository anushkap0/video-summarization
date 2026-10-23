"""
config_v3.py — Unified configuration for the v3 video summarization pipeline
=============================================================================
Single source of truth. Import this in every v3 script.

QUICK START:
  1. Set BACKBONE = 'vitl14'  for best results (needs raw videos)
     Set BACKBONE = 'vitb32'  if already have ViT-B/32 features
     Set BACKBONE = 'googlenet' to use existing eccv16 HDF5 (no raw videos)

  2. Set RAW_VIDEO_DIRS if you have raw videos for CLIP extraction.
     Otherwise point DATASET_H5 directly to existing HDF5 files.

  3. Run: python main_v3.py --device cuda
"""

import os
import torch

# ─────────────────────────────────────────────────────────────────────────────
# BACKBONE SELECTION
# ─────────────────────────────────────────────────────────────────────────────
# 'vitl14'   → CLIP ViT-L/14 (768-d) ← BEST, targets 0.60–0.65 TVSum
# 'vitb32'   → CLIP ViT-B/32 (512-d) ← good, targets 0.54–0.59 TVSum
# 'googlenet'→ GoogLeNet pool5 (1024-d) ← no raw videos needed
BACKBONE = 'vitl14'

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE DIMENSIONS (auto-set from BACKBONE)
# ─────────────────────────────────────────────────────────────────────────────
_FEAT_DIMS = {
    'vitl14':    768,
    'vitb32':    512,
    'googlenet': 1024,
}
FEATURE_DIM = _FEAT_DIMS[BACKBONE]

# ─────────────────────────────────────────────────────────────────────────────
# PATHS — edit these to match your server layout
# ─────────────────────────────────────────────────────────────────────────────

# Raw video directories (needed only for CLIP feature extraction)
RAW_VIDEO_DIRS = {
    'tvsum': 'raw_videos/tvsum',
    'summe': 'raw_videos/summe',
}

# HDF5 feature files — used for training
# If you already have extracted features, point directly here.
DATASET_H5 = {
    'vitl14': {
        'tvsum': 'data/tvsum_clip_l14.h5',
        'summe': 'data/summe_clip_l14.h5',
    },
    'vitb32': {
        'tvsum': 'data/tvsum_clip_b32.h5',
        'summe': 'data/summe_clip_b32.h5',
    },
    'googlenet': {
        'tvsum': 'data/tvsum.h5',            # eccv16 HDF5 (already on server)
        'summe': 'data/summe.h5',
    },
}

# Reference HDF5 — used only to copy ground-truth annotations during extraction
# These are the original eccv16 files you already have
REFERENCE_H5 = {
    'tvsum': 'data/tvsum.h5',
    'summe': 'data/summe.h5',
}

# Results directory
RESULTS_DIR = 'results_v3'

# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
D_MODEL = 256
N_HEADS_LOCAL    = 4         # heads in windowed attention
N_HEADS_GLOBAL   = 8         # heads in CLS global attention
N_LOCAL_LAYERS = 1
FFN_DIM = 1024
DROPOUT = 0.25
WINDOW_SIZE = 128

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
EPOCHS = 200
WARMUP_EPOCHS = 5
LR = 8e-5
WEIGHT_DECAY     = 1e-4
GRAD_CLIP        = 1.0
BATCH_ACCUM      = 4         # gradient accumulation steps

# Loss weights
LABEL_SMOOTHING  = 0.1
RANK_WEIGHT = 0.2
CONTRAST_WEIGHT = 0.05

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
N_FOLDS          = 5
SEED             = 42
SUMMARY_RATIO    = 0.15      # select top 15% frames as summary

# DPP parameters
DPP_LAMBDA       = 1.0

# Test-Time Augmentation
TTA_N_AUG        = 3         # 1=off, 3=fwd+rev+noise

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

MODE = 'clip' if BACKBONE in ('vitl14', 'vitb32') else 'enhanced'


def get_dataset_path(dataset: str) -> str:
    return DATASET_H5[BACKBONE][dataset]


def get_h5_path(dataset: str) -> str:
    """Return the HDF5 path for the current BACKBONE and dataset."""
    return DATASET_H5[BACKBONE][dataset]


def get_clip_model_name() -> str:
    """Return the CLIP model string for clip.load()."""
    return {'vitl14': 'ViT-L/14', 'vitb32': 'ViT-B/32'}.get(BACKBONE, '')


def is_clip_backbone() -> bool:
    return BACKBONE in ('vitl14', 'vitb32')


def print_config():
    print(f'  BACKBONE     : {BACKBONE}  (feat_dim={FEATURE_DIM})')
    print(f'  D_MODEL      : {D_MODEL}')
    print(f'  N_LOCAL_LAYERS: {N_LOCAL_LAYERS}  WINDOW={WINDOW_SIZE}')
    print(f'  EPOCHS       : {EPOCHS}  WARMUP={WARMUP_EPOCHS}')
    print(f'  LR           : {LR}  WD={WEIGHT_DECAY}')
    print(f'  LOSS         : MSE + {RANK_WEIGHT}×rank + {CONTRAST_WEIGHT}×contrast')
    print(f'  TTA          : {TTA_N_AUG} augmentations')
    print(f'  N_FOLDS      : {N_FOLDS}')
    print(f'  DEVICE       : {DEVICE}')
