"""
config_v2.py — Configuration for Option C: CLIP + Enhanced Transformer + DPP
=============================================================================
Two modes:
  MODE = 'clip'      → CLIP ViT-B/32 features (512-d), requires raw videos
  MODE = 'enhanced'  → GoogLeNet HDF5 features (1024-d) with stronger training

Switch MODE below and run train_v2.py.
"""

# ── Mode ──────────────────────────────────────────────────────────────────────
MODE = 'clip'          # 'clip' | 'enhanced'

# ── Dataset paths ─────────────────────────────────────────────────────────────
# GoogLeNet HDF5 (always needed for 'enhanced' mode; also used as fallback)
DATASET_PATHS_GOOGLENET = {
    'summe': 'data/summe.h5',
    'tvsum': 'data/tvsum.h5',
}

# CLIP HDF5 (produced by clip_feature_extractor.py)
DATASET_PATHS_CLIP = {
    'summe': 'data/summe_clip.h5',
    'tvsum': 'data/tvsum_clip.h5',
}

# Raw video directories (needed only for CLIP extraction)
RAW_VIDEO_DIRS = {
    'summe': 'raw_videos/summe/',   # folder containing .mp4 / .avi files
    'tvsum': 'raw_videos/tvsum/',   # folder containing .mp4 / .avi files
}

def get_dataset_path(dataset: str) -> str:
    if MODE == 'clip':
        return DATASET_PATHS_CLIP[dataset]
    return DATASET_PATHS_GOOGLENET[dataset]

# ── Feature dimensionality ────────────────────────────────────────────────────
FEATURE_DIM = 512 if MODE == 'clip' else 1024

# ── Transformer (v2 — deeper and wider) ──────────────────────────────────────
D_MODEL      = 512
N_HEADS      = 8
N_LAYERS     = 4          # v2: 4 layers (was 2) — stronger representation
FFN_DIM      = 2048
DROPOUT      = 0.1
PRE_LN       = True       # Pre-LayerNorm (more stable)

# ── Training (v2 — stronger schedule) ────────────────────────────────────────
N_FOLDS          = 5
EPOCHS           = 100        # v2: 100 (was 50)
LR               = 5e-5       # v2: lower LR for stability with deeper model
WEIGHT_DECAY     = 1e-4
WARMUP_EPOCHS    = 10         # v2: linear warmup
LABEL_SMOOTHING  = 0.1        # v2: smoother targets
GRAD_CLIP        = 1.0
SUMMARY_RATIO    = 0.15
BATCH_ACCUM      = 4          # v2: gradient accumulation (effective batch = 4)

# ── DPP ──────────────────────────────────────────────────────────────────────
DPP_LAMBDA   = 1.0

# ── Misc ─────────────────────────────────────────────────────────────────────
SEED         = 42
DEVICE       = 'cuda'
RESULTS_DIR  = 'results_v2'
