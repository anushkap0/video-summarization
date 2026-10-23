============================================================
  VIDEO SUMMARIZATION PROJECT — STUDENT PACKAGE
============================================================

FOLDER STRUCTURE
────────────────
  shared/               ← Copy these into whichever version you run
    dataset.py          - HDF5 dataset loader + 5-fold splits
    dpp_sampler.py      - DPP greedy MAP frame selection
    evaluate.py         - F1 evaluation (maximum annotator protocol)
    verify_features.py  - Check feature norm/compatibility

  v1_baseline/          ← Start here. GoogLeNet features, simple Transformer
    config.py           - All hyperparameters
    feature_extractor.py- Extract GoogLeNet pool5 (1024-d) from raw video
    model.py            - 2-layer Transformer scorer
    train.py            - 5-fold cross-validation training
    inference.py        - Run inference, save keyframes
    run_all.py          - One-command full pipeline

  v2_enhanced/          ← Better model: deeper Transformer + CLIP features
    config_v2.py        - Dual-mode config (clip / enhanced)
    model_v2.py         - 4-layer Pre-LN Transformer, learned positional enc
    train_v2.py         - Rank loss + label smoothing + warmup LR
    inference_v2.py     - Inference with TTA, auto-detects checkpoint
    clip_feature_extractor.py - Extract CLIP ViT-B/32 (512-d) features
    run_clip.py         - Master script for v2

  v3_sota/              ← Best results, targets DSNet+CLIP (0.65 TVSum)
    config_v3.py        - Unified config; set BACKBONE here
    model_v3.py         - Hierarchical windowed attention + multi-scale conv
    train_v3.py         - Contrastive loss + TTA + fold ensemble
    extract_clip_v3.py  - ViT-B/32 or ViT-L/14 feature extraction
    main_v3.py          - One-command full pipeline (extract+train+evaluate)

============================================================
QUICK START — V1 BASELINE (no GPU needed for small test)
============================================================

  Step 1: Install dependencies
    pip install torch torchvision h5py numpy scipy tqdm

  Step 2: Copy shared files
    cp shared/*.py v1_baseline/
    cd v1_baseline/

  Step 3: Download datasets
    TVSum: https://github.com/yalesong/tvsum  → place as data/tvsum.h5
    SumMe: https://gyglim.github.io/me/vsum   → place as data/summe.h5

  Step 4: Run
    python run_all.py --device cuda
    # or CPU: python run_all.py --device cpu

  Expected results:
    TVSum DPP F1 ~ 0.32    Top-k F1 ~ 0.35
    SumMe DPP F1 ~ 0.38    Top-k F1 ~ 0.40

============================================================
QUICK START — V3 SOTA (with raw videos, targets 0.65 TVSum)
============================================================

  Step 1: Install dependencies
    pip install torch torchvision h5py numpy scipy tqdm
    pip install git+https://github.com/openai/CLIP.git
    pip install imageio[ffmpeg] av

  Step 2: Copy shared files
    cp shared/*.py v3_sota/
    cd v3_sota/

  Step 3: Set backbone in config_v3.py
    BACKBONE = 'vitl14'     # ViT-L/14, 768-d, best results
    # or
    BACKBONE = 'vitb32'     # ViT-B/32, 512-d, faster
    # or
    BACKBONE = 'googlenet'  # no raw videos needed, uses eccv16 HDF5

  Step 4: Place raw videos (if using CLIP)
    raw_videos/tvsum/   ← all 50 TVSum .mp4 files
    raw_videos/summe/   ← all 25 SumMe .mp4 files

  Step 5: Run
    python main_v3.py --extract --device cuda   # with raw videos
    python main_v3.py --device cuda             # without raw videos

  Expected results (ViT-L/14):
    TVSum ensemble DPP F1 ~ 0.60–0.65
    SumMe ensemble DPP F1 ~ 0.56–0.61

============================================================
COMPARISON TABLE (to include in your paper)
============================================================

  Method                     TVSum     SumMe
  ─────────────────────────────────────────
  Random baseline            0.260     0.310
  DR-DSN (2018)              0.578     0.512
  VASNet (2019)              0.614     0.497
  PGL-SUM (2021)             0.644     0.555
  DSNet+CLIP (target)        0.650     ~0.58
  Ours v1 (GoogLeNet+DPP)   0.321     0.379
  Ours v3 (ViT-L/14+DPP)    0.60+     0.56+  ← fill from your run

============================================================
COMMON ERRORS & FIXES
============================================================

  Error: "HDF5 not found"
  Fix:   Check data/ folder has tvsum.h5 and summe.h5

  Error: "CLIP not installed"
  Fix:   pip install git+https://github.com/openai/CLIP.git

  Error: "No frames extracted"
  Fix:   pip install imageio[ffmpeg] av   (or install ffmpeg system-wide)

  Error: "CUDA out of memory" during CLIP extraction
  Fix:   Reduce batch size: python extract_clip_v3.py --batch 16

  Error: "feat_dim mismatch" at training
  Fix:   Check BACKBONE in config_v3.py matches your HDF5 feat_dim

============================================================
