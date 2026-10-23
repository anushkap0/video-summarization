"""
main_v3.py — Complete v3 pipeline in one command
=================================================
Runs: feature extraction → 5-fold training → TTA evaluation → fold ensemble
Targets DSNet+CLIP benchmark: TVSum ~0.65, SumMe ~0.58

THREE PATHS — pick one based on what you have:

  PATH A — Raw videos + ViT-L/14  (BEST, targets 0.65 TVSum)
  ───────────────────────────────
  1. Put raw videos in:
       raw_videos/tvsum/   ← all 50 TVSum .mp4 files
       raw_videos/summe/   ← all 25 SumMe .mp4 files
  2. In config_v3.py: set BACKBONE = 'vitl14'
  3. Run:
       python main_v3.py --extract --device cuda

  PATH B — Raw videos + ViT-B/32  (good, targets 0.54 TVSum, no re-download)
  ───────────────────────────────
  Same as A but set BACKBONE = 'vitb32' in config_v3.py

  PATH C — No raw videos, use existing eccv16 HDF5  (quick start, ~0.38 TVSum)
  ──────────────────────────────────────────────────
  1. In config_v3.py: set BACKBONE = 'googlenet'
  2. Make sure data/tvsum.h5 and data/summe.h5 exist
  3. Run:
       python main_v3.py --device cuda

Full usage:
    python main_v3.py --extract --device cuda                # PATH A or B
    python main_v3.py --device cuda                          # PATH C or pre-extracted
    python main_v3.py --extract --device cuda --epochs 50   # quick test
    python main_v3.py --device cuda --datasets tvsum        # single dataset
"""

import os, sys, json, argparse, subprocess
import numpy as np
import torch

import config_v3 as C

DATASETS = ['summe', 'tvsum']

# Published comparison numbers
PUBLISHED = [
    ('Random baseline',         0.260, 0.310),
    ('DR-DSN (2018)',            0.578, 0.512),
    ('VASNet (2019)',            0.614, 0.497),
    ('PGL-SUM (2021)',           0.644, 0.555),
    ('DSNet+CLIP (target)',      0.650, 0.580),
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def sep(title: str = ''):
    line = '=' * 68
    if title:
        print(f'\n{line}\n  {title}\n{line}')
    else:
        print(line)


def check_clip():
    try:
        import clip  # noqa
        return True
    except ImportError:
        return False


def install_clip():
    print('  Installing CLIP...')
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install',
        'git+https://github.com/openai/CLIP.git', '--quiet'
    ])
    print('  CLIP installed.')


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def run_extraction(datasets: list, device: str) -> dict:
    """
    Extract CLIP features for all requested datasets.
    Returns {dataset: h5_path} for successfully extracted datasets.
    """
    from extract_clip_v3 import process_dataset

    sep(f'STEP 1 — Feature Extraction  ({C.BACKBONE})')
    results = {}
    for dataset in datasets:
        h5 = process_dataset(dataset, C.BACKBONE, device)
        if h5:
            results[dataset] = h5
        else:
            print(f'  Extraction failed for {dataset} — '
                  f'will use existing HDF5 if available')
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def run_training(
    dataset:    str,
    device:     str,
    epochs:     int,
    results_dir: str,
    clip_model=None,
) -> dict | None:
    """Train 5-fold CV with v3 model. Returns summary dict."""

    from train_v3 import run as train_run
    return train_run(dataset, device, epochs, results_dir, clip_model)


# ─────────────────────────────────────────────────────────────────────────────
# Results table
# ─────────────────────────────────────────────────────────────────────────────

def print_results_table(all_results: list):
    sep('FINAL RESULTS')

    # Per-fold table
    print(f'  {"Dataset":<10} {"Backbone":<10} '
          f'{"Fold DPP":>10} {"Ensemble DPP":>14} {"Top-k":>8} {"Std":>6}')
    print('  ' + '-' * 62)
    for r in all_results:
        if r is None:
            continue
        ens = r.get('ensemble_dpp_f1', float('nan'))
        print(f'  {r["dataset"]:<10} {r["backbone"]:<10} '
              f'{r["dpp_mean"]:>10.4f} {ens:>14.4f} '
              f'{r["topk_mean"]:>8.4f} {r["dpp_std"]:>6.4f}')

    # Comparison with published
    print()
    print(f'  {"Method":<35} {"TVSum":>8} {"SumMe":>8}')
    print('  ' + '-' * 55)
    for name, tv, sm in PUBLISHED:
        print(f'  {name:<35} {tv:>8.3f} {sm:>8.3f}')

    print()
    tvsum_r = next((r for r in all_results if r and r['dataset'] == 'tvsum'), None)
    summe_r = next((r for r in all_results if r and r['dataset'] == 'summe'), None)

    tv_val = f'{tvsum_r["ensemble_dpp_f1"]:.3f}' if tvsum_r else '  —  '
    sm_val = f'{summe_r["ensemble_dpp_f1"]:.3f}' if summe_r else '  —  '
    label  = f'Ours v3 ({C.BACKBONE}, ensemble)'
    print(f'  {label:<35} {tv_val:>8} {sm_val:>8}')

    # Gap to target
    if tvsum_r and 'ensemble_dpp_f1' in tvsum_r:
        gap = 0.650 - tvsum_r['ensemble_dpp_f1']
        print()
        if gap <= 0.0:
            print(f'  ✓  Beat DSNet+CLIP on TVSum by {-gap:.3f} F1 points!')
        else:
            print(f'  Gap to DSNet+CLIP target (0.650): {gap:.3f} F1 on TVSum')
            if gap < 0.03:
                print(f'  Very close — consider running more epochs or ensemble more folds.')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='v3 pipeline: CLIP + hierarchical transformer + contrastive loss')
    parser.add_argument('--datasets',     nargs='+', default=DATASETS,
                        choices=DATASETS)
    parser.add_argument('--device',       default=C.DEVICE)
    parser.add_argument('--epochs',       type=int, default=C.EPOCHS)
    parser.add_argument('--results_dir',  default=C.RESULTS_DIR)
    parser.add_argument('--extract',      action='store_true',
                        help='Extract CLIP features before training '
                             '(needs raw videos in raw_videos/)')
    parser.add_argument('--backbone',     default=None,
                        choices=['vitl14', 'vitb32', 'googlenet'],
                        help='Override BACKBONE from config_v3.py')
    args = parser.parse_args()

    # Apply CLI backbone override
    if args.backbone:
        C.BACKBONE    = args.backbone
        C.FEATURE_DIM = C._FEAT_DIMS[args.backbone]

    sep(f'v3 Video Summarization Pipeline')
    C.print_config()
    print(f'  Datasets     : {args.datasets}')
    print(f'  Epochs       : {args.epochs}')
    print(f'  Results dir  : {args.results_dir}')
    print(f'  Extract first: {args.extract}')

    # ── CLIP install ─────────────────────────────────────────────────────────
    if C.is_clip_backbone():
        if not check_clip():
            install_clip()

    # ── Load CLIP model (shared for both contrastive loss + extraction) ──────
    clip_model = None
    if C.is_clip_backbone():
        try:
            import clip as _clip
            clip_name = C.get_clip_model_name()
            print(f'\n  Loading CLIP {clip_name} on {args.device}...')
            clip_model, _ = _clip.load(clip_name, device=args.device)
            clip_model.eval()
            print(f'  CLIP {clip_name} loaded.')
        except ImportError:
            print('  CLIP import failed — contrastive loss disabled.')

    # ── Feature extraction ───────────────────────────────────────────────────
    if args.extract:
        if not C.is_clip_backbone():
            print('\n  --extract is only for CLIP backbones (vitl14, vitb32).')
            print('  GoogLeNet HDF5 files must already be on disk.')
        else:
            run_extraction(args.datasets, args.device)

    # ── Training loop ─────────────────────────────────────────────────────────
    os.makedirs(args.results_dir, exist_ok=True)
    all_results = []

    for dataset in args.datasets:
        h5_path = C.get_h5_path(dataset)
        if not os.path.exists(h5_path):
            print(f'\n  SKIP: HDF5 not found for {dataset}: {h5_path}')
            if args.extract:
                print(f'  (extraction may have failed — check logs above)')
            else:
                print(f'  Tip: run with --extract to generate features first,')
                print(f'  or set BACKBONE="googlenet" to use existing eccv16 HDF5.')
            continue

        sep(f'STEP 2 — Training  {dataset.upper()}')
        print(f'  HDF5   : {h5_path}')
        print(f'  feat   : {C.FEATURE_DIM}-d  |  arch: TransformerScorerV3')

        result = run_training(dataset, args.device, args.epochs,
                              args.results_dir, clip_model)
        if result:
            result['backbone'] = C.BACKBONE
            all_results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    if all_results:
        print_results_table(all_results)

        combined = os.path.join(args.results_dir, 'all_results_v3.json')
        with open(combined, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'\n  All results saved → {combined}')
        print()
        print('  Paper metrics to report:')
        for r in all_results:
            ens = r.get('ensemble_dpp_f1', r['dpp_mean'])
            print(f'    {r["dataset"].upper():<8} DPP F1 = {ens:.4f}  '
                  f'(ensemble of {C.N_FOLDS} folds + TTA)')
    else:
        print('\n  No results to report — check errors above.')


if __name__ == '__main__':
    main()
