"""
run_clip.py — Complete Option C pipeline in one command
=======================================================
Runs the full CLIP + Enhanced Transformer + DPP pipeline:

  PATH A (raw videos available):
    1. Extract CLIP ViT-B/32 features for all videos
    2. Train 5-fold CV with enhanced Transformer (v2)
    3. Print final results table

  PATH B (only HDF5 available, no raw videos):
    1. Skip CLIP extraction
    2. Train 5-fold CV with enhanced Transformer (v2) on GoogLeNet HDF5
    3. Print final results table

Usage:
    # Full CLIP pipeline (raw videos in raw_videos/summe/ and raw_videos/tvsum/)
    python run_clip.py --device cuda --clip

    # Enhanced training only (no CLIP, uses existing HDF5)
    python run_clip.py --device cuda --enhanced

    # Both datasets
    python run_clip.py --device cuda --clip --datasets summe tvsum

    # Quick test (10 epochs)
    python run_clip.py --device cuda --enhanced --epochs 10
"""

import os, sys, json, argparse, subprocess
import numpy as np
import config_v2 as C

DATASETS = ['summe', 'tvsum']


def print_header(title: str):
    print('\n' + '=' * 65)
    print(f'  {title}')
    print('=' * 65)


def check_clip_installed() -> bool:
    try:
        import clip  # noqa: F401
        return True
    except ImportError:
        return False


def install_clip():
    print('Installing CLIP...')
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install',
        'git+https://github.com/openai/CLIP.git', '--quiet'
    ])
    print('CLIP installed.')


def extract_clip_features_for_dataset(dataset: str, device: str):
    """Extract CLIP features for all videos in a dataset."""
    raw_dir  = C.RAW_VIDEO_DIRS.get(dataset)
    out_h5   = C.DATASET_PATHS_CLIP.get(dataset)
    ref_h5   = C.DATASET_PATHS_GOOGLENET.get(dataset)

    if not raw_dir or not os.path.isdir(raw_dir):
        print(f'  Raw video directory not found: {raw_dir}')
        print(f'  Skipping CLIP extraction for {dataset}.')
        print(f'  To enable: place videos in {raw_dir}')
        return False

    print_header(f'CLIP Feature Extraction — {dataset.upper()}')
    print(f'  Input dir  : {raw_dir}')
    print(f'  Output HDF5: {out_h5}')
    if ref_h5 and os.path.exists(ref_h5):
        print(f'  Reference  : {ref_h5}  (annotations will be copied)')

    cmd = [
        sys.executable, 'clip_feature_extractor.py',
        '--folder', raw_dir,
        '--out',    out_h5,
        '--device', device,
    ]
    if ref_h5 and os.path.exists(ref_h5):
        cmd += ['--ref_h5', ref_h5]

    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f'  ERROR: CLIP extraction failed for {dataset}')
        return False

    print(f'  Done. CLIP features saved to {out_h5}')
    return True


def run_training(dataset: str, device: str, epochs: int, results_dir: str) -> dict | None:
    """Run enhanced training for one dataset."""
    from train_v2 import run
    return run(dataset, device, epochs, results_dir)


def print_results_table(all_results: list[dict]):
    print_header('FINAL RESULTS SUMMARY')
    print(f'  {"Dataset":<10} {"Mode":<10} {"DPP F1":>10} {"Top-k F1":>10} {"Std":>8}')
    print('  ' + '-' * 50)
    for r in all_results:
        if r is None:
            continue
        print(f'  {r["dataset"]:<10} {r["mode"]:<10} '
              f'{r["dpp_mean"]:>10.4f} {r["topk_mean"]:>10.4f} '
              f'{r["dpp_std"]:>8.4f}')

    print()
    print('  Comparison with published results:')
    print(f'  {"Method":<30} {"TVSum":>8} {"SumMe":>8}')
    print('  ' + '-' * 50)
    print(f'  {"Random baseline":<30} {"0.260":>8} {"0.310":>8}')
    print(f'  {"DR-DSN (2018)":<30} {"0.578":>8} {"0.512":>8}')
    print(f'  {"VASNet (2019)":<30} {"0.614":>8} {"0.497":>8}')
    print(f'  {"PGL-SUM (2021)":<30} {"0.644":>8} {"0.555":>8}')

    for r in all_results:
        if r is None:
            continue
        label = f'Ours v2 ({r["mode"]}, DPP)'
        val   = f'{r["dpp_mean"]:.3f}'
        if r['dataset'] == 'tvsum':
            print(f'  {label:<30} {val:>8} {"—":>8}')
        else:
            print(f'  {label:<30} {"—":>8} {val:>8}')


def main():
    parser = argparse.ArgumentParser()
    mode_grp = parser.add_mutually_exclusive_group(required=True)
    mode_grp.add_argument('--clip',     action='store_true',
                          help='Extract CLIP features then train (needs raw videos)')
    mode_grp.add_argument('--enhanced', action='store_true',
                          help='Train with enhanced model on existing GoogLeNet HDF5')

    parser.add_argument('--datasets', nargs='+', default=DATASETS,
                        choices=DATASETS)
    parser.add_argument('--device',   default=C.DEVICE)
    parser.add_argument('--epochs',   type=int, default=C.EPOCHS)
    parser.add_argument('--results_dir', default=C.RESULTS_DIR)
    args = parser.parse_args()

    # ── Set mode in config ────────────────────────────────────────────────────
    if args.clip:
        C.MODE        = 'clip'
        C.FEATURE_DIM = 512
        if not check_clip_installed():
            install_clip()
    else:
        C.MODE        = 'enhanced'
        C.FEATURE_DIM = 1024

    print_header(f'Option C Pipeline — Mode: {C.MODE.upper()}')
    print(f'  Datasets  : {args.datasets}')
    print(f'  Device    : {args.device}')
    print(f'  Epochs    : {args.epochs}')
    print(f'  Feat dim  : {C.FEATURE_DIM}')
    print(f'  Layers    : {C.N_LAYERS}  (v2: was 2)')
    print(f'  Results → : {args.results_dir}/')

    all_results = []

    for dataset in args.datasets:
        # ── CLIP extraction ───────────────────────────────────────────────────
        if args.clip:
            ok = extract_clip_features_for_dataset(dataset, args.device)
            if not ok:
                print(f'  Falling back to GoogLeNet HDF5 for {dataset}')
                C.MODE        = 'enhanced'
                C.FEATURE_DIM = 1024

        # ── Training ──────────────────────────────────────────────────────────
        print_header(f'Training — {dataset.upper()}')
        result = run_training(dataset, args.device, args.epochs, args.results_dir)
        if result:
            all_results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    if all_results:
        print_results_table(all_results)

        # Save combined results
        os.makedirs(args.results_dir, exist_ok=True)
        combined_path = os.path.join(args.results_dir, 'all_results_v2.json')
        with open(combined_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'\n  Combined results → {combined_path}')


if __name__ == '__main__':
    main()
