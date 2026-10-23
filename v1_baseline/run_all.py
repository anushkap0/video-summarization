"""
Quick-start runner — trains on all three datasets and prints a results table.

Usage:
    python run_all.py --device cuda
    python run_all.py --device cpu --epochs 20   # faster test run

After running, open results/ to find:
    summe_results.json
    tvsum_results.json
    vsumm_results.json
    <dataset>_fold<N>.pth   (model checkpoints)
"""

import os, argparse, json
from pathlib import Path
import torch

import config as C
from train import run_kfold


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=C.EPOCHS)
    parser.add_argument('--outdir', default='results')
    parser.add_argument('--nodpp',  action='store_true')
    args = parser.parse_args()

    # Allow overriding epochs from CLI without editing config.py
    C.EPOCHS = args.epochs

    device  = torch.device(args.device)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_dpp = not args.nodpp

    rows = []
    for ds, h5 in C.DATASET_PATHS.items():
        if not os.path.exists(h5):
            print(f'\n[SKIP] {ds}: HDF5 not found at {h5}')
            print(f'  → Download: https://github.com/KaiyangZhou/pytorch-vsumm-reinforce')
            rows.append({'dataset': ds, 'mean_f1': None, 'std_f1': None})
            continue
        m, s = run_kfold(ds, h5, device, out_dir, use_dpp)
        rows.append({'dataset': ds, 'mean_f1': m, 'std_f1': s})

    print('\n')
    print('┌──────────────────────────────────────────────┐')
    print('│        FINAL RESULTS SUMMARY                 │')
    print('├─────────┬───────────┬───────────┬────────────┤')
    print('│ Dataset │  Mean F1  │   Std F1  │  vs. SOTA  │')
    print('├─────────┼───────────┼───────────┼────────────┤')
    sota = {'summe': 0.512, 'tvsum': 0.624, 'vsumm': 0.360}
    for row in rows:
        ds = row['dataset']
        if row['mean_f1'] is None:
            print(f'│ {ds:7s} │   N/A     │   N/A     │    N/A     │')
        else:
            gap = sota.get(ds, 0) - row["mean_f1"]
            print(f'│ {ds:7s} │   {row["mean_f1"]:.4f}  │   {row["std_f1"]:.4f}  │  Δ={gap:.4f}  │')
    print('└─────────┴───────────┴───────────┴────────────┘')
    print()
    print('SOTA references: SumMe→VASNet+GL(0.512), TVSum→VASNet+GL(0.624), VSUMM→OV²S(0.360)')
