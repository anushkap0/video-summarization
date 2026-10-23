"""
Step 4 – Training with 5-Fold Cross-Validation
================================================
Loss function: MSE between predicted importance scores and normalised ground-truth scores.

This is a supervised regression formulation. The model learns to predict
how important each frame is relative to the human annotations. At inference,
the predicted scores are passed through DPP sampling to select the final keyframes.

Usage:
    python train.py --dataset summe --device cuda
    python train.py --dataset tvsum --device cpu
    python train.py --dataset vsumm --device cpu

Results are saved to results/<dataset>_fold<N>.pth and a summary CSV.
"""

import os, argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config as C
from dataset  import VideoDataset, collate_fn, build_kfold_splits
from model    import TransformerScorer
from dpp_sampler import select_keyframes, select_keyframes_topk
from evaluate import evaluate_summary, mean_f1


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_fold(model: nn.Module,
                   train_names: list[str],
                   h5_path: str,
                   device: torch.device,
                   epochs: int = C.EPOCHS) -> list[float]:
    """Train model for one fold. Returns list of per-epoch train losses."""
    dataset = VideoDataset(h5_path, train_names)
    loader  = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    optimiser = optim.Adam(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    model.train()
    losses = []
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for sample in loader:
            features = sample['features'].unsqueeze(0).to(device)   # (1, T, D)
            gtscore  = sample['gtscore'].unsqueeze(0).to(device)    # (1, T)

            optimiser.zero_grad()
            pred_scores = model(features)                            # (1, T)
            loss = criterion(pred_scores, gtscore)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimiser.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(dataset)
        losses.append(avg_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f'    Epoch [{epoch:3d}/{epochs}]  loss = {avg_loss:.6f}')

    return losses


@torch.no_grad()
def evaluate_fold(model: nn.Module,
                  test_names: list[str],
                  h5_path: str,
                  device: torch.device,
                  use_dpp: bool = True) -> tuple[list[dict], float]:
    """Evaluate model on one fold. Returns (per-video results, mean F1)."""
    dataset = VideoDataset(h5_path, test_names)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model.eval()
    results = []
    for sample in loader:
        features = sample['features'].unsqueeze(0).to(device)   # (1, T, D)
        pred_scores = model(features).squeeze(0).cpu().numpy()   # (T,)
        feat_np     = sample['features'].numpy()                 # (T, D)

        if use_dpp:
            pred_summary = select_keyframes(pred_scores, feat_np,
                                            ratio=C.SUMMARY_RATIO,
                                            lam=C.DPP_LAMBDA)
        else:
            pred_summary = select_keyframes_topk(pred_scores, ratio=C.SUMMARY_RATIO)

        result = evaluate_summary(pred_summary, sample)
        result['video'] = sample['name']
        results.append(result)
        print(f'    {sample["name"]:30s}  F1={result["f1"]:.4f}  '
              f'P={result["precision"]:.4f}  R={result["recall"]:.4f}')

    return results, mean_f1(results)


def run_kfold(dataset_name: str,
              h5_path: str,
              device: torch.device,
              out_dir: Path,
              use_dpp: bool = True):
    """Run full 5-fold CV and save results."""
    set_seed(C.SEED)
    splits = build_kfold_splits(h5_path, n_folds=C.N_FOLDS, seed=C.SEED)

    all_fold_f1 = []
    all_results = []

    print(f'\n{"="*60}')
    print(f'Dataset: {dataset_name.upper()}  |  Folds: {C.N_FOLDS}  |  Device: {device}')
    print(f'{"="*60}')

    for fold_idx, (train_names, test_names) in enumerate(splits, start=1):
        print(f'\n── Fold {fold_idx}/{C.N_FOLDS}  '
              f'(train={len(train_names)}, test={len(test_names)}) ──')

        model = TransformerScorer(
            feat_dim=C.FEATURE_DIM,
            d_model=C.D_MODEL,
            n_heads=C.N_HEADS,
            n_layers=C.N_LAYERS,
            dropout=C.DROPOUT,
        ).to(device)

        print(f'  Training ...')
        t0 = time.time()
        losses = train_one_fold(model, train_names, h5_path, device, C.EPOCHS)
        t1 = time.time()
        print(f'  Training done in {t1-t0:.1f}s')

        # Save checkpoint
        ckpt_path = out_dir / f'{dataset_name}_fold{fold_idx}.pth'
        torch.save({
            'fold': fold_idx,
            'model_state': model.state_dict(),
            'train_losses': losses,
            'config': {k: v for k, v in vars(C).items() if not k.startswith('_')},
        }, ckpt_path)
        print(f'  Checkpoint saved → {ckpt_path}')

        print(f'  Evaluating (DPP={use_dpp}) ...')
        fold_results, fold_f1 = evaluate_fold(model, test_names, h5_path, device, use_dpp)
        print(f'  ▶ Fold {fold_idx} mean F1 = {fold_f1:.4f}')

        all_fold_f1.append(fold_f1)
        all_results.extend(fold_results)

    # ── Summary ──────────────────────────────────────────────────────────────
    final_mean = float(np.mean(all_fold_f1))
    final_std  = float(np.std(all_fold_f1))

    print(f'\n{"="*60}')
    print(f'{dataset_name.upper()} 5-fold CV results:')
    for i, f1 in enumerate(all_fold_f1, 1):
        print(f'  Fold {i}: F1 = {f1:.4f}')
    print(f'  Mean  F1 = {final_mean:.4f}')
    print(f'  Std   F1 = {final_std:.4f}')
    print(f'{"="*60}')

    # Save JSON summary
    summary = {
        'dataset':    dataset_name,
        'n_folds':    C.N_FOLDS,
        'use_dpp':    use_dpp,
        'fold_f1':    all_fold_f1,
        'mean_f1':    final_mean,
        'std_f1':     final_std,
        'per_video':  all_results,
        'config': {
            'FEATURE_DIM': C.FEATURE_DIM,
            'D_MODEL':     C.D_MODEL,
            'N_HEADS':     C.N_HEADS,
            'N_LAYERS':    C.N_LAYERS,
            'EPOCHS':      C.EPOCHS,
            'LR':          C.LR,
            'SUMMARY_RATIO': C.SUMMARY_RATIO,
            'DPP_LAMBDA':  C.DPP_LAMBDA,
        },
    }
    json_path = out_dir / f'{dataset_name}_results.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Results saved → {json_path}')

    return final_mean, final_std


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['summe','tvsum','vsumm','all'],
                        default='summe')
    parser.add_argument('--device',  default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--nodpp',   action='store_true', help='Use top-k instead of DPP')
    parser.add_argument('--outdir',  default='results')
    args = parser.parse_args()

    device  = torch.device(args.device)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_dpp = not args.nodpp

    datasets = ['summe','tvsum','vsumm'] if args.dataset == 'all' else [args.dataset]
    summary_rows = []

    for ds in datasets:
        h5_path = C.DATASET_PATHS.get(ds)
        if not os.path.exists(h5_path):
            print(f'[SKIP] HDF5 not found: {h5_path}')
            print(f'       Download from: https://github.com/KaiyangZhou/pytorch-vsumm-reinforce')
            continue
        mean_f1_val, std_f1_val = run_kfold(ds, h5_path, device, out_dir, use_dpp)
        summary_rows.append({'dataset': ds, 'mean_f1': mean_f1_val, 'std_f1': std_f1_val})

    if summary_rows:
        print('\n── FINAL SUMMARY ──────────────────────────────────────')
        for row in summary_rows:
            print(f"  {row['dataset'].upper():8s}  F1 = {row['mean_f1']:.4f} ± {row['std_f1']:.4f}")
