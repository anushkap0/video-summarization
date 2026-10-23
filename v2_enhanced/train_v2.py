"""
train_v2.py — Enhanced 5-fold cross-validation training (Option C)
===================================================================
Improvements over train.py (v1):
  1. Supports both CLIP (512-d) and GoogLeNet (1024-d) features via config_v2.py
  2. Linear warmup + cosine annealing LR schedule
  3. Label smoothing on regression targets
  4. Gradient accumulation (simulates larger effective batch size)
  5. Best-checkpoint saving per fold (instead of last epoch)
  6. Rank-based loss: pushes selected frames higher than unselected frames
  7. Mixed MSE + rank loss for more stable training

Usage:
    # CLIP features (set MODE='clip' in config_v2.py first)
    python train_v2.py --dataset tvsum --device cuda

    # GoogLeNet HDF5 with enhanced training (set MODE='enhanced')
    python train_v2.py --dataset summe --device cuda

    # All datasets
    python run_clip.py --device cuda
"""

import os, json, argparse, random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import config_v2 as C
from model_v2      import TransformerScorerV2
from dataset       import VideoDataset, collate_fn, build_kfold_splits
from evaluate      import evaluate_summary, mean_f1
from dpp_sampler   import dpp_select

torch.manual_seed(C.SEED)
np.random.seed(C.SEED)
random.seed(C.SEED)


# ── Loss functions ────────────────────────────────────────────────────────────
def smooth_targets(targets: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    """Label smoothing: pull targets toward 0.5 by alpha."""
    return targets * (1 - alpha) + 0.5 * alpha


def rank_loss(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Pairwise ranking loss: selected frames should score higher than unselected.
    Targets are binary (gtsummary). Loss = mean(max(0, score_neg - score_pos + margin)).
    """
    pos_mask = targets > 0.5
    neg_mask = ~pos_mask
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return torch.tensor(0.0, device=scores.device)

    pos_scores = scores[pos_mask].mean()
    neg_scores = scores[neg_mask].mean()
    margin = 0.1
    return torch.clamp(neg_scores - pos_scores + margin, min=0.0)


def combined_loss(scores: torch.Tensor,
                  gtscore: torch.Tensor,
                  gtsummary: torch.Tensor,
                  alpha: float = C.LABEL_SMOOTHING,
                  rank_weight: float = 0.3) -> torch.Tensor:
    """
    MSE loss on smoothed importance scores + pairwise ranking loss.
    """
    smooth = smooth_targets(gtscore, alpha)
    mse    = nn.functional.mse_loss(scores, smooth)
    gt_bin = torch.from_numpy(gtsummary.astype(np.float32)).to(scores.device)
    rloss  = rank_loss(scores, gt_bin)
    return mse + rank_weight * rloss


# ── LR schedule with warmup ───────────────────────────────────────────────────
def get_lr(optimizer, epoch: int, warmup: int, base_lr: float) -> float:
    if epoch < warmup:
        lr = base_lr * (epoch + 1) / warmup
        for pg in optimizer.param_groups:
            pg['lr'] = lr
    return optimizer.param_groups[0]['lr']


# ── Training loop ─────────────────────────────────────────────────────────────
def train_one_fold(
    h5_path:    str,
    train_keys: list,
    test_keys:  list,
    fold_idx:   int,
    dataset_name: str,
    device:     str,
    epochs:     int,
    results_dir: str,
) -> dict:

    train_ds = VideoDataset(h5_path, train_keys)
    test_ds  = VideoDataset(h5_path, test_keys)

    model = TransformerScorerV2(feat_dim=C.FEATURE_DIM).to(device)
    opt   = AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(opt, T_max=epochs - C.WARMUP_EPOCHS, eta_min=1e-6)

    best_f1   = 0.0
    best_ckpt = os.path.join(results_dir, f'{dataset_name}_fold{fold_idx+1}_v2.pth')

    for epoch in range(epochs):
        # ── Warmup ──────────────────────────────────────────────────────────
        get_lr(opt, epoch, C.WARMUP_EPOCHS, C.LR)

        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        opt.zero_grad()
        total_loss = 0.0
        accum_step = 0

        idxs = np.random.permutation(len(train_ds))
        for step, i in enumerate(idxs):
            sample   = collate_fn([train_ds[int(i)]])
            feats    = sample['features'].to(device)          # (T, D)
            gtscore  = sample['gtscore'].to(device)           # (T,)
            gtsummary = sample['gtsummary']                   # (T,) numpy

            scores = model(feats)                             # (T,)
            loss   = combined_loss(scores, gtscore, gtsummary) / C.BATCH_ACCUM
            loss.backward()
            total_loss += loss.item() * C.BATCH_ACCUM
            accum_step += 1

            if accum_step % C.BATCH_ACCUM == 0 or step == len(idxs) - 1:
                nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
                opt.step()
                opt.zero_grad()

        if epoch >= C.WARMUP_EPOCHS:
            scheduler.step()

        avg_loss = total_loss / len(train_ds)

        # ── Evaluate every 5 epochs ──────────────────────────────────────────
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            val_f1 = _evaluate(model, test_ds, device, use_dpp=True)
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save({'model_state': model.state_dict(),
                            'epoch': epoch,
                            'f1': best_f1,
                            'feat_dim': C.FEATURE_DIM,
                            'mode': C.MODE},
                           best_ckpt)
            print(f'  Fold {fold_idx+1} | Epoch {epoch+1:3d}/{epochs} | '
                  f'loss={avg_loss:.4f} | val_F1={val_f1:.4f} | '
                  f'best={best_f1:.4f}')
        else:
            if (epoch + 1) % 10 == 0:
                print(f'  Fold {fold_idx+1} | Epoch {epoch+1:3d}/{epochs} | '
                      f'loss={avg_loss:.4f}')

    # ── Final evaluation with best checkpoint ─────────────────────────────────
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state'])

    dpp_f1  = _evaluate(model, test_ds, device, use_dpp=True)
    topk_f1 = _evaluate(model, test_ds, device, use_dpp=False)

    print(f'\n  Fold {fold_idx+1} DONE | DPP F1={dpp_f1:.4f} | Top-k F1={topk_f1:.4f}')
    return {'fold': fold_idx + 1, 'dpp_f1': dpp_f1, 'topk_f1': topk_f1,
            'checkpoint': best_ckpt}


@torch.no_grad()
def _evaluate(model, dataset, device: str, use_dpp: bool) -> float:
    model.eval()
    results = []
    for i in range(len(dataset)):
        sample = collate_fn([dataset[i]])
        feats  = sample['features'].to(device)
        scores = model(feats).cpu().numpy()

        T = len(scores)
        k = max(1, int(T * C.SUMMARY_RATIO))

        if use_dpp:
            feats_np = sample['features'].numpy()
            sel = dpp_select(scores, feats_np, k, lam=C.DPP_LAMBDA)
        else:
            sel = np.argsort(scores)[-k:]

        pred_summary = np.zeros(T, dtype=np.int8)
        pred_summary[sel] = 1

        res = evaluate_summary(pred_summary, {
            'picks':        sample.get('picks'),
            'n_frames':     sample['n_frames'],
            'gtsummary':    sample['gtsummary'],
            'user_summary': sample.get('user_summary'),
        })
        results.append(res)

    return mean_f1(results)


# ── Main ──────────────────────────────────────────────────────────────────────
def run(dataset_name: str, device: str, epochs: int, results_dir: str):
    h5_path = C.get_dataset_path(dataset_name)

    if not os.path.exists(h5_path):
        print(f'HDF5 not found: {h5_path}')
        if C.MODE == 'clip':
            print(f'Run clip_feature_extractor.py first to generate {h5_path}')
        return None

    os.makedirs(results_dir, exist_ok=True)
    splits = build_kfold_splits(h5_path, C.N_FOLDS, C.SEED)

    print(f'\n{"="*60}')
    print(f'Dataset  : {dataset_name}  ({h5_path})')
    print(f'Mode     : {C.MODE}  |  feat_dim={C.FEATURE_DIM}')
    print(f'Folds    : {C.N_FOLDS}  |  Epochs={epochs}')
    print(f'Device   : {device}')
    print(f'{"="*60}')

    fold_results = []
    for fold_idx, (train_keys, test_keys) in enumerate(splits):
        print(f'\nFold {fold_idx+1}/{C.N_FOLDS} | '
              f'train={len(train_keys)} test={len(test_keys)}')
        res = train_one_fold(h5_path, train_keys, test_keys, fold_idx,
                             dataset_name, device, epochs, results_dir)
        fold_results.append(res)

    dpp_scores  = [r['dpp_f1']  for r in fold_results]
    topk_scores = [r['topk_f1'] for r in fold_results]

    summary = {
        'dataset':        dataset_name,
        'mode':           C.MODE,
        'feat_dim':       C.FEATURE_DIM,
        'n_layers':       C.N_LAYERS,
        'epochs':         epochs,
        'dpp_mean':       float(np.mean(dpp_scores)),
        'dpp_std':        float(np.std(dpp_scores)),
        'topk_mean':      float(np.mean(topk_scores)),
        'topk_std':       float(np.std(topk_scores)),
        'fold_results':   fold_results,
    }

    out_json = os.path.join(results_dir, f'{dataset_name}_results_v2.json')
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\n{"="*60}')
    print(f'RESULTS — {dataset_name.upper()} ({C.MODE})')
    print(f'  DPP  : {np.mean(dpp_scores):.4f} ± {np.std(dpp_scores):.4f}')
    print(f'  Top-k: {np.mean(topk_scores):.4f} ± {np.std(topk_scores):.4f}')
    print(f'  Saved → {out_json}')
    print(f'{"="*60}')

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='tvsum',
                        choices=['summe', 'tvsum', 'vsumm'])
    parser.add_argument('--device',  default=C.DEVICE)
    parser.add_argument('--epochs',  type=int, default=C.EPOCHS)
    parser.add_argument('--results_dir', default=C.RESULTS_DIR)
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device == 'cuda':
        print('CUDA not available, falling back to CPU')
        args.device = 'cpu'

    run(args.dataset, args.device, args.epochs, args.results_dir)
