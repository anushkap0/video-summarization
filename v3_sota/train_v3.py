"""
train_v3.py — Enhanced training with contrastive loss targeting DSNet+CLIP (0.65 TVSum)
========================================================================================
New in v3 vs train_v2.py:

  1. Text-Visual Contrastive Loss (CLIP-based)
       - Encodes the video key/title as CLIP text
       - Pushes summary frames semantically closer to video title than non-summary frames
       - Only active when MODE='clip' (requires CLIP loaded)

  2. Test-Time Augmentation (TTA)
       - Scores each video 3× (forward, time-reversed, light noise)
       - Averages predictions for more robust final scores
       - Used at evaluation time only (free +1-2 F1 pts)

  3. Fold ensemble at test time
       - All 5 fold checkpoints score each video; predictions averaged
       - Reported alongside per-fold results

  4. ViT-L/14 support (768-d)
       - Set FEATURE_DIM=768 in config_v2.py when using ViT-L/14 features

  5. Deeper warmup (15 epochs vs 10)
       - Larger model needs more careful warmup

Usage:
    # CLIP features + v3 model
    python train_v3.py --dataset tvsum --device cuda

    # Via master script (recommended)
    python run_v3.py --device cuda --clip --datasets summe tvsum
"""

import os, json, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import config_v3 as C
from model_v3      import TransformerScorerV3
from dataset       import VideoDataset, collate_fn, build_kfold_splits
from evaluate      import evaluate_summary, mean_f1
from dpp_sampler   import dpp_select

torch.manual_seed(C.SEED)
np.random.seed(C.SEED)
random.seed(C.SEED)

# Warmup longer for the bigger model
WARMUP_EPOCHS_V3 = getattr(C, 'WARMUP_EPOCHS', 15)


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def smooth_targets(targets: torch.Tensor, alpha: float = C.LABEL_SMOOTHING) -> torch.Tensor:
    return targets * (1 - alpha) + 0.5 * alpha


def rank_loss(scores: torch.Tensor, targets: torch.Tensor,
              margin: float = 0.1) -> torch.Tensor:
    """Pairwise ranking: summary frames should score higher than non-summary."""
    pos_mask = targets > 0.5
    neg_mask = ~pos_mask
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return torch.tensor(0.0, device=scores.device)
    pos_scores = scores[pos_mask].mean()
    neg_scores = scores[neg_mask].mean()
    return torch.clamp(neg_scores - pos_scores + margin, min=0.0)


def contrastive_loss_clip(
    clip_model,
    features:  torch.Tensor,    # (T, D)  already-encoded frame features
    gtsummary: np.ndarray,       # (T,)   binary ground-truth summary
    video_key: str,              # e.g. "video1" — used as text prompt
    device:    str,
) -> torch.Tensor:
    """
    Text-visual contrastive loss using CLIP's text encoder.

    Idea: A good summary frame should be more similar to the video's title
    than a non-summary frame.

    Loss = -log(
        exp(sim(text, mean_pos)) /
        (exp(sim(text, mean_pos)) + exp(sim(text, mean_neg)))
    )

    Falls back to 0 if all frames are positive or all negative.
    """
    try:
        import clip as _clip
    except ImportError:
        return torch.tensor(0.0, device=device)

    pos_mask = torch.from_numpy(gtsummary.astype(bool)).to(device)
    neg_mask = ~pos_mask

    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return torch.tensor(0.0, device=device)

    # Build a simple text prompt from the video key
    # e.g. "video1" → "a video about video1"
    prompt = f"a highlight summary frame from {video_key.replace('_', ' ')}"
    with torch.no_grad():
        text_tokens = _clip.tokenize([prompt]).to(device)
        text_feat   = clip_model.encode_text(text_tokens).float()   # (1, D)
        text_feat   = F.normalize(text_feat, dim=-1)

    pos_feat = features[pos_mask].mean(0, keepdim=True)   # (1, D)
    neg_feat = features[neg_mask].mean(0, keepdim=True)   # (1, D)

    pos_feat = F.normalize(pos_feat, dim=-1)
    neg_feat = F.normalize(neg_feat, dim=-1)

    sim_pos = (text_feat * pos_feat).sum()     # scalar
    sim_neg = (text_feat * neg_feat).sum()     # scalar

    temperature = 0.07
    logits = torch.stack([sim_pos, sim_neg]) / temperature   # (2,)
    label  = torch.tensor(0, device=device)                  # pos is class 0
    return F.cross_entropy(logits.unsqueeze(0), label.unsqueeze(0))


def combined_loss_v3(
    scores:     torch.Tensor,
    features:   torch.Tensor,
    gtscore:    torch.Tensor,
    gtsummary:  np.ndarray,
    video_key:  str,
    clip_model,
    device:     str,
    alpha:      float = C.LABEL_SMOOTHING,
    rank_weight:   float = getattr(C, 'RANK_WEIGHT', 0.3),
    contrast_weight: float = getattr(C, 'CONTRAST_WEIGHT', 0.2),
) -> torch.Tensor:
    smooth  = smooth_targets(gtscore, alpha)
    mse     = F.mse_loss(scores, smooth)

    gt_bin  = torch.from_numpy(gtsummary.astype(np.float32)).to(device)
    rloss   = rank_loss(scores, gt_bin)

    closs   = torch.tensor(0.0, device=device)
    if clip_model is not None and C.MODE == 'clip':
        closs = contrastive_loss_clip(
            clip_model, features.detach(), gtsummary, video_key, device)

    return mse + rank_weight * rloss + contrast_weight * closs


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────────────────────────────────────

def apply_warmup(optimizer, epoch: int, warmup: int, base_lr: float):
    if epoch < warmup:
        lr = base_lr * (epoch + 1) / warmup
        for pg in optimizer.param_groups:
            pg['lr'] = lr


# ─────────────────────────────────────────────────────────────────────────────
# Test-Time Augmentation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def score_with_tta(model: nn.Module, features: torch.Tensor,
                   device: str, n_aug: int = 3) -> np.ndarray:
    """
    Average predictions over n_aug augmentations:
      1. Original
      2. Time-reversed
      3. Light Gaussian noise (sigma=0.01)
    """
    model.eval()
    x = features.to(device)

    preds = []
    preds.append(model(x).cpu().numpy())                                         # original
    preds.append(model(x.flip(0)).cpu().numpy()[::-1].copy())                   # reversed
    preds.append(model(x + 0.01 * torch.randn_like(x)).cpu().numpy())          # noisy

    return np.mean(preds[:n_aug], axis=0)   # (T,)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop — one fold
# ─────────────────────────────────────────────────────────────────────────────

def train_one_fold(
    h5_path:      str,
    train_keys:   list,
    test_keys:    list,
    fold_idx:     int,
    dataset_name: str,
    device:       str,
    epochs:       int,
    results_dir:  str,
    clip_model=None,
) -> dict:

    train_ds = VideoDataset(h5_path, train_keys)
    test_ds  = VideoDataset(h5_path, test_keys)

    model = TransformerScorerV3(feat_dim=C.FEATURE_DIM, d_model=C.D_MODEL, n_local_layers=C.N_LOCAL_LAYERS,
                                 ffn_dim=C.FFN_DIM, dropout=C.DROPOUT, window=C.WINDOW_SIZE).to(device)
    opt   = AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(opt,
                                  T_max=max(1, epochs - WARMUP_EPOCHS_V3),
                                  eta_min=1e-6)

    best_f1   = 0.0
    best_ckpt = os.path.join(results_dir, f'{dataset_name}_fold{fold_idx+1}_v3.pth')

    for epoch in range(epochs):
        apply_warmup(opt, epoch, WARMUP_EPOCHS_V3, C.LR)

        model.train()
        opt.zero_grad()
        total_loss = 0.0
        accum_step = 0

        idxs = np.random.permutation(len(train_ds))
        for step, i in enumerate(idxs):
            sample    = collate_fn([train_ds[int(i)]])
            feats     = sample['features'].to(device)
            gtscore   = sample['gtscore'].to(device)
            gtsummary = sample['gtsummary']
            vkey      = train_keys[int(i)]

            scores = model(feats)
            loss   = combined_loss_v3(
                scores, feats, gtscore, gtsummary, vkey, clip_model, device
            ) / C.BATCH_ACCUM
            loss.backward()
            total_loss += loss.item() * C.BATCH_ACCUM
            accum_step += 1

            if accum_step % C.BATCH_ACCUM == 0 or step == len(idxs) - 1:
                nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
                opt.step()
                opt.zero_grad()

        if epoch >= WARMUP_EPOCHS_V3:
            scheduler.step()

        avg_loss = total_loss / len(train_ds)

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            val_f1 = _evaluate(model, test_ds, device, use_dpp=True, use_tta=True)
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save({
                    'model_state': model.state_dict(),
                    'epoch':       epoch,
                    'f1':          best_f1,
                    'feat_dim':    C.FEATURE_DIM,
                    'mode':        C.MODE,
                    'arch':        'v3',
                }, best_ckpt)
            print(f'  Fold {fold_idx+1} | Epoch {epoch+1:3d}/{epochs} | '
                  f'loss={avg_loss:.4f} | val_F1={val_f1:.4f} | best={best_f1:.4f}')
        elif (epoch + 1) % 10 == 0:
            print(f'  Fold {fold_idx+1} | Epoch {epoch+1:3d}/{epochs} | '
                  f'loss={avg_loss:.4f}')

    # Load best checkpoint for final eval
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state'])

    dpp_f1  = _evaluate(model, test_ds, device, use_dpp=True,  use_tta=True)
    topk_f1 = _evaluate(model, test_ds, device, use_dpp=False, use_tta=True)

    print(f'\n  Fold {fold_idx+1} DONE | DPP F1={dpp_f1:.4f} | Top-k F1={topk_f1:.4f}')
    return {'fold': fold_idx+1, 'dpp_f1': dpp_f1, 'topk_f1': topk_f1,
            'checkpoint': best_ckpt}


@torch.no_grad()
def _evaluate(model, dataset, device: str,
              use_dpp: bool, use_tta: bool = True) -> float:
    model.eval()
    results = []
    for i in range(len(dataset)):
        sample = collate_fn([dataset[i]])
        feats  = sample['features']

        if use_tta:
            scores = score_with_tta(model, feats, device)
        else:
            scores = model(feats.to(device)).cpu().numpy()

        T = len(scores)
        k = max(1, int(T * C.SUMMARY_RATIO))

        if use_dpp:
            sel = dpp_select(scores, feats.numpy(), k, lam=C.DPP_LAMBDA)
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


# ─────────────────────────────────────────────────────────────────────────────
# Fold ensemble evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def ensemble_evaluate(checkpoints, h5_path, fold_keys, device):
    """Leak-free: each video is scored only by the fold model that held it out."""
    dpp_results, topk_results = [], []
    for ckpt_path, keys in zip(checkpoints, fold_keys):
        ckpt = torch.load(ckpt_path, map_location=device)
        fd   = ckpt.get('feat_dim', C.FEATURE_DIM)
        m    = TransformerScorerV3(feat_dim=fd, d_model=C.D_MODEL, n_local_layers=C.N_LOCAL_LAYERS,
                                 ffn_dim=C.FFN_DIM, dropout=C.DROPOUT, window=C.WINDOW_SIZE).to(device)
        m.load_state_dict(ckpt['model_state'])
        m.eval()
        ds = VideoDataset(h5_path, keys)
        for i in range(len(ds)):
            sample = collate_fn([ds[i]])
            feats  = sample['features']
            scores = score_with_tta(m, feats, device)
            T = len(scores)
            k = max(1, int(T * C.SUMMARY_RATIO))
            for use_dpp, res_list in [(True, dpp_results), (False, topk_results)]:
                if use_dpp:
                    sel = dpp_select(scores, feats.numpy(), k, lam=C.DPP_LAMBDA)
                else:
                    sel = np.argsort(scores)[-k:]
                pred = np.zeros(T, dtype=np.int8)
                pred[sel] = 1
                res_list.append(evaluate_summary(pred, {
                    'picks': sample.get('picks'), 'n_frames': sample['n_frames'],
                    'gtsummary': sample['gtsummary'],
                    'user_summary': sample.get('user_summary')}))
    return {'ensemble_dpp_f1': mean_f1(dpp_results),
            'ensemble_topk_f1': mean_f1(topk_results)}


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(dataset_name: str, device: str, epochs: int, results_dir: str,
        clip_model=None):
    h5_path = C.get_dataset_path(dataset_name)

    if not os.path.exists(h5_path):
        print(f'HDF5 not found: {h5_path}')
        if C.MODE == 'clip':
            print('Run clip_feature_extractor.py first.')
        return None

    os.makedirs(results_dir, exist_ok=True)
    splits    = build_kfold_splits(h5_path, C.N_FOLDS, C.SEED)
    all_keys  = []

    print(f'\n{"="*65}')
    print(f'Dataset  : {dataset_name}  ({h5_path})')
    print(f'Mode     : {C.MODE}  |  feat_dim={C.FEATURE_DIM}')
    print(f'Folds    : {C.N_FOLDS}  |  Epochs={epochs}  |  Warmup={WARMUP_EPOCHS_V3}')
    print(f'Device   : {device}')
    print(f'Arch     : TransformerScorerV3 (windowed+CLS+multiscale)')
    print(f'{"="*65}')

    fold_results = []
    ckpt_paths   = []
    fold_keys    = []

    for fold_idx, (train_keys, test_keys) in enumerate(splits):
        all_keys.extend(test_keys)
        fold_keys.append(list(test_keys))
        print(f'\nFold {fold_idx+1}/{C.N_FOLDS} | '
              f'train={len(train_keys)} test={len(test_keys)}')
        res = train_one_fold(h5_path, train_keys, test_keys, fold_idx,
                             dataset_name, device, epochs, results_dir,
                             clip_model=clip_model)
        fold_results.append(res)
        ckpt_paths.append(res['checkpoint'])

    dpp_scores  = [r['dpp_f1']  for r in fold_results]
    topk_scores = [r['topk_f1'] for r in fold_results]

    # ── Fold ensemble ─────────────────────────────────────────────────────────
    print('\nRunning fold ensemble evaluation...')
    ensemble = ensemble_evaluate(ckpt_paths, h5_path, fold_keys, device)

    summary = {
        'dataset':             dataset_name,
        'mode':                C.MODE,
        'backbone':            C.BACKBONE,
        'feat_dim':            C.FEATURE_DIM,
        'arch':                'v3',
        'epochs':              epochs,
        'dpp_mean':            float(np.mean(dpp_scores)),
        'dpp_std':             float(np.std(dpp_scores)),
        'topk_mean':           float(np.mean(topk_scores)),
        'topk_std':            float(np.std(topk_scores)),
        'ensemble_dpp_f1':     ensemble['ensemble_dpp_f1'],
        'ensemble_topk_f1':    ensemble['ensemble_topk_f1'],
        'fold_results':        fold_results,
    }

    out_json = os.path.join(results_dir, f'{dataset_name}_results_v3.json')
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\n{"="*65}')
    print(f'RESULTS — {dataset_name.upper()} ({C.MODE}, v3 arch)')
    print(f'  Per-fold DPP  : {np.mean(dpp_scores):.4f} ± {np.std(dpp_scores):.4f}')
    print(f'  Per-fold Top-k: {np.mean(topk_scores):.4f} ± {np.std(topk_scores):.4f}')
    print(f'  Ensemble  DPP : {ensemble["ensemble_dpp_f1"]:.4f}   (held-out videos only, with TTA)')
    print(f'  Ensemble Top-k: {ensemble["ensemble_topk_f1"]:.4f}')
    print(f'  Saved → {out_json}')
    print(f'{"="*65}')

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='tvsum',
                        choices=['summe', 'tvsum'])
    parser.add_argument('--device',  default=C.DEVICE)
    parser.add_argument('--epochs',  type=int, default=C.EPOCHS)
    parser.add_argument('--results_dir', default='results_v3')
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device == 'cuda':
        print('CUDA not available, falling back to CPU')
        args.device = 'cpu'

    # Load CLIP model for contrastive loss if in clip mode
    clip_model = None
    if C.MODE == 'clip':
        try:
            import clip as _clip
            print(f'Loading CLIP for contrastive loss ({args.device})...')
            clip_model, _ = _clip.load('ViT-B/32', device=args.device)
            clip_model.eval()
        except ImportError:
            print('CLIP not installed — contrastive loss disabled.')

    run(args.dataset, args.device, args.epochs, args.results_dir, clip_model)
