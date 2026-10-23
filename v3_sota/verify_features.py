"""
verify_features.py — Check whether your extracted features are compatible
with the training features in the HDF5 file.

Compares mean, std, norm, and cosine similarity distribution between:
  (a) features stored in the training HDF5 (eccv16 google_pool5)
  (b) features you extracted with feature_extractor.py

Usage:
    python verify_features.py \
        --train_h5  data/tvsum.h5 \
        --infer_h5  data/tvsum_test.h5 \
        --infer_key tvsum_test

If mean/std/norm are within ~10% of the training distribution, you're compatible.
"""

import argparse
import numpy as np
import h5py


def stats(feats: np.ndarray, label: str):
    """Print summary statistics for a feature matrix (T, D)."""
    norms = np.linalg.norm(feats, axis=1)
    print(f'\n  [{label}]  shape={feats.shape}')
    print(f'    Mean (per-dim avg) : {feats.mean():.4f}')
    print(f'    Std  (per-dim avg) : {feats.std():.4f}')
    print(f'    L2 norm  — mean    : {norms.mean():.4f}')
    print(f'    L2 norm  — std     : {norms.std():.4f}')
    print(f'    Min / Max value    : {feats.min():.4f} / {feats.max():.4f}')
    print(f'    Fraction zeros     : {(feats == 0).mean():.4f}  '
          f'(pool5 after ReLU + L2-norm: expect ~0.4–0.6 zeros, norm≈1.0)')
    return feats.mean(), feats.std(), norms.mean()


def cosine_sim_sample(a: np.ndarray, b: np.ndarray, n: int = 500) -> float:
    """Mean pairwise cosine similarity between random rows of a and b."""
    rng = np.random.default_rng(0)
    ia  = rng.choice(len(a), min(n, len(a)), replace=False)
    ib  = rng.choice(len(b), min(n, len(b)), replace=False)
    fa  = a[ia] / (np.linalg.norm(a[ia], axis=1, keepdims=True) + 1e-8)
    fb  = b[ib] / (np.linalg.norm(b[ib], axis=1, keepdims=True) + 1e-8)
    # mean cosine sim between all pairs
    sims = fa @ fb.T
    return float(sims.mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_h5',  required=True,
                        help='Path to training HDF5 (e.g. data/tvsum.h5)')
    parser.add_argument('--infer_h5',  required=True,
                        help='Path to inference HDF5 produced by feature_extractor.py')
    parser.add_argument('--infer_key', required=True,
                        help='Video key inside the inference HDF5')
    parser.add_argument('--train_key', default=None,
                        help='Video key inside training HDF5 (default: first key)')
    args = parser.parse_args()

    # Load training features (one video as representative sample)
    with h5py.File(args.train_h5, 'r') as f:
        keys = list(f.keys())
        key  = args.train_key or keys[0]
        train_feats = f[key]['features'][:].astype(np.float32)
        print(f'Training HDF5  : {args.train_h5}')
        print(f'Sample video   : {key}  ({len(keys)} videos total)')

    # Load inference features
    with h5py.File(args.infer_h5, 'r') as f:
        infer_feats = f[args.infer_key]['features'][:].astype(np.float32)
        print(f'\nInference HDF5 : {args.infer_h5}')
        print(f'Video key      : {args.infer_key}')

    # Print stats
    tm, ts, tn = stats(train_feats,  'Training  (HDF5)')
    im, is_, in_ = stats(infer_feats, 'Inference (extracted)')

    # Compatibility verdict
    print('\n' + '='*55)
    print('COMPATIBILITY CHECK')
    print('='*55)
    mean_diff = abs(tm - im) / (abs(tm) + 1e-8)
    std_diff  = abs(ts - is_) / (abs(ts) + 1e-8)
    norm_diff = abs(tn - in_) / (abs(tn) + 1e-8)

    cross_sim = cosine_sim_sample(train_feats, infer_feats)
    within_sim = cosine_sim_sample(train_feats, train_feats)

    print(f'  Mean  relative diff : {mean_diff*100:.1f}%  (OK if < 15%)')
    print(f'  Std   relative diff : {std_diff*100:.1f}%  (OK if < 15%)')
    print(f'  Norm  relative diff : {norm_diff*100:.1f}%  (OK if < 15%)')
    print(f'  Cross-set cosine sim : {cross_sim:.4f}')
    print(f'  Within-train cos sim : {within_sim:.4f}')
    print()

    ok = mean_diff < 0.15 and std_diff < 0.15 and norm_diff < 0.15
    if ok:
        print('  RESULT: COMPATIBLE — features match the training distribution.')
        print('  Inference results should be valid.')
    else:
        print('  RESULT: MISMATCH DETECTED — features differ from training.')
        print('  This may degrade inference quality.')
        print('  Check that both use GoogLeNet pool5 (1024-d) with the same')
        print('  preprocessing (224x224, ImageNet normalisation).')
    print('='*55)
