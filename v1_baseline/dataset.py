"""
Dataset loader for the standard SumMe / TVSum / VSUMM HDF5 format.

Expected HDF5 schema (used by DR-DSN, VASNet, and related works):
    /<video_name>/
        features      : (n_steps, feat_dim)  float32 — sampled frame features
        gtscore       : (n_steps,)           float32 — continuous importance scores
        gtsummary     : (n_steps,)           int8    — binary ground-truth summary
        user_summary  : (n_users, n_frames)  int8    — per-annotator binary summaries
        change_points : (n_segs, 2)          int32   — shot boundary [start, end)
        n_frame_per_seg : (n_segs,)          int32
        n_frames      : scalar               int     — total frames in original video
        picks         : (n_steps,)           int32   — sampled frame indices

Public download:
    https://github.com/KaiyangZhou/pytorch-vsumm-reinforce  (SumMe, TVSum)
    https://github.com/e-apostolidis/PGL-SUM               (SumMe, TVSum, OVP, YouTube)
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    """
    One item = one video.
    Returns a dict with features, scores, binary ground-truth, and metadata.
    """

    def __init__(self, h5_path: str, video_names: list[str] | None = None):
        """
        Args:
            h5_path     : path to the HDF5 file
            video_names : subset of video keys to load (None = all)
        """
        self.h5_path = h5_path
        with h5py.File(h5_path, 'r') as f:
            all_keys = list(f.keys())
        self.video_names = video_names if video_names is not None else all_keys

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx: int) -> dict:
        name = self.video_names[idx]
        with h5py.File(self.h5_path, 'r') as f:
            grp = f[name]
            features     = torch.from_numpy(grp['features'][:].astype(np.float32))
            gtscore      = torch.from_numpy(grp['gtscore'][:].astype(np.float32))
            gtsummary    = grp['gtsummary'][:].astype(np.int8)
            user_summary = grp['user_summary'][:].astype(np.int8)  if 'user_summary' in grp else None
            picks        = grp['picks'][:].astype(np.int32)         if 'picks'        in grp else None
            # n_frames is a dataset (scalar array) in the standard HDF5 format,
            # NOT an attribute — read it as a dataset first, fall back to attr.
            if 'n_frames' in grp:
                n_frames = int(grp['n_frames'][()])
            else:
                n_frames = int(grp.attrs.get('n_frames', features.shape[0]))

        # Normalise importance scores to [0, 1]
        mn, mx = gtscore.min(), gtscore.max()
        if mx > mn:
            gtscore = (gtscore - mn) / (mx - mn)

        return {
            'name':         name,
            'features':     features,           # (T, D)
            'gtscore':      gtscore,            # (T,)
            'gtsummary':    gtsummary,          # (T,)  binary
            'user_summary': user_summary,       # (n_users, n_frames) | None
            'picks':        picks,              # (T,)  original frame indices
            'n_frames':     n_frames,
        }


def collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate: keeps variable-length sequences as a list, not a padded tensor.
    Suitable for batch_size=1 (standard for video summarization training).
    """
    assert len(batch) == 1, 'Only batch_size=1 supported (variable-length sequences)'
    return batch[0]


def build_kfold_splits(h5_path: str, n_folds: int = 5,
                       seed: int = 42) -> list[tuple[list, list]]:
    """
    Returns a list of (train_names, test_names) tuples for k-fold CV.
    Splits are reproducible given the same seed.
    """
    with h5py.File(h5_path, 'r') as f:
        names = list(f.keys())

    rng = np.random.default_rng(seed)
    names = rng.permutation(names).tolist()

    fold_size = len(names) // n_folds
    splits = []
    for i in range(n_folds):
        test  = names[i * fold_size : (i + 1) * fold_size]
        train = names[:i * fold_size] + names[(i + 1) * fold_size:]
        splits.append((train, test))
    return splits


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys, os
    h5 = sys.argv[1] if len(sys.argv) > 1 else 'data/summe.h5'
    if not os.path.exists(h5):
        print(f'HDF5 not found: {h5}')
        print('Download from: https://github.com/KaiyangZhou/pytorch-vsumm-reinforce')
    else:
        ds = VideoDataset(h5)
        print(f'Dataset: {h5}   Videos: {len(ds)}')
        sample = ds[0]
        print(f'  name:      {sample["name"]}')
        print(f'  features:  {sample["features"].shape}')
        print(f'  gtscore:   {sample["gtscore"].shape}')
        print(f'  gtsummary: {sample["gtsummary"].shape}')

        splits = build_kfold_splits(h5)
        print(f'\n5-fold splits:')
        for i, (tr, te) in enumerate(splits):
            print(f'  Fold {i+1}: train={len(tr)}  test={len(te)}')
