"""
Step 5 – Inference on a New Video
====================================
Given a trained checkpoint and a video file (or pre-extracted HDF5),
outputs a list of selected keyframe indices and optionally extracts
the keyframe images to disk.

Usage:
    # From a video file (runs feature extraction first):
    python inference.py --video path/to/video.mp4 \
                        --ckpt  results/summe_fold1.pth \
                        --out   keyframes/

    # From a pre-extracted HDF5:
    python inference.py --h5    data/summe.h5 \
                        --name  video_name \
                        --ckpt  results/summe_fold1.pth

"""

import os, argparse
import numpy as np
import torch
import cv2
import h5py
from pathlib import Path

import config as C
from model        import TransformerScorer
from dpp_sampler  import select_keyframes
from feature_extractor import extract_features, TRANSFORM


@torch.no_grad()
def infer_from_features(features: np.ndarray,
                        ckpt_path: str,
                        device: torch.device,
                        ratio: float = C.SUMMARY_RATIO,
                        lam: float   = C.DPP_LAMBDA) -> np.ndarray:
    """
    Run the full inference pipeline on pre-extracted features.

    Args:
        features  : (T, D) float32 feature array
        ckpt_path : path to a .pth checkpoint saved by train.py
        device    : torch device
        ratio     : target summary length as fraction of T
        lam       : DPP quality weight

    Returns:
        keyframe_indices : 1D array of selected frame indices (in sampled-frame space)
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    model = TransformerScorer(
        feat_dim=ckpt['config'].get('FEATURE_DIM', C.FEATURE_DIM),
        d_model=ckpt['config'].get('D_MODEL',     C.D_MODEL),
        n_heads=ckpt['config'].get('N_HEADS',     C.N_HEADS),
        n_layers=ckpt['config'].get('N_LAYERS',   C.N_LAYERS),
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    feat_tensor = torch.from_numpy(features).unsqueeze(0).to(device)  # (1, T, D)
    scores = model(feat_tensor).squeeze(0).cpu().numpy()               # (T,)

    binary_summary = select_keyframes(scores, features, ratio=ratio, lam=lam)
    return np.where(binary_summary)[0]


def extract_keyframe_images(video_path: str,
                            frame_indices: np.ndarray,
                            picks: np.ndarray,
                            out_dir: str,
                            video_name: str = 'video'):
    """
    Extract keyframe images from the original video file.

    Args:
        video_path    : path to the original video
        frame_indices : selected indices in SAMPLED-frame space
        picks         : original frame indices of sampled frames
        out_dir       : directory to save keyframe PNGs
        video_name    : prefix for output filenames
    """
    os.makedirs(out_dir, exist_ok=True)
    original_indices = picks[frame_indices]  # map back to original frame space

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved = 0

    for orig_idx in sorted(original_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(orig_idx))
        ret, frame = cap.read()
        if not ret:
            continue
        ts_sec = orig_idx / max(cap.get(cv2.CAP_PROP_FPS), 1)
        out_path = os.path.join(out_dir, f'{video_name}_frame{orig_idx:06d}_{ts_sec:.1f}s.png')
        cv2.imwrite(out_path, frame)
        saved += 1

    cap.release()
    print(f'Saved {saved} keyframe images to {out_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default=None, help='Path to input video file')
    parser.add_argument('--h5',    default=None, help='Path to pre-extracted HDF5')
    parser.add_argument('--name',  default=None, help='Video name key inside HDF5')
    parser.add_argument('--ckpt',  required=True, help='Path to model checkpoint')
    parser.add_argument('--out',   default='keyframes', help='Output directory for images')
    parser.add_argument('--ratio', type=float, default=C.SUMMARY_RATIO)
    parser.add_argument('--lam',   type=float, default=C.DPP_LAMBDA)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.video:
        # Extract features on the fly
        import tempfile
        tmp_h5 = tempfile.mktemp(suffix='.h5')
        extract_features(args.video, tmp_h5, backbone=C.BACKBONE,
                         fps=C.FRAME_RATE, device=args.device)
        video_name = Path(args.video).stem
        with h5py.File(tmp_h5, 'r') as f:
            features = f[video_name]['features'][:]
            picks    = f[video_name]['picks'][:]
        os.remove(tmp_h5)

    elif args.h5 and args.name:
        with h5py.File(args.h5, 'r') as f:
            features = f[args.name]['features'][:]
            picks    = f[args.name]['picks'][:]
        video_name = args.name

    else:
        raise ValueError('Provide either --video or (--h5 + --name)')

    print(f'Video: {video_name}  |  Frames: {len(features)}  |  Feat dim: {features.shape[1]}')
    selected = infer_from_features(features, args.ckpt, device, args.ratio, args.lam)
    k = len(selected)
    total = len(features)
    print(f'Selected {k}/{total} frames ({100*k/total:.1f}%)')
    print(f'Selected sampled-frame indices: {selected.tolist()}')
    print(f'Original frame indices:         {picks[selected].tolist()}')

    if args.video:
        extract_keyframe_images(args.video, selected, picks, args.out, video_name)
