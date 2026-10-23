"""
inference_v2.py — Inference with v2 checkpoints (CLIP or GoogLeNet features)
=============================================================================
Automatically detects whether the checkpoint used CLIP or GoogLeNet features
from the saved metadata and loads the correct model configuration.

Usage:
    # With pre-extracted HDF5 features
    python inference_v2.py \
        --h5        data/tvsum_clip.h5 \
        --video_key video1 \
        --checkpoint results_v2/tvsum_fold1_v2.pth \
        --out_dir   output/keyframes_v2/

    # With raw video (extracts CLIP features on the fly)
    python inference_v2.py \
        --video      my_video.mp4 \
        --checkpoint results_v2/tvsum_fold1_v2.pth \
        --out_dir    output/keyframes_v2/
"""

import os, argparse, tempfile
import numpy as np
import torch
import h5py
from PIL import Image

import config_v2 as C
from model_v2    import TransformerScorerV2
from dpp_sampler import dpp_select
from evaluate    import upsample_summary


@torch.no_grad()
def run_inference(
    features:    np.ndarray,    # (T, D)
    picks:       np.ndarray,    # (T,)  original frame indices
    n_frames:    int,
    checkpoint:  str,
    out_dir:     str,
    ratio:       float = 0.15,
    use_dpp:     bool  = True,
    device:      str   = 'cpu',
    source_video: str | None = None,  # path to raw video for saving actual frames
) -> dict:
    """
    Run the v2 model on pre-extracted features and save keyframes.
    Returns dict with selected indices, scores, and F1 info.
    """
    # Load checkpoint metadata
    ckpt = torch.load(checkpoint, map_location=device)
    feat_dim = ckpt.get('feat_dim', C.FEATURE_DIM)
    mode     = ckpt.get('mode', C.MODE)
    epoch    = ckpt.get('epoch', '?')
    ckpt_f1  = ckpt.get('f1', '?')

    print(f'Checkpoint: {checkpoint}')
    print(f'  Mode: {mode}  |  feat_dim: {feat_dim}  |  '
          f'best epoch: {epoch}  |  val F1: {ckpt_f1}')

    model = TransformerScorerV2(feat_dim=feat_dim).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    T  = features.shape[0]
    x  = torch.from_numpy(features).to(device)
    scores = model(x).cpu().numpy()   # (T,)

    k = max(1, int(T * ratio))
    print(f'Features: ({T}, {feat_dim})  |  Selecting top {k}/{T} frames ({ratio*100:.0f}%)')

    if use_dpp:
        sel = dpp_select(scores, features, k, lam=C.DPP_LAMBDA)
        method = 'DPP'
    else:
        sel = np.argsort(scores)[-k:]
        method = 'Top-k'

    sel = np.sort(sel)
    print(f'Method: {method}  |  Selected {len(sel)} frames')

    # ── Save keyframes ────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)

    if source_video is not None:
        # Extract actual video frames for selected picks
        _save_video_frames(source_video, picks[sel], out_dir)
    else:
        print(f'No source video provided — saving frame index info only.')
        # Save a text file with selected frame indices
        idx_path = os.path.join(out_dir, 'selected_frames.txt')
        with open(idx_path, 'w') as f:
            for i, orig_idx in enumerate(picks[sel]):
                f.write(f'{i+1:04d}\t{orig_idx}\t{scores[sel[i]]:.4f}\n')
        print(f'Saved frame index list → {idx_path}')

    # Save scores for all frames
    scores_path = os.path.join(out_dir, 'frame_scores.npy')
    np.save(scores_path, scores)

    return {
        'n_selected':   len(sel),
        'n_total':      T,
        'selected_original_frames': picks[sel].tolist(),
        'method':       method,
        'ratio':        ratio,
    }


def _save_video_frames(video_path: str, frame_indices: np.ndarray, out_dir: str):
    """Save specific frames from a video as PNG files."""
    frame_set = set(frame_indices.tolist())
    saved = 0

    try:
        import imageio.v3 as iio
        for orig_idx, frame in enumerate(iio.imiter(video_path, plugin='pyav')):
            if orig_idx in frame_set:
                img = Image.fromarray(frame)
                img.save(os.path.join(out_dir, f'frame_{orig_idx:06d}.png'))
                saved += 1
                if saved == len(frame_indices):
                    break
    except ImportError:
        import cv2
        cap = cv2.VideoCapture(video_path)
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx in frame_set:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                Image.fromarray(rgb).save(
                    os.path.join(out_dir, f'frame_{idx:06d}.png'))
                saved += 1
                if saved == len(frame_indices):
                    break
            idx += 1
        cap.release()

    print(f'Saved {saved} keyframe images → {out_dir}/')


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--h5',       help='Pre-extracted feature HDF5')
    src.add_argument('--video',    help='Raw video (extracts CLIP features on the fly)')
    parser.add_argument('--video_key',   default=None,
                        help='Key inside HDF5 (default: first key)')
    parser.add_argument('--checkpoint',  required=True)
    parser.add_argument('--out_dir',     default='output/keyframes_v2/')
    parser.add_argument('--ratio',       type=float, default=0.15)
    parser.add_argument('--device',      default='cpu')
    parser.add_argument('--nodpp',       action='store_true',
                        help='Use Top-k instead of DPP')
    parser.add_argument('--source_video', default=None,
                        help='Raw video to extract keyframe images from '
                             '(when using --h5 mode)')
    args = parser.parse_args()

    if args.h5:
        # ── HDF5 mode ─────────────────────────────────────────────────────────
        with h5py.File(args.h5, 'r') as f:
            key = args.video_key or list(f.keys())[0]
            print(f'Loading features from {args.h5}[{key}]')
            features = f[key]['features'][:].astype(np.float32)
            picks    = f[key]['picks'][:].astype(np.int32) \
                       if 'picks' in f[key] else np.arange(len(features), dtype=np.int32)
            n_frames = int(f[key]['n_frames'][()]) \
                       if 'n_frames' in f[key] else int(picks[-1]) + 1

        result = run_inference(
            features, picks, n_frames,
            checkpoint=args.checkpoint,
            out_dir=args.out_dir,
            ratio=args.ratio,
            use_dpp=not args.nodpp,
            device=args.device,
            source_video=args.source_video,
        )

    else:
        # ── Raw video mode: extract CLIP on the fly ───────────────────────────
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        mode = ckpt.get('mode', 'clip')

        if mode == 'clip':
            try:
                import clip as _clip
            except ImportError:
                raise ImportError('pip install git+https://github.com/openai/CLIP.git')
            from clip_feature_extractor import extract_clip_features
            clip_model, preprocess = _clip.load('ViT-B/32', device=args.device)
            clip_model.eval()
            print('Extracting CLIP features from video...')
            features, picks = extract_clip_features(
                args.video, clip_model, preprocess, args.device)
        else:
            from feature_extractor import extract_features
            # Write to temp HDF5 then read back
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tf:
                tmp_path = tf.name
            extract_features(args.video, tmp_path, 'googlenet', 2, args.device)
            with h5py.File(tmp_path, 'r') as f:
                key      = list(f.keys())[0]
                features = f[key]['features'][:].astype(np.float32)
                picks    = f[key]['picks'][:].astype(np.int32)
            os.unlink(tmp_path)

        n_frames = int(picks[-1]) + 1
        result = run_inference(
            features, picks, n_frames,
            checkpoint=args.checkpoint,
            out_dir=args.out_dir,
            ratio=args.ratio,
            use_dpp=not args.nodpp,
            device=args.device,
            source_video=args.video,
        )

    print(f'\nInference complete:')
    print(f'  Selected {result["n_selected"]} / {result["n_total"]} frames')
    print(f'  Method : {result["method"]}')
    print(f'  Output : {args.out_dir}')
