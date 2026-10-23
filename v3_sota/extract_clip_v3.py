"""
extract_clip_v3.py — Extract CLIP features (ViT-B/32 or ViT-L/14) from raw videos
====================================================================================
Produces an HDF5 file with the same schema as the eccv16 google_pool5 files.
Ground-truth annotations (gtscore, gtsummary, user_summary, change_points)
are copied from the reference eccv16 HDF5 so training works without modification.

Backbone choice:
    ViT-L/14 (768-d) → best results, recommended  ← set BACKBONE='vitl14' in config_v3.py
    ViT-B/32 (512-d) → good results, faster        ← set BACKBONE='vitb32' in config_v3.py

Usage:
    # Extract TVSum with ViT-L/14 (reads backbone from config_v3.py)
    python extract_clip_v3.py --dataset tvsum --device cuda

    # Extract both datasets
    python extract_clip_v3.py --dataset tvsum summe --device cuda

    # Override backbone from CLI without editing config
    python extract_clip_v3.py --dataset tvsum --backbone vitb32 --device cuda

Requirements:
    pip install git+https://github.com/openai/CLIP.git
    pip install imageio[ffmpeg] av tqdm h5py
"""

import os, glob, argparse
import numpy as np
import torch
import h5py
from tqdm import tqdm
from PIL import Image

import config_v3 as C


# ─────────────────────────────────────────────────────────────────────────────
# Frame reader — imageio primary, OpenCV fallback
# ─────────────────────────────────────────────────────────────────────────────

def _frames_imageio(video_path: str, fps: int):
    import imageio.v3 as iio
    meta = iio.immeta(video_path, plugin='pyav')
    native_fps = float(meta.get('fps', 25))
    step = max(1, round(native_fps / fps))
    for orig_idx, frame in enumerate(iio.imiter(video_path, plugin='pyav')):
        if orig_idx % step == 0:
            yield orig_idx, frame   # (H, W, 3) uint8 RGB


def _frames_opencv(video_path: str, fps: int):
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open: {video_path}')
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(native_fps / fps))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            yield idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        idx += 1
    cap.release()


def iter_frames(video_path: str, fps: int = 2):
    """Yield (original_frame_index, RGB_array) at ~fps frames per second."""
    try:
        import imageio.v3, av   # noqa: F401
        yield from _frames_imageio(video_path, fps)
    except (ImportError, Exception):
        yield from _frames_opencv(video_path, fps)


# ─────────────────────────────────────────────────────────────────────────────
# CLIP feature extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_clip_features(
    video_path: str,
    clip_model,
    preprocess,
    device:     str,
    fps:        int = 2,
    batch_size: int = 64,
) -> tuple:
    """
    Extract CLIP visual features for every sampled frame.

    Returns:
        features : (T, D)  float32, L2-normalised
        picks    : (T,)    int32, original frame indices
    """
    frames_pil, indices = [], []
    for orig_idx, rgb in iter_frames(video_path, fps):
        frames_pil.append(Image.fromarray(rgb))
        indices.append(orig_idx)

    if not frames_pil:
        raise RuntimeError(f'No frames extracted from {video_path}')

    feats_list = []
    for i in range(0, len(frames_pil), batch_size):
        batch = torch.stack(
            [preprocess(f) for f in frames_pil[i:i+batch_size]]
        ).to(device)
        enc = clip_model.encode_image(batch).float()          # (B, D)
        enc = enc / (enc.norm(dim=1, keepdim=True) + 1e-8)   # L2-normalise
        feats_list.append(enc.cpu().numpy())

    features = np.concatenate(feats_list, axis=0).astype(np.float32)
    picks    = np.array(indices, dtype=np.int32)
    return features, picks


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 writer — copies annotations from reference file
# ─────────────────────────────────────────────────────────────────────────────

def _copy_annotations(src_grp, dst_grp, new_picks=None):
    """Copy GT; resample gtscore/gtsummary to the new frame sampling."""
    ref_picks = src_grp['picks'][:] if 'picks' in src_grp else None
    for field in ('gtscore', 'gtsummary', 'user_summary',
                  'change_points', 'n_frame_per_seg'):
        if field not in src_grp:
            continue
        data = src_grp[field][:]
        if (field in ('gtscore', 'gtsummary') and new_picks is not None
                and ref_picks is not None and len(ref_picks) == len(data)
                and (len(ref_picks) != len(new_picks)
                     or not np.array_equal(ref_picks, new_picks))):
            idx  = np.abs(ref_picks[None, :] - new_picks[:, None]).argmin(axis=1)
            data = data[idx]
        dst_grp.create_dataset(field, data=data)


def _find_ref_key(ref_h5: h5py.File, video_name: str):
    """Case-insensitive key lookup in reference HDF5."""
    if video_name in ref_h5:
        return video_name
    for k in ref_h5.keys():
        if k.lower() == video_name.lower():
            return k
    return None


def save_features_to_h5(
    video_path:  str,
    features:    np.ndarray,
    picks:       np.ndarray,
    out_h5:      str,
    backbone:    str,
    ref_h5_path: str | None = None,
):
    """Append one video's features to an HDF5 file, copying GT annotations."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    n_frames   = int(picks[-1]) + 1

    with h5py.File(out_h5, 'a') as f:
        if video_name in f:
            del f[video_name]
        grp = f.create_group(video_name)
        grp.create_dataset('features', data=features, compression='gzip')
        grp.create_dataset('picks',    data=picks)
        grp.create_dataset('n_frames', data=np.int32(n_frames))
        grp.attrs['n_steps']  = len(picks)
        grp.attrs['feat_dim'] = features.shape[1]
        grp.attrs['backbone'] = backbone

        # Copy ground-truth annotations
        if ref_h5_path and os.path.exists(ref_h5_path):
            with h5py.File(ref_h5_path, 'r') as ref:
                ref_key = _find_ref_key(ref, video_name)
                if ref_key:
                    _copy_annotations(ref[ref_key], grp, picks)
                    if 'n_frames' in ref[ref_key]:
                        del grp['n_frames']
                        grp.create_dataset('n_frames', data=np.int32(ref[ref_key]['n_frames'][()]))
                    print(f'    GT annotations copied from ref key "{ref_key}"')
                else:
                    print(f'    WARNING: "{video_name}" not in ref HDF5 — no annotations')

    feat_dim = features.shape[1]
    print(f'    Saved {len(picks)} frames × {feat_dim}-d  '
          f'(n_frames={n_frames}, backbone={backbone})')


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-level processing
# ─────────────────────────────────────────────────────────────────────────────

def process_dataset(
    dataset:    str,
    backbone:   str,
    device:     str,
    fps:        int = 2,
    batch_size: int = 64,
) -> str | None:
    """
    Extract CLIP features for all videos in a dataset.
    Returns the output HDF5 path, or None on failure.
    """
    raw_dir    = C.RAW_VIDEO_DIRS.get(dataset)
    out_h5     = C.DATASET_H5[backbone][dataset]
    ref_h5     = C.REFERENCE_H5.get(dataset)
    clip_name  = C.get_clip_model_name() if backbone != 'googlenet' \
                 else {'vitl14': 'ViT-L/14', 'vitb32': 'ViT-B/32'}.get(backbone, 'ViT-B/32')

    # Resolve backbone-specific clip name
    _clip_name_map = {'vitl14': 'ViT-L/14', 'vitb32': 'ViT-B/32'}
    clip_name = _clip_name_map.get(backbone, 'ViT-B/32')

    print(f'\n{"="*60}')
    print(f'Dataset  : {dataset}')
    print(f'Backbone : {backbone}  ({clip_name})')
    print(f'Raw dir  : {raw_dir}')
    print(f'Output   : {out_h5}')
    print(f'Ref HDF5 : {ref_h5}')
    print(f'{"="*60}')

    if not raw_dir or not os.path.isdir(raw_dir):
        print(f'ERROR: Raw video directory not found: {raw_dir}')
        print(f'  Place your {dataset} videos in: {raw_dir}/')
        return None

    # Find all video files
    exts   = ('*.mp4', '*.MP4', '*.avi', '*.AVI', '*.mkv', '*.MKV', '*.mov', '*.MOV')
    videos = []
    for ext in exts:
        videos.extend(glob.glob(os.path.join(raw_dir, ext)))
    videos = sorted(set(videos))

    if not videos:
        print(f'ERROR: No video files found in {raw_dir}')
        return None

    print(f'Found {len(videos)} video files.')

    # Load CLIP
    try:
        import clip as _clip
    except ImportError:
        raise ImportError(
            'CLIP not installed.\n'
            'Run: pip install git+https://github.com/openai/CLIP.git'
        )

    print(f'Loading CLIP {clip_name} on {device}...')
    clip_model, preprocess = _clip.load(clip_name, device=device)
    clip_model.eval()
    print(f'CLIP loaded.')

    os.makedirs(os.path.dirname(os.path.abspath(out_h5)), exist_ok=True)

    failed = []
    for vp in tqdm(videos, desc=f'{dataset} extraction', unit='video'):
        vname = os.path.basename(vp)
        print(f'\n  [{vname}]')
        try:
            features, picks = extract_clip_features(
                vp, clip_model, preprocess, device, fps, batch_size)
            save_features_to_h5(vp, features, picks, out_h5, backbone, ref_h5)
        except Exception as e:
            print(f'    ERROR: {e}')
            failed.append(vp)

    # Summary
    with h5py.File(out_h5, 'r') as f:
        keys = list(f.keys())

    print(f'\nDone. {len(keys)} videos in {out_h5}')
    if failed:
        print(f'Failed ({len(failed)}): {[os.path.basename(v) for v in failed]}')

    return out_h5


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract CLIP features from raw videos → HDF5')
    parser.add_argument('--dataset',  nargs='+', default=['tvsum', 'summe'],
                        choices=['tvsum', 'summe'],
                        help='Dataset(s) to extract')
    parser.add_argument('--backbone', default=None,
                        choices=['vitl14', 'vitb32'],
                        help='Override backbone from config_v3.py')
    parser.add_argument('--device',   default=C.DEVICE)
    parser.add_argument('--fps',      type=int, default=2)
    parser.add_argument('--batch',    type=int, default=64,
                        help='Frames per CLIP forward pass (reduce if OOM)')
    args = parser.parse_args()

    backbone = args.backbone or C.BACKBONE
    if backbone == 'googlenet':
        print('ERROR: This script is for CLIP only. '
              'Set backbone to vitl14 or vitb32.')
        raise SystemExit(1)

    print(f'Backbone : {backbone}')
    print(f'Device   : {args.device}')
    print(f'FPS      : {args.fps}')

    for dataset in args.dataset:
        process_dataset(dataset, backbone, args.device, args.fps, args.batch)
