"""
clip_feature_extractor.py — Extract CLIP ViT-B/32 features from raw videos
===========================================================================
Produces an HDF5 file with the same schema as the eccv16 google_pool5 files
so train_v2.py can load it without modification.

CLIP ViT-B/32 gives 512-d features that:
  - Encode rich semantic content (objects, scenes, actions, text)
  - Are pre-trained on 400M image-text pairs
  - Consistently boost F1 by 8-15 points over GoogLeNet on SumMe/TVSum

Usage:
    # Single video
    python clip_feature_extractor.py \
        --video  raw_videos/tvsum/video1.mp4 \
        --out    data/tvsum_clip.h5

    # Entire folder → one HDF5 (recommended for training)
    python clip_feature_extractor.py \
        --folder raw_videos/tvsum/ \
        --out    data/tvsum_clip.h5

    # Merge with existing eccv16 HDF5 to copy gtscore/gtsummary/user_summary
    python clip_feature_extractor.py \
        --folder raw_videos/tvsum/ \
        --out    data/tvsum_clip.h5 \
        --ref_h5 data/tvsum.h5

Requirements:
    pip install git+https://github.com/openai/CLIP.git
    pip install imageio[ffmpeg] av
"""

import os, argparse, glob
import numpy as np
import torch
import h5py
from tqdm import tqdm
import config_v2 as C

# Lazy CLIP import so the file is importable without CLIP installed
def _load_clip():
    try:
        import clip
        return clip
    except ImportError:
        raise ImportError(
            'CLIP not installed. Run:\n'
            '  pip install git+https://github.com/openai/CLIP.git'
        )


# ── Frame reader ─────────────────────────────────────────────────────────────
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
        raise RuntimeError(f'Cannot open {video_path}. Install imageio[ffmpeg] av.')
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


def extract_frames(video_path: str, fps: int = 2):
    try:
        import imageio.v3, av  # noqa: F401
        yield from _frames_imageio(video_path, fps)
    except ImportError:
        yield from _frames_opencv(video_path, fps)


# ── Main extraction ───────────────────────────────────────────────────────────
@torch.no_grad()
def extract_clip_features(
        video_path: str,
        model,
        preprocess,
        device: str,
        fps: int = 2,
        batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        features : (T, 512)  float32, L2-normalised
        picks    : (T,)      int32    original frame indices
    """
    from PIL import Image

    frames_pil, indices = [], []
    for orig_idx, rgb in extract_frames(video_path, fps):
        frames_pil.append(Image.fromarray(rgb))
        indices.append(orig_idx)

    if not frames_pil:
        raise RuntimeError(f'No frames extracted from {video_path}')

    feats = []
    for i in range(0, len(frames_pil), batch_size):
        batch = torch.stack([preprocess(f) for f in frames_pil[i:i+batch_size]]).to(device)
        enc = model.encode_image(batch).float()               # (B, 512)
        enc = enc / (enc.norm(dim=1, keepdim=True) + 1e-8)   # L2 normalise
        feats.append(enc.cpu().numpy())

    features = np.concatenate(feats, axis=0).astype(np.float32)
    picks    = np.array(indices, dtype=np.int32)
    return features, picks


def process_video(
        video_path: str,
        out_h5: str,
        model,
        preprocess,
        device: str,
        fps: int,
        ref_h5: str | None,
) -> None:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f'\n  [{video_name}]  {video_path}')

    features, picks = extract_clip_features(video_path, model, preprocess, device, fps)
    n_frames = int(picks[-1]) + 1

    with h5py.File(out_h5, 'a') as f:
        if video_name in f:
            del f[video_name]
        grp = f.create_group(video_name)
        grp.create_dataset('features', data=features, compression='gzip')
        grp.create_dataset('picks',    data=picks)
        grp.create_dataset('n_frames', data=np.int32(n_frames))
        grp.attrs['n_steps']  = len(picks)
        grp.attrs['feat_dim'] = 512
        grp.attrs['backbone'] = 'clip_vitb32'

        # Copy ground-truth annotations from reference HDF5 if provided
        if ref_h5 and os.path.exists(ref_h5):
            with h5py.File(ref_h5, 'r') as ref:
                # Try exact name match first, then case-insensitive
                ref_key = None
                if video_name in ref:
                    ref_key = video_name
                else:
                    for k in ref.keys():
                        if k.lower() == video_name.lower():
                            ref_key = k
                            break

                if ref_key:
                    for field in ('gtscore', 'gtsummary', 'user_summary',
                                  'change_points', 'n_frame_per_seg'):
                        if field in ref[ref_key]:
                            grp.create_dataset(field, data=ref[ref_key][field][:])
                    print(f'    Annotations copied from ref HDF5 key "{ref_key}"')
                else:
                    print(f'    WARNING: "{video_name}" not found in ref HDF5 '
                          f'— annotations not copied')

    print(f'    Saved {len(picks)} frames × 512-d  (n_frames={n_frames})')


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract CLIP ViT-B/32 features from video(s) → HDF5')
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--video',  help='Single video file path')
    src.add_argument('--folder', help='Folder of video files (processes all .mp4/.avi/.mkv)')
    parser.add_argument('--out',     required=True, help='Output HDF5 path')
    parser.add_argument('--ref_h5',  default=None,
                        help='Reference HDF5 (e.g. data/tvsum.h5) to copy annotations from')
    parser.add_argument('--fps',     type=int, default=2)
    parser.add_argument('--device',  default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch',   type=int, default=64, help='Frames per CLIP batch')
    args = parser.parse_args()

    clip = _load_clip()
    print(f'Loading CLIP ViT-B/32 on {args.device}...')
    model, preprocess = clip.load('ViT-B/32', device=args.device)
    model.eval()
    print('CLIP loaded.')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    if args.video:
        videos = [args.video]
    else:
        exts = ('*.mp4', '*.MP4', '*.avi', '*.AVI', '*.mkv', '*.MKV', '*.mov')
        videos = []
        for ext in exts:
            videos.extend(glob.glob(os.path.join(args.folder, ext)))
        videos = sorted(set(videos))
        print(f'Found {len(videos)} videos in {args.folder}')

    for vp in tqdm(videos, desc='Videos', unit='video'):
        try:
            process_video(vp, args.out, model, preprocess,
                          args.device, args.fps, args.ref_h5)
        except Exception as e:
            print(f'  ERROR on {vp}: {e}')

    print(f'\nDone. Output: {args.out}')
    with h5py.File(args.out, 'r') as f:
        print(f'Keys in HDF5: {list(f.keys())}')
