"""
Step 1 – Feature Extraction
============================
Extracts per-frame GoogLeNet pool5 (1024-d) features from a video file
and saves them to an HDF5 file compatible with the trained checkpoints.

IMPORTANT: The trained checkpoints use 1024-d GoogLeNet pool5 features,
matching the standard eccv16_dataset_*_google_pool5.h5 files.
Do NOT switch to ResNet50 (2048-d) unless you retrain from scratch.

NORMALISATION: The eccv16 HDF5 files store L2-normalised unit vectors
(norm ≈ 1.0, generated with Caffe GoogLeNet). Torchvision GoogLeNet
outputs raw pool5 features (norm ≈ 16.66). This script applies L2
normalisation automatically so extracted features match the training
distribution. Do NOT remove that step.

Video decoding uses imageio-ffmpeg (works on servers without OpenCV/FFmpeg
system libraries). Falls back to OpenCV if imageio is unavailable.

Usage:
    python feature_extractor.py --video path/to/video.mp4 --out data/my_video.h5

If you already have the standard SumMe/TVSum HDF5 files, skip this script.
"""

import os, argparse, h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from tqdm import tqdm
import config as C


# ── Caffe-compatible preprocessing ───────────────────────────────────────────
# The eccv16 google_pool5 HDF5 files were generated with BVLC GoogLeNet
# (Caffe) using:
#   1. Resize shorter side to 256, centre-crop to 224×224
#   2. Subtract BGR channel means [104.0, 117.0, 123.0]  (NOT divide by std)
#
# We replicate this in torchvision:
#   • Resize(256) + CenterCrop(224)  — matches Caffe's crop protocol
#   • Normalize with Caffe BGR means converted to RGB order,
#     std=[1,1,1] (no std division)  — replicates mean subtraction exactly
#
# After extraction we additionally L2-normalise each vector to unit norm,
# matching the per-frame normalisation applied before HDF5 storage.
TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize(256),                       # resize shorter side to 256
    T.CenterCrop(224),                   # centre-crop to 224×224
    T.ToTensor(),                        # [0,1] float, RGB order
    # Convert Caffe BGR means [104, 117, 123] → RGB order [123, 117, 104]
    # then divide by 255 to match ToTensor's [0,1] scale
    T.Normalize(mean=[123.0/255, 117.0/255, 104.0/255],
                std=[1.0,        1.0,        1.0]),
])


# ── Backbone builder ─────────────────────────────────────────────────────────
def build_backbone(name: str = 'googlenet') -> tuple[nn.Module, int]:
    """
    Return (feature_extractor_model, feature_dim).

    Backbone choices:
      'googlenet'       → 1024-d pool5  ← matches trained checkpoints
      'resnet50'        → 2048-d        ← only if you retrain
      'efficientnet_b3' → 1536-d        ← only if you retrain
    """
    name = name.lower()

    if name in ('googlenet', 'inception_v1'):
        # GoogLeNet pool5: remove the final classifier, keep avgpool output (1024-d)
        m = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1,
                             aux_logits=False)
        feat_dim = 1024
        # Strip the final FC layer; forward now returns the 1024-d pool5 vector
        m.fc = nn.Identity()
        m.eval()

    elif name == 'resnet50':
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        feat_dim = m.fc.in_features   # 2048
        m.fc = nn.Identity()
        m.eval()

    elif name == 'efficientnet_b3':
        m = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        feat_dim = m.classifier[1].in_features  # 1536
        m.classifier = nn.Identity()
        m.eval()

    else:
        raise ValueError(
            f'Unknown backbone: {name}. '
            'Choose "googlenet" (default), "resnet50", or "efficientnet_b3".')

    return m, feat_dim


# ── Video frame reader ───────────────────────────────────────────────────────
def _frames_imageio(video_path: str, fps: int):
    """
    Yield (original_frame_index, RGB_uint8_array) using imageio + ffmpeg.
    Works on servers where OpenCV has no video codec support.
    """
    import imageio.v3 as iio
    meta   = iio.immeta(video_path, plugin='pyav')
    native_fps = float(meta.get('fps', 25))
    step   = max(1, round(native_fps / fps))

    for orig_idx, frame in enumerate(iio.imiter(video_path, plugin='pyav')):
        # frame is (H, W, C) uint8 RGB
        if orig_idx % step == 0:
            yield orig_idx, frame


def _frames_opencv(video_path: str, fps: int):
    """
    Yield (original_frame_index, RGB_uint8_array) using OpenCV.
    Fallback when imageio is unavailable.
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(
            f'OpenCV cannot open {video_path}. '
            'Install imageio-ffmpeg instead: pip install imageio[ffmpeg] av')
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(native_fps / fps))
    idx  = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            yield idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        idx += 1
    cap.release()


def extract_frames(video_path: str, fps: int = 2):
    """
    Yield (original_frame_index, RGB_uint8_array).
    Tries imageio/pyav first (server-safe), falls back to OpenCV.
    """
    try:
        import imageio.v3  # noqa: F401
        import av          # noqa: F401
        yield from _frames_imageio(video_path, fps)
    except ImportError:
        print('[feature_extractor] imageio/av not found, falling back to OpenCV.')
        yield from _frames_opencv(video_path, fps)


# ── Main extraction function ─────────────────────────────────────────────────
@torch.no_grad()
def extract_features(video_path: str,
                     out_h5:     str,
                     backbone:   str = 'googlenet',
                     fps:        int = 2,
                     device:     str = 'cpu') -> None:
    """
    Extract features from a video and write them to HDF5.

    The output schema matches what the Transformer scorer and inference.py expect:
        <video_name>/features   : (T, D) float32
        <video_name>/picks      : (T,)   int32   — original frame indices
        <video_name>/n_frames   : scalar int
    """
    model, feat_dim = build_backbone(backbone)
    model = model.to(device)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames_list, indices = [], []

    print(f'Backbone : {backbone}  ({feat_dim}-d)')
    print(f'Video    : {video_path}')
    print(f'Sampling : {fps} fps')

    for orig_idx, rgb_frame in tqdm(extract_frames(video_path, fps),
                                     desc='Extracting'):
        tensor = TRANSFORM(rgb_frame).unsqueeze(0).to(device)
        feat   = model(tensor).squeeze(0).cpu().numpy()
        frames_list.append(feat)
        indices.append(orig_idx)

    if not frames_list:
        raise RuntimeError('No frames extracted — check video path and codec.')

    features = np.stack(frames_list, axis=0).astype(np.float32)  # (T, D)
    picks    = np.array(indices, dtype=np.int32)

    # ── L2-normalise to unit norm ────────────────────────────────────────────
    # The eccv16 google_pool5 HDF5 files were generated with Caffe GoogLeNet
    # and features were L2-normalised to unit norm before storage (norm ≈ 1.0).
    # Torchvision GoogLeNet outputs raw pool5 vectors (norm ≈ 16.66).
    # Without this step the feature distribution is incompatible with the
    # trained checkpoints and inference quality degrades significantly.
    norms    = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / (norms + 1e-8)                         # unit vectors

    os.makedirs(os.path.dirname(os.path.abspath(out_h5)), exist_ok=True)
    with h5py.File(out_h5, 'a') as f:
        if video_name in f:
            del f[video_name]          # overwrite if re-running
        grp = f.create_group(video_name)
        grp.create_dataset('features',  data=features, compression='gzip')
        grp.create_dataset('picks',     data=picks)
        grp.create_dataset('n_frames',  data=np.int32(int(picks[-1]) + 1))
        grp.attrs['n_steps']  = len(picks)
        grp.attrs['feat_dim'] = feat_dim
        grp.attrs['backbone'] = backbone

    print(f'Saved {len(picks)} frames × {feat_dim}-d → {out_h5}[{video_name}]')
    print(f'Feature shape: {features.shape}')


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract GoogLeNet pool5 features from a video for inference.')
    parser.add_argument('--video',    required=True,
                        help='Path to input video (MP4, AVI, etc.)')
    parser.add_argument('--out',      required=True,
                        help='Output HDF5 path (e.g. data/my_video.h5)')
    parser.add_argument('--backbone', default='googlenet',
                        choices=['googlenet', 'resnet50', 'efficientnet_b3'],
                        help='Feature backbone (default: googlenet = 1024-d, '
                             'matches trained checkpoints)')
    parser.add_argument('--fps',      type=int, default=C.FRAME_RATE,
                        help='Frames to sample per second (default: 2)')
    parser.add_argument('--device',   default='cuda' if torch.cuda.is_available()
                        else 'cpu')
    args = parser.parse_args()

    extract_features(args.video, args.out, args.backbone, args.fps, args.device)
