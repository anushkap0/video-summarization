"""
Evaluation utilities — F1-score computation against human annotations.

Protocol (matching Zhang et al., 2016):
  1. For each annotator, compute F1 between predicted summary and that
     annotator's binary summary.
  2. Report the MAXIMUM F1 across all annotators.
  3. Average max-F1 over all videos in the test split.

For datasets with only a single binary ground truth (gtsummary), step 1–2
reduce to a single F1 computation.
"""

import numpy as np


def upsample_summary(summary_sampled: np.ndarray,
                     picks: np.ndarray,
                     n_frames: int) -> np.ndarray:
    """
    Convert a summary defined on SAMPLED frames back to the FULL frame space
    by assigning the sampled frame's label to all frames in its temporal segment.

    Args:
        summary_sampled : (T,) binary array over sampled frames
        picks           : (T,) original frame indices of sampled frames
        n_frames        : total number of frames in the original video

    Returns:
        full_summary : (n_frames,) binary array
    """
    full = np.zeros(n_frames, dtype=np.int8)
    for i, pick in enumerate(picks):
        # Assign this label to the segment [pick, next_pick)
        next_pick = picks[i + 1] if i + 1 < len(picks) else n_frames
        full[pick:next_pick] = summary_sampled[i]
    return full


def f1_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Binary F1-score.

    Args:
        pred : (N,) binary predicted summary
        gt   : (N,) binary ground-truth summary

    Returns:
        f1 : float in [0, 1]
    """
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return f1


def evaluate_summary(pred_summary:  np.ndarray,
                     sample:        dict,
                     ratio:         float = 0.15) -> dict:
    """
    Compute F1-score for one video.

    Args:
        pred_summary : (T,) binary array over SAMPLED frames
        sample       : dict from VideoDataset (contains picks, n_frames, etc.)
        ratio        : target summary length (not used here, for logging only)

    Returns:
        dict with keys: f1, precision, recall, n_selected, n_total
    """
    picks    = sample.get('picks')
    n_frames = sample.get('n_frames', len(pred_summary))

    # Upsample to full frame space if picks are available
    if picks is not None and len(picks) == len(pred_summary):
        pred_full = upsample_summary(pred_summary, picks, n_frames)
    else:
        pred_full = pred_summary.astype(np.int8)

    user_summary = sample.get('user_summary')   # (n_users, n_frames) | None
    gtsummary    = sample['gtsummary']           # (T,) or (n_frames,)

    pred_bool = pred_full.astype(bool)

    # user_summary may be annotated at a slightly different length than n_frames
    # (common in the eccv16 HDF5 files). Trim pred_full to match if needed.
    if user_summary is not None:
        us_len = user_summary.shape[1]
        if len(pred_full) != us_len:
            pred_full = pred_full[:us_len] if len(pred_full) > us_len \
                        else np.pad(pred_full, (0, us_len - len(pred_full)))
        pred_bool = pred_full.astype(bool)

    if user_summary is not None:
        # Multiple annotators (SumMe, TVSum): max F1 across all annotators
        scores = []
        for u in range(len(user_summary)):
            gt_u = user_summary[u].astype(bool)
            scores.append(f1_score(pred_full, gt_u))
        best_f1 = max(scores)
        # Use the best-matching annotator for precision/recall stats
        best_u  = int(np.argmax(scores))
        gt_bool = user_summary[best_u].astype(bool)
    else:
        # Single ground truth (VSUMM or gtsummary fallback)
        if len(gtsummary) != len(pred_full) and picks is not None:
            gt_full = upsample_summary(gtsummary.astype(np.int8), picks, n_frames)
        else:
            gt_full = gtsummary
        gt_bool = gt_full.astype(bool)
        best_f1 = f1_score(pred_full, gt_bool)

    tp = int((pred_bool & gt_bool).sum())
    fp = int((pred_bool & ~gt_bool).sum())
    fn = int((~pred_bool & gt_bool).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        'f1':         best_f1,
        'precision':  precision,
        'recall':     recall,
        'n_selected': int(pred_bool.sum()),
        'n_total':    len(pred_full),
    }


def mean_f1(results: list[dict]) -> float:
    return float(np.mean([r['f1'] for r in results]))


# ── Unit test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Perfect prediction
    gt = np.array([1,1,0,0,1,0,1,0,0,1], dtype=np.int8)
    assert abs(f1_score(gt, gt) - 1.0) < 1e-6, 'Perfect F1 should be 1.0'

    # No overlap
    pred = 1 - gt
    assert f1_score(pred, gt) == 0.0, 'Zero-overlap F1 should be 0.0'

    # Partial overlap
    pred2 = np.array([1,0,0,0,1,0,0,0,0,1], dtype=np.int8)
    f1 = f1_score(pred2, gt)
    print(f'Partial overlap F1: {f1:.4f}  (expected ~0.667)')

    print('All evaluation unit tests passed.')
