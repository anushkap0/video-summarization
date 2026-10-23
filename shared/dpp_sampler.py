"""
Step 3 – DPP Diversity Sampling
==================================
Given per-frame importance scores and feature vectors, selects a diverse
and high-quality subset using a Determinantal Point Process (DPP).

A DPP defines a probability distribution over subsets S of a ground set
{1, …, T} via the kernel matrix:
    L_{ij} = q_i · S_{ij} · q_j

where:
    q_i  = importance score of frame i  (quality)
    S_{ij} = normalised cosine similarity between frames i and j  (similarity)

We use a greedy MAP (maximum-a-posteriori) approximation, which iteratively
adds the element that maximally increases log det(L_S). This is O(k²T) where
k is the target number of frames — fast enough for any video length.

Reference:
    Kulesza & Taskar (2012). Determinantal Point Processes for Machine Learning.
    Foundations and Trends in Machine Learning.
"""

import numpy as np
import torch


def cosine_similarity_matrix(features: np.ndarray) -> np.ndarray:
    """
    Compute (T, T) cosine similarity matrix from (T, D) feature matrix.
    Values in [0, 1] after shifting: S_{ij} = (cos + 1) / 2.
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    normed = features / norms
    cos = normed @ normed.T                     # (T, T), range [-1, 1]
    sim = (cos + 1.0) / 2.0                    # shift to [0, 1]
    np.fill_diagonal(sim, 1.0)
    return sim.astype(np.float64)


def dpp_greedy(scores: np.ndarray,
               features: np.ndarray,
               k: int,
               lam: float = 1.0) -> list[int]:
    """
    Greedy MAP inference for a quality-diversity DPP.

    Args:
        scores   : (T,) importance scores in [0, 1]
        features : (T, D) feature vectors (used to compute similarity)
        k        : number of frames to select
        lam      : quality-diversity trade-off weight on the quality term.
                   Higher lam → more quality-driven (less diversity).

    Returns:
        selected : sorted list of k selected frame indices
    """
    T = len(scores)
    k = min(k, T)

    sim = cosine_similarity_matrix(features)   # (T, T)
    q   = scores ** lam                        # quality weights

    # Build kernel L_{ij} = q_i * S_{ij} * q_j  (element-wise outer product)
    L = np.outer(q, q) * sim                  # (T, T)

    # ── Greedy MAP via Cholesky update ──────────────────────────────────────
    # We maintain a running Cholesky factor of L_S (the sub-matrix of selected
    # rows/cols) and update it as new elements are greedily added.
    selected = []
    remaining = list(range(T))

    # Marginal gain for element i: det(L_{S∪{i}}) / det(L_S)
    # Using the matrix determinant lemma, the log marginal gain equals
    # log(L_{ii} − L_{iS} @ L_{SS}^{-1} @ L_{Si}) for each candidate i.
    # We track the Schur complement diagonal via incremental Cholesky updates.

    # c[i]  : c_{i,j} coefficients (Gram-Schmidt vectors)
    # d[i]  : current "remaining variance" = L_{ii} - sum_j c_{ij}^2
    c = np.zeros((T, k))
    d = np.copy(np.diag(L))                  # initial = L_{ii}

    for step in range(k):
        if not remaining:
            break
        # Find frame with max remaining variance among candidates
        best = max(remaining, key=lambda i: d[i])
        selected.append(best)
        remaining.remove(best)

        # Update Gram-Schmidt coefficients for all remaining frames
        e_best = np.sqrt(d[best])
        if e_best < 1e-10:
            break                             # rank exhausted
        for i in remaining:
            c[i, step] = (L[i, best] - sum(c[i, j] * c[best, j]
                                            for j in range(step))) / e_best
            d[i] -= c[i, step] ** 2
            d[i] = max(d[i], 0.0)            # numerical clamp

    return sorted(selected)


def select_keyframes(scores: np.ndarray,
                     features: np.ndarray,
                     ratio: float = 0.15,
                     lam: float = 1.0) -> np.ndarray:
    """
    High-level wrapper: select keyframes using DPP given a compression ratio.

    Args:
        scores   : (T,) importance scores
        features : (T, D) feature vectors
        ratio    : target summary length as fraction of T
        lam      : DPP quality weight

    Returns:
        binary_summary : (T,) boolean array  (True = selected)
    """
    T = len(scores)
    k = max(1, round(ratio * T))
    indices = dpp_greedy(scores, features, k, lam)

    summary = np.zeros(T, dtype=bool)
    summary[indices] = True
    return summary


def select_keyframes_topk(scores: np.ndarray, ratio: float = 0.15) -> np.ndarray:
    """
    Baseline: plain top-k selection without diversity.
    Used to ablate the effect of DPP.
    """
    T = len(scores)
    k = max(1, round(ratio * T))
    indices = np.argsort(scores)[-k:]
    summary = np.zeros(T, dtype=bool)
    summary[indices] = True
    return summary


# ── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    np.random.seed(0)
    T, D = 200, 2048
    scores   = np.random.rand(T)
    features = np.random.randn(T, D).astype(np.float32)

    summary_dpp  = select_keyframes(scores, features, ratio=0.15, lam=1.0)
    summary_topk = select_keyframes_topk(scores, ratio=0.15)

    sel_dpp  = np.where(summary_dpp)[0]
    sel_topk = np.where(summary_topk)[0]

    def mean_pairwise_cos(feat, idx):
        f = feat[idx]; f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
        sims = f @ f.T; np.fill_diagonal(sims, 0)
        return sims.sum() / (len(idx) * (len(idx)-1)) if len(idx) > 1 else 0

    print(f'DPP  selected: {len(sel_dpp):3d} frames | '
          f'mean pairwise cos similarity = {mean_pairwise_cos(features, sel_dpp):.4f}')
    print(f'Top-k selected: {len(sel_topk):3d} frames | '
          f'mean pairwise cos similarity = {mean_pairwise_cos(features, sel_topk):.4f}')
    print('(Lower similarity = more diverse)')


def dpp_select(scores: np.ndarray,
               features: np.ndarray,
               k: int,
               lam: float = 1.0) -> np.ndarray:
    """
    Index-returning DPP selector used by v2/v3 (train_v2, inference_v2, train_v3).
    Thin wrapper over dpp_greedy: returns a sorted np.ndarray of k frame indices.
    """
    return np.asarray(dpp_greedy(scores, features, k, lam), dtype=int)
