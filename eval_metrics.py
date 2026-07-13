"""
eval_metrics.py — dependency-light statistical helpers for pipeline evaluation.

Adds the statistical rigor layer to evaluate_pipeline.py:

  - binary_auc()             : tie-aware rank AUC (no sklearn needed).
  - aggregate_video_level()  : collapse frame predictions to one prediction
                               per video (mean fake probability), because
                               frames of the same video are correlated and
                               frame-level metrics overstate the effective
                               sample size.
  - cluster_bootstrap_ci()   : bootstrap confidence intervals that resample
                               VIDEOS (clusters), not frames, respecting the
                               within-video correlation structure.
  - per_method_breakdown()   : FF++ per-manipulation-method metrics (each
                               method's fakes vs the shared real frames).

numpy-only, unit-testable in the fast CI tier.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Rank AUC (tie-aware, no sklearn)
# ---------------------------------------------------------------------------

def binary_auc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    """Mann-Whitney AUC. Returns None when only one class is present."""
    y = np.asarray(labels, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_scores = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0   # average rank, 1-based
        i = j + 1
    rank_sum_pos = float(ranks[y == 1].sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def accuracy_from_probs(labels: Sequence[int], probs: Sequence[float], threshold: float = 0.5) -> float:
    y = np.asarray(labels, dtype=np.int32)
    pred = (np.asarray(probs, dtype=np.float64) >= threshold).astype(np.int32)
    return float((y == pred).mean()) if len(y) else 0.0


# ---------------------------------------------------------------------------
# Video-level aggregation
# ---------------------------------------------------------------------------

def aggregate_video_level(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    video_ids: Sequence[str],
) -> Tuple[List[int], List[float], List[str]]:
    """
    Collapse frame-level predictions into one (label, mean fake probability)
    per video. A video's label is fake if any of its frames is labelled fake
    (labels are per-video in FF++/Celeb-DF anyway).

    Returns (video_labels, video_probs, video_id_order).
    """
    probs_by_video: Dict[str, List[float]] = defaultdict(list)
    label_by_video: Dict[str, int] = {}
    order: List[str] = []
    for t, p, v in zip(y_true, y_prob, video_ids):
        if v not in probs_by_video:
            order.append(v)
        probs_by_video[v].append(float(p))
        label_by_video[v] = max(label_by_video.get(v, 0), int(t))
    video_labels = [label_by_video[v] for v in order]
    video_probs = [float(np.mean(probs_by_video[v])) for v in order]
    return video_labels, video_probs, order


def video_level_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    video_ids: Sequence[str],
    threshold: float = 0.5,
) -> Dict:
    """Accuracy / AUC / confusion counts on video-level aggregated predictions."""
    v_labels, v_probs, order = aggregate_video_level(y_true, y_prob, video_ids)
    v_pred = [1 if p >= threshold else 0 for p in v_probs]
    tp = sum(t == 1 and p == 1 for t, p in zip(v_labels, v_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(v_labels, v_pred))
    tn = sum(t == 0 and p == 0 for t, p in zip(v_labels, v_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(v_labels, v_pred))
    auc = binary_auc(v_labels, v_probs)
    return {
        "n_videos": len(order),
        "n_fake_videos": int(sum(v_labels)),
        "n_real_videos": int(len(v_labels) - sum(v_labels)),
        "accuracy": round(accuracy_from_probs(v_labels, v_probs, threshold), 4),
        "auc_roc": round(auc, 4) if auc is not None else None,
        "recall": round(tp / max(tp + fn, 1), 4),
        "precision": round(tp / max(tp + fp, 1), 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "note": "One prediction per video = mean fake probability over its frames.",
    }


# ---------------------------------------------------------------------------
# Cluster (video-level) bootstrap confidence intervals
# ---------------------------------------------------------------------------

def cluster_bootstrap_ci(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    video_ids: Sequence[str],
    metric_fn: Callable[[np.ndarray, np.ndarray], Optional[float]],
    n_boot: int = 1000,
    coverage: float = 0.95,
    seed: int = 42,
) -> Optional[Dict]:
    """
    Bootstrap CI for a frame-level metric, resampling VIDEOS with replacement.

    Resampling whole videos (clusters) instead of individual frames keeps the
    within-video frame correlation intact, so the interval is honest about the
    effective sample size (172 frames from ~35 videos is not n=172).

    Args:
        metric_fn : f(y_true_array, y_prob_array) -> float or None
                    (None resamples are discarded, e.g. single-class draws).

    Returns:
        {"point", "ci_lower", "ci_upper", "n_boot_effective"} or None when
        the metric is undefined on the full sample.
    """
    y = np.asarray(y_true, dtype=np.int32)
    p = np.asarray(y_prob, dtype=np.float64)
    point = metric_fn(y, p)
    if point is None:
        return None

    frames_by_video: Dict[str, List[int]] = defaultdict(list)
    for idx, v in enumerate(video_ids):
        frames_by_video[v].append(idx)
    videos = list(frames_by_video.keys())
    rng = np.random.default_rng(seed)

    values: List[float] = []
    for _ in range(n_boot):
        sampled_videos = rng.choice(len(videos), size=len(videos), replace=True)
        idx = np.concatenate([np.asarray(frames_by_video[videos[k]]) for k in sampled_videos])
        val = metric_fn(y[idx], p[idx])
        if val is not None:
            values.append(float(val))
    if not values:
        return None

    alpha = (1.0 - coverage) / 2.0
    lo, hi = np.quantile(values, [alpha, 1.0 - alpha])
    return {
        "point": round(float(point), 4),
        "ci_lower": round(float(lo), 4),
        "ci_upper": round(float(hi), 4),
        "n_boot_effective": len(values),
    }


def standard_bootstrap_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    video_ids: Sequence[str],
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict:
    """Cluster-bootstrap CIs for the headline metrics (accuracy, AUC, recall)."""

    def _acc(y: np.ndarray, p: np.ndarray) -> float:
        return accuracy_from_probs(y, p)

    def _auc(y: np.ndarray, p: np.ndarray) -> Optional[float]:
        return binary_auc(y, p)

    def _recall(y: np.ndarray, p: np.ndarray) -> Optional[float]:
        pred = (p >= 0.5).astype(np.int32)
        n_pos = int((y == 1).sum())
        if n_pos == 0:
            return None
        return float(((y == 1) & (pred == 1)).sum() / n_pos)

    out = {}
    for name, fn in [("accuracy", _acc), ("auc_roc", _auc), ("recall", _recall)]:
        ci = cluster_bootstrap_ci(y_true, y_prob, video_ids, fn, n_boot=n_boot, seed=seed)
        if ci is not None:
            out[name] = ci
    out["note"] = (
        f"95% CIs from {n_boot} cluster-bootstrap resamples of whole videos "
        "(frames of one video are correlated, so frame-level resampling would "
        "understate the uncertainty)."
    )
    return out


# ---------------------------------------------------------------------------
# Per-method breakdown (FF++)
# ---------------------------------------------------------------------------

def per_method_breakdown(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    video_ids: Sequence[str],
    methods: Sequence[Optional[str]],
    threshold: float = 0.5,
) -> Dict[str, Dict]:
    """
    Per-manipulation-method classification metrics.

    Each method's metrics are computed on that method's fake frames plus ALL
    real frames (the standard FF++ per-method protocol): detection recall on
    the method's fakes, and AUC of method-fakes vs reals.

    Args:
        methods : per-frame method label (None/'' for real frames and frames
                  without a parseable method).
    """
    y = np.asarray(y_true, dtype=np.int32)
    p = np.asarray(y_prob, dtype=np.float64)
    vids = list(video_ids)
    meths = list(methods)

    real_idx = [i for i, t in enumerate(y) if t == 0]
    method_names = sorted({m for m, t in zip(meths, y) if t == 1 and m})

    out: Dict[str, Dict] = {}
    for m in method_names:
        fake_idx = [i for i, (mm, t) in enumerate(zip(meths, y)) if t == 1 and mm == m]
        idx = fake_idx + real_idx
        y_m, p_m = y[idx], p[idx]
        pred_fake = (p[fake_idx] >= threshold).astype(np.int32)
        auc = binary_auc(y_m, p_m)

        # video-level recall over the method's fake videos
        v_labels, v_probs, _ = aggregate_video_level(
            [1] * len(fake_idx), p[fake_idx], [vids[i] for i in fake_idx]
        )
        v_detected = sum(1 for pr in v_probs if pr >= threshold)

        out[m] = {
            "n_fake_frames": len(fake_idx),
            "n_fake_videos": len(v_probs),
            "frame_recall": round(float(pred_fake.mean()), 4) if len(fake_idx) else None,
            "video_recall": round(v_detected / max(len(v_probs), 1), 4),
            "auc_vs_reals": round(auc, 4) if auc is not None else None,
            "mean_fake_probability": round(float(p[fake_idx].mean()), 4) if len(fake_idx) else None,
        }
    return out


__all__ = [
    "binary_auc",
    "accuracy_from_probs",
    "aggregate_video_level",
    "video_level_metrics",
    "cluster_bootstrap_ci",
    "standard_bootstrap_metrics",
    "per_method_breakdown",
]
