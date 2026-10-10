"""
zone_localization.py — mask-based ground truth for zone-level attribution.

FaceForensics++ ships a manipulation mask for every fake video (the pixel
region actually altered by the synthesis method). This module converts those
masks into *zone-level* ground truth for the 9 SAEDF semantic zones, and
provides the metrics used by evaluate_zone_localization.py:

  Ground truth
    - crop_and_resize_mask()   : apply the SAME face-crop geometry used during
                                 preprocessing to the mask frame, resize to 224.
    - zone_pixel_masks()       : rasterise the 9 zone convex hulls (224x224).
    - zone_gt_from_mask()      : per-zone manipulated-coverage fraction and
                                 binary "zone is manipulated" labels.

  Localization metrics
    - localization_counts()    : TP/FP/FN/TN of flagged zones vs ground truth.
    - summarize_localization() : precision / recall / F1 / IoU from counts.
    - pointing_game_hit()      : is the top-scored zone actually manipulated?

  Causal faithfulness (deletion / insertion)
    - occlude_zones()          : grey-out a set of zones in a face image.
    - reveal_zones()           : restore a set of zones onto a blurred image.
    - curve_auc()              : normalised area under a deletion/insertion curve.

Everything here is numpy + OpenCV only (no torch), so the fast unit-test tier
can exercise it in CI.
"""

from __future__ import annotations

import base64
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from semantic_patch_mapper import ZONE_LANDMARK_INDICES, ZONE_NAMES

IMAGE_SIZE = 224


def is_likely_mask_frame(
    frame: np.ndarray,
    min_binary_fraction: float = 0.80,
) -> bool:
    """
    Heuristic: FF++ manipulation masks are near-binary (mostly 0 or 255).

    Rejects accidentally using a manipulated RGB video as a mask source.
    """
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    normed = gray.astype(np.float32) / 255.0
    binary_like = (normed < 0.15) | (normed > 0.85)
    return float(binary_like.mean()) >= min_binary_fraction


# ---------------------------------------------------------------------------
# Ground-truth construction
# ---------------------------------------------------------------------------

def crop_and_resize_mask(
    mask_frame: np.ndarray,
    crop_box: Sequence[int],
    target_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """
    Apply the stored face-crop geometry to an FF++ mask frame.

    Args:
        mask_frame : full-resolution mask frame (grayscale or BGR uint8).
        crop_box   : [x1, y1, x2, y2] in source-frame pixel coordinates, as
                     recorded by preprocess_landmarks.py (entry["crop_box"]).
        target_size: output side length (default 224).

    Returns:
        (target_size, target_size) float32 mask in [0, 1].
    """
    if mask_frame.ndim == 3:
        mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
    x1, y1, x2, y2 = (int(v) for v in crop_box)
    h, w = mask_frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid crop_box {list(crop_box)} for mask of shape {(h, w)}.")
    crop = mask_frame[y1:y2, x1:x2]
    # Nearest-neighbour keeps FF++ masks binary after resize (linear blur fills
    # the crop and looks like a solid green block in the demo overlay).
    resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    return resized.astype(np.float32) / 255.0


def binarize_manipulation_mask(
    mask_224: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    FF++ convention: bright pixels = manipulated. Returns float32 {0, 1}.
    """
    return (np.clip(mask_224.astype(np.float32), 0.0, 1.0) >= threshold).astype(
        np.float32
    )


def render_manipulation_overlay(
    face_bgr: np.ndarray,
    mask_224: np.ndarray,
    threshold: float = 0.5,
    alpha: float = 0.55,
    draw_contours: bool = True,
) -> np.ndarray:
    """
    Tint only manipulated pixels on the face crop; optional green outline.
    """
    binary = binarize_manipulation_mask(mask_224, threshold)
    out = face_bgr.astype(np.float32).copy()
    manip = binary > 0
    if manip.any():
        green = np.array([0.0, 255.0, 0.0], dtype=np.float32)
        out[manip] = out[manip] * (1.0 - alpha) + green * alpha
    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    if draw_contours and manip.any():
        mask_u8 = (binary * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out_u8, contours, -1, (0, 255, 0), 1)
    return out_u8


def render_manipulation_pixels_only(
    face_bgr: np.ndarray,
    mask_224: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Black background with only the manipulated pixels visible (face texture +
    green outline). Makes the exact edited region obvious in the demo.
    """
    binary = binarize_manipulation_mask(mask_224, threshold)
    out = np.zeros_like(face_bgr)
    manip = binary > 0
    out[manip] = face_bgr[manip]
    if manip.any():
        mask_u8 = (binary * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 255, 0), 1)
    return out


# BGR tint for system-claimed semantic zones (contrasts with green GT).
CLAIMED_ZONE_TINT_BGR = (80, 80, 255)
CLAIMED_ZONE_OUTLINE_BGR = (80, 80, 255)


def combine_zone_masks(
    zone_masks: Dict[str, np.ndarray],
    zones: Sequence[str],
) -> np.ndarray:
    """Union of rasterised convex-hull masks for the named zones."""
    if not zone_masks:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    shape = next(iter(zone_masks.values())).shape
    combined = np.zeros(shape, dtype=bool)
    for zone in zones:
        zm = zone_masks.get(zone)
        if zm is not None:
            combined |= zm
    return combined


def render_claimed_zones_overlay(
    face_bgr: np.ndarray,
    landmarks_68: Sequence[Sequence[float]],
    zones: Sequence[str],
    alpha: float = 0.55,
) -> np.ndarray:
    """Face crop with red tint on pixels inside explained semantic zones."""
    zone_masks = zone_pixel_masks(landmarks_68, image_size=face_bgr.shape[0])
    claimed = combine_zone_masks(zone_masks, zones)
    out = face_bgr.astype(np.float32).copy()
    if claimed.any():
        tint = np.array(CLAIMED_ZONE_TINT_BGR, dtype=np.float32)
        out[claimed] = out[claimed] * (1.0 - alpha) + tint * alpha
    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    if claimed.any():
        mask_u8 = claimed.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out_u8, contours, -1, CLAIMED_ZONE_OUTLINE_BGR, 1)
    return out_u8


def render_claimed_zones_pixels_only(
    face_bgr: np.ndarray,
    landmarks_68: Sequence[Sequence[float]],
    zones: Sequence[str],
) -> np.ndarray:
    """
    Black background with only explained-zone pixels visible (face texture +
    red outline). Comparable to render_manipulation_pixels_only for GT.
    """
    zone_masks = zone_pixel_masks(landmarks_68, image_size=face_bgr.shape[0])
    claimed = combine_zone_masks(zone_masks, zones)
    out = np.zeros_like(face_bgr)
    out[claimed] = face_bgr[claimed]
    if claimed.any():
        mask_u8 = claimed.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, CLAIMED_ZONE_OUTLINE_BGR, 1)
    return out


def encode_bgr_image_b64(image_bgr: np.ndarray, jpeg_quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise RuntimeError("Failed to encode mask visualization.")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def zone_pixel_masks(
    landmarks_68: Sequence[Sequence[float]],
    image_size: int = IMAGE_SIZE,
) -> Dict[str, np.ndarray]:
    """
    Rasterise each zone's convex hull into a boolean pixel mask.

    Uses the same ZONE_LANDMARK_INDICES geometry as the semantic patch mapper,
    so the ground truth is defined on exactly the regions the explanation head
    reports about.

    Returns:
        {zone_name: (image_size, image_size) bool array}
    """
    pts = np.asarray(landmarks_68, dtype=np.float32)
    if pts.shape != (68, 2):
        raise ValueError(f"Expected 68x2 landmarks, got {pts.shape}.")
    out: Dict[str, np.ndarray] = {}
    for zone, lm_ids in ZONE_LANDMARK_INDICES.items():
        zone_pts = pts[lm_ids]
        canvas = np.zeros((image_size, image_size), dtype=np.uint8)
        if len(zone_pts) >= 3:
            hull = cv2.convexHull(zone_pts)
            if hull is not None and len(hull) >= 3:
                cv2.fillConvexPoly(canvas, hull.astype(np.int32), 1)
        out[zone] = canvas.astype(bool)
    return out


def zone_gt_from_mask(
    mask_224: np.ndarray,
    zone_masks: Dict[str, np.ndarray],
    mask_threshold: float = 0.10,
    coverage_threshold: float = 0.30,
) -> Tuple[Dict[str, float], Dict[str, bool]]:
    """
    Convert a 224x224 manipulation mask into per-zone ground truth.

    Args:
        mask_224          : float mask in [0, 1] aligned with the face crop.
        zone_masks        : output of zone_pixel_masks().
        mask_threshold    : pixel intensity above which a mask pixel counts as
                            manipulated (FF++ masks are near-binary; a low
                            threshold absorbs compression noise).
        coverage_threshold: fraction of a zone's area that must be manipulated
                            for the zone to be labelled ground-truth positive.

    Returns:
        coverage : {zone: fraction of zone pixels that are manipulated}
        gt_label : {zone: bool — zone counts as manipulated}
    """
    binary = mask_224 > mask_threshold
    coverage: Dict[str, float] = {}
    gt_label: Dict[str, bool] = {}
    for zone, zmask in zone_masks.items():
        area = int(zmask.sum())
        if area == 0:
            coverage[zone] = 0.0
            gt_label[zone] = False
            continue
        frac = float(np.logical_and(binary, zmask).sum()) / area
        coverage[zone] = frac
        gt_label[zone] = frac >= coverage_threshold
    return coverage, gt_label


# ---------------------------------------------------------------------------
# Localization metrics
# ---------------------------------------------------------------------------

def localization_counts(
    flagged_zones: Set[str],
    gt_label: Dict[str, bool],
) -> Dict[str, int]:
    """TP/FP/FN/TN counts for one frame's flagged-zone set vs ground truth."""
    tp = fp = fn = tn = 0
    for zone in ZONE_NAMES:
        gt = bool(gt_label.get(zone, False))
        pred = zone in flagged_zones
        if pred and gt:
            tp += 1
        elif pred and not gt:
            fp += 1
        elif not pred and gt:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def summarize_localization(counts: Dict[str, int]) -> Dict[str, float]:
    """Precision / recall / F1 / IoU from accumulated TP/FP/FN/TN counts."""
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall / max(precision + recall, 1e-12)) if (tp + fp + fn) else 0.0
    iou = tp / max(tp + fp + fn, 1)
    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "iou": round(float(iou), 4),
    }


def pointing_game_hit(
    scores: Dict[str, float],
    gt_label: Dict[str, bool],
) -> Optional[bool]:
    """
    Pointing-game: does the single highest-scored zone land on a
    ground-truth-manipulated zone?

    Returns None when the frame has no ground-truth-positive zone at all
    (nothing to point at — excluded from the pointing-game denominator).
    """
    if not any(gt_label.get(z, False) for z in ZONE_NAMES):
        return None
    top_zone = max(ZONE_NAMES, key=lambda z: scores.get(z, float("-inf")))
    return bool(gt_label.get(top_zone, False))


def binary_auc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    """
    Rank-based (Mann-Whitney) AUC without sklearn, tie-aware.

    Returns None when only one class is present.
    """
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
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


# ---------------------------------------------------------------------------
# Deletion / insertion perturbations
# ---------------------------------------------------------------------------

def occlude_zones(
    image_bgr: np.ndarray,
    zone_masks: Dict[str, np.ndarray],
    zones: Sequence[str],
    fill: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """
    Return a copy of the face image with the given zones filled with a
    constant colour (default: the image's own mean colour, which avoids
    introducing a strong out-of-distribution black patch).
    """
    out = image_bgr.copy()
    if fill is None:
        fill_arr = image_bgr.reshape(-1, 3).mean(axis=0)
    else:
        fill_arr = np.asarray(fill, dtype=np.float64)
    combined = np.zeros(image_bgr.shape[:2], dtype=bool)
    for zone in zones:
        if zone in zone_masks:
            combined |= zone_masks[zone]
    out[combined] = fill_arr.astype(out.dtype)
    return out


def reveal_zones(
    image_bgr: np.ndarray,
    baseline_bgr: np.ndarray,
    zone_masks: Dict[str, np.ndarray],
    zones: Sequence[str],
) -> np.ndarray:
    """
    Return a copy of `baseline_bgr` (typically a heavily blurred version of
    the face) with the given zones restored from the original image.
    """
    out = baseline_bgr.copy()
    combined = np.zeros(image_bgr.shape[:2], dtype=bool)
    for zone in zones:
        if zone in zone_masks:
            combined |= zone_masks[zone]
    out[combined] = image_bgr[combined]
    return out


def blurred_baseline(image_bgr: np.ndarray, ksize: int = 31) -> np.ndarray:
    """Standard insertion-curve baseline: strong Gaussian blur of the input."""
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(image_bgr, (ksize, ksize), 0)


def curve_auc(probs: Sequence[float]) -> float:
    """
    Normalised area under a deletion/insertion curve.

    probs[k] is the classifier's fake probability after k steps
    (k = 0 … n). Trapezoidal integration over the unit interval, so the
    result is comparable across frames regardless of step count.
    """
    p = np.asarray(probs, dtype=np.float64)
    if len(p) < 2:
        return float(p[0]) if len(p) else 0.0
    x = np.linspace(0.0, 1.0, len(p))
    return float(np.trapz(p, x))


# ---------------------------------------------------------------------------
# Causal correctness gate (runtime claim filter)
# ---------------------------------------------------------------------------

def measure_zone_causal_drops(
    fake_prob_fn,
    image_bgr: np.ndarray,
    landmarks_68: Sequence[Sequence[float]],
    zones: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """
    Measure how much each claimed zone causally drives the fake score.

    For each zone z, occlude z and re-score P(fake). The drop
    ``baseline − occluded`` is the causal evidence that the claim about z
    is not just textually consistent but actually affects the detector.

    Args:
        fake_prob_fn : callable(List[np.ndarray]) -> List[float]
                       classifier-only fake probabilities for a batch of
                       224×224 BGR images (no explanation head).
        image_bgr    : (224, 224, 3) face crop used for the original claim.
        landmarks_68 : landmarks in the same 224 space.
        zones        : zone names to test (typically flagged explanation zones).

    Returns:
        {zone: {"baseline": p0, "occluded": p_z, "drop": p0 - p_z}}
    """
    unique_zones = list(dict.fromkeys(zones))  # preserve order, drop dupes
    if not unique_zones:
        return {}

    zmasks = zone_pixel_masks(landmarks_68, image_size=image_bgr.shape[0])
    images = [image_bgr]
    for zone in unique_zones:
        images.append(occlude_zones(image_bgr, zmasks, [zone]))

    probs = fake_prob_fn(images)
    if len(probs) != len(images):
        raise ValueError(
            f"fake_prob_fn returned {len(probs)} probs for {len(images)} images."
        )

    baseline = float(probs[0])
    out: Dict[str, Dict[str, float]] = {}
    for i, zone in enumerate(unique_zones):
        occluded = float(probs[i + 1])
        out[zone] = {
            "baseline": round(baseline, 6),
            "occluded": round(occluded, 6),
            "drop": round(baseline - occluded, 6),
        }
    return out


def zone_passes_causal_check(
    drop: float,
    min_drop: float = 0.02,
) -> bool:
    """True when occluding the zone reduces P(fake) by at least ``min_drop``."""
    return float(drop) >= float(min_drop)


__all__ = [
    "crop_and_resize_mask",
    "binarize_manipulation_mask",
    "render_manipulation_overlay",
    "render_manipulation_pixels_only",
    "encode_bgr_image_b64",
    "zone_pixel_masks",
    "zone_gt_from_mask",
    "localization_counts",
    "summarize_localization",
    "pointing_game_hit",
    "binary_auc",
    "occlude_zones",
    "reveal_zones",
    "blurred_baseline",
    "curve_auc",
    "measure_zone_causal_drops",
    "zone_passes_causal_check",
]
