"""
visualization.py — overlay artifact evidence results on face crops.

Visualization is separate from scoring. Heatmap / overlay pixels are never
fed back as quantitative inputs to ArtifactEvidenceAnalyzer.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from artifact_analysis.artifact_inference import ExtendedConceptVector
from artifact_analysis.artifact_ontology import ARTIFACT_DISPLAY_NAMES


def draw_artifact_result(
    image_rgb: np.ndarray,
    zone_mask: np.ndarray,
    concept: ExtendedConceptVector,
    alpha: float = 0.35,
) -> np.ndarray:
    """
    Overlay zone boundary, zone name, R(z), Q(z), artifact, and probability.

    Args:
        image_rgb: (H, W, 3) RGB face crop
        zone_mask: boolean mask for the reported zone
        concept: ExtendedConceptVector from analyze_zone_artifact
        alpha: overlay opacity for filled mask

    Returns:
        RGB uint8 image with annotations (copy; input not mutated).
    """
    img = np.asarray(image_rgb)
    if img.dtype != np.uint8:
        if float(np.max(img)) <= 1.0 + 1e-6:
            canvas = np.clip(img * 255.0, 0, 255).astype(np.uint8).copy()
        else:
            canvas = np.clip(img, 0, 255).astype(np.uint8).copy()
    else:
        canvas = img.copy()

    mask = np.asarray(zone_mask).astype(bool)
    if mask.shape[:2] != canvas.shape[:2]:
        raise ValueError("zone_mask shape must match image HxW")

    color = (220, 60, 60) if not concept.abstained else (160, 160, 160)
    overlay = canvas.copy()
    overlay[mask] = (
        (1.0 - alpha) * overlay[mask].astype(np.float32)
        + alpha * np.array(color, dtype=np.float32)
    ).astype(np.uint8)
    canvas = overlay

    # Contour
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(canvas, contours, -1, color, 2)

    display = ARTIFACT_DISPLAY_NAMES.get(
        concept.artifact_type, concept.artifact_type.replace("_", " ")
    )
    lines = [
        f"{concept.zone}",
        f"R={concept.zone_score_r:.2f}  Q={concept.zone_score_q:.2f}",
        f"{display}  p={concept.artifact_probability:.2f}",
    ]
    if concept.abstained:
        lines[-1] = "artifact: abstained"

    # Anchor text near mask centroid
    ys, xs = np.where(mask)
    if len(xs) > 0:
        cx, cy = int(xs.mean()), int(ys.mean())
    else:
        cx, cy = 10, 20

    y0 = max(cy - 30, 16)
    for i, line in enumerate(lines):
        y = y0 + i * 16
        cv2.putText(
            canvas,
            line[:48],
            (max(cx - 60, 4), y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            line[:48],
            (max(cx - 60, 4), y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return canvas
