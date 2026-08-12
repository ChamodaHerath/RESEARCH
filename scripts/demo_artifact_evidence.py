"""
demo_artifact_evidence.py — example usage of ArtifactEvidenceAnalyzer.

Creates a synthetic face crop with an artificial blending seam around the
mouth zone, runs measurable feature extraction + Bayesian inference, prints
the evidence-backed explanation, and optionally saves an overlay image.

Usage:
  python scripts/demo_artifact_evidence.py
  python scripts/demo_artifact_evidence.py --out results/demo_artifact_overlay.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifact_analysis import (
    ArtifactEvidenceAnalyzer,
    ExtendedFaithfulnessVerifier,
    draw_artifact_result,
    generate_artifact_explanation,
)
from tests.conftest import make_synthetic_landmarks_68
from zone_localization import zone_pixel_masks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/demo_artifact_overlay.png")
    args = ap.parse_args()

    lm = make_synthetic_landmarks_68()
    masks = zone_pixel_masks(lm)
    mouth = masks["mouth"]
    neigh = np.zeros_like(mouth)
    for z, m in masks.items():
        if z != "mouth":
            neigh |= m

    # Synthetic skin + hard compositing ring around mouth
    rng = np.random.default_rng(0)
    img = np.full((224, 224, 3), (180, 140, 120), dtype=np.uint8)
    img = np.clip(img.astype(np.int16) + rng.integers(-8, 9, img.shape), 0, 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ring = cv2.dilate(mouth.astype(np.uint8), k).astype(bool) & (~mouth)
    img[ring] = (25, 25, 25)

    analyzer = ArtifactEvidenceAnalyzer()
    concept = analyzer.analyze(
        image_rgb=img,
        zone_name="mouth",
        zone_mask=mouth,
        neighboring_mask=neigh,
        landmarks=lm,
        r_score=0.84,
        q_score=0.88,
    )
    text = generate_artifact_explanation(concept)
    ok, errors = ExtendedFaithfulnessVerifier().verify(text, concept)

    print("=== ArtifactEvidenceAnalyzer demo ===")
    print(f"zone              : {concept.zone}")
    print(f"artifact_type     : {concept.artifact_type}")
    print(f"abstained         : {concept.abstained}")
    print(f"artifact_prob     : {concept.artifact_probability:.3f}")
    print(f"evidence          : {concept.evidence_features}")
    print(f"explanation       : {text}")
    print(f"faithfulness      : {ok}  {errors if not ok else ''}")

    overlay = draw_artifact_result(img, mouth, concept)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"overlay saved     : {out}")


if __name__ == "__main__":
    main()
