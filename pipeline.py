"""
pipeline.py — SAEDFPipeline: unified dual-head orchestrator.

Combines the classification head (deepfake detection) and the semantic
explanation head (zone-level attribution) into a single class that:

  1. Loads one deepfake-detection ViT model (HF checkpoint).
  2. Runs a SINGLE forward pass per frame, extracting:
       - fake_probability  (softmax of classification logits)
       - alpha             (196,) last-layer CLS→patch attention  → A(z)
       - phi               (196,) blurred grad-rollout patch scores → S(z)
  3. Passes all signals through the calibrated fusion pipeline
     (SemanticExplanationGenerator loads beta, population stats, and
     Dirichlet artifact priors from configs/fusion_config.json).
  4. Returns a ForensicReport containing both heads.

Usage
-----
    from pipeline import SAEDFPipeline

    pipe = SAEDFPipeline()
    report = pipe.predict_from_landmark_entry(entry)   # landmark JSON dict
    report.print_summary()
    print(report.to_json())
    pipe.close()
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from attention_extractor import AttentionExtractor, DEFAULT_HF_DEEPFAKE_MODEL
from baseline_heatmap_generator import extract_full_pipeline_inputs
from semantic_patch_mapper import (
    ClassificationHead,
    ExplanationOutput,
    ForensicReport,
    SemanticExplanationGenerator,
    load_fusion_config,
)


# ---------------------------------------------------------------------------
# SAEDFPipeline
# ---------------------------------------------------------------------------

class SAEDFPipeline:
    """
    End-to-end SAEDF pipeline: one model, one forward pass, two output heads.

    Classification head  — fake_probability + label + confidence tier.
    Explanation head     — zone R(z) scores with 95 % credible intervals,
                           Bayesian artifact type selection, verified NL sentences.

    Parameters
    ----------
    model_name   : HF model id or timm/torchvision model name.
                   Must be the same model used during calibration
                   (recalibrate_v2.py default: prithivMLmods/Deep-Fake-Detector-v2-Model).
    config_path  : Path to fusion_config.json written by recalibrate_v2.py.
    min_severity : Minimum zone severity to generate explanations for.
                   "medium" (default) suppresses borderline low-signal zones.
    device       : Torch device string, e.g. "cuda" or "cpu".
                   None = auto-detect (GPU if available, else CPU).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_HF_DEEPFAKE_MODEL,
        config_path: str = "configs/fusion_config.json",
        min_severity: str = "medium",
        device: Optional[str] = None,
    ) -> None:
        self.model_name   = model_name
        self.config_path  = config_path
        self.min_severity = min_severity

        # Load the deepfake detection model
        self.extractor = AttentionExtractor(
            model_name=model_name,
            pretrained=True,
            device=device,
        )

        # Load calibrated explanation generator (beta, stats, artifact priors)
        self.explainer = SemanticExplanationGenerator(
            config_path=config_path,
            min_severity=min_severity,
        )

        # Pull calibration metadata for report stamping
        cfg, *_ = load_fusion_config(config_path)
        self._calibration_meta: Dict = {
            "config_path":    str(config_path),
            "beta_map":       cfg.get("beta", None),
            "beta_ci":        (
                [
                    cfg["beta_posterior"]["ci_lower"],
                    cfg["beta_posterior"]["ci_upper"],
                ]
                if "beta_posterior" in cfg
                else None
            ),
            "n_calibration_frames": cfg.get("calibration_n_frames", None),
        }

    # ------------------------------------------------------------------
    def predict_frame(
        self,
        image_bgr: np.ndarray,
        landmarks_68: Sequence[Sequence[float]],
        image_path: str = "",
    ) -> ForensicReport:
        """
        Run dual-head inference on one pre-cropped 224×224 face image.

        Args:
            image_bgr    : (224, 224, 3) BGR uint8/float32.
            landmarks_68 : (68, 2) dlib landmark coordinates in 224-px space.
            image_path   : Optional path string included in the report metadata.

        Returns:
            ForensicReport with classification + explanation heads.
        """
        if np.asarray(image_bgr).shape != (224, 224, 3):
            raise ValueError(
                f"Expected image_bgr shape (224, 224, 3), got {np.asarray(image_bgr).shape}."
            )

        # ------------------------------------------------------------------ #
        # Single forward pass → fake_prob, phi, alpha                         #
        # ------------------------------------------------------------------ #
        inputs = extract_full_pipeline_inputs(image_bgr, self.extractor)

        # ------------------------------------------------------------------ #
        # Classification head                                                  #
        # ------------------------------------------------------------------ #
        classification = ClassificationHead.from_probability(
            fake_probability=inputs["fake_probability"],
            model_name=self.model_name,
        )

        # ------------------------------------------------------------------ #
        # Explanation head (normalised fusion, Dirichlet artifacts, CIs)       #
        # ------------------------------------------------------------------ #
        explanation: ExplanationOutput = self.explainer.predict(
            attention_vec=inputs["alpha"],
            landmarks_68=landmarks_68,
            phi_vec=inputs["phi"],
            fake_probability=inputs["fake_probability"],
        )

        # ------------------------------------------------------------------ #
        # Wrap in ForensicReport                                               #
        # ------------------------------------------------------------------ #
        metadata = {
            "image_path":  image_path,
            "timestamp":   datetime.now(tz=timezone.utc).isoformat(),
            "model":       self.model_name,
            "calibration": self._calibration_meta,
        }
        return ForensicReport(
            classification=classification,
            explanation=explanation,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    def predict_from_landmark_entry(
        self,
        entry: Dict,
        image_path: str = "",
    ) -> ForensicReport:
        """
        Convenience wrapper: accepts a landmark JSON entry dict directly.

        Reads:
            entry["face_image"]       : path to 224×224 face image
            entry["landmarks_68"]     : list of 68 [x, y] pairs
            entry["zone_patch_matrix"]: 196×9 M matrix (used for display only;
                                        the pipeline recomputes M from landmarks)

        Args:
            entry      : dict from ffpp_landmarks*.json["frames"][key]
            image_path : override label; defaults to entry["face_image"]

        Returns:
            ForensicReport
        """
        face_path = entry.get("face_image", "")
        image_bgr = cv2.imread(face_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read face image: {face_path}")
        if image_bgr.shape != (224, 224, 3):
            image_bgr = cv2.resize(image_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)

        landmarks_68 = entry["landmarks_68"]
        return self.predict_frame(
            image_bgr=image_bgr,
            landmarks_68=landmarks_68,
            image_path=image_path or face_path,
        )

    # ------------------------------------------------------------------
    def predict_batch(
        self,
        entries: List[Dict],
        verbose: bool = False,
    ) -> List[ForensicReport]:
        """
        Run predict_from_landmark_entry over a list of landmark JSON entries.

        Args:
            entries : list of landmark JSON entry dicts.
            verbose : if True, print a progress counter every 10 frames.

        Returns:
            List of ForensicReport objects (same order as entries).
        """
        reports: List[ForensicReport] = []
        total = len(entries)
        for i, entry in enumerate(entries):
            if verbose and i % 10 == 0:
                print(f"  [{i}/{total}] processing frames ...")
            try:
                report = self.predict_from_landmark_entry(entry)
                reports.append(report)
            except Exception as exc:
                if verbose:
                    print(f"  [WARN] skipped frame {i}: {exc}")
        return reports

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Release model resources."""
        self.extractor.close()

    # Context-manager support
    def __enter__(self) -> "SAEDFPipeline":
        return self

    def __exit__(self, *_) -> None:
        self.close()
