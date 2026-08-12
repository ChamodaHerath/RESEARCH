"""
artifact_inference.py — Bayesian / probabilistic artifact-type inference.

BAYESIAN COMPONENT
------------------
Posterior (conceptual):

    P(a | f, z, R, Q)  proportional to  P(f | a, z) * P(a | z) * g(R, Q)

where:
  - P(a | z)     = zone-specific Dirichlet prior (weak; config)
  - P(f | a, z)  = feature likelihood from weighted evidence scores
  - g(R, Q)      = confidence adjustment only (does NOT select artifact type)

RULE-BASED COMPONENT: feature weights + abstention from config.
LEARNED COMPONENT (optional): ArtifactCalibrationModel likelihood blend.

Scientific constraint: report "most likely observable artifact"; abstain when weak.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import cv2
import numpy as np

from artifact_analysis.artifact_ontology import (
    ARTIFACT_DISPLAY_NAMES,
    ArtifactEvidenceConfig,
    load_artifact_config,
)
from artifact_analysis.feature_extractor import ZoneFeatureExtractor


@dataclass
class ArtifactPrediction:
    """Posterior artifact estimate for one zone."""

    artifact_type: str
    probability: float
    probabilities: Dict[str, float]
    evidence: Dict[str, float]
    confidence: float
    abstained: bool = False
    abstention_reason: Optional[str] = None

    def to_dict(self) -> Dict:
        d = {
            "artifact_type": self.artifact_type,
            "probability": round(float(self.probability), 4),
            "probabilities": {
                k: round(float(v), 4) for k, v in self.probabilities.items()
            },
            "evidence": {k: round(float(v), 4) for k, v in self.evidence.items()},
            "confidence": round(float(self.confidence), 4),
            "abstained": self.abstained,
        }
        if self.abstention_reason:
            d["abstention_reason"] = self.abstention_reason
        return d


@dataclass
class ExtendedConceptVector:
    """
    Evidence-supported concept vector for one flagged zone.

    Separates zone anomaly confidence (R/Q) from artifact-type confidence.
    """

    zone: str
    artifact_type: str
    severity: str
    zone_score_r: float
    zone_score_q: float
    artifact_probability: float
    credible_interval: Tuple[float, float]
    evidence_features: Dict[str, float]
    probabilities: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    abstained: bool = False
    features_raw: Dict[str, float] = field(default_factory=dict, repr=False)

    def to_dict(self) -> Dict:
        return {
            "zone": self.zone,
            "artifact_type": self.artifact_type,
            "severity": self.severity,
            "zone_score_r": round(float(self.zone_score_r), 4),
            "zone_score_q": round(float(self.zone_score_q), 4),
            "artifact_probability": round(float(self.artifact_probability), 4),
            "credible_interval": [
                round(float(self.credible_interval[0]), 4),
                round(float(self.credible_interval[1]), 4),
            ],
            "evidence_features": {
                k: round(float(v), 4) for k, v in self.evidence_features.items()
            },
            "probabilities": {
                k: round(float(v), 4) for k, v in self.probabilities.items()
            },
            "confidence": round(float(self.confidence), 4),
            "abstained": self.abstained,
        }


# Soft reference scales for robust normalization into [0, 1].
# Documented defaults — not hard decision thresholds inside scoring.
_FEATURE_REF_SCALES: Dict[str, float] = {
    "lbp_entropy": 4.0,
    "lbp_entropy_difference": 2.0,
    "laplacian_variance": 500.0,
    "laplacian_variance_difference": 400.0,
    "local_contrast": 40.0,
    "local_contrast_difference": 30.0,
    "gradient_magnitude_mean": 80.0,
    "gradient_magnitude_std": 50.0,
    "gray_level_entropy": 5.0,
    "boundary_gradient_mean": 80.0,
    "boundary_discontinuity": 60.0,
    "inside_outside_color_difference": 80.0,
    "edge_density_boundary": 1.0,
    "canny_edge_density": 1.0,
    "boundary_sharpness_ratio": 5.0,
    "delta_e_neighbor": 40.0,
    "hue_difference": 40.0,
    "saturation_difference": 60.0,
    "lab_a_mean_difference": 20.0,
    "lab_b_mean_difference": 20.0,
    "rgb_mean_difference": 80.0,
    "luminance_neighbor_difference": 40.0,
    "illumination_gradient": 80.0,
    "highlight_ratio": 1.0,
    "highlight_ratio_difference": 0.5,
    "shadow_ratio": 1.0,
    "left_right_luminance_imbalance": 40.0,
    "high_low_frequency_ratio": 5.0,
    "high_low_frequency_ratio_difference": 3.0,
    "spectral_entropy": 6.0,
    "spectral_entropy_difference": 3.0,
    "radial_spectrum_difference": 0.5,
    "high_frequency_energy": 1.0,
    "mid_frequency_energy": 1.0,
    "blocking_score": 5.0,
    "block_boundary_strength": 40.0,
    "block_boundary_difference": 30.0,
    "block_hf_discontinuity": 40.0,
    "landmark_symmetry_error": 0.15,
    "shape_ratio_deviation": 0.2,
    "contour_irregularity": 2.0,
    "eye_width_asymmetry": 0.1,
    "mouth_corner_asymmetry": 0.1,
    "nose_alignment": 0.1,
    "jaw_contour_irregularity": 0.1,
}


def normalize_features(
    features: Mapping[str, float],
    eps: float = 1e-8,
    ref_scales: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Map raw features into approximately [0, 1] via soft reference scales."""
    scales = ref_scales or _FEATURE_REF_SCALES
    out: Dict[str, float] = {}
    for k, v in features.items():
        if k == "zone_pixel_count":
            out[k] = float(v)
            continue
        scale = float(scales.get(k, max(abs(float(v)), 1.0)))
        out[k] = float(np.clip(abs(float(v)) / (scale + eps), 0.0, 1.0))
    return out


def g_rq(r_score: float, q_score: float, cfg: ArtifactEvidenceConfig) -> float:
    """Confidence adjustment from zone anomaly scores (not class selection)."""
    g = cfg.g_rq
    raw = g.r_weight * float(r_score) + g.q_weight * float(q_score)
    return float(np.clip(raw, g.floor, g.ceiling))


def assign_severity(score: float, thresholds: Mapping[str, float]) -> str:
    if score >= float(thresholds.get("high", 0.75)):
        return "high"
    if score >= float(thresholds.get("medium", 0.50)):
        return "medium"
    return "low"


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - np.max(logits)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


def _wilson_interval(p_hat: float, n: float, z: float = 1.96) -> Tuple[float, float]:
    """Approximate 95% interval for a binomial proportion (Wilson)."""
    n = max(float(n), 1.0)
    p = float(np.clip(p_hat, 0.0, 1.0))
    denom = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    margin = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    lo = (centre - margin) / denom
    hi = (centre + margin) / denom
    return float(np.clip(lo, 0.0, 1.0)), float(np.clip(hi, 0.0, 1.0))


class BayesianArtifactInferencer:
    """
    Transparent probabilistic artifact classifier (no neural net).

    Mode 1: rule-based Bayesian scoring (default; always available).
    Mode 2: optional calibrated likelihood from ArtifactCalibrationModel.
    """

    def __init__(
        self,
        config: Optional[ArtifactEvidenceConfig] = None,
        calibration_model=None,
        top_evidence_k: int = 3,
    ):
        self.config = config or load_artifact_config()
        self.calibration_model = calibration_model
        self.top_evidence_k = int(top_evidence_k)
        self.artifact_types: List[str] = list(self.config.artifact_types)

    def evidence_likelihoods(
        self,
        features_norm: Mapping[str, float],
        zone_name: str,
    ) -> Dict[str, float]:
        """RULE-BASED: weighted mean of features listed for each artifact type."""
        del zone_name  # reserved for zone-conditioned likelihood profiles
        scores: Dict[str, float] = {}
        for a in self.artifact_types:
            if a == "no_clear_artifact":
                scores[a] = 0.15
                continue
            weights = self.config.feature_weights(a)
            if not weights:
                scores[a] = 0.0
                continue
            num = 0.0
            den = 0.0
            for feat, w in weights.items():
                if feat not in features_norm:
                    continue
                num += float(w) * float(features_norm[feat])
                den += float(w)
            scores[a] = (num / den) if den > 0 else 0.0
        return scores

    def predict(
        self,
        zone_name: str,
        features: Dict[str, float],
        r_score: float,
        q_score: float,
        image_quality_laplacian_var: Optional[float] = None,
    ) -> ArtifactPrediction:
        """Estimate P(artifact | features, zone, R, Q) with abstention."""
        cfg = self.config
        abst = cfg.abstention

        n_pix = float(features.get("zone_pixel_count", 0.0))
        if n_pix < abst.min_zone_pixels:
            return self._abstain("zone_mask_too_small", r_score, q_score)
        if float(q_score) < abst.min_q_score:
            return self._abstain("q_score_below_threshold", r_score, q_score)
        if (
            image_quality_laplacian_var is not None
            and float(image_quality_laplacian_var) < abst.min_image_laplacian_var
        ):
            return self._abstain("poor_image_quality", r_score, q_score)

        feats_n = normalize_features(features, eps=cfg.feature_norm_eps)
        like = self.evidence_likelihoods(feats_n, zone_name)

        if self.calibration_model is not None:
            try:
                cal_probs = self.calibration_model.predict_proba(features, zone_name)
                for a in self.artifact_types:
                    like[a] = 0.5 * like[a] + 0.5 * float(cal_probs.get(a, 0.0))
            except Exception:
                pass

        prior = self.config.prior_vector(zone_name)
        prior_sum = sum(prior.values()) + 1e-12
        prior_p = {a: prior[a] / prior_sum for a in self.artifact_types}
        g = g_rq(r_score, q_score, cfg)

        logits = []
        for a in self.artifact_types:
            lf = math.log(max(like[a], 1e-6))
            lp = math.log(max(prior_p[a], 1e-6))
            if a == "no_clear_artifact":
                lg = math.log(max(1.0 - 0.5 * g, 1e-6))
            else:
                lg = math.log(max(0.5 + 0.5 * g, 1e-6))
            logits.append(lf + lp + lg)

        probs_arr = _softmax(np.asarray(logits, dtype=np.float64))
        probs = {a: float(probs_arr[i]) for i, a in enumerate(self.artifact_types)}
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        top_a, top_p = ranked[0]
        second_p = ranked[1][1] if len(ranked) > 1 else 0.0

        if top_p < abst.min_probability:
            return self._abstain(
                "top_probability_too_low",
                r_score,
                q_score,
                probabilities=probs,
                evidence=self._top_evidence(feats_n, top_a),
            )
        if (top_p - second_p) < abst.min_margin:
            return self._abstain(
                "top_two_too_close",
                r_score,
                q_score,
                probabilities=probs,
                evidence=self._top_evidence(feats_n, top_a),
            )
        if top_a == "no_clear_artifact":
            return self._abstain(
                "no_clear_artifact_selected",
                r_score,
                q_score,
                probabilities=probs,
            )

        evidence = self._top_evidence(feats_n, top_a)
        return ArtifactPrediction(
            artifact_type=top_a,
            probability=float(top_p),
            probabilities=probs,
            evidence=evidence,
            confidence=float(top_p * g),
            abstained=False,
        )

    def _top_evidence(
        self, features_norm: Mapping[str, float], artifact_type: str
    ) -> Dict[str, float]:
        weights = self.config.feature_weights(artifact_type)
        scored = []
        for feat, w in weights.items():
            if feat not in features_norm:
                continue
            scored.append((feat, float(features_norm[feat]) * float(w)))
        scored.sort(key=lambda x: x[1], reverse=True)
        out: Dict[str, float] = {}
        for feat, _ in scored[: self.top_evidence_k]:
            out[feat] = float(features_norm[feat])
        return out

    def _abstain(
        self,
        reason: str,
        r_score: float,
        q_score: float,
        probabilities: Optional[Dict[str, float]] = None,
        evidence: Optional[Dict[str, float]] = None,
    ) -> ArtifactPrediction:
        if probabilities is None:
            n = len(self.artifact_types)
            probabilities = {a: 0.05 for a in self.artifact_types}
            probabilities["no_clear_artifact"] = 1.0 - 0.05 * (n - 1)
        g = g_rq(r_score, q_score, self.config)
        p = float(probabilities.get("no_clear_artifact", 1.0))
        return ArtifactPrediction(
            artifact_type="no_clear_artifact",
            probability=p,
            probabilities=probabilities,
            evidence=evidence or {},
            confidence=float(p * g),
            abstained=True,
            abstention_reason=reason,
        )


class ArtifactEvidenceAnalyzer:
    """
    End-to-end measurable artifact evidence module.

    Image -> zone features -> normalize -> Bayesian inference -> abstention
    -> ExtendedConceptVector.
    """

    def __init__(
        self,
        config: Optional[ArtifactEvidenceConfig] = None,
        feature_extractor: Optional[ZoneFeatureExtractor] = None,
        inferencer: Optional[BayesianArtifactInferencer] = None,
        calibration_model=None,
    ):
        self.config = config or load_artifact_config()
        self.feature_extractor = feature_extractor or ZoneFeatureExtractor(
            min_zone_pixels=self.config.abstention.min_zone_pixels
        )
        self.inferencer = inferencer or BayesianArtifactInferencer(
            config=self.config,
            calibration_model=calibration_model,
        )

    def analyze(
        self,
        image_rgb: np.ndarray,
        zone_name: str,
        zone_mask: np.ndarray,
        neighboring_mask: np.ndarray,
        landmarks: np.ndarray,
        r_score: float,
        q_score: float,
    ) -> ExtendedConceptVector:
        return analyze_zone_artifact(
            image_rgb=image_rgb,
            zone_name=zone_name,
            zone_mask=zone_mask,
            neighboring_mask=neighboring_mask,
            landmarks=landmarks,
            r_score=r_score,
            q_score=q_score,
            artifact_analyzer=self,
        )


def analyze_zone_artifact(
    image_rgb: np.ndarray,
    zone_name: str,
    zone_mask: np.ndarray,
    neighboring_mask: np.ndarray,
    landmarks: np.ndarray,
    r_score: float,
    q_score: float,
    artifact_analyzer: ArtifactEvidenceAnalyzer,
) -> ExtendedConceptVector:
    """
    Integration function for the SAEDF pipeline.

    Only meaningful for zones that already exceed the configured Q(z) threshold;
    the inferencer also re-checks Q and abstains if below gate.
    """
    cfg = artifact_analyzer.config
    extractor = artifact_analyzer.feature_extractor
    inferencer = artifact_analyzer.inferencer

    features = extractor.extract_with_geometry(
        image_rgb=image_rgb,
        zone_mask=zone_mask,
        landmarks=landmarks,
        zone_name=zone_name,
        neighboring_mask=neighboring_mask,
    )

    img = np.asarray(image_rgb)
    if img.dtype != np.uint8:
        if float(np.max(img)) <= 1.0 + 1e-6:
            img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        else:
            img_u8 = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img_u8 = img
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())

    pred = inferencer.predict(
        zone_name=zone_name,
        features=features,
        r_score=float(r_score),
        q_score=float(q_score),
        image_quality_laplacian_var=lap_var,
    )

    n_eff = max(float(features.get("zone_pixel_count", 1.0)) / 50.0, 5.0)
    ci = _wilson_interval(pred.probability, n_eff)
    zone_score = float(q_score) if q_score is not None else float(r_score)
    severity = assign_severity(zone_score, cfg.severity_thresholds)

    return ExtendedConceptVector(
        zone=zone_name,
        artifact_type=pred.artifact_type,
        severity=severity,
        zone_score_r=float(r_score),
        zone_score_q=float(q_score),
        artifact_probability=float(pred.probability),
        credible_interval=ci,
        evidence_features=dict(pred.evidence),
        probabilities=dict(pred.probabilities),
        confidence=float(pred.confidence),
        abstained=bool(pred.abstained),
        features_raw=features,
    )


def generate_artifact_explanation(concept: ExtendedConceptVector) -> str:
    """
    Deterministic evidence-backed explanation.

    Uses "most likely observable artifact" wording. Never names a specific
    artifact type when the system abstains.
    """
    zone_label = concept.zone.replace("_", " ")
    sev_word = {"high": "High", "medium": "Moderate", "low": "Low"}.get(
        concept.severity, "Moderate"
    )

    if concept.abstained or concept.artifact_type == "no_clear_artifact":
        return (
            f"{sev_word} manipulation evidence was detected in the {zone_label} region, "
            f"but the visual artifact type could not be determined reliably. "
            f"Posterior zone score: {concept.zone_score_q:.2f}."
        )

    display = ARTIFACT_DISPLAY_NAMES.get(
        concept.artifact_type, concept.artifact_type.replace("_", " ")
    )
    cfg = load_artifact_config()
    evidence_names = [
        cfg.feature_display_names.get(feat, feat.replace("_", " "))
        for feat in concept.evidence_features
    ]
    if len(evidence_names) >= 2:
        evidence_clause = f"{evidence_names[0]} and {evidence_names[1]}"
    elif evidence_names:
        evidence_clause = evidence_names[0]
    else:
        evidence_clause = "measured local image statistics"

    lo, hi = concept.credible_interval
    return (
        f"{sev_word} manipulation evidence was detected in the {zone_label} region. "
        f"The most likely observable artifact is a {display}, supported by elevated "
        f"{evidence_clause}. "
        f"Posterior zone score: {concept.zone_score_q:.2f}. "
        f"Artifact probability: {concept.artifact_probability:.2f}. "
        f"95% credible interval: [{lo:.2f}, {hi:.2f}]."
    )


class ExtendedFaithfulnessVerifier:
    """Verify evidence-backed NL against ExtendedConceptVector; return errors."""

    def verify(
        self,
        explanation: str,
        concept: ExtendedConceptVector,
    ) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        text = explanation.lower()
        zone_label = concept.zone.replace("_", " ").lower()

        if zone_label not in text:
            errors.append(f"zone '{zone_label}' not found in explanation")

        if concept.abstained or concept.artifact_type == "no_clear_artifact":
            for a, display in ARTIFACT_DISPLAY_NAMES.items():
                if a == "no_clear_artifact":
                    continue
                claim = f"most likely observable artifact is a {display.lower()}"
                if claim in text:
                    errors.append(
                        f"specific artifact '{display}' stated despite abstention"
                    )
            if "could not be determined" not in text and "reliably" not in text:
                errors.append("abstention explanation missing unreliability wording")
        else:
            display = ARTIFACT_DISPLAY_NAMES.get(
                concept.artifact_type, concept.artifact_type.replace("_", " ")
            ).lower()
            if display not in text:
                errors.append(f"artifact '{display}' not found in explanation")
            if concept.evidence_features:
                cfg = load_artifact_config()
                any_ev = False
                for feat in concept.evidence_features:
                    phrase = cfg.feature_display_names.get(
                        feat, feat.replace("_", " ")
                    ).lower()
                    if phrase in text:
                        any_ev = True
                        break
                if not any_ev:
                    errors.append(
                        "none of the evidence features appear in the explanation"
                    )

        sev_map = {"high": "high", "medium": "moderate", "low": "low"}
        expected_sev = sev_map.get(concept.severity, concept.severity)
        if expected_sev not in text:
            errors.append(f"severity '{expected_sev}' not reflected in explanation")

        if "artifact probability:" in text:
            try:
                after = text.split("artifact probability:")[1].strip().split()[0]
                stated = float(after.rstrip("."))
                if abs(stated - round(concept.artifact_probability, 2)) > 0.015:
                    errors.append(
                        f"artifact probability mismatch: stated {stated}, "
                        f"expected {concept.artifact_probability:.2f}"
                    )
            except (ValueError, IndexError):
                errors.append("could not parse artifact probability from explanation")

        if "posterior zone score:" in text:
            try:
                after = text.split("posterior zone score:")[1].strip().split()[0]
                stated = float(after.rstrip("."))
                if abs(stated - round(concept.zone_score_q, 2)) > 0.015:
                    errors.append(
                        f"zone score mismatch: stated {stated}, "
                        f"expected {concept.zone_score_q:.2f}"
                    )
            except (ValueError, IndexError):
                errors.append("could not parse posterior zone score")

        return (len(errors) == 0, errors)
