"""
Unit tests for BayesianArtifactInferencer, abstention, NL, and faithfulness.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifact_analysis.artifact_inference import (
    ArtifactEvidenceAnalyzer,
    BayesianArtifactInferencer,
    ExtendedConceptVector,
    ExtendedFaithfulnessVerifier,
    analyze_zone_artifact,
    generate_artifact_explanation,
    normalize_features,
)
from artifact_analysis.artifact_ontology import ARTIFACT_TYPES, load_artifact_config
from artifact_analysis.feature_extractor import ZoneFeatureExtractor
from tests.conftest import make_synthetic_landmarks_68

IMAGE_SIZE = 224


def _disk(cx: int, cy: int, r: int) -> np.ndarray:
    yy, xx = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def _skin(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), (180, 140, 120), dtype=np.uint8)
    noise = rng.integers(-10, 11, size=base.shape, dtype=np.int16)
    return np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _boundary_features() -> dict:
    """Hand-crafted normalized-like raw features favoring blending_boundary."""
    return {
        "zone_pixel_count": 2000.0,
        "boundary_gradient_mean": 70.0,
        "boundary_discontinuity": 55.0,
        "inside_outside_color_difference": 50.0,
        "edge_density_boundary": 0.8,
        "canny_edge_density": 0.6,
        "boundary_sharpness_ratio": 3.0,
        "lbp_entropy": 1.0,
        "laplacian_variance": 20.0,
        "delta_e_neighbor": 2.0,
        "blocking_score": 0.5,
        "high_low_frequency_ratio": 0.5,
        "landmark_symmetry_error": 0.01,
    }


class TestNormalizeAndSoftmax:
    def test_normalize_clips(self) -> None:
        out = normalize_features({"delta_e_neighbor": 1000.0, "zone_pixel_count": 10})
        assert out["delta_e_neighbor"] == pytest.approx(1.0)
        assert out["zone_pixel_count"] == 10.0

    def test_missing_feature_values_safe(self) -> None:
        inf = BayesianArtifactInferencer()
        # Only a subset of features present
        feats = {"zone_pixel_count": 500.0, "boundary_discontinuity": 40.0}
        pred = inf.predict("mouth", feats, r_score=0.8, q_score=0.85)
        assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-6


class TestAbstention:
    def test_tiny_mask_abstains(self) -> None:
        inf = BayesianArtifactInferencer()
        pred = inf.predict(
            "mouth",
            {"zone_pixel_count": 5.0, "boundary_discontinuity": 50.0},
            r_score=0.9,
            q_score=0.9,
        )
        assert pred.abstained
        assert pred.artifact_type == "no_clear_artifact"
        assert pred.abstention_reason == "zone_mask_too_small"

    def test_low_q_abstains(self) -> None:
        inf = BayesianArtifactInferencer()
        pred = inf.predict(
            "mouth",
            _boundary_features(),
            r_score=0.9,
            q_score=0.1,
        )
        assert pred.abstained
        assert pred.abstention_reason == "q_score_below_threshold"

    def test_poor_quality_abstains(self) -> None:
        inf = BayesianArtifactInferencer()
        pred = inf.predict(
            "mouth",
            _boundary_features(),
            r_score=0.9,
            q_score=0.9,
            image_quality_laplacian_var=0.1,
        )
        assert pred.abstained
        assert pred.abstention_reason == "poor_image_quality"

    def test_low_probability_or_margin_abstains(self) -> None:
        cfg = load_artifact_config()
        cfg.abstention.min_probability = 0.99
        inf = BayesianArtifactInferencer(config=cfg)
        pred = inf.predict("mouth", _boundary_features(), 0.9, 0.9)
        assert pred.abstained


class TestInference:
    def test_probability_sum_equals_one(self) -> None:
        inf = BayesianArtifactInferencer()
        pred = inf.predict("mouth", _boundary_features(), 0.85, 0.88)
        s = sum(pred.probabilities.values())
        assert s == pytest.approx(1.0, abs=1e-6)
        assert set(pred.probabilities.keys()) == set(ARTIFACT_TYPES)

    def test_deterministic_output(self) -> None:
        inf = BayesianArtifactInferencer()
        a = inf.predict("mouth", _boundary_features(), 0.85, 0.88)
        b = inf.predict("mouth", _boundary_features(), 0.85, 0.88)
        assert a.artifact_type == b.artifact_type
        assert a.probability == pytest.approx(b.probability)
        assert a.probabilities == b.probabilities

    def test_boundary_features_favor_blending(self) -> None:
        inf = BayesianArtifactInferencer()
        pred = inf.predict("mouth", _boundary_features(), 0.9, 0.9)
        # May abstain if margin tight; if not abstaining, blending should rank high
        if not pred.abstained:
            assert pred.artifact_type == "blending_boundary"
        else:
            assert pred.probabilities["blending_boundary"] >= max(
                v for k, v in pred.probabilities.items() if k != "blending_boundary"
            ) - 1e-6 or pred.probabilities["blending_boundary"] == max(
                pred.probabilities.values()
            )

    def test_r_q_do_not_alone_select_artifact(self) -> None:
        """Changing R/Q should not invent a strong class from empty evidence."""
        inf = BayesianArtifactInferencer()
        weak = {"zone_pixel_count": 500.0}
        low = inf.predict("mouth", weak, 0.2, 0.55)
        high = inf.predict("mouth", weak, 0.99, 0.99)
        # Both should abstain or select no_clear — not a specific artifact from R/Q alone
        assert low.artifact_type == "no_clear_artifact" or low.abstained
        assert high.artifact_type == "no_clear_artifact" or high.abstained


class TestExplanationAndFaithfulness:
    def test_abstention_explanation_no_specific_artifact(self) -> None:
        concept = ExtendedConceptVector(
            zone="mouth",
            artifact_type="no_clear_artifact",
            severity="high",
            zone_score_r=0.8,
            zone_score_q=0.86,
            artifact_probability=0.4,
            credible_interval=(0.3, 0.5),
            evidence_features={},
            abstained=True,
        )
        text = generate_artifact_explanation(concept)
        assert "could not be determined reliably" in text
        assert "blending boundary" not in text.lower()
        ok, errors = ExtendedFaithfulnessVerifier().verify(text, concept)
        assert ok, errors

    def test_evidence_explanation_and_faithfulness(self) -> None:
        concept = ExtendedConceptVector(
            zone="mouth",
            artifact_type="blending_boundary",
            severity="high",
            zone_score_r=0.82,
            zone_score_q=0.86,
            artifact_probability=0.78,
            credible_interval=(0.74, 0.91),
            evidence_features={
                "boundary_discontinuity": 0.76,
                "inside_outside_color_difference": 0.69,
            },
            abstained=False,
        )
        text = generate_artifact_explanation(concept)
        assert "most likely observable artifact" in text
        assert "blending boundary" in text
        assert "0.86" in text
        assert "0.78" in text
        ok, errors = ExtendedFaithfulnessVerifier().verify(text, concept)
        assert ok, errors

    def test_faithfulness_detects_zone_mismatch(self) -> None:
        concept = ExtendedConceptVector(
            zone="mouth",
            artifact_type="blending_boundary",
            severity="high",
            zone_score_r=0.8,
            zone_score_q=0.8,
            artifact_probability=0.7,
            credible_interval=(0.6, 0.8),
            evidence_features={"boundary_discontinuity": 0.9},
        )
        bad = "High manipulation evidence was detected in the left eye region."
        ok, errors = ExtendedFaithfulnessVerifier().verify(bad, concept)
        assert not ok
        assert any("zone" in e for e in errors)


class TestEndToEndAnalyzer:
    def test_analyze_zone_artifact_runs(self) -> None:
        img = _skin(1)
        mask = _disk(112, 165, 28)
        neigh = _disk(112, 165, 50) & (~mask)
        # Inject a hard boundary to encourage blending evidence
        dil = _disk(112, 165, 30) & (~mask)
        img = img.copy()
        img[dil] = (15, 15, 15)
        lm = make_synthetic_landmarks_68()
        analyzer = ArtifactEvidenceAnalyzer()
        concept = analyze_zone_artifact(
            image_rgb=img,
            zone_name="mouth",
            zone_mask=mask,
            neighboring_mask=neigh,
            landmarks=lm,
            r_score=0.85,
            q_score=0.88,
            artifact_analyzer=analyzer,
        )
        assert concept.zone == "mouth"
        assert abs(sum(concept.probabilities.values()) - 1.0) < 1e-5
        text = generate_artifact_explanation(concept)
        ok, errors = ExtendedFaithfulnessVerifier().verify(text, concept)
        assert ok, errors
