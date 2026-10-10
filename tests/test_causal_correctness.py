"""
Unit tests for the causal correctness gate.

Covers: zone occlusion drop measurement, pass/fail threshold, and filtering
of ExplanationOutput so only causally-supported claims reach the final report.
"""

from __future__ import annotations

import numpy as np
import pytest

from semantic_patch_mapper import (
    ConceptVector,
    ExplanationOutput,
    filter_explanations_by_causal_correctness,
)
from zone_localization import (
    measure_zone_causal_drops,
    zone_passes_causal_check,
)


def _make_cv(zone: str = "mouth", confidence: float = 0.8) -> ConceptVector:
    return ConceptVector(
        zone=zone,
        artifact_type="blending boundary artefact",
        severity="high",
        confidence=confidence,
        R_score=confidence,
    )


def _make_explanation(zones: list[str]) -> ExplanationOutput:
    cvs = [_make_cv(z) for z in zones]
    return ExplanationOutput(
        label="Fake",
        fake_probability=0.9,
        concept_vectors=cvs,
        nl_explanations=[
            f"Severe blending boundary artefact detected in the {z.replace('_', ' ')} region "
            f"(confidence: 80.0%)."
            for z in zones
        ],
        faithfulness_passed=[True] * len(zones),
        R_scores=np.full(9, 0.5, dtype=np.float32),
    )


def test_zone_passes_causal_check_threshold():
    assert zone_passes_causal_check(0.05, min_drop=0.02) is True
    assert zone_passes_causal_check(0.01, min_drop=0.02) is False
    assert zone_passes_causal_check(0.02, min_drop=0.02) is True


def test_measure_zone_causal_drops_reports_baseline_and_drop(sample_landmarks_68):
    img = np.full((224, 224, 3), 128, dtype=np.uint8)
    # Fake classifier: baseline 0.9; mouth occlusion → 0.5; forehead → 0.88
    def fake_prob_fn(images):
        out = []
        for i, _ in enumerate(images):
            if i == 0:
                out.append(0.9)
            elif i == 1:
                out.append(0.5)   # mouth
            else:
                out.append(0.88)  # forehead
        return out

    drops = measure_zone_causal_drops(
        fake_prob_fn=fake_prob_fn,
        image_bgr=img,
        landmarks_68=sample_landmarks_68,
        zones=["mouth", "forehead"],
    )
    assert drops["mouth"]["drop"] == pytest.approx(0.4, abs=1e-5)
    assert drops["forehead"]["drop"] == pytest.approx(0.02, abs=1e-5)


def test_filter_keeps_only_causally_supported_claims(sample_landmarks_68):
    img = np.full((224, 224, 3), 128, dtype=np.uint8)
    explanation = _make_explanation(["mouth", "forehead"])

    # mouth drops a lot; forehead barely moves
    def fake_prob_fn(images):
        # images[0]=baseline, then one per unique zone in order
        return [0.90, 0.40, 0.89]

    filtered = filter_explanations_by_causal_correctness(
        explanation=explanation,
        image_bgr=img,
        landmarks_68=sample_landmarks_68,
        fake_prob_fn=fake_prob_fn,
        min_drop=0.02,
    )
    assert [cv.zone for cv in filtered.concept_vectors] == ["mouth"]
    assert len(filtered.nl_explanations) == 1
    assert "mouth" in filtered.nl_explanations[0]
    assert filtered.causal_correctness_passed == [True]
    assert filtered.causal_drop_by_zone["mouth"] >= 0.02
    assert filtered.causal_drop_by_zone["forehead"] < 0.02
    # Detector verdict unchanged even if some claims are dropped
    assert filtered.label == "Fake"
    assert filtered.fake_probability == 0.9


def test_soft_filter_keeps_threshold_claims_and_annotates_causal(sample_landmarks_68):
    """Live demo policy: hard_filter=False keeps all claims; causal is a badge."""
    img = np.full((224, 224, 3), 128, dtype=np.uint8)
    explanation = _make_explanation(["mouth", "forehead"])

    def fake_prob_fn(images):
        return [0.90, 0.40, 0.89]

    annotated = filter_explanations_by_causal_correctness(
        explanation=explanation,
        image_bgr=img,
        landmarks_68=sample_landmarks_68,
        fake_prob_fn=fake_prob_fn,
        min_drop=0.02,
        hard_filter=False,
    )
    assert [cv.zone for cv in annotated.concept_vectors] == ["mouth", "forehead"]
    assert len(annotated.nl_explanations) == 2
    assert annotated.causal_correctness_passed == [True, False]
    assert annotated.causal_drop_by_zone["mouth"] >= 0.02
    assert annotated.causal_drop_by_zone["forehead"] < 0.02


def test_filter_drops_all_when_no_zone_passes_causal(sample_landmarks_68):
    """Causal gate is strict: no Fake soft-keep of below-evidence zones."""
    img = np.full((224, 224, 3), 128, dtype=np.uint8)
    explanation = _make_explanation(["mouth", "forehead"])
    for cv in explanation.concept_vectors:
        cv.flag_reason = "calibrated_threshold"

    def fake_prob_fn(images):
        return [0.90] + [0.899] * (len(images) - 1)

    filtered = filter_explanations_by_causal_correctness(
        explanation=explanation,
        image_bgr=img,
        landmarks_68=sample_landmarks_68,
        fake_prob_fn=fake_prob_fn,
        min_drop=0.02,
    )
    assert filtered.concept_vectors == []
    assert filtered.nl_explanations == []
    assert filtered.label == "Fake"


def test_filter_empty_when_real_and_no_zone_moves_score(sample_landmarks_68):
    img = np.full((224, 224, 3), 128, dtype=np.uint8)
    explanation = _make_explanation(["mouth"])
    explanation.fake_probability = 0.1
    explanation.label = "Real"

    def fake_prob_fn(images):
        return [0.10] * len(images)

    filtered = filter_explanations_by_causal_correctness(
        explanation=explanation,
        image_bgr=img,
        landmarks_68=sample_landmarks_68,
        fake_prob_fn=fake_prob_fn,
        min_drop=0.02,
    )
    assert filtered.concept_vectors == []
    assert filtered.nl_explanations == []
    assert filtered.label == "Real"


def test_filter_noop_on_empty_explanations(sample_landmarks_68):
    img = np.full((224, 224, 3), 128, dtype=np.uint8)
    explanation = _make_explanation([])
    filtered = filter_explanations_by_causal_correctness(
        explanation=explanation,
        image_bgr=img,
        landmarks_68=sample_landmarks_68,
        fake_prob_fn=lambda imgs: [0.5] * len(imgs),
        min_drop=0.02,
    )
    assert filtered.concept_vectors == []
    assert filtered.causal_drop_by_zone == {}
