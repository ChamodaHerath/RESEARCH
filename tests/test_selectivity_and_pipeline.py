"""
Unit tests for zone-flagging selectivity (Phase 4) and the full
predict_label_and_explanation() orchestration function (pure-numpy inputs,
no ViT model required).
"""

from __future__ import annotations

import numpy as np
import pytest

from semantic_patch_mapper import (
    ZONE_NAMES,
    build_concept_vectors_from_R,
    get_flagged_zones,
    predict_label_and_explanation,
)
from zone_prior import ZonePriorModel


def test_get_flagged_zones_respects_min_severity():
    R = np.array([0.9, 0.6, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    flagged_high = get_flagged_zones(R, min_severity="high")
    flagged_low = get_flagged_zones(R, min_severity="low")
    assert {f["zone"] for f in flagged_high} == {"left_eye"}
    assert len(flagged_low) >= len(flagged_high)


def test_get_flagged_zones_sorted_descending_by_score():
    R = np.array([0.3, 0.9, 0.6, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    flagged = get_flagged_zones(R, min_severity="low")
    scores = [f["R_score"] for f in flagged]
    assert scores == sorted(scores, reverse=True)


def test_get_flagged_zones_max_zones_caps_output():
    R = np.full(9, 0.9, dtype=np.float32)  # every zone would qualify
    flagged_all = get_flagged_zones(R, min_severity="low")
    flagged_capped = get_flagged_zones(R, min_severity="low", max_zones=3)
    assert len(flagged_all) == 9
    assert len(flagged_capped) == 3


def test_get_flagged_zones_per_zone_thresholds_are_selective():
    """A per-zone threshold above the observed score should suppress flagging
    even though the global severity gate would have passed it — this is the
    Phase 4 fix for universal zone over-flagging."""
    R = np.array([0.6] * 9, dtype=np.float32)
    thresholds = {zone: 0.95 for zone in ZONE_NAMES}  # near-impossible to clear
    flagged = get_flagged_zones(R, min_severity="low", zone_thresholds=thresholds)
    assert flagged == []


def test_build_concept_vectors_from_r_uses_q_scores_for_flagging_when_given():
    R = np.array([0.6] * 9, dtype=np.float32)  # would all pass on R alone
    Q_scores = {zone: 0.1 for zone in ZONE_NAMES}  # but posterior says "not anomalous"
    cvs = build_concept_vectors_from_R(R, min_severity="low", Q_scores=Q_scores)
    assert cvs == []


def test_predict_label_and_explanation_shapes(sample_landmarks_68, sample_alpha, sample_phi):
    out = predict_label_and_explanation(
        attention_vec=sample_alpha,
        landmarks_68=sample_landmarks_68,
        phi_vec=sample_phi,
        beta=0.5,
        fake_probability=0.8,
        min_severity="low",
    )
    assert out.R_scores.shape == (9,)
    assert out.label in ("Fake", "Real")
    assert len(out.nl_explanations) == len(out.concept_vectors)
    assert len(out.faithfulness_passed) == len(out.concept_vectors)


def test_predict_label_and_explanation_real_label_when_probability_low(sample_landmarks_68, sample_alpha, sample_phi):
    out = predict_label_and_explanation(
        attention_vec=sample_alpha,
        landmarks_68=sample_landmarks_68,
        phi_vec=sample_phi,
        beta=0.5,
        fake_probability=0.1,
        min_severity="low",
    )
    assert out.label == "Real"


def test_predict_label_and_explanation_ci_is_on_same_scale_as_reported_confidence(
    sample_landmarks_68, sample_alpha, sample_phi
):
    """Regression test: when a ZonePriorModel is active, the confidence figure
    reported for a flagged zone is Q(z), so its 95% CI must also be expressed
    on the Q(z) scale (not R(z)) -- otherwise a sentence like
    "confidence: 73% [99-100%]" is internally contradictory (the point
    estimate doesn't even fall inside its own stated interval)."""
    out = predict_label_and_explanation(
        attention_vec=sample_alpha,
        landmarks_68=sample_landmarks_68,
        phi_vec=sample_phi,
        beta=0.5,
        fake_probability=0.9,
        min_severity="low",
        beta_mean=0.5,
        beta_std=0.15,
        zone_prior_model=ZonePriorModel(),
    )
    assert out.concept_vectors, "Expected at least one flagged zone for this fixture."
    for cv in out.concept_vectors:
        assert cv.zone_prior_score is not None  # confidence is Q(z)-based here
        if cv.R_ci_lower is not None and cv.R_ci_upper is not None:
            # The reported confidence must fall within its own stated interval
            # (within float rounding), which only holds if both are on the
            # same (Q(z)) scale.
            assert cv.R_ci_lower - 1e-6 <= cv.confidence <= cv.R_ci_upper + 1e-6


def test_predict_label_and_explanation_with_zone_prior_model_populates_method_fields(
    sample_landmarks_68, sample_alpha, sample_phi
):
    out = predict_label_and_explanation(
        attention_vec=sample_alpha,
        landmarks_68=sample_landmarks_68,
        phi_vec=sample_phi,
        beta=0.5,
        fake_probability=0.9,
        min_severity="low",
        zone_prior_model=ZonePriorModel(),
    )
    assert out.inferred_method is not None
    assert out.method_probabilities is not None
    assert out.Q_scores is not None
    assert out.Q_scores.shape == (9,)


def test_predict_label_and_explanation_max_flagged_zones_is_respected(
    sample_landmarks_68, sample_alpha, sample_phi
):
    out = predict_label_and_explanation(
        attention_vec=sample_alpha,
        landmarks_68=sample_landmarks_68,
        phi_vec=sample_phi,
        beta=0.5,
        fake_probability=0.9,
        min_severity="low",
        max_flagged_zones=2,
    )
    assert len(out.concept_vectors) <= 2


def test_predict_label_and_explanation_rejects_wrong_attention_shape(sample_landmarks_68):
    with pytest.raises(ValueError):
        predict_label_and_explanation(
            attention_vec=np.zeros(10, dtype=np.float32),
            landmarks_68=sample_landmarks_68,
        )


def test_predict_label_and_explanation_combines_zone_thresholds_with_q_scores(
    sample_landmarks_68, sample_alpha, sample_phi
):
    """End-to-end selectivity check for the exact configuration
    SemanticExplanationGenerator uses in production (a ZonePriorModel *and*
    calibrated zone_thresholds supplied together, as loaded from
    fusion_config.json): zone flagging must threshold the Bayesian Q(z)
    posterior, not raw R(z) -- this is the Phase 4 fix for universal
    zone over-flagging (see README "Selectivity")."""
    zone_prior_model = ZonePriorModel()
    kwargs = dict(
        attention_vec=sample_alpha,
        landmarks_68=sample_landmarks_68,
        phi_vec=sample_phi,
        beta=0.5,
        fake_probability=0.9,
        min_severity="low",
        zone_prior_model=zone_prior_model,
    )

    baseline = predict_label_and_explanation(**kwargs)
    assert baseline.Q_scores is not None and baseline.Q_scores.shape == (9,)

    # Threshold every zone just above its own Q(z) -- nothing should clear it,
    # which is only possible if flagging actually reads Q(z) (not R(z), which
    # could be arbitrarily different from Q(z) and might spuriously pass).
    strict_thresholds = {
        zone: float(baseline.Q_scores[i]) + 0.01 for i, zone in enumerate(ZONE_NAMES)
    }
    strict = predict_label_and_explanation(**kwargs, zone_thresholds=strict_thresholds)
    assert strict.concept_vectors == []

    # Zero out every threshold -- flagging should return for every zone whose
    # Q(z) still clears the independent severity floor.
    loose_thresholds = {zone: 0.0 for zone in ZONE_NAMES}
    loose = predict_label_and_explanation(**kwargs, zone_thresholds=loose_thresholds)
    assert len(loose.concept_vectors) >= len(strict.concept_vectors)

    # Confidence on every flagged zone must be Q(z), not the raw fused R(z),
    # confirming the posterior (not the fusion score) is what's being reported.
    for cv in loose.concept_vectors:
        z_idx = ZONE_NAMES.index(cv.zone)
        assert cv.confidence == pytest.approx(float(baseline.Q_scores[z_idx]), abs=1e-4)
