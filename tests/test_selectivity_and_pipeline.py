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


def test_get_flagged_zones_zone_thresholds_do_not_require_severity_floor():
    """Calibrated zone_thresholds alone decide flagging — a score that clears
    the per-zone cutoff must be kept even if assign_severity would return None
    (score ≤ 0.25). This was the double-gate bug."""
    R = np.array([0.20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    thresholds = {zone: 0.10 for zone in ZONE_NAMES}
    flagged = get_flagged_zones(R, min_severity="medium", zone_thresholds=thresholds)
    assert len(flagged) == 1
    assert flagged[0]["zone"] == "left_eye"


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
    assert out.concept_vectors == []
    assert out.nl_explanations == []


def test_real_never_flags_even_when_scores_clear_thresholds(
    sample_landmarks_68, sample_alpha, sample_phi
):
    """Real verdict ⇒ zero flags, even if Q(z) would clear calibrated bars."""
    out = predict_label_and_explanation(
        attention_vec=sample_alpha,
        landmarks_68=sample_landmarks_68,
        phi_vec=sample_phi,
        beta=0.5,
        fake_probability=0.2,
        min_severity="low",
        zone_thresholds={zone: 0.0 for zone in ZONE_NAMES},
        zone_prior_model=ZonePriorModel(),
    )
    assert out.label == "Real"
    assert out.concept_vectors == []
    assert out.nl_explanations == []
    # Scores are still computed for inspection
    assert out.R_scores.shape == (9,)


def test_weak_fake_below_confidence_gate_emits_no_flags(
    sample_landmarks_68, sample_alpha, sample_phi
):
    """P(fake)=0.55 is Fake but below the 0.70 research confidence gate."""
    out = predict_label_and_explanation(
        attention_vec=sample_alpha,
        landmarks_68=sample_landmarks_68,
        phi_vec=sample_phi,
        beta=0.5,
        fake_probability=0.55,
        min_severity="low",
        zone_thresholds={zone: 0.0 for zone in ZONE_NAMES},
        zone_prior_model=ZonePriorModel(),
        min_fake_probability_for_explanations=0.70,
    )
    assert out.label == "Fake"
    assert out.concept_vectors == []
    assert out.nl_explanations == []


def test_confident_fake_with_zero_thresholds_can_flag(
    sample_landmarks_68, sample_alpha, sample_phi
):
    out = predict_label_and_explanation(
        attention_vec=sample_alpha,
        landmarks_68=sample_landmarks_68,
        phi_vec=sample_phi,
        beta=0.5,
        fake_probability=0.91,
        min_severity="low",
        zone_thresholds={zone: 0.0 for zone in ZONE_NAMES},
        zone_prior_model=ZonePriorModel(),
        max_flagged_zones=3,
        min_fake_probability_for_explanations=0.70,
    )
    assert out.label == "Fake"
    assert len(out.concept_vectors) == 3
    assert len(out.nl_explanations) == 3


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


def test_get_flagged_zones_score_strictly_above_threshold_is_flagged():
    """Flag only when score > threshold (not ≥). Equal to the bar stays unflagged."""
    R = np.zeros(9, dtype=np.float32)
    mouth_i = ZONE_NAMES.index("mouth")
    thresholds = {zone: 0.99 for zone in ZONE_NAMES}
    thresholds["mouth"] = 0.669

    R[mouth_i] = 0.669  # equal → not flagged
    assert get_flagged_zones(R, min_severity="high", zone_thresholds=thresholds) == []

    R[mouth_i] = 0.739  # higher → flagged
    flagged = get_flagged_zones(R, min_severity="high", zone_thresholds=thresholds)
    assert len(flagged) == 1
    assert flagged[0]["zone"] == "mouth"


def test_threshold_clear_includes_highest_score_zone():
    """Mouth above its bar must be flagged; zones at/under their bars must not."""
    R = np.full(9, 0.2, dtype=np.float32)
    Q = np.full(9, 0.2, dtype=np.float32)
    Q[ZONE_NAMES.index("mouth")] = 0.739
    Q[ZONE_NAMES.index("nose_tip")] = 0.493  # equal to thr → not flagged
    Q[ZONE_NAMES.index("right_cheek")] = 0.567
    Q_scores = {z: float(Q[i]) for i, z in enumerate(ZONE_NAMES)}
    thresholds = {
        "left_eye": 0.57,
        "right_eye": 0.55,
        "nose_bridge": 0.47,
        "nose_tip": 0.493,
        "left_cheek": 0.66,
        "right_cheek": 0.68,
        "mouth": 0.669,
        "forehead": 0.55,
        "chin_jaw": 0.64,
    }
    cvs = build_concept_vectors_from_R(
        R,
        min_severity="medium",
        Q_scores=Q_scores,
        zone_thresholds=thresholds,
        max_zones=3,
    )
    zones = [cv.zone for cv in cvs]
    assert zones == ["mouth"]
    assert cvs[0].flag_reason == "calibrated_threshold"


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
    """Zone flagging must use Q(z) vs per-zone thresholds: score > thr only."""
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

    # Thresholds above every Q(z) → nothing flagged (no top-k fallback).
    strict_thresholds = {
        zone: float(baseline.Q_scores[i]) + 0.01 for i, zone in enumerate(ZONE_NAMES)
    }
    strict = predict_label_and_explanation(
        **kwargs, zone_thresholds=strict_thresholds, max_flagged_zones=3
    )
    assert strict.concept_vectors == []
    assert strict.nl_explanations == []
    assert strict.label == "Fake"

    # Zero thresholds → every zone with Q > 0 is eligible (capped by max_zones).
    loose_thresholds = {zone: 0.0 for zone in ZONE_NAMES}
    loose = predict_label_and_explanation(
        **kwargs, zone_thresholds=loose_thresholds, max_flagged_zones=3
    )
    assert len(loose.concept_vectors) == 3

    for cv in loose.concept_vectors:
        z_idx = ZONE_NAMES.index(cv.zone)
        assert cv.confidence == pytest.approx(float(baseline.Q_scores[z_idx]), abs=1e-4)


def test_impossible_thresholds_flag_nothing_even_when_fake(
    sample_landmarks_68, sample_alpha, sample_phi
):
    """Accuracy rule: Fake does not invent flags below the calibrated bar."""
    out = predict_label_and_explanation(
        attention_vec=sample_alpha,
        landmarks_68=sample_landmarks_68,
        phi_vec=sample_phi,
        beta=0.5,
        fake_probability=0.91,
        min_severity="high",
        zone_thresholds={zone: 0.999 for zone in ZONE_NAMES},
        zone_prior_model=ZonePriorModel(),
        max_flagged_zones=3,
    )
    assert out.label == "Fake"
    assert out.nl_explanations == []
    assert out.concept_vectors == []


def test_real_can_have_zero_nl_sentences(sample_landmarks_68, sample_alpha, sample_phi):
    out = predict_label_and_explanation(
        attention_vec=sample_alpha,
        landmarks_68=sample_landmarks_68,
        phi_vec=sample_phi,
        beta=0.5,
        fake_probability=0.1,
        min_severity="high",
        zone_thresholds={zone: 0.999 for zone in ZONE_NAMES},
    )
    assert out.label == "Real"
    assert out.nl_explanations == []


def test_semantic_generator_uses_beta_posterior_mean_for_fusion(tmp_path):
    """Accuracy fix: fusion β is posterior mean, not MAP=1.0 (which zeros S)."""
    import json
    from semantic_patch_mapper import SemanticExplanationGenerator, ZONE_NAMES

    cfg = {
        "beta": 1.0,
        "beta_for_fusion": "posterior_mean",
        "beta_posterior": {
            "map": 1.0,
            "mean": 0.9227,
            "std": 0.07,
            "ci_lower": 0.74,
            "ci_upper": 0.998,
        },
        "zone_thresholds": {z: 0.99 for z in ZONE_NAMES},
        "max_flagged_zones": 3,
        "min_fake_probability_for_explanations": 0.70,
        "causal_min_drop": 0.03,
        "evidence_weight": 10.0,
        "A_stats": {"mean": [0.0] * 9, "std": [1.0] * 9},
        "S_stats": {"mean": [0.0] * 9, "std": [1.0] * 9},
    }
    path = tmp_path / "fusion_test.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    gen = SemanticExplanationGenerator(config_path=str(path))
    assert gen.beta_map == pytest.approx(1.0)
    assert gen.beta == pytest.approx(0.9227)
    assert gen.causal_min_drop == pytest.approx(0.03)
