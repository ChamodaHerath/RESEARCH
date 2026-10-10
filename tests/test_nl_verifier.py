"""
Unit tests for NL template generation and the faithfulness verifier
(Module 4a/4b).

Covers plan item: "NL verifier: valid and invalid templates."
"""

from __future__ import annotations

import numpy as np
import pytest

import semantic_patch_mapper
from semantic_patch_mapper import (
    ConceptVector,
    build_concept_vector,
    generate_nl_explanation,
    generate_verified_explanation,
    verify_faithfulness,
)


def _make_cv(**overrides) -> ConceptVector:
    defaults = dict(
        zone="mouth",
        artifact_type="blending boundary artefact",
        severity="high",
        confidence=0.82,
        R_score=0.82,
    )
    defaults.update(overrides)
    return ConceptVector(**defaults)


def test_generate_nl_explanation_contains_zone_and_artifact():
    cv = _make_cv()
    sentence = generate_nl_explanation(cv)
    assert "mouth" in sentence.lower()
    assert "blending boundary artefact" in sentence.lower()
    assert "82.0%" in sentence


def test_generate_nl_explanation_includes_ci_when_present():
    cv = _make_cv(R_ci_lower=0.70, R_ci_upper=0.90)
    sentence = generate_nl_explanation(cv)
    assert "70.0" in sentence and "90.0" in sentence
    assert "%]" in sentence


def test_generate_nl_explanation_omits_ci_when_absent():
    cv = _make_cv()
    sentence = generate_nl_explanation(cv)
    assert "[" not in sentence


def test_verify_faithfulness_passes_for_generated_sentence():
    cv = _make_cv()
    sentence = generate_nl_explanation(cv)
    assert verify_faithfulness(sentence, cv) is True


@pytest.mark.parametrize(
    "bad_sentence",
    [
        "A high anomaly was detected somewhere on the face.",   # missing zone + artifact
        "High blending boundary artefact detected in the eye region (confidence: 82.0%).",  # wrong zone
        "detected in the mouth region (confidence: 82.0%).",     # missing severity adjective + artifact
    ],
)
def test_verify_faithfulness_fails_for_unfaithful_sentences(bad_sentence):
    cv = _make_cv()
    assert verify_faithfulness(bad_sentence, cv) is False


def test_generate_verified_explanation_always_passes_for_template_output():
    cv = _make_cv(zone="left_eye", artifact_type="iris texture inconsistency", severity="medium", confidence=0.55, R_score=0.55)
    sentence, passed = generate_verified_explanation(cv)
    assert passed is True
    assert "left eye" in sentence.lower()


def test_build_concept_vector_uses_rank0_artifact_by_default():
    cv = build_concept_vector(zone="mouth", R_score=0.8)
    assert cv.artifact_type == "blending boundary artefact"  # rank-0 in ARTIFACT_ONTOLOGY["mouth"]
    assert cv.severity == "high"


def test_build_concept_vector_uses_dirichlet_map_artifact_when_posterior_given():
    # Posterior strongly favouring rank-1 artifact.
    posterior = np.array([0.05, 0.7, 0.1, 0.1, 0.05], dtype=np.float64)
    cv = build_concept_vector(zone="mouth", R_score=0.8, artifact_posterior=posterior)
    assert cv.artifact_type == "lip sync inconsistency"  # rank-1 in ARTIFACT_ONTOLOGY["mouth"]
    assert cv.artifact_probs is not None


def test_build_concept_vector_rejects_unknown_zone():
    with pytest.raises(KeyError):
        build_concept_vector(zone="not_a_real_zone", R_score=0.5)


def test_build_concept_vector_zone_prior_score_overrides_severity_and_confidence():
    # R_score is high (would be "high" severity) but zone_prior_score (Q(z)) is low.
    cv = build_concept_vector(zone="mouth", R_score=0.9, zone_prior_score=0.2)
    assert cv.confidence == pytest.approx(0.2)
    assert cv.severity == "low"
    assert cv.R_score == pytest.approx(0.9)  # raw R_score is preserved separately


# ---------------------------------------------------------------------------
# Strengthened faithfulness checks: confidence % and CI bounds
# ---------------------------------------------------------------------------

def test_verify_faithfulness_fails_on_wrong_confidence_percentage():
    """Zone/artifact/severity all correct, but the stated number is wrong --
    the original (pre-strengthening) verifier could not catch this."""
    cv = _make_cv(confidence=0.82, R_score=0.82)
    sentence = (
        "High-confidence blending boundary artefact detected in the mouth region "
        "(confidence: 55.0%)."
    )
    assert verify_faithfulness(sentence, cv) is False


def test_verify_faithfulness_passes_confidence_within_rounding_tolerance():
    cv = _make_cv(confidence=0.8199999, R_score=0.82)  # rounds to 82.0%
    sentence = (
        "High-confidence blending boundary artefact detected in the mouth region "
        "(confidence: 82.0%)."
    )
    assert verify_faithfulness(sentence, cv) is True


def test_verify_faithfulness_fails_on_wrong_ci_bounds():
    cv = _make_cv(confidence=0.82, R_score=0.82, R_ci_lower=0.70, R_ci_upper=0.90)
    sentence = (
        "High-confidence blending boundary artefact detected in the mouth region "
        "(confidence: 82.0% [10.0\u201320.0%])."
    )
    assert verify_faithfulness(sentence, cv) is False


def test_verify_faithfulness_fails_when_cv_has_ci_but_sentence_omits_it():
    cv = _make_cv(confidence=0.82, R_score=0.82, R_ci_lower=0.70, R_ci_upper=0.90)
    sentence = (
        "High-confidence blending boundary artefact detected in the mouth region "
        "(confidence: 82.0%)."
    )
    assert verify_faithfulness(sentence, cv) is False


def test_verify_faithfulness_fails_when_sentence_invents_a_ci_cv_does_not_have():
    cv = _make_cv(confidence=0.82, R_score=0.82)  # no CI on the concept vector
    sentence = (
        "High-confidence blending boundary artefact detected in the mouth region "
        "(confidence: 82.0% [70.0\u201390.0%])."
    )
    assert verify_faithfulness(sentence, cv) is False


def test_generate_nl_explanation_ci_round_trips_through_verifier():
    cv = _make_cv(confidence=0.82, R_score=0.82, R_ci_lower=0.70, R_ci_upper=0.90)
    sentence = generate_nl_explanation(cv)
    assert verify_faithfulness(sentence, cv) is True


# ---------------------------------------------------------------------------
# Fallback path
# ---------------------------------------------------------------------------

def test_generate_verified_explanation_falls_back_when_primary_template_is_corrupted(monkeypatch):
    """If the primary NL generator unexpectedly produces an unfaithful sentence
    (e.g. a future template bug, or upstream corruption), the fallback must
    reconstruct a sentence directly from the ConceptVector's own fields, and
    that fallback sentence must itself pass verification -- `passed` should
    reflect the sentence actually returned, not the failed first attempt."""
    cv = _make_cv(
        zone="mouth",
        artifact_type="blending boundary artefact",
        severity="high",
        confidence=0.82,
        R_score=0.82,
        R_ci_lower=0.70,
        R_ci_upper=0.90,
    )

    def _broken_nl_explanation(_cv):
        return "A vague anomaly was detected somewhere on the face."

    monkeypatch.setattr(semantic_patch_mapper, "generate_nl_explanation", _broken_nl_explanation)

    sentence, passed = semantic_patch_mapper.generate_verified_explanation(cv)

    assert passed is True
    assert "mouth" in sentence.lower()
    assert "blending boundary artefact" in sentence.lower()
    assert "82.0%" in sentence
    assert "70.0" in sentence and "90.0" in sentence
    assert verify_faithfulness(sentence, cv) is True


def test_generate_verified_explanation_fallback_still_fails_if_cv_itself_is_unusable(monkeypatch):
    """Sanity check: the fallback re-verification is not a rubber stamp -- if
    even the reconstructed sentence can't satisfy verify_faithfulness (here,
    by breaking verify_faithfulness itself to always reject), passed must be
    False rather than being forced to True."""
    cv = _make_cv()

    def _broken_nl_explanation(_cv):
        return "unfaithful text"

    def _always_fail(_sentence, _cv):
        return False

    monkeypatch.setattr(semantic_patch_mapper, "generate_nl_explanation", _broken_nl_explanation)
    monkeypatch.setattr(semantic_patch_mapper, "verify_faithfulness", _always_fail)

    sentence, passed = semantic_patch_mapper.generate_verified_explanation(cv)
    assert passed is False
    assert isinstance(sentence, str) and len(sentence) > 0
