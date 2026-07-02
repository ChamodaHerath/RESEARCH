"""
Unit tests for the Bayesian manipulation-type-conditioned zone prior
(Option 2, zone_prior.py).

Covers plan item: "Zone prior: method probabilities sum to 1, unknown
threshold behavior."
"""

from __future__ import annotations

import numpy as np
import pytest

from semantic_patch_mapper import ZONE_NAMES
from zone_prior import MANIPULATION_METHODS, ZonePriorModel


@pytest.fixture
def model() -> ZonePriorModel:
    return ZonePriorModel()


def test_prior_mean_matches_alpha_over_alpha_plus_beta(model):
    for method in MANIPULATION_METHODS:
        for zone, (a, b) in model.priors[method].items():
            assert model.prior_mean(zone, method) == pytest.approx(a / (a + b))


def test_compute_zone_posterior_in_unit_interval(model):
    for method in MANIPULATION_METHODS:
        for zone in model.priors[method]:
            for R_z in (0.0, 0.25, 0.5, 0.75, 1.0):
                q = model.compute_zone_posterior(R_z, zone, method)
                assert 0.0 <= q <= 1.0


def test_compute_zone_posterior_moves_toward_evidence(model):
    """As R(z) increases, Q(z) should be monotonically non-decreasing."""
    zone, method = "mouth", "Deepfakes"
    prev = -1.0
    for R_z in np.linspace(0.0, 1.0, 11):
        q = model.compute_zone_posterior(float(R_z), zone, method)
        assert q >= prev - 1e-9
        prev = q


def test_compute_zone_posterior_at_r_zero_and_one_bounds_toward_prior_or_evidence(model):
    zone, method = "mouth", "Deepfakes"
    a, b = model.priors[method][zone]
    w = model.evidence_weight
    q0 = model.compute_zone_posterior(0.0, zone, method)
    q1 = model.compute_zone_posterior(1.0, zone, method)
    assert q0 == pytest.approx(a / (a + b + w))
    assert q1 == pytest.approx((a + w) / (a + b + w))


def test_infer_method_returns_valid_probability_distribution(model):
    # R(z) pattern that closely matches the Deepfakes signature.
    signature = np.array(
        [model._signatures["Deepfakes"].get(z, 0.5) for z in ZONE_NAMES]
    )
    best_method, probs = model.infer_method(signature, ZONE_NAMES)
    assert best_method in MANIPULATION_METHODS
    assert set(probs.keys()) == set(MANIPULATION_METHODS)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-3)
    assert all(0.0 <= p <= 1.0 for p in probs.values())


def test_infer_method_prefers_matching_signature(model):
    signature = np.array(
        [model._signatures["NeuralTextures"].get(z, 0.5) for z in ZONE_NAMES]
    )
    best_method, probs = model.infer_method(signature, ZONE_NAMES)
    assert best_method == "NeuralTextures"


def test_infer_method_returns_unknown_for_ambiguous_pattern(model):
    """A flat / low-signal R(z) vector should not confidently match any
    known method signature and should fall back to Unknown/Novel."""
    flat = np.full(len(ZONE_NAMES), 1e-6)
    best_method, probs = model.infer_method(
        flat, ZONE_NAMES, unknown_similarity_threshold=0.9999
    )
    assert best_method == "Unknown/Novel"
    assert probs == {"Unknown/Novel": 1.0}


def test_infer_method_low_threshold_never_triggers_unknown(model):
    signature = np.array(
        [model._signatures["FaceSwap"].get(z, 0.5) for z in ZONE_NAMES]
    )
    best_method, probs = model.infer_method(
        signature, ZONE_NAMES, unknown_similarity_threshold=-1.0
    )
    assert best_method != "Unknown/Novel"


def test_compute_marginal_posteriors_sums_are_bounded(model):
    R_scores = np.array([0.8, 0.2, 0.5, 0.5, 0.3, 0.3, 0.9, 0.4, 0.6])
    method_probs = {"Deepfakes": 0.7, "Face2Face": 0.3, "FaceSwap": 0.0, "NeuralTextures": 0.0}
    marginal = model.compute_marginal_posteriors(R_scores, ZONE_NAMES, method_probs=method_probs)
    assert set(marginal.keys()) == set(ZONE_NAMES)
    for v in marginal.values():
        assert 0.0 <= v <= 1.0


def test_compute_marginal_posteriors_uniform_when_method_probs_none(model):
    R_scores = np.full(len(ZONE_NAMES), 0.5)
    marginal_none = model.compute_marginal_posteriors(R_scores, ZONE_NAMES, method_probs=None)
    marginal_uniform = model.compute_marginal_posteriors(
        R_scores, ZONE_NAMES,
        method_probs={m: 1.0 / len(MANIPULATION_METHODS) for m in MANIPULATION_METHODS},
    )
    for zone in ZONE_NAMES:
        assert marginal_none[zone] == pytest.approx(marginal_uniform[zone], abs=1e-6)


def test_compute_marginal_posteriors_falls_back_to_uniform_for_unknown_method(model):
    """When method_probs is the degenerate Unknown/Novel dict, the marginal
    posterior must not crash and must fall back to a uniform mixture rather
    than silently returning all-zero posteriors."""
    R_scores = np.full(len(ZONE_NAMES), 0.6)
    marginal = model.compute_marginal_posteriors(
        R_scores, ZONE_NAMES, method_probs={"Unknown/Novel": 1.0}
    )
    for zone in ZONE_NAMES:
        assert marginal[zone] > 0.0
