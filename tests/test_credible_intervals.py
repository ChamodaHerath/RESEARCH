"""
Unit tests for Bayesian credible-interval propagation (Option 4).

Covers plan item: "CI propagation: lower/upper bounds, clipping, monotonic
sanity."
"""

from __future__ import annotations

import numpy as np
import pytest

from bayesian_calibration import (
    BayesianBetaCalibrator,
    DirichletArtifactPrior,
    compute_R_credible_interval,
)


@pytest.fixture
def A_norm() -> np.ndarray:
    return np.array([0.2, 0.8, 0.5, 0.5, 0.9, 0.1, 0.6, 0.4, 0.5], dtype=np.float32)


@pytest.fixture
def S_norm() -> np.ndarray:
    return np.array([0.6, 0.4, 0.5, 0.5, 0.1, 0.9, 0.4, 0.6, 0.5], dtype=np.float32)


def test_ci_lower_le_upper(A_norm, S_norm):
    lo, hi = compute_R_credible_interval(A_norm, S_norm, beta_mean=0.5, beta_std=0.1)
    assert np.all(lo <= hi + 1e-6)


def test_ci_bounds_clipped_to_unit_interval(A_norm, S_norm):
    # A very large beta_std should still produce bounds within [0, 1].
    lo, hi = compute_R_credible_interval(A_norm, S_norm, beta_mean=0.5, beta_std=10.0)
    assert np.all(lo >= 0.0) and np.all(lo <= 1.0)
    assert np.all(hi >= 0.0) and np.all(hi <= 1.0)


def test_ci_collapses_to_point_when_beta_std_zero(A_norm, S_norm):
    lo, hi = compute_R_credible_interval(A_norm, S_norm, beta_mean=0.5, beta_std=0.0)
    expected = 0.5 * A_norm + 0.5 * S_norm
    np.testing.assert_allclose(lo, expected, atol=1e-5)
    np.testing.assert_allclose(hi, expected, atol=1e-5)


def test_ci_width_zero_when_a_equals_s():
    """When A(z) == S(z), R(z) is independent of beta -> zero-width CI."""
    equal = np.full(9, 0.5, dtype=np.float32)
    lo, hi = compute_R_credible_interval(equal, equal, beta_mean=0.5, beta_std=0.3)
    np.testing.assert_allclose(hi - lo, np.zeros(9), atol=1e-6)


def test_ci_width_grows_monotonically_with_beta_std(A_norm, S_norm):
    prev_width = -1.0
    for beta_std in (0.0, 0.05, 0.1, 0.2, 0.3):
        lo, hi = compute_R_credible_interval(A_norm, S_norm, beta_mean=0.5, beta_std=beta_std)
        width = float(np.mean(hi - lo))
        assert width >= prev_width - 1e-9
        prev_width = width


def test_beta_calibrator_recovers_informative_beta():
    """
    Synthetic separable data where R(z, beta=1.0) perfectly separates
    fake/real (A high for fake, low for real) while S(z) is pure noise
    around 0.5. The posterior MAP beta should be pulled toward 1.0.
    """
    rng = np.random.default_rng(0)
    n = 200
    labels = np.array([1] * (n // 2) + [0] * (n // 2))
    A_raw = np.where(labels[:, None] == 1, 3.0, -3.0) + rng.normal(0, 0.1, size=(n, 9))
    S_raw = rng.normal(0, 0.1, size=(n, 9))  # pure noise, uninformative

    A_mean, A_std = A_raw.mean(axis=0), A_raw.std(axis=0) + 1e-6
    S_mean, S_std = S_raw.mean(axis=0), S_raw.std(axis=0) + 1e-6

    calibrator = BayesianBetaCalibrator(n_grid=200, scale=6.0)
    calibrator.fit(A_raw, S_raw, labels, A_mean, A_std, S_mean, S_std)
    summary = calibrator.summary()

    assert summary["map"] > 0.5, "MAP beta should favour the informative signal A(z)."
    assert summary["ci_lower"] <= summary["map"] <= summary["ci_upper"]


def test_beta_calibrator_requires_fit_before_summary():
    calibrator = BayesianBetaCalibrator()
    with pytest.raises(RuntimeError):
        calibrator.summary()


def test_dirichlet_artifact_prior_probs_sum_to_one():
    from semantic_patch_mapper import ARTIFACT_ONTOLOGY

    prior = DirichletArtifactPrior(ARTIFACT_ONTOLOGY)
    for zone in ARTIFACT_ONTOLOGY:
        probs = prior.get_posterior_probs(zone, R_score=0.7)
        assert probs.shape == (len(ARTIFACT_ONTOLOGY[zone]),)
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-6)
        assert np.all(probs >= 0.0)


def test_dirichlet_artifact_prior_roundtrip_serialisation():
    from semantic_patch_mapper import ARTIFACT_ONTOLOGY

    prior = DirichletArtifactPrior(ARTIFACT_ONTOLOGY)
    data = prior.to_dict()
    restored = DirichletArtifactPrior.from_dict(data, ARTIFACT_ONTOLOGY)
    for zone in ARTIFACT_ONTOLOGY:
        np.testing.assert_array_equal(prior._alpha[zone], restored._alpha[zone])


def test_dirichlet_high_r_score_concentrates_on_rank_zero():
    from semantic_patch_mapper import ARTIFACT_ONTOLOGY

    prior = DirichletArtifactPrior(ARTIFACT_ONTOLOGY)
    zone = "mouth"
    probs_high = prior.get_posterior_probs(zone, R_score=0.95)
    probs_low = prior.get_posterior_probs(zone, R_score=0.05)
    # Higher R_score should concentrate more mass on rank-0 than a low R_score.
    assert probs_high[0] >= probs_low[0]
