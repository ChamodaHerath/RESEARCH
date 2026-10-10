"""
Unit tests for dual-score fusion math (Module 2): R(z) = beta*A(z) + (1-beta)*S(z).

Covers plan item: "Fusion math: beta extremes and beta midpoint."
"""

from __future__ import annotations

import numpy as np
import pytest

from semantic_patch_mapper import (
    NUM_ZONES,
    fuse_scores,
    compute_population_zone_stats,
    normalize_zone_scores,
)


@pytest.fixture
def A() -> np.ndarray:
    return np.array([0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.5], dtype=np.float32)


@pytest.fixture
def S() -> np.ndarray:
    return np.array([0.9, 0.1, 0.7, 0.3, 0.8, 0.2, 0.6, 0.4, 0.5], dtype=np.float32)


def test_fuse_scores_beta_one_returns_attention_only(A, S):
    R = fuse_scores(A, S, beta=1.0)
    np.testing.assert_allclose(R, A, atol=1e-6)


def test_fuse_scores_beta_zero_returns_gradient_only(A, S):
    R = fuse_scores(A, S, beta=0.0)
    np.testing.assert_allclose(R, S, atol=1e-6)


def test_fuse_scores_beta_half_is_mean(A, S):
    R = fuse_scores(A, S, beta=0.5)
    np.testing.assert_allclose(R, (A + S) / 2.0, atol=1e-6)


def test_fuse_scores_output_shape_and_dtype(A, S):
    R = fuse_scores(A, S, beta=0.3)
    assert R.shape == (NUM_ZONES,)
    assert R.dtype == np.float32


@pytest.mark.parametrize("bad_beta", [-0.01, 1.01, -1.0, 2.0])
def test_fuse_scores_rejects_out_of_range_beta(A, S, bad_beta):
    with pytest.raises(ValueError):
        fuse_scores(A, S, beta=bad_beta)


def test_fuse_scores_is_convex_combination(A, S):
    """R(z) for any beta in [0,1] must lie between S(z) and A(z) per zone."""
    for beta in np.linspace(0.0, 1.0, 11):
        R = fuse_scores(A, S, beta=float(beta))
        lo = np.minimum(A, S)
        hi = np.maximum(A, S)
        assert np.all(R >= lo - 1e-6)
        assert np.all(R <= hi + 1e-6)


def test_population_zone_stats_uses_real_frames_only():
    scores = np.array([[1.0] * 9, [3.0] * 9, [100.0] * 9], dtype=np.float32)
    labels = np.array([0, 0, 1])  # third row is a fake outlier, must be excluded
    stats = compute_population_zone_stats(scores, labels)
    np.testing.assert_allclose(stats["mean"], np.full(9, 2.0), atol=1e-5)


def test_population_zone_stats_falls_back_to_all_frames_when_no_real():
    scores = np.array([[5.0] * 9, [7.0] * 9], dtype=np.float32)
    labels = np.array([1, 1])  # no real frames at all
    stats = compute_population_zone_stats(scores, labels)
    np.testing.assert_allclose(stats["mean"], np.full(9, 6.0), atol=1e-5)


def test_normalize_zone_scores_population_mean_maps_to_half():
    stats = {"mean": np.zeros(9, dtype=np.float32), "std": np.ones(9, dtype=np.float32)}
    normed = normalize_zone_scores(np.zeros(9, dtype=np.float32), stats)
    np.testing.assert_allclose(normed, np.full(9, 0.5), atol=1e-6)


def test_normalize_zone_scores_output_in_clip_range():
    stats = {"mean": np.zeros(9, dtype=np.float32), "std": np.ones(9, dtype=np.float32)}
    normed = normalize_zone_scores(np.array([-100.0] * 9, dtype=np.float32), stats)
    assert np.all(normed >= 0.0) and np.all(normed <= 1.0)
