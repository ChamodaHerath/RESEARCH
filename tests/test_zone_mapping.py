"""
Unit tests for zone-patch matrix construction (Module 1) and zone-score
aggregation (Module 2 inputs).

Covers plan item: "Zone mapping shape and bounds: (196, 9), non-negative,
stable zone ordering."
"""

from __future__ import annotations

import numpy as np
import pytest

from semantic_patch_mapper import (
    NUM_PATCHES,
    NUM_ZONES,
    ZONE_NAMES,
    build_overlap_fraction_matrix,
    build_binary_mask_matrix,
    compute_attention_zone_scores,
    compute_shap_zone_scores,
)


def test_zone_names_stable_order_and_count():
    expected = [
        "left_eye", "right_eye", "nose_bridge", "nose_tip",
        "left_cheek", "right_cheek", "mouth", "forehead", "chin_jaw",
    ]
    assert ZONE_NAMES == expected
    assert NUM_ZONES == 9
    assert NUM_PATCHES == 196


def test_overlap_fraction_matrix_shape_and_bounds(sample_landmarks_68):
    M = build_overlap_fraction_matrix(sample_landmarks_68)
    assert M.shape == (NUM_PATCHES, NUM_ZONES)
    assert M.dtype == np.float32
    assert np.all(M >= 0.0)
    assert np.all(M <= 1.0 + 1e-6)


def test_overlap_fraction_matrix_nonzero_coverage_per_zone(sample_landmarks_68):
    """Every zone should claim at least one patch for a normal face layout."""
    M = build_overlap_fraction_matrix(sample_landmarks_68)
    zone_area = M.sum(axis=0)
    for i, zone in enumerate(ZONE_NAMES):
        assert zone_area[i] > 0.0, f"Zone '{zone}' has zero patch coverage."


def test_binary_mask_matrix_shape_and_binary_values(sample_landmarks_68):
    M = build_binary_mask_matrix(sample_landmarks_68)
    assert M.shape == (NUM_PATCHES, NUM_ZONES)
    assert set(np.unique(M).tolist()).issubset({0.0, 1.0})


def test_overlap_fraction_matrix_rejects_bad_landmark_shape():
    with pytest.raises(ValueError):
        build_overlap_fraction_matrix(np.zeros((67, 2), dtype=np.float32))


def test_overlap_fraction_matrix_deterministic(sample_landmarks_68):
    M1 = build_overlap_fraction_matrix(sample_landmarks_68)
    M2 = build_overlap_fraction_matrix(sample_landmarks_68)
    np.testing.assert_array_equal(M1, M2)


def test_compute_attention_zone_scores_shape_and_bounds(sample_landmarks_68, sample_alpha):
    M = build_overlap_fraction_matrix(sample_landmarks_68)
    A = compute_attention_zone_scores(sample_alpha, M)
    assert A.shape == (NUM_ZONES,)
    # alpha sums to 1 and is non-negative -> area-normalised mean per zone
    # must also be within [0, 1].
    assert np.all(A >= 0.0)
    assert np.all(A <= 1.0 + 1e-6)


def test_compute_shap_zone_scores_shape_and_nonnegative(sample_landmarks_68, sample_phi):
    M = build_overlap_fraction_matrix(sample_landmarks_68)
    S = compute_shap_zone_scores(sample_phi, M)
    assert S.shape == (NUM_ZONES,)
    assert np.all(S >= 0.0)


def test_zone_scores_raise_on_shape_mismatch(sample_landmarks_68, sample_alpha):
    M = build_overlap_fraction_matrix(sample_landmarks_68)
    with pytest.raises(ValueError):
        compute_attention_zone_scores(sample_alpha[:100], M)
