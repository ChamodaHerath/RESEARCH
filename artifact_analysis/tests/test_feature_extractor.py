"""
Unit tests for ZoneFeatureExtractor and geometry features.

Uses synthetic 224x224 images and masks — no torch/dlib required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifact_analysis.feature_extractor import (
    ZoneFeatureExtractor,
    extract_geometry_features,
)
from tests.conftest import make_synthetic_landmarks_68

IMAGE_SIZE = 224


def _skin_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), (180, 140, 120), dtype=np.uint8)
    noise = rng.integers(-8, 9, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.int16)
    return np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _disk_mask(cx: int, cy: int, r: int) -> np.ndarray:
    yy, xx = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


@pytest.fixture
def extractor() -> ZoneFeatureExtractor:
    return ZoneFeatureExtractor(min_zone_pixels=64)


@pytest.fixture
def landmarks_68() -> np.ndarray:
    return make_synthetic_landmarks_68()


class TestInputValidation:
    def test_empty_zone_mask(self, extractor: ZoneFeatureExtractor) -> None:
        img = _skin_image()
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
        feats = extractor.extract(img, mask)
        assert feats["zone_pixel_count"] == 0.0
        assert feats["laplacian_variance"] == 0.0
        assert feats["delta_e_neighbor"] == 0.0

    def test_tiny_zone_mask(self, extractor: ZoneFeatureExtractor) -> None:
        img = _skin_image()
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
        mask[100:103, 100:103] = True
        feats = extractor.extract(img, mask)
        assert feats["zone_pixel_count"] == 9.0
        assert np.isfinite(feats["laplacian_variance"])

    def test_invalid_image_shape(self, extractor: ZoneFeatureExtractor) -> None:
        mask = _disk_mask(112, 112, 40)
        with pytest.raises(ValueError, match="HxWx3"):
            extractor.extract(np.zeros((224, 224), dtype=np.uint8), mask)

    def test_mask_shape_mismatch(self, extractor: ZoneFeatureExtractor) -> None:
        img = _skin_image()
        with pytest.raises(ValueError, match="zone_mask shape"):
            extractor.extract(img, np.zeros((100, 100), dtype=bool))

    def test_invalid_landmark_shape(self) -> None:
        with pytest.raises(ValueError, match="68, 2"):
            extract_geometry_features(np.zeros((10, 2)), "mouth")

    def test_unknown_zone_name(self, landmarks_68: np.ndarray) -> None:
        with pytest.raises(KeyError):
            extract_geometry_features(landmarks_68, "not_a_zone")


class TestDeterminismAndKeys:
    def test_deterministic_output(self, extractor: ZoneFeatureExtractor) -> None:
        img = _skin_image(1)
        mask = _disk_mask(112, 120, 35)
        a = extractor.extract(img, mask)
        b = extractor.extract(img, mask)
        assert a.keys() == b.keys()
        for k in a:
            assert a[k] == pytest.approx(b[k], rel=0, abs=1e-9)

    def test_finite_features(self, extractor: ZoneFeatureExtractor) -> None:
        img = _skin_image(2)
        mask = _disk_mask(112, 120, 35)
        feats = extractor.extract(img, mask)
        for k, v in feats.items():
            assert np.isfinite(v), f"{k} is not finite: {v}"

    def test_required_feature_keys_present(
        self, extractor: ZoneFeatureExtractor
    ) -> None:
        img = _skin_image()
        mask = _disk_mask(112, 120, 40)
        feats = extractor.extract(img, mask)
        required = [
            "lbp_entropy",
            "laplacian_variance",
            "local_contrast",
            "boundary_discontinuity",
            "delta_e_neighbor",
            "luminance_neighbor_difference",
            "high_low_frequency_ratio",
            "blocking_score",
            "gray_level_entropy",
            "canny_edge_density",
        ]
        for key in required:
            assert key in feats


class TestSyntheticArtifactSignals:
    """Synthetic images should move evidence in the expected direction."""

    def test_blurred_patch_increases_texture_smoothing_signal(
        self, extractor: ZoneFeatureExtractor
    ) -> None:
        img = _skin_image(3)
        # Add high-frequency texture everywhere
        rng = np.random.default_rng(3)
        img = np.clip(
            img.astype(np.int16) + rng.integers(-25, 26, img.shape), 0, 255
        ).astype(np.uint8)
        mask = _disk_mask(112, 120, 40)
        neigh = _disk_mask(112, 120, 70) & (~mask)

        sharp_feats = extractor.extract(img, mask, neigh)

        blurred = img.copy()
        patch = cv2.GaussianBlur(blurred, (21, 21), 0)
        blurred[mask] = patch[mask]
        blur_feats = extractor.extract(blurred, mask, neigh)

        # Blurred zone should have lower Laplacian variance than sharp zone
        assert blur_feats["laplacian_variance"] < sharp_feats["laplacian_variance"]
        # Difference vs noisy neighbor should increase
        assert (
            blur_feats["laplacian_variance_difference"]
            >= sharp_feats["laplacian_variance_difference"] - 1e-6
        )

    def test_artificial_edge_increases_blending_boundary_evidence(
        self, extractor: ZoneFeatureExtractor
    ) -> None:
        img = _skin_image(4)
        mask = _disk_mask(112, 120, 40)
        baseline = extractor.extract(img, mask)

        edged = img.copy()
        # Draw a hard ring around the zone (compositing seam proxy)
        band = mask.copy()
        dil = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        ring = dil & (~mask)
        edged[ring] = (20, 20, 20)
        edged_feats = extractor.extract(edged, mask)

        assert (
            edged_feats["boundary_gradient_mean"]
            > baseline["boundary_gradient_mean"]
        )
        assert (
            edged_feats["boundary_discontinuity"]
            >= baseline["boundary_discontinuity"] - 1e-6
        )

    def test_color_shifted_region_increases_delta_e(
        self, extractor: ZoneFeatureExtractor
    ) -> None:
        img = _skin_image(5)
        mask = _disk_mask(112, 120, 40)
        neigh = _disk_mask(112, 120, 70) & (~mask)
        baseline = extractor.extract(img, mask, neigh)

        shifted = img.copy()
        # Shift LAB-ish via strong green push in zone
        shifted[mask, 1] = np.clip(shifted[mask, 1].astype(np.int16) + 60, 0, 255).astype(
            np.uint8
        )
        shifted_feats = extractor.extract(shifted, mask, neigh)

        assert shifted_feats["delta_e_neighbor"] > baseline["delta_e_neighbor"]
        assert shifted_feats["rgb_mean_difference"] > baseline["rgb_mean_difference"]

    def test_periodic_blocks_increase_compression_evidence(
        self, extractor: ZoneFeatureExtractor
    ) -> None:
        img = _skin_image(6)
        mask = _disk_mask(112, 112, 60)
        baseline = extractor.extract(img, mask)

        blocked = img.copy().astype(np.float32)
        # Emphasize 8-pixel block borders
        for x in range(8, IMAGE_SIZE, 8):
            blocked[:, x] = np.clip(blocked[:, x] - 40, 0, 255)
            if x > 0:
                blocked[:, x - 1] = np.clip(blocked[:, x - 1] + 40, 0, 255)
        for y in range(8, IMAGE_SIZE, 8):
            blocked[y, :] = np.clip(blocked[y, :] - 40, 0, 255)
            if y > 0:
                blocked[y - 1, :] = np.clip(blocked[y - 1, :] + 40, 0, 255)
        blocked_u8 = blocked.astype(np.uint8)
        block_feats = extractor.extract(blocked_u8, mask)

        assert block_feats["block_boundary_strength"] > baseline["block_boundary_strength"]
        assert block_feats["blocking_score"] >= baseline["blocking_score"] - 1e-6

    def test_asymmetric_landmarks_increase_geometry_error(
        self, landmarks_68: np.ndarray
    ) -> None:
        base = extract_geometry_features(landmarks_68, "mouth")
        asym = landmarks_68.copy()
        # Push left eye landmarks outward
        asym[36:42, 0] -= 15.0
        asym[48, 1] += 12.0
        asym_feats = extract_geometry_features(asym, "mouth")
        assert (
            asym_feats["landmark_symmetry_error"]
            > base["landmark_symmetry_error"]
        )
        assert (
            asym_feats["eye_width_asymmetry"]
            >= base["eye_width_asymmetry"] - 1e-9
        )
        assert (
            asym_feats["mouth_corner_asymmetry"]
            > base["mouth_corner_asymmetry"]
        )


class TestUniformNoisyBlurred:
    def test_uniform_image(self, extractor: ZoneFeatureExtractor) -> None:
        img = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 128, dtype=np.uint8)
        mask = _disk_mask(112, 112, 40)
        feats = extractor.extract(img, mask)
        assert feats["laplacian_variance"] == pytest.approx(0.0, abs=1e-3)
        assert feats["gradient_magnitude_mean"] == pytest.approx(0.0, abs=1e-3)

    def test_noisy_image(self, extractor: ZoneFeatureExtractor) -> None:
        rng = np.random.default_rng(7)
        img = rng.integers(0, 256, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        mask = _disk_mask(112, 112, 40)
        feats = extractor.extract(img, mask)
        assert feats["laplacian_variance"] > 0.0
        assert feats["lbp_entropy"] > 0.0

    def test_blurred_image(self, extractor: ZoneFeatureExtractor) -> None:
        rng = np.random.default_rng(8)
        noisy = rng.integers(0, 256, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        blur = cv2.GaussianBlur(noisy, (31, 31), 0)
        mask = _disk_mask(112, 112, 40)
        noisy_f = extractor.extract(noisy, mask)
        blur_f = extractor.extract(blur, mask)
        assert blur_f["laplacian_variance"] < noisy_f["laplacian_variance"]


class TestGeometryIntegration:
    def test_extract_with_geometry(
        self, extractor: ZoneFeatureExtractor, landmarks_68: np.ndarray
    ) -> None:
        img = _skin_image(9)
        mask = _disk_mask(112, 165, 30)
        feats = extractor.extract_with_geometry(
            img, mask, landmarks_68, zone_name="mouth"
        )
        assert "landmark_symmetry_error" in feats
        assert "lbp_entropy" in feats
        assert feats["zone_pixel_count"] > 0
