"""
Unit tests for ArtifactCalibrationModel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifact_analysis.artifact_ontology import ARTIFACT_TYPES
from artifact_analysis.calibration import ArtifactCalibrationModel


def _row(artifact_hint: str) -> dict:
    """Synthetic feature row biased toward a given artifact family."""
    base = {k: 0.1 for k in [
        "boundary_discontinuity", "boundary_gradient_mean",
        "delta_e_neighbor", "hue_difference",
        "lbp_entropy_difference", "laplacian_variance_difference",
        "blocking_score", "block_boundary_difference",
        "high_low_frequency_ratio", "spectral_entropy_difference",
        "luminance_neighbor_difference", "illumination_gradient",
        "landmark_symmetry_error", "shape_ratio_deviation",
        "zone_pixel_count",
    ]}
    base["zone_pixel_count"] = 800.0
    if artifact_hint == "blending_boundary":
        base["boundary_discontinuity"] = 0.9
        base["boundary_gradient_mean"] = 0.85
    elif artifact_hint == "color_tone_mismatch":
        base["delta_e_neighbor"] = 0.95
        base["hue_difference"] = 0.8
    elif artifact_hint == "texture_inconsistency":
        base["lbp_entropy_difference"] = 0.9
        base["laplacian_variance_difference"] = 0.85
    elif artifact_hint == "compression_artifact":
        base["blocking_score"] = 0.9
        base["block_boundary_difference"] = 0.8
    return base


class TestCalibration:
    def test_fit_gaussian_nb_and_predict(self) -> None:
        rows, labels, zones = [], [], []
        for art in [
            "blending_boundary",
            "color_tone_mismatch",
            "texture_inconsistency",
            "compression_artifact",
        ]:
            for _ in range(8):
                rows.append(_row(art))
                labels.append(art)
                zones.append("mouth")
        model = ArtifactCalibrationModel(method="gaussian_nb")
        model.fit(rows, labels, zones)
        assert model.fitted_
        probs = model.predict_proba(_row("blending_boundary"), "mouth")
        assert abs(sum(probs.values()) - 1.0) < 1e-5
        assert max(probs, key=probs.get) == "blending_boundary"

    def test_fit_dirichlet_counts(self) -> None:
        rows = [_row("blending_boundary") for _ in range(5)] + [
            _row("color_tone_mismatch") for _ in range(2)
        ]
        labels = ["blending_boundary"] * 5 + ["color_tone_mismatch"] * 2
        zones = ["mouth"] * 7
        model = ArtifactCalibrationModel(method="dirichlet_counts")
        model.fit(rows, labels, zones)
        probs = model.predict_proba({}, "mouth")
        assert probs["blending_boundary"] > probs["color_tone_mismatch"]
        assert abs(sum(probs.values()) - 1.0) < 1e-5

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        rows = [_row("blending_boundary") for _ in range(6)] + [
            _row("texture_inconsistency") for _ in range(6)
        ]
        labels = ["blending_boundary"] * 6 + ["texture_inconsistency"] * 6
        zones = ["left_cheek"] * 12
        model = ArtifactCalibrationModel(method="gaussian_nb")
        model.fit(rows, labels, zones)
        path = tmp_path / "cal.json"
        model.save(str(path))
        loaded = ArtifactCalibrationModel.load(str(path))
        assert loaded.fitted_
        p1 = model.predict_proba(_row("blending_boundary"), "left_cheek")
        p2 = loaded.predict_proba(_row("blending_boundary"), "left_cheek")
        # Same argmax after round-trip
        assert max(p1, key=p1.get) == max(p2, key=p2.get)

    def test_unfitted_returns_uniform(self) -> None:
        model = ArtifactCalibrationModel()
        probs = model.predict_proba({}, "mouth")
        assert set(probs.keys()) == set(ARTIFACT_TYPES)
        assert all(abs(v - 1.0 / len(ARTIFACT_TYPES)) < 1e-9 for v in probs.values())

    def test_empty_fit_raises(self) -> None:
        model = ArtifactCalibrationModel()
        with pytest.raises(ValueError):
            model.fit([], [], [])
