"""
Unit tests for configs/fusion_config.json loading (load_fusion_config).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from semantic_patch_mapper import ZONE_NAMES, load_fusion_config


def test_load_fusion_config_missing_file_returns_empty_tuple(tmp_path):
    cfg, artifact_prior, A_stats, S_stats, beta_posterior = load_fusion_config(
        str(tmp_path / "does_not_exist.json")
    )
    assert cfg == {}
    assert artifact_prior is None
    assert A_stats is None
    assert S_stats is None
    assert beta_posterior is None


def test_load_fusion_config_parses_stats_and_beta_posterior(tmp_path):
    payload = {
        "beta": 0.6,
        "beta_posterior": {"map": 0.6, "mean": 0.61, "ci_lower": 0.5, "ci_upper": 0.72},
        "A_stats": {"mean": [0.1] * 9, "std": [0.05] * 9},
        "S_stats": {"mean": [0.2] * 9, "std": [0.07] * 9},
        "zone_names": ZONE_NAMES,
    }
    path = tmp_path / "fusion_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    cfg, artifact_prior, A_stats, S_stats, beta_posterior = load_fusion_config(str(path))

    assert cfg["beta"] == 0.6
    assert artifact_prior is None  # no artifact_posteriors key in payload
    np.testing.assert_allclose(A_stats["mean"], np.full(9, 0.1))
    np.testing.assert_allclose(S_stats["std"], np.full(9, 0.07))
    assert beta_posterior["map"] == 0.6
    # std is absent from the payload -> approximated from the 95% CI width
    assert "std" in beta_posterior
    assert beta_posterior["std"] == pytest.approx((0.72 - 0.5) / 3.92, abs=1e-4)


def test_load_fusion_config_parses_artifact_posteriors(tmp_path):
    from bayesian_calibration import DirichletArtifactPrior
    from semantic_patch_mapper import ARTIFACT_ONTOLOGY

    prior = DirichletArtifactPrior(ARTIFACT_ONTOLOGY)
    payload = {"artifact_posteriors": prior.to_dict()}
    path = tmp_path / "fusion_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    _, artifact_prior, *_ = load_fusion_config(str(path))
    assert artifact_prior is not None
    art, probs = artifact_prior.get_map_artifact("mouth", R_score=0.9)
    assert art in ARTIFACT_ONTOLOGY["mouth"]
