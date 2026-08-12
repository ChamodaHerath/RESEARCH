"""
Unit tests for inference_server.py response-shaping helpers.

These exercise the pure response-building functions with synthetic
ExplanationOutput objects. They deliberately avoid instantiating
ModelBundle (which loads dlib / MTCNN / the ViT classifier) so this module
runs fast in any environment with the project's Python dependencies
installed, without needing model weights or the landmark predictor file.

Full request/response integration tests for /infer/image and /infer/video
live in test_inference_server_integration.py and are skipped unless the
model bundle can actually be loaded (see that module's skip condition).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from semantic_patch_mapper import ZONE_NAMES, ConceptVector, ExplanationOutput

inference_server = pytest.importorskip(
    "inference_server", reason="inference_server.py requires flask/dlib/mtcnn to import."
)


def _make_bundle(beta: float = 0.4, min_severity: str = "medium") -> SimpleNamespace:
    return SimpleNamespace(explainer=SimpleNamespace(beta=beta, min_severity=min_severity))


def _make_explanation(n_flagged: int = 1, with_q: bool = True, with_method: bool = True) -> ExplanationOutput:
    R = np.linspace(0.1, 0.9, 9).astype(np.float32)
    Q = np.linspace(0.2, 0.8, 9).astype(np.float32) if with_q else None
    cvs = []
    sentences = []
    passed = []
    for i in range(n_flagged):
        zone = ZONE_NAMES[i]
        cv = ConceptVector(
            zone=zone,
            artifact_type="blending boundary artefact",
            severity="high",
            confidence=0.8,
            R_score=0.8,
            zone_prior_score=0.75 if with_q else None,
        )
        cvs.append(cv)
        sentences.append(f"High-confidence artefact detected in the {zone.replace('_', ' ')} region (confidence: 80.0%).")
        passed.append(True)
    return ExplanationOutput(
        label="Fake" if n_flagged else "Real",
        fake_probability=0.87,
        concept_vectors=cvs,
        nl_explanations=sentences,
        faithfulness_passed=passed,
        R_scores=R,
        inferred_method="Deepfakes" if with_method else None,
        method_probabilities={"Deepfakes": 0.6, "Face2Face": 0.4} if with_method else None,
        Q_scores=Q,
    )


def test_risk_level_from_probability_real_is_low():
    assert inference_server.risk_level_from_probability("Real", 0.99) == "Low"


@pytest.mark.parametrize(
    "prob,expected",
    [(0.96, "Critical"), (0.90, "High"), (0.60, "Moderate")],
)
def test_risk_level_from_probability_fake_tiers(prob, expected):
    assert inference_server.risk_level_from_probability("Fake", prob) == expected


def test_build_facial_regions_schema_and_aggregation():
    explanation = _make_explanation()
    regions = inference_server.build_facial_regions(explanation, _make_bundle())
    region_names = {r["region"] for r in regions}
    assert region_names == set(ZONE_NAMES)
    assert len(regions) == 9
    for r in regions:
        assert "anomalyScore" in r and "label" in r and "flagged" in r
        assert 0.0 <= r["anomalyScore"] <= 1.0
    # Only the concept-vector zone is flagged (authoritative pipeline set)
    flagged = [r for r in regions if r["flagged"]]
    assert len(flagged) == 1
    assert flagged[0]["region"] == ZONE_NAMES[0]


def test_build_facial_regions_real_never_flags():
    """Real verdict ⇒ Flagged No for every zone, regardless of score."""
    explanation = _make_explanation(n_flagged=0, with_q=True)
    explanation.label = "Real"
    explanation.fake_probability = 0.1
    mouth_i = ZONE_NAMES.index("mouth")
    Q = np.full(9, 0.2, dtype=np.float32)
    Q[mouth_i] = 0.95
    explanation.Q_scores = Q
    bundle = SimpleNamespace(
        explainer=SimpleNamespace(
            beta=0.4,
            min_severity="medium",
            zone_thresholds={z: 0.1 for z in ZONE_NAMES},
            min_fake_probability_for_explanations=0.70,
        )
    )
    regions = inference_server.build_facial_regions(explanation, bundle)
    assert all(r["flagged"] is False for r in regions)


def test_build_facial_regions_flagged_only_from_concept_vectors():
    """Display Flagged Yes tracks explanation survivors, not score alone."""
    explanation = _make_explanation(n_flagged=0, with_q=True)
    explanation.label = "Fake"
    explanation.fake_probability = 0.9
    mouth_i = ZONE_NAMES.index("mouth")
    Q = np.full(9, 0.2, dtype=np.float32)
    Q[mouth_i] = 0.739
    explanation.Q_scores = Q
    # No concept vectors → not flagged even if score > thr
    bundle = SimpleNamespace(
        explainer=SimpleNamespace(
            beta=0.4,
            min_severity="medium",
            zone_thresholds={z: 0.99 for z in ZONE_NAMES} | {"mouth": 0.669},
            min_fake_probability_for_explanations=0.70,
        )
    )
    regions = inference_server.build_facial_regions(explanation, bundle)
    mouth = next(r for r in regions if r["region"] == "mouth")
    assert mouth["clears_threshold"] is True
    assert mouth["flagged"] is False

    # Add mouth to concept vectors → flagged
    explanation.concept_vectors = [
        ConceptVector(
            zone="mouth",
            artifact_type="blending boundary artefact",
            severity="high",
            confidence=0.739,
            R_score=0.739,
            flag_reason="calibrated_threshold",
        )
    ]
    regions2 = inference_server.build_facial_regions(explanation, bundle)
    mouth2 = next(r for r in regions2 if r["region"] == "mouth")
    assert mouth2["flagged"] is True


def test_build_facial_regions_clears_threshold_without_confidence_gate():
    """Clears thr reflects score vs threshold on Fake verdicts (no P(fake) floor)."""
    explanation = _make_explanation(n_flagged=0, with_q=True)
    explanation.label = "Fake"
    explanation.fake_probability = 0.55
    mouth_i = ZONE_NAMES.index("mouth")
    Q = np.full(9, 0.2, dtype=np.float32)
    Q[mouth_i] = 0.739
    explanation.Q_scores = Q
    bundle = SimpleNamespace(
        explainer=SimpleNamespace(
            beta=0.4,
            min_severity="medium",
            zone_thresholds={z: 0.99 for z in ZONE_NAMES} | {"mouth": 0.669},
        )
    )
    regions = inference_server.build_facial_regions(explanation, bundle)
    mouth = next(r for r in regions if r["region"] == "mouth")
    assert mouth["clears_threshold"] is True
    explanation = _make_explanation(n_flagged=1)
    cv = explanation.concept_vectors[0]
    cv.zone_prior_score = 0.75
    bundle = SimpleNamespace(
        explainer=SimpleNamespace(
            beta=0.4,
            min_severity="medium",
            zone_thresholds={z: 0.5 for z in ZONE_NAMES},
        )
    )
    theory = inference_server.build_explanation_theory(cv, bundle, explanation=explanation)
    assert "per-zone" in theory["summary"].lower()
    assert "medium threshold" not in theory["summary"].lower()
    assert "0.5000" in theory["summary"] or "0.50" in theory["summary"]


def test_build_semantic_explanations_schema_and_sort_order():
    explanation = _make_explanation(n_flagged=3)
    bundle = _make_bundle()
    cards = inference_server.build_semantic_explanations(explanation, bundle)
    assert len(cards) == 3
    for card in cards:
        for key in ("id", "title", "description", "score", "severity", "theory"):
            assert key in card
        assert card["severity"] in {"high", "medium", "low"}
    scores = [c["score"] for c in cards]
    assert scores == sorted(scores, reverse=True)


def test_build_classification_reasoning_schema():
    explanation = _make_explanation(n_flagged=2)
    bundle = _make_bundle()
    reasoning = inference_server.build_classification_reasoning(explanation, 0.87, bundle)
    assert "summary" in reasoning and "steps" in reasoning
    assert isinstance(reasoning["steps"], list) and len(reasoning["steps"]) > 0
    for step in reasoning["steps"]:
        assert {"stage", "formula", "value", "interpretation"} <= step.keys()


def test_build_classification_reasoning_zero_flagged_zone_case():
    explanation = _make_explanation(n_flagged=0)
    bundle = _make_bundle()
    reasoning = inference_server.build_classification_reasoning(explanation, 0.3, bundle)
    assert "0 zones flagged" in reasoning["steps"][-1]["value"]


def test_build_claimed_zones_visual_schema(sample_landmarks_68):
    face = np.random.default_rng(3).integers(40, 200, (224, 224, 3), dtype=np.uint8)
    explanation = _make_explanation(n_flagged=2)
    payload = inference_server.build_claimed_zones_visual(
        face_224=face,
        landmarks_224=sample_landmarks_68,
        explanation=explanation,
    )
    assert payload["available"] is True
    assert len(payload["zones"]) == 2
    assert payload["zone_pixels"].startswith("data:image/jpeg;base64,")
    assert payload["overlay"].startswith("data:image/jpeg;base64,")
    assert 0.0 < payload["zone_coverage"] < 1.0


def test_build_claimed_zones_visual_unavailable_when_no_claims():
    face = np.zeros((224, 224, 3), dtype=np.uint8)
    explanation = _make_explanation(n_flagged=0)
    payload = inference_server.build_claimed_zones_visual(
        face_224=face,
        landmarks_224=np.zeros((68, 2), dtype=np.float32),
        explanation=explanation,
    )
    assert payload["available"] is False
    assert payload["zones"] == []


# def test_build_confidence_breakdown_schema():
#     explanation = _make_explanation(n_flagged=1)
#     bundle = _make_bundle()
#     items = inference_server.build_confidence_breakdown(explanation, bundle)
#     assert isinstance(items, list) and len(items) > 0
#     for item in items:
#         assert "metric" in item and "value" in item


def test_sha256_of_bytes_is_deterministic_and_hex():
    h1 = inference_server.sha256_of_bytes(b"hello world")
    h2 = inference_server.sha256_of_bytes(b"hello world")
    assert h1 == h2
    assert len(h1) == 64
    int(h1, 16)  # raises ValueError if not valid hex


def test_build_terminal_logs_image_mode_mentions_probability():
    logs = inference_server.build_terminal_logs(
        mode="image", filename="a.jpg", device="cpu", total_frames=None,
        sampled_frames=None, skipped_frames=0, per_frame_probs=[], label="Fake",
        fake_probability=0.91, beta=0.4, inferred_method="Deepfakes", elapsed=1.2,
    )
    joined = "\n".join(logs)
    assert "0.9100" in joined
    assert "FAKE" in joined
    assert "Deepfakes" in joined
