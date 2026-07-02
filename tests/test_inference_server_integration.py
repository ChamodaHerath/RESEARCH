"""
Integration tests for the real-time inference server (/health, /infer/image,
/infer/video).

These tests load the actual ModelBundle (dlib landmark predictor, MTCNN,
ViT classifier) and are therefore slower and heavier than the rest of the
suite. They are skipped automatically when the required model assets
(landmark predictor file, sample face image/video, network access for the
HF checkpoint) are not available, so the fast unit-test tier is unaffected.

Run explicitly with:
    pytest tests/test_inference_server_integration.py -m integration -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parent.parent
PREDICTOR_PATH = REPO_ROOT / "shape_predictor_68_face_landmarks.dat"
SAMPLE_IMAGE = REPO_ROOT / "inputs" / "001.jpg"
SAMPLE_VIDEO = REPO_ROOT / "inputs" / "001.mp4"
CONFIG_PATH = REPO_ROOT / "configs" / "fusion_config.json"

_SKIP_REASON = None
if not PREDICTOR_PATH.exists():
    _SKIP_REASON = f"Missing {PREDICTOR_PATH.name}; skipping live inference server tests."
elif not CONFIG_PATH.exists():
    _SKIP_REASON = "Missing configs/fusion_config.json; run recalibrate_v2.py first."
elif not SAMPLE_IMAGE.exists():
    _SKIP_REASON = f"Missing sample image {SAMPLE_IMAGE}; skipping."

pytestmark = [pytest.mark.integration, pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")]

REQUIRED_TOP_LEVEL_FIELDS = {
    "prediction", "confidence", "riskLevel", "processing_time", "model", "mode",
    "terminal_logs", "semantic_explanations", "facial_regions", "timeline",
    "heatmap", "attention_map", "faceCrop", "confidence_breakdown",
    "classification_reasoning", "report",
}


@pytest.fixture(scope="module")
def client():
    import inference_server

    inference_server.app.testing = True
    with inference_server.app.test_client() as c:
        yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert "beta_map" in data


def test_infer_image_schema(client):
    with open(SAMPLE_IMAGE, "rb") as fh:
        resp = client.post(
            "/infer/image",
            data={"file": (fh, "001.jpg")},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 200
    data = resp.get_json()
    assert REQUIRED_TOP_LEVEL_FIELDS <= data.keys()
    assert data["prediction"] in ("Fake", "Real")
    assert data["mode"] == "image"
    assert isinstance(data["semantic_explanations"], list)
    assert isinstance(data["facial_regions"], list)
    assert data["report"]["fileType"] == "image"


def test_infer_image_rejects_missing_file(client):
    resp = client.post("/infer/image", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_rate_limit_returns_429_after_threshold(client, monkeypatch):
    import inference_server

    monkeypatch.setattr(inference_server, "RATE_LIMIT_MAX_REQUESTS", 2)
    inference_server._request_log.clear()
    with open(SAMPLE_IMAGE, "rb") as fh:
        img_bytes = fh.read()

    def _post():
        return client.post(
            "/infer/image",
            data={"file": (__import__("io").BytesIO(img_bytes), "001.jpg")},
            content_type="multipart/form-data",
        )

    statuses = [_post().status_code for _ in range(3)]
    assert statuses[:2] == [200, 200]
    assert statuses[2] == 429


def test_api_key_required_when_configured(client, monkeypatch):
    import inference_server

    monkeypatch.setattr(inference_server, "API_KEY", "secret-key")
    inference_server._request_log.clear()
    with open(SAMPLE_IMAGE, "rb") as fh:
        resp = client.post(
            "/infer/image",
            data={"file": (fh, "001.jpg")},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 401

    inference_server._request_log.clear()
    with open(SAMPLE_IMAGE, "rb") as fh:
        resp = client.post(
            "/infer/image",
            data={"file": (fh, "001.jpg")},
            content_type="multipart/form-data",
            headers={"X-API-Key": "secret-key"},
        )
    assert resp.status_code == 200


@pytest.mark.skipif(not SAMPLE_VIDEO.exists(), reason="Missing sample video for video endpoint test.")
def test_infer_video_returns_single_aggregate_report(client):
    """
    The frontend/plan requirement: /infer/video must return ONE aggregate
    report for the whole clip, not one report per sampled frame.
    """
    with open(SAMPLE_VIDEO, "rb") as fh:
        resp = client.post(
            "/infer/video",
            data={"file": (fh, "001.mp4")},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 200
    data = resp.get_json()
    assert REQUIRED_TOP_LEVEL_FIELDS <= data.keys()
    assert data["mode"] == "video"
    assert data["report"]["fileType"] == "video"
    # A single verdict/confidence pair for the whole video, not a list.
    assert isinstance(data["prediction"], str)
    assert isinstance(data["confidence"], (int, float))
    # Per-frame detail is confined to the timeline, not duplicated top-level reports.
    assert isinstance(data["timeline"], list)
