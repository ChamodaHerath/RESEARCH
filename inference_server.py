"""
inference_server.py — Real-time HTTP inference server for the SAEDF pipeline.

This is the ONLY backend the web demo (webapp/) talks to for detection results.
Every field returned here is computed live from the actual uploaded file:

  - fake_probability   : real softmax output of the configured pretrained ViT
                          deepfake classifier
                          (attention_extractor.DEFAULT_HF_DEEPFAKE_MODEL).
  - alpha / phi         : real CLS-attention / grad-rollout signals from the
                          SAME forward pass on the uploaded image.
  - R(z), Q(z)          : real Bayesian dual-score fusion + zone-prior posterior,
                          using the calibration in configs/fusion_config.json
                          (beta/A_stats/S_stats/zone_thresholds are re-fit per
                          detector by recalibrate_v2.py -- see that file's
                          beta_posterior field for the current MAP/CI).
  - heatmap overlay     : real grad-weighted attention rollout rendered onto the
                          actual uploaded face crop.
  - hash / case id      : SHA-256 of the actual uploaded bytes.

No field is randomly generated or hardcoded. If a face cannot be detected, the
server returns a 422 error instead of fabricating a result.

Run:
    venv\\Scripts\\python.exe inference_server.py --port 8000
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import dlib
import numpy as np
from flask import Flask, jsonify, request
from mtcnn import MTCNN
from werkzeug.exceptions import RequestEntityTooLarge

from attention_extractor import AttentionExtractor, DEFAULT_HF_DEEPFAKE_MODEL
from baseline_heatmap_generator import extract_full_pipeline_inputs
from pipeline import classifier_fake_probabilities
from preprocess_landmarks import (
    detect_face_with_mtcnn,
    extract_68_landmarks_dlib,
    scale_landmarks_to_target,
)
from semantic_patch_mapper import (
    ExplanationOutput,
    SemanticExplanationGenerator,
    SEVERITY_THRESHOLDS,
    ZONE_NAMES,
    filter_explanations_by_causal_correctness,
)
from ffpp_utils import FFPP_METHODS, find_ffpp_mask_video
from zone_localization import crop_and_resize_mask, is_likely_mask_frame

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("inference_server")

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
CONFIG_PATH = "configs/fusion_config.json"
IMAGE_SIZE = 224
REPO_ROOT = Path(__file__).resolve().parent
FFPP_MASKS_ROOT = Path(
    os.environ.get(
        "SAEDF_FFPP_MASKS_ROOT",
        str(REPO_ROOT / "datasets" / "FaceForensics++" / "masks"),
    )
)
FFPP_ROOT = Path(
    os.environ.get(
        "SAEDF_FFPP_ROOT",
        str(REPO_ROOT / "datasets" / "FaceForensics++" / "FaceForensics++_C23"),
    )
)
# Minimum P(fake) drop when a claimed zone is occluded for the claim to survive
# Causal correctness gate — research-grade default ON.
# Only claims that move P(fake) by ≥ CAUSAL_MIN_DROP enter the final report.
# Override with SAEDF_CAUSAL_MIN_DROP / SAEDF_CAUSAL_CORRECTNESS=0.
CAUSAL_MIN_DROP = float(os.environ.get("SAEDF_CAUSAL_MIN_DROP", "0.03"))
CAUSAL_CORRECTNESS_ENABLED = os.environ.get("SAEDF_CAUSAL_CORRECTNESS", "1") != "0"

# 9-zone -> 5 display-region aggregation used by the frontend's facial-region UI.
DISPLAY_REGION_MAP: Dict[str, List[str]] = {
    "forehead": ["forehead"],
    "eyes":     ["left_eye", "right_eye"],
    "nose":     ["nose_bridge", "nose_tip"],
    "mouth":    ["mouth"],
    "jaw":      ["chin_jaw"],
}

app = Flask(__name__)
_inference_lock = threading.Lock()   # single-flight: one CUDA forward pass at a time

# ---------------------------------------------------------------------------
# Production hardening: upload size cap, simple rate limiting, optional
# API-key auth. All are configurable via env vars so the demo server keeps
# working out of the box (no auth, generous limits) unless explicitly
# tightened for a real deployment.
# ---------------------------------------------------------------------------
MAX_UPLOAD_MB = float(os.getenv("SAEDF_MAX_UPLOAD_MB", "100"))
app.config["MAX_CONTENT_LENGTH"] = int(MAX_UPLOAD_MB * 1024 * 1024)

RATE_LIMIT_MAX_REQUESTS = int(os.getenv("SAEDF_RATE_LIMIT_MAX_REQUESTS", "30"))
RATE_LIMIT_WINDOW_SECONDS = float(os.getenv("SAEDF_RATE_LIMIT_WINDOW_SECONDS", "60"))
API_KEY = os.getenv("SAEDF_API_KEY")   # if unset, auth is disabled (local/dev default)

_rate_limit_lock = threading.Lock()
_request_log: Dict[str, Deque[float]] = defaultdict(deque)
RATE_LIMITED_PATHS = ("/infer/image", "/infer/video")


def _client_ip() -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    return forwarded.split(",")[0].strip() if forwarded else (request.remote_addr or "unknown")


@app.before_request
def _enforce_upload_hardening():
    if request.path not in RATE_LIMITED_PATHS:
        return None

    if API_KEY:
        supplied = request.headers.get("X-API-Key")
        if supplied != API_KEY:
            return jsonify({"error": "Missing or invalid API key.", "code": "UNAUTHORIZED"}), 401

    ip = _client_ip()
    now = time.time()
    with _rate_limit_lock:
        window = _request_log[ip]
        while window and now - window[0] > RATE_LIMIT_WINDOW_SECONDS:
            window.popleft()
        if len(window) >= RATE_LIMIT_MAX_REQUESTS:
            retry_after = max(0.0, RATE_LIMIT_WINDOW_SECONDS - (now - window[0]))
            resp = jsonify({
                "error": f"Rate limit exceeded ({RATE_LIMIT_MAX_REQUESTS} requests / "
                         f"{RATE_LIMIT_WINDOW_SECONDS:.0f}s). Try again later.",
                "code": "RATE_LIMITED",
            })
            resp.status_code = 429
            resp.headers["Retry-After"] = str(int(retry_after) + 1)
            return resp
        window.append(now)
    return None


@app.errorhandler(RequestEntityTooLarge)
def _handle_upload_too_large(_exc):
    return jsonify({
        "error": f"Uploaded file exceeds the {MAX_UPLOAD_MB:.0f} MB limit.",
        "code": "PAYLOAD_TOO_LARGE",
    }), 413


class ModelBundle:
    """Loads every real component exactly once at process start."""

    def __init__(self) -> None:
        log.info("Loading MTCNN face detector ...")
        self.detector = MTCNN()

        log.info("Loading dlib 68-point landmark predictor ...")
        self.predictor = dlib.shape_predictor(PREDICTOR_PATH)

        log.info("Loading ViT deepfake classifier: %s ...", DEFAULT_HF_DEEPFAKE_MODEL)
        self.extractor = AttentionExtractor(
            model_name=DEFAULT_HF_DEEPFAKE_MODEL,
            pretrained=True,
        )

        log.info("Loading calibrated semantic explanation generator ...")
        self.explainer = SemanticExplanationGenerator(config_path=CONFIG_PATH)

        log.info("Model bundle ready on device: %s", self.extractor.device)


BUNDLE: Optional[ModelBundle] = None


def get_bundle() -> ModelBundle:
    global BUNDLE
    if BUNDLE is None:
        BUNDLE = ModelBundle()
    return BUNDLE


# ---------------------------------------------------------------------------
# Core single-frame pipeline
# ---------------------------------------------------------------------------

class NoFaceDetected(Exception):
    pass


def run_single_frame(image_bgr_native: np.ndarray, bundle: ModelBundle) -> Dict:
    """
    Run the full real SAEDF pipeline on one native-resolution BGR frame.

    Mirrors preprocess_landmarks.process_single_frame exactly (face detect ->
    landmarks on native crop -> scale to 224 -> resize crop to 224) followed by
    the real dual-head inference (classification + explanation), then a causal
    correctness gate that drops NL claims whose zones do not move P(fake).
    """
    face = detect_face_with_mtcnn(detector=bundle.detector, image_bgr=image_bgr_native)
    if face is None:
        raise NoFaceDetected("No face detected in frame.")

    landmarks = extract_68_landmarks_dlib(predictor=bundle.predictor, face_crop_bgr=face.image)
    if landmarks is None or len(landmarks) != 68:
        raise NoFaceDetected("Face detected but landmark extraction failed.")

    face_224 = cv2.resize(face.image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    landmarks_224 = scale_landmarks_to_target(landmarks, face.width, face.height, IMAGE_SIZE)

    # Real forward pass #1: fake_probability + alpha (attention) + phi (grad-rollout, zone-scoring variant)
    inputs = extract_full_pipeline_inputs(face_224, bundle.extractor)

    explanation: ExplanationOutput = bundle.explainer.predict(
        attention_vec=inputs["alpha"],
        landmarks_68=landmarks_224,
        phi_vec=inputs["phi"],
        fake_probability=inputs["fake_probability"],
    )

    # Causal correctness gate — only claims that move P(fake) enter the report.
    # Prefer the calibrated causal_min_drop from fusion_config when present.
    if CAUSAL_CORRECTNESS_ENABLED and explanation.concept_vectors:
        min_drop = float(
            getattr(bundle.explainer, "causal_min_drop", None) or CAUSAL_MIN_DROP
        )
        explanation = filter_explanations_by_causal_correctness(
            explanation=explanation,
            image_bgr=face_224,
            landmarks_68=landmarks_224,
            fake_prob_fn=lambda imgs: classifier_fake_probabilities(
                bundle.extractor, imgs
            ),
            min_drop=min_drop,
        )

    return {
        "face_224": face_224,
        "fake_probability": inputs["fake_probability"],
        "explanation": explanation,
        "heatmap_224": inputs["heatmap_224"],
        "crop_box": [int(face.x1), int(face.y1), int(face.x2), int(face.y2)],
    }


def render_heatmap_overlay_b64(face_224: np.ndarray, heatmap_224: np.ndarray) -> str:
    """
    Render the JPEG overlay from an already-computed grad-rollout heatmap.

    `heatmap_224` comes from the single forward+backward pass already done in
    run_single_frame() (extract_full_pipeline_inputs) -- this function does
    NOT run the model again. Previously this re-ran the full grad-weighted
    attention rollout a second time per request purely to render the overlay,
    doubling GPU/CPU cost for every /infer/image and /infer/video call.
    """
    overlay_bgr = AttentionExtractor.overlay_heatmap_on_image(
        image_bgr=face_224, heatmap_224=heatmap_224, alpha=0.55
    )
    ok, buf = cv2.imencode(".jpg", overlay_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("Failed to encode heatmap overlay.")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def face_crop_to_b64(face_224: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", face_224, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("Failed to encode face crop.")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _resolve_ffpp_mask_path(filename: Optional[str]) -> Optional[Path]:
    """
    Map an uploaded FF++-style filename to a local manipulation-mask video.

    Accepts ``Deepfakes/000_003.mp4``, ``000_003.mp4``, or a full path whose
    basename matches a mask clip. Returns None for non-FF++ uploads.
    """
    if not filename:
        return None
    name = Path(filename.replace("\\", "/")).name
    if not name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        return None

    # Prefer Method/name when the client preserves a relative path
    rel = filename.replace("\\", "/")
    parts = Path(rel).parts
    candidates: List[str] = []
    if len(parts) >= 2 and parts[-2] in FFPP_METHODS:
        candidates.append(f"{parts[-2]}/{name}")
    for method in FFPP_METHODS:
        candidates.append(f"{method}/{name}")

    seen: set = set()
    for rel_path in candidates:
        if rel_path in seen:
            continue
        seen.add(rel_path)
        found = find_ffpp_mask_video(rel_path, FFPP_MASKS_ROOT, FFPP_ROOT)
        if found is not None:
            return found
    return None


def render_mask_gt_overlay_b64(
    face_224: np.ndarray,
    mask_224: np.ndarray,
    alpha: float = 0.45,
) -> str:
    """Overlay FF++ manipulation mask (green) on the aligned face crop."""
    heat = np.clip(mask_224.astype(np.float32), 0.0, 1.0)
    if heat.ndim == 2:
        color = np.zeros_like(face_224, dtype=np.float32)
        color[:, :, 1] = heat * 255.0  # green channel = manipulated pixels
    else:
        color = heat
    blend = (
        face_224.astype(np.float32) * (1.0 - alpha * heat[..., None])
        + color * (alpha * heat[..., None])
    )
    out = np.clip(blend, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("Failed to encode mask GT overlay.")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def try_build_mask_ground_truth_visual(
    filename: Optional[str],
    face_224: np.ndarray,
    crop_box: Optional[Sequence[int]],
    frame_index: int = 0,
) -> Optional[Dict]:
    """
    If the upload matches an FF++ fake with a local mask video, return a
    side-by-side GT overlay payload. Otherwise None (arbitrary uploads).
    """
    if face_224 is None or crop_box is None:
        return None
    mask_path = _resolve_ffpp_mask_path(filename)
    if mask_path is None:
        return None
    try:
        cap = cv2.VideoCapture(str(mask_path))
        if not cap.isOpened():
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idx = int(max(0, min(frame_index, max(0, total - 1))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, mask_frame = cap.read()
        cap.release()
        if not ok or mask_frame is None:
            return None
        if not is_likely_mask_frame(mask_frame):
            log.warning("Rejected non-binary mask candidate: %s", mask_path)
            return None
        mask_224 = crop_and_resize_mask(mask_frame, crop_box, target_size=IMAGE_SIZE)
        coverage = float(mask_224.mean())
        return {
            "available": True,
            "overlay": render_mask_gt_overlay_b64(face_224, mask_224),
            "mask_coverage": round(coverage, 4),
            "frame_index": idx,
            "mask_source": mask_path.name,
            "note": (
                "Green overlay = FaceForensics++ manipulation-mask ground truth "
                "for this clip (true edited pixels), aligned to the same face crop. "
                "Compare with the model attention heatmap — not available for "
                "arbitrary non-FF++ uploads."
            ),
        }
    except Exception as exc:
        log.warning("Mask GT overlay failed for %s: %s", filename, exc)
        return None


# ---------------------------------------------------------------------------
# Response assembly (real values only)
# ---------------------------------------------------------------------------

SEVERITY_LEVELS = {"high", "medium", "low"}


def _format_zone(name: str) -> str:
    return name.replace("_", " ")


def build_explanation_theory(
    cv_,
    bundle: ModelBundle,
    explanation: Optional[ExplanationOutput] = None,
) -> Dict:
    """
    Per-explanation theoretical basis — every number comes from the live
    ConceptVector produced by semantic_patch_mapper for this request.
    """
    beta = bundle.explainer.beta
    min_severity = bundle.explainer.min_severity
    zone_thresholds = getattr(bundle.explainer, "zone_thresholds", None) or {}
    # Per-zone thresholds (calibrated on the Q(z) distribution of real
    # calibration frames -- see recalibrate_v2.py) take precedence over the
    # generic severity gate when present, since that is what actually
    # determines flagging at inference time.
    flag_threshold = float(zone_thresholds.get(cv_.zone, SEVERITY_THRESHOLDS.get(min_severity, 0.50)))
    threshold_source = "calibrated per-zone threshold" if cv_.zone in zone_thresholds else f"'{min_severity}' severity gate"
    flag_score = (
        float(cv_.zone_prior_score)
        if cv_.zone_prior_score is not None
        else float(cv_.R_score)
    )
    score_label = "Q(z)" if cv_.zone_prior_score is not None else "R(z)"

    steps: List[Dict[str, str]] = [
        {
            "label": "Zone flagging criterion",
            "detail": (
                f"The { _format_zone(cv_.zone) } zone was flagged because "
                f"{score_label} = {flag_score:.4f} > {flag_threshold:.4f} "
                f"({threshold_source})."
            ),
        },
        {
            "label": "Dual-score fusion R(z)",
            "detail": (
                f"R({ _format_zone(cv_.zone) }) = {cv_.R_score:.4f}, computed as "
                f"β·A_norm(z) + (1−β)·S_norm(z) with calibrated β = {beta:.4f}. "
                f"A_norm aggregates ViT CLS-to-patch attention; S_norm aggregates "
                f"grad-weighted attention rollout (φ) over the same semantic zone."
            ),
        },
    ]

    if cv_.zone_prior_score is not None:
        steps.append({
            "label": "Bayesian zone posterior Q(z)",
            "detail": (
                f"Q({ _format_zone(cv_.zone) }) = {cv_.zone_prior_score:.4f}. "
                f"This updates the zone anomaly belief using manipulation-type "
                f"literature priors conditioned on the inferred deepfake method."
            ),
        })

    if cv_.artifact_probs is not None:
        artifact_list = list(cv_.artifact_probs)
        top_idx = int(np.argmax(artifact_list))
        from semantic_patch_mapper import ARTIFACT_ONTOLOGY
        artifacts = ARTIFACT_ONTOLOGY.get(cv_.zone, [])
        top_name = artifacts[top_idx] if top_idx < len(artifacts) else cv_.artifact_type
        top_p = float(artifact_list[top_idx])
        dist_str = ", ".join(
            f"{art}={float(p):.3f}"
            for art, p in zip(artifacts, artifact_list)
        )
        steps.append({
            "label": "Dirichlet artifact attribution",
            "detail": (
                f"Artifact '{top_name}' selected as MAP type (posterior = {top_p:.3f}). "
                f"Full Dirichlet posterior over zone ontology: {dist_str}."
            ),
        })
    else:
        steps.append({
            "label": "Artifact ontology mapping",
            "detail": (
                f"Artifact '{cv_.artifact_type}' assigned from the fixed semantic "
                f"ontology for the { _format_zone(cv_.zone) } zone (deterministic rank-0 "
                f"fallback when Dirichlet calibration is unavailable)."
            ),
        })

    sev_detail = (
        f"Severity '{cv_.severity}' follows calibrated thresholds on the effective "
        f"zone score: high > {SEVERITY_THRESHOLDS['high']:.2f}, "
        f"medium > {SEVERITY_THRESHOLDS['medium']:.2f}, "
        f"low > {SEVERITY_THRESHOLDS['low']:.2f}."
    )
    if cv_.R_ci_lower is not None and cv_.R_ci_upper is not None:
        sev_detail += (
            f" 95% credible interval on R(z): "
            f"[{cv_.R_ci_lower:.4f}, {cv_.R_ci_upper:.4f}] from β posterior uncertainty."
        )
    steps.append({"label": "Severity assignment", "detail": sev_detail})

    steps.append({
        "label": "Natural-language generation",
        "detail": (
            "The displayed sentence is produced by a deterministic template grammar "
            "mapping ⟨zone, artifact_type, severity, confidence⟩ from the concept "
            "vector, then verified by a faithfulness checker that every claim is "
            "traceable to those fields."
        ),
    })

    if explanation is not None and explanation.causal_correctness_passed is not None:
        drop = (explanation.causal_drop_by_zone or {}).get(cv_.zone)
        drop_str = f"{drop:.4f}" if drop is not None else "n/a"
        min_drop = explanation.causal_min_drop
        steps.append({
            "label": "Causal correctness gate",
            "detail": (
                f"Zone '{_format_zone(cv_.zone)}' was occluded and the classifier "
                f"re-scored. P(fake) drop = {drop_str} "
                f"(threshold ≥ {min_drop}). "
                "Only claims that causally move the fake score are retained in "
                "the final report."
            ),
        })

    return {
        "summary": (
            f"Zone '{ _format_zone(cv_.zone) }' was flagged because its "
            f"{score_label} score ({flag_score:.4f}) exceeded the per-zone "
            f"calibrated threshold ({flag_threshold:.4f}; {threshold_source}). "
            f"The '{cv_.artifact_type}' label comes from the Bayesian / ontology "
            f"artifact mapping above."
        ),
        "steps": steps,
        "scores": {
            "R": round(float(cv_.R_score), 4),
            **({"Q": round(float(cv_.zone_prior_score), 4)} if cv_.zone_prior_score is not None else {}),
        },
    }


def build_semantic_explanations(explanation: ExplanationOutput, bundle: ModelBundle) -> List[Dict]:
    out = []
    for cv_, sentence in zip(explanation.concept_vectors, explanation.nl_explanations):
        out.append({
            "id": cv_.zone,
            "title": f"{cv_.artifact_type.capitalize()} — {cv_.zone.replace('_', ' ')}",
            "description": sentence,
            "score": round(float(cv_.confidence), 4),
            "severity": cv_.severity if cv_.severity in SEVERITY_LEVELS else "low",
            "theory": build_explanation_theory(cv_, bundle, explanation=explanation),
            "causal_drop": (
                round(float((explanation.causal_drop_by_zone or {}).get(cv_.zone, 0.0)), 4)
                if explanation.causal_drop_by_zone is not None else None
            ),
        })
    # Real ordering: highest confidence first (already sorted upstream by R_score,
    # but re-sort defensively on the actual displayed confidence field).
    out.sort(key=lambda d: d["score"], reverse=True)
    return out


def build_classification_reasoning(
    explanation: ExplanationOutput,
    fake_probability: float,
    bundle: ModelBundle,
) -> Dict:
    """
    Theoretical basis for the Real/Fake verdict — derived only from quantities
    computed during this inference request.
    """
    beta = bundle.explainer.beta
    label = explanation.label
    threshold = 0.5
    displayed_conf = (
        fake_probability if label == "Fake" else 1.0 - fake_probability
    )

    steps: List[Dict[str, str]] = [
        {
            "stage": "ViT classifier (softmax)",
            "formula": "P(fake | x) = softmax(logits)[fake]",
            "value": f"{fake_probability:.4f}",
            "interpretation": (
                f"The pretrained ViT deepfake head outputs P(fake|x) = {fake_probability:.4f} "
                f"from a single forward pass on the aligned 224×224 face crop."
            ),
        },
        {
            "stage": "Binary decision rule",
            "formula": "label = Fake if P(fake|x) ≥ 0.5 else Real",
            "value": f"{label} (threshold = {threshold:.2f})",
            "interpretation": (
                f"P(fake|x) = {fake_probability:.4f} is "
                f"{'≥' if fake_probability >= threshold else '<'} {threshold:.2f}, "
                f"so the classifier verdict is {label.upper()}. "
                f"Reported confidence = {displayed_conf * 100:.2f}% "
                f"({'P(fake|x)' if label == 'Fake' else '1 − P(fake|x)'})."
            ),
        },
    ]

    r_top_idx = int(np.argmax(explanation.R_scores))
    r_top_zone = ZONE_NAMES[r_top_idx]
    r_top_val = float(explanation.R_scores[r_top_idx])
    steps.append({
        "stage": "Semantic zone fusion (explainability layer)",
        "formula": "R(z) = β·A_norm(z) + (1−β)·S_norm(z)",
        "value": f"β = {beta:.4f}, max R(z) = {r_top_val:.4f} ({_format_zone(r_top_zone)})",
        "interpretation": (
            f"Attention (A) and gradient-rollout (S) patch scores are mapped to 9 "
            f"landmark-defined zones, population-normalised, then fused with "
            f"calibrated β = {beta:.4f} (MAP from configs/fusion_config.json). "
            f"Highest fused anomaly this frame: {_format_zone(r_top_zone)} "
            f"with R(z) = {r_top_val:.4f}."
        ),
    })

    if explanation.Q_scores is not None:
        q_top_idx = int(np.argmax(explanation.Q_scores))
        q_top_zone = ZONE_NAMES[q_top_idx]
        q_top_val = float(explanation.Q_scores[q_top_idx])
        steps.append({
            "stage": "Bayesian zone-prior update",
            "formula": "Q(z) = manipulation-type-conditioned posterior",
            "value": f"max Q(z) = {q_top_val:.4f} ({_format_zone(q_top_zone)})",
            "interpretation": (
                f"Zone beliefs are updated with literature priors for the inferred "
                f"manipulation family. Highest posterior zone anomaly: "
                f"{_format_zone(q_top_zone)} with Q(z) = {q_top_val:.4f}."
            ),
        })

    if explanation.inferred_method and explanation.method_probabilities:
        top_method = explanation.inferred_method
        top_p = explanation.method_probabilities.get(top_method, 0.0)
        steps.append({
            "stage": "Manipulation method inference",
            "formula": "argmax cosine(R(z), literature zone signatures)",
            "value": f"{top_method} ({top_p * 100:.1f}%)",
            "interpretation": (
                f"The R(z) pattern most closely matches the '{top_method}' deepfake "
                f"signature (posterior confidence {top_p * 100:.1f}%). This conditions "
                f"the Bayesian zone priors used for Q(z) and semantic explanations."
            ),
        })

    flagged = len(explanation.concept_vectors)
    min_severity = bundle.explainer.min_severity
    flat_threshold = SEVERITY_THRESHOLDS.get(min_severity, 0.50)
    zone_thresholds = getattr(bundle.explainer, "zone_thresholds", None) or {}

    # Mirrors the ACTUAL decision rule in semantic_patch_mapper.get_flagged_zones:
    # each zone is compared to its own calibrated Q(z)/R(z) threshold from
    # configs/fusion_config.json when one exists, and only falls back to the
    # flat severity-tier value (0.50 for "medium") for zones without a
    # calibrated threshold. Reporting a single flat number here regardless of
    # zone would misrepresent what the gate actually checked.
    score_arr = explanation.Q_scores if explanation.Q_scores is not None else explanation.R_scores
    score_label = "Q(z)" if explanation.Q_scores is not None else "R(z)"
    per_zone = []
    for i, zone in enumerate(ZONE_NAMES):
        thr = float(zone_thresholds.get(zone, flat_threshold))
        per_zone.append((zone, float(score_arr[i]), thr, float(score_arr[i]) - thr))
    closest_zone, closest_score, closest_thr, closest_margin = max(per_zone, key=lambda t: t[3])
    calibrated_note = (
        "per-zone calibrated thresholds (configs/fusion_config.json)"
        if zone_thresholds else f"the flat '{min_severity}' threshold ({flat_threshold:.2f})"
    )

    if flagged:
        steps.append({
            "stage": "Explanation generation gate",
            "formula": f"flag zone if {score_label} > that zone's own calibrated threshold",
            "value": f"{flagged} zone(s) flagged",
            "interpretation": (
                f"{flagged} facial zone(s) exceeded {calibrated_note} and received "
                f"concept vectors + NL sentences. Thresholds differ per zone — e.g. "
                f"mouth requires a higher {score_label} than nose_bridge — because each "
                f"was calibrated separately against real held-out frames."
            ),
        })
    else:
        steps.append({
            "stage": "Explanation generation gate",
            "formula": f"flag zone if {score_label} > that zone's own calibrated threshold",
            "value": "0 zones flagged",
            "interpretation": (
                f"No zone's {score_label} exceeded {calibrated_note}. Closest: "
                f"{_format_zone(closest_zone)} {score_label} = {closest_score:.4f}, "
                f"required > {closest_thr:.4f} "
                f"({closest_margin:+.4f} short) — so no semantic explanation cards are "
                f"emitted and the verdict rests on the ViT classifier probability alone. "
                f"Note: thresholds are calibrated per zone and are typically well above "
                f"the flat 0.50 'medium' severity tier (e.g. mouth > "
                f"{float(zone_thresholds.get('mouth', flat_threshold)):.2f}), so a "
                f"zone score above 0.50 does not by itself guarantee a flag."
            ),
        })

    summary = (
        f"Classified as {label.upper()} because P(fake|x) = {fake_probability:.4f} "
        f"{'≥' if fake_probability >= threshold else '<'} 0.5. "
    )
    if flagged:
        summary += (
            f"{flagged} zone(s) exceeded their own calibrated {score_label} threshold, "
            f"providing explainable forensic evidence below."
        )
    elif label == "Fake":
        summary += (
            f"The ViT head flags manipulation, but no individual zone crossed its "
            f"calibrated {score_label} threshold (closest: {_format_zone(closest_zone)} "
            f"at {closest_score:.4f} vs. {closest_thr:.4f} required) — a borderline / "
            f"global-artifact case."
        )
    else:
        summary += (
            "Both the classifier probability and zone-level fusion scores are "
            "consistent with an authentic face under the calibrated pipeline."
        )

    return {"summary": summary, "steps": steps}


def build_facial_regions(explanation: ExplanationOutput, bundle: ModelBundle) -> List[Dict]:
    """
    Always return all 9 SAEDF zones with anomaly scores.

    High score ⇒ region is more likely corrupted / manipulated.
    ``flagged`` marks zones that crossed the calibrated explanation threshold
    (and survived the causal gate when enabled).
    """
    zone_thresholds = getattr(bundle.explainer, "zone_thresholds", None) or {}
    min_severity = getattr(bundle.explainer, "min_severity", "medium")
    flagged_by_cv = {cv.zone: cv for cv in explanation.concept_vectors}
    # Research-grade: Flagged Yes only for zones that entered the explanation
    # set (Fake + confident P(fake) + score > thr, then causal survivors).
    # Never invent flags from score alone on Real / weak-Fake frames.
    min_p = float(
        getattr(bundle.explainer, "min_fake_probability_for_explanations", 0.70)
    )
    allow_flags = (
        explanation.label == "Fake"
        and float(explanation.fake_probability) > min_p
    )

    regions: List[Dict] = []
    for i, zone in enumerate(ZONE_NAMES):
        R = float(explanation.R_scores[i])
        Q = (
            float(explanation.Q_scores[i])
            if explanation.Q_scores is not None
            else None
        )
        # Display / corruption score: Q(z) when available (what drives flagging), else R(z)
        score = Q if Q is not None else R
        threshold = float(
            zone_thresholds.get(zone, SEVERITY_THRESHOLDS.get(min_severity, 0.50))
        )
        clears_thr = allow_flags and round(score, 4) > round(threshold, 4)
        cv = flagged_by_cv.get(zone)
        # Authoritative flag = survived the explanation pipeline (incl. causal).
        is_flagged = cv is not None
        regions.append({
            "region": zone,
            "label": zone.replace("_", " ").title(),
            "anomalyScore": round(score, 4),
            "flagged": is_flagged,
            "flag_reason": (cv.flag_reason if cv is not None else None),
            "R": round(R, 4),
            "Q": round(Q, 4) if Q is not None else None,
            "threshold": round(threshold, 4),
            "clears_threshold": clears_thr,
        })

    # Highest corruption score first so the UI reads as a ranked region list
    regions.sort(key=lambda d: d["anomalyScore"], reverse=True)
    return regions


def build_confidence_breakdown(explanation: ExplanationOutput, bundle: ModelBundle) -> List[Dict]:
    """
    Every entry here is a real, directly-computed quantity from the actual
    fusion + Bayesian zone-prior pipeline — nothing here is sampled/random.
    """
    beta = bundle.explainer.beta
    r_mean = float(np.mean(explanation.R_scores)) * 100.0
    coverage = (len(explanation.concept_vectors) / len(ZONE_NAMES)) * 100.0

    items = [
        {"metric": "Mean Fused Zone Anomaly  R̄(z)", "value": round(r_mean)},
        {"metric": "Calibrated Fusion Weight β (attention share)", "value": round(beta * 100)},
        {"metric": "Flagged Zone Coverage", "value": round(coverage)},
    ]
    if explanation.method_probabilities:
        top_method = max(explanation.method_probabilities, key=explanation.method_probabilities.get)
        top_p = explanation.method_probabilities[top_method]
        items.append({
            "metric": f"Inferred Manipulation Method Confidence ({top_method})",
            "value": round(top_p * 100),
        })
    faithful_rate = (
        100.0 * sum(explanation.faithfulness_passed) / len(explanation.faithfulness_passed)
        if explanation.faithfulness_passed else 100.0
    )
    items.append({"metric": "Explanation Faithfulness Pass Rate", "value": round(faithful_rate)})
    if explanation.causal_correctness_passed is not None:
        # Surviving claims already passed the gate; report retention vs all tested drops.
        n_tested = len(explanation.causal_drop_by_zone or {})
        n_kept = len(explanation.concept_vectors)
        retention = (100.0 * n_kept / n_tested) if n_tested else 100.0
        items.append({
            "metric": "Causal Correctness Retention",
            "value": round(retention),
        })
    return items


def build_explanation_validation(explanation: ExplanationOutput) -> Dict:
    """
    Dual-layer explanation audit for the forensic report.

    Live checks (this request):
      - textual faithfulness (sentence ↔ ConceptVector)
      - causal occlusion drop (zone must move P(fake))

    Offline research validation (FF++ mask GT, thesis numbers):
      - loaded from webapp/src/data/zone-localization.json when present
      - clearly scoped as dataset evaluation, not per-upload pixel GT
    """
    n_claims = len(explanation.nl_explanations)
    n_faithful = (
        int(sum(1 for p in explanation.faithfulness_passed if p))
        if explanation.faithfulness_passed else n_claims
    )
    faithful_rate = (n_faithful / n_claims) if n_claims else 1.0

    causal_enabled = explanation.causal_correctness_passed is not None
    n_tested = len(explanation.causal_drop_by_zone or {}) if causal_enabled else 0
    n_kept = len(explanation.concept_vectors) if causal_enabled else n_claims
    causal_retention = (n_kept / n_tested) if n_tested else (1.0 if causal_enabled else None)

    per_claim: List[Dict] = []
    for i, cv in enumerate(explanation.concept_vectors):
        drop = (explanation.causal_drop_by_zone or {}).get(cv.zone)
        per_claim.append({
            "zone": cv.zone,
            "artifact_type": cv.artifact_type,
            "faithfulness_passed": (
                bool(explanation.faithfulness_passed[i])
                if i < len(explanation.faithfulness_passed) else True
            ),
            "causal_drop": round(float(drop), 4) if drop is not None else None,
            "causal_passed": True if causal_enabled else None,
        })

    mask_gt: Optional[Dict] = None
    loc_path = Path("webapp/src/data/zone-localization.json")
    if not loc_path.exists():
        loc_path = Path("outputs/zone_localization_ffpp.json")
    if loc_path.exists():
        try:
            raw = json.loads(loc_path.read_text(encoding="utf-8"))
            # Accept both trimmed webapp JSON and full eval JSON.
            if "localization" in raw:
                micro = raw["localization"].get("micro", {})
                mask_gt = {
                    "dataset": "FaceForensics++",
                    "n_frames_evaluated": raw.get("evaluation_metadata", {}).get(
                        "fake_frames_evaluated"
                    ),
                    "precision": micro.get("precision"),
                    "pointing_game_accuracy": raw["localization"].get(
                        "pointing_game_accuracy"
                    ),
                    "zone_score_auc_micro": raw["localization"].get(
                        "zone_score_auc_micro"
                    ),
                    "scope_note": (
                        "Mask GT metrics are from offline FF++ evaluation, not this upload."
                    ),
                }
            else:
                mask_gt = {
                    "dataset": raw.get("dataset", "FaceForensics++"),
                    "n_frames_evaluated": raw.get("n_frames_evaluated"),
                    "precision": raw.get("precision"),
                    "pointing_game_accuracy": raw.get("pointing_game_accuracy"),
                    "zone_score_auc_micro": raw.get("zone_score_auc_micro"),
                    "scope_note": raw.get(
                        "scope_note",
                        "Mask GT metrics are from offline FF++ evaluation, not this upload.",
                    ),
                }
        except Exception as exc:
            log.warning("Could not load zone-localization metrics: %s", exc)

    return {
        "live": {
            "n_claims": n_claims,
            "faithfulness_passed_count": n_faithful,
            "faithfulness_pass_rate": round(faithful_rate, 4),
            "causal_gate_enabled": causal_enabled,
            "causal_min_drop": explanation.causal_min_drop,
            "causal_zones_tested": n_tested,
            "causal_claims_retained": n_kept,
            "causal_retention_rate": (
                round(causal_retention, 4) if causal_retention is not None else None
            ),
            "per_claim": per_claim,
            "note": (
                "Live checks verify (1) every NL claim matches its ConceptVector and "
                "(2) each retained zone causally moves P(fake) when occluded. "
                "They do not prove pixel-level manipulation without a mask."
            ),
        },
        "mask_ground_truth_validation": mask_gt,
        "novel_contribution_note": (
            "The semantic explanation head is validated independently of the "
            "classifier verdict: on FF++ mask ground truth, flagged zones are "
            "high-precision and the top-scored zone hits a truly manipulated "
            "region in the pointing game on nearly all frames — even when "
            "classification is imperfect."
        ),
    }


def risk_level_from_probability(label: str, fake_probability: float) -> str:
    if label == "Real":
        return "Low"
    if fake_probability > 0.95:
        return "Critical"
    if fake_probability > 0.85:
        return "High"
    return "Moderate"


def build_terminal_logs(
    mode: str,
    filename: str,
    device: str,
    total_frames: Optional[int],
    sampled_frames: Optional[int],
    skipped_frames: int,
    per_frame_probs: List[float],
    label: str,
    fake_probability: float,
    beta: float,
    inferred_method: Optional[str],
    elapsed: float,
) -> List[str]:
    """
    Every line below reports something that actually happened during THIS
    request (real device, real frame counts, real per-frame numbers) — not a
    pre-scripted script of flavor text.
    """
    logs = [
        "==================================================",
        f"SAEDF forensic pipeline :: {DEFAULT_HF_DEEPFAKE_MODEL}",
        "==================================================",
        f"Device: {device.upper()}",
        f"Input file: {filename}",
    ]
    if mode == "video":
        logs.append(f"Video opened — {total_frames} total frames, sampling {sampled_frames} frames")
        for i, p in enumerate(per_frame_probs):
            logs.append(f"Frame sample {i+1}/{sampled_frames}: face detected, fake_probability={p:.4f}")
        if skipped_frames:
            logs.append(f"Skipped {skipped_frames} sampled frame(s): no face / landmark failure")
    else:
        logs.append("Face detected (MTCNN) — extracting 68 landmarks (dlib)")
        logs.append(f"fake_probability (ViT softmax) = {fake_probability:.4f}")

    logs += [
        "Computing CLS-attention zone scores A(z) and grad-rollout zone scores S(z)",
        f"Fusing R(z) = beta*A(z) + (1-beta)*S(z), calibrated beta MAP = {beta:.4f}",
    ]
    if inferred_method:
        logs.append(f"Inferred manipulation-method signature: {inferred_method}")
    logs += [
        "Generating verified natural-language zone explanations",
        f"Verdict: {label.upper()}  (fake_probability={fake_probability:.4f})",
        f"Total inference time: {elapsed:.2f}s",
        "==================================================",
        "Completed.",
    ]
    return logs


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    bundle = get_bundle()
    return jsonify({
        "status": "ok",
        "device": str(bundle.extractor.device),
        "model": DEFAULT_HF_DEEPFAKE_MODEL,
        "beta_map": bundle.explainer.beta,
    })


@app.route("/infer/image", methods=["POST"])
def infer_image():
    bundle = get_bundle()
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "No file uploaded."}), 400

    raw_bytes = file.read()
    file_hash = sha256_of_bytes(raw_bytes)
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return jsonify({"error": "Could not decode image file."}), 400

    t0 = time.time()
    with _inference_lock:
        try:
            result = run_single_frame(image_bgr, bundle)
        except NoFaceDetected as exc:
            return jsonify({"error": str(exc), "code": "NO_FACE_DETECTED"}), 422
        heatmap_b64 = render_heatmap_overlay_b64(result["face_224"], result["heatmap_224"])
        face_crop_b64 = face_crop_to_b64(result["face_224"])
        mask_gt = try_build_mask_ground_truth_visual(
            filename=file.filename,
            face_224=result["face_224"],
            crop_box=result.get("crop_box"),
            frame_index=0,
        )
    elapsed = time.time() - t0

    explanation: ExplanationOutput = result["explanation"]
    label = explanation.label
    fake_probability = result["fake_probability"]

    now = datetime.now(timezone.utc)
    response = {
        "prediction": label,
        "confidence": round((fake_probability if label == "Fake" else 1.0 - fake_probability) * 100, 2),
        "riskLevel": risk_level_from_probability(label, fake_probability),
        "processing_time": f"{elapsed:.1f} sec",
        "model": DEFAULT_HF_DEEPFAKE_MODEL,
        "mode": "image",
        "terminal_logs": build_terminal_logs(
            mode="image", filename=file.filename or "uploaded_image",
            device=str(bundle.extractor.device), total_frames=None, sampled_frames=None,
            skipped_frames=0, per_frame_probs=[], label=label,
            fake_probability=fake_probability, beta=bundle.explainer.beta,
            inferred_method=explanation.inferred_method, elapsed=elapsed,
        ),
        "semantic_explanations": build_semantic_explanations(explanation, bundle),
        "facial_regions": build_facial_regions(explanation, bundle),
        "timeline": [],
        "heatmap": heatmap_b64,
        "attention_map": heatmap_b64,
        "faceCrop": face_crop_b64,
        "mask_ground_truth": mask_gt,
        "confidence_breakdown": build_confidence_breakdown(explanation, bundle),
        "classification_reasoning": build_classification_reasoning(
            explanation, fake_probability, bundle
        ),
        "explanation_validation": build_explanation_validation(explanation),
        "report": {
            "caseId": f"EDF-{now.year}-{file_hash[:8].upper()}",
            "date": now.strftime("%d/%m/%Y"),
            "time": now.strftime("%H:%M:%S"),
            "modelVersion": f"{DEFAULT_HF_DEEPFAKE_MODEL} (beta MAP={bundle.explainer.beta:.4f})",
            "inputFile": file.filename or "uploaded_image",
            "hashValue": file_hash,
            "fileType": "image",
        },
    }
    return jsonify(response)


@app.route("/infer/video", methods=["POST"])
def infer_video():
    bundle = get_bundle()
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "No file uploaded."}), 400

    raw_bytes = file.read()
    file_hash = sha256_of_bytes(raw_bytes)

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return jsonify({"error": "Could not open video file."}), 400
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if total_frames <= 0:
            cap.release()
            return jsonify({"error": "Video has no readable frames."}), 400

        max_samples = 10
        sample_count = min(max_samples, total_frames)
        indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)

        t0 = time.time()
        per_frame: List[Dict] = []
        skipped = 0
        best_face_224: Optional[np.ndarray] = None
        best_heatmap_224: Optional[np.ndarray] = None
        best_crop_box: Optional[List[int]] = None
        best_frame_idx: int = 0
        best_prob = -1.0
        best_explanation: Optional[ExplanationOutput] = None

        with _inference_lock:
            for frame_idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ok, frame = cap.read()
                if not ok or frame is None:
                    skipped += 1
                    continue
                try:
                    result = run_single_frame(frame, bundle)
                except NoFaceDetected:
                    skipped += 1
                    continue
                prob = result["fake_probability"]
                per_frame.append({
                    "frame": int(frame_idx),
                    "timeSec": round(float(frame_idx) / float(fps), 2),
                    "fake_probability": prob,
                    "explanation": result["explanation"],
                })
                if prob > best_prob:
                    best_prob = prob
                    best_face_224 = result["face_224"]
                    best_heatmap_224 = result["heatmap_224"]
                    best_explanation = result["explanation"]
                    best_crop_box = result.get("crop_box")
                    best_frame_idx = int(frame_idx)
            cap.release()

            if not per_frame:
                return jsonify({
                    "error": "No face could be detected in any sampled frame.",
                    "code": "NO_FACE_DETECTED",
                }), 422

            heatmap_b64 = render_heatmap_overlay_b64(best_face_224, best_heatmap_224)
            face_crop_b64 = face_crop_to_b64(best_face_224)
            mask_gt = try_build_mask_ground_truth_visual(
                filename=file.filename,
                face_224=best_face_224,
                crop_box=best_crop_box,
                frame_index=best_frame_idx,
            )
        elapsed = time.time() - t0
    finally:
        os.unlink(tmp_path)

    probs = [pf["fake_probability"] for pf in per_frame]
    mean_prob = float(np.mean(probs))
    label = "Fake" if mean_prob >= 0.5 else "Real"

    # Aggregate real R(z)/Q(z) across all successfully-processed frames.
    all_explanations: List[ExplanationOutput] = [pf["explanation"] for pf in per_frame]
    agg_R = np.mean([e.R_scores for e in all_explanations], axis=0)
    agg_Q = None
    if all(e.Q_scores is not None for e in all_explanations):
        agg_Q = np.mean([e.Q_scores for e in all_explanations], axis=0)

    representative = best_explanation  # frame with highest fake_probability — richest explanation set

    now = datetime.now(timezone.utc)
    timeline = [
        {
            "frame": pf["frame"],
            "timeSec": pf["timeSec"],
            "suspicion": round(pf["fake_probability"], 4),
            "note": (
                "High anomaly — elevated fake probability" if pf["fake_probability"] > 0.65 else
                "Moderate anomaly — borderline signal" if pf["fake_probability"] > 0.35 else
                "Low anomaly — within normal range"
            ),
        }
        for pf in per_frame
    ]

    response = {
        "prediction": label,
        "confidence": round((mean_prob if label == "Fake" else 1.0 - mean_prob) * 100, 2),
        "riskLevel": risk_level_from_probability(label, mean_prob),
        "processing_time": f"{elapsed:.1f} sec",
        "model": DEFAULT_HF_DEEPFAKE_MODEL,
        "mode": "video",
        "terminal_logs": build_terminal_logs(
            mode="video", filename=file.filename or "uploaded_video",
            device=str(bundle.extractor.device), total_frames=total_frames,
            sampled_frames=len(indices), skipped_frames=skipped, per_frame_probs=probs,
            label=label, fake_probability=mean_prob, beta=bundle.explainer.beta,
            inferred_method=representative.inferred_method if representative else None,
            elapsed=elapsed,
        ),
        "semantic_explanations": build_semantic_explanations(representative, bundle) if representative else [],
        "facial_regions": build_facial_regions(representative, bundle) if representative else [],
        "timeline": timeline,
        "heatmap": heatmap_b64,
        "attention_map": heatmap_b64,
        "faceCrop": face_crop_b64,
        "mask_ground_truth": mask_gt,
        "confidence_breakdown": build_confidence_breakdown(representative, bundle) if representative else [],
        "classification_reasoning": build_classification_reasoning(
            representative, mean_prob, bundle
        ) if representative else {
            "summary": (
                f"Classified as {label.upper()} from mean P(fake|x) = {mean_prob:.4f} "
                f"across {len(per_frame)} sampled frame(s)."
            ),
            "steps": [{
                "stage": "Video aggregation",
                "formula": "mean P(fake|x) over sampled frames",
                "value": f"{mean_prob:.4f}",
                "interpretation": (
                    f"{len(per_frame)} frame(s) with detectable faces were averaged; "
                    f"verdict = {'Fake' if mean_prob >= 0.5 else 'Real'}."
                ),
            }],
        },
        "explanation_validation": (
            build_explanation_validation(representative) if representative else {
                "live": {
                    "n_claims": 0,
                    "faithfulness_passed_count": 0,
                    "faithfulness_pass_rate": 1.0,
                    "causal_gate_enabled": False,
                    "causal_min_drop": None,
                    "causal_zones_tested": 0,
                    "causal_claims_retained": 0,
                    "causal_retention_rate": None,
                    "per_claim": [],
                    "note": "No face explanations available for this video.",
                },
                "mask_ground_truth_validation": None,
                "novel_contribution_note": (
                    "The semantic explanation head is validated independently of the "
                    "classifier verdict on FF++ mask ground truth."
                ),
            }
        ),
        "report": {
            "caseId": f"EDF-{now.year}-{file_hash[:8].upper()}",
            "date": now.strftime("%d/%m/%Y"),
            "time": now.strftime("%H:%M:%S"),
            "modelVersion": f"{DEFAULT_HF_DEEPFAKE_MODEL} (beta MAP={bundle.explainer.beta:.4f})",
            "inputFile": file.filename or "uploaded_video",
            "hashValue": file_hash,
            "fileType": "video",
        },
    }
    return jsonify(response)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAEDF real-time inference server.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--production", action="store_true",
        help="Serve with waitress (a production WSGI server) instead of the "
             "Flask development server. The Flask dev server is single-"
             "threaded/unstable under real traffic and prints its own "
             "'do not use in production' warning; --production silences that "
             "concern for actual deployments.",
    )
    p.add_argument(
        "--threads", type=int, default=4,
        help="Worker threads for the waitress server (--production only). "
             "Requests still serialise on the single GPU/model via "
             "_inference_lock, so this mainly helps concurrent I/O-bound "
             "requests (health checks, uploads) queue behind inference.",
    )
    return p.parse_args()


if __name__ == "__main__":
    from console_utf8 import ensure_utf8_stdout
    ensure_utf8_stdout()
    args = parse_args()
    log.info("Warming up model bundle ...")
    get_bundle()
    if API_KEY:
        log.info("API key auth ENABLED for %s", RATE_LIMITED_PATHS)
    log.info(
        "Rate limit: %d requests / %.0fs per client IP on %s",
        RATE_LIMIT_MAX_REQUESTS, RATE_LIMIT_WINDOW_SECONDS, RATE_LIMITED_PATHS,
    )
    log.info("Max upload size: %.0f MB", MAX_UPLOAD_MB)

    if args.production:
        from waitress import serve
        log.info("Starting production (waitress) server on http://%s:%d", args.host, args.port)
        serve(app, host=args.host, port=args.port, threads=args.threads)
    else:
        log.info("Starting development server on http://%s:%d", args.host, args.port)
        app.run(host=args.host, port=args.port, threaded=False)
