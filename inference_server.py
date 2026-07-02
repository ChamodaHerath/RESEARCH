"""
inference_server.py — Real-time HTTP inference server for the SAEDF pipeline.

This is the ONLY backend the web demo (webapp/) talks to for detection results.
Every field returned here is computed live from the actual uploaded file:

  - fake_probability   : real softmax output of the pretrained ViT deepfake
                          classifier (see attention_extractor.DEFAULT_HF_DEEPFAKE_MODEL,
                          currently hamzenium/ViT-Deepfake-Classifier -- FF++ AUC 1.00,
                          Celeb-DF cross-dataset AUC 0.90; see outputs/model_benchmarks/).
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
import logging
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import dlib
import numpy as np
from flask import Flask, jsonify, request
from mtcnn import MTCNN
from werkzeug.exceptions import RequestEntityTooLarge

from attention_extractor import AttentionExtractor, DEFAULT_HF_DEEPFAKE_MODEL
from baseline_heatmap_generator import extract_full_pipeline_inputs
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
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("inference_server")

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
CONFIG_PATH = "configs/fusion_config.json"
IMAGE_SIZE = 224

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
    the real dual-head inference (classification + explanation).
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

    return {
        "face_224": face_224,
        "fake_probability": inputs["fake_probability"],
        "explanation": explanation,
        "heatmap_224": inputs["heatmap_224"],
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


# ---------------------------------------------------------------------------
# Response assembly (real values only)
# ---------------------------------------------------------------------------

SEVERITY_LEVELS = {"high", "medium", "low"}


def _format_zone(name: str) -> str:
    return name.replace("_", " ")


def build_explanation_theory(cv_, bundle: ModelBundle) -> Dict:
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
                f"{score_label} = {flag_score:.4f} ≥ {flag_threshold:.4f} "
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

    return {
        "summary": (
            f"Zone '{ _format_zone(cv_.zone) }' was flagged because its "
            f"{score_label} score ({flag_score:.4f}) crossed the {min_severity} "
            f"threshold ({flag_threshold:.2f}). The '{cv_.artifact_type}' label "
            f"comes from the Bayesian / ontology artifact mapping above."
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
            "theory": build_explanation_theory(cv_, bundle),
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
    flag_threshold = SEVERITY_THRESHOLDS.get(min_severity, 0.50)
    if flagged:
        steps.append({
            "stage": "Explanation generation gate",
            "formula": f"flag zone if Q(z) or R(z) ≥ {flag_threshold:.2f} ({min_severity})",
            "value": f"{flagged} zone(s) flagged",
            "interpretation": (
                f"{flagged} facial zone(s) exceeded the '{min_severity}' threshold "
                f"({flag_threshold:.2f}) and received concept vectors + NL sentences. "
                f"These provide the semantic evidence supporting forensic review."
            ),
        })
    else:
        steps.append({
            "stage": "Explanation generation gate",
            "formula": f"flag zone if Q(z) or R(z) ≥ {flag_threshold:.2f} ({min_severity})",
            "value": "0 zones flagged",
            "interpretation": (
                f"No zone reached the '{min_severity}' anomaly threshold "
                f"({flag_threshold:.2f}), so no semantic explanation cards are "
                f"emitted — the verdict rests on the ViT classifier probability alone."
            ),
        })

    summary = (
        f"Classified as {label.upper()} because P(fake|x) = {fake_probability:.4f} "
        f"{'≥' if fake_probability >= threshold else '<'} 0.5. "
    )
    if flagged:
        summary += (
            f"{flagged} zone(s) show elevated fused anomaly scores (R/Q) that "
            f"support explainable forensic evidence below."
        )
    elif label == "Fake":
        summary += (
            "The ViT head flags manipulation, but no individual zone crossed the "
            f"{min_severity} explanation threshold — a borderline / global-artifact case."
        )
    else:
        summary += (
            "Both the classifier probability and zone-level fusion scores are "
            "consistent with an authentic face under the calibrated pipeline."
        )

    return {"summary": summary, "steps": steps}


def build_facial_regions(explanation: ExplanationOutput) -> List[Dict]:
    scores_by_zone = {
        ZONE_NAMES[i]: float(explanation.Q_scores[i]) if explanation.Q_scores is not None
        else float(explanation.R_scores[i])
        for i in range(len(ZONE_NAMES))
    }
    regions = []
    for region, zones in DISPLAY_REGION_MAP.items():
        vals = [scores_by_zone[z] for z in zones if z in scores_by_zone]
        anomaly = float(np.mean(vals)) if vals else 0.0
        regions.append({
            "region": region,
            "label": region.capitalize(),
            "anomalyScore": round(anomaly, 4),
        })
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
    return items


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
        "facial_regions": build_facial_regions(explanation),
        "timeline": [],
        "heatmap": heatmap_b64,
        "attention_map": heatmap_b64,
        "faceCrop": face_crop_b64,
        "confidence_breakdown": build_confidence_breakdown(explanation, bundle),
        "classification_reasoning": build_classification_reasoning(
            explanation, fake_probability, bundle
        ),
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
            cap.release()

            if not per_frame:
                return jsonify({
                    "error": "No face could be detected in any sampled frame.",
                    "code": "NO_FACE_DETECTED",
                }), 422

            heatmap_b64 = render_heatmap_overlay_b64(best_face_224, best_heatmap_224)
            face_crop_b64 = face_crop_to_b64(best_face_224)
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
        "facial_regions": build_facial_regions(representative) if representative else [],
        "timeline": timeline,
        "heatmap": heatmap_b64,
        "attention_map": heatmap_b64,
        "faceCrop": face_crop_b64,
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
