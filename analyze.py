"""
analyze.py — SAEDF forensic analysis from a raw image or video file.

Accepts any JPEG/PNG image or MP4/AVI/MOV video.  For each detected face
frame the pipeline runs:
  • MTCNN face detection
  • dlib 68-point landmark extraction
  • SAEDFPipeline (single ViT forward pass → classification + explanation)
  • Printed ForensicReport in the terminal (and optional JSON export)

For videos, every --frame-step frame is analysed (default 1 = all frames).
A per-video aggregate verdict is printed at the end.

Usage
-----
    # Single image
    python analyze.py --input face.jpg

    # Video — analyse every frame, save JSON report
    python analyze.py --input clip.mp4 --output report.json

    # Video — every 5th frame only (faster for long videos)
    python analyze.py --input clip.mp4 --frame-step 5

    # Specify severity threshold and dlib predictor path
    python analyze.py --input face.jpg --min-severity low \\
        --predictor shape_predictor_68_face_landmarks.dat
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import dlib
import numpy as np
from mtcnn import MTCNN

from pipeline import SAEDFPipeline
from semantic_patch_mapper import ForensicReport

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
VIDEO_EXTENSIONS  = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

DEFAULT_PREDICTOR = "shape_predictor_68_face_landmarks.dat"


# ---------------------------------------------------------------------------
# Face detection + landmark extraction helpers
# ---------------------------------------------------------------------------

def _load_predictor(predictor_path: str) -> dlib.shape_predictor:
    p = Path(predictor_path)
    if not p.exists():
        print(
            f"\n[ERROR] dlib shape predictor not found: {predictor_path}\n"
            "Download it from:\n"
            "  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n"
            "Extract and place shape_predictor_68_face_landmarks.dat in the project root.",
            file=sys.stderr,
        )
        sys.exit(1)
    return dlib.shape_predictor(str(p))


def detect_and_crop_face(
    detector: MTCNN,
    image_bgr: np.ndarray,
    margin_ratio: float = 0.10,
) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Detect the largest face in image_bgr, return (cropped_bgr, (x1,y1,x2,y2)).
    Returns None if no face detected.
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb)
    if not detections:
        return None

    best = max(detections, key=lambda d: d["box"][2] * d["box"][3])
    x, y, w, h = best["box"]
    ih, iw = image_bgr.shape[:2]
    mx, my = int(w * margin_ratio), int(h * margin_ratio)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(iw, x + w + mx)
    y2 = min(ih, y + h + my)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = image_bgr[y1:y2, x1:x2]
    return (crop, (x1, y1, x2, y2)) if crop.size > 0 else None


def extract_landmarks_68(
    predictor: dlib.shape_predictor,
    face_crop_bgr: np.ndarray,
    target_size: int = IMAGE_SIZE,
) -> Optional[List[List[float]]]:
    """
    Extract 68 dlib landmarks from a face crop, scale to target_size×target_size.
    Returns list of 68 [x, y] pairs, or None on failure.
    """
    gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    rect  = dlib.rectangle(0, 0, w - 1, h - 1)
    shape = predictor(gray, rect)
    if shape.num_parts != 68:
        return None

    sx = target_size / float(w)
    sy = target_size / float(h)
    return [[shape.part(i).x * sx, shape.part(i).y * sy] for i in range(68)]


def preprocess_frame(
    detector: MTCNN,
    predictor: dlib.shape_predictor,
    image_bgr: np.ndarray,
) -> Optional[Tuple[np.ndarray, List[List[float]]]]:
    """
    Full preprocessing for one frame: detect face → extract landmarks → resize.
    Returns (face_224, landmarks_68) or None if no face / landmarks found.
    """
    result = detect_and_crop_face(detector, image_bgr)
    if result is None:
        return None
    crop, _ = result

    landmarks = extract_landmarks_68(predictor, crop)
    if landmarks is None:
        return None

    face_224 = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    return face_224, landmarks


# ---------------------------------------------------------------------------
# Video aggregation — one report for the entire video
# ---------------------------------------------------------------------------

_SEVERITY_ADJ = {"high": "High-confidence", "medium": "Moderate", "low": "Low-confidence"}
_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}

ZONE_NAMES_ORDERED = [
    "mouth", "left_eye", "right_eye", "chin_jaw",
    "left_cheek", "right_cheek", "forehead", "nose_bridge", "nose_tip",
]


def aggregate_reports(reports: List[ForensicReport]) -> dict:
    """
    Aggregate all per-frame ForensicReports into a single rich dict used both
    for the printed forensic report and the saved JSON.
    """
    if not reports:
        return {}

    n_total  = len(reports)
    fake_probs = [r.classification.fake_probability for r in reports]
    mean_p   = float(np.mean(fake_probs))
    max_p    = float(np.max(fake_probs))
    n_fake   = sum(1 for r in reports if r.classification.label == "Fake")
    model    = reports[0].classification.model_name

    # ── zone R(z) and Q(z) averages ──────────────────────────────────────
    all_zones = ZONE_NAMES_ORDERED
    R_accum: dict = {z: [] for z in all_zones}
    Q_accum: dict = {z: [] for z in all_zones}
    for r in reports:
        r_dict = r.explanation.to_dict()
        for z in all_zones:
            R_accum[z].append(r_dict["R_scores"].get(z, 0.0))
            if "Q_scores" in r_dict:
                Q_accum[z].append(r_dict["Q_scores"].get(z, 0.0))

    mean_R = {z: round(float(np.mean(v)), 4) for z, v in R_accum.items()}
    mean_Q = {z: round(float(np.mean(v)), 4) for z, v in Q_accum.items()} if any(Q_accum.values()) else {}

    # ── per-zone flagging count + best artifact type ─────────────────────
    zone_info: dict = {}   # zone -> {count, artifact_counts, confidences, q_scores}
    for r in reports:
        for cv in r.explanation.concept_vectors:
            z = cv.zone
            if z not in zone_info:
                zone_info[z] = {
                    "count": 0,
                    "artifact_counts": {},
                    "confidences": [],
                    "q_scores": [],
                    "severities": [],
                }
            info = zone_info[z]
            info["count"] += 1
            art = cv.artifact_type
            info["artifact_counts"][art] = info["artifact_counts"].get(art, 0) + 1
            info["confidences"].append(cv.confidence)
            if cv.zone_prior_score is not None:
                info["q_scores"].append(cv.zone_prior_score)
            info["severities"].append(cv.severity)

    # Summarise per zone
    zone_summary = {}
    for z, info in zone_info.items():
        top_artifact = max(info["artifact_counts"], key=info["artifact_counts"].get)
        avg_conf     = round(float(np.mean(info["confidences"])), 4)
        avg_q        = round(float(np.mean(info["q_scores"])), 4) if info["q_scores"] else None
        top_severity = min(info["severities"], key=lambda s: _SEVERITY_ORDER.get(s, 9))
        zone_summary[z] = {
            "flagged_frames":  info["count"],
            "top_artifact":    top_artifact,
            "avg_confidence":  avg_conf,
            "avg_q_score":     avg_q,
            "top_severity":    top_severity,
        }

    # Sort zones by flag frequency
    sorted_zones = sorted(zone_summary.items(), key=lambda kv: kv[1]["flagged_frames"], reverse=True)

    # ── method inference — weighted average of per-frame probabilities ────
    method_probs_accum: dict = {}
    for r in reports:
        mp = r.explanation.method_probabilities or {}
        for m, p in mp.items():
            method_probs_accum[m] = method_probs_accum.get(m, 0.0) + p
    if method_probs_accum:
        total_w = sum(method_probs_accum.values())
        method_probs_avg = {m: round(v / total_w, 4) for m, v in method_probs_accum.items()}
        top_method = max(method_probs_avg, key=method_probs_avg.get)
    else:
        method_probs_avg = {}
        top_method = "Unknown"

    # ── confidence tier ────────────────────────────────────────────────────
    if mean_p > 0.75:
        tier = "high"
    elif mean_p > 0.50:
        tier = "medium"
    else:
        tier = "uncertain"

    verdict = "Fake" if mean_p >= 0.5 else "Real"

    return {
        "verdict":               verdict,
        "label":                 verdict,
        "mean_fake_probability": round(mean_p, 4),
        "max_fake_probability":  round(max_p, 4),
        "confidence_tier":       tier,
        "frames_analysed":       n_total,
        "frames_predicted_fake": n_fake,
        "model":                 model,
        "inferred_method":       top_method,
        "method_probabilities":  method_probs_avg,
        "mean_R_scores":         mean_R,
        "mean_Q_scores":         mean_Q,
        "flagged_zones":         [
            {
                "zone":           z,
                "flagged_frames": info["flagged_frames"],
                "top_artifact":   info["top_artifact"],
                "avg_confidence": info["avg_confidence"],
                "avg_q_score":    info["avg_q_score"],
                "top_severity":   info["top_severity"],
            }
            for z, info in sorted_zones
        ],
    }


def print_video_forensic_report(agg: dict, source: str) -> None:
    """
    Print a single unified forensic report for an entire video,
    mirroring the per-image ForensicReport layout exactly.
    """
    n       = agg["frames_analysed"]
    n_fake  = agg["frames_predicted_fake"]
    mean_p  = agg["mean_fake_probability"]
    tier    = agg["confidence_tier"]
    verdict = agg["label"]
    model   = agg.get("model", "")
    method  = agg.get("inferred_method", "Unknown")
    mprobs  = agg.get("method_probabilities", {})
    mean_R  = agg.get("mean_R_scores", {})
    mean_Q  = agg.get("mean_Q_scores", {})
    zones   = agg.get("flagged_zones", [])

    print("\n" + "=" * 62)
    print("  FORENSIC REPORT  (VIDEO)")
    print(f"  Source  : {source}")
    print(f"  Frames  : {n_fake}/{n} predicted fake  ({100*n_fake/max(n,1):.1f}%)")
    print("=" * 62)

    print("\n[HEAD 1 — CLASSIFICATION]")
    print(f"  Label            : {verdict}")
    print(f"  Fake probability : {mean_p:.4f}  ({tier})  [mean over {n} frames]")
    print(f"  Model            : {model}")

    print("\n[HEAD 2 — SEMANTIC EXPLANATION]")
    if mprobs:
        print(f"  Inferred method  : {method}")
        for m, p in sorted(mprobs.items(), key=lambda kv: kv[1], reverse=True):
            bar = "▪" * int(p * 20)
            print(f"    {m:20s} {p:.3f}  {bar}")

    if zones:
        print()
        for zinfo in zones[:5]:   # top 5 most flagged
            z        = zinfo["zone"]
            art      = zinfo["top_artifact"]
            sev      = zinfo["top_severity"]
            conf     = zinfo["avg_confidence"]
            ff       = zinfo["flagged_frames"]
            q        = zinfo["avg_q_score"]
            adj      = _SEVERITY_ADJ.get(sev, sev.capitalize())
            z_label  = z.replace("_", " ")
            q_str    = f"  Q(z)={q:.3f}" if q is not None else ""
            print(
                f"  • {adj} {art} detected in the {z_label} region"
                f"  (avg confidence: {conf*100:.1f}%{q_str},  flagged in {ff}/{n} frames)"
            )
    else:
        print("  No zones consistently above the severity threshold.")

    print(f"\n  Zone scores  [R(z) = fusion avg,  Q(z) = Bayesian posterior avg]:")
    for z in ZONE_NAMES_ORDERED:
        r_s = mean_R.get(z, 0.0)
        q_s = mean_Q.get(z)
        bar = "█" * int(r_s * 20)
        q_str = f"  Q={q_s:.3f}" if q_s is not None else ""
        print(f"    {z:14s} R={r_s:.3f}{q_str}  {bar}")

    print("=" * 62 + "\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SAEDF forensic analysis: image or video → dual-head report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--input", "-i",
        required=True,
        type=Path,
        help="Path to an image (.jpg/.png) or video (.mp4/.avi/.mov).",
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Optional path to save the JSON report (or aggregate JSON for video).",
    )
    p.add_argument(
        "--frame-step",
        type=int,
        default=1,
        metavar="N",
        help="For video: analyse every Nth frame (default 1 = all frames).",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        metavar="N",
        help="For video: stop after N frames analysed (default: no limit).",
    )
    p.add_argument(
        "--min-severity",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
        help="Minimum zone severity to flag (default: medium).",
    )
    p.add_argument(
        "--predictor",
        type=str,
        default=DEFAULT_PREDICTOR,
        help=f"Path to dlib 68-point shape predictor .dat (default: {DEFAULT_PREDICTOR}).",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/fusion_config.json"),
        help="Path to fusion_config.json (default: configs/fusion_config.json).",
    )
    return p.parse_args()


def run_on_image(
    image_path: Path,
    pipe: SAEDFPipeline,
    detector: MTCNN,
    predictor_model: dlib.shape_predictor,
) -> Optional[ForensicReport]:
    """Run the full pipeline on a single image file."""
    print(f"\nReading image: {image_path}")
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"[ERROR] Could not read image: {image_path}", file=sys.stderr)
        return None

    print("Detecting face and extracting landmarks ...")
    result = preprocess_frame(detector, predictor_model, image_bgr)
    if result is None:
        print("[WARN] No face detected or landmark extraction failed.", file=sys.stderr)
        return None

    face_224, landmarks_68 = result
    print("Running dual-head inference ...")
    report = pipe.predict_frame(
        image_bgr=face_224,
        landmarks_68=landmarks_68,
        image_path=str(image_path),
    )
    report.print_summary()
    return report


def run_on_video(
    video_path: Path,
    pipe: SAEDFPipeline,
    detector: MTCNN,
    predictor_model: dlib.shape_predictor,
    frame_step: int = 1,
    max_frames: Optional[int] = None,
) -> List[ForensicReport]:
    """
    Run the pipeline on a video, processing every frame_step-th frame.
    Per-frame output is suppressed — call print_video_forensic_report() on
    the returned list to show one unified report for the whole video.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}", file=sys.stderr)
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration     = total_frames / fps
    print(f"\nVideo : {video_path.name}")
    print(f"        {total_frames} frames  |  {fps:.1f} fps  |  {duration:.1f} s")
    print(f"        sampling every {frame_step} frame(s) — processing ...\n")

    reports:   List[ForensicReport] = []
    frame_idx  = 0
    analysed   = 0
    skipped    = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            result = preprocess_frame(detector, predictor_model, frame_bgr)
            if result is None:
                skipped += 1
                print(f"  frame {frame_idx:5d}  — no face, skipped ({skipped} total skipped)")
            else:
                face_224, landmarks_68 = result
                report = pipe.predict_frame(
                    image_bgr=face_224,
                    landmarks_68=landmarks_68,
                    image_path=f"{video_path.name}::f{frame_idx:06d}",
                )
                reports.append(report)
                analysed += 1
                p   = report.classification.fake_probability
                lbl = report.classification.label
                print(f"  frame {frame_idx:5d}  {lbl:4s}  p={p:.3f}  "
                      f"method={report.explanation.inferred_method or '?':16s}  "
                      f"zones={', '.join(cv.zone for cv in report.explanation.concept_vectors) or '—'}")

            if max_frames is not None and analysed >= max_frames:
                print(f"\n  Reached --max-frames {max_frames}, stopping early.")
                break

        frame_idx += 1

    cap.release()
    print(f"\n  Done — {analysed} frames analysed, {skipped} skipped (no face).")
    return reports


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Validate input
    # ------------------------------------------------------------------
    if not args.input.exists():
        print(f"[ERROR] Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    suffix = args.input.suffix.lower()
    is_image = suffix in IMAGE_EXTENSIONS
    is_video = suffix in VIDEO_EXTENSIONS
    if not is_image and not is_video:
        print(
            f"[ERROR] Unsupported file type: {suffix}\n"
            f"Images: {IMAGE_EXTENSIONS}\nVideos: {VIDEO_EXTENSIONS}",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    print("\nLoading MTCNN face detector ...")
    detector = MTCNN()

    print(f"Loading dlib landmark predictor: {args.predictor}")
    predictor_model = _load_predictor(args.predictor)

    print(f"Loading SAEDFPipeline (ViT deepfake model + Bayesian calibration) ...")
    pipe = SAEDFPipeline(
        config_path=str(args.config),
        min_severity=args.min_severity,
    )

    # ------------------------------------------------------------------
    # Run analysis
    # ------------------------------------------------------------------
    try:
        if is_image:
            report = run_on_image(args.input, pipe, detector, predictor_model)
            if report is None:
                sys.exit(1)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(report.to_json(), encoding="utf-8")
                print(f"JSON report saved to: {args.output}")

        else:  # video
            reports = run_on_video(
                args.input, pipe, detector, predictor_model,
                frame_step=args.frame_step,
                max_frames=args.max_frames,
            )
            if not reports:
                print("[WARN] No frames could be analysed.", file=sys.stderr)
                sys.exit(1)

            agg = aggregate_reports(reports)
            print_video_forensic_report(agg, source=args.input.name)

            if args.output is not None:
                payload = {
                    "source":    str(args.input),
                    "aggregate": agg,
                    "frames":    [json.loads(r.to_json()) for r in reports],
                }
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                print(f"Full JSON report ({len(reports)} frames) saved to: {args.output}")

    finally:
        pipe.close()


if __name__ == "__main__":
    from console_utf8 import ensure_utf8_stdout
    ensure_utf8_stdout()
    main()
