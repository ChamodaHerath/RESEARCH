"""
evaluate_method_inference.py — quantify the "zero-shot method inference"
contribution against FF++ ground-truth manipulation labels.

zone_prior.py infers the manipulation method (Deepfakes / Face2Face /
FaceSwap / NeuralTextures) from the R(z) zone pattern via cosine similarity
with literature signatures. FF++ frame keys encode the true method
('Face2Face/183_253.mp4::f000030'), so the claim is directly measurable.

Reports
-------
  - Frame-level accuracy, macro precision/recall/F1 per method.
  - 4x4 (+Unknown/Novel column) confusion matrix.
  - Top-2 accuracy (true method within the two highest-probability methods).
  - Video-level accuracy (majority vote over each video's frames).
  - Mean probability assigned to the true method (probabilistic calibration
    of the softmax output, not just argmax).
  - A chance baseline (25 % for 4 balanced methods) for context.

Usage
-----
    python evaluate_method_inference.py \
        --json outputs/ffpp_landmarks_test.json \
        --output outputs/method_inference_ffpp.json

Note: this requires a landmark JSON containing fake frames from FF++ method
directories. Frames whose method cannot be parsed (e.g. Celeb-DF) are skipped
and counted in the metadata.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ffpp_utils import FFPP_METHODS, parse_ffpp_method, video_id_from_key
from landmark_io import load_landmark_json
from pipeline import SAEDFPipeline

DEFAULT_JSON = "outputs/ffpp_landmarks_test_full.json"
DEFAULT_OUTPUT = "outputs/method_inference_ffpp.json"

UNKNOWN = "Unknown/Novel"
CONFUSION_COLUMNS = FFPP_METHODS + [UNKNOWN]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate zone-pattern method inference against FF++ labels."
    )
    p.add_argument("--json", type=Path, default=Path(DEFAULT_JSON))
    p.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    p.add_argument("--config", type=Path, default=Path("configs/fusion_config.json"))
    p.add_argument("--model-name", type=str, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    return p.parse_args()


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def main() -> None:
    args = parse_args()

    data = load_landmark_json(args.json)
    frames: Dict[str, Dict] = data.get("frames", data)

    # Fake frames with a parseable FF++ method label
    labelled = []
    n_unparseable = 0
    for key, entry in frames.items():
        if entry.get("label") != "fake":
            continue
        method = entry.get("method") or parse_ffpp_method(key)
        if method in FFPP_METHODS:
            labelled.append((key, entry, method))
        else:
            n_unparseable += 1
    if args.max_frames is not None:
        labelled = labelled[: args.max_frames]

    if not labelled:
        raise SystemExit(
            "No fake frames with a parseable FF++ method label found in the JSON. "
            "Re-run preprocess_landmarks.py on an FF++ root containing the method "
            "directories (Deepfakes/Face2Face/FaceSwap/NeuralTextures)."
        )

    methods_present = sorted({m for _, _, m in labelled})
    print(f"\nMethod-inference evaluation: {len(labelled)} fake frames "
          f"({n_unparseable} fake frames without a parseable method skipped)")
    print(f"Methods present: {', '.join(methods_present)}")

    confusion: Dict[str, Counter] = {m: Counter() for m in FFPP_METHODS}
    true_prob_sums: Dict[str, List[float]] = defaultdict(list)
    top2_hits: List[bool] = []
    video_votes: Dict[str, Counter] = defaultdict(Counter)
    video_truth: Dict[str, str] = {}
    n_skipped = 0
    t0 = time.time()

    pipe_kwargs = {"config_path": str(args.config)}
    if args.model_name:
        pipe_kwargs["model_name"] = args.model_name

    signature_source = "literature"
    with SAEDFPipeline(**pipe_kwargs) as pipe:
        zpm = getattr(pipe.explainer, "zone_prior_model", None)
        if zpm is not None:
            signature_source = getattr(zpm, "signature_source", "literature")
        print(f"Method-signature source: {signature_source}")
        for i, (key, entry, true_method) in enumerate(labelled):
            if i % 20 == 0:
                print(f"  [{i:4d}/{len(labelled)}]  elapsed: {time.time() - t0:.1f}s")
            try:
                report = pipe.predict_from_landmark_entry(entry, image_path=key)
            except Exception as exc:
                print(f"  [WARN] skipped '{key}': {exc}")
                n_skipped += 1
                continue

            expl = report.explanation
            pred = expl.inferred_method or UNKNOWN
            probs = expl.method_probabilities or {}

            confusion[true_method][pred] += 1
            true_prob_sums[true_method].append(float(probs.get(true_method, 0.0)))

            ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
            top2 = {m for m, _ in ranked[:2]}
            top2_hits.append(true_method in top2)

            vid = video_id_from_key(key)
            video_votes[vid][pred] += 1
            video_truth[vid] = true_method

    n_eval = sum(sum(c.values()) for c in confusion.values())
    if n_eval == 0:
        raise SystemExit("No frames were successfully evaluated.")

    # ---------------- frame-level metrics ----------------
    correct = sum(confusion[m][m] for m in FFPP_METHODS)
    accuracy = _safe_div(correct, n_eval)

    per_method = {}
    for m in FFPP_METHODS:
        support = sum(confusion[m].values())
        tp = confusion[m][m]
        predicted_as_m = sum(confusion[t][m] for t in FFPP_METHODS)
        precision = _safe_div(tp, predicted_as_m)
        recall = _safe_div(tp, support)
        per_method[m] = {
            "support": support,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(_safe_div(2 * precision * recall, precision + recall), 4),
            "mean_prob_assigned_to_true_method": (
                round(float(np.mean(true_prob_sums[m])), 4) if true_prob_sums[m] else None
            ),
        }
    supported = [m for m in FFPP_METHODS if per_method[m]["support"] > 0]
    macro_f1 = round(float(np.mean([per_method[m]["f1"] for m in supported])), 4) if supported else None

    # ---------------- video-level metrics ----------------
    video_correct = 0
    for vid, votes in video_votes.items():
        majority = votes.most_common(1)[0][0]
        if majority == video_truth[vid]:
            video_correct += 1
    video_accuracy = _safe_div(video_correct, len(video_votes))

    results = {
        "evaluation_metadata": {
            "landmark_json": str(args.json),
            "model_name": args.model_name or "pipeline_default",
            "method_signature_source": signature_source,
            "n_frames_evaluated": n_eval,
            "n_frames_skipped_errors": n_skipped,
            "n_fake_frames_without_method_label": n_unparseable,
            "n_videos": len(video_votes),
            "methods_present": methods_present,
            "chance_baseline_accuracy": round(1.0 / len(methods_present), 4)
            if methods_present else None,
            "elapsed_seconds": round(time.time() - t0, 1),
        },
        "frame_level": {
            "accuracy": round(accuracy, 4),
            "top2_accuracy": round(float(np.mean(top2_hits)), 4) if top2_hits else None,
            "macro_f1": macro_f1,
            "per_method": per_method,
            "confusion_matrix": {
                true_m: {col: int(confusion[true_m].get(col, 0)) for col in CONFUSION_COLUMNS}
                for true_m in FFPP_METHODS
            },
        },
        "video_level": {
            "accuracy_majority_vote": round(video_accuracy, 4),
            "n_videos": len(video_votes),
        },
    }

    # ---------------- console summary ----------------
    print("\n" + "=" * 68)
    print("  METHOD INFERENCE vs FF++ GROUND TRUTH")
    print("=" * 68)
    print(f"  frame accuracy       : {results['frame_level']['accuracy']}"
          f"  (chance: {results['evaluation_metadata']['chance_baseline_accuracy']})")
    print(f"  top-2 accuracy       : {results['frame_level']['top2_accuracy']}")
    print(f"  macro F1             : {results['frame_level']['macro_f1']}")
    print(f"  video-level accuracy : {results['video_level']['accuracy_majority_vote']} "
          f"(majority vote, {len(video_votes)} videos)")
    print("\n  Confusion matrix (rows = true, cols = predicted):")
    header = f"    {'':16s}" + "".join(f"{c[:12]:>14s}" for c in CONFUSION_COLUMNS)
    print(header)
    for true_m in FFPP_METHODS:
        row = results["frame_level"]["confusion_matrix"][true_m]
        cells = "".join(f"{row[c]:>14d}" for c in CONFUSION_COLUMNS)
        print(f"    {true_m:16s}{cells}")
    print("=" * 68 + "\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    from console_utf8 import ensure_utf8_stdout
    ensure_utf8_stdout()
    main()
