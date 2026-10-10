"""
detector_benchmark.py — fast classifier-only benchmark for detector candidates.

This script evaluates the detector head without running SAEDF explanations.
Use it to select a credible ViT backbone before recalibrating R(z)/Q(z).

Examples
--------
    python detector_benchmark.py --json outputs/ffpp_landmarks_test.json
    python detector_benchmark.py --json outputs/celebdf_landmarks_test.json --models model_a model_b
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from attention_extractor import DEFAULT_HF_DEEPFAKE_MODEL, AttentionExtractor


DEFAULT_MODELS = [
    DEFAULT_HF_DEEPFAKE_MODEL,
    "prithivMLmods/Deep-Fake-Detector-v2-Model",
    "prithivMLmods/Deepfake-Detection-Exp-02-21",
    "dima806/deepfake_vs_real_image_detection",
]


def _metrics(y_true: List[int], y_prob: List[float], threshold: float = 0.5) -> Dict:
    y_pred = [1 if p >= threshold else 0 for p in y_prob]
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    tn = sum(t == 0 and p == 0 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
    n = max(len(y_true), 1)

    out = {
        "accuracy": round((tp + tn) / n, 4),
        "precision": round(tp / max(tp + fp, 1), 4),
        "recall": round(tp / max(tp + fn, 1), 4),
        "specificity": round(tn / max(tn + fp, 1), 4),
        "f1": round((2 * tp) / max(2 * tp + fp + fn, 1), 4),
        "threshold": threshold,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "n_total": len(y_true),
        "n_fake": sum(y_true),
        "n_real": len(y_true) - sum(y_true),
    }

    try:
        from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

        out["auc_roc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
        out["average_precision"] = round(float(average_precision_score(y_true, y_prob)), 4)
        out["brier_score"] = round(float(brier_score_loss(y_true, y_prob)), 4)
    except Exception as exc:
        out["sklearn_note"] = f"Optional sklearn metrics unavailable: {exc}"

    out.update(_calibration_metrics(y_true, y_prob))
    return out


def _calibration_metrics(y_true: List[int], y_prob: List[float], n_bins: int = 10) -> Dict:
    if not y_true:
        return {"ece": None, "calibration_bins": []}
    y_true_arr = np.asarray(y_true, dtype=np.float32)
    y_prob_arr = np.asarray(y_prob, dtype=np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob_arr >= lo) & (y_prob_arr <= hi)
        else:
            mask = (y_prob_arr >= lo) & (y_prob_arr < hi)
        count = int(mask.sum())
        if count == 0:
            rows.append({"bin": [round(float(lo), 2), round(float(hi), 2)], "count": 0})
            continue
        conf = float(y_prob_arr[mask].mean())
        acc = float(y_true_arr[mask].mean())
        weight = count / len(y_true_arr)
        ece += weight * abs(acc - conf)
        rows.append({
            "bin": [round(float(lo), 2), round(float(hi), 2)],
            "count": count,
            "mean_confidence": round(conf, 4),
            "empirical_fake_rate": round(acc, 4),
            "gap": round(abs(acc - conf), 4),
        })
    return {"ece": round(float(ece), 4), "calibration_bins": rows}


def _load_entries(json_path: Path, max_frames: int | None) -> List[Tuple[str, Dict]]:
    with json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    entries = list(data.get("frames", data).items())
    return entries[:max_frames] if max_frames is not None else entries


def _predict_fake_probability(image_bgr: np.ndarray, extractor: AttentionExtractor) -> float:
    if image_bgr.shape != (224, 224, 3):
        image_bgr = cv2.resize(image_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
    model_input = extractor.preprocess_bgr_images_for_model(np.expand_dims(image_bgr, axis=0))
    x = model_input.to(extractor.device)
    with torch.no_grad():
        out = extractor.model(x, return_dict=True)
        logits = out.logits
        probs = torch.softmax(logits, dim=-1)
        fake_idx = extractor.get_fake_class_index(num_classes=logits.shape[-1])
        return float(probs[0, fake_idx].detach().cpu().item())


def benchmark_model(model_name: str, entries: List[Tuple[str, Dict]], device: str | None) -> Dict:
    print(f"\nBenchmarking {model_name} on {len(entries)} frames ...")
    t0 = time.time()
    y_true: List[int] = []
    y_prob: List[float] = []
    skipped = 0

    extractor = AttentionExtractor(model_name=model_name, pretrained=True, device=device)
    try:
        for i, (key, entry) in enumerate(entries):
            if i % 100 == 0:
                print(f"  [{i:5d}/{len(entries)}] elapsed={time.time() - t0:.1f}s")
            image_bgr = cv2.imread(entry.get("face_image", ""))
            if image_bgr is None:
                skipped += 1
                continue
            y_true.append(1 if entry.get("label") == "fake" else 0)
            y_prob.append(_predict_fake_probability(image_bgr, extractor))
    finally:
        extractor.close()

    metrics = _metrics(y_true, y_prob)
    metrics["skipped"] = skipped
    metrics["elapsed_seconds"] = round(time.time() - t0, 2)
    metrics["fps"] = round(len(y_true) / max(time.time() - t0, 1e-6), 2)
    return {
        "model": model_name,
        "fake_class_index": getattr(extractor, "_fake_class_index", None),
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast benchmark of candidate SAEDF detector heads.")
    p.add_argument("--json", type=Path, default=Path("outputs/ffpp_landmarks_test_full.json"))
    p.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    p.add_argument("--output", type=Path, default=Path("outputs/model_benchmarks/detector_benchmark.json"))
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    entries = _load_entries(args.json, args.max_frames)
    payload = {
        "landmark_json": str(args.json),
        "total_frames": len(entries),
        "results": [],
    }
    for model_name in args.models:
        try:
            payload["results"].append(benchmark_model(model_name, entries, device=args.device))
        except Exception as exc:
            payload["results"].append({"model": model_name, "error": repr(exc)})
            print(f"  [ERROR] {model_name}: {exc}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nBenchmark saved to: {args.output}")


if __name__ == "__main__":
    from console_utf8 import ensure_utf8_stdout
    ensure_utf8_stdout()
    main()
