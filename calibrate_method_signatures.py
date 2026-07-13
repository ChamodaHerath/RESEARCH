"""
calibrate_method_signatures.py — learn empirical method signatures from data.

Replaces the literature-based zone signatures used for zero-shot method
inference (zone_prior.py) with signatures learned from a leakage-free FF++
calibration split (train or val — NEVER the test split):

  1. For every fake frame with a known manipulation method, run the pipeline
     and record the fused zone vector R(z) (9 numbers).
  2. Average R(z) per method → 4 learned "signatures" (centroids).
  3. Optionally fit a multinomial logistic regression (9 features → 4 classes)
     on the same R(z) vectors — usually better than cosine-to-centroid.

The output JSON is auto-loaded by SemanticExplanationGenerator (and therefore
by SAEDFPipeline and evaluate_method_inference.py) when it exists at
configs/method_signatures.json, so method inference on the test split will
use the learned signatures instead of the paper values.

Note: R(z) is computed *before* the zone-prior model is applied, so learned
signatures do not contaminate their own calibration.

Usage
-----
    # 1. Preprocess a calibration split (val is the cheapest legitimate one):
    python preprocess_landmarks.py \
        --ffpp-root "datasets/FaceForensics++/FaceForensics++_C23" \
        --official-splits "datasets/FaceForensics++/splits" \
        --split val \
        --output-json outputs/ffpp_landmarks_val.json

    # 2. Learn signatures (add --fit-classifier for the logistic upgrade):
    python calibrate_method_signatures.py \
        --json outputs/ffpp_landmarks_val.json --fit-classifier

    # 3. Re-run the test-split evaluation — learned signatures now active:
    python evaluate_method_inference.py \
        --json outputs/ffpp_landmarks_test_full.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

from ffpp_utils import FFPP_METHODS, parse_ffpp_method
from landmark_io import load_landmark_json
from pipeline import SAEDFPipeline
from semantic_patch_mapper import ZONE_NAMES

DEFAULT_OUTPUT = "configs/method_signatures.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Learn empirical R(z) method signatures from an FF++ "
                    "calibration split (train or val — never test)."
    )
    p.add_argument("--json", type=Path, required=True,
                   help="Landmark JSON of the calibration split.")
    p.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    p.add_argument("--config", type=Path, default=Path("configs/fusion_config.json"))
    p.add_argument("--model-name", type=str, default=None)
    p.add_argument("--max-frames-per-method", type=int, default=1000,
                   help="Evenly subsample at most this many frames per method "
                        "(default 1000; R(z) needs a forward+backward pass per frame).")
    p.add_argument("--fit-classifier", action="store_true",
                   help="Also fit a multinomial logistic regression "
                        "(9 features -> methods) on the R(z) vectors; "
                        "requires scikit-learn.")
    return p.parse_args()


def _subsample_evenly(items: List, n_max: int) -> List:
    if len(items) <= n_max:
        return items
    idx = np.linspace(0, len(items) - 1, n_max).round().astype(int)
    return [items[i] for i in idx]


def _cosine_self_accuracy(
    vectors: Dict[str, List[np.ndarray]],
    signatures: Dict[str, Dict[str, float]],
) -> float:
    methods = list(signatures.keys())
    sig_mat = np.array(
        [[signatures[m][z] for z in ZONE_NAMES] for m in methods], dtype=np.float64
    )
    sig_norms = np.linalg.norm(sig_mat, axis=1) + 1e-8
    correct = total = 0
    for true_m, vecs in vectors.items():
        for v in vecs:
            sims = (sig_mat @ v) / (sig_norms * (np.linalg.norm(v) + 1e-8))
            if methods[int(np.argmax(sims))] == true_m:
                correct += 1
            total += 1
    return correct / total if total else 0.0


def main() -> None:
    args = parse_args()

    data = load_landmark_json(args.json)
    frames: Dict[str, Dict] = data.get("frames", data)

    # Group fake frames by ground-truth method
    by_method: Dict[str, List] = defaultdict(list)
    for key, entry in frames.items():
        if entry.get("label") != "fake":
            continue
        method = entry.get("method") or parse_ffpp_method(key)
        if method in FFPP_METHODS:
            by_method[method].append((key, entry))

    if not by_method:
        raise SystemExit(
            "No fake frames with a parseable FF++ method label found. "
            "Preprocess an FF++ train or val split first."
        )

    missing = [m for m in FFPP_METHODS if m not in by_method]
    if missing:
        print(f"[WARN] No frames for: {', '.join(missing)} — signatures for "
              f"these methods will keep the literature values.")

    selected: Dict[str, List] = {
        m: _subsample_evenly(items, args.max_frames_per_method)
        for m, items in by_method.items()
    }
    n_total = sum(len(v) for v in selected.values())
    for m in sorted(selected):
        print(f"  {m:16s}: {len(selected[m]):5d} frames selected "
              f"(of {len(by_method[m])} available)")
    print(f"  Total: {n_total} frames\n")

    # ---------------- collect R(z) vectors ----------------
    pipe_kwargs = {"config_path": str(args.config)}
    if args.model_name:
        pipe_kwargs["model_name"] = args.model_name

    vectors: Dict[str, List[np.ndarray]] = defaultdict(list)
    n_skipped = 0
    t0 = time.time()
    done = 0
    with SAEDFPipeline(**pipe_kwargs) as pipe:
        for method, items in selected.items():
            for key, entry in items:
                if done % 50 == 0:
                    print(f"  [{done:5d}/{n_total}]  elapsed: {time.time() - t0:.1f}s")
                done += 1
                try:
                    report = pipe.predict_from_landmark_entry(entry, image_path=key)
                except Exception as exc:
                    print(f"  [WARN] skipped '{key}': {exc}")
                    n_skipped += 1
                    continue
                vectors[method].append(
                    np.asarray(report.explanation.R_scores, dtype=np.float64)
                )

    vectors = {m: v for m, v in vectors.items() if v}
    if len(vectors) < 2:
        raise SystemExit("Need R(z) vectors for at least 2 methods to calibrate.")

    # ---------------- centroids (learned signatures) ----------------
    signatures: Dict[str, Dict[str, float]] = {}
    signature_stds: Dict[str, Dict[str, float]] = {}
    for method, vecs in vectors.items():
        arr = np.stack(vecs)
        signatures[method] = {
            z: round(float(mu), 6) for z, mu in zip(ZONE_NAMES, arr.mean(axis=0))
        }
        signature_stds[method] = {
            z: round(float(sd), 6) for z, sd in zip(ZONE_NAMES, arr.std(axis=0))
        }

    cos_acc = _cosine_self_accuracy(vectors, signatures)
    print(f"\n  Cosine-to-centroid self-accuracy (calibration split): {cos_acc:.4f}")

    # ---------------- optional logistic-regression upgrade ----------------
    classifier = None
    if args.fit_classifier:
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            raise SystemExit(
                "--fit-classifier requires scikit-learn: pip install scikit-learn"
            )
        classes = sorted(vectors.keys())
        X = np.concatenate([np.stack(vectors[m]) for m in classes])
        y = np.concatenate([
            np.full(len(vectors[m]), i) for i, m in enumerate(classes)
        ])
        mean, std = X.mean(axis=0), X.std(axis=0)
        Xs = (X - mean) / np.maximum(std, 1e-8)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xs, y)
        lr_acc = float(clf.score(Xs, y))
        print(f"  Logistic-regression self-accuracy (calibration split): {lr_acc:.4f}")

        coef = clf.coef_
        intercept = clf.intercept_
        # Binary sklearn models store a single row; expand to per-class rows
        if coef.shape[0] == 1 and len(classes) == 2:
            coef = np.vstack([-coef[0], coef[0]])
            intercept = np.array([-intercept[0], intercept[0]])
        classifier = {
            "type": "multinomial_logistic_regression",
            "classes": classes,
            "coef": coef.tolist(),
            "intercept": intercept.tolist(),
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
            "zone_names": ZONE_NAMES,
            "train_self_accuracy": round(lr_acc, 4),
        }

    # ---------------- save ----------------
    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_json": str(args.json),
        "model_name": args.model_name or "pipeline_default",
        "zone_names": ZONE_NAMES,
        "n_frames_used": {m: len(v) for m, v in vectors.items()},
        "n_frames_skipped_errors": n_skipped,
        "cosine_self_accuracy": round(cos_acc, 4),
        "signatures": signatures,
        "signature_stds": signature_stds,
    }
    if classifier is not None:
        out["classifier"] = classifier

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"\nLearned signatures saved to: {args.output}")
    print("They are auto-loaded by the pipeline; re-run "
          "evaluate_method_inference.py on the TEST split to measure the gain.")

    print("\n  Learned mean R(z) per method:")
    header = f"    {'Method':16s}" + "".join(f"{z[:10]:>12s}" for z in ZONE_NAMES)
    print(header)
    for m in sorted(signatures):
        cells = "".join(f"{signatures[m][z]:>12.4f}" for z in ZONE_NAMES)
        print(f"    {m:16s}{cells}")


if __name__ == "__main__":
    from console_utf8 import ensure_utf8_stdout
    ensure_utf8_stdout()
    main()
