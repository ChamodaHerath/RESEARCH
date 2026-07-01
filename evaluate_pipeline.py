"""
evaluate_pipeline.py — Batch evaluation of the SAEDF dual-head pipeline.

Runs SAEDFPipeline over every frame in a landmark JSON and computes:

  Classification metrics
    - Accuracy, AUC-ROC, Precision, Recall, F1  (requires scikit-learn)

  Explanation metrics
    - Faithfulness pass rate (% NL sentences passing verify_faithfulness)
    - Mean number of flagged zones per fake frame
    - Zone frequency distribution across all frames
    - Per-zone mean R(z) split by fake / real label

  Bayesian metrics
    - Mean 95 % credible interval width per zone  (R_ci_upper - R_ci_lower)
    - β posterior summary from fusion_config.json

Usage
-----
    python evaluate_pipeline.py
    python evaluate_pipeline.py --json outputs/ffpp_landmarks_test.json
    python evaluate_pipeline.py --json outputs/ffpp_landmarks_cal.json --output outputs/eval_cal.json
    python evaluate_pipeline.py --max-frames 50   # quick smoke-test
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from pipeline import SAEDFPipeline
from semantic_patch_mapper import ZONE_NAMES, load_fusion_config

DEFAULT_JSON   = "outputs/ffpp_landmarks_test.json"
DEFAULT_OUTPUT = "outputs/evaluation_results.json"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate the SAEDF dual-head pipeline on a landmark JSON."
    )
    p.add_argument(
        "--json",
        type=Path,
        default=DEFAULT_JSON,
        help="Landmark JSON produced by preprocess_landmarks.py.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write evaluation_results.json.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/fusion_config.json"),
        help="Path to fusion_config.json.",
    )
    p.add_argument(
        "--min-severity",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
        help="Zone severity threshold passed to the pipeline.",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process at most N frames (useful for quick smoke-tests).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _try_sklearn_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: List[float],
) -> Dict:
    """Compute AUC + classification report via scikit-learn if available."""
    try:
        from sklearn.metrics import (
            accuracy_score,
            roc_auc_score,
            precision_score,
            recall_score,
            f1_score,
        )
        return {
            "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
            "auc_roc":   round(float(roc_auc_score(y_true, y_prob)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        }
    except ImportError:
        # sklearn not installed — compute accuracy manually
        correct = sum(t == p for t, p in zip(y_true, y_pred))
        return {
            "accuracy":  round(correct / max(len(y_true), 1), 4),
            "note":      "scikit-learn not installed; AUC/precision/recall unavailable",
        }


def _zone_ci_width(report) -> Optional[Dict[str, float]]:
    """Extract per-zone CI width from a ForensicReport's concept vectors."""
    widths: Dict[str, List[float]] = defaultdict(list)
    for cv in report.explanation.concept_vectors:
        if cv.R_ci_lower is not None and cv.R_ci_upper is not None:
            widths[cv.zone].append(cv.R_ci_upper - cv.R_ci_lower)
    return {z: round(float(np.mean(ws)), 4) for z, ws in widths.items()} if widths else None


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Load landmark JSON
    landmark_path = Path(args.json)
    if not landmark_path.exists():
        raise FileNotFoundError(f"Landmark JSON not found: {landmark_path}")
    with open(landmark_path) as fh:
        data = json.load(fh)
    frames_dict: Dict = data.get("frames", data)
    all_entries = list(frames_dict.items())

    if args.max_frames is not None:
        all_entries = all_entries[: args.max_frames]

    total = len(all_entries)
    print(f"\nEvaluating {total} frames from {landmark_path}")

    # Load beta posterior summary for Bayesian metrics section
    cfg, *_ = load_fusion_config(str(args.config))
    beta_info = cfg.get("beta_posterior", {})

    # Accumulate results
    y_true:    List[int]   = []   # ground-truth label (1=fake, 0=real)
    y_pred:    List[int]   = []   # predicted label from classifier
    y_prob:    List[float] = []   # fake_probability

    # Explanation accumulators
    faithfulness_scores: List[float]       = []   # per frame: fraction passed
    flagged_zone_counts_fake: List[int]    = []
    zone_freq:    Dict[str, int]           = defaultdict(int)
    zone_R_fake:  Dict[str, List[float]]   = defaultdict(list)
    zone_R_real:  Dict[str, List[float]]   = defaultdict(list)
    ci_widths_all: Dict[str, List[float]]  = defaultdict(list)

    skipped = 0
    t0 = time.time()

    print(f"Loading pipeline ...")
    with SAEDFPipeline(
        config_path=str(args.config),
        min_severity=args.min_severity,
    ) as pipe:
        for i, (key, entry) in enumerate(all_entries):
            if i % 20 == 0:
                elapsed = time.time() - t0
                print(f"  [{i:4d}/{total}]  elapsed: {elapsed:.1f}s")

            gt_label = 1 if entry.get("label") == "fake" else 0

            try:
                report = pipe.predict_from_landmark_entry(entry, image_path=key)
            except Exception as exc:
                print(f"  [WARN] skipped '{key}': {exc}")
                skipped += 1
                continue

            # --- Classification ---
            y_true.append(gt_label)
            y_prob.append(report.classification.fake_probability)
            y_pred.append(1 if report.classification.label == "Fake" else 0)

            # --- Explanation faithfulness ---
            passed = report.explanation.faithfulness_passed
            if passed:
                faithfulness_scores.append(sum(passed) / len(passed))
            else:
                faithfulness_scores.append(1.0)  # no zones = trivially faithful

            # --- Flagged zones (fake frames) ---
            if gt_label == 1:
                flagged_zone_counts_fake.append(len(report.explanation.concept_vectors))

            # --- Zone R(z) distribution ---
            R = report.explanation.R_scores   # (9,)
            for z_idx, zone in enumerate(ZONE_NAMES):
                score = float(R[z_idx])
                if gt_label == 1:
                    zone_R_fake[zone].append(score)
                else:
                    zone_R_real[zone].append(score)

            # --- Zone frequency (flagged zones only) ---
            for cv in report.explanation.concept_vectors:
                zone_freq[cv.zone] += 1

            # --- Credible interval widths ---
            w = _zone_ci_width(report)
            if w:
                for zone, width in w.items():
                    ci_widths_all[zone].append(width)

    elapsed_total = time.time() - t0
    processed = total - skipped

    # -----------------------------------------------------------------------
    # Assemble metrics
    # -----------------------------------------------------------------------

    # Classification metrics
    classification_metrics = _try_sklearn_metrics(y_true, y_pred, y_prob)
    classification_metrics["n_total"]    = processed
    classification_metrics["n_fake"]     = int(sum(y_true))
    classification_metrics["n_real"]     = int(processed - sum(y_true))
    classification_metrics["n_skipped"]  = skipped

    # Explanation metrics
    faithfulness_rate = round(float(np.mean(faithfulness_scores)), 4) if faithfulness_scores else None
    mean_flagged_fake = round(float(np.mean(flagged_zone_counts_fake)), 3) if flagged_zone_counts_fake else None

    zone_separation = {}
    for zone in ZONE_NAMES:
        fake_mean = float(np.mean(zone_R_fake[zone])) if zone_R_fake[zone] else None
        real_mean = float(np.mean(zone_R_real[zone])) if zone_R_real[zone] else None
        zone_separation[zone] = {
            "fake_mean_R": round(fake_mean, 4) if fake_mean is not None else None,
            "real_mean_R": round(real_mean, 4) if real_mean is not None else None,
            "separation":  round(fake_mean - real_mean, 4) if (fake_mean is not None and real_mean is not None) else None,
            "flag_count":  zone_freq.get(zone, 0),
        }

    # Sort zones by flagging frequency for the summary
    zone_freq_sorted = dict(
        sorted(zone_freq.items(), key=lambda kv: kv[1], reverse=True)
    )

    # Bayesian metrics
    mean_ci_width = {
        zone: round(float(np.mean(ws)), 4)
        for zone, ws in ci_widths_all.items()
        if ws
    }

    results = {
        "evaluation_metadata": {
            "landmark_json":   str(landmark_path),
            "config_path":     str(args.config),
            "min_severity":    args.min_severity,
            "total_frames":    total,
            "processed_frames": processed,
            "skipped_frames":  skipped,
            "elapsed_seconds": round(elapsed_total, 1),
            "fps":             round(processed / max(elapsed_total, 1e-6), 2),
        },
        "classification_metrics": classification_metrics,
        "explanation_metrics": {
            "faithfulness_pass_rate":    faithfulness_rate,
            "mean_flagged_zones_per_fake_frame": mean_flagged_fake,
            "zone_flag_frequency":       zone_freq_sorted,
            "zone_R_separation":         zone_separation,
        },
        "bayesian_metrics": {
            "beta_posterior":            beta_info,
            "mean_CI_width_per_zone":    mean_ci_width,
            "note": (
                "CI width = R_ci_upper - R_ci_lower per zone per frame, averaged "
                "across all frames where a zone was flagged. Narrower = more certain."
            ),
        },
    }

    # -----------------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)

    print("\n[CLASSIFICATION]")
    cm = results["classification_metrics"]
    for k, v in cm.items():
        print(f"  {k:30s}: {v}")

    print("\n[EXPLANATION]")
    em = results["explanation_metrics"]
    print(f"  {'faithfulness_pass_rate':30s}: {em['faithfulness_pass_rate']}")
    print(f"  {'mean_flagged_zones (fake)':30s}: {em['mean_flagged_zones_per_fake_frame']}")
    print("\n  Zone flagging frequency (most → least):")
    for zone, cnt in list(em["zone_flag_frequency"].items())[:9]:
        print(f"    {zone:14s}: {cnt}")

    print("\n  Per-zone R(z) separation (fake mean − real mean):")
    for zone, info in em["zone_R_separation"].items():
        sep = info.get("separation")
        sep_str = f"{sep:+.4f}" if sep is not None else "N/A"
        print(f"    {zone:14s}: {sep_str}")

    print("\n[BAYESIAN]")
    bm = results["bayesian_metrics"]
    if bm["beta_posterior"]:
        bp = bm["beta_posterior"]
        print(f"  MAP β  = {bp.get('map', 'N/A')}")
        print(f"  Mean β = {bp.get('mean', 'N/A')}")
        print(f"  95% CI = [{bp.get('ci_lower', 'N/A')}, {bp.get('ci_upper', 'N/A')}]")
    if bm["mean_CI_width_per_zone"]:
        print("  Mean CI width per zone:")
        for zone, w in bm["mean_CI_width_per_zone"].items():
            print(f"    {zone:14s}: ±{w:.4f}")

    print("=" * 60 + "\n")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
