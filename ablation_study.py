"""
ablation_study.py — explanation ablations for SAEDF.

The detector probability is held fixed; each ablation changes only the
explanation score used for zone attribution.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from attention_extractor import DEFAULT_HF_DEEPFAKE_MODEL, AttentionExtractor
from baseline_heatmap_generator import extract_full_pipeline_inputs
from semantic_patch_mapper import ZONE_NAMES, load_fusion_config, predict_label_and_explanation
from zone_prior import ZonePriorModel


MODES = [
    "attention_only",
    "gradient_only",
    "fixed_beta",
    "bayesian_r",
    "bayesian_q",
    "full",
]


def _load_entries(path: Path, max_frames: int | None) -> List:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    entries = list(data.get("frames", data).items())
    return entries[:max_frames] if max_frames is not None else entries


def _mode_kwargs(mode: str, cfg, artifact_prior, A_stats, S_stats, beta_posterior):
    beta = float(cfg.get("beta", 0.5))
    beta_mean = float(beta_posterior["mean"]) if beta_posterior else None
    beta_std = float(beta_posterior["std"]) if beta_posterior and "std" in beta_posterior else None
    zone_thresholds = cfg.get("zone_thresholds")
    max_flagged_zones = cfg.get("max_flagged_zones")

    if mode == "attention_only":
        return dict(beta=1.0, artifact_prior=None, zone_prior_model=None)
    if mode == "gradient_only":
        return dict(beta=0.0, artifact_prior=None, zone_prior_model=None)
    if mode == "fixed_beta":
        return dict(beta=0.5, artifact_prior=None, zone_prior_model=None)
    if mode == "bayesian_r":
        return dict(
            beta=beta,
            artifact_prior=None,
            A_stats=A_stats,
            S_stats=S_stats,
            beta_mean=beta_mean,
            beta_std=beta_std,
            zone_prior_model=None,
        )
    if mode == "bayesian_q":
        return dict(
            beta=beta,
            artifact_prior=None,
            A_stats=A_stats,
            S_stats=S_stats,
            beta_mean=beta_mean,
            beta_std=beta_std,
            zone_prior_model=ZonePriorModel(),
            zone_thresholds=zone_thresholds,
            max_flagged_zones=max_flagged_zones,
        )
    if mode == "full":
        return dict(
            beta=beta,
            artifact_prior=artifact_prior,
            A_stats=A_stats,
            S_stats=S_stats,
            beta_mean=beta_mean,
            beta_std=beta_std,
            zone_prior_model=ZonePriorModel(),
            zone_thresholds=zone_thresholds,
            max_flagged_zones=max_flagged_zones,
        )
    raise ValueError(mode)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SAEDF explanation ablation study.")
    p.add_argument("--json", type=Path, default=Path("outputs/ffpp_landmarks_test_full.json"))
    p.add_argument("--config", type=Path, default=Path("configs/fusion_config.json"))
    p.add_argument("--model-name", default=DEFAULT_HF_DEEPFAKE_MODEL)
    p.add_argument("--output", type=Path, default=Path("outputs/ablation_results.json"))
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--min-severity", choices=["low", "medium", "high"], default="medium")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    entries = _load_entries(args.json, args.max_frames)
    cfg, artifact_prior, A_stats, S_stats, beta_posterior = load_fusion_config(str(args.config))
    accum = {
        mode: {
            "faithfulness": [],
            "flagged_fake": [],
            "zone_freq": defaultdict(int),
            "zone_fake": defaultdict(list),
            "zone_real": defaultdict(list),
        }
        for mode in MODES
    }
    # "classifier_only" isn't run through predict_label_and_explanation at
    # all -- it has no zone attribution step, which is exactly the point of
    # including it: it establishes what the classifier alone can and can't
    # tell a forensic analyst, as a baseline for every other row.
    classifier_y_true: List[int] = []
    classifier_y_prob: List[float] = []

    t0 = time.time()
    extractor = AttentionExtractor(model_name=args.model_name, pretrained=True, device=args.device)
    try:
        for idx, (key, entry) in enumerate(entries):
            if idx % 50 == 0:
                print(f"[{idx:5d}/{len(entries)}] elapsed={time.time() - t0:.1f}s")
            image = cv2.imread(entry.get("face_image", ""))
            if image is None:
                continue
            if image.shape != (224, 224, 3):
                image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
            inputs = extract_full_pipeline_inputs(image, extractor)
            gt_fake = entry.get("label") == "fake"
            classifier_y_true.append(1 if gt_fake else 0)
            classifier_y_prob.append(float(inputs["fake_probability"]))

            for mode in MODES:
                kwargs = _mode_kwargs(mode, cfg, artifact_prior, A_stats, S_stats, beta_posterior)
                phi_vec = None if mode == "attention_only" else inputs["phi"]
                report = predict_label_and_explanation(
                    attention_vec=inputs["alpha"],
                    landmarks_68=entry["landmarks_68"],
                    phi_vec=phi_vec,
                    fake_probability=inputs["fake_probability"],
                    min_severity=args.min_severity,
                    **kwargs,
                )
                passed = report.faithfulness_passed
                accum[mode]["faithfulness"].append(sum(passed) / len(passed) if passed else 1.0)
                if gt_fake:
                    accum[mode]["flagged_fake"].append(len(report.concept_vectors))
                for cv in report.concept_vectors:
                    accum[mode]["zone_freq"][cv.zone] += 1
                for z_idx, zone in enumerate(ZONE_NAMES):
                    target = accum[mode]["zone_fake"] if gt_fake else accum[mode]["zone_real"]
                    target[zone].append(float(report.R_scores[z_idx]))
    finally:
        extractor.close()

    results: Dict = {"metadata": {"json": str(args.json), "config": str(args.config), "model": args.model_name}, "modes": {}}

    # classifier_only baseline: the raw detector, no explanation layer at all.
    classifier_only: Dict = {
        "faithfulness_pass_rate": None,
        "mean_flagged_zones_per_fake_frame": None,
        "zone_flag_frequency": None,
        "zone_R_separation": None,
        "note": (
            "No zone-level explanation exists for this baseline (no M matrix, "
            "no A(z)/S(z)/R(z)/Q(z) attribution step), so faithfulness, "
            "flagged-zone counts, and per-zone R(z) separation are not "
            "applicable (null) by construction, not by omission. That gap is "
            "the entire motivation for every other row in this table: the "
            "classifier alone can say THAT a frame is fake, but never WHERE "
            "or WHAT KIND of artifact was found."
        ),
    }
    y_pred = [1 if p >= 0.5 else 0 for p in classifier_y_prob]
    if classifier_y_true:
        correct = sum(t == p for t, p in zip(classifier_y_true, y_pred))
        classifier_only["classification_accuracy"] = round(correct / len(classifier_y_true), 4)
        try:
            from sklearn.metrics import roc_auc_score

            classifier_only["classification_auc"] = round(
                float(roc_auc_score(classifier_y_true, classifier_y_prob)), 4
            )
        except Exception:
            classifier_only["classification_auc"] = None
    results["modes"]["classifier_only"] = classifier_only

    for mode, data in accum.items():
        zone_sep = {}
        for zone in ZONE_NAMES:
            fake_vals = data["zone_fake"][zone]
            real_vals = data["zone_real"][zone]
            fake_mean = float(np.mean(fake_vals)) if fake_vals else None
            real_mean = float(np.mean(real_vals)) if real_vals else None
            zone_sep[zone] = {
                "fake_mean_R": round(fake_mean, 4) if fake_mean is not None else None,
                "real_mean_R": round(real_mean, 4) if real_mean is not None else None,
                "separation": round(fake_mean - real_mean, 4) if fake_mean is not None and real_mean is not None else None,
            }
        results["modes"][mode] = {
            "faithfulness_pass_rate": round(float(np.mean(data["faithfulness"])), 4) if data["faithfulness"] else None,
            "mean_flagged_zones_per_fake_frame": round(float(np.mean(data["flagged_fake"])), 3) if data["flagged_fake"] else None,
            "zone_flag_frequency": dict(sorted(data["zone_freq"].items(), key=lambda kv: kv[1], reverse=True)),
            "zone_R_separation": zone_sep,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Ablation results saved to: {args.output}")


if __name__ == "__main__":
    from console_utf8 import ensure_utf8_stdout
    ensure_utf8_stdout()
    main()
