"""
sync_webapp_data.py — regenerate webapp/src/data/*.json from the real
pipeline outputs, so the website never drifts from the actual evaluation
results.

This is a thin, auditable copy step (no numbers are computed here, only
selected/trimmed) between:

    outputs/evaluation_results_ffpp.json     -> webapp/src/data/eval-ffpp.json
    outputs/evaluation_results_celebdf.json  -> webapp/src/data/eval-celebdf.json
    configs/fusion_config.json               -> webapp/src/data/fusion-config.json

Heavy per-frame arrays that the webapp does not use (`raw_predictions`,
`calibration_bins`, population `A_stats`/`S_stats`, and the raw Dirichlet
`artifact_posteriors` pseudo-count vectors) are dropped so the site bundle
stays small; every field that IS kept is copied verbatim, never recomputed.

Usage
-----
    python sync_webapp_data.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parent
WEBAPP_DATA = REPO_ROOT / "webapp" / "src" / "data"

EVAL_SOURCES = {
    "eval-ffpp.json": REPO_ROOT / "outputs" / "evaluation_results_ffpp.json",
    "eval-celebdf.json": REPO_ROOT / "outputs" / "evaluation_results_celebdf.json",
}
FUSION_CONFIG_SOURCE = REPO_ROOT / "configs" / "fusion_config.json"
ZONE_LOCALIZATION_SOURCE = REPO_ROOT / "outputs" / "zone_localization_ffpp.json"


def _trim_zone_localization(raw: Dict) -> Dict:
    """Keep thesis-facing localization metrics; drop bulky per-zone count tables."""
    loc = raw.get("localization", {})
    micro = loc.get("micro", {})
    causal = raw.get("causal_faithfulness", {})
    meta = raw.get("evaluation_metadata", {})
    return {
        "dataset": "FaceForensics++",
        "n_frames_evaluated": meta.get("fake_frames_evaluated"),
        "n_frames_total": meta.get("fake_frames_total"),
        "coverage_threshold": meta.get("coverage_threshold"),
        "precision": micro.get("precision"),
        "recall": micro.get("recall"),
        "f1": micro.get("f1"),
        "iou": micro.get("iou"),
        "pointing_game_accuracy": loc.get("pointing_game_accuracy"),
        "pointing_game_n": loc.get("pointing_game_n"),
        "zone_score_auc_micro": loc.get("zone_score_auc_micro"),
        "per_method": {
            method: {
                "precision": vals.get("precision"),
                "recall": vals.get("recall"),
                "f1": vals.get("f1"),
            }
            for method, vals in (loc.get("per_method") or {}).items()
        },
        "causal_faithfulness": {
            "deletion_auc_attribution_order": causal.get("deletion_auc_attribution_order"),
            "deletion_auc_random_order": causal.get("deletion_auc_random_order"),
            "insertion_auc_attribution_order": causal.get("insertion_auc_attribution_order"),
            "insertion_auc_random_order": causal.get("insertion_auc_random_order"),
            "n_frames": causal.get("n_frames"),
        },
        "claim": (
            "When a facial zone is flagged by the explanation head, it coincides with "
            "a truly manipulated region under FF++ manipulation-mask ground truth with "
            "high precision; the top-scored zone hits a GT-manipulated region in the "
            "pointing game on nearly all evaluated frames."
        ),
        "scope_note": (
            "Mask ground-truth validation applies to FaceForensics++ evaluation only. "
            "Live uploads without masks receive textual faithfulness + causal occlusion "
            "checks, not pixel-level manipulation GT."
        ),
    }


def _trim_eval_result(raw: Dict) -> Dict:
    classification = dict(raw["classification_metrics"])
    classification.pop("calibration_bins", None)  # webapp doesn't plot bins; keep bundle small

    return {
        "evaluation_metadata": {
            k: v
            for k, v in raw["evaluation_metadata"].items()
            if k in ("total_frames", "processed_frames", "skipped_frames", "elapsed_seconds", "fps")
        },
        "classification_metrics": classification,
        "explanation_metrics": raw["explanation_metrics"],
        "bayesian_metrics": raw["bayesian_metrics"],
        # raw_predictions intentionally dropped -- only used locally by
        # plot_evaluation.py, would otherwise ship ~5k raw floats per dataset
        # into the client bundle for no on-site benefit.
    }


def _trim_fusion_config(raw: Dict) -> Dict:
    return {
        "model_name": raw.get("model_name"),
        "beta": raw["beta"],
        "beta_posterior": raw["beta_posterior"],
        "severity_thresholds": raw["severity_thresholds"],
        "zone_thresholds": raw.get("zone_thresholds"),
        "zone_thresholds_scale": raw.get("zone_thresholds_scale"),
        "zone_threshold_quantile": raw.get("zone_threshold_quantile"),
        "evidence_weight": raw.get("evidence_weight"),
        "max_flagged_zones": raw.get("max_flagged_zones"),
        "min_fake_probability_for_explanations": raw.get(
            "min_fake_probability_for_explanations"
        ),
        "causal_min_drop": raw.get("causal_min_drop"),
        "beta_for_fusion": raw.get("beta_for_fusion"),
        "research_policy": raw.get("research_policy"),
        "zone_names": raw.get("zone_names"),
        "calibration_n_frames": raw["calibration_n_frames"],
        "calibration_n_fake": raw["calibration_n_fake"],
        "calibration_n_real": raw["calibration_n_real"],
        # A_stats / S_stats / artifact_posteriors intentionally dropped --
        # internal normalisation/Dirichlet pseudo-counts, not meaningful to
        # display and not needed by any current webapp component.
    }


def main() -> None:
    WEBAPP_DATA.mkdir(parents=True, exist_ok=True)

    for dest_name, src_path in EVAL_SOURCES.items():
        if not src_path.exists():
            raise FileNotFoundError(
                f"Missing {src_path} -- run evaluate_pipeline.py first."
            )
        raw = json.loads(src_path.read_text(encoding="utf-8"))
        trimmed = _trim_eval_result(raw)
        dest_path = WEBAPP_DATA / dest_name
        dest_path.write_text(json.dumps(trimmed, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  {src_path}  ->  {dest_path}")

    if not FUSION_CONFIG_SOURCE.exists():
        raise FileNotFoundError(
            f"Missing {FUSION_CONFIG_SOURCE} -- run recalibrate_v2.py first."
        )
    raw_cfg = json.loads(FUSION_CONFIG_SOURCE.read_text(encoding="utf-8"))
    trimmed_cfg = _trim_fusion_config(raw_cfg)
    dest_cfg = WEBAPP_DATA / "fusion-config.json"
    dest_cfg.write_text(json.dumps(trimmed_cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  {FUSION_CONFIG_SOURCE}  ->  {dest_cfg}")

    if ZONE_LOCALIZATION_SOURCE.exists():
        raw_loc = json.loads(ZONE_LOCALIZATION_SOURCE.read_text(encoding="utf-8"))
        trimmed_loc = _trim_zone_localization(raw_loc)
        dest_loc = WEBAPP_DATA / "zone-localization.json"
        dest_loc.write_text(
            json.dumps(trimmed_loc, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  {ZONE_LOCALIZATION_SOURCE}  ->  {dest_loc}")
    else:
        print(
            f"  [skip] {ZONE_LOCALIZATION_SOURCE} missing — "
            "run evaluate_zone_localization.py to publish mask-GT metrics."
        )

    print("\nwebapp/src/data/*.json is now in sync with the real pipeline outputs.")


if __name__ == "__main__":
    from console_utf8 import ensure_utf8_stdout

    ensure_utf8_stdout()
    main()
