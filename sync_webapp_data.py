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
        "max_flagged_zones": raw.get("max_flagged_zones"),
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

    print("\nwebapp/src/data/*.json is now in sync with the real pipeline outputs.")


if __name__ == "__main__":
    from console_utf8 import ensure_utf8_stdout

    ensure_utf8_stdout()
    main()
