"""
plot_evaluation.py — ROC/PR curves and calibration reliability diagrams for
one or more evaluate_pipeline.py output JSON files.

Usage
-----
    python plot_evaluation.py --results outputs/evaluation_results_ffpp.json outputs/evaluation_results_celebdf.json \
        --labels "FF++ (in-domain)" "Celeb-DF (cross-dataset)" \
        --output-dir outputs/plots

Requires each results JSON to contain a "raw_predictions" section
(y_true / y_prob), written by evaluate_pipeline.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot ROC/PR/calibration curves from evaluate_pipeline.py results.")
    p.add_argument("--results", nargs="+", required=True, type=Path,
                    help="One or more evaluation_results_*.json files.")
    p.add_argument("--labels", nargs="+", default=None,
                    help="Legend label per results file (defaults to filename stem).")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/plots"))
    return p.parse_args()


def _load(path: Path) -> Dict:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    raw = data.get("raw_predictions")
    if not raw:
        raise ValueError(
            f"{path} has no 'raw_predictions' field. Re-run evaluate_pipeline.py "
            "(with the updated version that saves y_true/y_prob) to generate it."
        )
    return {
        "y_true": np.array(raw["y_true"], dtype=np.int64),
        "y_prob": np.array(raw["y_prob"], dtype=np.float64),
        "classification_metrics": data.get("classification_metrics", {}),
    }


def plot_roc(datasets: List[Dict], labels: List[str], out_path: Path) -> None:
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(6, 6))
    for ds, label in zip(datasets, labels):
        fpr, tpr, _ = roc_curve(ds["y_true"], ds["y_prob"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve — SAEDF detector head")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pr(datasets: List[Dict], labels: List[str], out_path: Path) -> None:
    from sklearn.metrics import precision_recall_curve, average_precision_score

    fig, ax = plt.subplots(figsize=(6, 6))
    for ds, label in zip(datasets, labels):
        precision, recall, _ = precision_recall_curve(ds["y_true"], ds["y_prob"])
        ap = average_precision_score(ds["y_true"], ds["y_prob"])
        ax.plot(recall, precision, linewidth=2, label=f"{label} (AP = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve — SAEDF detector head")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_calibration(datasets: List[Dict], labels: List[str], out_path: Path, n_bins: int = 10) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for ds, label in zip(datasets, labels):
        y_true, y_prob = ds["y_true"], ds["y_prob"]
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_idx = np.clip(np.digitize(y_prob, bin_edges[1:-1]), 0, n_bins - 1)
        mean_conf = np.full(n_bins, np.nan)
        emp_rate = np.full(n_bins, np.nan)
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() > 0:
                mean_conf[b] = y_prob[mask].mean()
                emp_rate[b] = y_true[mask].mean()
        valid = ~np.isnan(mean_conf)
        ax.plot(mean_conf[valid], emp_rate[valid], marker="o", linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted fake_probability (bin)")
    ax.set_ylabel("Empirical fake rate (bin)")
    ax.set_title("Calibration reliability diagram")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    labels = args.labels or [p.stem for p in args.results]
    if len(labels) != len(args.results):
        raise ValueError("--labels must match --results in count.")

    datasets = [_load(p) for p in args.results]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_roc(datasets, labels, args.output_dir / "roc_curve.png")
    plot_pr(datasets, labels, args.output_dir / "pr_curve.png")
    plot_calibration(datasets, labels, args.output_dir / "calibration_curve.png")

    print(f"Saved plots to {args.output_dir}/ (roc_curve.png, pr_curve.png, calibration_curve.png)")


if __name__ == "__main__":
    from console_utf8 import ensure_utf8_stdout
    ensure_utf8_stdout()
    main()
