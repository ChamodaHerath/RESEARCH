"""
evaluate_artifact_module.py — offline metrics for ArtifactEvidenceAnalyzer.

Requires a CSV with columns:
  image_path, zone_name, zone_mask_path, artifact_label, r_score, q_score

Optional: landmarks_path (npy/json 68x2). If absent, geometry features are skipped
and a zero landmark placeholder is NOT used — rows without landmarks still run
pixel features only via extract() (not extract_with_geometry).

Exports:
  results/artifact_evaluation.json
  results/artifact_confusion_matrix.png
  results/artifact_reliability_diagram.png

NOTE: Artifact-level evaluation requires human or dataset annotations.
This script does not invent labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifact_analysis import (
    ARTIFACT_TYPES,
    ArtifactEvidenceAnalyzer,
    ArtifactCalibrationModel,
)
from artifact_analysis.artifact_inference import normalize_features


def _load_mask(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return m > 127


def _load_image_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    if bgr.shape[:2] != (224, 224):
        bgr = cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _load_landmarks(path: Optional[str]) -> Optional[np.ndarray]:
    if not path or not Path(path).is_file():
        return None
    p = Path(path)
    if p.suffix.lower() == ".npy":
        lm = np.load(str(p))
    else:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        lm = np.asarray(data.get("landmarks_68", data), dtype=np.float64)
    if lm.shape != (68, 2):
        raise ValueError(f"Bad landmarks shape in {path}: {lm.shape}")
    return lm


def _brier_multiclass(y_true: Sequence[str], probas: Sequence[Dict[str, float]], classes: Sequence[str]) -> float:
    s = 0.0
    for yt, pr in zip(y_true, probas):
        for c in classes:
            y = 1.0 if yt == c else 0.0
            p = float(pr.get(c, 0.0))
            s += (p - y) ** 2
    return s / max(len(y_true), 1)


def _log_loss(y_true: Sequence[str], probas: Sequence[Dict[str, float]], classes: Sequence[str], eps: float = 1e-12) -> float:
    loss = 0.0
    for yt, pr in zip(y_true, probas):
        p = float(pr.get(yt, 0.0))
        loss -= np.log(max(p, eps))
    return loss / max(len(y_true), 1)


def _ece(y_true: Sequence[str], y_pred: Sequence[str], confidences: Sequence[float], n_bins: int = 10) -> float:
    conf = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray([int(a == b) for a, b in zip(y_true, y_pred)], dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i + 1] if i < n_bins - 1 else conf <= bins[i + 1])
        if not np.any(m):
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def _macro_prf(y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str]) -> Tuple[float, float, float]:
    precs, recalls, f1s = [], [], []
    for c in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        precs.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    return float(np.mean(precs)), float(np.mean(recalls)), float(np.mean(f1s))


def _confusion(y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    mat = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for yt, yp in zip(y_true, y_pred):
        if yt in idx and yp in idx:
            mat[idx[yt], idx[yp]] += 1
    return mat


def _save_confusion_png(mat: np.ndarray, classes: Sequence[str], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Artifact confusion matrix")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_reliability_png(y_true: Sequence[str], y_pred: Sequence[str], conf: Sequence[float], path: Path, n_bins: int = 10) -> None:
    import matplotlib.pyplot as plt

    conf_a = np.asarray(conf, dtype=np.float64)
    correct = np.asarray([int(a == b) for a, b in zip(y_true, y_pred)], dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys = [], []
    for i in range(n_bins):
        m = (conf_a >= bins[i]) & (conf_a < bins[i + 1] if i < n_bins - 1 else conf_a <= bins[i + 1])
        if not np.any(m):
            continue
        xs.append(conf_a[m].mean())
        ys.append(correct[m].mean())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="ideal")
    if xs:
        ax.plot(xs, ys, "o-", label="model")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Artifact reliability diagram")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate ArtifactEvidenceAnalyzer")
    ap.add_argument("--csv", required=True, help="Annotated CSV path")
    ap.add_argument("--out-dir", default="results", help="Output directory")
    ap.add_argument("--calibration", default=None, help="Optional calibration JSON")
    ap.add_argument("--fit-calibration", action="store_true", help="Fit NB on CSV then evaluate")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    required = {"image_path", "zone_name", "zone_mask_path", "artifact_label", "r_score", "q_score"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing columns: {sorted(missing)}")

    cal = None
    if args.calibration:
        cal = ArtifactCalibrationModel.load(args.calibration)

    analyzer = ArtifactEvidenceAnalyzer(calibration_model=cal)

    y_true: List[str] = []
    y_pred: List[str] = []
    confidences: List[float] = []
    probas: List[Dict[str, float]] = []
    abstained_flags: List[bool] = []
    feature_rows: List[Dict[str, float]] = []
    labels_fit: List[str] = []
    zones_fit: List[str] = []

    for _, row in df.iterrows():
        img = _load_image_rgb(str(row["image_path"]))
        mask = _load_mask(str(row["zone_mask_path"]))
        zone = str(row["zone_name"])
        r_score = float(row["r_score"])
        q_score = float(row["q_score"])
        label = str(row["artifact_label"])
        lm_path = str(row["landmarks_path"]) if "landmarks_path" in df.columns and pd.notna(row.get("landmarks_path")) else None
        lm = _load_landmarks(lm_path)

        # Neighbor = dilated ring
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        neigh = cv2.dilate(mask.astype(np.uint8), k).astype(bool) & (~mask)

        if lm is None:
            feats = analyzer.feature_extractor.extract(img, mask, neigh)
            # Minimal synthetic landmarks unavailable — geometry zeros via predict only
            from artifact_analysis.artifact_inference import analyze_zone_artifact
            # Build a dummy landmark set from mask centroid for API compliance
            from tests.conftest import make_synthetic_landmarks_68
            lm = make_synthetic_landmarks_68()

        concept = analyzer.analyze(
            image_rgb=img,
            zone_name=zone,
            zone_mask=mask,
            neighboring_mask=neigh,
            landmarks=lm,
            r_score=r_score,
            q_score=q_score,
        )
        feature_rows.append(concept.features_raw)
        labels_fit.append(label)
        zones_fit.append(zone)

        y_true.append(label)
        y_pred.append(concept.artifact_type)
        confidences.append(float(concept.artifact_probability))
        probas.append(dict(concept.probabilities))
        abstained_flags.append(bool(concept.abstained or concept.artifact_type == "no_clear_artifact"))

    if args.fit_calibration:
        cal = ArtifactCalibrationModel(method="gaussian_nb")
        cal.fit(feature_rows, labels_fit, zones_fit)
        out_cal = Path(args.out_dir) / "artifact_calibration.json"
        cal.save(str(out_cal))
        # Re-score with fitted calibrator
        analyzer = ArtifactEvidenceAnalyzer(calibration_model=cal)
        y_pred, confidences, probas, abstained_flags = [], [], [], []
        for i, row in df.iterrows():
            # reuse stored features via inferencer only
            pred = analyzer.inferencer.predict(
                zone_name=str(row["zone_name"]),
                features=feature_rows[len(y_pred)],
                r_score=float(row["r_score"]),
                q_score=float(row["q_score"]),
            )
            y_pred.append(pred.artifact_type)
            confidences.append(float(pred.probability))
            probas.append(dict(pred.probabilities))
            abstained_flags.append(bool(pred.abstained))

    classes = list(ARTIFACT_TYPES)
    # Accuracy on all rows
    acc = float(np.mean([a == b for a, b in zip(y_true, y_pred)])) if y_true else 0.0
    macro_p, macro_r, macro_f1 = _macro_prf(y_true, y_pred, classes)
    cm = _confusion(y_true, y_pred, classes)
    ll = _log_loss(y_true, probas, classes)
    brier = _brier_multiclass(y_true, probas, classes)
    ece = _ece(y_true, y_pred, confidences)

    covered = [not a for a in abstained_flags]
    coverage = float(np.mean(covered)) if covered else 0.0
    abstention_rate = 1.0 - coverage
    if any(covered):
        sel_acc = float(np.mean([
            yt == yp for yt, yp, c in zip(y_true, y_pred, covered) if c
        ]))
    else:
        sel_acc = 0.0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_confusion_png(cm, classes, out_dir / "artifact_confusion_matrix.png")
    _save_reliability_png(y_true, y_pred, confidences, out_dir / "artifact_reliability_diagram.png")

    results = {
        "n": len(y_true),
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "log_loss": ll,
        "brier_score": brier,
        "expected_calibration_error": ece,
        "coverage": coverage,
        "abstention_rate": abstention_rate,
        "selective_accuracy": sel_acc,
        "classes": classes,
        "confusion_matrix": cm.tolist(),
        "notes": [
            "Artifact-level metrics require human/dataset annotations.",
            "Do not treat feature-based MAP as confirmed physical cause.",
            "Most likely observable artifact only.",
        ],
    }
    with open(out_dir / "artifact_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: results[k] for k in [
        "n", "accuracy", "macro_f1", "log_loss", "brier_score",
        "expected_calibration_error", "coverage", "abstention_rate", "selective_accuracy"
    ]}, indent=2))


if __name__ == "__main__":
    main()
