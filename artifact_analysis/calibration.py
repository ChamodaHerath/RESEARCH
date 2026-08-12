"""
calibration.py — optional learned calibration for artifact posteriors.

LEARNED / CALIBRATED COMPONENT
------------------------------
Fits an interpretable model when labeled artifact annotations exist:
  - Gaussian Naive Bayes (default; sklearn), or
  - Dirichlet-Categorical count updating (no sklearn required)

The SAEDF pipeline must still work in pure rule-based mode when no labels
are available. Calibration never invents ground-truth physical causes —
it only reweights artifact-class likelihoods from annotated examples.

Artifact-level evaluation requires human or dataset annotations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from artifact_analysis.artifact_ontology import ARTIFACT_TYPES, load_artifact_config
from artifact_analysis.artifact_inference import normalize_features


@dataclass
class ArtifactCalibrationModel:
    """
    Interpretable calibrator for artifact-type probabilities.

    Attributes:
        method: "gaussian_nb" | "dirichlet_counts"
        feature_names: ordered feature keys used at fit time
        class_names: ordered artifact labels
    """

    method: str = "gaussian_nb"
    feature_names: List[str] = field(default_factory=list)
    class_names: List[str] = field(default_factory=lambda: list(ARTIFACT_TYPES))
    # Gaussian NB params
    class_priors_: Dict[str, float] = field(default_factory=dict)
    means_: Dict[str, List[float]] = field(default_factory=dict)
    vars_: Dict[str, List[float]] = field(default_factory=dict)
    # Dirichlet-categorical counts: zone -> artifact -> count
    dirichlet_counts_: Dict[str, Dict[str, float]] = field(default_factory=dict)
    prior_strength_: float = 1.0
    fitted_: bool = False
    _sklearn_model: Any = field(default=None, repr=False)

    def fit(
        self,
        feature_rows: List[Dict[str, float]],
        labels: List[str],
        zone_names: List[str],
    ) -> None:
        """
        Fit calibrator on labeled zone feature rows.

        Args:
            feature_rows: list of raw feature dicts (same keys as extractor)
            labels: artifact type strings (must be in ARTIFACT_TYPES)
            zone_names: zone name per row (used by Dirichlet-count method)
        """
        if not (len(feature_rows) == len(labels) == len(zone_names)):
            raise ValueError("feature_rows, labels, zone_names must have equal length")
        if len(feature_rows) == 0:
            raise ValueError("Cannot fit on empty dataset")

        for lab in labels:
            if lab not in self.class_names:
                raise ValueError(f"Unknown artifact label: {lab}")

        # Stable feature order from union of keys (exclude counts)
        key_set = set()
        for row in feature_rows:
            key_set.update(row.keys())
        key_set.discard("zone_pixel_count")
        self.feature_names = sorted(key_set)

        if self.method == "dirichlet_counts":
            self._fit_dirichlet(labels, zone_names)
        else:
            self._fit_gaussian_nb(feature_rows, labels)
        self.fitted_ = True

    def _fit_dirichlet(self, labels: Sequence[str], zone_names: Sequence[str]) -> None:
        cfg = load_artifact_config()
        counts: Dict[str, Dict[str, float]] = {}
        for zone, lab in zip(zone_names, labels):
            if zone not in counts:
                # Start from weak prior concentrations
                counts[zone] = {
                    a: float(cfg.prior_vector(zone).get(a, self.prior_strength_))
                    for a in self.class_names
                }
            counts[zone][lab] = counts[zone].get(lab, self.prior_strength_) + 1.0
        self.dirichlet_counts_ = counts

    def _fit_gaussian_nb(
        self, feature_rows: Sequence[Dict[str, float]], labels: Sequence[str]
    ) -> None:
        X = self._rows_to_matrix(feature_rows)
        y = np.asarray(labels)

        try:
            from sklearn.naive_bayes import GaussianNB

            model = GaussianNB()
            model.fit(X, y)
            self._sklearn_model = model
            # Mirror params for JSON serialization / inspection
            for i, cls in enumerate(model.classes_):
                self.class_priors_[str(cls)] = float(model.class_prior_[i])
                self.means_[str(cls)] = [float(v) for v in model.theta_[i]]
                # sklearn >=1.0 uses var_; older used sigma_
                var = getattr(model, "var_", None)
                if var is None:
                    var = model.sigma_
                self.vars_[str(cls)] = [float(v) for v in var[i]]
            self.class_names = [str(c) for c in model.classes_]
        except ImportError:
            # Pure-numpy Gaussian NB fallback
            self._fit_gaussian_nb_numpy(X, y)

    def _fit_gaussian_nb_numpy(self, X: np.ndarray, y: np.ndarray) -> None:
        classes = sorted(set(y.tolist()))
        self.class_names = list(classes)
        n = len(y)
        for cls in classes:
            mask = y == cls
            self.class_priors_[cls] = float(mask.sum() / n)
            subset = X[mask]
            self.means_[cls] = subset.mean(axis=0).astype(float).tolist()
            self.vars_[cls] = (subset.var(axis=0) + 1e-6).astype(float).tolist()

    def _rows_to_matrix(self, rows: Sequence[Dict[str, float]]) -> np.ndarray:
        mats = []
        for row in rows:
            norm = normalize_features(row)
            mats.append([float(norm.get(k, 0.0)) for k in self.feature_names])
        return np.asarray(mats, dtype=np.float64)

    def predict_proba(
        self, features: Dict[str, float], zone_name: str
    ) -> Dict[str, float]:
        """
        Return calibrated P(artifact | features [, zone]).

        If not fitted, returns a uniform distribution (rule-based mode unchanged).
        """
        if not self.fitted_:
            n = len(ARTIFACT_TYPES)
            return {a: 1.0 / n for a in ARTIFACT_TYPES}

        if self.method == "dirichlet_counts":
            counts = self.dirichlet_counts_.get(zone_name)
            if counts is None:
                n = len(self.class_names)
                return {a: 1.0 / n for a in self.class_names}
            total = sum(counts.values()) + 1e-12
            return {a: float(counts.get(a, 0.0) / total) for a in self.class_names}

        # Gaussian NB
        if self._sklearn_model is not None:
            X = self._rows_to_matrix([features])
            proba = self._sklearn_model.predict_proba(X)[0]
            return {
                str(c): float(p)
                for c, p in zip(self._sklearn_model.classes_, proba)
            }

        # Pure-numpy Gaussian log-likelihood
        X = self._rows_to_matrix([features])[0]
        log_posts = []
        for cls in self.class_names:
            prior = max(self.class_priors_.get(cls, 1e-6), 1e-12)
            mean = np.asarray(self.means_[cls], dtype=np.float64)
            var = np.asarray(self.vars_[cls], dtype=np.float64)
            log_like = -0.5 * np.sum(np.log(2 * np.pi * var) + (X - mean) ** 2 / var)
            log_posts.append(np.log(prior) + log_like)
        log_posts_arr = np.asarray(log_posts, dtype=np.float64)
        log_posts_arr -= log_posts_arr.max()
        posts = np.exp(log_posts_arr)
        posts /= posts.sum() + 1e-12
        return {c: float(p) for c, p in zip(self.class_names, posts)}

    def save(self, path: str) -> None:
        """Save calibration parameters to JSON (interpretable, no pickle required)."""
        payload = {
            "method": self.method,
            "feature_names": self.feature_names,
            "class_names": self.class_names,
            "class_priors_": self.class_priors_,
            "means_": self.means_,
            "vars_": self.vars_,
            "dirichlet_counts_": self.dirichlet_counts_,
            "prior_strength_": self.prior_strength_,
            "fitted_": self.fitted_,
            "notes": [
                "LEARNED/CALIBRATED COMPONENT",
                "Requires labeled artifact annotations for meaningful use.",
                "Rule-based inference remains valid when this file is absent.",
            ],
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        # Optional joblib sidecar for sklearn model
        if self._sklearn_model is not None:
            try:
                import joblib

                joblib.dump(self._sklearn_model, str(out.with_suffix(".joblib")))
            except ImportError:
                pass

    @classmethod
    def load(cls, path: str) -> "ArtifactCalibrationModel":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        obj = cls(
            method=payload.get("method", "gaussian_nb"),
            feature_names=list(payload.get("feature_names", [])),
            class_names=list(payload.get("class_names", ARTIFACT_TYPES)),
            class_priors_=dict(payload.get("class_priors_", {})),
            means_=dict(payload.get("means_", {})),
            vars_=dict(payload.get("vars_", {})),
            dirichlet_counts_=dict(payload.get("dirichlet_counts_", {})),
            prior_strength_=float(payload.get("prior_strength_", 1.0)),
            fitted_=bool(payload.get("fitted_", False)),
        )
        joblib_path = Path(path).with_suffix(".joblib")
        if joblib_path.is_file():
            try:
                import joblib

                obj._sklearn_model = joblib.load(str(joblib_path))
            except ImportError:
                pass
        return obj
