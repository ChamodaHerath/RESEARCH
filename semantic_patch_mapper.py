# semantic_patch_mapper.py

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
except ImportError:
    torch = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_SIZE  = 224
GRID_SIZE   = 14
PATCH_SIZE  = 16
NUM_PATCHES = GRID_SIZE * GRID_SIZE   # 196

# 9-zone definitions — must mirror preprocess_landmarks.py SEMANTIC_GROUPS
ZONE_LANDMARK_INDICES: Dict[str, np.ndarray] = {
    "left_eye":    np.array([36,37,38,39,40,41],             dtype=np.int64),
    "right_eye":   np.array([42,43,44,45,46,47],             dtype=np.int64),
    "nose_bridge": np.array([27,28,29,30],                   dtype=np.int64),
    "nose_tip":    np.array([31,32,33,34,35],                dtype=np.int64),
    "left_cheek":  np.array([1,2,3,4,5,48,49],              dtype=np.int64),
    "right_cheek": np.array([11,12,13,14,15,54,55],         dtype=np.int64),
    "mouth":       np.arange(48, 68,                         dtype=np.int64),
    "forehead":    np.arange(17, 27,                         dtype=np.int64),
    "chin_jaw":    np.array([0,1,2,14,15,16,7,8,9],         dtype=np.int64),
}
ZONE_NAMES: List[str] = list(ZONE_LANDMARK_INDICES.keys())
NUM_ZONES: int = len(ZONE_NAMES)

# ---------------------------------------------------------------------------
# Semantic concept ontology  (Module 3 — primary novel contribution)
# ---------------------------------------------------------------------------
# For each zone, a ranked list of artifact types known from the manipulation
# literature (FaceForensics++, DeepFaceLab, SimSwap failure modes).
# Index 0 = most common artifact for that zone.

ARTIFACT_ONTOLOGY: Dict[str, List[str]] = {
    "mouth": [
        "blending boundary artefact",
        "lip sync inconsistency",
        "GAN upsampling ripple",
        "unnatural motion boundary",
        "over-smoothed lip texture",
    ],
    "left_eye": [
        "iris texture inconsistency",
        "specular highlight mismatch",
        "blink timing anomaly",
        "colour asymmetry between eyes",
        "eyelid boundary blending seam",
    ],
    "right_eye": [
        "iris texture inconsistency",
        "specular highlight mismatch",
        "blink timing anomaly",
        "colour asymmetry between eyes",
        "eyelid boundary blending seam",
    ],
    "nose_bridge": [
        "geometric distortion",
        "texture seam at bridge",
        "lighting gradient mismatch",
    ],
    "nose_tip": [
        "texture blending discontinuity",
        "over-sharpened tip boundary",
        "pore pattern erasure",
    ],
    "left_cheek": [
        "skin smoothing over-suppression",
        "pore texture erasure",
        "compression boundary artefact",
        "colour tone mismatch",
    ],
    "right_cheek": [
        "skin smoothing over-suppression",
        "pore texture erasure",
        "compression boundary artefact",
        "colour tone mismatch",
    ],
    "forehead": [
        "hair boundary artefact",
        "skin tone inconsistency",
        "forehead wrinkle over-smoothing",
    ],
    "chin_jaw": [
        "jawline blending seam",
        "shape warp distortion",
        "chin shadow inconsistency",
    ],
}

SEVERITY_THRESHOLDS = {"high": 0.75, "medium": 0.50, "low": 0.25}
SEVERITY_ADJECTIVES = {"high": "High-confidence", "medium": "Moderate", "low": "Low-level"}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ConceptVector:
    """
    Structured semantic descriptor for one flagged facial zone.
    C(z) = ⟨zone, artifact_type, severity, confidence⟩
    """
    zone: str
    artifact_type: str
    severity: str                           # "high" | "medium" | "low"
    confidence: float                       # R(z) score in [0, 1]
    R_score: float                          # raw fused score, for downstream sorting
    artifact_probs: Optional[np.ndarray] = field(default=None, repr=False)
    # artifact_probs: (K,) posterior probability vector over artifact types;
    # None when the Dirichlet prior has not been fitted (legacy path).
    R_ci_lower: Optional[float] = field(default=None, repr=False)
    R_ci_upper: Optional[float] = field(default=None, repr=False)
    # R_ci_lower / R_ci_upper: 95 % credible interval from β posterior
    # uncertainty propagation, on the SAME scale as `confidence` above --
    # i.e. on R(z) when zone_prior_score is None, or re-expressed on Q(z)
    # (via the zone prior's own monotonic transform) when a zone prior is
    # active. This keeps the reported interval consistent with the number it
    # brackets; see predict_label_and_explanation(). None when beta_std is
    # unavailable.
    zone_prior_score: Optional[float] = field(default=None, repr=False)
    # zone_prior_score: Q(z) — posterior zone anomaly probability from the
    # manipulation-type-conditioned Bayesian zone prior (Option 2).
    # Q(z) = (α + R(z)·w) / (α + β + w)  where (α,β) are literature priors.
    # None when ZonePriorModel is not loaded.
    flag_reason: Optional[str] = field(default=None, repr=False)
    # flag_reason: "calibrated_threshold" when score > per-zone cutoff.

    def to_dict(self) -> Dict:
        conf_pct = round(self.confidence * 100, 1)
        if self.R_ci_lower is not None and self.R_ci_upper is not None:
            conf_str = (
                f"{conf_pct}% "
                f"[{round(self.R_ci_lower * 100, 1)}%\u2013"
                f"{round(self.R_ci_upper * 100, 1)}%]"
            )
        else:
            conf_str = f"{conf_pct}%"

        d: Dict = {
            "zone": self.zone,
            "artifact_type": self.artifact_type,
            "severity": self.severity,
            "confidence_pct": conf_str,
            "R_score": round(self.R_score, 4),
        }
        if self.zone_prior_score is not None:
            d["zone_anomaly_posterior"] = round(self.zone_prior_score, 4)
        if self.flag_reason is not None:
            d["flag_reason"] = self.flag_reason
        if self.R_ci_lower is not None:
            d["R_ci"] = {
                "lower": round(self.R_ci_lower, 4),
                "upper": round(self.R_ci_upper, 4),
            }
        if self.artifact_probs is not None:
            artifact_list = ARTIFACT_ONTOLOGY.get(self.zone, [])
            d["artifact_distribution"] = {
                art: round(float(p), 4)
                for art, p in zip(artifact_list, self.artifact_probs)
            }
        return d


@dataclass
class ExplanationOutput:
    """Complete explanation output for one face frame."""
    label: str                              # "Fake" | "Real"
    fake_probability: float                 # detection confidence
    concept_vectors: List[ConceptVector]    # one per flagged zone
    nl_explanations: List[str]              # verified NL sentence per concept vector
    faithfulness_passed: List[bool]         # True if verifier passed for each sentence
    R_scores: np.ndarray                    # shape (9,) — fused scores for all zones
    # Bayesian zone prior fields (Option 2) — None when ZonePriorModel not loaded
    inferred_method: Optional[str] = field(default=None, repr=False)
    method_probabilities: Optional[Dict[str, float]] = field(default=None, repr=False)
    Q_scores: Optional[np.ndarray] = field(default=None, repr=False)
    # Q_scores: (9,) marginal posterior zone anomaly probabilities Q(z)
    # Causal correctness gate (occlusion drop) — None until filter_explanations_by_causal_correctness runs
    causal_correctness_passed: Optional[List[bool]] = field(default=None, repr=False)
    causal_drop_by_zone: Optional[Dict[str, float]] = field(default=None, repr=False)
    causal_min_drop: Optional[float] = field(default=None, repr=False)

    def to_dict(self) -> Dict:
        d: Dict = {
            "label": self.label,
            "fake_probability": round(self.fake_probability, 4),
            "flagged_zones": [cv.to_dict() for cv in self.concept_vectors],
            "nl_explanations": self.nl_explanations,
            "faithfulness_passed": self.faithfulness_passed,
            "R_scores": {ZONE_NAMES[i]: round(float(self.R_scores[i]), 4)
                         for i in range(NUM_ZONES)},
        }
        if self.inferred_method is not None:
            d["inferred_manipulation_method"] = self.inferred_method
            d["method_probabilities"] = self.method_probabilities
        if self.Q_scores is not None:
            d["Q_scores"] = {ZONE_NAMES[i]: round(float(self.Q_scores[i]), 4)
                             for i in range(NUM_ZONES)}
        if self.causal_correctness_passed is not None:
            d["causal_correctness_passed"] = self.causal_correctness_passed
            d["causal_drop_by_zone"] = self.causal_drop_by_zone
            d["causal_min_drop"] = self.causal_min_drop
        return d


# ---------------------------------------------------------------------------
# Dual-head output schema  (Stage 4 — joint inference report)
# ---------------------------------------------------------------------------

@dataclass
class ClassificationHead:
    """
    Classification result from the detection model.

    This is the first of two output heads in the final ForensicReport.
    It contains the raw classifier output and a human-readable tier label
    derived from the fake_probability score.
    """
    label: str             # "Fake" | "Real"
    fake_probability: float
    confidence_tier: str   # "high" (>0.75) | "medium" (>0.50) | "uncertain" (≤0.50)
    model_name: str

    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "fake_probability": round(self.fake_probability, 4),
            "confidence_tier": self.confidence_tier,
            "model_name": self.model_name,
        }

    @staticmethod
    def _tier(p: float) -> str:
        if p > 0.75:
            return "high"
        if p > 0.50:
            return "medium"
        return "uncertain"

    @classmethod
    def from_probability(
        cls,
        fake_probability: float,
        model_name: str,
    ) -> "ClassificationHead":
        label = "Fake" if fake_probability >= 0.5 else "Real"
        return cls(
            label=label,
            fake_probability=fake_probability,
            confidence_tier=cls._tier(fake_probability),
            model_name=model_name,
        )


@dataclass
class ForensicReport:
    """
    Dual-head forensic analysis report for one face image.

    Head 1 — classification : ClassificationHead
        Raw detection result: label, fake_probability, confidence tier.

    Head 2 — explanation    : ExplanationOutput
        Semantic attribution: flagged facial zones, artifact types, NL sentences,
        Bayesian credible intervals on each zone's anomaly score.

    The two heads are produced by a single model forward pass via SAEDFPipeline
    and are guaranteed to be consistent (same model, same image).
    """
    classification: ClassificationHead
    explanation: ExplanationOutput
    metadata: Dict                          # image_path, timestamp, calibration info

    def to_dict(self) -> Dict:
        return {
            "metadata":       self.metadata,
            "classification": self.classification.to_dict(),
            "explanation":    self.explanation.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        import json as _json
        return _json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def print_summary(self) -> None:
        """Human-readable console summary of the dual-head result."""
        c = self.classification
        e = self.explanation
        print("\n" + "=" * 62)
        print(f"  FORENSIC REPORT")
        print(f"  Image : {self.metadata.get('image_path', 'N/A')}")
        print("=" * 62)

        print(f"\n[HEAD 1 — CLASSIFICATION]")
        print(f"  Label            : {c.label}")
        print(f"  Fake probability : {c.fake_probability:.4f}  ({c.confidence_tier})")
        print(f"  Model            : {c.model_name}")

        print(f"\n[HEAD 2 — SEMANTIC EXPLANATION]")
        if e.inferred_method is not None:
            top_p = max(e.method_probabilities.values()) if e.method_probabilities else 0.0
            print(f"  Inferred method  : {e.inferred_method}  (p={top_p:.3f})")
            if e.method_probabilities:
                for m, p in sorted(e.method_probabilities.items(),
                                   key=lambda kv: kv[1], reverse=True):
                    bar = "▪" * int(p * 20)
                    print(f"    {m:20s} {p:.3f}  {bar}")
        if not e.concept_vectors:
            print("  No zones above the severity threshold.")
        else:
            print()
            for sentence in e.nl_explanations:
                print(f"  • {sentence}")

        print(f"\n  Zone scores  [R(z) = fusion,  Q(z) = Bayesian posterior]:")
        e_dict = e.to_dict()
        q_dict = e_dict.get("Q_scores", {})
        for zone, r_score in e_dict["R_scores"].items():
            q_score = q_dict.get(zone, None)
            bar = "█" * int(r_score * 20)
            q_str = f"  Q={q_score:.3f}" if q_score is not None else ""
            print(f"    {zone:14s} R={r_score:.3f}{q_str}  {bar}")
        print("=" * 62 + "\n")


# ---------------------------------------------------------------------------
# Zone-patch matrix  (continuous overlap-fraction version)
# ---------------------------------------------------------------------------

def _validate_grid(image_size: int, grid_size: int, patch_size: int) -> None:
    if image_size != grid_size * patch_size:
        raise ValueError(
            f"Inconsistent grid: image_size={image_size}, "
            f"grid_size={grid_size}, patch_size={patch_size}."
        )


def _to_landmark_array(landmarks_68: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(landmarks_68, dtype=np.float32)
    if arr.shape != (68, 2):
        raise ValueError(f"Expected landmarks shape (68, 2), got {arr.shape}.")
    return arr


def _clip_landmarks(landmarks_xy: np.ndarray, image_size: int) -> np.ndarray:
    max_val = np.nextafter(np.float32(image_size), np.float32(-np.inf))
    return np.clip(landmarks_xy, 0.0, max_val)


def build_overlap_fraction_matrix(
    landmarks_68: Sequence[Sequence[float]],
    image_size: int = IMAGE_SIZE,
    grid_size: int = GRID_SIZE,
    patch_size: int = PATCH_SIZE,
    as_torch: bool = False,
    torch_dtype=None,
) -> Union[np.ndarray, "torch.Tensor"]:
    """
    Build zone-patch membership matrix M ∈ ℝ^(196 × 9) with overlap fractions.

    M[i, z] = fraction of patch i's area that lies inside zone z's convex hull.
    This is continuous (not binary), so a patch spanning two zones gets partial
    credit in each.  Values per row need not sum to 1.

    This is the M matrix used in:
        A(z) = Σ_i M[i,z]·α[i] / Σ_i M[i,z]
        S(z) = Σ_i M[i,z]·φ[i]

    Args:
        landmarks_68: 68 (x,y) points scaled to [0, image_size).
        image_size, grid_size, patch_size: ViT grid constants.
        as_torch: if True, return a torch.Tensor.
        torch_dtype: dtype for torch output (default float32).

    Returns:
        M: shape (196, 9), dtype float32.
    """
    import cv2 as _cv2  # local import so the module loads without cv2 for testing

    _validate_grid(image_size, grid_size, patch_size)
    pts = _clip_landmarks(_to_landmark_array(landmarks_68), image_size)
    M = np.zeros((NUM_PATCHES, NUM_ZONES), dtype=np.float32)
    patch_area = float(patch_size * patch_size)

    for z_idx, (zone_name, lm_ids) in enumerate(ZONE_LANDMARK_INDICES.items()):
        zone_pts = pts[lm_ids]
        if len(zone_pts) < 3:
            continue
        hull = _cv2.convexHull(zone_pts)
        if hull is None or len(hull) < 3:
            continue
        zone_mask = np.zeros((image_size, image_size), dtype=np.uint8)
        _cv2.fillConvexPoly(zone_mask, hull.astype(np.int32), 1)

        # Bounding box optimisation: only test patches in zone's bbox
        bx1 = max(0, int(zone_pts[:, 0].min()) // patch_size)
        bx2 = min(grid_size - 1, int(zone_pts[:, 0].max()) // patch_size)
        by1 = max(0, int(zone_pts[:, 1].min()) // patch_size)
        by2 = min(grid_size - 1, int(zone_pts[:, 1].max()) // patch_size)

        for row in range(by1, by2 + 1):
            for col in range(bx1, bx2 + 1):
                py1, py2 = row * patch_size, (row + 1) * patch_size
                px1, px2 = col * patch_size, (col + 1) * patch_size
                overlap = int(zone_mask[py1:py2, px1:px2].sum())
                if overlap > 0:
                    M[row * grid_size + col, z_idx] = overlap / patch_area

    if as_torch:
        if torch is None:
            raise ImportError("PyTorch is not installed.")
        dtype = torch_dtype if torch_dtype is not None else torch.float32
        return torch.tensor(M, dtype=dtype)

    return M


def build_binary_mask_matrix(
    landmarks_68: Sequence[Sequence[float]],
    as_torch: bool = False,
    torch_dtype=None,
) -> Union[np.ndarray, "torch.Tensor"]:
    """
    Binary version of the zone-patch matrix (kept for ablation baseline).
    M[i, z] = 1 if ANY landmark from zone z maps to patch i, else 0.
    """
    pts = _clip_landmarks(_to_landmark_array(landmarks_68), IMAGE_SIZE)
    patch_xy = np.floor(pts / float(PATCH_SIZE)).astype(np.int64)
    patch_xy = np.clip(patch_xy, 0, GRID_SIZE - 1)
    patch_indices = patch_xy[:, 1] * GRID_SIZE + patch_xy[:, 0]

    M = np.zeros((NUM_PATCHES, NUM_ZONES), dtype=np.float32)
    for z_idx, (_, lm_ids) in enumerate(ZONE_LANDMARK_INDICES.items()):
        unique_patches = np.unique(patch_indices[lm_ids])
        M[unique_patches, z_idx] = 1.0

    if as_torch:
        if torch is None:
            raise ImportError("PyTorch is not installed.")
        dtype = torch_dtype if torch_dtype is not None else torch.float32
        return torch.tensor(M, dtype=dtype)

    return M


# Legacy name kept for backward compatibility
def build_semantic_mask_matrix(
    landmarks_68: Sequence[Sequence[float]],
    as_torch: bool = False,
    torch_dtype=None,
) -> Union[np.ndarray, "torch.Tensor"]:
    """Alias for build_overlap_fraction_matrix (new default). See that function."""
    return build_overlap_fraction_matrix(landmarks_68, as_torch=as_torch, torch_dtype=torch_dtype)


# ---------------------------------------------------------------------------
# Dual-score fusion  (Module 2 — R(z) computation)
# ---------------------------------------------------------------------------

def compute_attention_zone_scores(
    alpha: np.ndarray,
    M: np.ndarray,
) -> np.ndarray:
    """
    A(z) = Σ_i M[i,z]·α[i] / (Σ_i M[i,z] + ε)  -- area-normalised, raw scale.

    Per-frame min-max normalisation across zones was tried and rejected: it compares
    zones to each other within one frame, but zones have structurally different
    baseline magnitudes (e.g. chin_jaw is always higher than left_eye regardless of
    label, due to ViT positional-embedding boundary effects). The correct comparison
    is each zone against ITS OWN population baseline -- see normalize_zone_scores().

    Args:
        alpha : shape (196,) — CLS attention weights.
        M     : shape (196, 9) — overlap-fraction matrix.

    Returns:
        A : shape (9,) — area-normalised, raw scale (typically 1e-3 to 1e-2).
    """
    zone_area = M.sum(axis=0) + 1e-8
    return ((M * alpha[:, None]).sum(axis=0) / zone_area).astype(np.float32)


def compute_shap_zone_scores(
    phi: np.ndarray,
    M: np.ndarray,
) -> np.ndarray:
    """
    S(z) = Σ_i M[i,z]·φ[i] / (Σ_i M[i,z] + ε)  -- area-normalised, like A(z).

    Previously this was an unnormalised sum, which made large zones (chin_jaw,
    ~72 patches) always dominate small zones (left_eye, ~0.75 patches) by sheer
    patch count, independent of per-patch intensity. Area normalisation (mean per
    patch) removes this bias and puts S(z) on the same footing as A(z).

    Args:
        phi : shape (196,) — patch gradient/SHAP values (absolute).
        M   : shape (196, 9) — overlap-fraction matrix.

    Returns:
        S : shape (9,) — area-normalised, NOT min-max normalised here.
            Population-level normalisation happens in normalize_zone_scores().
    """
    zone_area = M.sum(axis=0) + 1e-8
    return ((M * phi[:, None]).sum(axis=0) / zone_area).astype(np.float32)


def fuse_scores(
    A: np.ndarray,
    S: np.ndarray,
    beta: float = 0.5,
) -> np.ndarray:
    """
    R(z) = β·A(z) + (1−β)·S(z)

    Args:
        A    : shape (9,) — attention scores.
        S    : shape (9,) — SHAP/gradient scores.
        beta : weighting in [0, 1]. Default 0.5 (equal weight).
                Calibrate on the FF++ validation set — see fusion calibration script.

    Returns:
        R : shape (9,), values in [0, 1].
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1].")
    return (beta * A + (1.0 - beta) * S).astype(np.float32)

def compute_population_zone_stats(
    scores_all: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute per-zone normalisation statistics from a reference set of frames.

    If labels is provided, statistics are computed from REAL frames only
    (label==0) -- this defines "normal" per-zone baseline, so that elevated
    scores in fake frames stand out as deviations from normal, which is the
    correct framing for anomaly detection.

    If labels is None, statistics are computed across all frames.

    Args:
        scores_all : shape (N, 9) — A_raw or S_raw across N reference frames.
        labels     : shape (N,) — optional, 1=fake, 0=real.

    Returns:
        Dict with "mean" and "std", each shape (9,).
    """
    if labels is not None:
        ref = scores_all[labels == 0]
        if len(ref) == 0:
            ref = scores_all  # fallback if no real frames available
    else:
        ref = scores_all

    return {
        "mean": ref.mean(axis=0).astype(np.float32),
        "std": (ref.std(axis=0) + 1e-8).astype(np.float32),
    }


def normalize_zone_scores(
    scores: np.ndarray,
    stats: Dict[str, np.ndarray],
    clip_range: Tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """
    Normalise per-zone scores using population statistics (z-score, then squash to [0,1]).

    z = (score - mean) / std
    normalised = sigmoid(z)  -- maps z-score to [0,1], with 0.5 = population mean

    This replaces per-frame min-max normalisation. A zone's score is now interpreted
    as "how many standard deviations above the NORMAL baseline for THIS zone",
    which is comparable across zones with different structural magnitudes.

    Args:
        scores     : shape (9,) — raw area-normalised A or S for one frame.
        stats      : Dict with "mean" and "std", shape (9,) each, from
                     compute_population_zone_stats().
        clip_range : output range, default [0, 1].

    Returns:
        normalised : shape (9,), values in clip_range.
    """
    z = (scores - stats["mean"]) / stats["std"]
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))
    lo, hi = clip_range
    return (lo + (hi - lo) * sigmoid).astype(np.float32)

def assign_severity(R_score: float) -> Optional[str]:
    """
    Map a fused R_score to a severity tier.

    Returns "high" / "medium" / "low" / None (below threshold → no explanation).
    """
    if R_score > SEVERITY_THRESHOLDS["high"]:
        return "high"
    elif R_score > SEVERITY_THRESHOLDS["medium"]:
        return "medium"
    elif R_score > SEVERITY_THRESHOLDS["low"]:
        return "low"
    return None


def get_flagged_zones(
    R: np.ndarray,
    min_severity: str = "low",
    zone_thresholds: Optional[Mapping[str, float]] = None,
    max_zones: Optional[int] = None,
) -> List[Dict]:
    """
    Return zones above the reporting threshold, sorted by score descending.

    When ``zone_thresholds`` are provided (calibrated on Q(z) in
    fusion_config.json), those alone decide flagging — flag only when
    ``score > threshold`` (strict). Do NOT also require the global severity
    floor. Combining both was a double-gate bug that silently dropped zones
    whose Q(z) cleared the calibrated threshold but sat near/under
    SEVERITY_THRESHOLDS["low"] after assign_severity.

    Without per-zone thresholds, fall back to the global severity gate
    (``min_severity``).

    Args:
        R            : shape (9,) — scores used for flagging (Q(z) or R(z)).
        min_severity : minimum severity when zone_thresholds is absent.
        zone_thresholds : optional per-zone calibrated cutoffs.
        max_zones    : optional cap on how many zones to return.

    Returns:
        List of dicts: [{"zone": ..., "R_score": ..., "severity": ...}, ...]
    """
    flagged = []
    use_zone_thresholds = zone_thresholds is not None and len(zone_thresholds) > 0
    for i, zone in enumerate(ZONE_NAMES):
        score = float(R[i])
        if use_zone_thresholds and zone in zone_thresholds:
            # Strict: flag only when score is higher than the calibrated bar.
            # Round to 4 decimals (same as API display) so float32 noise cannot
            # turn an exact threshold hit into a false positive.
            thr = float(zone_thresholds[zone])
            if round(score, 4) <= round(thr, 4):
                continue
            sev = assign_severity(score) or "low"
        else:
            sev = assign_severity(score)
            if sev is None:
                continue
            # Enforce min_severity ordering: high > medium > low
            order = {"low": 0, "medium": 1, "high": 2}
            if order.get(sev, -1) < order.get(min_severity, 0):
                continue
        flagged.append({"zone": zone, "R_score": score, "severity": sev})
    flagged = sorted(flagged, key=lambda d: d["R_score"], reverse=True)
    return flagged[:max_zones] if max_zones is not None else flagged


# ---------------------------------------------------------------------------
# Concept ontology  (Module 3)
# ---------------------------------------------------------------------------

def build_concept_vector(
    zone: str,
    R_score: float,
    artifact_rank: int = 0,
    artifact_posterior: Optional[np.ndarray] = None,
    R_ci_lower: Optional[float] = None,
    R_ci_upper: Optional[float] = None,
    zone_prior_score: Optional[float] = None,
) -> ConceptVector:
    """
    Build a concept vector C(z) = ⟨zone, artifact_type, severity, confidence⟩.

    Args:
        zone              : zone name (must be in ZONE_NAMES).
        R_score           : fused anomaly score in [0, 1] (MAP β estimate).
        artifact_rank     : fallback index into ARTIFACT_ONTOLOGY[zone] used
                            when artifact_posterior is None (0 = most common).
        artifact_posterior: (K,) posterior probability vector from
                            DirichletArtifactPrior.  When provided, the MAP
                            artifact type (argmax) is selected and the full
                            distribution is stored on the ConceptVector.
        R_ci_lower        : lower bound of 95 % credible interval on R(z).
        R_ci_upper        : upper bound of 95 % credible interval on R(z).

    Returns:
        ConceptVector
    """
    if zone not in ARTIFACT_ONTOLOGY:
        raise KeyError(f"Zone '{zone}' not in ARTIFACT_ONTOLOGY.")
    sev = assign_severity(R_score)
    if sev is None:
        sev = "low"
    artifact_list = ARTIFACT_ONTOLOGY[zone]

    if artifact_posterior is not None:
        probs        = np.asarray(artifact_posterior, dtype=np.float64)
        probs       /= probs.sum()
        artifact_idx = int(np.argmax(probs))
    else:
        probs        = None
        artifact_idx = min(artifact_rank, len(artifact_list) - 1)

    # When zone_prior_score is available, use Q(z) for confidence and severity
    effective_score = zone_prior_score if zone_prior_score is not None else R_score
    sev_effective   = assign_severity(effective_score) or "low"

    return ConceptVector(
        zone=zone,
        artifact_type=artifact_list[artifact_idx],
        severity=sev_effective,
        confidence=round(float(effective_score), 4),
        R_score=float(R_score),
        artifact_probs=probs,
        R_ci_lower=float(R_ci_lower) if R_ci_lower is not None else None,
        R_ci_upper=float(R_ci_upper) if R_ci_upper is not None else None,
        zone_prior_score=float(zone_prior_score) if zone_prior_score is not None else None,
        flag_reason=None,
    )


def build_concept_vectors_from_R(
    R: np.ndarray,
    min_severity: str = "medium",
    artifact_prior=None,
    ci_lower: Optional[np.ndarray] = None,
    ci_upper: Optional[np.ndarray] = None,
    Q_scores: Optional[Dict] = None,
    zone_thresholds: Optional[Mapping[str, float]] = None,
    max_zones: Optional[int] = None,
) -> List[ConceptVector]:
    """
    Build concept vectors for all zones above min_severity threshold.

    Args:
        R             : shape (9,) — fused anomaly scores (MAP β estimate).
        min_severity  : only generate concept vectors for this severity or above.
        artifact_prior: optional DirichletArtifactPrior instance.  When
                        provided, artifact types are selected from the Bayesian
                        posterior rather than from the fixed rank-0 default.
        ci_lower      : shape (9,) — lower bounds of 95 % credible intervals
                        on R(z), from compute_R_credible_interval().
        ci_upper      : shape (9,) — upper bounds of 95 % credible intervals.

    Returns:
        List of ConceptVector, sorted by R_score descending.
    """
    # Zone flagging uses Q_scores (zone posteriors) when available, else R
    score_for_flagging = R
    if Q_scores is not None:
        q_array = np.array(
            [Q_scores.get(z, float(R[i])) for i, z in enumerate(ZONE_NAMES)],
            dtype=np.float32,
        )
        score_for_flagging = q_array

    flagged = get_flagged_zones(
        score_for_flagging,
        min_severity=min_severity,
        zone_thresholds=zone_thresholds,
        max_zones=max_zones,
    )
    cvs: List[ConceptVector] = []
    for f in flagged:
        zone, eff_score = f["zone"], f["R_score"]
        z_idx = ZONE_NAMES.index(zone)
        R_score_raw = float(R[z_idx])

        if artifact_prior is not None:
            _, probs = artifact_prior.get_map_artifact(zone, R_score_raw)
        else:
            probs = None

        zps = Q_scores.get(zone) if Q_scores is not None else None

        cv = build_concept_vector(
            zone,
            R_score_raw,
            artifact_posterior=probs,
            R_ci_lower=float(ci_lower[z_idx]) if ci_lower is not None else None,
            R_ci_upper=float(ci_upper[z_idx]) if ci_upper is not None else None,
            zone_prior_score=float(zps) if zps is not None else None,
        )
        cv.flag_reason = "calibrated_threshold"
        cvs.append(cv)
    return cvs


# ---------------------------------------------------------------------------
# NL template grammar  (Module 4a — deterministic, auditable)
# ---------------------------------------------------------------------------

def generate_nl_explanation(cv: ConceptVector) -> str:
    """
    Deterministic template grammar: maps one ConceptVector to one NL sentence.

    Template:
        "{severity_adj} {artifact_type} detected in the {zone} region
         (confidence: {conf_pct}%)."

    Every word is traceable to a field in cv.  The faithfulness verifier
    (verify_faithfulness) checks this after generation.

    Args:
        cv: ConceptVector produced by build_concept_vector().

    Returns:
        A single verified NL sentence.
    """
    adj        = SEVERITY_ADJECTIVES[cv.severity]
    zone_label = cv.zone.replace("_", " ")
    conf_pct   = round(cv.confidence * 100, 1)

    if cv.R_ci_lower is not None and cv.R_ci_upper is not None:
        ci_str = (
            f" [{round(cv.R_ci_lower * 100, 1)}\u2013"
            f"{round(cv.R_ci_upper * 100, 1)}%]"
        )
    else:
        ci_str = ""

    sentence = (
        f"{adj} {cv.artifact_type} detected in the {zone_label} region "
        f"(confidence: {conf_pct}%{ci_str})."
    )
    return sentence


# ---------------------------------------------------------------------------
# Faithfulness verifier  (Module 4b)
# ---------------------------------------------------------------------------

# Numeric claims embedded in generate_nl_explanation()'s output. Kept as module
# constants (rather than inlined regexes) so verify_faithfulness and any future
# caller parse claims the same way the generator writes them.
_CONFIDENCE_PATTERN = re.compile(r"confidence:\s*([\d.]+)\s*%")
_CI_PATTERN = re.compile(r"\[\s*([\d.]+)\s*[\u2013-]\s*([\d.]+)\s*%\s*\]")
_NUMERIC_TOLERANCE = 0.05  # percentage points; generation always rounds to 1dp


def verify_faithfulness(sentence: str, cv: ConceptVector) -> bool:
    """
    Verify that every claim in the NL sentence is traceable to the concept vector.

    Checks:
      1. Zone name present in sentence.
      2. Artifact type (or its first noun) present in sentence.
      3. A recognised severity adjective is present.
      4. If a "confidence: X%" figure is stated, X must match cv.confidence
         (rounded to 1dp) within a small numeric tolerance -- catches a
         sentence that names the right zone/artifact but reports the wrong
         number, which the original zone/artifact/severity-only checks could
         not detect.
      5. Credible-interval bounds are consistent both ways: if cv carries a
         95% CI, the sentence must state it (and match it); if cv carries no
         CI, the sentence must not claim one either.

    Args:
        sentence: NL explanation string.
        cv      : ConceptVector used to generate the sentence.

    Returns:
        True if all checks pass; False triggers fallback to template.
    """
    sentence_lower = sentence.lower()
    zone_label = cv.zone.replace("_", " ").lower()

    zone_ok     = zone_label in sentence_lower
    artifact_ok = cv.artifact_type.lower() in sentence_lower
    severity_ok = any(
        adj.lower() in sentence_lower
        for adj in SEVERITY_ADJECTIVES.values()
    )

    conf_ok = True
    conf_match = _CONFIDENCE_PATTERN.search(sentence)
    if conf_match is not None:
        stated_conf = float(conf_match.group(1))
        expected_conf = round(cv.confidence * 100, 1)
        conf_ok = abs(stated_conf - expected_conf) <= _NUMERIC_TOLERANCE

    has_cv_ci = cv.R_ci_lower is not None and cv.R_ci_upper is not None
    ci_match = _CI_PATTERN.search(sentence)
    if has_cv_ci:
        if ci_match is None:
            ci_ok = False  # cv has a credible interval but the sentence omits it
        else:
            expected_lower = round(cv.R_ci_lower * 100, 1)
            expected_upper = round(cv.R_ci_upper * 100, 1)
            ci_ok = (
                abs(float(ci_match.group(1)) - expected_lower) <= _NUMERIC_TOLERANCE
                and abs(float(ci_match.group(2)) - expected_upper) <= _NUMERIC_TOLERANCE
            )
    else:
        ci_ok = ci_match is None  # sentence must not invent a CI the cv doesn't have

    return zone_ok and artifact_ok and severity_ok and conf_ok and ci_ok


def generate_verified_explanation(cv: ConceptVector) -> Tuple[str, bool]:
    """
    Generate a verified NL explanation.

    Returns:
        (sentence, faithfulness_passed): sentence is always a valid, fully
        faithful string built from cv's own fields. If the primary template
        output somehow fails verification (e.g. a corrupted/overwritten
        `cv.artifact_type` upstream, or a future template change that drifts
        from cv), a minimal fallback sentence is reconstructed directly from
        cv and re-verified -- `passed` always reflects whichever sentence is
        actually returned, not just the primary attempt.
    """
    sentence = generate_nl_explanation(cv)
    passed   = verify_faithfulness(sentence, cv)

    if not passed:
        adj        = SEVERITY_ADJECTIVES.get(cv.severity, cv.severity)
        zone_label = cv.zone.replace("_", " ")
        conf_pct   = round(cv.confidence * 100, 1)
        if cv.R_ci_lower is not None and cv.R_ci_upper is not None:
            ci_str = (
                f" [{round(cv.R_ci_lower * 100, 1)}\u2013"
                f"{round(cv.R_ci_upper * 100, 1)}%]"
            )
        else:
            ci_str = ""
        sentence = (
            f"{adj} {cv.artifact_type} anomaly detected in the {zone_label} region "
            f"(confidence: {conf_pct}%{ci_str})."
        )
        # Re-verify the fallback itself -- it is built straight from cv's
        # fields so it should always pass, but this guarantees `passed`
        # never lies about the sentence actually being returned.
        passed = verify_faithfulness(sentence, cv)

    return sentence, passed


# ---------------------------------------------------------------------------
# Causal correctness gate — filter claims before the final report
# ---------------------------------------------------------------------------

DEFAULT_CAUSAL_MIN_DROP = 0.02


def filter_explanations_by_causal_correctness(
    explanation: ExplanationOutput,
    image_bgr: np.ndarray,
    landmarks_68: Sequence[Sequence[float]],
    fake_prob_fn,
    min_drop: float = DEFAULT_CAUSAL_MIN_DROP,
) -> ExplanationOutput:
    """
    Keep only NL claims whose zones causally drive the fake score.

    After the textual faithfulness verifier, each remaining flagged zone is
    occluded and the classifier is re-run. A claim is retained only when
    P(fake) drops by at least ``min_drop``. Claims that fail are removed from
    ``concept_vectors`` / ``nl_explanations`` so they never enter the final
    forensic report.

    This is causal sensitivity (does this zone affect the detector?), not
    pixel-mask ground truth. Mask GT remains an offline evaluation path.

    Args:
        explanation  : output of predict_label_and_explanation / explainer.predict
        image_bgr    : (224, 224, 3) face crop used for the claim
        landmarks_68 : landmarks in the same 224 space
        fake_prob_fn : callable(List[np.ndarray]) -> List[float]
        min_drop     : minimum baseline−occluded drop to keep a claim

    Returns:
        A new ExplanationOutput with only causally-supported claims retained.
        Classification label is left unchanged (detector verdict is independent
        of which explanation sentences survive).
    """
    from zone_localization import measure_zone_causal_drops, zone_passes_causal_check

    if not explanation.concept_vectors:
        return ExplanationOutput(
            label=explanation.label,
            fake_probability=explanation.fake_probability,
            concept_vectors=[],
            nl_explanations=[],
            faithfulness_passed=[],
            R_scores=explanation.R_scores,
            inferred_method=explanation.inferred_method,
            method_probabilities=explanation.method_probabilities,
            Q_scores=explanation.Q_scores,
            causal_correctness_passed=[],
            causal_drop_by_zone={},
            causal_min_drop=float(min_drop),
        )

    zones = [cv.zone for cv in explanation.concept_vectors]
    drops = measure_zone_causal_drops(
        fake_prob_fn=fake_prob_fn,
        image_bgr=image_bgr,
        landmarks_68=landmarks_68,
        zones=zones,
    )

    kept_cvs: List[ConceptVector] = []
    kept_sentences: List[str] = []
    kept_faithful: List[bool] = []
    kept_causal: List[bool] = []
    drop_by_zone: Dict[str, float] = {
        z: float(info["drop"]) for z, info in drops.items()
    }

    for i, cv in enumerate(explanation.concept_vectors):
        drop = drop_by_zone.get(cv.zone, 0.0)
        ok = zone_passes_causal_check(drop, min_drop=min_drop)
        if not ok:
            continue
        kept_cvs.append(cv)
        kept_sentences.append(explanation.nl_explanations[i])
        kept_faithful.append(explanation.faithfulness_passed[i])
        kept_causal.append(True)

    return ExplanationOutput(
        label=explanation.label,
        fake_probability=explanation.fake_probability,
        concept_vectors=kept_cvs,
        nl_explanations=kept_sentences,
        faithfulness_passed=kept_faithful,
        R_scores=explanation.R_scores,
        inferred_method=explanation.inferred_method,
        method_probabilities=explanation.method_probabilities,
        Q_scores=explanation.Q_scores,
        causal_correctness_passed=kept_causal,
        causal_drop_by_zone=drop_by_zone,
        causal_min_drop=float(min_drop),
    )


# ---------------------------------------------------------------------------
# Config loading + Bayesian prior reconstruction
# ---------------------------------------------------------------------------

def load_fusion_config(
    config_path: str = "configs/fusion_config.json",
):
    """
    Load fusion_config.json and reconstruct Bayesian objects where available.

    Returns a 5-tuple:
        config_dict   : raw JSON dict, or {} if file not found.
        artifact_prior: DirichletArtifactPrior instance, or None.
        A_stats       : {"mean": np.ndarray(9), "std": np.ndarray(9)}, or None.
        S_stats       : same for gradient scores, or None.
        beta_posterior: dict with "mean", "std", "ci_lower", "ci_upper" from
                        the Bayesian calibration run; or None if not present.
                        "std" is approximated as (ci_upper-ci_lower)/3.92 when
                        the field is absent (i.e. config pre-dates this feature).
    """
    p = Path(config_path)
    if not p.exists():
        return {}, None, None, None, None

    with open(p) as fh:
        cfg = json.load(fh)

    # Dirichlet artifact prior
    artifact_prior = None
    if "artifact_posteriors" in cfg:
        try:
            from bayesian_calibration import DirichletArtifactPrior as _DAP
            artifact_prior = _DAP.from_dict(cfg["artifact_posteriors"], ARTIFACT_ONTOLOGY)
        except ImportError:
            pass

    # Population normalisation stats
    A_stats: Optional[Dict] = None
    S_stats: Optional[Dict] = None
    if "A_stats" in cfg and "S_stats" in cfg:
        A_stats = {
            "mean": np.array(cfg["A_stats"]["mean"], dtype=np.float32),
            "std":  np.array(cfg["A_stats"]["std"],  dtype=np.float32),
        }
        S_stats = {
            "mean": np.array(cfg["S_stats"]["mean"], dtype=np.float32),
            "std":  np.array(cfg["S_stats"]["std"],  dtype=np.float32),
        }

    # β posterior summary
    beta_posterior: Optional[Dict] = None
    if "beta_posterior" in cfg:
        bp = dict(cfg["beta_posterior"])
        if "std" not in bp and "ci_lower" in bp and "ci_upper" in bp:
            # Approximate σ_β from the 95 % CI: σ ≈ (upper - lower) / (2 × 1.96)
            bp["std"] = round((bp["ci_upper"] - bp["ci_lower"]) / 3.92, 4)
        beta_posterior = bp

    return cfg, artifact_prior, A_stats, S_stats, beta_posterior


# ---------------------------------------------------------------------------
# Full pipeline entry point  (replaces SemanticExplanationGenerator.predict_label_and_explanation)
# ---------------------------------------------------------------------------

def predict_label_and_explanation(
    attention_vec: Sequence[float],
    landmarks_68: Sequence[Sequence[float]],
    phi_vec: Optional[Sequence[float]] = None,
    beta: float = 0.5,
    fake_probability: float = 0.5,
    min_severity: str = "medium",
    artifact_prior=None,
    A_stats: Optional[Dict] = None,
    S_stats: Optional[Dict] = None,
    beta_mean: Optional[float] = None,
    beta_std: Optional[float] = None,
    zone_prior_model=None,
    zone_thresholds: Optional[Mapping[str, float]] = None,
    max_flagged_zones: Optional[int] = None,
    min_fake_probability_for_explanations: float = 0.70,
) -> ExplanationOutput:
    """
    Full SAEDF inference pipeline for one frame.

    Steps:
      1. Build overlap-fraction M matrix from landmarks.
      2. Compute A(z) from attention vector.
      3. Compute S(z) from gradient/SHAP vector (if provided), else use A(z) only.
      4. Population-normalise A(z) and S(z) when calibration stats are provided
         (required for the calibrated β to be applied correctly).
      5. Fuse: R(z) = β·A_norm(z) + (1−β)·S_norm(z).
      6. Propagate β posterior uncertainty to per-zone 95 % credible intervals
         when beta_mean and beta_std are available.
      7. Bayesian zone prior update (Option 2): infer manipulation method from
         R(z) pattern and compute per-zone posterior Q(z) when zone_prior_model
         is provided.
      8. Build concept vectors for flagged zones (Bayesian artifact selection if
         artifact_prior is provided, else deterministic rank-0 fallback).
         Zone flagging uses Q(z) when available, else R(z).
         Research-grade gate: no flags when label is Real OR when
         P(fake) < min_fake_probability_for_explanations.
      9. Generate and verify NL explanations.

    Args:
        attention_vec    : shape (196,) — CLS patch attention weights.
        landmarks_68     : shape (68, 2) — landmark coordinates (224×224 space).
        phi_vec          : shape (196,) — patch gradient scores from heatmap generator.
                           If None, S(z) = A(z) (attention-only fallback).
        beta             : MAP β (loaded from fusion_config.json via
                           SemanticExplanationGenerator, or a fixed value for
                           legacy callers).
        fake_probability : detection probability from the classifier head.
        min_severity     : minimum severity tier to generate explanations for.
        artifact_prior   : optional DirichletArtifactPrior loaded via
                           load_fusion_config().
        A_stats          : {"mean": (9,), "std": (9,)} real-population stats for
                           attention scores.  Required for correct normalisation
                           when using a calibrated β.
        S_stats          : same structure for gradient scores.
        beta_mean        : posterior mean of β — used as the central value when
                           computing per-zone credible intervals.
        beta_std         : posterior standard deviation of β — enables Option 4
                           credible-interval propagation.
        zone_prior_model : optional ZonePriorModel instance (Option 2).  When
                           provided, infers manipulation method from R(z) and
                           computes posterior zone anomaly probabilities Q(z).
        zone_thresholds  : optional per-zone Q(z) cutoffs from fusion_config.
        max_flagged_zones: optional cap on flagged zones.
        min_fake_probability_for_explanations: P(fake) must exceed this before
                           any zone is flagged (default 0.70).

    Returns:
        ExplanationOutput with label, concept vectors, NL sentences, and
        (when zone_prior_model is set) inferred method + Q_scores.
    """
    alpha = np.asarray(attention_vec, dtype=np.float32)
    if alpha.shape != (NUM_PATCHES,):
        raise ValueError(f"Expected attention_vec shape ({NUM_PATCHES},), got {alpha.shape}.")

    # Build M
    M = build_overlap_fraction_matrix(landmarks_68)

    # A(z) — raw zone scores
    A = compute_attention_zone_scores(alpha, M)

    # S(z) — use gradient scores if available, else fall back to A
    if phi_vec is not None:
        phi = np.asarray(phi_vec, dtype=np.float32)
        if phi.shape != (NUM_PATCHES,):
            raise ValueError(f"Expected phi_vec shape ({NUM_PATCHES},), got {phi.shape}.")
        S = compute_shap_zone_scores(phi, M)
    else:
        S = A.copy()

    # Population normalisation — mandatory when using a calibrated β.
    # Without this step, applying a β fitted on normalised scores to raw scores
    # produces incorrect R(z) values.
    if A_stats is not None and S_stats is not None:
        A_norm = normalize_zone_scores(A, A_stats)
        S_norm = normalize_zone_scores(S, S_stats)
    else:
        A_norm = A
        S_norm = S

    # R(z) — fuse normalised scores
    R = fuse_scores(A_norm, S_norm, beta=beta)

    # Credible interval propagation via β posterior uncertainty (Option 4)
    ci_lower_arr: Optional[np.ndarray] = None
    ci_upper_arr: Optional[np.ndarray] = None
    if beta_mean is not None and beta_std is not None and beta_std > 0:
        try:
            from bayesian_calibration import compute_R_credible_interval
            ci_lower_arr, ci_upper_arr = compute_R_credible_interval(
                A_norm, S_norm, beta_mean=beta_mean, beta_std=beta_std
            )
        except ImportError:
            pass

    # Bayesian zone prior update (Option 2)
    inferred_method: Optional[str]            = None
    method_probs:    Optional[Dict[str, float]] = None
    Q_dict:          Optional[Dict[str, float]] = None
    Q_arr:           Optional[np.ndarray]      = None
    if zone_prior_model is not None:
        inferred_method, method_probs = zone_prior_model.infer_method(R, ZONE_NAMES)
        Q_dict = zone_prior_model.compute_marginal_posteriors(
            R, ZONE_NAMES, method_probs=method_probs
        )
        Q_arr = np.array([Q_dict.get(z, float(R[i])) for i, z in enumerate(ZONE_NAMES)],
                         dtype=np.float32)

        # Re-express the R(z) credible interval on the Q(z) scale, since Q(z)
        # (not R(z)) is what build_concept_vectors_from_R reports as
        # `confidence` whenever a zone prior is active. compute_marginal_posteriors
        # is, for a *fixed* method_probs weighting, a positive convex
        # combination of terms each linear and non-decreasing in R(z); it is
        # therefore itself non-decreasing in R(z), so applying it to
        # [ci_lower_arr, ci_upper_arr] yields a valid credible interval on
        # Q(z). Skipping this step would leave a Q(z)-based confidence figure
        # paired with an R(z)-based interval on a different scale -- e.g. a
        # nonsensical-looking "confidence: 73% [99-100%]".
        if ci_lower_arr is not None and ci_upper_arr is not None:
            q_ci_lower_dict = zone_prior_model.compute_marginal_posteriors(
                ci_lower_arr, ZONE_NAMES, method_probs=method_probs
            )
            q_ci_upper_dict = zone_prior_model.compute_marginal_posteriors(
                ci_upper_arr, ZONE_NAMES, method_probs=method_probs
            )
            ci_lower_arr = np.array([q_ci_lower_dict[z] for z in ZONE_NAMES], dtype=np.float32)
            ci_upper_arr = np.array([q_ci_upper_dict[z] for z in ZONE_NAMES], dtype=np.float32)

    # Label + research-grade explanation gate.
    # Manipulation claims require: Fake verdict AND confident P(fake).
    # Real / weak-Fake still get R/Q scores for the region table, but zero
    # flags / NL sentences. Thresholds alone are real-face quantiles, so
    # authentic faces would otherwise produce false forensic claims.
    label = "Fake" if fake_probability >= 0.5 else "Real"
    min_p = float(min_fake_probability_for_explanations)
    allow_explanations = label == "Fake" and float(fake_probability) > min_p

    if not allow_explanations:
        cvs = []
    else:
        # Flag only when score > calibrated threshold (or severity floor
        # when no per-zone thresholds).
        cvs = build_concept_vectors_from_R(
            R,
            min_severity=min_severity,
            artifact_prior=artifact_prior,
            ci_lower=ci_lower_arr,
            ci_upper=ci_upper_arr,
            Q_scores=Q_dict,
            zone_thresholds=zone_thresholds,
            max_zones=max_flagged_zones,
        )

    # NL explanations with faithfulness verification
    sentences: List[str] = []
    faithfulness: List[bool] = []
    for cv in cvs:
        sentence, passed = generate_verified_explanation(cv)
        sentences.append(sentence)
        faithfulness.append(passed)

    return ExplanationOutput(
        label=label,
        fake_probability=fake_probability,
        concept_vectors=cvs,
        nl_explanations=sentences,
        faithfulness_passed=faithfulness,
        R_scores=R,
        inferred_method=inferred_method,
        method_probabilities=method_probs,
        Q_scores=Q_arr,
    )


# ---------------------------------------------------------------------------
# SemanticExplanationGenerator — preserved class with upgraded internals
# ---------------------------------------------------------------------------

class SemanticExplanationGenerator:
    """
    High-level interface for zone-level explanation generation.
    Wraps the functional API above for easy batch use.

    When config_path is provided (or the default path exists), the generator
    automatically loads the MAP β and Dirichlet artifact posteriors produced by
    recalibrate_v2.py so that Bayesian artifact selection is active out of the box.
    """

    def __init__(
        self,
        beta: float = 0.5,
        min_severity: str = "medium",
        image_size: int = IMAGE_SIZE,
        grid_size: int = GRID_SIZE,
        patch_size: int = PATCH_SIZE,
        config_path: str = "configs/fusion_config.json",
    ) -> None:
        self.min_severity = min_severity
        self.image_size   = image_size
        self.grid_size    = grid_size
        self.patch_size   = patch_size
        _validate_grid(image_size, grid_size, patch_size)

        # Auto-load all Bayesian config fields; fall back to supplied beta if absent
        cfg, artifact_prior, A_stats, S_stats, beta_posterior = load_fusion_config(config_path)
        self.artifact_prior = artifact_prior   # DirichletArtifactPrior or None
        self.A_stats        = A_stats          # population normalisation stats or None
        self.S_stats        = S_stats
        self.zone_thresholds: Optional[Dict[str, float]] = (
            {str(k): float(v) for k, v in cfg.get("zone_thresholds", {}).items()}
            if cfg.get("zone_thresholds")
            else None
        )
        self.max_flagged_zones: Optional[int] = (
            int(cfg["max_flagged_zones"]) if cfg.get("max_flagged_zones") is not None else None
        )
        self.min_fake_probability_for_explanations: float = float(
            cfg.get("min_fake_probability_for_explanations", 0.70)
        )
        self.causal_min_drop: float = float(cfg.get("causal_min_drop", 0.03))
        self.research_policy: Optional[Dict] = cfg.get("research_policy")
        self.beta_mean: Optional[float] = (
            float(beta_posterior["mean"]) if beta_posterior else None
        )
        self.beta_std: Optional[float] = (
            float(beta_posterior["std"]) if beta_posterior else None
        )
        # MAP β from calibration (often 1.0) — kept for reporting / provenance.
        self.beta_map: float = float(cfg.get("beta", beta))
        # Fusion uses posterior mean when available so S(z) is not zeroed out
        # when MAP β collapses to 1.0 (attention-only).
        use_mean = str(cfg.get("beta_for_fusion", "posterior_mean")).lower() in (
            "posterior_mean", "mean", "1", "true"
        )
        if use_mean and self.beta_mean is not None:
            self.beta = float(self.beta_mean)
        else:
            self.beta = self.beta_map
        # Bayesian zone prior (Option 2) — always active. Uses learned method
        # signatures from calibrate_method_signatures.py when the signature
        # file exists, otherwise falls back to the literature priors.
        # evidence_weight comes from fusion_config (recalibrate_v2.py) so
        # inference Q(z) matches the scale the thresholds were fit on.
        try:
            from zone_prior import ZonePriorModel as _ZPM
            ew = float(cfg.get("evidence_weight", 10.0))
            sig_path = Path(
                cfg.get("method_signatures_path") or "configs/method_signatures.json"
            )
            if sig_path.exists():
                self.zone_prior_model = _ZPM.from_signature_file(
                    sig_path, evidence_weight=ew
                )
            else:
                self.zone_prior_model = _ZPM(evidence_weight=ew)
        except ImportError:
            self.zone_prior_model = None

    def predict(
        self,
        attention_vec: Sequence[float],
        landmarks_68: Sequence[Sequence[float]],
        phi_vec: Optional[Sequence[float]] = None,
        fake_probability: float = 0.5,
    ) -> ExplanationOutput:
        """Single-frame inference. See predict_label_and_explanation() for details."""
        return predict_label_and_explanation(
            attention_vec=attention_vec,
            landmarks_68=landmarks_68,
            phi_vec=phi_vec,
            beta=self.beta,
            fake_probability=fake_probability,
            min_severity=self.min_severity,
            artifact_prior=self.artifact_prior,
            A_stats=self.A_stats,
            S_stats=self.S_stats,
            beta_mean=self.beta_mean,
            beta_std=self.beta_std,
            zone_prior_model=self.zone_prior_model,
            zone_thresholds=self.zone_thresholds,
            max_flagged_zones=self.max_flagged_zones,
            min_fake_probability_for_explanations=self.min_fake_probability_for_explanations,
        )

    def predict_batch(
        self,
        attention_batch: np.ndarray,
        landmarks_batch: List[Sequence[Sequence[float]]],
        phi_batch: Optional[np.ndarray] = None,
        fake_probabilities: Optional[np.ndarray] = None,
    ) -> List[ExplanationOutput]:
        """
        Run inference on a batch.

        Args:
            attention_batch   : shape (B, 196).
            landmarks_batch   : list of B × (68, 2) landmark arrays.
            phi_batch         : shape (B, 196) — optional gradient scores.
            fake_probabilities: shape (B,) — detection confidence per frame.

        Returns:
            List of B ExplanationOutput objects.
        """
        B = len(landmarks_batch)
        if attention_batch.shape[0] != B:
            raise ValueError("Batch size mismatch between attention_batch and landmarks_batch.")

        results = []
        for i in range(B):
            phi = phi_batch[i] if phi_batch is not None else None
            prob = float(fake_probabilities[i]) if fake_probabilities is not None else 0.5
            results.append(self.predict(
                attention_vec=attention_batch[i],
                landmarks_68=landmarks_batch[i],
                phi_vec=phi,
                fake_probability=prob,
            ))
        return results

    def visualise_R_scores(
        self,
        R: np.ndarray,
        save_path: Optional[Union[str, Path]] = None,
        title: str = "Fused zone scores R(z)",
    ) -> None:
        """Horizontal bar chart of R(z) scores with severity colour coding."""
        if R.shape != (NUM_ZONES,):
            raise ValueError(f"Expected R shape ({NUM_ZONES},), got {R.shape}.")

        colours = []
        for s in R:
            if s > SEVERITY_THRESHOLDS["high"]:
                colours.append("#D85A30")   # coral — high
            elif s > SEVERITY_THRESHOLDS["medium"]:
                colours.append("#BA7517")   # amber — medium
            elif s > SEVERITY_THRESHOLDS["low"]:
                colours.append("#888780")   # gray  — low
            else:
                colours.append("#D3D1C7")   # light gray — below threshold

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.barh(ZONE_NAMES, R, color=colours, edgecolor="none", height=0.6)
        for threshold, colour, label in [
            (0.75, "#D85A30", "high"),
            (0.50, "#BA7517", "medium"),
            (0.25, "#888780", "low"),
        ]:
            ax.axvline(threshold, color=colour, linestyle="--", linewidth=0.8, alpha=0.6,
                       label=f"{label} ({threshold})")
        for bar, val in zip(bars, R):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha="left", fontsize=9)
        ax.set_xlabel("R(z) fused score")
        ax.set_title(title)
        ax.set_xlim(0, 1.1)
        ax.legend(fontsize=8, loc="lower right")
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    def visualise_semantic_patch_assignment(
        self,
        image: np.ndarray,
        landmarks_68: Sequence[Sequence[float]],
        attention_vec: Optional[Sequence[float]] = None,
        R: Optional[np.ndarray] = None,
        alpha: float = 0.30,
        show: bool = True,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Overlay zone convex hulls and ViT patch grid on a face image.
        Optional: scale patch opacity by attention_vec.
        Optional: colour patches by R-score severity when R is provided.
        """
        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            raise ValueError(f"Expected image ({self.image_size}x{self.image_size}), got {image.shape}.")

        M = build_overlap_fraction_matrix(landmarks_68)
        attention = np.asarray(attention_vec, dtype=np.float32) if attention_vec is not None else None
        max_attn = float(np.max(attention)) if attention is not None else 1.0

        zone_colours_rgba = {
            "left_eye":    (0.1, 0.4, 1.0, alpha),
            "right_eye":   (0.1, 0.4, 1.0, alpha),
            "mouth":       (1.0, 0.0, 0.0, alpha),
            "nose_bridge": (0.0, 0.8, 0.2, alpha),
            "nose_tip":    (0.0, 0.8, 0.2, alpha),
            "left_cheek":  (0.8, 0.5, 0.0, alpha),
            "right_cheek": (0.8, 0.5, 0.0, alpha),
            "forehead":    (0.6, 0.0, 0.8, alpha),
            "chin_jaw":    (1.0, 0.7, 0.1, alpha),
        }

        fig, ax = plt.subplots(figsize=(7, 7))
        display = image if image.ndim == 3 and image.shape[2] == 3 else image
        ax.imshow(display if display.dtype == np.uint8 else (display * 255).astype(np.uint8))

        for patch_idx in range(NUM_PATCHES):
            row = patch_idx // self.grid_size
            col = patch_idx % self.grid_size
            x = col * self.patch_size
            y = row * self.patch_size

            # Pick dominant zone for this patch
            zone_idx = int(M[patch_idx].argmax()) if M[patch_idx].max() > 0 else -1
            if zone_idx < 0:
                continue

            zone_name = ZONE_NAMES[zone_idx]
            base_colour = zone_colours_rgba[zone_name]

            # Scale alpha by attention if provided
            if attention is not None and max_attn > 0:
                scale = 0.2 + 0.8 * (float(attention[patch_idx]) / max_attn)
                facecolour = (*base_colour[:3], min(0.85, base_colour[3] * scale + 0.05))
            else:
                facecolour = base_colour

            # Outline colour by R severity if R provided
            edgecolour = "yellow"
            if R is not None:
                sev = assign_severity(float(R[zone_idx]))
                if sev == "high":
                    edgecolour = "#D85A30"
                elif sev == "medium":
                    edgecolour = "#BA7517"

            rect = mpatches.Rectangle(
                (x, y), self.patch_size, self.patch_size,
                linewidth=0.8, edgecolor=edgecolour, facecolor=facecolour,
            )
            ax.add_patch(rect)

        # Landmarks
        pts = np.asarray(landmarks_68, dtype=np.float32)
        ax.scatter(pts[:, 0], pts[:, 1], s=6, c="white", marker="o", linewidths=0)

        legend_items = [
            mpatches.Patch(color=zone_colours_rgba[z], label=z.replace("_", " "))
            for z in ZONE_NAMES
        ]
        ax.legend(handles=legend_items, loc="upper right", fontsize=7, ncol=2)
        ax.set_title("Semantic patch assignment — 9-zone SAEDF grid")
        ax.set_xlim(0, self.image_size)
        ax.set_ylim(self.image_size, 0)
        ax.set_xticks(np.arange(0, self.image_size + 1, self.patch_size))
        ax.set_yticks(np.arange(0, self.image_size + 1, self.patch_size))
        ax.grid(False)
        ax.set_aspect("equal")

        if show:
            plt.show()

        return fig, ax


# ---------------------------------------------------------------------------
# JSON loading helpers
# ---------------------------------------------------------------------------

def build_frame_semantic_masks(
    landmarks_json_path: Union[str, Path],
    use_overlap_fraction: bool = True,
    as_torch: bool = False,
    torch_dtype=None,
) -> Dict[str, Union[np.ndarray, "torch.Tensor"]]:
    """
    Load landmarks JSON and return per-frame zone-patch matrices.

    If the JSON already contains 'zone_patch_matrix' (produced by the updated
    preprocess_landmarks.py), those are loaded directly.  Otherwise the matrix is
    recomputed from landmarks_68.

    Args:
        landmarks_json_path  : Path to JSON file.
        use_overlap_fraction : If True (default), use continuous overlap-fraction M.
                               If False, use binary M (ablation baseline).
    """
    json_path = Path(landmarks_json_path)
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    frames = payload.get("frames") if isinstance(payload, dict) else payload
    if not isinstance(frames, dict):
        raise ValueError("Unsupported landmarks JSON format.")

    result: Dict = {}
    for frame_key, frame_data in frames.items():
        if not isinstance(frame_data, dict):
            continue
        landmarks = frame_data.get("landmarks_68")
        if landmarks is None:
            continue

        # Use pre-computed M if available and we want overlap-fraction
        if use_overlap_fraction and "zone_patch_matrix" in frame_data:
            M = np.array(frame_data["zone_patch_matrix"], dtype=np.float32)
        elif use_overlap_fraction:
            M = build_overlap_fraction_matrix(landmarks)
        else:
            M = build_binary_mask_matrix(landmarks)

        if as_torch:
            if torch is None:
                raise ImportError("PyTorch is not installed.")
            dtype = torch_dtype if torch_dtype is not None else torch.float32
            result[frame_key] = torch.tensor(M, dtype=dtype)
        else:
            result[frame_key] = M

    return result


__all__ = [
    # Constants
    "ZONE_NAMES", "NUM_ZONES", "ARTIFACT_ONTOLOGY", "SEVERITY_THRESHOLDS",
    # Data classes
    "ConceptVector", "ExplanationOutput",
    # M matrix
    "build_overlap_fraction_matrix", "build_binary_mask_matrix", "build_semantic_mask_matrix",
    # Zone scoring
    "compute_attention_zone_scores", "compute_shap_zone_scores",
    # Fusion
    "fuse_scores", "assign_severity", "get_flagged_zones",
    # Concept ontology
    "build_concept_vector", "build_concept_vectors_from_R",
    # NLG
    "generate_nl_explanation", "verify_faithfulness", "generate_verified_explanation",
    "filter_explanations_by_causal_correctness", "DEFAULT_CAUSAL_MIN_DROP",
    # Pipeline
    "predict_label_and_explanation",
    # Class
    "SemanticExplanationGenerator",
    # JSON helpers
    "build_frame_semantic_masks",
]