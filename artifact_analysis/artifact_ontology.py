"""
artifact_ontology.py — artifact types, priors, and feature-weight configuration.

RULE-BASED / CONFIG COMPONENT
-----------------------------
Values here are *initial weak priors*, not scientific ground truth.
They must be recalibrated with labeled artifact annotations before any
artifact-level claims are treated as evaluated results.

R(z) and Q(z) never select the artifact class; they only modulate confidence
via g(R,Q) and abstention gates (see artifact_inference.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

# Canonical measurable artifact vocabulary for ArtifactEvidenceAnalyzer.
# Distinct from semantic_patch_mapper.ARTIFACT_ONTOLOGY (literature phrases).
ARTIFACT_TYPES: List[str] = [
    "texture_inconsistency",
    "blending_boundary",
    "color_tone_mismatch",
    "lighting_inconsistency",
    "geometric_distortion",
    "frequency_artifact",
    "compression_artifact",
    "no_clear_artifact",
]

# Human-readable phrases for deterministic NL generation.
ARTIFACT_DISPLAY_NAMES: Dict[str, str] = {
    "texture_inconsistency": "texture inconsistency",
    "blending_boundary": "blending boundary",
    "color_tone_mismatch": "color tone mismatch",
    "lighting_inconsistency": "lighting inconsistency",
    "geometric_distortion": "geometric distortion",
    "frequency_artifact": "frequency artifact",
    "compression_artifact": "compression artifact",
    "no_clear_artifact": "no clear artifact",
}

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "artifact_evidence_config.json"
)


@dataclass
class AbstentionConfig:
    """Configurable abstention gates — avoid unsupported forensic claims."""

    min_probability: float = 0.35
    min_margin: float = 0.08
    min_q_score: float = 0.50
    min_image_laplacian_var: float = 5.0
    min_zone_pixels: int = 64


@dataclass
class GRQConfig:
    """
    Confidence adjustment g(R, Q).

    BAYESIAN / CONFIDENCE COMPONENT — does not choose artifact type.
    """

    r_weight: float = 0.4
    q_weight: float = 0.6
    floor: float = 0.15
    ceiling: float = 1.0


@dataclass
class ArtifactEvidenceConfig:
    """
    Loaded configuration for feature weights, Dirichlet priors, and gates.

    Attributes marked as priors are intentionally low-concentration so that
    measured visual evidence can dominate the posterior.
    """

    artifact_types: List[str] = field(default_factory=lambda: list(ARTIFACT_TYPES))
    artifact_feature_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)
    zone_artifact_priors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    feature_display_names: Dict[str, str] = field(default_factory=dict)
    severity_thresholds: Dict[str, float] = field(
        default_factory=lambda: {"high": 0.75, "medium": 0.50, "low": 0.25}
    )
    abstention: AbstentionConfig = field(default_factory=AbstentionConfig)
    g_rq: GRQConfig = field(default_factory=GRQConfig)
    feature_norm_eps: float = 1e-8
    version: str = "0.1.0"
    notes: List[str] = field(default_factory=list)

    def prior_vector(self, zone_name: str) -> Dict[str, float]:
        """Return Dirichlet concentration α_a for zone z (weak prior)."""
        zone_priors = self.zone_artifact_priors.get(zone_name)
        if zone_priors is None:
            return {a: 1.0 for a in self.artifact_types}
        out: Dict[str, float] = {}
        for a in self.artifact_types:
            out[a] = float(zone_priors.get(a, 1.0))
        return out

    def feature_weights(self, artifact_type: str) -> Dict[str, float]:
        return dict(self.artifact_feature_weights.get(artifact_type, {}))


def _parse_abstention(raw: Mapping[str, Any], min_zone_pixels: int) -> AbstentionConfig:
    return AbstentionConfig(
        min_probability=float(raw.get("min_probability", 0.35)),
        min_margin=float(raw.get("min_margin", 0.08)),
        min_q_score=float(raw.get("min_q_score", 0.50)),
        min_image_laplacian_var=float(raw.get("min_image_laplacian_var", 5.0)),
        min_zone_pixels=int(raw.get("min_zone_pixels", min_zone_pixels)),
    )


def load_artifact_config(
    path: Optional[str | Path] = None,
) -> ArtifactEvidenceConfig:
    """
    Load artifact evidence configuration from JSON.

    Falls back to built-in defaults if the file is missing.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        return ArtifactEvidenceConfig(
            notes=["Config file missing; using in-code defaults."]
        )

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    min_zone_px = int(raw.get("min_zone_pixels", 64))
    abst_raw = dict(raw.get("abstention", {}))
    abst_raw.setdefault("min_zone_pixels", min_zone_px)

    g_raw = raw.get("g_rq", {})
    priors = {
        k: v
        for k, v in raw.get("zone_artifact_priors", {}).items()
        if not str(k).startswith("_")
    }

    return ArtifactEvidenceConfig(
        artifact_types=list(raw.get("artifact_types", ARTIFACT_TYPES)),
        artifact_feature_weights=dict(raw.get("artifact_feature_weights", {})),
        zone_artifact_priors=priors,
        feature_display_names=dict(raw.get("feature_display_names", {})),
        severity_thresholds=dict(
            raw.get("severity_thresholds", {"high": 0.75, "medium": 0.50, "low": 0.25})
        ),
        abstention=_parse_abstention(abst_raw, min_zone_px),
        g_rq=GRQConfig(
            r_weight=float(g_raw.get("r_weight", 0.4)),
            q_weight=float(g_raw.get("q_weight", 0.6)),
            floor=float(g_raw.get("floor", 0.15)),
            ceiling=float(g_raw.get("ceiling", 1.0)),
        ),
        feature_norm_eps=float(
            raw.get("feature_normalization", {}).get("eps", 1e-8)
        ),
        version=str(raw.get("version", "0.1.0")),
        notes=list(raw.get("notes", [])),
    )


def map_measurable_to_ontology_phrase(
    measurable_artifact: str,
    zone_name: str,
    ontology: Optional[Mapping[str, List[str]]] = None,
) -> str:
    """
    Soft bridge from measurable artifact IDs to existing ARTIFACT_ONTOLOGY phrases.

    Used only when integrating with the legacy NL path; the evidence module
    itself speaks in ARTIFACT_TYPES.
    """
    if measurable_artifact == "no_clear_artifact":
        return "no clear artifact"

    if ontology is None:
        from semantic_patch_mapper import ARTIFACT_ONTOLOGY

        ontology = ARTIFACT_ONTOLOGY

    phrases = ontology.get(zone_name, [])
    if not phrases:
        return ARTIFACT_DISPLAY_NAMES.get(measurable_artifact, measurable_artifact)

    keywords = {
        "texture_inconsistency": ("texture", "pore", "smooth", "iris"),
        "blending_boundary": ("blend", "seam", "boundary", "mask"),
        "color_tone_mismatch": ("colour", "color", "tone", "pigment"),
        "lighting_inconsistency": ("light", "specular", "shadow", "highlight"),
        "geometric_distortion": ("geometric", "shape", "warp", "distort"),
        "frequency_artifact": ("gan", "upsampl", "ripple", "frequency"),
        "compression_artifact": ("compression", "block"),
    }
    keys = keywords.get(measurable_artifact, ())
    for phrase in phrases:
        low = phrase.lower()
        if any(k in low for k in keys):
            return phrase
    return phrases[0]
