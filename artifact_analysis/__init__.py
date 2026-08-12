"""
artifact_analysis — measurable artifact evidence for SAEDF.

Extends the existing zone-scoring pipeline with interpretable image features
and Bayesian artifact-type inference. Does not replace A(z)/S(z)/R(z)/Q(z).
"""

from artifact_analysis.artifact_ontology import (
    ARTIFACT_DISPLAY_NAMES,
    ARTIFACT_TYPES,
    ArtifactEvidenceConfig,
    load_artifact_config,
)
from artifact_analysis.feature_extractor import (
    ZoneFeatureExtractor,
    extract_geometry_features,
)
from artifact_analysis.artifact_inference import (
    ArtifactEvidenceAnalyzer,
    ArtifactPrediction,
    BayesianArtifactInferencer,
    ExtendedConceptVector,
    ExtendedFaithfulnessVerifier,
    analyze_zone_artifact,
    generate_artifact_explanation,
    normalize_features,
)
from artifact_analysis.calibration import ArtifactCalibrationModel
from artifact_analysis.visualization import draw_artifact_result
from artifact_analysis.pipeline_integration import enrich_explanation_with_artifact_evidence

__all__ = [
    "ARTIFACT_TYPES",
    "ARTIFACT_DISPLAY_NAMES",
    "ArtifactEvidenceConfig",
    "load_artifact_config",
    "ZoneFeatureExtractor",
    "extract_geometry_features",
    "ArtifactEvidenceAnalyzer",
    "ArtifactPrediction",
    "BayesianArtifactInferencer",
    "ExtendedConceptVector",
    "ExtendedFaithfulnessVerifier",
    "analyze_zone_artifact",
    "generate_artifact_explanation",
    "normalize_features",
    "ArtifactCalibrationModel",
    "draw_artifact_result",
    "enrich_explanation_with_artifact_evidence",
]
