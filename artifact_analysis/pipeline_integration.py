"""
pipeline_integration.py — attach ArtifactEvidenceAnalyzer to SAEDF outputs.

Only processes zones already flagged by Q(z) thresholds. Does not recompute
A/S/R/Q. Visualization pixels are never used as scoring inputs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from artifact_analysis.artifact_inference import (
    ArtifactEvidenceAnalyzer,
    ExtendedConceptVector,
    ExtendedFaithfulnessVerifier,
    generate_artifact_explanation,
)
from artifact_analysis.artifact_ontology import map_measurable_to_ontology_phrase
from zone_localization import zone_pixel_masks


def _neighbor_mask(
    zone_mask: np.ndarray, all_masks: Dict[str, np.ndarray], zone_name: str
) -> np.ndarray:
    """Union of other facial-zone pixels as neighboring skin context."""
    neigh = np.zeros_like(zone_mask, dtype=bool)
    for z, m in all_masks.items():
        if z == zone_name:
            continue
        neigh |= m.astype(bool)
    neigh &= ~zone_mask.astype(bool)
    if not neigh.any():
        # Fallback ring
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dil = cv2.dilate(zone_mask.astype(np.uint8), k).astype(bool)
        neigh = dil & (~zone_mask.astype(bool))
    return neigh


def enrich_explanation_with_artifact_evidence(
    explanation,
    image_bgr: np.ndarray,
    landmarks_68: Sequence[Sequence[float]],
    analyzer: Optional[ArtifactEvidenceAnalyzer] = None,
    replace_nl: bool = True,
) -> Tuple[object, List[ExtendedConceptVector]]:
    """
    Run measurable artifact analysis on each flagged zone.

    Args:
        explanation: ExplanationOutput from the existing SAEDF head
        image_bgr: 224x224 BGR face crop (pipeline native)
        landmarks_68: (68, 2) landmarks in 224 space
        analyzer: optional shared ArtifactEvidenceAnalyzer
        replace_nl: if True, replace NL sentences with evidence-backed text

    Returns:
        (updated explanation, list of ExtendedConceptVector)
    """
    if analyzer is None:
        analyzer = ArtifactEvidenceAnalyzer()

    if not explanation.concept_vectors:
        explanation.artifact_evidence = []
        explanation.artifact_explanations = []
        explanation.artifact_faithfulness = []
        return explanation, []

    # BGR -> RGB for the feature extractor contract
    img = np.asarray(image_bgr)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lm = np.asarray(landmarks_68, dtype=np.float64)
    masks = zone_pixel_masks(lm)

    # Zone index map for R/Q
    from semantic_patch_mapper import ZONE_NAMES

    zone_to_idx = {z: i for i, z in enumerate(ZONE_NAMES)}
    verifier = ExtendedFaithfulnessVerifier()

    extended: List[ExtendedConceptVector] = []
    new_sentences: List[str] = []
    new_faith: List[bool] = []
    evidence_dicts: List[Dict] = []

    for i, cv in enumerate(explanation.concept_vectors):
        zmask = masks.get(cv.zone)
        if zmask is None or not zmask.any():
            # Keep legacy sentence if mask missing
            extended.append(
                ExtendedConceptVector(
                    zone=cv.zone,
                    artifact_type="no_clear_artifact",
                    severity=cv.severity,
                    zone_score_r=float(cv.R_score),
                    zone_score_q=float(
                        cv.zone_prior_score if cv.zone_prior_score is not None else cv.confidence
                    ),
                    artifact_probability=0.0,
                    credible_interval=(0.0, 0.0),
                    evidence_features={},
                    abstained=True,
                )
            )
            sent = explanation.nl_explanations[i] if i < len(explanation.nl_explanations) else ""
            new_sentences.append(sent)
            new_faith.append(
                explanation.faithfulness_passed[i]
                if i < len(explanation.faithfulness_passed)
                else False
            )
            evidence_dicts.append({"zone": cv.zone, "abstained": True, "reason": "missing_mask"})
            continue

        neigh = _neighbor_mask(zmask, masks, cv.zone)
        idx = zone_to_idx.get(cv.zone, 0)
        r_score = float(explanation.R_scores[idx]) if explanation.R_scores is not None else float(cv.R_score)
        if explanation.Q_scores is not None:
            q_score = float(explanation.Q_scores[idx])
        elif cv.zone_prior_score is not None:
            q_score = float(cv.zone_prior_score)
        else:
            q_score = float(cv.confidence)

        concept = analyzer.analyze(
            image_rgb=image_rgb,
            zone_name=cv.zone,
            zone_mask=zmask,
            neighboring_mask=neigh,
            landmarks=lm,
            r_score=r_score,
            q_score=q_score,
        )
        extended.append(concept)
        sent = generate_artifact_explanation(concept)
        ok, _errs = verifier.verify(sent, concept)
        new_sentences.append(sent)
        new_faith.append(ok)
        evidence_dicts.append(concept.to_dict())

        # Soft-update legacy ConceptVector artifact string when evidence is clear
        if not concept.abstained and concept.artifact_type != "no_clear_artifact":
            cv.artifact_type = map_measurable_to_ontology_phrase(
                concept.artifact_type, concept.zone
            )

    explanation.artifact_evidence = evidence_dicts
    explanation.artifact_explanations = new_sentences
    explanation.artifact_faithfulness = new_faith

    if replace_nl:
        explanation.nl_explanations = new_sentences
        explanation.faithfulness_passed = new_faith

    return explanation, extended
