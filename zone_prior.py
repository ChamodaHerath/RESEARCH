"""
zone_prior.py — Bayesian zone anomaly priors (Option 2).

This module implements the second major Bayesian contribution to the SAEDF
pipeline: manipulation-type-conditioned zone priors.

Scientific motivation
---------------------
Different deepfake synthesis methods leave characteristic artefacts in
different facial zones.  The FaceForensics++ benchmark (Rössler et al., 2019)
includes four manipulation methods with well-documented zone signatures:

  Deepfakes (DF)         — autoencoder face swap
  Face2Face  (F2F)       — expression reenactment
  FaceSwap   (FS)        — face replacement (cruder boundary blending)
  NeuralTextures (NT)    — neural texture rendering

This domain knowledge is encoded as a Beta prior P(zone z is anomalous | method m)
for each (zone, method) pair.  At inference the prior is updated with the
observed R(z) score through a soft Beta-Binomial model, producing a principled
posterior zone anomaly probability Q(z):

    Q(z) = E[Beta(α + R(z)·w,  β + (1−R(z))·w)]
         = (α + R(z)·w) / (α + R(z)·w + β + (1−R(z))·w)

where (α, β) are the literature-based prior parameters and w is an evidence
weight controlling how strongly R(z) updates the prior.

Because the manipulation method is typically unknown at inference time, the
module also provides:

  1. Method inference — cosine similarity between R(z) and each method's
     expected zone signature, converted to a probability distribution via
     softmax.  This is a novel contribution: the zone anomaly pattern is used
     to identify the manipulation class, which then conditions the posterior.

  2. Method-marginalised posteriors — Q(z) averaged over method posteriors
     using the inferred method probabilities as weights.

References
----------
Rössler A. et al. (2019). "FaceForensics++: Learning to Detect Manipulated
Facial Images." ICCV 2019.  arXiv:1901.08971.

Zhao H. et al. (2021). "Multi-attentional Deepfake Detection." CVPR 2021.

Li L. et al. (2020). "Face X-Ray for More General Face Forgery Detection."
CVPR 2020.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Literature-based zone priors  (α, β) → Beta(α, β)
# Prior mean = α / (α + β)
# ---------------------------------------------------------------------------
#
# Encoding guide (prior mean → (α, β)):
#   0.80  Very high  → (8, 2)
#   0.70  High       → (7, 3)
#   0.63  Med-high   → (5, 3)
#   0.50  Medium     → (4, 4)
#   0.38  Med-low    → (3, 5)
#   0.25  Low        → (2, 6)
#   0.17  Very low   → (1, 5)
#
# Sources: Rössler et al. 2019 (FaceForensics++), Zhao et al. 2021
#          (multi-attention analysis), Li et al. 2020 (Face X-Ray boundary
#          analysis).

ZONE_METHOD_PRIORS: Dict[str, Dict[str, Tuple[int, int]]] = {

    # ── Deepfakes (autoencoder face swap) ──────────────────────────────────
    # Artefacts arise from decoder reconstruction errors at the blending
    # boundary (mouth, eye rims, chin/jaw) and from colour/texture mismatches
    # at the face periphery.  Inner facial regions are less affected because
    # the autoencoder reconstructs the central face more faithfully.
    "Deepfakes": {
        "mouth":       (8, 2),   # 0.80 — blending boundary at lip outline
        "left_eye":    (7, 3),   # 0.70 — iris/eyelid boundary blending
        "right_eye":   (7, 3),   # 0.70
        "chin_jaw":    (5, 3),   # 0.63 — peripheral face boundary
        "left_cheek":  (4, 4),   # 0.50 — moderate peripheral blending
        "right_cheek": (4, 4),   # 0.50
        "forehead":    (3, 5),   # 0.38 — hairline boundary (present but weak)
        "nose_bridge": (2, 6),   # 0.25 — inner face, less affected
        "nose_tip":    (2, 6),   # 0.25
    },

    # ── Face2Face (expression reenactment) ─────────────────────────────────
    # Expression/motion transfer targets the mouth almost exclusively, with
    # secondary effects in the eye region (blink and gaze reenactment).
    # Skin texture is not altered; boundary regions are largely unchanged.
    "Face2Face": {
        "mouth":       (9, 1),   # 0.90 — primary target: lip/jaw expression
        "left_eye":    (6, 3),   # 0.67 — blink + gaze reenactment
        "right_eye":   (6, 3),   # 0.67
        "chin_jaw":    (3, 5),   # 0.38 — some jaw movement artefacts
        "left_cheek":  (2, 6),   # 0.25 — not targeted
        "right_cheek": (2, 6),   # 0.25
        "forehead":    (1, 5),   # 0.17 — not targeted
        "nose_bridge": (1, 5),   # 0.17
        "nose_tip":    (2, 5),   # 0.29 — minor warping during jaw movement
    },

    # ── FaceSwap (cruder geometric face replacement) ────────────────────────
    # Replaces the entire face region; boundary blending is less refined than
    # Deepfakes.  The peripheral boundary zones (chin/jaw, cheeks) show the
    # strongest artefacts, with secondary effects at eye/mouth.
    "FaceSwap": {
        "mouth":       (7, 3),   # 0.70 — boundary blending
        "left_eye":    (6, 3),   # 0.67 — boundary blending
        "right_eye":   (6, 3),   # 0.67
        "chin_jaw":    (8, 2),   # 0.80 — primary: outer face boundary
        "left_cheek":  (7, 3),   # 0.70 — face boundary region
        "right_cheek": (7, 3),   # 0.70
        "forehead":    (4, 4),   # 0.50 — hairline boundary
        "nose_bridge": (3, 5),   # 0.38 — some geometric warping
        "nose_tip":    (3, 5),   # 0.38
    },

    # ── NeuralTextures (neural texture rendering) ───────────────────────────
    # Renders face regions using learned neural textures; primarily affects
    # skin texture zones (cheeks, forehead).  Boundary regions are rendered
    # more smoothly than DF/FS, but texture inconsistencies remain.
    "NeuralTextures": {
        "mouth":       (6, 3),   # 0.67 — neural texture mouth rendering
        "left_eye":    (3, 5),   # 0.38 — less targeted
        "right_eye":   (3, 5),   # 0.38
        "chin_jaw":    (2, 6),   # 0.25 — smoother than DF/FS here
        "left_cheek":  (8, 2),   # 0.80 — primary: skin texture artefact
        "right_cheek": (8, 2),   # 0.80 — primary
        "forehead":    (7, 3),   # 0.70 — skin texture (pore erasure)
        "nose_bridge": (3, 5),   # 0.38 — moderate texture effect
        "nose_tip":    (3, 5),   # 0.38
    },
}

MANIPULATION_METHODS: List[str] = list(ZONE_METHOD_PRIORS.keys())


# ---------------------------------------------------------------------------
# ZonePriorModel
# ---------------------------------------------------------------------------

class ZonePriorModel:
    """
    Manipulation-type-conditioned Bayesian zone anomaly prior.

    For each (zone, method) pair the prior P(zone anomalous | method) is a
    Beta(α, β) distribution whose parameters come from the deepfake detection
    literature.  At inference the prior is updated with the observed R(z)
    score via a soft Beta-Binomial model:

        Q(z | method) = (α + R(z)·w) / (α + β + w)

    where w is the evidence weight (default 3.0 — R(z) counts as 3 soft
    binary observations).

    When the manipulation method is unknown the model infers the most likely
    method from the R(z) zone pattern via cosine similarity with each method's
    prior-mean signature.  The inferred method probabilities are then used to
    compute a marginalised posterior Q(z) = Σ_m P(method=m | R) · Q(z | m).
    """

    def __init__(
        self,
        priors: Dict[str, Dict[str, Tuple[int, int]]] = ZONE_METHOD_PRIORS,
        evidence_weight: float = 3.0,
    ) -> None:
        """
        Args:
            priors          : Nested dict {method: {zone: (alpha, beta)}}.
                              Defaults to ZONE_METHOD_PRIORS (literature values).
            evidence_weight : w in the Beta-Binomial update.  Higher values
                              give R(z) more influence relative to the prior.
        """
        self.priors          = priors
        self.evidence_weight = evidence_weight
        self.methods         = list(priors.keys())

        # Precompute prior-mean signatures per method for fast inference
        self._signatures: Dict[str, Dict[str, float]] = {}
        for method, zone_params in priors.items():
            self._signatures[method] = {
                zone: a / (a + b)
                for zone, (a, b) in zone_params.items()
            }

    # ------------------------------------------------------------------
    def prior_mean(self, zone: str, method: str) -> float:
        """Return α/(α+β) for the specified (zone, method) pair."""
        a, b = self.priors[method][zone]
        return a / (a + b)

    # ------------------------------------------------------------------
    def compute_zone_posterior(
        self,
        R_z: float,
        zone: str,
        method: str,
        weight: Optional[float] = None,
    ) -> float:
        """
        Compute Q(z) = E[Beta(α + R_z·w,  β + (1−R_z)·w)].

        The posterior mean has a closed form:

            Q(z) = (α + R_z · w) / (α + β + w)

        which is a convex combination of the prior mean α/(α+β) and the data
        signal R_z, weighted by w / (α + β + w).

        Args:
            R_z    : fused anomaly score in [0, 1].
            zone   : zone name.
            method : manipulation method name.
            weight : evidence weight (overrides instance default if provided).

        Returns:
            Posterior mean in [0, 1].
        """
        w    = weight if weight is not None else self.evidence_weight
        a, b = self.priors[method][zone]
        return float((a + R_z * w) / (a + b + w))

    # ------------------------------------------------------------------
    def compute_all_posteriors(
        self,
        R_scores: np.ndarray,
        zone_names: List[str],
        method: str,
        weight: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Compute Q(z) for all zones given a specific method.

        Args:
            R_scores   : (9,) array of fused zone scores.
            zone_names : list of zone name strings, matching R_scores order.
            method     : manipulation method name.
            weight     : evidence weight.

        Returns:
            Dict {zone_name: posterior_mean}.
        """
        return {
            zone: self.compute_zone_posterior(
                float(R_scores[i]), zone, method, weight
            )
            for i, zone in enumerate(zone_names)
            if zone in self.priors.get(method, {})
        }

    # ------------------------------------------------------------------
    def infer_method(
        self,
        R_scores: np.ndarray,
        zone_names: List[str],
        temperature: float = 10.0,
    ) -> Tuple[str, Dict[str, float]]:
        """
        Infer the most likely manipulation method from the R(z) zone pattern.

        Method
        ------
        Computes the cosine similarity between the observed R(z) vector and
        each method's prior-mean signature vector.  Similarity scores are
        converted to a probability distribution via softmax with a temperature
        parameter (higher temperature = sharper distribution).

        This implements a novel contribution: the zone anomaly pattern is used
        to classify the manipulation method, which then conditions the zone
        posterior Q(z).

        Args:
            R_scores   : (9,) array of fused zone scores.
            zone_names : list of zone name strings, matching R_scores order.
            temperature: softmax temperature.  Default 10.0 gives a reasonably
                         sharp distribution without collapsing to a delta mass.

        Returns:
            best_method      : name of the inferred manipulation method.
            method_probs     : {method: probability} summing to 1.
        """
        R_vec = np.asarray(R_scores, dtype=np.float64)
        R_norm = np.linalg.norm(R_vec) + 1e-8

        cosine_scores: Dict[str, float] = {}
        for method in self.methods:
            sig_vec = np.array(
                [self._signatures[method].get(z, 0.5) for z in zone_names],
                dtype=np.float64,
            )
            sig_norm = np.linalg.norm(sig_vec) + 1e-8
            cosine_scores[method] = float(np.dot(R_vec, sig_vec) / (R_norm * sig_norm))

        # Softmax with temperature
        vals     = np.array(list(cosine_scores.values()))
        shifted  = (vals - vals.max()) * temperature
        exp_vals = np.exp(shifted)
        probs    = exp_vals / exp_vals.sum()

        method_probs = {
            m: round(float(p), 4)
            for m, p in zip(self.methods, probs)
        }
        best_method = max(method_probs, key=method_probs.__getitem__)
        return best_method, method_probs

    # ------------------------------------------------------------------
    def compute_marginal_posteriors(
        self,
        R_scores: np.ndarray,
        zone_names: List[str],
        method_probs: Optional[Dict[str, float]] = None,
        weight: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Compute method-marginalised zone posteriors.

        Q(z) = Σ_m  P(method=m | R) · Q(z | method=m, R(z))

        When method_probs is None, a uniform prior over methods is used.
        When method_probs comes from infer_method(), the inferred probabilities
        are used as weights — this is the recommended inference path.

        Args:
            R_scores     : (9,) array of fused zone scores.
            zone_names   : list of zone name strings, matching R_scores order.
            method_probs : {method: probability} from infer_method(); or None
                           for uniform method prior.
            weight       : evidence weight (overrides instance default).

        Returns:
            Dict {zone_name: marginal_posterior_mean}.
        """
        if method_probs is None:
            method_probs = {m: 1.0 / len(self.methods) for m in self.methods}

        marginal: Dict[str, float] = {zone: 0.0 for zone in zone_names}
        for method in self.methods:
            p_m = float(method_probs.get(method, 0.0))
            if p_m < 1e-8:
                continue
            posteriors = self.compute_all_posteriors(
                R_scores, zone_names, method, weight
            )
            for zone in zone_names:
                marginal[zone] = marginal[zone] + p_m * posteriors.get(zone, 0.5)

        return {zone: round(float(v), 4) for zone, v in marginal.items()}

    # ------------------------------------------------------------------
    def signature_table(self) -> str:
        """Return a formatted table of prior means for all (method, zone) pairs."""
        col_w = 16
        zones_str = [z.replace("_", " ") for z in
                     list(next(iter(self.priors.values())).keys())]
        header = f"{'Method':{col_w}}" + "".join(f"{z:>14}" for z in zones_str)
        sep    = "-" * len(header)
        rows   = [header, sep]
        for method, zone_params in self.priors.items():
            vals = "".join(
                f"{a/(a+b):>14.2f}" for a, b in zone_params.values()
            )
            rows.append(f"{method:{col_w}}{vals}")
        return "\n".join(rows)
