"""
bayesian_calibration.py

Bayesian β calibration (Option 1) and Dirichlet artifact-type prior (Option 3)
for the SAEDF dual-score fusion pipeline.

Option 1 — BayesianBetaCalibrator
--------------------------------------
Places a Beta(1,1) (uniform) prior over β ∈ [0,1] and updates it with the
log-likelihood of correctly ordering fake/real frames under R(z,β).  The
posterior is evaluated on a dense grid of 1000 β values using population-
normalised A and S zone scores from the calibration set.

Outputs: MAP β, posterior mean, 95 % credible interval [2.5th, 97.5th
percentile].

Option 3 — DirichletArtifactPrior
--------------------------------------
For each semantic zone, maintains a Dirichlet distribution over the ranked list
of artifact types defined in ARTIFACT_ONTOLOGY.  Prior concentrations reflect
ontology rank order (rank-0 artifact gets the highest concentration).  The prior
is then updated with pseudo-counts derived from calibration fake-frame R(z)
scores: frames whose R(z) falls in a higher percentile contribute pseudo-counts
to lower-ranked (more common) artifact types.

At inference, the posterior expected probabilities are temperature-scaled by the
observed R(z) score so that a high-confidence zone concentrates on rank-0
artifacts while a borderline zone spreads mass across alternatives.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _normalize_scores(
    raw: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Z-score normalise then squash through sigmoid → values in (0, 1)."""
    z = (raw - mean) / (std + eps)
    return _sigmoid(z)


# ---------------------------------------------------------------------------
# Option 1 — Bayesian β calibration
# ---------------------------------------------------------------------------

class BayesianBetaCalibrator:
    """
    Posterior inference over the fusion weight β ∈ [0, 1].

    Prior
    -----
    Beta(1, 1) — uniform, encoding no strong prior belief about β.

    Likelihood
    ----------
    Binary cross-entropy on frame-level fake predictions:

        p_fake(i, β) = sigmoid( scale · mean_z R_i(z, β) )
        log L(β)     = Σ_i [ y_i · log p_fake + (1−y_i) · log(1−p_fake) ]

    where R_i(z, β) = β · A_norm_i(z) + (1−β) · S_norm_i(z).

    Posterior
    ---------
    Evaluated on a 1000-point grid; stored as a normalised PMF.
    Reported statistics: MAP, mean, 95 % credible interval.
    """

    def __init__(self, n_grid: int = 1000, scale: float = 6.0) -> None:
        """
        Args:
            n_grid : number of equally-spaced β values evaluated on [0, 1].
            scale  : logit scaling applied to mean_z R — higher values sharpen
                     the sigmoid and amplify separation between fake/real frames.
        """
        self.n_grid   = n_grid
        self.scale    = scale
        self.beta_grid: Optional[np.ndarray] = None
        self.posterior: Optional[np.ndarray] = None
        self._summary: Optional[dict]        = None

    # ------------------------------------------------------------------
    def fit(
        self,
        A_raw: np.ndarray,
        S_raw: np.ndarray,
        labels: np.ndarray,
        A_mean: np.ndarray,
        A_std: np.ndarray,
        S_mean: np.ndarray,
        S_std: np.ndarray,
    ) -> "BayesianBetaCalibrator":
        """
        Compute P(β | data) over the grid.

        Args:
            A_raw        : (N, 9) raw attention zone scores.
            S_raw        : (N, 9) raw gradient zone scores.
            labels       : (N,)  1 = fake, 0 = real.
            A_mean, A_std: (9,)  real-population statistics for attention.
            S_mean, S_std: (9,)  real-population statistics for gradient scores.

        Returns:
            self  (for method chaining).
        """
        A_norm = _normalize_scores(A_raw, A_mean, A_std)   # (N, 9)
        S_norm = _normalize_scores(S_raw, S_mean, S_std)   # (N, 9)

        beta_grid      = np.linspace(0.0, 1.0, self.n_grid)
        log_likelihood = np.zeros(self.n_grid)
        eps            = 1e-12

        for g, beta in enumerate(beta_grid):
            R      = beta * A_norm + (1.0 - beta) * S_norm   # (N, 9)
            p_fake = _sigmoid(self.scale * R.mean(axis=1))   # (N,)
            log_likelihood[g] = (
                labels * np.log(p_fake + eps)
                + (1.0 - labels) * np.log(1.0 - p_fake + eps)
            ).sum()

        # Beta(1,1) prior → log prior = 0 everywhere
        log_posterior  = log_likelihood
        log_posterior -= log_posterior.max()          # numerical stability
        posterior      = np.exp(log_posterior)
        posterior     /= posterior.sum()

        self.beta_grid = beta_grid
        self.posterior = posterior

        # Credible interval via CDF inversion
        cdf       = np.cumsum(posterior)
        ci_lower  = float(beta_grid[np.searchsorted(cdf, 0.025)])
        ci_upper  = float(beta_grid[np.searchsorted(cdf, 0.975)])
        map_beta  = float(beta_grid[np.argmax(posterior)])
        mean_beta = float((beta_grid * posterior).sum())
        var_beta  = float(((beta_grid - mean_beta) ** 2 * posterior).sum())
        std_beta  = float(np.sqrt(var_beta))

        self._summary = {
            "map":      round(map_beta,   4),
            "mean":     round(mean_beta,  4),
            "std":      round(std_beta,   4),
            "ci_lower": round(ci_lower,   4),
            "ci_upper": round(ci_upper,   4),
        }
        return self

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Return posterior summary statistics (MAP, mean, 95 % CI)."""
        if self._summary is None:
            raise RuntimeError("Call fit() before summary().")
        return dict(self._summary)

    # ------------------------------------------------------------------
    def print_report(self) -> None:
        """Pretty-print the Bayesian β calibration report."""
        if self._summary is None:
            raise RuntimeError("Call fit() before print_report().")
        s = self._summary
        print(
            "\nBayesian \u03b2 calibration"
            f"\n  MAP  \u03b2 = {s['map']:.4f}"
            f"\n  Mean \u03b2 = {s['mean']:.4f}"
            f"\n  95\u202f% CI : [{s['ci_lower']:.4f}, {s['ci_upper']:.4f}]"
        )


# ---------------------------------------------------------------------------
# R(z) credible-interval propagation (Option 4)
# ---------------------------------------------------------------------------

def compute_R_credible_interval(
    A_norm: np.ndarray,
    S_norm: np.ndarray,
    beta_mean: float,
    beta_std: float,
    coverage: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate β posterior uncertainty through the linear fusion R(z, β).

    Because R(z) = β·A_norm(z) + (1−β)·S_norm(z) is linear in β:

        E[R(z)]   = β_mean · A_norm(z) + (1 − β_mean) · S_norm(z)
        σ_R(z)    = β_std  · |A_norm(z) − S_norm(z)|

    The (1−α) credible interval is then:

        [E[R(z)] − z_{α/2} · σ_R(z),  E[R(z)] + z_{α/2} · σ_R(z)]

    clipped to [0, 1].

    Args:
        A_norm    : (9,) normalised attention zone scores for one frame.
        S_norm    : (9,) normalised gradient zone scores for one frame.
        beta_mean : posterior mean of β.
        beta_std  : posterior standard deviation of β.
        coverage  : credible interval coverage, default 0.95.

    Returns:
        ci_lower : (9,) lower bounds, clipped to [0, 1].
        ci_upper : (9,) upper bounds, clipped to [0, 1].
    """
    from scipy import stats as _stats  # lazy import — scipy is optional for the rest

    z_crit   = float(_stats.norm.ppf(0.5 + coverage / 2.0))
    R_mean   = beta_mean * A_norm + (1.0 - beta_mean) * S_norm
    R_margin = z_crit * beta_std * np.abs(A_norm - S_norm)

    ci_lower = np.clip(R_mean - R_margin, 0.0, 1.0)
    ci_upper = np.clip(R_mean + R_margin, 0.0, 1.0)
    return ci_lower.astype(np.float32), ci_upper.astype(np.float32)


# ---------------------------------------------------------------------------
# Option 3 — Dirichlet artifact-type prior
# ---------------------------------------------------------------------------

class DirichletArtifactPrior:
    """
    Per-zone Dirichlet distribution over artifact types.

    Prior
    -----
    For zone z with K artifact types ranked 0 … K−1 by the ontology:
        α_k^prior = K − k

    so rank-0 (most common) receives the highest concentration and rank-(K-1)
    the lowest.  This encodes the literature prior that rank-0 artifacts are
    most frequently observed for each zone.

    Update (calibration)
    --------------------
    Fake-frame R(z) scores are partitioned into K quantile buckets per zone.
    Frames in the top quantile (highest R) contribute a pseudo-count to α_0,
    the next bucket to α_1, and so on — reinforcing the prior that strong
    anomalies correspond to the most common artifact type while borderline
    anomalies spread uncertainty across alternatives.

    Inference
    ---------
    Expected Dirichlet probabilities E[θ_k] = α_k / Σ α are temperature-scaled
    by the current R(z) score, so that a high-confidence zone concentrates mass
    on rank-0 artifacts and a borderline zone retains uncertainty.
    """

    def __init__(self, artifact_ontology: Dict[str, List[str]]) -> None:
        self.ontology = artifact_ontology
        self._alpha: Dict[str, np.ndarray] = {}
        for zone, arts in artifact_ontology.items():
            K = len(arts)
            # α_k = K − k : rank-0 gets K, rank-(K-1) gets 1
            self._alpha[zone] = np.array(
                [K - k for k in range(K)], dtype=np.float64
            )

    # ------------------------------------------------------------------
    def update_from_calibration(
        self,
        R_norm_fake: np.ndarray,
        zone_names: List[str],
    ) -> "DirichletArtifactPrior":
        """
        Update α parameters with pseudo-counts from calibration fake frames.

        Args:
            R_norm_fake : (N_fake, 9) normalised R(z) scores for fake frames.
            zone_names  : zone name strings matching columns of R_norm_fake.

        Returns:
            self
        """
        for z_idx, zone in enumerate(zone_names):
            if zone not in self._alpha:
                continue
            K      = len(self._alpha[zone])
            scores = R_norm_fake[:, z_idx]   # (N_fake,)
            if len(scores) == 0:
                continue

            # K quantile boundaries → K buckets; high score → bucket 0
            quantile_edges = np.linspace(0.0, 100.0, K + 1)
            thresholds     = np.percentile(scores, quantile_edges)
            # digitize assigns bin indices 0 … K-1 from lowest to highest score
            bin_indices = np.digitize(scores, thresholds[1:-1])   # 0 … K-1
            # Invert: high score → bucket 0 (rank-0 artifact)
            bin_indices = (K - 1) - bin_indices.clip(0, K - 1)

            for b in bin_indices:
                self._alpha[zone][int(b)] += 1.0

        return self

    # ------------------------------------------------------------------
    def get_posterior_probs(
        self,
        zone: str,
        R_score: float,
        temperature: float = 2.0,
    ) -> np.ndarray:
        """
        Posterior probability vector over artifact types, scaled by R_score.

        logit_k = log(α_k) − rank_k · R_score · temperature

        High R_score (strong, confident anomaly) → large penalty on higher
        ranks → mass concentrates on the rank-0 (most common) artifact type.
        Low R_score (borderline anomaly) → penalty ≈ 0 → distribution stays
        close to the prior, i.e. retains more uncertainty across alternative
        artifact types.

        NOTE: this was previously implemented with a (1 - R_score) penalty
        term, which inverted the intended behaviour (it made LOW R_score
        concentrate on rank-0 and HIGH R_score spread out). Fixed to scale
        the penalty by R_score directly so it matches the class-level
        contract described above and in DirichletArtifactPrior's docstring.

        Args:
            zone        : zone name.
            R_score     : fused anomaly score in [0, 1].
            temperature : controls how strongly R_score modulates the distribution.

        Returns:
            p : (K,) probability vector summing to 1.
        """
        alpha  = self._alpha[zone]
        K      = len(alpha)
        ranks  = np.arange(K, dtype=np.float64)
        logits = np.log(alpha + 1e-12) - ranks * R_score * temperature
        logits -= logits.max()
        p       = np.exp(logits)
        p      /= p.sum()
        return p

    # ------------------------------------------------------------------
    def get_map_artifact(
        self,
        zone: str,
        R_score: float,
    ) -> Tuple[str, np.ndarray]:
        """
        Return the MAP artifact type and the full posterior probability vector.

        Args:
            zone    : zone name.
            R_score : fused anomaly score in [0, 1].

        Returns:
            (artifact_type_string, probability_vector of shape (K,))
        """
        probs   = self.get_posterior_probs(zone, R_score)
        map_idx = int(np.argmax(probs))
        return self.ontology[zone][map_idx], probs

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, list]:
        """Serialise posterior α parameters to a JSON-safe dict."""
        return {zone: alpha.tolist() for zone, alpha in self._alpha.items()}

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(
        cls,
        data: Dict[str, list],
        artifact_ontology: Dict[str, List[str]],
    ) -> "DirichletArtifactPrior":
        """Restore a DirichletArtifactPrior from a serialised dict."""
        obj = cls(artifact_ontology)
        for zone, alpha_list in data.items():
            if zone in obj._alpha:
                obj._alpha[zone] = np.array(alpha_list, dtype=np.float64)
        return obj
