import argparse
import json
import numpy as np
import cv2

from attention_extractor import AttentionExtractor, DEFAULT_HF_DEEPFAKE_MODEL
from baseline_heatmap_generator import extract_full_pipeline_inputs
from semantic_patch_mapper import (
    ARTIFACT_ONTOLOGY,
    compute_attention_zone_scores,
    compute_shap_zone_scores,
    compute_population_zone_stats,
    normalize_zone_scores,
    ZONE_NAMES,
)
from bayesian_calibration import BayesianBetaCalibrator, DirichletArtifactPrior
from zone_prior import ZonePriorModel


def parse_args():
    p = argparse.ArgumentParser(description="Recalibrate SAEDF fusion for a selected detector.")
    p.add_argument(
        "--json",
        default="outputs/ffpp_landmarks_cal.json",
        help="Calibration landmark JSON. Must come from val/train "
             "(build with build_cal_from_split.py) — never the test split.",
    )
    p.add_argument("--model-name", default=DEFAULT_HF_DEEPFAKE_MODEL)
    p.add_argument("--output", default="configs/fusion_config.json")
    p.add_argument("--device", default=None)
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on frames processed (even stride over the JSON). "
             "Prefer build_cal_from_split.py for video-level subsampling.",
    )
    p.add_argument(
        "--zone-threshold-quantile",
        type=float,
        default=0.85,
        help="Real-face Q(z) quantile used as the per-zone flagging cutoff. "
             "0.85 ≈ ≤15%% real-face pass rate per zone (accuracy-first default). "
             "0.95 is stricter but rarely emits Fake claims; 0.80 is demo-looser.",
    )
    p.add_argument(
        "--min-fake-probability",
        type=float,
        default=0.70,
        help="Classifier P(fake) must exceed this before any zone is flagged. "
             "Prevents weak/uncertain Fake verdicts from emitting forensic claims.",
    )
    p.add_argument(
        "--causal-min-drop",
        type=float,
        default=0.03,
        help="Minimum occlusion drop in P(fake) required to retain a claim "
             "at inference (written into fusion_config for the server).",
    )
    p.add_argument(
        "--evidence-weight",
        type=float,
        default=10.0,
        help="w in Q=(α+R·w)/(α+β+w). Higher w lets R(z) move Q away from "
             "the literature prior. Default 10 (was hardcoded 3).",
    )
    p.add_argument(
        "--threshold-ceiling-margin",
        type=float,
        default=0.05,
        help="Keep each zone threshold ≤ max_reachable_Q(z) − margin so a "
             "zone with strong R(z) can still clear the bar.",
    )
    p.add_argument("--max-flagged-zones", type=int, default=3)
    return p.parse_args()


def _max_reachable_q(zone_prior_model: ZonePriorModel, zone: str) -> float:
    """Upper bound on marginal Q(z) when R(z)=1 under equal method mix."""
    w = float(zone_prior_model.evidence_weight)
    vals = []
    for method in zone_prior_model.methods:
        a, b = zone_prior_model.priors[method][zone]
        vals.append((a + 1.0 * w) / (a + b + w))
    return float(np.max(vals))  # best-case method; conservative ceiling uses max


def main():
    args = parse_args()
    d = json.load(open(args.json))
    extractor = AttentionExtractor(model_name=args.model_name, pretrained=True, device=args.device)

    A_raw_all, S_raw_all, labels = [], [], []
    try:
        items = list(d["frames"].items())
        if args.max_frames is not None and len(items) > args.max_frames:
            step = len(items) / args.max_frames
            items = [items[int(i * step)] for i in range(args.max_frames)]
            print(f"Subsampled to {len(items)} frames (--max-frames={args.max_frames})")
        total = len(items)
        for idx, (key, entry) in enumerate(items):
            if idx % 50 == 0:
                print(f"  {idx}/{total}")
            image_bgr = cv2.imread(entry["face_image"])
            if image_bgr is None or image_bgr.shape != (224, 224, 3):
                continue
            M = np.array(entry["zone_patch_matrix"], dtype=np.float32)

            inputs = extract_full_pipeline_inputs(image_bgr, extractor)
            A_raw_all.append(compute_attention_zone_scores(inputs["alpha"], M))
            S_raw_all.append(compute_shap_zone_scores(inputs["phi"], M))

            labels.append(1 if entry.get("label") == "fake" else 0)
    finally:
        extractor.close()

    A_raw_all = np.array(A_raw_all)
    S_raw_all = np.array(S_raw_all)
    labels = np.array(labels)

    A_stats = compute_population_zone_stats(A_raw_all, labels)
    S_stats = compute_population_zone_stats(S_raw_all, labels)

    A_norm = np.array([normalize_zone_scores(a, A_stats) for a in A_raw_all])
    S_norm = np.array([normalize_zone_scores(s, S_stats) for s in S_raw_all])

    print(f"\nTotal: {len(labels)}  Fake: {(labels==1).sum()}  Real: {(labels==0).sum()}")

    # ------------------------------------------------------------------
    # Option 1 — Bayesian β calibration
    # ------------------------------------------------------------------
    print("\nBayesian \u03b2 calibration (1 000-point grid, Beta(1,1) prior) ...")
    calibrator = BayesianBetaCalibrator(n_grid=1000, scale=6.0)
    calibrator.fit(
        A_raw=A_raw_all,
        S_raw=S_raw_all,
        labels=labels,
        A_mean=A_stats["mean"],
        A_std=A_stats["std"],
        S_mean=S_stats["mean"],
        S_std=S_stats["std"],
    )
    calibrator.print_report()
    beta_summary = calibrator.summary()
    best_beta = beta_summary["map"]

    print("\nPer-zone R(z) at MAP \u03b2:")
    R_best = best_beta * A_norm + (1 - best_beta) * S_norm
    for i, zone in enumerate(ZONE_NAMES):
        f, r = R_best[labels==1, i].mean(), R_best[labels==0, i].mean()
        print(f"  {zone:14s} fake={f:.3f}  real={r:.3f}  diff={f-r:+.3f}")

    # ------------------------------------------------------------------
    # Option 2 — Bayesian zone-prior posterior Q(z)
    # ------------------------------------------------------------------
    # NOTE: zone flagging at inference time (build_concept_vectors_from_R)
    # uses Q(z) -- the manipulation-type-conditioned posterior -- not the
    # raw fused R(z), because ZonePriorModel is always auto-loaded by
    # SemanticExplanationGenerator. Q(z) is a shrinkage estimator pulled
    # toward literature priors (Beta(alpha,beta) means), so it lives on a
    # different, generally more compressed scale than R(z). Thresholds
    # MUST therefore be calibrated on the Q(z) distribution itself, not on
    # R(z) -- otherwise an R(z)-calibrated threshold can sit above the
    # structural ceiling of Q(z) for every zone/method combination and no
    # zone is ever flagged (silently breaking the explanation head).
    zone_prior_model = ZonePriorModel(evidence_weight=float(args.evidence_weight))
    print(f"\nZone-prior evidence_weight w = {zone_prior_model.evidence_weight}")
    Q_all = np.zeros_like(R_best)
    for row_idx in range(R_best.shape[0]):
        _, method_probs = zone_prior_model.infer_method(R_best[row_idx], ZONE_NAMES)
        Q_dict = zone_prior_model.compute_marginal_posteriors(
            R_best[row_idx], ZONE_NAMES, method_probs=method_probs
        )
        Q_all[row_idx] = [Q_dict[z] for z in ZONE_NAMES]

    print("\nPer-zone Q(z) (Bayesian zone-prior posterior) at MAP \u03b2:")
    for i, zone in enumerate(ZONE_NAMES):
        f, r = Q_all[labels == 1, i].mean(), Q_all[labels == 0, i].mean()
        print(f"  {zone:14s} fake={f:.3f}  real={r:.3f}  diff={f-r:+.3f}")

    real_Q = Q_all[labels == 0]
    fake_Q = Q_all[labels == 1]
    zone_thresholds = {}
    print(
        f"\nPer-zone reporting thresholds on Q(z) "
        f"(real q={args.zone_threshold_quantile:.2f}, "
        f"ceiling margin={args.threshold_ceiling_margin:.2f}):"
    )
    for i, zone in enumerate(ZONE_NAMES):
        raw_thr = float(np.quantile(real_Q[:, i], args.zone_threshold_quantile))
        q_ceil = _max_reachable_q(zone_prior_model, zone)
        capped = min(raw_thr, q_ceil - float(args.threshold_ceiling_margin))
        # Never set a threshold below the real median — keep some selectivity
        floor = float(np.median(real_Q[:, i]))
        thr = max(capped, floor)
        # But still never above the reachable ceiling − margin
        thr = min(thr, q_ceil - float(args.threshold_ceiling_margin))
        zone_thresholds[zone] = float(thr)
        fake_pass = float((fake_Q[:, i] > thr).mean())
        real_pass = float((real_Q[:, i] > thr).mean())
        print(
            f"  {zone:14s} raw_q={raw_thr:.3f}  ceil={q_ceil:.3f}  "
            f"thr={thr:.3f}  fake_pass={fake_pass:.1%}  real_pass={real_pass:.1%}"
        )

    # ------------------------------------------------------------------
    # Option 3 — Dirichlet artifact-type prior
    # ------------------------------------------------------------------
    print("\nFitting Dirichlet artifact prior ...")
    artifact_prior = DirichletArtifactPrior(ARTIFACT_ONTOLOGY)
    R_norm_fake = R_best[labels == 1]   # (N_fake, 9) — normalised R at MAP β
    artifact_prior.update_from_calibration(R_norm_fake, zone_names=ZONE_NAMES)

    # Brief per-zone report: MAP artifact type and its posterior probability
    print("\nPer-zone MAP artifact (posterior expected probability):")
    for zone in ZONE_NAMES:
        art, probs = artifact_prior.get_map_artifact(zone, R_score=0.7)
        print(f"  {zone:14s}  '{art}'  p={probs[0]:.3f}")

    # ------------------------------------------------------------------
    # Save the extended fusion config
    # ------------------------------------------------------------------
    # Expected real-face pass rate under the chosen quantile (empirical).
    real_pass_rates = {
        zone: float((real_Q[:, i] > zone_thresholds[zone]).mean())
        for i, zone in enumerate(ZONE_NAMES)
    }
    fake_pass_rates = {
        zone: float((fake_Q[:, i] > zone_thresholds[zone]).mean())
        for i, zone in enumerate(ZONE_NAMES)
    }

    config = {
        "beta": float(best_beta),
        "model_name": args.model_name,
        "beta_posterior": beta_summary,
        "severity_thresholds": {"high": 0.75, "medium": 0.50, "low": 0.25},
        "zone_thresholds": zone_thresholds,
        "zone_thresholds_scale": "Q(z)_marginal_posterior",
        "zone_threshold_quantile": args.zone_threshold_quantile,
        "evidence_weight": float(args.evidence_weight),
        "threshold_ceiling_margin": float(args.threshold_ceiling_margin),
        "max_flagged_zones": args.max_flagged_zones,
        "min_fake_probability_for_explanations": float(args.min_fake_probability),
        "causal_min_drop": float(args.causal_min_drop),
        "beta_for_fusion": "posterior_mean",
        "research_policy": {
            "name": "precision_first_v1",
            "rules": [
                "No zone flags when classifier label is Real",
                "No zone flags when P(fake) < min_fake_probability_for_explanations",
                "Flag only when Q(z) > per-zone calibrated threshold (strict, real q85)",
                "Live demos: retain claim only if causal occlusion drops P(fake)",
                "Fusion R(z) uses β posterior mean so attention and gradient both contribute",
            ],
            "claim_scope": (
                "Flagged zones are high-precision localisation of model anomaly "
                "mass. On FF++ masks, flagged-zone precision and pointing-game "
                "accuracy are the primary localisation claims — not that every "
                "manipulated pixel is recovered (recall is intentionally low)."
            ),
            "calibration_real_pass_rate_mean": round(
                float(np.mean(list(real_pass_rates.values()))), 4
            ),
            "calibration_fake_pass_rate_mean": round(
                float(np.mean(list(fake_pass_rates.values()))), 4
            ),
            "per_zone_real_pass_rate": {
                z: round(v, 4) for z, v in real_pass_rates.items()
            },
            "per_zone_fake_pass_rate": {
                z: round(v, 4) for z, v in fake_pass_rates.items()
            },
        },
        "A_stats": {"mean": A_stats["mean"].tolist(), "std": A_stats["std"].tolist()},
        "S_stats": {"mean": S_stats["mean"].tolist(), "std": S_stats["std"].tolist()},
        "zone_names": ZONE_NAMES,
        "artifact_posteriors": artifact_prior.to_dict(),
        "calibration_n_frames": int(len(labels)),
        "calibration_n_fake": int((labels==1).sum()),
        "calibration_n_real": int((labels==0).sum()),
    }
    with open(args.output, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    from console_utf8 import ensure_utf8_stdout
    ensure_utf8_stdout()
    main()