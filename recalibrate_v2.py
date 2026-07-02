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
    p.add_argument("--json", default="outputs/ffpp_landmarks_cal.json")
    p.add_argument("--model-name", default=DEFAULT_HF_DEEPFAKE_MODEL)
    p.add_argument("--output", default="configs/fusion_config.json")
    p.add_argument("--device", default=None)
    p.add_argument("--zone-threshold-quantile", type=float, default=0.95)
    p.add_argument("--max-flagged-zones", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    d = json.load(open(args.json))
    extractor = AttentionExtractor(model_name=args.model_name, pretrained=True, device=args.device)

    A_raw_all, S_raw_all, labels = [], [], []
    try:
        total = len(d["frames"])
        for idx, (key, entry) in enumerate(d["frames"].items()):
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
    zone_prior_model = ZonePriorModel()
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
    zone_thresholds = {
        zone: float(np.quantile(real_Q[:, i], args.zone_threshold_quantile))
        for i, zone in enumerate(ZONE_NAMES)
    }
    print(f"\nPer-zone reporting thresholds on Q(z) scale (real q={args.zone_threshold_quantile:.2f}):")
    for zone, threshold in zone_thresholds.items():
        print(f"  {zone:14s} threshold={threshold:.3f}")

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
    config = {
        "beta": float(best_beta),
        "model_name": args.model_name,
        "beta_posterior": beta_summary,
        "severity_thresholds": {"high": 0.75, "medium": 0.50, "low": 0.25},
        "zone_thresholds": zone_thresholds,
        "zone_thresholds_scale": "Q(z)_marginal_posterior",
        "zone_threshold_quantile": args.zone_threshold_quantile,
        "max_flagged_zones": args.max_flagged_zones,
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