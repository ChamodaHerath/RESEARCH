import json
import numpy as np
import cv2

from attention_extractor import AttentionExtractor
from baseline_heatmap_generator import extract_patch_gradient_scores
from semantic_patch_mapper import (
    ARTIFACT_ONTOLOGY,
    compute_attention_zone_scores,
    compute_shap_zone_scores,
    compute_population_zone_stats,
    normalize_zone_scores,
    ZONE_NAMES,
)
from bayesian_calibration import BayesianBetaCalibrator, DirichletArtifactPrior


def main():
    d = json.load(open("outputs/ffpp_landmarks_cal.json"))
    extractor = AttentionExtractor(pretrained=True)

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

            model_input = extractor.preprocess_bgr_images_for_model(
                np.expand_dims(image_bgr, axis=0)
            )
            maps = extractor.extract_attention_maps(model_input)
            alpha = maps.cls_patch_attention[0]
            A_raw_all.append(compute_attention_zone_scores(alpha, M))

            phi = extract_patch_gradient_scores(image_bgr, extractor)
            S_raw_all.append(compute_shap_zone_scores(phi, M))

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
        "beta_posterior": beta_summary,
        "severity_thresholds": {"high": 0.75, "medium": 0.50, "low": 0.25},
        "A_stats": {"mean": A_stats["mean"].tolist(), "std": A_stats["std"].tolist()},
        "S_stats": {"mean": S_stats["mean"].tolist(), "std": S_stats["std"].tolist()},
        "zone_names": ZONE_NAMES,
        "artifact_posteriors": artifact_prior.to_dict(),
        "calibration_n_frames": int(len(labels)),
        "calibration_n_fake": int((labels==1).sum()),
        "calibration_n_real": int((labels==0).sum()),
    }
    with open("configs/fusion_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("\nSaved configs/fusion_config.json")


if __name__ == "__main__":
    main()