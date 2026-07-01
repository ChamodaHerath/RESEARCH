import json
import numpy as np
import cv2

from attention_extractor import AttentionExtractor
from baseline_heatmap_generator import extract_patch_gradient_scores
from semantic_patch_mapper import (
    compute_attention_zone_scores,
    compute_shap_zone_scores,
    compute_population_zone_stats,
    normalize_zone_scores,
    ZONE_NAMES,
)


def main():
    d = json.load(open("outputs/ffpp_landmarks_test.json"))
    extractor = AttentionExtractor(pretrained=True)

    A_raw_all, S_raw_all, labels = [], [], []
    try:
        for key, entry in d["frames"].items():
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
            S_raw_all.append(compute_shap_zone_scores(phi, M))  # now area-normalised

            labels.append(1 if entry.get("label") == "fake" else 0)
    finally:
        extractor.close()

    A_raw_all = np.array(A_raw_all)
    S_raw_all = np.array(S_raw_all)
    labels = np.array(labels)

    # Calibrate population stats on REAL frames only ("normal" baseline)
    A_stats = compute_population_zone_stats(A_raw_all, labels)
    S_stats = compute_population_zone_stats(S_raw_all, labels)

    np.savez(
        "outputs/calibration_data_v2.npz",
        A_raw=A_raw_all, S_raw=S_raw_all, labels=labels,
        A_mean=A_stats["mean"], A_std=A_stats["std"],
        S_mean=S_stats["mean"], S_std=S_stats["std"],
        zone_names=ZONE_NAMES,
    )

    # Normalise every frame using the population stats
    A_norm = np.array([normalize_zone_scores(a, A_stats) for a in A_raw_all])
    S_norm = np.array([normalize_zone_scores(s, S_stats) for s in S_raw_all])

    print(f"Total: {len(labels)}  Fake: {(labels==1).sum()}  Real: {(labels==0).sum()}")
    print()
    print("S(z) area-normalised RAW -- fake vs real:")
    for i, zone in enumerate(ZONE_NAMES):
        f, r = S_raw_all[labels==1,i].mean(), S_raw_all[labels==0,i].mean()
        print(f"  {zone:14s} fake={f:.4f}  real={r:.4f}  rel_diff={((f-r)/(r+1e-8)):+.1%}")
    print()
    print("S(z) population-normalised -- fake vs real:")
    for i, zone in enumerate(ZONE_NAMES):
        f, r = S_norm[labels==1,i].mean(), S_norm[labels==0,i].mean()
        print(f"  {zone:14s} fake={f:.4f}  real={r:.4f}  diff={f-r:+.4f}")
    print()

    print("Beta sweep using population-normalised scores:")
    best_beta, best_sep = None, -1
    for beta in np.arange(0.0, 1.01, 0.1):
        R = beta * A_norm + (1 - beta) * S_norm
        sep = (R[labels==1].mean(axis=0) - R[labels==0].mean(axis=0)).mean()
        print(f"  beta={beta:.1f}  mean_separation={sep:+.4f}")
        if sep > best_sep:
            best_sep, best_beta = sep, beta
    print(f"\nBest beta: {best_beta:.1f} (separation={best_sep:.4f})")

    print("\nPer-zone R(z) at best beta:")
    R_best = best_beta * A_norm + (1 - best_beta) * S_norm
    for i, zone in enumerate(ZONE_NAMES):
        f, r = R_best[labels==1,i].mean(), R_best[labels==0,i].mean()
        print(f"  {zone:14s} fake={f:.3f}  real={r:.3f}  diff={f-r:+.3f}")


if __name__ == "__main__":
    main()