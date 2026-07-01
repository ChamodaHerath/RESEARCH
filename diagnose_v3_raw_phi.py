import json
import numpy as np
import cv2

from attention_extractor import AttentionExtractor
from baseline_heatmap_generator import extract_patch_gradient_scores_raw
from semantic_patch_mapper import (
    compute_attention_zone_scores,
    compute_shap_zone_scores,
    ZONE_NAMES,
)


def main():
    d = json.load(open("outputs/ffpp_landmarks_cal.json"))
    extractor = AttentionExtractor(pretrained=True)

    A_raw_all, S_raw_all, labels = [], [], []
    try:
        total = len(d["frames"])
        for idx, (key, entry) in enumerate(d["frames"].items()):
            if idx % 100 == 0:
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

            phi_raw = extract_patch_gradient_scores_raw(image_bgr, extractor)
            S_raw_all.append(compute_shap_zone_scores(phi_raw, M))

            labels.append(1 if entry.get("label") == "fake" else 0)
    finally:
        extractor.close()

    A_raw_all = np.array(A_raw_all)
    S_raw_all = np.array(S_raw_all)
    labels = np.array(labels)

    np.savez("outputs/raw_scores_v3_unblurred.npz", A_raw=A_raw_all, S_raw=S_raw_all, labels=labels)

    print(f"\nTotal: {len(labels)}  Fake: {(labels==1).sum()}  Real: {(labels==0).sum()}")
    print("\nS(z) from UNBLURRED grad-rollout -- fake vs real:")
    for i, zone in enumerate(ZONE_NAMES):
        f, r = S_raw_all[labels==1,i].mean(), S_raw_all[labels==0,i].mean()
        rel = (f - r) / (r + 1e-8)
        print(f"  {zone:14s} fake={f:.4f}  real={r:.4f}  rel_diff={rel:+.1%}")


if __name__ == "__main__":
    main()