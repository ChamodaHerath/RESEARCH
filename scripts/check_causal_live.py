"""Quick live check: causal occlusion drops with the real ViT classifier."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from attention_extractor import AttentionExtractor, DEFAULT_HF_DEEPFAKE_MODEL
from pipeline import classifier_fake_probabilities
from semantic_patch_mapper import (
    ConceptVector,
    ExplanationOutput,
    ZONE_NAMES,
    filter_explanations_by_causal_correctness,
)
from zone_localization import measure_zone_causal_drops, zone_passes_causal_check

MIN_DROP = 0.03


def main() -> None:
    rng = np.random.default_rng(42)
    face = rng.integers(40, 200, (224, 224, 3), dtype=np.uint8)
    landmarks = []
    for i in range(68):
        ang = 2 * np.pi * i / 68
        landmarks.append([112 + 80 * np.cos(ang), 112 + 90 * np.sin(ang)])
    landmarks_arr = np.array(landmarks, dtype=np.float32)

    print("Loading ViT extractor on CPU...")
    ext = AttentionExtractor(model_name=DEFAULT_HF_DEEPFAKE_MODEL, device="cpu")

    def prob_fn(imgs):
        return classifier_fake_probabilities(ext, imgs)

    baseline_p = prob_fn([face])[0]
    print(f"Baseline P(fake): {baseline_p:.4f}")

    drops = measure_zone_causal_drops(prob_fn, face, landmarks_arr, zones=ZONE_NAMES)
    passed = [z for z, d in drops.items() if zone_passes_causal_check(d["drop"], MIN_DROP)]
    print(f"Zones tested: {len(drops)}")
    print(f"Zones passing causal (drop >= {MIN_DROP}): {len(passed)}")
    if passed:
        print("  passed:", ", ".join(passed))

    ranked = sorted(drops.items(), key=lambda kv: kv[1]["drop"], reverse=True)
    print("\nTop 5 causal drops:")
    for z, info in ranked[:5]:
        ok = zone_passes_causal_check(info["drop"], MIN_DROP)
        print(
            f"  {z:12s}  drop={info['drop']:+.4f}  "
            f"(before={info['baseline']:.4f} after={info['occluded']:.4f})  pass={ok}"
        )

    cvs = [
        ConceptVector(
            zone=z,
            artifact_type="test artifact",
            severity="high",
            confidence=0.8,
            R_score=0.8,
        )
        for z in ["mouth", "nose_bridge", "forehead"]
    ]
    exp = ExplanationOutput(
        label="Fake",
        fake_probability=baseline_p,
        concept_vectors=cvs,
        nl_explanations=["a", "b", "c"],
        faithfulness_passed=[True] * 3,
        R_scores=np.full(9, 0.5, dtype=np.float32),
    )
    out = filter_explanations_by_causal_correctness(
        exp, face, landmarks_arr, prob_fn, min_drop=MIN_DROP, hard_filter=False
    )
    print("\nFilter (soft, demo mode):")
    print("  causal_correctness_passed:", out.causal_correctness_passed)
    for z in ["mouth", "nose_bridge", "forehead"]:
        d = out.causal_drop_by_zone.get(z)
        if d is not None:
            print(f"  {z}: drop={d:+.4f}")


if __name__ == "__main__":
    main()
