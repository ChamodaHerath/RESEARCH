"""
Unit tests for the ground-truth / statistical-rigor layer added to close the
Tier-1 research gaps:

  ffpp_utils.py         — method parsing, identity-safe splits
  eval_metrics.py       — video-level metrics, cluster bootstrap, per-method
  zone_localization.py  — mask-based zone ground truth, localization metrics,
                          deletion/insertion helpers

All tests are numpy/OpenCV-only (fast CI tier — no torch/transformers).
"""

from __future__ import annotations

import numpy as np
import pytest

from eval_metrics import (
    aggregate_video_level,
    binary_auc,
    cluster_bootstrap_ci,
    per_method_breakdown,
    standard_bootstrap_metrics,
    video_level_metrics,
)
from ffpp_utils import (
    ffpp_video_ids,
    find_ffpp_mask_video,
    hash_split_assignment,
    parse_ffpp_method,
    video_in_official_split,
    video_id_from_key,
)
from zone_localization import (
    binarize_manipulation_mask,
    blurred_baseline,
    combine_zone_masks,
    crop_and_resize_mask,
    curve_auc,
    is_likely_mask_frame,
    localization_counts,
    occlude_zones,
    pointing_game_hit,
    render_claimed_zones_overlay,
    render_claimed_zones_pixels_only,
    render_manipulation_overlay,
    render_manipulation_pixels_only,
    reveal_zones,
    summarize_localization,
    zone_gt_from_mask,
    zone_pixel_masks,
)
from semantic_patch_mapper import ZONE_NAMES


# ---------------------------------------------------------------------------
# ffpp_utils
# ---------------------------------------------------------------------------

class TestFFppUtils:
    def test_parse_method_from_frame_key(self):
        assert parse_ffpp_method("Deepfakes/000_003.mp4::f000000") == "Deepfakes"
        assert parse_ffpp_method("Face2Face/183_253.mp4::f000030") == "Face2Face"
        assert parse_ffpp_method("faceswap/001_002.mp4") == "FaceSwap"
        assert parse_ffpp_method("NeuralTextures/x.mp4") == "NeuralTextures"

    def test_parse_method_returns_none_for_real_and_celebdf(self):
        assert parse_ffpp_method("original/000.mp4::f000000") is None
        assert parse_ffpp_method("Celeb-synthesis/id1_id0_0007.mp4::f000010") is None

    def test_video_id_groups_frames_of_same_video(self):
        a = video_id_from_key("Deepfakes/000_003.mp4::f000000")
        b = video_id_from_key("Deepfakes/000_003.mp4::f000120")
        c = video_id_from_key("Deepfakes/001_003.mp4::f000000")
        assert a == b != c

    def test_ffpp_video_ids(self):
        assert ffpp_video_ids("Deepfakes/000_003.mp4") == ["000", "003"]
        assert ffpp_video_ids("original/000.mp4") == ["000"]
        assert ffpp_video_ids("Celeb-synthesis/id1_id0_0007.mp4") == []

    def test_hash_split_is_deterministic_and_identity_safe(self):
        # Original 000 and every fake with target id 000 must share a split.
        splits = {
            hash_split_assignment("original/000.mp4"),
            hash_split_assignment("Deepfakes/000_003.mp4"),
            hash_split_assignment("FaceSwap/000_999.mp4"),
        }
        assert len(splits) == 1

    def test_find_mask_video_rejects_manipulated_rgb_path(self, tmp_path):
        """Flat C23 manipulated videos must NOT be treated as masks."""
        fake_video = tmp_path / "Deepfakes" / "000_003.mp4"
        fake_video.parent.mkdir(parents=True)
        fake_video.write_bytes(b"not-a-real-video")
        assert find_ffpp_mask_video("Deepfakes/000_003.mp4", tmp_path) is None

    def test_find_mask_video_resolves_masks_subdirectory(self, tmp_path):
        mask_video = (
            tmp_path / "manipulated_sequences" / "Deepfakes" / "c23"
            / "masks" / "videos" / "000_003.mp4"
        )
        mask_video.parent.mkdir(parents=True, exist_ok=True)
        mask_video.write_bytes(b"mask")
        found = find_ffpp_mask_video("Deepfakes/000_003.mp4", tmp_path)
        assert found == mask_video

    def test_hash_split_is_deterministic(self):
        assert (hash_split_assignment("original/123.mp4")
                == hash_split_assignment("original/123.mp4"))

    def test_hash_split_fractions_validated(self):
        with pytest.raises(ValueError):
            hash_split_assignment("original/000.mp4", fractions=(0.5, 0.5, 0.5))

    def test_official_split_membership(self):
        pairs = {("000", "003"), ("100", "200")}
        assert video_in_official_split("Deepfakes/000_003.mp4", pairs)
        assert video_in_official_split("Deepfakes/003_000.mp4", pairs)  # either order
        assert video_in_official_split("original/100.mp4", pairs)
        assert not video_in_official_split("Deepfakes/000_100.mp4", pairs)  # cross pair
        assert not video_in_official_split("original/999.mp4", pairs)


# ---------------------------------------------------------------------------
# eval_metrics
# ---------------------------------------------------------------------------

class TestEvalMetrics:
    def test_binary_auc_perfect_and_random(self):
        assert binary_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
        assert binary_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5  # all tied
        assert binary_auc([1, 1], [0.5, 0.9]) is None                 # one class

    def test_binary_auc_matches_sklearn_when_available(self):
        sklearn_metrics = pytest.importorskip("sklearn.metrics")
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, size=200)
        y[0], y[1] = 0, 1  # both classes present
        s = rng.random(200)
        ours = binary_auc(y.tolist(), s.tolist())
        ref = float(sklearn_metrics.roc_auc_score(y, s))
        assert ours == pytest.approx(ref, abs=1e-9)

    def test_aggregate_video_level(self):
        y = [1, 1, 0, 0]
        p = [0.9, 0.7, 0.2, 0.4]
        v = ["fakeA", "fakeA", "realB", "realB"]
        labels, probs, order = aggregate_video_level(y, p, v)
        assert order == ["fakeA", "realB"]
        assert labels == [1, 0]
        assert probs[0] == pytest.approx(0.8)
        assert probs[1] == pytest.approx(0.3)

    def test_video_level_metrics_counts(self):
        y = [1, 1, 1, 0, 0, 0]
        p = [0.9, 0.8, 0.2, 0.1, 0.2, 0.9]
        v = ["f1", "f1", "f2", "r1", "r1", "r2"]
        m = video_level_metrics(y, p, v)
        assert m["n_videos"] == 4
        assert m["n_fake_videos"] == 2
        # f1 mean=0.85 (detected), f2=0.2 (missed), r1=0.15 (ok), r2=0.9 (FP)
        assert m["confusion_matrix"] == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}

    def test_cluster_bootstrap_ci_contains_point(self):
        rng = np.random.default_rng(1)
        n_videos, frames_per_video = 30, 5
        y, p, v = [], [], []
        for i in range(n_videos):
            label = i % 2
            for j in range(frames_per_video):
                y.append(label)
                p.append(np.clip(0.7 * label + 0.15 + rng.normal(0, 0.1), 0, 1))
                v.append(f"vid{i}")
        ci = cluster_bootstrap_ci(y, p, v, lambda yy, pp: binary_auc(yy, pp), n_boot=200)
        assert ci is not None
        assert ci["ci_lower"] <= ci["point"] <= ci["ci_upper"]
        assert 0.0 <= ci["ci_lower"] <= ci["ci_upper"] <= 1.0

    def test_standard_bootstrap_metrics_keys(self):
        y = [0, 1] * 20
        p = [0.2, 0.8] * 20
        v = [f"v{i // 2}" for i in range(40)]
        out = standard_bootstrap_metrics(y, p, v, n_boot=50)
        assert {"accuracy", "auc_roc", "recall"} <= set(out.keys())

    def test_per_method_breakdown(self):
        #                DF-fake DF-fake F2F-fake real  real
        y      = [1, 1, 1, 0, 0]
        p      = [0.9, 0.4, 0.8, 0.1, 0.2]
        vids   = ["a", "a", "b", "r1", "r2"]
        meths  = ["Deepfakes", "Deepfakes", "Face2Face", None, None]
        out = per_method_breakdown(y, p, vids, meths)
        assert set(out) == {"Deepfakes", "Face2Face"}
        assert out["Deepfakes"]["n_fake_frames"] == 2
        assert out["Deepfakes"]["frame_recall"] == pytest.approx(0.5)   # 0.9 yes, 0.4 no
        assert out["Face2Face"]["frame_recall"] == pytest.approx(1.0)
        assert out["Face2Face"]["auc_vs_reals"] == 1.0

    def test_per_method_breakdown_empty_for_non_ffpp(self):
        assert per_method_breakdown([1, 0], [0.9, 0.1], ["a", "b"], [None, None]) == {}


# ---------------------------------------------------------------------------
# zone_localization — ground truth
# ---------------------------------------------------------------------------

class TestZoneGroundTruth:
    def test_crop_and_resize_mask_geometry(self):
        # White square region in a 400x400 mask; crop it exactly.
        mask = np.zeros((400, 400), dtype=np.uint8)
        mask[100:200, 100:200] = 255
        out = crop_and_resize_mask(mask, [100, 100, 200, 200], target_size=224)
        assert out.shape == (224, 224)
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert out.mean() > 0.95  # crop contains only manipulated pixels

    def test_crop_and_resize_mask_invalid_box(self):
        with pytest.raises(ValueError):
            crop_and_resize_mask(np.zeros((100, 100), np.uint8), [50, 50, 40, 60])

    def test_is_likely_mask_frame_rejects_color_video(self):
        color = np.random.default_rng(0).integers(0, 256, (224, 224, 3), dtype=np.uint8)
        assert not is_likely_mask_frame(color)

    def test_is_likely_mask_frame_accepts_binary_mask(self):
        mask = np.zeros((224, 224), dtype=np.uint8)
        mask[50:170, 80:150] = 255
        assert is_likely_mask_frame(mask)

    def test_binarize_and_pixel_render_only_highlights_edited_region(self):
        face = np.random.default_rng(1).integers(40, 200, (224, 224, 3), dtype=np.uint8)
        mask = np.zeros((224, 224), dtype=np.float32)
        mask[80:120, 90:130] = 1.0
        binary = binarize_manipulation_mask(mask)
        assert binary.sum() == pytest.approx(40 * 40, rel=0.01)
        overlay = render_manipulation_overlay(face, mask)
        pixels = render_manipulation_pixels_only(face, mask)
        # Unedited pixels stay unchanged on overlay
        assert np.array_equal(overlay[~binary.astype(bool)], face[~binary.astype(bool)])
        # Pixels-only view is black outside edited region
        assert (pixels[~binary.astype(bool)] == 0).all()
        assert (pixels[binary.astype(bool)] != 0).any()

    def test_claimed_zones_pixels_only_highlights_zone_hulls(self, sample_landmarks_68):
        face = np.random.default_rng(2).integers(40, 200, (224, 224, 3), dtype=np.uint8)
        zmasks = zone_pixel_masks(sample_landmarks_68)
        claimed = combine_zone_masks(zmasks, ["mouth"])
        overlay = render_claimed_zones_overlay(face, sample_landmarks_68, ["mouth"])
        pixels = render_claimed_zones_pixels_only(face, sample_landmarks_68, ["mouth"])
        assert np.array_equal(overlay[~claimed], face[~claimed])
        assert (pixels[~claimed] == 0).all()
        assert (pixels[claimed] != 0).any()

    def test_zone_pixel_masks_cover_all_zones(self, sample_landmarks_68):
        masks = zone_pixel_masks(sample_landmarks_68)
        assert set(masks) == set(ZONE_NAMES)
        for zone, m in masks.items():
            assert m.shape == (224, 224)
            assert m.dtype == bool
            assert m.sum() > 0, f"zone {zone} rasterised to empty mask"

    def test_zone_gt_from_mask_full_and_empty(self, sample_landmarks_68):
        zmasks = zone_pixel_masks(sample_landmarks_68)
        all_manip = np.ones((224, 224), dtype=np.float32)
        cov, gt = zone_gt_from_mask(all_manip, zmasks)
        assert all(gt[z] for z in ZONE_NAMES)
        assert all(cov[z] == pytest.approx(1.0) for z in ZONE_NAMES)

        none_manip = np.zeros((224, 224), dtype=np.float32)
        cov, gt = zone_gt_from_mask(none_manip, zmasks)
        assert not any(gt.values())

    def test_zone_gt_selective_region(self, sample_landmarks_68):
        zmasks = zone_pixel_masks(sample_landmarks_68)
        # Manipulate exactly the mouth zone's pixels
        mask = np.zeros((224, 224), dtype=np.float32)
        mask[zmasks["mouth"]] = 1.0
        cov, gt = zone_gt_from_mask(mask, zmasks)
        assert gt["mouth"] is True
        # Zones spatially far from the mouth must stay negative
        assert gt["forehead"] is False
        assert gt["left_eye"] is False and gt["right_eye"] is False


# ---------------------------------------------------------------------------
# zone_localization — metrics
# ---------------------------------------------------------------------------

class TestLocalizationMetrics:
    def test_counts_and_summary(self):
        gt = {z: False for z in ZONE_NAMES}
        gt["mouth"] = gt["left_eye"] = True
        counts = localization_counts({"mouth", "chin_jaw"}, gt)
        assert counts == {"tp": 1, "fp": 1, "fn": 1, "tn": 6}
        s = summarize_localization(counts)
        assert s["precision"] == pytest.approx(0.5)
        assert s["recall"] == pytest.approx(0.5)
        assert s["iou"] == pytest.approx(1 / 3, abs=1e-4)

    def test_perfect_localization(self):
        gt = {z: z in {"mouth"} for z in ZONE_NAMES}
        s = summarize_localization(localization_counts({"mouth"}, gt))
        assert s["precision"] == s["recall"] == s["f1"] == s["iou"] == 1.0

    def test_pointing_game(self):
        gt = {z: z == "mouth" for z in ZONE_NAMES}
        scores_hit = {z: (0.9 if z == "mouth" else 0.1) for z in ZONE_NAMES}
        scores_miss = {z: (0.9 if z == "forehead" else 0.1) for z in ZONE_NAMES}
        assert pointing_game_hit(scores_hit, gt) is True
        assert pointing_game_hit(scores_miss, gt) is False
        # No GT-positive zone → excluded (None)
        assert pointing_game_hit(scores_hit, {z: False for z in ZONE_NAMES}) is None


# ---------------------------------------------------------------------------
# zone_localization — deletion / insertion helpers
# ---------------------------------------------------------------------------

class TestPerturbations:
    def test_occlude_zones_changes_only_zone_pixels(self, sample_landmarks_68):
        zmasks = zone_pixel_masks(sample_landmarks_68)
        img = np.full((224, 224, 3), 200, dtype=np.uint8)
        out = occlude_zones(img, zmasks, ["mouth"], fill=(0, 0, 0))
        assert (out[zmasks["mouth"]] == 0).all()
        outside = ~zmasks["mouth"]
        assert (out[outside] == 200).all()

    def test_reveal_zones_restores_original(self, sample_landmarks_68):
        zmasks = zone_pixel_masks(sample_landmarks_68)
        img = np.full((224, 224, 3), 200, dtype=np.uint8)
        base = np.zeros_like(img)
        out = reveal_zones(img, base, zmasks, ["left_eye"])
        assert (out[zmasks["left_eye"]] == 200).all()
        assert (out[~zmasks["left_eye"]] == 0).all()

    def test_blurred_baseline_shape_and_smoothing(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
        blurred = blurred_baseline(img)
        assert blurred.shape == img.shape
        assert float(np.std(blurred.astype(np.float64))) < float(np.std(img.astype(np.float64)))

    def test_curve_auc(self):
        assert curve_auc([1.0, 1.0, 1.0]) == pytest.approx(1.0)
        assert curve_auc([0.0, 0.0]) == pytest.approx(0.0)
        assert curve_auc([1.0, 0.0]) == pytest.approx(0.5)
        # Faster-dropping curve → smaller AUC (better deletion behaviour)
        fast = curve_auc([1.0, 0.1, 0.0, 0.0])
        slow = curve_auc([1.0, 0.9, 0.8, 0.0])
        assert fast < slow
