"""
evaluate_zone_localization.py — validate zone attributions against the
FaceForensics++ manipulation-mask ground truth, and measure causal
faithfulness via deletion/insertion curves.

This closes the "faithfulness ≠ forensic correctness" gap: the existing
faithfulness verifier only checks that generated sentences match the internal
ConceptVector; this script checks that the flagged zones correspond to the
regions FF++ actually manipulated.

Ground truth
------------
FF++ provides a mask video per fake clip (the altered pixel region). For each
evaluated fake frame we:
  1. read the same frame index from the mask video,
  2. crop it with the SAME face box stored by preprocess_landmarks.py
     (entry["crop_box"]) and resize to 224x224,
  3. rasterise the 9 zone convex hulls from the frame's landmarks,
  4. label a zone ground-truth-manipulated when >= --coverage-threshold of
     its pixels fall inside the manipulation mask.

Reported metrics
----------------
  Localization  : zone-level precision / recall / F1 / IoU of flagged zones
                  vs mask ground truth (micro over all zone decisions, plus
                  per-zone and per-method breakdowns).
  Pointing game : fraction of fake frames whose top-scored zone is a
                  ground-truth-manipulated zone.
  Zone AUC      : rank-AUC of the per-zone score (Q(z) when available, else
                  R(z)) against the zone ground-truth labels.
  Deletion AUC  : occlude zones in decreasing score order; lower area under
                  the fake-probability curve = attributions are causally
                  faithful. Compared against a random-order baseline.
  Insertion AUC : reveal zones onto a blurred baseline in decreasing score
                  order; higher = better. Compared against random order.

Usage
-----
    python evaluate_zone_localization.py \
        --json outputs/ffpp_landmarks_test.json \
        --ffpp-root datasets/FaceForensics++/FaceForensics++_C23 \
        --masks-root datasets/FaceForensics++/masks \
        --output outputs/zone_localization_ffpp.json

Mask lookup order for a fake video 'Deepfakes/000_003.mp4':
    <masks-root>/Deepfakes/000_003.mp4
    <masks-root>/Deepfakes/videos/000_003.mp4
    <masks-root>/manipulated_sequences/Deepfakes/masks/videos/000_003.mp4
    <ffpp-root>/<same three patterns>
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from ffpp_utils import find_ffpp_mask_video, parse_ffpp_method, video_path_from_key
from landmark_io import load_landmark_json
from pipeline import SAEDFPipeline
from semantic_patch_mapper import ZONE_NAMES
from zone_localization import (
    binary_auc,
    blurred_baseline,
    crop_and_resize_mask,
    curve_auc,
    is_likely_mask_frame,
    localization_counts,
    occlude_zones,
    pointing_game_hit,
    reveal_zones,
    summarize_localization,
    zone_gt_from_mask,
    zone_pixel_masks,
)

DEFAULT_JSON = "outputs/ffpp_landmarks_test_full.json"
DEFAULT_OUTPUT = "outputs/zone_localization_ffpp.json"
DEFAULT_MASKS_ROOT = Path("datasets/FaceForensics++/masks")


class MaskVideoCache:
    """Reads mask frames by index, keeping the most recent VideoCapture open."""

    def __init__(self) -> None:
        self._path: Optional[Path] = None
        self._cap: Optional[cv2.VideoCapture] = None

    def read_frame(self, mask_path: Path, frame_index: int) -> Optional[np.ndarray]:
        if self._path != mask_path:
            self.close()
            cap = cv2.VideoCapture(str(mask_path))
            if not cap.isOpened():
                return None
            self._path, self._cap = mask_path, cap
        assert self._cap is not None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._path, self._cap = None, None


# ---------------------------------------------------------------------------
# Classifier-only forward pass (for deletion/insertion curves)
# ---------------------------------------------------------------------------

def batch_fake_probabilities(pipe: SAEDFPipeline, images_bgr: List[np.ndarray]) -> List[float]:
    """Run a plain classification forward pass on a batch of 224x224 BGR images."""
    import torch

    extractor = pipe.extractor
    batch = np.stack([img.astype(np.uint8) for img in images_bgr], axis=0)
    model_input = extractor.preprocess_bgr_images_for_model(batch).to(extractor.device)
    with torch.no_grad():
        out = extractor.model(model_input)
    logits = out.logits if hasattr(out, "logits") else out
    probs = torch.softmax(logits, dim=-1)
    num_classes = logits.shape[-1]
    if getattr(extractor, "_fake_class_index", None) is not None:
        fake_idx = int(extractor._fake_class_index)
    else:
        fake_idx = extractor.get_fake_class_index(num_classes=num_classes)
    return [float(p) for p in probs[:, fake_idx].detach().cpu().numpy()]


def deletion_insertion_curves(
    pipe: SAEDFPipeline,
    image_bgr: np.ndarray,
    zone_masks: Dict[str, np.ndarray],
    zone_order: List[str],
) -> Tuple[List[float], List[float]]:
    """
    Deletion: occlude zones cumulatively in `zone_order`; probs[k] = fake
    probability with the first k zones occluded (k = 0 … 9).
    Insertion: reveal zones cumulatively onto a blurred baseline.
    """
    deletion_images = [image_bgr]
    for k in range(1, len(zone_order) + 1):
        deletion_images.append(occlude_zones(image_bgr, zone_masks, zone_order[:k]))

    baseline = blurred_baseline(image_bgr)
    insertion_images = [baseline]
    for k in range(1, len(zone_order) + 1):
        insertion_images.append(reveal_zones(image_bgr, baseline, zone_masks, zone_order[:k]))

    probs = batch_fake_probabilities(pipe, deletion_images + insertion_images)
    n = len(deletion_images)
    return probs[:n], probs[n:]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Zone-localization ground-truth evaluation on FF++ masks."
    )
    p.add_argument("--json", type=Path, default=Path(DEFAULT_JSON))
    p.add_argument("--ffpp-root", type=Path, default=None,
                   help="FF++ root (also searched for mask videos).")
    p.add_argument("--masks-root", type=Path, default=DEFAULT_MASKS_ROOT,
                   help="Root directory containing FF++ mask videos.")
    p.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    p.add_argument("--config", type=Path, default=Path("configs/fusion_config.json"))
    p.add_argument("--model-name", type=str, default=None)
    p.add_argument("--min-severity", type=str, default="medium",
                   choices=["low", "medium", "high"])
    p.add_argument("--coverage-threshold", type=float, default=0.30,
                   help="Zone GT-positive when >= this fraction of its pixels are manipulated.")
    p.add_argument("--mask-threshold", type=float, default=0.10,
                   help="Mask pixel intensity (0-1) above which a pixel counts as manipulated.")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Evaluate at most N fake frames (smoke test).")
    p.add_argument("--sample-every", type=int, default=1,
                   help="Evaluate every Nth fake frame (e.g. 4 -> quarter of "
                        "the frames; ~4x faster with near-identical metrics "
                        "because frames within a video are highly correlated).")
    p.add_argument("--checkpoint-every", type=int, default=1000,
                   help="Write partial results to the output path every N "
                        "evaluated frames so an interrupted run still leaves "
                        "usable numbers. 0 disables checkpointing.")
    p.add_argument("--perturbation-max-frames", type=int, default=200,
                   help="Run deletion/insertion curves on at most this many "
                        "frames (full test set is very slow). Use 0 to skip.")
    p.add_argument("--random-permutations", type=int, default=3,
                   help="Random zone orderings per frame for the deletion/insertion baseline.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    if args.masks_root is None and args.ffpp_root is None:
        raise SystemExit("Provide --masks-root and/or --ffpp-root so mask videos can be located.")

    data = load_landmark_json(args.json)
    frames: Dict[str, Dict] = data.get("frames", data)

    fake_entries = [
        (key, entry) for key, entry in frames.items() if entry.get("label") == "fake"
    ]
    if args.sample_every > 1:
        fake_entries = fake_entries[:: args.sample_every]
    if args.max_frames is not None:
        fake_entries = fake_entries[: args.max_frames]
    print(f"\nZone-localization evaluation: {len(fake_entries)} fake frames "
          f"from {args.json} (sample-every={args.sample_every})", flush=True)

    mask_cache = MaskVideoCache()

    # Accumulators
    micro_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    per_zone_counts: Dict[str, Dict[str, int]] = {
        z: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for z in ZONE_NAMES
    }
    per_method_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    )
    pointing_hits: List[bool] = []
    zone_auc_labels: List[int] = []
    zone_auc_scores: List[float] = []
    per_zone_auc_data: Dict[str, Tuple[List[int], List[float]]] = {
        z: ([], []) for z in ZONE_NAMES
    }
    deletion_aucs: List[float] = []
    insertion_aucs: List[float] = []
    deletion_aucs_random: List[float] = []
    insertion_aucs_random: List[float] = []

    n_no_mask = n_no_crop_box = n_bad_mask = n_errors = n_evaluated = 0
    n_perturbation = 0
    t0 = time.time()

    def _mean(xs: List[float]) -> Optional[float]:
        return round(float(np.mean(xs)), 4) if xs else None

    def _build_results(partial: bool) -> Dict:
        per_zone_summary = {}
        for zone in ZONE_NAMES:
            gt_l, sc = per_zone_auc_data[zone]
            per_zone_summary[zone] = {
                **summarize_localization(per_zone_counts[zone]),
                "counts": dict(per_zone_counts[zone]),
                "gt_positive_rate": round(float(np.mean(gt_l)), 4) if gt_l else None,
                "zone_auc": (round(a, 4) if (a := binary_auc(gt_l, sc)) is not None else None),
            }
        return {
            "evaluation_metadata": {
                "landmark_json": str(args.json),
                "masks_root": str(args.masks_root) if args.masks_root else None,
                "ffpp_root": str(args.ffpp_root) if args.ffpp_root else None,
                "model_name": args.model_name or "pipeline_default",
                "coverage_threshold": args.coverage_threshold,
                "mask_threshold": args.mask_threshold,
                "sample_every": args.sample_every,
                "partial_checkpoint": partial,
                "fake_frames_total": len(fake_entries),
                "fake_frames_evaluated": n_evaluated,
                "skipped_no_mask": n_no_mask,
                "skipped_bad_mask_frame": n_bad_mask,
                "skipped_no_crop_box": n_no_crop_box,
                "skipped_errors": n_errors,
                "elapsed_seconds": round(time.time() - t0, 1),
                "note_crop_box": (
                    "Frames without entry['crop_box'] are skipped: re-run "
                    "preprocess_landmarks.py to regenerate the landmark JSON with "
                    "crop boxes if skipped_no_crop_box > 0."
                ),
            },
            "localization": {
                "micro": {**summarize_localization(micro_counts), "counts": dict(micro_counts)},
                "per_zone": per_zone_summary,
                "per_method": {
                    m: {**summarize_localization(c), "counts": dict(c)}
                    for m, c in sorted(per_method_counts.items())
                },
                "pointing_game_accuracy": _mean([1.0 if h else 0.0 for h in pointing_hits]),
                "pointing_game_n": len(pointing_hits),
                "zone_score_auc_micro": (
                    round(a, 4) if (a := binary_auc(zone_auc_labels, zone_auc_scores)) is not None
                    else None
                ),
            },
            "causal_faithfulness": {
                "deletion_auc_attribution_order": _mean(deletion_aucs),
                "deletion_auc_random_order": _mean(deletion_aucs_random),
                "insertion_auc_attribution_order": _mean(insertion_aucs),
                "insertion_auc_random_order": _mean(insertion_aucs_random),
                "n_frames": len(deletion_aucs),
                "note": (
                    "Deletion: zones occluded in decreasing attribution order; lower "
                    "AUC than random = flagged zones causally drive the fake score. "
                    "Insertion: zones revealed onto a blurred baseline; higher AUC "
                    "than random = better."
                ),
            },
        }

    def _write_results(results: Dict) -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(args.output)

    # Disable the runtime causal-correctness gate so thesis localization
    # metrics measure the explanation head's native zone flagging (Q/R
    # thresholds), not the post-hoc occlusion filter used in the live demo.
    pipe_kwargs = {
        "config_path": str(args.config),
        "min_severity": args.min_severity,
        "causal_correctness": False,
    }
    if args.model_name:
        pipe_kwargs["model_name"] = args.model_name

    with SAEDFPipeline(**pipe_kwargs) as pipe:
        for i, (key, entry) in enumerate(fake_entries):
            if i % 20 == 0:
                print(f"  [{i:4d}/{len(fake_entries)}]  elapsed: {time.time() - t0:.1f}s",
                      flush=True)

            crop_box = entry.get("crop_box")
            if crop_box is None:
                n_no_crop_box += 1
                continue

            video_rel = entry.get("video_path") or video_path_from_key(key)
            frame_index = int(entry.get("frame_index", 0))
            mask_path = find_ffpp_mask_video(video_rel, args.masks_root, args.ffpp_root)
            if mask_path is None:
                n_no_mask += 1
                continue
            mask_frame = mask_cache.read_frame(mask_path, frame_index)
            if mask_frame is None:
                n_no_mask += 1
                continue
            if not is_likely_mask_frame(mask_frame):
                n_bad_mask += 1
                continue

            try:
                mask_224 = crop_and_resize_mask(mask_frame, crop_box)
                zmasks = zone_pixel_masks(entry["landmarks_68"])
                _, gt_label = zone_gt_from_mask(
                    mask_224, zmasks,
                    mask_threshold=args.mask_threshold,
                    coverage_threshold=args.coverage_threshold,
                )
                report = pipe.predict_from_landmark_entry(entry, image_path=key)
            except Exception as exc:
                print(f"  [WARN] skipped '{key}': {exc}")
                n_errors += 1
                continue

            expl = report.explanation
            flagged: Set[str] = {cv.zone for cv in expl.concept_vectors}
            # Q(z) is the score that actually drives flagging when the zone
            # prior is active; fall back to R(z) otherwise.
            if expl.Q_scores is not None:
                scores = {z: float(expl.Q_scores[j]) for j, z in enumerate(ZONE_NAMES)}
            else:
                scores = {z: float(expl.R_scores[j]) for j, z in enumerate(ZONE_NAMES)}

            counts = localization_counts(flagged, gt_label)
            for k in micro_counts:
                micro_counts[k] += counts[k]
            method = entry.get("method") or parse_ffpp_method(key) or "Unknown"
            for k in micro_counts:
                per_method_counts[method][k] += counts[k]
            for zone in ZONE_NAMES:
                gt = bool(gt_label.get(zone, False))
                pred = zone in flagged
                cell = ("tp" if gt else "fp") if pred else ("fn" if gt else "tn")
                per_zone_counts[zone][cell] += 1
                per_zone_auc_data[zone][0].append(int(gt))
                per_zone_auc_data[zone][1].append(scores[zone])
                zone_auc_labels.append(int(gt))
                zone_auc_scores.append(scores[zone])

            hit = pointing_game_hit(scores, gt_label)
            if hit is not None:
                pointing_hits.append(hit)

            if args.perturbation_max_frames > 0 and n_perturbation < args.perturbation_max_frames:
                image_bgr = cv2.imread(entry["face_image"])
                if image_bgr is not None:
                    if image_bgr.shape != (224, 224, 3):
                        image_bgr = cv2.resize(image_bgr, (224, 224))
                    order = sorted(ZONE_NAMES, key=lambda z: scores[z], reverse=True)
                    del_probs, ins_probs = deletion_insertion_curves(
                        pipe, image_bgr, zmasks, order
                    )
                    deletion_aucs.append(curve_auc(del_probs))
                    insertion_aucs.append(curve_auc(ins_probs))
                    for _ in range(args.random_permutations):
                        rand_order = list(ZONE_NAMES)
                        rng.shuffle(rand_order)
                        rd, ri = deletion_insertion_curves(pipe, image_bgr, zmasks, rand_order)
                        deletion_aucs_random.append(curve_auc(rd))
                        insertion_aucs_random.append(curve_auc(ri))
                    n_perturbation += 1

            n_evaluated += 1
            if args.checkpoint_every > 0 and n_evaluated % args.checkpoint_every == 0:
                _write_results(_build_results(partial=True))
                print(f"  [checkpoint] partial results written after "
                      f"{n_evaluated} frames -> {args.output}", flush=True)

    mask_cache.close()

    if n_evaluated == 0 and n_no_mask > 0:
        print(
            "\n[ERROR] No FF++ mask videos were found. The C23 manipulated "
            "videos are NOT masks.\n"
            "  1. Request the official FF++ download script (Google form on "
            "the FaceForensics++ repo).\n"
            "  2. Run: python download_ffpp_masks.py --help\n"
            "  3. Re-run this script with --masks-root pointing at the "
            "downloaded mask tree.\n"
        )
        raise SystemExit(1)

    results = _build_results(partial=False)

    print("\n" + "=" * 60)
    print("  ZONE LOCALIZATION vs FF++ MASK GROUND TRUTH")
    print("=" * 60)
    loc = results["localization"]
    print(f"  frames evaluated       : {n_evaluated} "
          f"(no mask: {n_no_mask}, bad mask frame: {n_bad_mask}, "
          f"no crop_box: {n_no_crop_box}, errors: {n_errors})")
    print(f"  micro precision        : {loc['micro']['precision']}")
    print(f"  micro recall           : {loc['micro']['recall']}")
    print(f"  micro F1               : {loc['micro']['f1']}")
    print(f"  micro IoU              : {loc['micro']['iou']}")
    print(f"  pointing game accuracy : {loc['pointing_game_accuracy']} (n={loc['pointing_game_n']})")
    print(f"  zone score AUC (micro) : {loc['zone_score_auc_micro']}")
    cf = results["causal_faithfulness"]
    if cf["n_frames"]:
        print(f"  deletion AUC  (attr / random) : {cf['deletion_auc_attribution_order']} / "
              f"{cf['deletion_auc_random_order']}   (lower attr = better)")
        print(f"  insertion AUC (attr / random) : {cf['insertion_auc_attribution_order']} / "
              f"{cf['insertion_auc_random_order']}   (higher attr = better)")
    print("=" * 60 + "\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    from console_utf8 import ensure_utf8_stdout
    ensure_utf8_stdout()
    main()
