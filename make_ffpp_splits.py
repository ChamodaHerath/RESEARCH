"""
make_ffpp_splits.py — split a landmark JSON into leakage-controlled
train / val / test JSONs at the VIDEO level.

Why this exists
---------------
The in-domain numbers previously reported for off-the-shelf HuggingFace
checkpoints are not defensible: those checkpoints have unknown training data,
so FF++ test videos may have been seen during their training. The defensible
path is to fine-tune a detector ourselves (train_vit_detector.py) on a split
whose provenance we control. Two leakage rules are enforced here:

  1. Video-level separation — all frames of a video go to exactly one split
     (frame-level random splits leak near-duplicate frames across splits).
  2. Identity-level separation — an FF++ original video and every fake
     derived from it (e.g. original/000.mp4, Deepfakes/000_003.mp4) share the
     same FF++ target id and are forced into the same split.

Split source
------------
  --official-splits DIR : directory containing the official FF++ split files
                          train.json / val.json / test.json (720/140/140
                          video pairs). This is the reproducible, standard
                          option — use it whenever the files are available
                          (they ship with the FF++ download scripts).
  (default)             : deterministic hash-based 70/15/15 assignment on the
                          FF++ target id (ffpp_utils.hash_split_assignment),
                          for partial local copies without the split files.

Usage
-----
    python make_ffpp_splits.py \
        --json outputs/ffpp_landmarks.json \
        --official-splits datasets/FaceForensics++/splits \
        --output-prefix outputs/ffpp_landmarks

    # writes outputs/ffpp_landmarks_train.json,
    #        outputs/ffpp_landmarks_val.json,
    #        outputs/ffpp_landmarks_test.json,
    #        outputs/ffpp_landmarks_split_manifest.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

from ffpp_utils import (
    ffpp_video_ids,
    hash_split_assignment,
    load_official_split,
    split_ids_from_pairs,
    video_in_official_split,
    video_path_from_key,
)

SPLIT_NAMES = ["train", "val", "test"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Video-level, identity-safe train/val/test split of a landmark JSON."
    )
    p.add_argument("--json", type=Path, required=True,
                   help="Landmark JSON produced by preprocess_landmarks.py.")
    p.add_argument("--official-splits", type=Path, default=None,
                   help="Directory with official FF++ train.json/val.json/test.json.")
    p.add_argument("--output-prefix", type=Path, default=None,
                   help="Prefix for output files (default: input path without .json).")
    p.add_argument("--fractions", type=float, nargs=3, default=[0.70, 0.15, 0.15],
                   metavar=("TRAIN", "VAL", "TEST"),
                   help="Hash-fallback split fractions (ignored with --official-splits).")
    return p.parse_args()


def assign_video(
    video_path: str,
    official: Optional[Dict[str, Tuple[Set, Set]]],
    fractions: Tuple[float, float, float],
) -> Optional[str]:
    """Return 'train'/'val'/'test' for a video, or None if not in any official split."""
    if official is not None:
        for split in SPLIT_NAMES:
            pairs, ids = official[split]
            if video_in_official_split(video_path, pairs, ids):
                return split
        return None
    return hash_split_assignment(video_path, fractions=fractions)


def main() -> None:
    args = parse_args()
    with args.json.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    frames: Dict[str, Dict] = data.get("frames", data)
    meta = data.get("meta", {})

    official = None
    if args.official_splits is not None:
        official = {}
        for split in SPLIT_NAMES:
            path = args.official_splits / f"{split}.json"
            if not path.exists():
                raise FileNotFoundError(f"Official split file missing: {path}")
            pairs = load_official_split(path)
            official[split] = (pairs, split_ids_from_pairs(pairs))
        print(f"Loaded official FF++ splits from {args.official_splits} "
              f"({', '.join(f'{s}: {len(official[s][0])} pairs' for s in SPLIT_NAMES)})")
    else:
        print("No --official-splits given: using deterministic hash-based "
              f"{args.fractions} split on FF++ target ids. Prefer the official "
              "split files when available.")

    fractions = tuple(args.fractions)
    split_frames: Dict[str, Dict[str, Dict]] = {s: {} for s in SPLIT_NAMES}
    video_assignment: Dict[str, Optional[str]] = {}
    unassigned = 0

    for key, entry in frames.items():
        video_path = entry.get("video_path") or video_path_from_key(key)
        if video_path not in video_assignment:
            video_assignment[video_path] = assign_video(video_path, official, fractions)
        split = video_assignment[video_path]
        if split is None:
            unassigned += 1
            continue
        split_frames[split][key] = entry

    # Sanity check: no FF++ target id may appear in more than one split.
    id_to_splits: Dict[str, Set[str]] = {}
    for video_path, split in video_assignment.items():
        if split is None:
            continue
        vids = ffpp_video_ids(video_path)
        if vids:
            id_to_splits.setdefault(vids[0], set()).add(split)
    leaked = {vid: sorted(s) for vid, s in id_to_splits.items() if len(s) > 1}
    if leaked:
        raise RuntimeError(
            f"Identity leakage detected — FF++ target ids in multiple splits: {leaked}"
        )

    prefix = args.output_prefix or args.json.with_suffix("")
    manifest = {
        "source_json": str(args.json),
        "split_source": (
            f"official:{args.official_splits}" if official is not None
            else f"hash_fallback:{list(fractions)}"
        ),
        "identity_leakage_check": "passed",
        "unassigned_frames": unassigned,
        "splits": {},
    }

    for split in SPLIT_NAMES:
        sf = split_frames[split]
        label_counts = Counter(e.get("label", "?") for e in sf.values())
        method_counts = Counter(
            e.get("method", "real" if e.get("label") == "real" else "?")
            for e in sf.values()
        )
        n_videos = len({e.get("video_path") or video_path_from_key(k) for k, e in sf.items()})
        out_path = Path(f"{prefix}_{split}.json")
        out_payload = {
            "meta": {
                **meta,
                "split": split,
                "split_source": manifest["split_source"],
                "num_processed": len(sf),
            },
            "frames": sf,
        }
        out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
        manifest["splits"][split] = {
            "path": str(out_path),
            "n_frames": len(sf),
            "n_videos": n_videos,
            "label_counts": dict(label_counts),
            "method_counts": dict(method_counts),
        }
        print(f"  {split:5s}: {len(sf):6d} frames, {n_videos:4d} videos, "
              f"labels={dict(label_counts)} -> {out_path}")

    manifest_path = Path(f"{prefix}_split_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Split manifest saved to: {manifest_path}")
    if unassigned:
        print(f"[NOTE] {unassigned} frames belonged to videos outside the official "
              "splits and were not written to any split.")


if __name__ == "__main__":
    from console_utf8 import ensure_utf8_stdout
    ensure_utf8_stdout()
    main()
