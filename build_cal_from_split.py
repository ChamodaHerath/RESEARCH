"""
build_cal_from_split.py — build a leakage-free calibration landmark JSON
from an official FF++ val (or train) split landmark file.

Never sample from the test split. Calibration frames are selected at the
VIDEO level (all kept frames of a video stay together) and stratified by
manipulation method + real originals.

Usage
-----
    python build_cal_from_split.py \
        --json outputs/ffpp_landmarks_val.json \
        --output outputs/ffpp_landmarks_cal.json \
        --max-videos-per-group 25 \
        --max-frames-per-video 20
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from ffpp_utils import parse_ffpp_method, video_path_from_key
from landmark_io import load_landmark_json, save_landmark_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a leakage-free calibration JSON from val/train landmarks."
    )
    p.add_argument("--json", type=Path, required=True,
                   help="Source landmark JSON (official val or train — never test).")
    p.add_argument("--output", type=Path, default=Path("outputs/ffpp_landmarks_cal.json"))
    p.add_argument("--max-videos-per-group", type=int, default=25,
                   help="Max videos kept per method group (and for originals).")
    p.add_argument("--max-frames-per-video", type=int, default=20,
                   help="Max frames kept per selected video.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _group_key(frame_key: str, entry: Dict) -> str:
    method = parse_ffpp_method(frame_key)
    if method is not None:
        return method
    label = str(entry.get("label", "")).lower()
    if label in {"real", "0"}:
        return "original"
    top = video_path_from_key(frame_key).split("/")[0]
    return top or "unknown"


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    data = load_landmark_json(args.json)
    frames: Dict[str, Dict] = data.get("frames", {})
    meta = dict(data.get("meta", {}))

    source_split = meta.get("split")
    if source_split == "test":
        raise SystemExit(
            f"Refusing to build calibration data from a test split ({args.json}). "
            "Use val or train."
        )

    by_group_video: Dict[str, Dict[str, List[Tuple[str, Dict]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for key, entry in frames.items():
        if not isinstance(entry, dict):
            continue
        if "zone_patch_matrix" not in entry or "face_image" not in entry:
            continue
        group = _group_key(key, entry)
        video = video_path_from_key(key)
        by_group_video[group][video].append((key, entry))

    selected: Dict[str, Dict] = {}
    selected_videos: Dict[str, List[str]] = {}
    for group, videos in sorted(by_group_video.items()):
        video_ids = list(videos.keys())
        rng.shuffle(video_ids)
        keep = video_ids[: args.max_videos_per_group]
        selected_videos[group] = keep
        for vid in keep:
            frame_items = list(videos[vid])
            frame_items.sort(key=lambda kv: kv[0])
            if len(frame_items) > args.max_frames_per_video:
                # even stride so we cover the video timeline
                step = len(frame_items) / args.max_frames_per_video
                idxs = [int(i * step) for i in range(args.max_frames_per_video)]
                frame_items = [frame_items[i] for i in idxs]
            for key, entry in frame_items:
                selected[key] = entry

    out_meta = {
        **meta,
        "calibration_source_json": str(args.json),
        "calibration_source_split": source_split,
        "calibration_built_at": datetime.now(timezone.utc).isoformat(),
        "calibration_seed": args.seed,
        "calibration_max_videos_per_group": args.max_videos_per_group,
        "calibration_max_frames_per_video": args.max_frames_per_video,
        "calibration_videos_per_group": {g: len(v) for g, v in selected_videos.items()},
        "num_processed": len(selected),
        "num_videos_processed": sum(len(v) for v in selected_videos.values()),
        "purpose": "leakage_free_fusion_calibration",
    }
    # Drop misleading full-split counters from the source file.
    for k in ("videos_skipped_by_split_filter", "num_skipped"):
        out_meta.pop(k, None)

    save_landmark_json(args.output, {"meta": out_meta, "frames": selected})
    print(f"Wrote {args.output}")
    print(f"  source split : {source_split}")
    print(f"  frames       : {len(selected)}")
    print(f"  videos       : {sum(len(v) for v in selected_videos.values())}")
    for g, vids in sorted(selected_videos.items()):
        n_frames = sum(1 for k in selected if video_path_from_key(k) in set(vids))
        print(f"  {g:16s} videos={len(vids):3d}  frames={n_frames}")


if __name__ == "__main__":
    main()
