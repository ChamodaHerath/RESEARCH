"""
repair_landmark_json.py — recover a truncated ffpp_landmarks*.json.

Large landmark JSON files (~600 MB+) can be truncated if the process is
interrupted during the final write. This script finds the last *complete*
frame entry in the "frames" object, truncates there, and closes the JSON
so evaluate_pipeline.py / evaluate_method_inference.py can load it.

Usage
-----
    python repair_landmark_json.py outputs/ffpp_landmarks_test.json
    python repair_landmark_json.py outputs/ffpp_landmarks_test.json --in-place
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


# Frame entries end with "label" (real/fake) or trailing "method" (fake only).
_FRAME_END_PATTERNS = (
    re.compile(r'      "method": "[^"]+"\n    \},?\n'),
    re.compile(r'      "label": "(?:fake|real)"\n    \},?\n'),
)


def find_last_complete_frame_end(text: str) -> int:
    """Return the index *after* the last complete frame entry's closing brace."""
    best = -1
    for pat in _FRAME_END_PATTERNS:
        for m in pat.finditer(text):
            best = max(best, m.end())
    if best < 0:
        raise ValueError(
            "Could not locate any complete frame entry in the file. "
            "The corruption may be too severe — re-run preprocess_landmarks.py."
        )
    return best


def repair_json_text(text: str) -> str:
    end = find_last_complete_frame_end(text)
    # Drop trailing comma on the last frame entry if present.
    body = text[:end].rstrip()
    if body.endswith(","):
        body = body[:-1]
    return body + '\n  },\n  "skipped_frames": []\n}\n'


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repair a truncated landmark JSON file.")
    p.add_argument("input", type=Path, help="Corrupted landmark JSON.")
    p.add_argument(
        "--output", type=Path, default=None,
        help="Repaired output path (default: <input>.repaired.json).",
    )
    p.add_argument(
        "--in-place", action="store_true",
        help="Overwrite input after writing a .bak backup.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = args.input
    if not src.exists():
        raise SystemExit(f"File not found: {src}")

    print(f"Reading {src} ({src.stat().st_size / 1e6:.1f} MB) ...")
    text = src.read_text(encoding="utf-8", errors="replace")

    try:
        json.loads(text)
        print("File is already valid JSON — nothing to repair.")
        return
    except json.JSONDecodeError as exc:
        print(f"Corruption detected: {exc}")

    repaired_text = repair_json_text(text)
    out = args.output or src.with_suffix(".repaired.json")
    out.write_text(repaired_text, encoding="utf-8")

    data = json.loads(repaired_text)
    n_frames = len(data.get("frames", {}))
    n_videos = len(data.get("videos", {}))
    print(f"Repaired: {n_frames} frames, {n_videos} videos -> {out}")

    if args.in_place:
        backup = src.with_suffix(".json.bak")
        if not backup.exists():
            src.rename(backup)
            print(f"Backup: {backup}")
        else:
            print(f"Backup already exists: {backup} (leaving original in place)")
        out.rename(src)
        print(f"In-place repair complete: {src}")


if __name__ == "__main__":
    main()
