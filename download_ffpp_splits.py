"""
download_ffpp_splits.py — fetch the official FaceForensics++ train/val/test
split JSON files (720 / 140 / 140 video pairs).

These files are NOT included in the C23 video download; they live in the
FaceForensics++ GitHub repo. Without them, preprocessing falls back to
processing every video on disk (7000+) or a tiny ad-hoc subset.

Usage
-----
    python download_ffpp_splits.py
    python download_ffpp_splits.py --output-dir datasets/FaceForensics++/splits
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

BASE_URL = (
    "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits"
)
SPLIT_FILES = ("train.json", "val.json", "test.json")
DEFAULT_OUT = Path("datasets/FaceForensics++/splits")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download official FF++ split JSON files.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_FILES:
        url = f"{BASE_URL}/{name}"
        dest = args.output_dir / name
        print(f"Downloading {url} -> {dest}")
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        dest.write_text(json.dumps(data, indent=2), encoding="utf-8")
        n_pairs = len(data) if isinstance(data, list) else "?"
        print(f"  saved {name} ({n_pairs} video pairs)")
    print(f"\nDone. Use with preprocess_landmarks.py --official-splits {args.output_dir}")


if __name__ == "__main__":
    main()
