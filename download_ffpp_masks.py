"""
download_ffpp_masks.py — helper for obtaining FF++ manipulation-mask videos.

Zone-localization evaluation (evaluate_zone_localization.py) needs the
*binary mask videos* that ship with FaceForensics++. These are NOT the same
as the C23 manipulated RGB videos you already have under
``FaceForensics++_C23/Deepfakes/`` etc.

The official masks are distributed only through the FaceForensics++ download
script, which requires a personal download link from their Google form:

  https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform

Once you have the script and server URL, download C23 masks for the four core
methods into a dedicated tree (recommended layout):

    datasets/FaceForensics++/masks/
        manipulated_sequences/Deepfakes/c23/masks/videos/*.mp4
        manipulated_sequences/Face2Face/c23/masks/videos/*.mp4
        ...

Usage
-----
    # Print step-by-step instructions:
    python download_ffpp_masks.py --instructions

    # Run the official downloader (after you receive it from FF++ authors):
    python download_ffpp_masks.py \
        --download-script path/to/download-FaceForensics.py \
        --output-dir datasets/FaceForensics++/masks

    # Verify that mask videos are discoverable:
    python download_ffpp_masks.py --verify-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ffpp_utils import FFPP_METHODS, find_ffpp_mask_video

DEFAULT_OUTPUT = Path("datasets/FaceForensics++/masks")
EXAMPLE_VIDEO = "Deepfakes/000_003.mp4"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download or verify FF++ manipulation-mask videos."
    )
    p.add_argument("--instructions", action="store_true",
                   help="Print manual download steps and exit.")
    p.add_argument("--download-script", type=Path, default=None,
                   help="Path to the official download-FaceForensics.py script.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--compression", type=str, default="c23",
                   choices=["raw", "c0", "c23", "hq", "c40", "lq"])
    p.add_argument("--methods", nargs="+", default=FFPP_METHODS)
    p.add_argument("--verify-only", action="store_true",
                   help="Check whether mask videos are already on disk.")
    return p.parse_args()


def print_instructions(output_dir: Path) -> None:
    print(
        "\n=== FF++ manipulation masks — download instructions ===\n\n"
        "1. Fill out the FaceForensics++ access form and wait for the email "
        "with your personal download script + server URL.\n\n"
        "2. Save download-FaceForensics.py somewhere on this machine.\n\n"
        "3. Download C23 masks for all four methods into:\n"
        f"     {output_dir}\n\n"
        "   Example commands (run once per method):\n"
        f"     python download-FaceForensics.py {output_dir} "
        f"-d Deepfakes -c c23 -t masks\n"
        f"     python download-FaceForensics.py {output_dir} "
        f"-d Face2Face -c c23 -t masks\n"
        f"     python download-FaceForensics.py {output_dir} "
        f"-d FaceSwap -c c23 -t masks\n"
        f"     python download-FaceForensics.py {output_dir} "
        f"-d NeuralTextures -c c23 -t masks\n\n"
        "4. Verify:\n"
        "     python download_ffpp_masks.py --verify-only\n\n"
        "5. Run zone-localization evaluation:\n"
        "     python evaluate_zone_localization.py \\\n"
        "         --json outputs/ffpp_landmarks_test_full.json \\\n"
        f"         --masks-root {output_dir}\n"
    )


def verify_masks(output_dir: Path) -> int:
    found = 0
    for method in FFPP_METHODS:
        rel = f"{method}/{EXAMPLE_VIDEO.split('/', 1)[1]}"
        path = find_ffpp_mask_video(rel, output_dir, output_dir)
        status = "OK" if path else "MISSING"
        print(f"  [{status:7s}] {rel}")
        if path:
            print(f"            -> {path}")
            found += 1
    print(f"\nResolved {found}/{len(FFPP_METHODS)} example mask videos under {output_dir}.")
    if found == 0:
        print(
            "No masks found. The C23 manipulated RGB videos are NOT valid "
            "ground truth — download the official mask videos first."
        )
        return 1
    return 0 if found == len(FFPP_METHODS) else 2


def run_official_download(script: Path, output_dir: Path, compression: str,
                          methods: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for method in methods:
        cmd = [
            sys.executable, str(script), str(output_dir),
            "-d", method, "-c", compression, "-t", "masks",
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    if args.instructions:
        print_instructions(args.output_dir)
        return
    if args.verify_only:
        raise SystemExit(verify_masks(args.output_dir))
    if args.download_script is None:
        print_instructions(args.output_dir)
        raise SystemExit(
            "Provide --download-script with the official FF++ downloader, "
            "or use --instructions / --verify-only."
        )
    run_official_download(
        args.download_script, args.output_dir, args.compression, args.methods
    )
    raise SystemExit(verify_masks(args.output_dir))


if __name__ == "__main__":
    main()
