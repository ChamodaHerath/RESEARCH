"""
run_pipeline_single_frame.py — End-to-end dual-head demo.

Loads a single frame from the landmark JSON, runs the full SAEDFPipeline
(one forward pass → classification head + semantic explanation head), and
prints the ForensicReport.

Usage
-----
    python run_pipeline_single_frame.py
    python run_pipeline_single_frame.py --json outputs/ffpp_landmarks_test.json
    python run_pipeline_single_frame.py --json outputs/ffpp_landmarks_test.json --key "Deepfakes/000_003.mp4::f000000"
    python run_pipeline_single_frame.py --image inputs/face.jpg --landmarks outputs/ffpp_landmarks_test.json --key "..."
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline import SAEDFPipeline

DEFAULT_JSON = "outputs/ffpp_landmarks_test_full.json"
DEFAULT_KEY  = "Deepfakes/000_003.mp4::f000000"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SAEDF dual-head demo: classification + semantic explanation."
    )
    p.add_argument(
        "--json",
        type=Path,
        default=DEFAULT_JSON,
        help="Path to ffpp_landmarks*.json (default: outputs/ffpp_landmarks_test_full.json).",
    )
    p.add_argument(
        "--key",
        type=str,
        default=DEFAULT_KEY,
        help="Frame key inside the JSON to run on.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON report to (e.g. outputs/report.json).",
    )
    p.add_argument(
        "--min-severity",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
        help="Minimum zone severity to generate explanations for (default: medium).",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/fusion_config.json"),
        help="Path to fusion_config.json (default: configs/fusion_config.json).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load landmark JSON
    landmark_path = Path(args.json)
    if not landmark_path.exists():
        raise FileNotFoundError(f"Landmark JSON not found: {landmark_path}")
    with open(landmark_path) as fh:
        data = json.load(fh)

    frames = data.get("frames", data)  # handle both {frames: {...}} and flat dicts
    if args.key not in frames:
        available = list(frames.keys())[:5]
        raise KeyError(
            f"Key '{args.key}' not found in {landmark_path}. "
            f"First 5 available keys: {available}"
        )
    entry = frames[args.key]

    # Run dual-head pipeline
    print(f"\nLoading pipeline (model + calibrated fusion config) ...")
    with SAEDFPipeline(
        config_path=str(args.config),
        min_severity=args.min_severity,
    ) as pipe:
        print(f"Running inference on frame: {args.key}")
        report = pipe.predict_from_landmark_entry(entry, image_path=args.key)

    # Print human-readable summary
    report.print_summary()

    # Print full JSON
    print("\n[FULL JSON REPORT]")
    print(report.to_json())

    # Optionally save to file
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report.to_json(), encoding="utf-8")
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    from console_utf8 import ensure_utf8_stdout
    ensure_utf8_stdout()
    main()
