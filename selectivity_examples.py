"""
selectivity_examples.py — concrete evidence that zone flagging is selective,
not universal.

Scans fake frames from a landmark JSON and prints/saves full forensic
reports for a handful of frames that flag only a small subset of the 9
zones (rather than all of them), which is the concrete, per-frame version of
the aggregate "mean flagged zones dropped from ~7 to ~3" claim in README.md.

Usage
-----
    python selectivity_examples.py --json outputs/ffpp_landmarks_test.json --n 4
    python selectivity_examples.py --json outputs/celebdf_landmarks_test.json --n 4 --max-scan 800
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from pipeline import SAEDFPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Print/save selective (non-universal) example forensic reports.")
    p.add_argument("--json", type=Path, default=Path("outputs/ffpp_landmarks_test_full.json"))
    p.add_argument("--config", type=Path, default=Path("configs/fusion_config.json"))
    p.add_argument("--n", type=int, default=4, help="How many example reports to keep.")
    p.add_argument("--max-flagged", type=int, default=3, help="Only keep frames flagging <= this many zones.")
    p.add_argument("--max-scan", type=int, default=None, help="Stop scanning after this many fake frames.")
    p.add_argument("--output", type=Path, default=Path("outputs/selectivity_examples.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.json) as fh:
        data = json.load(fh)
    frames = data.get("frames", data)

    examples: List[Dict] = []
    scanned = 0

    with SAEDFPipeline(config_path=str(args.config), min_severity="low") as pipe:
        for key, entry in frames.items():
            if entry.get("label") != "fake":
                continue
            scanned += 1
            if args.max_scan is not None and scanned > args.max_scan:
                break

            try:
                report = pipe.predict_from_landmark_entry(entry, image_path=key)
            except Exception as exc:
                print(f"  [WARN] skipped '{key}': {exc}")
                continue

            n_flagged = len(report.explanation.concept_vectors)
            if 0 < n_flagged <= args.max_flagged:
                examples.append({"key": key, "report": report})
                report.print_summary()

            if len(examples) >= args.n:
                break

    print(f"\nScanned {scanned} fake frames, kept {len(examples)} selective examples "
          f"(<= {args.max_flagged} flagged zones each).")

    payload = {
        "source_json": str(args.json),
        "max_flagged_zones_filter": args.max_flagged,
        "frames_scanned": scanned,
        "examples": [
            {"key": ex["key"], **ex["report"].to_dict()}
            for ex in examples
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    from console_utf8 import ensure_utf8_stdout

    ensure_utf8_stdout()
    main()
