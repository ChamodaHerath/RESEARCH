"""Shared helpers for reading/writing large landmark JSON files safely."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict


def load_landmark_json(path: Path | str) -> Dict[str, Any]:
    """Load a landmark JSON; raise a clear error if truncated/corrupt."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise json.JSONDecodeError(
            f"{exc.msg} — {path} looks truncated or corrupt "
            f"(~{path.stat().st_size / 1e6:.0f} MB). "
            "Run: python repair_landmark_json.py "
            f"{path.as_posix()} --in-place",
            exc.doc,
            exc.pos,
        ) from exc


def save_landmark_json(path: Path | str, payload: Dict[str, Any], *, compact: bool = True) -> None:
    """
    Atomically write a landmark JSON (temp file + replace).

    Uses compact JSON by default (no indent) — roughly 3–5× smaller and
    much faster to write than indent=2 for 30k+ frame files.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"ensure_ascii": False}
    if compact:
        kwargs["separators"] = (",", ":")
    else:
        kwargs["indent"] = 2

    fd, tmp_name = tempfile.mkstemp(
        suffix=".json.tmp",
        prefix=f"{path.stem}_",
        dir=str(path.parent),
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, **kwargs)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
