"""
console_utf8.py — ensure stdout/stderr can print unicode (beta, block bars,
bullets, en-dashes) used throughout the Bayesian calibration reports.

Windows consoles frequently default to a legacy code page (e.g. cp1252)
that cannot encode characters like '\u03b2' (beta) or '\u2588' (block),
causing a hard UnicodeEncodeError crash mid-script. Call ensure_utf8_stdout()
once at the top of a CLI entry point (inside `if __name__ == "__main__":`)
to make console output robust regardless of the host locale.
"""

from __future__ import annotations

import sys


def ensure_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        encoding = getattr(stream, "encoding", None) or ""
        if encoding.lower().replace("-", "") != "utf8" and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
