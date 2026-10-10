"""
ffpp_utils.py — FaceForensics++ metadata helpers shared by the evaluation
and training scripts.

Provides:
  - Method-label parsing from landmark-JSON frame keys / entries
    ("Deepfakes/000_003.mp4::f000000" -> "Deepfakes").
  - Video-id extraction for video-level (clustered) metrics.
  - Official FF++ train/val/test split loading (720/140/140 video pairs,
    the split JSONs shipped with the FaceForensics++ download scripts),
    with a deterministic hash-based fallback when the official split files
    are not available.

All functions here are dependency-light (stdlib + nothing else) so they can
be unit-tested in the fast CI tier.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# The four core FF++ manipulation methods (Rössler et al., 2019).
FFPP_METHODS: List[str] = [
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
]

# Case-insensitive lookup, plus common alternative directory spellings.
_METHOD_ALIASES: Dict[str, str] = {
    "deepfakes": "Deepfakes",
    "face2face": "Face2Face",
    "faceswap": "FaceSwap",
    "neuraltextures": "NeuralTextures",
}

_REAL_DIR_NAMES = {"original", "original_sequences", "originals", "real", "youtube"}


# ---------------------------------------------------------------------------
# Frame-key / entry parsing
# ---------------------------------------------------------------------------

def video_path_from_key(frame_key: str) -> str:
    """'Deepfakes/000_003.mp4::f000000' -> 'Deepfakes/000_003.mp4'."""
    return frame_key.split("::")[0]


def video_id_from_key(frame_key: str) -> str:
    """
    Return a stable per-video cluster id for a landmark-JSON frame key.

    Frames from the same source video must map to the same id so that
    video-level aggregation and cluster bootstrap resampling respect the
    within-video correlation of frames.
    """
    return video_path_from_key(frame_key)


def parse_ffpp_method(frame_key_or_path: str) -> Optional[str]:
    """
    Extract the FF++ manipulation-method label from a frame key or video path.

    Returns one of FFPP_METHODS, or None when the path does not correspond to
    a known FF++ manipulation method (e.g. original/real videos, Celeb-DF).
    """
    video_path = video_path_from_key(frame_key_or_path)
    for part in re.split(r"[\\/]+", video_path):
        canonical = _METHOD_ALIASES.get(part.lower())
        if canonical is not None:
            return canonical
    return None


def is_ffpp_real(frame_key_or_path: str) -> bool:
    """True when the path's leading directory names it as an original/real video."""
    video_path = video_path_from_key(frame_key_or_path)
    for part in re.split(r"[\\/]+", video_path):
        if part.lower() in _REAL_DIR_NAMES:
            return True
    return False


def ffpp_video_ids(video_path: str) -> List[str]:
    """
    Return the FF++ numeric video id(s) referenced by a video path.

    'Deepfakes/000_003.mp4' -> ['000', '003']  (target id first)
    'original/000.mp4'      -> ['000']
    Non-FF++-style names return [].
    """
    stem = Path(video_path_from_key(video_path)).stem
    if re.fullmatch(r"\d+_\d+", stem):
        return stem.split("_")
    if re.fullmatch(r"\d+", stem):
        return [stem]
    return []


# ---------------------------------------------------------------------------
# Official FF++ splits
# ---------------------------------------------------------------------------

def load_official_split(split_json_path: Path) -> Set[Tuple[str, str]]:
    """
    Load one official FF++ split file (train.json / val.json / test.json).

    The official files contain a list of [target_id, source_id] string pairs,
    e.g. [["953", "974"], ...]. Returns the set of ordered pairs.
    """
    with Path(split_json_path).open("r", encoding="utf-8") as fh:
        pairs = json.load(fh)
    out: Set[Tuple[str, str]] = set()
    for pair in pairs:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            out.add((str(pair[0]), str(pair[1])))
    return out


def split_ids_from_pairs(pairs: Set[Tuple[str, str]]) -> Set[str]:
    """Flatten a set of (target, source) pairs into the set of video ids."""
    ids: Set[str] = set()
    for a, b in pairs:
        ids.add(a)
        ids.add(b)
    return ids


def load_official_splits_dir(splits_dir: Path) -> Dict[str, Tuple[Set[Tuple[str, str]], Set[str]]]:
    """
    Load train/val/test split files from a directory.

    Returns {split_name: (pairs_set, ids_set)} for each file present.
    """
    out: Dict[str, Tuple[Set[Tuple[str, str]], Set[str]]] = {}
    for name in ("train", "val", "test"):
        path = splits_dir / f"{name}.json"
        if path.exists():
            pairs = load_official_split(path)
            out[name] = (pairs, split_ids_from_pairs(pairs))
    if not out:
        raise FileNotFoundError(
            f"No train.json/val.json/test.json found under {splits_dir}. "
            "Run: python download_ffpp_splits.py"
        )
    return out


def video_in_official_split(
    video_path: str,
    pairs: Set[Tuple[str, str]],
    ids: Optional[Set[str]] = None,
) -> bool:
    """
    Decide whether a video belongs to an official FF++ split.

    - Fake videos ('000_003.mp4'): in the split when the (target, source)
      pair appears in the split file in either order.
    - Original videos ('000.mp4'): in the split when the id appears anywhere
      in the split's pairs.
    """
    vids = ffpp_video_ids(video_path)
    if not vids:
        return False
    if ids is None:
        ids = split_ids_from_pairs(pairs)
    if len(vids) == 2:
        a, b = vids
        return (a, b) in pairs or (b, a) in pairs
    return vids[0] in ids


# ---------------------------------------------------------------------------
# Deterministic hash-based fallback split (video-level)
# ---------------------------------------------------------------------------

def hash_split_assignment(
    video_path: str,
    fractions: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: str = "saedf-ffpp-split-v1",
) -> str:
    """
    Deterministically assign a video to 'train' / 'val' / 'test'.

    The assignment key is the *target* FF++ id when the filename follows the
    FF++ convention, so an original video and every fake derived from it land
    in the same split (identity-level leakage control). Non-FF++ names fall
    back to hashing the full video path.

    Prefer the official FF++ splits (load_official_split) whenever the split
    JSONs are available; this fallback exists so the tooling still works on
    partial local copies of the dataset.
    """
    if not (abs(sum(fractions) - 1.0) < 1e-9):
        raise ValueError(f"Split fractions must sum to 1.0, got {fractions}.")
    vids = ffpp_video_ids(video_path)
    key = vids[0] if vids else video_path_from_key(video_path)
    digest = hashlib.md5(f"{seed}:{key}".encode("utf-8")).hexdigest()
    u = int(digest[:8], 16) / float(0xFFFFFFFF)
    if u < fractions[0]:
        return "train"
    if u < fractions[0] + fractions[1]:
        return "val"
    return "test"


# ---------------------------------------------------------------------------
# Manipulation-mask video lookup (FF++ ground-truth evaluation)
# ---------------------------------------------------------------------------

def find_ffpp_mask_video(
    video_rel_path: str,
    masks_root: Optional[Path],
    ffpp_root: Optional[Path] = None,
) -> Optional[Path]:
    """
    Locate the FF++ manipulation-mask video for a fake clip.

    IMPORTANT: manipulated RGB videos (e.g. ``Deepfakes/000_003.mp4`` under
    ``FaceForensics++_C23``) are *not* masks. Only paths that contain a
    ``masks`` directory segment are considered.
    """
    rel = Path(video_rel_path)
    method = rel.parts[0] if rel.parts else ""
    name = rel.name
    candidates: List[Path] = []
    for root in (masks_root, ffpp_root):
        if root is None:
            continue
        candidates += [
            root / "manipulated_sequences" / method / "c23" / "masks" / "videos" / name,
            root / "manipulated_sequences" / method / "masks" / "videos" / name,
            root / method / "c23" / "masks" / "videos" / name,
            root / method / "masks" / "videos" / name,
            root / method / "masks" / name,
            root / "masks" / method / name,
            root / "masks" / method / "videos" / name,
        ]
    seen: Set[str] = set()
    for cand in candidates:
        key = str(cand.resolve()) if cand.exists() else str(cand)
        if key in seen:
            continue
        seen.add(key)
        if not cand.exists():
            continue
        parts = {p.lower() for p in cand.parts}
        if "masks" not in parts:
            continue
        return cand
    return None


__all__ = [
    "FFPP_METHODS",
    "video_path_from_key",
    "video_id_from_key",
    "parse_ffpp_method",
    "is_ffpp_real",
    "ffpp_video_ids",
    "load_official_split",
    "split_ids_from_pairs",
    "load_official_splits_dir",
    "video_in_official_split",
    "hash_split_assignment",
    "find_ffpp_mask_video",
]
