"""
conftest.py — shared fixtures for the SAEDF pytest suite.

These fixtures avoid importing heavy runtime dependencies (dlib, mtcnn,
transformers, torch) so the fast unit-test tier can run in any CI
environment with just numpy / scipy / opencv-python installed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

IMAGE_SIZE = 224
NUM_PATCHES = 196


def _arc_points(cx: float, cy: float, rx: float, ry: float, angle_start: float, angle_end: float, n: int):
    """Generate n points along an elliptical arc (degrees)."""
    angles = np.linspace(math.radians(angle_start), math.radians(angle_end), n)
    return [(cx + rx * math.cos(a), cy + ry * math.sin(a)) for a in angles]


def make_synthetic_landmarks_68() -> np.ndarray:
    """
    Build a plausible, non-degenerate 68-point dlib-style landmark set inside
    a 224x224 frame. Coordinates are synthetic (procedurally generated from
    ellipses/arcs) rather than sampled from a real face, but every zone group
    used by semantic_patch_mapper.ZONE_LANDMARK_INDICES resolves to a valid
    (non-collinear, >=3 point) convex hull, which is all the geometry code
    requires.
    """
    pts = [None] * 68

    # Jaw line 0-16: lower arc of a face-outline ellipse.
    jaw = _arc_points(cx=112, cy=110, rx=85, ry=95, angle_start=200, angle_end=340, n=17)
    for i, p in enumerate(jaw):
        pts[i] = p

    # Right eyebrow 17-21, left eyebrow 22-26 (index convention only; not
    # anatomically meaningful for tests).
    brow_r = _arc_points(cx=75, cy=75, rx=22, ry=8, angle_start=200, angle_end=340, n=5)
    brow_l = _arc_points(cx=150, cy=75, rx=22, ry=8, angle_start=200, angle_end=340, n=5)
    for i, p in enumerate(brow_r):
        pts[17 + i] = p
    for i, p in enumerate(brow_l):
        pts[22 + i] = p

    # Nose bridge 27-30 (near-vertical but with a small x zig-zag so the
    # convex hull has non-degenerate area rather than collapsing to a line).
    for i in range(4):
        pts[27 + i] = (112 + (3 if i % 2 == 0 else -3), 85 + i * 10)
    nose_bottom = _arc_points(cx=112, cy=128, rx=14, ry=6, angle_start=200, angle_end=340, n=5)
    for i, p in enumerate(nose_bottom):
        pts[31 + i] = p

    # Eyes 36-41 / 42-47 (hexagons).
    eye_r = _arc_points(cx=80, cy=100, rx=14, ry=8, angle_start=0, angle_end=300, n=6)
    eye_l = _arc_points(cx=145, cy=100, rx=14, ry=8, angle_start=0, angle_end=300, n=6)
    for i, p in enumerate(eye_r):
        pts[36 + i] = p
    for i, p in enumerate(eye_l):
        pts[42 + i] = p

    # Mouth: outer lip 48-59 (12 points), inner lip 60-67 (8 points).
    outer_lip = _arc_points(cx=112, cy=165, rx=28, ry=14, angle_start=0, angle_end=330, n=12)
    inner_lip = _arc_points(cx=112, cy=165, rx=16, ry=7, angle_start=0, angle_end=315, n=8)
    for i, p in enumerate(outer_lip):
        pts[48 + i] = p
    for i, p in enumerate(inner_lip):
        pts[60 + i] = p

    assert all(p is not None for p in pts), "Synthetic landmark generation left gaps."
    arr = np.array(pts, dtype=np.float32)
    arr = np.clip(arr, 0.0, IMAGE_SIZE - 1.0)
    assert arr.shape == (68, 2)
    return arr


@pytest.fixture
def sample_landmarks_68() -> np.ndarray:
    return make_synthetic_landmarks_68()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=42)


@pytest.fixture
def sample_alpha(rng: np.random.Generator) -> np.ndarray:
    """A plausible CLS-attention vector: non-negative, sums to 1."""
    raw = rng.random(NUM_PATCHES).astype(np.float32) + 0.01
    return (raw / raw.sum()).astype(np.float32)


@pytest.fixture
def sample_phi(rng: np.random.Generator) -> np.ndarray:
    """A plausible patch gradient-magnitude vector: non-negative."""
    return rng.random(NUM_PATCHES).astype(np.float32)
