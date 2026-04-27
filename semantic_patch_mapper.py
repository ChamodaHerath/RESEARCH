#!/usr/bin/env python3
"""
Map 68-point facial landmarks to ViT 14x14 patches (224x224 image, patch size 16).

This module is designed for Explainable AI (XAI) pipelines where semantic regions
(mouth, eyes, nose) are aligned with transformer patch tokens.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


IMAGE_SIZE = 224
GRID_SIZE = 14
PATCH_SIZE = 16
NUM_PATCHES = GRID_SIZE * GRID_SIZE

# Required regions for explanation head supervision.
REGION_TO_INDICES: Dict[str, np.ndarray] = {
    "mouth": np.arange(48, 68, dtype=np.int64),
    "eyes": np.arange(36, 48, dtype=np.int64),
    "nose": np.arange(27, 36, dtype=np.int64),
}
REGION_ORDER = ("mouth", "eyes", "nose")
LANDMARK_GROUPS_68: Dict[str, np.ndarray] = {
    "jawline": np.arange(0, 17, dtype=np.int64),
    "nose": np.arange(27, 36, dtype=np.int64),
    "eyes": np.arange(36, 48, dtype=np.int64),
    "mouth": np.arange(48, 68, dtype=np.int64),
}


def _validate_image_grid(image_size: int, grid_size: int, patch_size: int) -> None:
    if image_size != grid_size * patch_size:
        raise ValueError(
            f"Inconsistent grid settings: image_size={image_size}, "
            f"grid_size={grid_size}, patch_size={patch_size}."
        )


def _to_landmark_array(landmarks_68: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(landmarks_68, dtype=np.float32)
    if arr.shape != (68, 2):
        raise ValueError(f"Expected landmarks shape (68, 2), got {arr.shape}.")
    return arr


def _clip_landmarks_to_image(landmarks_xy: np.ndarray, image_size: int) -> np.ndarray:
    # Keep coordinates in [0, image_size) so exact boundary values map safely.
    max_val = np.nextafter(np.float32(image_size), np.float32(-np.inf))
    return np.clip(landmarks_xy, 0.0, max_val)


def landmarks_to_patch_indices(
    landmarks_68: Sequence[Sequence[float]],
    image_size: int = IMAGE_SIZE,
    grid_size: int = GRID_SIZE,
    patch_size: int = PATCH_SIZE,
) -> np.ndarray:
    """
    Vectorized mapping from 68 landmark points to ViT patch indices [0, 195].

    Returns:
        np.ndarray shape (68,), dtype int64
    """
    _validate_image_grid(image_size=image_size, grid_size=grid_size, patch_size=patch_size)
    landmarks = _clip_landmarks_to_image(_to_landmark_array(landmarks_68), image_size=image_size)
    patch_xy = np.floor(landmarks / float(patch_size)).astype(np.int64)
    patch_xy = np.clip(patch_xy, 0, grid_size - 1)
    # row-major patch index: row * grid_size + col
    patch_indices = patch_xy[:, 1] * grid_size + patch_xy[:, 0]
    return patch_indices


def region_patch_sets_from_landmarks(
    landmarks_68: Sequence[Sequence[float]],
    region_to_indices: Mapping[str, np.ndarray] = REGION_TO_INDICES,
) -> Dict[str, np.ndarray]:
    """
    Map semantic landmark groups to unique patch indices.

    Returns:
        Dict[str, np.ndarray], each array sorted unique patch IDs.
    """
    patch_indices = landmarks_to_patch_indices(landmarks_68)
    region_patches: Dict[str, np.ndarray] = {}
    for region_name, landmark_ids in region_to_indices.items():
        region_patches[region_name] = np.unique(patch_indices[landmark_ids])
    return region_patches


def build_semantic_mask_matrix(
    landmarks_68: Sequence[Sequence[float]],
    region_order: Sequence[str] = REGION_ORDER,
    region_to_indices: Mapping[str, np.ndarray] = REGION_TO_INDICES,
    as_torch: bool = False,
    torch_dtype: Optional["torch.dtype"] = None,
) -> Union[np.ndarray, "torch.Tensor"]:
    """
    Build semantic mask matrix of shape (196, Num_Regions).

    matrix[p, r] = 1 if patch p belongs to region r, else 0.
    Region defaults: mouth, eyes, nose.
    """
    region_patches = region_patch_sets_from_landmarks(
        landmarks_68=landmarks_68,
        region_to_indices=region_to_indices,
    )
    num_regions = len(region_order)
    matrix = np.zeros((NUM_PATCHES, num_regions), dtype=np.float32)

    for region_idx, region_name in enumerate(region_order):
        if region_name not in region_patches:
            raise KeyError(f"Region '{region_name}' not found in region mapping.")
        matrix[region_patches[region_name], region_idx] = 1.0

    if as_torch:
        if torch is None:
            raise ImportError("PyTorch is not installed, but as_torch=True was requested.")
        dtype = torch_dtype if torch_dtype is not None else torch.float32
        return torch.tensor(matrix, dtype=dtype)

    return matrix


def build_frame_semantic_masks(
    landmarks_json_path: Union[str, Path],
    as_torch: bool = False,
    torch_dtype: Optional["torch.dtype"] = None,
) -> Dict[str, Union[np.ndarray, "torch.Tensor"]]:
    """
    Load Step-01 JSON and return per-frame semantic mask matrices.

    Supports both:
    - {"frames": {"frame_key": {"landmarks_68": ...}, ...}}
    - {"frame_key": {"landmarks_68": ...}, ...}
    """
    json_path = Path(landmarks_json_path)
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    frames = payload.get("frames") if isinstance(payload, dict) else None
    if frames is None and isinstance(payload, dict):
        frames = payload
    if not isinstance(frames, dict):
        raise ValueError("Unsupported landmarks JSON format.")

    frame_to_matrix: Dict[str, Union[np.ndarray, "torch.Tensor"]] = {}
    for frame_key, frame_data in frames.items():
        if not isinstance(frame_data, dict):
            continue
        landmarks = frame_data.get("landmarks_68")
        if landmarks is None:
            continue
        frame_to_matrix[frame_key] = build_semantic_mask_matrix(
            landmarks_68=landmarks,
            as_torch=as_torch,
            torch_dtype=torch_dtype,
        )
    return frame_to_matrix


class SemanticExplanationGenerator:
    """
    Generate region-level semantic explanations from ViT patch attention.
    """

    GROUP_PRIORITY = ("mouth", "eyes", "nose", "jawline")
    DEFAULT_EXPLANATION_GROUPS = ("eyes", "nose", "mouth")

    def __init__(
        self,
        image_size: int = IMAGE_SIZE,
        grid_size: int = GRID_SIZE,
        patch_size: int = PATCH_SIZE,
        landmark_groups: Optional[Mapping[str, np.ndarray]] = None,
    ) -> None:
        self.image_size = image_size
        self.grid_size = grid_size
        self.patch_size = patch_size
        _validate_image_grid(image_size=self.image_size, grid_size=self.grid_size, patch_size=self.patch_size)

        self.num_patches = self.grid_size * self.grid_size
        self.landmark_groups = dict(landmark_groups) if landmark_groups is not None else dict(LANDMARK_GROUPS_68)

        # Precompute rank for deterministic tie-breaking in overlap assignment.
        self._priority_rank = {name: idx for idx, name in enumerate(self.GROUP_PRIORITY)}

    def _to_landmarks(self, landmarks_68: Sequence[Sequence[float]]) -> np.ndarray:
        return _clip_landmarks_to_image(
            _to_landmark_array(landmarks_68),
            image_size=self.image_size,
        )

    def _to_attention(self, attention_vec: Sequence[float]) -> np.ndarray:
        attention = np.asarray(attention_vec, dtype=np.float32)
        if attention.shape != (self.num_patches,):
            raise ValueError(
                f"Expected attention shape ({self.num_patches},), got {attention.shape}."
            )
        return attention

    def landmarks_to_patch_indices(self, landmarks_68: Sequence[Sequence[float]]) -> np.ndarray:
        landmarks = self._to_landmarks(landmarks_68)
        patch_xy = np.floor(landmarks / float(self.patch_size)).astype(np.int64)
        patch_xy = np.clip(patch_xy, 0, self.grid_size - 1)
        return patch_xy[:, 1] * self.grid_size + patch_xy[:, 0]

    def build_patch_group_assignment(
        self,
        landmarks_68: Sequence[Sequence[float]],
    ) -> Dict[int, str]:
        """
        Map each patch to one semantic group by majority landmark count.
        Ties are resolved by GROUP_PRIORITY.
        """
        patch_indices = self.landmarks_to_patch_indices(landmarks_68)
        patch_group_counts: Dict[int, Dict[str, int]] = {}

        for group_name, lm_ids in self.landmark_groups.items():
            group_patch_indices = patch_indices[lm_ids]
            if group_patch_indices.size == 0:
                continue
            unique_ids, counts = np.unique(group_patch_indices, return_counts=True)
            for patch_id, count in zip(unique_ids.tolist(), counts.tolist()):
                if patch_id not in patch_group_counts:
                    patch_group_counts[patch_id] = {}
                patch_group_counts[patch_id][group_name] = patch_group_counts[patch_id].get(group_name, 0) + int(count)

        assignment: Dict[int, str] = {}
        for patch_id, group_counts in patch_group_counts.items():
            best_group = sorted(
                group_counts.items(),
                key=lambda kv: (
                    -kv[1],  # highest landmark count first
                    self._priority_rank.get(kv[0], len(self._priority_rank)),  # deterministic tie-break
                ),
            )[0][0]
            assignment[patch_id] = best_group

        return assignment

    def compute_region_attention_scores(
        self,
        attention_vec: Sequence[float],
        landmarks_68: Sequence[Sequence[float]],
        include_groups: Sequence[str] = DEFAULT_EXPLANATION_GROUPS,
    ) -> Dict[str, float]:
        attention = self._to_attention(attention_vec)
        assignment = self.build_patch_group_assignment(landmarks_68)
        include_set = set(include_groups)
        scores: Dict[str, float] = {group: 0.0 for group in include_groups}

        for patch_id, group in assignment.items():
            if group in include_set:
                scores[group] += float(attention[patch_id])

        return scores

    def predict_label_and_explanation(
        self,
        attention_vec: Sequence[float],
        landmarks_68: Sequence[Sequence[float]],
        threshold: float = 0.20,
    ) -> Dict[str, object]:
        if threshold < 0.0:
            raise ValueError("threshold must be >= 0.0")

        attention = self._to_attention(attention_vec)
        total_attention = float(np.sum(attention))
        if total_attention <= 0.0:
            raise ValueError("Attention vector sum must be positive.")

        scores = self.compute_region_attention_scores(
            attention_vec=attention,
            landmarks_68=landmarks_68,
            include_groups=self.DEFAULT_EXPLANATION_GROUPS,
        )

        top_region = max(scores, key=scores.get)
        top_score = float(scores[top_region])
        threshold_value = threshold * total_attention

        flagged_regions = [
            region
            for region, score in scores.items()
            if score >= threshold_value
        ]

        if top_score >= threshold_value:
            label = "Fake"
            explanation = (
                f"High confidence of manipulation detected in {top_region.capitalize()} "
                "due to structural inconsistency."
            )
        else:
            label = "Real"
            explanation = (
                "No dominant semantic inconsistency detected across eyes, nose, and mouth "
                "regions at the configured attention threshold."
            )

        return {
            "label": label,
            "top_region": top_region,
            "top_region_score": top_score,
            "region_scores": scores,
            "flagged_regions": flagged_regions,
            "threshold": float(threshold),
            "threshold_value": threshold_value,
            "total_attention": total_attention,
            "explanation": explanation,
        }

    def visualize_semantic_patch_assignment(
        self,
        image: np.ndarray,
        landmarks_68: Sequence[Sequence[float]],
        attention_vec: Optional[Sequence[float]] = None,
        alpha: float = 0.30,
        show: bool = True,
    ) -> Tuple[plt.Figure, plt.Axes]:
        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            raise ValueError(f"Expected image of shape ({self.image_size}, {self.image_size}, C), got {image.shape}.")

        assignment = self.build_patch_group_assignment(landmarks_68)
        attention = self._to_attention(attention_vec) if attention_vec is not None else None
        max_attention = float(np.max(attention)) if attention is not None else 0.0

        group_colors = {
            "eyes": (0.1, 0.4, 1.0, alpha),
            "mouth": (1.0, 0.0, 0.0, alpha),
            "nose": (0.0, 0.8, 0.2, alpha),
            "jawline": (1.0, 0.7, 0.1, alpha),
        }

        fig, ax = plt.subplots(figsize=(7, 7))
        if image.ndim == 2:
            ax.imshow(image, cmap="gray")
        else:
            ax.imshow(image)

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                patch_idx = row * self.grid_size + col
                x = col * self.patch_size
                y = row * self.patch_size

                group_name = assignment.get(patch_idx)
                facecolor = group_colors.get(group_name, "none")
                if attention is not None and group_name is not None and max_attention > 0.0:
                    # Scale alpha lightly by attention while preserving semantic color.
                    scale = 0.2 + 0.8 * (float(attention[patch_idx]) / max_attention)
                    color = group_colors[group_name]
                    facecolor = (color[0], color[1], color[2], min(0.85, alpha * scale + 0.1))

                rect = mpatches.Rectangle(
                    (x, y),
                    self.patch_size,
                    self.patch_size,
                    linewidth=0.6,
                    edgecolor="yellow",
                    facecolor=facecolor,
                )
                ax.add_patch(rect)

        landmarks = self._to_landmarks(landmarks_68)
        ax.scatter(landmarks[:, 0], landmarks[:, 1], s=7, c="white", marker="o", linewidths=0)

        legend_items = [
            mpatches.Patch(color=group_colors["eyes"], label="Eyes"),
            mpatches.Patch(color=group_colors["nose"], label="Nose"),
            mpatches.Patch(color=group_colors["mouth"], label="Mouth"),
            mpatches.Patch(color=group_colors["jawline"], label="Jawline"),
        ]
        ax.legend(handles=legend_items, loc="upper right")
        ax.set_title("Semantic Patch Assignment on 14x14 ViT Grid")
        ax.set_xlim(0, self.image_size)
        ax.set_ylim(self.image_size, 0)
        ax.set_xticks(np.arange(0, self.image_size + 1, self.patch_size))
        ax.set_yticks(np.arange(0, self.image_size + 1, self.patch_size))
        ax.grid(False)
        ax.set_aspect("equal")

        if show:
            plt.show()

        return fig, ax


def visualize_semantic_grid_overlay(
    image: np.ndarray,
    landmarks_68: Sequence[Sequence[float]],
    alpha: float = 0.35,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Overlay 14x14 patch grid and highlight semantic regions.

    Highlighting for quick verification:
    - Mouth region patches (red)
    - Eyes region patches (blue)
    """
    if image.shape[0] != IMAGE_SIZE or image.shape[1] != IMAGE_SIZE:
        raise ValueError(f"Expected image of shape (224, 224, C), got {image.shape}.")

    region_patches = region_patch_sets_from_landmarks(landmarks_68)
    mouth_patches = set(region_patches["mouth"].tolist())
    eyes_patches = set(region_patches["eyes"].tolist())

    fig, ax = plt.subplots(figsize=(7, 7))
    if image.ndim == 2:
        ax.imshow(image, cmap="gray")
    else:
        ax.imshow(image)

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            patch_idx = row * GRID_SIZE + col
            x = col * PATCH_SIZE
            y = row * PATCH_SIZE

            facecolor = "none"
            if patch_idx in mouth_patches:
                facecolor = (1.0, 0.0, 0.0, alpha)  # red
            elif patch_idx in eyes_patches:
                facecolor = (0.1, 0.4, 1.0, alpha)  # blue

            rect = mpatches.Rectangle(
                (x, y),
                PATCH_SIZE,
                PATCH_SIZE,
                linewidth=0.6,
                edgecolor="yellow",
                facecolor=facecolor,
            )
            ax.add_patch(rect)

    landmarks = _to_landmark_array(landmarks_68)
    ax.scatter(landmarks[:, 0], landmarks[:, 1], s=7, c="lime", marker="o", linewidths=0)

    legend_items = [
        mpatches.Patch(color=(1.0, 0.0, 0.0, alpha), label="Mouth patches"),
        mpatches.Patch(color=(0.1, 0.4, 1.0, alpha), label="Eyes patches"),
    ]
    ax.legend(handles=legend_items, loc="upper right")
    ax.set_title("14x14 ViT Patch Grid with Semantic Highlights")
    ax.set_xlim(0, IMAGE_SIZE)
    ax.set_ylim(IMAGE_SIZE, 0)
    ax.set_xticks(np.arange(0, IMAGE_SIZE + 1, PATCH_SIZE))
    ax.set_yticks(np.arange(0, IMAGE_SIZE + 1, PATCH_SIZE))
    ax.grid(False)
    ax.set_aspect("equal")

    if show:
        plt.show()

    return fig, ax


__all__ = [
    "IMAGE_SIZE",
    "GRID_SIZE",
    "PATCH_SIZE",
    "NUM_PATCHES",
    "REGION_TO_INDICES",
    "REGION_ORDER",
    "landmarks_to_patch_indices",
    "region_patch_sets_from_landmarks",
    "build_semantic_mask_matrix",
    "build_frame_semantic_masks",
    "LANDMARK_GROUPS_68",
    "SemanticExplanationGenerator",
    "visualize_semantic_grid_overlay",
]
