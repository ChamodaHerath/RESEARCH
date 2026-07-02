# attention_extractor.py

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Zone constants — must match preprocess_landmarks.py
ZONE_NAMES: List[str] = [
    "left_eye", "right_eye", "nose_bridge", "nose_tip",
    "left_cheek", "right_cheek", "mouth", "forehead", "chin_jaw",
]
NUM_ZONES: int = len(ZONE_NAMES)

DEFAULT_MODEL_NAME = "vit_base_patch16_224"
DEFAULT_IMAGE_SIZE = 224
DEFAULT_PATCH_GRID = 14
NUM_PATCHES = DEFAULT_PATCH_GRID * DEFAULT_PATCH_GRID   # 196

DEFAULT_HF_DEEPFAKE_MODEL = os.getenv(
    "SAEDF_MODEL_NAME",
    "hamzenium/ViT-Deepfake-Classifier",
)
# Winner of the Phase 1 detector tournament (see audit_vit_models.py /
# detector_benchmark.py, outputs/model_benchmarks/): FF++ AUC 1.000,
# Celeb-DF cross-dataset AUC 0.901 -- vs. 0.429 / 0.558 for the previous
# default (prithivMLmods/Deep-Fake-Detector-v2-Model). Drop-in compatible:
# 224px input, patch_size=16 (196 patches), binary id2label, supports
# output_attentions=True with eager attention.


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class AttentionMaps:
    """
    Stores attention outputs for a batch.

    cls_patch_attention : (B, 196)  — mean CLS-to-patch attention, last layer
    grid_attention      : (B, 14, 14)
    heatmaps_224        : (B, 224, 224) — normalised to [0, 1]
    zone_scores         : (B, 9) or None — A(z) per zone; populated by
                          get_zone_attention_scores() when M matrices are provided
    """
    cls_patch_attention: np.ndarray
    grid_attention: np.ndarray
    heatmaps_224: np.ndarray
    zone_scores: Optional[np.ndarray] = None   # NEW: shape (B, 9)


# ---------------------------------------------------------------------------
# Zone-level attention aggregation  (new in this version)
# ---------------------------------------------------------------------------

def compute_attention_zone_scores(
    alpha: np.ndarray,
    M: np.ndarray,
) -> np.ndarray:
    """
    Aggregate per-patch CLS attention into per-zone scores A(z).

    Formula: A(z) = Σ_i [ M[i,z] · α[i] ] / (Σ_i M[i,z] + ε)

    The denominator normalises by zone coverage so large zones (e.g. chin_jaw)
    do not automatically dominate solely because they cover more patches.

    Args:
        alpha : np.ndarray shape (196,) — CLS attention weights for one image,
                values in [0, 1], should sum to 1 (softmax output).
        M     : np.ndarray shape (196, 9) — zone-patch overlap-fraction matrix
                produced by preprocess_landmarks.compute_zone_patch_matrix().

    Returns:
        A : np.ndarray shape (9,) — attention score per zone, values in [0, 1].
    """
    if alpha.shape != (NUM_PATCHES,):
        raise ValueError(f"Expected alpha shape ({NUM_PATCHES},), got {alpha.shape}")
    if M.shape != (NUM_PATCHES, NUM_ZONES):
        raise ValueError(f"Expected M shape ({NUM_PATCHES}, {NUM_ZONES}), got {M.shape}")

    zone_area = M.sum(axis=0) + 1e-8          # (9,)
    weighted  = (M * alpha[:, None]).sum(axis=0)  # (9,)
    A = weighted / zone_area
    return A.astype(np.float32)


def load_zone_patch_matrices_from_json(
    json_path: Union[str, Path],
    frame_keys: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Load pre-computed zone-patch matrices serialised by preprocess_landmarks.py.

    Args:
        json_path  : Path to the landmarks JSON file.
        frame_keys : Optional subset of frame keys to load. If None, loads all.

    Returns:
        Dict mapping frame_key → M matrix of shape (196, 9).
    """
    import json
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    frames = payload.get("frames", payload)
    if not isinstance(frames, dict):
        raise ValueError("Unsupported landmarks JSON format.")

    result: Dict[str, np.ndarray] = {}
    for key, data in frames.items():
        if frame_keys is not None and key not in frame_keys:
            continue
        if not isinstance(data, dict):
            continue
        raw = data.get("zone_patch_matrix")
        if raw is None:
            # Fallback: JSON was produced by the old 5-zone preprocessor without M.
            # Caller should regenerate the JSON with the updated preprocess_landmarks.py.
            continue
        result[key] = np.array(raw, dtype=np.float32)

    return result


# ---------------------------------------------------------------------------
# AttentionExtractor — original class with one new public method added
# ---------------------------------------------------------------------------

class AttentionExtractor:
    """
    ViT attention extractor for interpretability.

    NEW: get_zone_attention_scores() wires CLS attention → M matrix → A(z).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        pretrained: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_name = model_name
        self.is_hf = self._looks_like_hf_model_id(model_name)
        self.image_processor = None
        self.model = self._load_model(model_name=model_name, pretrained=pretrained).to(self.device)
        self.model.eval()

        if self.is_hf and hasattr(self.model, "set_attn_implementation"):
            try:
                self.model.set_attn_implementation("eager")
            except Exception:
                pass

        self._last_attn: Optional[torch.Tensor] = None
        self._hook_handle: Optional[torch.utils.hooks.RemovableHook] = None

        if not self.is_hf:
            self._register_last_attention_hook()

    @staticmethod
    def _looks_like_hf_model_id(model_name: str) -> bool:
        return "/" in model_name

    def _load_model(self, model_name: str, pretrained: bool) -> nn.Module:
        if self.is_hf:
            try:
                from transformers import AutoImageProcessor, AutoModelForImageClassification
            except Exception as e:
                raise RuntimeError(
                    "transformers is required for HF models. pip install transformers"
                ) from e
            self.image_processor = AutoImageProcessor.from_pretrained(model_name)
            return AutoModelForImageClassification.from_pretrained(model_name)

        try:
            import timm
            return timm.create_model(model_name, pretrained=pretrained)
        except Exception:
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            weights = ViT_B_16_Weights.DEFAULT if pretrained else None
            return vit_b_16(weights=weights)

    def _register_last_attention_hook(self) -> None:
        attn_module = self._resolve_last_attention_module(self.model)
        if attn_module is None:
            raise RuntimeError("Could not resolve the last attention module in this ViT model.")

        def _hook(module, module_inputs, module_output):
            captured = None
            if isinstance(module_output, tuple) and len(module_output) >= 2:
                maybe_weights = module_output[1]
                if torch.is_tensor(maybe_weights):
                    captured = maybe_weights
            elif hasattr(module, "attn_map") and torch.is_tensor(module.attn_map):
                captured = module.attn_map
            if captured is None and hasattr(module, "attn") and torch.is_tensor(module.attn):
                captured = module.attn
            self._last_attn = captured

        self._hook_handle = attn_module.register_forward_hook(_hook)

    @staticmethod
    def _resolve_last_attention_module(model: nn.Module) -> Optional[nn.Module]:
        if hasattr(model, "blocks") and len(getattr(model, "blocks")) > 0:
            last_block = model.blocks[-1]
            if hasattr(last_block, "attn"):
                return last_block.attn
        if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            layers = model.encoder.layers
            if len(layers) > 0:
                last_layer = layers[-1]
                if hasattr(last_layer, "self_attention"):
                    return last_layer.self_attention
        return None

    @staticmethod
    def preprocess_bgr_images(images_bgr: np.ndarray) -> torch.Tensor:
        """Convert BGR uint8 batch [B,224,224,3] → normalised float tensor [B,3,224,224]."""
        if images_bgr.ndim != 4 or images_bgr.shape[-1] != 3:
            raise ValueError("Expected images shape (B, H, W, 3) in BGR format.")
        if images_bgr.shape[1] != DEFAULT_IMAGE_SIZE or images_bgr.shape[2] != DEFAULT_IMAGE_SIZE:
            raise ValueError("Expected image size 224x224.")
        images_rgb = images_bgr[..., ::-1].astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        images_rgb = (images_rgb - mean) / std
        return torch.from_numpy(images_rgb).permute(0, 3, 1, 2).contiguous()

    def preprocess_bgr_images_for_model(self, images_bgr: np.ndarray) -> torch.Tensor:
        """Preprocess using HF processor when loaded, otherwise ImageNet normalisation."""
        if images_bgr.ndim != 4 or images_bgr.shape[-1] != 3:
            raise ValueError("Expected images shape (B, H, W, 3) in BGR format.")
        if images_bgr.shape[1] != DEFAULT_IMAGE_SIZE or images_bgr.shape[2] != DEFAULT_IMAGE_SIZE:
            raise ValueError("Expected image size 224x224.")

        if not self.is_hf:
            return self.preprocess_bgr_images(images_bgr)

        if self.image_processor is None:
            raise RuntimeError("HF image processor is not initialised.")

        images_rgb = images_bgr[..., ::-1]
        pil_images = [Image.fromarray(img.astype(np.uint8), mode="RGB") for img in images_rgb]
        inputs = self.image_processor(images=pil_images, return_tensors="pt")
        if "pixel_values" not in inputs:
            raise RuntimeError("HF image processor did not return pixel_values.")
        return inputs["pixel_values"]

    def _extract_attention_from_model(self, model_input: torch.Tensor) -> torch.Tensor:
        """Run forward pass and return attention matrix shape [B, heads, tokens, tokens]."""
        if self.is_hf:
            with torch.no_grad():
                out = self.model(model_input, output_attentions=True, return_dict=True)
            attn = out.attentions[-1]
            if attn.ndim == 3:
                attn = attn.unsqueeze(1)
            if attn.ndim != 4:
                raise RuntimeError(f"Unexpected HF attention tensor shape: {tuple(attn.shape)}")
            return attn

        self._last_attn = None
        with torch.no_grad():
            _ = self.model(model_input)

        if self._last_attn is None:
            attn = self._recompute_last_block_attention_timm(model_input)
            if attn is None:
                raise RuntimeError("Could not capture attention weights from last block.")
            return attn

        attn = self._last_attn
        if attn.ndim == 3:
            attn = attn.unsqueeze(1)
        if attn.ndim != 4:
            raise RuntimeError(f"Unexpected attention tensor shape: {tuple(attn.shape)}")
        return attn

    def get_fake_class_index(self, num_classes: Optional[int] = None) -> int:
        """
        Infer the fake/deepfake class index from a model's id2label mapping.

        This is intentionally centralised because benchmark, calibration, and
        live inference must agree on which logit is interpreted as
        fake_probability.  If no label metadata is available, binary models
        fall back to index 1, matching the historical project convention.
        """
        cfg = getattr(self.model, "config", None)
        id2label = getattr(cfg, "id2label", None) if cfg is not None else None
        n = int(num_classes) if num_classes is not None else None

        fake_idx: Optional[int] = None
        real_idx: Optional[int] = None
        if isinstance(id2label, dict):
            for raw_idx, raw_label in id2label.items():
                if not isinstance(raw_label, str):
                    continue
                label = raw_label.lower()
                try:
                    idx = int(raw_idx)
                except Exception:
                    continue
                if any(token in label for token in ("fake", "deepfake", "forged", "manipulated", "synthetic")):
                    fake_idx = idx
                if any(token in label for token in ("real", "realism", "genuine", "authentic", "pristine")):
                    real_idx = idx

        if fake_idx is not None:
            return fake_idx
        if real_idx is not None and n == 2:
            return 1 - real_idx
        if n == 2:
            return 1
        if n is not None:
            raise ValueError(
                "Could not infer fake class index for non-binary classifier; "
                f"id2label={id2label!r}"
            )
        return 1

    def _recompute_last_block_attention_timm(
        self, model_input: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if not (hasattr(self.model, "patch_embed") and hasattr(self.model, "blocks")):
            return None
        if len(self.model.blocks) == 0:
            return None
        last_attn_module = self.model.blocks[-1].attn
        if not all(hasattr(last_attn_module, a) for a in ("qkv", "num_heads", "scale")):
            return None

        with torch.no_grad():
            x = self.model.patch_embed(model_input)
            cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            if hasattr(self.model, "pos_embed"):
                x = x + self.model.pos_embed
            if hasattr(self.model, "pos_drop"):
                x = self.model.pos_drop(x)
            for blk in self.model.blocks[:-1]:
                x = blk(x)
            x = self.model.blocks[-1].norm1(x)
            qkv = (
                last_attn_module.qkv(x)
                .reshape(x.shape[0], x.shape[1], 3,
                         last_attn_module.num_heads,
                         x.shape[2] // last_attn_module.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q, k = qkv[0], qkv[1]
            attn = (q @ k.transpose(-2, -1)) * last_attn_module.scale
            return attn.softmax(dim=-1)

    def extract_attention_maps(self, model_input: torch.Tensor) -> AttentionMaps:
        """
        Extract CLS-to-patch attention and return upsampled heatmaps.

        Args:
            model_input: [B, 3, 224, 224] normalised tensor.

        Returns:
            AttentionMaps with cls_patch_attention (B, 196), grid_attention (B, 14, 14),
            heatmaps_224 (B, 224, 224).  zone_scores is None until M matrices are supplied.
        """
        if model_input.ndim != 4:
            raise ValueError("Expected model_input shape (B, C, H, W).")

        x = model_input.to(self.device)
        attn = self._extract_attention_from_model(x)

        if attn.shape[-1] != (1 + DEFAULT_PATCH_GRID * DEFAULT_PATCH_GRID):
            raise RuntimeError(
                f"Unexpected token count: {attn.shape[-1]}. Expected 197 (CLS + 196 patches)."
            )

        cls_to_patches = attn[:, :, 0, 1:]    # (B, heads, 196)
        cls_patch_attention = cls_to_patches.mean(dim=1)   # (B, 196)
        grid_attention = cls_patch_attention.reshape(-1, DEFAULT_PATCH_GRID, DEFAULT_PATCH_GRID)

        heatmaps = []
        for i in range(grid_attention.shape[0]):
            grid_np = grid_attention[i].detach().cpu().numpy().astype(np.float32)
            upsampled = cv2.resize(grid_np, (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
                                   interpolation=cv2.INTER_LINEAR)
            heatmaps.append(self._normalize_map(upsampled))

        return AttentionMaps(
            cls_patch_attention=cls_patch_attention.detach().cpu().numpy().astype(np.float32),
            grid_attention=grid_attention.detach().cpu().numpy().astype(np.float32),
            heatmaps_224=np.stack(heatmaps, axis=0),
            zone_scores=None,
        )

    # ------------------------------------------------------------------
    # NEW: Zone-level attention scoring  (Module 2a)
    # ------------------------------------------------------------------

    def get_zone_attention_scores(
        self,
        images_bgr: np.ndarray,
        M_matrices: np.ndarray,
    ) -> Tuple[AttentionMaps, np.ndarray]:
        """
        Compute per-zone attention scores A(z) for a batch of face images.

        This is the A(z) signal used in the dual-score fusion:
            R(z) = β·A(z) + (1−β)·S(z)

        Args:
            images_bgr  : np.ndarray shape (B, 224, 224, 3) — BGR face images.
            M_matrices  : np.ndarray shape (B, 196, 9) — one zone-patch matrix per image,
                          loaded from the JSON produced by preprocess_landmarks.py.

        Returns:
            maps      : AttentionMaps with zone_scores field populated (B, 9).
            A_scores  : np.ndarray shape (B, 9) — A(z) per zone, values in [0, 1].
        """
        if images_bgr.ndim != 4:
            raise ValueError("Expected images_bgr shape (B, 224, 224, 3).")
        if M_matrices.ndim != 3 or M_matrices.shape[1:] != (NUM_PATCHES, NUM_ZONES):
            raise ValueError(
                f"Expected M_matrices shape (B, {NUM_PATCHES}, {NUM_ZONES}), "
                f"got {M_matrices.shape}."
            )
        if images_bgr.shape[0] != M_matrices.shape[0]:
            raise ValueError("Batch size mismatch between images_bgr and M_matrices.")

        model_input = self.preprocess_bgr_images_for_model(images_bgr)
        maps = self.extract_attention_maps(model_input)

        B = images_bgr.shape[0]
        A_scores = np.zeros((B, NUM_ZONES), dtype=np.float32)
        for i in range(B):
            A_scores[i] = compute_attention_zone_scores(
                alpha=maps.cls_patch_attention[i],
                M=M_matrices[i],
            )

        maps.zone_scores = A_scores
        return maps, A_scores

    def get_zone_attention_scores_single(
        self,
        image_bgr: np.ndarray,
        M: np.ndarray,
    ) -> np.ndarray:
        """
        Convenience wrapper for a single image.

        Args:
            image_bgr : np.ndarray shape (224, 224, 3).
            M         : np.ndarray shape (196, 9).

        Returns:
            A : np.ndarray shape (9,) — attention score per zone.
        """
        _, A_scores = self.get_zone_attention_scores(
            images_bgr=np.expand_dims(image_bgr, axis=0),
            M_matrices=np.expand_dims(M, axis=0),
        )
        return A_scores[0]

    # ------------------------------------------------------------------
    # Visualisation helpers (unchanged from original)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_map(attention_map: np.ndarray) -> np.ndarray:
        min_v, max_v = float(attention_map.min()), float(attention_map.max())
        if max_v <= min_v:
            return np.zeros_like(attention_map, dtype=np.float32)
        return ((attention_map - min_v) / (max_v - min_v)).astype(np.float32)

    @staticmethod
    def overlay_heatmap_on_image(
        image_bgr: np.ndarray,
        heatmap_224: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Overlay a normalised [0,1] heatmap onto a BGR image."""
        if image_bgr.shape[:2] != (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE):
            raise ValueError("Expected image size 224x224 for overlay.")
        if heatmap_224.shape != (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE):
            raise ValueError("Expected heatmap size 224x224.")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        heat_uint8 = np.clip(heatmap_224 * 255.0, 0, 255).astype(np.uint8)
        color_map  = cv2.applyColorMap(heat_uint8, colormap)
        return cv2.addWeighted(image_bgr, 1.0 - alpha, color_map, alpha, 0.0)

    def save_side_by_side(
        self,
        image_bgr: np.ndarray,
        heatmap_overlay_bgr: np.ndarray,
        output_path: Union[str, Path],
        original_title: str = "Original Face",
        heatmap_title: str = "Attention Heatmap",
    ) -> None:
        """Save side-by-side original and overlay image."""
        if image_bgr.shape != heatmap_overlay_bgr.shape:
            raise ValueError("Original and overlay images must have same shape.")
        left  = image_bgr.copy()
        right = heatmap_overlay_bgr.copy()
        cv2.putText(left,  original_title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(right, heatmap_title,  (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        panel = np.concatenate([left, right], axis=1)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), panel)

    def visualise_zone_bar_chart(
        self,
        A_scores: np.ndarray,
        save_path: Optional[Union[str, Path]] = None,
        title: str = "Attention scores per zone A(z)",
    ) -> None:
        """
        Draw a horizontal bar chart of A(z) scores per zone.

        Colour encodes severity:
          score > 0.5  → coral   (high)
          score > 0.25 → amber   (medium)
          score ≤ 0.25 → gray    (low / suppressed)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for visualise_zone_bar_chart().")

        if A_scores.shape != (NUM_ZONES,):
            raise ValueError(f"Expected A_scores shape ({NUM_ZONES},), got {A_scores.shape}.")

        colours = []
        for s in A_scores:
            if s > 0.5:
                colours.append("#D85A30")   # coral  — high
            elif s > 0.25:
                colours.append("#BA7517")   # amber  — medium
            else:
                colours.append("#888780")   # gray   — low

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.barh(ZONE_NAMES, A_scores, color=colours, edgecolor="none", height=0.6)
        ax.set_xlabel("A(z) score")
        ax.set_title(title)
        ax.set_xlim(0, max(0.05, float(A_scores.max()) * 1.15))
        for bar, val in zip(bars, A_scores):
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha="left", fontsize=9)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    def process_batch_bgr(
        self,
        images_bgr: np.ndarray,
        alpha: float = 0.45,
    ) -> Tuple[AttentionMaps, List[np.ndarray]]:
        """Batch pipeline: preprocess → extract attention → create overlays."""
        model_input = self.preprocess_bgr_images_for_model(images_bgr)
        maps = self.extract_attention_maps(model_input)
        overlays = [
            self.overlay_heatmap_on_image(images_bgr[i], maps.heatmaps_224[i], alpha=alpha)
            for i in range(images_bgr.shape[0])
        ]
        return maps, overlays

    def close(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def __del__(self) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Utility helpers (unchanged from original)
# ---------------------------------------------------------------------------

def load_images_from_paths(image_paths: Sequence[Union[str, Path]]) -> np.ndarray:
    """Load a list of image paths into a (B, 224, 224, 3) uint8 array."""
    images = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        if img.shape[:2] != (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE):
            img = cv2.resize(img, (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
                             interpolation=cv2.INTER_LINEAR)
        images.append(img)
    return np.stack(images, axis=0)


def save_batch_side_by_side(
    images_bgr: np.ndarray,
    overlays_bgr: Sequence[np.ndarray],
    output_dir: Union[str, Path],
    stem_prefix: str = "frame",
) -> None:
    """Save side-by-side panels for a full batch/video sequence."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(images_bgr.shape[0]):
        left  = images_bgr[idx].copy()
        right = overlays_bgr[idx].copy()
        cv2.putText(left,  "Original Face",    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(right, "Attention Heatmap",(8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        panel = np.concatenate([left, right], axis=1)
        cv2.imwrite(str(out_dir / f"{stem_prefix}_{idx:05d}.jpg"), panel)


def to_colormap_image(heatmap_224: np.ndarray, colormap: str = "jet") -> np.ndarray:
    """Optional matplotlib colormap conversion (RGB output in [0, 255])."""
    heat = np.clip(heatmap_224, 0.0, 1.0)
    cmap = cm.get_cmap(colormap)
    rgba = cmap(heat)
    return (rgba[..., :3] * 255.0).astype(np.uint8)


__all__ = [
    "ZONE_NAMES", "NUM_ZONES",
    "AttentionMaps",
    "AttentionExtractor",
    "compute_attention_zone_scores",
    "load_zone_patch_matrices_from_json",
    "load_images_from_paths",
    "save_batch_side_by_side",
    "to_colormap_image",
]