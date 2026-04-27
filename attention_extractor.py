
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn


DEFAULT_MODEL_NAME = "vit_base_patch16_224"
DEFAULT_IMAGE_SIZE = 224
DEFAULT_PATCH_GRID = 14


@dataclass
class AttentionMaps:
    """
    Stores attention outputs for a batch.

    cls_patch_attention: (B, 196)
    grid_attention: (B, 14, 14)
    heatmaps_224: (B, 224, 224), normalized to [0, 1]
    """

    cls_patch_attention: np.ndarray
    grid_attention: np.ndarray
    heatmaps_224: np.ndarray


class AttentionExtractor:
    """
    ViT attention extractor for interpretability.
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
        self.model = self._load_model(model_name=model_name, pretrained=pretrained).to(self.device)
        self.model.eval()
        self._last_attn: Optional[torch.Tensor] = None
        self._hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._register_last_attention_hook()

    def _load_model(self, model_name: str, pretrained: bool) -> nn.Module:
        try:
            import timm

            model = timm.create_model(model_name, pretrained=pretrained)
            return model
        except Exception:
            # Fallback path for torchvision if timm is unavailable.
            from torchvision.models import vit_b_16
            from torchvision.models import ViT_B_16_Weights

            weights = ViT_B_16_Weights.DEFAULT if pretrained else None
            model = vit_b_16(weights=weights)
            return model

    def _register_last_attention_hook(self) -> None:
        """
        Register hook on the last transformer attention module.

        Works with timm ViT (`model.blocks[-1].attn`) and torchvision ViT
        (`model.encoder.layers.encoder_layer_11.self_attention`).
        """
        attn_module = self._resolve_last_attention_module(self.model)
        if attn_module is None:
            raise RuntimeError("Could not resolve the last attention module in this ViT model.")

        def _hook(module: nn.Module, module_inputs: Tuple[torch.Tensor, ...], module_output):
            # Handle several output formats:
            # - timm attn returns tensor output, sometimes stores attn internally.
            # - torchvision MHA may return (attn_output, attn_weights) when need_weights=True.
            captured = None

            if isinstance(module_output, tuple) and len(module_output) >= 2:
                maybe_weights = module_output[1]
                if torch.is_tensor(maybe_weights):
                    captured = maybe_weights
            elif hasattr(module, "attn_drop") and hasattr(module, "qkv"):
                # timm often does not expose weights in output directly.
                # Some timm versions expose them as `module.attn_map` after forward.
                if hasattr(module, "attn_map") and torch.is_tensor(module.attn_map):
                    captured = module.attn_map

            # Some implementations expose `attn` attribute transiently.
            if captured is None and hasattr(module, "attn") and torch.is_tensor(module.attn):
                captured = module.attn

            self._last_attn = captured

        self._hook_handle = attn_module.register_forward_hook(_hook)

    @staticmethod
    def _resolve_last_attention_module(model: nn.Module) -> Optional[nn.Module]:
        # timm ViT style
        if hasattr(model, "blocks") and len(getattr(model, "blocks")) > 0:
            last_block = model.blocks[-1]
            if hasattr(last_block, "attn"):
                return last_block.attn

        # torchvision ViT style
        if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            layers = model.encoder.layers
            if len(layers) > 0:
                last_layer = layers[-1]
                if hasattr(last_layer, "self_attention"):
                    return last_layer.self_attention

        return None

    @staticmethod
    def preprocess_bgr_images(images_bgr: np.ndarray) -> torch.Tensor:
        """
        Convert BGR uint8 batch [B,224,224,3] to normalized float tensor [B,3,224,224].
        """
        if images_bgr.ndim != 4 or images_bgr.shape[-1] != 3:
            raise ValueError("Expected images shape (B, H, W, 3) in BGR format.")
        if images_bgr.shape[1] != DEFAULT_IMAGE_SIZE or images_bgr.shape[2] != DEFAULT_IMAGE_SIZE:
            raise ValueError("Expected image size 224x224.")

        # BGR -> RGB
        images_rgb = images_bgr[..., ::-1].astype(np.float32) / 255.0
        # ImageNet normalization for ViT
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        images_rgb = (images_rgb - mean) / std
        tensor = torch.from_numpy(images_rgb).permute(0, 3, 1, 2).contiguous()
        return tensor

    def _extract_attention_from_model(self, model_input: torch.Tensor) -> torch.Tensor:
        """
        Run forward pass and return attention matrix shape [B, heads, tokens, tokens].
        """
        self._last_attn = None

        with torch.no_grad():
            _ = self.model(model_input)

        if self._last_attn is None:
            # Try a best-effort direct recomputation for timm style blocks.
            attn = self._recompute_last_block_attention_timm(model_input)
            if attn is None:
                raise RuntimeError(
                    "Could not capture attention weights from last block. "
                    "Try a timm ViT model or adjust hook resolution."
                )
            return attn

        attn = self._last_attn
        if attn.ndim == 3:
            # Some APIs collapse heads => [B, tokens, tokens], add head dim.
            attn = attn.unsqueeze(1)
        if attn.ndim != 4:
            raise RuntimeError(f"Unexpected attention tensor shape: {tuple(attn.shape)}")
        return attn

    def _recompute_last_block_attention_timm(self, model_input: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Best-effort fallback to compute last block attention for common timm ViT.
        """
        if not (hasattr(self.model, "patch_embed") and hasattr(self.model, "blocks")):
            return None
        if len(self.model.blocks) == 0:
            return None
        last_attn_module = self.model.blocks[-1].attn
        if not all(hasattr(last_attn_module, attr) for attr in ("qkv", "num_heads", "scale")):
            return None

        with torch.no_grad():
            x = self.model.patch_embed(model_input)
            cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            if hasattr(self.model, "pos_embed"):
                x = x + self.model.pos_embed
            if hasattr(self.model, "pos_drop"):
                x = self.model.pos_drop(x)

            # Pass all blocks except last.
            for blk in self.model.blocks[:-1]:
                x = blk(x)
            x = self.model.blocks[-1].norm1(x)

            qkv = (
                last_attn_module.qkv(x)
                .reshape(x.shape[0], x.shape[1], 3, last_attn_module.num_heads, x.shape[2] // last_attn_module.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q, k = qkv[0], qkv[1]
            attn = (q @ k.transpose(-2, -1)) * last_attn_module.scale
            attn = attn.softmax(dim=-1)
            return attn

    def extract_attention_maps(self, model_input: torch.Tensor) -> AttentionMaps:
        """
        Extract CLS-to-patch attention and return upsampled heatmaps.

        Args:
            model_input: [B, 3, 224, 224] normalized tensor.
        """
        if model_input.ndim != 4:
            raise ValueError("Expected model_input shape (B, C, H, W).")
        if model_input.shape[2] != DEFAULT_IMAGE_SIZE or model_input.shape[3] != DEFAULT_IMAGE_SIZE:
            raise ValueError("Expected model_input size 224x224.")

        x = model_input.to(self.device)
        attn = self._extract_attention_from_model(x)

        # Attention shape: [B, heads, tokens, tokens], tokens = 1 + 196
        if attn.shape[-1] != (1 + DEFAULT_PATCH_GRID * DEFAULT_PATCH_GRID):
            raise RuntimeError(
                f"Unexpected token count in attention: {attn.shape[-1]}. "
                "Expected 197 (CLS + 196 patches)."
            )

        # CLS token extraction:
        # First query token (CLS) attending to all keys.
        # Keep patch keys only, skip CLS->CLS position 0.
        cls_to_all = attn[:, :, 0, :]  # [B, heads, 197]
        cls_to_patches = cls_to_all[:, :, 1:]  # [B, heads, 196]

        cls_patch_attention = cls_to_patches.mean(dim=1)  # [B, 196]
        grid_attention = cls_patch_attention.reshape(-1, DEFAULT_PATCH_GRID, DEFAULT_PATCH_GRID)

        heatmaps = []
        for i in range(grid_attention.shape[0]):
            grid_np = grid_attention[i].detach().cpu().numpy().astype(np.float32)
            # Bilinear upsampling from 14x14 -> 224x224
            upsampled = cv2.resize(
                grid_np,
                (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
                interpolation=cv2.INTER_LINEAR,
            )
            upsampled = self._normalize_map(upsampled)
            heatmaps.append(upsampled)

        return AttentionMaps(
            cls_patch_attention=cls_patch_attention.detach().cpu().numpy().astype(np.float32),
            grid_attention=grid_attention.detach().cpu().numpy().astype(np.float32),
            heatmaps_224=np.stack(heatmaps, axis=0),
        )

    @staticmethod
    def _normalize_map(attention_map: np.ndarray) -> np.ndarray:
        min_v = float(attention_map.min())
        max_v = float(attention_map.max())
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
        """
        Overlay a normalized [0,1] heatmap onto a BGR image.
        """
        if image_bgr.shape[:2] != (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE):
            raise ValueError("Expected image size 224x224 for overlay.")
        if heatmap_224.shape != (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE):
            raise ValueError("Expected heatmap size 224x224.")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1].")

        heat_uint8 = np.clip(heatmap_224 * 255.0, 0, 255).astype(np.uint8)
        color_map = cv2.applyColorMap(heat_uint8, colormap)
        blended = cv2.addWeighted(image_bgr, 1.0 - alpha, color_map, alpha, 0.0)
        return blended

    def save_side_by_side(
        self,
        image_bgr: np.ndarray,
        heatmap_overlay_bgr: np.ndarray,
        output_path: Union[str, Path],
        original_title: str = "Original Face",
        heatmap_title: str = "Attention Heatmap",
    ) -> None:
        """
        Save side-by-side original and overlay image.
        """
        if image_bgr.shape != heatmap_overlay_bgr.shape:
            raise ValueError("Original and overlay images must have same shape.")

        left = image_bgr.copy()
        right = heatmap_overlay_bgr.copy()
        cv2.putText(left, original_title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(right, heatmap_title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        panel = np.concatenate([left, right], axis=1)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), panel)

    def process_batch_bgr(
        self,
        images_bgr: np.ndarray,
        alpha: float = 0.45,
    ) -> Tuple[AttentionMaps, List[np.ndarray]]:
        """
        Batch pipeline:
        1) preprocess BGR batch
        2) extract attention heatmaps
        3) create overlays per frame
        """
        model_input = self.preprocess_bgr_images(images_bgr)
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


def load_images_from_paths(image_paths: Sequence[Union[str, Path]]) -> np.ndarray:
    """
    Utility helper for batch processing image files.
    """
    images = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        if img.shape[:2] != (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE):
            img = cv2.resize(img, (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        images.append(img)
    return np.stack(images, axis=0)


def save_batch_side_by_side(
    images_bgr: np.ndarray,
    overlays_bgr: Sequence[np.ndarray],
    output_dir: Union[str, Path],
    stem_prefix: str = "frame",
) -> None:
    """
    Save side-by-side panels for a full batch/video sequence.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(images_bgr.shape[0]):
        left = images_bgr[idx].copy()
        right = overlays_bgr[idx].copy()
        cv2.putText(left, "Original Face", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(right, "Attention Heatmap", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        panel = np.concatenate([left, right], axis=1)
        cv2.imwrite(str(out_dir / f"{stem_prefix}_{idx:05d}.jpg"), panel)


def to_colormap_image(heatmap_224: np.ndarray, colormap: str = "jet") -> np.ndarray:
    """
    Optional matplotlib colormap conversion helper (RGB output in [0,255]).
    """
    heat = np.clip(heatmap_224, 0.0, 1.0)
    cmap = cm.get_cmap(colormap)
    rgba = cmap(heat)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    return rgb


__all__ = [
    "AttentionMaps",
    "AttentionExtractor",
    "load_images_from_paths",
    "save_batch_side_by_side",
    "to_colormap_image",
]

