# baseline_heatmap_generator.py

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

import torch
import torch.nn.functional as F

from attention_extractor import AttentionExtractor, DEFAULT_HF_DEEPFAKE_MODEL
from semantic_patch_mapper import compute_shap_zone_scores

_PATCH_GRID = 14
_PATCH_SIZE = 16
_NUM_PATCHES = _PATCH_GRID * _PATCH_GRID  # 196


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate refined ViT attention heatmaps for an image or a sampled video grid."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image",
        type=Path,
        help="Path to input face image.",
    )
    input_group.add_argument(
        "--images",
        type=Path,
        nargs="+",
        help="One or more input face images. Prints a score per image.",
    )
    input_group.add_argument(
        "--video",
        type=Path,
        help="Path to input video. The script samples 4 frames and saves a 2x4 panel.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/baseline_attention_heatmap.jpg"),
        help="Path for side-by-side output image.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_HF_DEEPFAKE_MODEL,
        help="Model id/name. Supports Hugging Face ids (e.g. hamzenium/ViT-Deepfake-Classifier) or timm ViT names.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Overlay transparency in [0,1]. Default: 0.6",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display output panel in an OpenCV window.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional torch device, e.g. 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=4,
        help="Number of evenly sampled video frames. Default: 4",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Fake-probability threshold for FAKE/REAL label. Default: 0.5",
    )
    parser.add_argument(
        "--fake-class-index",
        type=int,
        default=None,
        help=(
            "Class index to treat as 'Fake' when extracting probability. "
            "If omitted, uses index 1 only when the head is binary (num_classes==2). "
            "For ImageNet-pretrained models (1000 classes), you must set this or load a binary deepfake checkpoint."
        ),
    )
    return parser.parse_args()


def read_and_resize_image(image_path: Path, target_size: int = 224) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    if image.shape[:2] != (target_size, target_size):
        image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return image


def _normalize_map01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    min_v = float(x.min())
    max_v = float(x.max())
    if max_v <= min_v:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - min_v) / (max_v - min_v)).astype(np.float32)


def _build_refined_heatmap(
    image_tensor: torch.Tensor,
    extractor: AttentionExtractor,
) -> tuple[np.ndarray, float, int]:
    """
    Internal helper that reuses a preloaded AttentionExtractor/model.
    """
    if image_tensor.ndim != 4 or image_tensor.shape[0] != 1 or image_tensor.shape[1] != 3:
        raise ValueError("image_tensor must have shape [1,3,H,W].")

    model = extractor.model
    device = extractor.device
    model.eval()
    x = image_tensor.to(device)

    # Forward with attention matrices that keep the graph.
    if getattr(extractor, "is_hf", False):
        # Some HF checkpoints/configs return empty `attentions` unless the config
        # explicitly enables it, even when passing output_attentions=True.
        cfg = getattr(model, "config", None)
        if cfg is not None:
            try:
                cfg.output_attentions = True
                cfg.return_dict = True
            except Exception:
                pass
        out = model(x, output_attentions=True, return_dict=True)
        logits = out.logits
        attn = getattr(out, "attentions", None)
        attn_mats = list(attn) if attn is not None else []  # list of [B, heads, T, T]
    elif hasattr(model, "blocks") and hasattr(model, "patch_embed"):
        logits, attn_mats = _timm_vit_forward_with_attn(model, x)
    else:
        logits, attn_mats = _torchvision_vit_forward_with_attn(model, x)

    # If we didn't actually receive any attention tensors, grad-based rollout cannot work.
    if not attn_mats:
        raise RuntimeError(
            "Model did not return any attention matrices (empty `attentions`). "
            "Grad-weighted attention rollout requires attention tensors to compute gradients. "
            "If you're using a Hugging Face model, try a ViTForImageClassification checkpoint that supports "
            "`output_attentions=True`, or use a timm/torchvision ViT model for refined heatmaps."
        )

    # Probability extraction from classification head logits.
    probs = torch.softmax(logits, dim=-1)

    num_classes = logits.shape[-1]
    if getattr(extractor, "_fake_class_index", None) is not None:
        fake_idx = int(extractor._fake_class_index)
    else:
        fake_idx = extractor.get_fake_class_index(num_classes=num_classes)

    if not (0 <= fake_idx < num_classes):
        raise ValueError(f"Fake class index {fake_idx} is out of range for num_classes={num_classes}.")
    fake_prob = float(probs[0, fake_idx].detach().cpu().item())
    score = logits[:, fake_idx].sum()

    grads = torch.autograd.grad(score, attn_mats, retain_graph=False, create_graph=False)

    # Grad-weighted Attention Rollout with residual mixing:
    # A_l = 0.5 I + 0.5 (W_l ⊙ ReLU(dscore/dW_l))
    B, _, T, _ = attn_mats[0].shape
    eye = torch.eye(T, device=device, dtype=attn_mats[0].dtype).unsqueeze(0)
    joint = eye.expand(B, -1, -1).clone()

    for attn, grad in zip(attn_mats, grads):
        attn_mean = attn.mean(dim=1)
        grad_mean = torch.relu(grad.mean(dim=1))
        weighted = attn_mean * grad_mean
        a = 0.5 * eye + 0.5 * weighted
        a = a / (a.sum(dim=-1, keepdim=True) + 1e-6)
        joint = a @ joint

    cls_to_patches = joint[0, 0, 1:]
    grid_14 = cls_to_patches.reshape(14, 14).detach().float().cpu().numpy()

    heat_224 = cv2.resize(grid_14, (224, 224), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    heat_224 = cv2.GaussianBlur(heat_224, ksize=(0, 0), sigmaX=7.0, sigmaY=7.0)

    thr = float(np.percentile(heat_224, 30.0))
    heat_224[heat_224 < thr] = 0.0

    return _normalize_map01(heat_224), fake_prob, int(fake_idx)


def _build_raw_patch_grid(
    image_tensor: torch.Tensor,
    extractor: AttentionExtractor,
) -> tuple[np.ndarray, float, int]:
    """
    Same grad-weighted attention rollout as _build_refined_heatmap(), but returns
    the raw 14x14 patch grid BEFORE upsampling, blurring, or percentile thresholding.

    The visualisation path (_build_refined_heatmap) applies a 7px Gaussian blur and
    a 30th-percentile hard cutoff to produce a clean-looking overlay. Both steps are
    appropriate for visual display but destructive for zone-level quantitative scoring:
    blurring smears signal across patch (and zone) boundaries, and the percentile cutoff
    discards low-but-informative values that may still carry discriminative signal
    between symmetric zones (e.g. left_eye vs right_eye).

    Returns:
        grid_14: np.ndarray (14, 14) float32, normalised to [0, 1], NO blur, NO threshold.
        fake_prob: float
        fake_idx: int
    """
    if image_tensor.ndim != 4 or image_tensor.shape[0] != 1 or image_tensor.shape[1] != 3:
        raise ValueError("image_tensor must have shape [1,3,H,W].")

    model = extractor.model
    device = extractor.device
    model.eval()
    x = image_tensor.to(device)

    if getattr(extractor, "is_hf", False):
        cfg = getattr(model, "config", None)
        if cfg is not None:
            try:
                cfg.output_attentions = True
                cfg.return_dict = True
            except Exception:
                pass
        out = model(x, output_attentions=True, return_dict=True)
        logits = out.logits
        attn = getattr(out, "attentions", None)
        attn_mats = list(attn) if attn is not None else []
    elif hasattr(model, "blocks") and hasattr(model, "patch_embed"):
        logits, attn_mats = _timm_vit_forward_with_attn(model, x)
    else:
        logits, attn_mats = _torchvision_vit_forward_with_attn(model, x)

    if not attn_mats:
        raise RuntimeError("Model did not return any attention matrices.")

    probs = torch.softmax(logits, dim=-1)
    num_classes = logits.shape[-1]

    if getattr(extractor, "_fake_class_index", None) is not None:
        fake_idx = int(extractor._fake_class_index)
    else:
        fake_idx = extractor.get_fake_class_index(num_classes=num_classes)

    fake_prob = float(probs[0, fake_idx].detach().cpu().item())
    score = logits[:, fake_idx].sum()

    grads = torch.autograd.grad(score, attn_mats, retain_graph=False, create_graph=False)

    B, _, T, _ = attn_mats[0].shape
    eye = torch.eye(T, device=device, dtype=attn_mats[0].dtype).unsqueeze(0)
    joint = eye.expand(B, -1, -1).clone()

    for attn, grad in zip(attn_mats, grads):
        attn_mean = attn.mean(dim=1)
        grad_mean = torch.relu(grad.mean(dim=1))
        weighted = attn_mean * grad_mean
        a = 0.5 * eye + 0.5 * weighted
        a = a / (a.sum(dim=-1, keepdim=True) + 1e-6)
        joint = a @ joint

    cls_to_patches = joint[0, 0, 1:]
    grid_14 = cls_to_patches.reshape(14, 14).detach().float().cpu().numpy()

    # Normalise only -- no blur, no percentile threshold
    grid_14 = _normalize_map01(grid_14)

    return grid_14, fake_prob, int(fake_idx)


def extract_patch_gradient_scores_raw(
    image_bgr: np.ndarray,
    extractor: AttentionExtractor,
) -> np.ndarray:
    """
    Unsmoothed alternative to extract_patch_gradient_scores().

    Returns the native 14x14 grad-rollout grid directly as 196 patch scores
    (row-major), WITHOUT the upsample->blur->threshold->downsample roundtrip.
    Use this for zone-level quantitative scoring (S(z) fusion); use
    extract_patch_gradient_scores() / _build_refined_heatmap() for visual overlays.

    Args:
        image_bgr: BGR uint8/float array, shape (224, 224, 3).
        extractor: Preloaded AttentionExtractor.

    Returns:
        phi: float32 array of shape (196,) — native patch-grid scores, row-major.
    """
    if image_bgr.shape != (224, 224, 3):
        raise ValueError(f"Expected image_bgr shape (224, 224, 3), got {image_bgr.shape}.")
    model_input = extractor.preprocess_bgr_images_for_model(np.expand_dims(image_bgr, axis=0))
    grid_14, _, _ = _build_raw_patch_grid(image_tensor=model_input, extractor=extractor)
    return grid_14.reshape(NUM_PATCHES).astype(np.float32)


def _heatmap224_to_patch_phi(heatmap_224: np.ndarray) -> np.ndarray:
    """Average each 16×16 ViT patch block of a 224×224 heatmap → (196,) row-major φ."""
    if heatmap_224.shape != (224, 224):
        raise ValueError(f"Expected heatmap shape (224, 224), got {heatmap_224.shape}.")
    blocks = heatmap_224.reshape(_PATCH_GRID, _PATCH_SIZE, _PATCH_GRID, _PATCH_SIZE)
    return blocks.mean(axis=(1, 3)).reshape(_NUM_PATCHES).astype(np.float32)


def extract_patch_gradient_scores(
    image_bgr: np.ndarray,
    extractor: AttentionExtractor,
) -> np.ndarray:
    """
    Run grad-weighted attention rollout and reduce the 224×224 heatmap to patch scores φ.

    Args:
        image_bgr: BGR uint8/float array, shape (224, 224, 3).
        extractor: Preloaded AttentionExtractor (model + preprocess).

    Returns:
        phi: float32 array of shape (196,) — mean heat per 16×16 ViT patch (row-major).
    """
    if image_bgr.shape != (224, 224, 3):
        raise ValueError(f"Expected image_bgr shape (224, 224, 3), got {image_bgr.shape}.")
    model_input = extractor.preprocess_bgr_images_for_model(np.expand_dims(image_bgr, axis=0))
    heatmap_224, _, _ = _build_refined_heatmap(image_tensor=model_input, extractor=extractor)
    return _heatmap224_to_patch_phi(heatmap_224)


def extract_full_pipeline_inputs(
    image_bgr: np.ndarray,
    extractor: AttentionExtractor,
) -> dict:
    """
    Single forward pass — returns all signals needed by the SAEDF pipeline.

    Runs the grad-weighted attention rollout exactly once, capturing:
      - fake_probability : classifier softmax score for the fake class.
      - phi              : (196,) blurred gradient-rollout patch scores (S(z) input).
                          Uses the same blur + 30th-percentile threshold as
                          extract_patch_gradient_scores() so the signal is
                          consistent with the calibrated beta in fusion_config.json.
      - alpha            : (196,) last-layer mean CLS→patch attention (A(z) input),
                          normalised to [0, 1].

    Using this function instead of calling extract_attention_maps() and
    extract_patch_gradient_scores() separately avoids two model forward passes
    and guarantees all three signals come from the same computation.

    Args:
        image_bgr: (224, 224, 3) BGR uint8/float32 array.
        extractor: Preloaded AttentionExtractor (use the HF deepfake model so
                   fake_probability is meaningful).

    Returns:
        dict with keys:
            "fake_probability" : float in [0, 1]
            "phi"              : np.ndarray shape (196,) float32
            "alpha"            : np.ndarray shape (196,) float32
            "heatmap_224"      : np.ndarray shape (224, 224) float32, normalised
                                  [0, 1] -- the same blurred/thresholded
                                  grad-rollout heatmap used to derive `phi`.
                                  Callers that need a visual overlay (e.g. the
                                  inference server) should reuse this instead
                                  of re-running the model via
                                  _build_refined_heatmap(), which would repeat
                                  an identical forward+backward pass.
    """
    if image_bgr.shape != (224, 224, 3):
        raise ValueError(
            f"Expected image_bgr shape (224, 224, 3), got {image_bgr.shape}."
        )

    model        = extractor.model
    device       = extractor.device
    model.eval()
    model_input  = extractor.preprocess_bgr_images_for_model(
        np.expand_dims(image_bgr, axis=0)
    )
    x = model_input.to(device)

    # ------------------------------------------------------------------ #
    # Forward pass — capture logits + all layer attention matrices        #
    # ------------------------------------------------------------------ #
    if getattr(extractor, "is_hf", False):
        cfg = getattr(model, "config", None)
        if cfg is not None:
            try:
                cfg.output_attentions = True
                cfg.return_dict = True
            except Exception:
                pass
        out       = model(x, output_attentions=True, return_dict=True)
        logits    = out.logits
        attn      = getattr(out, "attentions", None)
        attn_mats = list(attn) if attn is not None else []
    elif hasattr(model, "blocks") and hasattr(model, "patch_embed"):
        logits, attn_mats = _timm_vit_forward_with_attn(model, x)
    else:
        logits, attn_mats = _torchvision_vit_forward_with_attn(model, x)

    if not attn_mats:
        raise RuntimeError(
            "Model did not return any attention matrices. "
            "Use a ViTForImageClassification checkpoint that supports "
            "output_attentions=True."
        )

    # ------------------------------------------------------------------ #
    # Fake probability from classification logits                         #
    # ------------------------------------------------------------------ #
    probs       = torch.softmax(logits, dim=-1)
    num_classes = logits.shape[-1]
    if getattr(extractor, "_fake_class_index", None) is not None:
        fake_idx = int(extractor._fake_class_index)
    else:
        fake_idx = extractor.get_fake_class_index(num_classes=num_classes)
    fake_prob = float(probs[0, fake_idx].detach().cpu().item())

    # ------------------------------------------------------------------ #
    # CLS→patch attention from the last transformer layer (A(z) signal)  #
    # Shape of attn_mats[-1]: [B, heads, T, T] where T = 197             #
    # ------------------------------------------------------------------ #
    last_attn   = attn_mats[-1]           # [B, heads, T, T]
    alpha_raw   = (
        last_attn[0, :, 0, 1:]           # [heads, 196]
        .mean(dim=0)                      # [196]
        .detach().float().cpu().numpy()
    )
    alpha = _normalize_map01(alpha_raw)   # float32, [0, 1]

    # ------------------------------------------------------------------ #
    # Grad-weighted attention rollout → phi (S(z) signal)                 #
    # Matches the blurred calibration path in recalibrate_v2.py           #
    # ------------------------------------------------------------------ #
    score = logits[:, fake_idx].sum()
    grads = torch.autograd.grad(
        score, attn_mats, retain_graph=False, create_graph=False
    )

    B, _, T, _ = attn_mats[0].shape
    eye  = torch.eye(T, device=device, dtype=attn_mats[0].dtype).unsqueeze(0)
    joint = eye.expand(B, -1, -1).clone()

    for attn_l, grad_l in zip(attn_mats, grads):
        attn_mean = attn_l.mean(dim=1)
        grad_mean = torch.relu(grad_l.mean(dim=1))
        weighted  = attn_mean * grad_mean
        a         = 0.5 * eye + 0.5 * weighted
        a         = a / (a.sum(dim=-1, keepdim=True) + 1e-6)
        joint     = a @ joint

    cls_to_patches = joint[0, 0, 1:]
    grid_14 = cls_to_patches.reshape(14, 14).detach().float().cpu().numpy()

    # Blur + 30th-percentile threshold (matches calibration path)
    heat_224 = cv2.resize(
        grid_14, (224, 224), interpolation=cv2.INTER_CUBIC
    ).astype(np.float32)
    heat_224 = cv2.GaussianBlur(heat_224, ksize=(0, 0), sigmaX=7.0, sigmaY=7.0)
    thr = float(np.percentile(heat_224, 30.0))
    heat_224[heat_224 < thr] = 0.0
    heat_224 = _normalize_map01(heat_224)
    phi = _heatmap224_to_patch_phi(heat_224)   # (196,) float32

    return {
        "fake_probability": fake_prob,
        "phi":   phi,
        "alpha": alpha,
        "heatmap_224": heat_224,
    }


def get_shap_zone_scores(
    image_bgr: np.ndarray,
    M: np.ndarray,
    extractor: AttentionExtractor,
) -> np.ndarray:
    """
    Patch gradient scores φ fused into nine semantic zones via overlap matrix M.

    Args:
        image_bgr: BGR array, shape (224, 224, 3).
        M: Zone-patch overlap matrix, shape (196, 9).
        extractor: Preloaded AttentionExtractor.

    Returns:
        S: float32 array of shape (9,) — min-max normalised SHAP zone scores.
    """
    phi = extract_patch_gradient_scores(image_bgr=image_bgr, extractor=extractor)
    return compute_shap_zone_scores(phi, M)


def _sample_video_frames(video_path: Path, num_frames: int, target_size: int = 224) -> list[np.ndarray]:
    if num_frames <= 0:
        raise ValueError("--num-frames must be positive.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise RuntimeError(f"Could not determine frame count for video: {video_path}")

        sample_count = min(num_frames, frame_count)
        indices = np.linspace(0, frame_count - 1, sample_count, dtype=int)
        frames: list[np.ndarray] = []

        for idx in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Failed to read frame {idx} from video: {video_path}")
            if frame.shape[:2] != (target_size, target_size):
                frame = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            frames.append(frame)

        return frames
    finally:
        capture.release()


def _save_grid_panel(
    originals_bgr: list[np.ndarray],
    overlays_bgr: list[np.ndarray],
    output_path: Path,
) -> None:
    if len(originals_bgr) != len(overlays_bgr):
        raise ValueError("Original and overlay counts must match.")
    if not originals_bgr:
        raise ValueError("At least one frame is required to build the panel.")

    cell_count = len(originals_bgr)
    top_row = np.concatenate(originals_bgr, axis=1)
    bottom_row = np.concatenate(overlays_bgr, axis=1)
    panel = np.concatenate([top_row, bottom_row], axis=0)

    for i in range(cell_count):
        x = i * originals_bgr[i].shape[1]
        cv2.putText(panel, f"Frame {i + 1}", (x + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            panel,
            f"Heatmap {i + 1}",
            (x + 8, originals_bgr[i].shape[0] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), panel)


def _timm_vit_forward_with_attn(
    model: torch.nn.Module, x: torch.Tensor
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Forward a timm ViT while capturing per-block attention matrices (with grad graph).

    Returns:
      logits: [B, num_classes]
      attn_mats: list of [B, heads, T, T] (softmax attention), one per block.
    """
    if not (hasattr(model, "patch_embed") and hasattr(model, "blocks")):
        raise RuntimeError("Model does not look like a timm ViT (missing patch_embed/blocks).")
    if len(model.blocks) == 0:
        raise RuntimeError("Model has no transformer blocks.")

    B = x.shape[0]
    x = model.patch_embed(x)  # [B, N, C]

    if not hasattr(model, "cls_token"):
        raise RuntimeError("Expected timm ViT to have cls_token.")
    cls_token = model.cls_token.expand(B, -1, -1)
    x = torch.cat((cls_token, x), dim=1)  # [B, 1+N, C]

    if hasattr(model, "pos_embed"):
        x = x + model.pos_embed
    if hasattr(model, "pos_drop"):
        x = model.pos_drop(x)

    attn_mats: list[torch.Tensor] = []

    for blk in model.blocks:
        # timm Block variants commonly use drop_path1/drop_path2 instead of drop_path.
        if not (hasattr(blk, "norm1") and hasattr(blk, "attn")):
            raise RuntimeError("Unsupported timm ViT block structure (missing norm1/attn).")

        h = blk.norm1(x)
        attn = blk.attn
        if not all(hasattr(attn, a) for a in ("qkv", "num_heads", "scale", "proj")):
            raise RuntimeError("Unsupported timm ViT attention structure (missing qkv/num_heads/scale/proj).")

        qkv = (
            attn.qkv(h)
            .reshape(B, h.shape[1], 3, attn.num_heads, h.shape[2] // attn.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, T, dim]
        attn_w = (q @ k.transpose(-2, -1)) * attn.scale
        attn_w = attn_w.softmax(dim=-1)  # [B, heads, T, T]
        attn_mats.append(attn_w)

        attn_out = attn_w @ v  # [B, heads, T, dim]
        attn_out = attn_out.transpose(1, 2).reshape(B, h.shape[1], h.shape[2])  # [B, T, C]
        attn_out = attn.proj(attn_out)
        if hasattr(attn, "proj_drop"):
            attn_out = attn.proj_drop(attn_out)

        drop_path1 = getattr(blk, "drop_path1", None)
        if drop_path1 is None:
            drop_path1 = getattr(blk, "drop_path", None)
        if drop_path1 is None:
            drop_path1 = torch.nn.Identity()
        x = x + drop_path1(attn_out)

        if not (hasattr(blk, "norm2") and hasattr(blk, "mlp")):
            raise RuntimeError("Unsupported timm ViT block structure (missing norm2/mlp).")
        mlp_out = blk.mlp(blk.norm2(x))
        drop_path2 = getattr(blk, "drop_path2", None)
        if drop_path2 is None:
            drop_path2 = getattr(blk, "drop_path", None)
        if drop_path2 is None:
            drop_path2 = torch.nn.Identity()
        x = x + drop_path2(mlp_out)

    if hasattr(model, "norm"):
        x = model.norm(x)
    if hasattr(model, "fc_norm"):
        # Some timm ViT variants use fc_norm + head on pooled token.
        x_cls = model.fc_norm(x[:, 0])
    else:
        x_cls = x[:, 0]

    if hasattr(model, "head_drop"):
        x_cls = model.head_drop(x_cls)
    if not hasattr(model, "head"):
        raise RuntimeError("Expected timm ViT to have classification head (`head`).")
    logits = model.head(x_cls)
    return logits, attn_mats


def _torchvision_vit_forward_with_attn(
    model: torch.nn.Module, x: torch.Tensor
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Forward a torchvision VisionTransformer while capturing per-layer attention matrices
    (with grad graph).

    Returns:
      logits: [B, num_classes]
      attn_mats: list of [B, heads, T, T] (softmax attention), one per encoder layer.
    """
    if not (hasattr(model, "_process_input") and hasattr(model, "encoder") and hasattr(model, "class_token")):
        raise RuntimeError("Model does not look like a torchvision VisionTransformer.")
    if not hasattr(model.encoder, "layers"):
        raise RuntimeError("torchvision ViT encoder is missing layers.")

    B = x.shape[0]
    x = model._process_input(x)  # [B, 196, C]

    cls = model.class_token.expand(B, -1, -1)  # [B,1,C]
    x = torch.cat([cls, x], dim=1)  # [B,197,C]

    if hasattr(model.encoder, "pos_embedding"):
        x = x + model.encoder.pos_embedding
    if hasattr(model.encoder, "dropout"):
        x = model.encoder.dropout(x)

    attn_mats: list[torch.Tensor] = []

    # Encoder blocks
    for blk in model.encoder.layers.children():
        if not all(hasattr(blk, a) for a in ("ln_1", "self_attention", "dropout", "ln_2", "mlp")):
            raise RuntimeError("Unsupported torchvision ViT encoder block structure.")

        h = blk.ln_1(x)
        sa = blk.self_attention
        if not getattr(sa, "batch_first", False):
            raise RuntimeError("Expected torchvision MultiheadAttention with batch_first=True.")
        if sa.in_proj_weight is None or sa.out_proj is None:
            raise RuntimeError("Unsupported torchvision MultiheadAttention configuration.")

        # In-projection to QKV: [B,T,3C]
        qkv = F.linear(h, sa.in_proj_weight, sa.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        heads = sa.num_heads
        head_dim = sa.head_dim
        T = h.shape[1]

        q = q.reshape(B, T, heads, head_dim).transpose(1, 2)  # [B,H,T,D]
        k = k.reshape(B, T, heads, head_dim).transpose(1, 2)
        v = v.reshape(B, T, heads, head_dim).transpose(1, 2)

        scale = float(head_dim) ** -0.5
        attn_w = (q @ k.transpose(-2, -1)) * scale  # [B,H,T,T]
        attn_w = attn_w.softmax(dim=-1)
        attn_mats.append(attn_w)

        attn_out = attn_w @ v  # [B,H,T,D]
        attn_out = attn_out.transpose(1, 2).reshape(B, T, heads * head_dim)  # [B,T,C]
        attn_out = sa.out_proj(attn_out)

        x = x + blk.dropout(attn_out)
        x = x + blk.mlp(blk.ln_2(x))

    if hasattr(model.encoder, "ln"):
        x = model.encoder.ln(x)

    x_cls = x[:, 0]
    if not hasattr(model, "heads"):
        raise RuntimeError("Expected torchvision ViT to have `heads` classification module.")
    logits = model.heads(x_cls)
    return logits, attn_mats


def get_refined_heatmap(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Produce a refined deepfake heatmap using grad-weighted attention rollout.

    Args:
      image_tensor: torch.Tensor shaped [1,3,224,224] (float). Should be normalized to the
        model's expected input space (e.g., ImageNet normalization for ViT).

    Returns:
      heatmap_224: np.ndarray [224,224] float32 in [0,1]
    """
    if image_tensor.ndim != 4 or image_tensor.shape[0] != 1 or image_tensor.shape[1] != 3:
        raise ValueError("image_tensor must have shape [1,3,H,W].")

    extractor = AttentionExtractor(pretrained=True, device=image_tensor.device)
    try:
        heatmap, _, _ = _build_refined_heatmap(image_tensor=image_tensor, extractor=extractor)
        return heatmap
    finally:
        extractor.close()


def _generate_refined_overlay(
    image_bgr: np.ndarray,
    extractor: AttentionExtractor,
    alpha: float,
) -> tuple[np.ndarray, float]:
    model_input = extractor.preprocess_bgr_images_for_model(np.expand_dims(image_bgr, axis=0))
    heatmap, fake_prob, _ = _build_refined_heatmap(image_tensor=model_input, extractor=extractor)
    overlay = extractor.overlay_heatmap_on_image(image_bgr=image_bgr, heatmap_224=heatmap, alpha=alpha)
    return overlay, fake_prob


def _label_from_score(fake_prob: float, threshold: float) -> str:
    return "FAKE" if fake_prob >= threshold else "REAL"


def _print_detection_report(input_name: str, fake_prob: float, threshold: float) -> None:
    status = _label_from_score(fake_prob=fake_prob, threshold=threshold)
    conf_pct = 100.0 * float(fake_prob)
    print("--- DEEPFAKE DETECTION REPORT ---")
    print(f"Input File: {input_name}")
    print(f"Status: {status}")
    print(f"Confidence Score: {conf_pct:.2f}%")
    print("---------------------------------")


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be in [0,1].")
    if not (0.0 <= args.threshold <= 1.0):
        raise ValueError("--threshold must be in [0,1].")

    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

    extractor = AttentionExtractor(
        model_name=args.model_name,
        pretrained=True,
        device=args.device,
    )
    # Thread through chosen fake index without changing AttentionExtractor API.
    extractor._fake_class_index = args.fake_class_index

    try:
        # Be forgiving: if a video file is accidentally passed to --image, treat it as video.
        if args.image is not None and args.image.suffix.lower() in video_exts:
            args.video = args.image
            args.image = None

        if args.image is not None:
            image_bgr = read_and_resize_image(args.image, target_size=224)
            overlay, fake_prob = _generate_refined_overlay(image_bgr=image_bgr, extractor=extractor, alpha=args.alpha)
            extractor.save_side_by_side(
                image_bgr=image_bgr,
                heatmap_overlay_bgr=overlay,
                output_path=args.output,
                original_title="Original Input Frame",
                heatmap_title="Attention Heatmap",
            )
            if args.fake_class_index is None and getattr(extractor.model, "num_classes", None) not in (None, 2):
                print(
                    f"Warning: model head appears non-binary; score may be meaningless. "
                    f"Set --fake-class-index or load a 2-class deepfake model."
                )
            _print_detection_report(str(args.image), fake_prob=fake_prob, threshold=args.threshold)
            print(f"Saved refined heatmap panel to: {args.output}")
        elif args.images is not None:
            for p in args.images:
                image_bgr = read_and_resize_image(p, target_size=224)
                overlay, fake_prob = _generate_refined_overlay(image_bgr=image_bgr, extractor=extractor, alpha=args.alpha)
                if args.fake_class_index is None and getattr(extractor.model, "num_classes", None) not in (None, 2):
                    print(
                        f"Warning: model head appears non-binary; score may be meaningless. "
                        f"Set --fake-class-index or load a 2-class deepfake model."
                    )
                _print_detection_report(str(p), fake_prob=fake_prob, threshold=args.threshold)
            # No panel output for multi-image mode unless user explicitly requests later.
        else:
            frames_bgr = _sample_video_frames(args.video, num_frames=args.num_frames, target_size=224)
            overlays: list[np.ndarray] = []
            for i, frame in enumerate(frames_bgr):
                overlay, fake_prob = _generate_refined_overlay(image_bgr=frame, extractor=extractor, alpha=args.alpha)
                overlays.append(overlay)
                if args.fake_class_index is None and getattr(extractor.model, "num_classes", None) not in (None, 2):
                    print(
                        f"Warning: model head appears non-binary; score may be meaningless. "
                        f"Set --fake-class-index or load a 2-class deepfake model."
                    )
                _print_detection_report(f"{args.video} (frame {i + 1}/{len(frames_bgr)})", fake_prob=fake_prob, threshold=args.threshold)
            _save_grid_panel(frames_bgr, overlays, args.output)
            print(f"Saved sampled video heatmap grid to: {args.output}")
    finally:
        extractor.close()

    if args.show:
        panel = cv2.imread(str(args.output))
        if panel is None:
            raise RuntimeError(f"Failed to read saved panel from: {args.output}")
        cv2.imshow("Refined ViT Attention Heatmap", panel)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def _self_test() -> None:
    """Smoke tests for patch φ extraction and SHAP zone scoring."""
    print("[_self_test] heatmap -> patch phi ...")
    heat = np.zeros((224, 224), dtype=np.float32)
    heat[:16, :16] = 1.0
    phi = _heatmap224_to_patch_phi(heat)
    assert phi.shape == (_NUM_PATCHES,)
    assert abs(float(phi[0]) - 1.0) < 1e-5
    assert float(phi[1:].max()) < 1e-5
    phi_uniform = _heatmap224_to_patch_phi(np.ones((224, 224), dtype=np.float32))
    assert np.allclose(phi_uniform, 1.0)
    print("  OK")

    print("[_self_test] compute_shap_zone_scores wiring ...")
    rng = np.random.default_rng(0)
    M = rng.random((_NUM_PATCHES, 9), dtype=np.float32)
    M /= M.sum(axis=0, keepdims=True) + 1e-8
    phi_lin = np.linspace(0.0, 1.0, _NUM_PATCHES, dtype=np.float32)
    S = compute_shap_zone_scores(phi_lin, M)
    assert S.shape == (9,)
    assert float(S.min()) >= 0.0 and float(S.max()) <= 1.0 + 1e-5
    print("  OK")

    test_image = Path(__file__).resolve().parent / "inputs" / "1.jpg"
    if test_image.is_file():
        print(f"[_self_test] integration ({test_image.name}) ...")
        image_bgr = read_and_resize_image(test_image, target_size=224)
        extractor = AttentionExtractor(pretrained=True)
        try:
            phi_live = extract_patch_gradient_scores(image_bgr, extractor)
            assert phi_live.shape == (_NUM_PATCHES,)
            S_live = get_shap_zone_scores(image_bgr, M, extractor)
            assert S_live.shape == (9,)
            print(f"  phi range: [{phi_live.min():.4f}, {phi_live.max():.4f}]")
            print(f"  S zones:   {np.round(S_live, 4)}")
        finally:
            extractor.close()
        print("  OK")
    else:
        print(f"[_self_test] skipping integration (no file at {test_image})")

    print("All self-tests passed.")


if __name__ == "__main__":
    import sys

    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()

