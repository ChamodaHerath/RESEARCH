"""
feature_extractor.py — measurable visual features per semantic facial zone.

RULE-BASED / SIGNAL COMPONENT
-----------------------------
All features are classical, interpretable image statistics (LBP, gradients,
color spaces, FFT energy, JPEG-like blockiness, landmark geometry).
They do **not** prove the physical cause of manipulation; they quantify
observable local evidence that later feeds BayesianArtifactInferencer.

Expected inputs (aligned with SAEDF pipeline):
  - image_rgb: (224, 224, 3) uint8/float RGB face crop
  - zone_mask: (224, 224) bool/uint8 binary mask for one ZONE_NAMES region
  - neighboring_mask: optional complementary skin mask for difference features
  - landmarks: (68, 2) dlib points in 224-space for geometry features
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

try:
    from semantic_patch_mapper import ZONE_LANDMARK_INDICES, ZONE_NAMES
except ImportError:  # pragma: no cover
    ZONE_NAMES = [
        "left_eye", "right_eye", "nose_bridge", "nose_tip",
        "left_cheek", "right_cheek", "mouth", "forehead", "chin_jaw",
    ]
    ZONE_LANDMARK_INDICES = {}

IMAGE_SIZE = 224
_EPS = 1e-8


def _as_bool_mask(mask: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    m = np.asarray(mask)
    if m.shape != shape_hw:
        raise ValueError(f"zone_mask shape {m.shape} != image shape {shape_hw}")
    return m.astype(bool)


def _safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else 0.0


def _safe_std(x: np.ndarray) -> float:
    return float(np.std(x)) if x.size else 0.0


def _entropy_from_hist(hist: np.ndarray) -> float:
    p = hist.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p + _EPS)))


def _masked_pixels(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[mask]
    return arr[mask]


def _neighbor_diff(zone_val: float, neighbor_val: float) -> float:
    return float(abs(zone_val - neighbor_val))


def _dilate_mask(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _erode_mask(mask: np.ndarray, ksize: int = 3) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _boundary_band(mask: np.ndarray, width: int = 2) -> np.ndarray:
    dil = _dilate_mask(mask, ksize=2 * width + 1)
    ero = _erode_mask(mask, ksize=2 * width + 1)
    return dil & (~ero)


def local_binary_pattern_u2(
    gray: np.ndarray, radius: int = 1, n_points: int = 8
) -> np.ndarray:
    """
    Uniform LBP (P,R) without scikit-image. Codes in [0, P+1].

    Uses cv2.remap with border reflection so every neighbor sample is
    exactly HxW (avoids brittle slice arithmetic at image edges).
    """
    gray_f = gray.astype(np.float32)
    h, w = gray_f.shape
    codes = np.zeros((h, w), dtype=np.uint8)
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )
    for i, angle in enumerate(2 * np.pi * np.arange(n_points) / n_points):
        map_x = xx + radius * np.cos(angle)
        map_y = yy + radius * np.sin(angle)
        neigh = cv2.remap(
            gray_f,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        codes |= ((neigh >= gray_f).astype(np.uint8) << i)

    def _bit_transitions(v: int) -> int:
        b = ((v << 1) | (v >> (n_points - 1))) & ((1 << n_points) - 1)
        return bin(v ^ b).count("1")

    lut = np.zeros(256, dtype=np.uint8)
    uid = 0
    for v in range(1 << n_points):
        if _bit_transitions(v) <= 2:
            lut[v] = uid
            uid += 1
        else:
            lut[v] = n_points + 1
    return lut[codes]


def _lbp_hist_stats(
    gray: np.ndarray, mask: np.ndarray
) -> Tuple[float, float, float]:
    if mask.sum() < 4:
        return 0.0, 0.0, 0.0
    lbp = local_binary_pattern_u2(gray)
    vals = lbp[mask]
    hist, _ = np.histogram(vals, bins=np.arange(0, 12))
    return (
        _entropy_from_hist(hist),
        _safe_mean(vals.astype(np.float64)),
        _safe_std(vals.astype(np.float64)),
    )


def _gradient_mag(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _local_contrast(gray: np.ndarray, mask: np.ndarray, ksize: int = 7) -> float:
    if not mask.any():
        return 0.0
    blur = cv2.blur(gray.astype(np.float32), (ksize, ksize))
    contrast = np.abs(gray.astype(np.float32) - blur)
    return _safe_mean(contrast[mask])


def _gray_entropy(gray: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    hist, _ = np.histogram(gray[mask], bins=32, range=(0, 256))
    return _entropy_from_hist(hist)


def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _masked_crop(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bb = _bbox_from_mask(mask)
    if bb is None:
        return np.zeros((8, 8), dtype=np.float32)
    x0, y0, x1, y1 = bb
    crop = gray[y0:y1, x0:x1].astype(np.float32).copy()
    m = mask[y0:y1, x0:x1]
    crop[~m] = float(crop[m].mean()) if m.any() else 0.0
    return crop


def _fft_band_energies(gray_patch: np.ndarray) -> Dict[str, float]:
    if gray_patch.size < 16:
        return {
            "low_frequency_energy": 0.0,
            "mid_frequency_energy": 0.0,
            "high_frequency_energy": 0.0,
            "high_low_frequency_ratio": 0.0,
            "spectral_entropy": 0.0,
            "radial_frequency_mean": 0.0,
        }
    g = gray_patch.astype(np.float32)
    g = g - g.mean()
    h, w = g.shape
    n = int(2 ** np.ceil(np.log2(max(h, w, 16))))
    canvas = np.zeros((n, n), dtype=np.float32)
    canvas[:h, :w] = g
    F = np.fft.fftshift(np.fft.fft2(canvas))
    mag = np.abs(F) ** 2
    cy = cx = n // 2
    yy, xx = np.ogrid[:n, :n]
    r_norm = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_norm = r_norm / (r_norm.max() + _EPS)
    low = float(mag[r_norm <= 0.15].sum())
    mid = float(mag[(r_norm > 0.15) & (r_norm <= 0.45)].sum())
    high = float(mag[r_norm > 0.45].sum())
    total = low + mid + high + _EPS
    hist, _ = np.histogram(mag.ravel(), bins=32)
    return {
        "low_frequency_energy": low / total,
        "mid_frequency_energy": mid / total,
        "high_frequency_energy": high / total,
        "high_low_frequency_ratio": float(high / (low + _EPS)),
        "spectral_entropy": _entropy_from_hist(hist),
        "radial_frequency_mean": float((mag * r_norm).sum() / (mag.sum() + _EPS)),
    }


def _blocking_score(gray: np.ndarray, mask: np.ndarray, block: int = 8) -> Dict[str, float]:
    if mask.sum() < block * block:
        return {
            "blocking_score": 0.0,
            "block_boundary_strength": 0.0,
            "block_hf_discontinuity": 0.0,
        }
    g = gray.astype(np.float32)
    h, w = g.shape
    boundary: list = []
    interior: list = []
    for x in range(block, w, block):
        col_mask = mask[:, x] & mask[:, x - 1]
        if col_mask.any():
            boundary.extend(np.abs(g[col_mask, x] - g[col_mask, x - 1]).tolist())
        mid = x - block // 2
        if 1 <= mid < w:
            col_m = mask[:, mid] & mask[:, mid - 1]
            if col_m.any():
                interior.extend(np.abs(g[col_m, mid] - g[col_m, mid - 1]).tolist())
    for y in range(block, h, block):
        row_mask = mask[y, :] & mask[y - 1, :]
        if row_mask.any():
            boundary.extend(np.abs(g[y, row_mask] - g[y - 1, row_mask]).tolist())
        mid = y - block // 2
        if 1 <= mid < h:
            row_m = mask[mid, :] & mask[mid - 1, :]
            if row_m.any():
                interior.extend(np.abs(g[mid, row_m] - g[mid - 1, row_m]).tolist())
    b_mean = float(np.mean(boundary)) if boundary else 0.0
    i_mean = float(np.mean(interior)) if interior else 0.0
    lap = cv2.Laplacian(g, cv2.CV_32F)
    border = np.zeros_like(mask)
    border[:, np.arange(block, w, block)] = True
    border[np.arange(block, h, block), :] = True
    border &= mask
    hf = float(np.mean(np.abs(lap[border]))) if border.any() else 0.0
    return {
        "blocking_score": float(b_mean / (i_mean + _EPS)),
        "block_boundary_strength": b_mean,
        "block_hf_discontinuity": hf,
    }


def _delta_e_lab(lab_a: np.ndarray, lab_b: np.ndarray) -> float:
    if lab_a.size == 0 or lab_b.size == 0:
        return 0.0
    ma = lab_a.reshape(-1, 3).mean(axis=0)
    mb = lab_b.reshape(-1, 3).mean(axis=0)
    return float(np.linalg.norm(ma.astype(np.float64) - mb.astype(np.float64)))


def extract_geometry_features(
    landmarks: np.ndarray,
    zone_name: str,
) -> Dict[str, float]:
    """
    Scale-invariant geometric features from 68-point landmarks.

    Coordinates are normalized by face width/height so features are
    comparable across crops.
    """
    lm = np.asarray(landmarks, dtype=np.float64)
    if lm.shape != (68, 2):
        raise ValueError(f"Expected landmarks shape (68, 2), got {lm.shape}")
    if zone_name not in ZONE_NAMES and zone_name not in ZONE_LANDMARK_INDICES:
        raise KeyError(f"Unknown zone_name '{zone_name}'")

    face_w = float(np.linalg.norm(lm[16] - lm[0]) + _EPS)
    face_h = float(abs(lm[8, 1] - lm[27, 1]) + _EPS)
    lm_n = lm.copy()
    lm_n[:, 0] = (lm[:, 0] - lm[:, 0].mean()) / face_w
    lm_n[:, 1] = (lm[:, 1] - lm[:, 1].mean()) / face_h

    def _dist(i: int, j: int) -> float:
        return float(np.linalg.norm(lm_n[i] - lm_n[j]))

    pairs = [
        (0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9),
        (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),
        (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),
        (31, 35), (32, 34),
        (48, 54), (49, 53), (50, 52), (60, 64), (61, 63),
    ]
    sym_errs = []
    for li, ri in pairs:
        left_reflected = np.array([-lm_n[li, 0], lm_n[li, 1]])
        sym_errs.append(float(np.linalg.norm(left_reflected - lm_n[ri])))
    landmark_symmetry_error = float(np.mean(sym_errs)) if sym_errs else 0.0

    inter_ocular = _dist(36, 45)
    mouth_w = _dist(48, 54)
    nose_len = _dist(27, 33)
    shape_ratio_deviation = float(
        abs(inter_ocular - 0.35) + abs(mouth_w - 0.30) + abs(nose_len - 0.35)
    ) / 3.0

    eye_width_asymmetry = float(abs(_dist(36, 39) - _dist(42, 45)))
    mouth_corner_asymmetry = float(abs(lm_n[48, 1] - lm_n[54, 1]))
    nose_alignment = float(np.mean(np.abs(lm_n[27:31, 0])))

    jaw_n = lm_n[0:17]
    if len(jaw_n) >= 3:
        d2 = jaw_n[2:] - 2 * jaw_n[1:-1] + jaw_n[:-2]
        jaw_contour_irregularity = float(np.mean(np.linalg.norm(d2, axis=1)))
    else:
        jaw_contour_irregularity = 0.0

    contour_irregularity = 0.0
    idxs = ZONE_LANDMARK_INDICES.get(zone_name)
    if idxs is not None and len(idxs) >= 3:
        zpts = lm_n[np.asarray(idxs)]
        hull = cv2.convexHull(zpts.astype(np.float32))
        if hull is not None and len(hull) >= 3:
            area = float(cv2.contourArea(hull))
            peri = float(cv2.arcLength(hull, True))
            if area > _EPS:
                circ = (peri ** 2) / (4.0 * np.pi * area + _EPS)
                contour_irregularity = float(abs(circ - 1.0))

    return {
        "landmark_symmetry_error": landmark_symmetry_error,
        "shape_ratio_deviation": shape_ratio_deviation,
        "contour_irregularity": contour_irregularity,
        "eye_width_asymmetry": eye_width_asymmetry,
        "mouth_corner_asymmetry": mouth_corner_asymmetry,
        "nose_alignment": nose_alignment,
        "jaw_contour_irregularity": jaw_contour_irregularity,
        "inter_ocular_ratio": inter_ocular,
        "mouth_width_ratio": mouth_w,
    }


class ZoneFeatureExtractor:
    """
    Extract interpretable texture / boundary / color / lighting / frequency /
    compression features from one semantic facial zone on a 224×224 RGB crop.
    """

    def __init__(
        self,
        min_zone_pixels: int = 64,
        canny_lo: int = 50,
        canny_hi: int = 150,
    ):
        self.min_zone_pixels = int(min_zone_pixels)
        self.canny_lo = int(canny_lo)
        self.canny_hi = int(canny_hi)

    def extract(
        self,
        image_rgb: np.ndarray,
        zone_mask: np.ndarray,
        neighboring_mask: np.ndarray | None = None,
    ) -> Dict[str, float]:
        """
        Compute measurable zone features.

        Returns a dict of float feature names → raw (unnormalized) values.
        """
        img = np.asarray(image_rgb)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected RGB image HxWx3, got shape {img.shape}")
        h, w = img.shape[:2]
        mask = _as_bool_mask(zone_mask, (h, w))
        n_pix = int(mask.sum())
        if n_pix == 0:
            return self._zero_features(n_pix=0)

        rgb = img.astype(np.float32)
        if rgb.max() <= 1.0 + 1e-6:
            rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        else:
            rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)

        gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
        lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
        rgb_f = rgb_u8.astype(np.float32)

        if neighboring_mask is None:
            neigh = _dilate_mask(mask, ksize=15) & (~mask)
        else:
            neigh = _as_bool_mask(neighboring_mask, (h, w)) & (~mask)

        feats: Dict[str, float] = {"zone_pixel_count": float(n_pix)}
        feats.update(self._texture_features(gray, mask, neigh))
        feats.update(self._boundary_features(gray, rgb_f, mask))
        feats.update(self._color_features(rgb_f, hsv, lab, mask, neigh))
        feats.update(self._lighting_features(gray, mask, neigh))
        feats.update(self._frequency_features(gray, mask, neigh))
        feats.update(self._compression_features(gray, mask, neigh))
        return feats

    def extract_with_geometry(
        self,
        image_rgb: np.ndarray,
        zone_mask: np.ndarray,
        landmarks: np.ndarray,
        zone_name: str,
        neighboring_mask: np.ndarray | None = None,
    ) -> Dict[str, float]:
        """Pixel features plus scale-invariant geometry features."""
        feats = self.extract(image_rgb, zone_mask, neighboring_mask)
        feats.update(extract_geometry_features(landmarks, zone_name))
        return feats

    def _texture_features(
        self, gray: np.ndarray, mask: np.ndarray, neigh: np.ndarray
    ) -> Dict[str, float]:
        lbp_ent, lbp_mean, lbp_std = _lbp_hist_stats(gray, mask)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        lap_var = float(np.var(lap[mask])) if mask.any() else 0.0
        gmag = _gradient_mag(gray)
        g_mean = _safe_mean(gmag[mask])
        g_std = _safe_std(gmag[mask])
        contrast = _local_contrast(gray, mask)
        gray_ent = _gray_entropy(gray, mask)
        if neigh.any():
            n_lbp_ent, _, _ = _lbp_hist_stats(gray, neigh)
            n_lap_var = float(np.var(lap[neigh]))
            n_contrast = _local_contrast(gray, neigh)
            n_g_mean = _safe_mean(gmag[neigh])
        else:
            n_lbp_ent = n_lap_var = n_contrast = n_g_mean = 0.0
        return {
            "lbp_entropy": lbp_ent,
            "lbp_mean": lbp_mean,
            "lbp_std": lbp_std,
            "lbp_entropy_difference": _neighbor_diff(lbp_ent, n_lbp_ent),
            "laplacian_variance": lap_var,
            "laplacian_variance_difference": _neighbor_diff(lap_var, n_lap_var),
            "gradient_magnitude_mean": g_mean,
            "gradient_magnitude_std": g_std,
            "gradient_magnitude_difference": _neighbor_diff(g_mean, n_g_mean),
            "local_contrast": contrast,
            "local_contrast_difference": _neighbor_diff(contrast, n_contrast),
            "gray_level_entropy": gray_ent,
        }

    def _boundary_features(
        self, gray: np.ndarray, rgb: np.ndarray, mask: np.ndarray
    ) -> Dict[str, float]:
        band = _boundary_band(mask, width=2)
        inner = _erode_mask(mask, ksize=5)
        if not inner.any():
            inner = mask
        outer = _dilate_mask(mask, ksize=7) & (~mask)
        gmag = _gradient_mag(gray)
        boundary_grad = _safe_mean(gmag[band]) if band.any() else 0.0
        inner_grad = _safe_mean(gmag[inner]) if inner.any() else 0.0
        edges = cv2.Canny(gray, self.canny_lo, self.canny_hi)
        edge_density_boundary = (
            float(edges[band].mean() / 255.0) if band.any() else 0.0
        )
        canny_edge_density = (
            float(edges[mask].mean() / 255.0) if mask.any() else 0.0
        )
        if outer.any() and inner.any():
            inside_mean = rgb[inner].reshape(-1, 3).mean(axis=0)
            outside_mean = rgb[outer].reshape(-1, 3).mean(axis=0)
            inside_outside = float(np.linalg.norm(inside_mean - outside_mean))
        else:
            inside_outside = 0.0
        return {
            "edge_density_boundary": edge_density_boundary,
            "boundary_gradient_mean": boundary_grad,
            "boundary_discontinuity": float(abs(boundary_grad - inner_grad)),
            "inside_outside_color_difference": inside_outside,
            "canny_edge_density": canny_edge_density,
            "boundary_sharpness_ratio": float(boundary_grad / (inner_grad + _EPS)),
        }

    def _color_features(
        self,
        rgb: np.ndarray,
        hsv: np.ndarray,
        lab: np.ndarray,
        mask: np.ndarray,
        neigh: np.ndarray,
    ) -> Dict[str, float]:
        z_rgb = _masked_pixels(rgb, mask)
        z_hsv = _masked_pixels(hsv, mask)
        z_lab = _masked_pixels(lab, mask)
        feats = {
            "rgb_r_mean": _safe_mean(z_rgb[:, 0]) if z_rgb.size else 0.0,
            "rgb_g_mean": _safe_mean(z_rgb[:, 1]) if z_rgb.size else 0.0,
            "rgb_b_mean": _safe_mean(z_rgb[:, 2]) if z_rgb.size else 0.0,
            "rgb_r_std": _safe_std(z_rgb[:, 0]) if z_rgb.size else 0.0,
            "rgb_g_std": _safe_std(z_rgb[:, 1]) if z_rgb.size else 0.0,
            "rgb_b_std": _safe_std(z_rgb[:, 2]) if z_rgb.size else 0.0,
            "hsv_h_mean": _safe_mean(z_hsv[:, 0]) if z_hsv.size else 0.0,
            "hsv_s_mean": _safe_mean(z_hsv[:, 1]) if z_hsv.size else 0.0,
            "hsv_v_mean": _safe_mean(z_hsv[:, 2]) if z_hsv.size else 0.0,
            "hsv_h_std": _safe_std(z_hsv[:, 0]) if z_hsv.size else 0.0,
            "hsv_s_std": _safe_std(z_hsv[:, 1]) if z_hsv.size else 0.0,
            "hsv_v_std": _safe_std(z_hsv[:, 2]) if z_hsv.size else 0.0,
            "lab_l_mean": _safe_mean(z_lab[:, 0]) if z_lab.size else 0.0,
            "lab_a_mean": _safe_mean(z_lab[:, 1]) if z_lab.size else 0.0,
            "lab_b_mean": _safe_mean(z_lab[:, 2]) if z_lab.size else 0.0,
            "lab_l_std": _safe_std(z_lab[:, 0]) if z_lab.size else 0.0,
            "lab_a_std": _safe_std(z_lab[:, 1]) if z_lab.size else 0.0,
            "lab_b_std": _safe_std(z_lab[:, 2]) if z_lab.size else 0.0,
        }
        if neigh.any():
            n_rgb = _masked_pixels(rgb, neigh)
            n_hsv = _masked_pixels(hsv, neigh)
            n_lab = _masked_pixels(lab, neigh)
            feats["rgb_mean_difference"] = float(
                np.linalg.norm(z_rgb.mean(axis=0) - n_rgb.mean(axis=0))
            )
            feats["hue_difference"] = float(
                abs(_safe_mean(z_hsv[:, 0]) - _safe_mean(n_hsv[:, 0]))
            )
            feats["saturation_difference"] = float(
                abs(_safe_mean(z_hsv[:, 1]) - _safe_mean(n_hsv[:, 1]))
            )
            feats["delta_e_neighbor"] = _delta_e_lab(z_lab, n_lab)
            feats["lab_a_mean_difference"] = float(
                abs(_safe_mean(z_lab[:, 1]) - _safe_mean(n_lab[:, 1]))
            )
            feats["lab_b_mean_difference"] = float(
                abs(_safe_mean(z_lab[:, 2]) - _safe_mean(n_lab[:, 2]))
            )
        else:
            feats.update(
                {
                    "rgb_mean_difference": 0.0,
                    "hue_difference": 0.0,
                    "saturation_difference": 0.0,
                    "delta_e_neighbor": 0.0,
                    "lab_a_mean_difference": 0.0,
                    "lab_b_mean_difference": 0.0,
                }
            )
        return feats

    def _lighting_features(
        self, gray: np.ndarray, mask: np.ndarray, neigh: np.ndarray
    ) -> Dict[str, float]:
        lum = gray.astype(np.float32)
        z = lum[mask]
        mean_l = _safe_mean(z)
        var_l = float(np.var(z)) if z.size else 0.0
        bb = _bbox_from_mask(mask)
        if bb is not None:
            x0, y0, x1, y1 = bb
            mid = (x0 + x1) // 2
            left = mask.copy()
            right = mask.copy()
            left[:, mid:] = False
            right[:, :mid] = False
            lr_imbalance = abs(_safe_mean(lum[left]) - _safe_mean(lum[right]))
        else:
            lr_imbalance = 0.0
        n_mean = _safe_mean(lum[neigh]) if neigh.any() else mean_l
        gmag = _gradient_mag(lum)
        highlight_ratio = float((z > 220).mean()) if z.size else 0.0
        shadow_ratio = float((z < 40).mean()) if z.size else 0.0
        if neigh.any():
            nz = lum[neigh]
            hi_diff = abs(highlight_ratio - float((nz > 220).mean()))
        else:
            hi_diff = 0.0
        return {
            "mean_luminance": mean_l,
            "luminance_variance": var_l,
            "left_right_luminance_imbalance": float(lr_imbalance),
            "luminance_neighbor_difference": float(abs(mean_l - n_mean)),
            "illumination_gradient": float(_safe_mean(gmag[mask])),
            "highlight_ratio": highlight_ratio,
            "shadow_ratio": shadow_ratio,
            "highlight_ratio_difference": float(hi_diff),
        }

    def _frequency_features(
        self, gray: np.ndarray, mask: np.ndarray, neigh: np.ndarray
    ) -> Dict[str, float]:
        z_bands = _fft_band_energies(_masked_crop(gray, mask))
        feats = dict(z_bands)
        if neigh.any():
            n_bands = _fft_band_energies(_masked_crop(gray, neigh))
            feats["high_low_frequency_ratio_difference"] = abs(
                z_bands["high_low_frequency_ratio"]
                - n_bands["high_low_frequency_ratio"]
            )
            feats["spectral_entropy_difference"] = abs(
                z_bands["spectral_entropy"] - n_bands["spectral_entropy"]
            )
            feats["radial_spectrum_difference"] = abs(
                z_bands["radial_frequency_mean"] - n_bands["radial_frequency_mean"]
            )
        else:
            feats["high_low_frequency_ratio_difference"] = 0.0
            feats["spectral_entropy_difference"] = 0.0
            feats["radial_spectrum_difference"] = 0.0
        return feats

    def _compression_features(
        self, gray: np.ndarray, mask: np.ndarray, neigh: np.ndarray
    ) -> Dict[str, float]:
        z = _blocking_score(gray, mask)
        feats = dict(z)
        if neigh.any():
            n = _blocking_score(gray, neigh)
            feats["block_boundary_difference"] = abs(
                z["block_boundary_strength"] - n["block_boundary_strength"]
            )
        else:
            feats["block_boundary_difference"] = 0.0
        return feats

    def _zero_features(self, n_pix: int = 0) -> Dict[str, float]:
        keys = [
            "zone_pixel_count",
            "lbp_entropy", "lbp_mean", "lbp_std", "lbp_entropy_difference",
            "laplacian_variance", "laplacian_variance_difference",
            "gradient_magnitude_mean", "gradient_magnitude_std",
            "gradient_magnitude_difference",
            "local_contrast", "local_contrast_difference", "gray_level_entropy",
            "edge_density_boundary", "boundary_gradient_mean",
            "boundary_discontinuity", "inside_outside_color_difference",
            "canny_edge_density", "boundary_sharpness_ratio",
            "rgb_r_mean", "rgb_g_mean", "rgb_b_mean",
            "rgb_r_std", "rgb_g_std", "rgb_b_std",
            "hsv_h_mean", "hsv_s_mean", "hsv_v_mean",
            "hsv_h_std", "hsv_s_std", "hsv_v_std",
            "lab_l_mean", "lab_a_mean", "lab_b_mean",
            "lab_l_std", "lab_a_std", "lab_b_std",
            "rgb_mean_difference", "hue_difference", "saturation_difference",
            "delta_e_neighbor", "lab_a_mean_difference", "lab_b_mean_difference",
            "mean_luminance", "luminance_variance",
            "left_right_luminance_imbalance", "luminance_neighbor_difference",
            "illumination_gradient", "highlight_ratio", "shadow_ratio",
            "highlight_ratio_difference",
            "low_frequency_energy", "mid_frequency_energy", "high_frequency_energy",
            "high_low_frequency_ratio", "spectral_entropy", "radial_frequency_mean",
            "high_low_frequency_ratio_difference", "spectral_entropy_difference",
            "radial_spectrum_difference",
            "blocking_score", "block_boundary_strength", "block_hf_discontinuity",
            "block_boundary_difference",
        ]
        out = {k: 0.0 for k in keys}
        out["zone_pixel_count"] = float(n_pix)
        return out
