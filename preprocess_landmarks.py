
# preprocess_landmarks.py

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import dlib
import numpy as np
from mtcnn import MTCNN


# ---------------------------------------------------------------------------
# 9-zone semantic landmark groups (expanded from original 5)
# ---------------------------------------------------------------------------
SEMANTIC_GROUPS: Dict[str, List[int]] = {
    "left_eye":    list(range(36, 42)),          # 6 points
    "right_eye":   list(range(42, 48)),          # 6 points
    "nose_bridge": [27, 28, 29, 30],             # 4 points
    "nose_tip":    [31, 32, 33, 34, 35],         # 5 points
    "left_cheek":  [1, 2, 3, 4, 5, 48, 49],     # 7 points
    "right_cheek": [11, 12, 13, 14, 15, 54, 55],# 7 points
    "mouth":       list(range(48, 68)),          # 20 points
    "forehead":    list(range(17, 27)),          # eyebrow proxy, 10 points
    "chin_jaw":    [0, 1, 2, 14, 15, 16, 7, 8, 9],  # 9 points
}

ZONE_NAMES: List[str] = list(SEMANTIC_GROUPS.keys())
NUM_ZONES: int = len(ZONE_NAMES)

# ViT-Base/16 constants
IMAGE_SIZE: int = 224
PATCH_SIZE: int = 16
GRID_SIZE: int = IMAGE_SIZE // PATCH_SIZE   # 14
NUM_PATCHES: int = GRID_SIZE * GRID_SIZE    # 196

# Debug colours per zone (BGR)
ZONE_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "left_eye":    (255, 100, 100),
    "right_eye":   (100, 100, 255),
    "nose_bridge": (100, 255, 100),
    "nose_tip":    (100, 200, 200),
    "left_cheek":  (200, 100, 200),
    "right_cheek": (200, 200, 100),
    "mouth":       (50, 50, 255),
    "forehead":    (150, 150, 50),
    "chin_jaw":    (50, 150, 150),
}


# ---------------------------------------------------------------------------
# Zone-patch membership matrix  M ∈ ℝ^(196 × 9)
# ---------------------------------------------------------------------------

def compute_zone_patch_matrix(
    landmarks_2d: List[Tuple[float, float]],
    image_size: int = IMAGE_SIZE,
    patch_size: int = PATCH_SIZE,
) -> np.ndarray:
    """
    Compute the zone-patch membership matrix M.

    For each of the 196 ViT patches and each of the 9 semantic zones,
    M[i, z] = (area of patch i that lies inside zone z) / (patch area).

    This is a continuous overlap fraction in [0, 1], not a binary flag.
    A patch can partially overlap multiple zones; values per row need not sum to 1.

    Args:
        landmarks_2d: 68 (x, y) landmark coordinates scaled to [0, image_size).
        image_size:   Target face image size (default 224).
        patch_size:   ViT patch size (default 16).

    Returns:
        M: np.ndarray of shape (196, 9), dtype float32.
    """
    grid = image_size // patch_size
    pts = np.array(landmarks_2d, dtype=np.float32)   # (68, 2)
    M = np.zeros((grid * grid, NUM_ZONES), dtype=np.float32)

    for z_idx, (zone_name, lm_indices) in enumerate(SEMANTIC_GROUPS.items()):
        zone_pts = pts[lm_indices]

        # Need at least 3 points for a convex hull
        if len(zone_pts) < 3:
            continue

        hull = cv2.convexHull(zone_pts)
        if hull is None or len(hull) < 3:
            continue

        # Rasterise zone polygon onto a binary mask
        zone_mask = np.zeros((image_size, image_size), dtype=np.uint8)
        cv2.fillConvexPoly(zone_mask, hull.astype(np.int32), 1)

        # For efficiency, iterate only over patches that could overlap
        # by checking the zone bounding box first
        bx1 = max(0, int(zone_pts[:, 0].min()) // patch_size)
        bx2 = min(grid - 1, int(zone_pts[:, 0].max()) // patch_size)
        by1 = max(0, int(zone_pts[:, 1].min()) // patch_size)
        by2 = min(grid - 1, int(zone_pts[:, 1].max()) // patch_size)

        patch_area = float(patch_size * patch_size)

        for row in range(by1, by2 + 1):
            for col in range(bx1, bx2 + 1):
                py1 = row * patch_size
                py2 = py1 + patch_size
                px1 = col * patch_size
                px2 = px1 + patch_size
                overlap = int(zone_mask[py1:py2, px1:px2].sum())
                if overlap > 0:
                    patch_idx = row * grid + col
                    M[patch_idx, z_idx] = overlap / patch_area

    return M


def visualise_zone_matrix(
    image_bgr: np.ndarray,
    landmarks_2d: List[Tuple[float, float]],
    M: np.ndarray,
    save_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Overlay zone convex hulls and patch opacity (scaled by max zone score) for debug.

    Args:
        image_bgr:     224x224 BGR face image.
        landmarks_2d:  68 scaled landmark points.
        M:             Zone-patch matrix (196, 9).
        save_path:     Optional path to save the visualisation.

    Returns:
        Annotated BGR image.
    """
    canvas = image_bgr.copy()
    pts = np.array(landmarks_2d, dtype=np.float32)

    # Draw zone convex hulls with translucent fill
    overlay = canvas.copy()
    for z_idx, (zone_name, lm_indices) in enumerate(SEMANTIC_GROUPS.items()):
        zone_pts = pts[lm_indices]
        if len(zone_pts) < 3:
            continue
        hull = cv2.convexHull(zone_pts)
        if hull is None:
            continue
        colour = ZONE_COLOURS[zone_name]
        cv2.fillConvexPoly(overlay, hull.astype(np.int32), colour)
        cv2.polylines(canvas, [hull.astype(np.int32)], True, colour, 1)

    cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

    # Draw patch grid with opacity proportional to max zone score
    max_scores = M.max(axis=1)   # (196,)
    global_max = max_scores.max() + 1e-8
    grid = IMAGE_SIZE // PATCH_SIZE
    for idx in range(NUM_PATCHES):
        row = idx // grid
        col = idx % grid
        alpha = float(max_scores[idx]) / global_max
        if alpha > 0.05:
            x1, y1 = col * PATCH_SIZE, row * PATCH_SIZE
            cv2.rectangle(
                canvas,
                (x1, y1),
                (x1 + PATCH_SIZE - 1, y1 + PATCH_SIZE - 1),
                (255, 255, 0),
                1 if alpha > 0.3 else -1 if alpha > 0.6 else 1,
            )

    # Draw landmark points
    for x, y in landmarks_2d:
        cv2.circle(canvas, (int(round(x)), int(round(y))), 1, (0, 255, 0), -1)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), canvas)

    return canvas


# ---------------------------------------------------------------------------
# Original pipeline — preserved exactly, augmented with M matrix output
# ---------------------------------------------------------------------------

@dataclass
class FaceCrop:
    image: np.ndarray
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def setup_logger(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def detect_face_with_mtcnn(
    detector: MTCNN,
    image_bgr: np.ndarray,
    margin_ratio: float = 0.1,
) -> Optional[FaceCrop]:
    """Detect largest face and return cropped region with margin."""
    if image_bgr is None or image_bgr.size == 0:
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(image_rgb)
    if not detections:
        return None

    best = max(detections, key=lambda d: d["box"][2] * d["box"][3])
    x, y, w, h = best["box"]

    img_h, img_w = image_bgr.shape[:2]
    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(img_w, x + w + mx)
    y2 = min(img_h, y + h + my)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image_bgr[y1:y2, x1:x2]
    return None if crop.size == 0 else FaceCrop(image=crop, x1=x1, y1=y1, x2=x2, y2=y2)


def extract_68_landmarks_dlib(
    predictor: dlib.shape_predictor,
    face_crop_bgr: np.ndarray,
) -> Optional[List[Tuple[float, float]]]:
    """Extract exactly 68 facial landmarks from a cropped face using dlib."""
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        return None

    gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    rect = dlib.rectangle(0, 0, w - 1, h - 1)
    shape = predictor(gray, rect)

    if shape.num_parts != 68:
        return None

    return [(float(shape.part(i).x), float(shape.part(i).y)) for i in range(68)]


def scale_landmarks_to_target(
    landmarks: List[Tuple[float, float]],
    crop_width: int,
    crop_height: int,
    target_size: int = IMAGE_SIZE,
) -> List[Tuple[float, float]]:
    """Scale landmark coordinates from crop space to target_size × target_size."""
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError("Invalid crop dimensions for landmark scaling.")
    sx = target_size / float(crop_width)
    sy = target_size / float(crop_height)
    return [(x * sx, y * sy) for x, y in landmarks]


def landmarks_to_semantic_dict(
    points: List[Tuple[float, float]],
) -> Dict[str, List[List[float]]]:
    """Build 9-zone semantic grouping dict from 68 scaled points."""
    return {
        name: [[points[i][0], points[i][1]] for i in indices]
        for name, indices in SEMANTIC_GROUPS.items()
    }


def draw_debug_landmarks(
    resized_bgr: np.ndarray,
    scaled_points: List[Tuple[float, float]],
    output_path: Path,
) -> None:
    """Save image with plotted 68 landmarks and zone index labels."""
    canvas = resized_bgr.copy()
    for idx, (x, y) in enumerate(scaled_points):
        cx, cy = int(round(x)), int(round(y))
        cv2.circle(canvas, (cx, cy), 1, (0, 255, 0), -1)
        if idx % 5 == 0:
            cv2.putText(canvas, str(idx), (cx + 2, cy - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 0), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def sanitize_stem(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
    return safe.strip("_") or "sample"


def _build_frame_entry(
    frame_name: str,
    out_face_path: Path,
    resized: np.ndarray,
    scaled: List[Tuple[float, float]],
    face: FaceCrop,
) -> Dict:
    """
    Build the JSON entry for a single processed frame.

    NEW: zone_patch_matrix M is computed here and stored as a nested list
    so downstream modules can load it directly without recomputing.
    """
    M = compute_zone_patch_matrix(scaled)

    return {
        "frame": frame_name,
        "face_image": str(out_face_path.as_posix()),
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "landmarks_68": [[x, y] for x, y in scaled],
        "semantic_groups": landmarks_to_semantic_dict(scaled),
        # NEW: serialised zone-patch matrix for downstream modules
        "zone_patch_matrix": M.tolist(),   # shape (196, 9) as nested list
        "zone_names": ZONE_NAMES,
    }


def process_single_frame(
    frame_path: Path,
    out_face_dir: Path,
    predictor_path: Path,
    margin_ratio: float,
    target_size: int,
    save_debug: bool,
    debug_dir: Path,
) -> Tuple[str, Optional[Dict]]:
    """Process one frame file and return (filename, json_entry_or_none)."""
    try:
        detector = MTCNN()
        predictor = dlib.shape_predictor(str(predictor_path))
    except Exception as exc:
        logging.error("Failed to initialise MTCNN/dlib: %s", exc)
        return frame_path.name, None

    image = cv2.imread(str(frame_path))
    if image is None:
        logging.warning("Failed to read frame: %s", frame_path)
        return frame_path.name, None

    face = detect_face_with_mtcnn(detector=detector, image_bgr=image, margin_ratio=margin_ratio)
    if face is None:
        logging.info("No face detected: %s", frame_path.name)
        return frame_path.name, None

    landmarks = extract_68_landmarks_dlib(predictor=predictor, face_crop_bgr=face.image)
    if landmarks is None or len(landmarks) != 68:
        logging.info("No valid landmarks: %s", frame_path.name)
        return frame_path.name, None

    resized = cv2.resize(face.image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    scaled = scale_landmarks_to_target(landmarks, face.width, face.height, target_size)

    out_face_dir.mkdir(parents=True, exist_ok=True)
    out_face_path = out_face_dir / frame_path.name
    cv2.imwrite(str(out_face_path), resized)

    if save_debug:
        debug_path = debug_dir / frame_path.name
        draw_debug_landmarks(resized, scaled, debug_path)

    return frame_path.name, _build_frame_entry(frame_path.name, out_face_path, resized, scaled, face)


def _worker(args: Tuple) -> Tuple[str, Optional[Dict]]:
    return process_single_frame(*args)


def collect_frame_paths(frames_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    frames = [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    frames.sort()
    return frames


def collect_video_paths(videos_dir: Path) -> List[Path]:
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    videos = [p for p in videos_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    videos.sort()
    return videos


def extract_frames_from_video(video_path: Path, sample_every: int, max_frames: Optional[int] = None) -> List[Tuple[int, np.ndarray]]:
    if sample_every <= 0:
        raise ValueError("--sample-every must be > 0.")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.warning("Failed to open video: %s", video_path)
        return []
    frames: List[Tuple[int, np.ndarray]] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_every == 0:
            frames.append((idx, frame))
            if max_frames is not None and len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    return frames


def process_single_array_frame(
    frame_bgr: np.ndarray,
    frame_name: str,
    out_face_dir: Path,
    predictor: dlib.shape_predictor,
    detector: MTCNN,
    margin_ratio: float,
    target_size: int,
    save_debug: bool,
    debug_dir: Path,
) -> Optional[Dict]:
    """Process a raw BGR array frame (used in video mode)."""
    face = detect_face_with_mtcnn(detector=detector, image_bgr=frame_bgr, margin_ratio=margin_ratio)
    if face is None:
        return None

    landmarks = extract_68_landmarks_dlib(predictor=predictor, face_crop_bgr=face.image)
    if landmarks is None or len(landmarks) != 68:
        return None

    resized = cv2.resize(face.image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    scaled = scale_landmarks_to_target(landmarks, face.width, face.height, target_size)

    out_face_dir.mkdir(parents=True, exist_ok=True)
    out_face_path = out_face_dir / frame_name
    cv2.imwrite(str(out_face_path), resized)

    if save_debug:
        draw_debug_landmarks(resized, scaled, debug_dir / frame_name)

    return _build_frame_entry(frame_name, out_face_path, resized, scaled, face)


def process_ffpp_videos(
    ffpp_root: Path,
    predictor_path: Path,
    output_json: Path,
    faces_out_dir: Path,
    margin_ratio: float,
    target_size: int,
    sample_every: int,
    save_debug: bool,
    debug_dir: Path,
    resume_from_outputs: bool,
) -> None:
    """Process FaceForensics++ videos (Deepfakes + original) with resume support."""
    deepfakes_dir = ffpp_root / "Deepfakes"
    original_dir = ffpp_root / "original"
    if not deepfakes_dir.exists() or not original_dir.exists():
        raise FileNotFoundError(f"Expected 'Deepfakes' and 'original' under: {ffpp_root}")

    label_dirs = {"fake": deepfakes_dir, "real": original_dir}
    predictor = dlib.shape_predictor(str(predictor_path))
    detector = MTCNN()

    output: Dict[str, Dict] = {}
    skipped_set: set = set()
    video_stats: Dict[str, Dict] = {}

    if output_json.exists():
        try:
            with output_json.open("r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, dict):
                output.update(existing.get("frames", {}))
                video_stats.update(existing.get("videos", {}))
                skipped_set.update(str(x) for x in existing.get("skipped_frames", []))
            logging.info("Resume: loaded %d videos, %d frames from %s",
                         len(video_stats), len(output), output_json)
        except Exception as exc:
            logging.warning("Could not load existing JSON (%s). Starting fresh.", exc)

    for label, root_dir in label_dirs.items():
        videos = collect_video_paths(root_dir)
        logging.info("[%s] Found %d videos in %s", label, len(videos), root_dir)

        for video_path in videos:
            video_key = video_path.relative_to(ffpp_root).as_posix()
            if video_key in video_stats:
                logging.info("Skipping already processed: %s", video_key)
                continue

            frames = extract_frames_from_video(video_path, sample_every=sample_every)
            processed_count = skipped_count = 0
            video_stem = sanitize_stem(video_path.stem)
            out_video_faces = faces_out_dir / label / video_stem
            out_video_debug = debug_dir / label / video_stem

            for frame_idx, frame_bgr in frames:
                frame_name = f"{video_stem}_f{frame_idx:06d}.jpg"
                entry = process_single_array_frame(
                    frame_bgr=frame_bgr, frame_name=frame_name,
                    out_face_dir=out_video_faces, predictor=predictor,
                    detector=detector, margin_ratio=margin_ratio,
                    target_size=target_size, save_debug=save_debug,
                    debug_dir=out_video_debug,
                )
                frame_key = f"{video_key}::f{frame_idx:06d}"
                if entry is None:
                    skipped_set.add(frame_key)
                    skipped_count += 1
                    continue
                entry["video_path"] = video_key
                entry["frame_index"] = frame_idx
                entry["label"] = label
                output[frame_key] = entry
                processed_count += 1

            video_stats[video_key] = {
                "label": label,
                "sampled_frames": len(frames),
                "processed_frames": processed_count,
                "skipped_frames": skipped_count,
            }

    final_json = {
        "meta": {
            "mode": "ffpp",
            "ffpp_root": str(ffpp_root.as_posix()),
            "target_size": target_size,
            "sample_every": sample_every,
            "num_processed": len(output),
            "num_skipped": len(skipped_set),
            "num_videos_processed": len(video_stats),
            "semantic_groups_indices": SEMANTIC_GROUPS,
            "zone_names": ZONE_NAMES,
            "zone_patch_matrix_shape": [NUM_PATCHES, NUM_ZONES],
            "labels": {"fake": "Deepfakes", "real": "original"},
        },
        "videos": video_stats,
        "frames": output,
        "skipped_frames": sorted(skipped_set),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2)

    logging.info("Saved JSON: %s", output_json)
    logging.info("Processed: %d frames | Skipped: %d | Videos: %d",
                 len(output), len(skipped_set), len(video_stats))


def process_celebdf_videos(
    celebdf_root: Path,
    test_list: Path,
    predictor_path: Path,
    output_json: Path,
    faces_out_dir: Path,
    margin_ratio: float,
    target_size: int,
    sample_every: int,
    max_frames: Optional[int],
    save_debug: bool,
    debug_dir: Path,
    resume_from_outputs: bool,
) -> None:
    """Process Celeb-DF videos using List_of_testing_videos.txt."""
    if not test_list.exists():
        raise FileNotFoundError(f"Test list not found: {test_list}")

    with test_list.open("r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
    
    video_paths = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split()
        rel_path = parts[-1]  # last part is the path
        # Normalize path
        rel_path = rel_path.replace("\\", "/")
        full_path = celebdf_root / rel_path
        
        # Determine label based on folder name
        label = "fake" if "Celeb-synthesis" in rel_path else "real"
        
        if full_path.exists():
            video_paths.append((rel_path, full_path, label))
        else:
            logging.warning("Video not found: %s", full_path)

    predictor = dlib.shape_predictor(str(predictor_path))
    detector = MTCNN()

    output: Dict[str, Dict] = {}
    skipped_set: set = set()
    video_stats: Dict[str, Dict] = {}

    if output_json.exists():
        try:
            with output_json.open("r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, dict):
                output.update(existing.get("frames", {}))
                video_stats.update(existing.get("videos", {}))
                skipped_set.update(str(x) for x in existing.get("skipped_frames", []))
            logging.info("Resume: loaded %d videos, %d frames from %s",
                         len(video_stats), len(output), output_json)
        except Exception as exc:
            logging.warning("Could not load existing JSON (%s). Starting fresh.", exc)

    logging.info("Found %d videos in test list", len(video_paths))

    for rel_path, video_path, label in video_paths:
        video_key = rel_path
        if video_key in video_stats:
            logging.info("Skipping already processed: %s", video_key)
            continue

        frames = extract_frames_from_video(video_path, sample_every=sample_every, max_frames=max_frames)
        processed_count = skipped_count = 0
        video_stem = sanitize_stem(video_path.stem)
        
        # Append label directory so fakes and reals are separated
        out_video_faces = faces_out_dir / label / video_stem
        out_video_debug = debug_dir / label / video_stem

        for frame_idx, frame_bgr in frames:
            frame_name = f"{video_stem}_f{frame_idx:06d}.jpg"
            entry = process_single_array_frame(
                frame_bgr=frame_bgr, frame_name=frame_name,
                out_face_dir=out_video_faces, predictor=predictor,
                detector=detector, margin_ratio=margin_ratio,
                target_size=target_size, save_debug=save_debug,
                debug_dir=out_video_debug,
            )
            frame_key = f"{video_key}::f{frame_idx:06d}"
            if entry is None:
                skipped_set.add(frame_key)
                skipped_count += 1
                continue
            entry["video_path"] = video_key
            entry["frame_index"] = frame_idx
            entry["label"] = label
            output[frame_key] = entry
            processed_count += 1

        video_stats[video_key] = {
            "label": label,
            "sampled_frames": len(frames),
            "processed_frames": processed_count,
            "skipped_frames": skipped_count,
        }

    final_json = {
        "meta": {
            "mode": "celebdf",
            "celebdf_root": str(celebdf_root.as_posix()),
            "target_size": target_size,
            "sample_every": sample_every,
            "max_frames": max_frames,
            "num_processed": len(output),
            "num_skipped": len(skipped_set),
            "num_videos_processed": len(video_stats),
            "semantic_groups_indices": SEMANTIC_GROUPS,
            "zone_names": ZONE_NAMES,
            "zone_patch_matrix_shape": [NUM_PATCHES, NUM_ZONES],
            "labels": {"fake": "Celeb-synthesis", "real": "Celeb-real/YouTube-real"},
        },
        "videos": video_stats,
        "frames": output,
        "skipped_frames": sorted(skipped_set),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2)

    logging.info("Saved JSON: %s", output_json)
    logging.info("Processed: %d frames | Skipped: %d | Videos: %d",
                 len(output), len(skipped_set), len(video_stats))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAEDF Stage 1 — 9-zone landmark preprocessing with zone-patch matrix."
    )
    parser.add_argument("--frames-dir", type=Path, default=None,
                        help="Directory of extracted frame images (single-folder mode).")
    parser.add_argument("--ffpp-root", type=Path, default=None,
                        help="FaceForensics++ or Celeb-DF root directory.")
    parser.add_argument("--test-list", type=Path, default=None,
                        help="Path to Celeb-DF List_of_testing_videos.txt. If provided, triggers Celeb-DF mode.")
    parser.add_argument("--num-frames", type=int, default=None,
                        help="Maximum number of frames to process per video.")
    parser.add_argument("--output-json", type=Path, default=Path("outputs/ffpp_landmarks.json"))
    parser.add_argument("--faces-out-dir", type=Path, default=Path("outputs/faces_224"))
    parser.add_argument("--predictor-path", type=Path,
                        default=Path("shape_predictor_68_face_landmarks.dat"))
    parser.add_argument("--margin-ratio", type=float, default=0.1)
    parser.add_argument("--target-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--resume-from-outputs", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--save-debug", action="store_true",
                        help="Save zone visualisation debug images.")
    parser.add_argument("--debug-dir", type=Path, default=Path("outputs/debug_landmarks"))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(args.verbose)

    if args.frames_dir is None and args.ffpp_root is None:
        raise ValueError("Provide one mode: --frames-dir or --ffpp-root.")
    if args.frames_dir is not None and args.ffpp_root is not None:
        raise ValueError("Use only one mode at a time.")
    if not args.predictor_path.exists():
        raise FileNotFoundError(f"Predictor file not found: {args.predictor_path}")
    if args.target_size != IMAGE_SIZE:
        logging.warning("target_size=%d != IMAGE_SIZE=%d. Zone-patch matrix assumes 224.",
                        args.target_size, IMAGE_SIZE)

    if args.ffpp_root is not None:
        if not args.ffpp_root.exists():
            raise FileNotFoundError(f"Root does not exist: {args.ffpp_root}")
            
        if args.test_list is not None:
            # Celeb-DF mode
            process_celebdf_videos(
                celebdf_root=args.ffpp_root,
                test_list=args.test_list,
                predictor_path=args.predictor_path,
                output_json=args.output_json,
                faces_out_dir=args.faces_out_dir,
                margin_ratio=args.margin_ratio,
                target_size=args.target_size,
                sample_every=args.sample_every,
                max_frames=args.num_frames,
                save_debug=args.save_debug,
                debug_dir=args.debug_dir,
                resume_from_outputs=args.resume_from_outputs,
            )
        else:
            # FaceForensics++ mode
            process_ffpp_videos(
                ffpp_root=args.ffpp_root,
                predictor_path=args.predictor_path,
                output_json=args.output_json,
                faces_out_dir=args.faces_out_dir,
                margin_ratio=args.margin_ratio,
                target_size=args.target_size,
                sample_every=args.sample_every,
                save_debug=args.save_debug,
                debug_dir=args.debug_dir,
                resume_from_outputs=args.resume_from_outputs,
            )
        return

    if not args.frames_dir.exists():
        raise FileNotFoundError(f"Frames directory does not exist: {args.frames_dir}")

    frame_paths = collect_frame_paths(args.frames_dir)
    if not frame_paths:
        raise RuntimeError(f"No image frames found in: {args.frames_dir}")

    logging.info("Found %d frames. Workers: %d", len(frame_paths), args.workers)

    work_items = [
        (fp, args.faces_out_dir, args.predictor_path,
         args.margin_ratio, args.target_size, args.save_debug, args.debug_dir)
        for fp in frame_paths
    ]

    output: Dict[str, Dict] = {}
    skipped: List[str] = []

    with mp.Pool(processes=args.workers) as pool:
        for filename, entry in pool.imap_unordered(_worker, work_items):
            if entry is None:
                skipped.append(filename)
            else:
                output[filename] = entry

    final_json = {
        "meta": {
            "mode": "frames",
            "target_size": args.target_size,
            "num_frames": len(frame_paths),
            "num_processed": len(output),
            "num_skipped": len(skipped),
            "semantic_groups_indices": SEMANTIC_GROUPS,
            "zone_names": ZONE_NAMES,
            "zone_patch_matrix_shape": [NUM_PATCHES, NUM_ZONES],
        },
        "frames": output,
        "skipped_frames": sorted(skipped),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2)

    logging.info("Saved JSON: %s", args.output_json)
    logging.info("Processed: %d | Skipped: %d", len(output), len(skipped))


if __name__ == "__main__":
    main()