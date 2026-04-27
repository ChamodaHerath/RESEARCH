#!/usr/bin/env python3
"""
Preprocess facial landmarks for deepfake detection.

Pipeline:
1) Detect face using MTCNN
2) Crop face with margin
3) Extract 68 landmarks with dlib
4) Resize crop to 224x224
5) Scale landmarks to resized space
6) Save JSON keyed by frame filename

Assumptions:
- shape_predictor_68_face_landmarks.dat is in project root by default.
"""

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


SEMANTIC_GROUPS: Dict[str, List[int]] = {
    "Jawline": list(range(0, 17)),
    "Eyebrows": list(range(17, 27)),
    "Nose": list(range(27, 36)),
    "Eyes": list(range(36, 48)),
    "Mouth": list(range(48, 68)),
}


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

    # Choose largest detected face for stability in single-subject videos.
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
    if crop.size == 0:
        return None

    return FaceCrop(image=crop, x1=x1, y1=y1, x2=x2, y2=y2)


def extract_68_landmarks_dlib(
    predictor: dlib.shape_predictor,
    face_crop_bgr: np.ndarray,
) -> Optional[List[Tuple[float, float]]]:
    """Extract exactly 68 facial landmarks from cropped face using dlib."""
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        return None

    gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # On a cropped face, one full-frame rectangle works reliably.
    rect = dlib.rectangle(0, 0, w - 1, h - 1)
    shape = predictor(gray, rect)

    if shape.num_parts != 68:
        return None

    points = [(float(shape.part(i).x), float(shape.part(i).y)) for i in range(68)]
    return points


def scale_landmarks_to_target(
    landmarks: List[Tuple[float, float]],
    crop_width: int,
    crop_height: int,
    target_size: int = 224,
) -> List[Tuple[float, float]]:
    """
    Scale landmarks from crop space to target image space.

    New_X = Old_X * (target_size / Crop Width)
    New_Y = Old_Y * (target_size / Crop Height)
    """
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError("Invalid crop dimensions for landmark scaling.")

    sx = target_size / float(crop_width)
    sy = target_size / float(crop_height)
    return [(x * sx, y * sy) for x, y in landmarks]


def landmarks_to_semantic_dict(
    points: List[Tuple[float, float]],
) -> Dict[str, List[List[float]]]:
    """Build semantic grouping dict from 68 scaled points."""
    grouped: Dict[str, List[List[float]]] = {}
    for group_name, indices in SEMANTIC_GROUPS.items():
        grouped[group_name] = [[points[i][0], points[i][1]] for i in indices]
    return grouped


def draw_debug_landmarks(
    resized_bgr: np.ndarray,
    scaled_points: List[Tuple[float, float]],
    output_path: Path,
) -> None:
    """Save image with plotted 68 landmarks to verify scaling."""
    canvas = resized_bgr.copy()
    for idx, (x, y) in enumerate(scaled_points):
        cx = int(round(x))
        cy = int(round(y))
        cv2.circle(canvas, (cx, cy), 1, (0, 255, 0), -1)
        if idx % 5 == 0:
            cv2.putText(
                canvas,
                str(idx),
                (cx + 2, cy - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.25,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def sanitize_stem(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
    return safe.strip("_") or "sample"


def process_single_frame(
    frame_path: Path,
    out_face_dir: Path,
    predictor_path: Path,
    margin_ratio: float,
    target_size: int,
    save_debug: bool,
    debug_dir: Path,
) -> Tuple[str, Optional[Dict]]:
    """
    Process one frame and return (filename, json_entry_or_none).
    Returns None entry when no face/landmarks detected.
    """
    try:
        detector = MTCNN()
        predictor = dlib.shape_predictor(str(predictor_path))
    except Exception as exc:
        logging.error("Failed to initialize MTCNN/dlib: %s", exc)
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
        logging.info("No valid 68 landmarks: %s", frame_path.name)
        return frame_path.name, None

    resized = cv2.resize(face.image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    scaled = scale_landmarks_to_target(
        landmarks=landmarks,
        crop_width=face.width,
        crop_height=face.height,
        target_size=target_size,
    )

    out_face_dir.mkdir(parents=True, exist_ok=True)
    out_face_path = out_face_dir / frame_path.name
    cv2.imwrite(str(out_face_path), resized)

    if save_debug:
        debug_path = debug_dir / frame_path.name
        draw_debug_landmarks(resized_bgr=resized, scaled_points=scaled, output_path=debug_path)

    entry = {
        "frame": frame_path.name,
        "face_image": str(out_face_path.as_posix()),
        "image_size": [target_size, target_size],
        "landmarks_68": [[x, y] for x, y in scaled],
        "semantic_groups": landmarks_to_semantic_dict(scaled),
    }
    return frame_path.name, entry


def _worker(args: Tuple[Path, Path, Path, float, int, bool, Path]) -> Tuple[str, Optional[Dict]]:
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


def extract_frames_from_video(video_path: Path, sample_every: int) -> List[Tuple[int, np.ndarray]]:
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
    face = detect_face_with_mtcnn(detector=detector, image_bgr=frame_bgr, margin_ratio=margin_ratio)
    if face is None:
        return None

    landmarks = extract_68_landmarks_dlib(predictor=predictor, face_crop_bgr=face.image)
    if landmarks is None or len(landmarks) != 68:
        return None

    resized = cv2.resize(face.image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    scaled = scale_landmarks_to_target(
        landmarks=landmarks,
        crop_width=face.width,
        crop_height=face.height,
        target_size=target_size,
    )

    out_face_dir.mkdir(parents=True, exist_ok=True)
    out_face_path = out_face_dir / frame_name
    cv2.imwrite(str(out_face_path), resized)

    if save_debug:
        debug_path = debug_dir / frame_name
        draw_debug_landmarks(resized_bgr=resized, scaled_points=scaled, output_path=debug_path)

    return {
        "frame": frame_name,
        "face_image": str(out_face_path.as_posix()),
        "image_size": [target_size, target_size],
        "landmarks_68": [[x, y] for x, y in scaled],
        "semantic_groups": landmarks_to_semantic_dict(scaled),
    }


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
    deepfakes_dir = ffpp_root / "Deepfakes"
    original_dir = ffpp_root / "original"
    if not deepfakes_dir.exists() or not original_dir.exists():
        raise FileNotFoundError(
            f"Expected 'Deepfakes' and 'original' under: {ffpp_root}"
        )

    label_dirs = {
        "fake": deepfakes_dir,
        "real": original_dir,
    }

    predictor = dlib.shape_predictor(str(predictor_path))
    detector = MTCNN()

    # Resume support: if JSON already exists, skip videos already present there.
    output: Dict[str, Dict] = {}
    skipped_set: set[str] = set()
    video_stats: Dict[str, Dict[str, int]] = {}
    if output_json.exists():
        try:
            with output_json.open("r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, dict):
                existing_frames = existing.get("frames", {})
                existing_skipped = existing.get("skipped_frames", [])
                existing_videos = existing.get("videos", {})
                if isinstance(existing_frames, dict):
                    output.update(existing_frames)
                if isinstance(existing_videos, dict):
                    video_stats.update(existing_videos)
                if isinstance(existing_skipped, list):
                    skipped_set.update(str(x) for x in existing_skipped)
            logging.info(
                "Resume mode: loaded %d processed videos and %d frame records from %s",
                len(video_stats),
                len(output),
                output_json,
            )
        except Exception as exc:
            logging.warning(
                "Could not load existing output JSON (%s). Starting fresh. Error: %s",
                output_json,
                exc,
            )

    if resume_from_outputs:
        recovered = 0
        for label in ("fake", "real"):
            label_root = faces_out_dir / label
            if not label_root.exists():
                continue
            existing_dirs = [p for p in label_root.iterdir() if p.is_dir()]
            for d in existing_dirs:
                # Keep compatibility with current naming (sanitized video stem).
                video_stem = d.name
                dataset_subdir = "Deepfakes" if label == "fake" else "original"
                matched_video: Optional[Path] = None
                for candidate in collect_video_paths(ffpp_root / dataset_subdir):
                    if sanitize_stem(candidate.stem) == video_stem:
                        matched_video = candidate
                        break
                if matched_video is None:
                    continue
                video_key = matched_video.relative_to(ffpp_root).as_posix()
                if video_key in video_stats:
                    continue
                frames_done = len([x for x in d.iterdir() if x.is_file() and x.suffix.lower() in {".jpg", ".jpeg", ".png"}])
                video_stats[video_key] = {
                    "label": label,
                    "sampled_frames": 0,
                    "processed_frames": frames_done,
                    "skipped_frames": 0,
                }
                recovered += 1
        if recovered > 0:
            logging.info(
                "Resume mode: recovered %d processed videos from existing output folders.",
                recovered,
            )

    for label, root_dir in label_dirs.items():
        videos = collect_video_paths(root_dir)
        logging.info("[%s] Found %d videos in %s", label, len(videos), root_dir)

        for video_path in videos:
            rel_video = video_path.relative_to(ffpp_root)
            video_key = rel_video.as_posix()
            if video_key in video_stats:
                logging.info("Skipping already processed video: %s", video_key)
                continue
            frames = extract_frames_from_video(video_path, sample_every=sample_every)
            processed_count = 0
            skipped_count = 0

            video_stem = sanitize_stem(video_path.stem)
            out_video_faces = faces_out_dir / label / video_stem
            out_video_debug = debug_dir / label / video_stem

            for frame_idx, frame_bgr in frames:
                frame_name = f"{video_stem}_f{frame_idx:06d}.jpg"
                entry = process_single_array_frame(
                    frame_bgr=frame_bgr,
                    frame_name=frame_name,
                    out_face_dir=out_video_faces,
                    predictor=predictor,
                    detector=detector,
                    margin_ratio=margin_ratio,
                    target_size=target_size,
                    save_debug=save_debug,
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
    logging.info(
        "Processed frames: %d | Skipped frames: %d | Processed videos: %d",
        len(output),
        len(skipped_set),
        len(video_stats),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess 68 landmarks for ViT deepfake pipeline.")
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Directory of extracted frame images (single-folder mode).",
    )
    parser.add_argument(
        "--ffpp-root",
        type=Path,
        default=None,
        help="FaceForensics++_C23 root. Uses Deepfakes=fake and original=real.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("landmarks_68_scaled.json"),
        help="Output JSON path for scaled landmarks.",
    )
    parser.add_argument(
        "--faces-out-dir",
        type=Path,
        default=Path("faces_224"),
        help="Directory to save 224x224 cropped face images.",
    )
    parser.add_argument(
        "--predictor-path",
        type=Path,
        default=Path("shape_predictor_68_face_landmarks.dat"),
        help="Path to dlib 68-landmark predictor .dat file.",
    )
    parser.add_argument("--margin-ratio", type=float, default=0.1, help="Face crop margin ratio.")
    parser.add_argument("--target-size", type=int, default=224, help="Target resized face dimension.")
    parser.add_argument(
        "--sample-every",
        type=int,
        default=10,
        help="For --ffpp-root mode: process every Nth frame from each video.",
    )
    parser.add_argument(
        "--resume-from-outputs",
        action="store_true",
        help="For --ffpp-root mode: also skip videos already present in --faces-out-dir.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, mp.cpu_count() - 1),
        help="Number of worker processes.",
    )
    parser.add_argument("--save-debug", action="store_true", help="Save debug images with plotted landmarks.")
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=Path("debug_landmarks"),
        help="Directory for debug overlays.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(args.verbose)

    if args.frames_dir is None and args.ffpp_root is None:
        raise ValueError("Provide one mode: --frames-dir or --ffpp-root.")
    if args.frames_dir is not None and args.ffpp_root is not None:
        raise ValueError("Use only one mode at a time: --frames-dir or --ffpp-root.")

    if not args.predictor_path.exists():
        raise FileNotFoundError(
            f"Predictor file not found: {args.predictor_path}. "
            "Assume shape_predictor_68_face_landmarks.dat is in project root."
        )
    if args.target_size <= 0:
        raise ValueError("--target-size must be > 0.")

    if args.ffpp_root is not None:
        if not args.ffpp_root.exists():
            raise FileNotFoundError(f"FF++ root does not exist: {args.ffpp_root}")
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

    logging.info("Found %d frames. Processing with %d workers.", len(frame_paths), args.workers)
    work_items = [
        (
            frame_path,
            args.faces_out_dir,
            args.predictor_path,
            args.margin_ratio,
            args.target_size,
            args.save_debug,
            args.debug_dir,
        )
        for frame_path in frame_paths
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
            "skipped_frames": sorted(skipped),
            "semantic_groups_indices": SEMANTIC_GROUPS,
        },
        "frames": output,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2)

    logging.info("Saved JSON: %s", args.output_json)
    logging.info("Processed: %d | Skipped: %d", len(output), len(skipped))
    if args.save_debug:
        logging.info("Debug overlays saved to: %s", args.debug_dir)


if __name__ == "__main__":
    main()
