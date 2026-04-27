from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from attention_extractor import AttentionExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate baseline ViT attention heatmap for one face image.")
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to input face image.",
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
        default="vit_base_patch16_224",
        help="ViT model name for timm.create_model(...).",
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
    return parser.parse_args()


def read_and_resize_image(image_path: Path, target_size: int = 224) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    if image.shape[:2] != (target_size, target_size):
        image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return image


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be in [0,1].")

    image_bgr = read_and_resize_image(args.image, target_size=224)
    batch_bgr = np.expand_dims(image_bgr, axis=0)  # [1, 224, 224, 3]

    extractor = AttentionExtractor(
        model_name=args.model_name,
        pretrained=True,
        device=args.device,
    )

    try:
        maps, overlays = extractor.process_batch_bgr(batch_bgr, alpha=args.alpha)
        _ = maps  # Explicitly keep for future extensions/debugging.
        overlay = overlays[0]
        extractor.save_side_by_side(
            image_bgr=image_bgr,
            heatmap_overlay_bgr=overlay,
            output_path=args.output,
            original_title="Original Input Frame",
            heatmap_title="Attention Heatmap",
        )
    finally:
        extractor.close()

    print(f"Saved baseline heatmap panel to: {args.output}")

    if args.show:
        panel = cv2.imread(str(args.output))
        if panel is None:
            raise RuntimeError(f"Failed to read saved panel from: {args.output}")
        cv2.imshow("Baseline ViT Attention Heatmap", panel)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

