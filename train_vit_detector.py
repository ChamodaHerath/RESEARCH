"""
train_vit_detector.py — fine-tune a ViT/16 detector on SAEDF face crops with
leakage-controlled, provenance-documented training.

Why this script matters for the research claims
-----------------------------------------------
Off-the-shelf HuggingFace deepfake checkpoints have UNKNOWN training data, so
any in-domain FF++ number obtained with them cannot rule out train/test
leakage. This script produces a detector whose provenance is fully known:

  - Base model defaults to `google/vit-base-patch16-224-in21k` — a generic
    ImageNet-21k backbone with no deepfake exposure, so every deepfake-related
    parameter update comes from OUR documented training split.
  - Train/val separation is at the VIDEO level (never frame level), using the
    split JSONs produced by make_ffpp_splits.py. Frames of one video, and an
    FF++ original plus all fakes derived from it, are never split across
    train and val.
  - A provenance.json is written next to the checkpoint recording the base
    model, split manifest, video/frame counts, and hyperparameters, so the
    thesis can cite exactly what the detector saw.

Recommended workflow
--------------------
    # 1. Preprocess the full FF++ download (all 4 methods + originals)
    python preprocess_landmarks.py --ffpp-root <FFPP_ROOT> --output-json outputs/ffpp_landmarks.json

    # 2. Create leakage-controlled splits (use the official FF++ split files!)
    python make_ffpp_splits.py --json outputs/ffpp_landmarks.json \
        --official-splits <FFPP_ROOT>/splits

    # 3. Fine-tune
    python train_vit_detector.py \
        --train-json outputs/ffpp_landmarks_train.json \
        --val-json outputs/ffpp_landmarks_val.json \
        --output-dir models/saedf_vit16_detector --epochs 3

    # 4. Recalibrate + evaluate the full pipeline with the new detector
    python recalibrate_v2.py --json outputs/ffpp_landmarks_val.json \
        --model-name models/saedf_vit16_detector --output configs/fusion_config.json
    python evaluate_pipeline.py --json outputs/ffpp_landmarks_test_full.json \
        --model-name models/saedf_vit16_detector
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ffpp_utils import video_path_from_key

DEFAULT_BASE_MODEL = "google/vit-base-patch16-224-in21k"


class FaceCropDataset(Dataset):
    def __init__(self, entries: List[Tuple[str, Dict]], processor) -> None:
        self.entries = entries
        self.processor = processor

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        _, entry = self.entries[idx]
        image = cv2.imread(entry["face_image"])
        if image is None:
            raise FileNotFoundError(entry["face_image"])
        if image.shape[:2] != (224, 224):
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        image_rgb = image[..., ::-1]
        inputs = self.processor(images=image_rgb, return_tensors="pt")
        label = 1 if entry.get("label") == "fake" else 0
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def _load_entries(path: Path) -> List[Tuple[str, Dict]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    entries = list(data.get("frames", data).items())
    return [e for e in entries if Path(e[1].get("face_image", "")).exists()]


def _video_level_split(
    entries: List[Tuple[str, Dict]],
    val_fraction: float,
    seed: int,
) -> Tuple[List, List]:
    """
    Fallback when no --val-json is given: split by VIDEO, stratified by label,
    so no video contributes frames to both train and val. Prefer explicit
    make_ffpp_splits.py outputs — this fallback does not enforce
    identity-level (original vs derived fakes) separation.
    """
    by_video: Dict[str, List[Tuple[str, Dict]]] = defaultdict(list)
    for key, entry in entries:
        by_video[entry.get("video_path") or video_path_from_key(key)].append((key, entry))

    rng = random.Random(seed)
    fake_videos = sorted(v for v, es in by_video.items() if es[0][1].get("label") == "fake")
    real_videos = sorted(v for v, es in by_video.items() if es[0][1].get("label") != "fake")
    rng.shuffle(fake_videos)
    rng.shuffle(real_videos)

    n_fake_val = max(1, int(len(fake_videos) * val_fraction))
    n_real_val = max(1, int(len(real_videos) * val_fraction))
    val_videos = set(fake_videos[:n_fake_val] + real_videos[:n_real_val])

    train = [e for v, es in by_video.items() if v not in val_videos for e in es]
    val = [e for v, es in by_video.items() if v in val_videos for e in es]
    rng.shuffle(train)
    return train, val


def _count_stats(entries: List[Tuple[str, Dict]]) -> Dict:
    videos = {e[1].get("video_path") or video_path_from_key(e[0]) for e in entries}
    n_fake = sum(1 for e in entries if e[1].get("label") == "fake")
    return {
        "n_frames": len(entries),
        "n_videos": len(videos),
        "n_fake_frames": n_fake,
        "n_real_frames": len(entries) - n_fake,
    }


def _check_no_video_overlap(train: List[Tuple[str, Dict]], val: List[Tuple[str, Dict]]) -> None:
    train_videos = {e[1].get("video_path") or video_path_from_key(e[0]) for e in train}
    val_videos = {e[1].get("video_path") or video_path_from_key(e[0]) for e in val}
    overlap = train_videos & val_videos
    if overlap:
        raise RuntimeError(
            f"Video-level leakage between train and val: {sorted(overlap)[:5]}"
            f"{' ...' if len(overlap) > 5 else ''}"
        )


def _eval(model, loader, device) -> Dict:
    model.eval()
    y_true: List[int] = []
    y_prob: List[float] = []
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            out = model(pixel_values=pixel_values)
            probs = torch.softmax(out.logits, dim=-1)[:, 1]
            y_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
            y_prob.extend(probs.detach().cpu().numpy().astype(float).tolist())
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    tn = sum(t == 0 and p == 0 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
    metrics = {
        "accuracy": round((tp + tn) / max(len(y_true), 1), 4),
        "precision": round(tp / max(tp + fp, 1), 4),
        "recall": round(tp / max(tp + fn, 1), 4),
        "f1": round((2 * tp) / max(2 * tp + fp + fn, 1), 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }
    try:
        from sklearn.metrics import roc_auc_score, brier_score_loss

        metrics["auc_roc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
        metrics["brier_score"] = round(float(brier_score_loss(y_true, y_prob)), 4)
    except Exception as exc:
        metrics["sklearn_note"] = str(exc)
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune a ViT/16 deepfake detector with leakage-controlled splits."
    )
    p.add_argument("--train-json", type=Path, default=Path("outputs/ffpp_landmarks_train.json"),
                   help="Training-split landmark JSON (from make_ffpp_splits.py).")
    p.add_argument("--val-json", type=Path, default=None,
                   help="Validation-split landmark JSON. Strongly recommended; when "
                        "omitted, a video-level split of --train-json is used.")
    p.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL,
                   help="Base checkpoint. Default is a generic ImageNet-21k ViT with "
                        "no deepfake exposure, keeping training provenance fully known.")
    p.add_argument("--output-dir", type=Path, default=Path("models/saedf_vit16_detector"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--val-fraction", type=float, default=0.15,
                   help="Video-level val fraction when --val-json is omitted.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    from transformers import AutoImageProcessor, AutoModelForImageClassification

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.val_json is not None:
        train_entries = _load_entries(args.train_json)
        val_entries = _load_entries(args.val_json)
        split_source = f"explicit ({args.train_json} / {args.val_json})"
    else:
        entries = _load_entries(args.train_json)
        train_entries, val_entries = _video_level_split(entries, args.val_fraction, args.seed)
        split_source = f"video-level fallback split of {args.train_json} (val={args.val_fraction})"
        print("[NOTE] No --val-json given: using a video-level fallback split. "
              "Prefer make_ffpp_splits.py outputs for identity-level leakage control.")

    _check_no_video_overlap(train_entries, val_entries)
    train_stats = _count_stats(train_entries)
    val_stats = _count_stats(val_entries)
    print(f"Train: {train_stats} | Val: {val_stats} | Device: {device}")
    print(f"Split source: {split_source}")
    print("Video-level leakage check: passed (no video in both train and val)")

    processor = AutoImageProcessor.from_pretrained(args.base_model)
    model = AutoModelForImageClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        id2label={0: "Real", 1: "Fake"},
        label2id={"Real": 0, "Fake": 1},
        ignore_mismatched_sizes=True,
    ).to(device)

    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name and "head" not in name:
                param.requires_grad = False

    train_loader = DataLoader(FaceCropDataset(train_entries, processor), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(FaceCropDataset(val_entries, processor), batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    best_auc = -1.0
    best_metrics: Dict = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    provenance = {
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "base_model": args.base_model,
        "base_model_note": (
            "Generic ImageNet-21k backbone (no deepfake exposure) — all "
            "deepfake-specific learning comes from the documented split below."
            if args.base_model == DEFAULT_BASE_MODEL else
            "Non-default base model: verify its training data before claiming "
            "leakage-free in-domain results."
        ),
        "train_json": str(args.train_json),
        "val_json": str(args.val_json) if args.val_json else None,
        "split_source": split_source,
        "video_level_leakage_check": "passed",
        "train_stats": train_stats,
        "val_stats": val_stats,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "freeze_backbone": bool(args.freeze_backbone),
        },
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            out = model(pixel_values=pixel_values, labels=labels)
            loss = out.loss
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu().item())
            if step % 20 == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={running / step:.4f}")

        metrics = _eval(model, val_loader, device)
        print(f"epoch={epoch} validation={metrics}")
        auc = float(metrics.get("auc_roc", metrics["accuracy"]))
        if auc > best_auc:
            best_auc = auc
            best_metrics = metrics
            model.save_pretrained(args.output_dir)
            processor.save_pretrained(args.output_dir)
            (args.output_dir / "training_metrics.json").write_text(
                json.dumps({"best_epoch": epoch, "metrics": best_metrics}, indent=2),
                encoding="utf-8",
            )
            print(f"Saved best checkpoint to {args.output_dir}")

    print(
        "\nNext steps for defensible reporting:\n"
        f"  1. recalibrate_v2.py --model-name {args.output_dir}\n"
        f"  2. evaluate_pipeline.py --json outputs/ffpp_landmarks_test.json "
        f"--model-name {args.output_dir}\n"
        "  3. Cross-dataset check on Celeb-DF before quoting any number."
    )


if __name__ == "__main__":
    from console_utf8 import ensure_utf8_stdout
    ensure_utf8_stdout()
    main()
