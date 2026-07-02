"""
train_vit_detector.py — fine-tune a ViT/16 detector on SAEDF face crops.

This is the reproducible training path for improving detector AUC while keeping
the existing 224px / 196-patch SAEDF explanation stack intact.

Example
-------
    python train_vit_detector.py \
        --train-json outputs/ffpp_landmarks_cal.json \
        --base-model Wvolf/ViT-Deepfake-Detection \
        --output-dir models/saedf_vit16_detector \
        --epochs 3
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


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


def _stratified_split(entries: List[Tuple[str, Dict]], val_fraction: float, seed: int) -> Tuple[List, List]:
    rng = random.Random(seed)
    fake = [e for e in entries if e[1].get("label") == "fake"]
    real = [e for e in entries if e[1].get("label") != "fake"]
    rng.shuffle(fake)
    rng.shuffle(real)
    n_fake_val = max(1, int(len(fake) * val_fraction))
    n_real_val = max(1, int(len(real) * val_fraction))
    val = fake[:n_fake_val] + real[:n_real_val]
    train = fake[n_fake_val:] + real[n_real_val:]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


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
    p = argparse.ArgumentParser(description="Fine-tune a ViT/16 deepfake detector.")
    p.add_argument("--train-json", type=Path, default=Path("outputs/ffpp_landmarks_cal.json"))
    p.add_argument("--base-model", type=str, default="Wvolf/ViT-Deepfake-Detection")
    p.add_argument("--output-dir", type=Path, default=Path("models/saedf_vit16_detector"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--val-fraction", type=float, default=0.15)
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
    entries = _load_entries(args.train_json)
    train_entries, val_entries = _stratified_split(entries, args.val_fraction, args.seed)
    print(f"Train frames: {len(train_entries)} | Val frames: {len(val_entries)} | Device: {device}")

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


if __name__ == "__main__":
    from console_utf8 import ensure_utf8_stdout
    ensure_utf8_stdout()
    main()
