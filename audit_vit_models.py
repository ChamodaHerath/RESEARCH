"""
audit_vit_models.py — compatibility audit for candidate SAEDF detectors.

The SAEDF explanation stack currently assumes a 224px ViT/16 backbone:
14 x 14 patches = 196 patch scores.  This script checks candidate Hugging Face
image-classification checkpoints before expensive benchmarking/fine-tuning.

Example
-------
    python audit_vit_models.py
    python audit_vit_models.py --models model_a model_b --output outputs/model_audit.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch


DEFAULT_MODELS = [
    "hamzenium/ViT-Deepfake-Classifier",
    "prithivMLmods/Deep-Fake-Detector-v2-Model",
    "prithivMLmods/Deepfake-Detection-Exp-02-21",
    "dima806/deepfake_vs_real_image_detection",
    "buildborderless/CommunityForensics-DeepfakeDet-ViT",
]


def _fake_index_from_id2label(id2label: Dict) -> int | None:
    fake_idx = None
    real_idx = None
    for raw_idx, raw_label in (id2label or {}).items():
        if not isinstance(raw_label, str):
            continue
        label = raw_label.lower()
        try:
            idx = int(raw_idx)
        except Exception:
            continue
        if any(t in label for t in ("fake", "deepfake", "forged", "manipulated", "synthetic")):
            fake_idx = idx
        if any(t in label for t in ("real", "realism", "genuine", "authentic", "pristine")):
            real_idx = idx
    if fake_idx is not None:
        return fake_idx
    if real_idx is not None and len(id2label or {}) == 2:
        return 1 - real_idx
    return None


def audit_model(model_name: str) -> Dict:
    result: Dict = {
        "model": model_name,
        "loads": False,
        "compatible": False,
        "issues": [],
    }
    try:
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification

        cfg = AutoConfig.from_pretrained(model_name)
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        model.eval()
        if hasattr(model, "set_attn_implementation"):
            try:
                model.set_attn_implementation("eager")
            except Exception:
                pass

        result["loads"] = True
        result["architecture"] = getattr(cfg, "model_type", None)
        result["id2label"] = {str(k): v for k, v in getattr(cfg, "id2label", {}).items()}
        result["num_labels"] = int(getattr(cfg, "num_labels", len(result["id2label"]) or 0))
        result["image_size"] = getattr(cfg, "image_size", None)
        result["patch_size"] = getattr(cfg, "patch_size", None)
        result["fake_class_index"] = _fake_index_from_id2label(getattr(cfg, "id2label", {}))

        if result["fake_class_index"] is None:
            result["issues"].append("Could not infer fake/deepfake class from id2label.")
        if result["num_labels"] != 2:
            result["issues"].append(f"Expected binary classifier, got num_labels={result['num_labels']}.")
        if result["image_size"] not in (224, None):
            result["issues"].append(f"Expected image_size=224 or processor-driven resize, got {result['image_size']}.")
        if result["patch_size"] not in (16, None):
            result["issues"].append(f"Expected patch_size=16 for 196-patch SAEDF mapping, got {result['patch_size']}.")

        dummy = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy, output_attentions=True, return_dict=True)

        attentions = getattr(out, "attentions", None)
        result["supports_output_attentions"] = bool(attentions)
        if not attentions:
            result["issues"].append("Model did not return attentions with output_attentions=True.")
        else:
            last = attentions[-1]
            result["last_attention_shape"] = list(last.shape)
            token_count = int(last.shape[-1])
            result["token_count"] = token_count
            if token_count != 197:
                result["issues"].append(f"Expected 197 tokens (CLS + 196 patches), got {token_count}.")

        result["processor_class"] = processor.__class__.__name__
        result["compatible"] = len(result["issues"]) == 0
    except Exception as exc:
        result["error"] = repr(exc)
        result["issues"].append(str(exc))
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit candidate ViT/16 detectors for SAEDF.")
    p.add_argument("--models", nargs="*", default=DEFAULT_MODELS, help="HF model IDs to audit.")
    p.add_argument("--output", type=Path, default=Path("outputs/model_audit.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results: List[Dict] = []
    for model_name in args.models:
        print(f"\nAuditing {model_name} ...")
        res = audit_model(model_name)
        results.append(res)
        status = "COMPATIBLE" if res.get("compatible") else "NOT COMPATIBLE"
        print(f"  {status}")
        print(f"  id2label={res.get('id2label')}")
        print(f"  patch_size={res.get('patch_size')} token_count={res.get('token_count')}")
        if res.get("issues"):
            for issue in res["issues"]:
                print(f"  - {issue}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nAudit saved to: {args.output}")


if __name__ == "__main__":
    from console_utf8 import ensure_utf8_stdout
    ensure_utf8_stdout()
    main()
