"""Evaluate downstream fetal-head segmentation utility for HC18 SR methods.

This script freezes the improved HC18 U-Net from
scripts/39_improve_hc18_unet_segmentation.py and measures how different
reconstruction inputs affect fetal-head mask quality on the HC18 test split.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from hc18_dataset import HC18SRDataset
from models.swinir_light import build_swinir_light


MANIFEST = PROJECT_ROOT / "data" / "processed" / "HC18_SR" / "hc18_lq_manifest.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
SUMMARY_CSV = RESULTS_DIR / "downstream_utility_summary.csv"
PER_IMAGE_CSV = RESULTS_DIR / "downstream_utility_per_image.csv"
MASK_ROOT = RESULTS_DIR / "downstream_utility_masks"

SR_ONLY_CHECKPOINT = RESULTS_DIR / "models" / "sr_only_hc18_best.pth"
QA_CHECKPOINT = RESULTS_DIR / "models" / "quality_aware_sr_hc18_l1_mse_best.pth"
SR_PERCEPTUAL_CHECKPOINT = RESULTS_DIR / "models" / "sr_perceptual_hc18_final.pth"
QA_PERCEPTUAL_CHECKPOINT = RESULTS_DIR / "models" / "qa_sr_perceptual_hc18_final.pth"
SWINIR_LABEL = "SwinIR-Light Phase A"
SWINIR_CHECKPOINT = RESULTS_DIR / "models" / "swinir_light_phase_a_seed42" / "best_lpips.pth"
SWINIR_EVAL_DIR = RESULTS_DIR / "eval" / "swinir_light_phase_a" / "test"
CNN_RESIDUAL_OUTPUT_DIR = RESULTS_DIR / "sr_outputs" / "cnn_residual_diffusion_sr"


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    kind: str
    model_key: str | None = None
    checkpoint: Path | None = None


METHODS = [
    MethodSpec("hr", "HR", "hr"),
    MethodSpec("lq_mixed", "LQ_mixed", "lq"),
    MethodSpec("sr_only", "SR-only", "sr_model", "sr_only", SR_ONLY_CHECKPOINT),
    MethodSpec(
        "sr_only_perceptual",
        "SR-only+perceptual",
        "sr_model",
        "sr_only",
        SR_PERCEPTUAL_CHECKPOINT,
    ),
    MethodSpec("quality_aware", "Quality-aware", "sr_model", "quality_aware", QA_CHECKPOINT),
    MethodSpec(
        "quality_aware_perceptual",
        "Quality-aware+perceptual",
        "sr_model",
        "quality_aware",
        QA_PERCEPTUAL_CHECKPOINT,
    ),
    MethodSpec("swinir_light_phase_a", SWINIR_LABEL, "sr_model", "swinir_light", SWINIR_CHECKPOINT),
    MethodSpec("cnn_residual_diffusion", "CNN-Residual Diffusion", "png"),
]


SUMMARY_COLUMNS = [
    "method",
    "Dice_mean",
    "Dice_std",
    "IoU_mean",
    "AreaErr_mean",
    "DiamErr_mean",
]

PER_IMAGE_COLUMNS = [
    "image_id",
    "method",
    "Dice",
    "IoU",
    "AreaErr",
    "DiamErr",
    "pred_area",
    "GT_area",
    "pred_equivalent_diameter",
    "GT_equivalent_diameter",
    "pred_mask_path",
]


def load_module(module_name: str, script_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            state_dict = checkpoint.get(key)
            if isinstance(state_dict, dict):
                return state_dict
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu().squeeze().numpy().astype(np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {image.shape}")
    return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def ensure_grayscale_tensor(output: torch.Tensor) -> torch.Tensor:
    if output.dim() == 4 and output.size(1) == 3:
        weights = output.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (output * weights).sum(dim=1, keepdim=True)
    return output


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = (mask.astype(np.uint8) * 255).astype(np.uint8)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to save predicted mask: {path}")


def load_sr_models(
    device: torch.device,
    sr_module: Any,
    qa_module: Any,
) -> dict[str, torch.nn.Module]:
    models: dict[str, torch.nn.Module] = {}
    for method in METHODS:
        if method.kind != "sr_model":
            continue
        if method.checkpoint is None or not method.checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint for {method.label}: {method.checkpoint}")

        if method.model_key == "sr_only":
            model = sr_module.ResidualSRModel().to(device)
        elif method.model_key == "quality_aware":
            model = qa_module.QualityAwareSRModel().to(device)
        elif method.model_key == "swinir_light":
            model = build_swinir_light(use_checkpoint=False).to(device)
        else:
            raise ValueError(f"Unsupported SR model key: {method.model_key}")

        checkpoint = torch.load(method.checkpoint, map_location=device)
        model.load_state_dict(extract_state_dict(checkpoint), strict=True)
        model.eval()
        models[method.key] = model
    return models


def compute_quality_stats(
    train_dataset: HC18SRDataset,
    qa_module: Any,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = []
    with torch.no_grad():
        for index in range(len(train_dataset)):
            lq = train_dataset[index]["lq"].unsqueeze(0).to(device=device, dtype=torch.float32)
            feature = qa_module.compute_quality_features(lq).detach().cpu().view(-1)
            features.append(feature)
    stacked = torch.stack(features, dim=0)
    mean = stacked.mean(dim=0).to(device).view(1, -1)
    std = stacked.std(dim=0).to(device).view(1, -1)
    std = torch.where(std == 0.0, torch.ones_like(std), std)
    return mean, std


def resolve_manifest_path(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else MANIFEST.parent / path


@torch.no_grad()
def make_sr_image(
    method: MethodSpec,
    sample: dict[str, Any],
    models: dict[str, torch.nn.Module],
    qa_module: Any,
    quality_mean: torch.Tensor,
    quality_std: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    if method.kind == "hr":
        return tensor_to_uint8_image(sample["hr"])
    if method.kind == "lq":
        return tensor_to_uint8_image(sample["lq"])
    if method.kind == "png":
        image_path = CNN_RESIDUAL_OUTPUT_DIR / f"{sample['image_id']}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Missing CNN residual diffusion output: {image_path}")
        return np.squeeze(image).astype(np.uint8)
    if method.kind != "sr_model":
        raise ValueError(f"Unsupported method kind: {method.kind}")

    lq = sample["lq"].unsqueeze(0).to(device=device, dtype=torch.float32)
    model = models[method.key]
    if method.model_key == "quality_aware":
        quality = qa_module.compute_quality_features(lq)
        quality = (quality - quality_mean.to(quality.device)) / quality_std.to(quality.device)
        output = model(lq, quality).clamp(0.0, 1.0)
    else:
        output = model(lq).clamp(0.0, 1.0)
    output = ensure_grayscale_tensor(output).clamp(0.0, 1.0)
    return tensor_to_uint8_image(output)


def build_metric_row(
    seg_module: Any,
    image_id: str,
    method: MethodSpec,
    predicted_mask: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask_path: Path,
) -> dict[str, Any]:
    metrics = seg_module.compute_segmentation_metrics(predicted_mask, gt_mask)
    return {
        "image_id": image_id,
        "method": method.label,
        "Dice": float(metrics["dice"]),
        "IoU": float(metrics["iou"]),
        "AreaErr": float(metrics["fetal_head_area_error_percent"]) / 100.0,
        "DiamErr": float(metrics["equivalent_diameter_error_px"]),
        "pred_area": float(metrics["predicted_area_px"]),
        "GT_area": float(metrics["ground_truth_area_px"]),
        "pred_equivalent_diameter": float(metrics["predicted_equivalent_diameter_px"]),
        "GT_equivalent_diameter": float(metrics["ground_truth_equivalent_diameter_px"]),
        "pred_mask_path": str(pred_mask_path.relative_to(PROJECT_ROOT)),
    }


def summarize(rows: list[dict[str, Any]]) -> pd.DataFrame:
    per_image = pd.DataFrame(rows, columns=PER_IMAGE_COLUMNS)
    summary_rows = []
    for method in METHODS:
        method_df = per_image[per_image["method"] == method.label]
        if method_df.empty:
            continue
        summary_rows.append(
            {
                "method": method.label,
                "Dice_mean": float(method_df["Dice"].mean()),
                "Dice_std": float(method_df["Dice"].std(ddof=0)),
                "IoU_mean": float(method_df["IoU"].mean()),
                "AreaErr_mean": float(method_df["AreaErr"].mean()),
                "DiamErr_mean": float(method_df["DiamErr"].mean()),
            }
        )
    return pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)


def print_acceptance(summary_df: pd.DataFrame) -> None:
    dice_by_method = {
        str(row["method"]): float(row["Dice_mean"])
        for _, row in summary_df.iterrows()
    }
    lq = dice_by_method.get("LQ_mixed")
    sr_perc = dice_by_method.get("SR-only+perceptual")
    qa_perc = dice_by_method.get("Quality-aware+perceptual")

    print()
    if lq is None or sr_perc is None or qa_perc is None:
        print("ACCEPTANCE: FAIL - missing one or more required Dice rows.")
        return

    qa_vs_sr = qa_perc - sr_perc
    sr_vs_lq = sr_perc - lq
    qa_vs_lq = qa_perc - lq
    passed = qa_perc >= sr_perc and sr_perc > lq and qa_perc > lq
    status = "PASS" if passed else "FAIL"
    print(f"ACCEPTANCE: {status}")
    print(f"LQ_mixed Dice_mean: {lq:.6f}")
    print(f"SR-only+perceptual Dice_mean: {sr_perc:.6f} (delta vs LQ: {sr_vs_lq:+.6f})")
    print(
        "Quality-aware+perceptual Dice_mean: "
        f"{qa_perc:.6f} (delta vs LQ: {qa_vs_lq:+.6f}; "
        f"delta vs SR-only+perceptual: {qa_vs_sr:+.6f})"
    )


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MASK_ROOT.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    seg_module = load_module("hc18_improved_unet_segmentation", SCRIPT_DIR / "39_improve_hc18_unet_segmentation.py")
    sr_module = load_module("hc18_sr_only_train", SCRIPT_DIR / "05_train_sr_only_hc18.py")
    qa_module = load_module("hc18_quality_aware_train", SCRIPT_DIR / "07_train_quality_aware_sr_hc18.py")

    seg_model, seg_checkpoint = seg_module.load_best_model(device)
    threshold = float(seg_checkpoint.get("threshold", 0.50))
    image_size = int(seg_checkpoint.get("image_size", 256))
    print(
        "Loaded frozen HC18 U-Net: "
        f"epoch={seg_checkpoint.get('epoch')} threshold={threshold:.2f} "
        f"image_size={image_size}"
    )

    train_dataset = HC18SRDataset(
        csv_path=str(MANIFEST),
        input_type="LQ_mixed",
        split="train",
        return_mask=False,
    )
    test_dataset = HC18SRDataset(
        csv_path=str(MANIFEST),
        input_type="LQ_mixed",
        split="test",
        return_mask=False,
    )

    print("Loading SR checkpoints...")
    models = load_sr_models(device, sr_module, qa_module)
    print("Computing quality-aware feature normalization from the train split...")
    quality_mean, quality_std = compute_quality_stats(train_dataset, qa_module, device)

    rows: list[dict[str, Any]] = []
    skipped_empty_masks = 0
    skipped_unreadable_masks = 0

    print(f"Evaluating {len(test_dataset)} HC18 test images across {len(METHODS)} inputs...")
    for index in range(len(test_dataset)):
        sample = test_dataset[index]
        manifest_row = test_dataset.samples[index]
        image_id = str(sample["image_id"])
        mask_path = resolve_manifest_path(manifest_row.get("mask_path"))
        raw_gt_mask = seg_module.read_gray_uint8(mask_path)
        if raw_gt_mask is None:
            skipped_unreadable_masks += 1
            continue
        gt_mask = seg_module.preprocess_mask(raw_gt_mask).processed_mask.astype(bool)
        if not np.any(gt_mask):
            skipped_empty_masks += 1
            continue

        for method in METHODS:
            input_image = make_sr_image(
                method,
                sample,
                models,
                qa_module,
                quality_mean,
                quality_std,
                device,
            )
            predicted_mask = seg_module.predict_mask(
                model=seg_model,
                image=input_image,
                target_shape=gt_mask.shape,
                image_size=image_size,
                threshold=threshold,
                device=device,
            )
            pred_mask_path = MASK_ROOT / method.key / f"{image_id}.png"
            save_mask(pred_mask_path, predicted_mask)
            rows.append(
                build_metric_row(
                    seg_module,
                    image_id,
                    method,
                    predicted_mask,
                    gt_mask,
                    pred_mask_path,
                )
            )

    per_image_df = pd.DataFrame(rows, columns=PER_IMAGE_COLUMNS)
    summary_df = summarize(rows)

    with PER_IMAGE_CSV.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PER_IMAGE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    swinir_rows = [row for row in rows if row["method"] == SWINIR_LABEL]
    if swinir_rows:
        SWINIR_EVAL_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(swinir_rows, columns=PER_IMAGE_COLUMNS).to_csv(
            SWINIR_EVAL_DIR / "downstream_per_image.csv",
            index=False,
        )
        summary_df[summary_df["method"] == SWINIR_LABEL].to_csv(
            SWINIR_EVAL_DIR / "downstream_summary.csv",
            index=False,
        )

    if skipped_unreadable_masks or skipped_empty_masks:
        print(
            "Skipped masks: "
            f"unreadable={skipped_unreadable_masks}, empty_after_preprocessing={skipped_empty_masks}"
        )

    print()
    print("Downstream utility summary")
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    print(f"Wrote per-image metrics: {PER_IMAGE_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Wrote summary metrics: {SUMMARY_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Wrote predicted masks under: {MASK_ROOT.relative_to(PROJECT_ROOT)}")
    if swinir_rows:
        print(f"Wrote SwinIR downstream rows: {(SWINIR_EVAL_DIR / 'downstream_per_image.csv').relative_to(PROJECT_ROOT)}")
    print_acceptance(summary_df)

    if per_image_df.empty:
        raise RuntimeError("No downstream utility rows were generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
