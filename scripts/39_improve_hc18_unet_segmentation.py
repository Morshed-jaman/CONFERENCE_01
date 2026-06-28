r"""Improve HC18 fetal-head U-Net segmentation and evaluation.

PowerShell:

    python .\scripts\39_improve_hc18_unet_segmentation.py

The script intentionally stops before full training if the forced mask
preprocessing method, image-mask pairing, or the 20-sample overfit sanity check
appears unreliable.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import cv2
    import matplotlib
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard.
    missing_package = exc.name or "unknown"
    raise SystemExit(
        "Missing required dependency: "
        f"{missing_package}. Activate a Python environment with PyTorch, "
        "OpenCV, pandas, numpy, and matplotlib before rerunning this script."
    ) from exc


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HC18_ROOT = PROJECT_ROOT / "data" / "processed" / "HC18_SR"
TRAIN_CSV = HC18_ROOT / "train.csv"
VAL_CSV = HC18_ROOT / "val.csv"
TEST_CSV = HC18_ROOT / "test.csv"
LQ_MANIFEST_CSV = HC18_ROOT / "hc18_lq_manifest.csv"
HR_DIR = HC18_ROOT / "HR"
MASK_DIR = HC18_ROOT / "masks"
LQ_MIXED_DIR = HC18_ROOT / "LQ_mixed"
SR_OUTPUT_ROOT = PROJECT_ROOT / "results" / "sr_outputs"

MODEL_PATH = PROJECT_ROOT / "results" / "models" / "hc18_unet_segmentation_best_improved.pth"
MASK_STATS_CSV = PROJECT_ROOT / "results" / "hc18_unet_improved_mask_statistics.csv"
PER_IMAGE_CSV = PROJECT_ROOT / "results" / "hc18_unet_improved_metrics_per_image.csv"
SUMMARY_CSV = PROJECT_ROOT / "results" / "hc18_unet_improved_metrics_summary.csv"
TABLE_MD = PROJECT_ROOT / "results" / "hc18_unet_improved_metrics_table.md"
FIGURE_DIR = PROJECT_ROOT / "results" / "hc18_unet_segmentation_figures"

DEBUG_PAIRING_FIGURE = FIGURE_DIR / "debug_image_mask_pairing.png"
OVERFIT_FIGURE = FIGURE_DIR / "overfit_sanity_check.png"
TRAINING_CURVE_FIGURE = FIGURE_DIR / "improved_training_curve.png"
EXAMPLE_FIGURE = FIGURE_DIR / "improved_example_segmentation_outputs.png"
DICE_IOU_FIGURE = FIGURE_DIR / "improved_dice_iou_comparison.png"
MEASUREMENT_ERROR_FIGURE = FIGURE_DIR / "improved_measurement_error_comparison.png"

SELECTED_MASK_PREPROCESSING_METHOD = "F_ellipse_fitted_filled"
OVERFIT_DICE_PASS_THRESHOLD = 0.90
HR_TEST_DICE_PASS_THRESHOLD = 0.60

SEGMENTATION_NOTE = (
    "Dice, IoU, and fetal measurement error are computed using an HC18-trained "
    "U-Net segmentation model with ellipse-fitted filled mask preprocessing "
    "and HC18 fetal-head masks. These metrics are used as supporting "
    "anatomy-preservation evidence and should not be interpreted as expert "
    "clinical validation."
)

WEAK_SEGMENTATION_WARNING = (
    "The segmentation model did not reach sufficient reliability for strong "
    "Dice/IoU reporting."
)

THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

EVAL_METHODS = [
    {"key": "hr", "label": "HR ground truth", "kind": "hr", "output_dir": HR_DIR},
    {
        "key": "lq_mixed",
        "label": "LQ mixed baseline",
        "kind": "lq_mixed",
        "output_dir": LQ_MIXED_DIR,
    },
    {
        "key": "sr_only",
        "label": "SR-only CNN",
        "kind": "sr_output",
        "output_dir": SR_OUTPUT_ROOT / "sr_only",
    },
    {
        "key": "quality_aware",
        "label": "Quality-aware CNN",
        "kind": "sr_output",
        "output_dir": SR_OUTPUT_ROOT / "quality_aware",
    },
    {
        "key": "quality_aware_l1_mse",
        "label": "Quality-aware CNN + L1/MSE",
        "kind": "sr_output",
        "output_dir": SR_OUTPUT_ROOT / "quality_aware_l1_mse",
    },
    {
        "key": "diffusion_sr",
        "label": "Conditional Diffusion SR",
        "kind": "sr_output",
        "output_dir": SR_OUTPUT_ROOT / "diffusion_sr",
    },
    {
        "key": "quality_aware_diffusion_sr_v2",
        "label": "Quality-aware Diffusion SR v2",
        "kind": "sr_output",
        "output_dir": SR_OUTPUT_ROOT / "quality_aware_diffusion_sr_v2",
    },
    {
        "key": "cnn_residual_diffusion_sr",
        "label": "CNN-Residual Diffusion SR",
        "kind": "sr_output",
        "output_dir": SR_OUTPUT_ROOT / "cnn_residual_diffusion_sr",
    },
]

PER_IMAGE_COLUMNS = [
    "image_id",
    "method",
    "input_path",
    "mask_path",
    "dice",
    "iou",
    "boundary_error_px",
    "predicted_area_px",
    "ground_truth_area_px",
    "fetal_head_area_error_percent",
    "predicted_equivalent_diameter_px",
    "ground_truth_equivalent_diameter_px",
    "equivalent_diameter_error_px",
    "equivalent_diameter_error_percent",
    "reliability_note",
]

SUMMARY_COLUMNS = [
    "method",
    "n_images",
    "mean_dice",
    "std_dice",
    "mean_iou",
    "std_iou",
    "mean_boundary_error_px",
    "std_boundary_error_px",
    "mean_fetal_head_area_error_percent",
    "std_fetal_head_area_error_percent",
    "mean_equivalent_diameter_error_px",
    "std_equivalent_diameter_error_px",
    "mean_equivalent_diameter_error_percent",
    "std_equivalent_diameter_error_percent",
    "reliability_note",
]


@dataclass(frozen=True)
class SampleRecord:
    image_id: str
    split: str
    image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class PreprocessedMask:
    raw_mask: np.ndarray
    processed_mask: np.ndarray
    raw_area_ratio: float
    processed_area_ratio: float
    is_empty_mask: bool
    is_very_small_mask: bool
    is_suspicious_mask: bool


@dataclass
class MaskAudit:
    stats_df: pd.DataFrame
    records_by_split: dict[str, list[SampleRecord]]
    checked_masks: int
    suspicious_masks: int
    missing_images: int
    missing_masks: int

    @property
    def suspicious_fraction(self) -> float:
        if self.checked_masks == 0:
            return 1.0
        return self.suspicious_masks / self.checked_masks


@dataclass
class TrainingResult:
    history: list[dict[str, float]]
    best_val_dice: float
    best_val_iou: float
    best_epoch: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Debug HC18 masks, run an overfit sanity check, train an improved "
            "U-Net, tune threshold, and evaluate segmentation-based anatomy "
            "metrics for available SR outputs."
        )
    )
    parser.add_argument("--epochs", type=int, default=100, help="Full training epochs.")
    parser.add_argument(
        "--overfit-epochs",
        type=int,
        default=100,
        help="Maximum epochs for the 20-sample overfit sanity check.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--image-size", type=int, default=256, help="Model input size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Keep 0 for Windows PowerShell compatibility.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def warn(message: str) -> None:
    print(f"WARNING: {message}")


def resolve_csv_path(value: Any, base_dir: Path = HC18_ROOT) -> Path | None:
    if pd.isna(value):
        return None
    path_text = str(value).strip()
    if not path_text:
        return None
    path = Path(path_text)
    return path if path.is_absolute() else base_dir / path


def read_split_csv(csv_path: Path, split_name: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing split CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if "has_mask" in df.columns:
        df = df[df["has_mask"].astype(int) == 1].copy()
    df["split"] = split_name
    if "image_id" not in df.columns:
        raise ValueError(f"Missing image_id column in {csv_path}")
    return df.reset_index(drop=True)


def resolve_hr_path(row: pd.Series, image_id: str) -> Path:
    csv_path = resolve_csv_path(row.get("hr_path"))
    if csv_path is not None:
        return csv_path
    return HR_DIR / f"{image_id}.png"


def resolve_mask_path(row: pd.Series, image_id: str) -> Path:
    csv_path = resolve_csv_path(row.get("mask_path"))
    if csv_path is not None:
        return csv_path
    return MASK_DIR / f"{image_id}.png"


def read_gray_uint8(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return np.squeeze(image).astype(np.uint8)


def fill_holes(binary_mask: np.ndarray) -> np.ndarray:
    binary = (binary_mask > 0).astype(np.uint8)
    if not np.any(binary):
        return binary

    padded = np.pad(binary, 1, mode="constant", constant_values=0)
    flood = padded.copy()
    flood_mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 1)
    holes = (flood[1:-1, 1:-1] == 0) & (binary == 0)
    return np.logical_or(binary > 0, holes).astype(np.uint8)


def binary_mask(mask_gray: np.ndarray) -> np.ndarray:
    return (mask_gray > 0).astype(np.uint8)


def external_contours(mask: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return list(contours)


def largest_contour_filled(binary_mask: np.ndarray) -> np.ndarray:
    binary = (binary_mask > 0).astype(np.uint8)
    contours = external_contours(binary)
    if not contours:
        return np.zeros_like(binary, dtype=np.uint8)

    largest = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(binary, dtype=np.uint8)
    cv2.drawContours(filled, [largest], contourIdx=-1, color=1, thickness=cv2.FILLED)
    return filled


def preprocess_ellipse_fitted(mask_gray: np.ndarray) -> np.ndarray:
    raw = binary_mask(mask_gray)
    contours = external_contours(raw)
    if not contours:
        return np.zeros_like(raw, dtype=np.uint8)

    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return fill_holes(largest_contour_filled(raw))

    ellipse = cv2.fitEllipse(largest)
    output = np.zeros_like(raw, dtype=np.uint8)
    cv2.ellipse(output, ellipse, color=1, thickness=cv2.FILLED)
    return fill_holes(output)


def preprocess_mask(mask_gray: np.ndarray) -> PreprocessedMask:
    if SELECTED_MASK_PREPROCESSING_METHOD != "F_ellipse_fitted_filled":
        raise RuntimeError(
            "This script is fixed to F_ellipse_fitted_filled mask preprocessing."
        )

    raw_mask = binary_mask(mask_gray)
    raw_area_ratio = float(np.mean(raw_mask))

    processed = preprocess_ellipse_fitted(mask_gray)
    processed = (processed > 0).astype(np.uint8)

    processed_area_ratio = float(np.mean(processed))
    is_empty = processed_area_ratio <= 0.0
    is_very_small = processed_area_ratio < 0.005
    is_suspicious = is_empty or is_very_small or processed_area_ratio > 0.80

    return PreprocessedMask(
        raw_mask=raw_mask,
        processed_mask=processed,
        raw_area_ratio=raw_area_ratio,
        processed_area_ratio=processed_area_ratio,
        is_empty_mask=is_empty,
        is_very_small_mask=is_very_small,
        is_suspicious_mask=is_suspicious,
    )


def resize_image_01(image: np.ndarray, image_size: int) -> np.ndarray:
    resized = cv2.resize(
        image,
        (image_size, image_size),
        interpolation=cv2.INTER_LINEAR,
    )
    return np.clip(resized.astype(np.float32) / 255.0, 0.0, 1.0)


def resize_mask_01(mask: np.ndarray, image_size: int) -> np.ndarray:
    resized = cv2.resize(
        (mask > 0).astype(np.uint8),
        (image_size, image_size),
        interpolation=cv2.INTER_NEAREST,
    )
    return (resized > 0).astype(np.float32)


def normalize_panel(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.bool_:
        image = image.astype(np.float32)
    image_float = image.astype(np.float32)
    if image_float.max() > 1.0:
        image_float /= 255.0
    image_float = np.clip(image_float, 0.0, 1.0)
    return (image_float * 255.0).astype(np.uint8)


def mask_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = normalize_panel(image)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    mask_bool = mask > 0
    rgb[mask_bool, 1] = np.maximum(rgb[mask_bool, 1], 160)
    boundary = extract_boundary(mask_bool)
    rgb[boundary] = np.array([255, 0, 0], dtype=np.uint8)
    return rgb


def make_panel(image: np.ndarray, is_mask: bool = False) -> np.ndarray:
    panel = normalize_panel(image)
    if is_mask:
        panel = (panel > 0).astype(np.uint8) * 255
    return np.repeat(panel[..., None], 3, axis=2)


def resize_panel(panel: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(panel, (size, size), interpolation=cv2.INTER_AREA)


def save_debug_pairing_grid(records: list[dict[str, Any]], output_path: Path) -> None:
    if not records:
        warn("No readable samples were available for the image-mask pairing grid.")
        return

    panel_size = 128
    label_width = 180
    header_height = 28
    row_height = panel_size + 18
    n_cols = 4
    canvas_h = header_height + row_height * len(records)
    canvas_w = label_width + panel_size * n_cols
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    headers = ["HR image", "Raw mask", "Filled mask", "Overlay"]
    for col, title in enumerate(headers):
        x = label_width + col * panel_size + 6
        cv2.putText(canvas, title, (x, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    for row_idx, record in enumerate(records):
        y = header_height + row_idx * row_height
        label = f"{record['split']} {record['image_id']}"
        cv2.putText(
            canvas,
            label[:24],
            (6, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
        )
        panels = [
            make_panel(record["image"]),
            make_panel(record["raw_mask"], is_mask=True),
            make_panel(record["processed_mask"], is_mask=True),
            mask_overlay(record["image"], record["processed_mask"]),
        ]
        for col, panel in enumerate(panels):
            resized = resize_panel(panel, panel_size)
            x0 = label_width + col * panel_size
            y0 = y + 10
            canvas[y0 : y0 + panel_size, x0 : x0 + panel_size] = resized

    fig_w = canvas_w / 300.0
    fig_h = canvas_h / 300.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    ax.imshow(canvas)
    ax.axis("off")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def audit_masks(
    split_frames: dict[str, pd.DataFrame],
    seed: int,
) -> MaskAudit:
    rng = random.Random(seed)
    stats_rows: list[dict[str, Any]] = []
    debug_records: list[dict[str, Any]] = []
    records_by_split: dict[str, list[SampleRecord]] = {split: [] for split in split_frames}
    checked_masks = 0
    suspicious_masks = 0
    missing_images = 0
    missing_masks = 0

    for split_name, df in split_frames.items():
        readable_debug_candidates: list[dict[str, Any]] = []

        for _, row in df.iterrows():
            image_id = str(row["image_id"]).strip()
            image_path = resolve_hr_path(row, image_id)
            mask_path = resolve_mask_path(row, image_id)

            image = read_gray_uint8(image_path)
            raw_mask_gray = read_gray_uint8(mask_path)
            image_missing = image is None
            mask_missing = raw_mask_gray is None

            if image_missing:
                missing_images += 1
                print(f"Missing HR image, skipped: {image_path}")
            if mask_missing:
                missing_masks += 1
                print(f"Missing mask, skipped: {mask_path}")

            if image_missing or mask_missing:
                stats_rows.append(
                    {
                        "sample_id": image_id,
                        "split": split_name,
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                        "mask_preprocessing_method": SELECTED_MASK_PREPROCESSING_METHOD,
                        "raw_mask_area_ratio": np.nan,
                        "processed_mask_area_ratio": np.nan,
                        "is_empty_mask": True,
                        "is_very_small_mask": True,
                        "is_suspicious_mask": True,
                        "image_missing": image_missing,
                        "mask_missing": mask_missing,
                    }
                )
                continue

            mask_info = preprocess_mask(raw_mask_gray)
            checked_masks += 1
            suspicious_masks += int(mask_info.is_suspicious_mask)
            stats_rows.append(
                {
                    "sample_id": image_id,
                    "split": split_name,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "mask_preprocessing_method": SELECTED_MASK_PREPROCESSING_METHOD,
                    "raw_mask_area_ratio": mask_info.raw_area_ratio,
                    "processed_mask_area_ratio": mask_info.processed_area_ratio,
                    "is_empty_mask": mask_info.is_empty_mask,
                    "is_very_small_mask": mask_info.is_very_small_mask,
                    "is_suspicious_mask": mask_info.is_suspicious_mask,
                    "image_missing": False,
                    "mask_missing": False,
                }
            )

            if not mask_info.is_suspicious_mask:
                records_by_split[split_name].append(
                    SampleRecord(
                        image_id=image_id,
                        split=split_name,
                        image_path=image_path,
                        mask_path=mask_path,
                    )
                )

            readable_debug_candidates.append(
                {
                    "image_id": image_id,
                    "split": split_name,
                    "image": image,
                    "raw_mask": mask_info.raw_mask,
                    "processed_mask": mask_info.processed_mask,
                }
            )

        sample_count = min(20, len(readable_debug_candidates))
        debug_records.extend(rng.sample(readable_debug_candidates, sample_count))

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(MASK_STATS_CSV, index=False)
    save_debug_pairing_grid(debug_records, DEBUG_PAIRING_FIGURE)

    return MaskAudit(
        stats_df=stats_df,
        records_by_split=records_by_split,
        checked_masks=checked_masks,
        suspicious_masks=suspicious_masks,
        missing_images=missing_images,
        missing_masks=missing_masks,
    )


def assert_mask_audit_is_reliable(audit: MaskAudit) -> None:
    print(f"Forced mask preprocessing method: {SELECTED_MASK_PREPROCESSING_METHOD}")
    print(f"Mask statistics saved to: {rel_path(MASK_STATS_CSV)}")
    print(f"Debug image-mask grid saved to: {rel_path(DEBUG_PAIRING_FIGURE)}")
    print(
        "Mask audit: "
        f"{audit.checked_masks} readable masks, "
        f"{audit.suspicious_masks} suspicious "
        f"({100.0 * audit.suspicious_fraction:.2f}%), "
        f"{audit.missing_images} missing images, {audit.missing_masks} missing masks"
    )

    if audit.checked_masks == 0 or audit.suspicious_fraction > 0.20:
        raise SystemExit(
            "Mask preprocessing appears unreliable. Check image-mask pairing and mask format."
        )

    for split_name in ("train", "val", "test"):
        if not audit.records_by_split.get(split_name):
            raise SystemExit(f"No valid non-suspicious masks are available for {split_name}.")


class HC18SegmentationDataset(Dataset):
    def __init__(
        self,
        records: list[SampleRecord],
        image_size: int,
        augment: bool = False,
    ) -> None:
        if not records:
            raise ValueError("HC18SegmentationDataset received no records.")
        self.records = records
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        image = read_gray_uint8(record.image_path)
        raw_mask = read_gray_uint8(record.mask_path)
        if image is None:
            raise FileNotFoundError(f"Could not read HR image: {record.image_path}")
        if raw_mask is None:
            raise FileNotFoundError(f"Could not read mask: {record.mask_path}")

        processed_mask = preprocess_mask(raw_mask).processed_mask
        image_01 = resize_image_01(image, self.image_size)
        mask_01 = resize_mask_01(processed_mask, self.image_size)

        if self.augment:
            image_01, mask_01 = apply_training_augmentation(image_01, mask_01)

        image_tensor = torch.from_numpy(np.ascontiguousarray(image_01)).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_01)).unsqueeze(0).float()
        return image_tensor, mask_tensor


def apply_training_augmentation(
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    augmented_image = image.copy()
    augmented_mask = mask.copy()

    if random.random() < 0.5:
        augmented_image = np.fliplr(augmented_image)
        augmented_mask = np.fliplr(augmented_mask)

    angle = random.uniform(-10.0, 10.0)
    center = (augmented_image.shape[1] / 2.0, augmented_image.shape[0] / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    size = (augmented_image.shape[1], augmented_image.shape[0])
    augmented_image = cv2.warpAffine(
        augmented_image,
        matrix,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    augmented_mask = cv2.warpAffine(
        augmented_mask,
        matrix,
        size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    alpha = random.uniform(0.90, 1.10)
    beta = random.uniform(-0.05, 0.05)
    augmented_image = np.clip(alpha * augmented_image + beta, 0.0, 1.0)

    noise_sigma = random.uniform(0.0, 0.02)
    if noise_sigma > 0.0:
        augmented_image = np.clip(
            augmented_image + np.random.normal(0.0, noise_sigma, augmented_image.shape),
            0.0,
            1.0,
        )

    return augmented_image.astype(np.float32), (augmented_mask > 0.5).astype(np.float32)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(
                x,
                [
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2,
                ],
            )
        return self.conv(torch.cat([skip, x], dim=1))


class ImprovedUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc1 = DoubleConv(1, 32)
        self.enc2 = DownBlock(32, 64)
        self.enc3 = DownBlock(64, 128)
        self.enc4 = DownBlock(128, 256)
        self.bottleneck = DownBlock(256, 512)
        self.up1 = UpBlock(512, 256, 256)
        self.up2 = UpBlock(256, 128, 128)
        self.up3 = UpBlock(128, 64, 64)
        self.up4 = UpBlock(64, 32, 32)
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.bottleneck(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(x)


class BCEDiceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, targets) + dice_loss(logits, targets)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = torch.sum(probs * targets, dim=dims)
    denominator = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def dice_iou_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    dims = (1, 2, 3)
    intersection = torch.sum(preds * targets, dim=dims)
    pred_area = torch.sum(preds, dim=dims)
    target_area = torch.sum(targets, dim=dims)
    union = pred_area + target_area - intersection
    dice = (2.0 * intersection + eps) / (pred_area + target_area + eps)
    iou = (intersection + eps) / (union + eps)
    return dice.mean(), iou.mean()


def make_loader(
    records: list[SampleRecord],
    image_size: int,
    batch_size: int,
    num_workers: int,
    augment: bool,
    shuffle: bool,
) -> DataLoader:
    dataset = HC18SegmentationDataset(records, image_size=image_size, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_items = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = criterion(logits, masks)
        dice, iou = dice_iou_from_logits(logits, masks, threshold)
        batch_size = int(images.size(0))
        total_loss += float(loss.item()) * batch_size
        total_dice += float(dice.item()) * batch_size
        total_iou += float(iou.item()) * batch_size
        total_items += batch_size

    total_items = max(total_items, 1)
    return total_loss / total_items, total_dice / total_items, total_iou / total_items


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = int(images.size(0))
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)


def save_overfit_figure(
    model: nn.Module,
    records: list[SampleRecord],
    image_size: int,
    device: torch.device,
    threshold: float,
) -> None:
    model.eval()
    selected = records[: min(6, len(records))]
    if not selected:
        return

    fig, axes = plt.subplots(len(selected), 3, figsize=(8.0, 2.4 * len(selected)))
    axes_array = np.atleast_2d(axes)

    for row_idx, record in enumerate(selected):
        image = read_gray_uint8(record.image_path)
        raw_mask = read_gray_uint8(record.mask_path)
        if image is None or raw_mask is None:
            continue
        processed_mask = preprocess_mask(raw_mask).processed_mask
        predicted = predict_mask(
            model=model,
            image=image,
            target_shape=processed_mask.shape,
            image_size=image_size,
            threshold=threshold,
            device=device,
        )
        panels = [
            ("HR", image, "gray"),
            ("GT mask", processed_mask, "gray"),
            ("Pred overlay", overlay_prediction(image, predicted, processed_mask), None),
        ]
        for col_idx, (title, panel, cmap) in enumerate(panels):
            ax = axes_array[row_idx, col_idx]
            ax.imshow(panel, cmap=cmap)
            ax.set_title(f"{title}: {record.image_id}", fontsize=8)
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(OVERFIT_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_overfit_sanity_check(
    train_records: list[SampleRecord],
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    print(
        "Running 20-sample overfit sanity check with "
        f"{SELECTED_MASK_PREPROCESSING_METHOD} masks."
    )
    selected_records = train_records[:]
    random.shuffle(selected_records)
    selected_records = selected_records[:20]
    if len(selected_records) < 20:
        raise SystemExit("Fewer than 20 valid training samples are available for overfit sanity check.")

    model = ImprovedUNet().to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = make_loader(
        selected_records,
        image_size=args.image_size,
        batch_size=min(args.batch_size, 4),
        num_workers=args.num_workers,
        augment=False,
        shuffle=True,
    )
    eval_loader = make_loader(
        selected_records,
        image_size=args.image_size,
        batch_size=min(args.batch_size, 4),
        num_workers=args.num_workers,
        augment=False,
        shuffle=False,
    )

    final_dice = 0.0
    final_iou = 0.0
    best_dice = 0.0
    best_iou = 0.0
    best_epoch = 0
    for epoch in range(1, args.overfit_epochs + 1):
        train_loss = train_one_epoch(model, loader, criterion, optimizer, device)
        _, final_dice, final_iou = evaluate_loader(
            model,
            eval_loader,
            criterion,
            device,
            threshold=0.50,
        )
        if final_dice > best_dice:
            best_dice = final_dice
            best_iou = final_iou
            best_epoch = epoch

        if (
            epoch == 1
            or epoch % 10 == 0
            or best_dice >= OVERFIT_DICE_PASS_THRESHOLD
            or epoch == args.overfit_epochs
        ):
            print(
                f"Overfit epoch {epoch:03d}/{args.overfit_epochs:03d} "
                f"train_loss={train_loss:.5f} dice={final_dice:.5f} "
                f"iou={final_iou:.5f} best_dice={best_dice:.5f}"
            )
        if epoch >= 20 and best_dice >= OVERFIT_DICE_PASS_THRESHOLD:
            break

    save_overfit_figure(
        model,
        selected_records,
        image_size=args.image_size,
        device=device,
        threshold=0.50,
    )
    print(f"Overfit sanity-check figure saved to: {rel_path(OVERFIT_FIGURE)}")

    if best_dice < OVERFIT_DICE_PASS_THRESHOLD:
        raise SystemExit(
            "U-Net cannot overfit 20 samples. There is likely a mask preprocessing, "
            "pairing, or model issue."
        )

    print(
        f"Overfit sanity check passed at epoch {best_epoch}: "
        f"Dice={best_dice:.6f}, IoU={best_iou:.6f}"
    )
    return float(best_dice)


def train_full_model(
    records_by_split: dict[str, list[SampleRecord]],
    args: argparse.Namespace,
    device: torch.device,
) -> TrainingResult:
    print(f"Training target mask preprocessing: {SELECTED_MASK_PREPROCESSING_METHOD}")
    model = ImprovedUNet().to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=8,
    )

    train_loader = make_loader(
        records_by_split["train"],
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=True,
        shuffle=True,
    )
    val_loader = make_loader(
        records_by_split["val"],
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=False,
        shuffle=False,
    )

    history: list[dict[str, float]] = []
    best_val_dice = -1.0
    best_val_iou = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice, val_iou = evaluate_loader(
            model,
            val_loader,
            criterion,
            device,
            threshold=0.50,
        )
        scheduler.step(val_dice)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_val_iou = val_iou
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_val_dice": best_val_dice,
                    "best_val_iou": best_val_iou,
                    "image_size": args.image_size,
                    "threshold": 0.50,
                    "mask_preprocessing_method": SELECTED_MASK_PREPROCESSING_METHOD,
                    "loss": "BCEWithLogitsLoss + DiceLoss",
                },
                MODEL_PATH,
            )

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_dice": float(val_dice),
                "val_iou": float(val_iou),
                "best_val_dice": float(best_val_dice),
            }
        )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            f"val_dice={val_dice:.5f} val_iou={val_iou:.5f} "
            f"best_val_dice={best_val_dice:.5f}"
        )

    plot_training_curve(history)
    return TrainingResult(
        history=history,
        best_val_dice=float(best_val_dice),
        best_val_iou=float(best_val_iou),
        best_epoch=best_epoch,
    )


def load_best_model(device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing checkpoint: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = ImprovedUNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def tune_threshold(
    model: nn.Module,
    val_records: list[SampleRecord],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, pd.DataFrame]:
    criterion = BCEDiceLoss()
    val_loader = make_loader(
        val_records,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=False,
        shuffle=False,
    )
    rows = []
    for threshold in THRESHOLDS:
        _, dice, iou = evaluate_loader(model, val_loader, criterion, device, threshold)
        rows.append({"threshold": threshold, "val_dice": dice, "val_iou": iou})
        print(f"threshold={threshold:.2f} val_dice={dice:.5f} val_iou={iou:.5f}")

    threshold_df = pd.DataFrame(rows)
    best_row = threshold_df.sort_values(
        ["val_dice", "val_iou", "threshold"],
        ascending=[False, False, True],
    ).iloc[0]
    best_threshold = float(best_row["threshold"])

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    checkpoint["threshold"] = best_threshold
    checkpoint["threshold_tuning"] = rows
    torch.save(checkpoint, MODEL_PATH)
    print(f"Best validation threshold: {best_threshold:.2f}")
    return best_threshold, threshold_df


def plot_training_curve(history: list[dict[str, float]]) -> None:
    if not history:
        return
    df = pd.DataFrame(history)
    fig, ax1 = plt.subplots(figsize=(8.5, 5.0))
    ax1.plot(df["epoch"], df["train_loss"], label="train loss", color="#1f77b4")
    ax1.plot(df["epoch"], df["val_loss"], label="val loss", color="#ff7f0e")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["val_dice"], label="val Dice", color="#2ca02c")
    ax2.plot(df["epoch"], df["val_iou"], label="val IoU", color="#d62728")
    ax2.set_ylabel("Score")
    ax2.set_ylim(0.0, 1.0)

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="center right")
    fig.tight_layout()
    fig.savefig(TRAINING_CURVE_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curve saved to: {rel_path(TRAINING_CURVE_FIGURE)}")


def resize_bool_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == target_shape:
        return mask.astype(bool)
    resized = cv2.resize(
        (mask > 0).astype(np.uint8),
        (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized > 0


@torch.no_grad()
def predict_mask(
    model: nn.Module,
    image: np.ndarray,
    target_shape: tuple[int, int],
    image_size: int,
    threshold: float,
    device: torch.device,
) -> np.ndarray:
    image_01 = resize_image_01(image, image_size)
    tensor = torch.from_numpy(image_01).unsqueeze(0).unsqueeze(0).float().to(device)
    logits = model(tensor)
    probs = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
    pred_256 = probs >= threshold
    return resize_bool_mask(pred_256, target_shape)


def extract_boundary(mask: np.ndarray) -> np.ndarray:
    mask_uint8 = (mask > 0).astype(np.uint8)
    if not np.any(mask_uint8):
        return np.zeros(mask_uint8.shape, dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(mask_uint8, kernel, iterations=1)
    boundary = (mask_uint8 - eroded) > 0
    return boundary.astype(bool)


def average_symmetric_boundary_distance(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    pred_boundary = extract_boundary(predicted)
    gt_boundary = extract_boundary(ground_truth)

    if not np.any(pred_boundary) or not np.any(gt_boundary):
        return float("nan")

    distance_to_gt = cv2.distanceTransform(
        (~gt_boundary).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    distance_to_pred = cv2.distanceTransform(
        (~pred_boundary).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    distances = np.concatenate(
        [
            distance_to_gt[pred_boundary].astype(np.float32),
            distance_to_pred[gt_boundary].astype(np.float32),
        ]
    )
    if distances.size == 0:
        return float("nan")
    return float(np.mean(distances))


def equivalent_diameter(area: float) -> float:
    if area <= 0.0:
        return 0.0
    return float(math.sqrt(4.0 * area / math.pi))


def compute_segmentation_metrics(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
) -> dict[str, float]:
    pred = predicted.astype(bool)
    gt = ground_truth.astype(bool)
    pred_area = float(np.sum(pred))
    gt_area = float(np.sum(gt))
    intersection = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())

    dice = 0.0 if pred_area + gt_area <= 0.0 else 2.0 * intersection / (pred_area + gt_area)
    iou = 0.0 if union <= 0.0 else intersection / union
    area_error_percent = (
        float("nan") if gt_area <= 0.0 else 100.0 * abs(pred_area - gt_area) / gt_area
    )
    pred_diameter = equivalent_diameter(pred_area)
    gt_diameter = equivalent_diameter(gt_area)
    diameter_error_px = abs(pred_diameter - gt_diameter)
    diameter_error_percent = (
        float("nan") if gt_diameter <= 0.0 else 100.0 * diameter_error_px / gt_diameter
    )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "boundary_error_px": average_symmetric_boundary_distance(pred, gt),
        "predicted_area_px": pred_area,
        "ground_truth_area_px": gt_area,
        "fetal_head_area_error_percent": float(area_error_percent),
        "predicted_equivalent_diameter_px": pred_diameter,
        "ground_truth_equivalent_diameter_px": gt_diameter,
        "equivalent_diameter_error_px": float(diameter_error_px),
        "equivalent_diameter_error_percent": float(diameter_error_percent),
    }


def load_lq_manifest() -> dict[str, pd.Series]:
    if not LQ_MANIFEST_CSV.exists():
        warn(f"LQ manifest is missing: {LQ_MANIFEST_CSV}")
        return {}
    manifest = pd.read_csv(LQ_MANIFEST_CSV)
    if "image_id" not in manifest.columns:
        warn(f"LQ manifest has no image_id column: {LQ_MANIFEST_CSV}")
        return {}
    return {str(row["image_id"]): row for _, row in manifest.iterrows()}


def resolve_eval_input_path(
    method: dict[str, Any],
    record: SampleRecord,
    lq_manifest: dict[str, pd.Series],
) -> Path:
    if method["kind"] == "hr":
        return record.image_path
    if method["kind"] == "lq_mixed":
        manifest_row = lq_manifest.get(record.image_id)
        if manifest_row is not None:
            csv_path = resolve_csv_path(manifest_row.get("lq_mixed_path"))
            if csv_path is not None:
                return csv_path
        return LQ_MIXED_DIR / f"{record.image_id}.png"
    return Path(method["output_dir"]) / f"{record.image_id}.png"


def evaluate_methods(
    model: nn.Module,
    records: list[SampleRecord],
    methods: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    threshold: float,
    reliability_note: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    lq_manifest = load_lq_manifest()
    rows: list[dict[str, Any]] = []
    example_records: list[dict[str, Any]] = []
    missing_counts = {method["label"]: 0 for method in methods}
    unreadable_counts = {method["label"]: 0 for method in methods}

    for record in records:
        raw_gt_mask = read_gray_uint8(record.mask_path)
        if raw_gt_mask is None:
            warn(f"Skipping {record.image_id}: unreadable mask {record.mask_path}")
            continue
        gt_mask = preprocess_mask(raw_gt_mask).processed_mask.astype(bool)
        if not np.any(gt_mask):
            warn(f"Skipping {record.image_id}: empty processed mask {record.mask_path}")
            continue

        for method in methods:
            input_path = resolve_eval_input_path(method, record, lq_manifest)
            if not input_path.exists():
                missing_counts[method["label"]] += 1
                continue

            image = read_gray_uint8(input_path)
            if image is None:
                unreadable_counts[method["label"]] += 1
                continue

            predicted = predict_mask(
                model=model,
                image=image,
                target_shape=gt_mask.shape,
                image_size=args.image_size,
                threshold=threshold,
                device=device,
            )
            metrics = compute_segmentation_metrics(predicted, gt_mask)
            rows.append(
                {
                    "image_id": record.image_id,
                    "method": method["label"],
                    "input_path": rel_path(input_path),
                    "mask_path": rel_path(record.mask_path),
                    **metrics,
                    "reliability_note": reliability_note,
                }
            )

            if len(example_records) < 12 and len({r["image_id"] for r in example_records}) < 3:
                example_records.append(
                    {
                        "image_id": record.image_id,
                        "method": method["label"],
                        "image": image,
                        "predicted_mask": predicted,
                        "ground_truth_mask": gt_mask,
                    }
                )

    for method in methods:
        label = method["label"]
        if missing_counts[label]:
            warn(f"{label}: skipped {missing_counts[label]} missing inputs.")
        if unreadable_counts[label]:
            warn(f"{label}: skipped {unreadable_counts[label]} unreadable inputs.")

    return pd.DataFrame(rows, columns=PER_IMAGE_COLUMNS), example_records


def available_sr_methods() -> list[dict[str, Any]]:
    methods = []
    for method in EVAL_METHODS:
        output_dir = Path(method["output_dir"])
        if method["kind"] in {"hr", "lq_mixed"} or output_dir.exists():
            methods.append(method)
        else:
            warn(f"Skipping {method['label']} because folder is missing: {output_dir}")
    return methods


def summarize_metrics(per_image_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    metric_columns = [
        "dice",
        "iou",
        "boundary_error_px",
        "fetal_head_area_error_percent",
        "equivalent_diameter_error_px",
        "equivalent_diameter_error_percent",
    ]

    for method in EVAL_METHODS:
        label = method["label"]
        method_df = per_image_df[per_image_df["method"] == label]
        if method_df.empty:
            continue
        row: dict[str, Any] = {
            "method": label,
            "n_images": int(len(method_df)),
            "reliability_note": str(method_df["reliability_note"].iloc[0]),
        }
        for metric in metric_columns:
            values = method_df[metric].astype(float)
            row[f"mean_{metric}"] = float(values.mean())
            row[f"std_{metric}"] = float(values.std(ddof=0))
        summary_rows.append(row)

    return pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)


def overlay_prediction(
    image: np.ndarray,
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
) -> np.ndarray:
    if image.shape != ground_truth_mask.shape:
        image = cv2.resize(
            image,
            (ground_truth_mask.shape[1], ground_truth_mask.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    image_01 = normalize_panel(image).astype(np.float32) / 255.0
    rgb = np.repeat(image_01[..., None], 3, axis=2)
    gt_boundary = extract_boundary(ground_truth_mask)
    pred_boundary = extract_boundary(predicted_mask)
    rgb[ground_truth_mask.astype(bool), 1] = np.maximum(rgb[ground_truth_mask.astype(bool), 1], 0.45)
    rgb[gt_boundary] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    rgb[pred_boundary] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return np.clip(rgb, 0.0, 1.0)


def plot_example_outputs(example_records: list[dict[str, Any]]) -> None:
    if not example_records:
        warn("No example segmentation records were available for plotting.")
        return

    n_records = len(example_records)
    n_cols = min(4, n_records)
    n_rows = int(math.ceil(n_records / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.8 * n_rows))
    axes_array = np.atleast_1d(axes).ravel()

    for axis, record in zip(axes_array, example_records):
        overlay = overlay_prediction(
            record["image"],
            record["predicted_mask"],
            record["ground_truth_mask"],
        )
        axis.imshow(overlay)
        axis.set_title(f"{record['image_id']}\n{record['method']}", fontsize=8)
        axis.axis("off")

    for axis in axes_array[n_records:]:
        axis.axis("off")

    fig.suptitle("Improved U-Net examples (green=GT boundary, red=prediction boundary)")
    fig.tight_layout()
    fig.savefig(EXAMPLE_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dice_iou(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        return
    labels = summary_df["method"].tolist()
    x_values = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12.0, 5.5))
    ax.bar(x_values - width / 2, summary_df["mean_dice"], width, label="Dice", color="#1f77b4")
    ax.bar(x_values + width / 2, summary_df["mean_iou"], width, label="IoU", color="#ff7f0e")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Improved HC18 U-Net Segmentation Dice and IoU")
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(DICE_IOU_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_measurement_error(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        return
    labels = summary_df["method"].tolist()
    x_values = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12.0, 5.5))
    ax.bar(
        x_values - width / 2,
        summary_df["mean_fetal_head_area_error_percent"],
        width,
        label="Area error (%)",
        color="#2ca02c",
    )
    ax.bar(
        x_values + width / 2,
        summary_df["mean_equivalent_diameter_error_percent"],
        width,
        label="Equivalent diameter error (%)",
        color="#d62728",
    )
    ax.set_ylabel("Absolute error (%)")
    ax.set_title("Improved HC18 U-Net Fetal Measurement Error")
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(MEASUREMENT_ERROR_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(fig)


def format_metric(value: Any, digits: int = 4) -> str:
    if pd.isna(value):
        return "nan"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows available._"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, separator]
    for _, row in df[columns].iterrows():
        values = [format_metric(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown_report(
    audit: MaskAudit,
    overfit_dice: float,
    training_result: TrainingResult,
    threshold_df: pd.DataFrame,
    best_threshold: float,
    hr_test_dice: float,
    summary_df: pd.DataFrame,
    args: argparse.Namespace,
    supplementary_only: bool,
) -> None:
    summary_columns = [
        "method",
        "n_images",
        "mean_dice",
        "mean_iou",
        "mean_boundary_error_px",
        "mean_fetal_head_area_error_percent",
        "mean_equivalent_diameter_error_percent",
    ]
    threshold_table = markdown_table(threshold_df, ["threshold", "val_dice", "val_iou"])
    method_table = markdown_table(summary_df, summary_columns)

    warning_sections = []
    if training_result.best_val_dice < 0.60:
        warning_sections.append(
            f"""## Warning: Weak Segmentation

- "{WEAK_SEGMENTATION_WARNING}"
- "Main anatomy-preservation evidence should rely on mask-based MAE, boundary error, contrast preservation, and edge/boundary similarity."
- "Segmentation-based Dice/IoU should be treated as supplementary or future work."
"""
        )
    if supplementary_only:
        warning_sections.append(
            f"""## Reliability Gate

WARNING: Segmentation model is not reliable enough for final Dice/IoU reporting.

The saved segmentation-based outputs are supplementary only because HR test Dice is below {HR_TEST_DICE_PASS_THRESHOLD:.2f}.
"""
        )

    report = f"""# Improved HC18 U-Net Segmentation Metrics

{SEGMENTATION_NOTE}

## Mask Preprocessing Summary

- Checked readable masks: {audit.checked_masks}
- Suspicious masks: {audit.suspicious_masks} ({100.0 * audit.suspicious_fraction:.2f}%)
- Missing HR images: {audit.missing_images}
- Missing masks: {audit.missing_masks}
- Selected preprocessing method: `{SELECTED_MASK_PREPROCESSING_METHOD}`
- Mask statistics CSV: `{rel_path(MASK_STATS_CSV)}`
- Image-mask pairing figure: `{rel_path(DEBUG_PAIRING_FIGURE)}`

Masks were converted to grayscale, binarized with `mask > 0`, fitted to the largest external-contour ellipse, filled, hole-filled, and resized with nearest-neighbor interpolation.

## Overfit Sanity Check

- Samples: 20 training images
- Overfit Dice: {overfit_dice:.4f}
- Figure: `{rel_path(OVERFIT_FIGURE)}`

## Training Details

- Architecture: U-Net with encoder channels 32, 64, 128, 256 and bottleneck 512
- Epochs: {args.epochs}
- Batch size: {args.batch_size}
- Optimizer: AdamW, lr={args.lr}, weight_decay={args.weight_decay}
- Loss: `BCEWithLogitsLoss + DiceLoss`
- Target mask preprocessing: `{SELECTED_MASK_PREPROCESSING_METHOD}`
- Augmentation: horizontal flip, small rotation, brightness/contrast shift, Gaussian noise
- Best validation Dice: {training_result.best_val_dice:.4f}
- Best validation IoU: {training_result.best_val_iou:.4f}
- Best epoch: {training_result.best_epoch}
- Best threshold: {best_threshold:.2f}
- Training curve: `{rel_path(TRAINING_CURVE_FIGURE)}`

## Threshold Tuning

{threshold_table}

## HR Test Reliability

- HR test Dice: {hr_test_dice:.4f}

## Final Method Comparison

{method_table}

## Outputs

- Checkpoint: `{rel_path(MODEL_PATH)}`
- Per-image metrics: `{rel_path(PER_IMAGE_CSV)}`
- Summary metrics: `{rel_path(SUMMARY_CSV)}`
- Dice/IoU figure: `{rel_path(DICE_IOU_FIGURE)}`
- Measurement error figure: `{rel_path(MEASUREMENT_ERROR_FIGURE)}`
- Example segmentation figure: `{rel_path(EXAMPLE_FIGURE)}`

{''.join(warning_sections)}
"""
    TABLE_MD.write_text(report, encoding="utf-8")


def write_outputs(
    per_image_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    example_records: list[dict[str, Any]],
) -> None:
    per_image_df.to_csv(PER_IMAGE_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    plot_dice_iou(summary_df)
    plot_measurement_error(summary_df)
    plot_example_outputs(example_records)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    ensure_dirs([MODEL_PATH.parent, FIGURE_DIR, PER_IMAGE_CSV.parent])

    print("Improved HC18 U-Net segmentation pipeline")
    print(f"Project root: {PROJECT_ROOT}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    split_frames = {
        "train": read_split_csv(TRAIN_CSV, "train"),
        "val": read_split_csv(VAL_CSV, "val"),
        "test": read_split_csv(TEST_CSV, "test"),
    }
    audit = audit_masks(split_frames, seed=args.seed)
    assert_mask_audit_is_reliable(audit)

    overfit_dice = run_overfit_sanity_check(audit.records_by_split["train"], args, device)
    print(f"Overfit sanity-check Dice: {overfit_dice:.4f}")

    training_result = train_full_model(audit.records_by_split, args, device)
    model, checkpoint = load_best_model(device)
    print(f"Loaded best checkpoint from epoch {checkpoint.get('epoch')}: {rel_path(MODEL_PATH)}")

    best_threshold, threshold_df = tune_threshold(
        model,
        audit.records_by_split["val"],
        args,
        device,
    )

    hr_methods = [EVAL_METHODS[0]]
    hr_per_image_df, hr_examples = evaluate_methods(
        model,
        audit.records_by_split["test"],
        hr_methods,
        args,
        device,
        threshold=best_threshold,
        reliability_note="HR test reliability gate",
    )
    if hr_per_image_df.empty:
        raise RuntimeError("No HR test images were evaluated.")

    hr_test_dice = float(hr_per_image_df["dice"].astype(float).mean())
    supplementary_only = hr_test_dice < HR_TEST_DICE_PASS_THRESHOLD
    if supplementary_only:
        warn(
            "Segmentation model is not reliable enough for restored SR evaluation "
            f"(HR test Dice {hr_test_dice:.4f} < {HR_TEST_DICE_PASS_THRESHOLD:.2f})."
        )
        per_image_df = hr_per_image_df.copy()
        per_image_df["reliability_note"] = "supplementary only: HR test Dice below 0.60"
        example_records = hr_examples
    else:
        methods = available_sr_methods()
        per_image_df, example_records = evaluate_methods(
            model,
            audit.records_by_split["test"],
            methods,
            args,
            device,
            threshold=best_threshold,
            reliability_note="supporting anatomy-preservation evidence",
        )

    summary_df = summarize_metrics(per_image_df)
    write_outputs(per_image_df, summary_df, example_records)
    write_markdown_report(
        audit=audit,
        overfit_dice=overfit_dice,
        training_result=training_result,
        threshold_df=threshold_df,
        best_threshold=best_threshold,
        hr_test_dice=hr_test_dice,
        summary_df=summary_df,
        args=args,
        supplementary_only=supplementary_only,
    )

    print(f"Checkpoint: {rel_path(MODEL_PATH)}")
    print(f"Per-image metrics: {rel_path(PER_IMAGE_CSV)}")
    print(f"Summary metrics: {rel_path(SUMMARY_CSV)}")
    print(f"Markdown report: {rel_path(TABLE_MD)}")
    print(f"Figures: {rel_path(FIGURE_DIR)}")
    print(SEGMENTATION_NOTE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
