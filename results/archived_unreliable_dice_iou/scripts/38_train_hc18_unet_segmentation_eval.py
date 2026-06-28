"""
Train an HC18 fetal-head U-Net segmenter and evaluate restored SR outputs.

This script trains a lightweight U-Net on HC18 HR images and masks, then uses
the trained model to compute segmentation-based anatomy metrics for HR, LQ, and
available SR outputs. It does not retrain any SR model.
"""

from __future__ import annotations

import argparse
import random
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
        f"{missing_package}. Install pandas, numpy, opencv-python, "
        "matplotlib, and torch in the project Python environment, then rerun "
        "this script."
    ) from exc


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HC18_ROOT = PROJECT_ROOT / "data" / "processed" / "HC18_SR"
TRAIN_CSV = HC18_ROOT / "train.csv"
VAL_CSV = HC18_ROOT / "val.csv"
TEST_CSV = HC18_ROOT / "test.csv"
HR_DIR = HC18_ROOT / "HR"
MASK_DIR = HC18_ROOT / "masks"
LQ_MIXED_DIR = HC18_ROOT / "LQ_mixed"
SR_OUTPUT_ROOT = PROJECT_ROOT / "results" / "sr_outputs"

MODEL_PATH = PROJECT_ROOT / "results" / "models" / "hc18_unet_segmentation_best.pth"
PER_IMAGE_CSV = PROJECT_ROOT / "results" / "hc18_unet_segmentation_metrics_per_image.csv"
SUMMARY_CSV = PROJECT_ROOT / "results" / "hc18_unet_segmentation_metrics_summary.csv"
TABLE_MD = PROJECT_ROOT / "results" / "hc18_unet_segmentation_metrics_table.md"
FIGURE_DIR = PROJECT_ROOT / "results" / "hc18_unet_segmentation_figures"
DICE_IOU_FIGURE = FIGURE_DIR / "dice_iou_comparison.png"
MEASUREMENT_ERROR_FIGURE = FIGURE_DIR / "measurement_error_comparison.png"
EXAMPLE_SEGMENTATION_FIGURE = FIGURE_DIR / "example_segmentation_outputs.png"

SEGMENTATION_NOTE = (
    "Segmentation-based Dice, IoU, and fetal measurement error are computed "
    "using an HC18-trained U-Net segmentation model and HC18 fetal-head masks. "
    "These metrics provide anatomy-preservation evidence, but they are not "
    "expert clinical validation."
)

EVAL_METHODS = [
    {
        "key": "hr",
        "label": "HR ground-truth images",
        "output_dir": HR_DIR,
        "kind": "hr",
    },
    {
        "key": "lq_mixed",
        "label": "LQ mixed baseline",
        "output_dir": LQ_MIXED_DIR,
        "kind": "lq_mixed",
    },
    {
        "key": "sr_only",
        "label": "SR-only CNN",
        "output_dir": SR_OUTPUT_ROOT / "sr_only",
        "kind": "sr_output",
    },
    {
        "key": "quality_aware",
        "label": "Quality-aware CNN",
        "output_dir": SR_OUTPUT_ROOT / "quality_aware",
        "kind": "sr_output",
    },
    {
        "key": "quality_aware_l1_mse",
        "label": "Quality-aware CNN + L1/MSE",
        "output_dir": SR_OUTPUT_ROOT / "quality_aware_l1_mse",
        "kind": "sr_output",
    },
    {
        "key": "diffusion_sr",
        "label": "Conditional Diffusion SR",
        "output_dir": SR_OUTPUT_ROOT / "diffusion_sr",
        "kind": "sr_output",
    },
    {
        "key": "quality_aware_diffusion_sr_v2",
        "label": "Quality-aware Diffusion SR v2",
        "output_dir": SR_OUTPUT_ROOT / "quality_aware_diffusion_sr_v2",
        "kind": "sr_output",
    },
    {
        "key": "cnn_residual_diffusion_sr",
        "label": "CNN-Residual Diffusion SR",
        "output_dir": SR_OUTPUT_ROOT / "cnn_residual_diffusion_sr",
        "kind": "sr_output",
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
]

SUMMARY_MARKDOWN_COLUMNS = [
    "method",
    "n_images",
    "mean_dice",
    "mean_iou",
    "mean_boundary_error_px",
    "mean_fetal_head_area_error_percent",
    "mean_equivalent_diameter_error_percent",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an HC18 U-Net fetal-head segmenter and evaluate restored "
            "SR outputs with Dice, IoU, boundary, and measurement metrics."
        )
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--image-size", type=int, default=256, help="Training image size.")
    parser.add_argument(
        "--base-channels",
        type=int,
        default=16,
        help="Base channel count for the lightweight U-Net.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for binarizing U-Net outputs.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Keep 0 for Windows PowerShell compatibility.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Use the saved best model and run evaluation only.",
    )
    return parser.parse_args()


def rel_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def warn(message: str) -> None:
    print(f"WARNING: {message}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_csv_path(value: Any, base_dir: Path) -> Path | None:
    if pd.isna(value) or not str(value).strip():
        return None
    path = Path(str(value))
    return path if path.is_absolute() else base_dir / path


def resolve_hr_path(row: pd.Series, image_id: str) -> Path:
    csv_path = resolve_csv_path(row.get("hr_path"), HC18_ROOT)
    if csv_path is not None:
        return csv_path
    return HR_DIR / f"{image_id}.png"


def resolve_mask_path(row: pd.Series, image_id: str) -> Path:
    folder_mask = MASK_DIR / f"{image_id}.png"
    if folder_mask.exists():
        return folder_mask
    csv_path = resolve_csv_path(row.get("mask_path"), HC18_ROOT)
    if csv_path is not None:
        return csv_path
    return folder_mask


def read_grayscale_uint8(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return np.squeeze(image).astype(np.uint8)


def read_mask_bool(path: Path) -> np.ndarray | None:
    mask = read_grayscale_uint8(path)
    if mask is None:
        return None
    mask_bool = mask > 127
    return mask_bool if np.any(mask_bool) else None


def resize_image_01(image: np.ndarray, image_size: int) -> np.ndarray:
    resized = cv2.resize(
        image,
        (image_size, image_size),
        interpolation=cv2.INTER_LINEAR,
    )
    return resized.astype(np.float32) / 255.0


def resize_mask_01(mask: np.ndarray, image_size: int) -> np.ndarray:
    resized = cv2.resize(
        mask.astype(np.uint8) * 255,
        (image_size, image_size),
        interpolation=cv2.INTER_NEAREST,
    )
    return (resized > 127).astype(np.float32)


class HC18SegmentationDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        image_size: int,
        augment: bool = False,
    ) -> None:
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing split CSV: {csv_path}")

        df = pd.read_csv(csv_path)
        if "has_mask" in df.columns:
            df = df[df["has_mask"].astype(int) == 1].copy()
        self.rows = df.reset_index(drop=True)
        self.image_size = image_size
        self.augment = augment
        self.csv_path = csv_path

        if self.rows.empty:
            raise ValueError(f"No mask-backed rows found in {csv_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows.iloc[index]
        image_id = str(row["image_id"])
        image_path = resolve_hr_path(row, image_id)
        mask_path = resolve_mask_path(row, image_id)

        image = read_grayscale_uint8(image_path)
        mask = read_mask_bool(mask_path)
        if image is None:
            raise FileNotFoundError(f"Could not read HR image: {image_path}")
        if mask is None:
            raise FileNotFoundError(f"Could not read fetal-head mask: {mask_path}")

        image_01 = resize_image_01(image, self.image_size)
        mask_01 = resize_mask_01(mask, self.image_size)

        if self.augment and random.random() < 0.5:
            image_01 = np.ascontiguousarray(np.fliplr(image_01))
            mask_01 = np.ascontiguousarray(np.fliplr(mask_01))

        image_tensor = torch.from_numpy(image_01).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask_01).unsqueeze(0).float()
        return image_tensor, mask_tensor


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


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
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


class SmallUNet(nn.Module):
    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        channels = base_channels
        self.inc = DoubleConv(1, channels)
        self.down1 = Down(channels, channels * 2)
        self.down2 = Down(channels * 2, channels * 4)
        self.down3 = Down(channels * 4, channels * 8)
        self.down4 = Down(channels * 8, channels * 16)
        self.up1 = Up(channels * 16, channels * 8, channels * 8)
        self.up2 = Up(channels * 8, channels * 4, channels * 4)
        self.up3 = Up(channels * 4, channels * 2, channels * 2)
        self.up4 = Up(channels * 2, channels, channels)
        self.outc = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def dice_loss_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = torch.sum(probs * targets, dim=dims)
    denominator = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    return bce + dice_loss_from_logits(logits, targets)


def dice_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    dims = (1, 2, 3)
    intersection = torch.sum(preds * targets, dim=dims)
    denominator = torch.sum(preds, dim=dims) + torch.sum(targets, dim=dims)
    return ((2.0 * intersection + eps) / (denominator + eps)).mean()


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    train_dataset = HC18SegmentationDataset(
        TRAIN_CSV,
        image_size=args.image_size,
        augment=True,
    )
    val_dataset = HC18SegmentationDataset(
        VAL_CSV,
        image_size=args.image_size,
        augment=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


@torch.no_grad()
def validate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_items = 0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = bce_dice_loss(logits, masks)
        batch_size = int(images.size(0))
        total_loss += float(loss.item()) * batch_size
        total_dice += float(dice_from_logits(logits, masks, threshold).item()) * batch_size
        total_items += batch_size
    return total_loss / max(total_items, 1), total_dice / max(total_items, 1)


def train_segmentation_model(args: argparse.Namespace, device: torch.device) -> None:
    ensure_dirs([MODEL_PATH.parent])
    train_loader, val_loader = build_loaders(args)
    model = SmallUNet(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_dice = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = bce_dice_loss(logits, masks)
            loss.backward()
            optimizer.step()

            batch_size = int(images.size(0))
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size

        train_loss = total_loss / max(total_items, 1)
        val_loss, val_dice = validate_model(
            model,
            val_loader,
            device,
            threshold=args.threshold,
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            f"val_dice={val_dice:.5f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_dice": val_dice,
                    "image_size": args.image_size,
                    "base_channels": args.base_channels,
                    "threshold": args.threshold,
                },
                MODEL_PATH,
            )
            print(f"Saved best U-Net checkpoint to {rel_path(MODEL_PATH)}")


def load_trained_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing trained U-Net checkpoint: {MODEL_PATH}. "
            "Run without --skip-train to train it first."
        )

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    base_channels = int(checkpoint.get("base_channels", args.base_channels))
    model = SmallUNet(base_channels=base_channels).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def resolve_eval_input_path(
    method: dict[str, Any],
    row: pd.Series,
    image_id: str,
) -> Path:
    if method["kind"] == "hr":
        return resolve_hr_path(row, image_id)
    if method["kind"] == "lq_mixed":
        csv_path = resolve_csv_path(row.get("lq_mixed_path"), HC18_ROOT)
        if csv_path is not None:
            return csv_path
    return Path(method["output_dir"]) / f"{image_id}.png"


def available_eval_methods() -> list[dict[str, Any]]:
    available = []
    for method in EVAL_METHODS:
        output_dir = Path(method["output_dir"])
        if output_dir.exists():
            available.append(method)
        else:
            warn(f"Skipping {method['label']} because folder is missing: {output_dir}")
    return available


def resize_bool_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == target_shape:
        return mask.astype(bool)
    resized = cv2.resize(
        mask.astype(np.uint8) * 255,
        (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized > 127


@torch.no_grad()
def predict_unet_mask(
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


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    mask_uint8 = (mask > 0).astype(np.uint8)
    if not np.any(mask_uint8):
        return np.zeros(mask_uint8.shape, dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(mask_uint8, kernel, iterations=1)
    return (mask_uint8 - eroded) > 0


def symmetric_boundary_error(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    pred_boundary = mask_boundary(predicted)
    gt_boundary = mask_boundary(ground_truth)

    if not np.any(pred_boundary) and not np.any(gt_boundary):
        return 0.0
    if not np.any(pred_boundary) or not np.any(gt_boundary):
        return float(np.hypot(*ground_truth.shape))

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
    return float(np.mean(distances))


def equivalent_diameter(area: float) -> float:
    if area <= 0.0:
        return 0.0
    return float(np.sqrt(4.0 * area / np.pi))


def compute_segmentation_metrics(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
) -> dict[str, float]:
    predicted_bool = predicted.astype(bool)
    gt_bool = ground_truth.astype(bool)

    predicted_area = float(np.sum(predicted_bool))
    gt_area = float(np.sum(gt_bool))
    intersection = float(np.logical_and(predicted_bool, gt_bool).sum())
    union = float(np.logical_or(predicted_bool, gt_bool).sum())

    dice_denominator = predicted_area + gt_area
    dice = 0.0 if dice_denominator <= 0.0 else 2.0 * intersection / dice_denominator
    iou = 0.0 if union <= 0.0 else intersection / union
    area_error_percent = (
        np.nan if gt_area <= 0.0 else 100.0 * abs(predicted_area - gt_area) / gt_area
    )

    predicted_diameter = equivalent_diameter(predicted_area)
    gt_diameter = equivalent_diameter(gt_area)
    diameter_error_px = abs(predicted_diameter - gt_diameter)
    diameter_error_percent = (
        np.nan
        if gt_diameter <= 0.0
        else 100.0 * diameter_error_px / gt_diameter
    )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "boundary_error_px": symmetric_boundary_error(predicted_bool, gt_bool),
        "predicted_area_px": predicted_area,
        "ground_truth_area_px": gt_area,
        "fetal_head_area_error_percent": float(area_error_percent),
        "predicted_equivalent_diameter_px": predicted_diameter,
        "ground_truth_equivalent_diameter_px": gt_diameter,
        "equivalent_diameter_error_px": diameter_error_px,
        "equivalent_diameter_error_percent": float(diameter_error_percent),
    }


def evaluate_restored_outputs(
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Missing test CSV: {TEST_CSV}")

    test_df = pd.read_csv(TEST_CSV)
    if "has_mask" in test_df.columns:
        test_df = test_df[test_df["has_mask"].astype(int) == 1].copy()

    methods = available_eval_methods()
    rows: list[dict[str, Any]] = []
    example_records: list[dict[str, Any]] = []
    missing_counts = {method["label"]: 0 for method in methods}
    unreadable_counts = {method["label"]: 0 for method in methods}

    for _, row in test_df.iterrows():
        image_id = str(row["image_id"])
        mask_path = resolve_mask_path(row, image_id)
        gt_mask = read_mask_bool(mask_path)
        if gt_mask is None:
            warn(f"Skipping {image_id}: missing or unreadable mask {mask_path}")
            continue

        for method in methods:
            input_path = resolve_eval_input_path(method, row, image_id)
            if not input_path.exists():
                missing_counts[method["label"]] += 1
                continue

            image = read_grayscale_uint8(input_path)
            if image is None:
                unreadable_counts[method["label"]] += 1
                continue

            predicted_mask = predict_unet_mask(
                model=model,
                image=image,
                target_shape=gt_mask.shape,
                image_size=args.image_size,
                threshold=args.threshold,
                device=device,
            )
            metrics = compute_segmentation_metrics(predicted_mask, gt_mask)
            rows.append(
                {
                    "image_id": image_id,
                    "method": method["label"],
                    "input_path": rel_path(input_path),
                    "mask_path": rel_path(mask_path),
                    **metrics,
                }
            )

            if len(example_records) < len(methods) and image_id == str(test_df.iloc[0]["image_id"]):
                example_records.append(
                    {
                        "image_id": image_id,
                        "method": method["label"],
                        "image": image,
                        "predicted_mask": predicted_mask,
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


def summarize_metrics(per_image_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in EVAL_METHODS:
        label = method["label"]
        method_df = per_image_df[per_image_df["method"] == label]
        if method_df.empty:
            continue

        row: dict[str, Any] = {"method": label, "n_images": int(len(method_df))}
        metric_columns = [
            "dice",
            "iou",
            "boundary_error_px",
            "fetal_head_area_error_percent",
            "equivalent_diameter_error_px",
            "equivalent_diameter_error_percent",
        ]
        for metric in metric_columns:
            values = method_df[metric].astype(float)
            row[f"mean_{metric}"] = float(values.mean())
            row[f"std_{metric}"] = float(values.std(ddof=0))
        rows.append(row)

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def format_metric(value: Any, digits: int = 4) -> str:
    if pd.isna(value):
        return "nan"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, separator]
    for _, row in df[columns].iterrows():
        values = [format_metric(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def source_status_table() -> str:
    sources = [TRAIN_CSV, VAL_CSV, TEST_CSV, HR_DIR, MASK_DIR, SR_OUTPUT_ROOT]
    lines = ["| Source | Status |", "| --- | --- |"]
    for source in sources:
        status = "available" if source.exists() else "missing"
        lines.append(f"| `{rel_path(source)}` | {status} |")
    return "\n".join(lines)


def plot_dice_iou(summary_df: pd.DataFrame) -> None:
    labels = summary_df["method"].tolist()
    x_values = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(x_values - width / 2, summary_df["mean_dice"], width, label="Dice")
    ax.bar(x_values + width / 2, summary_df["mean_iou"], width, label="IoU")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("HC18 U-Net Segmentation Dice and IoU")
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(DICE_IOU_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_measurement_error(summary_df: pd.DataFrame) -> None:
    labels = summary_df["method"].tolist()
    x_values = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(
        x_values - width / 2,
        summary_df["mean_fetal_head_area_error_percent"],
        width,
        label="Area error (%)",
    )
    ax.bar(
        x_values + width / 2,
        summary_df["mean_equivalent_diameter_error_percent"],
        width,
        label="Equivalent diameter error (%)",
    )
    ax.set_ylabel("Absolute error (%)")
    ax.set_title("HC18 U-Net Fetal Head Measurement Error")
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(MEASUREMENT_ERROR_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(fig)


def overlay_masks(
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
    image_01 = image.astype(np.float32) / 255.0
    rgb = np.repeat(image_01[..., None], 3, axis=2)
    gt_boundary = mask_boundary(ground_truth_mask)
    pred_boundary = mask_boundary(predicted_mask)
    rgb[gt_boundary, 1] = 1.0
    rgb[gt_boundary, 0] = np.minimum(rgb[gt_boundary, 0], 0.25)
    rgb[pred_boundary, 0] = 1.0
    rgb[pred_boundary, 1] = np.minimum(rgb[pred_boundary, 1], 0.25)
    rgb[predicted_mask, 0] = np.maximum(rgb[predicted_mask, 0], 0.55)
    return np.clip(rgb, 0.0, 1.0)


def plot_example_segmentation_outputs(example_records: list[dict[str, Any]]) -> None:
    if not example_records:
        warn("No example segmentation records were available for the example figure.")
        return

    n_records = len(example_records)
    n_cols = min(4, n_records)
    n_rows = int(np.ceil(n_records / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.0 * n_rows))
    axes_array = np.atleast_1d(axes).ravel()

    for axis, record in zip(axes_array, example_records):
        overlay = overlay_masks(
            record["image"],
            record["predicted_mask"],
            record["ground_truth_mask"],
        )
        axis.imshow(overlay)
        axis.set_title(record["method"], fontsize=9)
        axis.axis("off")

    for axis in axes_array[n_records:]:
        axis.axis("off")

    image_id = example_records[0]["image_id"]
    fig.suptitle(
        f"Example HC18 U-Net segmentation: {image_id} "
        "(green=mask boundary, red=prediction boundary)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(EXAMPLE_SEGMENTATION_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_markdown(summary_df: pd.DataFrame, args: argparse.Namespace) -> None:
    report = f"""# HC18 U-Net Segmentation Metrics

{SEGMENTATION_NOTE}

## Model

- Architecture: lightweight U-Net
- Training inputs: HC18 HR images
- Training masks: HC18 fetal-head masks
- Resize size: {args.image_size} x {args.image_size}
- Epochs: {args.epochs}
- Loss: BCE + Dice loss
- Best checkpoint: `{rel_path(MODEL_PATH)}`

## Source Files

{source_status_table()}

## Summary Metrics

{markdown_table(summary_df, SUMMARY_MARKDOWN_COLUMNS)}

## Outputs

- Per-image metrics: `{rel_path(PER_IMAGE_CSV)}`
- Summary metrics: `{rel_path(SUMMARY_CSV)}`
- Dice/IoU figure: `{rel_path(DICE_IOU_FIGURE)}`
- Measurement error figure: `{rel_path(MEASUREMENT_ERROR_FIGURE)}`
- Example segmentation figure: `{rel_path(EXAMPLE_SEGMENTATION_FIGURE)}`

## Method Note

Final metrics use the trained U-Net probability map binarized at threshold
{args.threshold:.2f}; no image-threshold proxy segmentation is used for the
final Dice, IoU, boundary, or measurement metrics.
"""
    TABLE_MD.write_text(report, encoding="utf-8")


def write_outputs(
    per_image_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    example_records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    ensure_dirs([PER_IMAGE_CSV.parent, FIGURE_DIR])
    per_image_df.to_csv(PER_IMAGE_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    plot_dice_iou(summary_df)
    plot_measurement_error(summary_df)
    plot_example_segmentation_outputs(example_records)
    write_markdown(summary_df, args)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    ensure_dirs([MODEL_PATH.parent, FIGURE_DIR])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.skip_train:
        print(f"Skipping training and loading {rel_path(MODEL_PATH)}")
    else:
        train_segmentation_model(args, device)

    model = load_trained_model(args, device)
    per_image_df, example_records = evaluate_restored_outputs(model, args, device)
    if per_image_df.empty:
        raise RuntimeError("No valid images were evaluated.")

    summary_df = summarize_metrics(per_image_df)
    write_outputs(per_image_df, summary_df, example_records, args)

    print(f"Per-image metrics: {rel_path(PER_IMAGE_CSV)}")
    print(f"Summary metrics: {rel_path(SUMMARY_CSV)}")
    print(f"Markdown table: {rel_path(TABLE_MD)}")
    print(f"Figures: {rel_path(FIGURE_DIR)}")
    print(SEGMENTATION_NOTE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
