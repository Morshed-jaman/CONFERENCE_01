r"""Emergency debug script for HC18 mask pairing and U-Net overfitting.

PowerShell:

    python .\scripts\41_emergency_debug_hc18_mask_pairing_and_unet.py

This script does not train the full segmentation model and does not evaluate
SR outputs. It only diagnoses whether 20 HC18 training samples can be overfit
after selecting a reliable mask preprocessing method.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

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
HR_DIR = HC18_ROOT / "HR"
MASK_DIR = HC18_ROOT / "masks"

DEBUG_DIR = PROJECT_ROOT / "results" / "hc18_unet_debug"
PAIRING_GRID = DEBUG_DIR / "debug_pairing_large_grid.png"
PREPROCESSING_FIGURE = DEBUG_DIR / "debug_mask_preprocessing_examples.png"
MASK_STATS_CSV = DEBUG_DIR / "debug_mask_statistics.csv"
OVERFIT_CURVE = DEBUG_DIR / "debug_overfit_no_aug_curve.png"
OVERFIT_PREDICTIONS = DEBUG_DIR / "debug_overfit_predictions.png"
DEBUG_REPORT = DEBUG_DIR / "debug_report.md"
SELECTED_METHOD_TXT = DEBUG_DIR / "selected_mask_preprocessing.txt"

IMAGE_SIZE = 256
AREA_MIN = 0.03
AREA_MAX = 0.60
VERY_SMALL_AREA = 0.005
VERY_LARGE_AREA = 0.80


@dataclass(frozen=True)
class Sample:
    image_id: str
    image_path: Path
    mask_path: Path
    original_name: str


@dataclass(frozen=True)
class MethodStats:
    method: str
    mean_area_ratio: float
    empty_masks: int
    very_small_masks: int
    very_large_masks: int
    single_component_masks: int
    valid_area_masks: int
    score: float


@dataclass
class DebugState:
    selected_samples: list[Sample]
    selected_method: str
    method_stats: list[MethodStats]
    pairing_stats: pd.DataFrame
    aligned_mask_count: int
    suspicious_mask_count: int
    overfit_final_dice: float | None = None
    overfit_final_iou: float | None = None
    predicted_mask_mean: float | None = None
    target_mask_mean: float | None = None
    likely_cause: str = "Not determined yet."


MaskMethod = Callable[[np.ndarray], np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose HC18 image-mask pairing, compare mask preprocessing "
            "methods, and run a no-augmentation 20-sample U-Net overfit test."
        )
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=500, help="Overfit epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Overfit batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Keep 0 for Windows PowerShell compatibility.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def warn(message: str) -> None:
    print(f"WARNING: {message}")


def read_gray(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return np.squeeze(image).astype(np.uint8)


def resolve_csv_path(value: Any) -> Path | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else HC18_ROOT / path


def load_train_samples(seed: int, count: int | None = 30) -> list[Sample]:
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Missing train CSV: {TRAIN_CSV}")

    df = pd.read_csv(TRAIN_CSV)
    if "has_mask" in df.columns:
        df = df[df["has_mask"].astype(int) == 1].copy()
    if df.empty:
        raise RuntimeError(f"No mask-backed rows found in {TRAIN_CSV}")

    if count is None:
        selected = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    else:
        n_samples = min(count, len(df))
        if n_samples < count:
            warn(f"Requested {count} training samples, but only {n_samples} are available.")
        selected = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)

    samples: list[Sample] = []
    for _, row in selected.iterrows():
        image_id = str(row["image_id"]).strip()
        image_path = resolve_csv_path(row.get("hr_path")) or HR_DIR / f"{image_id}.png"
        mask_path = resolve_csv_path(row.get("mask_path")) or MASK_DIR / f"{image_id}.png"
        samples.append(
            Sample(
                image_id=image_id,
                image_path=image_path,
                mask_path=mask_path,
                original_name=str(row.get("original_name", "")),
            )
        )
    return samples


def binary_mask(mask_gray: np.ndarray) -> np.ndarray:
    return (mask_gray > 0).astype(np.uint8)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if not np.any(binary):
        return binary
    padded = np.pad(binary, 1, mode="constant", constant_values=0)
    flood = padded.copy()
    flood_mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 1)
    holes = (flood[1:-1, 1:-1] == 0) & (binary == 0)
    return np.logical_or(binary > 0, holes).astype(np.uint8)


def external_contours(mask: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return list(contours)


def preprocess_raw_binary(mask_gray: np.ndarray) -> np.ndarray:
    return binary_mask(mask_gray)


def preprocess_largest_contour_filled(mask_gray: np.ndarray) -> np.ndarray:
    raw = binary_mask(mask_gray)
    contours = external_contours(raw)
    if not contours:
        return np.zeros_like(raw, dtype=np.uint8)
    largest = max(contours, key=cv2.contourArea)
    output = np.zeros_like(raw, dtype=np.uint8)
    cv2.drawContours(output, [largest], -1, 1, thickness=cv2.FILLED)
    return fill_holes(output)


def preprocess_all_contours_filled(mask_gray: np.ndarray) -> np.ndarray:
    raw = binary_mask(mask_gray)
    contours = external_contours(raw)
    output = np.zeros_like(raw, dtype=np.uint8)
    if contours:
        cv2.drawContours(output, contours, -1, 1, thickness=cv2.FILLED)
    return fill_holes(output)


def preprocess_largest_convex_hull(mask_gray: np.ndarray) -> np.ndarray:
    raw = binary_mask(mask_gray)
    contours = external_contours(raw)
    if not contours:
        return np.zeros_like(raw, dtype=np.uint8)
    largest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest)
    output = np.zeros_like(raw, dtype=np.uint8)
    cv2.drawContours(output, [hull], -1, 1, thickness=cv2.FILLED)
    return fill_holes(output)


def preprocess_dilated_contour_then_filled(mask_gray: np.ndarray) -> np.ndarray:
    raw = binary_mask(mask_gray)
    if not np.any(raw):
        return np.zeros_like(raw, dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(raw, kernel, iterations=1)
    filled = preprocess_largest_contour_filled((dilated * 255).astype(np.uint8))
    closed = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel, iterations=1)
    return fill_holes(closed)


def preprocess_ellipse_fitted(mask_gray: np.ndarray) -> np.ndarray:
    raw = binary_mask(mask_gray)
    contours = external_contours(raw)
    if not contours:
        return np.zeros_like(raw, dtype=np.uint8)
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return preprocess_largest_contour_filled(mask_gray)
    ellipse = cv2.fitEllipse(largest)
    output = np.zeros_like(raw, dtype=np.uint8)
    cv2.ellipse(output, ellipse, color=1, thickness=cv2.FILLED)
    return fill_holes(output)


PREPROCESSING_METHODS: dict[str, MaskMethod] = {
    "A_raw_binary_mask_gt_0": preprocess_raw_binary,
    "B_largest_contour_filled": preprocess_largest_contour_filled,
    "C_all_contours_filled": preprocess_all_contours_filled,
    "D_convex_hull_largest_contour": preprocess_largest_convex_hull,
    "E_dilated_contour_then_filled": preprocess_dilated_contour_then_filled,
    "F_ellipse_fitted_filled": preprocess_ellipse_fitted,
}


def mask_area_ratio(mask: np.ndarray) -> float:
    return float(np.mean(mask > 0))


def connected_components_count(mask: np.ndarray) -> int:
    num_labels, _ = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    return max(int(num_labels) - 1, 0)


def bbox_touches_border(mask: np.ndarray) -> bool:
    binary = mask > 0
    if not np.any(binary):
        return False
    ys, xs = np.where(binary)
    h, w = binary.shape
    margin = max(2, int(round(min(h, w) * 0.01)))
    return (
        int(xs.min()) <= margin
        or int(ys.min()) <= margin
        or int(xs.max()) >= w - 1 - margin
        or int(ys.max()) >= h - 1 - margin
    )


def normalized_panel(image: np.ndarray) -> np.ndarray:
    panel = image.astype(np.float32)
    max_value = float(np.max(panel)) if panel.size else 0.0
    if max_value > 1.0:
        panel /= 255.0
    panel = np.clip(panel, 0.0, 1.0)
    return (panel * 255.0).astype(np.uint8)


def overlay_mask(image: np.ndarray, raw_mask: np.ndarray, processed_mask: np.ndarray) -> np.ndarray:
    image_u8 = normalized_panel(image)
    rgb = np.repeat(image_u8[..., None], 3, axis=2)
    if processed_mask.shape != image.shape:
        processed_mask = cv2.resize(
            (processed_mask > 0).astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    if raw_mask.shape != image.shape:
        raw_mask = cv2.resize(
            (raw_mask > 0).astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    processed = processed_mask > 0
    raw = raw_mask > 0
    rgb[processed, 1] = 255
    rgb[processed, 0] = np.minimum(rgb[processed, 0], 80)
    rgb[raw, 0] = 255
    rgb[raw, 1] = np.minimum(rgb[raw, 1], 80)
    return rgb


def resize_for_grid(image: np.ndarray, size: int = 160) -> np.ndarray:
    if image.ndim == 2:
        image = np.repeat(normalized_panel(image)[..., None], 3, axis=2)
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def add_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.42,
    color: tuple[int, int, int] = (0, 0, 0),
) -> None:
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def save_canvas_png(canvas: np.ndarray, output_path: Path) -> None:
    fig_w = max(canvas.shape[1] / 300.0, 4.0)
    fig_h = max(canvas.shape[0] / 300.0, 4.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    ax.imshow(canvas)
    ax.axis("off")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def inspect_pairing(samples: list[Sample]) -> tuple[pd.DataFrame, int, int]:
    rows: list[dict[str, Any]] = []
    grid_records: list[dict[str, Any]] = []
    aligned_count = 0
    suspicious_count = 0

    for index, sample in enumerate(samples, start=1):
        image = read_gray(sample.image_path)
        mask_gray = read_gray(sample.mask_path)
        image_missing = image is None
        mask_missing = mask_gray is None

        if image_missing:
            print(f"[{index:02d}] Missing image: {sample.image_path}")
        if mask_missing:
            print(f"[{index:02d}] Missing mask: {sample.mask_path}")

        image_shape = "" if image is None else str(tuple(image.shape))
        mask_shape = "" if mask_gray is None else str(tuple(mask_gray.shape))
        nonzero_count = 0 if mask_gray is None else int(np.count_nonzero(mask_gray))
        raw_ratio = float("nan") if mask_gray is None else mask_area_ratio(binary_mask(mask_gray))
        processed = (
            np.zeros_like(image, dtype=np.uint8)
            if image is not None and mask_gray is None
            else None
        )
        status_flags: list[str] = []

        if image_missing or mask_missing:
            status_flags.append("missing")
        elif image.shape != mask_gray.shape:
            status_flags.append("shape mismatch")
        else:
            processed = preprocess_largest_contour_filled(mask_gray)
            processed_ratio = mask_area_ratio(processed)
            if nonzero_count == 0:
                status_flags.append("empty")
            if raw_ratio < VERY_SMALL_AREA:
                status_flags.append("very small raw")
            if raw_ratio > VERY_LARGE_AREA:
                status_flags.append("very large raw")
            if processed_ratio < AREA_MIN or processed_ratio > AREA_MAX:
                status_flags.append("processed area outside expected range")
            if bbox_touches_border(processed):
                status_flags.append("mask touches border")
            if connected_components_count(processed) != 1:
                status_flags.append("not one component")

            if not status_flags:
                aligned_count += 1

        if status_flags:
            suspicious_count += 1

        status = "OK" if not status_flags else "; ".join(status_flags)
        processed_ratio = float("nan") if processed is None else mask_area_ratio(processed)
        print(
            f"[{index:02d}] image={sample.image_path.name} mask={sample.mask_path.name} "
            f"image_shape={image_shape} mask_shape={mask_shape} "
            f"mask_nonzero={nonzero_count} raw_area={raw_ratio:.6f} "
            f"processed_area={processed_ratio:.6f} status={status}"
        )

        rows.append(
            {
                "sample_id": sample.image_id,
                "original_name": sample.original_name,
                "image_path": str(sample.image_path),
                "mask_path": str(sample.mask_path),
                "image_filename": sample.image_path.name,
                "mask_filename": sample.mask_path.name,
                "image_shape": image_shape,
                "mask_shape": mask_shape,
                "nonzero_mask_pixel_count": nonzero_count,
                "raw_mask_area_ratio": raw_ratio,
                "largest_contour_area_ratio": processed_ratio,
                "status": status,
                "image_missing": image_missing,
                "mask_missing": mask_missing,
            }
        )

        if image is not None and mask_gray is not None:
            grid_records.append(
                {
                    "sample": sample,
                    "image": image,
                    "raw": binary_mask(mask_gray),
                    "processed": preprocess_largest_contour_filled(mask_gray),
                    "status": status,
                }
            )

    stats_df = pd.DataFrame(rows)
    save_pairing_grid(grid_records)
    return stats_df, aligned_count, suspicious_count


def save_pairing_grid(records: list[dict[str, Any]]) -> None:
    if not records:
        warn("No readable samples available for pairing grid.")
        return

    panel_size = 150
    row_height = panel_size + 54
    label_width = 235
    header_height = 32
    n_cols = 4
    canvas_h = header_height + len(records) * row_height
    canvas_w = label_width + n_cols * panel_size
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    for col, header in enumerate(["HR image", "Raw mask", "Processed mask", "Overlay"]):
        add_text(canvas, header, (label_width + col * panel_size + 8, 22), 0.45)

    for row_idx, record in enumerate(records):
        y0 = header_height + row_idx * row_height
        sample = record["sample"]
        status = record["status"]
        color = (0, 120, 0) if status == "OK" else (220, 0, 0)
        add_text(canvas, sample.image_id, (8, y0 + 22), 0.45, color)
        add_text(canvas, f"img: {sample.image_path.name}"[:34], (8, y0 + 42), 0.38, color)
        add_text(canvas, f"mask: {sample.mask_path.name}"[:34], (8, y0 + 62), 0.38, color)
        add_text(canvas, status[:34], (8, y0 + 84), 0.36, color)

        panels = [
            resize_for_grid(record["image"], panel_size),
            resize_for_grid(record["raw"] * 255, panel_size),
            resize_for_grid(record["processed"] * 255, panel_size),
            resize_for_grid(
                overlay_mask(record["image"], record["raw"], record["processed"]),
                panel_size,
            ),
        ]
        for col, panel in enumerate(panels):
            x0 = label_width + col * panel_size
            canvas[y0 + 10 : y0 + 10 + panel_size, x0 : x0 + panel_size] = panel

    save_canvas_png(canvas, PAIRING_GRID)
    print(f"Saved pairing grid: {rel_path(PAIRING_GRID)}")


def evaluate_preprocessing_methods(samples: list[Sample]) -> tuple[str, list[MethodStats]]:
    stats: list[MethodStats] = []
    per_method_masks: dict[str, list[np.ndarray]] = {name: [] for name in PREPROCESSING_METHODS}

    for sample in samples:
        mask_gray = read_gray(sample.mask_path)
        if mask_gray is None:
            continue
        for method_name, method_func in PREPROCESSING_METHODS.items():
            mask = method_func(mask_gray)
            per_method_masks[method_name].append(mask)

    for method_name, masks in per_method_masks.items():
        if not masks:
            stats.append(MethodStats(method_name, float("nan"), 999, 999, 999, 0, 0, -999.0))
            continue

        area_ratios = np.array([mask_area_ratio(mask) for mask in masks], dtype=np.float32)
        empty_count = int(np.sum(area_ratios <= 0.0))
        very_small_count = int(np.sum(area_ratios < VERY_SMALL_AREA))
        very_large_count = int(np.sum(area_ratios > VERY_LARGE_AREA))
        valid_area_count = int(np.sum((area_ratios >= AREA_MIN) & (area_ratios <= AREA_MAX)))
        single_component_count = int(
            sum(connected_components_count(mask) == 1 for mask in masks if np.any(mask))
        )
        mean_area = float(np.mean(area_ratios))

        valid_area_fraction = valid_area_count / max(len(masks), 1)
        single_component_fraction = single_component_count / max(len(masks), 1)
        mean_area_penalty = abs(mean_area - 0.22)
        thin_outline_penalty = 0.15 if method_name == "A_raw_binary_mask_gt_0" and mean_area < 0.03 else 0.0
        filled_bonus = 0.03 if method_name in {
            "B_largest_contour_filled",
            "E_dilated_contour_then_filled",
            "F_ellipse_fitted_filled",
        } else 0.0
        score = (
            valid_area_fraction
            + 0.5 * single_component_fraction
            - 0.02 * empty_count
            - 0.02 * very_large_count
            - mean_area_penalty
            - thin_outline_penalty
            + filled_bonus
        )

        stats.append(
            MethodStats(
                method=method_name,
                mean_area_ratio=mean_area,
                empty_masks=empty_count,
                very_small_masks=very_small_count,
                very_large_masks=very_large_count,
                single_component_masks=single_component_count,
                valid_area_masks=valid_area_count,
                score=float(score),
            )
        )

    stats = sorted(stats, key=lambda item: item.score, reverse=True)
    selected_method = stats[0].method
    print()
    print("Mask preprocessing method summary:")
    for item in stats:
        print(
            f"  {item.method}: mean_area={item.mean_area_ratio:.4f}, "
            f"empty={item.empty_masks}, very_small={item.very_small_masks}, "
            f"very_large={item.very_large_masks}, one_component={item.single_component_masks}, "
            f"valid_area={item.valid_area_masks}, score={item.score:.4f}"
        )
    print(f"Selected preprocessing method: {selected_method}")

    save_preprocessing_comparison(samples[:8])
    return selected_method, stats


def save_preprocessing_comparison(samples: list[Sample]) -> None:
    records: list[dict[str, Any]] = []
    for sample in samples:
        image = read_gray(sample.image_path)
        mask_gray = read_gray(sample.mask_path)
        if image is None or mask_gray is None:
            continue
        records.append(
            {
                "sample": sample,
                "image": image,
                "masks": {
                    method_name: method_func(mask_gray)
                    for method_name, method_func in PREPROCESSING_METHODS.items()
                },
            }
        )

    if not records:
        warn("No readable samples available for preprocessing comparison.")
        return

    method_names = list(PREPROCESSING_METHODS.keys())
    panel_size = 135
    row_height = panel_size + 50
    label_width = 190
    header_height = 42
    n_cols = 1 + len(method_names)
    canvas_h = header_height + len(records) * row_height
    canvas_w = label_width + n_cols * panel_size
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    headers = ["HR"] + [name.split("_", 1)[0] for name in method_names]
    for col, header in enumerate(headers):
        add_text(canvas, header, (label_width + col * panel_size + 8, 24), 0.43)

    for row_idx, record in enumerate(records):
        y0 = header_height + row_idx * row_height
        sample = record["sample"]
        add_text(canvas, sample.image_id, (8, y0 + 24), 0.45)
        add_text(canvas, sample.mask_path.name[:28], (8, y0 + 46), 0.36)

        panels = [resize_for_grid(record["image"], panel_size)]
        for method_name in method_names:
            mask = record["masks"][method_name]
            panels.append(resize_for_grid(mask * 255, panel_size))

        for col, panel in enumerate(panels):
            x0 = label_width + col * panel_size
            canvas[y0 + 8 : y0 + 8 + panel_size, x0 : x0 + panel_size] = panel
            if col > 0:
                method_name = method_names[col - 1]
                ratio = mask_area_ratio(record["masks"][method_name])
                add_text(
                    canvas,
                    f"area={ratio:.3f}",
                    (x0 + 6, y0 + panel_size + 28),
                    0.34,
                    (40, 40, 40),
                )

    save_canvas_png(canvas, PREPROCESSING_FIGURE)
    print(f"Saved preprocessing comparison: {rel_path(PREPROCESSING_FIGURE)}")


def resize_image_01(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    return np.clip(resized.astype(np.float32) / 255.0, 0.0, 1.0)


def resize_mask_01(mask: np.ndarray) -> np.ndarray:
    resized = cv2.resize(
        (mask > 0).astype(np.uint8),
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )
    return (resized > 0).astype(np.float32)


class OverfitDataset(Dataset):
    def __init__(self, samples: list[Sample], preprocessing_method: str) -> None:
        self.samples = samples
        self.method_func = PREPROCESSING_METHODS[preprocessing_method]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = read_gray(sample.image_path)
        mask_gray = read_gray(sample.mask_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {sample.image_path}")
        if mask_gray is None:
            raise FileNotFoundError(f"Could not read mask: {sample.mask_path}")

        mask = self.method_func(mask_gray)
        image_tensor = torch.from_numpy(resize_image_01(image)).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(resize_mask_01(mask)).unsqueeze(0).float()
        return image_tensor, mask_tensor


class TinyDoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc1 = TinyDoubleConv(1, 16)
        self.enc2 = TinyDoubleConv(16, 32)
        self.enc3 = TinyDoubleConv(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = TinyDoubleConv(64, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = TinyDoubleConv(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = TinyDoubleConv(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = TinyDoubleConv(32, 16)
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.bottleneck(self.pool(x3))
        x = self.up3(x4)
        x = self.dec3(torch.cat([x, x3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))
        return self.out(x)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = torch.sum(probs * targets, dim=(1, 2, 3))
    denominator = torch.sum(probs, dim=(1, 2, 3)) + torch.sum(targets, dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def bce_plus_dice(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets) + dice_loss(logits, targets)


def dice_iou_tensor(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    intersection = torch.sum(preds * targets, dim=(1, 2, 3))
    pred_area = torch.sum(preds, dim=(1, 2, 3))
    target_area = torch.sum(targets, dim=(1, 2, 3))
    union = pred_area + target_area - intersection
    dice = (2.0 * intersection + eps) / (pred_area + target_area + eps)
    iou = (intersection + eps) / (union + eps)
    return dice.mean(), iou.mean(), preds.mean(), targets.mean()


def select_overfit_samples(samples: list[Sample], method_name: str) -> list[Sample]:
    method_func = PREPROCESSING_METHODS[method_name]
    selected: list[Sample] = []
    for sample in samples:
        image = read_gray(sample.image_path)
        mask_gray = read_gray(sample.mask_path)
        if image is None or mask_gray is None or image.shape != mask_gray.shape:
            continue
        mask = method_func(mask_gray)
        ratio = mask_area_ratio(mask)
        if AREA_MIN <= ratio <= AREA_MAX and connected_components_count(mask) == 1:
            selected.append(sample)
        if len(selected) == 20:
            break
    return selected


def evaluate_overfit(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_pred_mean = 0.0
    total_target_mean = 0.0
    total_items = 0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = bce_plus_dice(logits, masks)
            dice, iou, pred_mean, target_mean = dice_iou_tensor(logits, masks)
            batch_size = int(images.size(0))
            total_loss += float(loss.item()) * batch_size
            total_dice += float(dice.item()) * batch_size
            total_iou += float(iou.item()) * batch_size
            total_pred_mean += float(pred_mean.item()) * batch_size
            total_target_mean += float(target_mean.item()) * batch_size
            total_items += batch_size

    total_items = max(total_items, 1)
    return (
        total_loss / total_items,
        total_dice / total_items,
        total_iou / total_items,
        total_pred_mean / total_items,
        total_target_mean / total_items,
    )


def run_no_aug_overfit(
    samples: list[Sample],
    method_name: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, float, float, float, list[dict[str, float]], nn.Module]:
    dataset = OverfitDataset(samples, method_name)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model = TinyUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history: list[dict[str, float]] = []

    final_loss = 0.0
    final_dice = 0.0
    final_iou = 0.0
    final_pred_mean = 0.0
    final_target_mean = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = bce_plus_dice(logits, masks)
            loss.backward()
            optimizer.step()

        final_loss, final_dice, final_iou, final_pred_mean, final_target_mean = evaluate_overfit(
            model,
            eval_loader,
            device,
        )
        history.append(
            {
                "epoch": float(epoch),
                "loss": final_loss,
                "dice": final_dice,
                "iou": final_iou,
                "predicted_mask_mean": final_pred_mean,
                "target_mask_mean": final_target_mean,
            }
        )

        if epoch == 1 or epoch % 25 == 0 or final_dice > 0.95 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} loss={final_loss:.6f} dice={final_dice:.6f} "
                f"iou={final_iou:.6f} predicted_mask_mean={final_pred_mean:.6f} "
                f"target_mask_mean={final_target_mean:.6f}"
            )
        if final_dice > 0.95:
            print("Early stopping: overfit Dice exceeded 0.95.")
            break

    plot_overfit_curve(history)
    save_overfit_predictions(model, samples, method_name, device)
    return final_dice, final_iou, final_pred_mean, final_target_mean, history, model


def plot_overfit_curve(history: list[dict[str, float]]) -> None:
    df = pd.DataFrame(history)
    if df.empty:
        return
    fig, ax1 = plt.subplots(figsize=(8.5, 5.0))
    ax1.plot(df["epoch"], df["loss"], label="loss", color="#1f77b4")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["dice"], label="Dice", color="#2ca02c")
    ax2.plot(df["epoch"], df["iou"], label="IoU", color="#d62728")
    ax2.set_ylabel("Score")
    ax2.set_ylim(0.0, 1.0)

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="center right")
    fig.tight_layout()
    fig.savefig(OVERFIT_CURVE, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved overfit curve: {rel_path(OVERFIT_CURVE)}")


@torch.no_grad()
def predict_probability(model: nn.Module, image: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    image_tensor = torch.from_numpy(resize_image_01(image)).unsqueeze(0).unsqueeze(0).float().to(device)
    logits = model(image_tensor)
    probs = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
    return probs.astype(np.float32)


def overlay_prediction(image: np.ndarray, target: np.ndarray, pred_binary: np.ndarray) -> np.ndarray:
    base = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    rgb = np.repeat(normalized_panel(base)[..., None], 3, axis=2)
    target_bool = target > 0
    pred_bool = pred_binary > 0
    rgb[target_bool, 1] = 255
    rgb[target_bool, 0] = np.minimum(rgb[target_bool, 0], 80)
    rgb[pred_bool, 0] = 255
    rgb[pred_bool, 1] = np.minimum(rgb[pred_bool, 1], 80)
    return rgb


def save_overfit_predictions(
    model: nn.Module,
    samples: list[Sample],
    method_name: str,
    device: torch.device,
) -> None:
    method_func = PREPROCESSING_METHODS[method_name]
    selected = samples[: min(8, len(samples))]
    if not selected:
        return

    panel_size = 138
    label_width = 190
    row_height = panel_size + 40
    header_height = 34
    n_cols = 5
    canvas_h = header_height + len(selected) * row_height
    canvas_w = label_width + n_cols * panel_size
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    for col, header in enumerate(
        ["HR image", "Target mask", "Pred prob", "Pred binary", "Overlay"]
    ):
        add_text(canvas, header, (label_width + col * panel_size + 5, 22), 0.38)

    for row_idx, sample in enumerate(selected):
        image = read_gray(sample.image_path)
        mask_gray = read_gray(sample.mask_path)
        if image is None or mask_gray is None:
            continue
        target = resize_mask_01(method_func(mask_gray))
        prob = predict_probability(model, image, device)
        pred_binary = (prob >= 0.5).astype(np.uint8)
        overlay = overlay_prediction(image, target, pred_binary)
        panels = [
            resize_for_grid(image, panel_size),
            resize_for_grid(target * 255, panel_size),
            resize_for_grid(prob * 255, panel_size),
            resize_for_grid(pred_binary * 255, panel_size),
            resize_for_grid(overlay, panel_size),
        ]
        y0 = header_height + row_idx * row_height
        add_text(canvas, sample.image_id, (8, y0 + 24), 0.45)
        for col, panel in enumerate(panels):
            x0 = label_width + col * panel_size
            canvas[y0 + 6 : y0 + 6 + panel_size, x0 : x0 + panel_size] = panel

    save_canvas_png(canvas, OVERFIT_PREDICTIONS)
    print(f"Saved overfit predictions: {rel_path(OVERFIT_PREDICTIONS)}")


def infer_likely_cause(
    state: DebugState,
    overfit_samples: list[Sample],
) -> str:
    if len(overfit_samples) < 20:
        return (
            "Fewer than 20 non-suspicious masks were available. The split likely has "
            "mask preprocessing, image-mask pairing, or mask-format problems."
        )
    if state.aligned_mask_count < 20:
        return (
            "The pairing audit found fewer than 20 clearly aligned samples among the "
            "30 inspected training rows."
        )
    if state.predicted_mask_mean is not None and state.predicted_mask_mean < 0.005:
        return "The U-Net predictions are nearly blank after overfit training."
    if state.predicted_mask_mean is not None and state.predicted_mask_mean > 0.95:
        return "The U-Net predictions are nearly all foreground after overfit training."
    if state.target_mask_mean is not None and (
        state.target_mask_mean < VERY_SMALL_AREA or state.target_mask_mean > VERY_LARGE_AREA
    ):
        return "The selected target masks have invalid foreground area ratios."
    return (
        "Masks appear usable but the no-augmentation U-Net still failed to overfit. "
        "Likely causes are target construction mismatch, insufficient optimization "
        "stability, or a deeper data-label alignment issue not visible in the grid."
    )


def method_stats_markdown(stats: list[MethodStats]) -> str:
    lines = [
        "| method | mean_area_ratio | empty | very_small | very_large | one_component | valid_area | score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in stats:
        lines.append(
            f"| {item.method} | {item.mean_area_ratio:.6f} | {item.empty_masks} | "
            f"{item.very_small_masks} | {item.very_large_masks} | "
            f"{item.single_component_masks} | {item.valid_area_masks} | {item.score:.6f} |"
        )
    return "\n".join(lines)


def write_debug_report(
    state: DebugState,
    overfit_samples: list[Sample],
) -> None:
    passed = state.overfit_final_dice is not None and state.overfit_final_dice >= 0.90
    if passed:
        pass_text = (
            "Overfit sanity check passed. Full U-Net training can now be restarted "
            "using the selected mask preprocessing method."
        )
        SELECTED_METHOD_TXT.write_text(state.selected_method + "\n", encoding="utf-8")
    else:
        pass_text = (
            "Overfit sanity check failed. Do not train the full segmentation model "
            "until the 20-sample overfit test reaches at least 0.90 Dice."
        )

    masks_aligned = "mostly aligned" if state.aligned_mask_count >= 20 else "questionable"
    thin_or_invalid = int(
        state.pairing_stats["status"].astype(str).str.contains(
            "empty|very small|very large|outside expected|shape mismatch|touches border",
            regex=True,
        ).sum()
    )
    predicted_behavior = "not available"
    if state.predicted_mask_mean is not None:
        if state.predicted_mask_mean < 0.005:
            predicted_behavior = "predicted masks are nearly blank"
        elif state.predicted_mask_mean > 0.95:
            predicted_behavior = "predicted masks are nearly all foreground"
        else:
            predicted_behavior = "predicted mask foreground fraction is plausible"

    target_validity = "valid"
    if state.target_mask_mean is not None:
        if state.target_mask_mean < VERY_SMALL_AREA:
            target_validity = "too thin or nearly empty"
        elif state.target_mask_mean > VERY_LARGE_AREA:
            target_validity = "too large"

    report = f"""# HC18 Emergency U-Net Debug Report

## Result

{pass_text}

## Files

- Pairing grid: `{rel_path(PAIRING_GRID)}`
- Mask preprocessing comparison: `{rel_path(PREPROCESSING_FIGURE)}`
- Mask statistics: `{rel_path(MASK_STATS_CSV)}`
- Overfit curve: `{rel_path(OVERFIT_CURVE)}`
- Overfit predictions: `{rel_path(OVERFIT_PREDICTIONS)}`

## Pairing Diagnosis

- Inspected training samples: {len(state.selected_samples)}
- Clearly aligned / non-suspicious samples: {state.aligned_mask_count}
- Suspicious samples: {state.suspicious_mask_count}
- Masks appear aligned: {masks_aligned}
- Thin, empty, too large, shifted, wrong-size, or suspicious masks: {thin_or_invalid}

## Mask Preprocessing Diagnosis

- Selected preprocessing method: `{state.selected_method}`

{method_stats_markdown(state.method_stats)}

## No-Augmentation Overfit Diagnosis

- Overfit samples used: {len(overfit_samples)}
- Final Dice: {state.overfit_final_dice if state.overfit_final_dice is not None else "not run"}
- Final IoU: {state.overfit_final_iou if state.overfit_final_iou is not None else "not run"}
- Predicted mask mean: {state.predicted_mask_mean if state.predicted_mask_mean is not None else "not run"}
- Target mask mean: {state.target_mask_mean if state.target_mask_mean is not None else "not run"}
- Prediction behavior: {predicted_behavior}
- Target masks are: {target_validity}

## Likely Cause

{state.likely_cause}

## Guardrail

Do not train the full segmentation model until this 20-sample overfit test reaches at least 0.90 Dice.
"""
    DEBUG_REPORT.write_text(report, encoding="utf-8")
    print(f"Saved debug report: {rel_path(DEBUG_REPORT)}")
    if passed:
        print(f"Saved selected preprocessing method: {rel_path(SELECTED_METHOD_TXT)}")


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    print("Emergency HC18 mask pairing and U-Net overfit debug")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Debug output folder: {DEBUG_DIR}")
    print("Full segmentation training is intentionally not implemented in this script.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for input_path in (TRAIN_CSV, VAL_CSV, TEST_CSV, HR_DIR, MASK_DIR):
        if not input_path.exists():
            raise FileNotFoundError(f"Required input is missing: {input_path}")

    selected_samples = load_train_samples(seed=args.seed, count=30)
    all_train_samples = load_train_samples(seed=args.seed, count=None)
    pairing_stats, aligned_count, suspicious_count = inspect_pairing(selected_samples)
    pairing_stats.to_csv(MASK_STATS_CSV, index=False)
    print(f"Saved mask statistics: {rel_path(MASK_STATS_CSV)}")

    selected_method, method_stats = evaluate_preprocessing_methods(selected_samples)
    state = DebugState(
        selected_samples=selected_samples,
        selected_method=selected_method,
        method_stats=method_stats,
        pairing_stats=pairing_stats,
        aligned_mask_count=aligned_count,
        suspicious_mask_count=suspicious_count,
    )

    overfit_samples = select_overfit_samples(all_train_samples, selected_method)
    if len(overfit_samples) != 20:
        state.likely_cause = infer_likely_cause(state, overfit_samples)
        write_debug_report(state, overfit_samples)
        raise SystemExit(
            "Could not select exactly 20 non-suspicious masks for overfit testing. "
            f"Selected {len(overfit_samples)}."
        )

    print()
    print("Starting no-augmentation 20-sample overfit test.")
    print(f"Selected preprocessing method: {selected_method}")
    final_dice, final_iou, pred_mean, target_mean, _, _ = run_no_aug_overfit(
        overfit_samples,
        selected_method,
        args,
        device,
    )
    state.overfit_final_dice = final_dice
    state.overfit_final_iou = final_iou
    state.predicted_mask_mean = pred_mean
    state.target_mask_mean = target_mean
    state.likely_cause = infer_likely_cause(state, overfit_samples)

    if final_dice >= 0.90:
        state.likely_cause = (
            "The overfit test passed. The previous failure was likely caused by "
            "the mask preprocessing method, augmentation, loss weighting, or a "
            "different target construction path used before this debug run."
        )

    write_debug_report(state, overfit_samples)

    if final_dice < 0.90:
        raise SystemExit(
            "Overfit Dice remained below 0.90. Full U-Net training must remain stopped."
        )

    print(
        "Overfit sanity check passed. Full U-Net training can now be restarted "
        "using the selected mask preprocessing method."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
