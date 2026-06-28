"""
Add reviewer-requested anatomy metrics for HC18 restored outputs.

Metrics are computed with a simple proxy fetal-head segmentation on each
restored output and the ground-truth HC18 fetal-head mask. The procedure is
intended for anatomy-preservation analysis, not clinical validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

try:
    import cv2
    import matplotlib
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard.
    missing_package = exc.name or "unknown"
    raise SystemExit(
        "Missing required dependency: "
        f"{missing_package}. Install pandas, numpy, opencv-python, and "
        "matplotlib in the project Python environment, then rerun this script."
    ) from exc


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HC18_ROOT = PROJECT_ROOT / "data" / "processed" / "HC18_SR"
TEST_CSV = HC18_ROOT / "test.csv"
MASK_DIR = HC18_ROOT / "masks"
SR_OUTPUT_ROOT = PROJECT_ROOT / "results" / "sr_outputs"
FINAL_COMPARISON_SUMMARY = (
    PROJECT_ROOT / "results" / "final_all_hc18_comparison_summary.csv"
)
ANATOMY_PRESERVATION_SUMMARY = (
    PROJECT_ROOT / "results" / "anatomy_preservation_all_models_summary.csv"
)

PER_IMAGE_CSV = PROJECT_ROOT / "results" / "hc18_dice_iou_measurement_metrics.csv"
SUMMARY_CSV = PROJECT_ROOT / "results" / "hc18_dice_iou_measurement_summary.csv"
TABLE_MD = PROJECT_ROOT / "results" / "hc18_dice_iou_measurement_table.md"
REVIEW_READY_MD = PROJECT_ROOT / "results" / "final_review_ready_anatomy_metrics.md"
FIGURE_DIR = PROJECT_ROOT / "results" / "hc18_dice_iou_measurement_figures"
DICE_IOU_FIGURE = FIGURE_DIR / "dice_iou_bar_chart.png"
MEASUREMENT_FIGURE = FIGURE_DIR / "fetal_measurement_error_bar_chart.png"

VALIDATION_NOTE = (
    "Dice, IoU, and fetal measurement error are computed using a proxy "
    "segmentation procedure on restored outputs and ground-truth HC18 "
    "fetal-head masks. These metrics support anatomy-preservation analysis "
    "but should not be interpreted as expert clinical validation."
)

METHODS = [
    {
        "key": "lq_mixed",
        "label": "LQ mixed baseline",
        "output_dir": HC18_ROOT / "LQ_mixed",
    },
    {
        "key": "sr_only",
        "label": "SR-only CNN",
        "output_dir": SR_OUTPUT_ROOT / "sr_only",
    },
    {
        "key": "quality_aware",
        "label": "Quality-aware CNN",
        "output_dir": SR_OUTPUT_ROOT / "quality_aware",
    },
    {
        "key": "quality_aware_l1_mse",
        "label": "Quality-aware CNN + L1/MSE",
        "output_dir": SR_OUTPUT_ROOT / "quality_aware_l1_mse",
    },
    {
        "key": "diffusion_sr",
        "label": "Conditional Diffusion SR",
        "output_dir": SR_OUTPUT_ROOT / "diffusion_sr",
    },
    {
        "key": "quality_aware_diffusion_sr_v2",
        "label": "Quality-aware Diffusion SR v2",
        "output_dir": SR_OUTPUT_ROOT / "quality_aware_diffusion_sr_v2",
    },
    {
        "key": "cnn_residual_diffusion_sr",
        "label": "CNN-Residual Diffusion SR",
        "output_dir": SR_OUTPUT_ROOT / "cnn_residual_diffusion_sr",
    },
]

PER_IMAGE_COLUMNS = [
    "image_id",
    "method",
    "output_path",
    "mask_path",
    "segmentation_method",
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


def resolve_csv_path(value: Any, base_dir: Path) -> Path | None:
    if pd.isna(value) or not str(value).strip():
        return None
    path = Path(str(value))
    return path if path.is_absolute() else base_dir / path


def read_grayscale_uint8(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return np.squeeze(image).astype(np.uint8)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    image_float = np.squeeze(image).astype(np.float32)
    min_value = float(np.min(image_float))
    max_value = float(np.max(image_float))
    if max_value <= min_value:
        return np.zeros(image_float.shape, dtype=np.uint8)
    normalized = (image_float - min_value) / (max_value - min_value)
    return np.clip(np.round(normalized * 255.0), 0, 255).astype(np.uint8)


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    mask_uint8 = (mask > 0).astype(np.uint8)
    if not np.any(mask_uint8):
        return np.zeros(mask_uint8.shape, dtype=bool)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8,
        connectivity=8,
    )
    if n_labels <= 1:
        return np.zeros(mask_uint8.shape, dtype=bool)

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1
    return labels == largest_label


def fill_holes(mask: np.ndarray) -> np.ndarray:
    mask_uint8 = (mask > 0).astype(np.uint8)
    if not np.any(mask_uint8):
        return mask_uint8.astype(bool)

    padded = np.pad(mask_uint8, pad_width=1, mode="constant", constant_values=0)
    flood = (padded * 255).astype(np.uint8)
    flood_mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood) > 0
    filled = (padded > 0) | holes
    return filled[1:-1, 1:-1]


def resize_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == target_shape:
        return mask.astype(bool)
    resized = cv2.resize(
        mask.astype(np.uint8) * 255,
        (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized > 127


def clean_component(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)
    component = largest_connected_component(mask_uint8 > 0)
    return fill_holes(component)


def adaptive_block_size(image_shape: tuple[int, int]) -> int:
    min_side = min(image_shape)
    block_size = min(51, max(15, (min_side // 8) * 2 + 1))
    if block_size % 2 == 0:
        block_size -= 1
    return max(block_size, 3)


def component_score(mask: np.ndarray) -> float:
    area = float(np.sum(mask))
    total = float(mask.size)
    if area <= 0.0 or total <= 0.0:
        return -1e9

    fraction = area / total
    size_score = -abs(fraction - 0.35)
    if fraction < 0.005:
        size_score -= 1.0
    if fraction > 0.90:
        size_score -= 2.0

    ys, xs = np.nonzero(mask)
    center_y = float(np.mean(ys))
    center_x = float(np.mean(xs))
    image_center_y = (mask.shape[0] - 1) / 2.0
    image_center_x = (mask.shape[1] - 1) / 2.0
    center_distance = np.hypot(center_y - image_center_y, center_x - image_center_x)
    max_distance = max(np.hypot(image_center_y, image_center_x), 1.0)
    center_score = -(center_distance / max_distance)
    return size_score + 0.25 * center_score


def proxy_segment_fetal_head(
    image: np.ndarray,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, str]:
    """
    Segment a proxy fetal-head region from a restored image.

    The method normalizes to 0-255, applies Gaussian blur, uses Otsu and
    adaptive threshold candidates, keeps the largest connected component,
    fills holes, and resizes the final mask to the ground-truth mask size.
    """
    normalized = normalize_to_uint8(image)
    blurred = cv2.GaussianBlur(normalized, (5, 5), sigmaX=0)

    candidates: list[tuple[str, np.ndarray]] = []
    _, otsu = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    candidates.append(("otsu", otsu > 0))

    block_size = adaptive_block_size(blurred.shape)
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        2,
    )
    candidates.append(("adaptive", adaptive > 0))

    # Inverse candidates are a defensive fallback for images where the head
    # region appears darker after restoration.
    _, otsu_inverse = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    candidates.append(("otsu_inverse", otsu_inverse > 0))

    best_mask: np.ndarray | None = None
    best_method = "none"
    best_score = -1e9
    for method_name, candidate in candidates:
        component = clean_component(candidate)
        score = component_score(component)
        if score > best_score:
            best_score = score
            best_mask = component
            best_method = method_name

    if best_mask is None or not np.any(best_mask):
        best_mask = clean_component(blurred > float(np.mean(blurred)))
        best_method = "mean_threshold_fallback"

    return resize_mask(best_mask, target_shape), best_method


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

    pred_area = float(np.sum(predicted_bool))
    gt_area = float(np.sum(gt_bool))
    intersection = float(np.logical_and(predicted_bool, gt_bool).sum())
    union = float(np.logical_or(predicted_bool, gt_bool).sum())

    dice_denominator = pred_area + gt_area
    dice = 0.0 if dice_denominator <= 0.0 else 2.0 * intersection / dice_denominator
    iou = 0.0 if union <= 0.0 else intersection / union

    area_error_percent = (
        np.nan if gt_area <= 0.0 else 100.0 * abs(pred_area - gt_area) / gt_area
    )
    pred_diameter = equivalent_diameter(pred_area)
    gt_diameter = equivalent_diameter(gt_area)
    diameter_error = abs(pred_diameter - gt_diameter)
    diameter_error_percent = (
        np.nan
        if gt_diameter <= 0.0
        else 100.0 * diameter_error / gt_diameter
    )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "boundary_error_px": symmetric_boundary_error(predicted_bool, gt_bool),
        "predicted_area_px": pred_area,
        "ground_truth_area_px": gt_area,
        "fetal_head_area_error_percent": float(area_error_percent),
        "predicted_equivalent_diameter_px": pred_diameter,
        "ground_truth_equivalent_diameter_px": gt_diameter,
        "equivalent_diameter_error_px": diameter_error,
        "equivalent_diameter_error_percent": float(diameter_error_percent),
    }


def read_ground_truth_mask(path: Path) -> np.ndarray | None:
    mask_image = read_grayscale_uint8(path)
    if mask_image is None:
        return None
    mask = mask_image > 127
    mask = largest_connected_component(mask)
    mask = fill_holes(mask)
    return mask if np.any(mask) else None


def resolve_mask_path(row: pd.Series, image_id: str) -> Path | None:
    folder_mask = MASK_DIR / f"{image_id}.png"
    if folder_mask.exists():
        return folder_mask
    return resolve_csv_path(row.get("mask_path"), HC18_ROOT)


def resolve_output_path(method: dict[str, Any], row: pd.Series, image_id: str) -> Path:
    if method["key"] == "lq_mixed":
        csv_path = resolve_csv_path(row.get("lq_mixed_path"), HC18_ROOT)
        if csv_path is not None:
            return csv_path
    return Path(method["output_dir"]) / f"{image_id}.png"


def available_methods() -> list[dict[str, Any]]:
    methods = []
    for method in METHODS:
        output_dir = Path(method["output_dir"])
        if output_dir.exists():
            methods.append(method)
        else:
            warn(f"Skipping {method['label']} because output folder is missing: {output_dir}")
    return methods


def evaluate() -> pd.DataFrame:
    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Missing test CSV: {TEST_CSV}")

    test_df = pd.read_csv(TEST_CSV)
    if "image_id" not in test_df.columns:
        raise ValueError(f"Expected image_id column in {TEST_CSV}")
    if "has_mask" in test_df.columns:
        test_df = test_df[test_df["has_mask"].astype(int) == 1].copy()

    methods = available_methods()
    rows: list[dict[str, Any]] = []
    skipped_missing_output = {method["label"]: 0 for method in methods}
    skipped_bad_image = {method["label"]: 0 for method in methods}

    for _, row in test_df.iterrows():
        image_id = str(row["image_id"])
        mask_path = resolve_mask_path(row, image_id)
        if mask_path is None or not mask_path.exists():
            warn(f"Skipping {image_id}: missing ground-truth mask {mask_path}")
            continue

        gt_mask = read_ground_truth_mask(mask_path)
        if gt_mask is None:
            warn(f"Skipping {image_id}: unreadable or empty mask {mask_path}")
            continue

        for method in methods:
            output_path = resolve_output_path(method, row, image_id)
            if not output_path.exists():
                skipped_missing_output[method["label"]] += 1
                continue

            output_image = read_grayscale_uint8(output_path)
            if output_image is None:
                skipped_bad_image[method["label"]] += 1
                continue

            pred_mask, segmentation_method = proxy_segment_fetal_head(
                output_image,
                gt_mask.shape,
            )
            metrics = compute_segmentation_metrics(pred_mask, gt_mask)
            rows.append(
                {
                    "image_id": image_id,
                    "method": method["label"],
                    "output_path": rel_path(output_path),
                    "mask_path": rel_path(mask_path),
                    "segmentation_method": segmentation_method,
                    **metrics,
                }
            )

    for method in methods:
        label = method["label"]
        if skipped_missing_output[label] > 0:
            warn(f"{label}: skipped {skipped_missing_output[label]} missing outputs.")
        if skipped_bad_image[label] > 0:
            warn(f"{label}: skipped {skipped_bad_image[label]} unreadable outputs.")

    return pd.DataFrame(rows, columns=PER_IMAGE_COLUMNS)


def summarize(per_image_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        label = method["label"]
        method_df = per_image_df[per_image_df["method"] == label]
        if method_df.empty:
            continue

        row: dict[str, Any] = {"method": label, "n_images": int(len(method_df))}
        metric_pairs = [
            ("dice", "dice"),
            ("iou", "iou"),
            ("boundary_error_px", "boundary_error_px"),
            (
                "fetal_head_area_error_percent",
                "fetal_head_area_error_percent",
            ),
            ("equivalent_diameter_error_px", "equivalent_diameter_error_px"),
            (
                "equivalent_diameter_error_percent",
                "equivalent_diameter_error_percent",
            ),
        ]
        for source_column, output_name in metric_pairs:
            values = method_df[source_column].astype(float)
            row[f"mean_{output_name}"] = float(values.mean())
            row[f"std_{output_name}"] = float(values.std(ddof=0))
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
    sources = [
        TEST_CSV,
        MASK_DIR,
        SR_OUTPUT_ROOT,
        FINAL_COMPARISON_SUMMARY,
        ANATOMY_PRESERVATION_SUMMARY,
    ]
    lines = ["| Source | Status |", "| --- | --- |"]
    for source in sources:
        lines.append(f"| `{rel_path(source)}` | {'available' if source.exists() else 'missing'} |")
    return "\n".join(lines)


def existing_summary_markdown(path: Path, columns: list[str]) -> str:
    if not path.exists():
        return f"`{rel_path(path)}` was not found."
    df = pd.read_csv(path)
    available_columns = [column for column in columns if column in df.columns]
    if not available_columns:
        return f"`{rel_path(path)}` does not contain the expected columns."
    return markdown_table(df, available_columns)


def plot_dice_iou(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        warn("Skipping Dice/IoU figure because the summary table is empty.")
        return

    labels = summary_df["method"].tolist()
    x_values = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x_values - width / 2, summary_df["mean_dice"], width, label="Dice")
    ax.bar(x_values + width / 2, summary_df["mean_iou"], width, label="IoU")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("HC18 Proxy Segmentation Dice and IoU")
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(DICE_IOU_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_measurement_error(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        warn("Skipping measurement-error figure because the summary table is empty.")
        return

    labels = summary_df["method"].tolist()
    x_values = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 5.5))
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
    ax.set_title("HC18 Proxy Fetal Head Measurement Error")
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(MEASUREMENT_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_table_markdown(summary_df: pd.DataFrame) -> None:
    report = f"""# HC18 Dice, IoU, and Fetal Measurement Metrics

{VALIDATION_NOTE}

## Source Files

{source_status_table()}

## Summary

{markdown_table(summary_df, SUMMARY_MARKDOWN_COLUMNS)}

## Outputs

- Per-image metrics: `{rel_path(PER_IMAGE_CSV)}`
- Summary metrics: `{rel_path(SUMMARY_CSV)}`
- Dice/IoU figure: `{rel_path(DICE_IOU_FIGURE)}`
- Fetal measurement error figure: `{rel_path(MEASUREMENT_FIGURE)}`

## Method Note

Each restored output is normalized to 0-255, blurred with a Gaussian kernel,
thresholded with Otsu and adaptive-threshold candidates, reduced to the largest
connected component, hole-filled, resized to the HC18 mask size, and compared
against the ground-truth fetal-head mask.
"""
    TABLE_MD.write_text(report, encoding="utf-8")


def write_review_ready_markdown(summary_df: pd.DataFrame) -> None:
    reconstruction_columns = [
        "method",
        "mean_psnr",
        "mean_ssim",
        "mean_mae",
        "mean_edge_score",
    ]
    anatomy_columns = [
        "method",
        "mean_mae_inside_mask",
        "mean_boundary_error",
        "mean_contrast_preservation",
        "mean_edge_similarity_boundary",
    ]

    report = f"""# Review-Ready Anatomy Preservation Metrics

{VALIDATION_NOTE}

## Reviewer-Requested Proxy Segmentation Metrics

{markdown_table(summary_df, SUMMARY_MARKDOWN_COLUMNS)}

## Existing HC18 Reconstruction Metrics

{existing_summary_markdown(FINAL_COMPARISON_SUMMARY, reconstruction_columns)}

## Existing Anatomy Preservation Metrics

{existing_summary_markdown(ANATOMY_PRESERVATION_SUMMARY, anatomy_columns)}

## Generated Files

- `{rel_path(PER_IMAGE_CSV)}`
- `{rel_path(SUMMARY_CSV)}`
- `{rel_path(TABLE_MD)}`
- `{rel_path(DICE_IOU_FIGURE)}`
- `{rel_path(MEASUREMENT_FIGURE)}`
"""
    REVIEW_READY_MD.write_text(report, encoding="utf-8")


def write_outputs(per_image_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    ensure_dirs([PER_IMAGE_CSV.parent, FIGURE_DIR])
    per_image_df.to_csv(PER_IMAGE_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    plot_dice_iou(summary_df)
    plot_measurement_error(summary_df)
    write_table_markdown(summary_df)
    write_review_ready_markdown(summary_df)


def main() -> int:
    per_image_df = evaluate()
    if per_image_df.empty:
        raise RuntimeError("No valid HC18 outputs were evaluated.")

    summary_df = summarize(per_image_df)
    write_outputs(per_image_df, summary_df)

    print(f"Per-image metrics: {rel_path(PER_IMAGE_CSV)}")
    print(f"Summary metrics: {rel_path(SUMMARY_CSV)}")
    print(f"Markdown table: {rel_path(TABLE_MD)}")
    print(f"Review-ready Markdown: {rel_path(REVIEW_READY_MD)}")
    print(f"Figures: {rel_path(FIGURE_DIR)}")
    print(VALIDATION_NOTE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
