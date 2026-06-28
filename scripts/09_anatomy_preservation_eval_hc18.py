"""
Evaluate fetal head anatomy preservation for HC18 super-resolution outputs.

This script compares CNN and diffusion SR outputs using fetal head masks from
the HC18 test split. Missing output images are skipped per method so other
available results for the same sample remain usable.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import matplotlib
import numpy as np
import pandas as pd

# Use a non-interactive backend so the script works from Windows PowerShell.
matplotlib.use("Agg")

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(".")
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "HC18_SR" / "hc18_lq_manifest.csv"
MANIFEST_ROOT = MANIFEST_PATH.parent
MASK_DIR = MANIFEST_ROOT / "masks"

PER_IMAGE_CSV = PROJECT_ROOT / "results" / "anatomy_preservation_all_models_per_image.csv"
SUMMARY_CSV = PROJECT_ROOT / "results" / "anatomy_preservation_all_models_summary.csv"
MARKDOWN_TABLE = PROJECT_ROOT / "results" / "anatomy_preservation_all_models_table.md"
VISUAL_DIR = PROJECT_ROOT / "results" / "anatomy_preservation_all_models_visuals"


# Ordered exactly as requested for tables and terminal output.
METHODS = [
    {
        "key": "lq_mixed",
        "label": "LQ mixed baseline",
        "output_dir": None,
    },
    {
        "key": "sr_only",
        "label": "SR-only CNN",
        "output_dir": PROJECT_ROOT / "results" / "sr_outputs" / "sr_only",
    },
    {
        "key": "quality_aware",
        "label": "Quality-aware CNN",
        "output_dir": PROJECT_ROOT / "results" / "sr_outputs" / "quality_aware",
    },
    {
        "key": "quality_aware_l1_mse",
        "label": "Quality-aware CNN + L1/MSE",
        "output_dir": PROJECT_ROOT / "results" / "sr_outputs" / "quality_aware_l1_mse",
    },
    {
        "key": "diffusion_sr",
        "label": "Conditional Diffusion SR",
        "output_dir": PROJECT_ROOT / "results" / "sr_outputs" / "diffusion_sr",
    },
    {
        "key": "quality_aware_diffusion_sr_v2",
        "label": "Quality-aware Diffusion SR v2",
        "output_dir": PROJECT_ROOT / "results" / "sr_outputs" / "quality_aware_diffusion_sr_v2",
    },
    {
        "key": "cnn_residual_diffusion_sr",
        "label": "CNN-Residual Diffusion SR",
        "output_dir": PROJECT_ROOT / "results" / "sr_outputs" / "cnn_residual_diffusion_sr",
    },
]
METHOD_BY_KEY = {method["key"]: method for method in METHODS}
METHOD_LABELS = {method["key"]: method["label"] for method in METHODS}

# Only these model images are required for the requested six-panel visuals.
VISUAL_METHOD_KEYS = [
    "lq_mixed",
    "quality_aware_l1_mse",
    "diffusion_sr",
    "cnn_residual_diffusion_sr",
]


def create_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def warn(message: str) -> None:
    print(f"WARNING: {message}")


def resolve_manifest_path(path_value: object) -> Optional[Path]:
    """Resolve absolute or manifest-relative paths from hc18_lq_manifest.csv."""
    if pd.isna(path_value):
        return None

    path_text = str(path_value).strip()
    if not path_text:
        return None

    path = Path(path_text)
    if path.is_absolute():
        return path
    return MANIFEST_ROOT / path


def resolve_mask_path(row: pd.Series, image_id: str) -> Optional[Path]:
    """Use the requested masks folder, with the manifest path as fallback."""
    folder_mask = MASK_DIR / f"{image_id}.png"
    if folder_mask.exists():
        return folder_mask
    return resolve_manifest_path(row.get("mask_path"))


def read_grayscale_01(path: Path) -> Optional[np.ndarray]:
    """Read a grayscale image and normalize to float32 in [0, 1]."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return image.astype(np.float32) / 255.0


def read_mask(path: Path, target_shape: tuple[int, int]) -> Optional[np.ndarray]:
    """Read a fetal head mask as a boolean array aligned to the HR shape."""
    mask_image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask_image is None:
        return None

    if mask_image.shape != target_shape:
        mask_image = cv2.resize(
            mask_image,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    mask = mask_image > 127
    if not np.any(mask):
        return None
    return mask


def resize_to_shape(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Align images defensively if an output PNG has a different shape."""
    if image.shape == target_shape:
        return image
    resized = cv2.resize(
        image,
        (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


def boundary_ring(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Create a narrow ring around the fetal head boundary."""
    mask_uint8 = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
    eroded = cv2.erode(mask_uint8, kernel, iterations=1)
    ring = (dilated - eroded) > 0
    return ring


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    """Mean over a mask, returning NaN if the mask is empty."""
    if not np.any(mask):
        return float("nan")
    return float(np.mean(values[mask]))


def contrast_preservation(output: np.ndarray, hr: np.ndarray, mask: np.ndarray) -> float:
    """
    Compare inside-mask contrast against HR.

    The score is clipped to [0, 1], where 1 means identical standard deviation
    inside the fetal head mask.
    """
    output_values = output[mask]
    hr_values = hr[mask]
    if output_values.size == 0 or hr_values.size == 0:
        return float("nan")

    output_contrast = float(np.std(output_values))
    hr_contrast = float(np.std(hr_values))
    score = 1.0 - abs(output_contrast - hr_contrast) / max(hr_contrast, 1e-8)
    return float(np.clip(score, 0.0, 1.0))


def normalized_sobel_edges(image: np.ndarray) -> np.ndarray:
    """Compute a normalized Sobel edge magnitude map."""
    image_float = image.astype(np.float32)
    edge_x = cv2.Sobel(image_float, cv2.CV_32F, 1, 0, ksize=3)
    edge_y = cv2.Sobel(image_float, cv2.CV_32F, 0, 1, ksize=3)
    edges = cv2.magnitude(edge_x, edge_y)
    max_edge = float(np.max(edges))
    if max_edge > 1e-8:
        edges = edges / max_edge
    return edges


def edge_similarity_near_boundary(output: np.ndarray, hr: np.ndarray, ring: np.ndarray) -> float:
    """Compare normalized Sobel edge maps near the fetal head boundary."""
    output_edges = normalized_sobel_edges(output)
    hr_edges = normalized_sobel_edges(hr)
    edge_mae = masked_mean(np.abs(output_edges - hr_edges), ring)
    if np.isnan(edge_mae):
        return float("nan")
    return float(np.clip(1.0 - edge_mae, 0.0, 1.0))


def compute_anatomy_metrics(output: np.ndarray, hr: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Compute mask-aware anatomy preservation metrics for one method."""
    error = np.abs(output - hr)
    outside_mask = ~mask
    ring = boundary_ring(mask)

    return {
        "mae_inside_mask": masked_mean(error, mask),
        "mae_outside_mask": masked_mean(error, outside_mask),
        "boundary_error": masked_mean(error, ring),
        "contrast_preservation": contrast_preservation(output, hr, mask),
        "edge_similarity_boundary": edge_similarity_near_boundary(output, hr, ring),
    }


def output_path_for_method(method_key: str, image_id: str) -> Optional[Path]:
    """Return the saved SR PNG path for model methods."""
    method = METHOD_BY_KEY[method_key]
    output_dir = method["output_dir"]
    return None if output_dir is None else output_dir / f"{image_id}.png"


def load_method_images(
    row: pd.Series,
    image_id: str,
    target_shape: tuple[int, int],
) -> Dict[str, np.ndarray]:
    """Load each available method independently, warning on missing outputs."""
    images: Dict[str, np.ndarray] = {}
    lq_path = resolve_manifest_path(row.get("lq_mixed_path"))
    lq_image = read_grayscale_01(lq_path) if lq_path is not None else None
    if lq_image is None:
        warn(f"Missing LQ mixed baseline for {image_id}: {lq_path}; skipping that method.")
    else:
        images["lq_mixed"] = resize_to_shape(lq_image, target_shape)

    for method in METHODS[1:]:
        method_key = method["key"]
        output_path = output_path_for_method(method_key, image_id)
        output_image = (
            read_grayscale_01(output_path)
            if output_path is not None and output_path.exists()
            else None
        )
        if output_image is None:
            warn(
                f"Missing {method['label']} output for {image_id}: "
                f"{output_path}; skipping that method."
            )
            continue
        images[method_key] = resize_to_shape(output_image, target_shape)

    return images


def add_metric_rows(
    rows: List[Dict[str, object]],
    image_id: str,
    method_key: str,
    metrics: Dict[str, float],
) -> None:
    row = {"image_id": image_id, "method": METHOD_LABELS[method_key]}
    row.update(metrics)
    rows.append(row)


def summarize(per_image_rows: List[Dict[str, object]]) -> pd.DataFrame:
    """Aggregate rows in requested method order, retaining empty methods."""
    per_image_df = pd.DataFrame(per_image_rows)
    metric_columns = [
        "mae_inside_mask",
        "mae_outside_mask",
        "boundary_error",
        "contrast_preservation",
        "edge_similarity_boundary",
    ]
    summary_columns = [
        "method",
        "mean_mae_inside_mask",
        "std_mae_inside_mask",
        "mean_mae_outside_mask",
        "std_mae_outside_mask",
        "mean_boundary_error",
        "std_boundary_error",
        "mean_contrast_preservation",
        "std_contrast_preservation",
        "mean_edge_similarity_boundary",
        "std_edge_similarity_boundary",
    ]

    summary_rows = []
    for method in METHODS:
        method_rows = per_image_df[per_image_df["method"] == method["label"]]
        if method_rows.empty:
            warn(f"No valid samples were available for {method['label']}.")
            values = {
                f"{stat}_{metric}": float("nan")
                for metric in metric_columns
                for stat in ("mean", "std")
            }
        else:
            values = {}
            for metric in metric_columns:
                values[f"mean_{metric}"] = method_rows[metric].mean()
                values[f"std_{metric}"] = method_rows[metric].std(ddof=0)
        summary_rows.append({"method": method["label"], **values})

    return pd.DataFrame(summary_rows, columns=summary_columns)


def write_markdown_table(summary_df: pd.DataFrame, path: Path) -> None:
    columns = [
        "method",
        "mean_mae_inside_mask",
        "mean_mae_outside_mask",
        "mean_boundary_error",
        "mean_contrast_preservation",
        "mean_edge_similarity_boundary",
    ]
    markdown = (
        "| method | mean_mae_inside_mask | mean_mae_outside_mask | "
        "mean_boundary_error | mean_contrast_preservation | mean_edge_similarity_boundary |\n"
        "|---|---:|---:|---:|---:|---:|\n"
    )
    for _, row in summary_df[columns].iterrows():
        markdown += (
            f"| {row['method']} | "
            f"{row['mean_mae_inside_mask']:.6f} | "
            f"{row['mean_mae_outside_mask']:.6f} | "
            f"{row['mean_boundary_error']:.6f} | "
            f"{row['mean_contrast_preservation']:.6f} | "
            f"{row['mean_edge_similarity_boundary']:.6f} |\n"
        )
    path.write_text(markdown, encoding="utf-8")


def print_summary_table(summary_df: pd.DataFrame) -> None:
    print(
        "method, mean_mae_inside_mask, mean_mae_outside_mask, "
        "mean_boundary_error, mean_contrast_preservation, "
        "mean_edge_similarity_boundary"
    )
    for _, row in summary_df.iterrows():
        print(
            f"{row['method']}, {row['mean_mae_inside_mask']:.6f}, "
            f"{row['mean_mae_outside_mask']:.6f}, "
            f"{row['mean_boundary_error']:.6f}, "
            f"{row['mean_contrast_preservation']:.6f}, "
            f"{row['mean_edge_similarity_boundary']:.6f}"
        )


def make_mask_overlay(hr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create an RGB HR image with the fetal head mask boundary overlaid."""
    base = np.stack([hr, hr, hr], axis=-1)
    ring = boundary_ring(mask, kernel_size=3)
    overlay = base.copy()
    overlay[mask, 1] = np.maximum(overlay[mask, 1], 0.55)
    overlay[ring] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return np.clip(overlay, 0.0, 1.0)


def save_visual_figure(
    image_id: str,
    hr: np.ndarray,
    mask: np.ndarray,
    method_images: Dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Save a six-panel visual anatomy comparison figure."""
    panels = [
        ("HR image", hr, "gray"),
        ("Mask overlay", make_mask_overlay(hr, mask), None),
        ("LQ mixed baseline", method_images["lq_mixed"], "gray"),
        (
            "Quality-aware CNN + L1/MSE",
            method_images["quality_aware_l1_mse"],
            "gray",
        ),
        ("Conditional Diffusion SR", method_images["diffusion_sr"], "gray"),
        (
            "CNN-Residual Diffusion SR",
            method_images["cnn_residual_diffusion_sr"],
            "gray",
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for axis, (title, image, cmap) in zip(axes.flat, panels):
        if image.ndim == 3:
            axis.imshow(image)
        else:
            axis.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.axis("off")

    fig.suptitle(image_id)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    create_dirs([PROJECT_ROOT / "results", VISUAL_DIR])

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    manifest = pd.read_csv(MANIFEST_PATH)
    test_rows = manifest[manifest["split"] == "test"].reset_index(drop=True)
    if test_rows.empty:
        raise ValueError("No test samples found in manifest")

    per_image_rows: List[Dict[str, object]] = []
    visual_count = 0
    skipped_count = 0

    for _, row in test_rows.iterrows():
        image_id = str(row.get("image_id", "")).strip()
        if not image_id:
            skipped_count += 1
            continue

        hr_path = resolve_manifest_path(row.get("hr_path"))
        mask_path = resolve_mask_path(row, image_id)
        if hr_path is None or mask_path is None or not mask_path.exists():
            warn(f"Skipping {image_id}: HR or mask path is missing")
            skipped_count += 1
            continue

        hr = read_grayscale_01(hr_path)
        if hr is None:
            warn(f"Skipping {image_id}: HR image is missing")
            skipped_count += 1
            continue

        mask = read_mask(mask_path, hr.shape)
        if mask is None:
            warn(f"Skipping {image_id}: mask is missing or empty")
            skipped_count += 1
            continue

        method_images = load_method_images(row, image_id, hr.shape)
        if not method_images:
            warn(f"Skipping {image_id}: no method images are available")
            skipped_count += 1
            continue

        for method_key, image in method_images.items():
            metrics = compute_anatomy_metrics(image, hr, mask)
            add_metric_rows(per_image_rows, image_id, method_key, metrics)

        if (
            visual_count < 10
            and all(key in method_images for key in VISUAL_METHOD_KEYS)
        ):
            save_visual_figure(
                image_id=image_id,
                hr=hr,
                mask=mask,
                method_images=method_images,
                output_path=VISUAL_DIR / f"anatomy_{visual_count + 1:02d}_{image_id}.png",
            )
            visual_count += 1

    if not per_image_rows:
        raise RuntimeError("No anatomy preservation results were produced")

    per_image_df = pd.DataFrame(
        per_image_rows,
        columns=[
            "image_id",
            "method",
            "mae_inside_mask",
            "mae_outside_mask",
            "boundary_error",
            "contrast_preservation",
            "edge_similarity_boundary",
        ],
    )
    summary_df = summarize(per_image_rows)

    per_image_df.to_csv(PER_IMAGE_CSV, index=False, float_format="%.6f")
    summary_df.to_csv(SUMMARY_CSV, index=False, float_format="%.6f")
    write_markdown_table(summary_df, MARKDOWN_TABLE)

    print()
    print_summary_table(summary_df)
    print()
    print(f"Per-image results saved to: {PER_IMAGE_CSV}")
    print(f"Summary results saved to: {SUMMARY_CSV}")
    print(f"Markdown table saved to: {MARKDOWN_TABLE}")
    print(f"Visual figures saved to: {VISUAL_DIR}")
    if visual_count < 10:
        warn(
            f"Saved {visual_count} visual examples; fewer than 10 samples had "
            "all requested visual panels."
        )
    print(f"Skipped samples: {skipped_count}")


if __name__ == "__main__":
    main()
