"""Create the final HC18 comparison across CNN and diffusion SR methods.

The script evaluates saved grayscale outputs on the manifest's test split. A
missing method output is reported and skipped only for that method and sample.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(".")
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "HC18_SR" / "hc18_lq_manifest.csv"

PER_IMAGE_PATH = PROJECT_ROOT / "results" / "final_all_hc18_comparison_per_image.csv"
SUMMARY_PATH = PROJECT_ROOT / "results" / "final_all_hc18_comparison_summary.csv"
MARKDOWN_PATH = PROJECT_ROOT / "results" / "final_all_hc18_comparison_table.md"
VISUAL_DIR = PROJECT_ROOT / "results" / "final_all_visual_comparison"


# Ordered exactly as the final HC18 comparison table should appear.
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

METHOD_LABELS = {method["key"]: method["label"] for method in METHODS}
SUMMARY_COLUMNS = [
    "method",
    "mean_psnr",
    "std_psnr",
    "mean_ssim",
    "std_ssim",
    "mean_mae",
    "std_mae",
    "mean_edge_score",
]

# The five requested inputs/outputs plus HR form a six-panel comparison.
VISUAL_METHODS = [
    ("lq_mixed", "LQ input"),
    ("quality_aware_l1_mse", "Quality-aware CNN + L1/MSE"),
    ("diffusion_sr", "Conditional Diffusion SR"),
    ("quality_aware_diffusion_sr_v2", "Quality-aware Diffusion SR v2"),
    ("cnn_residual_diffusion_sr", "CNN-Residual Diffusion SR"),
]


def warn(message: str) -> None:
    print(f"WARNING: {message}")


def resolve_manifest_path(value, manifest_dir: Path) -> Optional[Path]:
    """Resolve absolute or manifest-relative image paths."""
    if pd.isna(value) or not str(value).strip():
        return None
    path = Path(str(value))
    return path if path.is_absolute() else manifest_dir / path


def load_grayscale(path: Optional[Path], description: str) -> Optional[np.ndarray]:
    """Load a grayscale image as a normalized two-dimensional float array."""
    if path is None or not path.exists():
        warn(f"Missing {description}: {path}; skipping this sample for that method.")
        return None

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        warn(f"Could not read {description}: {path}; skipping this sample for that method.")
        return None

    image = np.squeeze(image)
    if image.ndim != 2:
        warn(f"Expected a 2D grayscale {description}, got shape {image.shape}: {path}")
        return None
    return image.astype(np.float32) / 255.0


def resize_reference(reference: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
    """Match the HR reference to a method's saved output resolution."""
    if reference.shape == output_shape:
        return reference

    target_height, target_width = output_shape
    shrinking = target_height < reference.shape[0] or target_width < reference.shape[1]
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    resized = cv2.resize(reference, (target_width, target_height), interpolation=interpolation)
    return np.clip(np.squeeze(resized), 0.0, 1.0).astype(np.float32)


def compute_edge_similarity(output: np.ndarray, target: np.ndarray) -> float:
    """Return cosine similarity between Sobel edge-magnitude maps."""
    output = np.squeeze(output).astype(np.float32)
    target = np.squeeze(target).astype(np.float32)

    output_dx = cv2.Sobel(output, cv2.CV_32F, 1, 0, ksize=3)
    output_dy = cv2.Sobel(output, cv2.CV_32F, 0, 1, ksize=3)
    target_dx = cv2.Sobel(target, cv2.CV_32F, 1, 0, ksize=3)
    target_dy = cv2.Sobel(target, cv2.CV_32F, 0, 1, ksize=3)

    output_edges = np.sqrt(output_dx ** 2 + output_dy ** 2).ravel()
    target_edges = np.sqrt(target_dx ** 2 + target_dy ** 2).ravel()
    output_norm = np.linalg.norm(output_edges)
    target_norm = np.linalg.norm(target_edges)

    if output_norm < 1e-8 or target_norm < 1e-8:
        return 1.0 if np.allclose(output_edges, target_edges, atol=1e-6) else 0.0
    return float(np.dot(output_edges, target_edges) / (output_norm * target_norm))


def compute_metrics(output: np.ndarray, hr: np.ndarray) -> Dict[str, float]:
    """Compute all metrics from normalized two-dimensional grayscale arrays."""
    output_2d = np.clip(np.squeeze(output), 0.0, 1.0).astype(np.float32)
    hr_2d = resize_reference(np.clip(np.squeeze(hr), 0.0, 1.0), output_2d.shape)
    if output_2d.ndim != 2 or hr_2d.ndim != 2:
        raise ValueError(f"Metrics require 2D images, got {output_2d.shape} and {hr_2d.shape}")

    min_side = min(output_2d.shape)
    win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
    if win_size < 3:
        raise ValueError(f"Image is too small for SSIM: {output_2d.shape}")

    return {
        "psnr": float(peak_signal_noise_ratio(hr_2d, output_2d, data_range=1.0)),
        "ssim": float(structural_similarity(hr_2d, output_2d, data_range=1.0, win_size=win_size)),
        "mae": float(np.mean(np.abs(hr_2d - output_2d))),
        "edge_score": compute_edge_similarity(output_2d, hr_2d),
    }


def add_metric_row(
    rows: List[Dict[str, object]],
    image_id: str,
    method_key: str,
    output: np.ndarray,
    hr: np.ndarray,
) -> None:
    metrics = compute_metrics(output, hr)
    rows.append(
        {
            "image_id": image_id,
            "method": METHOD_LABELS[method_key],
            "psnr": metrics["psnr"],
            "ssim": metrics["ssim"],
            "mae": metrics["mae"],
            "edge_score": metrics["edge_score"],
        }
    )


def summarize(per_image: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for method in METHODS:
        method_rows = per_image[per_image["method"] == method["label"]]
        if method_rows.empty:
            warn(f"No valid samples were available for {method['label']}.")
            values = {column: np.nan for column in SUMMARY_COLUMNS if column != "method"}
        else:
            values = {
                "mean_psnr": method_rows["psnr"].mean(),
                "std_psnr": method_rows["psnr"].std(ddof=0),
                "mean_ssim": method_rows["ssim"].mean(),
                "std_ssim": method_rows["ssim"].std(ddof=0),
                "mean_mae": method_rows["mae"].mean(),
                "std_mae": method_rows["mae"].std(ddof=0),
                "mean_edge_score": method_rows["edge_score"].mean(),
            }
        summary_rows.append({"method": method["label"], **values})

    return pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)


def format_metric(value: float, digits: int = 6) -> str:
    return "nan" if pd.isna(value) else f"{float(value):.{digits}f}"


def build_markdown_table(summary: pd.DataFrame) -> str:
    header = "| " + " | ".join(SUMMARY_COLUMNS) + " |\n"
    separator = "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    lines = []
    for _, row in summary.iterrows():
        values = [str(row["method"])] + [format_metric(row[column]) for column in SUMMARY_COLUMNS[1:]]
        lines.append("| " + " | ".join(values) + " |")
    return header + separator + "\n".join(lines) + "\n"


def save_visual_comparison(
    image_id: str,
    images: Dict[str, np.ndarray],
    hr: np.ndarray,
    visual_index: int,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    panels = [(images[key], title) for key, title in VISUAL_METHODS]
    panels.append((hr, "HR target"))

    for axis, (image, title) in zip(axes.ravel(), panels):
        axis.imshow(np.squeeze(image), cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(title, fontsize=10)
        axis.axis("off")

    fig.suptitle(f"HC18 comparison: {image_id}", fontsize=13)
    fig.tight_layout()
    output_path = VISUAL_DIR / f"comparison_{visual_index:02d}_{image_id}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_terminal_table(summary: pd.DataFrame) -> None:
    print(
        "method, mean_psnr, std_psnr, mean_ssim, std_ssim, "
        "mean_mae, std_mae, mean_edge_score"
    )
    for _, row in summary.iterrows():
        print(
            f"{row['method']}, {format_metric(row['mean_psnr'])}, "
            f"{format_metric(row['std_psnr'])}, {format_metric(row['mean_ssim'])}, "
            f"{format_metric(row['std_ssim'])}, {format_metric(row['mean_mae'])}, "
            f"{format_metric(row['std_mae'])}, {format_metric(row['mean_edge_score'])}"
        )


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    (PROJECT_ROOT / "results").mkdir(parents=True, exist_ok=True)
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_PATH)
    test_manifest = manifest[manifest["split"] == "test"].reset_index(drop=True)
    if test_manifest.empty:
        raise ValueError(f"No test samples found in {MANIFEST_PATH}")

    manifest_dir = MANIFEST_PATH.parent
    per_image_rows: List[Dict[str, object]] = []
    visual_count = 0

    for _, sample in test_manifest.iterrows():
        image_id = str(sample["image_id"])
        hr_path = resolve_manifest_path(sample.get("hr_path"), manifest_dir)
        hr = load_grayscale(hr_path, f"HR image for {image_id}")
        if hr is None:
            warn(f"Skipping all methods for {image_id} because its HR target is unavailable.")
            continue

        lq_path = resolve_manifest_path(sample.get("lq_mixed_path"), manifest_dir)
        lq = load_grayscale(lq_path, f"LQ mixed image for {image_id}")

        images: Dict[str, np.ndarray] = {}
        if lq is not None:
            images["lq_mixed"] = lq
            add_metric_row(per_image_rows, image_id, "lq_mixed", lq, hr)

        for method in METHODS[1:]:
            output_path = method["output_dir"] / f"{image_id}.png"
            output = load_grayscale(output_path, f"{method['label']} output for {image_id}")
            if output is None:
                continue
            images[method["key"]] = output
            add_metric_row(per_image_rows, image_id, method["key"], output, hr)

        if visual_count < 10 and all(key in images for key, _ in VISUAL_METHODS):
            visual_count = visual_count + 1
            save_visual_comparison(image_id, images, hr, visual_count)

    if not per_image_rows:
        raise RuntimeError("No valid HC18 samples or model outputs were available for comparison.")

    per_image = pd.DataFrame(
        per_image_rows,
        columns=["image_id", "method", "psnr", "ssim", "mae", "edge_score"],
    )
    summary = summarize(per_image)

    per_image.to_csv(PER_IMAGE_PATH, index=False, float_format="%.6f")
    summary.to_csv(SUMMARY_PATH, index=False, float_format="%.6f")
    MARKDOWN_PATH.write_text(build_markdown_table(summary), encoding="utf-8")

    if visual_count < 10:
        warn(f"Saved {visual_count} visual comparisons; fewer than 10 samples had all required panels.")

    print_terminal_table(summary)
    print(f"Per-image results saved to: {PER_IMAGE_PATH}")
    print(f"Summary table saved to: {SUMMARY_PATH}")
    print(f"Markdown table saved to: {MARKDOWN_PATH}")
    print(f"Visual comparisons saved to: {VISUAL_DIR}")


if __name__ == "__main__":
    main()
