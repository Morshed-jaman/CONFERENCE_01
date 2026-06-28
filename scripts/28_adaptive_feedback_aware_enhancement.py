"""
Adaptive feedback-aware enhancement for ACOUSLIC-AI frames.

Instead of blindly applying SR enhancement to every frame, this script uses
proxy acquisition-quality labels to decide whether to keep the frame unchanged,
enhance likely blur/visibility issues, denoise cautiously, or mark a frame as
not enhancement-suitable.

This is an adaptive no-reference proxy-quality analysis. Labels are heuristic
proxy labels, not expert clinical labels.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"

FRAME_MANIFEST_PATH = (
    PROJECT_ROOT / "data/processed/ACOUSLIC_AI/acouslic_proxy_quality_manifest.csv"
)
SWEEP_SUMMARY_PATH = (
    PROJECT_ROOT / "data/processed/ACOUSLIC_AI/acouslic_sweep_quality_summary.csv"
)
CHECKPOINT_PATH = (
    PROJECT_ROOT / "results/models/quality_aware_sr_hc18_l1_mse_best.pth"
)

ADAPTIVE_FRAME_ROOT = PROJECT_ROOT / "results/acouslic_ai/adaptive_enhanced_frames"
PER_FRAME_CSV = PROJECT_ROOT / "results/acouslic_ai/adaptive_feedback_enhancement_per_frame.csv"
SUMMARY_CSV = PROJECT_ROOT / "results/acouslic_ai/adaptive_feedback_enhancement_summary.csv"
REPORT_MD = PROJECT_ROOT / "results/acouslic_ai/adaptive_feedback_enhancement_report.md"
FIGURE_DIR = PROJECT_ROOT / "results/acouslic_ai/adaptive_feedback_enhancement_figures"
VISUAL_EXAMPLE_DIR = FIGURE_DIR / "before_after_examples"

ADAPTIVE_NOTE = (
    "This is an adaptive no-reference proxy-quality analysis. Labels are heuristic "
    "proxy labels, not expert clinical labels."
)

FEATURE_COLUMNS = [
    "blur_score",
    "contrast_score",
    "brightness_mean",
    "noise_proxy",
    "edge_density",
    "dark_ratio",
    "bright_ratio",
]

PER_FRAME_OUTPUT_COLUMNS = [
    "sweep_id",
    "frame_id",
    "frame_index",
    "frame_path",
    "original_frame_quality_label",
    "adaptive_frame_quality_label",
    "action_taken",
    "original_blur_score",
    "adaptive_blur_score",
    "delta_blur_score",
    "original_contrast_score",
    "adaptive_contrast_score",
    "delta_contrast_score",
    "original_edge_density",
    "adaptive_edge_density",
    "delta_edge_density",
    "original_noise_proxy",
    "adaptive_noise_proxy",
    "delta_noise_proxy",
    "improved_label_flag",
]

SUMMARY_COLUMNS = [
    "scope",
    "original_frame_quality_label",
    "action_taken",
    "total_frames",
    "number_of_frames_improved",
    "percentage_improved",
    "mean_delta_blur",
    "mean_delta_contrast",
    "mean_delta_edge_density",
    "mean_delta_noise_proxy",
]

LABEL_SEVERITY = {
    "good_frame": 0,
    "blurry_frame": 1,
    "low_contrast_frame": 1,
    "too_dark_frame": 1,
    "noisy_frame": 1,
    "poor_frame": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run adaptive feedback-aware enhancement on ACOUSLIC-AI frames."
    )
    parser.add_argument(
        "--frame_manifest",
        type=Path,
        default=FRAME_MANIFEST_PATH,
        help="Input ACOUSLIC per-frame proxy-quality manifest.",
    )
    parser.add_argument(
        "--sweep_summary",
        type=Path,
        default=SWEEP_SUMMARY_PATH,
        help="Input ACOUSLIC per-sweep proxy-quality summary.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=CHECKPOINT_PATH,
        help="Quality-aware CNN + L1/MSE checkpoint path.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Resize to image_size x image_size before CNN enhancement.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional cap for quick debugging. Defaults to all frames.",
    )
    parser.add_argument(
        "--max_sweeps",
        type=int,
        default=None,
        help="Optional cap for quick debugging. Defaults to all sweeps.",
    )
    parser.add_argument(
        "--max_visual_examples",
        type=int,
        default=20,
        help="Number of before/after examples to save.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path | str) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def rel_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def safe_name(value: object) -> str:
    text = str(value)
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    return safe.strip("._") or "item"


def markdown_escape(value: object) -> str:
    return str(value).replace("|", "\\|")


def ensure_output_dirs() -> None:
    ADAPTIVE_FRAME_ROOT.mkdir(parents=True, exist_ok=True)
    PER_FRAME_CSV.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    VISUAL_EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def validate_columns(df: pd.DataFrame, required_columns: list[str], path: Path) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")


def import_module_from_script(module_name: str, script_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def import_stage26_module():
    """Import Stage 26 helpers for exact model loading/enhancement."""
    return import_module_from_script(
        "feedback_aware_stage26_helpers",
        SCRIPT_DIR / "26_feedback_aware_enhancement_analysis.py",
    )


def import_proxy_quality_module():
    """Import Stage 24 feature calculation for consistent proxy features."""
    return import_module_from_script(
        "acouslic_proxy_quality_features_stage28",
        SCRIPT_DIR / "24_compute_acouslic_proxy_quality.py",
    )


def adaptive_output_path(sweep_id: object, frame_id: object) -> Path:
    return ADAPTIVE_FRAME_ROOT / safe_name(sweep_id) / f"{safe_name(frame_id)}.png"


def resize_to_shape(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if image.shape == target_shape:
        return np.clip(image, 0.0, 1.0).astype(np.float32)
    resized = cv2.resize(
        image,
        (int(target_shape[1]), int(target_shape[0])),
        interpolation=cv2.INTER_LINEAR,
    )
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


def to_uint8(image: np.ndarray) -> np.ndarray:
    image_np = np.asarray(image)
    if image_np.ndim == 3:
        image_np = np.squeeze(image_np)
    if image_np.max() <= 1.0:
        image_np = image_np * 255.0
    return np.clip(image_np, 0.0, 255.0).round().astype(np.uint8)


def mild_clahe(image: np.ndarray) -> np.ndarray:
    """Mild contrast correction for low-contrast proxy frames."""
    image_uint8 = to_uint8(image)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced_uint8 = clahe.apply(image_uint8)
    return (enhanced_uint8.astype(np.float32) / 255.0).clip(0.0, 1.0)


def mild_denoise(image: np.ndarray) -> np.ndarray:
    """Cautious denoising that avoids SR sharpening noisy frames."""
    image_uint8 = to_uint8(image)
    try:
        denoised = cv2.fastNlMeansDenoising(
            image_uint8,
            None,
            h=5,
            templateWindowSize=7,
            searchWindowSize=21,
        )
    except cv2.error:
        denoised = cv2.GaussianBlur(image_uint8, (3, 3), 0)
    return (denoised.astype(np.float32) / 255.0).clip(0.0, 1.0)


def features_for_image(image: np.ndarray, proxy_module: Any) -> dict[str, float]:
    features = proxy_module.compute_frame_features(image)
    return {column: float(features[column]) for column in FEATURE_COLUMNS}


def compute_label_thresholds(original_feature_df: pd.DataFrame) -> dict[str, float]:
    return {
        "blur_q25": float(original_feature_df["blur_score"].quantile(0.25)),
        "contrast_q25": float(original_feature_df["contrast_score"].quantile(0.25)),
        "brightness_q10": float(original_feature_df["brightness_mean"].quantile(0.10)),
        "noise_q75": float(original_feature_df["noise_proxy"].quantile(0.75)),
    }


def label_with_thresholds(row: pd.Series | dict[str, float], thresholds: dict[str, float]) -> str:
    bad_conditions: list[str] = []
    if float(row["blur_score"]) < thresholds["blur_q25"]:
        bad_conditions.append("blurry_frame")
    if float(row["contrast_score"]) < thresholds["contrast_q25"]:
        bad_conditions.append("low_contrast_frame")
    if float(row["brightness_mean"]) < thresholds["brightness_q10"]:
        bad_conditions.append("too_dark_frame")
    if float(row["noise_proxy"]) > thresholds["noise_q75"]:
        bad_conditions.append("noisy_frame")

    if len(bad_conditions) > 1:
        return "poor_frame"
    if len(bad_conditions) == 1:
        return bad_conditions[0]
    return "good_frame"


def label_severity(label: str) -> int:
    return LABEL_SEVERITY.get(str(label), 1)


def improved_label_flag(original_label: str, adaptive_label: str) -> int:
    return int(label_severity(adaptive_label) < label_severity(original_label))


def enhance_with_quality_cnn(
    image: np.ndarray,
    stage26_module: Any,
    checkpoint_path: Path,
    device: torch.device,
    bundle_holder: dict[str, Any],
    image_size: int,
) -> np.ndarray:
    if bundle_holder.get("bundle") is None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Enhancement checkpoint not found: {checkpoint_path}")
        print(f"Using device: {device}")
        bundle_holder["bundle"] = stage26_module.build_enhancement_bundle(
            checkpoint_path,
            device,
        )

    model_input = stage26_module.resize_frame(image, image_size=image_size)
    enhanced_small = stage26_module.enhance_frame(model_input, bundle_holder["bundle"])
    return resize_to_shape(enhanced_small, image.shape)


def apply_adaptive_policy(
    original_image: np.ndarray,
    source_label: str,
    stage26_module: Any,
    checkpoint_path: Path,
    device: torch.device,
    bundle_holder: dict[str, Any],
    image_size: int,
) -> tuple[np.ndarray, str]:
    """
    Decide whether to enhance, denoise, keep, or mark as not suitable.

    Poor frames are likely acquisition failures, so this script avoids forcing
    SR on them and leaves feedback to repeat/reacquire the sweep.
    """
    label = str(source_label)

    if label == "good_frame":
        return original_image.copy(), "keep_original"

    if label == "blurry_frame":
        adaptive_image = enhance_with_quality_cnn(
            image=original_image,
            stage26_module=stage26_module,
            checkpoint_path=checkpoint_path,
            device=device,
            bundle_holder=bundle_holder,
            image_size=image_size,
        )
        return adaptive_image, "enhance_blur"

    if label in {"low_contrast_frame", "too_dark_frame"}:
        clahe_image = mild_clahe(original_image)
        adaptive_image = enhance_with_quality_cnn(
            image=clahe_image,
            stage26_module=stage26_module,
            checkpoint_path=checkpoint_path,
            device=device,
            bundle_holder=bundle_holder,
            image_size=image_size,
        )
        return adaptive_image, "enhance_contrast"

    if label == "noisy_frame":
        return mild_denoise(original_image), "denoise_cautiously"

    if label == "poor_frame":
        return original_image.copy(), "feedback_repeat_or_reacquire"

    return original_image.copy(), "keep_original_unknown_label"


def process_frames(
    manifest_df: pd.DataFrame,
    stage26_module: Any,
    proxy_module: Any,
    checkpoint_path: Path,
    device: torch.device,
    image_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    bundle_holder: dict[str, Any] = {"bundle": None}
    total_frames = len(manifest_df)

    for row_number, row in enumerate(manifest_df.itertuples(index=False), start=1):
        if row_number % 100 == 0 or row_number == total_frames:
            print(f"Processed {row_number} / {total_frames} frames")

        sweep_id = getattr(row, "sweep_id")
        frame_id = getattr(row, "frame_id")
        frame_index = int(getattr(row, "frame_index"))
        frame_path = resolve_project_path(getattr(row, "frame_path"))
        source_label = str(getattr(row, "frame_quality_label"))

        original_image = stage26_module.read_grayscale_normalized(frame_path)
        if original_image is None:
            print(f"WARNING: Could not read frame, skipping: {frame_path}")
            continue

        original_features = features_for_image(original_image, proxy_module)
        adaptive_image, action_taken = apply_adaptive_policy(
            original_image=original_image,
            source_label=source_label,
            stage26_module=stage26_module,
            checkpoint_path=checkpoint_path,
            device=device,
            bundle_holder=bundle_holder,
            image_size=image_size,
        )
        adaptive_features = features_for_image(adaptive_image, proxy_module)

        output_path = adaptive_output_path(sweep_id, frame_id)
        stage26_module.save_grayscale_png(adaptive_image, output_path)

        rows.append(
            {
                "sweep_id": sweep_id,
                "frame_id": frame_id,
                "frame_index": frame_index,
                "frame_path": rel_path(frame_path),
                "source_frame_quality_label": source_label,
                "action_taken": action_taken,
                "adaptive_frame_path": rel_path(output_path),
                **{f"original_{key}": value for key, value in original_features.items()},
                **{f"adaptive_{key}": value for key, value in adaptive_features.items()},
            }
        )

    if not rows:
        raise ValueError("No readable ACOUSLIC frames were processed.")

    return pd.DataFrame(rows)


def build_per_frame_output(
    internal_df: pd.DataFrame,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for row in internal_df.itertuples(index=False):
        original_features = {
            "blur_score": getattr(row, "original_blur_score"),
            "contrast_score": getattr(row, "original_contrast_score"),
            "brightness_mean": getattr(row, "original_brightness_mean"),
            "noise_proxy": getattr(row, "original_noise_proxy"),
            "edge_density": getattr(row, "original_edge_density"),
            "dark_ratio": getattr(row, "original_dark_ratio"),
            "bright_ratio": getattr(row, "original_bright_ratio"),
        }
        adaptive_features = {
            "blur_score": getattr(row, "adaptive_blur_score"),
            "contrast_score": getattr(row, "adaptive_contrast_score"),
            "brightness_mean": getattr(row, "adaptive_brightness_mean"),
            "noise_proxy": getattr(row, "adaptive_noise_proxy"),
            "edge_density": getattr(row, "adaptive_edge_density"),
            "dark_ratio": getattr(row, "adaptive_dark_ratio"),
            "bright_ratio": getattr(row, "adaptive_bright_ratio"),
        }

        original_label = label_with_thresholds(original_features, thresholds)
        adaptive_label = label_with_thresholds(adaptive_features, thresholds)

        rows.append(
            {
                "sweep_id": getattr(row, "sweep_id"),
                "frame_id": getattr(row, "frame_id"),
                "frame_index": int(getattr(row, "frame_index")),
                "frame_path": getattr(row, "frame_path"),
                "original_frame_quality_label": original_label,
                "adaptive_frame_quality_label": adaptive_label,
                "action_taken": getattr(row, "action_taken"),
                "original_blur_score": float(original_features["blur_score"]),
                "adaptive_blur_score": float(adaptive_features["blur_score"]),
                "delta_blur_score": float(
                    adaptive_features["blur_score"] - original_features["blur_score"]
                ),
                "original_contrast_score": float(original_features["contrast_score"]),
                "adaptive_contrast_score": float(adaptive_features["contrast_score"]),
                "delta_contrast_score": float(
                    adaptive_features["contrast_score"] - original_features["contrast_score"]
                ),
                "original_edge_density": float(original_features["edge_density"]),
                "adaptive_edge_density": float(adaptive_features["edge_density"]),
                "delta_edge_density": float(
                    adaptive_features["edge_density"] - original_features["edge_density"]
                ),
                "original_noise_proxy": float(original_features["noise_proxy"]),
                "adaptive_noise_proxy": float(adaptive_features["noise_proxy"]),
                "delta_noise_proxy": float(
                    adaptive_features["noise_proxy"] - original_features["noise_proxy"]
                ),
                "improved_label_flag": improved_label_flag(original_label, adaptive_label),
            }
        )

    return pd.DataFrame(rows)[PER_FRAME_OUTPUT_COLUMNS]


def build_summary(per_frame_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def summarize(
        group: pd.DataFrame,
        scope: str,
        original_label: str,
        action: str,
    ) -> dict[str, object]:
        total = int(len(group))
        improved = int(group["improved_label_flag"].sum())
        percentage = 0.0 if total == 0 else 100.0 * improved / total
        return {
            "scope": scope,
            "original_frame_quality_label": original_label,
            "action_taken": action,
            "total_frames": total,
            "number_of_frames_improved": improved,
            "percentage_improved": percentage,
            "mean_delta_blur": float(group["delta_blur_score"].mean()) if total else 0.0,
            "mean_delta_contrast": float(group["delta_contrast_score"].mean()) if total else 0.0,
            "mean_delta_edge_density": float(group["delta_edge_density"].mean()) if total else 0.0,
            "mean_delta_noise_proxy": float(group["delta_noise_proxy"].mean()) if total else 0.0,
        }

    rows.append(summarize(per_frame_df, "overall", "all", "all"))

    for action, group in per_frame_df.groupby("action_taken", sort=True):
        rows.append(summarize(group, "by_action", "all", str(action)))

    for label, group in per_frame_df.groupby("original_frame_quality_label", sort=True):
        rows.append(summarize(group, "by_original_label", str(label), "all"))

    return pd.DataFrame(rows)[SUMMARY_COLUMNS]


def plot_bar(
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    output_path: Path,
    color: str,
    value_format: str = "{:.3f}",
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=30)
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            value_format.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_summary_figures(per_frame_df: pd.DataFrame) -> list[Path]:
    saved_paths: list[Path] = []

    action_counts = per_frame_df["action_taken"].value_counts().sort_index()
    action_path = FIGURE_DIR / "action_distribution.png"
    plot_bar(
        labels=[str(label) for label in action_counts.index],
        values=[float(value) for value in action_counts.values],
        title="Adaptive action distribution",
        ylabel="Number of frames",
        output_path=action_path,
        color="#4C78A8",
        value_format="{:.0f}",
    )
    saved_paths.append(action_path)

    grouped = per_frame_df.groupby("original_frame_quality_label", sort=True)
    labels = [str(label) for label in grouped.size().index]
    improvement_rates = [
        float(group["improved_label_flag"].mean() * 100.0) for _, group in grouped
    ]
    edge_changes = [float(group["delta_edge_density"].mean()) for _, group in grouped]
    contrast_changes = [float(group["delta_contrast_score"].mean()) for _, group in grouped]

    improvement_path = FIGURE_DIR / "improvement_rate_by_original_label.png"
    plot_bar(
        labels=labels,
        values=improvement_rates,
        title="Adaptive improvement rate by original frame-quality label",
        ylabel="Improved frames (%)",
        output_path=improvement_path,
        color="#72B7B2",
    )
    saved_paths.append(improvement_path)

    edge_path = FIGURE_DIR / "mean_edge_density_change_by_label.png"
    plot_bar(
        labels=labels,
        values=edge_changes,
        title="Mean edge-density change by original label",
        ylabel="Mean delta edge density",
        output_path=edge_path,
        color="#54A24B",
    )
    saved_paths.append(edge_path)

    contrast_path = FIGURE_DIR / "mean_contrast_change_by_label.png"
    plot_bar(
        labels=labels,
        values=contrast_changes,
        title="Mean contrast change by original label",
        ylabel="Mean delta contrast score",
        output_path=contrast_path,
        color="#F58518",
    )
    saved_paths.append(contrast_path)

    return saved_paths


def save_visual_examples(
    internal_df: pd.DataFrame,
    per_frame_df: pd.DataFrame,
    stage26_module: Any,
    max_examples: int,
) -> list[Path]:
    if max_examples <= 0:
        return []

    visual_df = per_frame_df.merge(
        internal_df[["sweep_id", "frame_id", "adaptive_frame_path"]],
        on=["sweep_id", "frame_id"],
        how="left",
    )
    examples = visual_df.sort_values(
        ["improved_label_flag", "delta_edge_density", "delta_contrast_score"],
        ascending=[False, False, False],
    ).head(max_examples)

    saved_paths: list[Path] = []
    for example_index, row in enumerate(examples.itertuples(index=False), start=1):
        original_path = resolve_project_path(getattr(row, "frame_path"))
        adaptive_path = resolve_project_path(getattr(row, "adaptive_frame_path"))
        original_image = stage26_module.read_grayscale_normalized(original_path)
        adaptive_image = stage26_module.read_grayscale_normalized(adaptive_path)
        if original_image is None or adaptive_image is None:
            print(
                "WARNING: Could not read before/after visual pair, skipping: "
                f"{original_path}, {adaptive_path}"
            )
            continue

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original_image, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0].set_title(f"Original\n{getattr(row, 'original_frame_quality_label')}")
        axes[0].axis("off")

        axes[1].imshow(adaptive_image, cmap="gray", vmin=0.0, vmax=1.0)
        axes[1].set_title(
            f"Adaptive\n{getattr(row, 'adaptive_frame_quality_label')}\n"
            f"{getattr(row, 'action_taken')}"
        )
        axes[1].axis("off")

        title = (
            f"sweep_id: {getattr(row, 'sweep_id')} | frame_id: {getattr(row, 'frame_id')}\n"
            f"blur Δ: {float(getattr(row, 'delta_blur_score')):.6f} | "
            f"contrast Δ: {float(getattr(row, 'delta_contrast_score')):.6f} | "
            f"edge Δ: {float(getattr(row, 'delta_edge_density')):.6f}"
        )
        fig.suptitle(title, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.88))

        output_path = (
            VISUAL_EXAMPLE_DIR
            / (
                f"adaptive_before_after_{example_index:03d}_"
                f"{safe_name(getattr(row, 'sweep_id'))}_{safe_name(getattr(row, 'frame_id'))}.png"
            )
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(output_path)

    return saved_paths


def summary_table_to_markdown(summary_df: pd.DataFrame) -> str:
    rows = [
        "| Scope | Original label | Action | Total frames | Improved | % improved | Mean Δ blur | Mean Δ contrast | Mean Δ edge | Mean Δ noise |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_df.itertuples(index=False):
        rows.append(
            "| "
            f"{markdown_escape(getattr(row, 'scope'))} | "
            f"`{markdown_escape(getattr(row, 'original_frame_quality_label'))}` | "
            f"`{markdown_escape(getattr(row, 'action_taken'))}` | "
            f"{int(getattr(row, 'total_frames'))} | "
            f"{int(getattr(row, 'number_of_frames_improved'))} | "
            f"{float(getattr(row, 'percentage_improved')):.2f} | "
            f"{float(getattr(row, 'mean_delta_blur')):.6f} | "
            f"{float(getattr(row, 'mean_delta_contrast')):.6f} | "
            f"{float(getattr(row, 'mean_delta_edge_density')):.6f} | "
            f"{float(getattr(row, 'mean_delta_noise_proxy')):.6f} |"
        )
    return "\n".join(rows)


def counts_to_markdown(counts: pd.Series) -> str:
    rows = ["| Item | Count |", "|---|---:|"]
    for item, count in counts.items():
        rows.append(f"| `{markdown_escape(item)}` | {int(count)} |")
    if len(rows) == 2:
        rows.append("| _none_ | 0 |")
    return "\n".join(rows)


def write_report(
    per_frame_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    sweep_summary_df: pd.DataFrame,
    thresholds: dict[str, float],
    figure_paths: list[Path],
    visual_paths: list[Path],
    frame_manifest_path: Path,
    sweep_summary_path: Path,
    checkpoint_path: Path,
) -> None:
    overall = summary_df[summary_df["scope"] == "overall"].iloc[0]
    action_counts = per_frame_df["action_taken"].value_counts().sort_index()
    original_counts = per_frame_df["original_frame_quality_label"].value_counts().sort_index()
    adaptive_counts = per_frame_df["adaptive_frame_quality_label"].value_counts().sort_index()

    figure_links = "\n".join(f"- `{rel_path(path)}`" for path in figure_paths)
    visual_links = "\n".join(f"- `{rel_path(path)}`" for path in visual_paths)
    if not visual_links:
        visual_links = "- _No visual examples were saved._"

    report_text = f"""# Adaptive Feedback-aware Enhancement Analysis

{ADAPTIVE_NOTE}

This stage avoids blind enhancement. It keeps good frames unchanged, enhances
likely blurry/low-contrast frames, cautiously denoises noisy frames, and leaves
poor acquisition-failure frames for repeat/reacquisition feedback.

## Inputs

- Per-frame proxy-quality manifest: `{rel_path(frame_manifest_path)}`
- Per-sweep proxy-quality summary: `{rel_path(sweep_summary_path)}`
- Quality-aware CNN + L1/MSE checkpoint: `{rel_path(checkpoint_path)}`

## Outputs

- Adaptive frames: `{rel_path(ADAPTIVE_FRAME_ROOT)}`
- Per-frame CSV: `{rel_path(PER_FRAME_CSV)}`
- Summary CSV: `{rel_path(SUMMARY_CSV)}`
- Figure folder: `{rel_path(FIGURE_DIR)}`

## Overall result

- Total frames processed: {int(overall["total_frames"])}
- Total sweeps in input summary: {len(sweep_summary_df)}
- Number of frames improved by proxy label: {int(overall["number_of_frames_improved"])}
- Percentage improved: {float(overall["percentage_improved"]):.2f}%

## Adaptive policy

| Original proxy label | Action |
|---|---|
| `good_frame` | `keep_original` |
| `blurry_frame` | `enhance_blur` |
| `low_contrast_frame` | `enhance_contrast` using mild CLAHE then Quality-aware CNN |
| `noisy_frame` | `denoise_cautiously` without SR sharpening |
| `poor_frame` | `feedback_repeat_or_reacquire` |

## Original-feature thresholds used for both original and adaptive labels

- blur_score < {thresholds["blur_q25"]:.8f}: `blurry_frame`
- contrast_score < {thresholds["contrast_q25"]:.8f}: `low_contrast_frame`
- brightness_mean < {thresholds["brightness_q10"]:.8f}: `too_dark_frame`
- noise_proxy > {thresholds["noise_q75"]:.8f}: `noisy_frame`
- Multiple bad conditions: `poor_frame`

## Action counts

{counts_to_markdown(action_counts)}

## Original label distribution

{counts_to_markdown(original_counts)}

## Adaptive label distribution

{counts_to_markdown(adaptive_counts)}

## Summary table

{summary_table_to_markdown(summary_df)}

## Saved figures

{figure_links}

## Visual before/after examples

{visual_links}
"""

    REPORT_MD.write_text(report_text, encoding="utf-8")


def run(args: argparse.Namespace) -> list[Path]:
    if args.image_size < 0:
        raise ValueError("--image_size must be 0 or greater.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max_frames must be greater than 0 when provided.")
    if args.max_sweeps is not None and args.max_sweeps <= 0:
        raise ValueError("--max_sweeps must be greater than 0 when provided.")
    if args.max_visual_examples < 0:
        raise ValueError("--max_visual_examples must be 0 or greater.")

    ensure_output_dirs()

    frame_manifest_path = resolve_project_path(args.frame_manifest)
    sweep_summary_path = resolve_project_path(args.sweep_summary)
    checkpoint_path = resolve_project_path(args.checkpoint_path)

    if not frame_manifest_path.exists():
        raise FileNotFoundError(f"Frame proxy-quality manifest not found: {frame_manifest_path}")
    if not sweep_summary_path.exists():
        raise FileNotFoundError(f"Sweep proxy-quality summary not found: {sweep_summary_path}")

    manifest_df = pd.read_csv(
        frame_manifest_path,
        dtype={
            "sweep_id": str,
            "frame_id": str,
            "frame_path": str,
            "frame_quality_label": str,
        },
    )
    sweep_summary_df = pd.read_csv(sweep_summary_path, dtype={"sweep_id": str})
    validate_columns(
        manifest_df,
        ["sweep_id", "frame_id", "frame_index", "frame_path", "frame_quality_label"],
        frame_manifest_path,
    )

    full_frames = len(manifest_df)
    full_sweeps = int(manifest_df["sweep_id"].nunique())

    if args.max_sweeps is not None:
        selected_sweeps = manifest_df["sweep_id"].drop_duplicates().head(args.max_sweeps)
        manifest_df = manifest_df[manifest_df["sweep_id"].isin(selected_sweeps)].copy()
    if args.max_frames is not None:
        manifest_df = manifest_df.head(args.max_frames).copy()

    print(
        "Processing ACOUSLIC adaptive enhancement rows: "
        f"{len(manifest_df)} / {full_frames} frames across "
        f"{manifest_df['sweep_id'].nunique()} / {full_sweeps} sweeps"
    )

    stage26_module = import_stage26_module()
    proxy_module = import_proxy_quality_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    internal_df = process_frames(
        manifest_df=manifest_df,
        stage26_module=stage26_module,
        proxy_module=proxy_module,
        checkpoint_path=checkpoint_path,
        device=device,
        image_size=args.image_size,
    )
    if len(internal_df) != len(manifest_df):
        print(
            "WARNING: Processed frame count differs from selected CSV rows: "
            f"{len(internal_df)} processed vs {len(manifest_df)} selected."
        )

    original_feature_df = internal_df[
        [f"original_{column}" for column in FEATURE_COLUMNS]
    ].copy()
    original_feature_df.columns = FEATURE_COLUMNS
    thresholds = compute_label_thresholds(original_feature_df)

    per_frame_df = build_per_frame_output(internal_df, thresholds)
    summary_df = build_summary(per_frame_df)

    per_frame_df.to_csv(PER_FRAME_CSV, index=False, float_format="%.8f")
    summary_df.to_csv(SUMMARY_CSV, index=False, float_format="%.8f")

    figure_paths = create_summary_figures(per_frame_df)
    visual_paths = save_visual_examples(
        internal_df=internal_df,
        per_frame_df=per_frame_df,
        stage26_module=stage26_module,
        max_examples=args.max_visual_examples,
    )
    write_report(
        per_frame_df=per_frame_df,
        summary_df=summary_df,
        sweep_summary_df=sweep_summary_df,
        thresholds=thresholds,
        figure_paths=figure_paths,
        visual_paths=visual_paths,
        frame_manifest_path=frame_manifest_path,
        sweep_summary_path=sweep_summary_path,
        checkpoint_path=checkpoint_path,
    )

    saved_paths = [
        ADAPTIVE_FRAME_ROOT,
        PER_FRAME_CSV,
        SUMMARY_CSV,
        REPORT_MD,
        FIGURE_DIR,
        VISUAL_EXAMPLE_DIR,
        *figure_paths,
        *visual_paths,
    ]

    total_frames_processed = int(len(per_frame_df))
    total_sweeps_processed = int(per_frame_df["sweep_id"].nunique())
    improved_frames = int(per_frame_df["improved_label_flag"].sum())
    percentage_improved = (
        0.0
        if total_frames_processed == 0
        else 100.0 * improved_frames / total_frames_processed
    )

    print(f"Total frames processed: {total_frames_processed}")
    print(f"Total sweeps processed: {total_sweeps_processed}")
    print("Action counts:")
    for action, count in per_frame_df["action_taken"].value_counts().sort_index().items():
        print(f"  {action}: {int(count)}")
    print(f"Percentage improved: {percentage_improved:.2f}%")
    print(ADAPTIVE_NOTE)
    print("Saved output paths:")
    for path in saved_paths:
        print(f"  {rel_path(path)}")

    return saved_paths


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except (
        FileNotFoundError,
        ValueError,
        ImportError,
        RuntimeError,
        IOError,
        pd.errors.EmptyDataError,
        cv2.error,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
