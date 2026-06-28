"""
Analyze whether Quality-aware CNN + L1/MSE enhancement improves ACOUSLIC-AI
proxy frame quality and feedback decisions.

ACOUSLIC-AI has no HR target in this workflow, so this script intentionally
does not compute PSNR or SSIM. It performs a no-reference proxy-quality
before/after analysis using heuristic image-quality features.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

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
HC18_MANIFEST_PATH = PROJECT_ROOT / "data/processed/HC18_SR/hc18_lq_manifest.csv"

ENHANCED_FRAME_ROOT = (
    PROJECT_ROOT / "results/acouslic_ai/enhanced_frames_quality_aware_l1_mse"
)
PER_FRAME_CSV = PROJECT_ROOT / "results/acouslic_ai/feedback_aware_enhancement_per_frame.csv"
SUMMARY_CSV = PROJECT_ROOT / "results/acouslic_ai/feedback_aware_enhancement_summary.csv"
REPORT_MD = PROJECT_ROOT / "results/acouslic_ai/feedback_aware_enhancement_report.md"
FIGURE_DIR = PROJECT_ROOT / "results/acouslic_ai/feedback_aware_enhancement_figures"
VISUAL_EXAMPLE_DIR = FIGURE_DIR / "before_after_examples"

NO_REFERENCE_NOTE = (
    "This is a no-reference proxy-quality analysis. Labels are heuristic proxy labels, "
    "not expert clinical labels."
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
    "enhanced_frame_quality_label",
    "original_blur_score",
    "enhanced_blur_score",
    "delta_blur_score",
    "original_contrast_score",
    "enhanced_contrast_score",
    "delta_contrast_score",
    "original_edge_density",
    "enhanced_edge_density",
    "delta_edge_density",
    "original_noise_proxy",
    "enhanced_noise_proxy",
    "delta_noise_proxy",
    "improved_label_flag",
]

SUMMARY_OUTPUT_COLUMNS = [
    "scope",
    "original_frame_quality_label",
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


class FallbackQualityAwareSRModel(nn.Module):
    """
    Exact fallback copy of scripts/07_train_quality_aware_sr_hc18.py's model.

    The script first tries to import the training module via importlib. This
    fallback keeps the analysis runnable if that import fails because of an
    unrelated optional dependency.
    """

    def __init__(self, num_channels: int = 64, quality_dim: int = 5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(num_channels, 1, kernel_size=3, padding=1)

        self.film_mlp = nn.Sequential(
            nn.Linear(quality_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_channels * 2),
        )

    def forward(self, x: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
        features = F.relu(self.conv1(x))
        features = F.relu(self.conv2(features))

        gamma_beta = self.film_mlp(quality)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        features = gamma * features + beta

        features = F.relu(self.conv3(features))
        features = F.relu(self.conv4(features))
        residual = self.conv_out(features)
        out = x + residual
        out = torch.clamp(out, 0.0, 1.0)
        return out


def fallback_compute_quality_features(lq: torch.Tensor) -> torch.Tensor:
    """Exact fallback copy of the 5-D quality feature calculation from script 07."""
    if lq.dim() == 2:
        lq = lq.unsqueeze(0).unsqueeze(0)
    elif lq.dim() == 3:
        lq = lq.unsqueeze(0)

    if lq.dim() != 4 or lq.size(1) != 1:
        raise ValueError("Expected LQ tensor shape [H, W], [1, H, W], or [B, 1, H, W]")

    lq = lq.float()
    lap_kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        dtype=lq.dtype,
        device=lq.device,
    ).view(1, 1, 3, 3)
    lap = F.conv2d(lq, lap_kernel, padding=1)

    blur_score = lap.var(dim=(1, 2, 3)).view(-1)
    std_value = lq.std(dim=(1, 2, 3)).view(-1)
    mean_value = lq.mean(dim=(1, 2, 3)).view(-1)
    blurred = F.avg_pool2d(lq, kernel_size=5, stride=1, padding=2)
    noise_proxy = torch.abs(lq - blurred).mean(dim=(1, 2, 3)).view(-1)
    degradation_strength = torch.ones_like(mean_value).view(-1)

    return torch.stack(
        [blur_score, std_value, mean_value, noise_proxy, degradation_strength],
        dim=1,
    )


@dataclass
class EnhancementBundle:
    model: nn.Module
    quality_module: Any
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    device: torch.device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run no-reference feedback-aware enhancement analysis on ACOUSLIC-AI frames."
        )
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
        help="Resize frames to image_size x image_size before CNN enhancement.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional cap for quick analysis/debugging. Defaults to all frames.",
    )
    parser.add_argument(
        "--max_sweeps",
        type=int,
        default=None,
        help="Optional cap on number of sweeps. Defaults to all sweeps in the CSV.",
    )
    parser.add_argument(
        "--force_regenerate",
        action="store_true",
        help="Regenerate enhanced PNGs even if they already exist.",
    )
    parser.add_argument(
        "--max_visual_examples",
        type=int,
        default=20,
        help="Number of before/after visual examples to save.",
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
    ENHANCED_FRAME_ROOT.mkdir(parents=True, exist_ok=True)
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


def import_quality_aware_module():
    script_path = SCRIPT_DIR / "07_train_quality_aware_sr_hc18.py"
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    try:
        module = import_module_from_script("quality_aware_sr_hc18_train", script_path)
        return module
    except Exception as exc:
        print(
            "WARNING: Could not import scripts/07_train_quality_aware_sr_hc18.py "
            f"directly ({exc}). Using an exact local fallback architecture."
        )

        class FallbackModule:
            QualityAwareSRModel = FallbackQualityAwareSRModel
            compute_quality_features = staticmethod(fallback_compute_quality_features)

        return FallbackModule()


def import_proxy_quality_module():
    script_path = SCRIPT_DIR / "24_compute_acouslic_proxy_quality.py"
    return import_module_from_script("acouslic_proxy_quality_features", script_path)


def load_checkpoint_state(checkpoint_path: Path, device: torch.device) -> tuple[dict[str, torch.Tensor], Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_key = key[7:] if str(key).startswith("module.") else key
        cleaned_state_dict[cleaned_key] = value

    return cleaned_state_dict, checkpoint


def checkpoint_tensor(checkpoint: Any, key: str) -> torch.Tensor | None:
    if isinstance(checkpoint, dict) and key in checkpoint:
        value = checkpoint[key]
        if isinstance(value, torch.Tensor):
            return value.detach().float()
        return torch.as_tensor(value, dtype=torch.float32)
    return None


def load_quality_feature_stats(
    quality_module: Any,
    checkpoint: Any,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Recover the same quality-feature normalization used by the HC18 model.

    The original training script saved a plain state_dict, so feature stats are
    usually not present in the checkpoint. In that case, recompute them from the
    HC18 training split to keep the FiLM conditioning consistent.
    """
    feature_mean = checkpoint_tensor(checkpoint, "feature_mean")
    feature_std = checkpoint_tensor(checkpoint, "feature_std")
    if feature_mean is not None and feature_std is not None:
        print("Loaded quality feature normalization from checkpoint.")
        return feature_mean.to(device), feature_std.to(device)

    if hasattr(quality_module, "compute_normalization_stats") and hasattr(
        quality_module, "HC18SRDataset"
    ):
        try:
            print("Recomputing HC18 quality feature normalization from the training split...")
            base_train = quality_module.HC18SRDataset(
                csv_path=str(HC18_MANIFEST_PATH),
                input_type="LQ_mixed",
                split="train",
                return_mask=False,
            )
            feature_mean, feature_std = quality_module.compute_normalization_stats(base_train)
            feature_std = torch.where(feature_std == 0.0, torch.ones_like(feature_std), feature_std)
            return feature_mean.float().to(device), feature_std.float().to(device)
        except Exception as exc:
            print(
                "WARNING: Could not recompute HC18 feature normalization "
                f"({exc}). Falling back to mean=0, std=1."
            )
    else:
        print(
            "WARNING: Imported quality module does not expose HC18 normalization helpers. "
            "Falling back to mean=0, std=1."
        )

    return torch.zeros(5, dtype=torch.float32, device=device), torch.ones(
        5, dtype=torch.float32, device=device
    )


def build_enhancement_bundle(checkpoint_path: Path, device: torch.device) -> EnhancementBundle:
    quality_module = import_quality_aware_module()
    state_dict, checkpoint = load_checkpoint_state(checkpoint_path, device)

    model = quality_module.QualityAwareSRModel().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    feature_mean, feature_std = load_quality_feature_stats(
        quality_module=quality_module,
        checkpoint=checkpoint,
        device=device,
    )

    return EnhancementBundle(
        model=model,
        quality_module=quality_module,
        feature_mean=feature_mean,
        feature_std=feature_std,
        device=device,
    )


def read_grayscale_normalized(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0
    return np.clip(image, 0.0, 1.0)


def resize_frame(image: np.ndarray, image_size: int) -> np.ndarray:
    if image_size <= 0:
        return image.astype(np.float32)
    if image.shape == (image_size, image_size):
        return image.astype(np.float32)
    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32)


def save_grayscale_png(image: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_uint8 = np.clip(image * 255.0, 0.0, 255.0).round().astype(np.uint8)
    ok = cv2.imwrite(str(output_path), image_uint8)
    if not ok:
        raise IOError(f"Failed to save PNG: {output_path}")


def enhance_frame(
    original_resized: np.ndarray,
    bundle: EnhancementBundle,
) -> np.ndarray:
    input_tensor = torch.from_numpy(original_resized).float().unsqueeze(0).unsqueeze(0)
    input_tensor = input_tensor.to(bundle.device)

    with torch.no_grad():
        raw_quality = bundle.quality_module.compute_quality_features(input_tensor)
        feature_mean = bundle.feature_mean.to(device=bundle.device, dtype=raw_quality.dtype).view(1, -1)
        feature_std = bundle.feature_std.to(device=bundle.device, dtype=raw_quality.dtype).view(1, -1)
        feature_std = torch.where(feature_std == 0.0, torch.ones_like(feature_std), feature_std)
        normalized_quality = (raw_quality - feature_mean) / feature_std
        enhanced = bundle.model(input_tensor, normalized_quality)

    enhanced_np = enhanced.squeeze().detach().cpu().float().numpy()
    return np.clip(enhanced_np, 0.0, 1.0).astype(np.float32)


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


def improvement_flag(original_label: str, enhanced_label: str) -> int:
    return int(label_severity(enhanced_label) < label_severity(original_label))


def enhancement_output_path(sweep_id: object, frame_id: object) -> Path:
    return ENHANCED_FRAME_ROOT / safe_name(sweep_id) / f"{safe_name(frame_id)}.png"


def process_frames(
    frame_df: pd.DataFrame,
    proxy_module: Any,
    checkpoint_path: Path,
    device: torch.device,
    image_size: int,
    force_regenerate: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    internal_rows: list[dict[str, object]] = []
    original_feature_rows: list[dict[str, float]] = []
    bundle: EnhancementBundle | None = None

    total_frames = len(frame_df)
    for row_number, row in enumerate(frame_df.itertuples(index=False), start=1):
        if row_number % 100 == 0 or row_number == total_frames:
            print(f"Processed {row_number} / {total_frames} frames")

        sweep_id = getattr(row, "sweep_id")
        frame_id = getattr(row, "frame_id")
        frame_index = getattr(row, "frame_index")
        frame_path = resolve_project_path(getattr(row, "frame_path"))
        output_path = enhancement_output_path(sweep_id, frame_id)

        original_image = read_grayscale_normalized(frame_path)
        if original_image is None:
            print(f"WARNING: Could not read frame, skipping: {frame_path}")
            continue

        original_resized = resize_frame(original_image, image_size=image_size)
        original_features = features_for_image(original_resized, proxy_module)

        if output_path.exists() and not force_regenerate:
            enhanced_image = read_grayscale_normalized(output_path)
            if enhanced_image is None:
                print(f"WARNING: Existing enhanced frame is unreadable, regenerating: {output_path}")
            else:
                enhanced_image = resize_frame(enhanced_image, image_size=image_size)

        else:
            enhanced_image = None

        if enhanced_image is None:
            if bundle is None:
                if not checkpoint_path.exists():
                    raise FileNotFoundError(f"Enhancement checkpoint not found: {checkpoint_path}")
                print(f"Using device: {device}")
                bundle = build_enhancement_bundle(checkpoint_path, device)
            enhanced_image = enhance_frame(original_resized, bundle)
            save_grayscale_png(enhanced_image, output_path)

        enhanced_features = features_for_image(enhanced_image, proxy_module)

        original_feature_rows.append(original_features)
        internal_rows.append(
            {
                "sweep_id": sweep_id,
                "frame_id": frame_id,
                "frame_index": int(frame_index),
                "frame_path": rel_path(frame_path),
                "original_frame_path": rel_path(frame_path),
                "enhanced_frame_path": rel_path(output_path),
                **{f"original_{key}": value for key, value in original_features.items()},
                **{f"enhanced_{key}": value for key, value in enhanced_features.items()},
            }
        )

    if not internal_rows:
        raise ValueError("No readable ACOUSLIC frames were processed.")

    internal_df = pd.DataFrame(internal_rows)
    original_feature_df = pd.DataFrame(original_feature_rows)
    return internal_df, original_feature_df


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
        enhanced_features = {
            "blur_score": getattr(row, "enhanced_blur_score"),
            "contrast_score": getattr(row, "enhanced_contrast_score"),
            "brightness_mean": getattr(row, "enhanced_brightness_mean"),
            "noise_proxy": getattr(row, "enhanced_noise_proxy"),
            "edge_density": getattr(row, "enhanced_edge_density"),
            "dark_ratio": getattr(row, "enhanced_dark_ratio"),
            "bright_ratio": getattr(row, "enhanced_bright_ratio"),
        }

        original_label = label_with_thresholds(original_features, thresholds)
        enhanced_label = label_with_thresholds(enhanced_features, thresholds)

        rows.append(
            {
                "sweep_id": getattr(row, "sweep_id"),
                "frame_id": getattr(row, "frame_id"),
                "frame_index": int(getattr(row, "frame_index")),
                "frame_path": getattr(row, "frame_path"),
                "original_frame_quality_label": original_label,
                "enhanced_frame_quality_label": enhanced_label,
                "original_blur_score": float(original_features["blur_score"]),
                "enhanced_blur_score": float(enhanced_features["blur_score"]),
                "delta_blur_score": float(enhanced_features["blur_score"] - original_features["blur_score"]),
                "original_contrast_score": float(original_features["contrast_score"]),
                "enhanced_contrast_score": float(enhanced_features["contrast_score"]),
                "delta_contrast_score": float(
                    enhanced_features["contrast_score"] - original_features["contrast_score"]
                ),
                "original_edge_density": float(original_features["edge_density"]),
                "enhanced_edge_density": float(enhanced_features["edge_density"]),
                "delta_edge_density": float(
                    enhanced_features["edge_density"] - original_features["edge_density"]
                ),
                "original_noise_proxy": float(original_features["noise_proxy"]),
                "enhanced_noise_proxy": float(enhanced_features["noise_proxy"]),
                "delta_noise_proxy": float(
                    enhanced_features["noise_proxy"] - original_features["noise_proxy"]
                ),
                "improved_label_flag": improvement_flag(original_label, enhanced_label),
            }
        )

    return pd.DataFrame(rows)[PER_FRAME_OUTPUT_COLUMNS]


def build_summary(per_frame_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def summarize(group: pd.DataFrame, scope: str, label: str) -> dict[str, object]:
        total = int(len(group))
        improved = int(group["improved_label_flag"].sum())
        percentage = 0.0 if total == 0 else 100.0 * improved / total
        return {
            "scope": scope,
            "original_frame_quality_label": label,
            "total_frames": total,
            "number_of_frames_improved": improved,
            "percentage_improved": percentage,
            "mean_delta_blur": float(group["delta_blur_score"].mean()) if total else 0.0,
            "mean_delta_contrast": float(group["delta_contrast_score"].mean()) if total else 0.0,
            "mean_delta_edge_density": float(group["delta_edge_density"].mean()) if total else 0.0,
            "mean_delta_noise_proxy": float(group["delta_noise_proxy"].mean()) if total else 0.0,
        }

    rows.append(summarize(per_frame_df, scope="overall", label="all"))
    for label, group in per_frame_df.groupby("original_frame_quality_label", sort=True):
        rows.append(summarize(group, scope="by_original_label", label=str(label)))

    return pd.DataFrame(rows)[SUMMARY_OUTPUT_COLUMNS]


def plot_bar(
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    output_path: Path,
    color: str = "#4C78A8",
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
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_summary_figures(per_frame_df: pd.DataFrame) -> list[Path]:
    saved_paths: list[Path] = []
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
        title="Enhancement improvement rate by original frame-quality label",
        ylabel="Improved frames (%)",
        output_path=improvement_path,
        color="#4C78A8",
    )
    saved_paths.append(improvement_path)

    edge_path = FIGURE_DIR / "mean_edge_density_change_by_label.png"
    plot_bar(
        labels=labels,
        values=edge_changes,
        title="Mean edge-density change after enhancement by original label",
        ylabel="Mean delta edge density",
        output_path=edge_path,
        color="#54A24B",
    )
    saved_paths.append(edge_path)

    contrast_path = FIGURE_DIR / "mean_contrast_change_by_label.png"
    plot_bar(
        labels=labels,
        values=contrast_changes,
        title="Mean contrast change after enhancement by original label",
        ylabel="Mean delta contrast score",
        output_path=contrast_path,
        color="#F58518",
    )
    saved_paths.append(contrast_path)

    return saved_paths


def save_visual_examples(
    internal_df: pd.DataFrame,
    per_frame_df: pd.DataFrame,
    max_examples: int,
) -> list[Path]:
    if max_examples <= 0:
        return []

    visual_df = per_frame_df.merge(
        internal_df[
            [
                "sweep_id",
                "frame_id",
                "original_frame_path",
                "enhanced_frame_path",
                "original_brightness_mean",
                "enhanced_brightness_mean",
                "original_dark_ratio",
                "enhanced_dark_ratio",
                "original_bright_ratio",
                "enhanced_bright_ratio",
            ]
        ],
        on=["sweep_id", "frame_id"],
        how="left",
    )

    examples = visual_df.sort_values(
        ["improved_label_flag", "delta_edge_density", "delta_contrast_score"],
        ascending=[False, False, False],
    ).head(max_examples)

    saved_paths: list[Path] = []
    for example_index, row in enumerate(examples.itertuples(index=False), start=1):
        original_path = resolve_project_path(getattr(row, "original_frame_path"))
        enhanced_path = resolve_project_path(getattr(row, "enhanced_frame_path"))
        original_image = read_grayscale_normalized(original_path)
        enhanced_image = read_grayscale_normalized(enhanced_path)
        if original_image is None or enhanced_image is None:
            print(
                "WARNING: Could not read before/after visual pair, skipping: "
                f"{original_path}, {enhanced_path}"
            )
            continue

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original_image, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0].set_title(f"Original\n{getattr(row, 'original_frame_quality_label')}")
        axes[0].axis("off")

        axes[1].imshow(enhanced_image, cmap="gray", vmin=0.0, vmax=1.0)
        axes[1].set_title(f"Enhanced\n{getattr(row, 'enhanced_frame_quality_label')}")
        axes[1].axis("off")

        delta_text = (
            f"sweep_id: {getattr(row, 'sweep_id')}\n"
            f"frame_id: {getattr(row, 'frame_id')}\n"
            f"blur Δ: {float(getattr(row, 'delta_blur_score')):.6f}\n"
            f"contrast Δ: {float(getattr(row, 'delta_contrast_score')):.6f}\n"
            f"edge Δ: {float(getattr(row, 'delta_edge_density')):.6f}\n"
            f"noise Δ: {float(getattr(row, 'delta_noise_proxy')):.6f}"
        )
        fig.suptitle(delta_text, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.86))

        output_path = (
            VISUAL_EXAMPLE_DIR
            / f"before_after_{example_index:03d}_{safe_name(getattr(row, 'sweep_id'))}_{safe_name(getattr(row, 'frame_id'))}.png"
        )
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(output_path)

    return saved_paths


def summary_table_to_markdown(summary_df: pd.DataFrame) -> str:
    rows = [
        "| Scope | Original label | Total frames | Improved | % improved | Mean Δ blur | Mean Δ contrast | Mean Δ edge | Mean Δ noise |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_df.itertuples(index=False):
        rows.append(
            "| "
            f"{markdown_escape(getattr(row, 'scope'))} | "
            f"`{markdown_escape(getattr(row, 'original_frame_quality_label'))}` | "
            f"{int(getattr(row, 'total_frames'))} | "
            f"{int(getattr(row, 'number_of_frames_improved'))} | "
            f"{float(getattr(row, 'percentage_improved')):.2f} | "
            f"{float(getattr(row, 'mean_delta_blur')):.6f} | "
            f"{float(getattr(row, 'mean_delta_contrast')):.6f} | "
            f"{float(getattr(row, 'mean_delta_edge_density')):.6f} | "
            f"{float(getattr(row, 'mean_delta_noise_proxy')):.6f} |"
        )
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
    label_counts_before = per_frame_df["original_frame_quality_label"].value_counts().sort_index()
    label_counts_after = per_frame_df["enhanced_frame_quality_label"].value_counts().sort_index()

    before_rows = "\n".join(
        f"| `{markdown_escape(label)}` | {int(count)} |"
        for label, count in label_counts_before.items()
    )
    after_rows = "\n".join(
        f"| `{markdown_escape(label)}` | {int(count)} |"
        for label, count in label_counts_after.items()
    )
    figure_links = "\n".join(f"- `{rel_path(path)}`" for path in figure_paths)
    visual_links = "\n".join(f"- `{rel_path(path)}`" for path in visual_paths)
    if not visual_links:
        visual_links = "- _No visual examples were saved._"

    report_text = f"""# Feedback-aware Enhancement Analysis for ACOUSLIC-AI

{NO_REFERENCE_NOTE}

ACOUSLIC-AI has no HR target in this workflow, so PSNR and SSIM are not computed.
This report compares no-reference proxy quality features before and after applying
the Quality-aware CNN + L1/MSE enhancement model.

## Inputs

- Per-frame proxy-quality manifest: `{rel_path(frame_manifest_path)}`
- Per-sweep proxy-quality summary: `{rel_path(sweep_summary_path)}`
- Enhancement checkpoint: `{rel_path(checkpoint_path)}`

## Outputs

- Enhanced frames: `{rel_path(ENHANCED_FRAME_ROOT)}`
- Per-frame CSV: `{rel_path(PER_FRAME_CSV)}`
- Summary CSV: `{rel_path(SUMMARY_CSV)}`
- Figure folder: `{rel_path(FIGURE_DIR)}`

## Overall result

- Total frames analyzed: {int(overall["total_frames"])}
- Number of frames improved by proxy label: {int(overall["number_of_frames_improved"])}
- Percentage improved: {float(overall["percentage_improved"]):.2f}%
- Total sweeps in input summary: {len(sweep_summary_df)}

## Quantile thresholds derived from original frames

- blur_score < {thresholds["blur_q25"]:.8f}: `blurry_frame`
- contrast_score < {thresholds["contrast_q25"]:.8f}: `low_contrast_frame`
- brightness_mean < {thresholds["brightness_q10"]:.8f}: `too_dark_frame`
- noise_proxy > {thresholds["noise_q75"]:.8f}: `noisy_frame`
- Multiple bad conditions: `poor_frame`

## Original frame-quality label distribution

| Label | Count |
|---|---:|
{before_rows}

## Enhanced frame-quality label distribution

| Label | Count |
|---|---:|
{after_rows}

## Summary by original label

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

    frame_df = pd.read_csv(
        frame_manifest_path,
        dtype={
            "sweep_id": "string",
            "frame_id": "string",
            "frame_path": "string",
        },
    )
    sweep_summary_df = pd.read_csv(sweep_summary_path)
    validate_columns(
        frame_df,
        ["sweep_id", "frame_id", "frame_index", "frame_path"],
        frame_manifest_path,
    )

    full_manifest_frames = len(frame_df)
    full_manifest_sweeps = int(frame_df["sweep_id"].nunique())

    if args.max_sweeps is not None:
        selected_sweeps = (
            frame_df["sweep_id"]
            .drop_duplicates()
            .head(args.max_sweeps)
            .tolist()
        )
        frame_df = frame_df[frame_df["sweep_id"].isin(selected_sweeps)].copy()
    if args.max_frames is not None:
        frame_df = frame_df.head(args.max_frames).copy()

    selected_frames = len(frame_df)
    selected_sweeps = int(frame_df["sweep_id"].nunique())
    if args.max_frames is None and args.max_sweeps is None:
        print(
            "Processing full ACOUSLIC proxy-quality manifest: "
            f"{full_manifest_frames} frames across {full_manifest_sweeps} sweeps"
        )
    else:
        print(
            "Processing selected ACOUSLIC proxy-quality rows: "
            f"{selected_frames} / {full_manifest_frames} frames across "
            f"{selected_sweeps} / {full_manifest_sweeps} sweeps"
        )

    proxy_module = import_proxy_quality_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    internal_df, original_feature_df = process_frames(
        frame_df=frame_df,
        proxy_module=proxy_module,
        checkpoint_path=checkpoint_path,
        device=device,
        image_size=args.image_size,
        force_regenerate=args.force_regenerate,
    )
    if len(internal_df) != selected_frames:
        print(
            "WARNING: Processed frame count differs from selected CSV rows: "
            f"{len(internal_df)} processed vs {selected_frames} selected. "
            "Unreadable frame paths may have been skipped."
        )

    thresholds = compute_label_thresholds(original_feature_df)
    per_frame_df = build_per_frame_output(internal_df, thresholds)
    summary_df = build_summary(per_frame_df)

    per_frame_df.to_csv(PER_FRAME_CSV, index=False, float_format="%.8f")
    summary_df.to_csv(SUMMARY_CSV, index=False, float_format="%.8f")

    figure_paths = create_summary_figures(per_frame_df)
    visual_paths = save_visual_examples(
        internal_df=internal_df,
        per_frame_df=per_frame_df,
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
        ENHANCED_FRAME_ROOT,
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
    improvement_percentage = (
        0.0
        if total_frames_processed == 0
        else 100.0 * improved_frames / total_frames_processed
    )

    print(f"Total frames processed: {total_frames_processed}")
    print(f"Total sweeps processed: {total_sweeps_processed}")
    print(f"Number of improved frames: {improved_frames}")
    print(
        "Percentage improved: "
        f"{improvement_percentage:.2f}%"
    )
    print(NO_REFERENCE_NOTE)
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
