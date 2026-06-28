"""
Learned adaptive feedback-aware enhancement for ACOUSLIC-AI frames.

This script uses the learned proxy-quality embedding MLP to choose a conservative
enhancement action, then evaluates before/after changes with the same
no-reference proxy feature thresholds. The labels are proxy labels, not expert
clinical labels.
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"

FRAME_MANIFEST_PATH = Path("data/processed/ACOUSLIC_AI/acouslic_proxy_quality_manifest.csv")
PROXY_MLP_CHECKPOINT_PATH = Path("results/models/proxy_quality_embedding_mlp.pth")
SR_CHECKPOINT_PATH = Path("results/models/quality_aware_sr_hc18_l1_mse_best.pth")

PER_FRAME_CSV = Path("results/acouslic_ai/learned_adaptive_enhancement_per_frame.csv")
SUMMARY_CSV = Path("results/acouslic_ai/learned_adaptive_enhancement_summary.csv")
REPORT_MD = Path("results/acouslic_ai/learned_adaptive_enhancement_report.md")
FIGURE_DIR = Path("results/acouslic_ai/learned_adaptive_enhancement_figures")
VISUAL_EXAMPLE_DIR = FIGURE_DIR / "before_after_examples"

PROXY_LABEL_NOTE = "Labels are proxy labels, not expert clinical labels."

FEATURE_COLUMNS = [
    "blur_score",
    "contrast_score",
    "brightness_mean",
    "brightness_std",
    "noise_proxy",
    "edge_density",
    "dark_ratio",
    "bright_ratio",
    "frame_difference_mean",
    "optical_flow_mean_magnitude",
    "optical_flow_std_magnitude",
    "jitter_score",
]

PROXY_FEATURE_COLUMNS = [
    "blur_score",
    "contrast_score",
    "brightness_mean",
    "noise_proxy",
    "edge_density",
    "dark_ratio",
    "bright_ratio",
]

LABELS = [
    "good_frame",
    "blurry_frame",
    "low_contrast_frame",
    "noisy_frame",
    "poor_frame",
]

LABEL_SEVERITY = {
    "good_frame": 0,
    "blurry_frame": 1,
    "low_contrast_frame": 1,
    "too_dark_frame": 1,
    "noisy_frame": 1,
    "poor_frame": 2,
}

PER_FRAME_COLUMNS = [
    "sweep_id",
    "frame_id",
    "frame_index",
    "frame_path",
    "original_proxy_label",
    "learned_predicted_label",
    "learned_confidence",
    "action_taken",
    "before_threshold_label",
    "after_threshold_label",
    "original_blur_score",
    "after_blur_score",
    "delta_blur_score",
    "original_contrast_score",
    "after_contrast_score",
    "delta_contrast_score",
    "original_edge_density",
    "after_edge_density",
    "delta_edge_density",
    "original_noise_proxy",
    "after_noise_proxy",
    "delta_noise_proxy",
    "improved_label_flag",
]

SUMMARY_COLUMNS = [
    "scope",
    "original_proxy_label",
    "action_taken",
    "total_frames",
    "number_of_frames_improved",
    "percentage_improved",
    "mean_delta_blur",
    "mean_delta_contrast",
    "mean_delta_edge_density",
    "mean_delta_noise_proxy",
]


class ProxyQualityEmbeddingMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 5):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.feature_net(x)
        logits = self.classifier(embedding)
        return logits, embedding


@dataclass
class LearnedQualityModel:
    model: ProxyQualityEmbeddingMLP
    feature_columns: list[str]
    labels: list[str]
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    device: torch.device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run learned adaptive feedback-aware enhancement on ACOUSLIC-AI frames."
        )
    )
    parser.add_argument(
        "--frame_manifest",
        type=Path,
        default=FRAME_MANIFEST_PATH,
        help="Input ACOUSLIC per-frame proxy-quality manifest.",
    )
    parser.add_argument(
        "--proxy_mlp_checkpoint",
        type=Path,
        default=PROXY_MLP_CHECKPOINT_PATH,
        help="Learned proxy-quality embedding MLP checkpoint.",
    )
    parser.add_argument(
        "--sr_checkpoint",
        type=Path,
        default=SR_CHECKPOINT_PATH,
        help="Quality-aware CNN + L1/MSE checkpoint for enhancement actions.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.60,
        help="Minimum learned-label confidence for non-conservative actions.",
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
        help="Optional cap on number of sweeps. Defaults to all sweeps.",
    )
    parser.add_argument(
        "--max_visual_examples",
        type=int,
        default=20,
        help="Number of before/after examples to save.",
    )
    return parser.parse_args()


def resolve_project_path(path_value: Path | str) -> Path:
    path = Path(path_value)
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
    resolve_project_path(PER_FRAME_CSV).parent.mkdir(parents=True, exist_ok=True)
    resolve_project_path(SUMMARY_CSV).parent.mkdir(parents=True, exist_ok=True)
    resolve_project_path(REPORT_MD).parent.mkdir(parents=True, exist_ok=True)
    resolve_project_path(FIGURE_DIR).mkdir(parents=True, exist_ok=True)
    resolve_project_path(VISUAL_EXAMPLE_DIR).mkdir(parents=True, exist_ok=True)


def import_module_from_script(module_name: str, script_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def import_stage26_module():
    return import_module_from_script(
        "feedback_aware_stage26_for_learned_adaptive",
        SCRIPT_DIR / "26_feedback_aware_enhancement_analysis.py",
    )


def import_proxy_quality_module():
    return import_module_from_script(
        "acouslic_proxy_quality_features_for_learned_adaptive",
        SCRIPT_DIR / "24_compute_acouslic_proxy_quality.py",
    )


def validate_manifest_columns(df: pd.DataFrame, path: Path) -> None:
    required = ["sweep_id", "frame_id", "frame_index", "frame_path", "frame_quality_label"]
    required.extend(FEATURE_COLUMNS)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={
            "sweep_id": str,
            "frame_id": str,
            "frame_path": str,
            "frame_quality_label": str,
        },
    )
    validate_manifest_columns(df, path)
    df = df.copy()
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if df[FEATURE_COLUMNS].isna().any().any():
        bad_columns = [column for column in FEATURE_COLUMNS if df[column].isna().any()]
        raise ValueError(f"Non-numeric or missing feature values in columns: {bad_columns}")
    return df


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        cleaned_key = str(key)[7:] if str(key).startswith("module.") else str(key)
        cleaned[cleaned_key] = value
    return cleaned


def checkpoint_array(checkpoint: Any, key: str, fallback: np.ndarray) -> np.ndarray:
    if isinstance(checkpoint, dict) and key in checkpoint:
        value = checkpoint[key]
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().float().numpy().astype(np.float32)
        return np.asarray(value, dtype=np.float32)
    return fallback.astype(np.float32)


def load_learned_quality_model(
    checkpoint_path: Path,
    device: torch.device,
) -> LearnedQualityModel:
    # weights_only=False is used because this local checkpoint stores model metadata in addition to tensor weights.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        feature_columns = list(checkpoint.get("feature_columns", FEATURE_COLUMNS))
        labels = list(checkpoint.get("label_names", LABELS))
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        feature_columns = FEATURE_COLUMNS
        labels = LABELS
    else:
        raise ValueError(f"Unsupported proxy MLP checkpoint format: {checkpoint_path}")

    if feature_columns != FEATURE_COLUMNS:
        print(
            "WARNING: Proxy MLP checkpoint feature columns differ from the current "
            "script defaults. Using checkpoint feature order."
        )
    if labels != LABELS:
        print(
            "WARNING: Proxy MLP checkpoint labels differ from the current script "
            "defaults. Using checkpoint label order."
        )

    fallback_mean = np.zeros(len(feature_columns), dtype=np.float32)
    fallback_std = np.ones(len(feature_columns), dtype=np.float32)
    feature_mean = checkpoint_array(checkpoint, "feature_mean", fallback_mean)
    feature_std = checkpoint_array(checkpoint, "feature_std", fallback_std)
    feature_std = np.where(np.abs(feature_std) < 1e-8, 1.0, feature_std).astype(np.float32)

    model = ProxyQualityEmbeddingMLP(
        input_dim=len(feature_columns),
        num_classes=len(labels),
    ).to(device)
    model.load_state_dict(clean_state_dict(state_dict))
    model.eval()

    return LearnedQualityModel(
        model=model,
        feature_columns=feature_columns,
        labels=labels,
        feature_mean=torch.as_tensor(feature_mean, dtype=torch.float32, device=device),
        feature_std=torch.as_tensor(feature_std, dtype=torch.float32, device=device),
        device=device,
    )


def predict_proxy_labels(
    df: pd.DataFrame,
    learned_model: LearnedQualityModel,
    batch_size: int = 4096,
) -> pd.DataFrame:
    missing = [column for column in learned_model.feature_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing MLP feature columns: {missing}")

    features = df[learned_model.feature_columns].to_numpy(dtype=np.float32)
    predictions: list[np.ndarray] = []
    confidences: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.as_tensor(
                features[start : start + batch_size],
                dtype=torch.float32,
                device=learned_model.device,
            )
            batch = (batch - learned_model.feature_mean) / learned_model.feature_std
            logits, _ = learned_model.model(batch)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = probabilities.max(dim=1)
            predictions.append(predicted.cpu().numpy())
            confidences.append(confidence.cpu().numpy())

    predicted_indices = np.concatenate(predictions)
    confidence_values = np.concatenate(confidences)

    result = df.copy()
    result["learned_predicted_label"] = [
        learned_model.labels[int(index)] for index in predicted_indices
    ]
    result["learned_confidence"] = confidence_values.astype(np.float32)
    return result


def to_uint8(image: np.ndarray) -> np.ndarray:
    image_np = np.asarray(image)
    if image_np.ndim == 3:
        image_np = np.squeeze(image_np)
    if image_np.max() <= 1.0:
        image_np = image_np * 255.0
    return np.clip(image_np, 0.0, 255.0).round().astype(np.uint8)


def mild_clahe(image: np.ndarray) -> np.ndarray:
    image_uint8 = to_uint8(image)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced_uint8 = clahe.apply(image_uint8)
    return (enhanced_uint8.astype(np.float32) / 255.0).clip(0.0, 1.0)


def mild_denoise(image: np.ndarray) -> np.ndarray:
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


def resize_to_shape(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if image.shape == target_shape:
        return np.clip(image, 0.0, 1.0).astype(np.float32)
    resized = cv2.resize(
        image,
        (int(target_shape[1]), int(target_shape[0])),
        interpolation=cv2.INTER_LINEAR,
    )
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


def display_resize(image: np.ndarray, max_side: int = 512) -> np.ndarray:
    height, width = image.shape[:2]
    largest = max(height, width)
    if largest <= max_side:
        return image.copy()
    scale = max_side / float(largest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


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
            raise FileNotFoundError(f"Quality-aware SR checkpoint not found: {checkpoint_path}")
        print(f"Loading Quality-aware CNN + L1/MSE on device: {device}")
        bundle_holder["bundle"] = stage26_module.build_enhancement_bundle(
            checkpoint_path,
            device,
        )

    model_input = stage26_module.resize_frame(image, image_size=image_size)
    enhanced_small = stage26_module.enhance_frame(model_input, bundle_holder["bundle"])
    return resize_to_shape(enhanced_small, image.shape)


def choose_action(
    predicted_label: str,
    confidence: float,
    original_label: str,
    confidence_threshold: float,
) -> str:
    if confidence < confidence_threshold:
        if original_label == "good_frame":
            return "keep_original_low_confidence"
        return "feedback_warning_low_confidence"

    if predicted_label == "good_frame":
        return "keep_original"
    if predicted_label == "blurry_frame":
        return "enhance_blur"
    if predicted_label == "low_contrast_frame":
        return "clahe_then_enhance"
    if predicted_label == "noisy_frame":
        return "denoise_cautiously"
    if predicted_label == "poor_frame":
        return "feedback_repeat_or_reacquire"
    return "feedback_warning_unknown_prediction"


def apply_action(
    image: np.ndarray,
    action: str,
    stage26_module: Any,
    sr_checkpoint_path: Path,
    device: torch.device,
    bundle_holder: dict[str, Any],
    image_size: int,
) -> np.ndarray:
    if action == "enhance_blur":
        return enhance_with_quality_cnn(
            image=image,
            stage26_module=stage26_module,
            checkpoint_path=sr_checkpoint_path,
            device=device,
            bundle_holder=bundle_holder,
            image_size=image_size,
        )

    if action == "clahe_then_enhance":
        clahe_image = mild_clahe(image)
        return enhance_with_quality_cnn(
            image=clahe_image,
            stage26_module=stage26_module,
            checkpoint_path=sr_checkpoint_path,
            device=device,
            bundle_holder=bundle_holder,
            image_size=image_size,
        )

    if action == "denoise_cautiously":
        return mild_denoise(image)

    return image.copy()


def features_for_image(image: np.ndarray, proxy_module: Any) -> dict[str, float]:
    features = proxy_module.compute_frame_features(image)
    return {column: float(features[column]) for column in PROXY_FEATURE_COLUMNS}


def compute_label_thresholds(original_feature_df: pd.DataFrame) -> dict[str, float]:
    return {
        "blur_q25": float(original_feature_df["blur_score"].quantile(0.25)),
        "contrast_q25": float(original_feature_df["contrast_score"].quantile(0.25)),
        "brightness_q10": float(original_feature_df["brightness_mean"].quantile(0.10)),
        "noise_q75": float(original_feature_df["noise_proxy"].quantile(0.75)),
    }


def label_with_thresholds(features: dict[str, float], thresholds: dict[str, float]) -> str:
    bad_conditions: list[str] = []
    if float(features["blur_score"]) < thresholds["blur_q25"]:
        bad_conditions.append("blurry_frame")
    if float(features["contrast_score"]) < thresholds["contrast_q25"]:
        bad_conditions.append("low_contrast_frame")
    if float(features["brightness_mean"]) < thresholds["brightness_q10"]:
        bad_conditions.append("too_dark_frame")
    if float(features["noise_proxy"]) > thresholds["noise_q75"]:
        bad_conditions.append("noisy_frame")

    if len(bad_conditions) > 1:
        return "poor_frame"
    if len(bad_conditions) == 1:
        return bad_conditions[0]
    return "good_frame"


def label_severity(label: str) -> int:
    return LABEL_SEVERITY.get(str(label), 1)


def improved_label_flag(before_label: str, after_label: str) -> int:
    return int(label_severity(after_label) < label_severity(before_label))


def collect_internal_rows(
    manifest_df: pd.DataFrame,
    stage26_module: Any,
    proxy_module: Any,
    sr_checkpoint_path: Path,
    device: torch.device,
    confidence_threshold: float,
    image_size: int,
    max_visual_examples: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    visual_examples: list[dict[str, object]] = []
    bundle_holder: dict[str, Any] = {"bundle": None}
    total_frames = len(manifest_df)

    for row_number, row in enumerate(manifest_df.itertuples(index=False), start=1):
        if row_number % 100 == 0 or row_number == total_frames:
            print(f"Processed {row_number} / {total_frames} frames")

        sweep_id = getattr(row, "sweep_id")
        frame_id = getattr(row, "frame_id")
        frame_index = int(getattr(row, "frame_index"))
        frame_path = resolve_project_path(getattr(row, "frame_path"))
        original_label = str(getattr(row, "frame_quality_label"))
        predicted_label = str(getattr(row, "learned_predicted_label"))
        confidence = float(getattr(row, "learned_confidence"))

        original_image = stage26_module.read_grayscale_normalized(frame_path)
        if original_image is None:
            print(f"WARNING: Could not read frame, skipping: {frame_path}")
            continue

        action = choose_action(
            predicted_label=predicted_label,
            confidence=confidence,
            original_label=original_label,
            confidence_threshold=confidence_threshold,
        )
        after_image = apply_action(
            image=original_image,
            action=action,
            stage26_module=stage26_module,
            sr_checkpoint_path=sr_checkpoint_path,
            device=device,
            bundle_holder=bundle_holder,
            image_size=image_size,
        )

        original_features = features_for_image(original_image, proxy_module)
        if action.startswith("keep_original") or action.startswith("feedback_"):
            after_features = dict(original_features)
        else:
            after_features = features_for_image(after_image, proxy_module)

        rows.append(
            {
                "sweep_id": sweep_id,
                "frame_id": frame_id,
                "frame_index": frame_index,
                "frame_path": rel_path(frame_path),
                "original_proxy_label": original_label,
                "learned_predicted_label": predicted_label,
                "learned_confidence": confidence,
                "action_taken": action,
                **{f"original_{key}": value for key, value in original_features.items()},
                **{f"after_{key}": value for key, value in after_features.items()},
            }
        )

        if (
            len(visual_examples) < max_visual_examples
            and action not in {"keep_original", "keep_original_low_confidence"}
        ):
            visual_examples.append(
                {
                    "sweep_id": sweep_id,
                    "frame_id": frame_id,
                    "original_proxy_label": original_label,
                    "learned_predicted_label": predicted_label,
                    "learned_confidence": confidence,
                    "action_taken": action,
                    "original_image": display_resize(original_image),
                    "after_image": display_resize(after_image),
                }
            )

    if not rows:
        raise ValueError("No readable ACOUSLIC frames were processed.")

    return pd.DataFrame(rows), visual_examples


def build_per_frame_output(
    internal_df: pd.DataFrame,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for row in internal_df.itertuples(index=False):
        original_features = {
            "blur_score": float(getattr(row, "original_blur_score")),
            "contrast_score": float(getattr(row, "original_contrast_score")),
            "brightness_mean": float(getattr(row, "original_brightness_mean")),
            "noise_proxy": float(getattr(row, "original_noise_proxy")),
            "edge_density": float(getattr(row, "original_edge_density")),
            "dark_ratio": float(getattr(row, "original_dark_ratio")),
            "bright_ratio": float(getattr(row, "original_bright_ratio")),
        }
        after_features = {
            "blur_score": float(getattr(row, "after_blur_score")),
            "contrast_score": float(getattr(row, "after_contrast_score")),
            "brightness_mean": float(getattr(row, "after_brightness_mean")),
            "noise_proxy": float(getattr(row, "after_noise_proxy")),
            "edge_density": float(getattr(row, "after_edge_density")),
            "dark_ratio": float(getattr(row, "after_dark_ratio")),
            "bright_ratio": float(getattr(row, "after_bright_ratio")),
        }

        before_label = label_with_thresholds(original_features, thresholds)
        after_label = label_with_thresholds(after_features, thresholds)

        rows.append(
            {
                "sweep_id": getattr(row, "sweep_id"),
                "frame_id": getattr(row, "frame_id"),
                "frame_index": int(getattr(row, "frame_index")),
                "frame_path": getattr(row, "frame_path"),
                "original_proxy_label": getattr(row, "original_proxy_label"),
                "learned_predicted_label": getattr(row, "learned_predicted_label"),
                "learned_confidence": float(getattr(row, "learned_confidence")),
                "action_taken": getattr(row, "action_taken"),
                "before_threshold_label": before_label,
                "after_threshold_label": after_label,
                "original_blur_score": original_features["blur_score"],
                "after_blur_score": after_features["blur_score"],
                "delta_blur_score": after_features["blur_score"] - original_features["blur_score"],
                "original_contrast_score": original_features["contrast_score"],
                "after_contrast_score": after_features["contrast_score"],
                "delta_contrast_score": (
                    after_features["contrast_score"] - original_features["contrast_score"]
                ),
                "original_edge_density": original_features["edge_density"],
                "after_edge_density": after_features["edge_density"],
                "delta_edge_density": (
                    after_features["edge_density"] - original_features["edge_density"]
                ),
                "original_noise_proxy": original_features["noise_proxy"],
                "after_noise_proxy": after_features["noise_proxy"],
                "delta_noise_proxy": (
                    after_features["noise_proxy"] - original_features["noise_proxy"]
                ),
                "improved_label_flag": improved_label_flag(before_label, after_label),
            }
        )

    return pd.DataFrame(rows)[PER_FRAME_COLUMNS]


def summarize_group(
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
        "original_proxy_label": original_label,
        "action_taken": action,
        "total_frames": total,
        "number_of_frames_improved": improved,
        "percentage_improved": percentage,
        "mean_delta_blur": float(group["delta_blur_score"].mean()) if total else 0.0,
        "mean_delta_contrast": float(group["delta_contrast_score"].mean()) if total else 0.0,
        "mean_delta_edge_density": float(group["delta_edge_density"].mean()) if total else 0.0,
        "mean_delta_noise_proxy": float(group["delta_noise_proxy"].mean()) if total else 0.0,
    }


def build_summary(per_frame_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = [
        summarize_group(per_frame_df, "overall", "all", "all")
    ]

    for action, group in per_frame_df.groupby("action_taken", sort=True):
        rows.append(summarize_group(group, "by_action", "all", str(action)))

    for label, group in per_frame_df.groupby("original_proxy_label", sort=True):
        rows.append(summarize_group(group, "by_original_label", str(label), "all"))

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

    for bar, value in zip(bars, values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            value_format.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_summary_figures(per_frame_df: pd.DataFrame) -> list[Path]:
    figure_dir = resolve_project_path(FIGURE_DIR)
    saved_paths: list[Path] = []

    action_counts = per_frame_df["action_taken"].value_counts().sort_index()
    action_path = figure_dir / "action_distribution.png"
    plot_bar(
        labels=[str(label) for label in action_counts.index],
        values=[float(value) for value in action_counts.values],
        title="Learned adaptive action distribution",
        ylabel="Number of frames",
        output_path=action_path,
        color="#4C78A8",
        value_format="{:.0f}",
    )
    saved_paths.append(action_path)

    grouped = per_frame_df.groupby("original_proxy_label", sort=True)
    labels = [str(label) for label in grouped.size().index]
    improvement_rates = [
        float(group["improved_label_flag"].mean() * 100.0) for _, group in grouped
    ]

    improvement_path = figure_dir / "improvement_rate_by_original_label.png"
    plot_bar(
        labels=labels,
        values=improvement_rates,
        title="Improvement rate by original proxy label",
        ylabel="Improved frames (%)",
        output_path=improvement_path,
        color="#72B7B2",
    )
    saved_paths.append(improvement_path)

    overall = per_frame_df[
        [
            "delta_blur_score",
            "delta_contrast_score",
            "delta_edge_density",
            "delta_noise_proxy",
        ]
    ].mean()
    metric_path = figure_dir / "mean_feature_changes.png"
    plot_bar(
        labels=["blur", "contrast", "edge", "noise"],
        values=[
            float(overall["delta_blur_score"]),
            float(overall["delta_contrast_score"]),
            float(overall["delta_edge_density"]),
            float(overall["delta_noise_proxy"]),
        ],
        title="Mean no-reference proxy feature changes",
        ylabel="Mean after minus before",
        output_path=metric_path,
        color="#F58518",
    )
    saved_paths.append(metric_path)

    plt.close("all")
    return saved_paths


def save_visual_examples(visual_examples: list[dict[str, object]]) -> list[Path]:
    saved_paths: list[Path] = []
    visual_dir = resolve_project_path(VISUAL_EXAMPLE_DIR)
    visual_dir.mkdir(parents=True, exist_ok=True)

    for index, example in enumerate(visual_examples, start=1):
        original_image = example["original_image"]
        after_image = example["after_image"]

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original_image, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0].set_title(f"Original\n{example['original_proxy_label']}")
        axes[0].axis("off")

        axes[1].imshow(after_image, cmap="gray", vmin=0.0, vmax=1.0)
        axes[1].set_title(f"After\n{example['action_taken']}")
        axes[1].axis("off")

        fig.suptitle(
            "sweep_id: "
            f"{example['sweep_id']} | frame_id: {example['frame_id']}\n"
            f"learned: {example['learned_predicted_label']} "
            f"({float(example['learned_confidence']):.3f})",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.88))

        output_path = (
            visual_dir
            / (
                f"learned_before_after_{index:03d}_"
                f"{safe_name(example['sweep_id'])}_{safe_name(example['frame_id'])}.png"
            )
        )
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(output_path)

    plt.close("all")
    return saved_paths


def counts_to_markdown(counts: pd.Series) -> str:
    rows = ["| Item | Count |", "|---|---:|"]
    for item, count in counts.items():
        rows.append(f"| `{markdown_escape(item)}` | {int(count)} |")
    if len(rows) == 2:
        rows.append("| _none_ | 0 |")
    return "\n".join(rows)


def summary_to_markdown(summary_df: pd.DataFrame) -> str:
    rows = [
        "| Scope | Original label | Action | Total frames | Improved | % improved | "
        "Mean delta blur | Mean delta contrast | Mean delta edge | Mean delta noise |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_df.itertuples(index=False):
        rows.append(
            "| "
            f"{markdown_escape(getattr(row, 'scope'))} | "
            f"`{markdown_escape(getattr(row, 'original_proxy_label'))}` | "
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


def write_report(
    per_frame_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    thresholds: dict[str, float],
    figure_paths: list[Path],
    visual_paths: list[Path],
    frame_manifest_path: Path,
    proxy_mlp_path: Path,
    sr_checkpoint_path: Path,
    confidence_threshold: float,
) -> None:
    overall = summary_df[summary_df["scope"] == "overall"].iloc[0]
    action_counts = per_frame_df["action_taken"].value_counts().sort_index()
    original_counts = per_frame_df["original_proxy_label"].value_counts().sort_index()
    predicted_counts = per_frame_df["learned_predicted_label"].value_counts().sort_index()

    figure_links = "\n".join(f"- `{rel_path(path)}`" for path in figure_paths)
    visual_links = "\n".join(f"- `{rel_path(path)}`" for path in visual_paths)
    if not visual_links:
        visual_links = "- _No visual examples were saved._"

    report = f"""# Learned Adaptive Feedback-aware Enhancement Report

{PROXY_LABEL_NOTE}

This run used a learned proxy-quality embedding MLP to choose conservative
feedback-aware enhancement actions. The Quality-aware CNN + L1/MSE model was
used only for frames where the learned policy requested enhancement.

## Inputs

- Proxy-quality manifest: `{rel_path(frame_manifest_path)}`
- Learned proxy-quality MLP: `{rel_path(proxy_mlp_path)}`
- Quality-aware CNN + L1/MSE checkpoint: `{rel_path(sr_checkpoint_path)}`
- Confidence threshold: {confidence_threshold:.2f}

## Outputs

- Per-frame CSV: `{rel_path(resolve_project_path(PER_FRAME_CSV))}`
- Summary CSV: `{rel_path(resolve_project_path(SUMMARY_CSV))}`
- Markdown report: `{rel_path(resolve_project_path(REPORT_MD))}`
- Figures: `{rel_path(resolve_project_path(FIGURE_DIR))}`

## Overall Metrics

- Total frames processed: {int(overall["total_frames"])}
- Number of frames improved by proxy threshold label: {int(overall["number_of_frames_improved"])}
- Percentage improved: {float(overall["percentage_improved"]):.2f}%
- Mean blur change: {float(overall["mean_delta_blur"]):.6f}
- Mean contrast change: {float(overall["mean_delta_contrast"]):.6f}
- Mean edge change: {float(overall["mean_delta_edge_density"]):.6f}
- Mean noise change: {float(overall["mean_delta_noise_proxy"]):.6f}

## Learned Policy

| Condition | Action |
|---|---|
| `good_frame` and confidence >= {confidence_threshold:.2f} | keep original |
| `blurry_frame` and confidence >= {confidence_threshold:.2f} | Quality-aware CNN + L1/MSE enhancement |
| `low_contrast_frame` and confidence >= {confidence_threshold:.2f} | CLAHE, then Quality-aware CNN + L1/MSE enhancement |
| `noisy_frame` and confidence >= {confidence_threshold:.2f} | cautious denoising |
| `poor_frame` and confidence >= {confidence_threshold:.2f} | repeat/reacquire feedback |
| confidence < {confidence_threshold:.2f} and original proxy label is `good_frame` | keep original |
| confidence < {confidence_threshold:.2f} and original proxy label is not `good_frame` | feedback warning |

## Original-feature thresholds used for before/after comparison

- blur_score < {thresholds["blur_q25"]:.8f}: `blurry_frame`
- contrast_score < {thresholds["contrast_q25"]:.8f}: `low_contrast_frame`
- brightness_mean < {thresholds["brightness_q10"]:.8f}: `too_dark_frame`
- noise_proxy > {thresholds["noise_q75"]:.8f}: `noisy_frame`
- Multiple bad conditions: `poor_frame`

## Action Counts

{counts_to_markdown(action_counts)}

## Original Proxy-label Counts

{counts_to_markdown(original_counts)}

## Learned Predicted-label Counts

{counts_to_markdown(predicted_counts)}

## Summary Table

{summary_to_markdown(summary_df)}

## Saved Figures

{figure_links}

## Visual Before/After Examples

{visual_links}
"""

    resolve_project_path(REPORT_MD).write_text(report, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    if args.confidence_threshold < 0.0 or args.confidence_threshold > 1.0:
        raise ValueError("--confidence_threshold must be between 0 and 1.")
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
    proxy_mlp_path = resolve_project_path(args.proxy_mlp_checkpoint)
    sr_checkpoint_path = resolve_project_path(args.sr_checkpoint)

    if not frame_manifest_path.exists():
        raise FileNotFoundError(f"Proxy-quality manifest not found: {frame_manifest_path}")
    if not proxy_mlp_path.exists():
        raise FileNotFoundError(f"Learned proxy MLP checkpoint not found: {proxy_mlp_path}")
    if not sr_checkpoint_path.exists():
        raise FileNotFoundError(f"Quality-aware SR checkpoint not found: {sr_checkpoint_path}")

    manifest_df = load_manifest(frame_manifest_path)
    full_frames = len(manifest_df)
    full_sweeps = int(manifest_df["sweep_id"].nunique())

    if args.max_sweeps is not None:
        selected_sweeps = manifest_df["sweep_id"].drop_duplicates().head(args.max_sweeps)
        manifest_df = manifest_df[manifest_df["sweep_id"].isin(selected_sweeps)].copy()
    if args.max_frames is not None:
        manifest_df = manifest_df.head(args.max_frames).copy()

    print(
        "Processing learned adaptive enhancement rows: "
        f"{len(manifest_df)} / {full_frames} frames across "
        f"{manifest_df['sweep_id'].nunique()} / {full_sweeps} sweeps"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learned_model = load_learned_quality_model(proxy_mlp_path, device=device)
    manifest_df = predict_proxy_labels(manifest_df, learned_model)

    stage26_module = import_stage26_module()
    proxy_module = import_proxy_quality_module()

    internal_df, visual_examples = collect_internal_rows(
        manifest_df=manifest_df,
        stage26_module=stage26_module,
        proxy_module=proxy_module,
        sr_checkpoint_path=sr_checkpoint_path,
        device=device,
        confidence_threshold=args.confidence_threshold,
        image_size=args.image_size,
        max_visual_examples=args.max_visual_examples,
    )

    original_feature_df = internal_df[
        [f"original_{column}" for column in PROXY_FEATURE_COLUMNS]
    ].copy()
    original_feature_df.columns = PROXY_FEATURE_COLUMNS
    thresholds = compute_label_thresholds(original_feature_df)

    per_frame_df = build_per_frame_output(internal_df, thresholds)
    summary_df = build_summary(per_frame_df)

    per_frame_df.to_csv(
        resolve_project_path(PER_FRAME_CSV),
        index=False,
        float_format="%.8f",
    )
    summary_df.to_csv(
        resolve_project_path(SUMMARY_CSV),
        index=False,
        float_format="%.8f",
    )

    figure_paths = create_summary_figures(per_frame_df)
    visual_paths = save_visual_examples(visual_examples)
    write_report(
        per_frame_df=per_frame_df,
        summary_df=summary_df,
        thresholds=thresholds,
        figure_paths=figure_paths,
        visual_paths=visual_paths,
        frame_manifest_path=frame_manifest_path,
        proxy_mlp_path=proxy_mlp_path,
        sr_checkpoint_path=sr_checkpoint_path,
        confidence_threshold=args.confidence_threshold,
    )

    overall = summary_df[summary_df["scope"] == "overall"].iloc[0]
    print(f"Total frames processed: {int(overall['total_frames'])}")
    print("Action counts:")
    for action, count in per_frame_df["action_taken"].value_counts().sort_index().items():
        print(f"  {action}: {int(count)}")
    print(f"Percentage improved: {float(overall['percentage_improved']):.2f}%")
    print(f"Per-frame CSV: {PER_FRAME_CSV}")
    print(f"Summary CSV: {SUMMARY_CSV}")
    print(f"Markdown report: {REPORT_MD}")
    print(f"Figure directory: {FIGURE_DIR}")
    print(PROXY_LABEL_NOTE)


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
