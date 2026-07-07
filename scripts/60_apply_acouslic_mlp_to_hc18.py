"""Apply the ACOUSLIC proxy-quality MLP to HC18 test images.

This is an evaluation-only script. It loads the frozen feature-based MLP from
scripts/29_train_proxy_quality_embedding.py, reuses the ACOUSLIC feature
helpers from scripts/24_compute_acouslic_proxy_quality.py, and writes only new
cross-dataset CSV outputs under results/cross_dataset/.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

import cv2
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
MANIFEST = PROJECT_ROOT / "data" / "processed" / "HC18_SR" / "hc18_lq_manifest.csv"
CHECKPOINT = PROJECT_ROOT / "results" / "models" / "proxy_quality_embedding_mlp.pth"
OUTPUT_DIR = PROJECT_ROOT / "results" / "cross_dataset"
PER_IMAGE_CSV = OUTPUT_DIR / "acouslic_quality_on_hc18_per_image.csv"
SUMMARY_CSV = OUTPUT_DIR / "acouslic_quality_on_hc18_summary.csv"

BASE_SOURCES = {
    "HR": "hr_path",
    "LQ_mixed": "lq_mixed_path",
    "LR_x2": "lr_x2_path",
    "LR_x4": "lr_x4_path",
}

SR_OUTPUT_ROOT = PROJECT_ROOT / "results" / "sr_outputs"
SWINIR_RECON_DIR = PROJECT_ROOT / "results" / "eval" / "swinir_light_phase_a" / "test" / "recon"

PROBABILITY_COLUMNS = {
    "good_frame": "prob_good_frame",
    "blurry_frame": "prob_blurry_frame",
    "low_contrast_frame": "prob_low_contrast_frame",
    "noisy_frame": "prob_noisy_frame",
    "poor_frame": "prob_poor_frame",
}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path
    image_paths: dict[str, Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the ACOUSLIC proxy-quality MLP to HC18 test images."
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return base_dir / path


def load_test_manifest() -> list[dict[str, str]]:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Missing HC18 manifest: {MANIFEST}")
    with MANIFEST.open("r", newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == "test"]
    if len(rows) != 150:
        raise ValueError(f"Expected 150 HC18 test rows, found {len(rows)}")
    return rows


def build_base_sources(test_rows: list[dict[str, str]]) -> list[SourceSpec]:
    sources: list[SourceSpec] = []
    for source_name, column in BASE_SOURCES.items():
        image_paths: dict[str, Path] = {}
        for row in test_rows:
            image_id = row["image_id"]
            image_path = resolve_path(row[column], MANIFEST.parent)
            if image_path.exists():
                image_paths[image_id] = image_path
        if len(image_paths) == len(test_rows):
            folder = image_paths[next(iter(image_paths))].parent
            sources.append(SourceSpec(source_name, folder, image_paths))
        else:
            print(
                f"WARNING: skipping {source_name}; found {len(image_paths)}/"
                f"{len(test_rows)} test images"
            )
    return sources


def image_map_from_folder(folder: Path, test_ids: list[str]) -> dict[str, Path]:
    image_paths: dict[str, Path] = {}
    for image_id in test_ids:
        path = folder / f"{image_id}.png"
        if path.exists():
            image_paths[image_id] = path
    return image_paths


def build_restoration_sources(test_rows: list[dict[str, str]]) -> list[SourceSpec]:
    test_ids = [row["image_id"] for row in test_rows]
    sources: list[SourceSpec] = []

    if SWINIR_RECON_DIR.exists():
        image_paths = image_map_from_folder(SWINIR_RECON_DIR, test_ids)
        if len(image_paths) == len(test_ids):
            sources.append(SourceSpec("swinir_light_phase_a", SWINIR_RECON_DIR, image_paths))
        else:
            print(
                "WARNING: skipping swinir_light_phase_a; found "
                f"{len(image_paths)}/{len(test_ids)} test recon images"
            )

    if SR_OUTPUT_ROOT.exists():
        for folder in sorted(SR_OUTPUT_ROOT.iterdir()):
            if not folder.is_dir():
                continue
            image_paths = image_map_from_folder(folder, test_ids)
            if len(image_paths) == len(test_ids):
                sources.append(SourceSpec(folder.name, folder, image_paths))
            else:
                png_count = len(list(folder.glob("*.png")))
                print(
                    f"WARNING: skipping {folder.name}; found {len(image_paths)}/"
                    f"{len(test_ids)} matching test images ({png_count} PNGs total)"
                )
    return sources


def load_checkpoint_and_model(proxy_module: Any) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = proxy_module.load_proxy_quality_checkpoint(CHECKPOINT, map_location="cpu")
    feature_columns = list(checkpoint.get("feature_columns", []))
    label_names = list(checkpoint.get("label_names", []))
    if feature_columns != list(proxy_module.FEATURE_COLUMNS):
        raise ValueError(
            "Checkpoint feature_columns do not match scripts/29 FEATURE_COLUMNS: "
            f"{feature_columns} vs {proxy_module.FEATURE_COLUMNS}"
        )
    if label_names != list(proxy_module.LABELS):
        raise ValueError(
            "Checkpoint label_names do not match scripts/29 LABELS: "
            f"{label_names} vs {proxy_module.LABELS}"
        )
    feature_mean = np.asarray(checkpoint.get("feature_mean"), dtype=np.float32)
    feature_std = np.asarray(checkpoint.get("feature_std"), dtype=np.float32)
    if feature_mean.shape != (len(feature_columns),) or feature_std.shape != (len(feature_columns),):
        raise ValueError("Checkpoint feature_mean/feature_std must both be length 12")

    model = proxy_module.ProxyQualityEmbeddingMLP(
        input_dim=int(checkpoint.get("input_dim", len(feature_columns))),
        num_classes=len(label_names),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def compute_source_features(source: SourceSpec, test_rows: list[dict[str, str]], feature_module: Any) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for frame_index, row in enumerate(test_rows):
        image_id = row["image_id"]
        image_path = source.image_paths[image_id]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        feature_values = feature_module.compute_frame_features(image, feature_size=256)
        records.append(
            {
                "sweep_id": source.name,
                "frame_id": image_id,
                "frame_index": frame_index,
                "frame_path": str(image_path),
                **feature_values,
            }
        )

    frame_df = pd.DataFrame(records)
    frame_df = feature_module.add_motion_features(frame_df)
    return frame_df


def predict_source(
    source: SourceSpec,
    test_rows: list[dict[str, str]],
    feature_module: Any,
    proxy_module: Any,
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
) -> list[dict[str, Any]]:
    label_names = list(proxy_module.LABELS)
    feature_columns = list(proxy_module.FEATURE_COLUMNS)
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std).astype(np.float32)

    feature_df = compute_source_features(source, test_rows, feature_module)
    features = feature_df[feature_columns].to_numpy(dtype=np.float32)
    normalized = (features - feature_mean) / feature_std

    with torch.no_grad():
        logits, _ = model(torch.from_numpy(normalized))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()

    rows: list[dict[str, Any]] = []
    for index, frame in feature_df.iterrows():
        probs = probabilities[index]
        argmax_idx = int(np.argmax(probs))
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        row: dict[str, Any] = {
            "source_folder": source.name,
            "image_id": str(frame["frame_id"]),
            "argmax_class_idx": argmax_idx,
            "argmax_class_name": label_names[argmax_idx],
            "entropy_5class": entropy,
        }
        for label in label_names:
            row[PROBABILITY_COLUMNS[label]] = float(probs[label_names.index(label)])
        rows.append(row)
    return rows


def summarize(per_image_rows: list[dict[str, Any]], labels: list[str]) -> list[dict[str, Any]]:
    df = pd.DataFrame(per_image_rows)
    summary_rows: list[dict[str, Any]] = []
    for source_name, group in df.groupby("source_folder", sort=False):
        n_images = int(len(group))
        row: dict[str, Any] = {
            "source_folder": source_name,
            "n_images": n_images,
            "mean_prob_good": float(group["prob_good_frame"].mean()),
            "mean_entropy": float(group["entropy_5class"].mean()),
        }
        for label in labels:
            short_name = label.removesuffix("_frame")
            row[f"pct_{short_name}"] = float((group["argmax_class_name"] == label).mean() * 100.0)
        summary_rows.append(row)
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    sys.path.insert(0, str(SCRIPT_DIR))
    proxy_module = load_module("proxy_quality_embedding_train_29", SCRIPT_DIR / "29_train_proxy_quality_embedding.py")
    feature_module = load_module("acouslic_proxy_features_24", SCRIPT_DIR / "24_compute_acouslic_proxy_quality.py")

    test_rows = load_test_manifest()
    sources = build_base_sources(test_rows) + build_restoration_sources(test_rows)
    if not sources:
        raise RuntimeError("No HC18 sources found for ACOUSLIC MLP evaluation")

    model, checkpoint = load_checkpoint_and_model(proxy_module)
    label_names = list(proxy_module.LABELS)

    print("Using checkpoint:", CHECKPOINT)
    print("Feature columns:", list(proxy_module.FEATURE_COLUMNS))
    print("Label order:", label_names)
    print("Sources:")
    for source in sources:
        print(f"  {source.name}: {source.path} ({len(source.image_paths)} test images)")

    all_rows: list[dict[str, Any]] = []
    for source in sources:
        print(f"Evaluating source: {source.name}")
        all_rows.extend(predict_source(source, test_rows, feature_module, proxy_module, model, checkpoint))

    per_image_columns = [
        "source_folder",
        "image_id",
        "argmax_class_idx",
        "argmax_class_name",
        "prob_good_frame",
        "prob_blurry_frame",
        "prob_low_contrast_frame",
        "prob_noisy_frame",
        "prob_poor_frame",
        "entropy_5class",
    ]
    summary_columns = [
        "source_folder",
        "n_images",
        "pct_good",
        "pct_blurry",
        "pct_low_contrast",
        "pct_noisy",
        "pct_poor",
        "mean_prob_good",
        "mean_entropy",
    ]

    summary_rows = summarize(all_rows, label_names)
    expected_rows = 150 * len(sources)
    if len(all_rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, generated {len(all_rows)}")

    write_csv(PER_IMAGE_CSV, all_rows, per_image_columns)
    write_csv(SUMMARY_CSV, summary_rows, summary_columns)
    print(f"Wrote per-image CSV: {PER_IMAGE_CSV}")
    print(f"Wrote summary CSV: {SUMMARY_CSV}")
    print(f"N_sources={len(sources)} per_image_rows={len(all_rows)} summary_rows={len(summary_rows)}")


def main() -> int:
    try:
        run(parse_args())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
