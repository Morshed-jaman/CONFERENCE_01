"""
Compute proxy image-quality and sweep-quality features for prepared
ACOUSLIC-AI PNG frames.

Important:
The labels produced here are heuristic proxy labels for quality-feedback
experiments. They are not clinical ground-truth labels and must not be reported
as diagnostic annotations.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_MANIFEST = PROJECT_ROOT / "data/processed/ACOUSLIC_AI/acouslic_frame_manifest.csv"
OUTPUT_ROOT = PROJECT_ROOT / "data/processed/ACOUSLIC_AI"
RESULTS_ROOT = PROJECT_ROOT / "results/acouslic_ai"

FRAME_QUALITY_MANIFEST = OUTPUT_ROOT / "acouslic_proxy_quality_manifest.csv"
SWEEP_QUALITY_SUMMARY = OUTPUT_ROOT / "acouslic_sweep_quality_summary.csv"
MARKDOWN_SUMMARY = RESULTS_ROOT / "acouslic_proxy_quality_summary.md"
VISUAL_DIR = RESULTS_ROOT / "proxy_quality_visuals"

FRAME_OUTPUT_COLUMNS = [
    "sweep_id",
    "frame_id",
    "frame_index",
    "frame_path",
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
    "frame_quality_label",
    "sweep_feedback_label",
]

SWEEP_OUTPUT_COLUMNS = [
    "sweep_id",
    "num_frames",
    "mean_blur_score",
    "mean_contrast_score",
    "mean_edge_density",
    "mean_optical_flow",
    "mean_jitter_score",
    "poor_frame_ratio",
    "final_sweep_feedback_label",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute proxy frame-quality and sweep-feedback features from "
            "ACOUSLIC-AI PNG frames."
        )
    )
    parser.add_argument(
        "--input_manifest",
        type=Path,
        default=DEFAULT_INPUT_MANIFEST,
        help="Input frame manifest created by the ACOUSLIC .mha sample extractor.",
    )
    parser.add_argument(
        "--max_visuals",
        type=int,
        default=20,
        help="Maximum number of proxy-quality visual examples to save.",
    )
    parser.add_argument(
        "--visual_dpi",
        type=int,
        default=200,
        help="DPI for saved visual examples.",
    )
    parser.add_argument(
        "--feature_size",
        type=int,
        default=256,
        help=(
            "Resize frames to feature_size x feature_size before blur, contrast, "
            "noise, and edge feature extraction."
        ),
    )
    return parser.parse_args()


def rel_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_project_path(path_value: object) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_required_columns(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")


def read_grayscale_normalized(frame_path: Path) -> np.ndarray | None:
    image = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    image_float = image.astype(np.float32)
    if image_float.size == 0:
        return None

    if image_float.max() > 1.0:
        image_float /= 255.0

    return np.clip(image_float, 0.0, 1.0)


def prepare_feature_frame(image: np.ndarray, size: int = 256) -> np.ndarray:
    img = np.asarray(image)

    if img.ndim == 3:
        img = np.squeeze(img)

    if img.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {img.shape}")

    if img.max() <= 1.0:
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
    else:
        img = img.clip(0, 255).astype(np.uint8)

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def compute_frame_features(image: np.ndarray, feature_size: int = 256) -> dict[str, float]:
    """Compute grayscale image-quality proxies on a resized frame."""
    img_uint8 = prepare_feature_frame(image, size=feature_size or 256)
    img_float32 = img_uint8.astype(np.float32) / 255.0
    img_float64 = img_uint8.astype(np.float64) / 255.0

    laplacian = cv2.Laplacian(img_float64, cv2.CV_64F)
    blur_score = float(laplacian.var())

    contrast_score = float(img_float32.std())
    brightness_mean = float(img_float32.mean())
    brightness_std = float(img_float32.std())

    blurred = cv2.blur(img_float32, (3, 3))
    noise_proxy = float(np.mean(np.abs(img_float32 - blurred)))

    sobel_x = cv2.Sobel(img_float32, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_float32, cv2.CV_32F, 0, 1, ksize=3)
    edge_magnitude = cv2.magnitude(sobel_x, sobel_y)
    edge_density = float(np.mean(edge_magnitude > 0.05))

    dark_ratio = float(np.mean(img_float32 < 0.05))
    bright_ratio = float(np.mean(img_float32 > 0.95))

    del sobel_x, sobel_y, edge_magnitude
    del laplacian, blurred, img_float64

    return {
        "blur_score": blur_score,
        "contrast_score": contrast_score,
        "brightness_mean": brightness_mean,
        "brightness_std": brightness_std,
        "noise_proxy": noise_proxy,
        "edge_density": edge_density,
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
    }


def label_frame_quality(df: pd.DataFrame) -> pd.Series:
    """Assign quantile-based frame-quality proxy labels."""
    blur_q25 = float(df["blur_score"].quantile(0.25))
    contrast_q25 = float(df["contrast_score"].quantile(0.25))
    brightness_q10 = float(df["brightness_mean"].quantile(0.10))
    noise_q75 = float(df["noise_proxy"].quantile(0.75))

    labels: list[str] = []
    for row in df.itertuples(index=False):
        bad_conditions: list[str] = []
        if row.blur_score < blur_q25:
            bad_conditions.append("blurry_frame")
        if row.contrast_score < contrast_q25:
            bad_conditions.append("low_contrast_frame")
        if row.brightness_mean < brightness_q10:
            bad_conditions.append("too_dark_frame")
        if row.noise_proxy > noise_q75:
            bad_conditions.append("noisy_frame")

        if len(bad_conditions) > 1:
            labels.append("poor_frame")
        elif len(bad_conditions) == 1:
            labels.append(bad_conditions[0])
        else:
            labels.append("good_frame")

    return pd.Series(labels, index=df.index, dtype="object")


def resize_like_current(previous: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Make consecutive frames the same size for optical flow if needed."""
    if previous.shape == current.shape:
        return previous, current

    resized_current = cv2.resize(
        current,
        dsize=(previous.shape[1], previous.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    return previous, resized_current.astype(np.float32)


def prepare_flow_frame(image: np.ndarray, size: int = 128) -> np.ndarray:
    """Resize one grayscale frame to a small uint8 image for safe optical flow."""
    img = np.asarray(image)

    if img.ndim == 3:
        img = np.squeeze(img)

    if img.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image for optical flow, got {img.shape}")

    if img.max() <= 1.0:
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
    else:
        img = img.clip(0, 255).astype(np.uint8)

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def add_motion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-frame motion proxies from consecutive frames in each sweep."""
    df = df.copy()
    df["frame_difference_mean"] = 0.0
    df["optical_flow_mean_magnitude"] = 0.0
    df["optical_flow_std_magnitude"] = 0.0
    df["jitter_score"] = 0.0

    processed_frames = 0
    total_frames = len(df)

    for _, group in df.groupby("sweep_id", sort=True):
        group = group.sort_values(["frame_index", "frame_id"])
        previous_for_flow: np.ndarray | None = None
        previous_flow_mean: float | None = None

        for idx, row in group.iterrows():
            frame_path = resolve_project_path(row["frame_path"])
            current_image = read_grayscale_normalized(frame_path)
            if current_image is None:
                print(f"WARNING: Could not read grayscale frame for motion, skipping: {frame_path}")
                continue

            current_for_flow = prepare_flow_frame(current_image, size=128)
            processed_frames += 1
            if processed_frames % 100 == 0:
                print(f"Motion feature progress: processed {processed_frames}/{total_frames} frames")
            if processed_frames % 200 == 0:
                gc.collect()

            if previous_for_flow is None:
                previous_for_flow = current_for_flow
                del current_image
                continue

            frame_difference = float(
                np.mean(cv2.absdiff(current_for_flow, previous_for_flow)) / 255.0
            )

            try:
                flow = cv2.calcOpticalFlowFarneback(
                    previous_for_flow,
                    current_for_flow,
                    None,
                    0.5,
                    3,
                    15,
                    3,
                    5,
                    1.2,
                    0,
                )
                magnitude, _ = cv2.cartToPolar(
                    flow[..., 0], flow[..., 1], angleInDegrees=False
                )
                flow_mean = float(np.mean(magnitude))
                flow_std = float(np.std(magnitude))
                jitter = 0.0 if previous_flow_mean is None else abs(flow_mean - previous_flow_mean)
                del flow, magnitude
            except Exception as exc:
                print(
                    "WARNING: Optical flow failed for "
                    f"sweep_id={row['sweep_id']}, frame_id={row['frame_id']}: {exc}. "
                    "Setting flow features to 0 for this pair."
                )
                flow_mean = 0.0
                flow_std = 0.0
                jitter = 0.0

            df.loc[idx, "frame_difference_mean"] = frame_difference
            df.loc[idx, "optical_flow_mean_magnitude"] = flow_mean
            df.loc[idx, "optical_flow_std_magnitude"] = flow_std
            df.loc[idx, "jitter_score"] = jitter

            previous_for_flow = current_for_flow
            previous_flow_mean = flow_mean
            del current_image

    return df


def minmax_normalize(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    minimum = float(values.min())
    maximum = float(values.max())
    denominator = maximum - minimum
    if denominator <= 1e-12:
        return pd.Series(np.zeros(len(values), dtype=np.float32), index=values.index)
    return (values - minimum) / denominator


def quality_drop_score_for_group(group: pd.DataFrame) -> float:
    """
    Estimate sudden sweep-quality drops from blur, contrast, and edge density.

    This is computed internally for feedback decisions while keeping the output
    CSV schema aligned with the requested columns.
    """
    ordered = group.sort_values(["frame_index", "frame_id"])
    if len(ordered) < 2:
        return 0.0

    quality_signal = (
        minmax_normalize(ordered["blur_score"])
        + minmax_normalize(ordered["contrast_score"])
        + minmax_normalize(ordered["edge_density"])
    ) / 3.0
    drops = quality_signal.shift(1) - quality_signal
    drops = drops.fillna(0.0).clip(lower=0.0)
    return float(drops.max())


def build_sweep_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sweep_id, group in df.groupby("sweep_id", sort=True):
        poor_frame_ratio = float(np.mean(group["frame_quality_label"] != "good_frame"))
        mean_flow = float(group["optical_flow_mean_magnitude"].mean())
        mean_jitter = float(group["jitter_score"].mean())

        rows.append(
            {
                "sweep_id": sweep_id,
                "num_frames": int(len(group)),
                "mean_blur_score": float(group["blur_score"].mean()),
                "mean_contrast_score": float(group["contrast_score"].mean()),
                "mean_edge_density": float(group["edge_density"].mean()),
                "mean_optical_flow": mean_flow,
                "mean_jitter_score": mean_jitter,
                "poor_frame_ratio": poor_frame_ratio,
                "_quality_drop_score": quality_drop_score_for_group(group),
                "_fast_sweep_score": mean_flow,
                "_stillness_score": 1.0 / (1.0 + mean_flow),
            }
        )

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        summary_df["final_sweep_feedback_label"] = []
        return summary_df

    flow_q75 = float(summary_df["mean_optical_flow"].quantile(0.75))
    jitter_q75 = float(summary_df["mean_jitter_score"].quantile(0.75))
    contrast_q25 = float(summary_df["mean_contrast_score"].quantile(0.25))
    edge_q25 = float(summary_df["mean_edge_density"].quantile(0.25))
    quality_drop_q75 = float(summary_df["_quality_drop_score"].quantile(0.75))

    labels: list[str] = []
    for _, row in summary_df.iterrows():
        if float(row["mean_optical_flow"]) > flow_q75:
            labels.append("move_slower")
        elif float(row["mean_jitter_score"]) > jitter_q75:
            labels.append("keep_probe_steady")
        elif (
            float(row["poor_frame_ratio"]) > 0.40
            or float(row["_quality_drop_score"]) > quality_drop_q75
        ):
            labels.append("repeat_sweep")
        elif (
            float(row["mean_contrast_score"]) < contrast_q25
            or float(row["mean_edge_density"]) < edge_q25
        ):
            labels.append("improve_visibility")
        else:
            labels.append("acceptable")

    summary_df["final_sweep_feedback_label"] = labels
    return summary_df


def write_markdown_summary(
    summary_path: Path,
    frame_df: pd.DataFrame,
    sweep_summary_df: pd.DataFrame,
) -> None:
    frame_counts = frame_df["frame_quality_label"].value_counts().sort_index()
    sweep_counts = sweep_summary_df["final_sweep_feedback_label"].value_counts().sort_index()

    frame_count_rows = "\n".join(
        f"| `{label}` | {count} |" for label, count in frame_counts.items()
    )
    sweep_count_rows = "\n".join(
        f"| `{label}` | {count} |" for label, count in sweep_counts.items()
    )

    summary_text = f"""# ACOUSLIC-AI Proxy Quality Summary

These labels are heuristic proxy labels for image-quality and sweep-feedback
experiments. They are not clinical ground-truth labels.

- Frames processed: {len(frame_df)}
- Sweeps processed: {len(sweep_summary_df)}
- Frame manifest: `{rel_path(FRAME_QUALITY_MANIFEST)}`
- Sweep summary: `{rel_path(SWEEP_QUALITY_SUMMARY)}`
- Visual examples: `{rel_path(VISUAL_DIR)}`

## Frame-quality proxy label counts

| Label | Count |
|---|---:|
{frame_count_rows}

## Final sweep-feedback proxy label counts

| Label | Count |
|---|---:|
{sweep_count_rows}
"""

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary_text, encoding="utf-8")


def save_visual_examples(
    frame_df: pd.DataFrame,
    visual_dir: Path,
    max_visuals: int,
    dpi: int,
) -> None:
    visual_dir.mkdir(parents=True, exist_ok=True)

    examples = frame_df.sort_values(["sweep_id", "frame_index", "frame_id"]).head(max_visuals)
    for visual_index, (_, row) in enumerate(examples.iterrows(), start=1):
        frame_path = resolve_project_path(row["frame_path"])
        image = read_grayscale_normalized(frame_path)
        if image is None:
            print(f"WARNING: Could not read visual example frame, skipping: {frame_path}")
            continue

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        ax.axis("off")

        title = (
            f"sweep_id: {row['sweep_id']}\n"
            f"frame_id: {row['frame_id']}\n"
            f"blur: {row['blur_score']:.6f} | contrast: {row['contrast_score']:.6f} | "
            f"edge density: {row['edge_density']:.6f}\n"
            f"frame label: {row['frame_quality_label']}\n"
            f"sweep feedback: {row['sweep_feedback_label']}"
        )
        ax.set_title(title, fontsize=9)
        fig.tight_layout()

        safe_sweep_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(row["sweep_id"]))
        safe_frame_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(row["frame_id"]))
        output_path = visual_dir / f"proxy_quality_{visual_index:03d}_{safe_sweep_id}_{safe_frame_id}.png"
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        del image
    plt.close("all")


def print_counts(title: str, counts: pd.Series) -> None:
    print(title)
    if counts.empty:
        print("  (none)")
        return
    for label, count in counts.sort_index().items():
        print(f"  {label}: {count}")


def run(args: argparse.Namespace) -> None:
    if args.max_visuals < 0:
        raise ValueError("--max_visuals must be 0 or greater.")
    if args.visual_dpi < 200:
        raise ValueError("--visual_dpi must be 200 or greater.")
    if args.feature_size <= 0:
        raise ValueError("--feature_size must be greater than 0.")

    input_manifest = resolve_project_path(args.input_manifest)
    if not input_manifest.exists():
        raise FileNotFoundError(f"Input frame manifest not found: {input_manifest}")

    manifest_df = pd.read_csv(input_manifest)
    ensure_required_columns(
        manifest_df,
        ["sweep_id", "frame_id", "frame_index", "frame_path"],
        input_manifest,
    )

    records: list[dict[str, object]] = []
    total_manifest_rows = len(manifest_df)

    for _, row in manifest_df.iterrows():
        frame_path = resolve_project_path(row["frame_path"])
        image = read_grayscale_normalized(frame_path)
        if image is None:
            print(f"WARNING: Could not read grayscale frame, skipping: {frame_path}")
            continue

        frame_record: dict[str, object] = {
            "sweep_id": row["sweep_id"],
            "frame_id": row["frame_id"],
            "frame_index": int(row["frame_index"]),
            "frame_path": rel_path(frame_path),
        }
        frame_record.update(compute_frame_features(image, feature_size=args.feature_size or 256))
        records.append(frame_record)

        processed_count = len(records)
        if processed_count % 100 == 0:
            print(
                f"Frame feature progress: processed "
                f"{processed_count}/{total_manifest_rows} frames"
            )
        del image
        if processed_count % 200 == 0:
            gc.collect()

    if not records:
        raise ValueError("No readable frames were found. Nothing to process.")

    frame_df = pd.DataFrame(records)
    frame_df["frame_quality_label"] = label_frame_quality(frame_df)
    frame_df = add_motion_features(frame_df)

    sweep_summary_df = build_sweep_summary(frame_df)
    sweep_label_map = dict(
        zip(
            sweep_summary_df["sweep_id"],
            sweep_summary_df["final_sweep_feedback_label"],
            strict=False,
        )
    )
    frame_df["sweep_feedback_label"] = frame_df["sweep_id"].map(sweep_label_map).fillna("acceptable")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)

    frame_df[FRAME_OUTPUT_COLUMNS].to_csv(FRAME_QUALITY_MANIFEST, index=False, float_format="%.8f")
    sweep_summary_df[SWEEP_OUTPUT_COLUMNS].to_csv(
        SWEEP_QUALITY_SUMMARY, index=False, float_format="%.8f"
    )
    write_markdown_summary(MARKDOWN_SUMMARY, frame_df, sweep_summary_df)
    save_visual_examples(
        frame_df=frame_df,
        visual_dir=VISUAL_DIR,
        max_visuals=args.max_visuals,
        dpi=args.visual_dpi,
    )

    print(f"Number of frames processed: {len(frame_df)}")
    print(f"Number of sweeps processed: {len(sweep_summary_df)}")
    print_counts(
        "Frame-quality proxy label counts:",
        frame_df["frame_quality_label"].value_counts(),
    )
    print_counts(
        "Final sweep-feedback proxy label counts:",
        sweep_summary_df["final_sweep_feedback_label"].value_counts(),
    )
    print(f"Frame proxy-quality manifest: {rel_path(FRAME_QUALITY_MANIFEST)}")
    print(f"Sweep proxy-quality summary: {rel_path(SWEEP_QUALITY_SUMMARY)}")
    print(f"Markdown summary: {rel_path(MARKDOWN_SUMMARY)}")
    print(f"Visual examples folder: {rel_path(VISUAL_DIR)}")
    print("Reminder: these are proxy quality labels, not real clinical labels.")


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except (FileNotFoundError, ValueError, pd.errors.EmptyDataError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
