"""
Generate a proxy sweep-quality feedback report from ACOUSLIC-AI proxy results.

These reports use heuristic image and motion features. They are designed for
proxy sweep-quality feedback experiments, not expert clinical labeling.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FRAME_MANIFEST = (
    PROJECT_ROOT / "data/processed/ACOUSLIC_AI/acouslic_proxy_quality_manifest.csv"
)
DEFAULT_SWEEP_SUMMARY = (
    PROJECT_ROOT / "data/processed/ACOUSLIC_AI/acouslic_sweep_quality_summary.csv"
)

REPORT_DIR = PROJECT_ROOT / "results/acouslic_ai/feedback_report"
TIMELINE_DIR = REPORT_DIR / "sweep_quality_timeline_examples"
FEEDBACK_EXAMPLES_DIR = REPORT_DIR / "feedback_examples"

REPORT_MD = REPORT_DIR / "acouslic_feedback_report.md"
FRAME_DISTRIBUTION_FIG = REPORT_DIR / "frame_quality_distribution.png"
SWEEP_DISTRIBUTION_FIG = REPORT_DIR / "sweep_feedback_distribution.png"

FRAME_LABEL_EXPLANATIONS = {
    "good_frame": "The frame has acceptable proxy image quality based on blur, contrast, brightness, noise, and edge features.",
    "blurry_frame": "The frame has a low Laplacian-variance blur score, suggesting weak sharpness.",
    "low_contrast_frame": "The frame has low pixel-intensity variation, suggesting low visible contrast.",
    "noisy_frame": "The frame has a high local difference score, suggesting stronger image noise or speckle-like variation.",
    "poor_frame": "Multiple proxy quality issues were detected in the same frame.",
}

SWEEP_LABEL_EXPLANATIONS = {
    "move_slower": "Large frame-to-frame motion was detected; the operator should move the probe more slowly.",
    "keep_probe_steady": "Motion variation suggests unstable probe movement; the operator should keep the probe steadier.",
    "repeat_sweep": "The sweep contains many low-quality frames; repeating the sweep is recommended.",
    "improve_visibility": "Low contrast or weak edge visibility was detected; the operator should adjust probe position or image visibility.",
    "acceptable": "The sweep has acceptable proxy quality based on image and motion features.",
}

REQUIRED_FRAME_COLUMNS = [
    "sweep_id",
    "frame_id",
    "frame_index",
    "frame_path",
    "blur_score",
    "contrast_score",
    "edge_density",
    "optical_flow_mean_magnitude",
    "frame_quality_label",
    "sweep_feedback_label",
]

REQUIRED_SWEEP_COLUMNS = [
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
        description="Generate an ACOUSLIC-AI proxy sweep-quality feedback report."
    )
    parser.add_argument(
        "--frame_manifest",
        type=Path,
        default=DEFAULT_FRAME_MANIFEST,
        help="Input per-frame proxy-quality manifest CSV.",
    )
    parser.add_argument(
        "--sweep_summary",
        type=Path,
        default=DEFAULT_SWEEP_SUMMARY,
        help="Input per-sweep proxy-quality summary CSV.",
    )
    parser.add_argument(
        "--max_feedback_examples",
        type=int,
        default=10,
        help="Number of frame feedback example images to save.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
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
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text).strip("_") or "item"


def markdown_escape(value: object) -> str:
    return str(value).replace("|", "\\|")


def validate_columns(df: pd.DataFrame, required_columns: list[str], csv_path: Path) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")


def create_output_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TIMELINE_DIR.mkdir(parents=True, exist_ok=True)
    FEEDBACK_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def plot_distribution(
    counts: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    labels = [str(label) for label in counts.index]
    values = counts.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=30)

    for bar, value in zip(bars, values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{int(value)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sweep_timeline(sweep_id: object, group: pd.DataFrame, output_path: Path) -> None:
    ordered = group.sort_values(["frame_index", "frame_id"])
    x = ordered["frame_index"].to_numpy()
    poor_mask = (ordered["frame_quality_label"] != "good_frame").to_numpy()

    metrics = [
        ("blur_score", "Blur score", "#4C78A8"),
        ("contrast_score", "Contrast score", "#F58518"),
    ]
    if "optical_flow_mean_magnitude" in ordered.columns:
        metrics.append(
            (
                "optical_flow_mean_magnitude",
                "Optical flow mean magnitude",
                "#54A24B",
            )
        )

    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=1,
        figsize=(10, 2.8 * len(metrics)),
        sharex=True,
    )
    if len(metrics) == 1:
        axes = [axes]

    for ax, (column, label, color) in zip(axes, metrics, strict=False):
        y = ordered[column].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=1.5, color=color, label=label)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)

        if poor_mask.any():
            poor_x = x[poor_mask]
            poor_y = y[poor_mask]
            ax.scatter(
                poor_x,
                poor_y,
                color="red",
                marker="x",
                s=55,
                linewidths=1.5,
                label="non-good/poor frame",
                zorder=5,
            )

        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("Frame index")
    fig.suptitle(f"Proxy sweep-quality timeline: {sweep_id}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def read_frame_image(frame_path: Path) -> np.ndarray | None:
    image = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0
    return np.clip(image, 0.0, 1.0)


def save_feedback_examples(
    frame_df: pd.DataFrame,
    output_dir: Path,
    max_examples: int,
) -> list[Path]:
    if max_examples <= 0:
        return []

    non_good = frame_df[frame_df["frame_quality_label"] != "good_frame"]
    good = frame_df[frame_df["frame_quality_label"] == "good_frame"]
    examples = pd.concat([non_good, good], ignore_index=False).head(max_examples)

    saved_paths: list[Path] = []
    for example_index, (_, row) in enumerate(examples.iterrows(), start=1):
        frame_path = resolve_project_path(Path(str(row["frame_path"])))
        image = read_frame_image(frame_path)
        if image is None:
            print(f"WARNING: Could not read feedback example frame, skipping: {frame_path}")
            continue

        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        ax.axis("off")

        overlay_text = (
            f"Frame label: {row['frame_quality_label']}\n"
            f"Sweep feedback: {row['sweep_feedback_label']}\n"
            f"Blur: {float(row['blur_score']):.6f}\n"
            f"Contrast: {float(row['contrast_score']):.6f}\n"
            f"Edge density: {float(row['edge_density']):.6f}"
        )
        ax.text(
            0.02,
            0.98,
            overlay_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="white",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "black",
                "alpha": 0.70,
                "edgecolor": "none",
            },
        )

        output_path = (
            output_dir
            / f"feedback_example_{example_index:03d}_{safe_name(row['sweep_id'])}_{safe_name(row['frame_id'])}.png"
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(output_path)

    return saved_paths


def counts_to_markdown(counts: pd.Series) -> str:
    if counts.empty:
        return "| Label | Count |\n|---|---:|\n| _none_ | 0 |"

    rows = ["| Label | Count |", "|---|---:|"]
    for label, count in counts.items():
        rows.append(f"| `{markdown_escape(label)}` | {int(count)} |")
    return "\n".join(rows)


def explanations_to_markdown() -> str:
    rows = ["| Label | Explanation |", "|---|---|"]

    ordered_labels = [
        "good_frame",
        "blurry_frame",
        "low_contrast_frame",
        "noisy_frame",
        "poor_frame",
        "move_slower",
        "keep_probe_steady",
        "repeat_sweep",
        "improve_visibility",
        "acceptable",
    ]

    explanations = {**FRAME_LABEL_EXPLANATIONS, **SWEEP_LABEL_EXPLANATIONS}
    for label in ordered_labels:
        rows.append(
            f"| `{label}` | {markdown_escape(explanations.get(label, 'Proxy feedback label.'))} |"
        )

    return "\n".join(rows)


def sweep_feedback_sentence(label: object) -> str:
    return SWEEP_LABEL_EXPLANATIONS.get(
        str(label),
        "The sweep has a proxy feedback label derived from image and motion features.",
    )


def sweeps_to_markdown(sweep_df: pd.DataFrame) -> str:
    rows = [
        "| Sweep ID | Frames | Poor frame ratio | Mean blur | Mean contrast | Mean edge density | Mean optical flow | Mean jitter | Final feedback | Feedback sentence |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]

    ordered = sweep_df.sort_values(["sweep_id"])
    for row in ordered.itertuples(index=False):
        label = getattr(row, "final_sweep_feedback_label")
        rows.append(
            "| "
            f"{markdown_escape(getattr(row, 'sweep_id'))} | "
            f"{int(getattr(row, 'num_frames'))} | "
            f"{float(getattr(row, 'poor_frame_ratio')):.4f} | "
            f"{float(getattr(row, 'mean_blur_score')):.6f} | "
            f"{float(getattr(row, 'mean_contrast_score')):.6f} | "
            f"{float(getattr(row, 'mean_edge_density')):.6f} | "
            f"{float(getattr(row, 'mean_optical_flow')):.6f} | "
            f"{float(getattr(row, 'mean_jitter_score')):.6f} | "
            f"`{markdown_escape(label)}` | "
            f"{markdown_escape(sweep_feedback_sentence(label))} |"
        )

    return "\n".join(rows)


def write_markdown_report(
    frame_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    frame_counts: pd.Series,
    sweep_counts: pd.Series,
    timeline_paths: list[Path],
    feedback_example_paths: list[Path],
) -> None:
    timeline_links = "\n".join(f"- `{rel_path(path)}`" for path in timeline_paths)
    if not timeline_links:
        timeline_links = "- _No timeline figures were saved._"

    feedback_links = "\n".join(f"- `{rel_path(path)}`" for path in feedback_example_paths)
    if not feedback_links:
        feedback_links = "- _No feedback example images were saved._"

    report_text = f"""# ACOUSLIC-AI Proxy Sweep-Quality Feedback Report

These labels are proxy quality-feedback labels derived from image and motion features, not expert clinical labels.

## Dataset summary

- Total number of frames: {len(frame_df)}
- Total number of sweeps: {len(sweep_df)}

## Frame-quality label distribution

{counts_to_markdown(frame_counts)}

## Sweep-feedback label distribution

{counts_to_markdown(sweep_counts)}

## Proxy feedback label explanations

{explanations_to_markdown()}

## Per-sweep feedback summary

{sweeps_to_markdown(sweep_df)}

## Saved figures

- Frame-quality distribution: `{rel_path(FRAME_DISTRIBUTION_FIG)}`
- Sweep-feedback distribution: `{rel_path(SWEEP_DISTRIBUTION_FIG)}`
- Timeline folder: `{rel_path(TIMELINE_DIR)}`
- Feedback examples folder: `{rel_path(FEEDBACK_EXAMPLES_DIR)}`

### Timeline figures

{timeline_links}

### Feedback example images

{feedback_links}
"""

    REPORT_MD.write_text(report_text, encoding="utf-8")


def run(args: argparse.Namespace) -> list[Path]:
    if args.max_feedback_examples < 0:
        raise ValueError("--max_feedback_examples must be 0 or greater.")

    frame_manifest_path = resolve_project_path(args.frame_manifest)
    sweep_summary_path = resolve_project_path(args.sweep_summary)

    if not frame_manifest_path.exists():
        raise FileNotFoundError(f"Frame proxy-quality manifest not found: {frame_manifest_path}")
    if not sweep_summary_path.exists():
        raise FileNotFoundError(f"Sweep proxy-quality summary not found: {sweep_summary_path}")

    frame_df = pd.read_csv(frame_manifest_path)
    sweep_df = pd.read_csv(sweep_summary_path)

    validate_columns(frame_df, REQUIRED_FRAME_COLUMNS, frame_manifest_path)
    validate_columns(sweep_df, REQUIRED_SWEEP_COLUMNS, sweep_summary_path)

    create_output_dirs()

    frame_counts = frame_df["frame_quality_label"].value_counts().sort_index()
    sweep_counts = sweep_df["final_sweep_feedback_label"].value_counts().sort_index()

    saved_paths: list[Path] = []

    plot_distribution(
        counts=frame_counts,
        title="Frame-quality proxy label distribution",
        xlabel="Frame-quality label",
        ylabel="Number of frames",
        output_path=FRAME_DISTRIBUTION_FIG,
    )
    saved_paths.append(FRAME_DISTRIBUTION_FIG)

    plot_distribution(
        counts=sweep_counts,
        title="Sweep-feedback proxy label distribution",
        xlabel="Sweep-feedback label",
        ylabel="Number of sweeps",
        output_path=SWEEP_DISTRIBUTION_FIG,
    )
    saved_paths.append(SWEEP_DISTRIBUTION_FIG)

    timeline_paths: list[Path] = []
    for sweep_id, group in frame_df.groupby("sweep_id", sort=True):
        timeline_path = TIMELINE_DIR / f"sweep_timeline_{safe_name(sweep_id)}.png"
        plot_sweep_timeline(sweep_id=sweep_id, group=group, output_path=timeline_path)
        timeline_paths.append(timeline_path)
        saved_paths.append(timeline_path)

    feedback_example_paths = save_feedback_examples(
        frame_df=frame_df,
        output_dir=FEEDBACK_EXAMPLES_DIR,
        max_examples=args.max_feedback_examples,
    )
    saved_paths.extend(feedback_example_paths)

    write_markdown_report(
        frame_df=frame_df,
        sweep_df=sweep_df,
        frame_counts=frame_counts,
        sweep_counts=sweep_counts,
        timeline_paths=timeline_paths,
        feedback_example_paths=feedback_example_paths,
    )
    saved_paths.insert(0, REPORT_MD)

    return saved_paths


def main() -> int:
    args = parse_args()
    try:
        saved_paths = run(args)
    except (
        FileNotFoundError,
        ValueError,
        pd.errors.EmptyDataError,
        cv2.error,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Saved output paths:")
    for path in saved_paths:
        print(f"  {rel_path(path)}")

    print(
        "Reminder: these are proxy quality-feedback labels derived from image "
        "and motion features, not expert clinical labels."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
