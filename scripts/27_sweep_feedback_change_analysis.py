"""
Analyze sweep-level proxy feedback changes after Stage 28 adaptive enhancement.

These are proxy feedback labels derived from image and motion features, not
expert clinical labels. Motion-based feedback such as move_slower and
keep_probe_steady is preserved, because image enhancement does not correct
sweep motion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ADAPTIVE_PER_FRAME = (
    PROJECT_ROOT / "results/acouslic_ai/adaptive_feedback_enhancement_per_frame.csv"
)
DEFAULT_SWEEP_SUMMARY = (
    PROJECT_ROOT / "data/processed/ACOUSLIC_AI/acouslic_sweep_quality_summary.csv"
)
DEFAULT_PROXY_FRAME_MANIFEST = (
    PROJECT_ROOT / "data/processed/ACOUSLIC_AI/acouslic_proxy_quality_manifest.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results/acouslic_ai"
ADAPTIVE_PER_SWEEP_CSV = OUTPUT_DIR / "adaptive_sweep_feedback_change_per_sweep.csv"
ADAPTIVE_SUMMARY_CSV = OUTPUT_DIR / "adaptive_sweep_feedback_change_summary.csv"
ADAPTIVE_REPORT_MD = OUTPUT_DIR / "adaptive_sweep_feedback_change_report.md"
ADAPTIVE_FIGURE_DIR = OUTPUT_DIR / "adaptive_sweep_feedback_change_figures"

PROXY_LABEL_NOTE = (
    "These are proxy feedback labels derived from image and motion features, "
    "not expert clinical labels."
)

COUNT_LABELS = [
    "good_frame",
    "blurry_frame",
    "low_contrast_frame",
    "noisy_frame",
    "poor_frame",
]
POOR_LIKE_LABELS = [
    "poor_frame",
    "blurry_frame",
    "low_contrast_frame",
    "noisy_frame",
]
MOTION_FEEDBACK_LABELS = {"move_slower", "keep_probe_steady"}

PER_SWEEP_COLUMNS = [
    "sweep_id",
    "total_frames",
    "original_good_frame_count",
    "adaptive_good_frame_count",
    "original_poor_like_count",
    "adaptive_poor_like_count",
    "original_poor_frame_ratio",
    "adaptive_poor_frame_ratio",
    "poor_frame_ratio_reduction",
    "original_good_frame_ratio",
    "adaptive_good_frame_ratio",
    "good_frame_ratio_improvement",
    "original_feedback_label",
    "adaptive_feedback_label",
    "feedback_changed_flag",
    "mean_optical_flow",
    "mean_jitter_score",
    "motion_summary_missing_flag",
    "adaptive_low_contrast_frame_ratio",
    "unchanged_motion_based_decision_flag",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze whether Stage 28 adaptive enhancement changes ACOUSLIC-AI "
            "sweep-level proxy feedback decisions."
        )
    )
    parser.add_argument(
        "--adaptive_per_frame",
        type=Path,
        default=DEFAULT_ADAPTIVE_PER_FRAME,
        help="Stage 28 adaptive per-frame enhancement analysis CSV.",
    )
    parser.add_argument(
        "--sweep_summary",
        type=Path,
        default=DEFAULT_SWEEP_SUMMARY,
        help="Original ACOUSLIC sweep-quality summary CSV.",
    )
    parser.add_argument(
        "--proxy_frame_manifest",
        type=Path,
        default=DEFAULT_PROXY_FRAME_MANIFEST,
        help="Original ACOUSLIC proxy frame-quality manifest CSV.",
    )
    parser.add_argument(
        "--poor_ratio_threshold",
        type=float,
        default=0.40,
        help="Adaptive poor-like frame ratio above this value maps to repeat_sweep.",
    )
    parser.add_argument(
        "--low_contrast_ratio_threshold",
        type=float,
        default=0.25,
        help=(
            "Adaptive low_contrast_frame ratio above this value maps to "
            "improve_visibility when no motion feedback is being preserved."
        ),
    )
    parser.add_argument(
        "--max_example_sweeps",
        type=int,
        default=10,
        help="Number of example sweep figures to save.",
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
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    return safe.strip("._") or "item"


def markdown_escape(value: object) -> str:
    return str(value).replace("|", "\\|")


def validate_columns(df: pd.DataFrame, required: list[str], path: Path) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")


def create_output_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTIVE_FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def label_count(group: pd.DataFrame, column: str, label: str) -> int:
    return int((group[column].astype(str) == label).sum())


def count_other_labels(group: pd.DataFrame, column: str) -> int:
    labels = group[column].astype(str)
    return int((~labels.isin(COUNT_LABELS)).sum())


def compute_adaptive_feedback(
    original_feedback: str,
    adaptive_poor_frame_ratio: float,
    adaptive_low_contrast_frame_ratio: float,
    poor_ratio_threshold: float,
    low_contrast_ratio_threshold: float,
) -> tuple[str, str]:
    """
    Recompute image/visibility-related feedback after enhancement.

    Motion labels are preserved in the final fallback branch because image
    enhancement does not change probe motion.
    """
    if adaptive_poor_frame_ratio > poor_ratio_threshold:
        return "repeat_sweep", "adaptive_poor_like_ratio_high"

    if adaptive_low_contrast_frame_ratio > low_contrast_ratio_threshold:
        return "improve_visibility", "adaptive_low_contrast_ratio_high"

    if original_feedback in MOTION_FEEDBACK_LABELS:
        return original_feedback, "unchanged_motion_based_feedback"

    return "acceptable", "adaptive_visibility_acceptable"


def build_per_sweep_dataframe(
    enhancement_df: pd.DataFrame,
    sweep_summary_df: pd.DataFrame,
    poor_ratio_threshold: float,
    low_contrast_ratio_threshold: float,
) -> pd.DataFrame:
    summary_lookup = sweep_summary_df.drop_duplicates("sweep_id").set_index("sweep_id", drop=False)
    rows: list[dict[str, object]] = []

    for sweep_id, group in enhancement_df.groupby("sweep_id", sort=True):
        total_frames = int(len(group))
        if total_frames == 0:
            continue

        original_labels = group["original_frame_quality_label"].astype(str)
        adaptive_labels = group["adaptive_frame_quality_label"].astype(str)

        original_good_count = int((original_labels == "good_frame").sum())
        adaptive_good_count = int((adaptive_labels == "good_frame").sum())
        original_poor_like = int(original_labels.isin(POOR_LIKE_LABELS).sum())
        adaptive_poor_like = int(adaptive_labels.isin(POOR_LIKE_LABELS).sum())

        original_poor_ratio = original_poor_like / total_frames
        adaptive_poor_ratio = adaptive_poor_like / total_frames
        original_good_ratio = original_good_count / total_frames
        adaptive_good_ratio = adaptive_good_count / total_frames
        adaptive_low_contrast_ratio = int((adaptive_labels == "low_contrast_frame").sum()) / total_frames

        if sweep_id in summary_lookup.index:
            summary_row = summary_lookup.loc[sweep_id]
            if isinstance(summary_row, pd.DataFrame):
                summary_row = summary_row.iloc[0]

            mean_optical_flow = float(summary_row.get("mean_optical_flow", np.nan))
            mean_jitter_score = float(summary_row.get("mean_jitter_score", np.nan))
            original_feedback = str(
                summary_row.get("final_sweep_feedback_label", "acceptable")
            )
            motion_summary_missing = 0
        else:
            mean_optical_flow = np.nan
            mean_jitter_score = np.nan
            original_feedback = "missing_motion_summary"
            motion_summary_missing = 1

        adaptive_feedback, reason = compute_adaptive_feedback(
            original_feedback=original_feedback,
            adaptive_poor_frame_ratio=adaptive_poor_ratio,
            adaptive_low_contrast_frame_ratio=adaptive_low_contrast_ratio,
            poor_ratio_threshold=poor_ratio_threshold,
            low_contrast_ratio_threshold=low_contrast_ratio_threshold,
        )

        rows.append(
            {
                "sweep_id": sweep_id,
                "total_frames": total_frames,
                "original_good_frame_count": original_good_count,
                "adaptive_good_frame_count": adaptive_good_count,
                "original_poor_like_count": original_poor_like,
                "adaptive_poor_like_count": adaptive_poor_like,
                "original_poor_frame_ratio": float(original_poor_ratio),
                "adaptive_poor_frame_ratio": float(adaptive_poor_ratio),
                "poor_frame_ratio_reduction": float(original_poor_ratio - adaptive_poor_ratio),
                "original_good_frame_ratio": float(original_good_ratio),
                "adaptive_good_frame_ratio": float(adaptive_good_ratio),
                "good_frame_ratio_improvement": float(adaptive_good_ratio - original_good_ratio),
                "original_feedback_label": original_feedback,
                "adaptive_feedback_label": adaptive_feedback,
                "feedback_changed_flag": int(original_feedback != adaptive_feedback),
                "mean_optical_flow": mean_optical_flow,
                "mean_jitter_score": mean_jitter_score,
                "motion_summary_missing_flag": motion_summary_missing,
                "adaptive_low_contrast_frame_ratio": float(adaptive_low_contrast_ratio),
                "unchanged_motion_based_decision_flag": int(
                    original_feedback in MOTION_FEEDBACK_LABELS
                    and adaptive_feedback == original_feedback
                ),
                "feedback_change_reason": reason,
            }
        )

    per_sweep_df = pd.DataFrame(rows)
    return per_sweep_df[[*PER_SWEEP_COLUMNS, "feedback_change_reason"]]


def build_summary_dataframe(per_sweep_df: pd.DataFrame) -> pd.DataFrame:
    total_sweeps = int(len(per_sweep_df))
    reduced_poor = int((per_sweep_df["poor_frame_ratio_reduction"] > 0).sum())
    improved_good = int((per_sweep_df["good_frame_ratio_improvement"] > 0).sum())
    repeat_changed = int(
        (
            (per_sweep_df["original_feedback_label"] == "repeat_sweep")
            & per_sweep_df["adaptive_feedback_label"].isin(
                ["acceptable", "improve_visibility"]
            )
        ).sum()
    )
    motion_unchanged = int(per_sweep_df["unchanged_motion_based_decision_flag"].sum())

    return pd.DataFrame(
        [
            {
                "number_of_sweeps_analyzed": total_sweeps,
                "number_of_sweeps_with_reduced_poor_frame_ratio": reduced_poor,
                "mean_poor_frame_ratio_reduction": float(
                    per_sweep_df["poor_frame_ratio_reduction"].mean()
                )
                if total_sweeps
                else 0.0,
                "number_of_sweeps_with_improved_good_frame_ratio": improved_good,
                "number_of_repeat_sweep_cases_changed_to_acceptable_or_improve_visibility": repeat_changed,
                "number_of_feedback_decisions_unchanged_due_to_motion_based_problems": motion_unchanged,
            }
        ]
    )


def plot_distribution(
    counts: pd.Series,
    title: str,
    ylabel: str,
    output_path: Path,
    color: str,
) -> None:
    labels = [str(label) for label in counts.index]
    values = counts.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=30)
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(value)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ratio_before_after(
    per_sweep_df: pd.DataFrame,
    before_column: str,
    after_column: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    ordered = per_sweep_df.sort_values("sweep_id")
    labels = [str(value) for value in ordered["sweep_id"]]
    x = np.arange(len(ordered))
    width = 0.38

    fig_width = min(max(10, 0.45 * len(ordered)), 30)
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    ax.bar(x - width / 2, ordered[before_column].to_numpy(dtype=float), width, label="Original")
    ax.bar(x + width / 2, ordered[after_column].to_numpy(dtype=float), width, label="Adaptive")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    if len(labels) <= 50:
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    else:
        ax.set_xticklabels([])
        ax.set_xlabel("Sweeps")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_example_sweep(row: pd.Series, output_path: Path) -> None:
    labels = ["Poor-like ratio", "Good-frame ratio"]
    original_values = [
        float(row["original_poor_frame_ratio"]),
        float(row["original_good_frame_ratio"]),
    ]
    adaptive_values = [
        float(row["adaptive_poor_frame_ratio"]),
        float(row["adaptive_good_frame_ratio"]),
    ]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, original_values, width, label="Original", color="#4C78A8")
    ax.bar(x + width / 2, adaptive_values, width, label="Adaptive", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Ratio")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    title = (
        f"Sweep: {row['sweep_id']}\n"
        f"Original feedback: {row['original_feedback_label']} | "
        f"Adaptive feedback: {row['adaptive_feedback_label']}\n"
        f"Poor ratio: {row['original_poor_frame_ratio']:.3f} -> "
        f"{row['adaptive_poor_frame_ratio']:.3f} | "
        f"Good ratio: {row['original_good_frame_ratio']:.3f} -> "
        f"{row['adaptive_good_frame_ratio']:.3f}"
    )
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_figures(per_sweep_df: pd.DataFrame, max_example_sweeps: int) -> list[Path]:
    saved_paths: list[Path] = []

    original_dist_path = ADAPTIVE_FIGURE_DIR / "original_sweep_feedback_distribution.png"
    plot_distribution(
        per_sweep_df["original_feedback_label"].value_counts().sort_index(),
        title="Original sweep-feedback proxy label distribution",
        ylabel="Number of sweeps",
        output_path=original_dist_path,
        color="#4C78A8",
    )
    saved_paths.append(original_dist_path)

    adaptive_dist_path = ADAPTIVE_FIGURE_DIR / "adaptive_sweep_feedback_distribution.png"
    plot_distribution(
        per_sweep_df["adaptive_feedback_label"].value_counts().sort_index(),
        title="Adaptive sweep-feedback proxy label distribution",
        ylabel="Number of sweeps",
        output_path=adaptive_dist_path,
        color="#54A24B",
    )
    saved_paths.append(adaptive_dist_path)

    poor_ratio_path = ADAPTIVE_FIGURE_DIR / "poor_frame_ratio_before_after_per_sweep.png"
    plot_ratio_before_after(
        per_sweep_df=per_sweep_df,
        before_column="original_poor_frame_ratio",
        after_column="adaptive_poor_frame_ratio",
        title="Poor-like frame ratio before vs after adaptive enhancement",
        ylabel="Poor-like frame ratio",
        output_path=poor_ratio_path,
    )
    saved_paths.append(poor_ratio_path)

    good_ratio_path = ADAPTIVE_FIGURE_DIR / "good_frame_ratio_before_after_per_sweep.png"
    plot_ratio_before_after(
        per_sweep_df=per_sweep_df,
        before_column="original_good_frame_ratio",
        after_column="adaptive_good_frame_ratio",
        title="Good-frame ratio before vs after adaptive enhancement",
        ylabel="Good-frame ratio",
        output_path=good_ratio_path,
    )
    saved_paths.append(good_ratio_path)

    examples = per_sweep_df.sort_values(
        ["poor_frame_ratio_reduction", "good_frame_ratio_improvement"],
        ascending=[False, False],
    ).head(max_example_sweeps)
    for idx, (_, row) in enumerate(examples.iterrows(), start=1):
        output_path = ADAPTIVE_FIGURE_DIR / f"example_sweep_{idx:03d}_{safe_name(row['sweep_id'])}.png"
        plot_example_sweep(row, output_path)
        saved_paths.append(output_path)

    return saved_paths


def counts_to_markdown(counts: pd.Series) -> str:
    rows = ["| Label | Count |", "|---|---:|"]
    for label, count in counts.items():
        rows.append(f"| `{markdown_escape(label)}` | {int(count)} |")
    if len(rows) == 2:
        rows.append("| _none_ | 0 |")
    return "\n".join(rows)


def per_sweep_to_markdown(per_sweep_df: pd.DataFrame) -> str:
    rows = [
        "| Sweep ID | Frames | Original feedback | Adaptive feedback | Poor ratio before | Poor ratio after | Poor ratio reduction | Good ratio before | Good ratio after | Good ratio improvement | Reason |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in per_sweep_df.sort_values("sweep_id").itertuples(index=False):
        rows.append(
            "| "
            f"{markdown_escape(getattr(row, 'sweep_id'))} | "
            f"{int(getattr(row, 'total_frames'))} | "
            f"`{markdown_escape(getattr(row, 'original_feedback_label'))}` | "
            f"`{markdown_escape(getattr(row, 'adaptive_feedback_label'))}` | "
            f"{float(getattr(row, 'original_poor_frame_ratio')):.4f} | "
            f"{float(getattr(row, 'adaptive_poor_frame_ratio')):.4f} | "
            f"{float(getattr(row, 'poor_frame_ratio_reduction')):.4f} | "
            f"{float(getattr(row, 'original_good_frame_ratio')):.4f} | "
            f"{float(getattr(row, 'adaptive_good_frame_ratio')):.4f} | "
            f"{float(getattr(row, 'good_frame_ratio_improvement')):.4f} | "
            f"{markdown_escape(getattr(row, 'feedback_change_reason'))} |"
        )
    return "\n".join(rows)


def write_markdown_report(
    per_sweep_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    figure_paths: list[Path],
    adaptive_per_frame_path: Path,
    sweep_summary_path: Path,
    proxy_frame_manifest_path: Path,
    poor_ratio_threshold: float,
    low_contrast_ratio_threshold: float,
) -> None:
    summary = summary_df.iloc[0]
    original_counts = per_sweep_df["original_feedback_label"].value_counts().sort_index()
    adaptive_counts = per_sweep_df["adaptive_feedback_label"].value_counts().sort_index()
    figure_links = "\n".join(f"- `{rel_path(path)}`" for path in figure_paths)

    report_text = f"""# ACOUSLIC-AI Adaptive Sweep Feedback Change Analysis

{PROXY_LABEL_NOTE}

This report uses Stage 28 adaptive-frame labels to estimate whether adaptive
super-resolution enhancement changes sweep-level proxy feedback decisions.
Motion-based labels such as `move_slower` and `keep_probe_steady` are preserved,
because image enhancement does not change probe motion.

## Inputs

- Stage 28 adaptive per-frame enhancement CSV: `{rel_path(adaptive_per_frame_path)}`
- Original sweep-quality summary CSV: `{rel_path(sweep_summary_path)}`
- Original frame proxy-quality manifest CSV: `{rel_path(proxy_frame_manifest_path)}`

## Decision rules

- Poor-like frame labels: `poor_frame`, `blurry_frame`, `low_contrast_frame`, `noisy_frame`
- Adaptive `poor_frame_ratio` > {poor_ratio_threshold:.2f}: `repeat_sweep`
- Adaptive `low_contrast_frame` ratio > {low_contrast_ratio_threshold:.2f}: `improve_visibility`
- Original `move_slower` / `keep_probe_steady`: preserved as motion-based feedback
- Otherwise: `acceptable`

## Summary

- Number of sweeps analyzed: {int(summary["number_of_sweeps_analyzed"])}
- Number of sweeps with reduced poor-frame ratio: {int(summary["number_of_sweeps_with_reduced_poor_frame_ratio"])}
- Mean poor-frame-ratio reduction: {float(summary["mean_poor_frame_ratio_reduction"]):.6f}
- Number of sweeps with improved good-frame ratio: {int(summary["number_of_sweeps_with_improved_good_frame_ratio"])}
- Number of `repeat_sweep` cases changed to `acceptable` or `improve_visibility`: {int(summary["number_of_repeat_sweep_cases_changed_to_acceptable_or_improve_visibility"])}
- Number of feedback decisions unchanged due to motion-based problems: {int(summary["number_of_feedback_decisions_unchanged_due_to_motion_based_problems"])}

## Original sweep-feedback distribution

{counts_to_markdown(original_counts)}

## Adaptive sweep-feedback distribution

{counts_to_markdown(adaptive_counts)}

## Per-sweep results

{per_sweep_to_markdown(per_sweep_df)}

## Outputs

- Per-sweep CSV: `{rel_path(ADAPTIVE_PER_SWEEP_CSV)}`
- Summary CSV: `{rel_path(ADAPTIVE_SUMMARY_CSV)}`
- Figure folder: `{rel_path(ADAPTIVE_FIGURE_DIR)}`

## Saved figures

{figure_links}
"""

    ADAPTIVE_REPORT_MD.write_text(report_text, encoding="utf-8")


def run(args: argparse.Namespace) -> list[Path]:
    if not 0.0 <= args.poor_ratio_threshold <= 1.0:
        raise ValueError("--poor_ratio_threshold must be between 0 and 1.")
    if not 0.0 <= args.low_contrast_ratio_threshold <= 1.0:
        raise ValueError("--low_contrast_ratio_threshold must be between 0 and 1.")
    if args.max_example_sweeps < 0:
        raise ValueError("--max_example_sweeps must be 0 or greater.")

    create_output_dirs()

    adaptive_per_frame_path = resolve_project_path(args.adaptive_per_frame)
    sweep_summary_path = resolve_project_path(args.sweep_summary)
    proxy_frame_manifest_path = resolve_project_path(args.proxy_frame_manifest)

    if not adaptive_per_frame_path.exists():
        raise FileNotFoundError(
            f"Stage 28 adaptive per-frame enhancement CSV not found: {adaptive_per_frame_path}"
        )
    if not sweep_summary_path.exists():
        raise FileNotFoundError(f"Sweep-quality summary CSV not found: {sweep_summary_path}")
    if not proxy_frame_manifest_path.exists():
        raise FileNotFoundError(
            f"Original proxy frame manifest CSV not found: {proxy_frame_manifest_path}"
        )

    adaptive_df = pd.read_csv(
        adaptive_per_frame_path,
        dtype={
            "sweep_id": "string",
            "frame_id": "string",
            "original_frame_quality_label": "string",
            "adaptive_frame_quality_label": "string",
        },
    )
    sweep_summary_df = pd.read_csv(
        sweep_summary_path,
        dtype={
            "sweep_id": "string",
            "final_sweep_feedback_label": "string",
        },
    )
    proxy_frame_df = pd.read_csv(
        proxy_frame_manifest_path,
        dtype={
            "sweep_id": "string",
            "frame_id": "string",
        },
    )

    validate_columns(
        adaptive_df,
        [
            "sweep_id",
            "frame_id",
            "original_frame_quality_label",
            "adaptive_frame_quality_label",
        ],
        adaptive_per_frame_path,
    )
    validate_columns(
        sweep_summary_df,
        [
            "sweep_id",
            "mean_optical_flow",
            "mean_jitter_score",
            "final_sweep_feedback_label",
        ],
        sweep_summary_path,
    )
    validate_columns(proxy_frame_df, ["sweep_id", "frame_id"], proxy_frame_manifest_path)

    stage28_sweep_count = int(adaptive_df["sweep_id"].nunique())
    print(
        "Stage 28 per-frame rows loaded: "
        f"{len(adaptive_df)} frames across {stage28_sweep_count} sweeps"
    )

    stage28_keys = set(zip(adaptive_df["sweep_id"], adaptive_df["frame_id"]))
    proxy_keys = set(zip(proxy_frame_df["sweep_id"], proxy_frame_df["frame_id"]))
    missing_from_proxy = len(stage28_keys - proxy_keys)
    if missing_from_proxy:
        print(
            "WARNING: "
            f"{missing_from_proxy} Stage 28 frame rows were not found in the original proxy manifest."
        )

    per_sweep_df = build_per_sweep_dataframe(
        enhancement_df=adaptive_df,
        sweep_summary_df=sweep_summary_df,
        poor_ratio_threshold=args.poor_ratio_threshold,
        low_contrast_ratio_threshold=args.low_contrast_ratio_threshold,
    )
    if per_sweep_df.empty:
        raise ValueError("No sweeps were available for analysis.")

    summary_df = build_summary_dataframe(per_sweep_df)

    per_sweep_df.to_csv(ADAPTIVE_PER_SWEEP_CSV, index=False, float_format="%.8f")
    summary_df.to_csv(ADAPTIVE_SUMMARY_CSV, index=False, float_format="%.8f")

    figure_paths = create_figures(
        per_sweep_df=per_sweep_df,
        max_example_sweeps=args.max_example_sweeps,
    )
    write_markdown_report(
        per_sweep_df=per_sweep_df,
        summary_df=summary_df,
        figure_paths=figure_paths,
        adaptive_per_frame_path=adaptive_per_frame_path,
        sweep_summary_path=sweep_summary_path,
        proxy_frame_manifest_path=proxy_frame_manifest_path,
        poor_ratio_threshold=args.poor_ratio_threshold,
        low_contrast_ratio_threshold=args.low_contrast_ratio_threshold,
    )

    saved_paths = [
        ADAPTIVE_PER_SWEEP_CSV,
        ADAPTIVE_SUMMARY_CSV,
        ADAPTIVE_REPORT_MD,
        ADAPTIVE_FIGURE_DIR,
        *figure_paths,
    ]

    summary = summary_df.iloc[0]
    print(f"Number of sweeps analyzed: {int(summary['number_of_sweeps_analyzed'])}")
    print(
        "Number of sweeps with reduced poor-frame ratio: "
        f"{int(summary['number_of_sweeps_with_reduced_poor_frame_ratio'])}"
    )
    print(
        "Mean poor-frame-ratio reduction: "
        f"{float(summary['mean_poor_frame_ratio_reduction']):.6f}"
    )
    print(
        "Number of sweeps with improved good-frame ratio: "
        f"{int(summary['number_of_sweeps_with_improved_good_frame_ratio'])}"
    )
    print(
        "Number of repeat_sweep cases changed to acceptable or improve_visibility: "
        f"{int(summary['number_of_repeat_sweep_cases_changed_to_acceptable_or_improve_visibility'])}"
    )
    print(
        "Number of feedback decisions unchanged due to motion-based problems: "
        f"{int(summary['number_of_feedback_decisions_unchanged_due_to_motion_based_problems'])}"
    )
    print(PROXY_LABEL_NOTE)
    print("Saved output paths:")
    for path in saved_paths:
        print(f"  {rel_path(path)}")

    return saved_paths


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
