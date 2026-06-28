"""
Create final ablation tables for the acquisition-aware fetal ultrasound framework.

The tables aggregate proxy ACOUSLIC-AI enhancement summaries and HC18
super-resolution/anatomy preservation metrics. These results are proxy and
benchmark metrics; they should not be presented as expert clinical validation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BLIND_SUMMARY = Path("results/acouslic_ai/feedback_aware_enhancement_summary.csv")
DEFAULT_HEURISTIC_SUMMARY = Path(
    "results/acouslic_ai/adaptive_feedback_enhancement_summary.csv"
)
DEFAULT_LEARNED_SUMMARY = Path(
    "results/acouslic_ai/learned_adaptive_enhancement_summary.csv"
)
DEFAULT_SWEEP_CHANGE_SUMMARY = Path(
    "results/acouslic_ai/adaptive_sweep_feedback_change_summary.csv"
)
DEFAULT_HC18_SUMMARY = Path("results/final_all_hc18_comparison_summary.csv")
DEFAULT_ANATOMY_SUMMARY = Path("results/anatomy_preservation_all_models_summary.csv")

OUTPUT_CSV = Path("results/final_ablation_study.csv")
OUTPUT_MD = Path("results/final_ablation_study.md")
FIGURE_DIR = Path("results/final_ablation_figures")

ACQUISITION_COLUMNS = [
    "Method",
    "Uses quality estimation",
    "Uses adaptive action",
    "Uses SR/CNN",
    "Uses diffusion refinement",
    "Frame improvement %",
    "Sweeps with reduced poor-frame ratio",
    "Repeat_sweep cases improved",
    "Notes",
]

HC18_METHODS = [
    "SR-only CNN",
    "Quality-aware CNN",
    "Quality-aware CNN + L1/MSE",
    "Conditional Diffusion SR",
    "CNN-Residual Diffusion SR",
]

COMBINED_COLUMNS = [
    "Table",
    *ACQUISITION_COLUMNS,
    "Mean PSNR",
    "Mean SSIM",
    "Mean MAE",
    "Mean edge score",
    "Mean anatomy MAE inside mask",
    "Mean boundary error",
    "Mean contrast preservation",
    "Mean boundary edge similarity",
]

NO_CLINICAL_NOTE = "Proxy/benchmark metrics only; not expert clinical validation."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create final acquisition-aware framework ablation tables."
    )
    parser.add_argument(
        "--blind_summary",
        type=Path,
        default=DEFAULT_BLIND_SUMMARY,
        help="Blind feedback-aware enhancement summary CSV.",
    )
    parser.add_argument(
        "--heuristic_summary",
        type=Path,
        default=DEFAULT_HEURISTIC_SUMMARY,
        help="Heuristic adaptive enhancement summary CSV.",
    )
    parser.add_argument(
        "--learned_summary",
        type=Path,
        default=DEFAULT_LEARNED_SUMMARY,
        help="Learned adaptive enhancement summary CSV.",
    )
    parser.add_argument(
        "--sweep_change_summary",
        type=Path,
        default=DEFAULT_SWEEP_CHANGE_SUMMARY,
        help="Adaptive sweep feedback change summary CSV.",
    )
    parser.add_argument(
        "--hc18_summary",
        type=Path,
        default=DEFAULT_HC18_SUMMARY,
        help="HC18 final model comparison summary CSV.",
    )
    parser.add_argument(
        "--anatomy_summary",
        type=Path,
        default=DEFAULT_ANATOMY_SUMMARY,
        help="Anatomy preservation all-model summary CSV.",
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


def read_csv_required(path_value: Path | str, required_columns: list[str]) -> pd.DataFrame:
    path = resolve_project_path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Required input CSV not found: {path}")

    df = pd.read_csv(path)
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return df


def get_overall_percentage(summary_df: pd.DataFrame) -> float:
    if "scope" not in summary_df.columns or "percentage_improved" not in summary_df.columns:
        return np.nan

    overall = summary_df[summary_df["scope"].astype(str) == "overall"]
    if overall.empty:
        return np.nan
    return float(overall.iloc[0]["percentage_improved"])


def get_sweep_value(sweep_df: pd.DataFrame, column: str) -> Any:
    if column not in sweep_df.columns or sweep_df.empty:
        return "N/A"
    value = sweep_df.iloc[0][column]
    if pd.isna(value):
        return "N/A"
    return int(value)


def format_percent_value(value: Any) -> Any:
    if value == "N/A" or pd.isna(value):
        return "N/A"
    return round(float(value), 2)


def numeric_or_nan(value: Any) -> float:
    if value == "N/A" or value is None:
        return np.nan
    try:
        if pd.isna(value):
            return np.nan
    except TypeError:
        return np.nan
    return float(value)


def blank_metric_row() -> dict[str, Any]:
    return {
        "Mean PSNR": "N/A",
        "Mean SSIM": "N/A",
        "Mean MAE": "N/A",
        "Mean edge score": "N/A",
        "Mean anatomy MAE inside mask": "N/A",
        "Mean boundary error": "N/A",
        "Mean contrast preservation": "N/A",
        "Mean boundary edge similarity": "N/A",
    }


def build_acquisition_ablation(
    blind_summary: pd.DataFrame,
    heuristic_summary: pd.DataFrame,
    learned_summary: pd.DataFrame,
    sweep_summary: pd.DataFrame,
    hc18_summary: pd.DataFrame,
    anatomy_summary: pd.DataFrame,
) -> pd.DataFrame:
    blind_improvement = format_percent_value(get_overall_percentage(blind_summary))
    heuristic_improvement = format_percent_value(get_overall_percentage(heuristic_summary))
    learned_improvement = format_percent_value(get_overall_percentage(learned_summary))

    sweeps_reduced = get_sweep_value(
        sweep_summary,
        "number_of_sweeps_with_reduced_poor_frame_ratio",
    )
    repeat_improved = get_sweep_value(
        sweep_summary,
        "number_of_repeat_sweep_cases_changed_to_acceptable_or_improve_visibility",
    )

    cnn_residual = model_row_lookup(hc18_summary, anatomy_summary, "CNN-Residual Diffusion SR")
    cnn_note = (
        "HC18 benchmark SR method; "
        f"PSNR {cnn_residual.get('Mean PSNR', 'N/A')}, "
        f"SSIM {cnn_residual.get('Mean SSIM', 'N/A')}, "
        f"boundary edge similarity {cnn_residual.get('Mean boundary edge similarity', 'N/A')}. "
        f"{NO_CLINICAL_NOTE}"
    )

    rows = [
        {
            "Method": "No enhancement",
            "Uses quality estimation": "No",
            "Uses adaptive action": "No",
            "Uses SR/CNN": "No",
            "Uses diffusion refinement": "No",
            "Frame improvement %": 0.0,
            "Sweeps with reduced poor-frame ratio": 0,
            "Repeat_sweep cases improved": 0,
            "Notes": f"Baseline reference with no image modification. {NO_CLINICAL_NOTE}",
        },
        {
            "Method": "Blind enhancement",
            "Uses quality estimation": "No adaptive decision",
            "Uses adaptive action": "No",
            "Uses SR/CNN": "Yes",
            "Uses diffusion refinement": "No",
            "Frame improvement %": blind_improvement,
            "Sweeps with reduced poor-frame ratio": "N/A",
            "Repeat_sweep cases improved": "N/A",
            "Notes": (
                "Quality-aware CNN enhancement applied without an adaptive "
                f"frame-action policy. {NO_CLINICAL_NOTE}"
            ),
        },
        {
            "Method": "Heuristic adaptive enhancement",
            "Uses quality estimation": "Heuristic proxy labels",
            "Uses adaptive action": "Yes",
            "Uses SR/CNN": "Yes",
            "Uses diffusion refinement": "No",
            "Frame improvement %": heuristic_improvement,
            "Sweeps with reduced poor-frame ratio": sweeps_reduced,
            "Repeat_sweep cases improved": repeat_improved,
            "Notes": (
                "Uses proxy frame labels to keep, enhance, denoise, or request "
                f"repeat/reacquisition. {NO_CLINICAL_NOTE}"
            ),
        },
        {
            "Method": "Learned adaptive enhancement",
            "Uses quality estimation": "Learned proxy embedding",
            "Uses adaptive action": "Yes",
            "Uses SR/CNN": "Yes",
            "Uses diffusion refinement": "No",
            "Frame improvement %": learned_improvement,
            "Sweeps with reduced poor-frame ratio": "N/A",
            "Repeat_sweep cases improved": "N/A",
            "Notes": (
                "Uses learned proxy-quality MLP confidence for conservative "
                f"frame actions. Sweep-level change was not recomputed here. {NO_CLINICAL_NOTE}"
            ),
        },
        {
            "Method": "CNN-residual diffusion SR",
            "Uses quality estimation": "No acquisition feedback",
            "Uses adaptive action": "No",
            "Uses SR/CNN": "Yes",
            "Uses diffusion refinement": "Yes",
            "Frame improvement %": "N/A",
            "Sweeps with reduced poor-frame ratio": "N/A",
            "Repeat_sweep cases improved": "N/A",
            "Notes": cnn_note,
        },
    ]

    return pd.DataFrame(rows, columns=ACQUISITION_COLUMNS)


def model_row_lookup(
    hc18_summary: pd.DataFrame,
    anatomy_summary: pd.DataFrame,
    method: str,
) -> dict[str, Any]:
    hc18_match = hc18_summary[hc18_summary["method"].astype(str) == method]
    anatomy_match = anatomy_summary[anatomy_summary["method"].astype(str) == method]

    row: dict[str, Any] = {
        "Mean PSNR": "N/A",
        "Mean SSIM": "N/A",
        "Mean MAE": "N/A",
        "Mean edge score": "N/A",
        "Mean anatomy MAE inside mask": "N/A",
        "Mean boundary error": "N/A",
        "Mean contrast preservation": "N/A",
        "Mean boundary edge similarity": "N/A",
    }

    if not hc18_match.empty:
        hc18 = hc18_match.iloc[0]
        row.update(
            {
                "Mean PSNR": round(float(hc18["mean_psnr"]), 4),
                "Mean SSIM": round(float(hc18["mean_ssim"]), 4),
                "Mean MAE": round(float(hc18["mean_mae"]), 6),
                "Mean edge score": round(float(hc18["mean_edge_score"]), 4),
            }
        )

    if not anatomy_match.empty:
        anatomy = anatomy_match.iloc[0]
        row.update(
            {
                "Mean anatomy MAE inside mask": round(
                    float(anatomy["mean_mae_inside_mask"]), 6
                ),
                "Mean boundary error": round(float(anatomy["mean_boundary_error"]), 6),
                "Mean contrast preservation": round(
                    float(anatomy["mean_contrast_preservation"]), 4
                ),
                "Mean boundary edge similarity": round(
                    float(anatomy["mean_edge_similarity_boundary"]), 4
                ),
            }
        )

    return row


def build_hc18_ablation(
    hc18_summary: pd.DataFrame,
    anatomy_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    flags = {
        "SR-only CNN": ("No", "No", "Yes", "No"),
        "Quality-aware CNN": ("Yes", "No", "Yes", "No"),
        "Quality-aware CNN + L1/MSE": ("Yes", "No", "Yes", "No"),
        "Conditional Diffusion SR": ("No", "No", "No", "Yes"),
        "CNN-Residual Diffusion SR": ("No acquisition feedback", "No", "Yes", "Yes"),
    }

    for method in HC18_METHODS:
        quality, adaptive, sr_cnn, diffusion = flags[method]
        metrics = model_row_lookup(hc18_summary, anatomy_summary, method)
        row = {
            "Method": method,
            "Uses quality estimation": quality,
            "Uses adaptive action": adaptive,
            "Uses SR/CNN": sr_cnn,
            "Uses diffusion refinement": diffusion,
            "Frame improvement %": "N/A",
            "Sweeps with reduced poor-frame ratio": "N/A",
            "Repeat_sweep cases improved": "N/A",
            "Notes": (
                "HC18 supervised SR/anatomy preservation benchmark. "
                f"{NO_CLINICAL_NOTE}"
            ),
        }
        row.update(metrics)
        rows.append(row)

    return pd.DataFrame(rows)


def combine_tables(acquisition_df: pd.DataFrame, hc18_df: pd.DataFrame) -> pd.DataFrame:
    acquisition_combined = acquisition_df.copy()
    acquisition_combined.insert(0, "Table", "acquisition_framework")
    for column, value in blank_metric_row().items():
        acquisition_combined[column] = value

    hc18_combined = hc18_df.copy()
    hc18_combined.insert(0, "Table", "hc18_model_ablation")

    return pd.concat([acquisition_combined, hc18_combined], ignore_index=True)[
        COMBINED_COLUMNS
    ]


def markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|")


def format_markdown_value(value: Any) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return "N/A"
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return markdown_escape(value)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [markdown_escape(column) for column in df.columns]
    align = ["---" for _ in headers]
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(align) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(format_markdown_value(value) for value in row) + " |")
    return "\n".join(rows)


def plot_acquisition_improvement(acquisition_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = acquisition_df.copy()
    plot_df["_numeric_improvement"] = plot_df["Frame improvement %"].map(numeric_or_nan)
    plot_df = plot_df.dropna(subset=["_numeric_improvement"])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        plot_df["Method"],
        plot_df["_numeric_improvement"],
        color=["#9E9E9E", "#4C78A8", "#72B7B2", "#54A24B"],
    )
    ax.set_ylabel("Frame improvement (%)")
    ax.set_title("ACOUSLIC-AI Proxy Frame Improvement by Method")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=0.25)
    for index, value in enumerate(plot_df["_numeric_improvement"]):
        ax.text(index, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sweep_change(acquisition_df: pd.DataFrame, output_path: Path) -> None:
    methods = acquisition_df["Method"].tolist()
    sweeps = [numeric_or_nan(value) for value in acquisition_df["Sweeps with reduced poor-frame ratio"]]
    repeat = [numeric_or_nan(value) for value in acquisition_df["Repeat_sweep cases improved"]]
    sweeps = [0.0 if np.isnan(value) else value for value in sweeps]
    repeat = [0.0 if np.isnan(value) else value for value in repeat]

    x = np.arange(len(methods))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, sweeps, width, label="Reduced poor-frame ratio", color="#4C78A8")
    ax.bar(x + width / 2, repeat, width, label="Repeat_sweep improved", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("ACOUSLIC-AI Sweep-level Proxy Feedback Changes")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_hc18_metrics(hc18_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    methods = hc18_df["Method"].tolist()

    axes[0].bar(methods, hc18_df["Mean PSNR"].astype(float), color="#4C78A8")
    axes[0].set_title("HC18 Mean PSNR")
    axes[0].set_ylabel("PSNR")
    axes[0].tick_params(axis="x", labelrotation=30)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(methods, hc18_df["Mean SSIM"].astype(float), color="#72B7B2")
    axes[1].set_title("HC18 Mean SSIM")
    axes[1].set_ylabel("SSIM")
    axes[1].tick_params(axis="x", labelrotation=30)
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_anatomy_metrics(hc18_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    methods = hc18_df["Method"].tolist()

    axes[0].bar(methods, hc18_df["Mean boundary error"].astype(float), color="#E45756")
    axes[0].set_title("Anatomy Boundary Error")
    axes[0].set_ylabel("Mean boundary error")
    axes[0].tick_params(axis="x", labelrotation=30)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(
        methods,
        hc18_df["Mean boundary edge similarity"].astype(float),
        color="#54A24B",
    )
    axes[1].set_title("Boundary Edge Similarity")
    axes[1].set_ylabel("Mean edge similarity")
    axes[1].tick_params(axis="x", labelrotation=30)
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_figures(acquisition_df: pd.DataFrame, hc18_df: pd.DataFrame) -> list[Path]:
    figure_dir = resolve_project_path(FIGURE_DIR)
    figure_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        figure_dir / "acquisition_frame_improvement.png",
        figure_dir / "acquisition_sweep_feedback_changes.png",
        figure_dir / "hc18_psnr_ssim_ablation.png",
        figure_dir / "hc18_anatomy_preservation_ablation.png",
    ]

    plot_acquisition_improvement(acquisition_df, paths[0])
    plot_sweep_change(acquisition_df, paths[1])
    plot_hc18_metrics(hc18_df, paths[2])
    plot_anatomy_metrics(hc18_df, paths[3])
    plt.close("all")

    return paths


def write_markdown_report(
    acquisition_df: pd.DataFrame,
    hc18_df: pd.DataFrame,
    output_csv: Path,
    figure_paths: list[Path],
    args: argparse.Namespace,
) -> None:
    figure_lines = "\n".join(f"- `{rel_path(path)}`" for path in figure_paths)
    report = f"""# Final Ablation Study

These tables summarize the complete acquisition-aware fetal ultrasound
framework using proxy ACOUSLIC-AI quality metrics and HC18 benchmark metrics.
They should not be interpreted as expert clinical validation.

## Inputs

- Blind enhancement summary: `{rel_path(resolve_project_path(args.blind_summary))}`
- Heuristic adaptive summary: `{rel_path(resolve_project_path(args.heuristic_summary))}`
- Learned adaptive summary: `{rel_path(resolve_project_path(args.learned_summary))}`
- Adaptive sweep feedback summary: `{rel_path(resolve_project_path(args.sweep_change_summary))}`
- HC18 final comparison: `{rel_path(resolve_project_path(args.hc18_summary))}`
- Anatomy preservation summary: `{rel_path(resolve_project_path(args.anatomy_summary))}`

## Acquisition-aware Framework Ablation

{dataframe_to_markdown(acquisition_df)}

## HC18 Model Ablation

{dataframe_to_markdown(hc18_df)}

## Saved Outputs

- Combined CSV: `{rel_path(output_csv)}`
- Figure directory: `{rel_path(resolve_project_path(FIGURE_DIR))}`

## Figures

{figure_lines}

## Interpretation Note

The ACOUSLIC-AI rows use no-reference proxy quality labels and sweep-feedback
proxies. The HC18 rows use supervised reconstruction and anatomy-preservation
benchmark metrics. These are useful ablations for method comparison, but they
are not clinical validation.
"""

    resolve_project_path(OUTPUT_MD).write_text(report, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    blind_summary = read_csv_required(
        args.blind_summary,
        ["scope", "percentage_improved"],
    )
    heuristic_summary = read_csv_required(
        args.heuristic_summary,
        ["scope", "percentage_improved"],
    )
    learned_summary = read_csv_required(
        args.learned_summary,
        ["scope", "percentage_improved"],
    )
    sweep_summary = read_csv_required(
        args.sweep_change_summary,
        [
            "number_of_sweeps_with_reduced_poor_frame_ratio",
            "number_of_repeat_sweep_cases_changed_to_acceptable_or_improve_visibility",
        ],
    )
    hc18_summary = read_csv_required(
        args.hc18_summary,
        ["method", "mean_psnr", "mean_ssim", "mean_mae", "mean_edge_score"],
    )
    anatomy_summary = read_csv_required(
        args.anatomy_summary,
        [
            "method",
            "mean_mae_inside_mask",
            "mean_boundary_error",
            "mean_contrast_preservation",
            "mean_edge_similarity_boundary",
        ],
    )

    acquisition_df = build_acquisition_ablation(
        blind_summary=blind_summary,
        heuristic_summary=heuristic_summary,
        learned_summary=learned_summary,
        sweep_summary=sweep_summary,
        hc18_summary=hc18_summary,
        anatomy_summary=anatomy_summary,
    )
    hc18_df = build_hc18_ablation(hc18_summary, anatomy_summary)
    combined_df = combine_tables(acquisition_df, hc18_df)

    output_csv = resolve_project_path(OUTPUT_CSV)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_csv, index=False)

    figure_paths = create_figures(acquisition_df, hc18_df)
    write_markdown_report(
        acquisition_df=acquisition_df,
        hc18_df=hc18_df,
        output_csv=output_csv,
        figure_paths=figure_paths,
        args=args,
    )

    print(f"Final ablation CSV: {OUTPUT_CSV}")
    print(f"Final ablation Markdown: {OUTPUT_MD}")
    print(f"Figure directory: {FIGURE_DIR}")
    print("Generated figures:")
    for path in figure_paths:
        print(f"  {rel_path(path)}")
    print("Note: results are proxy/benchmark metrics, not clinical validation.")


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
