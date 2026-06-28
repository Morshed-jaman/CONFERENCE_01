"""
Compute bootstrap confidence intervals for HC18 and ACOUSLIC-AI results.

The confidence intervals summarize benchmark/proxy metrics only. They are not
clinical validation.
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

DEFAULT_HC18_PER_IMAGE = Path("results/final_all_hc18_comparison_per_image.csv")
DEFAULT_ANATOMY_PER_IMAGE = Path(
    "results/anatomy_preservation_all_models_per_image.csv"
)
DEFAULT_ACOUSLIC_PER_SWEEP = Path(
    "results/acouslic_ai/adaptive_sweep_feedback_change_per_sweep.csv"
)
DEFAULT_ACOUSLIC_PER_FRAME = Path(
    "results/acouslic_ai/adaptive_feedback_enhancement_per_frame.csv"
)

OUTPUT_CSV = Path("results/bootstrap_confidence_intervals.csv")
OUTPUT_MD = Path("results/bootstrap_confidence_intervals.md")
FIGURE_DIR = Path("results/bootstrap_ci_figures")

HC18_METRICS = {
    "psnr": "PSNR",
    "ssim": "SSIM",
    "mae": "MAE",
    "edge_score": "edge_score",
}

ANATOMY_METRICS = {
    "mae_inside_mask": "inside-mask MAE",
    "boundary_error": "boundary error",
    "edge_similarity_boundary": "boundary edge similarity",
}

ACOUSLIC_METRICS = {
    "improved_label_flag": "frame improvement percentage",
    "poor_frame_ratio_reduction": "poor-frame-ratio reduction",
    "good_frame_ratio_improvement": "good-frame-ratio improvement",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute bootstrap confidence intervals for final result tables."
    )
    parser.add_argument(
        "--hc18_per_image",
        type=Path,
        default=DEFAULT_HC18_PER_IMAGE,
        help="HC18 final comparison per-image CSV.",
    )
    parser.add_argument(
        "--anatomy_per_image",
        type=Path,
        default=DEFAULT_ANATOMY_PER_IMAGE,
        help="Anatomy preservation per-image CSV.",
    )
    parser.add_argument(
        "--acouslic_per_sweep",
        type=Path,
        default=DEFAULT_ACOUSLIC_PER_SWEEP,
        help="ACOUSLIC adaptive sweep feedback change per-sweep CSV.",
    )
    parser.add_argument(
        "--acouslic_per_frame",
        type=Path,
        default=DEFAULT_ACOUSLIC_PER_FRAME,
        help="ACOUSLIC adaptive enhancement per-frame CSV.",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resamples.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
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


def bootstrap_mean_ci(
    values: np.ndarray | pd.Series,
    rng: np.random.Generator,
    n_bootstrap: int,
    scale: float = 1.0,
) -> tuple[float, float, float, int]:
    clean_values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(
        dtype=np.float64
    )
    if clean_values.size == 0:
        return np.nan, np.nan, np.nan, 0

    means = np.empty(n_bootstrap, dtype=np.float64)
    sample_count = clean_values.size
    for index in range(n_bootstrap):
        sample_indices = rng.integers(0, sample_count, size=sample_count)
        means[index] = clean_values[sample_indices].mean()

    point_estimate = clean_values.mean() * scale
    lower, upper = np.percentile(means * scale, [2.5, 97.5])
    return float(point_estimate), float(lower), float(upper), int(sample_count)


def ci_row(
    source: str,
    method: str,
    metric: str,
    values: np.ndarray | pd.Series,
    rng: np.random.Generator,
    n_bootstrap: int,
    unit: str = "",
    scale: float = 1.0,
    note: str = "",
) -> dict[str, Any]:
    mean, lower, upper, sample_count = bootstrap_mean_ci(
        values=values,
        rng=rng,
        n_bootstrap=n_bootstrap,
        scale=scale,
    )
    return {
        "source": source,
        "method": method,
        "metric": metric,
        "n": sample_count,
        "mean": mean,
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_level": "95%",
        "n_bootstrap": n_bootstrap,
        "unit": unit,
        "note": note,
    }


def compute_hc18_rows(
    hc18_df: pd.DataFrame,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, group in hc18_df.groupby("method", sort=True):
        for column, metric_name in HC18_METRICS.items():
            rows.append(
                ci_row(
                    source="HC18 reconstruction",
                    method=str(method),
                    metric=metric_name,
                    values=group[column],
                    rng=rng,
                    n_bootstrap=n_bootstrap,
                    unit="",
                )
            )
    return rows


def compute_anatomy_rows(
    anatomy_df: pd.DataFrame,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, group in anatomy_df.groupby("method", sort=True):
        for column, metric_name in ANATOMY_METRICS.items():
            rows.append(
                ci_row(
                    source="HC18 anatomy preservation",
                    method=str(method),
                    metric=metric_name,
                    values=group[column],
                    rng=rng,
                    n_bootstrap=n_bootstrap,
                    unit="",
                )
            )
    return rows


def compute_acouslic_rows(
    per_frame_df: pd.DataFrame,
    per_sweep_df: pd.DataFrame,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> list[dict[str, Any]]:
    rows = [
        ci_row(
            source="ACOUSLIC-AI adaptive enhancement",
            method="Heuristic adaptive enhancement",
            metric=ACOUSLIC_METRICS["improved_label_flag"],
            values=per_frame_df["improved_label_flag"],
            rng=rng,
            n_bootstrap=n_bootstrap,
            unit="%",
            scale=100.0,
            note="Bootstrap over frames.",
        ),
        ci_row(
            source="ACOUSLIC-AI sweep feedback",
            method="Heuristic adaptive enhancement",
            metric=ACOUSLIC_METRICS["poor_frame_ratio_reduction"],
            values=per_sweep_df["poor_frame_ratio_reduction"],
            rng=rng,
            n_bootstrap=n_bootstrap,
            unit="percentage points",
            scale=100.0,
            note="Bootstrap over sweeps.",
        ),
        ci_row(
            source="ACOUSLIC-AI sweep feedback",
            method="Heuristic adaptive enhancement",
            metric=ACOUSLIC_METRICS["good_frame_ratio_improvement"],
            values=per_sweep_df["good_frame_ratio_improvement"],
            rng=rng,
            n_bootstrap=n_bootstrap,
            unit="percentage points",
            scale=100.0,
            note="Bootstrap over sweeps.",
        ),
    ]
    return rows


def markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|")


def format_number(value: Any) -> str:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return markdown_escape(value)

    if np.isnan(value_float):
        return "N/A"
    return f"{value_float:.4f}".rstrip("0").rstrip(".")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [markdown_escape(column) for column in df.columns]
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, int, np.integer, np.floating)):
                values.append(format_number(value))
            else:
                values.append(markdown_escape(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def compact_ci_table(df: pd.DataFrame, source: str) -> pd.DataFrame:
    subset = df[df["source"] == source].copy()
    return subset[
        ["method", "metric", "n", "mean", "ci_lower", "ci_upper", "unit"]
    ].rename(
        columns={
            "method": "Method",
            "metric": "Metric",
            "n": "N",
            "mean": "Mean",
            "ci_lower": "95% CI lower",
            "ci_upper": "95% CI upper",
            "unit": "Unit",
        }
    )


def plot_ci_for_source(
    ci_df: pd.DataFrame,
    source: str,
    metrics: list[str],
    output_path: Path,
    title: str,
) -> None:
    source_df = ci_df[ci_df["source"] == source].copy()
    if source_df.empty:
        return

    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(10, max(4, 3.1 * len(metrics))),
        squeeze=False,
    )

    for axis, metric in zip(axes[:, 0], metrics, strict=False):
        metric_df = source_df[source_df["metric"] == metric].copy()
        if metric_df.empty:
            axis.set_visible(False)
            continue

        metric_df = metric_df.sort_values("method")
        means = metric_df["mean"].to_numpy(dtype=np.float64)
        lowers = metric_df["ci_lower"].to_numpy(dtype=np.float64)
        uppers = metric_df["ci_upper"].to_numpy(dtype=np.float64)
        x = np.arange(len(metric_df))
        yerr = np.vstack([means - lowers, uppers - means])
        yerr = np.maximum(yerr, 0.0)

        axis.errorbar(
            x,
            means,
            yerr=yerr,
            fmt="o",
            capsize=4,
            color="#4C78A8",
            ecolor="#F58518",
        )
        axis.set_xticks(x)
        axis.set_xticklabels(metric_df["method"], rotation=25, ha="right")
        axis.set_title(metric)
        axis.grid(axis="y", alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_acouslic_ci(ci_df: pd.DataFrame, output_path: Path) -> None:
    source_df = ci_df[ci_df["source"].str.startswith("ACOUSLIC-AI")].copy()
    if source_df.empty:
        return

    source_df = source_df.sort_values("metric")
    means = source_df["mean"].to_numpy(dtype=np.float64)
    lowers = source_df["ci_lower"].to_numpy(dtype=np.float64)
    uppers = source_df["ci_upper"].to_numpy(dtype=np.float64)
    x = np.arange(len(source_df))
    yerr = np.vstack([means - lowers, uppers - means])
    yerr = np.maximum(yerr, 0.0)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(
        x,
        means,
        yerr=yerr,
        fmt="o",
        capsize=5,
        color="#54A24B",
        ecolor="#F58518",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(source_df["metric"], rotation=25, ha="right")
    ax.set_ylabel("% / percentage points")
    ax.set_title("ACOUSLIC-AI Bootstrap 95% Confidence Intervals")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_figures(ci_df: pd.DataFrame) -> list[Path]:
    figure_dir = resolve_project_path(FIGURE_DIR)
    figure_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        figure_dir / "hc18_reconstruction_bootstrap_ci.png",
        figure_dir / "hc18_anatomy_bootstrap_ci.png",
        figure_dir / "acouslic_ai_bootstrap_ci.png",
    ]

    plot_ci_for_source(
        ci_df=ci_df,
        source="HC18 reconstruction",
        metrics=list(HC18_METRICS.values()),
        output_path=paths[0],
        title="HC18 Reconstruction Bootstrap 95% Confidence Intervals",
    )
    plot_ci_for_source(
        ci_df=ci_df,
        source="HC18 anatomy preservation",
        metrics=list(ANATOMY_METRICS.values()),
        output_path=paths[1],
        title="HC18 Anatomy Bootstrap 95% Confidence Intervals",
    )
    plot_acouslic_ci(ci_df=ci_df, output_path=paths[2])
    plt.close("all")

    return paths


def write_markdown_report(
    ci_df: pd.DataFrame,
    figure_paths: list[Path],
    args: argparse.Namespace,
) -> None:
    figure_lines = "\n".join(f"- `{rel_path(path)}`" for path in figure_paths)
    acouslic_df = ci_df[ci_df["source"].str.startswith("ACOUSLIC-AI")].copy()
    acouslic_table = acouslic_df[
        ["source", "metric", "n", "mean", "ci_lower", "ci_upper", "unit", "note"]
    ].rename(
        columns={
            "source": "Source",
            "metric": "Metric",
            "n": "N",
            "mean": "Mean",
            "ci_lower": "95% CI lower",
            "ci_upper": "95% CI upper",
            "unit": "Unit",
            "note": "Note",
        }
    )

    report = f"""# Bootstrap Confidence Intervals

Bootstrap settings: {args.n_bootstrap} resamples, random seed {args.seed}.

These confidence intervals summarize benchmark/proxy metrics only. They are not
clinical validation.

## Inputs

- HC18 per-image comparison: `{rel_path(resolve_project_path(args.hc18_per_image))}`
- Anatomy preservation per-image metrics: `{rel_path(resolve_project_path(args.anatomy_per_image))}`
- ACOUSLIC-AI per-sweep feedback changes: `{rel_path(resolve_project_path(args.acouslic_per_sweep))}`
- ACOUSLIC-AI per-frame adaptive enhancement metrics: `{rel_path(resolve_project_path(args.acouslic_per_frame))}`

## HC18 Reconstruction CIs

{dataframe_to_markdown(compact_ci_table(ci_df, "HC18 reconstruction"))}

## HC18 Anatomy Preservation CIs

{dataframe_to_markdown(compact_ci_table(ci_df, "HC18 anatomy preservation"))}

## ACOUSLIC-AI CIs

{dataframe_to_markdown(acouslic_table)}

## Outputs

- CSV: `{rel_path(resolve_project_path(OUTPUT_CSV))}`
- Figure directory: `{rel_path(resolve_project_path(FIGURE_DIR))}`

## Figures

{figure_lines}
"""

    resolve_project_path(OUTPUT_MD).write_text(report, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    if args.n_bootstrap <= 0:
        raise ValueError("--n_bootstrap must be greater than 0.")

    hc18_df = read_csv_required(
        args.hc18_per_image,
        ["image_id", "method", *HC18_METRICS.keys()],
    )
    anatomy_df = read_csv_required(
        args.anatomy_per_image,
        ["image_id", "method", *ANATOMY_METRICS.keys()],
    )
    acouslic_sweep_df = read_csv_required(
        args.acouslic_per_sweep,
        [
            "sweep_id",
            "poor_frame_ratio_reduction",
            "good_frame_ratio_improvement",
        ],
    )
    acouslic_frame_df = read_csv_required(
        args.acouslic_per_frame,
        ["sweep_id", "frame_id", "improved_label_flag"],
    )

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    rows.extend(compute_hc18_rows(hc18_df, rng=rng, n_bootstrap=args.n_bootstrap))
    rows.extend(compute_anatomy_rows(anatomy_df, rng=rng, n_bootstrap=args.n_bootstrap))
    rows.extend(
        compute_acouslic_rows(
            per_frame_df=acouslic_frame_df,
            per_sweep_df=acouslic_sweep_df,
            rng=rng,
            n_bootstrap=args.n_bootstrap,
        )
    )

    ci_df = pd.DataFrame(rows)
    output_csv = resolve_project_path(OUTPUT_CSV)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    ci_df.to_csv(output_csv, index=False, float_format="%.8f")

    figure_paths = create_figures(ci_df)
    write_markdown_report(ci_df=ci_df, figure_paths=figure_paths, args=args)

    print(f"Bootstrap confidence intervals CSV: {OUTPUT_CSV}")
    print(f"Bootstrap confidence intervals Markdown: {OUTPUT_MD}")
    print(f"Figure directory: {FIGURE_DIR}")
    print("Generated figures:")
    for path in figure_paths:
        print(f"  {rel_path(path)}")
    print("Note: intervals summarize proxy/benchmark metrics, not clinical validation.")


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
