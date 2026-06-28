"""
Create final report tables and figures for the HC18 super-resolution experiment.

The script reads the metric summaries produced by the comparison and anatomy
evaluation scripts, normalizes method labels, and writes publication-ready
tables plus clean Matplotlib figures.
"""

import csv
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

# Use a non-interactive backend so the script works from Windows PowerShell.
matplotlib.use("Agg")

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(".")

SR_SUMMARY_CSV = PROJECT_ROOT / "results" / "final_hc18_comparison_summary.csv"
ANATOMY_SUMMARY_CSV = PROJECT_ROOT / "results" / "anatomy_preservation_summary.csv"
SR_SOURCE_MD = PROJECT_ROOT / "results" / "final_hc18_comparison_table.md"
ANATOMY_SOURCE_MD = PROJECT_ROOT / "results" / "anatomy_preservation_table.md"

OUTPUT_DIR = PROJECT_ROOT / "results" / "report_assets"

FINAL_SR_TABLE_CSV = OUTPUT_DIR / "final_sr_comparison_table.csv"
FINAL_SR_TABLE_MD = OUTPUT_DIR / "final_sr_comparison_table.md"
ANATOMY_TABLE_CSV = OUTPUT_DIR / "anatomy_preservation_table.csv"
ANATOMY_TABLE_MD = OUTPUT_DIR / "anatomy_preservation_table.md"

FIGURE_PSNR = OUTPUT_DIR / "figure_psnr_comparison.png"
FIGURE_SSIM = OUTPUT_DIR / "figure_ssim_comparison.png"
FIGURE_MAE = OUTPUT_DIR / "figure_mae_comparison.png"
FIGURE_ANATOMY_MAE_INSIDE = OUTPUT_DIR / "figure_anatomy_mae_inside_mask.png"
FIGURE_BOUNDARY_ERROR = OUTPUT_DIR / "figure_boundary_error.png"


METHOD_ORDER = [
    "LQ mixed baseline",
    "SR-only",
    "Quality-aware SR",
    "Quality-aware SR + L1/MSE",
]

METHOD_NAME_MAP = {
    "LQ_mixed baseline": "LQ mixed baseline",
    "LQ mixed baseline": "LQ mixed baseline",
    "SR-only": "SR-only",
    "Quality-aware SR": "Quality-aware SR",
    "Quality-aware SR + L1/MSE": "Quality-aware SR + L1/MSE",
}

BAR_COLORS = ["#7a7a7a", "#4c78a8", "#59a14f", "#f28e2b"]


def ensure_inputs_exist(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required input file(s):\n{missing_text}")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def normalize_method_name(method: str) -> str:
    method = method.strip()
    if method not in METHOD_NAME_MAP:
        raise ValueError(f"Unexpected method name: {method}")
    return METHOD_NAME_MAP[method]


def as_float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"Missing value for column '{key}'")
    return float(value)


def order_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_method = {str(row["method"]): row for row in rows}
    missing = [method for method in METHOD_ORDER if method not in by_method]
    if missing:
        raise ValueError(f"Missing expected method(s): {', '.join(missing)}")
    return [by_method[method] for method in METHOD_ORDER]


def build_sr_table(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    table: List[Dict[str, object]] = []
    for row in rows:
        table.append(
            {
                "method": normalize_method_name(row["method"]),
                "mean_psnr": as_float(row, "mean_psnr"),
                "std_psnr": as_float(row, "std_psnr"),
                "mean_ssim": as_float(row, "mean_ssim"),
                "std_ssim": as_float(row, "std_ssim"),
                "mean_mae": as_float(row, "mean_mae"),
                "std_mae": as_float(row, "std_mae"),
                "mean_edge_similarity": as_float(row, "mean_edge_score"),
                "std_edge_similarity": as_float(row, "std_edge_score"),
            }
        )
    return order_rows(table)


def build_anatomy_table(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    table: List[Dict[str, object]] = []
    for row in rows:
        table.append(
            {
                "method": normalize_method_name(row["method"]),
                "mean_mae_inside_mask": as_float(row, "mean_mae_inside_mask"),
                "std_mae_inside_mask": as_float(row, "std_mae_inside_mask"),
                "mean_mae_outside_mask": as_float(row, "mean_mae_outside_mask"),
                "std_mae_outside_mask": as_float(row, "std_mae_outside_mask"),
                "mean_boundary_error": as_float(row, "mean_boundary_error"),
                "std_boundary_error": as_float(row, "std_boundary_error"),
                "mean_contrast_preservation": as_float(row, "mean_contrast_preservation"),
                "std_contrast_preservation": as_float(row, "std_contrast_preservation"),
                "mean_edge_similarity_boundary": as_float(row, "mean_edge_similarity_boundary"),
                "std_edge_similarity_boundary": as_float(row, "std_edge_similarity_boundary"),
            }
        )
    return order_rows(table)


def write_csv_table(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def metric_with_std(row: Dict[str, object], mean_key: str, std_key: str, digits: int) -> str:
    mean = float(row[mean_key])
    std = float(row[std_key])
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def write_sr_markdown(path: Path, rows: List[Dict[str, object]]) -> None:
    markdown = (
        "| Method | PSNR (dB) | SSIM | MAE | Edge similarity |\n"
        "|---|---:|---:|---:|---:|\n"
    )
    for row in rows:
        markdown += (
            f"| {row['method']} | "
            f"{metric_with_std(row, 'mean_psnr', 'std_psnr', 3)} | "
            f"{metric_with_std(row, 'mean_ssim', 'std_ssim', 4)} | "
            f"{metric_with_std(row, 'mean_mae', 'std_mae', 5)} | "
            f"{metric_with_std(row, 'mean_edge_similarity', 'std_edge_similarity', 4)} |\n"
        )
    path.write_text(markdown, encoding="utf-8")


def write_anatomy_markdown(path: Path, rows: List[Dict[str, object]]) -> None:
    markdown = (
        "| Method | MAE inside mask | MAE outside mask | Boundary error | "
        "Contrast preservation | Boundary edge similarity |\n"
        "|---|---:|---:|---:|---:|---:|\n"
    )
    for row in rows:
        markdown += (
            f"| {row['method']} | "
            f"{metric_with_std(row, 'mean_mae_inside_mask', 'std_mae_inside_mask', 5)} | "
            f"{metric_with_std(row, 'mean_mae_outside_mask', 'std_mae_outside_mask', 5)} | "
            f"{metric_with_std(row, 'mean_boundary_error', 'std_boundary_error', 5)} | "
            f"{metric_with_std(row, 'mean_contrast_preservation', 'std_contrast_preservation', 4)} | "
            f"{metric_with_std(row, 'mean_edge_similarity_boundary', 'std_edge_similarity_boundary', 4)} |\n"
        )
    path.write_text(markdown, encoding="utf-8")


def short_method_labels() -> List[str]:
    return [
        "LQ mixed\nbaseline",
        "SR-only",
        "Quality-aware\nSR",
        "Quality-aware SR\n+ L1/MSE",
    ]


def apply_academic_style(axis, ylabel: str, title: str, lower_is_better: bool) -> None:
    axis.set_ylabel(ylabel)
    axis.set_title(title, pad=12)
    axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    note = "Lower is better" if lower_is_better else "Higher is better"
    axis.text(
        0.99,
        0.96,
        note,
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#444444",
    )


def save_bar_figure(
    rows: List[Dict[str, object]],
    mean_key: str,
    std_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
    lower_is_better: bool,
) -> None:
    labels = short_method_labels()
    means = [float(row[mean_key]) for row in rows]
    stds = [float(row[std_key]) for row in rows]
    x_values = list(range(len(rows)))

    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    bars = axis.bar(
        x_values,
        means,
        yerr=stds,
        capsize=4,
        color=BAR_COLORS,
        edgecolor="#222222",
        linewidth=0.7,
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
    )

    axis.set_xticks(x_values)
    axis.set_xticklabels(labels)
    apply_academic_style(axis, ylabel, title, lower_is_better)

    value_range = max(means) - min(means)
    offset = max(value_range * 0.03, max(means) * 0.005, 0.001)
    for bar, mean in zip(bars, means):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{mean:.3f}" if mean >= 1.0 else f"{mean:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    ensure_inputs_exist([SR_SUMMARY_CSV, ANATOMY_SUMMARY_CSV, SR_SOURCE_MD, ANATOMY_SOURCE_MD])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sr_rows = build_sr_table(read_csv_rows(SR_SUMMARY_CSV))
    anatomy_rows = build_anatomy_table(read_csv_rows(ANATOMY_SUMMARY_CSV))

    write_csv_table(
        FINAL_SR_TABLE_CSV,
        sr_rows,
        [
            "method",
            "mean_psnr",
            "std_psnr",
            "mean_ssim",
            "std_ssim",
            "mean_mae",
            "std_mae",
            "mean_edge_similarity",
            "std_edge_similarity",
        ],
    )
    write_csv_table(
        ANATOMY_TABLE_CSV,
        anatomy_rows,
        [
            "method",
            "mean_mae_inside_mask",
            "std_mae_inside_mask",
            "mean_mae_outside_mask",
            "std_mae_outside_mask",
            "mean_boundary_error",
            "std_boundary_error",
            "mean_contrast_preservation",
            "std_contrast_preservation",
            "mean_edge_similarity_boundary",
            "std_edge_similarity_boundary",
        ],
    )

    write_sr_markdown(FINAL_SR_TABLE_MD, sr_rows)
    write_anatomy_markdown(ANATOMY_TABLE_MD, anatomy_rows)

    save_bar_figure(
        sr_rows,
        "mean_psnr",
        "std_psnr",
        "PSNR (dB)",
        "HC18 Test-Set PSNR",
        FIGURE_PSNR,
        lower_is_better=False,
    )
    save_bar_figure(
        sr_rows,
        "mean_ssim",
        "std_ssim",
        "SSIM",
        "HC18 Test-Set SSIM",
        FIGURE_SSIM,
        lower_is_better=False,
    )
    save_bar_figure(
        sr_rows,
        "mean_mae",
        "std_mae",
        "MAE",
        "HC18 Test-Set MAE",
        FIGURE_MAE,
        lower_is_better=True,
    )
    save_bar_figure(
        anatomy_rows,
        "mean_mae_inside_mask",
        "std_mae_inside_mask",
        "MAE inside fetal head mask",
        "Anatomy Preservation: Inside-Mask Error",
        FIGURE_ANATOMY_MAE_INSIDE,
        lower_is_better=True,
    )
    save_bar_figure(
        anatomy_rows,
        "mean_boundary_error",
        "std_boundary_error",
        "Boundary error",
        "Anatomy Preservation: Fetal Head Boundary Error",
        FIGURE_BOUNDARY_ERROR,
        lower_is_better=True,
    )

    saved_paths = [
        FINAL_SR_TABLE_CSV,
        FINAL_SR_TABLE_MD,
        ANATOMY_TABLE_CSV,
        ANATOMY_TABLE_MD,
        FIGURE_PSNR,
        FIGURE_SSIM,
        FIGURE_MAE,
        FIGURE_ANATOMY_MAE_INSIDE,
        FIGURE_BOUNDARY_ERROR,
    ]

    print("Saved report assets:")
    for path in saved_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
