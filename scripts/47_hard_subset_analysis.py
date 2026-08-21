"""Hard-subset downstream utility analysis for HC18 SR methods."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOWNSTREAM_PER_IMAGE = PROJECT_ROOT / "results" / "downstream_utility_per_image.csv"
RECON_PER_IMAGE = PROJECT_ROOT / "results" / "final_all_hc18_comparison_per_image.csv"
SUMMARY_CSV = PROJECT_ROOT / "results" / "downstream_hard_subset_summary.csv"
MEMBERSHIP_CSV = PROJECT_ROOT / "results" / "downstream_hard_subset_membership.csv"
FIGURE_PATH = PROJECT_ROOT / "results" / "downstream_hard_subset_figures" / "dice_hard_vs_full.png"
SWINIR_LABEL = "SwinIR-Light Phase A"
SWINIR_EVAL_DIR = PROJECT_ROOT / "results" / "eval" / "swinir_light_phase_a" / "test"
RESSHIFT_LABEL = "ResShift"
RESSHIFT_EVAL_DIR = PROJECT_ROOT / "results" / "eval" / "resshift" / "test"
OSEDIFF_LABEL = "OSEDiff"
OSEDIFF_EVAL_DIR = PROJECT_ROOT / "results" / "eval" / "osediff" / "test"

METHOD_ORDER = [
    "LQ_mixed",
    "SR-only",
    "Quality-aware",
    "SR-only+perceptual",
    "Quality-aware+perceptual",
    "CNN-Residual Diffusion",
    "SwinIR-Light Phase A",
    "ResShift",
    "OSEDiff",
]

RECON_METHOD_MAP = {
    "LQ_mixed": "LQ_mixed baseline",
    "SR-only": "SR-only",
    "Quality-aware": "Quality-aware SR",
    "SR-only+perceptual": "SR-only + LPIPS + GAN",
    "Quality-aware+perceptual": "Quality-aware SR + LPIPS + GAN",
    SWINIR_LABEL: SWINIR_LABEL,
    RESSHIFT_LABEL: RESSHIFT_LABEL,
    OSEDIFF_LABEL: OSEDIFF_LABEL,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute downstream metrics on hard HC18 subsets.")
    parser.add_argument("--downstream_per_image", default=str(DOWNSTREAM_PER_IMAGE))
    parser.add_argument("--recon_per_image", default=str(RECON_PER_IMAGE))
    parser.add_argument("--output_csv", default=str(SUMMARY_CSV))
    parser.add_argument("--membership_csv", default=str(MEMBERSHIP_CSV))
    parser.add_argument("--figure", default=str(FIGURE_PATH))
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: Iterable[str], path: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    downstream_path = Path(args.downstream_per_image)
    recon_path = Path(args.recon_per_image)
    downstream = pd.read_csv(downstream_path)
    recon = pd.read_csv(recon_path)
    require_columns(
        downstream,
        ["image_id", "method", "Dice", "IoU", "AreaErr", "DiamErr"],
        downstream_path,
    )
    require_columns(recon, ["image_id", "method", "LPIPS"], recon_path)
    return downstream, recon


def build_membership(downstream: pd.DataFrame, recon: pd.DataFrame) -> pd.DataFrame:
    lq_downstream = downstream[downstream["method"] == "LQ_mixed"].copy()
    if lq_downstream.empty:
        raise ValueError("No LQ_mixed rows found in downstream per-image CSV")
    lq_lpips = recon[recon["method"] == "LQ_mixed baseline"][["image_id", "LPIPS"]].copy()
    if lq_lpips.empty:
        raise ValueError("No LQ_mixed baseline rows found in reconstruction per-image CSV")

    lq_downstream["Dice"] = pd.to_numeric(lq_downstream["Dice"], errors="coerce")
    lq_lpips["LPIPS"] = pd.to_numeric(lq_lpips["LPIPS"], errors="coerce")
    merged = lq_downstream[["image_id", "Dice"]].merge(lq_lpips, on="image_id", how="inner")
    if merged.empty:
        raise ValueError("Could not align LQ downstream Dice with LQ reconstruction LPIPS")

    dice_q25 = float(merged["Dice"].quantile(0.25))
    lpips_q75 = float(merged["LPIPS"].quantile(0.75))
    rows = []
    for _, row in merged.iterrows():
        image_id = str(row["image_id"])
        dice = float(row["Dice"])
        lpips = float(row["LPIPS"])
        rows.append(
            {
                "image_id": image_id,
                "LQ_Dice": dice,
                "LQ_LPIPS": lpips,
                "bottom25_lq_dice": dice <= dice_q25,
                "lq_dice_lt_0_85": dice < 0.85,
                "top25_lq_lpips": lpips >= lpips_q75,
                "dice_q25_threshold": dice_q25,
                "lpips_q75_threshold": lpips_q75,
            }
        )
    return pd.DataFrame(rows)


def summarize_subset(
    downstream: pd.DataFrame,
    subset_name: str,
    image_ids: set[str],
) -> list[dict[str, float | str | int]]:
    subset_df = downstream[downstream["image_id"].astype(str).isin(image_ids)].copy()
    rows = []
    baseline_row = None
    method_summaries: dict[str, dict[str, float | str | int]] = {}
    for method in METHOD_ORDER:
        method_df = subset_df[subset_df["method"] == method].copy()
        if method_df.empty:
            continue
        for column in ["Dice", "IoU", "AreaErr", "DiamErr"]:
            method_df[column] = pd.to_numeric(method_df[column], errors="coerce")
        row: dict[str, float | str | int] = {
            "subset": subset_name,
            "method": method,
            "N": int(method_df["image_id"].nunique()),
            "Dice_mean": float(method_df["Dice"].mean()),
            "IoU_mean": float(method_df["IoU"].mean()),
            "AreaErr_mean": float(method_df["AreaErr"].mean()),
            "DiamErr_mean": float(method_df["DiamErr"].mean()),
        }
        method_summaries[method] = row
        if method == "LQ_mixed":
            baseline_row = row

    if baseline_row is None:
        raise ValueError(f"Subset {subset_name} has no LQ_mixed baseline rows")
    for method in METHOD_ORDER:
        row = method_summaries.get(method)
        if row is None:
            continue
        row["Dice_delta_vs_LQ"] = float(row["Dice_mean"]) - float(baseline_row["Dice_mean"])
        row["IoU_delta_vs_LQ"] = float(row["IoU_mean"]) - float(baseline_row["IoU_mean"])
        row["AreaErr_delta_vs_LQ"] = float(row["AreaErr_mean"]) - float(baseline_row["AreaErr_mean"])
        row["DiamErr_delta_vs_LQ"] = float(row["DiamErr_mean"]) - float(baseline_row["DiamErr_mean"])
        rows.append(row)
    return rows


def plot_dice(summary: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subsets = ["full_test", "bottom25_lq_dice", "lq_dice_lt_0_85", "top25_lq_lpips"]
    methods = [method for method in METHOD_ORDER if method in set(summary["method"])]
    x = np.arange(len(subsets), dtype=np.float64)
    width = 0.11 if len(methods) > 5 else 0.13

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    for index, method in enumerate(methods):
        values = []
        for subset in subsets:
            row = summary[(summary["subset"] == subset) & (summary["method"] == method)]
            values.append(float(row["Dice_mean"].iloc[0]) if not row.empty else np.nan)
        offset = (index - (len(methods) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(["Full", "Bottom 25% LQ Dice", "LQ Dice < 0.85", "Worst 25% LQ LPIPS"], rotation=12)
    ax.set_ylabel("Dice mean")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    downstream, recon = load_inputs(args)
    membership = build_membership(downstream, recon)
    membership_path = Path(args.membership_csv)
    membership_path.parent.mkdir(parents=True, exist_ok=True)
    membership.to_csv(membership_path, index=False)

    subset_defs = {
        "full_test": set(membership["image_id"].astype(str)),
        "bottom25_lq_dice": set(membership.loc[membership["bottom25_lq_dice"], "image_id"].astype(str)),
        "lq_dice_lt_0_85": set(membership.loc[membership["lq_dice_lt_0_85"], "image_id"].astype(str)),
        "top25_lq_lpips": set(membership.loc[membership["top25_lq_lpips"], "image_id"].astype(str)),
    }

    rows = []
    for subset_name, image_ids in subset_defs.items():
        rows.extend(summarize_subset(downstream, subset_name, image_ids))
    summary = pd.DataFrame(rows)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    swinir_summary = summary[summary["method"] == SWINIR_LABEL].copy()
    if not swinir_summary.empty:
        SWINIR_EVAL_DIR.mkdir(parents=True, exist_ok=True)
        swinir_summary.to_csv(SWINIR_EVAL_DIR / "hard_subsets.csv", index=False)
    resshift_summary = summary[summary["method"] == RESSHIFT_LABEL].copy()
    if not resshift_summary.empty:
        RESSHIFT_EVAL_DIR.mkdir(parents=True, exist_ok=True)
        resshift_summary.to_csv(RESSHIFT_EVAL_DIR / "hard_subsets.csv", index=False)
    osediff_summary = summary[summary["method"] == OSEDIFF_LABEL].copy()
    if not osediff_summary.empty:
        OSEDIFF_EVAL_DIR.mkdir(parents=True, exist_ok=True)
        osediff_summary.to_csv(OSEDIFF_EVAL_DIR / "hard_subsets.csv", index=False)
    plot_dice(summary, Path(args.figure))

    missing_methods = [method for method in METHOD_ORDER if method not in set(downstream["method"])]
    print("STEP 3a - HARD-SUBSET DOWNSTREAM ANALYSIS")
    print(f"Membership CSV: {membership_path.relative_to(PROJECT_ROOT)}")
    print(f"Summary CSV: {output_csv.relative_to(PROJECT_ROOT)}")
    print(f"Figure: {Path(args.figure).relative_to(PROJECT_ROOT)}")
    if not swinir_summary.empty:
        print(f"SwinIR hard subsets CSV: {(SWINIR_EVAL_DIR / 'hard_subsets.csv').relative_to(PROJECT_ROOT)}")
    if not resshift_summary.empty:
        print(f"ResShift hard subsets CSV: {(RESSHIFT_EVAL_DIR / 'hard_subsets.csv').relative_to(PROJECT_ROOT)}")
    if not osediff_summary.empty:
        print(f"OSEDiff hard subsets CSV: {(OSEDIFF_EVAL_DIR / 'hard_subsets.csv').relative_to(PROJECT_ROOT)}")
    print(f"Subset sizes: { {name: len(ids) for name, ids in subset_defs.items()} }")
    if missing_methods:
        print(f"Methods requested but unavailable in downstream per-image CSV: {missing_methods}")
        print("SwinIR-Light Phase A is not included because no full-test downstream per-image evaluation exists yet.")
    print()
    print("Hard-subset summary")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    print("Subset membership thresholds")
    threshold_cols = ["dice_q25_threshold", "lpips_q75_threshold"]
    print(membership[threshold_cols].head(1).to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
