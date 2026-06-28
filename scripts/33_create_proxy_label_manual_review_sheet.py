"""
Create a manual visual sanity-check sheet for ACOUSLIC-AI proxy labels.

This is a visual review support tool. The blank columns are intended for human
review notes and should not be treated as expert annotation by default.
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

DEFAULT_MANIFEST = Path("data/processed/ACOUSLIC_AI/acouslic_proxy_quality_manifest.csv")
OUTPUT_CSV = Path("results/acouslic_ai/manual_review/proxy_label_manual_review.csv")
CONTACT_SHEET_DIR = Path("results/acouslic_ai/manual_review/contact_sheets")

LABELS = [
    "good_frame",
    "blurry_frame",
    "low_contrast_frame",
    "noisy_frame",
    "poor_frame",
]

OUTPUT_COLUMNS = [
    "sample_id",
    "sweep_id",
    "frame_id",
    "frame_path",
    "proxy_label",
    "manual_label_blank",
    "reviewer_comment_blank",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ACOUSLIC-AI proxy-label manual review sheet and contact sheets."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Input ACOUSLIC-AI proxy-quality manifest CSV.",
    )
    parser.add_argument(
        "--samples_per_label",
        type=int,
        default=25,
        help="Number of frames to sample per proxy label.",
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


def validate_manifest(df: pd.DataFrame, path: Path) -> None:
    required = ["sweep_id", "frame_id", "frame_path", "frame_quality_label"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    available_labels = set(df["frame_quality_label"].astype(str))
    missing_labels = [label for label in LABELS if label not in available_labels]
    if missing_labels:
        raise ValueError(f"Missing required proxy labels in {path}: {missing_labels}")


def read_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={
            "sweep_id": str,
            "frame_id": str,
            "frame_path": str,
            "frame_quality_label": str,
        },
    )
    validate_manifest(df, path)
    return df


def sample_review_rows(
    manifest_df: pd.DataFrame,
    samples_per_label: int,
    seed: int,
) -> pd.DataFrame:
    if samples_per_label <= 0:
        raise ValueError("--samples_per_label must be greater than 0.")

    sampled_groups: list[pd.DataFrame] = []
    for label_index, label in enumerate(LABELS):
        label_df = manifest_df[manifest_df["frame_quality_label"].astype(str) == label]
        if len(label_df) < samples_per_label:
            raise ValueError(
                f"Label {label} has only {len(label_df)} frames; "
                f"need {samples_per_label}."
            )

        sampled = label_df.sample(
            n=samples_per_label,
            random_state=seed + label_index,
        ).sort_values(["sweep_id", "frame_id"])
        sampled_groups.append(sampled)

    sampled_df = pd.concat(sampled_groups, ignore_index=True)

    rows: list[dict[str, object]] = []
    for sample_number, row in enumerate(sampled_df.itertuples(index=False), start=1):
        label = str(getattr(row, "frame_quality_label"))
        sample_id = f"{label}_{sample_number:03d}"
        frame_path = resolve_project_path(getattr(row, "frame_path"))
        rows.append(
            {
                "sample_id": sample_id,
                "sweep_id": getattr(row, "sweep_id"),
                "frame_id": getattr(row, "frame_id"),
                "frame_path": rel_path(frame_path),
                "proxy_label": label,
                "manual_label_blank": "",
                "reviewer_comment_blank": "",
            }
        )

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def read_grayscale_for_plot(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0
    return np.clip(image, 0.0, 1.0)


def create_contact_sheet(
    label_df: pd.DataFrame,
    label: str,
    output_dir: Path,
    dpi: int = 300,
) -> Path:
    cols = 5
    rows = int(np.ceil(len(label_df) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.9))
    axes_array = np.atleast_1d(axes).reshape(rows, cols)

    for ax in axes_array.ravel():
        ax.axis("off")

    for ax, row in zip(axes_array.ravel(), label_df.itertuples(index=False), strict=False):
        frame_path = resolve_project_path(getattr(row, "frame_path"))
        image = read_grayscale_for_plot(frame_path)
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        ax.axis("off")
        overlay_text = f"{getattr(row, 'sample_id')}\n{getattr(row, 'proxy_label')}"
        ax.text(
            0.02,
            0.98,
            overlay_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="yellow",
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 2, "edgecolor": "none"},
        )

    fig.suptitle(
        f"Proxy-label manual review contact sheet: {label}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{label}_contact_sheet.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_contact_sheets(review_df: pd.DataFrame) -> list[Path]:
    output_dir = resolve_project_path(CONTACT_SHEET_DIR)
    saved_paths: list[Path] = []
    for label in LABELS:
        label_df = review_df[review_df["proxy_label"] == label].copy()
        saved_paths.append(create_contact_sheet(label_df, label, output_dir, dpi=300))
    plt.close("all")
    return saved_paths


def run(args: argparse.Namespace) -> None:
    manifest_path = resolve_project_path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Proxy-quality manifest not found: {manifest_path}")

    manifest_df = read_manifest(manifest_path)
    review_df = sample_review_rows(
        manifest_df=manifest_df,
        samples_per_label=args.samples_per_label,
        seed=args.seed,
    )

    output_csv = resolve_project_path(OUTPUT_CSV)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    review_df.to_csv(output_csv, index=False)

    saved_contact_sheets = create_contact_sheets(review_df)

    print(f"Manual review CSV: {OUTPUT_CSV}")
    print(f"Contact sheet directory: {CONTACT_SHEET_DIR}")
    print(f"Total sampled frames: {len(review_df)}")
    print("Contact sheets:")
    for path in saved_contact_sheets:
        print(f"  {rel_path(path)}")
    print("Note: this is a visual sanity-check support tool, not expert annotation.")


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except (FileNotFoundError, ValueError, pd.errors.EmptyDataError, cv2.error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
