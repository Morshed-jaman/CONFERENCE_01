r"""Archive unreliable Dice/IoU and first U-Net segmentation artifacts.

Run from PowerShell:

    python .\scripts\40_cleanup_unreliable_dice_results.py

Use --dry-run to preview the paths that would be moved.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
ARCHIVE_ROOT = RESULTS_ROOT / "archived_unreliable_dice_iou"

README_TEXT = (
    "These files were archived because threshold-based Dice/IoU and the first "
    "U-Net segmentation attempt produced unreliable anatomy metrics. They are "
    "not used in the final paper. The final paper uses mask-based anatomy "
    "preservation metrics and, if reliable, the improved U-Net segmentation "
    "results from script 39."
)

UNRELIABLE_PATHS = [
    PROJECT_ROOT / "scripts" / "37_add_dice_iou_fetal_measurement_metrics.py",
    PROJECT_ROOT / "scripts" / "38_train_hc18_unet_segmentation_eval.py",
    RESULTS_ROOT / "hc18_dice_iou_measurement_metrics.csv",
    RESULTS_ROOT / "hc18_dice_iou_measurement_summary.csv",
    RESULTS_ROOT / "hc18_dice_iou_measurement_table.md",
    RESULTS_ROOT / "final_review_ready_anatomy_metrics.md",
    RESULTS_ROOT / "hc18_dice_iou_measurement_figures",
    RESULTS_ROOT / "hc18_unet_segmentation_metrics_per_image.csv",
    RESULTS_ROOT / "hc18_unet_segmentation_metrics_summary.csv",
    RESULTS_ROOT / "hc18_unet_segmentation_metrics_table.md",
    RESULTS_ROOT / "models" / "hc18_unet_segmentation_best.pth",
]

OLD_UNET_FIGURE_FILES = [
    RESULTS_ROOT / "hc18_unet_segmentation_figures" / "example_segmentation_outputs.png",
    RESULTS_ROOT / "hc18_unet_segmentation_figures" / "dice_iou_comparison.png",
    RESULTS_ROOT / "hc18_unet_segmentation_figures" / "measurement_error_comparison.png",
    RESULTS_ROOT / "hc18_unet_segmentation_figures" / "debug_image_mask_pairing.png",
]

PROTECTED_PATHS = {
    PROJECT_ROOT / "scripts" / "39_improve_hc18_unet_segmentation.py",
    RESULTS_ROOT / "anatomy_preservation_all_models_summary.csv",
    RESULTS_ROOT / "anatomy_preservation_all_models_per_image.csv",
    RESULTS_ROOT / "anatomy_preservation_all_models_table.md",
    RESULTS_ROOT / "models" / "hc18_unet_segmentation_best_improved.pth",
    RESULTS_ROOT / "hc18_unet_improved_metrics_per_image.csv",
    RESULTS_ROOT / "hc18_unet_improved_metrics_summary.csv",
    RESULTS_ROOT / "hc18_unet_improved_metrics_table.md",
}


@dataclass
class CleanupSummary:
    moved_paths: int = 0
    moved_files: int = 0
    missing_paths: int = 0
    skipped_paths: int = 0
    failed_paths: int = 0


def resolve_path(path: Path) -> Path:
    return path.resolve(strict=False)


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def printable(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def archive_destination(path: Path) -> Path:
    return ARCHIVE_ROOT / path.relative_to(PROJECT_ROOT)


def protected_path(path: Path) -> bool:
    resolved = resolve_path(path)
    protected = {resolve_path(item) for item in PROTECTED_PATHS}
    if resolved in protected:
        return True

    old_unet_figures = resolve_path(RESULTS_ROOT / "hc18_unet_segmentation_figures")
    if path.name.startswith("improved_") and resolve_path(path.parent) == old_unet_figures:
        return True

    return False


def unique_destination(destination: Path) -> Path:
    if not destination.exists():
        return destination

    for index in range(1, 1000):
        candidate = destination.with_name(f"{destination.name}.archived_{index}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not find a free archive destination for {destination}")


def files_under(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(item for item in path.rglob("*") if item.is_file())
    return []


def print_planned_move(source: Path, destination: Path) -> None:
    if source.is_dir():
        files = files_under(source)
        if not files:
            print(f"Would move empty folder: {printable(source)} -> {printable(destination)}")
            return

        for file_path in files:
            relative_file = file_path.relative_to(source)
            print(
                "Would move: "
                f"{printable(file_path)} -> {printable(destination / relative_file)}"
            )
        return

    print(f"Would move: {printable(source)} -> {printable(destination)}")


def move_path(source: Path, summary: CleanupSummary, dry_run: bool) -> None:
    if protected_path(source):
        summary.skipped_paths += 1
        print(f"Skipped protected path: {printable(source)}")
        return

    if not source.exists():
        summary.missing_paths += 1
        print(f"Missing, skipped: {printable(source)}")
        return

    try:
        destination = unique_destination(archive_destination(source))
        moved_files = files_under(source)

        if dry_run:
            print_planned_move(source, destination)
            return

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        summary.moved_paths += 1
        summary.moved_files += len(moved_files)

        if source.is_dir():
            for file_path in moved_files:
                relative_file = file_path.relative_to(source)
                print(
                    "Moved: "
                    f"{printable(file_path)} -> {printable(destination / relative_file)}"
                )
        else:
            print(f"Moved: {printable(source)} -> {printable(destination)}")

        if source.is_dir() and not moved_files:
            print(f"Moved empty folder: {printable(source)} -> {printable(destination)}")

    except OSError as exc:
        summary.failed_paths += 1
        print(f"Failed: {printable(source)} ({exc})")
    except RuntimeError as exc:
        summary.failed_paths += 1
        print(f"Failed: {printable(source)} ({exc})")


def write_readme(dry_run: bool) -> Path:
    readme_path = ARCHIVE_ROOT / "README.md"
    if dry_run:
        print(f"Would write: {printable(readme_path)}")
        return readme_path

    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(README_TEXT + "\n", encoding="utf-8")
    print(f"Wrote: {printable(readme_path)}")
    return readme_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Archive unreliable threshold-based Dice/IoU artifacts and first "
            "U-Net segmentation outputs."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print planned moves without changing files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = CleanupSummary()

    print("Archiving unreliable Dice/IoU and first U-Net segmentation artifacts")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Archive folder: {ARCHIVE_ROOT}")
    if args.dry_run:
        print("Dry run only. No files will be moved.")
    print()

    for source in UNRELIABLE_PATHS + OLD_UNET_FIGURE_FILES:
        move_path(source, summary, args.dry_run)

    readme_path = write_readme(args.dry_run)

    print()
    print("Cleanup summary:")
    print(f"  Moved source paths: {summary.moved_paths}")
    print(f"  Moved files: {summary.moved_files}")
    print(f"  Missing source paths: {summary.missing_paths}")
    print(f"  Skipped protected paths: {summary.skipped_paths}")
    print(f"  Failed source paths: {summary.failed_paths}")
    print(f"  Archive folder: {printable(ARCHIVE_ROOT)}")
    print(f"  README: {printable(readme_path)}")

    return 1 if summary.failed_paths else 0


if __name__ == "__main__":
    raise SystemExit(main())
