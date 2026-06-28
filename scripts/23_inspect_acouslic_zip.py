"""
Inspect the ACOUSLIC-AI train-set zip without extracting it.

This script only reads the ZIP central directory metadata with Python's
standard zipfile module. It does not extract images, videos, arrays, or any
other payload files.
"""

from __future__ import annotations

import argparse
import csv
import sys
import zipfile
from collections import Counter
from pathlib import Path


DEFAULT_ZIP_CANDIDATES = [
    Path("D:/acouslic-ai-train-set.zip"),
    Path("D:/acouslic-ai-train-set"),
]

OUTPUT_DIR = Path("results/acouslic_ai")
CSV_OUTPUT_PATH = OUTPUT_DIR / "acouslic_zip_inspection.csv"
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "acouslic_zip_summary.md"

COMMON_EXTENSIONS = [
    ".png",
    ".jpg",
    ".jpeg",
    ".mp4",
    ".avi",
    ".npy",
    ".nii",
    ".nii.gz",
    ".json",
    ".csv",
    ".txt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the ACOUSLIC-AI train-set zip metadata without extraction."
    )
    parser.add_argument(
        "--zip_path",
        type=Path,
        default=None,
        help=(
            "Optional zip path. If omitted, the script tries "
            "D:/acouslic-ai-train-set.zip, then D:/acouslic-ai-train-set."
        ),
    )
    parser.add_argument(
        "--csv_output",
        type=Path,
        default=CSV_OUTPUT_PATH,
        help="Path for the per-file metadata CSV.",
    )
    parser.add_argument(
        "--summary_output",
        type=Path,
        default=SUMMARY_OUTPUT_PATH,
        help="Path for the markdown summary.",
    )
    return parser.parse_args()


def resolve_zip_path(user_path: Path | None) -> Path:
    candidates = [user_path] if user_path is not None else DEFAULT_ZIP_CANDIDATES

    checked_paths: list[Path] = []
    for candidate in candidates:
        if candidate is None:
            continue

        checked_paths.append(candidate)
        if candidate.exists() and candidate.is_file():
            if not zipfile.is_zipfile(candidate):
                raise ValueError(f"Found path but it is not a valid zip file: {candidate}")
            return candidate

    checked_text = "\n".join(f"  - {path}" for path in checked_paths)
    raise FileNotFoundError(
        "Could not find the ACOUSLIC-AI train-set zip. Checked:\n" + checked_text
    )


def zip_extension(zip_name: str) -> str:
    """Return a normalized extension for a path stored inside a zip."""
    lower_name = zip_name.lower()
    if lower_name.endswith(".nii.gz"):
        return ".nii.gz"
    return Path(lower_name).suffix


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def write_summary(
    summary_path: Path,
    zip_path: Path,
    total_files: int,
    total_dirs: int,
    extension_counts: Counter[str],
    first_100_files: list[str],
    total_compressed_size: int,
    total_uncompressed_size: int,
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    common_rows = []
    for extension in COMMON_EXTENSIONS:
        common_rows.append(f"| `{extension}` | {extension_counts.get(extension, 0)} |")

    top_rows = []
    for extension, count in extension_counts.most_common(20):
        shown_extension = extension if extension else "(no extension)"
        top_rows.append(f"| `{shown_extension}` | {count} |")

    first_paths = "\n".join(f"{idx}. `{name}`" for idx, name in enumerate(first_100_files, 1))
    if not first_paths:
        first_paths = "_No file entries found._"

    compression_ratio = ""
    if total_uncompressed_size > 0:
        ratio = total_compressed_size / total_uncompressed_size
        compression_ratio = f"\n- Compression ratio: {ratio:.4f}"

    summary_text = f"""# ACOUSLIC-AI Zip Inspection Summary

This inspection reads zip metadata only. It does not extract the dataset.

- Zip path: `{zip_path}`
- Total files: {total_files}
- Total directories: {total_dirs}
- Total compressed size: {format_bytes(total_compressed_size)}
- Total uncompressed size: {format_bytes(total_uncompressed_size)}{compression_ratio}

## Common extension counts

| Extension | Count |
|---|---:|
{chr(10).join(common_rows)}

## Top extensions

| Extension | Count |
|---|---:|
{chr(10).join(top_rows)}

## First 100 file paths

{first_paths}
"""

    summary_path.write_text(summary_text, encoding="utf-8")


def inspect_zip(zip_path: Path, csv_output_path: Path, summary_output_path: Path) -> None:
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)

    total_files = 0
    total_dirs = 0
    total_compressed_size = 0
    total_uncompressed_size = 0
    extension_counts: Counter[str] = Counter()
    first_100_files: list[str] = []

    with zipfile.ZipFile(zip_path, mode="r") as zf, csv_output_path.open(
        mode="w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "file_name",
                "extension",
                "compressed_size",
                "uncompressed_size",
            ],
        )
        writer.writeheader()

        for info in zf.infolist():
            if info.is_dir():
                total_dirs += 1
                continue

            extension = zip_extension(info.filename)
            total_files += 1
            total_compressed_size += int(info.compress_size)
            total_uncompressed_size += int(info.file_size)
            extension_counts[extension] += 1

            if len(first_100_files) < 100:
                first_100_files.append(info.filename)

            writer.writerow(
                {
                    "file_name": info.filename,
                    "extension": extension,
                    "compressed_size": int(info.compress_size),
                    "uncompressed_size": int(info.file_size),
                }
            )

    write_summary(
        summary_path=summary_output_path,
        zip_path=zip_path,
        total_files=total_files,
        total_dirs=total_dirs,
        extension_counts=extension_counts,
        first_100_files=first_100_files,
        total_compressed_size=total_compressed_size,
        total_uncompressed_size=total_uncompressed_size,
    )

    print(f"Zip path: {zip_path}")
    print(f"Total files: {total_files}")
    print(f"Total directories: {total_dirs}")
    print("Common extension counts:")
    for extension in COMMON_EXTENSIONS:
        print(f"  {extension}: {extension_counts.get(extension, 0)}")

    print("\nFirst 100 file paths inside the zip:")
    for idx, file_name in enumerate(first_100_files, 1):
        print(f"{idx:03d}. {file_name}")

    print(f"\nSaved CSV: {csv_output_path}")
    print(f"Saved summary: {summary_output_path}")
    print("Inspection complete. No files were extracted.")


def main() -> int:
    args = parse_args()

    try:
        zip_path = resolve_zip_path(args.zip_path)
        inspect_zip(
            zip_path=zip_path,
            csv_output_path=args.csv_output,
            summary_output_path=args.summary_output,
        )
    except (FileNotFoundError, ValueError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
