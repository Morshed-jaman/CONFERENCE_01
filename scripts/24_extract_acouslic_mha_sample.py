"""
Extract a small sample of ACOUSLIC-AI .mha ultrasound stacks and convert slices
to PNG frames.

Important:
This script is intentionally a sampler. It never extracts the full zip file.
The generated frames are intended for proxy sweep-quality feedback and quick
pipeline inspection only; they are not clinical labels or diagnostic targets.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np


DEFAULT_ZIP_CANDIDATES = [
    Path("D:/acouslic-ai-train-set.zip"),
    Path("D:/acouslic-ai-train-set"),
]

OUTPUT_ROOT = Path("data/processed/ACOUSLIC_AI")
TEMP_MHA_DIR = OUTPUT_ROOT / "temp_mha"
FRAMES_DIR = OUTPUT_ROOT / "frames"
MANIFEST_PATH = OUTPUT_ROOT / "acouslic_frame_manifest.csv"

STACKED_ULTRASOUND_HINT = "images/stacked_fetal_ultrasound/"

MANIFEST_COLUMNS = [
    "sweep_id",
    "frame_id",
    "frame_index",
    "source_mha_inside_zip",
    "extracted_mha_path",
    "frame_path",
    "width",
    "height",
    "num_slices",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract only a small sample of ACOUSLIC-AI .mha stacks from a zip "
            "and convert selected slices to grayscale PNG frames."
        )
    )
    parser.add_argument(
        "--zip_path",
        type=Path,
        default=None,
        help=(
            "Optional input zip path. If omitted, the script tries "
            "D:/acouslic-ai-train-set.zip, then D:/acouslic-ai-train-set."
        ),
    )
    parser.add_argument(
        "--max_mha",
        type=int,
        default=5,
        help="Maximum number of .mha files to extract from the zip.",
    )
    parser.add_argument(
        "--slice_interval",
        type=int,
        default=5,
        help="For 3D stacks, save one slice every N slices.",
    )
    parser.add_argument(
        "--max_slices_per_mha",
        type=int,
        default=100,
        help="Maximum number of PNG frames to save from each .mha file.",
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


def validate_args(args: argparse.Namespace) -> None:
    if args.max_mha <= 0:
        raise ValueError("--max_mha must be greater than 0.")
    if args.slice_interval <= 0:
        raise ValueError("--slice_interval must be greater than 0.")
    if args.max_slices_per_mha <= 0:
        raise ValueError("--max_slices_per_mha must be greater than 0.")


def safe_name(name: str) -> str:
    """Make a zip filename safe for a flat Windows-compatible output folder."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return cleaned.strip("._") or "unnamed"


def rel_path(path: Path) -> str:
    """Return a forward-slash relative path for CSV portability."""
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def list_mha_infos(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """List .mha files, preferring the expected stacked fetal ultrasound folder."""
    all_mha_infos = [
        info
        for info in zf.infolist()
        if not info.is_dir() and info.filename.lower().replace("\\", "/").endswith(".mha")
    ]

    stacked_infos = [
        info
        for info in all_mha_infos
        if STACKED_ULTRASOUND_HINT in info.filename.lower().replace("\\", "/")
    ]

    if stacked_infos:
        return sorted(stacked_infos, key=lambda item: item.filename)

    print(
        "WARNING: No .mha files were found under "
        f"'{STACKED_ULTRASOUND_HINT}'. Falling back to all .mha files in the zip."
    )
    return sorted(all_mha_infos, key=lambda item: item.filename)


def extract_one_mha(
    zf: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    temp_mha_dir: Path,
    selected_index: int,
) -> Path:
    """
    Extract exactly one selected .mha file into temp_mha.

    The zip member's internal directory structure is not recreated; using a flat
    destination avoids path traversal surprises and keeps the sample folder easy
    to inspect.
    """
    source_name = Path(info.filename).name
    flat_name = f"{selected_index:04d}_{safe_name(source_name)}"
    output_path = temp_mha_dir / flat_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zf.open(info, mode="r") as source, output_path.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)

    return output_path


def normalize_slice_to_uint8(slice_array: np.ndarray) -> np.ndarray:
    """Min-max normalize one ultrasound slice to uint8 [0, 255]."""
    slice_float = np.asarray(slice_array, dtype=np.float32)
    finite_mask = np.isfinite(slice_float)

    if not finite_mask.any():
        return np.zeros(slice_float.shape, dtype=np.uint8)

    valid_values = slice_float[finite_mask]
    min_value = float(valid_values.min())
    max_value = float(valid_values.max())

    if max_value <= min_value:
        return np.zeros(slice_float.shape, dtype=np.uint8)

    normalized = (slice_float - min_value) / (max_value - min_value)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    normalized = np.clip(normalized * 255.0, 0.0, 255.0)
    return normalized.astype(np.uint8)


def prepare_image_array(mha_path: Path, sitk_module) -> np.ndarray:
    """
    Read an .mha file and return a squeezed NumPy array.

    SimpleITK returns 3D images as [num_slices, height, width] and 2D images as
    [height, width]. Extra singleton dimensions are removed.
    """
    image = sitk_module.ReadImage(str(mha_path))
    array = sitk_module.GetArrayFromImage(image)
    array = np.asarray(array)
    array = np.squeeze(array)

    if array.ndim not in (2, 3):
        raise ValueError(
            f"Unsupported array shape after squeezing singleton dimensions: {array.shape}"
        )

    return array


def save_png_with_simpleitk(frame_array_uint8: np.ndarray, frame_path: Path, sitk_module) -> None:
    """Save a 2D uint8 array as a grayscale PNG using SimpleITK."""
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk_module.GetImageFromArray(frame_array_uint8)
    sitk_module.WriteImage(image, str(frame_path))


def convert_mha_to_frames(
    mha_path: Path,
    source_mha_inside_zip: str,
    sweep_id: str,
    slice_interval: int,
    max_slices_per_mha: int,
    sitk_module,
) -> list[dict[str, object]]:
    """
    Convert selected slices from one .mha file into PNG frames.

    These frames are proxy sweep-quality feedback artifacts only; they should
    not be interpreted as clinical annotations.
    """
    array = prepare_image_array(mha_path, sitk_module)
    rows: list[dict[str, object]] = []

    if array.ndim == 2:
        slice_plan = [(0, array)]
        num_slices = 1
    else:
        num_slices = int(array.shape[0])
        selected_indices = range(0, num_slices, slice_interval)
        slice_plan = [
            (slice_index, array[slice_index])
            for slice_index in selected_indices
        ][:max_slices_per_mha]

    safe_sweep_id = safe_name(sweep_id)
    sweep_frame_dir = FRAMES_DIR / safe_sweep_id

    for saved_index, (frame_index, slice_array) in enumerate(slice_plan, start=1):
        if slice_array.ndim != 2:
            raise ValueError(
                f"Expected a 2D slice for {source_mha_inside_zip}, got {slice_array.shape}"
            )

        frame_uint8 = normalize_slice_to_uint8(slice_array)
        height, width = int(frame_uint8.shape[0]), int(frame_uint8.shape[1])
        frame_id = f"frame_{saved_index:06d}"
        frame_path = sweep_frame_dir / f"{frame_id}.png"
        save_png_with_simpleitk(frame_uint8, frame_path, sitk_module)

        rows.append(
            {
                "sweep_id": sweep_id,
                "frame_id": frame_id,
                "frame_index": int(frame_index),
                "source_mha_inside_zip": source_mha_inside_zip,
                "extracted_mha_path": rel_path(mha_path),
                "frame_path": rel_path(frame_path),
                "width": width,
                "height": height,
                "num_slices": num_slices,
            }
        )

    return rows


def write_manifest(rows: list[dict[str, object]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    validate_args(args)

    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError(
            "SimpleITK is required to read .mha files. Install it in the active "
            "environment before running this script."
        ) from exc

    zip_path = resolve_zip_path(args.zip_path)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TEMP_MHA_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    extracted_count = 0
    saved_frame_count = 0

    with zipfile.ZipFile(zip_path, mode="r") as zf:
        mha_infos = list_mha_infos(zf)
        selected_infos = mha_infos[: args.max_mha]

        print(f"Input zip: {zip_path}")
        print(f"Number of .mha files found in zip: {len(mha_infos)}")

        for selected_index, info in enumerate(selected_infos, start=1):
            sweep_id = Path(info.filename).stem
            extracted_path = extract_one_mha(
                zf=zf,
                info=info,
                temp_mha_dir=TEMP_MHA_DIR,
                selected_index=selected_index,
            )
            extracted_count += 1

            try:
                frame_rows = convert_mha_to_frames(
                    mha_path=extracted_path,
                    source_mha_inside_zip=info.filename,
                    sweep_id=sweep_id,
                    slice_interval=args.slice_interval,
                    max_slices_per_mha=args.max_slices_per_mha,
                    sitk_module=sitk,
                )
            except Exception as exc:
                print(
                    "WARNING: Failed to convert "
                    f"{info.filename} after extraction to {extracted_path}: {exc}"
                )
                continue

            manifest_rows.extend(frame_rows)
            saved_frame_count += len(frame_rows)

    write_manifest(manifest_rows, MANIFEST_PATH)

    print(f"Number of .mha files extracted: {extracted_count}")
    print(f"Number of PNG frames saved: {saved_frame_count}")
    print(f"Manifest path: {MANIFEST_PATH}")
    print("Done. Only the selected .mha sample was extracted; the full zip was not extracted.")


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except (FileNotFoundError, ValueError, zipfile.BadZipFile, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
