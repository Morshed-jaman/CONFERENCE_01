r"""Safely find and optionally remove temporary project data.

Dry run is the default:

    python .\scripts\99_cleanup_unnecessary_data.py

Actual deletion requires both --delete and an interactive confirmation:

    python .\scripts\99_cleanup_unnecessary_data.py --delete
    python .\scripts\99_cleanup_unnecessary_data.py --delete --include_failed_experiments
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROTECTED_EXACT = {
    PROJECT_ROOT,
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "data" / "raw",
    PROJECT_ROOT / "results" / "models",
    PROJECT_ROOT / "results" / "training_logs",
}

PROTECTED_SUBTREES = {
    PROJECT_ROOT / "data" / "raw",
    PROJECT_ROOT / "results" / "models",
    PROJECT_ROOT / "results" / "training_logs",
}

PROTECTED_EXTERNAL = {
    Path("D:/acouslic-ai-train-set.zip"),
}

NEVER_DELETE_SUFFIXES = {
    ".csv",
    ".md",
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".rar",
    ".7z",
    ".pt",
    ".pth",
    ".pth.tar",
    ".ckpt",
    ".safetensors",
    ".onnx",
    ".h5",
    ".pkl",
    ".pickle",
}

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
}


@dataclass(frozen=True)
class Candidate:
    path: Path
    reason: str
    requires_extra_backup_confirmation: bool = False


def resolve_path(path: Path) -> Path:
    return path.resolve(strict=False)


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def format_size(num_bytes: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def path_size(path: Path) -> int:
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0

    total = 0
    if path.is_dir():
        for item in path.rglob("*"):
            if item.is_file():
                try:
                    total += item.stat().st_size
                except OSError:
                    continue
    return total


def printable(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def contains_never_delete_files(path: Path) -> bool:
    if path.is_file():
        lower_name = path.name.lower()
        return any(lower_name.endswith(suffix) for suffix in NEVER_DELETE_SUFFIXES)

    if not path.is_dir():
        return False

    for item in path.rglob("*"):
        lower_name = item.name.lower()
        if item.is_file() and any(
            lower_name.endswith(suffix) for suffix in NEVER_DELETE_SUFFIXES
        ):
            return True
    return False


def validate_candidate(candidate: Candidate) -> tuple[bool, str]:
    path = resolve_path(candidate.path)

    if not is_relative_to(path, PROJECT_ROOT):
        return False, "outside project root"

    if not path.exists():
        return False, "path does not exist"

    if path in PROTECTED_EXACT:
        return False, "protected exact path"

    for protected in PROTECTED_SUBTREES:
        if is_relative_to(path, resolve_path(protected)):
            return False, f"inside protected subtree: {printable(protected)}"

    if contains_never_delete_files(path):
        return False, "contains protected CSV/MD/ZIP/checkpoint-like files"

    return True, ""


def add_candidate(candidates: list[Candidate], path: Path, reason: str) -> None:
    candidates.append(Candidate(path=path, reason=reason))


def find_cache_candidates(candidates: list[Candidate]) -> None:
    for name in ("__pycache__", ".pytest_cache"):
        for path in PROJECT_ROOT.rglob(name):
            if path.is_dir():
                add_candidate(candidates, path, f"Python cache folder ({name})")


def find_acouslic_temp_candidates(candidates: list[Candidate]) -> None:
    temp_mha = PROJECT_ROOT / "data" / "processed" / "ACOUSLIC_AI" / "temp_mha"
    if temp_mha.is_dir():
        add_candidate(candidates, temp_mha, "temporary ACOUSLIC-AI extracted MHA files")


def has_large_acouslic_run() -> bool:
    acouslic_processed = PROJECT_ROOT / "data" / "processed" / "ACOUSLIC_AI"
    frames_dir = acouslic_processed / "frames"
    manifest = acouslic_processed / "acouslic_frame_manifest.csv"

    if frames_dir.is_dir():
        sweep_dirs = [item for item in frames_dir.iterdir() if item.is_dir()]
        if len(sweep_dirs) >= 50:
            return True

    if manifest.is_file():
        try:
            with manifest.open("r", encoding="utf-8", errors="ignore") as handle:
                row_count = sum(1 for _ in handle) - 1
            if row_count >= 50:
                return True
        except OSError:
            return False

    return False


def name_mentions_five_sweeps(name: str) -> bool:
    cleaned = name.lower().replace("-", "_").replace(" ", "_")
    tokens = [token for token in cleaned.split("_") if token]
    return (
        ("5" in tokens and any("sweep" in token for token in tokens))
        or "5sweep" in cleaned
        or "5_sweep" in cleaned
        or "sweep5" in cleaned
        or "sweep_5" in cleaned
    )


def name_mentions_large_sweeps(name: str) -> bool:
    cleaned = name.lower().replace("-", "_").replace(" ", "_")
    return (
        "50sweep" in cleaned
        or "50_sweep" in cleaned
        or "100sweep" in cleaned
        or "100_sweep" in cleaned
        or "sweep50" in cleaned
        or "sweep_50" in cleaned
        or "sweep100" in cleaned
        or "sweep_100" in cleaned
    )


def find_old_sample_acouslic_candidates(candidates: list[Candidate]) -> None:
    acouslic_roots = [
        PROJECT_ROOT / "data" / "processed" / "ACOUSLIC_AI",
        PROJECT_ROOT / "results" / "acouslic_ai",
    ]
    large_run_exists = has_large_acouslic_run()

    for root in acouslic_roots:
        if not root.is_dir():
            continue

        for path in root.rglob("*"):
            if not path.is_dir():
                continue

            lower_name = path.name.lower()
            parent_has_large_sibling = any(
                sibling.is_dir() and name_mentions_large_sweeps(sibling.name)
                for sibling in path.parent.iterdir()
            )

            if name_mentions_five_sweeps(path.name) and (
                large_run_exists or parent_has_large_sibling
            ):
                add_candidate(
                    candidates,
                    path,
                    "old 5-sweep ACOUSLIC sample output; larger run exists",
                )
                continue

            is_old_sample_frames = "sample" in lower_name and "frame" in lower_name
            is_current_frames_dir = path == root / "frames"
            if is_old_sample_frames and not is_current_frames_dir and large_run_exists:
                add_candidate(
                    candidates,
                    path,
                    "old sample extracted frames; larger ACOUSLIC run exists",
                )


def find_debug_image_candidates(candidates: list[Candidate]) -> None:
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        stem = path.stem.lower()
        tokens = {token for token in re.split(r"[^a-z0-9]+", stem) if token}
        parts = {part.lower() for part in path.parts}
        has_debug_marker = (
            bool(tokens & {"debug", "tmp", "temp"})
            or any(part in {"debug", "tmp", "temp"} for part in parts)
        )

        if has_debug_marker:
            add_candidate(candidates, path, "temporary/debug image file")


def find_failed_experiment_candidates(candidates: list[Candidate]) -> None:
    failed_paths = [
        PROJECT_ROOT / "results" / "sr_outputs" / "dmuisr_inspired_diffusion_sr",
        PROJECT_ROOT / "results" / "visual_dmuisr_inspired_diffusion_sr",
    ]
    for path in failed_paths:
        if path.exists():
            add_candidate(
                candidates,
                path,
                "optional failed DMUISR-inspired experiment output folder",
            )


def find_backup_candidates(candidates: list[Candidate]) -> None:
    backup_root = PROJECT_ROOT / "results" / "backups"
    if not backup_root.is_dir():
        return

    for path in backup_root.iterdir():
        if path.is_dir():
            candidates.append(
                Candidate(
                    path=path,
                    reason="optional old backup folder",
                    requires_extra_backup_confirmation=True,
                )
            )


def dedupe_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    valid_candidates: list[Candidate] = []
    skipped: list[tuple[Candidate, str]] = []

    for candidate in candidates:
        is_valid, reason = validate_candidate(candidate)
        if is_valid:
            valid_candidates.append(
                Candidate(
                    path=resolve_path(candidate.path),
                    reason=candidate.reason,
                    requires_extra_backup_confirmation=(
                        candidate.requires_extra_backup_confirmation
                    ),
                )
            )
        else:
            skipped.append((candidate, reason))

    valid_candidates.sort(key=lambda item: (len(item.path.parts), str(item.path).lower()))
    deduped: list[Candidate] = []

    for candidate in valid_candidates:
        if any(is_relative_to(candidate.path, existing.path) for existing in deduped):
            continue
        deduped.append(candidate)

    if skipped:
        print("Skipped protected/non-removable candidates:")
        for candidate, reason in skipped:
            print(f"  - {printable(candidate.path)} ({reason})")
        print()

    return deduped


def collect_candidates(args: argparse.Namespace) -> list[Candidate]:
    candidates: list[Candidate] = []

    find_acouslic_temp_candidates(candidates)
    find_old_sample_acouslic_candidates(candidates)
    find_cache_candidates(candidates)
    find_debug_image_candidates(candidates)

    if args.include_failed_experiments:
        find_failed_experiment_candidates(candidates)

    if args.include_backups:
        find_backup_candidates(candidates)

    return dedupe_candidates(candidates)


def print_protected_summary() -> None:
    print("Protected paths:")
    for path in sorted(PROTECTED_EXACT | PROTECTED_SUBTREES, key=lambda p: str(p).lower()):
        print(f"  - {printable(path)}")
    for path in sorted(PROTECTED_EXTERNAL, key=lambda p: str(p).lower()):
        print(f"  - {path}")
    print()


def print_candidates(candidates: list[Candidate]) -> int:
    if not candidates:
        print("No removable candidates found.")
        return 0

    total = 0
    print("Deletion candidates:")
    for index, candidate in enumerate(candidates, start=1):
        size = path_size(candidate.path)
        total += size
        backup_note = " [backup confirmation required]" if (
            candidate.requires_extra_backup_confirmation
        ) else ""
        print(
            f"{index:>3}. {format_size(size):>10}  {printable(candidate.path)}"
            f"{backup_note}"
        )
        print(f"      Reason: {candidate.reason}")
    print()
    print(f"Total removable size found: {format_size(total)}")
    return total


def confirm(prompt: str, expected: str) -> bool:
    response = input(prompt).strip()
    return response == expected


def delete_candidate(candidate: Candidate) -> tuple[bool, int, str]:
    is_valid, reason = validate_candidate(candidate)
    if not is_valid:
        return False, 0, reason

    size = path_size(candidate.path)
    try:
        if candidate.path.is_dir():
            shutil.rmtree(candidate.path)
        else:
            candidate.path.unlink()
    except OSError as exc:
        return False, 0, str(exc)

    return True, size, ""


def delete_candidates(candidates: list[Candidate]) -> int:
    if not candidates:
        return 0

    if not confirm(
        "Type DELETE to permanently remove the candidates listed above: ",
        "DELETE",
    ):
        print("Confirmation not received. Nothing was deleted.")
        return 0

    backup_candidates = [
        candidate for candidate in candidates if candidate.requires_extra_backup_confirmation
    ]
    if backup_candidates:
        print()
        print("Backup folders require an additional confirmation:")
        for candidate in backup_candidates:
            print(f"  - {printable(candidate.path)}")
        if not confirm("Type BACKUPS to delete backup folders too: ", "BACKUPS"):
            print("Backup confirmation not received. Backup folders will be skipped.")
            candidates = [
                candidate
                for candidate in candidates
                if not candidate.requires_extra_backup_confirmation
            ]

    total_deleted = 0
    print()
    print("Deleting:")
    for candidate in candidates:
        ok, size, error = delete_candidate(candidate)
        if ok:
            total_deleted += size
            print(f"  Deleted: {printable(candidate.path)} ({format_size(size)})")
        else:
            print(f"  Skipped: {printable(candidate.path)} ({error})")

    print()
    print(f"Total size deleted: {format_size(total_deleted)}")
    return total_deleted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run cleanup scanner for temporary and duplicate data in "
            "Fetal_SR_Project."
        )
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="actually delete candidates after interactive confirmation",
    )
    parser.add_argument(
        "--include_failed_experiments",
        action="store_true",
        help="include failed DMUISR-inspired output folders in the candidate list",
    )
    parser.add_argument(
        "--include_backups",
        action="store_true",
        help=(
            "include results/backups/* folders; still requires an extra BACKUPS "
            "confirmation"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("Fetal_SR_Project cleanup scanner")
    print(f"Project root: {PROJECT_ROOT}")
    print()
    print_protected_summary()

    candidates = collect_candidates(args)
    print_candidates(candidates)
    print()

    if not args.delete:
        print("Dry run only. Re-run with --delete to remove the candidates above.")
        if not args.include_failed_experiments:
            print(
                "Failed DMUISR-inspired outputs are excluded unless "
                "--include_failed_experiments is provided."
            )
        return 0

    delete_candidates(candidates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
