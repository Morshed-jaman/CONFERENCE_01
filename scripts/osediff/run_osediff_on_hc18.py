"""Run upstream OSEDiff RealSR x4 on the HC18 test split."""

from __future__ import annotations

import argparse
import csv
import faulthandler
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFERENCE_ROOT = PROJECT_ROOT.parent
OSEDIFF_ROOT = CONFERENCE_ROOT / "baselines" / "repos" / "OSEDiff"
VENV_PYTHON = CONFERENCE_ROOT / "baselines" / "venvs" / "osediff" / "Scripts" / "python.exe"
HF_HOME = Path(r"D:\hf_cache")
SD21_PATH = HF_HOME / "hub" / "models--Manojb--stable-diffusion-2-1-base" / "snapshots" / "0094d483a120f3f33dafbd187ea4aa60d10de75c"
RAM_PATH = HF_HOME / "hub" / "spaces--xinyu1205--recognize-anything" / "snapshots" / "7c76cec1377de0d46df8091a032c4b667aea4459" / "ram_swin_large_14m.pth"
OSEDIFF_WEIGHT = OSEDIFF_ROOT / "preset" / "models" / "osediff.pkl"
DAPE_WEIGHT = OSEDIFF_ROOT / "preset" / "models" / "DAPE.pth"
UPSTREAM_LAUNCHER = Path(__file__).with_name("_run_test_osediff.py")

DEFAULT_TEST_CSV = PROJECT_ROOT / "data" / "processed" / "HC18_SR" / "test.csv"
DEFAULT_LQ_DIR = PROJECT_ROOT / "data" / "processed" / "HC18_SR" / "LQ_mixed"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "eval" / "osediff" / "test" / "restored"
DEFAULT_LOG = PROJECT_ROOT / "results" / "logs" / "osediff_reproduction_log.md"
DEFAULT_PER_IMAGE_LOG = PROJECT_ROOT / "results" / "logs" / "osediff_batch_perimage.csv"
DEFAULT_DIAG_LOG = PROJECT_ROOT / "results" / "logs" / "osediff_batch_diag.jsonl"


@dataclass
class ImageRecord:
    image_id: str
    chop_size: int
    peak_vram_mib: float
    runtime_sec: float
    native_size: tuple[int, int]
    output_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OSEDiff RealSR x4 on HC18.")
    parser.add_argument("--test_csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--lq_dir", type=Path, default=DEFAULT_LQ_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--log_file", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--per_image_log", type=Path, default=DEFAULT_PER_IMAGE_LOG)
    parser.add_argument("--diag_log", type=Path, default=DEFAULT_DIAG_LOG)
    parser.add_argument(
        "--files",
        type=Path,
        default=None,
        help="Text manifest containing one input filename or image ID per line.",
    )
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--skip_existing", "--skip-existing", action="store_true")
    parser.add_argument("--confirm_full_run", action="store_true")
    parser.add_argument("--chunk", type=int, default=30)
    parser.add_argument("--cooldown", type=int, default=180)
    parser.add_argument(
        "--chop_size", "--chop-size", type=int, choices=(64, 128, 256), default=256
    )
    parser.add_argument("--text_encoder_cpu", action="store_true")
    parser.add_argument("--skip_dape", action="store_true")
    parser.add_argument("--watchdog_seconds", type=int, default=600)
    parser.add_argument("--heartbeat_seconds", type=int, default=900)
    return parser.parse_args()


def append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text.rstrip() + "\n")
    except PermissionError:
        fallback = path.with_name(f"{path.stem}.locked{path.suffix}")
        try:
            with fallback.open("a", encoding="utf-8") as handle:
                handle.write(text.rstrip() + "\n")
            print(f"WARNING: {path} was locked. Wrote to {fallback} instead.", flush=True)
        except OSError as exc:
            # Reproduction logging is diagnostic only; a locked log must never
            # prevent the resumable inference batch from producing images.
            print(
                f"WARNING: reproduction log unavailable ({exc}); continuing.",
                flush=True,
            )


def load_test_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sorted(row["image_id"] for row in csv.DictReader(handle) if row.get("image_id"))


def select_ids(args: argparse.Namespace) -> list[str]:
    if args.files is not None:
        ids = [
            Path(line.strip()).stem
            for line in args.files.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        ids = load_test_ids(args.test_csv)
    selected = [
        image_id
        for image_id in ids
        if not (args.skip_existing and (args.out_dir / f"{image_id}.png").exists())
    ]
    count = len(selected) if (args.confirm_full_run or args.files is not None) else args.limit
    return selected[:count]


def query_vram_mib() -> float:
    # Per-image peak memory is reported by torch.cuda in the inference
    # subprocess. Avoid repeated nvidia-smi child processes on Windows.
    return 0.0


def upstream_command(
    input_dir: Path,
    raw_dir: Path,
    chop_size: int,
    diag_log: Path,
    text_encoder_cpu: bool,
    skip_dape: bool,
) -> list[str]:
    latent_tile = max(16, chop_size // 4)
    latent_overlap = max(4, min(16, latent_tile // 3))
    decoder_tile = max(64, min(224, chop_size))
    command = [
        str(VENV_PYTHON),
        str(UPSTREAM_LAUNCHER),
        "-i",
        str(input_dir),
        "-o",
        str(raw_dir),
        "--osediff_path",
        str(OSEDIFF_WEIGHT),
        "--pretrained_model_name_or_path",
        str(SD21_PATH),
        "--ram_ft_path",
        str(DAPE_WEIGHT),
        "--ram_path",
        str(RAM_PATH),
        "--upscale",
        "4",
        "--process_size",
        str(chop_size),
        "--latent_tiled_size",
        str(latent_tile),
        "--latent_tiled_overlap",
        str(latent_overlap),
        "--vae_decoder_tiled_size",
        str(decoder_tile),
        "--mixed_precision",
        "fp16",
        "--align_method",
        "nofix",
        "--diag_log",
        str(diag_log),
    ]
    if text_encoder_cpu:
        command.append("--text_encoder_cpu")
    if skip_dape:
        command.append("--skip_dape")
    return command


def postprocess(raw_path: Path, output_path: Path) -> tuple[int, int]:
    with Image.open(raw_path) as image:
        native_size = image.size
        final = image.convert("L").resize((256, 256), Image.Resampling.BICUBIC)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final.save(output_path)
    return native_size


def run_attempt(
    image_ids: list[str],
    args: argparse.Namespace,
    chop_size: int,
    run_index: int,
    total_chunks: int,
) -> tuple[list[ImageRecord], str]:
    env = os.environ.copy()
    env["HF_HOME"] = str(HF_HOME)
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["OSEDIFF_STAGE_LOG"] = str(
        args.diag_log.with_name(f"{args.diag_log.stem}_stages.jsonl")
    )

    with tempfile.TemporaryDirectory(prefix="osediff_hc18_", dir=CONFERENCE_ROOT) as temp_name:
        temp_root = Path(temp_name)
        input_dir = temp_root / "input"
        raw_dir = temp_root / "raw"
        input_dir.mkdir()
        raw_dir.mkdir()
        for image_id in image_ids:
            source = args.lq_dir / f"{image_id}.png"
            if not source.exists():
                raise FileNotFoundError(source)
            shutil.copy2(source, input_dir / source.name)

        command = upstream_command(
            input_dir,
            raw_dir,
            chop_size,
            args.diag_log,
            args.text_encoder_cpu,
            args.skip_dape,
        )
        process = subprocess.Popen(
            command,
            cwd=OSEDIFF_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        started = time.monotonic()
        last_output = started
        last_heartbeat = started
        last_completed = started
        seen: set[str] = set()
        peak_since_output = 0.0
        records: list[ImageRecord] = []

        while process.poll() is None:
            now = time.monotonic()
            peak_since_output = max(peak_since_output, query_vram_mib())
            for raw_path in sorted(raw_dir.glob("*.png")):
                image_id = raw_path.stem
                if image_id in seen:
                    continue
                output_path = args.out_dir / raw_path.name
                native_size = postprocess(raw_path, output_path)
                runtime = now - last_completed
                records.append(
                    ImageRecord(
                        image_id=image_id,
                        chop_size=chop_size,
                        peak_vram_mib=peak_since_output,
                        runtime_sec=runtime,
                        native_size=native_size,
                        output_path=output_path,
                    )
                )
                seen.add(image_id)
                last_output = now
                last_completed = now
                peak_since_output = 0.0
            if args.watchdog_seconds > 0 and now - last_output > args.watchdog_seconds:
                process.kill()
                raise RuntimeError(f"OSEDiff watchdog: no new PNG for {args.watchdog_seconds}s")
            if now - last_heartbeat >= args.heartbeat_seconds:
                completed_total = len(list(args.out_dir.glob("*.png")))
                print(
                    f"[{datetime.now():%H:%M}] OSEDiff progress: {completed_total}/150 images, "
                    f"peak VRAM {max((record.peak_vram_mib for record in records), default=peak_since_output):.0f} MiB, "
                    f"last chop_size {chop_size}, chunk {run_index}/{total_chunks}.",
                    flush=True,
                )
                last_heartbeat = now
            time.sleep(2)

        stdout, stderr = process.communicate()
        for raw_path in sorted(raw_dir.glob("*.png")):
            if raw_path.stem in seen:
                continue
            now = time.monotonic()
            output_path = args.out_dir / raw_path.name
            native_size = postprocess(raw_path, output_path)
            records.append(
                ImageRecord(
                    image_id=raw_path.stem,
                    chop_size=chop_size,
                    peak_vram_mib=peak_since_output,
                    runtime_sec=now - last_completed,
                    native_size=native_size,
                    output_path=output_path,
                )
            )
            seen.add(raw_path.stem)
            last_completed = now

        combined = f"{stdout}\n{stderr}"
        if process.returncode != 0 and "out of memory" not in combined.lower():
            raise RuntimeError(f"Upstream OSEDiff failed ({process.returncode}):\n{combined[-4000:]}")
        upstream_metrics = {
            match.group("filename"): (
                float(match.group("runtime")),
                float(match.group("peak")),
            )
            for match in re.finditer(
                r"OSEDIFF_METRIC filename=(?P<filename>\S+) "
                r"runtime_sec=(?P<runtime>[0-9.]+) "
                r"peak_vram_mib=(?P<peak>[0-9.]+)",
                combined,
            )
        }
        for record in records:
            metric = upstream_metrics.get(record.output_path.name)
            if metric is not None:
                record.runtime_sec, record.peak_vram_mib = metric
        return records, combined


def log_record(record: ImageRecord, args: argparse.Namespace) -> None:
    with Image.open(record.output_path) as image:
        array = np.asarray(image)
    line = (
        f"{record.image_id}.png, chop_size_used={record.chop_size}, "
        f"peak_vram_mib={record.peak_vram_mib:.1f}, runtime_sec={record.runtime_sec:.2f}, "
        f"native={record.native_size[0]}x{record.native_size[1]}, final=256x256, "
        f"min={int(array.min())}, max={int(array.max())}, "
        f"mean={float(array.mean()):.3f}, std={float(array.std()):.3f}"
    )
    print(line, flush=True)
    append_log(args.log_file, line)
    args.per_image_log.parent.mkdir(parents=True, exist_ok=True)
    csv_exists = args.per_image_log.exists()
    try:
        with args.per_image_log.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            if not csv_exists:
                writer.writerow(
                    ["filename", "runtime_sec", "peak_vram_mib", "chop_size_used", "status"]
                )
            writer.writerow(
                [
                    record.output_path.name,
                    f"{record.runtime_sec:.6f}",
                    f"{record.peak_vram_mib:.3f}",
                    record.chop_size,
                    "ok" if record.chop_size == args.chop_size else "oom_fallback",
                ]
            )
    except PermissionError:
        fallback = args.per_image_log.with_name(f"{args.per_image_log.stem}.locked.csv")
        fallback_exists = fallback.exists()
        try:
            with fallback.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                if not fallback_exists:
                    writer.writerow(
                        ["filename", "runtime_sec", "peak_vram_mib", "chop_size_used", "status"]
                    )
                writer.writerow(
                    [
                        record.output_path.name,
                        f"{record.runtime_sec:.6f}",
                        f"{record.peak_vram_mib:.3f}",
                        record.chop_size,
                        "ok" if record.chop_size == args.chop_size else "oom_fallback",
                    ]
                )
            print(f"WARNING: {args.per_image_log} was locked. Wrote to {fallback}.", flush=True)
        except OSError as exc:
            print(f"WARNING: per-image log unavailable ({exc}); continuing.", flush=True)


def run_chunk(
    image_ids: list[str],
    args: argparse.Namespace,
    chunk_index: int,
    total_chunks: int,
    sticky_chop: int,
) -> int:
    pending = list(image_ids)
    chop_size = sticky_chop
    while pending:
        records, output = run_attempt(pending, args, chop_size, chunk_index, total_chunks)
        completed = {record.image_id for record in records}
        for record in records:
            log_record(record, args)
        pending = [image_id for image_id in pending if image_id not in completed]
        if not pending:
            return chop_size
        if "out of memory" not in output.lower() or chop_size <= 64:
            raise RuntimeError(f"OSEDiff failed with {len(pending)} unfinished images at chop_size={chop_size}")
        new_chop = chop_size // 2
        print(f"sticky chop_size downgrade {chop_size} -> {new_chop}", flush=True)
        append_log(args.log_file, f"sticky chop_size downgrade {chop_size} -> {new_chop}")
        chop_size = new_chop
    return chop_size


def validate_paths() -> None:
    for required in (
        VENV_PYTHON,
        SD21_PATH,
        RAM_PATH,
        OSEDIFF_WEIGHT,
        DAPE_WEIGHT,
        UPSTREAM_LAUNCHER,
    ):
        if not required.exists():
            raise FileNotFoundError(required)


def main() -> int:
    faulthandler.dump_traceback_later(150, repeat=True, file=sys.stderr)
    args = parse_args()
    validate_paths()
    if args.limit < 1 or args.chunk < 1:
        raise ValueError("--limit and --chunk must be positive")
    if args.limit > 3 and not args.confirm_full_run:
        raise RuntimeError("Sanity gate active: use --confirm_full_run for more than 3 images.")

    selected = select_ids(args)
    if not selected:
        print("No images selected.", flush=True)
        return 0
    chunks = [selected[index : index + args.chunk] for index in range(0, len(selected), args.chunk)]
    sticky_chop = args.chop_size
    append_log(
        args.log_file,
        f"\n## OSEDiff HC18 run\nids={len(selected)}, chunks={len(chunks)}, "
        f"HF_HOME={HF_HOME}, initial_chop_size={sticky_chop}",
    )
    for index, chunk_ids in enumerate(chunks, start=1):
        free_disk_gib = shutil.disk_usage(CONFERENCE_ROOT).free / (1024**3)
        print(
            f"Chunk {index}/{len(chunks)} starting at {datetime.now().isoformat(timespec='seconds')}, "
            f"free VRAM: {query_vram_mib():.0f} MiB, free disk D: {free_disk_gib:.2f} GiB.",
            flush=True,
        )
        sticky_chop = run_chunk(chunk_ids, args, index, len(chunks), sticky_chop)
        completed_total = len(list(args.out_dir.glob("*.png")))
        print(
            f"Chunk {index}/{len(chunks)} complete; restored PNG total={completed_total}.",
            flush=True,
        )
        if index < len(chunks) and args.cooldown:
            print(f"Cooldown {args.cooldown}s after chunk {index}/{len(chunks)}.", flush=True)
            time.sleep(args.cooldown)
    print(f"SANITY_GATE_STOP processed={len(selected)} outputs={args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
