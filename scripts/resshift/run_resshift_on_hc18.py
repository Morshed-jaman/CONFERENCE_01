from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

assert torch.cuda.is_available(), "CUDA unavailable - abort"

import _torchvision_compat  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESSHIFT_ROOT = PROJECT_ROOT.parent / "baselines" / "repos" / "ResShift"
if str(RESSHIFT_ROOT) not in sys.path:
    sys.path.insert(0, str(RESSHIFT_ROOT))

import cv2
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from sampler import ResShiftSampler
from utils.util_image import ImageSpliterTh


DEFAULT_TEST_CSV = PROJECT_ROOT / "data" / "processed" / "HC18_SR" / "test.csv"
DEFAULT_LQ_DIR = PROJECT_ROOT / "data" / "processed" / "HC18_SR" / "LQ_mixed"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "eval" / "resshift" / "test" / "restored"
DEFAULT_LOG = PROJECT_ROOT / "results" / "logs" / "resshift_reproduction_log.md"
CONFIG_PATH = RESSHIFT_ROOT / "configs" / "realsr_swinunet_realesrgan256.yaml"
SR_WEIGHT = RESSHIFT_ROOT / "weights" / "resshift_realsrx4_s15.pth"
AUTOENCODER_WEIGHT = RESSHIFT_ROOT / "weights" / "autoencoder_vq_f4.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ResShift v1 real-SR on HC18 test images.")
    parser.add_argument("--test_csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--lq_dir", type=Path, default=DEFAULT_LQ_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--log_file", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--confirm_full_run", action="store_true")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--chop_size", type=int, default=256)
    parser.add_argument("--chop_stride", type=int, default=224)
    parser.add_argument("--fallback_chop_size", type=int, default=128)
    parser.add_argument("--fallback_chop_stride", type=int, default=96)
    parser.add_argument("--no_fallback", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def load_test_ids(test_csv: Path) -> list[str]:
    with test_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    ids = [row["image_id"] for row in rows if row.get("image_id")]
    if not ids:
        raise RuntimeError(f"No image_id rows found in {test_csv}")
    return ids


def select_ids_for_run(all_ids: list[str], args: argparse.Namespace) -> list[str]:
    selected: list[str] = []
    for image_id in all_ids:
        out_path = args.out_dir / f"{image_id}.png"
        if args.skip_existing and out_path.exists():
            continue
        selected.append(image_id)
        if len(selected) >= args.limit:
            break
    return selected


def build_sampler(args: argparse.Namespace) -> ResShiftSampler:
    for required in (CONFIG_PATH, SR_WEIGHT, AUTOENCODER_WEIGHT):
        if not required.exists():
            raise FileNotFoundError(required)

    configs = OmegaConf.load(str(CONFIG_PATH))
    configs.model.ckpt_path = str(SR_WEIGHT)
    configs.autoencoder.ckpt_path = str(AUTOENCODER_WEIGHT)
    configs.diffusion.params.steps = args.steps
    configs.diffusion.params.sf = args.scale

    return ResShiftSampler(
        configs,
        chop_size=args.chop_size,
        chop_stride=args.chop_stride,
        chop_bs=1,
        use_fp16=True,
        seed=args.seed,
        desired_min_size=64,
    )


def stride_for_chop(chop_size: int, args: argparse.Namespace) -> int:
    if chop_size == args.chop_size:
        return args.chop_stride
    if chop_size == args.fallback_chop_size:
        return args.fallback_chop_stride
    return max(chop_size - 32, chop_size // 2)


def load_hc18_lq_rgb(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("L")
    if image.size != (256, 256):
        raise RuntimeError(f"{path} is {image.size}, expected 256x256")
    gray = np.asarray(image, dtype=np.float32) / 255.0
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor.cuda(non_blocking=True)


@torch.inference_mode()
def run_tensor(
    sampler: ResShiftSampler,
    lq_tensor: torch.Tensor,
    chop_size: int,
    chop_stride: int,
) -> torch.Tensor:
    old_chop_size = sampler.chop_size
    old_chop_stride = sampler.chop_stride
    sampler.chop_size = chop_size
    sampler.chop_stride = chop_stride
    try:
        if lq_tensor.shape[2] > chop_size or lq_tensor.shape[3] > chop_size:
            splitter = ImageSpliterTh(
                lq_tensor,
                chop_size,
                stride=chop_stride,
                sf=sampler.sf,
                extra_bs=sampler.chop_bs,
            )
            for patch, index_infos in splitter:
                sr_patch = sampler.sample_func((patch - 0.5) / 0.5, noise_repeat=False)
                splitter.update(sr_patch, index_infos)
            sr_tensor = splitter.gather()
        else:
            sr_tensor = sampler.sample_func((lq_tensor - 0.5) / 0.5, noise_repeat=False)
        return (sr_tensor * 0.5 + 0.5).clamp(0.0, 1.0)
    finally:
        sampler.chop_size = old_chop_size
        sampler.chop_stride = old_chop_stride


def tensor_to_rgb_uint8(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.squeeze(0).detach().float().cpu().permute(1, 2, 0).numpy()
    return (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def to_final_grayscale(native_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(native_rgb, cv2.COLOR_RGB2GRAY)
    final = Image.fromarray(gray).resize((256, 256), Image.Resampling.BICUBIC)
    return np.asarray(final, dtype=np.uint8)


def process_one(
    sampler: ResShiftSampler,
    image_id: str,
    args: argparse.Namespace,
    chop_size: int,
    chop_stride: int,
) -> dict[str, object]:
    lq_path = args.lq_dir / f"{image_id}.png"
    out_path = args.out_dir / f"{image_id}.png"
    if args.skip_existing and out_path.exists():
        return {"image_id": image_id, "skipped": True, "out_path": str(out_path)}
    if not lq_path.exists():
        raise FileNotFoundError(lq_path)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    effective_chop = chop_size
    effective_stride = chop_stride
    downgraded = False
    while True:
        lq_tensor = load_hc18_lq_rgb(lq_path)
        try:
            sr_tensor = run_tensor(sampler, lq_tensor, effective_chop, effective_stride)
            break
        except torch.cuda.OutOfMemoryError:
            del lq_tensor
            torch.cuda.empty_cache()
            if args.no_fallback or effective_chop <= 64:
                raise
            effective_chop = max(64, effective_chop // 2)
            effective_stride = stride_for_chop(effective_chop, args)
            downgraded = True
            torch.cuda.reset_peak_memory_stats()

    native_rgb = tensor_to_rgb_uint8(sr_tensor)
    final_gray = to_final_grayscale(native_rgb)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(final_gray, mode="L").save(out_path)

    peak_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
    seconds = time.perf_counter() - start
    native_h, native_w = native_rgb.shape[:2]
    final_min = int(final_gray.min())
    final_max = int(final_gray.max())
    del lq_tensor, sr_tensor
    torch.cuda.empty_cache()
    return {
        "image_id": image_id,
        "skipped": False,
        "native_hw": f"{native_h}x{native_w}",
        "final_hw": f"{final_gray.shape[0]}x{final_gray.shape[1]}",
        "dtype": str(final_gray.dtype),
        "range": f"{final_min}..{final_max}",
        "peak_mib": peak_mib,
        "seconds": seconds,
        "chop_size": effective_chop,
        "fallback": downgraded,
        "sticky_chop_size": effective_chop,
        "sticky_chop_stride": effective_stride,
        "out_path": str(out_path),
    }


def main() -> int:
    args = parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.limit > 3 and not args.confirm_full_run:
        raise RuntimeError("Sanity gate active: use --confirm_full_run to process more than 3 images.")

    ids = select_ids_for_run(load_test_ids(args.test_csv), args)
    header = (
        "\n## HC18 ResShift wrapper run\n\n"
        f"- limit: {args.limit}\n"
        f"- ids: {', '.join(ids)}\n"
        f"- out_dir: {args.out_dir}\n"
        f"- chop_size: {args.chop_size}, fallback_chop_size: {args.fallback_chop_size}\n"
    )
    append_log(args.log_file, header)

    sampler = build_sampler(args)
    current_chop_size = args.chop_size
    current_chop_stride = args.chop_stride
    sticky_downgrade_logged = False
    for image_id in ids:
        result = process_one(sampler, image_id, args, current_chop_size, current_chop_stride)
        if not result.get("skipped"):
            new_chop_size = int(result["sticky_chop_size"])
            new_chop_stride = int(result["sticky_chop_stride"])
            if new_chop_size < current_chop_size:
                current_chop_size = new_chop_size
                current_chop_stride = new_chop_stride
                if not sticky_downgrade_logged:
                    downgrade_line = f"sticky downgrade to chop_size={current_chop_size}"
                    print(downgrade_line, flush=True)
                    append_log(args.log_file, downgrade_line)
                    sticky_downgrade_logged = True
        if result.get("skipped"):
            line = f"{image_id}: skipped existing {result['out_path']}"
        else:
            line = (
                f"{image_id}: native={result['native_hw']} final={result['final_hw']} "
                f"dtype={result['dtype']} range={result['range']} "
                f"peak_mib={result['peak_mib']:.1f} seconds={result['seconds']:.2f} "
                f"chop={result['chop_size']} fallback={result['fallback']} "
                f"out={result['out_path']}"
            )
        print(line, flush=True)
        append_log(args.log_file, line)

    print(f"SANITY_GATE_STOP processed={len(ids)} outputs={args.out_dir}", flush=True)
    append_log(args.log_file, f"SANITY_GATE_STOP processed={len(ids)} outputs={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
