"""Evaluate the HC18 conditional diffusion super-resolution baseline."""

import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader

# Ensure the scripts directory is on sys.path so we can import HC18DiffusionDataset.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from hc18_diffusion_dataset import HC18DiffusionDataset


def load_training_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("train_diffusion_sr_hc18", str(script_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.squeeze().cpu().numpy().astype(np.float32)
    return np.clip(image, 0.0, 1.0)


def save_grayscale_png(path: Path, image: np.ndarray) -> None:
    if image.ndim == 2:
        out = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    else:
        out = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    cv2.imwrite(str(path), out)


def compute_edge_similarity(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    dx_pred = cv2.Sobel(pred, cv2.CV_32F, 1, 0, ksize=3)
    dy_pred = cv2.Sobel(pred, cv2.CV_32F, 0, 1, ksize=3)
    dx_target = cv2.Sobel(target, cv2.CV_32F, 1, 0, ksize=3)
    dy_target = cv2.Sobel(target, cv2.CV_32F, 0, 1, ksize=3)
    edge_pred = np.sqrt(dx_pred ** 2 + dy_pred ** 2).ravel()
    edge_target = np.sqrt(dx_target ** 2 + dy_target ** 2).ravel()
    norm_pred = np.linalg.norm(edge_pred)
    norm_target = np.linalg.norm(edge_target)
    if norm_pred < 1e-8 or norm_target < 1e-8:
        return 1.0 if np.allclose(edge_pred, edge_target, atol=1e-6) else 0.0
    return float(np.dot(edge_pred, edge_target) / (norm_pred * norm_target))


def build_visual_figure(lq: np.ndarray, sr: np.ndarray, hr: np.ndarray, error_map: np.ndarray) -> np.ndarray:
    def to_uchar(img):
        img_flat = np.clip(img, 0.0, 1.0)
        return (img_flat * 255.0).round().astype(np.uint8)

    h, w = lq.shape
    top = np.concatenate([to_uchar(lq), to_uchar(sr)], axis=1)
    bottom = np.concatenate([to_uchar(hr), to_uchar(error_map)], axis=1)
    figure = np.concatenate([top, bottom], axis=0)
    figure = cv2.cvtColor(figure, cv2.COLOR_GRAY2BGR)
    return figure


def sample_diffusion(
    model: torch.nn.Module,
    lq: torch.Tensor,
    schedule: dict,
    sampling_steps: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    batch_size, _, height, width = lq.shape
    x = torch.randn(batch_size, 1, height, width, device=device)

    timesteps = schedule["betas"].shape[0]
    step_indices = np.linspace(0, timesteps - 1, sampling_steps, dtype=int)[::-1]

    with torch.no_grad():
        for i, t in enumerate(step_indices):
            t_tensor = torch.full((batch_size,), int(t), device=device, dtype=torch.long)
            model_input = torch.cat([x, lq], dim=1)
            pred_noise = model(model_input, t_tensor)

            sqrt_alpha_t = schedule["sqrt_alpha_cumprod"][t]
            sqrt_one_minus_alpha_t = schedule["sqrt_one_minus_alpha_cumprod"][t]
            x0_pred = (x - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t

            if i + 1 < len(step_indices):
                alpha_prev = schedule["alpha_cumprod"][step_indices[i + 1]]
            else:
                alpha_prev = torch.tensor(1.0, device=device, dtype=sqrt_alpha_t.dtype)

            x = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1.0 - alpha_prev) * pred_noise

    return torch.clamp(x, 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HC18 diffusion SR model")
    parser.add_argument("--sampling_steps", type=int, default=50, help="Number of reverse sampling steps")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of test samples to evaluate")
    parser.add_argument("--image_size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    script_path = SCRIPT_DIR / "11_train_diffusion_sr_hc18.py"
    module = load_training_module(script_path)

    checkpoint_path = Path("results/models/diffusion_sr_hc18_best.pth")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    base_channels = int(checkpoint.get("base_channels", 64))
    if args.image_size is not None:
        image_size = args.image_size
    elif "image_size" in checkpoint:
        image_size = int(checkpoint["image_size"])
    else:
        image_size = 128
    print(f"Using image_size: {image_size}")
    timesteps = int(checkpoint.get("timesteps", 1000))

    model = module.ConditionalUNet(base_channels=base_channels, image_size=image_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    betas = checkpoint.get("betas")
    if betas is None:
        betas = module.linear_beta_schedule(timesteps)

    betas = betas.to(device)
    schedule = module.prepare_diffusion_schedule(betas)
    schedule = {k: v.to(device) if torch.is_tensor(v) else v for k, v in schedule.items()}

    results_root = Path("results")
    sr_dir = results_root / "sr_outputs" / "diffusion_sr"
    visual_dir = results_root / "visual_diffusion_sr"
    per_image_path = results_root / "diffusion_sr_hc18_per_image.csv"
    summary_path = results_root / "diffusion_sr_hc18_summary.csv"
    create_dirs(sr_dir, visual_dir)

    dataset = HC18DiffusionDataset(
        csv_path="data/processed/HC18_SR/hc18_lq_manifest.csv",
        split="test",
        image_size=image_size,
        input_type="LQ_mixed",
        return_quality=False,
    )
    sample_count = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))

    metrics = []
    visualization_count = 0

    with open(per_image_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "psnr", "ssim", "mae", "edge_score", "inference_time"])

        for index in range(sample_count):
            sample = dataset[index]
            image_id = sample["image_id"]
            lq = sample["lq"].unsqueeze(0).to(device)
            hr = sample["hr"].unsqueeze(0).to(device)

            start_time = time.perf_counter()
            sr = sample_diffusion(model, lq, schedule, args.sampling_steps, device)
            inference_time = time.perf_counter() - start_time

            lq_np = to_numpy_image(lq[0])
            hr_np = to_numpy_image(hr[0])
            sr_np = to_numpy_image(sr[0])
            error_np = np.abs(hr_np - sr_np)

            psnr = float(peak_signal_noise_ratio(hr_np, sr_np, data_range=1.0))
            ssim = float(structural_similarity(hr_np, sr_np, data_range=1.0))
            mae = float(np.mean(np.abs(hr_np - sr_np)))
            edge_score = compute_edge_similarity(sr_np, hr_np)

            writer.writerow([image_id, f"{psnr:.6f}", f"{ssim:.6f}", f"{mae:.6f}", f"{edge_score:.6f}", f"{inference_time:.6f}"])
            f.flush()

            metrics.append({
                "psnr": psnr,
                "ssim": ssim,
                "mae": mae,
                "edge_score": edge_score,
                "inference_time": inference_time,
            })

            save_grayscale_png(sr_dir / f"{image_id}.png", sr_np)

            if visualization_count < 10:
                figure = build_visual_figure(lq_np, sr_np, hr_np, error_np)
                visual_path = visual_dir / f"{image_id}.png"
                cv2.imwrite(str(visual_path), figure)
                visualization_count += 1

    if len(metrics) == 0:
        raise RuntimeError("No test samples were evaluated.")

    def mean_std(values):
        arr = np.array(values, dtype=np.float64)
        return float(arr.mean()), float(arr.std())

    psnr_mean, psnr_std = mean_std([m["psnr"] for m in metrics])
    ssim_mean, ssim_std = mean_std([m["ssim"] for m in metrics])
    mae_mean, mae_std = mean_std([m["mae"] for m in metrics])
    edge_mean, edge_std = mean_std([m["edge_score"] for m in metrics])
    time_mean, time_std = mean_std([m["inference_time"] for m in metrics])

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std"])
        writer.writerow(["psnr", f"{psnr_mean:.6f}", f"{psnr_std:.6f}"])
        writer.writerow(["ssim", f"{ssim_mean:.6f}", f"{ssim_std:.6f}"])
        writer.writerow(["mae", f"{mae_mean:.6f}", f"{mae_std:.6f}"])
        writer.writerow(["edge_score", f"{edge_mean:.6f}", f"{edge_std:.6f}"])
        writer.writerow(["inference_time", f"{time_mean:.6f}", f"{time_std:.6f}"])

    print(
        "method, mean_psnr, std_psnr, mean_ssim, std_ssim, mean_mae, std_mae, mean_edge_score, mean_inference_time"
    )
    print(
        f"diffusion_sr, {psnr_mean:.6f}, {psnr_std:.6f}, {ssim_mean:.6f}, {ssim_std:.6f}, "
        f"{mae_mean:.6f}, {mae_std:.6f}, {edge_mean:.6f}, {time_mean:.6f}"
    )


if __name__ == "__main__":
    main()
