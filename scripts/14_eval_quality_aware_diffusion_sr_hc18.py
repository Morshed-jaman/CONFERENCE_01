"""Evaluate the main quality-aware diffusion SR model on the HC18 test split.

This evaluator loads the exact model architecture from the quality-aware
training script and conditions every reverse-diffusion step on the five HC18
image-quality features.
"""

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

# Ensure the scripts directory is on sys.path so HC18DiffusionDataset can be imported.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from hc18_diffusion_dataset import HC18DiffusionDataset


def load_training_module(script_path: Path):
    """Load the training module so evaluation uses the checkpoint's exact architecture."""
    spec = importlib.util.spec_from_file_location("train_quality_aware_diffusion_sr_hc18", str(script_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a single grayscale tensor to a clipped two-dimensional image."""
    image = np.squeeze(tensor.detach().cpu().numpy()).astype(np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image after squeeze, got shape {image.shape}")
    return np.clip(image, 0.0, 1.0)


def to_uint8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def save_grayscale_png(path: Path, image: np.ndarray) -> None:
    output = to_uint8(image)
    if not cv2.imwrite(str(path), output):
        raise OSError(f"Failed to save image: {path}")


def compute_edge_similarity(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute cosine similarity between Sobel edge-magnitude images."""
    pred = np.squeeze(pred).astype(np.float32)
    target = np.squeeze(target).astype(np.float32)

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


def make_labeled_panel(image: np.ndarray, label: str) -> np.ndarray:
    panel = cv2.cvtColor(to_uint8(image), cv2.COLOR_GRAY2BGR)
    for line_index, line in enumerate(label.splitlines()):
        position = (5, 16 + line_index * 15)
        cv2.putText(panel, line, position, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(panel, line, position, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def build_visual_figure(
    lq: np.ndarray,
    sr: np.ndarray,
    hr: np.ndarray,
    error_map: np.ndarray,
) -> np.ndarray:
    """Build a labeled 2x2 comparison for the main quality-aware method."""
    lq_panel = make_labeled_panel(lq, "LQ input")
    sr_panel = make_labeled_panel(sr, "Quality-aware\nDiffusion SR")
    hr_panel = make_labeled_panel(hr, "HR target")
    error_panel = make_labeled_panel(error_map, "Absolute error")

    top = np.concatenate([lq_panel, sr_panel], axis=1)
    bottom = np.concatenate([hr_panel, error_panel], axis=1)
    return np.concatenate([top, bottom], axis=0)


def sample_diffusion(
    model: torch.nn.Module,
    lq: torch.Tensor,
    quality: torch.Tensor,
    schedule: dict,
    sampling_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate quality-aware SR images with deterministic reduced-step sampling."""
    model.eval()
    batch_size, _, height, width = lq.shape
    x = torch.randn(batch_size, 1, height, width, device=device, dtype=lq.dtype)

    timesteps = schedule["betas"].shape[0]
    step_indices = np.linspace(0, timesteps - 1, sampling_steps, dtype=int)[::-1]

    with torch.no_grad():
        for index, timestep in enumerate(step_indices):
            timestep_value = int(timestep)
            t_tensor = torch.full(
                (batch_size,),
                timestep_value,
                device=device,
                dtype=torch.long,
            )
            model_input = torch.cat([x, lq], dim=1)
            pred_noise = model(model_input, t_tensor, quality)

            sqrt_alpha_t = schedule["sqrt_alpha_cumprod"][timestep_value]
            sqrt_one_minus_alpha_t = schedule["sqrt_one_minus_alpha_cumprod"][timestep_value]
            x0_pred = (x - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t

            if index + 1 < len(step_indices):
                previous_timestep = int(step_indices[index + 1])
                alpha_prev = schedule["alpha_cumprod"][previous_timestep]
            else:
                alpha_prev = torch.ones((), device=device, dtype=sqrt_alpha_t.dtype)

            x = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1.0 - alpha_prev) * pred_noise

    return torch.clamp(x, 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the quality-aware HC18 diffusion SR model")
    parser.add_argument("--sampling_steps", type=int, default=50, help="Number of reverse sampling steps")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of test samples")
    parser.add_argument("--image_size", type=int, default=None, help="Override checkpoint image size")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="results/models/quality_aware_diffusion_sr_hc18_best.pth",
        help="Quality-aware diffusion checkpoint to evaluate",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="quality_aware_diffusion_sr",
        help="Name used for evaluation output directories and CSV files",
    )
    return parser.parse_args()


def mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std())


def main() -> None:
    args = parse_args()
    if args.sampling_steps <= 0:
        raise ValueError("--sampling_steps must be greater than zero")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max_samples must be greater than zero when provided")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the exact architecture used to train the main proposed HC18 model.
    script_path = SCRIPT_DIR / "13_train_quality_aware_diffusion_sr_hc18.py"
    module = load_training_module(script_path)

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    base_channels = int(checkpoint.get("base_channels", 64))
    quality_dim = int(checkpoint.get("quality_dim", 5))
    quality_scale = float(checkpoint.get("quality_scale", 1.0))
    if "quality_scale" in checkpoint:
        print(f"Using quality_scale: {quality_scale}")
    timesteps = int(checkpoint.get("timesteps", 1000))

    if args.image_size is not None:
        image_size = args.image_size
    elif "image_size" in checkpoint:
        image_size = int(checkpoint["image_size"])
    else:
        image_size = 128
    print(f"Using image_size: {image_size}")

    model = module.QualityAwareConditionalUNet(
        base_channels=base_channels,
        image_size=image_size,
        quality_dim=quality_dim,
        quality_scale=quality_scale,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    betas = checkpoint.get("betas")
    if betas is None:
        betas = module.linear_beta_schedule(timesteps)
    betas = betas.to(device)

    schedule = module.prepare_diffusion_schedule(betas)
    schedule = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in schedule.items()
    }

    results_root = Path("results")
    sr_dir = results_root / "sr_outputs" / args.output_name
    visual_dir = results_root / f"visual_{args.output_name}"
    per_image_path = results_root / f"{args.output_name}_hc18_per_image.csv"
    summary_path = results_root / f"{args.output_name}_hc18_summary.csv"
    create_dirs(sr_dir, visual_dir)

    dataset = HC18DiffusionDataset(
        csv_path="data/processed/HC18_SR/hc18_lq_manifest.csv",
        split="test",
        image_size=image_size,
        input_type="LQ_mixed",
        return_quality=True,
    )
    sample_count = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))

    metrics = []
    visualization_count = 0

    with open(per_image_path, "w", newline="") as per_image_file:
        writer = csv.writer(per_image_file)
        writer.writerow(["image_id", "psnr", "ssim", "mae", "edge_score", "inference_time"])

        for index in range(sample_count):
            sample = dataset[index]
            image_id = sample["image_id"]
            lq = sample["lq"].unsqueeze(0).to(device)
            hr = sample["hr"].unsqueeze(0).to(device)
            quality = sample["quality"].unsqueeze(0).to(device=device, dtype=torch.float32)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start_time = time.perf_counter()
            sr = sample_diffusion(model, lq, quality, schedule, args.sampling_steps, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_time = time.perf_counter() - start_time

            lq_np = to_numpy_image(lq[0])
            hr_np = to_numpy_image(hr[0])
            sr_np = to_numpy_image(sr[0])
            error_np = np.abs(hr_np - sr_np)

            # All metric inputs are explicitly squeezed to two-dimensional grayscale.
            hr_2d = np.squeeze(hr_np)
            sr_2d = np.squeeze(sr_np)
            psnr = float(peak_signal_noise_ratio(hr_2d, sr_2d, data_range=1.0))
            ssim = float(structural_similarity(hr_2d, sr_2d, data_range=1.0))
            mae = float(np.mean(np.abs(hr_2d - sr_2d)))
            edge_score = compute_edge_similarity(sr_2d, hr_2d)

            writer.writerow(
                [
                    image_id,
                    f"{psnr:.6f}",
                    f"{ssim:.6f}",
                    f"{mae:.6f}",
                    f"{edge_score:.6f}",
                    f"{inference_time:.6f}",
                ]
            )
            per_image_file.flush()

            metrics.append(
                {
                    "psnr": psnr,
                    "ssim": ssim,
                    "mae": mae,
                    "edge_score": edge_score,
                    "inference_time": inference_time,
                }
            )

            save_grayscale_png(sr_dir / f"{image_id}.png", sr_2d)

            if visualization_count < 10:
                figure = build_visual_figure(lq_np, sr_2d, hr_2d, error_np)
                visual_path = visual_dir / f"{image_id}.png"
                if not cv2.imwrite(str(visual_path), figure):
                    raise OSError(f"Failed to save visual figure: {visual_path}")
                visualization_count = visualization_count + 1

    if not metrics:
        raise RuntimeError("No test samples were evaluated.")

    psnr_mean, psnr_std = mean_std([metric["psnr"] for metric in metrics])
    ssim_mean, ssim_std = mean_std([metric["ssim"] for metric in metrics])
    mae_mean, mae_std = mean_std([metric["mae"] for metric in metrics])
    edge_mean, edge_std = mean_std([metric["edge_score"] for metric in metrics])
    time_mean, time_std = mean_std([metric["inference_time"] for metric in metrics])

    with open(summary_path, "w", newline="") as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(["metric", "mean", "std"])
        writer.writerow(["psnr", f"{psnr_mean:.6f}", f"{psnr_std:.6f}"])
        writer.writerow(["ssim", f"{ssim_mean:.6f}", f"{ssim_std:.6f}"])
        writer.writerow(["mae", f"{mae_mean:.6f}", f"{mae_std:.6f}"])
        writer.writerow(["edge_score", f"{edge_mean:.6f}", f"{edge_std:.6f}"])
        writer.writerow(["inference_time", f"{time_mean:.6f}", f"{time_std:.6f}"])

    print(
        "method, mean_psnr, std_psnr, mean_ssim, std_ssim, mean_mae, "
        "std_mae, mean_edge_score, mean_inference_time"
    )
    print(
        f"{args.output_name}, {psnr_mean:.6f}, {psnr_std:.6f}, "
        f"{ssim_mean:.6f}, {ssim_std:.6f}, {mae_mean:.6f}, {mae_std:.6f}, "
        f"{edge_mean:.6f}, {time_mean:.6f}"
    )


if __name__ == "__main__":
    main()
