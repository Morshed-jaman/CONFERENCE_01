"""Evaluate the DMUISR-inspired diffusion SR model on the HC18 test split."""

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


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from hc18_diffusion_dataset import HC18DiffusionDataset


METHOD_NAME = "dmuisr_inspired_diffusion_sr"


def load_training_module(script_path: Path):
    """Import the numbered training script to reuse its exact model and schedule."""
    spec = importlib.util.spec_from_file_location(
        "train_dmuisr_inspired_diffusion_sr_hc18",
        str(script_path),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load training module from: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def choose_output_dirs(results_root: Path) -> tuple[Path, Path]:
    """Choose paired output directories without reusing an earlier run's folders."""
    sr_base = results_root / "sr_outputs" / METHOD_NAME
    visual_base = results_root / f"visual_{METHOD_NAME}"
    if not sr_base.exists() and not visual_base.exists():
        return sr_base, visual_base

    run_number = 1
    while True:
        suffix = f"_run_{run_number:03d}"
        sr_candidate = sr_base.with_name(sr_base.name + suffix)
        visual_candidate = visual_base.with_name(visual_base.name + suffix)
        if not sr_candidate.exists() and not visual_candidate.exists():
            return sr_candidate, visual_candidate
        run_number += 1


def to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert one grayscale tensor to a clipped two-dimensional float image."""
    image = np.squeeze(tensor.detach().cpu().numpy()).astype(np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {image.shape}")
    return np.clip(image, 0.0, 1.0)


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a normalized image to the 0-255 range used by PNG files."""
    return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def save_grayscale_png(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), to_uint8(image)):
        raise OSError(f"Failed to save image: {path}")


def compute_edge_similarity(prediction: np.ndarray, target: np.ndarray) -> float:
    """Return cosine similarity between Sobel edge-magnitude maps."""
    prediction = np.asarray(prediction, dtype=np.float32).squeeze()
    target = np.asarray(target, dtype=np.float32).squeeze()
    if prediction.ndim != 2 or target.ndim != 2:
        raise ValueError("Sobel edge similarity expects two-dimensional images")

    pred_x = cv2.Sobel(prediction, cv2.CV_32F, 1, 0, ksize=3)
    pred_y = cv2.Sobel(prediction, cv2.CV_32F, 0, 1, ksize=3)
    target_x = cv2.Sobel(target, cv2.CV_32F, 1, 0, ksize=3)
    target_y = cv2.Sobel(target, cv2.CV_32F, 0, 1, ksize=3)

    pred_edges = np.hypot(pred_x, pred_y).ravel()
    target_edges = np.hypot(target_x, target_y).ravel()
    pred_norm = float(np.linalg.norm(pred_edges))
    target_norm = float(np.linalg.norm(target_edges))
    if pred_norm < 1e-8 or target_norm < 1e-8:
        return 1.0 if np.allclose(pred_edges, target_edges, atol=1e-6) else 0.0
    return float(np.dot(pred_edges, target_edges) / (pred_norm * target_norm))


def make_labeled_panel(image: np.ndarray, label: str) -> np.ndarray:
    panel = cv2.cvtColor(to_uint8(image), cv2.COLOR_GRAY2BGR)
    for line_number, line in enumerate(label.splitlines()):
        position = (5, 16 + 15 * line_number)
        cv2.putText(
            panel,
            line,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            line,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return panel


def build_visual_figure(
    lq: np.ndarray,
    sr: np.ndarray,
    hr: np.ndarray,
    absolute_error: np.ndarray,
) -> np.ndarray:
    """Build the requested labeled 2x2 visual comparison."""
    top = np.concatenate(
        [
            make_labeled_panel(lq, "LQ input"),
            make_labeled_panel(sr, "DMUISR-inspired\nDiffusion SR output"),
        ],
        axis=1,
    )
    bottom = np.concatenate(
        [
            make_labeled_panel(hr, "HR target"),
            make_labeled_panel(absolute_error, "Absolute error map"),
        ],
        axis=1,
    )
    return np.concatenate([top, bottom], axis=0)


def sample_diffusion(
    model: torch.nn.Module,
    lq: torch.Tensor,
    schedule: dict,
    sampling_steps: int,
) -> torch.Tensor:
    """Generate SR from Gaussian noise with deterministic reduced-step DDIM sampling."""
    batch_size, _, height, width = lq.shape
    x = torch.randn(
        batch_size,
        1,
        height,
        width,
        device=lq.device,
        dtype=lq.dtype,
    )
    diffusion_steps = int(schedule["betas"].shape[0])
    step_indices = np.linspace(
        0,
        diffusion_steps - 1,
        num=sampling_steps,
        dtype=np.int64,
    )[::-1].copy()

    model.eval()
    with torch.inference_mode():
        for step_index, timestep in enumerate(step_indices):
            timestep = int(timestep)
            t = torch.full(
                (batch_size,),
                timestep,
                device=lq.device,
                dtype=torch.long,
            )
            predicted_noise = model(torch.cat([x, lq], dim=1), t)
            sqrt_alpha = schedule["sqrt_alpha_cumprod"][timestep]
            sqrt_one_minus_alpha = schedule["sqrt_one_minus_alpha_cumprod"][timestep]
            predicted_x0 = (x - sqrt_one_minus_alpha * predicted_noise) / sqrt_alpha

            if step_index + 1 < len(step_indices):
                previous_timestep = int(step_indices[step_index + 1])
                alpha_previous = schedule["alpha_cumprod"][previous_timestep]
            else:
                alpha_previous = torch.ones(
                    (),
                    device=lq.device,
                    dtype=lq.dtype,
                )

            x = (
                torch.sqrt(alpha_previous) * predicted_x0
                + torch.sqrt(1.0 - alpha_previous) * predicted_noise
            )

    return x.clamp(0.0, 1.0)


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DMUISR-inspired diffusion SR on the HC18 test split"
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=50,
        help="Number of reverse diffusion sampling steps (default: 50)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of test images to evaluate (default: all)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Override the checkpoint image size",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="results/models/dmuisr_inspired_diffusion_sr_hc18_best.pth",
        help="Path to the trained DMUISR-inspired checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sampling_steps <= 0:
        raise ValueError("--sampling_steps must be greater than zero")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max_samples must be greater than zero when provided")
    if args.image_size is not None and args.image_size <= 0:
        raise ValueError("--image_size must be greater than zero when provided")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_module = load_training_module(
        SCRIPT_DIR / "16_train_dmuisr_inspired_diffusion_sr_hc18.py"
    )

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")

    image_size = (
        args.image_size
        if args.image_size is not None
        else int(checkpoint.get("image_size", 128))
    )
    timesteps = int(checkpoint.get("timesteps", 1000))
    if args.sampling_steps > timesteps:
        raise ValueError(
            f"--sampling_steps ({args.sampling_steps}) cannot exceed training "
            f"timesteps ({timesteps})"
        )

    model = training_module.DMUISRInspiredUNet(
        base_channels=int(checkpoint.get("base_channels", 64)),
        image_size=image_size,
        use_attention64=bool(checkpoint.get("use_attention64", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    betas = checkpoint.get("betas")
    if betas is None:
        betas = training_module.linear_beta_schedule(timesteps)
    betas = betas.to(device=device, dtype=torch.float32)
    schedule = training_module.prepare_diffusion_schedule(betas)
    schedule = training_module.move_schedule_to_device(schedule, device)

    dataset = HC18DiffusionDataset(
        csv_path="data/processed/HC18_SR/hc18_lq_manifest.csv",
        split="test",
        image_size=image_size,
        input_type="LQ_mixed",
        return_quality=False,
    )
    sample_count = len(dataset)
    if args.max_samples is not None:
        sample_count = min(args.max_samples, sample_count)

    results_root = Path("results")
    sr_dir, visual_dir = choose_output_dirs(results_root)
    sr_dir.mkdir(parents=True, exist_ok=False)
    visual_dir.mkdir(parents=True, exist_ok=False)
    per_image_path = results_root / f"{METHOD_NAME}_hc18_per_image.csv"
    summary_path = results_root / f"{METHOD_NAME}_hc18_summary.csv"

    print(f"Using device: {device}")
    print(f"Using image_size: {image_size}")
    print(f"Evaluating samples: {sample_count}")
    print(f"SR output directory: {sr_dir}")
    print(f"Visual output directory: {visual_dir}")

    metrics: list[dict[str, float]] = []
    with per_image_path.open("w", newline="") as per_image_file:
        writer = csv.writer(per_image_file)
        writer.writerow(
            ["image_id", "psnr", "ssim", "mae", "edge_score", "inference_time"]
        )

        for index in range(sample_count):
            sample = dataset[index]
            image_id = str(sample["image_id"])
            lq = sample["lq"].unsqueeze(0).to(device=device, dtype=torch.float32)
            hr = sample["hr"].unsqueeze(0).to(device=device, dtype=torch.float32)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start_time = time.perf_counter()
            sr = sample_diffusion(model, lq, schedule, args.sampling_steps)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_time = time.perf_counter() - start_time

            lq_2d = to_numpy_image(lq[0])
            hr_2d = to_numpy_image(hr[0])
            sr_2d = to_numpy_image(sr[0])
            absolute_error = np.abs(hr_2d - sr_2d)

            psnr = float(peak_signal_noise_ratio(hr_2d, sr_2d, data_range=1.0))
            ssim = float(structural_similarity(hr_2d, sr_2d, data_range=1.0))
            mae = float(np.mean(absolute_error))
            edge_score = compute_edge_similarity(sr_2d, hr_2d)
            metrics.append(
                {
                    "psnr": psnr,
                    "ssim": ssim,
                    "mae": mae,
                    "edge_score": edge_score,
                    "inference_time": inference_time,
                }
            )

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

            save_grayscale_png(sr_dir / f"{image_id}.png", sr_2d)
            if index < 10:
                visual = build_visual_figure(lq_2d, sr_2d, hr_2d, absolute_error)
                visual_path = visual_dir / f"{image_id}.png"
                if not cv2.imwrite(str(visual_path), visual):
                    raise OSError(f"Failed to save visual comparison: {visual_path}")

    if not metrics:
        raise RuntimeError("No test samples were evaluated")

    psnr_mean, psnr_std = mean_std([item["psnr"] for item in metrics])
    ssim_mean, ssim_std = mean_std([item["ssim"] for item in metrics])
    mae_mean, mae_std = mean_std([item["mae"] for item in metrics])
    edge_mean, edge_std = mean_std([item["edge_score"] for item in metrics])
    time_mean, time_std = mean_std([item["inference_time"] for item in metrics])

    with summary_path.open("w", newline="") as summary_file:
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
        f"{METHOD_NAME}, {psnr_mean:.6f}, {psnr_std:.6f}, "
        f"{ssim_mean:.6f}, {ssim_std:.6f}, {mae_mean:.6f}, {mae_std:.6f}, "
        f"{edge_mean:.6f}, {time_mean:.6f}"
    )


if __name__ == "__main__":
    main()
