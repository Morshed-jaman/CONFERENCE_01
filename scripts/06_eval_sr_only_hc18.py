"""
Evaluate the trained SR-only model on HC18 test data.

This script loads the best saved SR-only model and evaluates it on the HC18 test split.
It saves super-resolved images, per-image metrics, summary results, and visual comparison figures.
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader

# Ensure the scripts directory is on sys.path so we can import hc18_dataset.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from hc18_dataset import HC18SRDataset


class ResidualSRModel(nn.Module):
    """A small residual CNN for super-resolution."""

    def __init__(self, num_channels: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(1, num_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.body = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.tail = nn.Conv2d(num_channels, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.head(x)
        residual = self.body(residual)
        residual = self.tail(residual)
        out = x + residual
        out = torch.clamp(out, 0.0, 1.0)
        return out


def create_dirs(*dirs: Path):
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def image_to_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert a [1,H,W] tensor to a grayscale uint8 image."""
    array = image_tensor.detach().cpu().numpy()
    array = np.squeeze(array)
    array = np.clip(array, 0.0, 1.0)
    array = (array * 255.0).round().astype(np.uint8)
    return array


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float]:
    """Compute PSNR, SSIM, and MAE for a single prediction-target pair."""
    pred_np = pred.detach().cpu().float().numpy()
    target_np = target.detach().cpu().float().numpy()

    pred_np = np.squeeze(pred_np)
    target_np = np.squeeze(target_np)

    if pred_np.ndim == 3:
        pred_np = pred_np[0]
    if target_np.ndim == 3:
        target_np = target_np[0]

    pred_np = np.clip(pred_np, 0.0, 1.0)
    target_np = np.clip(target_np, 0.0, 1.0)

    psnr = peak_signal_noise_ratio(target_np, pred_np, data_range=1.0)
    ssim = structural_similarity(target_np, pred_np, data_range=1.0, win_size=7)
    mae = float(np.mean(np.abs(target_np - pred_np)))
    return psnr, ssim, mae


def save_image(array: np.ndarray, output_path: Path) -> None:
    """Save a 2D uint8 image to disk using PIL."""
    Image.fromarray(array).save(output_path)


def create_visual_figure(
    lq_img: np.ndarray,
    sr_img: np.ndarray,
    hr_img: np.ndarray,
    error_img: np.ndarray,
    output_path: Path,
    image_id: str,
) -> None:
    """Save a visual comparison figure showing LQ, SR, HR, and error."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"SR Comparison: {image_id}", fontsize=14)

    axes[0].imshow(lq_img, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("LQ input")
    axes[0].axis("off")

    axes[1].imshow(sr_img, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("SR output")
    axes[1].axis("off")

    axes[2].imshow(hr_img, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("HR target")
    axes[2].axis("off")

    axes[3].imshow(error_img, cmap="inferno", vmin=0, vmax=255)
    axes[3].set_title("Absolute error")
    axes[3].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    data_csv = Path("data/processed/HC18_SR/hc18_lq_manifest.csv")
    model_path = Path("results/models/sr_only_hc18_best.pth")
    sr_output_dir = Path("results/sr_outputs/sr_only")
    visual_dir = Path("results/visual_sr_only")
    per_image_csv = Path("results/sr_only_hc18_per_image.csv")
    summary_csv = Path("results/sr_only_hc18_summary.csv")

    create_dirs(sr_output_dir, visual_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = HC18SRDataset(
        csv_path=str(data_csv),
        input_type="LQ_mixed",
        split="test",
        return_mask=False,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = ResidualSRModel().to(device)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    per_image_results = []
    all_psnr = []
    all_ssim = []
    all_mae = []

    for idx, sample in enumerate(dataloader):
        image_id = sample["image_id"][0]
        lq = sample["lq"].to(device)
        hr = sample["hr"].to(device)

        with torch.no_grad():
            sr = model(lq)

        if lq.shape != hr.shape:
            raise ValueError(f"Shape mismatch for {image_id}: LQ {lq.shape} vs HR {hr.shape}")

        psnr, ssim, mae = calculate_metrics(sr, hr)
        all_psnr.append(psnr)
        all_ssim.append(ssim)
        all_mae.append(mae)

        per_image_results.append(
            {
                "image_id": image_id,
                "psnr": psnr,
                "ssim": ssim,
                "mae": mae,
            }
        )

        sr_uint8 = image_to_uint8(sr[0])
        sr_output_path = sr_output_dir / f"{image_id}.png"
        save_image(sr_uint8, sr_output_path)

        if idx < 10:
            lq_uint8 = image_to_uint8(lq[0])
            hr_uint8 = image_to_uint8(hr[0])
            error = np.abs(hr_uint8.astype(np.int16) - sr_uint8.astype(np.int16)).astype(np.uint8)
            visual_path = visual_dir / f"{image_id}.png"
            create_visual_figure(lq_uint8, sr_uint8, hr_uint8, error, visual_path, image_id)

    # Save per-image metrics
    with open(per_image_csv, "w", newline="") as f:
        fieldnames = ["image_id", "psnr", "ssim", "mae"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image_results)

    # Compute summary metrics
    summary = [
        {
            "method": "sr_only",
            "mean_psnr": float(np.mean(all_psnr)),
            "std_psnr": float(np.std(all_psnr)),
            "mean_ssim": float(np.mean(all_ssim)),
            "std_ssim": float(np.std(all_ssim)),
            "mean_mae": float(np.mean(all_mae)),
            "std_mae": float(np.std(all_mae)),
        }
    ]

    with open(summary_csv, "w", newline="") as f:
        fieldnames = [
            "method",
            "mean_psnr",
            "std_psnr",
            "mean_ssim",
            "std_ssim",
            "mean_mae",
            "std_mae",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    print("\nSR-only evaluation summary")
    print("method, mean_psnr, std_psnr, mean_ssim, std_ssim, mean_mae, std_mae")
    for row in summary:
        print(
            f"{row['method']}, "
            f"{row['mean_psnr']:.4f}, "
            f"{row['std_psnr']:.4f}, "
            f"{row['mean_ssim']:.4f}, "
            f"{row['std_ssim']:.4f}, "
            f"{row['mean_mae']:.4f}, "
            f"{row['std_mae']:.4f}"
        )
    print(f"\nSaved SR images to: {sr_output_dir}")
    print(f"Saved visual figures to: {visual_dir}")
    print(f"Saved per-image metrics to: {per_image_csv}")
    print(f"Saved summary metrics to: {summary_csv}")


if __name__ == "__main__":
    main()
