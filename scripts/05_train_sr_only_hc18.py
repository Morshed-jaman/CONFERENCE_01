"""
Train a lightweight super-resolution model on HC18 LQ images.

This script trains a small residual CNN model using LQ_mixed input and HR targets.
It saves the best model by validation PSNR and keeps a CSV training log.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train HC18 SR model")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--input_type",
        type=str,
        default="LQ_mixed",
        choices=["LR_x2", "LR_x4", "LQ_mixed"],
        help="Input type for the model",
    )
    return parser.parse_args()


def create_dirs(model_dir: Path, log_dir: Path):
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
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
    return psnr, ssim


def validate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    sum_psnr = 0.0
    sum_ssim = 0.0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            lq = batch["lq"].to(device)
            hr = batch["hr"].to(device)

            pred = model(lq)
            batch_size = lq.size(0)

            for i in range(batch_size):
                psnr, ssim = calculate_metrics(pred[i : i + 1], hr[i : i + 1])
                sum_psnr += psnr
                sum_ssim += ssim
                count += 1

    return (sum_psnr / count) if count else 0.0, (sum_ssim / count) if count else 0.0


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    count = 0

    for batch in dataloader:
        lq = batch["lq"].to(device)
        hr = batch["hr"].to(device)

        optimizer.zero_grad()
        pred = model(lq)
        loss = criterion(pred, hr)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * lq.size(0)
        count += lq.size(0)

    return total_loss / count if count else 0.0


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_root = Path("results")
    model_dir = results_root / "models"
    log_dir = results_root / "training_logs"
    create_dirs(model_dir, log_dir)

    model_path = model_dir / "sr_only_hc18_best.pth"
    log_path = log_dir / "sr_only_hc18_log.csv"

    train_dataset = HC18SRDataset(
        csv_path="data/processed/HC18_SR/hc18_lq_manifest.csv",
        input_type=args.input_type,
        split="train",
        return_mask=False,
    )
    val_dataset = HC18SRDataset(
        csv_path="data/processed/HC18_SR/hc18_lq_manifest.csv",
        input_type=args.input_type,
        split="val",
        return_mask=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ResidualSRModel().to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_psnr = -1.0
    header_written = False

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_psnr", "val_ssim"])

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_psnr, val_ssim = validate(model, val_loader, device)

            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_psnr:.4f}", f"{val_ssim:.4f}"])
            f.flush()

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(model.state_dict(), model_path)

            print(
                f"Epoch {epoch:02d}/{args.epochs:02d} "
                f"train_loss={train_loss:.6f} "
                f"val_psnr={val_psnr:.4f} "
                f"val_ssim={val_ssim:.4f}"
            )

    print("\nTraining complete")
    print(f"Best model saved to: {model_path}")
    print(f"Training log saved to: {log_path}")


if __name__ == "__main__":
    main()
