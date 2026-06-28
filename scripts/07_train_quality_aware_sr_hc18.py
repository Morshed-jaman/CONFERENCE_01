"""
Train a quality-aware super-resolution model on HC18.

This script trains a lightweight FiLM-conditioned residual CNN using
image-level quality degradation features extracted from LQ_mixed images.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader, Dataset

# Ensure the scripts directory is on sys.path so we can import hc18_dataset.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from hc18_dataset import HC18SRDataset


class QualityAwareDataset(Dataset):
    """Wrap HC18SRDataset with quality feature extraction."""

    def __init__(
        self,
        base_dataset: HC18SRDataset,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
    ):
        self.base_dataset = base_dataset
        self.feature_mean = feature_mean
        self.feature_std = feature_std

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        sample = self.base_dataset[index]
        lq = sample["lq"]
        hr = sample["hr"]
        quality = self.compute_normalized_quality_features(lq)
        return {
            "lq": lq,
            "hr": hr,
            "quality": quality,
        }

    def compute_normalized_quality_features(self, lq: torch.Tensor) -> torch.Tensor:
        single_input = lq.dim() < 4
        features = compute_quality_features(lq)
        mean = self.feature_mean.to(device=features.device, dtype=features.dtype).view(1, -1)
        std = self.feature_std.to(device=features.device, dtype=features.dtype).view(1, -1)
        normalized = (features - mean) / std
        if single_input:
            normalized = normalized.view(5)
        return normalized


class QualityAwareSRModel(nn.Module):
    """Lightweight FiLM-conditioned residual CNN for SR."""

    def __init__(self, num_channels: int = 64, quality_dim: int = 5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(num_channels, 1, kernel_size=3, padding=1)

        self.film_mlp = nn.Sequential(
            nn.Linear(quality_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_channels * 2),
        )

    def forward(self, x: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
        features = F.relu(self.conv1(x))
        features = F.relu(self.conv2(features))

        gamma_beta = self.film_mlp(quality)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        features = gamma * features + beta

        features = F.relu(self.conv3(features))
        features = F.relu(self.conv4(features))
        residual = self.conv_out(features)
        out = x + residual
        out = torch.clamp(out, 0.0, 1.0)
        return out


def parse_args():
    parser = argparse.ArgumentParser(description="Train quality-aware SR model on HC18")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--input_type",
        type=str,
        default="LQ_mixed",
        choices=["LR_x2", "LR_x4", "LQ_mixed"],
        help="Input type for the model",
    )
    return parser.parse_args()


def create_dirs(*dirs: Path) -> None:
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def gaussian_blur(image: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """Apply a Gaussian blur to a single-channel image tensor."""
    device = image.device
    axis = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2
    kernel_1d = torch.exp(-(axis ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel = kernel_2d.unsqueeze(0).unsqueeze(0)
    padding = kernel_size // 2
    return F.conv2d(image, kernel, padding=padding)


def compute_quality_features(lq: torch.Tensor) -> torch.Tensor:
    """Compute a 5-dimensional quality feature tensor from an LQ image batch."""
    if lq.dim() == 2:
        lq = lq.unsqueeze(0).unsqueeze(0)
    elif lq.dim() == 3:
        lq = lq.unsqueeze(0)

    if lq.dim() != 4 or lq.size(1) != 1:
        raise ValueError("Expected LQ tensor shape [H, W], [1, H, W], or [B, 1, H, W]")

    lq = lq.float()

    lap_kernel = torch.tensor(
        [[0, 1, 0],
         [1, -4, 1],
         [0, 1, 0]],
        dtype=lq.dtype,
        device=lq.device,
    ).view(1, 1, 3, 3)

    lap = F.conv2d(lq, lap_kernel, padding=1)

    blur_score = lap.var(dim=(1, 2, 3)).view(-1)
    std_value = lq.std(dim=(1, 2, 3)).view(-1)
    mean_value = lq.mean(dim=(1, 2, 3)).view(-1)

    blurred = F.avg_pool2d(lq, kernel_size=5, stride=1, padding=2)
    noise_proxy = torch.abs(lq - blurred).mean(dim=(1, 2, 3)).view(-1)

    degradation_strength = torch.ones_like(mean_value).view(-1)

    features = torch.stack(
        [blur_score, std_value, mean_value, noise_proxy, degradation_strength],
        dim=1,
    )

    assert features.dim() == 2 and features.shape[1] == 5

    return features


def compute_normalization_stats(dataset: HC18SRDataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute mean and std for quality feature normalization over a dataset."""
    features_list: List[torch.Tensor] = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        quality = compute_quality_features(sample["lq"])
        quality = quality.view(-1)
        if quality.shape[0] != 5:
            raise ValueError(f"Expected quality feature vector shape [5], got {tuple(quality.shape)}")
        features_list.append(quality)
    stacked = torch.stack(features_list, dim=0)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0)
    std = torch.where(std == 0.0, torch.ones_like(std), std)
    return mean, std


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    """Calculate PSNR and SSIM for a prediction and target pair."""
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


def validate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            lq = batch["lq"].to(device)
            hr = batch["hr"].to(device)
            quality = batch["quality"].to(device)
            pred = model(lq, quality)
            batch_size = lq.size(0)
            for i in range(batch_size):
                psnr, ssim = calculate_metrics(pred[i : i + 1], hr[i : i + 1])
                total_psnr += psnr
                total_ssim += ssim
                count += 1
    return (total_psnr / count) if count else 0.0, (total_ssim / count) if count else 0.0


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    l1_criterion: nn.Module,
    mse_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    count = 0
    for batch in dataloader:
        lq = batch["lq"].to(device)
        hr = batch["hr"].to(device)
        quality = batch["quality"].to(device)
        optimizer.zero_grad()
        pred = model(lq, quality)
        l1_loss = l1_criterion(pred, hr)
        mse_loss = mse_criterion(pred, hr)
        loss = 0.8 * l1_loss + 0.2 * mse_loss
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

    model_path = model_dir / "quality_aware_sr_hc18_l1_mse_best.pth"
    log_path = log_dir / "quality_aware_sr_hc18_l1_mse_log.csv"

    base_train = HC18SRDataset(
        csv_path="data/processed/HC18_SR/hc18_lq_manifest.csv",
        input_type=args.input_type,
        split="train",
        return_mask=False,
    )
    base_val = HC18SRDataset(
        csv_path="data/processed/HC18_SR/hc18_lq_manifest.csv",
        input_type=args.input_type,
        split="val",
        return_mask=False,
    )

    feature_mean, feature_std = compute_normalization_stats(base_train)
    train_dataset = QualityAwareDataset(base_train, feature_mean, feature_std)
    val_dataset = QualityAwareDataset(base_val, feature_mean, feature_std)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = QualityAwareSRModel().to(device)
    l1_criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_psnr = -1.0
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_psnr", "val_ssim"])

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model,
                train_loader,
                l1_criterion,
                mse_criterion,
                optimizer,
                device,
            )
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
