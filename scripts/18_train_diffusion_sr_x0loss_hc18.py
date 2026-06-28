"""Train the working HC18 conditional diffusion SR baseline with x0 loss."""

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure the scripts directory is on sys.path so we can import the shared dataset.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from hc18_diffusion_dataset import HC18DiffusionDataset


def sinusoidal_timestep_embedding(timesteps, dim):
    device = timesteps.device
    half_dim = dim // 2

    freqs = torch.exp(
        -math.log(10000)
        * torch.arange(half_dim, device=device, dtype=torch.float32)
        / max(half_dim - 1, 1)
    )

    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

    if dim % 2 == 1:
        embedding = torch.cat(
            [embedding, torch.zeros(timesteps.shape[0], 1, device=device)],
            dim=1
        )

    return embedding


class TimeConditionedBlock(nn.Module):
    """The baseline's non-inplace time-conditioned convolution block."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.activation(h)
        time_out = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_out
        h = self.conv2(h)
        h = self.activation(h)
        return h


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.block = TimeConditionedBlock(in_channels, out_channels, time_dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.block(x, t_emb)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = TimeConditionedBlock(out_channels + skip_channels, out_channels, time_dim)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x, t_emb)


class ConditionalUNet(nn.Module):
    """Exact working conditional U-Net baseline for noise prediction."""

    def __init__(self, base_channels: int = 64, image_size: int = 128, time_dim: int = 256):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.inc = TimeConditionedBlock(2, base_channels, time_dim)
        self.down1 = DownBlock(base_channels, base_channels * 2, time_dim)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_dim)

        self.mid = TimeConditionedBlock(base_channels * 4, base_channels * 4, time_dim)

        self.up1 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2, time_dim)
        self.up2 = UpBlock(base_channels * 2, base_channels, base_channels, time_dim)

        self.outc = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_timestep_embedding(timesteps, self.time_dim).to(x.device)
        t_emb = self.time_mlp(t_emb)

        x1 = self.inc(x, t_emb)
        x2 = self.down1(x1, t_emb)
        x3 = self.down2(x2, t_emb)

        x_mid = self.mid(x3, t_emb)
        x = self.up1(x_mid, x2, t_emb)
        x = self.up2(x, x1, t_emb)

        return self.outc(x)


def linear_beta_schedule(
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def prepare_diffusion_schedule(betas: torch.Tensor) -> dict:
    device = betas.device
    dtype = betas.dtype

    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    alpha_cumprod_prev = torch.cat(
        [
            torch.ones(1, device=device, dtype=dtype),
            alpha_cumprod[:-1],
        ],
        dim=0,
    )

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_cumprod": alpha_cumprod,
        "alpha_cumprod_prev": alpha_cumprod_prev,
        "sqrt_alpha_cumprod": torch.sqrt(alpha_cumprod),
        "sqrt_one_minus_alpha_cumprod": torch.sqrt(1.0 - alpha_cumprod),
        "sqrt_recip_alphas": torch.sqrt(1.0 / alphas),
        "posterior_variance": betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod),
    }


def move_schedule_to_device(schedule: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in schedule.items()
    }


def q_sample(
    x_start: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
    schedule: dict,
) -> torch.Tensor:
    sqrt_alpha = schedule["sqrt_alpha_cumprod"][t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha = schedule["sqrt_one_minus_alpha_cumprod"][t].view(-1, 1, 1, 1)
    return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise


def reconstruct_x0(
    x_t: torch.Tensor,
    t: torch.Tensor,
    pred_noise: torch.Tensor,
    schedule: dict,
) -> torch.Tensor:
    """Reconstruct x0 from x_t and the model's predicted noise."""
    sqrt_alpha = schedule["sqrt_alpha_cumprod"][t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha = schedule["sqrt_one_minus_alpha_cumprod"][t].view(-1, 1, 1, 1)
    return (x_t - sqrt_one_minus_alpha * pred_noise) / sqrt_alpha


def compute_losses(
    pred_noise: torch.Tensor,
    noise: torch.Tensor,
    pred_x0: torch.Tensor,
    hr: torch.Tensor,
    lambda_x0: float,
) -> Dict[str, torch.Tensor]:
    noise_loss = F.mse_loss(pred_noise, noise)
    x0_loss = F.l1_loss(pred_x0, hr)
    total_loss = noise_loss + lambda_x0 * x0_loss
    return {
        "total_loss": total_loss,
        "noise_loss": noise_loss,
        "x0_loss": x0_loss,
    }


def empty_loss_totals() -> Dict[str, float]:
    return {
        "total_loss": 0.0,
        "noise_loss": 0.0,
        "x0_loss": 0.0,
    }


def average_loss_totals(totals: Dict[str, float], count: int) -> Dict[str, float]:
    if count == 0:
        return {key: 0.0 for key in totals}
    return {key: value / count for key, value in totals.items()}


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    schedule: dict,
    device: torch.device,
    timesteps: int,
    lambda_x0: float,
) -> Dict[str, float]:
    model.train()
    totals = empty_loss_totals()
    count = 0

    for batch in dataloader:
        lq = batch["lq"].to(device)
        hr = batch["hr"].to(device)
        batch_size = hr.size(0)

        t = torch.randint(0, timesteps, (batch_size,), device=device).long()
        noise = torch.randn_like(hr)
        x_t = q_sample(hr, t, noise, schedule)

        model_input = torch.cat([x_t, lq], dim=1)
        pred_noise = model(model_input, t)
        pred_x0 = reconstruct_x0(x_t, t, pred_noise, schedule)
        losses = compute_losses(pred_noise, noise, pred_x0, hr, lambda_x0)

        optimizer.zero_grad(set_to_none=True)
        losses["total_loss"].backward()
        optimizer.step()

        for key in totals:
            totals[key] += losses[key].item() * batch_size
        count += batch_size

    return average_loss_totals(totals, count)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    schedule: dict,
    device: torch.device,
    timesteps: int,
    lambda_x0: float,
) -> Dict[str, float]:
    model.eval()
    totals = empty_loss_totals()
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            lq = batch["lq"].to(device)
            hr = batch["hr"].to(device)
            batch_size = hr.size(0)

            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
            noise = torch.randn_like(hr)
            x_t = q_sample(hr, t, noise, schedule)

            model_input = torch.cat([x_t, lq], dim=1)
            pred_noise = model(model_input, t)
            pred_x0 = reconstruct_x0(x_t, t, pred_noise, schedule)
            losses = compute_losses(pred_noise, noise, pred_x0, hr, lambda_x0)

            for key in totals:
                totals[key] += losses[key].item() * batch_size
            count += batch_size

    return average_loss_totals(totals, count)


def create_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    image_size: int,
    timesteps: int,
    base_channels: int,
    lambda_x0: float,
    betas: torch.Tensor,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "image_size": image_size,
        "timesteps": timesteps,
        "base_channels": base_channels,
        "lambda_x0": lambda_x0,
        "betas": betas.cpu(),
    }
    torch.save(checkpoint, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the HC18 diffusion SR baseline with x0 reconstruction loss"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Optimizer learning rate")
    parser.add_argument("--image_size", type=int, default=128, help="Input image size")
    parser.add_argument("--timesteps", type=int, default=1000, help="DDPM timesteps")
    parser.add_argument("--base_channels", type=int, default=64, help="Base channel count for U-Net")
    parser.add_argument(
        "--lambda_x0",
        type=float,
        default=0.1,
        help="Weight for L1 x0 reconstruction loss",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results_root = Path("results")
    model_dir = results_root / "models"
    log_dir = results_root / "training_logs"
    create_dirs(model_dir, log_dir)

    # These names deliberately keep the working noise-only baseline untouched.
    model_path = model_dir / "diffusion_sr_x0loss_hc18_best.pth"
    log_path = log_dir / "diffusion_sr_x0loss_hc18_log.csv"

    train_dataset = HC18DiffusionDataset(
        csv_path="data/processed/HC18_SR/hc18_lq_manifest.csv",
        split="train",
        image_size=args.image_size,
        input_type="LQ_mixed",
        return_quality=False,
    )
    val_dataset = HC18DiffusionDataset(
        csv_path="data/processed/HC18_SR/hc18_lq_manifest.csv",
        split="val",
        image_size=args.image_size,
        input_type="LQ_mixed",
        return_quality=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = ConditionalUNet(
        base_channels=args.base_channels,
        image_size=args.image_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    betas = linear_beta_schedule(args.timesteps)
    schedule = prepare_diffusion_schedule(betas)
    schedule = move_schedule_to_device(schedule, device)

    best_val_loss = float("inf")
    log_columns = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_noise_loss",
        "train_x0_loss",
        "val_noise_loss",
        "val_x0_loss",
    ]

    with log_path.open("w", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=log_columns)
        writer.writeheader()

        print("epoch, train_loss, val_loss, train_noise_loss, train_x0_loss")
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                schedule,
                device,
                args.timesteps,
                args.lambda_x0,
            )
            val_metrics = validate(
                model,
                val_loader,
                schedule,
                device,
                args.timesteps,
                args.lambda_x0,
            )

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_metrics['total_loss']:.6f}",
                    "val_loss": f"{val_metrics['total_loss']:.6f}",
                    "train_noise_loss": f"{train_metrics['noise_loss']:.6f}",
                    "train_x0_loss": f"{train_metrics['x0_loss']:.6f}",
                    "val_noise_loss": f"{val_metrics['noise_loss']:.6f}",
                    "val_x0_loss": f"{val_metrics['x0_loss']:.6f}",
                }
            )
            log_file.flush()

            print(
                f"{epoch}, {train_metrics['total_loss']:.6f}, "
                f"{val_metrics['total_loss']:.6f}, {train_metrics['noise_loss']:.6f}, "
                f"{train_metrics['x0_loss']:.6f}"
            )

            if val_metrics["total_loss"] < best_val_loss:
                best_val_loss = val_metrics["total_loss"]
                save_checkpoint(
                    model_path,
                    model,
                    optimizer,
                    epoch,
                    best_val_loss,
                    args.image_size,
                    args.timesteps,
                    args.base_channels,
                    args.lambda_x0,
                    betas,
                )

    print("Training complete")
    print(f"Best model saved to: {model_path}")
    print(f"Training log saved to: {log_path}")


if __name__ == "__main__":
    main()
