"""Train a practical DMUISR-inspired conditional diffusion SR model on HC18.

This is not an exact reproduction of DMUISR. It keeps the project's working
conditional DDPM training pipeline and adds ultrasound-oriented, wavelet-like
LQ detail conditioning, multi-layer spatial self-attention, x0 reconstruction
loss, and Sobel edge loss.
"""

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

# Allow the script to import the shared HC18 dataset when run from project root.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from hc18_diffusion_dataset import HC18DiffusionDataset


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Create device-safe sinusoidal timestep embeddings."""
    device = timesteps.device
    half_dim = dim // 2
    frequencies = torch.exp(
        -math.log(10000)
        * torch.arange(half_dim, device=device, dtype=torch.float32)
        / max(half_dim - 1, 1)
    )
    arguments = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.sin(arguments), torch.cos(arguments)], dim=1)

    if dim % 2 == 1:
        padding = torch.zeros(
            timesteps.shape[0],
            1,
            device=device,
            dtype=embedding.dtype,
        )
        embedding = torch.cat([embedding, padding], dim=1)

    return embedding


def group_count(channels: int) -> int:
    """Choose the largest valid GroupNorm group count up to 32."""
    return max(group for group in range(1, min(32, channels) + 1) if channels % group == 0)


class TimeConditionedBlock(nn.Module):
    """The working non-inplace convolution block from the baseline U-Net."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.activation(h)
        time_projection = self.time_mlp(time_embedding).unsqueeze(-1).unsqueeze(-1)
        h = h + time_projection
        h = self.conv2(h)
        h = self.activation(h)
        return h


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.block = TimeConditionedBlock(in_channels, out_channels, time_dim)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        h = self.pool(x)
        return self.block(h, time_embedding)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = TimeConditionedBlock(out_channels + skip_channels, out_channels, time_dim)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        h = self.up(x)
        if h.shape[-2:] != skip.shape[-2:]:
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        h = torch.cat([h, skip], dim=1)
        return self.block(h, time_embedding)


class SelfAttention2d(nn.Module):
    """Lightweight residual self-attention over flattened spatial tokens."""

    def __init__(self, channels: int):
        super().__init__()
        attention_channels = max(channels // 8, 1)
        self.scale = attention_channels ** -0.5
        self.norm = nn.GroupNorm(group_count(channels), channels)
        self.query = nn.Conv2d(channels, attention_channels, kernel_size=1)
        self.key = nn.Conv2d(channels, attention_channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.output = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        normalized = self.norm(x)

        query = self.query(normalized).flatten(2).transpose(1, 2)
        key = self.key(normalized).flatten(2)
        value = self.value(normalized).flatten(2)

        attention = torch.bmm(query, key) * self.scale
        attention = torch.softmax(attention, dim=-1)
        attended = torch.bmm(value, attention.transpose(1, 2))
        attended = attended.reshape(batch_size, channels, height, width)
        attention_output = self.output(attended)
        return x + attention_output


class WaveletLikeLQEncoder(nn.Module):
    """Approximate Haar detail conditioning using an LQ high-pass residual."""

    def __init__(self, base_channels: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
        )

    def forward(self, lq: torch.Tensor) -> torch.Tensor:
        blur = F.avg_pool2d(lq, kernel_size=3, stride=1, padding=1)
        high_frequency = lq - blur
        wavelet_like_input = torch.cat([lq, high_frequency], dim=1)
        return self.encoder(wavelet_like_input)


def resize_feature(feature: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if feature.shape[-2:] != reference.shape[-2:]:
        return F.interpolate(
            feature,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return feature


class DMUISRInspiredUNet(nn.Module):
    """Conditional U-Net with practical DMUISR-inspired detail and attention blocks."""

    def __init__(
        self,
        base_channels: int = 64,
        image_size: int = 128,
        time_dim: int = 256,
        use_attention64: bool = False,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.image_size = image_size
        self.use_attention64 = bool(use_attention64)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.wavelet_encoder = WaveletLikeLQEncoder(base_channels)
        self.wavelet_to_64 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=1)
        self.wavelet_to_32 = nn.Conv2d(base_channels, base_channels * 4, kernel_size=1)

        self.inc = TimeConditionedBlock(2, base_channels, time_dim)
        self.down1 = DownBlock(base_channels, base_channels * 2, time_dim)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_dim)

        self.attention64 = SelfAttention2d(base_channels * 2) if self.use_attention64 else nn.Identity()
        self.mid = TimeConditionedBlock(base_channels * 4, base_channels * 4, time_dim)
        # Two residual blocks provide multi-layer attention at the memory-safe 32x32 bottleneck.
        self.attention32 = nn.Sequential(
            SelfAttention2d(base_channels * 4),
            SelfAttention2d(base_channels * 4),
        )

        self.up1 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2, time_dim)
        self.up2 = UpBlock(base_channels * 2, base_channels, base_channels, time_dim)
        self.outc = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_embedding = sinusoidal_timestep_embedding(timesteps, self.time_dim).to(x.device)
        time_embedding = self.time_mlp(time_embedding)

        # Channel one is the LQ_mixed condition in [x_t, lq].
        lq = x[:, 1:2]
        wavelet_features = self.wavelet_encoder(lq)

        x1 = self.inc(x, time_embedding)
        x1 = x1 + resize_feature(wavelet_features, x1)

        x2 = self.down1(x1, time_embedding)
        wavelet_64 = resize_feature(wavelet_features, x2)
        x2 = x2 + self.wavelet_to_64(wavelet_64)
        x2 = self.attention64(x2)

        x3 = self.down2(x2, time_embedding)
        wavelet_32 = resize_feature(wavelet_features, x3)
        x3 = x3 + self.wavelet_to_32(wavelet_32)

        x_mid = self.mid(x3, time_embedding)
        x_mid = self.attention32(x_mid)

        x = self.up1(x_mid, x2, time_embedding)
        x = self.up2(x, x1, time_embedding)
        return self.outc(x)


def linear_beta_schedule(
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def prepare_diffusion_schedule(betas: torch.Tensor) -> dict:
    """Build the working DDPM schedule on the same device and dtype as betas."""
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
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    schedule: dict,
) -> torch.Tensor:
    sqrt_alpha = schedule["sqrt_alpha_cumprod"][timesteps].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha = schedule["sqrt_one_minus_alpha_cumprod"][timesteps].view(-1, 1, 1, 1)
    return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise


def reconstruct_x0(
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    predicted_noise: torch.Tensor,
    schedule: dict,
) -> torch.Tensor:
    sqrt_alpha = schedule["sqrt_alpha_cumprod"][timesteps].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha = schedule["sqrt_one_minus_alpha_cumprod"][timesteps].view(-1, 1, 1, 1)
    return (x_t - sqrt_one_minus_alpha * predicted_noise) / sqrt_alpha


def sobel_edges(image: torch.Tensor) -> torch.Tensor:
    """Compute differentiable Sobel edge magnitudes on the input tensor's device."""
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3)
    grad_x = F.conv2d(image, sobel_x, padding=1)
    grad_y = F.conv2d(image, sobel_y, padding=1)
    return torch.sqrt(grad_x.square() + grad_y.square() + 1e-12)


def compute_losses(
    predicted_noise: torch.Tensor,
    noise: torch.Tensor,
    x_t: torch.Tensor,
    hr: torch.Tensor,
    timesteps: torch.Tensor,
    schedule: dict,
    lambda_x0: float,
    lambda_edge: float,
) -> Dict[str, torch.Tensor]:
    predicted_x0 = reconstruct_x0(x_t, timesteps, predicted_noise, schedule)
    noise_loss = F.mse_loss(predicted_noise, noise)
    x0_loss = F.l1_loss(predicted_x0, hr)
    edge_loss = F.l1_loss(sobel_edges(predicted_x0), sobel_edges(hr))
    total_loss = noise_loss + lambda_x0 * x0_loss + lambda_edge * edge_loss
    return {
        "total_loss": total_loss,
        "noise_loss": noise_loss,
        "x0_loss": x0_loss,
        "edge_loss": edge_loss,
    }


def empty_loss_totals() -> Dict[str, float]:
    return {
        "total_loss": 0.0,
        "noise_loss": 0.0,
        "x0_loss": 0.0,
        "edge_loss": 0.0,
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
    lambda_edge: float,
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
        predicted_noise = model(model_input, t)
        losses = compute_losses(
            predicted_noise,
            noise,
            x_t,
            hr,
            t,
            schedule,
            lambda_x0,
            lambda_edge,
        )

        optimizer.zero_grad(set_to_none=True)
        losses["total_loss"].backward()
        optimizer.step()

        for key in totals:
            totals[key] = totals[key] + losses[key].item() * batch_size
        count = count + batch_size

    return average_loss_totals(totals, count)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    schedule: dict,
    device: torch.device,
    timesteps: int,
    lambda_x0: float,
    lambda_edge: float,
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
            predicted_noise = model(model_input, t)
            losses = compute_losses(
                predicted_noise,
                noise,
                x_t,
                hr,
                t,
                schedule,
                lambda_x0,
                lambda_edge,
            )

            for key in totals:
                totals[key] = totals[key] + losses[key].item() * batch_size
            count = count + batch_size

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
    lambda_edge: float,
    use_attention64: bool,
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
        "lambda_edge": lambda_edge,
        "use_attention64": use_attention64,
        "betas": betas.cpu(),
    }
    torch.save(checkpoint, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train practical DMUISR-inspired diffusion SR on HC18")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Optimizer learning rate")
    parser.add_argument("--image_size", type=int, default=128, help="Input image size")
    parser.add_argument("--timesteps", type=int, default=1000, help="DDPM timesteps")
    parser.add_argument("--base_channels", type=int, default=64, help="Base channel count")
    parser.add_argument("--lambda_x0", type=float, default=0.1, help="x0 reconstruction loss weight")
    parser.add_argument("--lambda_edge", type=float, default=0.05, help="Sobel edge loss weight")
    parser.add_argument(
        "--use_attention64",
        action="store_true",
        help="Enable optional self-attention at the 64x64 encoder feature",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using attention at 64x64: {args.use_attention64}")

    results_root = Path("results")
    model_dir = results_root / "models"
    log_dir = results_root / "training_logs"
    create_dirs(model_dir, log_dir)

    model_path = model_dir / "dmuisr_inspired_diffusion_sr_hc18_best.pth"
    log_path = log_dir / "dmuisr_inspired_diffusion_sr_hc18_log.csv"

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

    model = DMUISRInspiredUNet(
        base_channels=args.base_channels,
        image_size=args.image_size,
        use_attention64=args.use_attention64,
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
        "train_edge_loss",
        "val_noise_loss",
        "val_x0_loss",
        "val_edge_loss",
    ]

    with open(log_path, "w", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=log_columns)
        writer.writeheader()

        print("epoch, train_loss, val_loss, train_noise_loss, train_x0_loss, train_edge_loss")
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                schedule,
                device,
                args.timesteps,
                args.lambda_x0,
                args.lambda_edge,
            )
            val_metrics = validate(
                model,
                val_loader,
                schedule,
                device,
                args.timesteps,
                args.lambda_x0,
                args.lambda_edge,
            )

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_metrics['total_loss']:.6f}",
                    "val_loss": f"{val_metrics['total_loss']:.6f}",
                    "train_noise_loss": f"{train_metrics['noise_loss']:.6f}",
                    "train_x0_loss": f"{train_metrics['x0_loss']:.6f}",
                    "train_edge_loss": f"{train_metrics['edge_loss']:.6f}",
                    "val_noise_loss": f"{val_metrics['noise_loss']:.6f}",
                    "val_x0_loss": f"{val_metrics['x0_loss']:.6f}",
                    "val_edge_loss": f"{val_metrics['edge_loss']:.6f}",
                }
            )
            log_file.flush()

            print(
                f"{epoch}, {train_metrics['total_loss']:.6f}, {val_metrics['total_loss']:.6f}, "
                f"{train_metrics['noise_loss']:.6f}, {train_metrics['x0_loss']:.6f}, "
                f"{train_metrics['edge_loss']:.6f}"
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
                    args.lambda_edge,
                    args.use_attention64,
                    betas,
                )

    print("Training complete")
    print(f"Best model saved to: {model_path}")
    print(f"Training log saved to: {log_path}")


if __name__ == "__main__":
    main()
