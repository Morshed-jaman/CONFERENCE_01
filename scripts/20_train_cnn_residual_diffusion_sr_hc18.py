"""Train CNN-guided residual diffusion super-resolution on HC18.

The quality-aware CNN already supplies a strong SR estimate. Instead of asking
diffusion to generate the complete HR image, this trainer models only the
correction ``HR - CNN_SR``. At inference, the predicted residual is added back
to CNN_SR for a more stable refinement.
"""

import argparse
import csv
import importlib.util
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from hc18_diffusion_dataset import HC18DiffusionDataset


CNN_SR_ROOT = Path("results/sr_outputs/quality_aware_l1_mse")
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def load_baseline_module(script_path: Path):
    """Import the numbered working baseline training script with importlib."""
    spec = importlib.util.spec_from_file_location(
        "train_diffusion_sr_hc18_baseline",
        str(script_path),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load baseline module from: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Reuse the working implementation so activations, time conditioning, and skip
# connections remain identical to scripts/11_train_diffusion_sr_hc18.py.
BASELINE = load_baseline_module(SCRIPT_DIR / "11_train_diffusion_sr_hc18.py")


class CNNResidualConditionalUNet(BASELINE.ConditionalUNet):
    """Baseline ConditionalUNet modified only to accept three input channels."""

    def __init__(
        self,
        base_channels: int = 64,
        image_size: int = 128,
        time_dim: int = 256,
    ):
        super().__init__(
            base_channels=base_channels,
            image_size=image_size,
            time_dim=time_dim,
        )
        # Input is [noisy residual x_t, LQ_mixed, CNN_SR]; output is one noise map.
        self.inc = BASELINE.TimeConditionedBlock(3, base_channels, time_dim)


def find_cnn_sr_image(
    image_id: str,
    files_by_name: dict[str, Path],
    image_files: list[Path],
) -> Path | None:
    """Find an exact PNG/JPEG match first, then a file containing image_id."""
    image_id_lower = image_id.lower()
    for extension in (".png", ".jpg", ".jpeg"):
        exact_match = files_by_name.get(f"{image_id_lower}{extension}")
        if exact_match is not None:
            return exact_match

    fuzzy_matches = [
        path for path in image_files if image_id_lower in path.name.lower()
    ]
    if not fuzzy_matches:
        return None
    if len(fuzzy_matches) > 1:
        warnings.warn(
            f"Multiple CNN SR files contain image_id '{image_id}'; using "
            f"'{fuzzy_matches[0].name}'.",
            RuntimeWarning,
            stacklevel=2,
        )
    return fuzzy_matches[0]


class CNNGuidedResidualDataset(Dataset):
    """Add matched quality-aware CNN outputs to HC18DiffusionDataset samples."""

    def __init__(
        self,
        csv_path: str,
        split: str,
        image_size: int,
        cnn_sr_root: Path = CNN_SR_ROOT,
    ):
        self.split = split
        self.image_size = int(image_size)
        self.cnn_sr_root = Path(cnn_sr_root)
        self.base_dataset = HC18DiffusionDataset(
            csv_path=csv_path,
            split=split,
            image_size=image_size,
            input_type="LQ_mixed",
            return_quality=False,
        )

        if not self.cnn_sr_root.exists():
            raise FileNotFoundError(
                f"CNN SR output directory not found: {self.cnn_sr_root}"
            )

        image_files = sorted(
            (
                path
                for path in self.cnn_sr_root.rglob("*")
                if path.is_file()
                and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            ),
            key=lambda path: str(path).lower(),
        )
        files_by_name: dict[str, Path] = {}
        for path in image_files:
            files_by_name.setdefault(path.name.lower(), path)

        self.samples: list[tuple[int, Path]] = []
        missing_ids: list[str] = []
        for base_index, sample in enumerate(self.base_dataset.samples):
            image_id = str(sample.get("image_id", ""))
            cnn_sr_path = find_cnn_sr_image(image_id, files_by_name, image_files)
            if cnn_sr_path is None:
                missing_ids.append(image_id)
            else:
                self.samples.append((base_index, cnn_sr_path))

        if missing_ids:
            preview = ", ".join(missing_ids[:5])
            if len(missing_ids) > 5:
                preview += ", ..."
            warnings.warn(
                f"Skipping {len(missing_ids)} of {len(self.base_dataset)} "
                f"HC18 '{split}' samples without CNN SR output in "
                f"'{self.cnn_sr_root}'. Missing examples: {preview}",
                RuntimeWarning,
                stacklevel=2,
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        base_index, cnn_sr_path = self.samples[index]
        sample = self.base_dataset[base_index]

        cnn_sr_image = cv2.imread(str(cnn_sr_path), cv2.IMREAD_GRAYSCALE)
        if cnn_sr_image is None:
            raise OSError(f"Could not read CNN SR output: {cnn_sr_path}")
        cnn_sr_image = cv2.resize(
            cnn_sr_image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        cnn_sr = torch.from_numpy(cnn_sr_image.astype(np.float32) / 255.0).unsqueeze(0)

        sample["cnn_sr"] = cnn_sr
        sample["cnn_sr_path"] = str(cnn_sr_path)
        return sample


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    schedule: dict,
    device: torch.device,
    timesteps: int,
) -> float:
    """Train diffusion to predict noise added to the CNN correction residual."""
    model.train()
    total_loss = 0.0
    count = 0

    for batch in dataloader:
        lq = batch["lq"].to(device=device, dtype=torch.float32)
        hr = batch["hr"].to(device=device, dtype=torch.float32)
        cnn_sr = batch["cnn_sr"].to(device=device, dtype=torch.float32)
        batch_size = hr.size(0)

        residual_x0 = hr - cnn_sr
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()
        noise = torch.randn_like(residual_x0)
        x_t = BASELINE.q_sample(residual_x0, t, noise, schedule)

        # CNN-guided conditioning: diffuse only the correction, while both the
        # degraded input and strong CNN estimate guide noise prediction.
        model_input = torch.cat([x_t, lq, cnn_sr], dim=1)
        predicted_noise = model(model_input, t)
        loss = criterion(predicted_noise, noise)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        count += batch_size

    return total_loss / count if count else 0.0


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    schedule: dict,
    device: torch.device,
    timesteps: int,
) -> float:
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            lq = batch["lq"].to(device=device, dtype=torch.float32)
            hr = batch["hr"].to(device=device, dtype=torch.float32)
            cnn_sr = batch["cnn_sr"].to(device=device, dtype=torch.float32)
            batch_size = hr.size(0)

            residual_x0 = hr - cnn_sr
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
            noise = torch.randn_like(residual_x0)
            x_t = BASELINE.q_sample(residual_x0, t, noise, schedule)

            model_input = torch.cat([x_t, lq, cnn_sr], dim=1)
            predicted_noise = model(model_input, t)
            loss = criterion(predicted_noise, noise)

            total_loss += loss.item() * batch_size
            count += batch_size

    return total_loss / count if count else 0.0


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    image_size: int,
    timesteps: int,
    base_channels: int,
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
        "betas": betas.cpu(),
        "target_type": "residual",
        "input_channels": 3,
        "output_channels": 1,
    }
    torch.save(checkpoint, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CNN-guided residual diffusion SR on HC18"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--image_size", type=int, default=128, help="Input image size")
    parser.add_argument("--timesteps", type=int, default=1000, help="DDPM timesteps")
    parser.add_argument("--base_channels", type=int, default=64, help="U-Net base channels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than zero")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be greater than zero")
    if args.lr <= 0:
        raise ValueError("--lr must be greater than zero")
    if args.image_size <= 0:
        raise ValueError("--image_size must be greater than zero")
    if args.timesteps <= 0:
        raise ValueError("--timesteps must be greater than zero")
    if args.base_channels <= 0:
        raise ValueError("--base_channels must be greater than zero")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = CNNGuidedResidualDataset(
        csv_path="data/processed/HC18_SR/hc18_lq_manifest.csv",
        split="train",
        image_size=args.image_size,
    )
    val_dataset = CNNGuidedResidualDataset(
        csv_path="data/processed/HC18_SR/hc18_lq_manifest.csv",
        split="val",
        image_size=args.image_size,
    )

    empty_splits = [
        split_name
        for split_name, dataset in (("train", train_dataset), ("val", val_dataset))
        if len(dataset) == 0
    ]
    if empty_splits:
        raise RuntimeError(
            "No matched CNN SR guides are available for HC18 split(s): "
            f"{', '.join(empty_splits)}. Generate quality-aware L1/MSE CNN "
            "outputs for the train and validation image_ids under "
            f"'{CNN_SR_ROOT}' before training. Test-only guides must not be "
            "used for training."
        )

    print(f"Matched training samples: {len(train_dataset)}")
    print(f"Matched validation samples: {len(val_dataset)}")

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

    model = CNNResidualConditionalUNet(
        base_channels=args.base_channels,
        image_size=args.image_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    betas = BASELINE.linear_beta_schedule(
        args.timesteps,
        beta_start=1e-4,
        beta_end=0.02,
    )
    schedule = BASELINE.prepare_diffusion_schedule(betas)
    schedule = BASELINE.move_schedule_to_device(schedule, device)

    results_root = Path("results")
    model_dir = results_root / "models"
    log_dir = results_root / "training_logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Dedicated names protect all earlier CNN and diffusion checkpoints/logs.
    model_path = model_dir / "cnn_residual_diffusion_sr_hc18_best.pth"
    log_path = log_dir / "cnn_residual_diffusion_sr_hc18_log.csv"
    best_val_loss = float("inf")

    with log_path.open("w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["epoch", "train_loss", "val_loss"])

        print("epoch, train_loss, val_loss")
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                schedule,
                device,
                args.timesteps,
            )
            val_loss = validate(
                model,
                val_loader,
                criterion,
                schedule,
                device,
                args.timesteps,
            )

            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}"])
            log_file.flush()
            print(f"{epoch}, {train_loss:.6f}, {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model_path,
                    model,
                    optimizer,
                    epoch,
                    best_val_loss,
                    args.image_size,
                    args.timesteps,
                    args.base_channels,
                    betas,
                )

    print("Training complete")
    print(f"Best model saved to: {model_path}")
    print(f"Training log saved to: {log_path}")


if __name__ == "__main__":
    main()
