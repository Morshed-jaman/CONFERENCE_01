"""
Train HC18 SR models with perceptual and adversarial loss.

This script reuses the existing SR-only and quality-aware architectures from
05_train_sr_only_hc18.py and 07_train_quality_aware_sr_hc18.py while adding:
- L1/MSE reconstruction loss
- LPIPS perceptual loss through pyiqa when available
- optional patchGAN adversarial loss
- AMP mixed precision and gradient scaling
- automatic batch-size fallback on CUDA OOM
"""

import argparse
import csv
import importlib.util
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from hc18_dataset import HC18SRDataset
from metrics_utils import get_pyiqa_metric


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_training_module(module_name: str, script_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss that prefers pyiqa LPIPS and keeps image inputs in [0, 1]."""

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.backend = "none"
        self.model: nn.Module | None = None
        self.vgg: nn.Module | None = None

        pyiqa_lpips = get_pyiqa_metric("lpips", device, as_loss=True)
        if pyiqa_lpips is not None:
            self.model = pyiqa_lpips
            if hasattr(self.model, "eval"):
                self.model.eval()
            if hasattr(self.model, "parameters"):
                for param in self.model.parameters():
                    param.requires_grad_(False)
            self.backend = "pyiqa_lpips"
            return

        try:
            from lpips import LPIPS

            self.model = LPIPS(net="vgg").to(device).eval()
            for param in self.model.parameters():
                param.requires_grad_(False)
            self.backend = "lpips_package"
        except Exception:
            try:
                from torchvision.models import vgg16, VGG16_Weights

                vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[:16].to(device).eval()
                for param in vgg.parameters():
                    param.requires_grad_(False)
                self.vgg = vgg
                self.backend = "torchvision_vgg16"
            except Exception:
                self.vgg = nn.Sequential(
                    nn.Conv2d(3, 16, 3, 1, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, 16, 3, 1, 1),
                    nn.ReLU(inplace=True),
                ).to(device)
                for param in self.vgg.parameters():
                    param.requires_grad_(False)
                self.backend = "random_frozen_cnn"

    def _ensure_rgb(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(0)
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return x

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = self._ensure_rgb(pred)
        target = self._ensure_rgb(target)
        pred = pred.to(device=self.device, dtype=torch.float32).clamp(0.0, 1.0)
        target = target.to(device=self.device, dtype=torch.float32).clamp(0.0, 1.0)

        if self.backend == "pyiqa_lpips" and self.model is not None:
            return self.model(pred, target).mean()

        if self.backend == "lpips_package" and self.model is not None:
            return self.model(pred.mul(2.0).sub(1.0), target.mul(2.0).sub(1.0)).mean()

        if self.vgg is not None:
            feat_pred = self.vgg(pred)
            feat_target = self.vgg(target)
            return F.mse_loss(feat_pred, feat_target)

        return torch.zeros((), device=self.device)


class PatchDiscriminator(nn.Module):
    """A lightweight patch-based discriminator for adversarial training."""

    def __init__(self, input_channels: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QualityAwareDataset(Dataset):
    """Wrapper around HC18SRDataset that returns normalized quality features."""

    def __init__(
        self,
        base_dataset: HC18SRDataset,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        compute_quality_features,
    ):
        self.base_dataset = base_dataset
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.compute_quality_features_fn = compute_quality_features

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.base_dataset[index]
        lq = sample["lq"]
        hr = sample["hr"]
        quality_features = self.compute_normalized_quality_features(lq)
        return {"lq": lq, "hr": hr, "quality": quality_features}

    def compute_normalized_quality_features(self, lq: torch.Tensor) -> torch.Tensor:
        single_input = lq.dim() < 4
        features = self.compute_quality_features_fn(lq)
        mean = self.feature_mean.to(device=features.device, dtype=features.dtype).view(1, -1)
        std = self.feature_std.to(device=features.device, dtype=features.dtype).view(1, -1)
        normalized = (features - mean) / std
        if single_input:
            return normalized.view(5)
        return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an HC18 SR model with perceptual and adversarial loss")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Initial batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--input_type", type=str, default="LQ_mixed", choices=["LR_x2", "LR_x4", "LQ_mixed"], help="Input type")
    parser.add_argument("--arch", type=str, default="sr_only", choices=["sr_only", "quality_aware"], help="Architecture variant")
    parser.add_argument("--l1_weight", type=float, default=1.0, help="Weight for L1 loss")
    parser.add_argument("--mse_weight", type=float, default=0.0, help="Weight for MSE loss")
    parser.add_argument("--lpips_weight", type=float, default=0.1, help="Weight for perceptual loss")
    parser.add_argument("--gan_weight", type=float, default=0.05, help="Weight for adversarial loss")
    parser.add_argument("--no_gan", action="store_true", help="Disable adversarial training")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    parser.add_argument("--run_name", type=str, default="", help="Optional checkpoint/log prefix")
    parser.add_argument("--save_dir", type=str, default="results/models", help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="results/training_logs", help="Directory to save logs")
    return parser.parse_args()


def create_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def compute_normalization_stats(dataset: HC18SRDataset) -> Tuple[torch.Tensor, torch.Tensor]:
    qa_module = import_training_module("hc18_quality_aware_train", SCRIPT_DIR / "07_train_quality_aware_sr_hc18.py")
    features_list = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        features = qa_module.compute_quality_features(sample["lq"])
        features_list.append(features.view(-1))
    stacked = torch.stack(features_list, dim=0)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0)
    std = torch.where(std == 0.0, torch.ones_like(std), std)
    return mean, std


def build_dataloaders(args: argparse.Namespace, batch_size: int):
    csv_path = PROJECT_ROOT / "data" / "processed" / "HC18_SR" / "hc18_lq_manifest.csv"
    train_base = HC18SRDataset(csv_path=str(csv_path), input_type=args.input_type, split="train", return_mask=False)
    val_base = HC18SRDataset(csv_path=str(csv_path), input_type=args.input_type, split="val", return_mask=False)

    if args.arch == "quality_aware":
        qa_module = import_training_module("hc18_quality_aware_train", SCRIPT_DIR / "07_train_quality_aware_sr_hc18.py")
        feature_mean, feature_std = compute_normalization_stats(train_base)
        train_dataset = QualityAwareDataset(train_base, feature_mean, feature_std, qa_module.compute_quality_features)
        val_dataset = QualityAwareDataset(val_base, feature_mean, feature_std, qa_module.compute_quality_features)
    else:
        train_dataset = train_base
        val_dataset = val_base

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def build_model(arch: str, device: torch.device, sr_module: Any, qa_module: Any) -> nn.Module:
    if arch == "sr_only":
        return sr_module.ResidualSRModel().to(device)
    return qa_module.QualityAwareSRModel().to(device)


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
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

    metrics: Dict[str, float] = {}
    try:
        metrics["psnr"] = float(peak_signal_noise_ratio(target_np, pred_np, data_range=1.0))
    except Exception:
        metrics["psnr"] = float("nan")
    try:
        metrics["ssim"] = float(structural_similarity(target_np, pred_np, data_range=1.0, win_size=7))
    except Exception:
        metrics["ssim"] = float("nan")
    return metrics


def validate(model: nn.Module, dataloader: DataLoader, device: torch.device, use_quality: bool) -> Tuple[float, float]:
    model.eval()
    psnr_total = 0.0
    ssim_total = 0.0
    count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            lq = batch["lq"].to(device)
            hr = batch["hr"].to(device)
            if use_quality:
                quality = batch["quality"].to(device)
                preds = model(lq, quality)
            else:
                preds = model(lq)

            batch_size = lq.size(0)
            for i in range(batch_size):
                try:
                    metrics = calculate_metrics(preds[i : i + 1], hr[i : i + 1])
                    psnr_total += metrics["psnr"]
                    ssim_total += metrics["ssim"]
                    count += 1
                except Exception:
                    continue

    return (psnr_total / count) if count else 0.0, (ssim_total / count) if count else 0.0


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_quality: bool,
    discriminator: nn.Module | None,
    criterion_l1: nn.Module,
    criterion_mse: nn.Module,
    perceptual_loss: VGGPerceptualLoss | None,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer | None,
    l1_weight: float,
    mse_weight: float,
    lpips_weight: float,
    gan_weight: float,
    use_gan: bool,
    scaler: GradScaler,
) -> Tuple[float, float, float]:
    model.train()
    if discriminator is not None:
        discriminator.train()

    total_loss = 0.0
    total_disc_loss = 0.0
    total_gen_loss = 0.0
    count = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        lq = batch["lq"].to(device)
        hr = batch["hr"].to(device)
        if use_quality:
            quality = batch["quality"].to(device)
        else:
            quality = None

        optimizer_g.zero_grad(set_to_none=True)
        if optimizer_d is not None:
            optimizer_d.zero_grad(set_to_none=True)

        with autocast(enabled=torch.cuda.is_available()):
            if use_quality:
                pred = model(lq, quality)
            else:
                pred = model(lq)

            l1_loss = criterion_l1(pred, hr)
            mse_loss = criterion_mse(pred, hr) if mse_weight > 0.0 else pred.new_zeros(())
            if lpips_weight > 0.0 and perceptual_loss is not None:
                perceptual = perceptual_loss(pred, hr)
                if torch.is_tensor(perceptual):
                    perceptual = perceptual.mean()
            else:
                perceptual = pred.new_zeros(())
            gen_loss = l1_weight * l1_loss + mse_weight * mse_loss + lpips_weight * perceptual

            if use_gan and discriminator is not None:
                real_input = torch.cat([lq, hr], dim=1)
                fake_input = torch.cat([lq, pred.detach()], dim=1)
                real_logits = discriminator(real_input)
                fake_logits = discriminator(fake_input)
                valid = torch.ones_like(real_logits)
                fake = torch.zeros_like(fake_logits)
                d_real_loss = F.binary_cross_entropy_with_logits(real_logits, valid)
                d_fake_loss = F.binary_cross_entropy_with_logits(fake_logits, fake)
                disc_loss = 0.5 * (d_real_loss + d_fake_loss)

                scaler.scale(disc_loss).backward(retain_graph=True)
                scaler.step(optimizer_d)
                scaler.update()
                optimizer_d.zero_grad(set_to_none=True)

                with autocast(enabled=torch.cuda.is_available()):
                    adv_logits = discriminator(torch.cat([lq, pred], dim=1))
                    adv_loss = F.binary_cross_entropy_with_logits(adv_logits, valid)
                    gen_loss = gen_loss + gan_weight * adv_loss

        scaler.scale(gen_loss).backward()
        scaler.step(optimizer_g)
        scaler.update()

        total_loss += gen_loss.item() * lq.size(0)
        total_gen_loss += gen_loss.item() * lq.size(0)
        if use_gan and discriminator is not None:
            total_disc_loss += disc_loss.item() * lq.size(0)
        count += lq.size(0)

    return total_loss / count if count else 0.0, total_gen_loss / count if count else 0.0, total_disc_loss / count if count else 0.0


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)

    save_dir = Path(args.save_dir)
    log_dir = Path(args.log_dir)
    create_dirs(save_dir, log_dir)

    default_model_name = "sr_perceptual" if args.arch == "sr_only" else "qa_sr_perceptual"
    model_name = args.run_name.strip() if args.run_name.strip() else default_model_name
    checkpoint_path = save_dir / f"{model_name}_hc18_best.pth"
    final_checkpoint_path = save_dir / f"{model_name}_hc18_final.pth"
    metadata_path = save_dir / f"{model_name}_hc18_metadata.json"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_csv_path = log_dir / f"{model_name}_{timestamp}.csv"
    log_txt_path = log_dir / f"{model_name}_{timestamp}.log"
    create_dirs(save_dir, log_dir)

    sr_module = import_training_module("hc18_sr_only_train", SCRIPT_DIR / "05_train_sr_only_hc18.py")
    qa_module = import_training_module("hc18_quality_aware_train", SCRIPT_DIR / "07_train_quality_aware_sr_hc18.py")

    current_batch_size = args.batch_size
    attempted_batch_sizes = []

    while True:
        try:
            best_val_psnr = -1.0
            best_state = None
            train_loader, val_loader = build_dataloaders(args, current_batch_size)
            model = build_model(args.arch, device, sr_module, qa_module)
            criterion_l1 = nn.L1Loss()
            criterion_mse = nn.MSELoss()
            perceptual_loss = VGGPerceptualLoss(device) if args.lpips_weight > 0.0 else None
            perceptual_backend = perceptual_loss.backend if perceptual_loss is not None else "disabled"
            optimizer_g = torch.optim.Adam(model.parameters(), lr=args.lr)
            discriminator = None
            optimizer_d = None
            use_gan = (not args.no_gan) and args.gan_weight > 0.0
            if use_gan:
                discriminator = PatchDiscriminator().to(device)
                optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

            with open(log_csv_path, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "epoch",
                        "train_loss",
                        "generator_loss",
                        "discriminator_loss",
                        "val_psnr",
                        "val_ssim",
                        "batch_size",
                        "l1_weight",
                        "mse_weight",
                        "lpips_weight",
                        "gan_weight",
                        "gan_enabled",
                        "perceptual_backend",
                    ]
                )
                for epoch in range(1, args.epochs + 1):
                    train_loss, gen_loss, disc_loss = train_one_epoch(
                        model=model,
                        dataloader=train_loader,
                        device=device,
                        use_quality=(args.arch == "quality_aware"),
                        discriminator=discriminator,
                        criterion_l1=criterion_l1,
                        criterion_mse=criterion_mse,
                        perceptual_loss=perceptual_loss,
                        optimizer_g=optimizer_g,
                        optimizer_d=optimizer_d,
                        l1_weight=args.l1_weight,
                        mse_weight=args.mse_weight,
                        lpips_weight=args.lpips_weight,
                        gan_weight=args.gan_weight,
                        use_gan=use_gan,
                        scaler=scaler,
                    )
                    val_psnr, val_ssim = validate(model, val_loader, device, use_quality=(args.arch == "quality_aware"))
                    writer.writerow(
                        [
                            epoch,
                            f"{train_loss:.6f}",
                            f"{gen_loss:.6f}",
                            f"{disc_loss:.6f}",
                            f"{val_psnr:.4f}",
                            f"{val_ssim:.4f}",
                            current_batch_size,
                            f"{args.l1_weight:.6f}",
                            f"{args.mse_weight:.6f}",
                            f"{args.lpips_weight:.6f}",
                            f"{args.gan_weight:.6f}",
                            int(use_gan),
                            perceptual_backend,
                        ]
                    )
                    handle.flush()

                    if val_psnr > best_val_psnr:
                        best_val_psnr = val_psnr
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                        torch.save(best_state, checkpoint_path)

                    print(
                        f"Epoch {epoch:02d}/{args.epochs:02d} "
                        f"train_loss={train_loss:.6f} "
                        f"d_loss={disc_loss:.6f} "
                        f"val_psnr={val_psnr:.4f} "
                        f"val_ssim={val_ssim:.4f}"
                    )

            if best_state is not None:
                torch.save(best_state, checkpoint_path)
            final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(final_state, final_checkpoint_path)

            with open(log_txt_path, "w", encoding="utf-8") as handle:
                handle.write(
                    f"model={model_name}\n"
                    f"arch={args.arch}\n"
                    f"batch_size={current_batch_size}\n"
                    f"epochs={args.epochs}\n"
                    f"best_val_psnr={best_val_psnr:.4f}\n"
                    f"checkpoint={checkpoint_path}\n"
                    f"final_checkpoint={final_checkpoint_path}\n"
                    f"l1_weight={args.l1_weight}\n"
                    f"mse_weight={args.mse_weight}\n"
                    f"lpips_weight={args.lpips_weight}\n"
                    f"gan_weight={args.gan_weight}\n"
                    f"gan_enabled={use_gan}\n"
                    f"perceptual_backend={perceptual_backend}\n"
                )

            metadata = {
                "model": model_name,
                "arch": args.arch,
                "batch_size": current_batch_size,
                "epochs": args.epochs,
                "best_val_psnr": best_val_psnr,
                "best_checkpoint": str(checkpoint_path),
                "final_checkpoint": str(final_checkpoint_path),
                "log_csv": str(log_csv_path),
                "log_txt": str(log_txt_path),
                "l1_weight": args.l1_weight,
                "mse_weight": args.mse_weight,
                "lpips_weight": args.lpips_weight,
                "gan_weight": args.gan_weight,
                "gan_enabled": use_gan,
                "perceptual_backend": perceptual_backend,
                "seed": args.seed,
                "input_type": args.input_type,
            }
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            print("\nTraining complete")
            print(f"Best checkpoint saved to: {checkpoint_path}")
            print(f"Final checkpoint saved to: {final_checkpoint_path}")
            print(f"Training log saved to: {log_csv_path}")
            print(f"Summary: perceptual-loss training added for {args.arch}; best val PSNR={best_val_psnr:.4f}")
            return
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if current_batch_size <= 2:
                raise
            attempted_batch_sizes.append(current_batch_size)
            current_batch_size = max(2, current_batch_size // 2)
            print(f"CUDA OOM detected; reducing batch size from {current_batch_size * 2} to {current_batch_size}")
            if current_batch_size in attempted_batch_sizes:
                break


if __name__ == "__main__":
    main()
