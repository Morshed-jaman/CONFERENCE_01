r"""Phase A: train SwinIR-Light on HC18 with L1 + LPIPS + GAN loss.

PowerShell:

    python .\scripts\50_train_swinir_hc18.py --epochs 100 --batch_size 8

The script is additive and writes a complete run directory with:
best_lpips.pth, best_dice.pth, last.pth, ema.pth, metadata.json, CSV logs,
test-set reconstruction metrics, downstream Dice metrics, and SR outputs.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from hc18_dataset import HC18SRDataset
from metrics_utils import (
    PyiqaMetricBundle,
    compute_brisque,
    compute_edge_score,
    compute_lpips,
    compute_mae,
    compute_niqe,
    compute_psnr,
    compute_ssim,
    get_pyiqa_metric,
)
from models.swinir_light import SwinIRLight


MANIFEST = PROJECT_ROOT / "data" / "processed" / "HC18_SR" / "hc18_lq_manifest.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_ROOT = RESULTS_DIR / "models"
LOG_ROOT = RESULTS_DIR / "training_logs"
SR_OUTPUT_ROOT = RESULTS_DIR / "sr_outputs"
MASK_OUTPUT_ROOT = RESULTS_DIR / "downstream_utility_masks"
CURRENT_RECON_SUMMARY = RESULTS_DIR / "final_all_hc18_comparison_summary.csv"
CURRENT_DOWNSTREAM_SUMMARY = RESULTS_DIR / "downstream_utility_summary.csv"


class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def parse_depths(value: str) -> tuple[int, ...]:
    try:
        depths = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--depths must be comma-separated integers") from exc
    if not depths:
        raise argparse.ArgumentTypeError("--depths must contain at least one integer")
    return depths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SwinIR-Light on HC18 with perceptual and adversarial loss."
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Initial batch size; falls back on OOM")
    parser.add_argument("--lr", type=float, default=2e-4, help="Initial generator learning rate")
    parser.add_argument("--d_lr", type=float, default=1e-4, help="Discriminator learning rate")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Cosine LR warmup epochs")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    parser.add_argument("--input_type", type=str, default="LQ_mixed", choices=["LR_x2", "LR_x4", "LQ_mixed"])
    parser.add_argument("--l1_weight", type=float, default=1.0)
    parser.add_argument("--lpips_weight", type=float, default=0.2)
    parser.add_argument("--gan_weight", type=float, default=0.005)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_dice_every", type=int, default=5, help="Evaluate frozen-U-Net Dice every N epochs")
    parser.add_argument("--num_workers", type=int, default=0, help="Keep 0 for Windows PowerShell compatibility")
    parser.add_argument("--window_size", type=int, default=8, help="SwinIR window size")
    parser.add_argument("--depths", type=parse_depths, default=(6, 6, 6, 6), help="Comma-separated RSTB depths")
    parser.add_argument("--no_checkpoint", action="store_true", help="Disable activation checkpointing")
    parser.add_argument("--run_name", type=str, default="", help="Optional run directory name")
    parser.add_argument("--resume", type=str, default="", help="Resume from a Phase A checkpoint, usually last.pth")
    parser.add_argument("--current_best_lpips", type=float, default=0.1131)
    parser.add_argument("--save_outputs", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_module(module_name: str, script_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu().float().squeeze().numpy()
    if image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {image.shape}")
    return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def save_gray_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to save image: {path}")


def resolve_manifest_path(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else MANIFEST.parent / path


class AugmentedSRDataset(Dataset):
    """Apply paired geometric and photometric augmentation to LQ/HR tensors."""

    def __init__(self, base_dataset: HC18SRDataset, seed: int) -> None:
        self.base_dataset = base_dataset
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base_dataset[index]
        lq = sample["lq"].clone()
        hr = sample["hr"].clone()

        if self.rng.random() < 0.5:
            lq = torch.flip(lq, dims=(-1,))
            hr = torch.flip(hr, dims=(-1,))

        angle = self.rng.uniform(-5.0, 5.0)
        if abs(angle) > 1e-3:
            lq = TF.rotate(lq, angle, interpolation=InterpolationMode.BILINEAR, fill=0.0)
            hr = TF.rotate(hr, angle, interpolation=InterpolationMode.BILINEAR, fill=0.0)

        brightness = self.rng.uniform(0.95, 1.05)
        contrast = self.rng.uniform(0.95, 1.05)
        lq = self._jitter(lq, brightness, contrast)
        hr = self._jitter(hr, brightness, contrast)

        return {"lq": lq, "hr": hr, "image_id": sample["image_id"]}

    @staticmethod
    def _jitter(image: torch.Tensor, brightness: float, contrast: float) -> torch.Tensor:
        mean = image.mean(dim=(-2, -1), keepdim=True)
        image = (image - mean) * contrast + mean
        return torch.clamp(image * brightness, 0.0, 1.0)


class LPIPSLoss(nn.Module):
    """Differentiable LPIPS loss with [0, 1] grayscale inputs."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.backend = "none"
        self.model: nn.Module | None = None
        self.vgg: nn.Module | None = None

        metric = get_pyiqa_metric("lpips", device, as_loss=True)
        if metric is not None:
            self.model = metric
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad_(False)
            self.backend = "pyiqa_lpips"
            return

        try:
            from lpips import LPIPS

            self.model = LPIPS(net="vgg").to(device).eval()
            for param in self.model.parameters():
                param.requires_grad_(False)
            self.backend = "lpips_vgg_package"
            return
        except Exception:
            pass

        try:
            from torchvision.models import VGG16_Weights, vgg16

            self.vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[:16].to(device).eval()
            for param in self.vgg.parameters():
                param.requires_grad_(False)
            self.backend = "torchvision_vgg16_l2"
        except Exception:
            self.vgg = nn.Sequential(
                nn.Conv2d(3, 16, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, 3, 1, 1),
                nn.ReLU(inplace=True),
            ).to(device).eval()
            for param in self.vgg.parameters():
                param.requires_grad_(False)
            self.backend = "random_frozen_cnn_l2"

    @staticmethod
    def _rgb(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(0)
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return x

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = self._rgb(pred).to(device=self.device, dtype=torch.float32).clamp(0.0, 1.0)
        target = self._rgb(target).to(device=self.device, dtype=torch.float32).clamp(0.0, 1.0)

        if self.backend == "pyiqa_lpips" and self.model is not None:
            return self.model(pred, target).mean()
        if self.backend == "lpips_vgg_package" and self.model is not None:
            return self.model(pred.mul(2.0).sub(1.0), target.mul(2.0).sub(1.0)).mean()
        if self.vgg is not None:
            return F.mse_loss(self.vgg(pred), self.vgg(target))
        return pred.new_zeros(())


def spectral_conv(in_channels: int, out_channels: int, kernel: int, stride: int, padding: int) -> nn.Module:
    return nn.utils.spectral_norm(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding)
    )


class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            spectral_conv(input_channels, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_conv(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_conv(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_conv(128, 1, 4, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = model.state_dict()
        for key, value in state.items():
            if value.dtype.is_floating_point:
                self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[key].copy_(value.detach())

    def store(self, model: nn.Module) -> None:
        self.backup = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model: nn.Module) -> None:
        if self.backup:
            model.load_state_dict(self.backup, strict=True)
            self.backup = {}

    def cpu_state_dict(self) -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu().clone() for key, value in self.shadow.items()}


@dataclass
class RunPaths:
    run_name: str
    run_dir: Path
    log_csv: Path
    log_txt: Path
    sr_output_dir: Path
    mask_output_dir: Path
    recon_per_image_csv: Path
    recon_summary_csv: Path
    downstream_per_image_csv: Path
    downstream_summary_csv: Path


def make_run_paths(args: argparse.Namespace) -> RunPaths:
    run_name = args.run_name.strip() or f"swinir_light_phase_a_seed{args.seed}"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = MODEL_ROOT / run_name
    return RunPaths(
        run_name=run_name,
        run_dir=run_dir,
        log_csv=LOG_ROOT / f"50_train_swinir_hc18_{run_name}_{timestamp}.csv",
        log_txt=LOG_ROOT / f"50_train_swinir_hc18_{run_name}_{timestamp}.log",
        sr_output_dir=SR_OUTPUT_ROOT / run_name,
        mask_output_dir=MASK_OUTPUT_ROOT / run_name,
        recon_per_image_csv=RESULTS_DIR / f"{run_name}_reconstruction_per_image.csv",
        recon_summary_csv=RESULTS_DIR / f"{run_name}_reconstruction_summary.csv",
        downstream_per_image_csv=RESULTS_DIR / f"{run_name}_downstream_per_image.csv",
        downstream_summary_csv=RESULTS_DIR / f"{run_name}_downstream_summary.csv",
    )


def create_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def build_model(args: argparse.Namespace, device: torch.device) -> SwinIRLight:
    model = SwinIRLight(
        in_channels=1,
        out_channels=1,
        embed_dim=60,
        depths=args.depths,
        num_heads=(6,) * len(args.depths),
        window_size=args.window_size,
        mlp_ratio=2.0,
        upscale=1,
        use_checkpoint=not args.no_checkpoint,
    )
    return model.to(device)


def build_loaders(args: argparse.Namespace, batch_size: int) -> tuple[DataLoader, DataLoader, HC18SRDataset, HC18SRDataset]:
    train_base = HC18SRDataset(str(MANIFEST), input_type=args.input_type, split="train", return_mask=False)
    val_base = HC18SRDataset(str(MANIFEST), input_type=args.input_type, split="val", return_mask=False)
    train_dataset = AugmentedSRDataset(train_base, seed=args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_base,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_base, val_base


def lr_for_epoch(args: argparse.Namespace, epoch: int) -> float:
    if epoch <= args.warmup_epochs:
        return args.lr * epoch / max(1, args.warmup_epochs)
    progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.lr * cosine


def set_requires_grad(module: nn.Module | None, enabled: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad_(enabled)


def train_one_epoch(
    model: nn.Module,
    discriminator: nn.Module,
    train_loader: DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    scaler: GradScaler,
    ema: ModelEMA,
    lpips_loss: LPIPSLoss,
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.train()
    discriminator.train()
    totals = {"loss_g": 0.0, "loss_d": 0.0, "l1": 0.0, "lpips": 0.0, "gan": 0.0}
    count = 0
    use_amp = torch.cuda.is_available()

    progress = tqdm(train_loader, desc=f"Epoch {epoch:03d} train", leave=False)
    for batch_index, batch in enumerate(progress):
        try:
            lq = batch["lq"].to(device=device, dtype=torch.float32, non_blocking=True)
            hr = batch["hr"].to(device=device, dtype=torch.float32, non_blocking=True)

            optimizer_g.zero_grad(set_to_none=True)
            optimizer_d.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                pred = model(lq).clamp(0.0, 1.0)

            set_requires_grad(discriminator, True)
            with autocast(enabled=use_amp):
                real_logits = discriminator(torch.cat([lq, hr], dim=1))
                fake_logits = discriminator(torch.cat([lq, pred.detach()], dim=1))
                d_real = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                d_fake = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
                loss_d = 0.5 * (d_real + d_fake)
            scaler.scale(loss_d).backward()
            scaler.unscale_(optimizer_d)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), args.grad_clip)
            scaler.step(optimizer_d)
            scaler.update()

            optimizer_g.zero_grad(set_to_none=True)
            set_requires_grad(discriminator, False)
            with autocast(enabled=use_amp):
                loss_l1 = F.l1_loss(pred, hr)
                adv_logits = discriminator(torch.cat([lq, pred], dim=1))
                loss_gan = F.binary_cross_entropy_with_logits(adv_logits, torch.ones_like(adv_logits))
            with autocast(enabled=False):
                loss_lpips = lpips_loss(pred.float(), hr.float())

            loss_g = (
                args.l1_weight * loss_l1
                + args.lpips_weight * loss_lpips
                + args.gan_weight * loss_gan
            )
            scaler.scale(loss_g).backward()
            scaler.unscale_(optimizer_g)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer_g)
            scaler.update()
            ema.update(model)
            set_requires_grad(discriminator, True)

            batch_size = lq.size(0)
            totals["loss_g"] += float(loss_g.detach().item()) * batch_size
            totals["loss_d"] += float(loss_d.detach().item()) * batch_size
            totals["l1"] += float(loss_l1.detach().item()) * batch_size
            totals["lpips"] += float(loss_lpips.detach().item()) * batch_size
            totals["gan"] += float(loss_gan.detach().item()) * batch_size
            count += batch_size
            progress.set_postfix(
                loss=f"{totals['loss_g'] / max(count, 1):.4f}",
                lpips=f"{totals['lpips'] / max(count, 1):.4f}",
            )
        except RuntimeError:
            raise
        except Exception as exc:
            print(f"WARNING: skipped training batch {batch_index}: {exc}")
            traceback.print_exc()

    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def validate_reconstruction(
    model: nn.Module,
    val_loader: DataLoader,
    lpips_loss: LPIPSLoss,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.eval()
    totals = {"PSNR": 0.0, "SSIM": 0.0, "LPIPS": 0.0, "MAE": 0.0, "EdgeScore": 0.0}
    count = 0
    for batch_index, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch:03d} val", leave=False)):
        try:
            lq = batch["lq"].to(device=device, dtype=torch.float32, non_blocking=True)
            hr = batch["hr"].to(device=device, dtype=torch.float32, non_blocking=True)
            pred = model(lq).clamp(0.0, 1.0)
            lpips_value = float(lpips_loss(pred.float(), hr.float()).detach().cpu().item())
            batch_size = lq.size(0)
            totals["LPIPS"] += lpips_value * batch_size
            for item_index in range(batch_size):
                output = pred[item_index : item_index + 1]
                target = hr[item_index : item_index + 1]
                totals["PSNR"] += compute_psnr(output, target)
                totals["SSIM"] += compute_ssim(output, target)
                totals["MAE"] += compute_mae(output, target)
                totals["EdgeScore"] += compute_edge_score(output, target)
                count += 1
        except Exception as exc:
            print(f"WARNING: skipped validation batch {batch_index}: {exc}")
            traceback.print_exc()
    return {key: value / max(count, 1) for key, value in totals.items()}


def load_frozen_unet(device: torch.device):
    seg_module = load_module("hc18_phase_a_unet", SCRIPT_DIR / "39_improve_hc18_unet_segmentation.py")
    seg_model, checkpoint = seg_module.load_best_model(device)
    threshold = float(checkpoint.get("threshold", 0.50))
    image_size = int(checkpoint.get("image_size", 256))
    print(
        "Loaded frozen HC18 U-Net "
        f"epoch={checkpoint.get('epoch')} threshold={threshold:.2f} image_size={image_size}"
    )
    return seg_module, seg_model, threshold, image_size


@torch.no_grad()
def evaluate_dice(
    model: nn.Module,
    dataset: HC18SRDataset,
    seg_module: Any,
    seg_model: nn.Module,
    threshold: float,
    image_size: int,
    device: torch.device,
    desc: str,
    mask_output_dir: Path | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for index in tqdm(range(len(dataset)), desc=desc, leave=False):
        try:
            sample = dataset[index]
            manifest_row = dataset.samples[index]
            image_id = str(sample["image_id"])
            mask_path = resolve_manifest_path(manifest_row.get("mask_path"))
            raw_gt_mask = seg_module.read_gray_uint8(mask_path)
            if raw_gt_mask is None:
                continue
            gt_mask = seg_module.preprocess_mask(raw_gt_mask).processed_mask.astype(bool)
            if not np.any(gt_mask):
                continue

            lq = sample["lq"].unsqueeze(0).to(device=device, dtype=torch.float32)
            pred = model(lq).clamp(0.0, 1.0)
            pred_image = tensor_to_uint8_image(pred)
            pred_mask = seg_module.predict_mask(
                model=seg_model,
                image=pred_image,
                target_shape=gt_mask.shape,
                image_size=image_size,
                threshold=threshold,
                device=device,
            )
            if mask_output_dir is not None:
                save_gray_png(mask_output_dir / f"{image_id}.png", pred_mask.astype(np.uint8) * 255)
            metrics = seg_module.compute_segmentation_metrics(pred_mask, gt_mask)
            rows.append(
                {
                    "image_id": image_id,
                    "Dice": float(metrics["dice"]),
                    "IoU": float(metrics["iou"]),
                    "AreaErr": float(metrics["fetal_head_area_error_percent"]) / 100.0,
                    "DiamErr": float(metrics["equivalent_diameter_error_px"]),
                }
            )
        except Exception as exc:
            print(f"WARNING: skipped Dice sample index={index}: {exc}")
            traceback.print_exc()

    if not rows:
        return {"Dice": float("nan"), "IoU": float("nan"), "AreaErr": float("nan"), "DiamErr": float("nan")}, rows
    df = pd.DataFrame(rows)
    return {
        "Dice": float(df["Dice"].mean()),
        "IoU": float(df["IoU"].mean()),
        "AreaErr": float(df["AreaErr"].mean()),
        "DiamErr": float(df["DiamErr"].mean()),
    }, rows


def state_dict_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    ema: ModelEMA,
    discriminator: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    metrics: dict[str, float],
    batch_size: int,
    model_state_is_ema: bool = True,
    raw_model_state: dict[str, torch.Tensor] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_state = ema.cpu_state_dict() if model_state_is_ema else state_dict_cpu(model)
    torch.save(
        {
            "model_name": "swinir_light_phase_a",
            "epoch": epoch,
            "model_state_dict": model_state,
            "raw_model_state_dict": raw_model_state if raw_model_state is not None else state_dict_cpu(model),
            "ema_state_dict": ema.cpu_state_dict(),
            "discriminator_state_dict": state_dict_cpu(discriminator),
            "optimizer_g_state_dict": optimizer_g.state_dict(),
            "optimizer_d_state_dict": optimizer_d.state_dict(),
            "metrics": metrics,
            "batch_size": batch_size,
            "args": vars(args),
            "model_state_is_ema": model_state_is_ema,
        },
        path,
    )


def load_best_metric(path: Path, metric_name: str, default: float) -> float:
    if not path.exists():
        return default
    try:
        checkpoint = torch.load(path, map_location="cpu")
        metrics = checkpoint.get("metrics", {})
        value = metrics.get(metric_name)
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def load_resume_state(
    resume_path: Path,
    model: nn.Module,
    ema: ModelEMA,
    discriminator: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    checkpoint = torch.load(resume_path, map_location=device)
    state = checkpoint.get("raw_model_state_dict") or checkpoint.get("model_state_dict")
    if state is None:
        raise KeyError(f"Resume checkpoint has no model state: {resume_path}")
    model.load_state_dict(state, strict=True)
    if "ema_state_dict" in checkpoint:
        ema.shadow = {
            key: value.detach().to(device).clone()
            for key, value in checkpoint["ema_state_dict"].items()
        }
    if "discriminator_state_dict" in checkpoint:
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"], strict=True)
    if "optimizer_g_state_dict" in checkpoint:
        optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
    if "optimizer_d_state_dict" in checkpoint:
        optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    print(f"Resumed from {resume_path} at epoch {start_epoch}")
    return start_epoch


def write_rows_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@torch.no_grad()
def evaluate_full_test(
    args: argparse.Namespace,
    checkpoint_path: Path,
    paths: RunPaths,
    device: torch.device,
    seg_module: Any,
    seg_model: nn.Module,
    seg_threshold: float,
    seg_image_size: int,
) -> dict[str, float]:
    test_dataset = HC18SRDataset(str(MANIFEST), input_type=args.input_type, split="test", return_mask=False)
    model = build_model(args, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    metric_bundle = PyiqaMetricBundle.create(device)
    recon_rows: list[dict[str, Any]] = []
    if args.save_outputs:
        paths.sr_output_dir.mkdir(parents=True, exist_ok=True)

    for index in tqdm(range(len(test_dataset)), desc="Phase A test reconstruction", leave=False):
        try:
            sample = test_dataset[index]
            image_id = str(sample["image_id"])
            lq = sample["lq"].unsqueeze(0).to(device=device, dtype=torch.float32)
            hr = sample["hr"].unsqueeze(0).to(device=device, dtype=torch.float32)
            output = model(lq).clamp(0.0, 1.0)
            if args.save_outputs:
                save_gray_png(paths.sr_output_dir / f"{image_id}.png", tensor_to_uint8_image(output))
            recon_rows.append(
                {
                    "image_id": image_id,
                    "method": "SwinIR-Light Phase A",
                    "PSNR": compute_psnr(output, hr),
                    "SSIM": compute_ssim(output, hr),
                    "LPIPS": compute_lpips(output, hr, device, metric=metric_bundle.lpips),
                    "MAE": compute_mae(output, hr),
                    "EdgeScore": compute_edge_score(output, hr),
                    "NIQE": compute_niqe(output, device, metric=metric_bundle.niqe),
                    "BRISQUE": compute_brisque(output, device, metric=metric_bundle.brisque),
                }
            )
        except Exception as exc:
            print(f"WARNING: skipped test reconstruction sample index={index}: {exc}")
            traceback.print_exc()

    recon_columns = ["image_id", "method", "PSNR", "SSIM", "LPIPS", "MAE", "EdgeScore", "NIQE", "BRISQUE"]
    write_rows_csv(paths.recon_per_image_csv, recon_rows, recon_columns)
    recon_df = pd.DataFrame(recon_rows)
    recon_summary = {
        "method": "SwinIR-Light Phase A",
        "PSNR": float(recon_df["PSNR"].mean()),
        "SSIM": float(recon_df["SSIM"].mean()),
        "LPIPS": float(recon_df["LPIPS"].mean()),
        "MAE": float(recon_df["MAE"].mean()),
        "EdgeScore": float(recon_df["EdgeScore"].mean()),
        "NIQE": float(recon_df["NIQE"].mean()),
        "BRISQUE": float(recon_df["BRISQUE"].mean()),
    }
    pd.DataFrame([recon_summary]).to_csv(paths.recon_summary_csv, index=False)

    dice_summary, dice_rows = evaluate_dice(
        model=model,
        dataset=test_dataset,
        seg_module=seg_module,
        seg_model=seg_model,
        threshold=seg_threshold,
        image_size=seg_image_size,
        device=device,
        desc="Phase A test downstream Dice",
        mask_output_dir=paths.mask_output_dir if args.save_outputs else None,
    )
    downstream_columns = ["image_id", "Dice", "IoU", "AreaErr", "DiamErr"]
    write_rows_csv(paths.downstream_per_image_csv, dice_rows, downstream_columns)
    downstream_summary = {
        "method": "SwinIR-Light Phase A",
        "Dice_mean": dice_summary["Dice"],
        "IoU_mean": dice_summary["IoU"],
        "AreaErr_mean": dice_summary["AreaErr"],
        "DiamErr_mean": dice_summary["DiamErr"],
    }
    pd.DataFrame([downstream_summary]).to_csv(paths.downstream_summary_csv, index=False)

    return {**recon_summary, **downstream_summary}


def load_reference_row(recon_method: str, downstream_method: str, label: str) -> dict[str, Any]:
    row: dict[str, Any] = {"method": label}
    if CURRENT_RECON_SUMMARY.exists():
        recon = pd.read_csv(CURRENT_RECON_SUMMARY)
        match = recon[recon["method"] == recon_method]
        if not match.empty:
            for column in ["PSNR", "SSIM", "LPIPS", "MAE", "EdgeScore", "NIQE", "BRISQUE"]:
                if column in match.columns:
                    row[column] = float(match.iloc[0][column])
    if CURRENT_DOWNSTREAM_SUMMARY.exists():
        downstream = pd.read_csv(CURRENT_DOWNSTREAM_SUMMARY)
        match = downstream[downstream["method"] == downstream_method]
        if not match.empty:
            mapping = {
                "Dice_mean": "Dice_mean",
                "IoU_mean": "IoU_mean",
                "AreaErr_mean": "AreaErr_mean",
                "DiamErr_mean": "DiamErr_mean",
            }
            for source, target in mapping.items():
                if source in match.columns:
                    row[target] = float(match.iloc[0][source])
    return row


def print_comparison_table(phase_summary: dict[str, float], current_best_lpips: float) -> bool:
    rows = [
        load_reference_row(
            "Quality-aware SR + LPIPS + GAN",
            "Quality-aware+perceptual",
            "Current LPIPS champ: QA+perceptual",
        ),
        load_reference_row(
            "SR-only + LPIPS + GAN",
            "SR-only+perceptual",
            "Current Dice champ: SR-only+perceptual",
        ),
        phase_summary,
    ]
    columns = [
        "method",
        "PSNR",
        "SSIM",
        "LPIPS",
        "MAE",
        "EdgeScore",
        "NIQE",
        "BRISQUE",
        "Dice_mean",
        "IoU_mean",
        "AreaErr_mean",
        "DiamErr_mean",
    ]
    table = pd.DataFrame(rows)
    for column in columns:
        if column not in table.columns:
            table[column] = np.nan
    print()
    print("PHASE A COMPARISON TABLE")
    print(table[columns].to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    phase_lpips = float(phase_summary["LPIPS"])
    delta = phase_lpips - current_best_lpips
    passed = phase_lpips < current_best_lpips
    print()
    print(f"Phase A LPIPS: {phase_lpips:.6f}")
    print(f"Current best LPIPS gate: {current_best_lpips:.6f}")
    print(f"Delta vs gate: {delta:+.6f}")
    print(f"ACCEPTANCE: {'PASS' if passed else 'FAIL'}")
    if not passed:
        print(
            "Diagnosis queue for Phase A: try window_size=16, then depths=6,6,6,6,6,6, "
            "then 200 epochs, then gradient accumulation for effective batch 16."
        )
    return passed


def run_training(args: argparse.Namespace, paths: RunPaths, batch_size: int) -> tuple[bool, dict[str, float]]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    print(f"Using device: {device}")
    print(f"Run name: {paths.run_name}")
    print(f"Batch size: {batch_size}")
    print(f"SwinIR-Light config: embed_dim=60 depths={args.depths} heads={(6,) * len(args.depths)} window={args.window_size}")

    train_loader, val_loader, _, val_dataset = build_loaders(args, batch_size)
    test_dataset = HC18SRDataset(str(MANIFEST), input_type=args.input_type, split="test", return_mask=False)
    model = build_model(args, device)
    discriminator = PatchDiscriminator().to(device)
    ema = ModelEMA(model, decay=args.ema_decay)
    lpips_loss = LPIPSLoss(device)
    print(f"LPIPS backend: {lpips_loss.backend}")
    seg_module, seg_model, seg_threshold, seg_image_size = load_frozen_unet(device)

    optimizer_g = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=1e-4)
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=args.d_lr, betas=(0.9, 0.99), weight_decay=1e-4)
    scaler = GradScaler(enabled=use_amp)

    create_dirs(paths.run_dir, paths.log_csv.parent)
    best_lpips = load_best_metric(paths.run_dir / "best_lpips.pth", "val_LPIPS", float("inf"))
    best_dice = load_best_metric(paths.run_dir / "best_dice.pth", "val_Dice", -float("inf"))
    start_epoch = 1
    if args.resume.strip():
        resume_path = Path(args.resume.strip())
        if not resume_path.is_absolute():
            resume_path = PROJECT_ROOT / resume_path
        start_epoch = load_resume_state(
            resume_path=resume_path,
            model=model,
            ema=ema,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            device=device,
        )
        print(f"Loaded previous best_val_lpips={best_lpips:.6f} best_val_dice={best_dice:.6f}")
    phase_summary: dict[str, float] = {}

    csv_columns = [
        "epoch",
        "lr",
        "train_loss_g",
        "train_loss_d",
        "train_l1",
        "train_lpips",
        "train_gan",
        "val_PSNR",
        "val_SSIM",
        "val_LPIPS",
        "val_MAE",
        "val_EdgeScore",
        "val_Dice",
        "val_IoU",
        "batch_size",
        "lpips_backend",
    ]
    with paths.log_csv.open("w", newline="") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=csv_columns)
        writer.writeheader()

        for epoch in range(start_epoch, args.epochs + 1):
            lr = lr_for_epoch(args, epoch)
            for group in optimizer_g.param_groups:
                group["lr"] = lr
            train_metrics = train_one_epoch(
                model=model,
                discriminator=discriminator,
                train_loader=train_loader,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                scaler=scaler,
                ema=ema,
                lpips_loss=lpips_loss,
                args=args,
                device=device,
                epoch=epoch,
            )

            ema.store(model)
            ema.copy_to(model)
            val_metrics = validate_reconstruction(model, val_loader, lpips_loss, device, epoch)
            if epoch % args.eval_dice_every == 0 or epoch == args.epochs:
                val_dice_metrics, _ = evaluate_dice(
                    model=model,
                    dataset=val_dataset,
                    seg_module=seg_module,
                    seg_model=seg_model,
                    threshold=seg_threshold,
                    image_size=seg_image_size,
                    device=device,
                    desc=f"Epoch {epoch:03d} val Dice",
                    mask_output_dir=None,
                )
            else:
                val_dice_metrics = {"Dice": float("nan"), "IoU": float("nan"), "AreaErr": float("nan"), "DiamErr": float("nan")}

            checkpoint_metrics = {
                **{f"val_{key}": value for key, value in val_metrics.items()},
                **{f"val_{key}": value for key, value in val_dice_metrics.items()},
            }
            raw_state_for_checkpoint = {
                key: value.detach().cpu().clone()
                for key, value in ema.backup.items()
            }

            if val_metrics["LPIPS"] < best_lpips:
                best_lpips = val_metrics["LPIPS"]
                save_checkpoint(
                    paths.run_dir / "best_lpips.pth",
                    model,
                    ema,
                    discriminator,
                    optimizer_g,
                    optimizer_d,
                    args,
                    epoch,
                    checkpoint_metrics,
                    batch_size,
                    model_state_is_ema=True,
                    raw_model_state=raw_state_for_checkpoint,
                )
            if not math.isnan(val_dice_metrics["Dice"]) and val_dice_metrics["Dice"] > best_dice:
                best_dice = val_dice_metrics["Dice"]
                save_checkpoint(
                    paths.run_dir / "best_dice.pth",
                    model,
                    ema,
                    discriminator,
                    optimizer_g,
                    optimizer_d,
                    args,
                    epoch,
                    checkpoint_metrics,
                    batch_size,
                    model_state_is_ema=True,
                    raw_model_state=raw_state_for_checkpoint,
                )

            ema.restore(model)
            save_checkpoint(
                paths.run_dir / "last.pth",
                model,
                ema,
                discriminator,
                optimizer_g,
                optimizer_d,
                args,
                epoch,
                checkpoint_metrics,
                batch_size,
                model_state_is_ema=False,
            )
            torch.save({"ema_state_dict": ema.cpu_state_dict(), "epoch": epoch, "args": vars(args)}, paths.run_dir / "ema.pth")

            row = {
                "epoch": epoch,
                "lr": f"{lr:.8f}",
                "train_loss_g": f"{train_metrics['loss_g']:.6f}",
                "train_loss_d": f"{train_metrics['loss_d']:.6f}",
                "train_l1": f"{train_metrics['l1']:.6f}",
                "train_lpips": f"{train_metrics['lpips']:.6f}",
                "train_gan": f"{train_metrics['gan']:.6f}",
                "val_PSNR": f"{val_metrics['PSNR']:.6f}",
                "val_SSIM": f"{val_metrics['SSIM']:.6f}",
                "val_LPIPS": f"{val_metrics['LPIPS']:.6f}",
                "val_MAE": f"{val_metrics['MAE']:.6f}",
                "val_EdgeScore": f"{val_metrics['EdgeScore']:.6f}",
                "val_Dice": f"{val_dice_metrics['Dice']:.6f}",
                "val_IoU": f"{val_dice_metrics['IoU']:.6f}",
                "batch_size": batch_size,
                "lpips_backend": lpips_loss.backend,
            }
            writer.writerow(row)
            csv_handle.flush()

            print(
                f"epoch={epoch:03d}/{args.epochs:03d} lr={lr:.6g} "
                f"train_g={train_metrics['loss_g']:.5f} train_d={train_metrics['loss_d']:.5f} "
                f"val_lpips={val_metrics['LPIPS']:.6f} val_psnr={val_metrics['PSNR']:.4f} "
                f"val_ssim={val_metrics['SSIM']:.4f} "
                f"val_dice={val_dice_metrics['Dice']:.6f} "
                f"best_lpips={best_lpips:.6f} best_dice={best_dice:.6f}"
            )

    best_path = paths.run_dir / "best_lpips.pth"
    if not best_path.exists():
        raise RuntimeError("Training finished without writing best_lpips.pth")

    phase_summary = evaluate_full_test(
        args=args,
        checkpoint_path=best_path,
        paths=paths,
        device=device,
        seg_module=seg_module,
        seg_model=seg_model,
        seg_threshold=seg_threshold,
        seg_image_size=seg_image_size,
    )

    metadata = {
        "run_name": paths.run_name,
        "phase": "A",
        "model": "SwinIR-Light",
        "seed": args.seed,
        "batch_size": batch_size,
        "epochs": args.epochs,
        "best_val_lpips": best_lpips,
        "best_val_dice": best_dice,
        "best_lpips_checkpoint": str(paths.run_dir / "best_lpips.pth"),
        "best_dice_checkpoint": str(paths.run_dir / "best_dice.pth"),
        "last_checkpoint": str(paths.run_dir / "last.pth"),
        "ema_checkpoint": str(paths.run_dir / "ema.pth"),
        "log_csv": str(paths.log_csv),
        "log_txt": str(paths.log_txt),
        "reconstruction_summary": str(paths.recon_summary_csv),
        "downstream_summary": str(paths.downstream_summary_csv),
        "lpips_backend": lpips_loss.backend,
        "args": vars(args),
    }
    (paths.run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    passed = print_comparison_table(phase_summary, current_best_lpips=args.current_best_lpips)
    return passed, phase_summary


def batch_size_candidates(initial: int) -> list[int]:
    candidates = []
    for value in (initial, 8, 4, 2):
        if value <= initial and value not in candidates:
            candidates.append(value)
    if 2 not in candidates:
        candidates.append(2)
    return candidates


def main() -> int:
    args = parse_args()
    paths = make_run_paths(args)
    create_dirs(paths.run_dir, LOG_ROOT, RESULTS_DIR)
    log_handle = paths.log_txt.open("w", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(sys.stdout, log_handle)
    sys.stderr = Tee(sys.stderr, log_handle)
    try:
        print("Phase A SwinIR-Light training started")
        print(f"Project root: {PROJECT_ROOT}")
        print(f"Arguments: {json.dumps(vars(args), default=str, indent=2)}")
        last_error: RuntimeError | None = None
        for candidate_batch_size in batch_size_candidates(args.batch_size):
            try:
                passed, _ = run_training(args, paths, candidate_batch_size)
                print(f"Log CSV: {paths.log_csv}")
                print(f"Log TXT: {paths.log_txt}")
                print(f"Run directory: {paths.run_dir}")
                print(f"SR outputs: {paths.sr_output_dir}")
                print("STOPPING after Phase A for review.")
                return 0 if passed else 0
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                last_error = exc
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"CUDA OOM at batch size {candidate_batch_size}; trying next fallback.")
        if last_error is not None:
            raise last_error
        raise RuntimeError("No batch size candidates were attempted.")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
