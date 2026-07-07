r"""Phase B: segmentation-aware HC18 super-resolution training.

This script is prepared but should not be launched for full training until the
Phase A backbone decision is reviewed.

Smoke test:

    python .\scripts\51_train_segaware_hc18.py --dry-run --device cpu
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
from metrics_utils import compute_lpips, compute_psnr, compute_ssim, get_pyiqa_metric
from models.swinir_light import SwinIRLight


MANIFEST = PROJECT_ROOT / "data" / "processed" / "HC18_SR" / "hc18_lq_manifest.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_ROOT = RESULTS_DIR / "models"
LOG_ROOT = RESULTS_DIR / "training_logs"
HARD_MEMBERSHIP_CSV = RESULTS_DIR / "downstream_hard_subset_membership.csv"
DEFAULT_SWNIR_CKPT = MODEL_ROOT / "swinir_light_phase_a_seed42" / "best_lpips.pth"
DEFAULT_QA_CKPT = MODEL_ROOT / "qa_sr_perceptual_hc18_final.pth"
LQ_DICE_CACHE = RESULTS_DIR / "phaseB_lq_dice_weights.csv"


def load_module(module_name: str, script_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


phase_a = load_module("phase_a_swinir_trainer", SCRIPT_DIR / "50_train_swinir_hc18.py")
qa_module = load_module("hc18_quality_aware_sr_train", SCRIPT_DIR / "07_train_quality_aware_sr_hc18.py")
seg_module = load_module("hc18_improved_unet_segmentation", SCRIPT_DIR / "39_improve_hc18_unet_segmentation.py")


def parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--seeds must be comma-separated integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds must contain at least one seed")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase B segmentation-aware HC18 SR.")
    parser.add_argument("--backbone", choices=["swinir", "qa_cnn"], default="swinir")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=parse_seeds, default=None)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--d_lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--optimizer", choices=["adamw"], default="adamw")
    parser.add_argument("--scheduler", choices=["cosine"], default="cosine")
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--input_type", default="LQ_mixed", choices=["LR_x2", "LR_x4", "LQ_mixed"])
    parser.add_argument("--run_name", default="phaseB_segaware")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pretrained", default="")
    parser.add_argument("--seg_warmup_epochs", type=int, default=15)
    parser.add_argument("--r1_every", type=int, default=16)
    parser.add_argument("--r1_gamma", type=float, default=10.0)
    parser.add_argument("--eval_dice_every", type=int, default=5)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_manifest_path(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else MANIFEST.parent / path


def load_processed_mask(mask_path: Path) -> torch.Tensor:
    raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    processed = seg_module.preprocess_mask(raw).processed_mask.astype(np.float32)
    return torch.from_numpy(processed).unsqueeze(0)


def compute_quality_stats(dataset: HC18SRDataset) -> tuple[torch.Tensor, torch.Tensor]:
    features = []
    for index in range(len(dataset)):
        features.append(qa_module.compute_quality_features(dataset[index]["lq"]).view(-1))
    stacked = torch.stack(features, dim=0)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0)
    std = torch.where(std == 0.0, torch.ones_like(std), std)
    return mean, std


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().float().squeeze().numpy()
    return np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


@dataclass
class QualityStats:
    mean: torch.Tensor
    std: torch.Tensor


class SegAwareSRDataset(Dataset):
    def __init__(
        self,
        base_dataset: HC18SRDataset,
        weights_by_id: dict[str, tuple[float, float]],
        quality_stats: QualityStats | None,
        augment: bool,
        seed: int,
    ) -> None:
        self.base_dataset = base_dataset
        self.weights_by_id = weights_by_id
        self.quality_stats = quality_stats
        self.augment = augment
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base_dataset[index]
        manifest_row = self.base_dataset.samples[index]
        image_id = str(sample["image_id"])
        lq = sample["lq"].clone()
        hr = sample["hr"].clone()
        mask = load_processed_mask(resolve_manifest_path(manifest_row["mask_path"]))

        if self.augment:
            lq, hr, mask = self.apply_augment(lq, hr, mask)

        lq_dice, hard_weight = self.weights_by_id.get(image_id, (1.0, 1.0))
        output: dict[str, Any] = {
            "lq": lq,
            "hr": hr,
            "mask": mask,
            "image_id": image_id,
            "lq_dice": torch.tensor(float(lq_dice), dtype=torch.float32),
            "hard_weight": torch.tensor(float(hard_weight), dtype=torch.float32),
        }
        if self.quality_stats is not None:
            quality = qa_module.compute_quality_features(lq)
            mean = self.quality_stats.mean.to(quality.device, quality.dtype).view(1, -1)
            std = self.quality_stats.std.to(quality.device, quality.dtype).view(1, -1)
            output["quality"] = ((quality - mean) / std).view(-1)
        return output

    def apply_augment(
        self,
        lq: torch.Tensor,
        hr: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.rng.random() < 0.5:
            lq = torch.flip(lq, dims=(-1,))
            hr = torch.flip(hr, dims=(-1,))
            mask = torch.flip(mask, dims=(-1,))

        angle = self.rng.uniform(-5.0, 5.0)
        if abs(angle) > 1e-3:
            lq = TF.rotate(lq, angle, interpolation=InterpolationMode.BILINEAR, fill=0.0)
            hr = TF.rotate(hr, angle, interpolation=InterpolationMode.BILINEAR, fill=0.0)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0.0)

        brightness = self.rng.uniform(0.95, 1.05)
        contrast = self.rng.uniform(0.95, 1.05)
        lq = self.jitter(lq, brightness, contrast)
        hr = self.jitter(hr, brightness, contrast)
        return lq, hr, (mask > 0.5).float()

    @staticmethod
    def jitter(image: torch.Tensor, brightness: float, contrast: float) -> torch.Tensor:
        mean = image.mean(dim=(-2, -1), keepdim=True)
        image = (image - mean) * contrast + mean
        return torch.clamp(image * brightness, 0.0, 1.0)


def load_frozen_unet(device: torch.device) -> tuple[nn.Module, float, int]:
    model, checkpoint = seg_module.load_best_model(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    threshold = float(checkpoint.get("threshold", 0.50))
    image_size = int(checkpoint.get("image_size", 256))
    return model, threshold, image_size


@torch.no_grad()
def compute_or_load_lq_weights(
    split: str,
    device: torch.device,
    force: bool = False,
) -> dict[str, tuple[float, float]]:
    if LQ_DICE_CACHE.exists() and not force:
        cached = pd.read_csv(LQ_DICE_CACHE)
        split_df = cached[cached["split"] == split].copy()
        if not split_df.empty:
            return {
                str(row["image_id"]): (float(row["lq_dice"]), float(row["hard_weight"]))
                for _, row in split_df.iterrows()
            }

    unet, threshold, image_size = load_frozen_unet(device)
    rows: list[dict[str, Any]] = []
    existing = pd.read_csv(LQ_DICE_CACHE) if LQ_DICE_CACHE.exists() else pd.DataFrame()
    if not existing.empty:
        rows.extend(existing[existing["split"] != split].to_dict("records"))

    dataset = HC18SRDataset(str(MANIFEST), input_type="LQ_mixed", split=split, return_mask=False)
    for index in tqdm(range(len(dataset)), desc=f"Cache LQ Dice ({split})", leave=False):
        sample = dataset[index]
        manifest_row = dataset.samples[index]
        image_id = str(sample["image_id"])
        mask = load_processed_mask(resolve_manifest_path(manifest_row["mask_path"]))
        gt = mask.squeeze(0).numpy().astype(bool)
        pred_mask = seg_module.predict_mask(
            model=unet,
            image=tensor_to_uint8_image(sample["lq"]),
            target_shape=gt.shape,
            image_size=image_size,
            threshold=threshold,
            device=device,
        )
        metrics = seg_module.compute_segmentation_metrics(pred_mask, gt)
        lq_dice = float(metrics["dice"])
        hard_weight = float(np.clip(1.0 + 4.0 * (1.0 - lq_dice), 1.0, 5.0))
        rows.append({"split": split, "image_id": image_id, "lq_dice": lq_dice, "hard_weight": hard_weight})

    LQ_DICE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(LQ_DICE_CACHE, index=False)
    return {
        str(row["image_id"]): (float(row["lq_dice"]), float(row["hard_weight"]))
        for row in rows
        if row["split"] == split
    }


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((pred - target) ** 2 + self.eps**2).mean()


class EdgeSobelLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_x = F.conv2d(pred, self.kernel_x.to(pred.device, pred.dtype), padding=1)
        pred_y = F.conv2d(pred, self.kernel_y.to(pred.device, pred.dtype), padding=1)
        target_x = F.conv2d(target, self.kernel_x.to(target.device, target.dtype), padding=1)
        target_y = F.conv2d(target, self.kernel_y.to(target.device, target.dtype), padding=1)
        return F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)


class PyiqaPairLoss(nn.Module):
    def __init__(self, metric_name: str, device: torch.device) -> None:
        super().__init__()
        self.metric_name = metric_name
        self.metric = get_pyiqa_metric(metric_name, device, as_loss=True)
        if self.metric is not None:
            self.metric.eval()
            for param in self.metric.parameters():
                param.requires_grad_(False)

    @staticmethod
    def rgb(x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0.0, 1.0)
        return x.repeat(1, 3, 1, 1) if x.size(1) == 1 else x

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.metric is None:
            return pred.new_zeros(())
        return self.metric(self.rgb(pred.float()), self.rgb(target.float())).mean()


def spectral_conv(in_channels: int, out_channels: int, kernel: int, stride: int, padding: int) -> nn.Module:
    return nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel, stride, padding))


class PatchDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            spectral_conv(2, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_conv(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_conv(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_conv(128, 1, 4, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModelEMA(phase_a.ModelEMA):
    pass


def build_generator(backbone: str, device: torch.device) -> nn.Module:
    if backbone == "swinir":
        return SwinIRLight(
            in_channels=1,
            out_channels=1,
            embed_dim=60,
            depths=(6, 6, 6, 6),
            num_heads=(6, 6, 6, 6),
            window_size=8,
            mlp_ratio=2.0,
            upscale=1,
            use_checkpoint=True,
        ).to(device)
    return qa_module.QualityAwareSRModel().to(device)


def load_pretrained_if_available(model: nn.Module, args: argparse.Namespace, device: torch.device) -> None:
    path = Path(args.pretrained) if args.pretrained else (DEFAULT_SWNIR_CKPT if args.backbone == "swinir" else DEFAULT_QA_CKPT)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        print(f"WARNING: pretrained checkpoint not found; starting from scratch: {path}")
        return
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("raw_model_state_dict") or checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    model.load_state_dict(state, strict=True)
    print(f"Loaded pretrained {args.backbone} backbone from {path}")


def forward_generator(model: nn.Module, batch: dict[str, Any], backbone: str, device: torch.device) -> torch.Tensor:
    lq = batch["lq"].to(device=device, dtype=torch.float32, non_blocking=True)
    if backbone == "qa_cnn":
        quality = batch["quality"].to(device=device, dtype=torch.float32, non_blocking=True)
        return model(lq, quality).clamp(0.0, 1.0)
    return model(lq).clamp(0.0, 1.0)


def seg_weight_for_epoch(epoch: int, warmup_epochs: int) -> float:
    return 0.10 * min(1.0, max(0.0, epoch / max(1, warmup_epochs)))


def lr_for_epoch(args: argparse.Namespace, epoch: int) -> float:
    if epoch <= args.warmup:
        return args.lr * epoch / max(1, args.warmup)
    progress = (epoch - args.warmup) / max(1, args.epochs - args.warmup)
    return args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def seg_aware_loss(
    frozen_unet: nn.Module,
    pred: torch.Tensor,
    masks: torch.Tensor,
    hard_weights: torch.Tensor,
    image_size: int,
) -> torch.Tensor:
    pred_for_unet = F.interpolate(pred.float().clamp(0.0, 1.0), size=(image_size, image_size), mode="bilinear", align_corners=False)
    mask_for_unet = F.interpolate(masks.float(), size=(image_size, image_size), mode="nearest")
    logits = frozen_unet(pred_for_unet)
    bce = F.binary_cross_entropy_with_logits(logits, mask_for_unet, reduction="none").mean(dim=(1, 2, 3))
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = torch.sum(probs * mask_for_unet, dim=dims)
    denom = torch.sum(probs, dim=dims) + torch.sum(mask_for_unet, dim=dims)
    dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (denom + 1e-6)
    per_image = bce + dice_loss
    return (hard_weights.view(-1).to(per_image.device) * per_image).mean()


def r1_penalty(real_logits: torch.Tensor, real_input: torch.Tensor) -> torch.Tensor:
    grad = torch.autograd.grad(
        outputs=real_logits.sum(),
        inputs=real_input,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return grad.pow(2).view(grad.size(0), -1).sum(dim=1).mean()


def build_datasets(args: argparse.Namespace, seed: int, device: torch.device) -> tuple[Dataset, Dataset, QualityStats | None]:
    train_base = HC18SRDataset(str(MANIFEST), input_type=args.input_type, split="train", return_mask=False)
    val_base = HC18SRDataset(str(MANIFEST), input_type=args.input_type, split="val", return_mask=False)
    quality_stats = None
    if args.backbone == "qa_cnn":
        mean, std = compute_quality_stats(train_base)
        quality_stats = QualityStats(mean=mean, std=std)
    train_weights = compute_or_load_lq_weights("train", device)
    val_weights = compute_or_load_lq_weights("val", device)
    return (
        SegAwareSRDataset(train_base, train_weights, quality_stats, args.augment, seed),
        SegAwareSRDataset(val_base, val_weights, quality_stats, False, seed),
        quality_stats,
    )


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    backbone: str,
    device: torch.device,
    lpips_loss: PyiqaPairLoss,
    frozen_unet: nn.Module,
    seg_threshold: float,
    seg_image_size: int,
) -> dict[str, float]:
    model.eval()
    rows = []
    for batch in tqdm(loader, desc="Phase B val", leave=False):
        lq = batch["lq"].to(device=device, dtype=torch.float32)
        hr = batch["hr"].to(device=device, dtype=torch.float32)
        mask = batch["mask"].to(device=device, dtype=torch.float32)
        pred = forward_generator(model, batch, backbone, device)
        logits = frozen_unet(F.interpolate(pred, size=(seg_image_size, seg_image_size), mode="bilinear", align_corners=False))
        mask_256 = F.interpolate(mask, size=(seg_image_size, seg_image_size), mode="nearest")
        pred_mask = (torch.sigmoid(logits) >= seg_threshold).float()
        dims = (1, 2, 3)
        intersection = torch.sum(pred_mask * mask_256, dim=dims)
        pred_area = torch.sum(pred_mask, dim=dims)
        target_area = torch.sum(mask_256, dim=dims)
        dice = (2.0 * intersection + 1e-6) / (pred_area + target_area + 1e-6)
        iou = (intersection + 1e-6) / (pred_area + target_area - intersection + 1e-6)
        for item in range(lq.size(0)):
            rows.append(
                {
                    "image_id": str(batch["image_id"][item]),
                    "PSNR": compute_psnr(pred[item : item + 1], hr[item : item + 1]),
                    "SSIM": compute_ssim(pred[item : item + 1], hr[item : item + 1]),
                    "LPIPS": compute_lpips(pred[item : item + 1], hr[item : item + 1], device, metric=lpips_loss.metric),
                    "Dice": float(dice[item].detach().cpu()),
                    "IoU": float(iou[item].detach().cpu()),
                }
            )
    df = pd.DataFrame(rows)
    out = {
        "val_PSNR": float(df["PSNR"].mean()),
        "val_SSIM": float(df["SSIM"].mean()),
        "val_LPIPS": float(df["LPIPS"].mean()),
        "val_Dice": float(df["Dice"].mean()),
        "val_IoU": float(df["IoU"].mean()),
        "val_Dice_hard": float("nan"),
    }
    if HARD_MEMBERSHIP_CSV.exists():
        membership = pd.read_csv(HARD_MEMBERSHIP_CSV)
        hard_ids = set(
            membership.loc[
                membership[["bottom25_lq_dice", "lq_dice_lt_0_85", "top25_lq_lpips"]].any(axis=1),
                "image_id",
            ].astype(str)
        )
        hard_df = df[df["image_id"].astype(str).isin(hard_ids)]
        if not hard_df.empty:
            out["val_Dice_hard"] = float(hard_df["Dice"].mean())
    return out


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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": f"phaseB_segaware_{args.backbone}",
            "epoch": epoch,
            "model_state_dict": ema.cpu_state_dict(),
            "raw_model_state_dict": phase_a.state_dict_cpu(model),
            "ema_state_dict": ema.cpu_state_dict(),
            "discriminator_state_dict": phase_a.state_dict_cpu(discriminator),
            "optimizer_g_state_dict": optimizer_g.state_dict(),
            "optimizer_d_state_dict": optimizer_d.state_dict(),
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def train_one_seed(args: argparse.Namespace, seed: int, batch_size: int, device: torch.device) -> Path:
    set_seed(seed)
    run_dir = MODEL_ROOT / f"{args.run_name}_{args.backbone}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    train_dataset, val_dataset, _ = build_datasets(args, seed, device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    model = build_generator(args.backbone, device)
    load_pretrained_if_available(model, args, device)
    discriminator = PatchDiscriminator().to(device)
    frozen_unet, seg_threshold, seg_image_size = load_frozen_unet(device)
    ema = ModelEMA(model, args.ema)

    charbonnier = CharbonnierLoss()
    lpips_loss = PyiqaPairLoss("lpips", device)
    dists_loss = PyiqaPairLoss("dists", device)
    edge_loss = EdgeSobelLoss().to(device)
    optimizer_g = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=1e-4)
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=args.d_lr, betas=(0.9, 0.99), weight_decay=1e-4)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    best_lpips = float("inf")
    best_dice = -float("inf")
    best_dice_hard = -float("inf")
    global_step = 0
    log_path = LOG_ROOT / f"51_train_segaware_hc18_{args.run_name}_{args.backbone}_seed{seed}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", newline="") as handle:
        columns = ["epoch", "lr", "loss_g", "loss_d", "charbonnier", "lpips", "dists", "edge", "gan", "seg", "seg_weight", "val_PSNR", "val_SSIM", "val_LPIPS", "val_Dice", "val_Dice_hard"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            lr = lr_for_epoch(args, epoch)
            for group in optimizer_g.param_groups:
                group["lr"] = lr
            model.train()
            discriminator.train()
            totals = {key: 0.0 for key in ["loss_g", "loss_d", "charbonnier", "lpips", "dists", "edge", "gan", "seg"]}
            count = 0
            current_seg_weight = seg_weight_for_epoch(epoch, args.seg_warmup_epochs)

            for batch in tqdm(train_loader, desc=f"Phase B epoch {epoch:03d}", leave=False):
                lq = batch["lq"].to(device=device, dtype=torch.float32, non_blocking=True)
                hr = batch["hr"].to(device=device, dtype=torch.float32, non_blocking=True)
                masks = batch["mask"].to(device=device, dtype=torch.float32, non_blocking=True)
                hard_weights = batch["hard_weight"].to(device=device, dtype=torch.float32, non_blocking=True)

                optimizer_d.zero_grad(set_to_none=True)
                optimizer_g.zero_grad(set_to_none=True)

                with autocast(enabled=torch.cuda.is_available()):
                    pred = forward_generator(model, batch, args.backbone, device)

                real_pair = torch.cat([lq, hr], dim=1)
                fake_pair = torch.cat([lq, pred.detach()], dim=1)
                real_pair.requires_grad_(global_step % args.r1_every == 0)
                with autocast(enabled=torch.cuda.is_available()):
                    real_logits = discriminator(real_pair)
                    fake_logits = discriminator(fake_pair)
                    loss_d = 0.5 * (
                        F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                        + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
                    )
                    if global_step % args.r1_every == 0:
                        loss_d = loss_d + 0.5 * args.r1_gamma * r1_penalty(real_logits, real_pair) * args.r1_every
                scaler.scale(loss_d).backward()
                scaler.unscale_(optimizer_d)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), args.grad_clip)
                scaler.step(optimizer_d)
                scaler.update()

                optimizer_g.zero_grad(set_to_none=True)
                with autocast(enabled=torch.cuda.is_available()):
                    pred = forward_generator(model, batch, args.backbone, device)
                    loss_charb = charbonnier(pred, hr)
                    loss_edge = edge_loss(pred, hr)
                    adv_logits = discriminator(torch.cat([lq, pred], dim=1))
                    loss_gan = F.binary_cross_entropy_with_logits(adv_logits, torch.ones_like(adv_logits))
                with autocast(enabled=False):
                    loss_lpips = lpips_loss(pred.float(), hr.float())
                    loss_dists = dists_loss(pred.float(), hr.float())
                    loss_seg = seg_aware_loss(frozen_unet, pred.float(), masks.float(), hard_weights, seg_image_size)
                loss_g = (
                    1.00 * loss_charb
                    + 0.20 * loss_lpips
                    + 0.10 * loss_dists
                    + 0.05 * loss_edge
                    + 0.005 * loss_gan
                    + current_seg_weight * loss_seg
                )
                scaler.scale(loss_g).backward()
                scaler.unscale_(optimizer_g)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer_g)
                scaler.update()
                ema.update(model)
                global_step += 1

                batch_n = lq.size(0)
                count += batch_n
                for key, value in {
                    "loss_g": loss_g,
                    "loss_d": loss_d,
                    "charbonnier": loss_charb,
                    "lpips": loss_lpips,
                    "dists": loss_dists,
                    "edge": loss_edge,
                    "gan": loss_gan,
                    "seg": loss_seg,
                }.items():
                    totals[key] += float(value.detach().cpu()) * batch_n

            val_metrics = validate(model, val_loader, args.backbone, device, lpips_loss, frozen_unet, seg_threshold, seg_image_size) if epoch % args.eval_dice_every == 0 or epoch == args.epochs else {
                "val_PSNR": float("nan"),
                "val_SSIM": float("nan"),
                "val_LPIPS": float("nan"),
                "val_Dice": float("nan"),
                "val_Dice_hard": float("nan"),
            }
            metrics = {**{key: value / max(count, 1) for key, value in totals.items()}, **val_metrics}
            save_checkpoint(run_dir / "last.pth", model, ema, discriminator, optimizer_g, optimizer_d, args, epoch, metrics)
            torch.save({"ema_state_dict": ema.cpu_state_dict(), "epoch": epoch, "args": vars(args)}, run_dir / "ema.pth")
            if not math.isnan(val_metrics["val_LPIPS"]) and val_metrics["val_LPIPS"] < best_lpips:
                best_lpips = val_metrics["val_LPIPS"]
                save_checkpoint(run_dir / "best_lpips.pth", model, ema, discriminator, optimizer_g, optimizer_d, args, epoch, metrics)
            if not math.isnan(val_metrics["val_Dice"]) and val_metrics["val_Dice"] > best_dice:
                best_dice = val_metrics["val_Dice"]
                save_checkpoint(run_dir / "best_dice.pth", model, ema, discriminator, optimizer_g, optimizer_d, args, epoch, metrics)
            if not math.isnan(val_metrics["val_Dice_hard"]) and val_metrics["val_Dice_hard"] > best_dice_hard:
                best_dice_hard = val_metrics["val_Dice_hard"]
                save_checkpoint(run_dir / "best_dice_hard.pth", model, ema, discriminator, optimizer_g, optimizer_d, args, epoch, metrics)

            row = {"epoch": epoch, "lr": lr, "seg_weight": current_seg_weight, **metrics}
            writer.writerow({column: row.get(column, "") for column in columns})
            handle.flush()
            print(
                f"epoch={epoch:03d}/{args.epochs:03d} loss_g={metrics['loss_g']:.5f} "
                f"val_lpips={val_metrics['val_LPIPS']:.6f} val_dice={val_metrics['val_Dice']:.6f} "
                f"val_dice_hard={val_metrics['val_Dice_hard']:.6f}"
            )

    return run_dir / "best_dice_hard.pth" if (run_dir / "best_dice_hard.pth").exists() else run_dir / "last.pth"


def average_checkpoints(checkpoints: list[Path], output_path: Path, device: torch.device) -> None:
    if not checkpoints:
        return
    avg_state: dict[str, torch.Tensor] = {}
    for index, path in enumerate(checkpoints):
        checkpoint = torch.load(path, map_location=device)
        state = checkpoint["model_state_dict"]
        for key, value in state.items():
            value = value.detach().float().cpu()
            avg_state[key] = value.clone() if index == 0 else avg_state[key].add_(value)
    for key in avg_state:
        avg_state[key].div_(len(checkpoints))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": avg_state, "source_checkpoints": [str(path) for path in checkpoints]}, output_path)


def dry_run_smoke_test(args: argparse.Namespace, device: torch.device) -> None:
    set_seed(args.seed)
    args.augment = False
    train_base = HC18SRDataset(str(MANIFEST), input_type=args.input_type, split="train", return_mask=False)
    quality_stats = None
    if args.backbone == "qa_cnn":
        sample_features = [
            qa_module.compute_quality_features(train_base[index]["lq"]).view(-1)
            for index in range(min(8, len(train_base)))
        ]
        stacked = torch.stack(sample_features, dim=0)
        std = stacked.std(dim=0)
        std = torch.where(std == 0.0, torch.ones_like(std), std)
        quality_stats = QualityStats(mean=stacked.mean(dim=0), std=std)
    dry_weights = {
        str(train_base[index]["image_id"]): (1.0, 1.0)
        for index in range(min(4, len(train_base)))
    }
    train_dataset = SegAwareSRDataset(train_base, dry_weights, quality_stats, False, args.seed)
    loader = DataLoader(train_dataset, batch_size=min(2, args.batch_size), shuffle=False, num_workers=0)
    batch = next(iter(loader))
    model = build_generator(args.backbone, device)
    load_pretrained_if_available(model, args, device)
    frozen_unet, _, seg_image_size = load_frozen_unet(device)
    pred = forward_generator(model, batch, args.backbone, device)
    masks = batch["mask"].to(device=device, dtype=torch.float32)
    weights = batch["hard_weight"].to(device=device, dtype=torch.float32)
    loss = seg_aware_loss(frozen_unet, pred, masks, weights, seg_image_size)
    model.zero_grad(set_to_none=True)
    loss.backward()
    grads = [param.grad for param in model.parameters() if param.requires_grad]
    if not any(grad is not None and torch.isfinite(grad).all() for grad in grads):
        raise RuntimeError("Dry-run failed: no finite generator gradients after frozen-U-Net seg loss backward")
    frozen_grads = [param.grad for param in frozen_unet.parameters()]
    if any(grad is not None for grad in frozen_grads):
        raise RuntimeError("Dry-run failed: frozen U-Net accumulated parameter gradients")
    print("DRY-RUN PASSED: segmentation-aware loss backpropagated to generator through frozen U-Net.")
    print(f"Dry-run seg-aware loss: {float(loss.detach().cpu()):.6f}")


def batch_size_candidates(initial: int) -> list[int]:
    out = []
    for value in (initial, 8, 4, 2):
        if value <= initial and value not in out:
            out.append(value)
    return out or [2]


def main() -> int:
    args = parse_args()
    device = choose_device(args.device)
    print(f"Using device: {device}")
    if args.dry_run:
        dry_run_smoke_test(args, device)
        return 0

    seeds = args.seeds if args.seeds is not None else [args.seed]
    completed: list[Path] = []
    for seed in seeds:
        last_error: RuntimeError | None = None
        for batch_size in batch_size_candidates(args.batch_size):
            try:
                completed.append(train_one_seed(args, seed, batch_size, device))
                break
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                last_error = exc
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"CUDA OOM at batch size {batch_size}; trying fallback.")
        else:
            if last_error is not None:
                raise last_error

    if len(completed) > 1:
        average_checkpoints(completed, MODEL_ROOT / args.run_name / "phaseB_hero_swa.pth", device)
        print(f"Wrote SWA checkpoint from {len(completed)} seeds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
