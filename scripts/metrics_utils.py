"""Utility functions for SR evaluation metrics.

LPIPS, NIQE, and BRISQUE are computed through pyiqa when that package is
available. Images are normalized to [0, 1] and grayscale tensors are repeated to
RGB for metrics that expect three channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

try:
    from pyiqa import create_metric
except Exception:  # pragma: no cover - optional dependency guard
    create_metric = None


def _to_2d_numpy(image: Any) -> np.ndarray:
    """Convert a tensor or array to a normalized 2D float array in [0, 1]."""
    if isinstance(image, torch.Tensor):
        array = image.detach().cpu().float().numpy()
    else:
        array = np.asarray(image, dtype=np.float32)

    array = np.squeeze(array)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]

    if array.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {array.shape}")

    return np.clip(array, 0.0, 1.0).astype(np.float32)


def _to_3channel_tensor(image: Any, device: torch.device) -> torch.Tensor:
    """Convert a grayscale image to a 3-channel tensor for perceptual/no-reference metrics."""
    image_2d = _to_2d_numpy(image)
    image_rgb = np.repeat(image_2d[..., None], 3, axis=2)
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    return tensor


def get_pyiqa_metric(name: str, device: torch.device, *, as_loss: bool = False) -> Optional[Any]:
    """Create a pyiqa metric, returning None if the optional dependency is unavailable."""
    if create_metric is None:
        return None
    try:
        return create_metric(name, device=device, as_loss=as_loss)
    except TypeError:
        try:
            return create_metric(name, device=device)
        except Exception:
            return None
    except Exception:
        return None


@dataclass
class PyiqaMetricBundle:
    """Cache pyiqa metrics so evaluation does not rebuild networks per image."""

    device: torch.device
    lpips: Optional[Any] = None
    niqe: Optional[Any] = None
    brisque: Optional[Any] = None

    @classmethod
    def create(cls, device: torch.device) -> "PyiqaMetricBundle":
        return cls(
            device=device,
            lpips=get_pyiqa_metric("lpips", device),
            niqe=get_pyiqa_metric("niqe", device),
            brisque=get_pyiqa_metric("brisque", device),
        )


def compute_psnr(output: Any, target: Any) -> float:
    try:
        output_np = _to_2d_numpy(output)
        target_np = _to_2d_numpy(target)
        return float(peak_signal_noise_ratio(target_np, output_np, data_range=1.0))
    except Exception:
        return float("nan")


def compute_ssim(output: Any, target: Any) -> float:
    try:
        output_np = _to_2d_numpy(output)
        target_np = _to_2d_numpy(target)
        min_side = min(output_np.shape)
        win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
        if win_size < 3:
            return float("nan")
        return float(structural_similarity(target_np, output_np, data_range=1.0, win_size=win_size))
    except Exception:
        return float("nan")


def compute_mae(output: Any, target: Any) -> float:
    try:
        output_np = _to_2d_numpy(output)
        target_np = _to_2d_numpy(target)
        return float(np.mean(np.abs(target_np - output_np)))
    except Exception:
        return float("nan")


def compute_lpips(output: Any, target: Any, device: torch.device, metric: Optional[Any] = None) -> float:
    """Compute full-reference LPIPS with pyiqa; lower is better."""
    if metric is None:
        metric = get_pyiqa_metric("lpips", device)
    if metric is None:
        return float("nan")

    try:
        with torch.no_grad():
            output_tensor = _to_3channel_tensor(output, device)
            target_tensor = _to_3channel_tensor(target, device)
            score = metric(output_tensor, target_tensor)
            if isinstance(score, (tuple, list)):
                score = score[0]
            if torch.is_tensor(score):
                return float(score.detach().cpu().float().mean().item())
            return float(score)
    except Exception:
        return float("nan")


def compute_niqe(image: Any, device: torch.device, metric: Optional[Any] = None) -> float:
    if metric is None:
        metric = get_pyiqa_metric("niqe", device)
    if metric is None:
        return float("nan")

    try:
        with torch.no_grad():
            tensor = _to_3channel_tensor(image, device)
            score = metric(tensor)
            if isinstance(score, (tuple, list)):
                score = score[0]
            if torch.is_tensor(score):
                return float(score.detach().cpu().float().mean().item())
            return float(score)
    except Exception:
        return float("nan")


def compute_brisque(image: Any, device: torch.device, metric: Optional[Any] = None) -> float:
    if metric is None:
        metric = get_pyiqa_metric("brisque", device)
    if metric is None:
        return float("nan")

    try:
        with torch.no_grad():
            tensor = _to_3channel_tensor(image, device)
            score = metric(tensor)
            if isinstance(score, (tuple, list)):
                score = score[0]
            if torch.is_tensor(score):
                return float(score.detach().cpu().float().mean().item())
            return float(score)
    except Exception:
        return float("nan")


def compute_edge_score(output: Any, target: Any) -> float:
    try:
        output_np = _to_2d_numpy(output)
        target_np = _to_2d_numpy(target)
        output_dx = np.abs(np.gradient(output_np, axis=0)) + np.abs(np.gradient(output_np, axis=1))
        target_dx = np.abs(np.gradient(target_np, axis=0)) + np.abs(np.gradient(target_np, axis=1))
        return float(np.clip(1.0 - np.mean(np.abs(output_dx - target_dx)), 0.0, 1.0))
    except Exception:
        return float("nan")
