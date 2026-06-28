"""Generate quality-aware CNN + L1/MSE guide images for every HC18 split.

CNN-guided residual diffusion needs a CNN SR estimate for each training and
validation image. This script loads the exact quality-aware CNN architecture
from script 07 and writes grayscale guides for train, val, and test without
deleting the existing output directory.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from hc18_dataset import HC18SRDataset


MANIFEST_PATH = Path("data/processed/HC18_SR/hc18_lq_manifest.csv")
DEFAULT_CHECKPOINT_PATH = Path(
    "results/models/quality_aware_sr_hc18_l1_mse_best.pth"
)
OUTPUT_DIR = Path("results/sr_outputs/quality_aware_l1_mse")
SPLITS = ("train", "val", "test")


def load_training_module(script_path: Path):
    """Import script 07 so architecture and quality features match training."""
    spec = importlib.util.spec_from_file_location(
        "train_quality_aware_sr_hc18",
        str(script_path),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load training module from: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model(training_module, checkpoint_path: Path, device: torch.device):
    """Load the checkpoint strictly into the exact script-07 CNN architecture."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model = training_module.QualityAwareSRModel()
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "The quality-aware L1/MSE checkpoint does not match "
            "QualityAwareSRModel from scripts/07_train_quality_aware_sr_hc18.py."
        ) from error

    model = model.to(device)
    model.eval()
    return model


def normalize_quality_features(
    quality_features: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> torch.Tensor:
    """Apply the training-split feature normalization used by script 07."""
    mean = feature_mean.to(
        device=quality_features.device,
        dtype=quality_features.dtype,
    ).view(1, -1)
    std = feature_std.to(
        device=quality_features.device,
        dtype=quality_features.dtype,
    ).view(1, -1)
    return (quality_features - mean) / std


def save_png_safely(output_path: Path, image: np.ndarray) -> None:
    """Atomically create or replace one 0-255 grayscale PNG."""
    image_uint8 = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    if image_uint8.ndim != 2:
        raise ValueError(
            f"Expected a 2D grayscale output for {output_path.name}, "
            f"got shape {image_uint8.shape}"
        )

    temporary_path = output_path.with_name(f".{output_path.stem}.tmp.png")
    try:
        Image.fromarray(image_uint8, mode="L").save(temporary_path, format="PNG")
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def generate_split(
    split: str,
    model: torch.nn.Module,
    training_module,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> int:
    """Generate CNN SR guides for every image in one manifest split."""
    dataset = HC18SRDataset(
        csv_path=str(MANIFEST_PATH),
        input_type="LQ_mixed",
        split=split,
        return_mask=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    generated_count = 0
    with torch.inference_mode():
        for batch in dataloader:
            # HC18SRDataset already loads grayscale LQ_mixed as float32 in [0, 1].
            lq = batch["lq"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=device.type == "cuda",
            )
            quality = training_module.compute_quality_features(lq)
            quality = normalize_quality_features(
                quality,
                feature_mean,
                feature_std,
            )
            sr = model(lq, quality).clamp(0.0, 1.0)

            image_ids = batch["image_id"]
            for batch_index, image_id in enumerate(image_ids):
                sr_image = np.squeeze(
                    sr[batch_index].detach().cpu().numpy()
                ).astype(np.float32)
                save_png_safely(OUTPUT_DIR / f"{image_id}.png", sr_image)
                generated_count += 1

    return generated_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate quality-aware L1/MSE CNN guides for all HC18 splits"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Inference batch size (default: 16)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="Quality-aware CNN + L1/MSE checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be greater than zero")
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    training_module = load_training_module(
        SCRIPT_DIR / "07_train_quality_aware_sr_hc18.py"
    )
    model = load_model(
        training_module,
        Path(args.checkpoint_path),
        device,
    )

    # Script 07 normalized FiLM features with statistics from the train split.
    # Recompute those exact statistics once and reuse them for every split.
    normalization_dataset = HC18SRDataset(
        csv_path=str(MANIFEST_PATH),
        input_type="LQ_mixed",
        split="train",
        return_mask=False,
    )
    feature_mean, feature_std = training_module.compute_normalization_stats(
        normalization_dataset
    )

    # Preserve the directory and atomically update individual image_id PNGs.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generated_counts = {}
    for split in SPLITS:
        generated_counts[split] = generate_split(
            split,
            model,
            training_module,
            feature_mean,
            feature_std,
            device,
            args.batch_size,
        )
        print(f"Generated {split} images: {generated_counts[split]}")

    total_png_files = sum(1 for path in OUTPUT_DIR.glob("*.png") if path.is_file())
    print(f"Total PNG files in {OUTPUT_DIR}: {total_png_files}")


if __name__ == "__main__":
    main()
