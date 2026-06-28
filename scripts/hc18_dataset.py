"""
HC18 super-resolution dataset for PyTorch.

This module provides a reusable HC18SRDataset class that loads HR, LR, and mask
images from the processed HC18 manifest and returns normalized tensors.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image


class HC18SRDataset(torch.utils.data.Dataset):
    """PyTorch dataset for HC18 super-resolution data."""

    INPUT_KEYS = {
        "LR_x2": "lr_x2_path",
        "LR_x4": "lr_x4_path",
        "LQ_mixed": "lq_mixed_path",
    }

    def __init__(
        self,
        csv_path: str,
        input_type: str = "LR_x2",
        split: str = "train",
        return_mask: bool = False,
    ):
        """Initialize the HC18SRDataset.

        Args:
            csv_path: Path to hc18_lq_manifest.csv.
            input_type: One of "LR_x2", "LR_x4", "LQ_mixed".
            split: One of "train", "val", "test".
            return_mask: If True, include mask tensor when available.
        """
        self.csv_path = Path(csv_path)
        self.data_root = self.csv_path.parent
        self.input_type = input_type
        self.input_key = self._resolve_input_key(input_type)
        self.split = split
        self.return_mask = return_mask
        self.samples = self._load_manifest()

    def _resolve_input_key(self, input_type: str) -> str:
        if input_type not in self.INPUT_KEYS:
            valid = ", ".join(self.INPUT_KEYS.keys())
            raise ValueError(f"input_type must be one of: {valid}")
        return self.INPUT_KEYS[input_type]

    def _load_manifest(self) -> List[Dict[str, str]]:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.csv_path}")

        samples: List[Dict[str, str]] = []
        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split") != self.split:
                    continue
                samples.append(row)

        if not samples:
            raise ValueError(f"No samples found for split '{self.split}' in {self.csv_path}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        image_id = sample.get("image_id", "")

        hr_path = sample.get("hr_path")
        lq_path = sample.get(self.input_key)
        mask_path = sample.get("mask_path")

        hr = self._load_image(hr_path)
        lq = self._load_image(lq_path)

        if hr is None:
            raise FileNotFoundError(f"HR image missing: {hr_path}")
        if lq is None:
            raise FileNotFoundError(f"LQ image missing: {lq_path}")
        if hr.shape != lq.shape:
            raise ValueError(
                f"Shape mismatch for image_id={image_id}: HR {hr.shape} vs LQ {lq.shape}"
            )

        output: Dict[str, torch.Tensor] = {
            "lq": lq,
            "hr": hr,
            "image_id": image_id,
        }

        if self.return_mask:
            mask = self._load_image(mask_path, keep_dim=False)
            if mask is not None:
                output["mask"] = mask

        return output

    def _load_image(self, path_str: Optional[str], keep_dim: bool = True) -> Optional[torch.Tensor]:
        if not path_str:
            return None

        image_path = self.data_root / path_str
        if not image_path.exists():
            return None

        try:
            img = Image.open(image_path).convert("L")
            array = np.array(img, dtype=np.float32)
        except Exception:
            return None

        tensor = torch.from_numpy(array / 255.0).to(dtype=torch.float32)
        if keep_dim:
            tensor = tensor.unsqueeze(0)
        return tensor


if __name__ == "__main__":
    dataset = HC18SRDataset(
        csv_path="data/processed/HC18_SR/hc18_lq_manifest.csv",
        input_type="LQ_mixed",
        split="train",
        return_mask=True,
    )

    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"image_id: {sample['image_id']}")
    print(f"lq shape: {sample['lq'].shape}")
    print(f"hr shape: {sample['hr'].shape}")
    if "mask" in sample:
        print(f"mask shape: {sample['mask'].shape}")
    else:
        print("mask: not available")
