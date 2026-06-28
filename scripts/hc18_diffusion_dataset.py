"""HC18 diffusion dataset for conditional super-resolution training."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch


class HC18DiffusionDataset(torch.utils.data.Dataset):
    """PyTorch dataset for DMUISR-style conditional diffusion super-resolution."""

    INPUT_KEYS = {
        "LQ_mixed": "lq_mixed_path",
        "LR_x2": "lr_x2_path",
        "LR_x4": "lr_x4_path",
    }

    def __init__(
        self,
        csv_path: str = "data/processed/HC18_SR/hc18_lq_manifest.csv",
        split: str = "train",
        image_size: int = 128,
        input_type: str = "LQ_mixed",
        return_quality: bool = True,
    ):
        self.csv_path = Path(csv_path)
        self.data_root = self.csv_path.parent
        self.split = split
        self.image_size = int(image_size)
        self.input_type = input_type
        self.return_quality = return_quality
        self.input_key = self._resolve_input_key(input_type)
        self.samples = self._load_manifest()

    def _resolve_input_key(self, input_type: str) -> str:
        if input_type not in self.INPUT_KEYS:
            valid = ", ".join(self.INPUT_KEYS.keys())
            raise ValueError(f"input_type must be one of: {valid}")
        return self.INPUT_KEYS[input_type]

    def _load_manifest(self):
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.csv_path}")

        manifest = pd.read_csv(self.csv_path)
        samples = manifest[manifest["split"] == self.split].to_dict(orient="records")
        if len(samples) == 0:
            raise ValueError(f"No samples found for split '{self.split}' in {self.csv_path}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_id = str(sample.get("image_id", ""))

        lq_path = sample.get(self.input_key)
        hr_path = sample.get("hr_path")

        lq = self._load_image(lq_path)
        hr = self._load_image(hr_path)

        if lq is None:
            raise FileNotFoundError(f"LQ image missing: {lq_path}")
        if hr is None:
            raise FileNotFoundError(f"HR image missing: {hr_path}")

        # Use the low-quality mixed image as the condition and the HR image as the target.
        output = {
            "lq": lq,
            "hr": hr,
            "image_id": image_id,
        }

        if self.return_quality:
            quality = self._compute_quality_features(lq)
            output["quality"] = quality

        return output

    def _load_image(self, path_str, keep_dim: bool = True):
        if not path_str or not isinstance(path_str, str):
            return None

        image_path = Path(path_str)
        if not image_path.is_absolute():
            image_path = self.data_root / image_path

        if not image_path.exists():
            return None

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None

        resized = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)
        if keep_dim:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _compute_quality_features(self, lq_tensor: torch.Tensor):
        # Compute quality features from the low-quality condition image only.
        lq_image = lq_tensor.squeeze(0).cpu().numpy()
        lq_np = np.asarray(lq_image, dtype=np.float64)

        laplacian = cv2.Laplacian(lq_np, cv2.CV_64F)
        laplacian_variance = float(np.var(laplacian))
        blur_score = np.log1p(laplacian_variance) / 10.0

        contrast_score = float(np.std(lq_np))
        brightness_mean = float(np.mean(lq_np))

        blurred = cv2.blur(lq_np, ksize=(5, 5))
        noise_proxy = float(np.mean(np.abs(lq_np - blurred)))

        quality = np.array(
            [blur_score, contrast_score, brightness_mean, noise_proxy, 1.0],
            dtype=np.float32,
        )
        return torch.from_numpy(quality)


if __name__ == "__main__":
    train_dataset = HC18DiffusionDataset(
        csv_path="data/processed/HC18_SR/hc18_lq_manifest.csv",
        split="train",
        image_size=128,
        input_type="LQ_mixed",
        return_quality=True,
    )
    val_dataset = HC18DiffusionDataset(
        csv_path="data/processed/HC18_SR/hc18_lq_manifest.csv",
        split="val",
        image_size=128,
        input_type="LQ_mixed",
        return_quality=True,
    )
    test_dataset = HC18DiffusionDataset(
        csv_path="data/processed/HC18_SR/hc18_lq_manifest.csv",
        split="test",
        image_size=128,
        input_type="LQ_mixed",
        return_quality=True,
    )

    print(f"Train length: {len(train_dataset)}")
    print(f"Val length: {len(val_dataset)}")
    print(f"Test length: {len(test_dataset)}")

    sample = train_dataset[0]
    print(f"lq shape: {sample['lq'].shape}")
    print(f"hr shape: {sample['hr'].shape}")
    print(f"quality shape: {sample['quality'].shape}")
    print(f"lq range: {sample['lq'].min().item()} to {sample['lq'].max().item()}")
    print(f"hr range: {sample['hr'].min().item()} to {sample['hr'].max().item()}")
    print(f"image_id: {sample['image_id']}")
