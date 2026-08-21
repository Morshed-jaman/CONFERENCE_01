"""Download OSEDiff's README-documented dependent models."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


def main() -> int:
    hf_home = Path(os.environ["HF_HOME"])
    print(f"HF_HOME={hf_home}", flush=True)

    sd_path = snapshot_download(
        repo_id="Manojb/stable-diffusion-2-1-base",
        max_workers=1,
        allow_patterns=[
            "*.json",
            "*.txt",
            "*.safetensors",
            "*.bin",
            "tokenizer/**",
            "text_encoder/**",
            "vae/**",
            "unet/**",
            "scheduler/**",
            "feature_extractor/**",
        ],
    )
    print(f"SD21_PATH={sd_path}", flush=True)

    ram_path = hf_hub_download(
        repo_id="xinyu1205/recognize-anything",
        repo_type="space",
        filename="ram_swin_large_14m.pth",
    )
    print(f"RAM_PATH={ram_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
