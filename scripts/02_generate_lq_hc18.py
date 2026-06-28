"""
Generate low-quality (LQ) fetal ultrasound images from HR images.

Generates:
- LR_x2: Downsampled by 2x, upsampled with bicubic interpolation
- LR_x4: Downsampled by 4x, upsampled with bicubic interpolation
- LQ_mixed: LR_x4 with Gaussian blur, speckle noise, motion blur, and reduced contrast
"""

import os
import csv
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import cv2

# Set reproducible seed
np.random.seed(42)

# Configuration
DATA_ROOT = Path("data/processed/HC18_SR")
HR_DIR = DATA_ROOT / "HR"
MANIFEST_FILE = DATA_ROOT / "hc18_manifest.csv"

OUTPUT_DIRS = {
    "lr_x2": DATA_ROOT / "LR_x2",
    "lr_x4": DATA_ROOT / "LR_x4",
    "lq_mixed": DATA_ROOT / "LQ_mixed",
}

OUTPUT_MANIFEST = DATA_ROOT / "hc18_lq_manifest.csv"

TARGET_SIZE = (256, 256)


def ensure_output_dirs():
    """Create output directories if they don't exist."""
    for dir_path in OUTPUT_DIRS.values():
        dir_path.mkdir(parents=True, exist_ok=True)


def downsample_and_upsample(img_array, scale_factor):
    """Downsample and upsample image using bicubic interpolation.
    
    Args:
        img_array: NumPy array of image (H, W, C)
        scale_factor: Downsampling factor (2 or 4)
    
    Returns:
        Upsampled image array
    """
    h, w = img_array.shape[:2]
    new_h, new_w = h // scale_factor, w // scale_factor
    
    # Downsample
    img_pil = Image.fromarray(img_array)
    img_downsampled = img_pil.resize((new_w, new_h), Image.BICUBIC)
    
    # Upsample back to original size with bicubic
    img_upsampled = img_downsampled.resize((w, h), Image.BICUBIC)
    
    return np.array(img_upsampled)


def add_gaussian_blur(img_array, radius=1.5):
    """Add Gaussian blur."""
    img_pil = Image.fromarray(img_array)
    img_blurred = img_pil.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(img_blurred)


def add_speckle_noise(img_array, noise_sigma=0.02):
    """Add speckle noise (multiplicative noise)."""
    noise = 1 + noise_sigma * np.random.randn(*img_array.shape)
    noisy = np.clip(img_array.astype(float) * noise, 0, 255).astype(np.uint8)
    return noisy


def add_motion_blur(img_array, kernel_size=5):
    """Add horizontal motion blur."""
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Create horizontal motion blur kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
    kernel = kernel / (kernel_size)
    
    # Apply motion blur
    img_blurred = cv2.filter2D(img_cv, -1, kernel)
    
    img_rgb = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2RGB)
    return img_rgb


def reduce_contrast(img_array, factor=0.85):
    """Reduce contrast slightly."""
    img_pil = Image.fromarray(img_array)
    enhancer = ImageEnhance.Contrast(img_pil)
    img_reduced = enhancer.enhance(factor)
    return np.array(img_reduced)


def generate_lq_images(hr_image_path, image_id):
    """Generate LQ images from HR image.
    
    Args:
        hr_image_path: Path to HR image
        image_id: Image ID for output naming
    
    Returns:
        Dictionary with paths to generated images
    """
    # Read HR image
    hr_img = Image.open(hr_image_path).convert("RGB")
    hr_array = np.array(hr_img)
    
    # LR_x2: Downsample by 2x, upsample with bicubic
    lr_x2_array = downsample_and_upsample(hr_array, scale_factor=2)
    lr_x2_img = Image.fromarray(lr_x2_array.astype(np.uint8))
    lr_x2_path = OUTPUT_DIRS["lr_x2"] / f"{image_id}.png"
    lr_x2_img.save(lr_x2_path)
    
    # LR_x4: Downsample by 4x, upsample with bicubic
    lr_x4_array = downsample_and_upsample(hr_array, scale_factor=4)
    lr_x4_img = Image.fromarray(lr_x4_array.astype(np.uint8))
    lr_x4_path = OUTPUT_DIRS["lr_x4"] / f"{image_id}.png"
    lr_x4_img.save(lr_x4_path)
    
    # LQ_mixed: Start from LR_x4, add blur, speckle noise, motion blur, reduce contrast
    lq_array = lr_x4_array.astype(np.uint8)
    lq_array = add_gaussian_blur(lq_array, radius=1.5)
    lq_array = add_speckle_noise(lq_array, noise_sigma=0.02)
    lq_array = add_motion_blur(lq_array, kernel_size=5)
    lq_array = reduce_contrast(lq_array, factor=0.85)
    lq_img = Image.fromarray(np.clip(lq_array, 0, 255).astype(np.uint8))
    lq_path = OUTPUT_DIRS["lq_mixed"] / f"{image_id}.png"
    lq_img.save(lq_path)
    
    return {
        "lr_x2": lr_x2_path,
        "lr_x4": lr_x4_path,
        "lq_mixed": lq_path,
    }


def main():
    """Main processing pipeline."""
    # Create output directories
    ensure_output_dirs()
    
    # Read manifest
    manifest_data = []
    with open(MANIFEST_FILE, "r") as f:
        reader = csv.DictReader(f)
        manifest_data = list(reader)
    
    print(f"Processing {len(manifest_data)} images from manifest...")
    
    # Store output manifest data
    output_manifest = []
    
    # Process each image
    for idx, row in enumerate(manifest_data):
        image_id = row["image_id"]
        split = row["split"]
        hr_path = row["hr_path"]
        mask_path = row["mask_path"]
        
        full_hr_path = DATA_ROOT / hr_path
        
        # Check if HR image exists
        if not full_hr_path.exists():
            print(f"  WARNING: HR image not found: {full_hr_path}")
            continue
        
        try:
            # Generate LQ images
            lq_paths = generate_lq_images(full_hr_path, image_id)
            
            # Create relative paths for manifest
            lr_x2_rel = str(lq_paths["lr_x2"].relative_to(DATA_ROOT))
            lr_x4_rel = str(lq_paths["lr_x4"].relative_to(DATA_ROOT))
            lq_mixed_rel = str(lq_paths["lq_mixed"].relative_to(DATA_ROOT))
            
            # Add to output manifest
            output_manifest.append({
                "image_id": image_id,
                "split": split,
                "hr_path": hr_path,
                "lr_x2_path": lr_x2_rel,
                "lr_x4_path": lr_x4_rel,
                "lq_mixed_path": lq_mixed_rel,
                "mask_path": mask_path,
            })
            
            if (idx + 1) % 50 == 0 or idx == len(manifest_data) - 1:
                print(f"  Processed {idx + 1}/{len(manifest_data)} images")
        
        except Exception as e:
            print(f"  ERROR processing {image_id}: {str(e)}")
    
    # Save output manifest
    if output_manifest:
        with open(OUTPUT_MANIFEST, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "image_id",
                    "split",
                    "hr_path",
                    "lr_x2_path",
                    "lr_x4_path",
                    "lq_mixed_path",
                    "mask_path",
                ],
            )
            writer.writeheader()
            writer.writerows(output_manifest)
        print(f"\nManifest saved: {OUTPUT_MANIFEST}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("OUTPUT SUMMARY")
    print("=" * 60)
    print(f"Total images processed: {len(output_manifest)}")
    for name, dir_path in OUTPUT_DIRS.items():
        num_files = len(list(dir_path.glob("*.png")))
        print(f"  {name}: {dir_path} ({num_files} files)")
    print(f"  Manifest: {OUTPUT_MANIFEST}")
    print("=" * 60)


if __name__ == "__main__":
    main()
