"""
Visual comparison script for HC18 super-resolution data.

Creates comparison figures showing HR, LR_x2, LR_x4, LQ_mixed, and mask
for 5 random samples to verify that image degradation is working correctly.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
DATA_ROOT = Path("data/processed/HC18_SR")
MANIFEST_FILE = DATA_ROOT / "hc18_lq_manifest.csv"
OUTPUT_DIR = Path("results/visual_check")
NUM_SAMPLES = 5


def create_output_dir():
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_manifest():
    """Read the LQ manifest CSV file."""
    manifest_data = []
    with open(MANIFEST_FILE, "r") as f:
        reader = csv.DictReader(f)
        manifest_data = list(reader)
    return manifest_data


def load_image(image_path):
    """Load image from path. Returns None if image doesn't exist or can't be read."""
    try:
        full_path = DATA_ROOT / image_path
        if full_path.exists():
            img = Image.open(full_path).convert("RGB")
            return np.array(img)
        else:
            return None
    except Exception as e:
        print(f"    Error loading image {image_path}: {str(e)}")
        return None


def create_comparison_figure(sample_data):
    """Create a comparison figure for a single sample.
    
    Args:
        sample_data: Dictionary with image_id and path information from manifest
    
    Returns:
        Figure object or None if it fails
    """
    image_id = sample_data["image_id"]
    
    # Load all images
    hr_img = load_image(sample_data["hr_path"])
    lr_x2_img = load_image(sample_data["lr_x2_path"])
    lr_x4_img = load_image(sample_data["lr_x4_path"])
    lq_mixed_img = load_image(sample_data["lq_mixed_path"])
    mask_img = load_image(sample_data["mask_path"])
    
    # Check if critical images are available
    if hr_img is None or lr_x2_img is None or lr_x4_img is None or lq_mixed_img is None:
        print(f"  Skipping {image_id}: Missing critical image(s)")
        return None
    
    # Create figure with 5 subplots
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(f"Sample: {image_id} (Split: {sample_data['split']})", fontsize=14, fontweight="bold")
    
    # Convert RGB images to grayscale for ultrasound display
    hr_gray = np.mean(hr_img, axis=2)
    lr_x2_gray = np.mean(lr_x2_img, axis=2)
    lr_x4_gray = np.mean(lr_x4_img, axis=2)
    lq_mixed_gray = np.mean(lq_mixed_img, axis=2)
    
    # Plot HR
    axes[0].imshow(hr_gray, cmap="gray")
    axes[0].set_title("HR Original", fontweight="bold")
    axes[0].axis("off")
    
    # Plot LR_x2
    axes[1].imshow(lr_x2_gray, cmap="gray")
    axes[1].set_title("LR_x2\n(2x downsampled)", fontweight="bold")
    axes[1].axis("off")
    
    # Plot LR_x4
    axes[2].imshow(lr_x4_gray, cmap="gray")
    axes[2].set_title("LR_x4\n(4x downsampled)", fontweight="bold")
    axes[2].axis("off")
    
    # Plot LQ_mixed
    axes[3].imshow(lq_mixed_gray, cmap="gray")
    axes[3].set_title("LQ_mixed\n(degraded LR_x4)", fontweight="bold")
    axes[3].axis("off")
    
    # Plot mask (if available)
    if mask_img is not None:
        mask_gray = np.mean(mask_img, axis=2) if len(mask_img.shape) == 3 else mask_img
        axes[4].imshow(mask_gray, cmap="gray")
        axes[4].set_title("Mask", fontweight="bold")
    else:
        axes[4].text(0.5, 0.5, "Mask not available", 
                     ha="center", va="center", fontsize=12, color="red")
        axes[4].set_title("Mask", fontweight="bold")
    axes[4].axis("off")
    
    plt.tight_layout()
    return fig


def main():
    """Main processing pipeline."""
    # Create output directory
    create_output_dir()
    
    # Read manifest
    print(f"Reading manifest: {MANIFEST_FILE}")
    manifest_data = read_manifest()
    print(f"  Total samples in manifest: {len(manifest_data)}")
    
    # Select 5 random samples with reproducible seed
    if len(manifest_data) < NUM_SAMPLES:
        print(f"  Warning: Only {len(manifest_data)} samples available, using all")
        selected_indices = list(range(len(manifest_data)))
    else:
        selected_indices = np.random.choice(len(manifest_data), NUM_SAMPLES, replace=False)
    
    selected_samples = [manifest_data[i] for i in sorted(selected_indices)]
    
    # Generate figures for selected samples
    print(f"\nGenerating comparison figures for {len(selected_samples)} random samples...")
    saved_count = 0
    
    for idx, sample in enumerate(selected_samples):
        image_id = sample["image_id"]
        print(f"  ({idx + 1}/{len(selected_samples)}) Processing {image_id}...", end=" ")
        
        fig = create_comparison_figure(sample)
        
        if fig is not None:
            output_path = OUTPUT_DIR / f"sample_{image_id}.png"
            fig.savefig(output_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved")
            print(f"    -> {output_path}")
            saved_count += 1
        else:
            plt.close("all")
            print("Skipped")
    
    # Print summary
    print("\n" + "=" * 70)
    print("VISUAL CHECK SUMMARY")
    print("=" * 70)
    print(f"Figures saved: {saved_count}/{len(selected_samples)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
