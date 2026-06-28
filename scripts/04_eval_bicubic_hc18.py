"""
Evaluate degraded HC18 images against HR originals.

Computes PSNR, SSIM, and MAE metrics for:
- LR_x2 (bicubic_x2)
- LR_x4 (bicubic_x4)
- LQ_mixed (lq_mixed_baseline)

Only evaluates samples where split == "test".
"""

import csv
import numpy as np
from pathlib import Path
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Configuration
DATA_ROOT = Path("data/processed/HC18_SR")
MANIFEST_FILE = DATA_ROOT / "hc18_lq_manifest.csv"
RESULTS_DIR = Path("results")
PER_IMAGE_RESULTS = RESULTS_DIR / "bicubic_hc18_per_image.csv"
SUMMARY_RESULTS = RESULTS_DIR / "bicubic_hc18_summary.csv"

# Method names
METHODS = {
    "bicubic_x2": "lr_x2_path",
    "bicubic_x4": "lr_x4_path",
    "lq_mixed_baseline": "lq_mixed_path",
}


def create_results_dir():
    """Create results directory if it doesn't exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_grayscale_image(image_path):
    """Load image in grayscale format.
    
    Args:
        image_path: Path to image relative to DATA_ROOT
    
    Returns:
        NumPy array (H, W) or None if image doesn't exist
    """
    try:
        full_path = DATA_ROOT / image_path
        if full_path.exists():
            img = Image.open(full_path).convert("L")  # Convert to grayscale
            return np.array(img, dtype=np.float32)
        else:
            return None
    except Exception as e:
        print(f"    Error loading {image_path}: {str(e)}")
        return None


def compute_psnr(hr_img, degraded_img):
    """Compute Peak Signal-to-Noise Ratio."""
    return peak_signal_noise_ratio(hr_img, degraded_img, data_range=255.0)


def compute_ssim(hr_img, degraded_img):
    """Compute Structural Similarity Index."""
    return structural_similarity(hr_img, degraded_img, data_range=255.0)


def compute_mae(hr_img, degraded_img):
    """Compute Mean Absolute Error."""
    return np.mean(np.abs(hr_img - degraded_img))


def ensure_same_shape(hr_img, degraded_img):
    """Ensure HR and degraded images have the same shape.
    
    Args:
        hr_img: HR image array
        degraded_img: Degraded image array
    
    Returns:
        Tuple of (hr_img, degraded_img) with same shape, or (None, None) if mismatch
    """
    if hr_img.shape != degraded_img.shape:
        print(f"    Shape mismatch: HR {hr_img.shape} vs Degraded {degraded_img.shape}")
        return None, None
    return hr_img, degraded_img


def evaluate_sample(sample_data):
    """Evaluate a single sample for all methods.
    
    Args:
        sample_data: Dictionary with image paths from manifest
    
    Returns:
        Dictionary with evaluation results or None if evaluation fails
    """
    image_id = sample_data["image_id"]
    
    # Load HR image
    hr_img = load_grayscale_image(sample_data["hr_path"])
    if hr_img is None:
        return None
    
    results = {
        "image_id": image_id,
        "split": sample_data["split"],
    }
    
    # Evaluate each method
    for method_name, img_path_key in METHODS.items():
        degraded_img = load_grayscale_image(sample_data[img_path_key])
        
        if degraded_img is None:
            print(f"  Warning: Could not load {method_name} for {image_id}")
            results[f"{method_name}_psnr"] = None
            results[f"{method_name}_ssim"] = None
            results[f"{method_name}_mae"] = None
            continue
        
        # Ensure same shape
        hr_aligned, degraded_aligned = ensure_same_shape(hr_img, degraded_img)
        if hr_aligned is None:
            results[f"{method_name}_psnr"] = None
            results[f"{method_name}_ssim"] = None
            results[f"{method_name}_mae"] = None
            continue
        
        try:
            # Compute metrics
            psnr = compute_psnr(hr_aligned, degraded_aligned)
            ssim = compute_ssim(hr_aligned, degraded_aligned)
            mae = compute_mae(hr_aligned, degraded_aligned)
            
            results[f"{method_name}_psnr"] = psnr
            results[f"{method_name}_ssim"] = ssim
            results[f"{method_name}_mae"] = mae
        except Exception as e:
            print(f"  Error computing metrics for {method_name}/{image_id}: {str(e)}")
            results[f"{method_name}_psnr"] = None
            results[f"{method_name}_ssim"] = None
            results[f"{method_name}_mae"] = None
    
    return results


def main():
    """Main evaluation pipeline."""
    # Create results directory
    create_results_dir()
    
    # Read manifest
    print(f"Reading manifest: {MANIFEST_FILE}")
    manifest_data = []
    with open(MANIFEST_FILE, "r") as f:
        reader = csv.DictReader(f)
        manifest_data = list(reader)
    print(f"  Total samples: {len(manifest_data)}")
    
    # Filter for test split
    test_samples = [s for s in manifest_data if s["split"] == "test"]
    print(f"  Test samples: {len(test_samples)}")
    
    # Evaluate all test samples
    print(f"\nEvaluating {len(test_samples)} test samples...")
    per_image_results = []
    
    for idx, sample in enumerate(test_samples):
        image_id = sample["image_id"]
        if (idx + 1) % 50 == 0 or idx == len(test_samples) - 1:
            print(f"  Processed {idx + 1}/{len(test_samples)} samples")
        
        results = evaluate_sample(sample)
        if results is not None:
            per_image_results.append(results)
    
    print(f"  Successfully evaluated: {len(per_image_results)} samples")
    
    # Save per-image results
    if per_image_results:
        fieldnames = ["image_id", "split"]
        for method_name in METHODS.keys():
            fieldnames.extend([
                f"{method_name}_psnr",
                f"{method_name}_ssim",
                f"{method_name}_mae",
            ])
        
        with open(PER_IMAGE_RESULTS, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_image_results)
        print(f"\nPer-image results saved: {PER_IMAGE_RESULTS}")
    
    # Compute summary statistics
    print("\nComputing summary statistics...")
    summary_data = []
    
    for method_name in METHODS.keys():
        # Extract metric values for this method
        psnr_values = [
            r[f"{method_name}_psnr"] for r in per_image_results
            if r[f"{method_name}_psnr"] is not None
        ]
        ssim_values = [
            r[f"{method_name}_ssim"] for r in per_image_results
            if r[f"{method_name}_ssim"] is not None
        ]
        mae_values = [
            r[f"{method_name}_mae"] for r in per_image_results
            if r[f"{method_name}_mae"] is not None
        ]
        
        # Compute mean and std
        summary = {
            "method": method_name,
            "mean_psnr": np.mean(psnr_values) if psnr_values else None,
            "std_psnr": np.std(psnr_values) if psnr_values else None,
            "mean_ssim": np.mean(ssim_values) if ssim_values else None,
            "std_ssim": np.std(ssim_values) if ssim_values else None,
            "mean_mae": np.mean(mae_values) if mae_values else None,
            "std_mae": np.std(mae_values) if mae_values else None,
        }
        summary_data.append(summary)
    
    # Save summary results
    if summary_data:
        with open(SUMMARY_RESULTS, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "method",
                    "mean_psnr",
                    "std_psnr",
                    "mean_ssim",
                    "std_ssim",
                    "mean_mae",
                    "std_mae",
                ],
            )
            writer.writeheader()
            writer.writerows(summary_data)
        print(f"Summary results saved: {SUMMARY_RESULTS}")
    
    # Print summary table
    print("\n" + "=" * 95)
    print("EVALUATION SUMMARY")
    print("=" * 95)
    print(f"{'Method':<25} {'PSNR':>15} {'SSIM':>15} {'MAE':>15}")
    print(f"{'':25} {'Mean ± Std':>15} {'Mean ± Std':>15} {'Mean ± Std':>15}")
    print("-" * 95)
    
    for summary in summary_data:
        method = summary["method"]
        psnr_str = (
            f"{summary['mean_psnr']:.2f} ± {summary['std_psnr']:.2f}"
            if summary["mean_psnr"] is not None
            else "N/A"
        )
        ssim_str = (
            f"{summary['mean_ssim']:.4f} ± {summary['std_ssim']:.4f}"
            if summary["mean_ssim"] is not None
            else "N/A"
        )
        mae_str = (
            f"{summary['mean_mae']:.2f} ± {summary['std_mae']:.2f}"
            if summary["mean_mae"] is not None
            else "N/A"
        )
        print(f"{method:<25} {psnr_str:>15} {ssim_str:>15} {mae_str:>15}")
    
    print("=" * 95)


if __name__ == "__main__":
    main()
