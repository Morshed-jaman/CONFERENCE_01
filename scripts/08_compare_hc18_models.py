"""
Compare HC18 test-set performance for LQ_mixed, SR-only, and quality-aware SR.

The model architectures are imported from the original training scripts so this
evaluation uses the same definitions that produced the saved checkpoints.
"""

import csv
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader

# Keep all paths relative to the project root. The script is expected to be run
# from the project root with: python .\scripts\08_compare_hc18_models.py
PROJECT_ROOT = Path(".")
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from hc18_dataset import HC18SRDataset


MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "HC18_SR" / "hc18_lq_manifest.csv"
SR_ONLY_MODEL_PATH = PROJECT_ROOT / "results" / "models" / "sr_only_hc18_best.pth"
QUALITY_AWARE_MODEL_PATH = PROJECT_ROOT / "results" / "models" / "quality_aware_sr_hc18_best.pth"
QUALITY_AWARE_L1_MSE_MODEL_PATH = (
    PROJECT_ROOT / "results" / "models" / "quality_aware_sr_hc18_l1_mse_best.pth"
)

PER_IMAGE_CSV = PROJECT_ROOT / "results" / "final_hc18_comparison_per_image.csv"
SUMMARY_CSV = PROJECT_ROOT / "results" / "final_hc18_comparison_summary.csv"
MARKDOWN_TABLE = PROJECT_ROOT / "results" / "final_hc18_comparison_table.md"
QUALITY_AWARE_OUTPUT_DIR = PROJECT_ROOT / "results" / "sr_outputs" / "quality_aware"
QUALITY_AWARE_L1_MSE_OUTPUT_DIR = PROJECT_ROOT / "results" / "sr_outputs" / "quality_aware_l1_mse"
VISUAL_COMPARISON_DIR = PROJECT_ROOT / "results" / "final_visual_comparison"


METHOD_LABELS = {
    "lq_mixed": "LQ_mixed baseline",
    "sr_only": "SR-only",
    "quality_aware": "Quality-aware SR",
    "quality_aware_l1_mse": "Quality-aware SR + L1/MSE",
}


def load_module(module_name: str, script_path: Path):
    """Load a Python script whose filename is not a valid import name."""
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_training_components():
    """Import the exact model classes and quality feature function from training."""
    sr_module = load_module("hc18_sr_only_train", SCRIPT_DIR / "05_train_sr_only_hc18.py")
    qa_module = load_module("hc18_quality_aware_train", SCRIPT_DIR / "07_train_quality_aware_sr_hc18.py")

    return (
        sr_module.ResidualSRModel,
        qa_module.QualityAwareSRModel,
        qa_module.compute_quality_features,
    )


def create_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def extract_state_dict(checkpoint):
    """Support both raw state_dict checkpoints and wrapped checkpoint dicts."""
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def load_model(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(extract_state_dict(checkpoint))
    model.to(device)
    model.eval()
    return model


def tensor_to_2d_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert [1,H,W] or [B,1,H,W] tensors to a clipped 2D float array."""
    array = tensor.detach().cpu().float().numpy()
    array = np.squeeze(array)
    if array.ndim == 3:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image after squeeze, got {array.shape}")
    return np.clip(array, 0.0, 1.0).astype(np.float32)


def save_png(image: np.ndarray, path: Path) -> None:
    """Save a normalized grayscale image as an 8-bit PNG."""
    image_uint8 = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(image_uint8, mode="L").save(path)


def psnr_ssim_mae(output: np.ndarray, hr: np.ndarray) -> Tuple[float, float, float]:
    """Compute standard metrics using 2D arrays to avoid SSIM shape issues."""
    psnr = peak_signal_noise_ratio(hr, output, data_range=1.0)

    # HC18 images are expected to be 256x256, but this keeps SSIM robust.
    min_side = min(output.shape)
    win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
    if win_size < 3:
        raise ValueError(f"Image too small for SSIM: {output.shape}")

    ssim = structural_similarity(hr, output, data_range=1.0, win_size=win_size)
    mae = float(np.mean(np.abs(output - hr)))
    return float(psnr), float(ssim), mae


def sobel_edges(image: torch.Tensor) -> torch.Tensor:
    """Compute a normalized Sobel edge magnitude map for a grayscale image batch."""
    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.dim() == 3:
        image = image.unsqueeze(0)

    if image.dim() != 4 or image.size(1) != 1:
        raise ValueError(f"Expected grayscale image shape [H,W], [1,H,W], or [B,1,H,W], got {tuple(image.shape)}")

    image = image.float()
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=image.dtype,
        device=image.device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=image.dtype,
        device=image.device,
    ).view(1, 1, 3, 3)

    grad_x = F.conv2d(image, sobel_x, padding=1)
    grad_y = F.conv2d(image, sobel_y, padding=1)
    edges = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-12)

    # Normalize each edge map before comparison so the score is scale-stable.
    flat = edges.flatten(start_dim=1)
    max_values = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return edges / max_values.clamp_min(1e-8)


def edge_similarity(output: torch.Tensor, hr: torch.Tensor) -> float:
    """Return an edge similarity score where higher is better."""
    output_edges = sobel_edges(output)
    hr_edges = sobel_edges(hr)
    edge_mae = torch.abs(output_edges - hr_edges).mean()
    return float((1.0 - edge_mae).clamp(0.0, 1.0).item())


def compute_quality_normalization_stats(
    dataset: HC18SRDataset,
    compute_quality_features,
    device: torch.device,
    batch_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute feature normalization from the test split only.

    The quality-aware training script saves only model weights, not the feature
    mean/std tensors, so this script keeps the requested test-only scope.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    feature_batches: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            lq = batch["lq"].to(device)
            features = compute_quality_features(lq)
            if features.dim() != 2 or features.shape[1] != 5:
                raise ValueError(f"Expected quality features [B,5], got {tuple(features.shape)}")
            feature_batches.append(features.detach().cpu())

    stacked = torch.cat(feature_batches, dim=0)
    mean = stacked.mean(dim=0).to(device)
    std = stacked.std(dim=0).to(device)
    std = torch.where(std == 0.0, torch.ones_like(std), std)
    return mean.view(1, -1), std.view(1, -1)


def evaluate_metrics(output: torch.Tensor, hr: torch.Tensor) -> Dict[str, float]:
    output_np = tensor_to_2d_numpy(output)
    hr_np = tensor_to_2d_numpy(hr)
    psnr, ssim, mae = psnr_ssim_mae(output_np, hr_np)
    return {
        "psnr": psnr,
        "ssim": ssim,
        "mae": mae,
        "edge_score": edge_similarity(output, hr),
    }


def add_metric_rows(
    rows: List[Dict[str, object]],
    image_id: str,
    method_key: str,
    metrics: Dict[str, float],
) -> None:
    rows.append(
        {
            "image_id": image_id,
            "method": METHOD_LABELS[method_key],
            "psnr": metrics["psnr"],
            "ssim": metrics["ssim"],
            "mae": metrics["mae"],
            "edge_score": metrics["edge_score"],
        }
    )


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    summary_rows: List[Dict[str, object]] = []
    for method in METHOD_LABELS.values():
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue

        summary_rows.append(
            {
                "method": method,
                "mean_psnr": float(np.mean([row["psnr"] for row in method_rows])),
                "std_psnr": float(np.std([row["psnr"] for row in method_rows])),
                "mean_ssim": float(np.mean([row["ssim"] for row in method_rows])),
                "std_ssim": float(np.std([row["ssim"] for row in method_rows])),
                "mean_mae": float(np.mean([row["mae"] for row in method_rows])),
                "std_mae": float(np.std([row["mae"] for row in method_rows])),
                "mean_edge_score": float(np.mean([row["edge_score"] for row in method_rows])),
                "std_edge_score": float(np.std([row["edge_score"] for row in method_rows])),
            }
        )
    return summary_rows


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_float(value: object, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def build_markdown_table(summary_rows: List[Dict[str, object]]) -> str:
    header = (
        "| method | mean_psnr | std_psnr | mean_ssim | std_ssim | "
        "mean_mae | std_mae | mean_edge_score |\n"
    )
    separator = "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    body = ""
    for row in summary_rows:
        body += (
            f"| {row['method']} | "
            f"{format_float(row['mean_psnr'], 4)} | "
            f"{format_float(row['std_psnr'], 4)} | "
            f"{format_float(row['mean_ssim'], 4)} | "
            f"{format_float(row['std_ssim'], 4)} | "
            f"{format_float(row['mean_mae'], 6)} | "
            f"{format_float(row['std_mae'], 6)} | "
            f"{format_float(row['mean_edge_score'], 6)} |\n"
        )
    return header + separator + body


def print_terminal_table(summary_rows: List[Dict[str, object]]) -> None:
    columns = [
        "method",
        "mean_psnr",
        "std_psnr",
        "mean_ssim",
        "std_ssim",
        "mean_mae",
        "std_mae",
        "mean_edge_score",
    ]
    widths = {
        "method": max(len("method"), max(len(str(row["method"])) for row in summary_rows)),
        "mean_psnr": 10,
        "std_psnr": 9,
        "mean_ssim": 10,
        "std_ssim": 9,
        "mean_mae": 10,
        "std_mae": 9,
        "mean_edge_score": 15,
    }

    header = "  ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(
            "  ".join(
                [
                    str(row["method"]).ljust(widths["method"]),
                    format_float(row["mean_psnr"], 4).rjust(widths["mean_psnr"]),
                    format_float(row["std_psnr"], 4).rjust(widths["std_psnr"]),
                    format_float(row["mean_ssim"], 4).rjust(widths["mean_ssim"]),
                    format_float(row["std_ssim"], 4).rjust(widths["std_ssim"]),
                    format_float(row["mean_mae"], 6).rjust(widths["mean_mae"]),
                    format_float(row["std_mae"], 6).rjust(widths["std_mae"]),
                    format_float(row["mean_edge_score"], 6).rjust(widths["mean_edge_score"]),
                ]
            )
        )


def make_panel(image: np.ndarray, title: str, cell_size: int = 256, title_height: int = 24) -> Image.Image:
    """Create one labeled grayscale panel for the visual comparison figure."""
    image_uint8 = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    pil_image = Image.fromarray(image_uint8, mode="L").resize((cell_size, cell_size), Image.BILINEAR)
    panel = Image.new("RGB", (cell_size, cell_size + title_height), "white")
    panel.paste(pil_image.convert("RGB"), (0, title_height))

    draw = ImageDraw.Draw(panel)
    if hasattr(draw, "textlength"):
        text_width = draw.textlength(title)
    else:
        text_width = draw.textbbox((0, 0), title)[2]
    draw.text(((cell_size - text_width) / 2, 5), title, fill="black")
    return panel


def save_visual_comparison(
    image_id: str,
    lq: np.ndarray,
    sr_only: np.ndarray,
    quality_aware: np.ndarray,
    quality_aware_l1_mse: np.ndarray,
    hr: np.ndarray,
    path: Path,
) -> None:
    """Save a 2x4 figure with outputs, target, and absolute error maps."""
    sr_only_error = np.abs(sr_only - hr)
    quality_aware_error = np.abs(quality_aware - hr)
    quality_aware_l1_mse_error = np.abs(quality_aware_l1_mse - hr)

    panels = [
        make_panel(lq, "LQ input"),
        make_panel(sr_only, "SR-only output"),
        make_panel(quality_aware, "Quality-aware SR"),
        make_panel(quality_aware_l1_mse, "QA SR + L1/MSE"),
        make_panel(hr, "HR target"),
        make_panel(sr_only_error, "SR-only abs error"),
        make_panel(quality_aware_error, "QA abs error"),
        make_panel(quality_aware_l1_mse_error, "QA L1/MSE abs error"),
    ]

    cell_width, cell_height = panels[0].size
    padding = 8
    figure = Image.new(
        "RGB",
        (4 * cell_width + 5 * padding, 2 * cell_height + 3 * padding),
        "white",
    )

    for index, panel in enumerate(panels):
        row = index // 4
        col = index % 4
        x = padding + col * (cell_width + padding)
        y = padding + row * (cell_height + padding)
        figure.paste(panel, (x, y))

    # Include image_id in PNG metadata-friendly filename rather than inside the grid.
    figure.save(path)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    create_dirs(
        [
            PROJECT_ROOT / "results",
            QUALITY_AWARE_OUTPUT_DIR,
            QUALITY_AWARE_L1_MSE_OUTPUT_DIR,
            VISUAL_COMPARISON_DIR,
        ]
    )

    ResidualSRModel, QualityAwareSRModel, compute_quality_features = load_training_components()

    test_dataset = HC18SRDataset(
        csv_path=str(MANIFEST_PATH),
        input_type="LQ_mixed",
        split="test",
        return_mask=False,
    )

    sr_only_model = load_model(ResidualSRModel(), SR_ONLY_MODEL_PATH, device)
    quality_aware_model = load_model(QualityAwareSRModel(), QUALITY_AWARE_MODEL_PATH, device)
    quality_aware_l1_mse_model = load_model(
        QualityAwareSRModel(),
        QUALITY_AWARE_L1_MSE_MODEL_PATH,
        device,
    )

    quality_mean, quality_std = compute_quality_normalization_stats(
        test_dataset,
        compute_quality_features,
        device,
    )

    per_image_rows: List[Dict[str, object]] = []
    visual_count = min(10, len(test_dataset))

    with torch.no_grad():
        for index in range(len(test_dataset)):
            sample = test_dataset[index]
            image_id = str(sample["image_id"])

            lq = sample["lq"].unsqueeze(0).to(device)
            hr = sample["hr"].unsqueeze(0).to(device)

            sr_only_output = sr_only_model(lq).clamp(0.0, 1.0)

            quality_features = compute_quality_features(lq)
            mean = quality_mean.to(device=quality_features.device, dtype=quality_features.dtype)
            std = quality_std.to(device=quality_features.device, dtype=quality_features.dtype)
            quality_features = (quality_features - mean) / std
            quality_aware_output = quality_aware_model(lq, quality_features).clamp(0.0, 1.0)
            quality_aware_l1_mse_output = quality_aware_l1_mse_model(
                lq,
                quality_features,
            ).clamp(0.0, 1.0)

            lq_metrics = evaluate_metrics(lq, hr)
            sr_only_metrics = evaluate_metrics(sr_only_output, hr)
            quality_aware_metrics = evaluate_metrics(quality_aware_output, hr)
            quality_aware_l1_mse_metrics = evaluate_metrics(quality_aware_l1_mse_output, hr)

            add_metric_rows(per_image_rows, image_id, "lq_mixed", lq_metrics)
            add_metric_rows(per_image_rows, image_id, "sr_only", sr_only_metrics)
            add_metric_rows(per_image_rows, image_id, "quality_aware", quality_aware_metrics)
            add_metric_rows(
                per_image_rows,
                image_id,
                "quality_aware_l1_mse",
                quality_aware_l1_mse_metrics,
            )

            lq_np = tensor_to_2d_numpy(lq)
            hr_np = tensor_to_2d_numpy(hr)
            sr_only_np = tensor_to_2d_numpy(sr_only_output)
            quality_aware_np = tensor_to_2d_numpy(quality_aware_output)
            quality_aware_l1_mse_np = tensor_to_2d_numpy(quality_aware_l1_mse_output)

            save_png(quality_aware_np, QUALITY_AWARE_OUTPUT_DIR / f"{image_id}.png")
            save_png(
                quality_aware_l1_mse_np,
                QUALITY_AWARE_L1_MSE_OUTPUT_DIR / f"{image_id}.png",
            )

            if index < visual_count:
                save_visual_comparison(
                    image_id=image_id,
                    lq=lq_np,
                    sr_only=sr_only_np,
                    quality_aware=quality_aware_np,
                    quality_aware_l1_mse=quality_aware_l1_mse_np,
                    hr=hr_np,
                    path=VISUAL_COMPARISON_DIR / f"comparison_{index + 1:02d}_{image_id}.png",
                )

    summary_rows = summarize(per_image_rows)

    write_csv(
        PER_IMAGE_CSV,
        per_image_rows,
        ["image_id", "method", "psnr", "ssim", "mae", "edge_score"],
    )
    write_csv(
        SUMMARY_CSV,
        summary_rows,
        [
            "method",
            "mean_psnr",
            "std_psnr",
            "mean_ssim",
            "std_ssim",
            "mean_mae",
            "std_mae",
            "mean_edge_score",
            "std_edge_score",
        ],
    )

    markdown = build_markdown_table(summary_rows)
    MARKDOWN_TABLE.write_text(markdown, encoding="utf-8")

    print()
    print_terminal_table(summary_rows)
    print()
    print(f"Per-image CSV saved to: {PER_IMAGE_CSV}")
    print(f"Summary CSV saved to: {SUMMARY_CSV}")
    print(f"Markdown table saved to: {MARKDOWN_TABLE}")
    print(f"Quality-aware SR outputs saved to: {QUALITY_AWARE_OUTPUT_DIR}")
    print(f"Quality-aware SR + L1/MSE outputs saved to: {QUALITY_AWARE_L1_MSE_OUTPUT_DIR}")
    print(f"Visual comparisons saved to: {VISUAL_COMPARISON_DIR}")


if __name__ == "__main__":
    main()
