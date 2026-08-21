"""
Lightweight comparison script that computes PSNR, SSIM, LPIPS, MAE, EdgeScore, NIQE, BRISQUE
for selected models on the HC18 test set. Writes a CSV summary to results/final_all_hc18_comparison_summary.csv
and a per-image CSV.

Designed for quick verification runs against freshly trained checkpoints.
"""

import csv
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from hc18_dataset import HC18SRDataset
from metrics_utils import (
    PyiqaMetricBundle,
    compute_psnr,
    compute_ssim,
    compute_mae,
    compute_lpips,
    compute_niqe,
    compute_brisque,
    compute_edge_score,
    get_pyiqa_metric,
)
from models.swinir_light import build_swinir_light


MANIFEST = PROJECT_ROOT / "data" / "processed" / "HC18_SR" / "hc18_lq_manifest.csv"
OUTPUT_PER_IMAGE = PROJECT_ROOT / "results" / "final_all_hc18_comparison_per_image.csv"
OUTPUT_SUMMARY = PROJECT_ROOT / "results" / "final_all_hc18_comparison_summary.csv"
OUTPUT_TABLE = PROJECT_ROOT / "results" / "final_all_hc18_comparison_table.md"
SWINIR_LABEL = "SwinIR-Light Phase A"
SWINIR_CHECKPOINT = PROJECT_ROOT / "results" / "models" / "swinir_light_phase_a_seed42" / "best_lpips.pth"
SWINIR_EVAL_DIR = PROJECT_ROOT / "results" / "eval" / "swinir_light_phase_a" / "test"
RESSHIFT_LABEL = "ResShift"
RESSHIFT_EVAL_DIR = PROJECT_ROOT / "results" / "eval" / "resshift" / "test"
RESSHIFT_RECON_DIR = RESSHIFT_EVAL_DIR / "restored"
OSEDIFF_LABEL = "OSEDiff"
OSEDIFF_EVAL_DIR = PROJECT_ROOT / "results" / "eval" / "osediff" / "test"
OSEDIFF_RECON_DIR = OSEDIFF_EVAL_DIR / "restored"

MODELS = [
    ("identity", None, "LQ_mixed baseline"),
    ("sr_only", PROJECT_ROOT / "results" / "models" / "sr_only_hc18_best.pth", "SR-only"),
    (
        "quality_aware",
        PROJECT_ROOT / "results" / "models" / "quality_aware_sr_hc18_l1_mse_best.pth",
        "Quality-aware SR",
    ),
    (
        "sr_only",
        PROJECT_ROOT / "results" / "models" / "sr_perceptual_hc18_final.pth",
        "SR-only + LPIPS + GAN",
    ),
    (
        "quality_aware",
        PROJECT_ROOT / "results" / "models" / "qa_sr_perceptual_hc18_final.pth",
        "Quality-aware SR + LPIPS + GAN",
    ),
    ("swinir_light", SWINIR_CHECKPOINT, SWINIR_LABEL),
    ("png_dir", RESSHIFT_RECON_DIR, RESSHIFT_LABEL),
    ("png_dir", OSEDIFF_RECON_DIR, OSEDIFF_LABEL),
]


def load_module(module_name: str, script_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def extract_state_dict(checkpoint: Any) -> Any:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def ensure_grayscale_tensor(output: torch.Tensor) -> torch.Tensor:
    if output.dim() == 4 and output.size(1) == 3:
        weights = output.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (output * weights).sum(dim=1, keepdim=True)
    return output


def load_png_tensor(image_path: Path, reference: torch.Tensor, device: torch.device) -> torch.Tensor:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Missing precomputed reconstruction: {image_path}")
    target_h, target_w = int(reference.shape[-2]), int(reference.shape[-1])
    if image.shape != (target_h, target_w):
        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    image = np.squeeze(image).astype(np.float32) / 255.0
    return torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)


def to_rgb_tensor(image: torch.Tensor, device: torch.device) -> torch.Tensor:
    tensor = image.detach().to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.size(1) == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    return tensor


def compute_dists(output: torch.Tensor, target: torch.Tensor, device: torch.device, metric: Any) -> float:
    if metric is None:
        return float("nan")
    try:
        with torch.no_grad():
            score = metric(to_rgb_tensor(output, device), to_rgb_tensor(target, device))
            if isinstance(score, (tuple, list)):
                score = score[0]
            if torch.is_tensor(score):
                return float(score.detach().cpu().float().mean().item())
            return float(score)
    except Exception:
        return float("nan")


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_training_components(device: torch.device):
    sr_module = load_module("hc18_sr_only_train", SCRIPT_DIR / "05_train_sr_only_hc18.py")
    qa_module = load_module("hc18_quality_aware_train", SCRIPT_DIR / "07_train_quality_aware_sr_hc18.py")
    return {
        "sr_only": sr_module.ResidualSRModel,
        "quality_aware": qa_module.QualityAwareSRModel,
        "compute_quality_features": qa_module.compute_quality_features,
    }


def load_model_for_key(key: str, checkpoint: Path, device: torch.device, components: Dict[str, Any]):
    if key == "sr_only":
        model = components["sr_only"]().to(device)
    elif key == "quality_aware":
        model = components["quality_aware"]().to(device)
    elif key == "swinir_light":
        model = build_swinir_light(use_checkpoint=False).to(device)
    else:
        raise ValueError(f"Unsupported model key: {key}")

    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(extract_state_dict(ckpt))
    model.eval()
    return model


def compute_quality_normalization_stats(
    dataset: HC18SRDataset,
    compute_quality_features: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    features_list: list[torch.Tensor] = []
    for index in range(len(dataset)):
        lq = dataset[index]["lq"].unsqueeze(0).to(device)
        with torch.no_grad():
            features = compute_quality_features(lq).detach().cpu().view(-1)
        features_list.append(features)
    stacked = torch.stack(features_list, dim=0)
    mean = stacked.mean(dim=0).to(device).view(1, -1)
    std = stacked.std(dim=0).to(device).view(1, -1)
    std = torch.where(std == 0.0, torch.ones_like(std), std)
    return mean, std


def format_float(value: Any, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def build_markdown_table(summary_rows: List[Dict[str, Any]], metric_columns: List[str]) -> str:
    header = "| method | " + " | ".join(metric_columns) + " |\n"
    separator = "|---|" + "|".join("---:" for _ in metric_columns) + "|\n"
    body = ""
    for row in summary_rows:
        values = [format_float(row[column], 6) for column in metric_columns]
        body += f"| {row['method']} | " + " | ".join(values) + " |\n"
    return header + separator + body


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_dataset = HC18SRDataset(csv_path=str(MANIFEST), input_type="LQ_mixed", split="test", return_mask=False)
    train_dataset = HC18SRDataset(csv_path=str(MANIFEST), input_type="LQ_mixed", split="train", return_mask=False)
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    per_image_rows: List[Dict[str, Any]] = []

    metric_bundle = PyiqaMetricBundle.create(device)
    dists_metric = get_pyiqa_metric("dists", device)
    components = load_training_components(device)
    quality_mean, quality_std = compute_quality_normalization_stats(
        train_dataset,
        components["compute_quality_features"],
        device,
    )

    for idx, (key, checkpoint, label) in enumerate(MODELS):
        print(f"Evaluating: {label}")
        model = None
        if key == "png_dir":
            if checkpoint is None or not checkpoint.exists():
                raise FileNotFoundError(f"Missing reconstruction directory for {label}: {checkpoint}")
        elif checkpoint is not None and checkpoint.exists():
            model = load_model_for_key(key, checkpoint, device, components)
        elif checkpoint is not None:
            print(f"  missing checkpoint, falling back to LQ input: {checkpoint}")

        for i, sample in enumerate(loader):
            image_id_value = sample["image_id"]
            if isinstance(image_id_value, (list, tuple)):
                image_id = str(image_id_value[0])
            else:
                image_id = str(image_id_value)
            lq = sample["lq"].to(device)
            hr = sample["hr"].to(device)

            if key == "png_dir":
                output = load_png_tensor(checkpoint / f"{image_id}.png", hr, device)
            elif model is None:
                output = lq
            else:
                with torch.no_grad():
                    if key == "quality_aware":
                        quality = components["compute_quality_features"](lq)
                        quality = (quality - quality_mean.to(quality.device)) / quality_std.to(quality.device)
                        output = model(lq, quality).clamp(0.0, 1.0)
                    else:
                        output = model(lq).clamp(0.0, 1.0)
                    output = ensure_grayscale_tensor(output).clamp(0.0, 1.0)

            psnr = compute_psnr(output, hr)
            ssim = compute_ssim(output, hr)
            mae = compute_mae(output, hr)
            edge = compute_edge_score(output, hr)
            lpips = compute_lpips(output, hr, device, metric=metric_bundle.lpips)
            dists = compute_dists(output, hr, device, dists_metric)
            niqe = compute_niqe(output, device, metric=metric_bundle.niqe)
            brisque = compute_brisque(output, device, metric=metric_bundle.brisque)

            per_image_rows.append(
                {
                    "image_id": image_id,
                    "method": label,
                    "PSNR": psnr,
                    "SSIM": ssim,
                    "LPIPS": lpips,
                    "DISTS": dists,
                    "MAE": mae,
                    "EdgeScore": edge,
                    "NIQE": niqe,
                    "BRISQUE": brisque,
                }
            )

    # write per-image csv
    metric_columns = ["PSNR", "SSIM", "LPIPS", "DISTS", "MAE", "EdgeScore", "NIQE", "BRISQUE"]
    fieldnames = ["image_id", "method"] + metric_columns
    OUTPUT_PER_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PER_IMAGE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_image_rows:
            writer.writerow(row)

    # summarize
    summary = {}
    for row in per_image_rows:
        m = row["method"]
        summary.setdefault(m, {k: [] for k in metric_columns})
        for k in summary[m].keys():
            summary[m][k].append(float(row[k]))

    summary_rows = []
    for m, metrics in summary.items():
        avg_row = {"method": m}
        for k, vals in metrics.items():
            avg_row[k] = float(sum(vals) / len(vals)) if vals else float("nan")
        summary_rows.append(avg_row)

    with open(OUTPUT_SUMMARY, "w", newline="") as f:
        fieldnames2 = ["method"] + metric_columns
        writer = csv.DictWriter(f, fieldnames=fieldnames2)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    swinir_rows = [row for row in per_image_rows if row["method"] == SWINIR_LABEL]
    if swinir_rows:
        swinir_summary = [row for row in summary_rows if row["method"] == SWINIR_LABEL]
        write_csv_rows(SWINIR_EVAL_DIR / "recon_per_image.csv", swinir_rows, fieldnames)
        write_csv_rows(SWINIR_EVAL_DIR / "recon_summary.csv", swinir_summary, fieldnames2)
    resshift_rows = [row for row in per_image_rows if row["method"] == RESSHIFT_LABEL]
    if resshift_rows:
        resshift_summary = [row for row in summary_rows if row["method"] == RESSHIFT_LABEL]
        write_csv_rows(RESSHIFT_EVAL_DIR / "recon_per_image.csv", resshift_rows, fieldnames)
        write_csv_rows(RESSHIFT_EVAL_DIR / "recon_summary.csv", resshift_summary, fieldnames2)
    osediff_rows = [row for row in per_image_rows if row["method"] == OSEDIFF_LABEL]
    if osediff_rows:
        osediff_summary = [row for row in summary_rows if row["method"] == OSEDIFF_LABEL]
        write_csv_rows(OSEDIFF_EVAL_DIR / "recon_per_image.csv", osediff_rows, fieldnames)
        write_csv_rows(OSEDIFF_EVAL_DIR / "recon_summary.csv", osediff_summary, fieldnames2)

    OUTPUT_TABLE.write_text(build_markdown_table(summary_rows, metric_columns), encoding="utf-8")

    print("Per-image CSV:", OUTPUT_PER_IMAGE)
    print("Summary CSV:", OUTPUT_SUMMARY)
    print("Markdown table:", OUTPUT_TABLE)
    if swinir_rows:
        print("SwinIR recon per-image CSV:", SWINIR_EVAL_DIR / "recon_per_image.csv")
        print("SwinIR recon summary CSV:", SWINIR_EVAL_DIR / "recon_summary.csv")
    if resshift_rows:
        print("ResShift recon per-image CSV:", RESSHIFT_EVAL_DIR / "recon_per_image.csv")
        print("ResShift recon summary CSV:", RESSHIFT_EVAL_DIR / "recon_summary.csv")
    if osediff_rows:
        print("OSEDiff recon per-image CSV:", OSEDIFF_EVAL_DIR / "recon_per_image.csv")
        print("OSEDiff recon summary CSV:", OSEDIFF_EVAL_DIR / "recon_summary.csv")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
