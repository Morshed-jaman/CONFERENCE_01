"""Train and evaluate the short SR-only perceptual acceptance check.

This script trains two models on the same HC18 split and seed:
- SR-only L1/MSE baseline
- SR-only L1 + LPIPS + GAN perceptual model

It evaluates both final checkpoints on the test split and writes a compact CSV
with the requested columns:
method, PSNR, SSIM, LPIPS, MAE, EdgeScore, NIQE, BRISQUE
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = ROOT / "scripts"
MANIFEST = ROOT / "data" / "processed" / "HC18_SR" / "hc18_lq_manifest.csv"
MODEL_DIR = ROOT / "results" / "models"
SUMMARY_CSV = ROOT / "results" / "short_sr_perceptual_comparison.csv"
PER_IMAGE_CSV = ROOT / "results" / "short_sr_perceptual_comparison_per_image.csv"

sys.path.insert(0, str(SCRIPT_DIR))

from hc18_dataset import HC18SRDataset
from metrics_utils import (
    PyiqaMetricBundle,
    compute_brisque,
    compute_edge_score,
    compute_lpips,
    compute_mae,
    compute_niqe,
    compute_psnr,
    compute_ssim,
)


METRIC_COLUMNS = ["PSNR", "SSIM", "LPIPS", "MAE", "EdgeScore", "NIQE", "BRISQUE"]
SUMMARY_COLUMNS = ["method"] + METRIC_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Short perceptual acceptance check")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--input_type", type=str, default="LQ_mixed")
    parser.add_argument("--lpips_weight", type=float, default=0.1)
    parser.add_argument("--gan_weight", type=float, default=0.05)
    parser.add_argument("--checkpoint_kind", choices=["final", "best"], default="final")
    parser.add_argument("--skip_training", action="store_true")
    return parser.parse_args()


def load_module(module_name: str, script_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_command(cmd: list[str]) -> None:
    print("\nRunning:", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def train_short_models(args: argparse.Namespace) -> None:
    baseline_cmd = [
        sys.executable,
        "scripts/40_train_sr_perceptual_hc18.py",
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--input_type",
        args.input_type,
        "--arch",
        "sr_only",
        "--seed",
        str(args.seed),
        "--run_name",
        "short_sr_l1_mse",
        "--l1_weight",
        "0.8",
        "--mse_weight",
        "0.2",
        "--lpips_weight",
        "0.0",
        "--gan_weight",
        "0.0",
        "--no_gan",
    ]
    perceptual_cmd = [
        sys.executable,
        "scripts/40_train_sr_perceptual_hc18.py",
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--input_type",
        args.input_type,
        "--arch",
        "sr_only",
        "--seed",
        str(args.seed),
        "--run_name",
        "short_sr_lpips_gan",
        "--l1_weight",
        "1.0",
        "--mse_weight",
        "0.0",
        "--lpips_weight",
        str(args.lpips_weight),
        "--gan_weight",
        str(args.gan_weight),
    ]
    run_command(baseline_cmd)
    run_command(perceptual_cmd)


def extract_state_dict(checkpoint: Any) -> Any:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def load_sr_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    sr_module = load_module("hc18_sr_only_train", SCRIPT_DIR / "05_train_sr_only_hc18.py")
    model = sr_module.ResidualSRModel().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(extract_state_dict(checkpoint))
    model.eval()
    return model


def evaluate_model(
    model: torch.nn.Module,
    method: str,
    loader: DataLoader,
    device: torch.device,
    metrics: PyiqaMetricBundle,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for sample in loader:
            image_id_value = sample["image_id"]
            image_id = str(image_id_value[0]) if isinstance(image_id_value, (list, tuple)) else str(image_id_value)
            lq = sample["lq"].to(device)
            hr = sample["hr"].to(device)
            output = model(lq).clamp(0.0, 1.0)

            row = {
                "image_id": image_id,
                "method": method,
                "PSNR": compute_psnr(output, hr),
                "SSIM": compute_ssim(output, hr),
                "LPIPS": compute_lpips(output, hr, device, metric=metrics.lpips),
                "MAE": compute_mae(output, hr),
                "EdgeScore": compute_edge_score(output, hr),
                "NIQE": compute_niqe(output, device, metric=metrics.niqe),
                "BRISQUE": compute_brisque(output, device, metric=metrics.brisque),
            }
            rows.append(row)

    summary = {"method": method}
    for column in METRIC_COLUMNS:
        values = [float(row[column]) for row in rows if not math.isnan(float(row[column]))]
        summary[column] = float(sum(values) / len(values)) if values else float("nan")
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def print_table(summary_rows: list[dict[str, Any]]) -> None:
    widths = {
        "method": max(len("method"), max(len(str(row["method"])) for row in summary_rows)),
        "PSNR": 10,
        "SSIM": 10,
        "LPIPS": 10,
        "MAE": 10,
        "EdgeScore": 11,
        "NIQE": 10,
        "BRISQUE": 10,
    }
    header = "  ".join(column.ljust(widths[column]) for column in SUMMARY_COLUMNS)
    print()
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        values = [str(row["method"]).ljust(widths["method"])]
        for column in METRIC_COLUMNS:
            values.append(f"{float(row[column]):.6f}".rjust(widths[column]))
        print("  ".join(values))


def read_last_log_row(run_name: str) -> dict[str, str] | None:
    log_files = sorted((ROOT / "results" / "training_logs").glob(f"{run_name}_*.csv"))
    if not log_files:
        return None
    with open(log_files[-1], "r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else None


def diagnose_failure(args: argparse.Namespace, baseline_lpips: float, perceptual_lpips: float) -> None:
    last_row = read_last_log_row("short_sr_lpips_gan")
    print("\nACCEPTANCE CHECK FAILED")
    print(f"Baseline LPIPS:   {baseline_lpips:.6f}")
    print(f"Perceptual LPIPS: {perceptual_lpips:.6f}")
    print(f"Delta (baseline - perceptual): {baseline_lpips - perceptual_lpips:.6f}")
    print("\nDiagnostics:")
    print(f"- lpips_weight={args.lpips_weight}")
    print(f"- gan_weight={args.gan_weight}")
    print("- LPIPS tensors are clamped to [0,1] before pyiqa in training and evaluation.")
    if last_row is None:
        print("- No perceptual training log was found; cannot verify discriminator loss.")
    else:
        print(f"- gan_enabled={last_row.get('gan_enabled', 'unknown')}")
        print(f"- perceptual_backend={last_row.get('perceptual_backend', 'unknown')}")
        print(f"- final discriminator_loss={last_row.get('discriminator_loss', 'unknown')}")
        if last_row.get("gan_enabled") != "1":
            print("- Discriminator is not marked as enabled in the training log.")


def main() -> None:
    args = parse_args()
    if not args.skip_training:
        train_short_models(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nEvaluating short-run final checkpoints on device: {device}")

    dataset = HC18SRDataset(
        csv_path=str(MANIFEST),
        input_type=args.input_type,
        split="test",
        return_mask=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    metrics = PyiqaMetricBundle.create(device)

    model_specs = [
        ("SR-only L1/MSE", MODEL_DIR / f"short_sr_l1_mse_hc18_{args.checkpoint_kind}.pth"),
        ("SR-only L1+LPIPS+GAN", MODEL_DIR / f"short_sr_lpips_gan_hc18_{args.checkpoint_kind}.pth"),
    ]

    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for method, checkpoint in model_specs:
        print(f"Evaluating {method}: {checkpoint}")
        model = load_sr_model(checkpoint, device)
        rows, summary = evaluate_model(model, method, loader, device, metrics)
        all_rows.extend(rows)
        summary_rows.append(summary)

    write_csv(PER_IMAGE_CSV, all_rows, ["image_id"] + SUMMARY_COLUMNS)
    write_csv(SUMMARY_CSV, summary_rows, SUMMARY_COLUMNS)
    print_table(summary_rows)
    print(f"\nPer-image CSV saved to: {PER_IMAGE_CSV}")
    print(f"Comparison CSV saved to: {SUMMARY_CSV}")

    baseline_lpips = float(summary_rows[0]["LPIPS"])
    perceptual_lpips = float(summary_rows[1]["LPIPS"])
    if math.isnan(baseline_lpips) or math.isnan(perceptual_lpips) or perceptual_lpips >= baseline_lpips:
        diagnose_failure(args, baseline_lpips, perceptual_lpips)
        raise SystemExit(2)

    delta = baseline_lpips - perceptual_lpips
    print("\nACCEPTANCE CHECK PASSED")
    print(f"Baseline LPIPS:   {baseline_lpips:.6f}")
    print(f"Perceptual LPIPS: {perceptual_lpips:.6f}")
    print(f"LPIPS delta (baseline - perceptual): {delta:.6f}")


if __name__ == "__main__":
    main()
