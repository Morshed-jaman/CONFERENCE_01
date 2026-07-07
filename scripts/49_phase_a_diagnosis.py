"""Diagnose partial Phase A SwinIR-Light training without running GPU training."""

from __future__ import annotations

import argparse
import ast
import math
import re
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "results" / "training_logs"
OUTPUT_DIR = PROJECT_ROOT / "results" / "phase_a_diagnosis"
TRAINER_PATH = PROJECT_ROOT / "scripts" / "50_train_swinir_hc18.py"
LPIPS_GATE = 0.1131


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Phase A SwinIR-Light LPIPS trajectory.")
    parser.add_argument("--log_glob", default="50_train_swinir_hc18_swinir_light_phase_a_seed42_*.csv")
    parser.add_argument("--gate", type=float, default=LPIPS_GATE)
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    return parser.parse_args()


def load_phase_a_logs(pattern: str) -> pd.DataFrame:
    frames = []
    for path in sorted(LOG_DIR.glob(pattern)):
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["source_log"] = str(path.relative_to(PROJECT_ROOT))
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No Phase A logs matched: {LOG_DIR / pattern}")

    combined = pd.concat(frames, axis=0, ignore_index=True)
    for column in ["epoch", "train_loss_g", "val_LPIPS", "val_PSNR", "val_SSIM"]:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    combined = combined.dropna(subset=["epoch", "val_LPIPS"]).copy()
    combined["epoch"] = combined["epoch"].astype(int)
    combined = combined.sort_values(["epoch", "source_log"]).drop_duplicates("epoch", keep="last")
    return combined.sort_values("epoch").reset_index(drop=True)


def linear_fit(last_df: pd.DataFrame, gate: float) -> dict[str, float]:
    x = last_df["epoch"].to_numpy(dtype=np.float64)
    y = last_df["val_LPIPS"].to_numpy(dtype=np.float64)
    slope, intercept = np.polyfit(x, y, deg=1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
    cross_epoch = float("inf")
    if slope < 0.0:
        cross_epoch = float((gate - intercept) / slope)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "cross_epoch": cross_epoch,
    }


def literal_default(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def inspect_trainer(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(text)
    defaults: dict[str, Any] = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            continue
        flag = None
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                flag = arg.value.lstrip("-").replace("-", "_")
                break
        if flag is None:
            continue
        for keyword in node.keywords:
            if keyword.arg == "default":
                defaults[flag] = literal_default(keyword.value)

    recipe = {
        "l1_weight": defaults.get("l1_weight"),
        "lpips_weight": defaults.get("lpips_weight"),
        "gan_weight": defaults.get("gan_weight"),
        "lr": defaults.get("lr"),
        "d_lr": defaults.get("d_lr"),
        "warmup_epochs": defaults.get("warmup_epochs"),
        "ema_decay": defaults.get("ema_decay"),
        "grad_clip": defaults.get("grad_clip"),
        "window_size": defaults.get("window_size"),
        "depths": defaults.get("depths"),
        "loss_type": "L1 + LPIPS + GAN",
        "charbonnier": bool(re.search(r"Charbonnier", text, flags=re.IGNORECASE)),
        "lr_schedule": "linear warmup, then cosine decay via lr_for_epoch",
        "optimizer": "AdamW generator lr=args.lr betas=(0.9,0.99) wd=1e-4; AdamW discriminator lr=args.d_lr",
        "ema": "enabled via ModelEMA(model, decay=args.ema_decay)",
        "augmentations": "horizontal flip p=0.5, rotation +/-5 degrees, brightness/contrast jitter +/-5%",
    }
    recipe["matches_phase2_recipe"] = (
        recipe["lpips_weight"] == 0.2 and recipe["gan_weight"] == 0.005
    )
    return recipe


def save_figure(df: pd.DataFrame, fit: dict[str, float], gate: float, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = df["epoch"].to_numpy(dtype=np.float64)
    y = df["val_LPIPS"].to_numpy(dtype=np.float64)
    max_epoch = int(max(np.nanmax(x), min(fit["cross_epoch"], x[-1] + 40) if math.isfinite(fit["cross_epoch"]) else x[-1] + 40))
    x_line = np.arange(int(x[-1]), max_epoch + 1, dtype=np.float64)
    y_line = fit["slope"] * x_line + fit["intercept"]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(x, y, marker="o", linewidth=2, label="Observed val LPIPS")
    ax.plot(x_line, y_line, linestyle="--", linewidth=2, label="Last-10 linear extrapolation")
    ax.axhline(gate, color="crimson", linestyle=":", linewidth=2, label=f"LPIPS gate {gate:.4f}")
    if math.isfinite(fit["cross_epoch"]):
        ax.axvline(fit["cross_epoch"], color="gray", linestyle=":", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation LPIPS")
    ax.set_title("Phase A SwinIR-Light LPIPS Trajectory")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def verdict(fit: dict[str, float], last_epoch: int) -> str:
    slope = fit["slope"]
    cross_epoch = fit["cross_epoch"]
    if abs(slope) < 0.0005 or slope >= 0.0:
        return "C) ABANDON"
    if math.isfinite(cross_epoch) and cross_epoch <= last_epoch + 40:
        return "A) CONTINUE"
    return "B) RETUNE"


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    df = load_phase_a_logs(args.log_glob)
    last_df = df.tail(10).copy()
    fit = linear_fit(last_df, args.gate)
    recipe = inspect_trainer(TRAINER_PATH)
    output_path = output_dir / "lpips_trajectory.png"
    save_figure(df, fit, args.gate, output_path)

    last_values = [float(value) for value in last_df["val_LPIPS"]]
    last_epoch = int(df["epoch"].iloc[-1])
    delta_epochs = fit["cross_epoch"] - last_epoch if math.isfinite(fit["cross_epoch"]) else float("inf")
    plateaued = abs(fit["slope"]) < 0.0005
    result = verdict(fit, last_epoch)

    summary_path = output_dir / "phase_a_diagnosis_summary.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "last_epoch": last_epoch,
                "last_val_lpips": float(df["val_LPIPS"].iloc[-1]),
                "slope_last10": fit["slope"],
                "r2_last10": fit["r2"],
                "cross_epoch": fit["cross_epoch"],
                "delta_epochs_to_gate": delta_epochs,
                "plateaued": plateaued,
                "verdict": result,
            }
        ]
    ).to_csv(summary_path, index=False)

    print("STEP 2 - PHASE A DIAGNOSIS")
    print(f"Logs parsed: {len(sorted(LOG_DIR.glob(args.log_glob)))}")
    print(f"Epochs available: {int(df['epoch'].min())}-{last_epoch} ({len(df)} epochs)")
    print(f"Raw val_lpips values for last 10 recorded epochs: {[round(v, 6) for v in last_values]}")
    print(f"Linear-fit slope over last 10 epochs: {fit['slope']:.8f} LPIPS/epoch")
    print(f"Linear-fit R^2 over last 10 epochs: {fit['r2']:.6f}")
    print(f"Extrapolated epoch crossing {args.gate:.4f}: {fit['cross_epoch']:.2f}")
    print(f"Plateaued by |slope| < 0.0005: {plateaued}")
    print(f"Figure saved: {output_path.relative_to(PROJECT_ROOT)}")
    print(f"Summary CSV saved: {summary_path.relative_to(PROJECT_ROOT)}")
    print()
    print("Trainer recipe found in scripts/50_train_swinir_hc18.py")
    for key in [
        "loss_type",
        "l1_weight",
        "lpips_weight",
        "gan_weight",
        "charbonnier",
        "lr",
        "d_lr",
        "warmup_epochs",
        "lr_schedule",
        "optimizer",
        "ema_decay",
        "ema",
        "augmentations",
        "grad_clip",
    ]:
        print(f"- {key}: {recipe[key]}")
    print(f"- matches winning Phase 2 LPIPS/GAN weights: {recipe['matches_phase2_recipe']}")
    print()
    print(f"DIAGNOSIS VERDICT: {result}")
    if result.startswith("A"):
        print("5-line summary:")
        print("1. Phase A is still improving with a strong negative LPIPS slope.")
        print(f"2. Last recorded val LPIPS is {float(df['val_LPIPS'].iloc[-1]):.6f} at epoch {last_epoch}.")
        print(f"3. Linear extrapolation crosses the 0.1131 gate around epoch {fit['cross_epoch']:.2f}.")
        print("4. The loss weights match the winning perceptual recipe: LPIPS 0.2 and GAN 0.005.")
        print("5. Continue Phase A before starting Phase B.")
    elif result.startswith("B"):
        print("5-line summary:")
        print("1. Phase A is improving, but not fast enough to clear the gate within +40 epochs.")
        print("2. Retuning should target the listed recipe mismatches.")
        print("3. Do not start Phase B until the backbone decision is resolved.")
        print("4. Preserve current partial checkpoints for comparison.")
        print("5. Recommended next action depends on review.")
    else:
        print("5-line summary:")
        print("1. Phase A appears plateaued or diverging under the current recipe.")
        print("2. The observed trajectory does not support continuing this exact run.")
        print("3. Use the existing QA+perceptual champion as the practical backbone candidate.")
        print("4. Document the backbone decision before Phase B.")
        print("5. Do not start Phase B in this run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
