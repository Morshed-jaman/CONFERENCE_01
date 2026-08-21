"""Bootstrap CIs and paired Wilcoxon significance tests for current HC18 results."""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from hc18_dataset import HC18SRDataset


try:
    from pyiqa import create_metric
except Exception:
    create_metric = None


MANIFEST = PROJECT_ROOT / "data" / "processed" / "HC18_SR" / "hc18_lq_manifest.csv"
RECON_PER_IMAGE = PROJECT_ROOT / "results" / "final_all_hc18_comparison_per_image.csv"
DOWNSTREAM_PER_IMAGE = PROJECT_ROOT / "results" / "downstream_utility_per_image.csv"
HARD_MEMBERSHIP = PROJECT_ROOT / "results" / "downstream_hard_subset_membership.csv"
DISTS_PER_IMAGE = PROJECT_ROOT / "results" / "final_all_hc18_dists_per_image.csv"
BOOTSTRAP_SUMMARY = PROJECT_ROOT / "results" / "bootstrap_metric_summary.csv"
SIGNIFICANCE_REPORT = PROJECT_ROOT / "results" / "significance_report.csv"
SWINIR_LABEL = "SwinIR-Light Phase A"
SWINIR_CHECKPOINT = PROJECT_ROOT / "results" / "models" / "swinir_light_phase_a_seed42" / "best_lpips.pth"
EVAL_ROOT = PROJECT_ROOT / "results" / "eval"

RECON_TO_CANONICAL = {
    "LQ_mixed baseline": "LQ_mixed",
    "SR-only": "SR-only",
    "Quality-aware SR": "Quality-aware",
    "SR-only + LPIPS + GAN": "SR-only+perceptual",
    "Quality-aware SR + LPIPS + GAN": "Quality-aware+perceptual",
    SWINIR_LABEL: SWINIR_LABEL,
}

CANONICAL_TO_RECON = {value: key for key, value in RECON_TO_CANONICAL.items()}
CORE_SIGNIFICANCE_METHODS = {
    "LQ_mixed",
    "SR-only",
    "Quality-aware",
    "SR-only+perceptual",
    "Quality-aware+perceptual",
    SWINIR_LABEL,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute bootstrap CIs and paired significance tests.")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--fallback_bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_dists", action="store_true", help="Skip pyiqa DISTS computation/cache refresh")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Use cpu for non-GPU paper work")
    return parser.parse_args()


def load_module(module_name: str, script_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def compute_quality_normalization_stats(dataset: HC18SRDataset, qa_module: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    features = []
    with torch.no_grad():
        for index in range(len(dataset)):
            lq = dataset[index]["lq"].unsqueeze(0).to(device=device, dtype=torch.float32)
            features.append(qa_module.compute_quality_features(lq).detach().cpu().view(-1))
    stacked = torch.stack(features, dim=0)
    mean = stacked.mean(dim=0).to(device).view(1, -1)
    std = stacked.std(dim=0).to(device).view(1, -1)
    std = torch.where(std == 0.0, torch.ones_like(std), std)
    return mean, std


def to_rgb_tensor(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = x.detach().to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.size(1) == 1:
        x = x.repeat(1, 3, 1, 1)
    return x


@torch.no_grad()
def compute_dists_cache(device: torch.device) -> pd.DataFrame:
    if DISTS_PER_IMAGE.exists():
        cached = pd.read_csv(DISTS_PER_IMAGE)
        if "method" in cached.columns and SWINIR_LABEL in set(cached["method"].astype(str)):
            return cached
    if create_metric is None:
        print("WARNING: pyiqa unavailable; DISTS cache will be empty.")
        return pd.DataFrame(columns=["image_id", "method", "DISTS"])

    try:
        dists_metric = create_metric("dists", device=device)
    except Exception as exc:
        print(f"WARNING: could not create pyiqa DISTS metric: {exc}")
        return pd.DataFrame(columns=["image_id", "method", "DISTS"])

    sr_module = load_module("sr_only_for_dists", SCRIPT_DIR / "05_train_sr_only_hc18.py")
    qa_module = load_module("qa_for_dists", SCRIPT_DIR / "07_train_quality_aware_sr_hc18.py")
    swinir_module = load_module("swinir_light_for_dists", SCRIPT_DIR / "models" / "swinir_light.py")
    train_dataset = HC18SRDataset(str(MANIFEST), input_type="LQ_mixed", split="train", return_mask=False)
    test_dataset = HC18SRDataset(str(MANIFEST), input_type="LQ_mixed", split="test", return_mask=False)
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    quality_mean, quality_std = compute_quality_normalization_stats(train_dataset, qa_module, device)

    specs = [
        ("LQ_mixed", "identity", None),
        ("SR-only", "sr_only", PROJECT_ROOT / "results" / "models" / "sr_only_hc18_best.pth"),
        ("Quality-aware", "quality_aware", PROJECT_ROOT / "results" / "models" / "quality_aware_sr_hc18_l1_mse_best.pth"),
        ("SR-only+perceptual", "sr_only", PROJECT_ROOT / "results" / "models" / "sr_perceptual_hc18_final.pth"),
        ("Quality-aware+perceptual", "quality_aware", PROJECT_ROOT / "results" / "models" / "qa_sr_perceptual_hc18_final.pth"),
        (SWINIR_LABEL, "swinir_light", SWINIR_CHECKPOINT),
    ]
    models: dict[str, torch.nn.Module | None] = {}
    for label, model_type, checkpoint_path in specs:
        if model_type == "identity":
            models[label] = None
            continue
        if checkpoint_path is None or not checkpoint_path.exists():
            print(f"WARNING: missing checkpoint for DISTS: {checkpoint_path}")
            continue
        if model_type == "sr_only":
            model = sr_module.ResidualSRModel().to(device)
        elif model_type == "quality_aware":
            model = qa_module.QualityAwareSRModel().to(device)
        elif model_type == "swinir_light":
            model = swinir_module.build_swinir_light(use_checkpoint=False).to(device)
        else:
            raise ValueError(f"Unsupported DISTS model type: {model_type}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(extract_state_dict(checkpoint), strict=True)
        model.eval()
        models[label] = model

    rows = []
    for batch in loader:
        image_id_value = batch["image_id"]
        image_id = str(image_id_value[0] if isinstance(image_id_value, (list, tuple)) else image_id_value)
        lq = batch["lq"].to(device=device, dtype=torch.float32)
        hr = batch["hr"].to(device=device, dtype=torch.float32)
        for label, model_type, _ in specs:
            model = models.get(label)
            if label not in models:
                continue
            if model is None:
                output = lq
            elif model_type == "quality_aware":
                quality = qa_module.compute_quality_features(lq)
                quality = (quality - quality_mean.to(quality.device)) / quality_std.to(quality.device)
                output = model(lq, quality).clamp(0.0, 1.0)
            else:
                output = model(lq).clamp(0.0, 1.0)
            try:
                score = dists_metric(to_rgb_tensor(output, device), to_rgb_tensor(hr, device))
                if isinstance(score, (tuple, list)):
                    score = score[0]
                score_value = float(score.detach().cpu().float().mean().item()) if torch.is_tensor(score) else float(score)
            except Exception as exc:
                print(f"WARNING: DISTS failed for {label}/{image_id}: {exc}")
                score_value = float("nan")
            rows.append({"image_id": image_id, "method": label, "DISTS": score_value})

    df = pd.DataFrame(rows)
    DISTS_PER_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DISTS_PER_IMAGE, index=False)
    return df


def discover_eval_recon_rows() -> tuple[pd.DataFrame, set[str]]:
    rows: list[pd.DataFrame] = []
    methods: set[str] = set()
    for path in sorted(EVAL_ROOT.glob("*/test/recon_per_image.csv")):
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"WARNING: could not read discovered recon CSV {path}: {exc}")
            continue
        if not {"image_id", "method"}.issubset(df.columns):
            print(f"WARNING: discovered recon CSV is missing image_id/method columns: {path}")
            continue
        df = df.copy()
        df["method"] = df["method"].astype(str).map(RECON_TO_CANONICAL).fillna(df["method"].astype(str))
        methods.update(df["method"].dropna().astype(str).unique())
        rows.append(df)
    if not rows:
        return pd.DataFrame(), set()
    return pd.concat(rows, ignore_index=True, sort=False), methods


def load_metric_tables(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[str]]:
    recon = pd.read_csv(RECON_PER_IMAGE)
    recon["method"] = recon["method"].map(RECON_TO_CANONICAL).fillna(recon["method"])
    discovered_recon, discovered_methods = discover_eval_recon_rows()
    if not discovered_recon.empty:
        recon = pd.concat([recon, discovered_recon], ignore_index=True, sort=False)
        recon = recon.drop_duplicates(subset=["image_id", "method"], keep="last")
    downstream = pd.read_csv(DOWNSTREAM_PER_IMAGE)
    membership = pd.read_csv(HARD_MEMBERSHIP)

    for column in ["PSNR", "SSIM", "LPIPS"]:
        recon[column] = pd.to_numeric(recon[column], errors="coerce")
    if "DISTS" in recon.columns:
        recon["DISTS"] = pd.to_numeric(recon["DISTS"], errors="coerce")
    for column in ["Dice", "DiamErr"]:
        downstream[column] = pd.to_numeric(downstream[column], errors="coerce")

    needs_dists_cache = "DISTS" not in recon.columns or recon["DISTS"].isna().any()
    if not args.skip_dists and needs_dists_cache:
        device = torch.device("cpu" if args.device == "cpu" or not torch.cuda.is_available() else "cuda")
        dists = compute_dists_cache(device)
        if not dists.empty:
            recon = recon.merge(dists, on=["image_id", "method"], how="left", suffixes=("", "_cache"))
            if "DISTS_cache" in recon.columns:
                if "DISTS" in recon.columns:
                    recon["DISTS"] = recon["DISTS"].combine_first(recon["DISTS_cache"])
                else:
                    recon["DISTS"] = recon["DISTS_cache"]
                recon = recon.drop(columns=["DISTS_cache"])
    if "DISTS" not in recon.columns:
        recon["DISTS"] = np.nan
    return recon, downstream, membership, discovered_methods


def subset_ids(membership: pd.DataFrame) -> dict[str, set[str]]:
    return {
        "full_test": set(membership["image_id"].astype(str)),
        "bottom25_lq_dice": set(membership.loc[membership["bottom25_lq_dice"].astype(bool), "image_id"].astype(str)),
        "lq_dice_lt_0_85": set(membership.loc[membership["lq_dice_lt_0_85"].astype(bool), "image_id"].astype(str)),
        "top25_lq_lpips": set(membership.loc[membership["top25_lq_lpips"].astype(bool), "image_id"].astype(str)),
    }


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        sample = rng.choice(values, size=values.size, replace=True)
        means[index] = float(np.mean(sample))
    return float(np.mean(values)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap_delta(
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float, float]:
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    delta_values = b - a
    boot = np.empty(n_bootstrap, dtype=np.float64)
    indices = np.arange(delta_values.size)
    for index in range(n_bootstrap):
        sample_indices = rng.choice(indices, size=indices.size, replace=True)
        boot[index] = float(np.mean(delta_values[sample_indices]))
    return float(np.mean(delta_values)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def wilcoxon_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    differences = (b[mask] - a[mask]).astype(np.float64)
    differences = differences[np.abs(differences) > 1e-12]
    if differences.size == 0:
        return 1.0
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(differences, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        abs_diff = np.abs(differences)
        order = np.argsort(abs_diff)
        ranks = np.empty_like(abs_diff, dtype=np.float64)
        ranks[order] = np.arange(1, abs_diff.size + 1, dtype=np.float64)
        w_pos = float(np.sum(ranks[differences > 0]))
        n = differences.size
        mean = n * (n + 1) / 4.0
        var = n * (n + 1) * (2 * n + 1) / 24.0
        z = (w_pos - mean) / math.sqrt(var)
        return float(2.0 * min(normal_cdf(z), 1.0 - normal_cdf(z)))


def aligned_values(df: pd.DataFrame, metric: str, method: str, ids: set[str]) -> pd.Series:
    subset = df[(df["method"] == method) & (df["image_id"].astype(str).isin(ids))]
    return subset.set_index("image_id")[metric].sort_index()


def make_metric_summary(
    recon: pd.DataFrame,
    downstream: pd.DataFrame,
    subsets: dict[str, set[str]],
    rng: np.random.Generator,
    n_bootstrap: int,
) -> pd.DataFrame:
    rows = []
    for subset_name, ids in subsets.items():
        for metric in ["LPIPS", "DISTS", "PSNR", "SSIM"]:
            for method in sorted(set(recon["method"])):
                values = aligned_values(recon, metric, method, ids).to_numpy(dtype=np.float64)
                mean, low, high = bootstrap_ci(values, rng, n_bootstrap)
                rows.append({"subset": subset_name, "method": method, "metric": metric, "mean": mean, "ci_low": low, "ci_high": high, "n": int(np.isfinite(values).sum())})
        for metric in ["Dice", "DiamErr"]:
            for method in sorted(set(downstream["method"])):
                values = aligned_values(downstream, metric, method, ids).to_numpy(dtype=np.float64)
                mean, low, high = bootstrap_ci(values, rng, n_bootstrap)
                rows.append({"subset": subset_name, "method": method, "metric": metric, "mean": mean, "ci_low": low, "ci_high": high, "n": int(np.isfinite(values).sum())})
    return pd.DataFrame(rows)


def planned_tests(discovered_methods: set[str]) -> list[tuple[str, str, str, str]]:
    tests: list[tuple[str, str, str, str]] = []
    full_and_hard = ["full_test", "bottom25_lq_dice", "lq_dice_lt_0_85", "top25_lq_lpips"]
    hard_only = ["bottom25_lq_dice", "lq_dice_lt_0_85", "top25_lq_lpips"]
    for subset in full_and_hard:
        for metric in ["LPIPS", "Dice"]:
            tests.append(("SR-only", "SR-only+perceptual", metric, subset))
            tests.append(("SR-only+perceptual", "Quality-aware+perceptual", metric, subset))
            tests.append(("SR-only+perceptual", SWINIR_LABEL, metric, subset))
            tests.append(("Quality-aware+perceptual", SWINIR_LABEL, metric, subset))
    tests.append(("LQ_mixed", "Quality-aware+perceptual", "LPIPS", "full_test"))
    for subset in hard_only:
        tests.append(("LQ_mixed", "SR-only+perceptual", "Dice", subset))
        tests.append(("LQ_mixed", SWINIR_LABEL, "Dice", subset))
    for method in sorted(discovered_methods - CORE_SIGNIFICANCE_METHODS):
        tests.append((SWINIR_LABEL, method, "LPIPS", "full_test"))
        tests.append((SWINIR_LABEL, method, "Dice", "full_test"))
        tests.append(("LQ_mixed", method, "LPIPS", "full_test"))
        tests.append(("LQ_mixed", method, "Dice", "full_test"))
    return tests


def make_significance_report(
    recon: pd.DataFrame,
    downstream: pd.DataFrame,
    subsets: dict[str, set[str]],
    discovered_methods: set[str],
    rng: np.random.Generator,
    n_bootstrap: int,
    auto_discovery_seed: int,
) -> pd.DataFrame:
    rows = []
    available_by_metric = {
        "LPIPS": set(recon["method"].astype(str)),
        "Dice": set(downstream["method"].astype(str)),
    }
    auto_discovered_extras = discovered_methods - CORE_SIGNIFICANCE_METHODS
    auto_rng = np.random.default_rng(auto_discovery_seed)
    for method_a, method_b, metric, subset_name in planned_tests(discovered_methods):
        available_methods = available_by_metric[metric]
        if method_a not in available_methods or method_b not in available_methods:
            continue
        source = recon if metric == "LPIPS" else downstream
        is_auto_discovered_extra = method_b in auto_discovered_extras
        if metric == "LPIPS" and subset_name == "full_test" and is_auto_discovered_extra:
            ids = set(source["image_id"].astype(str))
        else:
            ids = subsets[subset_name]
        a_series = aligned_values(source, metric, method_a, ids)
        b_series = aligned_values(source, metric, method_b, ids)
        aligned = pd.concat([a_series.rename("a"), b_series.rename("b")], axis=1, join="inner").dropna()
        a = aligned["a"].to_numpy(dtype=np.float64)
        b = aligned["b"].to_numpy(dtype=np.float64)
        delta_rng = auto_rng if is_auto_discovered_extra else rng
        delta, low, high = paired_bootstrap_delta(a, b, delta_rng, n_bootstrap)
        p_raw = wilcoxon_pvalue(a, b)
        rows.append(
            {
                "method_A": method_a,
                "method_B": method_b,
                "metric": metric,
                "subset": subset_name,
                "delta": delta,
                "ci_low": low,
                "ci_high": high,
                "p_raw": p_raw,
                "n_pairs": int(aligned.shape[0]),
            }
        )
    report = pd.DataFrame(rows)
    if report.empty:
        return report
    m = len(report)
    report["p_bonf"] = np.minimum(report["p_raw"].astype(float) * m, 1.0)
    report["significant_at_0.05"] = report["p_bonf"] < 0.05
    return report[
        [
            "method_A",
            "method_B",
            "metric",
            "subset",
            "delta",
            "ci_low",
            "ci_high",
            "p_raw",
            "p_bonf",
            "significant_at_0.05",
            "n_pairs",
        ]
    ]


def print_markdown_summary(report: pd.DataFrame) -> None:
    print("STEP 3b - SIGNIFICANCE SUMMARY")
    if report.empty:
        print("No significance rows were produced.")
        return
    significant = report[report["significant_at_0.05"].astype(bool)].copy()
    print(f"Total tests: {len(report)}")
    print(f"Bonferroni-significant tests: {len(significant)}")
    if significant.empty:
        print("No paired tests are significant after Bonferroni correction at alpha=0.05.")
        return
    print("| comparison | metric | subset | delta (B-A) | p_bonf |")
    print("|---|---:|---|---:|---:|")
    for _, row in significant.iterrows():
        comparison = f"{row['method_A']} -> {row['method_B']}"
        print(
            f"| {comparison} | {row['metric']} | {row['subset']} | "
            f"{float(row['delta']):.6f} | {float(row['p_bonf']):.3e} |"
        )


def main() -> int:
    args = parse_args()
    n_bootstrap = args.n_bootstrap if args.n_bootstrap > 0 else args.fallback_bootstrap
    if n_bootstrap > 1000:
        n_bootstrap = 1000
    rng = np.random.default_rng(args.seed)
    recon, downstream, membership, discovered_methods = load_metric_tables(args)
    discovered_methods = discovered_methods & set(recon["method"].astype(str)) & set(downstream["method"].astype(str))
    subsets = subset_ids(membership)
    metric_summary = make_metric_summary(recon, downstream, subsets, rng, n_bootstrap)
    metric_summary.to_csv(BOOTSTRAP_SUMMARY, index=False)
    report = make_significance_report(recon, downstream, subsets, discovered_methods, rng, n_bootstrap, args.seed)
    report.to_csv(SIGNIFICANCE_REPORT, index=False)
    print(f"Bootstrap metric summary: {BOOTSTRAP_SUMMARY.relative_to(PROJECT_ROOT)}")
    print(f"Significance report: {SIGNIFICANCE_REPORT.relative_to(PROJECT_ROOT)}")
    if discovered_methods:
        print(f"Auto-discovered eval methods: {', '.join(sorted(discovered_methods))}")
    if report["p_raw"].isna().any() if not report.empty else False:
        print("WARNING: Some p-values are NaN due to missing paired data.")
    if recon["DISTS"].isna().all():
        print("WARNING: DISTS is unavailable; pyiqa DISTS cache was not produced.")
    print_markdown_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
