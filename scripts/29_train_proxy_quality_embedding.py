"""
Train a lightweight proxy-quality embedding model from ACOUSLIC-AI frame features.

Important:
The frame labels are heuristic proxy labels, not expert clinical labels.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_MANIFEST = Path("data/processed/ACOUSLIC_AI/acouslic_proxy_quality_manifest.csv")
MODEL_PATH = Path("results/models/proxy_quality_embedding_mlp.pth")
PREDICTION_CSV = Path("results/acouslic_ai/proxy_quality_embedding_predictions.csv")
METRICS_CSV = Path("results/acouslic_ai/proxy_quality_embedding_metrics.csv")
CONFUSION_MATRIX_FIG = Path("results/acouslic_ai/proxy_quality_embedding_confusion_matrix.png")
REPORT_PATH = Path("results/acouslic_ai/proxy_quality_embedding_report.md")

FEATURE_COLUMNS = [
    "blur_score",
    "contrast_score",
    "brightness_mean",
    "brightness_std",
    "noise_proxy",
    "edge_density",
    "dark_ratio",
    "bright_ratio",
    "frame_difference_mean",
    "optical_flow_mean_magnitude",
    "optical_flow_std_magnitude",
    "jitter_score",
]

LABELS = [
    "good_frame",
    "blurry_frame",
    "low_contrast_frame",
    "noisy_frame",
    "poor_frame",
]

LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a lightweight MLP that maps ACOUSLIC-AI proxy frame features "
            "to proxy-quality labels and a 32D embedding."
        )
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--input_manifest",
        type=Path,
        default=INPUT_MANIFEST,
        help="Input proxy-quality manifest CSV.",
    )
    return parser.parse_args()


def resolve_project_path(path_value: object) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def rel_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_required_columns(df: pd.DataFrame, path: Path) -> None:
    required = ["sweep_id", "frame_id", "frame_quality_label", *FEATURE_COLUMNS]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    ensure_required_columns(df, path)

    unknown_labels = sorted(set(df["frame_quality_label"].astype(str)) - set(LABELS))
    if unknown_labels:
        raise ValueError(
            "Unsupported frame_quality_label values found: "
            f"{unknown_labels}. Expected only: {LABELS}"
        )

    df = df.copy()
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if df[FEATURE_COLUMNS].isna().any().any():
        bad_columns = [
            column for column in FEATURE_COLUMNS if df[column].isna().any()
        ]
        raise ValueError(f"Non-numeric or missing feature values in columns: {bad_columns}")

    df["label_index"] = df["frame_quality_label"].map(LABEL_TO_INDEX).astype(int)
    return df


def split_sweeps(df: pd.DataFrame, seed: int) -> tuple[set[object], set[object], set[object]]:
    sweeps = np.array(sorted(df["sweep_id"].unique()))
    if len(sweeps) < 3:
        raise ValueError("Need at least 3 sweeps to create train/val/test splits.")

    rng = np.random.default_rng(seed)
    rng.shuffle(sweeps)

    num_sweeps = len(sweeps)
    num_train = max(1, int(round(num_sweeps * 0.70)))
    num_val = max(1, int(round(num_sweeps * 0.15)))

    if num_train + num_val >= num_sweeps:
        num_train = max(1, num_sweeps - 2)
        num_val = 1

    train_sweeps = set(sweeps[:num_train])
    val_sweeps = set(sweeps[num_train : num_train + num_val])
    test_sweeps = set(sweeps[num_train + num_val :])

    return train_sweeps, val_sweeps, test_sweeps


def assign_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    train_sweeps, val_sweeps, test_sweeps = split_sweeps(df, seed=seed)

    split_values: list[str] = []
    for sweep_id in df["sweep_id"]:
        if sweep_id in train_sweeps:
            split_values.append("train")
        elif sweep_id in val_sweeps:
            split_values.append("val")
        elif sweep_id in test_sweeps:
            split_values.append("test")
        else:
            raise ValueError(f"Sweep was not assigned to a split: {sweep_id}")

    df = df.copy()
    df["split"] = split_values
    return df


def normalize_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    train_features = df.loc[df["split"] == "train", FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    feature_mean = train_features.mean(axis=0).astype(np.float32)
    feature_std = train_features.std(axis=0).astype(np.float32)
    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std).astype(np.float32)

    normalized = (
        df[FEATURE_COLUMNS].to_numpy(dtype=np.float32) - feature_mean
    ) / feature_std
    df = df.copy()
    for index, column in enumerate(FEATURE_COLUMNS):
        df[f"normalized_{column}"] = normalized[:, index]

    return df, feature_mean, feature_std


class FeatureDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.features = df[[f"normalized_{column}" for column in FEATURE_COLUMNS]].to_numpy(
            dtype=np.float32
        )
        self.labels = df["label_index"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.from_numpy(self.features[index])
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return features, label


class ProxyQualityEmbeddingMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 5):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.feature_net(x)
        logits = self.classifier(embedding)
        return logits, embedding


def make_loader(df: pd.DataFrame, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        FeatureDataset(df),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, _ = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return total_loss / total_count if total_count else 0.0


def predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_confidence: list[np.ndarray] = []
    all_embeddings: list[np.ndarray] = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            logits, embeddings = model(features)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = probabilities.max(dim=1)

            all_true.append(labels.cpu().numpy())
            all_pred.append(predicted.cpu().numpy())
            all_confidence.append(confidence.cpu().numpy())
            all_embeddings.append(embeddings.cpu().numpy())

    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_confidence),
        np.concatenate(all_embeddings),
    )


def evaluate(
    model: nn.Module,
    df: pd.DataFrame,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    loader = make_loader(df, batch_size=batch_size, shuffle=False)
    true, predicted, confidence, embeddings = predict(model, loader, device=device)

    return {
        "true": true,
        "predicted": predicted,
        "confidence": confidence,
        "embeddings": embeddings,
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(
            f1_score(
                true,
                predicted,
                average="macro",
                labels=range(len(LABELS)),
                zero_division=0,
            )
        ),
        "per_class_f1": f1_score(
            true,
            predicted,
            average=None,
            labels=range(len(LABELS)),
            zero_division=0,
        ),
    }


def metrics_rows(split_results: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split_name in ("train", "val", "test"):
        result = split_results[split_name]
        row: dict[str, object] = {
            "split": split_name,
            "num_frames": int(len(result["true"])),
            "accuracy": float(result["accuracy"]),
            "macro_f1": float(result["macro_f1"]),
        }
        for label, value in zip(LABELS, result["per_class_f1"], strict=False):
            row[f"f1_{label}"] = float(value)
        rows.append(row)
    return rows


def save_predictions(
    df: pd.DataFrame,
    result: dict[str, object],
    output_path: Path,
) -> None:
    predicted = result["predicted"]
    confidence = result["confidence"]
    embeddings = result["embeddings"]

    rows = pd.DataFrame(
        {
            "sweep_id": df["sweep_id"].to_numpy(),
            "frame_id": df["frame_id"].to_numpy(),
            "true_proxy_label": df["frame_quality_label"].to_numpy(),
            "predicted_proxy_label": [LABELS[int(index)] for index in predicted],
            "confidence": confidence,
        }
    )

    embedding_df = pd.DataFrame(
        embeddings,
        columns=[f"embedding_{index}" for index in range(embeddings.shape[1])],
    )
    output_df = pd.concat([rows, embedding_df], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False, float_format="%.8f")


def save_confusion_matrix(
    true: np.ndarray,
    predicted: np.ndarray,
    output_path: Path,
) -> None:
    matrix = confusion_matrix(true, predicted, labels=range(len(LABELS)))

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(LABELS)))
    ax.set_yticks(np.arange(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=35, ha="right")
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted proxy label")
    ax.set_ylabel("True proxy label")
    ax.set_title("Proxy Quality Embedding Confusion Matrix (Test Split)")

    max_value = matrix.max() if matrix.size else 0
    threshold = max_value / 2.0 if max_value else 0.0
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            color = "white" if value > threshold else "black"
            ax.text(col_index, row_index, str(value), ha="center", va="center", color=color)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    plt.close("all")


def write_report(
    output_path: Path,
    split_results: dict[str, dict[str, object]],
    split_counts: pd.DataFrame,
) -> None:
    test_result = split_results["test"]
    per_class_rows = "\n".join(
        f"| `{label}` | {float(value):.4f} |"
        for label, value in zip(LABELS, test_result["per_class_f1"], strict=False)
    )
    split_counts = (
        split_counts.set_index("split")
        .reindex(["train", "val", "test"])
        .reset_index()
    )
    split_rows = "\n".join(
        f"| `{row.split}` | {int(row.num_sweeps)} | {int(row.num_frames)} |"
        for row in split_counts.itertuples(index=False)
    )
    metrics_rows_text = "\n".join(
        f"| `{split}` | {float(result['accuracy']):.4f} | {float(result['macro_f1']):.4f} |"
        for split, result in split_results.items()
    )

    report = f"""# Proxy Quality Embedding MLP Report

This model learns a 32-dimensional embedding from ACOUSLIC-AI proxy frame
features. The labels are heuristic proxy labels, not expert clinical labels.

## Split Summary

| Split | Sweeps | Frames |
|---|---:|---:|
{split_rows}

## Metrics

| Split | Accuracy | Macro-F1 |
|---|---:|---:|
{metrics_rows_text}

## Test Per-Class F1

| Proxy label | F1 |
|---|---:|
{per_class_rows}

## Outputs

- Model checkpoint: `{rel_path(resolve_project_path(MODEL_PATH))}`
- Per-frame predictions: `{rel_path(resolve_project_path(PREDICTION_CSV))}`
- Metrics CSV: `{rel_path(resolve_project_path(METRICS_CSV))}`
- Confusion matrix: `{rel_path(resolve_project_path(CONFUSION_MATRIX_FIG))}`
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


def save_checkpoint(
    model: nn.Module,
    model_path: Path,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    args: argparse.Namespace,
    best_val_macro_f1: float,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": FEATURE_COLUMNS,
            "label_names": LABELS,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "input_dim": len(FEATURE_COLUMNS),
            "embedding_dim": 32,
            "hidden_dims": [64, 32],
            "best_val_macro_f1": best_val_macro_f1,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "label_note": "Proxy labels only; not expert clinical labels.",
        },
        model_path,
    )


def run(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be greater than 0.")
    if args.lr <= 0:
        raise ValueError("--lr must be greater than 0.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_manifest = resolve_project_path(args.input_manifest)
    if not input_manifest.exists():
        raise FileNotFoundError(f"Input manifest not found: {input_manifest}")

    df = load_manifest(input_manifest)
    df = assign_split(df, seed=args.seed)
    df, feature_mean, feature_std = normalize_features(df)

    split_counts = (
        df.groupby("split", sort=False)
        .agg(num_sweeps=("sweep_id", "nunique"), num_frames=("frame_id", "count"))
        .reset_index()
    )

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    train_loader = make_loader(train_df, batch_size=args.batch_size, shuffle=True)

    model = ProxyQualityEmbeddingMLP(input_dim=len(FEATURE_COLUMNS), num_classes=len(LABELS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_macro_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_result = evaluate(model, val_df, batch_size=args.batch_size, device=device)
        val_macro_f1 = float(val_result["macro_f1"])

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_loss:.6f} "
            f"val_accuracy={float(val_result['accuracy']):.4f} "
            f"val_macro_f1={val_macro_f1:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    split_results = {
        "train": evaluate(model, train_df, batch_size=args.batch_size, device=device),
        "val": evaluate(model, val_df, batch_size=args.batch_size, device=device),
        "test": evaluate(model, test_df, batch_size=args.batch_size, device=device),
    }

    all_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    all_result = evaluate(model, all_df, batch_size=args.batch_size, device=device)

    save_checkpoint(
        model=model,
        model_path=resolve_project_path(MODEL_PATH),
        feature_mean=feature_mean,
        feature_std=feature_std,
        args=args,
        best_val_macro_f1=best_val_macro_f1,
    )

    save_predictions(all_df, all_result, resolve_project_path(PREDICTION_CSV))

    metrics_df = pd.DataFrame(metrics_rows(split_results))
    resolve_project_path(METRICS_CSV).parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(resolve_project_path(METRICS_CSV), index=False, float_format="%.8f")

    save_confusion_matrix(
        true=split_results["test"]["true"],
        predicted=split_results["test"]["predicted"],
        output_path=resolve_project_path(CONFUSION_MATRIX_FIG),
    )
    write_report(
        output_path=resolve_project_path(REPORT_PATH),
        split_results=split_results,
        split_counts=split_counts,
    )

    test_result = split_results["test"]
    print("\nTraining complete")
    print(f"Device: {device}")
    print(f"Test accuracy: {float(test_result['accuracy']):.4f}")
    print(f"Test macro-F1: {float(test_result['macro_f1']):.4f}")
    print(f"Model checkpoint: {MODEL_PATH}")
    print(f"Predictions CSV: {PREDICTION_CSV}")
    print(f"Metrics CSV: {METRICS_CSV}")
    print(f"Confusion matrix: {CONFUSION_MATRIX_FIG}")
    print(f"Markdown report: {REPORT_PATH}")
    print("Reminder: these are proxy labels, not expert clinical labels.")


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except (FileNotFoundError, ValueError, pd.errors.EmptyDataError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
