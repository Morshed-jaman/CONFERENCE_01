# Proxy Quality Embedding MLP Report

This model learns a 32-dimensional embedding from ACOUSLIC-AI proxy frame
features. The labels are heuristic proxy labels, not expert clinical labels.

## Split Summary

| Split | Sweeps | Frames |
|---|---:|---:|
| `train` | 70 | 7000 |
| `val` | 15 | 1500 |
| `test` | 15 | 1500 |

## Metrics

| Split | Accuracy | Macro-F1 |
|---|---:|---:|
| `train` | 0.9820 | 0.9797 |
| `val` | 0.9813 | 0.9779 |
| `test` | 0.9767 | 0.9688 |

## Test Per-Class F1

| Proxy label | F1 |
|---|---:|
| `good_frame` | 0.9762 |
| `blurry_frame` | 0.9565 |
| `low_contrast_frame` | 0.9457 |
| `noisy_frame` | 0.9732 |
| `poor_frame` | 0.9925 |

## Outputs

- Model checkpoint: `results/models/proxy_quality_embedding_mlp.pth`
- Per-frame predictions: `results/acouslic_ai/proxy_quality_embedding_predictions.csv`
- Metrics CSV: `results/acouslic_ai/proxy_quality_embedding_metrics.csv`
- Confusion matrix: `results/acouslic_ai/proxy_quality_embedding_confusion_matrix.png`
