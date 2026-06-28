# Learned Adaptive Feedback-aware Enhancement Report

Labels are proxy labels, not expert clinical labels.

This run used a learned proxy-quality embedding MLP to choose conservative
feedback-aware enhancement actions. The Quality-aware CNN + L1/MSE model was
used only for frames where the learned policy requested enhancement.

## Inputs

- Proxy-quality manifest: `data/processed/ACOUSLIC_AI/acouslic_proxy_quality_manifest.csv`
- Learned proxy-quality MLP: `results/models/proxy_quality_embedding_mlp.pth`
- Quality-aware CNN + L1/MSE checkpoint: `results/models/quality_aware_sr_hc18_l1_mse_best.pth`
- Confidence threshold: 0.60

## Outputs

- Per-frame CSV: `results/acouslic_ai/learned_adaptive_enhancement_per_frame.csv`
- Summary CSV: `results/acouslic_ai/learned_adaptive_enhancement_summary.csv`
- Markdown report: `results/acouslic_ai/learned_adaptive_enhancement_report.md`
- Figures: `results/acouslic_ai/learned_adaptive_enhancement_figures`

## Overall Metrics

- Total frames processed: 10000
- Number of frames improved by proxy threshold label: 1724
- Percentage improved: 17.24%
- Mean blur change: 0.004337
- Mean contrast change: 0.008988
- Mean edge change: -0.008032
- Mean noise change: 0.002087

## Learned Policy

| Condition | Action |
|---|---|
| `good_frame` and confidence >= 0.60 | keep original |
| `blurry_frame` and confidence >= 0.60 | Quality-aware CNN + L1/MSE enhancement |
| `low_contrast_frame` and confidence >= 0.60 | CLAHE, then Quality-aware CNN + L1/MSE enhancement |
| `noisy_frame` and confidence >= 0.60 | cautious denoising |
| `poor_frame` and confidence >= 0.60 | repeat/reacquire feedback |
| confidence < 0.60 and original proxy label is `good_frame` | keep original |
| confidence < 0.60 and original proxy label is not `good_frame` | feedback warning |

## Original-feature thresholds used for before/after comparison

- blur_score < 0.00405966: `blurry_frame`
- contrast_score < 0.14062477: `low_contrast_frame`
- brightness_mean < 0.00000000: `too_dark_frame`
- noise_proxy > 0.01146644: `noisy_frame`
- Multiple bad conditions: `poor_frame`

## Action Counts

| Item | Count |
|---|---:|
| `clahe_then_enhance` | 457 |
| `denoise_cautiously` | 2432 |
| `enhance_blur` | 467 |
| `feedback_repeat_or_reacquire` | 2000 |
| `feedback_warning_low_confidence` | 109 |
| `keep_original` | 4451 |
| `keep_original_low_confidence` | 84 |

## Original Proxy-label Counts

| Item | Count |
|---|---:|
| `blurry_frame` | 507 |
| `good_frame` | 4531 |
| `low_contrast_frame` | 469 |
| `noisy_frame` | 2462 |
| `poor_frame` | 2031 |

## Learned Predicted-label Counts

| Item | Count |
|---|---:|
| `blurry_frame` | 497 |
| `good_frame` | 4527 |
| `low_contrast_frame` | 472 |
| `noisy_frame` | 2486 |
| `poor_frame` | 2018 |

## Summary Table

| Scope | Original label | Action | Total frames | Improved | % improved | Mean delta blur | Mean delta contrast | Mean delta edge | Mean delta noise |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| overall | `all` | `all` | 10000 | 1724 | 17.24 | 0.004337 | 0.008988 | -0.008032 | 0.002087 |
| by_action | `all` | `clahe_then_enhance` | 457 | 11 | 2.41 | 0.087587 | 0.150012 | 0.038899 | 0.043241 |
| by_action | `all` | `denoise_cautiously` | 2432 | 1707 | 70.19 | -0.000984 | -0.000199 | -0.040247 | -0.001560 |
| by_action | `all` | `enhance_blur` | 467 | 6 | 1.28 | 0.012272 | 0.046697 | -0.000451 | 0.010504 |
| by_action | `all` | `feedback_repeat_or_reacquire` | 2000 | 0 | 0.00 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| by_action | `all` | `feedback_warning_low_confidence` | 109 | 0 | 0.00 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| by_action | `all` | `keep_original` | 4451 | 0 | 0.00 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| by_action | `all` | `keep_original_low_confidence` | 84 | 0 | 0.00 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| by_original_label | `blurry_frame` | `all` | 507 | 4 | 0.79 | 0.011200 | 0.042634 | -0.000432 | 0.009588 |
| by_original_label | `good_frame` | `all` | 4531 | 0 | 0.00 | 0.000021 | 0.000055 | -0.000564 | -0.000002 |
| by_original_label | `low_contrast_frame` | `all` | 469 | 5 | 1.07 | 0.083932 | 0.143727 | 0.037149 | 0.041417 |
| by_original_label | `noisy_frame` | `all` | 2462 | 1706 | 69.29 | -0.000951 | -0.000192 | -0.038674 | -0.001504 |
| by_original_label | `poor_frame` | `all` | 2031 | 9 | 0.44 | 0.000280 | 0.000530 | 0.000124 | 0.000147 |

## Saved Figures

- `results/acouslic_ai/learned_adaptive_enhancement_figures/action_distribution.png`
- `results/acouslic_ai/learned_adaptive_enhancement_figures/improvement_rate_by_original_label.png`
- `results/acouslic_ai/learned_adaptive_enhancement_figures/mean_feature_changes.png`

## Visual Before/After Examples

- `results/acouslic_ai/learned_adaptive_enhancement_figures/before_after_examples/learned_before_after_001_0199616b-bdeb-4119-97a3-a5a3571bd641_frame_000001.png`
- `results/acouslic_ai/learned_adaptive_enhancement_figures/before_after_examples/learned_before_after_002_0199616b-bdeb-4119-97a3-a5a3571bd641_frame_000002.png`
- `results/acouslic_ai/learned_adaptive_enhancement_figures/before_after_examples/learned_before_after_003_0199616b-bdeb-4119-97a3-a5a3571bd641_frame_000006.png`
- `results/acouslic_ai/learned_adaptive_enhancement_figures/before_after_examples/learned_before_after_004_0199616b-bdeb-4119-97a3-a5a3571bd641_frame_000007.png`
- `results/acouslic_ai/learned_adaptive_enhancement_figures/before_after_examples/learned_before_after_005_0199616b-bdeb-4119-97a3-a5a3571bd641_frame_000008.png`
- `results/acouslic_ai/learned_adaptive_enhancement_figures/before_after_examples/learned_before_after_006_0199616b-bdeb-4119-97a3-a5a3571bd641_frame_000009.png`
- `results/acouslic_ai/learned_adaptive_enhancement_figures/before_after_examples/learned_before_after_007_0199616b-bdeb-4119-97a3-a5a3571bd641_frame_000010.png`
- `results/acouslic_ai/learned_adaptive_enhancement_figures/before_after_examples/learned_before_after_008_0199616b-bdeb-4119-97a3-a5a3571bd641_frame_000013.png`
- `results/acouslic_ai/learned_adaptive_enhancement_figures/before_after_examples/learned_before_after_009_0199616b-bdeb-4119-97a3-a5a3571bd641_frame_000016.png`
- `results/acouslic_ai/learned_adaptive_enhancement_figures/before_after_examples/learned_before_after_010_0199616b-bdeb-4119-97a3-a5a3571bd641_frame_000017.png`
- `results/acouslic_ai/learned_adaptive_enhancement_figures/before_after_examples/learned_before_after_011_0199616b-bdeb-4119-97a3-a5a3571bd641_frame_000018.png`
- `results/acouslic_ai/learned_adaptive_enhancement_figures/before_after_examples/learned_before_after_012_0199616b-bdeb-4119-97a3-a5a3571bd641_frame_000019.png`
- `results/acouslic_ai/learned_adaptive_enhancement_figures/before_after_examples/learned_before_after_013_0199616b-bdeb-4119-97a3-a5a3571bd641_frame_000021.png`
- `results/acouslic_ai/learned_adaptive_enhancement_figures/before_after_examples/learned_before_after_014_0199616b-bdeb-4119-97a3-a5a3571bd641_frame_000022.png`
- `results/acouslic_ai/learned_adaptive_enhancement_figures/before_after_examples/learned_before_after_015_0199616b-bdeb-4119-97a3-a5a3571bd641_frame_000024.png`
- `results/acouslic_ai/learned_adaptive_enhancement_figures/before_after_examples/learned_before_after_016_0199616b-bdeb-4119-97a3-a5a3571bd641_frame_000025.png`
- `results/acouslic_ai/learned_adaptive_enhancement_figures/before_after_examples/learned_before_after_017_0199616b-bdeb-4119-97a3-a5a3571bd641_frame_000026.png`
- `results/acouslic_ai/learned_adaptive_enhancement_figures/before_after_examples/learned_before_after_018_0199616b-bdeb-4119-97a3-a5a3571bd641_frame_000027.png`
- `results/acouslic_ai/learned_adaptive_enhancement_figures/before_after_examples/learned_before_after_019_0199616b-bdeb-4119-97a3-a5a3571bd641_frame_000028.png`
- `results/acouslic_ai/learned_adaptive_enhancement_figures/before_after_examples/learned_before_after_020_0199616b-bdeb-4119-97a3-a5a3571bd641_frame_000029.png`
