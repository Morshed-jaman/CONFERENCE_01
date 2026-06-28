# Adaptive Feedback-aware Enhancement Analysis

This is an adaptive no-reference proxy-quality analysis. Labels are heuristic proxy labels, not expert clinical labels.

This stage avoids blind enhancement. It keeps good frames unchanged, enhances
likely blurry/low-contrast frames, cautiously denoises noisy frames, and leaves
poor acquisition-failure frames for repeat/reacquisition feedback.

## Inputs

- Per-frame proxy-quality manifest: `data/processed/ACOUSLIC_AI/acouslic_proxy_quality_manifest.csv`
- Per-sweep proxy-quality summary: `data/processed/ACOUSLIC_AI/acouslic_sweep_quality_summary.csv`
- Quality-aware CNN + L1/MSE checkpoint: `results/models/quality_aware_sr_hc18_l1_mse_best.pth`

## Outputs

- Adaptive frames: `results/acouslic_ai/adaptive_enhanced_frames`
- Per-frame CSV: `results/acouslic_ai/adaptive_feedback_enhancement_per_frame.csv`
- Summary CSV: `results/acouslic_ai/adaptive_feedback_enhancement_summary.csv`
- Figure folder: `results/acouslic_ai/adaptive_feedback_enhancement_figures`

## Overall result

- Total frames processed: 10000
- Total sweeps in input summary: 100
- Number of frames improved by proxy label: 1794
- Percentage improved: 17.94%

## Adaptive policy

| Original proxy label | Action |
|---|---|
| `good_frame` | `keep_original` |
| `blurry_frame` | `enhance_blur` |
| `low_contrast_frame` | `enhance_contrast` using mild CLAHE then Quality-aware CNN |
| `noisy_frame` | `denoise_cautiously` without SR sharpening |
| `poor_frame` | `feedback_repeat_or_reacquire` |

## Original-feature thresholds used for both original and adaptive labels

- blur_score < 0.00405966: `blurry_frame`
- contrast_score < 0.14062477: `low_contrast_frame`
- brightness_mean < 0.00000000: `too_dark_frame`
- noise_proxy > 0.01146644: `noisy_frame`
- Multiple bad conditions: `poor_frame`

## Action counts

| Item | Count |
|---|---:|
| `denoise_cautiously` | 2462 |
| `enhance_blur` | 507 |
| `enhance_contrast` | 469 |
| `feedback_repeat_or_reacquire` | 2031 |
| `keep_original` | 4531 |

## Original label distribution

| Item | Count |
|---|---:|
| `blurry_frame` | 507 |
| `good_frame` | 4531 |
| `low_contrast_frame` | 469 |
| `noisy_frame` | 2462 |
| `poor_frame` | 2031 |

## Adaptive label distribution

| Item | Count |
|---|---:|
| `good_frame` | 6325 |
| `low_contrast_frame` | 6 |
| `noisy_frame` | 1638 |
| `poor_frame` | 2031 |

## Summary table

| Scope | Original label | Action | Total frames | Improved | % improved | Mean Δ blur | Mean Δ contrast | Mean Δ edge | Mean Δ noise |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| overall | `all` | `all` | 10000 | 1794 | 17.94 | 0.004533 | 0.009392 | -0.008094 | 0.002199 |
| by_action | `all` | `denoise_cautiously` | 2462 | 1784 | 72.46 | -0.000984 | -0.000199 | -0.040315 | -0.001561 |
| by_action | `all` | `enhance_blur` | 507 | 5 | 0.99 | 0.012468 | 0.047032 | -0.000481 | 0.010651 |
| by_action | `all` | `enhance_contrast` | 469 | 5 | 1.07 | 0.088328 | 0.150455 | 0.039569 | 0.043578 |
| by_action | `all` | `feedback_repeat_or_reacquire` | 2031 | 0 | 0.00 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| by_action | `all` | `keep_original` | 4531 | 0 | 0.00 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| by_original_label | `blurry_frame` | `all` | 507 | 5 | 0.99 | 0.012468 | 0.047032 | -0.000481 | 0.010651 |
| by_original_label | `good_frame` | `all` | 4531 | 0 | 0.00 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| by_original_label | `low_contrast_frame` | `all` | 469 | 5 | 1.07 | 0.088328 | 0.150455 | 0.039569 | 0.043578 |
| by_original_label | `noisy_frame` | `all` | 2462 | 1784 | 72.46 | -0.000984 | -0.000199 | -0.040315 | -0.001561 |
| by_original_label | `poor_frame` | `all` | 2031 | 0 | 0.00 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Saved figures

- `results/acouslic_ai/adaptive_feedback_enhancement_figures/action_distribution.png`
- `results/acouslic_ai/adaptive_feedback_enhancement_figures/improvement_rate_by_original_label.png`
- `results/acouslic_ai/adaptive_feedback_enhancement_figures/mean_edge_density_change_by_label.png`
- `results/acouslic_ai/adaptive_feedback_enhancement_figures/mean_contrast_change_by_label.png`

## Visual before/after examples

- `results/acouslic_ai/adaptive_feedback_enhancement_figures/before_after_examples/adaptive_before_after_001_04a04f2e-840b-47f8-a907-abe7aeab3f41_frame_000082.png`
- `results/acouslic_ai/adaptive_feedback_enhancement_figures/before_after_examples/adaptive_before_after_002_3bd64075-6e7e-463d-86e1-19071f04327c_frame_000084.png`
- `results/acouslic_ai/adaptive_feedback_enhancement_figures/before_after_examples/adaptive_before_after_003_1af7894a-9f0a-41e8-9c10-a11c0341a93e_frame_000023.png`
- `results/acouslic_ai/adaptive_feedback_enhancement_figures/before_after_examples/adaptive_before_after_004_032b9024-3685-4a81-a62e-813d2da3d387_frame_000079.png`
- `results/acouslic_ai/adaptive_feedback_enhancement_figures/before_after_examples/adaptive_before_after_005_19202c0a-fade-4c7a-9878-e91d823963d6_frame_000079.png`
- `results/acouslic_ai/adaptive_feedback_enhancement_figures/before_after_examples/adaptive_before_after_006_1dccf454-18df-4bfb-b7bf-92acbea3d8e1_frame_000013.png`
- `results/acouslic_ai/adaptive_feedback_enhancement_figures/before_after_examples/adaptive_before_after_007_12941223-b27c-47f5-bcb0-7f7bfba18bba_frame_000025.png`
- `results/acouslic_ai/adaptive_feedback_enhancement_figures/before_after_examples/adaptive_before_after_008_356d0021-3560-421a-96f4-ecffd4eb6005_frame_000029.png`
- `results/acouslic_ai/adaptive_feedback_enhancement_figures/before_after_examples/adaptive_before_after_009_19202c0a-fade-4c7a-9878-e91d823963d6_frame_000016.png`
- `results/acouslic_ai/adaptive_feedback_enhancement_figures/before_after_examples/adaptive_before_after_010_1af7894a-9f0a-41e8-9c10-a11c0341a93e_frame_000078.png`
- `results/acouslic_ai/adaptive_feedback_enhancement_figures/before_after_examples/adaptive_before_after_011_30555893-2ac6-45c9-b404-8ca2d9f90b05_frame_000060.png`
- `results/acouslic_ai/adaptive_feedback_enhancement_figures/before_after_examples/adaptive_before_after_012_4454a0d6-325b-442b-b07a-54f26b52c0b2_frame_000057.png`
- `results/acouslic_ai/adaptive_feedback_enhancement_figures/before_after_examples/adaptive_before_after_013_40d97ff5-0d2f-4d9c-a6bc-8d07713a86c2_frame_000039.png`
- `results/acouslic_ai/adaptive_feedback_enhancement_figures/before_after_examples/adaptive_before_after_014_05f72776-bf41-40ea-bdea-a7bcfa1eb9ba_frame_000002.png`
- `results/acouslic_ai/adaptive_feedback_enhancement_figures/before_after_examples/adaptive_before_after_015_40d97ff5-0d2f-4d9c-a6bc-8d07713a86c2_frame_000040.png`
- `results/acouslic_ai/adaptive_feedback_enhancement_figures/before_after_examples/adaptive_before_after_016_12941223-b27c-47f5-bcb0-7f7bfba18bba_frame_000012.png`
- `results/acouslic_ai/adaptive_feedback_enhancement_figures/before_after_examples/adaptive_before_after_017_12941223-b27c-47f5-bcb0-7f7bfba18bba_frame_000092.png`
- `results/acouslic_ai/adaptive_feedback_enhancement_figures/before_after_examples/adaptive_before_after_018_30555893-2ac6-45c9-b404-8ca2d9f90b05_frame_000057.png`
- `results/acouslic_ai/adaptive_feedback_enhancement_figures/before_after_examples/adaptive_before_after_019_4adc2ad8-f29d-41e3-9bf6-cc935057ff71_frame_000042.png`
- `results/acouslic_ai/adaptive_feedback_enhancement_figures/before_after_examples/adaptive_before_after_020_1dc4fda1-c351-4f4f-8df8-9bb35eb46062_frame_000085.png`
