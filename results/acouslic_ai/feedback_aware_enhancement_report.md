# Feedback-aware Enhancement Analysis for ACOUSLIC-AI

This is a no-reference proxy-quality analysis. Labels are heuristic proxy labels, not expert clinical labels.

ACOUSLIC-AI has no HR target in this workflow, so PSNR and SSIM are not computed.
This report compares no-reference proxy quality features before and after applying
the Quality-aware CNN + L1/MSE enhancement model.

## Inputs

- Per-frame proxy-quality manifest: `data/processed/ACOUSLIC_AI/acouslic_proxy_quality_manifest.csv`
- Per-sweep proxy-quality summary: `data/processed/ACOUSLIC_AI/acouslic_sweep_quality_summary.csv`
- Enhancement checkpoint: `results/models/quality_aware_sr_hc18_l1_mse_best.pth`

## Outputs

- Enhanced frames: `results/acouslic_ai/enhanced_frames_quality_aware_l1_mse`
- Per-frame CSV: `results/acouslic_ai/feedback_aware_enhancement_per_frame.csv`
- Summary CSV: `results/acouslic_ai/feedback_aware_enhancement_summary.csv`
- Figure folder: `results/acouslic_ai/feedback_aware_enhancement_figures`

## Overall result

- Total frames analyzed: 5000
- Number of frames improved by proxy label: 400
- Percentage improved: 8.00%
- Total sweeps in input summary: 50

## Quantile thresholds derived from original frames

- blur_score < 0.00409857: `blurry_frame`
- contrast_score < 0.14103059: `low_contrast_frame`
- brightness_mean < 0.00000000: `too_dark_frame`
- noise_proxy > 0.01148253: `noisy_frame`
- Multiple bad conditions: `poor_frame`

## Original frame-quality label distribution

| Label | Count |
|---|---:|
| `blurry_frame` | 261 |
| `good_frame` | 2257 |
| `low_contrast_frame` | 243 |
| `noisy_frame` | 1232 |
| `poor_frame` | 1007 |

## Enhanced frame-quality label distribution

| Label | Count |
|---|---:|
| `good_frame` | 10 |
| `low_contrast_frame` | 62 |
| `noisy_frame` | 4321 |
| `poor_frame` | 607 |

## Summary by original label

| Scope | Original label | Total frames | Improved | % improved | Mean Δ blur | Mean Δ contrast | Mean Δ edge | Mean Δ noise |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| overall | `all` | 5000 | 400 | 8.00 | 0.059303 | 0.072127 | 0.002005 | 0.026761 |
| by_original_label | `blurry_frame` | 261 | 0 | 0.00 | 0.029988 | 0.059218 | 0.001629 | 0.017545 |
| by_original_label | `good_frame` | 2257 | 0 | 0.00 | 0.056637 | 0.077819 | 0.001736 | 0.027887 |
| by_original_label | `low_contrast_frame` | 243 | 0 | 0.00 | 0.044098 | 0.072741 | -0.018462 | 0.020994 |
| by_original_label | `noisy_frame` | 1232 | 0 | 0.00 | 0.113977 | 0.106289 | 0.010234 | 0.045319 |
| by_original_label | `poor_frame` | 1007 | 400 | 39.72 | 0.009656 | 0.020773 | -0.002426 | 0.005310 |

## Saved figures

- `results/acouslic_ai/feedback_aware_enhancement_figures/improvement_rate_by_original_label.png`
- `results/acouslic_ai/feedback_aware_enhancement_figures/mean_edge_density_change_by_label.png`
- `results/acouslic_ai/feedback_aware_enhancement_figures/mean_contrast_change_by_label.png`

## Visual before/after examples

- `results/acouslic_ai/feedback_aware_enhancement_figures/before_after_examples/before_after_001_1e927a7e-90cd-475b-848d-6dc6368f42ca_frame_000058.png`
- `results/acouslic_ai/feedback_aware_enhancement_figures/before_after_examples/before_after_002_1cd004a2-94fb-429a-9e14-fc950bbd8852_frame_000099.png`
- `results/acouslic_ai/feedback_aware_enhancement_figures/before_after_examples/before_after_003_0685cfaf-d585-40b3-a365-52f70f2536b6_frame_000073.png`
- `results/acouslic_ai/feedback_aware_enhancement_figures/before_after_examples/before_after_004_0428da3d-fef4-4c9b-96f2-5ffedaee82d0_frame_000035.png`
- `results/acouslic_ai/feedback_aware_enhancement_figures/before_after_examples/before_after_005_0685cfaf-d585-40b3-a365-52f70f2536b6_frame_000033.png`
- `results/acouslic_ai/feedback_aware_enhancement_figures/before_after_examples/before_after_006_21e6dac4-7606-4abb-9284-d5d9bd0d2b6e_frame_000087.png`
- `results/acouslic_ai/feedback_aware_enhancement_figures/before_after_examples/before_after_007_1dccf454-18df-4bfb-b7bf-92acbea3d8e1_frame_000092.png`
- `results/acouslic_ai/feedback_aware_enhancement_figures/before_after_examples/before_after_008_20a2b2a1-c91e-4ee0-93b6-c0d8fe2212db_frame_000079.png`
- `results/acouslic_ai/feedback_aware_enhancement_figures/before_after_examples/before_after_009_0685cfaf-d585-40b3-a365-52f70f2536b6_frame_000005.png`
- `results/acouslic_ai/feedback_aware_enhancement_figures/before_after_examples/before_after_010_12941223-b27c-47f5-bcb0-7f7bfba18bba_frame_000100.png`
- `results/acouslic_ai/feedback_aware_enhancement_figures/before_after_examples/before_after_011_0685cfaf-d585-40b3-a365-52f70f2536b6_frame_000087.png`
- `results/acouslic_ai/feedback_aware_enhancement_figures/before_after_examples/before_after_012_0d0a3298-a9c6-43c3-a9e3-df3a9c0afa06_frame_000009.png`
- `results/acouslic_ai/feedback_aware_enhancement_figures/before_after_examples/before_after_013_1a998ef3-797f-4a17-9aa6-96cfb9af87c9_frame_000037.png`
- `results/acouslic_ai/feedback_aware_enhancement_figures/before_after_examples/before_after_014_136db39d-90e4-4702-aee7-dc5cd3169268_frame_000021.png`
- `results/acouslic_ai/feedback_aware_enhancement_figures/before_after_examples/before_after_015_0685cfaf-d585-40b3-a365-52f70f2536b6_frame_000032.png`
- `results/acouslic_ai/feedback_aware_enhancement_figures/before_after_examples/before_after_016_12941223-b27c-47f5-bcb0-7f7bfba18bba_frame_000058.png`
- `results/acouslic_ai/feedback_aware_enhancement_figures/before_after_examples/before_after_017_0428da3d-fef4-4c9b-96f2-5ffedaee82d0_frame_000010.png`
- `results/acouslic_ai/feedback_aware_enhancement_figures/before_after_examples/before_after_018_12941223-b27c-47f5-bcb0-7f7bfba18bba_frame_000089.png`
- `results/acouslic_ai/feedback_aware_enhancement_figures/before_after_examples/before_after_019_1dccf454-18df-4bfb-b7bf-92acbea3d8e1_frame_000032.png`
- `results/acouslic_ai/feedback_aware_enhancement_figures/before_after_examples/before_after_020_12941223-b27c-47f5-bcb0-7f7bfba18bba_frame_000090.png`
