# Review-Ready Anatomy Preservation Metrics

Dice, IoU, and fetal measurement error are computed using a proxy segmentation procedure on restored outputs and ground-truth HC18 fetal-head masks. These metrics support anatomy-preservation analysis but should not be interpreted as expert clinical validation.

## Reviewer-Requested Proxy Segmentation Metrics

| method | n_images | mean_dice | mean_iou | mean_boundary_error_px | mean_fetal_head_area_error_percent | mean_equivalent_diameter_error_percent |
| --- | --- | --- | --- | --- | --- | --- |
| LQ mixed baseline | 150 | 0.0095 | 0.0048 | 73.9086 | 17366.5982 | 1164.7326 |
| SR-only CNN | 150 | 0.0106 | 0.0054 | 72.5946 | 16571.9853 | 1123.6447 |
| Quality-aware CNN | 150 | 0.0098 | 0.0049 | 73.9370 | 16963.6195 | 1138.9964 |
| Quality-aware CNN + L1/MSE | 150 | 0.0100 | 0.0050 | 73.3795 | 16110.8113 | 1111.9828 |
| Conditional Diffusion SR | 150 | 0.0103 | 0.0052 | 72.2309 | 16748.0244 | 1131.4720 |
| Quality-aware Diffusion SR v2 | 150 | 0.0110 | 0.0055 | 72.9218 | 16371.6625 | 1114.9390 |
| CNN-Residual Diffusion SR | 150 | 0.0090 | 0.0045 | 74.6302 | 16820.1143 | 1149.2797 |

## Existing HC18 Reconstruction Metrics

| method | mean_psnr | mean_ssim | mean_mae | mean_edge_score |
| --- | --- | --- | --- | --- |
| LQ mixed baseline | 25.3376 | 0.6733 | 0.0338 | 0.8147 |
| SR-only CNN | 30.7646 | 0.8510 | 0.0182 | 0.9082 |
| Quality-aware CNN | 30.4013 | 0.8714 | 0.0177 | 0.8952 |
| Quality-aware CNN + L1/MSE | 30.5746 | 0.8737 | 0.0174 | 0.8969 |
| Conditional Diffusion SR | 28.2914 | 0.8297 | 0.0254 | 0.9343 |
| Quality-aware Diffusion SR v2 | 28.2739 | 0.7995 | 0.0264 | 0.9353 |
| CNN-Residual Diffusion SR | 29.8726 | 0.8686 | 0.0206 | 0.9542 |

## Existing Anatomy Preservation Metrics

| method | mean_mae_inside_mask | mean_boundary_error | mean_contrast_preservation | mean_edge_similarity_boundary |
| --- | --- | --- | --- | --- |
| LQ mixed baseline | 0.0688 | 0.0670 | 0.7292 | 0.8217 |
| SR-only CNN | 0.0316 | 0.0303 | 0.9727 | 0.9582 |
| Quality-aware CNN | 0.0325 | 0.0312 | 0.9625 | 0.9561 |
| Quality-aware CNN + L1/MSE | 0.0320 | 0.0305 | 0.9678 | 0.9563 |
| Conditional Diffusion SR | 0.0449 | 0.0435 | 0.9240 | 0.9320 |
| Quality-aware Diffusion SR v2 | 0.0433 | 0.0419 | 0.9472 | 0.9408 |
| CNN-Residual Diffusion SR | 0.0363 | 0.0349 | 0.9440 | 0.9415 |

## Generated Files

- `results/hc18_dice_iou_measurement_metrics.csv`
- `results/hc18_dice_iou_measurement_summary.csv`
- `results/hc18_dice_iou_measurement_table.md`
- `results/hc18_dice_iou_measurement_figures/dice_iou_bar_chart.png`
- `results/hc18_dice_iou_measurement_figures/fetal_measurement_error_bar_chart.png`
