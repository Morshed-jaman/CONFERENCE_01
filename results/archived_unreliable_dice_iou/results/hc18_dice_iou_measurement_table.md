# HC18 Dice, IoU, and Fetal Measurement Metrics

Dice, IoU, and fetal measurement error are computed using a proxy segmentation procedure on restored outputs and ground-truth HC18 fetal-head masks. These metrics support anatomy-preservation analysis but should not be interpreted as expert clinical validation.

## Source Files

| Source | Status |
| --- | --- |
| `data/processed/HC18_SR/test.csv` | available |
| `data/processed/HC18_SR/masks` | available |
| `results/sr_outputs` | available |
| `results/final_all_hc18_comparison_summary.csv` | available |
| `results/anatomy_preservation_all_models_summary.csv` | available |

## Summary

| method | n_images | mean_dice | mean_iou | mean_boundary_error_px | mean_fetal_head_area_error_percent | mean_equivalent_diameter_error_percent |
| --- | --- | --- | --- | --- | --- | --- |
| LQ mixed baseline | 150 | 0.0095 | 0.0048 | 73.9086 | 17366.5982 | 1164.7326 |
| SR-only CNN | 150 | 0.0106 | 0.0054 | 72.5946 | 16571.9853 | 1123.6447 |
| Quality-aware CNN | 150 | 0.0098 | 0.0049 | 73.9370 | 16963.6195 | 1138.9964 |
| Quality-aware CNN + L1/MSE | 150 | 0.0100 | 0.0050 | 73.3795 | 16110.8113 | 1111.9828 |
| Conditional Diffusion SR | 150 | 0.0103 | 0.0052 | 72.2309 | 16748.0244 | 1131.4720 |
| Quality-aware Diffusion SR v2 | 150 | 0.0110 | 0.0055 | 72.9218 | 16371.6625 | 1114.9390 |
| CNN-Residual Diffusion SR | 150 | 0.0090 | 0.0045 | 74.6302 | 16820.1143 | 1149.2797 |

## Outputs

- Per-image metrics: `results/hc18_dice_iou_measurement_metrics.csv`
- Summary metrics: `results/hc18_dice_iou_measurement_summary.csv`
- Dice/IoU figure: `results/hc18_dice_iou_measurement_figures/dice_iou_bar_chart.png`
- Fetal measurement error figure: `results/hc18_dice_iou_measurement_figures/fetal_measurement_error_bar_chart.png`

## Method Note

Each restored output is normalized to 0-255, blurred with a Gaussian kernel,
thresholded with Otsu and adaptive-threshold candidates, reduced to the largest
connected component, hole-filled, resized to the HC18 mask size, and compared
against the ground-truth fetal-head mask.
