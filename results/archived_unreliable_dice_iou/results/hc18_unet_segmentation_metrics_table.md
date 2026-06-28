# HC18 U-Net Segmentation Metrics

Segmentation-based Dice, IoU, and fetal measurement error are computed using an HC18-trained U-Net segmentation model and HC18 fetal-head masks. These metrics provide anatomy-preservation evidence, but they are not expert clinical validation.

## Model

- Architecture: lightweight U-Net
- Training inputs: HC18 HR images
- Training masks: HC18 fetal-head masks
- Resize size: 256 x 256
- Epochs: 30
- Loss: BCE + Dice loss
- Best checkpoint: `results/models/hc18_unet_segmentation_best.pth`

## Source Files

| Source | Status |
| --- | --- |
| `data/processed/HC18_SR/train.csv` | available |
| `data/processed/HC18_SR/val.csv` | available |
| `data/processed/HC18_SR/test.csv` | available |
| `data/processed/HC18_SR/HR` | available |
| `data/processed/HC18_SR/masks` | available |
| `results/sr_outputs` | available |

## Summary Metrics

| method | n_images | mean_dice | mean_iou | mean_boundary_error_px | mean_fetal_head_area_error_percent | mean_equivalent_diameter_error_percent |
| --- | --- | --- | --- | --- | --- | --- |
| HR ground-truth images | 150 | 0.2537 | 0.1477 | 3.4521 | 150.5747 | 57.7369 |
| LQ mixed baseline | 150 | 0.2398 | 0.1386 | 4.4251 | 139.7814 | 54.3334 |
| SR-only CNN | 150 | 0.2466 | 0.1431 | 4.5774 | 153.3409 | 58.5032 |
| Quality-aware CNN | 150 | 0.2486 | 0.1444 | 4.4310 | 155.0294 | 59.0019 |
| Quality-aware CNN + L1/MSE | 150 | 0.2487 | 0.1444 | 4.5097 | 154.9232 | 58.9719 |
| Conditional Diffusion SR | 150 | 0.2439 | 0.1412 | 4.3807 | 148.9718 | 57.0496 |
| Quality-aware Diffusion SR v2 | 150 | 0.2467 | 0.1432 | 4.3732 | 152.1954 | 58.1178 |
| CNN-Residual Diffusion SR | 150 | 0.2476 | 0.1437 | 4.4853 | 151.8727 | 57.9754 |

## Outputs

- Per-image metrics: `results/hc18_unet_segmentation_metrics_per_image.csv`
- Summary metrics: `results/hc18_unet_segmentation_metrics_summary.csv`
- Dice/IoU figure: `results/hc18_unet_segmentation_figures/dice_iou_comparison.png`
- Measurement error figure: `results/hc18_unet_segmentation_figures/measurement_error_comparison.png`
- Example segmentation figure: `results/hc18_unet_segmentation_figures/example_segmentation_outputs.png`

## Method Note

Final metrics use the trained U-Net probability map binarized at threshold
0.50; no image-threshold proxy segmentation is used for the
final Dice, IoU, boundary, or measurement metrics.
