# Improved HC18 U-Net Segmentation Metrics

Dice, IoU, and fetal measurement error are computed using an HC18-trained U-Net segmentation model with ellipse-fitted filled mask preprocessing and HC18 fetal-head masks. These metrics are used as supporting anatomy-preservation evidence and should not be interpreted as expert clinical validation.

## Mask Preprocessing Summary

- Checked readable masks: 999
- Suspicious masks: 10 (1.00%)
- Missing HR images: 0
- Missing masks: 0
- Selected preprocessing method: `F_ellipse_fitted_filled`
- Mask statistics CSV: `results\hc18_unet_improved_mask_statistics.csv`
- Image-mask pairing figure: `results\hc18_unet_segmentation_figures\debug_image_mask_pairing.png`

Masks were converted to grayscale, binarized with `mask > 0`, fitted to the largest external-contour ellipse, filled, hole-filled, and resized with nearest-neighbor interpolation.

## Overfit Sanity Check

- Samples: 20 training images
- Overfit Dice: 0.9307
- Figure: `results\hc18_unet_segmentation_figures\overfit_sanity_check.png`

## Training Details

- Architecture: U-Net with encoder channels 32, 64, 128, 256 and bottleneck 512
- Epochs: 100
- Batch size: 8
- Optimizer: AdamW, lr=0.001, weight_decay=0.0001
- Loss: `BCEWithLogitsLoss + DiceLoss`
- Target mask preprocessing: `F_ellipse_fitted_filled`
- Augmentation: horizontal flip, small rotation, brightness/contrast shift, Gaussian noise
- Best validation Dice: 0.8776
- Best validation IoU: 0.8130
- Best epoch: 26
- Best threshold: 0.50
- Training curve: `results\hc18_unet_segmentation_figures\improved_training_curve.png`

## Threshold Tuning

| threshold | val_dice | val_iou |
| --- | --- | --- |
| 0.2000 | 0.8745 | 0.8071 |
| 0.2500 | 0.8758 | 0.8094 |
| 0.3000 | 0.8766 | 0.8109 |
| 0.3500 | 0.8771 | 0.8119 |
| 0.4000 | 0.8774 | 0.8125 |
| 0.4500 | 0.8775 | 0.8128 |
| 0.5000 | 0.8776 | 0.8130 |
| 0.5500 | 0.8774 | 0.8128 |
| 0.6000 | 0.8772 | 0.8125 |
| 0.6500 | 0.8766 | 0.8116 |
| 0.7000 | 0.8758 | 0.8102 |

## HR Test Reliability

- HR test Dice: 0.8897

## Final Method Comparison

| method | n_images | mean_dice | mean_iou | mean_boundary_error_px | mean_fetal_head_area_error_percent | mean_equivalent_diameter_error_percent |
| --- | --- | --- | --- | --- | --- | --- |
| HR ground truth | 148 | 0.8897 | 0.8225 | 9.7192 | 19.9109 | 9.4812 |
| LQ mixed baseline | 148 | 0.8850 | 0.8150 | 10.1920 | 20.2224 | 9.6149 |
| SR-only CNN | 148 | 0.8853 | 0.8157 | 10.4459 | 20.9446 | 9.8825 |
| Quality-aware CNN | 148 | 0.8842 | 0.8142 | 10.4615 | 21.2830 | 10.0189 |
| Quality-aware CNN + L1/MSE | 148 | 0.8847 | 0.8148 | 10.4485 | 21.1400 | 9.9617 |
| Conditional Diffusion SR | 148 | 0.8857 | 0.8163 | 10.2369 | 20.6967 | 9.7734 |
| Quality-aware Diffusion SR v2 | 148 | 0.8838 | 0.8135 | 10.5696 | 20.9578 | 9.9057 |
| CNN-Residual Diffusion SR | 148 | 0.8854 | 0.8156 | 10.4032 | 20.7863 | 9.8379 |

## Outputs

- Checkpoint: `results\models\hc18_unet_segmentation_best_improved.pth`
- Per-image metrics: `results\hc18_unet_improved_metrics_per_image.csv`
- Summary metrics: `results\hc18_unet_improved_metrics_summary.csv`
- Dice/IoU figure: `results\hc18_unet_segmentation_figures\improved_dice_iou_comparison.png`
- Measurement error figure: `results\hc18_unet_segmentation_figures\improved_measurement_error_comparison.png`
- Example segmentation figure: `results\hc18_unet_segmentation_figures\improved_example_segmentation_outputs.png`


