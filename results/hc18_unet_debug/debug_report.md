# HC18 Emergency U-Net Debug Report

## Result

Overfit sanity check passed. Full U-Net training can now be restarted using the selected mask preprocessing method.

## Files

- Pairing grid: `results\hc18_unet_debug\debug_pairing_large_grid.png`
- Mask preprocessing comparison: `results\hc18_unet_debug\debug_mask_preprocessing_examples.png`
- Mask statistics: `results\hc18_unet_debug\debug_mask_statistics.csv`
- Overfit curve: `results\hc18_unet_debug\debug_overfit_no_aug_curve.png`
- Overfit predictions: `results\hc18_unet_debug\debug_overfit_predictions.png`

## Pairing Diagnosis

- Inspected training samples: 30
- Clearly aligned / non-suspicious samples: 0
- Suspicious samples: 30
- Masks appear aligned: questionable
- Thin, empty, too large, shifted, wrong-size, or suspicious masks: 30

## Mask Preprocessing Diagnosis

- Selected preprocessing method: `F_ellipse_fitted_filled`

| method | mean_area_ratio | empty | very_small | very_large | one_component | valid_area | score |
|---|---:|---:|---:|---:|---:|---:|---:|
| F_ellipse_fitted_filled | 0.296658 | 0 | 0 | 0 | 30 | 28 | 1.386676 |
| E_dilated_contour_then_filled | 0.131182 | 0 | 0 | 0 | 30 | 23 | 1.207849 |
| D_convex_hull_largest_contour | 0.034373 | 0 | 1 | 0 | 30 | 13 | 0.747706 |
| B_largest_contour_filled | 0.001955 | 0 | 30 | 0 | 30 | 0 | 0.311955 |
| C_all_contours_filled | 0.007204 | 0 | 4 | 0 | 0 | 0 | -0.212796 |
| A_raw_binary_mask_gt_0 | 0.007204 | 0 | 4 | 0 | 0 | 0 | -0.362796 |

## No-Augmentation Overfit Diagnosis

- Overfit samples used: 20
- Final Dice: 0.9508016467094421
- Final IoU: 0.9115534424781799
- Predicted mask mean: 0.29942779541015624
- Target mask mean: 0.2956085205078125
- Prediction behavior: predicted mask foreground fraction is plausible
- Target masks are: valid

## Likely Cause

The overfit test passed. The previous failure was likely caused by the mask preprocessing method, augmentation, loss weighting, or a different target construction path used before this debug run.

## Guardrail

Do not train the full segmentation model until this 20-sample overfit test reaches at least 0.90 Dice.
