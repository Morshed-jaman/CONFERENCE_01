# Paper Summary Snapshot

Generated: 2026-07-07

Scope: non-GPU paper work through STEP 3c. Phase B/full SwinIR continuation is pending and was not started.

## Headline

Quality-aware SR + perceptual+GAN loss achieves LPIPS 0.1131 on the HC18 test set, a 44.6% reduction versus the Quality-aware SR baseline LPIPS 0.2042 and a 35.4% reduction versus the SR-only baseline LPIPS 0.1749. This trades PSNR from 30.58 dB to 28.74 dB relative to Quality-aware SR (-1.84 dB).

Downstream fetal-head Dice improves only slightly over the LQ baseline: 0.8858 vs 0.8850, an absolute gain of +0.00083 (+0.09% relative). This Dice effect is not statistically significant in the current frozen U-Net evaluation, so the paper claim should emphasize perceptual reconstruction quality and describe downstream segmentation as preserved/slightly improved rather than strongly improved.

## Reconstruction Metrics

| Method | PSNR | SSIM | LPIPS | NIQE | BRISQUE |
|---|---:|---:|---:|---:|---:|
| LQ_mixed baseline | 25.38 | 0.6723 | 0.3510 | 9.81 | 69.17 |
| SR-only | 30.99 | 0.8564 | 0.1749 | 7.45 | 54.51 |
| Quality-aware SR | 30.58 | 0.8743 | 0.2042 | 7.62 | 57.52 |
| SR-only + LPIPS + GAN | 27.86 | 0.7814 | 0.1237 | 6.90 | 29.09 |
| Quality-aware SR + LPIPS + GAN | 28.74 | 0.8266 | 0.1131 | 6.16 | 27.43 |

Full-test bootstrap LPIPS 95% CIs:

| Method | LPIPS mean | 95% CI |
|---|---:|---:|
| LQ_mixed | 0.3507 | [0.3443, 0.3569] |
| SR-only | 0.1748 | [0.1684, 0.1811] |
| SR-only + perceptual | 0.1239 | [0.1202, 0.1274] |
| Quality-aware + perceptual | 0.1130 | [0.1090, 0.1169] |

## Downstream Utility

Frozen HC18 U-Net downstream segmentation on the full test set:

| Method | Dice mean | 95% CI | Diameter error mean | 95% CI |
|---|---:|---:|---:|---:|
| HR | 0.8897 | [0.8637, 0.9119] | 16.2060 | [13.0423, 19.3802] |
| LQ_mixed | 0.8850 | [0.8616, 0.9063] | 16.3775 | [13.1917, 20.0110] |
| SR-only | 0.8853 | [0.8602, 0.9059] | 16.5769 | [13.6198, 20.0290] |
| SR-only + perceptual | 0.8873 | [0.8649, 0.9083] | 16.3414 | [13.2834, 19.5453] |
| Quality-aware + perceptual | 0.8858 | [0.8615, 0.9073] | 16.4891 | [13.3555, 19.9526] |

Hard-subset Dice deltas versus LQ_mixed:

| Subset | LQ Dice | SR-only + perceptual Dice | Delta | Quality-aware + perceptual Dice | Delta |
|---|---:|---:|---:|---:|---:|
| bottom25_lq_dice | 0.7048 | 0.7089 | +0.0041 | 0.7061 | +0.0013 |
| lq_dice_lt_0_85 | 0.7153 | 0.7211 | +0.0058 | 0.7182 | +0.0029 |
| top25_lq_lpips | 0.8635 | 0.8678 | +0.0043 | 0.8659 | +0.0024 |

## Significance Summary

Bootstrap/Wilcoxon testing used 1000 bootstrap resamples over the test set. There were 20 paired tests; 9 survived Bonferroni correction at alpha=0.05. All Bonferroni-significant effects were LPIPS improvements.

Key full-test paired results:

| Comparison | Metric | Delta | 95% CI | Bonferroni p | Significant |
|---|---|---:|---:|---:|---|
| SR-only -> SR-only + perceptual | LPIPS | -0.0510 | [-0.0571, -0.0447] | 1.82e-20 | yes |
| SR-only + perceptual -> Quality-aware + perceptual | LPIPS | -0.0108 | [-0.0132, -0.0085] | 9.01e-15 | yes |
| SR-only -> SR-only + perceptual | Dice | +0.0020 | [-0.0004, +0.0049] | 1.00 | no |
| SR-only + perceptual -> Quality-aware + perceptual | Dice | -0.0015 | [-0.0033, +0.0002] | 1.00 | no |
| LQ_mixed -> Quality-aware + perceptual | LPIPS | -0.2376 | [-0.2432, -0.2317] | 9.80e-25 | yes |

Interpretation for manuscript wording: Quality-aware + perceptual is the best perceptual reconstruction model by LPIPS/NIQE/BRISQUE. Downstream frozen U-Net Dice is essentially maintained, with small positive deltas over LQ_mixed but no statistically significant Dice advantage in this test.

