# Bootstrap Confidence Intervals

Bootstrap settings: 1000 resamples, random seed 42.

These confidence intervals summarize benchmark/proxy metrics only. They are not
clinical validation.

## Inputs

- HC18 per-image comparison: `results/final_all_hc18_comparison_per_image.csv`
- Anatomy preservation per-image metrics: `results/anatomy_preservation_all_models_per_image.csv`
- ACOUSLIC-AI per-sweep feedback changes: `results/acouslic_ai/adaptive_sweep_feedback_change_per_sweep.csv`
- ACOUSLIC-AI per-frame adaptive enhancement metrics: `results/acouslic_ai/adaptive_feedback_enhancement_per_frame.csv`

## HC18 Reconstruction CIs

| Method | Metric | N | Mean | 95% CI lower | 95% CI upper | Unit |
| --- | --- | --- | --- | --- | --- | --- |
| LQ_mixed baseline | PSNR | 150 | 25.3849 | 25.1905 | 25.6075 |  |
| LQ_mixed baseline | SSIM | 150 | 0.6723 | 0.6607 | 0.6837 |  |
| LQ_mixed baseline | LPIPS | 150 | 0.351 | 0.3445 | 0.3573 |  |
| LQ_mixed baseline | MAE | 150 | 0.0337 | 0.0324 | 0.035 |  |
| LQ_mixed baseline | Edge score | 150 | 0.9779 | 0.9771 | 0.9788 |  |
| LQ_mixed baseline | NIQE | 150 | 9.8086 | 9.677 | 9.9526 |  |
| LQ_mixed baseline | BRISQUE | 150 | 69.1681 | 68.6251 | 69.7372 |  |
| Quality-aware SR | PSNR | 150 | 30.5804 | 30.2978 | 30.8664 |  |
| Quality-aware SR | SSIM | 150 | 0.8743 | 0.8693 | 0.8795 |  |
| Quality-aware SR | LPIPS | 150 | 0.2042 | 0.1971 | 0.2107 |  |
| Quality-aware SR | MAE | 150 | 0.0174 | 0.0168 | 0.0181 |  |
| Quality-aware SR | Edge score | 150 | 0.985 | 0.9843 | 0.9856 |  |
| Quality-aware SR | NIQE | 150 | 7.6178 | 7.4693 | 7.7771 |  |
| Quality-aware SR | BRISQUE | 150 | 57.5157 | 56.7994 | 58.2671 |  |
| Quality-aware SR + LPIPS + GAN | PSNR | 150 | 28.7435 | 28.453 | 29.0224 |  |
| Quality-aware SR + LPIPS + GAN | SSIM | 150 | 0.8266 | 0.8206 | 0.8322 |  |
| Quality-aware SR + LPIPS + GAN | LPIPS | 150 | 0.1131 | 0.1093 | 0.1169 |  |
| Quality-aware SR + LPIPS + GAN | MAE | 150 | 0.022 | 0.0212 | 0.0228 |  |
| Quality-aware SR + LPIPS + GAN | Edge score | 150 | 0.9827 | 0.982 | 0.9833 |  |
| Quality-aware SR + LPIPS + GAN | NIQE | 150 | 6.1602 | 5.9602 | 6.3457 |  |
| Quality-aware SR + LPIPS + GAN | BRISQUE | 150 | 27.4303 | 26.0786 | 28.6 |  |
| SR-only | PSNR | 150 | 30.9856 | 30.708 | 31.2652 |  |
| SR-only | SSIM | 150 | 0.8564 | 0.8482 | 0.8635 |  |
| SR-only | LPIPS | 150 | 0.1749 | 0.1684 | 0.1813 |  |
| SR-only | MAE | 150 | 0.0177 | 0.017 | 0.0184 |  |
| SR-only | Edge score | 150 | 0.9857 | 0.9852 | 0.9863 |  |
| SR-only | NIQE | 150 | 7.4523 | 7.2688 | 7.6632 |  |
| SR-only | BRISQUE | 150 | 54.5148 | 53.6033 | 55.414 |  |
| SR-only + LPIPS + GAN | PSNR | 150 | 27.8637 | 27.6268 | 28.0702 |  |
| SR-only + LPIPS + GAN | SSIM | 150 | 0.7814 | 0.774 | 0.7895 |  |
| SR-only + LPIPS + GAN | LPIPS | 150 | 0.1237 | 0.1204 | 0.1277 |  |
| SR-only + LPIPS + GAN | MAE | 150 | 0.0251 | 0.0242 | 0.0261 |  |
| SR-only + LPIPS + GAN | Edge score | 150 | 0.9813 | 0.9806 | 0.9819 |  |
| SR-only + LPIPS + GAN | NIQE | 150 | 6.8983 | 6.7028 | 7.0922 |  |
| SR-only + LPIPS + GAN | BRISQUE | 150 | 29.0869 | 27.9074 | 30.3213 |  |

## HC18 Anatomy Preservation CIs

| Method | Metric | N | Mean | 95% CI lower | 95% CI upper | Unit |
| --- | --- | --- | --- | --- | --- | --- |
| CNN-Residual Diffusion SR | inside-mask MAE | 150 | 0.0363 | 0.0349 | 0.0378 |  |
| CNN-Residual Diffusion SR | boundary error | 150 | 0.0349 | 0.0336 | 0.0364 |  |
| CNN-Residual Diffusion SR | boundary edge similarity | 150 | 0.9415 | 0.9387 | 0.9442 |  |
| Conditional Diffusion SR | inside-mask MAE | 150 | 0.0449 | 0.0431 | 0.0468 |  |
| Conditional Diffusion SR | boundary error | 150 | 0.0435 | 0.0418 | 0.0452 |  |
| Conditional Diffusion SR | boundary edge similarity | 150 | 0.932 | 0.9284 | 0.9353 |  |
| LQ mixed baseline | inside-mask MAE | 150 | 0.0688 | 0.0653 | 0.0719 |  |
| LQ mixed baseline | boundary error | 150 | 0.067 | 0.064 | 0.0701 |  |
| LQ mixed baseline | boundary edge similarity | 150 | 0.8217 | 0.8152 | 0.8281 |  |
| Quality-aware CNN | inside-mask MAE | 150 | 0.0325 | 0.0309 | 0.034 |  |
| Quality-aware CNN | boundary error | 150 | 0.0312 | 0.0299 | 0.0327 |  |
| Quality-aware CNN | boundary edge similarity | 150 | 0.9561 | 0.9538 | 0.9582 |  |
| Quality-aware CNN + L1/MSE | inside-mask MAE | 150 | 0.032 | 0.0304 | 0.0335 |  |
| Quality-aware CNN + L1/MSE | boundary error | 150 | 0.0305 | 0.0292 | 0.032 |  |
| Quality-aware CNN + L1/MSE | boundary edge similarity | 150 | 0.9563 | 0.9541 | 0.9584 |  |
| Quality-aware Diffusion SR v2 | inside-mask MAE | 150 | 0.0433 | 0.0416 | 0.045 |  |
| Quality-aware Diffusion SR v2 | boundary error | 150 | 0.0419 | 0.0403 | 0.0434 |  |
| Quality-aware Diffusion SR v2 | boundary edge similarity | 150 | 0.9408 | 0.9379 | 0.9434 |  |
| SR-only CNN | inside-mask MAE | 150 | 0.0316 | 0.0302 | 0.0329 |  |
| SR-only CNN | boundary error | 150 | 0.0303 | 0.0291 | 0.0316 |  |
| SR-only CNN | boundary edge similarity | 150 | 0.9582 | 0.956 | 0.9602 |  |

## ACOUSLIC-AI CIs

| Source | Metric | N | Mean | 95% CI lower | 95% CI upper | Unit | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ACOUSLIC-AI adaptive enhancement | frame improvement percentage | 10000 | 17.94 | 17.18 | 18.7 | % | Bootstrap over frames. |
| ACOUSLIC-AI sweep feedback | poor-frame-ratio reduction | 100 | 17.94 | 16.2495 | 19.5802 | percentage points | Bootstrap over sweeps. |
| ACOUSLIC-AI sweep feedback | good-frame-ratio improvement | 100 | 17.94 | 16.28 | 19.6422 | percentage points | Bootstrap over sweeps. |

## Outputs

- CSV: `results/bootstrap_confidence_intervals.csv`
- Figure directory: `results/bootstrap_ci_figures`

## Figures

- `results/bootstrap_ci_figures/hc18_reconstruction_bootstrap_ci.png`
- `results/bootstrap_ci_figures/hc18_anatomy_bootstrap_ci.png`
- `results/bootstrap_ci_figures/acouslic_ai_bootstrap_ci.png`
