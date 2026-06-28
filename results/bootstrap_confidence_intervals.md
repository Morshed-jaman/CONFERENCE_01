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
| CNN-Residual Diffusion SR | PSNR | 150 | 29.8726 | 29.7001 | 30.0506 |  |
| CNN-Residual Diffusion SR | SSIM | 150 | 0.8686 | 0.8658 | 0.8714 |  |
| CNN-Residual Diffusion SR | MAE | 150 | 0.0206 | 0.02 | 0.0211 |  |
| CNN-Residual Diffusion SR | edge_score | 150 | 0.9542 | 0.9523 | 0.9559 |  |
| Conditional Diffusion SR | PSNR | 150 | 28.2914 | 28.1077 | 28.4791 |  |
| Conditional Diffusion SR | SSIM | 150 | 0.8297 | 0.8257 | 0.8338 |  |
| Conditional Diffusion SR | MAE | 150 | 0.0254 | 0.0247 | 0.0261 |  |
| Conditional Diffusion SR | edge_score | 150 | 0.9343 | 0.9318 | 0.9366 |  |
| LQ mixed baseline | PSNR | 150 | 25.3376 | 25.1361 | 25.5339 |  |
| LQ mixed baseline | SSIM | 150 | 0.6733 | 0.6624 | 0.684 |  |
| LQ mixed baseline | MAE | 150 | 0.0338 | 0.0326 | 0.0351 |  |
| LQ mixed baseline | edge_score | 150 | 0.8147 | 0.8098 | 0.8194 |  |
| Quality-aware CNN | PSNR | 150 | 30.4013 | 30.1584 | 30.6637 |  |
| Quality-aware CNN | SSIM | 150 | 0.8714 | 0.866 | 0.8763 |  |
| Quality-aware CNN | MAE | 150 | 0.0177 | 0.017 | 0.0185 |  |
| Quality-aware CNN | edge_score | 150 | 0.8952 | 0.8907 | 0.8994 |  |
| Quality-aware CNN + L1/MSE | PSNR | 150 | 30.5746 | 30.3064 | 30.8639 |  |
| Quality-aware CNN + L1/MSE | SSIM | 150 | 0.8737 | 0.8685 | 0.8788 |  |
| Quality-aware CNN + L1/MSE | MAE | 150 | 0.0174 | 0.0167 | 0.018 |  |
| Quality-aware CNN + L1/MSE | edge_score | 150 | 0.8969 | 0.8932 | 0.9013 |  |
| Quality-aware Diffusion SR v2 | PSNR | 150 | 28.2739 | 28.0808 | 28.4746 |  |
| Quality-aware Diffusion SR v2 | SSIM | 150 | 0.7995 | 0.7942 | 0.8053 |  |
| Quality-aware Diffusion SR v2 | MAE | 150 | 0.0264 | 0.0256 | 0.0273 |  |
| Quality-aware Diffusion SR v2 | edge_score | 150 | 0.9353 | 0.9328 | 0.9379 |  |
| SR-only CNN | PSNR | 150 | 30.7646 | 30.4716 | 31.0466 |  |
| SR-only CNN | SSIM | 150 | 0.851 | 0.8432 | 0.8588 |  |
| SR-only CNN | MAE | 150 | 0.0182 | 0.0175 | 0.019 |  |
| SR-only CNN | edge_score | 150 | 0.9082 | 0.9038 | 0.9122 |  |

## HC18 Anatomy Preservation CIs

| Method | Metric | N | Mean | 95% CI lower | 95% CI upper | Unit |
| --- | --- | --- | --- | --- | --- | --- |
| CNN-Residual Diffusion SR | inside-mask MAE | 150 | 0.0363 | 0.0349 | 0.0377 |  |
| CNN-Residual Diffusion SR | boundary error | 150 | 0.0349 | 0.0336 | 0.0363 |  |
| CNN-Residual Diffusion SR | boundary edge similarity | 150 | 0.9415 | 0.9387 | 0.9444 |  |
| Conditional Diffusion SR | inside-mask MAE | 150 | 0.0449 | 0.0431 | 0.0467 |  |
| Conditional Diffusion SR | boundary error | 150 | 0.0435 | 0.0418 | 0.0452 |  |
| Conditional Diffusion SR | boundary edge similarity | 150 | 0.932 | 0.9287 | 0.9352 |  |
| LQ mixed baseline | inside-mask MAE | 150 | 0.0688 | 0.0654 | 0.0719 |  |
| LQ mixed baseline | boundary error | 150 | 0.067 | 0.0641 | 0.07 |  |
| LQ mixed baseline | boundary edge similarity | 150 | 0.8217 | 0.8152 | 0.8283 |  |
| Quality-aware CNN | inside-mask MAE | 150 | 0.0325 | 0.0311 | 0.034 |  |
| Quality-aware CNN | boundary error | 150 | 0.0312 | 0.0299 | 0.0327 |  |
| Quality-aware CNN | boundary edge similarity | 150 | 0.9561 | 0.9539 | 0.9582 |  |
| Quality-aware CNN + L1/MSE | inside-mask MAE | 150 | 0.032 | 0.0305 | 0.0334 |  |
| Quality-aware CNN + L1/MSE | boundary error | 150 | 0.0305 | 0.0292 | 0.0319 |  |
| Quality-aware CNN + L1/MSE | boundary edge similarity | 150 | 0.9563 | 0.954 | 0.9584 |  |
| Quality-aware Diffusion SR v2 | inside-mask MAE | 150 | 0.0433 | 0.0417 | 0.0451 |  |
| Quality-aware Diffusion SR v2 | boundary error | 150 | 0.0419 | 0.0402 | 0.0435 |  |
| Quality-aware Diffusion SR v2 | boundary edge similarity | 150 | 0.9408 | 0.9379 | 0.9435 |  |
| SR-only CNN | inside-mask MAE | 150 | 0.0316 | 0.0301 | 0.0329 |  |
| SR-only CNN | boundary error | 150 | 0.0303 | 0.029 | 0.0316 |  |
| SR-only CNN | boundary edge similarity | 150 | 0.9582 | 0.9559 | 0.9604 |  |

## ACOUSLIC-AI CIs

| Source | Metric | N | Mean | 95% CI lower | 95% CI upper | Unit | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ACOUSLIC-AI adaptive enhancement | frame improvement percentage | 10000 | 17.94 | 17.1997 | 18.7 | % | Bootstrap over frames. |
| ACOUSLIC-AI sweep feedback | poor-frame-ratio reduction | 100 | 17.94 | 16.3 | 19.69 | percentage points | Bootstrap over sweeps. |
| ACOUSLIC-AI sweep feedback | good-frame-ratio improvement | 100 | 17.94 | 16.3998 | 19.55 | percentage points | Bootstrap over sweeps. |

## Outputs

- CSV: `results/bootstrap_confidence_intervals.csv`
- Figure directory: `results/bootstrap_ci_figures`

## Figures

- `results/bootstrap_ci_figures/hc18_reconstruction_bootstrap_ci.png`
- `results/bootstrap_ci_figures/hc18_anatomy_bootstrap_ci.png`
- `results/bootstrap_ci_figures/acouslic_ai_bootstrap_ci.png`
