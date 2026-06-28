# Final Ablation Study

These tables summarize the complete acquisition-aware fetal ultrasound
framework using proxy ACOUSLIC-AI quality metrics and HC18 benchmark metrics.
They should not be interpreted as expert clinical validation.

## Inputs

- Blind enhancement summary: `results/acouslic_ai/feedback_aware_enhancement_summary.csv`
- Heuristic adaptive summary: `results/acouslic_ai/adaptive_feedback_enhancement_summary.csv`
- Learned adaptive summary: `results/acouslic_ai/learned_adaptive_enhancement_summary.csv`
- Adaptive sweep feedback summary: `results/acouslic_ai/adaptive_sweep_feedback_change_summary.csv`
- HC18 final comparison: `results/final_all_hc18_comparison_summary.csv`
- Anatomy preservation summary: `results/anatomy_preservation_all_models_summary.csv`

## Acquisition-aware Framework Ablation

| Method | Uses quality estimation | Uses adaptive action | Uses SR/CNN | Uses diffusion refinement | Frame improvement % | Sweeps with reduced poor-frame ratio | Repeat_sweep cases improved | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No enhancement | No | No | No | No | 0 | 0 | 0 | Baseline reference with no image modification. Proxy/benchmark metrics only; not expert clinical validation. |
| Blind enhancement | No adaptive decision | No | Yes | No | 8 | N/A | N/A | Quality-aware CNN enhancement applied without an adaptive frame-action policy. Proxy/benchmark metrics only; not expert clinical validation. |
| Heuristic adaptive enhancement | Heuristic proxy labels | Yes | Yes | No | 17.94 | 100 | 36 | Uses proxy frame labels to keep, enhance, denoise, or request repeat/reacquisition. Proxy/benchmark metrics only; not expert clinical validation. |
| Learned adaptive enhancement | Learned proxy embedding | Yes | Yes | No | 17.24 | N/A | N/A | Uses learned proxy-quality MLP confidence for conservative frame actions. Sweep-level change was not recomputed here. Proxy/benchmark metrics only; not expert clinical validation. |
| CNN-residual diffusion SR | No acquisition feedback | No | Yes | Yes | N/A | N/A | N/A | HC18 benchmark SR method; PSNR 29.8726, SSIM 0.8686, boundary edge similarity 0.9415. Proxy/benchmark metrics only; not expert clinical validation. |

## HC18 Model Ablation

| Method | Uses quality estimation | Uses adaptive action | Uses SR/CNN | Uses diffusion refinement | Frame improvement % | Sweeps with reduced poor-frame ratio | Repeat_sweep cases improved | Notes | Mean PSNR | Mean SSIM | Mean MAE | Mean edge score | Mean anatomy MAE inside mask | Mean boundary error | Mean contrast preservation | Mean boundary edge similarity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SR-only CNN | No | No | Yes | No | N/A | N/A | N/A | HC18 supervised SR/anatomy preservation benchmark. Proxy/benchmark metrics only; not expert clinical validation. | 30.7646 | 0.851 | 0.0182 | 0.9082 | 0.0316 | 0.0303 | 0.9727 | 0.9582 |
| Quality-aware CNN | Yes | No | Yes | No | N/A | N/A | N/A | HC18 supervised SR/anatomy preservation benchmark. Proxy/benchmark metrics only; not expert clinical validation. | 30.4013 | 0.8714 | 0.0177 | 0.8952 | 0.0325 | 0.0312 | 0.9625 | 0.9561 |
| Quality-aware CNN + L1/MSE | Yes | No | Yes | No | N/A | N/A | N/A | HC18 supervised SR/anatomy preservation benchmark. Proxy/benchmark metrics only; not expert clinical validation. | 30.5746 | 0.8737 | 0.0174 | 0.8969 | 0.032 | 0.0305 | 0.9678 | 0.9563 |
| Conditional Diffusion SR | No | No | No | Yes | N/A | N/A | N/A | HC18 supervised SR/anatomy preservation benchmark. Proxy/benchmark metrics only; not expert clinical validation. | 28.2914 | 0.8297 | 0.0254 | 0.9343 | 0.0449 | 0.0435 | 0.924 | 0.932 |
| CNN-Residual Diffusion SR | No acquisition feedback | No | Yes | Yes | N/A | N/A | N/A | HC18 supervised SR/anatomy preservation benchmark. Proxy/benchmark metrics only; not expert clinical validation. | 29.8726 | 0.8686 | 0.0206 | 0.9542 | 0.0363 | 0.0349 | 0.944 | 0.9415 |

## Saved Outputs

- Combined CSV: `results/final_ablation_study.csv`
- Figure directory: `results/final_ablation_figures`

## Figures

- `results/final_ablation_figures/acquisition_frame_improvement.png`
- `results/final_ablation_figures/acquisition_sweep_feedback_changes.png`
- `results/final_ablation_figures/hc18_psnr_ssim_ablation.png`
- `results/final_ablation_figures/hc18_anatomy_preservation_ablation.png`

## Interpretation Note

The ACOUSLIC-AI rows use no-reference proxy quality labels and sweep-feedback
proxies. The HC18 rows use supervised reconstruction and anatomy-preservation
benchmark metrics. These are useful ablations for method comparison, but they
are not clinical validation.
