# Final Ablation Tables

These tables summarize the proposed framework using proxy quality labels, not
expert clinical annotations. Paths are relative to the project root.

## Source Files Checked

| Source File | Status |
| --- | --- |
| `results/final_all_hc18_comparison_summary.csv` | available |
| `results/acouslic_ai/feedback_aware_enhancement_summary.csv` | available |
| `results/acouslic_ai/feedback_aware_enhancement_report.md` | available |
| `results/acouslic_ai/adaptive_feedback_enhancement_summary.csv` | available |
| `results/acouslic_ai/adaptive_feedback_enhancement_report.md` | available |
| `results/acouslic_ai/adaptive_sweep_feedback_change_summary.csv` | available |
| `results/acouslic_ai/adaptive_sweep_feedback_change_report.md` | available |

## Table 1. Original vs Blind Enhancement vs Adaptive Enhancement

| Method | Frame-Level Improvement | Gain over Blind Enhancement | Main Interpretation |
| --- | --- | --- | --- |
| Original / No Enhancement | 0.00% |  | Reference condition with no enhancement; proxy frame-quality improvement is defined as zero. |
| Blind Enhancement | 8.00% |  | Applies enhancement without adaptive feedback; improves a limited subset under proxy quality labels. |
| Adaptive Feedback-Aware Enhancement | 20.10% |  | Uses proxy quality labels to select keep, enhancement, denoising, or repeat/reacquisition actions. |
| Gain over Blind Enhancement |  | +12.10 percentage points | Adaptive feedback-aware decisions add 12.10 percentage points beyond blind enhancement. |

Interpretation: Adaptive feedback-aware enhancement reaches 20.10% frame-level
proxy improvement, a +12.10 percentage-point gain over blind enhancement. The
comparison should be described as proxy quality-label evidence rather than
expert clinical annotation.

## Table 2. CNN Only vs CNN + Residual Diffusion

| Model | Model Type | PSNR | SSIM | MAE | Edge Score | Main Interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| Quality-aware CNN + L1/MSE | CNN only | 30.5746 | 0.8737 | 0.017368 | 0.8969 | Strong supervised reconstruction baseline using proxy quality conditioning; strongest of these two rows on PSNR, SSIM, and MAE. |
| CNN-Residual Diffusion SR | CNN + residual diffusion | 29.8726 | 0.8686 | 0.020557 | 0.9542 | Residual diffusion improves edge/boundary refinement, reflected in the higher edge score, but it is not the best model for all metrics. |

Interpretation: The CNN-only model remains stronger on PSNR, SSIM, and MAE,
whereas the CNN-residual diffusion model has the stronger edge score. Diffusion
is therefore best framed as edge/boundary refinement, not as the best model for
all metrics.

## Table 3. With vs Without Feedback Reassessment

| Configuration | Frame-Level Improvement | Feedback Reassessment | Sweeps with Reduced Poor-Frame Ratio | Sweeps with Improved Good-Frame Ratio | repeat_sweep Cases Changed to acceptable/improve_visibility | Main Interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| Adaptive Enhancement without Reassessment | 20.10% | No | Not reassessed | Not reassessed | Not reassessed | Frame-level proxy quality improves, but the acquisition feedback decision is not recomputed. |
| Adaptive Enhancement with Reassessment | 20.10% | Yes | 50 | 50 | 18 | Recomputing feedback after enhancement verifies the closed-loop pipeline by changing repeat_sweep cases to acceptable or improve_visibility. |

Interpretation: Feedback reassessment verifies the closed-loop pipeline: after
adaptive enhancement, the final reassessment subset shows 50 sweeps with reduced
poor-frame ratio, 50 sweeps with improved good-frame ratio, and 18 repeat_sweep
cases changed to acceptable or improve_visibility.
