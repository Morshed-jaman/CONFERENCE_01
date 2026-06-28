# Final Table and Figure Checklist

Paper title: Proxy Acquisition-Aware Fetal Ultrasound Enhancement with Adaptive Feedback and Diffusion-Based Boundary Refinement

This checklist tracks the tables and figures used by the final conference-style report and appendix.

## Main Report Tables

| ID | Table | Source artifact | Status |
|---|---|---|---|
| Table 1 | HC18 reconstruction comparison | `results/final_all_hc18_comparison_summary.csv` | Included in `reports/final_conference_report.md` |
| Table 2 | Mask-based anatomy preservation | `results/anatomy_preservation_all_models_summary.csv` | Included in `reports/final_conference_report.md` |
| Table 3 | Segmentation-based anatomy preservation | `results/hc18_unet_improved_metrics_summary.csv`, `results/hc18_unet_improved_metrics_table.md` | Included in `reports/final_conference_report.md` |
| Table 4 | Proxy frame-quality distribution | `data/processed/ACOUSLIC_AI/acouslic_proxy_quality_manifest.csv`, `results/acouslic_ai/acouslic_proxy_quality_summary.md` | Included in `reports/final_conference_report.md` |
| Table 5 | Sweep feedback distribution | `data/processed/ACOUSLIC_AI/acouslic_sweep_quality_summary.csv`, `results/acouslic_ai/feedback_report/acouslic_feedback_report.md` | Included in `reports/final_conference_report.md` |
| Table 6 | Main ablation table | `results/final_ablation_study.csv` | Included in `reports/final_conference_report.md` |
| Table 7 | Feedback reassessment table | `results/acouslic_ai/adaptive_sweep_feedback_change_summary.csv` | Included in `reports/final_conference_report.md` |
| Table 8 | Learned proxy-quality embedding | `results/acouslic_ai/proxy_quality_embedding_metrics.csv` | Included in `reports/final_conference_report.md` |

## Main Report Figures

| Figure | Requested artifact | Status | Suggested placement |
|---|---|---|---|
| Overall framework figure | `results/final_workflow_figure.png` | Not found in current results folder | Generate before submission if a visual overview is required |
| HC18 SR/anatomy branch figure | `results/report_assets/figure_psnr_comparison.png` and `results/report_assets/figure_boundary_error.png` | Available | Main report or appendix |
| Acquisition frame improvement | `results/final_ablation_figures/acquisition_frame_improvement.png` | Available | Main report Results |
| Acquisition sweep feedback changes | `results/final_ablation_figures/acquisition_sweep_feedback_changes.png` | Available | Main report Results |
| Improved U-Net Dice/IoU comparison | `results/hc18_unet_segmentation_figures/improved_dice_iou_comparison.png` | Available | Main report segmentation subsection |
| Frame quality distribution | `results/acouslic_ai/feedback_report/frame_quality_distribution.png` | Available | Main report ACOUSLIC-AI proxy quality subsection |
| Sweep feedback distribution | `results/acouslic_ai/feedback_report/sweep_feedback_distribution.png` | Available | Main report ACOUSLIC-AI feedback subsection |
| Proxy-quality embedding confusion matrix | `results/acouslic_ai/proxy_quality_embedding_confusion_matrix.png` | Available | Main report learned embedding subsection |

## Appendix Tables and Figures

| Artifact | Path | Status | Recommended use |
|---|---|---|---|
| U-Net threshold tuning table | `results/hc18_unet_improved_metrics_table.md` | Available | Appendix |
| U-Net training curve | `results/hc18_unet_segmentation_figures/improved_training_curve.png` | Available | Appendix |
| U-Net example segmentation outputs | `results/hc18_unet_segmentation_figures/improved_example_segmentation_outputs.png` | Available | Appendix |
| U-Net measurement error comparison | `results/hc18_unet_segmentation_figures/improved_measurement_error_comparison.png` | Available | Appendix |
| Mask statistics | `results/hc18_unet_improved_mask_statistics.csv` | Available | Appendix |
| Image-mask pairing grid | `results/hc18_unet_segmentation_figures/debug_image_mask_pairing.png` | Available | Appendix or development supplement |
| Overfit sanity-check figure | `results/hc18_unet_segmentation_figures/overfit_sanity_check.png` | Available | Appendix or development supplement |
| HC18 reconstruction bootstrap CI | `results/bootstrap_ci_figures/hc18_reconstruction_bootstrap_ci.png` | Available | Appendix |
| HC18 anatomy bootstrap CI | `results/bootstrap_ci_figures/hc18_anatomy_bootstrap_ci.png` | Available | Appendix |
| ACOUSLIC-AI bootstrap CI | `results/bootstrap_ci_figures/acouslic_ai_bootstrap_ci.png` | Available | Appendix |
| Manual review CSV | `results/acouslic_ai/manual_review/proxy_label_manual_review.csv` | Available | Appendix |
| Good-frame contact sheet | `results/acouslic_ai/manual_review/contact_sheets/good_frame_contact_sheet.png` | Available | Appendix |
| Blurry-frame contact sheet | `results/acouslic_ai/manual_review/contact_sheets/blurry_frame_contact_sheet.png` | Available | Appendix |
| Low-contrast-frame contact sheet | `results/acouslic_ai/manual_review/contact_sheets/low_contrast_frame_contact_sheet.png` | Available | Appendix |
| Noisy-frame contact sheet | `results/acouslic_ai/manual_review/contact_sheets/noisy_frame_contact_sheet.png` | Available | Appendix |
| Poor-frame contact sheet | `results/acouslic_ai/manual_review/contact_sheets/poor_frame_contact_sheet.png` | Available | Appendix |

## Claim-Safety Checklist

| Requirement | Status |
|---|---|
| Frame paper as quality-aware fetal ultrasound enhancement and feedback framework | Satisfied |
| Avoid expert clinical validation claim | Satisfied |
| Avoid real-time clinical deployment claim | Satisfied |
| Avoid real probe-motion / IMU label claim | Satisfied |
| Avoid exact DMUISR reproduction claim | Satisfied |
| Use proxy acquisition-quality labels wording | Satisfied |
| Use offline feedback analysis wording | Satisfied |
| Use DMUISR-guided / diffusion-inspired wording | Satisfied |
| Present diffusion as boundary/edge refinement | Satisfied |
| Exclude failed old Dice/IoU results | Satisfied |
| Exclude old weak U-Net result | Satisfied |
| Use latest 100-sweep ACOUSLIC-AI results where available | Satisfied |

## Files Created

| File | Purpose |
|---|---|
| `reports/final_conference_report.md` | Main conference-style report |
| `reports/final_conference_report_appendix.md` | Appendix with supporting details and figures |
| `reports/final_table_figure_checklist.md` | Table/figure and claim-safety checklist |

