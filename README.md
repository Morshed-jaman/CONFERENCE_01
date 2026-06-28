# Proxy Acquisition-Aware Fetal Ultrasound Enhancement with Adaptive Feedback and Diffusion-Based Boundary Refinement

This repository contains a conference-style research prototype for fetal ultrasound enhancement. The project is not only a diffusion super-resolution pipeline. Its main contribution is a quality-aware enhancement framework that estimates proxy acquisition quality, learns a quality embedding, selects an adaptive enhancement action, applies CNN and diffusion-inspired refinement, reassesses quality, and updates sweep-level feedback decisions.

The current implementation is an offline research workflow. It does not claim real-time clinical scanner deployment, expert clinical validation, real probe-motion or IMU supervision, or an exact DMUISR reproduction. Diffusion components should be interpreted as DMUISR-guided or diffusion-inspired, and ACOUSLIC-AI feedback labels are proxy labels rather than expert clinical annotations.

## Project Overview

Novice-acquired fetal ultrasound, especially in low-resource settings, can be blurry, noisy, low-contrast, unstable, incomplete, or otherwise poor quality. Conventional super-resolution enhances images, but it usually does not ask whether the frame or sweep was acquired well, whether enhancement should be applied differently for different quality states, or whether post-enhancement feedback should change.

This project addresses that gap by combining:

| Component | Role |
|---|---|
| HC18 fetal ultrasound SR | Controlled image reconstruction and anatomy-preservation evaluation |
| Proxy acquisition-quality estimation | Offline frame/sweep quality labels from image and sweep features |
| Learned proxy-quality embedding | Compact representation of quality state for adaptive enhancement |
| Adaptive enhancement decision | Heuristic and learned selection of enhancement action |
| CNN and residual diffusion refinement | Image enhancement with edge and boundary refinement |
| Feedback reassessment | Offline analysis of whether enhanced sweeps reduce poor-frame feedback |
| Improved HC18 U-Net segmentation | Supporting anatomy-preservation evidence using HC18 fetal-head masks |

## Main Workflow

```text
Low-quality fetal ultrasound image/sweep
-> proxy acquisition-quality estimation
-> learned quality embedding
-> adaptive enhancement decision
-> CNN / residual diffusion refinement
-> quality reassessment
-> updated feedback decision
```

## Key Contributions

- HC18-based super-resolution and anatomy-preservation evaluation.
- ACOUSLIC-AI proxy quality and sweep-feedback branch for blind-sweep fetal ultrasound.
- Adaptive enhancement instead of blind enhancement.
- Learned proxy-quality embedding for frame-level acquisition quality states.
- CNN-residual diffusion refinement for edge and boundary preservation.
- Feedback reassessment loop for offline sweep-level analysis.
- Segmentation-based anatomy preservation using an improved HC18-trained U-Net.

## Folder Structure

```text
Fetal_SR_Project/
|-- data/
|   |-- raw/
|   |   |-- HC18/
|   |   |   |-- training_set/
|   |   |   |-- test_set/
|   |   |   |-- training_set.zip
|   |   |   |-- test_set.zip
|   |-- processed/
|       |-- HC18_SR/
|       |   |-- HR/
|       |   |-- masks/
|       |   |-- LR_x2/
|       |   |-- LR_x4/
|       |   |-- LQ_mixed/
|       |   |-- train.csv
|       |   |-- val.csv
|       |   |-- test.csv
|       |-- ACOUSLIC_AI/
|           |-- frames/
|           |-- acouslic_frame_manifest.csv
|           |-- acouslic_proxy_quality_manifest.csv
|           |-- acouslic_sweep_quality_summary.csv
|-- scripts/
|   |-- 01_prepare_hc18.py
|   |-- 02_generate_lq_hc18.py
|   |-- 05_train_sr_only_hc18.py
|   |-- 07_train_quality_aware_sr_hc18.py
|   |-- 20_train_cnn_residual_diffusion_sr_hc18.py
|   |-- 24_extract_acouslic_mha_sample.py
|   |-- 24_compute_acouslic_proxy_quality.py
|   |-- 28_adaptive_feedback_aware_enhancement.py
|   |-- 29_train_proxy_quality_embedding.py
|   |-- 31_final_ablation_study.py
|   |-- 39_improve_hc18_unet_segmentation.py
|-- results/
|   |-- models/
|   |-- sr_outputs/
|   |-- acouslic_ai/
|   |-- final_ablation_figures/
|   |-- hc18_unet_segmentation_figures/
|   |-- final_all_hc18_comparison_summary.csv
|   |-- anatomy_preservation_all_models_summary.csv
|   |-- final_ablation_study.csv
|   |-- hc18_unet_improved_metrics_table.md
|   |-- novelty_comparison_table.md
|-- README.md
```

## Datasets

| Dataset | Use in this project |
|---|---|
| HC18 | Controlled single-frame fetal head ultrasound SR, reconstruction metrics, fetal-head mask anatomy preservation, and U-Net segmentation support. |
| ACOUSLIC-AI | Blind-sweep fetal ultrasound branch for proxy acquisition-quality estimation, adaptive enhancement analysis, and offline sweep-feedback reassessment. |

HC18 is used for paired low-quality/high-quality reconstruction experiments. ACOUSLIC-AI is used for proxy quality and sweep-feedback analysis; it is not treated as expert-labelled clinical feedback or real probe-motion supervision.

## Main Results

### HC18 Reconstruction

| Method | PSNR | SSIM | MAE | Edge score |
|---|---:|---:|---:|---:|
| LQ mixed baseline | 25.3376 | 0.6733 | 0.0338 | 0.8147 |
| SR-only CNN | **30.7646** | 0.8510 | 0.0182 | 0.9082 |
| Quality-aware CNN | 30.4013 | 0.8714 | 0.0177 | 0.8952 |
| Quality-aware CNN + L1/MSE | 30.5746 | **0.8737** | **0.0174** | 0.8969 |
| Conditional Diffusion SR | 28.2914 | 0.8297 | 0.0254 | 0.9343 |
| Quality-aware Diffusion SR v2 | 28.2739 | 0.7995 | 0.0264 | 0.9353 |
| CNN-Residual Diffusion SR | 29.8726 | 0.8686 | 0.0206 | **0.9542** |

Summary: SR-only CNN achieved the highest PSNR, Quality-aware CNN + L1/MSE achieved the best SSIM and MAE, and CNN-Residual Diffusion SR achieved the best edge score.

### Anatomy Preservation

| Method | Inside-mask MAE | Boundary error | Contrast preservation | Boundary edge similarity |
|---|---:|---:|---:|---:|
| LQ mixed baseline | 0.0688 | 0.0670 | 0.7292 | 0.8217 |
| SR-only CNN | **0.0316** | **0.0303** | **0.9727** | **0.9582** |
| Quality-aware CNN | 0.0325 | 0.0312 | 0.9625 | 0.9561 |
| Quality-aware CNN + L1/MSE | 0.0320 | 0.0305 | 0.9678 | 0.9563 |
| Conditional Diffusion SR | 0.0449 | 0.0435 | 0.9240 | 0.9320 |
| Quality-aware Diffusion SR v2 | 0.0433 | 0.0419 | 0.9472 | 0.9408 |
| CNN-Residual Diffusion SR | 0.0363 | 0.0349 | 0.9440 | 0.9415 |

### Acquisition-Aware Ablation

| Setting | Frame improvement | Sweep-level outcome |
|---|---:|---|
| Original / no enhancement | 0.00% | Baseline |
| Blind enhancement | 8.00% | No adaptive decision |
| Heuristic adaptive enhancement | 17.94% | Adaptive action from proxy quality labels |
| Learned adaptive enhancement | 17.24% | Uses learned proxy-quality embedding |
| With feedback reassessment | - | 100 sweeps reduced poor-frame ratio; 36 repeat-sweep cases improved |

### Learned Quality Embedding

| Split | Accuracy | Macro-F1 |
|---|---:|---:|
| Test | **0.9767** | **0.9688** |

### Improved HC18 U-Net Segmentation

Dice, IoU, and fetal measurement error are computed using an HC18-trained U-Net segmentation model with ellipse-fitted filled mask preprocessing and HC18 fetal-head masks. These metrics are used as supporting anatomy-preservation evidence and should not be interpreted as expert clinical validation.

| Metric | Value |
|---|---:|
| Best validation Dice | 0.8776 |
| Best validation IoU | 0.8130 |
| HR test Dice | 0.8897 |
| Best threshold | 0.50 |

## Important Tables and Figures

| File | Purpose |
|---|---|
| `results/final_all_hc18_comparison_summary.csv` | HC18 reconstruction comparison across CNN and diffusion-inspired models |
| `results/anatomy_preservation_all_models_summary.csv` | Mask-based anatomy-preservation summary |
| `results/final_ablation_study.csv` | Final acquisition-aware framework ablation |
| `results/final_ablation_figures/acquisition_frame_improvement.png` | Frame-level improvement figure |
| `results/final_ablation_figures/acquisition_sweep_feedback_changes.png` | Sweep feedback reassessment figure |
| `results/hc18_unet_improved_metrics_table.md` | Improved U-Net segmentation metrics report |
| `results/hc18_unet_segmentation_figures/improved_dice_iou_comparison.png` | U-Net Dice/IoU comparison figure |
| `results/acouslic_ai/proxy_quality_embedding_confusion_matrix.png` | Learned proxy-quality embedding confusion matrix |
| `results/novelty_comparison_table.md` | Positioning against related research directions |

## How to Run

Run commands from the project root in Windows PowerShell:

```powershell
cd D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project
```

Install dependencies in your Python environment:

```powershell
python -m pip install numpy pandas opencv-python pillow scikit-image matplotlib tqdm scikit-learn torch torchvision torchaudio
```

Prepare HC18 and generate low-quality paired data:

```powershell
python .\scripts\01_prepare_hc18.py
python .\scripts\02_generate_lq_hc18.py
```

Train and evaluate HC18 SR/enhancement models:

```powershell
python .\scripts\05_train_sr_only_hc18.py
python .\scripts\06_eval_sr_only_hc18.py
python .\scripts\07_train_quality_aware_sr_hc18.py
python .\scripts\11_train_diffusion_sr_hc18.py
python .\scripts\12_eval_diffusion_sr_hc18.py
python .\scripts\13_train_quality_aware_diffusion_sr_hc18.py
python .\scripts\14_eval_quality_aware_diffusion_sr_hc18.py
python .\scripts\20_train_cnn_residual_diffusion_sr_hc18.py
python .\scripts\21_eval_cnn_residual_diffusion_sr_hc18.py
python .\scripts\15_compare_all_hc18_models.py
```

Compute HC18 anatomy-preservation metrics:

```powershell
python .\scripts\09_anatomy_preservation_eval_hc18.py
```

Extract ACOUSLIC-AI samples and compute proxy quality:

```powershell
python .\scripts\23_inspect_acouslic_zip.py
python .\scripts\24_extract_acouslic_mha_sample.py
python .\scripts\24_compute_acouslic_proxy_quality.py
python .\scripts\25_generate_acouslic_feedback_report.py
```

Run adaptive enhancement and feedback reassessment:

```powershell
python .\scripts\26_feedback_aware_enhancement_analysis.py
python .\scripts\27_sweep_feedback_change_analysis.py
python .\scripts\28_adaptive_feedback_aware_enhancement.py
```

Train and evaluate the learned proxy-quality embedding:

```powershell
python .\scripts\29_train_proxy_quality_embedding.py
python .\scripts\30_learned_adaptive_feedback_enhancement.py
```

Run final ablations and supporting statistics:

```powershell
python .\scripts\31_final_ablation_study.py
python .\scripts\32_bootstrap_confidence_intervals.py
```

Run the improved HC18 U-Net segmentation analysis:

```powershell
python .\scripts\39_improve_hc18_unet_segmentation.py
```

Generate final report assets:

```powershell
python .\scripts\10_make_hc18_report_assets.py
python .\scripts\35_create_final_workflow_figure.py
python .\scripts\36_create_final_ablation_tables.py
```

## Evaluation Metrics

| Metric | What it measures |
|---|---|
| PSNR | Pixel-level reconstruction fidelity against HC18 HR targets |
| SSIM | Structural similarity between enhanced image and HR reference |
| MAE | Mean absolute reconstruction error |
| Edge score | Preservation of edge structure and boundary detail |
| Mask-based MAE | Reconstruction error inside or outside fetal-head mask regions |
| Boundary error | Error concentrated around fetal-head mask boundaries |
| Contrast preservation | Retention of foreground/background contrast around anatomy |
| Dice and IoU | Segmentation overlap from the HC18-trained U-Net support model |
| Fetal area error | Difference between predicted and target fetal-head mask area |
| Equivalent diameter error | Diameter proxy derived from fetal-head mask area |
| Frame-level improvement rate | Percentage of ACOUSLIC-AI frames improved by enhancement logic |
| Sweep feedback change | Offline change in sweep-level feedback after reassessment |
| Accuracy and macro-F1 | Learned proxy-quality embedding classification performance |

## Limitations

- Proxy labels are not expert clinical annotations.
- Feedback analysis is offline and should not be described as real-time clinical scanner feedback.
- The project does not use real probe-motion or IMU labels.
- The project does not demonstrate clinical deployment.
- Expert clinical validation is required before any clinical interpretation.
- The current adaptive policy is heuristic or proxy-learning based; a fully learned action policy is future work.
- Diffusion experiments are DMUISR-guided or diffusion-inspired rather than an exact DMUISR reproduction.

## Citation / Paper Status

This repository supports a conference-style research prototype. The final paper should frame the contribution as a combined applied machine-learning and medical-imaging framework.

## Research Framing

The strongest framing is not "diffusion SR for fetal ultrasound" alone. The contribution is a proxy acquisition-aware fetal ultrasound enhancement framework that links quality estimation, adaptive enhancement, CNN/diffusion-inspired restoration, anatomy-preservation evaluation, and offline sweep-feedback reassessment.
