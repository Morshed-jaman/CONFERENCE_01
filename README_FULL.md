# Fetal_SR_Project: Fetal Ultrasound Super-Resolution and Downstream Utility

This repository implements an end-to-end fetal ultrasound image restoration pipeline on HC18 fetal-head ultrasound images, with an auxiliary ACOUSLIC-AI branch for blind-sweep quality assessment and feedback-aware enhancement.

## 1. Project Overview

**Headline result:** SwinIR-Light + perceptual+GAN, LPIPS `0.1009 [0.0969, 0.1050]`, Dice `0.8877`, `p_bonf = 3.99e-10` vs CNN champion, commit `3d0f9fa`.

The main contribution is a clinically oriented comparison pipeline that evaluates super-resolution not only by reconstruction fidelity and perceptual quality, but also by downstream fetal-head segmentation utility and hard-subset robustness.

The canonical HC18 pipeline:

1. Prepares HC18 fetal-head images and masks at 256x256.
2. Generates realistic low-quality ultrasound inputs through downsampling, blur, speckle, motion blur, and contrast reduction.
3. Trains CNN, quality-aware CNN, diffusion, perceptual/GAN, SwinIR-Light, and segmentation-aware variants.
4. Evaluates full-test reconstruction quality using PSNR, SSIM, LPIPS-VGG, DISTS, MAE, edge similarity, NIQE, and BRISQUE.
5. Evaluates downstream fetal-head segmentation utility using a frozen HC18 U-Net.
6. Runs hard-subset analysis and paired significance tests with Bonferroni correction.

The best canonical reconstruction model is **SwinIR-Light Phase A**, saved at `results/models/swinir_light_phase_a_seed42/best_lpips.pth`. On the 150-image HC18 test split it reaches the lowest LPIPS and DISTS among the registered CNN/SwinIR comparison set while preserving downstream Dice at the LQ/HR ceiling observed for the frozen U-Net.

## 2. Repository Structure

Top-level folders audited in this repository:

| Folder | Purpose |
|---|---|
| `.agents/` | Local agent/task metadata. Not part of the scientific pipeline. |
| `.git/` | Git object database and history. |
| `data/` | Raw and processed datasets. Large processed/raw files are intentionally git-ignored. |
| `reports/` | Report artifacts and manuscript-facing generated summaries. |
| `results/` | Model checkpoints, evaluation CSVs, figures, visual comparisons, masks, and training logs. |
| `scripts/` | Data preparation, model training, evaluation, statistical analysis, and utility scripts. |

Representative tree to depth 3:

```text
Fetal_SR_Project/
|-- data/
|   |-- processed/
|   |   |-- HC18_SR/
|   |   `-- ACOUSLIC_AI/
|   `-- raw/
|       `-- HC18/
|-- reports/
|-- results/
|   |-- acouslic_ai/
|   |-- bootstrap_ci_figures/
|   |-- downstream_hard_subset_figures/
|   |-- downstream_utility_masks/
|   |-- eval/
|   |   `-- swinir_light_phase_a/
|   |-- final_ablation_figures/
|   |-- final_all_visual_comparison/
|   |-- hc18_unet_segmentation_figures/
|   |-- models/
|   |   |-- phaseB_segaware_swinir_seed42/
|   |   `-- swinir_light_phase_a_seed42/
|   |-- report_assets/
|   |-- sr_outputs/
|   `-- training_logs/
`-- scripts/
    |-- models/
    |   `-- swinir_light.py
    |-- 01_prepare_hc18.py
    |-- 02_generate_lq_hc18.py
    |-- 42_compare_all_models.py
    |-- 46_downstream_utility_eval.py
    |-- 47_hard_subset_analysis.py
    `-- 48_significance_report.py
```

Important generated-output conventions:

| Path | Role |
|---|---|
| `results/models/` | All trained checkpoint files; ignored by git because checkpoints are large. |
| `results/training_logs/` | Per-epoch CSV logs for training runs. |
| `results/eval/swinir_light_phase_a/test/` | SwinIR-specific reconstruction, downstream, and hard-subset test outputs. |
| `results/downstream_utility_masks/` | Frozen U-Net predicted masks for each evaluated method. |
| `results/final_all_visual_comparison/` | Curated HC18 visual comparisons for the final model set. |
| `results/bootstrap_ci_figures/` | Bootstrap CI plots for HC18 reconstruction, HC18 anatomy, and ACOUSLIC-AI. |
| `results/final_ablation_figures/` | Final ablation-study figures. |
| `results/report_assets/` | Manuscript/report figures and tables. |

There is no `results/figures/` directory in the audited tree. Publication-facing figures are instead under `results/bootstrap_ci_figures/`, `results/downstream_hard_subset_figures/`, `results/final_ablation_figures/`, `results/hc18_unet_segmentation_figures/`, and `results/report_assets/`.

## 3. Datasets

### HC18 Fetal Head Ultrasound

| Item | Value |
|---|---|
| Source page | `https://hc18.grand-challenge.org/` |
| Local raw path | `data/raw/HC18/training_set/` |
| Processed path | `data/processed/HC18_SR/` |
| Manifest | `data/processed/HC18_SR/hc18_lq_manifest.csv` |
| Image format | Grayscale fetal-head ultrasound PNG, resized to 256x256 |
| Mask format | Resized annotation/mask PNG, nearest-neighbor interpolation |
| Split counts | train 699, val 150, test 150 |
| Canonical test split | 150 images for reconstruction; 148 mask-valid images for downstream Dice |

`scripts/01_prepare_hc18.py` reads HC18 images and masks, excludes annotation/mask files from the raw image list, resizes to 256x256, creates HR and mask folders, and writes train/val/test manifests using a 70/15/15 split with seed 42.

### HC18 Low-Quality Degradation

`scripts/02_generate_lq_hc18.py` creates three degraded inputs:

| Output | Degradation |
|---|---|
| `LR_x2/` | Bicubic downsample by 2, then upsample to 256x256. |
| `LR_x4/` | Bicubic downsample by 4, then upsample to 256x256. |
| `LQ_mixed/` | LR_x4 plus Gaussian blur radius 1.5, multiplicative speckle noise sigma 0.02, horizontal motion blur kernel 5, and contrast factor 0.85. |

The final canonical SR comparison uses `LQ_mixed` as the low-quality input.

### ACOUSLIC-AI Branch

| Item | Value |
|---|---|
| Source page | `https://acouslic-ai.grand-challenge.org/` |
| Expected local archive | `D:/acouslic-ai-train-set.zip` or `D:/acouslic-ai-train-set/` |
| Processed path | `data/processed/ACOUSLIC_AI/` |
| Results path | `results/acouslic_ai/` |
| Purpose in this repo | Blind-sweep frame quality proxy labels, feedback-aware enhancement, adaptive enhancement, and learned quality embedding experiments. |

The ACOUSLIC-AI branch is not the source of the HC18 SR headline. It is an adjacent quality-assessment and acquisition-feedback branch that supports the broader fetal ultrasound quality theme.

## 4. Environment

Audited project environment:

| Component | Value |
|---|---|
| OS/shell | Windows 10, PowerShell |
| Python | 3.13.1 |
| PyTorch | 2.12.1+cu126 |
| CUDA runtime | 12.6 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| VRAM | 8 GB, user-specified |

`requirements.txt` contains:

```text
numpy
pillow
scikit-image
torch
torchvision
tqdm
lpips
pyiqa
timm
torchmetrics
pytorch-msssim
kornia
```

Known package notes:

| Package | Status |
|---|---|
| `torchmetrics` | Present in `requirements.txt`. |
| `pytorch-msssim` | Present in `requirements.txt`; imported in Python as `pytorch_msssim`. |
| `kornia` | Present in `requirements.txt`. |
| `cv2` / OpenCV | Used by several scripts, but not listed in `requirements.txt`; the user's project Python had it available during successful model runs. |
| `pyiqa` | Used for DISTS, NIQE, and BRISQUE. |
| `lpips` | Used for LPIPS-VGG and perceptual training/evaluation. |

Use the project Python environment, not an isolated sandbox environment, for any commands that require `cv2`, `torch`, `lpips`, or `pyiqa`.

## 5. Pipeline Overview

### Data Preparation

| Step | Script | Inputs | Outputs |
|---|---|---|---|
| HC18 preparation | `scripts/01_prepare_hc18.py` | `data/raw/HC18/training_set/` | `data/processed/HC18_SR/HR/`, `masks/`, `hc18_manifest.csv`, split CSVs |
| LQ generation | `scripts/02_generate_lq_hc18.py` | HC18 HR manifest | `LR_x2/`, `LR_x4/`, `LQ_mixed/`, `hc18_lq_manifest.csv` |
| Visual check | `scripts/03_visual_check_hc18.py` | Processed HC18 samples | `results/visual_check/*.png` |

### Training Scripts

| Script | Dataset/input | Architecture | Losses/objective | Main checkpoint/output |
|---|---|---|---|---|
| `scripts/05_train_sr_only_hc18.py` | HC18 `LQ_mixed` to HR | Small residual CNN `ResidualSRModel` | L1 reconstruction | `results/models/sr_only_hc18_best.pth` |
| `scripts/07_train_quality_aware_sr_hc18.py` | HC18 `LQ_mixed` + quality features | FiLM-conditioned residual CNN `QualityAwareSRModel` | `0.8*L1 + 0.2*MSE` | `results/models/quality_aware_sr_hc18_l1_mse_best.pth` |
| `scripts/11_train_diffusion_sr_hc18.py` | HC18 `LQ_mixed` to HR | Conditional diffusion U-Net | Noise-prediction MSE | `results/models/diffusion_sr_hc18_best.pth` |
| `scripts/13_train_quality_aware_diffusion_sr_hc18.py` | HC18 `LQ_mixed` + quality features | Quality-aware conditional diffusion U-Net | Noise-prediction MSE | `results/models/quality_aware_diffusion_sr_hc18_best.pth` and v2 checkpoint |
| `scripts/16_train_dmuisr_inspired_diffusion_sr_hc18.py` | HC18 `LQ_mixed` | DMUISR-inspired conditional diffusion | Noise, x0, and edge-aware objectives | `results/models/dmuisr_inspired_diffusion_sr_hc18_best.pth` |
| `scripts/18_train_diffusion_sr_x0loss_hc18.py` | HC18 `LQ_mixed` | Conditional diffusion U-Net | Noise loss plus x0 reconstruction loss | `results/models/diffusion_sr_x0loss_hc18_best.pth` |
| `scripts/20_train_cnn_residual_diffusion_sr_hc18.py` | HC18 with CNN guide | CNN-residual conditional diffusion U-Net | Residual/noise-prediction objective around CNN output | `results/models/cnn_residual_diffusion_sr_hc18_best.pth` |
| `scripts/29_train_proxy_quality_embedding.py` | ACOUSLIC proxy-quality manifest | MLP quality embedding/classifier | Cross-entropy proxy-label classification | `results/models/proxy_quality_embedding_mlp.pth` |
| `scripts/39_improve_hc18_unet_segmentation.py` | HC18 masks and image variants | Improved U-Net | BCE + Dice segmentation loss | `results/models/hc18_unet_segmentation_best_improved.pth` |
| `scripts/40_train_sr_perceptual_hc18.py` | HC18 `LQ_mixed` | Residual CNN or quality-aware CNN | L1/MSE + LPIPS + optional PatchGAN BCE | `results/models/sr_perceptual_hc18_*.pth`, `qa_sr_perceptual_hc18_*.pth` |
| `scripts/50_train_swinir_hc18.py` | HC18 `LQ_mixed` | `SwinIRLight` in `scripts/models/swinir_light.py` | L1 + LPIPS + PatchGAN BCE, EMA | `results/models/swinir_light_phase_a_seed42/{best_lpips,best_dice,last,ema}.pth` |
| `scripts/51_train_segaware_hc18.py` | HC18 `LQ_mixed`, optional pretrained SR | SwinIR-Light or QA CNN Phase B | Charbonnier + LPIPS + DISTS + Sobel edge + GAN + frozen U-Net segmentation BCE/Dice, hard-example weighting | `results/models/phaseB_segaware_swinir_seed42/{best_lpips,best_dice,last,ema}.pth` |

Phase B segmentation-aware training exists in the repository but is not part of the canonical full-test headline. The audited Phase B log ended at epoch 8 with no validation metrics recorded in the CSV rows inspected, so it should be treated as a deferred or incomplete ablation until a full validation/test report is produced.

### Evaluation and Analysis Scripts

| Script | Evaluated models/data | Metrics | Main CSV outputs |
|---|---|---|---|
| `scripts/04_eval_bicubic_hc18.py` | Bicubic x2, bicubic x4, `LQ_mixed` baseline | PSNR, SSIM, MAE | `results/bicubic_hc18_per_image.csv`, `results/bicubic_hc18_summary.csv` |
| `scripts/06_eval_sr_only_hc18.py` | SR-only CNN | PSNR, SSIM, MAE | `results/sr_only_hc18_per_image.csv`, `results/sr_only_hc18_summary.csv` |
| `scripts/08_compare_hc18_models.py` | Early HC18 model comparison | PSNR, SSIM, MAE and visual comparisons | Early comparison outputs |
| `scripts/09_anatomy_preservation_eval_hc18.py` | Anatomy-preservation variants | Inside/outside-mask MAE, boundary error, contrast preservation, edge similarity | `results/anatomy_preservation_*.csv` |
| `scripts/12_eval_diffusion_sr_hc18.py` | Diffusion SR | PSNR, SSIM, MAE, EdgeScore, inference time | `results/diffusion_sr_hc18_*.csv` |
| `scripts/14_eval_quality_aware_diffusion_sr_hc18.py` | Quality-aware diffusion SR | PSNR, SSIM, MAE, EdgeScore, inference time | `results/quality_aware_diffusion_sr_hc18_*.csv` |
| `scripts/15_compare_all_hc18_models.py` | Earlier all-model comparison | PSNR, SSIM, MAE, EdgeScore | `results/final_hc18_comparison_*.csv` |
| `scripts/17_eval_dmuisr_inspired_diffusion_sr_hc18.py` | DMUISR-inspired diffusion SR | PSNR, SSIM, MAE, EdgeScore, inference time | `results/dmuisr_inspired_diffusion_sr_hc18_*.csv` |
| `scripts/19_eval_diffusion_sr_x0loss_hc18.py` | x0-loss diffusion SR | PSNR, SSIM, MAE, EdgeScore, inference time | `results/diffusion_sr_x0loss_hc18_*.csv` |
| `scripts/21_eval_cnn_residual_diffusion_sr_hc18.py` | CNN-residual diffusion SR | PSNR, SSIM, MAE, EdgeScore, inference time | `results/cnn_residual_diffusion_sr_hc18_*.csv` |
| `scripts/32_bootstrap_confidence_intervals.py` | HC18 and ACOUSLIC result CSVs | Bootstrap mean CIs | `results/bootstrap_confidence_intervals.csv` |
| `scripts/36_create_final_ablation_tables.py` | Final ablation outputs | Report-table summaries | `results/final_ablation_*.csv` |
| `scripts/42_compare_all_models.py` | LQ, SR-only, QA, SR+perc, QA+perc, SwinIR | PSNR, SSIM, LPIPS, DISTS, MAE, EdgeScore, NIQE, BRISQUE | `results/final_all_hc18_comparison_per_image.csv`, `results/final_all_hc18_comparison_summary.csv`, SwinIR copies under `results/eval/swinir_light_phase_a/test/` |
| `scripts/46_downstream_utility_eval.py` | HR, LQ, SR-only, SR+perc, QA, QA+perc, SwinIR, CNN-residual diffusion | Dice, IoU, area error, equivalent-diameter error | `results/downstream_utility_per_image.csv`, `results/downstream_utility_summary.csv`, SwinIR copies under `results/eval/swinir_light_phase_a/test/` |
| `scripts/47_hard_subset_analysis.py` | Same downstream methods over hard subsets | Dice/IoU/area/diameter deltas vs LQ | `results/downstream_hard_subset_membership.csv`, `results/downstream_hard_subset_summary.csv`, `results/eval/swinir_light_phase_a/test/hard_subsets.csv` |
| `scripts/48_significance_report.py` | Registered reconstruction/downstream methods | Bootstrap CIs, paired Wilcoxon, Bonferroni correction | `results/bootstrap_metric_summary.csv`, `results/significance_report.csv` |
| `scripts/49_phase_a_diagnosis.py` | SwinIR Phase A log | LPIPS trend and gate-crossing diagnosis | `results/phase_a_diagnosis/phase_a_diagnosis_summary.csv` |

## 6. How to Reproduce

All commands below assume PowerShell in the repository root:

```powershell
cd D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project
```

### Prepare HC18

```powershell
python .\scripts\01_prepare_hc18.py
python .\scripts\02_generate_lq_hc18.py
python .\scripts\03_visual_check_hc18.py
```

### Train Canonical CNN Baselines

```powershell
python .\scripts\05_train_sr_only_hc18.py --epochs 30 --batch_size 8 --lr 1e-4

python .\scripts\07_train_quality_aware_sr_hc18.py --epochs 30 --batch_size 16 --lr 1e-4
```

### Train Perceptual/GAN CNN Champions

```powershell
python .\scripts\40_train_sr_perceptual_hc18.py `
    --arch sr_only `
    --epochs 40 --batch_size 8 --lr 1e-4 `
    --input_type LQ_mixed `
    --l1_weight 1.0 --mse_weight 0.0 `
    --lpips_weight 0.1 --gan_weight 0.05 `
    --seed 42 --run_name sr_perceptual

python .\scripts\40_train_sr_perceptual_hc18.py `
    --arch quality_aware `
    --epochs 40 --batch_size 8 --lr 1e-4 `
    --input_type LQ_mixed `
    --l1_weight 1.0 --mse_weight 0.0 `
    --lpips_weight 0.1 --gan_weight 0.05 `
    --seed 42 --run_name qa_sr_perceptual
```

### Train SwinIR-Light Phase A

```powershell
python .\scripts\50_train_swinir_hc18.py `
    --epochs 100 --batch_size 8 --lr 2e-4 --d_lr 1e-4 `
    --warmup_epochs 5 --seed 42 --input_type LQ_mixed `
    --l1_weight 1.0 --lpips_weight 0.2 --gan_weight 0.005 `
    --ema_decay 0.999 --grad_clip 1.0 `
    --eval_dice_every 5 --num_workers 0 `
    --window_size 8 --depths 6,6,6,6 `
    --run_name swinir_light_phase_a_seed42
```

The canonical hero checkpoint is:

```text
results/models/swinir_light_phase_a_seed42/best_lpips.pth
```

### Optional Phase B Segmentation-Aware Training

`scripts/51_train_segaware_hc18.py` exposes `--pretrained`, not `--init_from`. Its built-in loss stack includes Charbonnier, LPIPS, DISTS, edge, GAN, segmentation-aware loss, segmentation warmup, and hard-example weighting. A representative command is:

```powershell
python .\scripts\51_train_segaware_hc18.py `
    --backbone swinir `
    --pretrained results/models/swinir_light_phase_a_seed42/best_lpips.pth `
    --epochs 50 --batch_size 8 --seed 42 `
    --seg_warmup_epochs 10 `
    --eval_dice_every 5 `
    --run_name phaseB_segaware
```

Do not use Phase B numbers in the canonical table unless the run has completed and has been evaluated through scripts 42, 46, 47, and 48.

### Full-Test Reconstruction Evaluation

`scripts/42_compare_all_models.py` has no argparse CLI; it executes `main()` directly. To add a new model, edit only the `MODELS` registry in the script.

```powershell
python .\scripts\42_compare_all_models.py
```

Primary outputs:

```text
results/final_all_hc18_comparison_per_image.csv
results/final_all_hc18_comparison_summary.csv
results/final_all_hc18_comparison_table.md
results/eval/swinir_light_phase_a/test/recon_per_image.csv
results/eval/swinir_light_phase_a/test/recon_summary.csv
```

### Downstream Frozen U-Net Dice

`scripts/46_downstream_utility_eval.py` also has no argparse CLI; it executes `main()` directly. To add a model, edit only its `METHODS` list.

```powershell
python .\scripts\46_downstream_utility_eval.py
```

Primary outputs:

```text
results/downstream_utility_per_image.csv
results/downstream_utility_summary.csv
results/downstream_utility_masks/
results/eval/swinir_light_phase_a/test/downstream_per_image.csv
results/eval/swinir_light_phase_a/test/downstream_summary.csv
```

### Hard-Subset Analysis

```powershell
python .\scripts\47_hard_subset_analysis.py
```

Primary outputs:

```text
results/downstream_hard_subset_membership.csv
results/downstream_hard_subset_summary.csv
results/downstream_hard_subset_figures/dice_hard_vs_full.png
results/eval/swinir_light_phase_a/test/hard_subsets.csv
```

Hard subsets:

| Subset | Definition |
|---|---|
| `bottom25_lq_dice` | Worst quartile by LQ Dice. |
| `lq_dice_lt_0_85` | Images where LQ Dice is below 0.85. |
| `top25_lq_lpips` | Worst quartile by LQ LPIPS. This corresponds to bottom-25% quality by perceptual reconstruction. |

### Significance Report

Use CPU if CUDA import or memory pressure is unstable:

```powershell
python .\scripts\48_significance_report.py --device cpu
```

Primary outputs:

```text
results/bootstrap_metric_summary.csv
results/significance_report.csv
```

## 7. Trained Models

### Canonical Model Registry and Headline Metrics

Validation values are listed when they are present in audited training logs or canonical summaries. Test values come from the canonical full-test CSVs.

| Canonical name | Training script | Checkpoint | Val LPIPS | Val Dice | Test LPIPS | Test Dice |
|---|---|---|---:|---:|---:|---:|
| LQ_mixed baseline | `scripts/02_generate_lq_hc18.py` | None | n/a | n/a | 0.350968 | 0.884990 |
| SR-only | `scripts/05_train_sr_only_hc18.py` | `results/models/sr_only_hc18_best.pth` | n/a | n/a | 0.174942 | 0.885327 |
| Quality-aware | `scripts/07_train_quality_aware_sr_hc18.py` | `results/models/quality_aware_sr_hc18_l1_mse_best.pth` | n/a | n/a | 0.204216 | 0.884688 |
| SR-only+perceptual | `scripts/40_train_sr_perceptual_hc18.py` | `results/models/sr_perceptual_hc18_final.pth` | n/a | n/a | 0.123734 | 0.887286 |
| Quality-aware+perceptual | `scripts/40_train_sr_perceptual_hc18.py` | `results/models/qa_sr_perceptual_hc18_final.pth` | n/a | n/a | 0.113073 | 0.885817 |
| CNN-Residual Diffusion | `scripts/20_train_cnn_residual_diffusion_sr_hc18.py` | `results/models/cnn_residual_diffusion_sr_hc18_best.pth` | n/a | n/a | not in canonical LPIPS/DISTS table | 0.885430 |
| SwinIR-Light Phase A | `scripts/50_train_swinir_hc18.py` | `results/models/swinir_light_phase_a_seed42/best_lpips.pth` | 0.102597 | 0.875081 | 0.100915 | 0.887702 |
| Phase B SwinIR seg-aware | `scripts/51_train_segaware_hc18.py` | `results/models/phaseB_segaware_swinir_seed42/best_lpips.pth` | incomplete run | incomplete run | not evaluated | not evaluated |

### Checkpoint Inventory

All trained checkpoint files under `results/models/` at audit time:

| Checkpoint | Size MB | Last modified |
|---|---:|---|
| `results/models/cnn_residual_diffusion_sr_hc18_best.pth` | 39.03 | 2026-06-22 23:04:33 |
| `results/models/diffusion_sr_hc18_best.pth` | 39.02 | 2026-06-22 02:10:35 |
| `results/models/diffusion_sr_x0loss_hc18_best.pth` | 39.02 | 2026-06-22 18:09:09 |
| `results/models/dmuisr_inspired_diffusion_sr_hc18_best.pth` | 43.18 | 2026-06-22 17:31:50 |
| `results/models/hc18_unet_segmentation_best_improved.pth` | 29.68 | 2026-06-27 20:47:57 |
| `results/models/phaseB_segaware_swinir_seed42/best_dice.pth` | 22.56 | 2026-07-07 08:12:07 |
| `results/models/phaseB_segaware_swinir_seed42/best_lpips.pth` | 22.57 | 2026-07-07 08:12:07 |
| `results/models/phaseB_segaware_swinir_seed42/ema.pth` | 4.41 | 2026-07-07 13:25:53 |
| `results/models/phaseB_segaware_swinir_seed42/last.pth` | 22.52 | 2026-07-07 13:25:53 |
| `results/models/proxy_quality_embedding_mlp.pth` | 0.02 | 2026-06-24 01:51:22 |
| `results/models/qa_sr_perceptual_hc18_best.pth` | 0.45 | 2026-07-06 16:21:39 |
| `results/models/qa_sr_perceptual_hc18_final.pth` | 0.45 | 2026-07-06 16:21:39 |
| `results/models/quality_aware_diffusion_sr_hc18_best.pth` | 39.80 | 2026-06-22 02:45:10 |
| `results/models/quality_aware_diffusion_sr_hc18_v2_best.pth` | 39.80 | 2026-06-22 03:07:11 |
| `results/models/quality_aware_sr_hc18_best.pth` | 0.45 | 2026-06-21 02:43:45 |
| `results/models/quality_aware_sr_hc18_l1_mse_best.pth` | 0.45 | 2026-06-21 03:10:17 |
| `results/models/short_sr_l1_mse_hc18_best.pth` | 0.43 | 2026-07-06 15:51:40 |
| `results/models/short_sr_l1_mse_hc18_final.pth` | 0.43 | 2026-07-06 15:51:40 |
| `results/models/short_sr_lpips_gan_hc18_best.pth` | 0.43 | 2026-07-06 15:58:45 |
| `results/models/short_sr_lpips_gan_hc18_final.pth` | 0.43 | 2026-07-06 15:58:45 |
| `results/models/sr_only_hc18_best.pth` | 0.43 | 2026-07-06 15:24:57 |
| `results/models/sr_perceptual_hc18_best.pth` | 0.43 | 2026-07-06 16:09:28 |
| `results/models/sr_perceptual_hc18_final.pth` | 0.43 | 2026-07-06 16:09:28 |
| `results/models/swinir_light_phase_a_seed42/best_dice.pth` | 22.57 | 2026-07-07 03:17:30 |
| `results/models/swinir_light_phase_a_seed42/best_lpips.pth` | 22.57 | 2026-07-07 03:47:50 |
| `results/models/swinir_light_phase_a_seed42/ema.pth` | 4.41 | 2026-07-07 03:47:51 |
| `results/models/swinir_light_phase_a_seed42/last.pth` | 22.54 | 2026-07-07 03:47:51 |

## 8. Evaluation and Results

### Full-Test Reconstruction Summary

Canonical full-test results from `results/final_all_hc18_comparison_summary.csv`:

| Method | PSNR | SSIM | LPIPS | DISTS | MAE | EdgeScore | NIQE | BRISQUE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LQ_mixed baseline | 25.384944 | 0.672256 | 0.350968 | 0.241334 | 0.033746 | 0.977929 | 9.808565 | 69.168066 |
| SR-only | 30.985649 | 0.856411 | 0.174942 | 0.174088 | 0.017652 | 0.985700 | 7.452263 | 54.514774 |
| Quality-aware SR | 30.580420 | 0.874337 | 0.204216 | 0.181186 | 0.017394 | 0.984976 | 7.617840 | 57.515723 |
| SR-only + LPIPS + GAN | 27.863696 | 0.781360 | 0.123734 | 0.177299 | 0.025117 | 0.981262 | 6.898320 | 29.086917 |
| Quality-aware SR + LPIPS + GAN | 28.743527 | 0.826617 | 0.113073 | 0.167108 | 0.021958 | 0.982667 | 6.160169 | 27.430340 |
| SwinIR-Light Phase A | 29.727706 | 0.846230 | 0.100915 | 0.150868 | 0.019428 | 0.984159 | 6.072501 | 39.517253 |

Interpretation:

- SwinIR-Light Phase A is the best registered method on LPIPS and DISTS.
- Classical SR-only CNN has the highest PSNR and edge score, reflecting pixel fidelity rather than perceptual realism.
- Quality-aware SR has the highest SSIM among the canonical reconstruction rows.
- Perceptual/GAN losses improve LPIPS strongly but can reduce PSNR and SSIM.

### Full-Test Downstream Utility

Canonical downstream Dice results from `results/downstream_utility_summary.csv`:

| Method | Dice mean | Dice std | IoU mean | AreaErr mean | DiamErr mean |
|---|---:|---:|---:|---:|---:|
| HR | 0.889657 | 0.141001 | 0.822324 | 0.346928 | 7.884109 |
| LQ_mixed | 0.884990 | 0.141432 | 0.816676 | 0.341064 | 7.861875 |
| SR-only | 0.885327 | 0.142393 | 0.817207 | 0.355645 | 8.367206 |
| SR-only+perceptual | 0.887286 | 0.140725 | 0.819637 | 0.339348 | 8.007283 |
| Quality-aware | 0.884688 | 0.142371 | 0.816433 | 0.353884 | 8.410059 |
| Quality-aware+perceptual | 0.885817 | 0.141280 | 0.817821 | 0.337467 | 7.863954 |
| SwinIR-Light Phase A | 0.887702 | 0.141104 | 0.820139 | 0.341183 | 7.988348 |
| CNN-Residual Diffusion | 0.885430 | 0.141298 | 0.817341 | 0.341887 | 7.923502 |

Interpretation:

- The frozen U-Net has a high HR ceiling but low spread between LQ and SR variants.
- SwinIR-Light Phase A gives the best downstream Dice among registered SR outputs, but the absolute gain over LQ is small.
- Dice values are reported on 148 mask-valid test images, while reconstruction metrics use all 150 test images.

### Hard-Subset Summary for SwinIR-Light Phase A

From `results/downstream_hard_subset_summary.csv` and `results/eval/swinir_light_phase_a/test/hard_subsets.csv`:

| Subset | N | SwinIR Dice | Delta vs LQ | p_bonf |
|---|---:|---:|---:|---:|
| full_test | 148 | 0.887702 | +0.002712 | not a hard-subset test |
| bottom25_lq_dice | 37 | 0.708460 | +0.003661 | 1.0 |
| lq_dice_lt_0_85 | 40 | 0.720790 | +0.005496 | 1.0 |
| top25_lq_lpips | 37 | 0.867882 | +0.004413 | 1.0 |

Interpretation:

- SwinIR improves hard-subset Dice slightly over LQ, but these hard-subset deltas are not significant after Bonferroni correction.
- The hard bottom-25 LQ-Dice subset is the most clinically stressful downstream subset and remains difficult for all methods.

### Significance Highlights

From `results/significance_report.csv`:

| Comparison | Metric | Subset | Delta | p_bonf | Interpretation |
|---|---|---|---:|---:|---|
| SR-only+perceptual -> SwinIR-Light Phase A | LPIPS | full_test | -0.023025 | 3.63e-19 | SwinIR significantly lowers LPIPS vs SR+perceptual. |
| Quality-aware+perceptual -> SwinIR-Light Phase A | LPIPS | full_test | -0.012186 | 3.99e-10 | SwinIR significantly lowers LPIPS vs QA+perceptual. |
| SR-only+perceptual -> SwinIR-Light Phase A | Dice | full_test | +0.000416 | 0.545150 | Dice difference is not significant. |
| LQ_mixed -> SwinIR-Light Phase A | Dice | bottom25_lq_dice | +0.003661 | 1.0 | Hard-subset Dice gain is not significant. |
| LQ_mixed -> SwinIR-Light Phase A | Dice | lq_dice_lt_0_85 | +0.005496 | 1.0 | Hard-subset Dice gain is not significant. |
| LQ_mixed -> SwinIR-Light Phase A | Dice | top25_lq_lpips | +0.004413 | 1.0 | Hard-subset Dice gain is not significant. |

### SwinIR Full-Test Summary Block

```text
=== SwinIR full-test summary ===
LPIPS test:                    0.100915 [0.096945, 0.105029]
DISTS test:                    0.150868 [0.147990, 0.154013]
PSNR test:                     29.727706 +/- 1.349419
SSIM test:                     0.846230 +/- 0.040064
Dice test (full):              0.887702 +/- 0.141104
Dice hard bottom-25 LQ-Dice:   +0.003661 vs LQ, p_bonf 1.0
Dice hard LQD<0.85:            +0.005496 vs LQ, p_bonf 1.0
Dice hard bottom-25 LQ-LPIPS:  +0.004413 vs LQ, p_bonf 1.0
LPIPS SwinIR vs QA+perc:       -0.012186, p_bonf 3.99e-10
LPIPS SwinIR vs SR+perc:       -0.023025, p_bonf 3.63e-19
Dice  SwinIR vs SR+perc:       +0.000416, p_bonf 0.545150
```

### Figure and Visual Asset Inventory

Curated figure/report folders:

| Folder | Count | Contents |
|---|---:|---|
| `results/bootstrap_ci_figures/` | 3 | `acouslic_ai_bootstrap_ci.png`, `hc18_anatomy_bootstrap_ci.png`, `hc18_reconstruction_bootstrap_ci.png` |
| `results/downstream_hard_subset_figures/` | 1 | `dice_hard_vs_full.png` |
| `results/final_ablation_figures/` | 4 | `acquisition_frame_improvement.png`, `acquisition_sweep_feedback_changes.png`, `hc18_anatomy_preservation_ablation.png`, `hc18_psnr_ssim_ablation.png` |
| `results/hc18_unet_segmentation_figures/` | 6 | `debug_image_mask_pairing.png`, `improved_dice_iou_comparison.png`, `improved_example_segmentation_outputs.png`, `improved_measurement_error_comparison.png`, `improved_training_curve.png`, `overfit_sanity_check.png` |
| `results/report_assets/` | 5 PNG figures plus CSV tables | `figure_anatomy_mae_inside_mask.png`, `figure_boundary_error.png`, `figure_mae_comparison.png`, `figure_psnr_comparison.png`, `figure_ssim_comparison.png` |
| `results/visual_check/` | 5 | HC18 degradation sanity-check samples |
| `results/final_all_visual_comparison/` | 10 | Final all-model visual comparison panels |
| `results/final_visual_comparison/` | 10 | Earlier visual comparison panels |

The repository also contains thousands of generated PNG outputs under ACOUSLIC enhancement folders and downstream predicted-mask folders. Those are generated evidence artifacts, not curated publication figures.

## 9. Statistical Methodology

The canonical significance workflow is implemented in `scripts/48_significance_report.py`.

| Methodological component | Implementation |
|---|---|
| Unit of analysis | Paired image-level metrics on the HC18 test split. |
| Reconstruction metrics | PSNR, SSIM, LPIPS, DISTS; LPIPS and DISTS are lower-is-better. |
| Downstream metrics | Dice, IoU, area error, equivalent-diameter error from a frozen improved HC18 U-Net. |
| Confidence intervals | Bootstrap mean confidence intervals, default seed 42. |
| Paired tests | Wilcoxon signed-rank tests on paired image-level differences. |
| Multiple comparisons | Bonferroni correction across the registered comparison set, including SwinIR-vs-CNN comparisons. |
| Hard subsets | Defined once from LQ difficulty and reused for all methods. |

The statistical interpretation is deliberately conservative: SwinIR's LPIPS improvements over perceptual CNN champions are significant after Bonferroni correction, while downstream Dice gains are small and not significant after correction.

## 10. Repository Conventions

### Git and Artifact Policy

The `.gitignore` is configured to keep source, report tables, and curated result summaries while excluding large or reproducible artifacts:

| Ignored category | Examples |
|---|---|
| Raw datasets | `data/raw/`, archives, `.mha`, `.nii`, `.npz` |
| Processed image datasets | `data/processed/HC18_SR/`, processed ACOUSLIC frames/temp folders |
| Model checkpoints | `results/models/`, `*.pth`, `*.pt`, `*.ckpt` |
| Large generated outputs | frame dumps, predicted masks, visual-output folders |
| Local caches | Python caches and temporary files |

### Naming Conventions

| Name | Meaning |
|---|---|
| `LQ_mixed` | Canonical degraded HC18 input used for final SR evaluation. |
| `SR-only` | Residual CNN trained with pixel reconstruction losses. |
| `Quality-aware` | Quality-conditioned residual CNN. |
| `SR-only+perceptual` | SR-only CNN retrained with LPIPS and GAN losses. |
| `Quality-aware+perceptual` | Quality-aware CNN retrained with LPIPS and GAN losses. |
| `CNN-Residual Diffusion` | Diffusion residual refinement around a CNN guide. |
| `SwinIR-Light Phase A` | Canonical SwinIR-Light perceptual/GAN hero. |

The string `SwinIR-Light Phase A` is the canonical SwinIR label across scripts 42, 46, 47, and 48. Changing the label in only one script can silently break CSV joins.

## 11. Known Limitations

1. **Dataset scope:** The headline SR result is HC18-only. External generalization is not established.
2. **Downstream ceiling:** The frozen U-Net produces HR Dice around 0.8897 and LQ Dice around 0.8850, leaving little room to demonstrate downstream gains.
3. **Phase B status:** Segmentation-aware Phase B checkpoints exist, but the audited run is incomplete for canonical reporting and has not been fully evaluated on the test split.
4. **Clinical validation:** Metrics are image-processing and segmentation-utility metrics, not direct clinical outcome validation.
5. **ACOUSLIC proxy labels:** ACOUSLIC-AI quality labels in this repo are heuristic proxy labels and manual-review aids, not expert clinical annotations.
6. **OpenCV dependency:** Several scripts use `cv2`, but OpenCV is not listed in `requirements.txt`; it must exist in the project environment.
7. **Generated artifacts:** Many results are reproducible but large, and checkpoints/raw data are intentionally excluded from git.
8. **Diffusion comparison gap:** Earlier diffusion scripts report PSNR/SSIM/MAE/EdgeScore, but not every diffusion variant is registered in the final LPIPS/DISTS table.

## 12. Citation

Use the following citation skeletons and replace titles/URLs with the exact bibliographic source required by the venue.

```bibtex
@misc{hc18_challenge,
  title        = {HC18 Grand Challenge: Fetal Head Circumference Measurement},
  howpublished = {\url{https://hc18.grand-challenge.org/}},
  note         = {Accessed for the Fetal_SR_Project data-processing pipeline}
}

@misc{acouslic_ai_challenge,
  title        = {ACOUSLIC-AI Grand Challenge},
  howpublished = {\url{https://acouslic-ai.grand-challenge.org/}},
  note         = {Used for the blind-sweep ultrasound quality-assessment branch}
}

@inproceedings{wang2021swinir,
  title     = {SwinIR: Image Restoration Using Swin Transformer},
  author    = {Wang, Jingyun and Cao, Kelvin C. K. and others},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops},
  year      = {2021}
}

@article{zhang2018lpips,
  title   = {The Unreasonable Effectiveness of Deep Features as a Perceptual Metric},
  author  = {Zhang, Richard and Isola, Phillip and Efros, Alexei A. and Shechtman, Eli and Wang, Oliver},
  journal = {CVPR},
  year    = {2018}
}

@article{ding2020dists,
  title   = {Image Quality Assessment: Unifying Structure and Texture Similarity},
  author  = {Ding, Keyan and Ma, Kede and Wang, Shiqi and Simoncelli, Eero P.},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2020}
}
```

## 13. Git History

Last commits from `git log --oneline -30`:

| Commit | Message |
|---|---|
| `3d0f9fa` | Register SwinIR-Light in comparison/downstream/significance pipeline |
| `14bf4e7` | checkpoint: perceptual+GAN done, downstream eval done, SwinIR partial (epoch 32) |
| `0e95e59` | Initial conference project commit |

The most recent commit, `3d0f9fa`, is the audit point for the canonical SwinIR registration across scripts 42, 46, 47, and 48.

## Appendix A. Result CSV Inventory

Every CSV under `results/` at audit time, with row count and headers only:

| CSV | Rows | Headers |
|---|---:|---|
| `results/acouslic_ai/acouslic_zip_inspection.csv` | 601 | `file_name,extension,compressed_size,uncompressed_size` |
| `results/acouslic_ai/adaptive_feedback_enhancement_per_frame.csv` | 10000 | `sweep_id,frame_id,frame_index,frame_path,original_frame_quality_label,adaptive_frame_quality_label,action_taken,original_blur_score,adaptive_blur_score,delta_blur_score,original_contrast_score,adaptive_contrast_score,delta_contrast_score,original_edge_density,adaptive_edge_density,delta_edge_density,original_noise_proxy,adaptive_noise_proxy,delta_noise_proxy,improved_label_flag` |
| `results/acouslic_ai/adaptive_feedback_enhancement_summary.csv` | 11 | `scope,original_frame_quality_label,action_taken,total_frames,number_of_frames_improved,percentage_improved,mean_delta_blur,mean_delta_contrast,mean_delta_edge_density,mean_delta_noise_proxy` |
| `results/acouslic_ai/adaptive_sweep_feedback_change_per_sweep.csv` | 100 | `sweep_id,total_frames,original_good_frame_count,adaptive_good_frame_count,original_poor_like_count,adaptive_poor_like_count,original_poor_frame_ratio,adaptive_poor_frame_ratio,poor_frame_ratio_reduction,original_good_frame_ratio,adaptive_good_frame_ratio,good_frame_ratio_improvement,original_feedback_label,adaptive_feedback_label,feedback_changed_flag,mean_optical_flow,mean_jitter_score,motion_summary_missing_flag,adaptive_low_contrast_frame_ratio,unchanged_motion_based_decision_flag,feedback_change_reason` |
| `results/acouslic_ai/adaptive_sweep_feedback_change_summary.csv` | 1 | `number_of_sweeps_analyzed,number_of_sweeps_with_reduced_poor_frame_ratio,mean_poor_frame_ratio_reduction,number_of_sweeps_with_improved_good_frame_ratio,number_of_repeat_sweep_cases_changed_to_acceptable_or_improve_visibility,number_of_feedback_decisions_unchanged_due_to_motion_based_problems` |
| `results/acouslic_ai/feedback_aware_enhancement_per_frame.csv` | 5000 | `sweep_id,frame_id,frame_index,frame_path,original_frame_quality_label,enhanced_frame_quality_label,original_blur_score,enhanced_blur_score,delta_blur_score,original_contrast_score,enhanced_contrast_score,delta_contrast_score,original_edge_density,enhanced_edge_density,delta_edge_density,original_noise_proxy,enhanced_noise_proxy,delta_noise_proxy,improved_label_flag` |
| `results/acouslic_ai/feedback_aware_enhancement_summary.csv` | 6 | `scope,original_frame_quality_label,total_frames,number_of_frames_improved,percentage_improved,mean_delta_blur,mean_delta_contrast,mean_delta_edge_density,mean_delta_noise_proxy` |
| `results/acouslic_ai/learned_adaptive_enhancement_per_frame.csv` | 10000 | `sweep_id,frame_id,frame_index,frame_path,original_proxy_label,learned_predicted_label,learned_confidence,action_taken,before_threshold_label,after_threshold_label,original_blur_score,after_blur_score,delta_blur_score,original_contrast_score,after_contrast_score,delta_contrast_score,original_edge_density,after_edge_density,delta_edge_density,original_noise_proxy,after_noise_proxy,delta_noise_proxy,improved_label_flag` |
| `results/acouslic_ai/learned_adaptive_enhancement_summary.csv` | 13 | `scope,original_proxy_label,action_taken,total_frames,number_of_frames_improved,percentage_improved,mean_delta_blur,mean_delta_contrast,mean_delta_edge_density,mean_delta_noise_proxy` |
| `results/acouslic_ai/manual_review/proxy_label_manual_review.csv` | 125 | `sample_id,sweep_id,frame_id,frame_path,proxy_label,manual_label_blank,reviewer_comment_blank` |
| `results/acouslic_ai/proxy_quality_embedding_metrics.csv` | 3 | `split,num_frames,accuracy,macro_f1,f1_good_frame,f1_blurry_frame,f1_low_contrast_frame,f1_noisy_frame,f1_poor_frame` |
| `results/acouslic_ai/proxy_quality_embedding_predictions.csv` | 10000 | `sweep_id,frame_id,true_proxy_label,predicted_proxy_label,confidence,embedding_0..embedding_31` |
| `results/acouslic_ai/sweep_feedback_change_per_sweep.csv` | 50 | `sweep_id,total_frames,original_good_frame_count,enhanced_good_frame_count,original_poor_like_count,enhanced_poor_like_count,original_poor_frame_ratio,enhanced_poor_frame_ratio,poor_frame_ratio_reduction,original_good_frame_ratio,enhanced_good_frame_ratio,good_frame_ratio_improvement,original_feedback_label,enhanced_feedback_label,feedback_changed_flag,mean_optical_flow,mean_jitter_score,motion_summary_missing_flag,enhanced_low_contrast_frame_ratio,unchanged_motion_based_decision_flag,feedback_change_reason` |
| `results/acouslic_ai/sweep_feedback_change_summary.csv` | 1 | `number_of_sweeps_analyzed,number_of_sweeps_with_reduced_poor_frame_ratio,mean_poor_frame_ratio_reduction,number_of_sweeps_with_improved_good_frame_ratio,number_of_repeat_sweep_cases_changed_to_acceptable_or_improve_visibility,number_of_feedback_decisions_unchanged_due_to_motion_based_problems` |
| `results/anatomy_preservation_all_models_per_image.csv` | 1050 | `image_id,method,mae_inside_mask,mae_outside_mask,boundary_error,contrast_preservation,edge_similarity_boundary` |
| `results/anatomy_preservation_all_models_summary.csv` | 7 | `method,mean_mae_inside_mask,std_mae_inside_mask,mean_mae_outside_mask,std_mae_outside_mask,mean_boundary_error,std_boundary_error,mean_contrast_preservation,std_contrast_preservation,mean_edge_similarity_boundary,std_edge_similarity_boundary` |
| `results/anatomy_preservation_per_image.csv` | 600 | `image_id,method,mae_inside_mask,mae_outside_mask,boundary_error,contrast_preservation,edge_similarity_boundary` |
| `results/anatomy_preservation_summary.csv` | 4 | `method,mean_mae_inside_mask,std_mae_inside_mask,mean_mae_outside_mask,std_mae_outside_mask,mean_boundary_error,std_boundary_error,mean_contrast_preservation,std_contrast_preservation,mean_edge_similarity_boundary,std_edge_similarity_boundary` |
| `results/archived_unreliable_dice_iou/results/hc18_dice_iou_measurement_metrics.csv` | 1050 | `image_id,method,output_path,mask_path,segmentation_method,dice,iou,boundary_error_px,predicted_area_px,ground_truth_area_px,fetal_head_area_error_percent,predicted_equivalent_diameter_px,ground_truth_equivalent_diameter_px,equivalent_diameter_error_px,equivalent_diameter_error_percent` |
| `results/archived_unreliable_dice_iou/results/hc18_dice_iou_measurement_summary.csv` | 7 | `method,n_images,mean_dice,std_dice,mean_iou,std_iou,mean_boundary_error_px,std_boundary_error_px,mean_fetal_head_area_error_percent,std_fetal_head_area_error_percent,mean_equivalent_diameter_error_px,std_equivalent_diameter_error_px,mean_equivalent_diameter_error_percent,std_equivalent_diameter_error_percent` |
| `results/archived_unreliable_dice_iou/results/hc18_unet_segmentation_metrics_per_image.csv` | 1200 | `image_id,method,input_path,mask_path,dice,iou,boundary_error_px,predicted_area_px,ground_truth_area_px,fetal_head_area_error_percent,predicted_equivalent_diameter_px,ground_truth_equivalent_diameter_px,equivalent_diameter_error_px,equivalent_diameter_error_percent` |
| `results/archived_unreliable_dice_iou/results/hc18_unet_segmentation_metrics_summary.csv` | 8 | `method,n_images,mean_dice,std_dice,mean_iou,std_iou,mean_boundary_error_px,std_boundary_error_px,mean_fetal_head_area_error_percent,std_fetal_head_area_error_percent,mean_equivalent_diameter_error_px,std_equivalent_diameter_error_px,mean_equivalent_diameter_error_percent,std_equivalent_diameter_error_percent` |
| `results/backups/acouslic_ai_50sweeps/acouslic_zip_inspection.csv` | 601 | `file_name,extension,compressed_size,uncompressed_size` |
| `results/backups/acouslic_ai_50sweeps/adaptive_feedback_enhancement_per_frame.csv` | 5000 | `sweep_id,frame_id,frame_index,frame_path,original_frame_quality_label,adaptive_frame_quality_label,action_taken,original_blur_score,adaptive_blur_score,delta_blur_score,original_contrast_score,adaptive_contrast_score,delta_contrast_score,original_edge_density,adaptive_edge_density,delta_edge_density,original_noise_proxy,adaptive_noise_proxy,delta_noise_proxy,improved_label_flag` |
| `results/backups/acouslic_ai_50sweeps/adaptive_feedback_enhancement_summary.csv` | 11 | `scope,original_frame_quality_label,action_taken,total_frames,number_of_frames_improved,percentage_improved,mean_delta_blur,mean_delta_contrast,mean_delta_edge_density,mean_delta_noise_proxy` |
| `results/backups/acouslic_ai_50sweeps/adaptive_sweep_feedback_change_per_sweep.csv` | 50 | `sweep_id,total_frames,original_good_frame_count,adaptive_good_frame_count,original_poor_like_count,adaptive_poor_like_count,original_poor_frame_ratio,adaptive_poor_frame_ratio,poor_frame_ratio_reduction,original_good_frame_ratio,adaptive_good_frame_ratio,good_frame_ratio_improvement,original_feedback_label,adaptive_feedback_label,feedback_changed_flag,mean_optical_flow,mean_jitter_score,motion_summary_missing_flag,adaptive_low_contrast_frame_ratio,unchanged_motion_based_decision_flag,feedback_change_reason` |
| `results/backups/acouslic_ai_50sweeps/adaptive_sweep_feedback_change_summary.csv` | 1 | `number_of_sweeps_analyzed,number_of_sweeps_with_reduced_poor_frame_ratio,mean_poor_frame_ratio_reduction,number_of_sweeps_with_improved_good_frame_ratio,number_of_repeat_sweep_cases_changed_to_acceptable_or_improve_visibility,number_of_feedback_decisions_unchanged_due_to_motion_based_problems` |
| `results/backups/acouslic_ai_50sweeps/feedback_aware_enhancement_per_frame.csv` | 5000 | `sweep_id,frame_id,frame_index,frame_path,original_frame_quality_label,enhanced_frame_quality_label,original_blur_score,enhanced_blur_score,delta_blur_score,original_contrast_score,enhanced_contrast_score,delta_contrast_score,original_edge_density,enhanced_edge_density,delta_edge_density,original_noise_proxy,enhanced_noise_proxy,delta_noise_proxy,improved_label_flag` |
| `results/backups/acouslic_ai_50sweeps/feedback_aware_enhancement_summary.csv` | 6 | `scope,original_frame_quality_label,total_frames,number_of_frames_improved,percentage_improved,mean_delta_blur,mean_delta_contrast,mean_delta_edge_density,mean_delta_noise_proxy` |
| `results/backups/acouslic_ai_50sweeps/sweep_feedback_change_per_sweep.csv` | 50 | `sweep_id,total_frames,original_good_frame_count,enhanced_good_frame_count,original_poor_like_count,enhanced_poor_like_count,original_poor_frame_ratio,enhanced_poor_frame_ratio,poor_frame_ratio_reduction,original_good_frame_ratio,enhanced_good_frame_ratio,good_frame_ratio_improvement,original_feedback_label,enhanced_feedback_label,feedback_changed_flag,mean_optical_flow,mean_jitter_score,motion_summary_missing_flag,enhanced_low_contrast_frame_ratio,unchanged_motion_based_decision_flag,feedback_change_reason` |
| `results/backups/acouslic_ai_50sweeps/sweep_feedback_change_summary.csv` | 1 | `number_of_sweeps_analyzed,number_of_sweeps_with_reduced_poor_frame_ratio,mean_poor_frame_ratio_reduction,number_of_sweeps_with_improved_good_frame_ratio,number_of_repeat_sweep_cases_changed_to_acceptable_or_improve_visibility,number_of_feedback_decisions_unchanged_due_to_motion_based_problems` |
| `results/bicubic_hc18_per_image.csv` | 150 | `image_id,split,bicubic_x2_psnr,bicubic_x2_ssim,bicubic_x2_mae,bicubic_x4_psnr,bicubic_x4_ssim,bicubic_x4_mae,lq_mixed_baseline_psnr,lq_mixed_baseline_ssim,lq_mixed_baseline_mae` |
| `results/bicubic_hc18_summary.csv` | 3 | `method,mean_psnr,std_psnr,mean_ssim,std_ssim,mean_mae,std_mae` |
| `results/bootstrap_confidence_intervals.csv` | 59 | `source,method,metric,n,mean,ci_lower,ci_upper,ci_level,n_bootstrap,unit,note` |
| `results/bootstrap_metric_summary.csv` | 160 | `subset,method,metric,mean,ci_low,ci_high,n` |
| `results/cnn_residual_diffusion_sr_hc18_per_image.csv` | 150 | `image_id,psnr,ssim,mae,edge_score,inference_time` |
| `results/cnn_residual_diffusion_sr_hc18_summary.csv` | 5 | `metric,mean,std` |
| `results/diffusion_sr_hc18_per_image.csv` | 150 | `image_id,psnr,ssim,mae,edge_score,inference_time` |
| `results/diffusion_sr_hc18_summary.csv` | 5 | `metric,mean,std` |
| `results/diffusion_sr_x0loss_hc18_per_image.csv` | 10 | `image_id,psnr,ssim,mae,edge_score,inference_time` |
| `results/diffusion_sr_x0loss_hc18_summary.csv` | 5 | `metric,mean,std` |
| `results/dmuisr_inspired_diffusion_sr_hc18_per_image.csv` | 150 | `image_id,psnr,ssim,mae,edge_score,inference_time` |
| `results/dmuisr_inspired_diffusion_sr_hc18_summary.csv` | 5 | `metric,mean,std` |
| `results/downstream_hard_subset_membership.csv` | 148 | `image_id,LQ_Dice,LQ_LPIPS,bottom25_lq_dice,lq_dice_lt_0_85,top25_lq_lpips,dice_q25_threshold,lpips_q75_threshold` |
| `results/downstream_hard_subset_summary.csv` | 28 | `subset,method,N,Dice_mean,IoU_mean,AreaErr_mean,DiamErr_mean,Dice_delta_vs_LQ,IoU_delta_vs_LQ,AreaErr_delta_vs_LQ,DiamErr_delta_vs_LQ` |
| `results/downstream_utility_per_image.csv` | 1184 | `image_id,method,Dice,IoU,AreaErr,DiamErr,pred_area,GT_area,pred_equivalent_diameter,GT_equivalent_diameter,pred_mask_path` |
| `results/downstream_utility_summary.csv` | 8 | `method,Dice_mean,Dice_std,IoU_mean,AreaErr_mean,DiamErr_mean` |
| `results/eval/swinir_light_phase_a/test/downstream_per_image.csv` | 148 | `image_id,method,Dice,IoU,AreaErr,DiamErr,pred_area,GT_area,pred_equivalent_diameter,GT_equivalent_diameter,pred_mask_path` |
| `results/eval/swinir_light_phase_a/test/downstream_summary.csv` | 1 | `method,Dice_mean,Dice_std,IoU_mean,AreaErr_mean,DiamErr_mean` |
| `results/eval/swinir_light_phase_a/test/hard_subsets.csv` | 4 | `subset,method,N,Dice_mean,IoU_mean,AreaErr_mean,DiamErr_mean,Dice_delta_vs_LQ,IoU_delta_vs_LQ,AreaErr_delta_vs_LQ,DiamErr_delta_vs_LQ` |
| `results/eval/swinir_light_phase_a/test/recon_per_image.csv` | 150 | `image_id,method,PSNR,SSIM,LPIPS,DISTS,MAE,EdgeScore,NIQE,BRISQUE` |
| `results/eval/swinir_light_phase_a/test/recon_summary.csv` | 1 | `method,PSNR,SSIM,LPIPS,DISTS,MAE,EdgeScore,NIQE,BRISQUE` |
| `results/final_ablation_cnn_vs_residual_diffusion.csv` | 2 | `Model,Model Type,PSNR,SSIM,MAE,Edge Score,Main Interpretation` |
| `results/final_ablation_feedback_reassessment.csv` | 2 | `Configuration,Frame-Level Improvement,Feedback Reassessment,Sweeps with Reduced Poor-Frame Ratio,Sweeps with Improved Good-Frame Ratio,repeat_sweep Cases Changed to acceptable/improve_visibility,Main Interpretation` |
| `results/final_ablation_original_blind_adaptive.csv` | 4 | `Method,Frame-Level Improvement,Gain over Blind Enhancement,Main Interpretation` |
| `results/final_ablation_study.csv` | 10 | `Table,Method,Uses quality estimation,Uses adaptive action,Uses SR/CNN,Uses diffusion refinement,Frame improvement %,Sweeps with reduced poor-frame ratio,Repeat_sweep cases improved,Notes,Mean PSNR,Mean SSIM,Mean MAE,Mean edge score,Mean anatomy MAE inside mask,Mean boundary error,Mean contrast preservation,Mean boundary edge similarity` |
| `results/final_all_hc18_comparison_per_image.csv` | 900 | `image_id,method,PSNR,SSIM,LPIPS,DISTS,MAE,EdgeScore,NIQE,BRISQUE` |
| `results/final_all_hc18_comparison_summary.csv` | 6 | `method,PSNR,SSIM,LPIPS,DISTS,MAE,EdgeScore,NIQE,BRISQUE` |
| `results/final_all_hc18_dists_per_image.csv` | 750 | `image_id,method,DISTS` |
| `results/final_hc18_comparison_per_image.csv` | 600 | `image_id,method,psnr,ssim,mae,edge_score` |
| `results/final_hc18_comparison_summary.csv` | 4 | `method,mean_psnr,std_psnr,mean_ssim,std_ssim,mean_mae,std_mae,mean_edge_score,std_edge_score` |
| `results/hc18_unet_debug/debug_mask_statistics.csv` | 30 | `sample_id,original_name,image_path,mask_path,image_filename,mask_filename,image_shape,mask_shape,nonzero_mask_pixel_count,raw_mask_area_ratio,largest_contour_area_ratio,status,image_missing,mask_missing` |
| `results/hc18_unet_improved_mask_statistics.csv` | 999 | `sample_id,split,image_path,mask_path,mask_preprocessing_method,raw_mask_area_ratio,processed_mask_area_ratio,is_empty_mask,is_very_small_mask,is_suspicious_mask,image_missing,mask_missing` |
| `results/hc18_unet_improved_metrics_per_image.csv` | 1184 | `image_id,method,input_path,mask_path,dice,iou,boundary_error_px,predicted_area_px,ground_truth_area_px,fetal_head_area_error_percent,predicted_equivalent_diameter_px,ground_truth_equivalent_diameter_px,equivalent_diameter_error_px,equivalent_diameter_error_percent,reliability_note` |
| `results/hc18_unet_improved_metrics_summary.csv` | 8 | `method,n_images,mean_dice,std_dice,mean_iou,std_iou,mean_boundary_error_px,std_boundary_error_px,mean_fetal_head_area_error_percent,std_fetal_head_area_error_percent,mean_equivalent_diameter_error_px,std_equivalent_diameter_error_px,mean_equivalent_diameter_error_percent,std_equivalent_diameter_error_percent,reliability_note` |
| `results/phase_a_diagnosis/phase_a_diagnosis_summary.csv` | 1 | `last_epoch,last_val_lpips,slope_last10,r2_last10,cross_epoch,delta_epochs_to_gate,plateaued,verdict` |
| `results/phaseB_lq_dice_weights.csv` | 849 | `split,image_id,lq_dice,hard_weight` |
| `results/quality_aware_diffusion_sr_hc18_per_image.csv` | 150 | `image_id,psnr,ssim,mae,edge_score,inference_time` |
| `results/quality_aware_diffusion_sr_hc18_summary.csv` | 5 | `metric,mean,std` |
| `results/quality_aware_diffusion_sr_v2_hc18_per_image.csv` | 150 | `image_id,psnr,ssim,mae,edge_score,inference_time` |
| `results/quality_aware_diffusion_sr_v2_hc18_summary.csv` | 5 | `metric,mean,std` |
| `results/report_assets/anatomy_preservation_table.csv` | 4 | `method,mean_mae_inside_mask,std_mae_inside_mask,mean_mae_outside_mask,std_mae_outside_mask,mean_boundary_error,std_boundary_error,mean_contrast_preservation,std_contrast_preservation,mean_edge_similarity_boundary,std_edge_similarity_boundary` |
| `results/report_assets/final_sr_comparison_table.csv` | 4 | `method,mean_psnr,std_psnr,mean_ssim,std_ssim,mean_mae,std_mae,mean_edge_similarity,std_edge_similarity` |
| `results/short_sr_perceptual_comparison.csv` | 2 | `method,PSNR,SSIM,LPIPS,MAE,EdgeScore,NIQE,BRISQUE` |
| `results/short_sr_perceptual_comparison_per_image.csv` | 300 | `image_id,method,PSNR,SSIM,LPIPS,MAE,EdgeScore,NIQE,BRISQUE` |
| `results/significance_report.csv` | 39 | `method_A,method_B,metric,subset,delta,ci_low,ci_high,p_raw,p_bonf,significant_at_0.05,n_pairs` |
| `results/sr_only_hc18_per_image.csv` | 150 | `image_id,psnr,ssim,mae` |
| `results/sr_only_hc18_summary.csv` | 1 | `method,mean_psnr,std_psnr,mean_ssim,std_ssim,mean_mae,std_mae` |
| `results/training_logs/50_train_swinir_hc18_swinir_light_phase_a_seed42_20260706_165538.csv` | 12 | `epoch,lr,train_loss_g,train_loss_d,train_l1,train_lpips,train_gan,val_PSNR,val_SSIM,val_LPIPS,val_MAE,val_EdgeScore,val_Dice,val_IoU,batch_size,lpips_backend` |
| `results/training_logs/50_train_swinir_hc18_swinir_light_phase_a_seed42_20260706_185725.csv` | 20 | `epoch,lr,train_loss_g,train_loss_d,train_l1,train_lpips,train_gan,val_PSNR,val_SSIM,val_LPIPS,val_MAE,val_EdgeScore,val_Dice,val_IoU,batch_size,lpips_backend` |
| `results/training_logs/50_train_swinir_hc18_swinir_light_phase_a_seed42_20260707_010144.csv` | 16 | `epoch,lr,train_loss_g,train_loss_d,train_l1,train_lpips,train_gan,val_PSNR,val_SSIM,val_LPIPS,val_MAE,val_EdgeScore,val_Dice,val_IoU,batch_size,lpips_backend` |
| `results/training_logs/51_train_segaware_hc18_phaseB_segaware_swinir_seed42_20260707_041015.csv` | 8 | `epoch,lr,loss_g,loss_d,charbonnier,lpips,dists,edge,gan,seg,seg_weight,val_PSNR,val_SSIM,val_LPIPS,val_Dice,val_Dice_hard` |
| `results/training_logs/cnn_residual_diffusion_sr_hc18_log.csv` | 50 | `epoch,train_loss,val_loss` |
| `results/training_logs/diffusion_sr_hc18_log.csv` | 50 | `epoch,train_loss,val_loss` |
| `results/training_logs/diffusion_sr_x0loss_hc18_log.csv` | 100 | `epoch,train_loss,val_loss,train_noise_loss,train_x0_loss,val_noise_loss,val_x0_loss` |
| `results/training_logs/dmuisr_inspired_diffusion_sr_hc18_log.csv` | 100 | `epoch,train_loss,val_loss,train_noise_loss,train_x0_loss,train_edge_loss,val_noise_loss,val_x0_loss,val_edge_loss` |
| `results/training_logs/qa_sr_perceptual_20260706_161118.csv` | 40 | `epoch,train_loss,generator_loss,discriminator_loss,val_psnr,val_ssim,batch_size,l1_weight,mse_weight,lpips_weight,gan_weight,gan_enabled,perceptual_backend` |
| `results/training_logs/quality_aware_diffusion_sr_hc18_log.csv` | 50 | `epoch,train_loss,val_loss` |
| `results/training_logs/quality_aware_diffusion_sr_hc18_v2_log.csv` | 50 | `epoch,train_loss,val_loss` |
| `results/training_logs/quality_aware_sr_hc18_l1_mse_log.csv` | 30 | `epoch,train_loss,val_psnr,val_ssim` |
| `results/training_logs/quality_aware_sr_hc18_log.csv` | 30 | `epoch,train_loss,val_psnr,val_ssim` |
| `results/training_logs/short_sr_l1_mse_20260706_155008.csv` | 10 | `epoch,train_loss,generator_loss,discriminator_loss,val_psnr,val_ssim,batch_size,l1_weight,mse_weight,lpips_weight,gan_weight,gan_enabled,perceptual_backend` |
| `results/training_logs/short_sr_lpips_gan_20260706_155146.csv` | 10 | `epoch,train_loss,generator_loss,discriminator_loss,val_psnr,val_ssim,batch_size,l1_weight,mse_weight,lpips_weight,gan_weight,gan_enabled,perceptual_backend` |
| `results/training_logs/short_sr_lpips_gan_20260706_155657.csv` | 10 | `epoch,train_loss,generator_loss,discriminator_loss,val_psnr,val_ssim,batch_size,l1_weight,mse_weight,lpips_weight,gan_weight,gan_enabled,perceptual_backend` |
| `results/training_logs/sr_only_hc18_log.csv` | 10 | `epoch,train_loss,val_psnr,val_ssim` |
| `results/training_logs/sr_perceptual_20260706_015746.csv` | 1 | `epoch,train_loss,val_psnr,val_ssim` |
| `results/training_logs/sr_perceptual_20260706_151854.csv` | 10 | `epoch,train_loss,val_psnr,val_ssim` |
| `results/training_logs/sr_perceptual_20260706_160005.csv` | 40 | `epoch,train_loss,generator_loss,discriminator_loss,val_psnr,val_ssim,batch_size,l1_weight,mse_weight,lpips_weight,gan_weight,gan_enabled,perceptual_backend` |

