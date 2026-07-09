# THESIS_VIVA_DOSSIER.md

Project: QAFUE - Quality-Aware Fetal Ultrasound Enhancement  
Repository: `D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project`

## 1. Executive Summary

Evidence files read for this section: `README.md`, `README_FULL.md`, `results/final_all_hc18_comparison_summary.csv`, `results/bootstrap_metric_summary.csv`, `results/downstream_utility_summary.csv`, `results/significance_report.csv`, `results/acouslic_ai/proxy_quality_embedding_metrics.csv`, `results/acouslic_ai/adaptive_feedback_enhancement_summary.csv`, `results/acouslic_ai/learned_adaptive_enhancement_summary.csv`, `results/acouslic_ai/adaptive_sweep_feedback_change_summary.csv`, `results/cross_dataset/cross_dataset_stats.json`.

QAFUE addresses fetal ultrasound restoration under variable acquisition quality: blur, noise, low contrast, unstable sweep motion, and incomplete acquisition are explicitly named project motivations. `README.md`

The canonical current reconstruction method is SwinIR-Light Phase A, with checkpoint `results/models/swinir_light_phase_a_seed42/best_lpips.pth` and architecture builder `build_swinir_light`. `README_FULL.md`; `scripts/models/swinir_light.py`

On the HC18 reconstruction comparison table, SwinIR-Light Phase A has LPIPS `0.10091456212103367` and DISTS `0.15086822072664896`. `results/final_all_hc18_comparison_summary.csv`

The SwinIR-Light Phase A bootstrap LPIPS row reports mean `0.10084894095623005`, CI low `0.09694007307265858`, CI high `0.10476489356102266`, and `n` `148`. `results/bootstrap_metric_summary.csv`

Downstream segmentation utility is preservation-oriented: SwinIR-Light Phase A has Dice mean `0.8877019882187267`, while HR has Dice mean `0.8896568280296212` and LQ_mixed has Dice mean `0.8849903062929627`. `results/downstream_utility_summary.csv`

The ACOUSLIC proxy-quality branch trains a feature-based MLP with test accuracy `0.97666667` and macro-F1 `0.96882341`. `results/acouslic_ai/proxy_quality_embedding_metrics.csv`

Current adaptive feedback results are heuristic `17.94000000` percent frame improvement and learned `17.24000000` percent frame improvement on the 100-sweep branch. `results/acouslic_ai/adaptive_feedback_enhancement_summary.csv`; `results/acouslic_ai/learned_adaptive_enhancement_summary.csv`

Sweep reassessment reports `number_of_sweeps_analyzed` `100`, `number_of_sweeps_with_reduced_poor_frame_ratio` `100`, and `mean_poor_frame_ratio_reduction` `0.17940000`. `results/acouslic_ai/adaptive_sweep_feedback_change_summary.csv`

Cross-dataset transfer applies the ACOUSLIC-trained proxy MLP to HC18 zero-shot and reports Spearman rho `0.5688522618064861`, p-value `3.094730049128844e-14`, and verdict `STRONG TRANSFER`. `results/cross_dataset/cross_dataset_stats.json`

One-line limitation: this is an offline research prototype with proxy labels, simulated HC18 degradation, no expert clinical validation, no real-time scanner deployment, and incomplete Phase B segmentation-aware training. `README.md`; `README_FULL.md`; `results/training_logs/51_train_segaware_hc18_phaseB_segaware_swinir_seed42_20260707_041015.csv`

## 2. Motivation & Clinical Context

Evidence files read for this section: `README.md`, `README_FULL.md`, `reports/final_conference_report.md`, `reports/final_conference_report_appendix.md`, `reports/final_table_figure_checklist.md`, `data/processed/HC18_SR/hc18_manifest.csv`, `data/processed/HC18_SR/hc18_lq_manifest.csv`.

Fetal ultrasound quality varies because novice-acquired or low-resource scans can be blurry, noisy, low-contrast, unstable, incomplete, or otherwise poor quality. `README.md`

The repo grounds operator-skill context by saying experienced sonographers may not always be available in low-resource settings. `reports/final_conference_report.md`

Probe angle and acoustic shadowing are clinically plausible factors requested for this dossier, but they are UNVERIFIED by the repository text I read; a clinical background citation or dataset protocol file would be needed to ground them. UNVERIFIED: needed file = formal related-work bibliography or dataset/acquisition protocol.

HC18 is used for controlled paired reconstruction and fetal-head mask-based evaluation, with `999` rows in `hc18_manifest.csv` and `999` rows in `hc18_lq_manifest.csv`. `data/processed/HC18_SR/hc18_manifest.csv`; `data/processed/HC18_SR/hc18_lq_manifest.csv`

The project treats Dice, IoU, and fetal measurement errors as supporting anatomy-preservation evidence, not expert clinical validation. `reports/final_conference_report_appendix.md`

The key methodological motivation is that "pretty pictures" alone are not enough: QAFUE couples restoration with proxy acquisition-quality estimation, adaptive action selection, and downstream segmentation utility. `README.md`

The quality-aware framing is more honest because it checks whether restoration changes reconstruction metrics, whether U-Net segmentation remains near the HR ceiling, and whether ACOUSLIC-style proxy quality improves under a separate feature-based model. `results/final_all_hc18_comparison_summary.csv`; `results/downstream_utility_summary.csv`; `results/cross_dataset/cross_dataset_stats.json`

## 3. Related Work

Evidence files read for this section: `README.md`, `README_FULL.md`, `reports/final_conference_report.md`, `scripts/02_generate_lq_hc18.py`, `scripts/05_train_sr_only_hc18.py`, `scripts/07_train_quality_aware_sr_hc18.py`, `scripts/40_train_sr_perceptual_hc18.py`, `scripts/50_train_swinir_hc18.py`, `scripts/models/swinir_light.py`, `scripts/11_train_diffusion_sr_hc18.py`, `scripts/16_train_dmuisr_inspired_diffusion_sr_hc18.py`, `scripts/18_train_diffusion_sr_x0loss_hc18.py`, `scripts/20_train_cnn_residual_diffusion_sr_hc18.py`, `scripts/24_compute_acouslic_proxy_quality.py`.

Classical SR in this repo is represented concretely by bicubic downsample-upsample baselines: `downsample_and_upsample` creates LR_x2 and LR_x4 images with bicubic interpolation. `scripts/02_generate_lq_hc18.py`

Sparse-coding SR is requested as related work, but no sparse-coding source or bibliography entry was found in the repo files I read. UNVERIFIED: needed file = related-work bibliography containing sparse-coding SR citations.

CNN SR is represented by `ResidualSRModel` and `QualityAwareSRModel` training scripts, including SR-only and quality-aware CNN branches. `scripts/05_train_sr_only_hc18.py`; `scripts/07_train_quality_aware_sr_hc18.py`

SRCNN, EDSR, and RCAN are requested as related-work names, but I found no repo-grounded bibliography entries for those papers. UNVERIFIED: needed file = paper bibliography or literature-review draft.

Perceptual/GAN SR is represented by the residual CNN and quality-aware CNN perceptual branches, which combine reconstruction losses with LPIPS and optional PatchGAN-style adversarial training. `scripts/40_train_sr_perceptual_hc18.py`

LPIPS is grounded as both metric and loss by the citation skeleton for Zhang et al. and by the SwinIR Phase A training loss. `README_FULL.md`; `scripts/50_train_swinir_hc18.py`

SRGAN and ESRGAN are requested as related-work names, but I found no repo-grounded citation entries for them. UNVERIFIED: needed file = related-work bibliography containing SRGAN/ESRGAN citations.

Transformer SR is grounded by SwinIR: the repo includes a SwinIR citation skeleton and implements `SwinIRLight` with window attention, residual Swin Transformer blocks, and same-resolution residual output. `README_FULL.md`; `scripts/models/swinir_light.py`

Diffusion SR is represented by conditional diffusion, DMUISR-inspired diffusion, x0-loss diffusion, and CNN-residual diffusion scripts. `scripts/11_train_diffusion_sr_hc18.py`; `scripts/16_train_dmuisr_inspired_diffusion_sr_hc18.py`; `scripts/18_train_diffusion_sr_x0loss_hc18.py`; `scripts/20_train_cnn_residual_diffusion_sr_hc18.py`

SR3, one-step OSEDiff, DRAT, and diffusion transformer methods are not implemented in this repo. UNVERIFIED for implementation; allowed only as future work because searches for `OSEDiff`, `DRAT`, `Tier D`, and `diffusion transformer` returned no matching repo implementation.

Fetal-US quality assessment and acquisition guidance are represented in the ACOUSLIC branch through blur, contrast, brightness, noise, edge, frame-difference, optical-flow, and jitter proxies. `scripts/24_compute_acouslic_proxy_quality.py`

The position of QAFUE is quality-aware restoration plus downstream Dice utility plus zero-shot cross-dataset proxy transfer. `README.md`; `results/downstream_utility_summary.csv`; `docs/CROSS_DATASET_TRANSFER.md`

## 4. Datasets

Evidence files read for this section: `data/processed/HC18_SR/hc18_manifest.csv`, `data/processed/HC18_SR/hc18_lq_manifest.csv`, `data/processed/HC18_SR/train.csv`, `data/processed/HC18_SR/val.csv`, `data/processed/HC18_SR/test.csv`, `data/processed/ACOUSLIC_AI/acouslic_frame_manifest.csv`, `data/processed/ACOUSLIC_AI/acouslic_proxy_quality_manifest.csv`, `data/processed/ACOUSLIC_AI/acouslic_sweep_quality_summary.csv`, `README_FULL.md`, `scripts/01_prepare_hc18.py`, `scripts/02_generate_lq_hc18.py`.

HC18 has `999` processed manifest rows, with processed HR and mask images resized to `256` by `256` in `scripts/01_prepare_hc18.py`. `data/processed/HC18_SR/hc18_manifest.csv`; `scripts/01_prepare_hc18.py`

The split files contain `699` train rows, `150` validation rows, and `150` test rows. `data/processed/HC18_SR/train.csv`; `data/processed/HC18_SR/val.csv`; `data/processed/HC18_SR/test.csv`

The processed HC18 folders `HR`, `masks`, `LR_x2`, `LR_x4`, and `LQ_mixed` each contain `999` files in the audited workspace. `data/processed/HC18_SR/`

The HC18 LQ manifest contains `999` rows with `image_id`, `split`, `hr_path`, `lr_x2_path`, `lr_x4_path`, `lq_mixed_path`, and `mask_path`. `data/processed/HC18_SR/hc18_lq_manifest.csv`

ACOUSLIC-AI contains `100` sweep volumes under `data/processed/ACOUSLIC_AI/temp_mha/` and `10000` extracted frame records in the frame manifest. `data/processed/ACOUSLIC_AI/acouslic_frame_manifest.csv`

The ACOUSLIC proxy manifest contains `10000` frame rows with image-quality and motion-quality features plus `frame_quality_label` and `sweep_feedback_label`. `data/processed/ACOUSLIC_AI/acouslic_proxy_quality_manifest.csv`

The ACOUSLIC sweep summary contains `100` sweep-level rows with `poor_frame_ratio` and `final_sweep_feedback_label`. `data/processed/ACOUSLIC_AI/acouslic_sweep_quality_summary.csv`

The repo citation skeleton treats HC18 and ACOUSLIC-AI as public grand-challenge datasets. `README_FULL.md`

Exact licence and consent language is UNVERIFIED by repo files; official HC18 and ACOUSLIC-AI dataset licence pages would be needed for final submission wording. UNVERIFIED: needed files = dataset licence/terms documents.

## 5. Method

Evidence files read for this section: `scripts/01_prepare_hc18.py`, `scripts/02_generate_lq_hc18.py`, `scripts/hc18_dataset.py`, `scripts/hc18_diffusion_dataset.py`, `scripts/models/swinir_light.py`, `scripts/50_train_swinir_hc18.py`, `scripts/39_improve_hc18_unet_segmentation.py`, `scripts/46_downstream_utility_eval.py`, `scripts/24_compute_acouslic_proxy_quality.py`, `scripts/29_train_proxy_quality_embedding.py`, `scripts/28_adaptive_feedback_aware_enhancement.py`, `scripts/30_learned_adaptive_feedback_enhancement.py`, `scripts/60_apply_acouslic_mlp_to_hc18.py`, `scripts/61_cross_dataset_quality_analysis.py`.

### 5.1 Pipeline overview

Block diagram:

```text
HC18 raw images
  -> scripts/01_prepare_hc18.py
  -> HR + masks at 256x256
  -> scripts/02_generate_lq_hc18.py
  -> LR_x2 / LR_x4 / LQ_mixed
  -> scripts/50_train_swinir_hc18.py + scripts/models/swinir_light.py
  -> SwinIR-Light Phase A SR output
  -> scripts/46_downstream_utility_eval.py
  -> frozen improved U-Net Dice / IoU / measurement utility

ACOUSLIC-AI sweep volumes
  -> extracted frames
  -> scripts/24_compute_acouslic_proxy_quality.py
  -> 12 proxy image/motion features + frame/sweep labels
  -> scripts/29_train_proxy_quality_embedding.py
  -> proxy MLP
  -> scripts/28_adaptive_feedback_aware_enhancement.py or scripts/30_learned_adaptive_feedback_enhancement.py
  -> adaptive action and reassessment
  -> scripts/60_apply_acouslic_mlp_to_hc18.py + scripts/61_cross_dataset_quality_analysis.py
  -> zero-shot ACOUSLIC-to-HC18 proxy transfer
```

### 5.2 Degradation model

`scripts/01_prepare_hc18.py` resizes HC18 images to `256` by `256` with `cv2.INTER_AREA` and masks with `cv2.INTER_NEAREST`. `scripts/01_prepare_hc18.py`

`scripts/02_generate_lq_hc18.py` creates LR_x2 and LR_x4 through bicubic downsample-upsample. `scripts/02_generate_lq_hc18.py`

The LQ_mixed chain is LR_x4 -> Gaussian blur with radius `1.5` -> speckle noise with `noise_sigma=0.02` -> horizontal motion blur with `kernel_size=5` -> contrast reduction with factor `0.85`. `scripts/02_generate_lq_hc18.py`

### 5.3 SwinIR-Light Phase A

`SwinIRLight` is a grayscale same-resolution restoration model with `upscale=1`, `embed_dim=60`, default depths `(6, 6, 6, 6)`, default heads `(6, 6, 6, 6)`, window size `8`, and residual output `torch.clamp(x + residual, 0.0, 1.0)`. `scripts/models/swinir_light.py`

`build_swinir_light` constructs the canonical architecture with `embed_dim=60`, `mlp_ratio=2.0`, `upscale=1`, and activation checkpointing enabled by default. `scripts/models/swinir_light.py`

Phase A training defaults are `--epochs 100`, `--batch_size 8`, generator learning rate `2e-4`, discriminator learning rate `1e-4`, warmup epochs `5`, seed `42`, input type `LQ_mixed`, EMA decay `0.999`, and gradient clipping `1.0`. `scripts/50_train_swinir_hc18.py`

The Phase A generator loss is `args.l1_weight * loss_l1 + args.lpips_weight * loss_lpips + args.gan_weight * loss_gan`, with defaults `1.0`, `0.2`, and `0.005`. `scripts/50_train_swinir_hc18.py`

The canonical checkpoint path is `results/models/swinir_light_phase_a_seed42/best_lpips.pth`, and the binary checkpoint exists in the workspace. `README_FULL.md`; `results/models/swinir_light_phase_a_seed42/best_lpips.pth`

### 5.4 Downstream U-Net utility

The improved U-Net has encoder channels `32`, `64`, `128`, `256` and bottleneck `512`, with upsampling blocks back to a single-channel output. `scripts/39_improve_hc18_unet_segmentation.py`

The U-Net training summary reports `100` epochs, batch size `8`, AdamW, learning rate `0.001`, weight decay `0.0001`, BCEWithLogitsLoss + DiceLoss, best validation Dice `0.8776`, best validation IoU `0.8130`, best epoch `26`, HR test Dice `0.8897`, and threshold `0.50`. `reports/final_conference_report_appendix.md`

Downstream utility is evaluated by freezing this improved U-Net and scoring masks on HR, LQ, SR-only, perceptual CNNs, SwinIR-Light Phase A, and CNN-residual diffusion. `scripts/46_downstream_utility_eval.py`

This is anatomy-preservation support, not expert clinical validation. `reports/final_conference_report_appendix.md`

### 5.5 ACOUSLIC proxy quality

The ACOUSLIC feature set has 12 columns: `blur_score`, `contrast_score`, `brightness_mean`, `brightness_std`, `noise_proxy`, `edge_density`, `dark_ratio`, `bright_ratio`, `frame_difference_mean`, `optical_flow_mean_magnitude`, `optical_flow_std_magnitude`, and `jitter_score`. `scripts/29_train_proxy_quality_embedding.py`

The proxy labels are `good_frame`, `blurry_frame`, `low_contrast_frame`, `noisy_frame`, and `poor_frame`. `scripts/29_train_proxy_quality_embedding.py`

The labels are heuristic proxy labels, not clinical ground-truth annotations. `scripts/24_compute_acouslic_proxy_quality.py`; `scripts/29_train_proxy_quality_embedding.py`

`ProxyQualityEmbeddingMLP` is the learned quality embedding/classifier used for the ACOUSLIC branch. `scripts/29_train_proxy_quality_embedding.py`

### 5.6 Adaptive feedback policies

The heuristic policy maps `good_frame` to `keep_original`, `blurry_frame` to `enhance_blur`, `low_contrast_frame` to `enhance_contrast`, `noisy_frame` to `denoise_cautiously`, and `poor_frame` to `feedback_repeat_or_reacquire`. `scripts/28_adaptive_feedback_aware_enhancement.py`

The learned policy uses `choose_action`, with predicted labels and confidence deciding keep, enhance, CLAHE plus enhance, denoise, repeat/reacquire, or low-confidence conservative actions. `scripts/30_learned_adaptive_feedback_enhancement.py`

The available adaptive action vocabulary therefore includes keep, enhance, denoise, and request repeat/reacquisition behaviours. `scripts/28_adaptive_feedback_aware_enhancement.py`; `scripts/30_learned_adaptive_feedback_enhancement.py`

### 5.7 Cross-dataset transfer

`scripts/60_apply_acouslic_mlp_to_hc18.py` loads the frozen ACOUSLIC proxy-quality MLP, reuses the feature helpers from `scripts/24_compute_acouslic_proxy_quality.py`, validates feature order and label names, and applies the model to `150` HC18 test rows. `scripts/60_apply_acouslic_mlp_to_hc18.py`

`scripts/61_cross_dataset_quality_analysis.py` computes rankings, chi-square tests, paired probability deltas, Spearman correlation, and a Markdown report. `scripts/61_cross_dataset_quality_analysis.py`

The cross-dataset report states that the ACOUSLIC model is a feature-based 12-dimensional MLP, not a raw-image network. `docs/CROSS_DATASET_TRANSFER.md`

## 6. Experimental Setup

Evidence files read for this section: `README_FULL.md`, `requirements.txt`, `scripts/50_train_swinir_hc18.py`, `scripts/39_improve_hc18_unet_segmentation.py`, `scripts/29_train_proxy_quality_embedding.py`, `scripts/48_significance_report.py`, `scripts/42_compare_all_models.py`, `scripts/46_downstream_utility_eval.py`, `results/training_logs/50_train_swinir_hc18_swinir_light_phase_a_seed42_20260706_165538.csv`, `results/training_logs/50_train_swinir_hc18_swinir_light_phase_a_seed42_20260706_185725.csv`, `results/training_logs/50_train_swinir_hc18_swinir_light_phase_a_seed42_20260707_010144.csv`.

The audited environment is Windows 10 with PowerShell, Python `3.13.1`, PyTorch `2.12.1+cu126`, CUDA runtime `12.6`, NVIDIA GeForce RTX 4060 Laptop GPU, and `8 GB` VRAM. `README_FULL.md`

Canonical SwinIR-Light Phase A uses seed `42`, batch size `8`, generator learning rate `2e-4`, discriminator learning rate `1e-4`, warmup `5`, AdamW optimisers with betas `(0.9, 0.99)` and weight decay `1e-4`, EMA decay `0.999`, and cosine learning rate after warmup. `scripts/50_train_swinir_hc18.py`

The Phase A script default is `100` epochs, but the training logs opened in this session contain `12`, `20`, and `16` rows, covering epochs `1` through `48`. `scripts/50_train_swinir_hc18.py`; `results/training_logs/50_train_swinir_hc18_swinir_light_phase_a_seed42_20260706_165538.csv`; `results/training_logs/50_train_swinir_hc18_swinir_light_phase_a_seed42_20260706_185725.csv`; `results/training_logs/50_train_swinir_hc18_swinir_light_phase_a_seed42_20260707_010144.csv`

The U-Net script defaults are epochs `100`, batch size `8`, image size `256`, learning rate `1e-3`, and seed `42`. `scripts/39_improve_hc18_unet_segmentation.py`

The proxy-quality MLP script defaults are epochs `50`, batch size `128`, learning rate `1e-3`, and seed `42`. `scripts/29_train_proxy_quality_embedding.py`

Reconstruction evaluation metrics are PSNR, SSIM, LPIPS, DISTS, MAE, EdgeScore, NIQE, and BRISQUE. `scripts/42_compare_all_models.py`

Downstream evaluation metrics are Dice, IoU, area error, and equivalent-diameter error. `scripts/46_downstream_utility_eval.py`

PSNR, SSIM, and MAE are pixel/reconstruction metrics; LPIPS and DISTS are perceptual metrics; NIQE and BRISQUE are no-reference image-quality metrics; Dice and IoU are segmentation-overlap metrics. `README_FULL.md`

The statistical setup uses bootstrap mean confidence intervals with default seed `42`, Wilcoxon signed-rank tests on paired image-level differences, and Bonferroni correction across the registered comparison set. `scripts/48_significance_report.py`; `README_FULL.md`

## 7. Results

Evidence files read for this section: `results/final_all_hc18_comparison_summary.csv`, `results/bootstrap_metric_summary.csv`, `results/significance_report.csv`, `results/downstream_utility_summary.csv`, `results/acouslic_ai/proxy_quality_embedding_metrics.csv`, `results/acouslic_ai/adaptive_feedback_enhancement_summary.csv`, `results/acouslic_ai/learned_adaptive_enhancement_summary.csv`, `results/acouslic_ai/adaptive_sweep_feedback_change_summary.csv`, `results/acouslic_ai/adaptive_sweep_feedback_change_per_sweep.csv`, `results/backups/acouslic_ai_50sweeps/adaptive_feedback_enhancement_summary.csv`, `results/backups/acouslic_ai_50sweeps/adaptive_sweep_feedback_change_summary.csv`, `results/cross_dataset/cross_dataset_stats.json`, `results/cross_dataset/acouslic_quality_on_hc18_summary.csv`, `results/cross_dataset/method_ranking_by_acouslic_quality.csv`, `reports/final_table_figure_checklist.md`.

### 7.1 Reconstruction comparison - Table 1

Ranked by LPIPS, lower is better. Source for every row: `results/final_all_hc18_comparison_summary.csv`

| LPIPS rank | Method | PSNR | SSIM | LPIPS | DISTS | MAE | EdgeScore | NIQE | BRISQUE |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | SwinIR-Light Phase A | 29.72770638946017 | 0.8462304392579071 | 0.10091456212103367 | 0.15086822072664896 | 0.019428127873688937 | 0.984159038066864 | 6.07250085512797 | 39.51725341796875 |
| 2 | Quality-aware SR + LPIPS + GAN | 28.743526684454693 | 0.8266172298510083 | 0.1130732424557209 | 0.1671081018447876 | 0.021958102118223907 | 0.9826671433448791 | 6.160169426600138 | 27.430340169270835 |
| 3 | SR-only + LPIPS + GAN | 27.863696496405858 | 0.7813604113483116 | 0.12373436028758684 | 0.17729870637257894 | 0.02511708236609896 | 0.9812622725963592 | 6.8983197593688965 | 29.08691650390625 |
| 4 | SR-only | 30.98564898087667 | 0.8564114944492922 | 0.1749419870475928 | 0.17408846179644266 | 0.017652057191977898 | 0.9857001554965973 | 7.452263437906901 | 54.514773763020834 |
| 5 | Quality-aware SR | 30.58041972571543 | 0.8743366800436309 | 0.2042164137462775 | 0.1811862627665202 | 0.0173940657141308 | 0.9849764509995779 | 7.617840340932211 | 57.51572265625 |
| 6 | LQ_mixed baseline | 25.384943811967094 | 0.6722561101255639 | 0.350968104104201 | 0.24133360147476196 | 0.033745977462579804 | 0.9779292758305868 | 9.808565260569255 | 69.16806559244792 |

### 7.2 Bootstrap CIs - Table 2

Caption note: reconstruction metrics use 150 HC18 test images in `results/final_all_hc18_comparison_summary.csv`, but this bootstrap table uses `n` `148` aligned/mask-valid rows. `results/bootstrap_metric_summary.csv`

| Metric | Method | Mean | CI low | CI high | n | Source |
|---|---|---:|---:|---:|---:|---|
| LPIPS | LQ_mixed | 0.3506727686806305 | 0.3442903940238663 | 0.3569466292002314 | 148 | `results/bootstrap_metric_summary.csv` |
| LPIPS | Quality-aware | 0.20405285286943656 | 0.19711438928655275 | 0.21076973510302 | 148 | `results/bootstrap_metric_summary.csv` |
| LPIPS | Quality-aware+perceptual | 0.11303485361104071 | 0.10895135393114504 | 0.11686056593458188 | 148 | `results/bootstrap_metric_summary.csv` |
| LPIPS | SR-only | 0.1748358587658888 | 0.1683935107386394 | 0.18110409091679827 | 148 | `results/bootstrap_metric_summary.csv` |
| LPIPS | SR-only+perceptual | 0.1238735693733434 | 0.12023571541633553 | 0.1273959354182856 | 148 | `results/bootstrap_metric_summary.csv` |
| LPIPS | SwinIR-Light Phase A | 0.10084894095623005 | 0.09694007307265858 | 0.10476489356102266 | 148 | `results/bootstrap_metric_summary.csv` |
| DISTS | LQ_mixed | 0.24126080765917496 | 0.23825959495797353 | 0.24440243828135566 | 148 | `results/bootstrap_metric_summary.csv` |
| DISTS | Quality-aware | 0.18112999080000697 | 0.17752217030203019 | 0.18497552784109436 | 148 | `results/bootstrap_metric_summary.csv` |
| DISTS | Quality-aware+perceptual | 0.16714086162077413 | 0.16352529652618072 | 0.17102233101991376 | 148 | `results/bootstrap_metric_summary.csv` |
| DISTS | SR-only | 0.17400541821041624 | 0.16997573478399097 | 0.17776450199653973 | 148 | `results/bootstrap_metric_summary.csv` |
| DISTS | SR-only+perceptual | 0.17729988049816436 | 0.1742330178417064 | 0.1804981055291923 | 148 | `results/bootstrap_metric_summary.csv` |
| DISTS | SwinIR-Light Phase A | 0.15078885329736244 | 0.14757554924769975 | 0.15374142197338306 | 148 | `results/bootstrap_metric_summary.csv` |
| Dice | HR | 0.8896568280296211 | 0.8643529325771719 | 0.9104266438943754 | 148 | `results/bootstrap_metric_summary.csv` |
| Dice | LQ_mixed | 0.8849903062929627 | 0.8606901809692192 | 0.9056547213682631 | 148 | `results/bootstrap_metric_summary.csv` |
| Dice | SwinIR-Light Phase A | 0.8877019882187266 | 0.8637800827631902 | 0.9077806836890105 | 148 | `results/bootstrap_metric_summary.csv` |

### 7.3 Pairwise significance - Table 3

Full-test pairwise rows. Source for every row: `results/significance_report.csv`

| Method A | Method B | Metric | Delta | CI low | CI high | p_raw | p_bonf | Significant | n_pairs |
|---|---|---|---:|---:|---:|---:|---:|---|---:|
| SR-only | SR-only+perceptual | LPIPS | -0.050962289392545414 | -0.05726642634442731 | -0.0447339332410814 | 9.097485266357476e-22 | 3.548019253879416e-20 | True | 148 |
| SR-only+perceptual | Quality-aware+perceptual | LPIPS | -0.010838715762302683 | -0.01306449027505477 | -0.008251939617047038 | 4.50449559419414e-16 | 1.7567532817357146e-14 | True | 148 |
| SR-only+perceptual | SwinIR-Light Phase A | LPIPS | -0.023024628417113342 | -0.02642171496865214 | -0.01957347256735571 | 9.315147286221874e-21 | 3.632907441626531e-19 | True | 148 |
| Quality-aware+perceptual | SwinIR-Light Phase A | LPIPS | -0.01218591265481066 | -0.015609442373435648 | -0.00906372830913578 | 1.0223599873532368e-11 | 3.987203950677623e-10 | True | 148 |
| SR-only | SR-only+perceptual | Dice | 0.0019597252069589303 | -0.0002708456617061397 | 0.004963535758643626 | 0.3634174504841977 | 1.0 | False | 148 |
| SR-only+perceptual | Quality-aware+perceptual | Dice | -0.0014691163844730104 | -0.0033079115920721875 | 0.0003435826422968532 | 0.8285338229539583 | 1.0 | False | 148 |
| SR-only+perceptual | SwinIR-Light Phase A | Dice | 0.0004156285275811125 | -0.000837185079376294 | 0.0015862954492632281 | 0.013978197242663856 | 0.5451496924638903 | False | 148 |
| Quality-aware+perceptual | SwinIR-Light Phase A | Dice | 0.0018847449120541226 | 0.0003532878502746305 | 0.0037479954640823225 | 0.0006287310043785391 | 0.024520509170763025 | True | 148 |
| LQ_mixed | Quality-aware+perceptual | LPIPS | -0.23763791506958976 | -0.2432558720226626 | -0.2312236727429302 | 4.900420291456201e-26 | 1.9111639136679184e-24 | True | 148 |

### 7.4 Downstream Dice - Table 4

Source for every row: `results/downstream_utility_summary.csv`

| Method | Dice_mean | Dice_std | IoU_mean | AreaErr_mean | DiamErr_mean |
|---|---:|---:|---:|---:|---:|
| HR | 0.8896568280296212 | 0.14100148581871344 | 0.822474998718372 | 0.19910857293131018 | 16.205972770700587 |
| LQ_mixed | 0.8849903062929627 | 0.14143161413912664 | 0.8150281823989944 | 0.2019027575258408 | 16.377530370347838 |
| SR-only | 0.8853266344841867 | 0.1423929489302435 | 0.8158175674425396 | 0.20966380674464058 | 16.57690629901117 |
| SR-only+perceptual | 0.8872863596911454 | 0.14072515968918195 | 0.8184421331479875 | 0.2024589982767475 | 16.34140346066531 |
| Quality-aware | 0.8846884138798635 | 0.1423706881353163 | 0.8147794757826095 | 0.21140978243316016 | 16.65265340199787 |
| Quality-aware+perceptual | 0.8858172433066724 | 0.14127967377212142 | 0.8162364732093927 | 0.20425803129927317 | 16.489125559844812 |
| SwinIR-Light Phase A | 0.8877019882187267 | 0.1411037094661537 | 0.8192710507699988 | 0.20281509416480048 | 16.33906724232867 |
| CNN-Residual Diffusion | 0.8854304594623023 | 0.14129818886042322 | 0.8156440879874214 | 0.20786335413425644 | 16.550232474950814 |

SwinIR-Light Phase A Dice `0.8877019882187267` is close to HR Dice `0.8896568280296212`; this supports preservation near the frozen U-Net ceiling rather than a large clinical improvement claim. `results/downstream_utility_summary.csv`

### 7.5 ACOUSLIC proxy MLP metrics - Table 5

Source for every row: `results/acouslic_ai/proxy_quality_embedding_metrics.csv`

| Split | num_frames | accuracy | macro_f1 | f1_good_frame | f1_blurry_frame | f1_low_contrast_frame | f1_noisy_frame | f1_poor_frame |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 7000 | 0.98200000 | 0.97971677 | 0.98139975 | 0.95877863 | 0.98689956 | 0.97435897 | 0.99714693 |
| val | 1500 | 0.98133333 | 0.97788527 | 0.98351648 | 0.94680851 | 0.99200000 | 0.97404580 | 0.99305556 |
| test | 1500 | 0.97666667 | 0.96882341 | 0.97615262 | 0.95652174 | 0.94573643 | 0.97318008 | 0.99252616 |

### 7.6 Adaptive feedback - Table 6

| Branch | Scope | total_frames | number_of_frames_improved | percentage_improved | Source |
|---|---|---:|---:|---:|---|
| Current heuristic adaptive | overall | 10000 | 1794 | 17.94000000 | `results/acouslic_ai/adaptive_feedback_enhancement_summary.csv` |
| Current learned adaptive | overall | 10000 | 1724 | 17.24000000 | `results/acouslic_ai/learned_adaptive_enhancement_summary.csv` |
| Legacy heuristic adaptive | overall | 5000 | 1005 | 20.10000000 | `results/backups/acouslic_ai_50sweeps/adaptive_feedback_enhancement_summary.csv` |

The legacy `20.10000000` result is a 50-sweep backup with `5000` frames and must not be mixed with the current 100-sweep results. `results/backups/acouslic_ai_50sweeps/adaptive_feedback_enhancement_summary.csv`; `results/acouslic_ai/adaptive_feedback_enhancement_summary.csv`

### 7.7 Sweep reassessment - Table 7

| Quantity | Value | Source |
|---|---:|---|
| number_of_sweeps_analyzed | 100 | `results/acouslic_ai/adaptive_sweep_feedback_change_summary.csv` |
| number_of_sweeps_with_reduced_poor_frame_ratio | 100 | `results/acouslic_ai/adaptive_sweep_feedback_change_summary.csv` |
| mean_poor_frame_ratio_reduction | 0.17940000 | `results/acouslic_ai/adaptive_sweep_feedback_change_summary.csv` |
| number_of_sweeps_with_improved_good_frame_ratio | 100 | `results/acouslic_ai/adaptive_sweep_feedback_change_summary.csv` |
| number_of_repeat_sweep_cases_changed_to_acceptable_or_improve_visibility | 36 | `results/acouslic_ai/adaptive_sweep_feedback_change_summary.csv` |
| number_of_feedback_decisions_unchanged_due_to_motion_based_problems | 28 | `results/acouslic_ai/adaptive_sweep_feedback_change_summary.csv` |
| mean original poor-frame ratio | 0.5469 | computed from `results/acouslic_ai/adaptive_sweep_feedback_change_per_sweep.csv` |
| mean adaptive poor-frame ratio | 0.3675 | computed from `results/acouslic_ai/adaptive_sweep_feedback_change_per_sweep.csv` |

No file read in this session supports a `0.50 -> 0.21` poor-frame-ratio claim. UNVERIFIED: needed file = a result CSV with those two values and its producing script.

### 7.8 Cross-dataset transfer - Table 8

| Result | Exact value | Source |
|---|---:|---|
| best_sr | swinir_light_phase_a | `results/cross_dataset/cross_dataset_stats.json` |
| n_sources | 13 | `results/cross_dataset/cross_dataset_stats.json` |
| Spearman rho | 0.5688522618064861 | `results/cross_dataset/cross_dataset_stats.json` |
| Spearman p_value | 3.094730049128844e-14 | `results/cross_dataset/cross_dataset_stats.json` |
| Spearman ci_low | 0.455751144389122 | `results/cross_dataset/cross_dataset_stats.json` |
| Spearman ci_high | 0.6737837348922202 | `results/cross_dataset/cross_dataset_stats.json` |
| delta_prob_good n_pairs | 150 | `results/cross_dataset/cross_dataset_stats.json` |
| mean_delta_prob_good | 0.009160083369730758 | `results/cross_dataset/cross_dataset_stats.json` |
| delta_prob_good ci_low | 2.768901952996197e-09 | `results/cross_dataset/cross_dataset_stats.json` |
| delta_prob_good ci_high | 0.02335860834205065 | `results/cross_dataset/cross_dataset_stats.json` |
| verdict | STRONG TRANSFER | `results/cross_dataset/cross_dataset_stats.json` |

Chi-square tests:

| Comparison | chi2 | dof | p_value | Source |
|---|---:|---:|---:|---|
| LQ_mixed vs HR | 189.94248741912295 | 4 | 5.453232827129358e-40 | `results/cross_dataset/cross_dataset_stats.json` |
| swinir_light_phase_a vs LQ_mixed | 49.69281585466557 | 2 | 1.6193552814238292e-11 | `results/cross_dataset/cross_dataset_stats.json` |
| swinir_light_phase_a vs HR | 185.60614618863826 | 4 | 4.6596631034341996e-39 | `results/cross_dataset/cross_dataset_stats.json` |

ACOUSLIC proxy-quality ranking places HR rank `1`, SwinIR-Light Phase A rank `9`, and LQ_mixed rank `13`. `results/cross_dataset/method_ranking_by_acouslic_quality.csv`

### 7.9 Qualitative figure list

Workflow figure: `results/final_workflow_figure.png` is marked not found and should be generated before submission. `reports/final_table_figure_checklist.md`

Available figure candidates include `results/report_assets/figure_psnr_comparison.png`, `results/report_assets/figure_boundary_error.png`, `results/final_ablation_figures/acquisition_frame_improvement.png`, `results/final_ablation_figures/acquisition_sweep_feedback_changes.png`, `results/hc18_unet_segmentation_figures/improved_dice_iou_comparison.png`, and `results/acouslic_ai/proxy_quality_embedding_confusion_matrix.png`. `reports/final_table_figure_checklist.md`

Qualitative SR strips are UNVERIFIED as a named final figure because the checklist does not list a specific SR-strip artefact path. UNVERIFIED: needed file = final qualitative SR-strip PNG or figure manifest.

## 8. Ablations

Evidence files read for this section: `results/final_ablation_study.csv`, `results/final_all_hc18_comparison_summary.csv`, `results/acouslic_ai/adaptive_feedback_enhancement_summary.csv`, `results/acouslic_ai/learned_adaptive_enhancement_summary.csv`, `scripts/50_train_swinir_hc18.py`, `README_FULL.md`.

The repo has current comparison evidence for LQ_mixed baseline, SR-only, Quality-aware SR, SR-only + LPIPS + GAN, Quality-aware SR + LPIPS + GAN, and SwinIR-Light Phase A. `results/final_all_hc18_comparison_summary.csv`

An isolated "L1 + LPIPS without GAN" ablation is not present in the final comparison table. UNVERIFIED/planned: needed file = result CSV for a model trained with GAN weight `0` and LPIPS weight non-zero.

The final comparison shows SR-only LPIPS `0.1749419870475928`, SR-only + LPIPS + GAN LPIPS `0.12373436028758684`, Quality-aware SR LPIPS `0.2042164137462775`, and Quality-aware SR + LPIPS + GAN LPIPS `0.1130732424557209`. `results/final_all_hc18_comparison_summary.csv`

SwinIR-Light Phase A uses L1 + LPIPS + GAN with EMA, but there is no current final table that isolates "with EMA" versus "without EMA". `scripts/50_train_swinir_hc18.py`

With-vs-without EMA is therefore planned/not run as an ablation in the opened result files. UNVERIFIED/planned: needed file = paired SwinIR evaluation with EMA disabled or raw checkpoint evaluated separately.

Heuristic versus learned adaptive policy is backed by current results: heuristic improvement `17.94000000` and learned improvement `17.24000000`. `results/acouslic_ai/adaptive_feedback_enhancement_summary.csv`; `results/acouslic_ai/learned_adaptive_enhancement_summary.csv`

Older HC18 model ablation rows report SR-only CNN PSNR `30.7646`, Quality-aware CNN + L1/MSE PSNR `30.5746`, Conditional Diffusion SR PSNR `28.2914`, and CNN-Residual Diffusion SR PSNR `29.8726`. These are legacy ablation rows, not the current SwinIR headline table. `results/final_ablation_study.csv`

Phase B segmentation-aware SwinIR exists as code and checkpoints but is incomplete/deferred and not included in final comparison, downstream, or significance tables. `README_FULL.md`; `results/training_logs/51_train_segaware_hc18_phaseB_segaware_swinir_seed42_20260707_041015.csv`

## 9. Limitations

Evidence files read for this section: `README.md`, `README_FULL.md`, `reports/final_conference_report.md`, `reports/final_conference_report_appendix.md`, `reports/final_table_figure_checklist.md`, `docs/CROSS_DATASET_TRANSFER.md`, `results/training_logs/51_train_segaware_hc18_phaseB_segaware_swinir_seed42_20260707_041015.csv`, `results/downstream_utility_summary.csv`.

Proxy labels are not expert clinical labels; the ACOUSLIC frame labels are heuristic proxy labels. `scripts/24_compute_acouslic_proxy_quality.py`; `scripts/29_train_proxy_quality_embedding.py`

There was no expert clinical validation. `README.md`; `reports/final_table_figure_checklist.md`

HC18 degradation is simulated by a controlled chain of bicubic downsample-upsample, blur, speckle noise, motion blur, and contrast reduction, so it is not the same as real scanner acquisition artefacts. `scripts/02_generate_lq_hc18.py`

Feedback analysis is offline only and the system is not deployed for real-time clinical scanner use. `README.md`; `reports/final_table_figure_checklist.md`

Phase B segmentation-aware training is incomplete: the log file contains `8` rows, and Phase B is not in the final comparison, downstream utility, or significance tables. `results/training_logs/51_train_segaware_hc18_phaseB_segaware_swinir_seed42_20260707_041015.csv`; `README_FULL.md`

The HC18 reconstruction test set has `150` images, while downstream aligned/mask-valid significance and bootstrap rows use `148` pairs. `results/final_all_hc18_comparison_summary.csv`; `results/bootstrap_metric_summary.csv`

Downstream Dice improvement should be interpreted cautiously: SwinIR-Light Phase A Dice `0.8877019882187267` is close to HR Dice `0.8896568280296212` and LQ_mixed Dice `0.8849903062929627`. `results/downstream_utility_summary.csv`

Cross-dataset transfer is measured with proxy quality and LPIPS relationships, not clinical outcome. `docs/CROSS_DATASET_TRANSFER.md`; `results/cross_dataset/cross_dataset_stats.json`

## 10. Reproducibility

Evidence files read for this section: `README_FULL.md`, `requirements.txt`, `scripts/42_compare_all_models.py`, `scripts/46_downstream_utility_eval.py`, `scripts/48_significance_report.py`, `scripts/29_train_proxy_quality_embedding.py`, `scripts/28_adaptive_feedback_aware_enhancement.py`, `scripts/30_learned_adaptive_feedback_enhancement.py`, `scripts/27_sweep_feedback_change_analysis.py`, `scripts/60_apply_acouslic_mlp_to_hc18.py`, `scripts/61_cross_dataset_quality_analysis.py`, `scripts/35_create_final_workflow_figure.py`.

Key checkpoints are `results/models/swinir_light_phase_a_seed42/best_lpips.pth`, `results/models/hc18_unet_segmentation_best_improved.pth`, and `results/models/proxy_quality_embedding_mlp.pth`. `README_FULL.md`

The canonical seed is `42` in SwinIR-Light Phase A, the proxy MLP, cross-dataset transfer, and significance scripts. `scripts/50_train_swinir_hc18.py`; `scripts/29_train_proxy_quality_embedding.py`; `scripts/60_apply_acouslic_mlp_to_hc18.py`; `scripts/48_significance_report.py`

`requirements.txt` lists `numpy`, `pillow`, `scikit-image`, `torch`, `torchvision`, `tqdm`, `lpips`, `pyiqa`, `timm`, `torchmetrics`, `pytorch-msssim`, and `kornia`, but it does not list `opencv-python`. `requirements.txt`

Many scripts import `cv2`, including `scripts/01_prepare_hc18.py`, `scripts/02_generate_lq_hc18.py`, `scripts/24_compute_acouslic_proxy_quality.py`, `scripts/50_train_swinir_hc18.py`, `scripts/51_train_segaware_hc18.py`, and `scripts/60_apply_acouslic_mlp_to_hc18.py`. `scripts/01_prepare_hc18.py`; `scripts/02_generate_lq_hc18.py`; `scripts/24_compute_acouslic_proxy_quality.py`; `scripts/50_train_swinir_hc18.py`; `scripts/51_train_segaware_hc18.py`; `scripts/60_apply_acouslic_mlp_to_hc18.py`

Exact commands table:

| Claim/output | Script | Result file | Command |
|---|---|---|---|
| Reconstruction table | `scripts/42_compare_all_models.py` | `results/final_all_hc18_comparison_summary.csv` | `python .\scripts\42_compare_all_models.py` |
| Downstream Dice | `scripts/46_downstream_utility_eval.py` | `results/downstream_utility_summary.csv` | `python .\scripts\46_downstream_utility_eval.py` |
| Bootstrap CIs and significance | `scripts/48_significance_report.py` | `results/bootstrap_metric_summary.csv`, `results/significance_report.csv` | `python .\scripts\48_significance_report.py --device cpu` |
| Proxy MLP | `scripts/29_train_proxy_quality_embedding.py` | `results/acouslic_ai/proxy_quality_embedding_metrics.csv` | `python .\scripts\29_train_proxy_quality_embedding.py` |
| Heuristic adaptive feedback | `scripts/28_adaptive_feedback_aware_enhancement.py` | `results/acouslic_ai/adaptive_feedback_enhancement_summary.csv` | `python .\scripts\28_adaptive_feedback_aware_enhancement.py` |
| Learned adaptive feedback | `scripts/30_learned_adaptive_feedback_enhancement.py` | `results/acouslic_ai/learned_adaptive_enhancement_summary.csv` | `python .\scripts\30_learned_adaptive_feedback_enhancement.py` |
| Sweep reassessment | `scripts/27_sweep_feedback_change_analysis.py` | `results/acouslic_ai/adaptive_sweep_feedback_change_summary.csv` | `python .\scripts\27_sweep_feedback_change_analysis.py` |
| ACOUSLIC MLP on HC18 | `scripts/60_apply_acouslic_mlp_to_hc18.py` | `results/cross_dataset/acouslic_quality_on_hc18_per_image.csv` | `python .\scripts\60_apply_acouslic_mlp_to_hc18.py` |
| Cross-dataset statistics | `scripts/61_cross_dataset_quality_analysis.py` | `results/cross_dataset/cross_dataset_stats.json` | `python .\scripts\61_cross_dataset_quality_analysis.py` |
| Workflow figure | `scripts/35_create_final_workflow_figure.py` | `results/final_workflow_figure.png`, `results/final_workflow_figure.pdf` | `python .\scripts\35_create_final_workflow_figure.py` |

## 11. Ethical Considerations

Evidence files read for this section: `README.md`, `README_FULL.md`, `reports/final_conference_report.md`, `reports/final_table_figure_checklist.md`, `docs/CROSS_DATASET_TRANSFER.md`, `scripts/24_compute_acouslic_proxy_quality.py`, `scripts/29_train_proxy_quality_embedding.py`.

The repo treats HC18 and ACOUSLIC-AI as public challenge datasets via citation skeletons, but exact licence/consent text is not stored in the repo. `README_FULL.md`

No PHI handling claim can be fully verified from the repo files read; the safe thesis wording is that no additional private clinical data are introduced by this repository. UNVERIFIED: needed file = dataset terms or data-governance note.

There is no clinical deployment claim, no real-time scanner deployment claim, and no expert clinical validation claim. `README.md`; `reports/final_table_figure_checklist.md`

The main ethical risk is over-trust in restored images: a user may mistake perceptual sharpness for clinical correctness. `reports/final_conference_report.md`

Mitigations in this repo are multi-view evaluation: perceptual metrics, downstream Dice/IoU utility, ACOUSLIC proxy-quality flags, and explicit offline/proxy wording. `results/final_all_hc18_comparison_summary.csv`; `results/downstream_utility_summary.csv`; `results/cross_dataset/cross_dataset_stats.json`; `reports/final_table_figure_checklist.md`

## 12. Viva Q&A Bank

Evidence files read for this section: all files listed in sections 1-11, especially `README.md`, `README_FULL.md`, `scripts/02_generate_lq_hc18.py`, `scripts/50_train_swinir_hc18.py`, `scripts/models/swinir_light.py`, `scripts/39_improve_hc18_unet_segmentation.py`, `scripts/46_downstream_utility_eval.py`, `scripts/24_compute_acouslic_proxy_quality.py`, `scripts/29_train_proxy_quality_embedding.py`, `scripts/28_adaptive_feedback_aware_enhancement.py`, `scripts/30_learned_adaptive_feedback_enhancement.py`, `scripts/48_significance_report.py`, `results/final_all_hc18_comparison_summary.csv`, `results/downstream_utility_summary.csv`, `results/cross_dataset/cross_dataset_stats.json`.

**Q1. What problem does QAFUE solve?**  
QAFUE addresses fetal ultrasound enhancement when acquisition quality varies due to blur, noise, low contrast, unstable sweep motion, and incomplete acquisition. `README.md` It is not only an SR pipeline; it combines proxy quality estimation, restoration, adaptive actions, and feedback reassessment. `README.md` The clinical interpretation must remain cautious because this is an offline research workflow without expert clinical validation. `README.md`

**Q2. What is the canonical model in the current repo?**  
The canonical current reconstruction model is SwinIR-Light Phase A. `README_FULL.md` Its checkpoint is `results/models/swinir_light_phase_a_seed42/best_lpips.pth`. `README_FULL.md` Legacy CNN, perceptual+GAN, diffusion, and CNN-residual diffusion branches are baselines or historical branches, not the current headline reconstruction model. `README_FULL.md`

**Q3. Why not claim clinical superiority?**  
The repo explicitly says the project does not claim expert clinical validation or real-time clinical scanner deployment. `README.md` Downstream Dice is a frozen U-Net utility measure, not a sonographer-validated diagnosis. `reports/final_conference_report_appendix.md` Therefore the safest claim is anatomy-preservation support, not clinical superiority. `results/downstream_utility_summary.csv`

**Q4. Why use HC18?**  
HC18 provides fetal-head images and masks for controlled reconstruction and downstream segmentation utility. `README_FULL.md` The processed manifest contains `999` rows, split into `699` train, `150` validation, and `150` test rows. `data/processed/HC18_SR/hc18_manifest.csv`; `data/processed/HC18_SR/train.csv`; `data/processed/HC18_SR/val.csv`; `data/processed/HC18_SR/test.csv` This makes it suitable for paired LQ-to-HR restoration experiments. `scripts/02_generate_lq_hc18.py`

**Q5. Is the degradation realistic?**  
The degradation is controlled and simulated: LR_x4, Gaussian blur radius `1.5`, speckle noise `0.02`, motion blur kernel `5`, and contrast factor `0.85`. `scripts/02_generate_lq_hc18.py` It is useful for benchmark control but not the same as real scanner acquisition. `scripts/02_generate_lq_hc18.py` That is a limitation and should be stated clearly. `README_FULL.md`

**Q6. Why SwinIR-Light rather than a full SwinIR?**  
The repo implements `SwinIRLight`, not a full generic SwinIR training pipeline. `scripts/models/swinir_light.py` The model uses same-resolution residual restoration with `embed_dim=60`, window size `8`, and depths `(6, 6, 6, 6)`. `scripts/models/swinir_light.py` A full SwinIR comparison is UNVERIFIED because no final full-SwinIR result file was read. UNVERIFIED: needed file = full SwinIR evaluation CSV.

**Q7. Why does LPIPS matter more than PSNR here?**  
The current canonical claim is perceptual restoration: SwinIR-Light Phase A has the lowest LPIPS `0.10091456212103367` and DISTS `0.15086822072664896` in the registered final comparison. `results/final_all_hc18_comparison_summary.csv` PSNR and SSIM still matter, but the repo explicitly notes perceptual/GAN losses can improve LPIPS while reducing PSNR/SSIM. `README_FULL.md` Therefore the thesis should report both pixelwise and perceptual metrics. `results/final_all_hc18_comparison_summary.csv`

**Q8. Why is the GAN weight only 0.005?**  
The SwinIR Phase A defaults set `gan_weight` to `0.005`, much lower than L1 `1.0` and LPIPS `0.2`. `scripts/50_train_swinir_hc18.py` This makes adversarial training a small perceptual regulariser rather than the dominant objective. `scripts/50_train_swinir_hc18.py` Any stronger claim about why that value was chosen is UNVERIFIED unless a tuning note is added. UNVERIFIED: needed file = hyperparameter sweep record.

**Q9. Why combine L1, LPIPS, and GAN?**  
The loss stack in `train_one_epoch` explicitly combines L1, LPIPS, and GAN losses. `scripts/50_train_swinir_hc18.py` L1 anchors pixel fidelity, LPIPS targets perceptual similarity, and the small GAN term encourages perceptual realism. `scripts/50_train_swinir_hc18.py` This interpretation is consistent with the metric table where perceptual/GAN models lower LPIPS compared with their non-perceptual CNN versions. `results/final_all_hc18_comparison_summary.csv`

**Q10. What exactly did SwinIR improve?**  
SwinIR-Light Phase A has LPIPS `0.10091456212103367`, lower than Quality-aware SR + LPIPS + GAN at `0.1130732424557209`. `results/final_all_hc18_comparison_summary.csv` It has DISTS `0.15086822072664896`, lower than Quality-aware SR + LPIPS + GAN at `0.1671081018447876`. `results/final_all_hc18_comparison_summary.csv` It should not be described as dramatically improving Dice because the Dice changes are small. `results/downstream_utility_summary.csv`

**Q11. How should downstream Dice be interpreted?**  
SwinIR-Light Phase A Dice is `0.8877019882187267`, HR Dice is `0.8896568280296212`, and LQ_mixed Dice is `0.8849903062929627`. `results/downstream_utility_summary.csv` This means SwinIR preserves downstream segmentation utility near the HR ceiling of the frozen U-Net. `results/downstream_utility_summary.csv` It is not proof of clinical measurement improvement by expert readers. `reports/final_conference_report_appendix.md`

**Q12. Why use proxy labels in ACOUSLIC?**  
The ACOUSLIC branch needs quality labels for frame-level feedback experiments, but the repo labels them heuristically from image and motion features. `scripts/24_compute_acouslic_proxy_quality.py` The MLP learns these proxy classes and reaches test accuracy `0.97666667` and macro-F1 `0.96882341`. `results/acouslic_ai/proxy_quality_embedding_metrics.csv` They are acceptable for engineering experiments, not for clinical ground truth. `scripts/29_train_proxy_quality_embedding.py`

**Q13. What are the 12 proxy features?**  
The features are blur, contrast, brightness mean, brightness std, noise proxy, edge density, dark ratio, bright ratio, frame difference mean, optical-flow mean magnitude, optical-flow std magnitude, and jitter score. `scripts/29_train_proxy_quality_embedding.py` These combine image appearance and motion-derived signals. `scripts/24_compute_acouslic_proxy_quality.py` They support five proxy labels: good, blurry, low contrast, noisy, and poor. `scripts/29_train_proxy_quality_embedding.py`

**Q14. Why compare 100 sweeps and 50 sweeps carefully?**  
The current heuristic adaptive result uses `10000` frames and reports `17.94000000` percent improvement. `results/acouslic_ai/adaptive_feedback_enhancement_summary.csv` The legacy backup uses `5000` frames and reports `20.10000000` percent improvement. `results/backups/acouslic_ai_50sweeps/adaptive_feedback_enhancement_summary.csv` They are different experiments and must not be blended. `README_FULL.md`

**Q15. What changed after sweep reassessment?**  
Current sweep reassessment reports `100` analysed sweeps and `100` sweeps with reduced poor-frame ratio. `results/acouslic_ai/adaptive_sweep_feedback_change_summary.csv` The mean poor-frame-ratio reduction is `0.17940000`. `results/acouslic_ai/adaptive_sweep_feedback_change_summary.csv` The mean original-to-adaptive poor-frame ratio is `0.5469 -> 0.3675`, computed from the per-sweep file. `results/acouslic_ai/adaptive_sweep_feedback_change_per_sweep.csv`

**Q16. Why chi-square in cross-dataset analysis?**  
Chi-square tests compare proxy class distributions across HR, LQ_mixed, and SwinIR-restored HC18 sources. `scripts/61_cross_dataset_quality_analysis.py` The SwinIR versus LQ_mixed comparison reports chi2 `49.69281585466557`, dof `2`, and p-value `1.6193552814238292e-11`. `results/cross_dataset/cross_dataset_stats.json` This tests distributional shift in proxy classes, not clinical outcome. `docs/CROSS_DATASET_TRANSFER.md`

**Q17. Why Spearman in cross-dataset analysis?**  
Spearman relates paired improvement in ACOUSLIC proxy good-class probability to paired LPIPS improvement. `scripts/61_cross_dataset_quality_analysis.py` The reported rho is `0.5688522618064861` with p-value `3.094730049128844e-14`. `results/cross_dataset/cross_dataset_stats.json` This supports a monotonic association between proxy-quality shift and perceptual recovery, not a diagnostic claim. `docs/CROSS_DATASET_TRANSFER.md`

**Q18. What justifies the zero-shot claim?**  
`scripts/60_apply_acouslic_mlp_to_hc18.py` loads the frozen ACOUSLIC proxy-quality MLP and applies it to HC18 test images. `scripts/60_apply_acouslic_mlp_to_hc18.py` The cross-dataset report says this is done without fine-tuning on HC18. `docs/CROSS_DATASET_TRANSFER.md` The verdict is `STRONG TRANSFER` in the JSON stats, but that verdict is proxy-quality evidence only. `results/cross_dataset/cross_dataset_stats.json`

**Q19. What is Phase B and why is it not a headline result?**  
Phase B is segmentation-aware training with Charbonnier, LPIPS, DISTS, Sobel edge, GAN, and frozen U-Net segmentation loss. `scripts/51_train_segaware_hc18.py` The opened Phase B log has only `8` epochs. `results/training_logs/51_train_segaware_hc18_phaseB_segaware_swinir_seed42_20260707_041015.csv` It is not in the final comparison, downstream, or significance tables, so it is incomplete/deferred. `README_FULL.md`

**Q20. Are the NaNs in Phase B proof of numeric failure?**  
Not by themselves. `scripts/51_train_segaware_hc18.py` writes validation metrics only every `eval_dice_every` epochs or at the final epoch, otherwise placeholder `nan` values are expected. `scripts/51_train_segaware_hc18.py` The stronger risk is that the run is incomplete and early discriminator losses are very large, including `112972.30489404741` at epoch `4`. `results/training_logs/51_train_segaware_hc18_phaseB_segaware_swinir_seed42_20260707_041015.csv`

**Q21. What figure is missing before submission?**  
`results/final_workflow_figure.png` is listed as not found in the checklist. `reports/final_table_figure_checklist.md` The intended generator is `scripts/35_create_final_workflow_figure.py`, which writes PNG and PDF workflow figures. `scripts/35_create_final_workflow_figure.py` This should be generated before final paper submission. `reports/final_table_figure_checklist.md`

**Q22. What should be future work for OSEDiff or DRAT?**  
OSEDiff and DRAT are not present in the repo implementation. UNVERIFIED for current implementation: searches over scripts, docs, reports, and READMEs found no implementation files. They can be listed only as future one-step diffusion or diffusion-transformer directions. UNVERIFIED: needed file = implemented OSEDiff/DRAT training/evaluation script and result CSV.

**Q23. What is the clinical translation path?**  
The current system is offline and not deployed on a clinical scanner. `README.md` Translation would require expert sonographer validation, prospective testing, and real-time acquisition integration. `reports/final_conference_report.md` None of those steps is completed in the current repo. `reports/final_table_figure_checklist.md`

**Q24. Is n=148 statistically valid?**  
The reconstruction table uses all `150` HC18 test images, but the bootstrap/significance analysis uses `148` aligned pairs. `results/final_all_hc18_comparison_summary.csv`; `results/bootstrap_metric_summary.csv` The repo handles this by reporting `n` explicitly in the bootstrap and significance tables. `results/significance_report.csv` The viva answer should emphasise transparent sample accounting rather than overclaiming. `README_FULL.md`

**Q25. Why Bonferroni correction?**  
The significance script produces `p_bonf` and the README says multiple comparisons use Bonferroni correction across the registered comparison set. `scripts/48_significance_report.py`; `README_FULL.md` This is conservative and reduces false-positive risk when many paired comparisons are tested. `README_FULL.md` For SwinIR versus QA+perceptual LPIPS, the corrected p-value remains significant at `3.987203950677623e-10`. `results/significance_report.csv`

**Q26. What is the ethical risk of restored ultrasound images?**  
The main risk is over-trust: a sharper restored image may look clinically better even if it is not clinically validated. `reports/final_conference_report.md` The repo mitigates this by combining perceptual metrics, downstream Dice, and proxy quality checks. `results/final_all_hc18_comparison_summary.csv`; `results/downstream_utility_summary.csv`; `results/cross_dataset/cross_dataset_stats.json` It also repeatedly states that feedback and labels are proxy/offline. `README.md`

**Q27. What are likely failure cases?**  
Frames with poor acquisition, unstable motion, or motion-based feedback may remain problematic after image enhancement. `reports/final_conference_report.md` Current reassessment reports `28` feedback decisions unchanged due to motion-based problems. `results/acouslic_ai/adaptive_sweep_feedback_change_summary.csv` This means image restoration cannot fully fix bad acquisition. `reports/final_conference_report.md`

**Q28. Were ablation choices complete?**  
No. The repo has strong comparisons among current registered models, but an isolated L1+LPIPS-without-GAN SwinIR ablation is not in the final result files. `results/final_all_hc18_comparison_summary.csv` With-vs-without EMA is also not backed by a result table. `scripts/50_train_swinir_hc18.py` These should be described as planned/not run. UNVERIFIED: needed files = dedicated ablation CSVs.

**Q29. What is known about cost and inference time?**  
The environment is documented as RTX 4060 Laptop GPU with `8 GB` VRAM. `README_FULL.md` The final SwinIR comparison table opened here does not contain inference time columns. `results/final_all_hc18_comparison_summary.csv` Exact runtime/cost claims are therefore UNVERIFIED unless a timed benchmark is added. UNVERIFIED: needed file = inference-time benchmark CSV.

**Q30. What is the single safest thesis claim?**  
The safest claim is that SwinIR-Light Phase A gives the best LPIPS and DISTS among the registered current comparison set, while maintaining downstream Dice near the HR/LQ ceiling of the frozen U-Net. `results/final_all_hc18_comparison_summary.csv`; `results/downstream_utility_summary.csv` The ACOUSLIC branch adds proxy-quality feedback and zero-shot proxy transfer evidence. `results/acouslic_ai/proxy_quality_embedding_metrics.csv`; `results/cross_dataset/cross_dataset_stats.json` None of this should be phrased as expert clinical validation or real-time clinical deployment. `README.md`

## 13. Future Work

Evidence files read for this section: `README_FULL.md`, `scripts/51_train_segaware_hc18.py`, `results/training_logs/51_train_segaware_hc18_phaseB_segaware_swinir_seed42_20260707_041015.csv`, `reports/final_table_figure_checklist.md`, `README.md`.

Finish Phase B segmentation-aware training and add it to the final comparison, downstream, and significance tables before presenting it as a result. `scripts/51_train_segaware_hc18.py`; `results/training_logs/51_train_segaware_hc18_phaseB_segaware_swinir_seed42_20260707_041015.csv`

Generate the missing workflow figure before submission. `scripts/35_create_final_workflow_figure.py`; `reports/final_table_figure_checklist.md`

One-step diffusion such as OSEDiff and diffusion-transformer directions such as DRAT are future work only; they are not implemented in the current repo. UNVERIFIED for current implementation: needed files = training/evaluation scripts and result tables.

Run real clinical validation with expert sonographers, because the current labels are proxy labels and the current downstream utility is a frozen U-Net measurement, not expert interpretation. `README.md`; `reports/final_conference_report_appendix.md`

Explore real-time on-device deployment only after profiling inference time and validating safety, because the current workflow is offline. `README.md`

## CHECKLIST - Result Files Actually Opened

- `results/final_all_hc18_comparison_summary.csv`
- `results/downstream_utility_summary.csv`
- `results/bootstrap_metric_summary.csv`
- `results/significance_report.csv`
- `results/acouslic_ai/proxy_quality_embedding_metrics.csv`
- `results/acouslic_ai/adaptive_feedback_enhancement_summary.csv`
- `results/acouslic_ai/learned_adaptive_enhancement_summary.csv`
- `results/acouslic_ai/adaptive_sweep_feedback_change_summary.csv`
- `results/acouslic_ai/adaptive_sweep_feedback_change_per_sweep.csv`
- `results/backups/acouslic_ai_50sweeps/adaptive_feedback_enhancement_summary.csv`
- `results/backups/acouslic_ai_50sweeps/adaptive_sweep_feedback_change_summary.csv`
- `results/backups/acouslic_ai_50sweeps/adaptive_sweep_feedback_change_per_sweep.csv`
- `results/cross_dataset/cross_dataset_stats.json`
- `results/cross_dataset/acouslic_quality_on_hc18_summary.csv`
- `results/cross_dataset/method_ranking_by_acouslic_quality.csv`
- `results/final_ablation_study.csv`
- `results/eval/swinir_light_phase_a/test/hard_subsets.csv`
- `results/training_logs/50_train_swinir_hc18_swinir_light_phase_a_seed42_20260706_165538.csv`
- `results/training_logs/50_train_swinir_hc18_swinir_light_phase_a_seed42_20260706_185725.csv`
- `results/training_logs/50_train_swinir_hc18_swinir_light_phase_a_seed42_20260707_010144.csv`
- `results/training_logs/51_train_segaware_hc18_phaseB_segaware_swinir_seed42_20260707_041015.csv`

## CHECKLIST - Unable to Verify

- Exact official HC18 and ACOUSLIC-AI licence/consent wording was not stored in the repo; needed files are official dataset licence/terms documents.
- Probe angle and acoustic shadowing as motivation factors were not grounded in the opened repo files; needed file is a clinical-background source.
- Sparse-coding SR, SRCNN, EDSR, RCAN, SRGAN, ESRGAN, SR3, OSEDiff, and DRAT bibliographic details were not fully grounded in repo files; needed file is a complete related-work bibliography.
- Full SwinIR versus SwinIR-Light comparison was not found; needed file is a full-SwinIR evaluation CSV.
- Hyperparameter rationale for GAN weight `0.005` was not found; needed file is a hyperparameter sweep or design note.
- Isolated L1+LPIPS without GAN ablation was not found; needed file is a result CSV for that model.
- With-EMA versus without-EMA ablation was not found; needed file is a paired SwinIR evaluation table.
- A `0.50 -> 0.21` poor-frame-ratio claim was not found; needed file is a result CSV containing those exact values.
- Final qualitative SR-strip artefact was not identified in the checklist; needed file is a named final figure path.
- SwinIR inference-time/cost benchmark was not found in the opened final result files; needed file is an inference-time benchmark CSV.
