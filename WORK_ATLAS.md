# WORK_ATLAS.md

## Audit Note

This atlas is based only on files present in `D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project`. Paths are repository-relative unless stated otherwise.

The requested `tree -L 3 -I "__pycache__|.git|.venv|node_modules|*.pyc"` command is Unix-style and failed under Windows PowerShell. I reconstructed the same depth-limited repository view with PowerShell. No project pipeline files were changed during the audit.

`CIRCULAR.pdf` is an image-only PDF in this environment; text extraction was not possible without OCR. `DEMO_REPORTS.zip` contains the three report Markdown files under `reports/`, so it does not introduce a separate results source.

## Phase 1 - Repository Index

### Top-Level Structure

| Path | Role |
|---|---|
| `.agents/` | Local task/agent metadata; not part of the research pipeline. |
| `.git/` | Version history. Important commits include `3d0f9fa` for SwinIR registration and `9c8bab0` for cross-dataset transfer. |
| `data/` | Raw and processed HC18/ACOUSLIC data. |
| `docs/` | Cross-dataset transfer documentation. |
| `reports/` | Conference report, appendix, and final table/figure checklist. |
| `results/` | Evaluation CSVs, figures, checkpoints, reports, generated masks, and archived outputs. |
| `scripts/` | Data preparation, training, evaluation, statistics, plotting, and cleanup scripts. |
| `README.md` | Short project overview and scope caveats. |
| `README_FULL.md` | Main self-audit and canonical SwinIR-Light result summary. |
| `requirements.txt` | Python dependency list; OpenCV is used by scripts but is not listed. |
| `CIRCULAR.pdf` | Image-only conference/support PDF; not used as evidence for experimental claims. |
| `DEMO_REPORTS.zip` | Packaged copy of report Markdown files. |

### Data Inventory

| Path | Contents |
|---|---|
| `data/processed/HC18_SR/HR/` | 999 processed HR HC18 images. |
| `data/processed/HC18_SR/masks/` | 999 processed masks. |
| `data/processed/HC18_SR/LR_x2/` | 999 bicubic x2 degraded/restored inputs. |
| `data/processed/HC18_SR/LR_x4/` | 999 bicubic x4 degraded/restored inputs. |
| `data/processed/HC18_SR/LQ_mixed/` | 999 mixed low-quality inputs. |
| `data/processed/HC18_SR/train.csv` | 699 training rows. |
| `data/processed/HC18_SR/val.csv` | 150 validation rows. |
| `data/processed/HC18_SR/test.csv` | 150 test rows. |
| `data/processed/HC18_SR/hc18_manifest.csv` | 999 rows, HR/mask/source/split metadata. |
| `data/processed/HC18_SR/hc18_lq_manifest.csv` | 999 rows, HR/LR/LQ/mask metadata. |
| `data/processed/ACOUSLIC_AI/frames/` | 10,000 extracted ACOUSLIC frames. |
| `data/processed/ACOUSLIC_AI/temp_mha/` | 100 sampled sweep volumes. |
| `data/processed/ACOUSLIC_AI/acouslic_frame_manifest.csv` | 10,000 frame rows. |
| `data/processed/ACOUSLIC_AI/acouslic_proxy_quality_manifest.csv` | 10,000 proxy quality rows. |
| `data/processed/ACOUSLIC_AI/acouslic_sweep_quality_summary.csv` | 100 sweep-level rows. |

### Script Families

| Family | Main Files |
|---|---|
| HC18 preparation/degradation | `scripts/01_prepare_hc18.py`, `scripts/02_generate_lq_hc18.py`, `scripts/hc18_dataset.py` |
| CNN SR baselines | `scripts/05_train_sr_only_hc18.py`, `scripts/07_train_quality_aware_sr_hc18.py`, `scripts/40_train_sr_perceptual_hc18.py` |
| Diffusion baselines | `scripts/11_train_diffusion_sr_hc18.py` through `scripts/21_eval_cnn_residual_diffusion_sr_hc18.py` |
| SwinIR-Light Phase A | `scripts/models/swinir_light.py`, `scripts/50_train_swinir_hc18.py` |
| Segmentation-aware Phase B | `scripts/51_train_segaware_hc18.py` |
| U-Net downstream anatomy utility | `scripts/39_improve_hc18_unet_segmentation.py`, `scripts/46_downstream_utility_eval.py` |
| Final model comparison/statistics | `scripts/42_compare_all_models.py`, `scripts/47_hard_subset_analysis.py`, `scripts/48_significance_report.py` |
| ACOUSLIC proxy quality/feedback | `scripts/23_inspect_acouslic_zip.py` through `scripts/31_final_ablation_study.py` |
| Cross-dataset transfer | `scripts/60_apply_acouslic_mlp_to_hc18.py`, `scripts/61_cross_dataset_quality_analysis.py` |
| Figures/tables/cleanup | `scripts/35_create_final_workflow_figure.py`, `scripts/36_create_final_ablation_tables.py`, `scripts/40_cleanup_unreliable_dice_results.py`, `scripts/99_cleanup_unnecessary_data.py` |

### Drafts, Reports, and Assets

No `.tex`, `.docx`, or `.ipynb` files were found. The paper/report artefacts are Markdown:

- `reports/final_conference_report.md`
- `reports/final_conference_report_appendix.md`
- `reports/final_table_figure_checklist.md`
- `docs/CROSS_DATASET_TRANSFER.md`
- `results/report_assets/paper_summary.md`
- multiple generated result tables under `results/*.md`

The checklist explicitly says `results/final_workflow_figure.png` is missing, although `scripts/35_create_final_workflow_figure.py` is designed to generate both `results/final_workflow_figure.png` and `results/final_workflow_figure.pdf`.

## Phase 2 - A-to-Z Glossary of Real Components

**ACOUSLIC-AI branch**  
The ACOUSLIC branch extracts sweep frames, computes proxy image/motion quality, trains a proxy MLP, and evaluates adaptive feedback. Main files: `data/processed/ACOUSLIC_AI/acouslic_proxy_quality_manifest.csv`, `scripts/24_compute_acouslic_proxy_quality.py`, `scripts/29_train_proxy_quality_embedding.py`, `scripts/28_adaptive_feedback_aware_enhancement.py`.

**Bootstrap confidence intervals**  
There are two CI systems. Older broad CIs are in `scripts/32_bootstrap_confidence_intervals.py` and `results/bootstrap_confidence_intervals.csv`. The canonical SwinIR statistical CI/p-value source is `scripts/48_significance_report.py`, producing `results/bootstrap_metric_summary.csv` and `results/significance_report.csv`.

**Cross-dataset transfer**  
ACOUSLIC-trained proxy quality MLP is applied zero-shot to HC18 test images in `scripts/60_apply_acouslic_mlp_to_hc18.py`. Statistical analysis is in `scripts/61_cross_dataset_quality_analysis.py`, with the written summary in `docs/CROSS_DATASET_TRANSFER.md`.

**Degradation pipeline**  
HC18 HR images are converted into LR and mixed-LQ inputs by `scripts/02_generate_lq_hc18.py`. The mixed degradation uses bicubic x4, Gaussian blur, speckle noise, motion blur, and contrast reduction.

**EMA checkpoints**  
SwinIR-Light training maintains exponential moving average weights through `ModelEMA` in `scripts/50_train_swinir_hc18.py`. Stored checkpoints include `results/models/swinir_light_phase_a_seed42/ema.pth`, `best_lpips.pth`, `best_dice.pth`, and `last.pth`.

**Feedback-aware enhancement**  
Heuristic adaptive enhancement is implemented in `scripts/28_adaptive_feedback_aware_enhancement.py`, especially `apply_adaptive_policy`. Learned feedback-aware enhancement is in `scripts/30_learned_adaptive_feedback_enhancement.py`, especially `choose_action`.

**GAN/perceptual losses**  
CNN perceptual/GAN training is in `scripts/40_train_sr_perceptual_hc18.py`. SwinIR-Light Phase A uses L1 + LPIPS + GAN inside `train_one_epoch` in `scripts/50_train_swinir_hc18.py`.

**HC18_SR dataset**  
The processed HC18 SR dataset lives under `data/processed/HC18_SR/`. Dataset loading code is in `scripts/hc18_dataset.py` and `scripts/hc18_diffusion_dataset.py`.

**Improved U-Net**  
The final downstream segmentation utility model is built in `scripts/39_improve_hc18_unet_segmentation.py`. The appendix reports best validation Dice 0.8776 and HR test Dice 0.8897 in `reports/final_conference_report_appendix.md`.

**JSON cross-dataset stats**  
The main machine-readable cross-dataset statistics are in `results/cross_dataset/cross_dataset_stats.json`. This stores chi-square tests, Spearman correlation, and the `STRONG TRANSFER` verdict.

**Key checkpoints**  
Important checkpoint paths include `results/models/swinir_light_phase_a_seed42/best_lpips.pth`, `results/models/hc18_unet_segmentation_best_improved.pth`, `results/models/proxy_quality_embedding_mlp.pth`, and older CNN/diffusion checkpoints under `results/models/`.

**LPIPS/DISTS metrics**  
LPIPS and DISTS are computed in the final comparison pipeline `scripts/42_compare_all_models.py`. The canonical SwinIR summary is `results/final_all_hc18_comparison_summary.csv`.

**MLP proxy quality embedding**  
The proxy-quality MLP is in `scripts/29_train_proxy_quality_embedding.py`. It uses 12 image/motion features and predicts labels such as good, blurry, low contrast, noisy, and poor.

**No-reference metrics**  
NIQE and BRISQUE are included in `scripts/42_compare_all_models.py` and stored in `results/final_all_hc18_comparison_summary.csv`. They support image-quality comparison but are not the main anatomical utility claim.

**Orphan/archived results**  
`results/archived_unreliable_dice_iou/README.md` states that early threshold-based Dice/IoU results and the first U-Net attempt are unreliable and not used in the final paper.

**Phase A SwinIR-Light**  
The canonical current reconstruction model is SwinIR-Light Phase A. Training is in `scripts/50_train_swinir_hc18.py`, architecture is in `scripts/models/swinir_light.py`, and the main checkpoint is `results/models/swinir_light_phase_a_seed42/best_lpips.pth`.

**Quality-aware CNN**  
The older quality-aware CNN uses FiLM-like conditioning and 5D quality features in `scripts/07_train_quality_aware_sr_hc18.py`. Perceptual/GAN variants are trained in `scripts/40_train_sr_perceptual_hc18.py`.

**Reports**  
The older conference report is `reports/final_conference_report.md`. The newer canonical self-audit is `README_FULL.md`, which places SwinIR-Light Phase A as the best LPIPS/DISTS result.

**Segmentation-aware Phase B**  
Phase B is implemented in `scripts/51_train_segaware_hc18.py`. It combines reconstruction, LPIPS, DISTS, Sobel edge, GAN, and frozen U-Net segmentation loss. It is incomplete and not part of the canonical headline tables.

**Training logs**  
Training logs live in `results/training_logs/`. For Phase B, `results/training_logs/51_train_segaware_hc18_phaseB_segaware_swinir_seed42_20260707_041015.csv` records only 8 epochs of an intended longer run.

**U-Net downstream utility**  
Downstream anatomical utility is evaluated in `scripts/46_downstream_utility_eval.py`, producing `results/downstream_utility_summary.csv` and per-image masks.

**Validation/test split**  
HC18 has 699 train, 150 validation, and 150 test rows in `data/processed/HC18_SR/train.csv`, `val.csv`, and `test.csv`.

**Workflow figure**  
`results/final_workflow_figure.png` is missing. `scripts/35_create_final_workflow_figure.py` should generate it, and `reports/final_table_figure_checklist.md` marks it as needed before submission.

**x0-loss diffusion**  
`scripts/18_train_diffusion_sr_x0loss_hc18.py` and `scripts/19_eval_diffusion_sr_x0loss_hc18.py` implement the x0-loss diffusion branch. Its result files appear partial compared with the final SwinIR pipeline.

**Zero-shot transfer**  
The cross-dataset claim is zero-shot: ACOUSLIC proxy quality MLP is applied to HC18 without HC18 fine-tuning. This is documented in `docs/CROSS_DATASET_TRANSFER.md`.

## Phase 3 - End-to-End Pipeline Story

### 1. Raw HC18 to HR/mask splits

`scripts/01_prepare_hc18.py` is a top-level preparation script with no function wrapper. It reads raw HC18 files, filters annotation files, resizes images/masks to 256 x 256, writes `data/processed/HC18_SR/hc18_manifest.csv`, and writes `train.csv`, `val.csv`, and `test.csv`.

### 2. HR to LR/LQ degradation

`generate_lq_images` in `scripts/02_generate_lq_hc18.py` creates `LR_x2`, `LR_x4`, and `LQ_mixed`. The degradation functions are:

- `downsample_and_upsample`
- `add_gaussian_blur`
- `add_speckle_noise`
- `add_motion_blur`
- `reduce_contrast`

The effective LQ path is HR -> bicubic LR_x4 -> blur -> speckle -> motion blur -> contrast reduction.

### 3. Dataset loading

`HC18SRDataset` in `scripts/hc18_dataset.py` loads HR/LQ/mask examples for standard SR. `HC18DiffusionDataset` in `scripts/hc18_diffusion_dataset.py` adds quality features for diffusion-style experiments.

### 4. LQ to SR prediction

For the canonical model, `SwinIRLight` in `scripts/models/swinir_light.py` performs same-resolution restoration with a residual output. `build_swinir_light` constructs the architecture. Training calls the model inside `train_one_epoch` in `scripts/50_train_swinir_hc18.py`.

### 5. Loss computation

Phase A SwinIR-Light uses L1 + LPIPS + GAN loss in `train_one_epoch` in `scripts/50_train_swinir_hc18.py`. The defaults are L1 weight 1.0, LPIPS weight 0.2, and GAN weight 0.005 from `parse_args`.

The older CNN perceptual/GAN branch uses `VGGPerceptualLoss`, `PatchDiscriminator`, and `train_one_epoch` in `scripts/40_train_sr_perceptual_hc18.py`.

### 6. Backpropagation and checkpointing

`train_one_epoch` in `scripts/50_train_swinir_hc18.py` performs discriminator and generator updates, gradient clipping, and EMA update. `save_checkpoint` writes model state. `run_training` manages best LPIPS, best Dice, EMA, and last checkpoints.

Canonical SwinIR checkpoint: `results/models/swinir_light_phase_a_seed42/best_lpips.pth`.

### 7. Reconstruction evaluation

`scripts/42_compare_all_models.py` evaluates LQ, CNN baselines, perceptual/GAN baselines, and SwinIR. It computes PSNR, SSIM, LPIPS, DISTS, MAE, EdgeScore, NIQE, and BRISQUE. Outputs include:

- `results/final_all_hc18_comparison_per_image.csv`
- `results/final_all_hc18_comparison_summary.csv`
- `results/eval/swinir_light_phase_a/test/recon_summary.csv`

### 8. Downstream U-Net/Dice evaluation

`scripts/46_downstream_utility_eval.py` loads the improved U-Net from `scripts/39_improve_hc18_unet_segmentation.py`. It evaluates whether restorations preserve fetal-head segmentation utility. Main outputs:

- `results/downstream_utility_per_image.csv`
- `results/downstream_utility_summary.csv`
- `results/eval/swinir_light_phase_a/test/downstream_summary.csv`

### 9. ACOUSLIC proxy quality and feedback

`compute_frame_features`, `label_frame_quality`, `add_motion_features`, and `build_sweep_summary` in `scripts/24_compute_acouslic_proxy_quality.py` create frame/sweep-level proxy labels. The MLP in `scripts/29_train_proxy_quality_embedding.py` learns those labels.

Heuristic adaptive enhancement uses `apply_adaptive_policy` in `scripts/28_adaptive_feedback_aware_enhancement.py`. Learned adaptive enhancement uses `choose_action` in `scripts/30_learned_adaptive_feedback_enhancement.py`.

### 10. Closed-loop feedback reassessment

`compute_adaptive_feedback` and `build_per_sweep_dataframe` in `scripts/27_sweep_feedback_change_analysis.py` compare original and adaptive sweep feedback. Current 100-sweep files show mean poor-frame ratio changing from about 0.5469 to 0.3675, a 0.1794 reduction.

A previously reported 50-sweep branch shows 20.10% frame-level improvement, but its backup poor-frame ratio is about 0.547 to 0.346, not 0.50 to 0.21. I found no current result file supporting an exact 0.50 -> 0.21 poor-frame-ratio claim.

### 11. Cross-dataset transfer

`run` in `scripts/60_apply_acouslic_mlp_to_hc18.py` applies the ACOUSLIC-trained MLP to HC18 HR, LQ, and restoration sources. `run` and `build_report` in `scripts/61_cross_dataset_quality_analysis.py` compute chi-square tests, Spearman correlation, rankings, and the final report.

Outputs:

- `results/cross_dataset/acouslic_quality_on_hc18_per_image.csv`
- `results/cross_dataset/acouslic_quality_on_hc18_summary.csv`
- `results/cross_dataset/cross_dataset_stats.json`
- `docs/CROSS_DATASET_TRANSFER.md`

## Phase 4 - Results Reproducibility Map

| Claim | Exact result file | Producing script | Command / status |
|---|---|---|---|
| SwinIR LPIPS 0.1009 | `results/final_all_hc18_comparison_summary.csv` | `scripts/42_compare_all_models.py` | `python .\scripts\42_compare_all_models.py` |
| SwinIR LPIPS CI [0.0969, 0.1050] | `results/bootstrap_metric_summary.csv` | `scripts/48_significance_report.py` | `python .\scripts\48_significance_report.py` |
| SwinIR DISTS 0.1509 | `results/final_all_hc18_comparison_summary.csv` | `scripts/42_compare_all_models.py` | `python .\scripts\42_compare_all_models.py` |
| SwinIR Dice 0.8877 | `results/downstream_utility_summary.csv` | `scripts/46_downstream_utility_eval.py` | `python .\scripts\46_downstream_utility_eval.py` |
| SwinIR vs QA+perceptual LPIPS p_bonf 3.99e-10 | `results/significance_report.csv` | `scripts/48_significance_report.py` | `python .\scripts\48_significance_report.py` |
| SwinIR vs SR-only+perceptual LPIPS p_bonf 3.63e-19 | `results/significance_report.csv` | `scripts/48_significance_report.py` | `python .\scripts\48_significance_report.py` |
| Proxy MLP test accuracy 0.9767, macro-F1 0.9688 | `results/acouslic_ai/proxy_quality_embedding_metrics.csv` | `scripts/29_train_proxy_quality_embedding.py` | `python .\scripts\29_train_proxy_quality_embedding.py` |
| Current heuristic adaptive improvement 17.94% | `results/acouslic_ai/adaptive_feedback_enhancement_summary.csv` | `scripts/28_adaptive_feedback_aware_enhancement.py` | `python .\scripts\28_adaptive_feedback_aware_enhancement.py` |
| Current learned adaptive improvement 17.24% | `results/acouslic_ai/learned_adaptive_enhancement_summary.csv` | `scripts/30_learned_adaptive_feedback_enhancement.py` | `python .\scripts\30_learned_adaptive_feedback_enhancement.py` |
| Current feedback reassessment, 100/100 sweeps reduced poor-frame ratio | `results/acouslic_ai/adaptive_sweep_feedback_change_summary.csv` | `scripts/27_sweep_feedback_change_analysis.py` | `python .\scripts\27_sweep_feedback_change_analysis.py` |
| Legacy 20.10% adaptive improvement | `results/backups/acouslic_ai_50sweeps/adaptive_feedback_enhancement_summary.csv`; `results/final_ablation_original_blind_adaptive.csv` | `scripts/28_adaptive_feedback_aware_enhancement.py`; table likely from `scripts/36_create_final_ablation_tables.py` | UNCLEAR: exact historical command is not encoded in the result files. |
| Cross-dataset Spearman rho 0.5689 | `results/cross_dataset/cross_dataset_stats.json` | `scripts/61_cross_dataset_quality_analysis.py` | `python .\scripts\61_cross_dataset_quality_analysis.py` |
| Cross-dataset chi-square tests | `results/cross_dataset/cross_dataset_stats.json` | `scripts/61_cross_dataset_quality_analysis.py` | `python .\scripts\61_cross_dataset_quality_analysis.py` |

Important nuance: `results/final_all_hc18_comparison_summary.csv` uses 150 HC18 test images for reconstruction metrics. `results/bootstrap_metric_summary.csv` reports SwinIR full-test statistics over 148 aligned/mask-valid pairs.

## Phase 5 - Gaps, Risks, and TODOs

### Canonical Story Risk

The project has two narratives:

1. Older conference narrative: CNN, diffusion, adaptive feedback, and CNN-residual diffusion, mainly in `reports/final_conference_report.md` and `results/final_ablation_study.csv`.
2. Newer canonical self-audit: SwinIR-Light Phase A is the best reconstruction model, mainly in `README_FULL.md`, `scripts/42_compare_all_models.py`, and `scripts/48_significance_report.py`.

For submission, the paper should explicitly state that SwinIR-Light Phase A is the current canonical reconstruction result and that earlier CNN/diffusion tables are historical baselines.

### Phase B Seg-Aware Risk

`results/training_logs/51_train_segaware_hc18_phaseB_segaware_swinir_seed42_20260707_041015.csv` records only 8 epochs. Many validation fields are `nan`, but in the code these are largely placeholders for epochs where validation is skipped, not proof of numeric NaN divergence.

The real risk is that the discriminator loss is extremely large in early epochs and the run is incomplete. Phase B checkpoints exist in `results/models/phaseB_segaware_swinir_seed42/`, but Phase B is not registered in the final comparison, downstream utility, or significance tables. It should be described as incomplete/deferred.

### Missing Workflow Figure

`reports/final_table_figure_checklist.md` says `results/final_workflow_figure.png` is missing. `scripts/35_create_final_workflow_figure.py` should generate it. This is a submission TODO.

### Legacy/Orphan Outputs

`results/archived_unreliable_dice_iou/README.md` explicitly says early threshold-based Dice/IoU outputs are unreliable and not used. They should not be cited as final evidence.

`results/acouslic_ai/sweep_feedback_change_summary.csv` appears to be an older/poor blind reassessment branch where poor-frame ratio worsened. The current adaptive reassessment file is `results/acouslic_ai/adaptive_sweep_feedback_change_summary.csv`.

### Adaptive 20.10% vs 17.94%

The current 100-sweep heuristic result is 17.94%. The 20.10% number exists only in the older 50-sweep backup and old final ablation table. Do not mix these as if they are the same experiment.

### Unsupported 0.50 -> 0.21 Claim

I did not find a file supporting an exact poor-frame-ratio drop from 0.50 to 0.21. Current 100-sweep data support about 0.5469 -> 0.3675. The 50-sweep backup supports about 0.547 -> 0.346.

### Requirements Gap

`requirements.txt` does not list OpenCV, but many scripts import `cv2`. Add `opencv-python` or document the environment dependency before submission.

### OSEDiff/DRAT Scaffolding

Searches for `OSEDiff`, `DRAT`, `Tier D`, `oseddiff`, and `diffusion transformer` found no matching project scaffold in scripts, docs, reports, or READMEs. Any OSEDiff/DRAT discussion should be labelled future work unless new files are added.

### Clinical/Validation Limits

The project repeatedly states that labels are proxy labels, feedback is offline, there is no expert clinical validation, and this is not a real-time system. These limits are in `README.md`, `README_FULL.md`, `reports/final_conference_report.md`, and `reports/final_table_figure_checklist.md`.

## Submission TODO Checklist

- Choose one canonical results story: recommend SwinIR-Light Phase A as final reconstruction model.
- Keep older CNN/diffusion/adaptive results as baselines or historical ablations.
- Generate `results/final_workflow_figure.png` and `.pdf` with `scripts/35_create_final_workflow_figure.py`.
- Update `requirements.txt` to include OpenCV if the repo must be reproducible.
- Remove or clearly label archived unreliable Dice/IoU outputs.
- Decide whether to report current 17.94% adaptive improvement or legacy 20.10%; do not blend them.
- Do not claim 0.50 -> 0.21 poor-frame-ratio improvement unless a supporting result file is created.
- State Phase B as incomplete/deferred unless it is fully rerun and added to final tables.
- Keep cross-dataset transfer framed as proxy/zero-shot evidence, not clinical validation.

## Phase 6 - Teach-Me Summary

This project studies fetal ultrasound restoration under acquisition-aware quality control. The cleanest current story is: HC18 images are degraded into `LQ_mixed`, restored with SwinIR-Light Phase A, and evaluated both by reconstruction metrics and by downstream fetal-head segmentation utility.

The strongest reconstruction result is SwinIR-Light Phase A. Its main checkpoint is `results/models/swinir_light_phase_a_seed42/best_lpips.pth`. Its headline metrics are LPIPS 0.1009 and DISTS 0.1509 in `results/final_all_hc18_comparison_summary.csv`, with LPIPS CI around [0.0969, 0.1050] in `results/bootstrap_metric_summary.csv`.

The downstream utility story is conservative: SwinIR achieves Dice 0.8877, close to HR Dice 0.8897, in `results/downstream_utility_summary.csv`. This supports preservation of segmentation utility rather than dramatic anatomical improvement.

The ACOUSLIC side builds a proxy acquisition-quality system. It labels frames using blur, contrast, brightness, noise, edge, and motion cues; trains a small MLP; and uses feedback policies to decide whether to keep, enhance, denoise, or recommend repeat acquisition.

Current adaptive feedback results show 17.94% heuristic frame-level improvement and 17.24% learned improvement. A legacy 20.10% result also exists, but it belongs to an older 50-sweep backup and should not be mixed with the current 100-sweep result.

Cross-dataset transfer is a useful extra claim: an ACOUSLIC-trained proxy MLP applied zero-shot to HC18 finds a significant relationship between proxy quality change and LPIPS improvement. The main number is Spearman rho 0.5689 in `results/cross_dataset/cross_dataset_stats.json`.

The main caution is narrative cleanliness. Older reports centre CNN/diffusion results, while the newer self-audit centres SwinIR-Light. For a thesis or paper, make SwinIR-Light the canonical final model, retain older branches as baselines, and clearly mark Phase B segmentation-aware training as unfinished.
