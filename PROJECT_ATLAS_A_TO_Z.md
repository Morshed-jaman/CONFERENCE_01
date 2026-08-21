# QAFUE Project Atlas: A-to-Z Repository and Viva Walkthrough

**Project:** Quality-Aware Fetal Ultrasound Enhancement (QAFUE)  
**Audited root:** `D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project`  
**Audit date:** 2026-07-28  
**Mode:** documentation-only inventory; no training, inference, evaluation, or Git operation was performed.

This atlas explains how the repository turns raw fetal-ultrasound data into controlled HC18 super-resolution experiments, ACOUSLIC-AI proxy acquisition-quality experiments, frozen-U-Net downstream evaluation, diffusion-baseline comparisons, statistical tests, and paper artifacts. “Proxy” is important: ACOUSLIC quality labels are heuristic or learned from heuristic features, not expert clinical annotations.

## Table of contents

1. [Full inventory](#1-full-inventory)
2. [Top-level areas](#2-top-level-areas)
3. [Script map and true pipeline order](#3-script-map-and-true-pipeline-order)
4. [The complete A-to-Z process](#4-the-complete-a-to-z-process)
5. [Know every number: provenance table](#5-know-every-number-provenance-table)
6. [Glossary and viva cheat sheet](#6-glossary-and-viva-cheat-sheet)
7. [Open questions and stale-value note](#7-open-questions-and-stale-value-note)

---

# 1. Full inventory

## 1.1 Repository scale

| Path | Type | Files | Approximate bytes | Meaning |
|---|---|---:|---:|---|
| `.agents/` | directory | 0 | 0 | Empty agent-support directory. |
| `.git/` | directory | 968 | 176,978,189 | Version-control internals; existence recorded, contents intentionally not audited. |
| `.pycache_osediff_eval/` | directory | 37 | 905,060 | Python bytecode cache from OSEDiff evaluation compilation. |
| `data/` | directory | 17,442 | 36,954,067,521 | Raw and processed HC18/ACOUSLIC-AI data. |
| `docs/` | directory | 1 | 5,994 | Cross-dataset transfer documentation. |
| `reports/` | directory | 3 | 34,439 | Conference report and checklist documents. |
| `results/` | directory | 29,854 | 3,000,254,732 | Models, outputs, metrics, masks, logs, figures, and backups. |
| `scripts/` | directory | 104 including caches; 63 source files | 1,950,569 total | End-to-end experimental code. |

## 1.2 Clean collapsed tree

Folders with more than 20 files are collapsed as `<N files>`; representative filenames are shown for large image folders.

```text
Fetal_SR_Project/
├── .agents/                                  <0 files>
├── .git/                                     <968 internal files; not descended>
├── .pycache_osediff_eval/                    <37 bytecode files>
├── data/
│   ├── raw/HC18/
│   │   ├── training_set/training_set/        <1,999 files; e.g. 100_2HC.png, 100_2HC_Annotation.png>
│   │   ├── test_set/test_set/                <336 files; e.g. 100_HC.png, 101_HC.png>
│   │   ├── training_set.zip
│   │   └── test_set.zip
│   └── processed/
│       ├── HC18_SR/
│       │   ├── HR/                           <999 PNGs; hc18_0000.png …>
│       │   ├── masks/                        <999 PNGs; hc18_0000.png …>
│       │   ├── LR_x2/                        <999 PNGs>
│       │   ├── LR_x4/                        <999 PNGs>
│       │   ├── LQ_mixed/                     <999 PNGs>
│       │   ├── train.csv                     <699 rows>
│       │   ├── val.csv                       <150 rows>
│       │   ├── test.csv                      <150 rows>
│       │   ├── hc18_manifest.csv             <999 rows>
│       │   └── hc18_lq_manifest.csv          <999 rows>
│       └── ACOUSLIC_AI/
│           ├── temp_mha/                     <100 MHA volumes; 35.12 GB>
│           ├── frames/                       <100 sweep folders × 100 PNGs = 10,000 frames>
│           ├── acouslic_frame_manifest.csv   <10,000 rows>
│           ├── acouslic_proxy_quality_manifest.csv <10,000 rows>
│           └── acouslic_sweep_quality_summary.csv  <100 rows>
├── docs/
│   └── CROSS_DATASET_TRANSFER.md
├── reports/
│   ├── final_conference_report.md
│   ├── final_conference_report_appendix.md
│   └── final_table_figure_checklist.md
├── results/
│   ├── acouslic_ai/                          <15,318 files>
│   │   ├── adaptive_enhanced_frames/         <10,000 PNGs>
│   │   ├── enhanced_frames_quality_aware_l1_mse/ <5,000 PNGs>
│   │   ├── feedback_report/                  <121 files>
│   │   ├── proxy_quality_visuals/            <20 figures>
│   │   ├── manual_review/                    <6 files>
│   │   └── CSV/Markdown summaries for blind, heuristic, learned, and sweep feedback
│   ├── backups/acouslic_ai_50sweeps/         <10,194 files; historical snapshot>
│   ├── eval/
│   │   ├── swinir_light_phase_a/test/        <155 files>
│   │   ├── resshift/test/
│   │   │   └── restored/                     <150 PNGs; hc18_0007.png, hc18_0009.png, hc18_0010.png>
│   │   └── osediff/test/
│   │       └── restored/                     <150 PNGs; hc18_0007.png, hc18_0009.png, hc18_0010.png>
│   ├── downstream_utility_masks/             <1,480 masks: 148 × 10 methods>
│   ├── models/                               <31 checkpoint/metadata files>
│   ├── sr_outputs/                           <2,059 restored PNGs across legacy methods>
│   ├── logs/                                 <38 files>
│   ├── training_logs/                        <30 CSVs>
│   ├── sanity/                               <18 OSEDiff smoke-test artifacts>
│   ├── figures/                              final_workflow_figure.png
│   ├── downstream_hard_subset_figures/       dice_hard_vs_full.png
│   ├── final_all_visual_comparison/          <10 figures>
│   ├── report_assets/                        <10 files>
│   ├── archived_unreliable_dice_iou/         <16 archived files; not canonical>
│   ├── RESULTS_MANIFEST.md
│   ├── final_all_hc18_comparison_{per_image,summary}.csv
│   ├── downstream_utility_{per_image,summary}.csv
│   ├── downstream_hard_subset_{membership,summary}.csv
│   ├── significance_report.csv
│   └── <other top-level metric, ablation, bootstrap, and legacy CSV/MD files>
├── scripts/
│   ├── 01_prepare_hc18.py … 61_cross_dataset_quality_analysis.py
│   ├── 99_cleanup_unnecessary_data.py
│   ├── hc18_dataset.py
│   ├── hc18_diffusion_dataset.py
│   ├── metrics_utils.py
│   ├── run_short_compare.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── swinir_light.py
│   ├── resshift/
│   │   ├── _torchvision_compat.py
│   │   └── run_resshift_on_hc18.py
│   └── osediff/
│       ├── _run_test_osediff.py
│       ├── download_osediff_weights.py
│       ├── run_osediff_on_hc18.py
│       └── run_osediff_batch_orchestrator.ps1
├── .gitignore
├── requirements.txt
├── README.md
├── README_FULL.md
├── PROJECT_DESCRIPTION_FOR_BEGINNERS.md
├── WORK_ATLAS.md
├── THESIS_VIVA_DOSSIER.md
├── DEMO_REPORTS.zip
└── PROJECT_ATLAS_A_TO_Z.md                  (this document)
```

## 1.3 Root result files

The 52 direct files in `results/` fall into these groups:

- **Canonical reconstruction:** `final_all_hc18_comparison_per_image.csv`, `final_all_hc18_comparison_summary.csv`, `final_all_hc18_comparison_table.md`, `final_all_hc18_dists_per_image.csv`.
- **Canonical downstream utility:** `downstream_utility_per_image.csv`, `downstream_utility_summary.csv`, `downstream_hard_subset_membership.csv`, `downstream_hard_subset_summary.csv`.
- **Statistics:** `significance_report.csv`, `bootstrap_metric_summary.csv`, `bootstrap_confidence_intervals.csv`, `bootstrap_confidence_intervals.md`.
- **Earlier CNN-only comparison:** `final_hc18_comparison_*`.
- **Legacy diffusion evaluations:** `diffusion_sr_*`, `quality_aware_diffusion_sr_*`, `quality_aware_diffusion_sr_v2_*`, `dmuisr_inspired_*`, `diffusion_sr_x0loss_*`, `cnn_residual_diffusion_*`.
- **Segmentation/anatomy:** `hc18_unet_improved_*`, `anatomy_preservation_*`.
- **Ablations:** `final_ablation_study.*`, `final_ablation_*`, `final_ablation_tables.md`.
- **Paper support:** `novelty_comparison_table.md`, `RESULTS_MANIFEST.md`, `short_sr_perceptual_comparison*`, `phaseB_lq_dice_weights.csv`.

---

# 2. Top-level areas

## 2.1 `scripts/`

**What it is:** all executable research logic and reusable model/data/metric modules.  
**Why it exists:** it makes each research stage reproducible and connects inputs to saved artifacts.  
**Inside:** HC18 preparation/training/evaluation (`01–22`), ACOUSLIC processing (`23–36`), segmentation and canonical evaluation (`39–51`), cross-dataset evaluation (`60–61`), baseline adapters, and reusable modules.  
**Dependencies:** reads `data/`; loads `results/models/`; writes `results/`; uses packages in `requirements.txt`. Baseline wrappers also refer to a sibling `CONFERENCE\baselines` tree that is outside this audit root.

## 2.2 `data/`

**HC18:** 999 processed fetal-head ultrasound images, masks, and three degradation variants. HR means the clean reference grid. LR_x2/LR_x4 are bicubic downsample-and-restore variants. LQ_mixed adds blur, speckle noise, motion blur, and contrast reduction. The split is 699 train, 150 validation, and 150 test.

**ACOUSLIC-AI:** 100 sampled MHA sweeps and 10,000 extracted PNG frames. It has no paired HR target in this workflow. It supports proxy quality/acquisition feedback, not full-reference SR claims or clinical validation.

**Dependencies:** preparation scripts create it; loaders and all model/evaluation scripts consume it.

## 2.3 `results/`

**What it is:** the evidence layer.  
**Why it exists:** checkpoints, reconstructions, masks, per-image measurements, summary tables, significance tests, logs, and paper figures must be separable from source code.  
**Canonical reading order:** `RESULTS_MANIFEST.md` → `final_all_hc18_comparison_summary.csv` → `downstream_utility_summary.csv` → `downstream_hard_subset_summary.csv` → `significance_report.csv`.

`results/eval/<method>/test/` standardizes a method’s `restored/`, `recon_per_image.csv`, `recon_summary.csv`, `downstream_per_image.csv`, `downstream_summary.csv`, and `hard_subsets.csv`. `results/sr_outputs/` contains older method-specific output conventions. `results/backups/` and `archived_unreliable_dice_iou/` are provenance, not canonical headline sources.

## 2.4 `results/models/`

| Group | Files/purpose |
|---|---|
| CNN SR | `sr_only_hc18_best.pth`, `quality_aware_sr_hc18*.pth`, perceptual/GAN best/final checkpoints |
| Legacy diffusion | five ~40–45 MB DDPM/residual checkpoints |
| Hero model | `swinir_light_phase_a_seed42/{best_lpips,best_dice,last,ema}.pth` |
| Phase B | `phaseB_segaware_swinir_seed42/` checkpoints; incomplete/non-headline branch |
| Segmentation | `hc18_unet_segmentation_best_improved.pth` |
| Proxy quality | `proxy_quality_embedding_mlp.pth` |

## 2.5 Baselines

There is **no `baselines/` directory inside the permitted project root**. The wrapper code refers to sibling paths such as `Fetal_SR_Project.parent / "baselines" / "repos" / "ResShift"` and equivalent OSEDiff paths. From in-root evidence:

- `scripts/resshift/` adapts a sibling ResShift clone and its checkpoints/configuration.
- `scripts/osediff/` adapts a sibling OSEDiff clone, isolated Python, Stable Diffusion 2.1, RAM/DAPE, and OSEDiff weights.
- `results/eval/{resshift,osediff}/` proves 150 HC18 outputs and evaluations exist.
- `results/logs/` records environment, commands, OOM fallback, tiling, and metrics.

The sibling `repos/`, `venvs/`, and `weights/` contents could not be inventoried without leaving the explicitly permitted root.

## 2.6 Documentation and configuration

- `README.md`: concise original workflow.
- `README_FULL.md`: audited environment, results, limitations, and expanded commands.
- `PROJECT_DESCRIPTION_FOR_BEGINNERS.md`: plain-language orientation.
- `WORK_ATLAS.md`: prior repository audit and risk map.
- `THESIS_VIVA_DOSSIER.md`: defense questions, evidence, and limitations.
- `docs/CROSS_DATASET_TRANSFER.md`: frozen ACOUSLIC-to-HC18 transfer results.
- `reports/*.md`: manuscript, appendix, and table/figure checklist.
- `requirements.txt`: unpinned main dependencies plus pinned `omegaconf`, `loguru`, and ANTLR runtime. It includes OpenCV.
- `.gitignore`: exclusion policy.
- `DEMO_REPORTS.zip`: small archive of report Markdown files.
- `.git/`: version history; not part of the experimental pipeline.

---

# 3. Script map and true pipeline order

## 3.1 Every script: purpose, inputs, outputs, and important parameters

### HC18 preparation, CNNs, and legacy diffusion

| Script | Purpose | Main inputs → outputs | Key parameters |
|---|---|---|---|
| `01_prepare_hc18.py` | Normalize raw HC18 and assign splits. | raw training images/masks → `HR/`, `masks/`, manifests/split CSVs | 256×256; deterministic split logic in file |
| `02_generate_lq_hc18.py` | Create paired degradation variants. | HR manifest/images → `LR_x2/`, `LR_x4/`, `LQ_mixed/`, LQ manifest | target 256; x2/x4 bicubic; blur/noise/motion/contrast |
| `03_visual_check_hc18.py` | Sanity-check degradation visually. | LQ manifest/images/masks → five figures | random sample selection in script |
| `04_eval_bicubic_hc18.py` | Measure degraded baselines on test split. | HR and LR/LQ → bicubic per-image/summary CSVs | PSNR, SSIM, MAE |
| `05_train_sr_only_hc18.py` | Train residual CNN. | train/val LQ→HR → best checkpoint/log | 30 epochs, batch 8, LR 1e-4 |
| `06_eval_sr_only_hc18.py` | Evaluate residual CNN. | checkpoint + test split → PNGs, metrics, figures | fixed test evaluation |
| `07_train_quality_aware_sr_hc18.py` | Train FiLM-like quality-aware CNN. | LQ/HR + five quality features → checkpoint/log | 30 epochs, batch 16, LR 1e-4 |
| `08_compare_hc18_models.py` | Compare LQ and early CNN variants. | CNN checkpoints/test data → early comparison CSV/table/figures | imported matching architectures |
| `09_anatomy_preservation_eval_hc18.py` | Legacy mask-based anatomy analysis. | masks + available SR outputs → anatomy CSV/visuals | skips missing method images |
| `10_make_hc18_report_assets.py` | Format early result tables/figures. | early summary CSVs → `report_assets/` | method-label normalization |
| `11_train_diffusion_sr_hc18.py` | Train conditional DDPM. | diffusion dataset → checkpoint/log | 50 epochs, batch 8, LR 2e-4, 1,000 steps, 128 px |
| `12_eval_diffusion_sr_hc18.py` | Sample/evaluate conditional DDPM. | checkpoint/test LQ → outputs/metrics | 50 sampling steps |
| `13_train_quality_aware_diffusion_sr_hc18.py` | Add five-feature conditioning to DDPM. | LQ/HR/features → checkpoint | quality scale 0.1 |
| `14_eval_quality_aware_diffusion_sr_hc18.py` | Evaluate quality-aware DDPM. | selected checkpoint → outputs/CSV | 50 steps; configurable output name |
| `15_compare_all_hc18_models.py` | Older all-output comparison. | saved PNG directories → final comparison artifacts | skips missing files |
| `16_train_dmuisr_inspired_diffusion_sr_hc18.py` | Train practical DMUISR-inspired model, not exact reproduction. | LQ/HR → checkpoint/log | 100 epochs; x0 0.1; edge 0.05 |
| `17_eval_dmuisr_inspired_diffusion_sr_hc18.py` | Evaluate that model. | checkpoint/test set → PNGs/metrics | 50 sampling steps |
| `18_train_diffusion_sr_x0loss_hc18.py` | Add x0 reconstruction loss. | LQ/HR → checkpoint | x0 weight 0.1 |
| `19_eval_diffusion_sr_x0loss_hc18.py` | Evaluate x0-loss model. | checkpoint → partial outputs/metrics | 50 steps |
| `20_train_cnn_residual_diffusion_sr_hc18.py` | Diffuse only the residual after CNN guidance. | CNN guide + HR → checkpoint | 50 epochs, batch 8, 1,000 steps |
| `21_eval_cnn_residual_diffusion_sr_hc18.py` | Recombine sampled residual with CNN guide. | checkpoint + guide → 150 PNGs/metrics | 50 steps |
| `22_generate_cnn_guides_all_splits_hc18.py` | Generate CNN guides for all splits. | quality-aware CNN checkpoint + all LQ → 999 guide PNGs | batch 16 |

### ACOUSLIC-AI and acquisition-quality branch

| Script | Purpose | Main inputs → outputs | Key parameters |
|---|---|---|---|
| `23_inspect_acouslic_zip.py` | Read ZIP metadata without extraction. | ACOUSLIC ZIP → inspection CSV/summary | metadata-only |
| `24_extract_acouslic_mha_sample.py` | Sample MHA sweeps into PNG frames. | ZIP → `temp_mha/`, `frames/`, frame manifest | max 5 by default; interval 5; max 100 slices |
| `24_compute_acouslic_proxy_quality.py` | Extract heuristic frame/motion features and labels. | frame manifest/PNGs → proxy manifest, sweep summary, visuals | feature size 256 |
| `25_generate_acouslic_feedback_report.py` | Turn proxy labels into readable sweep feedback. | proxy frame/sweep CSVs → report, timelines, examples | max 10 examples |
| `26_feedback_aware_enhancement_analysis.py` | Blind CNN enhancement/no-reference analysis. | proxy manifest + CNN checkpoint → 5,000 enhanced frames and summary | 256 px; no HR metrics |
| `27_sweep_feedback_change_analysis.py` | Reassess sweep feedback after adaptive enhancement. | adaptive per-frame + original sweep data → per-sweep/summary/figures | poor-ratio threshold 0.4 |
| `28_adaptive_feedback_aware_enhancement.py` | Choose keep/enhance/denoise/reacquire by heuristic label. | proxy data + CNN checkpoint → 10,000 adaptive outputs and summaries | label-driven policy |
| `29_train_proxy_quality_embedding.py` | Train feature MLP on proxy labels. | proxy manifest → MLP, predictions, metrics, report | 50 epochs, batch 128, LR 1e-3, seed 42 |
| `30_learned_adaptive_feedback_enhancement.py` | Use MLP confidence to choose conservative actions. | MLP + CNN + proxy frames → learned summaries/figures | confidence 0.6 |
| `31_final_ablation_study.py` | Aggregate ACOUSLIC and HC18 ablations. | enhancement/sweep/HC18 summaries → CSV/MD/figures | source-backed aggregation |
| `32_bootstrap_confidence_intervals.py` | Bootstrap benchmark/proxy metrics. | per-image/per-frame/per-sweep CSVs → CI CSV/MD/figures | 1,000 resamples, seed 42 |
| `33_create_proxy_label_manual_review_sheet.py` | Make a human sanity-review sheet. | proxy manifest → review CSV/contact sheets | 25 per label, seed 42 |
| `35_create_final_workflow_figure.py` | Draw final two-row workflow. | code-defined layout → workflow PNG | fixed publication layout |
| `36_create_final_ablation_tables.py` | Create paper-ready ablation tables. | result summaries → three CSVs + Markdown | contains stale generated values; not canonical |

### Segmentation, canonical model, statistics, and transfer

| Script | Purpose | Main inputs → outputs | Key parameters |
|---|---|---|---|
| `39_improve_hc18_unet_segmentation.py` | Validate mask pairing, train improved U-Net, evaluate anatomy. | HR/masks/splits → U-Net checkpoint and metrics | seed 42; 100 epochs; threshold search 0.2–0.7 |
| `40_cleanup_unreliable_dice_results.py` | Archive known-unreliable legacy segmentation results. | listed paths → archive | dry-run option; historical maintenance |
| `40_train_sr_perceptual_hc18.py` | Train CNNs with L1/MSE + LPIPS + optional PatchGAN. | LQ/HR → checkpoints/logs | 40 epochs; seed 42; LPIPS 0.1; GAN 0.05 defaults |
| `41_emergency_debug_hc18_mask_pairing_and_unet.py` | Overfit 20 samples to diagnose masks/U-Net. | HR/masks → debug artifacts only | 500 max debug epochs, batch 4 |
| `41_metrics_utils.py` | Compatibility import shim. | imports canonical `metrics_utils.py` | no independent computation |
| `42_compare_all_models.py` | Canonical uniform reconstruction harness. | test data, checkpoints, ResShift/OSEDiff PNGs → final comparison + per-method eval CSVs | PSNR, SSIM, LPIPS, DISTS, MAE, EdgeScore, NIQE, BRISQUE |
| `46_downstream_utility_eval.py` | Frozen-U-Net utility evaluation. | reconstructions + U-Net + masks → Dice/IoU/error summaries and masks | threshold 0.50 loaded from checkpoint; 148 valid masks |
| `47_hard_subset_analysis.py` | Evaluate Dice on difficult subsets. | downstream/reconstruction per-image CSVs → membership, summary, figure | bottom quartile, Dice <0.85, top LPIPS quartile |
| `48_significance_report.py` | Bootstrap CIs and paired Wilcoxon tests. | canonical per-image tables + eval auto-discovery → bootstrap/significance CSVs | seed 42; max 1,000 bootstrap; 47 tests |
| `49_phase_a_diagnosis.py` | Diagnose partial SwinIR training without training. | training logs → diagnosis outputs | LPIPS gate from script |
| `50_train_swinir_hc18.py` | Train hero SwinIR-Light Phase A. | HC18 train/val/test + U-Net → checkpoints, logs, recon and Dice outputs | 100 epochs; batch 8; seed 42; L1 1.0, LPIPS 0.2, GAN 0.005, EMA 0.999 |
| `51_train_segaware_hc18.py` | Prepared Phase B segmentation-aware training. | Phase A backbone + masks/hard weights → partial checkpoints/log | 120 epochs, seed 42; incomplete/non-headline |
| `60_apply_acouslic_mlp_to_hc18.py` | Apply frozen ACOUSLIC MLP zero-shot to HC18. | MLP + HC18 method outputs → cross-dataset CSVs | seed 42; no HC18 fine-tuning |
| `61_cross_dataset_quality_analysis.py` | Analyze method rankings and transfer association. | cross-dataset per-image/summary → ranking, stats JSON, docs | seed 42; 1,000 bootstrap |
| `99_cleanup_unnecessary_data.py` | Find optional cleanup candidates. | project tree → dry-run report or confirmed deletion | dry-run default; destructive mode guarded |

### Reusable and baseline support

| Script | Role |
|---|---|
| `hc18_dataset.py` | Standard PyTorch HR/LQ/mask loader. |
| `hc18_diffusion_dataset.py` | Conditional diffusion dataset and quality features. |
| `metrics_utils.py` | Canonical metric implementations and pyiqa wrappers. |
| `models/swinir_light.py` | Compact same-grid grayscale SwinIR-style residual backbone. |
| `models/__init__.py` | Model-package marker. |
| `run_short_compare.py` | Short controlled L1/MSE vs LPIPS+GAN acceptance experiment; seed 42. |
| `resshift/_torchvision_compat.py` | Compatibility shim for old torchvision import paths. |
| `resshift/run_resshift_on_hc18.py` | Zero-shot ResShift wrapper; seed 12345, 15 steps, x4, chop 256→128 fallback. |
| `osediff/_run_test_osediff.py` | Launch upstream OSEDiff through compatibility setup. |
| `osediff/download_osediff_weights.py` | Download README-documented OSEDiff dependencies. |
| `osediff/run_osediff_on_hc18.py` | Offline HC18 wrapper; x4, fp16, chop choices 64/128/256, skip-existing and watchdog support. |
| `osediff/run_osediff_batch_orchestrator.ps1` | Resume, chunk, timeout, suspect-file retry, and logging controller. |

## 3.2 True dependency order

```text
Raw HC18
  → 01 prepare HR/masks/splits
  → 02 generate LR/LQ + paired manifest
  ├→ 04 bicubic baseline
  ├→ 05/07/40 CNN training
  ├→ 11–21 legacy diffusion training/evaluation
  ├→ 39 improved U-Net
  └→ 50 SwinIR-Light Phase A

HC18 LQ test + external baseline clones/weights
  ├→ ResShift wrapper → results/eval/resshift/test/restored
  └→ OSEDiff wrapper → results/eval/osediff/test/restored

All registered reconstructions
  → 42 uniform reconstruction metrics
  → 46 frozen-U-Net downstream utility
  → 47 hard subsets
  → 48 paired significance + Bonferroni
  → 35/report scripts → paper tables and figures

ACOUSLIC ZIP/MHA
  → 23 inspect → 24 extract frames → 24 proxy features/labels
  ├→ 25 feedback report
  ├→ 26 blind enhancement
  ├→ 28 heuristic adaptive enhancement → 27 sweep reassessment
  └→ 29 proxy MLP → 30 learned adaptive enhancement
       → 31/32/36 ablations and intervals
       → 60 frozen transfer to HC18 → 61 ranking/Spearman analysis
```

---

# 4. The complete A-to-Z process

## A. Environment setup

The audited documentation records Windows 10/PowerShell, Python `3.13.1`, PyTorch `2.12.1+cu126`, CUDA runtime `12.6`, and an RTX 4060 Laptop GPU with 8 GB VRAM. This is documented in `README_FULL.md` and `results/logs/resshift_reproduction_report.md`. `requirements.txt` lists the main scientific, image, PyTorch, perceptual-metric, and configuration packages.

The main environment is protected because changing its PyTorch/CUDA stack could invalidate trained checkpoints. ResShift and OSEDiff have older or conflicting dependency expectations, so wrapper scripts use compatibility shims and, for OSEDiff, an isolated sibling environment. The actual sibling environment directories are outside the allowed root and were not inventoried.

**Success check:** imports and versions are documented in reproduction logs; do not infer versions solely from unpinned requirements.

## B. Data preparation

1. Place raw HC18 archives/images under `data/raw/HC18`.
2. `01_prepare_hc18.py` creates 999 normalized HR images and masks and writes 699/150/150 train/val/test splits.
3. `02_generate_lq_hc18.py` creates LR_x2, LR_x4, and mixed-degradation paired inputs.
4. Verify all five processed image directories contain 999 files and the LQ manifest has 999 rows.
5. For ACOUSLIC, inspect the ZIP, sample MHA stacks, and create 10,000 frames from 100 sweeps.
6. Compute proxy quality features/labels. These are not diagnostic truth.

## C. Model training

The early ablations establish what quality conditioning and perceptual/adversarial losses do:

- SR-only residual CNN (`05`).
- Quality-aware FiLM-like CNN (`07`).
- Perceptual/GAN variants (`40`).
- Several local conditional-diffusion explorations (`11–21`).

The hero is **SwinIR-Light Phase A** (`50`): a same-resolution residual transformer trained with L1 + LPIPS + a small GAN term, deterministic seed 42, EMA, OOM-aware batch fallback, and periodic frozen-U-Net Dice evaluation. The canonical checkpoint is `results/models/swinir_light_phase_a_seed42/best_lpips.pth`.

Phase B (`51`) is incomplete and must not replace Phase A headline results.

## D. Reconstruction evaluation

`42_compare_all_models.py` is the current uniform harness. It loads the 150-test-image HC18 set and evaluates LQ, four CNN configurations, SwinIR, ResShift, and OSEDiff through the same functions. It writes:

- per-image values to `results/final_all_hc18_comparison_per_image.csv`;
- averages to `results/final_all_hc18_comparison_summary.csv`;
- method-specific eval CSVs under `results/eval/*/test/`.

Full-reference metrics need HR: PSNR, SSIM, LPIPS, DISTS, MAE, and EdgeScore. NIQE and BRISQUE are no-reference metrics but are still computed uniformly.

## E. Downstream utility

The improved U-Net is trained independently on HR/masks. `46` freezes it and asks: “If this reconstruction is fed to the anatomy model, is fetal-head segmentation preserved?” Two empty/unreadable masks are excluded, leaving `n=148`. Dice is the headline overlap measure.

`47` then evaluates full test plus difficult subsets: bottom quartile by LQ Dice, LQ Dice below 0.85, and top quartile by LQ LPIPS.

## F. ResShift and OSEDiff

Both are **zero-shot**: their published natural-image models are not fine-tuned on HC18.

ResShift uses 15 steps, x4 restoration, deterministic seed 12345, and chopping. Its wrapper falls from 256 to 128 tiles after OOM. OSEDiff uses one-step diffusion-style restoration with fp16, tiling, offline model paths, resumable manifests, watchdogs, and suspect-file retries. OSEDiff completed 150/150 at the stable 128 tile configuration. Their restored PNGs are protected inputs to the metric harness.

## G. Cross-dataset transfer

The feature MLP is trained on ACOUSLIC heuristic proxy labels, frozen, and applied to HC18 outputs without HC18 fine-tuning (`60`). `61` tests whether restoration-induced increases in predicted “good” probability align with LPIPS improvement. This supports dataset-external image-quality evidence, not clinical validation.

## H. Significance

`48_significance_report.py` aligns methods by image ID, computes paired differences, performs Wilcoxon signed-rank tests, and applies Bonferroni correction. Bonferroni controls family-wise false positives by multiplying each raw p-value by the number of planned tests. With automatically discovered ResShift and OSEDiff comparisons, `results/significance_report.csv` contains **47 tests**, so the current denominator is 47.

## I. Figures

`35_create_final_workflow_figure.py` creates the main two-row QAFUE workflow. Other scripts create visual degradation checks, qualitative reconstruction panels, bootstrap plots, U-Net figures, hard-subset Dice plots, ACOUSLIC timelines, and adaptive-action figures. Figures explain behavior; CSVs remain the numerical source of truth.

## J. Canonical-number verification

Read numbers, do not recalculate or copy them from prose:

1. Reconstruction: `results/final_all_hc18_comparison_summary.csv`.
2. Downstream Dice: `results/downstream_utility_summary.csv`.
3. Hard subsets: `results/downstream_hard_subset_summary.csv`.
4. Significance: `results/significance_report.csv`.
5. ACOUSLIC adaptive rates: the three `results/acouslic_ai/*enhancement_summary.csv` files.
6. Cross-dataset rho: `results/cross_dataset/cross_dataset_stats.json`.

---

# 5. Know every number: provenance table

Values below are copied exactly from current on-disk rows; no statistic was recomputed.

## 5.1 Reconstruction and downstream headline values

| Claim | File and row/column | Exact on-disk value | Status |
|---|---|---:|---|
| SwinIR PSNR | `final_all_hc18_comparison_summary.csv`, `SwinIR-Light Phase A`, `PSNR` | 29.72770638946017 | VERIFIED-ON-DISK |
| SwinIR SSIM | same row, `SSIM` | 0.8462304392579071 | VERIFIED-ON-DISK |
| SwinIR LPIPS | same row, `LPIPS` | 0.10091456212103367 | VERIFIED-ON-DISK |
| SwinIR DISTS | same row, `DISTS` | 0.15086822072664896 | VERIFIED-ON-DISK |
| SwinIR Dice (`n=148`) | `downstream_utility_summary.csv`, SwinIR, `Dice_mean` | 0.8877019882187267 | VERIFIED-ON-DISK |
| HR Dice ceiling | downstream summary, HR | 0.8896568280296212 | VERIFIED-ON-DISK |
| LQ Dice floor/reference | downstream summary, LQ_mixed | 0.8849903062929627 | VERIFIED-ON-DISK |
| SR-only LPIPS / DISTS / PSNR / SSIM | reconstruction summary, SR-only | 0.1749419870475928 / 0.17408846179644266 / 30.98564898087667 / 0.8564114944492922 | VERIFIED-ON-DISK |
| SR-only Dice | downstream summary | 0.8853266344841867 | VERIFIED-ON-DISK |
| Quality-aware LPIPS / DISTS / PSNR / SSIM | reconstruction summary, Quality-aware SR | 0.2042164137462775 / 0.1811862627665202 / 30.58041972571543 / 0.8743366800436309 | VERIFIED-ON-DISK |
| Quality-aware Dice | downstream summary | 0.8846884138798635 | VERIFIED-ON-DISK |
| SR-only+perc+GAN LPIPS / DISTS | reconstruction summary | 0.12373436028758684 / 0.17729870637257894 | VERIFIED-ON-DISK |
| SR-only+perc+GAN Dice | downstream summary | 0.8872863596911454 | VERIFIED-ON-DISK |
| Quality-aware+perc+GAN LPIPS / DISTS | reconstruction summary | 0.1130732424557209 / 0.1671081018447876 | VERIFIED-ON-DISK |
| Quality-aware+perc+GAN Dice | downstream summary | 0.8858172433066724 | VERIFIED-ON-DISK |
| ResShift LPIPS / DISTS | reconstruction summary, ResShift | 0.22272943139076232 / 0.20935875455538433 | VERIFIED-ON-DISK |
| ResShift PSNR / SSIM | same row | 26.571638913546884 / 0.7260506192561473 | VERIFIED-ON-DISK |
| ResShift Dice | downstream summary | 0.8849377099976218 | VERIFIED-ON-DISK |
| OSEDiff LPIPS / DISTS | reconstruction summary, OSEDiff | 0.37836109936237333 / 0.27610760847727456 | VERIFIED-ON-DISK |
| OSEDiff PSNR / SSIM | same row | 23.536646476366073 / 0.5739508913332453 | VERIFIED-ON-DISK |
| OSEDiff Dice | downstream summary | 0.875459653043834 | VERIFIED-ON-DISK |
| Bonferroni denominator | `significance_report.csv`, data-row count | 47 | VERIFIED-ON-DISK |

## 5.2 Acquisition-quality and transfer values

| Claim | File and row/column | Exact on-disk value | Status |
|---|---|---:|---|
| Heuristic-adaptive improvement | `results/acouslic_ai/adaptive_feedback_enhancement_summary.csv`, overall `percentage_improved` | 17.94000000% | VERIFIED-ON-DISK |
| Learned-adaptive improvement | `results/acouslic_ai/learned_adaptive_enhancement_summary.csv`, overall | 17.24000000% | VERIFIED-ON-DISK |
| Blind improvement | `results/acouslic_ai/feedback_aware_enhancement_summary.csv`, overall | 8.00000000% | VERIFIED-ON-DISK |
| Sweeps with reduced poor ratio | `adaptive_sweep_feedback_change_summary.csv` | 100 of 100 | VERIFIED-ON-DISK |
| Mean poor-ratio reduction | same file | 0.17940000 | VERIFIED-ON-DISK |
| Changed repeat-sweep cases | same file | 36 | VERIFIED-ON-DISK |
| Locked poor-frame ratio before/after pair | exact aggregate source requested by prompt | not found on disk | NOT-FOUND |
| Cross-dataset Spearman rho | `results/cross_dataset/cross_dataset_stats.json`, `spearman_delta_prob_good_vs_delta_lpips.rho` | 0.5688522618064861 | VERIFIED-ON-DISK |
| Spearman 95% CI | `docs/CROSS_DATASET_TRANSFER.md` | [0.455751, 0.673784] | VERIFIED-ON-DISK |
| Spearman p-value | same document | 3.095e-14 | VERIFIED-ON-DISK |

The locked explanatory pair supplied for the atlas is `0.5469 → 0.3675`, but an exact aggregate row containing that pair was not found. It must not be presented as verified until a source CSV/log is identified. The verified sweep summary instead records mean reduction `0.17940000`.

---

# 6. Glossary and viva cheat sheet

## 6.1 Glossary

| Term | Plain-language meaning |
|---|---|
| QAFUE | Quality-Aware Fetal Ultrasound Enhancement; the overall framework. |
| HC18 | Paired fetal-head ultrasound benchmark with images and head masks. |
| ACOUSLIC-AI | Ultrasound sweep dataset used here for proxy acquisition-quality analysis. |
| HR | High-resolution/clean reference image. |
| LR/LQ | Low-resolution or synthetically low-quality input. |
| SR | Super-resolution/restoration. |
| PSNR | Pixel-fidelity score in dB; higher is better. |
| SSIM | Structural similarity; higher is better. |
| MAE | Mean absolute pixel error; lower is better. |
| LPIPS | Learned perceptual distance; lower is better. |
| DISTS | Learned structure-and-texture distance; lower is better. |
| NIQE | No-reference natural-image quality estimate; lower is conventionally better, but ultrasound domain mismatch matters. |
| BRISQUE | No-reference spatial quality score; lower is conventionally better, with the same domain caveat. |
| EdgeScore | Edge-map similarity used for boundary preservation; higher is better. |
| Dice | Overlap between predicted and ground-truth masks; 1 is perfect. |
| IoU | Intersection over union, another segmentation-overlap metric. |
| Perceptual loss | Training loss based on deep feature similarity rather than only pixels. |
| GAN/PatchGAN | Adversarial discriminator encouraging locally realistic textures. |
| perc+GAN | Reconstruction plus perceptual and adversarial losses. |
| FiLM | Feature-wise linear modulation; quality features scale/shift model activations. |
| SwinIR | Image-restoration transformer built from shifted-window attention. |
| EMA | Exponential moving average of model weights for stable evaluation. |
| DDPM | Denoising diffusion probabilistic model. |
| x0 loss | Direct loss on the reconstructed clean image estimate. |
| Zero-shot | Applied without training or fine-tuning on HC18. |
| Proxy label | Algorithm-derived label used for research feedback, not expert clinical truth. |
| Proxy-quality MLP | Small neural network trained on extracted quality features/heuristic labels. |
| Frozen model | Weights are held fixed during evaluation. |
| Downstream utility | Whether restoration helps/preserves a later task such as segmentation. |
| Hard subset | Deliberately difficult test images selected by predeclared criteria. |
| Wilcoxon signed-rank | Nonparametric paired test comparing per-image differences. |
| Bonferroni | Multiple-testing correction; here the denominator is 47. |
| Bootstrap CI | Confidence interval obtained by repeated resampling. |
| Spearman rho | Rank-correlation coefficient; detects monotonic association. |
| Tiling/chopping | Splitting an image into smaller regions to reduce memory use. |
| OOM | Out of memory. |
| ResShift | Multi-step diffusion-based SR baseline. |
| OSEDiff | One-step diffusion-based SR baseline. |
| DAPE/RAM | OSEDiff semantic/prompt-related auxiliary components. |

## 6.2 “If a reviewer points at folder X”

| Folder/file | One-line defense |
|---|---|
| `data/raw/HC18` | “This preserves the source benchmark before any project preprocessing.” |
| `data/processed/HC18_SR` | “This is the paired 999-image experiment set with reproducible splits and degradations.” |
| `data/processed/ACOUSLIC_AI` | “These are sampled sweeps/frames for proxy acquisition-quality experiments, not paired HR targets.” |
| `scripts/` | “Each numbered stage is explicit code connecting data, models, metrics, and paper artifacts.” |
| `scripts/models/` | “This contains the hero SwinIR-Light architecture separate from orchestration.” |
| `scripts/resshift/` | “This safely adapts an external zero-shot diffusion baseline to HC18.” |
| `scripts/osediff/` | “This provides resumable, tiled, offline OSEDiff inference with failure recovery.” |
| `results/models/` | “These are learned weights; the canonical hero checkpoint is Phase A best-LPIPS.” |
| `results/eval/` | “This is the standardized evidence layout for SwinIR, ResShift, and OSEDiff.” |
| `results/sr_outputs/` | “These are older/local method reconstructions feeding comparison and anatomy analyses.” |
| `results/downstream_utility_masks/` | “These are frozen-U-Net predictions used to calculate the 148-case Dice results.” |
| `results/logs/` | “These preserve reproducibility, environment, OOM, and baseline-run evidence.” |
| `results/backups/` | “Historical evidence only; not the canonical source for current claims.” |
| `archived_unreliable_dice_iou/` | “Known unreliable early anatomy results are quarantined so they cannot be mistaken for final evidence.” |
| `final_all_hc18_comparison_summary.csv` | “This is the current reconstruction headline table.” |
| `downstream_utility_summary.csv` | “This is the current frozen-U-Net anatomy-utility table.” |
| `significance_report.csv` | “This is the paired statistical evidence with 47-test Bonferroni correction.” |
| `README_FULL.md` | “This documents the audited environment, workflow, limitations, and current interpretation.” |
| `RESULTS_MANIFEST.md` | “This is the guide to canonical Phase 1 result artifacts.” |

---

# 7. Open questions and stale-value note

## Open questions / things not verified on disk

1. The sibling `CONFERENCE\baselines` directory is outside the permitted root, so its exact cloned-repository, virtual-environment, weight, and environment-file inventory was not verified here.
2. The prompt-specified locked poor-frame ratio pair `0.5469 → 0.3675` has no exact aggregate source found under the project root. The current verified sweep summary reports mean reduction `0.17940000`.
3. Runtime/cost columns are absent from the canonical reconstruction summary; do not make comparative speed claims without a dedicated timing CSV.
4. Phase B training is partial and must not replace Phase A.
5. ACOUSLIC proxy labels and feedback are not expert or clinical validation.
6. OSEDiff was run with a fixed prompt/skip-DAPE recovery path according to its run artifacts; this implementation detail should be disclosed when describing the reproduction.
7. The main requirements are mostly unpinned; exact reproducibility depends on the audited environment logs.

## Stale forbidden values still present

The following files contain the forbidden historical percentage string. It is not reproduced here as a valid result and was not corrected during this read-only audit:

- `results/final_ablation_original_blind_adaptive.csv`
- `results/final_ablation_feedback_reassessment.csv`
- `results/final_ablation_tables.md`

No file under `results/` was found containing either forbidden threshold-change phrase.

**Correct current acquisition results to use:** heuristic-adaptive `17.94%`, learned-adaptive `17.24%`, blind `8.00%`. Use the poor-frame ratio pair only after its exact provenance is identified; the current on-disk sweep aggregate verifies mean reduction `0.17940000`.

