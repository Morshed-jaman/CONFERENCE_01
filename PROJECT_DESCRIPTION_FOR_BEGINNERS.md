# QAFUE Beginner Workflow Guide

Project name: **QAFUE - Quality-Aware Fetal Ultrasound Enhancement**  
Repository: `D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project`

This file explains the project from zero. Imagine you are new to machine learning, medical imaging, and this codebase. The goal is to help you understand what we are doing, why we are doing it, what each folder/script means, and how the full workflow fits together.

## 1. The One-Sentence Explanation

We are trying to improve low-quality fetal ultrasound images while checking whether the improvement is useful for downstream fetal-head analysis, not just visually nicer.

In plain language:

> The project takes fetal ultrasound images that look degraded, restores them with machine-learning models, checks whether the restored images preserve fetal-head anatomy, and also studies whether ultrasound sweep quality can be estimated and improved using proxy quality feedback.

## 2. The Main Goal

The main goal is **not** simply "make images pretty".

The main goal is:

1. Start with fetal ultrasound images.
2. Create low-quality versions in a controlled way.
3. Train restoration models to recover cleaner images.
4. Evaluate the restored images with image-quality metrics.
5. Check whether fetal-head segmentation still works after restoration.
6. Add a quality-aware feedback branch using ACOUSLIC-AI sweep frames.
7. Show that quality-aware restoration is more responsible than blind enhancement.

The safest thesis claim is:

> SwinIR-Light Phase A gives the best LPIPS and DISTS among the registered current reconstruction models, while preserving downstream fetal-head segmentation utility close to the HR reference level.

Important source files:

- `README.md`
- `README_FULL.md`
- `WORK_ATLAS.md`
- `THESIS_VIVA_DOSSIER.md`
- `results/final_all_hc18_comparison_summary.csv`
- `results/downstream_utility_summary.csv`

## 3. What Problem Are We Solving?

Fetal ultrasound images can be poor quality because of:

- blur
- noise
- low contrast
- motion
- unstable sweep acquisition
- incomplete or weak visibility
- variation between operators

If an image is poor, a doctor or an automated algorithm may struggle to identify anatomy clearly.

But there is a danger: if an AI model makes an image look sharper, it may also hallucinate or distort details. So we do not only ask:

> Does the image look better?

We also ask:

> Does the restored image still preserve fetal-head anatomy?

That is why this project uses both reconstruction metrics and downstream segmentation metrics.

## 4. The Two Big Branches of the Project

This project has two major branches.

### Branch A: HC18 Super-Resolution Branch

This is the main reconstruction branch.

It uses the HC18 fetal-head ultrasound dataset.

Purpose:

- prepare fetal-head ultrasound images
- create degraded low-quality images
- train restoration models
- compare model outputs
- test downstream fetal-head segmentation utility

Main dataset folder:

```text
data/processed/HC18_SR/
```

Important subfolders:

```text
HR/        clean/high-quality processed images
masks/     fetal-head masks
LR_x2/     x2 degraded images
LR_x4/     x4 degraded images
LQ_mixed/  mixed low-quality degraded images
```

Current counts:

- `999` HR images
- `999` masks
- `999` LR_x2 images
- `999` LR_x4 images
- `999` LQ_mixed images
- `699` train images
- `150` validation images
- `150` test images

Key files:

- `data/processed/HC18_SR/train.csv`
- `data/processed/HC18_SR/val.csv`
- `data/processed/HC18_SR/test.csv`
- `data/processed/HC18_SR/hc18_manifest.csv`
- `data/processed/HC18_SR/hc18_lq_manifest.csv`

### Branch B: ACOUSLIC-AI Quality Feedback Branch

This branch studies ultrasound sweep quality.

It uses ACOUSLIC-AI frames and creates proxy quality labels such as:

- `good_frame`
- `blurry_frame`
- `low_contrast_frame`
- `noisy_frame`
- `poor_frame`

Purpose:

- estimate frame quality
- train a quality classifier
- choose adaptive enhancement actions
- reassess sweep quality after enhancement
- test cross-dataset transfer from ACOUSLIC-AI to HC18

Main dataset folder:

```text
data/processed/ACOUSLIC_AI/
```

Important files:

- `data/processed/ACOUSLIC_AI/acouslic_frame_manifest.csv`
- `data/processed/ACOUSLIC_AI/acouslic_proxy_quality_manifest.csv`
- `data/processed/ACOUSLIC_AI/acouslic_sweep_quality_summary.csv`

Current counts:

- `100` sweep volumes
- `10000` extracted frames
- `10000` proxy-quality frame rows
- `100` sweep summary rows

## 5. What Is Super-Resolution?

Super-resolution means:

> Taking a low-quality or low-detail image and trying to reconstruct a cleaner, sharper, more useful version.

In this project:

- HR means high-resolution or high-quality reference image.
- LQ means low-quality input image.
- SR means super-resolved or restored output image.

Example:

```text
LQ ultrasound image -> restoration model -> restored ultrasound image
```

But again: the restored image must be checked carefully. A sharp wrong image is not useful.

## 6. What Is Quality-Aware Enhancement?

Blind enhancement means applying the same enhancement to every image.

Quality-aware enhancement means:

> First estimate what is wrong with the frame, then choose a suitable action.

Example:

| Frame problem | Better action |
|---|---|
| already good | keep original |
| blurry | enhance/sharpen |
| low contrast | improve contrast |
| noisy | denoise |
| very poor | recommend repeat/reacquire |

This is why the project is called **Quality-Aware Fetal Ultrasound Enhancement**.

## 7. Full Workflow in One Diagram

```text
                         PROJECT: QAFUE
                              |
          ------------------------------------------------
          |                                              |
          v                                              v
   HC18 reconstruction branch                    ACOUSLIC quality branch
          |                                              |
          v                                              v
   Prepare HR + masks                            Extract sweep frames
          |                                              |
          v                                              v
   Generate LQ images                            Compute proxy features
          |                                              |
          v                                              v
   Train restoration models                      Label frame quality
          |                                              |
          v                                              v
   Select canonical model                        Train proxy MLP
   SwinIR-Light Phase A                                  |
          |                                              v
          v                                      Adaptive feedback policy
   Evaluate reconstruction                               |
          |                                              v
          v                                      Sweep reassessment
   Evaluate downstream Dice                              |
          |                                              v
          ---------------------------------------- Cross-dataset transfer
                              |
                              v
                    Thesis/report conclusions
```

## 8. Step-by-Step Workflow

### Step 1: Prepare HC18 images

Script:

```text
scripts/01_prepare_hc18.py
```

What it does:

- reads raw HC18 ultrasound images
- ignores annotation files when collecting images
- resizes images to `256 x 256`
- resizes masks using nearest-neighbour interpolation
- creates processed HR images and masks
- creates train/validation/test split CSV files

Outputs:

```text
data/processed/HC18_SR/HR/
data/processed/HC18_SR/masks/
data/processed/HC18_SR/train.csv
data/processed/HC18_SR/val.csv
data/processed/HC18_SR/test.csv
data/processed/HC18_SR/hc18_manifest.csv
```

Beginner meaning:

> This step cleans and organises the dataset so the machine-learning code can use it.

### Step 2: Generate low-quality images

Script:

```text
scripts/02_generate_lq_hc18.py
```

What it does:

- starts from HR images
- creates `LR_x2`
- creates `LR_x4`
- creates `LQ_mixed`

The `LQ_mixed` degradation chain is:

```text
HR image
-> bicubic x4 downsample/upsample
-> Gaussian blur
-> speckle noise
-> motion blur
-> contrast reduction
-> LQ_mixed image
```

Outputs:

```text
data/processed/HC18_SR/LR_x2/
data/processed/HC18_SR/LR_x4/
data/processed/HC18_SR/LQ_mixed/
data/processed/HC18_SR/hc18_lq_manifest.csv
```

Beginner meaning:

> We intentionally damage clean images so we can train and test whether models can recover them.

### Step 3: Train restoration models

Many models were trained historically:

| Model family | Example script |
|---|---|
| SR-only CNN | `scripts/05_train_sr_only_hc18.py` |
| Quality-aware CNN | `scripts/07_train_quality_aware_sr_hc18.py` |
| Diffusion models | `scripts/11_train_diffusion_sr_hc18.py` |
| CNN-residual diffusion | `scripts/20_train_cnn_residual_diffusion_sr_hc18.py` |
| Perceptual/GAN CNN | `scripts/40_train_sr_perceptual_hc18.py` |
| SwinIR-Light Phase A | `scripts/50_train_swinir_hc18.py` |
| Phase B seg-aware SwinIR | `scripts/51_train_segaware_hc18.py` |

The current headline reconstruction model is:

```text
SwinIR-Light Phase A
```

Main files:

```text
scripts/models/swinir_light.py
scripts/50_train_swinir_hc18.py
results/models/swinir_light_phase_a_seed42/best_lpips.pth
```

Beginner meaning:

> This is the main AI model we currently trust most for the reconstruction story.

### Step 4: Understand the SwinIR-Light model

Architecture file:

```text
scripts/models/swinir_light.py
```

Training file:

```text
scripts/50_train_swinir_hc18.py
```

Checkpoint:

```text
results/models/swinir_light_phase_a_seed42/best_lpips.pth
```

It uses these loss weights:

```text
L1 loss weight     = 1.0
LPIPS loss weight  = 0.2
GAN loss weight    = 0.005
```

What those mean:

- L1 keeps the output close to the HR image at pixel level.
- LPIPS encourages perceptual similarity.
- GAN gives a small realism/sharpness signal.

Beginner meaning:

> The model learns by comparing its output to the clean reference image and getting punished when it is wrong.

### Step 5: Evaluate reconstruction quality

Script:

```text
scripts/42_compare_all_models.py
```

Output:

```text
results/final_all_hc18_comparison_summary.csv
```

Metrics:

| Metric | Simple meaning |
|---|---|
| PSNR | pixel-level similarity; higher is usually better |
| SSIM | structural similarity; higher is better |
| LPIPS | perceptual difference; lower is better |
| DISTS | perceptual/structure-texture difference; lower is better |
| MAE | average error; lower is better |
| EdgeScore | edge similarity; higher is better |
| NIQE | no-reference quality score; lower is usually better |
| BRISQUE | no-reference quality score; lower is usually better |

Current SwinIR-Light Phase A headline:

```text
LPIPS = 0.10091456212103367
DISTS = 0.15086822072664896
```

Source:

```text
results/final_all_hc18_comparison_summary.csv
```

Beginner meaning:

> SwinIR-Light Phase A is best in the current registered table for the perceptual metrics LPIPS and DISTS.

### Step 6: Evaluate downstream fetal-head segmentation

Script:

```text
scripts/46_downstream_utility_eval.py
```

Output:

```text
results/downstream_utility_summary.csv
```

Why we do this:

> An enhanced image should still preserve the fetal head. If a U-Net segmentation model fails on the restored image, the restoration may not be useful.

Important Dice values:

```text
HR Dice                  = 0.8896568280296212
LQ_mixed Dice            = 0.8849903062929627
SwinIR-Light Phase A Dice = 0.8877019882187267
```

Beginner meaning:

> SwinIR-Light does not magically beat the HR image. It stays close to HR and slightly above LQ in downstream Dice. That supports preservation, not clinical superiority.

### Step 7: Compute statistical confidence

Script:

```text
scripts/48_significance_report.py
```

Outputs:

```text
results/bootstrap_metric_summary.csv
results/significance_report.csv
```

Important SwinIR LPIPS confidence interval:

```text
mean    = 0.10084894095623005
ci_low  = 0.09694007307265858
ci_high = 0.10476489356102266
n       = 148
```

Important p-values:

```text
SwinIR vs Quality-aware+perceptual LPIPS p_bonf = 3.987203950677623e-10
SwinIR vs SR-only+perceptual LPIPS p_bonf      = 3.632907441626531e-19
```

Beginner meaning:

> We are checking that the improvement is not just random noise in the test set.

### Step 8: Build ACOUSLIC proxy quality labels

Script:

```text
scripts/24_compute_acouslic_proxy_quality.py
```

Output:

```text
data/processed/ACOUSLIC_AI/acouslic_proxy_quality_manifest.csv
data/processed/ACOUSLIC_AI/acouslic_sweep_quality_summary.csv
```

Features used:

- blur score
- contrast score
- brightness mean
- brightness std
- noise proxy
- edge density
- dark ratio
- bright ratio
- frame difference
- optical-flow mean magnitude
- optical-flow standard deviation
- jitter score

Beginner meaning:

> We calculate simple numbers that describe whether an ultrasound frame looks blurry, noisy, dark, low-contrast, or unstable.

Important warning:

> These are proxy labels, not expert clinical labels.

### Step 9: Train the proxy quality MLP

Script:

```text
scripts/29_train_proxy_quality_embedding.py
```

Output:

```text
results/models/proxy_quality_embedding_mlp.pth
results/acouslic_ai/proxy_quality_embedding_metrics.csv
```

Test result:

```text
accuracy = 0.97666667
macro_f1 = 0.96882341
```

Beginner meaning:

> We train a small classifier to recognise proxy frame-quality categories from those 12 features.

### Step 10: Run adaptive feedback enhancement

Heuristic script:

```text
scripts/28_adaptive_feedback_aware_enhancement.py
```

Learned script:

```text
scripts/30_learned_adaptive_feedback_enhancement.py
```

Current 100-sweep results:

```text
Heuristic adaptive improvement = 17.94000000%
Learned adaptive improvement   = 17.24000000%
```

Sources:

```text
results/acouslic_ai/adaptive_feedback_enhancement_summary.csv
results/acouslic_ai/learned_adaptive_enhancement_summary.csv
```

Beginner meaning:

> Instead of enhancing every frame the same way, we choose an action based on estimated quality.

### Step 11: Reassess sweep feedback

Script:

```text
scripts/27_sweep_feedback_change_analysis.py
```

Output:

```text
results/acouslic_ai/adaptive_sweep_feedback_change_summary.csv
```

Current 100-sweep result:

```text
number_of_sweeps_analyzed = 100
number_of_sweeps_with_reduced_poor_frame_ratio = 100
mean_poor_frame_ratio_reduction = 0.17940000
repeat_sweep cases improved = 36
unchanged due to motion-based problems = 28
```

Beginner meaning:

> After adaptive enhancement, all 100 analysed sweeps had lower poor-frame ratio, but some motion-related problems still remained.

### Step 12: Cross-dataset transfer

Scripts:

```text
scripts/60_apply_acouslic_mlp_to_hc18.py
scripts/61_cross_dataset_quality_analysis.py
```

Outputs:

```text
results/cross_dataset/acouslic_quality_on_hc18_per_image.csv
results/cross_dataset/acouslic_quality_on_hc18_summary.csv
results/cross_dataset/cross_dataset_stats.json
docs/CROSS_DATASET_TRANSFER.md
```

Main result:

```text
Spearman rho = 0.5688522618064861
p_value      = 3.094730049128844e-14
verdict      = STRONG TRANSFER
```

Beginner meaning:

> The ACOUSLIC-trained proxy quality model gives useful external evidence when applied to HC18, but this is still proxy-quality evidence, not clinical validation.

## 9. The Current Main Result

The current canonical model:

```text
SwinIR-Light Phase A
```

Main checkpoint:

```text
results/models/swinir_light_phase_a_seed42/best_lpips.pth
```

Main reconstruction result:

```text
LPIPS = 0.10091456212103367
DISTS = 0.15086822072664896
```

Main downstream result:

```text
Dice = 0.8877019882187267
```

Main interpretation:

> SwinIR-Light Phase A gives the best perceptual reconstruction among the registered current comparison models while keeping downstream segmentation utility near the HR/LQ ceiling.

What not to say:

```text
Do not say: This is clinically validated.
Do not say: This is real-time scanner feedback.
Do not say: This proves better diagnosis.
Do not say: This is state of the art.
```

What to say:

```text
This is an offline research prototype.
It uses proxy quality labels.
It evaluates reconstruction and downstream segmentation utility.
It needs expert clinical validation before clinical use.
```

## 10. Important Files and What They Mean

### Root explanation files

| File | Meaning |
|---|---|
| `README.md` | short project overview |
| `README_FULL.md` | detailed technical self-audit |
| `WORK_ATLAS.md` | map of repo, pipeline, results, risks |
| `THESIS_VIVA_DOSSIER.md` | viva-ready thesis dossier |
| `PROJECT_DESCRIPTION_FOR_BEGINNERS.md` | this beginner guide |

### Main data files

| File/folder | Meaning |
|---|---|
| `data/processed/HC18_SR/HR/` | clean processed HC18 images |
| `data/processed/HC18_SR/LQ_mixed/` | degraded low-quality images |
| `data/processed/HC18_SR/masks/` | fetal-head masks |
| `data/processed/HC18_SR/hc18_lq_manifest.csv` | table connecting HR, LQ, masks, split |
| `data/processed/ACOUSLIC_AI/acouslic_proxy_quality_manifest.csv` | ACOUSLIC frame quality table |

### Main scripts

| Script | Meaning |
|---|---|
| `scripts/01_prepare_hc18.py` | prepare HC18 images/masks |
| `scripts/02_generate_lq_hc18.py` | create degraded LQ images |
| `scripts/50_train_swinir_hc18.py` | train canonical SwinIR-Light Phase A |
| `scripts/42_compare_all_models.py` | compare reconstruction metrics |
| `scripts/46_downstream_utility_eval.py` | evaluate downstream Dice |
| `scripts/48_significance_report.py` | compute confidence intervals and p-values |
| `scripts/24_compute_acouslic_proxy_quality.py` | compute ACOUSLIC proxy quality |
| `scripts/29_train_proxy_quality_embedding.py` | train proxy quality MLP |
| `scripts/28_adaptive_feedback_aware_enhancement.py` | heuristic adaptive enhancement |
| `scripts/30_learned_adaptive_feedback_enhancement.py` | learned adaptive enhancement |
| `scripts/60_apply_acouslic_mlp_to_hc18.py` | apply ACOUSLIC MLP to HC18 |
| `scripts/61_cross_dataset_quality_analysis.py` | analyse cross-dataset transfer |

### Main result files

| File | Meaning |
|---|---|
| `results/final_all_hc18_comparison_summary.csv` | main reconstruction metric table |
| `results/downstream_utility_summary.csv` | downstream Dice and IoU table |
| `results/bootstrap_metric_summary.csv` | confidence intervals |
| `results/significance_report.csv` | p-values and Bonferroni tests |
| `results/acouslic_ai/proxy_quality_embedding_metrics.csv` | proxy MLP accuracy/F1 |
| `results/acouslic_ai/adaptive_feedback_enhancement_summary.csv` | heuristic adaptive improvement |
| `results/acouslic_ai/learned_adaptive_enhancement_summary.csv` | learned adaptive improvement |
| `results/acouslic_ai/adaptive_sweep_feedback_change_summary.csv` | sweep reassessment result |
| `results/cross_dataset/cross_dataset_stats.json` | cross-dataset transfer statistics |

## 11. Model Families in Simple Words

### Bicubic baseline

This is not deep learning. It is a traditional resizing method.

Purpose:

> Provides a simple baseline.

### SR-only CNN

A small neural network learns:

```text
LQ image -> HR image
```

Purpose:

> Basic supervised restoration.

### Quality-aware CNN

This model also uses quality features.

Purpose:

> Let the model know something about image quality while restoring.

### Perceptual/GAN CNN

This adds perceptual and adversarial losses.

Purpose:

> Improve perceptual appearance, not only pixel similarity.

### Diffusion models

Diffusion models try to refine images through noise/removal style learning.

Purpose in this repo:

> Legacy and baseline restoration experiments, especially edge/boundary refinement.

### SwinIR-Light Phase A

This is the current canonical model.

Purpose:

> Best current perceptual reconstruction story in the repo.

### Phase B segmentation-aware SwinIR

This tries to add segmentation-aware loss.

Status:

> Incomplete/deferred. Do not use it as a headline result yet.

## 12. How to Explain the Thesis in a Viva

If someone asks:

> What is your thesis about?

Answer:

> My thesis is about quality-aware fetal ultrasound enhancement. Instead of only making ultrasound images look sharper, I evaluate whether restoration improves perceptual quality while preserving fetal-head segmentation utility. I use HC18 for controlled reconstruction and ACOUSLIC-AI for proxy acquisition-quality feedback. The current main model is SwinIR-Light Phase A, and the broader framework includes proxy quality estimation, adaptive enhancement, and offline sweep reassessment.

If someone asks:

> What is your main model?

Answer:

> The current canonical reconstruction model is SwinIR-Light Phase A, trained on HC18 LQ_mixed images with L1, LPIPS, and GAN losses. Its checkpoint is `results/models/swinir_light_phase_a_seed42/best_lpips.pth`.

If someone asks:

> What is your best result?

Answer:

> In the current registered comparison, SwinIR-Light Phase A gives LPIPS `0.10091456212103367` and DISTS `0.15086822072664896`, the best perceptual reconstruction results in that table. Its downstream Dice is `0.8877019882187267`, close to HR Dice `0.8896568280296212`.

If someone asks:

> Is this clinically validated?

Answer:

> No. This is an offline research prototype. The labels in the ACOUSLIC branch are proxy labels, and the downstream Dice is from a frozen U-Net, not from expert clinical validation.

## 13. Beginner Glossary

| Term | Meaning |
|---|---|
| HR | High-resolution or clean reference image |
| LQ | Low-quality image |
| SR | Super-resolution/restored image |
| CNN | Convolutional neural network |
| SwinIR | Transformer-based image restoration model |
| GAN | Adversarial training signal for realism |
| LPIPS | Perceptual distance metric; lower is better |
| DISTS | Perceptual/structure-texture metric; lower is better |
| PSNR | Pixel similarity metric; higher is better |
| SSIM | Structure similarity metric; higher is better |
| MAE | Mean absolute error; lower is better |
| Dice | Segmentation overlap metric; higher is better |
| IoU | Intersection-over-union segmentation metric; higher is better |
| Proxy label | A label created from rules/features, not expert annotation |
| Checkpoint | Saved trained model weights |
| Manifest | CSV file listing images and paths |
| Train split | Data used to train the model |
| Validation split | Data used to tune/check model during training |
| Test split | Held-out data used for final evaluation |
| Bootstrap CI | Confidence interval estimated by resampling |
| Bonferroni correction | Conservative correction for many statistical tests |

## 14. What You Should Study First

If you are new, study in this order:

1. Read this file: `PROJECT_DESCRIPTION_FOR_BEGINNERS.md`
2. Read the short overview: `README.md`
3. Read the project map: `WORK_ATLAS.md`
4. Read the viva dossier: `THESIS_VIVA_DOSSIER.md`
5. Open the main result table: `results/final_all_hc18_comparison_summary.csv`
6. Open the downstream table: `results/downstream_utility_summary.csv`
7. Open the main training script: `scripts/50_train_swinir_hc18.py`
8. Open the architecture: `scripts/models/swinir_light.py`
9. Open the ACOUSLIC quality script: `scripts/24_compute_acouslic_proxy_quality.py`
10. Open the cross-dataset report: `docs/CROSS_DATASET_TRANSFER.md`

## 15. The Project Story in Human Language

Here is the whole project as a simple story:

We start with fetal ultrasound images from HC18. These images have fetal-head masks, so we can test whether anatomy is preserved. We resize and organise them, then create low-quality versions using blur, noise, motion blur, and contrast reduction. This gives us paired examples: a degraded input and a cleaner target.

We train several restoration models. Older models include CNNs and diffusion-inspired models. The current main model is SwinIR-Light Phase A. It learns to transform `LQ_mixed` images into restored images using L1, LPIPS, and a small GAN loss.

After training, we compare the models with reconstruction metrics. SwinIR-Light Phase A gives the best LPIPS and DISTS in the current comparison table. That means it performs best on the perceptual metrics used in this repo.

But image quality alone is not enough. We also test whether a frozen U-Net can still segment the fetal head from the restored images. SwinIR-Light Phase A has Dice close to the HR reference, so the result supports anatomy-preservation utility.

In parallel, we use ACOUSLIC-AI sweep frames to study acquisition quality. We compute proxy features such as blur, contrast, noise, edge density, optical flow, and jitter. These features create proxy quality labels. Then we train a small MLP to recognise those labels.

Using those proxy labels, we test adaptive feedback. Instead of applying the same enhancement to every frame, the system can keep good frames, enhance blurry/low-contrast frames, denoise noisy frames, or recommend repeat acquisition for poor frames.

Finally, we apply the ACOUSLIC-trained proxy quality model to HC18 outputs. This is a zero-shot cross-dataset check. It gives extra evidence that restoration shifts images in a direction recognised by the ACOUSLIC quality model, but it is still proxy evidence, not clinical validation.

## 16. The Honest Limitations

You must remember these limitations:

- This is not a clinical product.
- This is not real-time scanner guidance.
- ACOUSLIC labels are proxy labels, not expert labels.
- HC18 degradation is simulated.
- Phase B segmentation-aware training is incomplete.
- SwinIR improves perceptual metrics clearly, but downstream Dice gains are small.
- The project needs expert sonographer validation before clinical claims.
- Do not mix the current 100-sweep adaptive result with the older 50-sweep result.

## 17. Current vs Legacy Results

Current headline:

```text
SwinIR-Light Phase A
```

Current main reconstruction result:

```text
LPIPS = 0.10091456212103367
DISTS = 0.15086822072664896
```

Current adaptive feedback:

```text
Heuristic adaptive = 17.94000000%
Learned adaptive   = 17.24000000%
```

Legacy result:

```text
50-sweep adaptive = 20.10000000%
```

Important:

> The 20.10000000% result is from an older 50-sweep backup. Do not mix it with the current 100-sweep result.

## 18. Final Mental Model

If you remember only one thing, remember this:

```text
QAFUE is not just "make ultrasound sharper".

QAFUE is:

low-quality fetal ultrasound
-> restoration
-> quality checking
-> anatomy utility checking
-> adaptive feedback
-> honest limitations
```

That is the heart of the project.

