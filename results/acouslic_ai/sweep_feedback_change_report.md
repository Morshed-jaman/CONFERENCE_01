# ACOUSLIC-AI Sweep Feedback Change Analysis

These are proxy feedback labels derived from image and motion features, not expert clinical labels.

This report uses Stage 26 enhanced-frame labels to estimate whether
super-resolution enhancement changes sweep-level proxy feedback decisions.
Motion-based labels such as `move_slower` and `keep_probe_steady` are preserved,
because image enhancement does not change probe motion.

## Inputs

- Stage 26 per-frame enhancement CSV: `results/acouslic_ai/feedback_aware_enhancement_per_frame.csv`
- Original sweep-quality summary CSV: `data/processed/ACOUSLIC_AI/acouslic_sweep_quality_summary.csv`
- Original frame proxy-quality manifest CSV: `data/processed/ACOUSLIC_AI/acouslic_proxy_quality_manifest.csv`

## Decision rules

- Poor-like frame labels: `poor_frame`, `blurry_frame`, `low_contrast_frame`, `noisy_frame`
- Enhanced `poor_frame_ratio` > 0.40: `repeat_sweep`
- Enhanced `low_contrast_frame` ratio > 0.25: `improve_visibility`
- Original `move_slower` / `keep_probe_steady`: preserved as motion-based feedback
- Otherwise: `acceptable`

## Summary

- Number of sweeps analyzed: 50
- Number of sweeps with reduced poor-frame ratio: 0
- Mean poor-frame-ratio reduction: -0.449400
- Number of sweeps with improved good-frame ratio: 0
- Number of `repeat_sweep` cases changed to `acceptable` or `improve_visibility`: 0
- Number of feedback decisions unchanged due to motion-based problems: 0

## Original sweep-feedback distribution

| Label | Count |
|---|---:|
| `improve_visibility` | 1 |
| `keep_probe_steady` | 8 |
| `move_slower` | 13 |
| `repeat_sweep` | 28 |

## Enhanced sweep-feedback distribution

| Label | Count |
|---|---:|
| `repeat_sweep` | 50 |

## Per-sweep results

| Sweep ID | Frames | Original feedback | Enhanced feedback | Poor ratio before | Poor ratio after | Poor ratio reduction | Good ratio before | Good ratio after | Good ratio improvement | Reason |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 0199616b-bdeb-4119-97a3-a5a3571bd641 | 100 | `move_slower` | `repeat_sweep` | 0.5400 | 1.0000 | -0.4600 | 0.4600 | 0.0000 | -0.4600 | enhanced_poor_like_ratio_high |
| 02d3a9bc-63e2-4deb-9dc1-dba17e7d54c1 | 100 | `move_slower` | `repeat_sweep` | 0.5500 | 1.0000 | -0.4500 | 0.4500 | 0.0000 | -0.4500 | enhanced_poor_like_ratio_high |
| 02ee26a5-a665-4531-bec1-8bac83345a94 | 100 | `move_slower` | `repeat_sweep` | 0.5600 | 1.0000 | -0.4400 | 0.4400 | 0.0000 | -0.4400 | enhanced_poor_like_ratio_high |
| 032b9024-3685-4a81-a62e-813d2da3d387 | 100 | `repeat_sweep` | `repeat_sweep` | 0.5800 | 1.0000 | -0.4200 | 0.4200 | 0.0000 | -0.4200 | enhanced_poor_like_ratio_high |
| 0428da3d-fef4-4c9b-96f2-5ffedaee82d0 | 100 | `repeat_sweep` | `repeat_sweep` | 0.5500 | 1.0000 | -0.4500 | 0.4500 | 0.0000 | -0.4500 | enhanced_poor_like_ratio_high |
| 046ed03e-4b35-4519-bb5f-cd4b0474a060 | 100 | `move_slower` | `repeat_sweep` | 0.5100 | 1.0000 | -0.4900 | 0.4900 | 0.0000 | -0.4900 | enhanced_poor_like_ratio_high |
| 04a04f2e-840b-47f8-a907-abe7aeab3f41 | 100 | `repeat_sweep` | `repeat_sweep` | 0.4900 | 1.0000 | -0.5100 | 0.5100 | 0.0000 | -0.5100 | enhanced_poor_like_ratio_high |
| 04ef1176-7d54-4942-899d-f0fa69ad34bb | 100 | `repeat_sweep` | `repeat_sweep` | 0.5200 | 1.0000 | -0.4800 | 0.4800 | 0.0000 | -0.4800 | enhanced_poor_like_ratio_high |
| 056764b4-7c0c-4452-832d-3d1cbb5a0464 | 100 | `move_slower` | `repeat_sweep` | 0.4700 | 0.9900 | -0.5200 | 0.5300 | 0.0100 | -0.5200 | enhanced_poor_like_ratio_high |
| 05f72776-bf41-40ea-bdea-a7bcfa1eb9ba | 100 | `keep_probe_steady` | `repeat_sweep` | 0.4000 | 1.0000 | -0.6000 | 0.6000 | 0.0000 | -0.6000 | enhanced_poor_like_ratio_high |
| 0607eff0-2268-46b6-8fb8-16e40a521a42 | 100 | `move_slower` | `repeat_sweep` | 0.5800 | 1.0000 | -0.4200 | 0.4200 | 0.0000 | -0.4200 | enhanced_poor_like_ratio_high |
| 0658cdf9-53bc-46a9-ae86-cc99a6fc5e77 | 100 | `repeat_sweep` | `repeat_sweep` | 0.5900 | 0.9900 | -0.4000 | 0.4100 | 0.0100 | -0.4000 | enhanced_poor_like_ratio_high |
| 0685cfaf-d585-40b3-a365-52f70f2536b6 | 100 | `repeat_sweep` | `repeat_sweep` | 0.5700 | 1.0000 | -0.4300 | 0.4300 | 0.0000 | -0.4300 | enhanced_poor_like_ratio_high |
| 069c0349-1e8d-42d0-afa1-69c7114a742f | 100 | `repeat_sweep` | `repeat_sweep` | 0.6100 | 1.0000 | -0.3900 | 0.3900 | 0.0000 | -0.3900 | enhanced_poor_like_ratio_high |
| 071ee12c-a8a5-45fd-b191-fd5ec3b00766 | 100 | `repeat_sweep` | `repeat_sweep` | 0.6800 | 0.9900 | -0.3100 | 0.3200 | 0.0100 | -0.3100 | enhanced_poor_like_ratio_high |
| 09ddeecb-72e0-417a-85fd-77d397f6219d | 100 | `move_slower` | `repeat_sweep` | 0.5400 | 0.9900 | -0.4500 | 0.4600 | 0.0100 | -0.4500 | enhanced_poor_like_ratio_high |
| 0d0a3298-a9c6-43c3-a9e3-df3a9c0afa06 | 100 | `repeat_sweep` | `repeat_sweep` | 0.6300 | 1.0000 | -0.3700 | 0.3700 | 0.0000 | -0.3700 | enhanced_poor_like_ratio_high |
| 0d7c4d8f-6e07-4f2b-aa76-8915ce15a38f | 100 | `repeat_sweep` | `repeat_sweep` | 0.3900 | 0.9900 | -0.6000 | 0.6100 | 0.0100 | -0.6000 | enhanced_poor_like_ratio_high |
| 0e2631b6-ed14-4ce4-bc17-896590c406fd | 100 | `improve_visibility` | `repeat_sweep` | 0.4200 | 1.0000 | -0.5800 | 0.5800 | 0.0000 | -0.5800 | enhanced_poor_like_ratio_high |
| 0f32e98f-8257-4649-a8a2-3cccb2c1a2e5 | 100 | `repeat_sweep` | `repeat_sweep` | 0.7200 | 0.9900 | -0.2700 | 0.2800 | 0.0100 | -0.2700 | enhanced_poor_like_ratio_high |
| 1269a6c4-7f04-4927-9308-e4fd68883906 | 100 | `repeat_sweep` | `repeat_sweep` | 0.5800 | 1.0000 | -0.4200 | 0.4200 | 0.0000 | -0.4200 | enhanced_poor_like_ratio_high |
| 12941223-b27c-47f5-bcb0-7f7bfba18bba | 100 | `move_slower` | `repeat_sweep` | 0.6100 | 1.0000 | -0.3900 | 0.3900 | 0.0000 | -0.3900 | enhanced_poor_like_ratio_high |
| 136db39d-90e4-4702-aee7-dc5cd3169268 | 100 | `repeat_sweep` | `repeat_sweep` | 0.5000 | 0.9900 | -0.4900 | 0.5000 | 0.0100 | -0.4900 | enhanced_poor_like_ratio_high |
| 1613944c-ea48-4953-84e9-953b9d86f175 | 100 | `repeat_sweep` | `repeat_sweep` | 0.4300 | 1.0000 | -0.5700 | 0.5700 | 0.0000 | -0.5700 | enhanced_poor_like_ratio_high |
| 177786dd-a8f4-4d57-ac1b-13811480bd09 | 100 | `repeat_sweep` | `repeat_sweep` | 0.6100 | 0.9900 | -0.3800 | 0.3900 | 0.0100 | -0.3800 | enhanced_poor_like_ratio_high |
| 17b31515-75fb-41fc-8908-7c0264d21204 | 100 | `keep_probe_steady` | `repeat_sweep` | 0.4200 | 1.0000 | -0.5800 | 0.5800 | 0.0000 | -0.5800 | enhanced_poor_like_ratio_high |
| 19202c0a-fade-4c7a-9878-e91d823963d6 | 100 | `repeat_sweep` | `repeat_sweep` | 0.5100 | 1.0000 | -0.4900 | 0.4900 | 0.0000 | -0.4900 | enhanced_poor_like_ratio_high |
| 1a998ef3-797f-4a17-9aa6-96cfb9af87c9 | 100 | `keep_probe_steady` | `repeat_sweep` | 0.5000 | 1.0000 | -0.5000 | 0.5000 | 0.0000 | -0.5000 | enhanced_poor_like_ratio_high |
| 1af7894a-9f0a-41e8-9c10-a11c0341a93e | 100 | `repeat_sweep` | `repeat_sweep` | 0.5200 | 1.0000 | -0.4800 | 0.4800 | 0.0000 | -0.4800 | enhanced_poor_like_ratio_high |
| 1b250c39-7217-4921-838b-ed2681ebfef2 | 100 | `move_slower` | `repeat_sweep` | 0.4300 | 1.0000 | -0.5700 | 0.5700 | 0.0000 | -0.5700 | enhanced_poor_like_ratio_high |
| 1b3db504-c1fe-45f0-96dc-3cfb851d5bca | 100 | `move_slower` | `repeat_sweep` | 0.4300 | 1.0000 | -0.5700 | 0.5700 | 0.0000 | -0.5700 | enhanced_poor_like_ratio_high |
| 1c1a1114-3e30-436b-b656-101e6050ea32 | 100 | `repeat_sweep` | `repeat_sweep` | 0.5900 | 1.0000 | -0.4100 | 0.4100 | 0.0000 | -0.4100 | enhanced_poor_like_ratio_high |
| 1cd004a2-94fb-429a-9e14-fc950bbd8852 | 100 | `repeat_sweep` | `repeat_sweep` | 0.4500 | 1.0000 | -0.5500 | 0.5500 | 0.0000 | -0.5500 | enhanced_poor_like_ratio_high |
| 1d25be5c-d4ba-48bf-afd4-a133581f8491 | 100 | `keep_probe_steady` | `repeat_sweep` | 0.6300 | 1.0000 | -0.3700 | 0.3700 | 0.0000 | -0.3700 | enhanced_poor_like_ratio_high |
| 1d479820-767b-4d0f-b6cf-12ec5d3a26e7 | 100 | `repeat_sweep` | `repeat_sweep` | 0.5700 | 1.0000 | -0.4300 | 0.4300 | 0.0000 | -0.4300 | enhanced_poor_like_ratio_high |
| 1dab0e37-e5e4-41d4-b132-32103b385c78 | 100 | `repeat_sweep` | `repeat_sweep` | 0.6500 | 1.0000 | -0.3500 | 0.3500 | 0.0000 | -0.3500 | enhanced_poor_like_ratio_high |
| 1dc4fda1-c351-4f4f-8df8-9bb35eb46062 | 100 | `move_slower` | `repeat_sweep` | 0.5100 | 1.0000 | -0.4900 | 0.4900 | 0.0000 | -0.4900 | enhanced_poor_like_ratio_high |
| 1dccf454-18df-4bfb-b7bf-92acbea3d8e1 | 100 | `keep_probe_steady` | `repeat_sweep` | 0.6200 | 0.9900 | -0.3700 | 0.3800 | 0.0100 | -0.3700 | enhanced_poor_like_ratio_high |
| 1e927a7e-90cd-475b-848d-6dc6368f42ca | 100 | `keep_probe_steady` | `repeat_sweep` | 0.6100 | 1.0000 | -0.3900 | 0.3900 | 0.0000 | -0.3900 | enhanced_poor_like_ratio_high |
| 1f002210-be79-4c02-b948-7c27c5fd2f1e | 100 | `keep_probe_steady` | `repeat_sweep` | 0.5800 | 1.0000 | -0.4200 | 0.4200 | 0.0000 | -0.4200 | enhanced_poor_like_ratio_high |
| 1ff3221d-43d3-4065-9393-347f2d0edac0 | 100 | `keep_probe_steady` | `repeat_sweep` | 0.4400 | 1.0000 | -0.5600 | 0.5600 | 0.0000 | -0.5600 | enhanced_poor_like_ratio_high |
| 20a2b2a1-c91e-4ee0-93b6-c0d8fe2212db | 100 | `repeat_sweep` | `repeat_sweep` | 0.5400 | 1.0000 | -0.4600 | 0.4600 | 0.0000 | -0.4600 | enhanced_poor_like_ratio_high |
| 21127c13-cd7f-456b-ad07-71531d1cb054 | 100 | `repeat_sweep` | `repeat_sweep` | 0.6000 | 1.0000 | -0.4000 | 0.4000 | 0.0000 | -0.4000 | enhanced_poor_like_ratio_high |
| 21883c54-c2ac-4bc3-b557-aa6d41d51087 | 100 | `repeat_sweep` | `repeat_sweep` | 0.6500 | 1.0000 | -0.3500 | 0.3500 | 0.0000 | -0.3500 | enhanced_poor_like_ratio_high |
| 21e6dac4-7606-4abb-9284-d5d9bd0d2b6e | 100 | `move_slower` | `repeat_sweep` | 0.5200 | 1.0000 | -0.4800 | 0.4800 | 0.0000 | -0.4800 | enhanced_poor_like_ratio_high |
| 21ef25cf-842d-4adf-be9a-d07e2e259855 | 100 | `repeat_sweep` | `repeat_sweep` | 0.6400 | 0.9900 | -0.3500 | 0.3600 | 0.0100 | -0.3500 | enhanced_poor_like_ratio_high |
| 229525cd-637a-4560-ad44-45b6f04a5d1f | 100 | `repeat_sweep` | `repeat_sweep` | 0.5800 | 1.0000 | -0.4200 | 0.4200 | 0.0000 | -0.4200 | enhanced_poor_like_ratio_high |
| 23d90415-7355-4677-83aa-a214f6b17c03 | 100 | `move_slower` | `repeat_sweep` | 0.6700 | 1.0000 | -0.3300 | 0.3300 | 0.0000 | -0.3300 | enhanced_poor_like_ratio_high |
| 245dbf25-055e-4dba-8dfb-18cb27b4a880 | 100 | `repeat_sweep` | `repeat_sweep` | 0.5800 | 1.0000 | -0.4200 | 0.4200 | 0.0000 | -0.4200 | enhanced_poor_like_ratio_high |
| 246c5909-ac94-47b1-b343-cb358939e359 | 100 | `repeat_sweep` | `repeat_sweep` | 0.5600 | 1.0000 | -0.4400 | 0.4400 | 0.0000 | -0.4400 | enhanced_poor_like_ratio_high |

## Outputs

- Per-sweep CSV: `results/acouslic_ai/sweep_feedback_change_per_sweep.csv`
- Summary CSV: `results/acouslic_ai/sweep_feedback_change_summary.csv`
- Figure folder: `results/acouslic_ai/sweep_feedback_change_figures`

## Saved figures

- `results/acouslic_ai/sweep_feedback_change_figures/original_sweep_feedback_distribution.png`
- `results/acouslic_ai/sweep_feedback_change_figures/enhanced_sweep_feedback_distribution.png`
- `results/acouslic_ai/sweep_feedback_change_figures/poor_frame_ratio_before_after_per_sweep.png`
- `results/acouslic_ai/sweep_feedback_change_figures/good_frame_ratio_before_after_per_sweep.png`
- `results/acouslic_ai/sweep_feedback_change_figures/example_sweep_001_0f32e98f-8257-4649-a8a2-3cccb2c1a2e5.png`
- `results/acouslic_ai/sweep_feedback_change_figures/example_sweep_002_071ee12c-a8a5-45fd-b191-fd5ec3b00766.png`
- `results/acouslic_ai/sweep_feedback_change_figures/example_sweep_003_23d90415-7355-4677-83aa-a214f6b17c03.png`
- `results/acouslic_ai/sweep_feedback_change_figures/example_sweep_004_1dab0e37-e5e4-41d4-b132-32103b385c78.png`
- `results/acouslic_ai/sweep_feedback_change_figures/example_sweep_005_21883c54-c2ac-4bc3-b557-aa6d41d51087.png`
- `results/acouslic_ai/sweep_feedback_change_figures/example_sweep_006_21ef25cf-842d-4adf-be9a-d07e2e259855.png`
- `results/acouslic_ai/sweep_feedback_change_figures/example_sweep_007_0d0a3298-a9c6-43c3-a9e3-df3a9c0afa06.png`
- `results/acouslic_ai/sweep_feedback_change_figures/example_sweep_008_1d25be5c-d4ba-48bf-afd4-a133581f8491.png`
- `results/acouslic_ai/sweep_feedback_change_figures/example_sweep_009_1dccf454-18df-4bfb-b7bf-92acbea3d8e1.png`
- `results/acouslic_ai/sweep_feedback_change_figures/example_sweep_010_177786dd-a8f4-4d57-ac1b-13811480bd09.png`
