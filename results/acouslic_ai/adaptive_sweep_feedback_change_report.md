# ACOUSLIC-AI Adaptive Sweep Feedback Change Analysis

These are proxy feedback labels derived from image and motion features, not expert clinical labels.

This report uses Stage 28 adaptive-frame labels to estimate whether adaptive
super-resolution enhancement changes sweep-level proxy feedback decisions.
Motion-based labels such as `move_slower` and `keep_probe_steady` are preserved,
because image enhancement does not change probe motion.

## Inputs

- Stage 28 adaptive per-frame enhancement CSV: `results/acouslic_ai/adaptive_feedback_enhancement_per_frame.csv`
- Original sweep-quality summary CSV: `data/processed/ACOUSLIC_AI/acouslic_sweep_quality_summary.csv`
- Original frame proxy-quality manifest CSV: `data/processed/ACOUSLIC_AI/acouslic_proxy_quality_manifest.csv`

## Decision rules

- Poor-like frame labels: `poor_frame`, `blurry_frame`, `low_contrast_frame`, `noisy_frame`
- Adaptive `poor_frame_ratio` > 0.40: `repeat_sweep`
- Adaptive `low_contrast_frame` ratio > 0.25: `improve_visibility`
- Original `move_slower` / `keep_probe_steady`: preserved as motion-based feedback
- Otherwise: `acceptable`

## Summary

- Number of sweeps analyzed: 100
- Number of sweeps with reduced poor-frame ratio: 100
- Mean poor-frame-ratio reduction: 0.179400
- Number of sweeps with improved good-frame ratio: 100
- Number of `repeat_sweep` cases changed to `acceptable` or `improve_visibility`: 36
- Number of feedback decisions unchanged due to motion-based problems: 28

## Original sweep-feedback distribution

| Label | Count |
|---|---:|
| `acceptable` | 2 |
| `keep_probe_steady` | 13 |
| `move_slower` | 25 |
| `repeat_sweep` | 60 |

## Adaptive sweep-feedback distribution

| Label | Count |
|---|---:|
| `acceptable` | 38 |
| `keep_probe_steady` | 9 |
| `move_slower` | 19 |
| `repeat_sweep` | 34 |

## Per-sweep results

| Sweep ID | Frames | Original feedback | Adaptive feedback | Poor ratio before | Poor ratio after | Poor ratio reduction | Good ratio before | Good ratio after | Good ratio improvement | Reason |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 0199616b-bdeb-4119-97a3-a5a3571bd641 | 100 | `move_slower` | `move_slower` | 0.5300 | 0.2600 | 0.2700 | 0.4700 | 0.7400 | 0.2700 | unchanged_motion_based_feedback |
| 02d3a9bc-63e2-4deb-9dc1-dba17e7d54c1 | 100 | `move_slower` | `move_slower` | 0.5600 | 0.3400 | 0.2200 | 0.4400 | 0.6600 | 0.2200 | unchanged_motion_based_feedback |
| 02ee26a5-a665-4531-bec1-8bac83345a94 | 100 | `move_slower` | `repeat_sweep` | 0.5500 | 0.4400 | 0.1100 | 0.4500 | 0.5600 | 0.1100 | adaptive_poor_like_ratio_high |
| 032b9024-3685-4a81-a62e-813d2da3d387 | 100 | `repeat_sweep` | `acceptable` | 0.5700 | 0.3800 | 0.1900 | 0.4300 | 0.6200 | 0.1900 | adaptive_visibility_acceptable |
| 0428da3d-fef4-4c9b-96f2-5ffedaee82d0 | 100 | `repeat_sweep` | `repeat_sweep` | 0.5500 | 0.4800 | 0.0700 | 0.4500 | 0.5200 | 0.0700 | adaptive_poor_like_ratio_high |
| 046ed03e-4b35-4519-bb5f-cd4b0474a060 | 100 | `move_slower` | `move_slower` | 0.4900 | 0.3800 | 0.1100 | 0.5100 | 0.6200 | 0.1100 | unchanged_motion_based_feedback |
| 04a04f2e-840b-47f8-a907-abe7aeab3f41 | 100 | `repeat_sweep` | `acceptable` | 0.4700 | 0.3500 | 0.1200 | 0.5300 | 0.6500 | 0.1200 | adaptive_visibility_acceptable |
| 04ef1176-7d54-4942-899d-f0fa69ad34bb | 100 | `repeat_sweep` | `acceptable` | 0.5300 | 0.3500 | 0.1800 | 0.4700 | 0.6500 | 0.1800 | adaptive_visibility_acceptable |
| 056764b4-7c0c-4452-832d-3d1cbb5a0464 | 100 | `repeat_sweep` | `acceptable` | 0.4600 | 0.3100 | 0.1500 | 0.5400 | 0.6900 | 0.1500 | adaptive_visibility_acceptable |
| 05f72776-bf41-40ea-bdea-a7bcfa1eb9ba | 100 | `keep_probe_steady` | `keep_probe_steady` | 0.3900 | 0.2600 | 0.1300 | 0.6100 | 0.7400 | 0.1300 | unchanged_motion_based_feedback |
| 0607eff0-2268-46b6-8fb8-16e40a521a42 | 100 | `move_slower` | `move_slower` | 0.5700 | 0.3200 | 0.2500 | 0.4300 | 0.6800 | 0.2500 | unchanged_motion_based_feedback |
| 0658cdf9-53bc-46a9-ae86-cc99a6fc5e77 | 100 | `repeat_sweep` | `repeat_sweep` | 0.5700 | 0.4400 | 0.1300 | 0.4300 | 0.5600 | 0.1300 | adaptive_poor_like_ratio_high |
| 0685cfaf-d585-40b3-a365-52f70f2536b6 | 100 | `repeat_sweep` | `repeat_sweep` | 0.5700 | 0.4100 | 0.1600 | 0.4300 | 0.5900 | 0.1600 | adaptive_poor_like_ratio_high |
| 069c0349-1e8d-42d0-afa1-69c7114a742f | 100 | `repeat_sweep` | `acceptable` | 0.6200 | 0.3800 | 0.2400 | 0.3800 | 0.6200 | 0.2400 | adaptive_visibility_acceptable |
| 071ee12c-a8a5-45fd-b191-fd5ec3b00766 | 100 | `repeat_sweep` | `acceptable` | 0.6900 | 0.4000 | 0.2900 | 0.3100 | 0.6000 | 0.2900 | adaptive_visibility_acceptable |
| 09ddeecb-72e0-417a-85fd-77d397f6219d | 100 | `move_slower` | `move_slower` | 0.5300 | 0.3100 | 0.2200 | 0.4700 | 0.6900 | 0.2200 | unchanged_motion_based_feedback |
| 0d0a3298-a9c6-43c3-a9e3-df3a9c0afa06 | 100 | `repeat_sweep` | `repeat_sweep` | 0.6200 | 0.5000 | 0.1200 | 0.3800 | 0.5000 | 0.1200 | adaptive_poor_like_ratio_high |
| 0d7c4d8f-6e07-4f2b-aa76-8915ce15a38f | 100 | `acceptable` | `acceptable` | 0.3900 | 0.2300 | 0.1600 | 0.6100 | 0.7700 | 0.1600 | adaptive_visibility_acceptable |
| 0e2631b6-ed14-4ce4-bc17-896590c406fd | 100 | `repeat_sweep` | `acceptable` | 0.4200 | 0.3400 | 0.0800 | 0.5800 | 0.6600 | 0.0800 | adaptive_visibility_acceptable |
| 0f32e98f-8257-4649-a8a2-3cccb2c1a2e5 | 100 | `repeat_sweep` | `repeat_sweep` | 0.7100 | 0.4400 | 0.2700 | 0.2900 | 0.5600 | 0.2700 | adaptive_poor_like_ratio_high |
| 1269a6c4-7f04-4927-9308-e4fd68883906 | 100 | `repeat_sweep` | `acceptable` | 0.5900 | 0.4000 | 0.1900 | 0.4100 | 0.6000 | 0.1900 | adaptive_visibility_acceptable |
| 12941223-b27c-47f5-bcb0-7f7bfba18bba | 100 | `move_slower` | `move_slower` | 0.6200 | 0.3900 | 0.2300 | 0.3800 | 0.6100 | 0.2300 | unchanged_motion_based_feedback |
| 136db39d-90e4-4702-aee7-dc5cd3169268 | 100 | `repeat_sweep` | `acceptable` | 0.4800 | 0.3700 | 0.1100 | 0.5200 | 0.6300 | 0.1100 | adaptive_visibility_acceptable |
| 1613944c-ea48-4953-84e9-953b9d86f175 | 100 | `acceptable` | `acceptable` | 0.3900 | 0.3200 | 0.0700 | 0.6100 | 0.6800 | 0.0700 | adaptive_visibility_acceptable |
| 177786dd-a8f4-4d57-ac1b-13811480bd09 | 100 | `repeat_sweep` | `acceptable` | 0.6100 | 0.2200 | 0.3900 | 0.3900 | 0.7800 | 0.3900 | adaptive_visibility_acceptable |
| 17b31515-75fb-41fc-8908-7c0264d21204 | 100 | `repeat_sweep` | `acceptable` | 0.4200 | 0.3100 | 0.1100 | 0.5800 | 0.6900 | 0.1100 | adaptive_visibility_acceptable |
| 19202c0a-fade-4c7a-9878-e91d823963d6 | 100 | `repeat_sweep` | `repeat_sweep` | 0.4900 | 0.4400 | 0.0500 | 0.5100 | 0.5600 | 0.0500 | adaptive_poor_like_ratio_high |
| 1a998ef3-797f-4a17-9aa6-96cfb9af87c9 | 100 | `keep_probe_steady` | `repeat_sweep` | 0.5200 | 0.4400 | 0.0800 | 0.4800 | 0.5600 | 0.0800 | adaptive_poor_like_ratio_high |
| 1af7894a-9f0a-41e8-9c10-a11c0341a93e | 100 | `repeat_sweep` | `acceptable` | 0.5200 | 0.3700 | 0.1500 | 0.4800 | 0.6300 | 0.1500 | adaptive_visibility_acceptable |
| 1b250c39-7217-4921-838b-ed2681ebfef2 | 100 | `move_slower` | `move_slower` | 0.4000 | 0.2400 | 0.1600 | 0.6000 | 0.7600 | 0.1600 | unchanged_motion_based_feedback |
| 1b3db504-c1fe-45f0-96dc-3cfb851d5bca | 100 | `move_slower` | `move_slower` | 0.4200 | 0.3000 | 0.1200 | 0.5800 | 0.7000 | 0.1200 | unchanged_motion_based_feedback |
| 1c1a1114-3e30-436b-b656-101e6050ea32 | 100 | `repeat_sweep` | `acceptable` | 0.5900 | 0.2900 | 0.3000 | 0.4100 | 0.7100 | 0.3000 | adaptive_visibility_acceptable |
| 1cd004a2-94fb-429a-9e14-fc950bbd8852 | 100 | `repeat_sweep` | `acceptable` | 0.4400 | 0.2800 | 0.1600 | 0.5600 | 0.7200 | 0.1600 | adaptive_visibility_acceptable |
| 1d25be5c-d4ba-48bf-afd4-a133581f8491 | 100 | `repeat_sweep` | `repeat_sweep` | 0.6300 | 0.4300 | 0.2000 | 0.3700 | 0.5700 | 0.2000 | adaptive_poor_like_ratio_high |
| 1d479820-767b-4d0f-b6cf-12ec5d3a26e7 | 100 | `repeat_sweep` | `acceptable` | 0.5600 | 0.3600 | 0.2000 | 0.4400 | 0.6400 | 0.2000 | adaptive_visibility_acceptable |
| 1dab0e37-e5e4-41d4-b132-32103b385c78 | 100 | `repeat_sweep` | `repeat_sweep` | 0.6300 | 0.6000 | 0.0300 | 0.3700 | 0.4000 | 0.0300 | adaptive_poor_like_ratio_high |
| 1dc4fda1-c351-4f4f-8df8-9bb35eb46062 | 100 | `move_slower` | `move_slower` | 0.5200 | 0.2600 | 0.2600 | 0.4800 | 0.7400 | 0.2600 | unchanged_motion_based_feedback |
| 1dccf454-18df-4bfb-b7bf-92acbea3d8e1 | 100 | `keep_probe_steady` | `repeat_sweep` | 0.5900 | 0.4200 | 0.1700 | 0.4100 | 0.5800 | 0.1700 | adaptive_poor_like_ratio_high |
| 1e927a7e-90cd-475b-848d-6dc6368f42ca | 100 | `keep_probe_steady` | `keep_probe_steady` | 0.6100 | 0.3300 | 0.2800 | 0.3900 | 0.6700 | 0.2800 | unchanged_motion_based_feedback |
| 1f002210-be79-4c02-b948-7c27c5fd2f1e | 100 | `keep_probe_steady` | `keep_probe_steady` | 0.5700 | 0.3500 | 0.2200 | 0.4300 | 0.6500 | 0.2200 | unchanged_motion_based_feedback |
| 1ff3221d-43d3-4065-9393-347f2d0edac0 | 100 | `keep_probe_steady` | `keep_probe_steady` | 0.4300 | 0.2000 | 0.2300 | 0.5700 | 0.8000 | 0.2300 | unchanged_motion_based_feedback |
| 20a2b2a1-c91e-4ee0-93b6-c0d8fe2212db | 100 | `repeat_sweep` | `repeat_sweep` | 0.5300 | 0.4200 | 0.1100 | 0.4700 | 0.5800 | 0.1100 | adaptive_poor_like_ratio_high |
| 21127c13-cd7f-456b-ad07-71531d1cb054 | 100 | `repeat_sweep` | `repeat_sweep` | 0.5800 | 0.4600 | 0.1200 | 0.4200 | 0.5400 | 0.1200 | adaptive_poor_like_ratio_high |
| 21883c54-c2ac-4bc3-b557-aa6d41d51087 | 100 | `repeat_sweep` | `acceptable` | 0.6600 | 0.3200 | 0.3400 | 0.3400 | 0.6800 | 0.3400 | adaptive_visibility_acceptable |
| 21e6dac4-7606-4abb-9284-d5d9bd0d2b6e | 100 | `move_slower` | `move_slower` | 0.5300 | 0.2000 | 0.3300 | 0.4700 | 0.8000 | 0.3300 | unchanged_motion_based_feedback |
| 21ef25cf-842d-4adf-be9a-d07e2e259855 | 100 | `repeat_sweep` | `acceptable` | 0.6300 | 0.3400 | 0.2900 | 0.3700 | 0.6600 | 0.2900 | adaptive_visibility_acceptable |
| 229525cd-637a-4560-ad44-45b6f04a5d1f | 100 | `repeat_sweep` | `repeat_sweep` | 0.5700 | 0.4700 | 0.1000 | 0.4300 | 0.5300 | 0.1000 | adaptive_poor_like_ratio_high |
| 23d90415-7355-4677-83aa-a214f6b17c03 | 100 | `move_slower` | `repeat_sweep` | 0.6700 | 0.4200 | 0.2500 | 0.3300 | 0.5800 | 0.2500 | adaptive_poor_like_ratio_high |
| 245dbf25-055e-4dba-8dfb-18cb27b4a880 | 100 | `repeat_sweep` | `acceptable` | 0.5900 | 0.4000 | 0.1900 | 0.4100 | 0.6000 | 0.1900 | adaptive_visibility_acceptable |
| 246c5909-ac94-47b1-b343-cb358939e359 | 100 | `repeat_sweep` | `repeat_sweep` | 0.5500 | 0.4100 | 0.1400 | 0.4500 | 0.5900 | 0.1400 | adaptive_poor_like_ratio_high |
| 2594b3cd-e3c2-482d-879b-642f9cdb339d | 100 | `move_slower` | `move_slower` | 0.5100 | 0.3600 | 0.1500 | 0.4900 | 0.6400 | 0.1500 | unchanged_motion_based_feedback |
| 2697b349-9abe-4ec3-bb42-96ea329418eb | 100 | `repeat_sweep` | `acceptable` | 0.5900 | 0.3800 | 0.2100 | 0.4100 | 0.6200 | 0.2100 | adaptive_visibility_acceptable |
| 28c99f94-69b1-4e19-95de-2fc228c419cf | 100 | `repeat_sweep` | `acceptable` | 0.4600 | 0.3100 | 0.1500 | 0.5400 | 0.6900 | 0.1500 | adaptive_visibility_acceptable |
| 2a2817fe-e4d8-4fe6-a530-7456c3648c01 | 100 | `repeat_sweep` | `repeat_sweep` | 0.6800 | 0.5000 | 0.1800 | 0.3200 | 0.5000 | 0.1800 | adaptive_poor_like_ratio_high |
| 2b1767d0-ce44-4093-8713-9b9ca2823ec7 | 100 | `move_slower` | `move_slower` | 0.4700 | 0.3800 | 0.0900 | 0.5300 | 0.6200 | 0.0900 | unchanged_motion_based_feedback |
| 2b832993-22e9-4393-a9fe-9296864495ea | 100 | `repeat_sweep` | `acceptable` | 0.5600 | 0.3800 | 0.1800 | 0.4400 | 0.6200 | 0.1800 | adaptive_visibility_acceptable |
| 2c979adb-f7ee-4775-86d8-66ca4169884a | 100 | `repeat_sweep` | `acceptable` | 0.5200 | 0.3000 | 0.2200 | 0.4800 | 0.7000 | 0.2200 | adaptive_visibility_acceptable |
| 2c9934e1-ec1a-49ea-b24f-ff62de1965d1 | 100 | `repeat_sweep` | `acceptable` | 0.6700 | 0.3200 | 0.3500 | 0.3300 | 0.6800 | 0.3500 | adaptive_visibility_acceptable |
| 2cf6450c-a57e-41a8-8918-74218df6e9bf | 100 | `move_slower` | `move_slower` | 0.3500 | 0.1400 | 0.2100 | 0.6500 | 0.8600 | 0.2100 | unchanged_motion_based_feedback |
| 2cf8bcb5-3570-474b-a7bc-d86d9c037e08 | 100 | `repeat_sweep` | `acceptable` | 0.6000 | 0.4000 | 0.2000 | 0.4000 | 0.6000 | 0.2000 | adaptive_visibility_acceptable |
| 2dccc77d-0963-4199-8501-04925b3de876 | 100 | `repeat_sweep` | `acceptable` | 0.6100 | 0.3000 | 0.3100 | 0.3900 | 0.7000 | 0.3100 | adaptive_visibility_acceptable |
| 2de0ce54-f9ef-49ec-b4b8-59f304656111 | 100 | `repeat_sweep` | `repeat_sweep` | 0.5500 | 0.4600 | 0.0900 | 0.4500 | 0.5400 | 0.0900 | adaptive_poor_like_ratio_high |
| 2e99ee0e-506a-443b-8c16-e8a0619da2f7 | 100 | `repeat_sweep` | `acceptable` | 0.4300 | 0.3800 | 0.0500 | 0.5700 | 0.6200 | 0.0500 | adaptive_visibility_acceptable |
| 2fe70922-1d3a-4ce7-b8ad-55a0c07efe74 | 100 | `keep_probe_steady` | `repeat_sweep` | 0.5400 | 0.4400 | 0.1000 | 0.4600 | 0.5600 | 0.1000 | adaptive_poor_like_ratio_high |
| 300602fc-058e-456b-b872-8e9573428309 | 100 | `move_slower` | `repeat_sweep` | 0.5100 | 0.4400 | 0.0700 | 0.4900 | 0.5600 | 0.0700 | adaptive_poor_like_ratio_high |
| 30555893-2ac6-45c9-b404-8ca2d9f90b05 | 100 | `repeat_sweep` | `acceptable` | 0.5500 | 0.4000 | 0.1500 | 0.4500 | 0.6000 | 0.1500 | adaptive_visibility_acceptable |
| 30d9ee24-7603-46c5-9d05-7d8a6d6a292e | 100 | `repeat_sweep` | `acceptable` | 0.5400 | 0.3600 | 0.1800 | 0.4600 | 0.6400 | 0.1800 | adaptive_visibility_acceptable |
| 33397b96-5fad-441e-9b64-ff3bd5d0b2f9 | 100 | `repeat_sweep` | `acceptable` | 0.6600 | 0.3200 | 0.3400 | 0.3400 | 0.6800 | 0.3400 | adaptive_visibility_acceptable |
| 339c3f53-eaa7-4660-a2bb-b9e1726cd079 | 100 | `repeat_sweep` | `repeat_sweep` | 0.5500 | 0.4200 | 0.1300 | 0.4500 | 0.5800 | 0.1300 | adaptive_poor_like_ratio_high |
| 3479bfa3-6c07-462b-ab3d-409026598f0f | 100 | `keep_probe_steady` | `keep_probe_steady` | 0.5100 | 0.3300 | 0.1800 | 0.4900 | 0.6700 | 0.1800 | unchanged_motion_based_feedback |
| 356d0021-3560-421a-96f4-ecffd4eb6005 | 100 | `repeat_sweep` | `acceptable` | 0.5200 | 0.2300 | 0.2900 | 0.4800 | 0.7700 | 0.2900 | adaptive_visibility_acceptable |
| 36bcc878-7cf1-494c-a67d-7520145cdde2 | 100 | `repeat_sweep` | `repeat_sweep` | 0.5600 | 0.4200 | 0.1400 | 0.4400 | 0.5800 | 0.1400 | adaptive_poor_like_ratio_high |
| 36d10668-e159-4c9d-811e-b066f4872bda | 100 | `repeat_sweep` | `acceptable` | 0.5900 | 0.3800 | 0.2100 | 0.4100 | 0.6200 | 0.2100 | adaptive_visibility_acceptable |
| 379f256a-879c-4e1d-9b30-a0e9ee51e098 | 100 | `keep_probe_steady` | `repeat_sweep` | 0.5700 | 0.4100 | 0.1600 | 0.4300 | 0.5900 | 0.1600 | adaptive_poor_like_ratio_high |
| 3825d340-86df-41a7-b415-8c880818ca8e | 100 | `repeat_sweep` | `acceptable` | 0.4700 | 0.3900 | 0.0800 | 0.5300 | 0.6100 | 0.0800 | adaptive_visibility_acceptable |
| 3b3218ac-3616-4aaf-b0ae-b7287803a298 | 100 | `move_slower` | `move_slower` | 0.5000 | 0.3200 | 0.1800 | 0.5000 | 0.6800 | 0.1800 | unchanged_motion_based_feedback |
| 3bd64075-6e7e-463d-86e1-19071f04327c | 100 | `move_slower` | `move_slower` | 0.5800 | 0.3600 | 0.2200 | 0.4200 | 0.6400 | 0.2200 | unchanged_motion_based_feedback |
| 3d221d3e-749e-4b06-ac7a-762c3fc4c88a | 100 | `keep_probe_steady` | `keep_probe_steady` | 0.5800 | 0.4000 | 0.1800 | 0.4200 | 0.6000 | 0.1800 | unchanged_motion_based_feedback |
| 3d24bef2-3cfa-439f-b74f-6ae1e404ab06 | 100 | `move_slower` | `move_slower` | 0.6100 | 0.2100 | 0.4000 | 0.3900 | 0.7900 | 0.4000 | unchanged_motion_based_feedback |
| 3d902e8c-626f-46b6-8942-243b3270d929 | 100 | `keep_probe_steady` | `keep_probe_steady` | 0.4800 | 0.3000 | 0.1800 | 0.5200 | 0.7000 | 0.1800 | unchanged_motion_based_feedback |
| 3d9ebba3-0eff-4e57-8641-7099ba774574 | 100 | `repeat_sweep` | `acceptable` | 0.6000 | 0.2600 | 0.3400 | 0.4000 | 0.7400 | 0.3400 | adaptive_visibility_acceptable |
| 3daf4e87-136f-4644-9ec7-35d4de3650c9 | 100 | `repeat_sweep` | `repeat_sweep` | 0.5900 | 0.5400 | 0.0500 | 0.4100 | 0.4600 | 0.0500 | adaptive_poor_like_ratio_high |
| 3e42e61f-a6ec-47ae-bbb7-e84b47170a5f | 100 | `move_slower` | `move_slower` | 0.4900 | 0.3700 | 0.1200 | 0.5100 | 0.6300 | 0.1200 | unchanged_motion_based_feedback |
| 40d97ff5-0d2f-4d9c-a6bc-8d07713a86c2 | 100 | `repeat_sweep` | `repeat_sweep` | 0.5400 | 0.4600 | 0.0800 | 0.4600 | 0.5400 | 0.0800 | adaptive_poor_like_ratio_high |
| 42950f64-500a-46a8-9957-ea690a8292a4 | 100 | `repeat_sweep` | `acceptable` | 0.6000 | 0.3400 | 0.2600 | 0.4000 | 0.6600 | 0.2600 | adaptive_visibility_acceptable |
| 43ccd4c5-0644-402c-8220-d1c04cfc34ec | 100 | `keep_probe_steady` | `keep_probe_steady` | 0.4500 | 0.3400 | 0.1100 | 0.5500 | 0.6600 | 0.1100 | unchanged_motion_based_feedback |
| 43f564f8-6772-4341-9083-128fd6eed4c9 | 100 | `repeat_sweep` | `acceptable` | 0.5200 | 0.2400 | 0.2800 | 0.4800 | 0.7600 | 0.2800 | adaptive_visibility_acceptable |
| 4454a0d6-325b-442b-b07a-54f26b52c0b2 | 100 | `repeat_sweep` | `repeat_sweep` | 0.4900 | 0.4400 | 0.0500 | 0.5100 | 0.5600 | 0.0500 | adaptive_poor_like_ratio_high |
| 44b70504-4e9e-4e02-b13f-6c33808dd7f9 | 100 | `move_slower` | `repeat_sweep` | 0.7000 | 0.5900 | 0.1100 | 0.3000 | 0.4100 | 0.1100 | adaptive_poor_like_ratio_high |
| 4670a2b3-cad0-4d23-afaa-e823438f6944 | 100 | `move_slower` | `move_slower` | 0.5200 | 0.2400 | 0.2800 | 0.4800 | 0.7600 | 0.2800 | unchanged_motion_based_feedback |
| 47540baa-7ab8-425a-9e77-8005cb1c02f4 | 100 | `repeat_sweep` | `repeat_sweep` | 0.4900 | 0.4700 | 0.0200 | 0.5100 | 0.5300 | 0.0200 | adaptive_poor_like_ratio_high |
| 477c0589-f74a-4ecf-8b22-a27776fcbffe | 100 | `move_slower` | `repeat_sweep` | 0.6300 | 0.4400 | 0.1900 | 0.3700 | 0.5600 | 0.1900 | adaptive_poor_like_ratio_high |
| 484c03b0-441d-4966-b10e-06b31d84e55e | 100 | `move_slower` | `move_slower` | 0.5100 | 0.3800 | 0.1300 | 0.4900 | 0.6200 | 0.1300 | unchanged_motion_based_feedback |
| 489efa27-f374-4d48-a251-dd2e4d1b09aa | 100 | `repeat_sweep` | `repeat_sweep` | 0.5000 | 0.4300 | 0.0700 | 0.5000 | 0.5700 | 0.0700 | adaptive_poor_like_ratio_high |
| 4908826a-f3f0-4357-8c8e-8b3d8e6d48c4 | 100 | `repeat_sweep` | `repeat_sweep` | 0.6400 | 0.4900 | 0.1500 | 0.3600 | 0.5100 | 0.1500 | adaptive_poor_like_ratio_high |
| 4abd109b-71ff-4403-95a9-9a7d3592bf5f | 100 | `move_slower` | `repeat_sweep` | 0.6600 | 0.4300 | 0.2300 | 0.3400 | 0.5700 | 0.2300 | adaptive_poor_like_ratio_high |
| 4acfbc25-cf86-49c1-bb77-689f40db4d34 | 100 | `repeat_sweep` | `acceptable` | 0.4400 | 0.2400 | 0.2000 | 0.5600 | 0.7600 | 0.2000 | adaptive_visibility_acceptable |
| 4adc2ad8-f29d-41e3-9bf6-cc935057ff71 | 100 | `repeat_sweep` | `repeat_sweep` | 0.6200 | 0.4800 | 0.1400 | 0.3800 | 0.5200 | 0.1400 | adaptive_poor_like_ratio_high |
| 4b6b2a4e-8ad8-4f29-ad47-dad67dfccb6c | 100 | `keep_probe_steady` | `keep_probe_steady` | 0.5500 | 0.2600 | 0.2900 | 0.4500 | 0.7400 | 0.2900 | unchanged_motion_based_feedback |
| 4bf11a9a-cb81-4dd4-91d4-5f6a59deb47d | 100 | `repeat_sweep` | `repeat_sweep` | 0.7000 | 0.4900 | 0.2100 | 0.3000 | 0.5100 | 0.2100 | adaptive_poor_like_ratio_high |

## Outputs

- Per-sweep CSV: `results/acouslic_ai/adaptive_sweep_feedback_change_per_sweep.csv`
- Summary CSV: `results/acouslic_ai/adaptive_sweep_feedback_change_summary.csv`
- Figure folder: `results/acouslic_ai/adaptive_sweep_feedback_change_figures`

## Saved figures

- `results/acouslic_ai/adaptive_sweep_feedback_change_figures/original_sweep_feedback_distribution.png`
- `results/acouslic_ai/adaptive_sweep_feedback_change_figures/adaptive_sweep_feedback_distribution.png`
- `results/acouslic_ai/adaptive_sweep_feedback_change_figures/poor_frame_ratio_before_after_per_sweep.png`
- `results/acouslic_ai/adaptive_sweep_feedback_change_figures/good_frame_ratio_before_after_per_sweep.png`
- `results/acouslic_ai/adaptive_sweep_feedback_change_figures/example_sweep_001_3d24bef2-3cfa-439f-b74f-6ae1e404ab06.png`
- `results/acouslic_ai/adaptive_sweep_feedback_change_figures/example_sweep_002_177786dd-a8f4-4d57-ac1b-13811480bd09.png`
- `results/acouslic_ai/adaptive_sweep_feedback_change_figures/example_sweep_003_2c9934e1-ec1a-49ea-b24f-ff62de1965d1.png`
- `results/acouslic_ai/adaptive_sweep_feedback_change_figures/example_sweep_004_21883c54-c2ac-4bc3-b557-aa6d41d51087.png`
- `results/acouslic_ai/adaptive_sweep_feedback_change_figures/example_sweep_005_33397b96-5fad-441e-9b64-ff3bd5d0b2f9.png`
- `results/acouslic_ai/adaptive_sweep_feedback_change_figures/example_sweep_006_3d9ebba3-0eff-4e57-8641-7099ba774574.png`
- `results/acouslic_ai/adaptive_sweep_feedback_change_figures/example_sweep_007_21e6dac4-7606-4abb-9284-d5d9bd0d2b6e.png`
- `results/acouslic_ai/adaptive_sweep_feedback_change_figures/example_sweep_008_2dccc77d-0963-4199-8501-04925b3de876.png`
- `results/acouslic_ai/adaptive_sweep_feedback_change_figures/example_sweep_009_1c1a1114-3e30-436b-b656-101e6050ea32.png`
- `results/acouslic_ai/adaptive_sweep_feedback_change_figures/example_sweep_010_356d0021-3560-421a-96f4-ecffd4eb6005.png`
