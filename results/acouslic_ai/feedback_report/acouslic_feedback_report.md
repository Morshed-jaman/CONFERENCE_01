# ACOUSLIC-AI Proxy Sweep-Quality Feedback Report

These labels are proxy quality-feedback labels derived from image and motion features, not expert clinical labels.

## Dataset summary

- Total number of frames: 10000
- Total number of sweeps: 100

## Frame-quality label distribution

| Label | Count |
|---|---:|
| `blurry_frame` | 507 |
| `good_frame` | 4531 |
| `low_contrast_frame` | 469 |
| `noisy_frame` | 2462 |
| `poor_frame` | 2031 |

## Sweep-feedback label distribution

| Label | Count |
|---|---:|
| `acceptable` | 2 |
| `keep_probe_steady` | 13 |
| `move_slower` | 25 |
| `repeat_sweep` | 60 |

## Proxy feedback label explanations

| Label | Explanation |
|---|---|
| `good_frame` | The frame has acceptable proxy image quality based on blur, contrast, brightness, noise, and edge features. |
| `blurry_frame` | The frame has a low Laplacian-variance blur score, suggesting weak sharpness. |
| `low_contrast_frame` | The frame has low pixel-intensity variation, suggesting low visible contrast. |
| `noisy_frame` | The frame has a high local difference score, suggesting stronger image noise or speckle-like variation. |
| `poor_frame` | Multiple proxy quality issues were detected in the same frame. |
| `move_slower` | Large frame-to-frame motion was detected; the operator should move the probe more slowly. |
| `keep_probe_steady` | Motion variation suggests unstable probe movement; the operator should keep the probe steadier. |
| `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| `improve_visibility` | Low contrast or weak edge visibility was detected; the operator should adjust probe position or image visibility. |
| `acceptable` | The sweep has acceptable proxy quality based on image and motion features. |

## Per-sweep feedback summary

| Sweep ID | Frames | Poor frame ratio | Mean blur | Mean contrast | Mean edge density | Mean optical flow | Mean jitter | Final feedback | Feedback sentence |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0199616b-bdeb-4119-97a3-a5a3571bd641 | 100 | 0.5300 | 0.005194 | 0.149294 | 0.498712 | 2.693391 | 1.423664 | `move_slower` | Large frame-to-frame motion was detected; the operator should move the probe more slowly. |
| 02d3a9bc-63e2-4deb-9dc1-dba17e7d54c1 | 100 | 0.5600 | 0.005019 | 0.145703 | 0.478830 | 2.782896 | 1.278927 | `move_slower` | Large frame-to-frame motion was detected; the operator should move the probe more slowly. |
| 02ee26a5-a665-4531-bec1-8bac83345a94 | 100 | 0.5500 | 0.004197 | 0.123274 | 0.421484 | 2.800894 | 1.320657 | `move_slower` | Large frame-to-frame motion was detected; the operator should move the probe more slowly. |
| 032b9024-3685-4a81-a62e-813d2da3d387 | 100 | 0.5700 | 0.004880 | 0.137533 | 0.463828 | 2.118488 | 1.031319 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 0428da3d-fef4-4c9b-96f2-5ffedaee82d0 | 100 | 0.5500 | 0.004252 | 0.138682 | 0.468302 | 2.175940 | 1.048103 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 046ed03e-4b35-4519-bb5f-cd4b0474a060 | 100 | 0.4900 | 0.004256 | 0.134140 | 0.441213 | 2.629661 | 1.899516 | `move_slower` | Large frame-to-frame motion was detected; the operator should move the probe more slowly. |
| 04a04f2e-840b-47f8-a907-abe7aeab3f41 | 100 | 0.4700 | 0.004878 | 0.151765 | 0.489969 | 2.339091 | 1.309984 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 04ef1176-7d54-4942-899d-f0fa69ad34bb | 100 | 0.5300 | 0.004464 | 0.134022 | 0.452805 | 2.222019 | 1.067525 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 056764b4-7c0c-4452-832d-3d1cbb5a0464 | 100 | 0.4600 | 0.005554 | 0.161912 | 0.535574 | 2.514590 | 1.203014 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 05f72776-bf41-40ea-bdea-a7bcfa1eb9ba | 100 | 0.3900 | 0.005119 | 0.149167 | 0.485420 | 2.512142 | 1.434823 | `keep_probe_steady` | Motion variation suggests unstable probe movement; the operator should keep the probe steadier. |
| 0607eff0-2268-46b6-8fb8-16e40a521a42 | 100 | 0.5700 | 0.005024 | 0.146004 | 0.505319 | 2.587788 | 1.239289 | `move_slower` | Large frame-to-frame motion was detected; the operator should move the probe more slowly. |
| 0658cdf9-53bc-46a9-ae86-cc99a6fc5e77 | 100 | 0.5700 | 0.004276 | 0.139164 | 0.448816 | 1.953683 | 1.051726 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 0685cfaf-d585-40b3-a365-52f70f2536b6 | 100 | 0.5700 | 0.004640 | 0.138044 | 0.499668 | 2.288331 | 0.962230 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 069c0349-1e8d-42d0-afa1-69c7114a742f | 100 | 0.6200 | 0.004488 | 0.135703 | 0.467029 | 2.304125 | 1.095890 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 071ee12c-a8a5-45fd-b191-fd5ec3b00766 | 100 | 0.6900 | 0.005257 | 0.140571 | 0.491308 | 2.459072 | 1.224547 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 09ddeecb-72e0-417a-85fd-77d397f6219d | 100 | 0.5300 | 0.004753 | 0.138132 | 0.468950 | 2.767215 | 1.158567 | `move_slower` | Large frame-to-frame motion was detected; the operator should move the probe more slowly. |
| 0d0a3298-a9c6-43c3-a9e3-df3a9c0afa06 | 100 | 0.6200 | 0.004160 | 0.125081 | 0.417783 | 2.176867 | 1.254525 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 0d7c4d8f-6e07-4f2b-aa76-8915ce15a38f | 100 | 0.3900 | 0.005371 | 0.162033 | 0.546200 | 2.359701 | 1.152461 | `acceptable` | The sweep has acceptable proxy quality based on image and motion features. |
| 0e2631b6-ed14-4ce4-bc17-896590c406fd | 100 | 0.4200 | 0.004415 | 0.132960 | 0.468538 | 2.202925 | 1.174162 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 0f32e98f-8257-4649-a8a2-3cccb2c1a2e5 | 100 | 0.7100 | 0.004924 | 0.135772 | 0.467785 | 2.367991 | 1.158144 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 1269a6c4-7f04-4927-9308-e4fd68883906 | 100 | 0.5900 | 0.004719 | 0.139281 | 0.469725 | 2.223400 | 1.302014 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 12941223-b27c-47f5-bcb0-7f7bfba18bba | 100 | 0.6200 | 0.005017 | 0.143572 | 0.482641 | 2.723205 | 1.263170 | `move_slower` | Large frame-to-frame motion was detected; the operator should move the probe more slowly. |
| 136db39d-90e4-4702-aee7-dc5cd3169268 | 100 | 0.4800 | 0.005082 | 0.144284 | 0.477609 | 2.265310 | 1.143017 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 1613944c-ea48-4953-84e9-953b9d86f175 | 100 | 0.3900 | 0.004393 | 0.138930 | 0.466960 | 2.347479 | 1.238302 | `acceptable` | The sweep has acceptable proxy quality based on image and motion features. |
| 177786dd-a8f4-4d57-ac1b-13811480bd09 | 100 | 0.6100 | 0.005992 | 0.165471 | 0.553785 | 2.387477 | 1.075198 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 17b31515-75fb-41fc-8908-7c0264d21204 | 100 | 0.4200 | 0.004460 | 0.139983 | 0.482163 | 2.381233 | 1.326525 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 19202c0a-fade-4c7a-9878-e91d823963d6 | 100 | 0.4900 | 0.003866 | 0.131280 | 0.442215 | 1.979449 | 0.870533 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 1a998ef3-797f-4a17-9aa6-96cfb9af87c9 | 100 | 0.5200 | 0.004036 | 0.123640 | 0.415905 | 2.158006 | 1.365266 | `keep_probe_steady` | Motion variation suggests unstable probe movement; the operator should keep the probe steadier. |
| 1af7894a-9f0a-41e8-9c10-a11c0341a93e | 100 | 0.5200 | 0.004611 | 0.137035 | 0.436429 | 2.061210 | 1.137536 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 1b250c39-7217-4921-838b-ed2681ebfef2 | 100 | 0.4000 | 0.004646 | 0.146482 | 0.506567 | 2.868580 | 1.360205 | `move_slower` | Large frame-to-frame motion was detected; the operator should move the probe more slowly. |
| 1b3db504-c1fe-45f0-96dc-3cfb851d5bca | 100 | 0.4200 | 0.004743 | 0.140781 | 0.473761 | 2.614866 | 1.280814 | `move_slower` | Large frame-to-frame motion was detected; the operator should move the probe more slowly. |
| 1c1a1114-3e30-436b-b656-101e6050ea32 | 100 | 0.5900 | 0.005285 | 0.154615 | 0.559823 | 2.318032 | 1.034182 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 1cd004a2-94fb-429a-9e14-fc950bbd8852 | 100 | 0.4400 | 0.004848 | 0.145820 | 0.497712 | 2.376734 | 1.138780 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 1d25be5c-d4ba-48bf-afd4-a133581f8491 | 100 | 0.6300 | 0.005126 | 0.146140 | 0.486786 | 2.251882 | 1.348825 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 1d479820-767b-4d0f-b6cf-12ec5d3a26e7 | 100 | 0.5600 | 0.004456 | 0.130209 | 0.455097 | 2.161746 | 1.113435 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 1dab0e37-e5e4-41d4-b132-32103b385c78 | 100 | 0.6300 | 0.003451 | 0.117698 | 0.368987 | 1.883597 | 1.026116 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 1dc4fda1-c351-4f4f-8df8-9bb35eb46062 | 100 | 0.5200 | 0.005294 | 0.154787 | 0.523767 | 2.644415 | 1.132398 | `move_slower` | Large frame-to-frame motion was detected; the operator should move the probe more slowly. |
| 1dccf454-18df-4bfb-b7bf-92acbea3d8e1 | 100 | 0.5900 | 0.004565 | 0.129669 | 0.432147 | 2.299539 | 1.581008 | `keep_probe_steady` | Motion variation suggests unstable probe movement; the operator should keep the probe steadier. |
| 1e927a7e-90cd-475b-848d-6dc6368f42ca | 100 | 0.6100 | 0.004891 | 0.142234 | 0.471181 | 2.513134 | 1.408368 | `keep_probe_steady` | Motion variation suggests unstable probe movement; the operator should keep the probe steadier. |
| 1f002210-be79-4c02-b948-7c27c5fd2f1e | 100 | 0.5700 | 0.004823 | 0.141376 | 0.471331 | 2.314817 | 1.358862 | `keep_probe_steady` | Motion variation suggests unstable probe movement; the operator should keep the probe steadier. |
| 1ff3221d-43d3-4065-9393-347f2d0edac0 | 100 | 0.4300 | 0.005291 | 0.151667 | 0.524674 | 2.408255 | 1.381948 | `keep_probe_steady` | Motion variation suggests unstable probe movement; the operator should keep the probe steadier. |
| 20a2b2a1-c91e-4ee0-93b6-c0d8fe2212db | 100 | 0.5300 | 0.004083 | 0.128279 | 0.447164 | 2.085160 | 0.942645 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 21127c13-cd7f-456b-ad07-71531d1cb054 | 100 | 0.5800 | 0.004316 | 0.131016 | 0.430979 | 1.667400 | 0.865365 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 21883c54-c2ac-4bc3-b557-aa6d41d51087 | 100 | 0.6600 | 0.005376 | 0.151329 | 0.514406 | 2.387203 | 1.301781 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 21e6dac4-7606-4abb-9284-d5d9bd0d2b6e | 100 | 0.5300 | 0.005240 | 0.160396 | 0.516264 | 2.717738 | 1.270139 | `move_slower` | Large frame-to-frame motion was detected; the operator should move the probe more slowly. |
| 21ef25cf-842d-4adf-be9a-d07e2e259855 | 100 | 0.6300 | 0.004952 | 0.141667 | 0.488691 | 2.386954 | 1.077561 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 229525cd-637a-4560-ad44-45b6f04a5d1f | 100 | 0.5700 | 0.004093 | 0.130716 | 0.399795 | 1.992570 | 0.881080 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 23d90415-7355-4677-83aa-a214f6b17c03 | 100 | 0.6700 | 0.005747 | 0.155978 | 0.526879 | 2.894362 | 1.526179 | `move_slower` | Large frame-to-frame motion was detected; the operator should move the probe more slowly. |
| 245dbf25-055e-4dba-8dfb-18cb27b4a880 | 100 | 0.5900 | 0.004948 | 0.138109 | 0.482438 | 2.458592 | 1.100654 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 246c5909-ac94-47b1-b343-cb358939e359 | 100 | 0.5500 | 0.005001 | 0.137427 | 0.474294 | 2.333431 | 1.286375 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 2594b3cd-e3c2-482d-879b-642f9cdb339d | 100 | 0.5100 | 0.004902 | 0.141726 | 0.466513 | 2.751962 | 1.562641 | `move_slower` | Large frame-to-frame motion was detected; the operator should move the probe more slowly. |
| 2697b349-9abe-4ec3-bb42-96ea329418eb | 100 | 0.5900 | 0.005243 | 0.154098 | 0.538988 | 2.364018 | 1.015062 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 28c99f94-69b1-4e19-95de-2fc228c419cf | 100 | 0.4600 | 0.004482 | 0.139399 | 0.479640 | 1.967023 | 0.969011 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 2a2817fe-e4d8-4fe6-a530-7456c3648c01 | 100 | 0.6800 | 0.004366 | 0.123953 | 0.426481 | 2.111085 | 1.032506 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 2b1767d0-ce44-4093-8713-9b9ca2823ec7 | 100 | 0.4700 | 0.004782 | 0.146501 | 0.497035 | 2.650711 | 1.167228 | `move_slower` | Large frame-to-frame motion was detected; the operator should move the probe more slowly. |
| 2b832993-22e9-4393-a9fe-9296864495ea | 100 | 0.5600 | 0.004656 | 0.139395 | 0.451708 | 2.328035 | 1.051350 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 2c979adb-f7ee-4775-86d8-66ca4169884a | 100 | 0.5200 | 0.004773 | 0.139392 | 0.495307 | 2.230604 | 1.319923 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 2c9934e1-ec1a-49ea-b24f-ff62de1965d1 | 100 | 0.6700 | 0.005116 | 0.144228 | 0.494694 | 2.467037 | 1.243621 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 2cf6450c-a57e-41a8-8918-74218df6e9bf | 100 | 0.3500 | 0.005335 | 0.160510 | 0.556182 | 3.082587 | 1.406406 | `move_slower` | Large frame-to-frame motion was detected; the operator should move the probe more slowly. |
| 2cf8bcb5-3570-474b-a7bc-d86d9c037e08 | 100 | 0.6000 | 0.004482 | 0.138372 | 0.449534 | 1.923680 | 1.009082 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 2dccc77d-0963-4199-8501-04925b3de876 | 100 | 0.6100 | 0.005008 | 0.144853 | 0.498372 | 2.432535 | 1.298387 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 2de0ce54-f9ef-49ec-b4b8-59f304656111 | 100 | 0.5500 | 0.003932 | 0.128207 | 0.416176 | 2.239705 | 1.057133 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 2e99ee0e-506a-443b-8c16-e8a0619da2f7 | 100 | 0.4300 | 0.004190 | 0.143532 | 0.473126 | 2.199524 | 1.139091 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 2fe70922-1d3a-4ce7-b8ad-55a0c07efe74 | 100 | 0.5400 | 0.004354 | 0.134391 | 0.462849 | 1.935765 | 1.370405 | `keep_probe_steady` | Motion variation suggests unstable probe movement; the operator should keep the probe steadier. |
| 300602fc-058e-456b-b872-8e9573428309 | 100 | 0.5100 | 0.004077 | 0.127166 | 0.424629 | 2.573277 | 1.357284 | `move_slower` | Large frame-to-frame motion was detected; the operator should move the probe more slowly. |
| 30555893-2ac6-45c9-b404-8ca2d9f90b05 | 100 | 0.5500 | 0.004251 | 0.129477 | 0.405812 | 1.987290 | 0.831551 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 30d9ee24-7603-46c5-9d05-7d8a6d6a292e | 100 | 0.5400 | 0.004600 | 0.134485 | 0.454424 | 2.541663 | 1.313155 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 33397b96-5fad-441e-9b64-ff3bd5d0b2f9 | 100 | 0.6600 | 0.005310 | 0.145367 | 0.497961 | 2.391829 | 1.050976 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 339c3f53-eaa7-4660-a2bb-b9e1726cd079 | 100 | 0.5500 | 0.004066 | 0.126895 | 0.425645 | 2.175041 | 1.245658 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 3479bfa3-6c07-462b-ab3d-409026598f0f | 100 | 0.5100 | 0.004774 | 0.147503 | 0.485724 | 2.365891 | 1.407960 | `keep_probe_steady` | Motion variation suggests unstable probe movement; the operator should keep the probe steadier. |
| 356d0021-3560-421a-96f4-ecffd4eb6005 | 100 | 0.5200 | 0.005238 | 0.151583 | 0.524052 | 2.347636 | 1.221009 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 36bcc878-7cf1-494c-a67d-7520145cdde2 | 100 | 0.5600 | 0.004272 | 0.130598 | 0.433368 | 2.220569 | 1.097411 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 36d10668-e159-4c9d-811e-b066f4872bda | 100 | 0.5900 | 0.004729 | 0.132669 | 0.455260 | 2.419940 | 1.163208 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 379f256a-879c-4e1d-9b30-a0e9ee51e098 | 100 | 0.5700 | 0.004461 | 0.131067 | 0.457422 | 2.263492 | 1.615726 | `keep_probe_steady` | Motion variation suggests unstable probe movement; the operator should keep the probe steadier. |
| 3825d340-86df-41a7-b415-8c880818ca8e | 100 | 0.4700 | 0.004141 | 0.133840 | 0.459355 | 1.920605 | 0.896777 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 3b3218ac-3616-4aaf-b0ae-b7287803a298 | 100 | 0.5000 | 0.004722 | 0.141477 | 0.484385 | 2.715006 | 1.541361 | `move_slower` | Large frame-to-frame motion was detected; the operator should move the probe more slowly. |
| 3bd64075-6e7e-463d-86e1-19071f04327c | 100 | 0.5800 | 0.005719 | 0.156966 | 0.540378 | 2.734766 | 1.464281 | `move_slower` | Large frame-to-frame motion was detected; the operator should move the probe more slowly. |
| 3d221d3e-749e-4b06-ac7a-762c3fc4c88a | 100 | 0.5800 | 0.004220 | 0.130478 | 0.429903 | 2.423220 | 1.405315 | `keep_probe_steady` | Motion variation suggests unstable probe movement; the operator should keep the probe steadier. |
| 3d24bef2-3cfa-439f-b74f-6ae1e404ab06 | 100 | 0.6100 | 0.005606 | 0.155827 | 0.551786 | 2.888671 | 1.481075 | `move_slower` | Large frame-to-frame motion was detected; the operator should move the probe more slowly. |
| 3d902e8c-626f-46b6-8942-243b3270d929 | 100 | 0.4800 | 0.004784 | 0.137694 | 0.478084 | 2.542724 | 1.847856 | `keep_probe_steady` | Motion variation suggests unstable probe movement; the operator should keep the probe steadier. |
| 3d9ebba3-0eff-4e57-8641-7099ba774574 | 100 | 0.6000 | 0.005401 | 0.158937 | 0.514450 | 2.551011 | 1.310193 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 3daf4e87-136f-4644-9ec7-35d4de3650c9 | 100 | 0.5900 | 0.003814 | 0.125654 | 0.394254 | 1.876559 | 1.008003 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 3e42e61f-a6ec-47ae-bbb7-e84b47170a5f | 100 | 0.4900 | 0.004623 | 0.140352 | 0.450261 | 2.702072 | 1.253586 | `move_slower` | Large frame-to-frame motion was detected; the operator should move the probe more slowly. |
| 40d97ff5-0d2f-4d9c-a6bc-8d07713a86c2 | 100 | 0.5400 | 0.004011 | 0.133412 | 0.442327 | 2.023676 | 1.197698 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 42950f64-500a-46a8-9957-ea690a8292a4 | 100 | 0.6000 | 0.005136 | 0.145923 | 0.495403 | 2.515556 | 1.173791 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 43ccd4c5-0644-402c-8220-d1c04cfc34ec | 100 | 0.4500 | 0.004560 | 0.144384 | 0.481047 | 2.558242 | 1.374889 | `keep_probe_steady` | Motion variation suggests unstable probe movement; the operator should keep the probe steadier. |
| 43f564f8-6772-4341-9083-128fd6eed4c9 | 100 | 0.5200 | 0.004895 | 0.141913 | 0.493500 | 2.023941 | 1.108528 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 4454a0d6-325b-442b-b07a-54f26b52c0b2 | 100 | 0.4900 | 0.004029 | 0.133079 | 0.434458 | 1.804533 | 0.800847 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 44b70504-4e9e-4e02-b13f-6c33808dd7f9 | 100 | 0.7000 | 0.004121 | 0.125723 | 0.445520 | 2.595700 | 1.233136 | `move_slower` | Large frame-to-frame motion was detected; the operator should move the probe more slowly. |
| 4670a2b3-cad0-4d23-afaa-e823438f6944 | 100 | 0.5200 | 0.005554 | 0.162180 | 0.551958 | 2.949628 | 1.260805 | `move_slower` | Large frame-to-frame motion was detected; the operator should move the probe more slowly. |
| 47540baa-7ab8-425a-9e77-8005cb1c02f4 | 100 | 0.4900 | 0.003859 | 0.129620 | 0.435139 | 1.963510 | 0.967001 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 477c0589-f74a-4ecf-8b22-a27776fcbffe | 100 | 0.6300 | 0.004606 | 0.131009 | 0.428442 | 2.569787 | 1.337332 | `move_slower` | Large frame-to-frame motion was detected; the operator should move the probe more slowly. |
| 484c03b0-441d-4966-b10e-06b31d84e55e | 100 | 0.5100 | 0.004497 | 0.139261 | 0.452146 | 2.799275 | 1.421117 | `move_slower` | Large frame-to-frame motion was detected; the operator should move the probe more slowly. |
| 489efa27-f374-4d48-a251-dd2e4d1b09aa | 100 | 0.5000 | 0.004073 | 0.125744 | 0.397557 | 2.015720 | 1.059931 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 4908826a-f3f0-4357-8c8e-8b3d8e6d48c4 | 100 | 0.6400 | 0.004535 | 0.136720 | 0.465740 | 2.288377 | 1.117110 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 4abd109b-71ff-4403-95a9-9a7d3592bf5f | 100 | 0.6600 | 0.005439 | 0.144482 | 0.496900 | 2.716817 | 1.491080 | `move_slower` | Large frame-to-frame motion was detected; the operator should move the probe more slowly. |
| 4acfbc25-cf86-49c1-bb77-689f40db4d34 | 100 | 0.4400 | 0.004777 | 0.144232 | 0.502079 | 2.153901 | 1.148462 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 4adc2ad8-f29d-41e3-9bf6-cc935057ff71 | 100 | 0.6200 | 0.004031 | 0.133001 | 0.432086 | 1.914801 | 0.923624 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |
| 4b6b2a4e-8ad8-4f29-ad47-dad67dfccb6c | 100 | 0.5500 | 0.005187 | 0.150420 | 0.519891 | 2.443843 | 1.394711 | `keep_probe_steady` | Motion variation suggests unstable probe movement; the operator should keep the probe steadier. |
| 4bf11a9a-cb81-4dd4-91d4-5f6a59deb47d | 100 | 0.7000 | 0.004749 | 0.132990 | 0.471866 | 2.190591 | 0.967526 | `repeat_sweep` | The sweep contains many low-quality frames; repeating the sweep is recommended. |

## Saved figures

- Frame-quality distribution: `results/acouslic_ai/feedback_report/frame_quality_distribution.png`
- Sweep-feedback distribution: `results/acouslic_ai/feedback_report/sweep_feedback_distribution.png`
- Timeline folder: `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples`
- Feedback examples folder: `results/acouslic_ai/feedback_report/feedback_examples`

### Timeline figures

- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_0199616b-bdeb-4119-97a3-a5a3571bd641.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_02d3a9bc-63e2-4deb-9dc1-dba17e7d54c1.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_02ee26a5-a665-4531-bec1-8bac83345a94.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_032b9024-3685-4a81-a62e-813d2da3d387.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_0428da3d-fef4-4c9b-96f2-5ffedaee82d0.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_046ed03e-4b35-4519-bb5f-cd4b0474a060.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_04a04f2e-840b-47f8-a907-abe7aeab3f41.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_04ef1176-7d54-4942-899d-f0fa69ad34bb.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_056764b4-7c0c-4452-832d-3d1cbb5a0464.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_05f72776-bf41-40ea-bdea-a7bcfa1eb9ba.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_0607eff0-2268-46b6-8fb8-16e40a521a42.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_0658cdf9-53bc-46a9-ae86-cc99a6fc5e77.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_0685cfaf-d585-40b3-a365-52f70f2536b6.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_069c0349-1e8d-42d0-afa1-69c7114a742f.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_071ee12c-a8a5-45fd-b191-fd5ec3b00766.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_09ddeecb-72e0-417a-85fd-77d397f6219d.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_0d0a3298-a9c6-43c3-a9e3-df3a9c0afa06.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_0d7c4d8f-6e07-4f2b-aa76-8915ce15a38f.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_0e2631b6-ed14-4ce4-bc17-896590c406fd.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_0f32e98f-8257-4649-a8a2-3cccb2c1a2e5.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_1269a6c4-7f04-4927-9308-e4fd68883906.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_12941223-b27c-47f5-bcb0-7f7bfba18bba.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_136db39d-90e4-4702-aee7-dc5cd3169268.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_1613944c-ea48-4953-84e9-953b9d86f175.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_177786dd-a8f4-4d57-ac1b-13811480bd09.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_17b31515-75fb-41fc-8908-7c0264d21204.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_19202c0a-fade-4c7a-9878-e91d823963d6.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_1a998ef3-797f-4a17-9aa6-96cfb9af87c9.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_1af7894a-9f0a-41e8-9c10-a11c0341a93e.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_1b250c39-7217-4921-838b-ed2681ebfef2.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_1b3db504-c1fe-45f0-96dc-3cfb851d5bca.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_1c1a1114-3e30-436b-b656-101e6050ea32.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_1cd004a2-94fb-429a-9e14-fc950bbd8852.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_1d25be5c-d4ba-48bf-afd4-a133581f8491.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_1d479820-767b-4d0f-b6cf-12ec5d3a26e7.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_1dab0e37-e5e4-41d4-b132-32103b385c78.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_1dc4fda1-c351-4f4f-8df8-9bb35eb46062.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_1dccf454-18df-4bfb-b7bf-92acbea3d8e1.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_1e927a7e-90cd-475b-848d-6dc6368f42ca.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_1f002210-be79-4c02-b948-7c27c5fd2f1e.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_1ff3221d-43d3-4065-9393-347f2d0edac0.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_20a2b2a1-c91e-4ee0-93b6-c0d8fe2212db.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_21127c13-cd7f-456b-ad07-71531d1cb054.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_21883c54-c2ac-4bc3-b557-aa6d41d51087.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_21e6dac4-7606-4abb-9284-d5d9bd0d2b6e.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_21ef25cf-842d-4adf-be9a-d07e2e259855.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_229525cd-637a-4560-ad44-45b6f04a5d1f.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_23d90415-7355-4677-83aa-a214f6b17c03.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_245dbf25-055e-4dba-8dfb-18cb27b4a880.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_246c5909-ac94-47b1-b343-cb358939e359.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_2594b3cd-e3c2-482d-879b-642f9cdb339d.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_2697b349-9abe-4ec3-bb42-96ea329418eb.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_28c99f94-69b1-4e19-95de-2fc228c419cf.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_2a2817fe-e4d8-4fe6-a530-7456c3648c01.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_2b1767d0-ce44-4093-8713-9b9ca2823ec7.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_2b832993-22e9-4393-a9fe-9296864495ea.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_2c979adb-f7ee-4775-86d8-66ca4169884a.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_2c9934e1-ec1a-49ea-b24f-ff62de1965d1.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_2cf6450c-a57e-41a8-8918-74218df6e9bf.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_2cf8bcb5-3570-474b-a7bc-d86d9c037e08.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_2dccc77d-0963-4199-8501-04925b3de876.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_2de0ce54-f9ef-49ec-b4b8-59f304656111.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_2e99ee0e-506a-443b-8c16-e8a0619da2f7.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_2fe70922-1d3a-4ce7-b8ad-55a0c07efe74.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_300602fc-058e-456b-b872-8e9573428309.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_30555893-2ac6-45c9-b404-8ca2d9f90b05.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_30d9ee24-7603-46c5-9d05-7d8a6d6a292e.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_33397b96-5fad-441e-9b64-ff3bd5d0b2f9.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_339c3f53-eaa7-4660-a2bb-b9e1726cd079.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_3479bfa3-6c07-462b-ab3d-409026598f0f.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_356d0021-3560-421a-96f4-ecffd4eb6005.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_36bcc878-7cf1-494c-a67d-7520145cdde2.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_36d10668-e159-4c9d-811e-b066f4872bda.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_379f256a-879c-4e1d-9b30-a0e9ee51e098.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_3825d340-86df-41a7-b415-8c880818ca8e.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_3b3218ac-3616-4aaf-b0ae-b7287803a298.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_3bd64075-6e7e-463d-86e1-19071f04327c.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_3d221d3e-749e-4b06-ac7a-762c3fc4c88a.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_3d24bef2-3cfa-439f-b74f-6ae1e404ab06.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_3d902e8c-626f-46b6-8942-243b3270d929.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_3d9ebba3-0eff-4e57-8641-7099ba774574.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_3daf4e87-136f-4644-9ec7-35d4de3650c9.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_3e42e61f-a6ec-47ae-bbb7-e84b47170a5f.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_40d97ff5-0d2f-4d9c-a6bc-8d07713a86c2.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_42950f64-500a-46a8-9957-ea690a8292a4.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_43ccd4c5-0644-402c-8220-d1c04cfc34ec.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_43f564f8-6772-4341-9083-128fd6eed4c9.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_4454a0d6-325b-442b-b07a-54f26b52c0b2.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_44b70504-4e9e-4e02-b13f-6c33808dd7f9.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_4670a2b3-cad0-4d23-afaa-e823438f6944.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_47540baa-7ab8-425a-9e77-8005cb1c02f4.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_477c0589-f74a-4ecf-8b22-a27776fcbffe.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_484c03b0-441d-4966-b10e-06b31d84e55e.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_489efa27-f374-4d48-a251-dd2e4d1b09aa.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_4908826a-f3f0-4357-8c8e-8b3d8e6d48c4.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_4abd109b-71ff-4403-95a9-9a7d3592bf5f.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_4acfbc25-cf86-49c1-bb77-689f40db4d34.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_4adc2ad8-f29d-41e3-9bf6-cc935057ff71.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_4b6b2a4e-8ad8-4f29-ad47-dad67dfccb6c.png`
- `results/acouslic_ai/feedback_report/sweep_quality_timeline_examples/sweep_timeline_4bf11a9a-cb81-4dd4-91d4-5f6a59deb47d.png`

### Feedback example images

- `results/acouslic_ai/feedback_report/feedback_examples/feedback_example_001_0199616b-bdeb-4119-97a3-a5a3571bd641_frame_000001.png`
- `results/acouslic_ai/feedback_report/feedback_examples/feedback_example_002_0199616b-bdeb-4119-97a3-a5a3571bd641_frame_000002.png`
- `results/acouslic_ai/feedback_report/feedback_examples/feedback_example_003_0199616b-bdeb-4119-97a3-a5a3571bd641_frame_000006.png`
- `results/acouslic_ai/feedback_report/feedback_examples/feedback_example_004_0199616b-bdeb-4119-97a3-a5a3571bd641_frame_000007.png`
- `results/acouslic_ai/feedback_report/feedback_examples/feedback_example_005_0199616b-bdeb-4119-97a3-a5a3571bd641_frame_000008.png`
- `results/acouslic_ai/feedback_report/feedback_examples/feedback_example_006_0199616b-bdeb-4119-97a3-a5a3571bd641_frame_000009.png`
- `results/acouslic_ai/feedback_report/feedback_examples/feedback_example_007_0199616b-bdeb-4119-97a3-a5a3571bd641_frame_000010.png`
- `results/acouslic_ai/feedback_report/feedback_examples/feedback_example_008_0199616b-bdeb-4119-97a3-a5a3571bd641_frame_000013.png`
- `results/acouslic_ai/feedback_report/feedback_examples/feedback_example_009_0199616b-bdeb-4119-97a3-a5a3571bd641_frame_000016.png`
- `results/acouslic_ai/feedback_report/feedback_examples/feedback_example_010_0199616b-bdeb-4119-97a3-a5a3571bd641_frame_000017.png`
