# Novelty Comparison Table

| Method / research direction | Diffusion SR | Quality estimation | Sweep-quality analysis | Adaptive enhancement decision | Feedback reassessment loop | Anatomy preservation evaluation | Novice feedback | Main limitation |
|---|---|---|---|---|---|---|---|---|
| DMUISR-guided diffusion SR | Yes | Limited / implicit | No | No | No | Not typically fetal-anatomy specific | No | Focuses on image restoration; does not model fetal ultrasound acquisition quality or sweep-level feedback. |
| Fetal ultrasound SR methods | Sometimes | Usually no | No | No | No | Sometimes, depending on study | No | Improve image resolution but usually do not decide when enhancement is appropriate or link outputs to acquisition feedback. |
| Fetal ultrasound IQA methods | No | Yes | Usually frame/image-level | No | No | Rarely | Limited | Estimate image quality but generally do not perform adaptive restoration or reassess sweep feedback after enhancement. |
| ACOUSLIC-AI blind-sweep QA | No | Yes | Yes | No | Limited | No | Proxy / offline | Supports sweep-quality analysis, but does not integrate adaptive SR enhancement or anatomy-preservation evaluation. |
| Probe guidance methods | No | Sometimes | Sometimes | No | Sometimes | No | Yes | Provide acquisition guidance, but are usually not coupled with learned SR/restoration and post-enhancement quality reassessment. |
| Proposed framework | Yes | Yes | Yes | Yes | Yes | Yes | Proxy / offline | Uses proxy feedback rather than expert clinical feedback; current analysis is offline and not real-time clinical deployment. |

Note: The proposed framework is DMUISR-guided rather than an exact DMUISR implementation. Feedback labels are proxy labels derived from image/sweep features, not expert clinical annotations.
