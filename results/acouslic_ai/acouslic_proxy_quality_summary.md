# ACOUSLIC-AI Proxy Quality Summary

These labels are heuristic proxy labels for image-quality and sweep-feedback
experiments. They are not clinical ground-truth labels.

- Frames processed: 10000
- Sweeps processed: 100
- Frame manifest: `data/processed/ACOUSLIC_AI/acouslic_proxy_quality_manifest.csv`
- Sweep summary: `data/processed/ACOUSLIC_AI/acouslic_sweep_quality_summary.csv`
- Visual examples: `results/acouslic_ai/proxy_quality_visuals`

## Frame-quality proxy label counts

| Label | Count |
|---|---:|
| `blurry_frame` | 507 |
| `good_frame` | 4531 |
| `low_contrast_frame` | 469 |
| `noisy_frame` | 2462 |
| `poor_frame` | 2031 |

## Final sweep-feedback proxy label counts

| Label | Count |
|---|---:|
| `acceptable` | 2 |
| `keep_probe_steady` | 13 |
| `move_slower` | 25 |
| `repeat_sweep` | 60 |
