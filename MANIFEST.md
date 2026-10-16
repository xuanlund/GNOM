# Release manifest

## Core code

- `src/gnom.py` — GNOM and FNO ablation implementations.
- `src/baselines.py` — MLP, GWR-logit and GNNWLR definitions.
- `scripts/run_ablation_nested_cv.py` — leakage-resistant training/evaluation runner.
- `scripts/compute_metrics.py` — table regeneration.
- `scripts/plot_performance_box_rose.py` and `plot_performance_roc_pr.py` — final figures.
- `scripts/create_susceptibility_maps.py` — optional 30 m map generation.
- `scripts/audit_release.py` — release consistency tests.

## Deposited evidence

- One sample table and one aligned 8-channel patch cache.
- Two authoritative OOF prediction tables.
- Comparison and ablation metrics, paired effects, fold training log and protocol.
- Four final figure compositions, each in PNG and SVG.
- One manuscript framework figure in PNG and editable SVG.

No backup figures, temporary files, Python bytecode, or superseded metric tables are part of this release.
