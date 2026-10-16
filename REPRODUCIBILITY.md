# Reproducibility

1. Create a Python 3.10+ environment and install `requirements.txt`.
2. Run `python scripts/audit_release.py` to validate the deposited release.
3. Run `python scripts/compute_metrics.py` to recreate the tabulated metrics.
4. Run both `plot_performance` scripts to recreate all performance PNG/SVG files.
5. Optionally run `python scripts/run_ablation_nested_cv.py` for a full training rerun.

The archived OOF predictions are the source data for the manuscript tables and curves. A fixed threshold of 0.5 is used for OA, Precision, Recall and F1. ROC-AUC and AP use continuous OOF probabilities. Each sample appears in exactly one held-out outer fold.
