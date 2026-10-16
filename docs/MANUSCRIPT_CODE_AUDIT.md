# Manuscript-to-code audit

## Confirmed matches

- Eight raster channels, 32 x 32 sample-centred patches and a two-channel coordinate grid.
- Nine point-wise predictors.
- Local geographic radius 3 pixels (48 non-central offsets).
- GNOM width 24, four Fourier operator blocks and eight retained modes.
- Geographic affinity uses terrain contrast, aspect-direction consistency and lithology similarity.
- Geography-derived low-frequency Fourier gate.
- Auxiliary susceptibility-field head with mean, maximum and standard-deviation statistics.
- Tabular embedding 9 to 48 to 32 and fused classification head 59 to 64 to 1.
- AdamW, cosine schedule, gradient clipping, maximum 55 epochs and five outer folds.
- Weighted BCE plus geographic smoothness (0.08) and field-consistency (0.25) losses.
- Reported OOF metrics and the qualitative comparison/ablation conclusions.

## Corrected release issues

- Removed absolute paths from executable scripts.
- Kept outer test folds evaluation-only; epoch selection uses an inner validation split.
- Consolidated figures and results to one authoritative version.
- Enforced exact identity between comparison `GNOM` and ablation `Full GNOM` predictions.
- Standardized the manuscript metric name to AP (average precision); PR-AUC is not computed by trapezoidal integration.

## Unresolved manuscript text inconsistency

The supplied manuscript states 561 positive plus 561 negative samples. The released CSV, cached tensors, OOF files and all final metrics use 766 plus 766 (1,532 total). Selecting an arbitrary subset would invalidate the reported results, so the package preserves the actual analytical cohort. Correct the manuscript sentence to 766 + 766, or provide the exact 1,122-sample cohort and rerun every experiment.

## Reproducibility scope

The exact manuscript numbers are recomputed from the archived OOF probabilities. Neural retraining is deterministic within the documented software/hardware constraints but small cross-platform numerical variation can occur. Raw environmental rasters are not redistributed; the aligned patch cache is included for model reruns.
