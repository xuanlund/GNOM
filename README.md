# GNOM Nujiang landslide susceptibility

Reproducibility release accompanying the final manuscript. The repository contains the GNOM implementation, leakage-resistant nested five-fold ablation runner, saved out-of-fold (OOF) probabilities, manuscript result tables, and publication figures in PNG and editable SVG.

## Main result

GNOM achieves pooled five-fold OOF ROC-AUC 0.8920 and AP 0.8672. In the model comparison, it has the highest OA, Precision, F1, ROC-AUC and AP; Recall is reported without selective emphasis. In the ablation, the FNO branch supplies the main gain and the full geographic GNOM adds further ranking improvement. The comparison `GNOM` and ablation `Full GNOM` columns are the same 1,532 OOF probabilities.

## Repository layout

- `src/gnom.py`: GNOM architecture, geographic affinity, Fourier gate, risk-field branch and losses.
- `scripts/run_ablation_nested_cv.py`: outer five-fold evaluation with inner validation for epoch selection.
- `scripts/compute_metrics.py`: recomputes the six manuscript metrics from OOF probabilities.
- `scripts/plot_performance_*.py`: regenerates the final PNG/SVG performance figures.
- `scripts/create_susceptibility_maps.py`: optional 30 m OOF-constrained map workflow; requires the source elevation raster described in `data/README.md`.
- `scripts/audit_release.py`: checks row alignment, metrics, figures, and exact GNOM/Full-GNOM identity.
- `predictions/`: authoritative sample-level OOF probabilities.
- `results/`: manuscript tables, fold logs and paired-effect summaries.
- `figures/`: final, non-duplicated figure exports.

## Quick verification

```bash
python -m pip install -r requirements.txt
python scripts/audit_release.py
python scripts/compute_metrics.py
python scripts/plot_performance_box_rose.py
python scripts/plot_performance_roc_pr.py
```

Run the ablation from the cached patches:

```bash
python scripts/run_ablation_nested_cv.py
```

The rerun writes to `results/rerun/` and does not overwrite the archived manuscript OOF files.

## Evaluation protocol

- Five-fold stratified outer CV (`random_state=42`).
- Inner stratified validation split inside each outer-training fold for early-stopping epoch selection.
- Fold-specific raster-channel and tabular normalization; outer test folds are not used for fitting transformations or selecting epochs.
- Fixed decision threshold 0.5.
- Pooled OOF OA, Precision, Recall, F1, ROC-AUC and average precision (AP).
- GNOM loss weights: geographic smoothness 0.08 and field consistency 0.25.

## Data note

The executable data, cached patches, predictions and reported metrics all contain 1,532 samples (766 landslide and 766 non-landslide). The manuscript draft supplied for this release states 561 + 561 in one data-description sentence; that sentence is inconsistent with the deposited analysis and should be corrected before submission. No samples were removed or manufactured to force agreement.

## License and citation

Code is released under the MIT License. Please cite the associated manuscript; bibliographic metadata can be updated in `CITATION.cff` after acceptance.
