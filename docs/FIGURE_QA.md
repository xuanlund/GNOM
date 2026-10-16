# Figure QA

- Backend: Python/matplotlib only.
- Source data: complete pooled OOF predictions; no rows or metrics excluded.
- Export: PNG at 600 dpi and SVG with editable text.
- Palette: low-saturation grey/blue baselines; orange GNOM/Full GNOM highlight.
- ROC and precision-recall curves use continuous probabilities and thin solid lines; markers are not used.
- Box plots show the five held-out-fold ROC-AUC values and individual fold points.
- Rose charts use a declared radial baseline of 0.40; underlying values are not transformed and are available in CSV.
- Box-plot axis is 0.80 to 0.95, explicitly labelled and accompanied by raw fold points. It is used for fold-distribution readability, not as an effect-size substitute.
- Comparison GNOM and ablation Full GNOM are identical predictions and therefore have identical pooled metrics.
- Visual inspection: no clipped labels, obscured legends, rasterized SVG text, or inconsistent model order detected.
