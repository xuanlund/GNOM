"""Fail-fast audit for the manuscript release bundle."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
TOL = 5e-10

def metrics(y, p):
    z = p >= .5
    return np.array([accuracy_score(y,z), precision_score(y,z), recall_score(y,z),
                     f1_score(y,z), roc_auc_score(y,p), average_precision_score(y,p)])

def main():
    samples = pd.read_csv(ROOT/'data'/'nujiang_samples.csv', encoding='utf-8-sig')
    comp = pd.read_csv(ROOT/'predictions'/'comparison_oof_predictions.csv', encoding='utf-8-sig')
    abl = pd.read_csv(ROOT/'predictions'/'ablation_oof_predictions.csv', encoding='utf-8-sig')
    assert len(samples) == len(comp) == len(abl) == 1532
    assert samples.label.value_counts().to_dict() == {0: 766, 1: 766}
    assert np.array_equal(comp.label.to_numpy(), abl.label.to_numpy())
    assert np.allclose(comp.GNOM, abl['Full GNOM'], atol=0, rtol=0)
    for pred_file, result_file, models in [
        (comp, 'comparison_metrics.csv', ['MLP','GWR','GNNWLR','GNOM']),
        (abl, 'ablation_metrics.csv', ['Local only','FNO + local','Full GNOM'])]:
        table = pd.read_csv(ROOT/'results'/result_file, encoding='utf-8-sig').set_index('Model')
        for model in models:
            got = metrics(pred_file.label.to_numpy(), pred_file[model].to_numpy())
            expected = table.loc[model, ['OA','Precision','Recall','F1','ROC-AUC','AP']].to_numpy(float)
            assert np.allclose(got, expected, atol=TOL, rtol=0), (model, got, expected)
    required = [
        'comparison_box_rose_consistent.png','comparison_box_rose_consistent.svg',
        'comparison_roc_pr_consistent.png','comparison_roc_pr_consistent.svg',
        'ablation_box_rose_consistent.png','ablation_box_rose_consistent.svg',
        'ablation_roc_pr_consistent.png','ablation_roc_pr_consistent.svg']
    assert all((ROOT/'figures'/'performance'/x).exists() for x in required)
    print(json.dumps({'status':'PASS','n':len(samples),'class_counts':{'0':766,'1':766},
                      'shared_GNOM_Full_GNOM':True,'threshold':0.5}, indent=2))

if __name__ == '__main__':
    main()
