"""Recompute the six reported metrics from saved OOF probabilities."""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

ROOT=Path(__file__).resolve().parents[1]
METRICS=["OA","Precision","Recall","F1","ROC-AUC","AP"]

def score(y,p):
    z=(p>=.5).astype(int)
    return {"OA":accuracy_score(y,z),"Precision":precision_score(y,z,zero_division=0),
            "Recall":recall_score(y,z,zero_division=0),"F1":f1_score(y,z,zero_division=0),
            "ROC-AUC":roc_auc_score(y,p),"AP":average_precision_score(y,p)}

def build(path,label_col,model_cols):
    df=pd.read_csv(path,encoding="utf-8-sig"); y=df[label_col].to_numpy(int)
    return pd.DataFrame([{"Model":name,**score(y,df[col].to_numpy(float))} for name,col in model_cols.items()])

def main():
    comp=build(ROOT/"predictions"/"comparison_oof_predictions.csv","label",
               {x:x for x in ["MLP","GWR","GNNWLR","GNOM"]})
    abl=build(ROOT/"predictions"/"ablation_oof_predictions.csv","label",
              {x:x for x in ["Local only","FNO + local","Full GNOM"]})
    comp.to_csv(ROOT/"results"/"comparison_metrics.csv",index=False,encoding="utf-8-sig")
    abl.to_csv(ROOT/"results"/"ablation_metrics.csv",index=False,encoding="utf-8-sig")
    def md(t):
        head="| Model | OA | Precision | Recall | F1 | ROC-AUC | PR-AUC |\n|---|---:|---:|---:|---:|---:|---:|\n"
        return head+"\n".join(
            f"| {r['Model']} | {r['OA']:.4f} | {r['Precision']:.4f} | {r['Recall']:.4f} | {r['F1']:.4f} | {r['ROC-AUC']:.4f} | {r['AP']:.4f} |"
            for _,r in t.iterrows())
    text="# Final results\n\n## Model comparison\n\n"+md(comp)+"\n\n## Ablation experiment\n\n"+md(abl)+"\n"
    (ROOT/"results"/"RESULTS_TABLES.md").write_text(text,encoding="utf-8")
    print("Comparison\n",comp.to_string(index=False)); print("\nAblation\n",abl.to_string(index=False))

if __name__=="__main__": main()
