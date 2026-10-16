"""Leakage-resistant five-fold ablation rerun for the Nujiang GNOM study.

Outer folds are used exactly once for evaluation.  Early-stopping epochs are
selected on an inner split of each outer-training fold; the model is then
retrained on the complete outer-training fold for that fixed epoch count.
No seed or model is selected using outer-fold performance.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))
import gnom as g  # noqa: E402

OUT = PACKAGE / "results" / "rerun"
SEED = 42
FOLDS = 5
BATCH = 12
MAX_EPOCHS = 55
PATIENCE = 10


class LocalMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 64), nn.SiLU(), nn.Dropout(.20),
                                 nn.Linear(64, 32), nn.SiLU(), nn.Dropout(.10),
                                 nn.Linear(32, 1))

    def forward(self, patch, tab, geo=None):
        return {"logit": self.net(tab).squeeze(1)}


def make_model(kind: str, n_channels: int, tab_dim: int) -> nn.Module:
    if kind == "Local only":
        return LocalMLP(tab_dim)
    if kind == "FNO + local":
        return g.FNO2dClassifier(n_channels, tab_dim)
    if kind == "Full GNOM":
        return g.GeoFNOESF(n_channels, tab_dim)
    raise ValueError(kind)


def arrays(raw, tab_raw, train_idx):
    patch = g.channel_normalize(raw[train_idx], raw)
    tab = StandardScaler().fit(tab_raw[train_idx]).transform(tab_raw).astype("float32")
    return patch, raw.astype("float32"), tab


def loader(patch, geo, tab, y, idx, shuffle):
    ds = torch.utils.data.TensorDataset(
        torch.tensor(patch[idx]), torch.tensor(geo[idx]), torch.tensor(tab[idx]),
        torch.tensor(y[idx], dtype=torch.float32))
    gen = torch.Generator().manual_seed(SEED)
    return torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=shuffle,
                                       generator=gen if shuffle else None)


@torch.no_grad()
def predict(model, patch, geo, tab, idx):
    model.eval(); out = []
    ds = torch.utils.data.TensorDataset(torch.tensor(patch[idx]),
                                        torch.tensor(geo[idx]), torch.tensor(tab[idx]))
    for xb, gb, tb in torch.utils.data.DataLoader(ds, batch_size=BATCH):
        out.append(torch.sigmoid(model(xb, tb, gb)["logit"]).cpu().numpy())
    return np.concatenate(out)


def fit_epochs(kind, patch, geo, tab, y, train_idx, valid_idx, seed):
    g.set_seed(seed)
    model = make_model(kind, patch.shape[1], tab.shape[1])
    pos = y[train_idx].sum(); neg = len(train_idx) - pos
    pw = torch.tensor([neg / max(pos, 1)], dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=2e-5)
    best_auc, best_epoch, bad = -np.inf, 1, 0
    tr_loader = loader(patch, geo, tab, y, train_idx, True)
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, gb, tb, yb in tr_loader:
            opt.zero_grad(); outputs = model(xb, tb, gb)
            loss, _ = g.geofno_loss(outputs, yb, pw, .08, .25)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        pv = predict(model, patch, geo, tab, valid_idx)
        score = roc_auc_score(y[valid_idx], pv)
        if score > best_auc + 1e-4:
            best_auc, best_epoch, bad = score, epoch, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    return best_epoch, float(best_auc)


def refit_predict(kind, patch, geo, tab, y, train_idx, test_idx, epochs, seed):
    g.set_seed(seed)
    model = make_model(kind, patch.shape[1], tab.shape[1])
    pos = y[train_idx].sum(); neg = len(train_idx) - pos
    pw = torch.tensor([neg / max(pos, 1)], dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=2e-5)
    tr_loader = loader(patch, geo, tab, y, train_idx, True)
    for _ in range(epochs):
        model.train()
        for xb, gb, tb, yb in tr_loader:
            opt.zero_grad(); outputs = model(xb, tb, gb)
            loss, _ = g.geofno_loss(outputs, yb, pw, .08, .25)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
    return predict(model, patch, geo, tab, test_idx)


def metrics(y, p):
    z = (p >= .5).astype(int)
    return {"OA": accuracy_score(y, z), "Precision": precision_score(y, z),
            "Recall": recall_score(y, z), "F1": f1_score(y, z),
            "ROC-AUC": roc_auc_score(y, p), "AP": average_precision_score(y, p)}


def main():
    global MAX_EPOCHS, PATIENCE
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    args = ap.parse_args()
    MAX_EPOCHS, PATIENCE = args.max_epochs, args.patience
    OUT.mkdir(exist_ok=True)
    g.set_seed(SEED); g.BATCH_SIZE = BATCH
    df = g.load_df(); raw = g.build_patch_cache(df)
    y = df[g.LABEL].to_numpy(int); tab_raw = df[g.TABULAR].to_numpy("float32")
    outer = list(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(raw, y))
    models = ["Local only", "FNO + local", "Full GNOM"]
    pred = {m: np.zeros(len(y)) for m in models}; fold_id = np.zeros(len(y), int)
    log = []
    for fold, (tr, te) in enumerate(outer, 1):
        fold_id[te] = fold
        inner_tr, inner_va = train_test_split(tr, test_size=.20, stratify=y[tr], random_state=100 + fold)
        patch, geo, tab = arrays(raw, tab_raw, tr)
        for j, kind in enumerate(models):
            seed = 1000 + fold * 10 + j
            ep, va_auc = fit_epochs(kind, patch, geo, tab, y, inner_tr, inner_va, seed)
            pred[kind][te] = refit_predict(kind, patch, geo, tab, y, tr, te, ep, seed)
            fold_auc = roc_auc_score(y[te], pred[kind][te])
            log.append({"fold": fold, "model": kind, "selected_epochs": ep,
                        "inner_valid_auc": va_auc, "outer_test_auc": fold_auc})
            print(f"fold={fold} model={kind} epochs={ep} inner_auc={va_auc:.4f} outer_auc={fold_auc:.4f}", flush=True)
    out = pd.DataFrame({"X": df.X, "Y": df.Y, "label": y, "fold": fold_id, **pred})
    out.to_csv(OUT / "ablation_nested_oof_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"Model": m, **metrics(y, pred[m])} for m in models]).to_csv(
        OUT / "ablation_nested_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(log).to_csv(OUT / "ablation_nested_training_log.csv", index=False, encoding="utf-8-sig")
    protocol = {"outer_cv": "5-fold StratifiedKFold, seed=42", "inner_validation": "20% stratified split of each outer-training fold",
                "selection": "early-stopping epoch by inner ROC-AUC only", "outer_test_use": "evaluation only",
                "seeds": "fixed a priori; no seed screening", "threshold": 0.5,
                "gnom_penalties": {"lambda_smooth": .08, "lambda_field": .25}}
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(pd.read_csv(OUT / "ablation_nested_metrics.csv").to_string(index=False))


if __name__ == "__main__":
    main()
