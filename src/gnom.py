from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from rasterio.enums import Resampling
from rasterio.transform import rowcol
from rasterio.warp import transform
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "nujiang_samples.csv"
RASTER_DIR = ROOT / "data" / "rasters"
OUT_DIR = ROOT / "outputs"
PATCH_CACHE = ROOT / "data" / "processed" / "nujiang_factor_patches_32.npy"
PATCH_META = ROOT / "data" / "processed" / "nujiang_factor_patches_32_meta.json"
RESULT_CSV = OUT_DIR / "geofno_esf_results.csv"
PRED_CSV = OUT_DIR / "geofno_esf_oof_predictions.csv"
SUMMARY_TXT = OUT_DIR / "model_summary_and_parameters.txt"

SEED = 42
PATCH_SIZE = 32
BATCH_SIZE = 12
EPOCHS = 55
PATIENCE = 12
N_SPLITS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL = "label"
TABULAR = ["rivers", "Aspect", "Slope", "Precipitat", "Elevation", "faults", "Lithology", "road", "NDVI"]
RASTERS = {
    "elevation": "elevation.tif",
    "aspect": "aspect.tif",
    "slope": "slope.tif",
    "precipitation": "precipitation.tif",
    "rivers": "rivers.tif",
    "lithology": "lithology.tif",
    "road": "road.tif",
    "ndvi": "ndvi.tif",
}
CHANNEL = {name: i for i, name in enumerate(RASTERS)}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_df() -> pd.DataFrame:
    return pd.read_csv(DATA_CSV, encoding="utf-8-sig").dropna(subset=["X", "Y", LABEL] + TABULAR).reset_index(drop=True)


def read_patch(ds, x: float, y: float, size: int) -> np.ndarray:
    row, col = ds.index(x, y)
    half = size // 2
    window = rasterio.windows.Window(col - half, row - half, size, size)
    arr = ds.read(
        1,
        window=window,
        boundless=True,
        fill_value=np.nan,
        out_shape=(size, size),
        resampling=Resampling.bilinear,
    ).astype("float32")
    if ds.nodata is not None:
        arr = np.where(np.isclose(arr, ds.nodata), np.nan, arr)
    return np.where(np.abs(arr) > 1e20, np.nan, arr)


def build_patch_cache(df: pd.DataFrame) -> np.ndarray:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if PATCH_CACHE.exists() and PATCH_META.exists():
        meta = json.loads(PATCH_META.read_text(encoding="utf-8"))
        if meta.get("n_samples") == len(df) and meta.get("patch_size") == PATCH_SIZE and meta.get("rasters") == list(RASTERS):
            return np.load(PATCH_CACHE)

    patches = np.empty((len(df), len(RASTERS), PATCH_SIZE, PATCH_SIZE), dtype="float32")
    lon = df["X"].to_numpy().tolist()
    lat = df["Y"].to_numpy().tolist()
    for ch, (name, filename) in enumerate(RASTERS.items()):
        with rasterio.open(RASTER_DIR / filename) as ds:
            xs, ys = transform("EPSG:4326", ds.crs, lon, lat)
            for i, (x, y) in enumerate(zip(xs, ys)):
                patches[i, ch] = read_patch(ds, x, y, PATCH_SIZE)
        print(f"cached patches for {name}")

    for ch in range(patches.shape[1]):
        med = np.nanmedian(patches[:, ch])
        if not np.isfinite(med):
            med = 0.0
        patches[:, ch] = np.nan_to_num(patches[:, ch], nan=float(med), posinf=float(med), neginf=float(med))

    np.save(PATCH_CACHE, patches)
    PATCH_META.write_text(
        json.dumps({"n_samples": len(df), "patch_size": PATCH_SIZE, "rasters": list(RASTERS)}, indent=2),
        encoding="utf-8",
    )
    return patches


def channel_normalize(train_patches: np.ndarray, all_patches: np.ndarray) -> np.ndarray:
    mean = train_patches.mean(axis=(0, 2, 3), keepdims=True)
    std = train_patches.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    return ((all_patches - mean) / std).astype("float32")


def count_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def module_parameter_table(model: nn.Module) -> pd.DataFrame:
    rows = [{"module": "TOTAL", "parameters": count_parameters(model)}]
    for name, child in model.named_children():
        rows.append({"module": name, "parameters": count_parameters(child)})
    return pd.DataFrame(rows)


class SpectralConv2d(nn.Module):
    """Original FNO spectral convolution used as a baseline."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    @staticmethod
    def compl_mul2d(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            x.size(0),
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class GeoSpatialKernel(nn.Module):
    """Builds a differentiable geomorphology-constrained spatial affinity matrix.

    A_geo combines Euclidean distance, downslope/flow directional distance,
    terrain contrast, aspect alignment and lithology similarity. The dense
    matrix is used by the spatial-field energy losses, while a low-frequency
    projection of the same matrix gates Fourier modes in GeoSpectralConv2d.
    """

    def __init__(self, patch_size: int, modes: int, radius: int = 3, eps: float = 1e-6):
        super().__init__()
        self.patch_size = patch_size
        self.modes = modes
        self.radius = radius
        self.eps = eps

        offsets = []
        for oy in range(-radius, radius + 1):
            for ox in range(-radius, radius + 1):
                if oy == 0 and ox == 0:
                    continue
                offsets.append((oy, ox))
        offsets_t = torch.tensor(offsets, dtype=torch.float32)
        euclidean = torch.sqrt(offsets_t[:, 0].pow(2) + offsets_t[:, 1].pow(2) + eps) / max(radius, 1)
        theta = torch.atan2(offsets_t[:, 0], offsets_t[:, 1])
        self.register_buffer("offset_y", offsets_t[:, 0].long())
        self.register_buffer("offset_x", offsets_t[:, 1].long())
        self.register_buffer("euclidean", euclidean.view(1, -1, 1, 1))
        self.register_buffer("theta", theta.view(1, -1, 1, 1))

        self.raw_distance_weights = nn.Parameter(torch.tensor([1.0, 0.7, 0.8], dtype=torch.float32))
        self.raw_sigma = nn.Parameter(torch.tensor(0.6, dtype=torch.float32))
        self.raw_lithology_tau = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, geo_patch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, _, h, w = geo_patch.shape
        if h != self.patch_size or w != self.patch_size:
            raise ValueError(f"GeoSpatialKernel expects {self.patch_size}x{self.patch_size} patches.")

        elevation = geo_patch[:, CHANNEL["elevation"]]
        aspect_deg = geo_patch[:, CHANNEL["aspect"]]
        slope = geo_patch[:, CHANNEL["slope"]]
        lithology = geo_patch[:, CHANNEL["lithology"]]

        elevation_z = (elevation - elevation.mean(dim=(1, 2), keepdim=True)) / (elevation.std(dim=(1, 2), keepdim=True) + self.eps)
        slope_z = (slope - slope.mean(dim=(1, 2), keepdim=True)) / (slope.std(dim=(1, 2), keepdim=True) + self.eps)
        lith_z = (lithology - lithology.mean(dim=(1, 2), keepdim=True)) / (lithology.std(dim=(1, 2), keepdim=True) + self.eps)

        neigh_elev = shift_stack(elevation_z, self.offset_y, self.offset_x)
        neigh_slope = shift_stack(slope_z, self.offset_y, self.offset_x)
        neigh_lith = shift_stack(lith_z, self.offset_y, self.offset_x)
        valid = shift_stack(torch.ones_like(elevation_z), self.offset_y, self.offset_x)

        terrain_difference = 0.6 * (elevation_z.unsqueeze(1) - neigh_elev).abs()
        terrain_difference = terrain_difference + 0.4 * (slope_z.unsqueeze(1) - neigh_slope).abs()

        aspect_rad = torch.deg2rad(torch.remainder(aspect_deg, 360.0)).unsqueeze(1)
        direction_alignment = torch.relu(torch.cos(self.theta - aspect_rad))
        flow_distance = self.euclidean * (1.0 + (1.0 - direction_alignment))

        lithology_similarity = torch.exp(-(lith_z.unsqueeze(1) - neigh_lith).abs() / (F.softplus(self.raw_lithology_tau) + self.eps))

        weights = F.softmax(self.raw_distance_weights, dim=0)
        sigma = F.softplus(self.raw_sigma) + self.eps
        d_geo = weights[0] * self.euclidean + weights[1] * flow_distance + weights[2] * terrain_difference
        affinity = torch.exp(-d_geo / sigma) * direction_alignment * lithology_similarity
        affinity = affinity * valid
        affinity = affinity / (affinity.sum(dim=1, keepdim=True) + self.eps)

        support_field = affinity.amax(dim=1)
        gate = torch.fft.rfft2(support_field)
        gate = gate.abs()[:, : self.modes, : self.modes]
        gate = gate / (gate.amax(dim=(-2, -1), keepdim=True) + self.eps)
        gate = 0.25 + 0.75 * gate
        return affinity, gate, support_field


class GeoSpectralConv2d(nn.Module):
    """FNO spectral convolution modulated by a geography-derived Fourier gate."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    @staticmethod
    def compl_mul2d(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x: torch.Tensor, geo_gate: torch.Tensor) -> torch.Tensor:
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            x.size(0),
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        gate = geo_gate.unsqueeze(1)
        low = self.compl_mul2d(x_ft[:, :, : self.modes1, : self.modes2], self.weights1)
        high = self.compl_mul2d(x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2)
        out_ft[:, :, : self.modes1, : self.modes2] = low * gate
        out_ft[:, :, -self.modes1 :, : self.modes2] = high * gate
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNO2dClassifier(nn.Module):
    def __init__(self, in_channels: int, tab_dim: int, width: int = 24, modes: int = 8):
        super().__init__()
        self.lift = nn.Conv2d(in_channels + 2, width, 1)
        self.spec = nn.ModuleList([SpectralConv2d(width, width, modes, modes) for _ in range(4)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(4)])
        self.bn = nn.ModuleList([nn.BatchNorm2d(width) for _ in range(4)])
        self.tab = nn.Sequential(nn.Linear(tab_dim, 48), nn.SiLU(), nn.Dropout(0.10), nn.Linear(48, 32), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(width + 32, 64), nn.SiLU(), nn.Dropout(0.15), nn.Linear(64, 1))

    def forward(self, patch: torch.Tensor, tab: torch.Tensor, geo_patch: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        b, _, h, w = patch.shape
        grid = coordinate_grid(b, h, w, patch.device)
        x = self.lift(torch.cat([patch, grid], dim=1))
        for spec, local, bn in zip(self.spec, self.local, self.bn):
            x = F.gelu(bn(spec(x) + local(x)))
        feat = x.mean(dim=(-2, -1))
        logits = self.head(torch.cat([feat, self.tab(tab)], dim=1)).squeeze(1)
        return {"logit": logits}


class GeoFNOESF(nn.Module):
    """GNOM: geography-gated FNO with an explicit susceptibility-field branch."""

    def __init__(self, in_channels: int, tab_dim: int, width: int = 24, modes: int = 8):
        super().__init__()
        self.geo_kernel = GeoSpatialKernel(PATCH_SIZE, modes)
        self.lift = nn.Conv2d(in_channels + 2, width, 1)
        self.spec = nn.ModuleList([GeoSpectralConv2d(width, width, modes, modes) for _ in range(4)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(4)])
        self.bn = nn.ModuleList([nn.BatchNorm2d(width) for _ in range(4)])
        self.field_head = nn.Sequential(nn.Conv2d(width, width // 2, 1), nn.GELU(), nn.Conv2d(width // 2, 1, 1))
        self.tab = nn.Sequential(nn.Linear(tab_dim, 48), nn.SiLU(), nn.Dropout(0.10), nn.Linear(48, 32), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(width + 32 + 3, 64), nn.SiLU(), nn.Dropout(0.15), nn.Linear(64, 1))

    def forward(self, patch: torch.Tensor, tab: torch.Tensor, geo_patch: torch.Tensor) -> dict[str, torch.Tensor]:
        b, _, h, w = patch.shape
        affinity, geo_gate, center_kernel = self.geo_kernel(geo_patch)
        x = self.lift(torch.cat([patch, coordinate_grid(b, h, w, patch.device)], dim=1))
        for spec, local, bn in zip(self.spec, self.local, self.bn):
            x = F.gelu(bn(spec(x, geo_gate) + local(x)))

        field_logit = self.field_head(x).squeeze(1)
        field_prob = torch.sigmoid(field_logit)
        field_stats = torch.stack(
            [
                field_prob.mean(dim=(-2, -1)),
                field_prob.amax(dim=(-2, -1)),
                field_prob.std(dim=(-2, -1)),
            ],
            dim=1,
        )
        feat = x.mean(dim=(-2, -1))
        logits = self.head(torch.cat([feat, self.tab(tab), field_stats], dim=1)).squeeze(1)
        return {
            "logit": logits,
            "field_logit": field_logit,
            "field_prob": field_prob,
            "affinity": affinity,
            "geo_gate": geo_gate,
            "center_kernel": center_kernel,
        }


def coordinate_grid(batch: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.linspace(0, 1, h, device=device), torch.linspace(0, 1, w, device=device), indexing="ij")
    return torch.stack([xx, yy], dim=0).unsqueeze(0).repeat(batch, 1, 1, 1)


def shift_stack(field: torch.Tensor, offset_y: torch.Tensor, offset_x: torch.Tensor) -> torch.Tensor:
    """Returns neighbor values for a sparse local geographic matrix.

    Output shape is [B, K, H, W], where K is the number of offsets. Padding
    uses zero and a matching validity mask is built by shifting an all-one
    field with the same offsets.
    """

    b, h, w = field.shape
    padded = F.pad(field.unsqueeze(1), (1 + int(offset_x.abs().max()),) * 2 + (1 + int(offset_y.abs().max()),) * 2, mode="constant", value=0.0)
    pad_y = 1 + int(offset_y.abs().max())
    pad_x = 1 + int(offset_x.abs().max())
    shifted = []
    for oy, ox in zip(offset_y.tolist(), offset_x.tolist()):
        y0 = pad_y + int(oy)
        x0 = pad_x + int(ox)
        shifted.append(padded[:, 0, y0 : y0 + h, x0 : x0 + w])
    return torch.stack(shifted, dim=1)


def spatial_smoothness_loss(field_prob: torch.Tensor, affinity: torch.Tensor) -> torch.Tensor:
    radius = 3
    offsets = [(oy, ox) for oy in range(-radius, radius + 1) for ox in range(-radius, radius + 1) if not (oy == 0 and ox == 0)]
    offset_y = torch.tensor([v[0] for v in offsets], dtype=torch.long, device=field_prob.device)
    offset_x = torch.tensor([v[1] for v in offsets], dtype=torch.long, device=field_prob.device)
    neigh = shift_stack(field_prob, offset_y, offset_x)
    diff = field_prob.unsqueeze(1) - neigh
    return (affinity * diff.pow(2)).sum(dim=(1, 2, 3)).mean()


def field_consistency_loss(field_prob: torch.Tensor, class_logit: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    field_mean_logit = torch.logit(field_prob.mean(dim=(-2, -1)).clamp(1e-5, 1 - 1e-5))
    target_loss = F.binary_cross_entropy_with_logits(field_mean_logit, y)
    agreement_loss = F.mse_loss(torch.sigmoid(class_logit), torch.sigmoid(field_mean_logit))
    return target_loss + agreement_loss


def geofno_loss(outputs: dict[str, torch.Tensor], y: torch.Tensor, pos_weight: torch.Tensor, lambda_smooth: float, lambda_field: float) -> tuple[torch.Tensor, dict[str, float]]:
    bce = F.binary_cross_entropy_with_logits(outputs["logit"], y, pos_weight=pos_weight)
    if "field_prob" not in outputs:
        return bce, {"bce": float(bce.detach().cpu()), "smooth": 0.0, "field": 0.0}
    smooth = spatial_smoothness_loss(outputs["field_prob"], outputs["affinity"])
    field = field_consistency_loss(outputs["field_prob"], outputs["logit"], y)
    loss = bce + lambda_smooth * smooth + lambda_field * field
    return loss, {
        "bce": float(bce.detach().cpu()),
        "smooth": float(smooth.detach().cpu()),
        "field": float(field.detach().cpu()),
    }


def make_loaders(patches, geo_patches, tab, y, train_idx, valid_idx):
    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(patches[train_idx], dtype=torch.float32),
        torch.tensor(geo_patches[train_idx], dtype=torch.float32),
        torch.tensor(tab[train_idx], dtype=torch.float32),
        torch.tensor(y[train_idx], dtype=torch.float32),
    )
    valid_ds = torch.utils.data.TensorDataset(
        torch.tensor(patches[valid_idx], dtype=torch.float32),
        torch.tensor(geo_patches[valid_idx], dtype=torch.float32),
        torch.tensor(tab[valid_idx], dtype=torch.float32),
        torch.tensor(y[valid_idx], dtype=torch.float32),
    )
    return (
        torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        torch.utils.data.DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False),
    )


@torch.no_grad()
def predict_model(model: nn.Module, patches, geo_patches, tab, indices) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    probs, fields = [], []
    ds = torch.utils.data.TensorDataset(
        torch.tensor(patches[indices], dtype=torch.float32),
        torch.tensor(geo_patches[indices], dtype=torch.float32),
        torch.tensor(tab[indices], dtype=torch.float32),
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    for xb, gb, tb in loader:
        out = model(xb.to(DEVICE), tb.to(DEVICE), gb.to(DEVICE))
        probs.append(torch.sigmoid(out["logit"]).cpu().numpy())
        if "field_prob" in out:
            fields.append(out["field_prob"].cpu().numpy())
    return np.concatenate(probs), np.concatenate(fields) if fields else None


def train_fold(
    model_kind: str,
    patches,
    geo_patches,
    tab,
    y,
    train_idx,
    valid_idx,
    seed: int,
    lambda_smooth: float,
    lambda_field: float,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, float], nn.Module]:
    set_seed(seed)
    train_loader, _ = make_loaders(patches, geo_patches, tab, y, train_idx, valid_idx)
    if model_kind == "baseline_fno":
        model: nn.Module = FNO2dClassifier(patches.shape[1], tab.shape[1]).to(DEVICE)
    else:
        model = GeoFNOESF(patches.shape[1], tab.shape[1]).to(DEVICE)

    pos = float(y[train_idx].sum())
    neg = float(len(train_idx) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(EPOCHS, 1), eta_min=2e-5)
    best_auc, best_state, bad = -1.0, None, 0

    for epoch in range(EPOCHS):
        model.train()
        for xb, gb, tb, yb in train_loader:
            xb, gb, tb, yb = xb.to(DEVICE), gb.to(DEVICE), tb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb, tb, gb)
            loss, _ = geofno_loss(out, yb, pos_weight, lambda_smooth, lambda_field)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        scheduler.step()
        pv, _ = predict_model(model, patches, geo_patches, tab, valid_idx)
        auc = roc_auc_score(y[valid_idx], pv)
        if auc > best_auc + 1e-4:
            best_auc = float(auc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    prob, fields = predict_model(model, patches, geo_patches, tab, valid_idx)
    return prob, fields, {"best_valid_auc": best_auc, "epochs_used": epoch + 1}, model


def classification_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, np.clip(prob, 1e-6, 1 - 1e-6))),
    }


def morans_i_residual(x: np.ndarray, y: np.ndarray, residual: np.ndarray, k: int = 8) -> float:
    coords = np.column_stack([x, y]).astype("float64")
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    nn = np.argpartition(dist, kth=min(k, len(coords) - 2), axis=1)[:, :k]
    weights = np.zeros_like(dist, dtype="float64")
    rows = np.arange(len(coords))[:, None]
    weights[rows, nn] = 1.0 / (dist[rows, nn] + 1e-9)
    z = residual - residual.mean()
    denom = np.sum(z * z)
    if denom <= 0:
        return float("nan")
    return float((len(z) / weights.sum()) * (weights * (z[:, None] * z[None, :])).sum() / denom)


def spatial_block_metrics(df: pd.DataFrame, y: np.ndarray, prob: np.ndarray, n_blocks: int = 4) -> dict[str, float]:
    xb = pd.qcut(df["X"], q=n_blocks, labels=False, duplicates="drop")
    yb = pd.qcut(df["Y"], q=n_blocks, labels=False, duplicates="drop")
    block = (xb.astype(int) * n_blocks + yb.astype(int)).to_numpy()
    aucs = []
    for bid in np.unique(block):
        idx = np.where(block == bid)[0]
        if len(idx) >= 20 and len(np.unique(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], prob[idx]))
    return {"spatial_block_auc_mean": float(np.mean(aucs)), "spatial_block_auc_std": float(np.std(aucs)), "n_spatial_blocks": int(len(aucs))}


def save_probability_raster(df: pd.DataFrame, prob: np.ndarray, output_path: Path) -> None:
    reference = RASTER_DIR / "elevation.tif"
    with rasterio.open(reference) as src:
        profile = src.profile.copy()
        arr = np.full((src.height, src.width), -9999.0, dtype="float32")
        xs, ys = transform("EPSG:4326", src.crs, df["X"].to_numpy().tolist(), df["Y"].to_numpy().tolist())
        rr, cc = rowcol(src.transform, xs, ys)
        bucket: dict[tuple[int, int], list[float]] = {}
        for r, c, p in zip(rr, cc, prob):
            if 0 <= r < src.height and 0 <= c < src.width:
                bucket.setdefault((int(r), int(c)), []).append(float(p))
        for (r, c), vals in bucket.items():
            arr[r, c] = float(np.mean(vals))
        profile.update(count=1, dtype="float32", nodata=-9999.0, compress="lzw")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(arr, 1)


def save_geo_kernel_visual(model: nn.Module, geo_patch: np.ndarray, output_prefix: Path) -> None:
    if not isinstance(model, GeoFNOESF):
        return
    model.eval()
    with torch.no_grad():
        gb = torch.tensor(geo_patch[None], dtype=torch.float32, device=DEVICE)
        affinity, gate, center_kernel = model.geo_kernel(gb)
    np.save(str(output_prefix) + "_affinity.npy", affinity[0].cpu().numpy())
    np.save(str(output_prefix) + "_fourier_gate.npy", gate[0].cpu().numpy())
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), dpi=160)
    im0 = axes[0].imshow(center_kernel[0].cpu().numpy(), cmap="viridis")
    axes[0].set_title("A_geo from center pixel")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(gate[0].cpu().numpy(), cmap="magma")
    axes[1].set_title("Fourier geo gate")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(str(output_prefix) + ".png")
    plt.close(fig)


def run_cv_for_model(model_kind: str, raw_patches: np.ndarray, df: pd.DataFrame, lambda_smooth: float, lambda_field: float):
    y = df[LABEL].to_numpy(dtype=int)
    tab_raw = df[TABULAR].to_numpy(dtype="float32")
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(df), dtype="float64")
    field_mean = np.full(len(df), np.nan, dtype="float64")
    fold_info = []
    best_model, best_auc, best_geo_for_visual = None, -1.0, None

    for fold, (tr, va) in enumerate(cv.split(raw_patches, y), start=1):
        patches = channel_normalize(raw_patches[tr], raw_patches)
        geo_patches = raw_patches.astype("float32")
        scaler = StandardScaler().fit(tab_raw[tr])
        tab = scaler.transform(tab_raw).astype("float32")
        prob, fields, info, model = train_fold(model_kind, patches, geo_patches, tab, y, tr, va, SEED + fold, lambda_smooth, lambda_field)
        oof[va] = prob
        if fields is not None:
            field_mean[va] = fields.mean(axis=(1, 2))
        fold_auc = roc_auc_score(y[va], prob)
        fold_info.append({"fold": fold, "auc": float(fold_auc), **info})
        if fold_auc > best_auc:
            best_auc = float(fold_auc)
            best_model = copy.deepcopy(model).cpu()
            best_geo_for_visual = geo_patches[va[0]]
        print(f"{model_kind} fold {fold}: AUC={fold_auc:.4f}, epochs={info['epochs_used']}")

    metrics = classification_metrics(y, oof)
    metrics.update(spatial_block_metrics(df, y, oof))
    metrics["moran_i_residual"] = morans_i_residual(df["X"].to_numpy(), df["Y"].to_numpy(), y - oof)
    metrics["method"] = model_kind
    metrics["lambda_smooth"] = lambda_smooth
    metrics["lambda_field"] = lambda_field
    metrics["folds"] = json.dumps(fold_info, ensure_ascii=False)
    return metrics, oof, field_mean, best_model, best_geo_for_visual


def write_summary(models: dict[str, nn.Module]) -> None:
    parts = [f"Device: {DEVICE}", f"Patch size: {PATCH_SIZE}", ""]
    for name, model in models.items():
        parts.append(f"## {name}")
        parts.append(str(model))
        parts.append("")
        parts.append(module_parameter_table(model).to_string(index=False))
        parts.append("")
    SUMMARY_TXT.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="GeoFNO-ESF landslide susceptibility experiment for Nujiang.")
    parser.add_argument("--quick", action="store_true", help="Run two folds and fewer epochs for a smoke test.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--patience", type=int, default=None, help="Override early-stopping patience.")
    parser.add_argument("--folds", type=int, default=None, help="Override number of cross-validation folds.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override mini-batch size.")
    args = parser.parse_args()
    global EPOCHS, PATIENCE, N_SPLITS, BATCH_SIZE
    if args.quick:
        EPOCHS = 3
        PATIENCE = 2
        N_SPLITS = 2
    if args.epochs is not None:
        EPOCHS = args.epochs
    if args.patience is not None:
        PATIENCE = args.patience
    if args.folds is not None:
        N_SPLITS = args.folds
    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size

    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_df()
    raw_patches = build_patch_cache(df)

    write_summary(
        {
            "Original FNO baseline": FNO2dClassifier(raw_patches.shape[1], len(TABULAR)),
            "GeoFNO-ESF": GeoFNOESF(raw_patches.shape[1], len(TABULAR)),
        }
    )

    configs = [
        ("baseline_fno", 0.0, 0.0),
        ("geofno", 0.0, 0.0),
        ("geofno_esf", 0.08, 0.25),
    ]
    rows = []
    prediction_table = pd.DataFrame({"X": df["X"], "Y": df["Y"], "label": df[LABEL].astype(int)})
    best_esf_model, best_esf_geo = None, None

    for model_kind, lambda_smooth, lambda_field in configs:
        metrics, oof, field_mean, best_model, best_geo = run_cv_for_model(model_kind, raw_patches, df, lambda_smooth, lambda_field)
        rows.append(metrics)
        prediction_table[f"{model_kind}_prob"] = oof
        if not np.all(np.isnan(field_mean)):
            prediction_table[f"{model_kind}_field_mean"] = field_mean
        if model_kind == "geofno_esf":
            best_esf_model, best_esf_geo = best_model, best_geo
            torch.save(best_model.state_dict(), OUT_DIR / "best_geofno_esf.pt")
            save_probability_raster(df, oof, OUT_DIR / "geofno_esf_point_probability_raster.tif")

    result = pd.DataFrame(rows)
    result.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")
    prediction_table.to_csv(PRED_CSV, index=False, encoding="utf-8-sig")
    if best_esf_model is not None and best_esf_geo is not None:
        best_esf_model = best_esf_model.to(DEVICE)
        save_geo_kernel_visual(best_esf_model, best_esf_geo, OUT_DIR / "geofno_esf_geo_kernel_sample")

    print(result[["method", "auc", "accuracy", "f1", "precision", "recall", "moran_i_residual", "spatial_block_auc_mean"]].to_string(index=False))
    print(f"Saved results to {RESULT_CSV}")
    print(f"Saved predictions to {PRED_CSV}")
    print(f"Saved model summary to {SUMMARY_TXT}")


if __name__ == "__main__":
    raise SystemExit(
        "This module defines the GNOM architecture. For manuscript-consistent "
        "evaluation run: python scripts/run_ablation_nested_cv.py"
    )
