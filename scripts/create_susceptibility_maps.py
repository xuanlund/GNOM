"""Generate native 30 m OOF-constrained susceptibility maps.

The experiments provide point-wise out-of-fold probabilities rather than
deployable wall-to-wall checkpoints for every model.  All seven probability
fields are therefore spatialized directly on the original 30 m reference grid
with one shared projected k-nearest-neighbour IDW operator.  No coarse raster is
resampled and no model-specific cartographic adjustment is applied.
"""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from pyproj import Transformer
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import xy
from rasterio.warp import transform
from rasterio.windows import Window
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
LOCAL_ELEVATION = ROOT / "data" / "rasters" / "elevation.tif"
LEGACY_ELEVATION = ROOT.parent / "old" / "fno_data_ascii" / "elevation.tif"
SOURCE_ELEVATION = (
    LOCAL_ELEVATION if LOCAL_ELEVATION.exists() else LEGACY_ELEVATION
)
MAP_ROOT = ROOT / "maps"
RASTER_OUT = MAP_ROOT / "rasters"
FIGURE_OUT = MAP_ROOT / "figures"
RESULT_OUT = MAP_ROOT / "results"
MASK_PATH = RASTER_OUT / "study_area_mask_30m_clean.tif"
for folder in (RASTER_OUT, FIGURE_OUT, RESULT_OUT):
    folder.mkdir(parents=True, exist_ok=True)

TARGET_RESOLUTION_M = 30.0
IDW_NEIGHBORS = 64
IDW_POWER = 2.0
IDW_SMOOTHING_M = 1200.0
TILE_SIZE = 512
DISPLAY_MAX_ROWS = 3500
NODATA = -9999.0
THRESHOLDS = np.array([0, .2, .4, .6, .8, 1.000001])
CLASS_LABELS = ["Very low", "Low", "Moderate", "High", "Very high"]
CLASS_COLORS = ["#2A9D8F", "#8AB17D", "#E9C46A", "#F4A261", "#D55E00"]
MODEL_COLORS = {
    "MLP": "#B9BEC7", "GWR": "#79A9C5", "GNNWLR": "#3575A6",
    "GNOM": "#D55E00", "Local only": "#B9BEC7",
    "FNO + local": "#56B4E9", "Full GNOM": "#D55E00",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": .75,
    "axes.spines.right": False, "axes.spines.top": False,
    "legend.frameon": False, "savefig.dpi": 600,
})


def model_columns():
    comparison = pd.read_csv(
        ROOT / "predictions/comparison_oof_predictions.csv", encoding="utf-8-sig"
    )
    ablation = pd.read_csv(
        ROOT / "predictions/ablation_oof_predictions.csv", encoding="utf-8-sig"
    )
    if len(comparison) != len(ablation) or not np.allclose(
        comparison[["X", "Y"]], ablation[["X", "Y"]]
    ):
        raise RuntimeError("Comparison and ablation prediction rows do not align")
    columns = OrderedDict([
        ("MLP", comparison.MLP), ("GWR", comparison.GWR),
        ("GNNWLR", comparison.GNNWLR), ("GNOM", comparison.GNOM),
        ("Local only", ablation["Local only"]),
        ("FNO + local", ablation["FNO + local"]),
        ("Full GNOM", ablation["Full GNOM"]),
    ])
    return comparison, columns


def output_key(model):
    return model.lower().replace(" + ", "_plus_").replace(" ", "_")


def projected_training_coordinates(df, crs):
    xs, ys = transform(
        "EPSG:4326", crs, df.X.to_list(), df.Y.to_list()
    )
    return np.column_stack([xs, ys]).astype("float64")


def write_30m_surfaces(comparison, columns):
    """Interpolate all models in one tiled pass on the native source grid."""
    if not SOURCE_ELEVATION.exists():
        raise FileNotFoundError(SOURCE_ELEVATION)
    with rasterio.open(SOURCE_ELEVATION) as source:
        if not np.allclose(source.res, (TARGET_RESOLUTION_M, TARGET_RESOLUTION_M)):
            raise RuntimeError(f"Reference raster is not 30 m: {source.res}")
        profile = source.profile.copy()
        crs, affine = source.crs, source.transform
        width, height = source.width, source.height

        train_xy = projected_training_coordinates(comparison, crs)
        tree = cKDTree(train_xy)
        probability_matrix = np.column_stack(
            [series.to_numpy(float) for series in columns.values()]
        )

        mask_profile = profile.copy()
        mask_profile.update(
            driver="GTiff", count=1, dtype="uint8", nodata=0,
            compress="deflate", tiled=True, blockxsize=512, blockysize=512,
        )
        surface_profile = profile.copy()
        surface_profile.update(
            driver="GTiff", count=1, dtype="float32", nodata=NODATA,
            compress="deflate", predictor=3, tiled=True,
            blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER",
        )
        paths = OrderedDict(
            (model, RASTER_OUT / f"{output_key(model)}_30m_clean.tif")
            for model in columns
        )
        with rasterio.open(MASK_PATH, "w", **mask_profile) as mask_dst:
            destinations = [
                rasterio.open(path, "w", **surface_profile)
                for path in paths.values()
            ]
            try:
                total_tiles = int(np.ceil(height / TILE_SIZE) *
                                  np.ceil(width / TILE_SIZE))
                tile_index = 0
                for row_off in range(0, height, TILE_SIZE):
                    for col_off in range(0, width, TILE_SIZE):
                        tile_index += 1
                        h = min(TILE_SIZE, height - row_off)
                        w = min(TILE_SIZE, width - col_off)
                        window = Window(col_off, row_off, w, h)
                        elevation = source.read(1, window=window, masked=True)
                        valid = (
                            ~np.ma.getmaskarray(elevation) &
                            np.isfinite(elevation.filled(np.nan)) &
                            (elevation.filled(NODATA) != source.nodata)
                        )
                        mask_dst.write(valid.astype("uint8"), 1, window=window)
                        model_tiles = [
                            np.full((h, w), NODATA, dtype="float32")
                            for _ in columns
                        ]
                        rows, cols = np.nonzero(valid)
                        if len(rows):
                            global_rows = rows + row_off
                            global_cols = cols + col_off
                            xs, ys = xy(
                                affine, global_rows, global_cols, offset="center"
                            )
                            query = np.column_stack([xs, ys])
                            distances, indices = tree.query(
                                query, k=min(IDW_NEIGHBORS, len(train_xy)),
                                workers=-1,
                            )
                            distances = np.atleast_2d(distances)
                            indices = np.atleast_2d(indices)
                            if distances.shape[0] == 1 and len(query) > 1:
                                distances = distances.T
                                indices = indices.T
                            weights = 1.0 / (
                                distances ** IDW_POWER +
                                IDW_SMOOTHING_M ** IDW_POWER
                            )
                            values = np.einsum(
                                "nk,nkm->nm", weights,
                                probability_matrix[indices]
                            ) / weights.sum(axis=1, keepdims=True)
                            values = np.clip(values, 0, 1).astype("float32")
                            for model_id, tile in enumerate(model_tiles):
                                tile[rows, cols] = values[:, model_id]
                        for destination, tile in zip(destinations, model_tiles):
                            destination.write(tile, 1, window=window)
                        if tile_index % 25 == 0 or tile_index == total_tiles:
                            print(
                                f"30 m interpolation: {tile_index}/{total_tiles} "
                                f"tiles ({100 * tile_index / total_tiles:.1f}%)",
                                flush=True,
                            )
            finally:
                for destination in destinations:
                    destination.close()
    return crs, affine, paths


def read_display_surface(path):
    """Read a bounded preview only; the delivered GeoTIFF remains native 30 m."""
    with rasterio.open(path) as source:
        scale = max(1, int(np.ceil(source.height / DISPLAY_MAX_ROWS)))
        out_h = int(np.ceil(source.height / scale))
        out_w = int(np.ceil(source.width / scale))
        data = source.read(
            1, out_shape=(out_h, out_w), masked=True,
            resampling=Resampling.nearest,
        )
        display_affine = source.transform * source.transform.scale(
            source.width / out_w, source.height / out_h
        )
        return data, display_affine, source.crs


def degree_formatters(crs, center_x, center_y):
    inverse = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    def lon(value, _):
        return f"{inverse.transform(value, center_y)[0]:.1f}°E"
    def lat(value, _):
        return f"{inverse.transform(center_x, value)[1]:.1f}°N"
    return mpl.ticker.FuncFormatter(lon), mpl.ticker.FuncFormatter(lat)


def decorate_map(ax, crs, affine, shape, title, letter, outer_labels=True):
    height, width = shape
    left, top = affine * (0, 0)
    right, bottom = affine * (width, height)
    ax.set(xlim=(left, right), ylim=(bottom, top))
    ax.set_aspect("equal")
    ax.text(-.07, 1.025, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom")
    ax.text(.01, 1.025, title, transform=ax.transAxes, fontsize=7,
            fontweight="bold", va="bottom")
    if outer_labels:
        lon_fmt, lat_fmt = degree_formatters(
            crs, (left + right) / 2, (bottom + top) / 2
        )
        ax.xaxis.set_major_formatter(lon_fmt)
        ax.yaxis.set_major_formatter(lat_fmt)
        ax.tick_params(direction="out", length=2.5, width=.65)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.annotate(
        "N", xy=(x1 - .08*(x1-x0), y1 - .08*(y1-y0)),
        xytext=(x1 - .08*(x1-x0), y1 - .17*(y1-y0)),
        ha="center", va="center", fontsize=7, fontweight="bold",
        arrowprops={"arrowstyle": "-|>", "facecolor": "black",
                    "edgecolor": "black", "lw": .8},
    )
    length = 50000.0
    sx, sy = x1 - .08*(x1-x0) - length, y0 + .045*(y1-y0)
    ax.plot([sx, sx + length], [sy, sy], color="black", lw=2.1,
            solid_capstyle="butt")
    ax.text(sx, sy + .012*(y1-y0), "0", ha="center", fontsize=5.5)
    ax.text(sx + length, sy + .012*(y1-y0), "50 km",
            ha="center", fontsize=5.5)


def draw_surface(ax, path):
    surface, affine, crs = read_display_surface(path)
    height, width = surface.shape
    left, top = affine * (0, 0)
    right, bottom = affine * (width, height)
    cmap = ListedColormap(CLASS_COLORS)
    ax.imshow(
        surface, extent=(left, right, bottom, top), origin="upper",
        cmap=cmap, norm=BoundaryNorm(THRESHOLDS, cmap.N),
        interpolation="nearest", rasterized=True,
    )
    return surface, affine, crs


def class_legend(fig, y=.02):
    handles = [Patch(facecolor=c, edgecolor="none", label=l)
               for c, l in zip(CLASS_COLORS, CLASS_LABELS)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, y),
               ncol=5, columnspacing=1.0, handletextpad=.35)


def comparison_figure(paths):
    fig, axes = plt.subplots(2, 2, figsize=(7.205, 6.2))
    for index, (ax, model) in enumerate(
        zip(axes.flat, ["MLP", "GWR", "GNNWLR", "GNOM"])
    ):
        surface, affine, crs = draw_surface(ax, paths[model])
        decorate_map(ax, crs, affine, surface.shape, model,
                     chr(97 + index), index in (0, 2, 3))
        if index < 2:
            ax.tick_params(labelbottom=False)
    class_legend(fig, .015)
    fig.subplots_adjust(left=.07, right=.985, top=.97, bottom=.085,
                        wspace=.09, hspace=.17)
    fig.savefig(FIGURE_OUT / "comparison_susceptibility_maps.png",
                dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def ablation_figure(paths):
    fig, axes = plt.subplots(1, 3, figsize=(7.205, 4.15))
    for index, (ax, model) in enumerate(
        zip(axes, ["Local only", "FNO + local", "Full GNOM"])
    ):
        surface, affine, crs = draw_surface(ax, paths[model])
        decorate_map(ax, crs, affine, surface.shape, model,
                     chr(97 + index), index in (0, 2))
    class_legend(fig, .02)
    fig.subplots_adjust(left=.055, right=.985, top=.94, bottom=.12, wspace=.08)
    fig.savefig(FIGURE_OUT / "ablation_susceptibility_maps.png",
                dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def validation_summary(comparison):
    rows = []
    for model in ["MLP", "GWR", "GNNWLR", "GNOM"]:
        positive = comparison.loc[comparison.label.eq(1), model].to_numpy(float)
        negative = comparison.loc[comparison.label.eq(0), model].to_numpy(float)
        rows.append({
            "Model": model,
            "positive_capture_at_0_8": np.mean(positive >= .8),
            "negative_flag_rate_at_0_8": np.mean(negative >= .8),
            "very_high_observed_prevalence": comparison.loc[
                comparison[model].ge(.8), "label"
            ].mean(),
            "very_high_n": int(comparison[model].ge(.8).sum()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULT_OUT / "high_confidence_spatial_validation.csv",
                   index=False, encoding="utf-8-sig")
    return summary


def evidence_figure(comparison, paths):
    fig = plt.figure(figsize=(7.205, 3.7))
    ax_map, ax_tradeoff, ax_calibration = [
        fig.add_subplot(1, 3, i) for i in (1, 2, 3)
    ]
    surface, affine, crs = draw_surface(ax_map, paths["GNOM"])
    decorate_map(ax_map, crs, affine, surface.shape,
                 "GNOM and landslide inventory", "a")
    px, py = transform(
        "EPSG:4326", crs,
        comparison.loc[comparison.label.eq(1), "X"].to_list(),
        comparison.loc[comparison.label.eq(1), "Y"].to_list(),
    )
    ax_map.scatter(px, py, s=2.5, facecolor="#00A6D6", edgecolor="black",
                   linewidth=.18, alpha=.72, zorder=4)

    summary = validation_summary(comparison)
    for row in summary.itertuples():
        ax_tradeoff.scatter(
            row.negative_flag_rate_at_0_8, row.positive_capture_at_0_8,
            s=54 if row.Model == "GNOM" else 32,
            color=MODEL_COLORS[row.Model], edgecolor="white", linewidth=.5,
        )
        dx = -.004 if row.Model == "GNOM" else .004
        ax_tradeoff.text(
            row.negative_flag_rate_at_0_8 + dx,
            row.positive_capture_at_0_8 + .006, row.Model,
            ha="right" if dx < 0 else "left", fontsize=6,
            fontweight="bold" if row.Model == "GNOM" else "normal",
        )
    ax_tradeoff.set(
        xlabel="High-confidence false-positive rate",
        ylabel="High-confidence landslide capture",
        xlim=(.02, .115), ylim=(.18, .56),
    )
    ax_tradeoff.text(-.14, 1.03, "b", transform=ax_tradeoff.transAxes,
                     fontsize=8, fontweight="bold")
    ax_tradeoff.text(-.04, 1.03, "Confidence–specificity trade-off",
                     transform=ax_tradeoff.transAxes, fontsize=7,
                     fontweight="bold")

    centers = np.arange(5)
    for model in ["MLP", "GWR", "GNNWLR", "GNOM"]:
        bins = pd.cut(comparison[model], THRESHOLDS,
                      labels=False, include_lowest=True)
        prevalence = [
            comparison.loc[bins.eq(i), "label"].mean() for i in centers
        ]
        ax_calibration.plot(
            centers, prevalence, color=MODEL_COLORS[model],
            lw=1.8 if model == "GNOM" else 1.0, marker="o",
            ms=3.4 if model == "GNOM" else 2.6, label=model,
        )
    ax_calibration.set(
        xticks=centers, xticklabels=["VL", "L", "M", "H", "VH"],
        ylim=(0, 1), xlabel="Susceptibility class",
        ylabel="Observed landslide prevalence",
    )
    ax_calibration.legend(loc="upper left", ncol=2, handlelength=1.2,
                          columnspacing=.7, handletextpad=.3)
    ax_calibration.text(-.14, 1.03, "c", transform=ax_calibration.transAxes,
                        fontsize=8, fontweight="bold")
    ax_calibration.text(-.04, 1.03, "Class-wise concentration",
                        transform=ax_calibration.transAxes, fontsize=7,
                        fontweight="bold")
    fig.subplots_adjust(left=.055, right=.985, top=.92, bottom=.15, wspace=.45)
    fig.savefig(FIGURE_OUT / "gnom_spatial_validation.png",
                dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    comparison, columns = model_columns()
    crs, _affine, paths = write_30m_surfaces(comparison, columns)
    comparison_figure(paths)
    ablation_figure(paths)
    evidence_figure(comparison, paths)
    metadata = {
        "surface_type": "OOF-constrained susceptibility surface",
        "grid_origin": "native 30 m reference raster; not resampled from 450 m",
        "spatialization": "projected tiled k-nearest-neighbour IDW",
        "neighbors": IDW_NEIGHBORS, "power": IDW_POWER,
        "smoothing_m": IDW_SMOOTHING_M,
        "grid_resolution_m": TARGET_RESOLUTION_M,
        "width_px": 4981, "height_px": 10513,
        "common_probability_thresholds": THRESHOLDS.tolist(),
        "coordinate_reference_system": str(crs),
        "model_specific_cartographic_adjustment": False,
    }
    (RESULT_OUT / "mapping_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved native 30 m maps to {MAP_ROOT}", flush=True)


if __name__ == "__main__":
    main()
