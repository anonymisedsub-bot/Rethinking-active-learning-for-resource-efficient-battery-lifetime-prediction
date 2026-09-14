from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, FancyArrowPatch, Patch, Wedge
from PIL import Image

from config import (
    DATASET_ORDER,
    FEATURE_ORDER,
    MODEL_ORDER,
    SECTION1_MODEL_ORDER,
    SECTION_NAMES,
    AnalysisPaths,
)
from section1_layout import (
    ANALYTICAL_STEMS,
    AUXILIARY_SPECS,
    AUXILIARY_STEMS,
    DPI,
    LEGEND_SPECS,
    LEGEND_STEMS,
    PANEL_GEOMETRY,
)


SECTION1_MODEL_LABELS = {"AE_ENet": "AE-ENet"}
SECTION1_MODEL_PALETTE = [
    "#3B5BA9",
    "#D84A3A",
    "#3FA184",
    "#E2A62B",
    "#8064A2",
    "#56A8C7",
    "#7D873D",
]
SIGN_COLORS = {
    "Positive (>0)": "#4E9565",
    "Non-positive (<=0)": "#B9BDC2",
}
DATASET_CODES = {dataset: f"D{index}" for index, dataset in enumerate(DATASET_ORDER, start=1)}
FEATURE_CODES = {feature: f"F{index}" for index, feature in enumerate(FEATURE_ORDER, start=1)}
GROUPED_COLORS = {"A": "#3B6FB6", "B": "#4E9F62", "C": "#D97947", "D": "#8A68A8"}
TERM_CLASS_COLORS = {
    "Main effect": "#4776A8",
    "Pairwise interaction": "#D39A45",
    "Residual": "#9A9A9A",
}

AXIS_LINEWIDTH = 0.45
FILL_ALPHA = 0.80
BAR_ALPHA = 0.60
VIOLIN_ALPHA = 0.50
FONT_SIZE_PT = 8.0
SIGN_VALUE_FONT_SIZE_PT = 6.0
FEATURE_ORDER_LEGEND_FONT_SIZE_PT = 6.0
CONTRIBUTION_AXES_BOTTOM = 0.20
CONTRIBUTION_AXES_TOP = 0.86
RING_AXES_RECT = (0.04, 0.16, 0.92, 0.74)
RING_COLORBAR_RECT = (0.16, 0.105, 0.68, 0.025)
RING_RADIUS = 1.0
INNER_RADIUS = 0.38
RING_WIDTH = 0.18
CELL_LINEWIDTH = AXIS_LINEWIDTH
DATASET_LINEWIDTH = 0.60
SECTOR_GAP_DEG = 2.8
VERTICAL_COLORBAR_SPECS = {
    "fig_legend_colorbar_full_pool_performance": (
        "full_pool_mape",
        "MAPE (%)",
        "mako_r",
        False,
    ),
    "fig_legend_colorbar_attainable_full_pool_gain": (
        "attainable_gain_pp",
        "Attainable full-pool gain (pp)",
        "vlag",
        True,
    ),
    "fig_legend_colorbar_initialisation_gain": (
        "initialisation_gain_pp",
        "Initialisation gain (pp)",
        "vlag",
        True,
    ),
}


def apply_section1_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": FONT_SIZE_PT,
            "axes.labelsize": FONT_SIZE_PT,
            "axes.titlesize": FONT_SIZE_PT,
            "xtick.labelsize": FONT_SIZE_PT,
            "ytick.labelsize": FONT_SIZE_PT,
            "legend.fontsize": FONT_SIZE_PT,
            "axes.linewidth": AXIS_LINEWIDTH,
            "xtick.major.width": AXIS_LINEWIDTH,
            "ytick.major.width": AXIS_LINEWIDTH,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _apply_thin_axis(ax: mpl.axes.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_LINEWIDTH)
    ax.tick_params(axis="both", which="major", width=AXIS_LINEWIDTH, length=2.5)


def _figsize(panel: str) -> tuple[float, float]:
    geometry = PANEL_GEOMETRY[panel]
    return geometry.width_px / DPI, geometry.height_px / DPI


def save_fixed_png(
    fig: mpl.figure.Figure,
    output: Path,
    size_px: tuple[int, int],
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(size_px[0] / DPI, size_px[1] / DPI, forward=True)
    fig.savefig(
        output,
        format="png",
        dpi=DPI,
        bbox_inches=None,
        pad_inches=0,
        facecolor="white",
    )
    plt.close(fig)
    with Image.open(output) as image:
        if image.size != size_px:
            raise ValueError(
                f"Fixed-size export mismatch for {output.name}: {image.size} != {size_px}"
            )
    return output


def draw_violin_panel(
    data: pd.DataFrame,
    title: str = "Full-pool prediction performance",
) -> mpl.figure.Figure:
    required = {"model", "full_pool_mape"}
    missing_columns = sorted(required.difference(data.columns))
    if missing_columns:
        raise KeyError(f"Seven-model violin source missing columns: {missing_columns}")
    observed_models = set(data["model"].dropna())
    expected_models = set(SECTION1_MODEL_ORDER)
    if observed_models != expected_models:
        missing = sorted(expected_models - observed_models)
        extra = sorted(observed_models - expected_models)
        raise ValueError(
            f"Incomplete seven-model violin source: missing={missing}; extra={extra}"
        )
    if not np.isfinite(data["full_pool_mape"].to_numpy(dtype=float)).all():
        raise ValueError("Seven-model violin source contains non-finite MAPE values")

    apply_section1_style()
    order = SECTION1_MODEL_ORDER
    fig, ax = plt.subplots(figsize=_figsize("a"))
    sns.violinplot(
        data=data,
        x="model",
        y="full_pool_mape",
        order=order,
        hue="model",
        hue_order=order,
        palette=SECTION1_MODEL_PALETTE,
        legend=False,
        inner=None,
        cut=0,
        linewidth=AXIS_LINEWIDTH,
        saturation=1,
        ax=ax,
    )
    for collection in ax.collections:
        collection.set_alpha(VIOLIN_ALPHA)
    ax.boxplot(
        [
            data.loc[data["model"].eq(model), "full_pool_mape"].dropna().to_numpy()
            for model in order
        ],
        positions=np.arange(len(order)),
        widths=0.14,
        patch_artist=True,
        showfliers=False,
        zorder=3,
        boxprops={
            "facecolor": "white",
            "edgecolor": "#343434",
            "linewidth": AXIS_LINEWIDTH,
        },
        whiskerprops={"color": "#343434", "linewidth": AXIS_LINEWIDTH},
        capprops={"color": "#343434", "linewidth": AXIS_LINEWIDTH},
        medianprops={"color": "#1F1F1F", "linewidth": 0.70},
    )
    ax.set_xticks(
        np.arange(len(order)),
        labels=[SECTION1_MODEL_LABELS.get(model, model) for model in order],
        rotation=25,
        ha="right",
    )
    ax.set(xlabel="", ylabel="MAPE (%)", title=title)
    ax.grid(axis="y", color="#E5E7E9", linewidth=0.45)
    ax.set_axisbelow(True)
    _apply_thin_axis(ax)
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.25, top=0.88)
    return fig


def draw_sign_panel(
    data: pd.DataFrame,
    title: str,
    ylabel: str = "Configurations (%)",
) -> mpl.figure.Figure:
    apply_section1_style()
    order = [label for label in SIGN_COLORS if label in set(data["sign"])]
    frame = data.set_index("sign").reindex(order).reset_index()
    fig, ax = plt.subplots(figsize=_figsize("b"))
    values = frame["fraction"].to_numpy(dtype=float) * 100.0
    bars = ax.bar(
        np.arange(len(frame)),
        values,
        width=0.62,
        color=[SIGN_COLORS[label] for label in frame["sign"]],
        edgecolor="white",
        linewidth=AXIS_LINEWIDTH,
        alpha=BAR_ALPHA,
    )
    for bar, value, count in zip(bars, values, frame["count"]):
        center = bar.get_x() + bar.get_width() / 2
        if value < 15.0:
            ax.text(
                center,
                value + 2.0,
                f"{value:.2f}%\n(n={int(count)})",
                ha="center",
                va="bottom",
                color="#252525",
                fontsize=SIGN_VALUE_FONT_SIZE_PT,
                linespacing=1.0,
            )
            continue
        label_y = value - 3.0 if value >= 94.0 else value + 2.0
        va = "top" if value >= 94.0 else "bottom"
        color = "white" if value >= 94.0 else "#252525"
        ax.text(
            center,
            label_y,
            f"{value:.2f}%",
            ha="center",
            va=va,
            color=color,
            fontsize=SIGN_VALUE_FONT_SIZE_PT,
        )
        ax.text(
            center,
            max(3.0, min(value * 0.48, value - 7.0)),
            f"n={int(count)}",
            ha="center",
            va="center",
            color="white" if value >= 18.0 else "#252525",
            fontsize=SIGN_VALUE_FONT_SIZE_PT,
        )
    ax.set_xticks([0, 1], labels=["Positive", "Non-positive"], rotation=22, ha="right")
    ax.set(
        xlabel="",
        ylabel=ylabel,
        title=title,
        ylim=(0, 100),
        xlim=(-0.75, 1.75),
    )
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color="#E5E7E9", linewidth=0.45)
    ax.set_axisbelow(True)
    _apply_thin_axis(ax)
    fig.subplots_adjust(left=0.34, right=0.98, bottom=0.25, top=0.86)
    return fig


def validate_ring_source(data: pd.DataFrame, value: str) -> None:
    required = {"dataset", "feature_set", "model", value}
    missing_columns = sorted(required.difference(data.columns))
    if missing_columns:
        raise KeyError(f"Ring source missing columns: {missing_columns}")
    keys = ["dataset", "feature_set", "model"]
    duplicated = data.duplicated(keys, keep=False)
    if duplicated.any():
        preview = data.loc[duplicated, keys].head(5).to_dict("records")
        raise ValueError(f"Duplicate ring cells: {preview}")
    expected = pd.MultiIndex.from_product(
        [DATASET_ORDER, FEATURE_ORDER, MODEL_ORDER],
        names=keys,
    )
    observed = pd.MultiIndex.from_frame(data[keys])
    missing = expected.difference(observed)
    if len(missing):
        preview = [tuple(item) for item in missing[:6]]
        raise ValueError(f"Missing ring cells: {preview}")
    if not np.isfinite(data[value].to_numpy(dtype=float)).all():
        raise ValueError(f"Ring source contains non-finite {value} values")


def _ring_norm(data: pd.DataFrame, value: str, diverging: bool) -> Normalize:
    values = data[value].to_numpy(dtype=float)
    if diverging:
        limit = float(np.max(np.abs(values)))
        if limit == 0:
            limit = 1.0
        return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    upper = float(np.ceil(np.quantile(values, 0.98) / 5.0) * 5.0)
    if upper <= 0:
        upper = float(np.max(values)) or 1.0
    return Normalize(vmin=0.0, vmax=upper)


def _ring_colorbar_ticks(norm: Normalize, diverging: bool) -> np.ndarray:
    if diverging:
        return np.linspace(float(norm.vmin), float(norm.vmax), 5)

    lower = float(norm.vmin)
    upper = float(norm.vmax)
    step = max(5.0, float(np.ceil(((upper - lower) / 3.0) / 5.0) * 5.0))
    ticks = np.arange(lower, upper + step * 0.5, step)
    ticks = ticks[ticks <= upper + 1e-9]
    if not len(ticks) or not np.isclose(ticks[-1], upper):
        ticks = np.append(ticks, upper)
    return ticks


def draw_ring_panel(
    data: pd.DataFrame,
    value: str,
    title: str,
    cmap: str,
    diverging: bool,
) -> mpl.figure.Figure:
    validate_ring_source(data, value)
    apply_section1_style()
    fig = plt.figure(figsize=_figsize("d"))
    ax = fig.add_axes(RING_AXES_RECT, aspect="equal")
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.18, 1.18)

    lookup = data.set_index(["dataset", "feature_set", "model"])[value].to_dict()
    norm = _ring_norm(data, value, diverging)
    color_map = sns.color_palette(cmap, as_cmap=True)
    sector_width = 360.0 / len(DATASET_ORDER)
    feature_width = (sector_width - SECTOR_GAP_DEG) / len(FEATURE_ORDER)
    outer_radius = INNER_RADIUS + len(MODEL_ORDER) * RING_WIDTH
    patches: list[Wedge] = []
    values: list[float] = []

    for dataset_index, dataset in enumerate(DATASET_ORDER):
        sector_start = 90.0 - dataset_index * sector_width - SECTOR_GAP_DEG / 2.0
        sector_end = sector_start - (sector_width - SECTOR_GAP_DEG)
        sector_mid = (sector_start + sector_end) / 2.0
        theta_mid = np.deg2rad(sector_mid)
        dataset_label_radius = outer_radius + 0.075
        ax.text(
            dataset_label_radius * np.cos(theta_mid),
            dataset_label_radius * np.sin(theta_mid),
            DATASET_CODES[dataset],
            ha="center",
            va="center",
            fontsize=FONT_SIZE_PT,
            fontweight="bold",
        )
        boundary = np.deg2rad(sector_start + SECTOR_GAP_DEG / 2.0)
        ax.plot(
            [INNER_RADIUS * np.cos(boundary), outer_radius * np.cos(boundary)],
            [INNER_RADIUS * np.sin(boundary), outer_radius * np.sin(boundary)],
            color="white",
            linewidth=DATASET_LINEWIDTH,
            zorder=4,
        )

        for feature_index, feature in enumerate(FEATURE_ORDER):
            cell_start = sector_start - feature_index * feature_width
            cell_end = cell_start - feature_width
            for model_index, model in enumerate(MODEL_ORDER):
                inner = INNER_RADIUS + model_index * RING_WIDTH + 0.007
                outer = INNER_RADIUS + (model_index + 1) * RING_WIDTH - 0.007
                patches.append(
                    Wedge(
                        (0.0, 0.0),
                        outer,
                        theta1=cell_end,
                        theta2=cell_start,
                        width=outer - inner,
                    )
                )
                values.append(float(lookup[(dataset, feature, model)]))

    collection = PatchCollection(
        patches,
        array=np.asarray(values, dtype=float),
        cmap=color_map,
        norm=norm,
        edgecolor="white",
        linewidth=CELL_LINEWIDTH,
    )
    ax.add_collection(collection)
    ax.add_patch(plt.Circle((0, 0), INNER_RADIUS - 0.015, color="white", zorder=3))
    ax.text(
        0,
        0,
        title,
        ha="center",
        va="center",
        fontsize=FONT_SIZE_PT,
        fontweight="bold",
        linespacing=1.05,
        zorder=5,
    )

    fig.text(
        0.50,
        0.975,
        "Rings: GPR | RF | AE-ENet\n(inner to outer)",
        ha="center",
        va="top",
        fontsize=FONT_SIZE_PT,
        linespacing=1.05,
    )

    cax = fig.add_axes(RING_COLORBAR_RECT)
    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=color_map),
        cax=cax,
        orientation="horizontal",
    )
    label_map = {
        "full_pool_mape": "MAPE (%)",
        "attainable_gain_pp": "Attainable full-pool gain (pp)",
        "initialisation_gain_pp": "Initialisation gain (pp)",
    }
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.set_label(label_map.get(value, value), fontsize=FONT_SIZE_PT, labelpad=2)
    colorbar.set_ticks(_ring_colorbar_ticks(norm, diverging))
    colorbar.outline.set_visible(False)
    colorbar.outline.set_linewidth(0.0)
    colorbar.ax.tick_params(
        labelsize=FONT_SIZE_PT,
        length=2.5,
        width=AXIS_LINEWIDTH,
    )
    return fig


def draw_grouped_contribution_panel(data: pd.DataFrame) -> mpl.figure.Figure:
    apply_section1_style()
    frame = data.sort_values("factor_order").copy()
    value_column = (
        "variance_explained_pct"
        if "variance_explained_pct" in frame.columns
        else "grouped_contribution_pp"
    )
    fig, ax = plt.subplots(figsize=_figsize("j"))
    bars = ax.bar(
        frame["factor_code"],
        frame[value_column],
        width=0.64,
        color=[GROUPED_COLORS[code] for code in frame["factor_code"]],
        edgecolor="white",
        linewidth=AXIS_LINEWIDTH,
        alpha=BAR_ALPHA,
    )
    ymax = max(1.0, float(frame[value_column].max()) * 1.20)
    for bar, value in zip(bars, frame[value_column]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.025,
            f"{float(value):.2f}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_PT,
        )
    ax.set(
        xlabel="Factor",
        ylabel="Incremental variance explained (%)",
        title="Grouped factor contributions",
        ylim=(0, ymax),
    )
    ax.grid(axis="y", color="#E5E7E9", linewidth=0.45)
    ax.set_axisbelow(True)
    _apply_thin_axis(ax)
    fig.subplots_adjust(
        left=0.24,
        right=0.97,
        bottom=CONTRIBUTION_AXES_BOTTOM,
        top=CONTRIBUTION_AXES_TOP,
    )
    return fig


def draw_term_contribution_panel(data: pd.DataFrame) -> mpl.figure.Figure:
    apply_section1_style()
    frame = data.sort_values("term_order").copy()
    fig, ax = plt.subplots(figsize=_figsize("g"))
    ax.barh(
        frame["term_code"],
        frame["variance_contribution_pct"],
        color=[TERM_CLASS_COLORS[value] for value in frame["term_class"]],
        edgecolor="white",
        linewidth=AXIS_LINEWIDTH,
        height=0.68,
        alpha=BAR_ALPHA,
    )
    ax.invert_yaxis()
    maximum = float(frame["variance_contribution_pct"].max())
    upper = max(10.0, float(np.ceil(maximum / 10.0) * 10.0))
    ax.set_xlim(0, upper)
    ax.set_xticks(np.linspace(0, upper, 3))
    ax.set(
        xlabel="Share of total SS (%)",
        ylabel="",
        title="Term-level contributions",
    )
    ax.grid(axis="x", color="#E5E7E9", linewidth=0.45)
    ax.set_axisbelow(True)
    _apply_thin_axis(ax)
    fig.subplots_adjust(
        left=0.30,
        right=0.96,
        bottom=CONTRIBUTION_AXES_BOTTOM,
        top=CONTRIBUTION_AXES_TOP,
    )
    return fig


def draw_feature_order_legend(data: pd.DataFrame) -> mpl.figure.Figure:
    required = {
        "order",
        "code",
        "direction",
        "theta_start_deg",
        "theta_end_deg",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Feature-order legend is missing columns: {sorted(missing)}")

    frame = data.sort_values("order").reset_index(drop=True)
    expected_codes = [f"FS {index}" for index in range(1, 6)]
    if frame["code"].tolist() != expected_codes:
        raise ValueError("Feature-order legend must contain FS 1-FS 5 in order")
    if not frame["direction"].eq("clockwise").all():
        raise ValueError("Feature-order legend direction must be clockwise")
    if not (frame["theta_start_deg"] > frame["theta_end_deg"]).all():
        raise ValueError("Clockwise feature cells must use decreasing angles")

    spec = LEGEND_SPECS["fig_legend_feature_order_clockwise"]
    fig = plt.figure(figsize=(spec.width_px / DPI, spec.height_px / DPI))
    ax = fig.add_axes((0.03, 0.03, 0.94, 0.94), aspect="equal")
    ax.set_axis_off()
    ax.set_xlim(-1.12, 1.16)
    ax.set_ylim(0.02, 1.16)

    inner_radius = 0.53
    outer_radius = 0.96
    fills = ["#D9DEE5", "#EEF0F3", "#D9DEE5", "#EEF0F3", "#D9DEE5"]

    for row, color in zip(frame.itertuples(index=False), fills, strict=True):
        ax.add_patch(
            Wedge(
                (0.0, 0.0),
                outer_radius,
                theta1=float(row.theta_end_deg),
                theta2=float(row.theta_start_deg),
                width=outer_radius - inner_radius,
                facecolor=color,
                edgecolor="white",
                linewidth=0.45,
            )
        )
    arrow_radius = 0.43
    arrow_start_deg = float(frame.iloc[0]["theta_start_deg"]) - 5.0
    arrow_end_deg = float(frame.iloc[-1]["theta_end_deg"]) + 5.0
    ax.add_patch(
        Arc(
            (0.0, 0.0),
            2.0 * arrow_radius,
            2.0 * arrow_radius,
            theta1=arrow_end_deg,
            theta2=arrow_start_deg,
            color="#30343B",
            linewidth=0.65,
        )
    )
    arrow_tangent_start = np.deg2rad(arrow_end_deg + 8.0)
    arrow_end = np.deg2rad(arrow_end_deg)
    ax.add_patch(
        FancyArrowPatch(
            (
                arrow_radius * np.cos(arrow_tangent_start),
                arrow_radius * np.sin(arrow_tangent_start),
            ),
            (
                arrow_radius * np.cos(arrow_end),
                arrow_radius * np.sin(arrow_end),
            ),
            arrowstyle="-|>",
            mutation_scale=7.0,
            color="#30343B",
            linewidth=0.65,
        )
    )
    ax.text(
        0.70,
        0.10,
        "Clockwise",
        ha="center",
        va="center",
        fontsize=FEATURE_ORDER_LEGEND_FONT_SIZE_PT,
    )
    return fig


def draw_vertical_colorbar_asset(
    stem: str,
    data: pd.DataFrame,
) -> mpl.figure.Figure:
    if stem not in VERTICAL_COLORBAR_SPECS:
        raise KeyError(f"No vertical colorbar specification for {stem}")

    value, label, cmap, diverging = VERTICAL_COLORBAR_SPECS[stem]
    validate_ring_source(data, value)
    spec = LEGEND_SPECS[stem]
    fig = plt.figure(figsize=(spec.width_px / DPI, spec.height_px / DPI))
    cax = fig.add_axes((0.44, 0.08, 0.17, 0.84))
    norm = _ring_norm(data, value, diverging)
    color_map = sns.color_palette(cmap, as_cmap=True)
    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=color_map),
        cax=cax,
        orientation="vertical",
    )
    colorbar.set_ticks(_ring_colorbar_ticks(norm, diverging))
    colorbar.ax.yaxis.set_ticks_position("right")
    colorbar.ax.yaxis.set_label_position("left")
    colorbar.set_label(label, fontsize=FONT_SIZE_PT, labelpad=4)
    colorbar.outline.set_visible(False)
    colorbar.outline.set_linewidth(0.0)
    colorbar.ax.tick_params(
        labelsize=FONT_SIZE_PT,
        length=2.5,
        width=AXIS_LINEWIDTH,
    )
    return fig


def draw_legend_asset(stem: str, data: pd.DataFrame) -> mpl.figure.Figure:
    if stem not in LEGEND_SPECS:
        raise KeyError(f"No Section 1 legend specification for {stem}")
    if data.empty:
        raise ValueError(f"Legend source data are empty for {stem}")

    apply_section1_style()
    if stem == "fig_legend_feature_order_clockwise":
        return draw_feature_order_legend(data)
    if stem in VERTICAL_COLORBAR_SPECS:
        return draw_vertical_colorbar_asset(stem, data)

    spec = LEGEND_SPECS[stem]
    fig = plt.figure(figsize=(spec.width_px / DPI, spec.height_px / DPI))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_axis_off()
    frame = data.sort_values("order").reset_index(drop=True)

    if stem in {"fig_legend_dataset_codes", "fig_legend_feature_codes"}:
        handles = [
            Line2D([], [], linestyle="none", label=f"{row.code}  {row.label}")
            for row in frame.itertuples(index=False)
        ]
        ncol = 4 if stem == "fig_legend_dataset_codes" else 1
        fig.legend(
            handles=handles,
            loc="center",
            ncol=ncol,
            frameon=False,
            handlelength=0,
            handletextpad=0,
            columnspacing=0.7,
            labelspacing=0.22,
            fontsize=FONT_SIZE_PT,
        )
    else:
        factor_legend = stem in {
            "fig_legend_anova_factors_3",
            "fig_legend_anova_factors_4",
        }
        handles = [
            Patch(
                facecolor=row.color,
                edgecolor="none",
                label=f"{row.code}  {row.label}" if factor_legend else str(row.label),
                alpha=FILL_ALPHA,
            )
            for row in frame.itertuples(index=False)
        ]
        ncol = 2 if stem == "fig_legend_anova_factors_4" else len(handles)
        fig.legend(
            handles=handles,
            loc="center",
            ncol=ncol,
            frameon=False,
            handlelength=1.0,
            handleheight=0.8,
            handletextpad=0.4,
            columnspacing=0.7,
            labelspacing=0.35,
            fontsize=FONT_SIZE_PT,
        )
    return fig


def render_legend_asset(stem: str, data: pd.DataFrame, output: Path) -> Path:
    spec = LEGEND_SPECS[stem]
    return save_fixed_png(
        draw_legend_asset(stem, data),
        output,
        (spec.width_px, spec.height_px),
    )


def render_analytical_panel(stem: str, data: pd.DataFrame, output: Path) -> Path:
    geometry = next(
        (item for item in PANEL_GEOMETRY.values() if item.stem == stem),
        None,
    )
    if geometry is None:
        raise KeyError(f"No Section 1 panel geometry for {stem}")

    if geometry.panel == "a":
        fig = draw_violin_panel(data)
    elif geometry.panel == "b":
        fig = draw_sign_panel(data, "Attainable full-pool gain", ylabel="Trials (%)")
    elif geometry.panel == "c":
        fig = draw_sign_panel(data, "Initialisation gain", ylabel="Trials (%)")
    elif geometry.panel == "d":
        fig = draw_ring_panel(
            data,
            "full_pool_mape",
            "Full-pool\nperformance",
            "mako_r",
            diverging=False,
        )
    elif geometry.panel == "e":
        fig = draw_ring_panel(
            data,
            "attainable_gain_pp",
            "Attainable\nfull-pool gain",
            "vlag",
            diverging=True,
        )
    elif geometry.panel == "f":
        fig = draw_ring_panel(
            data,
            "initialisation_gain_pp",
            "Initialisation\ngain",
            "vlag",
            diverging=True,
        )
    elif geometry.panel in {"g", "h", "i"}:
        fig = draw_term_contribution_panel(data)
    elif geometry.panel in {"j", "k", "l"}:
        fig = draw_grouped_contribution_panel(data)
    else:
        raise KeyError(f"No Section 1 renderer for panel {geometry.panel}")

    return save_fixed_png(
        fig,
        output,
        (geometry.width_px, geometry.height_px),
    )


def render_auxiliary_asset(stem: str, data: pd.DataFrame, output: Path) -> Path:
    if stem not in AUXILIARY_SPECS:
        raise KeyError(f"No Section 1 auxiliary specification for {stem}")

    fig = draw_violin_panel(
        data,
        title="Full-pool prediction performance (trial level)",
    )
    spec = AUXILIARY_SPECS[stem]
    return save_fixed_png(fig, output, (spec.width_px, spec.height_px))


def render_section1_outputs(paths: AnalysisPaths, specs: list[object]) -> list[Path]:
    section_dir = paths.output_root / SECTION_NAMES[1]
    requested = list(specs)
    outputs_by_stem: dict[str, Path] = {}

    for spec in requested:
        stem = str(spec.stem)
        source = section_dir / "source_data" / f"{stem}.csv"
        if not source.exists():
            raise FileNotFoundError(f"Missing same-stem source data for {stem}: {source}")
        data = pd.read_csv(source)
        if data.empty:
            raise ValueError(f"Plot source data are empty for {stem}")
        output = section_dir / "outputs" / f"{stem}.png"
        if stem in ANALYTICAL_STEMS:
            outputs_by_stem[stem] = render_analytical_panel(stem, data, output)
        elif stem in LEGEND_STEMS:
            outputs_by_stem[stem] = render_legend_asset(stem, data, output)
        elif stem in AUXILIARY_STEMS:
            outputs_by_stem[stem] = render_auxiliary_asset(stem, data, output)
        else:
            raise KeyError(f"No independent Section 1 renderer for {stem}")

    return [outputs_by_stem[str(spec.stem)] for spec in requested]
