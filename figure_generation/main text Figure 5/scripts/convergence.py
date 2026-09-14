from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ANALYSIS_ROOT.parent / "common"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.neighbors import KernelDensity

from config import (
    ACQUISITION_ORDER,
    DATASET_ORDER,
    DEFAULT_PATHS,
    FEATURE_ORDER,
)
from convergence_profiles import (
    DatasetProfileBundle,
    build_dataset_profile_bundle,
    load_full_pool_reference,
    load_relative_alc_references,
    select_relative_alc_configurations,
    truncate_fraction_trajectory,
)


MODEL_ORDER = ["GPR", "RF", "AE_ENet"]
MODEL_LABELS = {"GPR": "GPR", "RF": "RF", "AE_ENet": "AE-ENet"}
DATASET_LABELS = {
    "MIT": "MIT",
    "ISU_ILCC": "ISU-ILCC",
    "LSD_Primary": "LSD-primary",
    "LSD_Second": "LSD-second",
    "HUST": "HUST",
    "KIT": "KIT",
    "Formation": "Formation",
    "TRI_Tesla": "TRI-Tesla",
}
ACQUISITION_LABELS = {
    "random_selection": "Random selection",
    "diversity_oneshot": "Diversity (One-shot)",
    "diversity_iterative": "Diversity (Iterative)",
    "coverage": "Coverage",
    "exploration": "Exploration",
    "exploitation": "Exploitation",
    "hybrid": "Hybrid",
}
ACQUISITION_COLORS = {
    "random_selection": "#777777",
    "diversity_oneshot": "#4C78A8",
    "diversity_iterative": "#8FB6D9",
    "coverage": "#59A14F",
    "exploration": "#B79A45",
    "exploitation": "#E07B54",
    "hybrid": "#B07AA1",
}
ROLE_ORDER = ["Best", "Worst", "Random"]
ROLE_LINESTYLES = {"Best": "-", "Worst": "--", "Random": ":"}
FIGURE_SIZE = (8.2, 10.3)
COORDINATE_LINEWIDTH = 0.40
ALL_CELLS_REGION_COLOR = "#D9D9D9"
ALL_CELLS_REGION_ALPHA = 0.48
PANEL_A_BEST_COLOR = "#E64B35"
PANEL_A_WORST_COLOR = "#4DBBD5"
PANEL_A_SELECTED_ALPHA = 0.76
LIFETIME_SIZE_RANGE = (20.0, 36.0)
CATEGORY_LEGEND_MARKER_SIZE = 7.5
PANEL_B_SHARED_YLABEL_OFFSET = 0.045
PANEL_C_SHARED_YLABEL_OFFSET = 0.065
PANEL_A_HORIZONTAL_SPACE = 0.32
LOWER_PANEL_WIDTH_RATIOS = (0.82, 1.50)
LOWER_PANEL_HORIZONTAL_SPACE = 0.35
PANEL_C_HORIZONTAL_SPACE = 0.25
PANEL_C_RIDGE_HEIGHT_FACTOR = 1.40
PANEL_C_LEGEND_X_OFFSET = 0.03
PANEL_C_MAX_LABELLED_FRACTION = 0.60
PANEL_C_ERROR_XLIM = (-100.0, 100.0)
PANEL_C_ERROR_XTICKS = (-100.0, 0.0, 100.0)
PANEL_C_BOTTOM_HEADROOM_SCALES = 0.20
PANEL_C_TOP_HEADROOM_SCALES = 3.60
PANEL_C_ZERO_REFERENCE_TOP_SCALE = 1.10
SHARED_XLABEL_OFFSET = 0.018
RIDGE_FILL_ALPHA = 0.18
PANEL_B_RANGE_ALPHA = 0.10
PANEL_B_INTERVAL_MODE = "sem"
PANEL_B_MEAN_LINEWIDTH = 1.60
PANEL_B_MEAN_ZORDER = 6
PANEL_B_MARKER_SIZE = 3.2
PANEL_B_MARKER_EDGEWIDTH = 0.45
PANEL_B_MARKER_ZORDER = 8
PANEL_B_MARKERS = ("o", "^", "s")
PANEL_B_MARKER_OFFSETS = (0, 1, 2)
PANEL_LABEL_FONT_SIZE = 9.5
PANEL_LABEL_X_OFFSET = 0.035
PANEL_LABEL_Y_OFFSET = 0.025
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results_v2"


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8.2,
        "axes.titlesize": 8.2,
        "axes.labelsize": 8.2,
        "axes.linewidth": COORDINATE_LINEWIDTH,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "legend.fontsize": 7.6,
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.6,
        "xtick.major.width": COORDINATE_LINEWIDTH,
        "ytick.major.width": COORDINATE_LINEWIDTH,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def role_rule_label(role: str, acquisition: str) -> str:
    label = ACQUISITION_LABELS.get(acquisition, str(acquisition))
    return label if role == "Random" else f"{role} · {label}"


def feature_set_label(feature_set: str) -> str:
    match = str(feature_set).removeprefix("set").split("_")[0]
    return f"FS {match}"


def size_by_lifetime(values: pd.Series | np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    numeric = np.asarray(values, dtype=float)
    low, high = LIFETIME_SIZE_RANGE
    if maximum <= minimum:
        return np.full(numeric.shape, (low + high) / 2.0)
    scaled = np.clip((numeric - minimum) / (maximum - minimum), 0.0, 1.0)
    return low + scaled * (high - low)


def _half_circle_marker(side: str, n_points: int = 32) -> MplPath:
    if side == "left":
        angles = np.linspace(np.pi / 2, 3 * np.pi / 2, n_points)
    elif side == "right":
        angles = np.linspace(-np.pi / 2, np.pi / 2, n_points)
    else:
        raise ValueError(f"Unknown half-circle side: {side}")
    vertices = [(0.0, 0.0), *zip(np.cos(angles), np.sin(angles)), (0.0, 0.0)]
    codes = [MplPath.MOVETO, *([MplPath.LINETO] * n_points), MplPath.CLOSEPOLY]
    return MplPath(vertices, codes)


OVERLAP_MARKERS = {
    "left": _half_circle_marker("left"),
    "right": _half_circle_marker("right"),
}


def _config_for_model(bundle: Any, model: str) -> pd.Series:
    rows = bundle.model_configs.loc[bundle.model_configs["model"].eq(model)]
    if len(rows) != 1:
        raise ValueError(f"Expected one configuration for model={model}; observed {len(rows)}")
    return rows.iloc[0]


def _style_axis(axis: plt.Axes) -> None:
    for spine in axis.spines.values():
        spine.set_linewidth(COORDINATE_LINEWIDTH)
    axis.tick_params(
        direction="out",
        length=3.0,
        width=COORDINATE_LINEWIDTH,
        pad=1.8,
    )
    axis.grid(False)


def draw_all_cells_region(axis: plt.Axes, embedding: pd.DataFrame) -> Any:
    """Draw all pool and hold-out cells as a translucent filled KDE region."""

    points = embedding[["embedding_x", "embedding_y"]].to_numpy(dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    if points.size == 0:
        raise ValueError("All-cell embedding region requires finite coordinates")
    grid_axis = np.linspace(-0.04, 1.04, 160)
    xx, yy = np.meshgrid(grid_axis, grid_axis)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    kde = KernelDensity(kernel="gaussian", bandwidth=0.065).fit(points)
    density = np.exp(kde.score_samples(grid)).reshape(xx.shape)
    threshold = float(np.quantile(density, 0.56))
    peak = float(np.max(density))
    if not np.isfinite(peak) or peak <= threshold:
        raise ValueError("All-cell embedding KDE does not define a filled region")
    region = axis.contourf(
        xx,
        yy,
        density,
        levels=[threshold, np.nextafter(peak, np.inf)],
        colors=[ALL_CELLS_REGION_COLOR],
        alpha=ALL_CELLS_REGION_ALPHA,
        antialiased=True,
        zorder=1,
    )
    region.set_gid("all_cells_region")
    return region


def _plot_split_overlap(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    best_color: str,
    worst_color: str,
    sizes: np.ndarray,
) -> None:
    for marker, color in (
        (OVERLAP_MARKERS["left"], best_color),
        (OVERLAP_MARKERS["right"], worst_color),
    ):
        axis.scatter(
            x,
            y,
            s=sizes,
            c=color,
            marker=marker,
            edgecolors="none",
            linewidths=0,
            alpha=PANEL_A_SELECTED_ALPHA,
            zorder=5,
        )


def _panel_a_legend_handles() -> list[Line2D | Patch]:
    return [
        Patch(
            facecolor=ALL_CELLS_REGION_COLOR,
            edgecolor="none",
            alpha=ALL_CELLS_REGION_ALPHA,
            label="All cells",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=PANEL_A_BEST_COLOR,
            markeredgecolor="none",
            markersize=CATEGORY_LEGEND_MARKER_SIZE,
            alpha=PANEL_A_SELECTED_ALPHA,
            label="Best-performing rule",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=PANEL_A_WORST_COLOR,
            markeredgecolor="none",
            markersize=CATEGORY_LEGEND_MARKER_SIZE,
            alpha=PANEL_A_SELECTED_ALPHA,
            label="Worst-performing rule",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=PANEL_A_BEST_COLOR,
            markerfacecoloralt=PANEL_A_WORST_COLOR,
            fillstyle="left",
            markeredgecolor="none",
            markersize=CATEGORY_LEGEND_MARKER_SIZE,
            alpha=PANEL_A_SELECTED_ALPHA,
            label="Overlap",
        ),
    ]


def _plot_embedding_axis(axis: plt.Axes, bundle: Any, model: str) -> None:
    config = _config_for_model(bundle, model)
    panel = bundle.embedding_source.loc[
        bundle.embedding_source["model"].eq(model)
    ].copy()
    if panel.empty:
        raise ValueError(f"Embedding source is empty for model={model}")
    draw_all_cells_region(axis, panel)
    best_rule = str(config["best_acquisition"])
    worst_rule = str(config["worst_acquisition"])
    lifetime_min = float(panel["lifetime"].min())
    lifetime_max = float(panel["lifetime"].max())
    best_color = PANEL_A_BEST_COLOR
    worst_color = PANEL_A_WORST_COLOR
    for membership, color in (
        ("best_only", best_color),
        ("worst_only", worst_color),
    ):
        selected = panel.loc[panel["selection_membership"].eq(membership)]
        axis.scatter(
            selected["embedding_x"],
            selected["embedding_y"],
            s=size_by_lifetime(selected["lifetime"], lifetime_min, lifetime_max),
            c=color,
            marker="o",
            edgecolors="none",
            linewidths=0,
            alpha=PANEL_A_SELECTED_ALPHA,
            zorder=4,
        )
    overlap = panel.loc[panel["selection_membership"].eq("overlap")]
    _plot_split_overlap(
        axis,
        overlap["embedding_x"].to_numpy(float),
        overlap["embedding_y"].to_numpy(float),
        best_color,
        worst_color,
        size_by_lifetime(overlap["lifetime"], lifetime_min, lifetime_max),
    )
    axis.set_xlim(-0.04, 1.04)
    axis.set_ylim(-0.04, 1.04)
    axis.set_box_aspect(1.0)
    axis.set_anchor("C")
    axis.set_xticks([0.0, 0.5, 1.0])
    axis.set_yticks([0.0, 0.5, 1.0])
    axis.set_xlabel("t-SNE dim 1", labelpad=1.5)
    if model == MODEL_ORDER[0]:
        axis.set_ylabel("t-SNE dim 2", labelpad=1.5)
    feature_label = feature_set_label(str(config["feature_set"]))
    axis.set_title(
        f"{MODEL_LABELS[model]} | {feature_label}\n"
        f"Best-performing rule: {ACQUISITION_LABELS[best_rule]}\n"
        f"Worst-performing rule: {ACQUISITION_LABELS[worst_rule]}\n"
        f"Labelled pool fraction: {100.0 * float(config['panel_a_labelled_fraction']):.0f}%",
        loc="center",
        pad=5.0,
        linespacing=1.10,
        fontsize=7.4,
    )
    axis.legend(
        handles=_panel_a_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=1,
        handletextpad=0.35,
        columnspacing=0.65,
        labelspacing=0.70,
        borderaxespad=0.0,
        fontsize=6.8,
    )
    _style_axis(axis)


def draw_panel_a(fig: plt.Figure, subplotspec: Any, bundle: Any) -> list[plt.Axes]:
    grid = subplotspec.subgridspec(1, 3, wspace=PANEL_A_HORIZONTAL_SPACE)
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    for axis, model in zip(axes, MODEL_ORDER, strict=True):
        _plot_embedding_axis(axis, bundle, model)
    return axes


def _plot_mape_axis(
    axis: plt.Axes,
    bundle: Any,
    model: str,
    row_index: int,
    *,
    interval_mode: str = PANEL_B_INTERVAL_MODE,
) -> None:
    if interval_mode not in {"minmax", "sem"}:
        raise ValueError(
            f"Unsupported panel-b interval mode {interval_mode!r}; "
            "expected 'minmax' or 'sem'"
        )
    panel = bundle.mape_source.loc[bundle.mape_source["model"].eq(model)].copy()
    if panel.empty:
        raise ValueError(f"MAPE convergence source is empty for model={model}")
    legend_handles: list[Line2D] = []
    legend_labels: list[str] = []
    mean_lines: list[Line2D] = []
    for role_index, role in enumerate(ROLE_ORDER):
        curve = panel.loc[panel["role"].eq(role)].sort_values("plot_fraction")
        if curve.empty:
            raise ValueError(f"MAPE source is missing role={role}, model={model}")
        acquisition = str(curve["acquisition"].iloc[0])
        color = ACQUISITION_COLORS[acquisition]
        x = curve["plot_fraction"].to_numpy(float)
        mean = curve["mape_mean"].to_numpy(float)
        if interval_mode == "sem":
            if "mape_sem" not in curve:
                raise ValueError("SEM interval mode requires a mape_sem source column")
            sem = curve["mape_sem"].to_numpy(float)
            lower = mean - sem
            upper = mean + sem
        else:
            lower = curve["mape_min"].to_numpy(float)
            upper = curve["mape_max"].to_numpy(float)
        axis.fill_between(
            x,
            lower,
            upper,
            color=color,
            alpha=PANEL_B_RANGE_ALPHA,
            linewidth=0,
            zorder=1,
        )
        mean_line = axis.plot(
            x,
            mean,
            color=color,
            linewidth=PANEL_B_MEAN_LINEWIDTH,
            linestyle="-",
            zorder=PANEL_B_MEAN_ZORDER,
        )[0]
        mean_line.set_path_effects([])
        mean_lines.append(mean_line)
        legend_handles.append(
            Line2D(
                [],
                [],
                color=color,
                linewidth=PANEL_B_MEAN_LINEWIDTH,
                linestyle="-",
                marker=PANEL_B_MARKERS[role_index],
                markersize=PANEL_B_MARKER_SIZE,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=PANEL_B_MARKER_EDGEWIDTH,
            )
        )
        legend_labels.append(ACQUISITION_LABELS.get(acquisition, acquisition))
    for mean_line, marker, offset in zip(
        mean_lines,
        PANEL_B_MARKERS,
        PANEL_B_MARKER_OFFSETS,
        strict=True,
    ):
        x_data = mean_line.get_xdata()
        y_data = mean_line.get_ydata()
        axis.plot(
            x_data[offset::3],
            y_data[offset::3],
            color=mean_line.get_color(),
            linestyle="None",
            marker=marker,
            markersize=PANEL_B_MARKER_SIZE,
            markerfacecolor=mean_line.get_color(),
            markeredgecolor="white",
            markeredgewidth=PANEL_B_MARKER_EDGEWIDTH,
            zorder=PANEL_B_MARKER_ZORDER,
            label="_nolegend_",
        )
    axis.set_xlim(float(panel["plot_fraction"].min()), float(panel["plot_fraction"].max()))
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.xaxis.set_major_locator(MaxNLocator(4))
    axis.yaxis.set_major_locator(MaxNLocator(4, min_n_ticks=3))
    if row_index == 2:
        axis.set_xlabel("Labelled pool fraction (%)", labelpad=5.0)
    else:
        axis.tick_params(labelbottom=False)
    axis.set_ylabel("MAPE (%)", labelpad=5.0)
    config = _config_for_model(bundle, model)
    axis.set_title(
        f"{MODEL_LABELS[model]} | {feature_set_label(str(config['feature_set']))}\n"
        f"Full-pool MAPE: {float(config['mean_full_pool_mape']):.1f}%",
        pad=5.0,
        linespacing=1.05,
    )
    axis.set_box_aspect(1.0)
    axis.set_anchor("C")
    _style_axis(axis)
    axis.grid(
        True,
        axis="y",
        color="#D9D9D9",
        linewidth=0.45,
        alpha=0.55,
    )
    axis.legend(
        legend_handles,
        legend_labels,
        title="Acquisition rule",
        loc="center left",
        bbox_to_anchor=(1.04, 0.5),
        ncol=1,
        handlelength=1.7,
        labelspacing=0.50,
        handletextpad=0.50,
        borderaxespad=0.0,
        fontsize=6.2,
        title_fontsize=6.6,
    )


def draw_panel_b(
    fig: plt.Figure,
    subplotspec: Any,
    bundle: Any,
    *,
    compact: bool = False,
    horizontal_shift: float = 0.0,
    interval_mode: str = PANEL_B_INTERVAL_MODE,
) -> list[plt.Axes]:
    grid = subplotspec.subgridspec(3, 1, hspace=0.40)
    axes = []
    for row_index, model in enumerate(MODEL_ORDER):
        axis = fig.add_subplot(grid[row_index, 0])
        _plot_mape_axis(
            axis,
            bundle,
            model,
            row_index,
            interval_mode=interval_mode,
        )
        if compact:
            axis.set_xlabel("")
            axis.set_ylabel("")
        axes.append(axis)
    if horizontal_shift:
        _translate_axes_horizontally(axes, horizontal_shift)
    return axes


def _ridge_scale(panel: pd.DataFrame, height_factor: float = 1.0) -> float:
    height_factor = float(height_factor)
    if not np.isfinite(height_factor) or height_factor <= 0.0:
        raise ValueError("Ridge height factor must be finite and positive")
    fractions = np.sort(panel["plot_fraction"].dropna().unique())
    if len(fractions) > 1:
        spacing = float(np.min(np.diff(fractions)))
    else:
        spacing = 0.20
    return max(spacing * 0.28, 0.012) * height_factor


def display_fraction_ticks(
    fractions: np.ndarray,
    minimum_gap: float = 0.06,
) -> np.ndarray:
    """Thin displayed fraction labels while retaining every plotted ridge."""

    values = np.sort(np.unique(np.asarray(fractions, dtype=float)))
    if values.size <= 1:
        return values
    kept = [float(values[0])]
    final = float(values[-1])
    for value in values[1:-1]:
        numeric = float(value)
        if numeric - kept[-1] >= minimum_gap and final - numeric >= minimum_gap:
            kept.append(numeric)
    if final - kept[-1] < minimum_gap and len(kept) > 1:
        kept.pop()
    kept.append(final)
    return np.asarray(kept, dtype=float)


def signed_error_ticks(
    lower: float,
    upper: float,
    max_ticks: int = 5,
) -> list[float]:
    """Return sparse symmetric-log ticks that remain legible in narrow axes."""

    lower = float(lower)
    upper = float(upper)
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("Signed-error tick bounds must be finite and increasing")
    if max_ticks < 3:
        raise ValueError("max_ticks must be at least three")
    maximum = max(abs(lower), abs(upper), 10.0)
    maximum_power = int(np.ceil(np.log10(maximum)))
    magnitudes = [float(10**power) for power in range(1, maximum_power + 1)]
    ticks = [
        value
        for value in [
            *[-magnitude for magnitude in reversed(magnitudes)],
            0.0,
            *magnitudes,
        ]
        if lower <= value <= upper
    ]
    if max_ticks == 3 and len(ticks) > max_ticks:
        negative = [value for value in ticks if value < 0.0]
        positive = [value for value in ticks if value > 0.0]
        if negative and positive and 0.0 in ticks:
            return [min(negative), 0.0, max(positive)]
    if len(ticks) > max_ticks:
        ticks = [value for value in ticks if abs(value) != 10.0]
    while len(ticks) > max_ticks:
        nonzero = [value for value in ticks if value != 0.0]
        smallest = min(nonzero, key=abs)
        ticks.remove(smallest)
    return ticks


def plain_number_tick(value: float, _position: int | None = None) -> str:
    """Format signed-error ticks as ordinary decimal numbers."""

    if not np.isfinite(value):
        return ""
    rounded = int(np.rint(value))
    if np.isclose(value, rounded, rtol=0.0, atol=1e-10):
        label = str(rounded)
    else:
        label = np.format_float_positional(
            float(value),
            trim="-",
            precision=6,
        )
    return label.replace("-", "−")


def _plot_ridgeline_axis(
    axis: plt.Axes,
    bundle: Any,
    model: str,
    *,
    ridge_height_factor: float = 1.0,
    legend_x_offset: float = 0.0,
    legend_fontsize: float = 6.2,
    legend_title_fontsize: float = 6.6,
) -> None:
    panel = bundle.kde_source.loc[bundle.kde_source["model"].eq(model)].copy()
    if panel.empty:
        raise ValueError(f"KDE source is empty for model={model}")
    panel = truncate_fraction_trajectory(
        panel,
        target_fraction=PANEL_C_MAX_LABELLED_FRACTION,
        group_columns=["model"],
    )
    scale = _ridge_scale(panel, ridge_height_factor)
    offsets = {role: 0.0 for role in ROLE_ORDER}
    x_min = float(panel["signed_error_grid"].min())
    x_max = float(panel["signed_error_grid"].max())
    for fraction in np.sort(panel["plot_fraction"].dropna().unique()):
        axis.hlines(
            float(fraction),
            x_min,
            x_max,
            color="#777777",
            linewidth=COORDINATE_LINEWIDTH,
            alpha=0.62,
            zorder=0,
        )
    for (fraction, role), curve in panel.groupby(
        ["plot_fraction", "role"],
        sort=True,
    ):
        curve = curve.sort_values("signed_error_grid")
        acquisition = str(curve["acquisition"].iloc[0])
        color = ACQUISITION_COLORS[acquisition]
        x = curve["signed_error_grid"].to_numpy(float)
        density = curve["density_mean"].to_numpy(float)
        sem = curve["density_sem"].to_numpy(float)
        peak = float(np.nanmax(density))
        if not np.isfinite(peak) or peak <= 0:
            continue
        baseline = float(fraction) + offsets[str(role)]
        mean_height = density / peak * scale
        if np.isfinite(sem).all() and np.any(sem > 0):
            lower = np.clip(density - sem, 0.0, None) / peak * scale
            upper = (density + sem) / peak * scale
            axis.fill_between(
                x,
                baseline + lower,
                baseline + upper,
                color=color,
                alpha=0.08,
                linewidth=0,
            )
        axis.fill_between(
            x,
            baseline,
            baseline + mean_height,
            color=color,
            alpha=RIDGE_FILL_ALPHA,
            linewidth=0,
        )
        axis.plot(
            x,
            baseline + mean_height,
            color="#111111",
            linewidth=0.85,
            linestyle=ROLE_LINESTYLES[str(role)],
        )
    axis.set_xlim(*PANEL_C_ERROR_XLIM)
    axis.set_xticks(PANEL_C_ERROR_XTICKS)
    axis.xaxis.set_major_formatter(FuncFormatter(plain_number_tick))
    fractions = np.sort(panel["plot_fraction"].dropna().unique())
    y_min = max(
        0.0,
        float(fractions.min()) - PANEL_C_BOTTOM_HEADROOM_SCALES * scale,
    )
    y_max = float(fractions.max()) + PANEL_C_TOP_HEADROOM_SCALES * scale
    zero_reference = axis.vlines(
        0.0,
        max(y_min, float(fractions.min()) - 0.25 * scale),
        float(fractions.max()) + PANEL_C_ZERO_REFERENCE_TOP_SCALE * scale,
        color="#222222",
        linewidth=COORDINATE_LINEWIDTH,
        linestyles="--",
        alpha=0.75,
        zorder=0,
    )
    zero_reference.set_gid("zero_error_reference")
    axis.set_ylim(y_min, y_max)
    axis.set_yticks(display_fraction_ticks(fractions))
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.set_xlabel("Predicted error (%)", labelpad=5.0)
    if model != MODEL_ORDER[0]:
        axis.tick_params(labelleft=False)
    config = _config_for_model(bundle, model)
    axis.set_title(
        f"{MODEL_LABELS[model]} | {feature_set_label(str(config['feature_set']))}",
        pad=5.0,
    )
    legend_handles = []
    legend_labels = []
    for role in ROLE_ORDER:
        key = {
            "Best": "best_acquisition",
            "Worst": "worst_acquisition",
            "Random": "random_acquisition",
        }[role]
        acquisition = str(config[key])
        legend_handles.append(
            Line2D(
                [],
                [],
                color=ACQUISITION_COLORS[acquisition],
                alpha=1.0,
                linewidth=1.45,
                linestyle=ROLE_LINESTYLES[role],
            )
        )
        legend_labels.append(ACQUISITION_LABELS.get(acquisition, acquisition))
    axis.legend(
        legend_handles,
        legend_labels,
        title="Acquisition rule",
        loc="upper left",
        bbox_to_anchor=(float(legend_x_offset), 1.0),
        handlelength=1.8,
        handletextpad=0.45,
        labelspacing=0.25,
        borderpad=0.0,
        borderaxespad=0.2,
        fontsize=float(legend_fontsize),
        title_fontsize=float(legend_title_fontsize),
    )
    _style_axis(axis)


def draw_panel_c(
    fig: plt.Figure,
    subplotspec: Any,
    bundle: Any,
    *,
    compact: bool = False,
    ridge_height_factor: float = 1.0,
    legend_x_offset: float = 0.0,
    legend_fontsize: float = 6.2,
    legend_title_fontsize: float = 6.6,
) -> list[plt.Axes]:
    grid = subplotspec.subgridspec(1, 3, wspace=PANEL_C_HORIZONTAL_SPACE)
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    for axis, model in zip(axes, MODEL_ORDER, strict=True):
        _plot_ridgeline_axis(
            axis,
            bundle,
            model,
            ridge_height_factor=ridge_height_factor,
            legend_x_offset=legend_x_offset,
            legend_fontsize=legend_fontsize,
            legend_title_fontsize=legend_title_fontsize,
        )
        if compact:
            axis.set_xlabel("")
    if not compact:
        axes[0].set_ylabel("Labelled pool fraction (%)", labelpad=5.0)
    return axes


def _legend_handles(bundle: Any) -> list[Line2D | Patch]:
    handles = [
        Patch(
            facecolor=ALL_CELLS_REGION_COLOR,
            edgecolor="none",
            alpha=ALL_CELLS_REGION_ALPHA,
            label="All cells (pool + hold-out)",
        )
    ]
    observed = []
    for model in MODEL_ORDER:
        config = _config_for_model(bundle, model)
        for role, key in (
            ("Best", "best_acquisition"),
            ("Worst", "worst_acquisition"),
            ("Random", "random_acquisition"),
        ):
            acquisition = str(config[key])
            identity = (role, acquisition)
            if identity not in observed:
                observed.append(identity)
    for role, acquisition in observed:
        handles.append(
            Line2D(
                [0],
                [0],
                color=ACQUISITION_COLORS[acquisition],
                linewidth=1.3,
                linestyle=ROLE_LINESTYLES[role],
                label=role_rule_label(role, acquisition),
            )
        )
    return handles


def _save_figure(
    fig: plt.Figure,
    stem: Path,
    dpi: int,
    formats: tuple[str, ...],
    *,
    tight: bool = True,
) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in formats:
        path = stem.with_suffix(f".{suffix}")
        save_kwargs = {"facecolor": "white"}
        if tight:
            save_kwargs["bbox_inches"] = "tight"
        if suffix == "png":
            save_kwargs["dpi"] = int(dpi)
        fig.savefig(path, **save_kwargs)
        paths.append(path)
    return paths


def render_standalone_panels(bundle: Any, output_dir: Path, dpi: int = 600) -> list[Path]:
    output_dir = Path(output_dir)
    paths = []
    specifications = [
        ("embedding", (7.2, 4.8), draw_panel_a),
        ("mape_convergence", (2.7, 6.5), draw_panel_b),
        ("signed_error_ridgeline", (7.2, 5.5), draw_panel_c),
    ]
    for suffix, size, drawer in specifications:
        figure = plt.figure(figsize=size, constrained_layout=False)
        grid = figure.add_gridspec(1, 1, left=0.09, right=0.985, bottom=0.17, top=0.88)
        drawer(figure, grid[0, 0], bundle)
        path = output_dir / f"{bundle.dataset}_panel_{suffix}.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        paths.append(path)
    return paths


def _add_panel_label(
    figure: plt.Figure,
    axes: list[plt.Axes],
    label: str,
    *,
    x: float | None = None,
    y: float | None = None,
) -> Any:
    if not axes:
        raise ValueError(f"Cannot place panel label {label!r} without axes")
    panel_left = min(axis.get_position().x0 for axis in axes)
    panel_top = max(axis.get_position().y1 for axis in axes)
    return figure.text(
        panel_left - PANEL_LABEL_X_OFFSET if x is None else float(x),
        panel_top + PANEL_LABEL_Y_OFFSET if y is None else float(y),
        label,
        ha="left",
        va="bottom",
        fontsize=PANEL_LABEL_FONT_SIZE,
        fontweight="bold",
    )


def _translate_axes_horizontally(axes: list[plt.Axes], shift: float) -> None:
    for axis in axes:
        position = axis.get_position()
        axis.set_position(position.translated(float(shift), 0.0))


def _text_center_x_in_figure(
    figure: plt.Figure,
    text_artist: Any,
    renderer: Any,
) -> float:
    bounds = text_artist.get_window_extent(renderer=renderer)
    center_display = ((bounds.x0 + bounds.x1) / 2.0, (bounds.y0 + bounds.y1) / 2.0)
    return float(figure.transFigure.inverted().transform(center_display)[0])


def render_dataset_figure(
    bundle: Any,
    output_dir: Path,
    dpi: int = 600,
    *,
    lower_width_ratios: tuple[float, float] = LOWER_PANEL_WIDTH_RATIOS,
    lower_panel_horizontal_space: float = LOWER_PANEL_HORIZONTAL_SPACE,
    panel_c_ridge_height_factor: float = PANEL_C_RIDGE_HEIGHT_FACTOR,
    panel_c_legend_x_offset: float = PANEL_C_LEGEND_X_OFFSET,
    panel_c_legend_fontsize: float = 6.2,
    panel_c_legend_title_fontsize: float = 6.6,
    panel_b_interval_mode: str = PANEL_B_INTERVAL_MODE,
    output_suffix: str = "",
    include_standalone: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=FIGURE_SIZE, constrained_layout=False)
    outer = figure.add_gridspec(
        2,
        1,
        height_ratios=[0.88, 1.22],
        hspace=0.22,
        left=0.105,
        right=0.985,
        bottom=0.070,
        top=0.945,
    )
    lower = outer[1, 0].subgridspec(
        1,
        2,
        width_ratios=lower_width_ratios,
        wspace=float(lower_panel_horizontal_space),
    )
    panel_a_axes = draw_panel_a(figure, outer[0, 0], bundle)
    panel_b_axes = draw_panel_b(
        figure,
        lower[0, 0],
        bundle,
        compact=True,
        interval_mode=panel_b_interval_mode,
    )
    panel_c_axes = draw_panel_c(
        figure,
        lower[0, 1],
        bundle,
        compact=True,
        ridge_height_factor=panel_c_ridge_height_factor,
        legend_x_offset=panel_c_legend_x_offset,
        legend_fontsize=panel_c_legend_fontsize,
        legend_title_fontsize=panel_c_legend_title_fontsize,
    )

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    panel_a_ylabel_center = _text_center_x_in_figure(
        figure,
        panel_a_axes[0].yaxis.label,
        renderer,
    )
    initial_b_left = min(axis.get_position().x0 for axis in panel_b_axes)
    target_b_left = panel_a_ylabel_center + PANEL_B_SHARED_YLABEL_OFFSET
    _translate_axes_horizontally(panel_b_axes, target_b_left - initial_b_left)

    a_left = min(axis.get_position().x0 for axis in panel_a_axes)
    a_top = max(axis.get_position().y1 for axis in panel_a_axes)
    b_left = min(axis.get_position().x0 for axis in panel_b_axes)
    b_right = max(axis.get_position().x1 for axis in panel_b_axes)
    b_bottom = min(axis.get_position().y0 for axis in panel_b_axes)
    b_top = max(axis.get_position().y1 for axis in panel_b_axes)
    c_left = min(axis.get_position().x0 for axis in panel_c_axes)
    c_right = max(axis.get_position().x1 for axis in panel_c_axes)
    c_bottom = min(axis.get_position().y0 for axis in panel_c_axes)
    c_top = max(axis.get_position().y1 for axis in panel_c_axes)

    left_panel_label_x = a_left - PANEL_LABEL_X_OFFSET
    lower_panel_label_y = max(b_top, c_top) + PANEL_LABEL_Y_OFFSET
    panel_c_ylabel_x = c_left - PANEL_C_SHARED_YLABEL_OFFSET
    _add_panel_label(
        figure,
        panel_a_axes,
        "a",
        x=left_panel_label_x,
        y=a_top + PANEL_LABEL_Y_OFFSET,
    )
    _add_panel_label(
        figure,
        panel_b_axes,
        "b",
        x=left_panel_label_x,
        y=lower_panel_label_y,
    )
    _add_panel_label(
        figure,
        panel_c_axes,
        "c",
        x=panel_c_ylabel_x,
        y=lower_panel_label_y,
    )

    figure.text(
        (b_left + b_right) / 2,
        b_bottom - SHARED_XLABEL_OFFSET,
        "Labelled pool fraction (%)",
        ha="center",
        va="top",
    )
    figure.text(
        b_left - PANEL_B_SHARED_YLABEL_OFFSET,
        (b_bottom + b_top) / 2,
        "MAPE (%)",
        rotation=90,
        ha="center",
        va="center",
    )
    figure.text(
        (c_left + c_right) / 2,
        c_bottom - SHARED_XLABEL_OFFSET,
        "Predicted error (%)",
        ha="center",
        va="top",
    )
    figure.text(
        panel_c_ylabel_x,
        (c_bottom + c_top) / 2,
        "Labelled pool fraction (%)",
        rotation=90,
        ha="center",
        va="center",
    )
    stem = output_dir / (
        f"{bundle.dataset}_convergence_selection_profiles{str(output_suffix)}"
    )
    paths = _save_figure(
        figure,
        stem,
        dpi=dpi,
        formats=("png",),
        tight=False,
    )
    standalone_paths = (
        render_standalone_panels(bundle, output_dir, dpi=dpi)
        if include_standalone
        else []
    )
    return {
        "figure": figure,
        "panel_a_axes": panel_a_axes,
        "panel_b_axes": panel_b_axes,
        "panel_c_axes": panel_c_axes,
        "paths": paths + standalone_paths,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def export_dataset_bundle(
    bundle: DatasetProfileBundle,
    output_root: Path,
    *,
    dpi: int = 600,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset_dir = Path(output_root) / bundle.dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    render = render_dataset_figure(bundle, dataset_dir, dpi=dpi)
    source_path = dataset_dir / f"{bundle.dataset}_source_data.csv"
    manifest_path = dataset_dir / f"{bundle.dataset}_manifest.json"
    bundle.source_data.to_csv(source_path, index=False)
    manifest = {
        **bundle.manifest,
        "figure_dpi": int(dpi),
        "figure_archetype": "asymmetric mixed-modality figure",
        "output_paths": [str(path) for path in render["paths"]]
        + [str(source_path), str(manifest_path)],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    qa_rows = qa_dataset_bundle(
        bundle,
        dataset_dir,
        dpi=dpi,
        source_path=source_path,
        manifest_path=manifest_path,
    )
    plt.close(render["figure"])
    return manifest, qa_rows


def _qa_row(dataset: str, check: str, passed: bool, detail: str) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "check": check,
        "passed": bool(passed),
        "detail": detail,
    }


def qa_dataset_bundle(
    bundle: DatasetProfileBundle,
    dataset_dir: Path,
    *,
    dpi: int,
    source_path: Path,
    manifest_path: Path,
) -> list[dict[str, Any]]:
    dataset = bundle.dataset
    kde_trial_support = pd.to_numeric(bundle.kde_source["n_trials"], errors="coerce")
    valid_kde_trial_support = (
        not kde_trial_support.empty
        and np.isfinite(kde_trial_support).all()
        and kde_trial_support.eq(10).all()
        and np.allclose(kde_trial_support, np.round(kde_trial_support))
    )
    kde_sample_support = pd.to_numeric(
        bundle.kde_source.get("n_holdout_samples"), errors="coerce"
    )
    valid_pooled_kde_metadata = (
        not kde_sample_support.empty
        and np.isfinite(kde_sample_support).all()
        and (kde_sample_support >= kde_trial_support).all()
        and np.allclose(kde_sample_support, np.round(kde_sample_support))
        and pd.to_numeric(
            bundle.kde_source["density_sem"], errors="coerce"
        ).eq(0.0).all()
    )
    panel_c_endpoints = (
        bundle.kde_source.groupby(["model", "role"], as_index=False)[
            "plot_fraction"
        ]
        .max()
    )
    panel_c_endpoint_passed = (
        not panel_c_endpoints.empty
        and panel_c_endpoints["plot_fraction"]
        .le(PANEL_C_MAX_LABELLED_FRACTION + 1e-9)
        .all()
        and panel_c_endpoints.groupby("model")["plot_fraction"].nunique().eq(1).all()
    )
    if (
        panel_c_endpoint_passed
        and not bundle.fraction_matches.empty
        and "panel" in bundle.fraction_matches.columns
    ):
        panel_b_fractions = bundle.fraction_matches.loc[
            bundle.fraction_matches["panel"].eq("b")
            & bundle.fraction_matches["plot_fraction"].le(
                PANEL_C_MAX_LABELLED_FRACTION + 1e-9
            )
        ]
        expected_endpoints = panel_b_fractions.groupby("model")[
            "plot_fraction"
        ].max()
        observed_endpoints = panel_c_endpoints.groupby("model")[
            "plot_fraction"
        ].first()
        panel_c_endpoint_passed = (
            set(expected_endpoints.index) == set(observed_endpoints.index)
            and np.allclose(
                expected_endpoints.sort_index().to_numpy(float),
                observed_endpoints.sort_index().to_numpy(float),
                rtol=0.0,
                atol=1e-9,
            )
        )
    expected_files = [
        dataset_dir / f"{dataset}_convergence_selection_profiles.png",
        dataset_dir / f"{dataset}_panel_a_embedding.png",
        dataset_dir / f"{dataset}_panel_b_mape_convergence.png",
        dataset_dir / f"{dataset}_panel_c_signed_error_ridgeline.png",
        source_path,
        manifest_path,
    ]
    rows = [
        _qa_row(
            dataset,
            "required_files",
            all(path.exists() and path.stat().st_size > 0 for path in expected_files),
            (
                f"{sum(path.exists() and path.stat().st_size > 0 for path in expected_files)}"
                f"/{len(expected_files)} non-empty"
            ),
        ),
        _qa_row(
            dataset,
            "model_configurations",
            len(bundle.model_configs) == 3
            and bundle.model_configs["model"].tolist() == MODEL_ORDER,
            f"rows={len(bundle.model_configs)}",
        ),
        _qa_row(
            dataset,
            "mape_trial_support",
            not bundle.mape_source.empty
            and bundle.mape_source["n_trials"].eq(10).all(),
            f"trial_counts={sorted(bundle.mape_source['n_trials'].unique().tolist())}",
        ),
        _qa_row(
            dataset,
            "kde_trial_support",
            valid_kde_trial_support,
            f"trial_counts={sorted(bundle.kde_source['n_trials'].unique().tolist())}",
        ),
        _qa_row(
            dataset,
            "pooled_kde_metadata",
            valid_pooled_kde_metadata,
            (
                f"sample_counts={sorted(kde_sample_support.dropna().unique().tolist())}; "
                "density_sem must be zero"
            ),
        ),
        _qa_row(
            dataset,
            "source_blocks",
            set(bundle.source_data["source_block"].unique())
            == {
                "configuration",
                "embedding",
                "mape_convergence",
                "kde",
                "fraction_match",
            },
            ",".join(sorted(bundle.source_data["source_block"].unique())),
        ),
        _qa_row(
            dataset,
            "embedding_population",
            set(bundle.embedding_source["split_assignment"].unique())
            == {"pool", "holdout"},
            str(bundle.embedding_source["split_assignment"].value_counts().to_dict()),
        ),
        _qa_row(
            dataset,
            "rule_specific_pool_only_overlays",
            not (
                bundle.embedding_source["best_selected"].astype(bool)
                & ~bundle.embedding_source["best_trial_pool"].astype(bool)
            ).any()
            and not (
                bundle.embedding_source["worst_selected"].astype(bool)
                & ~bundle.embedding_source["worst_trial_pool"].astype(bool)
            ).any(),
            "Best/Worst overlays must be pool cells in their own representative trial",
        ),
        _qa_row(
            dataset,
            "finite_summary_values",
            np.isfinite(
                bundle.mape_source[
                    [
                        "plot_fraction",
                        "mape_mean",
                        "mape_min",
                        "mape_max",
                        "mape_sem",
                    ]
                ].to_numpy(float)
            ).all()
            and np.isfinite(
                bundle.kde_source[
                    ["signed_error_grid", "density_mean", "density_sem"]
                ].to_numpy(float)
            ).all(),
            "MAPE and KDE plotted arrays",
        ),
        _qa_row(
            dataset,
            "mape_range_order",
            bundle.mape_source["mape_min"].le(bundle.mape_source["mape_mean"]).all()
            and bundle.mape_source["mape_mean"].le(bundle.mape_source["mape_max"]).all(),
            "mape_min <= mape_mean <= mape_max",
        ),
        _qa_row(
            dataset,
            "panel_c_early_trajectory",
            panel_c_endpoint_passed,
            (
                "model/rule endpoints="
                f"{panel_c_endpoints.to_dict(orient='records')}; target<=60%"
            ),
        ),
    ]
    composite_png = expected_files[0]
    if composite_png.exists():
        with Image.open(composite_png) as image:
            observed_dpi = image.info.get("dpi", (0.0, 0.0))
            width, height = image.size
        dpi_ok = min(observed_dpi) >= max(1.0, float(dpi) - 2.0)
        rows.extend(
            [
                _qa_row(
                    dataset,
                    "png_dpi",
                    dpi_ok,
                    f"observed={observed_dpi}, requested={dpi}",
                ),
                _qa_row(
                    dataset,
                    "png_dimensions",
                    (width, height)
                    == tuple(int(round(value * dpi)) for value in FIGURE_SIZE),
                    (
                        f"pixels={width}x{height}; expected="
                        f"{int(round(FIGURE_SIZE[0] * dpi))}x"
                        f"{int(round(FIGURE_SIZE[1] * dpi))}"
                    ),
                ),
            ]
        )
    else:
        rows.extend(
            [
                _qa_row(dataset, "png_dpi", False, "composite PNG missing"),
                _qa_row(dataset, "png_dimensions", False, "composite PNG missing"),
            ]
        )
    return rows


def preflight_configurations(datasets: list[str]) -> pd.DataFrame:
    full_pool, _ = load_full_pool_reference(DEFAULT_PATHS.output_root)
    relative_config, relative_trial, _ = load_relative_alc_references(
        DEFAULT_PATHS.output_root
    )
    frames = [
        select_relative_alc_configurations(
            full_pool,
            relative_config,
            relative_trial,
            dataset=dataset,
            feature_order=FEATURE_ORDER,
            model_order=MODEL_ORDER,
            acquisition_order=ACQUISITION_ORDER,
        )
        for dataset in datasets
    ]
    return pd.concat(frames, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render per-dataset relative-ALC convergence and selection profiles."
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def build_run_manifest(
    *,
    datasets: list[str],
    configurations: pd.DataFrame,
    dataset_manifests: dict[str, Any],
    qa_path: Path,
    dpi: int,
    qa_passed: bool,
) -> dict[str, Any]:
    return {
        "datasets": list(datasets),
        "n_datasets": len(datasets),
        "n_model_configurations": int(len(configurations)),
        "rule_ranking_metric": "median_relative_improvement_pct",
        "panel_a_target_labelled_fraction": 0.30,
        "panel_b_interval": (
            "10-trial arithmetic mean ± standard error of the mean (SEM)"
        ),
        "panel_c_checkpoint_policy": (
            "common trajectory through the closest checkpoint at or below 60%"
        ),
        "panel_c_max_labelled_fraction": PANEL_C_MAX_LABELLED_FRACTION,
        "panel_c_aggregation": "direct pooled hold-out signed errors across 10 trials",
        "dpi": int(dpi),
        "qa_passed": bool(qa_passed),
        "qa_report": str(qa_path),
        "dataset_manifests": dataset_manifests,
    }


def main() -> int:
    args = parse_args()
    datasets = args.dataset or (list(DATASET_ORDER) if args.all or args.preflight else [])
    unknown = sorted(set(datasets).difference(DATASET_ORDER))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    if not datasets:
        raise SystemExit("Specify --all, --preflight, or at least one --dataset.")
    configurations = preflight_configurations(datasets)
    columns = [
        "dataset",
        "model",
        "feature_set",
        "best_representative_trial",
        "worst_representative_trial",
        "best_acquisition",
        "worst_acquisition",
        "best_median_relative_improvement_pct",
        "worst_median_relative_improvement_pct",
    ]
    print(configurations[columns].to_string(index=False))
    if args.preflight:
        return 0

    output_root = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    output_root.mkdir(parents=True, exist_ok=True)
    all_qa = []
    dataset_manifests = {}
    for dataset in datasets:
        print(f"Building {dataset}...", flush=True)
        bundle = build_dataset_profile_bundle(
            dataset,
            DEFAULT_PATHS,
            feature_order=FEATURE_ORDER,
            model_order=MODEL_ORDER,
            acquisition_order=ACQUISITION_ORDER,
        )
        manifest, qa_rows = export_dataset_bundle(
            bundle,
            output_root,
            dpi=int(args.dpi),
        )
        dataset_manifests[dataset] = manifest
        all_qa.extend(qa_rows)
    qa = pd.DataFrame(all_qa)
    qa_path = output_root / "qa_report.csv"
    qa.to_csv(qa_path, index=False)
    build_manifest = build_run_manifest(
        datasets=datasets,
        configurations=configurations,
        dataset_manifests=dataset_manifests,
        qa_path=qa_path,
        dpi=int(args.dpi),
        qa_passed=bool(qa["passed"].all()),
    )
    build_manifest_path = output_root / "build_manifest.json"
    build_manifest_path.write_text(
        json.dumps(build_manifest, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    if not bool(qa["passed"].all()):
        failed = qa.loc[~qa["passed"], ["dataset", "check", "detail"]]
        raise RuntimeError(f"Required QA checks failed:\n{failed.to_string(index=False)}")
    print(f"QA passed: {len(qa)}/{len(qa)} checks")
    print(f"Output root: {output_root}")
    return 0


if __name__ == "__main__":
    main()
