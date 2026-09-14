from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from config import (
    FEATURE_ORDER,
    MODEL_ORDER,
    NONRANDOM_ACQUISITION_ORDER,
    AnalysisPaths,
    SECTION_NAMES,
)
from section2_layout import (
    AXIS_LINEWIDTH,
    COMPOSITE_STEM,
    DPI,
    FONT_SIZE_PT,
    HEATMAP_STANDALONE_AXIS_MM,
    LEGEND_GEOMETRY,
    LEGEND_STEMS,
    PAGE_MM,
    PAGE_PX,
    PANEL_LABEL_SIZE_PT,
    PANEL_GEOMETRY,
    PANEL_STEMS,
    STANDALONE_PANEL_MM,
    STANDALONE_PANEL_PX,
    TERM_STANDALONE_AXIS_HEIGHT_MM,
)

if TYPE_CHECKING:
    from pipeline import FigureSpec


FACTOR_CODES = {
    "Dataset": "A",
    "Feature": "B",
    "Model": "C",
    "Acquisition": "D",
}
FACTOR_LEGEND_LABELS = {
    "A": "A = Dataset",
    "B": "B = Feature set",
    "C": "C = Prediction model",
    "D": "D = Acquisition rule",
}
FACTOR_COLORS = {
    "A": "#3B6FB6",
    "B": "#4E9F62",
    "C": "#D97947",
    "D": "#8A68A8",
}
TERM_CLASS_COLORS = {
    "main": "#4776A8",
    "interaction": "#D39A45",
    "residual": "#9A9A9A",
}
ACQUISITION_LABELS = {
    "diversity_oneshot": "Diversity (One-shot)",
    "diversity_iterative": "Diversity (Iterative)",
    "coverage": "Coverage",
    "exploration": "Exploration",
    "exploitation": "Exploitation",
    "hybrid": "Hybrid",
}
ACQUISITION_COLORS = {
    "diversity_oneshot": "#3E9C98",
    "diversity_iterative": "#6EA7C9",
    "coverage": "#79A85B",
    "hybrid": "#C65B52",
    "exploration": "#8B7AB3",
    "exploitation": "#D88A4B",
}
ACQUISITION_MARKERS = {
    "diversity_oneshot": "o",
    "diversity_iterative": "s",
    "coverage": "^",
    "exploration": "D",
    "exploitation": "P",
    "hybrid": "X",
}
MODEL_LABELS = {"GPR": "GPR", "RF": "RF", "AE_ENet": "AE-ENet"}
MODEL_COLORS = {"GPR": "#3B6FB6", "RF": "#4E9F62", "AE_ENet": "#D97947"}
FEATURE_LABELS = {
    feature: f"FS {index}" for index, feature in enumerate(FEATURE_ORDER, start=1)
}
UNIT_FONT_FAMILIES = ["Arial", "Microsoft YaHei"]
ACTIVE_GAIN_AXIS_LABEL = "Active-learning performance\ngain (pp)"
DISTRIBUTION_GAIN_AXIS_LABEL = "Active-learning\nperformance gain\n(pp)"
MEDIAN_GAIN_COLORBAR_LABEL = "Median active-learning\nperformance gain (pp)"
AXES_SCALE = 0.64
DISTRIBUTION_AXIS_HEIGHT_SCALE = 2.0 / 3.0
DISTRIBUTION_POINT_SIZE = 18.0


def _center_scale_rect(
    rect: tuple[float, float, float, float],
    scale: float = AXES_SCALE,
) -> tuple[float, float, float, float]:
    x, y, width, height = rect
    return (
        x + 0.5 * width * (1.0 - scale),
        y + 0.5 * height * (1.0 - scale),
        width * scale,
        height * scale,
    )


_BASE_LOCAL_AXES_RECTS = {
    "a": (0.24, 0.07, 0.72, 0.76),
    "b": (0.20, 0.18, 0.76, 0.53),
    "c": (0.22, 0.18, 0.74, 0.53),
    "d": (0.13, 0.20, 0.76, 0.68),
    "e": (0.15, 0.25, 0.82, 0.64),
    "f": (0.15, 0.25, 0.82, 0.64),
}

_BASE_STANDALONE_AXES_RECTS = {
    "a": (0.30, 0.07, 0.66, 0.84),
    "b": (0.22, 0.18, 0.74, 0.68),
    "c": (0.25, 0.18, 0.71, 0.68),
    "d": (0.15, 0.22, 0.64, 0.66),
    "e": (0.18, 0.25, 0.79, 0.64),
    "f": (0.18, 0.25, 0.79, 0.64),
}

_LOCAL_AXES_RECTS = {
    panel: _center_scale_rect(rect)
    for panel, rect in _BASE_LOCAL_AXES_RECTS.items()
}
_STANDALONE_AXES_RECTS = {
    panel: _center_scale_rect(rect)
    for panel, rect in _BASE_STANDALONE_AXES_RECTS.items()
}
_term_x, _term_y, _term_width, _term_height = _STANDALONE_AXES_RECTS["a"]
_term_target_height = (
    TERM_STANDALONE_AXIS_HEIGHT_MM / STANDALONE_PANEL_MM["a"][1]
)
_STANDALONE_AXES_RECTS["a"] = (
    _term_x,
    _term_y + 0.5 * (_term_height - _term_target_height),
    _term_width,
    _term_target_height,
)
_heatmap_x, _heatmap_y, _heatmap_width, _heatmap_height = (
    _STANDALONE_AXES_RECTS["d"]
)
_heatmap_target_width = HEATMAP_STANDALONE_AXIS_MM[0] / STANDALONE_PANEL_MM["d"][0]
_heatmap_target_height = HEATMAP_STANDALONE_AXIS_MM[1] / STANDALONE_PANEL_MM["d"][1]
_STANDALONE_AXES_RECTS["d"] = (
    _heatmap_x + 0.5 * (_heatmap_width - _heatmap_target_width),
    _heatmap_y + 0.5 * (_heatmap_height - _heatmap_target_height),
    _heatmap_target_width,
    _heatmap_target_height,
)
for _distribution_panel in ("e", "f"):
    _distribution_x, _distribution_y, _distribution_width, _distribution_height = (
        _STANDALONE_AXES_RECTS[_distribution_panel]
    )
    _distribution_target_height = (
        _distribution_height * DISTRIBUTION_AXIS_HEIGHT_SCALE
    )
    _STANDALONE_AXES_RECTS[_distribution_panel] = (
        _distribution_x,
        _distribution_y
        + 0.5 * (_distribution_height - _distribution_target_height),
        _distribution_width,
        _distribution_target_height,
    )


def apply_section2_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Microsoft YaHei",
                "Helvetica",
                "DejaVu Sans",
                "sans-serif",
            ],
            "font.size": FONT_SIZE_PT,
            "axes.labelsize": FONT_SIZE_PT,
            "axes.titlesize": FONT_SIZE_PT,
            "xtick.labelsize": FONT_SIZE_PT,
            "ytick.labelsize": FONT_SIZE_PT,
            "legend.fontsize": FONT_SIZE_PT,
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "text.color": "black",
            "legend.labelcolor": "black",
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
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _apply_thin_axis(ax: mpl.axes.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_LINEWIDTH)
    ax.tick_params(axis="both", which="major", width=AXIS_LINEWIDTH, length=2.5)


def _factor_handles() -> list[Patch]:
    return [
        Patch(
            facecolor=FACTOR_COLORS[code],
            edgecolor="none",
            alpha=0.60,
            label=label,
        )
        for code, label in FACTOR_LEGEND_LABELS.items()
    ]


def _acquisition_handles() -> list[Line2D]:
    return [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=ACQUISITION_MARKERS[value],
            markersize=7.0,
            markerfacecolor=ACQUISITION_COLORS[value],
            markeredgecolor="none",
            label=ACQUISITION_LABELS[value],
        )
        for value in NONRANDOM_ACQUISITION_ORDER
    ]


def _symbolic_term(term: str) -> str:
    if term == "Residual":
        return term
    return "×".join(FACTOR_CODES.get(part, part) for part in term.split(" x "))


def _draw_term_contribution(
    ax: mpl.axes.Axes,
    data: pd.DataFrame,
    *,
    include_legend: bool = True,
) -> mpl.axes.Axes:
    frame = data.loc[:, ["term", "variance_contribution_pct"]].copy()
    frame["term_code"] = frame["term"].astype(str).map(_symbolic_term)
    frame["term_class"] = np.select(
        [frame["term_code"].eq("Residual"), frame["term_code"].str.contains("×")],
        ["residual", "interaction"],
        default="main",
    )
    ax.barh(
        frame["term_code"],
        frame["variance_contribution_pct"],
        color=[TERM_CLASS_COLORS[value] for value in frame["term_class"]],
        edgecolor="white",
        linewidth=AXIS_LINEWIDTH,
        height=0.68,
        alpha=0.60,
    )
    ax.invert_yaxis()
    ax.set(
        xlabel="Share of total SS (%)",
        ylabel="",
    )
    ax.grid(axis="x", color="#E5E7E9", linewidth=AXIS_LINEWIDTH)
    ax.set_axisbelow(True)
    if include_legend:
        ax.legend(
            handles=_factor_handles(),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.08),
            ncol=1,
            handlelength=0.9,
            labelspacing=0.25,
            borderaxespad=0,
        )
    _apply_thin_axis(ax)
    return ax


def add_bar_value_labels(
    ax: mpl.axes.Axes,
    bars: mpl.container.BarContainer,
    values: pd.Series | np.ndarray | list[float],
    *,
    center_on_bar_top: bool = False,
) -> list[mpl.text.Annotation]:
    labels = [f"{float(value):.2f}" for value in values]
    annotations = ax.bar_label(
        bars,
        labels=labels,
        padding=2,
        rotation=90,
        color="black",
        fontsize=FONT_SIZE_PT,
        clip_on=False,
    )
    if center_on_bar_top:
        for annotation in annotations:
            annotation.set_verticalalignment("center")
    return annotations


def _draw_grouped_contribution(
    ax: mpl.axes.Axes,
    data: pd.DataFrame,
    *,
    include_legend: bool = True,
) -> mpl.axes.Axes:
    frame = data.loc[:, ["factor", "grouped_contribution_pp"]].copy()
    frame["factor_code"] = frame["factor"].map(FACTOR_CODES)
    frame["factor_code"] = pd.Categorical(
        frame["factor_code"], categories=list("ABCD"), ordered=True
    )
    frame = frame.sort_values("factor_code")
    codes = frame["factor_code"].astype("object").astype(str).tolist()
    x = np.arange(len(frame))
    bars = ax.bar(
        x,
        frame["grouped_contribution_pp"],
        width=0.66,
        color=[FACTOR_COLORS[code] for code in codes],
        edgecolor="white",
        linewidth=AXIS_LINEWIDTH,
        alpha=0.60,
    )
    add_bar_value_labels(
        ax,
        bars,
        frame["grouped_contribution_pp"],
    )
    ax.margins(x=0.10)
    ax.set_xticks(x, labels=codes)
    ax.set_ylim(0.0, 100.0)
    ax.set(
        xlabel="Factor",
        ylabel="Incremental variance\nexplained (%)",
    )
    ax.grid(axis="y", color="#E5E7E9", linewidth=AXIS_LINEWIDTH)
    ax.set_axisbelow(True)
    if include_legend:
        ax.legend(
            handles=_factor_handles(),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.08),
            ncol=1,
            handlelength=0.8,
            columnspacing=0.7,
            labelspacing=0.25,
            borderaxespad=0,
        )
    _apply_thin_axis(ax)
    return ax


def _present_acquisitions(values: pd.Series) -> list[str]:
    present = set(values.astype(str))
    return [value for value in NONRANDOM_ACQUISITION_ORDER if value in present]


def _draw_moderation(
    ax: mpl.axes.Axes,
    data: pd.DataFrame,
    *,
    include_legend: bool = True,
) -> mpl.axes.Axes:
    order = _present_acquisitions(data["acquisition"])
    x_min = float(data["attainable_gain_pp"].min())
    x_max = float(data["attainable_gain_pp"].max())
    if np.isclose(x_min, x_max):
        x_min -= 0.5
        x_max += 0.5
    x_grid = np.linspace(x_min, x_max, 120)
    for acquisition in order:
        subset = data.loc[data["acquisition"].eq(acquisition)]
        color = ACQUISITION_COLORS[acquisition]
        ax.scatter(
            subset["attainable_gain_pp"],
            subset["early_gain_pp"],
            s=12,
            marker=ACQUISITION_MARKERS[acquisition],
            color=color,
            alpha=0.38,
            edgecolors="none",
            label=ACQUISITION_LABELS[acquisition],
        )
        row = subset.iloc[0]
        fitted = float(row["intercept_at_mean_gain"]) + float(row["slope_gain_per_pp"]) * (
            x_grid - float(row["attainable_gain_mean_pp"])
        )
        ax.plot(x_grid, fitted, color=color, linewidth=1.1)
    ax.axhline(0, color="#5D5D5D", linewidth=AXIS_LINEWIDTH)
    ax.set(
        xlabel="Attainable full-pool gain\n(pp)",
        ylabel=ACTIVE_GAIN_AXIS_LABEL,
    )
    ax.xaxis.label.set_fontfamily(UNIT_FONT_FAMILIES)
    ax.yaxis.label.set_fontfamily(UNIT_FONT_FAMILIES)
    ax.grid(color="#E5E7E9", linewidth=AXIS_LINEWIDTH)
    ax.set_axisbelow(True)
    if include_legend:
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.24),
            ncol=2,
            handlelength=1.0,
            columnspacing=0.7,
            labelspacing=0.25,
            borderaxespad=0,
        )
    _apply_thin_axis(ax)
    return ax


def _draw_feature_model_heatmap(ax: mpl.axes.Axes, data: pd.DataFrame) -> mpl.axes.Axes:
    rows = [value for value in MODEL_ORDER if value in set(data["model"])]
    columns = [value for value in FEATURE_ORDER if value in set(data["feature_set"])]
    matrix = data.pivot(
        index="model", columns="feature_set", values="median_early_gain_pp"
    ).reindex(index=rows, columns=columns)
    values = matrix.to_numpy(dtype=float)
    finite = np.abs(values[np.isfinite(values)])
    limit = float(finite.max()) if finite.size else 1.0
    if np.isclose(limit, 0):
        limit = 1.0
    heatmap = sns.heatmap(
        matrix,
        ax=ax,
        cmap="vlag",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        linewidths=AXIS_LINEWIDTH,
        linecolor="white",
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 9.0, "color": "black"},
        xticklabels=[FEATURE_LABELS[value] for value in columns],
        yticklabels=[MODEL_LABELS[value] for value in rows],
        cbar_kws={
            "shrink": 1.0,
            "pad": 0.03,
        },
    )
    if heatmap.collections[0].colorbar is not None:
        heatmap.collections[0].colorbar.set_label(
            MEDIAN_GAIN_COLORBAR_LABEL,
            fontfamily=UNIT_FONT_FAMILIES,
        )
    ax.set(
        xlabel="Feature set",
        ylabel="Prediction model",
    )
    ax.tick_params(axis="both", length=0)
    _apply_thin_axis(ax)
    return ax


def _draw_gain_distribution(
    ax: mpl.axes.Axes,
    data: pd.DataFrame,
    group_column: str,
) -> mpl.axes.Axes:
    if group_column == "feature_set":
        order = [value for value in FEATURE_ORDER if value in set(data[group_column])]
        labels = [FEATURE_LABELS[value] for value in order]
        colors = [FACTOR_COLORS["B"]] * len(order)
        xlabel = "Feature set"
    else:
        order = [value for value in MODEL_ORDER if value in set(data[group_column])]
        labels = [MODEL_LABELS[value] for value in order]
        colors = [MODEL_COLORS[value] for value in order]
        xlabel = "Prediction model"

    values = [
        data.loc[data[group_column].eq(value), "early_gain_pp"].dropna().to_numpy(dtype=float)
        for value in order
    ]
    positions = np.arange(1, len(order) + 1)
    box = ax.boxplot(
        values,
        positions=positions,
        widths=0.56,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#2A2A2A", "linewidth": 0.70},
        whiskerprops={"color": "#2A2A2A", "linewidth": AXIS_LINEWIDTH},
        capprops={"color": "#2A2A2A", "linewidth": AXIS_LINEWIDTH},
    )
    rng = np.random.default_rng(13)
    for index, (color, group_values) in enumerate(zip(colors, values, strict=True), start=1):
        box["boxes"][index - 1].set_facecolor(color)
        box["boxes"][index - 1].set_alpha(0.25)
        box["boxes"][index - 1].set_edgecolor("#2A2A2A")
        box["boxes"][index - 1].set_linewidth(AXIS_LINEWIDTH)
        if group_values.size:
            jitter = rng.normal(0.0, 0.055, size=group_values.size)
            ax.scatter(
                np.full(group_values.size, index) + jitter,
                group_values,
                s=DISTRIBUTION_POINT_SIZE,
                color=color,
                alpha=0.30,
                edgecolors="none",
                zorder=2,
            )
            ax.scatter(
                index,
                float(np.median(group_values)),
                s=26,
                marker="D",
                color=color,
                edgecolors="#2A2A2A",
                linewidths=AXIS_LINEWIDTH,
                zorder=3,
            )
    ax.axhline(0, color="#5D5D5D", linewidth=AXIS_LINEWIDTH)
    ax.set_xticks(positions, labels=labels)
    ax.set(
        xlabel=xlabel,
        ylabel=DISTRIBUTION_GAIN_AXIS_LABEL,
    )
    ax.yaxis.label.set_fontfamily(UNIT_FONT_FAMILIES)
    ax.grid(axis="y", color="#E5E7E9", linewidth=AXIS_LINEWIDTH)
    ax.set_axisbelow(True)
    _apply_thin_axis(ax)
    return ax


def _draw_panel(
    panel: str,
    ax: mpl.axes.Axes,
    data: pd.DataFrame,
    *,
    include_legend: bool = True,
) -> mpl.axes.Axes:
    if panel == "a":
        return _draw_term_contribution(ax, data, include_legend=include_legend)
    if panel == "b":
        return _draw_grouped_contribution(ax, data, include_legend=include_legend)
    if panel == "c":
        return _draw_moderation(ax, data, include_legend=include_legend)
    if panel == "d":
        return _draw_feature_model_heatmap(ax, data)
    if panel == "e":
        return _draw_gain_distribution(ax, data, "feature_set")
    if panel == "f":
        return _draw_gain_distribution(ax, data, "model")
    raise KeyError(f"Unknown Section 2 panel: {panel}")


def _panel_from_stem(stem: str) -> str:
    for panel, candidate in PANEL_STEMS.items():
        if candidate == stem:
            return panel
    raise KeyError(f"No Section 2 panel is registered for {stem}")


def _source_path(paths: AnalysisPaths, stem: str) -> Path:
    return paths.output_root / SECTION_NAMES[2] / "source_data" / f"{stem}.csv"


def _result_dir(paths: AnalysisPaths) -> Path:
    output = paths.output_root / SECTION_NAMES[2] / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    return output


def _save_fixed_png(
    fig: mpl.figure.Figure,
    output: Path,
    size_px: tuple[int, int],
) -> Path:
    fig.set_size_inches(size_px[0] / DPI, size_px[1] / DPI, forward=True)
    fig.savefig(output, format="png", dpi=DPI, facecolor="white")
    return output


def render_section2_outputs(
    paths: AnalysisPaths,
    specs: list[FigureSpec] | tuple[FigureSpec, ...],
) -> list[Path]:
    apply_section2_style()
    result_dir = _result_dir(paths)
    outputs = []
    for spec in specs:
        panel = _panel_from_stem(spec.stem)
        source = _source_path(paths, spec.stem)
        if not source.exists():
            raise FileNotFoundError(f"Missing same-stem source data for {spec.stem}: {source}")
        data = pd.read_csv(source)
        if data.empty:
            raise ValueError(f"Plot source data are empty for {spec.stem}")
        width_mm, height_mm = STANDALONE_PANEL_MM[panel]
        size_px = STANDALONE_PANEL_PX[panel]
        fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))
        ax = fig.add_axes(_STANDALONE_AXES_RECTS[panel])
        _draw_panel(panel, ax, data, include_legend=False)
        output = result_dir / f"{spec.stem}.png"
        outputs.append(_save_fixed_png(fig, output, size_px))
        plt.close(fig)
    return outputs


def render_section2_legends(paths: AnalysisPaths) -> list[Path]:
    apply_section2_style()
    result_dir = _result_dir(paths)
    outputs = []
    for panel, stem in LEGEND_STEMS.items():
        geometry = LEGEND_GEOMETRY[panel]
        fig = plt.figure(figsize=(geometry.width_mm / 25.4, geometry.height_mm / 25.4))
        handles = _factor_handles() if panel in {"a", "b"} else _acquisition_handles()
        fig.legend(
            handles=handles,
            loc="center",
            ncol=1 if panel in {"a", "b"} else 2,
            handlelength=0.9,
            columnspacing=0.8,
            labelspacing=0.25,
            borderaxespad=0,
        )
        png = result_dir / f"{stem}.png"
        outputs.append(_save_fixed_png(fig, png, (geometry.width_px, geometry.height_px)))
        plt.close(fig)
    return outputs


def _page_axes_rect(panel: str) -> tuple[float, float, float, float]:
    geometry = PANEL_GEOMETRY[panel]
    local_x, local_y, local_w, local_h = _LOCAL_AXES_RECTS[panel]
    x_mm = geometry.x_mm + local_x * geometry.width_mm
    y_bottom_mm = PAGE_MM[1] - geometry.y_mm - geometry.height_mm + local_y * geometry.height_mm
    return (
        x_mm / PAGE_MM[0],
        y_bottom_mm / PAGE_MM[1],
        local_w * geometry.width_mm / PAGE_MM[0],
        local_h * geometry.height_mm / PAGE_MM[1],
    )


def render_section2_composite(paths: AnalysisPaths) -> Path:
    apply_section2_style()
    data_by_panel = {}
    for panel, stem in PANEL_STEMS.items():
        source = _source_path(paths, stem)
        if not source.exists():
            raise FileNotFoundError(f"Missing same-stem source data for {stem}: {source}")
        data_by_panel[panel] = pd.read_csv(source)

    fig = plt.figure(figsize=(PAGE_MM[0] / 25.4, PAGE_MM[1] / 25.4))
    for panel in PANEL_STEMS:
        ax = fig.add_axes(_page_axes_rect(panel))
        _draw_panel(panel, ax, data_by_panel[panel], include_legend=False)
        geometry = PANEL_GEOMETRY[panel]
        fig.text(
            geometry.x_mm / PAGE_MM[0],
            1.0 - geometry.y_mm / PAGE_MM[1],
            panel,
            ha="left",
            va="top",
            fontsize=PANEL_LABEL_SIZE_PT,
            fontweight="bold",
        )

    result_dir = _result_dir(paths)
    png = result_dir / f"{COMPOSITE_STEM}.png"
    _save_fixed_png(fig, png, PAGE_PX)
    plt.close(fig)
    return png
