from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from config import (
    ACQUISITION_ORDER,
    DATASET_ORDER,
    DURATION_UNITS,
    FEATURE_ORDER,
    MODEL_ORDER,
    NONRANDOM_ACQUISITION_ORDER,
    AnalysisPaths,
    SECTION_NAMES,
)
from pipeline import (
    FIGURE_SPECS,
    SECTION4_NFP_ALPHA_BY_STEM,
    SECTION4_NFP_KIND_BY_STEM,
    FigureSpec,
)
from section1_figure import render_section1_outputs
from section2_figure import render_section2_outputs


DATASET_LABELS = {
    "MIT": "MIT",
    "ISU_ILCC": "ISU-ILCC",
    "LSD_Primary": "LSD Primary",
    "LSD_Second": "LSD Second",
    "HUST": "HUST",
    "KIT": "KIT",
    "Formation": "Formation",
    "TRI_Tesla": "TRI-Tesla",
}
SECTION4_DATASET_LABELS = {
    **DATASET_LABELS,
    "LSD_Primary": "LSD-primary",
    "LSD_Second": "LSD-second",
}
FEATURE_LABELS = {name: f"FS {i}" for i, name in enumerate(FEATURE_ORDER, start=1)}
COMPACT_DATASET_LABELS = {
    "MIT": "MIT",
    "ISU_ILCC": "ISU",
    "LSD_Primary": "LSD-P",
    "LSD_Second": "LSD-S",
    "HUST": "HUST",
    "KIT": "KIT",
    "Formation": "Form",
    "TRI_Tesla": "TRI",
}
COMPACT_MODEL_LABELS = {"GPR": "GPR", "RF": "RF", "AE_ENet": "AE"}
MODEL_LABELS = {"GPR": "GPR", "RF": "RF", "AE_ENet": "AE-ENet"}
ACQUISITION_LABELS = {
    "random_selection": "Random",
    "diversity_oneshot": "Diversity (One-shot)",
    "diversity_iterative": "Diversity (Iterative)",
    "coverage": "Coverage",
    "exploration": "Exploration",
    "exploitation": "Exploitation",
    "hybrid": "Hybrid",
}
ACQUISITION_SHORT_LABELS = {
    "diversity_oneshot": "D-OS",
    "diversity_iterative": "D-IT",
    "coverage": "Cov",
    "exploration": "Explore",
    "exploitation": "Exploit",
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
SECTION3_ACQUISITION_ORDER = [
    "diversity_oneshot",
    "diversity_iterative",
    "coverage",
    "hybrid",
    "exploration",
    "exploitation",
]
SECTION3_ACQUISITION_COLORS = {
    "diversity_oneshot": "#3E9C98",
    "diversity_iterative": "#6EA7C9",
    "coverage": "#79A85B",
    "hybrid": "#C65B52",
    "exploration": "#8B7AB3",
    "exploitation": "#D88A4B",
}
SECTION3_MARKERS = {
    "diversity_oneshot": "o",
    "diversity_iterative": "P",
    "coverage": "s",
    "hybrid": "D",
    "exploration": "^",
    "exploitation": "v",
}
SECTION3_HEATMAP_FIGSIZE = (4.0, 8.5)
SECTION3_COLORBAR_TICKS = [-4, -2, 0, 2, 4]
SECTION3_TITLE_SIZE = 14.0
SECTION3_LABEL_SIZE = 13.0
SECTION3_TICK_SIZE = 11.5
SECTION3_HEATMAP_TITLE_SIZE = 12.0
SECTION3_HEATMAP_LABEL_SIZE = 9.5
SECTION3_HEATMAP_TICK_SIZE = 8.6
SECTION3_HEATMAP_NUMBER_SIZE = 6.8
SECTION3_COLORBAR_LABEL_SIZE = 11.5
SECTION3_COLORBAR_TICK_SIZE = 10.5
SECTION3_HEATMAP_COLORBAR_LABEL_SIZE = 8.0
SECTION3_HEATMAP_COLORBAR_TICK_SIZE = 8.3
SECTION3_LEGEND_SIZE = 11.0
SECTION3_LEGEND_TITLE_SIZE = 11.5
SECTION4_FIGURE_WIDTH_IN = 13.2
SECTION4_STYLE = {
    "font.size": 9.0,
    "axes.labelsize": 9.0,
    "axes.titlesize": 10.0,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.0,
    "legend.title_fontsize": 9.0,
}
SECTION4_ANNOTATION_SIZE = 7.5
SECTION4_RAW_MARKER_AREA = 24.0
SECTION4_MEDIAN_MARKER_AREA = 64.0
MODEL_COLORS = {"GPR": "#4C78A8", "RF": "#59A14F", "AE_ENet": "#E07B54"}


def set_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.35,
            "xtick.major.width": 0.35,
            "ytick.major.width": 0.35,
            "legend.frameon": False,
            "legend.fontsize": 6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_png(fig: mpl.figure.Figure, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if bool(getattr(fig, "_section3_fixed_canvas", False)) or bool(getattr(fig, "_section4_fixed_canvas", False)):
        fig.savefig(output, dpi=600, facecolor="white")
    else:
        fig.savefig(output, dpi=600, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    return output


def _section3_fixed_canvas(fig: mpl.figure.Figure) -> mpl.figure.Figure:
    setattr(fig, "_section3_fixed_canvas", True)
    return fig


def render_example(source: Path, output: Path) -> Path:
    set_publication_style()
    data = pd.read_csv(source)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(data["x"], data["y"], color="#4C78A8", marker="o", linewidth=1.2, markersize=3)
    ax.set(xlabel="x", ylabel="y")
    fig.tight_layout()
    return save_png(fig, output)


def _present_order(values: pd.Series, preferred: list[str]) -> list[str]:
    observed = set(values.dropna().astype(str))
    return [value for value in preferred if value in observed] + sorted(observed.difference(preferred))


def _label_acquisition_axis(ax: mpl.axes.Axes, order: list[str], rotation: float = 28) -> None:
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([ACQUISITION_LABELS.get(value, value) for value in order], rotation=rotation, ha="right")


def _reader_acquisition_legend(ax: mpl.axes.Axes) -> None:
    legend = ax.get_legend()
    if legend is None:
        return
    for label in legend.get_texts():
        label.set_text(ACQUISITION_LABELS.get(label.get_text(), label.get_text()))


def _zero_line(ax: mpl.axes.Axes, orientation: str = "horizontal") -> None:
    if orientation == "horizontal":
        ax.axhline(0, color="#303030", linewidth=0.5, linestyle="--", zorder=1)
    else:
        ax.axvline(0, color="#303030", linewidth=0.5, linestyle="--", zorder=1)


def _heatmap_landscape(
    data: pd.DataFrame,
    value: str,
    ylabel: str,
    diverging: bool,
) -> mpl.figure.Figure:
    frame = data.copy()
    frame["column"] = frame["model"].astype(str) + "\n" + frame["feature_set"].map(
        lambda value: FEATURE_LABELS.get(value, value)
    )
    datasets = _present_order(frame["dataset"], DATASET_ORDER)
    models = _present_order(frame["model"], MODEL_ORDER)
    features = _present_order(frame["feature_set"], FEATURE_ORDER)
    columns = [f"{model}\n{FEATURE_LABELS.get(feature, feature)}" for model in models for feature in features]
    matrix = frame.pivot_table(index="dataset", columns="column", values=value, aggfunc="mean")
    matrix = matrix.reindex(index=datasets, columns=columns)
    fig, ax = plt.subplots(figsize=(7.15, 3.15))
    kwargs: dict[str, object] = {"cmap": "vlag" if diverging else "mako_r", "linewidths": 0.25, "linecolor": "white"}
    if diverging:
        finite = np.abs(matrix.to_numpy(dtype=float)[np.isfinite(matrix.to_numpy(dtype=float))])
        limit = float(finite.max()) if finite.size else 1.0
        if limit == 0:
            limit = 1.0
        kwargs["norm"] = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    heat = sns.heatmap(
        matrix,
        ax=ax,
        cbar_kws={"label": ylabel, "shrink": 0.78, "pad": 0.02},
        **kwargs,
    )
    heat.set(xlabel="Model and feature representation", ylabel="Ageing dataset")
    ax.set_yticklabels([DATASET_LABELS.get(value, value) for value in matrix.index], rotation=0)
    ax.set_xticklabels([FEATURE_LABELS.get(feature, feature) for _ in models for feature in features], rotation=0)
    for model_index, model in enumerate(models):
        center = model_index * len(features) + len(features) / 2
        ax.text(center, 1.025, model, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontweight="bold")
    for boundary in range(len(features), len(columns), len(features)):
        ax.axvline(boundary, color="#333333", linewidth=0.7)
    fig.tight_layout()
    return fig


def _box_by_acquisition(data: pd.DataFrame, value: str, ylabel: str, include_random: bool) -> mpl.figure.Figure:
    preferred = ACQUISITION_ORDER if include_random else NONRANDOM_ACQUISITION_ORDER
    order = _present_order(data["acquisition"], preferred)
    fig, ax = plt.subplots(figsize=(5.5, 3.05))
    sns.boxplot(
        data=data,
        x="acquisition",
        y=value,
        order=order,
        palette=ACQUISITION_COLORS,
        hue="acquisition",
        legend=False,
        showfliers=False,
        linewidth=0.7,
        width=0.66,
        ax=ax,
    )
    _label_acquisition_axis(ax, order)
    ax.set(xlabel="", ylabel=ylabel)
    _zero_line(ax)
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def _section3_acquisition_order(values: pd.Series) -> list[str]:
    return _present_order(values, SECTION3_ACQUISITION_ORDER)


def _section3_heatmap_colors() -> tuple[mpl.colors.Colormap, TwoSlopeNorm]:
    return mpl.colormaps["RdBu_r"], TwoSlopeNorm(vmin=-4.0, vcenter=0.0, vmax=4.0)


def _annotated_section3_heatmap(
    matrix: pd.DataFrame,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    figsize: tuple[float, float],
) -> mpl.figure.Figure:
    cmap, norm = _section3_heatmap_colors()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix,
        cmap=cmap,
        norm=norm,
        linewidths=0.25,
        linecolor="white",
        cbar=False,
        ax=ax,
    )
    for y_index, (_, row) in enumerate(matrix.iterrows()):
        for x_index, value in enumerate(row):
            if pd.isna(value):
                continue
            text_color = "white" if abs(float(value)) >= 2.8 else "#1F1F1F"
            ax.text(
                x_index + 0.5,
                y_index + 0.5,
                f"{float(value):.2f}",
                ha="center",
                va="center",
                fontsize=SECTION3_HEATMAP_NUMBER_SIZE,
                color=text_color,
    )
    ax.set_title(title, fontsize=SECTION3_HEATMAP_TITLE_SIZE, fontweight="normal", pad=10)
    ax.set_xlabel(xlabel, fontsize=SECTION3_HEATMAP_LABEL_SIZE)
    ax.set_ylabel("")
    ax.tick_params(axis="both", length=0)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=SECTION3_HEATMAP_TICK_SIZE,
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=SECTION3_HEATMAP_TICK_SIZE)
    cax = fig.add_axes([0.31, 0.125, 0.61, 0.018])
    colorbar = fig.colorbar(
        ax.collections[0],
        cax=cax,
        orientation="horizontal",
        ticks=SECTION3_COLORBAR_TICKS,
    )
    colorbar.ax.tick_params(labelsize=SECTION3_HEATMAP_COLORBAR_TICK_SIZE, length=2, width=0.35, pad=1)
    colorbar.outline.set_visible(False)
    for spine in colorbar.ax.spines.values():
        spine.set_visible(False)
    colorbar.ax.xaxis.set_label_position("bottom")
    colorbar.set_label(
        "Early-acquisition performance gain (pp)",
        fontsize=SECTION3_HEATMAP_COLORBAR_LABEL_SIZE,
        labelpad=2,
    )
    fig.subplots_adjust(left=0.31, right=0.94, bottom=0.30, top=0.92)
    return _section3_fixed_canvas(fig)


def _early_gain_effect_distribution(data: pd.DataFrame) -> mpl.figure.Figure:
    frame = data.copy()
    order = _section3_acquisition_order(frame["acquisition"])
    rng = np.random.default_rng(20260820)
    fig, ax = plt.subplots(figsize=(3.0, 4.0))
    for y_index, acquisition in enumerate(order):
        sub = frame.loc[frame["acquisition"].eq(acquisition), "early_gain_pp"].dropna().to_numpy(float)
        jitter = rng.normal(0.0, 0.045, size=len(sub))
        color = SECTION3_ACQUISITION_COLORS.get(acquisition, "#777777")
        ax.scatter(
            sub,
            np.full(len(sub), y_index) + jitter,
            s=16,
            alpha=0.22,
            color=color,
            edgecolors="none",
            zorder=2,
        )
        median = float(np.median(sub))
        ax.scatter(
            median,
            y_index,
            marker="D",
            s=48,
            color=color,
            edgecolor="#303030",
            linewidth=0.28,
            zorder=3,
        )
    ax.axvline(0, color="#606060", linewidth=0.55, zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([ACQUISITION_LABELS.get(value, value) for value in order], fontsize=SECTION3_TICK_SIZE)
    ax.invert_yaxis()
    ax.set_title("Overall distribution", fontsize=SECTION3_TITLE_SIZE, fontweight="normal", pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=SECTION3_TICK_SIZE)
    ax.grid(axis="x", color="#E7E7E7", linewidth=0.35)
    ax.set_axisbelow(True)
    fig.text(
        0.69,
        0.045,
        "Early-acquisition\nperformance gain (pp)",
        ha="center",
        va="bottom",
        fontsize=SECTION3_LABEL_SIZE,
    )
    fig.subplots_adjust(left=0.56, right=0.97, bottom=0.24, top=0.88)
    return _section3_fixed_canvas(fig)


def _early_dataset_model_heatmap(data: pd.DataFrame) -> mpl.figure.Figure:
    frame = data.copy()
    order = _section3_acquisition_order(frame["acquisition"])
    datasets = [value for value in DATASET_ORDER if value in set(frame["dataset"])]
    models = [value for value in MODEL_ORDER if value in set(frame["model"])]
    frame["column"] = (
        frame["dataset"].map(DATASET_LABELS).fillna(frame["dataset"])
        + "\n"
        + frame["model"].map(MODEL_LABELS).fillna(frame["model"])
    )
    columns = [
        f"{DATASET_LABELS.get(dataset, dataset)}\n{MODEL_LABELS.get(model, model)}"
        for dataset in datasets
        for model in models
    ]
    matrix = frame.pivot_table(index="acquisition", columns="column", values="early_gain_pp", aggfunc="mean")
    matrix = matrix.reindex(index=order, columns=columns).T
    matrix.columns = [ACQUISITION_LABELS.get(value, value) for value in matrix.columns]
    return _annotated_section3_heatmap(
        matrix,
        title="Dataset-model dependence",
        xlabel="",
        ylabel="Dataset and prediction model",
        figsize=SECTION3_HEATMAP_FIGSIZE,
    )


def _early_feature_model_heatmap(data: pd.DataFrame) -> mpl.figure.Figure:
    frame = data.copy()
    order = _section3_acquisition_order(frame["acquisition"])
    features = [value for value in FEATURE_ORDER if value in set(frame["feature_set"])]
    models = [value for value in MODEL_ORDER if value in set(frame["model"])]
    frame["column"] = (
        frame["feature_set"].map(FEATURE_LABELS).fillna(frame["feature_set"])
        + "\n"
        + frame["model"].map(MODEL_LABELS).fillna(frame["model"])
    )
    columns = [
        f"{FEATURE_LABELS.get(feature, feature)}\n{MODEL_LABELS.get(model, model)}"
        for feature in features
        for model in models
    ]
    matrix = frame.pivot_table(index="acquisition", columns="column", values="early_gain_pp", aggfunc="mean")
    matrix = matrix.reindex(index=order, columns=columns).T
    matrix.columns = [ACQUISITION_LABELS.get(value, value) for value in matrix.columns]
    return _annotated_section3_heatmap(
        matrix,
        title="Feature-model dependence",
        xlabel="",
        ylabel="Feature representation and prediction model",
        figsize=SECTION3_HEATMAP_FIGSIZE,
    )


def _early_heatmap(data: pd.DataFrame, row: str, row_order: list[str]) -> mpl.figure.Figure:
    frame = data.copy()
    frame["column"] = frame["model"].map(MODEL_LABELS).fillna(frame["model"]) + "\n" + frame["acquisition"].map(
        lambda value: ACQUISITION_LABELS.get(value, value)
    )
    models = _present_order(frame["model"], MODEL_ORDER)
    acquisitions = _present_order(frame["acquisition"], NONRANDOM_ACQUISITION_ORDER)
    columns = [f"{MODEL_LABELS.get(model, model)}\n{ACQUISITION_LABELS.get(acq, acq)}" for model in models for acq in acquisitions]
    rows = [value for value in row_order if value in set(frame[row])]
    matrix = frame.pivot_table(index=row, columns="column", values="early_gain_pp", aggfunc="mean")
    matrix = matrix.reindex(index=rows, columns=columns)
    finite = np.abs(matrix.to_numpy(dtype=float)[np.isfinite(matrix.to_numpy(dtype=float))])
    limit = float(finite.max()) if finite.size else 1.0
    if limit == 0:
        limit = 1.0
    fig, ax = plt.subplots(figsize=(8.15, 3.25 if row == "dataset" else 2.65))
    sns.heatmap(
        matrix,
        cmap="vlag",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "Early gain (MAPE pp)", "shrink": 0.78, "pad": 0.02},
        ax=ax,
    )
    if row == "dataset":
        ax.set_yticklabels([DATASET_LABELS.get(value, value) for value in matrix.index], rotation=0)
        ax.set_ylabel("Ageing dataset")
    else:
        ax.set_yticklabels([FEATURE_LABELS.get(value, value) for value in matrix.index], rotation=0)
        ax.set_ylabel("Feature representation")
    ax.set_xticklabels(
        [ACQUISITION_SHORT_LABELS.get(acquisition, acquisition) for _ in models for acquisition in acquisitions],
        rotation=40,
        ha="right",
    )
    ax.set(xlabel="Model and acquisition rule")
    for model_index, model in enumerate(models):
        center = model_index * len(acquisitions) + len(acquisitions) / 2
        ax.text(
            center,
            1.025,
            MODEL_LABELS.get(model, model),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    for boundary in range(len(acquisitions), len(columns), len(acquisitions)):
        ax.axvline(boundary, color="#333333", linewidth=0.7)
    fig.tight_layout()
    return fig


def _robustness_tradeoff(data: pd.DataFrame) -> mpl.figure.Figure:
    frame = data.copy()
    order = _section3_acquisition_order(frame["acquisition"])
    fig, ax = plt.subplots(figsize=(3.0, 4.0))
    for acquisition in order:
        row = frame.loc[frame["acquisition"].eq(acquisition)].iloc[0]
        acquisition = str(row["acquisition"])
        x = float(row["median_early_gain_pp"])
        y = float(row["positive_configuration_fraction"]) * 100
        ax.scatter(
            x,
            y,
            marker=SECTION3_MARKERS.get(acquisition, "o"),
            s=90,
            color=SECTION3_ACQUISITION_COLORS.get(acquisition, "#777777"),
            edgecolors="none",
            linewidth=0,
            alpha=0.72,
            label=ACQUISITION_LABELS.get(acquisition, acquisition),
            zorder=3,
        )
    ax.axvline(0, color="#606060", linewidth=0.55)
    ax.axhline(50, color="#606060", linewidth=0.5, linestyle="--")
    ax.set_title("Magnitude vs robustness", fontsize=SECTION3_TITLE_SIZE, fontweight="normal", pad=10)
    ax.set_xlabel("Median early-acquisitive\nperformance gain (pp)", fontsize=SECTION3_LABEL_SIZE)
    ax.set_ylabel("Positive configurations (%)", fontsize=SECTION3_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=SECTION3_TICK_SIZE)
    ax.set_ylim(0, 100)
    x_values = frame["median_early_gain_pp"].astype(float)
    x_margin = max(0.5, 0.18 * float(x_values.max() - x_values.min()))
    ax.set_xlim(float(x_values.min()) - x_margin, float(x_values.max()) + x_margin)
    ax.legend(
        title="Acquisition rule",
        loc="lower right",
        borderaxespad=0.25,
        handletextpad=0.35,
        labelspacing=0.5,
        fontsize=SECTION3_LEGEND_SIZE,
        title_fontsize=SECTION3_LEGEND_TITLE_SIZE,
    )
    ax.grid(color="#E7E7E7", linewidth=0.35)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.24, right=0.97, bottom=0.20, top=0.92)
    return _section3_fixed_canvas(fig)


def _robustness_tradeoff_with_labels(data: pd.DataFrame) -> mpl.figure.Figure:
    fig = _robustness_tradeoff(data)
    ax = fig.axes[0]
    offsets = {
        "diversity_oneshot": (5, 5),
        "diversity_iterative": (-8, -12),
        "coverage": (5, -1),
        "hybrid": (5, 5),
        "exploration": (-22, 7),
        "exploitation": (5, -2),
    }
    frame = data.copy()
    for acquisition in _section3_acquisition_order(frame["acquisition"]):
        row = frame.loc[frame["acquisition"].eq(acquisition)].iloc[0]
        x = float(row["median_early_gain_pp"])
        y = float(row["positive_configuration_fraction"]) * 100
        ax.annotate(
            ACQUISITION_LABELS.get(acquisition, acquisition),
            (x, y),
            xytext=offsets.get(acquisition, (4, 3)),
            textcoords="offset points",
            fontsize=5.8,
        )
    fig.tight_layout()
    return fig


def _nfp_feature_model_order(frame: pd.DataFrame) -> list[str]:
    preferred = [
        f"FS {feature_index}+{MODEL_LABELS.get(model, model)}"
        for feature_index, _feature in enumerate(FEATURE_ORDER, start=1)
        for model in MODEL_ORDER
    ]
    return _present_order(frame["feature_model_label"], preferred)


def _format_nfp_value(value: float) -> str:
    number = float(value)
    if not np.isfinite(number):
        return ""
    if np.isclose(number, round(number)):
        return f"{number:.0f}"
    return f"{number:.2f}"


def _nfp_dataset_title(dataset: str) -> str:
    return f"[{SECTION4_DATASET_LABELS.get(dataset, dataset)} dataset]"


def _section4_nfp_frame(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    for column in (
        "required_pool_fraction_pct",
        "required_full_life_tests",
        "required_duration",
        "nfp_percent",
    ):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "reached" in frame:
        frame["reached"] = frame["reached"].astype(str).str.lower().isin(["true", "1"])
    else:
        frame["reached"] = frame["status"].eq("crossed")
    return frame


def _nfp_heatmap_summary(frame: pd.DataFrame) -> pd.DataFrame:
    group = ["dataset", "acquisition", "acquisition_label", "feature_model_label"]
    return (
        frame.loc[frame["reached"]]
        .groupby(group, dropna=False)["required_pool_fraction_pct"]
        .median()
        .rename("median_required_pool_fraction_pct")
        .reset_index()
    )


def _nfp_interval_summary(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    group = ["dataset", "acquisition", "acquisition_label"]
    reached = frame.loc[frame["reached"]].dropna(subset=[value]).copy()
    if reached.empty:
        return pd.DataFrame(columns=group + ["median", "q25", "q75"])
    return (
        reached.groupby(group, dropna=False)[value]
        .agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        )
        .reset_index()
    )


def _nfp_relationship_summary(frame: pd.DataFrame) -> pd.DataFrame:
    group = ["dataset", "acquisition"]
    reached = frame.loc[frame["reached"]].dropna(
        subset=["required_pool_fraction_pct", "required_duration"]
    )
    return (
        reached.groupby(group, dropna=False)
        .agg(
            median_required_pool_fraction_pct=("required_pool_fraction_pct", "median"),
            median_required_duration=("required_duration", "median"),
        )
        .reset_index()
    )


def _plot_nfp_heatmap(
    ax: mpl.axes.Axes,
    summary: pd.DataFrame,
    dataset: str,
    acquisitions: list[str],
    feature_models: list[str],
    vmin: float | None,
    vmax: float | None,
    show_y: bool,
    show_x: bool,
) -> None:
    sub = summary.loc[summary["dataset"].eq(dataset)]
    if sub.empty:
        ax.axis("off")
        return
    values = (
        sub.pivot_table(
            index="acquisition",
            columns="feature_model_label",
            values="median_required_pool_fraction_pct",
            aggfunc="mean",
        )
        .reindex(index=acquisitions, columns=feature_models)
    )
    sns.heatmap(
        values,
        ax=ax,
        cmap="YlGnBu_r",
        vmin=vmin,
        vmax=vmax,
        cbar=False,
        linewidths=0.25,
        linecolor="white",
    )
    ax.set_title(
        _nfp_dataset_title(dataset),
        fontsize=SECTION4_STYLE["axes.titlesize"],
        fontweight="normal",
        pad=2,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    if show_y:
        ax.set_yticklabels([ACQUISITION_LABELS.get(value, value) for value in values.index], rotation=0)
    else:
        ax.set_yticklabels([])
    if show_x:
        ax.set_xticklabels(values.columns, rotation=90)
    else:
        ax.set_xticklabels([])


def _plot_nfp_interval(
    ax: mpl.axes.Axes,
    summary: pd.DataFrame,
    dataset: str,
    acquisitions: list[str],
    xlabel: str,
    show_y: bool,
) -> None:
    sub = summary.loc[summary["dataset"].eq(dataset)].set_index("acquisition").reindex(acquisitions)
    y = np.arange(len(acquisitions))
    for index, acquisition in enumerate(acquisitions):
        row = sub.loc[acquisition]
        if pd.isna(row["median"]):
            continue
        color = ACQUISITION_COLORS.get(acquisition, "#777777")
        ax.hlines(index, row["q25"], row["q75"], color=color, linewidth=2.0, alpha=0.55)
        ax.scatter(row["median"], index, s=15, color=color, edgecolor="#222222", linewidth=0.3, zorder=3)
        ax.text(
            row["median"],
            index - 0.18,
            _format_nfp_value(row["median"]),
            ha="center",
            va="bottom",
            fontsize=SECTION4_ANNOTATION_SIZE,
        )
    ax.set_yticks(y)
    if show_y:
        ax.set_yticklabels([ACQUISITION_LABELS.get(value, value) for value in acquisitions])
    else:
        ax.set_yticklabels([])
    ax.set_ylim(len(acquisitions) - 0.5, -0.5)
    ax.set_title(
        _nfp_dataset_title(dataset),
        fontsize=SECTION4_STYLE["axes.titlesize"],
        fontweight="normal",
        pad=8,
    )
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.5)
    ax.set_axisbelow(True)
    finite = sub[["q25", "q75", "median"]].to_numpy(dtype=float).ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size:
        ax.set_xlim(left=0, right=max(float(finite.max()) * 1.12, 1.0))


def _plot_nfp_relationship(
    ax: mpl.axes.Axes,
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    dataset: str,
    acquisitions: list[str],
    unit_label: str,
) -> None:
    columns = ["required_pool_fraction_pct", "required_duration"]
    raw = frame.loc[frame["dataset"].eq(dataset) & frame["reached"]].dropna(subset=columns)
    medians = summary.loc[summary["dataset"].eq(dataset)].set_index("acquisition")
    ax.scatter(
        raw["required_pool_fraction_pct"],
        raw["required_duration"],
        s=SECTION4_RAW_MARKER_AREA,
        color="#B8B8B8",
        alpha=0.28,
        edgecolors="none",
        zorder=1,
    )
    for acquisition in acquisitions:
        if acquisition not in medians.index:
            continue
        row = medians.loc[acquisition]
        ax.scatter(
            row["median_required_pool_fraction_pct"],
            row["median_required_duration"],
            s=SECTION4_MEDIAN_MARKER_AREA,
            color=ACQUISITION_COLORS.get(acquisition, "#777777"),
            edgecolor="#222222",
            linewidth=0.35,
            zorder=3,
        )
    ax.set_title(
        _nfp_dataset_title(dataset),
        fontsize=SECTION4_STYLE["axes.titlesize"],
        fontweight="normal",
        pad=2,
    )
    ax.set_xlabel("Required pool fraction (%)")
    ax.set_ylabel(f"Total duration ({unit_label})")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(color="#E8E8E8", linewidth=0.5)
    ax.set_axisbelow(True)


def _section4_dataset_axes(
    rows: int,
    cols: int,
    *,
    left: float,
    right: float,
    horizontal_gap: float,
    top_margin_in: float,
    bottom_margin_in: float,
    vertical_gap_in: float,
    physical_aspect_ratio: float = 1.5,
    axes_height_in: float | None = None,
) -> tuple[mpl.figure.Figure, list[mpl.axes.Axes]]:
    column_width = (right - left - horizontal_gap * (cols - 1)) / cols
    axes_width_in = column_width * SECTION4_FIGURE_WIDTH_IN
    if axes_height_in is None:
        axes_height_in = axes_width_in / physical_aspect_ratio
    figure_height_in = (
        top_margin_in
        + rows * axes_height_in
        + (rows - 1) * vertical_gap_in
        + bottom_margin_in
    )
    fig = plt.figure(
        figsize=(SECTION4_FIGURE_WIDTH_IN, figure_height_in),
        constrained_layout=False,
    )
    fig._section4_fixed_canvas = True
    axes: list[mpl.axes.Axes] = []
    for row in range(rows):
        y = 1.0 - (
            top_margin_in
            + (row + 1) * axes_height_in
            + row * vertical_gap_in
        ) / figure_height_in
        for col in range(cols):
            x = left + col * (column_width + horizontal_gap)
            axes.append(
                fig.add_axes(
                    [x, y, column_width, axes_height_in / figure_height_in]
                )
            )
    return fig, axes


def _nfp_heatmap_figure(data: pd.DataFrame, nfp_percent: int) -> mpl.figure.Figure:
    frame = _section4_nfp_frame(data)
    datasets = _present_order(frame["dataset"], DATASET_ORDER)
    acquisitions = _present_order(frame["acquisition"], ACQUISITION_ORDER)
    feature_models = _nfp_feature_model_order(frame)
    summary = _nfp_heatmap_summary(frame)
    finite = summary["median_required_pool_fraction_pct"].dropna()
    if finite.empty:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(finite.min())
        vmax = float(finite.max())
        if np.isclose(vmin, vmax):
            padding = max(abs(vmin) * 0.01, 0.5)
            vmin -= padding
            vmax += padding

    fig, axes = _section4_dataset_axes(
        2,
        4,
        left=0.12,
        right=0.91,
        horizontal_gap=0.04,
        top_margin_in=0.38,
        bottom_margin_in=1.05,
        vertical_gap_in=0.58,
        axes_height_in=1.35,
    )
    for index, ax in enumerate(axes):
        if index >= len(datasets):
            ax.axis("off")
            continue
        row, col = divmod(index, 4)
        _plot_nfp_heatmap(
            ax,
            summary,
            datasets[index],
            acquisitions,
            feature_models,
            vmin,
            vmax,
            show_y=col == 0,
            show_x=row == 1,
        )

    colorbar_bottom = axes[4].get_position().y0
    colorbar_top = axes[0].get_position().y1
    colorbar_ax = fig.add_axes(
        [0.935, colorbar_bottom, 0.012, colorbar_top - colorbar_bottom]
    )
    scalar = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
        cmap="YlGnBu_r",
    )
    colorbar = fig.colorbar(scalar, cax=colorbar_ax)
    colorbar.set_label("Median required pool fraction (%)")
    colorbar.outline.set_visible(False)
    for spine in colorbar.ax.spines.values():
        spine.set_visible(False)
    fig._section4_nfp_percent = nfp_percent
    return fig


def _nfp_interval_figure(
    data: pd.DataFrame,
    nfp_percent: int,
    value: str,
) -> mpl.figure.Figure:
    if value not in {"required_full_life_tests", "required_duration"}:
        raise KeyError(f"Unknown Section 4 interval metric: {value}")
    frame = _section4_nfp_frame(data)
    datasets = _present_order(frame["dataset"], DATASET_ORDER)
    acquisitions = _present_order(frame["acquisition"], ACQUISITION_ORDER)
    summary = _nfp_interval_summary(frame, value)
    fig, axes = _section4_dataset_axes(
        1,
        8,
        left=0.10,
        right=0.99,
        horizontal_gap=0.018,
        top_margin_in=0.38,
        bottom_margin_in=0.75,
        vertical_gap_in=0.0,
        physical_aspect_ratio=3.0 / 4.0,
    )
    for index, ax in enumerate(axes):
        if index >= len(datasets):
            ax.axis("off")
            continue
        dataset = datasets[index]
        unit = frame.loc[frame["dataset"].eq(dataset), "duration_unit"].dropna()
        unit_label = (
            unit.iloc[0]
            if not unit.empty
            else DURATION_UNITS.get(dataset, "native units")
        )
        xlabel = (
            "Full-life tests"
            if value == "required_full_life_tests"
            else f"Total duration\n({unit_label})"
        )
        _plot_nfp_interval(
            ax,
            summary,
            dataset,
            acquisitions,
            xlabel,
            show_y=index == 0,
        )
    fig._section4_nfp_percent = nfp_percent
    return fig


def _nfp_relationship_figure(
    data: pd.DataFrame,
    nfp_percent: int,
) -> mpl.figure.Figure:
    frame = _section4_nfp_frame(data)
    datasets = _present_order(frame["dataset"], DATASET_ORDER)
    acquisitions = _present_order(frame["acquisition"], ACQUISITION_ORDER)
    summary = _nfp_relationship_summary(frame)
    fig, axes = _section4_dataset_axes(
        2,
        4,
        left=0.075,
        right=0.99,
        horizontal_gap=0.055,
        top_margin_in=0.38,
        bottom_margin_in=1.02,
        vertical_gap_in=0.70,
    )
    for index, ax in enumerate(axes):
        if index >= len(datasets):
            ax.axis("off")
            continue
        dataset = datasets[index]
        unit = frame.loc[frame["dataset"].eq(dataset), "duration_unit"].dropna()
        unit_label = (
            unit.iloc[0]
            if not unit.empty
            else DURATION_UNITS.get(dataset, "native units")
        )
        _plot_nfp_relationship(
            ax,
            frame,
            summary,
            dataset,
            acquisitions,
            unit_label,
        )

    handles = [
        Line2D(
            [0],
            [0],
            linestyle="",
            marker="o",
            markerfacecolor="#B8B8B8",
            markeredgecolor="none",
            markersize=np.sqrt(SECTION4_RAW_MARKER_AREA),
            label="Independent trials",
        )
    ]
    handles.extend(
        Line2D(
            [0],
            [0],
            linestyle="",
            marker="o",
            markerfacecolor=ACQUISITION_COLORS.get(acquisition, "#777777"),
            markeredgecolor="#222222",
            markeredgewidth=0.35,
            markersize=np.sqrt(SECTION4_MEDIAN_MARKER_AREA),
            label=ACQUISITION_LABELS.get(acquisition, acquisition),
        )
        for acquisition in acquisitions
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.06),
        ncol=len(handles),
        title="Point color",
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.1,
    )
    fig._section4_nfp_percent = nfp_percent
    return fig


def _nfp_figure(
    data: pd.DataFrame,
    nfp_percent: int,
    kind: str,
) -> mpl.figure.Figure:
    if kind == "required_pool_fraction_heatmap":
        return _nfp_heatmap_figure(data, nfp_percent)
    if kind == "required_full_life_tests":
        return _nfp_interval_figure(data, nfp_percent, "required_full_life_tests")
    if kind == "total_experimental_duration":
        return _nfp_interval_figure(data, nfp_percent, "required_duration")
    if kind == "duration_vs_required_pool_fraction":
        return _nfp_relationship_figure(data, nfp_percent)
    raise KeyError(f"Unknown Section 4 NFP figure kind: {kind}")


def _render_figure(spec: FigureSpec, data: pd.DataFrame) -> mpl.figure.Figure:
    stem = spec.stem
    if stem == "fig_early_gain_by_acquisition":
        return _early_gain_effect_distribution(data)
    if stem == "fig_early_gain_dataset_model":
        return _early_dataset_model_heatmap(data)
    if stem == "fig_early_gain_feature_model":
        return _early_feature_model_heatmap(data)
    if stem == "fig_gain_robustness_tradeoff":
        return _robustness_tradeoff(data)
    if stem in SECTION4_NFP_ALPHA_BY_STEM:
        with mpl.rc_context(SECTION4_STYLE):
            return _nfp_figure(
                data,
                int(round(SECTION4_NFP_ALPHA_BY_STEM[stem] * 100)),
                SECTION4_NFP_KIND_BY_STEM[stem],
            )
    raise KeyError(f"No renderer is registered for {stem}")


def render_all(
    paths: AnalysisPaths,
    specs: list[FigureSpec] | tuple[FigureSpec, ...] = tuple(FIGURE_SPECS),
) -> list[Path]:
    set_publication_style()
    requested = list(specs)
    source_paths = []
    for spec in requested:
        section_dir = paths.output_root / SECTION_NAMES[spec.section]
        source = section_dir / "source_data" / f"{spec.stem}.csv"
        if not source.exists():
            raise FileNotFoundError(f"Missing same-stem source data for {spec.stem}: {source}")
        source_paths.append((spec, source, section_dir / "outputs" / f"{spec.stem}.png"))

    outputs_by_stem: dict[str, Path] = {}
    section1_specs = [spec for spec in requested if spec.section == 1]
    if section1_specs:
        for output in render_section1_outputs(paths, section1_specs):
            outputs_by_stem[output.stem] = output

    section2_specs = [spec for spec in requested if spec.section == 2]
    if section2_specs:
        with mpl.rc_context():
            for output in render_section2_outputs(paths, section2_specs):
                outputs_by_stem[output.stem] = output

    for spec, source, output in source_paths:
        if spec.section in {1, 2}:
            continue
        data = pd.read_csv(source)
        if data.empty:
            raise ValueError(f"Plot source data are empty for {spec.stem}")
        outputs_by_stem[spec.stem] = save_png(_render_figure(spec, data), output)
    return [outputs_by_stem[spec.stem] for spec in requested]
