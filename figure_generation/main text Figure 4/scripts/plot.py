from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
import seaborn as sns


SECTION_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SECTION_DIR.parents[1]
SCRIPT_DIR = SECTION_DIR.parent / "common"
RESULTS_DIR = SECTION_DIR / "outputs"
SOURCE_DIR = SECTION_DIR / "source_data"
ASSET_DIR = RESULTS_DIR
EARLY_GAIN_DISTRIBUTION_SOURCE = SOURCE_DIR / "early_gain_by_acquisition_config.csv"
EARLY_GAIN_TRADEOFF_SOURCE = SOURCE_DIR / "early_gain_robustness_tradeoff.csv"

sys.path.insert(0, str(SCRIPT_DIR))

from config import ACQUISITION_ORDER, DATASET_ORDER, DEFAULT_PATHS, FEATURE_ORDER, MODEL_ORDER  # noqa: E402
from io_tables import METRIC_USECOLS, normalise_metrics, read_feature_csvs  # noqa: E402
from plotting import ACQUISITION_LABELS, DATASET_LABELS, FEATURE_LABELS, MODEL_LABELS  # noqa: E402


BASELINE_ACQUISITION = "random_selection"
EARLY_MAX_FRACTION = 0.60
FIGSIZE = (7.2, 9.2)
DPI = 600
TITLE_PAD = 8
ACQUISITION_TICK_ROTATION = 35
VIOLIN_ALPHA = 0.28
SCHEMATIC_B0_FRACTION = 0.12
PANEL_CROP_PAD_INCHES = 0.04
COLORBAR_LABEL = "Relative AULC improvement (%)"
DATASET_HEATMAP_TITLE = (
    "Relative AULC improvement across\n"
    "dataset-acquisition combination"
)
FEATURE_MODEL_HEATMAP_TITLE = (
    "Relative AULC improvement across\n"
    "feature-model-acquisition combination"
)
BOTTOM_WIDTH_RATIOS = (1.0, 1.0, 0.055)
FOURTH_ROW_WIDTH_RATIOS = (1.28, 1.0)
EARLY_GAIN_ACQUISITION_ORDER = (
    "diversity_oneshot",
    "diversity_iterative",
    "coverage",
    "hybrid",
    "exploration",
    "exploitation",
)
EARLY_GAIN_MARKERS = {
    "diversity_oneshot": "o",
    "diversity_iterative": "P",
    "coverage": "s",
    "hybrid": "D",
    "exploration": "^",
    "exploitation": "v",
}
ACQUISITION_COLORS = {
    "random_selection": "#737373",
    "diversity_oneshot": "#3E9C98",
    "diversity_iterative": "#6EA7C9",
    "coverage": "#79A85B",
    "exploration": "#8B7AB3",
    "exploitation": "#D88A4B",
    "hybrid": "#C65B52",
}

PANEL_ASSET_STEMS = {
    "a": "panel_relative_alc_definition",
    "b": "panel_acquisition_distribution",
    "c": "panel_dataset_dependence",
    "d": "panel_feature_dependence",
    "e": "panel_model_dependence",
    "f": "panel_dataset_acquisition_heatmap",
    "g": "panel_feature_model_acquisition_heatmap",
    "h": "panel_improvement_robustness",
    "i": "panel_early_gain_by_acquisition",
    "j": "panel_gain_robustness_tradeoff",
}
CROPPED_ASSET_STEMS = {
    **PANEL_ASSET_STEMS,
    "colorbar": "colorbar_relative_alc_improvement",
}
AUXILIARY_ASSET_STEMS = (
    "legend_panel_acquisition_types",
    "legend_panel_robustness_thresholds",
    "legend_panel_acquisition_rules",
    "colorbar_relative_alc_improvement",
)


DATASET_DISPLAY_LABELS = {
    **DATASET_LABELS,
    "LSD_Primary": "LSD-primary",
    "LSD_Second": "LSD-second",
}

IMPROVEMENT_CMAP = LinearSegmentedColormap.from_list(
    "relative_alc_improvement",
    ["#B95F4E", "#E3B17A", "#F5F3ED", "#93C5C0", "#2F8F8A"],
)


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.5,
            "axes.labelsize": 6.5,
            "axes.titlesize": 7.2,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.35,
            "xtick.major.width": 0.35,
            "ytick.major.width": 0.35,
            "legend.frameon": False,
            "legend.fontsize": 5.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def present_order(values: pd.Series, preferred: list[str]) -> list[str]:
    observed = set(values.dropna().astype(str))
    return [value for value in preferred if value in observed] + sorted(observed.difference(preferred))


def non_random_config(config: pd.DataFrame) -> pd.DataFrame:
    return config.loc[~config["acquisition"].eq(BASELINE_ACQUISITION)].copy()


def display_dataset(value: str) -> str:
    return DATASET_DISPLAY_LABELS.get(value, value)


def acquisition_tick_label(value: str) -> str:
    label = ACQUISITION_LABELS.get(str(value), str(value))
    if label in {"Diversity (One-shot)", "Diversity (Iterative)"}:
        return label.replace(" (", "\n(", 1)
    return label


def standalone_asset_stems() -> tuple[str, ...]:
    return tuple(PANEL_ASSET_STEMS.values()) + AUXILIARY_ASSET_STEMS


def asset_output_paths(stem: str) -> dict[str, Path]:
    return {"png": ASSET_DIR / f"{stem}.png"}


def load_metrics() -> pd.DataFrame:
    raw = read_feature_csvs(DEFAULT_PATHS.al_root, "metrics.csv", usecols=METRIC_USECOLS)
    metrics = normalise_metrics(raw)
    metrics = metrics.loc[
        metrics["model"].isin(MODEL_ORDER) & metrics["acquisition"].isin(ACQUISITION_ORDER)
    ].copy()
    metrics["labelled_fraction"] = metrics["labelled_n"].div(metrics["pool_n"])
    return metrics


def load_early_profile_data() -> dict[str, pd.DataFrame]:
    return {
        "distribution": pd.read_csv(EARLY_GAIN_DISTRIBUTION_SOURCE),
        "tradeoff": pd.read_csv(EARLY_GAIN_TRADEOFF_SOURCE),
    }


def add_early_region_flags(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    b0_keys = ["dataset", "feature_set", "model", "trial"]
    out["b0_labelled_n"] = out.groupby(b0_keys, dropna=False)["labelled_n"].transform("min")
    out["early_alc_eligible"] = out["labelled_n"].gt(out["b0_labelled_n"]) & out[
        "labelled_fraction"
    ].le(EARLY_MAX_FRACTION)
    return out


def compute_absolute_alc(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["dataset", "feature_set", "model", "acquisition", "trial"]
    eligible = add_early_region_flags(metrics)
    eligible = eligible.loc[eligible["early_alc_eligible"]].copy()
    curve = (
        eligible.groupby(keys + ["labelled_n", "pool_n", "labelled_fraction"], dropna=False)
        .agg(holdout_mape=("holdout_mape", "mean"))
        .reset_index()
        .sort_values(keys + ["labelled_fraction"])
    )

    rows = []
    for group_key, group in curve.groupby(keys, dropna=False, sort=False):
        ordered = group.sort_values("labelled_fraction")
        x = ordered["labelled_fraction"].to_numpy(float)
        y = ordered["holdout_mape"].to_numpy(float)
        if len(np.unique(x)) < 2:
            continue
        area = float(np.trapezoid(y, x))
        span = float(x.max() - x.min())
        rows.append(
            {
                **dict(zip(keys, group_key)),
                "alc_area_mape_fraction": area,
                "alc_mape": area / span if span > 0 else np.nan,
                "start_fraction": float(x.min()),
                "end_fraction": float(x.max()),
                "fraction_span": span,
                "n_checkpoints": int(len(np.unique(x))),
                "early_max_fraction": EARLY_MAX_FRACTION,
                "excluded_initial_checkpoint": True,
                "initial_mape": float(y[0]),
                "final_mape": float(y[-1]),
            }
        )
    trial = pd.DataFrame(rows)
    config = (
        trial.groupby(["dataset", "feature_set", "model", "acquisition"], dropna=False)
        .agg(
            mean_alc_mape=("alc_mape", "mean"),
            median_alc_mape=("alc_mape", "median"),
            sd_trial_alc_mape=("alc_mape", "std"),
            mean_alc_area_mape_fraction=("alc_area_mape_fraction", "mean"),
            n_trials=("trial", "nunique"),
            min_checkpoints=("n_checkpoints", "min"),
            max_checkpoints=("n_checkpoints", "max"),
        )
        .reset_index()
    )
    return trial, config


def compute_relative_improvement(trial: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["dataset", "feature_set", "model", "trial"]
    baseline = trial.loc[trial["acquisition"].eq(BASELINE_ACQUISITION), keys + ["alc_mape"]].rename(
        columns={"alc_mape": "baseline_alc_mape"}
    )
    relative = trial.merge(baseline, on=keys, how="inner")
    relative = relative.loc[relative["baseline_alc_mape"].gt(0)].copy()
    relative["baseline_acquisition"] = BASELINE_ACQUISITION
    relative["delta_alc_mape"] = relative["baseline_alc_mape"] - relative["alc_mape"]
    relative["relative_improvement_pct"] = relative["delta_alc_mape"].div(relative["baseline_alc_mape"]) * 100.0
    relative["relative_direction"] = np.where(relative["relative_improvement_pct"].gt(0), "improved", "worse_or_equal")

    config = (
        relative.groupby(["dataset", "feature_set", "model", "acquisition"], dropna=False)
        .agg(
            mean_relative_improvement_pct=("relative_improvement_pct", "mean"),
            median_relative_improvement_pct=("relative_improvement_pct", "median"),
            q25_relative_improvement_pct=("relative_improvement_pct", lambda values: values.quantile(0.25)),
            q75_relative_improvement_pct=("relative_improvement_pct", lambda values: values.quantile(0.75)),
            mean_delta_alc_mape=("delta_alc_mape", "mean"),
            median_delta_alc_mape=("delta_alc_mape", "median"),
            mean_absolute_alc_mape=("alc_mape", "mean"),
            mean_baseline_alc_mape=("baseline_alc_mape", "mean"),
            n_trials=("trial", "nunique"),
        )
        .reset_index()
    )
    return relative, config


def build_summary_tables(config: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rule = (
        config.groupby("acquisition", dropna=False)
        .agg(
            mean_relative_improvement_pct=("mean_relative_improvement_pct", "mean"),
            median_relative_improvement_pct=("mean_relative_improvement_pct", "median"),
            q25_relative_improvement_pct=("mean_relative_improvement_pct", lambda values: values.quantile(0.25)),
            q75_relative_improvement_pct=("mean_relative_improvement_pct", lambda values: values.quantile(0.75)),
            n_configurations=("mean_relative_improvement_pct", "size"),
        )
        .reset_index()
    )

    robustness_base = config.copy()
    robustness_base["config_id"] = (
        robustness_base["dataset"].astype(str)
        + "|"
        + robustness_base["feature_set"].astype(str)
        + "|"
        + robustness_base["model"].astype(str)
    )
    robustness_base["positive_improvement"] = robustness_base["mean_relative_improvement_pct"].gt(0)
    robustness_base["at_least_5pct_improvement"] = robustness_base["mean_relative_improvement_pct"].ge(5)
    robustness_base["max_relative_improvement_pct"] = robustness_base.groupby("config_id", dropna=False)[
        "mean_relative_improvement_pct"
    ].transform("max")
    robustness_base["best"] = np.isclose(
        robustness_base["mean_relative_improvement_pct"],
        robustness_base["max_relative_improvement_pct"],
    )
    robustness_base["within_5pp_best"] = robustness_base["mean_relative_improvement_pct"].ge(
        robustness_base["max_relative_improvement_pct"] - 5
    )
    robustness = (
        robustness_base.groupby("acquisition", dropna=False)
        .agg(
            positive_fraction=("positive_improvement", "mean"),
            at_least_5pct_improvement_fraction=("at_least_5pct_improvement", "mean"),
            best_fraction=("best", "mean"),
            within_5pp_best_fraction=("within_5pp_best", "mean"),
            n_positive=("positive_improvement", "sum"),
            n_at_least_5pct_improvement=("at_least_5pct_improvement", "sum"),
            n_best=("best", "sum"),
            n_configurations=("config_id", "nunique"),
        )
        .reset_index()
    )
    return {"rule_summary": rule, "robustness": robustness}


def write_source_data(
    metrics: pd.DataFrame,
    absolute_trial: pd.DataFrame,
    absolute_config: pd.DataFrame,
    relative_trial: pd.DataFrame,
    relative_config: pd.DataFrame,
    summary: dict[str, pd.DataFrame],
) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    add_early_region_flags(metrics).to_csv(SOURCE_DIR / "relative_alc_learning_curve_points.csv", index=False)
    absolute_trial.to_csv(SOURCE_DIR / "relative_alc_absolute_trial_input.csv", index=False)
    absolute_config.to_csv(SOURCE_DIR / "relative_alc_absolute_config_input.csv", index=False)
    relative_trial.to_csv(SOURCE_DIR / "relative_alc_trial.csv", index=False)
    relative_config.to_csv(SOURCE_DIR / "relative_alc_config.csv", index=False)
    summary["rule_summary"].to_csv(SOURCE_DIR / "relative_alc_acquisition_summary.csv", index=False)
    summary["robustness"].to_csv(SOURCE_DIR / "relative_alc_improvement_robustness.csv", index=False)


def format_axis(ax: mpl.axes.Axes, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color="#E7E7E7", linewidth=0.35)
    ax.set_axisbelow(True)


def draw_definition(ax: mpl.axes.Axes, metrics: pd.DataFrame) -> None:
    x = np.linspace(0.0, 1.0, 401)
    random_curve = 18.0 + 70.0 * np.exp(-2.4 * x)
    early_gain = np.where(
        x > SCHEMATIC_B0_FRACTION,
        16.0
        * (1.0 - np.exp(-(x - SCHEMATIC_B0_FRACTION) / 0.06))
        * np.exp(-(x - SCHEMATIC_B0_FRACTION) / 0.45),
        0.0,
    )
    rule_curve = random_curve - early_gain
    early_mask = (x > SCHEMATIC_B0_FRACTION) & (x <= EARLY_MAX_FRACTION)

    ax.plot(x, random_curve, color="#6E6E6E", linewidth=1.2, label="Random", zorder=3)
    ax.plot(x, rule_curve, color="#2F8F8A", linewidth=1.2, label="Non-random rule", zorder=4)
    ax.fill_between(
        x,
        rule_curve,
        random_curve,
        where=early_mask,
        interpolate=True,
        color="#2F8F8A",
        alpha=0.22,
        linewidth=0,
        zorder=2,
    )
    ax.axvline(
        SCHEMATIC_B0_FRACTION,
        color="#8A8A8A",
        linewidth=0.45,
        linestyle="--",
        zorder=1,
    )
    ax.axvline(
        EARLY_MAX_FRACTION,
        color="#8A8A8A",
        linewidth=0.45,
        linestyle="--",
        zorder=1,
    )
    label_x = 0.34
    label_y = 0.5 * (
        np.interp(label_x, x, random_curve)
        + np.interp(label_x, x, rule_curve)
    )
    ax.text(label_x, label_y, r"$\Delta$AULC", color="#236F6C", ha="center", va="center", fontsize=5.8)
    ax.text(
        SCHEMATIC_B0_FRACTION,
        3.0,
        r"$b_0/\mathrm{pool}_n$",
        color="#555555",
        ha="center",
        va="bottom",
        fontsize=5.2,
    )
    ax.text(
        EARLY_MAX_FRACTION,
        3.0,
        "0.60",
        color="#555555",
        ha="center",
        va="bottom",
        fontsize=5.2,
    )
    ax.set_xlabel("Labelled fraction")
    ax.set_ylabel("MAPE (%)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)
    format_axis(ax)


def draw_acquisition_distribution(ax: mpl.axes.Axes, config: pd.DataFrame) -> None:
    frame = non_random_config(config)
    order = present_order(frame["acquisition"], ACQUISITION_ORDER)
    sns.boxplot(
        data=frame,
        x="acquisition",
        y="mean_relative_improvement_pct",
        order=order,
        hue="acquisition",
        palette=ACQUISITION_COLORS,
        legend=False,
        showfliers=False,
        linewidth=0.45,
        width=0.56,
        boxprops={"alpha": 0.45},
        ax=ax,
    )
    for patch in ax.patches:
        patch.set_alpha(0.45)
    rng = np.random.default_rng(20260820)
    for index, acquisition in enumerate(order):
        values = frame.loc[frame["acquisition"].eq(acquisition), "mean_relative_improvement_pct"].to_numpy(float)
        ax.scatter(
            np.full(len(values), index) + rng.normal(0, 0.055, len(values)),
            values,
            s=11,
            color=ACQUISITION_COLORS.get(acquisition, "#777777"),
            alpha=0.48,
            edgecolors="none",
            zorder=2,
        )
    ax.axhline(0, color="#3A3A3A", linewidth=0.5)
    ax.set_xlabel("")
    ax.set_ylabel("Improvement (%)")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(
        [acquisition_tick_label(value) for value in order],
        rotation=ACQUISITION_TICK_ROTATION,
        ha="right",
        rotation_mode="anchor",
    )
    format_axis(ax)


def draw_group_summary(
    ax: mpl.axes.Axes,
    config: pd.DataFrame,
    group_col: str,
    preferred: list[str],
    labels: dict[str, str],
    title: str,
    color: str,
) -> None:
    frame = config.loc[~config["acquisition"].eq(BASELINE_ACQUISITION)].copy()
    order = present_order(frame[group_col], preferred)
    summary = (
        frame.groupby(group_col, dropna=False)["mean_relative_improvement_pct"]
        .agg(median="median", q25=lambda values: values.quantile(0.25), q75=lambda values: values.quantile(0.75))
        .reindex(order)
        .reset_index()
    )
    sns.violinplot(
        data=frame,
        x="mean_relative_improvement_pct",
        y=group_col,
        order=order,
        orient="h",
        color=color,
        inner=None,
        cut=0,
        width=0.78,
        density_norm="width",
        common_norm=False,
        bw_adjust=0.8,
        linewidth=0,
        saturation=1,
        ax=ax,
    )
    for collection in ax.collections:
        collection.set_alpha(VIOLIN_ALPHA)
        collection.set_edgecolor("none")
        collection.set_zorder(1)
    y = np.arange(len(summary))
    ax.hlines(y, summary["q25"], summary["q75"], color="#8A8A8A", linewidth=1.0, zorder=2)
    ax.scatter(summary["median"], y, s=24, color=color, edgecolor="#303030", linewidth=0.25, zorder=3)
    ax.axvline(0, color="#3A3A3A", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([labels.get(str(value), str(value)) for value in summary[group_col]])
    ax.set_xlabel("Relative AULC improvement (%)")
    ax.set_ylabel("")
    format_axis(ax, grid_axis="x")


def heatmap_limits(*matrices: pd.DataFrame) -> tuple[float, float]:
    values = np.concatenate([matrix.to_numpy(float).ravel() for matrix in matrices])
    finite = values[np.isfinite(values)]
    low, high = np.quantile(finite, [0.03, 0.97])
    bound = max(abs(float(low)), abs(float(high)), 5.0)
    return -bound, bound


def draw_heatmap(
    ax: mpl.axes.Axes,
    matrix: pd.DataFrame,
    title: str,
    vmin: float,
    vmax: float,
    ytick_size: float = 5.8,
    annotate: bool = False,
) -> mpl.image.AxesImage:
    image = ax.imshow(
        matrix.to_numpy(float),
        cmap=IMPROVEMENT_CMAP,
        norm=TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax),
        aspect="auto",
    )
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(
        [acquisition_tick_label(value) for value in matrix.columns],
        rotation=ACQUISITION_TICK_ROTATION,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=ytick_size)
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", length=0, pad=1.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if annotate:
        threshold = 0.62 * max(abs(vmin), abs(vmax))
        for y_index, row in enumerate(matrix.to_numpy(float)):
            for x_index, value in enumerate(row):
                if np.isfinite(value):
                    display_value = 0.0 if np.isclose(value, 0.0, atol=0.005) else value
                    ax.text(
                        x_index,
                        y_index,
                        f"{display_value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=4.8,
                        color="white" if abs(value) >= threshold else "#1F1F1F",
                    )
    return image


def draw_robustness(ax: mpl.axes.Axes, robustness: pd.DataFrame) -> None:
    frame = non_random_config(robustness)
    order = present_order(frame["acquisition"], ACQUISITION_ORDER)
    frame = frame.set_index("acquisition").reindex(order).reset_index()
    y = np.arange(len(frame))
    ax.barh(
        y + 0.14,
        frame["positive_fraction"] * 100,
        height=0.24,
        color="#B7C9D8",
        edgecolor="none",
        label=">0%",
    )
    ax.barh(
        y - 0.14,
        frame["at_least_5pct_improvement_fraction"] * 100,
        height=0.24,
        color="#6E6E6E",
        edgecolor="none",
        label=">=5%",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([acquisition_tick_label(value) for value in frame["acquisition"]])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Configuration fraction (%)")
    format_axis(ax, grid_axis="x")


def draw_early_gain_distribution(ax: mpl.axes.Axes, data: pd.DataFrame) -> None:
    frame = non_random_config(data)
    order = present_order(frame["acquisition"], list(EARLY_GAIN_ACQUISITION_ORDER))
    rng = np.random.default_rng(20260820)
    for index, acquisition in enumerate(order):
        values = frame.loc[frame["acquisition"].eq(acquisition), "early_gain_pp"].dropna().to_numpy(float)
        color = ACQUISITION_COLORS.get(acquisition, "#777777")
        ax.scatter(
            np.full(len(values), index) + rng.normal(0.0, 0.055, len(values)),
            values,
            s=10,
            alpha=0.22,
            color=color,
            edgecolors="none",
            zorder=2,
        )
        ax.scatter(
            index,
            float(np.median(values)),
            marker="D",
            s=34,
            color=color,
            edgecolor="#303030",
            linewidth=0.25,
            zorder=3,
        )
    ax.axhline(0, color="#606060", linewidth=0.5, zorder=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(
        [acquisition_tick_label(value) for value in order],
        rotation=ACQUISITION_TICK_ROTATION,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_xlabel("Acquisition rule")
    ax.set_ylabel("Active-learning\nperformance gain (pp)")
    ax.tick_params(axis="x", pad=1.5)
    format_axis(ax)


def draw_gain_robustness_tradeoff(ax: mpl.axes.Axes, data: pd.DataFrame) -> None:
    frame = non_random_config(data)
    order = present_order(frame["acquisition"], list(EARLY_GAIN_ACQUISITION_ORDER))
    frame = frame.set_index("acquisition").reindex(order).reset_index()
    for row in frame.itertuples(index=False):
        acquisition = str(row.acquisition)
        ax.scatter(
            float(row.median_early_gain_pp),
            float(row.positive_configuration_fraction) * 100,
            marker=EARLY_GAIN_MARKERS.get(acquisition, "o"),
            s=48,
            color=ACQUISITION_COLORS.get(acquisition, "#777777"),
            edgecolors="none",
            linewidth=0,
            alpha=0.72,
            zorder=3,
        )
    ax.axvline(0, color="#606060", linewidth=0.5)
    ax.axhline(50, color="#606060", linewidth=0.4, linestyle="--")
    ax.set_xlabel("Median active-learning performance gain (pp)")
    ax.set_ylabel("Positive configurations (%)")
    ax.set_ylim(0, 100)
    x_values = frame["median_early_gain_pp"].astype(float)
    x_margin = max(0.5, 0.18 * float(x_values.max() - x_values.min()))
    ax.set_xlim(float(x_values.min()) - x_margin, float(x_values.max()) + x_margin)
    format_axis(ax, grid_axis="both")


def build_heatmap_matrices(config: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = non_random_config(config)
    acquisitions = present_order(frame["acquisition"], ACQUISITION_ORDER)
    datasets = present_order(frame["dataset"], DATASET_ORDER)
    dataset_matrix = (
        frame.pivot_table(
            index="dataset",
            columns="acquisition",
            values="mean_relative_improvement_pct",
            aggfunc="median",
        )
        .reindex(index=datasets, columns=acquisitions)
    )
    dataset_matrix.index = [display_dataset(value) for value in dataset_matrix.index]
    dataset_matrix.columns = [ACQUISITION_LABELS.get(value, value) for value in dataset_matrix.columns]

    features = present_order(frame["feature_set"], FEATURE_ORDER)
    models = present_order(frame["model"], MODEL_ORDER)
    frame["feature_model"] = (
        frame["feature_set"].map(FEATURE_LABELS).fillna(frame["feature_set"])
        + "+"
        + frame["model"].map(MODEL_LABELS).fillna(frame["model"])
    )
    rows = [
        f"{FEATURE_LABELS.get(feature, feature)}+{MODEL_LABELS.get(model, model)}"
        for feature in features
        for model in models
    ]
    feature_model_matrix = (
        frame.pivot_table(
            index="feature_model",
            columns="acquisition",
            values="mean_relative_improvement_pct",
            aggfunc="median",
        )
        .reindex(index=rows, columns=acquisitions)
    )
    feature_model_matrix.columns = [ACQUISITION_LABELS.get(value, value) for value in feature_model_matrix.columns]
    return dataset_matrix, feature_model_matrix


def save_figure_formats(
    fig: mpl.figure.Figure,
    base: Path,
    *,
    tight: bool = True,
) -> dict[str, Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = {"png": base.with_suffix(".png")}
    crop_options = {"bbox_inches": "tight", "pad_inches": 0.04} if tight else {}
    fig.savefig(paths["png"], dpi=DPI, **crop_options)
    return paths


def panel_crop_bboxes(
    fig: mpl.figure.Figure,
    panel_axes: dict[str, mpl.axes.Axes],
) -> dict[str, Bbox]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    display_bboxes = {}
    for key, ax in panel_axes.items():
        candidate_bboxes = [ax.get_window_extent(renderer), ax.get_tightbbox(renderer)]
        artists = [
            ax.title,
            ax.xaxis.label,
            ax.yaxis.label,
            *ax.get_xticklabels(),
            *ax.get_yticklabels(),
        ]
        candidate_bboxes.extend(
            artist.get_window_extent(renderer)
            for artist in artists
            if artist.get_visible() and artist.get_text()
        )
        artist_bboxes = [
            bbox
            for bbox in candidate_bboxes
            if bbox is not None and np.isfinite(bbox.extents).all()
        ]
        display_bboxes[key] = Bbox.union(artist_bboxes)

    if {"f", "g"}.issubset(panel_axes):
        axes_bboxes = {key: panel_axes[key].get_window_extent(renderer) for key in ("f", "g")}
        left = max(axes_bboxes[key].x0 - display_bboxes[key].x0 for key in ("f", "g"))
        right = max(display_bboxes[key].x1 - axes_bboxes[key].x1 for key in ("f", "g"))
        bottom = max(axes_bboxes[key].y0 - display_bboxes[key].y0 for key in ("f", "g"))
        top = max(display_bboxes[key].y1 - axes_bboxes[key].y1 for key in ("f", "g"))
        for key in ("f", "g"):
            axes_bbox = axes_bboxes[key]
            display_bboxes[key] = Bbox.from_extents(
                axes_bbox.x0 - left,
                axes_bbox.y0 - bottom,
                axes_bbox.x1 + right,
                axes_bbox.y1 + top,
            )

    padding = PANEL_CROP_PAD_INCHES * fig.dpi
    display_bboxes = {
        key: Bbox.from_extents(
            bbox.x0 - padding,
            bbox.y0 - padding,
            bbox.x1 + padding,
            bbox.y1 + padding,
        )
        for key, bbox in display_bboxes.items()
    }
    to_inches = fig.dpi_scale_trans.inverted()
    return {key: bbox.transformed(to_inches) for key, bbox in display_bboxes.items()}


def save_panel_crops(
    fig: mpl.figure.Figure,
    panel_axes: dict[str, mpl.axes.Axes],
    *,
    output_dir: Path = ASSET_DIR,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    crops = panel_crop_bboxes(fig, panel_axes)
    outputs = []
    for key, bbox in crops.items():
        stem = CROPPED_ASSET_STEMS[key]
        output = output_dir / f"{stem}.png"
        target_ax = panel_axes[key]
        visibility = {ax: ax.get_visible() for ax in fig.axes}
        try:
            for ax in fig.axes:
                ax.set_visible(ax is target_ax)
            fig.savefig(output, dpi=DPI, bbox_inches=bbox, pad_inches=0)
        finally:
            for ax, visible in visibility.items():
                ax.set_visible(visible)
        outputs.append(output)
    return outputs


def panel_a_legend_handles() -> tuple[list[mpl.artist.Artist], list[str]]:
    handles = [
        Line2D([0], [0], color="#6E6E6E", linewidth=1.2),
        Line2D([0], [0], color="#2F8F8A", linewidth=1.2),
        Patch(facecolor="#2F8F8A", edgecolor="none", alpha=0.22),
    ]
    return handles, ["Random", "Non-random rule", r"$\Delta$AULC"]


def robustness_legend_handles() -> tuple[list[Patch], list[str]]:
    handles = [
        Patch(facecolor="#B7C9D8", edgecolor="none"),
        Patch(facecolor="#6E6E6E", edgecolor="none"),
    ]
    return handles, [">0%", ">=5%"]


def early_gain_legend_handles() -> tuple[list[Line2D], list[str]]:
    handles = [
        Line2D(
            [0],
            [0],
            marker=EARLY_GAIN_MARKERS.get(acquisition, "o"),
            linestyle="none",
            markersize=5.2,
            markerfacecolor=ACQUISITION_COLORS.get(acquisition, "#777777"),
            markeredgecolor="none",
            alpha=0.78,
        )
        for acquisition in EARLY_GAIN_ACQUISITION_ORDER
    ]
    labels = [ACQUISITION_LABELS.get(acquisition, acquisition) for acquisition in EARLY_GAIN_ACQUISITION_ORDER]
    return handles, labels


def render_legend_asset(
    stem: str,
    handles: list[mpl.artist.Artist],
    labels: list[str],
    figsize: tuple[float, float],
    ncol: int,
) -> Path:
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    ax.legend(
        handles,
        labels,
        loc="center",
        ncol=ncol,
        handlelength=1.5,
        columnspacing=1.6,
        borderaxespad=0,
    )
    paths = save_figure_formats(fig, ASSET_DIR / stem)
    plt.close(fig)
    return paths["png"]


def build_composite_figure(
    metrics: pd.DataFrame,
    config: pd.DataFrame,
    summary: dict[str, pd.DataFrame],
    early_profile: dict[str, pd.DataFrame] | None = None,
) -> tuple[mpl.figure.Figure, dict[str, mpl.axes.Axes]]:
    set_style()
    if early_profile is None:
        early_profile = load_early_profile_data()
    dataset_matrix, feature_model_matrix = build_heatmap_matrices(config)
    vmin, vmax = heatmap_limits(dataset_matrix, feature_model_matrix)

    fig = plt.figure(figsize=FIGSIZE)
    outer = gridspec.GridSpec(
        4,
        4,
        figure=fig,
        height_ratios=[0.90, 1.28, 1.72, 1.10],
        width_ratios=[1.12, 0.90, 0.90, 1.55],
        hspace=0.78,
        wspace=0.92,
    )

    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1:4])
    ax_c = fig.add_subplot(outer[1, 0])
    ax_d = fig.add_subplot(outer[1, 1])
    ax_e = fig.add_subplot(outer[1, 2])
    ax_h = fig.add_subplot(outer[1, 3])
    bottom = outer[2, :].subgridspec(1, 3, width_ratios=BOTTOM_WIDTH_RATIOS, wspace=0.46)
    ax_f = fig.add_subplot(bottom[0, 0])
    ax_g = fig.add_subplot(bottom[0, 1])
    ax_cbar = fig.add_subplot(bottom[0, 2])
    fourth = outer[3, :].subgridspec(1, 2, width_ratios=FOURTH_ROW_WIDTH_RATIOS, wspace=0.44)
    ax_i = fig.add_subplot(fourth[0, 0])
    ax_j = fig.add_subplot(fourth[0, 1])

    draw_definition(ax_a, metrics)
    draw_acquisition_distribution(ax_b, config)
    draw_group_summary(
        ax_c,
        config,
        "dataset",
        DATASET_ORDER,
        DATASET_DISPLAY_LABELS,
        "Dataset dependence",
        "#557BA6",
    )
    draw_group_summary(
        ax_d,
        config,
        "feature_set",
        FEATURE_ORDER,
        FEATURE_LABELS,
        "Feature dependence",
        "#6E9E6B",
    )
    draw_group_summary(
        ax_e,
        config,
        "model",
        MODEL_ORDER,
        MODEL_LABELS,
        "Model dependence",
        "#C77C59",
    )
    draw_robustness(ax_h, summary["robustness"])
    draw_heatmap(
        ax_f,
        dataset_matrix,
        DATASET_HEATMAP_TITLE,
        vmin,
        vmax,
        ytick_size=5.2,
        annotate=True,
    )
    image_g = draw_heatmap(
        ax_g,
        feature_model_matrix,
        FEATURE_MODEL_HEATMAP_TITLE,
        vmin,
        vmax,
        ytick_size=4.8,
        annotate=True,
    )

    cbar = fig.colorbar(image_g, cax=ax_cbar, orientation="vertical")
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=5.6, length=2, width=0.35, pad=1)
    cbar.set_label(
        COLORBAR_LABEL,
        fontsize=6.2,
        labelpad=4,
    )
    draw_early_gain_distribution(ax_i, early_profile["distribution"])
    draw_gain_robustness_tradeoff(ax_j, early_profile["tradeoff"])

    fig.subplots_adjust(left=0.09, right=0.94, bottom=0.065, top=0.97)
    panel_axes = {
        "a": ax_a,
        "b": ax_b,
        "c": ax_c,
        "d": ax_d,
        "e": ax_e,
        "f": ax_f,
        "g": ax_g,
        "h": ax_h,
        "i": ax_i,
        "j": ax_j,
        "colorbar": ax_cbar,
    }
    return fig, panel_axes


def render_panel_assets(
    metrics: pd.DataFrame,
    config: pd.DataFrame,
    summary: dict[str, pd.DataFrame],
) -> list[Path]:
    fig, panel_axes = build_composite_figure(metrics, config, summary)
    outputs = save_panel_crops(fig, panel_axes)
    plt.close(fig)

    handles_a, labels_a = panel_a_legend_handles()
    outputs.append(
        render_legend_asset(
            "legend_panel_acquisition_types",
            handles_a,
            labels_a,
            (3.4, 0.42),
            ncol=3,
        )
    )
    handles_h, labels_h = robustness_legend_handles()
    outputs.append(
        render_legend_asset(
            "legend_panel_robustness_thresholds",
            handles_h,
            labels_h,
            (1.5, 0.42),
            ncol=2,
        )
    )
    handles_j, labels_j = early_gain_legend_handles()
    outputs.append(
        render_legend_asset(
            "legend_panel_acquisition_rules",
            handles_j,
            labels_j,
            (4.8, 0.72),
            ncol=3,
        )
    )
    return outputs


def render_figure(metrics: pd.DataFrame, config: pd.DataFrame, summary: dict[str, pd.DataFrame]) -> Path:
    fig, _ = build_composite_figure(metrics, config, summary)
    output = RESULTS_DIR / "Figure_relative_ALC"
    paths = save_figure_formats(fig, output)
    plt.close(fig)
    return paths["png"]


def main() -> int:
    metrics = pd.read_csv(SOURCE_DIR / "relative_alc_learning_curve_points.csv")
    relative_config = pd.read_csv(SOURCE_DIR / "relative_alc_config.csv")
    summary = build_summary_tables(relative_config)
    panel_outputs = render_panel_assets(metrics, relative_config, summary)
    output = render_figure(metrics, relative_config, summary)
    print(output)
    print(f"Standalone assets: {len(panel_outputs)} PNG files")
    print(ASSET_DIR)
    print(SOURCE_DIR / "relative_alc_config.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
