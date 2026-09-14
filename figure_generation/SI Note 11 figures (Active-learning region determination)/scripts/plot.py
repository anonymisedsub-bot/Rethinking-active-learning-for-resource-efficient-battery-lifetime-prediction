from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


DATASET_ORDER = [
    "MIT",
    "ISU_ILCC",
    "LSD_Primary",
    "LSD_Second",
    "HUST",
    "KIT",
    "Formation",
    "TRI_Tesla",
]
FEATURE_ORDER = [
    "set1_dQn_m",
    "set2_early_soh",
    "set3_features",
    "set4_features_metadata",
    "set5_features_metadata_corr95",
]
MODEL_ORDER = ["GPR", "RF", "AE_ENet"]
DATASET_LABELS = {
    "MIT": "[MIT dataset]",
    "ISU_ILCC": "[ISU-ILCC dataset]",
    "LSD_Primary": "[LSD-primary dataset]",
    "LSD_Second": "[LSD-second dataset]",
    "HUST": "[HUST dataset]",
    "KIT": "[KIT dataset]",
    "Formation": "[Formation dataset]",
    "TRI_Tesla": "[TRI-Tesla dataset]",
}
FEATURE_LABELS = {
    "set1_dQn_m": "FS 1",
    "set2_early_soh": "FS 2",
    "set3_features": "FS 3",
    "set4_features_metadata": "FS 4",
    "set5_features_metadata_corr95": "FS 5",
}
RANDOM_METHOD = "random_selection"
NONRANDOM_METHOD_ORDER = [
    "diversity_oneshot",
    "diversity_iterative",
    "coverage",
    "exploration",
    "exploitation",
    "hybrid",
]
METHOD_ORDER = [RANDOM_METHOD, *NONRANDOM_METHOD_ORDER]
METHOD_LABELS = {
    "diversity_oneshot": "Diversity (One-shot)",
    "diversity_iterative": "Diversity (Iterative)",
    "coverage": "Coverage",
    "exploration": "Exploration",
    "exploitation": "Exploitation",
    "hybrid": "Hybrid",
}
METHOD_COLORS = {
    "diversity_oneshot": "#0F4D92",
    "diversity_iterative": "#4F88C6",
    "coverage": "#76A85A",
    "exploration": "#8A68A8",
    "exploitation": "#D78734",
    "hybrid": "#B64B4A",
}
METHOD_LINESTYLES = {
    "diversity_oneshot": "-",
    "diversity_iterative": "--",
    "coverage": (0, (4.0, 1.5)),
    "exploration": "-.",
    "exploitation": ":",
    "hybrid": (0, (5.0, 1.4)),
}
METHOD_MARKERS = {
    "diversity_oneshot": "o",
    "diversity_iterative": "D",
    "coverage": "s",
    "exploration": "^",
    "exploitation": "v",
    "hybrid": "P",
}
FRACTION_TICKS = np.asarray([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
MATCH_KEYS = [
    "dataset",
    "feature_set",
    "regressor",
    "trial",
    "labeled_train_count",
]
REQUIRED_COLUMNS = {
    *MATCH_KEYS,
    "method",
    "max_budget_mean",
    "mape",
    "budget_total",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 9.0,
        "axes.linewidth": 0.7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    }
)


def validate_metrics(data: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Metrics missing columns: {sorted(missing)}")

    frame = data.loc[:, sorted(REQUIRED_COLUMNS)].copy()
    for column in ("trial", "labeled_train_count", "max_budget_mean", "budget_total"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").round().astype(int)
    frame["mape"] = pd.to_numeric(frame["mape"], errors="raise").astype(float)

    if not np.isfinite(frame["mape"]).all() or frame["mape"].lt(0).any():
        raise ValueError("MAPE values must be finite and non-negative.")
    if frame["max_budget_mean"].le(0).any():
        raise ValueError("Pool sizes must be positive.")
    if frame["labeled_train_count"].le(0).any():
        raise ValueError("Labelled-set sizes must be positive.")
    if frame["labeled_train_count"].gt(frame["max_budget_mean"]).any():
        raise ValueError("Labelled-set sizes cannot exceed pool sizes.")

    unknown_methods = sorted(set(frame["method"]).difference(METHOD_ORDER))
    if unknown_methods:
        raise ValueError(f"Unknown acquisition methods: {unknown_methods}")

    metric_keys = [*MATCH_KEYS, "method"]
    if frame.duplicated(metric_keys).any():
        raise ValueError("Metrics contain duplicate matched acquisition checkpoints.")
    return frame


def validate_complete_contract(data: pd.DataFrame) -> pd.DataFrame:
    frame = validate_metrics(data)
    expected_values = {
        "dataset": set(DATASET_ORDER),
        "feature_set": set(FEATURE_ORDER),
        "regressor": set(MODEL_ORDER),
        "method": set(METHOD_ORDER),
        "trial": set(range(1, 11)),
    }
    for column, expected in expected_values.items():
        observed = set(frame[column])
        if observed != expected:
            raise ValueError(
                f"Metrics contract is incomplete for {column}; "
                f"missing={sorted(expected.difference(observed))}, "
                f"extra={sorted(observed.difference(expected))}."
            )

    config_keys = ["dataset", "feature_set", "regressor"]
    for key, panel in frame.groupby(config_keys, sort=False):
        if panel["max_budget_mean"].nunique() != 1:
            raise ValueError(f"Metrics contract is incomplete: inconsistent pool size for {key}.")
        pool_n = int(panel["max_budget_mean"].iat[0])
        labelled_values = sorted(int(value) for value in panel["labeled_train_count"].unique())
        if not labelled_values or labelled_values[-1] != pool_n:
            raise ValueError(f"Metrics contract is incomplete: full-pool checkpoint missing for {key}.")
        expected = pd.MultiIndex.from_product(
            [range(1, 11), METHOD_ORDER, labelled_values],
            names=["trial", "method", "labeled_train_count"],
        )
        observed = pd.MultiIndex.from_frame(
            panel[["trial", "method", "labeled_train_count"]]
        )
        if len(observed) != len(expected) or not expected.difference(observed).empty:
            raise ValueError(f"Metrics contract is incomplete for configuration {key}.")
    return frame


def compute_trial_error_reduction(data: pd.DataFrame) -> pd.DataFrame:
    frame = validate_metrics(data)
    random = frame.loc[
        frame["method"].eq(RANDOM_METHOD),
        [*MATCH_KEYS, "mape"],
    ].rename(columns={"mape": "random_mape"})
    paired = frame.loc[frame["method"].isin(NONRANDOM_METHOD_ORDER)].merge(
        random,
        on=MATCH_KEYS,
        how="left",
        validate="many_to_one",
    )
    if paired["random_mape"].isna().any():
        missing = paired.loc[paired["random_mape"].isna(), MATCH_KEYS].head()
        raise ValueError(
            "Matched Random MAPE is missing for checkpoints: "
            f"{missing.to_dict('records')}"
        )
    paired["labelled_pool_fraction_pct"] = (
        100.0 * paired["labeled_train_count"] / paired["max_budget_mean"]
    )
    paired["error_reduction_pp"] = paired["random_mape"] - paired["mape"]
    return paired.sort_values(
        ["dataset", "feature_set", "regressor", "method", "trial", "labeled_train_count"],
        ignore_index=True,
    )


def summarise_error_reduction(trial_level: pd.DataFrame) -> pd.DataFrame:
    group_keys = [
        "dataset",
        "feature_set",
        "regressor",
        "method",
        "labelled_pool_fraction_pct",
        "labeled_train_count",
        "max_budget_mean",
    ]
    required = {*group_keys, "trial", "error_reduction_pp"}
    missing = required.difference(trial_level.columns)
    if missing:
        raise ValueError(f"Trial-level reductions missing columns: {sorted(missing)}")
    if trial_level.duplicated([*group_keys, "trial"]).any():
        raise ValueError("Trial-level reductions contain duplicate trial checkpoints.")

    summary = (
        trial_level.groupby(group_keys, as_index=False, sort=False)["error_reduction_pp"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "error_reduction_mean_pp",
                "std": "error_reduction_sd_pp",
                "count": "n_trials",
            }
        )
    )
    summary["error_reduction_sd_pp"] = summary["error_reduction_sd_pp"].fillna(0.0)
    summary["error_reduction_sem_pp"] = (
        summary["error_reduction_sd_pp"] / np.sqrt(summary["n_trials"])
    )
    return summary.sort_values(
        ["regressor", "dataset", "feature_set", "method", "labelled_pool_fraction_pct"],
        ignore_index=True,
    )


def _set_panel_y_limits(ax: plt.Axes, panel: pd.DataFrame) -> None:
    lower = panel["error_reduction_mean_pp"] - panel["error_reduction_sem_pp"]
    upper = panel["error_reduction_mean_pp"] + panel["error_reduction_sem_pp"]
    low = min(0.0, float(lower.min()))
    high = max(0.0, float(upper.max()))
    span = high - low
    if span <= 1e-12:
        ax.set_ylim(-1.0, 1.0)
        return
    padding = max(0.5, 0.12 * span)
    ax.set_ylim(low - padding, high + padding)


def _legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            marker=METHOD_MARKERS[method],
            linewidth=1.15,
            markersize=4.2,
            markeredgewidth=0.0,
            label=METHOD_LABELS[method],
        )
        for method in NONRANDOM_METHOD_ORDER
    ]


def render_model_figure(summary: pd.DataFrame, model: str):
    if model not in MODEL_ORDER:
        raise ValueError(f"Unknown model: {model}")
    model_summary = summary.loc[summary["regressor"].eq(model)].copy()
    if model_summary.empty:
        raise ValueError(f"No summary rows found for model {model}.")

    fig, axes = plt.subplots(
        nrows=len(DATASET_ORDER),
        ncols=len(FEATURE_ORDER),
        figsize=(12.6, 15.8),
        sharex=False,
        sharey=False,
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.94,
        top=0.955,
        bottom=0.075,
        wspace=0.28,
        hspace=0.42,
    )
    for row_index, dataset in enumerate(DATASET_ORDER):
        for column_index, feature_set in enumerate(FEATURE_ORDER):
            ax = axes[row_index, column_index]
            panel = model_summary.loc[
                model_summary["dataset"].eq(dataset)
                & model_summary["feature_set"].eq(feature_set)
            ].copy()
            if panel.empty:
                plt.close(fig)
                raise ValueError(
                    f"Summary is missing panel data for {model}, {dataset}, {feature_set}."
                )

            ax.axhline(0.0, color="#777777", linewidth=0.7, zorder=0)
            for method in NONRANDOM_METHOD_ORDER:
                curve = panel.loc[panel["method"].eq(method)].sort_values(
                    "labelled_pool_fraction_pct"
                )
                if curve.empty:
                    plt.close(fig)
                    raise ValueError(
                        f"Summary is missing {method} for {model}, {dataset}, {feature_set}."
                    )
                x = curve["labelled_pool_fraction_pct"].to_numpy(float)
                y = curve["error_reduction_mean_pp"].to_numpy(float)
                sem = curve["error_reduction_sem_pp"].to_numpy(float)
                color = METHOD_COLORS[method]
                ax.plot(
                    x,
                    y,
                    color=color,
                    linestyle=METHOD_LINESTYLES[method],
                    marker=METHOD_MARKERS[method],
                    markersize=2.8,
                    markeredgewidth=0.0,
                    linewidth=1.0,
                    label=METHOD_LABELS[method],
                    zorder=3,
                )
                ax.fill_between(
                    x,
                    y - sem,
                    y + sem,
                    color=color,
                    alpha=0.10,
                    linewidth=0.0,
                    zorder=1,
                )

            _set_panel_y_limits(ax, panel)
            ax.set_xlim(0.0, 100.0)
            ax.set_xticks(FRACTION_TICKS)
            ax.grid(axis="y", color="#E2E2E2", linewidth=0.45)
            ax.tick_params(labelsize=8.4, pad=1.5)

            if row_index == 0:
                ax.set_title(FEATURE_LABELS[feature_set], fontsize=9.6, pad=5.0)
            if column_index == 0:
                ax.set_ylabel("Error reduction (pp)", fontsize=9.0)
                ax.text(
                    -0.22,
                    1.05,
                    chr(ord("a") + row_index),
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=12.5,
                    fontweight="bold",
                )
            if column_index == len(FEATURE_ORDER) - 1:
                ax.text(
                    1.13,
                    0.5,
                    DATASET_LABELS[dataset],
                    transform=ax.transAxes,
                    rotation=270,
                    ha="center",
                    va="center",
                    fontsize=9.6,
                    fontweight="bold",
                )
            if row_index == len(DATASET_ORDER) - 1:
                ax.set_xlabel("Labelled pool fraction (%)", fontsize=9.0)

    handles = _legend_handles()
    fig.legend(
        handles=handles,
        labels=[METHOD_LABELS[method] for method in NONRANDOM_METHOD_ORDER],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=6,
        fontsize=9.2,
        handlelength=2.4,
        columnspacing=1.2,
        markerscale=1.25,
    )
    return fig


def save_model_figure(
    figure,
    output_stem: str | Path,
    dpi: int = 600,
) -> dict[str, Path]:
    if dpi <= 0:
        raise ValueError("DPI must be positive.")
    stem = Path(output_stem)
    if stem.suffix:
        raise ValueError("Figure output stem must not include a file extension.")
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = {"png": stem.with_suffix(".png")}
    figure.savefig(
        paths["png"],
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="white",
    )
    return paths


def load_metrics(input_root: str | Path) -> pd.DataFrame:
    root = Path(input_root)
    paths = sorted(root.glob("feature set */*/metrics.csv"), key=lambda path: str(path))
    if not paths:
        raise FileNotFoundError(f"No metrics.csv files found below {root}")
    columns = [
        "dataset",
        "feature_set",
        "regressor",
        "trial",
        "method",
        "labeled_train_count",
        "max_budget_mean",
        "mape",
        "budget_total",
    ]
    frames = [pd.read_csv(path, usecols=columns, low_memory=False) for path in paths]
    return pd.concat(frames, ignore_index=True, sort=False)


def render_all_from_frame(
    metrics: pd.DataFrame,
    section_root: str | Path,
    dpi: int = 600,
) -> dict[str, dict[str, Path]]:
    validated = validate_complete_contract(metrics)
    trial_level = compute_trial_error_reduction(validated)
    summary = summarise_error_reduction(trial_level)
    if not summary["n_trials"].eq(10).all():
        bad = summary.loc[~summary["n_trials"].eq(10)].head().to_dict("records")
        raise ValueError(f"Summary checkpoints do not contain ten trials: {bad}")

    root = Path(section_root)
    source_dir = root / "source_data"
    results_dir = root / "outputs"
    source_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    for model in MODEL_ORDER:
        for suffix in ("svg", "pdf"):
            stale_path = (
                results_dir
                / f"Figure_error_reduction_vs_random_{model}.{suffix}"
            )
            if stale_path.exists():
                stale_path.unlink()
    source_path = source_dir / "error_reduction_vs_random_learning_curves.csv"
    summary.to_csv(source_path, index=False)

    outputs: dict[str, dict[str, Path]] = {}
    for model in MODEL_ORDER:
        figure = render_model_figure(summary, model)
        try:
            outputs[model] = save_model_figure(
                figure,
                results_dir / f"Figure_error_reduction_vs_random_{model}",
                dpi=dpi,
            )
        finally:
            plt.close(figure)
    return outputs


def main() -> None:
    section_root = Path(__file__).resolve().parents[1]
    source_path = section_root / "source_data" / "error_reduction_vs_random_learning_curves.csv"
    summary = pd.read_csv(source_path)
    outputs = {}
    for model in MODEL_ORDER:
        figure = render_model_figure(summary, model)
        try:
            outputs[model] = save_model_figure(
                figure,
                section_root / "outputs" / f"Figure_error_reduction_vs_random_{model}",
                dpi=600,
            )
        finally:
            plt.close(figure)
    print(f"Source data: {source_path}")
    for model, bundle in outputs.items():
        for suffix, path in bundle.items():
            print(f"{model} {suffix.upper()}: {path}")


if __name__ == "__main__":
    main()
