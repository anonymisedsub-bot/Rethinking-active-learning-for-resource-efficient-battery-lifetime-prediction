from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from PIL import Image


FEATURE_SET_SPECS = {
    1: "set1_dQn_m",
    2: "set2_early_soh",
    3: "set3_features",
    4: "set4_features_metadata",
    5: "set5_features_metadata_corr95",
}
FEATURE_SET = FEATURE_SET_SPECS[5]
REGRESSOR = "GPR"
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
METHOD_ORDER = [
    "random_selection",
    "diversity_oneshot",
    "diversity_iterative",
    "coverage",
    "exploration",
    "exploitation",
    "hybrid",
]
METHOD_LABELS = {
    "random_selection": "Random",
    "diversity_oneshot": "Diversity (One-shot)",
    "diversity_iterative": "Diversity (Iterative)",
    "coverage": "Coverage",
    "exploration": "Exploration",
    "exploitation": "Exploitation",
    "hybrid": "Hybrid",
}
METHOD_TICK_LABELS = {
    method: str(index)
    for index, method in enumerate(METHOD_ORDER, start=1)
}
METHOD_LEGEND_LABELS = {
    method: f"{index}: {METHOD_LABELS[method]}"
    for index, method in enumerate(METHOD_ORDER, start=1)
}
METHOD_COLORS = {
    "random_selection": "#606060",
    "diversity_oneshot": "#0F4D92",
    "diversity_iterative": "#3775BA",
    "coverage": "#7BAA5B",
    "exploration": "#9A4D8E",
    "exploitation": "#E28E2C",
    "hybrid": "#B64342",
}
LEGEND_HEIGHT_INCHES = 0.8
LEGEND_PANEL_GAP_INCHES = 0.03

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 11.0,
        "axes.linewidth": 0.5,
        "legend.frameon": False,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
    }
)


def validate_metrics(
    data: pd.DataFrame,
    feature_set: str = FEATURE_SET,
) -> pd.DataFrame:
    required = {
        "dataset",
        "feature_set",
        "regressor",
        "target_transform",
        "pool_holdout_split_mode",
        "method",
        "trial",
        "budget_total",
        "mape",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Metrics missing columns: {sorted(missing)}")

    frame = data.copy()
    frame["trial"] = pd.to_numeric(frame["trial"], errors="raise").astype(int)
    frame["budget_total"] = pd.to_numeric(frame["budget_total"], errors="raise").astype(int)
    frame["mape"] = pd.to_numeric(frame["mape"], errors="raise").astype(float)
    if not np.isfinite(frame["mape"]).all() or frame["mape"].lt(0).any():
        raise ValueError("MAPE values must be finite and non-negative.")
    if not frame["feature_set"].eq(feature_set).all():
        raise ValueError(f"All rows must use feature set {feature_set}.")
    if not frame["regressor"].eq(REGRESSOR).all():
        raise ValueError(f"All rows must use regressor {REGRESSOR}.")
    if not frame["target_transform"].eq("log").all():
        raise ValueError("All rows must use log target transformation.")
    if not frame["pool_holdout_split_mode"].eq("protocol_holdout").all():
        raise ValueError("All rows must use protocol hold-out splits.")

    keys = ["dataset", "method", "trial", "budget_total"]
    if frame.duplicated(keys).any():
        raise ValueError("Metrics contain duplicate dataset-method-trial-checkpoint rows.")

    if set(frame["dataset"]) != set(DATASET_ORDER):
        missing_datasets = sorted(set(DATASET_ORDER).difference(frame["dataset"]))
        extra_datasets = sorted(set(frame["dataset"]).difference(DATASET_ORDER))
        raise ValueError(
            f"Dataset contract mismatch; missing={missing_datasets}, extra={extra_datasets}."
        )

    expected_trials = set(range(1, 11))
    for dataset in DATASET_ORDER:
        panel = frame.loc[frame["dataset"].eq(dataset)]
        if set(panel["method"]) != set(METHOD_ORDER):
            raise ValueError(f"{dataset} does not contain the seven acquisition rules.")
        if set(panel["trial"]) != expected_trials:
            raise ValueError(f"{dataset} does not contain trials 1-10.")
        budgets = sorted(panel["budget_total"].unique())
        expected = pd.MultiIndex.from_product(
            [METHOD_ORDER, range(1, 11), budgets],
            names=["method", "trial", "budget_total"],
        )
        observed = pd.MultiIndex.from_frame(panel[["method", "trial", "budget_total"]])
        if len(observed) != len(expected) or not expected.difference(observed).empty:
            raise ValueError(f"{dataset} has an incomplete rule-trial-checkpoint grid.")

    dataset_rank = {name: index for index, name in enumerate(DATASET_ORDER)}
    method_rank = {name: index for index, name in enumerate(METHOD_ORDER)}
    frame["_dataset_rank"] = frame["dataset"].map(dataset_rank)
    frame["_method_rank"] = frame["method"].map(method_rank)
    return (
        frame.sort_values(
            ["_dataset_rank", "_method_rank", "trial", "budget_total"],
            ignore_index=True,
        )
        .drop(columns=["_dataset_rank", "_method_rank"])
    )


def load_metrics(
    input_root: str | Path,
    feature_set: str = FEATURE_SET,
) -> pd.DataFrame:
    input_root = Path(input_root)
    frames = []
    for dataset in DATASET_ORDER:
        path = input_root / dataset / "metrics.csv"
        if not path.exists():
            raise FileNotFoundError(f"Metrics file not found: {path}")
        frame = pd.read_csv(path)
        frame = frame.loc[
            frame["feature_set"].eq(feature_set) & frame["regressor"].eq(REGRESSOR)
        ].copy()
        frames.append(frame)
    return validate_metrics(pd.concat(frames, ignore_index=True), feature_set=feature_set)


def summarise_trial_envelopes(data: pd.DataFrame) -> pd.DataFrame:
    summary = (
        data.groupby(["dataset", "method", "budget_total"], as_index=False)
        .agg(
            mape_mean=("mape", "mean"),
            mape_std=("mape", "std"),
            n_trials=("trial", "nunique"),
        )
        .sort_values(["dataset", "method", "budget_total"], ignore_index=True)
    )
    summary["mape_sem"] = summary["mape_std"] / np.sqrt(summary["n_trials"])
    summary["mape_sem_lower"] = summary["mape_mean"] - summary["mape_sem"]
    summary["mape_sem_upper"] = summary["mape_mean"] + summary["mape_sem"]
    return summary


def robust_early_zmax(panel: pd.DataFrame) -> float:
    budgets = sorted(panel["budget_total"].unique())
    early_count = max(1, int(np.ceil(len(budgets) / 2.0)))
    early_values = panel.loc[panel["budget_total"].isin(budgets[:early_count]), "mape"]
    percentile = float(np.quantile(early_values.to_numpy(float), 0.95))
    if percentile <= 50.0:
        increment = 5.0
    elif percentile <= 200.0:
        increment = 10.0
    else:
        increment = 25.0
    return float(max(increment, np.ceil(percentile / increment) * increment))


def robust_z_limits(panel: pd.DataFrame) -> tuple[float, float]:
    """Choose readable limits while retaining the observed minimum."""
    zmax = robust_early_zmax(panel)
    minimum = float(panel["mape"].min())

    if zmax <= 50.0:
        increment = 5.0
    elif zmax <= 200.0:
        increment = 10.0
    else:
        increment = 25.0

    zmin = max(0.0, float(np.floor(minimum / increment) * increment))
    if zmin >= zmax:
        zmin = max(0.0, zmax - increment)
    return zmin, zmax


def sparse_checkpoint_indices(n_checkpoints: int, max_ticks: int = 6) -> np.ndarray:
    if n_checkpoints <= 0:
        return np.asarray([], dtype=int)
    if n_checkpoints <= max_ticks:
        return np.arange(n_checkpoints, dtype=int)
    return np.unique(
        np.rint(np.linspace(0, n_checkpoints - 1, max_ticks)).astype(int)
    )


def _envelope_vertices(
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> list[tuple[float, float]]:
    return list(zip(x, lower, strict=True)) + list(
        zip(x[::-1], upper[::-1], strict=True)
    )


def _style_3d_axis(ax: plt.Axes) -> None:
    ax.view_init(elev=23, azim=-57)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.grid(True)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_linewidth(0.5)
        axis.line.set_color("#8A8A8A")
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor("#C8C8C8")
        axis._axinfo["grid"].update(color="#D8D8D8", linewidth=0.3, linestyle="-")


def _plot_dataset_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    summary: pd.DataFrame,
    dataset: str,
    panel_letter: str,
) -> None:
    raw_panel = data.loc[data["dataset"].eq(dataset)]
    panel = summary.loc[summary["dataset"].eq(dataset)]
    budgets = np.asarray(sorted(raw_panel["budget_total"].unique()), dtype=float)
    zmin, zmax = robust_z_limits(raw_panel)

    for method_index, method in enumerate(METHOD_ORDER):
        curve = panel.loc[panel["method"].eq(method)].sort_values("budget_total")
        x = np.arange(len(curve), dtype=float)
        lower = np.clip(curve["mape_sem_lower"].to_numpy(float), zmin, zmax)
        upper_raw = curve["mape_sem_upper"].to_numpy(float)
        upper = np.clip(upper_raw, zmin, zmax)
        mean_raw = curve["mape_mean"].to_numpy(float)
        mean_visible = np.clip(mean_raw, zmin, zmax)
        color = METHOD_COLORS[method]

        polygon = PolyCollection(
            [_envelope_vertices(x, lower, upper)],
            facecolors=[color],
            edgecolors=[color],
            linewidths=0.25,
            alpha=0.10,
            closed=True,
            zorder=1,
        )
        ax.add_collection3d(polygon, zs=method_index, zdir="y")
        ax.plot(
            x,
            np.full(len(x), method_index, dtype=float),
            mean_visible,
            color=color,
            linewidth=0.9,
            alpha=0.78,
            marker="o",
            markersize=2.4,
            markeredgewidth=0,
            zorder=4,
        )

    _style_3d_axis(ax)
    checkpoint_positions = np.arange(len(budgets), dtype=float)
    ax.set_xlim(-0.2, float(len(budgets) - 0.8))
    tick_indices = sparse_checkpoint_indices(len(budgets), max_ticks=6)
    ax.set_xticks(checkpoint_positions[tick_indices])
    ax.set_xticklabels(
        [f"{int(budgets[index])}" for index in tick_indices],
        fontsize=8.7,
        rotation=24,
    )
    ax.tick_params(axis="x", pad=-1.5)
    ax.set_ylim(-0.45, len(METHOD_ORDER) - 0.45)
    ax.set_yticks(np.arange(len(METHOD_ORDER)))
    ax.set_yticklabels(
        [METHOD_TICK_LABELS[method] for method in METHOD_ORDER],
        fontsize=9.0,
    )
    ax.tick_params(axis="y", pad=0)
    ax.set_zlim(zmin, zmax)
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.tick_params(axis="z", labelsize=9.2, pad=0)
    ax.set_xlabel("Labelled-set size", fontsize=10.1, labelpad=-2.0)
    ax.set_ylabel("Acquisition\nrule", fontsize=10.1, labelpad=1.5)
    ax.yaxis.label.set_linespacing(0.9)
    ax.yaxis.label.set_clip_on(False)
    ax.set_zlabel("MAPE (%)", fontsize=10.1, labelpad=4.0)
    ax.set_title(
        rf"$\mathbf{{{panel_letter}}}$  {DATASET_LABELS[dataset]}",
        fontsize=11.6,
        fontweight="normal",
        pad=2.5,
    )


def _rule_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            linewidth=1.2,
            alpha=0.78,
            label=METHOD_LEGEND_LABELS[method],
        )
        for method in METHOD_ORDER
    ]


def _prepare_png_output(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".png":
        raise ValueError("The figure output path must use the .png extension.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".svg", ".pdf", ".tiff"):
        output_path.with_suffix(suffix).unlink(missing_ok=True)
    return output_path


def _shrink_axes_positions(axes: list[plt.Axes], scale: float = 0.72) -> None:
    """Shrink every 3D panel around its centre without changing typography."""
    for index, axis in enumerate(axes):
        position = axis.get_position()
        width = position.width * scale
        height = position.height * scale
        column_shift = -0.035 if index % 2 else 0.0
        axis.set_position(
            [
                position.x0 + (position.width - width) / 2.0 + column_shift,
                position.y0 + (position.height - height) / 2.0,
                width,
                height,
            ]
        )


def render_figure(
    data: pd.DataFrame,
    output_path: str | Path,
    dpi: int = 600,
    feature_set: str = FEATURE_SET,
):
    validated = validate_metrics(data, feature_set=feature_set)
    summary = summarise_trial_envelopes(validated)
    fig = plt.figure(figsize=(8.6, 10.6))
    axes = [fig.add_subplot(4, 2, index + 1, projection="3d") for index in range(8)]
    for index, (axis, dataset) in enumerate(zip(axes, DATASET_ORDER, strict=True)):
        _plot_dataset_panel(
            axis,
            validated,
            summary,
            dataset,
            chr(ord("a") + index),
        )

    fig.subplots_adjust(
        left=0.02,
        right=0.92,
        top=0.97,
        bottom=0.03,
        wspace=-0.50,
        hspace=0.08,
    )
    _shrink_axes_positions(axes, scale=0.72)

    output_path = _prepare_png_output(output_path)
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.35,
        facecolor="white",
    )
    return fig


def render_legend(
    output_path: str | Path,
    target_width_px: int,
    dpi: int = 600,
):
    if target_width_px <= 0:
        raise ValueError("Legend target width must be a positive pixel count.")

    fig = plt.figure(
        figsize=(target_width_px / dpi, LEGEND_HEIGHT_INCHES),
        facecolor="white",
    )
    handles = _rule_legend_handles()
    fig.legend(
        handles=handles[:3],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        fontsize=10.1,
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=0.55,
    )
    fig.legend(
        handles=handles[3:],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=4,
        fontsize=10.1,
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=0.55,
    )

    output_path = _prepare_png_output(output_path)
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    return fig


def compose_figure_and_legend(
    panels_path: str | Path,
    legend_path: str | Path,
    output_path: str | Path,
    gap_px: int,
    dpi: int = 600,
) -> Path:
    if gap_px < 0:
        raise ValueError("Legend gap must be non-negative.")

    with Image.open(panels_path) as source:
        panels = source.convert("RGBA")
    with Image.open(legend_path) as source:
        legend = source.convert("RGBA")
    if panels.width != legend.width:
        raise ValueError("Panel and legend images must have the same width.")

    combined = Image.new(
        "RGBA",
        (panels.width, panels.height + gap_px + legend.height),
        (255, 255, 255, 255),
    )
    combined.paste(panels, (0, 0))
    combined.paste(legend, (0, panels.height + gap_px))
    output_path = _prepare_png_output(output_path)
    combined.save(output_path, dpi=(dpi, dpi))
    return output_path


def render_feature_set_artifacts(
    data: pd.DataFrame,
    feature_set_index: int,
    feature_set: str,
    source_dir: str | Path,
    results_dir: str | Path,
    dpi: int = 600,
    write_source_data: bool = True,
) -> tuple[Path, Path]:
    validated = validate_metrics(data, feature_set=feature_set)
    source_dir = Path(source_dir)
    results_dir = Path(results_dir)
    if write_source_data:
        source_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    source_columns = [
        "dataset",
        "feature_set",
        "regressor",
        "target_transform",
        "pool_holdout_split_mode",
        "method",
        "trial",
        "budget_total",
        "mape",
    ]
    source_path = source_dir / f"gpr_set{feature_set_index}_trial_mape.csv"
    if write_source_data:
        validated[source_columns].to_csv(source_path, index=False)

    stem = f"gpr_set{feature_set_index}_3d_mape_profiles"
    output_path = results_dir / f"Figure_{stem}.png"
    panels_path = results_dir / f"_Figure_{stem}_panels.png"
    legend_path = results_dir / f"_Figure_{stem}_legend.png"
    figure = None
    legend_figure = None
    try:
        figure = render_figure(
            validated,
            panels_path,
            dpi=dpi,
            feature_set=feature_set,
        )
        plt.close(figure)
        figure = None
        with Image.open(panels_path) as panels_image:
            panel_width_px = panels_image.width
        legend_figure = render_legend(
            legend_path,
            target_width_px=panel_width_px,
            dpi=dpi,
        )
        plt.close(legend_figure)
        legend_figure = None
        compose_figure_and_legend(
            panels_path=panels_path,
            legend_path=legend_path,
            output_path=output_path,
            gap_px=int(round(dpi * LEGEND_PANEL_GAP_INCHES)),
            dpi=dpi,
        )
    finally:
        if figure is not None:
            plt.close(figure)
        if legend_figure is not None:
            plt.close(legend_figure)
        panels_path.unlink(missing_ok=True)
        legend_path.unlink(missing_ok=True)

    (results_dir / f"Legend_{stem}.png").unlink(missing_ok=True)
    return source_path, output_path


def main() -> None:
    section_root = Path(__file__).resolve().parents[1]
    source_dir = section_root / "source_data"
    results_dir = section_root / "outputs"

    for feature_set_index, feature_set in FEATURE_SET_SPECS.items():
        data = pd.read_csv(source_dir / f"gpr_set{feature_set_index}_trial_mape.csv")
        source_path, output_path = render_feature_set_artifacts(
            data=data,
            feature_set_index=feature_set_index,
            feature_set=feature_set,
            source_dir=source_dir,
            results_dir=results_dir,
            write_source_data=False,
        )
        print(f"Set {feature_set_index} source data: {source_path}")
        print(f"Set {feature_set_index} figure: {output_path}")


if __name__ == "__main__":
    main()
