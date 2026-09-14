from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

from utils import ensure_dir, set_plot_style

BENCHMARK_MODE = "total_budget"

MODE_LABELS = {
    BENCHMARK_MODE: "Active-learning benchmark",
}

METHOD_LABELS = {
    "random_selection": "Random",
    "diversity_oneshot": "Diversity_oneshot",
    "diversity_iterative": "Diversity_iterative",
    "coverage": "Coverage",
    "exploration": "Exploration",
    "exploitation": "Exploitation",
    "hybrid": "Hybrid",
}

METHOD_ORDER = (
    "random_selection",
    "diversity_oneshot",
    "diversity_iterative",
    "coverage",
    "exploration",
    "exploitation",
    "hybrid",
)
REGRESSOR_ORDER = ("GPR", "RF", "AE_ENet")
REGRESSOR_LABELS = {
    "GPR": "GPR",
    "RF": "RF",
    "AE_ENet": "AE-ENet",
}
ITERATIVE_METHODS = {"random_selection", "diversity_iterative", "coverage", "exploration", "exploitation", "hybrid"}
DEFAULT_ITERATIVE_BATCH_SIZE = 8
ALL_POOL_ACCURACY_FRACTION = 0.95
ALL_POOL_TARGET_LABEL = "95% all-pool accuracy"
METRIC_PANEL_FONTSIZE = 11
METRIC_LABEL_FONTSIZE = 12
TARGET_MARKER_SIZE_2D = 46
TARGET_MARKER_SIZE_3D = 42
TARGET_ANNOTATION_FONTSIZE = 10
MAX_SAFE_PNG_PATH_LENGTH = 240
HIGH_CONTRAST_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
    "#F0E442",
    "#8B4513",
)
MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">")

PUBLICATION_COLORS = {
    "mutual_information": "#4F7C93",
    "exploitation": "#55B6A6",
    "similarity": "#F2C36B",
    "exploration": "#C98253",
    "no_retraining": "#7FB9D8",
    "random": "#A9A9A9",
}

METHOD_COLOR_KEYS = {
    "random_selection": "random",
    "coverage": "similarity",
    "diversity_oneshot": "no_retraining",
    "diversity_iterative": "no_retraining",
    "exploration": "exploration",
    "exploitation": "exploitation",
    "hybrid": "mutual_information",
}

METHOD_MARKERS = {
    "random_selection": "o",
    "coverage": "s",
    "diversity_oneshot": "D",
    "diversity_iterative": "v",
    "exploration": "p",
    "exploitation": "X",
    "hybrid": "h",
}

METHOD_LINESTYLES = {
    "random_selection": "-",
    "coverage": "--",
    "diversity_oneshot": "-.",
    "diversity_iterative": (0, (2.0, 1.2)),
    "exploration": ":",
    "exploitation": (0, (5.0, 1.5)),
    "hybrid": (0, (3.0, 1.2, 1.0, 1.2)),
}

METHOD_SHORT_LABELS = {
    "random_selection": "Random",
    "diversity_oneshot": "Diversity_oneshot",
    "diversity_iterative": "Diversity_iterative",
    "exploration": "Exploration",
    "coverage": "Coverage",
    "exploitation": "Exploitation",
    "hybrid": "Hybrid",
}

NATURE_COLORS = ["#313695", "#74add1", "#e0f3f8", "#fee090", "#f46d43", "#a50026"]
NATURE_CMAP = LinearSegmentedColormap.from_list("nature_cmap", NATURE_COLORS)

BUBBLE_COLORS = ["#F1EEF6", "#D4B9DA", "#C994C7", "#DF65B0", "#DD1C77", "#980043"]
BUBBLE_CMAP = LinearSegmentedColormap.from_list("bubble_cmap", BUBBLE_COLORS)

# A4 纸张学术排版物理尺寸控制常数 (英寸 = cm / 2.54)
CM2INCH = 1 / 2.54
W_MULTI = 16.0 * CM2INCH   # 多列布局总宽 16cm
W_SINGLE = 7.0 * CM2INCH   # 单列布局总宽 7cm


def _filter_iterative_batch_size(
    df: pd.DataFrame,
    iterative_batch_size: int | None = DEFAULT_ITERATIVE_BATCH_SIZE,
) -> pd.DataFrame:
    if df.empty or "method" not in df.columns or "iterative_batch_size" not in df.columns:
        return df.copy()
    if iterative_batch_size is None:
        return df.copy()
    methods = df["method"].astype(str)
    batch_sizes = pd.to_numeric(df["iterative_batch_size"], errors="coerce")
    keep = ~methods.isin(ITERATIVE_METHODS) | np.isclose(batch_sizes.astype(float), float(iterative_batch_size))
    return df[keep].copy()


def apply_global_style() -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 7,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 6,
        'axes.labelpad': 15.0,
        'axes.titlepad': 6.0,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
        'xtick.major.size': 3.0,
        'ytick.major.size': 3.0,
        'xtick.minor.size': 2.0,
        'ytick.minor.size': 2.0,
        'lines.linewidth': 1.0,
        'lines.markersize': 4.0,
        'legend.frameon': False,
    })


def _fallback_png_path(out_path: Path) -> Path:
    parent = out_path.parent
    safe_stem = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in out_path.stem)
    parent_len = len(str(parent.resolve()))
    for attempt in range(1, 100):
        suffix = f"_retry{attempt}.png"
        max_stem_len = max(12, 240 - parent_len - len(suffix) - 1)
        stem = (safe_stem or "figure")[:max_stem_len]
        candidate = parent / f"{stem}{suffix}"
        if not candidate.exists():
            return candidate
    return parent / "figure_retry.png"


def _path_text_length(path: Path) -> int:
    try:
        return len(str(path.resolve()))
    except OSError:
        return len(str(path.absolute()))


def _should_use_fallback_png_path(out_path: Path) -> bool:
    return _path_text_length(out_path) >= MAX_SAFE_PNG_PATH_LENGTH


def _save_png(fig, out_path: Path, use_tight_layout: bool = True) -> None:
    ensure_dir(out_path.parent)
    save_kwargs = {"dpi": 300}
    if use_tight_layout:
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig.tight_layout(pad=0.5)
        except Exception:
            pass
        save_kwargs.update({"bbox_inches": "tight", "pad_inches": 0.05})
    target_path = _fallback_png_path(out_path) if _should_use_fallback_png_path(out_path) else out_path
    try:
        fig.savefig(target_path, **save_kwargs)
    except OSError as exc:
        fallback = _fallback_png_path(out_path)
        if fallback == target_path:
            fallback = _fallback_png_path(fallback)
        try:
            fig.savefig(fallback, **save_kwargs)
            print(f"[Warn] Could not save figure to {target_path} ({exc}); saved fallback: {fallback}")
        except OSError as fallback_exc:
            print(f"[Warn] Could not save figure to {target_path} or fallback {fallback}: {fallback_exc}")


def _ordered_methods(values: pd.Series) -> list[str]:
    present = set(values.dropna().astype(str))
    ordered = [m for m in METHOD_ORDER if m in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def _display_method(method: str) -> str:
    return METHOD_LABELS.get(str(method), str(method))


def _display_method_short(method: str) -> str:
    return METHOD_SHORT_LABELS.get(str(method), _display_method(method))


def _method_plot_color(method: str) -> str:
    key = METHOD_COLOR_KEYS.get(str(method))
    return PUBLICATION_COLORS.get(key, HIGH_CONTRAST_COLORS[0])


def _method_marker(method: str) -> str:
    return METHOD_MARKERS.get(str(method), "o")


def _method_line_style(method: str):
    return METHOD_LINESTYLES.get(str(method), "-")


def _budget_denominator(df: pd.DataFrame, budget_col: str, fraction: float | None = None) -> int:
    values = df[budget_col].dropna()
    if values.empty:
        return 1
    return max(int(values.max()), 1)


def _axis_budget_labels(budgets: list[int], indices: list[int] = None) -> list[str]:
    if not budgets:
        return []
    if indices is None:
        return [str(b) for b in budgets]
    return [str(budgets[i]) for i in indices]


def _axis_percent_labels(budgets: list[int], denominator: int, show_percent: bool = True, indices: list[int] = None) -> list[str]:
    if not budgets:
        return []
    suffix = "%" if show_percent else ""
    idx_list = indices if indices is not None else range(len(budgets))
    return [f"{100.0 * budgets[i] / max(denominator, 1):.0f}{suffix}" for i in idx_list]


def _add_budget_ratio_axis(ax, budgets: list[int], denominator: int, label: str, show_percent: bool = True, indices: list[int] = None, show_label: bool = True) -> None:
    sec = ax.secondary_xaxis("top", functions=(lambda x: x, lambda x: x))
    idx_list = indices if indices is not None else np.arange(len(budgets))
    sec.set_xticks(idx_list)
    sec.set_xticklabels(_axis_percent_labels(budgets, denominator, show_percent, indices), fontsize=7)
    if show_label:
        sec.set_xlabel(label, fontsize=8, labelpad=5)
    else:
        sec.set_xlabel("")
    sec.tick_params(axis="x", length=0, pad=2)


def _line_summary(df: pd.DataFrame, group_cols: list[str], budget_col: str, metric_col: str = "mape") -> pd.DataFrame:
    cols = [c for c in group_cols if c in df.columns]
    by_trial = df.groupby(cols + [budget_col, "trial"], dropna=False, as_index=False)[metric_col].mean()
    return by_trial.groupby(cols + [budget_col], dropna=False, as_index=False).agg(
        **{
            f"{metric_col}_mean": (metric_col, "mean"),
            f"{metric_col}_std": (metric_col, "std"),
            f"{metric_col}_min": (metric_col, "min"),
            f"{metric_col}_max": (metric_col, "max"),
        },
        n_trials=("trial", "nunique"),
    )


def _match_group_mask(df: pd.DataFrame, group_cols: list[str], row: pd.Series) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col in group_cols:
        if col not in df.columns:
            continue
        value = row[col]
        if pd.isna(value):
            mask &= df[col].isna()
        elif np.issubdtype(df[col].dtype, np.number):
            mask &= np.isclose(pd.to_numeric(df[col], errors="coerce").astype(float), float(value))
        else:
            mask &= df[col].astype(str) == str(value)
    return mask


def _fill_iterative_cold_start_points(
    method: str,
    method_summary: pd.DataFrame,
    random_summary: pd.DataFrame,
    group_cols: list[str],
    budget_col: str,
) -> pd.DataFrame:
    if method not in ITERATIVE_METHODS or method_summary.empty or random_summary.empty:
        return method_summary
    additions = []
    unique_groups = random_summary[group_cols].drop_duplicates() if group_cols else pd.DataFrame([{}])
    for _, group in unique_groups.iterrows():
        method_group = method_summary[_match_group_mask(method_summary, group_cols, group)]
        random_group = random_summary[_match_group_mask(random_summary, group_cols, group)]
        if method_group.empty or random_group.empty:
            continue
        first_budget = float(method_group[budget_col].min())
        cold = random_group[pd.to_numeric(random_group[budget_col], errors="coerce").astype(float) < first_budget].copy()
        if cold.empty:
            continue
        if "method" in method_summary.columns or "method" in cold.columns:
            cold["method"] = method
        additions.append(cold)
    if not additions:
        return method_summary
    return pd.concat(additions + [method_summary], ignore_index=True).sort_values(group_cols + [budget_col])


def _half_tick_indices(values: list[int] | list[float]) -> np.ndarray:
    n = len(values)
    if n <= 5:
        return np.arange(n)
    max_ticks = min(4, n)
    numeric_values = np.asarray(values, dtype=float)
    if np.all(numeric_values > 0):
        log_values = np.log2(numeric_values)
        targets = np.linspace(float(log_values[0]), float(log_values[-1]), max_ticks)
        indices = [int(np.argmin(np.abs(log_values - target))) for target in targets]
        indices[0] = 0
        indices[-1] = n - 1
        return np.asarray(sorted(set(indices)), dtype=int)
    return np.unique(np.round(np.linspace(0, n - 1, max_ticks)).astype(int))


def _log2_budget_positions(budgets: list[int] | list[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(budgets, dtype=float)
    return np.log2(np.maximum(values, 1e-12))


def _axis_log_budget_labels(budgets: list[int] | list[float], indices: np.ndarray | None = None) -> list[str]:
    idx_list = indices if indices is not None else np.arange(len(budgets))
    labels = []
    for idx in idx_list:
        value = float(np.log2(float(budgets[int(idx)])))
        labels.append(f"{value:.0f}" if np.isclose(value, round(value)) else f"{value:.2f}")
    return labels


def _power_of_two_budget_ticks(budgets: list[int] | list[float]) -> list[int]:
    values = [float(v) for v in budgets if pd.notna(v) and float(v) > 0]
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    start = int(np.ceil(np.log2(lo)))
    end = int(np.floor(np.log2(hi)))
    powers = [int(2**exp) for exp in range(start, end + 1)]
    if len(powers) <= 4:
        return powers
    min_gap = 0.10 * max(hi - lo, 1.0)
    ticks: list[int] = []
    for power in powers:
        if not ticks or float(power - ticks[-1]) >= min_gap or power == powers[-1]:
            ticks.append(power)
    return ticks


def _axis_budget_value_labels(budgets: list[int] | list[float]) -> list[str]:
    labels = []
    for value in budgets:
        value = float(value)
        labels.append(f"{int(round(value))}" if np.isclose(value, round(value)) else f"{value:g}")
    return labels


def _axis_percent_value_labels(budgets: list[int] | list[float], denominator: int, show_percent: bool = True) -> list[str]:
    suffix = "%" if show_percent else ""
    return [f"{100.0 * float(budget) / max(denominator, 1):.0f}{suffix}" for budget in budgets]


def _budget_ticks_with_max_pool(display_budgets: list[int], global_budgets: list[int]) -> list[int]:
    if not global_budgets:
        return list(display_budgets)
    ticks = list(display_budgets)
    max_budget = int(max(global_budgets))
    if max_budget not in ticks:
        ticks.append(max_budget)
    return sorted(ticks)


def _true_runs(mask: np.ndarray) -> list[slice]:
    values = np.asarray(mask, dtype=bool)
    runs: list[slice] = []
    start: int | None = None
    for idx, keep in enumerate(values):
        if keep and start is None:
            start = idx
        elif not keep and start is not None:
            if idx - start >= 2:
                runs.append(slice(start, idx))
            start = None
    if start is not None and len(values) - start >= 2:
            runs.append(slice(start, len(values)))
    return runs


def _clip_polygon_by_z(
    vertices: list[tuple[float, float]],
    z_min: float,
    z_max: float,
) -> list[tuple[float, float]]:
    def clip_against(
        points: list[tuple[float, float]],
        boundary: float,
        keep_above: bool,
    ) -> list[tuple[float, float]]:
        if not points:
            return []
        clipped: list[tuple[float, float]] = []

        def inside(point: tuple[float, float]) -> bool:
            return point[1] >= boundary if keep_above else point[1] <= boundary

        prev = points[-1]
        prev_inside = inside(prev)
        for cur in points:
            cur_inside = inside(cur)
            if cur_inside != prev_inside:
                x0, z0 = prev
                x1, z1 = cur
                if np.isclose(z0, z1):
                    x_boundary = x1
                else:
                    frac = (boundary - z0) / (z1 - z0)
                    x_boundary = x0 + float(np.clip(frac, 0.0, 1.0)) * (x1 - x0)
                clipped.append((float(x_boundary), float(boundary)))
            if cur_inside:
                clipped.append(cur)
            prev = cur
            prev_inside = cur_inside
        return clipped

    return clip_against(clip_against(vertices, z_max, keep_above=False), z_min, keep_above=True)


def _format_budget_annotation(value: float) -> str:
    if np.isclose(value, round(value), atol=0.05):
        return f"{int(round(value))}"
    return f"{value:.1f}"


def _target_crossing(
    line: pd.DataFrame,
    budget_col: str,
    metric_prefix: str,
    target_value: float | None,
) -> tuple[float, float, float] | None:
    if target_value is None or not np.isfinite(target_value):
        return None
    mean_col = f"{metric_prefix}_mean"
    if budget_col not in line.columns or mean_col not in line.columns:
        return None

    data = line[[budget_col, mean_col]].copy()
    data[budget_col] = pd.to_numeric(data[budget_col], errors="coerce")
    data[mean_col] = pd.to_numeric(data[mean_col], errors="coerce")
    data = data.dropna().sort_values(budget_col)
    data = data[data[budget_col].astype(float) > 0]
    if data.empty:
        return None

    budgets = data[budget_col].to_numpy(float)
    values = data[mean_col].to_numpy(float)
    if metric_prefix == "accuracy":
        reached = values >= float(target_value)
    else:
        reached = values <= float(target_value)
    if not np.any(reached):
        return None

    idx = int(np.argmax(reached))
    if idx == 0:
        x_cross = float(budgets[0])
    else:
        x0 = float(budgets[idx - 1])
        x1 = float(budgets[idx])
        y0 = float(values[idx - 1])
        y1 = float(values[idx])
        if np.isclose(y0, y1):
            x_cross = x1
        else:
            frac = (float(target_value) - y0) / (y1 - y0)
            x_cross = x0 + float(np.clip(frac, 0.0, 1.0)) * (x1 - x0)
    budget_cross = float(x_cross)
    return x_cross, float(target_value), budget_cross


def _add_target_marker_2d(ax, crossing: tuple[float, float, float] | None, color: str) -> None:
    if crossing is None:
        return
    x_cross, y_cross, budget_cross = crossing
    ax.plot(
        x_cross,
        y_cross,
        marker="v",
        markersize=7.0,
        color=color,
        markeredgewidth=0,
        linestyle="None",
        label="_nolegend_",
        zorder=7,
    )
    ax.annotate(
        _format_budget_annotation(budget_cross),
        xy=(x_cross, y_cross),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=TARGET_ANNOTATION_FONTSIZE,
        color=color,
        zorder=8,
    )


def _positive_budget_list(values: pd.Series) -> list[int]:
    return [
        int(budget)
        for budget in sorted(pd.to_numeric(values, errors="coerce").dropna().astype(int).unique().tolist())
        if int(budget) > 0
    ]


def _apply_thin_spines(ax, lw: float = 0.6) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(lw)


def _set_3d_tick_params(ax) -> None:
    ax.tick_params(axis='x', pad=0, width=0.6, length=2.5)
    ax.tick_params(axis='y', pad=0, width=0.6, length=2.5)
    ax.tick_params(axis='z', pad=0, width=0.6, length=2.5)


def _set_surface_axis_pads(ax, labelpad: float = -2.0, tick_pad: float = -3.0) -> None:
    ax.xaxis.labelpad = labelpad
    ax.yaxis.labelpad = labelpad
    ax.zaxis.labelpad = labelpad
    ax.tick_params(axis="x", pad=tick_pad, width=0.6, length=2.5)
    ax.tick_params(axis="y", pad=tick_pad, width=0.6, length=2.5)
    ax.tick_params(axis="z", pad=tick_pad, width=0.6, length=2.5)


def _top_axis_label_for_mode(experiment_mode: str) -> str:
    return "Full-life test ratio (%)"


def _bottom_axis_label_for_mode(experiment_mode: str) -> str:
    return "Full-life test budget"


def _get_interp_x(val: float, budgets_list: list[int]) -> float:
    if not budgets_list:
        return 0.0
    if val <= budgets_list[0]: 
        return 0.0
    if val >= budgets_list[-1]: 
        return float(len(budgets_list) - 1)
    for idx in range(len(budgets_list) - 1):
        if budgets_list[idx] <= val <= budgets_list[idx+1]:
            frac = (val - budgets_list[idx]) / (budgets_list[idx+1] - budgets_list[idx])
            return idx + frac
    return 0.0


def _plot_metric_vs_budget(
    metrics: pd.DataFrame,
    out_dir: Path,
    dataset: str,
    experiment_mode: str,
    metric_col: str,
    metric_label: str,
    target_metric_col: str,
    output_metric_name: str,
    iterative_batch_size: int = DEFAULT_ITERATIVE_BATCH_SIZE,
) -> None:
    if metrics.empty:
        return
    import matplotlib.pyplot as plt

    set_plot_style()
    apply_global_style()
    sub = _filter_iterative_batch_size(
        metrics[(metrics["dataset"] == dataset) & (metrics["experiment_mode"] == experiment_mode)],
        iterative_batch_size,
    )
    if sub.empty:
        return
    if metric_col not in sub.columns:
        return

    budget_col = "budget_total"
    methods = _ordered_methods(sub["method"])
    if not methods or budget_col not in sub.columns:
        return

    method_colors = {m: _method_plot_color(m) for m in methods}

    target_value = _target_mape(sub) if metric_col == "mape" else None

    global_budgets = _positive_budget_list(sub[budget_col])
    if not global_budgets:
        return
    display_budgets = _power_of_two_budget_ticks(global_budgets)
    x_lookup = {budget: float(budget) for budget in global_budgets}
    display_positions = np.asarray(display_budgets, dtype=float)

    panel_values: list[float | str | None] = methods[:6]
    n_cols = 3
    n_rows = 2
    fig_w = W_MULTI * 1.18
    fig_h = W_MULTI * 0.98
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    y_min, y_max = float("inf"), float("-inf")
    for panel_value in panel_values:
        panel_df = sub[sub["method"].astype(str) == str(panel_value)]
        panel_methods = [str(panel_value)]
        random_summary = _line_summary(sub[sub["method"] == "random_selection"], [], budget_col, metric_col)
        summary = _line_summary(panel_df, ["method"], budget_col, metric_col)
        for method in panel_methods:
            line = summary[summary["method"] == method].copy()
            if method in ITERATIVE_METHODS:
                line = _fill_iterative_cold_start_points(method, line, random_summary, [], budget_col)
            line = line[pd.to_numeric(line[budget_col], errors="coerce").astype(float) > 0]
            if line.empty:
                continue
            y_min = min(y_min, line[f"{metric_col}_min"].min())
            y_max = max(y_max, line[f"{metric_col}_max"].max())
    if target_value is not None:
        y_min = min(y_min, target_value)
        y_max = max(y_max, target_value)

    global_ylim = None
    if np.isfinite(y_min) and np.isfinite(y_max):
        y_pad = (y_max - y_min) * 0.12 if y_max > y_min else 1.0
        global_ylim = (y_min - y_pad, y_max + y_pad)

    for idx, (ax, panel_value) in enumerate(zip(axes.ravel(), panel_values)):
        panel_df = sub[sub["method"].astype(str) == str(panel_value)]
        panel_methods = [str(panel_value)]
        random_summary = _line_summary(sub[sub["method"] == "random_selection"], [], budget_col, metric_col)
        summary = _line_summary(panel_df, ["method"], budget_col, metric_col)

        for method in panel_methods:
            line = summary[summary["method"] == method].copy()
            if method in ITERATIVE_METHODS:
                line = _fill_iterative_cold_start_points(method, line, random_summary, [], budget_col)
            line = line.sort_values(budget_col)
            line = line[pd.to_numeric(line[budget_col], errors="coerce").astype(float) > 0]
            if line.empty:
                continue
            budgets = line[budget_col].astype(int).tolist()
            x = np.asarray([x_lookup[int(b)] for b in budgets], dtype=float)
            y_mean = line[f"{metric_col}_mean"].to_numpy(float)
            y_min_band = line[f"{metric_col}_min"].to_numpy(float)
            y_max_band = line[f"{metric_col}_max"].to_numpy(float)
            color = method_colors[method]
            ax.fill_between(x, y_min_band, y_max_band, color=color, alpha=0.14, linewidth=0)
            ax.plot(
                x,
                y_mean,
                color=color,
                lw=1.25,
                linestyle="-",
                label="_nolegend_",
            )
            _add_target_marker_2d(
                ax,
                _target_crossing(line, budget_col, metric_col, target_value),
                color,
            )

        if target_value is not None:
            ax.axhline(target_value, color="gray", lw=0.8, ls="--", label=ALL_POOL_TARGET_LABEL)

        if global_ylim:
            ax.set_ylim(global_ylim)

        ax.set_title("")
        ax.set_title(_display_method_short(str(panel_value)), fontsize=12, color="#1F2430", pad=5)

        show_y_ticks = (idx % n_cols == 0)
        if not show_y_ticks:
            ax.set_yticklabels([])

        show_x_ticks = (idx >= n_cols * (n_rows - 1))
        ax.set_xticks(display_positions)
        if show_x_ticks:
            ax.set_xticklabels(
                _axis_budget_value_labels(display_budgets),
                fontsize=METRIC_PANEL_FONTSIZE,
                rotation=28,
                ha="right",
            )
        else:
            ax.set_xticklabels([])
        ax.tick_params(axis="y", labelsize=METRIC_PANEL_FONTSIZE)

        denominator = _budget_denominator(sub, budget_col)
        show_top_ticks = idx < n_cols
        sec = ax.secondary_xaxis("top", functions=(lambda x: x, lambda x: x))
        sec.set_xticks(display_positions)
        if show_top_ticks:
            sec.set_xticklabels(
                _axis_percent_value_labels(display_budgets, denominator, show_percent=False),
                fontsize=METRIC_PANEL_FONTSIZE,
                rotation=28,
                ha="center",
            )
        else:
            sec.set_xticklabels([])
        sec.set_xlabel("")
        sec.tick_params(axis="x", length=0, pad=2, width=0.6)

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xlim(float(min(global_budgets)), float(max(global_budgets)))
        ax.grid(True, axis="y", alpha=0.28)
        _apply_thin_spines(ax)

    for ax in axes.ravel()[len(panel_values):]:
        ax.axis("off")

    handles, labels = [], []
    for ax in axes.ravel():
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(
            handles, labels, loc="upper center",
            bbox_to_anchor=(0.5, 0.985), ncol=min(len(handles), 4),
            frameon=False,
            fontsize=METRIC_PANEL_FONTSIZE,
        )

    fig.supxlabel(_bottom_axis_label_for_mode(experiment_mode), y=0.045, fontsize=METRIC_LABEL_FONTSIZE)
    fig.supylabel(metric_label, x=0.018, fontsize=METRIC_LABEL_FONTSIZE)
    # Use only one top-axis label for the whole figure.
    top_label_y = 0.912
    fig.text(0.5, top_label_y, _top_axis_label_for_mode(experiment_mode), ha="center", va="top", fontsize=METRIC_LABEL_FONTSIZE)

    fig.suptitle("")
    top = 0.80
    fig.subplots_adjust(top=top, bottom=0.20, left=0.105, right=0.965, hspace=0.38, wspace=0.28)
    _save_png(fig, out_dir / f"{output_metric_name}_vs_budget.png", use_tight_layout=False)
    plt.close(fig)


def plot_mape_vs_budget(
    metrics: pd.DataFrame,
    out_dir: Path,
    dataset: str,
    experiment_mode: str,
    iterative_batch_size: int = DEFAULT_ITERATIVE_BATCH_SIZE,
) -> None:
    _plot_metric_vs_budget(
        metrics,
        out_dir,
        dataset,
        experiment_mode,
        "mape",
        "MAPE (%)",
        "all_pool_mape",
        "mape",
        iterative_batch_size,
    )


def plot_rmspe_vs_budget(
    metrics: pd.DataFrame,
    out_dir: Path,
    dataset: str,
    experiment_mode: str,
    iterative_batch_size: int = DEFAULT_ITERATIVE_BATCH_SIZE,
) -> None:
    _plot_metric_vs_budget(
        metrics,
        out_dir,
        dataset,
        experiment_mode,
        "rmspe",
        "RMSPE (%)",
        "all_pool_rmspe",
        "rmspe",
        iterative_batch_size,
    )


def plot_benchmark_methods(
    metrics: pd.DataFrame,
    out_dir: Path,
    dataset: str,
    iterative_batch_size: int = DEFAULT_ITERATIVE_BATCH_SIZE,
) -> None:
    sub = _filter_iterative_batch_size(
        metrics[(metrics["dataset"] == dataset) & (metrics["experiment_mode"] == BENCHMARK_MODE)],
        iterative_batch_size,
    )
    if sub.empty:
        return
    import matplotlib.pyplot as plt

    set_plot_style()
    apply_global_style()
    methods = _ordered_methods(sub["method"])[:6]
    if not methods:
        return
    global_budgets = _positive_budget_list(sub["budget_total"])
    if not global_budgets:
        return
    display_budgets = _power_of_two_budget_ticks(global_budgets)
    x_lookup = {budget: idx for idx, budget in enumerate(global_budgets)}

    target_mape = _target_mape(sub)

    lines_by_method = _benchmark_metric_lines_by_method(sub, methods, "mape")

    _plot_benchmark_methods_3d(
        out_dir,
        dataset,
        methods,
        lines_by_method,
        global_budgets,
        display_budgets,
        target_mape,
        metric_prefix="mape",
        z_label="MAPE (%)",
        output_name="all_methods_mape_vs_budget.png",
        target_label=ALL_POOL_TARGET_LABEL,
        target_style="plane",
    )
    _plot_benchmark_methods_3d(
        out_dir,
        dataset,
        methods,
        lines_by_method,
        global_budgets,
        display_budgets,
        target_mape,
        metric_prefix="mape",
        z_label="MAPE (%)",
        output_name="all_methods_mape_vs_budget_with_inset.png",
        target_label=ALL_POOL_TARGET_LABEL,
        target_style="plane",
        add_mape_inset=True,
    )
    _plot_benchmark_methods_2d(
        out_dir,
        dataset,
        methods,
        lines_by_method,
        global_budgets,
        display_budgets,
        target_mape,
        metric_prefix="mape",
        y_label="MAPE (%)",
        output_name="all_methods_mape_vs_budget_2D.png",
        target_label=ALL_POOL_TARGET_LABEL,
    )

    if "rmspe" in sub.columns:
        target_rmspe = None
        _plot_benchmark_methods_3d(
            out_dir,
            dataset,
            methods,
            _benchmark_metric_lines_by_method(sub, methods, "rmspe"),
            global_budgets,
            display_budgets,
            target_rmspe,
            metric_prefix="rmspe",
            z_label="RMSPE (%)",
            output_name="all_methods_rmspe_vs_budget.png",
            target_label=ALL_POOL_TARGET_LABEL,
            target_style="plane",
        )
        _plot_benchmark_methods_2d(
            out_dir,
            dataset,
            methods,
            _benchmark_metric_lines_by_method(sub, methods, "rmspe"),
            global_budgets,
            display_budgets,
            target_rmspe,
            metric_prefix="rmspe",
            y_label="RMSPE (%)",
            output_name="all_methods_rmspe_vs_budget_2D.png",
            target_label=ALL_POOL_TARGET_LABEL,
        )

    target_accuracy = None
    if "all_pool_mape" in sub.columns:
        baseline_mape = sub["all_pool_mape"].dropna().mean()
        if pd.notna(baseline_mape):
            target_accuracy = 0.95 * (100.0 - float(baseline_mape))
    _plot_benchmark_methods_3d(
        out_dir,
        dataset,
        methods,
        _accuracy_lines_by_method(lines_by_method),
        global_budgets,
        display_budgets,
        target_accuracy,
        metric_prefix="accuracy",
        z_label="Accuracy (%)",
        output_name="all_methods_accuracy_vs_budget.png",
        target_label="95% all-pool accuracy",
        target_style="dashed",
    )


def _benchmark_metric_lines_by_method(sub: pd.DataFrame, methods: list[str], metric_col: str) -> dict[str, pd.DataFrame]:
    summary = _line_summary(sub, ["method"], "budget_total", metric_col)
    random_summary = _line_summary(sub[sub["method"] == "random_selection"], [], "budget_total", metric_col)
    lines_by_method: dict[str, pd.DataFrame] = {}
    for method in methods:
        line = summary[summary["method"] == method].copy()
        if method in ITERATIVE_METHODS:
            line = _fill_iterative_cold_start_points(method, line, random_summary, [], "budget_total")
        lines_by_method[method] = line.sort_values("budget_total")
    return lines_by_method


def _accuracy_lines_by_method(lines_by_method: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for method, line in lines_by_method.items():
        converted = line.copy()
        if converted.empty:
            out[method] = converted
            continue
        converted["accuracy_mean"] = 100.0 - converted["mape_mean"].astype(float)
        converted["accuracy_min"] = 100.0 - converted["mape_max"].astype(float)
        converted["accuracy_max"] = 100.0 - converted["mape_min"].astype(float)
        out[method] = converted
    return out


def _plot_benchmark_methods_2d(
    out_dir: Path,
    dataset: str,
    methods: list[str],
    lines_by_method: dict[str, pd.DataFrame],
    global_budgets: list[int],
    display_budgets: list[int],
    target_value: float | None,
    metric_prefix: str,
    y_label: str,
    output_name: str,
    target_label: str | None,
) -> None:
    import matplotlib.pyplot as plt

    if not methods or not global_budgets:
        return

    fig, ax = plt.subplots(figsize=(W_MULTI, W_MULTI))
    y_min, y_max = float("inf"), float("-inf")
    for method in methods:
        line = lines_by_method.get(method, pd.DataFrame())
        if line.empty:
            continue
        line = line[pd.to_numeric(line["budget_total"], errors="coerce").astype(float) > 0]
        if line.empty:
            continue
        budgets = line["budget_total"].astype(int).to_numpy(float)
        x = budgets
        y_mean = line[f"{metric_prefix}_mean"].to_numpy(float)
        y_low = line[f"{metric_prefix}_min"].to_numpy(float)
        y_high = line[f"{metric_prefix}_max"].to_numpy(float)
        color = _method_plot_color(method)
        y_min = min(y_min, float(np.nanmin(y_low)))
        y_max = max(y_max, float(np.nanmax(y_high)))
        ax.fill_between(x, y_low, y_high, color=color, alpha=0.14, linewidth=0)
        ax.plot(
            x,
            y_mean,
            lw=1.35,
            color=color,
            label=_display_method_short(method),
        )

    if target_value is not None and np.isfinite(target_value):
        ax.axhline(target_value, color="gray", lw=1.0, ls="--", label=target_label or "_nolegend_")
        y_min = min(y_min, float(target_value))
        y_max = max(y_max, float(target_value))

    if np.isfinite(y_min) and np.isfinite(y_max):
        y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.set_xticks(np.asarray(display_budgets, dtype=float))
    ax.set_xticklabels(
        _axis_budget_value_labels(display_budgets),
        fontsize=METRIC_PANEL_FONTSIZE,
        rotation=28,
        ha="right",
    )
    ax.set_xlim(float(min(global_budgets)), float(max(global_budgets)))
    ax.set_xlabel("Full-life test budget", fontsize=METRIC_LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=METRIC_LABEL_FONTSIZE)
    ax.tick_params(axis="y", labelsize=METRIC_PANEL_FONTSIZE)
    ax.grid(True, axis="y", alpha=0.28)
    _apply_thin_spines(ax)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            fontsize=METRIC_PANEL_FONTSIZE,
            ncol=min(len(handles), 4),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            frameon=False,
        )
    fig.subplots_adjust(top=0.84, bottom=0.18, left=0.13, right=0.98)
    _save_png(fig, out_dir / output_name, use_tight_layout=False)
    plt.close(fig)


def _add_mape_inset_axis(
    fig,
    methods: list[str],
    lines_by_method: dict[str, pd.DataFrame],
    global_budgets: list[int],
    display_budgets: list[int],
    target_value: float | None,
    target_label: str | None,
) -> None:
    import matplotlib.pyplot as plt

    inset_ax = fig.add_axes([0.68, 0.58, 0.24, 0.22])
    y_values: list[float] = []
    for method in methods:
        line = lines_by_method.get(method, pd.DataFrame())
        if line.empty:
            continue
        line = line[pd.to_numeric(line["budget_total"], errors="coerce").astype(float) > 0]
        if line.empty:
            continue
        x = line["budget_total"].astype(int).to_numpy(float)
        y = line["mape_mean"].to_numpy(float)
        y_values.extend([float(v) for v in y if np.isfinite(v)])
        inset_ax.plot(
            x,
            y,
            lw=1.0,
            color=_method_plot_color(method),
            linestyle=_method_line_style(method),
            label=_display_method_short(method),
        )

    if target_value is not None and np.isfinite(target_value):
        inset_ax.axhline(target_value, color="0.42", lw=0.85, ls="--", label=target_label or "_nolegend_")
        y_values.append(float(target_value))

    if global_budgets:
        inset_ax.set_xlim(float(min(global_budgets)), float(max(global_budgets)))
    x_tick_budgets = _budget_ticks_with_max_pool(display_budgets, global_budgets)
    inset_ax.set_xticks(np.asarray(x_tick_budgets, dtype=float))
    inset_ax.set_xticklabels(_axis_budget_value_labels(x_tick_budgets), fontsize=6, rotation=28, ha="right")

    if y_values:
        y_min, y_max = min(y_values), max(y_values)
        y_pad = (y_max - y_min) * 0.08 if y_max > y_min else 1.0
        inset_ax.set_ylim(y_min - y_pad, y_max + y_pad)
    inset_ax.tick_params(axis="y", labelsize=6, pad=1.0)
    inset_ax.set_title("Mean MAPE", fontsize=6.5, pad=2.0)
    inset_ax.grid(True, axis="y", alpha=0.22, lw=0.45)
    inset_ax.patch.set_alpha(0.92)
    _apply_thin_spines(inset_ax)


def _plot_benchmark_methods_3d(
    out_dir: Path,
    dataset: str,
    methods: list[str],
    lines_by_method: dict[str, pd.DataFrame],
    global_budgets: list[int],
    display_budgets: list[int],
    target_value: float | None,
    metric_prefix: str,
    z_label: str,
    output_name: str,
    target_label: str | None,
    target_style: str,
    add_mape_inset: bool = False,
) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.legend_handler import HandlerTuple

    if not methods or not global_budgets:
        return

    fig = plt.figure(figsize=(W_MULTI * 1.18, W_MULTI * 1.02))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    use_line_styles = metric_prefix in {"mape", "rmspe", "accuracy"}

    z_min, z_max = float("inf"), float("-inf")
    target_crossings: list[tuple[float, float, float, float, str]] = []
    for y_pos, method in enumerate(methods, start=1):
        line = lines_by_method.get(method, pd.DataFrame())
        if line.empty:
            continue
        line = line[pd.to_numeric(line["budget_total"], errors="coerce").astype(float) > 0]
        if line.empty:
            continue
        budgets = line["budget_total"].astype(int).tolist()
        x = np.asarray(budgets, dtype=float)
        y = np.full(len(x), float(y_pos), dtype=float)
        z_mean = line[f"{metric_prefix}_mean"].to_numpy(float)
        z_low = line[f"{metric_prefix}_min"].to_numpy(float)
        z_high = line[f"{metric_prefix}_max"].to_numpy(float)
        if metric_prefix == "mape":
            z_mean = np.where((z_mean >= 0.0) & (z_mean <= 100.0), z_mean, np.nan)
        color = _method_plot_color(method)
        z_min_values = np.clip(z_low, 0.0, 100.0) if metric_prefix == "mape" else z_low
        z_max_values = np.clip(z_high, 0.0, 100.0) if metric_prefix == "mape" else z_high
        if len(z_min_values):
            z_min = min(z_min, float(np.nanmin(z_min_values)))
        if len(z_max_values):
            z_max = max(z_max, float(np.nanmax(z_max_values)))

        if len(x) >= 2:
            if metric_prefix == "mape":
                for seg_idx in range(len(x) - 1):
                    polygon = [
                        (float(x[seg_idx]), float(z_low[seg_idx])),
                        (float(x[seg_idx + 1]), float(z_low[seg_idx + 1])),
                        (float(x[seg_idx + 1]), float(z_high[seg_idx + 1])),
                        (float(x[seg_idx]), float(z_high[seg_idx])),
                    ]
                    clipped = _clip_polygon_by_z(polygon, 0.0, 100.0)
                    if len(clipped) < 3:
                        continue
                    ribbon = [(float(xi), float(y_pos), float(zi)) for xi, zi in clipped]
                    band = Poly3DCollection([ribbon], facecolor=color, alpha=0.15, edgecolor="none")
                    ax.add_collection3d(band)
            else:
                ribbon = [(float(xi), float(yi), float(zi)) for xi, yi, zi in zip(x, y, z_low)]
                ribbon.extend(
                    (float(xi), float(yi), float(zi))
                    for xi, yi, zi in zip(x[::-1], y[::-1], z_high[::-1])
                )
                band = Poly3DCollection([ribbon], facecolor=color, alpha=0.15, edgecolor="none")
                ax.add_collection3d(band)
        elif len(x) == 1:
            if metric_prefix == "mape":
                low = max(0.0, float(z_low[0]))
                high = min(100.0, float(z_high[0]))
                if low <= high:
                    ax.plot([x[0], x[0]], [y[0], y[0]], [low, high], color=color, alpha=0.24, lw=3.0)
            else:
                ax.plot([x[0], x[0]], [y[0], y[0]], [z_low[0], z_high[0]], color=color, alpha=0.24, lw=3.0)

        ax.plot(
            x,
            y,
            z_mean,
            lw=1.45,
            linestyle=_method_line_style(method) if use_line_styles else "-",
            marker="None" if use_line_styles else _method_marker(method),
            markersize=0.0 if use_line_styles else 4.8,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.8,
            color=color,
            label=_display_method_short(method),
        )
        if metric_prefix in {"mape", "rmspe", "accuracy"}:
            crossing = _target_crossing(line, "budget_total", metric_prefix, target_value)
            if crossing is not None:
                x_cross, z_cross, budget_cross = crossing
                target_crossings.append((x_cross, float(y_pos), z_cross, budget_cross, color))

    if target_value is not None and np.isfinite(target_value):
        x0, x1 = float(min(global_budgets)), float(max(global_budgets))
        y0, y1 = 0.55, float(len(methods)) + 0.45
        if target_style == "dashed":
            ax.plot(
                [x0, x1],
                [y1, y1],
                [target_value, target_value],
                color="gray",
                lw=1.2,
                ls="--",
                label=target_label or "_nolegend_",
            )
        else:
            target_plane = Poly3DCollection(
                [[(x0, y0, target_value), (x1, y0, target_value), (x1, y1, target_value), (x0, y1, target_value)]],
                facecolor="gray",
                alpha=0.08,
                edgecolor="none",
            )
            ax.add_collection3d(target_plane)
        z_min = min(z_min, float(target_value))
        z_max = max(z_max, float(target_value))

    if np.isfinite(z_min) and np.isfinite(z_max):
        z_pad = (z_max - z_min) * 0.12 if z_max > z_min else 1.0
        ax.set_zlim(z_min - z_pad, z_max + z_pad)
        z_text_offset = z_pad * 0.30
    else:
        z_text_offset = 1.0

    for x_cross, y_pos, z_cross, budget_cross, color in target_crossings:
        ax.plot(
            [x_cross],
            [y_pos],
            [z_cross],
            marker="v",
            markersize=7.2,
            color=color,
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=1.15,
            linestyle="None",
            label="_nolegend_",
            zorder=8,
        )
        ax.text(
            x_cross,
            y_pos,
            z_cross + z_text_offset,
            _format_budget_annotation(budget_cross),
            ha="center",
            va="bottom",
            fontsize=TARGET_ANNOTATION_FONTSIZE,
            color=color,
            zorder=9,
        )

    if metric_prefix == "mape":
        ax.set_zlim(0.0, 100.0)

    ax.set_xlabel("Full-life test budget", labelpad=10, fontsize=METRIC_LABEL_FONTSIZE)
    ax.set_ylabel("Acquisition function", labelpad=22, fontsize=METRIC_LABEL_FONTSIZE)
    ax.set_zlabel(z_label, labelpad=10, fontsize=METRIC_LABEL_FONTSIZE)
    x_tick_budgets = _budget_ticks_with_max_pool(display_budgets, global_budgets)
    ax.set_xticks(np.asarray(x_tick_budgets, dtype=float))
    ax.set_xticklabels(_axis_budget_value_labels(x_tick_budgets), fontsize=METRIC_PANEL_FONTSIZE)
    ax.set_xlim(float(min(global_budgets)), float(max(global_budgets)))
    ax.set_yticks(np.arange(1, len(methods) + 1, dtype=float))
    ax.set_yticklabels([_display_method_short(method) for method in methods], fontsize=METRIC_PANEL_FONTSIZE)
    for tick_label in ax.get_yticklabels():
        tick_label.set_horizontalalignment("left")
    ax.set_ylim(0.5, float(len(methods)) + 0.5)
    ax.tick_params(axis="z", labelsize=METRIC_PANEL_FONTSIZE)
    ax.view_init(elev=24, azim=-56)
    ax.set_box_aspect((1.45, 1.0, 0.82))
    ax.grid(True, alpha=0.22)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.98))
        axis.pane.set_edgecolor((0.86, 0.86, 0.86, 0.70))
    ax.tick_params(axis="both", which="major", pad=3)
    ax.tick_params(axis="y", which="major", pad=0.7)
    if use_line_styles:
        legend_handles = []
        legend_labels = []
        for method in methods:
            color = _method_plot_color(method)
            band_handle = mpatches.Patch(facecolor=color, edgecolor="none", alpha=0.15)
            line_handle = mlines.Line2D(
                [],
                [],
                color=color,
                linestyle=_method_line_style(method),
                linewidth=1.6,
            )
            legend_handles.append((band_handle, line_handle))
            legend_labels.append(_display_method_short(method))
        ax.legend(
            legend_handles,
            legend_labels,
            handler_map={tuple: HandlerTuple(ndivide=1, pad=0.0)},
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
            frameon=False,
            fontsize=METRIC_PANEL_FONTSIZE,
        )
    else:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=METRIC_PANEL_FONTSIZE)
    if add_mape_inset and metric_prefix == "mape":
        _add_mape_inset_axis(
            fig,
            methods,
            lines_by_method,
            global_budgets,
            display_budgets,
            target_value,
            target_label,
        )
    fig.subplots_adjust(top=0.90, bottom=0.06, left=0.02, right=0.98)
    _save_png(fig, out_dir / output_name, use_tight_layout=False)
    plt.close(fig)


def plot_all_pool_parity(all_pool_predictions: pd.DataFrame, out_dir: Path, dataset: str) -> None:
    sub = all_pool_predictions[all_pool_predictions["dataset"] == dataset].copy()
    if sub.empty:
        return
    import matplotlib.pyplot as plt

    set_plot_style()
    apply_global_style()

    trials = sorted(sub["trial"].unique())
    n_cols = min(5, len(trials))
    n_rows = int(np.ceil(len(trials) / n_cols))
    unit = "Week" if _unit_label(dataset) == "week" else "cycle"

    # 关键修改 1：显著增加 figure 高度，避免子图被压扁或截断
    fig_w = W_MULTI * 1.15 if len(trials) >= 8 else W_MULTI
    fig_h = fig_w * 0.50 if n_rows <= 2 else fig_w * (n_rows / n_cols) * 1.10

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False
    )

    lo = float(min(sub["lifetime"].min(), sub["pred_lifetime"].min()))
    hi = float(max(sub["lifetime"].max(), sub["pred_lifetime"].max()))

    # 给坐标轴范围加一点 padding，避免点或对角线贴边
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    lo_plot = lo - pad
    hi_plot = hi + pad

    vmin = float(sub["lifetime"].min())
    vmax = float(sub["lifetime"].max())

    sc = None
    for idx, (ax, trial) in enumerate(zip(axes.ravel(), trials)):
        part = sub[sub["trial"] == trial]

        sc = ax.scatter(
            part["lifetime"],
            part["pred_lifetime"],
            c=part["lifetime"],
            cmap=NATURE_CMAP,
            vmin=vmin,
            vmax=vmax,
            s=17,
            alpha=0.88,
            edgecolors="white",
            linewidths=0.15,
            rasterized=True
        )

        ax.plot(
            [lo_plot, hi_plot],
            [lo_plot, hi_plot],
            color="0.15",
            lw=0.9,
            ls="--"
        )

        ax.set_xlim(lo_plot, hi_plot)
        ax.set_ylim(lo_plot, hi_plot)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.grid(True, color="0.86", lw=0.45, alpha=0.80)

        # 保持每个子图近似正方形
        ax.set_aspect("equal", adjustable="box")

        ax.set_title("")
        ax.text(
            0.05,
            0.95,
            f"Trial {int(trial)}",
            transform=ax.transAxes,
            bbox=dict(
                facecolor="white",
                edgecolor="0.75",
                linewidth=0.35,
                alpha=0.92,
                boxstyle="square,pad=0.20"
            ),
            ha="left",
            va="top",
            fontsize=6.2
        )

        # 只保留最左侧和最底部的 tick labels
        if idx % n_cols != 0:
            ax.set_yticklabels([])

        if idx < n_cols * (n_rows - 1):
            ax.set_xticklabels([])

        ax.set_xlabel("")
        ax.set_ylabel("")
        _apply_thin_spines(ax)

    for ax in axes.ravel()[len(trials):]:
        ax.axis("off")

    # 关键修改 2：整图标签位置不要太靠边
    fig.supxlabel(
        f"True lifetime ({unit})",
        y=0.115,
        fontsize=8
    )

    fig.supylabel(
        f"Predicted lifetime ({unit})",
        x=0.04, y=0.6,
        fontsize=8
    )

    # 关键修改 3：上方水平 color bar，长度为整图一半
    # [left, bottom, width, height] 都是 figure 坐标，0~1
    cbar_ax = fig.add_axes([0.30, 0.045, 0.40, 0.020])

    sm = cm.ScalarMappable(
        cmap=NATURE_CMAP,
        norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        cax=cbar_ax,
        orientation="horizontal"
    )

    cbar.set_label(
        f"Battery lifetime ({unit})",
        size=7.5,
        labelpad=2
    )

    cbar.outline.set_visible(False)
    cbar.ax.tick_params(
        labelsize=7,
        width=0.6,
        length=2.5,
        pad=1
    )

    # 可选：让 colorbar 的 label 和 ticks 放在上方，更不容易和子图冲突
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.xaxis.set_label_position("bottom")

    # 关键修改 4：top 要给 colorbar 留空间，bottom 给总 x-label 留空间
    fig.subplots_adjust(
        bottom=0.22,
        left=0.11,
        right=0.985,
        top=0.975,
        hspace=0.10,
        wspace=0.12
    )

    fig.suptitle("")

    # 推荐直接 savefig，避免 tight_layout 再次改动布局
    _save_png(fig, out_dir / "all_pool_parity.png")

    plt.close(fig)


def _unit_label(dataset: str) -> str:
    return "week" if str(dataset) == "ISU_ILCC" else "cycle"


def _target_mape(df: pd.DataFrame) -> float:
    if "all_pool_accuracy" in df.columns:
        baseline_accuracy = df["all_pool_accuracy"].dropna()
        if not baseline_accuracy.empty:
            return float((1.0 - baseline_accuracy.mean() * ALL_POOL_ACCURACY_FRACTION) * 100.0)
    if "all_pool_mape" not in df.columns:
        return float("nan")
    baseline_mape = df["all_pool_mape"].dropna()
    if baseline_mape.empty:
        return float("nan")
    baseline_accuracy = 1.0 - baseline_mape.mean() / 100.0
    return float((1.0 - baseline_accuracy * ALL_POOL_ACCURACY_FRACTION) * 100.0)


def _interpolated_budget_required(summary: pd.DataFrame, budget_col: str, target: float) -> tuple[float, bool]:
    line = summary[[budget_col, "mape_mean"]].dropna().sort_values(budget_col)
    if line.empty or not np.isfinite(target):
        return float("nan"), False
    budgets = line[budget_col].to_numpy(float)
    mapes = line["mape_mean"].to_numpy(float)
    reached = mapes <= target
    if reached.any():
        first = int(np.argmax(reached))
        if first == 0:
            return float(budgets[0]), True
        x0, x1 = budgets[first - 1], budgets[first]
        y0, y1 = mapes[first - 1], mapes[first]
        if np.isclose(y0, y1):
            return float(x1), True
        frac = (target - y0) / (y1 - y0)
        return float(x0 + frac * (x1 - x0)), True
    return float(budgets.max()), False


def _budget_requirement_table(df: pd.DataFrame, group_cols: list[str], budget_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    target = _target_mape(df)
    rows = []
    summary = _line_summary(df, group_cols, budget_col)
    for keys, part in summary.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        required, reached = _interpolated_budget_required(part, budget_col, target)
        row = {col: value for col, value in zip(group_cols, keys)}
        row.update({"required_budget": required, "reached_target": reached, "target_mape": target})
        rows.append(row)
    return pd.DataFrame(rows)


def _bubble_sizes(values: pd.Series) -> np.ndarray:
    vals = values.astype(float).to_numpy()
    finite = np.isfinite(vals)
    if not finite.any():
        return np.full(len(vals), 20.0)
    lo = np.nanmin(vals)
    hi = np.nanmax(vals)
    if np.isclose(lo, hi):
        return np.full(len(vals), 60.0)
    return 15.0 + 300.0 * ((vals - lo) / (hi - lo)) ** 1.5


def plot_benchmark_budget_requirement_bubble(
    metrics: pd.DataFrame,
    out_dir: Path,
    dataset: str,
    iterative_batch_size: int = DEFAULT_ITERATIVE_BATCH_SIZE,
) -> None:
    sub = _filter_iterative_batch_size(
        metrics[(metrics["dataset"] == dataset) & (metrics["experiment_mode"] == BENCHMARK_MODE)],
        iterative_batch_size,
    )
    if sub.empty:
        return
    import matplotlib.pyplot as plt

    set_plot_style()
    apply_global_style()
    req = _budget_requirement_table(sub, ["method"], "budget_total")
    if req.empty:
        return
    methods = _ordered_methods(req["method"])
    method_to_y = {m: i for i, m in enumerate(methods)}
    fig, ax = plt.subplots(figsize=(W_SINGLE, W_SINGLE * 1.2))
    values = req["required_budget"].astype(float)
    sizes = _bubble_sizes(values)

    x_coords = np.zeros(len(req))
    y_coords = [method_to_y[m] for m in req["method"]]

    sc = ax.scatter(
        x_coords, y_coords,
        s=sizes, c=values.where(req["reached_target"], np.nan),
        cmap=BUBBLE_CMAP, edgecolor="none", alpha=0.90,
    )
    ax.scatter(
        x_coords - 0.02, np.array(y_coords) + 0.05,
        s=sizes * 0.15, c="white", edgecolor="none", alpha=0.6,
    )

    not_reached = req[~req["reached_target"]]
    if not not_reached.empty:
        nsizes = _bubble_sizes(not_reached["required_budget"])
        nx = np.zeros(len(not_reached))
        ny = [method_to_y[m] for m in not_reached["method"]]
        ax.scatter(nx, ny, s=nsizes, c="0.82", edgecolor="none", alpha=0.75)
        ax.scatter(nx - 0.02, np.array(ny) + 0.05, s=nsizes * 0.15, c="white", edgecolor="none", alpha=0.6)

    for _, row in req.iterrows():
        label = f"{row['required_budget']:.1f}" if row["reached_target"] else f">{row['required_budget']:.0f}"
        ax.text(0, method_to_y[row["method"]] + 0.35, label, va="bottom", ha="center", fontsize=6)

    ax.set_xlim(-0.55, 0.95)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([_display_method(m) for m in methods])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.set_ylim(-0.5, len(methods) - 0.2)
    _apply_thin_spines(ax)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.060, pad=0.08)
    cbar.outline.set_visible(False)
    cbar.set_label("Required full-life test budget", labelpad=15)
    _save_png(fig, out_dir / "budget_requirement_bubble.png")
    plt.close(fig)


def plot_benchmark_channel_occupancy(
    metrics: pd.DataFrame,
    out_dir: Path,
    dataset: str,
    iterative_batch_size: int = DEFAULT_ITERATIVE_BATCH_SIZE,
) -> None:
    sub = _filter_iterative_batch_size(
        metrics[(metrics["dataset"] == dataset) & (metrics["experiment_mode"] == BENCHMARK_MODE)],
        iterative_batch_size,
    )
    if sub.empty:
        return
    import matplotlib.pyplot as plt

    set_plot_style()
    apply_global_style()
    sub["avg_channel_occupancy"] = sub["total_test_cost"] / sub["labeled_train_count"].replace(0, np.nan)
    methods = _ordered_methods(sub["method"])[:6]
    req = _budget_requirement_table(sub, ["method"], "budget_total")
    required = {row["method"]: row["required_budget"] for _, row in req.iterrows()} if not req.empty else {}
    unit = _unit_label(dataset)
    
    global_budgets = sorted(sub["budget_total"].dropna().astype(int).unique().tolist())
    
    fig, axes = plt.subplots(2, 3, figsize=(W_MULTI, W_MULTI * 0.75), squeeze=False)
    for idx, (ax, method) in enumerate(zip(axes.ravel(), methods)):
        method_df = sub[sub["method"] == method]
        by_trial = method_df.groupby(["budget_total", "trial"], dropna=False, as_index=False).agg(
            total_test_cost=("total_test_cost", "mean"),
            avg_channel_occupancy=("avg_channel_occupancy", "mean"),
        )
        summary = by_trial.groupby("budget_total", as_index=False).agg(
            cost_mean=("total_test_cost", "mean"),
            occ_mean=("avg_channel_occupancy", "mean"),
        ).sort_values("budget_total")
        
        color = HIGH_CONTRAST_COLORS[idx % len(HIGH_CONTRAST_COLORS)]
        marker = MARKERS[idx % len(MARKERS)]
        
        budgets = summary["budget_total"].astype(int).tolist()
        x = [global_budgets.index(b) for b in budgets]
        
        ax.plot(x, summary["occ_mean"], marker=marker, ms=5, lw=1.5, ls="--", color=color, label=f"Average channel occupancy")
        
        if method in required and np.isfinite(required[method]):
            req_b = required[method]
            interp_x = _get_interp_x(req_b, global_budgets)
            interp_y = np.interp(req_b, summary["budget_total"], summary["occ_mean"])
            ax.scatter(interp_x, interp_y, marker="v", s=80, color="#E15759", edgecolor="black", zorder=5, label="Required budget to reach target")
            
        ax.set_xticks(np.arange(len(global_budgets)))
        ax.set_xticklabels(_axis_budget_labels(global_budgets), rotation=0)
        
        if idx % 3 == 0:
            ax.set_ylabel(f"Average channel occupancy ({unit})")
        else:
            ax.set_ylabel("")
            
        if idx >= 3:
            ax.set_xlabel("Full-life test budget")
        else:
            ax.set_xlabel("")
        
        ax.text(0.05, 0.92, _display_method(method), transform=ax.transAxes, bbox=dict(facecolor="white", edgecolor="black"), ha="left", va="top", fontsize=8)
        
    for ax in axes.ravel()[len(methods) :]:
        ax.axis("off")
        
    handles_dict = {}
    for ax in axes.ravel()[:len(methods)]:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in handles_dict:
                handles_dict[label] = handle
                
    if handles_dict:
        fig.legend(handles_dict.values(), handles_dict.keys(), loc="lower center", bbox_to_anchor=(0.5, 0.94), ncol=len(handles_dict), fontsize=8, frameon=False)
        
    fig.subplots_adjust(top=0.90, hspace=0.35, wspace=0.25)
    _save_png(fig, out_dir / "avg_channel_occupancy.png", use_tight_layout=False)
    plt.close(fig)


def _overview_mode_table(
    metrics: pd.DataFrame,
    experiment_mode: str,
    iterative_batch_size: int,
) -> tuple[pd.DataFrame, str]:
    sub = _filter_iterative_batch_size(metrics[metrics["experiment_mode"] == experiment_mode], iterative_batch_size)
    if sub.empty:
        return sub, "budget_total"
    budget_col = "budget_total"
    required = {"dataset", "regressor", "method", "trial", budget_col, "mape"}
    missing = required.difference(sub.columns)
    if missing:
        raise ValueError(f"Cannot plot MAPE overview; missing columns: {sorted(missing)}")
    trial_curve = (
        sub.groupby(["dataset", "regressor", "method", "trial", budget_col], dropna=False, as_index=False)
        .agg(mape=("mape", "mean"))
        .dropna(subset=[budget_col, "mape"])
    )
    summary = (
        trial_curve.groupby(["dataset", "regressor", "method", budget_col], dropna=False, as_index=False)
        .agg(
            mape_mean=("mape", "mean"),
            mape_min=("mape", "min"),
            mape_max=("mape", "max"),
            n_trials=("trial", "nunique"),
        )
        .sort_values(["dataset", "regressor", "method", budget_col])
    )
    return summary, budget_col


def plot_mape_budget_overview(
    metrics: pd.DataFrame,
    out_dir: Path,
    iterative_batch_size: int = DEFAULT_ITERATIVE_BATCH_SIZE,
) -> None:
    if metrics.empty:
        return
    import matplotlib.pyplot as plt

    set_plot_style()
    apply_global_style()
    ensure_dir(out_dir)

    dataset_order = [d for d in ["MIT", "ISU_ILCC", "LSD_Primary", "LSD_Second"] if d in set(metrics["dataset"].astype(str))]
    if not dataset_order:
        dataset_order = sorted(metrics["dataset"].dropna().astype(str).unique().tolist())
    present_regressors = set(metrics["regressor"].dropna().astype(str)) if "regressor" in metrics.columns else set()
    regressor_order = [r for r in REGRESSOR_ORDER if r in present_regressors]
    if not regressor_order and "regressor" in metrics.columns:
        regressor_order = sorted(metrics["regressor"].dropna().astype(str).unique().tolist())
    if not dataset_order or not regressor_order:
        return

    method_order = [m for m in METHOD_ORDER if m in set(metrics["method"].dropna().astype(str))]
    method_colors = {m: HIGH_CONTRAST_COLORS[i % len(HIGH_CONTRAST_COLORS)] for i, m in enumerate(method_order)}
    method_markers = {m: MARKERS[i % len(MARKERS)] for i, m in enumerate(method_order)}

    for experiment_mode, mode_label in [(BENCHMARK_MODE, MODE_LABELS.get(BENCHMARK_MODE, BENCHMARK_MODE))]:
        summary, budget_col = _overview_mode_table(metrics, experiment_mode, iterative_batch_size)
        if summary.empty:
            continue
        fig, axes = plt.subplots(
            len(dataset_order),
            len(regressor_order),
            figsize=(max(12.0, 3.6 * len(regressor_order)), max(10.0, 2.35 * len(dataset_order))),
            squeeze=False,
            sharey=False,
        )
        for row_idx, dataset in enumerate(dataset_order):
            for col_idx, regressor in enumerate(regressor_order):
                ax = axes[row_idx, col_idx]
                panel = summary[
                    (summary["dataset"].astype(str) == dataset)
                    & (summary["regressor"].astype(str) == regressor)
                ].copy()
                if panel.empty:
                    ax.axis("off")
                    continue
                budgets = sorted(panel[budget_col].dropna().astype(int).unique().tolist())
                x_lookup = {budget: idx for idx, budget in enumerate(budgets)}
                for method in method_order:
                    line = panel[panel["method"].astype(str) == method].sort_values(budget_col)
                    if line.empty:
                        continue
                    x = np.asarray([x_lookup[int(b)] for b in line[budget_col]], dtype=float)
                    y_mean = line["mape_mean"].to_numpy(float)
                    y_min = line["mape_min"].to_numpy(float)
                    y_max = line["mape_max"].to_numpy(float)
                    color = method_colors[method]
                    ax.fill_between(x, y_min, y_max, color=color, alpha=0.14, linewidth=0)
                    ax.plot(
                        x,
                        y_mean,
                        color=color,
                        marker=method_markers[method],
                        markersize=3.2,
                        linewidth=1.05,
                        label=_display_method(method),
                    )
                display_idx = _half_tick_indices(budgets)
                ax.set_xticks(display_idx)
                ax.set_xticklabels(_axis_budget_labels(budgets, display_idx), fontsize=7)
                ax.grid(True, axis="y", alpha=0.28)
                _apply_thin_spines(ax)
                if row_idx == 0:
                    ax.set_title(REGRESSOR_LABELS.get(regressor, regressor), fontsize=9, color="#1F2430", pad=6)
                if col_idx == 0:
                    ax.set_ylabel(f"{dataset}\nMAPE (%)", fontsize=8)
                else:
                    ax.set_ylabel("")
                if row_idx == len(dataset_order) - 1:
                    ax.set_xlabel("Full-life budget", fontsize=8)
                else:
                    ax.set_xlabel("")

        handles, labels = [], []
        for ax in axes.ravel():
            h, l = ax.get_legend_handles_labels()
            for handle, label in zip(h, l):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
        if handles:
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.978), ncol=min(len(handles), 6), frameon=False, fontsize=8)
        fig.text(
            0.02,
            0.992,
            f"{mode_label}: MAPE vs budget across datasets and lifetime predictors",
            ha="left",
            va="top",
            fontsize=13,
            color="#1F2430",
            weight="bold",
        )
        fig.text(
            0.02,
            0.958,
            "Lines show mean MAPE across trials; translucent bands show best-to-worst trial range at each budget.",
            ha="left",
            va="top",
            fontsize=9,
            color="#6F768A",
        )
        fig.subplots_adjust(top=0.90, bottom=0.07, left=0.08, right=0.985, hspace=0.42, wspace=0.22)
        _save_png(fig, out_dir / "MAPE_vs_budget_all_datasets_models.png", use_tight_layout=False)
        plt.close(fig)


def plot_dataset_outputs(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    out_dir: Path,
    dataset: str,
    all_pool_predictions: pd.DataFrame | None = None,
    iterative_batch_size: int = DEFAULT_ITERATIVE_BATCH_SIZE,
) -> None:
    ensure_dir(out_dir)
    if "regressor" in metrics.columns:
        regressors = sorted(metrics["regressor"].dropna().astype(str).unique().tolist())
        if len(regressors) > 1:
            for regressor in regressors:
                sub_metrics = metrics[metrics["regressor"].astype(str) == regressor].copy()
                sub_predictions = predictions
                if predictions is not None and not predictions.empty and "regressor" in predictions.columns:
                    sub_predictions = predictions[predictions["regressor"].astype(str) == regressor].copy()
                sub_all_pool = all_pool_predictions
                if all_pool_predictions is not None and not all_pool_predictions.empty and "regressor" in all_pool_predictions.columns:
                    sub_all_pool = all_pool_predictions[all_pool_predictions["regressor"].astype(str) == regressor].copy()
                regressor_out = ensure_dir(out_dir / regressor)
                plot_dataset_outputs(
                    sub_metrics,
                    sub_predictions,
                    regressor_out,
                    dataset,
                    sub_all_pool,
                    iterative_batch_size,
                )
            return
    plot_mape_vs_budget(metrics, out_dir, dataset, BENCHMARK_MODE, iterative_batch_size)
    plot_rmspe_vs_budget(metrics, out_dir, dataset, BENCHMARK_MODE, iterative_batch_size)
    plot_benchmark_methods(metrics, out_dir, dataset, iterative_batch_size)
    plot_benchmark_budget_requirement_bubble(metrics, out_dir, dataset, iterative_batch_size)
    plot_benchmark_channel_occupancy(metrics, out_dir, dataset, iterative_batch_size)
    
    if all_pool_predictions is not None and not all_pool_predictions.empty:
        plot_all_pool_parity(all_pool_predictions, out_dir, dataset)

plot_parity = plot_all_pool_parity


DL_DATASET_ORDER = ["MIT", "ISU_ILCC", "LSD_Primary", "LSD_Second"]
DL_MODEL_ORDER = ["CNN", "TabNet", "DeepVAE"]

DL_DATASET_DISPLAY = {
    "MIT": "MIT",
    "ISU_ILCC": "ISU-ILCC",
    "LSD_Primary": "LSD primary",
    "LSD_Second": "LSD second",
}

DL_MODEL_DISPLAY = {
    "CNN": "CNN",
    "TabNet": "TabNet",
    "DeepVAE": "Deep VAE",
}

DL_MODEL_COLORS = {
    "CNN": "#4E79A7",
    "TabNet": "#E15759",
    "DeepVAE": "#59A14F",
}

DL_MODEL_MARKERS = {
    "CNN": "o",
    "TabNet": "s",
    "DeepVAE": "^",
}


def _dl_ordered(values: pd.Series | list[str], preferred: list[str]) -> list[str]:
    seen = set(values)
    ordered = [item for item in preferred if item in seen]
    ordered.extend(sorted(seen.difference(ordered)))
    return ordered


def latest_dl_benchmark_run_dir(output_root: Path) -> Path:
    candidates = [
        path
        for path in output_root.iterdir()
        if path.is_dir()
        and (path / "metrics_summary.csv").exists()
        and (path / "predictions_summary.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No DL benchmark run with summary/predictions was found under {output_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _dl_unit_label(dataset: str) -> str:
    return "Lifetime (weeks)" if dataset == "ISU_ILCC" else "Lifetime (cycles)"


def read_dl_benchmark_results(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = run_dir / "metrics_summary.csv"
    predictions_path = run_dir / "predictions_summary.csv"
    summary = pd.read_csv(summary_path)
    predictions = pd.read_csv(predictions_path)

    required_summary = {"dataset", "trial", "model", "mape"}
    required_predictions = {"dataset", "trial", "model", "lifetime", "pred_lifetime"}
    missing_summary = required_summary.difference(summary.columns)
    missing_predictions = required_predictions.difference(predictions.columns)
    if missing_summary:
        raise ValueError(f"{summary_path} is missing columns: {sorted(missing_summary)}")
    if missing_predictions:
        raise ValueError(f"{predictions_path} is missing columns: {sorted(missing_predictions)}")

    for col in ("trial", "mape"):
        summary[col] = pd.to_numeric(summary[col], errors="coerce")
    for col in ("trial", "lifetime", "pred_lifetime"):
        predictions[col] = pd.to_numeric(predictions[col], errors="coerce")

    summary = summary.dropna(subset=["dataset", "trial", "model", "mape"]).copy()
    predictions = predictions.dropna(subset=["dataset", "trial", "model", "lifetime", "pred_lifetime"]).copy()
    summary["trial"] = summary["trial"].astype(int)
    predictions["trial"] = predictions["trial"].astype(int)
    return summary, predictions


def _dl_mape_table(summary: pd.DataFrame) -> pd.DataFrame:
    table = (
        summary.groupby(["dataset", "model"], as_index=False)
        .agg(
            mean_mape=("mape", "mean"),
            std_mape=("mape", "std"),
            n_trials=("trial", "nunique"),
        )
        .sort_values(["dataset", "model"])
    )
    table["std_mape"] = table["std_mape"].fillna(0.0)
    return table


def plot_dl_benchmark_mape_bars(summary: pd.DataFrame, out_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    datasets = _dl_ordered(summary["dataset"].unique().tolist(), DL_DATASET_ORDER)
    models = _dl_ordered(summary["model"].unique().tolist(), DL_MODEL_ORDER)
    table = _dl_mape_table(summary)

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    x = np.arange(len(datasets))
    width = min(0.22, 0.78 / max(len(models), 1))

    for i, model in enumerate(models):
        sub = table[table["model"] == model].set_index("dataset")
        means = [sub.loc[d, "mean_mape"] if d in sub.index else np.nan for d in datasets]
        stds = np.array([sub.loc[d, "std_mape"] if d in sub.index else 0.0 for d in datasets], dtype=float)
        means_array = np.array(means, dtype=float)
        yerr = np.vstack([np.minimum(stds, means_array), stds])
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width=width,
            yerr=yerr,
            label=DL_MODEL_DISPLAY.get(model, model),
            color=DL_MODEL_COLORS.get(model, "#777777"),
            edgecolor="none",
            linewidth=0,
            capsize=3,
            error_kw={"elinewidth": 1.0, "ecolor": "#4A4A4A", "capthick": 1.0},
            alpha=0.92,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([DL_DATASET_DISPLAY.get(d, d) for d in datasets])
    ax.set_ylabel("MAPE (%)")
    ax.set_xlabel("")
    fig.suptitle("Deep learning benchmark MAPE", x=0.08, y=0.985, ha="left", fontsize=13, fontweight="bold")
    fig.text(
        0.08,
        0.935,
        "Mean +/- SD across independent trials; lower is better.",
        ha="left",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    ax.legend(frameon=False, ncol=len(models), loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.05)

    out_path = out_dir / "mape_by_dataset.png"
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save_png(fig, out_path, use_tight_layout=False)
    plt.close(fig)
    return out_path


def _dl_axis_limits(values: pd.DataFrame) -> tuple[float, float]:
    clean = values[["lifetime", "pred_lifetime"]].replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0, 1.0
    arr = clean.to_numpy(dtype=float).ravel()
    low = float(np.nanpercentile(arr, 0.5))
    high = float(np.nanpercentile(arr, 99.0))
    high = max(high, float(clean["lifetime"].max()))
    if np.isclose(low, high):
        pad = max(abs(high) * 0.05, 1.0)
    else:
        pad = (high - low) * 0.06
    return max(0.0, low - pad), high + pad


def plot_dl_benchmark_model_parity(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    dataset: str,
    model: str,
    out_dir: Path,
) -> Path | None:
    import matplotlib.pyplot as plt

    data = predictions[(predictions["dataset"] == dataset) & (predictions["model"] == model)].copy()
    if data.empty:
        return None

    trials = sorted(data["trial"].unique().tolist())[:10]
    lim_low, lim_high = _dl_axis_limits(data)

    fig, axes = plt.subplots(2, 5, figsize=(16.0, 7.2), sharex=True, sharey=True)
    axes_vector = axes.ravel()
    unit = _dl_unit_label(dataset)

    for ax_idx, ax in enumerate(axes_vector):
        if ax_idx >= len(trials):
            ax.axis("off")
            continue
        trial = trials[ax_idx]
        trial_data = data[data["trial"] == trial]

        ax.scatter(
            trial_data["lifetime"],
            trial_data["pred_lifetime"],
            s=22,
            alpha=0.62,
            color=DL_MODEL_COLORS.get(model, "#777777"),
            marker=DL_MODEL_MARKERS.get(model, "o"),
            edgecolors="none",
            label=DL_MODEL_DISPLAY.get(model, model),
        )

        ax.plot([lim_low, lim_high], [lim_low, lim_high], color="#2F2F2F", linewidth=1.0, linestyle="--", alpha=0.8)
        ax.set_xlim(lim_low, lim_high)
        ax.set_ylim(lim_low, lim_high)
        ax.set_title(f"Trial {trial}", fontweight="bold", pad=6)
        ax.grid(True, axis="both")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        off_scale = trial_data[
            (trial_data["lifetime"] < lim_low)
            | (trial_data["lifetime"] > lim_high)
            | (trial_data["pred_lifetime"] < lim_low)
            | (trial_data["pred_lifetime"] > lim_high)
        ]
        if not off_scale.empty:
            ax.text(
                0.97,
                0.03,
                f"off-scale n={len(off_scale)}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.2,
                color="#8A3A3A",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.78),
            )

        metric_rows = summary[(summary["dataset"] == dataset) & (summary["trial"] == trial) & (summary["model"] == model)]
        metric_text = []
        if not metric_rows.empty:
            metric_text.append(f"{DL_MODEL_DISPLAY.get(model, model)} {metric_rows['mape'].iloc[0]:.1f}%")
        if metric_text:
            ax.text(
                0.03,
                0.97,
                "\n".join(metric_text),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.2,
                color="#333333",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.72),
            )

    for ax in axes[-1, :]:
        ax.set_xlabel(f"True {unit.lower()}")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"Predicted {unit.lower()}")

    fig.suptitle(
        f"{DL_DATASET_DISPLAY.get(dataset, dataset)} {DL_MODEL_DISPLAY.get(model, model)} parity",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.08, top=0.86, wspace=0.20, hspace=0.30)

    out_path = out_dir / "parity.png"
    _save_png(fig, out_path, use_tight_layout=False)
    plt.close(fig)
    return out_path


def create_dl_benchmark_visualizations(run_dir: Path, out_dir: Path | None = None) -> list[Path]:
    set_plot_style()
    feature_runs_path = run_dir / "feature_set_runs.csv"
    if out_dir is None and feature_runs_path.exists() and not (run_dir / "metrics_summary.csv").exists():
        written: list[Path] = []
        feature_runs = pd.read_csv(feature_runs_path)
        for _, item in feature_runs.iterrows():
            child = Path(str(item["feature_run_dir"]))
            if not child.is_absolute():
                child = run_dir / child
            written.extend(create_dl_benchmark_visualizations(child.resolve()))
        return written

    summary, predictions = read_dl_benchmark_results(run_dir)
    out_dir = ensure_dir(out_dir or (run_dir / "figures"))

    written: list[Path] = []
    summary_table = _dl_mape_table(summary)
    summary_table.to_csv(out_dir / "mape_summary.csv", index=False)
    written.append(out_dir / "mape_summary.csv")

    written.append(plot_dl_benchmark_mape_bars(summary, out_dir))
    for dataset in _dl_ordered(predictions["dataset"].unique().tolist(), DL_DATASET_ORDER):
        dataset_data = predictions[predictions["dataset"] == dataset]
        for model in _dl_ordered(dataset_data["model"].unique().tolist(), DL_MODEL_ORDER):
            model_dir = ensure_dir(run_dir / str(dataset) / "figures" / str(model))
            out_path = plot_dl_benchmark_model_parity(predictions, summary, dataset, model, model_dir)
            if out_path is not None:
                written.append(out_path)
    return written

