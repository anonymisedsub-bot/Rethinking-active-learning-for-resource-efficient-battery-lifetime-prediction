from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
from PIL import Image

MODULE_ROOT = Path(__file__).resolve().parents[1]
COMMON_DIR = MODULE_ROOT.parent / "common"
sys.path.insert(0, str(COMMON_DIR))

from config import (
    ACQUISITION_ORDER,
    DATASET_ORDER,
    DURATION_UNITS,
    FEATURE_ORDER,
    MODEL_ORDER,
)


NFP_THRESHOLDS = (92, 95, 98)
PRIMARY_NFP_THRESHOLD = 95
SENSITIVITY_NFP_THRESHOLDS = (92, 98)
THRESHOLD_DISPLAY_ORDER = (95, 92, 98)
FIGURE_STEM = "fig_nfp_performance_indicator_summary"
DEFAULT_EXPORT_FORMATS = ("png",)
FIGURE_SIZE_INCHES = (8.4, 9.8)
FIGURE_ARCHETYPE = "asymmetric mixed-modality figure"
ANALYSIS_ROOT = MODULE_ROOT
DEFAULT_SOURCE_DIR = MODULE_ROOT / "source_data"
DEFAULT_OUTPUT_DIR = MODULE_ROOT
DEFAULT_ALC_CURVE_SOURCE = (
    ANALYSIS_ROOT
    / "Section 3-Relative ALC Profiles"
    / "source_data"
    / "relative_alc_learning_curve_points.csv"
)
DEFAULT_FULL_POOL_SOURCE = (
    ANALYSIS_ROOT
    / "Section 1-Full-Pool Headroom and Initialisation Effects"
    / "source_data"
    / "fig_full_pool_primary_vs_deep_violin_trial_level.csv"
)
PANEL_A_TARGET_FRACTION = 0.30
PANEL_A_REQUIRED_TRIALS = 10
MARKER_ALPHA = 0.6
AXIS_LINEWIDTH = 0.4
MEDIAN_CONNECTOR_COLOR = "#000000"
PANEL_C_MARKER_AREA_SENSITIVITY = 15.0
PANEL_C_MARKER_AREA_PRIMARY = 32.0
FONT_SIZE_DELTA_PT = 2.0
HEATMAP_CMAP_COLORS = ("#3C5488", "#F7F7F3", "#E64B35")
HEATMAP_VALUE_RANGE = (30.0, 75.0)
HEATMAP_COLORBAR_TICKS = (30.0, 45.0, 60.0, 75.0)
HEATMAP_MATRIX_WIDTH_MM = 52.0
HEATMAP_MATRIX_HEIGHT_MM = 52.0
PANEL_A_PLOT_HEIGHT_MM = 55.0
PANEL_CD_CLEARANCE_SHIFT_MM = 1.0
PANEL_H_THRESHOLD_OFFSETS = {92: -0.24, 95: 0.0, 98: 0.24}
DURATION_Y_AXIS_UPPER_FACTOR = 1.25
PANEL_H_Y_AXIS_LIMITS = (7.55, -0.55)
VIOLIN_PANEL_LEGEND_BBOX_Y = 1.015
EMBEDDED_SPLIT_PANEL_LEGENDS = ("e", "f", "g", "h")
EXPORT_ONLY_LEGEND_PANELS = ("c",)

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
MODEL_LABELS = {"GPR": "GPR", "RF": "RF", "AE_ENet": "AE-ENet"}
FEATURE_LABELS = {
    feature_set: f"FS {index}"
    for index, feature_set in enumerate(FEATURE_ORDER, start=1)
}
ACQUISITION_LABELS = {
    "random_selection": "Random",
    "diversity_oneshot": "Diversity (One-shot)",
    "diversity_iterative": "Diversity (Iterative)",
    "coverage": "Coverage",
    "exploration": "Exploration",
    "exploitation": "Exploitation",
    "hybrid": "Hybrid",
}
MARKER_BY_ACQUISITION = {
    "random_selection": "o",
    "diversity_oneshot": "s",
    "diversity_iterative": "^",
    "coverage": "D",
    "exploration": "v",
    "exploitation": "P",
    "hybrid": "X",
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
MARKER_BY_THRESHOLD = {92: "o", 95: "s", 98: "D"}
NFP_COLORS = {92: "#B9CBDC", 95: "#275D8C", 98: "#D8A99E"}
PANEL_TYPES = {
    "a": "tradeoff_2x3_small_multiples",
    "b": "nfp95_heatmap",
    "c": "tests_threshold_trend",
    "d": "fastest_win_count",
    "e": "duration_cycles",
    "f": "duration_efc",
    "g": "duration_weeks",
    "h": "nfp95_sensitivity_delta",
}
PANEL_FILE_STEMS = {
    "a": "fig_efficiency_tradeoff",
    "b": "fig_nfp95_pool_fraction",
    "c": "fig_full_life_test_sensitivity",
    "d": "fig_primary_strategy_wins",
    "e": "fig_duration_cycles",
    "f": "fig_duration_efc",
    "g": "fig_duration_weeks",
    "h": "fig_pool_fraction_sensitivity",
}

REQUIRED_COLUMNS = {
    "dataset",
    "feature_set",
    "model",
    "trial",
    "acquisition",
    "nfp_percent",
    "positive_attainable_gain",
    "reached",
    "pool_n",
    "required_pool_fraction_pct",
    "required_full_life_tests",
    "required_duration",
    "duration_unit",
}
SUMMARY_COLUMNS = {
    "dataset",
    "acquisition",
    "nfp_percent",
    "pool_n",
    "duration_unit",
    "n_observations",
    "required_full_life_tests_median",
    "required_full_life_tests_q25",
    "required_full_life_tests_q75",
    "required_pool_fraction_pct_median",
    "required_pool_fraction_pct_q25",
    "required_pool_fraction_pct_q75",
    "required_duration_median",
    "required_duration_q25",
    "required_duration_q75",
}
TRIAL_SUMMARY_COLUMNS = {
    "dataset",
    "acquisition",
    "nfp_percent",
    "trial",
    "duration_unit",
    "n_configurations",
    "required_duration",
    "required_pool_fraction_pct",
}
PANEL_A_REQUIRED_COLUMNS = {
    "dataset",
    "model",
    "feature_set",
    "role",
    "acquisition",
    "median_relative_alc_improvement_pct",
    "q25_relative_alc_improvement_pct",
    "q75_relative_alc_improvement_pct",
    "n_trials",
    "pool_n",
    "labelled_n",
    "achieved_labelled_fraction",
    "target_labelled_fraction",
    "mean_full_pool_mape",
}


def _require_columns(frame: pd.DataFrame, required: set[str], table_name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"{table_name} is missing required columns: {missing}")


def _as_bool(values: pd.Series, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        if values.isna().any():
            raise ValueError(f"Unrecognised boolean values in {column}: [null]")
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
    invalid = sorted(set(normalized).difference(mapping))
    if invalid:
        raise ValueError(f"Unrecognised boolean values in {column}: {invalid}")
    return normalized.map(mapping).astype(bool)


def _strict_numeric(
    values: pd.Series,
    column: str,
    *,
    integer: bool = False,
) -> pd.Series:
    parsed = pd.to_numeric(values, errors="raise").astype(float)
    if not np.isfinite(parsed.to_numpy(dtype=float)).all():
        raise ValueError(f"{column} must contain only finite numeric values")
    if integer and not np.allclose(parsed, np.rint(parsed), rtol=0.0, atol=1e-10):
        raise ValueError(f"{column} must contain integer values")
    return pd.Series(np.rint(parsed) if integer else parsed, index=values.index)


def load_nfp_sources(
    source_dir: Path,
    thresholds: Iterable[int] = NFP_THRESHOLDS,
) -> pd.DataFrame:
    source_dir = Path(source_dir)
    frames: list[pd.DataFrame] = []
    for threshold in thresholds:
        threshold = int(threshold)
        path = source_dir / f"fig_nfp{threshold}_required_full_life_tests.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing canonical NFP source table: {path}")
        frame = pd.read_csv(path, low_memory=False)
        _require_columns(frame, REQUIRED_COLUMNS, path.name)
        numeric_threshold = _strict_numeric(frame["nfp_percent"], "nfp_percent", integer=False)
        if not np.allclose(numeric_threshold, np.rint(numeric_threshold), rtol=0.0, atol=1e-10):
            raise ValueError(f"{path.name} must contain integer NFP thresholds")
        observed = np.rint(numeric_threshold).astype(int)
        if not observed.eq(threshold).all():
            unique = sorted(observed.unique().tolist())
            raise ValueError(f"{path.name} contains NFP values {unique}, expected only {threshold}")
        frame["nfp_percent"] = observed
        frames.append(frame)
    if not frames:
        raise ValueError("At least one NFP threshold is required")
    return pd.concat(frames, ignore_index=True, sort=False)


def summarise_nfp_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, REQUIRED_COLUMNS, "NFP source data")
    data = frame.copy()
    data["reached"] = _as_bool(data["reached"], "reached")
    data["positive_attainable_gain"] = _as_bool(
        data["positive_attainable_gain"], "positive_attainable_gain"
    )
    for column in ("trial", "nfp_percent", "pool_n"):
        data[column] = _strict_numeric(data[column], column, integer=True)

    if bool(data["pool_n"].le(0).any()):
        raise ValueError("pool_n must be positive")
    unexpected_datasets = sorted(set(data["dataset"]).difference(DATASET_ORDER))
    unexpected_acquisitions = sorted(set(data["acquisition"]).difference(ACQUISITION_ORDER))
    unexpected_thresholds = sorted(set(data["nfp_percent"].astype(int)).difference(NFP_THRESHOLDS))
    if unexpected_datasets:
        raise ValueError(f"Unexpected datasets: {unexpected_datasets}")
    if unexpected_acquisitions:
        raise ValueError(f"Unexpected acquisition rules: {unexpected_acquisitions}")
    if unexpected_thresholds:
        raise ValueError(f"Unexpected NFP thresholds: {unexpected_thresholds}")
    expected_units = data["dataset"].map(DURATION_UNITS)
    unit_mismatch = ~data["duration_unit"].astype(str).eq(expected_units.astype(str))
    if bool(unit_mismatch.any()):
        preview = data.loc[unit_mismatch, ["dataset", "duration_unit"]].head().to_dict("records")
        raise ValueError(f"Dataset duration units are inconsistent: {preview}")

    # Canonical source tables leave performance indicators blank when an NFP
    # target is not reached. Validate those indicators only after applying the
    # documented eligibility filter, while retaining strict validation of core
    # identifiers, booleans, pool sizes, categories, and units above.
    data = data.loc[data["reached"] & data["positive_attainable_gain"]].copy()
    if data.empty:
        raise ValueError("No positive, reached NFP observations are available")
    for column in (
        "required_pool_fraction_pct",
        "required_full_life_tests",
        "required_duration",
    ):
        data[column] = _strict_numeric(data[column], column)
        if bool(data[column].lt(0).any()):
            raise ValueError(f"{column} must be nonnegative")
    if bool(data["required_pool_fraction_pct"].gt(100.0 + 1e-9).any()):
        raise ValueError("required_pool_fraction_pct must not exceed 100")
    expected_fraction = data["required_full_life_tests"] / data["pool_n"] * 100.0
    fraction_match = np.isclose(
        data["required_pool_fraction_pct"],
        expected_fraction,
        rtol=1e-9,
        atol=1e-9,
    )
    if not bool(fraction_match.all()):
        raise ValueError("required pool fraction does not equal full-life tests / pool_n * 100")

    group_keys = ["dataset", "acquisition", "nfp_percent"]
    grouped = data.groupby(group_keys, dropna=False)
    if bool(grouped["pool_n"].nunique().gt(1).any()):
        raise ValueError("pool_n must be constant within dataset/acquisition/NFP groups")
    if bool(grouped["duration_unit"].nunique().gt(1).any()):
        raise ValueError("duration_unit must be constant within dataset/acquisition/NFP groups")

    data["feature_model"] = data["feature_set"].astype(str) + "+" + data["model"].astype(str)
    summary = (
        data.groupby(group_keys, dropna=False)
        .agg(
            pool_n=("pool_n", "first"),
            duration_unit=("duration_unit", "first"),
            n_observations=("required_full_life_tests", "size"),
            n_trials=("trial", "nunique"),
            n_feature_model_combinations=("feature_model", "nunique"),
            required_full_life_tests_median=("required_full_life_tests", "median"),
            required_full_life_tests_q25=(
                "required_full_life_tests",
                lambda values: values.quantile(0.25),
            ),
            required_full_life_tests_q75=(
                "required_full_life_tests",
                lambda values: values.quantile(0.75),
            ),
            required_pool_fraction_pct_median=("required_pool_fraction_pct", "median"),
            required_pool_fraction_pct_q25=(
                "required_pool_fraction_pct",
                lambda values: values.quantile(0.25),
            ),
            required_pool_fraction_pct_q75=(
                "required_pool_fraction_pct",
                lambda values: values.quantile(0.75),
            ),
            required_duration_median=("required_duration", "median"),
            required_duration_q25=("required_duration", lambda values: values.quantile(0.25)),
            required_duration_q75=("required_duration", lambda values: values.quantile(0.75)),
        )
        .reset_index()
    )
    dataset_rank = {value: index for index, value in enumerate(DATASET_ORDER)}
    acquisition_rank = {value: index for index, value in enumerate(ACQUISITION_ORDER)}
    summary["_dataset_rank"] = summary["dataset"].map(dataset_rank).fillna(len(dataset_rank))
    summary["_acquisition_rank"] = summary["acquisition"].map(acquisition_rank).fillna(
        len(acquisition_rank)
    )
    summary = summary.sort_values(
        ["_dataset_rank", "_acquisition_rank", "nfp_percent"]
    ).drop(columns=["_dataset_rank", "_acquisition_rank"])
    summary["nfp_percent"] = summary["nfp_percent"].astype(int)
    summary["pool_n"] = summary["pool_n"].astype(int)
    return summary.reset_index(drop=True)


def summarise_nfp_trial_distributions(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, REQUIRED_COLUMNS, "NFP source data")
    data = frame.copy()
    data["reached"] = _as_bool(data["reached"], "reached")
    data["positive_attainable_gain"] = _as_bool(
        data["positive_attainable_gain"], "positive_attainable_gain"
    )
    for column in ("trial", "nfp_percent"):
        data[column] = _strict_numeric(data[column], column, integer=True)
    data = data.loc[data["reached"] & data["positive_attainable_gain"]].copy()
    if data.empty:
        raise ValueError("No positive, reached NFP observations are available")
    for column in ("required_duration", "required_pool_fraction_pct"):
        data[column] = _strict_numeric(data[column], column)
    if bool(data[["required_duration", "required_pool_fraction_pct"]].lt(0).any().any()):
        raise ValueError("Trial-level NFP indicators must be nonnegative")
    if bool(data["required_pool_fraction_pct"].gt(100.0 + 1e-9).any()):
        raise ValueError("required_pool_fraction_pct must not exceed 100")

    unexpected_datasets = sorted(set(data["dataset"]).difference(DATASET_ORDER))
    unexpected_acquisitions = sorted(set(data["acquisition"]).difference(ACQUISITION_ORDER))
    unexpected_thresholds = sorted(set(data["nfp_percent"].astype(int)).difference(NFP_THRESHOLDS))
    unexpected_trials = sorted(set(data["trial"].astype(int)).difference(range(1, 11)))
    if unexpected_datasets:
        raise ValueError(f"Unexpected datasets: {unexpected_datasets}")
    if unexpected_acquisitions:
        raise ValueError(f"Unexpected acquisition rules: {unexpected_acquisitions}")
    if unexpected_thresholds:
        raise ValueError(f"Unexpected NFP thresholds: {unexpected_thresholds}")
    if unexpected_trials:
        raise ValueError(f"Unexpected trial identifiers: {unexpected_trials}")
    expected_units = data["dataset"].map(DURATION_UNITS)
    if not bool(data["duration_unit"].astype(str).eq(expected_units.astype(str)).all()):
        raise ValueError("NFP source data contains non-canonical dataset duration units")

    observation_keys = [
        "dataset",
        "feature_set",
        "model",
        "trial",
        "acquisition",
        "nfp_percent",
    ]
    if bool(data.duplicated(observation_keys).any()):
        raise ValueError("NFP source data contains duplicate configuration observations")

    group_keys = ["dataset", "acquisition", "nfp_percent", "trial"]
    trial_summary = (
        data.groupby(group_keys, as_index=False, sort=False)
        .agg(
            duration_unit=("duration_unit", "first"),
            n_configurations=("required_duration", "size"),
            required_duration=("required_duration", "median"),
            required_pool_fraction_pct=("required_pool_fraction_pct", "median"),
        )
    )
    expected_groups = {
        (dataset, acquisition, threshold, trial)
        for dataset in DATASET_ORDER
        for acquisition in ACQUISITION_ORDER
        for threshold in NFP_THRESHOLDS
        for trial in range(1, 11)
    }
    observed_groups = set(
        zip(
            trial_summary["dataset"],
            trial_summary["acquisition"],
            trial_summary["nfp_percent"].astype(int),
            trial_summary["trial"].astype(int),
        )
    )
    missing = sorted(expected_groups.difference(observed_groups))
    extra = sorted(observed_groups.difference(expected_groups))
    if missing or extra or len(trial_summary) != len(expected_groups):
        raise ValueError(
            "Trial-level NFP grid must contain exactly 10 paired trials per "
            "dataset, acquisition, and threshold; "
            f"missing={missing[:5]}; extra={extra[:5]}"
        )

    dataset_rank = {value: index for index, value in enumerate(DATASET_ORDER)}
    acquisition_rank = {value: index for index, value in enumerate(ACQUISITION_ORDER)}
    threshold_rank = {value: index for index, value in enumerate(NFP_THRESHOLDS)}
    trial_summary["_dataset_rank"] = trial_summary["dataset"].map(dataset_rank)
    trial_summary["_acquisition_rank"] = trial_summary["acquisition"].map(acquisition_rank)
    trial_summary["_threshold_rank"] = trial_summary["nfp_percent"].map(threshold_rank)
    trial_summary = trial_summary.sort_values(
        ["_dataset_rank", "_acquisition_rank", "_threshold_rank", "trial"]
    ).drop(columns=["_dataset_rank", "_acquisition_rank", "_threshold_rank"])
    for column in ("nfp_percent", "trial", "n_configurations"):
        trial_summary[column] = trial_summary[column].astype(int)
    return trial_summary.reset_index(drop=True)


def tests_to_pool_fraction(values: np.ndarray | float, pool_n: float) -> np.ndarray | float:
    if float(pool_n) <= 0:
        raise ValueError("pool_n must be positive")
    return np.asarray(values, dtype=float) / float(pool_n) * 100.0


def pool_fraction_to_tests(values: np.ndarray | float, pool_n: float) -> np.ndarray | float:
    if float(pool_n) <= 0:
        raise ValueError("pool_n must be positive")
    return np.asarray(values, dtype=float) / 100.0 * float(pool_n)


def build_panel_a_relative_alc_summary(
    curve_points: pd.DataFrame,
    full_pool_trials: pd.DataFrame,
    *,
    feature_order: Iterable[str] = FEATURE_ORDER,
    model_order: Iterable[str] = MODEL_ORDER,
    acquisition_order: Iterable[str] = ACQUISITION_ORDER,
    target_fraction: float = PANEL_A_TARGET_FRACTION,
    required_trials: int = PANEL_A_REQUIRED_TRIALS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select model-specific best FSs and best/worst rules at a common budget.

    Feature sets are selected independently within every dataset-model context
    by the lowest mean full-pool MAPE across the required trials. Relative ALC
    is then integrated from the first post-initialisation checkpoint through
    the observed checkpoint nearest ``target_fraction``. Each non-random rule
    is compared with Random from the same dataset, model, feature set, and
    trial before rule-level trial medians are ranked.
    """

    curve_required = {
        "dataset",
        "feature_set",
        "model",
        "trial",
        "acquisition",
        "labelled_n",
        "pool_n",
        "holdout_mape",
    }
    full_pool_required = {
        "dataset",
        "feature_set",
        "model",
        "trial",
        "full_pool_mape",
    }
    _require_columns(curve_points, curve_required, "panel-a learning curves")
    _require_columns(full_pool_trials, full_pool_required, "panel-a full-pool trials")
    if not 0.0 < float(target_fraction) < 1.0:
        raise ValueError("target_fraction must lie strictly between 0 and 1")
    if int(required_trials) <= 0:
        raise ValueError("required_trials must be positive")

    feature_order = tuple(str(value) for value in feature_order)
    model_order = tuple(str(value) for value in model_order)
    acquisition_order = tuple(str(value) for value in acquisition_order)
    if "random_selection" not in acquisition_order:
        raise ValueError("Random must be present as the relative-ALC baseline")
    nonrandom_order = tuple(
        acquisition for acquisition in acquisition_order if acquisition != "random_selection"
    )
    if len(nonrandom_order) < 2:
        raise ValueError("At least two non-random acquisition rules are required")

    curves = curve_points.copy()
    full_pool = full_pool_trials.copy()
    for frame, numeric_columns in (
        (curves, ("trial", "labelled_n", "pool_n", "holdout_mape")),
        (full_pool, ("trial", "full_pool_mape")),
    ):
        for column in numeric_columns:
            frame[column] = _strict_numeric(
                frame[column],
                column,
                integer=column in {"trial", "labelled_n", "pool_n"},
            )
    if bool(curves["pool_n"].le(0).any()):
        raise ValueError("panel-a pool_n must be positive")
    if bool(curves["holdout_mape"].lt(0).any()):
        raise ValueError("panel-a holdout_mape must be nonnegative")
    if bool(full_pool["full_pool_mape"].lt(0).any()):
        raise ValueError("panel-a full_pool_mape must be nonnegative")

    curves = curves.loc[
        curves["feature_set"].isin(feature_order)
        & curves["model"].isin(model_order)
        & curves["acquisition"].isin(acquisition_order)
    ].copy()
    full_pool = full_pool.loc[
        full_pool["feature_set"].isin(feature_order)
        & full_pool["model"].isin(model_order)
    ].copy()
    if curves.empty or full_pool.empty:
        raise ValueError("panel-a source tables contain no eligible primary-model rows")

    full_pool_trial = (
        full_pool.groupby(
            ["dataset", "model", "feature_set", "trial"],
            as_index=False,
            sort=False,
        )["full_pool_mape"]
        .mean()
    )
    trial_counts = full_pool_trial.groupby(
        ["dataset", "model", "feature_set"], sort=False
    )["trial"].nunique()
    invalid_trial_counts = trial_counts.loc[trial_counts.ne(int(required_trials))]
    if not invalid_trial_counts.empty:
        raise ValueError(
            f"Every panel-a full-pool feature candidate requires {required_trials} trials; "
            f"observed={invalid_trial_counts.head().to_dict()}"
        )
    observed_candidates = full_pool_trial.groupby(
        ["dataset", "model"], sort=False
    )["feature_set"].agg(lambda values: set(values.astype(str)))
    missing_candidates = {
        context: sorted(set(feature_order).difference(values))
        for context, values in observed_candidates.items()
        if set(feature_order).difference(values)
    }
    if missing_candidates:
        raise ValueError(
            "Every panel-a dataset-model context requires all candidate feature sets; "
            f"examples={dict(list(missing_candidates.items())[:3])}"
        )

    feature_summary = (
        full_pool_trial.groupby(
            ["dataset", "model", "feature_set"],
            as_index=False,
            sort=False,
        )
        .agg(
            mean_full_pool_mape=("full_pool_mape", "mean"),
            n_feature_trials=("trial", "nunique"),
        )
    )
    feature_rank = {feature: index for index, feature in enumerate(feature_order)}
    feature_summary["_feature_rank"] = feature_summary["feature_set"].map(feature_rank)
    best_features = (
        feature_summary.sort_values(
            ["dataset", "model", "mean_full_pool_mape", "_feature_rank"],
            kind="mergesort",
        )
        .groupby(["dataset", "model"], as_index=False, sort=False)
        .first()
        .drop(columns="_feature_rank")
    )

    selected_curves = curves.merge(
        best_features[["dataset", "model", "feature_set"]],
        on=["dataset", "model", "feature_set"],
        how="inner",
        validate="many_to_one",
    )
    expected_contexts = set(
        zip(best_features["dataset"], best_features["model"], strict=True)
    )
    observed_contexts = set(
        zip(selected_curves["dataset"], selected_curves["model"], strict=True)
    )
    if observed_contexts != expected_contexts:
        missing = sorted(expected_contexts.difference(observed_contexts))
        raise ValueError(f"Best-FS learning curves are missing contexts: {missing[:5]}")

    target_by_dataset: dict[str, dict[str, float | int]] = {}
    sequence_columns = ["dataset", "model", "trial", "acquisition"]
    for dataset, dataset_frame in selected_curves.groupby("dataset", sort=False):
        if dataset_frame["pool_n"].nunique() != 1:
            raise ValueError(f"panel-a pool_n must be common within dataset={dataset}")
        pool_n = int(dataset_frame["pool_n"].iloc[0])
        sequences = dataset_frame.groupby(sequence_columns[1:], sort=False)[
            "labelled_n"
        ].agg(lambda values: tuple(sorted(pd.unique(values.astype(int)))))
        if sequences.nunique() != 1:
            raise ValueError(
                f"All panel-a trials and rules must share checkpoints for dataset={dataset}"
            )
        checkpoints = np.asarray(sequences.iloc[0], dtype=int)
        distances = np.abs(checkpoints / float(pool_n) - float(target_fraction))
        order = np.lexsort((checkpoints, distances))
        labelled_n = int(checkpoints[order[0]])
        target_by_dataset[str(dataset)] = {
            "pool_n": pool_n,
            "labelled_n": labelled_n,
            "achieved_labelled_fraction": labelled_n / float(pool_n),
        }

    trial_rows: list[dict[str, object]] = []
    group_columns = ["dataset", "feature_set", "model", "trial", "acquisition"]
    for group_key, group in selected_curves.groupby(group_columns, sort=False):
        dataset, feature_set, model, trial, acquisition = group_key
        target = target_by_dataset[str(dataset)]
        checkpoint = int(target["labelled_n"])
        ordered = (
            group.groupby("labelled_n", as_index=False, sort=True)
            .agg(pool_n=("pool_n", "first"), holdout_mape=("holdout_mape", "mean"))
            .sort_values("labelled_n")
        )
        initial_labelled_n = int(ordered["labelled_n"].min())
        window = ordered.loc[
            ordered["labelled_n"].gt(initial_labelled_n)
            & ordered["labelled_n"].le(checkpoint)
        ].copy()
        if window["labelled_n"].nunique() < 2:
            raise ValueError(
                "Relative ALC through the target checkpoint requires at least two "
                f"post-initialisation checkpoints for {group_columns}={group_key}"
            )
        x = window["labelled_n"].to_numpy(dtype=float) / float(target["pool_n"])
        y = window["holdout_mape"].to_numpy(dtype=float)
        span = float(x.max() - x.min())
        alc_mape = float(np.trapezoid(y, x) / span)
        trial_rows.append(
            {
                "dataset": str(dataset),
                "feature_set": str(feature_set),
                "model": str(model),
                "trial": int(trial),
                "acquisition": str(acquisition),
                "pool_n": int(target["pool_n"]),
                "labelled_n": checkpoint,
                "target_labelled_fraction": float(target_fraction),
                "achieved_labelled_fraction": float(
                    target["achieved_labelled_fraction"]
                ),
                "start_labelled_fraction": float(x.min()),
                "end_labelled_fraction": float(x.max()),
                "n_checkpoints": int(window["labelled_n"].nunique()),
                "alc_mape": alc_mape,
                "excluded_initial_checkpoint": True,
            }
        )
    absolute_trial = pd.DataFrame(trial_rows)
    trial_support = absolute_trial.groupby(
        ["dataset", "model", "feature_set", "acquisition"], sort=False
    )["trial"].nunique()
    invalid_support = trial_support.loc[trial_support.ne(int(required_trials))]
    if not invalid_support.empty:
        raise ValueError(
            f"Every panel-a rule requires {required_trials} trials; "
            f"observed={invalid_support.head().to_dict()}"
        )

    baseline_keys = ["dataset", "feature_set", "model", "trial"]
    baseline = absolute_trial.loc[
        absolute_trial["acquisition"].eq("random_selection"),
        baseline_keys + ["alc_mape"],
    ].rename(columns={"alc_mape": "baseline_alc_mape"})
    if bool(baseline.duplicated(baseline_keys).any()):
        raise ValueError("Random relative-ALC baseline must be unique per trial context")
    relative_trial = absolute_trial.merge(
        baseline,
        on=baseline_keys,
        how="left",
        validate="many_to_one",
    )
    if relative_trial["baseline_alc_mape"].isna().any() or bool(
        relative_trial["baseline_alc_mape"].le(0).any()
    ):
        raise ValueError("Every panel-a trial context requires a positive Random ALC")
    relative_trial["baseline_acquisition"] = "random_selection"
    relative_trial["delta_alc_mape"] = (
        relative_trial["baseline_alc_mape"] - relative_trial["alc_mape"]
    )
    relative_trial["relative_alc_improvement_pct"] = (
        relative_trial["delta_alc_mape"]
        / relative_trial["baseline_alc_mape"]
        * 100.0
    )

    nonrandom = relative_trial.loc[
        relative_trial["acquisition"].isin(nonrandom_order)
    ].copy()
    rule_summary = (
        nonrandom.groupby(
            ["dataset", "feature_set", "model", "acquisition"],
            as_index=False,
            sort=False,
        )
        .agg(
            median_relative_alc_improvement_pct=(
                "relative_alc_improvement_pct",
                "median",
            ),
            q25_relative_alc_improvement_pct=(
                "relative_alc_improvement_pct",
                lambda values: values.quantile(0.25),
            ),
            q75_relative_alc_improvement_pct=(
                "relative_alc_improvement_pct",
                lambda values: values.quantile(0.75),
            ),
            n_trials=("trial", "nunique"),
            pool_n=("pool_n", "first"),
            labelled_n=("labelled_n", "first"),
            achieved_labelled_fraction=("achieved_labelled_fraction", "first"),
            target_labelled_fraction=("target_labelled_fraction", "first"),
        )
    )
    acquisition_rank = {
        acquisition: index for index, acquisition in enumerate(nonrandom_order)
    }
    rule_summary["_acquisition_rank"] = rule_summary["acquisition"].map(
        acquisition_rank
    )
    selected_rows: list[pd.Series] = []
    for _context, context_frame in rule_summary.groupby(
        ["dataset", "model", "feature_set"], sort=False
    ):
        best = context_frame.sort_values(
            ["median_relative_alc_improvement_pct", "_acquisition_rank"],
            ascending=[False, True],
            kind="mergesort",
        ).iloc[0].copy()
        worst = context_frame.sort_values(
            ["median_relative_alc_improvement_pct", "_acquisition_rank"],
            ascending=[True, False],
            kind="mergesort",
        ).iloc[0].copy()
        best["role"] = "best"
        worst["role"] = "worst"
        selected_rows.extend([best, worst])
    selected = pd.DataFrame(selected_rows).drop(columns="_acquisition_rank")
    selected = selected.merge(
        best_features[
            ["dataset", "model", "feature_set", "mean_full_pool_mape"]
        ],
        on=["dataset", "model", "feature_set"],
        how="left",
        validate="many_to_one",
    )
    dataset_rank = {dataset: index for index, dataset in enumerate(DATASET_ORDER)}
    model_rank = {model: index for index, model in enumerate(model_order)}
    role_rank = {"best": 0, "worst": 1}
    selected["_dataset_rank"] = selected["dataset"].map(dataset_rank).fillna(
        len(dataset_rank)
    )
    selected["_model_rank"] = selected["model"].map(model_rank)
    selected["_role_rank"] = selected["role"].map(role_rank)
    selected = selected.sort_values(
        ["_dataset_rank", "_model_rank", "_role_rank"], kind="mergesort"
    ).drop(columns=["_dataset_rank", "_model_rank", "_role_rank"])
    relative_trial = relative_trial.sort_values(
        ["dataset", "model", "feature_set", "acquisition", "trial"],
        kind="mergesort",
    )
    return selected.reset_index(drop=True), relative_trial.reset_index(drop=True)


def _set_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "Liberation Sans",
                "sans-serif",
            ],
            "svg.fonttype": "none",
            "svg.hashsalt": "section4-nfp-summary-v1",
            "pdf.fonttype": 42,
            "font.size": 9.0,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.4,
            "axes.linewidth": AXIS_LINEWIDTH,
            "xtick.major.width": 0.4,
            "ytick.major.width": 0.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def _interval_error(median: float, q25: float, q75: float) -> np.ndarray:
    return np.asarray([[max(median - q25, 0.0)], [max(q75 - median, 0.0)]], dtype=float)


def _add_relative_duration_index(summary: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset", "acquisition", "nfp_percent", "required_duration_median"}
    _require_columns(summary, required, "NFP summary data")
    data = summary.copy()
    baseline = data.loc[
        data["acquisition"].eq("random_selection"),
        ["dataset", "nfp_percent", "required_duration_median"],
    ].copy()
    key_columns = ["dataset", "nfp_percent"]
    if bool(baseline.duplicated(key_columns).any()):
        raise ValueError("Random-selection duration baseline must be unique per dataset/NFP")
    baseline = baseline.set_index(key_columns)["required_duration_median"]
    keys = pd.MultiIndex.from_frame(data[key_columns])
    random_duration = baseline.reindex(keys).to_numpy(dtype=float)
    if not np.isfinite(random_duration).all() or bool(np.less_equal(random_duration, 0.0).any()):
        raise ValueError("Every dataset/NFP group requires a positive random-selection baseline")
    data["random_duration_median"] = random_duration
    data["relative_duration_pct"] = (
        data["required_duration_median"].to_numpy(dtype=float) / random_duration * 100.0
    )
    if not np.isfinite(data["relative_duration_pct"].to_numpy(dtype=float)).all():
        raise ValueError("Relative duration index must be finite")
    return data


def _add_panel_label(ax: mpl.axes.Axes, label: str, *, x: float = -0.12) -> None:
    ax.text(
        x,
        1.035,
        label,
        transform=ax.transAxes,
        fontsize=10.0,
        fontweight="bold",
        ha="right",
        va="bottom",
    )


def _draw_tradeoff_panel(
    ax: mpl.axes.Axes,
    data: pd.DataFrame,
) -> tuple[
    dict[str, mpl.axes.Axes],
    dict[int, list[mpl.collections.PathCollection]],
    list[mpl.collections.PathCollection],
]:
    indexed = _add_relative_duration_index(data)
    grouped = (
        indexed.groupby(["acquisition", "nfp_percent"], sort=False)
        .agg(
            pool_fraction_median=("required_pool_fraction_pct_median", "median"),
            pool_fraction_q25=(
                "required_pool_fraction_pct_median",
                lambda values: values.quantile(0.25),
            ),
            pool_fraction_q75=(
                "required_pool_fraction_pct_median",
                lambda values: values.quantile(0.75),
            ),
            relative_duration_median=("relative_duration_pct", "median"),
            relative_duration_q25=(
                "relative_duration_pct",
                lambda values: values.quantile(0.25),
            ),
            relative_duration_q75=(
                "relative_duration_pct",
                lambda values: values.quantile(0.75),
            ),
            full_life_tests_median=("required_full_life_tests_median", "median"),
        )
        .reset_index()
    )
    threshold_artists: dict[int, list[mpl.collections.PathCollection]] = {
        threshold: [] for threshold in NFP_THRESHOLDS
    }
    random_artists: list[mpl.collections.PathCollection] = []
    nonrandom_rules = ACQUISITION_ORDER[1:]
    subaxes: dict[str, mpl.axes.Axes] = {}
    subpanel_bounds = (
        (0.095, 0.555, 0.270, 0.300),
        (0.405, 0.555, 0.270, 0.300),
        (0.715, 0.555, 0.270, 0.300),
        (0.095, 0.125, 0.270, 0.300),
        (0.405, 0.125, 0.270, 0.300),
        (0.715, 0.125, 0.270, 0.300),
    )
    x_min = float(grouped["pool_fraction_q25"].min())
    x_max = float(grouped["pool_fraction_q75"].max())
    y_min = min(float(grouped["relative_duration_q25"].min()), 100.0)
    y_max = max(float(grouped["relative_duration_q75"].max()), 100.0)
    x_limits = (max(0.0, x_min - 4.0), min(100.0, x_max + 4.0))
    y_limits = (max(0.0, y_min - 8.0), y_max + 8.0)
    random_trajectory = grouped.loc[
        grouped["acquisition"].eq("random_selection")
    ].sort_values("nfp_percent")

    for index, (acquisition, bounds) in enumerate(zip(nonrandom_rules, subpanel_bounds)):
        subax = ax.inset_axes(bounds)
        subaxes[acquisition] = subax
        trajectory = grouped.loc[grouped["acquisition"].eq(acquisition)].sort_values(
            "nfp_percent"
        )
        color = ACQUISITION_COLORS[acquisition]
        subax.plot(
            random_trajectory["pool_fraction_median"],
            random_trajectory["relative_duration_median"],
            color=ACQUISITION_COLORS["random_selection"],
            linewidth=0.75,
            linestyle="--",
            alpha=0.65,
            zorder=2,
        )
        subax.plot(
            trajectory["pool_fraction_median"],
            trajectory["relative_duration_median"],
            color=color,
            linewidth=0.95,
            alpha=MARKER_ALPHA,
            zorder=2,
        )
        for random_row, row in zip(
            random_trajectory.itertuples(index=False),
            trajectory.itertuples(index=False),
        ):
            threshold = int(row.nfp_percent)
            is_primary = threshold == PRIMARY_NFP_THRESHOLD
            subax.errorbar(
                float(row.pool_fraction_median),
                float(row.relative_duration_median),
                xerr=_interval_error(
                    float(row.pool_fraction_median),
                    float(row.pool_fraction_q25),
                    float(row.pool_fraction_q75),
                ),
                yerr=_interval_error(
                    float(row.relative_duration_median),
                    float(row.relative_duration_q25),
                    float(row.relative_duration_q75),
                ),
                fmt="none",
                ecolor=mpl.colors.to_rgba(color, 0.38 if is_primary else 0.18),
                elinewidth=0.75 if is_primary else 0.5,
                capsize=1.8 if is_primary else 1.3,
                capthick=0.65 if is_primary else 0.45,
                zorder=1,
            )
            subax.errorbar(
                float(random_row.pool_fraction_median),
                float(random_row.relative_duration_median),
                xerr=_interval_error(
                    float(random_row.pool_fraction_median),
                    float(random_row.pool_fraction_q25),
                    float(random_row.pool_fraction_q75),
                ),
                fmt="none",
                ecolor=mpl.colors.to_rgba("#777777", 0.30 if is_primary else 0.16),
                elinewidth=0.65 if is_primary else 0.45,
                capsize=1.6 if is_primary else 1.1,
                capthick=0.55 if is_primary else 0.40,
                zorder=1,
            )
            tests = float(row.full_life_tests_median)
            marker_area = (
                18.0 + 1.05 * tests if is_primary else 7.0 + 0.42 * tests
            )
            scatter = subax.scatter(
                float(row.pool_fraction_median),
                float(row.relative_duration_median),
                s=marker_area,
                marker=MARKER_BY_ACQUISITION[acquisition],
                facecolor=color,
                edgecolor="none",
                linewidth=0.0,
                alpha=MARKER_ALPHA,
                zorder=4 if is_primary else 3,
            )
            scatter.set_gid(f"hero-{acquisition}-nfp{threshold}")
            threshold_artists[threshold].append(scatter)
            random_marker_area = 30.0 if is_primary else 14.0
            random_scatter = subax.scatter(
                float(random_row.pool_fraction_median),
                float(random_row.relative_duration_median),
                s=random_marker_area,
                marker=MARKER_BY_ACQUISITION["random_selection"],
                facecolor=ACQUISITION_COLORS["random_selection"],
                edgecolor="none",
                linewidth=0.0,
                alpha=MARKER_ALPHA,
                zorder=4 if is_primary else 3,
            )
            random_scatter.set_gid(
                f"hero-random_selection-vs-{acquisition}-nfp{threshold}"
            )
            random_artists.append(random_scatter)

        rule_label = subax.text(
            0.500,
            1.025,
            ACQUISITION_LABELS[acquisition],
            transform=subax.transAxes,
            color=color,
            fontsize=7.1,
            fontweight="bold",
            ha="center",
            va="bottom",
            clip_on=False,
        )
        rule_label.set_gid(f"panel-a-rule-label-{acquisition}")
        subax.set_xlim(*x_limits)
        subax.set_ylim(*y_limits)
        subax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
        subax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        subax.grid(axis="y", color="#E7E7E7", linewidth=0.30)
        subax.set_axisbelow(True)
        row, column = divmod(index, 3)
        subax.tick_params(
            axis="x",
            labelbottom=row == 1,
            labelsize=7.1,
            pad=1.0,
            length=2.2,
            width=AXIS_LINEWIDTH,
        )
        subax.tick_params(
            axis="y",
            labelleft=column == 0,
            labelsize=7.1,
            pad=1.0,
            length=2.2,
            width=AXIS_LINEWIDTH,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.patch.set_alpha(0.0)
    ax.set_xlabel("Required pool fraction (%)")
    ax.set_ylabel("Relative total duration\n(% of random)")
    ax.xaxis.set_label_coords(0.54, 0.025)
    ax.yaxis.set_label_coords(0.020, 0.49)
    return subaxes, threshold_artists, random_artists


def _draw_heatmap_panel(
    ax: mpl.axes.Axes,
    data: pd.DataFrame,
) -> tuple[mpl.colorbar.Colorbar, None]:
    heat = (
        data.loc[data["nfp_percent"].eq(95)]
        .pivot(index="dataset", columns="acquisition", values="required_pool_fraction_pct_median")
        .reindex(index=DATASET_ORDER, columns=ACQUISITION_ORDER)
    )
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "nfp_pool_fraction",
        HEATMAP_CMAP_COLORS,
    )
    norm = mpl.colors.Normalize(vmin=HEATMAP_VALUE_RANGE[0], vmax=HEATMAP_VALUE_RANGE[1])
    image = ax.imshow(
        heat.to_numpy(dtype=float),
        cmap=cmap,
        norm=norm,
        aspect="auto",
    )
    for row_index in range(heat.shape[0]):
        for column_index in range(heat.shape[1]):
            value = float(heat.iat[row_index, column_index])
            ax.text(
                column_index,
                row_index,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=7.1,
                color="#222222",
                zorder=2,
            )
    ax.set_xticks(range(len(ACQUISITION_ORDER)))
    ax.set_xticklabels(
        [
            "Random",
            "Diversity (One-shot)",
            "Diversity (Iterative)",
            "Coverage",
            "Exploration",
            "Exploitation",
            "Hybrid",
        ],
        rotation=35,
        ha="right",
        va="top",
        rotation_mode="anchor",
        fontsize=7.1,
    )
    ax.set_yticks(range(len(DATASET_ORDER)))
    ax.set_yticklabels([DATASET_LABELS[dataset] for dataset in DATASET_ORDER], fontsize=7.9)
    ax.tick_params(axis="x", length=0, pad=2.2)
    ax.tick_params(axis="y", length=0, pad=1.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = ax.figure.colorbar(
        image,
        ax=ax,
        orientation="vertical",
        fraction=0.075,
        pad=0.045,
        aspect=18,
    )
    colorbar.set_ticks(HEATMAP_COLORBAR_TICKS)
    colorbar.set_label("Required pool fraction (%)", labelpad=4.0)
    colorbar.ax.tick_params(
        axis="y",
        labelsize=7.1,
        length=2.0,
        width=0.4,
        pad=1.5,
    )
    colorbar.outline.set_visible(False)
    return colorbar, None


def _draw_tests_trend_panel(ax: mpl.axes.Axes, data: pd.DataFrame) -> None:
    trend = (
        data.groupby(["acquisition", "nfp_percent"], sort=False)[
            "required_full_life_tests_median"
        ]
        .median()
        .rename("median_tests")
        .reset_index()
    )
    for acquisition in ACQUISITION_ORDER:
        sub = trend.loc[trend["acquisition"].eq(acquisition)].sort_values("nfp_percent")
        color = ACQUISITION_COLORS[acquisition]
        ax.plot(
            sub["nfp_percent"],
            sub["median_tests"],
            color=color,
            linewidth=0.85,
            alpha=0.48,
        )
        for row in sub.itertuples(index=False):
            threshold = int(row.nfp_percent)
            is_primary = threshold == PRIMARY_NFP_THRESHOLD
            ax.scatter(
                threshold,
                float(row.median_tests),
                s=(
                    PANEL_C_MARKER_AREA_PRIMARY
                    if is_primary
                    else PANEL_C_MARKER_AREA_SENSITIVITY
                ),
                marker=MARKER_BY_ACQUISITION[acquisition],
                facecolor=color,
                edgecolor="none",
                linewidth=0.0,
                alpha=MARKER_ALPHA,
                zorder=4 if is_primary else 3,
            )
    ax.axvline(PRIMARY_NFP_THRESHOLD, color="#9AA9B6", linewidth=0.65, alpha=0.45)
    ax.set_xticks(NFP_THRESHOLDS)
    ax.set_xlabel("NFP target (%)")
    ax.set_ylabel("Required full-life tests")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    acquisition_handles = [
        Line2D(
            [0],
            [0],
            color=ACQUISITION_COLORS[acquisition],
            linewidth=0.85,
            marker=MARKER_BY_ACQUISITION[acquisition],
            markersize=4.0,
            markerfacecolor=ACQUISITION_COLORS[acquisition],
            markeredgecolor="none",
            alpha=MARKER_ALPHA,
            label=ACQUISITION_LABELS[acquisition],
        )
        for acquisition in ACQUISITION_ORDER
    ]
    acquisition_legend = ax.legend(
        handles=acquisition_handles,
        loc="center",
        ncol=4,
        title="Acquisition rule",
        fontsize=7.1,
        title_fontsize=7.1,
        handlelength=1.35,
        handletextpad=0.35,
        columnspacing=0.75,
        borderaxespad=0.0,
    )
    acquisition_legend.set_visible(False)


def _draw_win_count_panel(ax: mpl.axes.Axes, data: pd.DataFrame) -> None:
    primary = data.loc[data["nfp_percent"].eq(PRIMARY_NFP_THRESHOLD)].copy()
    minimum = primary.groupby("dataset")["required_duration_median"].transform("min")
    winners = primary.loc[
        np.isclose(
            primary["required_duration_median"].to_numpy(dtype=float),
            minimum.to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-9,
        )
    ]
    counts = winners.groupby("acquisition").size().reindex(ACQUISITION_ORDER, fill_value=0)
    rank = {acquisition: index for index, acquisition in enumerate(ACQUISITION_ORDER)}
    order = sorted(ACQUISITION_ORDER, key=lambda value: (-int(counts[value]), rank[value]))
    y_positions = np.arange(len(order))
    values = np.asarray([int(counts[value]) for value in order], dtype=float)
    bars = ax.barh(
        y_positions,
        values,
        color=[ACQUISITION_COLORS[value] for value in order],
        edgecolor="none",
        height=0.68,
        alpha=MARKER_ALPHA,
    )
    for bar, value in zip(bars, values):
        if value > 0:
            ax.text(
                value + 0.25,
                bar.get_y() + bar.get_height() / 2.0,
                f"{int(value)}",
                va="center",
                ha="left",
                fontsize=7.8,
            )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [ACQUISITION_LABELS[value].replace("Diversity (", "Diversity\n(") for value in order],
        fontsize=7.8,
        linespacing=0.85,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Fastest datasets at NFP target of 95%")
    ax.set_xlim(0.0, max(float(values.max()) * 1.18, 1.0))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))


def _duration_tick_label(value: float, _position: int) -> str:
    if abs(value) >= 1000.0:
        return f"{value / 1000.0:g}k"
    return f"{value:g}"


def _draw_duration_panel(
    ax: mpl.axes.Axes,
    data: pd.DataFrame,
    *,
    unit: str,
    title: str,
) -> None:
    datasets = [dataset for dataset in DATASET_ORDER if DURATION_UNITS[dataset] == unit]
    duration = (
        data.loc[data["dataset"].isin(datasets)]
        .groupby(["dataset", "nfp_percent"], sort=False)["required_duration_median"]
        .agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        )
        .reset_index()
    )
    base_y = np.arange(len(datasets), dtype=float)
    pivot = duration.pivot(index="dataset", columns="nfp_percent", values="median").reindex(
        datasets
    )
    for y_position, dataset in zip(base_y, datasets):
        ax.plot(
            pivot.loc[dataset, list(NFP_THRESHOLDS)].to_numpy(dtype=float),
            np.repeat(y_position, len(NFP_THRESHOLDS)),
            color=MEDIAN_CONNECTOR_COLOR,
            linewidth=0.65,
            alpha=0.65,
            zorder=1,
        )
    for threshold in (*SENSITIVITY_NFP_THRESHOLDS, PRIMARY_NFP_THRESHOLD):
        threshold_data = (
            duration.loc[duration["nfp_percent"].eq(threshold)]
            .set_index("dataset")
            .reindex(datasets)
        )
        for y_position, (dataset, row) in zip(base_y, threshold_data.iterrows()):
            median = float(row["median"])
            ax.errorbar(
                median,
                y_position,
                xerr=_interval_error(median, float(row["q25"]), float(row["q75"])),
                fmt=MARKER_BY_THRESHOLD[threshold],
                markersize=4.2 if threshold == PRIMARY_NFP_THRESHOLD else 2.8,
                markerfacecolor=NFP_COLORS[threshold],
                markeredgecolor="none",
                markeredgewidth=0.0,
                color=NFP_COLORS[threshold],
                ecolor=mpl.colors.to_rgba(
                    NFP_COLORS[threshold],
                    0.75 if threshold == PRIMARY_NFP_THRESHOLD else 0.40,
                ),
                elinewidth=0.8 if threshold == PRIMARY_NFP_THRESHOLD else 0.55,
                capsize=1.6 if threshold == PRIMARY_NFP_THRESHOLD else 1.1,
                alpha=MARKER_ALPHA,
                zorder=4 if threshold == PRIMARY_NFP_THRESHOLD else 3,
            )
    ax.set_yticks(base_y)
    ax.set_yticklabels([DATASET_LABELS[dataset] for dataset in datasets], fontsize=7.7)
    ax.invert_yaxis()
    if unit == "EFC":
        ax.set_ylim(1.7, -0.7)
    elif unit == "cycles":
        ax.set_ylim(5.25, -1.25)
    ax.set_xlim(left=0.0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="upper"))
    if unit in {"cycles", "EFC"}:
        ax.set_xlabel(rf"Total duration ($\times10^3$ {unit})")
        ax.xaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda value, _position: f"{value / 1000.0:g}")
        )
    else:
        ax.set_xlabel(f"Total duration ({unit})")
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(_duration_tick_label))
    ax.tick_params(axis="x", labelsize=7.7)
    ax.xaxis.get_offset_text().set_fontsize(7.7)


def _draw_half_violin(
    ax: mpl.axes.Axes,
    values: np.ndarray,
    *,
    position: float,
    side: str,
    color: str,
    width: float,
    gid: str,
) -> mpl.collections.PolyCollection:
    if side not in {"upper", "lower", "left", "right"}:
        raise ValueError(f"Unsupported half-violin side: {side}")
    orientation = "horizontal" if side in {"upper", "lower"} else "vertical"
    violin = ax.violinplot(
        [np.asarray(values, dtype=float)],
        positions=[position],
        orientation=orientation,
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=0.30,
        points=100,
    )
    body = violin["bodies"][0]
    vertices = body.get_paths()[0].vertices
    if side == "upper":
        vertices[:, 1] = np.maximum(vertices[:, 1], position)
    elif side == "lower":
        vertices[:, 1] = np.minimum(vertices[:, 1], position)
    elif side == "right":
        vertices[:, 0] = np.maximum(vertices[:, 0], position)
    else:
        vertices[:, 0] = np.minimum(vertices[:, 0], position)
    body.set_facecolor(color)
    body.set_edgecolor("none")
    body.set_linewidth(0.0)
    body.set_alpha(MARKER_ALPHA)
    body.set_gid(gid)
    return body


def _draw_duration_dataset_half_violin_panel(
    ax: mpl.axes.Axes,
    trial_data: pd.DataFrame,
    *,
    panel: str,
    unit: str,
) -> tuple[
    list[mpl.collections.PolyCollection],
    list[mpl.collections.PathCollection],
    list[mpl.lines.Line2D],
]:
    datasets = [dataset for dataset in DATASET_ORDER if DURATION_UNITS[dataset] == unit]
    base_x = np.arange(len(datasets), dtype=float)
    threshold_step = 0.34 if panel == "g" else 0.28
    threshold_offsets = {92: -threshold_step, 95: 0.0, 98: threshold_step}
    acquisition_offsets = dict(
        zip(ACQUISITION_ORDER, np.linspace(-0.12, -0.02, len(ACQUISITION_ORDER)))
    )
    bodies: list[mpl.collections.PolyCollection] = []
    point_artists: list[mpl.collections.PathCollection] = []
    median_connectors: list[mpl.lines.Line2D] = []
    expected_n = len(ACQUISITION_ORDER) * 10
    unit_data = trial_data.loc[trial_data["dataset"].isin(datasets)]
    for x_position, dataset in zip(base_x, datasets):
        dataset_data = unit_data.loc[unit_data["dataset"].eq(dataset)]
        threshold_values = {
            threshold: dataset_data.loc[
                dataset_data["nfp_percent"].eq(threshold),
                "required_duration",
            ].to_numpy(dtype=float)
            for threshold in NFP_THRESHOLDS
        }
        for threshold, values in threshold_values.items():
            if len(values) != expected_n:
                raise ValueError(
                    f"Panel {panel} requires {expected_n} values for "
                    f"{dataset} at NFP{threshold}; observed={len(values)}"
                )
        connector = ax.plot(
            [x_position + threshold_offsets[threshold] for threshold in NFP_THRESHOLDS],
            [float(np.median(threshold_values[threshold])) for threshold in NFP_THRESHOLDS],
            color=MEDIAN_CONNECTOR_COLOR,
            linewidth=0.65,
            alpha=0.65,
            zorder=1,
        )[0]
        connector.set_gid(f"threshold-median-connector-{panel}-{dataset}")
        median_connectors.append(connector)
        for threshold in NFP_THRESHOLDS:
            values = threshold_values[threshold]
            center = x_position + threshold_offsets[threshold]
            body = _draw_half_violin(
                ax,
                values,
                position=center,
                side="right",
                color=NFP_COLORS[threshold],
                width=0.24,
                gid=f"half-violin-{panel}-{dataset}-nfp{threshold}-x{center}",
            )
            bodies.append(body)
            for acquisition in ACQUISITION_ORDER:
                acquisition_values = dataset_data.loc[
                    dataset_data["acquisition"].eq(acquisition)
                    & dataset_data["nfp_percent"].eq(threshold),
                    "required_duration",
                ].to_numpy(dtype=float)
                if len(acquisition_values) != 10:
                    raise ValueError(
                        f"Panel {panel} requires 10 values for {dataset}, {acquisition}, "
                        f"NFP{threshold}; observed={len(acquisition_values)}"
                    )
                point_x = (
                    center
                    + acquisition_offsets[acquisition]
                    + np.linspace(-0.007, 0.007, len(acquisition_values))
                )
                points = ax.scatter(
                    point_x,
                    acquisition_values,
                    s=7.0,
                    marker=MARKER_BY_ACQUISITION[acquisition],
                    color=NFP_COLORS[threshold],
                    edgecolor="none",
                    linewidth=0.0,
                    alpha=MARKER_ALPHA,
                    zorder=4,
                )
                points.set_gid(
                    f"trial-points-{panel}-dataset-{dataset}-"
                    f"acquisition-{acquisition}-nfp{threshold}"
                )
                point_artists.append(points)
            q25, median, q75 = np.quantile(values, [0.25, 0.50, 0.75])
            ax.plot(
                [center, center],
                [q25, q75],
                color=NFP_COLORS[threshold],
                linewidth=1.0 if threshold == PRIMARY_NFP_THRESHOLD else 0.75,
                solid_capstyle="round",
                alpha=0.95,
                zorder=5,
            )
            ax.scatter(
                center,
                median,
                s=17 if threshold == PRIMARY_NFP_THRESHOLD else 11,
                marker=MARKER_BY_THRESHOLD[threshold],
                color=NFP_COLORS[threshold],
                edgecolor="none",
                linewidth=0.0,
                alpha=MARKER_ALPHA,
                zorder=6,
            )

    ax.set_xticks(base_x)
    dataset_tick_labels = [
        (
            DATASET_LABELS[dataset].replace("-", "-\n")
            if panel == "e"
            else DATASET_LABELS[dataset]
        )
        for dataset in datasets
    ]
    ax.set_xticklabels(
        dataset_tick_labels,
        rotation=0,
        ha="center",
        rotation_mode="anchor",
        fontsize=7.1,
    )
    x_margin = 0.70 if panel == "g" else 0.55
    ax.set_xlim(-x_margin, len(datasets) - 1.0 + x_margin)
    ax.set_xlabel("")
    unit_values = trial_data.loc[trial_data["dataset"].isin(datasets), "required_duration"]
    top = float(unit_values.max())
    ax.set_ylim(
        0.0,
        top * DURATION_Y_AXIS_UPPER_FACTOR if top > 0 else 1.0,
    )
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="upper"))
    if unit in {"cycles", "EFC"}:
        ax.set_ylabel("Total duration\n" + rf"($\times10^3$ {unit})")
        ax.yaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda value, _position: f"{value / 1000.0:g}")
        )
    else:
        ax.set_ylabel(f"Total duration\n({unit})")
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(_duration_tick_label))
    ax.tick_params(axis="y", labelsize=7.2)
    ax.yaxis.get_offset_text().set_fontsize(7.2)
    threshold_handles = [
        Line2D(
            [0],
            [0],
            marker=MARKER_BY_THRESHOLD[threshold],
            linestyle="none",
            markersize=4.2 if threshold == PRIMARY_NFP_THRESHOLD else 3.4,
            markerfacecolor=NFP_COLORS[threshold],
            markeredgecolor="none",
            alpha=MARKER_ALPHA,
            label=f"{threshold}%" + (" (primary)" if threshold == 95 else ""),
        )
        for threshold in NFP_THRESHOLDS
    ]
    threshold_legend = ax.legend(
        handles=threshold_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, VIOLIN_PANEL_LEGEND_BBOX_Y),
        ncol=3,
        title="NFP target",
        fontsize=7.1,
        title_fontsize=7.1,
        handletextpad=0.25,
        columnspacing=0.55,
        borderaxespad=0.2,
    )
    ax.add_artist(threshold_legend)
    return bodies, point_artists, median_connectors


def _draw_dataset_pool_fraction_half_violin_panel(
    ax: mpl.axes.Axes,
    trial_data: pd.DataFrame,
) -> tuple[
    list[mpl.collections.PolyCollection],
    list[mpl.collections.PathCollection],
    dict[str, list[mpl.artist.Artist]],
]:
    dataset_trials = (
        trial_data.groupby(["dataset", "nfp_percent", "trial"], as_index=False)[
            "required_pool_fraction_pct"
        ]
        .median()
    )
    group_sizes = dataset_trials.groupby(["dataset", "nfp_percent"]).size()
    if set(group_sizes) != {10}:
        raise ValueError("Panel h requires 10 paired trials per dataset and NFP threshold")
    threshold_offsets = PANEL_H_THRESHOLD_OFFSETS
    bodies: list[mpl.collections.PolyCollection] = []
    point_artists: list[mpl.collections.PathCollection] = []
    connectors: list[mpl.lines.Line2D] = []
    markers: list[mpl.collections.PathCollection] = []
    y_positions = np.arange(len(DATASET_ORDER), dtype=float)
    for y_position, dataset in zip(y_positions, DATASET_ORDER):
        dataset_data = dataset_trials.loc[dataset_trials["dataset"].eq(dataset)]
        summaries: dict[int, tuple[float, float, float]] = {}
        for threshold in NFP_THRESHOLDS:
            values = dataset_data.loc[
                dataset_data["nfp_percent"].eq(threshold),
                "required_pool_fraction_pct",
            ].to_numpy(dtype=float)
            q25, median, q75 = np.quantile(values, [0.25, 0.50, 0.75])
            summaries[threshold] = (float(q25), float(median), float(q75))
        centers = {
            threshold: y_position + threshold_offsets[threshold]
            for threshold in NFP_THRESHOLDS
        }
        connector = ax.plot(
            [summaries[threshold][1] for threshold in NFP_THRESHOLDS],
            [centers[threshold] for threshold in NFP_THRESHOLDS],
            color=MEDIAN_CONNECTOR_COLOR,
            linewidth=0.65,
            alpha=0.65,
            zorder=1,
        )[0]
        connector.set_gid(f"threshold-median-connector-h-{dataset}")
        connectors.append(connector)
        for threshold in NFP_THRESHOLDS:
            q25, median, q75 = summaries[threshold]
            center = centers[threshold]
            values = dataset_data.loc[
                dataset_data["nfp_percent"].eq(threshold),
                "required_pool_fraction_pct",
            ].to_numpy(dtype=float)
            body = _draw_half_violin(
                ax,
                values,
                position=center,
                side="upper",
                color=NFP_COLORS[threshold],
                width=0.18,
                gid=f"half-violin-h-{dataset}-nfp{threshold}-y{center}",
            )
            bodies.append(body)
            point_y = center - 0.070 + np.linspace(-0.012, 0.012, len(values))
            points = ax.scatter(
                values,
                point_y,
                s=7.0,
                marker=MARKER_BY_THRESHOLD[threshold],
                color=NFP_COLORS[threshold],
                edgecolor="none",
                linewidth=0.0,
                alpha=MARKER_ALPHA,
                zorder=4,
            )
            points.set_gid(f"trial-points-h-dataset-{dataset}-nfp{threshold}")
            point_artists.append(points)
            ax.plot(
                [q25, q75],
                [center, center],
                color=NFP_COLORS[threshold],
                linewidth=1.05 if threshold == PRIMARY_NFP_THRESHOLD else 0.65,
                alpha=0.85,
                solid_capstyle="round",
                zorder=2,
            )
            marker = ax.scatter(
                median,
                center,
                s=23 if threshold == PRIMARY_NFP_THRESHOLD else 13,
                marker=MARKER_BY_THRESHOLD[threshold],
                color=NFP_COLORS[threshold],
                edgecolor="none",
                linewidth=0.0,
                alpha=MARKER_ALPHA,
                zorder=4 if threshold == PRIMARY_NFP_THRESHOLD else 3,
            )
            marker.set_gid(f"threshold-marker-h-{dataset}-nfp{threshold}")
            markers.append(marker)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([DATASET_LABELS[dataset] for dataset in DATASET_ORDER], fontsize=7.9)
    ax.tick_params(axis="y", pad=1.0)
    ax.invert_yaxis()
    ax.set_ylim(*PANEL_H_Y_AXIS_LIMITS)
    all_values = dataset_trials["required_pool_fraction_pct"].to_numpy(dtype=float)
    data_min = float(all_values.min())
    data_max = float(all_values.max())
    margin = max((data_max - data_min) * 0.08, 1.0)
    ax.set_xlim(max(0.0, data_min - margin), min(100.0, data_max + margin))
    ax.set_xlabel("Required pool fraction (%)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    threshold_handles = [
        Line2D(
            [0],
            [0],
            marker=MARKER_BY_THRESHOLD[threshold],
            linestyle="none",
            markersize=4.5 if threshold == PRIMARY_NFP_THRESHOLD else 3.6,
            markerfacecolor=NFP_COLORS[threshold],
            markeredgecolor="none",
            alpha=MARKER_ALPHA,
            label=f"{threshold}%" + (" (primary)" if threshold == 95 else ""),
        )
        for threshold in NFP_THRESHOLDS
    ]
    ax.legend(
        handles=threshold_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, VIOLIN_PANEL_LEGEND_BBOX_Y),
        ncol=3,
        title="NFP target",
        fontsize=7.1,
        title_fontsize=7.1,
        handletextpad=0.25,
        columnspacing=0.55,
        borderaxespad=0.2,
    )
    return bodies, point_artists, {"connectors": connectors, "markers": markers}


def _draw_dataset_penalty_panel(ax: mpl.axes.Axes, data: pd.DataFrame) -> None:
    levels = (
        data.groupby(["dataset", "nfp_percent"])["required_pool_fraction_pct_median"]
        .median()
        .unstack("nfp_percent")
        .reindex(DATASET_ORDER)
    )
    sensitivity = pd.DataFrame(
        {
            92: levels[92] - levels[PRIMARY_NFP_THRESHOLD],
            98: levels[98] - levels[PRIMARY_NFP_THRESHOLD],
        }
    )
    y_positions = np.arange(len(DATASET_ORDER), dtype=float)
    for y_position, dataset in zip(y_positions, DATASET_ORDER):
        start = float(sensitivity.loc[dataset, 92])
        end = float(sensitivity.loc[dataset, 98])
        ax.plot([start, end], [y_position, y_position], color="#A9A9A9", linewidth=1.0)
        ax.scatter(
            start,
            y_position,
            s=24,
            marker=MARKER_BY_THRESHOLD[92],
            color=NFP_COLORS[92],
            edgecolor="none",
            linewidth=0.0,
            alpha=MARKER_ALPHA,
            zorder=3,
        )
        ax.scatter(
            end,
            y_position,
            s=28,
            marker=MARKER_BY_THRESHOLD[98],
            color=NFP_COLORS[98],
            edgecolor="none",
            linewidth=0.0,
            alpha=MARKER_ALPHA,
            zorder=3,
        )
    ax.set_yticks(y_positions)
    ax.set_yticklabels([DATASET_LABELS[dataset] for dataset in DATASET_ORDER], fontsize=7.9)
    ax.tick_params(axis="y", pad=1.0)
    ax.invert_yaxis()
    max_abs = max(float(np.abs(sensitivity.to_numpy(dtype=float)).max()), 1.0)
    ax.set_xlim(-max_abs * 1.18, max_abs * 1.18)
    ax.axvline(0.0, color="#275D8C", linestyle="--", linewidth=0.8, alpha=0.75)
    ax.set_xlabel("Variation relative to NFP target of 95%")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))


def _normalise_panel_a_summary(panel_a_summary: pd.DataFrame) -> pd.DataFrame:
    _require_columns(panel_a_summary, PANEL_A_REQUIRED_COLUMNS, "panel-a relative ALC summary")
    data = panel_a_summary.copy()
    for column in ("n_trials", "pool_n", "labelled_n"):
        data[column] = _strict_numeric(data[column], column, integer=True)
    for column in (
        "median_relative_alc_improvement_pct",
        "q25_relative_alc_improvement_pct",
        "q75_relative_alc_improvement_pct",
        "achieved_labelled_fraction",
        "target_labelled_fraction",
        "mean_full_pool_mape",
    ):
        data[column] = _strict_numeric(data[column], column)
    if not bool(data["n_trials"].eq(PANEL_A_REQUIRED_TRIALS).all()):
        raise ValueError(f"Every panel-a endpoint requires {PANEL_A_REQUIRED_TRIALS} trials")
    if bool(data["pool_n"].le(0).any()) or bool(data["labelled_n"].le(0).any()):
        raise ValueError("panel-a pool_n and labelled_n must be positive")
    if not bool(
        np.isclose(
            data["achieved_labelled_fraction"],
            data["labelled_n"] / data["pool_n"],
            rtol=0.0,
            atol=1e-12,
        ).all()
    ):
        raise ValueError("panel-a achieved fraction must equal labelled_n / pool_n")
    if not bool(
        np.isclose(
            data["target_labelled_fraction"],
            PANEL_A_TARGET_FRACTION,
            rtol=0.0,
            atol=1e-12,
        ).all()
    ):
        raise ValueError("panel-a target fraction must be 0.30")
    ordered = (
        data["q25_relative_alc_improvement_pct"]
        .le(data["median_relative_alc_improvement_pct"])
        & data["median_relative_alc_improvement_pct"].le(
            data["q75_relative_alc_improvement_pct"]
        )
    )
    if not bool(ordered.all()):
        raise ValueError("panel-a relative ALC quantiles must satisfy Q25 <= median <= Q75")
    if set(data["dataset"]) != set(DATASET_ORDER):
        raise ValueError("panel-a summary must contain all eight datasets")
    if set(data["model"]) != set(MODEL_ORDER):
        raise ValueError("panel-a summary must contain all three primary models")
    if not set(data["feature_set"]).issubset(FEATURE_ORDER):
        raise ValueError("panel-a summary contains an unknown feature set")
    if not set(data["acquisition"]).issubset(set(ACQUISITION_ORDER[1:])):
        raise ValueError("panel-a best/worst endpoints must be non-random rules")
    expected_groups = {
        (dataset, model, role)
        for dataset in DATASET_ORDER
        for model in MODEL_ORDER
        for role in ("best", "worst")
    }
    observed_groups = set(zip(data["dataset"], data["model"], data["role"], strict=True))
    if observed_groups != expected_groups or len(data) != len(expected_groups):
        raise ValueError("panel-a summary requires one best and one worst endpoint per dataset-model")
    context_groups = data.groupby(["dataset", "model"], sort=False)
    if bool(context_groups["feature_set"].nunique().ne(1).any()):
        raise ValueError("panel-a best and worst endpoints must share the selected feature set")
    if bool(context_groups["labelled_n"].nunique().ne(1).any()):
        raise ValueError("panel-a endpoints must share the target checkpoint within each context")
    pivot = data.pivot(
        index=["dataset", "model"],
        columns="role",
        values="median_relative_alc_improvement_pct",
    )
    if bool(pivot["best"].lt(pivot["worst"] - 1e-12).any()):
        raise ValueError("panel-a best-rule median must not be below the worst-rule median")
    dataset_rank = {dataset: index for index, dataset in enumerate(DATASET_ORDER)}
    model_rank = {model: index for index, model in enumerate(MODEL_ORDER)}
    role_rank = {"best": 0, "worst": 1}
    data["_dataset_rank"] = data["dataset"].map(dataset_rank)
    data["_model_rank"] = data["model"].map(model_rank)
    data["_role_rank"] = data["role"].map(role_rank)
    return data.sort_values(
        ["_model_rank", "_dataset_rank", "_role_rank"], kind="mergesort"
    ).drop(columns=["_dataset_rank", "_model_rank", "_role_rank"])


def _draw_relative_alc_panel(
    ax: mpl.axes.Axes,
    panel_a_summary: pd.DataFrame,
) -> tuple[list[mpl.lines.Line2D], list[mpl.collections.PathCollection]]:
    data = _normalise_panel_a_summary(panel_a_summary)
    dataset_offsets = dict(
        zip(DATASET_ORDER, np.linspace(-0.23, 0.23, len(DATASET_ORDER)), strict=True)
    )
    model_positions = {model: float(index) for index, model in enumerate(MODEL_ORDER)}
    pair_lines: list[mpl.lines.Line2D] = []
    endpoint_artists: list[mpl.collections.PathCollection] = []
    for model in MODEL_ORDER:
        for dataset in DATASET_ORDER:
            pair = data.loc[data["model"].eq(model) & data["dataset"].eq(dataset)]
            best = pair.loc[pair["role"].eq("best")].iloc[0]
            worst = pair.loc[pair["role"].eq("worst")].iloc[0]
            x_position = model_positions[model] + dataset_offsets[dataset]
            line = ax.plot(
                [x_position, x_position],
                [
                    float(worst["median_relative_alc_improvement_pct"]),
                    float(best["median_relative_alc_improvement_pct"]),
                ],
                color="#A9A9A9",
                linewidth=0.65,
                alpha=0.52,
                zorder=1,
            )[0]
            line.set_gid(f"relative-alc-pair-{dataset}-{model}")
            pair_lines.append(line)
            for row in (worst, best):
                acquisition = str(row["acquisition"])
                median = float(row["median_relative_alc_improvement_pct"])
                ax.errorbar(
                    x_position,
                    median,
                    yerr=_interval_error(
                        median,
                        float(row["q25_relative_alc_improvement_pct"]),
                        float(row["q75_relative_alc_improvement_pct"]),
                    ),
                    fmt="none",
                    ecolor=mpl.colors.to_rgba(ACQUISITION_COLORS[acquisition], 0.30),
                    elinewidth=0.55,
                    capsize=1.25,
                    capthick=0.45,
                    zorder=2,
                )
                endpoint = ax.scatter(
                    x_position,
                    median,
                    s=27.0,
                    marker=MARKER_BY_ACQUISITION[acquisition],
                    facecolor=ACQUISITION_COLORS[acquisition],
                    edgecolor="none",
                    linewidth=0.0,
                    alpha=MARKER_ALPHA,
                    zorder=3,
                )
                endpoint.set_gid(
                    f"relative-alc-{dataset}-{model}-{row['role']}-{acquisition}"
                )
                endpoint_artists.append(endpoint)

    q25 = float(data["q25_relative_alc_improvement_pct"].min())
    q75 = float(data["q75_relative_alc_improvement_pct"].max())
    lower_data = min(q25, 0.0)
    upper_data = max(q75, 0.0)
    span = max(upper_data - lower_data, 1.0)
    ax.set_ylim(lower_data - 0.17 * span, upper_data + 0.56 * span)
    ax.set_xlim(-0.50, len(MODEL_ORDER) - 0.50)
    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=0.70, alpha=0.75)
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([MODEL_LABELS[model] for model in MODEL_ORDER])
    ax.set_xlabel("Model (dataset-specific best FS)")
    ax.set_ylabel("Relative ALC improvement\nat ~30% labelled (%)")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(axis="y", color="#E7E7E7", linewidth=0.35)
    ax.set_axisbelow(True)
    ax.text(
        0.012,
        0.018,
        "Each segment: one dataset; endpoints: independent 10-trial medians\n"
        "Best FS: lowest mean full-pool MAPE; bars: Q25-Q75",
        transform=ax.transAxes,
        color="#555555",
        fontsize=7.1,
        ha="left",
        va="bottom",
    )
    return pair_lines, endpoint_artists


def _add_panel_a_legends(ax: mpl.axes.Axes) -> tuple[Legend]:
    shared_handles = [
        Line2D(
            [0],
            [0],
            linestyle="--",
            color=ACQUISITION_COLORS["random_selection"],
            linewidth=0.75,
            marker=MARKER_BY_ACQUISITION["random_selection"],
            markerfacecolor=ACQUISITION_COLORS["random_selection"],
            markeredgecolor="none",
            markeredgewidth=0.0,
            markersize=3.4,
            alpha=MARKER_ALPHA,
            label="Random",
        ),
        Line2D(
            [0],
            [0],
            linestyle="-",
            color="#4C78A8",
            linewidth=0.95,
            marker="s",
            markerfacecolor="#4C78A8",
            markeredgecolor="none",
            markeredgewidth=0.0,
            markersize=3.4,
            alpha=MARKER_ALPHA,
            label="Panel rule",
        ),
        Line2D(
            [0],
            [0],
            color="#7A7A7A",
            linewidth=0.8,
            marker="o",
            markerfacecolor="#7A7A7A",
            markeredgecolor="none",
            markersize=3.0,
            alpha=MARKER_ALPHA,
            label="92% → 95% → 98%",
        ),
        Line2D(
            [0],
            [0],
            linestyle="",
            marker="o",
            markerfacecolor=NFP_COLORS[PRIMARY_NFP_THRESHOLD],
            markeredgecolor="none",
            markersize=5.0,
            alpha=MARKER_ALPHA,
            label="Larger marker: 95% (primary)",
        ),
    ]
    shared_legend = ax.legend(
        handles=shared_handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 1.025),
        ncol=4,
        fontsize=7.1,
        columnspacing=0.75,
        handlelength=1.35,
        handletextpad=0.35,
        borderaxespad=0.0,
    )
    return (shared_legend,)


def build_summary_figure(
    summary: pd.DataFrame,
    panel_a_summary: pd.DataFrame | None = None,
    *,
    trial_summary: pd.DataFrame | None = None,
) -> mpl.figure.Figure:
    _require_columns(summary, SUMMARY_COLUMNS, "NFP summary data")
    data = summary.copy()
    for column in ("nfp_percent", "pool_n", "n_observations"):
        data[column] = _strict_numeric(data[column], column, integer=True)
    metric_columns = sorted(
        column
        for column in SUMMARY_COLUMNS
        if column.endswith(("_median", "_q25", "_q75"))
    )
    for column in metric_columns:
        data[column] = _strict_numeric(data[column], column)
        if bool(data[column].lt(0).any()):
            raise ValueError(f"{column} must be nonnegative")
    for stem in ("required_full_life_tests", "required_pool_fraction_pct", "required_duration"):
        ordered = (
            data[f"{stem}_q25"].le(data[f"{stem}_median"])
            & data[f"{stem}_median"].le(data[f"{stem}_q75"])
        )
        if not bool(ordered.all()):
            raise ValueError(f"Invalid quantile ordering for {stem}")
    expected_groups = {
        (dataset, acquisition, threshold)
        for dataset in DATASET_ORDER
        for acquisition in ACQUISITION_ORDER
        for threshold in NFP_THRESHOLDS
    }
    observed_groups = set(
        zip(data["dataset"], data["acquisition"], data["nfp_percent"].astype(int))
    )
    missing_groups = sorted(expected_groups.difference(observed_groups))
    extra_groups = sorted(observed_groups.difference(expected_groups))
    if missing_groups:
        raise ValueError(
            "NFP summary is missing dataset/acquisition/NFP groups: "
            f"count={len(missing_groups)}; examples={missing_groups[:5]}"
        )
    if extra_groups or len(data) != len(expected_groups):
        raise ValueError(
            "NFP summary contains duplicate or unexpected dataset/acquisition/NFP groups: "
            f"rows={len(data)}; expected={len(expected_groups)}; extras={extra_groups[:5]}"
        )
    expected_units = data["dataset"].map(DURATION_UNITS)
    if not bool(data["duration_unit"].astype(str).eq(expected_units.astype(str)).all()):
        raise ValueError("NFP summary contains non-canonical dataset duration units")
    for statistic in ("median", "q25", "q75"):
        expected_fraction = (
            data[f"required_full_life_tests_{statistic}"] / data["pool_n"] * 100.0
        )
        if not bool(
            np.isclose(
                data[f"required_pool_fraction_pct_{statistic}"],
                expected_fraction,
                rtol=1e-9,
                atol=1e-9,
            ).all()
        ):
            raise ValueError(f"Summary pool-fraction {statistic} is inconsistent with full-life tests")

    trial_data: pd.DataFrame | None = None
    if trial_summary is not None:
        _require_columns(trial_summary, TRIAL_SUMMARY_COLUMNS, "NFP trial summary data")
        trial_data = trial_summary.copy()
        for column in ("nfp_percent", "trial", "n_configurations"):
            trial_data[column] = _strict_numeric(trial_data[column], column, integer=True)
        for column in ("required_duration", "required_pool_fraction_pct"):
            trial_data[column] = _strict_numeric(trial_data[column], column)
        expected_trial_groups = {
            (dataset, acquisition, threshold, trial)
            for dataset in DATASET_ORDER
            for acquisition in ACQUISITION_ORDER
            for threshold in NFP_THRESHOLDS
            for trial in range(1, 11)
        }
        observed_trial_groups = set(
            zip(
                trial_data["dataset"],
                trial_data["acquisition"],
                trial_data["nfp_percent"].astype(int),
                trial_data["trial"].astype(int),
            )
        )
        if observed_trial_groups != expected_trial_groups or len(trial_data) != len(
            expected_trial_groups
        ):
            raise ValueError(
                "NFP trial summary must contain 10 paired trials per dataset, "
                "acquisition, and threshold"
            )
        trial_units = trial_data["dataset"].map(DURATION_UNITS)
        if not bool(
            trial_data["duration_unit"].astype(str).eq(trial_units.astype(str)).all()
        ):
            raise ValueError("NFP trial summary contains non-canonical duration units")

    _set_publication_style()
    fig = plt.figure(figsize=FIGURE_SIZE_INCHES, constrained_layout=False)
    grid = fig.add_gridspec(
        4,
        6,
        height_ratios=[1.25, 0.70, 0.70, 0.70],
        width_ratios=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        left=0.115,
        right=0.955,
        top=0.940,
        bottom=0.130,
        wspace=1.25,
        hspace=0.65,
    )
    panel_axes = {
        "a": fig.add_subplot(grid[0, 0:4]),
        "b": fig.add_subplot(grid[0, 4:6]),
        "c": fig.add_subplot(grid[1, 0:2]),
        "d": fig.add_subplot(grid[1, 2:4]),
        "e": fig.add_subplot(grid[2, 0:4]),
        "f": fig.add_subplot(grid[3, 0:2]),
        "g": fig.add_subplot(grid[3, 2:4]),
        "h": fig.add_subplot(grid[1:4, 4:6]),
    }
    panel_types = dict(PANEL_TYPES)
    panel_a_subaxes, threshold_artists, panel_a_random_artists = _draw_tradeoff_panel(
        panel_axes["a"], data
    )
    heatmap_colorbar, heatmap_cell_bubbles = _draw_heatmap_panel(panel_axes["b"], data)
    heatmap_position = panel_axes["b"].get_position()
    heatmap_target_width = HEATMAP_MATRIX_WIDTH_MM / (25.4 * fig.get_figwidth())
    heatmap_target_height = HEATMAP_MATRIX_HEIGHT_MM / (25.4 * fig.get_figheight())
    heatmap_width_delta = heatmap_target_width - heatmap_position.width
    heatmap_target_y0 = heatmap_position.y1 - heatmap_target_height
    panel_axes["b"].set_position(
        [
            heatmap_position.x1 - heatmap_target_width,
            heatmap_target_y0,
            heatmap_target_width,
            heatmap_target_height,
        ]
    )
    hero_position = panel_axes["a"].get_position()
    hero_right = hero_position.x1 - heatmap_width_delta
    hero_left = max(hero_position.x0 - heatmap_width_delta, 0.08)
    hero_target_height = PANEL_A_PLOT_HEIGHT_MM / (25.4 * fig.get_figheight())
    panel_axes["a"].set_position(
        [
            hero_left,
            hero_position.y1 - hero_target_height,
            hero_right - hero_left,
            hero_target_height,
        ]
    )
    panel_cd_shift = PANEL_CD_CLEARANCE_SHIFT_MM / (25.4 * fig.get_figheight())
    for panel in ("c", "d"):
        position = panel_axes[panel].get_position()
        panel_axes[panel].set_position(
            [position.x0, position.y0 - panel_cd_shift, position.width, position.height]
        )
    h_position = panel_axes["h"].get_position()
    h_right_shift = 0.018
    h_top_clearance = 0.040
    panel_axes["h"].set_position(
        [
            h_position.x0 + h_right_shift,
            h_position.y0,
            h_position.width - h_right_shift,
            h_position.height - h_top_clearance,
        ]
    )
    _draw_tests_trend_panel(panel_axes["c"], data)
    _draw_win_count_panel(panel_axes["d"], data)
    half_violin_bodies: dict[str, list[mpl.collections.PolyCollection]] = {}
    trial_point_artists: dict[str, list[mpl.collections.PathCollection]] = {}
    duration_threshold_connectors: dict[str, list[mpl.lines.Line2D]] = {}
    threshold_trajectory_artists: dict[str, list[mpl.artist.Artist]] = {
        "connectors": [],
        "markers": [],
    }
    if trial_data is None:
        _draw_duration_panel(
            panel_axes["e"],
            data,
            unit="cycles",
            title="Cycle-based datasets",
        )
        _draw_duration_panel(
            panel_axes["f"],
            data,
            unit="EFC",
            title="EFC datasets",
        )
        _draw_duration_panel(
            panel_axes["g"],
            data,
            unit="weeks",
            title="Week-based dataset",
        )
        _draw_dataset_penalty_panel(panel_axes["h"], data)
    else:
        for panel, unit in (("e", "cycles"), ("f", "EFC"), ("g", "weeks")):
            bodies, points, connectors = _draw_duration_dataset_half_violin_panel(
                panel_axes[panel],
                trial_data,
                panel=panel,
                unit=unit,
            )
            half_violin_bodies[panel] = bodies
            trial_point_artists[panel] = points
            duration_threshold_connectors[panel] = connectors
        h_bodies, h_points, threshold_trajectory_artists = (
            _draw_dataset_pool_fraction_half_violin_panel(
                panel_axes["h"], trial_data
            )
        )
        half_violin_bodies["h"] = h_bodies
        trial_point_artists["h"] = h_points
        panel_types.update(
            {
                "e": "duration_cycles_dataset_threshold_half_violin",
                "f": "duration_efc_dataset_threshold_half_violin",
                "g": "duration_weeks_dataset_threshold_half_violin",
                "h": "dataset_pool_fraction_threshold_horizontal_half_violin",
            }
        )

    panel_a_legends = _add_panel_a_legends(panel_axes["a"])
    fig._nfp_panel_axes = panel_axes
    fig._nfp_auxiliary_axes_by_panel = {
        "a": tuple(panel_a_subaxes.values()),
        "b": (heatmap_colorbar.ax,),
    }
    fig._nfp_heatmap_colorbar = heatmap_colorbar
    fig._nfp_heatmap_cell_bubbles = heatmap_cell_bubbles
    fig._nfp_half_violin_bodies = half_violin_bodies
    fig._nfp_trial_point_artists = trial_point_artists
    fig._nfp_duration_threshold_connectors = duration_threshold_connectors
    fig._nfp_threshold_trajectory_artists = threshold_trajectory_artists
    fig._nfp_panel_types = panel_types
    fig._nfp_figure_archetype = FIGURE_ARCHETYPE
    fig._nfp_marker_by_threshold = dict(MARKER_BY_THRESHOLD)
    fig._nfp_marker_by_acquisition = dict(MARKER_BY_ACQUISITION)
    fig._nfp_panel_a_legends = panel_a_legends
    fig._nfp_panel_a_subaxes = panel_a_subaxes
    fig._nfp_panel_a_random_artists = panel_a_random_artists
    fig._nfp_primary_threshold = PRIMARY_NFP_THRESHOLD
    fig._nfp_sensitivity_thresholds = tuple(SENSITIVITY_NFP_THRESHOLDS)
    fig._nfp_threshold_roles = {
        threshold: (
            "primary" if threshold == PRIMARY_NFP_THRESHOLD else "sensitivity"
        )
        for threshold in NFP_THRESHOLDS
    }
    fig._nfp_threshold_artists = threshold_artists
    return fig


def _bbox_within(inner: mpl.transforms.Bbox, outer: mpl.transforms.Bbox, tolerance: float = 1.0) -> bool:
    return bool(
        inner.x0 >= outer.x0 - tolerance
        and inner.y0 >= outer.y0 - tolerance
        and inner.x1 <= outer.x1 + tolerance
        and inner.y1 <= outer.y1 + tolerance
    )


def _audit_figure_layout(fig: mpl.figure.Figure) -> dict[str, bool]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas_bounds = fig.bbox
    panel_axes: dict[str, mpl.axes.Axes] = dict(fig._nfp_panel_axes)

    text_by_panel: dict[str, list[mpl.text.Text]] = {}
    all_text: list[mpl.text.Text] = list(fig.texts)
    for panel, ax in panel_axes.items():
        owned_axes = [
            ax,
            *getattr(fig, "_nfp_auxiliary_axes_by_panel", {}).get(panel, ()),
        ]
        artists = [
            artist
            for owned_ax in owned_axes
            for artist in (
                owned_ax._left_title,
                owned_ax.title,
                owned_ax._right_title,
                owned_ax.xaxis.label,
                owned_ax.yaxis.label,
                *owned_ax.texts,
                *owned_ax.get_xticklabels(),
                *owned_ax.get_yticklabels(),
            )
        ]
        visible = [
            artist
            for artist in artists
            if artist.get_visible() and bool(artist.get_text().strip())
        ]
        text_by_panel[panel] = visible
        all_text.extend(visible)

    text_inside = all(
        _bbox_within(artist.get_window_extent(renderer=renderer), canvas_bounds)
        for artist in all_text
        if artist.get_visible() and bool(artist.get_text().strip())
    )
    legend_owners: list[tuple[Legend, str | None]] = [
        (legend, None) for legend in fig.legends
    ]
    legend_owners.extend(
        (legend, "a") for legend in getattr(fig, "_nfp_panel_a_legends", ())
    )
    legend_bounds = [
        (legend.get_window_extent(renderer=renderer), owner)
        for legend, owner in legend_owners
        if legend.get_visible()
    ]
    legends_inside = all(
        _bbox_within(bounds, canvas_bounds) for bounds, _owner in legend_bounds
    )
    axes_bounds = {
        panel: ax.get_window_extent(renderer=renderer) for panel, ax in panel_axes.items()
    }
    legends_clear = all(
        not legend_bounds_item.overlaps(axis_bounds)
        for legend_bounds_item, owner in legend_bounds
        for panel, axis_bounds in axes_bounds.items()
        if panel != owner
    )
    cross_panel_clear = True
    for owner, artists in text_by_panel.items():
        for artist in artists:
            bounds = artist.get_window_extent(renderer=renderer)
            if any(
                bounds.overlaps(other_bounds)
                for other, other_bounds in axes_bounds.items()
                if other != owner
            ):
                cross_panel_clear = False
                break
        if not cross_panel_clear:
            break
    return {
        "all_text_within_canvas": text_inside,
        "all_legends_within_canvas": legends_inside,
        "legends_clear_of_data_axes": legends_clear,
        "panel_text_clear_of_other_data_axes": cross_panel_clear,
    }


def _panel_statistics_metadata(
    summary: pd.DataFrame,
    panel_a_summary: pd.DataFrame | None = None,
    trial_summary: pd.DataFrame | None = None,
) -> dict[str, dict[str, object]]:
    observation_values = pd.to_numeric(summary["n_observations"], errors="raise").astype(int)
    input_group_n = {
        "column": "n_observations",
        "definition": (
            "positive, reached feature-set x model x trial observations contributing "
            "to each dataset x acquisition x NFP group median"
        ),
        "minimum": int(observation_values.min()),
        "maximum": int(observation_values.max()),
        "exact_values": "source_data/fig_nfp_performance_indicator_summary.csv",
    }
    metadata: dict[str, dict[str, object]] = {
        "a": {
            "statistical_unit": "dataset-level dataset x acquisition x NFP group median",
            "center": "median across datasets",
            "spread": "Q25-Q75 across datasets",
            "n_definition": "8 datasets per acquisition x NFP point",
            "baseline": "random selection within the same dataset and NFP",
            "threshold_role": "NFP95 primary; NFP92 and NFP98 sensitivity",
        },
        "b": {
            "statistical_unit": "one dataset x acquisition x NFP95 group",
            "center": "within-group median across positive, reached observations",
            "spread": "not drawn; within-group Q25-Q75 retained in source data",
            "n_definition": (
                "one dataset x acquisition group per cell; exact observation n in source data"
            ),
            "threshold_role": "primary NFP95 only",
        },
        "c": {
            "statistical_unit": "dataset-level dataset x acquisition x NFP group median",
            "center": "median across datasets",
            "spread": "not drawn",
            "n_definition": "8 datasets per acquisition x NFP point",
            "threshold_role": "NFP95 primary; NFP92 and NFP98 sensitivity",
        },
        "d": {
            "statistical_unit": "dataset at NFP95",
            "center": "count of datasets with minimum median total duration at NFP95",
            "spread": "not applicable; count statistic",
            "n_definition": "8 datasets at primary NFP95",
            "threshold_role": "primary",
        },
        "e": {
            "statistical_unit": "acquisition-rule group median within dataset x NFP",
            "center": "median across acquisition rules",
            "spread": "Q25-Q75 across acquisition rules",
            "n_definition": "7 acquisition rules per dataset x NFP point",
            "absolute_duration_unit": "cycles",
            "threshold_role": "NFP95 primary; NFP92 and NFP98 sensitivity",
        },
        "f": {
            "statistical_unit": "acquisition-rule group median within dataset x NFP",
            "center": "median across acquisition rules",
            "spread": "Q25-Q75 across acquisition rules",
            "n_definition": "7 acquisition rules per dataset x NFP point",
            "absolute_duration_unit": "EFC",
            "threshold_role": "NFP95 primary; NFP92 and NFP98 sensitivity",
        },
        "g": {
            "statistical_unit": "acquisition-rule group median within dataset x NFP",
            "center": "median across acquisition rules",
            "spread": "Q25-Q75 across acquisition rules",
            "n_definition": "7 acquisition rules per dataset x NFP point",
            "absolute_duration_unit": "weeks",
            "threshold_role": "NFP95 primary; NFP92 and NFP98 sensitivity",
        },
        "h": {
            "statistical_unit": "acquisition-rule group median within dataset x NFP threshold",
            "center": "median across acquisition rules, expressed as NFP92 or NFP98 minus NFP95",
            "spread": "not drawn",
            "n_definition": "7 acquisition rules per dataset x threshold; deltas use NFP95 as reference",
            "threshold_role": "sensitivity vs primary",
        },
    }
    if trial_summary is not None:
        for panel, unit in (("e", "cycles"), ("f", "EFC"), ("g", "weeks")):
            metadata[panel] = {
                "statistical_unit": (
                    "trial-level median across eligible feature-set x model configurations "
                    "within dataset x acquisition x NFP x trial"
                ),
                "center": "median across acquisition-trial medians within dataset x NFP",
                "spread": "half-violin density and Q25-Q75 across acquisition-trial medians",
                "n_definition": (
                    "70 acquisition-trial medians per dataset x NFP threshold "
                    "(7 acquisition rules x 10 trials)"
                ),
                "absolute_duration_unit": unit,
                "threshold_role": "NFP95 primary; NFP92 and NFP98 sensitivity distributions",
            }
        metadata["h"] = {
            "statistical_unit": (
                "dataset-trial median across 7 acquisition-specific trial medians"
            ),
            "center": "median across 10 dataset-trial medians",
            "spread": (
                "horizontal half-violin density and Q25-Q75 across 10 "
                "dataset-trial medians"
            ),
            "n_definition": (
                "10 dataset-trial medians per dataset x NFP threshold; each trial median "
                "is across 7 acquisition rules"
            ),
            "threshold_role": "NFP95 primary; NFP92 and NFP98 sensitivity",
        }
    for panel in metadata.values():
        panel["input_group_n"] = dict(input_group_n)
    return metadata


def _readme_text() -> str:
    return """# Section 4 NFP performance-indicator summary

This directory contains an asymmetric, eight-panel Nature-style analysis of operational near-full-pool performance. NFP95 is the primary analysis threshold. NFP92 and NFP98 are sensitivity analyses. The composite is retained as a geometry reference, while publication-ready independent panel exports are provided under `results/panels`. Panels a and c remain legend-free in their independent exports; panels e-h retain their centered legends above the plotting axes. Standalone copies of every categorical legend are also exported under `results/legends`. Panel c's acquisition-rule key is export-only so it can be placed during final assembly without reducing the compact data area. The composite retains its displayed legends. The continuous color bar remains attached to panel b because it is the heatmap's numerical scale rather than a categorical legend.

The saved figure-delivery preference is **PNG-only**. Re-running the builder produces only opaque RGB PNG images; SVG, PDF, TIFF, and other image formats are rejected. Production PNG files use 600 dpi.

The composite uses an exact 3:3.5 width-to-height ratio. Panel a has a 55 mm plotting height while retaining its established width and top alignment. All visible typography is 2 pt larger than the preceding layout. Panel b uses a restrained Nature-style blue-white-red scale: lower required pool fractions are blue, the midpoint is warm white, and higher fractions are red. The heatmap contains no cell bubbles; values are written directly over the color field, acquisition-rule labels are centered on their cells, and the colorbar is a plain, borderless continuous scale.

- Panel a is a 2 x 3 small-multiple comparison of the six non-random acquisition rules against Random. Every sub-panel shows required pool fraction versus total-duration index relative to the within-dataset, within-threshold Random baseline, uses identical x/y limits, and retains cross-dataset Q25-Q75. The six single-line rule names are centered directly above each sub-panel, while the x/y labels and untitled encoding legend are shared once across panel a.
- Panel b is the primary NFP95 dataset-by-acquisition heatmap of required pool fraction.
- Panel c shows required full-life-test sensitivity around the emphasized NFP95 estimate. Its seven-rule acquisition legend is delivered as a separate PNG and is intentionally not embedded in the panel.
- Panel d counts the eight NFP95 datasets in which each acquisition rule is fastest; the sensitivity thresholds do not enter this primary ranking.
- Panels e-g show dataset on the x axis and absolute total duration on the y axis, separately for cycles, EFC, and weeks. Every dataset has adjacent NFP92, NFP95, and NFP98 right-half violins with the underlying acquisition-trial values and median/Q25-Q75 summaries; thin black lines connect their NFP92, NFP95, and NFP98 medians to show the absolute duration change across targets, while NFP95 remains visually primary. Threshold is encoded by color, while acquisition rule is encoded by marker shape using the shared marker key in panel a. Their NFP-target legends sit horizontally above and centered on the plotting axes, and the y-axis upper bound is 1.25 times the observed maximum so the violins use the data region efficiently without touching the frame. Panel e contains five cycle datasets, f contains two EFC datasets, and g contains one week-based dataset (eight datasets total). Panel g uses slightly wider threshold spacing to improve separation of its three violins.
- Panel h uses horizontal half-violins for each dataset at NFP92, NFP95, and NFP98. Each distribution contains ten dataset-trial medians aggregated across acquisition rules, with the individual trial points and median/Q25-Q75 overlaid. Thin black lines connect the three threshold medians so the sensitivity direction remains visible, while NFP95 is emphasized as primary. Its horizontal NFP-target legend is centered above the axes, and the categorical y range is tightened to remove the former legend-reserved blank row.

Acquisition rules use fixed colors and seven distinct marker shapes. `Diversity (One-shot)` and `Diversity (Iterative)` are written in full. Panel a identifies its six rules with single-line rule names centered directly above each sub-panel and uses one untitled shared legend for Random, the panel rule, the 92% → 95% → 98% trajectory, and the larger 95% primary marker. Panel c's standalone `Acquisition rule` legend lists all seven rules with the same line, color, and marker encodings. Panel b uses 35° acquisition-rule tick labels centered on the heatmap cells. Dataset labels in panels e, f, and g use 0° centered text and no x-axis title. Panel e splits its two hyphenated dataset names after the hyphen; Panels f and g keep every dataset name on one line. Their y-axis labels place the duration unit on a second line. Visible coordinate-axis lines use 0.4 pt strokes. The enlarged canvas and wider c–g plotting areas preserve the complete labels and increase separation between the three threshold violins. Acquisition-level markers have no border and use alpha 0.6; panel d bars also use alpha 0.6. In panels a, c, and e-g, marker shape identifies acquisition rule; the larger point identifies primary NFP95 within panel-a/c trajectories and threshold summaries. Panel titles and a–h subplot letters are intentionally off. Each source observation is one positive-attainable-gain dataset × feature-set × model × trial × acquisition combination at one NFP threshold. Raw duration units are never pooled across cycles, EFC, and weeks. All intervals are descriptive interquartile ranges, not confidence intervals. Reproduce the bundle by running `scripts/build_section4_nfp_summary.py` from the analysis package.

Panels e and f use a mathematical ×10^3 y-axis multiplier rather than abbreviated `k` labels. Dataset names are shown directly on the x axes, while acquisition-rule legend labels remain written in full.

Statistical units and exact n definitions:

- Panels a and c use n=8 dataset-level group medians per acquisition × NFP point; panel a shows cross-dataset Q25-Q75 and uses random selection from the same dataset and NFP as the duration baseline.
- Panel b contains one dataset × acquisition group median per NFP95 cell. The exact number of positive, reached feature-set × model × trial observations behind every cell is the `n_observations` column in the same-stem source CSV.
- Panel d counts n=8 datasets at the primary NFP95 threshold; ties are counted for every tied acquisition rule.
- Panels e-g use acquisition-specific trial medians. Each value is the median across eligible feature-set × model configurations within a dataset × acquisition × NFP × trial. Each dataset × NFP violin therefore contains n=70 acquisition-trial values (7 acquisition rules × 10 trials); the half violin and bar show the distribution and Q25-Q75.
- Panel h uses n=10 dataset-trial medians per dataset × NFP threshold. Each trial value is the median across the seven acquisition rules; horizontal half-violins show the distribution, while connected threshold medians and Q25-Q75 preserve the absolute sensitivity pattern.

The real-data group source CSV retains `n_trials` and `n_feature_model_combinations` for each of the 168 groups. `fig_nfp_trial_level_distributions.csv` contains the 1,680 acquisition-specific trial-level distribution rows used by panels e-h. No inferential test, multiple-comparison correction, or p-value is used in this descriptive figure.
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_provenance(
    source_dir: Path,
) -> list[dict[str, object]]:
    provenance: list[dict[str, object]] = []
    for threshold in NFP_THRESHOLDS:
        path = Path(source_dir) / f"fig_nfp{threshold}_required_full_life_tests.csv"
        rows = len(pd.read_csv(path, usecols=["nfp_percent"]))
        provenance.append(
            {
                "kind": "NFP performance indicators",
                "nfp_percent": threshold,
                "filename": path.name,
                "rows": int(rows),
                "sha256": _sha256(path),
            }
        )
    return provenance


def _ensure_rgb_raster(path: Path, extension: str, dpi: int) -> None:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    if extension == "png":
        rgb.save(path, format="PNG", dpi=(dpi, dpi), optimize=True)
    else:
        rgb.save(path, format="TIFF", dpi=(dpi, dpi), compression="tiff_lzw")


def _measure_export(path: Path) -> dict[str, object]:
    extension = path.suffix.lower().lstrip(".")
    measured: dict[str, object] = {
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }
    if extension in {"png", "tiff"}:
        with Image.open(path) as image:
            measured["pixels"] = [int(image.width), int(image.height)]
            measured["mode"] = image.mode
            measured["dpi"] = [float(value) for value in image.info.get("dpi", (0.0, 0.0))]
    elif extension == "svg":
        text_content = path.read_text(encoding="utf-8")
        width = re.search(r'<svg[^>]*\bwidth="([0-9.]+)pt"', text_content)
        height = re.search(r'<svg[^>]*\bheight="([0-9.]+)pt"', text_content)
        measured["width_pt"] = float(width.group(1)) if width else None
        measured["height_pt"] = float(height.group(1)) if height else None
        measured["editable_text_nodes"] = int(text_content.count("<text"))
    elif extension == "pdf":
        payload = path.read_bytes()
        measured["valid_pdf_signature"] = payload[:4] == b"%PDF"
        measured["embedded_truetype_fonts"] = int(payload.count(b"/FontFile2"))
        measured["type0_font_resources"] = int(payload.count(b"/Subtype /Type0"))
        measured["cid_truetype_font_resources"] = int(payload.count(b"/CIDFontType2"))
        measured["editable_text_resources"] = bool(
            measured["embedded_truetype_fonts"]
            and measured["type0_font_resources"]
            and measured["cid_truetype_font_resources"]
        )
    return measured


def _save_figure_path(
    fig: mpl.figure.Figure,
    path: Path,
    *,
    extension: str,
    dpi: int,
    bbox_inches: Bbox | None = None,
) -> None:
    save_kwargs: dict[str, object] = {
        "facecolor": "white",
        "transparent": False,
    }
    if bbox_inches is not None:
        save_kwargs["bbox_inches"] = bbox_inches
    if extension in {"png", "tiff"}:
        save_kwargs["dpi"] = int(dpi)
    elif extension == "svg":
        save_kwargs["metadata"] = {"Date": None}
    elif extension == "pdf":
        save_kwargs["metadata"] = {"CreationDate": None, "ModDate": None}
    if extension == "tiff":
        save_kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
    fig.savefig(path, **save_kwargs)
    if extension in {"png", "tiff"}:
        _ensure_rgb_raster(path, extension, int(dpi))


def _panel_legends(fig: mpl.figure.Figure) -> dict[str, tuple[Legend, ...]]:
    """Return every rendered legend grouped by its owning panel."""
    legends_by_panel: dict[str, tuple[Legend, ...]] = {}
    panel_a_legends = tuple(getattr(fig, "_nfp_panel_a_legends", ()))
    if panel_a_legends:
        legends_by_panel["a"] = panel_a_legends
    for panel, ax in getattr(fig, "_nfp_panel_axes", {}).items():
        if panel == "a":
            continue
        panel_legends_list: list[Legend] = []
        for legend in ax.findobj(Legend):
            if not any(legend is existing for existing in panel_legends_list):
                panel_legends_list.append(legend)
        panel_legends = tuple(panel_legends_list)
        if panel_legends:
            legends_by_panel[panel] = panel_legends
    return legends_by_panel


def _flatten_panel_legends(fig: mpl.figure.Figure) -> list[Legend]:
    legends: list[Legend] = []
    for panel_legends in _panel_legends(fig).values():
        for legend in panel_legends:
            if not any(legend is existing for existing in legends):
                legends.append(legend)
    return legends


def _export_standalone_legends(
    fig: mpl.figure.Figure,
    results_dir: Path,
    *,
    formats: tuple[str, ...],
    dpi: int,
    pad_points: float = 2.5,
) -> tuple[list[Path], dict[str, dict[str, dict[str, object]]]]:
    """Export one tightly cropped legend-only PNG for each legend-bearing panel."""
    legends_dir = Path(results_dir) / "legends"
    legends_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    measured: dict[str, dict[str, dict[str, object]]] = {}
    for panel, panel_legends in _panel_legends(fig).items():
        if len(panel_legends) != 1:
            raise ValueError(
                f"Standalone legend export requires one legend for panel {panel}; "
                f"found {len(panel_legends)}"
            )
        source = panel_legends[0]
        handles = [copy.copy(handle) for handle in source.legend_handles]
        labels = [text_item.get_text() for text_item in source.get_texts()]
        title = source.get_title().get_text()
        text_size = (
            float(source.get_texts()[0].get_fontsize())
            if source.get_texts()
            else float(mpl.rcParams["legend.fontsize"])
        )
        title_size = float(source.get_title().get_fontsize())
        legend_fig = plt.figure(figsize=(FIGURE_SIZE_INCHES[0], 2.2))
        try:
            standalone = legend_fig.legend(
                handles=handles,
                labels=labels,
                loc="center",
                ncol=int(getattr(source, "_ncols", 1)),
                title=title or None,
                frameon=False,
                fontsize=text_size,
                title_fontsize=title_size,
                handlelength=float(source.handlelength),
                handleheight=float(source.handleheight),
                handletextpad=float(source.handletextpad),
                columnspacing=float(source.columnspacing),
                labelspacing=float(source.labelspacing),
                markerscale=float(source.markerscale),
                borderpad=float(source.borderpad),
            )
            legend_fig.canvas.draw()
            renderer = legend_fig.canvas.get_renderer()
            legend_bounds = standalone.get_window_extent(renderer=renderer).transformed(
                legend_fig.dpi_scale_trans.inverted()
            )
            pad_inches = float(pad_points) / 72.0
            crop = Bbox.from_extents(
                legend_bounds.x0 - pad_inches,
                legend_bounds.y0 - pad_inches,
                legend_bounds.x1 + pad_inches,
                legend_bounds.y1 + pad_inches,
            )
            measured[panel] = {}
            for extension in formats:
                path = legends_dir / f"{PANEL_FILE_STEMS[panel]}_legend.{extension}"
                _save_figure_path(
                    legend_fig,
                    path,
                    extension=extension,
                    dpi=int(dpi),
                    bbox_inches=crop,
                )
                outputs.append(path)
                measured[panel][extension] = _measure_export(path)
        finally:
            plt.close(legend_fig)
    return outputs, measured


def _panel_export_geometry(
    fig: mpl.figure.Figure,
    *,
    pad_points: float = 2.5,
) -> tuple[dict[str, Bbox], dict[str, list[float]], dict[str, list[float]]]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    panel_bboxes: dict[str, Bbox] = {}
    axes_size_inches: dict[str, list[float]] = {}
    crop_size_inches: dict[str, list[float]] = {}
    pad_pixels = float(pad_points) / 72.0 * float(fig.dpi)
    for panel, ax in fig._nfp_panel_axes.items():
        boxes = [ax.get_tightbbox(renderer=renderer)]
        text_artists = [
            ax.xaxis.label,
            ax.yaxis.label,
            ax.xaxis.get_offset_text(),
            ax.yaxis.get_offset_text(),
            *ax.get_xticklabels(),
            *ax.get_yticklabels(),
        ]
        boxes.extend(
            artist.get_window_extent(renderer=renderer)
            for artist in text_artists
            if artist.get_visible() and bool(artist.get_text().strip())
        )
        boxes.extend(
            legend.get_window_extent(renderer=renderer)
            for legend in _panel_legends(fig).get(panel, ())
            if legend.get_visible()
        )
        boxes.extend(
            auxiliary_ax.get_tightbbox(renderer=renderer)
            for auxiliary_ax in getattr(fig, "_nfp_auxiliary_axes_by_panel", {}).get(panel, ())
            if auxiliary_ax.get_visible()
        )
        tight = Bbox.union(boxes)
        expanded = Bbox.from_extents(
            tight.x0 - pad_pixels,
            tight.y0 - pad_pixels,
            tight.x1 + pad_pixels,
            tight.y1 + pad_pixels,
        )
        bbox_inches = expanded.transformed(fig.dpi_scale_trans.inverted())
        panel_bboxes[panel] = bbox_inches
        position = ax.get_position()
        axes_size_inches[panel] = [
            float(position.width * FIGURE_SIZE_INCHES[0]),
            float(position.height * FIGURE_SIZE_INCHES[1]),
        ]
        crop_size_inches[panel] = [float(bbox_inches.width), float(bbox_inches.height)]
    return panel_bboxes, axes_size_inches, crop_size_inches


def _export_split_panels(
    fig: mpl.figure.Figure,
    results_dir: Path,
    *,
    formats: tuple[str, ...],
    dpi: int,
) -> tuple[
    list[Path],
    dict[str, dict[str, dict[str, object]]],
    dict[str, list[float]],
    dict[str, list[float]],
    bool,
]:
    panels_dir = Path(results_dir) / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    axes_visibility = {
        panel: ax.get_visible() for panel, ax in fig._nfp_panel_axes.items()
    }
    auxiliary_visibility = {
        auxiliary_ax: auxiliary_ax.get_visible()
        for auxiliary_axes in getattr(fig, "_nfp_auxiliary_axes_by_panel", {}).values()
        for auxiliary_ax in auxiliary_axes
    }
    legends_by_panel = _panel_legends(fig)
    embedded_legend_panels = set(EMBEDDED_SPLIT_PANEL_LEGENDS).intersection(
        legends_by_panel
    )
    panel_legends = _flatten_panel_legends(fig)
    legend_visibility = [legend.get_visible() for legend in panel_legends]
    outputs: list[Path] = []
    measured: dict[str, dict[str, dict[str, object]]] = {
        panel: {} for panel in PANEL_FILE_STEMS
    }
    embedded_legend_policy_valid = True
    try:
        for owner, legends in legends_by_panel.items():
            for legend in legends:
                legend.set_visible(owner in embedded_legend_panels)
        panel_bboxes, axes_size_inches, crop_size_inches = _panel_export_geometry(fig)
        for panel, stem in PANEL_FILE_STEMS.items():
            for other_panel, ax in fig._nfp_panel_axes.items():
                ax.set_visible(other_panel == panel)
            for owner, auxiliary_axes in getattr(
                fig, "_nfp_auxiliary_axes_by_panel", {}
            ).items():
                for auxiliary_ax in auxiliary_axes:
                    auxiliary_ax.set_visible(owner == panel)
            for owner, legends in legends_by_panel.items():
                for legend in legends:
                    legend.set_visible(
                        owner == panel and panel in embedded_legend_panels
                    )
            visible_legend_owners = {
                owner
                for owner, legends in legends_by_panel.items()
                if any(legend.get_visible() for legend in legends)
            }
            expected_visible_owners = (
                {panel} if panel in embedded_legend_panels else set()
            )
            embedded_legend_policy_valid = bool(
                embedded_legend_policy_valid
                and visible_legend_owners == expected_visible_owners
            )
            for extension in formats:
                path = panels_dir / f"{stem}.{extension}"
                _save_figure_path(
                    fig,
                    path,
                    extension=extension,
                    dpi=int(dpi),
                    bbox_inches=panel_bboxes[panel],
                )
                outputs.append(path)
                measured[panel][extension] = _measure_export(path)
    finally:
        for panel, ax in fig._nfp_panel_axes.items():
            ax.set_visible(axes_visibility[panel])
        for auxiliary_ax, visible in auxiliary_visibility.items():
            auxiliary_ax.set_visible(visible)
        for legend, visible in zip(panel_legends, legend_visibility):
            legend.set_visible(visible)
        fig.canvas.draw()
    return (
        outputs,
        measured,
        axes_size_inches,
        crop_size_inches,
        embedded_legend_policy_valid,
    )


def export_summary_bundle(
    summary: pd.DataFrame,
    output_dir: Path,
    trial_summary: pd.DataFrame | None = None,
    formats: tuple[str, ...] = DEFAULT_EXPORT_FORMATS,
    dpi: int = 600,
    input_provenance: list[dict[str, object]] | None = None,
    source_data_dir: Path | None = None,
    enforce_publication_dpi: bool = True,
) -> list[Path]:
    normalized_formats = tuple(dict.fromkeys(extension.lower().lstrip(".") for extension in formats))
    unsupported = sorted(set(normalized_formats).difference({"png"}))
    if unsupported:
        raise ValueError(
            f"Only PNG export is supported; requested unsupported formats: {unsupported}"
        )
    if not normalized_formats:
        raise ValueError("At least one PNG export format is required")
    if int(dpi) <= 0:
        raise ValueError("dpi must be positive")
    includes_raster = bool(set(normalized_formats).intersection({"png", "tiff"}))
    if enforce_publication_dpi and includes_raster and int(dpi) != 600:
        raise ValueError("Publication raster exports require 600 dpi")

    output_dir = Path(output_dir)
    results_dir = output_dir / "outputs"
    source_dir = Path(source_data_dir) if source_data_dir is not None else output_dir / "source_data"
    results_dir.mkdir(parents=True, exist_ok=True)
    if source_data_dir is None:
        source_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    source_path = source_dir / f"{FIGURE_STEM}.csv"
    if source_data_dir is None:
        summary.to_csv(source_path, index=False)
        outputs.append(source_path)
    trial_source_path: Path | None = None
    if trial_summary is not None:
        trial_source_path = source_dir / "fig_nfp_trial_level_distributions.csv"
        if source_data_dir is None:
            trial_summary.to_csv(trial_source_path, index=False)
            outputs.append(trial_source_path)

    fig = build_summary_figure(summary, trial_summary=trial_summary)
    rendered_archetype = str(fig._nfp_figure_archetype)
    rendered_panel_types = dict(fig._nfp_panel_types)
    rendered_duration_threshold_connector_counts = {
        panel: len(connectors)
        for panel, connectors in fig._nfp_duration_threshold_connectors.items()
    }
    layout_audit = _audit_figure_layout(fig)
    rendered_panel_bounds = {
        label: [float(value) for value in ax.get_position().bounds]
        for label, ax in fig._nfp_panel_axes.items()
    }
    fig.canvas.draw()
    rendered_titles_off = all(
        not title.get_text()
        for ax in fig._nfp_panel_axes.values()
        for title in (ax._left_title, ax.title, ax._right_title)
    )
    rendered_panel_letters_off = all(
        panel not in {text_item.get_text() for text_item in ax.texts}
        for panel, ax in fig._nfp_panel_axes.items()
    )
    rendered_bar_alpha = all(
        patch.get_alpha() == MARKER_ALPHA for patch in fig._nfp_panel_axes["d"].patches
    )
    rendered_heatmap_ticks = np.asarray(fig._nfp_heatmap_colorbar.get_ticks(), dtype=float)
    rendered_heatmap_has_no_cell_bubbles = bool(
        fig._nfp_heatmap_cell_bubbles is None
        and not any(
            collection.get_gid() == "heatmap-cell-value-bubbles"
            for collection in fig._nfp_panel_axes["b"].collections
        )
    )
    rendered_heatmap_x_labels_centered = bool(
        all(
            label.get_ha() == "right"
            for label in fig._nfp_panel_axes["b"].get_xticklabels()
        )
        and np.allclose(
            [
                label.get_position()[0]
                for label in fig._nfp_panel_axes["b"].get_xticklabels()
            ],
            np.arange(len(ACQUISITION_ORDER), dtype=float),
        )
    )
    rendered_legend_titles = [
        legend.get_title().get_text() for legend in fig._nfp_panel_a_legends
    ]
    rendered_legend_labels = [
        text_item.get_text()
        for legend in fig._nfp_panel_a_legends
        for text_item in legend.get_texts()
    ]
    visible_text_sizes = [
        text_item.get_fontsize()
        for text_item in fig.findobj(mpl.text.Text)
        if text_item.get_visible() and text_item.get_text().strip()
    ]
    rendered_x_tick_rotations = {
        panel: sorted(
            {
                float(label.get_rotation())
                for label in fig._nfp_panel_axes[panel].get_xticklabels()
                if label.get_visible() and label.get_text().strip()
            }
        )
        for panel in ("b", "e", "f", "g")
    }
    renderer = fig.canvas.get_renderer()
    violin_legends_above_center_valid = True
    if trial_summary is not None:
        violin_legends_above_center_valid = all(
            (legend := fig._nfp_panel_axes[panel].get_legend()) is not None
            and legend.get_title().get_text() == "NFP target"
            and int(getattr(legend, "_ncols", 1)) == 3
            and (
                legend_bounds := legend.get_window_extent(renderer=renderer)
            ).y0
            >= (
                axes_bounds := fig._nfp_panel_axes[panel].get_window_extent(
                    renderer=renderer
                )
            ).y1
            and np.isclose(
                legend_bounds.x0 + legend_bounds.width / 2.0,
                axes_bounds.x0 + axes_bounds.width / 2.0,
                atol=1.0,
            )
            for panel in EMBEDDED_SPLIT_PANEL_LEGENDS
        )
    cycle_ylim = sorted(fig._nfp_panel_axes["e"].get_ylim())
    duration_and_h_axes_valid = bool(
        (
            trial_summary is None
            and cycle_ylim[0] <= -1.2
            and cycle_ylim[1] >= 5.2
        )
        or (
            trial_summary is not None
            and all(
                fig._nfp_panel_axes[panel].get_xlabel() == ""
                for panel in ("e", "f", "g")
            )
            and [
                label.get_text()
                for label in fig._nfp_panel_axes["e"].get_xticklabels()
            ]
            == ["MIT", "LSD-\nprimary", "LSD-\nsecond", "HUST", "Formation"]
            and [
                label.get_text()
                for label in fig._nfp_panel_axes["f"].get_xticklabels()
            ]
            == ["KIT", "TRI-Tesla"]
            and [
                label.get_text()
                for label in fig._nfp_panel_axes["g"].get_xticklabels()
            ]
            == ["ISU-ILCC"]
            and fig._nfp_panel_axes["e"].get_ylabel()
            == "Total duration\n" + r"($\times10^3$ cycles)"
            and fig._nfp_panel_axes["f"].get_ylabel()
            == "Total duration\n" + r"($\times10^3$ EFC)"
            and fig._nfp_panel_axes["g"].get_ylabel() == "Total duration\n(weeks)"
            and fig._nfp_panel_axes["h"].get_xlabel() == "Required pool fraction (%)"
            and all(
                np.allclose(
                    fig._nfp_panel_axes[panel].get_ylim(),
                    (
                        0.0,
                        float(
                            trial_summary.loc[
                                trial_summary["duration_unit"].eq(unit),
                                "required_duration",
                            ].max()
                        )
                        * DURATION_Y_AXIS_UPPER_FACTOR,
                    ),
                )
                for panel, unit in (("e", "cycles"), ("f", "EFC"), ("g", "weeks"))
            )
            and np.allclose(
                fig._nfp_panel_axes["h"].get_ylim(), PANEL_H_Y_AXIS_LIMITS
            )
            and violin_legends_above_center_valid
        )
    )
    panel_a_subaxes = dict(fig._nfp_panel_a_subaxes)
    panel_a_positions = [
        panel_a_subaxes[rule].get_position() for rule in ACQUISITION_ORDER[1:]
    ]
    panel_a_small_multiples_valid = bool(
        list(panel_a_subaxes) == list(ACQUISITION_ORDER[1:])
        and len(panel_a_subaxes) == 6
        and len({round(position.y0, 6) for position in panel_a_positions}) == 2
        and len({round(position.x0, 6) for position in panel_a_positions[:3]}) == 3
        and len({round(position.x0, 6) for position in panel_a_positions[3:]}) == 3
        and len(
            {
                tuple(np.round(subax.get_xlim(), 8))
                for subax in panel_a_subaxes.values()
            }
        )
        == 1
        and len(
            {
                tuple(np.round(subax.get_ylim(), 8))
                for subax in panel_a_subaxes.values()
            }
        )
        == 1
        and all(
            sum(
                (collection.get_gid() or "").startswith(f"hero-{rule}-")
                for collection in subax.collections
            )
            == len(NFP_THRESHOLDS)
            and sum(
                (collection.get_gid() or "").startswith(
                    f"hero-random_selection-vs-{rule}-"
                )
                for collection in subax.collections
            )
            == len(NFP_THRESHOLDS)
            for rule, subax in panel_a_subaxes.items()
        )
        and all(
            len(
                rule_labels := [
                    text_item
                    for text_item in subax.texts
                    if text_item.get_gid() == f"panel-a-rule-label-{rule}"
                ]
            )
            == 1
            and rule_labels[0].get_text() == ACQUISITION_LABELS[rule]
            and "\n" not in rule_labels[0].get_text()
            and rule_labels[0].get_ha() == "center"
            and rule_labels[0].get_va() == "bottom"
            and np.isclose(rule_labels[0].get_position()[0], 0.5)
            and rule_labels[0].get_position()[1] > 1.0
            for rule, subax in panel_a_subaxes.items()
        )
    )
    revised_visual_contract_valid = bool(
        np.isclose(FIGURE_SIZE_INCHES[0] / FIGURE_SIZE_INCHES[1], 3.0 / 3.5)
        and np.isclose(
            fig._nfp_panel_axes["a"].get_position().height
            * fig.get_figheight()
            * 25.4,
            PANEL_A_PLOT_HEIGHT_MM,
        )
        and np.isclose(float(mpl.rcParams["font.size"]), 9.0)
        and np.isclose(float(mpl.rcParams["axes.labelsize"]), 8.8)
        and bool(visible_text_sizes)
        and min(visible_text_sizes) >= 7.1 - 1e-9
        and fig._nfp_heatmap_colorbar.ax.get_ylabel() == "Required pool fraction (%)"
        and np.allclose(rendered_heatmap_ticks, HEATMAP_COLORBAR_TICKS)
        and not fig._nfp_heatmap_colorbar.outline.get_visible()
        and rendered_heatmap_has_no_cell_bubbles
        and rendered_heatmap_x_labels_centered
        and panel_a_small_multiples_valid
        and rendered_legend_titles == [""]
        and "Random" in rendered_legend_labels
        and "Panel rule" in rendered_legend_labels
        and "92% → 95% → 98%" in rendered_legend_labels
        and "Larger marker: 95% (primary)" in rendered_legend_labels
        and fig._nfp_panel_axes["a"].get_xlabel()
        == "Required pool fraction (%)"
        and fig._nfp_panel_axes["a"].get_ylabel()
        == "Relative total duration\n(% of random)"
        and not any(
            "Bubble area" in text_item.get_text()
            for text_item in fig._nfp_panel_axes["a"].texts
        )
        and fig._nfp_panel_axes["d"].get_xlabel()
        == "Fastest datasets at NFP target of 95%"
        and rendered_x_tick_rotations["b"] == [35.0]
        and (
            trial_summary is None
            or all(
                rendered_x_tick_rotations[panel] == [0.0]
                for panel in ("e", "f", "g")
            )
        )
        and duration_and_h_axes_valid
    )
    trial_violin_contract_valid = True
    duration_threshold_connectors_valid = True
    duration_datasets_by_panel = {
        panel: [
            dataset
            for dataset in DATASET_ORDER
            if DURATION_UNITS[dataset] == unit
        ]
        for panel, unit in (("e", "cycles"), ("f", "EFC"), ("g", "weeks"))
    }
    if trial_summary is not None:
        dataset_counts_by_panel = {
            panel: len(datasets)
            for panel, datasets in duration_datasets_by_panel.items()
        }
        trial_violin_contract_valid = bool(
            len(trial_summary)
            == len(DATASET_ORDER)
            * len(ACQUISITION_ORDER)
            * len(NFP_THRESHOLDS)
            * 10
            and set(
                trial_summary.groupby(
                    ["dataset", "acquisition", "nfp_percent"]
                ).size()
            )
            == {10}
            and [dataset_counts_by_panel[panel] for panel in ("e", "f", "g")]
            == [5, 2, 1]
            and {
                dataset
                for datasets in duration_datasets_by_panel.values()
                for dataset in datasets
            }
            == set(DATASET_ORDER)
            and all(
                len(fig._nfp_half_violin_bodies.get(panel, ()))
                == dataset_count * len(NFP_THRESHOLDS)
                for panel, dataset_count in dataset_counts_by_panel.items()
            )
            and all(
                len(fig._nfp_trial_point_artists.get(panel, ()))
                == len(ACQUISITION_ORDER)
                * len(NFP_THRESHOLDS)
                * dataset_count
                and all(
                    len(artist.get_offsets()) == 10
                    for artist in fig._nfp_trial_point_artists[panel]
                )
                for panel, dataset_count in dataset_counts_by_panel.items()
            )
            and all(
                all(
                    sum(
                        f"-acquisition-{acquisition}-" in str(artist.get_gid())
                        for artist in fig._nfp_trial_point_artists[panel]
                    )
                    == len(datasets) * len(NFP_THRESHOLDS)
                    for acquisition in ACQUISITION_ORDER
                )
                for panel, datasets in duration_datasets_by_panel.items()
            )
            and len(fig._nfp_threshold_trajectory_artists["connectors"])
            == len(DATASET_ORDER)
            and all(
                connector.get_color() == MEDIAN_CONNECTOR_COLOR
                for connector in fig._nfp_threshold_trajectory_artists["connectors"]
            )
            and len(fig._nfp_threshold_trajectory_artists["markers"])
            == len(DATASET_ORDER) * len(NFP_THRESHOLDS)
            and len(fig._nfp_half_violin_bodies.get("h", ()))
            == len(DATASET_ORDER) * len(NFP_THRESHOLDS)
            and len(fig._nfp_trial_point_artists.get("h", ()))
            == len(DATASET_ORDER) * len(NFP_THRESHOLDS)
            and all(
                len(artist.get_offsets()) == 10
                for artist in fig._nfp_trial_point_artists["h"]
            )
        )
        expected_duration_connector_counts = {"e": 5, "f": 2, "g": 1}
        duration_threshold_connectors_valid = bool(
            set(fig._nfp_duration_threshold_connectors)
            == set(expected_duration_connector_counts)
            and all(
                len(fig._nfp_duration_threshold_connectors[panel]) == count
                and all(
                    len(connector.get_xdata()) == len(NFP_THRESHOLDS)
                    and len(connector.get_ydata()) == len(NFP_THRESHOLDS)
                    and connector.get_color() == MEDIAN_CONNECTOR_COLOR
                    and np.isclose(connector.get_linewidth(), 0.65)
                    and np.isclose(float(connector.get_alpha()), 0.65)
                    and np.isclose(float(connector.get_zorder()), 1.0)
                    for connector in fig._nfp_duration_threshold_connectors[panel]
                )
                for panel, count in expected_duration_connector_counts.items()
            )
        )
    split_panel_exports: dict[str, dict[str, dict[str, object]]] = {}
    standalone_legend_exports: dict[str, dict[str, dict[str, object]]] = {}
    panel_axes_size_inches: dict[str, list[float]] = {}
    panel_crop_size_inches: dict[str, list[float]] = {}
    split_panel_legend_policy_valid = False
    rendered_legend_panels = set(_panel_legends(fig))
    rendered_embedded_legend_panels = sorted(
        set(EMBEDDED_SPLIT_PANEL_LEGENDS).intersection(rendered_legend_panels)
    )
    rendered_export_only_legend_panels = sorted(
        set(EXPORT_ONLY_LEGEND_PANELS).intersection(rendered_legend_panels)
    )
    composite_legends = [
        legend
        for panel, panel_legends in _panel_legends(fig).items()
        if panel not in EXPORT_ONLY_LEGEND_PANELS
        for legend in panel_legends
    ]
    export_only_legends = [
        legend
        for panel, panel_legends in _panel_legends(fig).items()
        if panel in EXPORT_ONLY_LEGEND_PANELS
        for legend in panel_legends
    ]
    composite_retains_legends = bool(
        composite_legends
        and all(legend.get_visible() for legend in composite_legends)
        and all(not legend.get_visible() for legend in export_only_legends)
    )
    try:
        for extension in normalized_formats:
            path = results_dir / f"{FIGURE_STEM}.{extension}"
            _save_figure_path(
                fig,
                path,
                extension=extension,
                dpi=int(dpi),
            )
            outputs.append(path)
        (
            split_outputs,
            split_panel_exports,
            panel_axes_size_inches,
            panel_crop_size_inches,
            split_panel_legend_policy_valid,
        ) = _export_split_panels(
            fig,
            results_dir,
            formats=normalized_formats,
            dpi=int(dpi),
        )
        outputs.extend(split_outputs)
        legend_outputs, standalone_legend_exports = _export_standalone_legends(
            fig,
            results_dir,
            formats=normalized_formats,
            dpi=int(dpi),
        )
        outputs.extend(legend_outputs)
    finally:
        plt.close(fig)

    measured_exports = {
        path.suffix.lower().lstrip("."): _measure_export(path)
        for path in outputs
        if path.parent == results_dir
    }
    expected_pixels = [
        int(FIGURE_SIZE_INCHES[0] * int(dpi)),
        int(FIGURE_SIZE_INCHES[1] * int(dpi)),
    ]
    asymmetric_geometry = (
        rendered_panel_bounds["a"][2] > rendered_panel_bounds["b"][2]
        and rendered_panel_bounds["a"][3] > rendered_panel_bounds["c"][3]
        and rendered_panel_bounds["e"][2] > rendered_panel_bounds["f"][2]
    )
    panel_statistics = _panel_statistics_metadata(summary, trial_summary=trial_summary)
    exact_group_n_columns = [
        column
        for column in (
            "dataset",
            "acquisition",
            "nfp_percent",
            "n_observations",
            "n_trials",
            "n_feature_model_combinations",
        )
        if column in summary.columns
    ]
    exact_group_n = summary.loc[:, exact_group_n_columns].to_dict("records")
    statistics_complete = all(
        bool(panel.get(key))
        for panel in panel_statistics.values()
        for key in ("statistical_unit", "center", "spread", "n_definition")
    )
    duration_units_valid = (
        set(summary["duration_unit"].astype(str)) == set(DURATION_UNITS.values())
        and panel_statistics["e"]["absolute_duration_unit"] == "cycles"
        and panel_statistics["f"]["absolute_duration_unit"] == "EFC"
        and panel_statistics["g"]["absolute_duration_unit"] == "weeks"
    )
    threshold_roles_valid = bool(
        PRIMARY_NFP_THRESHOLD == 95
        and tuple(SENSITIVITY_NFP_THRESHOLDS) == (92, 98)
        and set(summary["nfp_percent"].astype(int))
        == {PRIMARY_NFP_THRESHOLD, *SENSITIVITY_NFP_THRESHOLDS}
        and panel_statistics["d"]["threshold_role"] == "primary"
        and panel_statistics["h"]["threshold_role"]
        == (
            "NFP95 primary; NFP92 and NFP98 sensitivity"
            if trial_summary is not None
            else "sensitivity vs primary"
        )
    )
    split_bundle_complete = bool(
        set(split_panel_exports) == set(PANEL_FILE_STEMS)
        and all(
            set(split_panel_exports[panel]) == set(normalized_formats)
            for panel in PANEL_FILE_STEMS
        )
        and all(
            path.exists() and path.stat().st_size > 0
            for path in outputs
            if path.parent.name == "panels"
        )
    )
    standalone_legend_bundle_complete = bool(
        rendered_legend_panels == set(standalone_legend_exports)
        and rendered_legend_panels
        == (
            {"a", "c", "e", "f", "g", "h"}
            if trial_summary is not None
            else {"a", "c"}
        )
        and all(
            set(standalone_legend_exports[panel]) == set(normalized_formats)
            for panel in rendered_legend_panels
        )
        and all(
            path.exists() and path.stat().st_size > 0
            for path in outputs
            if path.parent.name == "legends"
        )
    )
    panel_geometry_preserved = all(
        np.allclose(
            panel_axes_size_inches[panel],
            [
                rendered_panel_bounds[panel][2] * FIGURE_SIZE_INCHES[0],
                rendered_panel_bounds[panel][3] * FIGURE_SIZE_INCHES[1],
            ],
            rtol=0.0,
            atol=1e-12,
        )
        for panel in PANEL_FILE_STEMS
    )
    polish_contract_valid = bool(
        rendered_titles_off
        and rendered_panel_letters_off
        and rendered_bar_alpha
        and len(set(MARKER_BY_ACQUISITION.values())) == len(ACQUISITION_ORDER)
        and MARKER_ALPHA == 0.6
        and AXIS_LINEWIDTH == 0.4
    )
    qa_checks: list[tuple[str, bool, str]] = [
        (
            "Complete dataset-acquisition-threshold grid",
            len(summary) == 168,
            f"summary_groups={len(summary)}; expected=168",
        ),
        (
            "Heterogeneous panel types",
            len(rendered_panel_types) == 8 and len(set(rendered_panel_types.values())) >= 6,
            f"panels={len(rendered_panel_types)}; unique_types={len(set(rendered_panel_types.values()))}",
        ),
        (
            "Panel-a 2x3 rule-versus-random small multiples",
            panel_a_small_multiples_valid,
            "six_nonrandom_rules; shared_xy_limits; shared_axis_labels; one_shared_legend",
        ),
        (
            "Asymmetric panel geometry",
            asymmetric_geometry,
            "hero_wider_than_heatmap; hero_taller_than_trend; cycles_wider_than_EFC",
        ),
        (
            "Figure text and legends stay inside canvas",
            bool(
                layout_audit["all_text_within_canvas"]
                and layout_audit["all_legends_within_canvas"]
            ),
            json.dumps(layout_audit, sort_keys=True),
        ),
        (
            "Legends and panel text avoid other data axes",
            bool(
                layout_audit["legends_clear_of_data_axes"]
                and layout_audit["panel_text_clear_of_other_data_axes"]
            ),
            json.dumps(layout_audit, sort_keys=True),
        ),
        (
            "Exact n disclosure",
            statistics_complete and len(exact_group_n) == len(summary),
            f"panel_definitions={len(panel_statistics)}; exact_group_records={len(exact_group_n)}",
        ),
        (
            "Variability definition",
            statistics_complete,
            "panel-specific center, spread, statistical unit, and n definition recorded",
        ),
        (
            "Duration units remain dataset-native",
            duration_units_valid,
            f"observed={sorted(set(summary['duration_unit'].astype(str)))}",
        ),
        (
            "Publication raster resolution contract",
            (not enforce_publication_dpi) or (not includes_raster) or int(dpi) == 600,
            f"enforced={bool(enforce_publication_dpi)}; dpi={int(dpi)}",
        ),
        (
            "PNG-only export policy",
            normalized_formats == DEFAULT_EXPORT_FORMATS,
            f"formats={list(normalized_formats)}",
        ),
        (
            "Complete NFP thresholds",
            set(summary["nfp_percent"].astype(int)) == set(NFP_THRESHOLDS),
            f"observed={sorted(set(summary['nfp_percent'].astype(int)))}",
        ),
        (
            "Primary and sensitivity NFP roles",
            threshold_roles_valid,
            "primary=NFP95; sensitivity=NFP92,NFP98; panel_d_primary_only; panel_h_absolute_threshold_half_violins",
        ),
        (
            "Requested panel polish contract",
            polish_contract_valid,
            "titles=off; panel_letters=off; unique_acquisition_markers=7; marker_and_bar_alpha=0.6; axis_linewidth=0.4",
        ),
        (
            "Heatmap palette, typography, labels, and 3:3.5 layout",
            revised_visual_contract_valid,
            (
                f"panel_a_height={PANEL_A_PLOT_HEIGHT_MM:g}mm; "
                "heatmap=blue_white_red_without_cell_bubbles_and_centered_x_labels; "
                "colorbar=plain_borderless; font_delta=2pt; labels=revised; "
                    "x_tick_rotation_b=35deg_e_f_g=0deg; "
                    "violin_legends=above_center; y_ranges=tightened; aspect=3:3.5"
            ),
        ),
        (
            "Trial-level grouped half-violin contract",
            trial_violin_contract_valid,
            (
                "e-g=5+2+1_datasets_x_3_vertical_half_violins_with_7_acquisition_markers; "
                    "h=8_datasets_x_3_horizontal_half_violins_with_10_trial_points_and_black_median_connectors"
                if trial_summary is not None
                else "trial-level grouped half violins not requested for this compatibility export"
            ),
        ),
        (
            "Duration threshold median connectors",
            duration_threshold_connectors_valid,
            (
                "e=5_f=2_g=1_dataset_connectors; order=NFP92_to_NFP95_to_NFP98; "
                "color=black; absolute_duration_preserved"
                if trial_summary is not None
                else "trial-level duration connectors not requested for this compatibility export"
            ),
        ),
        (
            "Complete split-panel export bundle",
            split_bundle_complete,
            f"panels={len(split_panel_exports)}; formats={list(normalized_formats)}",
        ),
        (
            "Embedded violin-panel legends and standalone legend bundle",
            bool(
                split_panel_legend_policy_valid
                and composite_retains_legends
                and standalone_legend_bundle_complete
            ),
            (
                f"embedded_split_panel_legends={rendered_embedded_legend_panels}; "
                f"export_only_legend_panels={rendered_export_only_legend_panels}; "
                f"policy_valid={split_panel_legend_policy_valid}; "
                f"composite_retains_legends={composite_retains_legends}; "
                f"legend_panels={sorted(standalone_legend_exports)}; "
                "panel_b_colorbar_attached=True"
            ),
        ),
        (
            "Split-panel axes preserve composite geometry",
            panel_geometry_preserved,
            "independent crops retain the original axes width and height in inches",
        ),
        (
            "Source-data round trip",
            len(pd.read_csv(source_path)) == len(summary)
            and (
                trial_summary is None
                or (
                    trial_source_path is not None
                    and len(pd.read_csv(trial_source_path)) == len(trial_summary)
                )
            ),
            (
                f"summary_rows={len(pd.read_csv(source_path))}; "
                f"trial_rows={0 if trial_source_path is None else len(pd.read_csv(trial_source_path))}"
            ),
        ),
    ]
    if "svg" in measured_exports:
        svg_measure = measured_exports["svg"]
        qa_checks.append(
            (
                "Editable SVG text",
                int(svg_measure.get("editable_text_nodes", 0)) > 0,
                f"text_nodes={svg_measure.get('editable_text_nodes', 0)}",
            )
        )
    if "svg" in normalized_formats:
        split_svg_text = [
            int(split_panel_exports[panel]["svg"].get("editable_text_nodes", 0))
            for panel in PANEL_FILE_STEMS
        ]
        qa_checks.append(
            (
                "Split-panel editable SVG text",
                all(value > 0 for value in split_svg_text),
                f"text_nodes_by_panel={dict(zip(PANEL_FILE_STEMS, split_svg_text))}",
            )
        )
    for extension in ("png", "tiff"):
        if extension in measured_exports:
            raster = measured_exports[extension]
            dpi_values = raster.get("dpi", [0.0, 0.0])
            qa_checks.extend(
                [
                    (
                        f"Exact raster dimensions ({extension})",
                        raster.get("pixels") == expected_pixels,
                        f"observed={raster.get('pixels')}; expected={expected_pixels}",
                    ),
                    (
                        f"Opaque RGB raster ({extension})",
                        raster.get("mode") == "RGB",
                        f"mode={raster.get('mode')}",
                    ),
                    (
                        f"Raster resolution ({extension})",
                        all(abs(float(value) - int(dpi)) < 0.1 for value in dpi_values),
                        f"observed={dpi_values}; expected={int(dpi)}",
                    ),
                ]
            )
    if "pdf" in measured_exports:
        qa_checks.extend(
            [
                (
                "Valid PDF signature",
                bool(measured_exports["pdf"].get("valid_pdf_signature")),
                f"valid={measured_exports['pdf'].get('valid_pdf_signature')}",
                ),
                (
                    "Editable PDF text resources",
                    bool(measured_exports["pdf"].get("editable_text_resources")),
                    (
                        f"FontFile2={measured_exports['pdf'].get('embedded_truetype_fonts')}; "
                        f"Type0={measured_exports['pdf'].get('type0_font_resources')}; "
                        f"CIDFontType2={measured_exports['pdf'].get('cid_truetype_font_resources')}"
                    ),
                ),
            ]
        )
    if "pdf" in normalized_formats:
        split_pdf_editable = all(
            bool(split_panel_exports[panel]["pdf"].get("editable_text_resources"))
            for panel in PANEL_FILE_STEMS
        )
        qa_checks.append(
            (
                "Split-panel editable PDF text resources",
                split_pdf_editable,
                f"panels={len(PANEL_FILE_STEMS)}",
            )
        )
    for extension in ("png", "tiff"):
        if extension not in normalized_formats:
            continue
        split_rasters = [split_panel_exports[panel][extension] for panel in PANEL_FILE_STEMS]
        raster_contract = all(
            raster.get("mode") == "RGB"
            and all(
                abs(float(value) - int(dpi)) < 0.1
                for value in raster.get("dpi", [0.0, 0.0])
            )
            for raster in split_rasters
        )
        qa_checks.append(
            (
                f"Split-panel opaque RGB raster resolution ({extension})",
                raster_contract,
                f"panels={len(split_rasters)}; dpi={int(dpi)}",
            )
        )
        legend_rasters = [
            standalone_legend_exports[panel][extension]
            for panel in sorted(standalone_legend_exports)
        ]
        qa_checks.append(
            (
                f"Standalone-legend opaque RGB raster resolution ({extension})",
                bool(
                    legend_rasters
                    and all(
                        raster.get("mode") == "RGB"
                        and all(
                            abs(float(value) - int(dpi)) < 0.1
                            for value in raster.get("dpi", [0.0, 0.0])
                        )
                        for raster in legend_rasters
                    )
                ),
                f"legends={len(legend_rasters)}; dpi={int(dpi)}",
            )
        )

    manifest = {
        "figure_stem": FIGURE_STEM,
        "backend": "Python/matplotlib",
        "figure_archetype": rendered_archetype,
        "figure_size_inches": list(FIGURE_SIZE_INCHES),
        "figure_aspect_ratio_width_to_height": [3, 3.5],
        "panel_a_plot_height_mm": PANEL_A_PLOT_HEIGHT_MM,
        "font_size_delta_points": FONT_SIZE_DELTA_PT,
        "heatmap_colorbar": {
            "label": "Required pool fraction (%)",
            "ticks": list(HEATMAP_COLORBAR_TICKS),
            "value_range": list(HEATMAP_VALUE_RANGE),
            "colormap_colors": list(HEATMAP_CMAP_COLORS),
            "bubbles": "none",
            "outline": "none",
        },
        "heatmap_cell_bubbles": {"count": 0, "encoding": "none"},
        "panel_a_small_multiples": {
            "rows": 2,
            "columns": 3,
            "rules": list(ACQUISITION_ORDER[1:]),
            "comparator": "random_selection",
            "shared_x_axis_label": "Required pool fraction (%)",
            "shared_y_axis_label": "Relative total duration (% of random)",
            "shared_limits": True,
            "shared_legend": True,
            "shared_legend_title": "",
            "rule_label_position": "above_center",
            "rule_label_lines": 1,
        },
        "publication_dpi_required": 600,
        "publication_dpi_enforced": bool(enforce_publication_dpi),
        "export_policy": "PNG only",
        "export_formats": list(normalized_formats),
        "panel_types": rendered_panel_types,
        "panel_bounds": rendered_panel_bounds,
        "panel_axes_size_inches": panel_axes_size_inches,
        "panel_crop_size_inches": panel_crop_size_inches,
        "split_panel_exports": split_panel_exports,
        "standalone_legend_exports": standalone_legend_exports,
        "split_panels_contain_legends": bool(rendered_embedded_legend_panels),
        "embedded_split_panel_legends": rendered_embedded_legend_panels,
        "export_only_legend_panels": rendered_export_only_legend_panels,
        "violin_panel_legend_position": "above_center",
        "duration_y_axis_upper_factor": DURATION_Y_AXIS_UPPER_FACTOR,
        "panel_h_y_axis_limits": list(PANEL_H_Y_AXIS_LIMITS),
        "composite_retains_legends": composite_retains_legends,
        "heatmap_colorbar_separated": False,
        "panel_file_stems": PANEL_FILE_STEMS,
        "panel_titles": "off",
        "panel_letters": False,
        "axis_linewidth_points": AXIS_LINEWIDTH,
        "x_tick_label_rotation_degrees": rendered_x_tick_rotations,
        "marker_alpha": MARKER_ALPHA,
        "acquisition_marker_by_rule": MARKER_BY_ACQUISITION,
        "duration_x_axis": "dataset",
        "duration_x_axis_label": "",
        "duration_tick_label_wrapping": {
            "e": "hyphenated names split after the hyphen",
            "f": "single line",
            "g": "single line",
        },
        "duration_point_marker_encoding": "acquisition rule",
        "duration_point_marker_by_acquisition": MARKER_BY_ACQUISITION,
        "duration_threshold_median_connectors": {
            "panels": {
                panel: count
                for panel, count in rendered_duration_threshold_connector_counts.items()
            },
            "threshold_order": list(NFP_THRESHOLDS),
            "color": MEDIAN_CONNECTOR_COLOR,
            "linewidth_points": 0.65,
            "alpha": 0.65,
            "meaning": "absolute duration change across NFP targets within dataset",
        },
        "duration_threshold_offsets_by_panel": {
            "e": [-0.28, 0.0, 0.28],
            "f": [-0.28, 0.0, 0.28],
            "g": [-0.34, 0.0, 0.34],
        },
        "panel_h_threshold_offsets_y": PANEL_H_THRESHOLD_OFFSETS,
        "duration_panel_datasets": duration_datasets_by_panel,
        "layout_audit": layout_audit,
        "panel_statistics": panel_statistics,
        "exact_group_n": exact_group_n,
        "nfp_thresholds": list(NFP_THRESHOLDS),
        "primary_nfp_threshold": PRIMARY_NFP_THRESHOLD,
        "sensitivity_nfp_thresholds": list(SENSITIVITY_NFP_THRESHOLDS),
        "center": "median",
        "interval": "Q25-Q75",
        "interval_definition": "Descriptive interquartile range; not a confidence interval",
        "filter": "positive_attainable_gain == True and reached == True",
        "dataset_duration_units": DURATION_UNITS,
        "dataset_order": list(DATASET_ORDER),
        "acquisition_order": list(ACQUISITION_ORDER),
        "summary_rows": int(len(summary)),
        "trial_distribution_rows": 0 if trial_summary is None else int(len(trial_summary)),
        "trial_distribution_source": (
            None if trial_source_path is None else trial_source_path.name
        ),
        "represented_input_observations": int(summary["n_observations"].sum()),
        "input_sources": input_provenance or [],
        "runtime": {
            "python": platform.python_version(),
            "matplotlib": mpl.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "exports": [path.name for path in outputs if path.parent == results_dir],
        "measured_exports": measured_exports,
    }
    manifest_path = results_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs.append(manifest_path)

    readme_path = results_dir / "rendering_metadata.md"
    readme_path.write_text(_readme_text(), encoding="utf-8")
    outputs.append(readme_path)

    qa_lines = [
        f"{'PASS' if passed else 'FAIL'} | {name} | {detail}"
        for name, passed, detail in qa_checks
    ]
    qa_path = results_dir / "qa_report.txt"
    qa_path.write_text("\n".join(qa_lines) + "\n", encoding="utf-8")
    outputs.append(qa_path)
    failed_checks = [name for name, passed, _detail in qa_checks if not passed]
    if failed_checks:
        raise ValueError(f"Export QA failed: {failed_checks}")
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Section 4 NFP summary figure bundle")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args(argv)

    summary = pd.read_csv(args.source_dir / f"{FIGURE_STEM}.csv")
    trial_summary = pd.read_csv(args.source_dir / "fig_nfp_trial_level_distributions.csv")
    outputs = export_summary_bundle(
        summary,
        args.output_dir,
        trial_summary=trial_summary,
        dpi=args.dpi,
        source_data_dir=args.source_dir,
        input_provenance=[
            {
                "filename": f"{FIGURE_STEM}.csv",
                "sha256": _sha256(args.source_dir / f"{FIGURE_STEM}.csv"),
                "rows": int(len(summary)),
            },
            {
                "filename": "fig_nfp_trial_level_distributions.csv",
                "sha256": _sha256(args.source_dir / "fig_nfp_trial_level_distributions.csv"),
                "rows": int(len(trial_summary)),
            },
        ],
    )
    print(f"Built {FIGURE_STEM} from packaged source-data summaries")
    print(f"Summary rows: {len(summary):,}")
    print(f"Trial-distribution rows: {len(trial_summary):,}")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
