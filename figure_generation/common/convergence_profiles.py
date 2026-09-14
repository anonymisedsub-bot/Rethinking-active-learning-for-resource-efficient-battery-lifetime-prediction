from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


REQUIRED_TRIALS = 10
PRIMARY_NFP_ALPHA = 0.95
TARGET_FRACTIONS = (0.20, 0.50, 0.80)
EMBEDDING_TARGET_FRACTION = 0.30
PANEL_C_MAX_LABELLED_FRACTION = 0.60
RANDOM_ACQUISITION = "random_selection"
FRACTION_ALIGNMENT_ATOL = 5e-10

FEATURE_DIRECTORY = {
    "set1_dQn_m": "feature set 1",
    "set2_early_soh": "feature set 2",
    "set3_features": "feature set 3",
    "set4_features_metadata": "feature set 4",
    "set5_features_metadata_corr95": "feature set 5",
}
FEATURE_MATRIX_KEY = {
    "set1_dQn_m": "dQn_m_matrix",
    "set2_early_soh": "early_soh_degradation_matrix",
    "set3_features": "features_matrix",
    "set4_features_metadata": "features_metadata_matrix",
    "set5_features_metadata_corr95": "features_metadata_matrix",
}
DATASET_MAT_FILE = {
    "MIT": "data_MIT.mat",
    "ISU_ILCC": "data_ISU_ILCC.mat",
    "LSD_Primary": "data_LSD_primary.mat",
    "LSD_Second": "data_LSD_second.mat",
    "HUST": "data_HUST.mat",
    "KIT": "data_KIT.mat",
    "Formation": "data_Formation.mat",
    "TRI_Tesla": "data_TRI_Tesla.mat",
}


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise KeyError(f"{label} is missing required columns: {missing}")


def _single_context(frame: pd.DataFrame) -> str:
    context_columns = [
        column
        for column in ("dataset", "feature_set", "model", "trial")
        if column in frame.columns
    ]
    if not context_columns or frame.empty:
        return "input"
    values = frame.iloc[0][context_columns].to_dict()
    return ", ".join(f"{key}={value}" for key, value in values.items())


def select_best_feature_set(
    full_pool: pd.DataFrame,
    feature_order: Sequence[str],
    required_trials: int = REQUIRED_TRIALS,
) -> dict[str, Any]:
    """Select the lowest-mean full-pool MAPE feature with deterministic ties."""

    _require_columns(
        full_pool,
        ["feature_set", "trial", "full_pool_mape"],
        "full-pool table",
    )
    if full_pool.empty:
        raise ValueError("Full-pool table is empty")
    feature_order = list(feature_order)
    available = set(full_pool["feature_set"].astype(str))
    missing_features = [feature for feature in feature_order if feature not in available]
    if missing_features:
        raise ValueError(f"Full-pool candidates are missing feature sets: {missing_features}")

    trial_counts = full_pool.groupby("feature_set", sort=False)["trial"].nunique()
    invalid = trial_counts.loc[trial_counts.ne(required_trials)]
    if not invalid.empty:
        raise ValueError(
            f"Each candidate feature set must contain {required_trials} unique trials; "
            f"observed {invalid.to_dict()}"
        )

    trial_level = (
        full_pool.groupby(["feature_set", "trial"], as_index=False, sort=False)[
            "full_pool_mape"
        ]
        .mean()
    )
    summary = (
        trial_level.groupby("feature_set", as_index=False, sort=False)
        .agg(
            mean_full_pool_mape=("full_pool_mape", "mean"),
            n_trials=("trial", "nunique"),
        )
    )
    order_lookup = {feature: index for index, feature in enumerate(feature_order)}
    summary["feature_order"] = summary["feature_set"].map(order_lookup)
    if summary["feature_order"].isna().any():
        unexpected = summary.loc[summary["feature_order"].isna(), "feature_set"].tolist()
        raise ValueError(f"Feature sets are absent from the fixed order: {unexpected}")
    selected = summary.sort_values(
        ["mean_full_pool_mape", "feature_order"],
        kind="mergesort",
    ).iloc[0]
    return {
        "feature_set": str(selected["feature_set"]),
        "mean_full_pool_mape": float(selected["mean_full_pool_mape"]),
        "n_trials": int(selected["n_trials"]),
    }


def select_relative_alc_configurations(
    full_pool: pd.DataFrame,
    relative_config: pd.DataFrame,
    relative_trial: pd.DataFrame,
    *,
    dataset: str,
    feature_order: Sequence[str],
    model_order: Sequence[str],
    acquisition_order: Sequence[str],
    required_trials: int = REQUIRED_TRIALS,
) -> pd.DataFrame:
    """Select feature sets and median-relative-ALC Best/Worst rule profiles."""

    _require_columns(
        full_pool,
        ["dataset", "feature_set", "model", "trial", "full_pool_mape"],
        "full-pool table",
    )
    _require_columns(
        relative_config,
        [
            "dataset",
            "feature_set",
            "model",
            "acquisition",
            "median_relative_improvement_pct",
            "n_trials",
        ],
        "relative-ALC configuration table",
    )
    _require_columns(
        relative_trial,
        [
            "dataset",
            "feature_set",
            "model",
            "acquisition",
            "trial",
            "relative_improvement_pct",
        ],
        "relative-ALC trial table",
    )
    acquisition_order = list(acquisition_order)
    nonrandom_order = [
        acquisition
        for acquisition in acquisition_order
        if acquisition != RANDOM_ACQUISITION
    ]
    if RANDOM_ACQUISITION not in acquisition_order or not nonrandom_order:
        raise ValueError("Relative-ALC selection requires Random and non-random rules")
    acquisition_rank = {
        acquisition: index for index, acquisition in enumerate(nonrandom_order)
    }

    rows: list[dict[str, Any]] = []
    for model in model_order:
        model_full_pool = full_pool.loc[
            full_pool["dataset"].eq(dataset) & full_pool["model"].eq(model)
        ].copy()
        feature = select_best_feature_set(
            model_full_pool,
            feature_order=feature_order,
            required_trials=required_trials,
        )
        feature_set = str(feature["feature_set"])
        summary = relative_config.loc[
            relative_config["dataset"].eq(dataset)
            & relative_config["model"].eq(model)
            & relative_config["feature_set"].eq(feature_set)
            & relative_config["acquisition"].isin(nonrandom_order)
        ].copy()
        missing = sorted(set(nonrandom_order).difference(summary["acquisition"]))
        if missing:
            raise ValueError(
                f"Relative-ALC configuration is missing rules for dataset={dataset}, "
                f"model={model}, feature_set={feature_set}: {missing}"
            )
        if summary["acquisition"].duplicated().any():
            raise ValueError("Relative-ALC configuration has duplicate acquisition rows")
        summary["n_trials"] = pd.to_numeric(summary["n_trials"], errors="raise")
        invalid_counts = summary.loc[summary["n_trials"].ne(required_trials)]
        if not invalid_counts.empty:
            observed = dict(
                zip(
                    invalid_counts["acquisition"],
                    invalid_counts["n_trials"],
                    strict=True,
                )
            )
            raise ValueError(
                f"Relative-ALC rules require {required_trials} trials; observed {observed}"
            )
        summary["median_relative_improvement_pct"] = pd.to_numeric(
            summary["median_relative_improvement_pct"], errors="raise"
        )
        if not np.isfinite(summary["median_relative_improvement_pct"]).all():
            raise ValueError("Relative-ALC medians must be finite")
        candidate_trials = relative_trial.loc[
            relative_trial["dataset"].eq(dataset)
            & relative_trial["model"].eq(model)
            & relative_trial["feature_set"].eq(feature_set)
            & relative_trial["acquisition"].isin(nonrandom_order)
        ].copy()
        candidate_trials["relative_improvement_pct"] = pd.to_numeric(
            candidate_trials["relative_improvement_pct"], errors="raise"
        )
        if not np.isfinite(candidate_trials["relative_improvement_pct"]).all():
            raise ValueError("Relative-ALC trial improvements must be finite")
        trial_support = (
            candidate_trials.groupby("acquisition", as_index=False, sort=False)
            .agg(
                n_trial_rows=("trial", "size"),
                n_unique_trials=("trial", "nunique"),
                observed_median_relative_improvement_pct=(
                    "relative_improvement_pct",
                    "median",
                ),
            )
        )
        summary = summary.merge(
            trial_support,
            on="acquisition",
            how="left",
            validate="one_to_one",
        )
        invalid_support = summary.loc[
            summary["n_trial_rows"].ne(required_trials)
            | summary["n_unique_trials"].ne(required_trials)
        ]
        if not invalid_support.empty:
            observed = {
                str(row.acquisition): {
                    "rows": int(row.n_trial_rows) if pd.notna(row.n_trial_rows) else 0,
                    "unique_trials": (
                        int(row.n_unique_trials)
                        if pd.notna(row.n_unique_trials)
                        else 0
                    ),
                }
                for row in invalid_support.itertuples()
            }
            raise ValueError(
                f"Relative-ALC candidate rules require {required_trials} unique trials "
                f"and one row per trial; observed {observed}"
            )
        configured_medians = summary[
            "median_relative_improvement_pct"
        ].to_numpy(dtype=float)
        observed_medians = summary[
            "observed_median_relative_improvement_pct"
        ].to_numpy(dtype=float)
        if not np.allclose(
            configured_medians,
            observed_medians,
            rtol=1e-10,
            atol=1e-10,
        ):
            mismatches = summary.loc[
                ~np.isclose(
                    configured_medians,
                    observed_medians,
                    rtol=1e-10,
                    atol=1e-10,
                ),
                [
                    "acquisition",
                    "median_relative_improvement_pct",
                    "observed_median_relative_improvement_pct",
                ],
            ]
            raise ValueError(
                "Relative-ALC configuration medians disagree with trial-level rows: "
                f"{mismatches.to_dict(orient='records')}"
            )
        summary["median_relative_improvement_pct"] = observed_medians
        summary["acquisition_order"] = summary["acquisition"].map(acquisition_rank)
        best_candidates = summary.loc[
            summary["median_relative_improvement_pct"].eq(
                summary["median_relative_improvement_pct"].max()
            )
        ].sort_values("acquisition_order", kind="mergesort")
        worst_candidates = summary.loc[
            summary["median_relative_improvement_pct"].eq(
                summary["median_relative_improvement_pct"].min()
            )
        ].sort_values("acquisition_order", kind="mergesort")
        best = best_candidates.iloc[0]
        worst = worst_candidates.iloc[0]
        best_rule = str(best["acquisition"])
        worst_rule = str(worst["acquisition"])
        if best_rule == worst_rule:
            raise ValueError("Best and Worst relative-ALC rules must be distinct")

        trial_subset = candidate_trials.loc[
            candidate_trials["acquisition"].isin([best_rule, worst_rule])
        ].copy()
        representative_trials: dict[str, int] = {}
        representative_values: dict[str, float] = {}
        for role, rule, median in (
            ("best", best_rule, float(best["median_relative_improvement_pct"])),
            ("worst", worst_rule, float(worst["median_relative_improvement_pct"])),
        ):
            trials = trial_subset.loc[trial_subset["acquisition"].eq(rule)].copy()
            if trials["trial"].nunique() != required_trials or len(trials) != required_trials:
                raise ValueError(
                    f"Relative-ALC trial rows require {required_trials} unique trials for "
                    f"dataset={dataset}, model={model}, acquisition={rule}"
                )
            trials["relative_improvement_pct"] = pd.to_numeric(
                trials["relative_improvement_pct"], errors="raise"
            )
            trials["median_distance"] = (
                trials["relative_improvement_pct"] - median
            ).abs()
            representative = trials.sort_values(
                ["median_distance", "trial"], kind="mergesort"
            ).iloc[0]
            representative_trials[role] = int(representative["trial"])
            representative_values[role] = float(
                representative["relative_improvement_pct"]
            )

        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "feature_set": feature_set,
                "mean_full_pool_mape": float(feature["mean_full_pool_mape"]),
                "n_feature_trials": int(feature["n_trials"]),
                "best_acquisition": best_rule,
                "best_median_relative_improvement_pct": float(
                    best["median_relative_improvement_pct"]
                ),
                "best_representative_trial": representative_trials["best"],
                "best_representative_relative_improvement_pct": representative_values[
                    "best"
                ],
                "worst_acquisition": worst_rule,
                "worst_median_relative_improvement_pct": float(
                    worst["median_relative_improvement_pct"]
                ),
                "worst_representative_trial": representative_trials["worst"],
                "worst_representative_relative_improvement_pct": representative_values[
                    "worst"
                ],
                "random_acquisition": RANDOM_ACQUISITION,
            }
        )
    result = pd.DataFrame(rows)
    order = {model: index for index, model in enumerate(model_order)}
    result["model_order"] = result["model"].map(order)
    return (
        result.sort_values("model_order", kind="mergesort")
        .drop(columns="model_order")
        .reset_index(drop=True)
    )


def select_representative_trial(attainable: pd.DataFrame) -> dict[str, Any]:
    """Select the maximum positive attainable gain, breaking ties by trial."""

    _require_columns(
        attainable,
        ["trial", "attainable_gain_pp", "positive_attainable_gain"],
        "attainable-gain table",
    )
    eligible = attainable.loc[
        attainable["positive_attainable_gain"].astype(bool)
        & attainable["attainable_gain_pp"].gt(0)
    ].copy()
    if eligible.empty:
        raise ValueError(f"No positive attainable gain exists for {_single_context(attainable)}")
    selected = eligible.sort_values(
        ["attainable_gain_pp", "trial"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]
    return {
        key: (value.item() if isinstance(value, np.generic) else value)
        for key, value in selected.to_dict().items()
    }


def rank_best_worst_rules(
    crossings: pd.DataFrame,
    acquisition_order: Sequence[str],
    alpha: float = PRIMARY_NFP_ALPHA,
) -> dict[str, Any]:
    """Rank non-random rules by continuous NFP crossing for one configuration."""

    _require_columns(
        crossings,
        [
            "acquisition",
            "status",
            "continuous_labelled_n",
            "operational_labelled_n",
        ],
        "NFP crossing table",
    )
    data = crossings.copy()
    if "alpha" in data.columns:
        data = data.loc[np.isclose(data["alpha"].astype(float), float(alpha))].copy()
    nonrandom_order = [
        acquisition
        for acquisition in acquisition_order
        if acquisition != RANDOM_ACQUISITION
    ]
    present = set(data["acquisition"].astype(str))
    if RANDOM_ACQUISITION not in present:
        raise ValueError("Random must be present as a distinct comparator")
    missing = [acquisition for acquisition in nonrandom_order if acquisition not in present]
    valid = data.loc[
        data["acquisition"].isin(nonrandom_order)
        & data["status"].eq("crossed")
        & np.isfinite(data["continuous_labelled_n"])
    ].copy()
    invalid = [
        acquisition
        for acquisition in nonrandom_order
        if acquisition not in set(valid["acquisition"].astype(str))
    ]
    if missing or invalid or len(valid) != len(nonrandom_order):
        raise ValueError(
            "All non-random rules require valid NFP95 crossings; "
            f"missing={missing}, invalid={invalid}"
        )

    order_lookup = {
        acquisition: index for index, acquisition in enumerate(nonrandom_order)
    }
    valid["acquisition_order"] = valid["acquisition"].map(order_lookup)
    valid = valid.sort_values(
        ["continuous_labelled_n", "acquisition_order"],
        kind="mergesort",
    )
    best = valid.iloc[0]
    worst = valid.iloc[-1]
    best_name = str(best["acquisition"])
    worst_name = str(worst["acquisition"])
    if len({best_name, worst_name, RANDOM_ACQUISITION}) != 3:
        raise ValueError("Best, Worst, and Random must be three distinct rules")
    return {
        "best_acquisition": best_name,
        "best_continuous_labelled_n": float(best["continuous_labelled_n"]),
        "best_operational_labelled_n": int(best["operational_labelled_n"]),
        "worst_acquisition": worst_name,
        "worst_continuous_labelled_n": float(worst["continuous_labelled_n"]),
        "worst_operational_labelled_n": int(worst["operational_labelled_n"]),
        "random_acquisition": RANDOM_ACQUISITION,
    }


def match_nearest_checkpoint(
    checkpoints: pd.DataFrame,
    target_fraction: float,
) -> dict[str, Any]:
    """Match a nominal fraction to an observed checkpoint for one trial/rule."""

    _require_columns(checkpoints, ["labelled_n", "pool_n"], "checkpoint table")
    if checkpoints.empty:
        raise ValueError("Checkpoint table is empty")
    if checkpoints["pool_n"].nunique() != 1:
        raise ValueError("Checkpoint table must have one common pool_n")
    pool_n = float(checkpoints["pool_n"].iloc[0])
    if pool_n <= 0:
        raise ValueError("pool_n must be positive")
    if not 0 <= float(target_fraction) <= 1:
        raise ValueError("target_fraction must lie in [0, 1]")

    candidates = checkpoints.copy()
    candidates["achieved_fraction"] = candidates["labelled_n"].astype(float) / pool_n
    candidates["fraction_distance"] = (
        candidates["achieved_fraction"] - float(target_fraction)
    ).abs()
    selected = candidates.sort_values(
        ["fraction_distance", "labelled_n"],
        kind="mergesort",
    ).iloc[0]
    result = {
        key: (value.item() if isinstance(value, np.generic) else value)
        for key, value in selected.to_dict().items()
    }
    result["target_fraction"] = float(target_fraction)
    return result


def _available_group_columns(frame: pd.DataFrame, candidates: Sequence[str]) -> list[str]:
    return [column for column in candidates if column in frame.columns]


def _iter_groups(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
):
    if group_columns:
        yield from frame.groupby(list(group_columns), sort=False, dropna=False)
    else:
        yield (), frame


def _group_values(group_key: Any, group_columns: Sequence[str]) -> dict[str, Any]:
    if not group_columns:
        return {}
    values = group_key if isinstance(group_key, tuple) else (group_key,)
    return dict(zip(group_columns, values, strict=True))


def trial_balanced_ecdf(
    errors: pd.DataFrame,
    x_grid: np.ndarray,
    required_trials: int = REQUIRED_TRIALS,
) -> pd.DataFrame:
    """Average per-trial empirical CDFs, assigning equal weight to each trial."""

    _require_columns(errors, ["trial", "ape"], "ECDF error table")
    grid = np.asarray(x_grid, dtype=float)
    if grid.ndim != 1 or grid.size == 0 or not np.isfinite(grid).all():
        raise ValueError("ECDF x_grid must be a non-empty finite one-dimensional array")
    group_columns = _available_group_columns(
        errors,
        ("dataset", "model", "role", "acquisition", "target_fraction"),
    )
    rows: list[dict[str, Any]] = []
    for group_key, group in _iter_groups(errors, group_columns):
        trial_ids = sorted(group["trial"].dropna().unique().tolist())
        if len(trial_ids) != required_trials:
            raise ValueError(
                f"ECDF group {_group_values(group_key, group_columns)} requires "
                f"{required_trials} trials; observed {len(trial_ids)}"
            )
        curves = []
        for trial in trial_ids:
            values = np.sort(
                group.loc[group["trial"].eq(trial), "ape"]
                .to_numpy(dtype=float)
            )
            values = values[np.isfinite(values)]
            if values.size == 0:
                raise ValueError(f"ECDF trial {trial} has no finite APE values")
            curves.append(np.searchsorted(values, grid, side="right") / values.size)
        matrix = np.vstack(curves)
        sem = (
            matrix.std(axis=0, ddof=1) / np.sqrt(matrix.shape[0])
            if matrix.shape[0] > 1
            else np.zeros(grid.size, dtype=float)
        )
        context = _group_values(group_key, group_columns)
        for x_value, mean_value, sem_value in zip(
            grid,
            matrix.mean(axis=0),
            sem,
            strict=True,
        ):
            rows.append(
                {
                    **context,
                    "ape_grid": float(x_value),
                    "ecdf_mean": float(mean_value),
                    "ecdf_sem": float(sem_value),
                    "n_trials": int(matrix.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def common_fraction_alignment(
    checkpoints: pd.DataFrame,
    acquisitions: Sequence[str],
    required_trials: int = REQUIRED_TRIALS,
    nominal_fractions: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Match a common fraction grid and retain levels complete across trials/rules."""

    _require_columns(
        checkpoints,
        ["trial", "acquisition", "labelled_n", "pool_n"],
        "fraction-alignment table",
    )
    acquisitions = list(acquisitions)
    subset = checkpoints.loc[checkpoints["acquisition"].isin(acquisitions)].copy()
    missing = sorted(set(acquisitions).difference(subset["acquisition"].unique()))
    if missing:
        raise ValueError(f"Fraction alignment is missing acquisitions: {missing}")
    trial_counts = subset.groupby("acquisition")["trial"].nunique()
    incomplete = trial_counts.loc[trial_counts.ne(required_trials)]
    if not incomplete.empty:
        raise ValueError(
            f"Fraction alignment requires {required_trials} trials per rule; "
            f"observed {incomplete.to_dict()}"
        )

    if nominal_fractions is None:
        nominal = sorted(
            np.unique(
                np.round(
                    subset["labelled_n"].to_numpy(dtype=float)
                    / subset["pool_n"].to_numpy(dtype=float),
                    decimals=10,
                )
            ).tolist()
        )
    else:
        nominal = sorted({float(value) for value in nominal_fractions})
    if not nominal or min(nominal) < 0 or max(nominal) > 1:
        raise ValueError("Nominal fractions must be non-empty and lie in [0, 1]")

    identity_columns = _available_group_columns(subset, ("dataset", "model"))
    group_columns = [*identity_columns, "acquisition", "trial"]
    rows: list[dict[str, Any]] = []
    for group_key, group in subset.groupby(group_columns, sort=False, dropna=False):
        group = group.drop_duplicates(["labelled_n", "pool_n"]).copy()
        context = _group_values(group_key, group_columns)
        for fraction in nominal:
            matched = match_nearest_checkpoint(group, fraction)
            rows.append(
                {
                    **context,
                    "nominal_fraction": float(fraction),
                    "labelled_n": int(matched["labelled_n"]),
                    "pool_n": int(matched["pool_n"]),
                    "achieved_fraction": float(matched["achieved_fraction"]),
                    "fraction_distance": float(matched["fraction_distance"]),
                }
            )
    aligned = pd.DataFrame(rows)
    aligned = (
        aligned.sort_values(
            group_columns + ["labelled_n", "fraction_distance", "nominal_fraction"],
            kind="mergesort",
        )
        .drop_duplicates(group_columns + ["labelled_n"], keep="first")
        .reset_index(drop=True)
    )

    support_group = [*identity_columns, "nominal_fraction", "acquisition"]
    support = (
        aligned.groupby(support_group, as_index=False, dropna=False)["trial"]
        .nunique()
        .rename(columns={"trial": "n_trials"})
    )
    complete = support.loc[support["n_trials"].eq(required_trials)].copy()
    level_group = [*identity_columns, "nominal_fraction"]
    complete_levels = (
        complete.groupby(level_group, as_index=False, dropna=False)["acquisition"]
        .nunique()
        .loc[lambda frame: frame["acquisition"].eq(len(acquisitions)), level_group]
    )
    if complete_levels.empty:
        raise ValueError(
            f"No labelled-fraction level has {required_trials}-trial support for all rules"
        )
    aligned = aligned.merge(
        complete_levels,
        on=level_group,
        how="inner",
        validate="many_to_one",
    )
    plot_positions = (
        aligned.groupby(level_group, as_index=False, dropna=False)["achieved_fraction"]
        .median()
        .rename(columns={"achieved_fraction": "plot_fraction"})
    )
    aligned = aligned.merge(
        plot_positions,
        on=level_group,
        how="left",
        validate="many_to_one",
    )
    return aligned.sort_values(
        [*identity_columns, "nominal_fraction", "acquisition", "trial"],
        kind="mergesort",
    ).reset_index(drop=True)


def truncate_fraction_trajectory(
    frame: pd.DataFrame,
    target_fraction: float,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    """Keep each group through its last checkpoint not exceeding the target."""

    group_columns = list(group_columns)
    _require_columns(
        frame,
        [*group_columns, "plot_fraction"],
        "fraction-trajectory table",
    )
    target = float(target_fraction)
    if not np.isfinite(target) or not 0.0 <= target <= 1.0:
        raise ValueError("target_fraction must be finite and lie in [0, 1]")
    data = frame.copy()
    data["plot_fraction"] = pd.to_numeric(data["plot_fraction"], errors="raise")
    if not np.isfinite(data["plot_fraction"]).all():
        raise ValueError("plot_fraction values must be finite")
    retained = []
    for group_key, group in _iter_groups(data, group_columns):
        eligible = group.loc[
            group["plot_fraction"].le(target + FRACTION_ALIGNMENT_ATOL)
        ]
        if eligible.empty:
            raise ValueError(
                "No checkpoint lies at or below the target fraction for "
                f"{_group_values(group_key, group_columns)}"
            )
        endpoint = float(eligible["plot_fraction"].max())
        subset = group.loc[
            group["plot_fraction"].le(endpoint + FRACTION_ALIGNMENT_ATOL)
        ].copy()
        subset["panel_c_endpoint_fraction"] = endpoint
        retained.append(subset)
    return pd.concat(retained, axis=0).sort_index(kind="mergesort")


def summarise_mape_convergence(
    errors: pd.DataFrame,
    alignment: pd.DataFrame,
    required_trials: int = REQUIRED_TRIALS,
) -> pd.DataFrame:
    """Summarise hold-out MAPE within trial, then aggregate across trials."""

    _require_columns(errors, ["trial", "acquisition", "labelled_n", "ape"], "MAPE error table")
    _require_columns(
        alignment,
        [
            "trial",
            "acquisition",
            "labelled_n",
            "nominal_fraction",
            "plot_fraction",
            "achieved_fraction",
        ],
        "MAPE alignment table",
    )
    merge_keys = _available_group_columns(
        errors,
        ("dataset", "model", "role", "acquisition", "trial", "labelled_n"),
    )
    for required in ("acquisition", "trial", "labelled_n"):
        if required not in merge_keys:
            merge_keys.append(required)
    alignment_columns = [
        *merge_keys,
        "nominal_fraction",
        "plot_fraction",
        "achieved_fraction",
    ]
    aligned = errors.merge(
        alignment[alignment_columns].drop_duplicates(),
        on=merge_keys,
        how="inner",
        validate="many_to_one",
    )
    if aligned.empty:
        raise ValueError("No prediction errors match the MAPE convergence alignment")
    trial_group = [
        *_available_group_columns(aligned, ("dataset", "model", "role")),
        "acquisition",
        "trial",
        "nominal_fraction",
        "plot_fraction",
        "achieved_fraction",
        "labelled_n",
    ]
    trial_mape = (
        aligned.groupby(trial_group, as_index=False, dropna=False)["ape"]
        .mean()
        .rename(columns={"ape": "trial_mape"})
    )
    summary_group = [
        *_available_group_columns(trial_mape, ("dataset", "model", "role")),
        "acquisition",
        "nominal_fraction",
        "plot_fraction",
    ]
    summary = (
        trial_mape.groupby(summary_group, as_index=False, dropna=False)
        .agg(
            mape_mean=("trial_mape", "mean"),
            mape_min=("trial_mape", "min"),
            mape_max=("trial_mape", "max"),
            mape_sd=("trial_mape", "std"),
            n_trials=("trial", "nunique"),
            labelled_n=("labelled_n", "median"),
            achieved_fraction=("achieved_fraction", "median"),
        )
    )
    if not summary["n_trials"].eq(required_trials).all():
        observed = summary.loc[
            summary["n_trials"].ne(required_trials),
            [*summary_group, "n_trials"],
        ].to_dict("records")
        raise ValueError(
            f"MAPE convergence requires {required_trials} trials per curve point; "
            f"observed {observed[:5]}"
        )
    summary["mape_sem"] = summary["mape_sd"] / np.sqrt(summary["n_trials"])
    summary["labelled_n"] = summary["labelled_n"].round().astype(int)
    return summary.sort_values(
        [*_available_group_columns(summary, ("dataset", "model", "role")), "plot_fraction"],
        kind="mergesort",
    ).reset_index(drop=True)


def truncate_checkpoints_at_nfp(
    checkpoints: pd.DataFrame,
    crossings: pd.DataFrame,
) -> pd.DataFrame:
    """Keep each trial/rule through its first observed post-NFP checkpoint."""

    _require_columns(
        checkpoints,
        ["trial", "acquisition", "labelled_n", "pool_n"],
        "checkpoint truncation table",
    )
    _require_columns(
        crossings,
        ["trial", "acquisition", "continuous_labelled_n"],
        "NFP crossing table",
    )
    checkpoint_data = checkpoints.copy()
    crossing_data = crossings.copy()
    crossing_data["continuous_labelled_n"] = pd.to_numeric(
        crossing_data["continuous_labelled_n"], errors="coerce"
    )
    crossing_data = crossing_data.loc[
        np.isfinite(crossing_data["continuous_labelled_n"])
    ].copy()
    merge_keys = [
        column
        for column in ("dataset", "model", "feature_set", "trial", "acquisition")
        if column in checkpoint_data.columns and column in crossing_data.columns
    ]
    for mandatory in ("trial", "acquisition"):
        if mandatory not in merge_keys:
            merge_keys.append(mandatory)
    crossing_columns = [*merge_keys, "continuous_labelled_n"]
    crossing_data = crossing_data[crossing_columns].drop_duplicates()
    if crossing_data.duplicated(merge_keys).any():
        raise ValueError("NFP crossing table contains multiple crossings per trial/rule")
    merged = checkpoint_data.merge(
        crossing_data.rename(
            columns={"continuous_labelled_n": "nfp_continuous_labelled_n"}
        ),
        on=merge_keys,
        how="inner",
        validate="many_to_one",
    )
    if merged.empty:
        raise ValueError("No checkpoint sequence has a valid NFP crossing")

    frames: list[pd.DataFrame] = []
    for group_key, group in merged.groupby(merge_keys, sort=False, dropna=False):
        group = group.drop_duplicates(["labelled_n", "pool_n"]).copy()
        pool_values = group["pool_n"].dropna().unique()
        if len(pool_values) != 1 or float(pool_values[0]) <= 0:
            raise ValueError(
                f"Checkpoint group {_group_values(group_key, merge_keys)} "
                "must contain one positive pool_n"
            )
        crossing = float(group["nfp_continuous_labelled_n"].iloc[0])
        labelled = pd.to_numeric(group["labelled_n"], errors="coerce")
        if not np.isfinite(labelled).all():
            raise ValueError(
                f"Checkpoint group {_group_values(group_key, merge_keys)} "
                "contains a non-finite labelled_n"
            )
        if crossing < float(labelled.min()) or crossing > float(labelled.max()):
            continue
        stop_candidates = group.loc[
            labelled.ge(crossing)
        ].copy()
        stop = int(stop_candidates["labelled_n"].min())
        retained = group.loc[group["labelled_n"].le(stop)].copy()
        retained["nfp_stop_labelled_n"] = stop
        retained["nfp_stop_fraction"] = stop / float(pool_values[0])
        frames.append(retained)
    if not frames:
        raise ValueError("No NFP crossing lies within its observed checkpoint range")
    return (
        pd.concat(frames, ignore_index=True, sort=False)
        .sort_values([*merge_keys, "labelled_n"], kind="mergesort")
        .reset_index(drop=True)
    )


def variable_support_fraction_alignment(
    checkpoints: pd.DataFrame,
    nominal_fractions: Sequence[float],
) -> pd.DataFrame:
    """Align truncated sequences without forcing common rule/trial endpoints."""

    _require_columns(
        checkpoints,
        [
            "trial",
            "acquisition",
            "labelled_n",
            "pool_n",
            "nfp_continuous_labelled_n",
            "nfp_stop_labelled_n",
            "nfp_stop_fraction",
        ],
        "variable-support fraction-alignment table",
    )
    nominal = sorted({float(value) for value in nominal_fractions})
    if not nominal or min(nominal) < 0 or max(nominal) > 1:
        raise ValueError("Nominal fractions must be non-empty and lie in [0, 1]")
    identity_columns = _available_group_columns(
        checkpoints,
        ("dataset", "model", "feature_set", "role"),
    )
    group_columns = [*identity_columns, "acquisition", "trial"]
    rows: list[dict[str, Any]] = []
    for group_key, group in checkpoints.groupby(
        group_columns, sort=False, dropna=False
    ):
        group = group.drop_duplicates(["labelled_n", "pool_n"]).copy()
        context = _group_values(group_key, group_columns)
        stop_fraction = float(group["nfp_stop_fraction"].iloc[0])
        stop_labelled_n = int(group["nfp_stop_labelled_n"].iloc[0])
        continuous = float(group["nfp_continuous_labelled_n"].iloc[0])
        for fraction in nominal:
            if fraction > stop_fraction and not np.isclose(
                fraction,
                stop_fraction,
                rtol=0.0,
                atol=FRACTION_ALIGNMENT_ATOL,
            ):
                continue
            matched = match_nearest_checkpoint(group, fraction)
            rows.append(
                {
                    **context,
                    "nominal_fraction": float(fraction),
                    "labelled_n": int(matched["labelled_n"]),
                    "pool_n": int(matched["pool_n"]),
                    "achieved_fraction": float(matched["achieved_fraction"]),
                    "fraction_distance": float(matched["fraction_distance"]),
                    "nfp_continuous_labelled_n": continuous,
                    "nfp_stop_labelled_n": stop_labelled_n,
                    "nfp_stop_fraction": stop_fraction,
                }
            )
    if not rows:
        raise ValueError("No truncated checkpoint can be aligned to the nominal grid")
    aligned = pd.DataFrame(rows)
    aligned = (
        aligned.sort_values(
            group_columns + ["labelled_n", "fraction_distance", "nominal_fraction"],
            kind="mergesort",
        )
        .drop_duplicates(group_columns + ["labelled_n"], keep="first")
        .reset_index(drop=True)
    )
    endpoint_check = aligned.groupby(group_columns, dropna=False).agg(
        maximum_labelled_n=("labelled_n", "max"),
        expected_stop_labelled_n=("nfp_stop_labelled_n", "first"),
    )
    if not endpoint_check["maximum_labelled_n"].eq(
        endpoint_check["expected_stop_labelled_n"]
    ).all():
        raise ValueError("Fraction alignment failed to retain an NFP stop checkpoint")
    support_group = [*identity_columns, "nominal_fraction", "acquisition"]
    support = (
        aligned.groupby(support_group, as_index=False, dropna=False)["trial"]
        .nunique()
        .rename(columns={"trial": "n_trials"})
    )
    aligned = aligned.merge(
        support,
        on=support_group,
        how="left",
        validate="many_to_one",
    )
    plot_identity = _available_group_columns(aligned, ("dataset", "model"))
    level_group = [*plot_identity, "nominal_fraction"]
    plot_positions = (
        aligned.groupby(level_group, as_index=False, dropna=False)["achieved_fraction"]
        .median()
        .rename(columns={"achieved_fraction": "plot_fraction"})
    )
    aligned = aligned.merge(
        plot_positions,
        on=level_group,
        how="left",
        validate="many_to_one",
    )
    return aligned.sort_values(
        [*identity_columns, "nominal_fraction", "acquisition", "trial"],
        kind="mergesort",
    ).reset_index(drop=True)


def robust_common_bandwidth(values: np.ndarray, x_span: float) -> float:
    """Return a deterministic robust Gaussian-KDE bandwidth with a span floor."""

    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        raise ValueError("Bandwidth calculation requires at least one finite value")
    span = float(x_span)
    if not np.isfinite(span) or span <= 0:
        raise ValueError("x_span must be finite and positive")
    std = float(np.std(clean, ddof=1)) if clean.size > 1 else 0.0
    q25, q75 = np.quantile(clean, [0.25, 0.75])
    robust_sigma = float((q75 - q25) / 1.349)
    positive_scales = [value for value in (std, robust_sigma) if value > 0]
    sigma = min(positive_scales) if positive_scales else 0.0
    rule = 0.9 * sigma * clean.size ** (-0.2) if sigma > 0 else 0.0
    return float(max(rule, span * 0.01, np.finfo(float).eps))


def trial_balanced_signed_kde(
    errors: pd.DataFrame,
    x_grid: np.ndarray,
    bandwidth: float,
    required_trials: int | None = REQUIRED_TRIALS,
) -> pd.DataFrame:
    """Average fixed-bandwidth signed-error KDEs with equal trial weights."""

    _require_columns(errors, ["trial", "signed_error"], "signed-error table")
    grid = np.asarray(x_grid, dtype=float)
    if grid.ndim != 1 or grid.size == 0 or not np.isfinite(grid).all():
        raise ValueError("KDE x_grid must be a non-empty finite one-dimensional array")
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("KDE bandwidth must be finite and positive")
    group_columns = _available_group_columns(
        errors,
        (
            "dataset",
            "model",
            "role",
            "acquisition",
            "nominal_fraction",
            "plot_fraction",
        ),
    )
    rows: list[dict[str, Any]] = []
    normalizer = float(bandwidth) * np.sqrt(2.0 * np.pi)
    for group_key, group in _iter_groups(errors, group_columns):
        trial_ids = sorted(group["trial"].dropna().unique().tolist())
        if not trial_ids:
            raise ValueError(
                f"KDE group {_group_values(group_key, group_columns)} has no trials"
            )
        if required_trials is not None and len(trial_ids) != required_trials:
            raise ValueError(
                f"KDE group {_group_values(group_key, group_columns)} requires "
                f"{required_trials} trials; observed {len(trial_ids)}"
            )
        densities = []
        for trial in trial_ids:
            values = group.loc[group["trial"].eq(trial), "signed_error"].to_numpy(float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                raise ValueError(f"KDE trial {trial} has no finite signed-error values")
            scaled = (grid[:, None] - values[None, :]) / float(bandwidth)
            density = np.exp(-0.5 * scaled**2).mean(axis=1) / normalizer
            densities.append(density)
        matrix = np.vstack(densities)
        sem = (
            matrix.std(axis=0, ddof=1) / np.sqrt(matrix.shape[0])
            if matrix.shape[0] > 1
            else np.zeros(grid.size, dtype=float)
        )
        context = _group_values(group_key, group_columns)
        for x_value, mean_value, sem_value in zip(
            grid,
            matrix.mean(axis=0),
            sem,
            strict=True,
        ):
            rows.append(
                {
                    **context,
                    "signed_error_grid": float(x_value),
                    "density_mean": float(mean_value),
                    "density_sem": float(sem_value),
                    "bandwidth": float(bandwidth),
                    "n_trials": int(matrix.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def pooled_signed_kde(
    errors: pd.DataFrame,
    x_grid: np.ndarray,
    bandwidth: float,
    required_trials: int | None = None,
) -> pd.DataFrame:
    """Fit one KDE to all finite hold-out errors pooled across trials."""

    _require_columns(errors, ["trial", "signed_error"], "signed-error table")
    grid = np.asarray(x_grid, dtype=float)
    if grid.ndim != 1 or grid.size == 0 or not np.isfinite(grid).all():
        raise ValueError("KDE x_grid must be a non-empty finite one-dimensional array")
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("KDE bandwidth must be finite and positive")
    group_columns = _available_group_columns(
        errors,
        (
            "dataset",
            "model",
            "role",
            "acquisition",
            "nominal_fraction",
            "plot_fraction",
        ),
    )
    rows: list[dict[str, Any]] = []
    normalizer = float(bandwidth) * np.sqrt(2.0 * np.pi)
    for group_key, group in _iter_groups(errors, group_columns):
        finite = group.loc[
            np.isfinite(pd.to_numeric(group["signed_error"], errors="coerce"))
        ].copy()
        trial_ids = sorted(finite["trial"].dropna().unique().tolist())
        if not trial_ids:
            raise ValueError(
                f"KDE group {_group_values(group_key, group_columns)} has no trials"
            )
        if required_trials is not None and len(trial_ids) != required_trials:
            raise ValueError(
                f"KDE group {_group_values(group_key, group_columns)} requires "
                f"{required_trials} trials; observed {len(trial_ids)}"
            )
        values = finite["signed_error"].to_numpy(float)
        scaled = (grid[:, None] - values[None, :]) / float(bandwidth)
        density = np.exp(-0.5 * scaled**2).mean(axis=1) / normalizer
        context = _group_values(group_key, group_columns)
        for x_value, density_value in zip(grid, density, strict=True):
            rows.append(
                {
                    **context,
                    "signed_error_grid": float(x_value),
                    "density_mean": float(density_value),
                    "density_sem": 0.0,
                    "bandwidth": float(bandwidth),
                    "n_trials": int(len(trial_ids)),
                    "n_holdout_samples": int(values.size),
                }
            )
    return pd.DataFrame(rows)


def _parse_retained_feature_indices(value: Any, n_available: int) -> list[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return list(range(n_available))
    text = str(value).strip()
    if not text:
        return list(range(n_available))
    retained = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not retained:
        return list(range(n_available))
    if min(retained) < 0 or max(retained) >= n_available:
        raise ValueError(
            f"Retained feature indices {retained} exceed {n_available} available features"
        )
    if len(set(retained)) != len(retained):
        raise ValueError("Retained feature indices contain duplicates")
    return retained


def _feature_columns_in_original_order(feature_table: pd.DataFrame) -> list[str]:
    numbered = []
    for column in feature_table.columns:
        match = re.fullmatch(r"(?:f|feature_?)(\d+)", str(column))
        if match:
            numbered.append((int(match.group(1)), str(column)))
    if not numbered:
        raise ValueError("Feature table contains no numbered feature columns")
    return [column for _, column in sorted(numbered)]


def _selected_indices_at_checkpoint(
    selected_table: pd.DataFrame,
    trial: int,
    acquisition: str,
    operational_n: int,
) -> set[int]:
    subset = selected_table.loc[
        selected_table["trial"].eq(trial)
        & selected_table["acquisition"].eq(acquisition)
        & selected_table["budget_total"].eq(int(operational_n))
        & selected_table["selection_order"].le(int(operational_n))
    ].copy()
    if subset.empty:
        raise ValueError(
            f"No selected cells found for trial={trial}, acquisition={acquisition}, "
            f"operational_n={operational_n}"
        )
    return set(subset["selected_index"].astype(int).tolist())


def prepare_embedding_table(
    feature_table: pd.DataFrame,
    split_table: pd.DataFrame,
    selected_table: pd.DataFrame,
    audit_table: pd.DataFrame,
    *,
    reference_trial: int,
    best_trial: int,
    worst_trial: int,
    best_rule: str,
    worst_rule: str,
    best_operational_n: int,
    worst_operational_n: int,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one full-cell t-SNE background with rule-specific trial overlays."""

    _require_columns(feature_table, ["index"], "feature table")
    _require_columns(
        split_table,
        ["trial", "index", "split_assignment"],
        "split table",
    )
    _require_columns(
        selected_table,
        [
            "trial",
            "acquisition",
            "budget_total",
            "selection_order",
            "selected_index",
        ],
        "selected-cell table",
    )
    _require_columns(
        audit_table,
        ["trial", "retained_feature_indices"],
        "feature-processing audit",
    )
    if best_rule == worst_rule or RANDOM_ACQUISITION in {best_rule, worst_rule}:
        raise ValueError("Embedding Best and Worst must be distinct non-random rules")

    def trial_assignments(trial: int, column: str) -> pd.DataFrame:
        split = split_table.loc[split_table["trial"].eq(trial)].copy()
        if split.empty or split["index"].duplicated().any():
            raise ValueError(f"Trial {trial} split rows must be non-empty and unique by index")
        return split[["index", "split_assignment"]].rename(
            columns={"split_assignment": column}
        )

    reference_assignments = trial_assignments(
        reference_trial, "reference_split_assignment"
    )
    best_assignments = trial_assignments(best_trial, "best_split_assignment")
    worst_assignments = trial_assignments(worst_trial, "worst_split_assignment")
    eligible_indices = set(
        best_assignments.loc[
            best_assignments["best_split_assignment"].isin(["pool", "holdout"]),
            "index",
        ].astype(int)
    ) | set(
        worst_assignments.loc[
            worst_assignments["worst_split_assignment"].isin(["pool", "holdout"]),
            "index",
        ].astype(int)
    )
    background = feature_table.loc[
        feature_table["index"].astype(int).isin(eligible_indices)
    ].copy()
    if background["index"].duplicated().any() or set(
        background["index"].astype(int)
    ) != eligible_indices:
        missing = sorted(eligible_indices.difference(background["index"].astype(int)))
        raise ValueError(
            "Feature table must contain the union of Best/Worst pool and hold-out cells; "
            f"missing={missing}"
        )
    for assignments in (
        reference_assignments,
        best_assignments,
        worst_assignments,
    ):
        background = background.merge(
            assignments,
            on="index",
            how="left",
            validate="one_to_one",
        )
    background["split_assignment"] = background[
        "reference_split_assignment"
    ].where(
        background["reference_split_assignment"].isin(["pool", "holdout"]),
        background["best_split_assignment"].where(
            background["best_split_assignment"].isin(["pool", "holdout"]),
            background["worst_split_assignment"],
        ),
    )
    if not background["split_assignment"].isin(["pool", "holdout"]).all():
        raise ValueError("Common embedding union contains no usable split assignment")
    pool_mask = background["split_assignment"].eq("pool").to_numpy()
    reference_pool_mask = background["reference_split_assignment"].eq("pool").to_numpy()
    if not reference_pool_mask.any() or bool(pool_mask.all()):
        raise ValueError("Embedding requires both pool and hold-out cells")

    audit = audit_table.loc[audit_table["trial"].eq(reference_trial)].copy()
    if len(audit) != 1:
        raise ValueError(
            f"Expected one feature-processing audit row for trial {reference_trial}"
        )
    feature_columns = _feature_columns_in_original_order(feature_table)
    retained = _parse_retained_feature_indices(
        audit.iloc[0]["retained_feature_indices"],
        len(feature_columns),
    )
    retained_columns = [feature_columns[index] for index in retained]
    if "n_features" in audit.columns:
        audited_n = int(audit.iloc[0]["n_features"])
        if audited_n != len(retained_columns):
            raise ValueError(
                f"Audit reports {audited_n} retained features but parsed "
                f"{len(retained_columns)}"
            )

    raw_features = background[retained_columns].to_numpy(dtype=float)
    imputer = SimpleImputer(strategy="median").fit(raw_features[reference_pool_mask])
    imputed = imputer.transform(raw_features)
    scaler = StandardScaler().fit(imputed[reference_pool_mask])
    scaled = scaler.transform(imputed)
    if scaled.shape[1] == 1:
        pca_init = np.column_stack([scaled[:, 0], np.zeros(len(scaled))])
    else:
        pca_init = PCA(n_components=2, svd_solver="full").fit_transform(scaled)
    if len(background) < 4:
        raise ValueError("t-SNE embedding requires at least four cells")
    perplexity = min(30.0, max(2.0, (len(background) - 1) / 3.0))
    coordinates = TSNE(
        n_components=2,
        perplexity=perplexity,
        init=pca_init,
        learning_rate="auto",
        metric="euclidean",
        random_state=int(random_seed),
        max_iter=1000,
    ).fit_transform(scaled)
    minimum = coordinates.min(axis=0)
    span = coordinates.max(axis=0) - minimum
    span[span == 0] = 1.0
    coordinates = (coordinates - minimum) / span

    best_indices = _selected_indices_at_checkpoint(
        selected_table,
        best_trial,
        best_rule,
        int(best_operational_n),
    )
    worst_indices = _selected_indices_at_checkpoint(
        selected_table,
        worst_trial,
        worst_rule,
        int(worst_operational_n),
    )
    def own_trial_pool(trial: int) -> set[int]:
        split = split_table.loc[
            split_table["trial"].eq(trial)
            & split_table["split_assignment"].isin(["pool", "holdout"])
        ].copy()
        if split.empty or split["index"].duplicated().any():
            raise ValueError(f"Trial {trial} split rows must be non-empty and unique")
        return set(split.loc[split["split_assignment"].eq("pool"), "index"].astype(int))

    best_pool_indices = own_trial_pool(best_trial)
    worst_pool_indices = own_trial_pool(worst_trial)
    illegal_best = sorted(best_indices.difference(best_pool_indices))
    illegal_worst = sorted(worst_indices.difference(worst_pool_indices))
    if illegal_best or illegal_worst:
        raise ValueError(
            "Acquisition overlay contains hold-out cell indices in its own trial: "
            f"best={illegal_best}, worst={illegal_worst}"
        )
    missing_background = sorted((best_indices | worst_indices).difference(background["index"]))
    if missing_background:
        raise ValueError(
            f"Acquisition overlay indices are absent from the common embedding: {missing_background}"
        )
    if len(best_indices) != int(best_operational_n):
        raise ValueError(
            f"Best overlay contains {len(best_indices)} cells, expected {best_operational_n}"
        )
    if len(worst_indices) != int(worst_operational_n):
        raise ValueError(
            f"Worst overlay contains {len(worst_indices)} cells, expected {worst_operational_n}"
        )

    background = background.copy()
    background["embedding_x"] = coordinates[:, 0]
    background["embedding_y"] = coordinates[:, 1]
    background["best_selected"] = background["index"].isin(best_indices)
    background["worst_selected"] = background["index"].isin(worst_indices)
    background["best_trial_pool"] = background["index"].isin(best_pool_indices)
    background["worst_trial_pool"] = background["index"].isin(worst_pool_indices)
    background["selection_membership"] = np.select(
        [
            background["best_selected"] & background["worst_selected"],
            background["best_selected"],
            background["worst_selected"],
        ],
        ["overlap", "best_only", "worst_only"],
        default="background",
    )
    metadata = {
        "preprocessing_fit_scope": "pool_only",
        "reference_trial": int(reference_trial),
        "best_trial": int(best_trial),
        "worst_trial": int(worst_trial),
        "preprocessing_fit_n": int(reference_pool_mask.sum()),
        "n_all_cells": int(len(background)),
        "n_holdout": int((~pool_mask).sum()),
        "retained_feature_indices": retained,
        "imputer_statistics": imputer.statistics_.astype(float).tolist(),
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "tsne_random_seed": int(random_seed),
        "tsne_perplexity": float(perplexity),
    }
    return background, metadata


def select_dataset_model_configurations(
    nfp95: pd.DataFrame,
    *,
    dataset: str,
    feature_order: Sequence[str],
    model_order: Sequence[str],
    acquisition_order: Sequence[str],
    full_pool_trials: pd.DataFrame | None = None,
    required_trials: int = REQUIRED_TRIALS,
) -> pd.DataFrame:
    """Apply the feature, representative-trial, and rule selection contract."""

    _require_columns(
        nfp95,
        [
            "dataset",
            "feature_set",
            "model",
            "trial",
            "acquisition",
            "full_pool_mape",
            "attainable_gain_pp",
            "positive_attainable_gain",
            "status",
            "continuous_labelled_n",
            "operational_labelled_n",
        ],
        "NFP95 table",
    )
    data = nfp95.loc[nfp95["dataset"].eq(dataset)].copy()
    if "alpha" in data.columns:
        data = data.loc[
            np.isclose(data["alpha"].astype(float), PRIMARY_NFP_ALPHA)
        ].copy()
    if data.empty:
        raise ValueError(f"No NFP95 rows found for dataset={dataset}")
    if full_pool_trials is not None:
        _require_columns(
            full_pool_trials,
            ["dataset", "feature_set", "model", "trial", "full_pool_mape"],
            "full-pool trial reference",
        )

    rows: list[dict[str, Any]] = []
    for model in model_order:
        model_data = data.loc[data["model"].eq(model)].copy()
        if model_data.empty:
            raise ValueError(f"No NFP95 rows found for dataset={dataset}, model={model}")
        if full_pool_trials is None:
            consistency = (
                model_data.groupby(["feature_set", "trial"], as_index=False)
                .agg(
                    full_pool_min=("full_pool_mape", "min"),
                    full_pool_max=("full_pool_mape", "max"),
                )
            )
            if not np.allclose(
                consistency["full_pool_min"],
                consistency["full_pool_max"],
                equal_nan=False,
            ):
                raise ValueError(
                    f"Full-pool MAPE varies by acquisition for dataset={dataset}, model={model}"
                )
            full_pool = consistency.rename(
                columns={"full_pool_min": "full_pool_mape"}
            )[["feature_set", "trial", "full_pool_mape"]]
        else:
            full_pool = full_pool_trials.loc[
                full_pool_trials["dataset"].eq(dataset)
                & full_pool_trials["model"].eq(model),
                ["feature_set", "trial", "full_pool_mape"],
            ].copy()
        feature = select_best_feature_set(
            full_pool,
            feature_order=feature_order,
            required_trials=required_trials,
        )
        chosen = model_data.loc[
            model_data["feature_set"].eq(feature["feature_set"])
        ].copy()

        attainable_columns = [
            "trial",
            "attainable_gain_pp",
            "positive_attainable_gain",
        ]
        optional_reference = [
            column
            for column in ("cold_mape", "full_pool_mape", "pool_n", "target_mape")
            if column in chosen.columns
        ]
        attainable = chosen[
            [*attainable_columns, *optional_reference]
        ].drop_duplicates()
        if attainable["trial"].duplicated().any():
            raise ValueError(
                f"Attainable gain varies by acquisition for dataset={dataset}, "
                f"model={model}, feature_set={feature['feature_set']}"
            )
        representative = select_representative_trial(attainable)
        trial = int(representative["trial"])
        crossings = chosen.loc[chosen["trial"].eq(trial)].copy()
        ranking = rank_best_worst_rules(
            crossings,
            acquisition_order=acquisition_order,
            alpha=PRIMARY_NFP_ALPHA,
        )
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "feature_set": feature["feature_set"],
                "mean_full_pool_mape": feature["mean_full_pool_mape"],
                "n_feature_trials": feature["n_trials"],
                "representative_trial": trial,
                "representative_attainable_gain_pp": float(
                    representative["attainable_gain_pp"]
                ),
                **{
                    key: representative[key]
                    for key in ("cold_mape", "full_pool_mape", "pool_n", "target_mape")
                    if key in representative
                },
                **ranking,
            }
        )
    result = pd.DataFrame(rows)
    model_lookup = {model: index for index, model in enumerate(model_order)}
    result["model_order"] = result["model"].map(model_lookup)
    return result.sort_values("model_order").drop(columns="model_order").reset_index(drop=True)


def prediction_error_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Add APE, signed percentage error, and labelled fraction to predictions."""

    data = predictions.copy()
    rename = {}
    if "method" in data.columns and "acquisition" not in data.columns:
        rename["method"] = "acquisition"
    if "budget_total" in data.columns and "labelled_n" not in data.columns:
        rename["budget_total"] = "labelled_n"
    if "max_budget_mean" in data.columns and "pool_n" not in data.columns:
        rename["max_budget_mean"] = "pool_n"
    if "regressor" in data.columns and "model" not in data.columns:
        rename["regressor"] = "model"
    data = data.rename(columns=rename)
    _require_columns(
        data,
        [
            "trial",
            "acquisition",
            "labelled_n",
            "pool_n",
            "lifetime",
            "pred_lifetime",
        ],
        "prediction table",
    )
    numeric_columns = ["labelled_n", "pool_n", "lifetime", "pred_lifetime"]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if data["lifetime"].le(0).any():
        examples = data.loc[data["lifetime"].le(0), "lifetime"].head().tolist()
        raise ValueError(f"Prediction rows require positive lifetime values; found {examples}")
    if data["pool_n"].le(0).any():
        raise ValueError("Prediction rows require positive pool_n values")
    residual = data["pred_lifetime"] - data["lifetime"]
    data["ape"] = 100.0 * residual.abs() / data["lifetime"]
    data["signed_error"] = 100.0 * residual / data["lifetime"]
    data["labelled_fraction"] = data["labelled_n"] / data["pool_n"]
    return data


@dataclass
class DatasetProfileBundle:
    dataset: str
    model_configs: pd.DataFrame
    embedding_source: pd.DataFrame
    mape_source: pd.DataFrame
    kde_source: pd.DataFrame
    fraction_matches: pd.DataFrame
    source_data: pd.DataFrame
    manifest: dict[str, Any]


def _full_pool_source_path(output_root: Path) -> Path:
    path = (
        Path(output_root)
        / "Section 1-Full-Pool Headroom and Initialisation Effects"
        / "source_data"
        / "fig_full_pool_primary_vs_deep_violin_trial_level.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Required analysis source does not exist: {path}")
    return path


def load_full_pool_reference(output_root: Path) -> tuple[pd.DataFrame, Path]:
    """Load the full-pool trial table without requiring any NFP source."""

    path = _full_pool_source_path(output_root)
    return pd.read_csv(path, low_memory=False), path


def _analysis_source_paths(output_root: Path) -> tuple[Path, Path]:
    full_pool_path = _full_pool_source_path(output_root)
    nfp_path = (
        Path(output_root)
        / "Section 4-Sample and Time Efficiency"
        / "source_data"
        / "fig_nfp95_required_full_life_tests.csv"
    )
    if not nfp_path.exists():
        raise FileNotFoundError(f"Required analysis source does not exist: {nfp_path}")
    return full_pool_path, nfp_path


def load_configuration_references(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the complete full-pool trials and positive-gain NFP95 crossings."""

    full_pool_path, nfp_path = _analysis_source_paths(output_root)
    full_pool = pd.read_csv(full_pool_path, low_memory=False)
    nfp95 = pd.read_csv(nfp_path, low_memory=False)
    return full_pool, nfp95


def load_relative_alc_references(
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[Path, Path]]:
    """Load audited relative-ALC configuration and trial-level source tables."""

    source_dir = (
        Path(output_root) / "Section 3-Relative ALC Profiles" / "source_data"
    )
    config_path = source_dir / "relative_alc_config.csv"
    trial_path = source_dir / "relative_alc_trial.csv"
    for path in (config_path, trial_path):
        if not path.exists():
            raise FileNotFoundError(f"Required relative-ALC source does not exist: {path}")
    return (
        pd.read_csv(config_path, low_memory=False),
        pd.read_csv(trial_path, low_memory=False),
        (config_path, trial_path),
    )


def _feature_dataset_dir(al_root: Path, feature_set: str, dataset: str) -> Path:
    if feature_set not in FEATURE_DIRECTORY:
        raise KeyError(f"Unknown feature set: {feature_set}")
    path = Path(al_root) / FEATURE_DIRECTORY[feature_set] / dataset
    if not path.exists():
        raise FileNotFoundError(f"Active-learning output directory does not exist: {path}")
    return path


def _read_predictions(path: Path) -> pd.DataFrame:
    columns = [
        "index",
        "source_index",
        "lifetime",
        "pred_lifetime",
        "trial",
        "regressor",
        "method",
        "budget_total",
    ]
    data = pd.read_csv(path, usecols=columns, low_memory=False)
    return data.rename(
        columns={
            "regressor": "model",
            "method": "acquisition",
            "budget_total": "labelled_n",
        }
    )


def _read_selected(path: Path) -> pd.DataFrame:
    columns = [
        "trial",
        "regressor",
        "method",
        "budget_total",
        "selection_order",
        "selected_index",
    ]
    data = pd.read_csv(path, usecols=columns, low_memory=False)
    return data.rename(columns={"regressor": "model", "method": "acquisition"})


def _read_split(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        usecols=[
            "trial",
            "index",
            "source_index",
            "cell_name",
            "lifetime",
            "split_assignment",
        ],
        low_memory=False,
    )


def attach_true_pool_sizes(
    predictions: pd.DataFrame,
    split_table: pd.DataFrame,
    *,
    required_trials: int = REQUIRED_TRIALS,
) -> pd.DataFrame:
    """Attach independently counted pool sizes and require a true full-pool endpoint."""

    _require_columns(
        predictions,
        ["trial", "acquisition", "labelled_n"],
        "prediction checkpoint table",
    )
    _require_columns(
        split_table,
        ["trial", "index", "split_assignment"],
        "split-assignment table",
    )
    data = predictions.copy()
    data["labelled_n"] = pd.to_numeric(data["labelled_n"], errors="raise")
    trial_counts = data.groupby("acquisition", sort=False)["trial"].nunique()
    invalid_trials = trial_counts.loc[trial_counts.ne(required_trials)]
    if not invalid_trials.empty:
        raise ValueError(
            f"Prediction acquisitions require {required_trials} unique trials; "
            f"observed {invalid_trials.to_dict()}"
        )
    pool_sizes = (
        split_table.loc[split_table["split_assignment"].eq("pool")]
        .groupby("trial", sort=False)["index"]
        .nunique()
    )
    data["pool_n"] = data["trial"].map(pool_sizes)
    if data["pool_n"].isna().any() or data["pool_n"].le(0).any():
        missing_trials = sorted(
            data.loc[data["pool_n"].isna(), "trial"].drop_duplicates().tolist()
        )
        raise ValueError(
            "Split assignments require a positive pool count for every prediction "
            f"trial; missing={missing_trials}"
        )
    endpoint = (
        data.groupby(["trial", "acquisition"], as_index=False, sort=False)
        .agg(max_labelled_n=("labelled_n", "max"), pool_n=("pool_n", "first"))
    )
    incomplete = endpoint.loc[endpoint["max_labelled_n"].ne(endpoint["pool_n"])]
    if not incomplete.empty:
        raise ValueError(
            "Every trial/rule trajectory must contain the true full-pool checkpoint; "
            f"observed {incomplete.to_dict(orient='records')}"
        )
    if data["labelled_n"].gt(data["pool_n"]).any():
        raise ValueError("Prediction checkpoint labelled_n cannot exceed true pool_n")
    data["pool_n"] = data["pool_n"].astype(int)
    return data


def _read_audit(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        usecols=[
            "trial",
            "n_features",
            "retained_feature_indices",
            "preprocessing_fit_scope",
        ],
        low_memory=False,
    )


def _raw_feature_table(
    project_root: Path,
    dataset: str,
    feature_set: str,
    split_table: pd.DataFrame,
    trials: int | Sequence[int],
) -> pd.DataFrame:
    if dataset not in DATASET_MAT_FILE:
        raise KeyError(f"Unknown dataset: {dataset}")
    data_path = Path(project_root) / "cloud_run_package - v2" / "data" / DATASET_MAT_FILE[dataset]
    if not data_path.exists():
        raise FileNotFoundError(f"Raw dataset does not exist: {data_path}")
    payload = loadmat(data_path, simplify_cells=True)
    matrix_key = FEATURE_MATRIX_KEY[feature_set]
    if matrix_key not in payload:
        raise KeyError(f"{data_path.name} does not contain {matrix_key}")
    matrix = np.asarray(payload[matrix_key], dtype=float)
    trial_ids = [int(trials)] if np.isscalar(trials) else [int(value) for value in trials]
    trial_split = split_table.loc[
        split_table["trial"].isin(trial_ids)
        & split_table["split_assignment"].isin(["pool", "holdout"])
    ].copy()
    if trial_split.empty:
        raise ValueError(
            f"No pool/hold-out split rows for dataset={dataset}, trials={trial_ids}"
        )
    identity_columns = ["index", "source_index", "cell_name", "lifetime"]
    conflicts = trial_split.groupby("index")[identity_columns[1:]].nunique(dropna=False)
    if conflicts.gt(1).any(axis=None):
        raise ValueError(
            f"Split identity varies across representative trials for dataset={dataset}"
        )
    trial_split = trial_split[identity_columns].drop_duplicates("index").copy()
    source_indices = trial_split["source_index"].astype(int).to_numpy()
    if source_indices.min() < 0 or source_indices.max() >= matrix.shape[0]:
        raise ValueError(
            f"Split source indices exceed raw feature rows for dataset={dataset}, "
            f"feature_set={feature_set}"
        )
    features = matrix[source_indices]
    table = trial_split[identity_columns].reset_index(drop=True)
    feature_frame = pd.DataFrame(
        features,
        columns=[f"f{index}" for index in range(features.shape[1])],
    )
    return pd.concat([table, feature_frame], axis=1)


def _role_mapping(config: pd.Series) -> dict[str, str]:
    return {
        str(config["best_acquisition"]): "Best",
        str(config["worst_acquisition"]): "Worst",
        str(config["random_acquisition"]): "Random",
    }


def panel_c_nfp_crossings(
    nfp95: pd.DataFrame,
    configs: pd.DataFrame,
    *,
    dataset: str,
) -> pd.DataFrame:
    """Return valid positive-gain NFP95 crossings for selected Panel c rules."""

    _require_columns(
        nfp95,
        [
            "dataset",
            "feature_set",
            "model",
            "trial",
            "acquisition",
            "positive_attainable_gain",
            "status",
            "continuous_labelled_n",
        ],
        "Panel-c NFP95 table",
    )
    _require_columns(
        configs,
        [
            "dataset",
            "feature_set",
            "model",
            "best_acquisition",
            "worst_acquisition",
            "random_acquisition",
        ],
        "Panel-c configuration table",
    )
    rows: list[pd.DataFrame] = []
    for _, config in configs.loc[configs["dataset"].eq(dataset)].iterrows():
        roles = _role_mapping(config)
        subset = nfp95.loc[
            nfp95["dataset"].eq(dataset)
            & nfp95["feature_set"].eq(str(config["feature_set"]))
            & nfp95["model"].eq(str(config["model"]))
            & nfp95["acquisition"].isin(roles)
        ].copy()
        if "alpha" in subset.columns:
            subset = subset.loc[
                np.isclose(
                    pd.to_numeric(subset["alpha"], errors="coerce"),
                    PRIMARY_NFP_ALPHA,
                )
            ].copy()
        positive = (
            subset["positive_attainable_gain"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("true")
        )
        crossed = subset["status"].astype(str).str.lower().isin(["crossed", "reached"])
        continuous = pd.to_numeric(
            subset["continuous_labelled_n"], errors="coerce"
        )
        subset = subset.loc[positive & crossed & np.isfinite(continuous)].copy()
        subset["continuous_labelled_n"] = pd.to_numeric(
            subset["continuous_labelled_n"], errors="raise"
        )
        subset["role"] = subset["acquisition"].map(roles)
        rows.append(
            subset[
                [
                    "dataset",
                    "feature_set",
                    "model",
                    "trial",
                    "role",
                    "acquisition",
                    "continuous_labelled_n",
                ]
            ]
        )
    if not rows:
        raise ValueError(f"No selected Panel-c configurations for dataset={dataset}")
    result = pd.concat(rows, ignore_index=True, sort=False).drop_duplicates()
    if result.empty:
        raise ValueError(f"No valid positive-gain NFP95 crossings for dataset={dataset}")
    duplicate_keys = ["dataset", "feature_set", "model", "trial", "acquisition"]
    if result.duplicated(duplicate_keys).any():
        raise ValueError("Panel-c NFP95 table has duplicate trial/rule crossings")
    return result.sort_values(
        ["model", "role", "acquisition", "trial"],
        kind="mergesort",
    ).reset_index(drop=True)


def _ape_grid(values: np.ndarray, n_points: int = 321) -> np.ndarray:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean) & (clean >= 0)]
    if clean.size == 0:
        raise ValueError("APE grid requires finite non-negative values")
    maximum = float(clean.max())
    if maximum <= 1.0:
        return np.linspace(0.0, max(maximum, 1e-6), n_points)
    linear = np.linspace(0.0, 1.0, max(16, n_points // 8), endpoint=False)
    logarithmic = np.geomspace(1.0, maximum, n_points - len(linear))
    return np.unique(np.concatenate([linear, logarithmic]))


def _signed_grid(values: np.ndarray, n_points: int = 401) -> np.ndarray:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        raise ValueError("Signed-error grid requires finite values")
    lower = float(clean.min())
    upper = float(clean.max())
    if np.isclose(lower, upper):
        lower -= 1.0
        upper += 1.0
    padding = 0.02 * (upper - lower)
    return np.linspace(lower - padding, upper + padding, n_points)


def _nominal_fraction_grid(checkpoints: pd.DataFrame) -> list[float]:
    sequences = []
    for _, group in checkpoints.groupby(["model", "acquisition", "trial"], sort=False):
        values = np.sort(group["labelled_fraction"].dropna().unique())
        if len(values):
            sequences.append(values)
    if not sequences:
        raise ValueError("No checkpoint sequences are available for fraction alignment")
    common_length = min(len(values) for values in sequences)
    if common_length < 2:
        raise ValueError("At least two convergence checkpoints are required")
    nominal = [
        float(np.median([sequence[index] for sequence in sequences]))
        for index in range(common_length)
    ]
    return sorted(set(np.round(nominal, 10).tolist()))


def _source_block(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    out = frame.copy()
    out.insert(0, "source_block", name)
    return out


def build_dataset_profile_bundle(
    dataset: str,
    paths: Any,
    *,
    feature_order: Sequence[str],
    model_order: Sequence[str],
    acquisition_order: Sequence[str],
    random_seed: int = 42,
    required_trials: int = REQUIRED_TRIALS,
) -> DatasetProfileBundle:
    """Build all source data required by panels a-c for one dataset."""

    full_pool, full_pool_path = load_full_pool_reference(paths.output_root)
    relative_config, relative_trial, relative_paths = load_relative_alc_references(
        paths.output_root
    )
    configs = select_relative_alc_configurations(
        full_pool,
        relative_config,
        relative_trial,
        dataset=dataset,
        feature_order=feature_order,
        model_order=model_order,
        acquisition_order=acquisition_order,
        required_trials=required_trials,
    )

    prediction_cache: dict[str, pd.DataFrame] = {}
    selected_cache: dict[str, pd.DataFrame] = {}
    split_cache: dict[str, pd.DataFrame] = {}
    audit_cache: dict[str, pd.DataFrame] = {}
    embedding_frames = []
    error_frames = []
    embedding_metadata: dict[str, Any] = {}
    input_paths: set[str] = {str(full_pool_path), *(str(path) for path in relative_paths)}

    for model_position, config in configs.iterrows():
        model = str(config["model"])
        feature_set = str(config["feature_set"])
        data_dir = _feature_dataset_dir(paths.al_root, feature_set, dataset)
        prediction_path = data_dir / "predictions.csv"
        selected_path = data_dir / "selected_indices.csv"
        split_path = data_dir / "split_assignments.csv"
        audit_path = data_dir / "feature_processing_audit.csv"
        input_paths.update(map(str, (prediction_path, selected_path, split_path, audit_path)))
        if feature_set not in prediction_cache:
            prediction_cache[feature_set] = _read_predictions(prediction_path)
            selected_cache[feature_set] = _read_selected(selected_path)
            split_cache[feature_set] = _read_split(split_path)
            audit_cache[feature_set] = _read_audit(audit_path)

        roles = _role_mapping(config)
        acquisitions = list(roles)
        model_predictions = prediction_cache[feature_set].loc[
            prediction_cache[feature_set]["model"].eq(model)
            & prediction_cache[feature_set]["acquisition"].isin(acquisitions)
        ].copy()
        observed_trials = model_predictions["trial"].nunique()
        if observed_trials != required_trials:
            raise ValueError(
                f"Predictions require {required_trials} trials for dataset={dataset}, "
                f"model={model}; observed {observed_trials}"
            )
        model_predictions = attach_true_pool_sizes(
            model_predictions,
            split_cache[feature_set],
            required_trials=required_trials,
        )
        model_predictions["role"] = model_predictions["acquisition"].map(roles)
        errors = prediction_error_table(model_predictions)
        errors["dataset"] = dataset
        errors["feature_set"] = feature_set
        error_frames.append(errors)

        best_trial = int(config["best_representative_trial"])
        worst_trial = int(config["worst_representative_trial"])
        checkpoints = errors[
            ["trial", "acquisition", "labelled_n", "pool_n"]
        ].drop_duplicates()
        best_match = match_nearest_checkpoint(
            checkpoints.loc[
                checkpoints["trial"].eq(best_trial)
                & checkpoints["acquisition"].eq(str(config["best_acquisition"]))
            ],
            EMBEDDING_TARGET_FRACTION,
        )
        worst_match = match_nearest_checkpoint(
            checkpoints.loc[
                checkpoints["trial"].eq(worst_trial)
                & checkpoints["acquisition"].eq(str(config["worst_acquisition"]))
            ],
            EMBEDDING_TARGET_FRACTION,
        )
        if int(best_match["labelled_n"]) != int(worst_match["labelled_n"]) or not np.isclose(
            float(best_match["achieved_fraction"]),
            float(worst_match["achieved_fraction"]),
            rtol=0.0,
            atol=FRACTION_ALIGNMENT_ATOL,
        ):
            raise ValueError(
                f"Panel-a Best/Worst checkpoints differ at ~30% for dataset={dataset}, "
                f"model={model}: best={best_match}, worst={worst_match}"
            )
        configs.loc[model_position, "panel_a_labelled_n"] = int(
            best_match["labelled_n"]
        )
        configs.loc[model_position, "panel_a_labelled_fraction"] = float(
            best_match["achieved_fraction"]
        )
        configs.loc[model_position, "best_operational_labelled_n"] = int(
            best_match["labelled_n"]
        )
        configs.loc[model_position, "worst_operational_labelled_n"] = int(
            worst_match["labelled_n"]
        )
        configs.loc[model_position, "embedding_reference_trial"] = best_trial
        model_selected = selected_cache[feature_set].loc[
            selected_cache[feature_set]["model"].eq(model)
            & (
                (
                    selected_cache[feature_set]["trial"].eq(best_trial)
                    & selected_cache[feature_set]["acquisition"].eq(
                        str(config["best_acquisition"])
                    )
                )
                | (
                    selected_cache[feature_set]["trial"].eq(worst_trial)
                    & selected_cache[feature_set]["acquisition"].eq(
                        str(config["worst_acquisition"])
                    )
                )
            )
        ].copy()
        raw_features = _raw_feature_table(
            paths.project_root,
            dataset,
            feature_set,
            split_cache[feature_set],
            [best_trial, worst_trial],
        )
        embedding, metadata = prepare_embedding_table(
            raw_features,
            split_cache[feature_set],
            model_selected,
            audit_cache[feature_set],
            reference_trial=best_trial,
            best_trial=best_trial,
            worst_trial=worst_trial,
            best_rule=str(config["best_acquisition"]),
            worst_rule=str(config["worst_acquisition"]),
            best_operational_n=int(best_match["labelled_n"]),
            worst_operational_n=int(worst_match["labelled_n"]),
            random_seed=int(random_seed + model_position),
        )
        embedding.insert(0, "dataset", dataset)
        embedding.insert(1, "model", model)
        embedding.insert(2, "feature_set", feature_set)
        embedding.insert(3, "best_representative_trial", best_trial)
        embedding.insert(4, "worst_representative_trial", worst_trial)
        embedding.insert(5, "panel_a_labelled_fraction", float(best_match["achieved_fraction"]))
        embedding.insert(6, "panel_a_labelled_n", int(best_match["labelled_n"]))
        embedding["best_acquisition"] = str(config["best_acquisition"])
        embedding["worst_acquisition"] = str(config["worst_acquisition"])
        embedding = embedding[
            [
                "dataset",
                "model",
                "feature_set",
                "best_representative_trial",
                "worst_representative_trial",
                "panel_a_labelled_fraction",
                "panel_a_labelled_n",
                "index",
                "source_index",
                "cell_name",
                "lifetime",
                "split_assignment",
                "embedding_x",
                "embedding_y",
                "best_selected",
                "worst_selected",
                "best_trial_pool",
                "worst_trial_pool",
                "selection_membership",
                "best_acquisition",
                "worst_acquisition",
            ]
        ].copy()
        embedding_frames.append(embedding)
        embedding_metadata[model] = metadata

    all_errors = pd.concat(error_frames, ignore_index=True, sort=False)
    embedding_source = pd.concat(embedding_frames, ignore_index=True, sort=False)

    alignment_frames = []
    mape_frames = []
    kde_frames = []
    global_signed_grid = _signed_grid(all_errors["signed_error"].to_numpy(float))
    global_span = float(global_signed_grid[-1] - global_signed_grid[0])
    for _, config in configs.iterrows():
        model = str(config["model"])
        roles = _role_mapping(config)
        model_errors = all_errors.loc[all_errors["model"].eq(model)].copy()
        model_checkpoints = model_errors.loc[
            model_errors["acquisition"].isin(roles),
            ["dataset", "model", "trial", "acquisition", "labelled_n", "pool_n"],
        ].drop_duplicates()
        alignment = common_fraction_alignment(
            model_checkpoints,
            acquisitions=list(roles),
            required_trials=required_trials,
        )
        alignment["role"] = alignment["acquisition"].map(roles)
        alignment["panel"] = "b"
        panel_c_alignment = truncate_fraction_trajectory(
            alignment,
            target_fraction=PANEL_C_MAX_LABELLED_FRACTION,
            group_columns=["dataset", "model"],
        )
        panel_c_alignment["panel"] = "c"
        alignment_frames.extend([alignment, panel_c_alignment])
        configs.loc[
            configs["model"].eq(model), "panel_c_endpoint_labelled_fraction"
        ] = float(panel_c_alignment["panel_c_endpoint_fraction"].iloc[0])
        mape_frames.append(
            summarise_mape_convergence(
                model_errors,
                alignment,
                required_trials=required_trials,
            )
        )
        aligned_errors = model_errors.merge(
            panel_c_alignment[
                [
                    "dataset",
                    "model",
                    "trial",
                    "role",
                    "acquisition",
                    "labelled_n",
                    "nominal_fraction",
                    "plot_fraction",
                    "achieved_fraction",
                ]
            ],
            on=["dataset", "model", "trial", "role", "acquisition", "labelled_n"],
            how="inner",
            validate="many_to_many",
        )
        bandwidth = robust_common_bandwidth(
            model_errors["signed_error"].to_numpy(float),
            x_span=global_span,
        )
        kde_frames.append(
            pooled_signed_kde(
                aligned_errors,
                x_grid=global_signed_grid,
                bandwidth=bandwidth,
                required_trials=required_trials,
            )
        )
    fraction_matches = pd.concat(alignment_frames, ignore_index=True, sort=False)
    mape_source = pd.concat(mape_frames, ignore_index=True, sort=False)
    kde_source = pd.concat(kde_frames, ignore_index=True, sort=False)

    source_data = pd.concat(
        [
            _source_block(configs, "configuration"),
            _source_block(embedding_source, "embedding"),
            _source_block(mape_source, "mape_convergence"),
            _source_block(kde_source, "kde"),
            _source_block(fraction_matches, "fraction_match"),
        ],
        ignore_index=True,
        sort=False,
    )
    manifest = {
        "dataset": dataset,
        "error_metric": "MAPE",
        "required_trials": int(required_trials),
        "panel_a_target_labelled_fraction": EMBEDDING_TARGET_FRACTION,
        "model_order": list(model_order),
        "model_configurations": configs.to_dict("records"),
        "embedding_preprocessing": embedding_metadata,
        "rule_ranking": (
            "Best/Worst are the maximum/minimum median relative ALC improvement "
            "across ten trials; Random is an unranked comparator"
        ),
        "panel_a_representative_trial_policy": (
            "separately for Best and Worst, choose the trial closest to that "
            "rule's ten-trial median relative ALC improvement; ties use trial order"
        ),
        "panel_b_aggregation": (
            "hold-out MAPE within trial and checkpoint, followed by arithmetic "
            "mean ± standard error of the mean (SEM) across ten trials"
        ),
        "panel_b_uncertainty_alpha": 0.10,
        "kde_aggregation": (
            "one fixed-bandwidth KDE fitted to all directly pooled hold-out "
            "signed errors across ten trials at each aligned early-trajectory ridge"
        ),
        "panel_c_sample_aggregation": "direct pooled hold-out samples",
        "panel_c_uncertainty": (
            "no trial-level SEM; density_sem is zero for schema compatibility"
        ),
        "panel_c_checkpoint_policy": (
            "common trajectory through the closest checkpoint at or below 60%"
        ),
        "panel_c_max_labelled_fraction": PANEL_C_MAX_LABELLED_FRACTION,
        "panel_c_trial_support": "exactly ten trials per ridge",
        "panel_c_sample_support": "n_holdout_samples recorded per ridge",
        "signed_error_definition": "100 * (pred_lifetime - lifetime) / lifetime",
        "ape_definition": "100 * abs(pred_lifetime - lifetime) / lifetime",
        "input_paths": sorted(input_paths),
    }
    return DatasetProfileBundle(
        dataset=dataset,
        model_configs=configs,
        embedding_source=embedding_source,
        mape_source=mape_source,
        kde_source=kde_source,
        fraction_matches=fraction_matches,
        source_data=source_data,
        manifest=manifest,
    )
