from __future__ import annotations

import numpy as np
import pandas as pd


KEY_DFMR = ["dataset", "feature_set", "model", "trial"]
RANDOM = "random_selection"
DIVERSITY_METHODS = ["diversity_oneshot", "diversity_iterative"]
MATCH_KEYS = KEY_DFMR + ["labelled_n"]


def _assert_constant(grouped: pd.core.groupby.SeriesGroupBy, label: str, tolerance: float = 1e-10) -> None:
    spread = grouped.max() - grouped.min()
    if bool((spread > tolerance).any()):
        examples = spread[spread > tolerance].head().to_dict()
        raise ValueError(f"Inconsistent {label} values: {examples}")


def build_full_pool(metrics: pd.DataFrame, tolerance: float = 1e-10) -> pd.DataFrame:
    full = metrics.loc[metrics["labelled_n"].eq(metrics["pool_n"])].copy()
    if full.empty:
        raise ValueError("No full-pool checkpoints were found")
    grouped = full.groupby(KEY_DFMR, dropna=False)["holdout_mape"]
    _assert_constant(grouped, "full-pool", tolerance)
    out = grouped.agg(full_pool_mape="mean", n_full_pool_copies="size").reset_index()
    return out


def build_cold_start(metrics: pd.DataFrame, tolerance: float = 1e-10) -> pd.DataFrame:
    random = metrics.loc[metrics["acquisition"].eq(RANDOM)].copy()
    min_labelled = random.groupby(KEY_DFMR, dropna=False)["labelled_n"].transform("min")
    cold_rows = random.loc[random["labelled_n"].eq(min_labelled)].copy()
    grouped = cold_rows.groupby(KEY_DFMR, dropna=False)["holdout_mape"]
    _assert_constant(grouped, "common cold-start", tolerance)
    out = (
        cold_rows.groupby(KEY_DFMR, dropna=False)
        .agg(cold_mape=("holdout_mape", "mean"), initial_labelled_n=("labelled_n", "first"), pool_n=("pool_n", "first"))
        .reset_index()
    )
    return out


def build_attainable_gain(cold: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    out = cold.merge(full, on=KEY_DFMR, how="inner", validate="one_to_one")
    out["attainable_gain_pp"] = out["cold_mape"] - out["full_pool_mape"]
    out["positive_attainable_gain"] = out["attainable_gain_pp"].gt(0)
    return out


def _first_checkpoint_rows(metrics: pd.DataFrame, acquisitions: list[str]) -> pd.DataFrame:
    subset = metrics.loc[metrics["acquisition"].isin(acquisitions)].copy()
    group = KEY_DFMR + ["acquisition"]
    minimum = subset.groupby(group, dropna=False)["labelled_n"].transform("min")
    return subset.loc[subset["labelled_n"].eq(minimum)].copy()


def _initial_id_table(selected: pd.DataFrame, acquisitions: list[str]) -> pd.DataFrame:
    subset = selected.loc[selected["acquisition"].isin(acquisitions)].copy()
    group = KEY_DFMR + ["acquisition"]
    minimum = subset.groupby(group, dropna=False)["budget_total"].transform("min")
    subset = subset.loc[subset["budget_total"].eq(minimum)].copy()
    subset = subset.sort_values(group + ["selection_order"])
    return (
        subset.groupby(group, dropna=False)
        .agg(
            selected_ids=("selected_index", lambda values: tuple(int(value) for value in values)),
            selected_count=("selected_index", "size"),
        )
        .reset_index()
    )


def build_initialisation_table(
    metrics: pd.DataFrame,
    selected: pd.DataFrame,
    tolerance: float = 1e-10,
) -> pd.DataFrame:
    acquisitions = [RANDOM, *DIVERSITY_METHODS]
    first = _first_checkpoint_rows(metrics, acquisitions)
    first_grouped = first.groupby(KEY_DFMR + ["acquisition"], dropna=False)
    first_summary = first_grouped.agg(
        initial_labelled_n=("labelled_n", "first"),
        initial_mape=("holdout_mape", "mean"),
        n_rows=("holdout_mape", "size"),
    ).reset_index()
    ids = _initial_id_table(selected, acquisitions)
    summary = first_summary.merge(ids, on=KEY_DFMR + ["acquisition"], how="left", validate="one_to_one")

    hot = summary.loc[summary["acquisition"].isin(DIVERSITY_METHODS)].copy()
    if hot["selected_ids"].isna().any():
        raise ValueError("Missing Diversity initialisation cell IDs")
    for _, group in hot.groupby(KEY_DFMR, dropna=False):
        if set(group["acquisition"]) != set(DIVERSITY_METHODS):
            raise ValueError("Diversity initialisation is incomplete")
        if group["selected_ids"].nunique() != 1:
            raise ValueError("Diversity initialisation cell IDs differ")
        if group["initial_labelled_n"].nunique() != 1:
            raise ValueError("Diversity initialisation labelled-set sizes differ")
        if float(group["initial_mape"].max() - group["initial_mape"].min()) > tolerance:
            raise ValueError("Diversity initialisation MAPE values differ")

    cold = summary.loc[summary["acquisition"].eq(RANDOM)].copy()
    cold["initialisation"] = "Random cold start"
    hot_one = summary.loc[summary["acquisition"].eq("diversity_oneshot")].copy()
    hot_one["initialisation"] = "Diversity hot start"
    out = pd.concat([cold, hot_one], ignore_index=True, sort=False)
    out["initial_cell_ids"] = out["selected_ids"].map(lambda values: "|".join(map(str, values)))
    return out[
        KEY_DFMR
        + ["initialisation", "initial_labelled_n", "initial_mape", "initial_cell_ids", "selected_count"]
    ].sort_values(KEY_DFMR + ["initialisation"]).reset_index(drop=True)


def build_initialisation_gain(initialisation: pd.DataFrame) -> pd.DataFrame:
    wide = initialisation.pivot(index=KEY_DFMR, columns="initialisation", values="initial_mape")
    required = {"Random cold start", "Diversity hot start"}
    if not required.issubset(wide.columns):
        raise ValueError("Both logical initialisation levels are required")
    out = wide.reset_index()
    out["cold_mape"] = out["Random cold start"]
    out["hot_mape"] = out["Diversity hot start"]
    out["initialisation_gain_pp"] = out["cold_mape"] - out["hot_mape"]
    out["positive_initialisation_gain"] = out["initialisation_gain_pp"].gt(0)
    return out[
        KEY_DFMR + ["cold_mape", "hot_mape", "initialisation_gain_pp", "positive_initialisation_gain"]
    ]


def compute_pointwise_gain(metrics: pd.DataFrame) -> pd.DataFrame:
    random = metrics.loc[metrics["acquisition"].eq(RANDOM), MATCH_KEYS + ["holdout_mape"]].copy()
    random = random.rename(columns={"holdout_mape": "random_mape"})
    out = metrics.merge(random, on=MATCH_KEYS, how="left", validate="many_to_one")
    if out["random_mape"].isna().any():
        missing = out.loc[out["random_mape"].isna(), MATCH_KEYS].head().to_dict("records")
        raise ValueError(f"Matched Random MAPE is missing: {missing}")
    out["active_learning_gain_pp"] = out["random_mape"] - out["holdout_mape"]
    return out


def _summarise_gain(pointwise: pd.DataFrame, early: bool) -> pd.DataFrame:
    data = pointwise.copy()
    b0 = data.groupby(KEY_DFMR, dropna=False)["labelled_n"].transform("min")
    eligible = data.loc[data["labelled_n"].gt(b0) & data["labelled_n"].lt(data["pool_n"])].copy()
    if early:
        # Shared Section 2/3 window: after the initial labelled set and up to 60% of the pool.
        eligible = eligible.loc[eligible["labelled_n"].div(eligible["pool_n"]).le(0.60)].copy()
    value_name = "early_gain_pp" if early else "overall_gain_pp"
    trial = (
        eligible.groupby(KEY_DFMR + ["acquisition"], dropna=False)
        .agg(
            trial_gain_pp=("active_learning_gain_pp", "mean"),
            n_checkpoint_values=("labelled_n", "nunique"),
        )
        .reset_index()
    )
    config = (
        trial.groupby(["dataset", "feature_set", "model", "acquisition"], dropna=False)
        .agg(
            **{
                value_name: ("trial_gain_pp", "mean"),
                "sd_trial_gain_pp": ("trial_gain_pp", "std"),
                "n_trials": ("trial", "nunique"),
                "n_checkpoint_values": ("n_checkpoint_values", "min"),
                "max_checkpoint_values": ("n_checkpoint_values", "max"),
            }
        )
        .reset_index()
    )
    return config


def summarise_overall_gain(pointwise: pd.DataFrame) -> pd.DataFrame:
    return _summarise_gain(pointwise, early=False)


def summarise_early_gain(pointwise: pd.DataFrame) -> pd.DataFrame:
    return _summarise_gain(pointwise, early=True)


def positive_configuration_fraction(summary: pd.DataFrame, value_column: str) -> pd.DataFrame:
    return (
        summary.groupby("acquisition", dropna=False)
        .agg(
            positive_configuration_fraction=(value_column, lambda values: float((values > 0).mean())),
            n_positive=(value_column, lambda values: int((values > 0).sum())),
            n_non_positive=(value_column, lambda values: int((values <= 0).sum())),
            n_configurations=(value_column, "count"),
        )
        .reset_index()
    )
