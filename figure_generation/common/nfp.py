from __future__ import annotations

import numpy as np
import pandas as pd


NFP_MATCH_KEYS = ["dataset", "feature_set", "model", "trial", "alpha"]
KEY_DFMR = ["dataset", "feature_set", "model", "trial"]
CURVE_KEYS = KEY_DFMR + ["acquisition"]


def nfp_target(cold_mape: float, full_pool_mape: float, alpha: float) -> float:
    if not 0 < alpha < 1:
        raise ValueError("NFP alpha must be between zero and one")
    if cold_mape <= full_pool_mape:
        return float("nan")
    return float(cold_mape - alpha * (cold_mape - full_pool_mape))


def crossing(curve: pd.DataFrame, target_mape: float, pool_n: int | float) -> dict[str, float | str]:
    data = curve[["labelled_n", "holdout_mape"]].dropna().sort_values("labelled_n").copy()
    data = data.groupby("labelled_n", as_index=False)["holdout_mape"].min()
    data["best_mape"] = data["holdout_mape"].cummin()
    reached = data.loc[data["best_mape"].le(target_mape)]
    if reached.empty:
        return {
            "status": "not_reached",
            "continuous_labelled_n": np.nan,
            "continuous_pool_fraction": np.nan,
            "operational_labelled_n": np.nan,
            "operational_pool_fraction": np.nan,
        }

    op_index = int(reached.index[0])
    op = data.loc[op_index]
    op_n = float(op["labelled_n"])
    continuous_n = op_n
    if op_index > 0:
        prev = data.loc[op_index - 1]
        x0, x1 = float(prev["labelled_n"]), op_n
        y0, y1 = float(prev["best_mape"]), float(op["best_mape"])
        if y1 != y0 and y0 > target_mape >= y1:
            continuous_n = x0 + (target_mape - y0) / (y1 - y0) * (x1 - x0)
    return {
        "status": "crossed",
        "continuous_labelled_n": float(continuous_n),
        "continuous_pool_fraction": float(continuous_n / float(pool_n)),
        "operational_labelled_n": int(op_n),
        "operational_pool_fraction": float(op_n / float(pool_n)),
    }


def campaign_duration(selected: pd.DataFrame) -> float:
    if selected.empty:
        return float("nan")
    return float(selected.groupby("acquisition_batch")["selected_lifetime"].max().sum())


def build_duration_table(selected: pd.DataFrame) -> pd.DataFrame:
    data = selected.loc[selected["selection_order"].le(selected["budget_total"])].copy()
    batch = (
        data.groupby(CURVE_KEYS + ["budget_total", "acquisition_batch"], dropna=False)["selected_lifetime"]
        .max()
        .rename("batch_duration")
        .reset_index()
    )
    return (
        batch.groupby(CURVE_KEYS + ["budget_total"], dropna=False)
        .agg(
            operational_duration=("batch_duration", "sum"),
            n_completed_batches=("acquisition_batch", "nunique"),
        )
        .reset_index()
    )


def build_nfp_crossings(
    metrics: pd.DataFrame,
    attainable: pd.DataFrame,
    durations: pd.DataFrame,
    alphas: list[float],
) -> pd.DataFrame:
    duration_lookup = durations.set_index(CURVE_KEYS + ["budget_total"])["operational_duration"].to_dict()
    curve_lookup = {tuple(key): group for key, group in metrics.groupby(CURVE_KEYS, dropna=False)}
    rows: list[dict[str, object]] = []
    for reference in attainable.to_dict("records"):
        reference_key = tuple(reference[key] for key in KEY_DFMR)
        acquisitions = sorted(
            {key[-1] for key in curve_lookup if tuple(key[:-1]) == reference_key}
        )
        for alpha in alphas:
            target = nfp_target(float(reference["cold_mape"]), float(reference["full_pool_mape"]), float(alpha))
            for acquisition in acquisitions:
                curve_key = (*reference_key, acquisition)
                if not bool(reference["positive_attainable_gain"]):
                    crossed = {
                        "status": "non_positive_attainable_gain",
                        "continuous_labelled_n": np.nan,
                        "continuous_pool_fraction": np.nan,
                        "operational_labelled_n": np.nan,
                        "operational_pool_fraction": np.nan,
                    }
                else:
                    crossed = crossing(curve_lookup[curve_key], target, float(reference["pool_n"]))
                op_n = crossed["operational_labelled_n"]
                duration = np.nan
                if crossed["status"] == "crossed" and np.isfinite(op_n):
                    duration = duration_lookup.get((*curve_key, int(op_n)), np.nan)
                rows.append(
                    {
                        **{key: reference[key] for key in KEY_DFMR},
                        "acquisition": acquisition,
                        "alpha": float(alpha),
                        "cold_mape": reference["cold_mape"],
                        "full_pool_mape": reference["full_pool_mape"],
                        "attainable_gain_pp": reference["attainable_gain_pp"],
                        "positive_attainable_gain": reference["positive_attainable_gain"],
                        "pool_n": reference["pool_n"],
                        "target_mape": target,
                        **crossed,
                        "operational_duration": duration,
                    }
                )
    return pd.DataFrame(rows)


def matched_savings(crossings: pd.DataFrame) -> pd.DataFrame:
    random = crossings.loc[crossings["acquisition"].eq("random_selection")].copy()
    random = random[
        NFP_MATCH_KEYS + ["status", "operational_labelled_n", "operational_duration"]
    ].rename(
        columns={
            "status": "random_status",
            "operational_labelled_n": "random_operational_labelled_n",
            "operational_duration": "random_operational_duration",
        }
    )
    nonrandom = crossings.loc[~crossings["acquisition"].eq("random_selection")].copy()
    out = nonrandom.merge(random, on=NFP_MATCH_KEYS, how="left", validate="many_to_one")
    non_reach = out["status"].eq("crossed")
    random_reach = out["random_status"].eq("crossed")
    out["attainment_outcome"] = np.select(
        [non_reach & random_reach, non_reach & ~random_reach, ~non_reach & random_reach],
        ["both_reach", "nonrandom_only", "random_only"],
        default="neither_reaches",
    )
    both = non_reach & random_reach
    out["test_saving_pct"] = np.where(
        both,
        100.0
        * (out["random_operational_labelled_n"] - out["operational_labelled_n"])
        / out["random_operational_labelled_n"],
        np.nan,
    )
    out["duration_saving_pct"] = np.where(
        both,
        100.0
        * (out["random_operational_duration"] - out["operational_duration"])
        / out["random_operational_duration"],
        np.nan,
    )
    return out
