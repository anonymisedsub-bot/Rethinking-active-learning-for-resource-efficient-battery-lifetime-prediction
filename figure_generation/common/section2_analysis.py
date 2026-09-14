from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


DFM_KEYS = ["dataset", "feature_set", "model"]


def build_feature_model_median(early: pd.DataFrame) -> pd.DataFrame:
    return (
        early.groupby(["feature_set", "model"], as_index=False, dropna=False)
        .agg(
            median_early_gain_pp=("early_gain_pp", "median"),
            n_configurations=("early_gain_pp", "size"),
        )
        .sort_values(["feature_set", "model"])
        .reset_index(drop=True)
    )


def build_moderation_table(
    early: pd.DataFrame,
    attainable: pd.DataFrame,
) -> pd.DataFrame:
    attainable_reference = attainable.loc[:, DFM_KEYS + ["attainable_gain_pp"]].copy()
    if attainable_reference.duplicated(DFM_KEYS).any():
        raise ValueError("Attainable gain rows must be unique by dataset-feature-model")

    merged = early.merge(
        attainable_reference,
        on=DFM_KEYS,
        how="left",
        validate="many_to_one",
    )
    if merged["attainable_gain_pp"].isna().any():
        raise ValueError("Every early-gain row must match an attainable-gain reference")

    attainable_mean = float(merged["attainable_gain_pp"].mean())
    merged["attainable_gain_centered"] = (
        merged["attainable_gain_pp"] - attainable_mean
    )
    full = smf.ols(
        "early_gain_pp ~ attainable_gain_centered * C(acquisition, Sum)",
        data=merged,
    ).fit()
    additive = smf.ols(
        "early_gain_pp ~ attainable_gain_centered + C(acquisition, Sum)",
        data=merged,
    ).fit()
    if additive.ssr <= np.finfo(float).eps:
        interaction_partial_r2 = 0.0
    else:
        interaction_partial_r2 = float(
            np.clip((additive.ssr - full.ssr) / additive.ssr, 0.0, 1.0)
        )

    rule_rows = []
    for acquisition in sorted(merged["acquisition"].astype(str).unique()):
        prediction_frame = pd.DataFrame(
            {
                "attainable_gain_centered": [0.0, 1.0],
                "acquisition": [acquisition, acquisition],
            }
        )
        predictions = np.asarray(full.predict(prediction_frame), dtype=float)
        rule_rows.append(
            {
                "acquisition": acquisition,
                "intercept_at_mean_gain": float(predictions[0]),
                "slope_gain_per_pp": float(predictions[1] - predictions[0]),
                "n_configurations": int(
                    merged["acquisition"].astype(str).eq(acquisition).sum()
                ),
            }
        )
    rule_summary = pd.DataFrame(rule_rows)

    merged = merged.merge(rule_summary, on="acquisition", how="left", validate="many_to_one")
    merged["attainable_gain_mean_pp"] = attainable_mean
    merged["interaction_partial_r2"] = interaction_partial_r2
    merged["moderation_fitted_pp"] = np.asarray(full.fittedvalues, dtype=float)
    return merged.reset_index(drop=True)
