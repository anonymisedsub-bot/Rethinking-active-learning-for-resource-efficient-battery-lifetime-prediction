from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy.contrasts import Sum
from statsmodels.formula.api import ols


FACTOR_LABELS = {
    "dataset": "Dataset",
    "feature_set": "Feature",
    "model": "Model",
    "acquisition": "Acquisition",
    "initialisation": "Initialisation",
}


def _coded(factor: str) -> str:
    return f"C({factor}, Sum)"


def _model_terms(factors: list[str]) -> list[tuple[tuple[str, ...], str]]:
    terms = [((factor,), _coded(factor)) for factor in factors]
    terms.extend(((left, right), f"{_coded(left)}:{_coded(right)}") for left, right in combinations(factors, 2))
    return terms


def _term_label(raw: str, terms: list[tuple[tuple[str, ...], str]]) -> str:
    if raw == "Residual":
        return "Residual"
    for factor_tuple, expression in terms:
        if raw == expression:
            return " x ".join(FACTOR_LABELS.get(factor, factor.title()) for factor in factor_tuple)
    return raw


def factorial_anova(
    data: pd.DataFrame,
    response: str,
    factors: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = [response, *factors]
    frame = data[required].dropna().copy()
    if frame.empty:
        raise ValueError("ANOVA input is empty after removing missing values")
    terms = _model_terms(factors)
    formula = f"{response} ~ " + " + ".join(expression for _, expression in terms)
    full_model = ols(formula, data=frame).fit()
    raw = sm.stats.anova_lm(full_model, typ=3).reset_index().rename(
        columns={"index": "raw_term", "sum_sq": "ss", "PR(>F)": "p_value"}
    )
    raw = raw.loc[raw["raw_term"].ne("Intercept")].copy()
    total_ss = float(((frame[response] - frame[response].mean()) ** 2).sum())
    raw["term"] = raw["raw_term"].map(lambda value: _term_label(str(value), terms))
    raw["variance_contribution_pct"] = np.where(total_ss > 0, raw["ss"] / total_ss * 100.0, np.nan)
    raw["full_model_r2"] = float(full_model.rsquared)
    term_table = raw[["term", "raw_term", "ss", "df", "F", "p_value", "variance_contribution_pct", "full_model_r2"]]

    grouped_rows = []
    for factor in factors:
        reduced_terms = [expression for factor_tuple, expression in terms if factor not in factor_tuple]
        reduced_formula = f"{response} ~ " + (" + ".join(reduced_terms) if reduced_terms else "1")
        reduced_model = ols(reduced_formula, data=frame).fit()
        drop = max(0.0, float(full_model.rsquared - reduced_model.rsquared))
        grouped_rows.append(
            {
                "factor": FACTOR_LABELS.get(factor, factor.title()),
                "grouped_contribution_pp": drop * 100.0,
                "full_model_r2": float(full_model.rsquared),
                "reduced_model_r2": float(reduced_model.rsquared),
            }
        )
    grouped_table = pd.DataFrame(grouped_rows)
    return term_table.reset_index(drop=True), grouped_table
