from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import AnalysisPaths


METRIC_USECOLS = [
    "dataset", "feature_set", "regressor", "trial", "method", "labeled_train_count",
    "max_budget_mean", "mape", "budget_total",
]
SELECTED_USECOLS = [
    "dataset", "feature_set", "regressor", "trial", "method", "budget_total",
    "selection_order", "acquisition_batch", "selection_phase", "selected_index", "selected_lifetime",
]
SPLIT_USECOLS = ["dataset", "feature_set", "trial", "index", "cell_name", "lifetime", "split_assignment"]
DL_METRIC_USECOLS = ["dataset", "feature_set", "model", "trial", "mape"]


def read_feature_csvs(root: Path, filename: str, usecols: list[str] | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob(f"feature set */*/{filename}"), key=lambda item: str(item)):
        frame = pd.read_csv(path, usecols=usecols, low_memory=False)
        frame["source_path"] = str(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No {filename} files found below {root}")
    return pd.concat(frames, ignore_index=True, sort=False)


def load_inputs(paths: AnalysisPaths, slim: bool = False) -> dict[str, pd.DataFrame]:
    metric_columns = METRIC_USECOLS if slim else None
    selected_columns = SELECTED_USECOLS if slim else None
    split_columns = SPLIT_USECOLS if slim else None
    dl_columns = DL_METRIC_USECOLS if slim else None
    return {
        "metrics": read_feature_csvs(paths.al_root, "metrics.csv", usecols=metric_columns),
        "selected": read_feature_csvs(paths.al_root, "selected_indices.csv", usecols=selected_columns),
        "splits": read_feature_csvs(paths.al_root, "split_assignments.csv", usecols=split_columns),
        "dl_metrics": read_feature_csvs(paths.dl_root, "metrics.csv", usecols=dl_columns),
    }


def _require_columns(frame: pd.DataFrame, columns: list[str], table_name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise KeyError(f"{table_name} is missing required columns: {missing}")


def normalise_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    source_columns = [
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
    _require_columns(raw, source_columns, "metrics")
    out = raw[source_columns].rename(
        columns={
            "regressor": "model",
            "method": "acquisition",
            "labeled_train_count": "labelled_n",
            "max_budget_mean": "pool_n",
            "mape": "holdout_mape",
        }
    )
    for column in ["trial", "labelled_n", "pool_n", "budget_total"]:
        out[column] = pd.to_numeric(out[column], errors="raise").round().astype(int)
    out["holdout_mape"] = pd.to_numeric(out["holdout_mape"], errors="raise")
    return out


def normalise_selected(raw: pd.DataFrame) -> pd.DataFrame:
    source_columns = [
        "dataset",
        "feature_set",
        "regressor",
        "trial",
        "method",
        "budget_total",
        "selection_order",
        "acquisition_batch",
        "selection_phase",
        "selected_index",
        "selected_lifetime",
    ]
    _require_columns(raw, source_columns, "selected indices")
    out = raw[source_columns].rename(columns={"regressor": "model", "method": "acquisition"})
    for column in ["trial", "budget_total", "selection_order", "acquisition_batch", "selected_index"]:
        out[column] = pd.to_numeric(out[column], errors="raise").round().astype(int)
    out["selected_lifetime"] = pd.to_numeric(out["selected_lifetime"], errors="raise")
    return out


def normalise_dl_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    source_columns = ["dataset", "feature_set", "model", "trial", "mape"]
    _require_columns(raw, source_columns, "deep-learning metrics")
    out = raw[source_columns].rename(columns={"mape": "full_pool_mape"})
    out["trial"] = pd.to_numeric(out["trial"], errors="raise").round().astype(int)
    out["full_pool_mape"] = pd.to_numeric(out["full_pool_mape"], errors="raise")
    return out
