from __future__ import annotations

import numpy as np
import pandas as pd


def lifetime_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - true
    denom = np.maximum(np.abs(true), 1e-6)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    mape = float(np.mean(np.abs(err) / denom) * 100.0)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": mape,
        "rmspe": float(np.sqrt(np.mean((err / denom) ** 2)) * 100.0),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "accuracy": float(1.0 - mape / 100.0),
    }


def subset_lifetime_metrics(true: np.ndarray, pred: np.ndarray, mask: np.ndarray, prefix: str) -> dict[str, float]:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return {
            f"{prefix}_mae": float("nan"),
            f"{prefix}_rmse": float("nan"),
            f"{prefix}_mape": float("nan"),
            f"{prefix}_rmspe": float("nan"),
            f"{prefix}_r2": float("nan"),
            f"{prefix}_accuracy": float("nan"),
        }
    values = lifetime_metrics(np.asarray(true, dtype=float)[mask], np.asarray(pred, dtype=float)[mask])
    return {f"{prefix}_{key}": value for key, value in values.items()}


def prediction_frame(test_table: pd.DataFrame, pred: np.ndarray, pred_std: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "index": test_table["index"].to_numpy(dtype=int),
            "source_index": test_table["source_index"].to_numpy(dtype=int),
            "cell_name": test_table["cell_name"].to_numpy(),
            "lifetime": test_table["lifetime"].to_numpy(float),
            "pred_lifetime": pred.astype(float),
            "pred_lifetime_std": pred_std.astype(float),
        }
    )


DEFAULT_ALL_POOL_ACCURACY_FRACTION = 0.95


def _accuracy_from_mape(mape: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return 1.0 - np.asarray(mape, dtype=float) / 100.0


def _target_mape_from_accuracy(target_accuracy: float) -> float:
    return float((1.0 - float(target_accuracy)) * 100.0)


def add_pool_accuracy_targets(metrics: pd.DataFrame, all_pool_accuracy_fraction: float = DEFAULT_ALL_POOL_ACCURACY_FRACTION) -> pd.DataFrame:
    if metrics.empty or "all_pool_mape" not in metrics.columns:
        return metrics
    out = metrics.copy()
    if "accuracy" not in out.columns:
        out["accuracy"] = _accuracy_from_mape(out["mape"])
    if "all_pool_accuracy" not in out.columns:
        out["all_pool_accuracy"] = _accuracy_from_mape(out["all_pool_mape"])
    suffix = f"{int(round(all_pool_accuracy_fraction * 100))}"
    out[f"pool_{suffix}_accuracy_threshold"] = out["all_pool_accuracy"] * float(all_pool_accuracy_fraction)
    out[f"pool_{suffix}_mape_equivalent_threshold"] = _target_mape_from_accuracy(0.0) - out[f"pool_{suffix}_accuracy_threshold"] * 100.0
    out[f"reaches_pool_{suffix}_all_pool_accuracy"] = out["accuracy"] >= out[f"pool_{suffix}_accuracy_threshold"]
    return out


def summarize_pool_accuracy_cost(metrics: pd.DataFrame, all_pool_accuracy_fraction: float = DEFAULT_ALL_POOL_ACCURACY_FRACTION) -> pd.DataFrame:
    if metrics.empty or "all_pool_mape" not in metrics.columns:
        return pd.DataFrame()
    metrics = add_pool_accuracy_targets(metrics, all_pool_accuracy_fraction=all_pool_accuracy_fraction)
    group_cols = ["dataset", "experiment_mode", "trial", "regressor", "method", "iterative_batch_size"]
    group_cols = [c for c in group_cols if c in metrics.columns]
    rows = []
    for keys, sub in metrics.groupby(group_cols, dropna=False):
        budget_col = "budget_total"
        if budget_col not in sub.columns:
            continue
        curve = (
            sub[[budget_col, "mape", "accuracy", "total_test_cost", "full_life_test_count", "labeled_train_count", "all_pool_mape", "all_pool_accuracy"]]
            .dropna(subset=[budget_col, "accuracy", "all_pool_accuracy"])
            .groupby(budget_col, as_index=False)
            .agg(
                mape=("mape", "mean"),
                accuracy=("accuracy", "mean"),
                total_test_cost=("total_test_cost", "mean"),
                full_life_test_count=("full_life_test_count", "mean"),
                labeled_train_count=("labeled_train_count", "mean"),
                all_pool_mape=("all_pool_mape", "mean"),
                all_pool_accuracy=("all_pool_accuracy", "mean"),
            )
            .sort_values(budget_col)
        )
        if curve.empty:
            continue
        target_accuracy = float(curve["all_pool_accuracy"].dropna().mean() * float(all_pool_accuracy_fraction))
        target_mape = _target_mape_from_accuracy(target_accuracy)
        budgets = curve[budget_col].to_numpy(float)
        accuracies = curve["accuracy"].to_numpy(float)
        best_so_far = np.maximum.accumulate(accuracies)
        reached = best_so_far >= target_accuracy
        row = sub.iloc[0].to_dict()
        row["target_accuracy"] = target_accuracy
        row["target_mape"] = target_mape
        row["budget_axis"] = budget_col
        if not reached.any():
            row["status"] = "not_reached"
            row["required_budget_axis_cells"] = float("nan")
            row["required_total_test_cost"] = float("nan")
            row["required_full_life_test_count"] = float("nan")
            row["required_labeled_train_count"] = float("nan")
        else:
            row["status"] = "reached"
            idx = int(np.argmax(reached))
            if idx == 0:
                required_budget = float(budgets[0])
            else:
                x0, x1 = budgets[idx - 1], budgets[idx]
                y0, y1 = best_so_far[idx - 1], best_so_far[idx]
                required_budget = float(x1 if np.isclose(y0, y1) else x0 + (target_accuracy - y0) / (y1 - y0) * (x1 - x0))
            row["required_budget_axis_cells"] = required_budget
            row["required_total_test_cost"] = float(np.interp(required_budget, budgets, curve["total_test_cost"].to_numpy(float)))
            row["required_full_life_test_count"] = float(np.interp(required_budget, budgets, curve["full_life_test_count"].to_numpy(float)))
            row["required_labeled_train_count"] = float(np.interp(required_budget, budgets, curve["labeled_train_count"].to_numpy(float)))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_target_accuracy_cost(metrics: pd.DataFrame) -> pd.DataFrame:
    return summarize_pool_accuracy_cost(metrics, all_pool_accuracy_fraction=DEFAULT_ALL_POOL_ACCURACY_FRACTION)

