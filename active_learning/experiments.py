from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from config import DEFAULT_RUN_TRIALS, ExperimentConfig, FeatureSetSpec
from data import canonical_split_mode, feature_matrix, prepare_feature_data, split_mode_label
from metrics import DEFAULT_ALL_POOL_ACCURACY_FRACTION, add_pool_accuracy_targets, lifetime_metrics, prediction_frame, subset_lifetime_metrics, summarize_pool_accuracy_cost
from models import error_risk_table, fit_regressor, uncertainty_table
from targeting import canonical_target_transform
from active_learning import select_indices
from utils import ensure_dir, save_json, set_seed
from visualize import plot_dataset_outputs, plot_mape_budget_overview


def _dataset_iterative_batch_size(cfg: ExperimentConfig, dataset_name: str) -> int:
    batch_sizes = getattr(cfg, "dataset_iterative_batch_sizes", None)
    if batch_sizes is None:
        raise AttributeError("ExperimentConfig.dataset_iterative_batch_sizes is required for dataset-specific budgets.")
    if dataset_name not in batch_sizes:
        raise KeyError(f"Missing iterative batch size for dataset: {dataset_name}")
    value = int(batch_sizes[dataset_name])
    if value <= 0:
        raise ValueError(f"Iterative batch size must be positive for dataset {dataset_name}: {value}")
    return value


def _dataset_budget_grid(cfg: ExperimentConfig, dataset_name: str, max_budget: int, include_zero: bool) -> list[int]:
    max_budget = int(max_budget)
    if max_budget <= 0:
        return [0] if include_zero else []
    budget_specs = getattr(cfg, "dataset_budget_grid_specs", None)
    if budget_specs is None:
        raise AttributeError("ExperimentConfig.dataset_budget_grid_specs is required for dataset-specific budget grids.")
    if dataset_name not in budget_specs:
        raise KeyError(f"Missing budget grid spec for dataset: {dataset_name}")
    start, step = (int(v) for v in budget_specs[dataset_name])
    if start <= 0 or step <= 0:
        raise ValueError(f"Budget grid start/step must be positive for dataset {dataset_name}: {(start, step)}")
    budgets: list[int] = [0] if include_zero else []
    if start <= max_budget:
        budgets.append(start)
    value = step
    while value < max_budget:
        budgets.append(int(value))
        value += step
    if max_budget not in budgets:
        budgets.append(max_budget)
    return sorted(set(int(v) for v in budgets))


def _configured_trial_ids(cfg: ExperimentConfig) -> list[int]:
    run_trials = getattr(cfg, "run_trials", None)
    if run_trials is None:
        run_trials = DEFAULT_RUN_TRIALS
    trial_ids = [int(trial) for trial in run_trials]
    if not trial_ids:
        raise ValueError("run_trials must contain at least one trial id when provided.")
    if any(trial <= 0 for trial in trial_ids):
        raise ValueError(f"run_trials must contain positive trial ids, got {trial_ids}.")
    return sorted(dict.fromkeys(trial_ids))


def _active_learning_parallel_jobs(cfg: ExperimentConfig) -> int:
    jobs = int(getattr(cfg, "parallel_jobs", 1) or 1)
    if jobs < 1:
        raise ValueError(f"parallel_jobs must be a positive integer, got {jobs}.")
    return jobs


def _fresh_trial_job_config(cfg: ExperimentConfig) -> ExperimentConfig:
    job_cfg = copy.copy(cfg)
    setattr(job_cfg, "_runtime_events", [])
    setattr(job_cfg, "_runtime_context", {})
    return job_cfg


def _merge_existing_dataset_output(path: Path, new_frame: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    new_frame = new_frame.copy()
    if not path.exists():
        return new_frame
    existing = pd.read_csv(path)
    if existing.empty:
        return new_frame
    if new_frame.empty:
        return existing
    missing = [col for col in key_cols if col not in existing.columns or col not in new_frame.columns]
    if missing:
        raise KeyError(f"Cannot merge {path.name}; missing key column(s): {missing}")
    merged = pd.concat([existing, new_frame], ignore_index=True, sort=False)
    merged = merged.drop_duplicates(subset=key_cols, keep="last")
    sort_cols = [
        col
        for col in [
            "feature_set",
            "dataset",
            "trial",
            "regressor",
            "model",
            "method",
            "budget_total",
            "index",
            "selection_order",
            "epoch",
        ]
        if col in merged.columns
    ]
    if sort_cols:
        merged = merged.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return merged


def _merge_partial_trial_output(
    cfg: ExperimentConfig,
    path: Path,
    new_frame: pd.DataFrame,
    key_cols: list[str],
) -> pd.DataFrame:
    if getattr(cfg, "run_trials", None) is None:
        return new_frame.copy()
    return _merge_existing_dataset_output(path, new_frame, key_cols)


def _write_merged_summary(path: Path, new_frame: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    merged = _merge_existing_dataset_output(path, new_frame, key_cols)
    merged.to_csv(path, index=False)
    return merged


PREDICTION_SUMMARY_FILENAMES = (
    "predictions_summary.csv",
    "selected_indices_summary.csv",
    "acquisition_trajectory_summary.csv",
)

DATASET_PREDICTION_FILENAMES = (
    "predictions.csv",
    "selected_indices.csv",
    "acquisition_trajectory.csv",
)


def _cleanup_prediction_outputs_when_disabled(feature_root: Path) -> list[Path]:
    removed: list[Path] = []
    root = Path(feature_root)
    for filename in PREDICTION_SUMMARY_FILENAMES:
        path = root / filename
        if path.exists():
            path.unlink()
            removed.append(path)
    if root.exists():
        for dataset_dir in root.iterdir():
            if not dataset_dir.is_dir():
                continue
            for filename in DATASET_PREDICTION_FILENAMES:
                path = dataset_dir / filename
                if path.exists():
                    path.unlink()
                    removed.append(path)
    return removed


REGRESSOR_DISPLAY_LABELS = {
    "GPR": "GPR",
    "RF": "RF",
    "AE_ENet": "AE-ENet",
}

ACQUISITION_RULE_LABELS = {
    "random_selection": "Random",
    "diversity_oneshot": "Diversity_oneshot",
    "diversity_iterative": "Diversity_iterative",
    "coverage": "Coverage",
    "exploration": "Exploration",
    "exploitation": "Exploitation",
    "hybrid": "Hybrid",
}


def _canonical_regressor_name(regressor_name: str) -> str:
    upper = str(regressor_name).upper().replace("-", "_")
    if upper in {"AE_ENET", "AEENET", "AENET"}:
        return "AE_ENet"
    return upper


def _regressor_display_label(regressor_name: str) -> str:
    canonical = _canonical_regressor_name(regressor_name)
    return REGRESSOR_DISPLAY_LABELS.get(canonical, REGRESSOR_DISPLAY_LABELS.get(canonical.upper(), canonical))


def _acquisition_rule_label(method: str) -> str:
    return ACQUISITION_RULE_LABELS.get(str(method), str(method))


def _model_acquisition_label(regressor_name: str, method: str) -> str:
    return f"{_regressor_display_label(regressor_name)}__{_acquisition_rule_label(method)}"


def _acquisition_implementation_for_regressor(method: str, regressor_name: str) -> str:
    if method == "exploitation":
        return "max_error_exploitation"
    if method != "exploration":
        return str(method)
    upper = _canonical_regressor_name(regressor_name).upper()
    if upper == "GPR":
        return "uncertainty"
    if upper in {"RF", "AE_ENET"}:
        return "qbc"
    raise ValueError(f"Exploration is not configured for regressor: {regressor_name}")


def _is_iterative_method(method: str) -> bool:
    return method in {
        "random_selection",
        "diversity_iterative",
        "coverage",
        "exploration",
        "exploitation",
        "hybrid",
    }


def _is_model_aware_iterative_method(method: str) -> bool:
    return method in {"exploration", "exploitation", "hybrid"}


def _is_model_agnostic_iterative_method(method: str) -> bool:
    return method in {"random_selection", "diversity_iterative", "coverage"}


def _is_method_allowed_for_regressor(method: str, regressor_name: str) -> bool:
    if method not in {
        "random_selection",
        "diversity_oneshot",
        "diversity_iterative",
        "coverage",
        "exploration",
        "exploitation",
        "hybrid",
    }:
        return False
    upper = _canonical_regressor_name(regressor_name).upper()
    if method == "exploration":
        return upper in {"GPR", "RF", "AE_ENET"}
    if method == "exploitation":
        return upper in {"GPR", "RF", "AE_ENET"}
    return True


def _acquisition_mode(method: str) -> str:
    if method == "diversity_oneshot":
        return "one_shot_batched"
    return "iterative" if _is_iterative_method(method) else "one_shot"


def _model_seed_offset(regressor_name: str) -> int:
    return sum(ord(ch) for ch in str(regressor_name))


def _dataset_seed_offset(dataset_name: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(str(dataset_name)))


def _cold_start_seed(
    cfg: ExperimentConfig,
    dataset_name: str,
    trial: int,
    batch_size: int,
    threshold: float | None = None,
) -> int:
    return int(cfg.random_seed) + int(trial) * 200003


def _evaluation_train_seed(
    cfg: ExperimentConfig,
    dataset_name: str,
    trial: int,
    regressor_name: str,
    experiment_mode: str,
    budget: int,
    batch_size: int = 0,
    threshold: float | None = None,
) -> int:
    if experiment_mode != "total_budget":
        raise ValueError(f"Unknown experiment_mode for training seed: {experiment_mode}")
    # Final model evaluation randomness is tied only to the trial. This keeps
    # identical labelled sets comparable across acquisition methods, budgets,
    # and channel/batch-size settings.
    return int(cfg.random_seed) + int(trial) * 200003


def _selection_seed(base_seed: int, method: str, regressor_name: str) -> int:
    # Model-agnostic acquisitions must select identical cells across lifetime
    # predictors; model-aware acquisitions are allowed to depend on predictor.
    return base_seed + (_model_seed_offset(regressor_name) if _is_model_aware_iterative_method(method) else 0)


def _feature_set_context(prepared) -> dict[str, object]:
    dataset = prepared.dataset
    retained = dataset.retained_feature_indices
    return {
        "feature_set": dataset.feature_set,
        "feature_set_label": dataset.feature_set_label,
        "feature_matrix_key": dataset.feature_matrix_key,
        "n_original_features": int(dataset.original_feature_count),
        "n_features": int(dataset.features.shape[1]),
        "retained_feature_indices": "" if retained is None else ",".join(str(int(v)) for v in retained),
    }


def _split_assignment_table(cfg: ExperimentConfig, prepared, spec, trial: int) -> pd.DataFrame:
    table = prepared.table.copy()
    pool_set = set(np.asarray(prepared.pool_indices, dtype=int).tolist())
    test_set = set(np.asarray(prepared.test_indices, dtype=int).tolist())
    assignment = []
    for idx in table["index"].to_numpy(dtype=int):
        if int(idx) in pool_set:
            assignment.append("pool")
        elif int(idx) in test_set:
            assignment.append("holdout")
        else:
            assignment.append("unused")
    audit = table[["index", "source_index", "cell_name", "lifetime", "feature_set", "feature_set_label", "feature_matrix_key"]].copy()
    audit.insert(0, "dataset", spec.name)
    audit.insert(1, "trial", int(trial))
    audit["split_assignment"] = assignment
    audit["pool_holdout_split_mode"] = canonical_split_mode(getattr(cfg, "pool_holdout_split_mode", "random"))
    audit["pool_holdout_split_label"] = split_mode_label(getattr(cfg, "pool_holdout_split_mode", "random"))
    protocol_cols = [c for c in table.columns if c.startswith("protocol_")]
    for col in protocol_cols:
        audit[col] = table[col].to_numpy(float)
    if protocol_cols:
        audit["protocol_key"] = table[protocol_cols].round(10).astype(str).agg("|".join, axis=1)
    else:
        audit["protocol_key"] = ""
    audit["is_pool"] = audit["split_assignment"].eq("pool")
    audit["is_holdout"] = audit["split_assignment"].eq("holdout")
    audit["is_unused"] = audit["split_assignment"].eq("unused")
    return audit


def _feature_processing_audit(cfg: ExperimentConfig, prepared, spec, trial: int) -> dict[str, object]:
    table = prepared.table
    protocol_cols = [c for c in table.columns if c.startswith("protocol_")]
    return {
        **_feature_set_context(prepared),
        "dataset": spec.name,
        "trial": int(trial),
        "pool_holdout_split_mode": canonical_split_mode(getattr(cfg, "pool_holdout_split_mode", "random")),
        "pool_holdout_split_label": split_mode_label(getattr(cfg, "pool_holdout_split_mode", "random")),
        "n_filtered_samples": int(len(table)),
        "n_pool": int(len(prepared.pool_indices)),
        "n_holdout": int(len(prepared.test_indices)),
        "n_unused": int(len(table) - len(prepared.pool_indices) - len(prepared.test_indices)),
        "n_protocol_parameters": int(len(protocol_cols)),
        "correlation_threshold": prepared.dataset.correlation_threshold,
        "imputer": type(prepared.imputer).__name__,
        "scaler": type(prepared.scaler).__name__,
        "preprocessing_fit_scope": "pool_only",
        "lifetime_min": float(table["lifetime"].min()),
        "lifetime_max": float(table["lifetime"].max()),
        "lifetime_median": float(table["lifetime"].median()),
    }


def _prediction_calibration_summary(predictions: pd.DataFrame, source_label: str) -> pd.DataFrame:
    if predictions.empty or "pred_lifetime_std" not in predictions.columns:
        return pd.DataFrame()
    df = predictions.copy()
    df["prediction_source"] = source_label
    df["abs_error"] = (df["pred_lifetime"].astype(float) - df["lifetime"].astype(float)).abs()
    df["absolute_percentage_error"] = df["abs_error"] / np.maximum(df["lifetime"].abs().astype(float), 1e-6) * 100.0
    df["uncertainty_z"] = df["abs_error"] / np.maximum(df["pred_lifetime_std"].abs().astype(float), 1e-6)
    group_cols = [
        "prediction_source",
        "dataset",
        "feature_set",
        "trial",
        "experiment_mode",
        "regressor",
        "method",
        "budget_total",
        "iterative_batch_size",
    ]
    group_cols = [c for c in group_cols if c in df.columns]
    rows: list[pd.DataFrame] = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=["pred_lifetime_std", "abs_error"])
        if sub.empty:
            continue
        try:
            bins = pd.qcut(sub["pred_lifetime_std"], q=min(5, len(sub)), duplicates="drop")
        except ValueError:
            bins = pd.Series(["all"] * len(sub), index=sub.index)
        tmp = sub.copy()
        tmp["uncertainty_bin"] = bins.astype(str)
        summary = (
            tmp.groupby("uncertainty_bin", dropna=False, as_index=False)
            .agg(
                n_samples=("abs_error", "size"),
                pred_lifetime_std_mean=("pred_lifetime_std", "mean"),
                pred_lifetime_std_min=("pred_lifetime_std", "min"),
                pred_lifetime_std_max=("pred_lifetime_std", "max"),
                abs_error_mean=("abs_error", "mean"),
                abs_error_median=("abs_error", "median"),
                absolute_percentage_error_mean=("absolute_percentage_error", "mean"),
                uncertainty_z_mean=("uncertainty_z", "mean"),
            )
        )
        if not isinstance(keys, tuple):
            keys = (keys,)
        for col, value in zip(group_cols, keys):
            summary[col] = value
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    ordered_cols = group_cols + [
        "uncertainty_bin",
        "n_samples",
        "pred_lifetime_std_mean",
        "pred_lifetime_std_min",
        "pred_lifetime_std_max",
        "abs_error_mean",
        "abs_error_median",
        "absolute_percentage_error_mean",
        "uncertainty_z_mean",
    ]
    return pd.concat(rows, ignore_index=True)[ordered_cols]


def _runtime_events_frame(cfg: ExperimentConfig) -> pd.DataFrame:
    events = getattr(cfg, "_runtime_events", [])
    return pd.DataFrame(events) if isinstance(events, list) and events else pd.DataFrame()


def _runtime_summary(runtime_events: pd.DataFrame) -> pd.DataFrame:
    if runtime_events.empty:
        return pd.DataFrame()
    group_cols = [
        "feature_set",
        "dataset",
        "trial",
        "phase",
        "experiment_mode",
        "regressor",
        "method",
        "budget_total",
        "iterative_batch_size",
        "event",
    ]
    group_cols = [c for c in group_cols if c in runtime_events.columns]
    return (
        runtime_events.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            n_events=("elapsed_sec", "size"),
            elapsed_sec_total=("elapsed_sec", "sum"),
            elapsed_sec_mean=("elapsed_sec", "mean"),
            n_train_mean=("n_train", "mean"),
            n_features_mean=("n_features", "mean"),
        )
        .sort_values(["elapsed_sec_total"], ascending=False)
    )


def _set_runtime_context(cfg: ExperimentConfig, **context: object) -> dict[str, object]:
    previous = getattr(cfg, "_runtime_context", {})
    setattr(cfg, "_runtime_context", dict(context))
    return dict(previous) if isinstance(previous, dict) else {}


def _restore_runtime_context(cfg: ExperimentConfig, previous: dict[str, object]) -> None:
    setattr(cfg, "_runtime_context", previous)


def _baseline_group_columns(metrics: pd.DataFrame) -> list[str]:
    cols = []
    if "feature_set" in metrics.columns:
        cols.append("feature_set")
    return cols + ["dataset", "regressor"]


def _max_budget_all_pool_baseline_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or "budget_total" not in metrics.columns:
        return pd.DataFrame()
    required = {"dataset", "trial", "regressor", "budget_total"}
    if not required.issubset(metrics.columns):
        return pd.DataFrame()
    df = metrics.copy()
    if "experiment_mode" in df.columns:
        df = df[df["experiment_mode"].astype(str) == "total_budget"].copy()
    df = df.dropna(subset=["budget_total"])
    if df.empty:
        return pd.DataFrame()
    group_cols = _baseline_group_columns(df)
    trial_cols = group_cols + ["trial"]
    max_budget = df.groupby(trial_cols, dropna=False)["budget_total"].transform("max")
    max_rows = df[df["budget_total"].astype(float).eq(max_budget.astype(float))].copy()
    if max_rows.empty:
        return pd.DataFrame()

    trial_agg_spec: dict[str, tuple[str, str]] = {
        "n_max_budget_rows": ("budget_total", "size"),
        "max_budget": ("budget_total", "mean"),
    }
    for metric in ("mae", "rmse", "mape", "rmspe", "r2", "accuracy"):
        if metric in max_rows.columns:
            trial_agg_spec[metric] = (metric, "mean")
    trial_metrics = max_rows.groupby(trial_cols, dropna=False, as_index=False).agg(**trial_agg_spec)

    agg_spec: dict[str, tuple[str, str]] = {
        "n_max_budget_rows": ("n_max_budget_rows", "sum"),
        "n_trials": ("trial", "nunique"),
        "max_budget_mean": ("max_budget", "mean"),
    }
    for metric in ("mae", "rmse", "mape", "rmspe", "r2", "accuracy"):
        if metric in trial_metrics.columns:
            agg_spec[f"all_pool_{metric}"] = (metric, "mean")
    baseline = trial_metrics.groupby(group_cols, dropna=False, as_index=False).agg(**agg_spec)
    baseline["all_pool_baseline_source"] = "max_budget_trial_mean"
    return baseline


def _apply_max_budget_all_pool_baseline(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    baseline = _max_budget_all_pool_baseline_metrics(metrics)
    if baseline.empty:
        return metrics
    group_cols = _baseline_group_columns(metrics)
    drop_cols = [
        col
        for col in metrics.columns
        if col.startswith("all_pool_") or col in {"all_pool_baseline_source", "n_max_budget_rows", "n_trials", "max_budget_mean"}
    ]
    out = metrics.drop(columns=drop_cols, errors="ignore")
    return out.merge(baseline, on=group_cols, how="left")


def _evaluate_setup(
    cfg: ExperimentConfig,
    prepared,
    dataset_name: str,
    trial: int,
    regressor_name: str,
    method: str,
    experiment_mode: str,
    initial_label_table: pd.DataFrame,
    candidate_table: pd.DataFrame,
    selected_indices: np.ndarray,
    selection_trace: pd.DataFrame | None,
    test_table: pd.DataFrame,
    pool_lifetime_median: float,
    seed: int,
    budget_total: int | None,
    iterative_batch_size: int,
    evaluation_cache: dict[tuple[object, ...], dict[str, object]] | None = None,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    selected_indices = np.asarray(selected_indices, dtype=int)
    if len(selected_indices):
        selected_table = candidate_table.set_index("index").loc[selected_indices].reset_index()
    else:
        selected_table = candidate_table.iloc[0:0].copy()
    train_table = pd.concat([initial_label_table, selected_table], ignore_index=True)
    train_key = tuple(sorted(train_table["index"].to_numpy(dtype=int).tolist()))
    test_key = tuple(test_table["index"].to_numpy(dtype=int).tolist())
    cache_key = (
        regressor_name,
        int(seed),
        train_key,
        test_key,
        prepared.dataset.feature_set,
    )
    cached = evaluation_cache.get(cache_key) if evaluation_cache is not None else None
    if cached is None:
        train_fit_table = train_table.sort_values("index").reset_index(drop=True)
        previous_context = _set_runtime_context(
            cfg,
            dataset=dataset_name,
            feature_set=prepared.dataset.feature_set,
            trial=int(trial),
            phase="final_evaluation",
            experiment_mode=experiment_mode,
            regressor=regressor_name,
            method=method,
            budget_total=np.nan if budget_total is None else int(budget_total),
            iterative_batch_size=int(iterative_batch_size),
        )
        try:
            model = fit_regressor(regressor_name, train_fit_table, cfg, seed)
            pred, pred_std = model.predict(test_table)
        finally:
            _restore_runtime_context(cfg, previous_context)
        metric = lifetime_metrics(test_table["lifetime"].to_numpy(float), pred)
        if evaluation_cache is not None:
            evaluation_cache[cache_key] = {
                "pred": np.asarray(pred, dtype=float).copy(),
                "pred_std": np.asarray(pred_std, dtype=float).copy(),
                "metric": dict(metric),
            }
    else:
        pred = np.asarray(cached["pred"], dtype=float).copy()
        pred_std = np.asarray(cached["pred_std"], dtype=float).copy()
        metric = dict(cached["metric"])
        events = getattr(cfg, "_runtime_events", None)
        if isinstance(events, list):
            events.append(
                {
                    "dataset": dataset_name,
                    "feature_set": prepared.dataset.feature_set,
                    "trial": int(trial),
                    "phase": "final_evaluation",
                    "experiment_mode": experiment_mode,
                    "regressor": regressor_name,
                    "method": method,
                    "budget_total": np.nan if budget_total is None else int(budget_total),
                    "iterative_batch_size": int(iterative_batch_size),
                    "event": "evaluation_cache_hit",
                    "n_train": int(len(train_table)),
                    "n_features": int(feature_matrix(train_table).shape[1]) if not train_table.empty else 0,
                    "elapsed_sec": 0.0,
                }
            )
    pred_df = prediction_frame(test_table, pred, pred_std)
    for key, value in _feature_set_context(prepared).items():
        pred_df[key] = value
    pred_df["dataset"] = dataset_name
    pred_df["trial"] = trial
    pred_df["experiment_mode"] = experiment_mode
    pred_df["regressor"] = regressor_name
    pred_df["model"] = _regressor_display_label(regressor_name)
    acquisition_implementation = _acquisition_implementation_for_regressor(method, regressor_name)
    pred_df["method"] = method
    pred_df["acquisition_implementation"] = acquisition_implementation
    pred_df["acquisition_rule"] = _acquisition_rule_label(method)
    pred_df["model_acquisition"] = f"{regressor_name}__{method}"
    pred_df["model_acquisition_label"] = _model_acquisition_label(regressor_name, method)
    pred_df["acquisition_mode"] = _acquisition_mode(method)
    pred_df["budget_total"] = budget_total
    pred_df["iterative_batch_size"] = iterative_batch_size
    pred_df["pool_lifetime_median"] = pool_lifetime_median
    pred_df["is_long_life"] = pred_df["lifetime"] > pool_lifetime_median
    pred_df["target_transform"] = canonical_target_transform(getattr(cfg, "target_transform", "log"))
    pred_df["prediction_clip_enabled"] = bool(getattr(cfg, "prediction_clip_enabled", True))
    pred_df["pool_holdout_split_mode"] = canonical_split_mode(getattr(cfg, "pool_holdout_split_mode", "random"))
    pred_df["pool_holdout_split_label"] = split_mode_label(getattr(cfg, "pool_holdout_split_mode", "random"))

    full_life_cost = float(selected_table["lifetime"].sum())
    true_lifetime = test_table["lifetime"].to_numpy(float)
    long_life_mask = true_lifetime > pool_lifetime_median
    row = {
        **metric,
        **subset_lifetime_metrics(true_lifetime, pred, long_life_mask, "holdout_long_life"),
        **_feature_set_context(prepared),
        "dataset": dataset_name,
        "trial": trial,
        "experiment_mode": experiment_mode,
        "regressor": regressor_name,
        "model": _regressor_display_label(regressor_name),
        "method": method,
        "acquisition_implementation": acquisition_implementation,
        "acquisition_rule": _acquisition_rule_label(method),
        "model_acquisition": f"{regressor_name}__{method}",
        "model_acquisition_label": _model_acquisition_label(regressor_name, method),
        "acquisition_mode": _acquisition_mode(method),
        "budget_total": np.nan if budget_total is None else int(budget_total),
        "iterative_batch_size": int(iterative_batch_size),
        "pool_lifetime_median": pool_lifetime_median,
        "full_life_test_count": int(len(selected_table)),
        "labeled_train_count": int(len(train_table)),
        "full_life_long_life_count": int((selected_table["lifetime"] > pool_lifetime_median).sum()) if not selected_table.empty else 0,
        "holdout_long_life_count": int(long_life_mask.sum()),
        "total_test_cost": full_life_cost,
        "full_life_test_cost": full_life_cost,
        "target_transform": canonical_target_transform(getattr(cfg, "target_transform", "log")),
        "prediction_clip_enabled": bool(getattr(cfg, "prediction_clip_enabled", True)),
        "log_clip_lower": float(getattr(cfg, "log_clip_bounds", (-3.0, 3.0))[0]),
        "log_clip_upper": float(getattr(cfg, "log_clip_bounds", (-3.0, 3.0))[1]),
        "raw_clip_lower": float(getattr(cfg, "raw_clip_lower", 0.0)),
        "raw_clip_upper_sigma": float(getattr(cfg, "raw_clip_upper_sigma", 5.0)),
        "pool_holdout_split_mode": canonical_split_mode(getattr(cfg, "pool_holdout_split_mode", "random")),
        "pool_holdout_split_label": split_mode_label(getattr(cfg, "pool_holdout_split_mode", "random")),
    }
    selected_df = selected_table[["index", "source_index", "cell_name", "lifetime"]].copy()
    selected_df = selected_df.rename(columns={"index": "selected_index", "lifetime": "selected_lifetime"})
    if selection_trace is None:
        selection_trace = pd.DataFrame()
    if not selection_trace.empty:
        selected_df = selected_df.merge(selection_trace, on="selected_index", how="left")
    else:
        selected_df["selection_order"] = np.arange(1, len(selected_df) + 1, dtype=int)
        selected_df["selection_phase"] = np.nan
        selected_df["predicted_lifetime_at_selection"] = np.nan
        selected_df["uncertainty_score"] = np.nan
        selected_df["acquisition_score"] = np.nan
        selected_df["acquisition_rank"] = np.nan
    selected_df["dataset"] = dataset_name
    for key, value in _feature_set_context(prepared).items():
        selected_df[key] = value
    selected_df["trial"] = trial
    selected_df["experiment_mode"] = experiment_mode
    selected_df["regressor"] = regressor_name
    selected_df["model"] = _regressor_display_label(regressor_name)
    selected_df["method"] = method
    selected_df["acquisition_implementation"] = acquisition_implementation
    selected_df["acquisition_rule"] = _acquisition_rule_label(method)
    selected_df["model_acquisition"] = f"{regressor_name}__{method}"
    selected_df["model_acquisition_label"] = _model_acquisition_label(regressor_name, method)
    selected_df["target_transform"] = canonical_target_transform(getattr(cfg, "target_transform", "log"))
    selected_df["pool_holdout_split_mode"] = canonical_split_mode(getattr(cfg, "pool_holdout_split_mode", "random"))
    selected_df["pool_holdout_split_label"] = split_mode_label(getattr(cfg, "pool_holdout_split_mode", "random"))
    selected_df["acquisition_mode"] = _acquisition_mode(method)
    selected_df["budget_total"] = budget_total
    selected_df["iterative_batch_size"] = iterative_batch_size
    selected_df["pool_lifetime_median"] = pool_lifetime_median
    selected_df["is_long_life"] = selected_df["selected_lifetime"] > pool_lifetime_median
    selected_df["labelled_train_count_at_selection"] = len(initial_label_table) + selected_df["selection_order"].astype(float)
    if int(iterative_batch_size) > 0:
        selected_df["acquisition_batch"] = np.ceil(selected_df["selection_order"].astype(float) / int(iterative_batch_size)).astype(int)
    else:
        selected_df["acquisition_batch"] = 1
    base_cols = [
        "dataset",
        "feature_set",
        "feature_set_label",
        "feature_matrix_key",
        "n_original_features",
        "n_features",
        "retained_feature_indices",
        "trial",
        "experiment_mode",
        "regressor",
        "model",
        "method",
        "acquisition_implementation",
        "acquisition_rule",
        "model_acquisition",
        "model_acquisition_label",
        "acquisition_mode",
        "budget_total",
        "iterative_batch_size",
        "pool_lifetime_median",
        "target_transform",
        "prediction_clip_enabled",
        "pool_holdout_split_mode",
        "pool_holdout_split_label",
        "selection_order",
        "acquisition_batch",
        "labelled_train_count_at_selection",
        "selection_phase",
        "selected_index",
        "source_index",
        "cell_name",
        "selected_lifetime",
        "is_long_life",
        "predicted_lifetime_at_selection",
        "uncertainty_score",
        "target_std",
        "acquisition_score",
        "acquisition_rank",
    ]
    optional_cols = [
        "coverage_score",
        "coverage_rank_score",
        "uncertainty_rank_score",
    ]
    selected_df = selected_df[[c for c in base_cols + optional_cols if c in selected_df.columns]]
    return row, pred_df, selected_df


def _run_total_budget_scan(cfg: ExperimentConfig, prepared, spec, trial: int):
    rows = []
    preds = []
    selected = []
    pool_table = prepared.pool_table
    test_table = prepared.test_table
    pool_lifetime_median = float(pool_table["lifetime"].median())
    budgets = _dataset_budget_grid(cfg, spec.name, len(pool_table), include_zero=False)
    initial_label_table = pool_table.iloc[0:0].copy()
    candidate_table = pool_table.copy()
    model_agnostic_trajectory_cache: dict[tuple[str, int, int], tuple[np.ndarray, pd.DataFrame]] = {}
    evaluation_cache: dict[tuple[object, ...], dict[str, object]] = {}
    for configured_regressor in cfg.regressors:
        regressor = _canonical_regressor_name(configured_regressor)
        for method in cfg.active_learning_methods:
            if not _is_method_allowed_for_regressor(method, regressor):
                continue
            acquisition_implementation = _acquisition_implementation_for_regressor(method, regressor)
            if _is_iterative_method(method):
                max_budget = max(budgets) if budgets else 0
                for batch_size in (_dataset_iterative_batch_size(cfg, spec.name),):
                    print(
                        f"[Run] feature_set={prepared.dataset.feature_set} dataset={spec.name} "
                        f"trial={trial} mode=active_learning model={regressor} "
                        f"method={method} max_budget_total={max_budget} p={batch_size}"
                    )
                    base_seed = cfg.random_seed + trial * 200003 + len(method)
                    select_seed = _selection_seed(base_seed, method, regressor)
                    cold_start_seed = _cold_start_seed(cfg, spec.name, trial, batch_size)
                    rng = np.random.default_rng(select_seed)

                    def u_fn(train_table, cand_table, name=regressor, u_seed=select_seed):
                        previous_context = _set_runtime_context(
                            cfg,
                            dataset=spec.name,
                            feature_set=prepared.dataset.feature_set,
                            trial=int(trial),
                            phase="acquisition_uncertainty",
                            experiment_mode="total_budget",
                            regressor=name,
                            method=method,
                            budget_total=max_budget,
                            iterative_batch_size=int(batch_size),
                            labelled_train_count=int(len(train_table)),
                            candidate_count=int(len(cand_table)),
                        )
                        try:
                            return uncertainty_table(name, train_table, cand_table, cfg, u_seed + len(train_table) + len(cand_table))
                        finally:
                            _restore_runtime_context(cfg, previous_context)

                    def e_fn(train_table, cand_table, name=regressor, e_seed=select_seed):
                        previous_context = _set_runtime_context(
                            cfg,
                            dataset=spec.name,
                            feature_set=prepared.dataset.feature_set,
                            trial=int(trial),
                            phase="acquisition_error_risk",
                            experiment_mode="total_budget",
                            regressor=name,
                            method=method,
                            budget_total=max_budget,
                            iterative_batch_size=int(batch_size),
                            labelled_train_count=int(len(train_table)),
                            candidate_count=int(len(cand_table)),
                        )
                        try:
                            return error_risk_table(name, train_table, cand_table, cfg, e_seed + len(train_table) + len(cand_table))
                        finally:
                            _restore_runtime_context(cfg, previous_context)

                    cache_key = (method, int(max_budget), int(batch_size))
                    if _is_model_agnostic_iterative_method(method) and cache_key in model_agnostic_trajectory_cache:
                        trajectory, trajectory_trace = model_agnostic_trajectory_cache[cache_key]
                    else:
                        trajectory, trajectory_trace = select_indices(
                            acquisition_implementation,
                            max_budget,
                            initial_label_table,
                            candidate_table,
                            feature_matrix,
                            u_fn,
                            rng,
                            batch_size or 1,
                            return_trace=True,
                            error_risk_fn=e_fn,
                            cold_start_seed=cold_start_seed,
                        )
                        if _is_model_agnostic_iterative_method(method):
                            model_agnostic_trajectory_cache[cache_key] = (trajectory, trajectory_trace)
                    for budget_total in budgets:
                        full_life_budget = min(int(budget_total), len(candidate_table))
                        chosen = trajectory[:full_life_budget]
                        trace = trajectory_trace[trajectory_trace["selection_order"] <= full_life_budget].copy()
                        train_seed = _evaluation_train_seed(
                            cfg,
                            spec.name,
                            trial,
                            regressor,
                            "total_budget",
                            int(budget_total),
                            batch_size,
                        )
                        row, pred_df, selected_df = _evaluate_setup(
                            cfg,
                            prepared,
                            spec.name,
                            trial,
                            regressor,
                            method,
                            "total_budget",
                            initial_label_table,
                            candidate_table,
                            chosen,
                            trace,
                            test_table,
                            pool_lifetime_median,
                            train_seed,
                            budget_total,
                            batch_size,
                            evaluation_cache,
                        )
                        rows.append(row)
                        preds.append(pred_df)
                        selected.append(selected_df)
                continue

            for budget_total in budgets:
                full_life_budget = min(int(budget_total), len(candidate_table))
                batch_size = _dataset_iterative_batch_size(cfg, spec.name) if method == "diversity_oneshot" else 0
                print(
                    f"[Run] feature_set={prepared.dataset.feature_set} dataset={spec.name} "
                    f"trial={trial} mode=active_learning model={regressor} "
                    f"method={method} budget_total={budget_total} p={batch_size}"
                )
                if method == "diversity_oneshot":
                    base_seed = cfg.random_seed + trial * 200003 + len("diversity_iterative")
                else:
                    base_seed = cfg.random_seed + trial * 200003 + budget_total * 9173 + len(method)
                select_seed = _selection_seed(base_seed, method, regressor)
                cold_start_seed = _cold_start_seed(cfg, spec.name, trial, _dataset_iterative_batch_size(cfg, spec.name))
                train_seed = _evaluation_train_seed(
                    cfg,
                    spec.name,
                    trial,
                    regressor,
                    "total_budget",
                    int(budget_total),
                    batch_size,
                )
                rng = np.random.default_rng(select_seed)

                def u_fn(train_table, cand_table, name=regressor, u_seed=select_seed):
                    previous_context = _set_runtime_context(
                        cfg,
                        dataset=spec.name,
                        feature_set=prepared.dataset.feature_set,
                        trial=int(trial),
                        phase="acquisition_uncertainty",
                        experiment_mode="total_budget",
                        regressor=name,
                        method=method,
                        budget_total=int(budget_total),
                        iterative_batch_size=int(batch_size),
                        labelled_train_count=int(len(train_table)),
                        candidate_count=int(len(cand_table)),
                    )
                    try:
                        return uncertainty_table(name, train_table, cand_table, cfg, u_seed + len(train_table) + len(cand_table))
                    finally:
                        _restore_runtime_context(cfg, previous_context)

                chosen, trace = select_indices(
                    acquisition_implementation,
                    full_life_budget,
                    initial_label_table,
                    candidate_table,
                    feature_matrix,
                    u_fn,
                    rng,
                    batch_size or 1,
                    return_trace=True,
                    cold_start_seed=cold_start_seed,
                )
                row, pred_df, selected_df = _evaluate_setup(
                    cfg,
                    prepared,
                    spec.name,
                    trial,
                    regressor,
                    method,
                    "total_budget",
                    initial_label_table,
                    candidate_table,
                    chosen,
                    trace,
                    test_table,
                    pool_lifetime_median,
                    train_seed,
                    budget_total,
                    batch_size,
                    evaluation_cache,
                )
                rows.append(row)
                preds.append(pred_df)
                selected.append(selected_df)
    return rows, preds, selected


def _run_trial_job(cfg: ExperimentConfig, spec, feature_set: FeatureSetSpec, trial: int) -> dict[str, object]:
    worker_cfg = copy.copy(cfg)
    setattr(worker_cfg, "_runtime_events", [])
    setattr(worker_cfg, "_runtime_context", {})
    set_seed(int(worker_cfg.random_seed) + int(trial))

    prepared = prepare_feature_data(worker_cfg, spec, trial, feature_set=feature_set)
    split_audit = _split_assignment_table(worker_cfg, prepared, spec, trial)
    feature_audit = _feature_processing_audit(worker_cfg, prepared, spec, trial)
    pool_lifetime_median = float(prepared.pool_table["lifetime"].median())
    inventory_row = {
        **_feature_set_context(prepared),
        "dataset": spec.name,
        "trial": trial,
        "n_filtered_samples": len(prepared.table),
        "n_pool": len(prepared.pool_indices),
        "n_holdout": len(prepared.test_indices),
        "pool_lifetime_median": pool_lifetime_median,
        "n_pool_long_life": int((prepared.pool_table["lifetime"] > pool_lifetime_median).sum()),
        "n_holdout_long_life": int((prepared.test_table["lifetime"] > pool_lifetime_median).sum()),
        "min_lifetime": float(prepared.table["lifetime"].min()),
        "max_lifetime": float(prepared.table["lifetime"].max()),
    }
    rows, preds, selected = _run_total_budget_scan(worker_cfg, prepared, spec, trial)
    return {
        "dataset": spec.name,
        "trial": int(trial),
        "metrics_rows": rows,
        "predictions": pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(),
        "selected": pd.concat(selected, ignore_index=True) if selected else pd.DataFrame(),
        "split_assignments": split_audit,
        "feature_processing_row": feature_audit,
        "inventory_row": inventory_row,
        "runtime_events": _runtime_events_frame(worker_cfg),
    }


def _run_trial_jobs_for_dataset(
    cfg: ExperimentConfig,
    spec,
    feature_set: FeatureSetSpec,
    trial_ids: list[int],
) -> list[dict[str, object]]:
    parallel_jobs = _active_learning_parallel_jobs(cfg)
    if parallel_jobs == 1 or len(trial_ids) <= 1:
        return [_run_trial_job(_fresh_trial_job_config(cfg), spec, feature_set, trial) for trial in trial_ids]

    max_workers = min(parallel_jobs, len(trial_ids))
    print(
        f"[Parallel] feature_set={feature_set.name} dataset={spec.name} "
        f"trials={trial_ids} parallel_jobs={max_workers}"
    )
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_trial_job, _fresh_trial_job_config(cfg), spec, feature_set, trial): int(trial)
            for trial in trial_ids
        }
        for future in as_completed(futures):
            trial = futures[future]
            result = future.result()
            print(f"[Parallel done] feature_set={feature_set.name} dataset={spec.name} trial={trial}")
            results.append(result)
    return sorted(results, key=lambda item: int(item["trial"]))


def _safe_run_label(value: str) -> str:
    label = str(value).strip().replace("-", "_").replace(" ", "_")
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in label)
    safe = "_".join(part for part in safe.split("_") if part)
    return safe or "unknown"


def _feature_set_number(feature_set: FeatureSetSpec | str) -> str | None:
    value = feature_set.name if isinstance(feature_set, FeatureSetSpec) else str(feature_set)
    digits = []
    collecting = False
    for char in value:
        if char.isdigit():
            digits.append(char)
            collecting = True
        elif collecting:
            break
    return "".join(digits) if digits else None


def _feature_set_directory_name(feature_set: FeatureSetSpec | str) -> str:
    number = _feature_set_number(feature_set)
    if number:
        return f"feature set {int(number)}"
    value = feature_set.name if isinstance(feature_set, FeatureSetSpec) else str(feature_set)
    return _safe_run_label(value).replace("_", " ")


def _target_split_directory_name(cfg: ExperimentConfig) -> str:
    target = canonical_target_transform(getattr(cfg, "target_transform", "log"))
    split = split_mode_label(getattr(cfg, "pool_holdout_split_mode", "protocol_holdout"))
    return f"target-{target}__split-{split}"


def _scoped_output_root(cfg: ExperimentConfig, root: Path) -> Path:
    return Path(root) / _target_split_directory_name(cfg)


def _feature_set_output_root(cfg: ExperimentConfig, feature_set: FeatureSetSpec | str) -> Path:
    return _scoped_output_root(cfg, cfg.output_root) / _feature_set_directory_name(feature_set)


def _portable_relative_path(path: Path, base: Path) -> str:
    return Path(path).relative_to(Path(base)).as_posix()


def _portable_run_config(cfg: ExperimentConfig) -> dict[str, object]:
    payload = cfg.to_jsonable()
    for field in ("project_root", "output_root", "dl_output_root"):
        payload.pop(field, None)
    return payload


def _write_feature_set_run_index(output_root: Path, rows: list[dict[str, str]]) -> Path:
    index_path = output_root / "feature_set_runs.csv"
    new_rows = pd.DataFrame(rows)
    frames = []
    if index_path.exists():
        frames.append(pd.read_csv(index_path))
    if not new_rows.empty:
        frames.append(new_rows)
    if frames:
        merged = pd.concat(frames, ignore_index=True)
        if "feature_set" in merged.columns:
            merged = merged.drop_duplicates(subset=["feature_set"], keep="last")
            merged["_feature_set_order"] = merged["feature_set"].map(
                lambda value: int(_feature_set_number(str(value)) or 10_000)
            )
            merged = merged.sort_values(["_feature_set_order", "feature_set"]).drop(columns=["_feature_set_order"])
    else:
        merged = pd.DataFrame(columns=["feature_set", "feature_set_label", "feature_run_dir", "metrics_summary"])
    merged.to_csv(index_path, index=False)
    return index_path


def _feature_dataset_dirs(feature_root: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in Path(feature_root).iterdir()
            if path.is_dir() and (path / "metrics.csv").exists()
        ],
        key=lambda path: path.name,
    )


def _read_feature_dataset_frames(feature_root: Path, filename: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dataset_dir in _feature_dataset_dirs(feature_root):
        path = dataset_dir / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        if "dataset" not in frame.columns:
            frame.insert(0, "dataset", dataset_dir.name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _bool_split_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    values = frame[column]
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def _rebuild_data_inventory_from_split_summary(split_summary: pd.DataFrame, feature_processing_summary: pd.DataFrame) -> pd.DataFrame:
    if split_summary.empty or "lifetime" not in split_summary.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    audit_lookup = {}
    if not feature_processing_summary.empty and {"dataset", "trial"}.issubset(feature_processing_summary.columns):
        for _, audit_row in feature_processing_summary.iterrows():
            audit_lookup[(str(audit_row["dataset"]), int(audit_row["trial"]))] = audit_row
    group_cols = [col for col in ["feature_set", "dataset", "trial"] if col in split_summary.columns]
    if not {"dataset", "trial"}.issubset(group_cols):
        return pd.DataFrame()
    for keys, part in split_summary.groupby(group_cols, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        group = dict(zip(group_cols, key_values))
        dataset = str(group.get("dataset", ""))
        trial = int(group.get("trial", 0))
        audit_row = audit_lookup.get((dataset, trial))
        pool_mask = _bool_split_column(part, "is_pool")
        holdout_mask = _bool_split_column(part, "is_holdout")
        pool = part[pool_mask]
        holdout = part[holdout_mask]
        pool_median = float(pool["lifetime"].median()) if not pool.empty else float("nan")

        def audit_value(name: str, default: object = "") -> object:
            if audit_row is not None and name in audit_row.index and pd.notna(audit_row[name]):
                return audit_row[name]
            if name in part.columns and not part[name].dropna().empty:
                return part[name].dropna().iloc[0]
            return default

        rows.append(
            {
                "feature_set": audit_value("feature_set", group.get("feature_set", "")),
                "feature_set_label": audit_value("feature_set_label", ""),
                "feature_matrix_key": audit_value("feature_matrix_key", ""),
                "n_original_features": audit_value("n_original_features", np.nan),
                "n_features": audit_value("n_features", np.nan),
                "retained_feature_indices": audit_value("retained_feature_indices", ""),
                "dataset": dataset,
                "trial": trial,
                "n_filtered_samples": int(len(part)),
                "n_pool": int(pool_mask.sum()),
                "n_holdout": int(holdout_mask.sum()),
                "pool_lifetime_median": pool_median,
                "n_pool_long_life": int((pool["lifetime"].astype(float) > pool_median).sum()) if not pool.empty else 0,
                "n_holdout_long_life": int((holdout["lifetime"].astype(float) > pool_median).sum())
                if not holdout.empty and np.isfinite(pool_median)
                else 0,
                "min_lifetime": float(part["lifetime"].min()),
                "max_lifetime": float(part["lifetime"].max()),
            }
        )
    return pd.DataFrame(rows)


def _collect_feature_set_dataset_outputs(feature_root: Path, save_predictions: bool = True) -> dict[str, pd.DataFrame]:
    metrics_summary = _read_feature_dataset_frames(feature_root, "metrics.csv")
    metrics_summary = add_pool_accuracy_targets(_apply_max_budget_all_pool_baseline(metrics_summary))
    predictions_summary = _read_feature_dataset_frames(feature_root, "predictions.csv") if save_predictions else pd.DataFrame()
    selected_summary = _read_feature_dataset_frames(feature_root, "selected_indices.csv") if save_predictions else pd.DataFrame()
    trajectory_summary = _read_feature_dataset_frames(feature_root, "acquisition_trajectory.csv") if save_predictions else pd.DataFrame()
    split_summary = _read_feature_dataset_frames(feature_root, "split_assignments.csv")
    feature_processing_summary = _read_feature_dataset_frames(feature_root, "feature_processing_audit.csv")
    baseline_df = _max_budget_all_pool_baseline_metrics(metrics_summary)
    calibration_summary = _prediction_calibration_summary(predictions_summary, "active_learning")
    cost_summary = summarize_pool_accuracy_cost(
        metrics_summary,
        all_pool_accuracy_fraction=DEFAULT_ALL_POOL_ACCURACY_FRACTION,
    )
    inventory_df = _rebuild_data_inventory_from_split_summary(split_summary, feature_processing_summary)
    return {
        "metrics_summary": metrics_summary,
        "predictions_summary": predictions_summary,
        "selected_indices_summary": selected_summary,
        "acquisition_trajectory_summary": trajectory_summary,
        "all_pool_baseline_metrics": baseline_df,
        "split_assignments_summary": split_summary,
        "feature_processing_audit": feature_processing_summary,
        "uncertainty_calibration_summary": calibration_summary,
        "cost_to_95pct_all_pool_accuracy": cost_summary,
        "data_inventory": inventory_df,
    }


def run_experiments(cfg: ExperimentConfig) -> Path:
    target_modes = getattr(cfg, "run_target_transforms", None)
    split_modes = getattr(cfg, "run_split_modes", None)
    if target_modes is not None or split_modes is not None:
        output_root = ensure_dir(cfg.output_root)
        target_values = tuple(target_modes) if target_modes is not None else (getattr(cfg, "target_transform", "log"),)
        split_values = tuple(split_modes) if split_modes is not None else (getattr(cfg, "pool_holdout_split_mode", "protocol_holdout"),)
        run_rows: list[dict[str, str]] = []
        for target in target_values:
            for split in split_values:
                subcfg = copy.copy(cfg)
                subcfg.target_transform = canonical_target_transform(target)
                subcfg.pool_holdout_split_mode = canonical_split_mode(split)
                subcfg.run_target_transforms = None
                subcfg.run_split_modes = None
                out_dir = run_experiments(subcfg)
                run_rows.append(
                    {
                        "target_transform": subcfg.target_transform,
                        "pool_holdout_split_mode": subcfg.pool_holdout_split_mode,
                        "pool_holdout_split_label": split_mode_label(subcfg.pool_holdout_split_mode),
                        "run_dir": _portable_relative_path(out_dir, output_root),
                    }
                )
        pd.DataFrame(run_rows).to_csv(output_root / "target_split_runs.csv", index=False)
        save_json({"target_split_runs": "target_split_runs.csv"}, output_root / "run_summary.json")
        return output_root

    set_seed(cfg.random_seed)
    output_root = ensure_dir(_scoped_output_root(cfg, cfg.output_root))
    setattr(cfg, "_runtime_events", [])
    setattr(cfg, "_runtime_context", {})

    enabled = set(cfg.run_datasets) if cfg.run_datasets else None
    feature_run_summaries: list[dict[str, str]] = []
    selected_feature_sets = cfg.selected_feature_sets()
    last_feature_root: Path | None = None

    for feature_set in selected_feature_sets:
        feature_root = ensure_dir(_feature_set_output_root(cfg, feature_set))
        last_feature_root = feature_root
        save_json(_portable_run_config(cfg), feature_root / "run_config.json")
        feature_metrics: list[pd.DataFrame] = []
        feature_predictions: list[pd.DataFrame] = []
        feature_selected: list[pd.DataFrame] = []
        feature_split_assignments: list[pd.DataFrame] = []
        feature_processing_rows: list[dict[str, object]] = []
        inventory: list[dict[str, object]] = []

        for spec in cfg.datasets:
            if enabled and spec.name not in enabled:
                continue
            dataset_metrics = []
            dataset_predictions = []
            dataset_selected = []
            dataset_split_assignments = []
            dataset_feature_processing_rows = []
            trial_results = _run_trial_jobs_for_dataset(cfg, spec, feature_set, _configured_trial_ids(cfg))
            for result in trial_results:
                split_audit = result["split_assignments"]
                feature_audit = result["feature_processing_row"]
                predictions = result["predictions"]
                selected = result["selected"]
                runtime_events = result["runtime_events"]
                dataset_split_assignments.append(split_audit)
                dataset_feature_processing_rows.append(feature_audit)
                feature_split_assignments.append(split_audit)
                feature_processing_rows.append(feature_audit)
                inventory.append(result["inventory_row"])
                dataset_metrics.extend(result["metrics_rows"])
                if isinstance(predictions, pd.DataFrame) and not predictions.empty:
                    dataset_predictions.append(predictions)
                if isinstance(selected, pd.DataFrame) and not selected.empty:
                    dataset_selected.append(selected)
                if isinstance(runtime_events, pd.DataFrame) and not runtime_events.empty:
                    events = getattr(cfg, "_runtime_events", None)
                    if isinstance(events, list):
                        events.extend(runtime_events.to_dict("records"))

            dataset_dir = ensure_dir(feature_root / spec.name)
            metrics_df = _merge_partial_trial_output(
                cfg,
                dataset_dir / "metrics.csv",
                pd.DataFrame(dataset_metrics),
                ["feature_set", "dataset", "trial", "regressor", "method", "budget_total"],
            )
            metrics_df = add_pool_accuracy_targets(_apply_max_budget_all_pool_baseline(metrics_df))
            pred_df = _merge_partial_trial_output(
                cfg,
                dataset_dir / "predictions.csv",
                pd.concat(dataset_predictions, ignore_index=True) if dataset_predictions else pd.DataFrame(),
                ["feature_set", "dataset", "trial", "regressor", "method", "budget_total", "index"],
            )
            selected_df = _merge_partial_trial_output(
                cfg,
                dataset_dir / "selected_indices.csv",
                pd.concat(dataset_selected, ignore_index=True) if dataset_selected else pd.DataFrame(),
                ["feature_set", "dataset", "trial", "regressor", "method", "budget_total", "selection_order"],
            )
            baseline_df = _max_budget_all_pool_baseline_metrics(metrics_df)
            split_df = _merge_partial_trial_output(
                cfg,
                dataset_dir / "split_assignments.csv",
                pd.concat(dataset_split_assignments, ignore_index=True) if dataset_split_assignments else pd.DataFrame(),
                ["feature_set", "dataset", "trial", "index"],
            )
            feature_processing_df = _merge_partial_trial_output(
                cfg,
                dataset_dir / "feature_processing_audit.csv",
                pd.DataFrame(dataset_feature_processing_rows),
                ["feature_set", "dataset", "trial"],
            )
            calibration_df = _prediction_calibration_summary(pred_df, "active_learning")
            dataset_cost_summary = summarize_pool_accuracy_cost(
                metrics_df,
                all_pool_accuracy_fraction=DEFAULT_ALL_POOL_ACCURACY_FRACTION,
            )
            metrics_df.to_csv(dataset_dir / "metrics.csv", index=False)
            baseline_df.to_csv(dataset_dir / "all_pool_baseline_metrics.csv", index=False)
            split_df.to_csv(dataset_dir / "split_assignments.csv", index=False)
            feature_processing_df.to_csv(dataset_dir / "feature_processing_audit.csv", index=False)
            calibration_df.to_csv(dataset_dir / "uncertainty_calibration_summary.csv", index=False)
            dataset_cost_summary.to_csv(dataset_dir / "cost_to_95pct_all_pool_accuracy.csv", index=False)
            if cfg.save_predictions:
                pred_df.to_csv(dataset_dir / "predictions.csv", index=False)
                selected_df.to_csv(dataset_dir / "selected_indices.csv", index=False)
                selected_df.to_csv(dataset_dir / "acquisition_trajectory.csv", index=False)
            if cfg.save_figures and not metrics_df.empty:
                fig_dir = ensure_dir(dataset_dir / "figures")
                plot_dataset_outputs(
                    metrics_df,
                    pred_df,
                    fig_dir,
                    spec.name,
                    iterative_batch_size=_dataset_iterative_batch_size(cfg, spec.name),
                )
            feature_metrics.append(metrics_df)
            if not pred_df.empty:
                feature_predictions.append(pred_df)
            if not selected_df.empty:
                feature_selected.append(selected_df)

        summaries = _collect_feature_set_dataset_outputs(feature_root, save_predictions=bool(cfg.save_predictions))
        metrics_summary = summaries["metrics_summary"]
        predictions_summary = summaries["predictions_summary"]
        selected_summary = summaries["selected_indices_summary"]
        trajectory_summary = summaries["acquisition_trajectory_summary"]
        baseline_df = summaries["all_pool_baseline_metrics"]
        inventory_df = summaries["data_inventory"]
        split_summary = summaries["split_assignments_summary"]
        feature_processing_summary = summaries["feature_processing_audit"]
        calibration_summary = summaries["uncertainty_calibration_summary"]
        cost_summary = summaries["cost_to_95pct_all_pool_accuracy"]
        runtime_events = _runtime_events_frame(cfg)
        feature_runtime_events = (
            runtime_events[runtime_events["feature_set"].astype(str) == feature_set.name].copy()
            if not runtime_events.empty and "feature_set" in runtime_events.columns
            else pd.DataFrame()
        )
        feature_runtime_summary = _runtime_summary(feature_runtime_events)
        if cfg.save_figures and not metrics_summary.empty:
            plot_mape_budget_overview(metrics_summary, ensure_dir(feature_root / "figures"), iterative_batch_size=None)

        metrics_summary = _write_merged_summary(
            feature_root / "metrics_summary.csv",
            metrics_summary,
            ["feature_set", "dataset", "trial", "regressor", "method", "budget_total"],
        )
        baseline_df = _write_merged_summary(
            feature_root / "all_pool_baseline_metrics.csv",
            baseline_df,
            ["feature_set", "dataset", "regressor", "method", "budget_total"],
        )
        inventory_df = _write_merged_summary(
            feature_root / "data_inventory.csv",
            inventory_df,
            ["feature_set", "dataset", "trial"],
        )
        split_summary = _write_merged_summary(
            feature_root / "split_assignments_summary.csv",
            split_summary,
            ["feature_set", "dataset", "trial", "index"],
        )
        feature_processing_summary = _write_merged_summary(
            feature_root / "feature_processing_audit.csv",
            feature_processing_summary,
            ["feature_set", "dataset", "trial"],
        )
        calibration_summary = _write_merged_summary(
            feature_root / "uncertainty_calibration_summary.csv",
            calibration_summary,
            ["prediction_source", "feature_set", "dataset", "trial", "experiment_mode", "regressor", "method", "budget_total", "iterative_batch_size", "uncertainty_bin"],
        )
        cost_summary = _write_merged_summary(
            feature_root / "cost_to_95pct_all_pool_accuracy.csv",
            cost_summary,
            ["feature_set", "dataset", "trial", "experiment_mode", "regressor", "method", "budget_total", "iterative_batch_size"],
        )
        feature_runtime_events.to_csv(feature_root / "runtime_events.csv", index=False)
        feature_runtime_summary.to_csv(feature_root / "runtime_summary.csv", index=False)
        if cfg.save_predictions:
            predictions_summary = _write_merged_summary(
                feature_root / "predictions_summary.csv",
                predictions_summary,
                ["feature_set", "dataset", "trial", "regressor", "method", "budget_total", "index"],
            )
            selected_summary = _write_merged_summary(
                feature_root / "selected_indices_summary.csv",
                selected_summary,
                ["feature_set", "dataset", "trial", "regressor", "method", "budget_total", "selection_order"],
            )
            trajectory_summary = _write_merged_summary(
                feature_root / "acquisition_trajectory_summary.csv",
                trajectory_summary,
                ["feature_set", "dataset", "trial", "regressor", "method", "budget_total", "selection_order"],
            )
        else:
            _cleanup_prediction_outputs_when_disabled(feature_root)
        save_json(
            {
                "feature_set": feature_set.name,
                "feature_set_label": feature_set.label,
                "metrics_summary": "metrics_summary.csv",
                "baseline_metrics": "all_pool_baseline_metrics.csv",
                "baseline_definition": "mean performance of max-budget active-learning checkpoints across trials",
                "cost_to_95pct_all_pool_accuracy": "cost_to_95pct_all_pool_accuracy.csv",
                "updated_datasets_this_run": [spec.name for spec in cfg.datasets if not enabled or spec.name in enabled],
                "summary_merged_from_dataset_dirs": [path.name for path in _feature_dataset_dirs(feature_root)],
            },
            feature_root / "run_summary.json",
        )
        feature_run_summaries.append(
            {
                "feature_set": feature_set.name,
                "feature_set_label": feature_set.label,
                "feature_run_dir": _portable_relative_path(feature_root, output_root),
                "metrics_summary": _portable_relative_path(feature_root / "metrics_summary.csv", output_root),
            }
        )
    feature_set_runs_path = _write_feature_set_run_index(output_root, feature_run_summaries)
    save_json(
        {
            "feature_set_runs": _portable_relative_path(feature_set_runs_path, output_root),
            "output_layout": "Each feature set is saved in a fixed top-level directory: feature set 1, ..., feature set 5.",
        },
        output_root / "run_summary.json",
    )
    return last_feature_root if len(selected_feature_sets) == 1 and last_feature_root is not None else output_root

