from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


MatrixFn = Callable[[pd.DataFrame], np.ndarray]
UncertaintyFn = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]
ErrorRiskFn = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]

ACTIVE_LEARNING_METHODS = (
    "random_selection",
    "diversity_oneshot",
    "diversity_iterative",
    "coverage",
    "uncertainty",
    "qbc",
    "max_error_exploitation",
    "hybrid",
)


def select_by_coverage(
    x_labeled: np.ndarray,
    x_candidate: np.ndarray,
    budget: int,
    rng: np.random.Generator,
) -> np.ndarray:
    # Coverage: farthest-first traversal covers sparse regions of feature space,
    # using already labeled cells as anchors when available.
    if budget <= 0 or len(x_candidate) == 0:
        return np.array([], dtype=int)
    budget = min(int(budget), len(x_candidate))
    if len(x_labeled):
        anchors = x_labeled
    else:
        # In the total-budget benchmark there is no labeled seed set. Use a
        # random anchor so coverage is not biased by the candidate table order.
        anchor_idx = int(rng.integers(0, len(x_candidate)))
        anchors = x_candidate[[anchor_idx]]
    min_dist = np.min(((x_candidate[:, None, :] - anchors[None, :, :]) ** 2).sum(axis=2), axis=1)
    selected: list[int] = []
    for _ in range(budget):
        idx = int(np.argmax(min_dist))
        selected.append(idx)
        dist_new = ((x_candidate - x_candidate[idx]) ** 2).sum(axis=1)
        min_dist = np.minimum(min_dist, dist_new)
        min_dist[selected] = -np.inf
    return np.asarray(selected, dtype=int)


def select_by_kmeans_diversity(x_candidate: np.ndarray, budget: int, rng: np.random.Generator) -> np.ndarray:
    # Diversity: cluster the candidate pool and pick one representative cell
    # closest to each k-means centroid.
    if budget <= 0 or len(x_candidate) == 0:
        return np.array([], dtype=int)
    budget = min(int(budget), len(x_candidate))
    if budget == len(x_candidate):
        return np.arange(len(x_candidate), dtype=int)
    random_state = int(rng.integers(0, np.iinfo(np.int32).max))
    kmeans = KMeans(n_clusters=budget, n_init=10, random_state=random_state)
    kmeans.fit(x_candidate)
    distances = pairwise_distances(x_candidate, kmeans.cluster_centers_)
    selected: list[int] = []
    for center_id in range(budget):
        for idx in np.argsort(distances[:, center_id]):
            idx = int(idx)
            if idx not in selected:
                selected.append(idx)
                break
    if len(selected) < budget:
        selected.extend([i for i in range(len(x_candidate)) if i not in selected][: budget - len(selected)])
    return np.asarray(selected[:budget], dtype=int)


def _rank_coverage_candidates(
    train_table: pd.DataFrame,
    candidate_table: pd.DataFrame,
    matrix_fn: MatrixFn,
) -> pd.DataFrame:
    # feature_matrix() already returns coordinates standardized by the
    # pool-fitted scaler in data.prepare_feature_data().
    x_labeled = matrix_fn(train_table)
    x_candidate = matrix_fn(candidate_table)
    if len(x_labeled):
        min_dist = np.min(((x_candidate[:, None, :] - x_labeled[None, :, :]) ** 2).sum(axis=2), axis=1)
    else:
        min_dist = np.zeros(len(candidate_table), dtype=float)
    ranked = pd.DataFrame(
        {
            "index": candidate_table["index"].to_numpy(dtype=int),
            "acquisition_score": min_dist.astype(float),
            "lifetime_mean": np.nan,
            "target_std": np.nan,
        }
    ).set_index("index")
    ranked = ranked.sort_values("acquisition_score", ascending=False)
    ranked["acquisition_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
    return ranked


def _coverage_scores(
    train_table: pd.DataFrame,
    candidate_table: pd.DataFrame,
    matrix_fn: MatrixFn,
) -> np.ndarray:
    x_labeled = matrix_fn(train_table)
    x_candidate = matrix_fn(candidate_table)
    if len(x_labeled):
        return np.min(((x_candidate[:, None, :] - x_labeled[None, :, :]) ** 2).sum(axis=2), axis=1)
    return np.zeros(len(candidate_table), dtype=float)


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    finite = np.isfinite(values)
    clean = np.where(finite, values, np.nanmin(values[finite]) if finite.any() else 0.0)
    return pd.Series(clean).rank(pct=True).to_numpy(float)


def _uncertainty_scores(uncertainty_fn: UncertaintyFn, train_table: pd.DataFrame, candidate_table: pd.DataFrame) -> pd.DataFrame:
    scores = uncertainty_fn(train_table, candidate_table)
    required = {"index", "lifetime_mean", "target_std"}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"Uncertainty function missing columns: {sorted(missing)}")
    return scores.set_index("index").loc[candidate_table["index"].to_numpy(dtype=int)]


def _rank_model_aware_candidates(
    train_table: pd.DataFrame,
    candidate_table: pd.DataFrame,
    uncertainty_fn: UncertaintyFn,
    mode: str,
) -> pd.DataFrame:
    scores = _uncertainty_scores(uncertainty_fn, train_table, candidate_table)
    if mode == "std":
        # Exploration: query cells where the current predictor is most uncertain
        # in the configured learning-target space.
        acquisition_score = scores["target_std"].fillna(0.0).to_numpy(float)
    else:
        raise ValueError(f"Unknown uncertainty mode: {mode}")
    ranked = scores.copy()
    ranked["acquisition_score"] = acquisition_score
    ranked = ranked.sort_values("acquisition_score", ascending=False)
    ranked["acquisition_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
    return ranked


def _rank_error_risk_candidates(
    train_table: pd.DataFrame,
    candidate_table: pd.DataFrame,
    error_risk_fn: ErrorRiskFn,
) -> pd.DataFrame:
    scores = error_risk_fn(train_table, candidate_table)
    required = {"index", "acquisition_score"}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"Error-risk function missing columns: {sorted(missing)}")
    ranked = scores.set_index("index").loc[candidate_table["index"].to_numpy(dtype=int)]
    if "lifetime_mean" not in ranked.columns:
        ranked["lifetime_mean"] = np.nan
    if "target_std" not in ranked.columns:
        ranked["target_std"] = np.nan
    ranked = ranked.sort_values("acquisition_score", ascending=False)
    ranked["acquisition_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
    return ranked


def _rank_hybrid_candidates(
    uncertainty_train_table: pd.DataFrame,
    coverage_anchor_table: pd.DataFrame,
    candidate_table: pd.DataFrame,
    matrix_fn: MatrixFn,
    uncertainty_fn: UncertaintyFn,
) -> pd.DataFrame:
    scores = _uncertainty_scores(uncertainty_fn, uncertainty_train_table, candidate_table)
    coverage = _coverage_scores(coverage_anchor_table, candidate_table, matrix_fn)
    uncertainty = scores["target_std"].fillna(0.0).to_numpy(float)
    coverage_rank = _percentile_rank(coverage)
    uncertainty_rank = _percentile_rank(uncertainty)
    ranked = scores.copy()
    ranked["coverage_score"] = coverage.astype(float)
    ranked["coverage_rank_score"] = coverage_rank.astype(float)
    ranked["uncertainty_rank_score"] = uncertainty_rank.astype(float)
    ranked["acquisition_score"] = coverage_rank * uncertainty_rank
    ranked = ranked.sort_values("acquisition_score", ascending=False)
    ranked["acquisition_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
    return ranked


def _rank_hybrid_candidates_from_uncertainty(
    uncertainty_scores: pd.DataFrame,
    coverage_anchor_table: pd.DataFrame,
    candidate_table: pd.DataFrame,
    matrix_fn: MatrixFn,
) -> pd.DataFrame:
    scores_with_index = uncertainty_scores
    if "index" in scores_with_index.columns:
        scores_with_index = scores_with_index.set_index("index")
    required = {"lifetime_mean", "target_std"}
    missing = required.difference(scores_with_index.columns)
    if missing:
        raise ValueError(f"Hybrid uncertainty scores missing columns: {sorted(missing)}")
    scores = scores_with_index.loc[candidate_table["index"].to_numpy(dtype=int)]
    coverage = _coverage_scores(coverage_anchor_table, candidate_table, matrix_fn)
    uncertainty = scores["target_std"].fillna(0.0).to_numpy(float)
    coverage_rank = _percentile_rank(coverage)
    uncertainty_rank = _percentile_rank(uncertainty)
    ranked = scores.copy()
    ranked["coverage_score"] = coverage.astype(float)
    ranked["coverage_rank_score"] = coverage_rank.astype(float)
    ranked["uncertainty_rank_score"] = uncertainty_rank.astype(float)
    ranked["acquisition_score"] = coverage_rank * uncertainty_rank
    ranked = ranked.sort_values("acquisition_score", ascending=False)
    ranked["acquisition_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
    return ranked


def _trace_frame(indices: np.ndarray, method: str, phase: str, score_table: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    score_table = score_table if score_table is not None else pd.DataFrame()
    for order, idx in enumerate(np.asarray(indices, dtype=int).tolist(), start=1):
        values = score_table.loc[idx].to_dict() if idx in score_table.index else {}
        rows.append(
            {
                "selected_index": int(idx),
                "selection_order": order,
                "selection_phase": phase,
                "predicted_lifetime_at_selection": values.get("lifetime_mean", np.nan),
                "uncertainty_score": values.get("target_std", np.nan),
                "target_std": values.get("target_std", np.nan),
                "acquisition_score": values.get("acquisition_score", np.nan),
                "acquisition_rank": values.get("acquisition_rank", np.nan),
            }
        )
    return pd.DataFrame(rows)


def _cold_start_selection(
    candidate_table: pd.DataFrame,
    initial_step: int,
    rng: np.random.Generator,
    cold_start_seed: int | None,
) -> np.ndarray:
    cold_rng = np.random.default_rng(int(cold_start_seed)) if cold_start_seed is not None else rng
    initial_rel = cold_rng.choice(len(candidate_table), size=initial_step, replace=False)
    return candidate_table.iloc[initial_rel]["index"].to_numpy(dtype=int)


def _select_random_in_batches(
    candidate_table: pd.DataFrame,
    budget: int,
    rng: np.random.Generator,
    batch_size: int,
    return_trace: bool = False,
    cold_start_seed: int | None = None,
) -> np.ndarray | tuple[np.ndarray, pd.DataFrame]:
    budget = min(int(budget), len(candidate_table))
    if budget <= 0:
        empty = np.array([], dtype=int)
        return (empty, _trace_frame(empty, "random_selection", "")) if return_trace else empty
    batch_size = max(1, int(batch_size))
    selected: list[int] = []
    selected_set: set[int] = set()
    trace_parts: list[pd.DataFrame] = []

    first_step = min(batch_size, budget, len(candidate_table))
    first = _cold_start_selection(candidate_table, first_step, rng, cold_start_seed)
    selected.extend(int(idx) for idx in first.tolist())
    selected_set.update(int(idx) for idx in first.tolist())
    first_phase = "random_cold_start" if cold_start_seed is not None else "random_batch_acquisition"
    trace_parts.append(_trace_frame(first, "random_selection", first_phase))

    while len(selected) < budget:
        remaining = candidate_table[~candidate_table["index"].isin(selected_set)]
        if remaining.empty:
            break
        step = min(batch_size, budget - len(selected), len(remaining))
        rel = rng.choice(len(remaining), size=step, replace=False)
        chosen = remaining.iloc[rel]["index"].to_numpy(dtype=int)
        trace_parts.append(_trace_frame(chosen, "random_selection", "random_batch_acquisition"))
        selected.extend(int(idx) for idx in chosen.tolist())
        selected_set.update(int(idx) for idx in chosen.tolist())

    selected_arr = np.asarray(selected[:budget], dtype=int)
    trace = pd.concat(trace_parts, ignore_index=True)
    trace = trace[trace["selected_index"].isin(selected_arr)].copy()
    trace["selection_order"] = np.arange(1, len(trace) + 1, dtype=int)
    return (selected_arr, trace) if return_trace else selected_arr


def _select_diversity_oneshot(
    candidate_table: pd.DataFrame,
    matrix_fn: MatrixFn,
    budget: int,
    rng: np.random.Generator,
    return_trace: bool = False,
) -> np.ndarray | tuple[np.ndarray, pd.DataFrame]:
    x_candidate = matrix_fn(candidate_table)
    rel = select_by_kmeans_diversity(x_candidate, budget, rng)
    selected = candidate_table.iloc[rel]["index"].to_numpy(dtype=int)
    trace = _trace_frame(selected, "diversity_oneshot", "diversity_oneshot_acquisition")
    return (selected, trace) if return_trace else selected


def _select_diversity_iteratively(
    candidate_table: pd.DataFrame,
    matrix_fn: MatrixFn,
    budget: int,
    rng: np.random.Generator,
    batch_size: int,
    return_trace: bool = False,
) -> np.ndarray | tuple[np.ndarray, pd.DataFrame]:
    budget = min(int(budget), len(candidate_table))
    if budget <= 0:
        empty = np.array([], dtype=int)
        return (empty, _trace_frame(empty, "diversity_iterative", "")) if return_trace else empty
    batch_size = max(1, int(batch_size))
    selected: list[int] = []
    selected_set: set[int] = set()
    trace_parts: list[pd.DataFrame] = []
    while len(selected) < budget:
        remaining = candidate_table[~candidate_table["index"].isin(selected_set)]
        if remaining.empty:
            break
        step = min(batch_size, budget - len(selected), len(remaining))
        rel = select_by_kmeans_diversity(matrix_fn(remaining), step, rng)
        chosen = remaining.iloc[rel]["index"].to_numpy(dtype=int)
        trace_parts.append(_trace_frame(chosen, "diversity_iterative", "diversity_iterative_acquisition"))
        selected.extend(int(idx) for idx in chosen.tolist())
        selected_set.update(int(idx) for idx in chosen.tolist())
    selected_arr = np.asarray(selected[:budget], dtype=int)
    trace = pd.concat(trace_parts, ignore_index=True) if trace_parts else _trace_frame(selected_arr, "diversity_iterative", "")
    trace = trace[trace["selected_index"].isin(selected_arr)].copy()
    trace["selection_order"] = np.arange(1, len(trace) + 1, dtype=int)
    return (selected_arr, trace) if return_trace else selected_arr


def _select_iteratively(
    initial_label_table: pd.DataFrame,
    candidate_table: pd.DataFrame,
    matrix_fn: MatrixFn,
    uncertainty_fn: UncertaintyFn,
    budget: int,
    rng: np.random.Generator,
    batch_size: int,
    mode: str,
    return_trace: bool = False,
    cold_start_seed: int | None = None,
) -> np.ndarray | tuple[np.ndarray, pd.DataFrame]:
    budget = min(int(budget), len(candidate_table))
    if budget <= 0:
        empty = np.array([], dtype=int)
        return (empty, _trace_frame(empty, "", "")) if return_trace else empty
    batch_size = max(1, int(batch_size))
    selected: list[int] = []
    selected_set: set[int] = set()
    trace_parts: list[pd.DataFrame] = []
    if initial_label_table.empty:
        initial_step = min(batch_size, budget, len(candidate_table))
        initial = _cold_start_selection(candidate_table, initial_step, rng, cold_start_seed)
        for idx in initial.tolist():
            selected.append(int(idx))
            selected_set.add(int(idx))
        trace_parts.append(_trace_frame(initial, "", "random_cold_start"))
        if len(selected) >= budget:
            selected_arr = np.asarray(selected[:budget], dtype=int)
            trace = pd.concat(trace_parts, ignore_index=True)
            trace["selection_order"] = np.arange(1, len(trace) + 1, dtype=int)
            return (selected_arr, trace) if return_trace else selected_arr
    while len(selected) < budget:
        remaining = candidate_table[~candidate_table["index"].isin(selected_set)]
        if remaining.empty:
            break
        train = pd.concat([initial_label_table, candidate_table[candidate_table["index"].isin(selected_set)]], ignore_index=True)
        step = min(batch_size, budget - len(selected), len(remaining))
        ranked = _rank_model_aware_candidates(train, remaining, uncertainty_fn, mode)
        chosen = ranked.head(step).index.to_numpy(dtype=int)
        trace_parts.append(_trace_frame(chosen, "", "model_aware_acquisition", ranked))
        for idx in chosen.tolist():
            if idx not in selected_set:
                selected.append(int(idx))
                selected_set.add(int(idx))
    selected_arr = np.asarray(selected[:budget], dtype=int)
    trace = pd.concat(trace_parts, ignore_index=True) if trace_parts else _trace_frame(selected_arr, "", "")
    trace = trace[trace["selected_index"].isin(selected_arr)].copy()
    trace["selection_order"] = np.arange(1, len(trace) + 1, dtype=int)
    return (selected_arr, trace) if return_trace else selected_arr


def _select_error_risk_iteratively(
    initial_label_table: pd.DataFrame,
    candidate_table: pd.DataFrame,
    error_risk_fn: ErrorRiskFn,
    budget: int,
    rng: np.random.Generator,
    batch_size: int,
    return_trace: bool = False,
    cold_start_seed: int | None = None,
) -> np.ndarray | tuple[np.ndarray, pd.DataFrame]:
    budget = min(int(budget), len(candidate_table))
    if budget <= 0:
        empty = np.array([], dtype=int)
        return (empty, _trace_frame(empty, "max_error_exploitation", "")) if return_trace else empty
    batch_size = max(1, int(batch_size))
    selected: list[int] = []
    selected_set: set[int] = set()

    trace_parts: list[pd.DataFrame] = []
    if initial_label_table.empty:
        initial_step = min(batch_size, budget, len(candidate_table))
        initial = _cold_start_selection(candidate_table, initial_step, rng, cold_start_seed)
        for idx in initial.tolist():
            selected.append(int(idx))
            selected_set.add(int(idx))
        trace_parts.append(_trace_frame(initial, "max_error_exploitation", "random_cold_start"))
        if len(selected) >= budget:
            selected_arr = np.asarray(selected[:budget], dtype=int)
            trace = pd.concat(trace_parts, ignore_index=True)
            trace["selection_order"] = np.arange(1, len(trace) + 1, dtype=int)
            return (selected_arr, trace) if return_trace else selected_arr

    while len(selected) < budget:
        remaining = candidate_table[~candidate_table["index"].isin(selected_set)]
        if remaining.empty:
            break
        train = pd.concat([initial_label_table, candidate_table[candidate_table["index"].isin(selected_set)]], ignore_index=True)
        step = min(batch_size, budget - len(selected), len(remaining))
        ranked = _rank_error_risk_candidates(train, remaining, error_risk_fn)
        chosen = ranked.head(step).index.to_numpy(dtype=int)
        trace_parts.append(_trace_frame(chosen, "max_error_exploitation", "maximum_error_acquisition", ranked))
        for idx in chosen.tolist():
            if idx not in selected_set:
                selected.append(int(idx))
                selected_set.add(int(idx))

    selected_arr = np.asarray(selected[:budget], dtype=int)
    trace = pd.concat(trace_parts, ignore_index=True) if trace_parts else _trace_frame(selected_arr, "max_error_exploitation", "")
    trace = trace[trace["selected_index"].isin(selected_arr)].copy()
    trace["selection_order"] = np.arange(1, len(trace) + 1, dtype=int)
    return (selected_arr, trace) if return_trace else selected_arr


def _select_hybrid_iteratively(
    initial_label_table: pd.DataFrame,
    candidate_table: pd.DataFrame,
    matrix_fn: MatrixFn,
    uncertainty_fn: UncertaintyFn,
    budget: int,
    rng: np.random.Generator,
    batch_size: int,
    return_trace: bool = False,
    cold_start_seed: int | None = None,
) -> np.ndarray | tuple[np.ndarray, pd.DataFrame]:
    budget = min(int(budget), len(candidate_table))
    if budget <= 0:
        empty = np.array([], dtype=int)
        return (empty, _trace_frame(empty, "hybrid", "")) if return_trace else empty
    batch_size = max(1, int(batch_size))
    selected: list[int] = []
    selected_set: set[int] = set()

    trace_parts: list[pd.DataFrame] = []
    if initial_label_table.empty:
        initial_step = min(batch_size, budget, len(candidate_table))
        initial = _cold_start_selection(candidate_table, initial_step, rng, cold_start_seed)
        for idx in initial.tolist():
            selected.append(int(idx))
            selected_set.add(int(idx))
        trace_parts.append(_trace_frame(initial, "hybrid", "random_cold_start"))
        if len(selected) >= budget:
            selected_arr = np.asarray(selected[:budget], dtype=int)
            trace = pd.concat(trace_parts, ignore_index=True)
            trace["selection_order"] = np.arange(1, len(trace) + 1, dtype=int)
            return (selected_arr, trace) if return_trace else selected_arr

    while len(selected) < budget:
        batch_selected: list[int] = []
        model_train = pd.concat([initial_label_table, candidate_table[candidate_table["index"].isin(selected_set)]], ignore_index=True)
        batch_remaining = candidate_table[~candidate_table["index"].isin(selected_set)]
        if batch_remaining.empty:
            break
        step = min(batch_size, budget - len(selected), len(batch_remaining))
        uncertainty_scores = _uncertainty_scores(uncertainty_fn, model_train, batch_remaining)
        for _ in range(step):
            anchors = selected_set.union(batch_selected)
            remaining = candidate_table[~candidate_table["index"].isin(anchors)]
            if remaining.empty:
                break
            coverage_anchor = pd.concat([model_train, candidate_table[candidate_table["index"].isin(batch_selected)]], ignore_index=True)
            ranked = _rank_hybrid_candidates_from_uncertainty(uncertainty_scores, coverage_anchor, remaining, matrix_fn)
            chosen = ranked.head(1).index.to_numpy(dtype=int)
            trace_parts.append(_trace_frame(chosen, "hybrid", "hybrid_acquisition", ranked))
            idx = int(chosen[0])
            selected.append(idx)
            batch_selected.append(idx)
        selected_set.update(batch_selected)

    selected_arr = np.asarray(selected[:budget], dtype=int)
    trace = pd.concat(trace_parts, ignore_index=True) if trace_parts else _trace_frame(selected_arr, "hybrid", "")
    trace = trace[trace["selected_index"].isin(selected_arr)].copy()
    trace["selection_order"] = np.arange(1, len(trace) + 1, dtype=int)
    return (selected_arr, trace) if return_trace else selected_arr


def _select_coverage_iteratively(
    initial_label_table: pd.DataFrame,
    candidate_table: pd.DataFrame,
    matrix_fn: MatrixFn,
    budget: int,
    rng: np.random.Generator,
    batch_size: int,
    return_trace: bool = False,
    cold_start_seed: int | None = None,
) -> np.ndarray | tuple[np.ndarray, pd.DataFrame]:
    budget = min(int(budget), len(candidate_table))
    if budget <= 0:
        empty = np.array([], dtype=int)
        return (empty, _trace_frame(empty, "coverage", "")) if return_trace else empty
    batch_size = max(1, int(batch_size))
    selected: list[int] = []
    selected_set: set[int] = set()
    trace_parts: list[pd.DataFrame] = []

    if initial_label_table.empty:
        initial_step = min(batch_size, budget, len(candidate_table))
        initial = _cold_start_selection(candidate_table, initial_step, rng, cold_start_seed)
        for idx in initial.tolist():
            selected.append(int(idx))
            selected_set.add(int(idx))
        trace_parts.append(_trace_frame(initial, "coverage", "random_cold_start"))
        if len(selected) >= budget:
            selected_arr = np.asarray(selected[:budget], dtype=int)
            trace = pd.concat(trace_parts, ignore_index=True)
            trace["selection_order"] = np.arange(1, len(trace) + 1, dtype=int)
            return (selected_arr, trace) if return_trace else selected_arr

    while len(selected) < budget:
        remaining = candidate_table[~candidate_table["index"].isin(selected_set)]
        if remaining.empty:
            break
        train = pd.concat([initial_label_table, candidate_table[candidate_table["index"].isin(selected_set)]], ignore_index=True)
        step = min(batch_size, budget - len(selected), len(remaining))
        ranked = _rank_coverage_candidates(train, remaining, matrix_fn)
        chosen = ranked.head(step).index.to_numpy(dtype=int)
        trace_parts.append(_trace_frame(chosen, "coverage", "coreset_acquisition", ranked))
        for idx in chosen.tolist():
            if idx not in selected_set:
                selected.append(int(idx))
                selected_set.add(int(idx))

    selected_arr = np.asarray(selected[:budget], dtype=int)
    trace = pd.concat(trace_parts, ignore_index=True) if trace_parts else _trace_frame(selected_arr, "coverage", "")
    trace = trace[trace["selected_index"].isin(selected_arr)].copy()
    trace["selection_order"] = np.arange(1, len(trace) + 1, dtype=int)
    return (selected_arr, trace) if return_trace else selected_arr


def select_indices(
    method: str,
    budget: int,
    initial_label_table: pd.DataFrame,
    candidate_table: pd.DataFrame,
    matrix_fn: MatrixFn,
    uncertainty_fn: UncertaintyFn,
    rng: np.random.Generator,
    iterative_batch_size: int = 4,
    return_trace: bool = False,
    error_risk_fn: ErrorRiskFn | None = None,
    cold_start_seed: int | None = None,
) -> np.ndarray | tuple[np.ndarray, pd.DataFrame]:
    if method not in ACTIVE_LEARNING_METHODS:
        raise ValueError(f"Unknown active learning method: {method}")
    if budget <= 0 or candidate_table.empty:
        empty = np.array([], dtype=int)
        return (empty, _trace_frame(empty, method, "")) if return_trace else empty
    budget = min(int(budget), len(candidate_table))
    if method == "random_selection":
        # Random acquisition is the benchmark: uniformly sample cells without
        # using feature geometry or model predictions. It is still executed in
        # successive acquisition batches so the trajectory aligns with other
        # batch-mode rules.
        return _select_random_in_batches(
            candidate_table,
            budget,
            rng,
            iterative_batch_size,
            return_trace=return_trace,
            cold_start_seed=cold_start_seed,
        )
    if method == "coverage":
        # Coreset/Coverage is iterative: first a random cold-start batch, then
        # farthest-first batches anchored on the currently labeled cells.
        return _select_coverage_iteratively(
            initial_label_table,
            candidate_table,
            matrix_fn,
            budget,
            rng,
            iterative_batch_size,
            return_trace=return_trace,
            cold_start_seed=cold_start_seed,
        )
    if method == "diversity_oneshot":
        # One-shot diversity ranks the full budget once by k-means
        # representatives, while downstream accounting still groups the
        # resulting trajectory into acquisition batches.
        return _select_diversity_oneshot(candidate_table, matrix_fn, budget, rng, return_trace=return_trace)
    if method == "diversity_iterative":
        # Iterative diversity re-runs k-means on the remaining unlabelled pool
        # at each batch and queries representatives for that batch only.
        return _select_diversity_iteratively(
            candidate_table,
            matrix_fn,
            budget,
            rng,
            iterative_batch_size,
            return_trace=return_trace,
        )
    if method == "uncertainty":
        return _select_iteratively(
            initial_label_table,
            candidate_table,
            matrix_fn,
            uncertainty_fn,
            budget,
            rng,
            iterative_batch_size,
            mode="std",
            return_trace=return_trace,
            cold_start_seed=cold_start_seed,
        )
    if method == "qbc":
        return _select_iteratively(
            initial_label_table,
            candidate_table,
            matrix_fn,
            uncertainty_fn,
            budget,
            rng,
            iterative_batch_size,
            mode="std",
            return_trace=return_trace,
            cold_start_seed=cold_start_seed,
        )
    if method == "max_error_exploitation":
        if error_risk_fn is None:
            raise ValueError("max_error_exploitation acquisition requires an error_risk_fn.")
        return _select_error_risk_iteratively(
            initial_label_table,
            candidate_table,
            error_risk_fn,
            budget,
            rng,
            iterative_batch_size,
            return_trace=return_trace,
            cold_start_seed=cold_start_seed,
        )
    if method == "hybrid":
        return _select_hybrid_iteratively(
            initial_label_table,
            candidate_table,
            matrix_fn,
            uncertainty_fn,
            budget,
            rng,
            iterative_batch_size,
            return_trace=return_trace,
            cold_start_seed=cold_start_seed,
        )
    raise ValueError(f"Unknown active learning method: {method}")

