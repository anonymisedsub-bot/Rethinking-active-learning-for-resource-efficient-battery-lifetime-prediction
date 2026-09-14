from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from config import DatasetSpec, ExperimentConfig, FeatureSetSpec
from utils import matlab_round


@dataclass
class RawFeatureDataset:
    name: str
    features: np.ndarray
    lifetimes: np.ndarray
    cell_names: list[str]
    source_indices: np.ndarray
    protocols: np.ndarray | None = None
    feature_set: str = "set3_features"
    feature_set_label: str = "Set 3: early features"
    feature_matrix_key: str = "features_matrix"
    original_feature_count: int = 0
    retained_feature_indices: np.ndarray | None = None
    correlation_threshold: float | None = None


@dataclass
class PreparedFeatureData:
    dataset: RawFeatureDataset
    table: pd.DataFrame
    pool_indices: np.ndarray
    test_indices: np.ndarray
    imputer: SimpleImputer
    scaler: StandardScaler

    @property
    def pool_table(self) -> pd.DataFrame:
        return self.table[self.table["index"].isin(self.pool_indices)].copy()

    @property
    def test_table(self) -> pd.DataFrame:
        return self.table[self.table["index"].isin(self.test_indices)].copy()


def canonical_split_mode(value: str) -> str:
    name = str(value).strip().lower().replace("-", "_")
    if name in {"protocol_holdout", "protocol_heldout", "ph"}:
        return "protocol_holdout"
    if name in {"random", "r"}:
        return "random"
    raise ValueError(f"Unknown pool_holdout_split_mode: {value}")


def split_mode_label(value: str) -> str:
    mode = canonical_split_mode(value)
    return "PH" if mode == "protocol_holdout" else "R"


def _as_names(value: Any, n: int) -> list[str]:
    if value is None:
        return [f"Cell_{i + 1}" for i in range(n)]
    arr = np.atleast_1d(value)
    names = [str(v) for v in arr.tolist()]
    if len(names) < n:
        names.extend(f"Cell_{i + 1}" for i in range(len(names), n))
    return names[:n]


def _median_imputed_for_correlation(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"Correlation filtering expects a 2D feature matrix, got shape {x.shape}.")
    out = x.copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        finite = np.isfinite(col)
        fill_value = float(np.nanmedian(col[finite])) if finite.any() else 0.0
        out[~finite, j] = fill_value
    return out


def _absolute_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    value = float(np.corrcoef(a, b)[0, 1])
    return abs(value) if np.isfinite(value) else 0.0


def correlation_reduced_features(features: np.ndarray, threshold: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    x = _median_imputed_for_correlation(features)
    if x.shape[1] == 0:
        return x, np.array([], dtype=int)
    retained: list[int] = []
    for j in range(x.shape[1]):
        candidate = x[:, j]
        if all(_absolute_correlation(candidate, x[:, kept]) <= float(threshold) for kept in retained):
            retained.append(j)
    retained_arr = np.asarray(retained, dtype=int)
    return np.asarray(features, dtype=float)[:, retained_arr], retained_arr


def load_feature_dataset(
    cfg: ExperimentConfig,
    spec: DatasetSpec,
    feature_set: str | FeatureSetSpec | None = None,
) -> RawFeatureDataset:
    mat_path = cfg.data_dir / spec.filename
    data = loadmat(mat_path, simplify_cells=True)
    feature_spec = cfg.resolve_feature_set(feature_set)
    if feature_spec.matrix_key not in data:
        raise KeyError(f"{spec.filename} does not contain {feature_spec.matrix_key}.")
    if "lifetimes" not in data:
        raise KeyError(f"{spec.filename} does not contain lifetimes.")

    features = np.asarray(data[feature_spec.matrix_key], dtype=float)
    lifetimes = np.asarray(data["lifetimes"], dtype=float).reshape(-1)
    if features.ndim != 2:
        raise ValueError(f"{spec.filename}: {feature_spec.matrix_key} must be 2D, got shape {features.shape}.")
    if features.shape[0] != lifetimes.shape[0]:
        raise ValueError(f"{spec.filename}: feature/lifetime row mismatch {features.shape[0]} != {lifetimes.shape[0]}.")

    finite_life = np.isfinite(lifetimes)
    keep = finite_life & (lifetimes >= spec.min_lifetime)
    if spec.max_lifetime is not None:
        keep &= lifetimes <= spec.max_lifetime
    keep_indices = np.where(keep)[0].astype(int)
    if len(keep_indices) == 0:
        raise ValueError(f"{spec.name}: no samples remain after lifetime filtering.")

    cell_names = _as_names(data.get("cell_names"), len(lifetimes))
    protocols = _protocol_matrix(data.get("protocols"), len(lifetimes), spec.name)
    filtered_features = features[keep_indices]
    original_feature_count = int(filtered_features.shape[1])
    retained_feature_indices = np.arange(original_feature_count, dtype=int)
    return RawFeatureDataset(
        name=spec.name,
        features=filtered_features,
        lifetimes=lifetimes[keep_indices],
        cell_names=[cell_names[i] for i in keep_indices],
        source_indices=keep_indices,
        protocols=protocols[keep_indices] if protocols is not None else None,
        feature_set=feature_spec.name,
        feature_set_label=feature_spec.label,
        feature_matrix_key=feature_spec.matrix_key,
        original_feature_count=original_feature_count,
        retained_feature_indices=retained_feature_indices,
        correlation_threshold=feature_spec.correlation_threshold,
    )


def _protocol_matrix(value: Any, n_samples: int, dataset_name: str) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 2 and arr.shape[0] == n_samples:
        return arr
    if arr.ndim == 2 and arr.shape[1] == n_samples:
        return arr.T
    if arr.ndim == 1 and arr.size % n_samples == 0:
        return arr.reshape(n_samples, arr.size // n_samples)
    raise ValueError(f"{dataset_name}: protocols shape {arr.shape} cannot be aligned with {n_samples} samples.")


def random_pool_holdout_split(
    lifetimes: np.ndarray,
    pool_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(lifetimes), dtype=int)
    if len(indices) == 0:
        return indices, indices
    shuffled = indices.copy()
    rng.shuffle(shuffled)
    n_pool = matlab_round(len(shuffled) * pool_ratio)
    if len(shuffled) == 1 and pool_ratio > 0:
        n_pool = 1
    n_pool = min(max(n_pool, 0), len(shuffled))
    return shuffled[:n_pool].astype(int), shuffled[n_pool:].astype(int)


def protocol_holdout_pool_split(
    protocols: np.ndarray,
    pool_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    protocol_matrix = np.asarray(protocols, dtype=float)
    if protocol_matrix.ndim != 2:
        raise ValueError(f"Protocol-heldout split expects a 2D protocols matrix, got shape {protocol_matrix.shape}.")
    n_samples = int(protocol_matrix.shape[0])
    indices = np.arange(n_samples, dtype=int)
    if n_samples == 0:
        return indices, indices
    target_pool_size = matlab_round(n_samples * pool_ratio)
    if n_samples == 1 and pool_ratio > 0:
        target_pool_size = 1
    target_pool_size = min(max(target_pool_size, 0), n_samples)
    if target_pool_size == 0:
        return np.array([], dtype=int), indices
    if target_pool_size == n_samples:
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        return shuffled, np.array([], dtype=int)

    _, inverse = np.unique(np.round(protocol_matrix, 10), axis=0, return_inverse=True)
    group_ids = np.unique(inverse)
    rng.shuffle(group_ids)

    selected_group_ids: list[int] = []
    candidate_pool: list[int] = []
    for group_id in group_ids.tolist():
        selected_group_ids.append(int(group_id))
        candidate_pool.extend(indices[inverse == group_id].tolist())
        if len(candidate_pool) >= target_pool_size:
            break

    candidate_pool_arr = np.asarray(candidate_pool, dtype=int)
    if len(candidate_pool_arr) > target_pool_size:
        keep_rel = rng.choice(len(candidate_pool_arr), size=target_pool_size, replace=False)
        pool = candidate_pool_arr[keep_rel]
    else:
        pool = candidate_pool_arr
    rng.shuffle(pool)

    selected_group_set = set(selected_group_ids)
    holdout = indices[[int(group_id) not in selected_group_set for group_id in inverse]]
    rng.shuffle(holdout)
    return pool.astype(int), holdout.astype(int)


def feature_columns(n_features: int) -> list[str]:
    return [f"f{i + 1}" for i in range(n_features)]


def make_feature_table(dataset: RawFeatureDataset, features_scaled: np.ndarray) -> pd.DataFrame:
    data = {
        "index": np.arange(features_scaled.shape[0], dtype=int),
        "source_index": dataset.source_indices.astype(int),
        "cell_name": dataset.cell_names,
        "lifetime": dataset.lifetimes.astype(float),
        "feature_set": dataset.feature_set,
        "feature_set_label": dataset.feature_set_label,
        "feature_matrix_key": dataset.feature_matrix_key,
    }
    if dataset.protocols is not None:
        for j in range(dataset.protocols.shape[1]):
            data[f"protocol_{j + 1}"] = dataset.protocols[:, j].astype(float)
    for j, col in enumerate(feature_columns(features_scaled.shape[1])):
        data[col] = features_scaled[:, j]
    return pd.DataFrame(data)


def prepare_feature_data(
    cfg: ExperimentConfig,
    spec: DatasetSpec,
    trial: int,
    feature_set: str | FeatureSetSpec | None = None,
) -> PreparedFeatureData:
    dataset = load_feature_dataset(cfg, spec, feature_set=feature_set)
    rng = np.random.default_rng(cfg.random_seed + trial)
    split_mode = canonical_split_mode(getattr(cfg, "pool_holdout_split_mode", "random"))
    if split_mode == "random":
        pool_idx, test_idx = random_pool_holdout_split(dataset.lifetimes, cfg.pool_ratio, rng)
    elif split_mode == "protocol_holdout":
        if dataset.protocols is None:
            raise ValueError(f"{spec.name}: protocol_holdout split requires a protocols matrix.")
        pool_idx, test_idx = protocol_holdout_pool_split(dataset.protocols, cfg.pool_ratio, rng)
    else:
        raise ValueError(f"Unknown pool_holdout_split_mode: {cfg.pool_holdout_split_mode}")
    if dataset.correlation_threshold is not None:
        _, retained = correlation_reduced_features(
            dataset.features[pool_idx],
            threshold=float(dataset.correlation_threshold),
        )
        dataset.features = dataset.features[:, retained]
        dataset.retained_feature_indices = retained
    imputer = SimpleImputer(strategy="median").fit(dataset.features[pool_idx])
    features_imputed = imputer.transform(dataset.features)
    scaler = StandardScaler().fit(features_imputed[pool_idx])
    features_scaled = scaler.transform(features_imputed)
    table = make_feature_table(dataset, features_scaled)
    return PreparedFeatureData(dataset, table, pool_idx, test_idx, imputer, scaler)


def feature_matrix(table: pd.DataFrame) -> np.ndarray:
    cols = [c for c in table.columns if c.startswith("f") and c[1:].isdigit()]
    return table[cols].to_numpy(float)
