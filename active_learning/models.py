from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.model_selection import KFold

from config import ExperimentConfig
from data import feature_matrix
from targeting import TargetTransformer


def _canonical_model_name(name: str) -> str:
    upper = str(name).upper().replace("-", "_")
    if upper in {"AE_ENET", "AEENET", "AENET"}:
        return "AE_ENet"
    return upper


class AEENetRegressor:
    def __init__(self, cfg: ExperimentConfig, seed: int, n_features: int) -> None:
        self.cfg = cfg
        self.seed = int(seed)
        self.n_features = int(n_features)
        self.models: list[AnchoredElasticNetMember] = []
        self.anchors: list[np.ndarray] = []

    def _member_seed(self, member_idx: int) -> int:
        return self.seed + 65537 * (int(member_idx) + 1)

    def fit(self, x: np.ndarray, y_scaled: np.ndarray) -> "AEENetRegressor":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y_scaled, dtype=float)
        if x.ndim != 2:
            raise ValueError(f"AE-ENet expects a 2D feature matrix, got shape {x.shape}.")
        if len(x) != len(y):
            raise ValueError(f"AE-ENet feature/target length mismatch: {len(x)} != {len(y)}.")
        n_features = x.shape[1]
        base = _fit_anchored_elastic_net(
            x,
            y,
            alpha=float(self.cfg.ae_enet_alpha),
            l1_ratio=float(self.cfg.ae_enet_l1_ratio),
            anchor=np.zeros(n_features, dtype=float),
            anchor_lambda=0.0,
            max_iter=10000,
            tol=1e-7,
        )
        base_coef = np.asarray(base.coef_, dtype=float)
        anchor_lambda = max(0.0, float(self.cfg.ae_enet_anchor_lambda))
        anchor_scale = float(self.cfg.ae_enet_anchor_noise_scale) / max(1.0, np.sqrt(float(n_features)))
        self.models = []
        self.anchors = []
        for member_idx in range(max(1, int(self.cfg.ae_enet_members))):
            rng = np.random.default_rng(self._member_seed(member_idx))
            anchor = base_coef + rng.normal(0.0, anchor_scale, size=n_features)
            model = _fit_anchored_elastic_net(
                x,
                y,
                alpha=float(self.cfg.ae_enet_alpha),
                l1_ratio=float(self.cfg.ae_enet_l1_ratio),
                anchor=anchor,
                anchor_lambda=anchor_lambda,
                max_iter=10000,
                tol=1e-7,
                initial_coef=base_coef,
            )
            self.models.append(model)
            self.anchors.append(anchor.astype(float))
        return self

    def predict_scaled_ensemble(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
        if not self.models:
            return np.full(len(x), np.nan, dtype=float), np.full(len(x), np.nan, dtype=float)
        preds = np.stack([model.predict(x) for model in self.models], axis=0)
        return preds.mean(axis=0), preds.std(axis=0)


@dataclass
class AnchoredElasticNetMember:
    coef_: np.ndarray
    intercept_: float
    alpha: float
    l1_ratio: float
    anchor: np.ndarray
    anchor_lambda: float
    n_iter_: int

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float) @ self.coef_ + float(self.intercept_)


def _soft_threshold(value: float, threshold: float) -> float:
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0.0


def _fit_anchored_elastic_net(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float,
    l1_ratio: float,
    anchor: np.ndarray,
    anchor_lambda: float,
    max_iter: int,
    tol: float,
    initial_coef: np.ndarray | None = None,
) -> AnchoredElasticNetMember:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    anchor = np.asarray(anchor, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"Anchored ElasticNet expects a 2D feature matrix, got shape {x.shape}.")
    if y.ndim != 1:
        y = y.reshape(-1)
    if len(x) != len(y):
        raise ValueError(f"Anchored ElasticNet feature/target length mismatch: {len(x)} != {len(y)}.")
    if anchor.shape != (x.shape[1],):
        raise ValueError(f"Anchor shape {anchor.shape} does not match n_features={x.shape[1]}.")
    n_samples, n_features = x.shape
    if n_samples == 0:
        raise ValueError("Anchored ElasticNet requires at least one training sample.")

    alpha = max(0.0, float(alpha))
    l1_ratio = float(np.clip(l1_ratio, 0.0, 1.0))
    anchor_lambda = max(0.0, float(anchor_lambda))
    max_iter = max(1, int(max_iter))
    tol = max(0.0, float(tol))

    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    x_centered = x - x_mean
    y_centered = y - y_mean
    if initial_coef is None:
        coef = np.zeros(n_features, dtype=float)
    else:
        coef = np.asarray(initial_coef, dtype=float).copy()
        if coef.shape != (n_features,):
            coef = np.zeros(n_features, dtype=float)
    feature_scale = np.sum(x_centered * x_centered, axis=0) / float(n_samples)
    l1_penalty = alpha * l1_ratio
    l2_penalty = alpha * (1.0 - l1_ratio)

    for iteration in range(1, max_iter + 1):
        old_coef = coef.copy()
        prediction = x_centered @ coef
        for j in range(n_features):
            residual_plus_j = y_centered - prediction + x_centered[:, j] * coef[j]
            rho = float(x_centered[:, j] @ residual_plus_j) / float(n_samples) + anchor_lambda * float(anchor[j])
            denom = float(feature_scale[j] + l2_penalty + anchor_lambda)
            coef[j] = _soft_threshold(rho, l1_penalty) / denom if denom > 0 else 0.0
            prediction = y_centered - residual_plus_j + x_centered[:, j] * coef[j]
        if np.max(np.abs(coef - old_coef)) <= tol * max(1.0, float(np.max(np.abs(old_coef)))):
            break

    intercept = y_mean - float(x_mean @ coef)
    return AnchoredElasticNetMember(
        coef_=coef.astype(float),
        intercept_=float(intercept),
        alpha=alpha,
        l1_ratio=l1_ratio,
        anchor=anchor.astype(float),
        anchor_lambda=anchor_lambda,
        n_iter_=int(iteration),
    )


@dataclass
class FittedRegressor:
    name: str
    model: object
    probabilistic: bool
    is_ensemble: bool = False
    target_mean: float = 0.0
    target_std: float = 1.0
    target_transformer: TargetTransformer | None = None

    def _inverse_target(self, target_scaled: np.ndarray) -> np.ndarray:
        if self.target_transformer is not None:
            return self.target_transformer.inverse(target_scaled, clip=True)
        return inverse_log_lifetime(target_scaled)

    def predict_target(self, table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = feature_matrix(table)
        model_name = _canonical_model_name(self.name)
        if model_name == "GPR":
            mean, std = self.model.predict(x, return_std=True)
            return np.asarray(mean, dtype=float), np.asarray(std, dtype=float)
        if model_name == "RF" and hasattr(self.model, "estimators_"):
            preds = np.stack([tree.predict(x) for tree in self.model.estimators_], axis=0)
            return preds.mean(axis=0), preds.std(axis=0)
        if model_name == "AE_ENet":
            return self.model.predict_scaled_ensemble(x)
        if self.is_ensemble and hasattr(self.model, "estimators_"):
            preds = np.stack([est.predict(x) for est in self.model.estimators_], axis=0)
            return preds.mean(axis=0), preds.std(axis=0)
        pred = np.asarray(self.model.predict(x), dtype=float)
        return pred, np.full(len(x), np.nan, dtype=float)

    def predict(self, table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = feature_matrix(table)
        model_name = _canonical_model_name(self.name)
        if model_name == "GPR":
            target_mean, target_std = self.model.predict(x, return_std=True)
            pred = self._inverse_target(target_mean)
            pred_hi = self._inverse_target(np.asarray(target_mean, dtype=float) + np.asarray(target_std, dtype=float))
            return pred, np.abs(pred_hi - pred)
        if model_name == "RF":
            if hasattr(self.model, "estimators_"):
                scaled_preds = np.stack([tree.predict(x) for tree in self.model.estimators_], axis=0)
            else:
                scaled_preds = np.asarray(self.model.predict(x), dtype=float)[None, :]
            lifetime_preds = self._inverse_target(scaled_preds)
            pred = lifetime_preds.mean(axis=0)
            std = lifetime_preds.std(axis=0)
            return pred, std
        if model_name == "AE_ENet":
            scaled_mean, _ = self.model.predict_scaled_ensemble(x)
            scaled_preds = np.stack([model.predict(x) for model in self.model.models], axis=0)
            lifetime_preds = self._inverse_target(scaled_preds)
            return self._inverse_target(scaled_mean), lifetime_preds.std(axis=0)
        if self.is_ensemble and hasattr(self.model, "estimators_"):
            scaled_preds = np.stack([est.predict(x) for est in self.model.estimators_], axis=0)
            lifetime_preds = self._inverse_target(scaled_preds)
            pred = lifetime_preds.mean(axis=0)
            std = lifetime_preds.std(axis=0)
            return pred, std
        target_pred = np.asarray(self.model.predict(x), dtype=float)
        return self._inverse_target(target_pred), np.full(len(x), np.nan, dtype=float)


def _base_regressor(name: str, seed: int):
    upper = _canonical_model_name(name)
    if upper == "GPR":
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(
            noise_level=1.0,
            noise_level_bounds=(1e-5, 1e3),
        )
        return GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed), True
    raise ValueError(f"Unknown regressor: {name}")


def _record_runtime_event(cfg: ExperimentConfig, payload: dict[str, object]) -> None:
    events = getattr(cfg, "_runtime_events", None)
    if isinstance(events, list):
        context = getattr(cfg, "_runtime_context", {})
        row = dict(context) if isinstance(context, dict) else {}
        row.update(payload)
        events.append(row)


def fit_regressor(name: str, train_table: pd.DataFrame, cfg: ExperimentConfig, seed: int) -> FittedRegressor:
    if train_table.empty:
        raise ValueError("Cannot fit a lifetime regressor with an empty training table.")
    x = feature_matrix(train_table)
    target_transformer = TargetTransformer.fit(
        train_table["lifetime"].to_numpy(float),
        mode=getattr(cfg, "target_transform", "log"),
        log_clip_bounds=getattr(cfg, "log_clip_bounds", (-3.0, 3.0)),
        raw_clip_lower=getattr(cfg, "raw_clip_lower", 0.0),
        raw_clip_upper_sigma=getattr(cfg, "raw_clip_upper_sigma", 5.0),
    )
    y = target_transformer.transform(train_table["lifetime"].to_numpy(float))
    start_time = time.perf_counter()
    model_name = _canonical_model_name(name)
    if model_name == "RF":
        model = RandomForestRegressor(
            n_estimators=int(cfg.rf_n_estimators),
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(x, y)
        _record_runtime_event(
            cfg,
            {
                "event": "fit_regressor",
                "regressor": name,
                "seed": int(seed),
                "n_train": int(len(train_table)),
                "n_features": int(x.shape[1]),
                "n_estimators": int(cfg.rf_n_estimators),
                "target_transform": target_transformer.mode,
                "target_mean": target_transformer.mean,
                "target_std": target_transformer.std,
                "elapsed_sec": float(time.perf_counter() - start_time),
            },
        )
        return FittedRegressor(
            name=name,
            model=model,
            probabilistic=True,
            is_ensemble=True,
            target_mean=target_transformer.mean,
            target_std=target_transformer.std,
            target_transformer=target_transformer,
        )
    if model_name == "AE_ENet":
        model = AEENetRegressor(cfg, seed, x.shape[1]).fit(x, y)
        _record_runtime_event(
            cfg,
            {
                "event": "fit_regressor",
                "regressor": name,
                "seed": int(seed),
                "n_train": int(len(train_table)),
                "n_features": int(x.shape[1]),
                "target_transform": target_transformer.mode,
                "target_mean": target_transformer.mean,
                "target_std": target_transformer.std,
                "ae_enet_members": int(cfg.ae_enet_members),
                "ae_enet_anchor_lambda": float(cfg.ae_enet_anchor_lambda),
                "elapsed_sec": float(time.perf_counter() - start_time),
            },
        )
        return FittedRegressor(
            name=name,
            model=model,
            probabilistic=True,
            is_ensemble=True,
            target_mean=target_transformer.mean,
            target_std=target_transformer.std,
            target_transformer=target_transformer,
        )
    model, probabilistic = _base_regressor(name, seed)
    if model_name == "GPR":
        model.alpha = cfg.gpr_alpha
        model.n_restarts_optimizer = cfg.gpr_n_restarts_optimizer
    is_ensemble = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(x, y)
    _record_runtime_event(
        cfg,
        {
            "event": "fit_regressor",
            "regressor": name,
            "seed": int(seed),
            "n_train": int(len(train_table)),
            "n_features": int(x.shape[1]),
            "gpr_n_restarts_optimizer": int(cfg.gpr_n_restarts_optimizer) if model_name == "GPR" else np.nan,
            "ensemble_members": int(cfg.ensemble_members) if is_ensemble else np.nan,
            "target_transform": target_transformer.mode,
            "target_mean": target_transformer.mean,
            "target_std": target_transformer.std,
            "elapsed_sec": float(time.perf_counter() - start_time),
        },
    )
    return FittedRegressor(
        name=name,
        model=model,
        probabilistic=probabilistic,
        is_ensemble=is_ensemble,
        target_mean=target_transformer.mean,
        target_std=target_transformer.std,
        target_transformer=target_transformer,
    )


def uncertainty_table(name: str, train_table: pd.DataFrame, candidate_table: pd.DataFrame, cfg: ExperimentConfig, seed: int) -> pd.DataFrame:
    if candidate_table.empty:
        return pd.DataFrame(columns=["index", "lifetime_mean", "target_std"])
    model = fit_regressor(name, train_table, cfg, seed)
    target_mean, target_std = model.predict_target(candidate_table)
    mean, _ = model.predict(candidate_table)
    return pd.DataFrame(
        {
            "index": candidate_table["index"].to_numpy(dtype=int),
            "lifetime_mean": mean.astype(float),
            "target_std": target_std.astype(float),
        }
    )


def _fit_error_surrogate(x: np.ndarray, residual: np.ndarray, cfg: ExperimentConfig, seed: int):
    start_time = time.perf_counter()
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(
        noise_level=1.0,
        noise_level_bounds=(1e-5, 1e3),
    )
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=float(cfg.gpr_alpha),
        normalize_y=True,
        n_restarts_optimizer=int(cfg.gpr_n_restarts_optimizer),
        random_state=int(seed),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(x, residual)
    _record_runtime_event(
        cfg,
        {
            "event": "fit_error_surrogate",
            "regressor": "GPR_error_surrogate",
            "seed": int(seed),
            "n_train": int(x.shape[0]),
            "n_features": int(x.shape[1]),
            "gpr_n_restarts_optimizer": int(cfg.gpr_n_restarts_optimizer),
            "elapsed_sec": float(time.perf_counter() - start_time),
        },
    )
    return model


def error_risk_table(name: str, train_table: pd.DataFrame, candidate_table: pd.DataFrame, cfg: ExperimentConfig, seed: int) -> pd.DataFrame:
    if candidate_table.empty:
        return pd.DataFrame(columns=["index", "lifetime_mean", "target_std", "acquisition_score"])
    if len(train_table) < 2:
        return pd.DataFrame(
            {
                "index": candidate_table["index"].to_numpy(dtype=int),
                "lifetime_mean": np.nan,
                "target_std": np.nan,
                "acquisition_score": np.zeros(len(candidate_table), dtype=float),
            }
        )

    train_table = train_table.reset_index(drop=True)
    y_true = train_table["lifetime"].to_numpy(float)
    residual = np.zeros(len(train_table), dtype=float)
    n_splits = min(4, len(train_table))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    for fold_id, (fit_idx, valid_idx) in enumerate(splitter.split(train_table)):
        fold_model = fit_regressor(name, train_table.iloc[fit_idx].reset_index(drop=True), cfg, int(seed) + 7919 * (fold_id + 1))
        pred, _ = fold_model.predict(train_table.iloc[valid_idx].reset_index(drop=True))
        residual[valid_idx] = np.abs(y_true[valid_idx] - pred)

    surrogate = _fit_error_surrogate(feature_matrix(train_table), residual, cfg, int(seed) + 104729)
    risk = np.asarray(surrogate.predict(feature_matrix(candidate_table)), dtype=float)
    risk = np.clip(risk, 0.0, None)
    return pd.DataFrame(
        {
            "index": candidate_table["index"].to_numpy(dtype=int),
            "lifetime_mean": np.nan,
            "target_std": np.nan,
            "acquisition_score": risk,
        }
    )



