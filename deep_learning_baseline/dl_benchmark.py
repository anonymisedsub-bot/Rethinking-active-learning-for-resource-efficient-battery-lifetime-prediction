from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - script-level dependency guard
    raise ImportError("main_dl_benchmark.py requires PyTorch.") from exc

try:
    from pytorch_tabnet.tab_model import TabNetRegressor as StandardTabNetRegressor
except ImportError:  # pragma: no cover - dependency installed by benchmark setup
    StandardTabNetRegressor = None

from config import DEFAULT_RUN_TRIALS, ExperimentConfig, FeatureSetSpec
from data import canonical_split_mode, feature_matrix, prepare_feature_data, split_mode_label
from metrics import lifetime_metrics, prediction_frame
from targeting import TargetTransformer, canonical_target_transform
from utils import ensure_dir, save_json, set_seed


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


def _portable_relative_path(path: Path, base: Path) -> str:
    return Path(path).relative_to(Path(base)).as_posix()


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


DL_MODELS = ("CNN", "TabNet", "DeepVAE", "MLP")
DL_EARLY_STOPPING_PATIENCE = 50

TABNET_CONFIG_LARGER = {
    "n_d": 4,
    "n_a": 4,
    "n_steps": 2,
    "gamma": 1.1,
    "n_shared": 1,
    "n_independent": 1,
    "lambda_sparse": 5e-3,
    "optimizer": "AdamW",
    "weight_decay": 1e-3,
    "virtual_batch_size": 4,
    "patience": DL_EARLY_STOPPING_PATIENCE,
}


def torch_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNNRegressor(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x.unsqueeze(1)).squeeze(-1)


class MLPRegressor(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DeepVAERegressor(nn.Module):
    def __init__(self, n_features: int, latent_dim: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
        )
        self.mu = nn.Linear(16, latent_dim)
        self.logvar = nn.Linear(16, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, n_features),
        )
        self.regression_head = nn.Sequential(nn.Linear(latent_dim, 16), nn.ReLU(), nn.Linear(16, 1))

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), torch.clamp(self.logvar(h), min=-8.0, max=6.0)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if self.training else mu
        recon = self.decoder(z)
        pred = self.regression_head(z).squeeze(-1)
        return pred, recon, mu, logvar


def _make_model(name: str, n_features: int) -> nn.Module:
    if name == "CNN":
        return CNNRegressor(n_features)
    if name == "MLP":
        return MLPRegressor(n_features)
    if name == "DeepVAE":
        return DeepVAERegressor(n_features)
    raise ValueError(f"Unknown deep-learning benchmark model: {name}")


def _loss(model_name: str, model_out, x, y) -> torch.Tensor:
    if model_name == "DeepVAE":
        pred, recon, mu, logvar = model_out
        reg = F.mse_loss(pred, y)
        recon_loss = F.mse_loss(recon, x)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return reg + 0.20 * recon_loss + 1e-3 * kl
    return F.mse_loss(model_out, y)


def _predict(model_name: str, model: nn.Module, x: np.ndarray, transformer: TargetTransformer, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_t = torch.as_tensor(x.astype(np.float32), device=device)
        out = model(x_t)
        pred_scaled = out[0] if model_name == "DeepVAE" else out
    return transformer.inverse(pred_scaled.cpu().numpy(), clip=True)


def _tabnet_history_to_logs(history) -> list[dict[str, float]]:
    history_dict = getattr(history, "history", history)
    if not isinstance(history_dict, dict):
        return []
    train_values = history_dict.get("loss", [])
    val_key = next((key for key in ("val_0_mse", "val_mse", "valid_mse") if key in history_dict), None)
    val_values = history_dict.get(val_key, []) if val_key else []
    n_epochs = max(len(train_values), len(val_values))
    rows = []
    for epoch in range(n_epochs):
        rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_values[epoch]) if epoch < len(train_values) else float("nan"),
                "val_loss": float(val_values[epoch]) if epoch < len(val_values) else float("nan"),
            }
        )
    return rows


def _tabnet_safe_batch_settings(
    n_samples: int,
    requested_batch_size: int,
    requested_virtual_batch_size: int,
) -> tuple[int, int]:
    n = int(n_samples)
    if n < 2:
        raise ValueError(f"TabNet requires at least 2 training samples, got {n}.")

    preferred_batch = max(2, int(requested_batch_size))
    upper = min(preferred_batch, n)
    batch_size = None
    for candidate in range(upper, 1, -1):
        if n % candidate != 1:
            batch_size = candidate
            break
    if batch_size is None:
        batch_size = n

    batch_sizes_seen = [batch_size]
    remainder = n % batch_size
    if remainder:
        batch_sizes_seen.append(remainder)

    preferred_virtual = max(2, int(requested_virtual_batch_size))
    virtual_upper = min(preferred_virtual, min(batch_sizes_seen))
    virtual_batch_size = None
    for candidate in range(virtual_upper, 1, -1):
        if all(size % candidate != 1 for size in batch_sizes_seen):
            virtual_batch_size = candidate
            break
    if virtual_batch_size is None:
        virtual_batch_size = min(batch_sizes_seen)

    return int(batch_size), int(virtual_batch_size)


def _train_tabnet_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    target_transformer: TargetTransformer,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
):
    if StandardTabNetRegressor is None:
        raise ImportError("TabNet benchmark requires pytorch-tabnet. Install it with: pip install pytorch-tabnet")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    safe_batch_size, safe_virtual_batch_size = _tabnet_safe_batch_settings(
        n_samples=len(train_x),
        requested_batch_size=batch_size,
        requested_virtual_batch_size=int(TABNET_CONFIG_LARGER["virtual_batch_size"]),
    )
    model = StandardTabNetRegressor(
        n_d=int(TABNET_CONFIG_LARGER["n_d"]),
        n_a=int(TABNET_CONFIG_LARGER["n_a"]),
        n_steps=int(TABNET_CONFIG_LARGER["n_steps"]),
        gamma=float(TABNET_CONFIG_LARGER["gamma"]),
        n_shared=int(TABNET_CONFIG_LARGER["n_shared"]),
        n_independent=int(TABNET_CONFIG_LARGER["n_independent"]),
        lambda_sparse=float(TABNET_CONFIG_LARGER["lambda_sparse"]),
        optimizer_fn=torch.optim.AdamW,
        optimizer_params={
            "lr": float(learning_rate),
            "weight_decay": float(TABNET_CONFIG_LARGER["weight_decay"]),
        },
        mask_type="sparsemax",
        seed=int(seed),
        verbose=0,
        device_name=device_name,
    )
    model.fit(
        train_x.astype(np.float32),
        train_y.reshape(-1, 1).astype(np.float32),
        eval_set=[(val_x.astype(np.float32), val_y.reshape(-1, 1).astype(np.float32))],
        eval_name=["val"],
        eval_metric=["mse"],
        max_epochs=int(epochs),
        patience=int(TABNET_CONFIG_LARGER["patience"]),
        batch_size=safe_batch_size,
        virtual_batch_size=safe_virtual_batch_size,
        num_workers=0,
        drop_last=False,
    )
    pred_scaled = model.predict(test_x.astype(np.float32)).reshape(-1)
    pred = target_transformer.inverse(pred_scaled, clip=True)
    return model, _tabnet_history_to_logs(model.history), pred


def _train_one_model(
    model_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    target_transformer: TargetTransformer,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[object, list[dict[str, float]], np.ndarray]:
    if model_name == "TabNet":
        return _train_tabnet_model(train_x, train_y, val_x, val_y, test_x, target_transformer, epochs, batch_size, learning_rate, seed)
    torch.manual_seed(seed)
    device = torch_device()
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = _make_model(model_name, train_x.shape[1]).to(device)
    lr = learning_rate
    weight_decay = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    train_x_t = torch.as_tensor(train_x.astype(np.float32), device=device)
    train_y_t = torch.as_tensor(train_y.astype(np.float32), device=device)
    val_x_t = torch.as_tensor(val_x.astype(np.float32), device=device)
    val_y_t = torch.as_tensor(val_y.astype(np.float32), device=device)
    batch_size = min(max(1, int(batch_size)), len(train_x_t))
    logs: list[dict[str, float]] = []
    best_state = None
    best_val = float("inf")
    patience = DL_EARLY_STOPPING_PATIENCE
    stale = 0
    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = rng.permutation(len(train_x_t))
        losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            xb = train_x_t[idx]
            yb = train_y_t[idx]
            loss = _loss(model_name, model(xb), xb, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_loss = float(_loss(model_name, model(val_x_t), val_x_t, val_y_t).detach().cpu())
        train_loss = float(np.mean(losses)) if losses else float("nan")
        logs.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    pred = _predict(model_name, model, test_x, target_transformer, device)
    return model, logs, pred


def _read_dl_dataset_frames(feature_root: Path, filename: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dataset_dir in sorted(path for path in Path(feature_root).iterdir() if path.is_dir()):
        path = dataset_dir / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def run_dl_benchmark(cfg: ExperimentConfig) -> Path:
    target_modes = getattr(cfg, "run_target_transforms", None)
    split_modes = getattr(cfg, "run_split_modes", None)
    if target_modes is not None or split_modes is not None:
        output_root = ensure_dir(cfg.dl_output_root)
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
                out_dir = run_dl_benchmark(subcfg)
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
    device = torch_device()
    print(f"[DL] Using device: {device}")
    output_root = ensure_dir(_scoped_output_root(cfg, cfg.dl_output_root))
    enabled = set(cfg.run_datasets) if cfg.run_datasets else None
    feature_run_summaries: list[dict[str, str]] = []
    selected_feature_sets = cfg.selected_feature_sets()
    last_feature_root: Path | None = None

    for feature_set in selected_feature_sets:
        feature_root = ensure_dir(output_root / _feature_set_directory_name(feature_set))
        last_feature_root = feature_root
        summary_rows = []
        prediction_frames = []
        training_rows = []

        for spec in cfg.datasets:
            if enabled and spec.name not in enabled:
                continue
            dataset_dir = ensure_dir(feature_root / spec.name)
            dataset_summary_rows = []
            dataset_prediction_frames = []
            dataset_training_rows = []

            for trial in _configured_trial_ids(cfg):
                prepared = prepare_feature_data(cfg, spec, trial, feature_set=feature_set)
                pool_table = prepared.pool_table.reset_index(drop=True)
                test_table = prepared.test_table
                train_table = pool_table.copy()
                val_table = pool_table.copy()
                train_x = feature_matrix(train_table)
                val_x = feature_matrix(val_table)
                test_x = feature_matrix(test_table)
                transformer = TargetTransformer.fit(
                    train_table["lifetime"].to_numpy(float),
                    mode=getattr(cfg, "target_transform", "log"),
                    log_clip_bounds=getattr(cfg, "log_clip_bounds", (-3.0, 3.0)),
                    raw_clip_lower=getattr(cfg, "raw_clip_lower", 0.0),
                    raw_clip_upper_sigma=getattr(cfg, "raw_clip_upper_sigma", 5.0),
                )
                train_y = transformer.transform(train_table["lifetime"].to_numpy(float))
                val_y = transformer.transform(val_table["lifetime"].to_numpy(float))
                for model_name in cfg.dl_models:
                    seed = cfg.random_seed + trial * 9109 + sum(ord(c) for c in model_name)
                    print(f"[DL] feature_set={feature_set.name} dataset={spec.name} trial={trial} model={model_name}")
                    model, logs, pred = _train_one_model(
                        model_name,
                        train_x,
                        train_y,
                        val_x,
                        val_y,
                        test_x,
                        transformer,
                        epochs=cfg.dl_epochs,
                        batch_size=cfg.dl_batch_size,
                        learning_rate=cfg.dl_learning_rate,
                        seed=seed,
                    )
                    metric = lifetime_metrics(test_table["lifetime"].to_numpy(float), pred)
                    metric_row = {
                        **metric,
                        "feature_set": feature_set.name,
                        "feature_set_label": feature_set.label,
                        "dataset": spec.name,
                        "trial": trial,
                        "model": model_name,
                        "pool_train_count": int(len(train_table)),
                        "pool_val_count": 0,
                        "holdout_count": int(len(test_table)),
                        "target_transform": canonical_target_transform(getattr(cfg, "target_transform", "log")),
                        "prediction_clip_enabled": bool(getattr(cfg, "prediction_clip_enabled", True)),
                        "pool_holdout_split_mode": canonical_split_mode(getattr(cfg, "pool_holdout_split_mode", "random")),
                        "pool_holdout_split_label": split_mode_label(getattr(cfg, "pool_holdout_split_mode", "random")),
                    }
                    summary_rows.append(metric_row)
                    dataset_summary_rows.append(metric_row)

                    pred_df = prediction_frame(test_table, pred, np.full(len(pred), np.nan))
                    pred_df["feature_set"] = feature_set.name
                    pred_df["feature_set_label"] = feature_set.label
                    pred_df["dataset"] = spec.name
                    pred_df["trial"] = trial
                    pred_df["model"] = model_name
                    pred_df["target_transform"] = canonical_target_transform(getattr(cfg, "target_transform", "log"))
                    pred_df["prediction_clip_enabled"] = bool(getattr(cfg, "prediction_clip_enabled", True))
                    pred_df["pool_holdout_split_mode"] = canonical_split_mode(getattr(cfg, "pool_holdout_split_mode", "random"))
                    pred_df["pool_holdout_split_label"] = split_mode_label(getattr(cfg, "pool_holdout_split_mode", "random"))
                    prediction_frames.append(pred_df)
                    dataset_prediction_frames.append(pred_df)

                    for row in logs:
                        log_row = {"feature_set": feature_set.name, "dataset": spec.name, "trial": trial, "model": model_name, **row}
                        training_rows.append(log_row)
                        dataset_training_rows.append(log_row)

                    model_checkpoint_dir = ensure_dir(dataset_dir / "checkpoints" / model_name)
                    if model_name == "TabNet":
                        ckpt_base = model_checkpoint_dir / f"trial{trial}"
                        model.save_model(str(ckpt_base))
                        save_json({"target_transformer": transformer.to_jsonable()}, model_checkpoint_dir / f"trial{trial}_target.json")
                    else:
                        ckpt_path = model_checkpoint_dir / f"trial{trial}.pt"
                        state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                        torch.save({"model": state_dict, "target_transformer": transformer.to_jsonable()}, ckpt_path)

            dataset_metrics_df = _merge_partial_trial_output(
                cfg,
                dataset_dir / "metrics.csv",
                pd.DataFrame(dataset_summary_rows),
                ["feature_set", "dataset", "trial", "model"],
            )
            dataset_predictions_df = _merge_partial_trial_output(
                cfg,
                dataset_dir / "predictions.csv",
                pd.concat(dataset_prediction_frames, ignore_index=True) if dataset_prediction_frames else pd.DataFrame(),
                ["feature_set", "dataset", "trial", "model", "index"],
            )
            dataset_training_df = _merge_partial_trial_output(
                cfg,
                dataset_dir / "training_log.csv",
                pd.DataFrame(dataset_training_rows),
                ["feature_set", "dataset", "trial", "model", "epoch"],
            )
            dataset_metrics_df.to_csv(dataset_dir / "metrics.csv", index=False)
            dataset_predictions_df.to_csv(dataset_dir / "predictions.csv", index=False)
            dataset_training_df.to_csv(dataset_dir / "training_log.csv", index=False)

        summary = _read_dl_dataset_frames(feature_root, "metrics.csv")
        predictions = _read_dl_dataset_frames(feature_root, "predictions.csv")
        training_log = _read_dl_dataset_frames(feature_root, "training_log.csv")
        summary = _write_merged_summary(
            feature_root / "metrics_summary.csv",
            summary,
            ["feature_set", "dataset", "trial", "model"],
        )
        predictions = _write_merged_summary(
            feature_root / "predictions_summary.csv",
            predictions,
            ["feature_set", "dataset", "trial", "model", "index"],
        )
        training_log = _write_merged_summary(
            feature_root / "training_log_summary.csv",
            training_log,
            ["feature_set", "dataset", "trial", "model", "epoch"],
        )
        save_json(
            {
                "models": list(cfg.dl_models),
                "tabnet_backend": "pytorch-tabnet",
                "tabnet_config_larger": TABNET_CONFIG_LARGER,
                "epochs": int(cfg.dl_epochs),
                "batch_size": int(cfg.dl_batch_size),
                "learning_rate": float(cfg.dl_learning_rate),
                "run_trials": _configured_trial_ids(cfg),
                "n_trials": int(len(_configured_trial_ids(cfg))),
                "random_seed": int(cfg.random_seed),
                "target_transform": canonical_target_transform(getattr(cfg, "target_transform", "log")),
                "prediction_clip_enabled": bool(getattr(cfg, "prediction_clip_enabled", True)),
                "pool_holdout_split_mode": canonical_split_mode(cfg.pool_holdout_split_mode),
                "pool_holdout_split_label": split_mode_label(cfg.pool_holdout_split_mode),
                "feature_set": feature_set.name,
                "feature_set_label": feature_set.label,
                "device": str(device),
                "cuda_available": bool(torch.cuda.is_available()),
                "metrics_summary": "metrics_summary.csv",
                "predictions_summary": "predictions_summary.csv",
                "training_log_summary": "training_log_summary.csv",
            },
            feature_root / "run_config.json",
        )
        save_json(
            {
                "feature_set": feature_set.name,
                "feature_set_label": feature_set.label,
                "metrics_summary": "metrics_summary.csv",
                "predictions_summary": "predictions_summary.csv",
                "training_log_summary": "training_log_summary.csv",
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
            "output_layout": "Each DL benchmark feature set is saved in a fixed top-level directory: feature set 1, ..., feature set 5.",
        },
        output_root / "run_summary.json",
    )
    return last_feature_root if len(selected_feature_sets) == 1 and last_feature_root is not None else output_root
