from __future__ import annotations

import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


NATURE_COLORS = ["#4E79A7", "#E15759", "#59A14F", "#F28E2B", "#76B7B2", "#B07AA1", "#EDC948", "#9C755F"]
LIFETIME_EPS = 1e-6


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def matlab_round(value: float) -> int:
    return int(math.floor(value + 0.5))


def log_lifetime(values: np.ndarray | float) -> np.ndarray:
    return np.log(np.maximum(np.asarray(values, dtype=float), LIFETIME_EPS))


def inverse_log_lifetime(values: np.ndarray | float) -> np.ndarray:
    return np.exp(np.asarray(values, dtype=float))


def inverse_log_lifetime_std(log_mean: np.ndarray, log_std: np.ndarray) -> np.ndarray:
    log_mean = np.asarray(log_mean, dtype=float)
    log_std = np.maximum(np.asarray(log_std, dtype=float), 0.0)
    variance = (np.exp(log_std**2) - 1.0) * np.exp(2.0 * log_mean + log_std**2)
    return np.sqrt(np.maximum(variance, 0.0))


def budget_tick_label(budget: int, max_budget: int) -> str:
    if budget == 0:
        return "0"
    if budget == max_budget and budget & (budget - 1):
        return "max"
    exponent = int(np.log2(budget)) if budget > 0 else 0
    if 2**exponent == budget:
        return rf"$2^{exponent}$"
    return str(budget)


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def set_plot_style() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "Arial",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
