from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils import inverse_log_lifetime, log_lifetime


def canonical_target_transform(value: str) -> str:
    name = str(value).strip().lower().replace("-", "_")
    if name in {"log", "log_lifetime", "log_life"}:
        return "log"
    if name in {"raw", "lifetime", "raw_lifetime"}:
        return "raw"
    raise ValueError(f"Unknown target_transform: {value}")


@dataclass(frozen=True)
class TargetTransformer:
    mode: str
    mean: float
    std: float
    log_clip_lower: float = -3.0
    log_clip_upper: float = 3.0
    raw_clip_lower: float = 0.0
    raw_clip_upper_sigma: float = 5.0

    @classmethod
    def fit(
        cls,
        lifetimes: np.ndarray,
        mode: str = "log",
        log_clip_bounds: tuple[float, float] = (-3.0, 3.0),
        raw_clip_lower: float = 0.0,
        raw_clip_upper_sigma: float = 5.0,
    ) -> "TargetTransformer":
        canonical = canonical_target_transform(mode)
        life = np.asarray(lifetimes, dtype=float)
        if life.size == 0:
            raise ValueError("Cannot fit target transformer with an empty lifetime array.")
        target = log_lifetime(life) if canonical == "log" else life
        mean = float(np.mean(target))
        std = float(np.std(target))
        if std <= 1e-8 or not np.isfinite(std):
            std = 1.0
        return cls(
            mode=canonical,
            mean=mean,
            std=std,
            log_clip_lower=float(log_clip_bounds[0]),
            log_clip_upper=float(log_clip_bounds[1]),
            raw_clip_lower=float(raw_clip_lower),
            raw_clip_upper_sigma=float(raw_clip_upper_sigma),
        )

    @property
    def raw_clip_upper(self) -> float:
        return float(self.mean + self.raw_clip_upper_sigma * self.std)

    def transform(self, lifetimes: np.ndarray) -> np.ndarray:
        life = np.asarray(lifetimes, dtype=float)
        target = log_lifetime(life) if self.mode == "log" else life
        return (target - self.mean) / self.std

    def inverse(self, scaled_prediction: np.ndarray, clip: bool = True) -> np.ndarray:
        scaled = np.asarray(scaled_prediction, dtype=float)
        if self.mode == "log":
            if clip:
                scaled = np.clip(scaled, self.log_clip_lower, self.log_clip_upper)
            return inverse_log_lifetime(scaled * self.std + self.mean)
        target = scaled * self.std + self.mean
        if clip:
            target = np.clip(target, self.raw_clip_lower, self.raw_clip_upper)
        return target

    def to_jsonable(self) -> dict[str, float | str]:
        return {
            "mode": self.mode,
            "mean": float(self.mean),
            "std": float(self.std),
            "log_clip_lower": float(self.log_clip_lower),
            "log_clip_upper": float(self.log_clip_upper),
            "raw_clip_lower": float(self.raw_clip_lower),
            "raw_clip_upper_sigma": float(self.raw_clip_upper_sigma),
            "raw_clip_upper": float(self.raw_clip_upper),
        }
