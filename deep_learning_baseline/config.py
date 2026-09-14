from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_FEATURE_SET = "set3_features"
DEFAULT_RUN_TRIALS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    filename: str
    min_lifetime: float = 0.0
    max_lifetime: float | None = None


@dataclass(frozen=True)
class FeatureSetSpec:
    name: str
    label: str
    matrix_key: str
    description: str
    correlation_threshold: float | None = None


@dataclass
class ExperimentConfig:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    output_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "outputs_AL")
    dl_output_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "outputs_DL_benchmark")
    pipeline: str = "active_learning"  # "active_learning" or "dl_benchmark"

    run_datasets: list[str] | None = None
    run_feature_sets: list[str] | None = None
    run_trials: tuple[int, ...] | None = DEFAULT_RUN_TRIALS
    random_seed: int = 2026
    pool_ratio: float = 0.80
    pool_holdout_split_mode: str = "protocol_holdout"
    target_transform: str = "log"
    run_target_transforms: tuple[str, ...] | None = None
    run_split_modes: tuple[str, ...] | None = None
    prediction_clip_enabled: bool = True
    log_clip_bounds: tuple[float, float] = (-3.0, 3.0)
    raw_clip_lower: float = 0.0
    raw_clip_upper_sigma: float = 5.0

    regressors: tuple[str, ...] = ("GPR", "RF", "AE_ENet")
    active_learning_methods: tuple[str, ...] = (
        "random_selection",
        "diversity_oneshot",
        "diversity_iterative",
        "coverage",
        "exploration",
        "exploitation",
        "hybrid",
    )
    dataset_iterative_batch_sizes: dict[str, int] = field(
        default_factory=lambda: {
            "MIT": 8,
            "ISU_ILCC": 16,
            "LSD_Primary": 6,
            "LSD_Second": 6,
            "HUST": 6,
            "KIT": 6,
            "Formation": 12,
            "TRI_Tesla": 8,
        }
    )
    dataset_budget_grid_specs: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "MIT": (8, 8),
            "ISU_ILCC": (16, 16),
            "LSD_Primary": (6, 6),
            "LSD_Second": (6, 6),
            "HUST": (6, 6),
            "KIT": (6, 6),
            "Formation": (12, 12),
            "TRI_Tesla": (8, 8),
        }
    )

    # GPR and ensemble uncertainty.
    gpr_alpha: float = 1e-6
    gpr_n_restarts_optimizer: int = 3
    rf_n_estimators: int = 100
    ensemble_members: int = 5

    # Coefficient-space anchored ElasticNet ensemble used as the third primary
    # lifetime predictor and as the committee for RF-like exploration.
    ae_enet_members: int = 10
    ae_enet_alpha: float = 0.01
    ae_enet_l1_ratio: float = 0.5
    ae_enet_anchor_lambda: float = 0.50
    ae_enet_anchor_noise_scale: float = 0.15

    # Deep-learning all-pool benchmark settings. This pipeline reuses the same
    # trial seeds and pool/hold-out splits as the active-learning pipeline.
    dl_models: tuple[str, ...] = ("CNN", "TabNet", "DeepVAE", "MLP")
    dl_epochs: int = 200
    dl_batch_size: int = 8
    dl_learning_rate: float = 1e-3

    save_predictions: bool = True
    save_figures: bool = True

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def feature_sets(self) -> tuple[FeatureSetSpec, ...]:
        return (
            FeatureSetSpec(
                "set1_dQn_m",
                "Set 1: dQn_m",
                "dQn_m_matrix",
                "Capacity-sequence difference between the early reference cycle/week n and cycle/week m.",
            ),
            FeatureSetSpec(
                "set2_early_soh",
                "Set 2: early SOH degradation",
                "early_soh_degradation_matrix",
                "Early capacity-degradation trajectory.",
            ),
            FeatureSetSpec(
                "set3_features",
                "Set 3: early features",
                "features_matrix",
                "Handcrafted early-life features.",
            ),
            FeatureSetSpec(
                "set4_features_metadata",
                "Set 4: early features + metadata",
                "features_metadata_matrix",
                "Handcrafted early-life features plus ageing-protocol metadata.",
            ),
            FeatureSetSpec(
                "set5_features_metadata_corr95",
                "Set 5: early features + metadata (corr95)",
                "features_metadata_matrix",
                "Set 4 after forward correlation filtering at |r| > 0.95.",
                correlation_threshold=0.95,
            ),
        )

    def selected_feature_sets(self) -> tuple[FeatureSetSpec, ...]:
        if not self.run_feature_sets:
            return self.feature_sets
        by_name = {item.name: item for item in self.feature_sets}
        missing = [name for name in self.run_feature_sets if name not in by_name]
        if missing:
            raise KeyError(f"Unknown feature set(s): {missing}")
        return tuple(by_name[name] for name in self.run_feature_sets)

    def resolve_feature_set(self, feature_set: str | FeatureSetSpec | None = None) -> FeatureSetSpec:
        if isinstance(feature_set, FeatureSetSpec):
            return feature_set
        name = DEFAULT_FEATURE_SET if feature_set is None else str(feature_set)
        by_name = {item.name: item for item in self.feature_sets}
        if name not in by_name:
            raise KeyError(f"Unknown feature set: {name}")
        return by_name[name]

    @property
    def datasets(self) -> list[DatasetSpec]:
        return [
            DatasetSpec("MIT", "data_MIT.mat"),
            DatasetSpec("ISU_ILCC", "data_ISU_ILCC.mat"),
            DatasetSpec("LSD_Primary", "data_LSD_primary.mat"),
            DatasetSpec("LSD_Second", "data_LSD_second.mat"),
            DatasetSpec("Formation", "data_Formation.mat"),
            DatasetSpec("HUST", "data_HUST.mat"),
            DatasetSpec("KIT", "data_KIT.mat"),
            DatasetSpec("TRI_Tesla", "data_TRI_Tesla.mat"),
        ]

    def validate_data_files(self) -> list[Path]:
        """Return the configured dataset paths after checking the release layout."""
        paths = [self.data_dir / spec.filename for spec in self.datasets]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            details = "\n".join(f"- {path}" for path in missing)
            raise FileNotFoundError(
                "Required processed datasets were not found. Run `git lfs pull` "
                f"from {self.project_root} and retry. Missing files:\n{details}"
            )
        return paths

    def to_jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["project_root"] = str(self.project_root)
        payload["output_root"] = str(self.output_root)
        payload["dl_output_root"] = str(self.dl_output_root)
        return payload
