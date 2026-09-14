from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "figure_generation"
# Raw training outputs are deliberately absent from the public release. These
# placeholders are retained only for source-table regeneration by data owners.
AL_ROOT = PROJECT_ROOT / "active_learning" / "outputs_AL" / "target-log__split-PH"
DL_ROOT = PROJECT_ROOT / "deep_learning_baseline" / "outputs_DL_benchmark" / "target-log__split-PH"

SECTION_NAMES = {
    1: "main text Figure 2",
    2: "main text Figure 3",
    3: "main text Figure 4",
    4: "Extended data Figure 4-7",
}

DATASET_ORDER = [
    "MIT",
    "ISU_ILCC",
    "LSD_Primary",
    "LSD_Second",
    "HUST",
    "KIT",
    "Formation",
    "TRI_Tesla",
]
FEATURE_ORDER = [
    "set1_dQn_m",
    "set2_early_soh",
    "set3_features",
    "set4_features_metadata",
    "set5_features_metadata_corr95",
]
MODEL_ORDER = ["GPR", "RF", "AE_ENet"]
DEEP_MODEL_ORDER = ["CNN", "DeepVAE", "MLP", "TabNet"]
SECTION1_MODEL_ORDER = [*MODEL_ORDER, *DEEP_MODEL_ORDER]
ACQUISITION_ORDER = [
    "random_selection",
    "diversity_oneshot",
    "diversity_iterative",
    "coverage",
    "exploration",
    "exploitation",
    "hybrid",
]
NONRANDOM_ACQUISITION_ORDER = ACQUISITION_ORDER[1:]
DURATION_UNITS = {
    "MIT": "cycles",
    "ISU_ILCC": "weeks",
    "LSD_Primary": "cycles",
    "LSD_Second": "cycles",
    "HUST": "cycles",
    "KIT": "EFC",
    "Formation": "cycles",
    "TRI_Tesla": "EFC",
}


@dataclass(frozen=True)
class AnalysisPaths:
    project_root: Path
    output_root: Path
    al_root: Path
    dl_root: Path


DEFAULT_PATHS = AnalysisPaths(
    project_root=PROJECT_ROOT,
    output_root=OUTPUT_ROOT,
    al_root=AL_ROOT,
    dl_root=DL_ROOT,
)
