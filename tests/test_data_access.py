from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FILES = {
    "data_Formation.mat",
    "data_HUST.mat",
    "data_ISU_ILCC.mat",
    "data_KIT.mat",
    "data_LSD_primary.mat",
    "data_LSD_second.mat",
    "data_MIT.mat",
    "data_TRI_Tesla.mat",
}


def _load_config(module_dir: str):
    path = PACKAGE_ROOT / module_dir / "config.py"
    spec = importlib.util.spec_from_file_location(f"release_{module_dir}_config", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_both_frameworks_resolve_peer_data_directory(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    for module_dir in ("active_learning", "deep_learning_baseline"):
        module = _load_config(module_dir)
        cfg = module.ExperimentConfig()
        assert cfg.project_root == PACKAGE_ROOT
        assert cfg.data_dir == PACKAGE_ROOT / "data"


def test_both_frameworks_declare_all_eight_dataset_files() -> None:
    for module_dir in ("active_learning", "deep_learning_baseline"):
        module = _load_config(module_dir)
        cfg = module.ExperimentConfig()
        declared = {spec.filename for spec in cfg.datasets}
        assert declared == EXPECTED_FILES
        assert all((cfg.data_dir / name).is_file() for name in declared)


def test_both_frameworks_validate_the_distributed_data() -> None:
    for module_dir in ("active_learning", "deep_learning_baseline"):
        module = _load_config(module_dir)
        cfg = module.ExperimentConfig()
        resolved = cfg.validate_data_files()
        assert {path.name for path in resolved} == EXPECTED_FILES
        assert all(path.parent == PACKAGE_ROOT / "data" for path in resolved)
