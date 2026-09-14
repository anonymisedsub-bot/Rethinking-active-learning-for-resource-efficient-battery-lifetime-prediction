from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = PACKAGE_ROOT / "figure_generation"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_script(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_figure_6_cli_does_not_rewrite_source_tables(tmp_path: Path) -> None:
    original_root = FIGURE_ROOT / "main text Figure 6"
    isolated_root = tmp_path / "figure_6"
    source_dir = isolated_root / "source_data"
    source_dir.mkdir(parents=True)
    source_names = [
        "fig_nfp_performance_indicator_summary.csv",
        "fig_nfp_trial_level_distributions.csv",
    ]
    for name in source_names:
        shutil.copy2(original_root / "source_data" / name, source_dir / name)
    before = {name: _sha256(source_dir / name) for name in source_names}

    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-B",
            str(original_root / "scripts" / "plot.py"),
            "--source-dir",
            str(source_dir),
            "--output-dir",
            str(isolated_root),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    after = {name: _sha256(source_dir / name) for name in source_names}
    assert after == before


def test_si_note_13_renderer_does_not_rewrite_source_table(tmp_path: Path) -> None:
    original_root = FIGURE_ROOT / "SI Note 13 figures (3D convergence profiles)"
    module = _load_script(
        "release_si_note_13_plot",
        original_root / "scripts" / "plot.py",
    )
    source_dir = tmp_path / "source_data"
    results_dir = tmp_path / "results"
    source_dir.mkdir()
    source_path = source_dir / "gpr_set5_trial_mape.csv"
    shutil.copy2(original_root / "source_data" / source_path.name, source_path)
    before = _sha256(source_path)

    module.render_feature_set_artifacts(
        data=pd.read_csv(source_path),
        feature_set_index=5,
        feature_set=module.FEATURE_SET_SPECS[5],
        source_dir=source_dir,
        results_dir=results_dir,
        dpi=100,
        write_source_data=False,
    )

    assert _sha256(source_path) == before
