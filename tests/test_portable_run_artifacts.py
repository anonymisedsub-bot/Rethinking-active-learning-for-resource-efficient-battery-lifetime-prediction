from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ABSOLUTE_PATH = re.compile(
    r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]|(?<![A-Za-z0-9_.-])/(?:home|Users|tmp|var|private|mnt)/"
)


RUNNERS = {
    "active_learning": """
from config import ExperimentConfig
from experiments import run_experiments

cfg = ExperimentConfig(
    pipeline="active_learning",
    run_datasets=("__metadata_only__",),
    run_feature_sets=("set3_features",),
    run_trials=(1,),
    run_target_transforms=("log", "raw"),
    run_split_modes=("protocol_holdout",),
    output_root=output_root,
)
run_experiments(cfg)
""",
    "deep_learning_baseline": """
from config import ExperimentConfig
from dl_benchmark import run_dl_benchmark

cfg = ExperimentConfig(
    pipeline="dl_benchmark",
    run_datasets=("__metadata_only__",),
    run_feature_sets=("set3_features",),
    run_trials=(1,),
    run_target_transforms=("log", "raw"),
    run_split_modes=("protocol_holdout",),
    dl_models=("MLP",),
    dl_epochs=1,
    dl_output_root=output_root,
)
run_dl_benchmark(cfg)
""",
}


@pytest.mark.parametrize("module_name", RUNNERS)
def test_generated_run_metadata_contains_only_portable_paths(
    module_name: str,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / module_name
    program = f"""
import sys
from pathlib import Path

package_root = Path(sys.argv[1])
output_root = Path(sys.argv[2])
sys.path.insert(0, str(package_root / {module_name!r}))
{RUNNERS[module_name]}
"""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [sys.executable, "-c", program, str(PACKAGE_ROOT), str(output_root)],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    violations: list[str] = []
    for path in output_root.rglob("*"):
        if path.suffix.lower() not in {".csv", ".json"}:
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
            if ABSOLUTE_PATH.search(line):
                violations.append(f"{path.relative_to(output_root)}:{line_number}")

    assert not violations, "Generated metadata contains absolute paths:\n" + "\n".join(violations)
    for path in output_root.rglob("run_config.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "data_directory" not in payload


@pytest.mark.parametrize("module_name", ["active_learning", "deep_learning_baseline"])
def test_feature_run_index_paths_resolve_against_the_index_directory(
    module_name: str,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "indexed_run"
    actual_child = run_dir / "nested" / "feature set 3"
    decoy_child = tmp_path / "nested" / "feature set 3"
    actual_child.mkdir(parents=True)
    decoy_child.mkdir(parents=True)
    (run_dir / "feature_set_runs.csv").write_text(
        "feature_set,feature_run_dir\nset3_features,nested/feature set 3\n",
        encoding="utf-8",
    )
    program = """
import os
import sys
from pathlib import Path

import pandas as pd

module_root = Path(sys.argv[1])
run_dir = Path(sys.argv[2])
expected = Path(sys.argv[3]).resolve()
sys.path.insert(0, str(module_root))
import visualize

seen = []
visualize.read_dl_benchmark_results = lambda path: (
    seen.append(Path(path).resolve())
    or (
        pd.DataFrame(columns=["dataset", "trial", "model", "mape"]),
        pd.DataFrame(columns=["dataset", "trial", "model", "lifetime", "pred_lifetime"]),
    )
)
visualize._dl_mape_table = lambda frame: pd.DataFrame()
visualize.plot_dl_benchmark_mape_bars = lambda summary, out_dir: Path(out_dir) / "mape.png"
visualize.create_dl_benchmark_visualizations(run_dir)
if seen != [expected]:
    raise AssertionError(f"resolved {seen}, expected {[expected]}")
"""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            str(PACKAGE_ROOT / module_name),
            str(run_dir),
            str(actual_child),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
