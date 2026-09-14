from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ROOT = PACKAGE_ROOT / "active_learning"


@pytest.mark.parametrize("legacy_name", ["ElasticNet", "SVR", "XGB", "LGB"])
def test_undocumented_legacy_regressors_are_not_part_of_the_release(legacy_name: str) -> None:
    program = """
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1])))
from models import _base_regressor

try:
    _base_regressor(sys.argv[2], seed=2026)
except ValueError as exc:
    if "Unknown regressor" not in str(exc):
        raise
else:
    raise AssertionError(f"Undocumented regressor remains callable: {sys.argv[2]}")
"""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [sys.executable, "-c", program, str(ACTIVE_ROOT), legacy_name],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
