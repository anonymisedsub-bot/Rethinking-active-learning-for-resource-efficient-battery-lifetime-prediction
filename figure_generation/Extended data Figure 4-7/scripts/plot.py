from __future__ import annotations

import sys
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
COMMON = MODULE_ROOT.parent / "common"
sys.path.insert(0, str(COMMON))

from config import DEFAULT_PATHS  # noqa: E402
from pipeline import FIGURE_SPECS  # noqa: E402
from plotting import render_all  # noqa: E402


def main() -> None:
    specs = [spec for spec in FIGURE_SPECS if spec.section == 4]
    outputs = render_all(DEFAULT_PATHS, specs)
    print(f"Rendered {len(outputs)} Extended Data Figure 4-7 assets in {MODULE_ROOT / 'outputs'}")


if __name__ == "__main__":
    main()

