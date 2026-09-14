from __future__ import annotations

import sys
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
COMMON = MODULE_ROOT.parent / "common"
sys.path.insert(0, str(COMMON))

from config import DEFAULT_PATHS  # noqa: E402
from pipeline import FIGURE_SPECS  # noqa: E402
from section1_figure import render_section1_outputs  # noqa: E402


def main() -> None:
    specs = [spec for spec in FIGURE_SPECS if spec.section == 1]
    outputs = render_section1_outputs(DEFAULT_PATHS, specs)
    print(f"Rendered {len(outputs)} Figure 2 assets in {MODULE_ROOT / 'outputs'}")


if __name__ == "__main__":
    main()

