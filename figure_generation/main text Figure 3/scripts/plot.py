from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODULE_ROOT = Path(__file__).resolve().parents[1]
COMMON = MODULE_ROOT.parent / "common"
sys.path.insert(0, str(COMMON))

from config import DATASET_ORDER, DEFAULT_PATHS  # noqa: E402
from pipeline import FIGURE_SPECS  # noqa: E402
from section2_additional import (  # noqa: E402
    COMPOSITE_FILENAME,
    LEGEND_FILENAME,
    draw_all_dataset_grouped_contribution,
    draw_dataset_grouped_contribution,
    draw_grouped_contribution_legend,
    save_composite_figure,
    save_dataset_figure,
    save_grouped_contribution_legend,
)
from section2_figure import render_section2_legends, render_section2_outputs  # noqa: E402


def _render_additional() -> list[Path]:
    source = MODULE_ROOT / "source_data" / "grouped_factor_contribution_by_dataset.csv"
    grouped = pd.read_csv(source)
    output_dir = MODULE_ROOT / "outputs"
    outputs: list[Path] = []
    for dataset in DATASET_ORDER:
        figure = draw_dataset_grouped_contribution(grouped, dataset)
        try:
            outputs.append(save_dataset_figure(figure, output_dir / f"grouped_factor_contribution_{dataset}.png"))
        finally:
            plt.close(figure)
    composite = draw_all_dataset_grouped_contribution(grouped)
    try:
        outputs.append(save_composite_figure(composite, output_dir / COMPOSITE_FILENAME))
    finally:
        plt.close(composite)
    legend = draw_grouped_contribution_legend()
    try:
        outputs.append(save_grouped_contribution_legend(legend, output_dir / LEGEND_FILENAME))
    finally:
        plt.close(legend)
    return outputs


def main() -> None:
    specs = [spec for spec in FIGURE_SPECS if spec.section == 2]
    outputs = render_section2_outputs(DEFAULT_PATHS, specs)
    outputs.extend(render_section2_legends(DEFAULT_PATHS))
    outputs.extend(_render_additional())
    print(f"Rendered {len(set(outputs))} Figure 3 assets in {MODULE_ROOT / 'outputs'}")


if __name__ == "__main__":
    main()

