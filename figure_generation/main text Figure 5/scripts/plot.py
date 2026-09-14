from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd


MODULE_ROOT = Path(__file__).resolve().parents[1]
COMMON_DIR = MODULE_ROOT.parent / "common"
sys.path.insert(0, str(COMMON_DIR))

from config import DATASET_ORDER  # noqa: E402
from convergence import render_dataset_figure  # noqa: E402


def _block(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    selected = frame.loc[frame["source_block"].eq(name)].drop(columns="source_block")
    return selected.dropna(axis=1, how="all").reset_index(drop=True)


def _render_dataset_profiles() -> int:
    output_dir = MODULE_ROOT / "outputs"
    rendered = 0
    for dataset in DATASET_ORDER:
        source_path = MODULE_ROOT / "source_data" / f"{dataset}_source_data.csv"
        source = pd.read_csv(source_path, low_memory=False)
        bundle = SimpleNamespace(
            dataset=dataset,
            model_configs=_block(source, "configuration"),
            embedding_source=_block(source, "embedding"),
            mape_source=_block(source, "mape_convergence"),
            kde_source=_block(source, "kde"),
            fraction_matches=_block(source, "fraction_match"),
            source_data=source,
            manifest={"source": source_path.name},
        )
        result = render_dataset_figure(bundle, output_dir, dpi=600, include_standalone=True)
        plt.close(result["figure"])
        rendered += len(result["paths"])
    return rendered


def main() -> None:
    rendered = _render_dataset_profiles()
    print(f"Rendered {rendered} convergence-profile assets")


if __name__ == "__main__":
    main()
