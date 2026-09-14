from __future__ import annotations

import csv
import hashlib
import importlib.util
import re
from pathlib import Path

import pandas as pd
from PIL import Image


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = PACKAGE_ROOT / "figure_generation"

FIGURE_COUNTS = {
    "main text Figure 2": 23,
    "main text Figure 3": 7,
    "main text Figure 4": 9,
    "main text Figure 5": 8,
    "main text Figure 6": 2,
    "Extended data Figure 4-7": 12,
    "SI Note 11 figures (Active-learning region determination)": 1,
    "SI Note 13 figures (3D convergence profiles)": 5,
}

STALE_NAME = re.compile(r"^(?:panel_(?:[A-Za-z]|\d+)_|fig\d+[A-Za-z]?_)", re.IGNORECASE)


def load_main_text_figure_4_module():
    module_path = FIGURE_ROOT / "main text Figure 4" / "scripts" / "plot.py"
    spec = importlib.util.spec_from_file_location("github_package_main_text_figure_4", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_figure_directory_inventory() -> None:
    present = {
        path.name
        for path in FIGURE_ROOT.iterdir()
        if path.is_dir() and path.name != "common"
    }
    assert present == set(FIGURE_COUNTS)


def test_each_figure_module_is_self_contained() -> None:
    for name, expected_count in FIGURE_COUNTS.items():
        root = FIGURE_ROOT / name
        assert (root / "README.md").is_file(), name
        assert (root / "scripts" / "plot.py").is_file(), name
        assert (root / "source_data" / "manifest.csv").is_file(), name
        csv_files = [path for path in (root / "source_data").glob("*.csv") if path.name != "manifest.csv"]
        assert len(csv_files) == expected_count, (name, len(csv_files), expected_count)
        assert any((root / "outputs").glob("*.png")), name


def test_source_data_manifests_cover_every_csv() -> None:
    for name in FIGURE_COUNTS:
        source_dir = FIGURE_ROOT / name / "source_data"
        with (source_dir / "manifest.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        recorded = {row["filename"] for row in rows}
        actual = {path.name for path in source_dir.glob("*.csv") if path.name != "manifest.csv"}
        assert recorded == actual, name
        assert all(row["sha256"] and row["n_rows"] for row in rows), name
        for row in rows:
            path = source_dir / row["filename"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], path
            with path.open(newline="", encoding="utf-8-sig") as handle:
                n_rows = sum(1 for _ in csv.reader(handle)) - 1
            assert n_rows == int(row["n_rows"]), path


def test_old_standalone_number_prefixes_are_removed() -> None:
    stale = [
        str(path.relative_to(FIGURE_ROOT))
        for path in FIGURE_ROOT.rglob("*")
        if path.is_file() and STALE_NAME.match(path.name)
    ]
    assert not stale, "Stale numbered figure/panel names:\n" + "\n".join(stale)


def test_main_text_figure_5_excludes_mechanism_asset_and_code() -> None:
    root = FIGURE_ROOT / "main text Figure 5"
    assert not (root / "outputs" / "mechanism_core.png").exists()
    assert not (root / "scripts" / "mechanism.py").exists()
    assert not list((root / "source_data").glob("mechanism_*.csv"))
    assert "mechanism" not in (root / "scripts" / "plot.py").read_text(encoding="utf-8").lower()


def test_main_text_figure_4_uses_aulc_display_labels() -> None:
    module = load_main_text_figure_4_module()
    assert module.COLORBAR_LABEL == "Relative AULC improvement (%)"
    assert module.DATASET_HEATMAP_TITLE.startswith("Relative AULC improvement")
    assert module.FEATURE_MODEL_HEATMAP_TITLE.startswith("Relative AULC improvement")

    fig, ax = module.plt.subplots()
    module.draw_definition(ax, pd.DataFrame())
    assert any(text.get_text() == r"$\Delta$AULC" for text in ax.texts)
    module.plt.close(fig)

    _, labels = module.panel_a_legend_handles()
    assert labels[-1] == r"$\Delta$AULC"


def test_main_text_figure_4_uses_two_line_active_learning_gain_ylabel() -> None:
    module = load_main_text_figure_4_module()
    data = pd.DataFrame(
        [
            {"acquisition": "random_selection", "early_gain_pp": 0.0},
            {"acquisition": "coverage", "early_gain_pp": 1.0},
        ]
    )
    fig, ax = module.plt.subplots()
    module.draw_early_gain_distribution(ax, data)
    assert ax.get_ylabel() == "Active-learning\nperformance gain (pp)"
    module.plt.close(fig)


def test_png_outputs_are_valid_high_resolution_rasters() -> None:
    pngs = list(FIGURE_ROOT.rglob("*.png"))
    assert pngs
    for path in pngs:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            dpi = image.info.get("dpi", (0.0, 0.0))
            assert image.width >= 200 and image.height >= 100, path
            assert min(float(value) for value in dpi) >= 295.0, (path, dpi)
