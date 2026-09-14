from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.transforms import Affine2D
from PIL import Image

from anova import factorial_anova
from config import DATASET_ORDER, FEATURE_ORDER, MODEL_ORDER, NONRANDOM_ACQUISITION_ORDER
from pipeline import DATASET_DISPLAY
from section2_figure import (
    FACTOR_COLORS,
    FACTOR_LEGEND_LABELS,
    add_bar_value_labels,
    apply_section2_style,
)
from section2_layout import AXIS_LINEWIDTH, DPI, FONT_SIZE_PT


ADDITIONAL_SECTION_NAME = "Section 2-Drivers of Active-Learning Gain - additional"
SOURCE_FILENAME = "grouped_factor_contribution_by_dataset.csv"
EXPECTED_CONFIGURATIONS_PER_DATASET = 90
FACTOR_CODES = {"Feature": "B", "Model": "C", "Acquisition": "D"}
OUTPUT_COLUMNS = [
    "dataset",
    "factor",
    "factor_code",
    "grouped_contribution_pp",
    "full_model_r2",
    "reduced_model_r2",
    "n_configurations",
]
FIGURE_WIDTH_MM = 40.0
FIGURE_WIDTH_PX = round(FIGURE_WIDTH_MM / 25.4 * DPI)
BASE_FIGURE_HEIGHT_MM = 52.0
COMPOSITE_FILENAME = "grouped_factor_contribution_all_datasets.png"
COMPOSITE_TARGET_FIGURE_WIDTH_MM = 160.0
COMPOSITE_CONTENT_WIDTH_PX = round(
    COMPOSITE_TARGET_FIGURE_WIDTH_MM / 25.4 * DPI
)
COMPOSITE_CONTENT_WIDTH_MM = COMPOSITE_CONTENT_WIDTH_PX / DPI * 25.4
COMPOSITE_BASE_LEFT_PADDING_MM = 1.0
COMPOSITE_ADDITIONAL_LEFT_PADDING_MM = 3.0
COMPOSITE_LEFT_PADDING_MM = (
    COMPOSITE_BASE_LEFT_PADDING_MM + COMPOSITE_ADDITIONAL_LEFT_PADDING_MM
)
COMPOSITE_BASE_LEFT_PADDING_PX = round(
    COMPOSITE_BASE_LEFT_PADDING_MM / 25.4 * DPI
)
COMPOSITE_ADDITIONAL_LEFT_PADDING_PX = round(
    COMPOSITE_ADDITIONAL_LEFT_PADDING_MM / 25.4 * DPI
)
COMPOSITE_LEFT_PADDING_PX = (
    COMPOSITE_BASE_LEFT_PADDING_PX + COMPOSITE_ADDITIONAL_LEFT_PADDING_PX
)
COMPOSITE_FIGURE_WIDTH_PX = (
    COMPOSITE_CONTENT_WIDTH_PX + COMPOSITE_LEFT_PADDING_PX
)
COMPOSITE_FIGURE_WIDTH_MM = COMPOSITE_FIGURE_WIDTH_PX / DPI * 25.4
COMPOSITE_CONTENT_CENTER_X = (
    COMPOSITE_LEFT_PADDING_MM + COMPOSITE_CONTENT_WIDTH_MM / 2.0
) / COMPOSITE_FIGURE_WIDTH_MM
COMPOSITE_FIGURE_HEIGHT_PX = 2187
COMPOSITE_FIGURE_HEIGHT_MM = COMPOSITE_FIGURE_HEIGHT_PX / DPI * 25.4
COMPOSITE_AXIS_WIDTH_MM = 20.0
COMPOSITE_AXIS_HEIGHT_MM = 25.0
COMPOSITE_X_LEFT_MM = (17.0, 55.0, 92.0, 132.0)
COMPOSITE_Y_BOTTOM_MM = (55.0, 18.0)
COMPOSITE_TITLE_FONT_SIZE_PT = FONT_SIZE_PT - 2.0
COMPOSITE_Y_LIM = (0.0, 120.0)
COMPOSITE_Y_TICKS = (0.0, 50.0, 100.0)
COMPOSITE_Y_LABEL_X = -0.008
COMPOSITE_X_LABEL_Y = 0.08
LEGEND_FILENAME = "grouped_factor_contribution_legend.png"
LEGEND_WIDTH_PX = COMPOSITE_CONTENT_WIDTH_PX
LEGEND_WIDTH_MM = LEGEND_WIDTH_PX / DPI * 25.4
LEGEND_HEIGHT_MM = 12.0
LEGEND_HEIGHT_PX = round(LEGEND_HEIGHT_MM / 25.4 * DPI)


def build_per_dataset_grouped_contribution(early: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset", "feature_set", "model", "acquisition", "early_gain_pp"}
    missing_columns = sorted(required.difference(early.columns))
    if missing_columns:
        raise ValueError(f"Missing per-dataset ANOVA columns: {missing_columns}")
    if early.empty:
        raise ValueError("Per-dataset ANOVA input is empty")

    present_datasets = set(early["dataset"].astype(str))
    missing_datasets = [
        dataset for dataset in DATASET_ORDER if dataset not in present_datasets
    ]
    unexpected_datasets = sorted(present_datasets.difference(DATASET_ORDER))
    if missing_datasets or unexpected_datasets:
        raise ValueError(
            "Dataset inventory mismatch: "
            f"missing={missing_datasets}; unexpected={unexpected_datasets}"
        )

    outputs = []
    for dataset in DATASET_ORDER:
        subset = early.loc[early["dataset"].eq(dataset)].copy()
        if len(subset) != EXPECTED_CONFIGURATIONS_PER_DATASET:
            raise ValueError(
                f"{dataset} has {len(subset)} configurations; "
                f"expected {EXPECTED_CONFIGURATIONS_PER_DATASET}"
            )
        response = pd.to_numeric(subset["early_gain_pp"], errors="coerce")
        if response.isna().any() or not np.isfinite(
            response.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"{dataset} early_gain_pp contains null or non-finite values"
            )
        if subset.duplicated(["feature_set", "model", "acquisition"]).any():
            raise ValueError(f"{dataset} configuration grid is not unique")
        level_inventory = {
            "feature_set": set(subset["feature_set"].astype(str)),
            "model": set(subset["model"].astype(str)),
            "acquisition": set(subset["acquisition"].astype(str)),
        }
        expected_levels = {
            "feature_set": set(FEATURE_ORDER),
            "model": set(MODEL_ORDER),
            "acquisition": set(NONRANDOM_ACQUISITION_ORDER),
        }
        if level_inventory != expected_levels:
            raise ValueError(
                f"{dataset} factor level inventory mismatch: "
                f"observed={level_inventory}; expected={expected_levels}"
            )

        _, grouped = factorial_anova(
            subset,
            "early_gain_pp",
            ["feature_set", "model", "acquisition"],
        )
        grouped = grouped.copy()
        grouped_factors = grouped["factor"].astype(str)
        if (
            len(grouped) != len(FACTOR_CODES)
            or grouped_factors.duplicated().any()
            or set(grouped_factors) != set(FACTOR_CODES)
        ):
            raise ValueError(
                f"{dataset} grouped factors must be exactly "
                f"{list(FACTOR_CODES)}"
            )
        grouped["dataset"] = dataset
        grouped["factor_code"] = grouped["factor"].map(FACTOR_CODES)
        grouped["n_configurations"] = len(subset)
        outputs.append(grouped)

    out = pd.concat(outputs, ignore_index=True)
    out["dataset"] = pd.Categorical(out["dataset"], DATASET_ORDER, ordered=True)
    out["factor_code"] = pd.Categorical(
        out["factor_code"],
        ["B", "C", "D"],
        ordered=True,
    )
    out = out.sort_values(["dataset", "factor_code"]).reset_index(drop=True)
    out["dataset"] = out["dataset"].astype("object")
    out["factor_code"] = out["factor_code"].astype("object")
    return out.loc[:, OUTPUT_COLUMNS]


def draw_dataset_grouped_contribution(
    data: pd.DataFrame,
    dataset: str,
) -> mpl.figure.Figure:
    with mpl.rc_context():
        apply_section2_style()
        frame = data.loc[data["dataset"].eq(dataset)].copy()
        frame["factor_code"] = pd.Categorical(
            frame["factor_code"],
            ["B", "C", "D"],
            ordered=True,
        )
        frame = frame.sort_values("factor_code")
        codes = frame["factor_code"].astype("object").tolist()
        if codes != ["B", "C", "D"]:
            raise ValueError(f"{dataset} must contain factor codes B, C, and D")

        figure = plt.figure(
            figsize=(FIGURE_WIDTH_MM / 25.4, BASE_FIGURE_HEIGHT_MM / 25.4),
            layout="constrained",
        )
        axis = figure.add_subplot(111)
        positions = np.arange(len(frame))
        axis.bar(
            positions,
            frame["grouped_contribution_pp"],
            width=0.66,
            color=[FACTOR_COLORS[code] for code in codes],
            edgecolor="white",
            linewidth=AXIS_LINEWIDTH,
            alpha=0.60,
        )
        axis.set_xticks(positions, labels=codes)
        axis.set_ylim(0.0, 100.0)
        axis.set_xlabel("Factor")
        axis.set_ylabel("Incremental variance\nexplained (%)")
        axis.set_title(DATASET_DISPLAY.get(dataset, dataset))
        axis.set_box_aspect(1.0)
        axis.grid(axis="y", color="#E5E7E9", linewidth=AXIS_LINEWIDTH)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_linewidth(AXIS_LINEWIDTH)
        axis.tick_params(
            axis="both",
            which="major",
            width=AXIS_LINEWIDTH,
            length=2.5,
        )
        return figure


def draw_all_dataset_grouped_contribution(
    data: pd.DataFrame,
) -> mpl.figure.Figure:
    with mpl.rc_context():
        apply_section2_style()
        figure = plt.figure(
            figsize=(
                COMPOSITE_FIGURE_WIDTH_PX / DPI,
                COMPOSITE_FIGURE_HEIGHT_PX / DPI,
            ),
        )
        content_to_figure = (
            Affine2D()
            .scale(
                COMPOSITE_CONTENT_WIDTH_MM / COMPOSITE_FIGURE_WIDTH_MM,
                1.0,
            )
            .translate(
                COMPOSITE_LEFT_PADDING_MM / COMPOSITE_FIGURE_WIDTH_MM,
                0.0,
            )
            + figure.transFigure
        )
        axes: list[mpl.axes.Axes] = []
        for dataset_index, dataset in enumerate(DATASET_ORDER):
            row = dataset_index // 4
            column = dataset_index % 4
            rect = (
                COMPOSITE_X_LEFT_MM[column] / COMPOSITE_FIGURE_WIDTH_MM,
                COMPOSITE_Y_BOTTOM_MM[row] / COMPOSITE_FIGURE_HEIGHT_MM,
                COMPOSITE_AXIS_WIDTH_MM / COMPOSITE_FIGURE_WIDTH_MM,
                COMPOSITE_AXIS_HEIGHT_MM / COMPOSITE_FIGURE_HEIGHT_MM,
            )
            shared_axes = (
                {}
                if not axes
                else {"sharex": axes[0], "sharey": axes[0]}
            )
            axis = figure.add_axes(rect, **shared_axes)
            axes.append(axis)
            frame = data.loc[data["dataset"].eq(dataset)].copy()
            frame["factor_code"] = pd.Categorical(
                frame["factor_code"],
                ["B", "C", "D"],
                ordered=True,
            )
            frame = frame.sort_values("factor_code")
            codes = frame["factor_code"].astype("object").tolist()
            if codes != ["B", "C", "D"]:
                raise ValueError(
                    f"{dataset} must contain factor codes B, C, and D"
                )

            positions = np.arange(len(frame))
            bars = axis.bar(
                positions,
                frame["grouped_contribution_pp"],
                width=0.66,
                color=[FACTOR_COLORS[code] for code in codes],
                edgecolor="white",
                linewidth=AXIS_LINEWIDTH,
                alpha=0.60,
            )
            add_bar_value_labels(
                axis,
                bars,
                frame["grouped_contribution_pp"],
            )
            axis.set_xticks(positions, labels=codes)
            axis.set_ylim(*COMPOSITE_Y_LIM)
            axis.set_yticks(COMPOSITE_Y_TICKS)
            axis.set_title(
                f"[{DATASET_DISPLAY.get(dataset, dataset)} dataset]",
                fontsize=COMPOSITE_TITLE_FONT_SIZE_PT,
            )
            axis.grid(
                axis="y",
                color="#E5E7E9",
                linewidth=AXIS_LINEWIDTH,
            )
            axis.set_axisbelow(True)
            for spine in axis.spines.values():
                spine.set_linewidth(AXIS_LINEWIDTH)
            axis.tick_params(
                axis="both",
                which="major",
                width=AXIS_LINEWIDTH,
                length=2.5,
                labelbottom=row == 1,
                labelleft=column == 0,
            )

        figure.supxlabel(
            "Factor",
            x=COMPOSITE_CONTENT_CENTER_X,
            y=COMPOSITE_X_LABEL_Y,
            fontsize=FONT_SIZE_PT,
        )
        figure.supylabel(
            "Incremental variance explained (%)",
            x=COMPOSITE_Y_LABEL_X,
            transform=content_to_figure,
            fontsize=FONT_SIZE_PT,
        )
        return figure


def draw_grouped_contribution_legend() -> mpl.figure.Figure:
    with mpl.rc_context():
        apply_section2_style()
        figure = plt.figure(
            figsize=(LEGEND_WIDTH_PX / DPI, LEGEND_HEIGHT_PX / DPI)
        )
        handles = [
            Patch(
                facecolor=FACTOR_COLORS[code],
                edgecolor="none",
                alpha=0.60,
                label=FACTOR_LEGEND_LABELS[code],
            )
            for code in FACTOR_CODES.values()
        ]
        figure.legend(
            handles=handles,
            loc="center",
            ncol=3,
            handlelength=0.9,
            columnspacing=0.8,
            labelspacing=0.25,
            borderaxespad=0,
        )
        return figure


def _fit_figure_height(
    figure: mpl.figure.Figure,
    *,
    width_px: int = FIGURE_WIDTH_PX,
    base_height_mm: float = BASE_FIGURE_HEIGHT_MM,
) -> None:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    tight = figure.get_tightbbox(renderer)
    required_inches = max(
        base_height_mm / 25.4,
        float(tight.height) + 0.08,
    )
    figure.set_size_inches(
        width_px / DPI,
        required_inches,
        forward=True,
    )
    figure.canvas.draw()


def save_dataset_figure(
    figure: mpl.figure.Figure,
    output: Path,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    _fit_figure_height(figure)
    figure.set_size_inches(
        FIGURE_WIDTH_PX / DPI,
        figure.get_figheight(),
        forward=True,
    )
    figure.savefig(
        output,
        format="png",
        dpi=DPI,
        facecolor="white",
    )
    return output


def save_composite_figure(
    figure: mpl.figure.Figure,
    output: Path,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.set_size_inches(
        COMPOSITE_FIGURE_WIDTH_PX / DPI,
        COMPOSITE_FIGURE_HEIGHT_PX / DPI,
        forward=True,
    )
    figure.savefig(
        output,
        format="png",
        dpi=DPI,
        facecolor="white",
    )
    return output


def save_grouped_contribution_legend(
    figure: mpl.figure.Figure,
    output: Path,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.set_size_inches(
        LEGEND_WIDTH_PX / DPI,
        LEGEND_HEIGHT_PX / DPI,
        forward=True,
    )
    figure.savefig(
        output,
        format="png",
        dpi=DPI,
        facecolor="white",
    )
    return output


def _verify_staged_artifacts(
    csv_path: Path,
    png_paths: list[Path],
) -> None:
    source = pd.read_csv(csv_path)
    if source.shape != (len(DATASET_ORDER) * len(FACTOR_CODES), len(OUTPUT_COLUMNS)):
        raise ValueError(
            f"Unexpected supplementary CSV shape: {source.shape}; expected (24, 7)"
        )
    if source.columns.tolist() != OUTPUT_COLUMNS:
        raise ValueError(
            f"Unexpected supplementary CSV columns: {source.columns.tolist()}"
        )

    expected_names = [
        f"grouped_factor_contribution_{dataset}.png"
        for dataset in DATASET_ORDER
    ] + [COMPOSITE_FILENAME, LEGEND_FILENAME]
    observed_names = [path.name for path in png_paths]
    if observed_names != expected_names:
        raise ValueError(
            f"Unexpected supplementary PNG inventory: {observed_names}"
        )
    for path in png_paths:
        with Image.open(path) as image:
            dpi = image.info.get("dpi", (None, None))
            if path.name == COMPOSITE_FILENAME:
                expected_width = COMPOSITE_FIGURE_WIDTH_PX
                expected_height = COMPOSITE_FIGURE_HEIGHT_PX
            elif path.name == LEGEND_FILENAME:
                expected_width = LEGEND_WIDTH_PX
                expected_height = LEGEND_HEIGHT_PX
            else:
                expected_width = FIGURE_WIDTH_PX
                expected_height = None
            dpi_valid = (
                dpi[0] is not None
                and dpi[1] is not None
                and abs(float(dpi[0]) - DPI) <= 1.0
                and abs(float(dpi[1]) - DPI) <= 1.0
            )
            if (
                image.width != expected_width
                or image.height <= 0
                or (expected_height is not None and image.height != expected_height)
                or not dpi_valid
            ):
                raise ValueError(
                    f"Invalid supplementary PNG geometry for {path.name}: "
                    f"size={image.size}; dpi={dpi}"
                )


def render_section2_additional(
    early: pd.DataFrame,
    output_root: Path,
) -> tuple[Path, list[Path]]:
    grouped = build_per_dataset_grouped_contribution(early)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=".section2-additional-staging-",
        dir=output_root,
    ) as temporary_root:
        stage_root = Path(temporary_root)
        stage_source = stage_root / "source_data" / SOURCE_FILENAME
        stage_results = stage_root / "outputs"
        stage_source.parent.mkdir(parents=True, exist_ok=True)
        stage_results.mkdir(parents=True, exist_ok=True)
        grouped.to_csv(stage_source, index=False)

        staged_pngs = []
        for dataset in DATASET_ORDER:
            figure = draw_dataset_grouped_contribution(grouped, dataset)
            try:
                staged_pngs.append(
                    save_dataset_figure(
                        figure,
                        stage_results
                        / f"grouped_factor_contribution_{dataset}.png",
                    )
                )
            finally:
                plt.close(figure)

        composite = draw_all_dataset_grouped_contribution(grouped)
        try:
            staged_pngs.append(
                save_composite_figure(
                    composite,
                    stage_results / COMPOSITE_FILENAME,
                )
            )
        finally:
            plt.close(composite)

        legend = draw_grouped_contribution_legend()
        try:
            staged_pngs.append(
                save_grouped_contribution_legend(
                    legend,
                    stage_results / LEGEND_FILENAME,
                )
            )
        finally:
            plt.close(legend)
        _verify_staged_artifacts(stage_source, staged_pngs)

        target_root = output_root / ADDITIONAL_SECTION_NAME
        target_source = target_root / "source_data" / SOURCE_FILENAME
        target_results = target_root / "outputs"
        target_source.parent.mkdir(parents=True, exist_ok=True)
        target_results.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stage_source, target_source)

        outputs = []
        for staged in staged_pngs:
            target = target_results / staged.name
            shutil.copy2(staged, target)
            outputs.append(target)

    return target_source, outputs
