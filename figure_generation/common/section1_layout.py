from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


DPI = 600
PAGE_MM = (210.0, 315.0)
PAGE_PX = (4961, 7441)

ANALYTICAL_STEMS = (
    "fig_full_pool_primary_vs_deep_violin",
    "fig_attainable_gain_sign_fraction_vertical",
    "fig_initialisation_gain_sign_fraction_vertical",
    "fig_full_pool_performance_ring_heatmap",
    "fig_attainable_full_pool_gain_ring_heatmap",
    "fig_initialisation_gain_ring_heatmap",
    "fig_full_pool_term_contribution",
    "fig_attainable_full_pool_gain_term_contribution",
    "fig_initial_prediction_performance_term_contribution",
    "fig_full_pool_grouped_contribution",
    "fig_attainable_full_pool_gain_grouped_contribution",
    "fig_initial_prediction_performance_grouped_contribution",
)
LEGEND_STEMS = (
    "fig_legend_dataset_codes",
    "fig_legend_feature_codes",
    "fig_legend_feature_order_clockwise",
    "fig_legend_colorbar_full_pool_performance",
    "fig_legend_colorbar_attainable_full_pool_gain",
    "fig_legend_colorbar_initialisation_gain",
    "fig_legend_anova_factors_3",
    "fig_legend_anova_factors_4",
    "fig_legend_term_classes",
)
AUXILIARY_STEMS = (
    "fig_full_pool_primary_vs_deep_violin_trial_level",
)

# Kept during the migration so older callers fail gracefully rather than at import.
COMPOSITE_STEM = "fig_section1_composite_210x250mm"

_LEFT_MM = 8.0
_TOP_MM = 7.0
_H_GAP_MM = 3.0
_V_GAP_MM = 4.0
_BASE_COLUMN_MM = 46.25
_THIRD_COLUMN_MM = 62.6666666667
_ROW_HEIGHT_MM = 72.25

PANEL_MM_RECTS = {
    "a": (_LEFT_MM, _TOP_MM, 95.5, _ROW_HEIGHT_MM),
    "b": (106.5, _TOP_MM, _BASE_COLUMN_MM, _ROW_HEIGHT_MM),
    "c": (155.75, _TOP_MM, _BASE_COLUMN_MM, _ROW_HEIGHT_MM),
    "d": (_LEFT_MM, 83.25, _THIRD_COLUMN_MM, _ROW_HEIGHT_MM),
    "e": (73.6666666667, 83.25, _THIRD_COLUMN_MM, _ROW_HEIGHT_MM),
    "f": (139.3333333334, 83.25, _THIRD_COLUMN_MM, _ROW_HEIGHT_MM),
    "g": (_LEFT_MM, 159.5, _THIRD_COLUMN_MM, _ROW_HEIGHT_MM),
    "h": (73.6666666667, 159.5, _THIRD_COLUMN_MM, _ROW_HEIGHT_MM),
    "i": (139.3333333334, 159.5, _THIRD_COLUMN_MM, _ROW_HEIGHT_MM),
    "j": (_LEFT_MM, 235.75, _THIRD_COLUMN_MM, _ROW_HEIGHT_MM),
    "k": (73.6666666667, 235.75, _THIRD_COLUMN_MM, _ROW_HEIGHT_MM),
    "l": (139.3333333334, 235.75, _THIRD_COLUMN_MM, _ROW_HEIGHT_MM),
}

PANEL_DEFINITIONS = {
    "a": "Seven-model full-pool MAPE violin distributions",
    "b": "Positive and non-positive attainable full-pool gain fractions",
    "c": "Positive and non-positive initialisation gain fractions",
    "d": "Dataset-feature-model full-pool MAPE ring heatmap",
    "e": "Dataset-feature-model attainable full-pool gain ring heatmap",
    "f": "Dataset-feature-model initialisation gain ring heatmap",
    "g": "Three-factor full-pool MAPE term-level contributions",
    "h": "Three-factor attainable full-pool gain term-level contributions",
    "i": "Four-factor initial prediction MAPE term-level contributions",
    "j": "Three-factor full-pool MAPE grouped factor contributions",
    "k": "Three-factor attainable full-pool gain grouped factor contributions",
    "l": "Four-factor initial prediction MAPE grouped factor contributions",
}

PANEL_RESPONSES = {
    "a": "Full-pool prediction performance",
    "b": "Attainable full-pool gain",
    "c": "Initialisation gain",
    "d": "Full-pool prediction performance",
    "e": "Attainable full-pool gain",
    "f": "Initialisation gain",
    "g": "Full-pool prediction performance",
    "h": "Attainable full-pool gain",
    "i": "Initialisation prediction performance",
    "j": "Full-pool prediction performance",
    "k": "Attainable full-pool gain",
    "l": "Initialisation prediction performance",
}


def _mm_to_px(value: float) -> int:
    return int(round(value * DPI / 25.4))


@dataclass(frozen=True)
class PanelGeometry:
    panel: str
    stem: str
    source_data_stem: str
    row: int
    x_mm: float
    y_mm: float
    width_mm: float
    height_mm: float
    x_px: int
    y_px: int
    width_px: int
    height_px: int
    response: str
    definition: str


PANEL_GEOMETRY = {
    panel: PanelGeometry(
        panel=panel,
        stem=stem,
        source_data_stem=stem,
        row=(index // 3) + 1 if index >= 3 else 1,
        x_mm=PANEL_MM_RECTS[panel][0],
        y_mm=PANEL_MM_RECTS[panel][1],
        width_mm=PANEL_MM_RECTS[panel][2],
        height_mm=PANEL_MM_RECTS[panel][3],
        x_px=_mm_to_px(PANEL_MM_RECTS[panel][0]),
        y_px=_mm_to_px(PANEL_MM_RECTS[panel][1]),
        width_px=_mm_to_px(PANEL_MM_RECTS[panel][2]),
        height_px=_mm_to_px(PANEL_MM_RECTS[panel][3]),
        response=PANEL_RESPONSES[panel],
        definition=PANEL_DEFINITIONS[panel],
    )
    for index, (panel, stem) in enumerate(zip("abcdefghijkl", ANALYTICAL_STEMS))
}


@dataclass(frozen=True)
class AuxiliarySpec:
    stem: str
    width_px: int
    height_px: int
    definition: str


AUXILIARY_SPECS = {
    stem: AuxiliarySpec(
        stem=stem,
        width_px=PANEL_GEOMETRY["a"].width_px,
        height_px=PANEL_GEOMETRY["a"].height_px,
        definition="Seven-model trial-level full-pool MAPE violin distributions",
    )
    for stem in AUXILIARY_STEMS
}


@dataclass(frozen=True)
class LegendSpec:
    stem: str
    width_mm: float
    height_mm: float
    width_px: int
    height_px: int
    definition: str


_LEGEND_DEFINITIONS = {
    "fig_legend_dataset_codes": (92.0, 18.0, "Dataset code mapping"),
    "fig_legend_feature_codes": (78.0, 24.0, "Feature-set code mapping"),
    "fig_legend_feature_order_clockwise": (
        78.0,
        28.0,
        "Clockwise within-sector feature-set order",
    ),
    "fig_legend_colorbar_full_pool_performance": (
        24.0,
        60.0,
        "Full-pool performance vertical colorbar",
    ),
    "fig_legend_colorbar_attainable_full_pool_gain": (
        24.0,
        60.0,
        "Attainable full-pool gain vertical colorbar",
    ),
    "fig_legend_colorbar_initialisation_gain": (
        24.0,
        60.0,
        "Initialisation gain vertical colorbar",
    ),
    "fig_legend_anova_factors_3": (105.0, 20.0, "Three-factor ANOVA code mapping"),
    "fig_legend_anova_factors_4": (92.0, 24.0, "Four-factor ANOVA code mapping"),
    "fig_legend_term_classes": (82.0, 16.0, "ANOVA term-class colors"),
}

LEGEND_SPECS = {
    stem: LegendSpec(
        stem=stem,
        width_mm=values[0],
        height_mm=values[1],
        width_px=_mm_to_px(values[0]),
        height_px=_mm_to_px(values[1]),
        definition=values[2],
    )
    for stem, values in _LEGEND_DEFINITIONS.items()
}


def build_layout_manifest() -> pd.DataFrame:
    return pd.DataFrame([asdict(PANEL_GEOMETRY[panel]) for panel in "abcdefghijkl"])


def build_composite_manifest() -> pd.DataFrame:
    return build_layout_manifest()
