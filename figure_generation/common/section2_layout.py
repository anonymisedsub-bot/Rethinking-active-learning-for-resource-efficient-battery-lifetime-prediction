from __future__ import annotations

from dataclasses import dataclass


DPI = 600
PAGE_MM = (210.0, 315.0)
PAGE_PX = (4961, 7441)
FONT_SIZE_PT = 10.0
PANEL_LABEL_SIZE_PT = 11.0
AXIS_LINEWIDTH = 0.45
TERM_STANDALONE_AXIS_HEIGHT_MM = 90.0
HEATMAP_STANDALONE_AXIS_MM = (68.0, 32.0)
COMPOSITE_STEM = "fig_section2_composite_210x315mm"

PANEL_STEMS = {
    "a": "fig_active_learning_gain_term_anova",
    "b": "fig_active_learning_gain_grouped_contribution",
    "c": "fig_acquisition_rule_vs_attainable_full_pool_gain",
    "d": "fig_median_active_learning_gain_feature_model",
    "e": "fig_active_learning_gain_by_feature_set",
    "f": "fig_active_learning_gain_by_prediction_model",
}

LEGEND_STEMS = {
    "a": "fig_legend_factor_codes",
    "b": "fig_legend_factor_codes",
    "c": "fig_legend_acquisition_rules",
}


def _mm_to_px(value: float) -> int:
    return int(round(value * DPI / 25.4))


@dataclass(frozen=True)
class PanelGeometry:
    panel: str
    stem: str
    x_mm: float
    y_mm: float
    width_mm: float
    height_mm: float
    width_px: int
    height_px: int


@dataclass(frozen=True)
class LegendGeometry:
    panel: str
    stem: str
    width_mm: float
    height_mm: float
    width_px: int
    height_px: int


_MARGIN_MM = 8.0
_H_GAP_MM = 4.0
_V_GAP_MM = 5.0
_COLUMN_MM = (PAGE_MM[0] - 2 * _MARGIN_MM - 2 * _H_GAP_MM) / 3
_ROW_MM = (PAGE_MM[1] - 2 * _MARGIN_MM - 2 * _V_GAP_MM) / 3
_BOTTOM_PANEL_MM = (PAGE_MM[0] - 2 * _MARGIN_MM - _H_GAP_MM) / 2

_PANEL_RECTS = {
    "a": (
        _MARGIN_MM,
        _MARGIN_MM,
        _COLUMN_MM,
        2 * _ROW_MM + _V_GAP_MM,
    ),
    "b": (
        _MARGIN_MM + _COLUMN_MM + _H_GAP_MM,
        _MARGIN_MM,
        _COLUMN_MM,
        _ROW_MM,
    ),
    "c": (
        _MARGIN_MM + 2 * (_COLUMN_MM + _H_GAP_MM),
        _MARGIN_MM,
        _COLUMN_MM,
        _ROW_MM,
    ),
    "d": (
        _MARGIN_MM + _COLUMN_MM + _H_GAP_MM,
        _MARGIN_MM + _ROW_MM + _V_GAP_MM,
        2 * _COLUMN_MM + _H_GAP_MM,
        _ROW_MM,
    ),
    "e": (
        _MARGIN_MM,
        _MARGIN_MM + 2 * (_ROW_MM + _V_GAP_MM),
        _BOTTOM_PANEL_MM,
        _ROW_MM,
    ),
    "f": (
        _MARGIN_MM + _BOTTOM_PANEL_MM + _H_GAP_MM,
        _MARGIN_MM + 2 * (_ROW_MM + _V_GAP_MM),
        _BOTTOM_PANEL_MM,
        _ROW_MM,
    ),
}

PANEL_GEOMETRY = {
    panel: PanelGeometry(
        panel=panel,
        stem=PANEL_STEMS[panel],
        x_mm=rect[0],
        y_mm=rect[1],
        width_mm=rect[2],
        height_mm=rect[3],
        width_px=_mm_to_px(rect[2]),
        height_px=_mm_to_px(rect[3]),
    )
    for panel, rect in _PANEL_RECTS.items()
}

STANDALONE_PANEL_MM = {
    panel: (
        160.02 if panel == "d" else geometry.width_mm,
        geometry.height_mm,
    )
    for panel, geometry in PANEL_GEOMETRY.items()
}
STANDALONE_PANEL_PX = {
    panel: (_mm_to_px(width_mm), _mm_to_px(height_mm))
    for panel, (width_mm, height_mm) in STANDALONE_PANEL_MM.items()
}

_LEGEND_WIDTH_MM = {"a": _COLUMN_MM, "b": _COLUMN_MM, "c": 92.0}

LEGEND_GEOMETRY = {
    panel: LegendGeometry(
        panel=panel,
        stem=stem,
        width_mm=_LEGEND_WIDTH_MM[panel],
        height_mm=24.0 if panel in {"a", "b"} else 20.0,
        width_px=_mm_to_px(_LEGEND_WIDTH_MM[panel]),
        height_px=_mm_to_px(24.0 if panel in {"a", "b"} else 20.0),
    )
    for panel, stem in LEGEND_STEMS.items()
}
