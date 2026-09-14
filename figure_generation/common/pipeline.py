from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from anova import factorial_anova
from config import (
    DATASET_ORDER,
    DURATION_UNITS,
    FEATURE_ORDER,
    AnalysisPaths,
    SECTION_NAMES,
)
from metrics import (
    RANDOM,
    build_attainable_gain,
    build_cold_start,
    build_full_pool,
    build_initialisation_gain,
    build_initialisation_table,
    compute_pointwise_gain,
    positive_configuration_fraction,
    summarise_early_gain,
    summarise_overall_gain,
)
from nfp import build_duration_table, build_nfp_crossings
from section2_analysis import build_feature_model_median, build_moderation_table
from section1_layout import build_layout_manifest


@dataclass(frozen=True)
class FigureSpec:
    section: int
    stem: str
    source_table: str


NFP_FRACTIONS = [0.92, 0.95, 0.98]
SECTION4_NFP_FIGURE_KINDS = (
    "required_pool_fraction_heatmap",
    "required_full_life_tests",
    "total_experimental_duration",
    "duration_vs_required_pool_fraction",
)
SECTION4_NFP_FIGURE_SPECS = [
    FigureSpec(
        4,
        f"fig_nfp{int(round(alpha * 100))}_{kind}",
        "nfp_operational",
    )
    for alpha in NFP_FRACTIONS
    for kind in SECTION4_NFP_FIGURE_KINDS
]
SECTION4_NFP_ALPHA_BY_STEM = {
    f"fig_nfp{int(round(alpha * 100))}_{kind}": alpha
    for alpha in NFP_FRACTIONS
    for kind in SECTION4_NFP_FIGURE_KINDS
}
SECTION4_NFP_KIND_BY_STEM = {
    f"fig_nfp{int(round(alpha * 100))}_{kind}": kind
    for alpha in NFP_FRACTIONS
    for kind in SECTION4_NFP_FIGURE_KINDS
}
FEATURE_SHORT_LABELS = {
    "set1_dQn_m": "FS 1",
    "set2_early_soh": "FS 2",
    "set3_features": "FS 3",
    "set4_features_metadata": "FS 4",
    "set5_features_metadata_corr95": "FS 5",
}
MODEL_SHORT_LABELS = {"GPR": "GPR", "RF": "RF", "AE_ENet": "AE-ENet"}
ACQUISITION_DISPLAY_LABELS = {
    "random_selection": "Random",
    "diversity_oneshot": "Diversity OS",
    "diversity_iterative": "Diversity IT",
    "coverage": "Coverage",
    "exploration": "Exploration",
    "exploitation": "Exploitation",
    "hybrid": "Hybrid",
}


FIGURE_SPECS = [
    FigureSpec(1, "fig_full_pool_primary_vs_deep_violin", "full_pool_all_models"),
    FigureSpec(
        1,
        "fig_full_pool_primary_vs_deep_violin_trial_level",
        "full_pool_all_models_trial",
    ),
    FigureSpec(1, "fig_attainable_gain_sign_fraction_vertical", "attainable_gain_sign_counts"),
    FigureSpec(1, "fig_initialisation_gain_sign_fraction_vertical", "initialisation_gain_sign_counts"),
    FigureSpec(1, "fig_full_pool_performance_ring_heatmap", "full_pool_primary_config"),
    FigureSpec(1, "fig_attainable_full_pool_gain_ring_heatmap", "attainable_gain_config"),
    FigureSpec(1, "fig_initialisation_gain_ring_heatmap", "initialisation_gain_config"),
    FigureSpec(1, "fig_full_pool_term_contribution", "section1_full_pool_anova_term"),
    FigureSpec(1, "fig_attainable_full_pool_gain_term_contribution", "section1_attainable_anova_term"),
    FigureSpec(1, "fig_initial_prediction_performance_term_contribution", "section1_initial_prediction_anova_term"),
    FigureSpec(1, "fig_full_pool_grouped_contribution", "section1_full_pool_anova_grouped"),
    FigureSpec(1, "fig_attainable_full_pool_gain_grouped_contribution", "section1_attainable_anova_grouped"),
    FigureSpec(1, "fig_initial_prediction_performance_grouped_contribution", "section1_initial_prediction_anova_grouped"),
    FigureSpec(1, "fig_legend_dataset_codes", "legend_dataset_codes"),
    FigureSpec(1, "fig_legend_feature_codes", "legend_feature_codes"),
    FigureSpec(
        1,
        "fig_legend_feature_order_clockwise",
        "legend_feature_order_clockwise",
    ),
    FigureSpec(
        1,
        "fig_legend_colorbar_full_pool_performance",
        "full_pool_primary_config",
    ),
    FigureSpec(
        1,
        "fig_legend_colorbar_attainable_full_pool_gain",
        "attainable_gain_config",
    ),
    FigureSpec(
        1,
        "fig_legend_colorbar_initialisation_gain",
        "initialisation_gain_config",
    ),
    FigureSpec(1, "fig_legend_anova_factors_3", "legend_anova_factors_3"),
    FigureSpec(1, "fig_legend_anova_factors_4", "legend_anova_factors_4"),
    FigureSpec(1, "fig_legend_term_classes", "legend_term_classes"),
    FigureSpec(2, "fig_active_learning_gain_term_anova", "active_gain_anova_term"),
    FigureSpec(2, "fig_active_learning_gain_grouped_contribution", "active_gain_anova_grouped"),
    FigureSpec(2, "fig_acquisition_rule_vs_attainable_full_pool_gain", "section2_moderation"),
    FigureSpec(2, "fig_median_active_learning_gain_feature_model", "section2_feature_model_median"),
    FigureSpec(2, "fig_active_learning_gain_by_feature_set", "early_gain_config"),
    FigureSpec(2, "fig_active_learning_gain_by_prediction_model", "early_gain_config"),
    *SECTION4_NFP_FIGURE_SPECS,
]


FACTOR_DISPLAY = {
    "Dataset": ("A", "Dataset"),
    "Feature": ("B", "Feature set"),
    "Model": ("C", "Prediction model"),
    "Initialisation": ("D", "Initialisation strategy"),
}
TERM_CODE_ORDER = [
    "A", "B", "C", "D", "A\u00d7B", "A\u00d7C", "A\u00d7D",
    "B\u00d7C", "B\u00d7D", "C\u00d7D", "Residual",
]
FACTOR_COLORS = {"A": "#3B6FB6", "B": "#4E9F62", "C": "#D97947", "D": "#8A68A8"}
TERM_CLASS_COLORS = {
    "Main effect": "#4776A8",
    "Pairwise interaction": "#D39A45",
    "Residual": "#9A9A9A",
}
DATASET_DISPLAY = {
    "MIT": "MIT",
    "ISU_ILCC": "ISU-ILCC",
    "LSD_Primary": "LSD-primary",
    "LSD_Second": "LSD-second",
    "HUST": "HUST",
    "KIT": "KIT",
    "Formation": "Formation",
    "TRI_Tesla": "TRI-Tesla",
}


def _trial_mean(frame: pd.DataFrame, group: list[str], value: str, output: str) -> pd.DataFrame:
    return (
        frame.groupby(group, dropna=False)
        .agg(**{output: (value, "mean"), "n_trials": ("trial", "nunique")})
        .reset_index()
    )


def _sign_counts(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    positive = int(frame[value].gt(0).sum())
    non_positive = int(frame[value].le(0).sum())
    total = positive + non_positive
    return pd.DataFrame(
        {
            "sign": ["Positive (>0)", "Non-positive (<=0)"],
            "count": [positive, non_positive],
            "fraction": [positive / total if total else float("nan"), non_positive / total if total else float("nan")],
        }
    )


def prepare_contribution_sources(
    term: pd.DataFrame,
    grouped: pd.DataFrame,
    response_code: str,
    response_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped_out = grouped.copy()
    grouped_out["factor_code"] = grouped_out["factor"].map(
        lambda value: FACTOR_DISPLAY[value][0]
    )
    grouped_out["factor"] = grouped_out["factor"].map(
        lambda value: FACTOR_DISPLAY[value][1]
    )
    grouped_out["factor_order"] = grouped_out["factor_code"].map(
        {value: index for index, value in enumerate("ABCD")}
    )
    grouped_out["variance_explained_pct"] = grouped_out["grouped_contribution_pp"]
    grouped_out["response_code"] = response_code
    grouped_out["response_label"] = response_label
    grouped_out = grouped_out.sort_values("factor_order").reset_index(drop=True)

    label_to_code = {
        label: code for label, (code, _) in FACTOR_DISPLAY.items()
    }
    term_out = term.copy()
    term_out["term_code"] = term_out["term"].map(
        lambda value: "Residual" if value == "Residual" else "\u00d7".join(
            label_to_code[item] for item in value.split(" x ")
        )
    )
    term_out["term_class"] = term_out["term_code"].map(
        lambda value: "Residual" if value == "Residual" else (
            "Pairwise interaction" if "\u00d7" in value else "Main effect"
        )
    )
    term_out["term_order"] = term_out["term_code"].map(
        {value: index for index, value in enumerate(TERM_CODE_ORDER)}
    )
    term_out["response_code"] = response_code
    term_out["response_label"] = response_label
    term_out = term_out.sort_values("term_order").reset_index(drop=True)
    return term_out, grouped_out


def build_section1_legend_tables() -> dict[str, pd.DataFrame]:
    dataset = pd.DataFrame(
        {
            "code": [f"D{index}" for index in range(1, len(DATASET_ORDER) + 1)],
            "label": [DATASET_DISPLAY[value] for value in DATASET_ORDER],
            "order": range(len(DATASET_ORDER)),
        }
    )
    feature = pd.DataFrame(
        {
            "code": [f"F{index}" for index in range(1, len(FEATURE_ORDER) + 1)],
            "label": FEATURE_ORDER,
            "order": range(len(FEATURE_ORDER)),
        }
    )
    feature_order_clockwise = pd.DataFrame(
        {
            "order": range(5),
            "code": [f"FS {index}" for index in range(1, 6)],
            "direction": ["clockwise"] * 5,
            "theta_start_deg": [140.0, 120.0, 100.0, 80.0, 60.0],
            "theta_end_deg": [120.0, 100.0, 80.0, 60.0, 40.0],
        }
    )
    factor_rows = [
        {"code": code, "label": label, "color": FACTOR_COLORS[code], "order": order}
        for order, (code, label) in enumerate(
            [
                ("A", "Dataset"),
                ("B", "Feature set"),
                ("C", "Prediction model"),
                ("D", "Initialisation strategy"),
            ]
        )
    ]
    term_class = pd.DataFrame(
        [
            {"label": label, "color": color, "order": order}
            for order, (label, color) in enumerate(TERM_CLASS_COLORS.items())
        ]
    )
    return {
        "legend_dataset_codes": dataset,
        "legend_feature_codes": feature,
        "legend_feature_order_clockwise": feature_order_clockwise,
        "legend_anova_factors_3": pd.DataFrame(factor_rows[:3]),
        "legend_anova_factors_4": pd.DataFrame(factor_rows),
        "legend_term_classes": term_class,
    }


def derive_tables(inputs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    metrics = inputs["metrics"]
    selected = inputs["selected"]
    dl_metrics = inputs["dl_metrics"]

    full_trial = build_full_pool(metrics)
    cold = build_cold_start(metrics)
    attainable_trial = build_attainable_gain(cold, full_trial)
    initialisation_trial = build_initialisation_table(metrics, selected)
    initialisation_gain_trial = build_initialisation_gain(initialisation_trial)

    full_config = _trial_mean(full_trial, ["dataset", "feature_set", "model"], "full_pool_mape", "full_pool_mape")
    attainable_config = _trial_mean(
        attainable_trial, ["dataset", "feature_set", "model"], "attainable_gain_pp", "attainable_gain_pp"
    )
    initialisation_config = _trial_mean(
        initialisation_trial,
        ["dataset", "feature_set", "model", "initialisation"],
        "initial_mape",
        "initial_mape",
    )
    initialisation_gain_config = _trial_mean(
        initialisation_gain_trial,
        ["dataset", "feature_set", "model"],
        "initialisation_gain_pp",
        "initialisation_gain_pp",
    )

    primary_model_comparison = full_config.copy()
    primary_model_comparison["model_family"] = "Primary ML"
    dl_config = _trial_mean(
        dl_metrics,
        ["dataset", "feature_set", "model"],
        "full_pool_mape",
        "full_pool_mape",
    )
    dl_config["model_family"] = "Deep learning"
    full_all_models = pd.concat([primary_model_comparison, dl_config], ignore_index=True, sort=False)

    trial_columns = [
        "dataset",
        "feature_set",
        "model",
        "trial",
        "full_pool_mape",
    ]
    primary_trial_comparison = full_trial.loc[:, trial_columns].copy()
    primary_trial_comparison["model_family"] = "Primary ML"
    dl_trial_comparison = dl_metrics.loc[:, trial_columns].copy()
    dl_trial_comparison["model_family"] = "Deep learning"
    full_all_models_trial = pd.concat(
        [primary_trial_comparison, dl_trial_comparison],
        ignore_index=True,
        sort=False,
    )
    trial_keys = ["dataset", "feature_set", "model", "trial"]
    duplicated_trials = full_all_models_trial.duplicated(trial_keys, keep=False)
    if duplicated_trials.any():
        preview = (
            full_all_models_trial.loc[duplicated_trials, trial_keys]
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )
        raise ValueError(f"Duplicate full-pool model trials: {preview}")

    full_term, full_grouped = factorial_anova(
        full_config, "full_pool_mape", ["dataset", "feature_set", "model"]
    )
    full_term["response"] = "Full-pool MAPE"
    full_grouped["response"] = "Full-pool MAPE"
    section1_full_term, section1_full_grouped = prepare_contribution_sources(
        full_term,
        full_grouped,
        "full_pool",
        "Full-pool prediction performance",
    )
    attainable_term, attainable_grouped = factorial_anova(
        attainable_config, "attainable_gain_pp", ["dataset", "feature_set", "model"]
    )
    attainable_term["response"] = "Attainable gain"
    attainable_grouped["response"] = "Attainable gain"
    section1_attainable_term, section1_attainable_grouped = prepare_contribution_sources(
        attainable_term,
        attainable_grouped,
        "attainable_gain",
        "Attainable full-pool gain",
    )
    part1_term = pd.concat([full_term, attainable_term], ignore_index=True, sort=False)
    part1_grouped = pd.concat([full_grouped, attainable_grouped], ignore_index=True, sort=False)

    init_term, init_grouped = factorial_anova(
        initialisation_config,
        "initial_mape",
        ["dataset", "feature_set", "model", "initialisation"],
    )
    section1_init_term, section1_init_grouped = prepare_contribution_sources(
        init_term,
        init_grouped,
        "initial_prediction",
        "Initialisation prediction performance",
    )
    legend_tables = build_section1_legend_tables()

    pointwise = compute_pointwise_gain(metrics)
    overall = summarise_overall_gain(pointwise)
    early = summarise_early_gain(pointwise)
    overall_nonrandom = overall.loc[~overall["acquisition"].eq(RANDOM)].reset_index(drop=True)
    early_nonrandom = early.loc[~early["acquisition"].eq(RANDOM)].reset_index(drop=True)
    active_term, active_grouped = factorial_anova(
        early_nonrandom,
        "early_gain_pp",
        ["dataset", "feature_set", "model", "acquisition"],
    )
    section2_moderation = build_moderation_table(early_nonrandom, attainable_config)
    section2_feature_model_median = build_feature_model_median(early_nonrandom)
    early_rule_summary = (
        early_nonrandom.groupby("acquisition", dropna=False)
        .agg(
            mean_early_gain_pp=("early_gain_pp", "mean"),
            median_early_gain_pp=("early_gain_pp", "median"),
            q25_early_gain_pp=("early_gain_pp", lambda values: values.quantile(0.25)),
            q75_early_gain_pp=("early_gain_pp", lambda values: values.quantile(0.75)),
            positive_configuration_fraction=("early_gain_pp", lambda values: float((values > 0).mean())),
            n_configurations=("early_gain_pp", "size"),
        )
        .reset_index()
    )

    durations = build_duration_table(selected)
    nfp_operational = build_nfp_crossings(metrics, attainable_trial, durations, alphas=NFP_FRACTIONS)
    durations["duration_unit"] = durations["dataset"].map(DURATION_UNITS)
    nfp_operational["duration_unit"] = nfp_operational["dataset"].map(DURATION_UNITS)

    return {
        "full_pool_trial": full_trial,
        "full_pool_primary_config": full_config,
        "full_pool_all_models": full_all_models,
        "full_pool_all_models_trial": full_all_models_trial,
        "cold_start": cold,
        "attainable_gain_trial": attainable_trial,
        "attainable_gain_config": attainable_config,
        "attainable_gain_sign_counts": _sign_counts(attainable_trial, "attainable_gain_pp"),
        "initialisation_table": initialisation_trial,
        "initialisation_config": initialisation_config,
        "initialisation_gain_trial": initialisation_gain_trial,
        "initialisation_gain_config": initialisation_gain_config,
        "initialisation_gain_sign_counts": _sign_counts(initialisation_gain_trial, "initialisation_gain_pp"),
        "part1_term_anova": part1_term,
        "part1_grouped_anova": part1_grouped,
        "initialisation_anova_term": init_term,
        "initialisation_anova_grouped": init_grouped,
        "section1_full_pool_anova_term": section1_full_term,
        "section1_full_pool_anova_grouped": section1_full_grouped,
        "section1_attainable_anova_term": section1_attainable_term,
        "section1_attainable_anova_grouped": section1_attainable_grouped,
        "section1_initial_prediction_anova_term": section1_init_term,
        "section1_initial_prediction_anova_grouped": section1_init_grouped,
        "section1_layout_manifest": build_layout_manifest(),
        **legend_tables,
        "active_gain_pointwise": pointwise,
        "overall_gain_config": overall_nonrandom,
        "early_gain_config": early_nonrandom,
        "early_gain_positive_fraction": positive_configuration_fraction(early_nonrandom, "early_gain_pp"),
        "early_gain_rule_summary": early_rule_summary,
        "active_gain_anova_term": active_term,
        "active_gain_anova_grouped": active_grouped,
        "section2_moderation": section2_moderation,
        "section2_feature_model_median": section2_feature_model_median,
        "duration_by_budget": durations,
        "nfp_operational": nfp_operational,
    }


def export_plot_source(frame: pd.DataFrame, section_dir: Path, stem: str) -> Path:
    path = section_dir / "source_data" / f"{stem}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _prepare_section4_nfp_source(frame: pd.DataFrame, stem: str) -> pd.DataFrame:
    alpha = SECTION4_NFP_ALPHA_BY_STEM[stem]
    out = frame.loc[frame["alpha"].eq(alpha) & frame["positive_attainable_gain"].eq(True)].copy()
    out["nfp_percent"] = int(round(alpha * 100))
    out["feature_model_label"] = (
        out["feature_set"].map(FEATURE_SHORT_LABELS).fillna(out["feature_set"].astype(str))
        + "+"
        + out["model"].map(MODEL_SHORT_LABELS).fillna(out["model"].astype(str))
    )
    out["acquisition_label"] = out["acquisition"].map(ACQUISITION_DISPLAY_LABELS).fillna(
        out["acquisition"].astype(str)
    )
    out["reached"] = out["status"].eq("crossed")
    out["required_pool_fraction_pct"] = out["operational_pool_fraction"].where(out["reached"]) * 100.0
    out["required_full_life_tests"] = out["operational_labelled_n"].where(out["reached"])
    out["required_duration"] = out["operational_duration"].where(out["reached"])
    return out


def _source_for_spec(tables: dict[str, pd.DataFrame], spec: FigureSpec) -> pd.DataFrame:
    frame = tables[spec.source_table].copy()
    if spec.stem in SECTION4_NFP_ALPHA_BY_STEM:
        return _prepare_section4_nfp_source(frame, spec.stem)
    return frame


def export_all_plot_sources(
    tables: dict[str, pd.DataFrame],
    paths: AnalysisPaths,
    specs: list[FigureSpec] | tuple[FigureSpec, ...] = tuple(FIGURE_SPECS),
) -> list[Path]:
    outputs = []
    for spec in specs:
        section_dir = paths.output_root / SECTION_NAMES[spec.section]
        outputs.append(export_plot_source(_source_for_spec(tables, spec), section_dir, spec.stem))
    if any(spec.section == 1 for spec in specs):
        section1_source = paths.output_root / SECTION_NAMES[1] / "source_data"
        section1_source.mkdir(parents=True, exist_ok=True)
        tables["section1_layout_manifest"].to_csv(
            section1_source / "layout_manifest.csv",
            index=False,
        )
    return outputs


def write_manifest(
    output_root: Path,
    table_inventory: dict[str, object],
    figure_inventory: list[object],
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "analysis_version": "v4",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "definitions": {
            "full_pool_performance": "Hold-out MAPE under complete candidate-pool supervision",
            "attainable_gain": "E_cold - E_full",
            "initialisation_performance_gain": "E_cold - E_hot",
            "active_learning_performance_gain": "E_random - E_non-random",
            "nfp_fractions": NFP_FRACTIONS,
            "main_nfp_fraction": 0.95,
        },
        "tables": table_inventory,
        "figures": figure_inventory,
        "figure_specs": [asdict(spec) for spec in FIGURE_SPECS],
    }
    path = output_root / "analysis_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return path
