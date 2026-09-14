from __future__ import annotations

import argparse
from pathlib import Path

from config import ExperimentConfig
from dl_benchmark import run_dl_benchmark
from visualize import create_dl_benchmark_visualizations


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the deep-learning benchmark on the packaged datasets.")
    parser.add_argument("--datasets", nargs="+", help="Dataset identifiers; omit to run all eight datasets.")
    parser.add_argument("--feature-sets", nargs="+", help="Feature-set identifiers; omit to run all five sets.")
    parser.add_argument("--trials", nargs="+", type=int, help="Positive trial identifiers; omit to run trials 1-10.")
    parser.add_argument("--models", nargs="+", help="Model identifiers; omit to run CNN, TabNet, DeepVAE and MLP.")
    parser.add_argument("--epochs", type=int, default=200, help="Maximum training epochs (default: 200).")
    parser.add_argument("--output", type=Path, help="Output directory (default: deep_learning_baseline/outputs_DL_benchmark).")
    parser.add_argument("--save-figures", action="store_true", help="Generate exploratory training-output plots.")
    parser.add_argument("--check-data", action="store_true", help="Validate the eight packaged .mat files and exit.")
    return parser


def main() -> None:
    args = _parser().parse_args()
    cfg = ExperimentConfig(
        pipeline="dl_benchmark",
        pool_holdout_split_mode="protocol_holdout",
        run_datasets=args.datasets,
        run_feature_sets=args.feature_sets,
        run_trials=tuple(args.trials) if args.trials else None,
        dl_models=tuple(args.models) if args.models else ("CNN", "TabNet", "DeepVAE", "MLP"),
        dl_epochs=args.epochs,
        dl_output_root=args.output.resolve() if args.output else Path(__file__).resolve().parent / "outputs_DL_benchmark",
        save_figures=args.save_figures,
    )
    paths = cfg.validate_data_files()
    if args.check_data:
        print(f"[OK] Found {len(paths)} datasets in {cfg.data_dir}")
        return
    out_dir = run_dl_benchmark(cfg)
    if cfg.save_figures:
        create_dl_benchmark_visualizations(out_dir)
    print(f"[Done] Deep-learning benchmark results saved to: {out_dir}")


if __name__ == "__main__":
    main()
