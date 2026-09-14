from __future__ import annotations

import argparse
from pathlib import Path

from config import ExperimentConfig
from experiments import run_experiments


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the active-learning benchmark on the packaged datasets.")
    parser.add_argument("--datasets", nargs="+", help="Dataset identifiers; omit to run all eight datasets.")
    parser.add_argument("--feature-sets", nargs="+", help="Feature-set identifiers; omit to run all five sets.")
    parser.add_argument("--trials", nargs="+", type=int, help="Positive trial identifiers; omit to run trials 1-10.")
    parser.add_argument("--parallel-jobs", type=int, default=1, help="Concurrent dataset jobs (default: 1).")
    parser.add_argument("--output", type=Path, help="Output directory (default: active_learning/outputs_AL).")
    parser.add_argument("--save-figures", action="store_true", help="Generate exploratory training-output plots.")
    parser.add_argument("--check-data", action="store_true", help="Validate the eight packaged .mat files and exit.")
    return parser


def main() -> None:
    args = _parser().parse_args()
    cfg = ExperimentConfig(
        pipeline="active_learning",
        pool_holdout_split_mode="protocol_holdout",
        run_datasets=args.datasets,
        run_feature_sets=args.feature_sets,
        run_trials=tuple(args.trials) if args.trials else None,
        parallel_jobs=args.parallel_jobs,
        output_root=args.output.resolve() if args.output else Path(__file__).resolve().parent / "outputs_AL",
        save_figures=args.save_figures,
    )
    paths = cfg.validate_data_files()
    if args.check_data:
        print(f"[OK] Found {len(paths)} datasets in {cfg.data_dir}")
        return
    out_dir = run_experiments(cfg)
    print(f"[Done] Active-learning results saved to: {out_dir}")


if __name__ == "__main__":
    main()
