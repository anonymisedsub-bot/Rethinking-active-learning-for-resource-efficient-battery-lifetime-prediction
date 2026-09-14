from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path


def _comma_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _job_counts(max_jobs: int, job_list: str | None) -> list[int]:
    if job_list:
        values = sorted({int(item) for item in _comma_list(job_list)})
    else:
        values = []
        current = 1
        while current < max_jobs:
            values.append(current)
            current *= 2
        values.append(max_jobs)
    return [value for value in values if value >= 1]


def _free_memory_gb() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2:
                return float(parts[1]) / 1024.0 / 1024.0
    return None


def _set_inner_thread_env(inner_threads: int) -> None:
    value = str(max(1, int(inner_threads)))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, value)


def _run_probe(args: argparse.Namespace, jobs: int) -> dict[str, object]:
    from config import ExperimentConfig
    from experiments import run_experiments

    output_root = Path(args.output_root).resolve() / f"parallel_jobs_{jobs}"
    if output_root.exists() and not args.keep_existing_outputs:
        shutil.rmtree(output_root)

    total_trials = max(int(args.min_trials), int(args.trials_per_job) * int(jobs))
    cfg = ExperimentConfig(
        pipeline="active_learning",
        output_root=output_root,
        run_datasets=[args.dataset],
        run_feature_sets=[args.feature_set],
        run_trials=tuple(range(1, total_trials + 1)),
        regressors=_comma_list(args.models),
        active_learning_methods=_comma_list(args.methods),
        pool_holdout_split_mode=args.split_mode,
        target_transform=args.target_transform,
        parallel_jobs=int(jobs),
        save_predictions=bool(args.save_predictions),
        save_figures=False,
    )

    row: dict[str, object] = {
        "parallel_jobs": int(jobs),
        "total_trials": int(total_trials),
        "dataset": args.dataset,
        "feature_set": args.feature_set,
        "models": args.models,
        "methods": args.methods,
        "output_root": str(output_root),
        "free_memory_before_gb": _free_memory_gb(),
    }
    start = time.perf_counter()
    try:
        run_dir = run_experiments(cfg)
        row.update(
            {
                "success": True,
                "elapsed_sec": float(time.perf_counter() - start),
                "run_dir": str(run_dir),
                "error": "",
            }
        )
    except Exception as exc:  # pragma: no cover - this script records cloud failures.
        row.update(
            {
                "success": False,
                "elapsed_sec": float(time.perf_counter() - start),
                "run_dir": "",
                "error": repr(exc),
            }
        )
    row["free_memory_after_gb"] = _free_memory_gb()
    return row


def _recommend(rows: list[dict[str, object]], min_efficiency: float, min_free_memory_gb: float) -> int | None:
    successful = [row for row in rows if bool(row.get("success"))]
    if not successful:
        return None
    baseline = next((row for row in successful if int(row["parallel_jobs"]) == 1), successful[0])
    baseline_elapsed = float(baseline["elapsed_sec"])
    eligible = []
    for row in successful:
        jobs = int(row["parallel_jobs"])
        elapsed = max(float(row["elapsed_sec"]), 1e-9)
        efficiency = baseline_elapsed / elapsed / jobs
        row["speedup_vs_1"] = baseline_elapsed / elapsed
        row["parallel_efficiency"] = efficiency
        free_after = row.get("free_memory_after_gb")
        memory_ok = free_after is None or float(free_after) >= min_free_memory_gb
        if efficiency >= min_efficiency and memory_ok:
            eligible.append(row)
    if eligible:
        return max(int(row["parallel_jobs"]) for row in eligible)
    return max(int(row["parallel_jobs"]) for row in successful)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe suitable active-learning parallel_jobs on a cloud server.")
    parser.add_argument("--max-jobs", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--job-list", default=None, help="Comma-separated job counts, for example: 1,2,4,8.")
    parser.add_argument("--dataset", default="MIT")
    parser.add_argument("--feature-set", default="set3_features")
    parser.add_argument("--models", default="AE_ENet")
    parser.add_argument("--methods", default="random_selection")
    parser.add_argument("--min-trials", type=int, default=2)
    parser.add_argument("--trials-per-job", type=int, default=1)
    parser.add_argument("--split-mode", default="protocol_holdout")
    parser.add_argument("--target-transform", default="log")
    parser.add_argument("--inner-threads", type=int, default=1)
    parser.add_argument("--min-efficiency", type=float, default=0.45)
    parser.add_argument("--min-free-memory-gb", type=float, default=1.0)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--keep-existing-outputs", action="store_true")
    parser.add_argument(
        "--output-root",
        default=str(Path(__file__).resolve().parent / "parallel_job_benchmark_outputs"),
    )
    args = parser.parse_args()

    _set_inner_thread_env(args.inner_threads)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for jobs in _job_counts(args.max_jobs, args.job_list):
        print(f"[Probe] parallel_jobs={jobs}")
        row = _run_probe(args, jobs)
        rows.append(row)
        print(f"[Probe result] parallel_jobs={jobs} success={row['success']} elapsed_sec={row['elapsed_sec']:.2f}")

    recommended = _recommend(rows, args.min_efficiency, args.min_free_memory_gb)
    csv_path = output_root / "parallel_jobs_benchmark.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "recommended_parallel_jobs": recommended,
        "benchmark_csv": str(csv_path),
        "min_efficiency": args.min_efficiency,
        "min_free_memory_gb": args.min_free_memory_gb,
    }
    summary_path = output_root / "parallel_jobs_benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
