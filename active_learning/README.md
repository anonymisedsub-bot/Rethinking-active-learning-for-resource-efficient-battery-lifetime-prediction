# Active-learning framework

This module benchmarks seven acquisition strategies with Gaussian-process regression, random forests and anchored Elastic Net ensembles. It uses lifetime-stratified protocol hold-out evaluation, deterministic trial seeds and five early-life feature representations.

Training results are not distributed in this release. New outputs are written to `active_learning/outputs_AL/` by default.

## Data resolution

`ExperimentConfig` resolves the dataset directory as `<repository>/data` from `config.py`; the current working directory is not used. After cloning, validate the Git LFS payload:

```bash
python active_learning/run_active_learning.py --check-data
```

## Minimal run

From the repository root:

```bash
python active_learning/run_active_learning.py \
  --datasets MIT \
  --feature-sets set3_features \
  --trials 1
```

Omitting `--datasets`, `--feature-sets` and `--trials` runs the complete configured benchmark. Use `--parallel-jobs N` only after confirming memory requirements with `benchmark_parallel_jobs.py`.

## Feature sets

| Identifier | Input |
|---|---|
| `set1_dQn_m` | Early differential capacity curve |
| `set2_early_soh` | Early SOH trajectory |
| `set3_features` | Engineered early-life features |
| `set4_features_metadata` | Engineered features and protocol metadata |
| `set5_features_metadata_corr95` | Set 4 with training-only correlation filtering at \|r\| > 0.95 |

The full experimental configuration, dataset filters and randomisation settings are defined in `config.py`. Run `python active_learning/run_active_learning.py --help` for command-line options.

