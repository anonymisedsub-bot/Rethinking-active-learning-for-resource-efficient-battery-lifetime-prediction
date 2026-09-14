# Deep-learning baseline

This module evaluates CNN, TabNet, variational-autoencoder and multilayer-perceptron regressors using the same datasets, feature representations, trial seeds and protocol hold-out splits as the active-learning framework.

Training results are not distributed in this release. New outputs are written to `deep_learning_baseline/outputs_DL_benchmark/` by default.

## Data resolution

`ExperimentConfig` resolves the dataset directory as `<repository>/data` from `config.py`; the current working directory is not used. After cloning, validate the Git LFS payload:

```bash
python deep_learning_baseline/run_deep_learning_baseline.py --check-data
```

## Minimal run

From the repository root:

```bash
python deep_learning_baseline/run_deep_learning_baseline.py \
  --datasets MIT \
  --feature-sets set3_features \
  --trials 1 \
  --models MLP \
  --epochs 20
```

Omitting the selection flags runs the complete configured benchmark. PyTorch is required; TabNet additionally requires `pytorch-tabnet`. GPU-specific PyTorch builds should be installed from the official PyTorch package index before installing the remaining requirements.

The full optimisation, architecture and reproducibility settings are defined in `config.py` and `dl_benchmark.py`. Run `python deep_learning_baseline/run_deep_learning_baseline.py --help` for command-line options.

