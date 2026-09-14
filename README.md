# Active-learning battery lifetime prediction: reproducibility package

This repository contains the processed datasets, active-learning framework, deep-learning baselines and manuscript figure-generation modules for **“Navigating Battery Ageing Space with Active Learning for Test-Efficient Lifetime Prediction from Early-Life Data.”**

The release is modular: both training frameworks read the peer `data/` directory automatically, whereas every figure module can be rerun directly from its packaged CSV source tables. Complete active-learning and deep-learning result directories are not included in this release.

## Repository structure

```text
.
├── active_learning/             # Active-learning benchmark and command-line entry point
├── deep_learning_baseline/      # CNN, TabNet, DeepVAE and MLP baselines
├── data/                        # Eight processed MATLAB datasets and provenance
├── figure_generation/           # Eight independent manuscript-figure modules
│   ├── common/                  # Shared styles, layouts and statistical helpers
│   ├── main text Figure 2/
│   ├── main text Figure 3/
│   ├── main text Figure 4/
│   ├── main text Figure 5/
│   ├── main text Figure 6/
│   ├── Extended data Figure 4-7/
│   ├── SI Note 11 figures (Active-learning region determination)/
│   └── SI Note 13 figures (3D convergence profiles)/
├── tests/                       # Release-layout and data-resolution checks
├── tools/                       # Deterministic source-manifest utility
├── CITATION.cff
├── LICENSE                      # Software licence
├── LICENSE-DATA                 # Data and figure reuse notice
└── requirements.txt
```

## Installation

Python 3.12 was used for release verification. The two largest `.mat` files exceed GitHub's ordinary file-size limit and are tracked with Git LFS.

```bash
git lfs install
git lfs pull
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a CUDA-enabled deep-learning run, install the appropriate PyTorch build from the official PyTorch package index before installing the remaining requirements.

## Data validation

```bash
python active_learning/run_active_learning.py --check-data
python deep_learning_baseline/run_deep_learning_baseline.py --check-data
```

Expected SHA-256 digests, variable definitions and source links are provided in `data/`.

## Training

Minimal active-learning run:

```bash
python active_learning/run_active_learning.py --datasets MIT --feature-sets set3_features --trials 1
```

Minimal deep-learning run:

```bash
python deep_learning_baseline/run_deep_learning_baseline.py --datasets MIT --feature-sets set3_features --trials 1 --models MLP --epochs 20
```

Omit the selection arguments to run the complete configured benchmark. Training outputs are created locally and ignored by Git.

## Figure reproduction

Each figure directory contains `README.md`, `scripts/plot.py`, `source_data/manifest.csv` and curated PNG outputs. For example:

```bash
python "figure_generation/main text Figure 2/scripts/plot.py"
python "figure_generation/Extended data Figure 4-7/scripts/plot.py"
```

See `figure_generation/README.md` for the complete manuscript mapping.

## Release checks

```bash
python -m pytest tests -q
python tools/build_source_manifests.py
```

The second command refreshes figure-source hashes after an authorised change to a source table.

## Licensing and citation

Source code is released under the MIT License. Original figure source tables and rendered figures are covered by the terms in [LICENSE-DATA](LICENSE-DATA). Third-party-derived datasets retain their upstream licences and attribution requirements; see [data/DATASET_SOURCES.md](data/DATASET_SOURCES.md). Citation metadata for this release are provided in [CITATION.cff](CITATION.cff).
