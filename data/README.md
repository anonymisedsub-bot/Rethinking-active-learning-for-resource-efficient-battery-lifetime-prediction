# Processed battery datasets

This directory contains the eight processed MATLAB files used by both training modules. The files are aligned, analysis-ready derivatives of the public datasets cited in [DATASET_SOURCES.md](DATASET_SOURCES.md). Upstream licences and attribution requirements remain applicable.

## Inventory

| File | Dataset identifier | Cells | Lifetime unit | `dQn_m_matrix` | `early_soh_degradation_matrix` | `features_matrix` | `features_metadata_matrix` | `protocols` |
|---|---|---:|---|---|---|---|---|---|
| `data_Formation.mat` | Formation | 182 | cycles | 182 × 100 | 182 × 21 | 182 × 17 | 182 × 21 | 182 × 4 |
| `data_HUST.mat` | HUST | 77 | cycles | 77 × 100 | 77 × 100 | 77 × 17 | 77 × 21 | 77 × 4 |
| `data_ISU_ILCC.mat` | ISU_ILCC | 241 | weeks | 241 × 100 | 241 × 4 | 241 × 11 | 241 × 14 | 241 × 3 |
| `data_KIT.mat` | KIT | 82 | equivalent full cycles (EFC) | 82 × 100 | 82 × 5 | 82 × 17 | 82 × 21 | 82 × 4 |
| `data_LSD_primary.mat` | LSD_Primary | 73 | cycles | 73 × 100 | 73 × 20 | 73 × 11 | 73 × 15 | 73 × 4 |
| `data_LSD_second.mat` | LSD_Second | 72 | cycles | 72 × 100 | 72 × 20 | 72 × 11 | 72 × 15 | 72 × 4 |
| `data_MIT.mat` | MIT | 124 | cycles | 124 × 100 | 124 × 100 | 124 × 12 | 124 × 15 | 124 × 3 |
| `data_TRI_Tesla.mat` | TRI_Tesla | 134 | equivalent full cycles (EFC) | 134 × 100 | 134 × 3 | 134 × 18 | 134 × 24 | 134 × 6 |

The dimensions above describe the stored arrays before dataset-specific lifetime filters are applied by the training code.

## Common variables

| Variable | Description |
|---|---|
| `lifetimes` | Regression target. The physical unit is dataset-specific and is listed above. |
| `dQn_m_matrix` | Fixed-grid early-life differential capacity-curve representation (feature set 1). |
| `early_soh_degradation_matrix` | Early state-of-health trajectory (feature set 2). |
| `features_matrix` | Engineered early-life descriptors (feature set 3). |
| `features_metadata_matrix` | Engineered descriptors concatenated with protocol metadata (feature sets 4 and 5). |
| `protocols` | Numeric cycling or formation conditions used for protocol-aware analyses. |
| `cell_names` | Source-aligned cell identifiers. |

Feature set 5 uses `features_metadata_matrix` and removes predictors with an absolute training-set Pearson correlation greater than 0.95. Filtering is fitted within each training split.

## Integrity and use

Verify the files before analysis:

```powershell
Get-FileHash data\*.mat -Algorithm SHA256
```

Expected digests are listed in [SHA256SUMS.txt](SHA256SUMS.txt). The repository tracks these files with Git LFS; run `git lfs pull` after cloning. Both `active_learning` and `deep_learning_baseline` resolve this directory automatically from their installed file locations.

## Provenance

The `.mat` files standardise public source records for the analyses in this repository. They do not replace the source records. Consult [DATASET_SOURCES.md](DATASET_SOURCES.md) for persistent identifiers, attribution and licence information.

