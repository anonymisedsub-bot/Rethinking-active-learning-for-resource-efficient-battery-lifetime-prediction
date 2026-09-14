# Dataset provenance and reuse conditions

Source records were last checked on 2 September 2026. The processed `.mat` files preserve the scientific content required by this repository but use a common variable layout. Cite both the source dataset and its associated article where applicable.

| Processed file(s) | Public source record | Associated article | Upstream licence or reuse condition |
|---|---|---|---|
| `data_MIT.mat` | [MATR dataset 1](https://data.matr.io/1/) | Severson *et al.*, *Nature Energy* (2019), [doi:10.1038/s41560-019-0356-8](https://doi.org/10.1038/s41560-019-0356-8) | CC BY 4.0, as stated by the source record. |
| `data_ISU_ILCC.mat` | Thelen *et al.*, ISU-ILCC Battery Aging Dataset, version 2, [doi:10.25380/iastate.22582234.v2](https://doi.org/10.25380/iastate.22582234.v2) | Thelen *et al.*, *Cell Reports Physical Science* (2024), [doi:10.1016/j.xcrp.2024.101891](https://doi.org/10.1016/j.xcrp.2024.101891) | CC BY 4.0, as stated by the source record. |
| `data_LSD_primary.mat`; `data_LSD_second.mat` | Wang *et al.*, Data for “Deep sorting of reused batteries for enabling long-term consistency grouping with unknown prior conditions”, [doi:10.5281/zenodo.14859405](https://doi.org/10.5281/zenodo.14859405) | Wang *et al.*, *Cell Reports Physical Science* (2025), [doi:10.1016/j.xcrp.2025.102657](https://doi.org/10.1016/j.xcrp.2025.102657) | CC BY 4.0, as stated in the cited Zenodo record. |
| `data_Formation.mat` | [MATR dataset 8](https://data.matr.io/8/) | “Data-driven analysis of battery formation reveals the role of electrode utilization in extending cycle life”, *Joule* (2024) | CC BY-NC 4.0, as stated by the source record. Commercial reuse is not granted by this repository. |
| `data_HUST.mat` | Yuan, Ma and Xu, “The Dataset for: Real-time personalized health status prediction of lithium-ion batteries using deep transfer learning”, version 2, [doi:10.17632/nsc7hnsg4s.2](https://doi.org/10.17632/nsc7hnsg4s.2) | Yuan, Ma and Xu, *Energy & Environmental Science* (2022), [doi:10.1039/D2EE01676A](https://doi.org/10.1039/D2EE01676A) | CC BY 4.0, as stated by the source record. |
| `data_KIT.mat` | Luh *et al.*, Battery aging dataset based on electric vehicle operation, [doi:10.35097/1947](https://doi.org/10.35097/1947) | Luh *et al.*, *Scientific Data* (2024), [doi:10.1038/s41597-024-03831-x](https://doi.org/10.1038/s41597-024-03831-x) | CC BY 4.0, as stated by the source record. |
| `data_TRI_Tesla.mat` | [MATR dataset 11: TRI Aging Matrix](https://data.matr.io/11/) | van Vlijmen *et al.*, *Energy & Environmental Science* (2025) | CC BY-NC 4.0, as stated by the source record. Commercial reuse is not granted by this repository. |

## Processing scope

- Raw source files were harmonised into cell-aligned numeric arrays.
- Early-life curve representations, engineered predictors and protocol metadata were retained.
- The stored sample counts reflect cells with aligned lifetime labels and the variables needed by the benchmark.
- Dataset-specific filters used during modelling are defined in each training module's `config.py`; they do not alter the distributed `.mat` files.

No ownership claim is made over third-party source data. The repository-level data notice does not supersede any upstream licence, access condition or citation requirement.
