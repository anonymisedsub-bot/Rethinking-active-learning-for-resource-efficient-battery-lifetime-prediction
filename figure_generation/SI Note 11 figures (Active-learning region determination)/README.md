# Supplementary Information Note 11 figures

This module visualises error reduction relative to random acquisition across the predefined early active-learning region for GPR, RF and AE-ENet.

## Reproduction

```bash
python "figure_generation/SI Note 11 figures (Active-learning region determination)/scripts/plot.py"
```

`source_data/error_reduction_vs_random_learning_curves.csv` contains the complete model-, dataset-, feature- and checkpoint-level plotting summary. Three 600-dpi PNG files are written to `outputs/`.

