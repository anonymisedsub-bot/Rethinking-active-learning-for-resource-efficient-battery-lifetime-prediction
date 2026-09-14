# Main text Figure 2

This module renders the full-pool prediction benchmark, attainable headroom, initialisation effects, variance contributions, ring heatmaps and associated legends.

## Reproduction

```bash
python "figure_generation/main text Figure 2/scripts/plot.py"
```

The script reads 22 same-stem analytical or legend tables from `source_data/`; `layout_manifest.csv` records the fixed panel geometry. Outputs are written to `outputs/` as 600-dpi PNG files. The source tables are analysis summaries derived from the complete benchmark; raw training-result directories are not required for rendering and are not distributed in this release.

