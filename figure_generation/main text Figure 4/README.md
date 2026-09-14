# Main text Figure 4

This module renders relative area-under-the-learning-curve (AULC) profiles, acquisition-rule distributions, factor dependence, heatmaps, robustness analyses and the early-gain trade-off.

## Reproduction

```bash
python "figure_generation/main text Figure 4/scripts/plot.py"
```

The script reads the packaged learning-curve, configuration and robustness tables directly. It does not access the excluded training-result directories. The composite and standalone 600-dpi PNG assets are written to `outputs/`.
