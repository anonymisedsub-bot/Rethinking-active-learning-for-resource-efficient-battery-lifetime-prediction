# Main text Figure 5

This module renders dataset-specific convergence and selection profiles. Each dataset table stores configuration, embedding, MAPE-convergence, signed-error density and fraction-matching blocks in long format.

## Reproduction

```bash
python "figure_generation/main text Figure 5/scripts/plot.py"
```

The script rebuilds eight composite profiles and their standalone components exclusively from the eight CSV tables in `source_data/`. Outputs are written to `outputs/` at 600 dpi. `convergence.py` contains the specialised renderer.
