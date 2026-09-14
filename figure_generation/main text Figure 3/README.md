# Main text Figure 3

This module renders the factorial drivers of active-learning gain, acquisition-rule moderation, feature/model effects and dataset-resolved grouped factor contributions.

## Reproduction

```bash
python "figure_generation/main text Figure 3/scripts/plot.py"
```

The seven CSV tables in `source_data/` are the complete plotting inputs. Outputs are written to `outputs/` as 600-dpi PNG files. Shared factor orders and layout rules are defined in `figure_generation/common/`.

