# Main text Figure 6

This module renders the NFP performance-indicator summary, including efficiency trade-offs, required pool fractions, full-life-test sensitivity, strategy wins and duration estimates.

## Reproduction

```bash
python "figure_generation/main text Figure 6/scripts/plot.py"
```

The two source tables provide configuration-level summaries and trial-level distributions. The script validates the fixed figure contract and writes the composite, standalone panels, legends and rendering metadata to `outputs/` as 600-dpi PNG assets.

