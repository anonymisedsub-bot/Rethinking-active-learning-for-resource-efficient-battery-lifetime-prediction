# Figure-generation modules

Each manuscript figure is distributed as an independent rendering module with three components:

- `source_data/`: the numerical values used by the plotting functions;
- `scripts/plot.py`: the rendering entry point;
- `outputs/`: the curated 600-dpi PNG renderings.

The source tables are sufficient to regenerate the figures without the excluded active-learning and deep-learning result directories. Shared plotting functions are in `common/`. Run each entry point from the repository root, for example:

```bash
python "figure_generation/main text Figure 2/scripts/plot.py"
```

`manifest.csv` in each `source_data/` directory reports the SHA-256 digest and dimensions of every distributed table. Filenames use descriptive stems; obsolete standalone panel or figure numbers have been removed.

| Module | Manuscript item | Source tables |
|---|---|---:|
| `main text Figure 2` | Full-pool headroom and initialisation effects | 23 |
| `main text Figure 3` | Drivers of active-learning gain | 7 |
| `main text Figure 4` | Relative ALC profiles | 9 |
| `main text Figure 5` | Convergence and selection profiles | 8 |
| `main text Figure 6` | NFP performance-indicator summary | 2 |
| `Extended data Figure 4-7` | Sample and time efficiency | 12 |
| `SI Note 11 figures (Active-learning region determination)` | Early active-learning region | 1 |
| `SI Note 13 figures (3D convergence profiles)` | GPR early-acquisition profiles | 5 |
