# Shared rendering code

This directory contains plotting styles, fixed factor orders, layout specifications and statistical helper functions used by the main-text and Extended Data figure modules. It is not a standalone analysis entry point. Public rendering commands are provided in each figure module's `scripts/plot.py`.

The constants in `config.py` point to the modular release layout. Paths to raw training outputs are retained only to support source-table regeneration by data owners; public figure rendering reads the packaged CSV tables directly.

