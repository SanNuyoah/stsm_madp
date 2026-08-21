# Scripts Layout

Current entry points:

- `run_experiment.py`: Python entry point for the main experiment runner.
- `run_experiments.sh`: shell implementation used by existing workflows.

Organized implementation directories:

- `visualization/`: unified result and topology diagnostics plotting.
- `analysis/`: result management and ADP training utilities.
- `tests/`: script-level check documentation and helpers.
- `legacy/`: archived compatibility modules and historical experiment scripts.
- `lib/`: shell helpers shared by runners.

Do not add new top-level plotting scripts. Extend `visualization/plot.py` or
`visualization/plot_topology.py`.
