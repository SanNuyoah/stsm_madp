# Analysis Entry Points

Use the shared result manager:

```bash
python3 stsm_madp/scripts/analysis/results_manager.py collect
python3 stsm_madp/scripts/analysis/results_manager.py validate --run-id <RUN_ID>
```

Implementation lives in:

- `scripts/analysis/results_manager.py`
- `scripts/analysis/train_adp_critic.py`

Historical collectors remain archived under `scripts/archive/legacy/` and are
imported by the analysis modules only when explicitly needed. Top-level
analysis scripts are intentionally not kept.
