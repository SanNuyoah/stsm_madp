# Script Checks

Current lightweight checks:

```bash
python3 -m py_compile $(find stsm_madp -name '*.py')
bash -n stsm_madp/scripts/run_experiments.sh
bash -n stsm_madp/scripts/run_stress.sh
python3 stsm_madp/scripts/visualization/plot.py paper --run-id <RUN_ID>
python3 stsm_madp/scripts/analysis/results_manager.py validate --run-id <RUN_ID>
```

Repository tests live in `stsm_madp/test/`.
