# STSM-MADP Results

Use `summary/all_metrics.csv` for aggregate analysis. Each experiment lives under `runs/<run_id>/<robot>/<variant>/`.

## Run Policy

Use `YYYYMMDD_R###` for run IDs. The ID is only an identifier; experiment meaning belongs in `summary/experiment_index.csv` and the run `README.md`.

`20260701_R001` is retained as wheelchair ADP stage evidence. `20260701_R002` is retained as Arm ADP-DLS integration evidence, but it is not a final ablation result because wheelchair STSM-ADP failed and arm risk metrics did not improve.

Create `R003` only after a real code, parameter, critic, or scenario change. Directory cleanup, plot movement, or a plain rerun of R002 should not create a new final candidate run.

Minimum R003 candidate requirements:

- Wheelchair STSM-ADP: `success_safe=1`, `stop_triggered=0`, and `final_dist_to_goal < 0.25`.
- Arm STSM-ADP: `arm_dls_adp_used=1`, `arm_qp_used=0`, nonzero ADP soft/alignment or v/dq delta fields.
- Arm risk: STSM-ADP improves at least one key metric versus STSM-noADP (`mean_phi_s`, `risk_exceed_pct`, `min_head_dist`, or `min_chest_dist`).
- Corridor ADP: ADP term is nonzero and recorded in total corridor cost; rank changes are recorded when present.

Only figures selected from a valid final candidate should be copied into `paper_figures/final/`.
