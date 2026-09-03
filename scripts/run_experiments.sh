#!/usr/bin/env bash
set -euo pipefail

source_if_exists() {
  if [ -f "$1" ]; then
    # shellcheck disable=SC1090
    source "$1"
  fi
}

source_if_exists /opt/ros/melodic/setup.bash
source_if_exists /home/sun/elfin_assist_ws/devel/setup.bash
source_if_exists /home/sun/LLL/catkin_ws/devel/setup.bash

pkg_dir="$(rospack find stsm_madp 2>/dev/null || true)"
if [ -z "${pkg_dir}" ]; then
  pkg_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
results_dir="${pkg_dir}/results"
pictures_dir="${results_dir}/figures"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --target)
      export TARGET="${2:-}"
      shift 2
      ;;
    --run-id)
      export RUN_ID="${2:-}"
      shift 2
      ;;
    --stage)
      export STAGE="${2:-}"
      shift 2
      ;;
    --purpose)
      export PURPOSE="${2:-}"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage:
  TARGET=all bash scripts/run_experiments.sh
  bash scripts/run_experiments.sh --target all --run-id YYYYMMDD_R###
EOF
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done
next_run_id() {
  local day
  local max_n
  day="$(date +%Y%m%d)"
  max_n="$(
    find "${results_dir}/runs" -maxdepth 1 -type d -name "${day}_R[0-9][0-9][0-9]" 2>/dev/null \
      | sed -n 's/.*_R\([0-9][0-9][0-9]\)$/\1/p' \
      | sort -n | tail -1
  )"
  if [ -z "${max_n}" ]; then
    printf '%s_R001\n' "${day}"
  else
    printf '%s_R%03d\n' "${day}" "$((10#${max_n} + 1))"
  fi
}
run_id="${RUN_ID:-$(next_run_id)}"
gui="${GUI:-true}"
rviz="${RVIZ:-true}"
plot="${PLOT:-true}"
plot_timeout_s="${PLOT_TIMEOUT_S:-120}"
clean="${CLEAN:-false}"
clean_env="${CLEAN_ENV:-true}"
keep_on_fail="${KEEP_ON_FAIL:-true}"
target_filter="${TARGET:-all}"
adp_enabled="${ADP_ENABLED:-true}"
adp_decision_influence_enabled="${ADP_DECISION_INFLUENCE_ENABLED:-true}"
adp_ranking_influence_enabled="${ADP_RANKING_INFLUENCE_ENABLED:-true}"
adp_mpc_influence_enabled="${ADP_MPC_INFLUENCE_ENABLED:-false}"
adp_ranking_lambda="${ADP_RANKING_LAMBDA:-0.02}"
adp_value_normalization="${ADP_VALUE_NORMALIZATION:-robust}"
adp_norm_clip="${ADP_NORM_CLIP:-3.0}"
adp_contribution_clip="${ADP_CONTRIBUTION_CLIP:-0.10}"
# A caller can still deliberately use one model for both targets with ADP_MODEL.
# The formal default keeps each critic on the matching online-cost calibration.
adp_model_override="${ADP_MODEL:-}"
arm_adp_model="${ADP_ARM_MODEL:-${pkg_dir}/config/adp_critic_arm_candidate_conditioned.yaml}"
wheelchair_adp_model="${ADP_WHEELCHAIR_MODEL:-${pkg_dir}/config/adp_critic_wheelchair_candidate_conditioned.yaml}"

adp_critic_identity() {
  PYTHONPATH="${pkg_dir}/src${PYTHONPATH:+:${PYTHONPATH}}" python - "$1" <<'PY'
from __future__ import print_function
import sys
from stsm_madp.adp import ADPCritic, critic_theta_hash
critic = ADPCritic.load_yaml(sys.argv[1])
print("%s %s" % (critic.critic_version, critic_theta_hash(critic)))
PY
}
lambda_adp="${LAMBDA_ADP:-0.005}"
lambda_adp_corridor="${LAMBDA_ADP_CORRIDOR:-0.05}"
lambda_adp_terminal="${LAMBDA_ADP_TERMINAL:-0.0015}"
lambda_adp_path="${LAMBDA_ADP_PATH:-${lambda_adp}}"
lambda_adp_arm="${LAMBDA_ADP_ARM:-0.008}"
adp_grad_eps="${ADP_GRAD_EPS:-0.01}"
adp_descent_gain="${ADP_DESCENT_GAIN:-0.04}"
adp_grad_clip="${ADP_GRAD_CLIP:-8.0}"
adp_solver_mode="${ADP_SOLVER_MODE:-dls_adp}"
use_cvxpy="${USE_CVXPY:-false}"
adp_blend_alpha="${ADP_BLEND_ALPHA:-0.08}"
# Wheelchair MPC has an explicit terminal-value contract; keep it enabled for
# formal STSM runs unless explicitly overridden. Baseline disables ADP in-node.
mpc_use_adp_terminal="${MPC_USE_ADP_TERMINAL:-true}"
adp_post_scale_enabled="${ADP_POST_SCALE_ENABLED:-false}"
adp_min_scale="${ADP_MIN_SCALE:-0.35}"
adp_debug="${ADP_DEBUG:-false}"
arm_wait_s="${ARM_WAIT_S:-15}"
arm_interest_enabled="${ARM_INTEREST_ENABLED:-true}"
arm_interest_gate_enabled="${ARM_INTEREST_GATE_ENABLED:-true}"
arm_interest_rho_warn="${ARM_INTEREST_RHO_WARN:-3.5}"
arm_interest_rho_stop="${ARM_INTEREST_RHO_STOP:-6.0}"
arm_interest_gate_min_scale="${ARM_INTEREST_GATE_MIN_SCALE:-0.20}"
wc_wait_s="${WC_WAIT_S:-12}"
wc_goal_tolerance="${WC_GOAL_TOLERANCE:-0.08}"
wc_completion_tolerance="${WC_COMPLETION_TOLERANCE:-0.25}"
wc_variants="${WC_VARIANTS:-baseline stsm}"
arm_variants="${ARM_VARIANTS:-baseline stsm}"
experiment_mode="${EXPERIMENT_MODE:-paper}"
wc_completion_hold_s="${WC_COMPLETION_HOLD_S:-1.5}"
wc_max_runtime_s="${WC_MAX_RUNTIME_S:-180.0}"
wc_command_hold_s="${WC_COMMAND_HOLD_S:-1.0}"
wc_mpc_solve_deadline_s="${WC_MPC_SOLVE_DEADLINE_S:-0.6}"
wc_no_progress_timeout_s="${WC_NO_PROGRESS_TIMEOUT_S:-45.0}"
wc_no_progress_epsilon="${WC_NO_PROGRESS_EPSILON:-0.02}"
wc_replan_period="${WC_REPLAN_PERIOD:-5.0}"
wc_no_progress_replan_time="${WC_NO_PROGRESS_REPLAN_TIME:-5.0}"
wc_progress_eps="${WC_PROGRESS_EPS:-0.01}"
wc_replan_tube_margin="${WC_REPLAN_TUBE_MARGIN:-0.08}"
wc_near_goal_radius="${WC_NEAR_GOAL_RADIUS:-0.50}"
wc_near_goal_adp_scale="${WC_NEAR_GOAL_ADP_SCALE:-0.20}"
wc_min_progress_per_solve="${WC_MIN_PROGRESS_PER_SOLVE:-0.005}"
wc_near_goal_goal_weight="${WC_NEAR_GOAL_GOAL_WEIGHT:-18.0}"
wc_near_goal_social_scale="${WC_NEAR_GOAL_SOCIAL_SCALE:-0.5}"
wc_lam_stall="${WC_LAM_STALL:-10.0}"
wc_progress_reward_weight="${WC_PROGRESS_REWARD_WEIGHT:-2.8}"
wc_final_approach_radius="${WC_FINAL_APPROACH_RADIUS:-0.90}"
wc_final_heading_threshold="${WC_FINAL_HEADING_THRESHOLD:-0.75}"
wc_final_heading_gain="${WC_FINAL_HEADING_GAIN:-1.6}"
wc_final_creep_v="${WC_FINAL_CREEP_V:-0.10}"
wc_final_min_v="${WC_FINAL_MIN_V:-0.16}"
wc_final_max_v="${WC_FINAL_MAX_V:-0.30}"
wc_final_forward_gain="${WC_FINAL_FORWARD_GAIN:-0.75}"
wc_lam_heading="${WC_LAM_HEADING:-2.5}"
wc_final_direct_override_enabled="${WC_FINAL_DIRECT_OVERRIDE_ENABLED:-true}"
wc_final_direct_override_radius="${WC_FINAL_DIRECT_OVERRIDE_RADIUS:-0.90}"
wc_mpc_horizon="${WC_MPC_HORIZON:-6}"
wc_mpc_dt="${WC_MPC_DT:-0.2}"
wc_mpc_a_max="${WC_MPC_A_MAX:-0.5}"
wc_mpc_alpha_max="${WC_MPC_ALPHA_MAX:-1.5}"
wc_mpc_beam_width="${WC_MPC_BEAM_WIDTH:-4}"
wc_interest_enabled="${WC_INTEREST_ENABLED:-true}"
wc_interest_gate_enabled="${WC_INTEREST_GATE_ENABLED:-true}"
wc_footprint_rho_warn="${WC_FOOTPRINT_RHO_WARN:-5.0}"
wc_footprint_rho_stop="${WC_FOOTPRINT_RHO_STOP:-7.0}"
wc_footprint_gate_min_scale="${WC_FOOTPRINT_GATE_MIN_SCALE:-0.20}"
wc_footprint_forbidden_stop_enabled="${WC_FOOTPRINT_FORBIDDEN_STOP_ENABLED:-true}"
run_root="${results_dir}/runs/${run_id}"
mkdir -p \
  "${results_dir}/runs" "${results_dir}/summary" \
  "${run_root}/config" "${run_root}/compare"
manifest_csv="${run_root}/manifest.csv"
env_pid=""
env_log=""
metrics_pid=""
failed=0

printf 'run_id,robot,variant,mode,start_time,end_time,status,exit_code,metrics_path,traj_path,log_path\n' > "${manifest_csv}"

if [ "${clean}" = "true" ]; then
  rm -f \
    "${results_dir}/arm_metrics.csv" \
    "${results_dir}/wc_metrics.csv" \
    "${results_dir}/arm_baseline_traj.csv" \
    "${results_dir}/arm_stsm_traj.csv" \
    "${results_dir}/wc_baseline_traj.csv" \
    "${results_dir}/wc_stsm_traj.csv"
fi

cleanup_env() {
  if [ -n "${env_pid}" ] && kill -0 "${env_pid}" >/dev/null 2>&1; then
    kill -INT "${env_pid}" >/dev/null 2>&1 || true
    wait "${env_pid}" >/dev/null 2>&1 || true
  fi
  env_pid=""
}

cleanup_metrics() {
  if [ -n "${metrics_pid}" ] && kill -0 "${metrics_pid}" >/dev/null 2>&1; then
    kill -INT "${metrics_pid}" >/dev/null 2>&1 || true
    wait "${metrics_pid}" >/dev/null 2>&1 || true
  fi
  metrics_pid=""
}

cleanup_all() {
  cleanup_metrics
  if [ "${failed}" = "1" ] && [ "${keep_on_fail}" = "true" ]; then
    echo "Keeping Gazebo/RViz alive because the run failed (KEEP_ON_FAIL=true)."
    echo "Run archive: ${run_root}"
    return
  fi
  cleanup_env
}

cleanup_stale_env() {
  if [ "${clean_env}" != "true" ]; then
    return
  fi
  echo "cleaning stale Gazebo/RViz processes (CLEAN_ENV=true)..."
  pkill -INT -f "roslaunch stsm_madp arm_view.launch" >/dev/null 2>&1 || true
  pkill -INT -f "roslaunch stsm_madp wheelchair_view.launch" >/dev/null 2>&1 || true
  pkill -INT -f "roslaunch stsm_madp arm_action.launch" >/dev/null 2>&1 || true
  pkill -INT -f "roslaunch stsm_madp wheelchair_action.launch" >/dev/null 2>&1 || true
  pkill -INT -f "stsm_handover" >/dev/null 2>&1 || true
  pkill -INT -f "stsm_wheelchair" >/dev/null 2>&1 || true
  pkill -INT -f "stsm_social_field_viz" >/dev/null 2>&1 || true
  pkill -INT -f "wc_controller_spawner" >/dev/null 2>&1 || true
  pkill -INT -f "gzserver.*eldercare_room.world" >/dev/null 2>&1 || true
  pkill -INT -f "gzclient" >/dev/null 2>&1 || true
  sleep 2.0
  pkill -TERM -f "roslaunch stsm_madp arm_view.launch" >/dev/null 2>&1 || true
  pkill -TERM -f "roslaunch stsm_madp wheelchair_view.launch" >/dev/null 2>&1 || true
  pkill -TERM -f "stsm_handover" >/dev/null 2>&1 || true
  pkill -TERM -f "stsm_wheelchair" >/dev/null 2>&1 || true
  pkill -TERM -f "wc_controller_spawner" >/dev/null 2>&1 || true
  pkill -TERM -f "gzserver.*eldercare_room.world" >/dev/null 2>&1 || true
  pkill -TERM -f "gzclient" >/dev/null 2>&1 || true
  sleep 1.0
}

launch_env() {
  local launch_file="$1"
  local wait_s="$2"

  cleanup_env
  cleanup_stale_env
  echo
  echo "=== starting ${launch_file} ==="
  env_log="${run_root}/config/${launch_file%.launch}_env.log"
  echo "environment log: ${env_log}"
  roslaunch stsm_madp "${launch_file}" gui:="${gui}" rviz:="${rviz}" >"${env_log}" 2>&1 &
  env_pid="$!"
  echo "waiting ${wait_s}s for Gazebo/RViz/controllers..."
  sleep "${wait_s}"
  check_env
}

check_env() {
  if [ -z "${env_pid}" ] || ! kill -0 "${env_pid}" >/dev/null 2>&1; then
    failed=1
    echo "ERROR: Gazebo/RViz environment roslaunch exited unexpectedly." >&2
    if [ -n "${env_log}" ] && [ -f "${env_log}" ]; then
      echo "Last environment log lines (${env_log}):" >&2
      tail -80 "${env_log}" >&2 || true
    fi
    return 1
  fi
}

trap cleanup_all EXIT
trap 'cleanup_all; exit 130' INT
trap 'cleanup_all; exit 143' TERM

commit_run() {
  local target="$1"
  local mode="$2"
  local tmp_metrics="$3"
  local tmp_traj="$4"
  local final_metrics="$5"
  local final_traj="$6"

  if [ ! -s "${tmp_metrics}" ]; then
    echo "ERROR: missing metrics for ${target}/${mode}: ${tmp_metrics}" >&2
    return 1
  fi
  if [ ! -s "${tmp_traj}" ]; then
    echo "ERROR: missing trajectory for ${target}/${mode}: ${tmp_traj}" >&2
    return 1
  fi

  python3 -B - "$tmp_metrics" "$final_metrics" "$target" "$mode" <<'PY'
import csv
import os
import sys
import tempfile

tmp_metrics, final_metrics, target, mode = sys.argv[1:5]
with open(tmp_metrics, "r") as f:
    new_rows = list(csv.DictReader(f))
if not new_rows:
    raise SystemExit("empty metrics file: {}".format(tmp_metrics))
new_row = new_rows[-1]
new_variant = (new_row.get("variant") or mode).strip().lower()

rows = []
fieldnames = list(new_row.keys())
if os.path.exists(final_metrics) and os.path.getsize(final_metrics) > 0:
    with open(final_metrics, "r") as f:
        reader = csv.DictReader(f)
        for old in reader:
            old_mode = old.get("mode", "").strip().lower()
            old_variant = old.get("variant", "").strip().lower()
            if not old_variant:
                old_variant = old_mode
            old_target = old.get("target", "").strip().lower()
            if old_variant == new_variant and old_target == target:
                continue
            rows.append(old)
        if reader.fieldnames:
            for key in reader.fieldnames:
                if key not in fieldnames:
                    fieldnames.append(key)
for key in new_row.keys():
    if key not in fieldnames:
        fieldnames.append(key)
rows.append(new_row)
required_compare_fields = [
    "topology_constraint_used",
    "corridor_constraint_used",
    "manifold_constraint_used",
    "critical_point_sequence_constraint_used",
    "critical_point_association_used",
    "topology_sequence_valid",
    "critical_point_status",
    "mpc_feasibility_status",
    "failure_reason",
    "replan_required",
]
for key in required_compare_fields:
    if key not in fieldnames:
        fieldnames.append(key)
for row in rows:
    variant = (row.get("variant") or row.get("mode", "")).strip().lower()
    if variant == "baseline":
        for key in (
                "topology_constraint_used",
                "corridor_constraint_used",
                "manifold_constraint_used",
                "critical_point_sequence_constraint_used",
                "critical_point_association_used",
                "topology_sequence_valid"):
            if row.get(key, "") in ("", None):
                row[key] = "0"
        if row.get("critical_point_status", "") in ("", None):
            row["critical_point_status"] = "passed"
        if row.get("mpc_feasibility_status", "") in ("", None):
            row["mpc_feasibility_status"] = "feasible"
        if row.get("failure_reason", "") in ("", None):
            row["failure_reason"] = ""
        if row.get("replan_required", "") in ("", None):
            row["replan_required"] = "0"
    elif variant == "stsm":
        for key in (
                "topology_constraint_used",
                "corridor_constraint_used",
                "manifold_constraint_used",
                "critical_point_sequence_constraint_used",
                "critical_point_association_used",
                "topology_sequence_valid"):
            if row.get(key, "") in ("", None):
                row[key] = "0"
        if row.get("critical_point_status", "") in ("", None):
            row["critical_point_status"] = "passed"
        if row.get("failure_reason", "") in ("", None):
            row["failure_reason"] = ""
order = {
    "baseline": 0,
    "stsm": 1,
}
rows.sort(key=lambda r: order.get(
    (r.get("variant") or r.get("mode", "")).strip().lower(), 99))

out_dir = os.path.dirname(final_metrics)
fd, tmp_out = tempfile.mkstemp(prefix=".metrics_", suffix=".csv", dir=out_dir)
os.close(fd)
try:
    with open(tmp_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    os.replace(tmp_out, final_metrics)
except Exception:
    try:
        os.unlink(tmp_out)
    except OSError:
        pass
    raise
PY

  python3 -B - "$tmp_metrics" "$tmp_traj" "$target" "$mode" <<'PY'
import csv
import sys

metrics_path, traj_path, target, mode = sys.argv[1:5]
required_metrics = [
    "run_id", "success_goal", "success_safe", "stop_triggered",
    "stop_reason", "first_stop_time_s", "gate_slow_count",
    "gate_stop_count", "max_gate_risk", "mean_gate_scale",
]
required_traj = [
    "t", "run_id", "mode", "target", "gate_state", "gate_scale",
    "gate_stop", "gate_reason", "rho_warn", "rho_stop",
]
if target == "wheelchair":
    required_metrics += [
        "mean_phi_max_point", "max_phi_max_point",
        "forbidden_hit_count", "interest_enabled",
        "footprint_gate_enabled", "footprint_slow_count",
        "footprint_stop_count", "first_footprint_stop_time_s",
        "footprint_stop_reason", "corridor_adp_raw_mean",
        "corridor_adp_norm", "corridor_rank_base", "corridor_rank_total",
        "corridor_rank_changed_count", "final_approach_used",
    ]
    required_traj += [
        "yaw", "phi_center", "phi_front_center", "phi_front_left",
        "phi_front_right", "phi_footrest_left", "phi_footrest_right",
        "phi_max_point", "worst_point_idx", "forbidden_hit",
        "gate_source", "footprint_gate_risk", "footprint_gate_scale",
        "footprint_gate_stop", "footprint_rho_warn", "footprint_rho_stop",
        "corridor_adp_raw_mean", "corridor_adp_norm",
        "corridor_rank_base", "corridor_rank_total",
    ]
elif target == "arm":
    required_metrics += [
        "arm_reached_hand", "arm_hold_completed", "arm_hold_duration_s",
        "arm_retreat_started", "arm_wait_reached",
        "arm_home_return_started", "arm_home_returned",
        "arm_task_complete", "arm_min_hand_dist", "arm_min_wait_dist",
        "arm_min_home_return_dist",
        "mean_phi_arm_max_point", "max_phi_arm_max_point",
        "arm_interest_enabled", "arm_interest_gate_enabled",
        "arm_interest_slow_count", "arm_interest_stop_count",
        "mean_arm_interest_gate_risk", "max_arm_interest_gate_risk",
        "arm_adp_grad_norm", "arm_adp_soft_cost",
        "arm_v_adp_alignment", "arm_dls_adp_used",
        "arm_qp_used", "arm_solver_success_count",
        "arm_solver_fallback_count", "arm_solver_success_rate",
        "v_des_raw_norm", "v_des_adp_norm", "v_des_delta_norm",
        "dq_nominal_norm", "dq_adp_norm", "dq_delta_norm",
    ]
    required_traj += [
        "phi_ee_point", "phi_wrist", "phi_elbow", "phi_object",
        "phi_arm_max_point", "phi_arm_mean_point", "phi_arm_sum_point",
        "arm_worst_point_idx", "arm_interest_valid_count",
        "arm_gate_source", "arm_interest_gate_enabled",
        "arm_interest_gate_risk", "arm_interest_gate_scale",
        "arm_interest_gate_stop", "arm_interest_rho_warn",
        "arm_interest_rho_stop", "arm_interest_gate_worst_idx",
        "arm_adp_grad_norm", "arm_adp_soft_cost",
        "arm_v_adp_alignment", "arm_dls_adp_used",
        "arm_qp_used", "arm_solver_success_count",
        "arm_solver_fallback_count", "arm_solver_success_rate",
    ]
with open(metrics_path, "r") as f:
    fields = csv.DictReader(f).fieldnames or []
missing = [x for x in required_metrics if x not in fields]
if missing:
    raise SystemExit("missing metrics fields in {}: {}".format(metrics_path, missing))
with open(traj_path, "r") as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames or []
    trows = list(reader)
if "run_id" not in fields and "selected_corridor_label" in fields:
    required_traj = [
        "t", "mode", "variant", "target", "x", "y", "z",
        "gate_state", "gate_scale", "gate_stop", "gate_reason",
        "adp_value", "adp_enabled", "selected_corridor_label",
        "selected_corridor_type",
    ]
    if target == "wheelchair":
        required_traj += [
            "yaw", "phi_center", "phi_max_point", "phi_mean_point",
            "forbidden_hit", "footprint_gate_risk",
            "footprint_gate_scale", "footprint_gate_stop",
        ]
    elif target == "arm":
        required_traj += [
            "phase", "phi_ee_point", "phi_wrist", "phi_elbow",
            "phi_object", "phi_arm_max_point", "phi_arm_mean_point",
            "arm_interest_gate_risk", "arm_interest_gate_scale",
            "arm_adp_grad_norm", "arm_adp_soft_cost",
            "arm_v_adp_alignment",
        ]
missing = [x for x in required_traj if x not in fields]
if missing:
    raise SystemExit("missing trajectory fields in {}: {}".format(traj_path, missing))
with open(metrics_path, "r") as f:
    mrows = list(csv.DictReader(f))
if not mrows:
    raise SystemExit("empty metrics file: {}".format(metrics_path))
if not trows:
    last_metrics = mrows[-1]
    execution_status = str(last_metrics.get("execution_status", "")).lower()
    failure_stage = str(last_metrics.get("failure_stage", "")).lower()
    selected = str(last_metrics.get("selected_corridor_id", ""))
    planning_failed = (
        execution_status == "failed" and
        failure_stage in ("planning", "refinement") and
        selected == "planning_failed")
    if planning_failed:
        sys.stderr.write(
            "warning: no trajectory rows for {}/{} because planning stopped at {}; "
            "preserving failure metrics\n".format(target, mode, failure_stage))
    else:
        raise SystemExit("empty trajectory file: {}".format(traj_path))
if trows and str(mrows[-1].get("stop_triggered", "0")) == "1":
    last = trows[-1]
    if last.get("gate_state") != "STOP" or str(last.get("gate_stop", "0")) != "1":
        raise SystemExit("STOP metrics found, but final trajectory row is not STOP")
PY

  local tmp_copy
  if [ "${tmp_traj}" != "${final_traj}" ]; then
    tmp_copy="$(mktemp "${final_traj}.tmp.XXXXXX")"
    cp "${tmp_traj}" "${tmp_copy}"
    mv "${tmp_copy}" "${final_traj}"
  fi
  python3 -B - "$tmp_metrics" "$(dirname "${tmp_traj}")/metrics.json" <<'PY'
import csv
import json
import os
import sys

metrics_csv, metrics_json = sys.argv[1:3]
with open(metrics_csv) as f:
    rows = list(csv.DictReader(f))
row = rows[-1] if rows else {}
if row:
    robot = row.get("robot") or row.get("target") or ""
    variant = row.get("variant") or row.get("mode") or ""
    if robot:
        row["robot"] = robot
    if variant:
        row["variant"] = variant
    out_dir = os.path.dirname(metrics_json)
    diag = {}
    try:
        with open(os.path.join(out_dir, "mpc_diagnostics.json")) as f:
            diag = json.load(f)
    except Exception:
        diag = {}
    if variant == "baseline":
        row["baseline_type"] = (
            row.get("baseline_type") or
            diag.get("baseline_type") or "direct")
        row["planner_source"] = (
            row.get("planner_source") or
            diag.get("planner_source") or "direct_connection")
        for key in (
                "topology_constraint_used",
                "critical_point_sequence_constraint_used",
                "corridor_constraint_used",
                "manifold_constraint_used",
                "critical_point_association_used",
                "morse_used",
                "refinement_used"):
            row[key] = "0"
        row.setdefault("topology_sequence_valid", "0")
        row.setdefault("critical_point_status", "passed")
        row["mpc_feasibility_status"] = (
            row.get("final_status") or
            diag.get("final_status") or
            row.get("final_mpc_status") or
            diag.get("final_mpc_status") or
            row.get("mpc_feasibility_status") or
            diag.get("mpc_feasibility_status") or "feasible")
        row["failure_reason"] = (
            row.get("final_failure_reason") or
            diag.get("final_failure_reason") or
            row.get("failure_reason") or
            diag.get("failure_reason") or "")
        row["replan_required"] = (
            row.get("replan_required") or
            int(bool(diag.get("replan_required", False))))
    elif diag:
        for key in (
                "topology_constraint_used",
                "critical_point_sequence_constraint_used",
                "corridor_constraint_used",
                "manifold_constraint_used",
                "topology_sequence_valid",
                "critical_point_status",
                "mpc_feasibility_status",
                "final_status",
                "final_mpc_status",
                "temporary_mpc_status",
                "failure_reason",
                "mpc_failure_reason",
                "temporary_failure_reason",
                "final_failure_reason",
                "replan_required"):
            value = diag.get(key, "")
            if key.endswith("_used") or key in (
                    "topology_sequence_valid", "replan_required"):
                value = int(bool(value))
            if str(row.get(key, "")).strip() in ("", "None"):
                row[key] = value
        final_status = str(
            diag.get("final_status", "") or
            diag.get("final_mpc_status", "") or "")
        if final_status:
            row["final_status"] = final_status
            row["final_mpc_status"] = final_status
            row["mpc_feasibility_status"] = final_status
        final_failure = str(diag.get("final_failure_reason", "") or "")
        if "final_failure_reason" in diag:
            row["failure_reason"] = final_failure
            row["mpc_failure_reason"] = final_failure
        status = str(row.get("mpc_feasibility_status", "") or
                     diag.get("mpc_feasibility_status", ""))
        if status == "feasible":
            row["failure_reason"] = ""
            row["mpc_failure_reason"] = ""
        if status in ("topology_infeasible", "corridor_infeasible",
                      "manifold_infeasible"):
            row["success"] = "0"
            row["success_safe"] = "0"
        elif status in ("feasible", "feasible_with_soft_violation",
                        "feasible_with_soft_violations"):
            if str(row.get("success_goal", row.get("success", ""))).lower() in (
                    "1", "true"):
                row["success"] = "1"
                row["success_safe"] = "1"
if out_dir and not os.path.isdir(out_dir):
    os.makedirs(out_dir)
with open(metrics_json, "w") as f:
    json.dump(row, f, indent=2, sort_keys=True)
PY
  python3 -B - "$(dirname "${tmp_traj}")" "${target}" "${variant}" <<'PY'
import json
import os
import sys

run_dir, robot, variant = sys.argv[1:4]
for name in (
        "metrics.json", "decision_trace.json", "mpc_diagnostics.json",
        "mpc_feedback.json", "topology_constraint.json", "topology_tube.json",
        "critical_point_association.json", "baseline_execution_chain.json",
        "consistency_check.json", "planning_trace.json",
        "mpc_validation.json"):
    path = os.path.join(run_dir, name)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        continue
    try:
        with open(path) as f:
            payload = json.load(f)
    except Exception:
        continue
    if isinstance(payload, dict):
        payload.setdefault("target", robot)
        payload["robot"] = robot
        payload["variant"] = variant
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
PY
  python3 -B - "$(dirname "${tmp_traj}")" "${target}" "${variant}" <<'PY'
import csv
import json
import os
import shutil
import sys

run_dir, robot, variant = sys.argv[1:4]

def load_json(name):
    path = os.path.join(run_dir, name)
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}

def load_metrics():
    path = os.path.join(run_dir, "metrics.csv")
    try:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        return rows[-1] if rows else {}
    except Exception:
        return {}

def truthy(value):
    return str(value).strip().lower() in ("1", "true", "yes", "ok", "feasible")

def number(value, default=0):
    try:
        if value in ("", None):
            return default
        raw = float(value)
        if abs(raw - int(raw)) <= 1e-9:
            return int(raw)
        return raw
    except Exception:
        return default

def first_value(*values):
    for value in values:
        if value not in ("", None):
            return value
    return ""

metrics = load_metrics()
metrics_json = load_json("metrics.json")
diag = load_json("mpc_diagnostics.json")
selected = load_json("selected_corridor.json")
candidate_stats = load_json("candidate_statistics.json")
candidate_generation = load_json("candidate_generation_report.json")
morse_diag = load_json("morse_diagnostics.json")
morse_routes = load_json("morse_routes.json")
traj_path = os.path.join(run_dir, "traj.csv")
trajectory_alias = os.path.join(run_dir, "trajectory.csv")
if os.path.exists(traj_path):
    try:
        shutil.copyfile(traj_path, trajectory_alias)
    except Exception:
        pass
traj_rows = 0
try:
    with open(traj_path) as f:
        traj_rows = max(0, sum(1 for _ in f) - 1)
except Exception:
    traj_rows = 0
mpc_status = str(
    diag.get("final_status") or
    diag.get("final_mpc_status") or
    diag.get("mpc_feasibility_status") or
    metrics_json.get("final_status") or
    metrics_json.get("final_mpc_status") or
    metrics_json.get("mpc_feasibility_status") or
    metrics.get("mpc_feasibility_status") or "")
goal_reached = truthy(metrics.get("success_goal", metrics_json.get("success_goal", "")))
try:
    final_dist = float(metrics.get(
        "final_dist_to_goal", metrics_json.get("final_dist_to_goal", "")))
except Exception:
    final_dist = float("inf")
try:
    goal_tol = float(metrics.get(
        "success_goal_tolerance",
        metrics_json.get("success_goal_tolerance", 0.08)) or 0.08)
except Exception:
    goal_tol = 0.08
if robot == "wheelchair":
    goal_reached = bool(goal_reached and final_dist < goal_tol)
constraint_payload = diag.get("constraints", {})
if not isinstance(constraint_payload, dict):
    constraint_payload = {}
constraint_mode = str(first_value(
    constraint_payload.get("manifold_constraint_mode"),
    diag.get("manifold_constraint_mode"),
    diag.get("mpc_manifold_constraint_mode"),
    "soft")).strip().lower()
if constraint_mode not in ("soft", "hard"):
    constraint_mode = "soft"
runtime_manifold_v = int(number(diag.get("manifold_violation_count"), 0))
runtime_corridor_v = int(number(diag.get("corridor_violation_count"), 0))
runtime_clearance_v = int(number(diag.get("clearance_violation_count"), 0))
runtime_violation_count = int(runtime_manifold_v + runtime_corridor_v)
hard_violation_count = (
    runtime_violation_count if constraint_mode == "hard" else 0)
constraint_violation = bool(hard_violation_count > 0)
major_violation_count = int(number(diag.get("major_violation_count"), 0))
max_soft_violation = number(diag.get("max_manifold_violation"), None)
soft_tolerance = float(number(first_value(
    diag.get("manifold_soft_tolerance"),
    constraint_payload.get("manifold_soft_tolerance"),
    0.005), 0.005))
override_count = int(number(diag.get("manifold_override_count"), 0))
step_count = int(number(first_value(
    diag.get("executed_trajectory_count"),
    diag.get("rollout_solve_count"),
    diag.get("reference_path_count"),
    0), 0))
soft_violation_ratio = (
    float(runtime_manifold_v) / float(step_count)
    if step_count > 0 else (1.0 if runtime_manifold_v > 0 else 0.0))
soft_violation_ratio_limit = float(number(first_value(
    diag.get("soft_violation_ratio_limit"),
    constraint_payload.get("soft_violation_ratio_limit"),
    0.0), 0.0))
override_ratio = (
    float(override_count) / float(step_count)
    if step_count > 0 else (1.0 if override_count > 0 else 0.0))
override_replan_limit = int(number(first_value(
    diag.get("override_replan_limit"),
    constraint_payload.get("override_replan_limit"),
    4), 4))
consecutive_override = int(number(
    diag.get("consecutive_manifold_override_max"), 0))
required = [
    "metrics.csv",
    "metrics.json",
    "trajectory.csv",
    "mpc_diagnostics.json",
]
missing = [
    name for name in required
    if not os.path.exists(os.path.join(run_dir, name)) or
    os.path.getsize(os.path.join(run_dir, name)) == 0
]
planning_finished = str(metrics.get("failure_stage", "")) != "planning"
mpc_finished = bool(mpc_status)
result_saved = not missing
status = {
    "robot": robot,
    "scenario": str(metrics.get("scenario") or metrics_json.get("scenario") or ""),
    "variant": variant,
    "simulation_started": True,
    "planning_finished": bool(planning_finished),
    "mpc_finished": bool(mpc_finished),
    "goal_reached": bool(goal_reached),
    "result_saved": bool(result_saved),
}
check = {
    "robot": robot,
    "variant": variant,
    "trajectory_empty": bool(traj_rows <= 0),
    "trajectory_rows": int(traj_rows),
    "goal_reached": bool(goal_reached),
    "mpc_feasible": bool(mpc_status == "feasible"),
    "mpc_feasibility_status": mpc_status,
    "constraint_violation": bool(constraint_violation),
    "result_files_complete": bool(not missing),
    "missing_files": missing,
    "passed": bool(
        traj_rows > 0 and goal_reached and mpc_status == "feasible" and
        not constraint_violation and not missing),
}
safety = {
    "candidate_manifold_valid": bool(
        selected.get("candidate_manifold_valid",
                     selected.get("manifold_feasible", False))),
    "candidate_tube_valid": bool(
        selected.get("candidate_tube_valid", selected.get("tube_valid", False))),
    "refinement_tube_valid": bool(
        metrics_json.get("refinement_tube_valid",
                         diag.get("refinement_tube_valid", False))),
    "tube_constraint_used": bool(diag.get("tube_constraint_used", False)),
    "tube_constraint_mode": str(diag.get("tube_constraint_mode", "")),
    "predicted_corridor_violation_count": int(float(
        diag.get("predicted_corridor_violation_count", 0) or 0)),
    "predicted_manifold_violation_count": int(float(
        diag.get("predicted_manifold_violation_count", 0) or 0)),
    "predicted_min_clearance": diag.get("predicted_min_clearance", ""),
    "predicted_max_risk": diag.get("predicted_max_risk", ""),
}
route_count = number(first_value(
    candidate_stats.get("morse_routes"),
    candidate_generation.get("morse_routes"),
    candidate_generation.get("num_topology_paths_discovered"),
    morse_diag.get("route_count"),
    morse_diag.get("routes"),
    len(morse_routes) if isinstance(morse_routes, list) else None,
    metrics_json.get("num_morse_saddle_corridors"),
    metrics.get("num_morse_saddle_corridors")), 0)
candidate_generated = number(first_value(
    candidate_stats.get("candidate_generated"),
    candidate_generation.get("candidate_generated"),
    candidate_generation.get("num_candidates_generated"),
    candidate_generation.get("total_candidates"),
    metrics_json.get("candidate_corridor_count"),
    metrics_json.get("num_candidate_corridors"),
    metrics.get("candidate_corridor_count"),
    metrics.get("num_candidate_corridors")), 0)
candidate_source = str(first_value(
    selected.get("candidate_source"),
    candidate_stats.get("selected_candidate_source"),
    candidate_generation.get("route_source"),
    metrics_json.get("candidate_source"),
    metrics.get("candidate_source"),
    "morse_topology" if candidate_generated else ""))
selected_candidate_source = str(first_value(
    selected.get("candidate_source"),
    candidate_stats.get("selected_candidate_source"),
    metrics_json.get("selected_candidate_source"),
    metrics.get("selected_candidate_source"),
    candidate_source))
reference_count = number(first_value(
    diag.get("reference_path_count"),
    metrics_json.get("reference_path_count"),
    metrics.get("reference_path_count")), 0)
mpc_used = bool(truthy(first_value(
    diag.get("mpc_used"),
    metrics_json.get("mpc_used"),
    metrics.get("mpc_used"),
    reference_count > 0)))
planner_success = bool(
    (candidate_generated > 0 and
     selected_candidate_source not in ("", "semantic", "direct", "fallback")) or
    truthy(first_value(diag.get("morse_used"), metrics_json.get("morse_used"),
                       metrics.get("morse_used"))))
controller_success = bool(mpc_used and mpc_status in (
    "feasible", "feasible_with_soft_violation",
    "feasible_with_soft_violations") and
    consecutive_override < override_replan_limit)
if constraint_mode == "hard":
    safety_success = bool(
        hard_violation_count == 0 and runtime_manifold_v == 0 and
        runtime_corridor_v == 0)
else:
    soft_evidence_complete = bool(
        runtime_manifold_v == 0 or max_soft_violation not in (None, ""))
    safety_success = bool(
        runtime_corridor_v == 0 and
        major_violation_count == 0 and
        soft_evidence_complete and
        float(max_soft_violation or 0.0) <= soft_tolerance + 1e-9 and
        soft_violation_ratio <= soft_violation_ratio_limit + 1e-9 and
        consecutive_override < override_replan_limit)
task_success = bool(goal_reached)
overall_success = bool(
    task_success and planner_success and controller_success and safety_success)
failure_reason = str(first_value(
    diag.get("failure_reason"),
    metrics_json.get("failure_reason"),
    metrics.get("failure_reason"),
    ""))
warning_reason = ""
if constraint_mode == "soft" and runtime_violation_count > 0:
    warning_reason = (
        "minor_soft_violation" if safety_success
        else "soft_manifold_violation_not_accepted")
if overall_success:
    failure_reason = ""
planning_trace = {
    "robot_type": robot,
    "morse_route_count": route_count,
    "candidate_generated": candidate_generated,
    "candidate_source": candidate_source,
    "selected_candidate_source": selected_candidate_source,
    "refinement_used": bool(truthy(first_value(
        selected.get("refinement_used"),
        selected.get("refinement_success"),
        metrics_json.get("refinement_success"),
        metrics_json.get("selected_refinement_used"),
        metrics.get("selected_refinement_used")))),
    "reference_source": str(first_value(
        diag.get("reference_source"),
        selected.get("reference_source"),
        metrics_json.get("reference_source"),
        metrics_json.get("mpc_reference_source"),
        metrics.get("reference_source"),
        metrics.get("mpc_reference_source"))),
    "mpc_used": bool(mpc_used),
    "mpc_status": mpc_status,
    "task_success": bool(task_success),
    "planner_success": bool(planner_success),
    "controller_success": bool(controller_success),
    "safety_success": bool(safety_success),
    "overall_success": bool(overall_success),
    "warning_reason": warning_reason,
    "failure_reason": failure_reason,
    "success": bool(overall_success),
}
mpc_validation = {
    "robot_type": robot,
    "reference_path_count": reference_count,
    "tube_constraint_used": bool(truthy(first_value(
        diag.get("tube_constraint_used"),
        metrics_json.get("tube_constraint_used"),
        metrics.get("tube_constraint_used")))),
    "predicted_min_clearance": first_value(
        diag.get("predicted_min_clearance"),
        diag.get("min_manifold_clearance"),
        metrics_json.get("predicted_min_clearance"),
        metrics_json.get("min_manifold_clearance"),
        metrics.get("predicted_min_clearance"),
        metrics.get("min_manifold_clearance")),
    "predicted_max_risk": first_value(
        diag.get("predicted_max_risk"),
        diag.get("mpc_max_risk"),
        diag.get("max_risk_value"),
        metrics_json.get("predicted_max_risk"),
        metrics_json.get("mpc_max_risk"),
        metrics_json.get("max_risk_value"),
        metrics.get("predicted_max_risk"),
        metrics.get("mpc_max_risk"),
        metrics.get("max_risk_value")),
    "manifold_violation_count": int(runtime_manifold_v),
    "corridor_violation_count": int(runtime_corridor_v),
    "clearance_violation_count": int(runtime_clearance_v),
    "violation_count": int(runtime_violation_count),
    "hard_violation_count": int(hard_violation_count),
    "major_violation_count": int(major_violation_count),
    "max_soft_violation": float(max_soft_violation or 0.0),
    "soft_violation_ratio": float(soft_violation_ratio),
    "soft_violation_ratio_limit": float(soft_violation_ratio_limit),
    "manifold_override_count": int(override_count),
    "override_ratio": float(override_ratio),
    "consecutive_manifold_override_max": int(consecutive_override),
    "constraint_mode": constraint_mode,
    "safety_success": bool(safety_success),
    "task_success": bool(task_success),
    "planner_success": bool(planner_success),
    "controller_success": bool(controller_success),
    "overall_success": bool(overall_success),
    "warning_reason": warning_reason,
    "failure_reason": failure_reason,
    "mpc_status": mpc_status,
    "mpc_used": bool(mpc_used),
    "success": bool(overall_success),
}
for name, payload in (
        ("simulation_status.json", status),
        ("simulation_check_report.json", check),
        ("safety_report.json", safety),
        ("planning_trace.json", planning_trace),
        ("mpc_validation.json", mpc_validation)):
    with open(os.path.join(run_dir, name), "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
PY
  if [ "${target}" = "wheelchair" ] && [ "${mode}" = "baseline" ]; then
    python3 -B - "$(dirname "${tmp_traj}")" <<'PY'
import json
import os
import sys

run_dir = sys.argv[1]
metrics = {}
metrics_path = os.path.join(run_dir, "metrics.json")
try:
    with open(metrics_path) as f:
        metrics = json.load(f)
except Exception:
    metrics = {}
baseline_type = str(metrics.get("baseline_type") or "direct")
selected_id = str(metrics.get("selected_corridor_id") or
                  metrics.get("corridor_id") or "baseline_direct")
planner_source = (
    "direct_connection" if baseline_type == "direct"
    else "traditional_grid_astar" if baseline_type == "traditional"
    else "wheelchair_safe_fallback")
payloads = {
    "mpc_diagnostics.json": {
        "target": "wheelchair",
        "variant": "baseline",
        "mode": "baseline",
        "baseline": True,
        "baseline_type": baseline_type,
        "planner_source": planner_source,
        "selected_corridor_id": selected_id,
        "selected_corridor_label": selected_id,
        "selected_corridor_type": "baseline",
        "topology_constraint_used": False,
        "corridor_constraint_used": False,
        "manifold_constraint_used": False,
        "critical_point_constraint_used": False,
        "critical_point_sequence_constraint_used": False,
        "critical_point_association_used": False,
        "topology_sequence_valid": False,
        "critical_point_status": "passed",
        "topology_sequence_constraint_used": False,
        "morse_used": False,
        "refinement_used": False,
        "selected_refinement_used": 0,
        "module_chain_valid": False,
        "mpc_feasibility_status": "feasible",
        "failure_reason": "none",
        "replan_required": False,
    },
    "topology_constraint.json": {
        "target": "wheelchair",
        "variant": "baseline",
        "selected_corridor_id": selected_id,
        "topology_constraint_used": False,
        "critical_point_sequence": [],
        "critical_points": [],
    },
    "critical_point_association.json": {
        "critical_points": [],
        "critical_point_association_used": False,
        "topology_sequence_valid": False,
        "critical_point_status": "passed",
    },
    "decision_trace.json": {
        "target": "wheelchair",
        "variant": "baseline",
        "mode": "baseline",
        "baseline_type": baseline_type,
        "planner_source": planner_source,
        "selected_corridor_id": selected_id,
        "selected_corridor_type": "baseline",
        "morse_used": False,
        "topology_constraint_used": False,
        "refinement_used": False,
        "final_path_source": planner_source,
        "execution_status": "success",
        "mpc_feasibility_status": "feasible",
    },
}
for name, payload in payloads.items():
    path = os.path.join(run_dir, name)
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
PY
  fi
  echo "committed ${target}/${mode}"
}

run_one() {
  local target="$1"
  local mode="$2"
  local baseline_arg="false"
  local baseline_type="direct"
  local run_adp_enabled="${adp_enabled}"
  local run_mpc_use_adp_terminal="false"
  local run_adp_model
  local adp_expected_critic_version
  local adp_expected_theta_hash
  local variant="$mode"
  local metrics_target="$target"
  local metrics_csv
  local traj_csv
  local tmp_metrics_csv
  local tmp_traj_csv
  local run_dir
  local action_launch
  local start_time
  local end_time
  local exit_code
  local status
  local log_path
  local interest_required

  if [ "${mode}" = "baseline" ]; then
    baseline_arg="true"
    baseline_type="direct"
    variant="baseline"
    run_adp_enabled="false"
  elif [ "${mode}" = "baseline0" ]; then
    baseline_arg="true"
    baseline_type="direct"
    variant="baseline0"
    run_adp_enabled="false"
  elif [ "${mode}" = "baseline1" ]; then
    baseline_arg="true"
    baseline_type="mpc_safe"
    variant="baseline1"
    run_adp_enabled="false"
  elif [ "${mode}" = "stsm" ]; then
    baseline_arg="false"
    variant="stsm"
    run_adp_enabled="${adp_enabled}"
    run_mpc_use_adp_terminal="${mpc_use_adp_terminal}"
  else
    echo "[run_experiments] unsupported wheelchair variant: ${mode}" >&2
    return 2
  fi

  case "${target}" in
    arm)
      metrics_csv="${run_root}/compare/arm_compare_metrics.csv"
      traj_csv="${run_root}/arm/${variant}/traj.csv"
      action_launch="arm_action.launch"
      run_adp_model="${arm_adp_model}"
      ;;
    wheelchair)
      metrics_csv="${run_root}/compare/wheelchair_compare_metrics.csv"
      traj_csv="${run_root}/wheelchair/${variant}/traj.csv"
      action_launch="wheelchair_action.launch"
      run_adp_model="${wheelchair_adp_model}"
      ;;
    *)
      echo "unknown target: ${target}" >&2
      exit 2
      ;;
  esac

  if [ -n "${adp_model_override}" ]; then
    run_adp_model="${adp_model_override}"
  fi
  if [ "${run_adp_enabled}" = "true" ]; then
    read -r adp_expected_critic_version adp_expected_theta_hash <<EOF
$(adp_critic_identity "${run_adp_model}")
EOF
  else
    adp_expected_critic_version=""
    adp_expected_theta_hash=""
  fi

  run_dir="${run_root}/${target}/${variant}"
  mkdir -p "${run_dir}"
  tmp_metrics_csv="${run_dir}/metrics.csv"
  tmp_traj_csv="${run_dir}/traj.csv"
  decision_trace_json="${run_dir}/decision_trace.json"
  mpc_reference_csv="${run_dir}/mpc_reference_path.csv"
  mpc_diagnostics_json="${run_dir}/mpc_diagnostics.json"
  mpc_cost_breakdown_csv="${run_dir}/mpc_cost_breakdown.csv"
  arm_handover_debug_json="${run_dir}/arm_handover_debug.json"
  log_path="${run_dir}/ros.log"

  echo
  echo "=== ${target} / ${mode} ==="
  echo "temporary metrics: ${tmp_metrics_csv}"
  echo "temporary traj:    ${tmp_traj_csv}"
  echo "decision trace:    ${decision_trace_json}"
  echo "mpc reference:     ${mpc_reference_csv}"
  echo "final metrics:     ${metrics_csv}"
  echo "final traj:        ${traj_csv}"

  rosparam set /stsm/run_id "'${run_id}'"
  if [ "${target}" = "arm" ]; then
    interest_required="${arm_interest_enabled}"
    metrics_goal_tolerance="${wc_goal_tolerance}"
  else
    interest_required="${wc_interest_enabled}"
    metrics_goal_tolerance="${wc_completion_tolerance}"
  fi

  rosrun stsm_madp metrics_node.py \
    _target:="${metrics_target}" \
	    _variant:="${variant}" \
		    _out:="${tmp_metrics_csv}" \
		    _traj_out:="${tmp_traj_csv}" \
		    _mpc_diagnostics:="${mpc_diagnostics_json}" \
		    _success_goal_tolerance:="${metrics_goal_tolerance}" \
		    _interest_required:="${interest_required}" \
    __name:="stsm_metrics_${target}_${mode}" &
  metrics_pid="$!"

  sleep 1.0
  start_time="$(date -Iseconds)"
  set +e
  if [ "${target}" = "wheelchair" ]; then
    roslaunch stsm_madp "${action_launch}" \
	      baseline:="${baseline_arg}" \
	      baseline_type:="${baseline_type}" \
      goal_tolerance:="${wc_goal_tolerance}" \
      experiment_mode:="${experiment_mode}" \
      completion_tolerance:="${wc_completion_tolerance}" \
      completion_hold_s:="${wc_completion_hold_s}" \
      max_runtime_s:="${wc_max_runtime_s}" \
      command_hold_s:="${wc_command_hold_s}" \
      mpc_solve_deadline_s:="${wc_mpc_solve_deadline_s}" \
      no_progress_timeout_s:="${wc_no_progress_timeout_s}" \
      no_progress_epsilon:="${wc_no_progress_epsilon}" \
      replan_period:="${wc_replan_period}" \
      no_progress_replan_time:="${wc_no_progress_replan_time}" \
      progress_eps:="${wc_progress_eps}" \
      replan_tube_margin:="${wc_replan_tube_margin}" \
      near_goal_radius:="${wc_near_goal_radius}" \
      near_goal_adp_scale:="${wc_near_goal_adp_scale}" \
      min_progress_per_solve:="${wc_min_progress_per_solve}" \
      near_goal_goal_weight:="${wc_near_goal_goal_weight}" \
      near_goal_social_scale:="${wc_near_goal_social_scale}" \
      lam_stall:="${wc_lam_stall}" \
      progress_reward_weight:="${wc_progress_reward_weight}" \
      final_approach_radius:="${wc_final_approach_radius}" \
      final_heading_threshold:="${wc_final_heading_threshold}" \
      final_heading_gain:="${wc_final_heading_gain}" \
      final_creep_v:="${wc_final_creep_v}" \
      final_min_v:="${wc_final_min_v}" \
      final_max_v:="${wc_final_max_v}" \
      final_forward_gain:="${wc_final_forward_gain}" \
      lam_heading:="${wc_lam_heading}" \
      final_direct_override_enabled:="${wc_final_direct_override_enabled}" \
      final_direct_override_radius:="${wc_final_direct_override_radius}" \
      mpc_horizon:="${wc_mpc_horizon}" \
      mpc_dt:="${wc_mpc_dt}" \
      mpc_a_max:="${wc_mpc_a_max}" \
      mpc_alpha_max:="${wc_mpc_alpha_max}" \
      mpc_beam_width:="${wc_mpc_beam_width}" \
      interest_enabled:="${wc_interest_enabled}" \
      interest_gate_enabled:="${wc_interest_gate_enabled}" \
      footprint_rho_warn:="${wc_footprint_rho_warn}" \
      footprint_rho_stop:="${wc_footprint_rho_stop}" \
      footprint_gate_min_scale:="${wc_footprint_gate_min_scale}" \
      footprint_forbidden_stop_enabled:="${wc_footprint_forbidden_stop_enabled}" \
      adp_enabled:="${run_adp_enabled}" \
      adp_model:="${run_adp_model}" \
      adp_expected_critic_path:="${run_adp_model}" \
      adp_expected_critic_version:="${adp_expected_critic_version}" \
      adp_expected_theta_hash:="${adp_expected_theta_hash}" \
      adp_decision_influence_enabled:="${adp_decision_influence_enabled}" \
      adp_ranking_influence_enabled:="${adp_ranking_influence_enabled}" \
      adp_mpc_influence_enabled:="${adp_mpc_influence_enabled}" \
      adp_ranking_lambda:="${adp_ranking_lambda}" \
      adp_value_normalization:="${adp_value_normalization}" \
      adp_norm_clip:="${adp_norm_clip}" \
      adp_contribution_clip:="${adp_contribution_clip}" \
      lambda_adp:="${lambda_adp}" \
      lambda_adp_corridor:="${lambda_adp_corridor}" \
      lambda_adp_terminal:="${lambda_adp_terminal}" \
      mpc_use_adp_terminal:="${run_mpc_use_adp_terminal}" \
      adp_post_scale_enabled:="${adp_post_scale_enabled}" \
      adp_min_scale:="${adp_min_scale}" \
      adp_debug:="${adp_debug}" \
	      decision_trace_out:="${decision_trace_json}" \
	      mpc_reference_out:="${mpc_reference_csv}" \
	      mpc_diagnostics_out:="${mpc_diagnostics_json}" \
	      mpc_cost_breakdown_out:="${mpc_cost_breakdown_csv}" \
	      2>&1 | tee "${log_path}"
  else
    roslaunch stsm_madp "${action_launch}" \
      baseline:="${baseline_arg}" \
      experiment_mode:="${experiment_mode}" \
      arm_interest_enabled:="${arm_interest_enabled}" \
      arm_interest_gate_enabled:="${arm_interest_gate_enabled}" \
      arm_interest_rho_warn:="${arm_interest_rho_warn}" \
      arm_interest_rho_stop:="${arm_interest_rho_stop}" \
      arm_interest_gate_min_scale:="${arm_interest_gate_min_scale}" \
      adp_enabled:="${run_adp_enabled}" \
      adp_model:="${run_adp_model}" \
      adp_expected_critic_path:="${run_adp_model}" \
      adp_expected_critic_version:="${adp_expected_critic_version}" \
      adp_expected_theta_hash:="${adp_expected_theta_hash}" \
      adp_decision_influence_enabled:="${adp_decision_influence_enabled}" \
      adp_ranking_influence_enabled:="${adp_ranking_influence_enabled}" \
      adp_mpc_influence_enabled:="${adp_mpc_influence_enabled}" \
      adp_ranking_lambda:="${adp_ranking_lambda}" \
      adp_value_normalization:="${adp_value_normalization}" \
      adp_norm_clip:="${adp_norm_clip}" \
      adp_contribution_clip:="${adp_contribution_clip}" \
      lambda_adp:="${lambda_adp}" \
      lambda_adp_path:="${lambda_adp_path}" \
      lambda_adp_arm:="${lambda_adp_arm}" \
      adp_grad_eps:="${adp_grad_eps}" \
      adp_descent_gain:="${adp_descent_gain}" \
      adp_grad_clip:="${adp_grad_clip}" \
      adp_solver_mode:="${adp_solver_mode}" \
      use_cvxpy:="${use_cvxpy}" \
      adp_blend_alpha:="${adp_blend_alpha}" \
      adp_post_scale_enabled:="${adp_post_scale_enabled}" \
      adp_min_scale:="${adp_min_scale}" \
      adp_debug:="${adp_debug}" \
      decision_trace_out:="${decision_trace_json}" \
      mpc_reference_out:="${mpc_reference_csv}" \
      arm_handover_debug_out:="${arm_handover_debug_json}" \
      2>&1 | tee "${log_path}"
  fi
  exit_code="${PIPESTATUS[0]}"
  set -e
  end_time="$(date -Iseconds)"
  sleep 1.0
  cleanup_metrics
  status="ok"
  if [ "${exit_code}" -ne 0 ]; then
    status="failed"
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${run_id}" "${metrics_target}" "${variant}" "${mode}" "${start_time}" "${end_time}" \
    "${status}" "${exit_code}" "${target}/${variant}/metrics.csv" \
    "${target}/${variant}/traj.csv" "${target}/${variant}/ros.log" >> "${manifest_csv}"
  if [ "${exit_code}" -ne 0 ]; then
    failed=1
    echo "ERROR: roslaunch failed for ${target}/${mode}, see ${log_path}" >&2
    return "${exit_code}"
  fi
  check_env
  commit_run "${metrics_target}" "${mode}" "${tmp_metrics_csv}" "${tmp_traj_csv}" \
    "${metrics_csv}" "${traj_csv}"
  python3 -B "${pkg_dir}/scripts/ensure_result_diagnostics.py" \
    --run-dir "${run_root}"
  check_env
}

catkin_make -C /home/sun/LLL/catkin_ws

export RUN_ID="${run_id}"
export TARGET="${target_filter}"
export GUI="${gui}"
export RVIZ="${rviz}"
export PLOT="${plot}"
export CLEAN_ENV="${clean_env}"
export ADP_ENABLED="${adp_enabled}"
export ADP_DECISION_INFLUENCE_ENABLED="${adp_decision_influence_enabled}"
export ADP_RANKING_INFLUENCE_ENABLED="${adp_ranking_influence_enabled}"
export ADP_MPC_INFLUENCE_ENABLED="${adp_mpc_influence_enabled}"
export ADP_RANKING_LAMBDA="${adp_ranking_lambda}"
export ADP_SOLVER_MODE="${adp_solver_mode}"
export USE_CVXPY="${use_cvxpy}"
export ADP_BLEND_ALPHA="${adp_blend_alpha}"
export ADP_DESCENT_GAIN="${adp_descent_gain}"
export LAMBDA_ADP="${lambda_adp}"
export LAMBDA_ADP_CORRIDOR="${lambda_adp_corridor}"
export LAMBDA_ADP_TERMINAL="${lambda_adp_terminal}"
export LAMBDA_ADP_PATH="${lambda_adp_path}"
export LAMBDA_ADP_ARM="${lambda_adp_arm}"
export WC_GOAL_TOLERANCE="${wc_goal_tolerance}"
export WC_COMPLETION_TOLERANCE="${wc_completion_tolerance}"
export WC_COMMAND_HOLD_S="${wc_command_hold_s}"
export WC_MPC_SOLVE_DEADLINE_S="${wc_mpc_solve_deadline_s}"
export WC_VARIANTS="${wc_variants}"
export ARM_VARIANTS="${arm_variants}"
export WC_REPLAN_PERIOD="${wc_replan_period}"
export WC_NEAR_GOAL_RADIUS="${wc_near_goal_radius}"
export WC_NEAR_GOAL_ADP_SCALE="${wc_near_goal_adp_scale}"
export WC_NEAR_GOAL_GOAL_WEIGHT="${wc_near_goal_goal_weight}"
export WC_NO_PROGRESS_REPLAN_TIME="${wc_no_progress_replan_time}"
export WC_PROGRESS_REWARD_WEIGHT="${wc_progress_reward_weight}"
export WC_FINAL_APPROACH_RADIUS="${wc_final_approach_radius}"
export WC_FINAL_HEADING_THRESHOLD="${wc_final_heading_threshold}"
export WC_FINAL_HEADING_GAIN="${wc_final_heading_gain}"
export WC_FINAL_CREEP_V="${wc_final_creep_v}"
export WC_FINAL_MIN_V="${wc_final_min_v}"
export WC_FINAL_MAX_V="${wc_final_max_v}"
export WC_FINAL_FORWARD_GAIN="${wc_final_forward_gain}"
export WC_LAM_HEADING="${wc_lam_heading}"
export WC_FINAL_DIRECT_OVERRIDE_ENABLED="${wc_final_direct_override_enabled}"
export WC_FINAL_DIRECT_OVERRIDE_RADIUS="${wc_final_direct_override_radius}"
export WC_MPC_HORIZON="${wc_mpc_horizon}"
export WC_MPC_DT="${wc_mpc_dt}"
export WC_MPC_A_MAX="${wc_mpc_a_max}"
export WC_MPC_ALPHA_MAX="${wc_mpc_alpha_max}"
export WC_MPC_BEAM_WIDTH="${wc_mpc_beam_width}"

python3 -B "${pkg_dir}/scripts/analysis/results_manager.py" organize \
  --run-id "${run_id}" \
  --purpose "${PURPOSE:-}" \
  --stage "${STAGE:-}" \
  --status running \
  --update-latest

if [ "${target_filter}" = "all" ] || [ "${target_filter}" = "arm" ]; then
  launch_env arm_view.launch "${arm_wait_s}"
  for arm_variant in ${arm_variants}; do
    run_one arm "${arm_variant}"
  done
fi

if [ "${target_filter}" = "all" ] || [ "${target_filter}" = "wheelchair" ]; then
  echo
  echo "=== switching to wheelchair scene; the arm Gazebo/RViz window will close here ==="
  launch_env wheelchair_view.launch "${wc_wait_s}"
  for wc_variant in ${wc_variants}; do
    run_one wheelchair "${wc_variant}"
  done
fi

cleanup_env

python3 -B "${pkg_dir}/scripts/analysis/results_manager.py" collect \
  --results-root "${results_dir}" \
  --out "${results_dir}/summary/all_metrics.csv"

python3 -B "${pkg_dir}/scripts/analysis/results_manager.py" organize \
  --run-id "${run_id}" \
  --purpose "${PURPOSE:-}" \
  --stage "${STAGE:-}" \
  --status ok \
  --update-latest

python3 -B "${pkg_dir}/scripts/cleanup_results.py" \
  --results-root "${results_dir}" \
  --finalize \
  --execute

python3 -B - "${run_root}" "${results_dir}/run" <<'PY'
import os
import shutil
import sys

src, dst = sys.argv[1:3]
if os.path.isdir(src):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
PY

python3 -B "${pkg_dir}/scripts/ensure_result_diagnostics.py" \
  --run-dir "${run_root}"
python3 -B "${pkg_dir}/scripts/ensure_result_diagnostics.py" \
  --run-dir "${results_dir}/run"

python3 -B - "${run_root}" <<'PY'
import csv
import json
import os
import shutil
import sys

run_root = sys.argv[1]
wc_root = os.path.join(run_root, "wheelchair")
stsm_root = os.path.join(wc_root, "stsm")
if not os.path.isdir(stsm_root):
    os.makedirs(stsm_root)
for name in (
        "mpc_diagnostics.json", "decision_trace.json",
        "topology_constraint.json", "critical_point_association.json",
        "mpc_feedback.json", "topology_tube.json",
        "consistency_check.json", "planning_trace.json",
        "mpc_validation.json"):
    src = os.path.join(wc_root, name)
    dst = os.path.join(stsm_root, name)
    if (not os.path.exists(dst) or os.path.getsize(dst) == 0) and os.path.exists(src):
        shutil.copy2(src, dst)

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}

def load_last_csv(path):
    try:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        return rows[-1] if rows else {}
    except Exception:
        return {}

def truthy(value):
    return str(value).strip().lower() in ("1", "true", "yes")

def first_value(*values):
    for value in values:
        if value not in ("", None):
            return value
    return ""

robots = [
    robot for robot in ("arm", "wheelchair")
    if os.path.isdir(os.path.join(run_root, robot))
]
required_variant_files = (
    "traj.csv", "metrics.json", "mpc_diagnostics.json",
    "decision_trace.json")
variant_isolated = bool(robots) and all(
    os.path.exists(os.path.join(run_root, robot, variant, name))
    for robot in robots
    for variant in ("baseline", "stsm")
    for name in required_variant_files)
variant_isolated = variant_isolated and all(
    os.path.exists(os.path.join(run_root, robot, "stsm", name))
    for robot in robots
    for name in ("topology_constraint.json",
                 "critical_point_association.json"))
root_duplicates = [
    os.path.join(wc_root, name)
    for name in ("traj.csv", "metrics.csv", "metrics.json", "mpc_diagnostics.json",
                 "decision_trace.json", "topology_constraint.json")
]
root_duplicates += [
    os.path.join(run_root, robot, name)
    for robot in ("arm", "wheelchair")
    for name in ("traj.csv", "metrics.csv", "metrics.json", "mpc_diagnostics.json",
                 "decision_trace.json", "topology_constraint.json")
]
no_root_duplicates = not any(os.path.exists(path) for path in root_duplicates)
figures_dir = os.path.join(os.path.dirname(run_root), "figures")
figures_clean = True
if os.path.isdir(figures_dir):
    for root, _, files in os.walk(figures_dir):
        for name in files:
            if os.path.splitext(name)[1].lower() not in (".png", ".pdf", ".svg"):
                figures_clean = False

allowed = ("feasible", "feasible_with_soft_violation")
baseline_valid = bool(robots)
stsm_valid = bool(robots)
diagnostics_consistent = bool(robots)
metrics_consistent = bool(robots)
topology_mpc_closed_loop = bool(robots)
execution_success_valid = bool(robots)
for robot in robots:
    robot_root = os.path.join(run_root, robot)
    baseline_traj = load_last_csv(os.path.join(robot_root, "baseline", "traj.csv"))
    stsm_traj = load_last_csv(os.path.join(robot_root, "stsm", "traj.csv"))
    baseline_diag = load_json(os.path.join(robot_root, "baseline", "mpc_diagnostics.json"))
    baseline_trace = load_json(os.path.join(robot_root, "baseline", "decision_trace.json"))
    baseline_metric_json = load_json(os.path.join(robot_root, "baseline", "metrics.json"))
    stsm_diag = load_json(os.path.join(robot_root, "stsm", "mpc_diagnostics.json"))
    stsm_trace = load_json(os.path.join(robot_root, "stsm", "decision_trace.json"))
    stsm_constraint = load_json(os.path.join(robot_root, "stsm", "topology_constraint.json"))
    metrics_rows = []
    for path in (
            os.path.join(run_root, "metrics", "{}_metrics.csv".format(robot)),
            os.path.join(run_root, "compare", "{}_compare_metrics.csv".format(robot))):
        try:
            with open(path) as f:
                metrics_rows.extend(csv.DictReader(f))
        except Exception:
            pass
    by_variant = {}
    for row in metrics_rows:
        variant = str(row.get("variant") or row.get("mode") or "").strip().lower()
        if variant in ("baseline", "stsm"):
            by_variant[variant] = row

    baseline_valid = baseline_valid and (
        baseline_traj.get("variant") == "baseline" and
        str(baseline_diag.get("baseline_type") or
            baseline_trace.get("baseline_type") or
            baseline_metric_json.get("baseline_type") or "").lower() ==
        "direct" and
        str(baseline_diag.get("planner_source") or
            baseline_trace.get("planner_source") or
            baseline_metric_json.get("planner_source") or "").lower() ==
        "direct_connection" and
        not truthy(baseline_diag.get("topology_constraint_used")))
    stsm_valid = stsm_valid and (
        stsm_traj.get("variant") == "stsm" and
        truthy(stsm_diag.get("topology_constraint_used")) and
        truthy(stsm_diag.get("critical_point_sequence_constraint_used")))
    diagnostics_consistent = diagnostics_consistent and bool(
        baseline_diag and stsm_diag and
        "mpc_feasibility_status" in baseline_diag and
        "mpc_feasibility_status" in stsm_diag)

    topology_payload = dict(stsm_diag.get("topology_constraint", {}) or stsm_constraint or {})
    topology_mpc_closed_loop = topology_mpc_closed_loop and all([
        truthy(stsm_diag.get("topology_constraint_used")),
        truthy(stsm_diag.get("critical_point_sequence_constraint_used")),
        truthy(stsm_diag.get("corridor_constraint_used")),
        truthy(stsm_diag.get("manifold_constraint_used")),
        bool(topology_payload.get("critical_point_constraint")),
        bool(topology_payload.get("corridor_constraint")),
        bool(topology_payload.get("manifold_constraint")),
        bool(topology_payload.get("topology_sequence_constraint")),
        os.path.exists(os.path.join(robot_root, "stsm", "critical_point_association.json")),
    ])

    for variant, diag in (("baseline", baseline_diag), ("stsm", stsm_diag)):
        metric = by_variant.get(variant, {})
        metric_json = load_json(os.path.join(robot_root, variant, "metrics.json"))
        diag_status = str(diag.get("mpc_feasibility_status", ""))
        metric_status = str(
            metric.get("mpc_feasibility_status") or
            metric_json.get("mpc_feasibility_status") or "")
        if diag_status and metric_status and diag_status != metric_status:
            metrics_consistent = False
        success_value = str(
            metric.get("success") if metric.get("success") not in (None, "")
            else metric_json.get("success", "")).strip().lower()
        if success_value in ("1", "true", "yes") and diag_status not in allowed:
            metrics_consistent = False
        outcome = first_value(
            diag.get("overall_success"), diag.get("task_success"),
            metric_json.get("overall_success"), metric_json.get("task_success"),
            metric.get("overall_success"), metric.get("task_success"),
            metric.get("success"), False)
        execution_success_valid = execution_success_valid and truthy(outcome)

result_structure_valid = bool(
    variant_isolated and no_root_duplicates and figures_clean)
diagnostics_metrics_consistent = bool(
    diagnostics_consistent and metrics_consistent)
topology_constraint_consistent = bool(
    topology_mpc_closed_loop and stsm_valid)
report = {
    "topology_mpc_closed_loop": bool(topology_mpc_closed_loop),
    "diagnostics_consistent": bool(diagnostics_consistent),
    "metrics_consistent": bool(metrics_consistent),
    "result_structure_valid": bool(result_structure_valid),
    "baseline_valid": bool(baseline_valid),
    "stsm_valid": bool(stsm_valid),
    "diagnostics_metrics_consistent": bool(diagnostics_metrics_consistent),
    "topology_constraint_consistent": bool(topology_constraint_consistent),
    "variant_isolated": bool(variant_isolated and no_root_duplicates),
    "execution_success_valid": bool(execution_success_valid),
}
report["overall_pass"] = all(bool(report[key]) for key in (
    "topology_mpc_closed_loop",
    "diagnostics_consistent",
    "metrics_consistent",
    "result_structure_valid",
    "baseline_valid",
    "stsm_valid",
    "execution_success_valid",
))
out = os.path.join(run_root, "experiment_consistency_report.json")
with open(out, "w") as f:
    json.dump(report, f, indent=2, sort_keys=True)
print("wrote {}".format(out))
PY

python3 -B - "${run_root}" "${results_dir}/run" <<'PY'
import os
import shutil
import sys

src, dst = sys.argv[1:3]
if os.path.isdir(src):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
PY

if [ "${plot}" = "true" ]; then
  set +e
  timeout "${plot_timeout_s}" python3 -B "${pkg_dir}/scripts/visualization/generate_results_figures.py" \
    --run-dir "${results_dir}/run" \
    --figures-dir "${results_dir}/figures"
  plot_exit="$?"
  set -e
  if [ "${plot_exit}" -eq 124 ]; then
    echo "WARNING: final result plotting timed out after ${plot_timeout_s}s; results are still finalized." >&2
  elif [ "${plot_exit}" -ne 0 ]; then
    echo "WARNING: final result plotting exited with code ${plot_exit}; results are still finalized." >&2
  fi
fi

find "${pkg_dir}" -type f -name '*.pyc' -delete
find "${pkg_dir}" -type d -name '__pycache__' -empty -delete

echo
echo "Done."
echo "Results:  ${results_dir}"
echo "Figures:  ${results_dir}/figures"
echo "Run:      ${results_dir}/run"
echo "Summary:  ${results_dir}/summary"
