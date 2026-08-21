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
pictures_dir="${results_dir}/paper_figures"
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
clean="${CLEAN:-false}"
clean_env="${CLEAN_ENV:-true}"
keep_on_fail="${KEEP_ON_FAIL:-true}"
target_filter="${TARGET:-all}"
adp_enabled="${ADP_ENABLED:-true}"
adp_model="${ADP_MODEL:-${pkg_dir}/config/adp_critic.yaml}"
lambda_adp="${LAMBDA_ADP:-0.005}"
lambda_adp_corridor="${LAMBDA_ADP_CORRIDOR:-0.003}"
lambda_adp_terminal="${LAMBDA_ADP_TERMINAL:-0.0015}"
lambda_adp_path="${LAMBDA_ADP_PATH:-${lambda_adp}}"
lambda_adp_arm="${LAMBDA_ADP_ARM:-0.02}"
adp_grad_eps="${ADP_GRAD_EPS:-0.01}"
adp_descent_gain="${ADP_DESCENT_GAIN:-0.12}"
adp_grad_clip="${ADP_GRAD_CLIP:-8.0}"
adp_solver_mode="${ADP_SOLVER_MODE:-dls_adp}"
use_cvxpy="${USE_CVXPY:-false}"
adp_blend_alpha="${ADP_BLEND_ALPHA:-0.35}"
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
wc_completion_tolerance="${WC_COMPLETION_TOLERANCE:-0.25}"
wc_completion_hold_s="${WC_COMPLETION_HOLD_S:-1.5}"
wc_max_runtime_s="${WC_MAX_RUNTIME_S:-180.0}"
wc_no_progress_timeout_s="${WC_NO_PROGRESS_TIMEOUT_S:-45.0}"
wc_no_progress_epsilon="${WC_NO_PROGRESS_EPSILON:-0.02}"
wc_replan_period="${WC_REPLAN_PERIOD:-5.0}"
wc_no_progress_replan_time="${WC_NO_PROGRESS_REPLAN_TIME:-5.0}"
wc_progress_eps="${WC_PROGRESS_EPS:-0.01}"
wc_replan_tube_margin="${WC_REPLAN_TUBE_MARGIN:-0.08}"
wc_near_goal_radius="${WC_NEAR_GOAL_RADIUS:-0.50}"
wc_near_goal_adp_scale="${WC_NEAR_GOAL_ADP_SCALE:-0.25}"
wc_min_progress_per_solve="${WC_MIN_PROGRESS_PER_SOLVE:-0.005}"
wc_near_goal_goal_weight="${WC_NEAR_GOAL_GOAL_WEIGHT:-12.0}"
wc_near_goal_social_scale="${WC_NEAR_GOAL_SOCIAL_SCALE:-0.5}"
wc_lam_stall="${WC_LAM_STALL:-10.0}"
wc_interest_enabled="${WC_INTEREST_ENABLED:-true}"
wc_interest_gate_enabled="${WC_INTEREST_GATE_ENABLED:-true}"
wc_footprint_rho_warn="${WC_FOOTPRINT_RHO_WARN:-5.0}"
wc_footprint_rho_stop="${WC_FOOTPRINT_RHO_STOP:-7.0}"
wc_footprint_gate_min_scale="${WC_FOOTPRINT_GATE_MIN_SCALE:-0.20}"
wc_footprint_forbidden_stop_enabled="${WC_FOOTPRINT_FORBIDDEN_STOP_ENABLED:-true}"
run_root="${results_dir}/runs/${run_id}"
mkdir -p \
  "${results_dir}/runs" "${results_dir}/summary" \
  "${results_dir}/paper_figures" "${results_dir}/archive" \
  "${pictures_dir}" "${run_root}/config" "${run_root}/compare"
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
order = {"baseline": 0, "stsm": 1}
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

  python3 -B - "$tmp_metrics" "$tmp_traj" "$target" <<'PY'
import csv
import sys

metrics_path, traj_path, target = sys.argv[1:4]
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
if not mrows or not trows:
    raise SystemExit("empty metrics or trajectory")
if str(mrows[-1].get("stop_triggered", "0")) == "1":
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
  echo "committed ${target}/${mode}"
}

run_one() {
  local target="$1"
  local mode="$2"
  local baseline_arg="false"
  local run_adp_enabled="${adp_enabled}"
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
    run_adp_enabled="false"
  else
    variant="${mode}"
  fi

  case "${target}" in
    arm)
      metrics_csv="${run_root}/compare/arm_compare_metrics.csv"
      traj_csv="${run_root}/arm/${variant}/traj.csv"
      action_launch="arm_action.launch"
      ;;
    wheelchair)
      metrics_csv="${run_root}/compare/wheelchair_compare_metrics.csv"
      traj_csv="${run_root}/wheelchair/${variant}/traj.csv"
      action_launch="wheelchair_action.launch"
      ;;
    *)
      echo "unknown target: ${target}" >&2
      exit 2
      ;;
  esac

  run_dir="${run_root}/${target}/${variant}"
  mkdir -p "${run_dir}"
  tmp_metrics_csv="${run_dir}/metrics.csv"
  tmp_traj_csv="${run_dir}/traj.csv"
  log_path="${run_dir}/ros.log"

  echo
  echo "=== ${target} / ${mode} ==="
  echo "temporary metrics: ${tmp_metrics_csv}"
  echo "temporary traj:    ${tmp_traj_csv}"
  echo "final metrics:     ${metrics_csv}"
  echo "final traj:        ${traj_csv}"

  rosparam set /stsm/run_id "'${run_id}'"
  if [ "${target}" = "arm" ]; then
    interest_required="${arm_interest_enabled}"
  else
    interest_required="${wc_interest_enabled}"
  fi

  rosrun stsm_madp metrics_node.py \
    _target:="${metrics_target}" \
    _variant:="${variant}" \
    _out:="${tmp_metrics_csv}" \
    _traj_out:="${tmp_traj_csv}" \
    _interest_required:="${interest_required}" \
    __name:="stsm_metrics_${target}_${mode}" &
  metrics_pid="$!"

  sleep 1.0
  start_time="$(date -Iseconds)"
  set +e
  if [ "${target}" = "wheelchair" ]; then
    roslaunch stsm_madp "${action_launch}" \
      baseline:="${baseline_arg}" \
      completion_tolerance:="${wc_completion_tolerance}" \
      completion_hold_s:="${wc_completion_hold_s}" \
      max_runtime_s:="${wc_max_runtime_s}" \
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
      interest_enabled:="${wc_interest_enabled}" \
      interest_gate_enabled:="${wc_interest_gate_enabled}" \
      footprint_rho_warn:="${wc_footprint_rho_warn}" \
      footprint_rho_stop:="${wc_footprint_rho_stop}" \
      footprint_gate_min_scale:="${wc_footprint_gate_min_scale}" \
      footprint_forbidden_stop_enabled:="${wc_footprint_forbidden_stop_enabled}" \
      adp_enabled:="${run_adp_enabled}" \
      adp_model:="${adp_model}" \
      lambda_adp:="${lambda_adp}" \
      lambda_adp_corridor:="${lambda_adp_corridor}" \
      lambda_adp_terminal:="${lambda_adp_terminal}" \
      mpc_use_adp_terminal:="${mpc_use_adp_terminal}" \
      adp_post_scale_enabled:="${adp_post_scale_enabled}" \
      adp_min_scale:="${adp_min_scale}" \
      adp_debug:="${adp_debug}" \
      2>&1 | tee "${log_path}"
  else
    roslaunch stsm_madp "${action_launch}" \
      baseline:="${baseline_arg}" \
      arm_interest_enabled:="${arm_interest_enabled}" \
      arm_interest_gate_enabled:="${arm_interest_gate_enabled}" \
      arm_interest_rho_warn:="${arm_interest_rho_warn}" \
      arm_interest_rho_stop:="${arm_interest_rho_stop}" \
      arm_interest_gate_min_scale:="${arm_interest_gate_min_scale}" \
      adp_enabled:="${run_adp_enabled}" \
      adp_model:="${adp_model}" \
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
    "${status}" "${exit_code}" "${tmp_metrics_csv}" "${tmp_traj_csv}" \
    "${log_path}" >> "${manifest_csv}"
  if [ "${exit_code}" -ne 0 ]; then
    failed=1
    echo "ERROR: roslaunch failed for ${target}/${mode}, see ${log_path}" >&2
    return "${exit_code}"
  fi
  check_env
  commit_run "${metrics_target}" "${mode}" "${tmp_metrics_csv}" "${tmp_traj_csv}" \
    "${metrics_csv}" "${traj_csv}"
  check_env
}

catkin_make -C /home/sun/LLL/catkin_ws

python3 -B "${pkg_dir}/scripts/organize_results.py" \
  --run-id "${run_id}" \
  --purpose "${PURPOSE:-}" \
  --stage "${STAGE:-}" \
  --status running \
  --update-latest

if [ "${target_filter}" = "all" ] || [ "${target_filter}" = "arm" ]; then
  launch_env arm_view.launch "${arm_wait_s}"
  run_one arm baseline
  run_one arm stsm
fi

if [ "${target_filter}" = "all" ] || [ "${target_filter}" = "wheelchair" ]; then
  echo
  echo "=== switching to wheelchair scene; the arm Gazebo/RViz window will close here ==="
  launch_env wheelchair_view.launch "${wc_wait_s}"
  run_one wheelchair baseline
  run_one wheelchair stsm
fi

cleanup_env

if [ "${plot}" = "true" ]; then
  python3 -B "${pkg_dir}/scripts/plot_results.py" \
    --results "${run_root}" \
    --out "${pictures_dir}"
  {
    echo "# Figure Sources"
    echo
    echo "- Run ID: ${run_id}"
    echo "- Source run: ${run_root}"
    echo "- Generated at: $(date -Iseconds)"
    echo "- Metrics summary: ${results_dir}/summary/all_metrics.csv"
    echo "- Plot output: ${pictures_dir}"
  } > "${pictures_dir}/figure_source.md"
fi

python3 -B "${pkg_dir}/scripts/collect_metrics.py" \
  --results-root "${results_dir}" \
  --out "${results_dir}/summary/all_metrics.csv"

python3 -B "${pkg_dir}/scripts/organize_results.py" \
  --run-id "${run_id}" \
  --purpose "${PURPOSE:-}" \
  --stage "${STAGE:-}" \
  --status ok \
  --update-latest

find "${pkg_dir}" -type f -name '*.pyc' -delete
find "${pkg_dir}" -type d -name '__pycache__' -empty -delete

echo
echo "Done."
echo "Run temp archive: ${run_root}"
echo "Manifest: ${manifest_csv}"
echo "Results:  ${results_dir}"
echo "Pictures: ${pictures_dir}"
