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

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  cat <<'EOF'
Usage:
  RUN_ID=wc_footprint_gate_stress GUI=true run_wc_footprint_gate_stress.sh

Runs an isolated wheelchair footprint-gate STOP stress test. Results are written under:
  results/runs/<run_id>/wheelchair_footprint_gate_stress/
EOF
  exit 0
fi

pkg_dir="$(rospack find stsm_madp 2>/dev/null || true)"
if [ -z "${pkg_dir}" ]; then
  pkg_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

results_dir="${pkg_dir}/results"
run_id="${RUN_ID:-wc_footprint_gate_stress_$(date +%Y%m%d_%H%M%S)}"
gui="${GUI:-true}"
wait_s="${WC_WAIT_S:-12}"
clean_env="${CLEAN_ENV:-true}"
start_pose="${START_POSE:-[-0.25, 0.66, -2.70]}"
footprint_warn="${FOOTPRINT_RHO_WARN:-3.0}"
footprint_stop="${FOOTPRINT_RHO_STOP:-4.5}"
max_stop_s="${MAX_STOP_S:-5.0}"
run_dir="${results_dir}/runs/${run_id}/wheelchair_footprint_gate_stress"
mkdir -p "${run_dir}"

metrics_csv="${run_dir}/metrics.csv"
traj_csv="${run_dir}/traj.csv"
log_path="${run_dir}/roslaunch.log"
env_log="${run_dir}/wheelchair_view_env.log"
metrics_pid=""
env_pid=""

cleanup_stale_wheelchair_env() {
  if [ "${clean_env}" != "true" ]; then
    return
  fi
  pkill -INT -f "roslaunch stsm_madp wheelchair_view.launch" >/dev/null 2>&1 || true
  pkill -INT -f "roslaunch stsm_madp wheelchair_action.launch" >/dev/null 2>&1 || true
  pkill -INT -f "stsm_metrics_wc_footprint_gate_stress" >/dev/null 2>&1 || true
  pkill -INT -f "stsm_wheelchair" >/dev/null 2>&1 || true
  pkill -INT -f "wc_controller_spawner" >/dev/null 2>&1 || true
  pkill -INT -f "gzserver.*eldercare_room.world" >/dev/null 2>&1 || true
  pkill -INT -f "gzclient" >/dev/null 2>&1 || true
  sleep 2.0
  pkill -TERM -f "roslaunch stsm_madp wheelchair_view.launch" >/dev/null 2>&1 || true
  pkill -TERM -f "roslaunch stsm_madp wheelchair_action.launch" >/dev/null 2>&1 || true
  pkill -TERM -f "stsm_wheelchair" >/dev/null 2>&1 || true
  pkill -TERM -f "wc_controller_spawner" >/dev/null 2>&1 || true
  pkill -TERM -f "gzserver.*eldercare_room.world" >/dev/null 2>&1 || true
  pkill -TERM -f "gzclient" >/dev/null 2>&1 || true
  sleep 1.0
}

cleanup() {
  if [ -n "${metrics_pid}" ] && kill -0 "${metrics_pid}" >/dev/null 2>&1; then
    kill -INT "${metrics_pid}" >/dev/null 2>&1 || true
    wait "${metrics_pid}" >/dev/null 2>&1 || true
  fi
  if [ -n "${env_pid}" ] && kill -0 "${env_pid}" >/dev/null 2>&1; then
    kill -INT "${env_pid}" >/dev/null 2>&1 || true
    wait "${env_pid}" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

echo "=== wheelchair footprint gate stress ==="
echo "run_id: ${run_id}"
echo "start_pose: ${start_pose}"
echo "footprint gate: warn=${footprint_warn}, stop=${footprint_stop}"
echo "metrics: ${metrics_csv}"
echo "trajectory: ${traj_csv}"

cleanup_stale_wheelchair_env
roslaunch stsm_madp wheelchair_view.launch gui:="${gui}" >"${env_log}" 2>&1 &
env_pid="$!"
sleep "${wait_s}"
if ! kill -0 "${env_pid}" >/dev/null 2>&1; then
  echo "ERROR: wheelchair environment exited unexpectedly" >&2
  tail -100 "${env_log}" >&2 || true
  exit 1
fi

rosparam set /stsm/run_id "'${run_id}'"
rosrun stsm_madp metrics_node.py \
  _target:=wheelchair \
  _out:="${metrics_csv}" \
  _traj_out:="${traj_csv}" \
  _controller_version:=footprint_gate_stress \
  __name:=stsm_metrics_wc_footprint_gate_stress &
metrics_pid="$!"

sleep 1.0
set +e
roslaunch stsm_madp wheelchair_action.launch \
  baseline:=true \
  interest_enabled:=true \
  interest_gate_enabled:=true \
  footprint_rho_warn:="${footprint_warn}" \
  footprint_rho_stop:="${footprint_stop}" \
  footprint_gate_min_scale:=0.20 \
  footprint_forbidden_stop_enabled:=true \
  start_pose:="${start_pose}" \
  2>&1 | tee "${log_path}"
exit_code="${PIPESTATUS[0]}"
set -e

sleep 1.0
if [ -n "${metrics_pid}" ] && kill -0 "${metrics_pid}" >/dev/null 2>&1; then
  kill -INT "${metrics_pid}" >/dev/null 2>&1 || true
  wait "${metrics_pid}" >/dev/null 2>&1 || true
fi
metrics_pid=""

python3 -B - "${metrics_csv}" "${traj_csv}" "${max_stop_s}" <<'PY'
import csv
import sys

metrics_path, traj_path, max_stop_s = sys.argv[1:4]
max_stop_s = float(max_stop_s)
with open(metrics_path, "r") as f:
    m = list(csv.DictReader(f))[-1]
with open(traj_path, "r") as f:
    trows = list(csv.DictReader(f))
last = trows[-1] if trows else {}
if m.get("stop_triggered") != "1":
    raise SystemExit("footprint gate stress failed: stop_triggered != 1")
if m.get("success_safe") != "0":
    raise SystemExit("footprint gate stress failed: success_safe != 0")
if int(float(m.get("footprint_stop_count") or 0)) < 1:
    raise SystemExit("footprint gate stress failed: footprint_stop_count < 1")
if last.get("gate_state") != "STOP" or str(last.get("gate_stop", "0")) != "1":
    raise SystemExit("footprint gate stress failed: final traj row is not STOP")
if last.get("gate_source") not in ("footprint", "footprint_forbidden"):
    raise SystemExit("footprint gate stress failed: gate_source={}".format(last.get("gate_source")))
if not (m.get("stop_reason") or "").startswith("footprint:"):
    raise SystemExit("footprint gate stress failed: stop_reason={}".format(m.get("stop_reason")))
first_stop = float(m.get("first_stop_time_s") or 1e9)
if first_stop > max_stop_s:
    raise SystemExit("footprint gate stress failed: first_stop_time_s={} > {}".format(first_stop, max_stop_s))
print("PASS: wheelchair footprint gate STOP source={} t={}".format(
    last.get("gate_source"), m.get("first_stop_time_s")))
PY

exit "${exit_code}"
