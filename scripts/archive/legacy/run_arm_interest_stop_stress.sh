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
  RUN_ID=arm_interest_stop_stress GUI=false run_arm_interest_stop_stress.sh

Runs an isolated arm multi-interest-gate STOP stress test. Results are written under:
  results/runs/<run_id>/arm_interest_stop_stress/
EOF
  exit 0
fi

pkg_dir="$(rospack find stsm_madp 2>/dev/null || true)"
if [ -z "${pkg_dir}" ]; then
  pkg_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

results_dir="${pkg_dir}/results"
run_id="${RUN_ID:-arm_interest_stop_stress_$(date +%Y%m%d_%H%M%S)}"
gui="${GUI:-true}"
wait_s="${ARM_WAIT_S:-15}"
clean_env="${CLEAN_ENV:-true}"
mode="${MODE:-stsm}"
baseline_arg="false"
if [ "${mode}" = "baseline" ]; then
  baseline_arg="true"
fi
interest_warn="${ARM_INTEREST_RHO_WARN:-1.2}"
interest_stop="${ARM_INTEREST_RHO_STOP:-1.6}"
max_stop_s="${MAX_STOP_S:-25.0}"
run_dir="${results_dir}/runs/${run_id}/arm_interest_stop_stress"
mkdir -p "${run_dir}"

metrics_csv="${run_dir}/metrics.csv"
traj_csv="${run_dir}/traj.csv"
log_path="${run_dir}/roslaunch.log"
env_log="${run_dir}/arm_view_env.log"
metrics_pid=""
env_pid=""

cleanup_stale_arm_env() {
  if [ "${clean_env}" != "true" ]; then
    return
  fi
  pkill -INT -f "roslaunch stsm_madp arm_view.launch" >/dev/null 2>&1 || true
  pkill -INT -f "roslaunch stsm_madp arm_action.launch" >/dev/null 2>&1 || true
  pkill -INT -f "stsm_metrics_arm_interest_stop_stress" >/dev/null 2>&1 || true
  pkill -INT -f "stsm_handover" >/dev/null 2>&1 || true
  pkill -INT -f "gzserver.*eldercare_room.world" >/dev/null 2>&1 || true
  pkill -INT -f "gzclient" >/dev/null 2>&1 || true
  sleep 2.0
  pkill -TERM -f "roslaunch stsm_madp arm_view.launch" >/dev/null 2>&1 || true
  pkill -TERM -f "roslaunch stsm_madp arm_action.launch" >/dev/null 2>&1 || true
  pkill -TERM -f "stsm_handover" >/dev/null 2>&1 || true
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

echo "=== arm interest gate STOP stress ==="
echo "run_id: ${run_id}"
echo "mode: ${mode}"
echo "arm interest gate: warn=${interest_warn}, stop=${interest_stop}"
echo "metrics: ${metrics_csv}"
echo "trajectory: ${traj_csv}"

cleanup_stale_arm_env
roslaunch stsm_madp arm_view.launch gui:="${gui}" >"${env_log}" 2>&1 &
env_pid="$!"
sleep "${wait_s}"
if ! kill -0 "${env_pid}" >/dev/null 2>&1; then
  echo "ERROR: arm environment exited unexpectedly" >&2
  tail -100 "${env_log}" >&2 || true
  exit 1
fi

rosparam set /stsm/run_id "'${run_id}'"
rosrun stsm_madp metrics_node.py \
  _target:=arm \
  _out:="${metrics_csv}" \
  _traj_out:="${traj_csv}" \
  _controller_version:=arm_interest_stop_stress \
  _interest_required:=true \
  __name:=stsm_metrics_arm_interest_stop_stress &
metrics_pid="$!"

sleep 1.0
set +e
roslaunch stsm_madp arm_action.launch \
  baseline:="${baseline_arg}" \
  arm_interest_enabled:=true \
  arm_interest_gate_enabled:=true \
  arm_interest_rho_warn:="${interest_warn}" \
  arm_interest_rho_stop:="${interest_stop}" \
  arm_interest_gate_min_scale:=0.20 \
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
    raise SystemExit("arm interest stress failed: stop_triggered != 1")
if m.get("success_safe") != "0":
    raise SystemExit("arm interest stress failed: success_safe != 0")
if int(float(m.get("arm_interest_stop_count") or 0)) < 1:
    raise SystemExit("arm interest stress failed: arm_interest_stop_count < 1")
if last.get("gate_state") != "STOP" or str(last.get("gate_stop", "0")) != "1":
    raise SystemExit("arm interest stress failed: final traj row is not STOP")
if last.get("arm_gate_source") != "arm_interest":
    raise SystemExit("arm interest stress failed: arm_gate_source={}".format(last.get("arm_gate_source")))
if not (m.get("stop_reason") or "").startswith("arm_interest:risk_stop"):
    raise SystemExit("arm interest stress failed: stop_reason={}".format(m.get("stop_reason")))
first_stop = float(m.get("first_stop_time_s") or 1e9)
if first_stop > max_stop_s:
    raise SystemExit("arm interest stress failed: first_stop_time_s={} > {}".format(first_stop, max_stop_s))
if int(float(m.get("arm_interest_slow_count") or 0)) != 0 and first_stop <= 0.01:
    raise SystemExit("arm interest stress failed: immediate STOP counted as SLOW")
print("PASS: arm interest gate STOP source={} t={}".format(
    last.get("arm_gate_source"), m.get("first_stop_time_s")))
PY

exit "${exit_code}"
