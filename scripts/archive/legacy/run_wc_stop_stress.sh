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
  RUN_ID=wc_stop_stress_check GUI=true run_wc_stop_stress.sh

Runs an isolated wheelchair STOP stress test. Results are written under:
  results/runs/<run_id>/wheelchair_stop_stress/
EOF
  exit 0
fi

pkg_dir="$(rospack find stsm_madp 2>/dev/null || true)"
if [ -z "${pkg_dir}" ]; then
  pkg_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
results_dir="${pkg_dir}/results"
run_id="${RUN_ID:-wc_stop_stress_$(date +%Y%m%d_%H%M%S)}"
gui="${GUI:-true}"
wait_s="${WC_WAIT_S:-12}"
clean_env="${CLEAN_ENV:-true}"
stress_rho_warn="${STRESS_RHO_WARN:-0.8}"
stress_rho_stop="${STRESS_RHO_STOP:-1.0}"
stress_min_scale="${STRESS_MIN_SCALE:-0.20}"
stress_start_pose="${STRESS_START_POSE:-[0.17, 0.85, -2.78]}"
max_stop_s="${MAX_STOP_S:-5.0}"
run_dir="${results_dir}/runs/${run_id}/wheelchair_stop_stress"
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
  echo "cleaning stale wheelchair/Gazebo processes (CLEAN_ENV=true)..."
  pkill -INT -f "roslaunch stsm_madp wheelchair_view.launch" >/dev/null 2>&1 || true
  pkill -INT -f "roslaunch stsm_madp wheelchair_action.launch" >/dev/null 2>&1 || true
  pkill -INT -f "stsm_metrics_wc_stop_stress" >/dev/null 2>&1 || true
  pkill -INT -f "stsm_wheelchair" >/dev/null 2>&1 || true
  pkill -INT -f "stsm_social_field_viz" >/dev/null 2>&1 || true
  pkill -INT -f "wc_controller_spawner" >/dev/null 2>&1 || true
  pkill -INT -f "gazebo_ros/spawn_model.*wheelchair" >/dev/null 2>&1 || true
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

check_wheelchair_env() {
  python -B - <<'PY'
import sys
import time
import rospy
from gazebo_msgs.msg import ModelStates

rospy.init_node("stsm_wc_env_check", anonymous=True, disable_signals=True)
deadline = time.time() + 20.0
last_names = []
while time.time() < deadline and not rospy.is_shutdown():
    try:
        msg = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=2.0)
        last_names = list(msg.name)
        if "wheelchair" in last_names:
            sys.exit(0)
    except Exception:
        pass
sys.stderr.write("wheelchair model not available in /gazebo/model_states; last names={0}\n".format(last_names))
sys.exit(1)
PY
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

echo "=== wheelchair STOP stress ==="
echo "run_id: ${run_id}"
echo "metrics: ${metrics_csv}"
echo "trajectory: ${traj_csv}"
echo "action log: ${log_path}"
echo "environment log: ${env_log}"
echo "stress gate: rho_warn=${stress_rho_warn}, rho_stop=${stress_rho_stop}, min_scale=${stress_min_scale}"
echo "stress start_pose: ${stress_start_pose}"
echo "max allowed first_stop_time_s: ${max_stop_s}"

cleanup_stale_wheelchair_env
roslaunch stsm_madp wheelchair_view.launch gui:="${gui}" >"${env_log}" 2>&1 &
env_pid="$!"
sleep "${wait_s}"
if ! kill -0 "${env_pid}" >/dev/null 2>&1; then
  echo "ERROR: wheelchair environment exited unexpectedly" >&2
  tail -100 "${env_log}" >&2 || true
  exit 1
fi
if ! check_wheelchair_env; then
  echo "ERROR: wheelchair model did not spawn correctly" >&2
  tail -120 "${env_log}" >&2 || true
  exit 1
fi

rosparam set /stsm/run_id "'${run_id}'"
rosrun stsm_madp metrics_node.py \
  _target:=wheelchair \
  _out:="${metrics_csv}" \
  _traj_out:="${traj_csv}" \
  _controller_version:=gate_v1_stress \
  __name:=stsm_metrics_wc_stop_stress &
metrics_pid="$!"

sleep 1.0
set +e
roslaunch stsm_madp wheelchair_action.launch \
  baseline:=true \
  rho_warn:="${stress_rho_warn}" \
  rho_stop:="${stress_rho_stop}" \
  gate_min_scale:="${stress_min_scale}" \
  abort_on_stop:=true \
  start_pose:="${stress_start_pose}" \
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
    mrows = list(csv.DictReader(f))
with open(traj_path, "r") as f:
    trows = list(csv.DictReader(f))
if not mrows:
    raise SystemExit("wheelchair STOP stress failed: empty metrics")
if not trows:
    raise SystemExit("wheelchair STOP stress failed: empty trajectory")
m = mrows[-1]
last = trows[-1]
if m.get("stop_triggered") != "1":
    raise SystemExit("wheelchair STOP stress failed: stop_triggered != 1")
if m.get("success_safe") != "0":
    raise SystemExit("wheelchair STOP stress failed: success_safe != 0")
if int(float(m.get("gate_stop_count") or 0)) < 1:
    raise SystemExit("wheelchair STOP stress failed: gate_stop_count < 1")
if last.get("gate_state") != "STOP" or str(last.get("gate_stop", "0")) != "1":
    raise SystemExit("wheelchair STOP stress failed: final traj row is not STOP")
valid_reasons = ("risk_stop", "center:risk_stop")
if m.get("stop_reason") not in valid_reasons or last.get("gate_reason") not in valid_reasons:
    raise SystemExit("wheelchair STOP stress failed: stop_reason is not risk_stop")
first_stop = float(m.get("first_stop_time_s") or 1e9)
if first_stop > max_stop_s:
    raise SystemExit(
        "wheelchair STOP stress failed: first_stop_time_s={} > {}".format(
            first_stop, max_stop_s))
print("PASS: wheelchair STOP stress triggered at t={}".format(
    m.get("first_stop_time_s")))
PY

exit "${exit_code}"
