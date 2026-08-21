#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 {arm|wheelchair} {baseline|stsm}" >&2
  echo "Start this script before running the matching action node; press Ctrl+C after the run finishes." >&2
}

if [ "$#" -ne 2 ]; then
  usage
  exit 2
fi

target="$1"
mode="$2"

case "$target" in
  arm)
    metric_target="arm"
    csv_prefix="arm"
    metrics_name="arm_metrics.csv"
    ;;
  wheelchair|wc)
    metric_target="wheelchair"
    csv_prefix="wc"
    metrics_name="wc_metrics.csv"
    ;;
  *)
    usage
    exit 2
    ;;
esac

case "$mode" in
  baseline|stsm)
    ;;
  *)
    usage
    exit 2
    ;;
esac

if [ -f /opt/ros/melodic/setup.bash ]; then
  source /opt/ros/melodic/setup.bash
fi
if [ -f /home/sun/elfin_assist_ws/devel/setup.bash ]; then
  source /home/sun/elfin_assist_ws/devel/setup.bash
fi
source /home/sun/LLL/catkin_ws/devel/setup.bash

pkg_dir="$(rospack find stsm_madp)"
results_dir="${pkg_dir}/results"
mkdir -p "${results_dir}"

out_csv="${results_dir}/${metrics_name}"
traj_csv="${results_dir}/${csv_prefix}_${mode}_traj.csv"

echo "Recording ${target}/${mode}"
echo "metrics:    ${out_csv}"
echo "trajectory: ${traj_csv} (overwritten on each run)"
echo "Press Ctrl+C after the matching action node finishes so metrics_node writes the CSV files."

rosrun stsm_madp metrics_node.py \
  _target:="${metric_target}" \
  _out:="${out_csv}" \
  _traj_out:="${traj_csv}" \
  __name:=stsm_metrics
