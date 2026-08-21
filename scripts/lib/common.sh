#!/usr/bin/env bash

source_if_exists() {
  if [ -f "$1" ]; then
    # shellcheck disable=SC1090
    source "$1"
  fi
}

load_ros_env() {
  source_if_exists /opt/ros/melodic/setup.bash
  source_if_exists /home/sun/elfin_assist_ws/devel/setup.bash
  source_if_exists /home/sun/LLL/catkin_ws/devel/setup.bash
}

stsm_pkg_dir() {
  local pkg_dir
  pkg_dir="$(rospack find stsm_madp 2>/dev/null || true)"
  if [ -z "${pkg_dir}" ]; then
    pkg_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  fi
  printf '%s\n' "${pkg_dir}"
}

next_run_id() {
  local results_dir="$1"
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
