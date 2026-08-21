#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

scenario=""
run_id=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --scenario)
      scenario="${2:-}"
      shift 2
      ;;
    --run-id)
      run_id="${2:-}"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage:
  bash scripts/run_stress.sh --scenario wc_stop [--run-id YYYYMMDD_R###]
  bash scripts/run_stress.sh --scenario wc_footprint_gate [--run-id YYYYMMDD_R###]
  bash scripts/run_stress.sh --scenario arm_interest_stop [--run-id YYYYMMDD_R###]
EOF
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [ -n "${run_id}" ]; then
  export RUN_ID="${run_id}"
fi

case "${scenario}" in
  wc_stop)
    exec bash "${script_dir}/legacy/run_wc_stop_stress.sh"
    ;;
  wc_footprint_gate)
    exec bash "${script_dir}/legacy/run_wc_footprint_gate_stress.sh"
    ;;
  arm_interest_stop)
    exec bash "${script_dir}/legacy/run_arm_interest_stop_stress.sh"
    ;;
  *)
    echo "missing or invalid --scenario: ${scenario}" >&2
    exit 2
    ;;
esac
