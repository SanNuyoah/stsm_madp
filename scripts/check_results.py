#!/usr/bin/env python3
from __future__ import print_function

import argparse
import json
import os
import sys

sys.dont_write_bytecode = True

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RUN = os.path.join(ROOT, "results", "run")
ROBOTS = ("wheelchair", "arm")
VARIANTS = ("baseline", "stsm")
REQUIRED = (
    "traj.csv",
    "metrics.csv",
    "metrics.json",
    "decision_trace.json",
    "mpc_diagnostics.json",
)
BASELINE_FORBIDDEN = (
    "critical_point_association.json",
    "topology_constraint.json",
    "topology_tube.json",
)
ROOT_DUPLICATES = (
    "traj.csv",
    "metrics.csv",
    "metrics.json",
    "mpc_diagnostics.json",
    "decision_trace.json",
    "topology_constraint.json",
)


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=DEFAULT_RUN)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    errors = []
    summary_dir = os.path.join(os.path.dirname(args.run_dir), "summary")
    if os.path.exists(summary_dir):
        errors.append("derived summary directory exists: {}".format(summary_dir))
    figures = os.path.join(os.path.dirname(args.run_dir), "figures")
    if os.path.isdir(figures):
        for base, _dirs, files in os.walk(figures):
            for name in files:
                if os.path.splitext(name)[1].lower() not in (".png", ".pdf", ".svg"):
                    errors.append("non-figure file in figures: {}".format(
                        os.path.join(base, name)))
    for robot in ROBOTS:
        robot_dir = os.path.join(args.run_dir, robot)
        for name in ROOT_DUPLICATES:
            path = os.path.join(robot_dir, name)
            if os.path.exists(path):
                errors.append("root duplicate exists: {}".format(path))
        for variant in VARIANTS:
            base = os.path.join(robot_dir, variant)
            if not os.path.isdir(base):
                errors.append("missing variant dir: {}".format(base))
                continue
            for name in REQUIRED:
                path = os.path.join(base, name)
                if not os.path.exists(path):
                    errors.append("missing required file: {}".format(path))
                elif os.path.getsize(path) == 0:
                    errors.append("empty required file: {}".format(path))
            if variant == "baseline":
                for name in BASELINE_FORBIDDEN:
                    path = os.path.join(base, name)
                    if os.path.exists(path):
                        errors.append("baseline unused file exists: {}".format(path))
            for name in (
                    "metrics.json", "decision_trace.json",
                    "mpc_diagnostics.json", "topology_constraint.json"):
                path = os.path.join(base, name)
                if not os.path.exists(path):
                    continue
                payload = load_json(path)
                if isinstance(payload, dict):
                    if str(payload.get("variant", "")) != variant:
                        errors.append(
                            "{} variant mismatch: expected {}".format(
                                path, variant))
                    robot_value = str(
                        payload.get("robot", payload.get("target", "")))
                    if robot_value and robot_value != robot:
                        errors.append(
                            "{} robot mismatch: expected {}".format(
                                path, robot))

    report = {
        "result_structure_valid": not errors,
        "errors": errors,
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("result_structure_valid={}".format(
            str(report["result_structure_valid"]).lower()))
        for error in errors:
            print("ERROR: {}".format(error))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
