#!/usr/bin/env python3
"""Calibrate one ADP critic from recorded, executed online transitions."""
import argparse
import json
import os
import sys
import time

PACKAGE_SRC = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "src"))
if PACKAGE_SRC not in sys.path:
    sys.path.insert(0, PACKAGE_SRC)

from stsm_madp.adp import (  # noqa: E402
    ADPCritic, fit_critic_from_transition_records)


def main():
    parser = argparse.ArgumentParser(
        description="Fit a critic to real ADP transition cost-to-go records.")
    parser.add_argument("--diagnostics", nargs="+", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--robot", required=True, choices=("arm", "wheelchair"))
    parser.add_argument("--ridge", type=float, default=1e-4)
    args = parser.parse_args()

    template = ADPCritic.load_yaml(args.template)
    records = []
    sources = []
    for path in args.diagnostics:
        with open(path, "r") as handle:
            payload = json.load(handle)
        if str(payload.get("robot", "")) != args.robot:
            continue
        records.extend(payload.get("records", []))
        sources.append(os.path.abspath(path))
    critic, summary = fit_critic_from_transition_records(
        records, template, gamma=template.gamma, ridge=args.ridge)
    critic.metadata.update({
        "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "calibration_method": "real_online_transition_cost_to_go",
        "calibration_robot": args.robot,
        "calibration_sources": sources,
        "seed_critic_target_mismatch": True,
    })
    critic.metadata.update(summary)
    critic.save_yaml(args.out)
    print("calibrated {} samples={} episodes={} target_mean={:.6f}".format(
        args.out, summary["sample_count"], summary["episode_count"],
        summary["target_mean"]))


if __name__ == "__main__":
    main()
