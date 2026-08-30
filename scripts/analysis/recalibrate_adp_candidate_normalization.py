#!/usr/bin/env python3
"""Recenter existing candidate features from executed runtime corridors."""
import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

PACKAGE_SRC = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "src"))
if PACKAGE_SRC not in sys.path:
    sys.path.insert(0, PACKAGE_SRC)

from stsm_madp.adp import (  # noqa: E402
    ADPCritic, CANDIDATE_FEATURE_NAMES, candidate_feature_values)


def _runtime_candidate_values(path):
    with open(path, "r") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        return []
    values = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw = dict(row.get("candidate_adp_features_raw", {}) or {})
        if not raw:
            raw = dict(row.get("adp_ranking_audit", {}).get(
                "candidate_adp_features_raw", {}) or {})
        normalized, missing = candidate_feature_values(raw)
        if not any(missing.values()):
            values.append(normalized)
    return values


def _feature_stats(samples):
    stats = {}
    for name in CANDIDATE_FEATURE_NAMES:
        values = np.asarray([sample[name] for sample in samples], float)
        if not len(values):
            raise ValueError("no runtime samples for %s" % name)
        center = float(np.median(values))
        p5, p95 = [float(value) for value in np.percentile(values, [5, 95])]
        scale = float((p95 - p5) / 1.349)
        if scale <= 1e-6:
            # Constant runtime quantities normalize to zero around their
            # measured center; do not retain a stale, unrelated tiny scale.
            scale = 1.0
        stats[name] = {
            "mean": center,
            "std": scale,
            "p5": p5,
            "p95": p95,
            "sample_count": int(len(values)),
            "source": "runtime_selected_candidate_context",
        }
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--candidate-corridors", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    samples = []
    sources = []
    for path in args.candidate_corridors:
        values = _runtime_candidate_values(path)
        if values:
            samples.extend(values)
            sources.append(os.path.normpath(path))
    if not samples:
        raise ValueError("no complete runtime candidate contexts")
    with open(args.template, "r") as handle:
        template_data = yaml.safe_load(handle) or {}
    critic = ADPCritic.load_yaml(args.template)
    # Do not materialize unrelated default learning parameters into the seed.
    critic.learning_config = dict(template_data.get("learning", {}) or {})
    stats = _feature_stats(samples)
    # Keep the learned weights untouched.  This is a normalization-only
    # recalibration, deliberately not a refit or a coordinate-preserving
    # parameter transform.
    for name, values in stats.items():
        index = critic.feature_names.index(name)
        critic.mean[index] = values["mean"]
        critic.std[index] = values["std"]
    critic.metadata["candidate_feature_normalization"] = stats
    critic.metadata["candidate_feature_normalization_sources"] = sources
    critic.metadata["candidate_feature_normalization_recalibrated_at"] = (
        time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    critic.save_yaml(args.out)
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
