#!/usr/bin/env python3
"""Calibrate one ADP critic from recorded, executed online transitions."""
import argparse
import json
import os
import sys
import time
import numpy as np

PACKAGE_SRC = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "src"))
if PACKAGE_SRC not in sys.path:
    sys.path.insert(0, PACKAGE_SRC)

from stsm_madp.adp import (  # noqa: E402
    ADPCritic, candidate_feature_values, fit_critic_from_transition_records)


def _selected_candidate_features(diagnostics_path):
    ranking_path = os.path.join(os.path.dirname(diagnostics_path),
                                "candidate_ranking.json")
    if not os.path.isfile(ranking_path):
        return None, {}, ranking_path
    with open(ranking_path, "r") as handle:
        rows = json.load(handle)
    selected = next((row for row in rows if row.get("selected")), None)
    if not isinstance(selected, dict):
        return None, {}, ranking_path
    raw = dict(selected.get("candidate_adp_features_raw", {}) or {})
    if not raw:
        raw = dict(selected.get("adp_ranking_audit", {}).get(
            "candidate_context", {}) or {})
    if not raw:
        raw = dict(selected)
    if "mean_phi_on_path" in raw and "risk_mean" not in raw:
        raw["risk_mean"] = raw["mean_phi_on_path"]
    if "max_phi_on_path" in raw and "risk_max" not in raw:
        raw["risk_max"] = raw["max_phi_on_path"]
    values, missing = candidate_feature_values(raw)
    return str(selected.get("candidate_id") or selected.get("corridor_id") or ""), {
        "values": values, "missing": missing}, ranking_path


def _ranking_value_distribution(diagnostics_paths, critic):
    values = []
    for diagnostics_path in diagnostics_paths:
        _selected_id, _candidate, ranking_path = _selected_candidate_features(
            diagnostics_path)
        if not os.path.isfile(ranking_path):
            continue
        with open(ranking_path, "r") as handle:
            rows = json.load(handle)
        for row in rows if isinstance(rows, list) else []:
            audit = dict(row.get("adp_ranking_audit", {}) or {})
            vector = audit.get("feature_vector", []) or []
            if not vector:
                continue
            features = {item.get("feature_name"): item.get("raw_value", 0.0)
                        for item in vector if isinstance(item, dict)}
            raw = dict(row.get("candidate_adp_features_raw", {}) or {})
            if not raw:
                raw = dict(audit.get("candidate_context", {}) or {})
            if not raw:
                raw = dict(row)
            features.update(candidate_feature_values(raw)[0])
            value = critic.predict_detail(features)["raw"]
            if value == value and abs(value) != float("inf"):
                values.append(float(value))
    if not values:
        return {}
    array = np.asarray(values, float)
    center = float(np.median(array))
    q25, q75 = np.percentile(array, [25, 75])
    scale = max(float((q75 - q25) / 1.349), 1e-6)
    return {"ranking_value_center": center, "ranking_value_scale": scale,
            "ranking_value_p5": float(np.percentile(array, 5)),
            "ranking_value_p95": float(np.percentile(array, 95)),
            "ranking_value_sample_count": int(array.size)}


def main():
    parser = argparse.ArgumentParser(
        description="Fit a critic to real ADP transition cost-to-go records.")
    parser.add_argument("--diagnostics", nargs="+", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--robot", required=True, choices=("arm", "wheelchair"))
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--candidate-conditioned", action="store_true",
                        help="Attach the actually executed selected corridor summary.")
    args = parser.parse_args()

    template = ADPCritic.load_yaml(args.template)
    records = []
    sources = []
    excluded_sources = []
    missing_counts = {}
    for path in args.diagnostics:
        with open(path, "r") as handle:
            payload = json.load(handle)
        if str(payload.get("robot", "")) != args.robot:
            continue
        source_records = list(payload.get("records", []))
        if args.candidate_conditioned:
            selected_id, candidate, ranking_path = _selected_candidate_features(path)
            if not candidate:
                excluded_sources.append({"diagnostics": os.path.abspath(path),
                                         "ranking": ranking_path,
                                         "reason": "missing_selected_candidate"})
                continue
            enriched = []
            for record in source_records:
                if not record.get("updated", False):
                    continue
                if str(record.get("corridor_id", "")) != selected_id:
                    continue
                record = dict(record)
                for key in ("features_t", "features_next"):
                    features = dict(record.get(key, {}) or {})
                    features.update(candidate["values"])
                    record[key] = features
                enriched.append(record)
            if not enriched:
                excluded_sources.append({"diagnostics": os.path.abspath(path),
                                         "ranking": ranking_path,
                                         "reason": "no_executed_selected_transitions"})
                continue
            for name, missing in candidate["missing"].items():
                missing_counts[name] = missing_counts.get(name, 0) + int(missing)
            source_records = enriched
        records.extend(source_records)
        sources.append(os.path.abspath(path))
    critic, summary = fit_critic_from_transition_records(
        records, template, gamma=template.gamma, ridge=args.ridge)
    if args.candidate_conditioned:
        critic.critic_version = "linear_adp_candidate_conditioned_v2_calibrated"
    critic.metadata.update({
        "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "calibration_method": "real_online_transition_cost_to_go",
        "calibration_robot": args.robot,
        "calibration_sources": sources,
        "seed_critic_target_mismatch": True,
        "candidate_conditioned": bool(args.candidate_conditioned),
        "feature_schema_version": (
            "candidate_conditioned_v2" if args.candidate_conditioned else
            template.metadata.get("feature_schema_version", "")),
        "feature_count": len(critic.feature_names),
        "feature_names": list(critic.feature_names),
        "candidate_feature_missing_source_count": missing_counts,
        "excluded_sources": excluded_sources,
    })
    critic.metadata.update(summary)
    if args.candidate_conditioned:
        critic.metadata.update(_ranking_value_distribution(sources, critic))
    critic.save_yaml(args.out)
    print("calibrated {} samples={} episodes={} target_mean={:.6f}".format(
        args.out, summary["sample_count"], summary["episode_count"],
        summary["target_mean"]))


if __name__ == "__main__":
    main()
