#!/usr/bin/env python3
"""Offline lineage audit for a saved Wheelchair MPC reference trace.

This tool is deliberately read-only.  In particular, it does not construct a
new corridor, alter a SafetyContext, or run ROS/Gazebo.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _load(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _point(row):
    return np.array([float(row["x"]), float(row["y"])], float)


def _stage_from_trace(points):
    invalid = [int(row["index"]) for row in points
               if not bool(row.get("hard_valid", False))]
    manifold = [int(row["index"]) for row in points
                if not bool(row.get("manifold_valid", False))]
    worst = min(points, key=lambda row: float(row.get("clearance", 0.0)))
    return {
        "point_count": len(points),
        "min_clearance": float(worst.get("clearance", 0.0)),
        "max_risk": max(float(row.get("risk", 0.0)) for row in points),
        "first_hard_invalid_index": invalid[0] if invalid else None,
        "hard_invalid_indices": invalid,
        "manifold_violation_count": len(manifold),
        "manifold_violation_indices": manifold,
        "worst_index": int(worst["index"]),
        "worst_point": [float(worst["x"]), float(worst["y"])],
    }


def _lineage(point, planning):
    """Map a logged horizon point back to a final point or its segment."""
    final = np.asarray([_point(row) for row in planning], float)
    distances = np.linalg.norm(final - point, axis=1)
    exact = int(np.argmin(distances))
    if float(distances[exact]) <= 1e-8:
        return {"planning_final_index": exact, "generated_by": "identity",
                "coordinate_changed": False}
    best = None
    for idx, (a, b) in enumerate(zip(final[:-1], final[1:])):
        segment = b - a
        denom = float(np.dot(segment, segment))
        alpha = 0.0 if denom <= 1e-12 else float(np.clip(
            np.dot(point - a, segment) / denom, 0.0, 1.0))
        projected = a + alpha * segment
        distance = float(np.linalg.norm(point - projected))
        if best is None or distance < best[0]:
            best = (distance, idx, alpha)
    return {
        "planning_final_index": None,
        "planning_segment_start_index": int(best[1]),
        "planning_segment_end_index": int(best[1] + 1),
        "segment_alpha": float(best[2]),
        "generated_by": "horizon_reference_interpolation",
        "coordinate_changed": False,
        "distance_to_source_segment": float(best[0]),
    }


def run(run_dir, output_path):
    trace = _load(os.path.join(run_dir, "candidate_path_trace.json"))
    candidate = next(item for item in trace["candidates"]
                     if item["candidate_id"] == "wheelchair_c0001")
    planning = list(candidate["final_reference"]["points"])
    diagnostics = _load(os.path.join(run_dir, "mpc_diagnostics.json"))
    with open(os.path.join(run_dir, "mpc_reference_path.csv"), newline="") as handle:
        flat = list(csv.DictReader(handle))

    # Formal diagnostics intentionally downsample the flattened, multi-cycle
    # trace.  Reproduce its index map exactly, so an audit index cannot be
    # mistaken for a spatial-path index.
    sampled_indices = np.linspace(0, len(flat) - 1, 160).astype(int).tolist()
    audit = diagnostics["reference_safety_audit"]
    audit_records = list(audit.get("records", []))
    invalid = [row for row in audit_records if bool(row.get("violation", False))]
    first = invalid[0] if invalid else None
    worst_index = int(audit.get("worst_index", -1))
    worst_flat_index = (sampled_indices[worst_index]
                        if 0 <= worst_index < len(sampled_indices) else -1)
    worst_row = flat[worst_flat_index] if worst_flat_index >= 0 else {}
    worst_point = _point(worst_row) if worst_row else np.zeros(2, float)

    output = {
        "run_dir": os.path.abspath(run_dir),
        "planning_final": _stage_from_trace(planning),
        "mpc_input_reference": {
            "point_count": None,
            "equal_to_planning_final_reference": False,
            "reason": "MPC receives one per-cycle rolling horizon, not the 54-point path as one object.",
        },
        "transform_stages": [{
            "stage": "horizon_reference_interpolation_and_window_extraction",
            "point_count": len(flat),
            "representation": "all control-cycle horizon points accumulated in mpc_reference_path.csv",
            "geometry_change": "new samples on final-reference segments; no alignment translation is applied",
        }, {
            "stage": "formal_reference_safety_audit_input",
            "point_count": len(audit_records),
            "source_flat_point_count": len(flat),
            "min_clearance": float(audit.get("min_clearance", 0.0)),
            "max_risk": None,
            "first_hard_invalid_index": (int(first["index"]) if first else None),
            "hard_invalid_indices": [int(row["index"]) for row in invalid],
            "manifold_violation_count": int(audit.get("violation_count", 0)),
            "worst_index": worst_index,
            "worst_point": worst_point.tolist(),
        }],
        "first_invalid_stage": "formal_reference_safety_audit_input" if invalid else None,
        "worst_runtime_point": {
            "flat_index": worst_flat_index,
            "cycle_id": int(worst_row.get("solve_index", -1)),
            "horizon_index": int(worst_row.get("horizon_point_index", -1)),
            "global_reference_index": None,
            "x": float(worst_point[0]), "y": float(worst_point[1]),
            "clearance": float(audit_records[worst_index].get("min_clearance", 0.0)),
            "lineage": _lineage(worst_point, planning),
        },
    }
    output["worst_runtime_point"]["global_reference_index"] = (
        output["worst_runtime_point"]["lineage"].get("planning_final_index"))
    with open(output_path, "w") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=os.path.join(
        ROOT, "results", "runs", "20260901_R001", "wheelchair", "stsm"))
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    output = args.output or os.path.join(args.run_dir, "mpc_reference_lineage_audit.json")
    result = run(args.run_dir, output)
    print(json.dumps({"output": output,
                      "first_invalid_stage": result["first_invalid_stage"],
                      "worst_runtime_point": result["worst_runtime_point"]},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
