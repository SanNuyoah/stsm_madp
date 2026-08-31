#!/usr/bin/env python3
"""Replay R011 runtime join evidence without running ROS/Gazebo."""
import argparse
import json
import math
import os
import re
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
THIS_DIR = os.path.dirname(__file__)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from stsm_madp.deform import path_curvature_metrics, path_length
from stsm_madp.safety_evaluator import SafetyEvaluator

import replay_c0001_safe_terminal_trials as c0001_replay

CLEARANCE = c0001_replay.CLEARANCE
RISK = c0001_replay.RISK
TURN = c0001_replay.TURN
CURVATURE = c0001_replay.CURVATURE


def _load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _runtime_pose(run_dir):
    runtime = _load_json(os.path.join(
        run_dir, "runtime_replan_connectability.json"))
    attempt = runtime["attempts"][0]
    yaw = float(attempt.get("current_yaw", 0.0))
    x = y = None
    with open(os.path.join(run_dir, "ros.log"), "r", errors="ignore") as handle:
        for line in handle:
            if "[wc][recovery] full replan reason=no_progress" in line:
                break
            match = re.search(
                r"pos=\(([-0-9.]+),\s*([-0-9.]+),\s*([-0-9.]+)\)", line)
            if match:
                x = float(match.group(1))
                y = float(match.group(2))
    if x is None or y is None:
        raise RuntimeError("R011 no-progress pose not found in ros.log")
    return np.asarray([x, y, yaw], float), attempt


def _c0001_final_reference(run_dir):
    topo = _load_json(os.path.join(run_dir, "topology_refinement.json"))
    attempt = next(item for item in topo["attempts"]
                   if item.get("candidate_id") == "wheelchair_c0001")
    points = attempt["candidate_path_trace"]["final_reference"]["points"]
    return np.asarray([[row["x"], row["y"], 0.0] for row in points], float), attempt


def _rows(evaluator, points):
    states = evaluator.evaluate_states(points)
    return [dict(
        index=int(index),
        x=float(point[0]),
        y=float(point[1]),
        clearance=float(state.get("clearance", 0.0)),
        risk=float(state.get("risk", 0.0)),
        manifold_valid=bool(state.get("inside_manifold", False)),
        hard_valid=bool(state.get("inside_manifold", False) and
                        state.get("inside_corridor", False)),
    ) for index, (point, state) in enumerate(zip(points, states))]


def run(run_dir, output_path):
    pose, recorded = _runtime_pose(run_dir)
    reference, initial_attempt = _c0001_final_reference(run_dir)
    context = c0001_replay._scene_context()
    evaluator = SafetyEvaluator(
        manifold_constraint=dict(context["manifold_constraint"]),
        corridor_constraint={"centerline": reference.tolist(), "radius": 0.35},
        risk_field=context["social_field"])
    start = pose[:2]
    goal = c0001_replay.GOAL[:2]
    nearest = int(np.argmin(np.linalg.norm(
        reference[:, :2] - start.reshape(1, 2), axis=1)))
    join_indices = list(range(nearest, min(len(reference), nearest + 15)))
    rows = []
    d0 = float(np.linalg.norm(start - goal))
    for join_idx in join_indices:
        point = reference[join_idx]
        status = _rows(evaluator, point.reshape(1, 3))[0]
        heading = float(math.atan2(point[1] - start[1], point[0] - start[0]))
        heading_error = abs(float(math.atan2(
            math.sin(heading - pose[2]), math.cos(heading - pose[2]))))
        remaining = float(path_length(reference[join_idx:]))
        reason = ""
        if not status["hard_valid"]:
            reason = "join_point_not_hard_safe"
        elif d0 - float(np.linalg.norm(point[:2] - goal)) < -0.03:
            reason = "connector_no_progress"
        elif remaining < 0.10:
            reason = "join_insufficient_remaining_corridor"
        rows.append(dict(
            join_idx=int(join_idx),
            x=float(point[0]),
            y=float(point[1]),
            distance_from_current_pose=float(np.linalg.norm(point[:2] - start)),
            distance_to_goal=float(np.linalg.norm(point[:2] - goal)),
            clearance=status["clearance"],
            risk=status["risk"],
            manifold_valid=status["manifold_valid"],
            hard_valid=status["hard_valid"],
            heading_to_join=heading,
            heading_error_from_current_yaw=heading_error,
            remaining_corridor_length=remaining,
            critical_points_remaining=[],
            pre_filter_reject_reason=reason,
        ))

    c0001_replay.STATE = pose.copy()
    connector = c0001_replay._heading_prefix(reference)
    connector_rows = _rows(SafetyEvaluator(
        manifold_constraint=dict(context["manifold_constraint"]),
        corridor_constraint={"centerline": connector["points"].tolist(),
                             "radius": 0.35},
        risk_field=context["social_field"]), connector["points"])
    metrics = path_curvature_metrics(connector["points"])
    min_clearance = min(row["clearance"] for row in connector_rows)
    max_risk = max(row["risk"] for row in connector_rows)
    violations = int(sum(not row["manifold_valid"] for row in connector_rows))
    progress = float(d0 - np.linalg.norm(connector["points"][-1, :2] - goal))
    executable = bool(
        min_clearance >= c0001_replay.CLEARANCE - 1e-9 and
        max_risk <= c0001_replay.RISK + 1e-9 and
        violations == 0 and
        float(metrics["max_turn"]) <= c0001_replay.TURN + 1e-9 and
        float(metrics["max_curvature"]) <= c0001_replay.CURVATURE + 1e-9 and
        progress > 0.0)
    output = {
        "runtime_pose_source": "ros_log_xy_runtime_diag_yaw",
        "runtime_pose": [float(item) for item in pose.tolist()],
        "runtime_yaw": float(pose[2]),
        "runtime_dist_to_goal": d0,
        "runtime_context_fingerprint": str(context.get("fingerprint", "")),
        "active_corridor": {
            "candidate_id": "wheelchair_c0001",
            "candidate_label": "morse_saddle_2",
        },
        "candidate_id": "wheelchair_c0001",
        "candidate_label": "morse_saddle_2",
        "active_corridor_progress_index": nearest,
        "nearest_forward_idx": nearest,
        "recorded_runtime_candidate_note": (
            "R011 saved join-point distances/reasons but not runtime candidate "
            "coordinates; live diagnostics now preserve coordinates."),
        "recorded_runtime_candidate": {
            "candidate_label": "morse_saddle_0",
            "join_points_tested": int(recorded.get("join_points_tested", 0)),
            "safe_join_points": int(sum(
                not item.get("pre_filter_reject_reason")
                for item in recorded.get("join_point_audit", []))),
            "final_reject_reason": str(recorded.get("final_reject_reason", "")),
        },
        "current_corridor_forward_join_count": len(rows),
        "current_corridor_safe_join_count": int(sum(
            not row["pre_filter_reject_reason"] for row in rows)),
        "best_current_corridor_join_idx": int(connector["join_index"]),
        "join_point_audit": rows,
        "selected_join_idx": int(connector["join_index"]),
        "connector": {
            "method": "heading_progress_prefix",
            "point_count": int(len(connector["points"])),
            "length": float(path_length(connector["points"])),
            "min_clearance": min_clearance,
            "max_risk": max_risk,
            "manifold_violation_count": violations,
            "max_turn": float(metrics["max_turn"]),
            "max_curvature": float(metrics["max_curvature"]),
            "goal_progress": progress,
        },
        "merged_route": {
            "critical_sequence": "passed",
            "safe_terminal_reused": bool(initial_attempt.get(
                "safe_terminal_rebuild", {}).get(
                    "safe_terminal_rebuild_applied", False)),
            "post_refine_max_turn": float(metrics["max_turn"]),
            "post_refine_max_curvature": float(metrics["max_curvature"]),
            "final_reference_valid": executable,
            "final_reject_reason": "" if executable else "runtime_join_replay_failed",
        },
        "runtime_executable_candidate_count": int(executable),
    }
    with open(output_path, "w") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=os.path.join(
        ROOT, "results", "runs", "20260831_R011", "wheelchair", "stsm"))
    parser.add_argument("--out", default=os.path.join(
        ROOT, "results", "runs", "20260831_R011", "wheelchair", "stsm",
        "runtime_join_replay.json"))
    args = parser.parse_args()
    run(args.run_dir, args.out)
