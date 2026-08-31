#!/usr/bin/env python3
"""Offline R009 c0001 safe-terminal replay; it never changes live planning."""
import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from stsm_madp.deform import path_curvature_metrics
from stsm_madp.mpc import (
    wheelchair_nonholonomic_execution_profile, wheelchair_sharp_turn_audit)
from stsm_madp.safety_evaluator import (
    SafetyEvaluator, build_safety_context, terminal_acceptance_preflight)
from stsm_madp.social_field import (
    HumanState, SemanticAnchor, SocialField, SocialFieldParams)
from stsm_madp.topology_refinement import smooth_wheelchair_corners


GOAL = np.array([-0.55, 0.55, 0.0], float)
# R009 records the measured planning-time yaw as ``current_yaw=0.0``.
# Do not substitute the launch-file reset yaw (-2.4) during replay.
STATE = np.array([2.0, 1.5, 0.0], float)
CLEARANCE = 0.10
RISK = 2.0
TURN = 0.40


def _scene_context():
    """Recreate the R009 authoritative Wheelchair safety world exactly."""
    field = SocialField(SocialFieldParams(
        lam_prox=1.2, lam_close=1.0, lam_dir=0.5, lam_body=0.0,
        lam_env=1.5, sigma_env=0.4, direction_model="continuous",
        task_aware_enabled=True))
    field.set_scene(
        [HumanState(pos=[-1.6, 0.2, 0.0], heading=np.pi / 2.0,
                    posture="transferring", vulnerability=1.4)],
        [SemanticAnchor("bed", [-1.6, -1.0, 0.0], [0.5, 1.0, 0.5],
                        weight=2.0, forbidden=True),
         SemanticAnchor("transfer-zone", [-0.7, -1.0, 0.0], [0.4, 1.0, 0.5],
                        weight=2.5, forbidden=True),
         SemanticAnchor("table", [0.55, 0.0, 0.0], [0.3, 0.5, 0.4],
                        weight=1.0, forbidden=True)])
    field.set_task_context({
        "task_mode": "navigation", "task_state": "moving",
        "phase": "navigation", "current_phase": "navigation",
        "progress": 0.0, "near_narrow_passage": False,
        "near_critical_point": False, "obstacle_ahead": False,
    })
    return build_safety_context(
        field, {"minimum_clearance": CLEARANCE, "risk_threshold": RISK},
        strict=True)


def _points(stage):
    return np.asarray([[row["x"], row["y"], 0.0]
                       for row in stage["points"]], float)


def _rows(evaluator, points):
    states = evaluator.evaluate_states(points)
    return [{
        "index": int(index), "x": float(point[0]), "y": float(point[1]),
        "clearance": float(status["clearance"]),
        "risk": float(status["risk"]),
        "manifold_valid": bool(status["inside_manifold"]),
        "hard_valid": bool(status["inside_manifold"] and
                           status["inside_corridor"]),
        "distance_to_fixed_goal": float(np.linalg.norm(point[:2] - GOAL[:2])),
    } for index, (point, status) in enumerate(zip(points, states))]


def _segment(start, terminal, step=0.03):
    distance = float(np.linalg.norm(terminal[:2] - start[:2]))
    count = max(1, int(math.ceil(distance / step)))
    return np.asarray([
        start + (terminal - start) * float(index) / float(count)
        for index in range(1, count + 1)], float)


def _heading_prefix(reference):
    """Analysis-only equivalent of the existing bounded launch-prefix stage."""
    ref = np.asarray(reference, float)
    start, goal = STATE[:2], GOAL[:2]
    to_goal = goal - start
    goal_dist = float(np.linalg.norm(to_goal))
    heading = np.array([math.cos(STATE[2]), math.sin(STATE[2])])
    goal_dir = to_goal / goal_dist
    blend = heading + goal_dir
    if np.linalg.norm(blend) <= 1e-6 or np.dot(heading, goal_dir) < 0.0:
        blend = heading
    launch_dir = blend / max(float(np.linalg.norm(blend)), 1e-9)
    launch_len = min(0.55, max(0.24, 0.22 * goal_dist))
    step = 0.06
    prefix_count = max(3, int(math.ceil(launch_len / step)))
    join = start + launch_dir * launch_len
    dists = np.linalg.norm(ref[:, :2] - join.reshape(1, 2), axis=1)
    nearest = int(np.argmin(dists))
    candidates = []
    for join_idx in sorted(set(min(max(index, 1), len(ref) - 1) for index in (
            nearest, nearest + 2, min(len(ref) - 1, 8), min(len(ref) - 1, 10)))):
        end = ref[join_idx]
        tail_vec = (ref[join_idx + 1, :2] - end[:2]
                    if join_idx + 1 < len(ref) else goal - end[:2])
        tail_dir = tail_vec / max(float(np.linalg.norm(tail_vec)), 1e-9)
        length = float(np.linalg.norm(end[:2] - start))
        for scale in (0.45, 0.55):
            c1 = start + heading * length * scale
            c2 = end[:2] - tail_dir * length * scale
            samples = min(14, max(8, int(math.ceil(length / 0.10))))
            bridge = []
            for index in range(samples + 1):
                u = float(index) / float(samples)
                p2 = ((1.0 - u) ** 3 * start + 3.0 * (1.0 - u) ** 2 * u * c1 +
                      3.0 * (1.0 - u) * u ** 2 * c2 + u ** 3 * end[:2])
                bridge.append([p2[0], p2[1], 0.0])
            candidates.append({
                "points": np.vstack([np.asarray(bridge, float),
                                     ref[join_idx + 1:]]),
                "bridge_point_count": int(len(bridge)),
                "join_index": int(join_idx),
            })
    def score(candidate):
        path = candidate["points"]
        profile = wheelchair_nonholonomic_execution_profile(
            path, STATE, GOAL, min_step=0.03, initial_lookahead=0.12,
            horizon_points=min(10, max(4, len(path))))
        turns = path_curvature_metrics(path)
        return (max(0.0, float(turns["max_turn"]) - TURN) * 20.0 +
                max(0.0, float(turns["max_curvature"]) - 8.0) * 4.0 +
                float(profile["execution_profile_cost"]))
    return min(candidates, key=score)


def _point_lineage(rebuilt_count, rebuild_start_idx, prefix_metadata=None,
                   smoothed=False):
    """Return analysis provenance without changing live path construction."""
    if smoothed:
        return [{"source_stage": "turn_repair_generated",
                 "source_index": int(index), "source_refined_index": None}
                for index in range(rebuilt_count)]
    if prefix_metadata is not None:
        bridge_count = int(prefix_metadata["bridge_point_count"])
        join_index = int(prefix_metadata["join_index"])
        lineage = [{"source_stage": "launch_prefix", "source_index": int(index),
                    "source_refined_index": None}
                   for index in range(bridge_count)]
        for rebuilt_index in range(join_index + 1, rebuilt_count):
            stage = ("refined_main_path" if rebuilt_index <= rebuild_start_idx
                     else "terminal_rebuild")
            lineage.append({"source_stage": stage,
                            "source_index": int(rebuilt_index),
                            "source_refined_index": (
                                int(rebuilt_index) if stage == "refined_main_path"
                                else int(rebuild_start_idx))})
        return lineage
    return [{"source_stage": ("refined_main_path"
                              if index <= rebuild_start_idx else "terminal_rebuild"),
             "source_index": int(index),
             "source_refined_index": (int(index) if index <= rebuild_start_idx
                                      else int(rebuild_start_idx))}
            for index in range(rebuilt_count)]


def _turn_origin_audit(reference, lineage):
    """Locate the exact three-point turn and classify its construction stage."""
    pts = np.asarray(reference, float)
    audit = wheelchair_sharp_turn_audit(pts, turn_limit=TURN)
    if len(pts) < 3:
        return {"max_turn_index": None, "max_turn": 0.0,
                "turn_origin": "none", "points": []}
    entries = audit["sharp_turns"]
    if entries:
        maximum = max(entries, key=lambda item: item["local_turn"])
        index = int(maximum["index"])
    else:
        # The audit also defines the index convention when every turn is legal.
        index, maximum_turn = 1, -1.0
        for candidate in range(1, len(pts) - 1):
            before = math.atan2(pts[candidate, 1] - pts[candidate - 1, 1],
                                pts[candidate, 0] - pts[candidate - 1, 0])
            after = math.atan2(pts[candidate + 1, 1] - pts[candidate, 1],
                               pts[candidate + 1, 0] - pts[candidate, 0])
            turn = abs(math.atan2(math.sin(after - before), math.cos(after - before)))
            if turn > maximum_turn:
                index, maximum_turn = candidate, turn
        maximum = {"local_turn": maximum_turn,
                   "heading_before": math.atan2(pts[index, 1] - pts[index - 1, 1],
                                                pts[index, 0] - pts[index - 1, 0]),
                   "heading_after": math.atan2(pts[index + 1, 1] - pts[index, 1],
                                               pts[index + 1, 0] - pts[index, 0])}
    related = []
    for point_index in (index - 1, index, index + 1):
        related.append({"index": int(point_index), "x": float(pts[point_index, 0]),
                        "y": float(pts[point_index, 1]),
                        "lineage": dict(lineage[point_index])})
    stages = {item["lineage"]["source_stage"] for item in related}
    origin = next(iter(stages)) if len(stages) == 1 else "stage_boundary"
    return {"max_turn_index": int(index), "max_turn": float(maximum["local_turn"]),
            "heading_before": float(maximum["heading_before"]),
            "heading_after": float(maximum["heading_after"]),
            "turn_origin": origin, "points": related}


def _execution_reference(rebuilt):
    """Reuse the existing profile and bounded smoothing policy offline."""
    base_profile = wheelchair_nonholonomic_execution_profile(
        rebuilt, STATE, GOAL, min_step=0.03, initial_lookahead=0.12,
        horizon_points=min(10, max(4, len(rebuilt))))
    needs_prefix = (base_profile["initial_heading_error"] > 1.85 or
                    base_profile["monotonic_regression_ratio"] > 0.18 or
                    base_profile["nonmonotonic_fraction"] > 0.30 or
                    base_profile["heading_oscillation"] > 0.50)
    prefix_metadata = _heading_prefix(rebuilt) if needs_prefix else None
    reference = prefix_metadata["points"] if prefix_metadata is not None else rebuilt
    metrics = path_curvature_metrics(reference)
    if float(metrics["max_turn"]) > TURN + 0.03:
        smooth = smooth_wheelchair_corners(reference, samples_per_segment=6, passes=1)
        if len(smooth) > 64:
            keep = sorted(set(int(round(value)) for value in np.linspace(
                0, len(smooth) - 1, 64)))
            smooth = np.asarray([smooth[index] for index in keep], float)
        smooth_metrics = path_curvature_metrics(smooth)
        if (float(smooth_metrics["max_turn"]) <= TURN + 0.03 and
                float(smooth_metrics["max_curvature"]) <= 8.0):
            reference, metrics = smooth, smooth_metrics
            prefix_metadata = None
            smoothed = True
        else:
            smoothed = False
    else:
        smoothed = False
    lineage = _point_lineage(len(reference), len(rebuilt) - 1,
                             prefix_metadata=prefix_metadata, smoothed=smoothed)
    return (np.asarray(reference, float), dict(metrics), bool(needs_prefix),
            _turn_origin_audit(reference, lineage))


def run(trace_path, output_path):
    with open(trace_path, "r") as handle:
        trace = json.load(handle)
    candidate = next(item for item in trace["candidates"]
                     if item["candidate_id"] == "wheelchair_c0001")
    refined = _points(candidate["refinement"])
    context = _scene_context()
    # Keep the original selected corridor as the execution tube for the tail.
    evaluator = SafetyEvaluator(
        manifold_constraint=dict(context["manifold_constraint"]),
        corridor_constraint={"centerline": refined.tolist(), "radius": 0.35},
        risk_field=context["social_field"])
    refined_rows = _rows(evaluator, refined)
    invalid = [row["index"] for row in refined_rows if not row["hard_valid"]]
    last_safe = max([row["index"] for row in refined_rows if row["hard_valid"]],
                    default=None)
    terminals = terminal_acceptance_preflight(GOAL, 0.25, context)[
        "safe_terminal_candidates"]
    terminals = sorted(terminals, key=lambda row: (row["distance_to_goal"], row["index"]))
    trials = []
    for rank, terminal in enumerate(terminals, start=1):
        terminal_point = np.array([terminal["x"], terminal["y"], 0.0])
        for start_index in (last_safe, last_safe - 1, last_safe - 2):
            suffix = _segment(refined[start_index], terminal_point)
            suffix_rows = _rows(evaluator, suffix)
            suffix_invalid = [row["index"] for row in suffix_rows
                              if not row["hard_valid"]]
            trial = {
                "terminal_rank": int(rank), "selected_terminal": terminal_point.tolist(),
                "distance_to_fixed_goal": float(terminal["distance_to_goal"]),
                "rebuild_start_idx": int(start_index),
                "rebuilt_point_count": int(len(suffix)),
                "terminal_segment_min_clearance": min(
                    [row["clearance"] for row in suffix_rows] or [0.0]),
                "terminal_segment_max_risk": max(
                    [row["risk"] for row in suffix_rows] or [0.0]),
                "terminal_segment_manifold_valid": not bool(suffix_invalid),
                "first_invalid_segment_point": (suffix_invalid[0]
                                                if suffix_invalid else None),
                "critical_sequence_status": "passed",
                "critical_sequence_source": "R009 preserved topology prefix",
            }
            if suffix_invalid:
                trial.update({"final_reference_point_count": 0,
                              "final_reference_valid": False,
                              "reject_reason": "terminal_segment_safety"})
                trials.append(trial)
                continue
            rebuilt = np.vstack([refined[:start_index + 1], suffix])
            final, geometry, prefix_used, turn_origin = _execution_reference(rebuilt)
            final_rows = _rows(evaluator, final)
            manifold_bad = [row["index"] for row in final_rows
                            if not row["manifold_valid"]]
            max_turn = float(wheelchair_sharp_turn_audit(
                final, turn_limit=TURN)["max_turn"])
            valid = bool(not manifold_bad and max_turn <= TURN + 1e-9 and
                         np.linalg.norm(final[-1, :2] - GOAL[:2]) <= 0.25 + 1e-9)
            trial.update({
                "execution_prefix_used": bool(prefix_used),
                "final_reference_point_count": int(len(final)),
                "final_min_clearance": min(row["clearance"] for row in final_rows),
                "final_max_risk": max(row["risk"] for row in final_rows),
                "final_manifold_violation_count": int(len(manifold_bad)),
                "final_max_turn": max_turn,
                "final_reference_valid": valid,
                "reject_reason": ("" if valid else
                                  "final_reference_manifold_violation" if manifold_bad else
                                  "refined_execution_turn_limit" if max_turn > TURN + 1e-9 else
                                  "completion_region_not_reached"),
                "execution_geometry": geometry,
                "turn_origin_audit": turn_origin,
            })
            trials.append(trial)
            if valid:
                break
        if trials and trials[-1].get("final_reference_valid"):
            break
    output = {
        "candidate_id": candidate["candidate_id"], "label": candidate["label"],
        "refined_point_count": int(len(refined)), "refined_safety": refined_rows,
        "first_hard_invalid_index": invalid[0] if invalid else None,
        "last_hard_safe_index": last_safe, "hard_invalid_indices": invalid,
        "safe_terminal_count": int(len(terminals)), "trials": trials,
        "executable_candidate_count": int(any(
            item.get("final_reference_valid") for item in trials)),
    }
    with open(output_path, "w") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default=os.path.join(
        ROOT, "results", "runs", "20260831_R009", "wheelchair", "stsm",
        "candidate_path_trace.json"))
    parser.add_argument("--out", default=os.path.join(
        ROOT, "results", "runs", "20260831_R009", "wheelchair", "stsm",
        "c0001_safe_terminal_trials.json"))
    args = parser.parse_args()
    run(args.trace, args.out)
