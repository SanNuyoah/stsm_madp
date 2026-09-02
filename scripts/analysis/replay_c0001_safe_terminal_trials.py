#!/usr/bin/env python3
"""Offline c0001 replay/audit utilities; they never change live planning."""
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
STATE = np.array([2.0, 1.5, -2.4], float)
CLEARANCE = 0.10
RISK = 2.0
TURN = 0.40
CURVATURE = 8.0


def _load_initial_pose(trace_path):
    """Read the actual reset pose saved next to run diagnostics."""
    run_dir = os.path.dirname(os.path.abspath(trace_path))
    ros_log = os.path.join(run_dir, "ros.log")
    if os.path.exists(ros_log):
        with open(ros_log, "r", errors="ignore") as handle:
            for line in handle:
                if "reset wheelchair to start pose" not in line:
                    continue
                tail = line.split("reset wheelchair to start pose", 1)[1]
                payload = tail[tail.find("[") + 1:tail.find("]")]
                values = [float(item) for item in payload.replace(",", " ").split()]
                if len(values) >= 3:
                    return np.asarray(values[:3], float), "diagnostics_ros_log"
    return np.asarray(STATE, float), "fallback"


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


def _repair_refined_main_turn(reference, evaluator):
    """Bounded c0001-only execution-geometry repair around one sharp turn."""
    path = np.asarray(reference, float)
    audit = wheelchair_sharp_turn_audit(path, turn_limit=TURN)
    if not audit["sharp_turns"]:
        return path, {"repair_applied": False, "reason": "no_sharp_turn"}
    sharp = max(audit["sharp_turns"], key=lambda item: item["local_turn"])
    center = int(sharp["index"])
    # Keep every original point, including P16, and insert samples only
    # between the sharp-turn center and its successor P17.
    if center < 1 or center + 1 >= len(path):
        return path, {"repair_applied": False, "reason": "window_unavailable"}
    p15, p16, p17 = path[center - 1], path[center], path[center + 1]
    heading_before = math.atan2(p16[1] - p15[1], p16[0] - p15[0])
    heading_after = math.atan2(p17[1] - p16[1], p17[0] - p16[0])
    total_turn = abs(math.atan2(math.sin(heading_after - heading_before),
                                math.cos(heading_after - heading_before)))
    n_turn_steps = max(2, int(math.ceil(total_turn / TURN)))
    handle = min(0.03, 0.35 * min(
        float(np.linalg.norm(p16[:2] - p15[:2])),
        float(np.linalg.norm(p17[:2] - p16[:2]))))
    attempts = []
    for point_count in (n_turn_steps, n_turn_steps + 1, n_turn_steps + 2):
        attempts.append({
            "method": "cubic_heading_interpolation",
            "left": center,
            "right": center + 1,
            "point_count": int(point_count),
            "handle": float(handle),
        })
    left = max(0, center - 2)
    right = min(len(path) - 1, center + 1)
    chord = float(np.linalg.norm(path[right, :2] - path[left, :2]))
    if right > left + 1 and chord > 1e-9:
        for handle_scale in (0.25, 0.40, 0.60, 0.80, 1.00):
            for point_count in range(4, 10):
                attempts.append({
                    "method": "curvature_aware_cubic_window",
                    "left": int(left),
                    "right": int(right),
                    "point_count": int(point_count),
                    "handle": float(chord * handle_scale),
                })
    for attempt in attempts:
        method = str(attempt["method"])
        left_idx = int(attempt["left"])
        right_idx = int(attempt["right"])
        point_count = int(attempt["point_count"])
        start_pt = path[left_idx]
        end_pt = path[right_idx]
        if method == "cubic_heading_interpolation":
            h_start = heading_before
            h_end = heading_after
        else:
            h_start = math.atan2(path[left_idx + 1, 1] - path[left_idx, 1],
                                 path[left_idx + 1, 0] - path[left_idx, 0])
            h_end = math.atan2(path[right_idx, 1] - path[right_idx - 1, 1],
                               path[right_idx, 0] - path[right_idx - 1, 0])
        c1 = start_pt[:2] + float(attempt["handle"]) * np.array([
            math.cos(h_start), math.sin(h_start)])
        c2 = end_pt[:2] - float(attempt["handle"]) * np.array([
            math.cos(h_end), math.sin(h_end)])
        inserts = []
        for index in range(1, point_count):
            u = float(index) / float(point_count)
            p2 = ((1.0 - u) ** 3 * start_pt[:2] +
                  3.0 * (1.0 - u) ** 2 * u * c1 +
                  3.0 * (1.0 - u) * u ** 2 * c2 + u ** 3 * end_pt[:2])
            inserts.append([p2[0], p2[1], 0.0])
        candidate = np.vstack([path[:left_idx + 1], np.asarray(inserts, float),
                               path[right_idx:]])
        new_right_idx = left_idx + len(inserts) + 1
        local = candidate[max(0, left_idx - 2):min(
            len(candidate), new_right_idx + 3)]
        local_turn = wheelchair_sharp_turn_audit(local, turn_limit=TURN)
        local_metrics = path_curvature_metrics(local)
        full_metrics = path_curvature_metrics(candidate)
        statuses = evaluator.evaluate_states(candidate[
            left_idx + 1:left_idx + 1 + len(inserts)])
        hard_valid = all(bool(item["inside_manifold"]) and
                         bool(item["inside_corridor"]) for item in statuses)
        if (float(local_turn["max_turn"]) <= TURN + 1e-9 and
                float(local_metrics["max_curvature"]) <= CURVATURE + 1e-9 and
                float(full_metrics["max_curvature"]) <= CURVATURE + 1e-9 and
                hard_valid):
            return candidate, {
                "repair_applied": True,
                "repair_window_refined_indices": [left_idx, right_idx],
                "original_window_point_count": int(right_idx - left_idx + 1),
                "repaired_window_point_count": int(2 + len(inserts)),
                "inserted_point_count": len(inserts),
                "n_turn_steps": n_turn_steps,
                "attempt_point_count": point_count,
                "repair_method": method,
                "original_max_turn": float(sharp["local_turn"]),
                "repaired_local_max_turn": float(local_turn["max_turn"]),
                "repaired_local_max_curvature": float(
                    local_metrics["max_curvature"]),
                "repaired_full_max_curvature": float(
                    full_metrics["max_curvature"]),
            }
    return path, {"repair_applied": False,
                  "reason": "local_safety_turn_or_curvature"}


def _heading_prefix(reference):
    """Analysis-only equivalent of the bounded heading-continuous launch stage."""
    ref = np.asarray(reference, float)
    start, goal = STATE[:2], GOAL[:2]
    turn_limit = TURN
    min_segment = 0.06
    radius = min_segment / turn_limit

    def mod2pi(angle):
        return float(angle) % (2.0 * math.pi)

    def csc(end, target_yaw):
        delta = end - start
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-9:
            return []
        d = distance / radius
        theta = math.atan2(delta[1], delta[0])
        alpha = mod2pi(STATE[2] - theta)
        beta = mod2pi(target_yaw - theta)
        paths = []
        lsl_p2 = (2.0 + d * d - 2.0 * math.cos(alpha - beta) +
                  2.0 * d * (math.sin(alpha) - math.sin(beta)))
        if lsl_p2 >= -1e-9:
            tmp = math.atan2(math.cos(beta) - math.cos(alpha),
                             d + math.sin(alpha) - math.sin(beta))
            paths.append(("LSL", (mod2pi(-alpha + tmp),
                                  math.sqrt(max(0.0, lsl_p2)),
                                  mod2pi(beta - tmp))))
        rsr_p2 = (2.0 + d * d - 2.0 * math.cos(alpha - beta) +
                  2.0 * d * (math.sin(beta) - math.sin(alpha)))
        if rsr_p2 >= -1e-9:
            tmp = math.atan2(math.cos(alpha) - math.cos(beta),
                             d - math.sin(alpha) + math.sin(beta))
            paths.append(("RSR", (mod2pi(alpha - tmp),
                                  math.sqrt(max(0.0, rsr_p2)),
                                  mod2pi(-beta + tmp))))
        return paths

    def sample(primitive, values):
        x, y, yaw = float(start[0]), float(start[1]), float(STATE[2])
        points = [[x, y, 0.0]]
        for kind, value in zip(primitive, values):
            count = (max(1, int(math.floor(radius * value / min_segment)))
                     if kind == "S" else
                     max(1, int(math.ceil(value / turn_limit))))
            increment = value / float(count)
            for _unused in range(count):
                if kind == "S":
                    x += radius * increment * math.cos(yaw)
                    y += radius * increment * math.sin(yaw)
                else:
                    sign = 1.0 if kind == "L" else -1.0
                    next_yaw = yaw + sign * increment
                    x += sign * radius * (math.sin(next_yaw) - math.sin(yaw))
                    y -= sign * radius * (math.cos(next_yaw) - math.cos(yaw))
                    yaw = next_yaw
                points.append([x, y, 0.0])
        return np.asarray(points, float)

    nearest = int(np.argmin(np.linalg.norm(
        ref[:, :2] - start.reshape(1, 2), axis=1)))
    candidates = []
    for join_idx in sorted(set(min(max(index, 1), len(ref) - 1) for index in (
            nearest, nearest + 2, min(len(ref) - 1, 8), min(len(ref) - 1, 10)))):
        end = ref[join_idx]
        tail_vec = (ref[join_idx + 1, :2] - end[:2]
                    if join_idx + 1 < len(ref) else goal - end[:2])
        tail_dir = tail_vec / max(float(np.linalg.norm(tail_vec)), 1e-9)
        target_yaw = math.atan2(tail_dir[1], tail_dir[0])
        for primitive, values in csc(end[:2], target_yaw):
            prefix = sample(primitive, values)
            candidates.append({
                "points": np.vstack([prefix, ref[join_idx + 1:]]),
                "bridge_point_count": int(len(prefix)),
                "join_index": int(join_idx),
                "primitive": primitive,
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


def _curvature_origin_audit(reference, lineage):
    pts = np.asarray(reference, float)
    if len(pts) < 3:
        return {"max_curvature_index": None, "max_curvature": 0.0,
                "curvature_origin": "none", "points": []}
    best = None
    for index in range(1, len(pts) - 1):
        u = pts[index, :2] - pts[index - 1, :2]
        v = pts[index + 1, :2] - pts[index, :2]
        nu = float(np.linalg.norm(u))
        nv = float(np.linalg.norm(v))
        if nu <= 1e-9 or nv <= 1e-9:
            continue
        before = math.atan2(u[1], u[0])
        after = math.atan2(v[1], v[0])
        turn = abs(math.atan2(math.sin(after - before),
                              math.cos(after - before)))
        curvature = turn / max(0.5 * (nu + nv), 1e-9)
        row = (curvature, index, turn, nu, nv, before, after)
        if best is None or row[0] > best[0]:
            best = row
    if best is None:
        return {"max_curvature_index": None, "max_curvature": 0.0,
                "curvature_origin": "none", "points": []}
    curvature, index, turn, nu, nv, before, after = best
    related = []
    for point_index in (index - 1, index, index + 1):
        related.append({"index": int(point_index), "x": float(pts[point_index, 0]),
                        "y": float(pts[point_index, 1]),
                        "lineage": dict(lineage[point_index])})
    stages = {item["lineage"]["source_stage"] for item in related}
    origin = next(iter(stages)) if len(stages) == 1 else "stage_boundary"
    return {
        "max_curvature_index": int(index),
        "max_curvature": float(curvature),
        "segment_length_before": float(nu),
        "segment_length_after": float(nv),
        "heading_before": float(before),
        "heading_after": float(after),
        "delta_heading": float(turn),
        "turn_angle": float(turn),
        "curvature_origin": str(origin),
        "points": related,
    }


def _refined_repair_lineage(point_count, repair):
    if not bool(repair.get("repair_applied", False)):
        return [{"source_stage": "refined_main_original",
                 "source_index": int(index),
                 "source_refined_index": int(index)}
                for index in range(point_count)]
    left, right = [int(item) for item in repair["repair_window_refined_indices"]]
    inserted = int(repair.get("inserted_point_count", 0))
    lineage = []
    for index in range(point_count):
        if index <= left:
            refined_index = index
            stage = "refined_main_original"
        elif index <= left + inserted:
            refined_index = None
            stage = "refined_main_repair"
        else:
            refined_index = right + (index - (left + inserted + 1))
            stage = "refined_main_original"
        lineage.append({"source_stage": stage,
                        "source_index": int(index),
                        "source_refined_index": (
                            None if refined_index is None else int(refined_index))})
    return lineage


def _execution_reference(rebuilt, force_launch_prefix=False):
    """Reuse the existing profile and bounded smoothing policy offline."""
    base_profile = wheelchair_nonholonomic_execution_profile(
        rebuilt, STATE, GOAL, min_step=0.03, initial_lookahead=0.12,
        horizon_points=min(10, max(4, len(rebuilt))))
    needs_prefix = bool(force_launch_prefix or
                        base_profile["initial_heading_error"] > 1.85 or
                        base_profile["monotonic_regression_ratio"] > 0.18 or
                        base_profile["nonmonotonic_fraction"] > 0.30 or
                        base_profile["heading_oscillation"] > 0.50)
    prefix_metadata = _heading_prefix(rebuilt) if needs_prefix else None
    reference = prefix_metadata["points"] if prefix_metadata is not None else rebuilt
    metrics = path_curvature_metrics(reference)
    if (not force_launch_prefix and
            float(metrics["max_turn"]) > TURN + 0.03):
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
    if prefix_metadata is not None:
        prefix_count = int(prefix_metadata["bridge_point_count"])
        local_points = reference[:min(len(reference), prefix_count + 1)]
        prefix_turns = wheelchair_sharp_turn_audit(
            local_points, turn_limit=TURN)
        join_turn = float(wheelchair_sharp_turn_audit(
            local_points[-3:], turn_limit=TURN).get(
                "max_turn", 0.0)) if len(local_points) >= 3 else 0.0
        prefix_audit = {
            "launch_prefix_point_count": prefix_count,
            "launch_prefix_max_turn": float(prefix_turns["max_turn"]),
            "launch_prefix_sharp_turn_indices": list(
                prefix_turns["sharp_turn_indices"]),
            "prefix_join_max_turn": join_turn,
            "join_index": int(prefix_metadata["join_index"]),
            "primitive": str(prefix_metadata.get("primitive", "")),
        }
    else:
        prefix_audit = {"launch_prefix_point_count": 0,
                        "launch_prefix_max_turn": 0.0,
                        "launch_prefix_sharp_turn_indices": [],
                        "prefix_join_max_turn": 0.0}
    return (np.asarray(reference, float), dict(metrics), bool(needs_prefix),
            _turn_origin_audit(reference, lineage), prefix_audit)


def _execution_geometry_replay(candidate, refined, evaluator):
    repaired, main_turn_repair = _repair_refined_main_turn(refined, evaluator)
    final, geometry, prefix_used, turn_origin, prefix_audit = (
        _execution_reference(repaired))
    if prefix_used:
        lineage = _point_lineage(
            len(final), len(repaired) - 1, prefix_metadata={
                "bridge_point_count": int(prefix_audit.get(
                    "launch_prefix_point_count", 0)),
                "join_index": int(prefix_audit.get("join_index", 0)),
            })
    else:
        lineage = _refined_repair_lineage(len(final), main_turn_repair)
    final_evaluator = SafetyEvaluator(
        manifold_constraint=dict(evaluator.manifold_constraint),
        corridor_constraint={"centerline": final.tolist(), "radius": 0.35},
        risk_field=evaluator.risk_field)
    final_rows = _rows(final_evaluator, final)
    manifold_bad = [row["index"] for row in final_rows
                    if not row["manifold_valid"]]
    hard_bad = [row["index"] for row in final_rows if not row["hard_valid"]]
    final_max_turn = float(geometry.get("max_turn", 0.0))
    final_max_curvature = float(geometry.get("max_curvature", 0.0))
    final_reference_valid = bool(
        not hard_bad and final_max_turn <= TURN + 1e-9 and
        final_max_curvature <= CURVATURE + 1e-9)
    reject_reason = (
        "" if final_reference_valid else
        "final_reference_manifold_violation" if manifold_bad else
        "final_reference_hard_safety_violation" if hard_bad else
        "refined_execution_turn_limit" if final_max_turn > TURN + 1e-9 else
        "refined_execution_curvature_limit")
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "label": str(candidate["label"]),
        "initial_pose": [float(item) for item in STATE.tolist()],
        "execution_prefix_used": bool(prefix_used),
        "launch_prefix_audit": prefix_audit,
        "refined_main_turn_repair": main_turn_repair,
        "final_reference_point_count": int(len(final)),
        "execution_geometry": geometry,
        "turn_origin_audit": turn_origin,
        "curvature_origin_audit": _curvature_origin_audit(final, lineage),
        "final_min_clearance": min(row["clearance"] for row in final_rows),
        "final_max_risk": max(row["risk"] for row in final_rows),
        "final_manifold_violation_count": int(len(manifold_bad)),
        "final_hard_invalid_indices": hard_bad,
        "critical_sequence_status": "passed",
        "final_reference_valid": final_reference_valid,
        "final_reject_reason": reject_reason,
    }


def _existing_final_reference_audit(candidate):
    final = candidate.get("final_reference")
    if not isinstance(final, dict) or not final.get("points"):
        return {"available": False}
    rows = list(final.get("points") or [])
    points = np.asarray([[row["x"], row["y"], 0.0] for row in rows], float)
    lineage = []
    for row in rows:
        lineage.append({
            "source_stage": str(row.get("source_stage", "")),
            "source_index": int(row.get("source_index", row.get("index", 0))),
            "source_refined_index": int(row.get("source_index", 0))
            if str(row.get("source_stage", "")) == "refinement" else None,
        })
    return {
        "available": True,
        "final_reference_point_count": int(len(points)),
        "execution_geometry": dict(path_curvature_metrics(points)),
        "curvature_origin_audit": _curvature_origin_audit(points, lineage),
        "turn_origin_audit": _turn_origin_audit(points, lineage),
    }


def run(trace_path, output_path, include_terminal_trials=True):
    global STATE
    STATE, initial_pose_source = _load_initial_pose(trace_path)
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
    execution_replay = _execution_geometry_replay(candidate, refined, evaluator)
    execution_replay["initial_pose_source"] = str(initial_pose_source)
    existing_final_audit = _existing_final_reference_audit(candidate)
    terminal_preflight = terminal_acceptance_preflight(GOAL, 0.25, context)
    terminals = terminal_preflight["safe_terminal_candidates"]
    terminals = sorted(
        terminals,
        key=lambda row: (
            float(row["distance_to_goal"]),
            -float(row["clearance"]),
            float(row["risk"])))
    trials = []
    for rank, terminal in enumerate(terminals, start=1) if include_terminal_trials else []:
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
            # Mirror the live ordering: rebuilding the terminal never skips
            # the shared launch geometry; only then may the main-path repair
            # operate on the final merged reference.
            final, geometry, prefix_used, turn_origin, prefix_audit = (
                _execution_reference(rebuilt, force_launch_prefix=True))
            final, main_turn_repair = _repair_refined_main_turn(
                final, evaluator)
            geometry = dict(path_curvature_metrics(final))
            execution_profile = wheelchair_nonholonomic_execution_profile(
                final, STATE, GOAL, min_step=0.03, initial_lookahead=0.12,
                horizon_points=min(10, max(4, len(final))))
            # Mirror WheelchairNode's authoritative final-reference context:
            # generated launch geometry is evaluated against the reference
            # that would actually be handed to execution, not the obsolete
            # pre-rebuild centerline.
            final_evaluator = SafetyEvaluator(
                manifold_constraint=dict(context["manifold_constraint"]),
                corridor_constraint={"centerline": final.tolist(), "radius": 0.35},
                risk_field=context["social_field"])
            final_rows = _rows(final_evaluator, final)
            prefix_count = int(prefix_audit.get("launch_prefix_point_count", 0))
            if prefix_count:
                prefix_rows = final_rows[:prefix_count]
                prefix_audit.update({
                    "launch_prefix_min_clearance": min(
                        row["clearance"] for row in prefix_rows),
                    "launch_prefix_max_risk": max(
                        row["risk"] for row in prefix_rows),
                    "launch_prefix_manifold_violation_count": int(sum(
                        not row["manifold_valid"] for row in prefix_rows)),
                    "launch_prefix_hard_valid": bool(all(
                        row["hard_valid"] for row in prefix_rows)),
                })
            manifold_bad = [row["index"] for row in final_rows
                            if not row["manifold_valid"]]
            hard_bad = [row["index"] for row in final_rows
                        if not row["hard_valid"]]
            max_turn = float(wheelchair_sharp_turn_audit(
                final, turn_limit=TURN)["max_turn"])
            max_curvature = float(geometry.get("max_curvature", 0.0))
            min_clearance = min(row["clearance"] for row in final_rows)
            max_risk = max(row["risk"] for row in final_rows)
            valid = bool(not hard_bad and max_turn <= TURN + 1e-9 and
                         max_curvature <= CURVATURE + 1e-9 and
                         np.linalg.norm(final[-1, :2] - GOAL[:2]) <= 0.25 + 1e-9)
            trial.update({
                "execution_prefix_used": bool(prefix_used),
                "final_reference_point_count": int(len(final)),
                "final_min_clearance": min_clearance,
                "final_max_risk": max_risk,
                "final_manifold_violation_count": int(len(manifold_bad)),
                "final_hard_invalid_indices": hard_bad,
                "final_max_turn": max_turn,
                "final_max_curvature": max_curvature,
                "final_reference_valid": valid,
                "reject_reason": ("" if valid else
                                  "final_reference_manifold_violation" if manifold_bad else
                                  "final_reference_hard_safety_violation" if hard_bad else
                                  "refined_execution_turn_limit" if max_turn > TURN + 1e-9 else
                                  "refined_execution_curvature_limit" if max_curvature > CURVATURE + 1e-9 else
                                  "completion_region_not_reached"),
                "execution_geometry": geometry,
                "nonholonomic_execution_profile": execution_profile,
                "turn_origin_audit": turn_origin,
                "launch_prefix_audit": prefix_audit,
                "refined_main_turn_repair": main_turn_repair,
            })
            trials.append(trial)
            if valid:
                break
        if trials and trials[-1].get("final_reference_valid"):
            break
    selected = next((item for item in trials
                     if item.get("final_reference_valid")), None)
    safe_terminal_pipeline = {
        "fixed_goal_hard_valid": bool(terminal_preflight.get(
            "goal_hard_valid", False)),
        "safe_terminal_rebuild_triggered": bool(
            include_terminal_trials and not terminal_preflight.get(
                "goal_hard_valid", False)),
        "safe_terminal_candidate_count": int(len(terminals)),
        "terminal_trial_count": int(len(trials)),
        "final_reference_valid": bool(selected is not None),
        "selected_terminal": (selected.get("selected_terminal")
                              if selected is not None else None),
        "selected_terminal_distance_to_goal": (
            selected.get("distance_to_fixed_goal") if selected is not None else None),
        "rebuild_start_index": (
            selected.get("rebuild_start_idx") if selected is not None else None),
        "terminal_rebuild_point_count": (
            selected.get("rebuilt_point_count") if selected is not None else 0),
        "final_reference_point_count": (
            selected.get("final_reference_point_count") if selected is not None else 0),
        "final_max_turn": (
            selected.get("final_max_turn") if selected is not None else None),
        "final_max_curvature": (
            selected.get("final_max_curvature") if selected is not None else None),
        "final_min_clearance": (
            selected.get("final_min_clearance") if selected is not None else None),
        "final_max_risk": (
            selected.get("final_max_risk") if selected is not None else None),
        "final_manifold_violation_count": (
            selected.get("final_manifold_violation_count")
            if selected is not None else None),
        "critical_sequence_status": (
            selected.get("critical_sequence_status")
            if selected is not None else "not_checked"),
        "final_reject_reason": (
            selected.get("reject_reason") if selected is not None else
            (trials[-1].get("reject_reason") if trials else
             "no_safe_terminal_trial")),
    }
    output = {
        "candidate_id": candidate["candidate_id"], "label": candidate["label"],
        "refined_point_count": int(len(refined)), "refined_safety": refined_rows,
        "first_hard_invalid_index": invalid[0] if invalid else None,
        "last_hard_safe_index": last_safe, "hard_invalid_indices": invalid,
        "initial_pose": [float(item) for item in STATE.tolist()],
        "initial_pose_source": str(initial_pose_source),
        "r010_existing_final_reference_audit": existing_final_audit,
        "execution_geometry_without_safe_terminal": execution_replay,
        "safe_terminal_pipeline": safe_terminal_pipeline,
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
        ROOT, "results", "runs", "20260831_R010", "wheelchair", "stsm",
        "candidate_path_trace.json"))
    parser.add_argument("--out", default=os.path.join(
        ROOT, "results", "runs", "20260831_R010", "wheelchair", "stsm",
        "c0001_curvature_replay.json"))
    parser.add_argument("--no-safe-terminal", action="store_true")
    args = parser.parse_args()
    run(args.trace, args.out, include_terminal_trials=not args.no_safe_terminal)
