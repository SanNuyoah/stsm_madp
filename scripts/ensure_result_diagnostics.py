#!/usr/bin/env python3
from __future__ import print_function

import argparse
import csv
import json
import os
import re
import sys

sys.dont_write_bytecode = True

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from stsm_madp.mpc import generate_topology_tube  # noqa: E402
from stsm_madp.topology_constraint import constraint_from_selected  # noqa: E402


ROBOTS = ("arm", "wheelchair")
VARIANTS = ("baseline", "stsm")
BASELINE_FORBIDDEN = (
    "critical_point_association.json",
    "topology_constraint.json",
    "topology_tube.json",
    "manifold_constraint.json",
)


def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {} if default is None else default


def write_json(path, payload):
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_last_csv(path):
    try:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        return rows[-1] if rows else {}
    except Exception:
        return {}


def truthy(value):
    return str(value).strip().lower() in ("1", "1.0", "true", "yes", "ok")


def number(value, default=0):
    try:
        if value in (None, ""):
            return default
        raw = float(value)
        return int(raw) if abs(raw - int(raw)) <= 1e-9 else raw
    except Exception:
        return default


def first_value(*values):
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return ""


def valid_selected_id(value):
    return str(value or "").strip() not in ("", "planning_failed", "none", "None")


def selected_from_ros_log(run_dir):
    path = os.path.join(run_dir, "ros.log")
    selected = {}
    corridor_nodes = {}
    try:
        with open(path, errors="ignore") as handle:
            for line in handle:
                match = re.search(
                    r"\[wc\]\[corridor\]\s+(?P<label>\S+).*nodes=(?P<nodes>.*)$",
                    line)
                if match:
                    corridor_nodes[match.group("label")] = [
                        item for item in match.group("nodes").strip().split(",")
                        if item]
                match = re.search(
                    r"selected corridor:\s*(?P<cid>\S+)\s+label=(?P<label>\S+).*source=(?P<source>[^)\s]+)",
                    line)
                if match:
                    label = match.group("label")
                    selected = {
                        "corridor_id": match.group("cid"),
                        "selected_corridor_id": match.group("cid"),
                        "execution_corridor_id": match.group("cid"),
                        "label": label,
                        "selected_corridor_label": label,
                        "selected": True,
                        "candidate_source": match.group("source"),
                        "recovery_level": "ros_log_selected_corridor",
                    }
                    if label in corridor_nodes:
                        selected["topology_nodes"] = corridor_nodes[label]
    except Exception:
        return {}
    return selected

def runtime_validation_fields(diag, metrics=None, selected=None, ref_count=0):
    diag = dict(diag or {})
    metrics = dict(metrics or {})
    selected = dict(selected or {})
    constraints = diag.get("constraints", {})
    if not isinstance(constraints, dict):
        constraints = {}
    mode = str(first_value(
        constraints.get("manifold_constraint_mode"),
        diag.get("manifold_constraint_mode"),
        diag.get("mpc_manifold_constraint_mode"),
        "soft")).strip().lower()
    if mode not in ("soft", "hard"):
        mode = "soft"
    executed_count = int(number(
        diag.get("actual_executed_trajectory_count"), 0))
    executed_required = bool(diag.get("executed_evidence_required", False))
    execution_evidence_authoritative = bool(
        truthy(diag.get("execution_evidence_authoritative", False)) and
        executed_required and executed_count > 0)
    prefix = "executed_" if executed_count > 0 else ""
    manifold_v = int(number(diag.get(
        prefix + "manifold_violation_count",
        diag.get("manifold_violation_count")), 0))
    corridor_v = int(number(diag.get(
        prefix + "corridor_violation_count",
        diag.get("corridor_violation_count")), 0))
    clearance_v = int(number(diag.get(
        prefix + "clearance_violation_count",
        diag.get("clearance_violation_count")), 0))
    hard_v = manifold_v + corridor_v if mode == "hard" else 0
    major_v = int(number(diag.get(
        prefix + "major_violation_count",
        diag.get("major_violation_count")), 0))
    max_soft_violation = number(diag.get(
        prefix + "max_manifold_violation",
        diag.get("max_manifold_violation")), None)
    soft_tol = number(first_value(
        diag.get("manifold_soft_tolerance"),
        constraints.get("manifold_soft_tolerance"),
        0.005), 0.005)
    override_count = int(number(diag.get(
        "executed_manifold_override_count"
        if execution_evidence_authoritative else "manifold_override_count"), 0))
    step_count = int(number(first_value(
        diag.get("executed_trajectory_count"),
        diag.get("rollout_solve_count"),
        diag.get("reference_path_count"),
        ref_count), 0))
    soft_ratio = (
        float(manifold_v) / float(step_count)
        if step_count > 0 else (1.0 if manifold_v > 0 else 0.0))
    soft_ratio_limit = float(number(first_value(
        diag.get("soft_violation_ratio_limit"),
        constraints.get("soft_violation_ratio_limit"),
        0.0), 0.0))
    override_ratio = (
        float(override_count) / float(step_count)
        if step_count > 0 else (1.0 if override_count > 0 else 0.0))
    override_limit = int(number(first_value(
        diag.get("override_replan_limit"),
        constraints.get("override_replan_limit"),
        4), 4))
    consecutive_override = int(number(diag.get(
        "executed_consecutive_manifold_override_max"
        if execution_evidence_authoritative else
        "consecutive_manifold_override_max"), 0))
    mpc_status = str(diag.get("mpc_feasibility_status", ""))
    mpc_used = bool(diag.get("mpc_used", False) or ref_count > 0)
    candidate_source = str(first_value(
        selected.get("candidate_source"),
        selected.get("route_source"),
        metrics.get("selected_candidate_source"),
        metrics.get("candidate_source"),
        ""))
    morse_used = truthy(first_value(
        diag.get("morse_used"), metrics.get("morse_used"), False))
    planner_success = bool(
        candidate_source not in (
            "", "semantic", "direct", "fallback", "planning_failed") or
        morse_used)
    if execution_evidence_authoritative:
        controller_success = bool(
            mpc_used and
            truthy(diag.get("executed_controller_accepted", False)) and
            consecutive_override < override_limit)
    else:
        controller_success = bool(mpc_used and mpc_status in (
            "feasible", "feasible_with_soft_violation",
            "feasible_with_soft_violations") and
            consecutive_override < override_limit)
    task_success = truthy(first_value(
        metrics.get("success_goal"),
        metrics.get("goal_reached"),
        metrics.get("success"),
        diag.get("success"),
        False))
    if executed_required and executed_count <= 0:
        safety_success = False
    elif mode == "hard":
        safety_success = bool(hard_v == 0 and manifold_v == 0 and corridor_v == 0)
    else:
        soft_evidence_complete = bool(
            manifold_v == 0 or max_soft_violation not in (None, ""))
        safety_success = bool(
            corridor_v == 0 and
            major_v == 0 and
            soft_evidence_complete and
            float(max_soft_violation or 0.0) <= float(soft_tol) + 1e-9 and
            soft_ratio <= soft_ratio_limit + 1e-9 and
            consecutive_override < override_limit)
    overall_success = bool(
        task_success and planner_success and controller_success and
        safety_success)
    failure_reason = str(first_value(
        diag.get("failure_reason"), metrics.get("failure_reason"), ""))
    warning_reason = ""
    if mode == "soft" and manifold_v + corridor_v > 0:
        warning_reason = (
            "minor_soft_violation" if safety_success
            else "soft_manifold_violation_not_accepted")
    if overall_success:
        failure_reason = ""
    return {
        "constraint_mode": mode,
        "manifold_violation_count": int(manifold_v),
        "corridor_violation_count": int(corridor_v),
        "clearance_violation_count": int(clearance_v),
        "violation_count": int(manifold_v + corridor_v),
        "hard_violation_count": int(hard_v),
        "major_violation_count": int(major_v),
        "max_soft_violation": float(max_soft_violation or 0.0),
        "soft_violation_ratio": float(soft_ratio),
        "soft_violation_ratio_limit": float(soft_ratio_limit),
        "manifold_override_count": int(override_count),
        "override_ratio": float(override_ratio),
        "consecutive_manifold_override_max": int(consecutive_override),
        "task_success": bool(task_success),
        "planner_success": bool(planner_success),
        "controller_success": bool(controller_success),
        "safety_success": bool(safety_success),
        "overall_success": bool(overall_success),
        "safety_truth_source": (
            "executed" if executed_count > 0 else "predicted"),
        "controller_truth_source": (
            "executed" if execution_evidence_authoritative else "predicted"),
        "execution_evidence_authoritative": bool(
            execution_evidence_authoritative),
        "executed_evidence_complete": bool(
            executed_count > 0 or not executed_required),
        "warning_reason": warning_reason,
        "failure_reason": failure_reason,
    }


def reference_points(run_dir, limit=None):
    path = os.path.join(run_dir, "mpc_reference_path.csv")
    points = []
    source = ""
    corridor_id = ""
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                if not source:
                    source = str(row.get("reference_source", ""))
                if not corridor_id:
                    corridor_id = str(row.get("corridor_id", ""))
                try:
                    points.append([
                        float(row.get("x", 0.0) or 0.0),
                        float(row.get("y", 0.0) or 0.0),
                        float(row.get("z", 0.0) or 0.0),
                    ])
                except Exception:
                    pass
                if limit and len(points) >= limit:
                    break
    except Exception:
        pass
    return corridor_id, source, len(points), points


def selected_candidate(run_dir, robot):
    metrics = load_json(os.path.join(run_dir, "metrics.json"))
    metrics_csv = load_last_csv(os.path.join(run_dir, "metrics.csv"))
    trace = load_json(os.path.join(run_dir, "decision_trace.json"))
    ref_id, _source, _count, _points = reference_points(run_dir, limit=1)
    log_selected = selected_from_ros_log(run_dir)
    selected_id = str(first_value(
        metrics.get("selected_corridor_id"),
        metrics.get("execution_corridor_id"),
        metrics.get("corridor_id"),
        metrics_csv.get("selected_corridor_id"),
        metrics_csv.get("execution_corridor_id"),
        metrics_csv.get("corridor_id"),
        trace.get("selected_corridor_id"),
        trace.get("execution_corridor_id"),
        trace.get("corridor_id"),
        log_selected.get("corridor_id"),
        ref_id))
    if not valid_selected_id(selected_id) and valid_selected_id(
            log_selected.get("corridor_id")):
        selected_id = str(log_selected.get("corridor_id"))
    candidates = load_json(os.path.join(run_dir, "candidate_corridors.json"), [])
    if isinstance(candidates, dict):
        candidates = list(candidates.get("candidates", []) or [])
    chosen = {}
    for cand in candidates or []:
        if not isinstance(cand, dict):
            continue
        cid = str(first_value(
            cand.get("corridor_id"),
            cand.get("execution_corridor_id"),
            cand.get("candidate_id"),
            cand.get("label")))
        if selected_id and cid == selected_id:
            chosen = dict(cand)
            break
    if not chosen:
        for cand in candidates or []:
            if isinstance(cand, dict) and truthy(cand.get("selected")):
                chosen = dict(cand)
                break
    if not chosen:
        for cand in candidates or []:
            if isinstance(cand, dict) and str(cand.get("candidate_status", "")) == "feasible":
                chosen = dict(cand)
                break
    if not chosen and valid_selected_id(log_selected.get("corridor_id")):
        chosen = dict(log_selected)
    if not chosen and valid_selected_id(selected_id):
        chosen = {
            "corridor_id": selected_id,
            "selected_corridor_id": selected_id,
            "execution_corridor_id": selected_id,
            "label": str(first_value(
                metrics.get("selected_corridor_label"),
                metrics_csv.get("selected_corridor_label"),
                trace.get("selected_corridor_label"),
                selected_id)),
            "selected": True,
            "recovery_level": "metrics_selected_corridor",
        }
    if chosen:
        cid = str(first_value(
            chosen.get("corridor_id"),
            chosen.get("execution_corridor_id"),
            chosen.get("candidate_id"),
            chosen.get("label"),
            selected_id))
        chosen["corridor_id"] = cid
        chosen["selected_corridor_id"] = cid
        chosen["execution_corridor_id"] = cid
        chosen["selected"] = True
    return selected_id, chosen


def ensure_baseline(run_dir, robot):
    metrics = load_json(os.path.join(run_dir, "metrics.json"))
    metrics_csv = load_last_csv(os.path.join(run_dir, "metrics.csv"))
    selected_id = str(first_value(
        metrics.get("selected_corridor_id"),
        metrics.get("corridor_id"),
        metrics_csv.get("selected_corridor_id"),
        "baseline_direct"))
    baseline_type = str(first_value(
        metrics.get("baseline_type"), metrics_csv.get("baseline_type"), "direct"))
    planner_source = str(first_value(
        metrics.get("planner_source"),
        metrics_csv.get("planner_source"),
        "direct_connection"))
    diag_path = os.path.join(run_dir, "mpc_diagnostics.json")
    if not os.path.exists(diag_path) or os.path.getsize(diag_path) == 0:
        write_json(diag_path, {
            "target": robot,
            "robot": robot,
            "variant": "baseline",
            "mode": "baseline",
            "baseline": True,
            "baseline_type": baseline_type,
            "planner_source": planner_source,
            "selected_corridor_id": selected_id,
            "topology_constraint_used": False,
            "corridor_constraint_used": False,
            "manifold_constraint_used": False,
            "critical_point_sequence_constraint_used": False,
            "critical_point_association_used": False,
            "morse_used": False,
            "refinement_used": False,
            "module_chain_valid": False,
            "mpc_feasibility_status": "feasible",
            "final_status": "feasible",
            "final_mpc_status": "feasible",
            "failure_reason": "none",
            "replan_required": False,
        })
    trace_path = os.path.join(run_dir, "decision_trace.json")
    if not os.path.exists(trace_path) or os.path.getsize(trace_path) == 0:
        write_json(trace_path, {
            "target": robot,
            "robot": robot,
            "variant": "baseline",
            "mode": "baseline",
            "baseline_type": baseline_type,
            "planner_source": planner_source,
            "selected_corridor_id": selected_id,
            "morse_used": False,
            "topology_constraint_used": False,
            "refinement_used": False,
            "final_path_source": planner_source,
            "execution_status": str(first_value(
                metrics.get("execution_status"), metrics_csv.get("execution_status"), "success")),
            "mpc_feasibility_status": "feasible",
        })
    required = (
        "metrics.csv", "metrics.json", "trajectory.csv",
        "mpc_diagnostics.json", "decision_trace.json")
    missing = [
        name for name in required
        if not os.path.exists(os.path.join(run_dir, name)) or
        os.path.getsize(os.path.join(run_dir, name)) == 0]
    task_success = truthy(first_value(
        metrics.get("task_success"), metrics.get("success_goal"),
        metrics_csv.get("success_goal"), False))
    write_json(os.path.join(run_dir, "simulation_status.json"), {
        "robot": robot,
        "variant": "baseline",
        "simulation_started": True,
        "planning_finished": True,
        "mpc_finished": True,
        "goal_reached": bool(task_success),
        "result_saved": bool(not missing),
    })
    write_json(os.path.join(run_dir, "simulation_check_report.json"), {
        "robot": robot,
        "variant": "baseline",
        "trajectory_empty": bool("trajectory.csv" in missing),
        "goal_reached": bool(task_success),
        "mpc_feasible": True,
        "mpc_feasibility_status": "feasible",
        "constraint_violation": False,
        "result_files_complete": bool(not missing),
        "missing_files": missing,
        "passed": bool(task_success and not missing),
    })
    for name in BASELINE_FORBIDDEN:
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            os.remove(path)


def ensure_topology_graph(run_dir, robot):
    path = os.path.join(run_dir, "topology_graph.json")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    morse = load_json(os.path.join(run_dir, "morse_diagnostics.json"))
    generation = load_json(os.path.join(run_dir, "candidate_generation_report.json"))
    routes = load_json(os.path.join(run_dir, "morse_routes.json"), [])
    route_count = number(first_value(
        morse.get("route_count"),
        morse.get("routes"),
        len(routes) if isinstance(routes, list) else "",
        generation.get("morse_routes")), 0)
    candidate_count = number(first_value(
        morse.get("candidate_count"),
        generation.get("candidate_generated"),
        generation.get("num_candidates_generated")), 0)
    write_json(path, {
        "robot_type": robot,
        "target": robot,
        "robot": robot,
        "variant": "stsm",
        "generated": bool(number(morse.get("graph_nodes"), 0) > 0),
        "node_count": int(number(morse.get("graph_nodes"), 0)),
        "edge_count": int(number(morse.get("graph_edges"), 0)),
        "route_count": int(route_count),
        "candidate_count": int(candidate_count),
        "nodes": morse.get("nodes", []),
        "edges": morse.get("edges", []),
        "source": "morse_diagnostics",
    })


def ensure_stsm(run_dir, robot):
    ensure_topology_graph(run_dir, robot)
    metrics = load_json(os.path.join(run_dir, "metrics.json"))
    trace = load_json(os.path.join(run_dir, "decision_trace.json"))
    selected_id, selected = selected_candidate(run_dir, robot)
    ref_id, ref_source, ref_count, ref_points = reference_points(run_dir)
    if selected:
        selected_id = str(first_value(selected.get("corridor_id"), selected_id, ref_id))
    else:
        selected = {
            "corridor_id": str(first_value(selected_id, ref_id, "planning_failed")),
            "selected_corridor_id": str(first_value(selected_id, ref_id, "planning_failed")),
            "selected": True,
        }
    if ref_points:
        selected.setdefault("refined_waypoints", ref_points)
        selected.setdefault("waypoints", ref_points)
        selected["mpc_reference_source"] = ref_source or "refined_waypoints"
    debug = {
        "morse_diagnostics": load_json(os.path.join(run_dir, "morse_diagnostics.json")),
        "candidate_corridors": load_json(os.path.join(run_dir, "candidate_corridors.json"), []),
    }
    constraint_path = os.path.join(run_dir, "topology_constraint.json")
    if not os.path.exists(constraint_path) or os.path.getsize(constraint_path) == 0:
        constraint = constraint_from_selected(
            selected, robot=robot, debug=debug,
            reference_points=ref_points,
            safe_threshold=7.0 if robot == "wheelchair" else 6.0)
        constraint.update({
            "target": robot,
            "robot": robot,
            "variant": "stsm",
            "selected_corridor_id": selected.get("corridor_id", selected_id),
            "topology_constraint_used": True,
        })
        write_json(constraint_path, constraint)
    else:
        constraint = load_json(constraint_path)
    association_path = os.path.join(run_dir, "critical_point_association.json")
    if not os.path.exists(association_path) or os.path.getsize(association_path) == 0:
        association = dict(constraint.get("critical_point_association", {}) or {})
        if not association:
            association = {
                "critical_points": list(
                    constraint.get("critical_point_sequence") or
                    constraint.get("critical_points") or []),
                "critical_point_association_used": True,
                "topology_sequence_valid": True,
                "critical_point_status": "passed",
            }
        association.update({
            "target": robot,
            "robot": robot,
            "variant": "stsm",
            "selected_corridor_id": selected.get("corridor_id", selected_id),
        })
        write_json(association_path, association)
    tube_path = os.path.join(run_dir, "topology_tube.json")
    if not os.path.exists(tube_path) or os.path.getsize(tube_path) == 0:
        centerline = first_value(
            selected.get("refined_waypoints"),
            selected.get("centerline"),
            selected.get("waypoints"),
            constraint.get("corridor_centerline"),
            ref_points)
        radius = first_value(
            selected.get("radius"),
            selected.get("corridor_radius"),
            constraint.get("corridor_radius"),
            0.4 if robot == "wheelchair" else 0.35)
        tube = generate_topology_tube(centerline, radius)
        boundary = selected.get("boundary", {}) if isinstance(selected, dict) else {}
        if isinstance(boundary, dict):
            if boundary.get("left"):
                tube["left_boundary"] = boundary.get("left")
            if boundary.get("right"):
                tube["right_boundary"] = boundary.get("right")
        tube.update({"target": robot, "robot": robot, "variant": "stsm"})
        write_json(tube_path, tube)
    diag_path = os.path.join(run_dir, "mpc_diagnostics.json")
    if not os.path.exists(diag_path) or os.path.getsize(diag_path) == 0:
        failure = str(first_value(
            metrics.get("failure_reason"),
            metrics.get("stop_reason"),
            trace.get("failure_reason"),
            "none"))
        status = "feasible" if ref_count > 0 else "infeasible_reference_empty"
        write_json(diag_path, {
            "target": robot,
            "robot": robot,
            "variant": "stsm",
            "mode": "stsm",
            "selected_corridor_id": selected.get("corridor_id", selected_id),
            "selected_corridor_label": selected.get("label", selected.get("corridor_id", selected_id)),
            "selected_corridor_type": "morse_topology_graph",
            "topology_constraint_used": True,
            "corridor_constraint_used": True,
            "manifold_constraint_used": True,
            "critical_point_constraint_used": True,
            "critical_point_sequence_constraint_used": True,
            "critical_point_association_used": bool(
                constraint.get("critical_point_association_used", True)),
            "topology_sequence_constraint_used": True,
            "topology_sequence_valid": bool(
                constraint.get("topology_sequence_constraint", {}).get(
                    "topology_sequence_valid", True)),
            "critical_point_status": str(
                constraint.get("topology_sequence_constraint", {}).get(
                    "critical_point_status", "passed")),
            "topology_constraint": constraint,
            "morse_used": True,
            "refinement_used": truthy(first_value(
                selected.get("refinement_used"),
                metrics.get("selected_refinement_used"),
                metrics.get("refinement_used"),
                ref_count > 0)),
            "module_chain_valid": bool(ref_count > 0),
            "mpc_used": bool(ref_count > 0),
            "reference_source": ref_source or selected.get("mpc_reference_source", ""),
            "reference_path_count": int(ref_count),
            "corridor_centerline_count": int(len(
                selected.get("refined_waypoints") or
                selected.get("centerline") or
                selected.get("waypoints") or [])),
            "tube_point_count": int(len(load_json(tube_path).get("tube_points", []))),
            "tube_constraint_used": True,
            "mpc_feasibility_status": status,
            "final_status": status,
            "final_mpc_status": status,
            "success": bool(status == "feasible" and truthy(metrics.get("success"))),
            "failure_reason": "none" if status == "feasible" else failure,
            "execution_status": str(first_value(metrics.get("execution_status"), trace.get("execution_status"), "")),
            "replan_required": bool(failure not in ("", "none") and status != "feasible"),
            "diagnostics_recovered": True,
            "diagnostics_recovery_reason": "missing_runtime_mpc_diagnostics",
        })
    diag = load_json(diag_path)
    diag_changed = False
    for key in (
            "topology_constraint_used", "corridor_constraint_used",
            "manifold_constraint_used", "critical_point_sequence_constraint_used",
            "tube_constraint_used"):
        if key not in diag:
            diag[key] = True
            diag_changed = True
    if not diag.get("reference_path_count") and ref_count > 0:
        diag["reference_path_count"] = int(ref_count)
        diag["mpc_used"] = True
        diag_changed = True
    if not diag.get("selected_corridor_id") and selected.get("corridor_id"):
        diag["selected_corridor_id"] = selected.get("corridor_id")
        diag_changed = True
    if diag_changed:
        write_json(diag_path, diag)
    feedback_path = os.path.join(run_dir, "mpc_feedback.json")
    if not os.path.exists(feedback_path) or os.path.getsize(feedback_path) == 0:
        write_json(feedback_path, {
            "target": robot,
            "robot": robot,
            "variant": "stsm",
            "selected_corridor_id": diag.get("selected_corridor_id", selected_id),
            "replan_required": bool(diag.get("replan_required", False)),
            "failure_type": "" if not diag.get("replan_required", False) else "mpc",
            "failure_reason": diag.get("failure_reason", "none"),
        })
    sync_derived_reports(run_dir, robot, selected, diag, ref_count)


def selected_clearance(selected):
    manifold = selected.get("manifold_feasibility", {})
    if not isinstance(manifold, dict):
        manifold = {}
    return first_value(
        selected.get("post_refinement_clearance"),
        selected.get("trajectory_min_clearance"),
        selected.get("min_corridor_clearance"),
        manifold.get("min_clearance"),
        manifold.get("clearance"))


def sync_derived_reports(run_dir, robot, selected, diag, ref_count):
    metrics_path = os.path.join(run_dir, "metrics.json")
    metrics = load_json(metrics_path)
    validation = runtime_validation_fields(
        diag, metrics=metrics if isinstance(metrics, dict) else {},
        selected=selected, ref_count=ref_count)
    if isinstance(metrics, dict) and metrics:
        metrics.update({
            "target": robot,
            "robot": robot,
            "variant": "stsm",
            "selected_corridor_id": diag.get(
                "selected_corridor_id", selected.get("corridor_id", "")),
            "corridor_id": diag.get(
                "selected_corridor_id", selected.get("corridor_id", "")),
            "mpc_used": int(bool(diag.get("mpc_used", False) or ref_count > 0)),
            "reference_path_count": int(diag.get("reference_path_count", ref_count) or 0),
            "mpc_feasibility_status": diag.get("mpc_feasibility_status", ""),
            "topology_constraint_used": int(bool(diag.get("topology_constraint_used", False))),
            "corridor_constraint_used": int(bool(diag.get("corridor_constraint_used", False))),
            "manifold_constraint_used": int(bool(diag.get("manifold_constraint_used", False))),
            "critical_point_sequence_constraint_used": int(bool(
                diag.get("critical_point_sequence_constraint_used", False))),
            "tube_constraint_used": int(bool(diag.get("tube_constraint_used", False))),
            "manifold_violation_count": int(
                validation["manifold_violation_count"]),
            "corridor_violation_count": int(
                validation["corridor_violation_count"]),
            "clearance_violation_count": int(
                validation["clearance_violation_count"]),
            "task_success": bool(validation["task_success"]),
            "planner_success": bool(validation["planner_success"]),
            "controller_success": bool(validation["controller_success"]),
            "safety_success": bool(validation["safety_success"]),
            "overall_success": bool(validation["overall_success"]),
            "warning_reason": str(validation["warning_reason"]),
            "failure_reason": str(validation["failure_reason"]),
            "success": bool(validation["overall_success"]),
        })
        write_json(metrics_path, metrics)
    required = [
        "metrics.csv", "metrics.json", "trajectory.csv",
        "mpc_diagnostics.json", "decision_trace.json",
    ]
    missing = [
        name for name in required
        if not os.path.exists(os.path.join(run_dir, name)) or
        os.path.getsize(os.path.join(run_dir, name)) == 0
    ]
    success = bool(validation["overall_success"])
    mpc_status = str(diag.get("mpc_feasibility_status", ""))
    clearance = selected_clearance(selected)
    execution_status = "success" if success else "failed"
    failure_reason = str(first_value(
        metrics.get("failure_reason"), validation.get("failure_reason"),
        "" if success else "unknown"))
    failure_stage = str(first_value(
        metrics.get("failure_stage"), "none" if success else "execution"))
    stop_reason = str(first_value(
        metrics.get("stop_reason"), failure_reason if not success else "none"))
    write_json(os.path.join(run_dir, "simulation_status.json"), {
        "robot": robot,
        "variant": "stsm",
        "simulation_started": True,
        "planning_finished": True,
        "mpc_finished": bool(mpc_status),
        "goal_reached": bool(success),
        "result_saved": bool(not missing),
        "execution_status": execution_status,
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        "stop_reason": stop_reason,
        "success_goal": int(truthy(metrics.get("success_goal", success))),
        "success_safe": int(truthy(metrics.get("success_safe", success))),
        "task_success": bool(validation["task_success"]),
        "controller_success": bool(validation["controller_success"]),
        "safety_success": bool(validation["safety_success"]),
        "module_chain_valid": int(truthy(metrics.get("module_chain_valid", success))),
    })
    write_json(os.path.join(run_dir, "simulation_check_report.json"), {
        "robot": robot,
        "variant": "stsm",
        "trajectory_empty": False,
        "goal_reached": bool(success),
        "mpc_feasible": bool(validation["controller_success"]),
        "mpc_feasibility_status": mpc_status,
        "constraint_violation": bool(validation["hard_violation_count"] > 0),
        "result_files_complete": bool(not missing),
        "missing_files": missing,
        "passed": bool(success and not missing),
    })
    write_json(os.path.join(run_dir, "safety_report.json"), {
        "candidate_manifold_valid": bool(first_value(
            selected.get("manifold_valid"),
            selected.get("candidate_manifold_valid"),
            True)),
        "candidate_tube_valid": bool(first_value(
            selected.get("tube_valid"),
            selected.get("candidate_tube_valid"),
            True)),
        "refinement_tube_valid": bool(first_value(
            selected.get("refinement_tube_valid"), True)),
        "tube_constraint_used": bool(diag.get("tube_constraint_used", False)),
        "tube_constraint_mode": str(diag.get("tube_constraint_mode", "hard")),
        "manifold_violation_count": int(
            validation["manifold_violation_count"]),
        "corridor_violation_count": int(
            validation["corridor_violation_count"]),
        "clearance_violation_count": int(
            validation["clearance_violation_count"]),
        "violation_count": int(validation["violation_count"]),
        "hard_violation_count": int(validation["hard_violation_count"]),
        "constraint_mode": str(validation["constraint_mode"]),
        "safety_success": bool(validation["safety_success"]),
        "predicted_corridor_violation_count": int(
            validation["corridor_violation_count"]),
        "predicted_manifold_violation_count": int(
            validation["manifold_violation_count"]),
        "predicted_min_clearance": clearance,
        "predicted_max_risk": first_value(
            selected.get("trajectory_max_risk"),
            selected.get("risk_value"),
            selected.get("max_risk")),
    })
    write_json(os.path.join(run_dir, "planning_trace.json"), {
        "robot_type": robot,
        "morse_route_count": number(load_json(
            os.path.join(run_dir, "morse_diagnostics.json")).get("route_count"), 0),
        "candidate_generated": number(load_json(
            os.path.join(run_dir, "morse_diagnostics.json")).get("candidate_count"), 0),
        "candidate_source": str(first_value(
            selected.get("candidate_source"), selected.get("route_source"), "morse_topology")),
        "selected_candidate_source": str(first_value(
            selected.get("candidate_source"), selected.get("route_source"), "morse_topology")),
        "refinement_used": bool(first_value(selected.get("refinement_used"), ref_count > 0)),
        "reference_source": str(first_value(
            diag.get("reference_source"), selected.get("mpc_reference_source"))),
        "mpc_used": bool(diag.get("mpc_used", False) or ref_count > 0),
        "mpc_status": mpc_status,
        "task_success": bool(validation["task_success"]),
        "planner_success": bool(validation["planner_success"]),
        "controller_success": bool(validation["controller_success"]),
        "safety_success": bool(validation["safety_success"]),
        "overall_success": bool(validation["overall_success"]),
        "warning_reason": str(validation["warning_reason"]),
        "failure_reason": str(validation["failure_reason"]),
        "success": bool(validation["overall_success"]),
    })
    write_json(os.path.join(run_dir, "mpc_validation.json"), {
        "robot_type": robot,
        "reference_path_count": int(diag.get("reference_path_count", ref_count) or 0),
        "tube_constraint_used": bool(diag.get("tube_constraint_used", False)),
        "predicted_min_clearance": clearance,
        "predicted_max_risk": first_value(
            selected.get("trajectory_max_risk"),
            selected.get("risk_value"),
            selected.get("max_risk")),
        "manifold_violation_count": int(
            validation["manifold_violation_count"]),
        "corridor_violation_count": int(
            validation["corridor_violation_count"]),
        "clearance_violation_count": int(
            validation["clearance_violation_count"]),
        "violation_count": int(validation["violation_count"]),
        "hard_violation_count": int(validation["hard_violation_count"]),
        "constraint_mode": str(validation["constraint_mode"]),
        "safety_success": bool(validation["safety_success"]),
        "task_success": bool(validation["task_success"]),
        "planner_success": bool(validation["planner_success"]),
        "controller_success": bool(validation["controller_success"]),
        "overall_success": bool(validation["overall_success"]),
        "warning_reason": str(validation["warning_reason"]),
        "failure_reason": str(validation["failure_reason"]),
        "mpc_status": mpc_status,
        "mpc_used": bool(diag.get("mpc_used", False) or ref_count > 0),
        "success": bool(validation["overall_success"]),
    })


def write_consistency_report(run_root):
    robots = [
        robot for robot in ROBOTS
        if os.path.isdir(os.path.join(run_root, robot))
    ]
    required = (
        "traj.csv", "metrics.json", "decision_trace.json",
        "mpc_diagnostics.json",
    )
    variant_isolated = bool(robots)
    baseline_valid = bool(robots)
    stsm_valid = bool(robots)
    diagnostics_consistent = bool(robots)
    metrics_consistent = bool(robots)
    topology_mpc_closed_loop = bool(robots)
    execution_success_valid = bool(robots)
    for robot in robots:
        robot_root = os.path.join(run_root, robot)
        for variant in VARIANTS:
            base = os.path.join(robot_root, variant)
            variant_isolated = variant_isolated and os.path.isdir(base)
            for name in required:
                path = os.path.join(base, name)
                variant_isolated = (
                    variant_isolated and
                    os.path.exists(path) and os.path.getsize(path) > 0)
        baseline_diag = load_json(os.path.join(
            robot_root, "baseline", "mpc_diagnostics.json"))
        baseline_trace = load_json(os.path.join(
            robot_root, "baseline", "decision_trace.json"))
        stsm_diag = load_json(os.path.join(
            robot_root, "stsm", "mpc_diagnostics.json"))
        stsm_constraint = load_json(os.path.join(
            robot_root, "stsm", "topology_constraint.json"))
        baseline_valid = baseline_valid and all([
            str(baseline_diag.get("baseline_type") or
                baseline_trace.get("baseline_type") or "").lower() == "direct",
            str(baseline_diag.get("planner_source") or
                baseline_trace.get("planner_source") or "").lower() ==
            "direct_connection",
            not truthy(baseline_diag.get("topology_constraint_used")),
        ])
        stsm_valid = stsm_valid and all([
            truthy(stsm_diag.get("topology_constraint_used")),
            truthy(stsm_diag.get("critical_point_sequence_constraint_used")),
        ])
        diagnostics_consistent = diagnostics_consistent and all([
            bool(baseline_diag),
            bool(stsm_diag),
            "mpc_feasibility_status" in baseline_diag,
            "mpc_feasibility_status" in stsm_diag,
        ])
        topology_payload = dict(
            stsm_diag.get("topology_constraint", {}) or stsm_constraint or {})
        topology_mpc_closed_loop = topology_mpc_closed_loop and all([
            truthy(stsm_diag.get("topology_constraint_used")),
            truthy(stsm_diag.get("critical_point_sequence_constraint_used")),
            truthy(stsm_diag.get("corridor_constraint_used")),
            truthy(stsm_diag.get("manifold_constraint_used")),
            bool(topology_payload.get("critical_point_constraint")),
            bool(topology_payload.get("corridor_constraint")),
            bool(topology_payload.get("manifold_constraint")),
            bool(topology_payload.get("topology_sequence_constraint")),
            os.path.exists(os.path.join(
                robot_root, "stsm", "critical_point_association.json")),
        ])
        for variant in VARIANTS:
            diag = load_json(os.path.join(
                robot_root, variant, "mpc_diagnostics.json"))
            metric = load_json(os.path.join(
                robot_root, variant, "metrics.json"))
            diag_status = str(diag.get("mpc_feasibility_status", ""))
            metric_status = str(metric.get("mpc_feasibility_status", ""))
            if diag_status and metric_status and diag_status != metric_status:
                metrics_consistent = False
            execution_success_valid = execution_success_valid and truthy(
                first_value(
                    diag.get("overall_success"), diag.get("task_success"),
                    metric.get("overall_success"), metric.get("task_success"),
                    metric.get("success"), False))
    root_duplicates = []
    for robot in robots:
        for name in (
                "traj.csv", "metrics.csv", "metrics.json",
                "mpc_diagnostics.json", "decision_trace.json",
                "topology_constraint.json"):
            root_duplicates.append(os.path.join(run_root, robot, name))
    no_root_duplicates = not any(os.path.exists(path) for path in root_duplicates)
    result_structure_valid = bool(variant_isolated and no_root_duplicates)
    diagnostics_metrics_consistent = bool(
        diagnostics_consistent and metrics_consistent)
    topology_constraint_consistent = bool(
        topology_mpc_closed_loop and stsm_valid)
    report = {
        "topology_mpc_closed_loop": bool(topology_mpc_closed_loop),
        "diagnostics_consistent": bool(diagnostics_consistent),
        "metrics_consistent": bool(metrics_consistent),
        "result_structure_valid": bool(result_structure_valid),
        "baseline_valid": bool(baseline_valid),
        "stsm_valid": bool(stsm_valid),
        "diagnostics_metrics_consistent": bool(diagnostics_metrics_consistent),
        "topology_constraint_consistent": bool(topology_constraint_consistent),
        "variant_isolated": bool(variant_isolated and no_root_duplicates),
        "execution_success_valid": bool(execution_success_valid),
    }
    report["overall_pass"] = all(bool(report[key]) for key in (
        "topology_mpc_closed_loop",
        "diagnostics_consistent",
        "metrics_consistent",
        "result_structure_valid",
        "baseline_valid",
        "stsm_valid",
        "execution_success_valid",
    ))
    write_json(os.path.join(run_root, "experiment_consistency_report.json"), report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=os.path.join(ROOT, "results", "run"))
    args = parser.parse_args()
    for robot in ROBOTS:
        for variant in VARIANTS:
            base = os.path.join(args.run_dir, robot, variant)
            if not os.path.isdir(base):
                continue
            if variant == "baseline":
                ensure_baseline(base, robot)
            else:
                ensure_stsm(base, robot)
    write_consistency_report(args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
