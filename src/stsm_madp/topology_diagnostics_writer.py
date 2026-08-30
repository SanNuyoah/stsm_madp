import sys
sys.dont_write_bytecode = True

import json
import os

import numpy as np

from stsm_madp.task_config import (
    resolve_task_mode,
    resolve_task_weight,
    weighted_task_candidate_cost,
)
from stsm_madp.task_semantics import infer_task_state


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "__dict__"):
        return _jsonable(dict(value.__dict__))
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)
    return value


def _failure_tokens(value):
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            out.extend(_failure_tokens(item))
        return out
    return [
        token.strip()
        for token in str(value or "").replace(";", ",").split(",")
        if token.strip()
    ]


def _recoverable_level(row):
    level = str(row.get("recoverable_level", "") or "")
    if level:
        return level
    reasons = set(_failure_tokens(row.get("failure_reason", "")))
    if reasons.intersection(set([
            "link_collision", "arm_link_collision", "self_collision",
            "collision", "ik_failed", "ik_failure", "ik_or_link_collision",
            "workspace_violation", "workspace_exceeded",
            "workspace_out_of_bounds"])):
        return "level3"
    if reasons.intersection(set([
            "pose_adjustment", "pose_adjustment_required",
            "pose_optimization_required", "position_error"])):
        return "level2"
    if reasons.intersection(set([
            "end_effector_clearance_violation", "orientation_error"])):
        return "level1"
    return "none"


def _int_value(payload, key, default=0):
    try:
        return int(payload.get(key, default) or default)
    except Exception:
        return int(default)


def _float_value(payload, key, default=0.0):
    try:
        value = payload.get(key, default)
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _float_any(payload, keys, default=0.0):
    for key in keys:
        value = payload.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except Exception:
            continue
    return float(default)


def _candidate_id(row):
    return str(row.get(
        "candidate_id",
        row.get("corridor_id", row.get("route_id", row.get("label", "")))))


def _points_from_row(row):
    for key in ("refined_waypoints", "waypoints", "centerline",
                "raw_topology_waypoints", "points"):
        pts = row.get(key)
        if isinstance(pts, (list, tuple)) and len(pts) >= 2:
            out = []
            for point in pts:
                try:
                    arr = np.asarray(point, float).reshape(-1)
                    if arr.size >= 2:
                        out.append(arr[:min(3, arr.size)])
                except Exception:
                    pass
            if len(out) >= 2:
                return out
    return []


def _path_length(points):
    total = 0.0
    for a, b in zip(points[:-1], points[1:]):
        aa = np.asarray(a, float)
        bb = np.asarray(b, float)
        dim = min(aa.size, bb.size)
        total += float(np.linalg.norm(bb[:dim] - aa[:dim]))
    return total


def _path_smoothness(points):
    total = 0.0
    for a, b, c in zip(points[:-2], points[1:-1], points[2:]):
        aa = np.asarray(a, float)
        bb = np.asarray(b, float)
        cc = np.asarray(c, float)
        dim = min(aa.size, bb.size, cc.size)
        v1 = bb[:dim] - aa[:dim]
        v2 = cc[:dim] - bb[:dim]
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 <= 1e-9 or n2 <= 1e-9:
            continue
        cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        total += float(np.arccos(cosang))
    return total


def _candidate_ranking_rows(debug, candidate_report, filter_report,
                            morse_routes=None, morse_route_evaluation=None):
    # A final ranking is the canonical set of execution-legal candidates.  Do
    # not mix it with earlier generator/filter records from another stage.
    final_ranking = list(debug.get("final_candidate_ranking") or [])
    has_final_ranking = bool(final_ranking)
    primary_candidates = list(
        final_ranking or
        debug.get("candidate_corridors") or
        debug.get("candidate_after_filter") or
        debug.get("candidate_after_top_k") or
        debug.get("arm_route_ranking") or
        candidate_report.get("candidate_corridors", []) or
        candidate_report.get("candidate_after_filter", []) or
        candidate_report.get("arm_route_ranking", []) or [])
    candidates = []
    seen_candidates = set()
    sources = [primary_candidates]
    if not has_final_ranking:
        sources.extend((
            candidate_report.get("candidate_filter_report", []) or [],
            filter_report or []))
    for source in sources:
        for item in source:
            if not isinstance(item, dict):
                continue
            cid = _candidate_id(item)
            if cid and cid in seen_candidates:
                continue
            status = str(item.get("candidate_status", ""))
            filter_class = str(item.get(
                "candidate_filter_class", status))
            eligible_status = (
                filter_class in ("safe", "recoverable") or
                status in ("feasible", "safe", "recoverable"))
            has_candidate_score = any(
                key in item for key in (
                    "candidate_cost", "total_cost", "total_score",
                    "risk_cost", "task_cost"))
            if not eligible_status and not has_candidate_score:
                continue
            candidates.append(item)
            if cid:
                seen_candidates.add(cid)
    selected_id = str(
        debug.get("selected_corridor_id") or
        debug.get("execution_corridor_id") or
        debug.get("selected_candidate") or "")
    task_mode = resolve_task_mode(
        debug.get("task_mode") or candidate_report.get("task_mode", ""))
    task_weight = resolve_task_weight(
        task_mode,
        task_config=debug.get("task_config") or candidate_report.get(
            "task_config", {}),
        task_weight=debug.get("task_weight") or candidate_report.get(
            "task_weight", {}))
    route_by_id = {}
    for sources in (morse_routes or [], morse_route_evaluation or []):
        for source in sources:
            if not isinstance(source, dict):
                continue
            rid = _candidate_id(source)
            if rid:
                route_by_id.setdefault(rid, {}).update(source)
    rows = []
    for idx, item in enumerate(candidates):
        if not isinstance(item, dict):
            continue
        cid = _candidate_id(item) or "candidate_{:04d}".format(idx + 1)
        route = route_by_id.get(cid, {})
        merged = dict(route)
        merged.update(item)
        points = _points_from_row(merged)
        risk_cost = _float_any(
            merged, ("risk_cost", "risk_value", "max_risk", "risk"), 0.0)
        length_cost = _float_any(
            merged, ("length_cost", "distance_cost", "path_length", "length"),
            _path_length(points))
        smoothness_cost = _float_any(
            merged, ("smoothness_cost", "smooth_cost", "curvature_cost",
                     "turning_cost", "max_route_turn"),
            _path_smoothness(points))
        task_cost = _float_any(merged, ("task_cost",), 0.0)
        clearance_cost = _float_any(merged, ("clearance_cost",), 0.0)
        feasibility_cost = _float_any(merged, ("feasibility_cost",), 0.0)
        raw_recovery_cost = _float_any(
            merged, ("raw_recovery_cost", "recovery_cost"), 0.0)
        normalized_recovery_cost = _float_any(
            merged, ("normalized_recovery_cost", "recovery_cost"), 0.0)
        recovery_cost = _float_any(merged, ("recovery_cost",), 0.0)
        recovery_weight = _float_any(merged, ("recovery_weight",), 0.3)
        task_specific_cost = _float_any(merged, ("task_specific_cost",), 0.0)
        task_candidate_cost = weighted_task_candidate_cost(
            risk_cost, length_cost, task_cost, task_weight)
        total_cost = _float_any(
            merged,
            (("total_cost_with_adp", "total_cost", "total_score",
              "candidate_cost") if has_final_ranking else
             ("candidate_cost", "total_cost", "total_score")),
            risk_cost + length_cost + smoothness_cost + task_cost +
            feasibility_cost + 0.3 * normalized_recovery_cost)
        selected = bool(merged.get("selected", False)) or (
            bool(selected_id) and cid == selected_id)
        filter_class = str(merged.get(
            "candidate_filter_class", merged.get("candidate_status", "")))
        candidate_status = str(merged.get("candidate_status", ""))
        if filter_class == "safe" or candidate_status == "feasible":
            candidate_status = "safe"
        cost_breakdown = dict(merged.get("candidate_cost_breakdown", {}) or {})
        cost_breakdown.update({
            "risk_cost": float(risk_cost),
            "length_cost": float(length_cost),
            "smoothness_cost": float(smoothness_cost),
            "task_cost": float(task_cost),
            "clearance_cost": float(clearance_cost),
            "feasibility_cost": float(feasibility_cost),
            "raw_recovery_cost": float(raw_recovery_cost),
            "normalized_recovery_cost": float(normalized_recovery_cost),
            "recovery_cost": float(recovery_cost),
            "recovery_weight": float(recovery_weight),
            "ranking_score": float(total_cost),
            "task_specific_cost": float(task_specific_cost),
            "candidate_cost": float(total_cost),
        })
        # Keep the final ranking record whole, then enrich it with the shared
        # diagnostics.  Rebuilding a fresh generic record loses ADP evidence.
        row = dict(merged)
        row.update({
            "rank": int(merged.get("rank_after_adp", merged.get("rank", 0)) or 0),
            "candidate_id": cid,
            "risk_cost": float(risk_cost),
            "length_cost": float(length_cost),
            "smoothness_cost": float(smoothness_cost),
            "task_cost": float(task_cost),
            "clearance_cost": float(clearance_cost),
            "feasibility_cost": float(feasibility_cost),
            "raw_recovery_cost": float(raw_recovery_cost),
            "normalized_recovery_cost": float(normalized_recovery_cost),
            "recovery_cost": float(recovery_cost),
            "recovery_weight": float(recovery_weight),
            "recoverable_level": _recoverable_level(merged),
            "ranking_score": float(total_cost),
            "task_specific_cost": float(task_specific_cost),
            "candidate_cost": float(total_cost),
            "total_score": float(total_cost),
            "candidate_cost_breakdown": cost_breakdown,
            "task_candidate_cost": float(task_candidate_cost),
            "total_cost": total_cost,
            "selected": bool(selected),
            "selected_reason": str(
                merged.get("selected_reason") or
                ("lowest_candidate_cost_after_task_safety_feasibility_ranking"
                 if selected else "")),
            "task_mode_influence": dict(merged.get(
                "task_mode_influence", {}) or {
                    "task_mode": str(task_mode),
                    "task_cost": float(task_cost),
                    "task_candidate_cost": float(task_candidate_cost),
                    "task_weight": dict(task_weight),
                }),
            "task_mode": str(task_mode),
            "task_state": str(merged.get("task_state", "")),
            "task_cost_breakdown": dict(
                merged.get("task_cost_breakdown", {}) or {}),
            "task_weight": dict(task_weight),
            "task_weight_used": True,
            "candidate_status": candidate_status,
            "candidate_filter_class": filter_class,
            "failure_reason": merged.get("failure_reason", ""),
        })
        if has_final_ranking:
            row["ranking_record_stage"] = "final_legal"
            row.setdefault("base_total_cost", float(total_cost))
            row.setdefault("total_cost_with_adp", float(total_cost))
            row.setdefault("rank_before_adp", int(row["rank"]))
            row.setdefault("rank_after_adp", int(row["rank"]))
        rows.append(row)
    if has_final_ranking:
        rows.sort(key=lambda row: (
            int(row.get("rank_after_adp", row.get("rank", 0)) or 0),
            float(row.get("total_cost_with_adp", row.get("total_cost", 0.0)) or 0.0),
            row["candidate_id"]))
        for rank, row in enumerate(rows, start=1):
            row["rank"] = int(row.get("rank_after_adp", rank) or rank)
        return rows
    max_raw_recovery = max(
        [float(row.get("raw_recovery_cost", 0.0) or 0.0) for row in rows] or
        [0.0])
    for row in rows:
        raw_recovery_cost = float(row.get("raw_recovery_cost", 0.0) or 0.0)
        normalized_recovery_cost = (
            raw_recovery_cost / max_raw_recovery
            if max_raw_recovery > 1e-9 else 0.0)
        normalized_recovery_cost = max(0.0, min(1.0, normalized_recovery_cost))
        total_cost = float(
            float(row.get("risk_cost", 0.0) or 0.0) +
            float(row.get("length_cost", 0.0) or 0.0) +
            float(row.get("smoothness_cost", 0.0) or 0.0) +
            float(row.get("task_cost", 0.0) or 0.0) +
            float(row.get("feasibility_cost", 0.0) or 0.0) +
            0.3 * normalized_recovery_cost)
        row["normalized_recovery_cost"] = float(normalized_recovery_cost)
        row["recovery_cost"] = float(normalized_recovery_cost)
        row["total_cost"] = float(total_cost)
        row["total_score"] = float(total_cost)
        row["candidate_cost"] = float(total_cost)
        row["ranking_score"] = float(total_cost)
        breakdown = dict(row.get("candidate_cost_breakdown", {}) or {})
        breakdown.update({
            "normalized_recovery_cost": float(normalized_recovery_cost),
            "recovery_cost": float(normalized_recovery_cost),
            "weighted_recovery_cost": float(0.3 * normalized_recovery_cost),
            "candidate_cost": float(total_cost),
            "ranking_score": float(total_cost),
        })
        row["candidate_cost_breakdown"] = breakdown
    rows.sort(key=lambda row: (
        float(row["total_cost"]),
        0 if bool(row["selected"]) else 1,
        row["candidate_id"]))
    if rows and not any(bool(row.get("selected", False)) for row in rows):
        rows[0]["selected"] = True
        rows[0]["selected_reason"] = (
            "lowest_candidate_cost_after_task_safety_feasibility_ranking")
    for rank, row in enumerate(rows, start=1):
        row["rank"] = int(rank)
    return rows


def _stsm_alias_dir(base_dir, robot_type):
    if not base_dir:
        return ""
    norm = os.path.abspath(base_dir).replace("\\", "/")
    robot = str(robot_type or "").strip().lower()
    upper_suffix = "/{}/STSM".format(robot)
    lower_suffix = "/{}/stsm".format(robot)
    if norm.endswith(upper_suffix):
        return norm[:-len(upper_suffix)] + lower_suffix
    if norm.lower().endswith(lower_suffix):
        return norm[:-len(lower_suffix)] + lower_suffix
    return norm


def write_stsm_candidate_ranking_alias(base_dir, robot_type, ranking_rows):
    alias = _stsm_alias_dir(base_dir, robot_type)
    if not alias:
        return
    if not os.path.isdir(alias):
        os.makedirs(alias)
    with open(os.path.join(alias, "candidate_ranking.json"), "w") as f:
        json.dump(_jsonable(ranking_rows), f, indent=2, sort_keys=True)


def write_failed_topology_diagnostics(base_dir, robot_type, debug=None,
                                      failure_reason="planning_failed"):
    if not base_dir:
        return
    base_dir = _stsm_alias_dir(base_dir, robot_type)
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    debug = dict(debug or {})
    candidate_report = dict(debug.get("candidate_generation_report", {}) or {})
    filter_report = list(
        debug.get("candidate_filter_report") or
        candidate_report.get("candidate_filter_report", []) or [])
    morse_routes = list(debug.get("morse_routes") or [])
    morse_route_evaluation = list(debug.get("morse_route_evaluation") or [])
    task_mode = resolve_task_mode(
        debug.get("task_mode") or candidate_report.get("task_mode", ""),
        robot_type=robot_type)
    task_weight = resolve_task_weight(
        task_mode,
        task_config=debug.get("task_config") or candidate_report.get(
            "task_config", {}),
        task_weight=debug.get("task_weight") or candidate_report.get(
            "task_weight", {}),
        robot_type=robot_type)
    debug["task_mode"] = task_mode
    debug["task_weight"] = task_weight
    debug["task_weight_used"] = True
    candidate_report.setdefault("task_mode", task_mode)
    candidate_report.setdefault("task_weight", task_weight)
    candidate_report.setdefault("task_weight_used", True)
    morse_diag = dict(debug.get("morse_diagnostics", {}) or {})
    if not morse_diag:
        route_count = _int_value(debug, "route_count", len(morse_routes))
        candidate_count = _int_value(
            debug, "num_candidate_corridors",
            _int_value(candidate_report, "num_candidates_generated", 0))
        num_minima = _int_value(debug, "num_minima",
                                _int_value(debug, "num_critical_minima", 0))
        num_saddle = _int_value(debug, "num_saddle",
                                _int_value(debug, "num_critical_saddles", 0))
        graph_nodes = _int_value(debug, "graph_nodes",
                                 _int_value(debug, "num_topology_nodes", 0))
        graph_edges = _int_value(debug, "graph_edges",
                                 _int_value(debug, "num_topology_edges", 0))
        morse_diag = {
            "robot_type": str(robot_type),
            "num_minima": int(num_minima),
            "num_saddle": int(num_saddle),
            "num_critical_points": int(num_minima + num_saddle),
            "graph_nodes": int(graph_nodes),
            "graph_edges": int(graph_edges),
            "route_count": int(route_count),
            "candidate_count": int(candidate_count),
            "route_generation_status": (
                "ok" if route_count > 0 and candidate_count > 0 else
                "route_search_failed"),
            "route_source": str(debug.get("route_source", "morse_topology")),
            "failure_reason": str(failure_reason),
            "semantic_topology_recovery_used": bool(debug.get(
                "semantic_topology_recovery_used", False)),
        }
    else:
        morse_diag.setdefault("robot_type", str(robot_type))
        morse_diag.setdefault("failure_reason", str(failure_reason))
        morse_diag.setdefault("candidate_count", _int_value(
            debug, "num_candidate_corridors",
            _int_value(candidate_report, "num_candidates_generated", 0)))

    if not candidate_report:
        candidate_report = {
            "candidate_generated": int(morse_diag.get("candidate_count", 0) or 0),
            "num_candidates_generated": int(morse_diag.get("candidate_count", 0) or 0),
            "candidate_filter_report": list(filter_report),
            "failure_reason": str(failure_reason),
        }
    if not filter_report and int(morse_diag.get("candidate_count", 0) or 0) == 0:
        filter_report = [{
            "candidate_id": "planning_failed",
            "candidate_status": "invalid",
            "failure_reason": [str(failure_reason)],
            "geometry_valid": False,
            "manifold_valid": False,
            "tube_valid": False,
            "selected": False,
        }]

    graph = {
        "robot_type": str(robot_type),
        "generated": bool(int(morse_diag.get("graph_nodes", 0) or 0) > 0),
        "node_count": int(morse_diag.get("graph_nodes", 0) or 0),
        "edge_count": int(morse_diag.get("graph_edges", 0) or 0),
        "route_count": int(morse_diag.get("route_count", 0) or 0),
        "candidate_count": int(morse_diag.get("candidate_count", 0) or 0),
        "nodes": _jsonable(debug.get("nodes", [])),
        "edges": _jsonable(debug.get("edges", [])),
        "failure_reason": str(failure_reason),
    }
    refinement_attempts = list(debug.get("refinement_attempts") or [])
    refinement_trace = list(debug.get("refinement_trace") or [])
    if refinement_attempts and (
            not refinement_trace or
            (len(refinement_trace) == 1 and isinstance(refinement_trace[0], dict)
             and "attempts" in refinement_trace[0])):
        refinement_trace = list(refinement_attempts)
    topology_refinement = dict(debug.get("topology_refinement", {}) or {})
    if not topology_refinement and (refinement_attempts or refinement_trace):
        attempts = refinement_attempts or refinement_trace
        topology_refinement = {
            "robot_type": str(robot_type),
            "attempted": bool(attempts),
            "attempt_count": int(len(attempts)),
            "accepted_count": int(sum(
                1 for item in attempts
                if isinstance(item, dict) and bool(item.get("accepted", False)))),
            "rejected_count": int(sum(
                1 for item in attempts
                if not (isinstance(item, dict) and
                        bool(item.get("accepted", False))))),
            "failure_reason": str(failure_reason),
            "attempts": list(attempts),
        }

    ranking_rows = _candidate_ranking_rows(
        debug, candidate_report, filter_report, morse_routes,
        morse_route_evaluation)
    recovery_ranking = list(ranking_rows)
    candidate_task_breakdown = list(
        debug.get("candidate_task_cost_breakdown") or
        candidate_report.get("candidate_task_cost_breakdown") or
        [{
            "rank": row.get("rank", 0),
            "candidate_id": row.get("candidate_id", ""),
            "selected": bool(row.get("selected", False)),
            "task_mode": row.get("task_mode", ""),
            "task_state": row.get("task_state", ""),
            "distance_cost": (
                row.get("task_cost_breakdown", {}).get("terms", {})
                if isinstance(row.get("task_cost_breakdown", {}), dict)
                else {}).get("distance_cost", 0.0),
            "orientation_cost": (
                row.get("task_cost_breakdown", {}).get("terms", {})
                if isinstance(row.get("task_cost_breakdown", {}), dict)
                else {}).get("orientation_cost", 0.0),
            "feasibility_cost": (
                row.get("task_cost_breakdown", {}).get("terms", {})
                if isinstance(row.get("task_cost_breakdown", {}), dict)
                else {}).get("feasibility_cost", row.get("feasibility_cost", 0.0)),
            "interaction_cost": (
                row.get("task_cost_breakdown", {}).get("terms", {})
                if isinstance(row.get("task_cost_breakdown", {}), dict)
                else {}).get("interaction_cost", 0.0),
            "task_cost": row.get("task_cost", 0.0),
            "total_task_cost": row.get("task_cost", 0.0),
            "task_cost_breakdown": row.get("task_cost_breakdown", {}),
            "ranking_score": row.get("ranking_score", row.get("total_cost", 0.0)),
            "selection_reason": row.get("selected_reason", ""),
        } for row in ranking_rows])
    task_state_diag = dict(debug.get("task_state_diagnostics", {}) or {})
    if not task_state_diag:
        task_state_diag = infer_task_state(
            robot_type, task_mode,
            phase=debug.get("current_phase", debug.get("phase", "")),
            progress=debug.get("progress", 0.0))
    outputs = {
        "morse_diagnostics.json": morse_diag,
        "candidate_filter_report.json": filter_report,
        "candidate_generation_report.json": candidate_report,
        "candidate_ranking.json": ranking_rows,
        "candidate_feature_delta_summary.json": dict(debug.get(
            "candidate_feature_delta_summary", {}) or {}),
        "candidate_recovery_ranking.json": recovery_ranking,
        "candidate_task_cost_breakdown.json": candidate_task_breakdown,
        "task_state_diagnostics.json": task_state_diag,
        "morse_routes.json": morse_routes,
        "morse_route_evaluation.json": morse_route_evaluation,
        "candidate_statistics.json": dict(debug.get(
            "candidate_statistics", {}) or {}),
        "topology_graph.json": graph,
        "refinement_trace.json": refinement_trace,
        "topology_refinement.json": topology_refinement,
    }
    for name, payload in outputs.items():
        with open(os.path.join(base_dir, name), "w") as f:
            json.dump(_jsonable(payload), f, indent=2, sort_keys=True)
    write_stsm_candidate_ranking_alias(base_dir, robot_type, ranking_rows)
