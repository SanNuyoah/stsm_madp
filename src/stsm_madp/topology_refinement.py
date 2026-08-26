import sys
sys.dont_write_bytecode = True

import numpy as np

from stsm_madp.deform import (
    bezier_smooth_polyline,
    path_curvature_metrics,
    path_length,
)
from stsm_madp.critical_point_association import associate_corridor_critical_points
from stsm_madp.manifold_constraint import (
    distance_to_manifold_boundary,
    manifold_risk_value,
)
from stsm_madp.safety_evaluator import SafetyEvaluator


def _as_points(path):
    if path is None or isinstance(path, str):
        return np.zeros((0, 3), float)
    pts = np.asarray(path, float)
    if pts.size == 0:
        return np.zeros((0, 3), float)
    if pts.ndim == 0:
        return np.zeros((0, 3), float)
    if pts.ndim == 1:
        pts = pts.reshape((1, pts.shape[0]))
    if pts.shape[1] == 2:
        pts = np.hstack([pts, np.zeros((pts.shape[0], 1), float)])
    return pts[:, :3]


def _limit_refinement_points(points, max_points=0, protected_points=None):
    """Bound refinement work while preserving path endpoints.

    Topology enumeration may hand refinement a dense centerline.  For
    wheelchair execution we only need a bounded representative polyline here:
    safety/tube validity is still checked after refinement, and the downstream
    MPC reference is generated from the accepted geometry.
    """
    pts = _as_points(points)
    max_points = int(max_points or 0)
    if max_points <= 0 or len(pts) <= max_points:
        return pts.copy(), False
    if max_points <= 2:
        return np.asarray([pts[0], pts[-1]], float), True
    protected = _as_points(protected_points)
    protected_indices = []
    if len(protected):
        for waypoint in protected:
            protected_indices.append(int(np.argmin(np.linalg.norm(
                pts - waypoint.reshape(1, 3), axis=1))))
    budget = max(2, max_points - len(set(protected_indices)))
    keep = np.linspace(0, len(pts) - 1, budget)
    indices = sorted(set(
        [int(round(v)) for v in keep] + protected_indices))
    if indices[0] != 0:
        indices.insert(0, 0)
    if indices[-1] != len(pts) - 1:
        indices.append(len(pts) - 1)
    return np.asarray([pts[i] for i in indices], float), True


def _boundary_points(boundary):
    if boundary is None or isinstance(boundary, str):
        return np.zeros((0, 3), float)
    if isinstance(boundary, dict):
        pts = []
        for key in ("left", "right", "boundary", "points"):
            pts.extend(boundary.get(key, []) or [])
        return _as_points(pts)
    return _as_points(boundary)


def _point_to_polyline_distance(point, boundary):
    return distance_to_manifold_boundary(point, boundary)


def _risk_value(point, risk_field):
    return manifold_risk_value(point, risk_field)


def _manifold_payload(manifold_constraint=None, corridor=None, risk_field=None):
    payload = dict(manifold_constraint or {})
    if not payload and corridor is not None:
        payload = dict(getattr(corridor, "manifold_constraint", {}) or {})
    boundary = payload.get("boundary")
    if not boundary and corridor is not None:
        boundary = getattr(corridor, "boundary", {})
    threshold = payload.get(
        "risk_threshold", payload.get("safe_threshold", None))
    if threshold in (None, ""):
        threshold = getattr(risk_field, "rho", 1.0) if risk_field is not None else 1.0
    minimum_clearance = payload.get(
        "minimum_clearance", payload.get("min_clearance", None))
    if minimum_clearance in (None, ""):
        minimum_clearance = getattr(corridor, "min_corridor_clearance", 0.0) if corridor is not None else 0.0
    return {
        "boundary": boundary,
        "risk_threshold": float(threshold),
        "minimum_clearance": float(minimum_clearance or 0.0),
    }


def _corridor_payload(corridor_constraint=None, corridor=None):
    payload = dict(corridor_constraint or {})
    centerline = payload.get("centerline")
    if centerline is None and corridor is not None:
        centerline = getattr(corridor, "centerline", None)
    if centerline is None and corridor is not None:
        centerline = getattr(corridor, "waypoints", None)
    radius = payload.get("radius", None)
    if radius in (None, "") and corridor is not None:
        radius = getattr(corridor, "radius", 0.0)
    return {
        "centerline": _as_points(centerline),
        "radius": float(radius or 0.0),
    }


def _distance_to_corridor(point, corridor_constraint):
    centerline = corridor_constraint.get("centerline")
    radius = float(corridor_constraint.get("radius", 0.0))
    if centerline is None or len(centerline) == 0 or radius <= 0.0:
        return 0.0, True
    d = _point_to_polyline_distance(point, centerline)
    return float(d), bool(d <= radius + 1e-9)


def check_refinement_manifold_validity(refined_trajectory,
                                       manifold_constraint=None,
                                       corridor_constraint=None,
                                       risk_field=None):
    pts = _as_points(refined_trajectory)
    manifold = _manifold_payload(manifold_constraint, risk_field=risk_field)
    corridor_payload = _corridor_payload(corridor_constraint)
    evaluator = SafetyEvaluator(
        manifold_constraint=manifold,
        corridor_constraint=corridor_payload,
        risk_field=risk_field,
        planning_clearance_margin=0.0)
    status = evaluator.evaluate_trajectory(pts)
    tube_status = evaluator.evaluate_corridor({
        "centerline": corridor_payload.get("centerline", []),
    })
    clearance = float(status.get("min_clearance", 0.0))
    risk = float(status.get("max_risk", 0.0))
    minimum_clearance = float(manifold["minimum_clearance"])
    risk_threshold = float(manifold["risk_threshold"])
    clearance_valid = bool(clearance + 1e-9 >= minimum_clearance)
    risk_valid = bool(risk <= risk_threshold + 1e-9)
    tube_valid = bool(
        tube_status.get("tube_valid", True) and
        int(status.get("corridor_violation_count", 0)) == 0)
    manifold_valid = bool(
        int(status.get("manifold_violation_count", 0)) == 0 and
        clearance_valid and risk_valid)
    return {
        "valid": bool(manifold_valid and tube_valid),
        "manifold_valid": bool(manifold_valid),
        "tube_valid": bool(tube_valid),
        "clearance_valid": bool(clearance_valid),
        "risk_valid": bool(risk_valid),
        "min_clearance": float(clearance),
        "max_risk": float(risk),
        "violation_count": int(status.get("manifold_violation_count", 0)),
        "corridor_violation_count": int(
            status.get("corridor_violation_count", 0)),
        "manifold_violation_count": int(
            status.get("manifold_violation_count", 0)),
        "refinement_tube_valid": bool(tube_valid),
        "minimum_clearance": float(minimum_clearance),
        "risk_threshold": float(risk_threshold),
    }


def _manifold_penalty(path, manifold_constraint, risk_field=None):
    status = check_refinement_manifold_validity(
        path, manifold_constraint=manifold_constraint, risk_field=risk_field)
    clearance_gap = max(
        0.0,
        float(status["minimum_clearance"]) - float(status["min_clearance"]))
    risk_gap = max(
        0.0,
        float(status["max_risk"]) - float(status["risk_threshold"]))
    return float(clearance_gap ** 2 + risk_gap ** 2), status


def _tube_penalty(status):
    return float(
        max(0, int(status.get("corridor_violation_count", 0))) +
        max(0, int(status.get("manifold_violation_count", 0))))


def _safety_cost(path, status, manifold_constraint):
    smooth_cost = float(path_length(path))
    clearance_gap = max(
        0.0,
        float(status.get("minimum_clearance", 0.0)) -
        float(status.get("min_clearance", 0.0)))
    risk_gap = max(
        0.0,
        float(status.get("max_risk", 0.0)) -
        float(status.get("risk_threshold", 0.0)))
    tube_gap = _tube_penalty(status)
    # Safety dominates smoothing; smoothing only breaks ties between safe paths.
    return float(
        1.0 * smooth_cost +
        200.0 * clearance_gap * clearance_gap +
        80.0 * tube_gap +
        40.0 * risk_gap * risk_gap)


def _safety_failure_reason(status, pre_clearance=None, tolerance=0.005):
    if pre_clearance is not None:
        if float(status.get("min_clearance", 0.0)) + 1e-9 < (
                float(pre_clearance) - float(tolerance)):
            return "clearance_regression"
    if not bool(status.get("manifold_valid", False)):
        if not bool(status.get("clearance_valid", False)):
            return "clearance_violation"
        if not bool(status.get("risk_valid", False)):
            return "risk_violation"
        return "manifold_violation"
    if not bool(status.get("tube_valid", False)):
        return "tube_violation"
    return ""


def refine_trajectory(trajectory, corridor_constraint=None,
                      manifold_constraint=None, risk_field=None,
                      samples_per_segment=12, max_iterations=4):
    original = _as_points(trajectory)
    trace = []
    pre_status = check_refinement_manifold_validity(
        original, manifold_constraint=manifold_constraint,
        corridor_constraint=corridor_constraint, risk_field=risk_field)
    best = original.copy()
    best_status = dict(pre_status)
    pre_clearance = float(pre_status.get("min_clearance", 0.0))
    best_cost = _safety_cost(best, best_status, manifold_constraint)
    pre_reason = _safety_failure_reason(best_status)
    trace.append({
        "iteration": 0,
        "trajectory_cost": float(best_cost),
        "cost": float(best_cost),
        "clearance": float(best_status["min_clearance"]),
        "min_clearance": float(best_status["min_clearance"]),
        "risk": float(best_status["max_risk"]),
        "tube_valid": bool(best_status.get("tube_valid", False)),
        "manifold_valid": bool(best_status.get("manifold_valid", False)),
        "accepted": bool(best_status["valid"]),
        "failure_reason": pre_reason,
        "trajectory_valid": bool(best_status["valid"]),
    })
    if len(original) <= 2:
        return best, {
            "trace": trace,
            "pre_status": pre_status,
            "post_status": best_status,
            "fallback": False,
        }
    candidate = bezier_smooth_polyline(
        original, samples_per_segment=samples_per_segment)
    for iteration in range(1, max(1, int(max_iterations)) + 1):
        alpha = 1.0 / float(iteration)
        proposal = original + alpha * (candidate[:len(original)] - original) if len(candidate) == len(original) else candidate
        status = check_refinement_manifold_validity(
            proposal, manifold_constraint=manifold_constraint,
            corridor_constraint=corridor_constraint, risk_field=risk_field)
        cost = _safety_cost(proposal, status, manifold_constraint)
        reason = _safety_failure_reason(
            status, pre_clearance=pre_clearance, tolerance=0.005)
        accepted = bool(not reason and status["valid"] and (
            not best_status["valid"] or cost <= best_cost))
        trace.append({
            "iteration": int(iteration),
            "trajectory_cost": float(cost),
            "cost": float(cost),
            "clearance": float(status["min_clearance"]),
            "min_clearance": float(status["min_clearance"]),
            "risk": float(status["max_risk"]),
            "tube_valid": bool(status.get("tube_valid", False)),
            "manifold_valid": bool(status.get("manifold_valid", False)),
            "accepted": bool(accepted),
            "failure_reason": reason,
            "trajectory_valid": bool(status["valid"]),
        })
        if accepted:
            best = proposal.copy()
            best_status = dict(status)
            best_cost = float(cost)
    fallback = False
    if not best_status["valid"]:
        best = original.copy()
        best_status = dict(pre_status)
        fallback = True
    return best, {
        "trace": trace,
        "pre_status": pre_status,
        "post_status": best_status,
        "fallback": bool(fallback),
    }


def _protected_indices_for_corridor(corridor, path):
    pts = np.asarray(path, float)
    protected = [0, len(pts) - 1]
    ordered = np.asarray(getattr(corridor, "topology_ordered_waypoints", []), float)
    if len(ordered) >= 3:
        for waypoint in ordered[1:-1]:
            if len(pts) == 0:
                continue
            idx = int(np.argmin(np.linalg.norm(
                pts - np.asarray(waypoint, float), axis=1)))
            protected.append(idx)
        return sorted(set(i for i in protected if 0 <= i < len(pts)))
    for waypoint in list(getattr(corridor, "channel_waypoints", [])):
        if len(pts) == 0:
            continue
        idx = int(np.argmin(np.linalg.norm(pts - np.asarray(waypoint, float), axis=1)))
        protected.append(idx)
    for waypoint in list(getattr(corridor, "task_minima_waypoints", [])):
        if len(pts) == 0:
            continue
        idx = int(np.argmin(np.linalg.norm(pts - np.asarray(waypoint, float), axis=1)))
        protected.append(idx)
    return sorted(set(i for i in protected if 0 <= i < len(pts)))


def _topology_order_preserved(corridor, path):
    pts = np.asarray(path, float)
    ordered = np.asarray(getattr(corridor, "topology_ordered_waypoints", []), float)
    if len(ordered) < 3 or len(pts) == 0:
        return True
    indices = [
        int(np.argmin(np.linalg.norm(pts - waypoint, axis=1)))
        for waypoint in ordered[1:-1]
    ]
    return indices == sorted(indices)


def _project_to_corridor(path, corridor, margin=0.85):
    pts = np.asarray(path, float).copy()
    radius = float(getattr(corridor, "radius", 0.0)) * float(margin)
    if radius <= 0.0:
        return pts
    for i, p in enumerate(pts):
        q, d = corridor.project(p)
        if d > radius:
            dim = min(len(q), pts.shape[1])
            pull = q[:dim] - pts[i, :dim]
            pts[i, :dim] += pull * (1.0 - radius / max(float(d), 1e-9))
    return pts


def smooth_wheelchair_corners(path, samples_per_segment=8, passes=2):
    """Densify and smooth sharp wheelchair corners while keeping endpoints fixed."""
    pts = _as_points(path)
    if len(pts) <= 2:
        return pts.copy()
    samples_per_segment = max(2, int(samples_per_segment))
    passes = max(1, int(passes))
    working = pts.copy()
    for _ in range(passes):
        dense = bezier_smooth_polyline(
            working, samples_per_segment=samples_per_segment)
        if len(dense) <= 2:
            break
        smoothed = dense.copy()
        for idx in range(1, len(dense) - 1):
            smoothed[idx] = (
                0.25 * dense[idx - 1] +
                0.50 * dense[idx] +
                0.25 * dense[idx + 1])
        smoothed[0] = pts[0]
        smoothed[-1] = pts[-1]
        working = smoothed
    return np.asarray(working, float)


def refine_topology_path(corridor, samples_per_segment=12,
                         max_curvature=None, max_turn=None,
                         footprint_checker=None, corridor_constraint=None,
                         manifold_constraint=None,
                         max_refinement_points=0):
    raw_original = np.asarray(getattr(corridor, "waypoints", []), float)
    protected_waypoints = []
    for attr in (
            "topology_ordered_waypoints", "channel_waypoints",
            "task_minima_waypoints"):
        values = getattr(corridor, attr, None)
        if values is None:
            continue
        values = _as_points(values)
        if len(values):
            protected_waypoints.extend(list(values))
    original, refinement_points_limited = _limit_refinement_points(
        raw_original, max_refinement_points,
        protected_points=protected_waypoints)
    risk_fn = getattr(corridor, "risk_field", None)
    if risk_fn is None:
        risk_fn = getattr(corridor, "field", None)
    corridor_constraint = _corridor_payload(
        corridor_constraint, corridor=corridor)
    manifold_constraint = _manifold_payload(
        manifold_constraint, corridor=corridor, risk_field=risk_fn)
    pre_status = check_refinement_manifold_validity(
        original, manifold_constraint=manifold_constraint,
        corridor_constraint=corridor_constraint, risk_field=risk_fn)
    wheelchair_fast_refinement = (
        max_refinement_points and
        (footprint_checker is not None or
         str(getattr(corridor, "robot_type", "")).lower() == "wheelchair" or
         str(getattr(corridor, "profile", "")).lower() == "wheelchair"))
    if len(original) <= 2 or wheelchair_fast_refinement:
        refined = original.copy()
        refinement_info = {
            "trace": [{
                "iteration": (
                    "bounded_wheelchair_refinement"
                    if wheelchair_fast_refinement else 0),
                "cost": float(path_length(refined)),
                "min_clearance": float(pre_status["min_clearance"]),
                "risk": float(pre_status["max_risk"]),
                "trajectory_valid": bool(pre_status["valid"]),
                "refinement_points_limited": bool(refinement_points_limited),
                "raw_reference_path_count": int(len(raw_original)),
                "bounded_reference_path_count": int(len(original)),
                "generic_smoothing_skipped": bool(wheelchair_fast_refinement),
            }],
            "pre_status": pre_status,
            "post_status": pre_status,
            "fallback": False,
        }
    else:
        refined, refinement_info = refine_trajectory(
            original,
            corridor_constraint=corridor_constraint,
            manifold_constraint=manifold_constraint,
            risk_field=risk_fn,
            samples_per_segment=samples_per_segment)
        if refinement_points_limited:
            refinement_info.setdefault("trace", []).append({
                "iteration": "bounded_refinement_points",
                "accepted": True,
                "failure_reason": "",
                "trajectory_valid": True,
                "raw_reference_path_count": int(len(raw_original)),
                "bounded_reference_path_count": int(len(original)),
            })
        projected = _project_to_corridor(refined, corridor)
        projected_status = check_refinement_manifold_validity(
            projected, manifold_constraint=manifold_constraint,
            corridor_constraint=corridor_constraint, risk_field=risk_fn)
        projected_reason = _safety_failure_reason(
            projected_status,
            pre_clearance=float(pre_status.get("min_clearance", 0.0)),
            tolerance=0.005)
        refinement_info.setdefault("trace", []).append({
            "iteration": "project_to_corridor",
            "trajectory_cost": float(_safety_cost(
                projected, projected_status, manifold_constraint)),
            "cost": float(_safety_cost(
                projected, projected_status, manifold_constraint)),
            "clearance": float(projected_status.get("min_clearance", 0.0)),
            "min_clearance": float(projected_status.get("min_clearance", 0.0)),
            "risk": float(projected_status.get("max_risk", 0.0)),
            "tube_valid": bool(projected_status.get("tube_valid", False)),
            "manifold_valid": bool(projected_status.get("manifold_valid", False)),
            "accepted": bool(not projected_reason and projected_status["valid"]),
            "failure_reason": projected_reason,
            "trajectory_valid": bool(projected_status["valid"]),
        })
        if not projected_reason and projected_status["valid"]:
            refined = projected
            refinement_info["post_status"] = dict(projected_status)
        else:
            refinement_info["fallback"] = True
            refinement_info["fallback_reason"] = projected_reason or "projection_invalid"
            refinement_info["post_status"] = dict(pre_status)

    protected = _protected_indices_for_corridor(corridor, refined)
    for idx in protected:
        if idx == 0:
            refined[idx] = original[0]
        elif idx == len(refined) - 1:
            refined[idx] = original[-1]
    protected_status = check_refinement_manifold_validity(
        refined, manifold_constraint=manifold_constraint,
        corridor_constraint=corridor_constraint, risk_field=risk_fn)
    protected_reason = _safety_failure_reason(
        protected_status,
        pre_clearance=float(pre_status.get("min_clearance", 0.0)),
        tolerance=0.005)
    refinement_info.setdefault("trace", []).append({
        "iteration": "protected_waypoints",
        "trajectory_cost": float(_safety_cost(
            refined, protected_status, manifold_constraint)),
        "cost": float(_safety_cost(
            refined, protected_status, manifold_constraint)),
        "clearance": float(protected_status.get("min_clearance", 0.0)),
        "min_clearance": float(protected_status.get("min_clearance", 0.0)),
        "risk": float(protected_status.get("max_risk", 0.0)),
        "tube_valid": bool(protected_status.get("tube_valid", False)),
        "manifold_valid": bool(protected_status.get("manifold_valid", False)),
        "accepted": bool(not protected_reason and protected_status["valid"]),
        "failure_reason": protected_reason,
        "trajectory_valid": bool(protected_status["valid"]),
    })
    if protected_reason or not protected_status["valid"]:
        refined = original.copy()
        refinement_info["fallback"] = True
        refinement_info["fallback_reason"] = (
            protected_reason or "protected_waypoint_invalid")
        refinement_info["post_status"] = dict(pre_status)
    else:
        refinement_info["post_status"] = dict(protected_status)

    metrics = path_curvature_metrics(refined)
    turn_limit = None if max_turn is None else 1.15 * float(max_turn)
    curvature_limit = None if max_curvature is None else 1.15 * float(max_curvature)
    if (str(getattr(corridor, "robot_type", "")).lower() == "wheelchair" or
            str(getattr(corridor, "profile", "")).lower() == "wheelchair"):
        target_turn = turn_limit if turn_limit is not None else 0.70
        if float(metrics.get("max_turn", 0.0)) > float(target_turn) + 1e-9:
            corner_refined = smooth_wheelchair_corners(
                refined,
                samples_per_segment=max(4, int(samples_per_segment)),
                passes=2)
            corner_refined = _project_to_corridor(corner_refined, corridor)
            corner_status = check_refinement_manifold_validity(
                corner_refined, manifold_constraint=manifold_constraint,
                corridor_constraint=corridor_constraint, risk_field=risk_fn)
            corner_reason = _safety_failure_reason(
                corner_status,
                pre_clearance=float(pre_status.get("min_clearance", 0.0)),
                tolerance=0.005)
            corner_metrics = path_curvature_metrics(corner_refined)
            refinement_info.setdefault("trace", []).append({
                "iteration": "wheelchair_corner_smoothing",
                "trajectory_cost": float(_safety_cost(
                    corner_refined, corner_status, manifold_constraint)),
                "cost": float(_safety_cost(
                    corner_refined, corner_status, manifold_constraint)),
                "clearance": float(corner_status.get("min_clearance", 0.0)),
                "min_clearance": float(corner_status.get("min_clearance", 0.0)),
                "risk": float(corner_status.get("max_risk", 0.0)),
                "tube_valid": bool(corner_status.get("tube_valid", False)),
                "manifold_valid": bool(corner_status.get("manifold_valid", False)),
                "accepted": bool(
                    not corner_reason and corner_status.get("valid", False) and
                    float(corner_metrics.get("max_turn", 0.0)) <=
                    float(target_turn) + 1e-9),
                "failure_reason": str(corner_reason),
                "trajectory_valid": bool(corner_status.get("valid", False)),
                "max_turn": float(corner_metrics.get("max_turn", 0.0)),
                "max_curvature": float(corner_metrics.get("max_curvature", 0.0)),
            })
            if (not corner_reason and corner_status.get("valid", False) and
                    float(corner_metrics.get("max_turn", 0.0)) <
                    float(metrics.get("max_turn", 0.0))):
                refined = corner_refined
                post_status = dict(corner_status)
                refinement_info["post_status"] = dict(corner_status)
                metrics = dict(corner_metrics)
    ordered = np.asarray(getattr(corridor, "topology_ordered_waypoints", []), float)
    if len(ordered) >= 3 and len(refined):
        errors = [
            float(np.min(np.linalg.norm(refined - waypoint, axis=1)))
            for waypoint in ordered[1:-1]
        ]
        topology_tracking_error = float(np.max(errors)) if errors else 0.0
    else:
        topology_tracking_error = 0.0
    risk_before = 0.0
    risk_after = 0.0
    if callable(risk_fn):
        try:
            risk_before = float(np.mean([risk_fn(p) for p in original])) if len(original) else 0.0
            risk_after = float(np.mean([risk_fn(p) for p in refined])) if len(refined) else 0.0
        except Exception:
            risk_before = 0.0
            risk_after = 0.0
    ok = True
    reason = ""
    if turn_limit is not None and metrics["max_turn"] > turn_limit + 1e-9:
        ok = False
        reason = "refined_turn_limit"
    if ok and curvature_limit is not None and metrics["max_curvature"] > curvature_limit + 1e-9:
        ok = False
        reason = "refined_curvature_limit"
    if ok and footprint_checker is not None:
        ok, reason = footprint_checker(refined)
    if ok and not _topology_order_preserved(corridor, refined):
        ok = False
        reason = "refined_topology_order_changed"
    post_status = dict(refinement_info.get("post_status", {}))
    if ok and not bool(post_status.get("valid", True)):
        ok = False
        reason = "refined_manifold_violation"
    if not ok or bool(refinement_info.get("fallback", False)):
        refined = original.copy()
        fallback_status = check_refinement_manifold_validity(
            refined, manifold_constraint=manifold_constraint,
            corridor_constraint=corridor_constraint, risk_field=risk_fn)
        post_status = dict(fallback_status)
        refinement_info["fallback"] = True
        refinement_info["post_status"] = dict(fallback_status)
        if fallback_status.get("valid", False):
            ok = True
            reason = ""
        else:
            reason = str(refinement_info.get(
                "fallback_reason", reason or "manifold_violation"))
    fallback_used = bool(refinement_info.get("fallback", False))
    fallback_reason = str(refinement_info.get("fallback_reason", reason or ""))
    if fallback_used and ok:
        reason = ""
    metrics = path_curvature_metrics(refined)

    corridor.raw_topology_waypoints = original.copy()
    corridor.refined_waypoints = refined
    corridor.refinement_used = int(ok and not fallback_used)
    corridor.refinement_success = bool(ok and not fallback_used)
    refinement_changed_path = bool(
        corridor.refinement_success and
        original.shape == refined.shape and
        original.size > 0 and
        np.max(np.abs(original - refined)) > 1e-9)
    if corridor.refinement_success and original.shape != refined.shape:
        refinement_changed_path = True
    reference_source = (
        "refined_waypoints" if refinement_changed_path
        else "candidate_fallback" if fallback_used
        else "selected_candidate_waypoints")
    corridor.refinement_reject_reason = "" if ok else str(reason)
    corridor.refined_path_length = path_length(refined)
    corridor.refined_max_turn_angle = metrics["max_turn"]
    corridor.refined_mean_turn_angle = metrics["mean_turn"]
    corridor.refined_max_curvature = metrics["max_curvature"]
    corridor.topology_tracking_error = float(topology_tracking_error)
    corridor.refinement_manifold_checked = True
    corridor.refinement_manifold_valid = bool(post_status.get("valid", False))
    corridor.refinement_tube_valid = bool(post_status.get(
        "refinement_tube_valid",
        post_status.get("corridor_violation_count", 0) == 0))
    corridor.post_refinement_corridor_valid = bool(corridor.refinement_tube_valid)
    corridor.pre_refinement_clearance = float(pre_status.get("min_clearance", 0.0))
    corridor.post_refinement_clearance = float(post_status.get("min_clearance", 0.0))
    corridor.trajectory_manifold_violation_count = int(
        post_status.get("violation_count", 0))
    corridor.trajectory_corridor_violation_count = int(
        post_status.get("corridor_violation_count", 0))
    corridor.refinement_fallback = bool(fallback_used)
    corridor.refinement_trace = list(refinement_info.get("trace", []))
    association = associate_corridor_critical_points(corridor, refined)
    corridor.critical_point_association = association
    corridor.critical_point_projection_index = {
        str(item.get("id", "")): int(item.get("trajectory_index", -1))
        for item in association.get("critical_points", [])
    }
    stage_sequence = [
        {
            "id": str(item.get("id", "")),
            "type": str(item.get("type", "")),
            "trajectory_index": int(item.get("trajectory_index", -1)),
            "stage_order": int(item.get("stage_order", item.get("order", idx + 1))),
            "critical_point_status": str(item.get("critical_point_status", "")),
        }
        for idx, item in enumerate(association.get("critical_points", []))
    ]
    corridor.topology_stage_sequence = stage_sequence
    corridor.refinement_output = {
        "trajectory": refined.tolist(),
        "final_trajectory": refined.tolist(),
        "critical_point_association": association,
        "topology_stage_sequence": stage_sequence,
        "refinement_manifold_checked": True,
        "refinement_manifold_valid": bool(corridor.refinement_manifold_valid),
        "refinement_tube_valid": bool(corridor.refinement_tube_valid),
        "pre_refinement_clearance": float(corridor.pre_refinement_clearance),
        "post_refinement_clearance": float(corridor.post_refinement_clearance),
        "post_refinement_corridor_valid": bool(
            corridor.post_refinement_corridor_valid),
        "refinement_fallback": bool(corridor.refinement_fallback),
        "refinement_success": bool(corridor.refinement_success),
        "fallback_used": bool(corridor.refinement_fallback),
        "fallback_reason": str(fallback_reason),
        "refinement_changed_path": bool(refinement_changed_path),
        "reference_source": reference_source,
        "trajectory_manifold_violation_count": int(
            corridor.trajectory_manifold_violation_count),
        "trajectory_corridor_violation_count": int(
            corridor.trajectory_corridor_violation_count),
    }
    corridor.risk_before_refinement = float(risk_before)
    corridor.risk_after_refinement = float(risk_after)
    corridor.final_reference_source = reference_source
    metrics["success"] = bool(ok)
    metrics["failure_reason"] = "" if ok else str(reason)
    metrics["fallback_reason"] = str(fallback_reason)
    metrics["refined_path_length"] = float(corridor.refined_path_length)
    metrics["topology_tracking_error"] = float(topology_tracking_error)
    metrics["critical_point_association_used"] = bool(
        association.get("critical_point_association_used", False))
    metrics["critical_point_status"] = association.get(
        "critical_point_status", "")
    metrics["topology_sequence_valid"] = bool(
        association.get("topology_sequence_valid", False))
    metrics["topology_stage_sequence"] = stage_sequence
    metrics["risk_before_refinement"] = float(risk_before)
    metrics["risk_after_refinement"] = float(risk_after)
    metrics["refinement_manifold_checked"] = True
    metrics["refinement_manifold_valid"] = bool(corridor.refinement_manifold_valid)
    metrics["refinement_tube_valid"] = bool(corridor.refinement_tube_valid)
    metrics["pre_refinement_clearance"] = float(corridor.pre_refinement_clearance)
    metrics["post_refinement_clearance"] = float(corridor.post_refinement_clearance)
    metrics["post_refinement_corridor_valid"] = bool(
        corridor.post_refinement_corridor_valid)
    metrics["refinement_fallback"] = bool(corridor.refinement_fallback)
    metrics["refinement_success"] = bool(corridor.refinement_success)
    metrics["refinement_changed_path"] = bool(refinement_changed_path)
    metrics["fallback_used"] = bool(corridor.refinement_fallback)
    metrics["reference_source"] = str(corridor.final_reference_source)
    metrics["reference_path_count"] = int(len(refined))
    metrics["raw_reference_path_count"] = int(len(raw_original))
    metrics["bounded_reference_path_count"] = int(len(original))
    metrics["refinement_points_limited"] = bool(refinement_points_limited)
    metrics["trajectory_manifold_violation_count"] = int(
        corridor.trajectory_manifold_violation_count)
    metrics["trajectory_corridor_violation_count"] = int(
        corridor.trajectory_corridor_violation_count)
    metrics["refinement_trace"] = list(corridor.refinement_trace)
    return ok, refined, metrics, reason
