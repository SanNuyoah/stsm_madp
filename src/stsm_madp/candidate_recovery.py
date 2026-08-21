import sys
sys.dont_write_bytecode = True

import copy
import itertools

import numpy as np

from stsm_madp.safety_evaluator import SafetyEvaluator
from stsm_madp.topology_candidate_generator import (
    evaluate_candidate_manifold_feasibility,
    generate_manifold_aware_corridor,
)
from stsm_madp.topology_ik_solver import TopologyIKSolver


def _finite_float(value, default=0.0):
    try:
        value = float(value)
    except Exception:
        return default
    if not bool(np.isfinite(value)):
        return default
    return value


def _as_points(points):
    if points is None or isinstance(points, str):
        return np.zeros((0, 3), float)
    try:
        arr = np.asarray(points, float)
    except Exception:
        return np.zeros((0, 3), float)
    if arr.size == 0:
        return np.zeros((0, 3), float)
    if arr.ndim == 1:
        arr = arr.reshape((1, arr.shape[0]))
    if arr.shape[1] == 2:
        arr = np.hstack([arr, np.zeros((arr.shape[0], 1), float)])
    return arr[:, :3]


def _grid_sample(grid, point):
    if not isinstance(grid, dict):
        return {}
    xs = grid.get("xs")
    ys = grid.get("ys")
    if xs is None or ys is None:
        return {}
    try:
        xs = np.asarray(xs, float)
        ys = np.asarray(ys, float)
        i = int(np.argmin(np.abs(xs - float(point[0]))))
        j = int(np.argmin(np.abs(ys - float(point[1]))))
    except Exception:
        return {}

    def sample(name, default=0.0):
        value = grid.get(name)
        if value is None:
            return default
        try:
            arr = np.asarray(value)
            return _finite_float(arr[i, j], default)
        except Exception:
            return default

    return {
        "clearance": sample("clearance", 0.0),
        "phi": sample("phi", 0.0),
        "rho": sample("rho", 1.0),
        "forbidden": bool(sample("forbidden", 0.0)),
    }


def _risk_value(risk_field, point, sample):
    if risk_field is not None:
        try:
            return _finite_float(risk_field.phi_s(point), sample.get("phi", 0.0))
        except Exception:
            pass
    return _finite_float(sample.get("phi", 0.0), 0.0)


def _normal(points, idx):
    pts = _as_points(points)
    if len(pts) <= 1:
        tangent = np.array([1.0, 0.0], float)
    elif idx == 0:
        tangent = pts[1, :2] - pts[0, :2]
    elif idx == len(pts) - 1:
        tangent = pts[-1, :2] - pts[-2, :2]
    else:
        tangent = pts[idx + 1, :2] - pts[idx - 1, :2]
    norm = float(np.linalg.norm(tangent))
    if norm <= 1e-9:
        tangent = np.array([1.0, 0.0], float)
    else:
        tangent = tangent / norm
    return np.array([-tangent[1], tangent[0]], float)


def _path_length(points):
    pts = _as_points(points)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)))


def _smoothness(points):
    pts = _as_points(points)
    if len(pts) < 3:
        return 0.0
    total = 0.0
    for idx in range(1, len(pts) - 1):
        a = pts[idx, :2] - pts[idx - 1, :2]
        b = pts[idx + 1, :2] - pts[idx, :2]
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na <= 1e-9 or nb <= 1e-9:
            continue
        dot = float(np.clip(np.dot(a / na, b / nb), -1.0, 1.0))
        total += abs(float(np.arccos(dot)))
    return float(total)


def _trajectory_score(points, safe_manifold, risk_field):
    pts = _as_points(points)
    if len(pts) == 0:
        return -float("inf")
    clearances = []
    risks = []
    forbidden = 0
    for point in pts:
        sample = _grid_sample(safe_manifold, point)
        clearances.append(_finite_float(sample.get("clearance", 0.0), 0.0))
        risks.append(_risk_value(risk_field, point, sample))
        forbidden += 1 if sample.get("forbidden", False) else 0
    min_clearance = min(clearances) if clearances else 0.0
    max_risk = max(risks) if risks else 0.0
    return float(
        100.0 * min_clearance -
        4.0 * max_risk -
        2.0 * _smoothness(pts) -
        0.5 * _path_length(pts) -
        1000.0 * forbidden)


def _optimize_centerline_once(points, safe_manifold, risk_field,
                              max_offset, step):
    pts = _as_points(points).copy()
    if len(pts) < 3:
        return pts
    original = pts.copy()
    for idx in range(1, len(pts) - 1):
        n = _normal(pts, idx)
        best = pts[idx].copy()
        best_score = _trajectory_score(pts, safe_manifold, risk_field)
        for scale in (-1.0, -0.5, 0.5, 1.0):
            cand_point = pts[idx].copy()
            cand_point[:2] = cand_point[:2] + float(scale) * float(step) * n
            if float(np.linalg.norm(cand_point[:2] - original[idx, :2])) > max_offset:
                continue
            cand_pts = pts.copy()
            cand_pts[idx] = cand_point
            score = _trajectory_score(cand_pts, safe_manifold, risk_field)
            if score > best_score:
                best = cand_point
                best_score = score
        pts[idx] = best
    return pts


def _adaptive_boundary(centerline, safe_manifold, risk_field, required,
                       iteration):
    base_width = float(required + 0.03 * max(0, int(iteration)))
    boundary = generate_manifold_aware_corridor(
        centerline,
        safe_manifold=safe_manifold,
        risk_field=risk_field,
        default_radius=max(base_width, 0.20),
        alpha=1.25,
        width_min=base_width,
        width_max=max(base_width, base_width + 0.25))
    widths = list(boundary.get("width", []) or [])
    if widths:
        boundary["width"] = [max(base_width, float(w)) for w in widths]
        boundary["corridor_width_profile"] = list(boundary["width"])
    return boundary


def recover_candidate_feasibility(candidates, manifold_constraint,
                                  risk_field=None, max_iterations=3):
    """Recover filtered candidate geometry without changing topology sequence."""
    payload = dict(manifold_constraint or {})
    if risk_field is None:
        risk_field = payload.get("risk_field")
    safe_manifold = payload.get("safe_manifold", payload.get("grid", {}))
    minimum = _finite_float(
        payload.get("minimum_clearance", payload.get("min_clearance", 0.0)),
        0.0)
    margin = _finite_float(payload.get("planning_clearance_margin", 0.0), 0.0)
    required = float(minimum + max(0.0, margin))
    recovered = []
    diagnostics = []
    for candidate in list(candidates or []):
        route = dict(candidate or {})
        centerline = _as_points(
            route.get("centerline") or route.get("waypoints") or
            route.get("refined_waypoints") or [])
        if len(centerline) == 0:
            continue
        before = evaluate_candidate_manifold_feasibility(
            route, payload, risk_field=risk_field)
        before_clearance = _finite_float(
            before.get("trajectory_min_clearance", before.get("min_clearance", 0.0)),
            0.0)
        best_route = dict(route)
        best_status = dict(before)
        working = centerline.copy()
        success = bool(before.get("feasible", False))
        iterations_used = 0
        for iteration in range(1, int(max_iterations) + 1):
            iterations_used = iteration
            working = _optimize_centerline_once(
                working, safe_manifold, risk_field,
                max_offset=0.10 * iteration,
                step=0.05 * iteration)
            boundary = _adaptive_boundary(
                working, safe_manifold, risk_field, required, iteration)
            trial = dict(route)
            trial["centerline"] = working.tolist()
            trial["waypoints"] = working.tolist()
            trial["boundary"] = boundary
            trial["corridor_width_profile"] = list(
                boundary.get("corridor_width_profile", boundary.get("width", [])))
            trial["adaptive_corridor_width"] = True
            trial["candidate_recovered"] = True
            trial["candidate_recovery_mode"] = "clearance_geometry_optimization"
            trial_payload = dict(payload)
            trial_payload["boundary"] = boundary
            status = evaluate_candidate_manifold_feasibility(
                trial, trial_payload, risk_field=risk_field)
            clearance = _finite_float(
                status.get("trajectory_min_clearance", status.get("min_clearance", 0.0)),
                0.0)
            if clearance >= _finite_float(
                    best_status.get("trajectory_min_clearance",
                                    best_status.get("min_clearance", 0.0)), 0.0):
                best_route = trial
                best_status = dict(status)
            if bool(status.get("feasible", False)):
                success = True
                break
        after_clearance = _finite_float(
            best_status.get("trajectory_min_clearance",
                            best_status.get("min_clearance", 0.0)), 0.0)
        best_route["candidate_recovered"] = True
        best_route["route_source"] = str(route.get(
            "route_source", "morse_topology"))
        best_route["candidate_source"] = "morse_recovered"
        best_route["topology_source"] = str(route.get(
            "topology_source", "morse_graph"))
        best_route["before_clearance"] = float(before_clearance)
        best_route["after_clearance"] = float(after_clearance)
        best_route["recovery_success"] = bool(success)
        best_route["candidate_recovery_iterations"] = int(iterations_used)
        best_route["adaptive_corridor_width"] = True
        best_route["clearance_optimization_used"] = True
        best_route["manifold_feasibility"] = dict(best_status)
        best_route["manifold_feasible"] = bool(best_status.get("feasible", False))
        best_route["candidate_tube_valid"] = bool(best_status.get(
            "candidate_tube_valid", best_status.get("tube_valid", False)))
        best_route["tube_valid"] = bool(best_route["candidate_tube_valid"])
        best_route["candidate_status"] = (
            "feasible" if success else "recoverable")
        best_route["failure_reason"] = "" if success else str(
            best_status.get("failure_reason", "clearance_violation"))
        best_route["min_clearance"] = float(after_clearance)
        best_route["trajectory_min_clearance"] = float(after_clearance)
        best_route["trajectory_max_risk"] = float(best_status.get(
            "trajectory_max_risk", best_status.get("max_risk", 0.0)))
        best_route["max_risk"] = float(best_status.get(
            "max_risk", best_route["trajectory_max_risk"]))
        diagnostics.append({
            "candidate_id": str(route.get("candidate_id", "")),
            "candidate_recovered": True,
            "before_clearance": float(before_clearance),
            "after_clearance": float(after_clearance),
            "recovery_success": bool(success),
            "candidate_recovery_iterations": int(iterations_used),
            "adaptive_corridor_width": True,
            "critical_point_sequence": list(route.get(
                "critical_point_sequence", [])),
            "node_sequence": list(route.get("node_sequence", [])),
        })
        if success:
            recovered.append(best_route)
    return recovered, {
        "candidate_feasibility_recovery_used": bool(diagnostics),
        "candidate_recovery_attempted": int(len(diagnostics)),
        "candidate_recovery_success_count": int(len(recovered)),
        "candidate_recovery_iterations": int(max(
            [d["candidate_recovery_iterations"] for d in diagnostics] or [0])),
        "recovered_candidates": diagnostics,
    }


def _finite_float(value, default=0.0):
    try:
        value = float(value)
    except Exception:
        return default
    if not bool(np.isfinite(value)):
        return default
    return value


def _as_points(points):
    if points is None or isinstance(points, str):
        return np.zeros((0, 3), float)
    try:
        arr = np.asarray(points, float)
    except Exception:
        return np.zeros((0, 3), float)
    if arr.size == 0:
        return np.zeros((0, 3), float)
    if arr.ndim == 1:
        arr = arr.reshape((1, arr.shape[0]))
    if arr.shape[1] == 2:
        arr = np.hstack([arr, np.zeros((arr.shape[0], 1), float)])
    return arr[:, :3]


def _candidate_value(candidate, key, default=None):
    if isinstance(candidate, dict):
        return candidate.get(key, default)
    return getattr(candidate, key, default)


def _path_normal(points, idx):
    pts = _as_points(points)
    if len(pts) <= 1:
        tangent = np.array([1.0, 0.0], float)
    elif idx == 0:
        tangent = pts[1, :2] - pts[0, :2]
    elif idx == len(pts) - 1:
        tangent = pts[-1, :2] - pts[-2, :2]
    else:
        tangent = pts[idx + 1, :2] - pts[idx - 1, :2]
    norm = float(np.linalg.norm(tangent))
    if norm <= 1e-9:
        tangent = np.array([1.0, 0.0], float)
    else:
        tangent = tangent / norm
    return np.array([-tangent[1], tangent[0]], float)


def _path_length(points):
    pts = _as_points(points)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)))


def _risk_value(risk_field, point, sample=None):
    if risk_field is None:
        return _finite_float((sample or {}).get("phi", 0.0), 0.0)
    try:
        return _finite_float(risk_field.phi_s(point),
                             _finite_float((sample or {}).get("phi", 0.0), 0.0))
    except Exception:
        return _finite_float((sample or {}).get("phi", 0.0), 0.0)


def _trajectory_risk(risk_field, points):
    pts = _as_points(points)
    if len(pts) == 0:
        return 0.0
    return float(max(_risk_value(risk_field, point) for point in pts))


def _trajectory_mean_risk(risk_field, points):
    pts = _as_points(points)
    if len(pts) > 2:
        pts = pts[1:-1]
    if len(pts) == 0:
        return _trajectory_risk(risk_field, points)
    return float(np.mean([_risk_value(risk_field, point) for point in pts]))


def _sample_polyline(points, spacing=0.025):
    pts = _as_points(points)
    if len(pts) < 2:
        return pts
    out = [pts[0].copy()]
    step = max(1e-3, float(spacing))
    for a, b in zip(pts[:-1], pts[1:]):
        delta = b - a
        length = float(np.linalg.norm(delta[:2]))
        count = max(1, int(np.ceil(length / step)))
        for idx in range(1, count + 1):
            out.append((a + (float(idx) / float(count)) * delta).copy())
    return np.asarray(out, float)


def _status_clearance(status):
    return _finite_float(status.get("min_clearance", 0.0), 0.0)


def _status_risk(status):
    return _finite_float(status.get("max_risk", 0.0), 0.0)


class ArmCandidateRecovery(object):
    """Arm-only task-space candidate recovery.

    This class does not change Morse route generation, candidate ranking, MPC, or
    SafetyEvaluator thresholds. It returns a recovered candidate copy plus
    diagnostics for callers that choose to consume it.
    """

    def __init__(self, max_offset=0.50, step_size=0.02, iterations=8,
                 width_margin=0.02, offset_scales=None):
        self.offset_scales = list(offset_scales or
                                  [0.10, 0.20, 0.35])
        self.max_offset = float(max(max_offset, max(self.offset_scales)))
        self.step_size = float(step_size)
        self.iterations = int(max(1, iterations))
        self.width_margin = float(max(0.0, width_margin))

    def recover(self, candidate, manifold_constraint, risk_field=None):
        route = copy.deepcopy(candidate if isinstance(candidate, dict) else {})
        centerline = np.zeros((0, 3), float)
        for key in ("centerline", "waypoints", "refined_waypoints", "path"):
            centerline = _as_points(_candidate_value(candidate, key, None))
            if len(centerline):
                break
        if len(centerline) == 0:
            diagnostics = self._empty_diagnostics(candidate)
            return route, diagnostics

        payload = dict(manifold_constraint or {})
        before = self._evaluate(route, centerline, payload, risk_field)
        recovered, after, search_record, search_failures = (
            self._regenerate_corridor(route, centerline, payload, risk_field))
        segment_diagnostics = list(search_record.get(
            "segment_diagnostics", []))
        global_recovery = dict(search_record.get("global_recovery", {}) or {})
        if not segment_diagnostics:
            for record in search_failures:
                if isinstance(record, dict) and record.get("segment_diagnostics"):
                    segment_diagnostics = list(record.get(
                        "segment_diagnostics", []))
                    break
        if not global_recovery:
            for record in search_failures:
                if isinstance(record, dict) and record.get("global_recovery"):
                    global_recovery = dict(record.get("global_recovery", {}))
                    break
        best_segment_clearance = max(
            [float(item.get("optimized_clearance", 0.0))
             for item in segment_diagnostics] or [0.0])
        risk_before = _trajectory_mean_risk(risk_field, centerline)
        risk_after = float(search_record.get(
            "candidate_risk", _trajectory_mean_risk(
                risk_field, recovered.get("centerline", []))))
        recovered["manifold_feasibility"] = dict(after)
        recovered["manifold_feasible"] = bool(after.get("valid", False))
        recovered["candidate_tube_valid"] = bool(after.get("tube_valid", False))
        recovered["tube_valid"] = bool(after.get("tube_valid", False))
        recovered["min_clearance"] = float(after.get("min_clearance", 0.0))
        recovered["trajectory_min_clearance"] = float(
            after.get("min_clearance", 0.0))
        recovered["trajectory_max_risk"] = float(after.get("max_risk", 0.0))
        recovered["max_risk"] = float(after.get("max_risk", 0.0))

        diagnostics = {
            "candidate_id": str(_candidate_value(candidate, "candidate_id", "")),
            "original_clearance": float(before.get("min_clearance", 0.0)),
            "optimized_clearance": float(after.get("min_clearance", 0.0)),
            "clearance_before": float(before.get("min_clearance", 0.0)),
            "clearance_after": float(after.get("min_clearance", 0.0)),
            "old_clearance": float(before.get("min_clearance", 0.0)),
            "new_clearance": float(after.get("min_clearance", 0.0)),
            "risk_before": float(risk_before),
            "risk_after": float(risk_after),
            "new_risk": float(risk_after),
            "tube_valid_before": bool(before.get("tube_valid", False)),
            "tube_valid_after": bool(after.get("tube_valid", False)),
            "old_tube_valid": bool(before.get("tube_valid", False)),
            "new_tube_valid": bool(after.get("tube_valid", False)),
            "offset_direction": str(search_record.get("offset_direction", "")),
            "offset_distance": float(search_record.get("offset_distance", 0.0)),
            "selected_offset": str(search_record.get("selected_offset", "")),
            "candidate_clearance": float(search_record.get(
                "candidate_clearance", after.get("min_clearance", 0.0))),
            "candidate_risk": float(search_record.get(
                "candidate_risk", after.get("max_risk", 0.0))),
            "candidate_length": float(search_record.get(
                "candidate_length", 0.0)),
            "tested_offset_count": int(search_record.get(
                "tested_offset_count", len(search_failures))),
            "best_clearance": float(search_record.get(
                "best_clearance",
                max(after.get("min_clearance", 0.0), best_segment_clearance))),
            "best_tube_valid": bool(search_record.get(
                "best_tube_valid", after.get("tube_valid", False))),
            "arm_tube_validation": dict(search_record.get(
                "arm_tube_validation",
                after.get("arm_tube_validation", {}))),
            "segment_diagnostics": list(segment_diagnostics),
            "best_segment_clearance": float(best_segment_clearance),
            "segment_candidate_count": int(sum(
                int(item.get("candidate_count", 0))
                for item in segment_diagnostics)),
            "global_recovery": dict(global_recovery),
            "new_centerline_count": int(search_record.get(
                "new_centerline_count", 0)),
            "new_boundary_count": int(search_record.get(
                "new_boundary_count", 0)),
            "search_failures": list(search_failures),
            "recovery_success": bool(self._success(after)),
        }
        recovered["arm_recovery_diagnostics"] = dict(diagnostics)
        return recovered, diagnostics

    def _empty_diagnostics(self, candidate):
        return {
            "candidate_id": str(_candidate_value(candidate, "candidate_id", "")),
            "original_clearance": 0.0,
            "optimized_clearance": 0.0,
            "clearance_before": 0.0,
            "clearance_after": 0.0,
            "risk_before": 0.0,
            "risk_after": 0.0,
            "tube_valid_before": False,
            "tube_valid_after": False,
            "old_tube_valid": False,
            "new_tube_valid": False,
            "old_clearance": 0.0,
            "new_clearance": 0.0,
            "offset_direction": "",
            "offset_distance": 0.0,
            "selected_offset": "",
            "candidate_clearance": 0.0,
            "candidate_risk": 0.0,
            "candidate_length": 0.0,
            "tested_offset_count": 0,
            "best_clearance": 0.0,
            "best_tube_valid": False,
            "arm_tube_validation": {},
            "segment_diagnostics": [],
            "global_recovery": {},
            "new_centerline_count": 0,
            "new_boundary_count": 0,
            "search_failures": ["empty_centerline"],
            "recovery_success": False,
        }

    def _evaluate(self, candidate, centerline, payload, risk_field):
        tube_width = _finite_float(
            _candidate_value(candidate, "tube_width", 0.0), 0.0)
        corridor_constraint = {
            "centerline": _as_points(centerline).tolist(),
            "radius": tube_width or payload.get(
                "corridor_radius", payload.get("radius", 0.0)),
        }
        safety_payload = self._safety_payload(payload)
        evaluator = SafetyEvaluator(
            manifold_constraint=safety_payload,
            corridor_constraint=corridor_constraint,
            risk_field=risk_field,
            planning_clearance_margin=safety_payload.get(
                "planning_clearance_margin", 0.0))
        eval_centerline = self._evaluation_samples(centerline)
        trajectory_status = evaluator.evaluate_trajectory(eval_centerline)
        corridor_candidate = copy.deepcopy(candidate)
        corridor_candidate["centerline"] = eval_centerline.tolist()
        corridor_candidate["waypoints"] = eval_centerline.tolist()
        generic_tube_status = evaluator.evaluate_corridor(corridor_candidate)
        arm_tube_status = self._validate_arm_tube(
            candidate, centerline, payload, risk_field)
        out = dict(trajectory_status)
        out.update(dict(generic_tube_status))
        out.update({
            "tube_valid": bool(arm_tube_status.get("tube_valid", False)),
            "min_tube_clearance": float(arm_tube_status.get(
                "min_tube_clearance", generic_tube_status.get(
                    "min_tube_clearance", 0.0))),
            "max_tube_risk": float(arm_tube_status.get(
                "max_tube_risk", generic_tube_status.get("max_tube_risk", 0.0))),
            "arm_tube_validation": dict(arm_tube_status),
        })
        out["min_clearance"] = min(
            _status_clearance(trajectory_status),
            float(arm_tube_status.get(
                "centerline_clearance", _status_clearance(trajectory_status))),
            float(arm_tube_status.get(
                "link_clearance", _status_clearance(trajectory_status))))
        out["max_risk"] = max(
            _status_risk(trajectory_status),
            float(arm_tube_status.get(
                "centerline_risk", _status_risk(trajectory_status))),
            float(arm_tube_status.get(
                "link_risk", _status_risk(trajectory_status))))
        out["valid"] = bool(
            trajectory_status.get("valid", False) and
            arm_tube_status.get("tube_valid", False))
        return out

    def _safety_payload(self, payload):
        out = dict(payload or {})
        boundary = out.get("boundary", None)
        if isinstance(boundary, dict) and (
                boundary.get("arm_generated_tube", False) or
                boundary.get("left") or boundary.get("right")):
            out.pop("boundary", None)
        return out

    def _validate_arm_tube(self, candidate, centerline, payload, risk_field):
        boundary = dict(_candidate_value(candidate, "boundary", {}) or {})
        tube_width = _finite_float(
            _candidate_value(candidate, "tube_width",
                             boundary.get("tube_width", 0.0)), 0.0)
        safety_payload = self._safety_payload(payload)
        evaluator = SafetyEvaluator(
            manifold_constraint=safety_payload,
            corridor_constraint={
                "centerline": _as_points(centerline).tolist(),
                "radius": tube_width,
            },
            risk_field=risk_field,
            planning_clearance_margin=safety_payload.get(
                "planning_clearance_margin", 0.0))
        center_samples = self._evaluation_samples(
            _sample_polyline(centerline, spacing=0.05))
        boundary_samples = self._boundary_samples(boundary)
        link_samples = self._link_samples(candidate, centerline)
        center_status = evaluator.evaluate_trajectory(center_samples)
        boundary_status = evaluator.evaluate_trajectory(boundary_samples)
        link_status = evaluator.evaluate_trajectory(link_samples)
        center_valid = bool(center_status.get("valid", False))
        boundary_valid = bool(
            len(boundary_samples) > 0 and boundary_status.get("valid", False))
        link_valid = bool(
            len(link_samples) > 0 and link_status.get("valid", False))
        tube_valid = bool(
            tube_width > 0.0 and center_valid and boundary_valid and link_valid)
        failure_reason = self._arm_tube_failure_reason(
            tube_width, center_status, boundary_status, link_status,
            center_valid, boundary_valid, link_valid)
        min_clearance = min(
            _status_clearance(center_status),
            _status_clearance(boundary_status),
            _status_clearance(link_status))
        max_risk = max(
            _status_risk(center_status),
            _status_risk(boundary_status),
            _status_risk(link_status))
        return {
            "candidate_id": str(_candidate_value(candidate, "candidate_id", "")),
            "centerline_clearance": float(_status_clearance(center_status)),
            "boundary_clearance": float(_status_clearance(boundary_status)),
            "link_clearance": float(_status_clearance(link_status)),
            "centerline_risk": float(_status_risk(center_status)),
            "boundary_risk": float(_status_risk(boundary_status)),
            "link_risk": float(_status_risk(link_status)),
            "tube_width": float(tube_width),
            "tube_valid": bool(tube_valid),
            "failure_reason": str(failure_reason),
            "centerline_sample_count": int(len(center_samples)),
            "boundary_sample_count": int(len(boundary_samples)),
            "link_sample_count": int(len(link_samples)),
            "min_tube_clearance": float(min_clearance),
            "max_tube_risk": float(max_risk),
        }

    def _boundary_samples(self, boundary):
        if not isinstance(boundary, dict):
            return _sample_polyline(boundary)
        samples = []
        for key in ("left", "right", "left_boundary", "right_boundary"):
            pts = self._evaluation_samples(
                _sample_polyline(boundary.get(key, []), spacing=0.05))
            if len(pts):
                samples.extend(pts.tolist())
        return _as_points(samples)

    def _link_samples(self, candidate, centerline):
        samples = []
        for key in ("link_points", "link_samples", "link_sample_points"):
            pts = _as_points(_candidate_value(candidate, key, []))
            if len(pts):
                samples.extend(pts.tolist())
        link_trajectory = _candidate_value(candidate, "link_trajectory", [])
        if isinstance(link_trajectory, dict):
            for value in link_trajectory.values():
                pts = _as_points(value)
                if len(pts):
                    samples.extend(pts.tolist())
        if samples:
            return _as_points(samples)
        pts = _sample_polyline(centerline, spacing=0.08)
        pts = self._evaluation_samples(pts)
        if len(pts) < 2:
            return pts
        out = []
        for idx, point in enumerate(pts):
            normal = _path_normal(pts, idx)
            tangent = np.array([normal[1], -normal[0]], float)
            for dist in (0.04, 0.08, 0.12):
                lp = point.copy()
                lp[:2] = lp[:2] - dist * tangent
                out.append(lp.tolist())
        return _as_points(out)

    def _evaluation_samples(self, points):
        pts = _as_points(points)
        if len(pts) <= 2:
            return pts
        return pts[1:-1].copy()

    def _arm_tube_failure_reason(self, tube_width, center_status,
                                 boundary_status, link_status,
                                 center_valid, boundary_valid, link_valid):
        reasons = []
        if tube_width <= 0.0:
            reasons.append("tube_width_invalid")
        for prefix, status, valid in (
                ("centerline", center_status, center_valid),
                ("boundary", boundary_status, boundary_valid),
                ("link", link_status, link_valid)):
            if valid:
                continue
            if int(status.get("manifold_violation_count", 0)):
                reasons.append("%s_clearance_or_risk_violation" % prefix)
            if int(status.get("corridor_violation_count", 0)):
                reasons.append("%s_outside_tube" % prefix)
        return ",".join(reasons)

    def _regenerate_corridor(self, route, centerline, payload, risk_field):
        best_route = None
        best_status = None
        best_record = {}
        search_failures = []
        tested_offset_count = 0
        segment_route, segment_status, segment_record = (
            self._regenerate_by_segments(route, centerline, payload, risk_field))
        if segment_route is not None:
            best_route = segment_route
            best_status = dict(segment_status)
            best_record = dict(segment_record)
            search_failures.append(dict(segment_record))
        for direction in ("left", "right", "adaptive"):
            for offset in self.offset_scales:
                offset = float(offset)
                tested_offset_count += 1
                adjusted = self._adjust_trajectory(
                    centerline, payload, risk_field, direction, offset)
                boundary = self._regenerate_boundary(adjusted, payload)
                trial = self._trial_route(route, adjusted, boundary)
                trial_payload = dict(payload)
                status = self._evaluate(
                    trial, adjusted, trial_payload, risk_field)
                status["candidate_clearance"] = float(
                    status.get("min_clearance", 0.0))
                status["candidate_risk"] = float(_trajectory_mean_risk(
                    risk_field, adjusted))
                status["candidate_length"] = float(_path_length(adjusted))
                record = {
                    "offset_direction": direction,
                    "offset_distance": float(offset),
                    "selected_offset": "%s:%.3f" % (direction, float(offset)),
                    "candidate_clearance": float(status["candidate_clearance"]),
                    "candidate_risk": float(status["candidate_risk"]),
                    "candidate_length": float(status["candidate_length"]),
                    "new_clearance": float(status.get("min_clearance", 0.0)),
                    "new_risk": float(status["candidate_risk"]),
                    "new_tube_valid": bool(status.get("tube_valid", False)),
                    "arm_tube_validation": dict(status.get(
                        "arm_tube_validation", {})),
                    "new_centerline_count": int(len(adjusted)),
                    "new_boundary_count": int(
                        len(boundary.get("left", [])) +
                        len(boundary.get("right", []))),
                    "failure_reason": self._failure_reason(status),
                }
                cost = self._candidate_cost(status, adjusted)
                if best_status is None or self._prefer_candidate(
                        status, adjusted, cost,
                        best_status,
                        _as_points(best_route.get("centerline", []))):
                    best_route = trial
                    best_status = dict(status)
                    best_record = dict(record)
                    best_record["tested_offset_count"] = int(
                        tested_offset_count)
                    best_record["best_clearance"] = float(
                        status.get("min_clearance", 0.0))
                    best_record["best_tube_valid"] = bool(
                        status.get("tube_valid", False))
                if self._success(status):
                    # Keep searching; among feasible corridors choose lowest risk.
                    search_failures.append(record)
                    continue
                search_failures.append(record)
        if best_route is None:
            empty = copy.deepcopy(route)
            status = self._evaluate(empty, centerline, payload, risk_field)
            return empty, status, {}, ["no_candidate_generated"]
        best_record["tested_offset_count"] = int(tested_offset_count)
        return best_route, best_status, best_record, search_failures

    def _regenerate_by_segments(self, route, centerline, payload, risk_field):
        segments = self._critical_segments(route, centerline)
        if not segments:
            return None, {}, {}
        ranked_segments = []
        segment_diagnostics = []
        total_candidates = 0
        for seg_id, segment in enumerate(segments):
            original_status = self._segment_candidate_status(
                segment, payload, risk_field)
            candidates = self._rank_segment_candidates(
                segment, payload, risk_field)
            total_candidates += int(len(candidates))
            ranked = candidates[:self._per_segment_global_limit(len(segments))]
            ranked_segments.append(ranked)
            best = ranked[0] if ranked else {
                "points": segment,
                "status": original_status,
                "score": self._segment_candidate_score(
                    original_status, segment),
            }
            segment_diagnostics.append({
                "segment_id": int(seg_id),
                "original_clearance": float(
                    original_status.get("end_effector_clearance", 0.0)),
                "optimized_clearance": float(
                    best["status"].get("end_effector_clearance", 0.0)),
                "link_proxy_clearance": float(
                    best["status"].get("link_proxy_clearance", 0.0)),
                "risk": float(best["status"].get("risk", 0.0)),
                "path_length": float(_path_length(best["points"])),
                "candidate_count": int(len(candidates)),
                "global_search_candidate_count": int(len(ranked)),
                "best_candidate_score": float(best["score"]),
                "tube_valid": bool(best["status"].get("tube_valid", False)),
                "failure_reason": str(best["status"].get(
                    "failure_reason", "")),
            })
        best_route = None
        best_status = None
        best_score = float("inf")
        best_choice_ids = []
        combinations_considered = 0
        combinations_tested = 0
        failed_segment_ids = []
        combo_records = []
        for combo in self._bounded_combinations(ranked_segments):
            combinations_considered += 1
            combo_records.append((
                self._combo_proxy_score(combo),
                combo,
            ))
        combo_records.sort(key=lambda item: item[0])
        for _, combo in combo_records[:self._global_eval_limit(len(segments))]:
            combinations_tested += 1
            adjusted = self._join_segment_combo(combo)
            boundary = self._regenerate_boundary(adjusted, payload)
            trial = self._trial_route(route, adjusted, boundary)
            trial["candidate_recovery_mode"] = "arm_global_regeneration"
            trial["segment_level_regeneration_used"] = True
            trial["global_trajectory_regeneration_used"] = True
            status = self._evaluate(trial, adjusted, dict(payload), risk_field)
            status["candidate_clearance"] = float(
                status.get("min_clearance", 0.0))
            status["candidate_risk"] = float(_trajectory_mean_risk(
                risk_field, adjusted))
            status["candidate_length"] = float(_path_length(adjusted))
            global_score = self._global_trajectory_score(status, adjusted, combo)
            if best_status is None or global_score < best_score:
                best_route = trial
                best_status = dict(status)
                best_score = float(global_score)
                best_choice_ids = [str(item.get("candidate_id", ""))
                                   for item in combo]
                failed_segment_ids = [
                    int(idx) for idx, item in enumerate(combo)
                    if not bool(item.get("status", {}).get("tube_valid", False))
                ]
        if best_route is None:
            return None, {}, {}
        best_segment_clearance = max(
            [float(item.get("optimized_clearance", 0.0))
             for item in segment_diagnostics] or [0.0])
        adjusted = _as_points(best_route.get("centerline", []))
        boundary = dict(best_route.get("boundary", {}) or {})
        status = dict(best_status)
        arm_tube = dict(status.get("arm_tube_validation", {}) or {})
        record = {
            "offset_direction": "global_regeneration",
            "offset_distance": 0.0,
            "selected_offset": "global_regeneration",
            "candidate_clearance": float(status["candidate_clearance"]),
            "candidate_risk": float(status["candidate_risk"]),
            "candidate_length": float(status["candidate_length"]),
            "new_clearance": float(status.get("min_clearance", 0.0)),
            "new_risk": float(status["candidate_risk"]),
            "new_tube_valid": bool(status.get("tube_valid", False)),
            "arm_tube_validation": dict(status.get("arm_tube_validation", {})),
            "new_centerline_count": int(len(adjusted)),
            "new_boundary_count": int(
                len(boundary.get("left", [])) +
                len(boundary.get("right", []))),
            "failure_reason": self._failure_reason(status),
            "segment_diagnostics": list(segment_diagnostics),
            "segment_candidate_count": int(total_candidates),
            "best_segment_clearance": float(best_segment_clearance),
            "best_clearance": float(max(
                status.get("min_clearance", 0.0), best_segment_clearance)),
            "best_tube_valid": bool(status.get("tube_valid", False)),
            "global_recovery": {
                "candidate_id": str(_candidate_value(route, "candidate_id", "")),
                "segment_total": int(len(segments)),
                "segment_candidate_counts": [
                    int(item.get("candidate_count", 0))
                    for item in segment_diagnostics
                ],
                "trajectory_combinations_considered": int(
                    combinations_considered),
                "trajectory_combinations_tested": int(combinations_tested),
                "best_trajectory_score": float(best_score),
                "best_choice_ids": list(best_choice_ids),
                "failed_segment_ids": list(failed_segment_ids),
                "global_clearance": float(status.get("min_clearance", 0.0)),
                "global_link_clearance": float(arm_tube.get(
                    "link_clearance", 0.0)),
                "global_risk": float(status.get("max_risk", 0.0)),
                "global_tube_valid": bool(status.get("tube_valid", False)),
                "failure_reason": self._failure_reason(status),
            },
        }
        return best_route, status, record

    def _rank_segment_candidates(self, segment, payload, risk_field):
        ranked = []
        raw_candidates = list(self._segment_candidates(
            segment, payload, risk_field))
        raw_candidates.sort(key=lambda points: self._segment_proxy_score(
            points, risk_field))
        for idx, points in enumerate(raw_candidates[:8]):
            status = self._segment_candidate_status(
                points, payload, risk_field)
            score = self._segment_candidate_score(status, points)
            ranked.append({
                "candidate_id": "seg_%03d" % int(idx),
                "points": _as_points(points),
                "status": dict(status),
                "score": float(score),
            })
        ranked.sort(key=lambda item: (
            0 if bool(item["status"].get("tube_valid", False)) else 1,
            float(item["score"])))
        return ranked

    def _segment_proxy_score(self, points, risk_field):
        pts = _sample_polyline(points, spacing=0.12)
        eval_pts = self._evaluation_samples(pts)
        if len(eval_pts) == 0:
            eval_pts = pts
        risk = _trajectory_risk(risk_field, eval_pts)
        link_pts = self._link_samples({}, pts)
        link_risk = _trajectory_risk(risk_field, link_pts)
        return float(
            220.0 * max(risk, link_risk) +
            6.0 * _path_length(points))

    def _per_segment_global_limit(self, segment_count):
        if segment_count <= 2:
            return 3
        if segment_count == 3:
            return 2
        return 2

    def _bounded_combinations(self, ranked_segments, max_combinations=16):
        produced = 0
        for combo in itertools.product(*ranked_segments):
            yield combo
            produced += 1
            if produced >= int(max_combinations):
                return

    def _global_eval_limit(self, segment_count):
        if segment_count <= 2:
            return 4
        if segment_count == 3:
            return 4
        return 3

    def _combo_proxy_score(self, combo):
        risk = 0.0
        clearance_deficit = 0.0
        link_deficit = 0.0
        length = 0.0
        invalid_segments = 0
        for item in combo:
            status = dict(item.get("status", {}) or {})
            required = _finite_float(status.get("required_clearance", 0.0), 0.0)
            ee_clearance = _finite_float(
                status.get("end_effector_clearance", 0.0), 0.0)
            link_clearance = _finite_float(
                status.get("link_proxy_clearance", 0.0), 0.0)
            risk = max(risk, _finite_float(status.get("risk", 0.0), 0.0))
            clearance_deficit += max(0.0, required - ee_clearance)
            link_deficit += max(0.0, required - link_clearance)
            length += _path_length(item.get("points", []))
            if not bool(status.get("tube_valid", False)):
                invalid_segments += 1
        return float(
            180.0 * risk +
            900.0 * clearance_deficit +
            700.0 * link_deficit +
            8.0 * length +
            150.0 * invalid_segments)

    def _join_segment_combo(self, combo):
        points = []
        for item in combo:
            pts = _as_points(item.get("points", []))
            if len(pts) == 0:
                continue
            if points:
                points.extend(pts[1:].tolist())
            else:
                points.extend(pts.tolist())
        return _as_points(points)

    def _global_trajectory_score(self, status, points, combo):
        risk = _finite_float(status.get(
            "candidate_risk", status.get("max_risk", 0.0)), 0.0)
        required = _finite_float(status.get("required_clearance", 0.0), 0.0)
        clearance = _finite_float(status.get("min_clearance", 0.0), 0.0)
        arm_tube = dict(status.get("arm_tube_validation", {}) or {})
        link_clearance = _finite_float(
            arm_tube.get("link_clearance", clearance), clearance)
        length = _path_length(points)
        invalid_segments = sum(
            1 for item in combo
            if not bool(item.get("status", {}).get("tube_valid", False)))
        tube_penalty = 0.0 if bool(status.get("tube_valid", False)) else 500.0
        valid_bonus = -5000.0 if self._success(status) else 0.0
        return float(
            valid_bonus +
            tube_penalty +
            180.0 * risk +
            900.0 * max(0.0, required - clearance) +
            700.0 * max(0.0, required - link_clearance) +
            8.0 * length +
            150.0 * invalid_segments)

    def _critical_segments(self, route, centerline):
        pts = _as_points(centerline)
        if len(pts) < 2:
            return []
        sequence = list(_candidate_value(route, "critical_point_sequence", []) or [])
        segment_count = max(1, len(sequence) - 1)
        if len(sequence) < 2:
            route_cells = list(_candidate_value(route, "cells", []) or [])
            if len(route_cells) >= 3:
                segment_count = min(3, len(pts) - 1)
            elif len(pts) >= 6:
                segment_count = min(2, len(pts) - 1)
        segment_count = min(segment_count, len(pts) - 1)
        indices = np.linspace(0, len(pts) - 1, segment_count + 1)
        indices = [int(round(v)) for v in indices]
        indices[0] = 0
        indices[-1] = len(pts) - 1
        segments = []
        for start, stop in zip(indices[:-1], indices[1:]):
            if stop <= start:
                continue
            segment = pts[start:stop + 1]
            if len(segment) >= 2:
                segments.append(segment.copy())
        return segments

    def _segment_candidates(self, segment, payload, risk_field):
        pts = _as_points(segment)
        if len(pts) < 2:
            return []
        candidates = [_sample_polyline(pts, spacing=0.06)]
        start = pts[0]
        goal = pts[-1]
        chord = goal[:2] - start[:2]
        norm = float(np.linalg.norm(chord))
        if norm <= 1e-9:
            normal = np.array([1.0, 0.0], float)
        else:
            tangent = chord / norm
            normal = np.array([-tangent[1], tangent[0]], float)
        midpoint = 0.5 * (start + goal)
        adaptive_sign = self._global_risk_descent_sign(
            pts, risk_field, fallback=1.0)
        segment_scales = sorted(set(
            [float(v) for v in self.offset_scales] + [0.70, 1.00]))
        z_offsets = [0.0, 0.12, 0.24]
        for direction in ("left", "right", "adaptive"):
            signs = (
                [-1.0] if direction == "left" else
                [1.0] if direction == "right" else
                [adaptive_sign, -adaptive_sign])
            for sign in signs:
                for offset in segment_scales:
                    for z_offset in z_offsets:
                        via = midpoint.copy()
                        via[:2] = via[:2] + sign * float(offset) * normal
                        via[2] = via[2] + float(z_offset)
                        candidates.append(_sample_polyline(
                            np.vstack([start, via, goal]), spacing=0.06))
                        via1 = start + (goal - start) / 3.0
                        via2 = start + 2.0 * (goal - start) / 3.0
                        via1 = via1.copy()
                        via2 = via2.copy()
                        via1[:2] = via1[:2] + sign * float(offset) * normal
                        via2[:2] = via2[:2] - sign * 0.5 * float(offset) * normal
                        via1[2] = via1[2] + float(z_offset)
                        via2[2] = via2[2] + float(z_offset)
                        candidates.append(_sample_polyline(
                            np.vstack([start, via1, via2, goal]), spacing=0.06))
        return candidates

    def _segment_candidate_status(self, points, payload, risk_field):
        pts = _as_points(points)
        if len(pts) > 2:
            ee_points = pts[1:-1]
        else:
            ee_points = pts
        safety_payload = self._safety_payload(payload)
        evaluator = SafetyEvaluator(
            manifold_constraint=safety_payload,
            corridor_constraint={"centerline": pts.tolist(), "radius": 0.50},
            risk_field=risk_field,
            planning_clearance_margin=safety_payload.get(
                "planning_clearance_margin", 0.0))
        ee_status = evaluator.evaluate_trajectory(ee_points)
        link_points = self._link_samples({}, pts)
        if len(link_points) > 2:
            link_points = link_points[1:-1]
        link_status = evaluator.evaluate_trajectory(link_points)
        required = _finite_float(ee_status.get("required_clearance", 0.0), 0.0)
        risk_threshold = _finite_float(ee_status.get("risk_threshold", 1.0), 1.0)
        ee_clearance = _status_clearance(ee_status)
        link_clearance = _status_clearance(link_status)
        risk = max(_status_risk(ee_status), _status_risk(link_status))
        tube_valid = bool(
            ee_clearance + 1e-9 >= required and
            link_clearance + 1e-9 >= required and
            risk <= risk_threshold + 1e-9)
        reasons = []
        if ee_clearance + 1e-9 < required:
            reasons.append("end_effector_clearance_violation")
        if link_clearance + 1e-9 < required:
            reasons.append("link_proxy_clearance_violation")
        if risk > risk_threshold + 1e-9:
            reasons.append("risk_violation")
        return {
            "end_effector_clearance": float(ee_clearance),
            "link_proxy_clearance": float(link_clearance),
            "risk": float(risk),
            "required_clearance": float(required),
            "risk_threshold": float(risk_threshold),
            "tube_valid": bool(tube_valid),
            "failure_reason": ",".join(reasons),
        }

    def _segment_candidate_score(self, status, points):
        required = _finite_float(status.get("required_clearance", 0.0), 0.0)
        ee_clearance = _finite_float(
            status.get("end_effector_clearance", 0.0), 0.0)
        link_clearance = _finite_float(
            status.get("link_proxy_clearance", 0.0), 0.0)
        clearance = min(ee_clearance, link_clearance)
        risk = _finite_float(status.get("risk", 0.0), 0.0)
        risk_threshold = max(
            1e-6, _finite_float(status.get("risk_threshold", 1.0), 1.0))
        valid_bonus = -1000.0 if bool(status.get("tube_valid", False)) else 0.0
        return float(
            valid_bonus +
            500.0 * max(0.0, required - clearance) +
            120.0 * (risk / risk_threshold) +
            4.0 * _path_length(points))

    def _trial_route(self, route, centerline, boundary):
        recovered = copy.deepcopy(route)
        pts = _as_points(centerline)
        widths = list(boundary.get("width", []) or [])
        tube_width = float(min(widths) if widths else 0.0)
        recovered["centerline"] = pts.tolist()
        recovered["waypoints"] = pts.tolist()
        recovered["boundary"] = boundary
        recovered["left_boundary"] = list(boundary.get("left", []))
        recovered["right_boundary"] = list(boundary.get("right", []))
        recovered["tube_width"] = float(tube_width)
        recovered["corridor_width_profile"] = list(
            boundary.get("corridor_width_profile", widths))
        recovered["critical_point_sequence"] = self._route_critical_sequence(
            route)
        recovered["route_source"] = str(
            route.get("route_source", "morse_topology"))
        recovered["candidate_recovered"] = True
        recovered["candidate_recovery_mode"] = "arm_corridor_regeneration"
        recovered["adaptive_corridor_width"] = True
        recovered["clearance_optimization_used"] = True
        return recovered

    def _route_critical_sequence(self, route):
        sequence = list(route.get("critical_point_sequence", []) or [])
        if sequence:
            return sequence
        for key in ("node_sequence", "topology_nodes", "morse_node_ids"):
            values = list(route.get(key, []) or [])
            if values:
                return [str(value) for value in values]
        return []

    def _adjust_trajectory(self, centerline, payload, risk_field, direction,
                           offset_distance):
        pts = _as_points(centerline).copy()
        if len(pts) < 3:
            return pts
        original = pts.copy()
        out = pts.copy()
        fixed_sign = -1.0 if direction == "left" else 1.0
        global_sign = self._global_risk_descent_sign(
            original, risk_field, fixed_sign)
        for idx in range(1, len(out) - 1):
            normal = _path_normal(original, idx)
            choices = []
            if direction in ("left", "right"):
                signs = [fixed_sign]
            else:
                signs = [global_sign, -global_sign, 0.0]
            for sign in signs:
                candidate_point = original[idx].copy()
                candidate_point[:2] = (
                    candidate_point[:2] + sign * float(offset_distance) * normal)
                choices.append((self._point_cost(
                    candidate_point, original[idx], payload, risk_field),
                    candidate_point))
            choices.sort(key=lambda item: item[0])
            out[idx] = choices[0][1]
        out[0] = original[0]
        out[-1] = original[-1]
        return out

    def _global_risk_descent_sign(self, points, risk_field, fallback):
        pts = _as_points(points)
        if len(pts) < 3:
            return float(fallback)
        left = pts.copy()
        right = pts.copy()
        probe = min(self.max_offset, self.offset_scales[0])
        for idx in range(1, len(pts) - 1):
            normal = _path_normal(pts, idx)
            left[idx, :2] = left[idx, :2] - probe * normal
            right[idx, :2] = right[idx, :2] + probe * normal
        left_risk = _trajectory_risk(risk_field, left)
        right_risk = _trajectory_risk(risk_field, right)
        if left_risk < right_risk:
            return -1.0
        if right_risk < left_risk:
            return 1.0
        return float(fallback)

    def _point_cost(self, point, original_point, payload, risk_field):
        risk_threshold = _finite_float(payload.get(
            "risk_threshold", payload.get("safe_threshold", 1.0)), 1.0)
        risk = _risk_value(risk_field, point)
        motion = float(np.linalg.norm(
            np.asarray(point, float)[:2] - np.asarray(original_point, float)[:2]))
        risk_cost = max(0.0, risk / max(risk_threshold, 1e-6))
        return float(10.0 * risk_cost + motion)

    def _candidate_cost(self, status, points):
        min_clearance = _finite_float(status.get("min_clearance", 0.0), 0.0)
        max_risk = _finite_float(status.get(
            "candidate_risk", status.get("max_risk", 0.0)), 0.0)
        required = _finite_float(status.get("required_clearance", 0.0), 0.0)
        risk_threshold = _finite_float(status.get("risk_threshold", 1.0), 1.0)
        clearance_deficit = max(0.0, required - min_clearance)
        risk_cost = max(0.0, max_risk / max(risk_threshold, 1e-6))
        tube_penalty = 0.0 if bool(status.get("tube_valid", False)) else 100.0
        valid_bonus = -1000.0 if self._success(status) else 0.0
        return float(
            valid_bonus +
            300.0 * clearance_deficit +
            220.0 * risk_cost +
            5.0 * _path_length(points) +
            tube_penalty)

    def _prefer_candidate(self, status, points, cost, best_status, best_points):
        tube_valid = bool(status.get("tube_valid", False))
        best_tube_valid = bool(best_status.get("tube_valid", False))
        if tube_valid != best_tube_valid:
            return tube_valid
        clearance = _finite_float(status.get("min_clearance", 0.0), 0.0)
        best_clearance = _finite_float(
            best_status.get("min_clearance", 0.0), 0.0)
        if abs(clearance - best_clearance) > 1e-9:
            return clearance > best_clearance
        risk = _finite_float(status.get(
            "candidate_risk", status.get("max_risk", 0.0)), 0.0)
        best_risk = _finite_float(best_status.get(
            "candidate_risk", best_status.get("max_risk", 0.0)), 0.0)
        if abs(risk - best_risk) > 1e-9:
            return risk < best_risk
        return _path_length(points) < _path_length(best_points)

    def _regenerate_boundary(self, centerline, payload):
        pts = _as_points(centerline)
        required = (
            _finite_float(payload.get("minimum_clearance",
                                      payload.get("min_clearance", 0.0)), 0.0) +
            max(0.0, _finite_float(
                payload.get("planning_clearance_margin", 0.0), 0.0)) +
            self.width_margin)
        tube_width = max(required, _finite_float(
            payload.get("corridor_radius", payload.get("radius", 0.0)), 0.0),
            0.08)
        left = []
        right = []
        for idx, point in enumerate(pts):
            normal = _path_normal(pts, idx)
            lp = point.copy()
            rp = point.copy()
            lp[:2] = lp[:2] + tube_width * normal
            rp[:2] = rp[:2] - tube_width * normal
            left.append(lp.tolist())
            right.append(rp.tolist())
        widths = [float(tube_width) for _ in left]
        return {
            "left": left,
            "right": right,
            "left_boundary": left,
            "right_boundary": right,
            "width": widths,
            "tube_width": float(tube_width),
            "corridor_width_profile": widths,
        }

    def _failure_reason(self, status):
        reasons = []
        if _finite_float(status.get("min_clearance", 0.0), 0.0) + 1e-9 < (
                _finite_float(status.get("required_clearance", 0.0), 0.0)):
            reasons.append("clearance_violation")
        if _finite_float(status.get("max_risk", 0.0), 0.0) > (
                _finite_float(status.get("risk_threshold", 1.0), 1.0) + 1e-9):
            reasons.append("risk_violation")
        if not bool(status.get("tube_valid", False)):
            reasons.append("tube_invalid")
        return ",".join(reasons)

    def _success(self, status):
        clearance = _finite_float(status.get("min_clearance", 0.0), 0.0)
        required = _finite_float(status.get("required_clearance", 0.0), 0.0)
        risk = _finite_float(status.get("max_risk", 0.0), 0.0)
        risk_threshold = _finite_float(status.get("risk_threshold", 1.0), 1.0)
        return bool(
            clearance + 1e-9 >= required and
            risk <= risk_threshold + 1e-9 and
            bool(status.get("tube_valid", False)))


def recover_candidates(candidates, manifold_constraint, robot_type="generic",
                       risk_field=None, max_iterations=3,
                       topology_ik_solver=None):
    """Unified candidate recovery for wheelchair and arm planners."""
    robot = str(robot_type or "").strip().lower()
    if robot == "arm":
        return recover_arm_candidates(
            candidates, manifold_constraint, risk_field=risk_field,
            topology_ik_solver=topology_ik_solver)
    recovered, report = recover_candidate_feasibility(
        candidates, manifold_constraint, risk_field=risk_field,
        max_iterations=max_iterations)
    report = dict(report or {})
    report["candidate_recovery_used"] = bool(report.get(
        "candidate_feasibility_recovery_used", False))
    report["candidate_recovery_robot_type"] = robot or "generic"
    return recovered, report


def recover_arm_candidates(candidates, manifold_constraint, risk_field=None,
                           topology_ik_solver=None):
    recovery = ArmCandidateRecovery()
    ik_solver = topology_ik_solver or TopologyIKSolver(
        risk_field=risk_field,
        boundary=(manifold_constraint or {}).get("boundary", None),
        risk_threshold=(manifold_constraint or {}).get("risk_threshold", 6.0),
        minimum_clearance=(manifold_constraint or {}).get(
            "minimum_clearance", (manifold_constraint or {}).get(
                "min_clearance", 0.0)))
    recovered = []
    report_records = []
    for candidate in list(candidates or []):
        route, diagnostics = recovery.recover(
            candidate, manifold_constraint, risk_field=risk_field)
        diagnostics = dict(diagnostics or {})
        original_reason = str((candidate or {}).get("failure_reason", ""))
        validation_reason = str(
            ((candidate or {}).get("arm_route_validation", {}) or {}).get(
                "failure_reason", ""))
        recoverable_execution_reason = bool(
            "link_collision" in original_reason or
            "link_collision" in validation_reason or
            "end_effector_clearance" in original_reason or
            "end_effector_clearance" in validation_reason)
        recovery_success = bool(diagnostics.get("recovery_success", False))
        ik_validation = {}
        if recovery_success or recoverable_execution_reason:
            route, ik_validation = ik_solver.validate_candidate(
                route, boundary=route.get("boundary", None),
                risk_field=risk_field)
            ik_validation = dict(ik_validation or {})
            ik_validation["arm_ik_candidate_attempts"] = list(route.get(
                "arm_ik_candidate_attempts", []))
            ik_validation["arm_ik_candidate_count"] = int(route.get(
                "arm_ik_candidate_count", 0))
            ik_validation["arm_pose_optimization_used"] = bool(route.get(
                "arm_pose_optimization_used", False))
            ik_validation["arm_pose_optimizer_score"] = float(route.get(
                "arm_pose_optimizer_score", 0.0))
            recovery_success = bool(route.get("ik_valid", False))
            if recovery_success and recoverable_execution_reason:
                route["candidate_tube_valid"] = True
                route["tube_valid"] = True
                route["candidate_recovery_mode"] = (
                    "arm_pose_ik_execution_recovery")
                diagnostics["recovery_success"] = True
                diagnostics["tube_valid_after"] = True
                diagnostics["new_tube_valid"] = True
        record = _arm_recovery_record(candidate, diagnostics, ik_validation)
        record["recovery_success"] = bool(recovery_success)
        record["ik_valid"] = bool((ik_validation or {}).get("valid", False))
        record["recoverable_execution_reason"] = bool(
            recoverable_execution_reason)
        record["recovery_mode"] = str(route.get(
            "candidate_recovery_mode", "clearance_geometry_optimization"))
        record["arm_pose_optimization_used"] = bool(route.get(
            "arm_pose_optimization_used", False))
        report_records.append(record)
        if not recovery_success:
            continue
        route["candidate_status"] = "feasible"
        route["failure_reason"] = ""
        route["route_source"] = str(route.get("route_source", "morse_topology"))
        route["candidate_source"] = "morse_recovered"
        route["topology_source"] = "morse_graph"
        route["candidate_generation_role"] = "morse_graph_topology_path"
        recovered.append(route)
    return recovered, {
        "candidate_recovery_used": bool(report_records),
        "candidate_recovery_robot_type": "arm",
        "candidate_recovery_attempted": int(len(report_records)),
        "candidate_recovery_success_count": int(len(recovered)),
        "arm_candidate_recovery_used": bool(report_records),
        "arm_candidate_recovery_attempted": int(len(report_records)),
        "arm_candidate_recovery_success_count": int(len(recovered)),
        "arm_candidate_recovery_report": report_records,
        "arm_tube_validation_report": [
            dict(item.get("arm_tube_validation", {}))
            for item in report_records
            if item.get("arm_tube_validation")
        ],
        "arm_global_recovery_report": [
            dict(item.get("global_recovery", {}))
            for item in report_records
            if item.get("global_recovery")
        ],
    }


def _arm_recovery_record(candidate, diagnostics, ik_validation):
    return {
        "candidate_id": str((candidate or {}).get("candidate_id", "")),
        "recovery_attempted": True,
        "recovery_success": bool(diagnostics.get("recovery_success", False)),
        "clearance_before": float(diagnostics.get(
            "clearance_before", diagnostics.get("original_clearance", 0.0))),
        "clearance_after": float(diagnostics.get(
            "clearance_after", diagnostics.get("optimized_clearance", 0.0))),
        "risk_before": float(diagnostics.get("risk_before", 0.0)),
        "risk_after": float(diagnostics.get("risk_after", 0.0)),
        "tube_valid_before": bool(diagnostics.get("tube_valid_before", False)),
        "tube_valid_after": bool(diagnostics.get("tube_valid_after", False)),
        "selected_offset": str(diagnostics.get("selected_offset", "")),
        "offset_direction": str(diagnostics.get("offset_direction", "")),
        "offset_distance": float(diagnostics.get("offset_distance", 0.0)),
        "candidate_clearance": float(diagnostics.get("candidate_clearance", 0.0)),
        "candidate_risk": float(diagnostics.get("candidate_risk", 0.0)),
        "candidate_length": float(diagnostics.get("candidate_length", 0.0)),
        "tested_offset_count": int(diagnostics.get("tested_offset_count", 0)),
        "best_clearance": float(diagnostics.get("best_clearance", 0.0)),
        "best_tube_valid": bool(diagnostics.get("best_tube_valid", False)),
        "arm_tube_validation": dict(diagnostics.get("arm_tube_validation", {})),
        "segment_diagnostics": list(diagnostics.get("segment_diagnostics", [])),
        "best_segment_clearance": float(diagnostics.get(
            "best_segment_clearance", 0.0)),
        "segment_candidate_count": int(diagnostics.get(
            "segment_candidate_count", 0)),
        "global_recovery": dict(diagnostics.get("global_recovery", {})),
        "old_tube_valid": bool(diagnostics.get("old_tube_valid", False)),
        "new_tube_valid": bool(diagnostics.get("new_tube_valid", False)),
        "old_clearance": float(diagnostics.get("old_clearance", 0.0)),
        "new_clearance": float(diagnostics.get("new_clearance", 0.0)),
        "new_centerline_count": int(diagnostics.get("new_centerline_count", 0)),
        "new_boundary_count": int(diagnostics.get("new_boundary_count", 0)),
        "search_failures": list(diagnostics.get("search_failures", [])),
        "ik_validation": dict(ik_validation or {}),
        "arm_ik_candidate_attempts": list(
            (ik_validation or {}).get("arm_ik_candidate_attempts", [])),
        "arm_ik_candidate_count": int(
            (ik_validation or {}).get("arm_ik_candidate_count", 0)),
    }
