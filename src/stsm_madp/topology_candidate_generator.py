import sys
sys.dont_write_bytecode = True

import heapq
import numpy as np

from stsm_madp.safety_evaluator import SafetyEvaluator


def _finite_float(value, default=None):
    try:
        value = float(value)
    except Exception:
        return default
    if not bool(np.isfinite(value)):
        return default
    return value


def _candidate_value(candidate, name, default=None):
    if isinstance(candidate, dict):
        return candidate.get(name, default)
    return getattr(candidate, name, default)


def _set_candidate_value(candidate, name, value):
    if isinstance(candidate, dict):
        candidate[name] = value
    else:
        setattr(candidate, name, value)


def candidate_topology_identity(candidate):
    """Stable Morse identity shared by original and recovered geometry."""
    explicit = str(_candidate_value(
        candidate, "original_topology_identity", "") or "")
    if explicit:
        return explicit
    topology_class = str(_candidate_value(
        candidate, "topology_class",
        _candidate_value(candidate, "topology_route_class", "")) or "")
    sequence = list(_candidate_value(
        candidate, "critical_point_sequence",
        _candidate_value(candidate, "node_sequence",
                         _candidate_value(candidate, "topology_nodes", []))) or [])
    node_ids = []
    for item in sequence:
        if isinstance(item, dict):
            node_ids.append(str(item.get("id", item.get("node_id", ""))))
        else:
            node_ids.append(str(item))
    node_ids = [item for item in node_ids if item]
    source_graph_id = str(_candidate_value(candidate, "source_graph_id", "") or "")
    identity = "|".join((source_graph_id, topology_class, ">".join(node_ids)))
    if identity.strip("|"):
        return identity
    return str(_candidate_value(
        candidate, "candidate_id",
        _candidate_value(candidate, "corridor_id", "")) or "")


def candidate_decision_record(candidate):
    """Return the explicit P0-4 hard-feasibility decision chain."""
    topology_valid = bool(_candidate_value(candidate, "topology_valid", True))
    geometry_valid = bool(len(_candidate_points(candidate)) >= 2)
    manifold_feasible = bool(_candidate_value(
        candidate, "manifold_feasible",
        _candidate_value(candidate, "manifold_valid", True)))
    tube_valid = bool(_candidate_value(
        candidate, "candidate_tube_valid",
        _candidate_value(candidate, "tube_valid", True)))
    risk_valid = bool(_candidate_value(candidate, "risk_valid", True))
    safety_feasible = bool(manifold_feasible and tube_valid and risk_valid)
    execution_feasible = bool(_candidate_value(
        candidate, "execution_feasible", geometry_valid))
    decision_robot_type = str(_candidate_value(
        candidate, "decision_robot_type", "") or "").lower()
    if (decision_robot_type == "arm" and
            _candidate_value(candidate, "ik_valid", None) is not None):
        execution_feasible = bool(
            execution_feasible and _candidate_value(candidate, "ik_valid", False))
    if (decision_robot_type == "arm" and
            _candidate_value(candidate, "link_collision_valid", None) is not None):
        execution_feasible = bool(
            execution_feasible and
            _candidate_value(candidate, "link_collision_valid", False))
    hard_feasible = bool(
        topology_valid and safety_feasible and execution_feasible)
    reasons = []
    if not topology_valid:
        reasons.append("topology_invalid")
    if not geometry_valid:
        reasons.append("geometry_invalid")
    if not manifold_feasible:
        reasons.append("manifold_infeasible")
    if not tube_valid:
        reasons.append("tube_infeasible")
    if not risk_valid:
        reasons.append("risk_infeasible")
    if not execution_feasible:
        reasons.append("execution_infeasible")
    stage = (
        "topology_valid" if not topology_valid else
        "safety_feasible" if not safety_feasible else
        "execution_feasible" if not execution_feasible else
        "ranked")
    return {
        "candidate_id": str(_candidate_value(
            candidate, "candidate_id",
            _candidate_value(candidate, "corridor_id", "")) or ""),
        "original_topology_identity": candidate_topology_identity(candidate),
        "topology_class": str(_candidate_value(
            candidate, "topology_class",
            _candidate_value(candidate, "topology_route_class", "")) or ""),
        "topology_valid": topology_valid,
        "geometry_valid": geometry_valid,
        "safety_feasible": safety_feasible,
        "execution_feasible": execution_feasible,
        "hard_feasible": hard_feasible,
        "decision_stage": stage,
        "decision_reason": "eligible" if hard_feasible else ",".join(reasons),
        "recovery_used": bool(_candidate_value(
            candidate, "candidate_recovered",
            _candidate_value(candidate, "recovery_used", False))),
        "recovery_cost": float(_finite_float(_candidate_value(
            candidate, "normalized_recovery_cost",
            _candidate_value(candidate, "recovery_cost", 0.0)), 0.0)),
    }


def rank_feasible_candidates(candidates):
    """Hard-filter then deterministically rank candidates by explained cost."""
    candidates = list(candidates or [])
    eligible = []
    records = []
    for candidate in candidates:
        record = candidate_decision_record(candidate)
        score = float(_finite_float(_candidate_value(
            candidate, "total_score",
            _candidate_value(candidate, "cost", float("inf"))), float("inf")))
        topology_value = float(_finite_float(
            _candidate_value(candidate, "topology_value", 0.0), 0.0))
        geometry_tie_breaker = float(_finite_float(
            _candidate_value(candidate, "path_length",
                             _candidate_value(candidate, "length_cost", 0.0)), 0.0))
        record.update({
            "ranking_score": score,
            "topology_value": topology_value,
            "geometry_tie_breaker": geometry_tie_breaker,
            "ranking_eligible": bool(record["hard_feasible"]),
            "ranking_decomposition": {
                "comparison_cost": score,
                "topology_value": topology_value,
                "geometry_tie_breaker": geometry_tie_breaker,
                "recovery_cost": record["recovery_cost"],
            },
        })
        records.append(record)
        if record["hard_feasible"]:
            eligible.append((candidate, record))
        else:
            _set_candidate_value(candidate, "hard_feasible", False)
            _set_candidate_value(candidate, "decision_stage", record["decision_stage"])
            _set_candidate_value(candidate, "decision_reason", record["decision_reason"])
    eligible.sort(key=lambda item: (
        float(item[1]["ranking_score"]),
        -float(item[1]["topology_value"]),
        float(item[1]["geometry_tie_breaker"]),
        str(item[1]["original_topology_identity"]),
        str(item[1]["candidate_id"])))
    for rank, (candidate, record) in enumerate(eligible):
        record["rank"] = int(rank)
        record["selected"] = bool(rank == 0)
        record["decision_reason"] = (
            "minimum_explained_cost" if rank == 0 else
            "higher_explained_cost_or_tie_breaker")
        _set_candidate_value(candidate, "hard_feasible", True)
        _set_candidate_value(candidate, "decision_stage", "ranked")
        _set_candidate_value(candidate, "decision_reason", record["decision_reason"])
        _set_candidate_value(candidate, "original_topology_identity",
                             record["original_topology_identity"])
        _set_candidate_value(candidate, "ranking_decomposition",
                             dict(record["ranking_decomposition"]))
    return [item[0] for item in eligible], records


def _nearest_grid_sample(safe_manifold, point):
    grid = safe_manifold or {}
    xs = grid.get("xs")
    ys = grid.get("ys")
    if xs is None or ys is None:
        return {}
    try:
        xs_arr = np.asarray(xs, float)
        ys_arr = np.asarray(ys, float)
        i = int(np.argmin(np.abs(xs_arr - float(point[0]))))
        j = int(np.argmin(np.abs(ys_arr - float(point[1]))))
    except Exception:
        return {}

    def sample(name, default=None):
        data = grid.get(name)
        if data is None:
            return default
        try:
            return _finite_float(np.asarray(data, float)[i, j], default)
        except Exception:
            return default

    return {
        "i": i,
        "j": j,
        "clearance": sample("clearance"),
        "rho": sample("rho"),
        "phi": sample("phi"),
        "forbidden": bool(sample("forbidden", 0.0) or 0.0),
        "hard_clearance": _finite_float(grid.get("hard_clearance"), 0.0),
        "min_clearance": _finite_float(grid.get("min_clearance"), 0.0),
    }


def _path_tangent(pts, idx):
    if len(pts) == 1:
        return np.array([1.0, 0.0], float)
    if idx == 0:
        vec = pts[1, :2] - pts[0, :2]
    elif idx == len(pts) - 1:
        vec = pts[-1, :2] - pts[-2, :2]
    else:
        vec = pts[idx + 1, :2] - pts[idx - 1, :2]
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-9:
        return np.array([1.0, 0.0], float)
    return vec / norm


def _point_risk(risk_field, point, grid_sample):
    risk = None
    if risk_field is not None:
        try:
            risk = _finite_float(risk_field.phi_s(point))
        except Exception:
            risk = None
    if risk is None:
        risk = _finite_float(grid_sample.get("phi"), 0.0)
    return float(risk if risk is not None else 0.0)


def generate_manifold_aware_corridor(centerline, safe_manifold=None,
                                     risk_field=None, default_radius=0.35,
                                     alpha=1.0, width_min=None,
                                     width_max=None):
    """Generate adaptive corridor boundaries from manifold clearance and risk."""
    pts = np.asarray(centerline, float)
    if len(pts) == 0:
        return {
            "left": [],
            "right": [],
            "width": [],
            "corridor_width_profile": [],
            "local_clearance_profile": [],
            "risk_profile": [],
        }
    radius = float(default_radius)
    width_min = float(width_min if width_min is not None else max(0.08, 0.25 * radius))
    width_max = float(width_max if width_max is not None else max(width_min, 2.0 * radius))
    left = []
    right = []
    widths = []
    clearances = []
    risks = []
    for idx, point in enumerate(pts):
        sample = _nearest_grid_sample(safe_manifold, point)
        clearance = _finite_float(sample.get("clearance"))
        if clearance is None:
            clearance = width_max
        rho = _finite_float(sample.get("rho"))
        phi = _point_risk(risk_field, point, sample)
        if rho is not None and rho > 1e-9:
            safe_fraction = np.clip((rho - max(0.0, phi)) / rho, 0.15, 1.0)
        else:
            safe_fraction = 1.0 / (1.0 + max(0.0, phi))
            safe_fraction = np.clip(safe_fraction, 0.15, 1.0)
        d_safe = max(0.0, min(float(clearance), width_max))
        width = float(np.clip(float(alpha) * d_safe * safe_fraction,
                              width_min, width_max))
        tangent = _path_tangent(pts, idx)
        normal = np.array([-tangent[1], tangent[0]], float)
        nlen = float(np.linalg.norm(normal))
        if nlen <= 1e-9:
            normal = np.array([0.0, 1.0], float)
        else:
            normal = normal / nlen
        lp = np.asarray(point, float).copy()
        rp = np.asarray(point, float).copy()
        lp[:2] = lp[:2] + width * normal
        rp[:2] = rp[:2] - width * normal
        left.append(lp.tolist())
        right.append(rp.tolist())
        widths.append(float(width))
        clearances.append(float(clearance))
        risks.append(float(phi))
    return {
        "left": left,
        "right": right,
        "width": widths,
        "corridor_width_profile": widths,
        "local_clearance_profile": clearances,
        "risk_profile": risks,
    }


def validate_manifold_corridor(corridor, safe_manifold=None, risk_field=None,
                               risk_threshold=None):
    """Validate that a generated corridor stays inside the safe manifold."""
    centerline = corridor.get("centerline", []) if isinstance(corridor, dict) else []
    boundary = corridor.get("boundary", {}) if isinstance(corridor, dict) else {}
    pts = []
    for series in (centerline, boundary.get("left", []), boundary.get("right", [])):
        for point in series or []:
            try:
                pts.append(np.asarray(point, float))
            except Exception:
                continue
    width_profile = (
        corridor.get("corridor_width_profile") or
        boundary.get("corridor_width_profile") or
        boundary.get("width") or [])
    clearances = []
    risks = []
    forbidden_hits = 0
    hard_clearance = _finite_float((safe_manifold or {}).get("hard_clearance"), 0.0)
    for point in pts:
        sample = _nearest_grid_sample(safe_manifold, point)
        clearance = _finite_float(sample.get("clearance"))
        if clearance is not None:
            clearances.append(clearance)
        if sample.get("forbidden"):
            forbidden_hits += 1
        risk = _point_risk(risk_field, point, sample)
        risks.append(float(risk))
    min_clearance = min(clearances) if clearances else 0.0
    average_width = (
        float(np.mean(np.asarray(width_profile, float)))
        if width_profile else 0.0)
    max_risk = max(risks) if risks else 0.0
    if risk_threshold is None:
        rho = (safe_manifold or {}).get("rho")
        risk_threshold = (
            float(np.nanmax(np.asarray(rho, float)))
            if rho is not None and np.asarray(rho).size else None)
    risk_valid = True
    if risk_threshold is not None and bool(np.isfinite(float(risk_threshold))):
        risk_valid = bool(max_risk <= float(risk_threshold) + 1e-9)
    manifold_valid = bool(
        forbidden_hits == 0 and
        min_clearance >= max(0.0, float(hard_clearance)) and
        risk_valid)
    return {
        "manifold_valid": manifold_valid,
        "min_clearance": float(min_clearance),
        "average_width": float(average_width),
        "average_corridor_width": float(average_width),
        "max_risk": float(max_risk),
        "forbidden_hits": int(forbidden_hits),
        "risk_valid": bool(risk_valid),
    }


def _constraint_value(manifold_constraint, key, default=None):
    if isinstance(manifold_constraint, dict):
        return manifold_constraint.get(key, default)
    return getattr(manifold_constraint, key, default)


def _candidate_points(candidate_corridor):
    if candidate_corridor is None:
        return []
    points = []

    def value(name, default=None):
        if isinstance(candidate_corridor, dict):
            return candidate_corridor.get(name, default)
        return getattr(candidate_corridor, name, default)

    for name in ("centerline", "waypoints", "refined_waypoints"):
        series = value(name, [])
        if series is None:
            series = []
        for point in series:
            try:
                points.append(np.asarray(point, float))
            except Exception:
                continue
        if points:
            break
    return points


def _normal_at(points, idx):
    pts = np.asarray(points, float)
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


def _offset_boundary(centerline, half_width):
    pts = np.asarray(centerline, float)
    if pts.size == 0:
        return {"left": [], "right": [], "width": []}
    if pts.ndim == 1:
        pts = pts.reshape((1, pts.shape[0]))
    if pts.shape[1] == 2:
        pts = np.hstack([pts, np.zeros((pts.shape[0], 1), float)])
    left = []
    right = []
    width = float(max(0.0, half_width))
    for idx, point in enumerate(pts[:, :3]):
        normal = _normal_at(pts, idx)
        lp = point.copy()
        rp = point.copy()
        lp[:2] = lp[:2] + width * normal
        rp[:2] = rp[:2] - width * normal
        left.append(lp.tolist())
        right.append(rp.tolist())
    widths = [width for _ in left]
    return {
        "left": left,
        "right": right,
        "width": widths,
        "corridor_width_profile": widths,
    }


def _turning_cost(points):
    pts = _candidate_points({"centerline": points})
    if len(pts) < 3:
        return 0.0, 0.0
    turns = []
    for a, b, c in zip(pts[:-2], pts[1:-1], pts[2:]):
        u = b[:2] - a[:2]
        v = c[:2] - b[:2]
        nu = float(np.linalg.norm(u))
        nv = float(np.linalg.norm(v))
        if nu <= 1e-9 or nv <= 1e-9:
            continue
        dot = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
        turns.append(float(np.arccos(dot)))
    if not turns:
        return 0.0, 0.0
    return float(np.sum(np.square(turns))), float(np.max(turns))


def recover_candidate_corridor_feasibility(candidate_corridor,
                                           manifold_constraint,
                                           risk_field=None,
                                           clearance_padding=1e-3):
    """Post-generation tube recovery without changing topology generation."""
    if candidate_corridor is None:
        return None, {}
    route = dict(candidate_corridor)
    centerline = (
        route.get("centerline") or route.get("waypoints") or
        route.get("refined_waypoints") or [])
    pts = np.asarray(centerline, float)
    if pts.size == 0:
        return None, {}
    payload = dict(manifold_constraint or {})
    minimum = _finite_float(
        payload.get("minimum_clearance", payload.get("min_clearance", 0.0)),
        0.0)
    margin = _finite_float(payload.get("planning_clearance_margin", 0.0), 0.0)
    required = float(minimum + max(0.0, margin) + max(0.0, clearance_padding))
    boundary = _offset_boundary(centerline, required)
    route["boundary"] = boundary
    route["corridor_width_profile"] = list(boundary.get("width", []))
    route["average_corridor_width"] = float(required)
    route["recovery_used"] = True
    route["topology_recovery_used"] = True
    route["route_source"] = str(route.get("route_source", "morse_topology"))
    route["candidate_source"] = "morse_recovered"
    route["candidate_recovery_mode"] = "topology_tube_expansion"
    route["candidate_generation_role"] = str(route.get(
        "candidate_generation_role", "morse_graph_topology_path"))
    recovery_constraint = dict(payload)
    recovery_constraint["boundary"] = boundary
    status = evaluate_candidate_manifold_feasibility(
        route, recovery_constraint, risk_field=risk_field)
    route["manifold_feasibility"] = dict(status)
    route["manifold_feasible"] = bool(status.get("feasible", False))
    route["candidate_status"] = str(status.get(
        "candidate_status", "manifold_infeasible"))
    route["failure_reason"] = str(status.get("failure_reason", ""))
    route["risk_valid"] = bool(status.get("risk_valid", True))
    route["candidate_tube_valid"] = bool(status.get(
        "candidate_tube_valid", status.get("tube_valid", False)))
    route["tube_valid"] = bool(route["candidate_tube_valid"])
    route["min_clearance"] = float(status.get("min_clearance", 0.0))
    route["trajectory_min_clearance"] = float(status.get(
        "trajectory_min_clearance", route["min_clearance"]))
    route["trajectory_max_risk"] = float(status.get("trajectory_max_risk", 0.0))
    route["max_risk"] = float(status.get("max_risk", 0.0))
    return route, status


def evaluate_candidate_manifold_feasibility(candidate_trajectory,
                                            manifold_constraint,
                                            risk_field=None):
    """Trajectory-level candidate hard gate using the shared manifold evaluator."""
    manifold_constraint = manifold_constraint or {}
    if risk_field is None:
        risk_field = _constraint_value(manifold_constraint, "risk_field", None)
    trajectory = (
        _candidate_points(candidate_trajectory)
        if not isinstance(candidate_trajectory, (list, tuple, np.ndarray)) else
        candidate_trajectory)
    payload = dict(manifold_constraint or {})
    if not payload.get("boundary"):
        boundary = (
            candidate_trajectory.get("boundary", {})
            if isinstance(candidate_trajectory, dict) else
            getattr(candidate_trajectory, "boundary", {}))
        if boundary:
            payload["boundary"] = boundary
    margin = _finite_float(payload.get("planning_clearance_margin", 0.0), 0.0)
    corridor_constraint = {
        "centerline": (
            candidate_trajectory.get("centerline", [])
            if isinstance(candidate_trajectory, dict) else
            getattr(candidate_trajectory, "centerline", [])),
        "radius": payload.get("corridor_radius", payload.get("radius", 0.0)),
    }
    evaluator = SafetyEvaluator(
        manifold_constraint=payload,
        corridor_constraint=corridor_constraint,
        risk_field=risk_field,
        planning_clearance_margin=margin)
    status = evaluator.evaluate_trajectory(trajectory)
    tube_status = evaluator.evaluate_corridor(candidate_trajectory)
    feasible = bool(status.get("valid", False) and
                    tube_status.get("tube_valid", False))
    reason = ""
    if not feasible:
        if int(status.get("manifold_violation_count", 0)):
            reason = "clearance_violation"
        elif int(status.get("corridor_violation_count", 0)) or int(
                tube_status.get("tube_corridor_violation_count", 0)):
            reason = "corridor_violation"
        else:
            reason = "manifold_violation"
    return {
        "feasible": feasible,
        "manifold_feasible": feasible,
        "clearance": float(status.get("min_clearance", 0.0)),
        "risk": float(status.get("max_risk", 0.0)),
        "manifold_valid": bool(status.get("valid", False)),
        "min_clearance": float(status.get("min_clearance", 0.0)),
        "trajectory_min_clearance": float(status.get("min_clearance", 0.0)),
        "minimum_clearance": float(status.get("minimum_clearance", 0.0)),
        "required_clearance": float(status.get("required_clearance", 0.0)),
        "planning_clearance_margin": float(status.get(
            "planning_clearance_margin", 0.0)),
        "max_risk": float(status.get("max_risk", 0.0)),
        "trajectory_max_risk": float(status.get("max_risk", 0.0)),
        "risk_threshold": float(status.get("risk_threshold", 0.0)),
        "risk_valid": bool(float(status.get("max_risk", 0.0)) <=
                           float(status.get("risk_threshold", 0.0)) + 1e-9),
        "clearance_valid": bool(float(status.get("min_clearance", 0.0)) + 1e-9 >=
                                float(status.get("required_clearance", 0.0))),
        "violation_count": int(status.get("manifold_violation_count", 0)) + int(
            status.get("corridor_violation_count", 0)),
        "clearance_violation_count": int(status.get(
            "manifold_violation_count", 0)),
        "risk_violation_count": int(
            1 if float(status.get("max_risk", 0.0)) >
            float(status.get("risk_threshold", 0.0)) + 1e-9 else 0),
        "tube_valid": bool(tube_status.get("tube_valid", False)),
        "candidate_tube_valid": bool(tube_status.get("tube_valid", False)),
        "min_tube_clearance": float(tube_status.get("min_tube_clearance", 0.0)),
        "candidate_status": "feasible" if feasible else "manifold_infeasible",
        "failure_reason": reason,
    }


def evaluate_candidate(candidate_trajectory, manifold_constraint,
                       risk_field=None):
    """Single candidate safety entry point used by generation/recovery/ranking."""
    status = evaluate_candidate_manifold_feasibility(
        candidate_trajectory,
        manifold_constraint,
        risk_field=risk_field)
    geometry_valid = bool(len(_candidate_points(candidate_trajectory)) >= 2)
    failure = []
    if not geometry_valid:
        failure.append("geometry_invalid")
    if not bool(status.get("clearance_valid", False)):
        failure.append("clearance_violation")
    if not bool(status.get("risk_valid", True)):
        failure.append("risk_violation")
    if not bool(status.get("tube_valid", False)):
        failure.append("tube_invalid")
    candidate_status = (
        "safe" if bool(status.get("feasible", False)) else
        "recoverable" if geometry_valid else
        "invalid")
    out = dict(status)
    out.update({
        "clearance": float(status.get("clearance", status.get("min_clearance", 0.0))),
        "risk": float(status.get("risk", status.get("max_risk", 0.0))),
        "geometry_valid": bool(geometry_valid),
        "manifold_valid": bool(status.get("manifold_valid", False)),
        "tube_valid": bool(status.get("tube_valid", False)),
        "candidate_status": candidate_status,
        "failure_reason": failure,
    })
    return out


def check_candidate_manifold_feasibility(candidate_corridor,
                                         manifold_constraint):
    return evaluate_candidate(
        candidate_corridor,
        manifold_constraint,
        risk_field=_constraint_value(manifold_constraint or {}, "risk_field", None))


class TopologyDrivenCandidateGenerator(object):
    """Derive candidate corridor routes from the Morse connectivity graph."""

    def __init__(self, grid=None, world_from_ij=None, field=None,
                 default_radius=0.35, max_paths=512, max_routes=256,
                 min_corridor_width=None, arm_search_enabled=False,
                 arm_search_validator=None, arm_search_weights=None,
                 robot_type="generic"):
        self.grid = grid or {}
        self.world_from_ij = world_from_ij
        self.field = field
        self.robot_type = str(robot_type or "generic").strip().lower()
        self.default_radius = float(default_radius)
        self.max_paths = max(1, int(max_paths))
        self.max_routes = max(1, int(max_routes))
        self.min_corridor_width = (
            None if min_corridor_width is None else float(min_corridor_width))
        self.arm_search_enabled = bool(
            arm_search_enabled or self.robot_type == "arm")
        self.arm_search_validator = arm_search_validator
        self.arm_search_weights = dict(arm_search_weights or {})
        self.last_report = {}

    def build_dense_route_trajectory(self, node_ids, node_by_id,
                                     cells=None, start=None, goal=None,
                                     spacing=None):
        points = []
        if start is not None:
            try:
                points.append(np.asarray(start, float)[:3].tolist())
            except Exception:
                pass
        if cells:
            points.extend(self._centerline_from_cells(cells).tolist())
        else:
            for node_id in node_ids or []:
                node = node_by_id.get(str(node_id))
                if node is None:
                    continue
                points.append(np.asarray(getattr(node, "point", []), float).tolist())
        if goal is not None:
            try:
                points.append(np.asarray(goal, float)[:3].tolist())
            except Exception:
                pass
        pts = np.asarray(points, float)
        if pts.size == 0:
            return np.empty((0, 3), float)
        if pts.ndim == 1:
            pts = pts.reshape((1, pts.shape[0]))
        if pts.shape[1] == 2:
            pts = np.hstack([pts, np.zeros((pts.shape[0], 1), float)])
        if len(pts) < 2:
            return np.empty((0, 3), float)
        dense = []
        step = float(spacing) if spacing is not None else 0.10
        xs = self.grid.get("xs") if isinstance(self.grid, dict) else None
        if spacing is None and xs is not None:
            try:
                xs_arr = np.asarray(xs, float)
                if len(xs_arr) > 1:
                    step = max(0.05, float(np.min(np.diff(xs_arr))) * 1.5)
            except Exception:
                step = 0.10
        for a, b in zip(pts[:-1], pts[1:]):
            a = np.asarray(a, float)[:3]
            b = np.asarray(b, float)[:3]
            dist = float(np.linalg.norm(b[:2] - a[:2]))
            count = max(2, int(np.ceil(dist / max(step, 1e-6))) + 1)
            for idx in range(count):
                if dense and idx == 0:
                    continue
                alpha = float(idx) / float(max(count - 1, 1))
                dense.append((a + alpha * (b - a)).tolist())
        return np.asarray(dense, float)

    def route_interpolation(self, node_ids, node_by_id, cells=None):
        return self.build_dense_route_trajectory(
            node_ids, node_by_id, cells=cells)

    def validate_route(self, trajectory, start=None, goal=None):
        pts = np.asarray(trajectory, float)
        reasons = []
        if pts.size == 0:
            reasons.append("empty_trajectory")
            return {"route_valid": False, "failure_reason": reasons}
        if pts.ndim == 1:
            pts = pts.reshape((1, pts.shape[0]))
        if pts.shape[1] == 2:
            pts = np.hstack([pts, np.zeros((pts.shape[0], 1), float)])
        if len(pts) < 2:
            reasons.append("insufficient_points")
        if len(pts) >= 2:
            seg = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
            max_step = float(np.max(seg)) if len(seg) else 0.0
            nominal = 0.25
            xs = self.grid.get("xs") if isinstance(self.grid, dict) else None
            if xs is not None:
                try:
                    xs_arr = np.asarray(xs, float)
                    if len(xs_arr) > 1:
                        nominal = max(nominal, 3.0 * float(np.min(np.diff(xs_arr))))
                except Exception:
                    pass
            if max_step > nominal + 1e-9:
                reasons.append("trajectory_discontinuous")
        if start is not None and len(pts):
            try:
                if float(np.linalg.norm(pts[0, :2] - np.asarray(start, float)[:2])) > 0.30:
                    reasons.append("start_disconnected")
            except Exception:
                pass
        if goal is not None and len(pts):
            try:
                if float(np.linalg.norm(pts[-1, :2] - np.asarray(goal, float)[:2])) > 0.30:
                    reasons.append("goal_disconnected")
            except Exception:
                pass
        return {
            "route_valid": bool(not reasons),
            "failure_reason": reasons,
            "point_count": int(len(pts)),
            "max_step": float(max_step if len(pts) >= 2 else 0.0),
        }

    def _smooth_trajectory(self, trajectory):
        pts = np.asarray(trajectory, float)
        if len(pts) < 5:
            return pts
        out = pts.copy()
        for idx in range(1, len(pts) - 1):
            out[idx, :2] = 0.25 * pts[idx - 1, :2] + 0.50 * pts[idx, :2] + 0.25 * pts[idx + 1, :2]
        return out

    def _adaptive_corridor(self, centerline):
        boundary = generate_manifold_aware_corridor(
            centerline, self.grid, self.field,
            default_radius=self.default_radius,
            width_min=self.min_corridor_width)
        widths = list(boundary.get(
            "corridor_width_profile", boundary.get("width", [])) or [])
        if not widths:
            return boundary
        adapted = []
        for idx, point in enumerate(np.asarray(centerline, float)):
            sample = _nearest_grid_sample(self.grid, point)
            risk = _point_risk(self.field, point, sample)
            rho = sample.get("rho")
            if rho is None or abs(float(rho)) <= 1e-9:
                risk_ratio = min(1.0, max(0.0, risk / (1.0 + risk)))
            else:
                risk_ratio = min(1.0, max(0.0, risk / max(abs(float(rho)), 1e-6)))
            base = float(widths[min(idx, len(widths) - 1)])
            adapted.append(float(base * (1.0 + 0.75 * risk_ratio)))
        boundary["width"] = adapted
        boundary["corridor_width_profile"] = list(adapted)
        return boundary

    def _adaptive_speed_profile(self, centerline, nominal_speed=0.35):
        speeds = []
        for point in np.asarray(centerline, float):
            sample = _nearest_grid_sample(self.grid, point)
            risk = _point_risk(self.field, point, sample)
            rho = sample.get("rho")
            if rho is None or abs(float(rho)) <= 1e-9:
                risk_ratio = min(1.0, max(0.0, risk / (1.0 + risk)))
            else:
                risk_ratio = min(1.0, max(0.0, risk / max(abs(float(rho)), 1e-6)))
            speeds.append(float(max(0.08, nominal_speed * (1.0 - 0.55 * risk_ratio))))
        return speeds

    def _route_length(self, trajectory):
        pts = np.asarray(trajectory, float)
        if len(pts) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)))

    def _arm_morse_search_cost(self, topology_cost, route, centerline,
                               route_eval):
        validation = {}
        if self.arm_search_validator is not None:
            payload = {
                "candidate_id": str(route.get("candidate_id", "")),
                "route_id": str(route.get("candidate_id", "")),
                "critical_point_sequence": list(route.get(
                    "critical_point_sequence", []) or []),
                "centerline": np.asarray(centerline, float).tolist(),
                "waypoints": np.asarray(centerline, float).tolist(),
                "boundary": dict(route.get("boundary", {}) or {}),
            }
            validation = dict(
                self.arm_search_validator.validate_route(payload) or {})
        clearance = float(validation.get(
            "min_end_effector_clearance",
            route_eval.get("min_clearance", 0.0)))
        link_clearance = float(validation.get(
            "min_link_clearance", clearance))
        risk = float(validation.get(
            "max_risk", route_eval.get("max_risk", 0.0)))
        weights = self.arm_search_weights
        w_clearance = float(weights.get("clearance", 1.0))
        w_link = float(weights.get("link_clearance", 1.0))
        w_risk = float(weights.get("risk", 0.25))
        clearance_cost = w_clearance / max(1e-6, max(0.0, clearance))
        link_cost = w_link / max(1e-6, max(0.0, link_clearance))
        risk_cost = w_risk * max(0.0, risk)
        total_cost = float(topology_cost + clearance_cost + link_cost + risk_cost)
        validation_failed = bool(
            validation and not validation.get("route_valid", True))
        collision_link = str(validation.get("worst_link_id", ""))
        return {
            "robot_type": str(self.robot_type),
            "topology_cost": float(topology_cost),
            "J_topology": float(topology_cost),
            "clearance_cost": float(clearance_cost),
            "link_cost": float(link_cost),
            "J_link": float(link_cost),
            "risk_cost": float(risk_cost),
            "J_risk": float(risk_cost),
            "total_cost": float(total_cost),
            "J": float(total_cost),
            "clearance": float(clearance),
            "link_clearance": float(link_clearance),
            "risk": float(risk),
            "upper_arm_collision": bool(
                validation_failed and "upper_arm" in collision_link),
            "link_collision": bool(
                validation_failed and not validation.get(
                    "link_collision_valid", True)),
            "collision_link": collision_link,
            "arm_route_validation": dict(validation),
        }

    def apply_arm_route_ranking(self, routes, ranking, top_k=None):
        """Sort Morse routes by arm-aware scores and keep only selected routes."""
        routes = list(routes or [])
        ranking = [dict(item) for item in list(ranking or [])]
        score_by_id = {
            str(item.get("route_id", "")): float(item.get("score", 0.0))
            for item in ranking
        }
        top_k = len(routes) if top_k is None else int(top_k)
        top_k = max(0, min(len(routes), top_k))

        sorted_routes = sorted(
            routes,
            key=lambda route: (
                -float(score_by_id.get(str(route.get("candidate_id", "")), 0.0)),
                float(route.get("base_cost", 0.0)),
                str(route.get("candidate_id", ""))))
        selected_ids = set(
            str(route.get("candidate_id", "")) for route in sorted_routes[:top_k])
        sorted_ranking = sorted(
            ranking,
            key=lambda item: (
                -float(item.get("score", 0.0)),
                str(item.get("route_id", ""))))
        for item in sorted_ranking:
            item["selected"] = bool(str(item.get("route_id", "")) in selected_ids)
        for route in sorted_routes:
            route["arm_route_rank_score"] = float(score_by_id.get(
                str(route.get("candidate_id", "")), 0.0))
            route["route_source"] = str(route.get(
                "route_source", "morse_topology"))
            route["critical_point_sequence"] = list(route.get(
                "critical_point_sequence", []) or [])
        return sorted_routes[:top_k], sorted_ranking

    def _evaluate_route(self, route, centerline, boundary):
        minimum = 0.0
        if isinstance(self.grid, dict):
            hard = self.grid.get("hard_clearance")
            try:
                minimum = float(np.nanmax(np.asarray(hard, float)))
            except Exception:
                minimum = _finite_float(hard, 0.0) or 0.0
        risk_threshold = getattr(self.field, "rho", None)
        if risk_threshold is None and isinstance(self.grid, dict):
            rho = self.grid.get("rho")
            try:
                risk_threshold = float(np.nanmax(np.asarray(rho, float)))
            except Exception:
                risk_threshold = 1.0
        payload = {
            "boundary": boundary,
            "minimum_clearance": float(minimum),
            "min_clearance": float(minimum),
            "risk_threshold": float(risk_threshold),
            "safe_threshold": float(risk_threshold),
            "planning_clearance_margin": 0.0,
        }
        evaluator = SafetyEvaluator(
            manifold_constraint=payload,
            corridor_constraint={
                "centerline": np.asarray(centerline, float).tolist(),
                "radius": float(self.default_radius),
            },
            risk_field=self.field)
        status = evaluator.evaluate_trajectory(centerline)
        failure = []
        if len(centerline) < 2:
            failure.append("geometry_failure")
        if int(status.get("manifold_violation_count", 0)):
            failure.append("clearance_failure")
        if float(status.get("max_risk", 0.0)) > float(status.get("risk_threshold", risk_threshold)) + 1e-9:
            failure.append("risk_failure")
        return {
            "route_id": str(route.get("candidate_id", "")),
            "length": float(self._route_length(centerline)),
            "min_clearance": float(status.get("min_clearance", 0.0)),
            "max_risk": float(status.get("max_risk", 0.0)),
            "manifold_valid": bool(status.get("valid", False)),
            "route_valid": bool(len(failure) == 0 and status.get("valid", False)),
            "failure_reason": failure,
            "safety_status": dict(status),
        }

    def generate(self, nodes, edges, start=None, goal=None):
        node_by_id = {str(getattr(node, "id", "")): node for node in nodes or []}
        route_levels = [
            ("saddle_sequence", True),
            ("critical_sequence", False),
        ]
        routes = []
        selected_level = ""
        level_reports = []
        for level_name, require_saddle in route_levels:
            level_routes, level_report = self._generate_level(
                edges, node_by_id, level_name, require_saddle)
            level_reports.append(level_report)
            if level_routes:
                routes = level_routes
                selected_level = level_name
                break
        for idx, route in enumerate(routes):
            route["candidate_id"] = "morse_topology_candidate_%04d" % (idx + 1)
            route["morse_route_evaluation"]["route_id"] = route["candidate_id"]
        route_evaluations = [
            dict(route.get("morse_route_evaluation", {}) or {})
            for route in routes
        ]
        candidate_attempts = []
        for route in routes:
            candidate_attempts.append({
                "route_id": str(route.get("candidate_id", "")),
                "candidate_generated": True,
                "failure_reason": list(route.get(
                    "morse_route_evaluation", {}).get(
                    "failure_reason", []) or []),
                "candidate_status": str(route.get("candidate_status", "")),
                "route_source": str(route.get(
                    "route_source", "morse_topology")),
            })
        route_validation_report = [
            dict(route.get("route_validation", {}) or {})
            for route in routes
        ]
        saddle_route_debug = []
        for level_report in level_reports:
            saddle_route_debug.extend(list(
                level_report.get("morse_saddle_route_debug", []) or []))
        arm_morse_search_diagnostics = []
        for route in routes:
            search_cost = dict(route.get("arm_morse_search_cost", {}) or {})
            if not search_cost:
                continue
            arm_morse_search_diagnostics.append({
                "route_id": str(route.get("candidate_id", "")),
                "topology_cost": float(search_cost.get(
                    "topology_cost", route.get("topology_cost", 0.0))),
                "clearance_cost": float(search_cost.get("clearance_cost", 0.0)),
                "link_cost": float(search_cost.get("link_cost", 0.0)),
                "risk_cost": float(search_cost.get("risk_cost", 0.0)),
                "total_cost": float(search_cost.get(
                    "total_cost", route.get("base_cost", 0.0))),
            })
        self.last_report = {
            "num_candidates_generated": int(len(routes)),
            "candidate_generated": int(len(routes)),
            "generation_method": (
                "morse_topology_induced" if routes else "no_topology_candidate"),
            "route_generation_level": selected_level,
            "route_generation_levels": level_reports,
            "morse_route_evaluation": route_evaluations,
            "route_validation_report": route_validation_report,
            "morse_saddle_route_debug": saddle_route_debug,
            "morse_saddle_route_debug_file": "morse_saddle_route_debug.json",
            "arm_morse_search_diagnostics": arm_morse_search_diagnostics,
            "arm_morse_search_diagnostics_file": (
                "arm_morse_search_diagnostics.json"),
            "candidate_generation_attempts": candidate_attempts,
            "critical_points_used": bool(routes),
            "manifold_used": True,
            "heuristic_sampling_used": False,
            "heuristic_fallback_used": False,
            "morse_induced_candidate": bool(routes),
            "manifold_adaptive": bool(routes),
            "manifold_adaptive_corridor": bool(routes),
            "risk_adaptive_width": bool(routes),
            "route_source": "morse_topology" if routes else "",
            "average_corridor_width": float(np.mean([
                float(route.get("average_corridor_width", 0.0))
                for route in routes
            ])) if routes else 0.0,
            "min_corridor_clearance": float(min([
                float(route.get("min_clearance", 0.0))
                for route in routes
            ])) if routes else 0.0,
        }
        return routes

    def _generate_level(self, edges, node_by_id, level_name, require_saddle):
        best_by_sequence = {}
        considered = 0
        topology_rejected = 0
        manifold_rejected = 0
        manifold_rejections = []
        truncated = False
        saddle_debug = self._init_saddle_route_debug(edges, node_by_id)
        for cost, node_ids, edge_parts in self._enumerate_morse_paths(edges):
            considered += 1
            if considered > self.max_paths:
                truncated = True
                break
            status = self.check_candidate_topology(
                node_ids, node_by_id, require_saddle=require_saddle)
            if not status["topology_valid"]:
                topology_rejected += 1
                self._record_saddle_path_debug(
                    saddle_debug, node_ids, node_by_id, status,
                    rejected=True, reject_reason=status.get(
                        "reason", "topology_rejected"))
                continue
            self._record_saddle_path_debug(
                saddle_debug, node_ids, node_by_id, status)
            cells = self._merge_cells(edge_parts)
            centerline = self.build_dense_route_trajectory(
                node_ids, node_by_id, cells)
            centerline = self._smooth_trajectory(centerline)
            validation = self.validate_route(centerline)
            if not bool(validation.get("route_valid", False)):
                topology_rejected += 1
                self._record_saddle_path_debug(
                    saddle_debug, node_ids, node_by_id, status,
                    rejected=True,
                    reject_reason="route_validation_failed")
                continue
            boundary = self._adaptive_corridor(centerline)
            speed_profile = self._adaptive_speed_profile(centerline)
            route_eval_seed = {
                "candidate_id": "",
                "node_sequence": list(node_ids),
            }
            route_eval = self._evaluate_route(route_eval_seed, centerline, boundary)
            if not bool(route_eval.get("route_valid", False)):
                topology_rejected += 1
                manifold_rejected += 1
                rejection = {
                    "node_sequence": list(node_ids),
                    "critical_point_sequence": list(
                        status.get("critical_point_sequence", []) or []),
                    "failure_reason": list(
                        route_eval.get("failure_reason", []) or []),
                    "min_clearance": float(
                        route_eval.get("min_clearance", 0.0)),
                    "max_risk": float(route_eval.get("max_risk", 0.0)),
                }
                manifold_rejections.append(rejection)
                self._record_saddle_path_debug(
                    saddle_debug, node_ids, node_by_id, status,
                    rejected=True,
                    reject_reason=",".join(rejection["failure_reason"]) or
                    "manifold_infeasible")
                continue
            turning_cost, max_route_turn = _turning_cost(centerline)
            sequence = tuple(status["critical_point_sequence"])
            previous = best_by_sequence.get(sequence)
            route = {
                "candidate_id": "",
                "topology_source": "morse_graph",
                "generation_method": "morse_topology_induced",
                "route_source": "morse_topology",
                "route_generation_level": level_name,
                "candidate_generation_role": "morse_graph_topology_path",
                "critical_point_sequence": list(status["critical_point_sequence"]),
                "critical_point_types": list(status["critical_point_types"]),
                "node_sequence": list(node_ids),
                "centerline": centerline.tolist(),
                "waypoints": centerline.tolist(),
                "boundary": boundary,
                "corridor_width_profile": list(boundary.get(
                    "corridor_width_profile", [])),
                "speed_profile": list(speed_profile),
                "cells": cells,
                "base_cost": float(cost + 0.25 * turning_cost),
                "topology_cost": float(cost),
                "turning_cost": float(turning_cost),
                "max_route_turn": float(max_route_turn),
                "manifold_valid": bool(route_eval.get("manifold_valid", False)),
                "manifold_feasible": bool(route_eval.get("route_valid", False)),
                "route_validation": dict(validation),
                "candidate_status": (
                    "feasible" if route_eval.get("route_valid", False)
                    else "manifold_infeasible"),
                "failure_reason": ",".join(route_eval.get("failure_reason", [])),
                "manifold_adaptive": True,
                "risk_adaptive_width": True,
                "adaptive_corridor_width": True,
                "min_clearance": float(route_eval.get("min_clearance", 0.0)),
                "trajectory_min_clearance": float(route_eval.get(
                    "min_clearance", 0.0)),
                "max_risk": float(route_eval.get("max_risk", 0.0)),
                "trajectory_max_risk": float(route_eval.get("max_risk", 0.0)),
                "average_corridor_width": float(np.mean(np.asarray(
                    boundary.get("corridor_width_profile", [0.0]), float))),
                "manifold_validation": dict(route_eval.get("safety_status", {})),
                "morse_route_evaluation": dict(route_eval),
                "topology_valid": True,
                "topology_status": status,
            }
            route["morse_route_evaluation"]["turning_cost"] = float(turning_cost)
            route["morse_route_evaluation"]["max_route_turn"] = float(max_route_turn)
            if self.arm_search_enabled:
                search_cost = self._arm_morse_search_cost(
                    float(cost), route, centerline, route_eval)
                if bool(search_cost.get("upper_arm_collision", False)) or bool(
                        search_cost.get("link_collision", False)):
                    route["candidate_status"] = "recoverable"
                    route["failure_reason"] = "arm_link_collision"
                    route["arm_link_recoverable"] = True
                    self._record_saddle_path_debug(
                        saddle_debug, node_ids, node_by_id, status,
                        rejected=True,
                        reject_reason="arm_link_collision_recoverable")
                route["arm_morse_search_cost"] = dict(search_cost)
                route["base_cost"] = float(search_cost["total_cost"])
                route["topology_cost"] = float(search_cost["topology_cost"])
                route["arm_search_clearance_cost"] = float(
                    search_cost["clearance_cost"])
                route["arm_search_link_cost"] = float(search_cost["link_cost"])
                route["arm_search_risk_cost"] = float(search_cost["risk_cost"])
                if search_cost.get("arm_route_validation"):
                    route["arm_route_validation"] = dict(
                        search_cost["arm_route_validation"])
            if previous is None or float(route["base_cost"]) < float(previous["base_cost"]):
                best_by_sequence[sequence] = route
            if len(best_by_sequence) >= self.max_routes:
                truncated = True
                break
        routes = sorted(
            best_by_sequence.values(),
            key=lambda item: (
                len(item.get("critical_point_sequence", [])),
                float(item.get("base_cost", 0.0))))
        for idx, route in enumerate(routes):
            rid = "morse_topology_candidate_%04d" % (idx + 1)
            route["candidate_id"] = rid
            route["morse_route_evaluation"]["route_id"] = rid
            if route.get("arm_morse_search_cost"):
                route["arm_morse_search_cost"]["route_id"] = rid
            for saddle_id in [
                    str(node_id)
                    for node_id in route.get("node_sequence", [])
                    if (str(node_id) in node_by_id and
                        str(getattr(node_by_id[str(node_id)], "kind", "")) ==
                        "saddle")]:
                rec = saddle_debug.get(saddle_id)
                if rec is None:
                    continue
                rec["generated_routes"].append({
                    "route_id": rid,
                    "critical_point_sequence": list(route.get(
                        "critical_point_sequence", []) or []),
                    "node_sequence": list(route.get("node_sequence", []) or []),
                })
        report = {
            "level": str(level_name),
            "require_saddle": bool(require_saddle),
            "paths_considered": int(considered),
            "topology_rejected": int(topology_rejected),
            "manifold_rejected": int(manifold_rejected),
            "morse_route_rejections": manifold_rejections,
            "routes": int(len(routes)),
            "candidate_generated": int(len(routes)),
            "truncated": bool(truncated),
            "max_paths": int(self.max_paths),
            "max_routes": int(self.max_routes),
            "morse_saddle_route_debug": self._finalize_saddle_route_debug(
                saddle_debug),
        }
        return routes, report

    def check_candidate_topology(self, node_ids, node_by_id,
                                 require_saddle=True):
        critical_ids = []
        critical_types = []
        has_saddle = False
        for node_id in node_ids[1:-1]:
            node = node_by_id.get(str(node_id))
            if node is None:
                continue
            kind = str(getattr(node, "kind", ""))
            if kind not in ("saddle", "minimum"):
                continue
            if kind == "saddle":
                has_saddle = True
            critical_ids.append(str(node_id))
            critical_types.append(kind)
        sequence_valid = bool(critical_ids)
        valid = bool(sequence_valid)
        if require_saddle and not has_saddle:
            valid = False
        reason = ""
        if not critical_ids:
            reason = "missing_critical_sequence"
        elif require_saddle and not has_saddle:
            reason = "missing_saddle_sequence"
        return {
            "topology_valid": bool(valid),
            "critical_point_sequence": critical_ids,
            "critical_point_types": critical_types,
            "same_morse_branch": bool(critical_ids),
            "sequence_valid": bool(sequence_valid),
            "reason": reason,
        }

    def _critical_status_for_path(self, node_ids, node_by_id):
        critical_ids = []
        critical_types = []
        for node_id in node_ids[1:-1]:
            node = node_by_id.get(str(node_id))
            if node is None:
                continue
            kind = str(getattr(node, "kind", ""))
            if kind not in ("saddle", "minimum"):
                continue
            critical_ids.append(str(node_id))
            critical_types.append(kind)
        return critical_ids, critical_types

    def _init_saddle_route_debug(self, edges, node_by_id):
        debug = {}
        for node_id, node in node_by_id.items():
            if str(getattr(node, "kind", "")) != "saddle":
                continue
            connected = []
            for edge in list((edges or {}).get(str(node_id), []) or []):
                target_id = str(edge.get("to", ""))
                target = node_by_id.get(target_id)
                if target is not None and str(getattr(
                        target, "kind", "")) == "minimum":
                    connected.append(target_id)
            debug[str(node_id)] = {
                "saddle_id": str(node_id),
                "connected_minima": sorted(set(connected)),
                "candidate_paths": [],
                "rejected_paths": [],
                "reject_reason": {},
                "generated_routes": [],
            }
        return debug

    def _record_saddle_path_debug(self, saddle_debug, node_ids, node_by_id,
                                  status=None, rejected=False,
                                  reject_reason=""):
        if not saddle_debug:
            return
        saddle_ids = [
            str(node_id)
            for node_id in node_ids[1:-1]
            if (str(node_id) in node_by_id and
                str(getattr(node_by_id[str(node_id)], "kind", "")) ==
                "saddle")
        ]
        if not saddle_ids:
            return
        status = dict(status or {})
        critical_ids = list(status.get("critical_point_sequence", []) or [])
        critical_types = list(status.get("critical_point_types", []) or [])
        if not critical_ids:
            critical_ids, critical_types = self._critical_status_for_path(
                node_ids, node_by_id)
        path_record = {
            "node_sequence": list(node_ids),
            "critical_point_sequence": list(critical_ids),
            "critical_point_types": list(critical_types),
        }
        reason = str(reject_reason or status.get("reason", "") or "")
        if rejected:
            path_record["reject_reason"] = reason
        for saddle_id in saddle_ids:
            rec = saddle_debug.get(saddle_id)
            if rec is None:
                continue
            if rejected:
                if len(rec["rejected_paths"]) < 64:
                    rec["rejected_paths"].append(dict(path_record))
                if reason:
                    rec["reject_reason"][reason] = int(
                        rec["reject_reason"].get(reason, 0)) + 1
            else:
                if len(rec["candidate_paths"]) < 64:
                    rec["candidate_paths"].append(dict(path_record))

    def _finalize_saddle_route_debug(self, saddle_debug):
        out = []
        for saddle_id in sorted(saddle_debug):
            rec = dict(saddle_debug[saddle_id])
            rec["candidate_paths"] = list(rec.get("candidate_paths", []) or [])
            rec["rejected_paths"] = list(rec.get("rejected_paths", []) or [])
            rec["generated_routes"] = list(rec.get("generated_routes", []) or [])
            rec["reject_reason"] = dict(rec.get("reject_reason", {}) or {})
            out.append(rec)
        return out

    def _enumerate_morse_paths(self, edges):
        # Uniform-cost enumeration is essential here: the route budget must
        # contain the globally cheapest Morse paths, not whichever deep paths
        # a depth-first traversal happens to finish first.
        queue = [(0.0, 0, ["start"], [])]
        tie_breaker = 0
        while queue:
            cost, _order, node_ids, parts = heapq.heappop(queue)
            cur = node_ids[-1]
            if cur == "goal":
                yield cost, node_ids, parts
                continue
            outgoing = sorted(
                list((edges or {}).get(cur, []) or []),
                key=lambda edge: (
                    float(edge.get("cost", 0.0)),
                    str(edge.get("to", ""))))
            for edge in outgoing:
                nxt = str(edge.get("to", ""))
                if not nxt or nxt in node_ids:
                    continue
                tie_breaker += 1
                heapq.heappush(queue, (
                    float(cost) + float(edge.get("cost", 0.0)),
                    tie_breaker,
                    node_ids + [nxt],
                    parts + [edge],
                ))

    def _merge_cells(self, edge_parts):
        cells = []
        for part in edge_parts or []:
            part_cells = list(part.get("cells", []) or [])
            if cells and part_cells:
                part_cells = part_cells[1:]
            cells.extend(part_cells)
        return cells

    def _centerline_from_cells(self, cells):
        points = []
        for ij in cells or []:
            if self.world_from_ij is not None:
                points.append(np.asarray(self.world_from_ij(ij), float))
        if not points:
            return np.empty((0, 3), float)
        return np.asarray(points, float)

    def _boundary_from_centerline(self, centerline):
        return generate_manifold_aware_corridor(
            centerline, self.grid, self.field,
            default_radius=self.default_radius)

    def _path_tangent(self, pts, idx):
        return _path_tangent(pts, idx)

    def _safe_width(self, point):
        radius = float(self.default_radius)
        clearance = None
        if self.grid:
            xs = self.grid.get("xs")
            ys = self.grid.get("ys")
            clearance_grid = self.grid.get("clearance")
            if xs is not None and ys is not None and clearance_grid is not None:
                i = int(np.argmin(np.abs(np.asarray(xs, float) - float(point[0]))))
                j = int(np.argmin(np.abs(np.asarray(ys, float) - float(point[1]))))
                clearance = float(clearance_grid[i, j])
        risk_scale = 1.0
        if self.field is not None:
            try:
                risk = float(self.field.phi_s(point))
                risk_scale = 1.0 / (1.0 + max(0.0, risk))
            except Exception:
                risk_scale = 1.0
        if clearance is None or not bool(np.isfinite(clearance)):
            return float(2.0 * radius * risk_scale)
        width = min(2.0 * radius, 1.8 * max(0.0, clearance))
        width = max(0.25 * radius, width * risk_scale)
        return float(width)
