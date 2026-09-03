import sys
sys.dont_write_bytecode = True

import numpy as np


PHASE_MANIFOLD_DEFAULTS = {
    "arm": {
        "approach": {
            "safety_distance_priority": 1.0,
            "task_completion_priority": 0.6,
            "interaction_region_allowed": False,
            "start_clearance": 0.35,
            "end_clearance": 0.20,
        },
        "handover": {
            "safety_distance_priority": 0.45,
            "task_completion_priority": 1.0,
            "interaction_region_allowed": True,
            "clearance": 0.15,
        },
        "return": {
            "safety_distance_priority": 1.0,
            "task_completion_priority": 0.5,
            "interaction_region_allowed": False,
            "start_clearance": 0.20,
            "end_clearance": 0.35,
        },
    },
    "wheelchair": {
        "navigation": {
            "safety_distance_priority": 1.0,
            "passage_width_priority": 1.0,
            "task_completion_priority": 0.7,
            "interaction_region_allowed": False,
        },
    },
}


def _phase_name(value, robot_type="generic"):
    phase = str(value if value not in (None, "") else "").strip().lower()
    aliases = {
        "0": "approach",
        "1": "approach",
        "2": "approach",
        "3": "handover",
        "4": "return",
        "phase0": "approach",
        "phase1": "approach",
        "phase2": "approach",
        "phase3": "handover",
        "phase4": "return",
    }
    phase = aliases.get(phase, phase)
    robot = str(robot_type or "generic").strip().lower()
    if not phase:
        phase = "navigation" if robot == "wheelchair" else "approach"
    if robot == "wheelchair" and phase not in PHASE_MANIFOLD_DEFAULTS["wheelchair"]:
        phase = "navigation"
    return phase


def phase_manifold_parameters(phase=None, robot_type="generic",
                              phase_params=None):
    robot = str(robot_type or "generic").strip().lower()
    if robot in ("manipulator", "handover", "mechanical_arm"):
        robot = "arm"
    if robot not in PHASE_MANIFOLD_DEFAULTS:
        robot = "wheelchair" if _phase_name(phase, robot) == "navigation" else "arm"
    name = _phase_name(phase, robot)
    defaults = dict(PHASE_MANIFOLD_DEFAULTS.get(robot, {}).get(name, {}) or {})
    overrides = dict((phase_params or {}).get(name, {}) or {})
    defaults.update(overrides)
    return name, robot, defaults


def build_phase_clearance_schedule(robot_type, phase_params=None):
    robot = str(robot_type or "generic").strip().lower()
    if robot in ("manipulator", "handover", "mechanical_arm"):
        robot = "arm"
    if phase_params:
        return dict(phase_params)
    if robot == "arm":
        return {
            "approach": {"start_clearance": 0.35, "end_clearance": 0.20},
            "handover": {"clearance": 0.15},
            "return": {"start_clearance": 0.20, "end_clearance": 0.35},
        }
    if robot == "wheelchair":
        return {"navigation": {}}
    return {}


def assert_manifold_mode_consistency(*modes):
    values = set()
    for mode in modes:
        if mode in (None, ""):
            continue
        value = str(mode).strip().lower()
        if value:
            values.add(value)
    invalid = [value for value in values if value not in ("soft", "hard")]
    if invalid:
        raise ValueError("invalid manifold mode: %s" % sorted(invalid))
    if len(values) > 1:
        raise ValueError("inconsistent manifold modes: %s" % sorted(values))
    return next(iter(values)) if values else "soft"


def effective_phase_manifold_thresholds(minimum_clearance, risk_threshold,
                                        phase_parameters):
    params = dict(phase_parameters or {})
    safety_priority = float(params.get("safety_distance_priority", 1.0))
    safety_priority = max(0.20, min(1.50, safety_priority))
    clearance = float(minimum_clearance or 0.0)
    risk = float(risk_threshold)
    if "start_clearance" in params or "end_clearance" in params:
        effective_clearance = float(params.get(
            "start_clearance", params.get("end_clearance", clearance)))
    elif "clearance" in params:
        effective_clearance = float(params.get("clearance", clearance))
    else:
        effective_clearance = clearance * safety_priority
    return float(effective_clearance), float(risk)


def evaluate_dynamic_state_constraint(state, reference_heading=None,
                                      limits=None):
    """Evaluate the kinodynamic part of the safety manifold for one state.

    ``state`` is ``[x, y, theta, v]`` (extra fields are ignored).  The
    function is pure and takes all limits explicitly so planning and runtime
    callers cannot silently use different defaults.
    """
    values = np.asarray(state, float).reshape(-1)
    limits = dict(limits or {})
    if values.size < 4 or not np.all(np.isfinite(values[:4])):
        return {"valid": False, "reason": "state_invalid"}
    heading_error = 0.0
    if reference_heading is not None:
        heading_error = abs(float((values[2] - float(reference_heading) + np.pi) %
                                  (2.0 * np.pi) - np.pi))
    speed = float(values[3])
    checks = {
        "heading_error": (heading_error, float(limits.get("heading_max", np.inf))),
        "speed": (abs(speed), float(limits.get("speed_max", np.inf))),
    }
    return {
        "valid": bool(all(v <= lim + 1e-9 for v, lim in checks.values())),
        "heading_error": heading_error,
        "speed": speed,
        "checks": {k: {"value": v, "limit": lim,
                        "valid": bool(v <= lim + 1e-9)}
                    for k, (v, lim) in checks.items()},
    }


class ManifoldConstraint(dict):
    """MPC-facing safety manifold constraint payload."""

    def __init__(self, safe_region=None, boundary_distance=None,
                 minimum_clearance=0.0, risk_threshold=1.0,
                 constraint_type="safe_manifold", boundary=None,
                 mode=None, payload=None, phase=None,
                 robot_type="generic", phase_params=None):
        data = dict(payload or {})
        phase_name, robot, phase_cfg = phase_manifold_parameters(
            data.get("phase", phase), data.get("robot_type", robot_type),
            phase_params=phase_params)
        resolved_mode = assert_manifold_mode_consistency(
            data.get("manifold_constraint_mode"),
            data.get("mode"),
            mode if mode not in (None, "") else None)
        data.setdefault("type", constraint_type)
        data.setdefault("constraint_type", constraint_type)
        data.setdefault("constraint_used", True)
        data.setdefault("used", True)
        data.setdefault("phase_aware", True)
        data.setdefault("phase", phase_name)
        data.setdefault("task_phase", phase_name)
        data.setdefault("robot_type", robot)
        data.setdefault("phase_parameters", dict(phase_cfg))
        data.setdefault("phase_manifold_weights", dict(phase_cfg))
        data.setdefault(
            "phase_clearance_schedule",
            build_phase_clearance_schedule(robot, phase_params))
        data.setdefault("safety_distance_priority", float(
            phase_cfg.get("safety_distance_priority", 1.0)))
        data.setdefault("task_completion_priority", float(
            phase_cfg.get("task_completion_priority", 0.0)))
        data.setdefault("interaction_region_allowed", bool(
            phase_cfg.get("interaction_region_allowed", False)))
        data.setdefault("safe_region", safe_region or [])
        data.setdefault("boundary_distance", boundary_distance)
        data.setdefault("minimum_clearance", float(minimum_clearance or 0.0))
        data.setdefault("min_clearance", float(minimum_clearance or 0.0))
        data.setdefault("risk_threshold", float(risk_threshold))
        data.setdefault("safe_threshold", float(risk_threshold))
        effective_clearance, effective_risk = (
            effective_phase_manifold_thresholds(
                data.get("minimum_clearance", minimum_clearance or 0.0),
                data.get("risk_threshold", risk_threshold),
                phase_cfg))
        data.setdefault("effective_minimum_clearance", float(effective_clearance))
        data.setdefault("effective_min_clearance", float(effective_clearance))
        data.setdefault("effective_risk_threshold", float(effective_risk))
        data.setdefault("effective_safe_threshold", float(effective_risk))
        data.setdefault("boundary", boundary or [])
        data["mode"] = resolved_mode
        data["manifold_constraint_mode"] = resolved_mode
        data.setdefault(
            "distance_rule",
            "distance_to_boundary >= effective_minimum_clearance")
        data.setdefault("risk_rule", "risk_value <= effective_risk_threshold")
        super(ManifoldConstraint, self).__init__(data)

    def to_dict(self):
        return dict(self)


def _as_points(points):
    if points is None or isinstance(points, str):
        return []
    try:
        arr = np.asarray(points, float)
    except Exception:
        return []
    if arr.size == 0:
        return []
    if arr.ndim == 1:
        arr = arr.reshape((1, arr.shape[0]))
    if arr.shape[1] == 2:
        arr = np.hstack([arr, np.zeros((arr.shape[0], 1), float)])
    return arr[:, :3].tolist()


def _boundary_points(boundary):
    if boundary is None or isinstance(boundary, str):
        return np.zeros((0, 3), float)
    if isinstance(boundary, dict):
        pts = []
        for key in ("left", "right", "boundary", "points"):
            pts.extend(boundary.get(key, []) or [])
        boundary = pts
    try:
        pts = np.asarray(boundary, float)
    except Exception:
        return np.zeros((0, 3), float)
    if pts.size == 0:
        return np.zeros((0, 3), float)
    if pts.ndim == 1:
        pts = pts.reshape((1, pts.shape[0]))
    if pts.shape[1] == 2:
        pts = np.hstack([pts, np.zeros((pts.shape[0], 1), float)])
    return pts[:, :3]


def distance_to_manifold_boundary(point, boundary):
    """Distance evaluator used by Candidate, Refinement, and MPC checks."""
    if isinstance(boundary, dict):
        distances = []
        for key in ("left", "right", "boundary", "points"):
            pts = boundary.get(key, []) or []
            if pts:
                distances.append(distance_to_manifold_boundary(point, pts))
        return float(min(distances)) if distances else float("inf")
    pts = _boundary_points(boundary)
    if len(pts) == 0:
        return float("inf")
    p = np.asarray(point, float)[:3]
    dim = min(p.size, pts.shape[1])
    p = p[:dim]
    wps = pts[:, :dim]
    if len(wps) == 1:
        return float(np.linalg.norm(p - wps[0]))
    starts = wps[:-1]
    segments = wps[1:] - starts
    denom = np.einsum("ij,ij->i", segments, segments)
    projection = np.zeros(len(segments), float)
    nondegenerate = denom > 1e-12
    if np.any(nondegenerate):
        offsets = p - starts[nondegenerate]
        projection[nondegenerate] = (
            np.einsum(
                "ij,ij->i", offsets, segments[nondegenerate]) /
            denom[nondegenerate])
    projection = np.clip(projection, 0.0, 1.0)
    closest = starts + projection[:, None] * segments
    return float(np.min(np.linalg.norm(closest - p, axis=1)))


def manifold_risk_value(point, risk_field=None):
    if risk_field is None:
        return 0.0
    try:
        if hasattr(risk_field, "phi_s"):
            return float(risk_field.phi_s(point))
        if callable(risk_field):
            return float(risk_field(point))
    except Exception:
        return 0.0
    return 0.0


def evaluate_trajectory_manifold_feasibility(
        trajectory, manifold_constraint=None, risk_field=None,
        planning_clearance_margin=0.0):
    payload = dict(manifold_constraint or {})
    boundary = payload.get("boundary", [])
    risk_threshold = payload.get(
        "risk_threshold", payload.get("safe_threshold", None))
    if risk_threshold in (None, ""):
        risk_threshold = getattr(risk_field, "rho", 1.0) if risk_field is not None else 1.0
    minimum_clearance = payload.get(
        "effective_minimum_clearance",
        payload.get(
            "effective_min_clearance",
            payload.get("minimum_clearance", payload.get("min_clearance", 0.0))))
    required_clearance = (
        float(minimum_clearance or 0.0) +
        max(0.0, float(planning_clearance_margin or 0.0)))
    points = _boundary_points(trajectory)
    min_clearance = float("inf")
    max_risk = 0.0
    violation_count = 0
    clearance_violation_count = 0
    risk_violation_count = 0
    for point in points:
        clearance = distance_to_manifold_boundary(point, boundary)
        risk = manifold_risk_value(point, risk_field)
        if not np.isfinite(clearance):
            clearance = max(0.0, float(risk_threshold) - risk)
        min_clearance = min(float(min_clearance), float(clearance))
        max_risk = max(float(max_risk), float(risk))
        clearance_bad = clearance + 1e-9 < required_clearance
        risk_bad = risk > float(risk_threshold) + 1e-9
        if clearance_bad:
            clearance_violation_count += 1
        if risk_bad:
            risk_violation_count += 1
        if clearance_bad or risk_bad:
            violation_count += 1
    if not np.isfinite(min_clearance):
        min_clearance = 0.0
    feasible = bool(
        len(points) > 0 and
        violation_count == 0 and
        min_clearance + 1e-9 >= required_clearance and
        max_risk <= float(risk_threshold) + 1e-9)
    failure_reason = ""
    if not feasible:
        if clearance_violation_count:
            failure_reason = "clearance_violation"
        elif risk_violation_count:
            failure_reason = "risk_violation"
        else:
            failure_reason = "manifold_violation"
    return {
        "feasible": feasible,
        "min_clearance": float(min_clearance),
        "max_risk": float(max_risk),
        "violation_count": int(violation_count),
        "clearance_violation_count": int(clearance_violation_count),
        "risk_violation_count": int(risk_violation_count),
        "minimum_clearance": float(minimum_clearance or 0.0),
        "required_clearance": float(required_clearance),
        "planning_clearance_margin": float(planning_clearance_margin or 0.0),
        "risk_threshold": float(risk_threshold),
        "failure_reason": failure_reason,
    }


def _boundary_from_candidate(candidate):
    if candidate is None:
        return []
    boundary = candidate.get("boundary", {}) if isinstance(candidate, dict) else getattr(candidate, "boundary", {})
    if isinstance(boundary, dict):
        left = _as_points(boundary.get("left", []))
        right = _as_points(boundary.get("right", []))
        if left or right:
            return {"left": left, "right": right}
    return _as_points(boundary)


def _safe_region_from_manifold(safe_manifold):
    grid = getattr(safe_manifold, "last_safe_grid", None)
    if isinstance(safe_manifold, dict):
        grid = safe_manifold
    if not isinstance(grid, dict):
        return []
    safe = grid.get("safe")
    xs = grid.get("xs")
    ys = grid.get("ys")
    if safe is None or xs is None or ys is None:
        return []
    try:
        safe_arr = np.asarray(safe, bool)
        xs_arr = np.asarray(xs, float)
        ys_arr = np.asarray(ys, float)
    except Exception:
        return []
    points = []
    for i, j in np.argwhere(safe_arr):
        if len(points) >= 500:
            break
        points.append([float(xs_arr[int(i)]), float(ys_arr[int(j)]), 0.0])
    return points


def _min_clearance_from_sources(safe_manifold=None, footprint=None,
                                selected_corridor=None, default=0.0):
    values = []
    for key in ("minimum_clearance",):
        if selected_corridor is not None:
            value = (
                selected_corridor.get(key)
                if isinstance(selected_corridor, dict) else
                getattr(selected_corridor, key, None))
            if value not in (None, ""):
                values.append(value)
    summary = getattr(safe_manifold, "last_safe_manifold_summary", None)
    if isinstance(summary, dict):
        for key in ("target_clearance", "hard_clearance", "min_clearance"):
            if summary.get(key) not in (None, ""):
                values.append(summary.get(key))
    if isinstance(safe_manifold, dict):
        for key in ("target_clearance", "minimum_clearance", "min_clearance", "hard_clearance"):
            if safe_manifold.get(key) not in (None, ""):
                values.append(safe_manifold.get(key))
    if footprint not in (None, ""):
        try:
            values.append(float(footprint))
        except Exception:
            pass
    for value in values:
        try:
            value = float(value)
        except Exception:
            continue
        if bool(np.isfinite(value)) and value >= 0.0:
            return float(value)
    return float(default)


def build_manifold_constraint(safe_manifold=None, risk_field=None,
                              footprint=None, selected_corridor=None,
                              boundary=None, minimum_clearance=None,
                              risk_threshold=None, constraint_type="safe_manifold",
                              mode=None, phase=None, robot_type="generic",
                              phase_params=None):
    if risk_threshold in (None, ""):
        risk_threshold = getattr(safe_manifold, "rho", None)
    if risk_threshold in (None, "") and isinstance(safe_manifold, dict):
        risk_threshold = safe_manifold.get("risk_threshold", safe_manifold.get("safe_threshold", None))
    if risk_threshold in (None, ""):
        risk_threshold = 1.0
    min_clearance = (
        float(minimum_clearance)
        if minimum_clearance not in (None, "") else
        _min_clearance_from_sources(
            safe_manifold=safe_manifold,
            footprint=footprint,
            selected_corridor=selected_corridor,
            default=0.0))
    out_boundary = boundary if boundary not in (None, "", [], {}) else []
    if not out_boundary and isinstance(safe_manifold, dict):
        out_boundary = safe_manifold.get(
            "risk_manifold_boundary", safe_manifold.get("boundary", []))
    return ManifoldConstraint(
        safe_region=_safe_region_from_manifold(safe_manifold),
        boundary_distance=None,
        minimum_clearance=min_clearance,
        risk_threshold=float(risk_threshold),
        constraint_type=constraint_type,
        boundary=out_boundary,
        mode=mode,
        phase=phase,
        robot_type=robot_type,
        phase_params=phase_params)
