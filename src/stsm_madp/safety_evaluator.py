import sys
sys.dont_write_bytecode = True

import hashlib
import json

import numpy as np

from stsm_madp.manifold_constraint import (
    distance_to_manifold_boundary,
    manifold_risk_value,
)


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


def build_safety_context(social_field=None, manifold_constraint=None,
                         task_context=None, source="", strict=False):
    """Build the explicit, serializable identity for hard social safety."""
    payload = dict(manifold_constraint or {})
    context = dict(task_context if task_context is not None else
                   getattr(social_field, "task_context", {}) or {})
    humans = list(getattr(social_field, "humans", []) or [])
    anchors = list(getattr(social_field, "anchors", []) or [])
    weights = (dict(social_field.get_effective_weights())
               if social_field is not None and
               hasattr(social_field, "get_effective_weights") else {})
    identity = {
        "task_state": str(context.get("task_state", "")),
        "humans": [{"position": np.asarray(h.pos, float).round(9).tolist(),
                    "heading": None if getattr(h, "heading", None) is None
                    else round(float(h.heading), 9)} for h in humans],
        "anchors": [{"type": str(a.type),
                     "center": np.asarray(a.center, float).round(9).tolist(),
                     "half_extent": np.asarray(a.half_extent, float).round(9).tolist(),
                     "weight": round(float(a.weight), 9),
                     "forbidden": bool(a.forbidden)} for a in anchors],
        "weights": weights,
        "rho": payload.get("rho", payload.get("risk_threshold", None)),
        "risk_threshold": payload.get("effective_risk_threshold",
                                      payload.get("risk_threshold", None)),
        "clearance_threshold": payload.get("effective_minimum_clearance",
                                            payload.get("minimum_clearance", None)),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return {
        "social_field": social_field,
        "manifold_constraint": payload,
        "task_context": context,
        "source": str(source),
        "strict": bool(strict),
        "fingerprint": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    }


def safety_context_audit(point, manifold_constraint=None, corridor_constraint=None,
                         risk_field=None, stage="", task_context_source=""):
    """Return a JSON-stable, point-level snapshot of one safety context."""
    payload = dict(manifold_constraint or {})
    evaluator = SafetyEvaluator(
        manifold_constraint=payload,
        corridor_constraint=corridor_constraint,
        risk_field=risk_field)
    p = _as_points(point)[0]
    task_context = dict(getattr(risk_field, "task_context", {}) or {})
    humans = list(getattr(risk_field, "humans", []) or [])
    anchors = list(getattr(risk_field, "anchors", []) or [])
    weights = (dict(risk_field.get_effective_weights())
               if risk_field is not None and
               hasattr(risk_field, "get_effective_weights") else {})
    phi = dict(phi_prox=0.0, phi_close=0.0, phi_dir=0.0,
               phi_body=0.0, phi_env=0.0, phi_total=0.0)
    if risk_field is not None:
        for human in humans:
            if hasattr(risk_field, "phi_prox"):
                phi["phi_prox"] += float(risk_field.phi_prox(p, human))
            if hasattr(risk_field, "phi_close"):
                phi["phi_close"] += float(risk_field.phi_close(
                    p, np.zeros_like(p), human))
            if hasattr(risk_field, "phi_dir"):
                phi["phi_dir"] += float(risk_field.phi_dir(p, human))
            if hasattr(risk_field, "phi_body"):
                phi["phi_body"] += float(risk_field.phi_body(p, human))
        if hasattr(risk_field, "phi_env"):
            phi["phi_env"] = float(risk_field.phi_env(p))
        if hasattr(risk_field, "phi_s"):
            phi["phi_total"] = float(risk_field.phi_s(p))
    status = evaluator.evaluate_state(p)
    fingerprint_payload = {
        "task_state": str(task_context.get("task_state", "")),
        "humans": [{
            "position": np.asarray(h.pos, float).round(9).tolist(),
            "heading": None if getattr(h, "heading", None) is None else
            round(float(h.heading), 9),
        } for h in humans],
        "anchors": [{
            "type": str(a.type), "center": np.asarray(a.center, float).round(9).tolist(),
            "half_extent": np.asarray(a.half_extent, float).round(9).tolist(),
            "weight": round(float(a.weight), 9), "forbidden": bool(a.forbidden),
        } for a in anchors],
        "weights": weights,
        "rho": payload.get("rho", payload.get(
            "risk_threshold", getattr(risk_field, "rho", None))),
        "risk_threshold": evaluator.risk_threshold,
        "clearance_threshold": evaluator.required_clearance,
    }
    fingerprint_json = json.dumps(
        fingerprint_payload, sort_keys=True, separators=(",", ":"))
    audit = {
        "stage": str(stage), "x": float(p[0]), "y": float(p[1]),
        "task_state": str(task_context.get("task_state", "")),
        "task_context_source": str(task_context_source),
        "human_count": int(len(humans)),
        "human_positions": [np.asarray(h.pos, float).tolist() for h in humans],
        "human_headings": [getattr(h, "heading", None) for h in humans],
        "anchor_count": int(len(anchors)),
        "anchor_types": [str(a.type) for a in anchors],
        "social_field_id": (None if risk_field is None else
                            risk_field.__class__.__name__),
        "manifold_id": str(payload.get("type", "")),
        "rho": fingerprint_payload["rho"],
        "risk_threshold": float(evaluator.risk_threshold),
        "clearance_threshold": float(evaluator.required_clearance),
        "social_field_weights": weights,
        "safety_context_fingerprint": hashlib.sha256(
            fingerprint_json.encode("utf-8")).hexdigest(),
        "clearance": float(status["clearance"]),
        "risk": float(status["risk"]),
        "manifold_valid": bool(status["inside_manifold"]),
        "hard_valid": bool(status["inside_manifold"] and
                           status["inside_corridor"]),
    }
    audit.update(phi)
    return audit


def _polyline_project(point, centerline):
    pts = _as_points(centerline)
    p = np.asarray(point, float)[:3]
    if len(pts) == 0:
        return p, float("inf"), 0.0
    dim = min(p.size, pts.shape[1])
    p = p[:dim]
    wps = pts[:, :dim]
    if len(wps) == 1:
        return wps[0], float(np.linalg.norm(p - wps[0])), 0.0
    best_q = wps[0]
    best_d = float("inf")
    best_s = 0.0
    acc = 0.0
    for a, b in zip(wps[:-1], wps[1:]):
        ab = b - a
        seg_len = float(np.linalg.norm(ab))
        denom = float(np.dot(ab, ab))
        t = 0.0 if denom <= 1e-12 else np.clip(
            float(np.dot(p - a, ab)) / denom, 0.0, 1.0)
        q = a + t * ab
        d = float(np.linalg.norm(p - q))
        if d < best_d:
            best_q = q
            best_d = d
            best_s = acc + t * seg_len
        acc += seg_len
    return best_q, best_d, best_s


def _constraint_value(payload, key, default=None):
    if isinstance(payload, dict):
        return payload.get(key, default)
    return getattr(payload, key, default)


def _extend_points(out, value):
    pts = _as_points(value)
    if len(pts):
        out.extend(pts.tolist())


class SafetyEvaluator(object):
    """Shared Candidate/Refinement/MPC safety evaluator."""

    def __init__(self, manifold_constraint=None, corridor_constraint=None,
                 risk_field=None, planning_clearance_margin=0.0):
        self.manifold_constraint = dict(manifold_constraint or {})
        self.corridor_constraint = dict(corridor_constraint or {})
        self.risk_field = risk_field
        self.planning_clearance_margin = float(
            planning_clearance_margin or
            self.manifold_constraint.get("planning_clearance_margin", 0.0) or 0.0)

    @property
    def minimum_clearance(self):
        value = self.manifold_constraint.get(
            "effective_minimum_clearance",
            self.manifold_constraint.get(
                "effective_min_clearance",
                self.manifold_constraint.get(
                    "minimum_clearance",
                    self.manifold_constraint.get("min_clearance", 0.0))))
        return float(value or 0.0)

    @property
    def nominal_minimum_clearance(self):
        value = self.manifold_constraint.get(
            "minimum_clearance",
            self.manifold_constraint.get("min_clearance", self.minimum_clearance))
        return float(value or 0.0)

    @property
    def required_clearance(self):
        return float(self.minimum_clearance + max(
            0.0, self.planning_clearance_margin))

    @property
    def risk_threshold(self):
        value = self.manifold_constraint.get(
            "effective_risk_threshold",
            self.manifold_constraint.get(
                "effective_safe_threshold",
                self.manifold_constraint.get(
                    "risk_threshold",
                    self.manifold_constraint.get("safe_threshold", None))))
        if value in (None, ""):
            value = getattr(self.risk_field, "rho", 1.0)
        return float(value)

    def _boundary(self):
        return self.manifold_constraint.get("boundary", [])

    def _centerline(self):
        return _as_points(
            self.corridor_constraint.get(
                "centerline",
                self.corridor_constraint.get("corridor_centerline", [])))

    def _corridor_radius(self):
        radius = self.corridor_constraint.get(
            "radius",
            self.corridor_constraint.get(
                "tube_width",
                self.corridor_constraint.get("corridor_radius", 0.0)))
        return float(radius or 0.0)

    def _corridor_distance(self, state):
        centerline = self._centerline()
        radius = self._corridor_radius()
        if len(centerline) == 0 or radius <= 0.0:
            return 0.0, True
        _q, dist, _s = _polyline_project(state, centerline)
        return float(dist), bool(dist <= radius + 1e-9)

    def evaluate_state(self, state, risk=None):
        point = np.asarray(state, float)[:3]
        clearance = distance_to_manifold_boundary(point, self._boundary())
        risk = (manifold_risk_value(point, self.risk_field)
                if risk is None else float(risk))
        clearance_source = "risk_manifold_boundary"
        if not np.isfinite(clearance) and self.risk_field is not None and hasattr(
                self.risk_field, "grad_phi_s"):
            try:
                grad = np.asarray(self.risk_field.grad_phi_s(point), float)[:3]
                grad_norm = float(np.linalg.norm(grad))
                if grad_norm > 1e-9:
                    clearance = float((self.risk_threshold - risk) / grad_norm)
                    clearance_source = "risk_gradient_distance"
            except Exception:
                pass
        clearance_available = bool(np.isfinite(clearance))
        if not clearance_available:
            clearance_source = "risk_threshold_only"
        corridor_distance, inside_corridor = self._corridor_distance(point)
        inside_manifold = bool(
            (not clearance_available or
             clearance + 1e-9 >= self.required_clearance) and
            risk <= self.risk_threshold + 1e-9)
        return {
            "clearance": float(clearance),
            "clearance_available": bool(clearance_available),
            "clearance_source": clearance_source,
            "risk": float(risk),
            "inside_manifold": bool(inside_manifold),
            "inside_corridor": bool(inside_corridor),
            "corridor_distance": float(corridor_distance),
            "corridor_radius": float(self._corridor_radius()),
            "minimum_clearance": float(self.minimum_clearance),
            "nominal_minimum_clearance": float(self.nominal_minimum_clearance),
            "effective_minimum_clearance": float(self.minimum_clearance),
            "required_clearance": float(self.required_clearance),
            "risk_threshold": float(self.risk_threshold),
        }

    def evaluate_states(self, states):
        """Batch only the risk query; retain the existing per-state checks."""
        points = _as_points(states)
        if len(points) == 0:
            return []
        if self.risk_field is not None and hasattr(self.risk_field, "phi_s_batch"):
            risks = np.asarray(self.risk_field.phi_s_batch(points), float)
            return [self.evaluate_state(point, risk=risk)
                    for point, risk in zip(points, risks)]
        return [self.evaluate_state(point) for point in points]

    def evaluate_state_or_points(self, state=None, interest_points=None,
                                 robot_type="", task_phase="",
                                 phase_progress=0.0,
                                 effective_minimum_clearance=None,
                                 paper_mode=False):
        points = _as_points(interest_points)
        clearance_source = (
            "risk_manifold_boundary" if bool(self._boundary()) else
            "risk_gradient_distance" if (
                self.risk_field is not None and
                hasattr(self.risk_field, "grad_phi_s")) else
            "risk_threshold_only")
        if len(points) == 0:
            if bool(paper_mode):
                required = (
                    self.minimum_clearance if effective_minimum_clearance is None
                    else float(effective_minimum_clearance or 0.0))
                return {
                    "valid": False,
                    "min_clearance": 0.0,
                    "clearance": 0.0,
                    "risk": 0.0,
                    "min_clearance_point": "",
                    "effective_minimum_clearance": float(required),
                    "clearance_margin": -float(required),
                    "clearance_violation": bool(required > 0.0),
                    "clearance_source": "invalid",
                    "interest_point_count": 0,
                    "worst_interest_point": "",
                    "robot_type": str(robot_type or ""),
                    "task_phase": str(task_phase or ""),
                    "phase_progress": float(phase_progress or 0.0),
                }
            points = _as_points(state)
            if len(points) == 0:
                points = np.zeros((1, 3), float)
            clearance_source = "fallback_state_point"
        required = (
            self.minimum_clearance if effective_minimum_clearance is None
            else float(effective_minimum_clearance or 0.0))
        worst = None
        worst_status = None
        for idx, point in enumerate(points):
            status = self.evaluate_state(point)
            clearance = float(status.get("clearance", 0.0))
            if worst_status is None or clearance < float(
                    worst_status.get("clearance", 0.0)):
                worst = idx
                worst_status = status
        worst_status = worst_status or self.evaluate_state(points[0])
        min_clearance = float(worst_status.get("clearance", 0.0))
        margin = float(min_clearance - required)
        risk = float(worst_status.get("risk", 0.0))
        valid = bool(
            margin >= -1e-9 and
            risk <= self.risk_threshold + 1e-9 and
            bool(worst_status.get("inside_corridor", True)))
        label = "point_{}".format(int(worst or 0))
        return {
            "valid": bool(valid),
            "min_clearance": min_clearance,
            "clearance": min_clearance,
            "risk": risk,
            "min_clearance_point": label,
            "effective_minimum_clearance": float(required),
            "clearance_margin": margin,
            "clearance_violation": bool(margin < -1e-9),
            "clearance_source": clearance_source,
            "interest_point_count": int(len(points)),
            "worst_interest_point": label,
            "robot_type": str(robot_type or ""),
            "task_phase": str(task_phase or ""),
            "phase_progress": float(phase_progress or 0.0),
        }

    def evaluate_trajectory(self, traj, require_social_context=False):
        pts = _as_points(traj)
        if require_social_context and self.risk_field is None:
            return {
                "min_clearance": 0.0, "max_risk": 0.0,
                "manifold_violation_count": int(len(pts)),
                "corridor_violation_count": 0, "valid": False,
                "minimum_clearance": float(self.minimum_clearance),
                "nominal_minimum_clearance": float(self.nominal_minimum_clearance),
                "effective_minimum_clearance": float(self.minimum_clearance),
                "required_clearance": float(self.required_clearance),
                "planning_clearance_margin": float(self.planning_clearance_margin),
                "risk_threshold": float(self.risk_threshold),
                "failure_reason": "missing_safety_context",
            }
        min_clearance = float("inf")
        max_risk = 0.0
        manifold_violation_count = 0
        corridor_violation_count = 0
        for point in pts:
            state = self.evaluate_state(point)
            min_clearance = min(min_clearance, float(state["clearance"]))
            max_risk = max(max_risk, float(state["risk"]))
            if not bool(state["inside_manifold"]):
                manifold_violation_count += 1
            if not bool(state["inside_corridor"]):
                corridor_violation_count += 1
        if not np.isfinite(min_clearance):
            min_clearance = 0.0
        valid = bool(
            len(pts) > 0 and
            manifold_violation_count == 0 and
            corridor_violation_count == 0)
        return {
            "min_clearance": float(min_clearance),
            "max_risk": float(max_risk),
            "manifold_violation_count": int(manifold_violation_count),
            "corridor_violation_count": int(corridor_violation_count),
            "valid": bool(valid),
            "minimum_clearance": float(self.minimum_clearance),
            "nominal_minimum_clearance": float(self.nominal_minimum_clearance),
            "effective_minimum_clearance": float(self.minimum_clearance),
            "required_clearance": float(self.required_clearance),
            "planning_clearance_margin": float(self.planning_clearance_margin),
            "risk_threshold": float(self.risk_threshold),
        }

    def evaluate_corridor(self, candidate):
        samples = []
        _extend_points(samples, _constraint_value(candidate, "centerline", []))
        _extend_points(samples, _constraint_value(candidate, "waypoints", []))
        if not samples:
            valid_region = _constraint_value(candidate, "valid_region", {}) or {}
            if isinstance(valid_region, dict):
                _extend_points(samples, valid_region.get("centerline", []))
        status = self.evaluate_trajectory(samples)
        return {
            "tube_valid": bool(status["valid"]),
            "min_tube_clearance": float(status["min_clearance"]),
            "max_tube_risk": float(status["max_risk"]),
            "tube_manifold_violation_count": int(
                status["manifold_violation_count"]),
            "tube_corridor_violation_count": int(
                status["corridor_violation_count"]),
        }
