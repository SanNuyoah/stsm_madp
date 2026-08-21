import sys
sys.dont_write_bytecode = True

import math

import numpy as np

from stsm_madp.manifold_constraint import (
    distance_to_manifold_boundary,
    manifold_risk_value,
)


def _finite_float(value, default=0.0):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


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


def _sample_polyline(points, spacing):
    pts = _as_points(points)
    if len(pts) <= 1:
        return pts.copy()
    spacing = max(1e-6, float(spacing or 0.05))
    samples = [pts[0].copy()]
    for start, goal in zip(pts[:-1], pts[1:]):
        segment = goal - start
        length = float(np.linalg.norm(segment))
        steps = max(1, int(math.ceil(length / spacing)))
        for step in range(1, steps + 1):
            alpha = float(step) / float(steps)
            samples.append(start + alpha * segment)
    return _as_points(samples)


def _route_value(route, key, default=None):
    if isinstance(route, dict):
        return route.get(key, default)
    return getattr(route, key, default)


class ArmTopologyValidator(object):
    """Minimal arm route validator used before candidate recovery.

    The validator evaluates the route centerline as an end-effector path and
    uses a deterministic arm-link proxy for pre-MPC route screening. The proxy
    is intentionally lightweight: it maps each Morse waypoint to shoulder,
    elbow, wrist, and tool-link positions on a static base-to-EE chain, then
    checks swept link segments against the same route boundary/risk interface
    used by the end-effector check.
    """

    def __init__(self, risk_field=None, risk_threshold=6.0,
                 minimum_clearance=0.0, planning_clearance_margin=0.0,
                 sample_spacing=0.05, boundary=None, **_kwargs):
        self.risk_field = risk_field
        self.risk_threshold = _finite_float(risk_threshold, 6.0)
        self.minimum_clearance = max(0.0, _finite_float(
            minimum_clearance, 0.0))
        self.planning_clearance_margin = max(0.0, _finite_float(
            planning_clearance_margin, 0.0))
        self.sample_spacing = max(1e-6, _finite_float(sample_spacing, 0.05))
        self.boundary = boundary
        self.link_sample_spacing = max(1e-6, _finite_float(
            _kwargs.get("link_sample_spacing", sample_spacing), 0.05))
        self.default_link_radius = max(0.0, _finite_float(
            _kwargs.get("link_radius", 0.015), 0.015))
        self.link_proxy_fractions = list(_kwargs.get(
            "link_proxy_fractions", [0.0, 0.45, 0.75, 0.92]) or
            [0.0, 0.45, 0.75, 0.92])
        self.link_proxy_names = list(_kwargs.get(
            "link_proxy_names", ["upper_arm", "forearm", "wrist_link"]) or
            ["upper_arm", "forearm", "wrist_link"])
        self.link_proxy_radii = list(_kwargs.get(
            "link_proxy_radii",
            [0.025, 0.020, self.default_link_radius]) or
            [0.025, 0.020, self.default_link_radius])
        self.link_collision_penalty_scale = max(0.0, _finite_float(
            _kwargs.get("link_collision_penalty_scale", 10.0), 10.0))
        self.clearance_score_scale = max(1e-6, _finite_float(
            _kwargs.get("clearance_score_scale", 1.0), 1.0))

    @property
    def required_clearance(self):
        return float(self.minimum_clearance + self.planning_clearance_margin)

    def _route_id(self, route):
        return str(
            _route_value(route, "route_id", None) or
            _route_value(route, "candidate_id", None) or
            _route_value(route, "corridor_id", None) or "")

    def _route_points(self, route):
        for key in ("centerline", "waypoints", "points", "path"):
            pts = _as_points(_route_value(route, key, None))
            if len(pts):
                return pts
        valid_region = _route_value(route, "valid_region", {}) or {}
        if isinstance(valid_region, dict):
            return _as_points(valid_region.get("centerline", []))
        return np.zeros((0, 3), float)

    def _route_boundary(self, route):
        boundary = _route_value(route, "boundary", None)
        if boundary is None:
            boundary = self.boundary
        return boundary

    def _point_clearance(self, point, boundary):
        clearance = distance_to_manifold_boundary(point, boundary)
        if not np.isfinite(clearance):
            risk = manifold_risk_value(point, self.risk_field)
            clearance = max(0.0, self.risk_threshold - risk)
        return _finite_float(clearance, 0.0)

    def _arm_base_position(self, route, points):
        base = (
            _route_value(route, "arm_base_position", None) or
            _route_value(route, "base_position", None) or
            _route_value(route, "shoulder_position", None))
        pts = _as_points(base)
        if len(pts):
            return pts[0].copy()
        pts = _as_points(points)
        if len(pts):
            return pts[0].copy()
        return np.zeros(3, float)

    def _proxy_joint_positions(self, ee_point, base_point):
        ee = np.asarray(ee_point, float)[:3]
        base = np.asarray(base_point, float)[:3]
        vec = ee - base
        joints = []
        for fraction in self.link_proxy_fractions:
            alpha = min(0.98, max(0.0, _finite_float(fraction, 0.0)))
            joints.append(base + alpha * vec)
        return joints

    def _segment_samples(self, start, goal, spacing):
        start = np.asarray(start, float)[:3]
        goal = np.asarray(goal, float)[:3]
        length = float(np.linalg.norm(goal - start))
        steps = max(1, int(math.ceil(length / max(1e-6, spacing))))
        samples = []
        for idx in range(steps + 1):
            alpha = float(idx) / float(max(steps, 1))
            samples.append(start + alpha * (goal - start))
        return samples

    def _link_radius(self, idx):
        if idx < len(self.link_proxy_radii):
            return max(0.0, _finite_float(self.link_proxy_radii[idx], 0.0))
        return float(self.default_link_radius)

    def _link_name(self, idx):
        if idx < len(self.link_proxy_names):
            return str(self.link_proxy_names[idx])
        return "link_%d" % int(idx)

    def _evaluate_link_clearance(self, route, samples, boundary):
        if len(samples) == 0:
            return {
                "min_link_clearance": 0.0,
                "link_collision_valid": False,
                "link_collision_risk": 1.0,
                "worst_link_id": "",
                "worst_link_clearance_position": [],
            }
        base = self._arm_base_position(route, samples)
        min_clearance = float("inf")
        worst_link_id = ""
        worst_position = []
        previous_joints = None
        for ee_point in samples:
            joints = self._proxy_joint_positions(ee_point, base)
            for idx in range(max(0, len(joints) - 1)):
                link_id = self._link_name(idx)
                radius = self._link_radius(idx)
                for probe in self._segment_samples(
                        joints[idx], joints[idx + 1],
                        self.link_sample_spacing):
                    clearance = self._point_clearance(probe, boundary) - radius
                    if clearance < min_clearance:
                        min_clearance = float(clearance)
                        worst_link_id = link_id
                        worst_position = np.asarray(probe, float)[:3].tolist()
            if previous_joints is not None:
                for idx in range(min(len(previous_joints), len(joints))):
                    link_id = "%s_swept" % self._link_name(
                        min(idx, max(0, len(joints) - 2)))
                    radius = self._link_radius(min(idx, max(0, len(joints) - 2)))
                    for probe in self._segment_samples(
                            previous_joints[idx], joints[idx],
                            self.link_sample_spacing):
                        clearance = self._point_clearance(probe, boundary) - radius
                        if clearance < min_clearance:
                            min_clearance = float(clearance)
                            worst_link_id = link_id
                            worst_position = np.asarray(probe, float)[:3].tolist()
            previous_joints = joints
        if not np.isfinite(min_clearance):
            min_clearance = 0.0
        return {
            "min_link_clearance": float(min_clearance),
            "link_collision_valid": bool(min_clearance >= -1e-9),
            "link_collision_risk": float(max(0.0, -min_clearance)),
            "worst_link_id": worst_link_id,
            "worst_link_clearance_position": worst_position,
        }

    def _overall_route_score(self, min_ee_clearance, link_collision_risk,
                             max_risk):
        clearance_reward = float(min_ee_clearance) * self.clearance_score_scale
        link_penalty = (
            float(max(0.0, link_collision_risk)) *
            self.link_collision_penalty_scale)
        risk_penalty = float(max(0.0, max_risk)) / max(
            self.risk_threshold, 1e-6)
        return float(clearance_reward - link_penalty - risk_penalty)

    def validate_route(self, route):
        route_id = self._route_id(route)
        sequence = list(_route_value(route, "critical_point_sequence", []) or [])
        points = self._route_points(route)
        samples = _sample_polyline(points, self.sample_spacing)
        boundary = self._route_boundary(route)

        min_ee_clearance = float("inf")
        max_risk = 0.0
        for point in samples:
            min_ee_clearance = min(
                min_ee_clearance, self._point_clearance(point, boundary))
            max_risk = max(
                max_risk, _finite_float(
                    manifold_risk_value(point, self.risk_field), 0.0))

        if not np.isfinite(min_ee_clearance):
            min_ee_clearance = 0.0
        link_status = self._evaluate_link_clearance(route, samples, boundary)
        min_link_clearance = float(link_status.get("min_link_clearance", 0.0))
        link_collision_risk = float(link_status.get(
            "link_collision_risk", max(0.0, -min_link_clearance)))
        ee_collision_valid = bool(min_ee_clearance >= -1e-9)
        link_collision_valid = bool(link_status.get(
            "link_collision_valid", False))
        topology_valid = bool(_route_value(route, "topology_valid", True))
        ik_reachable = self._ik_reachable(route, points)
        joint_continuity_valid = self._joint_continuity_valid(samples)
        posture_valid = self._posture_valid(points)
        overall_route_score = self._overall_route_score(
            min_ee_clearance, link_collision_risk, max_risk)

        failure_reason = ""
        if len(samples) == 0:
            failure_reason = "empty_route"
        elif not topology_valid:
            failure_reason = "topology_invalid"
        elif not ik_reachable:
            failure_reason = "ik_unreachable"
        elif not joint_continuity_valid:
            failure_reason = "joint_discontinuity"
        elif not posture_valid:
            failure_reason = "posture_invalid"
        elif not ee_collision_valid:
            failure_reason = "end_effector_collision"
        elif min_ee_clearance + 1e-9 < self.required_clearance:
            failure_reason = "end_effector_clearance_violation"
        elif max_risk > self.risk_threshold + 1e-9:
            failure_reason = "risk_threshold_violation"

        route_valid = bool(not failure_reason)
        result = {
            "route_valid": bool(route_valid),
            "route_id": route_id,
            "critical_point_sequence": sequence,
            "min_end_effector_clearance": float(min_ee_clearance),
            "min_link_clearance": float(min_link_clearance),
            "ee_collision_valid": bool(ee_collision_valid),
            "link_collision_valid": bool(link_collision_valid),
            "ik_reachable": bool(ik_reachable),
            "joint_continuity_valid": bool(joint_continuity_valid),
            "posture_valid": bool(posture_valid),
            "link_collision_risk": float(link_collision_risk),
            "overall_route_score": float(overall_route_score),
            "worst_link_id": str(link_status.get("worst_link_id", "")),
            "worst_link_clearance_position": list(link_status.get(
                "worst_link_clearance_position", []) or []),
            "max_risk": float(max_risk),
            "failure_reason": failure_reason,
        }
        if isinstance(route, dict):
            route["arm_route_validation"] = dict(result)
        return result

    def _ik_reachable(self, route, points):
        pts = _as_points(points)
        if len(pts) == 0:
            return False
        base = self._arm_base_position(route, pts)
        reach = np.linalg.norm(pts - base, axis=1)
        return bool(float(np.max(reach)) <= 1.25 + 1e-9)

    def _joint_continuity_valid(self, samples):
        pts = _as_points(samples)
        if len(pts) <= 1:
            return bool(len(pts) == 1)
        steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        return bool(float(np.max(steps)) <= max(0.20, 4.0 * self.sample_spacing))

    def _posture_valid(self, points):
        pts = _as_points(points)
        if len(pts) == 0:
            return False
        return bool(float(np.min(pts[:, 2])) >= -0.10)

    def validate_routes(self, routes):
        valid_routes = []
        report = []
        for route in list(routes or []):
            validation = self.validate_route(route)
            report.append(dict(validation))
            if validation.get("route_valid"):
                valid_routes.append(route)
        return valid_routes, report
