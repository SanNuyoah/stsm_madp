import sys
sys.dont_write_bytecode = True

import numpy as np


def _finite_float(value, default=0.0):
    try:
        out = float(value)
    except Exception:
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


class ArmPoseOptimizer(object):
    """Rank arm IK/posture variants by execution safety and smoothness."""

    BODY_WEIGHTS = {
        "hand": 0.35,
        "head": 3.0,
        "chest": 2.4,
        "torso": 2.0,
        "leg": 1.2,
    }

    def __init__(self, risk_field=None, clearance_weight=120.0,
                 risk_weight=8.0, motion_weight=0.08,
                 posture_weight=2.0, invalid_penalty=10000.0):
        self.risk_field = risk_field
        self.clearance_weight = float(clearance_weight)
        self.risk_weight = float(risk_weight)
        self.motion_weight = float(motion_weight)
        self.posture_weight = float(posture_weight)
        self.invalid_penalty = float(invalid_penalty)

    def score(self, route, validation, risk_field=None):
        risk_field = risk_field if risk_field is not None else self.risk_field
        validation = dict(validation or {})
        link_status = dict(validation.get("link_collision", {}) or {})
        min_clearance = _finite_float(validation.get(
            "min_clearance", link_status.get("min_clearance", 0.0)), 0.0)
        max_risk = _finite_float(validation.get(
            "max_risk", link_status.get("max_risk", 0.0)), 0.0)
        joint_motion = self._joint_motion(validation.get("joint_trajectory", []))
        posture_cost = self._posture_cost(route)
        body_cost = self._body_part_cost(
            link_status.get("link_sample_records",
                            validation.get("link_sample_records", [])),
            risk_field)
        valid_bonus = 0.0 if bool(validation.get("valid", False)) else -self.invalid_penalty
        return float(
            valid_bonus +
            self.clearance_weight * min_clearance -
            self.risk_weight * max_risk -
            self.motion_weight * joint_motion -
            self.posture_weight * posture_cost -
            body_cost)

    def _joint_motion(self, joint_trajectory):
        if joint_trajectory is None or isinstance(joint_trajectory, str):
            return 0.0
        try:
            q = np.asarray(joint_trajectory, float)
        except Exception:
            return 0.0
        if q.size == 0:
            return 0.0
        if q.ndim == 1:
            q = q.reshape((1, q.shape[0]))
        if len(q) <= 1:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(q, axis=0), axis=1)))

    def _posture_cost(self, route):
        variant = dict((route or {}).get("arm_ik_variant", {}) or {})
        cost = 0.0
        elbow = str(variant.get("elbow_posture", "middle"))
        if elbow == "middle":
            cost += 0.0
        elif elbow in ("up", "down"):
            cost += 0.35
        else:
            cost += 0.75
        wrist = abs(_finite_float(variant.get("wrist_angle", 0.0), 0.0))
        cost += wrist / max(np.pi, 1e-6)
        approach = str(variant.get("approach_direction", "front"))
        if approach != "front":
            cost += 0.25
        return float(cost)

    def _body_part_cost(self, link_records, risk_field):
        if risk_field is None:
            return 0.0
        humans = list(getattr(risk_field, "humans", []) or [])
        if not humans:
            return 0.0
        total = 0.0
        for rec in list(link_records or []):
            point = np.asarray(rec.get("point", []), float)
            if point.size < 2:
                continue
            for human in humans:
                for name, item in getattr(human, "body_parts", {}).items():
                    try:
                        center, _weight, sigma = item
                    except Exception:
                        continue
                    center = np.asarray(center, float)
                    n = min(len(point), len(center))
                    dist = float(np.linalg.norm(point[:n] - center[:n]))
                    weight = self.BODY_WEIGHTS.get(str(name), 1.0)
                    scale = max(_finite_float(sigma, 0.15), 1e-6)
                    total += float(weight * np.exp(-(dist * dist) / (2.0 * scale * scale)))
        return float(total)

    def select_best(self, attempts):
        best = None
        best_score = -float("inf")
        for item in list(attempts or []):
            score = _finite_float(item.get("score", -float("inf")), -float("inf"))
            if best is None or score > best_score:
                best = item
                best_score = score
        return best
