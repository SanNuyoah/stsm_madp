import sys
sys.dont_write_bytecode = True

import math

import numpy as np

from stsm_madp.arm_interest_sampler import ArmInterestSampler
from stsm_madp.manifold_constraint import (
    distance_to_manifold_boundary,
    manifold_risk_value,
)


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


class ArmConfigurationSpaceValidator(object):
    """Validate Cartesian arm candidates in joint and link space.

    A caller may inject a real IK solver and FK solver. Without one, this class
    uses a deterministic proxy chain so topology candidates are still filtered
    by joint-limit continuity and link-level clearance before MPC.
    """

    def __init__(self, ik_solver=None, fk_solver=None, joint_limits=None,
                 risk_field=None, boundary=None, risk_threshold=6.0,
                 minimum_clearance=0.0, link_radius=0.015,
                 base_position=None, interest_sampler=None, **_kwargs):
        self.ik_solver = ik_solver
        self.fk_solver = fk_solver
        self.joint_limits = list(joint_limits or [
            (-math.pi, math.pi),
            (-2.6, 2.6),
            (-2.6, 2.6),
            (-math.pi, math.pi),
            (-math.pi, math.pi),
            (-math.pi, math.pi),
        ])
        self.risk_field = risk_field
        self.boundary = boundary
        self.risk_threshold = _finite_float(risk_threshold, 6.0)
        self.minimum_clearance = max(0.0, _finite_float(minimum_clearance, 0.0))
        self.link_radius = max(0.0, _finite_float(link_radius, 0.015))
        self.base_position = np.asarray(
            base_position if base_position is not None else [0.0, 0.0, 0.0],
            float)[:3]
        self.interest_sampler = interest_sampler or ArmInterestSampler()

    def solve_ik_trajectory(self, cartesian_trajectory, seed=None):
        pts = _as_points(cartesian_trajectory)
        if len(pts) == 0:
            return np.zeros((0, len(self.joint_limits)), float), {
                "ik_success": False,
                "failure_reason": "empty_cartesian_trajectory",
            }
        joints = []
        failures = []
        q_seed = seed
        for idx, point in enumerate(pts):
            q, status = self._solve_ik(point, q_seed)
            if not bool(status.get("ik_success", False)):
                failures.append({
                    "index": int(idx),
                    "failure_reason": str(status.get("failure_reason", "ik_failed")),
                })
                continue
            q_seed = q
            joints.append(np.asarray(q, float).tolist())
        return np.asarray(joints, float), {
            "ik_success": bool(len(failures) == 0 and len(joints) == len(pts)),
            "failure_reason": "" if not failures else "ik_failed",
            "ik_failure_count": int(len(failures)),
            "ik_failures": failures,
            "joint_count": int(len(joints)),
        }

    def validate_cartesian_trajectory(self, cartesian_trajectory, seed=None,
                                      boundary=None, risk_field=None,
                                      grasp_object_trajectory=None,
                                      link_sample_points=None,
                                      link_sample_records=None):
        pts = _as_points(cartesian_trajectory)
        joint_traj, ik_status = self.solve_ik_trajectory(pts, seed=seed)
        if not bool(ik_status.get("ik_success", False)):
            return self._invalid_result(
                "ik_failed", joint_traj, ik_status, pts, collision_link="")
        joint_status = self._validate_joint_limits(joint_traj)
        fk_status = self._forward_check(joint_traj, pts)
        link_status = self._validate_links(
            pts, boundary=boundary, risk_field=risk_field,
            grasp_object_trajectory=grasp_object_trajectory,
            link_sample_points=link_sample_points,
            link_sample_records=link_sample_records)
        valid = bool(
            joint_status.get("joint_limits_valid", False) and
            fk_status.get("fk_valid", False) and
            link_status.get("link_collision_valid", False))
        reason = ""
        if not valid:
            if not joint_status.get("joint_limits_valid", False):
                reason = "joint_limit_violation"
            elif not fk_status.get("fk_valid", False):
                reason = "fk_forward_check_failed"
            else:
                reason = "link_collision"
        out = {
            "valid": bool(valid),
            "failure_reason": reason,
            "collision_link": str(link_status.get("collision_link", "")),
            "min_clearance": float(link_status.get("min_clearance", 0.0)),
            "max_risk": float(link_status.get("max_risk", 0.0)),
            "ik": dict(ik_status),
            "joint_limits": dict(joint_status),
            "fk": dict(fk_status),
            "link_collision": dict(link_status),
            "joint_trajectory": joint_traj.tolist(),
            "cartesian_trajectory": pts.tolist(),
            "link_sample_points": list(link_status.get("link_sample_points", [])),
            "link_sample_records": list(link_status.get("link_sample_records", [])),
        }
        return out

    def _solve_ik(self, point, seed=None):
        if self.ik_solver is not None:
            try:
                q = self.ik_solver(point, seed=seed)
            except TypeError:
                q = self.ik_solver(point)
            except Exception as exc:
                return None, {
                    "ik_success": False,
                    "failure_reason": "ik_exception:%s" % type(exc).__name__,
                }
            q = np.asarray(q, float).reshape((-1,))
            return q, {
                "ik_success": bool(self._q_within_limits(q)),
                "failure_reason": "" if self._q_within_limits(q) else "joint_limit_violation",
            }
        p = np.asarray(point, float)[:3] - self.base_position
        yaw = math.atan2(float(p[1]), float(p[0]))
        reach = float(np.linalg.norm(p[:2]))
        height = float(p[2])
        shoulder = math.atan2(height, max(1e-6, reach))
        elbow = -0.5 * shoulder
        wrist = -0.5 * shoulder
        q = np.asarray([yaw, shoulder, elbow, wrist, 0.0, 0.0], float)
        if len(self.joint_limits) != len(q):
            if len(self.joint_limits) < len(q):
                q = q[:len(self.joint_limits)]
            else:
                q = np.hstack([q, np.zeros(len(self.joint_limits) - len(q), float)])
        return q, {
            "ik_success": bool(self._q_within_limits(q)),
            "failure_reason": "" if self._q_within_limits(q) else "joint_limit_violation",
            "ik_source": "proxy_geometric",
        }

    def _q_within_limits(self, q):
        q = np.asarray(q, float).reshape((-1,))
        if len(q) != len(self.joint_limits):
            return False
        for value, limit in zip(q, self.joint_limits):
            lo, hi = limit
            if value < float(lo) - 1e-9 or value > float(hi) + 1e-9:
                return False
        return True

    def _validate_joint_limits(self, joint_traj):
        violations = []
        for row_idx, q in enumerate(np.asarray(joint_traj, float)):
            for joint_idx, (value, limit) in enumerate(zip(q, self.joint_limits)):
                lo, hi = limit
                if value < float(lo) - 1e-9 or value > float(hi) + 1e-9:
                    violations.append({
                        "trajectory_index": int(row_idx),
                        "joint_index": int(joint_idx),
                        "value": float(value),
                        "limit": [float(lo), float(hi)],
                    })
        return {
            "joint_limits_valid": bool(not violations),
            "joint_limit_violation_count": int(len(violations)),
            "violations": violations,
        }

    def _forward_check(self, joint_traj, cartesian_trajectory):
        if self.fk_solver is None:
            return {
                "fk_valid": True,
                "fk_error_max": 0.0,
                "fk_source": "proxy_pass_through",
            }
        errors = []
        for q, point in zip(joint_traj, cartesian_trajectory):
            try:
                fk = np.asarray(self.fk_solver(q), float).reshape((-1,))[:3]
                errors.append(float(np.linalg.norm(fk - np.asarray(point, float)[:3])))
            except Exception:
                errors.append(float("inf"))
        max_error = max(errors) if errors else float("inf")
        return {
            "fk_valid": bool(np.isfinite(max_error) and max_error <= 0.05),
            "fk_error_max": float(max_error if np.isfinite(max_error) else 1e9),
            "fk_source": "external_fk_solver",
        }

    def _validate_links(self, cartesian_trajectory, boundary=None,
                        risk_field=None, grasp_object_trajectory=None,
                        link_sample_points=None, link_sample_records=None):
        boundary = boundary if boundary is not None else self.boundary
        risk_field = risk_field if risk_field is not None else self.risk_field
        if link_sample_records:
            records = []
            points = []
            points_by_link = {}
            for rec in list(link_sample_records or []):
                point = _as_points(rec.get("point", []))
                if len(point) == 0:
                    continue
                out = dict(rec)
                out["point"] = point[0].tolist()
                link = str(out.get("link", out.get("name", "posture_proxy")))
                out["link"] = link
                records.append(out)
                points.append(out["point"])
                points_by_link.setdefault(link, []).append(out["point"])
            sampled = {
                "records": records,
                "points": points,
                "points_by_link": points_by_link,
            }
        elif link_sample_points is not None:
            points = _as_points(link_sample_points).tolist()
            records = [{
                "trajectory_index": int(idx),
                "name": "posture_proxy",
                "link": "posture_proxy",
                "point": point,
            } for idx, point in enumerate(points)]
            sampled = {
                "records": records,
                "points": points,
                "points_by_link": {"posture_proxy": points},
            }
        else:
            sampled = self.interest_sampler.sample_trajectory(
                cartesian_trajectory, base_point=self.base_position,
                grasp_object_trajectory=grasp_object_trajectory)
        min_clearance = float("inf")
        max_risk = 0.0
        collision_link = ""
        for rec in sampled.get("records", []):
            point = rec.get("point", [])
            clearance = distance_to_manifold_boundary(point, boundary)
            if not np.isfinite(clearance):
                clearance = self.risk_threshold - manifold_risk_value(
                    point, risk_field)
            radius = max(0.0, _finite_float(
                rec.get("radius", self.link_radius), self.link_radius))
            clearance = _finite_float(clearance, 0.0) - radius
            risk = _finite_float(manifold_risk_value(point, risk_field), 0.0)
            if clearance < min_clearance:
                min_clearance = float(clearance)
                collision_link = str(rec.get("link", ""))
            max_risk = max(max_risk, risk)
        if not np.isfinite(min_clearance):
            min_clearance = 0.0
        valid = bool(
            min_clearance + 1e-9 >= self.minimum_clearance and
            max_risk <= self.risk_threshold + 1e-9)
        return {
            "link_collision_valid": bool(valid),
            "collision_link": "" if valid else collision_link,
            "min_clearance": float(min_clearance),
            "max_risk": float(max_risk),
            "risk_threshold": float(self.risk_threshold),
            "minimum_clearance": float(self.minimum_clearance),
            "link_sample_count": int(len(sampled.get("points", []))),
            "link_sample_points": list(sampled.get("points", [])),
            "link_sample_records": list(sampled.get("records", [])),
            "points_by_link": dict(sampled.get("points_by_link", {})),
        }

    def _invalid_result(self, reason, joint_traj, ik_status, cartesian, collision_link=""):
        return {
            "valid": False,
            "failure_reason": str(reason),
            "collision_link": str(collision_link),
            "min_clearance": 0.0,
            "max_risk": 0.0,
            "ik": dict(ik_status),
            "joint_limits": {},
            "fk": {},
            "link_collision": {},
            "joint_trajectory": np.asarray(joint_traj, float).tolist(),
            "cartesian_trajectory": _as_points(cartesian).tolist(),
            "link_sample_points": [],
            "link_sample_records": [],
        }
