import sys
sys.dont_write_bytecode = True

import copy

import numpy as np

from stsm_madp.arm_configuration_space import ArmConfigurationSpaceValidator
from stsm_madp.arm_ik_candidate_generator import ArmIKCandidateGenerator
from stsm_madp.arm_pose_optimizer import ArmPoseOptimizer


def _route_value(route, key, default=None):
    if isinstance(route, dict):
        return route.get(key, default)
    return getattr(route, key, default)


def _route_points(route):
    for key in ("centerline", "waypoints", "refined_waypoints", "path"):
        pts = _route_value(route, key, None)
        try:
            arr = np.asarray(pts, float)
        except Exception:
            continue
        if arr.size:
            if arr.ndim == 1:
                arr = arr.reshape((1, arr.shape[0]))
            if arr.shape[1] == 2:
                arr = np.hstack([arr, np.zeros((arr.shape[0], 1), float)])
            return arr[:, :3]
    return np.zeros((0, 3), float)


class TopologyIKSolver(object):
    """Couple Morse candidates with IK and joint-space validation."""

    def __init__(self, configuration_validator=None, **kwargs):
        self.configuration_validator = (
            configuration_validator or ArmConfigurationSpaceValidator(**kwargs))
        self.candidate_generator = kwargs.get(
            "arm_ik_candidate_generator", ArmIKCandidateGenerator())
        self.pose_optimizer = kwargs.get(
            "arm_pose_optimizer", ArmPoseOptimizer())

    def validate_candidate(self, candidate, boundary=None, risk_field=None,
                           seed=None):
        route = copy.deepcopy(candidate if isinstance(candidate, dict) else {})
        attempts = []
        variants = [(route, seed)]
        variants.extend(self.candidate_generator.generate(route, seed=seed))
        best_route = route
        best_validation = None
        best_score = -float("inf")
        for variant, variant_seed in variants:
            pts = _route_points(variant)
            variant_boundary = (
                boundary if boundary is not None else variant.get("boundary", None))
            validation = self.configuration_validator.validate_cartesian_trajectory(
                pts, seed=variant_seed, boundary=variant_boundary,
                risk_field=risk_field,
                link_sample_points=variant.get("link_sample_points", None),
                link_sample_records=variant.get("link_sample_records", None))
            link_status = dict(validation.get("link_collision", {}) or {})
            score = self.pose_optimizer.score(
                variant, validation, risk_field=risk_field)
            attempts.append({
                "variant": dict(variant.get("arm_ik_variant", {}) or {}),
                "valid": bool(validation.get("valid", False)),
                "failure_reason": str(validation.get("failure_reason", "")),
                "link_collision_valid": bool(link_status.get(
                    "link_collision_valid", validation.get("valid", False))),
                "min_clearance": float(validation.get("min_clearance", 0.0)),
                "max_risk": float(validation.get("max_risk", 0.0)),
                "score": float(score),
            })
            if bool(validation.get("valid", False)) and score > best_score:
                best_route = variant
                best_validation = validation
                best_score = float(score)
            if best_validation is None or score > best_score:
                best_route = variant
                best_validation = validation
                best_score = float(score)
        route = copy.deepcopy(best_route)
        validation = dict(best_validation or {})
        route["ik_validation"] = dict(validation)
        route["arm_ik_candidate_attempts"] = attempts
        route["arm_ik_candidate_count"] = int(len(attempts))
        route["ik_valid"] = bool(validation.get("valid", False))
        route["joint_trajectory"] = list(validation.get("joint_trajectory", []))
        route["link_sample_points"] = list(validation.get("link_sample_points", []))
        route["link_sample_records"] = list(validation.get("link_sample_records", []))
        route["link_collision_valid"] = bool(validation.get(
            "link_collision", {}).get("link_collision_valid",
            validation.get("valid", False)))
        route["collision_link"] = str(validation.get("collision_link", ""))
        route["configuration_min_clearance"] = float(
            validation.get("min_clearance", 0.0))
        route["configuration_max_risk"] = float(validation.get("max_risk", 0.0))
        route["arm_pose_optimization_used"] = True
        route["arm_pose_optimizer_score"] = float(best_score)
        if not route["ik_valid"]:
            reason = str(validation.get("failure_reason", "ik_or_link_collision"))
            route["candidate_status"] = (
                "recoverable" if reason in (
                    "link_collision", "fk_forward_check_failed",
                    "joint_limit_violation") else "invalid")
            route["failure_reason"] = str(
                validation.get("failure_reason", "ik_or_link_collision"))
        else:
            route["candidate_status"] = "feasible"
            route["failure_reason"] = ""
        return route, validation

    def filter_candidates(self, candidates, boundary=None, risk_field=None):
        valid = []
        report = []
        invalid = []
        for candidate in list(candidates or []):
            route, validation = self.validate_candidate(
                candidate, boundary=boundary, risk_field=risk_field)
            report.append({
                "candidate_id": str(route.get("candidate_id", "")),
                "ik_valid": bool(route.get("ik_valid", False)),
                "link_collision_valid": bool(route.get("link_collision_valid", False)),
                "collision_link": str(route.get("collision_link", "")),
                "min_clearance": float(route.get("configuration_min_clearance", 0.0)),
                "max_risk": float(route.get("configuration_max_risk", 0.0)),
                "failure_reason": str(validation.get("failure_reason", "")),
                "candidate_status": str(route.get("candidate_status", "")),
                "arm_ik_candidate_count": int(route.get(
                    "arm_ik_candidate_count", 0)),
                "arm_ik_candidate_attempts": list(route.get(
                    "arm_ik_candidate_attempts", [])),
            })
            if bool(route.get("ik_valid", False)):
                valid.append(route)
            else:
                invalid.append(route)
        return valid, {
            "topology_ik_validation_used": True,
            "topology_ik_attempted": int(len(report)),
            "topology_ik_valid_count": int(len(valid)),
            "topology_ik_invalid_count": int(len(invalid)),
            "topology_ik_report": report,
        }, invalid
