import sys
sys.dont_write_bytecode = True

import numpy as np

from .manifold import Corridor


CORRIDOR_CONTRACT_VERSION = "stsm_corridor_contract_v1"


class CorridorContractError(ValueError):
    """Raised when an STSM corridor cannot be traced into MPC."""


def _points(value):
    if value is None or isinstance(value, str):
        return np.zeros((0, 3), float)
    try:
        arr = np.asarray(value, float)
    except (TypeError, ValueError):
        return np.zeros((0, 3), float)
    if arr.size == 0:
        return np.zeros((0, 3), float)
    if arr.ndim == 0:
        return np.zeros((0, 3), float)
    if arr.ndim == 1:
        arr = arr.reshape((1, arr.shape[0]))
    if arr.shape[1] == 2:
        arr = np.hstack([arr, np.zeros((arr.shape[0], 1), float)])
    return arr[:, :3]


def _first_points(*values):
    for value in values:
        points = _points(value)
        if len(points):
            return points
    return np.zeros((0, 3), float)


def corridor_contract(corridor, reference_path=None):
    """Return the canonical identity/geometry payload for one corridor."""
    if corridor is None:
        return {
            "contract_version": CORRIDOR_CONTRACT_VERSION,
            "valid": False,
            "failure_reason": "corridor_missing",
        }
    corridor_id = str(getattr(corridor, "corridor_id", "") or "")
    topology_class = str(
        getattr(corridor, "topology_route_class", "") or
        getattr(corridor, "topology_class", "") or "")
    node_sequence = list(
        getattr(corridor, "node_sequence", None) or
        getattr(corridor, "topology_nodes", None) or [])
    critical_ids = list(
        getattr(corridor, "critical_point_ids", None) or
        getattr(corridor, "morse_node_ids", None) or [])
    raw_waypoints = _points(getattr(corridor, "waypoints", []))
    refined = _first_points(
        getattr(corridor, "refined_waypoints", None),
        getattr(corridor, "execution_reference_path", None))
    reference = _points(reference_path)
    centerline = _first_points(refined, reference, raw_waypoints)
    radius = float(getattr(corridor, "radius", 0.0) or 0.0)
    boundary = dict(getattr(corridor, "boundary", {}) or {})
    source = str(
        getattr(corridor, "candidate_source", "") or
        getattr(corridor, "route_source", "") or
        getattr(corridor, "source", "") or "")
    recovery_level = str(
        getattr(corridor, "recovery_level", "") or
        getattr(corridor, "candidate_recovery_mode", "") or "none")
    return {
        "contract_version": CORRIDOR_CONTRACT_VERSION,
        "corridor_id": corridor_id,
        "topology_class": topology_class,
        "node_sequence": node_sequence,
        "critical_point_ids": critical_ids,
        "waypoints": raw_waypoints.tolist(),
        "refined_waypoints": refined.tolist(),
        "centerline": centerline.tolist(),
        "tube": {
            "radius": radius,
            "boundary": boundary,
            "centerline": centerline.tolist(),
        },
        "source": source,
        "recovery_level": recovery_level,
    }


def validate_corridor_contract(corridor, reference_path=None,
                               expected_corridor_id=None,
                               require_morse=False, require_tube=True):
    payload = corridor_contract(corridor, reference_path=reference_path)
    reasons = []
    corridor_id = str(payload.get("corridor_id", ""))
    if not corridor_id:
        reasons.append("corridor_id_missing")
    if (expected_corridor_id not in (None, "") and
            corridor_id != str(expected_corridor_id)):
        reasons.append("corridor_id_mismatch")
    if len(payload.get("centerline", [])) < 2:
        reasons.append("corridor_centerline_missing")
    if require_tube and float(payload.get("tube", {}).get("radius", 0.0)) <= 0.0:
        reasons.append("trajectory_tube_missing")
    if require_morse:
        if not payload.get("topology_class"):
            reasons.append("topology_class_missing")
        if not payload.get("node_sequence"):
            reasons.append("node_sequence_missing")
        if not payload.get("critical_point_ids"):
            reasons.append("critical_point_ids_missing")
        source = str(payload.get("source", "")).lower()
        if "morse" not in source and not bool(
                getattr(corridor, "morse_induced", False)):
            reasons.append("candidate_source_not_morse")
    payload["valid"] = not reasons
    payload["failure_reason"] = "|".join(reasons)
    return payload


def require_corridor_contract(corridor, reference_path=None,
                              expected_corridor_id=None,
                              require_morse=False, require_tube=True):
    payload = validate_corridor_contract(
        corridor, reference_path=reference_path,
        expected_corridor_id=expected_corridor_id,
        require_morse=require_morse, require_tube=require_tube)
    if not payload.get("valid", False):
        raise CorridorContractError(payload.get(
            "failure_reason", "invalid_corridor_contract"))
    corridor.corridor_contract_version = CORRIDOR_CONTRACT_VERSION
    corridor.corridor_contract = payload
    corridor.critical_point_ids = list(payload.get("critical_point_ids", []))
    return payload

__all__ = [
    "Corridor", "CORRIDOR_CONTRACT_VERSION", "CorridorContractError",
    "corridor_contract", "validate_corridor_contract",
    "require_corridor_contract",
]
