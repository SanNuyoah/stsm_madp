import sys
sys.dont_write_bytecode = True

import json
import os

import numpy as np


def _as_path(points):
    if points is None:
        return np.zeros((0, 3), float)
    pts = np.asarray(points, float)
    if pts.size == 0:
        return np.zeros((0, 3), float)
    if pts.ndim == 1:
        pts = pts.reshape((1, pts.shape[0]))
    if pts.shape[1] == 2:
        pts = np.hstack([pts, np.zeros((pts.shape[0], 1), float)])
    return pts[:, :3]


def normalize_critical_points(critical_points):
    out = []
    for idx, item in enumerate(critical_points or []):
        if not isinstance(item, dict):
            continue
        point = item.get("point", item.get("position", None))
        try:
            p = _as_path([point])[0]
        except Exception:
            continue
        try:
            order = int(item.get("order", idx + 1))
        except Exception:
            order = idx + 1
        kind = str(item.get("type", item.get("kind", "critical")))
        if kind == "minima":
            kind = "minimum"
        out.append({
            "id": str(item.get("id", "critical_{}".format(idx))),
            "type": kind,
            "position": [float(p[0]), float(p[1]), float(p[2])],
            "point": [float(p[0]), float(p[1]), float(p[2])],
            "stage_order": order,
            "order": order,
        })
    out.sort(key=lambda item: int(item.get("stage_order", 0)))
    return out


def _critical_points_from_corridor(corridor):
    if corridor is None:
        return []
    existing = getattr(corridor, "topology_constraint_info", {}) or {}
    explicit = (
        getattr(corridor, "critical_point_sequence", None) or
        existing.get("critical_point_sequence") or
        getattr(corridor, "critical_points", None) or
        existing.get("critical_points") or [])
    return normalize_critical_points(explicit)


def associate_critical_points(critical_points, trajectory,
                              soft_radius=0.5, hard_radius=1.0):
    path = _as_path(trajectory)
    soft_radius = float(soft_radius)
    hard_radius = max(float(hard_radius), soft_radius)
    normalized = normalize_critical_points(critical_points)
    associated = []
    for item in normalized:
        q = np.asarray(item["position"], float)[:3]
        if len(path) == 0:
            idx = -1
            distance = float("inf")
        else:
            ds = np.linalg.norm(path[:, :3] - q[None, :3], axis=1)
            idx = int(np.argmin(ds))
            distance = float(ds[idx])
        if distance <= soft_radius:
            status = "passed"
        elif distance <= hard_radius:
            status = "soft_violation"
        else:
            status = "hard_violation"
        row = dict(item)
        row.update({
            "trajectory_index": int(idx),
            "stage_order": int(item.get("stage_order", item.get("order", 0))),
            "distance_to_trajectory": float(distance),
            "critical_point_status": status,
        })
        associated.append(row)
    indices = [int(item.get("trajectory_index", -1)) for item in associated]
    orders = [int(item.get("stage_order", 0)) for item in associated]
    order_valid = (
        orders == sorted(orders) and
        all(idx >= 0 for idx in indices) and
        indices == sorted(indices))
    for i, item in enumerate(associated):
        prev_idx = 0 if i == 0 else int((indices[i - 1] + indices[i]) / 2)
        next_idx = (
            max(0, len(path) - 1) if i == len(associated) - 1
            else int((indices[i] + indices[i + 1]) / 2))
        item["stage_window"] = [int(prev_idx), int(next_idx)]
    hard_count = sum(
        1 for item in associated
        if str(item.get("critical_point_status")) == "hard_violation")
    soft_count = sum(
        1 for item in associated
        if str(item.get("critical_point_status")) == "soft_violation")
    if not order_valid or hard_count:
        status = "hard_violation"
    elif soft_count:
        status = "soft_violation"
    else:
        status = "passed"
    return {
        "critical_points": associated,
        "critical_point_count": int(len(associated)),
        "critical_point_association_used": True,
        "topology_sequence_valid": bool(order_valid and hard_count == 0),
        "critical_point_status": status,
        "soft_violation_count": int(soft_count),
        "hard_violation_count": int(hard_count),
        "trajectory_count": int(len(path)),
        "soft_radius": float(soft_radius),
        "hard_radius": float(hard_radius),
    }


def associate_corridor_critical_points(corridor, trajectory=None,
                                       soft_radius=0.5, hard_radius=1.0):
    if trajectory is None and corridor is not None:
        refined = getattr(corridor, "refined_waypoints", None)
        if refined is not None and np.asarray(refined).size:
            trajectory = refined
        else:
            trajectory = getattr(corridor, "waypoints", None)
    return associate_critical_points(
        _critical_points_from_corridor(corridor), trajectory,
        soft_radius=soft_radius, hard_radius=hard_radius)


def write_critical_point_association(path, association):
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    with open(path, "w") as f:
        json.dump(association or {}, f, indent=2, sort_keys=True)
