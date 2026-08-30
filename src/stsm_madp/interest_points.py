import sys
sys.dont_write_bytecode = True

import numpy as np


DEFAULT_WC_LOCAL_POINTS = {
    "center": [0.00, 0.00],
    "front_center": [0.35, 0.00],
    "front_left": [0.35, 0.30],
    "front_right": [0.35, -0.30],
    "footrest_left": [0.50, 0.22],
    "footrest_right": [0.50, -0.22],
    "rear_left": [-0.35, 0.30],
    "rear_right": [-0.35, -0.30],
}

WC_LABELS = [
    "center",
    "front_center",
    "front_left",
    "front_right",
    "footrest_left",
    "footrest_right",
    "rear_left",
    "rear_right",
]


def _as_point3(p, z=0.0):
    arr = np.asarray(p, float)
    if arr.shape[0] >= 3:
        return arr[:3].astype(float)
    return np.array([arr[0], arr[1], float(z)], dtype=float)


def transform_points_2d(state, local_points):
    x, y, yaw = float(state[0]), float(state[1]), float(state[2])
    c, s = np.cos(yaw), np.sin(yaw)
    out = {}
    for label, p in local_points.items():
        lx, ly = float(p[0]), float(p[1])
        wx = x + c * lx - s * ly
        wy = y + s * lx + c * ly
        out[label] = np.array([wx, wy, 0.0], dtype=float)
    return out


def points_from_offsets(anchor, offsets, labels=None):
    anchor = _as_point3(anchor)
    offsets = offsets or {}
    if labels is None:
        labels = list(offsets.keys())
    points = []
    out_labels = []
    for label in labels:
        if label not in offsets:
            continue
        points.append(anchor + _as_point3(offsets[label]))
        out_labels.append(label)
    return out_labels, points


def aggregate_point_risks(field, labels, points, vels=None):
    use_batch = bool(
        hasattr(field, "phi_s_batch") and
        (vels is None or all(value is None for value in vels)))
    if use_batch:
        phi_each = np.asarray(field.phi_s_batch(points), float).tolist()
    else:
        if vels is None:
            vels = [None] * len(points)
        phi_each = [float(field.phi_s(p, v)) for p, v in zip(points, vels)]
    worst_idx = int(np.argmax(phi_each)) if phi_each else -1
    return {
        "phi_each": phi_each,
        "phi_max": float(np.max(phi_each)) if phi_each else 0.0,
        "phi_mean": float(np.mean(phi_each)) if phi_each else 0.0,
        "phi_sum": float(np.sum(phi_each)) if phi_each else 0.0,
        "risk_gate": float(np.max(phi_each)) if phi_each else 0.0,
        "worst_idx": worst_idx,
        "worst_label": labels[worst_idx] if worst_idx >= 0 else "",
    }


def pose_interest_risk(field, state, local_points=None, offsets=None,
                       labels=None, vels=None):
    if offsets is not None:
        out_labels, points = points_from_offsets(state, offsets, labels)
    else:
        local_points = local_points or DEFAULT_WC_LOCAL_POINTS
        points_map = transform_points_2d(state, local_points)
        out_labels = list(labels or points_map.keys())
        points = [points_map[k] for k in out_labels if k in points_map]
        out_labels = [k for k in out_labels if k in points_map]
    summary = aggregate_point_risks(field, out_labels, points, vels)
    summary["labels"] = out_labels
    summary["points"] = points
    return summary


def pose_interest_risk_batch(field, states, local_points=None, labels=None):
    """Evaluate fixed local interest points for several planar poses exactly.

    Wheelchair MPC evaluates a short set of sibling rollouts at each beam
    layer.  Their footprint risks do not depend on one another, so the social
    field can evaluate the flattened footprint points in one vectorized call
    without changing the per-pose risk definition.
    """
    pose_array = np.asarray(states, float)
    if pose_array.size == 0:
        return []
    if pose_array.ndim == 1:
        pose_array = pose_array.reshape((1, pose_array.shape[0]))
    points_map = local_points or DEFAULT_WC_LOCAL_POINTS
    out_labels = [label for label in (labels or points_map.keys())
                  if label in points_map]
    if not out_labels:
        return [{"phi_each": [], "phi_max": 0.0, "phi_mean": 0.0,
                 "phi_sum": 0.0, "risk_gate": 0.0, "worst_idx": -1,
                 "worst_label": "", "labels": [], "points": []}
                for _ in pose_array]
    local = np.asarray([points_map[label] for label in out_labels], float)
    yaw = pose_array[:, 2] if pose_array.shape[1] >= 3 else np.zeros(
        len(pose_array), float)
    c, s = np.cos(yaw), np.sin(yaw)
    world = np.zeros((len(pose_array), len(out_labels), 3), float)
    world[:, :, 0] = (pose_array[:, None, 0] +
                      c[:, None] * local[None, :, 0] -
                      s[:, None] * local[None, :, 1])
    world[:, :, 1] = (pose_array[:, None, 1] +
                      s[:, None] * local[None, :, 0] +
                      c[:, None] * local[None, :, 1])
    if hasattr(field, "phi_s_batch"):
        values = np.asarray(field.phi_s_batch(
            world.reshape((-1, 3))), float).reshape(world.shape[:2])
    else:
        values = np.asarray([
            float(field.phi_s(point)) for point in world.reshape((-1, 3))
        ], float).reshape(world.shape[:2])
    summaries = []
    for row, points in zip(values, world):
        phi_each = row.tolist()
        worst_idx = int(np.argmax(row)) if len(row) else -1
        summaries.append({
            "phi_each": phi_each,
            "phi_max": float(np.max(row)) if len(row) else 0.0,
            "phi_mean": float(np.mean(row)) if len(row) else 0.0,
            "phi_sum": float(np.sum(row)) if len(row) else 0.0,
            "risk_gate": float(np.max(row)) if len(row) else 0.0,
            "worst_idx": worst_idx,
            "worst_label": out_labels[worst_idx] if worst_idx >= 0 else "",
            "labels": list(out_labels),
            "points": [point.copy() for point in points],
        })
    return summaries


def point_inside_anchor(anchor, p):
    return float(anchor.signed_distance(p)) <= 0.0


def forbidden_anchor_hit(field, labels, points):
    for label, p in zip(labels, points):
        for anchor in getattr(field, "anchors", []):
            if getattr(anchor, "forbidden", False) and point_inside_anchor(anchor, p):
                anchor_type = getattr(anchor, "type", "unknown")
                reason = "footprint:forbidden_zone:{0}:{1}".format(
                    label, anchor_type)
                return True, label, anchor_type, reason
    return False, "", "", ""
