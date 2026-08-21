import sys
sys.dont_write_bytecode = True

import numpy as np


def _as_p2(point):
    try:
        arr = np.asarray(point, float).reshape((-1,))
    except Exception:
        return None
    if arr.size < 2:
        return None
    return arr[:2]


def _clip_to_bounds(p2, bounds):
    p2 = np.asarray(p2, float).copy()
    (xmin, xmax), (ymin, ymax) = bounds
    p2[0] = min(max(p2[0], xmin), xmax)
    p2[1] = min(max(p2[1], ymin), ymax)
    return p2


def _append_unique(out, kind, point, semantic_type, bounds, min_sep):
    p2 = _as_p2(point)
    if p2 is None:
        return
    p2 = _clip_to_bounds(p2, bounds)
    for item in out:
        if item["kind"] != kind:
            continue
        if float(np.linalg.norm(np.asarray(item["point"], float)[:2] - p2)) < min_sep:
            return
    out.append({
        "kind": str(kind),
        "point": [float(p2[0]), float(p2[1]), 0.0],
        "semantic_type": str(semantic_type),
        "source": "semantic_topology_recovery",
    })


def semantic_topology_specs(topology_profile, start, goal, bounds,
                            task_minima_points=None, semantic_nodes=None,
                            merge_radius=0.20):
    """Create semantic minimum/saddle specs when numeric Morse extraction fails."""
    profile = str(topology_profile or "generic").strip().lower()
    start2 = _as_p2(start)
    goal2 = _as_p2(goal)
    if start2 is None or goal2 is None:
        return []
    direction = goal2 - start2
    length = float(np.linalg.norm(direction))
    if length <= 1e-9:
        return []
    normal = np.array([-direction[1], direction[0]], float) / length
    mid = 0.5 * (start2 + goal2)
    min_sep = max(1e-6, 0.5 * float(merge_radius))
    out = []

    for item in list(task_minima_points or []):
        if isinstance(item, dict):
            point = item.get("position", item.get("point", None))
            role = str(item.get("type", "task"))
        else:
            point = item
            role = "task"
        if point is not None:
            _append_unique(out, "minimum", point, role, bounds, min_sep)

    for point in list(semantic_nodes or []):
        _append_unique(out, "minimum", point, "semantic", bounds, min_sep)

    if profile == "arm":
        _append_unique(out, "minimum", goal2, "handover", bounds, min_sep)
        saddle_offsets = [0.45, -0.45, 0.30]
        saddle_roles = ["side_approach", "front_approach", "handover_connection"]
    elif profile == "wheelchair":
        _append_unique(out, "minimum", goal2, "parking", bounds, min_sep)
        _append_unique(out, "minimum", start2 + 0.35 * direction, "waiting", bounds, min_sep)
        saddle_offsets = [0.55, -0.55, 0.30]
        saddle_roles = ["door_passage", "bypass", "connection_channel"]
    else:
        _append_unique(out, "minimum", goal2, "goal_region", bounds, min_sep)
        saddle_offsets = [0.45, -0.45]
        saddle_roles = ["passage", "bypass"]

    for offset, role in zip(saddle_offsets, saddle_roles):
        _append_unique(out, "saddle", mid + float(offset) * normal,
                       role, bounds, min_sep)
    return out
