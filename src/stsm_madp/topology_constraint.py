import json
import os
import sys
sys.dont_write_bytecode = True

import numpy as np

from stsm_madp.critical_point_association import associate_critical_points
from stsm_madp.manifold_constraint import build_manifold_constraint


class TopologyConstraint(dict):
    """Unified topology constraint payload passed into MPC."""

    def __init__(self, critical_point_constraint=None,
                 corridor_constraint=None,
                 manifold_constraint=None,
                 topology_sequence=None,
                 payload=None):
        if payload is None and isinstance(critical_point_constraint, dict):
            candidate = dict(critical_point_constraint)
            payload_keys = (
                "topology_constraint_used",
                "critical_point_constraint",
                "corridor_constraint",
                "manifold_constraint",
                "topology_sequence_constraint",
                "corridor_centerline",
                "critical_point_sequence",
            )
            if (
                    corridor_constraint is None and
                    manifold_constraint is None and
                    topology_sequence is None and
                    any(key in candidate for key in payload_keys)):
                payload = candidate
                critical_point_constraint = candidate.get(
                    "critical_point_constraint")
                corridor_constraint = candidate.get("corridor_constraint")
                manifold_constraint = candidate.get("manifold_constraint")
                topology_sequence = candidate.get(
                    "topology_sequence_constraint")

        data = dict(payload or {})
        if critical_point_constraint is not None:
            data["critical_point_constraint"] = dict(
                critical_point_constraint or {})
        if corridor_constraint is not None:
            data["corridor_constraint"] = dict(corridor_constraint or {})
        if manifold_constraint is not None:
            data["manifold_constraint"] = dict(manifold_constraint or {})
        if topology_sequence is not None:
            if isinstance(topology_sequence, dict):
                data["topology_sequence_constraint"] = dict(
                    topology_sequence or {})
            else:
                data["topology_sequence_constraint"] = {
                    "used": True,
                    "stage_sequence": list(topology_sequence or []),
                    "rule": "critical points must be visited in stage_order",
                }

        data.setdefault("topology_constraint_used", True)
        data.setdefault(
            "critical_point_sequence_constraint_used",
            bool(data.get("critical_point_constraint") or
                 data.get("topology_sequence_constraint")))
        data.setdefault("corridor_constraint_used",
                        bool(data.get("corridor_constraint")))
        data.setdefault("manifold_constraint_used",
                        bool(data.get("manifold_constraint")))
        super(TopologyConstraint, self).__init__(data)

    @property
    def critical_point_constraint(self):
        return self.get("critical_point_constraint", {})

    @property
    def corridor_constraint(self):
        return self.get("corridor_constraint", {})

    @property
    def manifold_constraint(self):
        return self.get("manifold_constraint", {})

    @property
    def topology_sequence_constraint(self):
        return self.get("topology_sequence_constraint", {})

    def to_dict(self):
        return dict(self)


class TopologyTubeConstraint(dict):
    """Selected corridor tube hard-constraint payload."""

    def __init__(self, corridor_id="", centerline=None, left_boundary=None,
                 right_boundary=None, tube_width=0.0, valid_region=None,
                 mode="hard", payload=None):
        data = dict(payload or {})
        boundary = data.get("boundary", {}) or {}
        if isinstance(boundary, dict):
            left_boundary = (
                left_boundary if left_boundary is not None else
                boundary.get("left", []))
            right_boundary = (
                right_boundary if right_boundary is not None else
                boundary.get("right", []))
        data.setdefault("used", True)
        data.setdefault("tube_constraint_used", True)
        data.setdefault("tube_constraint_mode", str(mode or "hard"))
        data.setdefault("corridor_id", str(corridor_id or ""))
        data.setdefault("centerline", _as_points(centerline))
        data.setdefault("left_boundary", _as_points(left_boundary))
        data.setdefault("right_boundary", _as_points(right_boundary))
        data.setdefault("tube_width", float(tube_width or 0.0))
        data.setdefault("radius", float(tube_width or 0.0))
        data.setdefault("valid_region", valid_region or {
            "left": data.get("left_boundary", []),
            "right": data.get("right_boundary", []),
        })
        data.setdefault("rule", "x_prediction in selected topology corridor tube")
        super(TopologyTubeConstraint, self).__init__(data)

    def to_dict(self):
        return dict(self)


def _as_points(points):
    if points is None:
        return []
    if isinstance(points, str):
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


def _first_points(*values):
    for value in values:
        pts = _as_points(value)
        if pts:
            return pts
    return []


def _point_from_debug_node(node):
    if not isinstance(node, dict):
        return None
    for key in ("point", "position", "pos"):
        if key in node:
            pts = _as_points([node.get(key)])
            if pts:
                return pts[0]
    if all(k in node for k in ("x", "y")):
        return [
            float(node.get("x", 0.0)),
            float(node.get("y", 0.0)),
            float(node.get("z", 0.0)),
        ]
    return None


def _debug_nodes_by_id(debug):
    out = {}
    for node in (debug or {}).get("nodes", []) or []:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id", ""))
        if node_id:
            out[node_id] = node
    return out


def _critical_from_sequence(node_sequence, debug):
    by_id = _debug_nodes_by_id(debug)
    out = []
    for node_id in [str(x) for x in (node_sequence or [])]:
        node = by_id.get(node_id)
        if not node:
            continue
        kind = str(node.get("kind", node.get("node_type", "")))
        node_type = str(node.get("node_type", kind))
        if kind not in ("saddle", "minimum", "minima") and node_type not in (
                "saddle", "minimum", "minima"):
            continue
        point = _point_from_debug_node(node)
        if point is None:
            continue
        out.append({
            "id": node_id,
            "type": "minimum" if kind == "minima" else kind,
            "point": point,
        })
    return out


def _critical_from_corridor(corridor):
    out = []
    if corridor is None:
        return out
    for idx, item in enumerate(list(getattr(corridor, "morse_nodes", []) or [])):
        if isinstance(item, dict):
            kind = str(item.get("type", item.get("kind", "")))
            point = item.get("point", item.get("position", None))
            node_id = str(item.get("id", "critical_{}".format(idx)))
        else:
            kind = str(getattr(item, "kind", getattr(item, "type", "")))
            point = getattr(item, "point", getattr(item, "position", None))
            node_id = str(getattr(item, "id", "critical_{}".format(idx)))
        if kind not in ("saddle", "minimum", "minima") or point is None:
            continue
        pts = _as_points([point])
        if not pts:
            continue
        out.append({
            "id": node_id,
            "type": "minimum" if kind == "minima" else kind,
            "point": pts[0],
        })
    return out


def _critical_from_selected(selected, debug):
    selected = selected or {}
    existing = selected.get("topology_constraint_info", {}) or {}
    out = []
    explicit_sequence = (
        selected.get("critical_point_sequence") or
        existing.get("critical_point_sequence") or [])
    for idx, item in enumerate(explicit_sequence or []):
        if not isinstance(item, dict):
            continue
        point = item.get("point", item.get("position", None))
        pts = _as_points([point])
        if not pts:
            continue
        try:
            order = int(item.get("order", idx + 1))
        except Exception:
            order = idx + 1
        out.append({
            "id": str(item.get("id", "critical_{}".format(idx))),
            "type": str(item.get("type", item.get("kind", "critical"))),
            "point": pts[0],
            "position": pts[0],
            "order": order,
        })
    if out:
        return sorted(out, key=lambda x: int(x.get("order", 0)))
    for key, kind in (("saddle_points", "saddle"),
                      ("minimum_points", "minimum")):
        for idx, point in enumerate(existing.get(key) or selected.get(key) or []):
            pts = _as_points([point])
            if pts:
                out.append({
                    "id": "{}_{}".format(kind, idx),
                    "type": kind,
                    "point": pts[0],
                    "position": pts[0],
                    "order": len(out) + 1,
                })
    if out:
        return out
    for idx, item in enumerate(selected.get("critical_points", []) or []):
        if not isinstance(item, dict):
            continue
        point = item.get("point", item.get("position", None))
        pts = _as_points([point])
        if pts:
            out.append({
                "id": str(item.get("id", "critical_{}".format(idx))),
                "type": str(item.get("type", item.get("kind", "critical"))),
                "point": pts[0],
                "position": pts[0],
                "order": int(item.get("order", len(out) + 1))
                if str(item.get("order", "")).strip() else len(out) + 1,
            })
    if out:
        return out
    node_sequence = selected.get(
        "node_sequence",
        selected.get("topology_nodes", selected.get("morse_node_ids", [])))
    return _critical_from_sequence(node_sequence, debug)


def _infer_topology_class(centerline):
    pts = _as_points(centerline)
    if len(pts) < 3:
        return "direct_safe_channel"
    start = np.asarray(pts[0], float)[:2]
    goal = np.asarray(pts[-1], float)[:2]
    axis = goal - start
    vals = []
    for p in pts[1:-1]:
        q = np.asarray(p, float)[:2]
        vals.append(float(axis[0] * (q[1] - start[1]) -
                          axis[1] * (q[0] - start[0])))
    mean = float(np.mean(vals)) if vals else 0.0
    if abs(mean) < 1e-6:
        return "direct_safe_channel"
    return "left_bypass" if mean > 0.0 else "right_bypass"


def _constraint_dict(topology_class, node_sequence, centerline, radius,
                     critical_points, safe_threshold=None,
                     manifold_constraint=None, corridor_id="",
                     boundary=None):
    radius = float(radius if radius not in (None, "") else 0.35)
    critical_point_radius = max(0.20, 1.25 * radius)
    normalized_critical = []
    for idx, item in enumerate(critical_points or []):
        if not isinstance(item, dict):
            continue
        point = item.get("point", item.get("position", None))
        pts = _as_points([point])
        if not pts:
            continue
        try:
            order = int(item.get("order", idx + 1))
        except Exception:
            order = idx + 1
        kind = str(item.get("type", item.get("kind", "critical")))
        if kind == "minima":
            kind = "minimum"
        normalized_critical.append({
            "id": str(item.get("id", "critical_{}".format(idx))),
            "type": kind,
            "point": pts[0],
            "position": pts[0],
            "order": order,
        })
    normalized_critical.sort(key=lambda x: int(x.get("order", 0)))
    saddle_points = [
        item["point"] for item in normalized_critical
        if str(item.get("type", "")) == "saddle"
    ]
    minimum_points = [
        item["point"] for item in normalized_critical
        if str(item.get("type", "")) in ("minimum", "minima")
    ]
    status = "feasible"
    if not centerline:
        status = "infeasible_no_centerline"
    elif radius <= 0.0:
        status = "infeasible_no_radius"
    payload = {
        "topology_constraint_used": True,
        "critical_point_sequence_constraint_used": True,
        "corridor_constraint_used": True,
        "manifold_constraint_used": True,
        "topology_class": str(topology_class or _infer_topology_class(centerline)),
        "node_sequence": list(node_sequence or []),
        "corridor_centerline": _as_points(centerline),
        "corridor_radius": radius,
        "critical_point_sequence": normalized_critical,
        "critical_points": normalized_critical,
        "saddle_points": saddle_points,
        "minimum_points": minimum_points,
        "critical_point_radius": critical_point_radius,
        "critical_point_soft_radius": 0.5,
        "critical_point_hard_radius": max(1.0, 2.5 * radius),
        "critical_point_constraint_mode": "soft_hard",
        "safe_threshold": (
            float(safe_threshold) if safe_threshold not in (None, "") else ""),
        "constraint_status": status,
    }
    payload["critical_point_constraint"] = {
        "used": True,
        "mode": payload["critical_point_constraint_mode"],
        "soft_radius": payload["critical_point_soft_radius"],
        "hard_radius": payload["critical_point_hard_radius"],
        "critical_points": normalized_critical,
    }
    payload["corridor_constraint"] = {
        "used": True,
        "centerline": payload["corridor_centerline"],
        "radius": radius,
        "rule": "distance_to_corridor <= corridor_radius",
    }
    payload["topology_tube_constraint"] = TopologyTubeConstraint(
        corridor_id=corridor_id,
        centerline=payload["corridor_centerline"],
        left_boundary=(boundary or {}).get("left", [])
        if isinstance(boundary, dict) else [],
        right_boundary=(boundary or {}).get("right", [])
        if isinstance(boundary, dict) else [],
        tube_width=radius,
        valid_region=boundary or {},
        mode="hard").to_dict()
    payload["corridor_constraint"]["tube_constraint"] = dict(
        payload["topology_tube_constraint"])
    if manifold_constraint is None:
        manifold_constraint = build_manifold_constraint(
            risk_threshold=payload["safe_threshold"])
    payload["manifold_constraint"] = dict(manifold_constraint or {})
    payload["topology_sequence_constraint"] = {
        "used": True,
        "stage_sequence": [
            {
                "id": item["id"],
                "type": item["type"],
                "stage_order": int(item.get("order", idx + 1)),
            }
            for idx, item in enumerate(normalized_critical)
        ],
        "allowed_transitions": [[0, 1], [1, 2], [2, 3]],
        "rule": "critical points must be visited in stage_order",
    }
    return TopologyConstraint(payload)


def _corridor_critical_points(corridor, node_sequence, manifold):
    critical_points = _critical_from_corridor(corridor)
    if critical_points:
        return critical_points
    debug = getattr(manifold, "last_topology_debug", {}) if manifold is not None else {}
    return _critical_from_sequence(node_sequence, debug or {})


def build_topology_constraint(selected_topology_graph=None,
                              selected_corridor=None,
                              critical_points=None,
                              safe_manifold=None,
                              refined_reference=None,
                              safe_threshold=None,
                              minimum_clearance=None,
                              phase=None,
                              robot_type="generic",
                              manifold_constraint_mode=None,
                              phase_params=None):
    corridor = selected_corridor
    refinement_output = (
        getattr(corridor, "refinement_output", {}) if corridor is not None else {})
    if not isinstance(refinement_output, dict):
        refinement_output = {}
    # Keep the selected route's execution tube independent from the MPC
    # reference.  Otherwise every reference is trivially valid against a tube
    # rebuilt from itself and candidate/refinement violations are hidden.
    centerline = _first_points(
        getattr(corridor, "execution_tube_centerline", None)
        if corridor is not None else None,
        refinement_output.get("final_trajectory"),
        refinement_output.get("trajectory"),
        getattr(corridor, "refined_waypoints", None) if corridor is not None else None,
        getattr(corridor, "centerline", None) if corridor is not None else None,
        getattr(corridor, "waypoints", None) if corridor is not None else None,
        refined_reference)
    radius = float(getattr(corridor, "radius", 0.35)) if corridor is not None else 0.35
    topology_class = str(
        getattr(corridor, "topology_route_class",
                getattr(corridor, "topology_class", "")) if corridor is not None else "")
    if not topology_class and isinstance(selected_topology_graph, dict):
        topology_class = str(
            selected_topology_graph.get("topology_class") or
            selected_topology_graph.get("topology_route_class") or "")
    node_sequence = list(
        getattr(corridor, "node_sequence",
                getattr(corridor, "topology_nodes", [])) if corridor is not None else [])
    if not node_sequence and isinstance(selected_topology_graph, dict):
        node_sequence = list(
            selected_topology_graph.get("node_sequence") or
            selected_topology_graph.get("topology_nodes") or [])
    cps = list(critical_points or [])
    if not cps:
        cps = _corridor_critical_points(corridor, node_sequence, safe_manifold)
    threshold = (
        safe_threshold if safe_threshold not in (None, "") else
        float(getattr(safe_manifold, "rho", 1.0)) if safe_manifold is not None else "")
    constraint = _constraint_dict(
        topology_class, node_sequence, centerline, radius,
        cps, safe_threshold=threshold,
        corridor_id=str(getattr(corridor, "corridor_id",
                                getattr(corridor, "label", "")))
        if corridor is not None else "",
        boundary=getattr(corridor, "boundary", {}) if corridor is not None else {},
        manifold_constraint=build_manifold_constraint(
            safe_manifold=safe_manifold,
            selected_corridor=corridor,
            minimum_clearance=minimum_clearance,
            risk_threshold=threshold,
            mode=manifold_constraint_mode,
            phase=phase,
            robot_type=robot_type,
            phase_params=phase_params))
    association = {}
    if corridor is not None:
        association = getattr(corridor, "critical_point_association", None) or {}
    if not association:
        association = associate_critical_points(
            cps, centerline,
            soft_radius=constraint.get("critical_point_soft_radius", 0.5),
            hard_radius=constraint.get("critical_point_hard_radius", max(1.0, 2.5 * radius)))
    if association:
        constraint["critical_point_association"] = association
        constraint["critical_point_association_used"] = bool(
            association.get("critical_point_association_used", False))
        constraint["critical_point_constraint"]["association"] = association
        constraint["topology_sequence_constraint"]["association"] = association
        constraint["topology_sequence_constraint"]["topology_sequence_valid"] = bool(
            association.get("topology_sequence_valid", False))
        constraint["topology_sequence_constraint"]["critical_point_status"] = str(
            association.get("critical_point_status", ""))
    projection = {}
    if corridor is not None:
        projection = getattr(corridor, "critical_point_projection_index", None) or {}
    if not projection and association:
        projection = {
            str(item.get("id", "")): int(item.get("trajectory_index", -1))
            for item in association.get("critical_points", [])
        }
    if projection:
        constraint["critical_point_projection_index"] = dict(projection)
    if corridor is not None:
        for key in (
                "candidate_id", "ik_valid", "link_collision_valid",
                "link_collision", "tube_valid", "arm_pose_optimization_used",
                "arm_ik_candidate_count", "arm_ik_candidate_attempts"):
            if hasattr(corridor, key):
                constraint[key] = getattr(corridor, key)
        ik_validation = getattr(corridor, "ik_validation", None)
        if isinstance(ik_validation, dict) and ik_validation:
            constraint["ik_validation"] = dict(ik_validation)
    return constraint


def constraint_from_corridor(corridor=None, manifold=None, reference_path=None,
                             safe_threshold=None, phase=None,
                             robot_type="generic", phase_params=None):
    return build_topology_constraint(
        selected_corridor=corridor,
        safe_manifold=manifold,
        refined_reference=reference_path,
        safe_threshold=safe_threshold,
        phase=phase,
        robot_type=robot_type,
        phase_params=phase_params)


def constraint_from_selected(selected=None, robot="", debug=None,
                             reference_points=None, safe_threshold=None,
                             phase=None, phase_params=None):
    selected = dict(selected or {})
    debug = debug or {}
    selected_id = str(
        selected.get("corridor_id", selected.get("selected_corridor_id", "")))
    if debug:
        for item in debug.get("candidate_corridors", []) or []:
            if not isinstance(item, dict):
                continue
            ids = [
                str(item.get("corridor_id", "")),
                str(item.get("execution_corridor_id", "")),
                str(item.get("label", "")),
            ]
            if selected_id and selected_id in ids:
                merged = dict(item)
                for key, value in selected.items():
                    if value not in (None, "", [], {}):
                        merged[key] = value
                selected = merged
                break
    existing = selected.get("topology_constraint_info", {}) or {}
    centerline = _first_points(
        existing.get("corridor_centerline"),
        selected.get("refined_waypoints"),
        selected.get("corridor_centerline"),
        selected.get("waypoints"),
        selected.get("raw_topology_waypoints"),
        reference_points)
    radius = selected.get(
        "corridor_radius",
        existing.get("corridor_radius",
                     selected.get("radius", 0.4 if robot == "wheelchair" else 0.35)))
    topology_class = (
        existing.get("topology_class") or
        selected.get("topology_class") or
        selected.get("topology_route_class") or
        _infer_topology_class(centerline))
    node_sequence = selected.get(
        "node_sequence",
        existing.get(
            "node_sequence",
            selected.get("topology_nodes", selected.get("morse_node_ids", []))))
    critical_points = _critical_from_selected(selected, debug or {})
    threshold = (
        safe_threshold if safe_threshold not in (None, "") else
        selected.get("safe_threshold", 7.0 if robot == "wheelchair" else 6.0))
    constraint = _constraint_dict(
        topology_class, node_sequence, centerline, radius,
        critical_points, safe_threshold=threshold,
        corridor_id=selected_id,
        boundary=selected.get("boundary", {}),
        manifold_constraint=build_manifold_constraint(
            selected_corridor=selected,
            risk_threshold=threshold,
            phase=phase or selected.get("phase", selected.get("task_phase", "")),
            robot_type=robot,
            phase_params=phase_params))
    association = (
        selected.get("critical_point_association") or
        existing.get("critical_point_association") or {})
    if not association:
        association = associate_critical_points(
            critical_points, centerline,
            soft_radius=constraint.get("critical_point_soft_radius", 0.5),
            hard_radius=constraint.get("critical_point_hard_radius", max(1.0, 2.5 * float(radius))))
    if association:
        constraint["critical_point_association"] = association
        constraint["critical_point_association_used"] = bool(
            association.get("critical_point_association_used", False))
    projection = (
        selected.get("critical_point_projection_index") or
        existing.get("critical_point_projection_index") or {})
    if projection:
        constraint["critical_point_projection_index"] = dict(projection)
    return constraint


def mpc_inputs_from_constraint(constraint, corridor_id="", waypoints=None):
    constraint = dict(constraint or {})
    centerline = _as_points(constraint.get("corridor_centerline", []))
    radius = float(constraint.get("corridor_radius", 0.35))
    critical_points = list(
        constraint.get("critical_point_sequence") or
        constraint.get("critical_points") or [])
    topology_info = {
        "topology_class": constraint.get("topology_class", ""),
        "node_sequence": list(constraint.get("node_sequence", []) or []),
        "critical_point_sequence": critical_points,
        "critical_points": critical_points,
        "saddle_points": list(constraint.get("saddle_points", []) or []),
        "minimum_points": list(constraint.get("minimum_points", []) or []),
        "corridor_centerline": centerline,
        "corridor_radius": radius,
        "critical_point_radius": float(
            constraint.get("critical_point_radius", max(0.20, 1.25 * radius))),
        "critical_point_soft_radius": float(
            constraint.get("critical_point_soft_radius", 0.5)),
        "critical_point_hard_radius": float(
            constraint.get("critical_point_hard_radius", max(1.0, 2.5 * radius))),
        "critical_point_constraint_mode": str(
            constraint.get("critical_point_constraint_mode", "soft_hard")),
        "critical_point_association": dict(
            constraint.get("critical_point_association", {}) or {}),
        "critical_point_association_used": bool(
            constraint.get("critical_point_association_used", False)),
        "topology_tube_constraint": dict(
            constraint.get("topology_tube_constraint", {}) or {}),
        "constraint_status": constraint.get("constraint_status", ""),
    }
    corridor_info = {
        "corridor_id": str(corridor_id or ""),
        "centerline": centerline,
        "radius": radius,
        "topology_tube_constraint": dict(
            constraint.get("topology_tube_constraint", {}) or {}),
        "waypoints": _as_points(waypoints) or centerline,
        "topology_route_class": constraint.get("topology_class", ""),
        "critical_point_radius": topology_info["critical_point_radius"],
        "critical_point_soft_radius": topology_info["critical_point_soft_radius"],
        "critical_point_hard_radius": topology_info["critical_point_hard_radius"],
        "critical_point_constraint_mode": topology_info["critical_point_constraint_mode"],
        "critical_point_association": topology_info["critical_point_association"],
        "critical_point_association_used": topology_info["critical_point_association_used"],
    }
    for key in (
            "candidate_id", "ik_valid", "link_collision_valid",
            "link_collision", "tube_valid", "arm_pose_optimization_used",
            "arm_ik_candidate_count", "arm_ik_candidate_attempts",
            "ik_validation"):
        if key in constraint:
            corridor_info[key] = constraint.get(key)
            topology_info[key] = constraint.get(key)
    manifold_info = {
        "safe_threshold": float(
            constraint.get("safe_threshold")
            if constraint.get("safe_threshold") not in (None, "") else 1.0),
        "risk_boundary": [],
        "distance_function": "safe_threshold_minus_phi_s",
    }
    manifold_constraint = dict(constraint.get("manifold_constraint", {}) or {})
    if manifold_constraint:
        topology_info["manifold_constraint"] = manifold_constraint
        manifold_info.update(manifold_constraint)
        manifold_info["safe_threshold"] = float(
            manifold_constraint.get(
                "safe_threshold",
                manifold_constraint.get(
                    "risk_threshold", manifold_info["safe_threshold"])))
    return topology_info, corridor_info, manifold_info, constraint


def write_topology_constraint(path, constraint):
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    with open(path, "w") as f:
        json.dump(constraint, f, indent=2, sort_keys=True)
    manifold = dict((constraint or {}).get("manifold_constraint", {}) or {})
    if manifold:
        manifold_path = os.path.join(directory or ".", "manifold_constraint.json")
        payload = dict(manifold)
        payload.setdefault("constraint_used", bool(payload.get("used", True)))
        with open(manifold_path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
