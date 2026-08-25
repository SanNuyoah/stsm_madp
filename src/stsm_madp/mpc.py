import sys
sys.dont_write_bytecode = True

import numpy as np
import importlib
import csv
import json
import os
import time

from stsm_madp.interest_points import forbidden_anchor_hit, pose_interest_risk
from stsm_madp.critical_point_tracker import CriticalPointTracker
from stsm_madp.topology_stage_tracker import TopologyStageTracker
from stsm_madp.topology_constraint import (
    build_topology_constraint, mpc_inputs_from_constraint)
from stsm_madp.manifold_constraint import distance_to_manifold_boundary
from stsm_madp.manifold_constraint import assert_manifold_mode_consistency
from stsm_madp.manifold_constraint_evaluator import ManifoldConstraintEvaluator
from stsm_madp.topology_ik_solver import TopologyIKSolver
from stsm_madp.task_config import resolve_task_mode, resolve_task_weight
from stsm_madp.task_semantics import infer_task_state


DEFAULT_MPC_CONFIG = {
    "enabled": True,
    "horizon": 10,
    "dt": 0.1,
    "task_mode": "",
    "task_config": {},
    "task_weight": {},
    "phase_cost_weights": {
        "approach": {
            "risk_multiplier": 1.8,
            "tracking_multiplier": 1.0,
            "smooth_multiplier": 1.0,
            "task_multiplier": 1.0,
        },
        "handover": {
            "risk_multiplier": 0.7,
            "tracking_multiplier": 3.0,
            "smooth_multiplier": 1.0,
            "task_multiplier": 3.0,
        },
        "return": {
            "risk_multiplier": 1.8,
            "tracking_multiplier": 1.0,
            "smooth_multiplier": 1.0,
            "task_multiplier": 1.0,
        },
        "navigation": {
            "risk_multiplier": 1.5,
            "tracking_multiplier": 1.0,
            "smooth_multiplier": 1.5,
            "task_multiplier": 1.0,
        },
    },
    "weights": {
        "track": 1.0,
        "control": 0.1,
        "smooth": 0.2,
        "risk": 1.0,
        "topology": 5.0,
        "corridor": 10.0,
        "manifold": 10.0,
        "safety_violation": 5.0,
        "constraint_violation": 10.0,
        "handover": 1.0,
    },
    "manifold_constraint_mode": "soft",
    "manifold_soft_tolerance": 0.005,
    "manifold_hard_tolerance": 0.25,
    "wheelchair": {
        "v_max": 0.5,
        "v_min": 0.0,
        "omega_max": 1.0,
        "a_max": 0.5,
        "alpha_max": 1.0,
        "curvature_max": 1.5,
    },
    "arm": {
        "joint_velocity_max": 0.5,
        "ee_speed_max": 0.3,
        "control_delta_max": 0.1,
        "risk_threshold": 1.0,
        "manifold_constraint_mode": "soft",
        "manifold_soft_tolerance": 0.005,
        "manifold_hard_tolerance": 0.25,
    },
}


PHASE_ALIASES = {
    "0": "approach",
    "1": "approach",
    "2": "approach",
    "3": "handover",
    "4": "return",
    "approach": "approach",
    "handover": "handover",
    "return": "return",
    "navigation": "navigation",
    "nav": "navigation",
}


def _merge_dict(base, override):
    out = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out.get(key), value)
        else:
            out[key] = value
    return out


def query_social_risk(point, robot_type, social_field, fallback_value=0.0):
    p = np.asarray(point, float)
    if p.size < 3:
        p = np.array([p[0], p[1], 0.0], float)
    if social_field is None:
        return {
            "risk_value": float(fallback_value),
            "risk_query_valid": False,
            "risk_query_source": "fallback_zero",
            "risk_query_failure_reason": "social_field_unavailable",
        }
    try:
        source = str(getattr(social_field, "risk_query_source", "") or "")
        if not source:
            source = "social_field_phi" if hasattr(social_field, "phi_s") else "risk_grid_interpolation"
        if hasattr(social_field, "phi_s"):
            value = float(social_field.phi_s(p))
        elif callable(social_field):
            value = float(social_field(p, robot_type=robot_type))
        elif hasattr(social_field, "query"):
            value = float(social_field.query(p))
        else:
            return {
                "risk_value": float(fallback_value),
                "risk_query_valid": False,
                "risk_query_source": "unavailable",
                "risk_query_failure_reason": "unsupported_risk_evaluator",
            }
        return {
            "risk_value": max(0.0, value),
            "risk_query_valid": True,
            "risk_query_source": source,
            "risk_query_failure_reason": "none",
        }
    except Exception as exc:
        return {
            "risk_value": float(fallback_value),
            "risk_query_valid": False,
            "risk_query_source": "unavailable",
            "risk_query_failure_reason": "{}:{}".format(
                type(exc).__name__, str(exc)[:120]),
        }


def _as_points(reference_path):
    if isinstance(reference_path, (list, tuple)) and reference_path:
        if isinstance(reference_path[0], dict):
            rows = []
            for item in reference_path:
                if not isinstance(item, dict):
                    continue
                rows.append([
                    item.get("x", item.get("ref_x", 0.0)),
                    item.get("y", item.get("ref_y", 0.0)),
                    item.get("z", item.get("ref_z", 0.0)),
                ])
            reference_path = rows
    pts = np.asarray(reference_path, float)
    if pts.size == 0:
        return np.zeros((0, 3), float)
    if pts.ndim == 1:
        pts = pts.reshape((1, pts.shape[0]))
    if pts.shape[1] == 2:
        z = np.zeros((pts.shape[0], 1), float)
        pts = np.hstack([pts, z])
    return pts[:, :3]


def _phase_name(value, default="approach"):
    text = str(value if value is not None else "").strip().lower()
    return PHASE_ALIASES.get(text, default)


def _reference_phases(reference_path, count, robot_type):
    phases = []
    if isinstance(reference_path, (list, tuple)):
        for item in reference_path:
            if isinstance(item, dict):
                phases.append(_phase_name(item.get("phase", item.get("task_phase"))))
            elif isinstance(item, (list, tuple)) and len(item) >= 4:
                phases.append(_phase_name(item[3]))
    if len(phases) >= int(count):
        return phases[:int(count)]
    robot = str(robot_type or "").lower()
    count = int(count)
    if count <= 0:
        return []
    inferred = []
    for idx in range(count):
        progress = float(idx) / float(max(count - 1, 1))
        if robot == "arm":
            if progress < 0.45:
                inferred.append("approach")
            elif progress < 0.70:
                inferred.append("handover")
            else:
                inferred.append("return")
        else:
            inferred.append("navigation")
    return inferred


def _phase_local_progress(phase_sequence, index):
    phase_sequence = list(phase_sequence or [])
    if not phase_sequence:
        return 0.0
    index = int(max(0, min(int(index), len(phase_sequence) - 1)))
    phase = phase_sequence[index]
    indices = [i for i, item in enumerate(phase_sequence) if item == phase]
    if len(indices) <= 1:
        return 1.0
    first = int(indices[0])
    last = int(indices[-1])
    return float(np.clip(
        float(index - first) / float(max(last - first, 1)), 0.0, 1.0))


ARM_PHASE_CLEARANCE_SCHEDULE = {
    "approach": {
        "start_clearance": 0.35,
        "end_clearance": 0.20,
    },
    "handover": {
        "clearance": 0.15,
    },
    "return": {
        "start_clearance": 0.20,
        "end_clearance": 0.35,
    },
}


def _phase_clearance_schedule(context):
    constraint = dict((context or {}).get("manifold_constraint", {}) or {})
    schedule = dict(
        constraint.get("phase_clearance_schedule") or
        (context or {}).get("phase_clearance_schedule") or {})
    robot = str(
        constraint.get("robot_type", (context or {}).get("robot_type", ""))
    ).strip().lower()
    if not schedule and robot == "arm":
        schedule = dict(ARM_PHASE_CLEARANCE_SCHEDULE)
    return schedule


def _phase_clearance_threshold(context, phase, progress):
    constraint = dict((context or {}).get("manifold_constraint", {}) or {})
    default_phase = "navigation" if str(
        constraint.get("robot_type", (context or {}).get("robot_type", ""))
    ).strip().lower() == "wheelchair" else "approach"
    phase = _phase_name(phase, default=default_phase)
    progress = float(np.clip(float(progress or 0.0), 0.0, 1.0))
    schedule = _phase_clearance_schedule(context)
    entry = dict(schedule.get(phase, {}) or {})
    if phase == "approach":
        start = float(entry.get("start_clearance", 0.35))
        end = float(entry.get("end_clearance", 0.20))
        return float(start + (end - start) * progress)
    if phase == "handover":
        return float(entry.get(
            "interaction_clearance", entry.get("clearance", 0.15)))
    if phase == "return":
        start = float(entry.get(
            "start_clearance", entry.get(
                "handover_clearance", entry.get("clearance_start", 0.20))))
        end = float(entry.get(
            "end_clearance", entry.get(
                "clearance_threshold", entry.get("clearance", 0.35))))
        return float(start + (end - start) * progress)
    return float(
        constraint.get(
            "effective_minimum_clearance",
            constraint.get(
                "effective_min_clearance",
                (context or {}).get("minimum_clearance", 0.0))) or 0.0)


def audit_reference_safety(reference_path, context, robot_type="",
                           phase_sequence=None, planning_margin=0.0):
    pts = _as_optional_points(reference_path)
    robot = str(robot_type or (context or {}).get("robot_type", "")).lower()
    phases = list(phase_sequence or _reference_phases(pts, len(pts), robot))
    worst = {
        "min_clearance": float("inf"),
        "required_min_clearance": 0.0,
        "worst_phase": "",
        "worst_index": -1,
        "worst_interest_point": "",
    }
    violations = 0
    records = []
    for idx, point in enumerate(pts):
        phase = phases[min(idx, len(phases) - 1)] if phases else (
            "navigation" if robot == "wheelchair" else "approach")
        progress = _phase_local_progress(phases, idx)
        required = float(
            _phase_clearance_threshold(context, phase, progress) +
            max(0.0, float(planning_margin or 0.0)))
        evaluator = ManifoldConstraintEvaluator(
            manifold_constraint={
                "boundary": (context or {}).get("manifold_boundary", []),
                "minimum_clearance": required,
                "effective_minimum_clearance": required,
                "risk_threshold": (context or {}).get("safe_threshold", 1.0),
                "effective_risk_threshold": (context or {}).get("safe_threshold", 1.0),
            },
            corridor_constraint={
                "centerline": (context or {}).get("centerline", []),
                "radius": (context or {}).get("radius", 0.0),
            },
            risk_field=(context or {}).get("social_field"),
            planning_clearance_margin=0.0,
            soft_tolerance=(context or {}).get("manifold_soft_tolerance", 0.005),
            hard_tolerance=(context or {}).get("manifold_hard_tolerance", 0.25))
        status = evaluator.evaluate_state_or_points(
            state=point, interest_points=[point], robot_type=robot,
            task_phase=phase, phase_progress=progress,
            effective_minimum_clearance=required)
        clearance = float(status.get("min_clearance", 0.0))
        margin = float(status.get("clearance_margin", clearance - required))
        violated = bool(margin < -1e-9 or not bool(status.get("valid", False)))
        if violated:
            violations += 1
        if clearance < float(worst["min_clearance"]):
            worst.update({
                "min_clearance": clearance,
                "required_min_clearance": required,
                "worst_phase": phase,
                "worst_index": int(idx),
                "worst_interest_point": str(status.get(
                    "worst_interest_point", "")),
            })
        records.append({
            "trajectory_source": "reference",
            "index": int(idx),
            "phase": phase,
            "progress": float(progress),
            "min_clearance": clearance,
            "required_min_clearance": required,
            "clearance_margin": margin,
            "violation": bool(violated),
            "worst_interest_point": str(status.get(
                "worst_interest_point", "")),
        })
    if not np.isfinite(float(worst["min_clearance"])):
        worst["min_clearance"] = 0.0
    return {
        "feasible": bool(len(pts) > 0 and violations == 0),
        "min_clearance": float(worst["min_clearance"]),
        "required_min_clearance": float(worst["required_min_clearance"]),
        "violation_count": int(violations),
        "worst_phase": str(worst["worst_phase"]),
        "worst_index": int(worst["worst_index"]),
        "worst_interest_point": str(worst["worst_interest_point"]),
        "records": records,
    }


def _phase_weight_entry(weights, phase):
    phase = _phase_name(phase)
    phase_cfg = dict((weights or {}).get("phase_cost_weights", {}) or {})
    defaults = DEFAULT_MPC_CONFIG["phase_cost_weights"]
    cfg = _merge_dict(defaults.get(phase, {}), phase_cfg.get(phase, {}))
    risk_weight = float(weights.get("risk", 1.0)) * float(
        cfg.get("risk_multiplier", 1.0))
    tracking_weight = float(weights.get("track", 1.0)) * float(
        cfg.get("tracking_multiplier", 1.0))
    smoothness_weight = float(weights.get("smooth", 0.2)) * float(
        cfg.get("smooth_multiplier", cfg.get("smoothness_multiplier", 1.0)))
    task_base = float(weights.get("task", weights.get("handover", 1.0)))
    task_weight = task_base * float(cfg.get("task_multiplier", 1.0))
    out = dict(weights or {})
    out["risk"] = risk_weight
    out["track"] = tracking_weight
    out["smooth"] = smoothness_weight
    out["handover"] = task_weight
    out["task"] = task_weight
    out["phase"] = phase
    out["risk_weight"] = risk_weight
    out["tracking_weight"] = tracking_weight
    out["smoothness_weight"] = smoothness_weight
    out["task_weight_effective"] = task_weight
    return out


def _as_optional_points(points):
    if points is None:
        return np.zeros((0, 3), float)
    if isinstance(points, str) and not points:
        return np.zeros((0, 3), float)
    try:
        return _as_points(points)
    except Exception:
        return np.zeros((0, 3), float)


def _polyline_project(point, centerline):
    p = np.asarray(point, float)
    pts = _as_optional_points(centerline)
    if pts.size == 0:
        return p[:min(3, p.size)], 0.0, 0.0
    dim = min(p.size, pts.shape[1])
    p = p[:dim]
    wps = pts[:, :dim]
    if len(wps) == 1:
        return wps[0], float(np.linalg.norm(p - wps[0])), 0.0
    best_pt = wps[0]
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
            best_d = d
            best_pt = q
            best_s = acc + t * seg_len
        acc += seg_len
    return best_pt, best_d, best_s


def _boundary_points(boundary):
    if boundary is None or isinstance(boundary, str):
        return np.zeros((0, 3), float)
    if isinstance(boundary, dict):
        pts = []
        for key in ("left", "right", "boundary", "points"):
            pts.extend(boundary.get(key, []) or [])
        return _as_optional_points(pts)
    return _as_optional_points(boundary)


def _point_to_polyline_distance(point, boundary):
    return distance_to_manifold_boundary(point, boundary)


def _has_points(value):
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value)
    try:
        return np.asarray(value).size > 0
    except Exception:
        return False


def _first_point_source(*values):
    for value in values:
        if _has_points(value):
            return value
    return []


def generate_topology_tube(corridor_centerline, corridor_radius,
                           samples_per_segment=8):
    centerline = _as_optional_points(corridor_centerline)
    radius = float(corridor_radius if corridor_radius not in (None, "") else 0.0)
    tube_points = []
    if len(centerline) == 1:
        tube_points.append(centerline[0].tolist())
    elif len(centerline) > 1:
        samples = max(1, int(samples_per_segment))
        for a, b in zip(centerline[:-1], centerline[1:]):
            for idx in range(samples + 1):
                if tube_points and idx == 0:
                    continue
                alpha = float(idx) / float(samples)
                tube_points.append((a + alpha * (b - a)).tolist())
    return {
        "centerline": centerline.tolist(),
        "radius": radius,
        "tube_points": tube_points,
    }


def _extract_constraint_context(topology_info, corridor_info, manifold_info,
                                reference_path, social_field, constraints):
    topology_info = dict(topology_info or {})
    corridor_info = dict(corridor_info or {})
    manifold_info = dict(manifold_info or {})
    centerline = _first_point_source(
        corridor_info.get("centerline"),
        corridor_info.get("corridor_centerline"),
        topology_info.get("corridor_centerline"),
        corridor_info.get("waypoints"),
        reference_path)
    radius = (
        corridor_info.get("radius")
        if corridor_info.get("radius") not in (None, "") else
        topology_info.get("corridor_radius"))
    if radius in (None, ""):
        radius = constraints.get("corridor_radius", 0.35)
    safe_threshold = manifold_info.get("safe_threshold")
    if safe_threshold in (None, ""):
        safe_threshold = constraints.get(
            "manifold_safe_threshold",
            constraints.get("risk_threshold", 1.0))
    manifold_constraint = dict(
        manifold_info.get(
            "manifold_constraint",
            topology_info.get("manifold_constraint", manifold_info)) or {})
    risk_threshold = manifold_constraint.get(
        "effective_risk_threshold",
        manifold_constraint.get(
            "effective_safe_threshold",
            manifold_constraint.get(
                "risk_threshold",
                manifold_constraint.get("safe_threshold", safe_threshold))))
    minimum_clearance = manifold_constraint.get(
        "effective_minimum_clearance",
        manifold_constraint.get(
            "effective_min_clearance",
            manifold_constraint.get(
                "minimum_clearance",
                manifold_constraint.get(
                    "min_clearance",
                    constraints.get("minimum_clearance", 0.0)))))
    boundary = manifold_constraint.get(
        "boundary",
        manifold_info.get("boundary", manifold_info.get("risk_boundary", [])))
    mode = assert_manifold_mode_consistency(
        constraints.get(
        "mpc_manifold_constraint_mode",
        constraints.get(
            "manifold_constraint_mode",
            manifold_constraint.get(
                "manifold_constraint_mode",
                manifold_constraint.get("mode", None)))),
        constraints.get("manifold_constraint_mode"),
        manifold_constraint.get("manifold_constraint_mode"),
        manifold_constraint.get("mode"))
    return {
        "topology_info": topology_info,
        "corridor_info": corridor_info,
        "manifold_info": manifold_info,
        "social_field": social_field,
        "centerline": _as_optional_points(centerline),
        "reference_path": _as_optional_points(reference_path),
        "radius": float(radius),
        "safe_threshold": float(risk_threshold),
        "manifold_constraint": manifold_constraint,
        "robot_type": str(manifold_constraint.get(
            "robot_type", constraints.get("robot_type", ""))),
        "manifold_constraint_mode": mode or "soft",
        "strict_risk_query": bool(constraints.get(
            "strict_risk_query", (mode == "hard"))),
        "minimum_clearance": float(minimum_clearance or 0.0),
        "nominal_minimum_clearance": float(manifold_constraint.get(
            "minimum_clearance", minimum_clearance or 0.0) or 0.0),
        "nominal_safe_threshold": float(manifold_constraint.get(
            "risk_threshold",
            manifold_constraint.get("safe_threshold", risk_threshold)) or 0.0),
        "manifold_boundary": boundary,
        "topology_tube_constraint": dict(
            corridor_info.get(
                "topology_tube_constraint",
                topology_info.get("topology_tube_constraint", {})) or {}),
        "topology_class": str(
            topology_info.get("topology_class") or
            corridor_info.get("topology_route_class") or
            corridor_info.get("topology_class") or ""),
        "node_sequence": list(topology_info.get(
            "node_sequence", corridor_info.get("node_sequence", [])) or []),
        "critical_point_sequence": list(
            topology_info.get(
                "critical_point_sequence",
                topology_info.get("critical_points", [])) or []),
        "critical_point_association": dict(
            topology_info.get(
                "critical_point_association",
                corridor_info.get("critical_point_association", {})) or {}),
        "critical_point_association_used": bool(
            topology_info.get(
                "critical_point_association_used",
                corridor_info.get("critical_point_association_used", False))),
        "risk_boundary": manifold_info.get("risk_boundary", []),
        "distance_function": manifold_info.get(
            "distance_function", "risk_phi_minus_safe_threshold"),
        "critical_point_radius": float(
            topology_info.get(
                "critical_point_radius",
                corridor_info.get(
                    "critical_point_radius",
                    constraints.get("critical_point_radius", 0.20)))),
        "critical_point_constraint_mode": str(
            constraints.get(
                "critical_point_constraint_mode",
                topology_info.get("critical_point_constraint_mode", "soft_hard"))
        ).strip().lower(),
        "critical_point_soft_radius": float(
            topology_info.get(
                "critical_point_soft_radius",
                corridor_info.get(
                    "critical_point_soft_radius",
                    constraints.get(
                        "critical_point_soft_radius",
                        topology_info.get(
                            "critical_point_radius",
                            corridor_info.get(
                                "critical_point_radius",
                                constraints.get("critical_point_radius", 0.40))))))),
        "critical_point_hard_radius": float(
            topology_info.get(
                "critical_point_hard_radius",
                corridor_info.get(
                    "critical_point_hard_radius",
                    constraints.get(
                        "critical_point_hard_radius",
                        max(1.0, 2.5 * float(radius)))))),
        "corridor_hard_tolerance": float(
            constraints.get("corridor_hard_tolerance", 1e-6)),
        "manifold_soft_tolerance": float(
            constraints.get("manifold_soft_tolerance", 0.08)),
        "manifold_hard_tolerance": float(
            constraints.get("manifold_hard_tolerance", 0.25)),
        "override_replan_limit": int(
            constraints.get("override_replan_limit",
                            constraints.get("manifold_override_replan_limit", 4))),
    }


def _signed_lateral(point, centerline):
    pts = _as_optional_points(centerline)
    if len(pts) < 2:
        return 0.0
    p = np.asarray(point, float)[:2]
    start = pts[0, :2]
    goal = pts[-1, :2]
    axis = goal - start
    denom = float(np.linalg.norm(axis))
    if denom <= 1e-12:
        return 0.0
    return float((axis[0] * (p[1] - start[1]) -
                  axis[1] * (p[0] - start[0])) / denom)


def _topology_step_penalty(point, context):
    topology_class = str(context.get("topology_class", "") or "")
    if topology_class not in ("left_bypass", "right_bypass",
                              "direct_safe_channel", "center_passage"):
        return 0.0, 0
    centerline = context.get("centerline")
    radius = float(context.get("radius", 0.0))
    _q, dist_to_corridor, _s = _polyline_project(point, centerline)
    if radius > 0.0 and dist_to_corridor <= radius + float(
            context.get("corridor_hard_tolerance", 1e-6)):
        return 0.0, 0
    signed = _signed_lateral(point, centerline)
    margin = max(0.05, 0.25 * radius)
    if topology_class == "left_bypass":
        violation = max(0.0, -signed + margin)
    elif topology_class == "right_bypass":
        violation = max(0.0, signed + margin)
    else:
        violation = max(0.0, abs(signed) - max(radius, margin))
    return float(violation ** 2), int(violation > 1e-9)


def _critical_points_from_context(context):
    points = []
    seen = set()
    def add_point(node_id, kind, point):
        pts = _as_optional_points([point])
        if not len(pts):
            return
        q = pts[0]
        key = "{:.6f},{:.6f},{:.6f}".format(float(q[0]), float(q[1]), float(q[2]))
        id_key = str(node_id or "")
        if key in seen or (id_key and id_key in seen):
            return
        seen.add(key)
        if id_key:
            seen.add(id_key)
        order = len(points) + 1
        points.append({
            "id": id_key or "{}_{}".format(kind, len(points)),
            "type": kind,
            "point": q,
            "position": q,
            "order": order,
        })
    explicit = []
    for idx, item in enumerate(context.get("critical_point_sequence", []) or []):
        if not isinstance(item, dict):
            continue
        point = item.get("point", item.get("position", None))
        pts = _as_optional_points([point])
        if not len(pts):
            continue
        try:
            order = int(item.get("order", idx + 1))
        except Exception:
            order = idx + 1
        explicit.append({
            "id": str(item.get("id", "critical_{}".format(idx))),
            "type": str(item.get("type", item.get("kind", "critical"))),
            "point": pts[0],
            "position": pts[0],
            "order": order,
        })
    if explicit:
        explicit.sort(key=lambda x: int(x.get("order", 0)))
        return explicit
    for key, kind in (("saddle_points", "saddle"),
                      ("minimum_points", "minimum")):
        for idx, point in enumerate(context.get("topology_info", {}).get(key, []) or []):
            add_point("{}_{}".format(kind, idx), kind, point)
    for idx, item in enumerate(
            context.get("topology_info", {}).get("critical_points", []) or []):
        if not isinstance(item, dict):
            continue
        point = item.get("point", item.get("position", None))
        if point is None:
            continue
        add_point(
            str(item.get("id", "critical_{}".format(idx))),
            str(item.get("type", item.get("kind", "critical"))),
            point)
    return points


def _critical_sequence_valid(context):
    sequence = _critical_points_from_context(context)
    orders = []
    ids = set()
    for item in sequence:
        try:
            order = int(item.get("order", 0))
        except Exception:
            return False, "critical_point_order_invalid"
        if order <= 0 or order in orders:
            return False, "critical_point_order_invalid"
        point_id = str(item.get("id", ""))
        if not point_id or point_id in ids:
            return False, "critical_point_sequence_invalid"
        ids.add(point_id)
        orders.append(order)
    if sequence and orders != sorted(orders):
        return False, "critical_point_order_invalid"
    return bool(sequence), "" if sequence else "critical_point_sequence_missing"


def _annotate_critical_sequence(rows, context):
    rows = list(rows or [])
    sequence = _critical_points_from_context(context)
    valid_input, input_reason = _critical_sequence_valid(context)
    soft_radius = float(context.get(
        "critical_point_soft_radius",
        context.get("critical_point_radius", 0.20)))
    hard_radius = max(
        soft_radius,
        float(context.get("critical_point_hard_radius", soft_radius)))
    tracker = CriticalPointTracker(
        sequence, threshold=hard_radius)
    invalid_count = 0
    for row in rows:
        point = [
            float(row.get("pred_x", 0.0) or 0.0),
            float(row.get("pred_y", 0.0) or 0.0),
            float(row.get("pred_z", 0.0) or 0.0),
        ]
        status = tracker.update(point) if valid_input else tracker.status()
        if not valid_input:
            status["sequence_valid"] = False
            status["invalid_reason"] = input_reason
        row["critical_point_target"] = status.get("critical_point_target", "")
        row["critical_point_distance"] = float(
            status.get("critical_point_distance", row.get(
                "critical_point_distance", 0.0)) or 0.0)
        cp_distance = float(row["critical_point_distance"])
        if cp_distance <= soft_radius:
            cp_status = "feasible"
        elif cp_distance <= hard_radius:
            cp_status = "soft_violation"
        else:
            cp_status = "hard_violation"
        row["critical_point_constraint_status"] = cp_status
        row["critical_sequence_state"] = status.get(
            "critical_sequence_state", status.get("state", "INIT"))
        row["topology_sequence_valid"] = bool(status.get("sequence_valid", False))
        row["critical_point_sequence_valid"] = bool(
            status.get("sequence_valid", False))
        row["current_target_critical_point"] = status.get(
            "current_target", row.get("critical_point_target", ""))
        row["critical_points_passed"] = json.dumps(
            status.get("passed_points", []))
        if not bool(row["topology_sequence_valid"]):
            invalid_count += 1
    final_status = tracker.status()
    if not valid_input:
        final_status["sequence_valid"] = False
        final_status["invalid_reason"] = input_reason
    return {
        "topology_sequence_constraint_used": True,
        "critical_point_sequence_constraint_used": True,
        "critical_point_sequence_valid": bool(final_status.get("sequence_valid", False)),
        "critical_sequence_status": {
            "current_target": str(final_status.get("current_target", "")),
            "passed_points": list(final_status.get("passed_points", [])),
            "sequence_valid": bool(final_status.get("sequence_valid", False)),
        },
        "critical_points_passed": list(final_status.get("passed_points", [])),
        "current_target_critical_point": str(final_status.get("current_target", "")),
        "critical_sequence_state": str(final_status.get("state", "INIT")),
        "sequence_progress": int(len(final_status.get("passed_points", []))),
        "critical_point_constraint_mode": str(context.get(
            "critical_point_constraint_mode", "soft_hard")),
        "critical_point_soft_radius": float(soft_radius),
        "critical_point_hard_radius": float(hard_radius),
        "critical_sequence_invalid_reason": str(
            final_status.get("invalid_reason", "")),
        "topology_infeasible_count": int(
            invalid_count + (0 if bool(final_status.get("sequence_valid", False)) else 1)),
    }


def _nearest_critical(point, context):
    critical = _critical_points_from_context(context)
    if not critical:
        return "", 0.0, "not_applicable"
    p = np.asarray(point, float)[:3]
    best = None
    best_d = float("inf")
    for item in critical:
        q = np.asarray(item["point"], float)[:3]
        d = float(np.linalg.norm(p - q))
        if d < best_d:
            best = item
            best_d = d
    soft_radius = float(context.get(
        "critical_point_soft_radius",
        context.get("critical_point_radius", 0.20)))
    hard_radius = max(
        soft_radius,
        float(context.get("critical_point_hard_radius", soft_radius)))
    if best_d <= soft_radius:
        status = "feasible"
    elif best_d <= hard_radius:
        status = "soft_violation"
    else:
        status = "hard_violation"
    return str(best.get("id", "")), float(best_d), status


def _status_from_violation(violation, hard_tol):
    v = float(violation)
    if v <= 0.0:
        return "feasible"
    if v <= float(hard_tol):
        return "soft_violation"
    return "infeasible"


def _apply_corridor_hard_constraint(point, context):
    centerline = context.get("centerline")
    radius = float(context.get("radius", 0.0))
    p = np.asarray(point, float).copy()
    q, dist, _s = _polyline_project(p, centerline)
    if radius <= 0.0 or dist <= radius:
        return p
    dim = min(len(q), len(p))
    delta = p[:dim] - q[:dim]
    norm = float(np.linalg.norm(delta))
    if norm <= 1e-12:
        return p
    p[:dim] = q[:dim] + delta * (radius / norm)
    return p


def _apply_manifold_hard_constraint(point, robot, field, context):
    threshold = float(context.get("safe_threshold", 1.0))
    best = np.asarray(point, float).copy()
    best_info = query_social_risk(best, robot, field)
    best_risk = float(best_info.get("risk_value", 0.0))
    if best_risk <= threshold:
        return best, best_info

    centerline = context.get("centerline")
    radius = max(0.0, float(context.get("radius", 0.0)))
    q, _dist, _s = _polyline_project(best, centerline)
    candidates = [np.asarray(q, float)]
    if radius > 1e-9:
        for scale in (0.25, 0.5, 0.75, 1.0):
            r = radius * scale
            for angle in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False):
                cand = np.asarray(q, float).copy()
                cand[0] += r * float(np.cos(angle))
                cand[1] += r * float(np.sin(angle))
                if len(cand) > 2 and len(best) > 2:
                    cand[2] = best[2]
                candidates.append(cand)
    for cand in candidates:
        info = query_social_risk(cand, robot, field)
        risk = float(info.get("risk_value", 0.0))
        if risk < best_risk:
            best = cand
            best_info = info
            best_risk = risk
            if best_risk <= threshold:
                break
    return best, best_info


def _apply_hard_constraints(point, robot, field, context):
    constrained = _apply_corridor_hard_constraint(point, context)
    constrained, risk_info = _apply_manifold_hard_constraint(
        constrained, robot, field, context)
    constrained = _apply_corridor_hard_constraint(constrained, context)
    risk_info = query_social_risk(constrained, robot, field)
    return constrained, risk_info


def _constraint_step_metrics(point, risk_value, context, phase=None,
                             progress=0.0):
    centerline = context.get("centerline")
    radius = float(context.get("radius", 0.0))
    safe_threshold = float(context.get("safe_threshold", 1.0))
    minimum_clearance = (
        _phase_clearance_threshold(context, phase, progress)
        if phase not in (None, "") else
        float(context.get("minimum_clearance", 0.0) or 0.0))
    boundary = context.get("manifold_boundary", [])
    tube = dict(context.get("topology_tube_constraint", {}) or {})
    corridor_constraint = {
        "centerline": centerline,
        "radius": radius,
    }
    if tube:
        corridor_constraint.update({
            "centerline": tube.get("centerline", centerline),
            "radius": tube.get("radius", tube.get("tube_width", radius)),
            "tube_width": tube.get("tube_width", radius),
            "left_boundary": tube.get("left_boundary", []),
            "right_boundary": tube.get("right_boundary", []),
        })
    evaluator = ManifoldConstraintEvaluator(
        manifold_constraint={
            "boundary": boundary,
            "minimum_clearance": minimum_clearance,
            "risk_threshold": safe_threshold,
            "safe_threshold": safe_threshold,
        },
        corridor_constraint=corridor_constraint,
        risk_field=context.get("social_field"),
        planning_clearance_margin=0.0,
        soft_tolerance=context.get("manifold_soft_tolerance", 0.08),
        hard_tolerance=context.get("manifold_hard_tolerance", 0.25))
    safety = evaluator.evaluate_state(point)
    distance_to_corridor = float(safety.get("corridor_distance", 0.0))
    active_radius = float(safety.get("corridor_radius", radius))
    corridor_violation = max(0.0, distance_to_corridor - active_radius)
    boundary_distance = float(safety.get("clearance", 0.0))
    boundary_available = True
    risk_margin = safe_threshold - float(risk_value)
    clearance_margin = boundary_distance - minimum_clearance
    distance_to_manifold = min(risk_margin, clearance_margin)
    risk_violation = max(0.0, -risk_margin)
    clearance_violation = max(0.0, -clearance_margin)
    manifold_violation = max(float(risk_violation), float(clearance_violation))
    topology_cost, topology_violated = _topology_step_penalty(point, context)
    nearest_cp, cp_dist, cp_status = _nearest_critical(point, context)
    corridor_status = _status_from_violation(
        corridor_violation, context.get("corridor_hard_tolerance", 1e-6))
    mode = str(context.get("manifold_constraint_mode", "hard")).lower()
    violation_class = evaluator.classify_violation(manifold_violation, 1)
    clearance_only_soft = bool(
        phase in ("approach", "handover", "return") and
        risk_violation <= 1e-9 and
        clearance_violation > 1e-9)
    if clearance_only_soft:
        violation_class["major_violation"] = False
        violation_class["minor_violation"] = True
        violation_class["level"] = "minor"
    if clearance_only_soft:
        manifold_status = "soft_violation"
    elif mode == "soft":
        manifold_status = (
            "infeasible" if violation_class["major_violation"] else
            "soft_violation" if manifold_violation > 1e-9 else "feasible")
    else:
        manifold_status = "infeasible" if manifold_violation > 1e-9 else "feasible"
    topology_class_status = "infeasible" if topology_violated else "feasible"
    return {
        "topology_class": context.get("topology_class", ""),
        "distance_to_corridor": float(distance_to_corridor),
        "corridor_violation": float(corridor_violation),
        "corridor_constraint_violation": float(corridor_violation),
        "corridor_constraint_status": corridor_status,
        "inside_corridor": bool(safety.get("inside_corridor", True)),
        "tube_constraint_used": True,
        "tube_constraint_mode": "hard",
        "tube_constraint_status": corridor_status,
        "tube_constraint_violation": float(corridor_violation),
        "distance_to_manifold": float(distance_to_manifold),
        "boundary_distance": (
            float(boundary_distance) if boundary_available else ""),
        "manifold_clearance": (
            float(boundary_distance) if boundary_available else
            float(max(0.0, risk_margin))),
        "minimum_clearance": float(minimum_clearance),
        "clearance_threshold": float(minimum_clearance),
        "actual_clearance": (
            float(boundary_distance) if boundary_available else
            float(max(0.0, risk_margin))),
        "progress": float(np.clip(float(progress or 0.0), 0.0, 1.0)),
        "risk_threshold": float(safe_threshold),
        "risk_constraint_violation": float(risk_violation),
        "clearance_constraint_violation": float(clearance_violation),
        "manifold_violation": float(manifold_violation),
        "manifold_constraint_violation": float(manifold_violation),
        "manifold_constraint_status": manifold_status,
        "manifold_constraint_mode": mode,
        "override_replan_limit": int(context.get("override_replan_limit", 4)),
        "soft_constraint_used": bool(mode == "soft"),
        "minor_violation": bool(violation_class["minor_violation"]),
        "major_violation": bool(violation_class["major_violation"]),
        "safety_violation_level": violation_class["level"],
        "inside_manifold": bool(
            clearance_violation <= 1e-9 and risk_violation <= 1e-9),
        "nearest_critical_point": nearest_cp,
        "critical_point_distance": float(cp_dist),
        "critical_point_constraint_status": cp_status,
        "topology_consistency_cost": float(topology_cost),
        "topology_violation": int(topology_violated),
        "topology_class_violation": int(topology_violated),
        "topology_class_constraint_status": topology_class_status,
        "corridor_cost": float(corridor_violation ** 2),
        "manifold_cost": float(manifold_violation ** 2),
    }


def _override_thresholds(context, constraints):
    radius = max(0.0, float(context.get("radius", 0.0)))
    corridor_stop = float(
        constraints.get("corridor_stop_violation", max(0.10, 0.5 * radius)))
    manifold_stop = float(
        constraints.get("manifold_stop_violation", 0.25))
    alpha = float(constraints.get("safety_override_alpha", 0.5))
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha, corridor_stop, manifold_stop


def _wheelchair_safety_override(x, v_cmd, w_cmd, dt, field, context,
                                constraints):
    alpha, corridor_stop, manifold_stop = _override_thresholds(context, constraints)
    probe = np.array(x, float)
    probe[0] += float(v_cmd) * np.cos(probe[2]) * dt
    probe[1] += float(v_cmd) * np.sin(probe[2]) * dt
    probe[2] += float(w_cmd) * dt
    risk_info = query_social_risk([probe[0], probe[1], 0.0], "wheelchair", field)
    risk = float(risk_info.get("risk_value", 0.0))
    metrics = _constraint_step_metrics([probe[0], probe[1], 0.0], risk, context)
    corridor_override = bool(metrics["corridor_violation"] > 1e-9)
    manifold_override = bool(metrics["manifold_violation"] > 1e-9)
    severe = (
        metrics["corridor_violation"] > corridor_stop or
        metrics["manifold_violation"] > manifold_stop)
    if severe:
        return 0.0, 0.0, corridor_override, manifold_override, True
    if corridor_override or manifold_override:
        v_cmd = float(v_cmd) * alpha
        if manifold_override:
            safe_point, _safe_info = _apply_manifold_hard_constraint(
                [probe[0], probe[1], 0.0], "wheelchair", field, context)
            avoid_heading = float(np.arctan2(
                safe_point[1] - x[1], safe_point[0] - x[0]))
            heading_error = float(np.arctan2(
                np.sin(avoid_heading - x[2]), np.cos(avoid_heading - x[2])))
            w_max = float(constraints.get("omega_max", constraints.get("w_max", 1.0)))
            w_cmd = float(np.clip(w_cmd + alpha * heading_error, -w_max, w_max))
    return float(v_cmd), float(w_cmd), corridor_override, manifold_override, False


def _arm_safety_override(x, u, dt, field, context, constraints):
    alpha, corridor_stop, manifold_stop = _override_thresholds(context, constraints)
    probe = np.asarray(x, float)[:3] + np.asarray(u, float)[:3] * dt
    risk_info = query_social_risk(probe, "arm", field)
    risk = float(risk_info.get("risk_value", 0.0))
    metrics = _constraint_step_metrics(probe, risk, context)
    corridor_override = bool(metrics["corridor_violation"] > 1e-9)
    manifold_override = bool(metrics["manifold_violation"] > 1e-9)
    severe = (
        metrics["corridor_violation"] > corridor_stop or
        metrics["manifold_violation"] > manifold_stop)
    u = np.asarray(u, float)[:3]
    if severe:
        return np.zeros(3, float), corridor_override, manifold_override, True
    if corridor_override or manifold_override:
        u = u * alpha
        if manifold_override:
            safe_point, _safe_info = _apply_manifold_hard_constraint(
                probe, "arm", field, context)
            correction = (np.asarray(safe_point, float)[:3] - np.asarray(x, float)[:3])
            u = u + alpha * correction / max(dt, 1e-9)
            ee_speed_max = float(constraints.get("ee_speed_max", 0.3))
            speed = float(np.linalg.norm(u))
            if speed > ee_speed_max and speed > 1e-9:
                u = u * (ee_speed_max / speed)
    return u, corridor_override, manifold_override, False


def _constraint_summary(rows):
    rows = list(rows or [])
    corridor_dist = [
        float(row.get("distance_to_corridor", 0.0) or 0.0) for row in rows]
    corridor_viol = [
        float(row.get("corridor_violation", 0.0) or 0.0) for row in rows]
    manifold_viol = [
        float(row.get("manifold_violation", 0.0) or 0.0) for row in rows]
    manifold_clearance = [
        float(row.get("manifold_clearance", 0.0) or 0.0)
        for row in rows
        if str(row.get("manifold_clearance", "")).strip() not in ("", "inf")]
    risk_viol = [
        float(row.get("risk_constraint_violation", 0.0) or 0.0)
        for row in rows]
    clearance_viol = [
        float(row.get("clearance_constraint_violation", 0.0) or 0.0)
        for row in rows]
    risk_values = [
        float(row.get("risk_value", row.get("risk", 0.0)) or 0.0)
        for row in rows]
    topology_costs = [
        float(row.get("topology_consistency_cost", 0.0) or 0.0)
        for row in rows]
    corridor_costs = [
        float(row.get("corridor_cost", 0.0) or 0.0) for row in rows]
    manifold_costs = [
        float(row.get("manifold_cost", 0.0) or 0.0) for row in rows]
    corridor_hard = int(sum(
        1 for row in rows
        if str(row.get("corridor_constraint_status", "")) == "infeasible"))
    corridor_soft = int(sum(
        1 for row in rows
        if str(row.get("corridor_constraint_status", "")) == "soft_violation"))
    manifold_hard = int(sum(
        1 for row in rows
        if str(row.get("manifold_constraint_status", "")) == "infeasible"))
    manifold_soft = int(sum(
        1 for row in rows
        if str(row.get("manifold_constraint_status", "")) == "soft_violation"))
    minor_violation_count = int(sum(
        1 for row in rows if bool(row.get("minor_violation", False))))
    major_violation_count = int(sum(
        1 for row in rows if bool(row.get("major_violation", False))))
    topology_class_viol = int(sum(
        1 for row in rows
        if int(row.get("topology_class_violation", 0) or 0)))
    corridor_override_count = int(sum(
        1 for row in rows if bool(row.get("corridor_safety_override", False)) or
        bool(row.get("corridor_override", False))))
    manifold_override_count = int(sum(
        1 for row in rows if bool(row.get("manifold_safety_override", False)) or
        bool(row.get("manifold_override", False))))
    consecutive = 0
    consecutive_manifold_override_max = 0
    for row in rows:
        if bool(row.get("manifold_safety_override", False)) or bool(
                row.get("manifold_override", False)):
            consecutive += 1
            consecutive_manifold_override_max = max(
                consecutive_manifold_override_max, consecutive)
        else:
            consecutive = 0
    override_ratio = float(manifold_override_count / float(len(rows))) if rows else 0.0
    max_corridor_violation = float(max(corridor_viol) if corridor_viol else 0.0)
    max_manifold_violation = float(max(manifold_viol) if manifold_viol else 0.0)
    min_manifold_clearance = float(
        min(manifold_clearance) if manifold_clearance else 0.0)
    average_manifold_clearance = float(
        np.mean(manifold_clearance) if manifold_clearance else 0.0)
    max_risk_value = float(max(risk_values) if risk_values else 0.0)
    manifold_status = (
        "infeasible" if manifold_hard else
        "soft_violation" if manifold_soft else "feasible")
    return {
        "topology_consistency_cost": float(sum(topology_costs)),
        "topology_violation_count": int(sum(
            1 for row in rows if int(row.get("topology_violation", 0) or 0))),
        "topology_class_constraint_used": True,
        "selected_topology_class": str(
            rows[0].get("topology_class", "")) if rows else "",
        "topology_class_violation_count": topology_class_viol,
        "corridor_cost": float(sum(corridor_costs)),
        "manifold_cost": float(sum(manifold_costs)),
        "max_corridor_distance": float(max(corridor_dist) if corridor_dist else 0.0),
        "mean_corridor_distance": float(
            np.mean(corridor_dist) if corridor_dist else 0.0),
        "max_corridor_deviation": float(max(corridor_dist) if corridor_dist else 0.0),
        "mean_corridor_deviation": float(
            np.mean(corridor_dist) if corridor_dist else 0.0),
        "corridor_violation_count": int(sum(1 for v in corridor_viol if v > 1e-9)),
        "predicted_corridor_violation_count": int(
            sum(1 for v in corridor_viol if v > 1e-9)),
        "corridor_constraint_violation_count": int(
            sum(1 for v in corridor_viol if v > 1e-9)),
        "max_corridor_violation": max_corridor_violation,
        "mean_corridor_violation": float(
            np.mean(corridor_viol) if corridor_viol else 0.0),
        "corridor_constraint_status": (
            "infeasible" if corridor_hard else
            "soft_violation" if corridor_soft else "feasible"),
        "tube_constraint_used": True,
        "tube_constraint_mode": "hard",
        "tube_constraint_status": (
            "infeasible" if corridor_hard else
            "soft_violation" if corridor_soft else "feasible"),
        "corridor_override_count": corridor_override_count,
        "corridor_safety_override": bool(corridor_override_count > 0),
        "max_manifold_violation": max_manifold_violation,
        "mean_manifold_violation": float(
            np.mean(manifold_viol) if manifold_viol else 0.0),
        "manifold_violation_count": int(sum(1 for v in manifold_viol if v > 1e-9)),
        "predicted_manifold_violation_count": int(
            sum(1 for v in manifold_viol if v > 1e-9)),
        "manifold_constraint_status": manifold_status,
        "manifold_feasibility_status": (
            "manifold_infeasible" if manifold_status == "infeasible"
            else "feasible_with_soft_violation"
            if manifold_status == "soft_violation" else "feasible"),
        "soft_constraint_used": bool(any(
            str(row.get("manifold_constraint_mode", "hard")) == "soft"
            for row in rows)),
        "minor_violation": bool(minor_violation_count > 0),
        "major_violation": bool(major_violation_count > 0),
        "minor_violation_count": int(minor_violation_count),
        "major_violation_count": int(major_violation_count),
        "mpc_warning": (
            "minor_manifold_violation"
            if minor_violation_count > 0 and major_violation_count == 0 else ""),
        "min_manifold_clearance": min_manifold_clearance,
        "average_manifold_clearance": average_manifold_clearance,
        "planning_clearance": min_manifold_clearance,
        "predicted_clearance": min_manifold_clearance,
        "execution_clearance": min_manifold_clearance,
        "predicted_min_clearance": min_manifold_clearance,
        "predicted_max_risk": max_risk_value,
        "max_risk_value": max_risk_value,
        "risk_violation_count": int(sum(1 for v in risk_viol if v > 1e-9)),
        "clearance_violation_count": int(
            sum(1 for v in clearance_viol if v > 1e-9)),
        "manifold_constraint_mode": str(
            rows[0].get("manifold_constraint_mode", "hard")) if rows else "hard",
        "override_replan_limit": int(
            rows[0].get("override_replan_limit", 4) if rows else 4),
        "manifold_override_count": manifold_override_count,
        "override_ratio": override_ratio,
        "manifold_override_ratio": override_ratio,
        "consecutive_manifold_override_max": int(
            consecutive_manifold_override_max),
        "manifold_safety_override": bool(manifold_override_count > 0),
    }


def _critical_sequence_summary(rows, context):
    rows = list(rows or [])
    critical = _critical_points_from_context(context)
    soft_radius = float(context.get(
        "critical_point_soft_radius",
        context.get("critical_point_radius", 0.20)))
    hard_radius = max(
        soft_radius,
        float(context.get("critical_point_hard_radius", soft_radius)))
    mode = str(context.get("critical_point_constraint_mode", "soft_hard"))
    association = dict(context.get("critical_point_association", {}) or {})
    if association.get("critical_point_association_used"):
        cps = list(association.get("critical_points", []) or [])
        hard_violations = int(association.get("hard_violation_count", 0) or 0)
        soft_violations = int(association.get("soft_violation_count", 0) or 0)
        order_valid = bool(association.get("topology_sequence_valid", False))
        status = str(association.get("critical_point_status", ""))
        if not status:
            status = (
                "hard_violation" if hard_violations or not order_valid else
                "soft_violation" if soft_violations else "passed")
        tracker = TopologyStageTracker(association)
        if cps:
            final_idx = max(
                int(item.get("trajectory_index", -1)) for item in cps)
            tracker.update_index(final_idx)
        stage_status = tracker.finish()
        return {
            "critical_point_constraint_used": True,
            "critical_point_association_used": True,
            "critical_point_sequence_valid": bool(
                order_valid and hard_violations == 0),
            "topology_sequence_valid": bool(order_valid and hard_violations == 0),
            "critical_point_status": status,
            "critical_point_count": int(len(cps)),
            "critical_point_violation_count": int(hard_violations),
            "critical_point_soft_violation_count": int(soft_violations),
            "critical_point_hard_violation_count": int(hard_violations),
            "max_critical_point_distance": float(max(
                [float(item.get("distance_to_trajectory", 0.0) or 0.0)
                 for item in cps] or [0.0])),
            "topology_sequence_constraint_status": (
                "infeasible" if hard_violations or not order_valid else
                "soft_violation" if soft_violations else "feasible"),
            "topology_constraint_status": (
                "infeasible" if hard_violations or not order_valid else
                "soft_violation" if soft_violations else "feasible"),
            "critical_point_constraint_mode": mode,
            "critical_point_soft_radius": float(
                association.get("soft_radius", soft_radius)),
            "critical_point_hard_radius": float(
                association.get("hard_radius", hard_radius)),
            "topology_infeasible_count": int(
                hard_violations + (0 if order_valid else 1)),
            "current_topology_stage": int(stage_status.get("current_stage", 0)),
            "passed_critical_points": list(
                stage_status.get("passed_critical_points", [])),
            "passed_topology_stages": list(stage_status.get("passed_stages", [])),
            "critical_point_association": association,
        }
    if not critical:
        return {
            "critical_point_constraint_used": True,
            "critical_point_association_used": False,
            "critical_point_count": 0,
            "critical_point_violation_count": 0,
            "critical_point_soft_violation_count": 0,
            "critical_point_hard_violation_count": 0,
            "max_critical_point_distance": 0.0,
            "topology_sequence_constraint_status": "feasible",
            "topology_constraint_status": "feasible",
            "critical_point_constraint_mode": mode,
            "critical_point_soft_radius": float(soft_radius),
            "critical_point_hard_radius": float(hard_radius),
            "critical_point_status": "passed",
            "topology_sequence_valid": True,
            "current_topology_stage": 0,
            "passed_critical_points": [],
        }
    reference_path = _as_optional_points(context.get("reference_path", []))
    if len(reference_path) > 0:
        pred = np.asarray(reference_path, float)
    else:
        pred = []
        for row in rows:
            try:
                pred.append([
                    float(row.get("pred_x", 0.0)),
                    float(row.get("pred_y", 0.0)),
                    float(row.get("pred_z", 0.0) or 0.0),
                ])
            except Exception:
                pass
        pred = np.asarray(pred, float)
    next_start = 0
    distances = []
    soft_violations = 0
    hard_violations = 0
    for item in critical:
        if len(pred) == 0:
            d = float("inf")
            idx = -1
        else:
            q = np.asarray(item["point"], float)[:3]
            tail = pred[next_start:]
            ds = np.linalg.norm(tail[:, :3] - q[None, :3], axis=1)
            rel_idx = int(np.argmin(ds))
            d = float(ds[rel_idx])
            idx = next_start + rel_idx
        distances.append(d)
        if d <= soft_radius and idx >= next_start:
            next_start = idx + 1
        elif d <= hard_radius and idx >= next_start:
            soft_violations += 1
            next_start = idx + 1
        else:
            hard_violations += 1
    if str(mode) == "ignore":
        status = "feasible"
        hard_violations = 0
        soft_violations = 0
    elif hard_violations > 0:
        status = "infeasible"
    elif soft_violations > 0:
        status = "soft_violation"
    else:
        status = "feasible"
    return {
        "critical_point_constraint_used": True,
        "critical_point_association_used": False,
        "critical_point_sequence_valid": bool(hard_violations == 0),
        "topology_sequence_valid": bool(hard_violations == 0),
        "critical_point_status": (
            "hard_violation" if hard_violations else
            "soft_violation" if soft_violations else "passed"),
        "critical_point_count": int(len(critical)),
        "critical_point_violation_count": int(hard_violations),
        "critical_point_soft_violation_count": int(soft_violations),
        "critical_point_hard_violation_count": int(hard_violations),
        "max_critical_point_distance": float(
            max(distances) if distances and all(np.isfinite(distances)) else
            (float("inf") if distances else 0.0)),
        "topology_sequence_constraint_status": status,
        "topology_constraint_status": status,
        "critical_point_constraint_mode": mode,
        "critical_point_soft_radius": float(soft_radius),
        "critical_point_hard_radius": float(hard_radius),
        "topology_infeasible_count": int(hard_violations),
        "current_topology_stage": int(len(critical) + 1 if hard_violations == 0 else next_start),
        "passed_critical_points": [],
    }


def _feasibility_from_constraints(summary, risk_exceed_count=0):
    if int(summary.get("consecutive_manifold_override_max", 0) or 0) >= int(
            summary.get("override_replan_limit", 4) or 4):
        return (
            "manifold_infeasible",
            "persistent_manifold_violation",
            "manifold")
    if int(summary.get("major_violation_count", 0) or 0) > 0:
        reason = (
            "major_risk_violation"
            if int(summary.get("risk_violation_count", 0) or 0) > 0
            else "major_clearance_violation")
        return "manifold_infeasible", reason, "manifold"
    if str(summary.get("manifold_constraint_status", "")) == "infeasible":
        reason = (
            "risk_violation"
            if int(summary.get("risk_violation_count", 0) or 0) > 0
            else "clearance_violation")
        return "manifold_infeasible", reason, "manifold"
    if str(summary.get("corridor_constraint_status", "")) == "infeasible":
        return "corridor_infeasible", "corridor_hard_constraint_violation", "corridor"
    if str(summary.get("critical_point_status", "")) == "hard_violation":
        return "topology_infeasible", "critical_point_sequence_violation", "topology"
    if not bool(summary.get("critical_point_sequence_valid", True)):
        return "topology_infeasible", "critical_point_sequence_violation", "topology"
    if not bool(summary.get("topology_sequence_valid", True)):
        return "topology_infeasible", "critical_point_sequence_violation", "topology"
    if int(summary.get("topology_infeasible_count", 0) or 0) > 0:
        return "topology_infeasible", "critical_point_sequence_violation", "topology"
    if int(summary.get("critical_point_violation_count", 0) or 0) > 0:
        return "topology_infeasible", "critical_point_sequence_violation", "topology"
    if int(summary.get("topology_class_violation_count", 0) or 0) > 0:
        return "topology_infeasible", "topology_class_violation", "topology"
    if str(summary.get("manifold_constraint_status", "")) == "soft_violation":
        return "feasible_with_soft_violation", "", ""
    if str(summary.get("corridor_constraint_status", "")) == "soft_violation":
        return "feasible_with_soft_violation", "", ""
    if (str(summary.get("topology_sequence_constraint_status", "")) == "soft_violation" or
            str(summary.get("critical_point_status", "")) == "soft_violation" or
            int(summary.get("critical_point_soft_violation_count", 0) or 0) > 0):
        return "feasible_with_soft_violation", "", ""
    return "feasible", "none", ""


def _apply_acceptance_diagnostics(result, context):
    result = dict(result or {})
    context = dict(context or {})
    constraint = dict(context.get("manifold_constraint", {}) or {})
    mode = str(context.get(
        "manifold_constraint_mode",
        constraint.get("manifold_constraint_mode", "soft"))).strip().lower()
    nominal_clearance = float(context.get(
        "nominal_minimum_clearance",
        constraint.get(
            "minimum_clearance",
            constraint.get("min_clearance", context.get("minimum_clearance", 0.0)))) or 0.0)
    effective_clearance = float(context.get(
        "minimum_clearance",
        constraint.get(
            "effective_minimum_clearance",
            constraint.get("effective_min_clearance", nominal_clearance))) or 0.0)
    planning_margin = float(
        constraint.get("planning_clearance_margin",
                       context.get("planning_clearance_margin", 0.0)) or 0.0)
    boundary = constraint.get("boundary", context.get("manifold_boundary", []))
    clearance_source = (
        "risk_manifold_boundary" if bool(boundary) else
        "risk_threshold_only")
    reference_path = context.get("reference_path", [])
    if _as_optional_points(reference_path).size:
        evaluator = ManifoldConstraintEvaluator(
            manifold_constraint={
                "boundary": boundary,
                "minimum_clearance": nominal_clearance,
                "effective_minimum_clearance": effective_clearance,
                "risk_threshold": context.get("nominal_safe_threshold",
                                              context.get("safe_threshold", 1.0)),
                "effective_risk_threshold": context.get("safe_threshold", 1.0),
            },
            corridor_constraint={
                "centerline": context.get("centerline", []),
                "radius": context.get("radius", 0.0),
            },
            risk_field=context.get("social_field"),
            planning_clearance_margin=0.0)
        reference_status = evaluator.evaluate_trajectory(reference_path)
        reference_clearance = float(reference_status.get("min_clearance", 0.0))
    else:
        reference_clearance = float(result.get("min_manifold_clearance", 0.0) or 0.0)
    execution_clearance = float(result.get(
        "min_manifold_clearance",
        result.get("predicted_min_clearance", 0.0)) or 0.0)
    result.update({
        "mpc_manifold_constraint_mode": mode,
        "manifold_constraint_mode": mode,
        "minimum_clearance": nominal_clearance,
        "effective_minimum_clearance": effective_clearance,
        "required_candidate_clearance": float(effective_clearance + max(0.0, planning_margin)),
        "required_execution_clearance": effective_clearance,
        "candidate_reference_min_clearance": reference_clearance,
        "refined_reference_min_clearance": reference_clearance,
        "predicted_execution_min_clearance": execution_clearance,
        "actual_execution_min_clearance": execution_clearance,
        "clearance_source": clearance_source,
        "distance_rule": "distance_to_boundary >= effective_minimum_clearance",
        "risk_rule": "risk_value <= effective_risk_threshold",
    })
    if (mode == "soft" and
            int(result.get("manifold_violation_count", 0) or 0) > 0 and
            not str(result.get("failure_reason", "") or "").strip()):
        result["warning_reason"] = "soft_manifold_violation"
    elif mode == "hard" and int(result.get("manifold_violation_count", 0) or 0) > 0:
        result["failure_reason"] = result.get("failure_reason") or "hard_manifold_violation"
    else:
        result.setdefault("warning_reason", "")
    return result


def _apply_success_contract(result, reference_audit=None):
    result = dict(result or {})
    mode = str(result.get(
        "manifold_constraint_mode",
        result.get("mpc_manifold_constraint_mode", "soft"))).strip().lower()
    if mode not in ("soft", "hard"):
        mode = "soft"
    mpc_status = str(result.get("mpc_feasibility_status", ""))
    planner_success = bool(
        (reference_audit or result.get("reference_safety_audit", {}) or {}).get(
            "feasible", True))
    override_limit = int(result.get("override_replan_limit", 4) or 4)
    consecutive_override = int(
        result.get("consecutive_manifold_override_max", 0) or 0)
    controller_success = bool(
        mpc_status in (
            "feasible", "feasible_with_soft_violation",
            "feasible_with_soft_violations") and
        consecutive_override < override_limit)
    executed_required = bool(result.get("executed_evidence_required", False))
    executed_count = int(result.get("actual_executed_trajectory_count", 0) or 0)
    use_executed = bool(executed_count > 0)
    prefix = "executed_" if use_executed else ""
    manifold_v = int(result.get(
        prefix + "manifold_violation_count",
        result.get("manifold_violation_count", 0)) or 0)
    corridor_v = int(result.get(
        prefix + "corridor_violation_count",
        result.get("corridor_violation_count", 0)) or 0)
    major_v = int(result.get(
        prefix + "major_violation_count",
        result.get("major_violation_count", 0)) or 0)
    max_soft = float(result.get(
        prefix + "max_manifold_violation",
        result.get("max_manifold_violation", 0.0)) or 0.0)
    soft_tol = float(result.get("manifold_soft_tolerance", 0.005) or 0.005)
    step_count = int(
        executed_count if use_executed else result.get(
            "executed_trajectory_count",
            result.get("rollout_solve_count", 0)) or 0)
    soft_ratio = (
        float(manifold_v) / float(step_count)
        if step_count > 0 else (1.0 if manifold_v > 0 else 0.0))
    soft_ratio_limit = float(result.get("soft_violation_ratio_limit", 0.0) or 0.0)
    if executed_required and not use_executed:
        safety_success = False
    elif mode == "hard":
        safety_success = bool(manifold_v == 0 and corridor_v == 0)
    else:
        safety_success = bool(
            corridor_v == 0 and major_v == 0 and
            max_soft <= soft_tol + 1e-9 and
            soft_ratio <= soft_ratio_limit + 1e-9 and
            consecutive_override < override_limit)
    task_success = bool(result.get("rolling_goal_reached", True))
    overall_success = bool(
        task_success and planner_success and controller_success and
        safety_success)
    result.update({
        "task_success": bool(task_success),
        "planner_success": bool(planner_success),
        "controller_success": bool(controller_success),
        "safety_success": bool(safety_success),
        "overall_success": bool(overall_success),
        "success": bool(overall_success),
        "soft_violation_ratio": float(soft_ratio),
        "soft_violation_ratio_limit": float(soft_ratio_limit),
        "safety_truth_source": "executed" if use_executed else "predicted",
        "executed_evidence_complete": bool(
            use_executed or not executed_required),
    })
    if not overall_success and not str(result.get("failure_reason", "")).strip():
        result["failure_reason"] = (
            "safety_contract_failed" if not safety_success else
            "controller_failed" if not controller_success else
            "planner_failed" if not planner_success else
            "task_failed")
    if overall_success:
        result["failure_reason"] = ""
    return result


def evaluate_executed_trajectory(trajectory, context, social_field=None,
                                 robot_type="", phase_sequence=None,
                                 corridor_active_sequence=None):
    """Evaluate measured states with the same constraint truth as MPC rollout."""
    points = _as_points(trajectory)
    phases = list(phase_sequence or _reference_phases(
        trajectory, len(points), robot_type))
    corridor_active = list(corridor_active_sequence or [])
    rows = []
    for index, point in enumerate(points):
        phase = phases[min(index, len(phases) - 1)] if phases else (
            "navigation" if str(robot_type).lower() == "wheelchair" else "approach")
        progress = _phase_local_progress(phases, index) if phases else 0.0
        risk_info = query_social_risk(point, robot_type, social_field)
        metrics = _constraint_step_metrics(
            point, float(risk_info.get("risk_value", 0.0)), context,
            phase=phase, progress=progress)
        active = bool(
            corridor_active[min(index, len(corridor_active) - 1)]) \
            if corridor_active else True
        raw_corridor_violation = float(metrics.get("corridor_violation", 0.0))
        if not active:
            metrics["corridor_out_of_scope_violation"] = raw_corridor_violation
            metrics["corridor_violation"] = 0.0
            metrics["corridor_constraint_violation"] = 0.0
            metrics["corridor_constraint_status"] = "not_applicable"
        else:
            metrics["corridor_out_of_scope_violation"] = 0.0
        metrics["corridor_active"] = bool(active)
        row = {
            "step": int(index),
            "global_step": int(index),
            "x": float(point[0]),
            "y": float(point[1]),
            "z": float(point[2]),
            "pred_x": float(point[0]),
            "pred_y": float(point[1]),
            "pred_z": float(point[2]),
            "phase": str(phase),
            "progress": float(progress),
            "trajectory_source": "executed",
            "risk": float(risk_info.get("risk_value", 0.0)),
            "risk_value": float(risk_info.get("risk_value", 0.0)),
            "risk_query_valid": int(bool(risk_info.get("risk_query_valid", False))),
            "risk_query_source": str(risk_info.get("risk_query_source", "")),
            "feasibility_status": "measured",
            "manifold_override": False,
            "corridor_override": False,
        }
        row.update(metrics)
        rows.append(row)
    return rows, _constraint_summary(rows)


def _weighted_total(costs, weights):
    return (
        float(weights.get("track", 1.0)) * float(costs.get("tracking_cost", 0.0)) +
        float(weights.get("control", 0.1)) * float(costs.get("control_cost", 0.0)) +
        float(weights.get("smooth", 0.2)) * float(costs.get("smoothness_cost", 0.0)) +
        float(weights.get("risk", 1.0)) * float(costs.get("risk_cost", 0.0)) +
        float(weights.get("topology", 5.0)) * float(costs.get("topology_cost", 0.0)) +
        float(weights.get("corridor", 10.0)) * float(costs.get("corridor_cost", 0.0)) +
        float(weights.get("manifold", 10.0)) * float(costs.get("manifold_cost", 0.0)) +
        float(weights.get("safety_violation", 5.0)) *
        float(costs.get("safety_violation_cost", 0.0)) +
        float(weights.get("constraint_violation", 10.0)) *
        float(costs.get("constraint_violation_cost", 0.0)) +
        float(weights.get("handover", 1.0)) * float(costs.get("handover_cost", 0.0)) +
        float(weights.get(
            "task_weight_effective", weights.get("task", 0.0))) *
        float(costs.get("task_cost", 0.0)))


def _jsonable_points(points):
    pts = _as_optional_points(points)
    return pts.tolist()


def build_mpc_constraint_inputs(corridor=None, manifold=None, reference_path=None,
                                safe_threshold=None, phase=None,
                                robot_type="generic", phase_params=None,
                                manifold_constraint_mode=None,
                                minimum_clearance=None, strict_stsm=False,
                                expected_corridor_id=None):
    if strict_stsm:
        from stsm_madp.corridor import require_corridor_contract
        require_corridor_contract(
            corridor, reference_path=reference_path,
            expected_corridor_id=expected_corridor_id,
            require_morse=True, require_tube=True)
    constraint = build_topology_constraint(
        selected_corridor=corridor,
        safe_manifold=manifold,
        refined_reference=reference_path,
        safe_threshold=safe_threshold,
        minimum_clearance=minimum_clearance,
        phase=phase,
        robot_type=robot_type,
        manifold_constraint_mode=manifold_constraint_mode,
        phase_params=phase_params)
    corridor_id = str(
        getattr(corridor, "corridor_id",
                getattr(corridor, "label", "")) if corridor is not None else "")
    waypoints = (
        getattr(corridor, "waypoints", None) if corridor is not None
        else reference_path)
    inputs = mpc_inputs_from_constraint(
        constraint, corridor_id=corridor_id, waypoints=waypoints)
    if strict_stsm:
        tube_id = str(inputs[3].get(
            "topology_tube_constraint", {}).get("corridor_id", ""))
        mpc_id = str(inputs[1].get("corridor_id", ""))
        if not corridor_id or mpc_id != corridor_id or tube_id != corridor_id:
            from stsm_madp.corridor import CorridorContractError
            raise CorridorContractError("mpc_corridor_id_mismatch")
    return inputs


def _arm_mpc_input_validation(ref, topology_info, corridor_info, manifold_info,
                              social_field, constraints, context):
    def _count(value):
        if isinstance(value, (list, tuple, dict)):
            return len(value)
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    for source in (corridor_info, topology_info, manifold_info):
        if not isinstance(source, dict):
            continue
        validation = source.get("ik_validation")
        if isinstance(validation, dict) and validation:
            return dict(validation)
        ik_valid = source.get("ik_valid", None)
        link_valid = source.get("link_collision_valid", None)
        if ik_valid is not None or link_valid is not None:
            ik_ok = bool(ik_valid)
            link_ok = bool(link_valid) if link_valid is not None else True
            link_collision = bool(source.get("link_collision", False))
            return {
                "valid": bool(ik_ok and link_ok and not link_collision),
                "ik_valid": bool(ik_ok),
                "link_collision_valid": bool(link_ok),
                "link_collision": bool(link_collision),
                "failure_reason": (
                    "" if ik_ok and link_ok and not link_collision else
                    "link_collision" if link_collision or not link_ok else
                    "ik_invalid"),
                "source": "candidate_execution_validation",
                "arm_pose_optimization_used": bool(source.get(
                    "arm_pose_optimization_used", False)),
                "arm_ik_candidate_count": int(
                    _count(source.get("arm_ik_candidate_count", 0))),
                "arm_ik_candidate_attempts": int(
                    _count(source.get("arm_ik_candidate_attempts", 0))),
            }
    solver = TopologyIKSolver(
        risk_field=social_field,
        boundary=context.get("manifold_boundary", None),
        risk_threshold=constraints.get("risk_threshold", 1.0),
        minimum_clearance=constraints.get("minimum_clearance", 0.0))
    candidate = {
        "candidate_id": str((corridor_info or {}).get(
            "candidate_id", (corridor_info or {}).get("corridor_id", ""))),
        "centerline": np.asarray(ref, float).tolist(),
        "boundary": (corridor_info or {}).get(
            "boundary", context.get("manifold_boundary", None)),
    }
    _, validation = solver.validate_candidate(
        candidate, boundary=candidate.get("boundary"), risk_field=social_field)
    return dict(validation)


def _empty_mpc_result(robot_type, corridor_id, source, horizon, dt,
                      constraints, weights, status, reason):
    return {
        "success": False,
        "robot_type": str(robot_type),
        "selected_corridor_id": str(corridor_id or ""),
        "reference_source": str(source or ""),
        "reference_path_file": "mpc_reference_path.csv",
        "reference_path_count": 0,
        "horizon": int(horizon),
        "dt": float(dt),
        "constraints": constraints or {},
        "weights": weights or {},
        "task_mode": str((weights or {}).get("task_mode", "")),
        "task_weight": dict((weights or {}).get("task_weight", {}) or {}),
        "task_weight_used": bool((weights or {}).get("task_weight_used", False)),
        "constraint_version": "topology_hard_v2",
        "module_chain_valid": False,
        "mpc_feasibility_status": status,
        "failure_reason": reason,
        "executable_trajectory": [],
        "control_sequence": [],
        "predicted_states": [],
        "cost_breakdown_rows": [],
        "rollout_rows": [],
        "executed_trajectory_rows": [],
        "tracking_cost": 0.0,
        "control_cost": 0.0,
        "smoothness_cost": 0.0,
        "risk_cost": 0.0,
        "topology_cost": 0.0,
        "corridor_cost": 0.0,
        "manifold_cost": 0.0,
        "safety_violation_cost": 0.0,
        "constraint_violation_cost": 0.0,
        "handover_cost": 0.0,
        "task_cost": 0.0,
        "total_cost": 0.0,
        "max_tracking_error": 0.0,
        "mean_tracking_error": 0.0,
        "max_risk": 0.0,
        "mean_risk": 0.0,
        "risk_exceed_count": 0,
        "risk_exceed_ratio": 0.0,
        "risk_query_called": False,
        "risk_query_valid_count": 0,
        "risk_query_invalid_count": 0,
        "risk_query_source": "unavailable",
        "risk_sanity_status": "failed_no_risk_query",
        "risk_sanity_warning": "reference_path_empty",
        "rollout_mode": "none",
        "rollout_solve_count": 0,
        "rollout_horizon_rows": 0,
        "executed_trajectory_count": 0,
        "rollout_log_file": "mpc_rollout_log.csv",
        "executed_trajectory_file": "mpc_executed_trajectory.csv",
        "cost_breakdown_file": "mpc_cost_breakdown.csv",
        "rolling_goal_reached": False,
        "rolling_stop_reason": reason,
        "mpc_risk_cost": 0.0,
        "mpc_risk_exceed_count": 0,
        "mpc_mean_risk": 0.0,
        "mpc_max_risk": 0.0,
        "max_control": 0.0,
        "max_control_delta": 0.0,
        "control_sequence_count": 0,
        "predicted_state_count": 0,
        "mpc_affects_candidate_ranking": 0,
        "mpc_candidate_feasibility_used": 0,
        "mpc_candidate_selection": 0,
        "mpc_execution_cost_in_score": 0,
        "topology_constraint_used": False,
        "topology_sequence_constraint_used": False,
        "corridor_constraint_used": False,
        "tube_constraint_used": False,
        "tube_constraint_mode": "hard",
        "predicted_corridor_violation_count": 0,
        "predicted_manifold_violation_count": 0,
        "manifold_constraint_used": False,
        "corridor_hard_constraint_used": False,
        "manifold_hard_constraint_used": False,
        "manifold_constraint_mode": "soft",
        "soft_constraint_used": True,
        "minor_violation": False,
        "major_violation": False,
        "minor_violation_count": 0,
        "major_violation_count": 0,
        "mpc_warning": "",
        "planning_clearance": 0.0,
        "predicted_clearance": 0.0,
        "execution_clearance": 0.0,
        "predicted_min_clearance": 0.0,
        "predicted_max_risk": 0.0,
        "min_manifold_clearance": 0.0,
        "average_manifold_clearance": 0.0,
        "max_risk_value": 0.0,
        "manifold_feasibility_status": "feasible",
        "critical_point_constraint_used": False,
        "critical_point_sequence_constraint_used": False,
        "critical_point_association_used": False,
        "critical_point_sequence_valid": False,
        "topology_sequence_valid": False,
        "critical_point_status": "hard_violation",
        "current_topology_stage": 0,
        "passed_critical_points": [],
        "critical_points_passed": [],
        "current_target_critical_point": "",
        "topology_infeasible_count": 0,
        "corridor_override_count": 0,
        "manifold_override_count": 0,
        "topology_class_constraint_used": False,
        "replan_required": False,
        "mpc_feedback": {
            "replan_required": False,
            "failure_type": "",
            "failed_constraint": "",
            "selected_corridor_id": str(corridor_id or ""),
            "failure_reason": reason,
            "failed_constraint_type": "",
        },
        "topology_class": "",
        "topology_consistency_cost": 0.0,
        "topology_violation_count": 0,
        "max_corridor_deviation": 0.0,
        "mean_corridor_deviation": 0.0,
        "corridor_violation_count": 0,
        "max_manifold_violation": 0.0,
        "mean_manifold_violation": 0.0,
        "manifold_violation_count": 0,
    }


def _mpc_candidate_decision_flags(context):
    corridor_info = dict((context or {}).get("corridor_info", {}) or {})
    topology_info = dict((context or {}).get("topology_info", {}) or {})
    breakdown = dict(corridor_info.get(
        "candidate_cost_breakdown",
        topology_info.get("candidate_cost_breakdown", {})) or {})

    execution_term = float(breakdown.get(
        "execution_cost_term",
        corridor_info.get("execution_cost", 0.0)) or 0.0)
    execution_cost_in_score = bool(
        breakdown.get("mpc_execution_cost_in_score", False) or
        abs(execution_term) > 1e-12)
    feasibility_used = bool(
        corridor_info.get("hard_feasible", False) or
        corridor_info.get("execution_feasible", False) or
        topology_info.get("hard_feasible", False) or
        topology_info.get("execution_feasible", False))
    affects_ranking = bool(
        corridor_info.get("mpc_affects_candidate_ranking", False) or
        topology_info.get("mpc_affects_candidate_ranking", False) or
        execution_cost_in_score or feasibility_used)

    return {
        "mpc_affects_candidate_ranking": int(affects_ranking),
        "mpc_candidate_feasibility_used": int(feasibility_used),
        "mpc_candidate_selection": int(affects_ranking or feasibility_used),
        "mpc_execution_cost_in_score": int(execution_cost_in_score),
        "mpc_candidate_cost_breakdown": breakdown,
    }


def run_mpc_tracking(robot_type, current_state, reference_path,
                     topology_info=None, corridor_info=None,
                     manifold_info=None, social_field=None, constraints=None,
                     horizon=None, dt=None,
                     selected_corridor_id="", risk_threshold=None,
                     config=None, rollout_mode="rolling_window"):
    if social_field is None and manifold_info is None and (
            hasattr(topology_info, "phi_s") or callable(topology_info) or
            (topology_info is not None and not isinstance(topology_info, dict))):
        social_field = topology_info
        constraints = corridor_info if constraints is None else constraints
        topology_info = {}
        corridor_info = {}
        manifold_info = {}
    cfg = _merge_dict(DEFAULT_MPC_CONFIG, config or {})
    robot = str(robot_type or "").lower()
    h = int(horizon if horizon is not None else cfg.get("horizon", 10))
    step_dt = float(dt if dt is not None else cfg.get("dt", 0.1))
    weights = dict(cfg.get("weights", {}))
    task_mode = resolve_task_mode(
        cfg.get("task_mode", ""), robot_type=robot)
    task_weight = resolve_task_weight(
        task_mode,
        task_config=cfg.get("task_config", {}),
        task_weight=cfg.get("task_weight", {}),
        robot_type=robot)
    weights["task_mode"] = str(task_mode)
    weights["task_weight"] = dict(task_weight)
    weights["task_weight_used"] = True
    weights["phase_cost_weights"] = dict(cfg.get("phase_cost_weights", {}) or {})
    weights["phase_weight_used"] = True
    robot_constraints = _merge_dict(
        cfg.get(robot, {}), constraints or {})
    threshold = float(
        risk_threshold if risk_threshold is not None else
        robot_constraints.get("risk_threshold", 1.0))
    ref = _as_points(reference_path)
    phase_sequence = _reference_phases(reference_path, len(ref), robot)
    if ref.size == 0:
        result = _empty_mpc_result(
            robot, selected_corridor_id, "refined_waypoints", h, step_dt,
            robot_constraints, weights, "infeasible_reference_empty",
            "reference_path_empty")
        result.update({
            "task_mode": str(task_mode),
            "task_weight": dict(task_weight),
            "task_weight_used": True,
        })
        return result
    context = _extract_constraint_context(
        topology_info, corridor_info, manifold_info, ref, social_field,
        robot_constraints)
    planning_margin = float(
        context.get("manifold_constraint", {}).get(
            "planning_clearance_margin",
            robot_constraints.get("planning_clearance_margin", 0.0)) or 0.0)
    reference_audit = audit_reference_safety(
        ref, context, robot_type=robot, phase_sequence=phase_sequence,
        planning_margin=planning_margin)
    if not bool(reference_audit.get("feasible", False)):
        candidate_decision_flags = _mpc_candidate_decision_flags(context)
        result = _empty_mpc_result(
            robot, selected_corridor_id, "selected_candidate_waypoints",
            h, step_dt, robot_constraints, weights,
            "reference_manifold_infeasible",
            "reference_manifold_infeasible")
        result.update({
            "reference_path_count": int(len(ref)),
            "reference_safety_audit": reference_audit,
            "planner_success": False,
            "controller_success": False,
            "safety_success": False,
            "overall_success": False,
            "module_chain_valid": False,
            "mpc_affects_candidate_ranking": int(
                candidate_decision_flags["mpc_affects_candidate_ranking"]),
            "mpc_candidate_feasibility_used": int(
                candidate_decision_flags["mpc_candidate_feasibility_used"]),
            "mpc_candidate_selection": int(
                candidate_decision_flags["mpc_candidate_selection"]),
            "mpc_execution_cost_in_score": int(
                candidate_decision_flags["mpc_execution_cost_in_score"]),
            "mpc_candidate_cost_breakdown": dict(
                candidate_decision_flags["mpc_candidate_cost_breakdown"]),
            "replan_required": True,
            "mpc_feedback": {
                "replan_required": True,
                "failure_type": "planner",
                "failed_constraint": "manifold",
                "selected_corridor_id": str(selected_corridor_id or ""),
                "failure_reason": "reference_manifold_infeasible",
                "failed_constraint_type": "manifold",
            },
        })
        return _apply_success_contract(result, reference_audit)
    if robot == "arm":
        arm_validation = _arm_mpc_input_validation(
            ref, topology_info, corridor_info, manifold_info, social_field,
            robot_constraints, context)
        if not bool(arm_validation.get("valid", False)):
            reason = str(arm_validation.get(
                "failure_reason", "ik_or_link_collision"))
            result = _empty_mpc_result(
                robot, selected_corridor_id, "ik_validated_trajectory",
                h, step_dt, robot_constraints, weights,
                "infeasible_arm_configuration_space", reason)
            result.update({
                "reference_path_count": int(len(ref)),
                "task_mode": str(task_mode),
                "task_weight": dict(task_weight),
                "task_weight_used": True,
                "module_chain_valid": False,
                "ik_validation": dict(arm_validation),
                "ik_valid": False,
                "link_collision_valid": bool(arm_validation.get(
                    "link_collision", {}).get("link_collision_valid", False)),
                "collision_link": str(arm_validation.get("collision_link", "")),
                "mpc_input_requires_ik_validation": True,
                "mpc_input_validation_stage": "pre_mpc",
            })
            return result

    state = np.asarray(current_state if current_state is not None else ref[0], float)
    mode = str(
        rollout_mode or cfg.get("rollout_mode", "rolling_window")
    ).strip().lower()
    if mode == "rolling_window" and len(ref) > 1:
        result = _run_rolling_tracking(
            robot, state, ref, social_field, robot_constraints, weights,
            step_dt, threshold, max(2, h), context, phase_sequence)
    else:
        horizon_count = min(max(1, h), len(ref))
        refs = ref[:horizon_count]
        if robot == "wheelchair":
            result = _run_wheelchair_tracking(
                state, refs, social_field, robot_constraints, weights,
                step_dt, threshold, context, phase_sequence[:horizon_count])
        else:
            result = _run_arm_tracking(
                state, refs, social_field, robot_constraints, weights,
                step_dt, threshold, context, phase_sequence[:horizon_count])
    result = _apply_acceptance_diagnostics(result, context)
    result["reference_safety_audit"] = reference_audit
    reference_source = str(
        context.get("corridor_info", {}).get(
            "reference_source",
            context.get("corridor_info", {}).get(
                "mpc_reference_source",
                context.get("manifold_info", {}).get(
                    "reference_source", ""))) or "")
    if not reference_source:
        reference_source = "selected_candidate_waypoints"
    candidate_decision_flags = _mpc_candidate_decision_flags(context)
    result.update({
        "robot_type": robot,
        "selected_corridor_id": str(selected_corridor_id or ""),
        "reference_source": reference_source,
        "reference_path_file": "mpc_reference_path.csv",
        "reference_path_count": int(len(ref)),
        "horizon": int(h),
        "dt": float(step_dt),
        "constraints": robot_constraints,
        "weights": weights,
        "task_mode": str(task_mode),
        "task_weight": dict(task_weight),
        "task_weight_used": True,
        "phase_cost_weights": dict(weights.get("phase_cost_weights", {})),
        "phase_weight_used": True,
        "constraint_version": "topology_soft_manifold_v3",
        "module_chain_valid": True,
        "ik_validation": dict(
            _arm_mpc_input_validation(
                ref, topology_info, corridor_info, manifold_info,
                social_field, robot_constraints, context)
            if robot == "arm" else {}),
        "mpc_input_requires_ik_validation": bool(robot == "arm"),
        "mpc_input_validation_stage": (
            "pre_mpc" if robot == "arm" else "not_required"),
        "control_sequence_count": len(result.get("control_sequence", [])),
        "predicted_state_count": len(result.get("predicted_states", [])),
        "rollout_horizon_rows": len(result.get("rollout_rows", [])),
        "executed_trajectory_count": len(result.get("executable_trajectory", [])),
        "executed_trajectory_file": "mpc_executed_trajectory.csv",
        "cost_breakdown_file": "mpc_cost_breakdown.csv",
        "mpc_affects_candidate_ranking": int(
            candidate_decision_flags["mpc_affects_candidate_ranking"]),
        "mpc_candidate_feasibility_used": int(
            candidate_decision_flags["mpc_candidate_feasibility_used"]),
        "mpc_candidate_selection": int(
            candidate_decision_flags["mpc_candidate_selection"]),
        "mpc_execution_cost_in_score": int(
            candidate_decision_flags["mpc_execution_cost_in_score"]),
        "mpc_candidate_cost_breakdown": dict(
            candidate_decision_flags["mpc_candidate_cost_breakdown"]),
        "topology_info": context["topology_info"],
        "corridor_info": context["corridor_info"],
        "manifold_info": context["manifold_info"],
        "topology_constraint": {
            "critical_point_constraint": {
                "used": True,
                "critical_points": context.get("critical_point_sequence", []),
                "association": context.get("critical_point_association", {}),
                "soft_radius": context.get("critical_point_soft_radius", 0.5),
                "hard_radius": context.get("critical_point_hard_radius", 1.0),
                "mode": context.get("critical_point_constraint_mode", "soft_hard"),
            },
            "corridor_constraint": {
                "used": True,
                "centerline": _jsonable_points(context.get("centerline", [])),
                "radius": context.get("radius", 0.0),
                "tube_constraint": dict(
                    context.get("topology_tube_constraint", {}) or {}),
            },
            "topology_tube_constraint": dict(
                context.get("topology_tube_constraint", {}) or {}),
            "manifold_constraint": {
                "used": True,
                "safe_threshold": context.get("safe_threshold", 1.0),
                "risk_threshold": context.get("safe_threshold", 1.0),
                "minimum_clearance": context.get("minimum_clearance", 0.0),
                "effective_minimum_clearance": result.get(
                    "effective_minimum_clearance", context.get("minimum_clearance", 0.0)),
                "boundary": context.get("manifold_boundary", []),
                "manifold_constraint_mode": context.get(
                    "manifold_constraint_mode", "soft"),
                "soft_constraint_used": bool(
                    context.get("manifold_constraint_mode", "soft") == "soft"),
                "distance_rule": "distance_to_boundary >= effective_minimum_clearance",
                "risk_rule": "risk_value <= effective_risk_threshold",
            },
            "topology_sequence_constraint": {
                "used": True,
                "topology_sequence_valid": result.get(
                    "topology_sequence_valid", True),
                "critical_point_status": result.get(
                    "critical_point_status", ""),
            },
        },
        "topology_class": context["topology_class"],
        "topology_constraint_used": True,
        "corridor_constraint_used": True,
        "tube_constraint_used": True,
        "tube_constraint_mode": "hard",
        "manifold_constraint_used": True,
        "manifold_constraint_mode": context.get(
            "manifold_constraint_mode", "soft"),
        "mpc_manifold_constraint_mode": context.get(
            "manifold_constraint_mode", "soft"),
        "soft_constraint_used": bool(
            context.get("manifold_constraint_mode", "soft") == "soft"),
        "minimum_clearance": result.get(
            "minimum_clearance", context.get("minimum_clearance", 0.0)),
        "risk_threshold": context.get("safe_threshold", threshold),
        "corridor_hard_constraint_used": True,
        "manifold_hard_constraint_used": bool(
            context.get("manifold_constraint_mode", "soft") != "soft"),
        "critical_point_constraint_used": True,
        "critical_point_sequence_constraint_used": True,
        "critical_point_association_used": bool(
            context.get("critical_point_association_used", False)),
        "topology_sequence_constraint_used": True,
        "topology_class_constraint_used": True,
    })
    feedback = dict(result.get("mpc_feedback", {}) or {})
    feedback["selected_corridor_id"] = str(selected_corridor_id or "")
    feedback.setdefault("failure_type", result.get("mpc_feasibility_status", ""))
    feedback.setdefault("failed_constraint", result.get("failed_constraint_type", ""))
    result["mpc_feedback"] = feedback
    measured = cfg.get("executed_trajectory", [])
    if _as_optional_points(measured).size:
        executed_rows, executed_summary = evaluate_executed_trajectory(
            measured, context, social_field=social_field, robot_type=robot,
            phase_sequence=cfg.get("executed_phase_sequence", []),
            corridor_active_sequence=cfg.get(
                "executed_corridor_active_sequence", []))
        result["executed_trajectory_rows"] = executed_rows
        result["actual_executed_trajectory_count"] = len(executed_rows)
        result["executed_trajectory_count"] = len(executed_rows)
        result["predicted_trajectory_count"] = len(
            result.get("predicted_states", []))
        result["actual_executable_trajectory"] = _jsonable_points(measured)
        for key, value in executed_summary.items():
            result["executed_" + key] = value
        result["executed_corridor_active_count"] = int(sum(
            1 for row in executed_rows if bool(row.get("corridor_active", True))))
        result["executed_corridor_out_of_scope_count"] = int(sum(
            1 for row in executed_rows
            if (not bool(row.get("corridor_active", True)) and
                float(row.get("corridor_out_of_scope_violation", 0.0)) > 1e-9)))
        result["actual_execution_min_clearance"] = float(
            executed_summary.get("min_manifold_clearance", 0.0))
        result["execution_clearance"] = float(
            executed_summary.get("min_manifold_clearance", 0.0))
        result["executed_min_clearance"] = float(
            executed_summary.get("min_manifold_clearance", 0.0))
        result["trajectory_evidence"] = {
            "reference": "mpc_reference_path.csv",
            "predicted": "mpc_rollout_log.csv",
            "executed": "mpc_executed_trajectory.csv",
        }
    else:
        result["actual_executed_trajectory_count"] = 0
    result["executed_evidence_required"] = bool(
        cfg.get("executed_evidence_required", False))
    return _apply_success_contract(result, reference_audit)


def _wheelchair_step(x, ref, prev_control, field, constraints, weights, dt,
                     risk_threshold, context, phase="approach"):
    phase_weights = _phase_weight_entry(weights, phase)
    v_prev = float(prev_control[0]) if len(prev_control) > 0 else 0.0
    w_prev = float(prev_control[1]) if len(prev_control) > 1 else 0.0
    v_max = float(constraints.get("v_max", 0.5))
    v_min = float(constraints.get("v_min", 0.0))
    w_max = float(constraints.get("omega_max", constraints.get("w_max", 1.0)))
    a_max = float(constraints.get("a_max", 0.5))
    alpha_max = float(constraints.get("alpha_max", 1.0))
    curvature_max = float(constraints.get("curvature_max", 1.5))
    task_ref = np.asarray(ref, float).copy()
    ref_path = _as_optional_points((context or {}).get("reference_path", []))
    goal_distance_error = 0.0
    task_reference_blend = 0.0
    if len(ref_path):
        goal_xy = ref_path[-1, :2]
        goal_delta = goal_xy - np.asarray(x, float)[:2]
        goal_distance_error = float(np.linalg.norm(goal_delta))
        task_reference_blend = 0.35
        if goal_distance_error < 1.2:
            task_reference_blend = 0.70
        if goal_distance_error < 0.55:
            task_reference_blend = 0.92
        task_ref[:2] = (
            (1.0 - task_reference_blend) * task_ref[:2] +
            task_reference_blend * goal_xy)
    dx = float(task_ref[0] - x[0])
    dy = float(task_ref[1] - x[1])
    desired = np.arctan2(dy, dx)
    heading_error = float(np.arctan2(
        np.sin(desired - x[2]), np.cos(desired - x[2])))
    dist = float(np.hypot(dx, dy))
    v_cmd = min(v_max, max(v_min, dist / max(dt, 1e-9)))
    v_cmd *= max(0.0, np.cos(heading_error))
    w_cmd = float(np.clip(2.0 * heading_error, -w_max, w_max))
    v_cmd = float(np.clip(v_cmd, v_prev - a_max * dt, v_prev + a_max * dt))
    v_cmd = float(np.clip(v_cmd, v_min, v_max))
    w_cmd = float(np.clip(w_cmd, w_prev - alpha_max * dt, w_prev + alpha_max * dt))
    w_cmd = float(np.clip(w_cmd, -w_max, w_max))
    curvature = abs(w_cmd) / max(abs(v_cmd), 1e-6)
    if abs(v_cmd) < 0.05:
        curvature = 0.0
    constraint_violation = max(0.0, curvature - curvature_max)
    v_cmd, w_cmd, corridor_override, manifold_override, safety_stop = (
        _wheelchair_safety_override(
            x, v_cmd, w_cmd, dt, field, context, constraints))
    nxt = np.array(x, float)
    nxt[0] += v_cmd * np.cos(nxt[2]) * dt
    nxt[1] += v_cmd * np.sin(nxt[2]) * dt
    nxt[2] += w_cmd * dt
    nxt[2] = np.arctan2(np.sin(nxt[2]), np.cos(nxt[2]))
    constrained, risk_info = _apply_hard_constraints(
        [nxt[0], nxt[1], 0.0], "wheelchair", field, context)
    nxt[0] = float(constrained[0])
    nxt[1] = float(constrained[1])
    risk = float(risk_info.get("risk_value", 0.0))
    tracking_error = float(np.linalg.norm(nxt[:2] - ref[:2]))
    control_norm = float(np.linalg.norm([v_cmd, w_cmd]))
    control_delta = float(np.linalg.norm([v_cmd - v_prev, w_cmd - w_prev]))
    safety_violation = max(0.0, risk - risk_threshold)
    task_optimization_cost = float(
        goal_distance_error ** 2 + heading_error ** 2)
    extra = _constraint_step_metrics([nxt[0], nxt[1], 0.0], risk, context)
    step_cost = (
        float(phase_weights.get("track", 1.0)) * tracking_error ** 2 +
        float(phase_weights.get("control", 0.1)) * control_norm ** 2 +
        float(phase_weights.get("smooth", 0.2)) * control_delta ** 2 +
        float(phase_weights.get("risk", 1.0)) * risk +
        float(phase_weights.get("topology", 5.0)) *
        extra["topology_consistency_cost"] +
        float(phase_weights.get("corridor", 10.0)) * extra["corridor_cost"] +
        float(phase_weights.get("manifold", 10.0)) * extra["manifold_cost"] +
        float(phase_weights.get("safety_violation", 5.0)) * safety_violation ** 2 +
        float(phase_weights.get("constraint_violation", 10.0)) *
        constraint_violation ** 2 +
        float(phase_weights.get("task_weight_effective", 0.0)) *
        task_optimization_cost)
    row = {
        "pred_x": float(nxt[0]), "pred_y": float(nxt[1]),
        "pred_z": 0.0, "pred_theta": float(nxt[2]),
        "v_cmd": v_cmd, "omega_cmd": w_cmd,
        "corridor_safety_override": bool(corridor_override),
        "manifold_safety_override": bool(manifold_override),
        "corridor_override": bool(corridor_override),
        "manifold_override": bool(manifold_override),
        "safety_stop_override": bool(safety_stop),
        "tracking_error": tracking_error,
        "heading_error": heading_error,
        "goal_distance_error": goal_distance_error,
        "task_reference_blend": task_reference_blend,
        "control_norm": control_norm,
        "control_delta": control_delta,
        "risk": risk, "risk_value": risk,
        "risk_query_valid": int(bool(risk_info.get("risk_query_valid"))),
        "risk_query_source": risk_info.get("risk_query_source", ""),
        "risk_query_failure_reason": risk_info.get("risk_query_failure_reason", ""),
        "safety_violation": safety_violation,
        "risk_threshold": risk_threshold,
        "risk_exceeded": int(risk > risk_threshold),
        "constraint_violation": constraint_violation,
        "handover_cost": 0.0,
        "task_cost": float(task_optimization_cost),
        "task_optimization_cost": float(task_optimization_cost),
        "progress": 0.0,
        "phase": str(phase_weights.get("phase", phase)),
        "risk_weight": float(phase_weights.get("risk_weight", 0.0)),
        "tracking_weight": float(phase_weights.get("tracking_weight", 0.0)),
        "smoothness_weight": float(phase_weights.get("smoothness_weight", 0.0)),
        "task_weight": float(phase_weights.get("task_weight_effective", 0.0)),
        "step_cost": step_cost,
    }
    row.update(extra)
    return nxt, [v_cmd, w_cmd], row


def _arm_step(x, ref, prev_control, field, constraints, weights, dt,
              risk_threshold, goal, context, phase="approach", progress=0.0):
    phase_weights = _phase_weight_entry(weights, phase)
    ee_speed_max = float(constraints.get("ee_speed_max", 0.3))
    delta_max = float(constraints.get("control_delta_max", 0.1))
    prev = np.asarray(prev_control if len(prev_control) >= 3 else [0.0, 0.0, 0.0], float)[:3]
    u = (ref[:3] - x[:3]) / max(dt, 1e-9)
    speed = float(np.linalg.norm(u))
    constraint_violation = 0.0
    if speed > ee_speed_max and speed > 1e-9:
        constraint_violation += speed - ee_speed_max
        u = u * (ee_speed_max / speed)
    du = u - prev
    du_norm = float(np.linalg.norm(du))
    if du_norm > delta_max and du_norm > 1e-9:
        constraint_violation += du_norm - delta_max
        u = prev + du * (delta_max / du_norm)
    u, corridor_override, manifold_override, safety_stop = _arm_safety_override(
        x, u, dt, field, context, constraints)
    nxt = x[:3] + u * dt
    nxt, risk_info = _apply_hard_constraints(nxt, "arm", field, context)
    risk = float(risk_info.get("risk_value", 0.0))
    tracking_error = float(np.linalg.norm(nxt[:3] - ref[:3]))
    control_norm = float(np.linalg.norm(u))
    control_delta = float(np.linalg.norm(u - prev))
    safety_violation = max(0.0, risk - risk_threshold)
    handover_cost = float(np.linalg.norm(nxt[:3] - goal[:3]) ** 2)
    task_optimization_cost = float(handover_cost + safety_violation ** 2)
    extra = _constraint_step_metrics(
        nxt, risk, context, phase=phase, progress=progress)
    step_cost = (
        float(phase_weights.get("track", 1.0)) * tracking_error ** 2 +
        float(phase_weights.get("control", 0.1)) * control_norm ** 2 +
        float(phase_weights.get("smooth", 0.2)) * control_delta ** 2 +
        float(phase_weights.get("risk", 1.0)) * risk +
        float(phase_weights.get("topology", 5.0)) *
        extra["topology_consistency_cost"] +
        float(phase_weights.get("corridor", 10.0)) * extra["corridor_cost"] +
        float(phase_weights.get("manifold", 10.0)) * extra["manifold_cost"] +
        float(phase_weights.get("safety_violation", 5.0)) * safety_violation ** 2 +
        float(phase_weights.get("constraint_violation", 10.0)) *
        constraint_violation ** 2 +
        float(phase_weights.get("handover", 1.0)) * handover_cost +
        float(phase_weights.get("task_weight_effective", 0.0)) *
        task_optimization_cost)
    row = {
        "pred_x": float(nxt[0]), "pred_y": float(nxt[1]),
        "pred_z": float(nxt[2]), "pred_theta": float(nxt[2]),
        "corridor_safety_override": bool(corridor_override),
        "manifold_safety_override": bool(manifold_override),
        "corridor_override": bool(corridor_override),
        "manifold_override": bool(manifold_override),
        "safety_stop_override": bool(safety_stop),
        "tracking_error": tracking_error,
        "control_norm": control_norm,
        "control_delta": control_delta,
        "risk": risk, "risk_value": risk,
        "risk_query_valid": int(bool(risk_info.get("risk_query_valid"))),
        "risk_query_source": risk_info.get("risk_query_source", ""),
        "risk_query_failure_reason": risk_info.get("risk_query_failure_reason", ""),
        "safety_violation": safety_violation,
        "risk_threshold": risk_threshold,
        "risk_exceeded": int(risk > risk_threshold),
        "constraint_violation": constraint_violation,
        "handover_cost": handover_cost,
        "task_cost": float(task_optimization_cost),
        "task_optimization_cost": float(task_optimization_cost),
        "progress": float(np.clip(float(progress or 0.0), 0.0, 1.0)),
        "phase": str(phase_weights.get("phase", phase)),
        "risk_weight": float(phase_weights.get("risk_weight", 0.0)),
        "tracking_weight": float(phase_weights.get("tracking_weight", 0.0)),
        "smoothness_weight": float(phase_weights.get("smoothness_weight", 0.0)),
        "task_weight": float(phase_weights.get("task_weight_effective", 0.0)),
        "step_cost": step_cost,
    }
    row.update(extra)
    return nxt, [float(u[0]), float(u[1]), float(u[2])], row


def _arm_early_stop_satisfied(state, ref, goal, control_norm, constraints):
    state = np.asarray(state, float)[:3]
    ref = np.asarray(ref, float)[:3]
    goal = np.asarray(goal, float)[:3]
    goal_tol = float(constraints.get(
        "early_stop_goal_tolerance",
        constraints.get("target_region_tolerance",
                        constraints.get("goal_tolerance", 0.03))))
    tracking_tol = float(constraints.get(
        "early_stop_tracking_tolerance",
        constraints.get("tracking_tolerance", 0.02)))
    speed_tol = float(constraints.get(
        "early_stop_speed_tolerance",
        constraints.get("speed_tolerance", 0.01)))
    goal_error = float(np.linalg.norm(state - goal))
    tracking_error = float(np.linalg.norm(state - ref))
    return (
        goal_error <= goal_tol and
        tracking_error <= tracking_tol and
        float(control_norm) <= speed_tol)


def _run_rolling_tracking(robot, state, refs, field, constraints, weights, dt,
                          risk_threshold, horizon, context,
                          phase_sequence=None):
    if robot == "wheelchair":
        actual = np.array([
            float(state[0]), float(state[1]),
            float(state[2]) if len(state) > 2 else 0.0], float)
        prev_control = [float(state[3]) if len(state) > 3 else 0.0,
                        float(state[4]) if len(state) > 4 else 0.0]
    else:
        actual = np.asarray(state if state is not None and len(state) >= 3 else refs[0], float)[:3]
        prev_control = [0.0, 0.0, 0.0]
    rollout_rows = []
    executed_rows = []
    summary_rows = []
    controls = []
    executed_states = []
    predicted_states = []
    tracking_errors = []
    risks = []
    control_norms = []
    control_deltas = []
    costs = {
        "tracking_cost": 0.0, "control_cost": 0.0,
        "smoothness_cost": 0.0, "risk_cost": 0.0,
        "topology_cost": 0.0, "corridor_cost": 0.0,
        "manifold_cost": 0.0,
        "safety_violation_cost": 0.0,
        "constraint_violation_cost": 0.0, "handover_cost": 0.0,
        "task_cost": 0.0,
    }
    violation_count = 0
    cumulative = 0.0
    max_solves = max(1, len(refs) - 1)
    goal = refs[-1]
    early_terminated = False
    rolling_stop_reason = "reference_end"
    if (robot == "arm" and
            _arm_early_stop_satisfied(actual, goal, goal, 0.0, constraints)):
        refs = np.asarray([goal], float)
        max_solves = 1
    status = "feasible"
    phase_sequence = list(phase_sequence or _reference_phases(
        refs, len(refs), robot))
    for solve_index in range(max_solves):
        segment = refs[solve_index:min(len(refs), solve_index + int(horizon))]
        if len(segment) < 1:
            break
        rollout_start = len(rollout_rows)
        pred = np.array(actual, float)
        pred_prev = list(prev_control)
        first_row = None
        first_control = None
        first_state = None
        horizon_tracking = 0.0
        horizon_control = 0.0
        horizon_smooth = 0.0
        horizon_risk = 0.0
        horizon_safety = 0.0
        horizon_constraint = 0.0
        horizon_handover = 0.0
        horizon_topology = 0.0
        horizon_corridor = 0.0
        horizon_manifold = 0.0
        for horizon_step, ref in enumerate(segment):
            ref_index = min(solve_index + horizon_step, len(phase_sequence) - 1)
            phase = phase_sequence[ref_index] if phase_sequence else "approach"
            if robot == "wheelchair":
                pred, ctrl, row = _wheelchair_step(
                    pred, ref, pred_prev, field, constraints, weights, dt,
                    risk_threshold, context, phase=phase)
            else:
                progress = _phase_local_progress(phase_sequence, ref_index)
                pred, ctrl, row = _arm_step(
                    pred, ref, pred_prev, field, constraints, weights, dt,
                    risk_threshold, goal, context, phase=phase,
                    progress=progress)
            pred_prev = ctrl
            pred_xyz = [float(pred[0]), float(pred[1]), float(pred[2])]
            predicted_states.append(pred_xyz)
            row.update({
                "robot": robot,
                "corridor_id": "",
                "rollout_mode": "rolling_window",
                "solve_index": solve_index,
                "horizon_step": horizon_step,
                "global_step": solve_index,
                "ref_index": solve_index + horizon_step,
                "ref_x": float(ref[0]), "ref_y": float(ref[1]),
                "ref_z": float(ref[2]),
                "ref_theta": float(ref[2]),
                "phase": str(row.get("phase", phase)),
                "first_control_applied": bool(horizon_step == 0),
                "feasibility_status": status,
            })
            rollout_rows.append(row)
            horizon_tracking += row["tracking_error"] ** 2
            horizon_control += row["control_norm"] ** 2
            horizon_smooth += row["control_delta"] ** 2
            horizon_risk += row["risk_value"]
            horizon_topology += row["topology_consistency_cost"]
            horizon_corridor += row["corridor_cost"]
            horizon_manifold += row["manifold_cost"]
            horizon_safety += row["safety_violation"] ** 2
            horizon_constraint += row["constraint_violation"] ** 2
            horizon_handover += row.get("handover_cost", 0.0)
            if first_row is None:
                first_row = dict(row)
                first_control = list(ctrl)
                first_state = np.array(pred, float)
        if first_row is None:
            break
        actual = first_state
        prev_control = first_control
        for row in rollout_rows[rollout_start:]:
            row["executed_x"] = float(actual[0])
            row["executed_y"] = float(actual[1])
            row["executed_z"] = 0.0 if robot == "wheelchair" else float(actual[2])
            row["executed_theta"] = float(actual[2]) if robot == "wheelchair" else ""
        controls.append(first_control)
        executed = [float(actual[0]), float(actual[1]), float(actual[2])]
        executed_states.append(executed)
        tracking_errors.append(first_row["tracking_error"])
        risks.append(first_row["risk_value"])
        control_norms.append(first_row["control_norm"])
        control_deltas.append(first_row["control_delta"])
        if (first_row["risk_exceeded"] or
                first_row["constraint_violation"] > 1e-9 or
                first_row["corridor_violation"] > 1e-9 or
                first_row["manifold_violation"] > 1e-9 or
                int(first_row.get("topology_violation", 0))):
            violation_count += 1
        costs["tracking_cost"] += first_row["tracking_error"] ** 2
        costs["control_cost"] += first_row["control_norm"] ** 2
        costs["smoothness_cost"] += first_row["control_delta"] ** 2
        costs["risk_cost"] += first_row["risk_value"]
        costs["topology_cost"] += first_row["topology_consistency_cost"]
        costs["corridor_cost"] += first_row["corridor_cost"]
        costs["manifold_cost"] += first_row["manifold_cost"]
        costs["safety_violation_cost"] += first_row["safety_violation"] ** 2
        costs["constraint_violation_cost"] += first_row["constraint_violation"] ** 2
        costs["handover_cost"] += first_row.get("handover_cost", 0.0)
        costs["task_cost"] += float(first_row.get(
            "task_optimization_cost", first_row.get("task_cost", 0.0)) or 0.0)
        first_step_total = first_row["step_cost"]
        cumulative += first_step_total
        summary_rows.append({
            "robot": robot, "corridor_id": "",
            "rollout_mode": "rolling_window",
            "solve_index": solve_index,
            "horizon_step": 0,
            "global_step": solve_index,
            "tracking_cost": first_row["tracking_error"] ** 2,
            "control_cost": first_row["control_norm"] ** 2,
            "smoothness_cost": first_row["control_delta"] ** 2,
            "risk_cost": first_row["risk_value"],
            "topology_cost": first_row["topology_consistency_cost"],
            "corridor_cost": first_row["corridor_cost"],
            "manifold_cost": first_row["manifold_cost"],
            "safety_violation_cost": first_row["safety_violation"] ** 2,
            "constraint_violation_cost": first_row["constraint_violation"] ** 2,
            "handover_cost": first_row.get("handover_cost", 0.0),
            "task_cost": float(first_row.get(
                "task_optimization_cost", first_row.get("task_cost", 0.0)) or 0.0),
            "task_optimization_cost": float(first_row.get(
                "task_optimization_cost", first_row.get("task_cost", 0.0)) or 0.0),
            "phase": str(first_row.get("phase", "")),
            "risk_weight": float(first_row.get("risk_weight", 0.0)),
            "tracking_weight": float(first_row.get("tracking_weight", 0.0)),
            "task_weight": float(first_row.get("task_weight", 0.0)),
            "step_total_cost": first_step_total,
            "cumulative_total_cost": cumulative,
            "feasibility_status": status,
        })
        executed_rows.append({
            "robot": robot, "corridor_id": "",
            "global_step": solve_index,
            "x": executed[0], "y": executed[1], "z": 0.0 if robot == "wheelchair" else executed[2],
            "theta": executed[2] if robot == "wheelchair" else "",
            "joint_state": "" if robot == "wheelchair" else json.dumps(executed),
            "control_norm": first_row["control_norm"],
            "tracking_error": first_row["tracking_error"],
            "risk_value": first_row["risk_value"],
            "phase": str(first_row.get("phase", "")),
            "risk_weight": float(first_row.get("risk_weight", 0.0)),
            "tracking_weight": float(first_row.get("tracking_weight", 0.0)),
            "task_weight": float(first_row.get("task_weight", 0.0)),
            "feasibility_status": status,
        })
        if (robot == "arm" and
                _arm_early_stop_satisfied(
                    actual, goal, goal, first_row["control_norm"], constraints)):
            early_terminated = True
            rolling_stop_reason = "arm_goal_tracking_speed_converged"
            break
    result = _finish_mpc_result(costs, [], controls, executed_states,
                                tracking_errors, risks, control_norms,
                                control_deltas, violation_count,
                                risk_threshold, weights, context)
    sequence_summary = _annotate_critical_sequence(rollout_rows, context)
    summary = _constraint_summary(rollout_rows)
    summary.update(sequence_summary)
    summary.update(_critical_sequence_summary(rollout_rows, context))
    status, failure_reason, failed_type = _feasibility_from_constraints(
        summary, result.get("risk_exceed_count", 0))
    result.update({
        "mpc_feasibility_status": status,
        "failure_reason": failure_reason,
        "mpc_failure_reason": failure_reason,
        "failed_constraint_type": failed_type,
        "replan_required": bool(status in (
            "topology_infeasible", "manifold_infeasible",
            "corridor_infeasible")),
        "mpc_feedback": {
            "replan_required": bool(status in (
                "topology_infeasible", "manifold_infeasible",
                "corridor_infeasible")),
            "failure_type": failed_type,
            "failed_constraint": failed_type,
            "selected_corridor_id": str(context.get(
                "corridor_info", {}).get("corridor_id", "")),
            "failure_reason": failure_reason,
            "failed_constraint_type": failed_type,
        },
        "executable_trajectory": executed_states,
        "control_sequence": controls,
        "predicted_states": predicted_states,
        "cost_breakdown_rows": summary_rows,
        "rollout_rows": rollout_rows,
        "executed_trajectory_rows": executed_rows,
        "rollout_mode": "rolling_window",
        "rollout_solve_count": len(summary_rows),
        "rollout_horizon_rows": len(rollout_rows),
        "executed_trajectory_count": len(executed_rows),
        "total_cost": float(cumulative),
        "executed_trajectory_file": "mpc_executed_trajectory.csv",
        "cost_breakdown_file": "mpc_cost_breakdown.csv",
        "rolling_goal_reached": bool(early_terminated or len(executed_rows) > 0),
        "rolling_early_terminated": bool(early_terminated),
        "rolling_stop_reason": (
            rolling_stop_reason if len(executed_rows) > 0
            else "no_reference_segment"),
    })
    result.update(summary)
    valid_count = sum(
        1 for row in rollout_rows if bool(int(row.get("risk_query_valid", 0) or 0)))
    invalid_count = max(0, len(rollout_rows) - valid_count)
    sources = [
        str(row.get("risk_query_source", ""))
        for row in rollout_rows if str(row.get("risk_query_source", "")).strip()
    ]
    source = sources[0] if sources else "unavailable"
    if not rollout_rows or valid_count <= 0:
        sanity = "failed_no_risk_query"
        warning = "risk_query_not_valid"
    elif source == "fallback_zero":
        sanity = "warning_fallback_zero"
        warning = "social_field_unavailable_fallback_zero"
    elif max(risks) <= 1e-12:
        sanity = "pass_all_low_risk"
        warning = ""
    else:
        sanity = "pass"
        warning = ""
    if bool(context.get("strict_risk_query", False)) and invalid_count > 0:
        status = "risk_query_invalid"
        failure_reason = "risk_query_invalid"
        failed_type = "risk_query"
        result.update({
            "mpc_feasibility_status": status,
            "final_status": status,
            "final_mpc_status": status,
            "failure_reason": failure_reason,
            "mpc_failure_reason": failure_reason,
            "failed_constraint_type": failed_type,
            "replan_required": True,
        })
        feedback = dict(result.get("mpc_feedback", {}) or {})
        feedback.update({
            "replan_required": True,
            "failure_type": failed_type,
            "failed_constraint": failed_type,
            "failure_reason": failure_reason,
            "failed_constraint_type": failed_type,
        })
        result["mpc_feedback"] = feedback
    result.update({
        "risk_query_called": bool(rollout_rows),
        "risk_query_valid_count": int(valid_count),
        "risk_query_invalid_count": int(invalid_count),
        "risk_query_source": source,
        "risk_sanity_status": sanity,
        "risk_sanity_warning": warning,
    })
    for row in rollout_rows:
        row["feasibility_status"] = status
        row["risk_sanity_status"] = sanity
    for row in summary_rows:
        row["feasibility_status"] = status
    for row in executed_rows:
        row["feasibility_status"] = status
    return result


def _run_wheelchair_tracking(state, refs, field, constraints, weights, dt,
                             risk_threshold, context, phase_sequence=None):
    x = np.array([
        float(state[0]), float(state[1]),
        float(state[2]) if len(state) > 2 else 0.0], float)
    v_prev = float(state[3]) if len(state) > 3 else 0.0
    w_prev = float(state[4]) if len(state) > 4 else 0.0
    v_max = float(constraints.get("v_max", 0.5))
    v_min = float(constraints.get("v_min", 0.0))
    w_max = float(constraints.get("omega_max", constraints.get("w_max", 1.0)))
    a_max = float(constraints.get("a_max", 0.5))
    alpha_max = float(constraints.get("alpha_max", 1.0))
    curvature_max = float(constraints.get("curvature_max", 1.5))
    rows = []
    controls = []
    states = []
    costs = {
        "tracking_cost": 0.0,
        "control_cost": 0.0,
        "smoothness_cost": 0.0,
        "risk_cost": 0.0,
        "topology_cost": 0.0,
        "corridor_cost": 0.0,
        "manifold_cost": 0.0,
        "safety_violation_cost": 0.0,
        "constraint_violation_cost": 0.0,
        "handover_cost": 0.0,
        "task_cost": 0.0,
    }
    tracking_errors = []
    risks = []
    control_norms = []
    control_deltas = []
    violation_count = 0
    phase_sequence = list(phase_sequence or _reference_phases(
        refs, len(refs), "wheelchair"))
    goal_xy = np.asarray(refs[-1], float)[:2]
    for step, ref in enumerate(refs):
        phase = phase_sequence[min(step, len(phase_sequence) - 1)]
        progress = _phase_local_progress(phase_sequence, step)
        phase_weights = _phase_weight_entry(weights, phase)
        task_ref = np.asarray(ref, float).copy()
        goal_delta = goal_xy - x[:2]
        goal_distance_error = float(np.linalg.norm(goal_delta))
        task_reference_blend = 0.35
        if goal_distance_error < 1.2:
            task_reference_blend = 0.70
        if goal_distance_error < 0.55:
            task_reference_blend = 0.92
        task_ref[:2] = (
            (1.0 - task_reference_blend) * task_ref[:2] +
            task_reference_blend * goal_xy)
        dx = float(task_ref[0] - x[0])
        dy = float(task_ref[1] - x[1])
        desired = np.arctan2(dy, dx)
        heading_error = float(np.arctan2(
            np.sin(desired - x[2]), np.cos(desired - x[2])))
        dist = float(np.hypot(dx, dy))
        v_cmd = min(v_max, max(v_min, dist / max(dt, 1e-9)))
        v_cmd *= max(0.0, np.cos(heading_error))
        w_cmd = float(np.clip(2.0 * heading_error, -w_max, w_max))
        v_cmd = float(np.clip(
            v_cmd, v_prev - a_max * dt, v_prev + a_max * dt))
        v_cmd = float(np.clip(v_cmd, v_min, v_max))
        w_cmd = float(np.clip(
            w_cmd, w_prev - alpha_max * dt, w_prev + alpha_max * dt))
        w_cmd = float(np.clip(w_cmd, -w_max, w_max))
        curvature = abs(w_cmd) / max(abs(v_cmd), 1e-6)
        if abs(v_cmd) < 0.05:
            curvature = 0.0
        constraint_violation = max(0.0, curvature - curvature_max)
        v_cmd, w_cmd, corridor_override, manifold_override, safety_stop = (
            _wheelchair_safety_override(
                x, v_cmd, w_cmd, dt, field, context, constraints))
        if constraint_violation > 1e-9:
            violation_count += 1
        x[0] += v_cmd * np.cos(x[2]) * dt
        x[1] += v_cmd * np.sin(x[2]) * dt
        x[2] += w_cmd * dt
        x[2] = np.arctan2(np.sin(x[2]), np.cos(x[2]))
        constrained, risk_info = _apply_hard_constraints(
            [x[0], x[1], 0.0], "wheelchair", field, context)
        x[0] = float(constrained[0])
        x[1] = float(constrained[1])
        risk = float(risk_info.get("risk_value", 0.0))
        risk_exceeded = risk > risk_threshold
        if risk_exceeded:
            violation_count += 1
        tracking_error = float(np.linalg.norm(x[:2] - ref[:2]))
        control_norm = float(np.linalg.norm([v_cmd, w_cmd]))
        control_delta = float(np.linalg.norm([v_cmd - v_prev, w_cmd - w_prev]))
        safety_violation = max(0.0, risk - risk_threshold)
        task_optimization_cost = float(
            goal_distance_error ** 2 + heading_error ** 2)
        extra = _constraint_step_metrics([x[0], x[1], 0.0], risk, context)
        if (extra["corridor_violation"] > 1e-9 or
                extra["manifold_violation"] > 1e-9 or
                int(extra["topology_violation"])):
            violation_count += 1
        step_cost = (
            float(phase_weights.get("track", 1.0)) * tracking_error ** 2 +
            float(phase_weights.get("control", 0.1)) * control_norm ** 2 +
            float(phase_weights.get("smooth", 0.2)) * control_delta ** 2 +
            float(phase_weights.get("risk", 1.0)) * risk +
            float(phase_weights.get("topology", 5.0)) *
            extra["topology_consistency_cost"] +
            float(phase_weights.get("corridor", 10.0)) * extra["corridor_cost"] +
            float(phase_weights.get("manifold", 10.0)) * extra["manifold_cost"] +
            float(phase_weights.get("safety_violation", 5.0)) * safety_violation ** 2 +
            float(phase_weights.get("constraint_violation", 10.0)) *
            constraint_violation ** 2 +
            float(phase_weights.get("task_weight_effective", 0.0)) *
            task_optimization_cost)
        costs["tracking_cost"] += tracking_error ** 2
        costs["control_cost"] += control_norm ** 2
        costs["smoothness_cost"] += control_delta ** 2
        costs["risk_cost"] += risk
        costs["topology_cost"] += extra["topology_consistency_cost"]
        costs["corridor_cost"] += extra["corridor_cost"]
        costs["manifold_cost"] += extra["manifold_cost"]
        costs["safety_violation_cost"] += safety_violation ** 2
        costs["constraint_violation_cost"] += constraint_violation ** 2
        costs["task_cost"] += task_optimization_cost
        costs["task_cost"] += task_optimization_cost
        controls.append([v_cmd, w_cmd])
        states.append([float(x[0]), float(x[1]), float(x[2])])
        tracking_errors.append(tracking_error)
        risks.append(risk)
        control_norms.append(control_norm)
        control_deltas.append(control_delta)
        row = {
            "step": step,
            "ref_x": float(ref[0]),
            "ref_y": float(ref[1]),
            "ref_z": float(ref[2]),
            "pred_x": float(x[0]),
            "pred_y": float(x[1]),
            "pred_z": 0.0,
            "corridor_safety_override": bool(corridor_override),
            "manifold_safety_override": bool(manifold_override),
            "corridor_override": bool(corridor_override),
            "manifold_override": bool(manifold_override),
            "safety_stop_override": bool(safety_stop),
            "tracking_error": tracking_error,
            "heading_error": heading_error,
            "goal_distance_error": goal_distance_error,
            "task_reference_blend": task_reference_blend,
            "control_norm": control_norm,
            "control_delta": control_delta,
            "risk": risk,
            "risk_value": risk,
            "risk_query_valid": int(bool(risk_info.get("risk_query_valid"))),
            "risk_query_source": risk_info.get("risk_query_source", ""),
            "risk_query_failure_reason": risk_info.get(
                "risk_query_failure_reason", ""),
            "safety_violation": safety_violation,
            "risk_threshold": risk_threshold,
            "risk_exceeded": int(risk_exceeded),
            "constraint_violation": constraint_violation,
            "task_cost": float(task_optimization_cost),
            "task_optimization_cost": float(task_optimization_cost),
            "progress": float(progress),
            "phase": str(phase_weights.get("phase", phase)),
            "risk_weight": float(phase_weights.get("risk_weight", 0.0)),
            "tracking_weight": float(phase_weights.get("tracking_weight", 0.0)),
            "smoothness_weight": float(phase_weights.get("smoothness_weight", 0.0)),
            "task_weight": float(phase_weights.get("task_weight_effective", 0.0)),
            "step_cost": step_cost,
        }
        row.update(extra)
        rows.append(row)
        v_prev, w_prev = v_cmd, w_cmd
    return _finish_mpc_result(costs, rows, controls, states, tracking_errors,
                              risks, control_norms, control_deltas,
                              violation_count, risk_threshold, weights, context)


def _run_arm_tracking(state, refs, field, constraints, weights, dt,
                      risk_threshold, context, phase_sequence=None):
    x = np.asarray(state if state is not None and len(state) >= 3 else refs[0], float)[:3]
    ee_speed_max = float(constraints.get("ee_speed_max", 0.3))
    delta_max = float(constraints.get("control_delta_max", 0.1))
    rows = []
    controls = []
    states = []
    costs = {
        "tracking_cost": 0.0,
        "control_cost": 0.0,
        "smoothness_cost": 0.0,
        "risk_cost": 0.0,
        "topology_cost": 0.0,
        "corridor_cost": 0.0,
        "manifold_cost": 0.0,
        "safety_violation_cost": 0.0,
        "constraint_violation_cost": 0.0,
        "handover_cost": 0.0,
        "task_cost": 0.0,
    }
    prev_u = np.zeros(3, float)
    tracking_errors = []
    risks = []
    control_norms = []
    control_deltas = []
    violation_count = 0
    goal = refs[-1]
    phase_sequence = list(phase_sequence or _reference_phases(
        refs, len(refs), "arm"))
    for step, ref in enumerate(refs):
        phase = phase_sequence[min(step, len(phase_sequence) - 1)]
        progress = _phase_local_progress(phase_sequence, step)
        phase_weights = _phase_weight_entry(weights, phase)
        err = ref[:3] - x[:3]
        u = err / max(dt, 1e-9)
        speed = float(np.linalg.norm(u))
        constraint_violation = 0.0
        if speed > ee_speed_max and speed > 1e-9:
            constraint_violation += speed - ee_speed_max
            u = u * (ee_speed_max / speed)
        du = u - prev_u
        du_norm = float(np.linalg.norm(du))
        if du_norm > delta_max and du_norm > 1e-9:
            constraint_violation += du_norm - delta_max
            u = prev_u + du * (delta_max / du_norm)
        u, corridor_override, manifold_override, safety_stop = _arm_safety_override(
            x, u, dt, field, context, constraints)
        if constraint_violation > 1e-9:
            violation_count += 1
        x = x + u * dt
        x, risk_info = _apply_hard_constraints(x, "arm", field, context)
        risk = float(risk_info.get("risk_value", 0.0))
        risk_exceeded = risk > risk_threshold
        if risk_exceeded:
            violation_count += 1
        tracking_error = float(np.linalg.norm(x[:3] - ref[:3]))
        control_norm = float(np.linalg.norm(u))
        control_delta = float(np.linalg.norm(u - prev_u))
        safety_violation = max(0.0, risk - risk_threshold)
        handover_cost = float(np.linalg.norm(x[:3] - goal[:3]) ** 2)
        task_optimization_cost = float(handover_cost + safety_violation ** 2)
        extra = _constraint_step_metrics(
            x, risk, context, phase=phase, progress=progress)
        if (extra["corridor_violation"] > 1e-9 or
                extra["manifold_violation"] > 1e-9 or
                int(extra["topology_violation"])):
            violation_count += 1
        step_cost = (
            float(phase_weights.get("track", 1.0)) * tracking_error ** 2 +
            float(phase_weights.get("control", 0.1)) * control_norm ** 2 +
            float(phase_weights.get("smooth", 0.2)) * control_delta ** 2 +
            float(phase_weights.get("risk", 1.0)) * risk +
            float(phase_weights.get("topology", 5.0)) *
            extra["topology_consistency_cost"] +
            float(phase_weights.get("corridor", 10.0)) * extra["corridor_cost"] +
            float(phase_weights.get("manifold", 10.0)) * extra["manifold_cost"] +
            float(phase_weights.get("safety_violation", 5.0)) * safety_violation ** 2 +
            float(phase_weights.get("constraint_violation", 10.0)) *
            constraint_violation ** 2 +
            float(phase_weights.get("handover", 1.0)) * handover_cost +
            float(phase_weights.get("task_weight_effective", 0.0)) *
            task_optimization_cost)
        costs["tracking_cost"] += tracking_error ** 2
        costs["control_cost"] += control_norm ** 2
        costs["smoothness_cost"] += control_delta ** 2
        costs["risk_cost"] += risk
        costs["topology_cost"] += extra["topology_consistency_cost"]
        costs["corridor_cost"] += extra["corridor_cost"]
        costs["manifold_cost"] += extra["manifold_cost"]
        costs["safety_violation_cost"] += safety_violation ** 2
        costs["constraint_violation_cost"] += constraint_violation ** 2
        costs["handover_cost"] += handover_cost
        costs["task_cost"] += task_optimization_cost
        controls.append([float(u[0]), float(u[1]), float(u[2])])
        states.append([float(x[0]), float(x[1]), float(x[2])])
        tracking_errors.append(tracking_error)
        risks.append(risk)
        control_norms.append(control_norm)
        control_deltas.append(control_delta)
        row = {
            "step": step,
            "ref_x": float(ref[0]),
            "ref_y": float(ref[1]),
            "ref_z": float(ref[2]),
            "pred_x": float(x[0]),
            "pred_y": float(x[1]),
            "pred_z": float(x[2]),
            "corridor_safety_override": bool(corridor_override),
            "manifold_safety_override": bool(manifold_override),
            "corridor_override": bool(corridor_override),
            "manifold_override": bool(manifold_override),
            "safety_stop_override": bool(safety_stop),
            "tracking_error": tracking_error,
            "control_norm": control_norm,
            "control_delta": control_delta,
            "risk": risk,
            "risk_value": risk,
            "risk_query_valid": int(bool(risk_info.get("risk_query_valid"))),
            "risk_query_source": risk_info.get("risk_query_source", ""),
            "risk_query_failure_reason": risk_info.get(
                "risk_query_failure_reason", ""),
            "safety_violation": safety_violation,
            "risk_threshold": risk_threshold,
            "risk_exceeded": int(risk_exceeded),
            "constraint_violation": constraint_violation,
            "task_cost": float(task_optimization_cost),
            "task_optimization_cost": float(task_optimization_cost),
            "progress": float(progress),
            "phase": str(phase_weights.get("phase", phase)),
            "risk_weight": float(phase_weights.get("risk_weight", 0.0)),
            "tracking_weight": float(phase_weights.get("tracking_weight", 0.0)),
            "smoothness_weight": float(phase_weights.get("smoothness_weight", 0.0)),
            "task_weight": float(phase_weights.get("task_weight_effective", 0.0)),
            "step_cost": step_cost,
        }
        row.update(extra)
        rows.append(row)
        prev_u = u
    return _finish_mpc_result(costs, rows, controls, states, tracking_errors,
                              risks, control_norms, control_deltas,
                              violation_count, risk_threshold, weights, context)


def _finish_mpc_result(costs, rows, controls, states, tracking_errors, risks,
                       control_norms, control_deltas, violation_count,
                       risk_threshold, weights, context):
    if rows and any("step_cost" in row for row in rows):
        total = float(sum(float(row.get("step_cost", 0.0) or 0.0)
                          for row in rows))
    else:
        total = _weighted_total(costs, weights)
    risk_exceed_count = sum(1 for r in risks if r > risk_threshold)
    sequence_summary = _annotate_critical_sequence(rows, context)
    summary = _constraint_summary(rows)
    summary.update(sequence_summary)
    summary.update(_critical_sequence_summary(rows, context))
    status, failure_reason, failed_type = _feasibility_from_constraints(
        summary, risk_exceed_count)
    valid_count = sum(
        1 for row in rows if bool(int(row.get("risk_query_valid", 0) or 0)))
    invalid_count = max(0, len(rows) - valid_count)
    sources = [
        str(row.get("risk_query_source", ""))
        for row in rows if str(row.get("risk_query_source", "")).strip()
    ]
    source = sources[0] if sources else "unavailable"
    called = len(rows) > 0
    warning = ""
    if not called or valid_count <= 0:
        sanity = "failed_no_risk_query"
        warning = "risk_query_not_valid"
    elif source == "fallback_zero":
        sanity = "warning_fallback_zero"
        warning = "social_field_unavailable_fallback_zero"
    elif max(risks) <= 1e-12:
        sanity = "pass_all_low_risk"
    else:
        sanity = "pass"
    rollout_rows = []
    for row in rows:
        rollout = dict(row)
        step = int(row.get("step", 0))
        rollout.update({
            "robot": "",
            "corridor_id": "",
            "solve_index": 0,
            "horizon_step": step,
            "global_step": step,
            "pred_theta": row.get("pred_z", ""),
            "ref_theta": row.get("ref_z", ""),
            "feasibility_status": status,
        })
        rollout_rows.append(rollout)
        row["risk_sanity_status"] = sanity
        row["horizon"] = len(rows)
    risk_cost = float(costs["risk_cost"])
    task_cost = float(costs.get("task_cost", 0.0))
    max_risk = float(max(risks) if risks else 0.0)
    mean_risk = float(np.mean(risks) if risks else 0.0)
    if bool(context.get("strict_risk_query", False)) and invalid_count > 0:
        status = "risk_query_invalid"
        failure_reason = "risk_query_invalid"
        failed_type = "risk_query"
    result = {
        "mpc_feasibility_status": status,
        "failure_reason": failure_reason,
        "mpc_failure_reason": failure_reason,
        "failed_constraint_type": failed_type,
        "replan_required": bool(status in (
            "topology_infeasible", "manifold_infeasible",
            "corridor_infeasible")),
        "mpc_feedback": {
            "replan_required": bool(status in (
                "topology_infeasible", "manifold_infeasible",
                "corridor_infeasible")),
            "failure_type": failed_type,
            "failed_constraint": failed_type,
            "selected_corridor_id": str(context.get(
                "corridor_info", {}).get("corridor_id", "")),
            "failure_reason": failure_reason,
            "failed_constraint_type": failed_type,
        },
        "executable_trajectory": states,
        "control_sequence": controls,
        "predicted_states": states,
        "cost_breakdown_rows": rows,
        "rollout_rows": rollout_rows,
        "rollout_mode": "single_window_diagnostic",
        "rollout_solve_count": 1 if rows else 0,
        "rollout_log_file": "mpc_rollout_log.csv",
        "tracking_cost": float(costs["tracking_cost"]),
        "control_cost": float(costs["control_cost"]),
        "smoothness_cost": float(costs["smoothness_cost"]),
        "risk_cost": risk_cost,
        "mpc_risk_cost": risk_cost,
        "topology_cost": float(costs.get("topology_cost", 0.0)),
        "corridor_cost": float(costs.get("corridor_cost", 0.0)),
        "manifold_cost": float(costs.get("manifold_cost", 0.0)),
        "safety_violation_cost": float(costs["safety_violation_cost"]),
        "constraint_violation_cost": float(costs["constraint_violation_cost"]),
        "handover_cost": float(costs.get("handover_cost", 0.0)),
        "task_cost": task_cost,
        "mpc_task_cost": task_cost,
        "total_cost": float(total),
        "max_tracking_error": float(max(tracking_errors) if tracking_errors else 0.0),
        "mean_tracking_error": float(np.mean(tracking_errors) if tracking_errors else 0.0),
        "max_risk": max_risk,
        "mpc_max_risk": max_risk,
        "mean_risk": mean_risk,
        "mpc_mean_risk": mean_risk,
        "risk_exceed_count": int(risk_exceed_count),
        "mpc_risk_exceed_count": int(risk_exceed_count),
        "risk_exceed_ratio": float(
            risk_exceed_count / float(len(risks))) if risks else 0.0,
        "mpc_risk_exceed_ratio": float(
            risk_exceed_count / float(len(risks))) if risks else 0.0,
        "risk_query_called": bool(called),
        "risk_query_valid_count": int(valid_count),
        "risk_query_invalid_count": int(invalid_count),
        "risk_query_source": source,
        "risk_sanity_status": sanity,
        "risk_sanity_warning": warning,
        "max_control": float(max(control_norms) if control_norms else 0.0),
        "max_control_delta": float(max(control_deltas) if control_deltas else 0.0),
    }
    result.update(summary)
    return result


def _task_weight_diagnostics_payload(diag, rows, rollout_rows):
    source_rows = list(rollout_rows or rows or [])
    phase_rows = []
    phase_counts = {}
    seen = set()
    for idx, row in enumerate(source_rows):
        phase = _phase_name(row.get("phase", ""), default="approach")
        risk_weight = float(row.get("risk_weight", 0.0) or 0.0)
        tracking_weight = float(row.get("tracking_weight", 0.0) or 0.0)
        smoothness_weight = float(row.get("smoothness_weight", 0.0) or 0.0)
        task_weight = float(row.get("task_weight", 0.0) or 0.0)
        phase_counts[phase] = int(phase_counts.get(phase, 0) + 1)
        key = (phase, risk_weight, tracking_weight, smoothness_weight,
               task_weight)
        if key in seen:
            continue
        seen.add(key)
        phase_rows.append({
            "phase": phase,
            "risk_weight": risk_weight,
            "tracking_weight": tracking_weight,
            "smoothness_weight": smoothness_weight,
            "task_weight": task_weight,
            "first_step_index": int(row.get(
                "ref_index", row.get("global_step", row.get("step", idx))) or 0),
        })
    if not phase_rows:
        weights = dict(diag.get("weights", {}) or {})
        phase = "navigation" if str(diag.get("robot_type", "")).lower() == "wheelchair" else "approach"
        phase_weights = _phase_weight_entry(weights, phase)
        phase_rows.append({
            "phase": phase,
            "risk_weight": float(phase_weights.get("risk_weight", 0.0)),
            "tracking_weight": float(phase_weights.get("tracking_weight", 0.0)),
            "smoothness_weight": float(phase_weights.get("smoothness_weight", 0.0)),
            "task_weight": float(phase_weights.get("task_weight_effective", 0.0)),
            "first_step_index": 0,
        })
    return {
        "robot_type": str(diag.get("robot_type", "")),
        "selected_corridor_id": str(diag.get("selected_corridor_id", "")),
        "task_mode": str(diag.get("task_mode", "")),
        "task_weight_used": bool(diag.get("task_weight_used", False)),
        "phase_weight_used": bool(diag.get("phase_weight_used", True)),
        "phase_counts": phase_counts,
        "phase_weights": phase_rows,
    }


def _phase_constraint_diagnostics_payload(diag, rows, rollout_rows):
    source_rows = list(rows or rollout_rows or [])
    phases = {}
    records = []
    for row in source_rows:
        phase = _phase_name(row.get("phase", ""), default="approach")
        progress = float(row.get("progress", 0.0) or 0.0)
        clearance_threshold = float(row.get(
            "clearance_threshold",
            row.get("minimum_clearance", 0.0)) or 0.0)
        actual_clearance = float(row.get(
            "actual_clearance",
            row.get("manifold_clearance", 0.0)) or 0.0)
        constraint_status = str(row.get(
            "manifold_constraint_status",
            row.get("constraint_status", "")))
        hard_clearance_violation = bool(
            bool(row.get("major_violation", False)) or
            constraint_status == "infeasible")
        numeric_violation = bool(
            float(row.get("risk_constraint_violation", 0.0) or 0.0) > 1e-9 or
            float(row.get("clearance_constraint_violation", 0.0) or 0.0) > 1e-9 or
            float(row.get("manifold_violation", 0.0) or 0.0) > 1e-9)
        geometric_violation = bool(
            clearance_threshold > 0.0 and
            actual_clearance + 1e-9 < clearance_threshold)
        violation_count = int(
            numeric_violation or geometric_violation or
            constraint_status in ("soft_violation", "infeasible"))
        records.append({
            "trajectory_source": str(row.get(
                "trajectory_source", "executed" if rows else "predicted")),
            "phase": phase,
            "progress": float(progress),
            "return_progress": float(progress) if phase == "return" else "",
            "threshold": float(clearance_threshold),
            "clearance_threshold": float(clearance_threshold),
            "return_clearance_threshold": (
                float(clearance_threshold) if phase == "return" else ""),
            "actual_clearance": float(actual_clearance),
            "violation_count": int(violation_count),
            "constraint_status": constraint_status,
        })
        entry = phases.setdefault(phase, {
            "phase": phase,
            "count": 0,
            "risk_weight": float(row.get("risk_weight", 0.0) or 0.0),
            "tracking_weight": float(row.get("tracking_weight", 0.0) or 0.0),
            "smoothness_weight": float(row.get("smoothness_weight", 0.0) or 0.0),
            "task_weight": float(row.get("task_weight", 0.0) or 0.0),
            "max_manifold_violation": 0.0,
            "max_corridor_violation": 0.0,
            "min_actual_clearance": float("inf"),
            "max_clearance_threshold": 0.0,
            "risk_violation_count": 0,
            "clearance_violation_count": 0,
        })
        entry["count"] += 1
        entry["min_actual_clearance"] = min(
            float(entry["min_actual_clearance"]),
            float(actual_clearance))
        entry["max_clearance_threshold"] = max(
            float(entry["max_clearance_threshold"]),
            float(clearance_threshold))
        entry["max_manifold_violation"] = max(
            float(entry["max_manifold_violation"]),
            float(row.get("manifold_violation", 0.0) or 0.0))
        entry["max_corridor_violation"] = max(
            float(entry["max_corridor_violation"]),
            float(row.get("corridor_violation", 0.0) or 0.0))
        if float(row.get("risk_constraint_violation", 0.0) or 0.0) > 1e-9:
            entry["risk_violation_count"] += 1
        if hard_clearance_violation:
            entry["clearance_violation_count"] += 1
        elif float(row.get("clearance_constraint_violation", 0.0) or 0.0) > 1e-9:
            entry["clearance_violation_count"] += 1
    manifold_info = dict(diag.get("manifold_info", {}) or {})
    manifold_constraint = dict(
        manifold_info.get("manifold_constraint", {}) or manifold_info)
    phase_values = []
    for item in phases.values():
        if not np.isfinite(float(item.get("min_actual_clearance", 0.0))):
            item["min_actual_clearance"] = 0.0
        phase_values.append(item)
    executed_total = int(sum(int(r.get("violation_count", 0) or 0)
                             for r in records
                             if r.get("trajectory_source") == "executed"))
    validation_total = int(diag.get(
        "executed_manifold_violation_count",
        diag.get("manifold_violation_count", executed_total)) or 0)
    return {
        "robot_type": str(diag.get("robot_type", "")),
        "selected_corridor_id": str(diag.get("selected_corridor_id", "")),
        "current_phase": str(
            source_rows[-1].get("phase", "") if source_rows else
            manifold_constraint.get("phase", "")),
        "phase_count": int(len(phases)),
        "phases": phase_values,
        "records": records,
        "executed_violation_total": int(executed_total),
        "validation_manifold_violation_count": int(validation_total),
        "consistency_check": bool(executed_total == validation_total),
        "constraint_status": {
            "mpc_feasibility_status": str(diag.get("mpc_feasibility_status", "")),
            "manifold_constraint_status": str(
                diag.get("manifold_constraint_status", "")),
            "corridor_constraint_status": str(
                diag.get("corridor_constraint_status", "")),
            "tube_constraint_status": str(diag.get("tube_constraint_status", "")),
            "critical_point_status": str(diag.get("critical_point_status", "")),
        },
        "manifold_phase_parameters": {
            "phase": str(manifold_constraint.get("phase", "")),
            "robot_type": str(manifold_constraint.get("robot_type", "")),
            "phase_aware": bool(manifold_constraint.get("phase_aware", False)),
            "phase_parameters": dict(
                manifold_constraint.get("phase_parameters", {}) or {}),
            "phase_manifold_weights": dict(
                manifold_constraint.get("phase_manifold_weights", {}) or {}),
            "minimum_clearance": manifold_constraint.get("minimum_clearance", ""),
            "risk_threshold": manifold_constraint.get("risk_threshold", ""),
            "effective_minimum_clearance": manifold_constraint.get(
                "effective_minimum_clearance", ""),
            "effective_risk_threshold": manifold_constraint.get(
                "effective_risk_threshold", ""),
        },
    }


def _task_state_diagnostics_payload(diag, rows, rollout_rows):
    source_rows = list(rollout_rows or rows or [])
    records = []
    transitions = []
    prev = ""
    robot = str(diag.get("robot_type", ""))
    mode = str(diag.get("task_mode", ""))
    progress_values = []
    for row in source_rows:
        try:
            progress_values.append(float(row.get("progress", 0.0) or 0.0))
        except (TypeError, ValueError):
            progress_values.append(0.0)
    progress_missing = bool(source_rows) and max(progress_values or [0.0]) <= 1e-9
    denom = float(max(1, len(source_rows) - 1))
    for idx, row in enumerate(source_rows):
        progress = (
            float(idx) / denom if progress_missing
            else float(progress_values[idx] if idx < len(progress_values) else 0.0))
        state = str(infer_task_state(
            robot, mode, phase=row.get("phase", ""),
            progress=progress).get("task_state", ""))
        transition = "{}->{}".format(prev or "start", state)
        if state != prev:
            transitions.append(transition)
        prev = state
        records.append({
            "task_mode": mode,
            "task_state": state,
            "phase": str(row.get("phase", "")),
            "current_phase": str(row.get("phase", "")),
            "progress": float(progress),
            "state_transition": transition,
            "timestamp": float(row.get("timestamp", 0.0) or time.time()),
        })
    if not records:
        info = infer_task_state(robot, mode)
        records.append(dict(info))
    return {
        "robot_type": robot,
        "task_mode": mode,
        "current_phase": str(records[-1].get("current_phase", "")),
        "task_state": str(records[-1].get("task_state", "")),
        "state_transition": str(records[-1].get("state_transition", "")),
        "transitions": transitions,
        "records": records,
    }


def _mpc_task_optimization_diagnostics_payload(diag, rows, rollout_rows):
    source_rows = list(rollout_rows or rows or [])
    records = []
    phase_summary = {}
    for row in source_rows:
        phase = _phase_name(row.get("phase", ""), default="approach")
        task_cost = float(row.get(
            "task_optimization_cost", row.get("task_cost", 0.0)) or 0.0)
        tracking = float(row.get("tracking_error", 0.0) or 0.0) ** 2
        status = str(row.get(
            "manifold_constraint_status",
            row.get("feasibility_status",
                    row.get("constraint_status", ""))))
        records.append({
            "phase": phase,
            "task_state": str(row.get("task_state", "")),
            "task_cost": float(task_cost),
            "tracking_cost": float(tracking),
            "constraint_status": status,
        })
        entry = phase_summary.setdefault(phase, {
            "phase": phase,
            "count": 0,
            "task_cost": 0.0,
            "tracking_cost": 0.0,
            "constraint_status": status,
        })
        entry["count"] += 1
        entry["task_cost"] += float(task_cost)
        entry["tracking_cost"] += float(tracking)
        if status and status != "feasible":
            entry["constraint_status"] = status
    return {
        "robot_type": str(diag.get("robot_type", "")),
        "selected_corridor_id": str(diag.get("selected_corridor_id", "")),
        "task_mode": str(diag.get("task_mode", "")),
        "records": records,
        "phase_summary": list(phase_summary.values()),
        "total_task_cost": float(sum(r["task_cost"] for r in records)),
        "total_tracking_cost": float(sum(r["tracking_cost"] for r in records)),
    }


def _task_error_payload(robot, state, row):
    task_cost = float(row.get(
        "task_optimization_cost", row.get("task_cost", 0.0)) or 0.0)
    tracking_error = float(row.get("tracking_error", 0.0) or 0.0)
    control_delta = float(row.get("control_delta", 0.0) or 0.0)
    safety_violation = float(row.get("safety_violation", 0.0) or 0.0)
    handover_cost = float(row.get("handover_cost", 0.0) or 0.0)
    if str(robot).lower() == "arm":
        ee_position_error = float(np.sqrt(max(0.0, handover_cost)))
        return {
            "task_error": float(np.sqrt(max(0.0, task_cost))),
            "ee_position_error": ee_position_error,
            "ee_orientation_error": float(row.get("ee_orientation_error", 0.0) or 0.0),
            "interaction_distance_error": ee_position_error,
            "hold_stability_error": (
                control_delta if str(state) == "hold" else 0.0),
            "safety_violation_error": safety_violation,
        }
    return {
        "task_error": float(np.sqrt(max(0.0, task_cost))),
        "goal_distance_error": tracking_error,
        "goal_direction_error": float(row.get("heading_error", 0.0) or 0.0),
        "passage_alignment_error": float(
            row.get("corridor_violation", row.get("manifold_violation", 0.0)) or 0.0),
        "stability_error": control_delta,
    }


def _mpc_task_objective_diagnostics_payload(diag, rows, rollout_rows):
    source_rows = list(rollout_rows or rows or [])
    robot = str(diag.get("robot_type", ""))
    mode = str(diag.get("task_mode", ""))
    records = []
    state_summary = {}
    progress_values = []
    for row in source_rows:
        try:
            progress_values.append(float(row.get("progress", 0.0) or 0.0))
        except (TypeError, ValueError):
            progress_values.append(0.0)
    progress_missing = bool(source_rows) and max(progress_values or [0.0]) <= 1e-9
    denom = float(max(1, len(source_rows) - 1))
    for idx, row in enumerate(source_rows):
        phase = _phase_name(row.get("phase", ""), default="approach")
        progress = (
            float(idx) / denom if progress_missing
            else float(progress_values[idx] if idx < len(progress_values) else 0.0))
        inferred_state = infer_task_state(
            robot, mode, phase=phase, progress=progress).get("task_state", "")
        state = str(inferred_state if progress_missing else (
            row.get("task_state", "") or inferred_state))
        tracking_base = float(row.get("tracking_error", 0.0) or 0.0) ** 2
        risk_base = float(row.get("risk_value", row.get("risk", 0.0)) or 0.0)
        task_base = float(row.get(
            "task_optimization_cost", row.get("task_cost", 0.0)) or 0.0)
        if str(robot).lower() == "wheelchair":
            heading_base = min(
                1.0, abs(float(row.get("heading_error", 0.0) or 0.0)) /
                max(float(np.pi), 1e-6))
            stability_base = min(
                1.0, float(row.get("control_delta", 0.0) or 0.0))
            task_base = (
                min(1.0, float(row.get("tracking_error", 0.0) or 0.0)) ** 2 +
                heading_base ** 2 +
                0.25 * stability_base ** 2)
        J_tracking = float(row.get("tracking_weight", 0.0) or 0.0) * tracking_base
        J_risk = float(row.get("risk_weight", 0.0) or 0.0) * risk_base
        J_task = float(row.get("task_weight", 0.0) or 0.0) * task_base
        J_total = float(row.get(
            "step_cost", row.get("step_total_cost",
                                 J_tracking + J_risk + J_task)) or 0.0)
        task_error = _task_error_payload(robot, state, row)
        step = int(row.get(
            "global_step", row.get("horizon_step", row.get("step", idx))) or 0)
        record = {
            "step": step,
            "phase": phase,
            "task_state": state,
            "J_tracking": J_tracking,
            "J_risk": J_risk,
            "J_task": J_task,
            "J_total": J_total,
            "task_error": float(task_error.get("task_error", 0.0)),
            "task_error_breakdown": task_error,
        }
        records.append(record)
        entry = state_summary.setdefault(state, {
            "task_state": state,
            "count": 0,
            "J_tracking": 0.0,
            "J_risk": 0.0,
            "J_task": 0.0,
            "J_total": 0.0,
        })
        entry["count"] += 1
        entry["J_tracking"] += J_tracking
        entry["J_risk"] += J_risk
        entry["J_task"] += J_task
        entry["J_total"] += J_total
    return {
        "robot_type": robot,
        "selected_corridor_id": str(diag.get("selected_corridor_id", "")),
        "task_mode": mode,
        "records": records,
        "state_summary": list(state_summary.values()),
        "J_tracking": float(sum(r["J_tracking"] for r in records)),
        "J_risk": float(sum(r["J_risk"] for r in records)),
        "J_task": float(sum(r["J_task"] for r in records)),
        "J_total": float(sum(r["J_total"] for r in records)),
    }


def _stsm_alias_path(path):
    if not path:
        return ""
    norm = os.path.abspath(path).replace("\\", "/")
    marker = "/stsm/"
    idx = norm.lower().rfind(marker)
    if idx < 0:
        return norm
    return norm[:idx] + "/stsm/" + norm[idx + len(marker):]


def _write_stsm_alias_json(path, payload):
    alias = _stsm_alias_path(path)
    if not alias:
        return
    directory = os.path.dirname(alias)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    with open(alias, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_mpc_outputs(result, diagnostics_path, breakdown_path, rollout_path=None,
                      executed_path=None):
    diag = dict(result or {})
    rows = list(diag.pop("cost_breakdown_rows", []) or [])
    rollout_rows = list(diag.pop("rollout_rows", []) or [])
    executed_rows = list(diag.pop("executed_trajectory_rows", []) or [])
    diagnostics_path = _stsm_alias_path(diagnostics_path)
    breakdown_path = _stsm_alias_path(breakdown_path)
    if rollout_path:
        rollout_path = _stsm_alias_path(rollout_path)
    if executed_path:
        executed_path = _stsm_alias_path(executed_path)
    rollout_path = rollout_path or os.path.join(
        os.path.dirname(diagnostics_path), "mpc_rollout_log.csv")
    executed_path = executed_path or os.path.join(
        os.path.dirname(diagnostics_path), "mpc_executed_trajectory.csv")
    diag["rollout_log_file"] = os.path.basename(rollout_path)
    diag["executed_trajectory_file"] = os.path.basename(executed_path)
    diag["cost_breakdown_file"] = os.path.basename(breakdown_path)
    diag["topology_constraint_file"] = "topology_constraint.json"
    diag["mpc_feedback_file"] = "mpc_feedback.json"
    task_weight_path = os.path.join(
        os.path.dirname(diagnostics_path), "mpc_task_weight_diagnostics.json")
    diag["mpc_task_weight_diagnostics_file"] = os.path.basename(task_weight_path)
    phase_diag_path = os.path.join(
        os.path.dirname(diagnostics_path), "mpc_phase_diagnostics.json")
    diag["mpc_phase_diagnostics_file"] = os.path.basename(phase_diag_path)
    manifold_phase_diag_path = os.path.join(
        os.path.dirname(diagnostics_path), "manifold_phase_diagnostics.json")
    diag["manifold_phase_diagnostics_file"] = os.path.basename(
        manifold_phase_diag_path)
    task_state_diag_path = os.path.join(
        os.path.dirname(diagnostics_path), "task_state_diagnostics.json")
    diag["task_state_diagnostics_file"] = os.path.basename(
        task_state_diag_path)
    task_opt_diag_path = os.path.join(
        os.path.dirname(diagnostics_path),
        "mpc_task_optimization_diagnostics.json")
    diag["mpc_task_optimization_diagnostics_file"] = os.path.basename(
        task_opt_diag_path)
    task_obj_diag_path = os.path.join(
        os.path.dirname(diagnostics_path),
        "mpc_task_objective_diagnostics.json")
    diag["mpc_task_objective_diagnostics_file"] = os.path.basename(
        task_obj_diag_path)
    for key in ("executable_trajectory", "control_sequence", "predicted_states"):
        if key in diag and len(diag.get(key) or []) > 50:
            diag[key] = diag[key][:50]
    robot = str(diag.get("robot_type", ""))
    cid = str(diag.get("selected_corridor_id", ""))
    horizon = int(diag.get("horizon", len(rows) or 0))
    dt = float(diag.get("dt", 0.0))
    for row in rows:
        row["robot"] = row.get("robot") or robot
        row["corridor_id"] = row.get("corridor_id") or cid
        row["horizon"] = horizon
        row["dt"] = dt
        row.setdefault("risk_value", row.get("risk", 0.0))
        row.setdefault("risk_query_valid", 0)
        row.setdefault("risk_query_source", "")
        row.setdefault("safety_violation", 0.0)
        row.setdefault("risk_threshold", "")
        row.setdefault("risk_sanity_status", diag.get("risk_sanity_status", ""))
        row.setdefault("topology_class", diag.get("topology_class", ""))
        row.setdefault("distance_to_corridor", "")
        row.setdefault("corridor_violation", "")
        row.setdefault("corridor_constraint_status", "")
        row.setdefault("distance_to_manifold", "")
        row.setdefault("boundary_distance", "")
        row.setdefault("manifold_clearance", "")
        row.setdefault("minimum_clearance", diag.get("minimum_clearance", ""))
        row.setdefault("progress", 0.0)
        row.setdefault("clearance_threshold", row.get("minimum_clearance", ""))
        row.setdefault("actual_clearance", row.get("manifold_clearance", ""))
        row.setdefault("manifold_violation", "")
        row.setdefault("manifold_constraint_status", "")
        row.setdefault("manifold_constraint_mode", diag.get(
            "manifold_constraint_mode", "soft"))
        row.setdefault("soft_constraint_used", diag.get(
            "soft_constraint_used", False))
        row.setdefault("minor_violation", False)
        row.setdefault("major_violation", False)
        row.setdefault("risk_constraint_violation", "")
        row.setdefault("clearance_constraint_violation", "")
        row.setdefault("nearest_critical_point", "")
        row.setdefault("critical_point_target", "")
        row.setdefault("critical_point_distance", "")
        row.setdefault("critical_sequence_state", "")
        row.setdefault("topology_sequence_valid", "")
        row.setdefault("critical_point_status", diag.get("critical_point_status", ""))
        row.setdefault("current_topology_stage", diag.get("current_topology_stage", ""))
        row.setdefault("corridor_override", False)
        row.setdefault("manifold_override", False)
        row.setdefault("corridor_safety_override", False)
        row.setdefault("manifold_safety_override", False)
        row.setdefault("critical_point_constraint_status", "")
        row.setdefault("topology_class_constraint_status", "")
        row.setdefault("topology_consistency_cost", 0.0)
        row.setdefault("topology_cost", row.get("topology_consistency_cost", 0.0))
        row.setdefault("corridor_cost", 0.0)
        row.setdefault("manifold_cost", 0.0)
        row.setdefault("phase", "approach")
        state_info = infer_task_state(
            robot, diag.get("task_mode", ""), phase=row.get("phase", ""),
            progress=row.get("progress", 0.0))
        row.setdefault("task_state", state_info.get("task_state", ""))
        row.setdefault("state_transition", state_info.get("state_transition", ""))
        row.setdefault("timestamp", time.time())
        row.setdefault("task_cost", row.get("task_optimization_cost", 0.0))
        row.setdefault("task_optimization_cost", row.get("task_cost", 0.0))
        row.setdefault("risk_weight", 0.0)
        row.setdefault("tracking_weight", 0.0)
        row.setdefault("smoothness_weight", 0.0)
        row.setdefault("task_weight", 0.0)
    for row in rollout_rows:
        row["robot"] = row.get("robot") or robot
        row["corridor_id"] = row.get("corridor_id") or cid
        row.setdefault("risk_value", row.get("risk", 0.0))
        row.setdefault("risk_query_valid", 0)
        row.setdefault("risk_query_source", "")
        row.setdefault("risk_exceeded", 0)
        row.setdefault("safety_violation", 0.0)
        row.setdefault("constraint_violation", 0.0)
        row.setdefault("topology_class", diag.get("topology_class", ""))
        row.setdefault("distance_to_corridor", "")
        row.setdefault("corridor_violation", "")
        row.setdefault("corridor_constraint_status", "")
        row.setdefault("distance_to_manifold", "")
        row.setdefault("boundary_distance", "")
        row.setdefault("manifold_clearance", "")
        row.setdefault("minimum_clearance", diag.get("minimum_clearance", ""))
        row.setdefault("progress", 0.0)
        row.setdefault("clearance_threshold", row.get("minimum_clearance", ""))
        row.setdefault("actual_clearance", row.get("manifold_clearance", ""))
        row.setdefault("manifold_violation", "")
        row.setdefault("manifold_constraint_status", "")
        row.setdefault("manifold_constraint_mode", diag.get(
            "manifold_constraint_mode", "soft"))
        row.setdefault("soft_constraint_used", diag.get(
            "soft_constraint_used", False))
        row.setdefault("minor_violation", False)
        row.setdefault("major_violation", False)
        row.setdefault("risk_constraint_violation", "")
        row.setdefault("clearance_constraint_violation", "")
        row.setdefault("nearest_critical_point", "")
        row.setdefault("phase", "approach")
        state_info = infer_task_state(
            robot, diag.get("task_mode", ""), phase=row.get("phase", ""),
            progress=row.get("progress", 0.0))
        row.setdefault("task_state", state_info.get("task_state", ""))
        row.setdefault("state_transition", state_info.get("state_transition", ""))
        row.setdefault("timestamp", time.time())
        row.setdefault("task_cost", row.get("task_optimization_cost", 0.0))
        row.setdefault("task_optimization_cost", row.get("task_cost", 0.0))
        row.setdefault("risk_weight", 0.0)
        row.setdefault("tracking_weight", 0.0)
        row.setdefault("smoothness_weight", 0.0)
        row.setdefault("task_weight", 0.0)
        row.setdefault("critical_point_target", "")
        row.setdefault("critical_point_distance", "")
        row.setdefault("critical_sequence_state", "")
        row.setdefault("topology_sequence_valid", "")
        row.setdefault("critical_point_status", diag.get("critical_point_status", ""))
        row.setdefault("current_topology_stage", diag.get("current_topology_stage", ""))
        row.setdefault("corridor_override", False)
        row.setdefault("manifold_override", False)
        row.setdefault("corridor_safety_override", False)
        row.setdefault("manifold_safety_override", False)
        row.setdefault("critical_point_constraint_status", "")
        row.setdefault("topology_class_constraint_status", "")
        row.setdefault("topology_consistency_cost", 0.0)
        row.setdefault("corridor_cost", 0.0)
        row.setdefault("manifold_cost", 0.0)
    for row in executed_rows:
        row["robot"] = row.get("robot") or robot
        row["corridor_id"] = row.get("corridor_id") or cid
    for path in (diagnostics_path, breakdown_path, rollout_path, executed_path,
                 task_weight_path, phase_diag_path, manifold_phase_diag_path,
                 task_state_diag_path, task_opt_diag_path, task_obj_diag_path):
        directory = os.path.dirname(path)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
    task_weight_payload = _task_weight_diagnostics_payload(
        diag, rows, rollout_rows)
    phase_payload = _phase_constraint_diagnostics_payload(
        diag, executed_rows, rollout_rows)
    task_state_payload = _task_state_diagnostics_payload(
        diag, rows, rollout_rows)
    task_opt_payload = _mpc_task_optimization_diagnostics_payload(
        diag, rows, rollout_rows)
    task_obj_payload = _mpc_task_objective_diagnostics_payload(
        diag, rows, rollout_rows)
    diag["current_phase"] = str(phase_payload.get("current_phase", ""))
    diag["phase_counts"] = {
        str(item.get("phase", "")): int(item.get("count", 0))
        for item in phase_payload.get("phases", [])
    }
    diag["constraint_status_summary"] = dict(
        phase_payload.get("constraint_status", {}) or {})
    diag["manifold_phase_parameters"] = dict(
        phase_payload.get("manifold_phase_parameters", {}) or {})
    with open(task_weight_path, "w") as f:
        json.dump(task_weight_payload, f, indent=2, sort_keys=True)
    _write_stsm_alias_json(task_weight_path, task_weight_payload)
    with open(phase_diag_path, "w") as f:
        json.dump(phase_payload, f, indent=2, sort_keys=True)
    _write_stsm_alias_json(phase_diag_path, phase_payload)
    with open(manifold_phase_diag_path, "w") as f:
        json.dump(phase_payload, f, indent=2, sort_keys=True)
    _write_stsm_alias_json(manifold_phase_diag_path, phase_payload)
    with open(task_state_diag_path, "w") as f:
        json.dump(task_state_payload, f, indent=2, sort_keys=True)
    _write_stsm_alias_json(task_state_diag_path, task_state_payload)
    with open(task_opt_diag_path, "w") as f:
        json.dump(task_opt_payload, f, indent=2, sort_keys=True)
    _write_stsm_alias_json(task_opt_diag_path, task_opt_payload)
    with open(task_obj_diag_path, "w") as f:
        json.dump(task_obj_payload, f, indent=2, sort_keys=True)
    _write_stsm_alias_json(task_obj_diag_path, task_obj_payload)
    with open(diagnostics_path, "w") as f:
        json.dump(diag, f, indent=2, sort_keys=True)
    feedback_path = os.path.join(os.path.dirname(diagnostics_path),
                                 "mpc_feedback.json")
    feedback = dict(diag.get("mpc_feedback", {}) or {})
    feedback.setdefault("replan_required", bool(diag.get("replan_required", False)))
    if bool(diag.get("replan_required", False)):
        feedback["failure_type"] = str(
            diag.get("failed_constraint_type") or feedback.get("failure_type", ""))
    else:
        feedback.setdefault("failure_type", "")
    feedback.setdefault("failed_constraint", diag.get("failed_constraint_type", ""))
    feedback.setdefault("selected_corridor_id", cid)
    feedback.setdefault("failure_reason", diag.get("failure_reason", ""))
    with open(feedback_path, "w") as f:
        json.dump(feedback, f, indent=2, sort_keys=True)
    fields = [
        "robot", "corridor_id", "rollout_mode", "solve_index",
        "horizon_step", "global_step", "tracking_cost", "control_cost",
        "smoothness_cost", "risk_cost", "topology_cost", "corridor_cost",
        "manifold_cost", "safety_violation_cost",
        "constraint_violation_cost", "handover_cost", "step_total_cost",
        "cumulative_total_cost", "phase", "task_state", "state_transition",
        "task_cost", "task_optimization_cost", "risk_weight",
        "tracking_weight", "smoothness_weight", "task_weight",
        "feasibility_status",
        "step", "ref_x", "ref_y", "ref_z", "pred_x", "pred_y", "pred_z",
        "tracking_error", "control_norm", "control_delta", "risk",
        "risk_value", "risk_query_valid", "risk_query_source",
        "safety_violation", "risk_threshold", "risk_sanity_status",
        "horizon", "dt", "risk_exceeded", "constraint_violation",
        "topology_class", "distance_to_corridor", "corridor_violation",
        "corridor_constraint_status", "distance_to_manifold",
        "manifold_clearance", "minimum_clearance", "clearance_threshold",
        "actual_clearance", "progress", "boundary_distance",
        "manifold_violation", "manifold_constraint_status",
        "manifold_constraint_mode", "risk_constraint_violation",
        "soft_constraint_used", "minor_violation", "major_violation",
        "clearance_constraint_violation",
        "nearest_critical_point", "critical_point_target",
        "critical_point_distance", "critical_sequence_state",
        "topology_sequence_valid", "critical_point_status",
        "current_topology_stage", "corridor_override", "manifold_override",
        "critical_point_constraint_status",
        "topology_class_constraint_status", "topology_consistency_cost",
        "step_cost",
    ]
    with open(breakdown_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    rollout_fields = [
        "robot", "corridor_id", "rollout_mode", "solve_index",
        "horizon_step", "global_step", "ref_index",
        "ref_x", "ref_y", "ref_z", "ref_theta",
        "pred_x", "pred_y", "pred_z", "pred_theta",
        "executed_x", "executed_y", "executed_z", "executed_theta",
        "v_cmd", "omega_cmd", "first_control_applied",
        "critical_point_target", "critical_point_distance",
        "critical_sequence_state", "topology_sequence_valid",
        "critical_point_status", "current_topology_stage",
        "corridor_override", "manifold_override",
        "corridor_safety_override", "manifold_safety_override",
        "phase", "task_state", "state_transition", "task_cost",
        "task_optimization_cost", "risk_weight", "tracking_weight",
        "smoothness_weight", "task_weight",
        "tracking_error", "control_norm", "control_delta", "risk_value",
        "risk_query_valid", "risk_query_source", "risk_exceeded",
        "safety_violation", "constraint_violation",
        "topology_class", "distance_to_corridor", "corridor_violation",
        "corridor_constraint_status", "distance_to_manifold",
        "manifold_clearance", "minimum_clearance", "clearance_threshold",
        "actual_clearance", "progress", "boundary_distance",
        "manifold_violation", "manifold_constraint_status",
        "manifold_constraint_mode", "risk_constraint_violation",
        "soft_constraint_used", "minor_violation", "major_violation",
        "clearance_constraint_violation",
        "nearest_critical_point",
        "critical_point_constraint_status",
        "topology_class_constraint_status",
        "topology_consistency_cost", "corridor_cost", "manifold_cost",
        "step_cost",
        "feasibility_status",
    ]
    with open(rollout_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=rollout_fields)
        writer.writeheader()
        for row in rollout_rows:
            writer.writerow({key: row.get(key, "") for key in rollout_fields})
    executed_fields = [
        "robot", "corridor_id", "global_step", "x", "y", "z", "theta",
        "joint_state", "control_norm", "tracking_error", "risk_value",
        "phase", "task_state", "state_transition", "risk_weight",
        "tracking_weight", "smoothness_weight", "task_weight",
        "feasibility_status", "trajectory_source", "clearance_source",
        "manifold_clearance", "minimum_clearance", "clearance_threshold",
        "actual_clearance", "clearance_constraint_violation",
        "risk_threshold", "risk_constraint_violation",
        "manifold_violation", "manifold_constraint_status",
        "corridor_violation", "corridor_constraint_status",
    ]
    with open(executed_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=executed_fields)
        writer.writeheader()
        for row in executed_rows:
            writer.writerow({key: row.get(key, "") for key in executed_fields})

_CVXPY = None
_CVXPY_CHECKED = False
_HAVE_CVXPY = False

def _get_cvxpy():
    global _CVXPY, _CVXPY_CHECKED, _HAVE_CVXPY
    if not _CVXPY_CHECKED:
        try:
            _CVXPY = importlib.import_module("cvxpy")
        except Exception:
            _CVXPY = None
        _CVXPY_CHECKED = True
        _HAVE_CVXPY = _CVXPY is not None
    return _CVXPY

def _cvx_matvec(cp, A, x):
    if hasattr(cp, "matmul"):
        return cp.matmul(A, x)
    return A * x

class ArmMPC:
    def __init__(self, n_joints=6, dq_max=0.8, v_cap=0.25, damping=0.05,
                 lam_nominal=0.3, adp_grad_clip=8.0, horizon=6,
                 beam_width=10, ddq_max=1.2, joint_lower=None,
                 joint_upper=None, phase_cost_weights=None,
                 min_terminal_progress_ratio=0.01,
                 task_progress_tolerance=1e-3):
        self.n = n_joints
        self.dq_max = dq_max
        self.v_cap = v_cap
        self.damping = damping
        self.lam_nominal = lam_nominal
        self.adp_grad_clip = float(adp_grad_clip)
        self.N = max(2, int(horizon))
        self.beam_width = max(2, int(beam_width))
        self.ddq_max = float(ddq_max)
        self.joint_lower = self._joint_bound(joint_lower, -np.inf)
        self.joint_upper = self._joint_bound(joint_upper, np.inf)
        self.phase_cost_weights = dict(phase_cost_weights or {})
        self.min_terminal_progress_ratio = max(
            0.0, float(min_terminal_progress_ratio))
        self.task_progress_tolerance = max(
            0.0, float(task_progress_tolerance))
        self.solve_count = 0
        self.solve_success_count = 0
        self.fallback_count = 0
        self.last_solver_status = "not_called"
        self.last_adp_grad_norm = 0.0
        self.last_adp_soft_cost = 0.0
        self.last_v_adp_alignment = 0.0
        self.last_dls_adp_used = 0
        self.last_qp_used = 0
        self.last_v_des_raw_norm = 0.0
        self.last_v_des_adp_norm = 0.0
        self.last_v_des_delta_norm = 0.0
        self.last_dq_nominal_norm = 0.0
        self.last_dq_adp_norm = 0.0
        self.last_dq_delta_norm = 0.0
        self.last_reject_forbidden_count = 0
        self.last_reject_interest_phi_count = 0
        self.first_predicted_forbidden_reason = ""
        self.last_handover_protection = {}
        self.last_predicted_joint_states = []
        self.last_predicted_controls = []
        self.last_predicted_ee_states = []
        self.last_objective_terms = {}
        self.last_constraint_violation = {}
        self.last_control_sequence_varies = False
        self.last_kinematics_source = "unreported"
        self.last_interest_kinematics_source = "unreported"
        self.last_prediction_model = "none"

    def _joint_bound(self, values, default):
        if values is None:
            return np.full(self.n, float(default), float)
        arr = np.asarray(values, float).reshape(-1)
        if arr.size != self.n:
            raise ValueError("joint bound must contain {} values".format(self.n))
        return arr

    def solve(self, J, v_des, dq_nom=None, v_cap=None, ee_pos=None, dt=0.1,
              critic=None, feature_builder=None, field=None, gate_info=None,
              interest_risk=None, target_pos=None, phase="handover",
              lambda_adp_arm=0.0, adp_grad_eps=0.01,
              adp_descent_gain=0.04, solver_mode="dls_adp",
              adp_blend_alpha=0.08, use_cvxpy=False,
               interest_constraints=None, handover_protect=False,
               handover_target=None, handover_tracking_weight=8.0,
               q=None, corridor=None, predictive=False,
               kinematics_source="real", phase_cost_weights=None):
        self.solve_count += 1
        J = np.asarray(J, float)
        v_des = np.asarray(v_des, float)
        cap = self.v_cap if v_cap is None else float(v_cap)
        dq_nom = np.zeros(self.n) if dq_nom is None else np.asarray(dq_nom, float)
        protect = bool(handover_protect and ee_pos is not None and
                       handover_target is not None)
        self.last_handover_protection = {
            "active": bool(protect),
            "phase": phase,
        }
        self.last_predicted_joint_states = []
        self.last_predicted_controls = []
        self.last_predicted_ee_states = []
        self.last_objective_terms = {}
        self.last_constraint_violation = {}
        self.last_control_sequence_varies = False
        self.last_kinematics_source = str(kinematics_source or "unreported")
        self.last_interest_kinematics_source = "unreported"
        self.last_prediction_model = "none"
        gradV, v_avoid = self._adp_avoidance(
            ee_pos, critic, feature_builder, field, gate_info, interest_risk,
            target_pos, phase, lambda_adp_arm, adp_grad_eps, adp_descent_gain)
        if protect:
            v_avoid = None
            lambda_adp_arm = 0.0

        if bool(predictive):
            return self._sampled_predictive_solve(
                J, v_des, dq_nom, cap, q, ee_pos, dt, field,
                interest_constraints, target_pos, phase, corridor, v_avoid,
                lambda_adp_arm, adp_blend_alpha, handover_target, protect,
                handover_tracking_weight, phase_cost_weights)

        mode = str(solver_mode or "dls_adp").lower()
        if mode in ("dls", "dls_adp") or not bool(use_cvxpy):
            v_cmd = self._compose_adp_velocity(
                v_des, v_avoid, cap, lambda_adp_arm=lambda_adp_arm,
                adp_blend_alpha=adp_blend_alpha, preserve_progress=True)
            dq_nominal = self._dls(J, v_des, dq_nom, cap)
            dq = self._dls(J, v_cmd, dq_nom, cap)
            self.solve_success_count += 1
            self.last_dls_adp_used = 1 if v_avoid is not None else 0
            self.last_qp_used = 0
            self.last_solver_status = (
                "dls_adp" if v_avoid is not None else "dls")
            self._update_control_delta_stats(v_des, v_cmd, dq_nominal, dq)
            dq = self._enforce_interest_constraints(
                J, dq, ee_pos, dt, field, interest_constraints)
            dq = self._apply_handover_target_guard(
                J, dq, dq_nominal, ee_pos, dt, handover_target, protect)
            self._update_adp_solution_stats(J, dq, gradV, v_avoid)
            return dq

        cp = _get_cvxpy()
        if cp is None:
            v_cmd = self._compose_adp_velocity(
                v_des, v_avoid, cap, lambda_adp_arm=lambda_adp_arm,
                adp_blend_alpha=adp_blend_alpha, preserve_progress=True)
            dq_nominal = self._dls(J, v_des, dq_nom, cap)
            dq = self._dls(J, v_cmd, dq_nom, cap)
            self.solve_success_count += 1
            self.last_dls_adp_used = 1 if v_avoid is not None else 0
            self.last_qp_used = 0
            self.last_solver_status = (
                "dls_adp_no_cvxpy" if v_avoid is not None else "dls_no_cvxpy")
            self._update_control_delta_stats(v_des, v_cmd, dq_nominal, dq)
            dq = self._enforce_interest_constraints(
                J, dq, ee_pos, dt, field, interest_constraints)
            dq = self._apply_handover_target_guard(
                J, dq, dq_nominal, ee_pos, dt, handover_target, protect)
            self._update_adp_solution_stats(J, dq, gradV, v_avoid)
            return dq

        dq = cp.Variable(self.n)
        ee_vel = _cvx_matvec(cp, J, dq)
        track_weight = (
            max(1.0, float(handover_tracking_weight))
            if protect else 1.0)
        cost = track_weight * cp.sum_squares(ee_vel - v_des)
        cost += self.damping * cp.sum_squares(dq)
        cost += self.lam_nominal * cp.sum_squares(dq - dq_nom)
        if v_avoid is not None and float(lambda_adp_arm) > 0.0:
            cost += float(lambda_adp_arm) * cp.sum_squares(ee_vel - v_avoid)
        cons = [cp.norm(dq, "inf") <= self.dq_max,
                cp.norm(ee_vel) <= cap]
        prob = cp.Problem(cp.Minimize(cost), cons)
        try:
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            if dq.value is None:
                self.fallback_count += 1
                self.last_solver_status = "fallback: {}".format(prob.status)
                dq_nominal = self._dls(J, v_des, dq_nom, cap)
                return self._apply_handover_target_guard(
                    J, dq_nominal, dq_nominal, ee_pos, dt,
                    handover_target, protect)
            self.solve_success_count += 1
            self.last_solver_status = str(prob.status)
            self.last_dls_adp_used = 0
            self.last_qp_used = 1
            out = np.asarray(dq.value, float)
            out = self._enforce_interest_constraints(
                J, out, ee_pos, dt, field, interest_constraints)
            dq_nominal = self._dls(J, v_des, dq_nom, cap)
            self._update_control_delta_stats(v_des, np.dot(J, out), dq_nominal, out)
            out = self._apply_handover_target_guard(
                J, out, dq_nominal, ee_pos, dt, handover_target, protect)
            self._update_adp_solution_stats(J, out, gradV, v_avoid)
            return out
        except Exception as exc:
            self.fallback_count += 1
            self.last_solver_status = "fallback: {}".format(type(exc).__name__)
            v_cmd = self._compose_adp_velocity(
                v_des, v_avoid, cap, lambda_adp_arm=lambda_adp_arm,
                adp_blend_alpha=adp_blend_alpha, preserve_progress=True)
            dq_nominal = self._dls(J, v_des, dq_nom, cap)
            dq_out = self._dls(J, v_cmd, dq_nom, cap)
            dq_out = self._enforce_interest_constraints(
                J, dq_out, ee_pos, dt, field, interest_constraints)
            self.last_dls_adp_used = 1 if v_avoid is not None else 0
            self.last_qp_used = 0
            self._update_control_delta_stats(v_des, v_cmd, dq_nominal, dq_out)
            dq_out = self._apply_handover_target_guard(
                J, dq_out, dq_nominal, ee_pos, dt, handover_target, protect)
            self._update_adp_solution_stats(J, dq_out, gradV, v_avoid)
            return dq_out

    def _phase_weights(self, phase, override=None):
        phase_name = str(phase or "approach")
        if phase_name.lstrip("-").isdigit():
            phase_name = (
                "handover" if int(phase_name) == 3 else
                "return" if int(phase_name) == 4 else "approach")
        configured = dict(self.phase_cost_weights)
        configured.update(dict(override or {}))
        row = dict(configured.get(phase_name, {}) or {})
        return phase_name, {
            "tracking": float(row.get("tracking_multiplier", 1.0)),
            "risk": float(row.get("risk_multiplier", 1.0)),
            "smooth": float(row.get("smooth_multiplier", 1.0)),
            "task": float(row.get("task_multiplier", 1.0)),
        }

    def _arm_step_controls(self, previous, warm, dt, alternatives=None):
        previous = np.asarray(previous, float)
        warm = np.asarray(warm, float)
        delta = max(0.0, self.ddq_max) * float(dt)
        seeds = [previous, warm, 0.75 * warm, 0.5 * warm, np.zeros(self.n)]
        for alternative in list(alternatives or []):
            alternative = np.asarray(alternative, float)
            seeds.extend([alternative, 0.75 * alternative,
                          0.5 * alternative])
        if delta > 0.0:
            order = np.argsort(-np.abs(warm - previous))[:min(3, self.n)]
            for idx in order:
                for sign in (-1.0, 1.0):
                    cand = previous.copy()
                    cand[int(idx)] += sign * delta
                    seeds.append(cand)
        unique = []
        seen = set()
        for seed in seeds:
            cand = np.clip(seed, previous - delta, previous + delta)
            cand = np.clip(cand, -self.dq_max, self.dq_max)
            key = tuple(np.round(cand, 8).tolist())
            if key not in seen:
                seen.add(key)
                unique.append(cand)
        return unique

    def _sampled_predictive_solve(
            self, J, v_des, dq_nom, cap, q, ee_pos, dt, field,
            interest_constraints, target_pos, phase, corridor, v_avoid,
            lambda_adp_arm, adp_blend_alpha, handover_target, protect,
            handover_tracking_weight, phase_cost_weights):
        """Optimize a short joint-velocity sequence; DLS is only its seed."""
        J = np.asarray(J, float)
        q0 = np.zeros(self.n, float) if q is None else np.asarray(q, float)
        ee0 = (np.zeros(J.shape[0], float) if ee_pos is None
               else np.asarray(ee_pos, float))
        if q0.size != self.n or J.shape[1] != self.n:
            raise ValueError("ArmMPC predictive state/Jacobian dimension mismatch")
        dt = max(float(dt), 1e-4)
        v_cmd = self._compose_adp_velocity(
            v_des, v_avoid, cap, lambda_adp_arm=lambda_adp_arm,
            adp_blend_alpha=adp_blend_alpha, preserve_progress=True)
        warm = self._dls(J, v_cmd, dq_nom, cap)
        task_warm = self._dls(J, v_des, dq_nom, cap)
        target = (ee0 + v_cmd * dt * self.N if target_pos is None
                  else np.asarray(target_pos, float))
        phase_name, weights = self._phase_weights(phase, phase_cost_weights)
        cfg = dict(interest_constraints or {})
        interest_enabled = bool(cfg.get("enabled", False) and field is not None)
        interest_rho = float(cfg.get("rho", float("inf")))
        offsets = cfg.get("offsets")
        labels = cfg.get("labels")
        self.last_interest_kinematics_source = (
            "proxy_rigid_offset" if offsets is not None else "not_used")
        violations = {
            "joint_position": 0, "joint_velocity": 0,
            "joint_acceleration": 0, "ee_speed": 0,
            "trajectory_tube": 0, "interest_point": 0, "forbidden": 0,
            "task_progress": 0, "first_step_task_progress": 0,
        }
        initial_target_error = float(np.linalg.norm(ee0[:3] - target[:3]))
        max_task_regression = max(0.0, float(cfg.get(
            "max_task_regression", 1e-6)))
        task_progress_tolerance = max(0.0, float(cfg.get(
            "task_progress_tolerance", self.task_progress_tolerance)))
        progress_scale = min(
            max(0.0, initial_target_error - task_progress_tolerance),
            max(0.0, float(cap)) * float(self.N) * float(dt))
        required_terminal_progress = float(
            self.min_terminal_progress_ratio * progress_scale)
        first_step_progress_scale = min(
            max(0.0, initial_target_error - task_progress_tolerance),
            max(0.0, float(cap)) * float(dt))
        required_first_step_progress = float(
            self.min_terminal_progress_ratio * first_step_progress_scale)
        empty = {"tracking": 0.0, "task": 0.0, "risk": 0.0,
                 "tube": 0.0, "control": 0.0, "smooth": 0.0}
        beam = [{"cost": 0.0, "q": q0.copy(), "ee": ee0.copy(),
                 "dq": np.asarray(dq_nom, float).copy(), "controls": [],
                 "joints": [], "ees": [], "parts": dict(empty)}]
        for k in range(self.N):
            expanded = []
            for item in beam:
                previous = np.asarray(item["dq"], float)
                for cand in self._arm_step_controls(
                        previous, warm, dt, alternatives=[task_warm]):
                    # Project sampled velocity onto the one-step position
                    # interval.  This retains a boundary-reaching control
                    # instead of leaving only an oversized sample or hold.
                    position_velocity_lower = (
                        self.joint_lower - np.asarray(item["q"], float)) / dt
                    position_velocity_upper = (
                        self.joint_upper - np.asarray(item["q"], float)) / dt
                    cand = np.minimum(
                        np.maximum(cand, position_velocity_lower),
                        position_velocity_upper)
                    if np.any(np.abs(cand) > self.dq_max + 1e-9):
                        violations["joint_velocity"] += 1
                        continue
                    if np.any(np.abs(cand - previous) >
                              self.ddq_max * dt + 1e-9):
                        violations["joint_acceleration"] += 1
                        continue
                    ee_vel = np.dot(J, cand)
                    if np.linalg.norm(ee_vel) > cap + 1e-9:
                        violations["ee_speed"] += 1
                        continue
                    q_next = np.asarray(item["q"], float) + cand * dt
                    if (np.any(q_next < self.joint_lower - 1e-9) or
                            np.any(q_next > self.joint_upper + 1e-9)):
                        violations["joint_position"] += 1
                        continue
                    ee_next = np.asarray(item["ee"], float) + ee_vel * dt
                    # Only the first control is applied before replanning.  Do
                    # not allow the first step to move away from the active
                    # task target, but keep horizon-level progress as the hard
                    # criterion.  Real arm Jacobians near limits can require a
                    # neutral first step before the sequence makes progress.
                    if k == 0 and required_first_step_progress > 0.0:
                        first_step_error = float(np.linalg.norm(
                            ee_next[:3] - target[:3]))
                        if first_step_error > initial_target_error + max(
                                max_task_regression,
                                task_progress_tolerance) + 1e-9:
                            violations["first_step_task_progress"] += 1
                            continue
                    tube_cost = 0.0
                    if corridor is not None:
                        _projection, distance = corridor.project(ee_next[:3])
                        if float(distance) > float(corridor.radius) + 1e-9:
                            violations["trajectory_tube"] += 1
                            continue
                    risk_cost = 0.0
                    if interest_enabled and offsets is not None:
                        risk = pose_interest_risk(
                            field, ee_next[:3], offsets=offsets, labels=labels)
                        hit, _label, _anchor, reason = forbidden_anchor_hit(
                            field, risk.get("labels", []), risk.get("points", []))
                        if hit:
                            violations["forbidden"] += 1
                            self.last_reject_forbidden_count += 1
                            if not self.first_predicted_forbidden_reason:
                                self.first_predicted_forbidden_reason = reason
                            continue
                        phi = float(risk.get("phi_max", 0.0))
                        if phi > interest_rho:
                            violations["interest_point"] += 1
                            self.last_reject_interest_phi_count += 1
                            continue
                        normalized_risk = phi / max(interest_rho, 1e-6)
                        risk_cost = (
                            weights["risk"] * normalized_risk ** 2 /
                            float(self.N))
                    fraction = float(k + 1) / float(self.N)
                    reference = ee0 + fraction * (target - ee0)
                    tracking = weights["tracking"] * float(
                        np.sum((ee_next - reference) ** 2))
                    task_cost = weights["task"] * float(
                        np.sum((ee_next - target) ** 2)) / float(self.N)
                    if k == self.N - 1 and initial_target_error > 1e-6:
                        terminal_error = float(np.linalg.norm(
                            ee_next[:3] - target[:3]))
                        if required_terminal_progress > 0.0:
                            if terminal_error > (
                                    initial_target_error -
                                    required_terminal_progress + 1e-9):
                                violations["task_progress"] += 1
                                continue
                        elif terminal_error > (
                                initial_target_error +
                                max_task_regression):
                            violations["task_progress"] += 1
                            continue
                    control = self.damping * float(np.dot(cand, cand))
                    smooth = weights["smooth"] * self.lam_nominal * float(
                        np.sum((cand - previous) ** 2))
                    if protect:
                        tracking *= max(1.0, float(handover_tracking_weight))
                    parts = dict(item["parts"])
                    for key, value in (("tracking", tracking), ("task", task_cost),
                                       ("risk", risk_cost), ("tube", tube_cost),
                                       ("control", control), ("smooth", smooth)):
                        parts[key] += float(value)
                    step_cost = sum((tracking, task_cost, risk_cost,
                                     tube_cost, control, smooth))
                    expanded.append({
                        "cost": float(item["cost"] + step_cost),
                        "q": q_next, "ee": ee_next, "dq": cand,
                        "controls": item["controls"] + [cand.copy()],
                        "joints": item["joints"] + [q_next.copy()],
                        "ees": item["ees"] + [ee_next.copy()], "parts": parts,
                    })
            if not expanded:
                self.fallback_count += 1
                self.last_solver_status = "safe_stop: no_feasible_joint_sequence"
                self.last_constraint_violation = violations
                self.last_prediction_model = "linearized_jacobian"
                return np.zeros(self.n, float)
            expanded.sort(key=lambda item: item["cost"])
            if k == 0 and required_first_step_progress > 0.0:
                live_first_step = []
                for item in expanded:
                    ees = list(item.get("ees", []))
                    if not ees:
                        continue
                    first_error = float(np.linalg.norm(
                        np.asarray(ees[0], float)[:3] - target[:3]))
                    first_progress = initial_target_error - first_error
                    if first_progress + 1e-9 >= required_first_step_progress:
                        live_first_step.append(item)
                if live_first_step:
                    keep = []
                    seen = set()
                    for item in live_first_step + expanded:
                        first = np.asarray(item["controls"][0], float)
                        key = tuple(np.round(first, 8).tolist())
                        if key in seen:
                            continue
                        seen.add(key)
                        keep.append(item)
                        if len(keep) >= self.beam_width:
                            break
                    beam = keep
                else:
                    beam = expanded[:self.beam_width]
            else:
                beam = expanded[:self.beam_width]
        first_step_live = []
        if required_first_step_progress > 0.0:
            for item in beam:
                ees = list(item.get("ees", []))
                if not ees:
                    continue
                first_error = float(np.linalg.norm(
                    np.asarray(ees[0], float)[:3] - target[:3]))
                first_progress = initial_target_error - first_error
                if first_progress + 1e-9 >= required_first_step_progress:
                    first_step_live.append(item)
        selection_pool = first_step_live if first_step_live else beam
        best = min(selection_pool, key=lambda item: item["cost"])
        controls = list(best["controls"])
        out = np.asarray(controls[0], float)
        predicted_joints = [np.asarray(x, float).copy()
                            for x in best["joints"]]
        predicted_ees = [np.asarray(x, float).copy()
                         for x in best["ees"]]
        task_warm_first_step_used = False
        selected_first_progress = 0.0
        if len(predicted_ees):
            selected_first_error = float(np.linalg.norm(
                predicted_ees[0][:3] - target[:3]))
            selected_first_progress = float(
                initial_target_error - selected_first_error)
        if (selected_first_progress <= 1e-9 and
                initial_target_error > task_progress_tolerance + 1e-9):
            cand = np.asarray(task_warm, float).copy()
            previous = np.asarray(dq_nom, float)
            delta = max(0.0, self.ddq_max) * float(dt)
            cand = np.clip(cand, previous - delta, previous + delta)
            cand = np.clip(cand, -self.dq_max, self.dq_max)
            position_velocity_lower = (self.joint_lower - q0) / dt
            position_velocity_upper = (self.joint_upper - q0) / dt
            cand = np.minimum(
                np.maximum(cand, position_velocity_lower),
                position_velocity_upper)
            ee_vel = np.dot(J, cand)
            q_next = q0 + cand * dt
            ee_next = ee0 + ee_vel * dt
            cand_error = float(np.linalg.norm(ee_next[:3] - target[:3]))
            cand_progress = float(initial_target_error - cand_error)
            cand_ok = (
                cand_progress > 1e-9 and
                np.all(np.abs(cand) <= self.dq_max + 1e-9) and
                np.all(np.abs(cand - previous) <=
                       self.ddq_max * dt + 1e-9) and
                np.linalg.norm(ee_vel) <= cap + 1e-9 and
                np.all(q_next >= self.joint_lower - 1e-9) and
                np.all(q_next <= self.joint_upper + 1e-9))
            if cand_ok and corridor is not None:
                _projection, distance = corridor.project(ee_next[:3])
                cand_ok = (
                    float(distance) <= float(corridor.radius) + 1e-9)
            if cand_ok and interest_enabled and offsets is not None:
                risk = pose_interest_risk(
                    field, ee_next[:3], offsets=offsets, labels=labels)
                hit, _label, _anchor, _reason = forbidden_anchor_hit(
                    field, risk.get("labels", []), risk.get("points", []))
                phi = float(risk.get("phi_max", 0.0))
                cand_ok = (not hit and phi <= interest_rho + 1e-9)
            if cand_ok:
                out = cand
                controls[0] = cand.copy()
                if predicted_joints:
                    predicted_joints[0] = q_next.copy()
                if predicted_ees:
                    predicted_ees[0] = ee_next.copy()
                task_warm_first_step_used = True
                selected_first_progress = cand_progress
        self.solve_success_count += 1
        self.last_solver_status = (
            "predictive_joint_beam_task_warm_first_step"
            if task_warm_first_step_used else "predictive_joint_beam")
        self.last_prediction_model = "linearized_jacobian"
        self.last_dls_adp_used = 0
        self.last_qp_used = 0
        self.last_predicted_joint_states = [
            x.tolist() for x in predicted_joints]
        self.last_predicted_controls = [x.tolist() for x in controls]
        self.last_predicted_ee_states = [x.tolist() for x in predicted_ees]
        self.last_objective_terms = dict(best["parts"])
        self.last_objective_terms["initial_target_error"] = float(
            initial_target_error)
        self.last_objective_terms["terminal_target_error"] = float(
            np.linalg.norm(np.asarray(best["ee"], float)[:3] - target[:3]))
        self.last_objective_terms["maximum_task_regression"] = float(
            max_task_regression)
        self.last_objective_terms["required_terminal_progress"] = float(
            required_terminal_progress)
        self.last_objective_terms["required_first_step_progress"] = float(
            required_first_step_progress)
        self.last_objective_terms["first_step_live_candidate_count"] = int(
            len(first_step_live))
        self.last_objective_terms["first_step_live_selection_used"] = int(
            bool(first_step_live))
        self.last_objective_terms["task_warm_first_step_used"] = int(
            task_warm_first_step_used)
        self.last_objective_terms["selected_first_step_progress"] = float(
            selected_first_progress)
        self.last_objective_terms["first_step_target_error"] = float(
            np.linalg.norm(np.asarray(predicted_ees[0], float)[:3] - target[:3]))
        self.last_objective_terms["task_progress_tolerance"] = float(
            task_progress_tolerance)
        self.last_objective_terms["phase"] = phase_name
        self.last_objective_terms["phase_weights"] = weights
        self.last_constraint_violation = violations
        self.last_control_sequence_varies = bool(any(
            not np.allclose(controls[0], item) for item in controls[1:]))
        self._update_control_delta_stats(v_des, np.dot(J, out), warm, out)
        self._update_adp_solution_stats(J, out, np.zeros(3), v_avoid)
        return out

    def _apply_handover_target_guard(self, J, dq, dq_nominal, ee_pos, dt,
                                     handover_target, protect):
        if not protect:
            return dq
        J = np.asarray(J, float)
        dq = np.asarray(dq, float)
        dq_nominal = np.asarray(dq_nominal, float)
        ee = np.asarray(ee_pos, float)[:3]
        target = np.asarray(handover_target, float)[:3]
        raw_point = ee + np.dot(J, dq)[:3] * float(dt)
        nominal_point = ee + np.dot(J, dq_nominal)[:3] * float(dt)
        hold = np.zeros_like(dq)
        hold_point = ee + np.dot(J, hold)[:3] * float(dt)
        current_error = float(np.linalg.norm(ee - target))
        raw_error = float(np.linalg.norm(raw_point - target))
        nominal_error = float(np.linalg.norm(nominal_point - target))
        candidates = [
            ("raw_mpc", dq, raw_point, raw_error),
            ("tracking_nominal", dq_nominal, nominal_point, nominal_error),
            ("hold_current", hold, hold_point, current_error),
        ]
        best_name, out, protected_point, protected_error = min(
            candidates, key=lambda item: item[3])
        if protected_error > current_error + 1e-9:
            best_name, out, protected_point, protected_error = (
                "hold_current", hold, hold_point, current_error)
        guarded = bool(best_name != "raw_mpc")
        self.last_handover_protection = {
            "active": True,
            "guard_applied": bool(guarded),
            "selected_output": str(best_name),
            "current_point": [float(v) for v in ee],
            "handover_reference_point": [float(v) for v in target],
            "raw_mpc_output_point": [float(v) for v in raw_point],
            "protected_mpc_output_point": [float(v) for v in protected_point],
            "current_error": current_error,
            "before_error": raw_error,
            "after_error": protected_error,
            "nominal_tracking_error": nominal_error,
        }
        return out

    def _enforce_interest_constraints(self, J, dq, ee_pos, dt, field,
                                      interest_constraints):
        cfg = interest_constraints or {}
        self.last_reject_forbidden_count = 0
        self.last_reject_interest_phi_count = 0
        self.first_predicted_forbidden_reason = ""
        if (not cfg or not bool(cfg.get("enabled", False)) or field is None or
                ee_pos is None):
            return dq
        rho = float(cfg.get("rho", float("inf")))
        offsets = cfg.get("offsets")
        if offsets is None or not np.isfinite(rho):
            return dq
        labels = cfg.get("labels")
        J = np.asarray(J, float)
        dq = np.asarray(dq, float)
        ee_pos = np.asarray(ee_pos, float)
        for scale in (1.0, 0.75, 0.5, 0.25, 0.0):
            cand = dq * scale
            pred_ee = ee_pos + np.dot(J, cand) * float(dt)
            risk = pose_interest_risk(
                field, pred_ee, offsets=offsets,
                labels=labels)
            hit, _label, _anchor, reason = forbidden_anchor_hit(
                field, risk.get("labels", []), risk.get("points", []))
            if hit:
                self.last_reject_forbidden_count += 1
                if not self.first_predicted_forbidden_reason:
                    self.first_predicted_forbidden_reason = reason
                continue
            if float(risk.get("phi_max", 0.0)) > rho:
                self.last_reject_interest_phi_count += 1
                continue
            else:
                if scale < 1.0:
                    self.last_solver_status += "_ip_limited"
                return cand
        return np.zeros_like(dq)

    def _dls(self, J, v_des, dq_nom, cap):
        speed = np.linalg.norm(v_des)
        if speed > cap and speed > 1e-9:
            v_des = v_des * (cap / speed)
        JJ = np.dot(J, J.T) + (self.damping ** 2) * np.eye(J.shape[0])
        dq = np.dot(J.T, np.linalg.solve(JJ, v_des))
        dq = dq + self.lam_nominal * (dq_nom - dq) * 0.0
        return np.clip(dq, -self.dq_max, self.dq_max)

    def _compose_adp_velocity(self, v_des, v_avoid, cap,
                              lambda_adp_arm=0.0, adp_blend_alpha=0.08,
                              preserve_progress=True):
        v_goal = np.asarray(v_des, float)
        speed = float(np.linalg.norm(v_goal))
        if speed > float(cap) and speed > 1e-9:
            v_goal = v_goal * (float(cap) / speed)
        if v_avoid is None or float(lambda_adp_arm) <= 0.0:
            return v_goal
        v_adp = np.asarray(v_avoid, float)
        goal_speed = float(np.linalg.norm(v_goal))
        if preserve_progress and goal_speed > 1e-9:
            e_goal = v_goal / goal_speed
            backward = float(np.dot(v_adp, e_goal))
            if backward < 0.0:
                v_adp = v_adp - backward * e_goal
        alpha = max(0.0, min(float(adp_blend_alpha), 0.35))
        v_cmd = v_goal + alpha * v_adp
        cmd_speed = float(np.linalg.norm(v_cmd))
        if cmd_speed > float(cap) and cmd_speed > 1e-9:
            v_cmd = v_cmd * (float(cap) / cmd_speed)
        return v_cmd

    def _adp_value_at(self, p, critic, feature_builder, field, gate_info,
                      interest_risk, target_pos, phase):
        features = feature_builder.build_arm(
            np.asarray(p, float), target_pos, field,
            gate_info=gate_info or {}, interest_risk=interest_risk or {},
            phase=phase)
        return float(critic.predict(features))

    def _adp_grad_ee(self, ee_pos, critic, feature_builder, field, gate_info,
                     interest_risk, target_pos, phase, eps=0.01):
        grad = np.zeros(3, float)
        ee = np.asarray(ee_pos, float)
        for j in range(3):
            d = np.zeros(3, float)
            d[j] = float(eps)
            vp = self._adp_value_at(
                ee + d, critic, feature_builder, field, gate_info,
                interest_risk, target_pos, phase)
            vm = self._adp_value_at(
                ee - d, critic, feature_builder, field, gate_info,
                interest_risk, target_pos, phase)
            grad[j] = (vp - vm) / max(2.0 * float(eps), 1e-9)
        return np.clip(grad, -self.adp_grad_clip, self.adp_grad_clip)

    def _adp_avoidance(self, ee_pos, critic, feature_builder, field, gate_info,
                       interest_risk, target_pos, phase, lambda_adp_arm,
                       adp_grad_eps, adp_descent_gain):
        self.last_adp_grad_norm = 0.0
        self.last_adp_soft_cost = 0.0
        self.last_v_adp_alignment = 0.0
        if (critic is None or feature_builder is None or field is None or
                ee_pos is None or target_pos is None or
                float(lambda_adp_arm) <= 0.0):
            return np.zeros(3, float), None
        grad = self._adp_grad_ee(
            ee_pos, critic, feature_builder, field, gate_info,
            interest_risk, target_pos, phase, eps=adp_grad_eps)
        norm = float(np.linalg.norm(grad))
        self.last_adp_grad_norm = norm
        if norm <= 1e-9:
            return grad, None
        unit = grad / max(norm, 1.0)
        return grad, -float(adp_descent_gain) * unit

    def _update_control_delta_stats(self, v_raw, v_adp, dq_nominal, dq_adp):
        v_raw = np.asarray(v_raw, float)
        v_adp = np.asarray(v_adp, float)
        dq_nominal = np.asarray(dq_nominal, float)
        dq_adp = np.asarray(dq_adp, float)
        self.last_v_des_raw_norm = float(np.linalg.norm(v_raw))
        self.last_v_des_adp_norm = float(np.linalg.norm(v_adp))
        self.last_v_des_delta_norm = float(np.linalg.norm(v_adp - v_raw))
        self.last_dq_nominal_norm = float(np.linalg.norm(dq_nominal))
        self.last_dq_adp_norm = float(np.linalg.norm(dq_adp))
        self.last_dq_delta_norm = float(np.linalg.norm(dq_adp - dq_nominal))

    def _update_adp_solution_stats(self, J, dq, gradV, v_avoid):
        if v_avoid is None:
            self.last_adp_soft_cost = 0.0
            self.last_v_adp_alignment = 0.0
            return
        ee_vel = np.dot(np.asarray(J, float), np.asarray(dq, float))
        diff = ee_vel - v_avoid
        self.last_adp_soft_cost = float(np.dot(diff, diff))
        gnorm = float(np.linalg.norm(gradV))
        vnorm = float(np.linalg.norm(ee_vel))
        if gnorm > 1e-9 and vnorm > 1e-9:
            self.last_v_adp_alignment = float(
                np.dot(ee_vel, -gradV) / (vnorm * gnorm))
        else:
            self.last_v_adp_alignment = 0.0

class WheelchairMPC:
    def __init__(self, horizon=12, dt=0.2, v_max=0.6, w_max=1.0,
                 a_max=0.5, lam_track=1.0, lam_social=0.6, lam_u=0.05,
                 lam_du=0.2, alpha_max=1.5, beam_width=12):
        self.N = horizon
        self.dt = dt
        self.v_max = v_max
        self.w_max = w_max
        self.a_max = a_max
        self.alpha_max = alpha_max
        self.beam_width = max(2, int(beam_width))
        self.lam_track = lam_track
        self.lam_social = lam_social
        self.lam_u = lam_u
        self.lam_du = lam_du
        self.lam_tube = 0.8
        self.lam_progress = 2.8
        self.lam_speed = 0.25
        self.lam_ref_progress = 1.0
        self.lam_goal_terminal = 8.0
        self.lam_stall = 10.0
        self.min_progress_per_solve = 0.005
        self.near_goal_radius = 0.50
        self.near_goal_goal_weight = 18.0
        self.near_goal_adp_scale = 0.20
        self.near_goal_social_scale = 0.5
        self.final_approach_radius = 0.90
        self.final_heading_threshold = 0.75
        self.final_heading_gain = 1.6
        self.final_creep_v = 0.10
        self.final_min_v = 0.16
        self.final_max_v = 0.30
        self.final_forward_gain = 0.75
        self.lam_heading = 2.5
        self.lam_heading_stage = 1.5
        self.min_heading_improvement = 0.08
        self.first_step_progress_ratio = 0.50
        self.heading_recovery_w_max = 0.45
        self.last_final_approach_used = 0
        self.last_terminal_adp_cost = 0.0
        self.last_total_cost = 0.0
        self.last_social_cost = 0.0
        self.last_tube_cost = 0.0
        self.last_track_cost = 0.0
        self.last_control_cost = 0.0
        self.last_reject_forbidden_count = 0
        self.last_reject_interest_phi_count = 0
        self.first_predicted_forbidden_reason = ""
        self.last_solver_status = "not_called"
        self.last_topology_constraint = {}
        self.last_predicted_states = []
        self.last_predicted_controls = []
        self.last_objective_terms = {}
        self.last_constraint_violation = {}
        self.last_control_sequence_varies = False
        self.last_sequence_progress = 0.0
        self.last_heading_improvement = 0.0
        self.last_alignment_translation = 0.0

    def solve(self, state, ref_points, field, corridor=None, u_prev=None,
              critic=None, feature_builder=None, lambda_adp_terminal=0.0,
              goal=None, gate_info=None, interest_risk=None,
              use_adp_terminal=False, interest_constraints=None,
              topology_constraint=None, predictive=True):
        x0 = np.asarray(state, float)
        ref = np.asarray(ref_points, float)
        u_prev = np.zeros(2) if u_prev is None else np.asarray(u_prev, float)
        self.last_final_approach_used = 0
        self.last_topology_constraint = dict(topology_constraint or {})
        self.last_predicted_states = []
        self.last_predicted_controls = []
        self.last_objective_terms = {}
        self.last_constraint_violation = {}
        self.last_control_sequence_varies = False
        self.last_sequence_progress = 0.0
        self.last_heading_improvement = 0.0
        self.last_alignment_translation = 0.0

        if ref.size == 0:
            self.last_solver_status = "safe_stop: empty_ref"
            return 0.0, 0.0

        goal_arr = ref[-1] if goal is None else np.asarray(goal, float)
        dist_goal = float(np.linalg.norm(x0[:2] - goal_arr[:2]))
        if not bool(predictive):
            warm = self._pure_pursuit_u(x0, ref, field, u_prev)
            self.last_predicted_controls = [warm.tolist()]
            self.last_predicted_states = [self._step(x0, warm).tolist()]
            self.last_objective_terms = {"baseline_pure_pursuit": 0.0}
            self.last_solver_status = "baseline_pure_pursuit"
            return float(warm[0]), float(warm[1])

        return self._sampled_predictive_solve(
            x0, ref, field, corridor, u_prev, critic, feature_builder,
            lambda_adp_terminal, goal_arr, gate_info, interest_risk,
            interest_constraints)

    def _step(self, x, u):
        v, w = float(u[0]), float(u[1])
        out = np.array(x, float)
        out[0] += v * np.cos(out[2]) * self.dt
        out[1] += v * np.sin(out[2]) * self.dt
        out[2] += w * self.dt
        out[2] = np.arctan2(np.sin(out[2]), np.cos(out[2]))
        return out

    def _goal_heading_error(self, x0, goal):
        goal = np.asarray(goal, float)
        desired = np.arctan2(goal[1] - x0[1], goal[0] - x0[0])
        return float(np.arctan2(np.sin(desired - x0[2]),
                                np.cos(desired - x0[2])))

    def _goal_seek_u(self, x0, goal):
        dist = float(np.linalg.norm(np.asarray(goal, float)[:2] - x0[:2]))
        herr = self._goal_heading_error(x0, goal)
        w = float(np.clip(
            self.final_heading_gain * herr, -self.w_max, self.w_max))
        if abs(herr) > self.final_heading_threshold:
            v = 0.0
        else:
            v = self.final_forward_gain * dist
            v = float(np.clip(v, 0.0, self.final_max_v))
        return np.array([v, w], float)

    def _sequence_step_controls(self, u_prev, warm_u=None, goal_u=None):
        """Return rate-limited controls for one prediction step."""
        prev = np.asarray(u_prev, float)
        dv = float(self.a_max) * float(self.dt)
        dw = float(self.alpha_max) * float(self.dt)
        candidates = []
        for delta_v in (-dv, 0.0, dv):
            for delta_w in (-dw, 0.0, dw):
                candidates.append(prev + np.array([delta_v, delta_w], float))
        for seed in (warm_u, goal_u):
            if seed is None:
                continue
            seed = np.asarray(seed, float)
            candidates.append(np.array([
                np.clip(seed[0], prev[0] - dv, prev[0] + dv),
                np.clip(seed[1], prev[1] - dw, prev[1] + dw),
            ], float))
        unique = []
        seen = set()
        for item in candidates:
            item = np.array([
                np.clip(item[0], 0.0, self.v_max),
                np.clip(item[1], -self.w_max, self.w_max),
            ], float)
            key = (round(float(item[0]), 8), round(float(item[1]), 8))
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique

    def _sampled_predictive_solve(self, x0, ref, field, corridor, u_prev,
                                  critic, feature_builder,
                                  lambda_adp_terminal, goal, gate_info,
                                  interest_risk, interest_constraints=None):
        has_adp_terminal = (
            lambda_adp_terminal > 0.0 and
            critic is not None and feature_builder is not None)
        horizon = min(self.N, max(1, ref.shape[0]))
        goal = ref[-1] if goal is None else np.asarray(goal, float)
        gate_info = gate_info or {}
        interest_risk = interest_risk or {}
        dist0 = float(np.linalg.norm(x0[:2] - goal[:2]))
        if dist0 < self.final_approach_radius:
            self.last_final_approach_used = 1
        interest_constraints = interest_constraints or {}
        interest_enabled = bool(interest_constraints.get("enabled", False))
        interest_rho = float(interest_constraints.get("rho", float("inf")))
        self.last_reject_forbidden_count = 0
        self.last_reject_interest_phi_count = 0
        self.first_predicted_forbidden_reason = ""
        manifold_payload = dict(
            self.last_topology_constraint.get("manifold_constraint", {}) or {})
        manifold_mode = str(manifold_payload.get(
            "manifold_constraint_mode", manifold_payload.get("mode", "soft"))).lower()
        manifold_evaluator = ManifoldConstraintEvaluator(
            manifold_constraint=manifold_payload,
            corridor_constraint=dict(
                self.last_topology_constraint.get("corridor_constraint", {}) or {}),
            risk_field=field)
        corridor_payload = dict(
            self.last_topology_constraint.get("corridor_constraint", {}) or {})
        tube_payload = dict(corridor_payload.get("tube_constraint", {}) or {})
        tube_mode = str(
            tube_payload.get(
                "mode", self.last_topology_constraint.get(
                    "tube_constraint_mode", "hard")) or "hard").lower()
        soft_tolerance = float(manifold_payload.get("soft_tolerance", 0.08) or 0.08)
        violation_counts = {
            "control_bounds": 0,
            "control_rate": 0,
            "trajectory_tube": 0,
            "manifold": 0,
            "interest_point": 0,
            "forbidden": 0,
            "insufficient_progress": 0,
            "nonprogressive_rollout": 0,
        }
        min_alignment_first_v = min(0.02, self.a_max * self.dt)
        empty_parts = {
            "tracking": 0.0,
            "heading_tracking": 0.0,
            "social": 0.0,
            "tube": 0.0,
            "control": 0.0,
            "smooth": 0.0,
        }
        # Beam branches frequently converge to the exact same predicted pose.
        # Safety and interest evaluations are deterministic within one solve,
        # so reuse them instead of repeating the social-field calculation for
        # every equivalent branch.
        manifold_state_cache = {}
        interest_state_cache = {}
        corridor_projection_cache = {}
        beam = [{
            "cost": 0.0,
            "state": np.array(x0, float),
            "controls": [],
            "states": [],
            "parts": dict(empty_parts),
        }]
        for k in range(horizon):
            expanded = []
            for item in beam:
                x = np.asarray(item["state"], float)
                previous = (
                    np.asarray(item["controls"][-1], float)
                    if item["controls"] else np.asarray(u_prev, float))
                local_ref = ref[min(k, ref.shape[0] - 1):]
                warm = self._pure_pursuit_u(
                    x, local_ref if len(local_ref) else ref, field, previous)
                local_goal_u = self._goal_seek_u(x, goal)
                for u in self._sequence_step_controls(
                        previous, warm_u=warm, goal_u=local_goal_u):
                    if (abs(float(u[0] - previous[0])) >
                            self.a_max * self.dt + 1e-9 or
                            abs(float(u[1] - previous[1])) >
                            self.alpha_max * self.dt + 1e-9):
                        violation_counts["control_rate"] += 1
                        continue
                    # The first command is the only part of the horizon that
                    # will actually execute.  Preserve live branches before
                    # beam pruning can prefer a zero-speed hold whose motion
                    # is perpetually deferred to later controls.
                    if (k == 0 and dist0 >= 0.12 and
                            float(u[0]) + 1e-9 < min_alignment_first_v):
                        violation_counts["insufficient_progress"] += 1
                        continue
                    x_next = self._step(x, u)
                    state_key = tuple(float(value) for value in x_next[:3])
                    position_key = state_key[:2]
                    parts = dict(item["parts"])
                    hard_violation = False
                    step_soft_cost = 0.0
                    manifold_state = manifold_state_cache.get(position_key)
                    if manifold_state is None:
                        manifold_state = manifold_evaluator.evaluate_state(
                            [x_next[0], x_next[1], 0.0])
                        manifold_state_cache[position_key] = manifold_state
                    manifold_risk_violation = max(
                        0.0, float(manifold_state.get("risk", 0.0)) -
                        float(manifold_evaluator.risk_threshold))
                    manifold_clearance_violation = 0.0
                    if bool(manifold_state.get("clearance_available", False)):
                        manifold_clearance_violation = max(
                            0.0, float(manifold_evaluator.required_clearance) -
                            float(manifold_state.get("clearance", 0.0)))
                    manifold_violation = max(
                        manifold_risk_violation, manifold_clearance_violation)
                    if manifold_violation > 1e-9:
                        if (manifold_mode == "hard" or
                                manifold_violation > soft_tolerance):
                            hard_violation = True
                            violation_counts["manifold"] += 1
                        else:
                            step_soft_cost += 10.0 * float(
                                manifold_violation ** 2)
                    if hard_violation:
                        continue
                    if interest_enabled:
                        cached_interest = interest_state_cache.get(state_key)
                        if cached_interest is None:
                            summary = pose_interest_risk(
                                field, x_next,
                                local_points=interest_constraints.get(
                                    "local_points"),
                                labels=interest_constraints.get("labels"))
                            hit, _label, _anchor, reason = forbidden_anchor_hit(
                                field, summary.get("labels", []),
                                summary.get("points", []))
                            cached_interest = (summary, hit, reason)
                            interest_state_cache[state_key] = cached_interest
                        summary, hit, reason = cached_interest
                        if hit:
                            violation_counts["forbidden"] += 1
                            self.last_reject_forbidden_count += 1
                            if not self.first_predicted_forbidden_reason:
                                self.first_predicted_forbidden_reason = reason
                            continue
                        if float(summary.get("phi_max", 0.0)) > interest_rho:
                            violation_counts["interest_point"] += 1
                            self.last_reject_interest_phi_count += 1
                            continue
                    r = ref[min(k, ref.shape[0] - 1)]
                    track = self.lam_track * float(
                        np.sum((x_next[:2] - r[:2]) ** 2))
                    heading_target = ref[min(k + 1, ref.shape[0] - 1)]
                    heading_tracking = self.lam_heading_stage * float(
                        self._goal_heading_error(x_next, heading_target) ** 2)
                    social_weight = self.lam_social
                    if dist0 < self.near_goal_radius:
                        social_weight *= self.near_goal_social_scale
                    social = social_weight * float(
                        manifold_state.get("risk", 0.0))
                    tube = 0.0
                    if corridor is not None:
                        d = corridor_projection_cache.get(position_key)
                        if d is None:
                            _, d = corridor.project(np.array([
                                x_next[0], x_next[1], 0.0]))
                            corridor_projection_cache[position_key] = float(d)
                        tube_violation = max(
                            0.0, float(d) - float(corridor.radius))
                        if tube_violation > 1e-9 and tube_mode == "hard":
                            violation_counts["trajectory_tube"] += 1
                            continue
                        tube = self.lam_tube * float(tube_violation ** 2)
                    control = self.lam_u * float(np.dot(u, u))
                    smooth = self.lam_du * float(np.sum((u - previous) ** 2))
                    parts["tracking"] += track
                    parts["heading_tracking"] += heading_tracking
                    parts["social"] += social + step_soft_cost
                    parts["tube"] += tube
                    parts["control"] += control
                    parts["smooth"] += smooth
                    expanded.append({
                        "cost": float(item["cost"] + track + heading_tracking + social +
                                      step_soft_cost + tube + control + smooth),
                        "state": x_next,
                        "controls": item["controls"] + [u.copy()],
                        "states": item["states"] + [x_next.copy()],
                        "parts": parts,
                    })
            if not expanded:
                self.last_solver_status = "safe_stop: no_feasible_sequence"
                self.last_constraint_violation = violation_counts
                return 0.0, 0.0
            expanded.sort(key=lambda value: value["cost"])
            beam = expanded[:self.beam_width]

        records = []
        for item in beam:
            x = np.asarray(item["state"], float)
            controls = list(item["controls"])
            parts = dict(item["parts"])
            u_terminal = controls[-1] if controls else np.asarray(u_prev, float)
            terminal_adp = 0.0
            if has_adp_terminal:
                features = feature_builder.build_wheelchair(
                    x, goal, field, gate_info=gate_info,
                    interest_risk=interest_risk, corridor=corridor,
                    u=u_terminal)
                terminal_adp = max(0.0, critic.predict(features))
            distN = float(np.linalg.norm(x[:2] - goal[:2]))
            progress = dist0 - distN
            initial_heading_error = abs(self._goal_heading_error(x0, ref[0]))
            final_heading_error = abs(self._goal_heading_error(x, ref[0]))
            heading_improvement = initial_heading_error - final_heading_error
            ref_goal = ref[min(horizon - 1, ref.shape[0] - 1)]
            ref_progress = float(
                np.linalg.norm(x0[:2] - ref_goal[:2]) -
                np.linalg.norm(x[:2] - ref_goal[:2]))
            first_state = (
                np.asarray(item["states"][0], float)
                if item["states"] else np.asarray(x0, float))
            first_u = (
                np.asarray(controls[0], float)
                if controls else np.asarray(u_prev, float))
            first_goal_progress = float(
                dist0 - np.linalg.norm(first_state[:2] - goal[:2]))
            first_ref_progress = float(
                np.linalg.norm(x0[:2] - ref_goal[:2]) -
                np.linalg.norm(first_state[:2] - ref_goal[:2]))
            first_heading_error = abs(self._goal_heading_error(
                first_state, ref[0]))
            first_heading_improvement = (
                initial_heading_error - first_heading_error)
            first_positive_progress = max(
                first_goal_progress, first_ref_progress)
            progress_reward = self.lam_progress * max(0.0, progress)
            ref_progress_reward = self.lam_ref_progress * max(0.0, ref_progress)
            speed_reward = self.lam_speed * sum(
                max(0.0, float(u[0])) for u in controls) * self.dt
            sequence_translation = sum(
                max(0.0, float(u[0])) for u in controls) * self.dt
            terminal_goal_cost = self.lam_goal_terminal * float(distN ** 2)
            if dist0 < self.near_goal_radius:
                terminal_goal_cost += self.near_goal_goal_weight * float(distN ** 2)
            stall_cost = 0.0
            if dist0 > 0.12 and progress < self.min_progress_per_solve:
                stall_cost = self.lam_stall * float(
                    dist0 - min(progress, 0.0))
            adp_scale = (
                self.near_goal_adp_scale
                if dist0 < self.near_goal_radius else 1.0)
            heading_cost = 0.0
            if dist0 < self.final_approach_radius:
                heading_err = self._goal_heading_error(x, goal)
                heading_cost = self.lam_heading * float(heading_err ** 2)
            total = (
                float(item["cost"]) + terminal_goal_cost + stall_cost +
                heading_cost + float(lambda_adp_terminal) * adp_scale *
                terminal_adp - progress_reward - ref_progress_reward -
                speed_reward)
            objective = dict(parts)
            objective.update({
                "terminal_goal": terminal_goal_cost,
                "stall": stall_cost,
                "heading": heading_cost,
                "terminal_adp": float(lambda_adp_terminal) * adp_scale * terminal_adp,
                "progress_reward": -progress_reward,
                "reference_progress_reward": -ref_progress_reward,
                "speed_reward": -speed_reward,
                "alignment_translation": float(sequence_translation),
                "reference_progress": float(ref_progress),
                "first_step_goal_progress": float(first_goal_progress),
                "first_step_reference_progress": float(first_ref_progress),
                "first_step_positive_progress": float(first_positive_progress),
                "first_step_heading_improvement": float(first_heading_improvement),
                "first_step_angular_speed": abs(float(first_u[1])),
            })
            records.append((total, progress, heading_improvement, distN,
                            sequence_translation, ref_progress,
                            first_goal_progress, first_ref_progress,
                            item, terminal_adp, objective,
                            first_positive_progress,
                            first_heading_improvement,
                            abs(float(first_u[1]))))

        min_alignment_translation = max(0.005, self.min_progress_per_solve)
        min_rollout_progress = 0.5 * float(self.min_progress_per_solve)
        min_first_step_progress = max(
            0.001,
            float(self.first_step_progress_ratio) *
            float(self.min_progress_per_solve))
        max_heading_recovery_w = min(
            float(self.w_max), max(0.0, float(self.heading_recovery_w_max)))
        max_heading_recovery_backtrack = max(
            0.02, 4.0 * float(self.min_progress_per_solve))
        # A liveness gate needs a positive command now, but must not require
        # the exact maximum acceleration step.  The latter rejects otherwise
        # safe sequences under tiny state or floating-point differences.
        valid = []
        for item in records:
            first_u = np.asarray(item[8]["controls"][0], float)
            first_speed_ok = (
                float(first_u[0]) + 1e-9 >= min_alignment_first_v)
            first_progress_ok = (
                float(item[11]) + 1e-9 >= min_first_step_progress)
            rollout_progress_ok = (
                item[1] + 1e-9 >= self.min_progress_per_solve or
                item[5] + 1e-9 >= self.min_progress_per_solve)
            first_step_live = bool(
                first_speed_ok and
                (first_progress_ok or rollout_progress_ok))
            heading_recovery_live = bool(
                first_speed_ok and
                item[2] + 1e-9 >= self.min_heading_improvement and
                item[13] <= max_heading_recovery_w + 1e-9 and
                item[4] + 1e-9 >= min_alignment_translation and
                item[1] + max_heading_recovery_backtrack >= 0.0 and
                item[5] + max_heading_recovery_backtrack >= 0.0)
            if dist0 < 0.12 or first_step_live or heading_recovery_live:
                objective = dict(item[10])
                objective["first_step_live"] = bool(first_step_live)
                objective["heading_recovery_live"] = bool(heading_recovery_live)
                item = (
                    item[0], item[1], item[2], item[3], item[4],
                    item[5], item[6], item[7], item[8], item[9],
                    objective, item[11], item[12], item[13])
                valid.append(item)
        if not valid:
            self.last_solver_status = "safe_stop: insufficient_progress"
            violation_counts["insufficient_progress"] = int(len(records))
            violation_counts["nonprogressive_rollout"] = int(len(records))
            self.last_constraint_violation = violation_counts
            self.last_objective_terms = {
                "required_first_speed": float(min_alignment_first_v),
                "required_first_step_progress": float(
                    min_first_step_progress),
                "required_sequence_progress": float(
                    self.min_progress_per_solve),
                "required_heading_improvement": float(
                    self.min_heading_improvement),
                "max_heading_recovery_w": float(max_heading_recovery_w),
                "required_alignment_translation": float(
                    min_alignment_translation),
                "required_rollout_progress": float(min_rollout_progress),
                "max_heading_recovery_backtrack": float(
                    max_heading_recovery_backtrack),
                "best_sequence_progress": float(max(
                    [item[1] for item in records] or [0.0])),
                "best_reference_progress": float(max(
                    [item[5] for item in records] or [0.0])),
                "best_first_step_goal_progress": float(max(
                    [item[6] for item in records] or [0.0])),
                "best_first_step_reference_progress": float(max(
                    [item[7] for item in records] or [0.0])),
                "best_first_step_positive_progress": float(max(
                    [item[11] for item in records] or [0.0])),
                "best_first_step_heading_improvement": float(max(
                    [item[12] for item in records] or [0.0])),
                "best_first_step_angular_speed": float(min(
                    [item[13] for item in records] or [0.0])),
                "best_heading_improvement": float(max(
                    [item[2] for item in records] or [0.0])),
                "best_alignment_translation": float(max(
                    [item[4] for item in records] or [0.0])),
                "best_first_speed": float(max(
                    [item[8]["controls"][0][0] for item in records] or [0.0])),
            }
            return 0.0, 0.0
        best = min(valid, key=lambda value: value[0])
        (best_cost, progress, heading_improvement, _distN,
         alignment_translation, ref_progress, first_goal_progress,
         first_ref_progress, best_item, terminal_adp, objective,
         first_positive_progress, first_heading_improvement,
         first_angular_speed) = best
        controls = list(best_item["controls"])
        states = list(best_item["states"])
        best_u = controls[0]
        self.last_terminal_adp_cost = float(terminal_adp)
        self.last_track_cost = float(objective.get("tracking", 0.0))
        self.last_social_cost = float(objective.get("social", 0.0))
        self.last_tube_cost = float(objective.get("tube", 0.0))
        self.last_control_cost = float(
            objective.get("control", 0.0) + objective.get("smooth", 0.0))
        self.last_total_cost = float(best_cost)
        self.last_predicted_controls = [
            np.asarray(u, float).tolist() for u in controls]
        self.last_predicted_states = [
            np.asarray(x, float).tolist() for x in states]
        self.last_objective_terms = dict(objective)
        self.last_objective_terms["required_first_speed"] = float(
            min_alignment_first_v)
        self.last_objective_terms["required_sequence_progress"] = float(
            self.min_progress_per_solve)
        self.last_objective_terms["sequence_progress"] = float(progress)
        self.last_objective_terms["heading_improvement"] = float(
            heading_improvement)
        self.last_objective_terms["alignment_translation"] = float(
            alignment_translation)
        self.last_objective_terms["reference_progress"] = float(ref_progress)
        self.last_objective_terms["first_step_goal_progress"] = float(
            first_goal_progress)
        self.last_objective_terms["first_step_reference_progress"] = float(
            first_ref_progress)
        self.last_objective_terms["first_step_positive_progress"] = float(
            first_positive_progress)
        self.last_objective_terms["first_step_heading_improvement"] = float(
            first_heading_improvement)
        self.last_objective_terms["first_step_angular_speed"] = float(
            first_angular_speed)
        self.last_objective_terms["required_first_step_progress"] = float(
            min_first_step_progress)
        self.last_objective_terms["max_heading_recovery_w"] = float(
            max_heading_recovery_w)
        self.last_objective_terms["required_rollout_progress"] = float(
            min_rollout_progress)
        self.last_constraint_violation = violation_counts
        self.last_control_sequence_varies = bool(any(
            not np.allclose(controls[0], item) for item in controls[1:]))
        self.last_sequence_progress = float(progress)
        self.last_heading_improvement = float(heading_improvement)
        self.last_alignment_translation = float(alignment_translation)
        if self.last_final_approach_used and has_adp_terminal:
            self.last_solver_status = "predictive_beam_adp_terminal_final"
        elif self.last_final_approach_used:
            self.last_solver_status = "predictive_beam_final_approach"
        elif has_adp_terminal:
            self.last_solver_status = "predictive_beam_adp_terminal"
        else:
            self.last_solver_status = "predictive_beam"
        return float(best_u[0]), float(best_u[1])

    def _pure_pursuit_u(self, x0, ref, field, u_prev):
        look = ref[min(3, ref.shape[0] - 1)]
        desired = np.arctan2(look[1] - x0[1], look[0] - x0[0])
        herr = np.arctan2(np.sin(desired - x0[2]), np.cos(desired - x0[2]))
        w = float(np.clip(2.0 * herr, -self.w_max, self.w_max))
        dist = float(np.linalg.norm(look - x0[:2]))
        align = max(0.0, np.cos(herr))
        v = self.v_max * align * np.clip(dist / 0.6, 0.0, 1.0)
        risk = field.phi_s(np.array([x0[0], x0[1], 0.0]))
        slow = 1.0 / (1.0 + self.lam_social * min(risk, 3.0))
        v *= max(slow, 0.35)
        v = float(np.clip(v, u_prev[0] - self.a_max * self.dt,
                          u_prev[0] + self.a_max * self.dt))
        v = float(np.clip(v, 0.0, self.v_max))
        return np.array([v, w], float)

    def _heading_ctrl(self, x0, ref):
        target = ref[min(1, ref.shape[0] - 1)]
        desired = np.arctan2(target[1] - x0[1], target[0] - x0[0])
        err = np.arctan2(np.sin(desired - x0[2]), np.cos(desired - x0[2]))
        return float(np.clip(1.5 * err, -self.w_max, self.w_max))

    def _greedy(self, x0, ref, field, corridor):
        target = ref[min(1, ref.shape[0] - 1)]
        dist = np.linalg.norm(target - x0[:2])
        w = self._heading_ctrl(x0, ref)
        v = float(np.clip(0.8 * dist, 0.0, self.v_max))

        risk = field.phi_s(np.array([x0[0], x0[1], 0.0]))
        v *= 1.0 / (1.0 + 0.6 * risk)
        return v, w
