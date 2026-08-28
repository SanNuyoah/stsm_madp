import sys
sys.dont_write_bytecode = True

import time

import numpy as np


def node_semantic_type(profile, node_kind, node_id="", task_type=""):
    profile = str(profile or "generic")
    kind = str(node_kind or "")
    task_type = str(task_type or "")
    if kind == "saddle":
        return "bypass" if profile == "wheelchair" else "passage"
    if kind == "minimum":
        if task_type:
            return task_type
        if profile == "arm":
            return "handover"
        if profile == "wheelchair":
            text = str(node_id or "")
            return "waiting" if "waiting" in text else "parking"
        return "stable_safe_region"
    if kind in ("start", "goal"):
        return kind
    return kind or "unknown"


def semantic_sequence(corridor):
    explicit = list(getattr(corridor, "topology_semantics", []))
    if explicit:
        return explicit
    node_types = list(getattr(corridor, "node_type_sequence", []))
    semantics = list(getattr(corridor, "topology_semantic_kinds", []))
    if not node_types:
        return []
    out = []
    middle = list(semantics)
    for idx, kind in enumerate(node_types):
        if idx == 0:
            out.append("start")
        elif idx == len(node_types) - 1:
            out.append("goal")
        else:
            out.append(middle[idx - 1] if idx - 1 < len(middle) else str(kind))
    return out


def task_semantic_class(corridor, profile):
    profile = str(profile or "generic")
    semantics = semantic_sequence(corridor)
    if profile == "arm":
        return "handover_approach" if "handover" in semantics else "arm_task_missing"
    if profile == "wheelchair":
        if "parking" in semantics:
            return "parking_approach"
        if "waiting" in semantics:
            return "waiting_approach"
        return "wheelchair_task_missing"
    return "generic_task"


def topology_route_class(corridor, start=None, goal=None):
    waypoints = np.asarray(getattr(corridor, "topology_ordered_waypoints",
                                   getattr(corridor, "waypoints", [])), float)
    if len(waypoints) < 3:
        return "direct_safe_channel"
    if start is None:
        start = waypoints[0]
    if goal is None:
        goal = waypoints[-1]
    start = np.asarray(start, float)
    goal = np.asarray(goal, float)
    axis = goal[:2] - start[:2]
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9:
        return "center_passage"
    side = np.array([-axis[1], axis[0]], float) / norm
    offsets = []
    for point in waypoints[1:-1]:
        rel = np.asarray(point, float)[:2] - start[:2]
        offsets.append(float(np.dot(rel, side)))
    if not offsets:
        return "center_passage"
    mean_offset = float(np.mean(offsets))
    max_abs = float(np.max(np.abs(offsets)))
    threshold = max(0.03, 0.10 * norm)
    if max_abs <= threshold:
        return "center_passage"
    vertical = abs(axis[1]) > abs(axis[0])
    if vertical:
        return "right_bypass" if mean_offset > 0.0 else "left_bypass"
    return "left_bypass" if mean_offset > 0.0 else "right_bypass"


def build_task_context(profile, task_state, state_trigger,
                       progress=None, dist_to_goal=None,
                       risk_ahead=None, near_narrow_passage=False,
                       near_critical_point=False,
                       interaction_target=None):
    """Return the lightweight, serializable context shared with risk models."""
    def _optional_float(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    return {
        "profile": str(profile or "generic").strip().lower(),
        "task_state": str(task_state or "generic").strip().lower(),
        "state_trigger": str(state_trigger or "default_motion"),
        "progress": _optional_float(progress),
        "dist_to_goal": _optional_float(dist_to_goal),
        "risk_ahead": _optional_float(risk_ahead),
        "near_narrow_passage": bool(near_narrow_passage),
        "near_critical_point": bool(near_critical_point),
        "interaction_target": interaction_target,
    }


def infer_task_context(profile, task_mode=None, phase=None, progress=0.0,
                       context=None, config=None):
    """Infer task semantics with event evidence ahead of progress fallback."""
    profile = str(profile or "generic").strip().lower()
    mode = str(task_mode or "").strip().lower()
    phase = str(phase or "").strip().lower()
    context = dict(context or {})
    config = dict(config or {})
    try:
        progress = float(progress or 0.0)
    except (TypeError, ValueError):
        progress = 0.0
    progress = float(np.clip(progress, 0.0, 1.0))
    trigger = "default_motion"
    if profile == "arm":
        if phase == "return" or progress >= 0.80:
            state = "return"
        elif phase == "hold" or 0.60 <= progress < 0.80:
            state = "hold"
        elif phase == "handover" or 0.40 <= progress < 0.60:
            state = "handover"
        elif phase == "align" or 0.20 <= progress < 0.40:
            state = "align"
        else:
            state = "approach"
        if mode == "holding":
            state = "hold"
        trigger = "explicit_phase" if phase else "progress_fallback"
    elif profile == "wheelchair":
        wheelchair_cfg = dict(config.get("wheelchair", config) or {})
        explicit_states = set((
            "moving", "avoiding", "passing", "arriving", "arrival",
            "goal_reaching", "narrow_passage", "passage",
            "obstacle_avoidance", "avoidance"))
        arriving_radius = float(wheelchair_cfg.get("arriving_radius", 0.80))
        avoiding_threshold = float(wheelchair_cfg.get(
            "avoiding_risk_threshold", 1.6))
        risk_ahead = context.get("risk_ahead", None)
        dist_to_goal = context.get("dist_to_goal", None)
        near_narrow = bool(context.get("near_narrow_passage", False))
        near_critical = bool(context.get("near_critical_point", False))
        try:
            risk_ahead = float(risk_ahead)
        except (TypeError, ValueError):
            risk_ahead = None
        try:
            dist_to_goal = float(dist_to_goal)
        except (TypeError, ValueError):
            dist_to_goal = None

        if phase in explicit_states:
            state = {
                "arrival": "arriving", "goal_reaching": "arriving",
                "narrow_passage": "passing", "passage": "passing",
                "obstacle_avoidance": "avoiding", "avoidance": "avoiding",
            }.get(phase, phase)
            trigger = "explicit_phase"
        elif risk_ahead is not None and np.isfinite(risk_ahead) and \
                risk_ahead >= avoiding_threshold:
            state = "avoiding"
            trigger = "social_risk_ahead"
        elif near_narrow or near_critical:
            state = "passing"
            trigger = ("narrow_passage" if near_narrow else
                       "topology_critical_segment")
        elif dist_to_goal is not None and np.isfinite(dist_to_goal) and \
                dist_to_goal < arriving_radius:
            state = "arriving"
            trigger = "goal_proximity"
        elif bool(wheelchair_cfg.get("progress_fallback_enabled", True)):
            if progress >= 0.75:
                state = "arriving"
            elif progress >= 0.50:
                state = "passing"
            elif progress >= 0.25:
                state = "avoiding"
            else:
                state = "moving"
            trigger = "progress_fallback"
        else:
            state = "moving"
            trigger = "default_motion"
    else:
        state = mode or "generic"
        trigger = "explicit_mode" if mode else "default"
    current_phase = phase or (
        "navigation" if profile == "wheelchair" else
        "return" if state == "return" else
        "handover" if state in ("handover", "hold") else
        "approach")
    result = {
        "task_mode": mode or ("handover" if profile == "arm" else "navigation"),
        "task_state": state,
        "phase": current_phase,
        "current_phase": current_phase,
        "progress": float(progress),
        "state_transition": str(context.get(
            "state_transition",
            "{}->{}".format(mode or "task", state))),
        "timestamp": float(context.get("timestamp", time.time())),
    }
    result.update(build_task_context(
        profile, state, trigger, progress=progress,
        dist_to_goal=context.get("dist_to_goal", None),
        risk_ahead=context.get("risk_ahead", None),
        near_narrow_passage=context.get("near_narrow_passage", False),
        near_critical_point=context.get("near_critical_point", False),
        interaction_target=context.get("interaction_target", None)))
    return result


def infer_task_state(profile, task_mode=None, phase=None, progress=0.0,
                     context=None, config=None):
    """Backward-compatible task-state entry point."""
    return infer_task_context(
        profile, task_mode=task_mode, phase=phase, progress=progress,
        context=context, config=config)


def evaluate_task_cost_breakdown(corridor, profile, start=None, goal=None,
                                 task_mode=None, task_state=None):
    profile = str(profile or "generic")
    mode = str(task_mode or "").strip().lower()
    if not mode:
        mode = "handover" if profile == "arm" else "navigation"
    waypoints = np.asarray(getattr(corridor, "waypoints", []), float)
    semantics = semantic_sequence(corridor)
    terms = {}

    state_info = infer_task_state(profile, mode, task_state or "", 0.0)
    state = str(task_state or state_info.get("task_state", ""))

    if mode == "handover":
        if "handover" not in semantics:
            terms["semantic_handover_missing_cost"] = 3.0
        route_class = str(topology_route_class(corridor, start=start, goal=goal))
        if route_class == "center_passage":
            terms["center_passage_penalty"] = 1.0
        distance_cost = 0.0
        orientation_cost = float(getattr(corridor, "orientation_error", 0.0))
        interaction_cost = 0.0
        if len(waypoints) >= 2 and goal is not None:
            final_vec = waypoints[-1, :2] - waypoints[-2, :2]
            goal_vec = np.asarray(goal, float)[:2] - waypoints[-2, :2]
            denom = float(np.linalg.norm(final_vec) * np.linalg.norm(goal_vec))
            if denom > 1e-9:
                align = float(np.dot(final_vec, goal_vec) / denom)
                terms["approach_direction_cost"] = max(0.0, 1.0 - align)
            goal3 = np.asarray(goal, float)[:min(3, waypoints.shape[1])]
            end3 = waypoints[-1, :goal3.shape[0]]
            distance_cost = float(np.linalg.norm(end3 - goal3))
        ik_validation = getattr(corridor, "ik_validation", {}) or {}
        ik_count = float(ik_validation.get("arm_ik_candidate_count", 0) or 0)
        ik_attempts = list(ik_validation.get("arm_ik_candidate_attempts", []) or [])
        ik_den = float(max(len(ik_attempts), ik_count, 1.0))
        ik_success_rate = 1.0 if bool(getattr(corridor, "ik_valid", True)) else 0.0
        if ik_count > 0 or ik_attempts:
            ik_success_rate = float(np.clip(ik_count / ik_den, 0.0, 1.0))
        ik_cost = float(1.0 - ik_success_rate)
        interaction_cost = float(max(
            0.0, getattr(corridor, "trajectory_max_risk", 0.0) -
            getattr(corridor, "risk_threshold", 6.0))) / 6.0
        state_weights = {
            "approach": (1.2, 0.6, 1.0, 0.7),
            "align": (0.8, 1.5, 1.0, 0.9),
            "handover": (1.4, 1.2, 1.2, 1.6),
            "hold": (0.7, 1.0, 0.8, 1.8),
            "return": (0.5, 0.6, 1.0, 0.7),
        }
        w_dist, w_orient, w_ik, w_interact = state_weights.get(
            state, (1.0, 1.0, 1.0, 1.0))
        terms.update({
            "distance_cost": float(w_dist * distance_cost),
            "handover_distance_cost": float(w_dist * distance_cost),
            "orientation_cost": float(w_orient * orientation_cost),
            "orientation_match_cost": float(w_orient * orientation_cost),
            "feasibility_cost": float(w_ik * ik_cost),
            "ik_feasibility_cost": float(w_ik * ik_cost),
            "interaction_cost": float(w_interact * interaction_cost),
            "interaction_region_cost": float(w_interact * interaction_cost),
        })
    elif mode == "navigation":
        if "parking" not in semantics and "waiting" not in semantics:
            terms["semantic_goal_missing_cost"] = 3.0
        route_class = str(topology_route_class(corridor, start=start, goal=goal))
        if route_class == "center_passage":
            terms["center_passage_penalty"] = 0.5
        goal_alignment_cost = 0.0
        goal_distance_cost = 0.0
        if len(waypoints) >= 2 and goal is not None:
            final_dist = float(np.linalg.norm(
                waypoints[-1, :2] - np.asarray(goal, float)[:2]))
            goal_distance_cost = 2.0 * final_dist
            final_vec = waypoints[-1, :2] - waypoints[-2, :2]
            goal_vec = np.asarray(goal, float)[:2] - waypoints[-2, :2]
            denom = float(np.linalg.norm(final_vec) * np.linalg.norm(goal_vec))
            goal_alignment_cost = (
                max(0.0, 1.0 - float(np.dot(final_vec, goal_vec) / denom))
                if denom > 1e-9 else 0.0)
        widths = list(getattr(corridor, "corridor_width_profile", []) or [])
        width_min = float(np.min(np.asarray(widths, float))) if widths else 0.0
        passage_width_cost = min(1.0, max(0.0, 0.20 - width_min) / 0.20)
        turning_cost = min(1.0, float(getattr(
            corridor, "max_turn_angle", 0.0)) / max(float(np.pi), 1e-6))
        goal_stability_cost = min(
            1.0, float(getattr(corridor, "curvature_violation", 0.0)) / 20.0)
        state_weights = {
            "moving": (1.2, 0.8, 0.8, 0.8),
            "avoiding": (0.8, 1.5, 1.3, 0.8),
            "passing": (1.0, 1.6, 1.1, 0.9),
            "arriving": (1.5, 0.8, 0.8, 1.6),
        }
        w_goal, w_width, w_turn, w_stable = state_weights.get(
            state, (1.0, 1.0, 1.0, 1.0))
        terms.update({
            "distance_cost": float(w_goal * goal_distance_cost),
            "goal_region_distance_cost": float(w_goal * goal_distance_cost),
            "goal_alignment_cost": float(w_goal * goal_alignment_cost),
            "goal_direction_cost": float(w_goal * goal_alignment_cost),
            "passage_width_cost": float(w_width * passage_width_cost),
            "turning_cost": float(w_turn * turning_cost),
            "turning_complexity_cost": float(w_turn * turning_cost),
            "goal_stability_cost": float(w_stable * goal_stability_cost),
            "arrival_stability_cost": float(w_stable * goal_stability_cost),
        })
    else:
        terms["semantic_missing_cost"] = 0.5 if not semantics else 0.0
    alias_terms = set([
        "handover_distance_cost",
        "orientation_match_cost",
        "ik_feasibility_cost",
        "interaction_region_cost",
        "goal_region_distance_cost",
        "goal_direction_cost",
        "turning_complexity_cost",
        "arrival_stability_cost",
    ])
    cost = float(sum(
        float(v) for k, v in terms.items() if k not in alias_terms))
    return {
        "task_mode": mode,
        "task_state": state,
        "task_cost": float(cost),
        "terms": dict((k, float(v)) for k, v in terms.items()),
        "semantic_sequence": semantics,
        "topology_route_class": str(topology_route_class(
            corridor, start=start, goal=goal)),
    }


def evaluate_task_cost(corridor, profile, start=None, goal=None, task_mode=None):
    return float(evaluate_task_cost_breakdown(
        corridor, profile, start=start, goal=goal,
        task_mode=task_mode).get("task_cost", 0.0))
