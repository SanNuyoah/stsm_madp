import sys
sys.dont_write_bytecode = True

import heapq
import math
import time

import numpy as np

from stsm_madp.manifold import Corridor
from stsm_madp.topology_candidate_generator import (
    TopologyDrivenCandidateGenerator,
    evaluate_candidate,
    evaluate_candidate_manifold_feasibility,
)
from stsm_madp.candidate_ranker import (
    candidate_topology_identity,
    rank_feasible_candidates,
)
from stsm_madp.topology_components import default_topology_components
from stsm_madp.candidate_recovery import recover_candidates
from stsm_madp.arm_topology_validator import ArmTopologyValidator
from stsm_madp.topology_ik_solver import TopologyIKSolver
from stsm_madp.topology_refinement import refine_topology_path
from stsm_madp.safety_evaluator import (
    build_safety_context, safety_context_audit)
from stsm_madp.interest_points import forbidden_anchor_hit, pose_interest_risk
from stsm_madp.task_semantics import (
    evaluate_task_cost,
    evaluate_task_cost_breakdown,
    infer_task_state,
    node_semantic_type,
    semantic_sequence,
    task_semantic_class,
    topology_route_class,
)
from stsm_madp.task_config import (
    resolve_task_mode,
    resolve_task_weight,
    weighted_task_candidate_cost,
)
from stsm_madp.semantic_topology_node import semantic_topology_specs


ARM_RECOVERABLE_FAILURE_REASONS = (
    "end_effector_clearance_violation",
    "orientation_error",
)


def _safe_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(value):
        return float(default)
    return float(value)


def _failure_tokens(value):
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            out.extend(_failure_tokens(item))
        return out
    return [
        token.strip()
        for token in str(value or "").replace(";", ",").split(",")
        if token.strip()
    ]


def _arm_candidate_filter_classification(reason, validation=None,
                                         feasibility=None):
    validation = dict(validation or {})
    feasibility = dict(feasibility or {})
    reasons = set(_failure_tokens(reason))
    reasons.update(_failure_tokens(validation.get("failure_reason", "")))
    reasons.update(_failure_tokens(feasibility.get("failure_reason", "")))
    collision_reasons = set([
        "end_effector_collision",
        "link_collision",
        "arm_link_collision",
        "self_collision",
        "collision",
    ])
    if reasons.intersection(collision_reasons):
        return "invalid"
    if any(token in reasons for token in ARM_RECOVERABLE_FAILURE_REASONS):
        return "recoverable"
    if "orientation_error" in validation or "orientation_error" in feasibility:
        return "recoverable"
    if not reasons:
        return "safe"
    return "invalid"


def _arm_recovery_cost(reason, validation=None, feasibility=None):
    validation = dict(validation or {})
    feasibility = dict(feasibility or {})
    required = max(
        _safe_float(validation.get("required_clearance", 0.0), 0.0),
        _safe_float(feasibility.get("required_clearance", 0.0), 0.0),
        _safe_float(feasibility.get("minimum_clearance", 0.0), 0.0))
    clearance = max(
        _safe_float(validation.get("min_end_effector_clearance", 0.0), 0.0),
        _safe_float(feasibility.get(
            "trajectory_min_clearance",
            feasibility.get("min_clearance", 0.0)), 0.0))
    clearance_deficit = max(0.0, required - clearance)
    orientation_error = max(
        _safe_float(validation.get("orientation_error", 0.0), 0.0),
        _safe_float(feasibility.get("orientation_error", 0.0), 0.0))
    risk_threshold = max(
        _safe_float(validation.get("risk_threshold", 0.0), 0.0),
        _safe_float(feasibility.get("risk_threshold", 0.0), 0.0),
        1e-6)
    risk = max(
        _safe_float(validation.get("max_risk", 0.0), 0.0),
        _safe_float(feasibility.get(
            "trajectory_max_risk",
            feasibility.get("max_risk", 0.0)), 0.0))
    risk_excess = max(0.0, risk - risk_threshold)
    base = 0.0 if _arm_candidate_filter_classification(
        reason, validation, feasibility) == "safe" else 1.0
    return float(
        base +
        20.0 * clearance_deficit +
        2.0 * orientation_error +
        5.0 * risk_excess / risk_threshold)


def _arm_recoverable_level(reason="", validation=None, feasibility=None,
                           cost=None):
    validation = dict(validation or {})
    feasibility = dict(feasibility or {})
    reasons = set(_failure_tokens(reason))
    reasons.update(_failure_tokens(validation.get("failure_reason", "")))
    reasons.update(_failure_tokens(feasibility.get("failure_reason", "")))
    level3 = set([
        "link_collision",
        "arm_link_collision",
        "self_collision",
        "collision",
        "ik_failed",
        "ik_failure",
        "ik_or_link_collision",
        "workspace_violation",
        "workspace_exceeded",
        "workspace_out_of_bounds",
    ])
    if reasons.intersection(level3):
        return "level3"
    if any("workspace" in token and "violation" in token for token in reasons):
        return "level3"
    if any(token in reasons for token in (
            "pose_adjustment",
            "pose_adjustment_required",
            "pose_optimization_required",
            "position_error")):
        return "level2"
    orientation_error = max(
        _safe_float(validation.get("orientation_error", 0.0), 0.0),
        _safe_float(feasibility.get("orientation_error", 0.0), 0.0))
    if ("end_effector_clearance_violation" in reasons or
            ("orientation_error" in reasons and orientation_error <= 0.35)):
        return "level1"
    if "orientation_error" in reasons:
        return "level2"
    if cost is None and not reasons:
        try:
            cost = float(reason)
        except (TypeError, ValueError):
            cost = 0.0
    cost = _safe_float(cost, 0.0)
    if cost <= 0.0:
        return "none"
    if cost <= 0.35:
        return "level1"
    if cost <= 0.70:
        return "level2"
    return "level3"


TOPOLOGY_PROFILES = {
    "generic": {
        "motion_space": "planar",
        "workspace_dimension": 2,
        "robot_scale": 0.45,
        "grid_resolution": 0.10,
        "grid_scale": 0.22,
        "merge_radius": 0.20,
        "merge_grid_factor": 2.0,
        "min_clearance": 0.15,
        "clearance_scale": 0.33,
        "hard_clearance": 0.06,
        "hard_clearance_scale": 0.40,
        "neighbor_k": 8,
        "neighbor_strategy": "balanced",
        "max_graph_nodes": 40,
        "max_saddle_nodes": 8,
        "max_minima_nodes": 4,
        "max_semantic_nodes": 8,
        "max_ring_nodes": 8,
        "morse_w_goal": 0.35,
        "morse_w_social": 1.0,
        "morse_w_barrier": 0.6,
        "morse_core_required": True,
        "lambda_tracking": 0.15,
        "lambda_saddle_value": 0.20,
        "max_tracking_turn": 1.35,
        "nominal_speed": 0.30,
        "max_curvature": 4.0,
        "candidate_max_risk": None,
    },
    "wheelchair": {
        "motion_space": "planar_body",
        "workspace_dimension": 2,
        "robot_scale": 0.65,
        "grid_resolution": 0.15,
        "grid_scale": 0.23,
        "merge_radius": 0.20,
        "merge_grid_factor": 1.35,
        "min_clearance": 0.15,
        "clearance_scale": 0.23,
        "hard_clearance": 0.06,
        "hard_clearance_scale": 0.40,
        "neighbor_k": 8,
        "neighbor_strategy": "sparse_body",
        "max_graph_nodes": 32,
        "max_saddle_nodes": 8,
        "max_minima_nodes": 4,
        "max_semantic_nodes": 4,
        "max_ring_nodes": 6,
        "morse_w_goal": 0.35,
        "morse_w_social": 1.0,
        "morse_w_barrier": 0.6,
        "morse_core_required": True,
        "lambda_tracking": 0.35,
        "lambda_saddle_value": 0.35,
        "max_tracking_turn": 1.05,
        "nominal_speed": 0.35,
        "max_curvature": 3.0,
        "candidate_max_risk": 2.5,
    },
    "arm": {
        "motion_space": "projected_3d_handover",
        "workspace_dimension": 3,
        "robot_scale": 0.16,
        "grid_resolution": 0.04,
        "grid_scale": 0.25,
        "merge_radius": 0.12,
        "merge_grid_factor": 3.0,
        "min_clearance": 0.08,
        "clearance_scale": 0.50,
        "hard_clearance": 0.03,
        "hard_clearance_scale": 0.38,
        "neighbor_k": 12,
        "neighbor_strategy": "dense_projected_3d",
        "max_graph_nodes": 40,
        "max_saddle_nodes": 8,
        "max_minima_nodes": 4,
        "max_semantic_nodes": 8,
        "max_ring_nodes": 8,
        "morse_w_goal": 0.25,
        "morse_w_social": 1.20,
        "morse_w_barrier": 0.80,
        "morse_core_required": True,
        "lambda_tracking": 0.20,
        "lambda_saddle_value": 0.30,
        "max_tracking_turn": 1.25,
        "nominal_speed": 0.08,
        "max_curvature": 8.0,
        "candidate_max_risk": None,
    },
}


def topology_profile_defaults(profile):
    key = str(profile or "generic").strip().lower()
    if key in ("mechanical_arm", "manipulator", "handover"):
        key = "arm"
    if key not in TOPOLOGY_PROFILES:
        key = "generic"
    data = dict(TOPOLOGY_PROFILES[key])
    data["profile"] = key
    data.update(_derive_topology_profile_params(data))
    return data


def _derive_topology_profile_params(profile):
    scale = max(1e-6, float(profile.get("robot_scale", 0.45)))
    grid = float(profile.get(
        "grid_resolution", scale * float(profile.get("grid_scale", 0.22))))
    grid = max(0.01, grid)
    clearance = float(profile.get(
        "min_clearance", scale * float(profile.get("clearance_scale", 0.33))))
    hard = float(profile.get(
        "hard_clearance",
        clearance * float(profile.get("hard_clearance_scale", 0.40))))
    merge = float(profile.get(
        "merge_radius", grid * float(profile.get("merge_grid_factor", 2.0))))
    strategy = str(profile.get("neighbor_strategy", "balanced"))
    if "neighbor_k" in profile:
        neighbor_k = int(profile["neighbor_k"])
    elif strategy == "dense_projected_3d":
        neighbor_k = 12
    elif strategy == "sparse_body":
        neighbor_k = 8
    else:
        neighbor_k = 10
    return {
        "grid_resolution": grid,
        "merge_radius": max(grid, merge),
        "min_clearance": max(0.0, clearance),
        "hard_clearance": max(0.0, min(hard, clearance)),
        "neighbor_k": max(1, neighbor_k),
        "neighbor_strategy": strategy,
    }


def topology_param_or_auto(value):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("", "auto", "profile", "default"):
            return None
    return value


def _profile_value(profile, key, value):
    value = topology_param_or_auto(value)
    if value is not None:
        return value
    return profile[key]


class TopologyNode:

    def __init__(self, node_id, kind, point, ij=None, semantic_type=None):
        self.id = str(node_id)
        self.kind = str(kind)
        self.node_type = self.kind if self.kind != "minimum" else "minima"
        self.semantic_type = str(semantic_type or "")
        self.point = np.asarray(point, float)
        self.ij = ij


class TopologicalCorridorPlanner:

    def __init__(self, field, rho, bounds, grid_resolution=None,
                 merge_radius=None, min_clearance=None,
                 hard_clearance=None, neighbor_k=None,
                 max_graph_nodes=None, lam_s=1.0, eps_m=0.02,
                 z_height=0.0, to_world=None, to_plane=None,
                 interest_config=None, topology_profile="generic",
                 morse_w_goal=None, morse_w_social=None,
                 morse_w_barrier=None,
                 morse_texture_eps=0.01, morse_texture_freq=3.0,
                 morse_grad_eps=0.6, morse_degenerate_eps=1e-5,
                 max_saddle_nodes=None, max_minima_nodes=None,
                 max_semantic_nodes=None, max_ring_nodes=None,
                 lambda_morse_bonus=0.0, saddle_tie_ratio=0.05,
                 morse_priority_ratio=0.25,
                 morse_saddle_priority_ratio=0.50,
                 morse_mix_priority_ratio=0.50,
                 morse_minima_priority_ratio=0.25,
                 morse_primary=True, morse_core_required=None,
                 safety_regions=None,
                 allow_semantic_with_morse=False,
                 allow_semantic_topology_recovery=True,
                 allow_ring_with_morse=False,
                 allow_graph_fallback_with_morse=False,
                 lambda_execution=0.20, lambda_tracking=None,
                 lambda_saddle_value=None, dynamics_profile=None,
                 min_saddle_offset=0.0, goal_saddle_exclusion=0.0,
                 corridor_dedupe_distance=0.0, max_corridor_turn=None,
                 max_corridor_curvature=None, min_segment_length=0.0,
                 require_risk_improvement=False, candidate_max_risk=None,
                 morse_decision_mode="balanced", corridor_score_weights=None,
                 task_minima_points=None, planning_clearance_margin=None,
                 task_mode=None, task_config=None, task_weight=None,
                 manifold_constraint_mode="soft",
                 candidate_pool_min=None, route_max_paths=None,
                 route_max_routes=None):
        profile = topology_profile_defaults(topology_profile)
        self.components = default_topology_components()
        self.field = field
        self.rho = float(rho)
        self.bounds = bounds
        self.topology_profile = profile["profile"]
        self.motion_space = profile["motion_space"]
        self.workspace_dimension = int(profile["workspace_dimension"])
        self.neighbor_strategy = str(profile.get("neighbor_strategy", "balanced"))
        self.grid_resolution = float(
            _profile_value(profile, "grid_resolution", grid_resolution))
        self.merge_radius = float(
            _profile_value(profile, "merge_radius", merge_radius))
        self.min_clearance = float(
            _profile_value(profile, "min_clearance", min_clearance))
        hard_clearance = topology_param_or_auto(hard_clearance)
        if hard_clearance is None:
            hard_clearance = profile.get("hard_clearance")
        if hard_clearance is None:
            hard_clearance = min(0.08, max(0.0, 0.4 * self.min_clearance))
        self.hard_clearance = float(hard_clearance)
        self.neighbor_k = max(
            1, int(_profile_value(profile, "neighbor_k", neighbor_k)))
        self.max_graph_nodes = int(
            _profile_value(profile, "max_graph_nodes", max_graph_nodes))
        self.lam_s = float(lam_s)
        self.eps_m = float(eps_m)
        self.z_height = float(z_height)
        self.to_world = to_world
        self.to_plane = to_plane
        self.interest_config = interest_config or {}
        self.morse_w_goal = float(
            _profile_value(profile, "morse_w_goal", morse_w_goal))
        self.morse_w_social = float(
            _profile_value(profile, "morse_w_social", morse_w_social))
        self.morse_w_barrier = float(
            _profile_value(profile, "morse_w_barrier", morse_w_barrier))
        self.morse_texture_eps = float(morse_texture_eps)
        self.morse_texture_freq = float(morse_texture_freq)
        self.morse_grad_eps = float(morse_grad_eps)
        self.morse_degenerate_eps = float(morse_degenerate_eps)
        self.max_saddle_nodes = int(
            _profile_value(profile, "max_saddle_nodes", max_saddle_nodes))
        self.max_minima_nodes = int(
            _profile_value(profile, "max_minima_nodes", max_minima_nodes))
        self.max_semantic_nodes = int(
            _profile_value(profile, "max_semantic_nodes", max_semantic_nodes))
        self.max_ring_nodes = int(
            _profile_value(profile, "max_ring_nodes", max_ring_nodes))
        self.lambda_morse_bonus = float(lambda_morse_bonus)
        self.saddle_tie_ratio = max(0.0, float(saddle_tie_ratio))
        self.morse_priority_ratio = max(
            self.saddle_tie_ratio, float(morse_priority_ratio))
        self.morse_saddle_priority_ratio = max(
            self.morse_priority_ratio, float(morse_saddle_priority_ratio))
        self.morse_mix_priority_ratio = max(
            self.morse_priority_ratio, float(morse_mix_priority_ratio))
        self.morse_minima_priority_ratio = max(
            self.morse_priority_ratio, float(morse_minima_priority_ratio))
        self.morse_primary = bool(morse_primary)
        self.morse_core_required = bool(_profile_value(
            profile, "morse_core_required", morse_core_required))
        self.safety_regions = list(safety_regions or [])
        self.allow_semantic_with_morse = bool(allow_semantic_with_morse)
        self.allow_semantic_topology_recovery = bool(
            allow_semantic_topology_recovery)
        self.allow_ring_with_morse = bool(allow_ring_with_morse)
        self.allow_graph_fallback_with_morse = bool(
            allow_graph_fallback_with_morse)
        self.lambda_execution = float(lambda_execution)
        self.lambda_tracking = float(
            _profile_value(profile, "lambda_tracking", lambda_tracking))
        self.lambda_saddle_value = float(_profile_value(
            profile, "lambda_saddle_value", lambda_saddle_value))
        self.morse_decision_mode = str(
            morse_decision_mode or "balanced").strip().lower()
        if self.morse_decision_mode not in ("balanced", "priority"):
            self.morse_decision_mode = "balanced"
        self.dynamics_profile = dict(dynamics_profile or {})
        self.dynamics_profile.setdefault(
            "nominal_speed", profile.get("nominal_speed", 0.30))
        self.dynamics_profile.setdefault(
            "max_tracking_turn", profile.get("max_tracking_turn", 1.35))
        self.dynamics_profile.setdefault(
            "max_curvature", profile.get("max_curvature", 4.0))
        self.min_saddle_offset = max(0.0, float(min_saddle_offset))
        self.goal_saddle_exclusion = max(0.0, float(goal_saddle_exclusion))
        self.corridor_dedupe_distance = max(0.0, float(corridor_dedupe_distance))
        self.candidate_pool_min = max(
            1, int(3 if candidate_pool_min in (None, "", "auto") else
                   candidate_pool_min))
        default_route_max_paths = (
            32 if self.topology_profile == "arm" else 512)
        default_route_max_routes = (
            16 if self.topology_profile == "arm" else 256)
        self.route_max_paths = max(1, int(
            default_route_max_paths if route_max_paths in (None, "", "auto")
            else route_max_paths))
        self.route_max_routes = max(1, int(
            default_route_max_routes if route_max_routes in (None, "", "auto")
            else route_max_routes))
        self.max_corridor_turn = float(
            self.dynamics_profile.get("max_tracking_turn", 1.35)
            if max_corridor_turn is None else max_corridor_turn)
        self.max_corridor_curvature = float(
            self.dynamics_profile.get("max_curvature", 4.0)
            if max_corridor_curvature is None else max_corridor_curvature)
        self.min_segment_length = max(0.0, float(min_segment_length))
        self.require_risk_improvement = bool(require_risk_improvement)
        default_planning_margin = (
            0.0 if self.topology_profile == "wheelchair" else 0.03)
        self.planning_clearance_margin = float(
            default_planning_margin
            if planning_clearance_margin in (None, "", "auto") else
            planning_clearance_margin)
        self.manifold_constraint_mode = str(
            manifold_constraint_mode or "soft").strip().lower()
        if self.manifold_constraint_mode not in ("soft", "hard"):
            self.manifold_constraint_mode = "soft"
        candidate_max_risk = _profile_value(
            profile, "candidate_max_risk", candidate_max_risk)
        self.candidate_max_risk = (
            self.rho if candidate_max_risk is None
            else float(candidate_max_risk))
        self.corridor_score_weights = dict(corridor_score_weights or {})
        self.task_mode = resolve_task_mode(
            task_mode, robot_type=self.topology_profile)
        self.task_weight = resolve_task_weight(
            self.task_mode, task_config=task_config, task_weight=task_weight,
            robot_type=self.topology_profile)
        self.task_minima_points = list(task_minima_points or [])
        self.last_debug = {}
        self._grid = None

    def _manifold_robot_phase(self):
        robot = str(self.topology_profile or "generic").strip().lower()
        if robot == "wheelchair":
            return "wheelchair", "navigation"
        if robot == "arm":
            return "arm", "approach"
        return robot, "navigation" if robot == "wheelchair" else "approach"

    def _debug_inc(self, key, amount=1):
        self.last_debug[key] = int(self.last_debug.get(key, 0)) + int(amount)

    def _clearance_penalty(self, clearance):
        if not np.isfinite(clearance):
            return 0.0
        gap = max(0.0, self.min_clearance - float(clearance))
        return gap * gap

    def _point_in_region(self, p2, region):
        p2 = np.asarray(p2, float)
        shape = str(region.get("shape", "rect"))
        center = np.asarray(region.get("center", [0.0, 0.0]), float)[:2]
        if shape == "circle":
            radius = float(region.get("radius", 0.0))
            return float(np.linalg.norm(p2 - center)) <= radius
        half_extent = np.asarray(region.get("half_extent", [0.0, 0.0]), float)[:2]
        return bool(np.all(np.abs(p2 - center) <= half_extent))

    def _safety_profile(self, p2):
        profile = {
            "region": "default",
            "rho": self.rho,
            "interest_rho": float(self.interest_config.get("rho", self.rho)),
            "min_clearance": self.min_clearance,
            "hard_clearance": self.hard_clearance,
        }
        for region in self.safety_regions:
            if not self._point_in_region(p2, region):
                continue
            profile["region"] = str(region.get("name", "region"))
            for key in ("rho", "interest_rho", "min_clearance", "hard_clearance"):
                if key in region:
                    profile[key] = float(region[key])
        return profile

    def _world(self, p2):
        p2 = np.asarray(p2, float)
        if self.to_world is not None:
            return np.asarray(self.to_world(p2), float)
        return np.array([p2[0], p2[1], self.z_height], float)

    def _plane(self, p):
        p = np.asarray(p, float)
        if self.to_plane is not None:
            return np.asarray(self.to_plane(p), float)[:2]
        return p[:2]

    def psi(self, p2, goal2):
        p2 = np.asarray(p2, float)
        goal2 = np.asarray(goal2, float)
        gamma = float(np.sum((p2 - goal2) ** 2))
        m = self.eps_m * float(np.sin(3.0 * p2[0]) * np.cos(3.0 * p2[1]))
        return gamma + self.lam_s * self.field.phi_s(self._world(p2)) + m

    def _normalized_phi(self, phi):
        phi = max(0.0, float(phi))
        scale = max(abs(float(self.rho)), 1.0)
        return phi / (scale + phi)

    def _safe_boundary_barrier(self, p2):
        (xmin, xmax), (ymin, ymax) = self.bounds
        p2 = np.asarray(p2, float)
        dx = max(0.0, min(p2[0] - xmin, xmax - p2[0]))
        dy = max(0.0, min(p2[1] - ymin, ymax - p2[1]))
        d = max(min(dx, dy), self.grid_resolution)
        return float((self.grid_resolution / d) ** 2)

    def morse_potential(self, p2, goal2):
        p2 = np.asarray(p2, float)
        goal2 = np.asarray(goal2, float)
        d_goal = float(np.sum((p2 - goal2) ** 2))
        phi = self._normalized_phi(self.field.phi_s(self._world(p2)))
        barrier = self._safe_boundary_barrier(p2)
        texture = self.morse_texture_eps * float(
            math.sin(self.morse_texture_freq * p2[0]) *
            math.cos(self.morse_texture_freq * p2[1]))
        return (
            self.morse_w_goal * d_goal +
            self.morse_w_social * phi +
            self.morse_w_barrier * barrier +
            texture)

    def _interest_summary(self, z):
        cfg = self.interest_config
        if not cfg or not bool(cfg.get("enabled", False)):
            return {
                "phi_max": 0.0,
                "labels": [],
                "points": [],
                "forbidden_hit": False,
                "forbidden_reason": "",
            }
        if cfg.get("offsets") is not None:
            summary = pose_interest_risk(
                self.field, z, offsets=cfg.get("offsets"),
                labels=cfg.get("labels"))
        else:
            pose = np.array([z[0], z[1], float(cfg.get("yaw", 0.0))], float)
            summary = pose_interest_risk(
                self.field, pose,
                local_points=cfg.get("local_points"),
                labels=cfg.get("labels"))
        hit, label, anchor, reason = forbidden_anchor_hit(
            self.field, summary.get("labels", []), summary.get("points", []))
        summary["forbidden_hit"] = bool(hit)
        summary["forbidden_label"] = label
        summary["forbidden_anchor"] = anchor
        summary["forbidden_reason"] = reason
        return summary

    def _interest_risk(self, z):
        return float(self._interest_summary(z).get("phi_max", 0.0))

    def _safe_components(self, center_phi, interest_phi, forbidden_hit,
                         clearance=None, profile=None):
        profile = profile or {
            "rho": self.rho,
            "interest_rho": float(self.interest_config.get("rho", self.rho)),
            "hard_clearance": self.hard_clearance,
        }
        rho = float(profile.get("rho", self.rho))
        interest_rho = float(profile.get("interest_rho", rho))
        hard_clearance = float(profile.get("hard_clearance", self.hard_clearance))
        base_safe = (
            float(center_phi) <= rho and
            float(interest_phi) <= interest_rho and
            not bool(forbidden_hit))
        clearance_safe = (
            True if clearance is None else
            float(clearance) >= hard_clearance)
        return {
            "center_safe": float(center_phi) <= rho,
            "interest_safe": float(interest_phi) <= interest_rho,
            "forbidden_safe": not bool(forbidden_hit),
            "clearance_safe": bool(clearance_safe),
            "base_safe": bool(base_safe),
            "safe": bool(base_safe and clearance_safe),
        }

    def _safety_record(self, stage, action, reason="", p2=None, ij=None,
                       center_phi=None, interest_phi=None,
                       forbidden_hit=False, clearance=None, profile=None,
                       node_id="", edge="", corridor_label="", extra=None):
        if profile is None:
            if p2 is not None:
                profile = self._safety_profile(p2)
            else:
                profile = {}
        rho = float(profile.get("rho", self.rho))
        interest_rho = float(profile.get(
            "interest_rho", self.interest_config.get("rho", rho)))
        hard_clearance = float(profile.get(
            "hard_clearance", self.hard_clearance))
        min_clearance = float(profile.get(
            "min_clearance", self.min_clearance))
        center = 0.0 if center_phi is None else float(center_phi)
        interest = 0.0 if interest_phi is None else float(interest_phi)
        components = self._safe_components(
            center, interest, forbidden_hit,
            clearance=clearance, profile=profile)
        rec = {
            "stage": str(stage),
            "action": str(action),
            "reason": str(reason or ""),
            "node_id": str(node_id or ""),
            "edge": str(edge or ""),
            "corridor_label": str(corridor_label or ""),
            "region": str(profile.get("region", "default")),
            "center_phi": center,
            "center_rho": rho,
            "interest_phi": interest,
            "interest_rho": interest_rho,
            "forbidden_hit": bool(forbidden_hit),
            "clearance": (
                float(clearance) if clearance is not None else float("inf")),
            "hard_clearance": hard_clearance,
            "min_clearance": min_clearance,
            "center_safe": bool(components["center_safe"]),
            "interest_safe": bool(components["interest_safe"]),
            "forbidden_safe": bool(components["forbidden_safe"]),
            "clearance_safe": bool(components["clearance_safe"]),
            "base_safe": bool(components["base_safe"]),
            "safe": bool(components["safe"]),
        }
        if ij is not None:
            rec["ij"] = [int(ij[0]), int(ij[1])]
        if p2 is not None:
            arr = np.asarray(p2, float)
            rec["p2"] = [float(arr[0]), float(arr[1])]
        if extra:
            rec.update(extra)
        return rec

    def _grid_safety_record(self, ij, stage, action, reason="", **extra):
        p2 = self._ij_to_p2(ij)
        grid = self._grid
        profile = {
            "region": grid.get("region_name")[ij],
            "rho": float(grid.get("rho")[ij]),
            "interest_rho": float(grid.get("interest_rho")[ij]),
            "hard_clearance": float(grid.get("hard_clearance")[ij]),
            "min_clearance": float(grid.get("min_clearance")[ij]),
        }
        return self._safety_record(
            stage, action, reason=reason, p2=p2, ij=ij,
            center_phi=float(grid.get("center_phi", grid["phi"])[ij]),
            interest_phi=float(grid["interest_phi"][ij]),
            forbidden_hit=bool(grid["forbidden"][ij]),
            clearance=float(grid["clearance"][ij]),
            profile=profile, extra=extra)

    def safe_check(self, x, interest_phi=0.0, forbidden_hit=False,
                   clearance=None):
        arr = np.asarray(x, float)
        if arr.shape:
            p2 = self._plane(arr)
            z = self._world(p2)
            center_phi = float(self.field.phi_s(z))
            summary = self._interest_summary(z)
            interest_phi = float(summary.get("phi_max", interest_phi))
            forbidden_hit = bool(summary.get("forbidden_hit", forbidden_hit))
            if clearance is None and self._grid is not None:
                ij = self._p2_to_ij(p2)
                clearance = self._clearance_cells(ij)
            profile = self._safety_profile(p2)
        else:
            center_phi = float(arr)
            profile = None
        return self._safe_components(
            center_phi, interest_phi, forbidden_hit, clearance, profile)["safe"]

    def build_grid(self, goal):
        goal2 = self._plane(goal)
        (xmin, xmax), (ymin, ymax) = self.bounds
        nx = max(3, int(math.ceil((xmax - xmin) / self.grid_resolution)) + 1)
        ny = max(3, int(math.ceil((ymax - ymin) / self.grid_resolution)) + 1)
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        center_phi_grid = np.zeros((nx, ny), float)
        phi = np.zeros((nx, ny), float)
        interest_phi = np.zeros((nx, ny), float)
        psi = np.zeros((nx, ny), float)
        morse_u = np.zeros((nx, ny), float)
        safe = np.zeros((nx, ny), bool)
        base_safe = np.zeros((nx, ny), bool)
        forbidden = np.zeros((nx, ny), bool)
        rho_grid = np.zeros((nx, ny), float)
        interest_rho_grid = np.zeros((nx, ny), float)
        hard_clearance_grid = np.zeros((nx, ny), float)
        min_clearance_grid = np.zeros((nx, ny), float)
        region_name = np.empty((nx, ny), dtype=object)
        region_cell_counts = {}
        num_forbidden_cells = 0
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                p2 = np.array([x, y], float)
                z = self._world(p2)
                profile = self._safety_profile(p2)
                region = str(profile.get("region", "default"))
                region_name[i, j] = region
                region_cell_counts[region] = int(region_cell_counts.get(region, 0)) + 1
                rho_grid[i, j] = float(profile.get("rho", self.rho))
                interest_rho_grid[i, j] = float(
                    profile.get("interest_rho", rho_grid[i, j]))
                hard_clearance_grid[i, j] = float(
                    profile.get("hard_clearance", self.hard_clearance))
                min_clearance_grid[i, j] = float(
                    profile.get("min_clearance", self.min_clearance))
                center_phi = float(self.field.phi_s(z))
                center_phi_grid[i, j] = center_phi
                summary = self._interest_summary(z)
                ip_phi = float(summary.get("phi_max", 0.0))
                forbidden_hit = bool(summary.get("forbidden_hit", False))
                forbidden[i, j] = forbidden_hit
                if forbidden_hit:
                    num_forbidden_cells += 1
                interest_phi[i, j] = ip_phi
                phi[i, j] = max(center_phi, ip_phi)
                psi[i, j] = self.psi(p2, goal2)
                morse_u[i, j] = self.morse_potential(p2, goal2)
                base_safe[i, j] = self._safe_components(
                    center_phi, ip_phi, forbidden_hit, profile=profile)["base_safe"]
        unsafe_idx = np.argwhere(~base_safe)
        clearance = np.full((nx, ny), float("inf"), float)
        if len(unsafe_idx):
            unsafe_pts = np.array([
                np.array([xs[int(i)], ys[int(j)]], float)
                for i, j in unsafe_idx
            ])
            for i, x in enumerate(xs):
                for j, y in enumerate(ys):
                    p2 = np.array([x, y], float)
                    clearance[i, j] = float(np.min(
                        np.linalg.norm(unsafe_pts - p2[None, :], axis=1)))
        for i in range(nx):
            for j in range(ny):
                safe[i, j] = bool(
                    base_safe[i, j] and clearance[i, j] >= hard_clearance_grid[i, j])
        self._grid = {
            "xs": xs, "ys": ys, "center_phi": center_phi_grid, "phi": phi,
            "interest_phi": interest_phi, "forbidden": forbidden,
            "psi": psi, "morse_u": morse_u, "base_safe": base_safe,
            "clearance": clearance, "safe": safe,
            "rho": rho_grid, "interest_rho": interest_rho_grid,
            "hard_clearance": hard_clearance_grid,
            "min_clearance": min_clearance_grid,
            "region_name": region_name,
            "goal2": goal2,
        }
        grid_audit_counts = {
            "kept": 0,
            "reject_center": 0,
            "reject_interest": 0,
            "reject_forbidden": 0,
            "reject_clearance": 0,
        }
        grid_samples = []
        for i in range(nx):
            for j in range(ny):
                reason = ""
                action = "kept"
                if center_phi_grid[i, j] > rho_grid[i, j]:
                    action = "rejected"
                    reason = "center_risk"
                    grid_audit_counts["reject_center"] += 1
                elif interest_phi[i, j] > interest_rho_grid[i, j]:
                    action = "rejected"
                    reason = "interest_risk"
                    grid_audit_counts["reject_interest"] += 1
                elif forbidden[i, j]:
                    action = "rejected"
                    reason = "forbidden"
                    grid_audit_counts["reject_forbidden"] += 1
                elif clearance[i, j] < hard_clearance_grid[i, j]:
                    action = "rejected"
                    reason = "clearance"
                    grid_audit_counts["reject_clearance"] += 1
                else:
                    grid_audit_counts["kept"] += 1
                if len(grid_samples) < 24 and (action == "rejected" or len(grid_samples) < 6):
                    grid_samples.append(self._grid_safety_record(
                        (i, j), "grid", action, reason))
        self.last_debug["num_forbidden_cells"] = int(num_forbidden_cells)
        self.last_debug["safety_region_cell_counts"] = region_cell_counts
        self.last_debug["safety_regions"] = self.safety_regions
        self.last_debug["safety_audit_grid_counts"] = grid_audit_counts
        self.last_debug["safety_audit_grid_samples"] = grid_samples
        return self._grid

    def _ij_to_p2(self, ij):
        xs = self._grid["xs"]
        ys = self._grid["ys"]
        return np.array([xs[ij[0]], ys[ij[1]]], float)

    def _p2_to_ij(self, p2):
        xs = self._grid["xs"]
        ys = self._grid["ys"]
        i = int(np.argmin(np.abs(xs - p2[0])))
        j = int(np.argmin(np.abs(ys - p2[1])))
        return (i, j)

    def _nearest_safe_ij(self, p2):
        safe = self._grid["safe"]
        base_safe = self._grid.get("base_safe", safe)
        ij0 = self._p2_to_ij(p2)
        if safe[ij0]:
            return ij0
        safe_idx = np.argwhere(safe)
        if len(safe_idx) == 0:
            return None
        pts = np.array([self._ij_to_p2((int(i), int(j))) for i, j in safe_idx])
        d = np.linalg.norm(pts - np.asarray(p2, float)[None, :], axis=1)
        k = int(np.argmin(d))
        return (int(safe_idx[k][0]), int(safe_idx[k][1]))

    def associate_pose_to_topology_node(self, pose, nodes=None,
                                        preferred_id=None):
        p2 = self._plane(pose)
        projected_ij = self._nearest_safe_ij(p2)
        projected = (
            self._world(self._ij_to_p2(projected_ij))
            if projected_ij is not None else self._world(p2))
        nodes = list(nodes or [])
        if preferred_id:
            for node in nodes:
                if str(getattr(node, "id", "")) == str(preferred_id):
                    if node.ij is None:
                        node.ij = projected_ij
                        node.point = np.asarray(projected, float)
                    return {
                        "node_id": str(node.id),
                        "ij": list(node.ij) if node.ij is not None else None,
                        "projected": bool(np.linalg.norm(
                            self._plane(node.point) - p2) >
                            0.5 * self.grid_resolution),
                        "distance": float(np.linalg.norm(
                            self._plane(node.point) - p2)),
                    }
        nearest = None
        nearest_dist = float("inf")
        for node in nodes:
            if node.ij is None:
                continue
            dist = float(np.linalg.norm(self._plane(node.point) - p2))
            if dist < nearest_dist:
                nearest = node
                nearest_dist = dist
        if nearest is None:
            return {
                "node_id": None,
                "ij": list(projected_ij) if projected_ij is not None else None,
                "projected": bool(projected_ij is not None),
                "distance": 0.0,
            }
        return {
            "node_id": str(nearest.id),
            "ij": list(nearest.ij),
            "projected": bool(np.linalg.norm(
                self._ij_to_p2(nearest.ij) - p2) >
                0.5 * self.grid_resolution),
            "distance": float(nearest_dist),
        }

    def _clearance_cells(self, ij):
        clearance = self._grid.get("clearance")
        if clearance is not None:
            return float(clearance[ij])
        safe_idx = np.argwhere(~self._grid["safe"])
        if len(safe_idx) == 0:
            return float("inf")
        p = self._ij_to_p2(ij)
        pts = np.array([self._ij_to_p2((int(i), int(j))) for i, j in safe_idx])
        return float(np.min(np.linalg.norm(pts - p[None, :], axis=1)))

    def detect_morse_points(self):
        if self._grid is None:
            raise RuntimeError("build_grid must be called first")
        psi = self._grid.get("morse_u", self._grid["psi"])
        safe = self._grid["safe"]
        base_safe = self._grid.get("base_safe", safe)
        forbidden = self._grid.get("forbidden")
        nx, ny = psi.shape
        raw = {"minima": [], "saddles": [], "maxima": []}
        raw_points = {"minima": [], "saddles": [], "maxima": []}
        safe_points = {"minima": [], "saddles": [], "maxima": []}
        critical_chain = {"minima": [], "saddles": [], "maxima": []}
        raw_counts = {"minima": 0, "saddles": 0, "maxima": 0}
        safe_counts = {"minima": 0, "saddles": 0, "maxima": 0}
        clearance_reject_counts = {"minima": 0, "saddles": 0, "maxima": 0}
        reject = {
            "reject_by_gradient_count": 0,
            "reject_by_degenerate_count": 0,
            "reject_by_forbidden_count": 0,
            "reject_by_clearance_count": 0,
            "reject_by_unsafe_count": 0,
        }
        step = max(self.grid_resolution, 1e-6)
        grad_eps = max(self.morse_grad_eps, 0.25 * step)
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                c = psi[i, j]
                fxx = (psi[i + 1, j] - 2.0 * c + psi[i - 1, j]) / (step * step)
                fyy = (psi[i, j + 1] - 2.0 * c + psi[i, j - 1]) / (step * step)
                fxy = (psi[i + 1, j + 1] - psi[i + 1, j - 1] -
                       psi[i - 1, j + 1] + psi[i - 1, j - 1]) / (4.0 * step * step)
                eig = np.linalg.eigvalsh(np.array([[fxx, fxy], [fxy, fyy]], float))
                gx = (psi[i + 1, j] - psi[i - 1, j]) / (2.0 * step)
                gy = (psi[i, j + 1] - psi[i, j - 1]) / (2.0 * step)
                grad_norm = math.sqrt(gx * gx + gy * gy)
                if grad_norm > grad_eps:
                    reject["reject_by_gradient_count"] += 1
                    continue
                if min(abs(float(eig[0])), abs(float(eig[1]))) < self.morse_degenerate_eps:
                    reject["reject_by_degenerate_count"] += 1
                    continue
                point_kind = ""
                if eig[0] > 0.0 and eig[1] > 0.0:
                    point_kind = "minima"
                elif eig[0] < 0.0 and eig[1] < 0.0:
                    point_kind = "maxima"
                elif eig[0] < 0.0 and eig[1] > 0.0:
                    point_kind = "saddles"
                if not point_kind:
                    continue
                raw_counts[point_kind] += 1
                raw_idx = raw_counts[point_kind] - 1
                p2 = self._ij_to_p2((i, j))
                point_rec = {
                    "id": "%s_raw_%d" % (point_kind[:-1], raw_idx),
                    "raw_index": raw_idx,
                    "kind": point_kind[:-1],
                    "point": self._world(p2),
                    "p2": p2,
                    "ij": (i, j),
                    "psi": float(c),
                    "grad_norm": float(grad_norm),
                    "eig": [float(eig[0]), float(eig[1])],
                    "phi": float(self._grid["phi"][i, j]),
                    "clearance": float(self._clearance_cells((i, j))),
                    "stage": "raw",
                    "status": "raw",
                    "reason": "",
                    "safety": self._grid_safety_record(
                        (i, j), "critical_raw", "candidate"),
                }
                raw_points[point_kind].append(point_rec)
                if forbidden is not None and bool(forbidden[i, j]):
                    reject["reject_by_forbidden_count"] += 1
                    rejected = dict(point_rec)
                    rejected["stage"] = "filtered"
                    rejected["status"] = "rejected"
                    rejected["reason"] = "forbidden"
                    rejected["safety"] = self._grid_safety_record(
                        (i, j), "critical_filter", "rejected", "forbidden")
                    critical_chain[point_kind].append(rejected)
                    continue
                if not base_safe[i, j]:
                    reject["reject_by_unsafe_count"] += 1
                    rejected = dict(point_rec)
                    rejected["stage"] = "filtered"
                    rejected["status"] = "rejected"
                    rejected["reason"] = "unsafe"
                    rejected["safety"] = self._grid_safety_record(
                        (i, j), "critical_filter", "rejected", "unsafe")
                    critical_chain[point_kind].append(rejected)
                    continue
                safe_counts[point_kind] += 1
                safe_points[point_kind].append(point_rec)
                clearance = float(point_rec["clearance"])
                if clearance < self._hard_clearance_at_ij((i, j)):
                    clearance_reject_counts[point_kind] += 1
                    reject["reject_by_clearance_count"] += 1
                    rejected = dict(point_rec)
                    rejected["stage"] = "filtered"
                    rejected["status"] = "rejected"
                    rejected["reason"] = "clearance"
                    rejected["safety"] = self._grid_safety_record(
                        (i, j), "critical_filter", "rejected", "clearance")
                    critical_chain[point_kind].append(rejected)
                    continue
                raw[point_kind].append(point_rec)
        merged = {}
        merge_rejects = {"minima": [], "saddles": [], "maxima": []}
        for kind in ("minima", "saddles", "maxima"):
            merged[kind], merge_rejects[kind] = self._merge_points_with_reasons(
                raw[kind])
            for item in merged[kind]:
                rec = dict(item)
                rec["stage"] = "filtered"
                rec["status"] = "filtered"
                rec["reason"] = ""
                rec["safety"] = self._grid_safety_record(
                    rec["ij"], "critical_filter", "kept")
                critical_chain[kind].append(rec)
            for item in merge_rejects[kind]:
                rec = dict(item)
                rec["stage"] = "filtered"
                rec["status"] = "rejected"
                rec["reason"] = "merged_duplicate"
                rec["safety"] = self._grid_safety_record(
                    rec["ij"], "critical_filter", "rejected",
                    "merged_duplicate")
                critical_chain[kind].append(rec)
        merge_counts = {
            kind: len(merge_rejects[kind])
            for kind in ("minima", "saddles", "maxima")
        }
        merged["_raw_counts"] = raw_counts
        merged["_safe_counts"] = safe_counts
        merged["_raw_points"] = raw_points
        merged["_safe_points"] = safe_points
        merged["_critical_chain"] = critical_chain
        merged["_merge_rejects"] = merge_rejects
        merged["_merge_counts"] = merge_counts
        merged["_clearance_reject_counts"] = clearance_reject_counts
        merged["_reject_counts"] = reject
        return merged

    def _apply_semantic_topology_recovery(self, critical, start, goal,
                                          semantic_nodes=None):
        if not self.allow_semantic_topology_recovery:
            return critical
        needs_recovery = bool(
            not critical.get("minima") or not critical.get("saddles"))
        if not needs_recovery:
            self.last_debug["semantic_topology_recovery_used"] = False
            return critical

        specs = semantic_topology_specs(
            self.topology_profile, start, goal, self.bounds,
            task_minima_points=self.task_minima_points,
            semantic_nodes=semantic_nodes,
            merge_radius=self.merge_radius)
        if not specs:
            self.last_debug["semantic_topology_recovery_used"] = False
            self.last_debug["semantic_topology_recovery_reason"] = (
                "no_semantic_specs")
            return critical

        out = dict(critical)
        chain = out.setdefault("_critical_chain", {
            "minima": [], "saddles": [], "maxima": []})
        raw_counts = dict(out.get("_raw_counts", {}) or {})
        safe_counts = dict(out.get("_safe_counts", {}) or {})
        raw_points = out.setdefault("_raw_points", {
            "minima": [], "saddles": [], "maxima": []})
        safe_points = out.setdefault("_safe_points", {
            "minima": [], "saddles": [], "maxima": []})
        added = {"minima": 0, "saddles": 0}
        for spec in specs:
            kind = str(spec.get("kind", "")).strip().lower()
            plural = "saddles" if kind == "saddle" else "minima"
            if plural == "minima" and out.get("minima"):
                continue
            if plural == "saddles" and out.get("saddles"):
                continue
            p2 = self._plane(spec.get("point", []))
            ij = self._nearest_safe_ij(p2)
            if ij is None:
                continue
            p2 = self._ij_to_p2(ij)
            idx = len(out.get(plural, []))
            crit_id = "semantic_%s_%d" % (
                "saddle" if plural == "saddles" else "minimum", idx)
            rec = {
                "id": crit_id,
                "raw_index": idx,
                "kind": "saddle" if plural == "saddles" else "minimum",
                "point": self._world(p2),
                "p2": p2,
                "ij": ij,
                "psi": float(self.morse_potential(p2, self._plane(goal))),
                "grad_norm": 0.0,
                "eig": [-1.0, 1.0] if plural == "saddles" else [1.0, 1.0],
                "phi": float(self._grid["phi"][ij]),
                "clearance": float(self._clearance_cells(ij)),
                "stage": "semantic_recovery",
                "status": "filtered",
                "reason": "",
                "semantic_type": str(spec.get("semantic_type", "")),
                "source": "semantic_topology_recovery",
                "safety": self._grid_safety_record(
                    ij, "critical_semantic_recovery", "kept",
                    semantic_type=str(spec.get("semantic_type", ""))),
            }
            out.setdefault(plural, []).append(rec)
            raw_points.setdefault(plural, []).append(rec)
            safe_points.setdefault(plural, []).append(rec)
            chain.setdefault(plural, []).append(dict(rec))
            raw_counts[plural] = int(raw_counts.get(plural, 0)) + 1
            safe_counts[plural] = int(safe_counts.get(plural, 0)) + 1
            added[plural] += 1
        out["_raw_counts"] = raw_counts
        out["_safe_counts"] = safe_counts
        self.last_debug["semantic_topology_recovery_used"] = bool(
            added["minima"] or added["saddles"])
        self.last_debug["semantic_topology_recovery_added_minima"] = int(
            added["minima"])
        self.last_debug["semantic_topology_recovery_added_saddles"] = int(
            added["saddles"])
        self.last_debug["semantic_topology_recovery_reason"] = (
            "morse_missing_minima_or_saddle")
        return out

    def _merge_points(self, items):
        kept, _rejected = self._merge_points_with_reasons(items)
        return kept

    def _merge_points_with_reasons(self, items):
        ordered = sorted(items, key=lambda x: (x["psi"], -x["clearance"]))
        kept = []
        rejected = []
        for item in ordered:
            p = item["p2"]
            duplicate = False
            merged_into = ""
            for old in kept:
                if np.linalg.norm(p - old["p2"]) < self.merge_radius:
                    duplicate = True
                    merged_into = old.get("id", "")
                    break
            if not duplicate:
                kept.append(item)
            else:
                rec = dict(item)
                rec["merged_into"] = merged_into
                rejected.append(rec)
        return kept, rejected

    def _astar(self, start_ij, goal_ij):
        safe = self._grid["safe"]
        phi = self._grid["phi"]
        nx, ny = safe.shape
        if start_ij is None or goal_ij is None:
            return None, float("inf")
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]
        openq = []
        heapq.heappush(openq, (0.0, start_ij))
        came = {}
        g = {start_ij: 0.0}

        def h(ij):
            a = self._ij_to_p2(ij)
            b = self._ij_to_p2(goal_ij)
            return float(np.linalg.norm(a - b))

        while openq:
            _, cur = heapq.heappop(openq)
            if cur == goal_ij:
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                path.reverse()
                return path, g[goal_ij]
            for di, dj in moves:
                ni, nj = cur[0] + di, cur[1] + dj
                nxt = (ni, nj)
                if ni < 0 or nj < 0 or ni >= nx or nj >= ny or not safe[nxt]:
                    continue
                step_len = self.grid_resolution * (math.sqrt(2.0) if di and dj else 1.0)
                risk = 0.5 * (float(phi[cur]) + float(phi[nxt]))
                cost = step_len * (1.0 + 0.35 * risk + 0.25 * max(0.0, risk - self.rho))
                ng = g[cur] + cost
                if ng < g.get(nxt, float("inf")):
                    came[nxt] = cur
                    g[nxt] = ng
                    heapq.heappush(openq, (ng + h(nxt), nxt))
        return None, float("inf")

    def _resample(self, points, max_points=18):
        points = np.asarray(points, float)
        if len(points) <= 2:
            return points
        seg_lens = [float(np.linalg.norm(b - a)) for a, b in zip(points[:-1], points[1:])]
        total = sum(seg_lens)
        if total <= 1e-9:
            return points[:1]
        n = min(max_points, max(3, int(math.ceil(total / max(self.grid_resolution * 2.0, 1e-6))) + 1))
        dists = np.linspace(0.0, total, n)
        out = []
        for d in dists:
            acc = 0.0
            for idx, length in enumerate(seg_lens):
                if d <= acc + length or idx == len(seg_lens) - 1:
                    a = points[idx]
                    b = points[idx + 1]
                    alpha = (d - acc) / max(length, 1e-9)
                    out.append(a + alpha * (b - a))
                    break
                acc += length
        return np.asarray(out, float)

    def _candidate_waypoints_from_nodes(self, node_ids, node_by_id, cell_points):
        skeleton = []
        has_morse_node = False
        for node_id in node_ids:
            node = node_by_id.get(node_id)
            if node is None:
                continue
            skeleton.append(np.asarray(node.point, float))
            if node.kind in ("saddle", "minimum"):
                has_morse_node = True
        if not has_morse_node or len(skeleton) < 2:
            return self._resample(cell_points)

        # Kinematic checks must evaluate the Morse skeleton, not jagged grid
        # cells. Otherwise valid saddle corridors can be rejected before the
        # protected waypoint ever reaches the executor.
        pieces = []
        for a, b in zip(skeleton[:-1], skeleton[1:]):
            a = np.asarray(a, float)
            b = np.asarray(b, float)
            dist = float(np.linalg.norm(b - a))
            steps = max(2, int(math.ceil(dist / max(self.grid_resolution * 2.5, 1e-6))) + 1)
            for idx in range(steps):
                if pieces and idx == 0:
                    continue
                alpha = float(idx) / float(max(steps - 1, 1))
                pieces.append(a + alpha * (b - a))
        return np.asarray(pieces, float)

    def _edge_metrics(self, cell_path):
        phi = self._grid["phi"]
        vals = [float(phi[ij]) for ij in cell_path]
        pts = [self._ij_to_p2(ij) for ij in cell_path]
        length = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            length += float(np.linalg.norm(b - a))
        return length, float(np.mean(vals)), float(np.max(vals))

    def _cell_path_clearance(self, cell_path):
        if not cell_path:
            return 0.0
        return float(min(self._clearance_cells(ij) for ij in cell_path))

    def _hard_clearance_at_ij(self, ij):
        grid = self._grid.get("hard_clearance") if self._grid is not None else None
        if grid is None:
            return self.hard_clearance
        return float(grid[ij])

    def _min_clearance_at_ij(self, ij):
        grid = self._grid.get("min_clearance") if self._grid is not None else None
        if grid is None:
            return self.min_clearance
        return float(grid[ij])

    def _cell_path_clearance_margin(self, cell_path):
        if not cell_path:
            return 0.0
        return float(min(
            self._clearance_cells(ij) - self._hard_clearance_at_ij(ij)
            for ij in cell_path))

    def _cell_path_clearance_penalty(self, cell_path):
        if not cell_path:
            return 0.0
        penalty = 0.0
        for ij in cell_path:
            gap = max(0.0, self._min_clearance_at_ij(ij) - self._clearance_cells(ij))
            penalty = max(penalty, gap * gap)
        return float(penalty)

    def _cell_path_forbidden_hits(self, cell_path):
        forbidden = self._grid.get("forbidden")
        if forbidden is None:
            return 0
        return int(sum(1 for ij in cell_path if bool(forbidden[ij])))

    def _graph_has_path(self, edges, start_id="start", goal_id="goal"):
        stack = [str(start_id)]
        seen = set()
        goal_id = str(goal_id)
        while stack:
            cur = stack.pop()
            if cur == goal_id:
                return True
            if cur in seen:
                continue
            seen.add(cur)
            for edge in list((edges or {}).get(cur, []) or []):
                nxt = str(edge.get("to", ""))
                if nxt and nxt not in seen:
                    stack.append(nxt)
        return False

    def _edge_record_from_cells(self, node_a, node_b, cell_path, astar_cost):
        edge_clearance = self._cell_path_clearance(cell_path)
        edge_clearance_margin = self._cell_path_clearance_margin(cell_path)
        edge_forbidden_hits = self._cell_path_forbidden_hits(cell_path)
        length, mean_phi, max_phi = self._edge_metrics(cell_path)
        clearance_penalty = self._cell_path_clearance_penalty(cell_path)
        cost = (length + 0.8 * mean_phi + 0.4 * max_phi +
                0.15 * astar_cost + 8.0 * clearance_penalty)
        return {
            "to": node_b.id,
            "cost": float(cost),
            "cells": cell_path,
            "length": length,
            "mean_phi": mean_phi,
            "max_phi": max_phi,
            "clearance": edge_clearance,
            "clearance_margin": edge_clearance_margin,
            "clearance_penalty": clearance_penalty,
            "forbidden_hits": edge_forbidden_hits,
        }

    def _add_graph_edge(self, edges, node_a, node_b):
        if node_a is None or node_b is None or node_a.ij is None or node_b.ij is None:
            return False, "missing_node"
        for edge in list((edges or {}).get(node_a.id, []) or []):
            if str(edge.get("to", "")) == str(node_b.id):
                return False, "duplicate"
        cell_path, astar_cost = self._astar(node_a.ij, node_b.ij)
        if not cell_path:
            return False, "astar_fail"
        if self._cell_path_clearance_margin(cell_path) < 0.0:
            return False, "clearance"
        if self._cell_path_forbidden_hits(cell_path):
            return False, "forbidden"
        rec = self._edge_record_from_cells(node_a, node_b, cell_path, astar_cost)
        rev = dict(rec)
        rev["to"] = node_a.id
        rev["cells"] = list(reversed(cell_path))
        edges.setdefault(node_a.id, []).append(rec)
        edges.setdefault(node_b.id, []).append(rev)
        return True, ""

    def _repair_graph_connectivity(self, nodes, edges, start_id="start",
                                   goal_id="goal"):
        if self._graph_has_path(edges, start_id, goal_id):
            return {
                "attempted": False,
                "repaired": True,
                "added_edges": 0,
                "reason": "",
            }
        node_by_id = {str(node.id): node for node in nodes or []}
        endpoints = [
            node_by_id.get(str(start_id)),
            node_by_id.get(str(goal_id)),
        ]
        critical_nodes = [
            node for node in nodes or []
            if str(getattr(node, "kind", "")) in ("saddle", "minimum")
        ]
        attempts = []
        added = 0
        for endpoint in endpoints:
            if endpoint is None:
                continue
            nearby = sorted(
                critical_nodes,
                key=lambda node: float(np.linalg.norm(
                    self._plane(endpoint.point) - self._plane(node.point))))
            for node in nearby[:max(2, min(len(nearby), self.neighbor_k))]:
                ok, reason = self._add_graph_edge(edges, endpoint, node)
                attempts.append({
                    "edge": "%s->%s" % (endpoint.id, node.id),
                    "added": bool(ok),
                    "reason": str(reason),
                })
                if ok:
                    added += 1
                if self._graph_has_path(edges, start_id, goal_id):
                    return {
                        "attempted": True,
                        "repaired": True,
                        "added_edges": int(added),
                        "attempts": attempts,
                        "reason": "",
                    }
        for node in critical_nodes:
            nearby = sorted(
                [other for other in critical_nodes if other.id != node.id],
                key=lambda other: float(np.linalg.norm(
                    self._plane(node.point) - self._plane(other.point))))
            for other in nearby[:2]:
                ok, reason = self._add_graph_edge(edges, node, other)
                attempts.append({
                    "edge": "%s->%s" % (node.id, other.id),
                    "added": bool(ok),
                    "reason": str(reason),
                })
                if ok:
                    added += 1
                if self._graph_has_path(edges, start_id, goal_id):
                    return {
                        "attempted": True,
                        "repaired": True,
                        "added_edges": int(added),
                        "attempts": attempts,
                        "reason": "",
                    }
        return {
            "attempted": True,
            "repaired": False,
            "added_edges": int(added),
            "attempts": attempts,
            "reason": "start_goal_disconnected",
        }

    def _point_segment_distance(self, p, a, b):
        p = np.asarray(p, float)[:2]
        a = np.asarray(a, float)[:2]
        b = np.asarray(b, float)[:2]
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            return float(np.linalg.norm(p - a))
        t = np.clip(float(np.dot(p - a, ab)) / denom, 0.0, 1.0)
        q = a + t * ab
        return float(np.linalg.norm(p - q))

    def _saddle_valid_for_corridor(self, node, start2, goal2):
        if node.kind != "saddle":
            return True, "", 0.0
        p2 = self._plane(node.point)
        offset = self._point_segment_distance(p2, start2, goal2)
        if offset < self.min_saddle_offset:
            return False, "saddle_offset_too_small", offset
        if self.goal_saddle_exclusion > 0.0:
            if float(np.linalg.norm(p2 - goal2)) < self.goal_saddle_exclusion:
                return False, "saddle_too_close_to_goal", offset
        if node.ij is None or not self._grid["safe"][node.ij]:
            return False, "saddle_not_in_safe_manifold", offset
        return True, "", offset

    def _path_execution_metrics(self, pts):
        pts = np.asarray(pts, float)
        if len(pts) < 2:
            return 0.0, 0.0, 0.0, 0.0
        seg = np.diff(pts[:, :2], axis=0)
        lens = np.linalg.norm(seg, axis=1)
        path_length = float(np.sum(lens))
        angles = []
        for a, b in zip(seg[:-1], seg[1:]):
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            if na < 1e-9 or nb < 1e-9:
                continue
            cosang = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
            angles.append(float(math.acos(cosang)))
        max_turn = float(max(angles)) if angles else 0.0
        mean_turn = float(np.mean(angles)) if angles else 0.0
        execution_cost = path_length * (mean_turn + 0.5 * max_turn)
        return path_length, max_turn, mean_turn, float(execution_cost)

    def _path_tracking_metrics(self, pts):
        pts = np.asarray(pts, float)
        out = {
            "tracking_cost": 0.0,
            "max_curvature": 0.0,
            "curvature_violation": 0.0,
            "turn_violation": 0.0,
            "min_segment_length": 0.0,
            "expected_progress": 0.0,
        }
        if len(pts) < 3:
            if len(pts) >= 2:
                out["expected_progress"] = float(np.linalg.norm(
                    pts[-1, :2] - pts[0, :2]))
            return out
        seg = np.diff(pts[:, :2], axis=0)
        lens = np.linalg.norm(seg, axis=1)
        valid_lens = lens[lens > 1e-9]
        if valid_lens.size == 0:
            return out
        min_seg = float(np.min(valid_lens))
        angles = []
        curvatures = []
        for idx, (a, b) in enumerate(zip(seg[:-1], seg[1:])):
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            if na < 1e-9 or nb < 1e-9:
                continue
            cosang = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
            angle = float(math.acos(cosang))
            angles.append(angle)
            local_len = max(0.5 * (na + nb), self.grid_resolution, 1e-6)
            curvatures.append(angle / local_len)
        if not angles:
            out["min_segment_length"] = min_seg
            out["expected_progress"] = float(np.linalg.norm(
                pts[-1, :2] - pts[0, :2]))
            return out

        max_turn = float(max(angles))
        mean_turn = float(np.mean(angles))
        max_curvature = float(max(curvatures)) if curvatures else 0.0
        dyn = self.dynamics_profile
        nominal_speed = max(0.0, float(dyn.get("nominal_speed", 0.0)))
        max_curvature_allowed = float(dyn.get("max_curvature", 0.0))
        v_max = float(dyn.get("v_max", 0.0))
        w_max = float(dyn.get("w_max", 0.0))
        if v_max > 1e-9 and w_max > 1e-9:
            speed_for_curve = max(
                min(nominal_speed if nominal_speed > 0.0 else v_max, v_max),
                0.05)
            max_curvature_allowed = min(
                max_curvature_allowed if max_curvature_allowed > 0.0 else float("inf"),
                w_max / speed_for_curve)
        if max_curvature_allowed <= 0.0:
            max_curvature_allowed = float("inf")
        max_turn_allowed = max(
            0.05, float(dyn.get("max_tracking_turn", 1.35)))
        curvature_violation = max(0.0, max_curvature - max_curvature_allowed)
        turn_violation = max(0.0, max_turn - max_turn_allowed)
        path_length = float(np.sum(valid_lens))
        chord = float(np.linalg.norm(pts[-1, :2] - pts[0, :2]))
        tortuosity = max(0.0, path_length / max(chord, self.grid_resolution) - 1.0)
        smooth_cost = float(np.sum(np.asarray(angles, float) ** 2))
        expected_progress = chord / (1.0 + 0.5 * mean_turn + tortuosity)
        progress_deficit = max(
            0.0, float(dyn.get("min_progress", 0.0)) - expected_progress)
        tracking_cost = (
            1.5 * smooth_cost +
            2.0 * turn_violation * turn_violation +
            1.2 * curvature_violation * curvature_violation +
            0.8 * tortuosity +
            4.0 * progress_deficit)
        out.update({
            "tracking_cost": float(tracking_cost),
            "max_curvature": max_curvature,
            "curvature_violation": float(curvature_violation),
            "turn_violation": float(turn_violation),
            "min_segment_length": min_seg,
            "expected_progress": float(expected_progress),
        })
        return out

    def _segments_intersect(self, a, b, c, d):
        a = np.asarray(a, float)[:2]
        b = np.asarray(b, float)[:2]
        c = np.asarray(c, float)[:2]
        d = np.asarray(d, float)[:2]

        def orient(p, q, r):
            return float((q[0] - p[0]) * (r[1] - p[1]) -
                         (q[1] - p[1]) * (r[0] - p[0]))

        o1 = orient(a, b, c)
        o2 = orient(a, b, d)
        o3 = orient(c, d, a)
        o4 = orient(c, d, b)
        return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)

    def _path_self_intersection(self, pts):
        pts = np.asarray(pts, float)
        if len(pts) < 4:
            return False
        for i in range(len(pts) - 1):
            for j in range(i + 2, len(pts) - 1):
                if i == 0 and j == len(pts) - 2:
                    continue
                if self._segments_intersect(pts[i], pts[i + 1], pts[j], pts[j + 1]):
                    return True
        return False

    def _baseline_risk_reference(self, start, goal):
        pts = self._resample(np.asarray([start, goal], float), max_points=24)
        length = 0.0
        vals = []
        for p in pts:
            vals.append(max(float(self.field.phi_s(p)), self._interest_risk(p)))
        for a, b in zip(pts[:-1], pts[1:]):
            length += float(np.linalg.norm(b - a))
        mean_risk = float(np.mean(vals)) if vals else 0.0
        max_risk = float(np.max(vals)) if vals else 0.0
        return {
            "length": float(length),
            "risk_per_meter": mean_risk,
            "max_risk": max_risk,
            "waypoints": pts,
        }

    def _candidate_baseline_similarity(self, corridor, baseline_ref):
        stats = self._candidate_baseline_stats(corridor, baseline_ref)
        return float(stats.get("similarity", 0.0))

    def _candidate_baseline_stats(self, corridor, baseline_ref):
        baseline = np.asarray(baseline_ref.get("waypoints", []), float)
        pts = np.asarray(getattr(corridor, "waypoints", []), float)
        if len(baseline) == 0 or len(pts) == 0:
            return {
                "mean_distance": 0.0,
                "max_lateral_offset": 0.0,
                "side": 0,
                "similarity": 0.0,
            }
        samples = min(24, max(len(baseline), len(pts)))
        base = self._resample(baseline, max_points=samples)
        cand = self._resample(pts, max_points=samples)
        if len(base) == 0 or len(cand) == 0:
            return {
                "mean_distance": 0.0,
                "max_lateral_offset": 0.0,
                "side": 0,
                "similarity": 0.0,
            }
        n = min(len(base), len(cand))
        dist = np.linalg.norm(base[:n, :2] - cand[:n, :2], axis=1)
        mean_dist = float(np.mean(dist)) if len(dist) else float("inf")
        start2 = base[0, :2]
        goal2 = base[-1, :2]
        axis = goal2 - start2
        axis_len = float(np.linalg.norm(axis))
        signed = []
        if axis_len > 1e-9:
            for p in cand[:, :2]:
                rel = p - start2
                signed.append(float((axis[0] * rel[1] - axis[1] * rel[0]) / axis_len))
        max_offset = float(np.max(np.abs(signed))) if signed else 0.0
        mean_signed = float(np.mean(signed)) if signed else 0.0
        side = 1 if mean_signed > 0.05 else -1 if mean_signed < -0.05 else 0
        scale = max(self.grid_resolution * 4.0, 0.10 * float(
            baseline_ref.get("length", 0.0)), 1e-6)
        return {
            "mean_distance": float(mean_dist),
            "max_lateral_offset": max_offset,
            "side": int(side),
            "similarity": float(max(0.0, 1.0 - mean_dist / scale)),
        }

    def _assign_topology_class(self, corridor, start, goal):
        semantics = set(str(x) for x in getattr(
            corridor, "semantic_sequence", semantic_sequence(corridor)))
        side = self._corridor_side(corridor, self._plane(start), self._plane(goal))
        pts = np.asarray(getattr(corridor, "waypoints", []), float)
        max_offset = 0.0
        if len(pts):
            start2 = self._plane(start)
            goal2 = self._plane(goal)
            axis = goal2 - start2
            axis_len = float(np.linalg.norm(axis))
            if axis_len > 1e-9:
                offsets = [
                    abs(float((axis[0] * (p[1] - start2[1]) -
                               axis[1] * (p[0] - start2[0])) / axis_len))
                    for p in pts[:, :2]
                ]
                max_offset = float(max(offsets)) if offsets else 0.0
        if self.topology_profile == "wheelchair":
            if any(sem in ("parking", "waiting") for sem in semantics):
                return "parking_approach"
            if max_offset < max(0.10, 1.5 * self.grid_resolution):
                return "center_passage"
            return "left_bypass" if side > 0 else "right_bypass" if side < 0 else "center_passage"
        if self.topology_profile == "arm":
            if "handover" in semantics:
                return "handover_approach"
            if max_offset >= max(0.06, 1.5 * self.grid_resolution):
                return "head_avoidance_passage"
            return "center_passage"
        if max_offset < max(0.10, 1.5 * self.grid_resolution):
            return "center_passage"
        return "left_bypass" if side > 0 else "right_bypass" if side < 0 else "center_passage"

    def _required_task_minima_types(self):
        if not self.task_minima_points:
            return []
        if self.topology_profile == "arm":
            return ["handover"]
        if self.topology_profile == "wheelchair":
            return ["parking", "waiting"]
        return []

    def _candidate_hard_filter_reason(self, corridor, baseline_ref):
        source = str(getattr(corridor, "source", "morse_topology_graph"))
        role = str(getattr(corridor, "candidate_generation_role", ""))
        label = str(getattr(corridor, "label", ""))
        if "fallback" in source or "fallback" in role or "fallback" in label:
            return "candidate_fallback_forbidden"
        if not bool(getattr(corridor, "morse_induced", False)):
            return "candidate_not_from_morse_graph"
        morse_types = list(getattr(corridor, "morse_node_types", []))
        route_level = str(getattr(corridor, "route_generation_level", ""))
        if not morse_types:
            return "candidate_not_from_morse_graph"
        if (not any(t == "saddle" for t in morse_types) and
                route_level != "critical_sequence"):
            return "candidate_missing_valid_saddle"
        required = self._required_task_minima_types()
        if required:
            semantics = set(str(x) for x in getattr(
                corridor, "semantic_sequence", semantic_sequence(corridor)))
            minima_type = str(getattr(corridor, "minima_semantic_type", ""))
            if minima_type:
                semantics.add(minima_type)
            if not any(req in sem for req in required for sem in semantics):
                if self.topology_profile == "arm":
                    return "candidate_missing_handover_minima"
                return "candidate_missing_parking_waiting_minima"
        risk_per_meter = float(getattr(corridor, "mean_phi_on_path", 0.0))
        baseline_rpm = float(baseline_ref.get("risk_per_meter", 0.0))
        corridor.risk_per_meter = risk_per_meter
        corridor.baseline_risk_per_meter = baseline_rpm
        corridor.baseline_length = float(baseline_ref.get("length", 0.0))
        corridor.baseline_max_risk = float(baseline_ref.get("max_risk", 0.0))
        corridor.risk_gain = float(baseline_rpm - risk_per_meter)
        if baseline_rpm > 1e-9 and risk_per_meter > max(
                1.50 * baseline_rpm, baseline_rpm + 0.25):
            return "candidate_risk_per_meter_over_baseline"
        base_stats = self._candidate_baseline_stats(corridor, baseline_ref)
        similarity = float(base_stats["similarity"])
        corridor.baseline_similarity = similarity
        corridor.mean_distance_to_baseline = float(base_stats["mean_distance"])
        corridor.max_lateral_offset = float(base_stats["max_lateral_offset"])
        corridor.baseline_side = int(base_stats["side"])
        risk_gain = float(corridor.risk_gain)
        max_gain = float(baseline_ref.get("max_risk", 0.0)) - float(
            getattr(corridor, "max_phi_on_path", 0.0))
        if (float(base_stats["mean_distance"]) < 0.10 and
                risk_gain < 0.10 * max(baseline_rpm, 1e-6) and
                max_gain <= 0.0):
            return "similar_to_baseline_no_risk_gain"
        return ""

    def _kinematic_reject_reason(self, corridor, baseline_ref=None):
        corridor.corridor_self_intersection = bool(
            self._path_self_intersection(corridor.waypoints))
        corridor.corridor_max_turn = float(getattr(corridor, "max_turn_angle", 0.0))
        corridor.corridor_max_curvature = float(getattr(corridor, "max_curvature", 0.0))
        corridor.corridor_kinematic_valid = 1
        corridor.smoothing_preserved_saddle = 1
        if corridor.corridor_self_intersection:
            corridor.corridor_kinematic_valid = 0
            return "candidate_self_intersection"
        if corridor.corridor_max_turn > self.max_corridor_turn + 1e-9:
            corridor.corridor_kinematic_valid = 0
            return "candidate_turn_limit"
        if corridor.corridor_max_curvature > self.max_corridor_curvature + 1e-9:
            corridor.corridor_kinematic_valid = 0
            return "candidate_curvature_limit"
        max_risk = float(getattr(corridor, "max_phi_on_path", 0.0))
        if max_risk > self.candidate_max_risk + 1e-9:
            corridor.corridor_kinematic_valid = 0
            return "candidate_risk_over_rho"
        if (self.min_segment_length > 0.0 and
                float(getattr(corridor, "min_segment_length", 0.0)) <
                self.min_segment_length):
            corridor.corridor_kinematic_valid = 0
            return "candidate_segment_too_short"
        if self.require_risk_improvement and baseline_ref:
            risk_per_meter = float(getattr(corridor, "mean_phi_on_path", 0.0))
            length = float(getattr(corridor, "path_length", 0.0))
            corridor.baseline_risk_per_meter = float(
                baseline_ref.get("risk_per_meter", 0.0))
            corridor.baseline_max_risk = float(baseline_ref.get("max_risk", 0.0))
            corridor.baseline_length = float(baseline_ref.get("length", 0.0))
            corridor.risk_not_improved = int(
                risk_per_meter >= corridor.baseline_risk_per_meter or
                max_risk >= corridor.baseline_max_risk)
            corridor.length_over_baseline = int(
                corridor.baseline_length > 1e-9 and
                length > 1.3 * corridor.baseline_length)
        return ""

    def _public_node_type(self, kind):
        kind = str(kind or "")
        if kind == "minimum":
            return "minima"
        return kind

    def _corridor_node_type_sequence(self, corridor):
        seq = []
        for kind in list(getattr(corridor, "topology_node_kinds", [])):
            seq.append(self._public_node_type(kind))
        if seq:
            return seq
        nodes = list(getattr(corridor, "topology_nodes", []))
        kinds = list(getattr(corridor, "topology_kinds", []))
        if not nodes:
            return []
        out = []
        middle = list(kinds)
        for idx, node_id in enumerate(nodes):
            if idx == 0:
                out.append("start")
            elif idx == len(nodes) - 1:
                out.append("goal")
            else:
                out.append(self._public_node_type(
                    middle[idx - 1] if idx - 1 < len(middle) else "unknown"))
        return out

    def _candidate_record(self, corridor, selected=False, reject_reason=""):
        default_status = "feasible"
        if reject_reason:
            default_status = (
                "manifold_infeasible"
                if str(reject_reason) == "manifold_infeasible" else
                "infeasible")
        manifold_status = str(getattr(
            corridor, "candidate_status",
            default_status))
        manifold_check = dict(getattr(
            corridor, "manifold_feasibility", {}) or {})
        return {
            "corridor_id": str(getattr(corridor, "corridor_id", "")),
            "candidate_id": str(getattr(
                corridor, "candidate_id", getattr(corridor, "corridor_id", ""))),
            "label": str(getattr(corridor, "label", "")),
            "source": str(getattr(corridor, "source", "")),
            "topology_source": str(getattr(
                corridor, "topology_source",
                getattr(corridor, "source", ""))),
            "route_source": str(getattr(corridor, "route_source", "")),
            "candidate_source": str(getattr(
                corridor, "candidate_source",
                getattr(corridor, "route_source", ""))),
            "route_generation_level": str(getattr(
                corridor, "route_generation_level", "")),
            "generation_method": str(getattr(
                corridor, "candidate_generation_method", "")),
            "candidate_generation_role": str(getattr(
                corridor, "candidate_generation_role", "")),
            "critical_point_sequence": list(getattr(
                corridor, "critical_point_sequence", [])),
            "centerline": np.asarray(
                getattr(corridor, "centerline",
                        getattr(corridor, "waypoints", [])), float).tolist(),
            "boundary": getattr(corridor, "boundary", {}),
            "corridor_width_profile": list(getattr(
                corridor, "corridor_width_profile",
                getattr(corridor, "boundary", {}).get(
                    "corridor_width_profile",
                    getattr(corridor, "boundary", {}).get("width", [])))),
            "manifold_adaptive": bool(getattr(
                corridor, "manifold_adaptive", False)),
            "risk_adaptive_width": bool(getattr(
                corridor, "risk_adaptive_width", False)),
            "manifold_valid": bool(getattr(corridor, "manifold_valid", True)),
            "manifold_feasible": bool(getattr(
                corridor, "manifold_feasible",
                manifold_status == "feasible")),
            "candidate_manifold_valid": bool(getattr(
                corridor, "manifold_feasible",
                manifold_status == "feasible")),
            "candidate_tube_valid": bool(manifold_check.get(
                "candidate_tube_valid",
                manifold_check.get("tube_valid", manifold_status == "feasible"))),
            "topology_corridor_recovery_used": bool(getattr(
                corridor, "topology_recovery_used",
                getattr(corridor, "recovery_used", False))),
            "candidate_recovery_mode": str(getattr(
                corridor, "candidate_recovery_mode",
                manifold_check.get("candidate_recovery_mode", ""))),
            "candidate_recovered": bool(getattr(
                corridor, "candidate_recovered", False)),
            "before_clearance": float(getattr(
                corridor, "before_clearance", manifold_check.get(
                    "before_clearance", 0.0))),
            "after_clearance": float(getattr(
                corridor, "after_clearance", manifold_check.get(
                    "after_clearance", 0.0))),
            "recovery_success": bool(getattr(
                corridor, "recovery_success", False)),
            "candidate_recovery_iterations": int(getattr(
                corridor, "candidate_recovery_iterations", 0)),
            "adaptive_corridor_width": bool(getattr(
                corridor, "adaptive_corridor_width", False)),
            "clearance_optimization_used": bool(getattr(
                corridor, "clearance_optimization_used", False)),
            "tube_valid": bool(manifold_check.get(
                "tube_valid", manifold_status == "feasible")),
            "min_tube_clearance": float(manifold_check.get(
                "min_tube_clearance", 0.0)),
            "candidate_status": manifold_status,
            "failure_reason": str(getattr(
                corridor, "failure_reason", reject_reason or "")),
            "manifold_feasibility": manifold_check,
            "trajectory_min_clearance": float(getattr(
                corridor, "trajectory_min_clearance",
                manifold_check.get(
                    "trajectory_min_clearance",
                    manifold_check.get("min_clearance", getattr(
                        corridor, "min_corridor_clearance",
                        getattr(corridor, "min_clearance", 0.0)))))),
            "trajectory_max_risk": float(getattr(
                corridor, "trajectory_max_risk",
                manifold_check.get(
                    "trajectory_max_risk",
                    manifold_check.get("max_risk", getattr(
                        corridor, "max_phi_on_path", 0.0))))),
            "planning_clearance_margin": float(getattr(
                corridor, "planning_clearance_margin",
                manifold_check.get(
                    "planning_clearance_margin",
                    getattr(self, "planning_clearance_margin", 0.0)))),
            "min_corridor_clearance": float(getattr(
                corridor, "min_corridor_clearance",
                getattr(corridor, "min_clearance", 0.0))),
            "average_corridor_width": float(getattr(
                corridor, "average_corridor_width", 0.0)),
            "manifold_validation": dict(getattr(
                corridor, "manifold_validation", {}) or {}),
            "topology_valid": bool(getattr(corridor, "topology_valid", True)),
            "node_sequence": list(getattr(corridor, "node_sequence", [])),
            "node_type_sequence": self._corridor_node_type_sequence(corridor),
            "semantic_sequence": list(getattr(
                corridor, "semantic_sequence", semantic_sequence(corridor))),
            "topology_class": str(getattr(corridor, "topology_class", "")),
            "topology_route_class": str(getattr(
                corridor, "topology_route_class",
                getattr(corridor, "topology_class", ""))),
            "task_semantic_class": str(getattr(
                corridor, "task_semantic_class", "")),
            "source_graph_id": str(getattr(corridor, "source_graph_id", "")),
            "source_saddle_ids": list(getattr(corridor, "source_saddle_ids", [])),
            "source_minima_ids": list(getattr(corridor, "source_minima_ids", [])),
            "waypoints": np.asarray(
                getattr(corridor, "waypoints", []), float).tolist(),
            "topology_ordered_waypoints": np.asarray(
                getattr(corridor, "topology_ordered_waypoints", []),
                float).tolist(),
            "raw_topology_waypoints": np.asarray(
                getattr(corridor, "raw_topology_waypoints", []), float).tolist(),
            "refined_waypoints": np.asarray(
                getattr(corridor, "refined_waypoints", []), float).tolist(),
            "refinement_used": int(getattr(corridor, "refinement_used", 0)),
            "refinement_reject_reason": str(getattr(
                corridor, "refinement_reject_reason", "")),
            "refinement_manifold_checked": bool(getattr(
                corridor, "refinement_manifold_checked", False)),
            "refinement_manifold_valid": bool(getattr(
                corridor, "refinement_manifold_valid", False)),
            "pre_refinement_clearance": float(getattr(
                corridor, "pre_refinement_clearance", 0.0)),
            "post_refinement_clearance": float(getattr(
                corridor, "post_refinement_clearance", 0.0)),
            "refinement_fallback": bool(getattr(
                corridor, "refinement_fallback", False)),
            "trajectory_manifold_violation_count": int(getattr(
                corridor, "trajectory_manifold_violation_count", 0)),
            "refinement_trace": list(getattr(
                corridor, "refinement_trace", [])),
            "refined_path_length": float(getattr(
                corridor, "refined_path_length", 0.0)),
            "refined_max_turn_angle": float(getattr(
                corridor, "refined_max_turn_angle", 0.0)),
            "refined_mean_turn_angle": float(getattr(
                corridor, "refined_mean_turn_angle", 0.0)),
            "refined_max_curvature": float(getattr(
                corridor, "refined_max_curvature", 0.0)),
            "path_length": float(getattr(corridor, "path_length", 0.0)),
            "base_cost": float(getattr(corridor, "base_cost", 0.0)),
            "cost": float(getattr(corridor, "cost", 0.0)),
            "risk_cost": float(getattr(corridor, "risk_cost", 0.0)),
            "length_cost": float(getattr(
                corridor, "length_cost",
                getattr(corridor, "distance_cost", 0.0))),
            "smoothness_cost": float(getattr(
                corridor, "smoothness_cost",
                getattr(corridor, "smooth_cost", 0.0))),
            "task_cost": float(getattr(corridor, "task_cost", 0.0)),
            "task_state": str(getattr(corridor, "task_state", "")),
            "task_cost_breakdown": dict(getattr(
                corridor, "task_cost_breakdown", {}) or {}),
            "clearance_cost": float(getattr(corridor, "clearance_cost", 0.0)),
            "feasibility_cost": float(getattr(
                corridor, "feasibility_cost", 0.0)),
            "task_specific_cost": float(getattr(
                corridor, "task_specific_cost", 0.0)),
            "raw_recovery_cost": float(getattr(
                corridor, "raw_recovery_cost",
                getattr(corridor, "recovery_cost", 0.0))),
            "recovery_cost": float(getattr(corridor, "recovery_cost", 0.0)),
            "recovery_weight": float(
                (getattr(corridor, "candidate_cost_breakdown", {}) or {}).get(
                    "recovery_weight", 0.3)),
            "normalized_recovery_cost": float(getattr(
                corridor, "normalized_recovery_cost",
                getattr(corridor, "recovery_cost", 0.0))),
            "recoverable_level": str(getattr(
                corridor, "recoverable_level",
                _arm_recoverable_level(getattr(
                    corridor, "normalized_recovery_cost",
                    getattr(corridor, "recovery_cost", 0.0))))),
            "candidate_cost": float(getattr(
                corridor, "candidate_cost",
                getattr(corridor, "total_score",
                        getattr(corridor, "cost", 0.0)))),
            "candidate_cost_breakdown": dict(getattr(
                corridor, "candidate_cost_breakdown", {}) or {}),
            "task_candidate_cost": float(getattr(
                corridor, "task_candidate_cost", 0.0)),
            "task_mode": str(getattr(corridor, "task_mode",
                                     getattr(self, "task_mode", ""))),
            "task_weight": dict(getattr(
                corridor, "task_weight", getattr(self, "task_weight", {})) or {}),
            "task_weight_used": bool(getattr(
                corridor, "task_weight_used", True)),
            "risk_norm": float(getattr(corridor, "risk_norm", 0.0)),
            "length_norm": float(getattr(corridor, "length_norm", 0.0)),
            "smooth_norm": float(getattr(corridor, "smooth_norm", 0.0)),
            "task_norm": float(getattr(corridor, "task_norm", 0.0)),
            "execution_norm": float(getattr(corridor, "execution_norm", 0.0)),
            "execution_cost": float(getattr(corridor, "execution_cost", 0.0)),
            "topology_diversity": float(getattr(
                corridor, "topology_diversity", 0.0)),
            "total_score": float(getattr(
                corridor, "total_score",
                getattr(corridor, "total_cost", getattr(corridor, "cost", 0.0)))),
            "ranking_score": float(getattr(
                corridor, "total_score",
                getattr(corridor, "total_cost", getattr(corridor, "cost", 0.0)))),
            "total_cost": float(getattr(
                corridor, "total_cost", getattr(corridor, "cost", 0.0))),
            "topology_cost": float(getattr(corridor, "topology_cost", 0.0)),
            "distance_cost": float(getattr(corridor, "distance_cost", 0.0)),
            "smooth_cost": float(getattr(corridor, "smooth_cost", 0.0)),
            "motion_cost": float(getattr(corridor, "motion_cost", 0.0)),
            "execution_cost": float(getattr(corridor, "execution_cost", 0.0)),
            "curvature_cost": float(getattr(corridor, "curvature_cost", 0.0)),
            "topology_value": float(getattr(corridor, "topology_value", 0.0)),
            "mean_phi_on_path": float(getattr(corridor, "mean_phi_on_path", 0.0)),
            "max_phi_on_path": float(getattr(corridor, "max_phi_on_path", 0.0)),
            "risk_per_meter": float(getattr(corridor, "risk_per_meter", 0.0)),
            "baseline_risk_per_meter": float(getattr(
                corridor, "baseline_risk_per_meter", 0.0)),
            "baseline_similarity": float(getattr(
                corridor, "baseline_similarity", 0.0)),
            "mean_distance_to_baseline": float(getattr(
                corridor, "mean_distance_to_baseline", 0.0)),
            "max_lateral_offset": float(getattr(
                corridor, "max_lateral_offset", 0.0)),
            "risk_gain": float(getattr(corridor, "risk_gain", 0.0)),
            "selected": bool(selected or getattr(corridor, "selected", 0)),
            "geometry_duplicate_retained": bool(getattr(
                corridor, "geometry_duplicate_retained", False)),
            "diversity_retention_reason": str(getattr(
                corridor, "diversity_retention_reason", "")),
            "selected_reason": (
                "lowest_candidate_cost_after_task_safety_feasibility_ranking"
                if bool(selected or getattr(corridor, "selected", 0)) else ""),
            "candidate_filter_class": str(getattr(
                corridor, "candidate_filter_class",
                getattr(corridor, "candidate_status", ""))),
            "task_mode_influence": {
                "task_mode": str(getattr(corridor, "task_mode",
                                         getattr(self, "task_mode", ""))),
                "task_cost": float(getattr(corridor, "task_cost", 0.0)),
                "task_candidate_cost": float(getattr(
                    corridor, "task_candidate_cost", 0.0)),
                "task_weight": dict(getattr(
                    corridor, "task_weight",
                    getattr(self, "task_weight", {})) or {}),
            },
            "reject_reason": str(reject_reason or getattr(corridor, "reject_reason", "")),
            "morse_node_ids": list(getattr(corridor, "morse_node_ids", [])),
            "morse_node_types": [
                self._public_node_type(t)
                for t in list(getattr(corridor, "morse_node_types", []))
            ],
            "morse_induced": bool(getattr(corridor, "morse_induced", False)),
        }

    def evaluate_corridor(self, corridor, node_ids=None, graph_search_cost=0.0,
                          start=None, goal=None):
        return self._corridor_score(
            corridor,
            list(node_ids or getattr(corridor, "topology_nodes", [])),
            graph_search_cost,
            start=start,
            goal=goal)

    def _corridor_points(self, corridor):
        pts = np.asarray(
            getattr(corridor, "refined_waypoints",
                    getattr(corridor, "waypoints", [])), float)
        if pts.size == 0:
            pts = np.asarray(getattr(corridor, "waypoints", []), float)
        if pts.size == 0:
            return np.zeros((0, 3), float)
        if pts.ndim == 1:
            pts = pts.reshape((1, pts.shape[0]))
        if pts.shape[1] == 2:
            pts = np.hstack([pts, np.zeros((pts.shape[0], 1), float)])
        return pts[:, :3]

    def _final_alignment_cost(self, points, goal):
        if goal is None or len(points) < 2:
            return 0.0
        final_vec = points[-1, :2] - points[-2, :2]
        goal_vec = np.asarray(goal, float)[:2] - points[-2, :2]
        denom = float(np.linalg.norm(final_vec) * np.linalg.norm(goal_vec))
        if denom <= 1e-9:
            return 0.0
        align = float(np.dot(final_vec, goal_vec) / denom)
        return float(max(0.0, 1.0 - np.clip(align, -1.0, 1.0)))

    def _clearance_cost(self, corridor):
        minimum = float(getattr(corridor, "trajectory_min_clearance",
                                getattr(corridor, "min_clearance", 0.0)))
        required = max(
            float(getattr(corridor, "required_clearance", 0.0)),
            float(getattr(corridor, "planning_clearance_margin", 0.0)),
            float(getattr(corridor, "min_corridor_clearance", 0.0)))
        deficit = max(0.0, required - minimum)
        margin_cost = 0.0 if required <= 1e-9 else deficit / max(required, 1e-6)
        proximity_cost = 1.0 / (1.0 + max(0.0, minimum))
        return float(margin_cost + 0.25 * proximity_cost)

    def _feasibility_cost(self, corridor):
        cost = 0.0
        if not bool(getattr(corridor, "manifold_feasible", True)):
            cost += 10.0
        if not bool(getattr(corridor, "candidate_tube_valid", True)):
            cost += 5.0
        if self.topology_profile == "arm":
            if not bool(getattr(corridor, "ik_valid", True)):
                cost += 3.0
            if not bool(getattr(corridor, "link_collision_valid", True)):
                cost += 2.0
        return float(cost)

    def _task_specific_candidate_terms(self, corridor, points, start=None,
                                       goal=None):
        terms = {}
        if self.topology_profile == "arm":
            target = np.asarray(goal if goal is not None else points[-1], float)[:3]
            end = points[-1] if len(points) else target
            handover_distance = float(np.linalg.norm(end[:3] - target[:3]))
            reachability_cost = 0.0 if bool(getattr(corridor, "ik_valid", True)) else 3.0
            approach_direction_cost = self._final_alignment_cost(points, target)
            terms.update({
                "handover_distance_cost": handover_distance,
                "end_effector_reachability_cost": float(reachability_cost),
                "approach_direction_cost": float(approach_direction_cost),
            })
        elif self.topology_profile == "wheelchair":
            widths = list(getattr(corridor, "corridor_width_profile", []) or [])
            if not widths:
                boundary = getattr(corridor, "boundary", {}) or {}
                widths = list(boundary.get(
                    "corridor_width_profile", boundary.get("width", [])) or [])
            width_min = float(np.min(np.asarray(widths, float))) if widths else 0.0
            required = max(float(self.min_clearance), float(self.hard_clearance), 1e-6)
            passage_width_cost = max(0.0, required - width_min) / required
            turning_cost = float(getattr(corridor, "max_turn_angle", 0.0))
            goal_approach_cost = (
                float(np.linalg.norm(points[-1, :2] - np.asarray(goal, float)[:2]))
                if goal is not None and len(points) else 0.0)
            terms.update({
                "passage_width_cost": float(passage_width_cost),
                "turning_smoothness_cost": float(turning_cost),
                "goal_approach_cost": float(goal_approach_cost),
            })
        return terms

    def _saddle_value_bonus(self, kinds, base_cost):
        if getattr(self, "morse_decision_mode", "balanced") == "balanced":
            return 0.0
        if self.lambda_saddle_value <= 0.0:
            return 0.0
        kinds = list(kinds or [])
        has_saddle = any(kind == "saddle" for kind in kinds)
        if not has_saddle:
            return 0.0
        has_minimum = any(kind == "minimum" for kind in kinds)
        role_weight = 0.80 if has_minimum else 1.0
        scale = max(1.0, abs(float(base_cost)))
        return float(self.lambda_saddle_value * role_weight * scale)

    def _neighbor_pairs(self, nodes, k=None):
        if k is None:
            k = self.neighbor_k
        strategy = self.neighbor_strategy
        if strategy == "dense_projected_3d":
            k = max(int(k), 12)
        elif strategy == "sparse_body":
            k = min(int(k), 8)
        else:
            k = int(k)
        pairs = set()
        safe_nodes = [(idx, node) for idx, node in enumerate(nodes)
                      if node.ij is not None]
        ids = {node.id: idx for idx, node in enumerate(nodes)}
        if "start" in ids and "goal" in ids:
            pairs.add(tuple(sorted((ids["start"], ids["goal"]))))
        sparse_radius = max(3.0 * self.merge_radius, 6.0 * self.grid_resolution)
        dense_radius = max(4.0 * self.merge_radius, 8.0 * self.grid_resolution)
        for idx, node in safe_nodes:
            dists = []
            for jdx, other in safe_nodes:
                if idx == jdx:
                    continue
                dist = float(np.linalg.norm(node.point - other.point))
                endpoint_or_semantic = (
                    node.kind in ("start", "goal") or
                    other.kind in ("start", "goal") or
                    node.kind == "semantic" or other.kind == "semantic")
                if (strategy == "sparse_body" and not endpoint_or_semantic and
                        dist > sparse_radius):
                    continue
                priority = 0 if (
                    endpoint_or_semantic
                ) else 1
                dists.append((priority, dist, jdx))
            dists.sort()
            for _priority, _dist, jdx in dists[:int(k)]:
                pairs.add(tuple(sorted((idx, jdx))))
            if strategy == "dense_projected_3d":
                for _priority, dist, jdx in dists[int(k):]:
                    if dist > dense_radius:
                        break
                    pairs.add(tuple(sorted((idx, jdx))))
        return sorted(pairs)

    def _ring_nodes(self, center2, label, count=6):
        nodes = []
        radius = max(self.merge_radius, self.grid_resolution * 2.0)
        (xmin, xmax), (ymin, ymax) = self.bounds
        for idx, theta in enumerate(np.linspace(0.0, 2.0 * math.pi, int(count), endpoint=False)):
            p2 = np.asarray(center2, float) + radius * np.array(
                [math.cos(theta), math.sin(theta)], float)
            p2[0] = min(max(p2[0], xmin), xmax)
            p2[1] = min(max(p2[1], ymin), ymax)
            ij = self._nearest_safe_ij(p2)
            if ij is None:
                continue
            safe_p2 = self._ij_to_p2(ij)
            duplicate = False
            for old in nodes:
                if old.ij == ij or np.linalg.norm(old.point - self._world(safe_p2)) < self.merge_radius:
                    duplicate = True
                    break
            if duplicate:
                continue
            nodes.append(TopologyNode("%s_ring_%d" % (label, idx),
                                      "%s_ring" % label,
                                      self._world(safe_p2), ij))
        return nodes

    def build_graph(self, start, goal, critical=None, semantic_nodes=None):
        if self._grid is None:
            self.build_grid(goal)
        critical = critical if critical is not None else self.detect_morse_points()
        critical_chain = critical.get("_critical_chain", {})

        def mark_critical(kind_plural, item, stage, status, reason="", **extra):
            rec_id = item.get("id", "")
            if not rec_id:
                return
            for rec in critical_chain.get(kind_plural, []):
                if rec.get("id", "") != rec_id:
                    continue
                rec["stage"] = stage
                rec["status"] = status
                rec["reason"] = reason
                rec.update(extra)
                break

        semantic_nodes = semantic_nodes or []
        has_morse_critical = bool(critical.get("saddles") or critical.get("minima"))
        semantic_suppressed = 0
        ring_suppressed = 0
        if self.morse_primary and has_morse_critical and not self.allow_semantic_with_morse:
            semantic_suppressed = len(semantic_nodes)
            semantic_nodes = []
        start2 = self._plane(start)
        goal2 = self._plane(goal)
        nodes = [
            TopologyNode(
                "start", "start", self._world(start2),
                self._nearest_safe_ij(start2),
                semantic_type=node_semantic_type(
                    self.topology_profile, "start")),
            TopologyNode(
                "goal", "goal", self._world(goal2),
                self._nearest_safe_ij(goal2),
                semantic_type=node_semantic_type(
                    self.topology_profile, "goal")),
        ]
        start_assoc = self.associate_pose_to_topology_node(
            start, nodes=nodes, preferred_id="start")
        goal_assoc = self.associate_pose_to_topology_node(
            goal, nodes=nodes, preferred_id="goal")
        self.last_debug["start_node"] = str(start_assoc.get("node_id") or "")
        self.last_debug["goal_node"] = str(goal_assoc.get("node_id") or "")
        self.last_debug["topology_pose_association"] = {
            "start": start_assoc,
            "goal": goal_assoc,
        }
        idx = 0

        def add_node(node):
            if node.ij is None:
                return False
            if len(nodes) >= self.max_graph_nodes:
                return False
            if any(n.ij == node.ij for n in nodes if n.ij is not None):
                return False
            nodes.append(node)
            return True

        for kind, node_kind, quota in (
                ("saddles", "saddle", self.max_saddle_nodes),
                ("minima", "minimum", self.max_minima_nodes)):
            added = 0
            items = list(critical.get(kind, []))
            for item_idx, item in enumerate(items):
                if added >= quota:
                    for rest in items[item_idx:]:
                        mark_critical(
                            kind, rest, "node", "rejected",
                            "node_quota_exceeded", quota=int(quota))
                    break
                ij = item.get("ij")
                if ij is None or not self._grid["safe"][ij]:
                    ij = self._nearest_safe_ij(item.get("p2", self._plane(item["point"])))
                point = self._world(self._ij_to_p2(ij)) if ij is not None else item["point"]
                if ij is None:
                    mark_critical(kind, item, "node", "rejected", "no_safe_cell",
                                  safety={"stage": "node", "action": "rejected",
                                          "reason": "no_safe_cell"})
                    continue
                node = TopologyNode(
                    "%s_%d" % (node_kind, idx),
                    node_kind, point, ij,
                    semantic_type=node_semantic_type(
                        self.topology_profile, node_kind,
                        "%s_%d" % (node_kind, idx)))
                node.critical_id = item.get("id", "")
                if add_node(node):
                    mark_critical(
                        kind, item, "node", "node_added", "",
                        node_id=node.id,
                        safety=self._grid_safety_record(
                            ij, "node", "kept", node_id=node.id))
                    added += 1
                    idx += 1
                else:
                    reason = (
                        "max_graph_nodes" if len(nodes) >= self.max_graph_nodes
                        else "duplicate_node")
                    mark_critical(
                        kind, item, "node", "rejected", reason,
                        safety=self._grid_safety_record(
                            ij, "node", "rejected", reason, node_id=node.id))
        semantic_added = 0
        for p in semantic_nodes:
            if semantic_added >= self.max_semantic_nodes:
                break
            p2 = self._plane(p)
            if add_node(TopologyNode(
                    "semantic_%d" % idx, "semantic",
                    self._world(p2), self._nearest_safe_ij(p2),
                    semantic_type=node_semantic_type(
                        self.topology_profile, "semantic"))):
                semantic_added += 1
                idx += 1
        ring_added = 0
        ring_candidates = self._ring_nodes(start2, "start") + self._ring_nodes(goal2, "goal")
        if self.morse_primary and has_morse_critical and not self.allow_ring_with_morse:
            ring_suppressed = len(ring_candidates)
        else:
            for ring in ring_candidates:
                if ring_added >= self.max_ring_nodes:
                    break
                if add_node(ring):
                    ring_added += 1
        self.last_debug["semantic_nodes_added"] = int(semantic_added)
        self.last_debug["ring_nodes_added"] = int(ring_added)
        self.last_debug["semantic_nodes_suppressed_by_morse"] = int(semantic_suppressed)
        self.last_debug["ring_nodes_suppressed_by_morse"] = int(ring_suppressed)

        edges = {}
        edge_audit = []
        edge_audit_counts = {
            "kept": 0,
            "reject_astar": 0,
            "reject_clearance": 0,
            "reject_forbidden": 0,
        }
        attempted_pairs = 0
        astar_fail_count = 0
        for i, j in self._neighbor_pairs(nodes):
            attempted_pairs += 1
            a = nodes[i]
            b = nodes[j]
            if a.ij is None or b.ij is None:
                continue
            cell_path, astar_cost = self._astar(a.ij, b.ij)
            if not cell_path:
                astar_fail_count += 1
                edge_audit_counts["reject_astar"] += 1
                if len(edge_audit) < 40:
                    edge_audit.append({
                        "stage": "edge",
                        "action": "rejected",
                        "reason": "astar_fail",
                        "edge": "%s->%s" % (a.id, b.id),
                    })
                continue
            edge_clearance = self._cell_path_clearance(cell_path)
            edge_clearance_margin = self._cell_path_clearance_margin(cell_path)
            if edge_clearance_margin < 0.0:
                self._debug_inc("edge_clearance_reject_count")
                edge_audit_counts["reject_clearance"] += 1
                if len(edge_audit) < 40:
                    worst_ij = min(
                        cell_path,
                        key=lambda ij: self._clearance_cells(ij) -
                        self._hard_clearance_at_ij(ij))
                    edge_audit.append(self._grid_safety_record(
                        worst_ij, "edge", "rejected", "clearance",
                        edge="%s->%s" % (a.id, b.id),
                        path_clearance=edge_clearance,
                        clearance_margin=edge_clearance_margin))
                continue
            edge_forbidden_hits = self._cell_path_forbidden_hits(cell_path)
            if edge_forbidden_hits:
                self._debug_inc("edge_forbidden_reject_count")
                edge_audit_counts["reject_forbidden"] += 1
                if len(edge_audit) < 40:
                    hit_ij = next(
                        (ij for ij in cell_path
                         if bool(self._grid.get("forbidden")[ij])),
                        cell_path[0])
                    edge_audit.append(self._grid_safety_record(
                        hit_ij, "edge", "rejected", "forbidden",
                        edge="%s->%s" % (a.id, b.id),
                        forbidden_hits=edge_forbidden_hits))
                continue
            length, mean_phi, max_phi = self._edge_metrics(cell_path)
            clearance_penalty = self._cell_path_clearance_penalty(cell_path)
            cost = (length + 0.8 * mean_phi + 0.4 * max_phi +
                    0.15 * astar_cost + 8.0 * clearance_penalty)
            rec = {
                "to": b.id, "cost": float(cost), "cells": cell_path,
                "length": length, "mean_phi": mean_phi,
                "max_phi": max_phi,
                "clearance": edge_clearance,
                "clearance_margin": edge_clearance_margin,
                "clearance_penalty": clearance_penalty,
                "forbidden_hits": edge_forbidden_hits,
            }
            edges.setdefault(a.id, []).append(rec)
            rev = dict(rec)
            rev["to"] = a.id
            rev["cells"] = list(reversed(cell_path))
            edges.setdefault(b.id, []).append(rev)
            edge_audit_counts["kept"] += 1
            if len(edge_audit) < 40:
                min_ij = min(cell_path, key=lambda ij: self._clearance_cells(ij))
                edge_audit.append(self._grid_safety_record(
                    min_ij, "edge", "kept", "",
                    edge="%s->%s" % (a.id, b.id),
                    path_clearance=edge_clearance,
                    clearance_margin=edge_clearance_margin,
                    mean_phi=mean_phi,
                    max_phi=max_phi))
        self.last_debug["neighbor_pair_attempt_count"] = int(attempted_pairs)
        self.last_debug["edge_astar_fail_count"] = int(astar_fail_count)
        self.last_debug["safety_audit_edge_counts"] = edge_audit_counts
        self.last_debug["safety_audit_edges"] = edge_audit
        connectivity_before = self._graph_has_path(edges, "start", "goal")
        repair_report = self._repair_graph_connectivity(
            nodes, edges, "start", "goal")
        connectivity_after = self._graph_has_path(edges, "start", "goal")
        self.last_debug["topology_start_goal_connected_before_repair"] = bool(
            connectivity_before)
        self.last_debug["topology_start_goal_connected"] = bool(
            connectivity_after)
        self.last_debug["local_graph_repair"] = repair_report
        return nodes, edges, critical

    def _k_paths(self, edges, k):
        openq = [(0.0, ["start"], [])]
        paths = []
        seen = set()
        while openq and len(paths) < k:
            cost, nodes, cell_parts = heapq.heappop(openq)
            cur = nodes[-1]
            key = tuple(nodes)
            if key in seen:
                continue
            seen.add(key)
            if cur == "goal":
                paths.append((cost, nodes, cell_parts))
                continue
            for edge in edges.get(cur, []):
                nxt = edge["to"]
                if nxt in nodes:
                    continue
                heapq.heappush(openq, (
                    cost + edge["cost"],
                    nodes + [nxt],
                    cell_parts + [edge],
                ))
        return paths

    def corridor_from_morse_graph(self, start, goal, k=3, radius=0.35,
                                  critic=None, feature_builder=None,
                                  lambda_adp=0.0, feature_context=None,
                                  semantic_nodes=None):
        """Generate executable corridors from the Morse topology graph.

        Normal candidates are restricted to start -> saddle... -> goal.
        Generic graph search is recorded for diagnostics only; callers should
        use their outer fallback path when this method returns no corridors.
        """
        return self.enumerate_corridors(
            start, goal, k=k, radius=radius, critic=critic,
            feature_builder=feature_builder, lambda_adp=lambda_adp,
            feature_context=feature_context, semantic_nodes=semantic_nodes)

    def enumerate_corridors(self, start, goal, k=3, radius=0.35,
                            critic=None, feature_builder=None, lambda_adp=0.0,
                            feature_context=None, semantic_nodes=None):
        feature_context = feature_context or {}
        self.build_grid(goal)
        critical = self.detect_morse_points()
        critical = self._apply_semantic_topology_recovery(
            critical, start, goal, semantic_nodes=semantic_nodes)
        has_morse_critical = bool(critical.get("saddles") or critical.get("minima"))
        if semantic_nodes is None:
            if self.morse_primary and has_morse_critical:
                semantic_nodes = []
            else:
                semantic_nodes = self._default_semantic_nodes(start, goal)
        elif self.morse_primary and has_morse_critical and not self.allow_semantic_with_morse:
            self.last_debug["semantic_nodes_requested_with_morse"] = int(len(semantic_nodes))
        nodes, edges, critical = self.build_graph(
            start, goal, critical=critical, semantic_nodes=semantic_nodes)
        corridors = []
        node_by_id = {}
        for n in nodes:
            node_by_id[n.id] = n
        start2 = self._plane(start)
        goal2 = self._plane(goal)

        task_node_count = 0
        for item in self.task_minima_points:
            if isinstance(item, dict):
                point = item.get("position", item.get("point", None))
                role = str(item.get("type", "task"))
            else:
                point = item
                role = "task"
            if point is None:
                continue
            p2 = self._plane(point)
            ij = self._nearest_safe_ij(p2)
            if ij is None:
                continue
            existing = next(
                (n for n in nodes if n.ij is not None and n.ij == ij),
                None)
            if existing is not None and existing.kind == "minimum":
                existing.semantic_type = node_semantic_type(
                    self.topology_profile, "minimum", existing.id,
                    task_type=role)
                existing.task_minima_type = role
                continue
            if existing is not None:
                continue
            node = TopologyNode(
                "task_minimum_%d" % task_node_count,
                "minimum",
                self._world(self._ij_to_p2(ij)),
                ij,
                semantic_type=node_semantic_type(
                    self.topology_profile, "minimum",
                    "task_minimum_%d" % task_node_count,
                    task_type=role))
            node.critical_id = "task_minimum_%d" % task_node_count
            node.task_minima_type = role
            nodes.append(node)
            node_by_id[node.id] = node
            task_node_count += 1

        baseline_risk_ref = self._baseline_risk_reference(start, goal)
        critical_chain = critical.get("_critical_chain", {})
        corridor_audit = []
        corridor_audit_counts = {
            "kept": 0,
            "reject_empty": 0,
            "reject_clearance": 0,
            "reject_forbidden": 0,
        }
        candidate_before_filter = []

        def mark_critical_id(crit_id, stage, status, reason="", **extra):
            if not crit_id:
                return
            for kind in ("minima", "saddles"):
                for rec in critical_chain.get(kind, []):
                    if rec.get("id", "") != crit_id:
                        continue
                    rec["stage"] = stage
                    rec["status"] = status
                    rec["reason"] = reason
                    rec.update(extra)
                    return

        def mark_critical_nodes(node_ids, stage, status, reason="", **extra):
            for node_id in node_ids[1:-1]:
                node = node_by_id.get(node_id)
                if node is None:
                    continue
                crit_id = getattr(node, "critical_id", "")
                mark_critical_id(crit_id, stage, status, reason, **extra)

        last_append_reject = {"reason": ""}

        def append_corridor(cells, node_ids, base_cost, label=None, route=None):
            route = dict(route or {})
            last_append_reject["reason"] = ""
            arm_execution_valid = bool(
                self.topology_profile == "arm" and
                route.get("ik_valid", False) and
                route.get("link_collision_valid", False))
            if not cells:
                last_append_reject["reason"] = "empty_cells"
                corridor_audit_counts["reject_empty"] += 1
                return None
            route_centerline = route.get("centerline", [])
            try:
                route_centerline = np.asarray(route_centerline, float)
                if route_centerline.ndim == 1 and route_centerline.size:
                    route_centerline = route_centerline.reshape(
                        (1, route_centerline.shape[0]))
                if route_centerline.size and route_centerline.shape[1] == 2:
                    route_centerline = np.hstack([
                        route_centerline,
                        np.zeros((route_centerline.shape[0], 1), float)])
            except Exception:
                route_centerline = np.empty((0, 3), float)
            has_prebuilt_route_geometry = bool(len(route_centerline) >= 2)
            kinds = [node_by_id[n].kind for n in node_ids[1:-1]
                     if n in node_by_id]
            route_level = str(route.get("route_generation_level", ""))
            has_saddle_kind = any(kind == "saddle" for kind in kinds)
            has_critical_kind = any(kind in ("saddle", "minimum") for kind in kinds)
            if not has_critical_kind:
                last_append_reject["reason"] = "candidate_missing_critical_point"
                self._debug_inc("candidate_missing_valid_saddle_count")
                corridor_audit_counts["reject_missing_saddle"] = (
                    corridor_audit_counts.get("reject_missing_saddle", 0) + 1)
                return None
            if not has_saddle_kind and route_level != "critical_sequence":
                last_append_reject["reason"] = "candidate_missing_valid_saddle"
                self._debug_inc("candidate_missing_valid_saddle_count")
                corridor_audit_counts["reject_missing_saddle"] = (
                    corridor_audit_counts.get("reject_missing_saddle", 0) + 1)
                return None
            cell_pts = np.asarray([
                self._world(self._ij_to_p2(ij)) for ij in cells
            ], float)
            pts = (
                route_centerline.copy() if has_prebuilt_route_geometry else
                self._candidate_waypoints_from_nodes(
                    node_ids, node_by_id, cell_pts))
            if label is None:
                idx = len(corridors)
                if sum(1 for n in node_ids if "saddle" in n) > 1:
                    label = "morse_saddle_pair_%d" % idx
                elif any("saddle" in n for n in node_ids):
                    label = "morse_saddle_%d" % idx
                elif any("minimum" in n for n in node_ids):
                    label = "morse_minima_%d" % idx
                elif any(kind in ("semantic", "start_ring", "goal_ring") for kind in kinds):
                    label = "graph_semantic_%d" % idx
                else:
                    label = "graph_direct_%d" % idx
            corridor = Corridor(pts, radius, label, base_cost)
            if has_prebuilt_route_geometry:
                corridor.centerline = route_centerline.copy()
                corridor.raw_waypoints = route_centerline.copy()
            route_boundary = dict(route.get("boundary", {}) or {})
            if route_boundary:
                corridor.boundary = route_boundary
                corridor.corridor_width_profile = list(route.get(
                    "corridor_width_profile",
                    route_boundary.get("corridor_width_profile",
                                       route_boundary.get("width", []))))
                corridor.average_corridor_width = float(np.mean(
                    np.asarray(corridor.corridor_width_profile, float))
                    if corridor.corridor_width_profile else 0.0)
            corridor.base_cost = float(base_cost)
            corridor.topology_nodes = list(node_ids)
            corridor.topology_kinds = kinds
            corridor.topology_node_kinds = [
                node_by_id[n].kind if n in node_by_id else ""
                for n in node_ids
            ]
            corridor.topology_semantic_kinds = [
                str(getattr(node_by_id[n], "semantic_type", ""))
                for n in node_ids[1:-1]
                if n in node_by_id
            ]
            morse_nodes = [
                {
                    "id": n,
                    "type": node_by_id[n].kind,
                }
                for n in node_ids[1:-1]
                if n in node_by_id and node_by_id[n].kind in ("saddle", "minimum")
            ]
            corridor.morse_nodes = morse_nodes
            corridor.morse_node_ids = [m["id"] for m in morse_nodes]
            corridor.morse_node_types = [m["type"] for m in morse_nodes]
            corridor.node_sequence = list(corridor.topology_nodes)
            corridor.node_type_sequence = self._corridor_node_type_sequence(
                corridor)
            corridor.semantic_sequence = semantic_sequence(corridor)
            corridor.source = "morse_topology_graph"
            corridor.route_source = str(route.get(
                "route_source", "morse_topology"))
            corridor.candidate_source = str(route.get(
                "candidate_source", "morse_topology"))
            corridor.route_generation_level = route_level
            corridor.topology_class = self._assign_topology_class(
                corridor, start, goal)
            corridor.topology_route_class = topology_route_class(
                corridor, start, goal)
            corridor.task_semantic_class = task_semantic_class(
                corridor, self.topology_profile)
            corridor.source_graph_id = "morse_topology_graph"
            corridor.source_saddle_ids = [
                m["id"] for m in morse_nodes if m["type"] == "saddle"
            ]
            corridor.source_minima_ids = [
                m["id"] for m in morse_nodes if m["type"] == "minimum"
            ]
            corridor.topology_ordered_waypoints = np.asarray([
                node_by_id[n].point for n in node_ids if n in node_by_id
            ], float)
            corridor.raw_waypoints = np.asarray(pts, float).copy()
            if not has_prebuilt_route_geometry:
                corridor.centerline = np.asarray(cell_pts, float).copy()
            corridor.raw_topology_waypoints = np.asarray(
                corridor.topology_ordered_waypoints, float).copy()
            corridor.protected_waypoints = np.asarray([
                node_by_id[n].point
                for n in node_ids[1:-1]
                if n in node_by_id and node_by_id[n].kind == "saddle"
            ], float)
            if self.morse_decision_mode == "balanced":
                corridor.channel_waypoints = np.asarray(
                    corridor.protected_waypoints, float)
                corridor.protected_waypoints = np.empty((0, pts.shape[1]), float)
            corridor.task_minima_waypoints = np.asarray([
                node_by_id[n].point
                for n in node_ids[1:-1]
                if n in node_by_id and node_by_id[n].kind == "minimum"
            ], float)
            corridor.auxiliary_node_ids = [
                n for n in node_ids[1:-1]
                if n in node_by_id and node_by_id[n].kind not in ("saddle", "minimum")
            ]
            corridor.auxiliary_node_count = len(corridor.auxiliary_node_ids)
            corridor.morse_induced = bool(corridor.morse_node_ids)
            corridor.topology_role = self._corridor_role(label, kinds)
            corridor.morse_priority_class = self._morse_priority_class(
                corridor.topology_role)
            corridor.mean_phi_on_path = self._path_phi(pts, "mean")
            corridor.max_phi_on_path = self._path_phi(pts, "max")
            (
                corridor.path_length,
                corridor.max_turn_angle,
                corridor.mean_turn_angle,
                corridor.execution_cost,
            ) = self._path_execution_metrics(pts)
            tracking = self._path_tracking_metrics(pts)
            corridor.tracking_cost = float(tracking["tracking_cost"])
            corridor.max_curvature = float(tracking["max_curvature"])
            corridor.curvature_violation = float(
                tracking["curvature_violation"])
            corridor.turn_violation = float(tracking["turn_violation"])
            corridor.min_segment_length = float(
                tracking["min_segment_length"])
            corridor.expected_progress = float(
                tracking["expected_progress"])
            corridor.min_clearance = self._path_clearance(cells)
            corridor.clearance_margin = self._cell_path_clearance_margin(cells)
            corridor.clearance_penalty = self._cell_path_clearance_penalty(cells)
            corridor.forbidden_hits = self._cell_path_forbidden_hits(cells)
            reject_reason = ""
            if not arm_execution_valid:
                reject_reason = self._kinematic_reject_reason(
                    corridor, baseline_ref=baseline_risk_ref)
                if not reject_reason:
                    reject_reason = self._candidate_hard_filter_reason(
                        corridor, baseline_risk_ref)
            if reject_reason:
                corridor.reject_reason = reject_reason
                candidate_before_filter.append(
                    self._candidate_record(corridor, reject_reason=reject_reason))
                last_append_reject["reason"] = reject_reason
                self._debug_inc("candidate_kinematic_reject_count")
                corridor_audit_counts.setdefault("reject_kinematic", 0)
                corridor_audit_counts["reject_kinematic"] += 1
                if len(corridor_audit) < 40:
                    corridor_audit.append(self._safety_record(
                        "corridor", "rejected", reject_reason,
                        corridor_label=label,
                        extra={
                            "nodes": list(node_ids),
                            "corridor_max_turn": float(corridor.corridor_max_turn),
                            "corridor_max_curvature": float(
                                corridor.corridor_max_curvature),
                            "corridor_self_intersection": bool(
                                corridor.corridor_self_intersection),
                            "mean_phi": float(corridor.mean_phi_on_path),
                            "max_phi": float(corridor.max_phi_on_path),
                        }))
                return None
            if corridor.forbidden_hits:
                corridor.stage1_forbidden_hits = int(corridor.forbidden_hits)
                if not has_prebuilt_route_geometry:
                    corridor.reject_reason = "candidate_forbidden"
                    candidate_before_filter.append(self._candidate_record(
                        corridor, reject_reason=corridor.reject_reason))
                    self._debug_inc("candidate_forbidden_reject_count")
                    last_append_reject["reason"] = "candidate_forbidden"
                    corridor_audit_counts["reject_forbidden"] += 1
                    if len(corridor_audit) < 40:
                        hit_ij = next(
                            (ij for ij in cells if bool(self._grid.get("forbidden")[ij])),
                            cells[0])
                        corridor_audit.append(self._grid_safety_record(
                            hit_ij, "corridor", "rejected", "forbidden",
                            corridor_label=label,
                            nodes=list(node_ids),
                            min_clearance=float(corridor.min_clearance),
                            clearance_margin=float(corridor.clearance_margin),
                            forbidden_hits=int(corridor.forbidden_hits),
                            mean_phi=float(corridor.mean_phi_on_path),
                            max_phi=float(corridor.max_phi_on_path)))
                    return None
            if corridor.clearance_margin < 0.0:
                corridor.stage1_clearance_margin = float(corridor.clearance_margin)
                if not has_prebuilt_route_geometry:
                    corridor.reject_reason = "candidate_clearance"
                    candidate_before_filter.append(self._candidate_record(
                        corridor, reject_reason=corridor.reject_reason))
                    self._debug_inc("clearance_reject_count")
                    last_append_reject["reason"] = "candidate_clearance"
                    corridor_audit_counts["reject_clearance"] += 1
                    if len(corridor_audit) < 40:
                        worst_ij = min(
                            cells,
                            key=lambda ij: self._clearance_cells(ij) -
                            self._hard_clearance_at_ij(ij))
                        corridor_audit.append(self._grid_safety_record(
                            worst_ij, "corridor", "rejected", "clearance",
                            corridor_label=label,
                            nodes=list(node_ids),
                            min_clearance=float(corridor.min_clearance),
                            clearance_margin=float(corridor.clearance_margin),
                            forbidden_hits=int(corridor.forbidden_hits),
                            mean_phi=float(corridor.mean_phi_on_path),
                            max_phi=float(corridor.max_phi_on_path)))
                    return None
            constraint_robot, constraint_phase = self._manifold_robot_phase()
            manifold_constraint = {
                "safe_manifold": self._grid,
                "risk_field": self.field,
                "minimum_clearance": float(getattr(
                    self, "hard_clearance", 0.0)),
                "min_clearance": float(getattr(self, "hard_clearance", 0.0)),
                "planning_clearance_margin": float(
                    self.planning_clearance_margin),
                "risk_threshold": float(getattr(self, "rho", 1.0)),
                "safe_threshold": float(getattr(self, "rho", 1.0)),
                "manifold_constraint_mode": self.manifold_constraint_mode,
                "robot_type": constraint_robot,
                "phase": constraint_phase,
                "task_phase": constraint_phase,
                "type": "safe_manifold",
                "used": True,
            }
            if arm_execution_valid:
                feasibility = {
                    "feasible": True,
                    "valid": True,
                    "manifold_valid": True,
                    "geometry_valid": True,
                    "tube_valid": True,
                    "candidate_tube_valid": True,
                    "risk_valid": True,
                    "candidate_status": "safe_candidate",
                    "candidate_filter_class": "safe",
                    "recovery_cost": 0.0,
                    "failure_reason": "",
                    "min_clearance": float(route.get(
                        "configuration_min_clearance",
                        route.get("min_clearance", 0.0))),
                    "trajectory_min_clearance": float(route.get(
                        "configuration_min_clearance",
                        route.get("trajectory_min_clearance",
                                  route.get("min_clearance", 0.0)))),
                    "max_risk": float(route.get(
                        "configuration_max_risk",
                        route.get("max_risk", 0.0))),
                    "trajectory_max_risk": float(route.get(
                        "configuration_max_risk",
                        route.get("trajectory_max_risk",
                                  route.get("max_risk", 0.0)))),
                    "arm_pose_optimization_used": bool(route.get(
                        "arm_pose_optimization_used", False)),
                }
            else:
                feasibility = evaluate_candidate(
                    corridor, manifold_constraint, risk_field=self.field)
            points_for_audit = np.asarray(
                getattr(corridor, "waypoints", []), float)
            corridor.planning_terminal_safety_context = (
                safety_context_audit(
                    points_for_audit[-1],
                    manifold_constraint=manifold_constraint,
                    corridor_constraint={
                        "centerline": points_for_audit,
                        "radius": float(getattr(corridor, "radius", 0.0)),
                    },
                    risk_field=self.field,
                    stage="planning_candidate_validation",
                    task_context_source="TopologyPlanner.field")
                if len(points_for_audit) else {})
            corridor.manifold_feasibility = dict(feasibility)
            corridor.manifold_feasible = bool(feasibility.get("feasible", False))
            corridor.candidate_tube_valid = bool(feasibility.get(
                "candidate_tube_valid", feasibility.get("tube_valid", False)))
            corridor.tube_valid = bool(corridor.candidate_tube_valid)
            corridor.manifold_valid = bool(feasibility.get(
                "manifold_valid", corridor.manifold_feasible))
            corridor.min_corridor_clearance = float(feasibility.get(
                "min_clearance", getattr(corridor, "min_clearance", 0.0)))
            corridor.min_clearance = float(corridor.min_corridor_clearance)
            corridor.trajectory_min_clearance = float(feasibility.get(
                "trajectory_min_clearance", corridor.min_corridor_clearance))
            corridor.trajectory_max_risk = float(feasibility.get(
                "trajectory_max_risk", getattr(corridor, "max_phi_on_path", 0.0)))
            corridor.planning_clearance_margin = float(feasibility.get(
                "planning_clearance_margin", self.planning_clearance_margin))
            corridor.max_phi_on_path = float(feasibility.get(
                "max_risk", getattr(corridor, "max_phi_on_path", 0.0)))
            corridor.risk_valid = bool(feasibility.get("risk_valid", True))
            raw_candidate_status = str(feasibility.get(
                "candidate_status", "unsafe_candidate"))
            corridor.candidate_status = (
                "feasible" if raw_candidate_status in ("safe", "safe_candidate") else
                "recoverable" if raw_candidate_status in (
                    "recoverable", "recoverable_candidate") else
                raw_candidate_status)
            failure_value = feasibility.get("failure_reason", "")
            if isinstance(failure_value, (list, tuple)):
                corridor.failure_reason = ",".join(str(x) for x in failure_value)
            else:
                corridor.failure_reason = str(failure_value)
            if (not corridor.manifold_feasible and
                    corridor.candidate_status != "recoverable"):
                corridor.reject_reason = "manifold_infeasible"
                candidate_before_filter.append(self._candidate_record(
                    corridor, reject_reason=corridor.reject_reason))
                self._debug_inc("num_manifold_filtered_candidates")
                self._debug_inc("candidate_manifold_reject_count")
                last_append_reject["reason"] = (
                    corridor.failure_reason or "manifold_infeasible")
                corridor_audit_counts.setdefault("reject_manifold", 0)
                corridor_audit_counts["reject_manifold"] += 1
                if len(corridor_audit) < 40:
                    corridor_audit.append(self._safety_record(
                        "corridor", "rejected",
                        corridor.failure_reason or "manifold_infeasible",
                        corridor_label=label,
                        extra={
                            "nodes": list(node_ids),
                            "min_clearance": float(corridor.min_clearance),
                            "minimum_clearance": float(feasibility.get(
                                "minimum_clearance", 0.0)),
                            "max_risk": float(feasibility.get("max_risk", 0.0)),
                            "risk_threshold": float(feasibility.get(
                                "risk_threshold", getattr(self, "rho", 1.0))),
                        }))
                return None
            corridor.manifold_constraint = {
                "boundary": getattr(corridor, "boundary", {}),
                "minimum_clearance": float(getattr(
                    self, "hard_clearance", 0.0)),
                "min_clearance": float(getattr(
                    self, "hard_clearance", 0.0)),
                "risk_threshold": float(getattr(self, "rho", 1.0)),
                "safe_threshold": float(getattr(self, "rho", 1.0)),
                "manifold_constraint_mode": self.manifold_constraint_mode,
                "robot_type": constraint_robot,
                "phase": constraint_phase,
                "task_phase": constraint_phase,
                "type": "safe_manifold",
                "used": True,
            }
            planning_context = build_safety_context(
                social_field=self.field,
                manifold_constraint=corridor.manifold_constraint,
                source="TopologyPlanner.field", strict=True)
            corridor.planning_safety_context_fingerprint = str(
                planning_context.get("fingerprint", ""))
            refine_topology_path(
                corridor,
                samples_per_segment=12,
                max_curvature=self.max_corridor_curvature,
                max_turn=self.max_corridor_turn,
                corridor_constraint={
                    "centerline": np.asarray(getattr(
                        corridor, "centerline",
                        getattr(corridor, "waypoints", [])), float).tolist(),
                    "radius": float(getattr(corridor, "radius", 0.0)),
                },
                manifold_constraint=corridor.manifold_constraint,
                safety_context=planning_context,
                require_social_context=True)
            score = self.evaluate_corridor(
                corridor, node_ids, base_cost, start=start, goal=goal)
            corridor.topology_cost = float(score["topology_cost"])
            corridor.risk_cost = float(score["risk_cost"])
            corridor.distance_cost = float(score["distance_cost"])
            corridor.length_cost = float(score["length_cost"])
            corridor.task_cost = float(score["task_cost"])
            corridor.task_state = str(score.get("task_state", ""))
            corridor.task_cost_breakdown = dict(score.get(
                "task_cost_breakdown", {}) or {})
            corridor.task_candidate_cost = float(score["task_candidate_cost"])
            corridor.task_mode = str(score["task_mode"])
            corridor.task_weight = dict(score["task_weight"])
            corridor.task_weight_used = bool(score["task_weight_used"])
            corridor.smooth_cost = float(score["smooth_cost"])
            corridor.motion_cost = float(score["motion_cost"])
            corridor.execution_cost = float(score["execution_cost"])
            corridor.curvature_cost = float(score["curvature_cost"])
            corridor.topology_value = float(score["topology_value"])
            corridor.risk_improvement_penalty = float(
                score["risk_improvement_penalty"])
            corridor.length_penalty = float(score["length_penalty"])
            corridor.graph_search_cost = float(score["graph_search_cost"])
            corridor.topology_diversity = float(score["topology_diversity"])
            corridor.total_score = float(score["score"])
            corridor.base_cost = float(score["score"])
            corridor.total_cost = float(score["score"])
            corridor.total_score = float(corridor.base_cost)
            corridor.saddle_value_bonus = self._saddle_value_bonus(
                kinds, corridor.base_cost)
            corridor.topological_value_bonus = float(corridor.topology_value)
            corridor.morse_bonus = 0.0
            corridor.cost = corridor.base_cost
            corridor.total_cost = float(corridor.cost)
            corridor.total_score = float(corridor.cost)
            corridor.adp_raw_mean = 0.0
            corridor.adp_raw_max = 0.0
            corridor.adp_raw_end = 0.0
            corridor._adp_raw_score = 0.0
            if critic is not None and feature_builder is not None and lambda_adp > 0.0:
                self._apply_wheelchair_adp(
                    corridor, goal, critic, feature_builder, feature_context)
            candidate_before_filter.append(self._candidate_record(corridor))
            corridors.append(corridor)
            corridor_audit_counts["kept"] += 1
            if len(corridor_audit) < 40:
                worst_ij = max(cells, key=lambda ij: float(self._grid["phi"][ij]))
                corridor_audit.append(self._grid_safety_record(
                    worst_ij, "corridor", "kept", "",
                    corridor_label=label,
                    nodes=list(node_ids),
                    min_clearance=float(corridor.min_clearance),
                    clearance_margin=float(corridor.clearance_margin),
                    forbidden_hits=int(corridor.forbidden_hits),
                    mean_phi=float(corridor.mean_phi_on_path),
                    max_phi=float(corridor.max_phi_on_path),
                    tracking_cost=float(corridor.tracking_cost)))
            mark_critical_nodes(
                node_ids, "candidate", "candidate_used", "",
                corridor_label=label)
            return corridor

        saddles = [n for n in nodes if n.kind == "saddle"]
        forced_count = 0
        pair_count = 0
        task_minima_count = 0
        used_graph_fallback_count = 0
        suppressed_graph_non_morse_count = 0
        route_max_paths = int(self.route_max_paths)
        route_max_routes = int(self.route_max_routes)
        arm_route_risk_threshold = None
        arm_search_validator = None
        arm_search_weights = {}
        if self.topology_profile == "arm":
            arm_route_risk_threshold = max(
                float(getattr(self, "rho", 1.0)),
                float(self.interest_config.get(
                    "arm_topology_risk_threshold", 6.0)
                    if isinstance(self.interest_config, dict) else 6.0))
            arm_search_validator = ArmTopologyValidator(
                risk_field=self.field,
                risk_threshold=arm_route_risk_threshold,
                minimum_clearance=min(
                    0.01, max(0.0, float(getattr(
                        self, "hard_clearance", 0.0)))),
                planning_clearance_margin=0.0,
                sample_spacing=max(0.02, min(
                    0.05, float(getattr(self, "grid_resolution", 0.05)))))
            arm_search_weights = {
                "clearance": float(self.interest_config.get(
                    "arm_morse_search_w_clearance", 1.0)
                    if isinstance(self.interest_config, dict) else 1.0),
                "link_clearance": float(self.interest_config.get(
                    "arm_morse_search_w_link_clearance", 1.0)
                    if isinstance(self.interest_config, dict) else 1.0),
                "risk": float(self.interest_config.get(
                    "arm_morse_search_w_risk", 0.25)
                    if isinstance(self.interest_config, dict) else 0.25),
            }
        generator = TopologyDrivenCandidateGenerator(
            grid=self._grid,
            world_from_ij=lambda ij: self._world(self._ij_to_p2(ij)),
            field=self.field,
            default_radius=radius,
            max_paths=route_max_paths,
            max_routes=route_max_routes,
            arm_search_enabled=bool(self.topology_profile == "arm"),
            arm_search_validator=arm_search_validator,
            arm_search_weights=arm_search_weights,
            robot_type=self.topology_profile)
        print("[topology] route_to_candidate start profile=%s max_paths=%d max_routes=%d nodes=%d edges=%d" % (
            self.topology_profile, route_max_paths, route_max_routes,
            len(nodes), sum(len(v) for v in edges.values()) // 2))
        route_generation_t0 = time.time()
        topology_routes = generator.generate(
            nodes, edges, start=start, goal=goal)
        route_generation_time = float(time.time() - route_generation_t0)
        print("[topology] route_to_candidate done profile=%s routes=%d time=%.3fs" % (
            self.topology_profile, len(topology_routes), route_generation_time))
        candidate_generation_report = dict(generator.last_report)
        candidate_generation_report["route_to_candidate_time_s"] = (
            route_generation_time)
        candidate_generation_report["route_to_candidate_max_paths"] = int(
            route_max_paths)
        candidate_generation_report["route_to_candidate_max_routes"] = int(
            route_max_routes)
        all_generated_topology_routes = [dict(route) for route in topology_routes]
        generated_route_count = int(len(topology_routes))
        feasible_routes = []
        recoverable_routes = []
        route_filter_records = []
        candidate_filter_report = []
        rejected_routes = []
        recovered_route_count = 0
        arm_recovery_report = []
        arm_route_validation_report = []
        arm_route_invalid_records = []
        arm_route_ranking_report = []
        if self.topology_profile == "arm" and topology_routes:
            arm_route_validator = arm_search_validator
            arm_rank_w_clearance = float(self.interest_config.get(
                "arm_route_rank_w_clearance", 1.0)
                if isinstance(self.interest_config, dict) else 1.0)
            arm_rank_w_link = float(self.interest_config.get(
                "arm_route_rank_w_link_clearance", 1.0)
                if isinstance(self.interest_config, dict) else 1.0)
            arm_rank_w_risk = float(self.interest_config.get(
                "arm_route_rank_w_risk", 0.25)
                if isinstance(self.interest_config, dict) else 0.25)
            arm_rank_top_k = int(self.interest_config.get(
                "arm_route_rank_top_k", min(
                    self.candidate_pool_min, len(topology_routes)))
                if isinstance(self.interest_config, dict) else
                min(self.candidate_pool_min, len(topology_routes)))
            arm_rank_top_k = max(1, min(int(len(topology_routes)),
                                        int(arm_rank_top_k)))
            arm_route_validation_report = []
            arm_route_ranking = []
            for route in topology_routes:
                validation = arm_route_validator.validate_route(route)
                arm_route_validation_report.append(dict(validation))
                clearance = float(validation.get(
                    "min_end_effector_clearance", 0.0))
                link_clearance = float(validation.get(
                    "min_link_clearance", clearance))
                risk = float(validation.get("max_risk", 0.0))
                score = (
                    arm_rank_w_clearance * clearance +
                    arm_rank_w_link * link_clearance -
                    arm_rank_w_risk * risk)
                arm_route_ranking.append({
                    "route_id": str(validation.get("route_id", "")),
                    "clearance": float(clearance),
                    "link_clearance": float(link_clearance),
                    "risk": float(risk),
                    "score": float(score),
                    "selected": False,
                })
            topology_routes, arm_route_ranking_report = (
                generator.apply_arm_route_ranking(
                    topology_routes, arm_route_ranking, top_k=arm_rank_top_k))
            selected_ranked_ids = set(str(route.get("candidate_id", ""))
                                      for route in topology_routes)
            arm_route_validation_report = [
                dict(item)
                for item in arm_route_validation_report
                if str(item.get("route_id", "")) in selected_ranked_ids
            ]
            valid_topology_routes = [
                route for route in topology_routes
                if bool(dict(route.get(
                    "arm_route_validation", {}) or {}).get(
                    "route_valid", False))
            ]
            ik_solver = TopologyIKSolver(
                risk_field=self.field,
                risk_threshold=arm_route_risk_threshold,
                minimum_clearance=min(
                    0.01, max(0.0, float(getattr(
                        self, "hard_clearance", 0.0)))))
            ik_valid_routes = []
            ik_valid_ids = set()
            topology_ik_report = []
            topology_ik_invalid_routes = []
            for route in valid_topology_routes:
                ik_route, ik_validation = ik_solver.validate_candidate(
                    route, boundary=route.get("boundary", None),
                    risk_field=self.field)
                topology_ik_report.append({
                    "candidate_id": str(ik_route.get("candidate_id", "")),
                    "ik_valid": bool(ik_route.get("ik_valid", False)),
                    "link_collision": bool(not ik_route.get(
                        "link_collision_valid", False)),
                    "link_collision_valid": bool(ik_route.get(
                        "link_collision_valid", False)),
                    "collision_link": str(ik_route.get("collision_link", "")),
                    "min_clearance": float(ik_route.get(
                        "configuration_min_clearance", 0.0)),
                    "max_risk": float(ik_route.get(
                        "configuration_max_risk", 0.0)),
                    "failure_reason": str(ik_validation.get(
                        "failure_reason", "")),
                })
                if bool(ik_route.get("ik_valid", False)):
                    ik_valid_routes.append(ik_route)
                    ik_valid_ids.add(str(ik_route.get("candidate_id", "")))
                else:
                    topology_ik_invalid_routes.append(ik_route)
            for route in topology_ik_invalid_routes:
                validation = dict(route.get("ik_validation", {}) or {})
                failure_reason = str(validation.get(
                    "failure_reason", "ik_or_link_collision"))
                status_class = _arm_candidate_filter_classification(
                    failure_reason, validation=validation)
                recoverable_reason = status_class == "recoverable"
                recovery_cost = _arm_recovery_cost(
                    failure_reason, validation=validation)
                arm_route_invalid_records.append({
                    "corridor_id": str(route.get("candidate_id", "")),
                    "label": str(route.get("candidate_id", "")),
                    "source": str(route.get("topology_source", "morse_graph")),
                    "topology_source": str(route.get(
                        "topology_source", "morse_graph")),
                    "generation_method": str(route.get("generation_method", "")),
                    "candidate_generation_role": str(route.get(
                        "candidate_generation_role", "")),
                    "critical_point_sequence": list(route.get(
                        "critical_point_sequence", [])),
                    "centerline": list(route.get("centerline", [])),
                    "boundary": dict(route.get("boundary", {}) or {}),
                    "corridor_width_profile": list(route.get(
                        "corridor_width_profile", [])),
                    "manifold_valid": bool(recoverable_reason),
                    "manifold_feasible": bool(recoverable_reason),
                    "min_clearance": float(validation.get("min_clearance", 0.0)),
                    "trajectory_min_clearance": float(validation.get(
                        "min_clearance", 0.0)),
                    "trajectory_max_risk": float(validation.get("max_risk", 0.0)),
                    "planning_clearance_margin": float(
                        self.planning_clearance_margin),
                    "min_corridor_clearance": float(validation.get(
                        "min_clearance", 0.0)),
                    "max_phi_on_path": float(validation.get("max_risk", 0.0)),
                    "risk_valid": bool(recoverable_reason),
                    "candidate_tube_valid": bool(recoverable_reason),
                    "tube_valid": bool(recoverable_reason),
                    "min_tube_clearance": 0.0,
                    "candidate_status": status_class,
                    "candidate_filter_class": status_class,
                    "raw_recovery_cost": float(recovery_cost),
                    "recovery_cost": float(recovery_cost),
                    "failure_reason": failure_reason,
                    "reject_reason": failure_reason,
                    "manifold_feasibility": {},
                    "ik_validation": validation,
                    "selected": False,
                    "arm_ik_candidate_attempts": list(route.get(
                        "arm_ik_candidate_attempts", [])),
                    "arm_ik_candidate_count": int(route.get(
                        "arm_ik_candidate_count", 0)),
                })
                candidate_filter_report.append({
                    "candidate_id": str(route.get("candidate_id", "")),
                    "geometry_valid": True,
                    "clearance_value": float(validation.get(
                        "min_clearance", 0.0)),
                    "risk_value": float(validation.get("max_risk", 0.0)),
                    "manifold_valid": False,
                    "tube_valid": False,
                    "candidate_status": status_class,
                    "candidate_filter_class": status_class,
                    "raw_recovery_cost": float(recovery_cost),
                    "recovery_cost": float(recovery_cost),
                    "failure_reason": [failure_reason] if failure_reason else [],
                    "arm_route_valid": True,
                    "ik_valid": False,
                    "arm_ik_candidate_count": int(route.get(
                        "arm_ik_candidate_count", 0)),
                })
                if recoverable_reason:
                    route["candidate_status"] = status_class
                    route["candidate_filter_class"] = status_class
                    route["failure_reason"] = failure_reason
                    route["raw_recovery_cost"] = float(recovery_cost)
                    route["recovery_cost"] = float(recovery_cost)
                    recoverable_routes.append(route)
                    continue
                rejected_routes.append(dict(route))
            valid_topology_routes = ik_valid_routes
            ik_invalid_ids = set(str(route.get("candidate_id", ""))
                                 for route in topology_ik_invalid_routes)
            valid_route_ids = set(str(route.get("candidate_id", ""))
                                  for route in valid_topology_routes)
            for route in topology_routes:
                if str(route.get("candidate_id", "")) in valid_route_ids:
                    continue
                if str(route.get("candidate_id", "")) in ik_invalid_ids:
                    continue
                validation = dict(route.get("arm_route_validation", {}) or {})
                failure_reason = str(validation.get(
                    "failure_reason", "arm_route_invalid"))
                status_class = _arm_candidate_filter_classification(
                    failure_reason, validation=validation)
                recoverable_reason = status_class == "recoverable"
                recovery_cost = _arm_recovery_cost(
                    failure_reason, validation=validation)
                route["candidate_status"] = status_class
                route["candidate_filter_class"] = status_class
                route["failure_reason"] = failure_reason
                route["manifold_feasible"] = bool(recoverable_reason)
                route["manifold_valid"] = bool(recoverable_reason)
                route["raw_recovery_cost"] = float(recovery_cost)
                route["recovery_cost"] = float(recovery_cost)
                arm_route_invalid_records.append({
                    "corridor_id": str(route.get("candidate_id", "")),
                    "label": str(route.get("candidate_id", "")),
                    "source": str(route.get("topology_source", "morse_graph")),
                    "topology_source": str(route.get(
                        "topology_source", "morse_graph")),
                    "generation_method": str(route.get("generation_method", "")),
                    "candidate_generation_role": str(route.get(
                        "candidate_generation_role", "")),
                    "critical_point_sequence": list(route.get(
                        "critical_point_sequence", [])),
                    "centerline": list(route.get("centerline", [])),
                    "boundary": dict(route.get("boundary", {}) or {}),
                    "corridor_width_profile": list(route.get(
                        "corridor_width_profile", [])),
                    "manifold_valid": bool(recoverable_reason),
                    "manifold_feasible": bool(recoverable_reason),
                    "min_clearance": float(validation.get(
                        "min_end_effector_clearance", 0.0)),
                    "trajectory_min_clearance": float(validation.get(
                        "min_end_effector_clearance", 0.0)),
                    "trajectory_max_risk": float(validation.get(
                        "max_risk", 0.0)),
                    "planning_clearance_margin": float(
                        self.planning_clearance_margin),
                    "min_corridor_clearance": float(validation.get(
                        "min_end_effector_clearance", 0.0)),
                    "max_phi_on_path": float(validation.get("max_risk", 0.0)),
                    "risk_valid": bool(recoverable_reason),
                    "candidate_tube_valid": bool(recoverable_reason),
                    "tube_valid": bool(recoverable_reason),
                    "min_tube_clearance": 0.0,
                    "candidate_status": status_class,
                    "candidate_filter_class": status_class,
                    "raw_recovery_cost": float(recovery_cost),
                    "recovery_cost": float(recovery_cost),
                    "failure_reason": failure_reason,
                    "reject_reason": failure_reason,
                    "manifold_feasibility": {},
                    "arm_route_validation": validation,
                    "selected": False,
                })
                candidate_filter_report.append({
                    "candidate_id": str(route.get("candidate_id", "")),
                    "geometry_valid": False,
                    "clearance_value": float(validation.get(
                        "min_end_effector_clearance", 0.0)),
                    "risk_value": float(validation.get("max_risk", 0.0)),
                    "manifold_valid": False,
                    "tube_valid": False,
                    "candidate_status": status_class,
                    "candidate_filter_class": status_class,
                    "raw_recovery_cost": float(recovery_cost),
                    "recovery_cost": float(recovery_cost),
                    "failure_reason": [failure_reason] if failure_reason else [],
                    "arm_route_valid": False,
                })
                if recoverable_reason:
                    recoverable_routes.append(route)
                    continue
                rejected_routes.append(dict(route))
            topology_routes = valid_topology_routes
            candidate_generation_report["arm_route_validation_used"] = True
            candidate_generation_report["arm_route_validation_report"] = list(
                arm_route_validation_report)
            candidate_generation_report["arm_route_validation"] = list(
                arm_route_validation_report)
            candidate_generation_report["arm_route_ranking"] = list(
                arm_route_ranking_report)
            candidate_generation_report["arm_route_ranking_file"] = (
                "arm_route_ranking.json")
            candidate_generation_report["arm_route_rank_top_k"] = int(
                arm_rank_top_k)
            candidate_generation_report["arm_route_rank_weights"] = {
                "clearance": float(arm_rank_w_clearance),
                "link_clearance": float(arm_rank_w_link),
                "risk": float(arm_rank_w_risk),
            }
            candidate_generation_report["arm_route_valid_count"] = int(
                len(valid_topology_routes))
            candidate_generation_report["arm_route_invalid_count"] = int(
                generated_route_count - len(valid_topology_routes))
            candidate_generation_report["topology_ik_validation_used"] = True
            candidate_generation_report["topology_ik_report"] = list(
                topology_ik_report)
            candidate_generation_report["topology_ik_valid_count"] = int(
                len(ik_valid_ids))
            candidate_generation_report["topology_ik_invalid_count"] = int(
                len(topology_ik_invalid_routes))
        elif self.topology_profile == "arm":
            candidate_generation_report["arm_route_validation_used"] = False
            candidate_generation_report["arm_route_validation_report"] = []
            candidate_generation_report["arm_route_validation"] = []
            candidate_generation_report["arm_route_ranking"] = []
            candidate_generation_report["arm_route_ranking_file"] = (
                "arm_route_ranking.json")
            candidate_generation_report["arm_route_valid_count"] = 0
            candidate_generation_report["arm_route_invalid_count"] = 0
        for route in topology_routes:
            constraint_robot, constraint_phase = self._manifold_robot_phase()
            cells = list(route.get("cells", []) or [])
            route_minimum_clearance = (
                max(float(self._hard_clearance_at_ij(ij)) for ij in cells)
                if cells else float(getattr(self, "hard_clearance", 0.0)))
            if self.topology_profile == "wheelchair":
                route_minimum_clearance = max(float(route_minimum_clearance), 0.10)
            route_filter_constraint = {
                "safe_manifold": self._grid,
                "risk_field": self.field,
                "boundary": dict(route.get("boundary", {}) or {}),
                "minimum_clearance": float(route_minimum_clearance),
                "min_clearance": float(route_minimum_clearance),
                "planning_clearance_margin": float(
                    self.planning_clearance_margin),
                "risk_threshold": float(getattr(self, "rho", 1.0)),
                "safe_threshold": float(getattr(self, "rho", 1.0)),
                "manifold_constraint_mode": self.manifold_constraint_mode,
                "robot_type": constraint_robot,
                "phase": constraint_phase,
                "task_phase": constraint_phase,
                "type": "safe_manifold",
                "used": True,
            }
            if (self.topology_profile == "arm" and
                    bool(route.get("ik_valid", False)) and
                    bool(route.get("link_collision_valid", False))):
                min_clearance = float(route.get(
                    "configuration_min_clearance", route.get("min_clearance", 0.0)))
                max_risk = float(route.get(
                    "configuration_max_risk", route.get("max_risk", 0.0)))
                feasibility = {
                    "feasible": True,
                    "valid": True,
                    "manifold_valid": True,
                    "geometry_valid": True,
                    "tube_valid": True,
                    "candidate_tube_valid": True,
                    "risk_valid": True,
                    "candidate_status": "safe_candidate",
                    "failure_reason": "",
                    "min_clearance": float(min_clearance),
                    "trajectory_min_clearance": float(min_clearance),
                    "max_risk": float(max_risk),
                    "trajectory_max_risk": float(max_risk),
                    "ik_valid": True,
                    "link_collision": False,
                    "link_collision_valid": True,
                    "arm_pose_optimization_used": bool(route.get(
                        "arm_pose_optimization_used", False)),
                    "arm_pose_optimizer_score": float(route.get(
                        "arm_pose_optimizer_score", 0.0)),
                }
                route["manifold_feasibility"] = dict(feasibility)
                route["manifold_feasible"] = True
                route["candidate_status"] = "feasible"
                route["candidate_filter_class"] = "safe"
                route["failure_reason"] = ""
                route["raw_recovery_cost"] = 0.0
                route["recovery_cost"] = 0.0
                route["risk_valid"] = True
                route["candidate_tube_valid"] = True
                route["tube_valid"] = True
                route["min_clearance"] = float(min_clearance)
                route["trajectory_min_clearance"] = float(min_clearance)
                route["trajectory_max_risk"] = float(max_risk)
                route["max_risk"] = float(max_risk)
                candidate_filter_report.append({
                    "candidate_id": str(route.get("candidate_id", "")),
                    "geometry_valid": True,
                    "clearance_value": float(min_clearance),
                    "risk_value": float(max_risk),
                    "manifold_valid": True,
                    "tube_valid": True,
                    "candidate_status": "feasible",
                    "candidate_filter_class": "safe",
                    "raw_recovery_cost": 0.0,
                    "recovery_cost": 0.0,
                    "failure_reason": [],
                    "arm_route_valid": True,
                    "ik_valid": True,
                    "link_collision_valid": True,
                    "arm_ik_candidate_count": int(route.get(
                        "arm_ik_candidate_count", 0)),
                    "arm_pose_optimization_used": bool(route.get(
                        "arm_pose_optimization_used", False)),
                })
                feasible_routes.append(route)
                continue
            feasibility = evaluate_candidate(
                route, route_filter_constraint, risk_field=self.field)
            route["manifold_feasibility"] = dict(feasibility)
            route["manifold_feasible"] = bool(feasibility.get("feasible", False))
            raw_status = str(feasibility.get(
                "candidate_status", "unsafe_candidate"))
            failure_value = feasibility.get("failure_reason", "")
            status_class = (
                "safe" if raw_status in ("safe", "safe_candidate") else
                _arm_candidate_filter_classification(
                    failure_value, validation=route.get(
                        "arm_route_validation", {}),
                    feasibility=feasibility)
                if self.topology_profile == "arm" else
                "recoverable" if raw_status in (
                    "recoverable", "recoverable_candidate") else
                "invalid")
            route["candidate_status"] = (
                "feasible" if status_class == "safe" else status_class)
            route["candidate_filter_class"] = status_class
            route["raw_recovery_cost"] = _arm_recovery_cost(
                failure_value, validation=route.get("arm_route_validation", {}),
                feasibility=feasibility) if self.topology_profile == "arm" else 0.0
            route["recovery_cost"] = float(route["raw_recovery_cost"])
            route["failure_reason"] = (
                ",".join(str(x) for x in failure_value)
                if isinstance(failure_value, (list, tuple)) else
                str(failure_value))
            route["risk_valid"] = bool(feasibility.get("risk_valid", True))
            route["candidate_tube_valid"] = bool(feasibility.get(
                "candidate_tube_valid", feasibility.get("tube_valid", False)))
            route["tube_valid"] = bool(route["candidate_tube_valid"])
            route["min_clearance"] = float(feasibility.get(
                "min_clearance", route.get("min_clearance", 0.0)))
            route["trajectory_min_clearance"] = float(feasibility.get(
                "trajectory_min_clearance", route.get("min_clearance", 0.0)))
            route["trajectory_max_risk"] = float(feasibility.get(
                "trajectory_max_risk", route.get("max_risk", 0.0)))
            route["max_risk"] = float(feasibility.get(
                "max_risk", route.get("max_risk", 0.0)))
            failure_reasons = (
                list(failure_value)
                if isinstance(failure_value, (list, tuple)) else
                ([str(failure_value)] if str(failure_value) else []))
            candidate_filter_report.append({
                "candidate_id": str(route.get("candidate_id", "")),
                "geometry_valid": bool(feasibility.get("geometry_valid", True)),
                "clearance_value": float(route.get(
                    "trajectory_min_clearance", route.get("min_clearance", 0.0))),
                "risk_value": float(route.get(
                    "trajectory_max_risk", route.get("max_risk", 0.0))),
                "manifold_valid": bool(feasibility.get("manifold_valid", False)),
                "tube_valid": bool(feasibility.get(
                    "tube_valid", route.get("tube_valid", False))),
                "candidate_status": str(route.get("candidate_status", "")),
                "candidate_filter_class": str(route.get(
                    "candidate_filter_class", route.get("candidate_status", ""))),
                "raw_recovery_cost": float(route.get(
                    "raw_recovery_cost", route.get("recovery_cost", 0.0))),
                "recovery_cost": float(route.get("recovery_cost", 0.0)),
                "failure_reason": failure_reasons,
            })
            if route["candidate_status"] == "feasible":
                feasible_routes.append(route)
                continue
            if route["candidate_status"] == "recoverable":
                recoverable_routes.append(route)
            rejected_routes.append(dict(route))
            route_filter_records.append({
                "corridor_id": str(route.get("candidate_id", "")),
                "label": str(route.get("candidate_id", "")),
                "source": str(route.get("topology_source", "morse_graph")),
                "topology_source": str(route.get("topology_source", "morse_graph")),
                "generation_method": str(route.get("generation_method", "")),
                "candidate_generation_role": str(route.get(
                    "candidate_generation_role", "")),
                "critical_point_sequence": list(route.get(
                    "critical_point_sequence", [])),
                "centerline": list(route.get("centerline", [])),
                "boundary": dict(route.get("boundary", {}) or {}),
                "corridor_width_profile": list(route.get(
                    "corridor_width_profile", [])),
                "manifold_valid": False,
                "manifold_feasible": False,
                "min_clearance": float(route.get("min_clearance", 0.0)),
                "trajectory_min_clearance": float(route.get(
                    "trajectory_min_clearance", route.get("min_clearance", 0.0))),
                "trajectory_max_risk": float(route.get(
                    "trajectory_max_risk", route.get("max_risk", 0.0))),
                "planning_clearance_margin": float(
                    self.planning_clearance_margin),
                "min_corridor_clearance": float(route.get("min_clearance", 0.0)),
                "max_phi_on_path": float(route.get("max_risk", 0.0)),
                "risk_valid": bool(route.get("risk_valid", False)),
                "candidate_tube_valid": bool(route.get(
                    "candidate_tube_valid", False)),
                "tube_valid": bool(route.get("tube_valid", False)),
                "min_tube_clearance": float(feasibility.get(
                    "min_tube_clearance", 0.0)),
                "candidate_status": str(route.get(
                    "candidate_status", "unsafe_candidate")),
                "candidate_filter_class": str(route.get(
                    "candidate_filter_class",
                    route.get("candidate_status", "unsafe_candidate"))),
                "raw_recovery_cost": float(route.get(
                    "raw_recovery_cost", route.get("recovery_cost", 0.0))),
                "recovery_cost": float(route.get("recovery_cost", 0.0)),
                "failure_reason": str(route.get(
                    "failure_reason", "manifold_infeasible")),
                "reject_reason": str(route.get(
                    "failure_reason", "manifold_infeasible")),
                "manifold_feasibility": dict(feasibility),
                "selected": False,
            })
            self._debug_inc("num_manifold_filtered_candidates")
        raw_route_count_before_recovery = int(len(feasible_routes) +
                                              len(recoverable_routes) +
                                              len(rejected_routes))
        safe_route_count_before_recovery = int(len(feasible_routes))
        recoverable_route_count_before_recovery = int(len(recoverable_routes))
        invalid_route_count_before_recovery = int(len(rejected_routes))
        recovery_report = {}
        if (generated_route_count and recoverable_routes and
                not feasible_routes and
                self.topology_profile == "arm"):
            recovery_constraint = {
                "safe_manifold": self._grid,
                "risk_field": self.field,
                "minimum_clearance": float(getattr(
                    self, "hard_clearance", 0.0)),
                "min_clearance": float(getattr(self, "hard_clearance", 0.0)),
                "planning_clearance_margin": float(
                    self.planning_clearance_margin),
                "risk_threshold": float(getattr(self, "rho", 1.0)),
                "safe_threshold": float(getattr(self, "rho", 1.0)),
                "manifold_constraint_mode": self.manifold_constraint_mode,
                "type": "safe_manifold",
                "used": True,
            }
            topology_ik_solver = TopologyIKSolver(
                risk_field=self.field,
                risk_threshold=float(getattr(self, "rho", 1.0)),
                minimum_clearance=float(getattr(self, "hard_clearance", 0.0)))
            recovered_routes, recovery_report = recover_candidates(
                recoverable_routes,
                recovery_constraint,
                robot_type="arm",
                risk_field=self.field,
                topology_ik_solver=topology_ik_solver)
            arm_recovery_report = list(recovery_report.get(
                "arm_candidate_recovery_report", []))
            if recovered_routes:
                recovered_route_count += int(len(recovered_routes))
                feasible_routes.extend(recovered_routes)
                recovered_ids = set(str(route.get("candidate_id", ""))
                                    for route in recovered_routes)
                route_filter_records = [
                    rec for rec in route_filter_records
                    if str(rec.get("corridor_id", "")) not in recovered_ids
                ]
        if (generated_route_count and recoverable_routes and
                self.topology_profile == "arm"):
            existing_ids = set(str(route.get("candidate_id", ""))
                               for route in feasible_routes)
            ranked_recoverable = []
            for recoverable_route in recoverable_routes:
                cid = str(recoverable_route.get("candidate_id", ""))
                if cid in existing_ids:
                    continue
                recoverable_route["route_source"] = str(
                    recoverable_route.get("route_source", "morse_topology"))
                recoverable_route["candidate_source"] = str(
                    recoverable_route.get(
                        "candidate_source", "morse_recoverable"))
                recoverable_route["topology_source"] = "morse_graph"
                recoverable_route["candidate_status"] = "recoverable"
                recoverable_route["candidate_filter_class"] = "recoverable"
                recoverable_route["raw_recovery_cost"] = float(
                    recoverable_route.get(
                        "raw_recovery_cost",
                        recoverable_route.get(
                            "recovery_cost",
                        _arm_recovery_cost(
                            recoverable_route.get("failure_reason", ""),
                            validation=recoverable_route.get(
                                "arm_route_validation", {}),
                            feasibility=recoverable_route.get(
                                "manifold_feasibility", {})))))
                recoverable_route["recovery_cost"] = float(
                    recoverable_route["raw_recovery_cost"])
                recoverable_route["recoverable_level"] = _arm_recoverable_level(
                    recoverable_route.get("failure_reason", ""),
                    validation=recoverable_route.get("arm_route_validation", {}),
                    feasibility=recoverable_route.get("manifold_feasibility", {}),
                    cost=recoverable_route["raw_recovery_cost"])
                if recoverable_route["recoverable_level"] == "level3":
                    recoverable_route["candidate_status"] = "invalid"
                    recoverable_route["candidate_filter_class"] = "invalid"
                    rejected_routes.append(dict(recoverable_route))
                    continue
                ranked_recoverable.append(recoverable_route)
                existing_ids.add(cid)
            feasible_routes.extend(ranked_recoverable)
        if (generated_route_count and recoverable_routes and
                not feasible_routes and
                self.topology_profile == "wheelchair" and
                self.allow_semantic_topology_recovery):
            constraint_robot, constraint_phase = self._manifold_robot_phase()
            recovery_constraint = {
                "safe_manifold": self._grid,
                "risk_field": self.field,
                "minimum_clearance": float(
                    max(float(getattr(self, "hard_clearance", 0.0)),
                        0.10 if self.topology_profile == "wheelchair" else 0.0)),
                "min_clearance": float(
                    max(float(getattr(self, "hard_clearance", 0.0)),
                        0.10 if self.topology_profile == "wheelchair" else 0.0)),
                "planning_clearance_margin": float(
                    self.planning_clearance_margin),
                "risk_threshold": float(getattr(self, "rho", 1.0)),
                "safe_threshold": float(getattr(self, "rho", 1.0)),
                "manifold_constraint_mode": self.manifold_constraint_mode,
                "robot_type": constraint_robot,
                "phase": constraint_phase,
                "task_phase": constraint_phase,
                "type": "safe_manifold",
                "used": True,
            }
            recovered_routes, recovery_report = recover_candidates(
                recoverable_routes,
                recovery_constraint,
                robot_type="wheelchair",
                risk_field=self.field,
                max_iterations=3)
            if recovered_routes:
                recovered_route_count += int(len(recovered_routes))
                for recovered_route in recovered_routes:
                    recovered_route["route_source"] = "morse_topology"
                    recovered_route["candidate_source"] = "morse_recovered"
                    recovered_route["topology_source"] = "morse_graph"
                    recovered_route["candidate_generation_role"] = (
                        "morse_graph_topology_path")
                feasible_routes.extend(recovered_routes)
                recovered_ids = set(str(route.get("candidate_id", ""))
                                    for route in recovered_routes)
                route_filter_records = [
                    rec for rec in route_filter_records
                    if str(rec.get("corridor_id", "")) not in recovered_ids
                ]
            elif recoverable_routes and self.allow_semantic_topology_recovery:
                recoverable_routes = sorted(
                    recoverable_routes,
                    key=lambda item: (
                        -float(item.get("trajectory_min_clearance",
                                        item.get("min_clearance", 0.0))),
                        float(item.get("trajectory_max_risk",
                                       item.get("max_risk", 0.0)))))
                for recovered_route in recoverable_routes[:max(1, int(k))]:
                    recovered_route["route_source"] = "morse_topology"
                    recovered_route["candidate_source"] = "morse_recovered"
                    recovered_route["topology_source"] = "morse_graph"
                    recovered_route["candidate_status"] = "recoverable"
                    recovered_route["candidate_recovered"] = True
                    recovered_route["recovery_success"] = False
                    recovered_route["candidate_recovery_mode"] = (
                        "staged_recoverable_ranking")
                feasible_routes.extend(recoverable_routes[:max(1, int(k))])
                recovered_route_count += int(len(recoverable_routes[:max(1, int(k))]))
                recovered_ids = set(str(route.get("candidate_id", ""))
                                    for route in recoverable_routes[:max(1, int(k))])
                route_filter_records = [
                    rec for rec in route_filter_records
                    if str(rec.get("corridor_id", "")) not in recovered_ids
                ]
        topology_routes = feasible_routes
        if arm_route_invalid_records:
            route_filter_records = [
                rec for rec in list(arm_route_invalid_records) + list(
                    route_filter_records)
                if str(rec.get("candidate_status", "invalid")) == "invalid"
            ]
        candidate_before_filter.extend(route_filter_records)
        recovery_ranking_records = []
        if self.topology_profile == "arm":
            for route in topology_routes:
                status = str(route.get("candidate_status", "feasible"))
                filter_class = str(route.get(
                    "candidate_filter_class",
                    "safe" if status == "feasible" else status))
                recovery_ranking_records.append({
                    "candidate_id": str(route.get("candidate_id", "")),
                    "candidate_status": status,
                    "candidate_filter_class": filter_class,
                    "raw_recovery_cost": float(route.get(
                        "raw_recovery_cost", route.get("recovery_cost", 0.0))),
                    "recovery_cost": float(route.get("recovery_cost", 0.0)),
                    "failure_reason": str(route.get("failure_reason", "")),
                    "min_clearance": float(route.get(
                        "trajectory_min_clearance",
                        route.get("min_clearance", 0.0))),
                    "max_risk": float(route.get(
                        "trajectory_max_risk",
                        route.get("max_risk", 0.0))),
                    "ik_valid": bool(route.get("ik_valid", False)),
                    "link_collision_valid": bool(route.get(
                        "link_collision_valid", True)),
                    "ranking_participates": bool(
                        filter_class in ("safe", "recoverable")),
                })
        candidate_generation_report["total_candidates"] = int(generated_route_count)
        candidate_generation_report["candidate_raw_count"] = int(
            raw_route_count_before_recovery)
        candidate_generation_report["candidate_safe_before_recovery"] = int(
            safe_route_count_before_recovery)
        candidate_generation_report["candidate_recoverable_before_recovery"] = int(
            recoverable_route_count_before_recovery)
        candidate_generation_report["candidate_invalid_before_recovery"] = int(
            invalid_route_count_before_recovery)
        candidate_generation_report["feasible_candidates"] = int(len(topology_routes))
        candidate_generation_report["safe_candidates"] = int(sum(
            1 for route in topology_routes
            if str(route.get("candidate_filter_class", "")) == "safe" or
            str(route.get("candidate_status", "")) == "feasible"))
        candidate_generation_report["recoverable_candidates"] = int(sum(
            1 for route in topology_routes
            if str(route.get("candidate_filter_class", "")) == "recoverable" or
            str(route.get("candidate_status", "")) == "recoverable"))
        candidate_generation_report["invalid_candidates"] = int(
            len(route_filter_records))
        candidate_generation_report["removed_candidates"] = int(
            len(route_filter_records))
        candidate_generation_report["num_manifold_filtered_candidates"] = int(
            len(route_filter_records))
        candidate_generation_report["topology_corridor_recovery_used"] = bool(
            recovered_route_count > 0)
        candidate_generation_report["candidate_recovery_attempted"] = int(
            recovered_route_count > 0)
        candidate_generation_report["candidate_recovery_success_count"] = int(
            recovered_route_count)
        candidate_generation_report["candidate_safe_after_recovery"] = int(
            candidate_generation_report["safe_candidates"])
        candidate_generation_report["raw_feasible_ratio"] = float(
            safe_route_count_before_recovery /
            float(max(1, raw_route_count_before_recovery)))
        candidate_generation_report["recovery_dependency_ratio"] = float(
            recovered_route_count /
            float(max(1, candidate_generation_report["safe_candidates"])))
        candidate_generation_report["recovered_feasible_candidates"] = int(
            recovered_route_count)
        candidate_generation_report["candidate_after_recovery"] = int(
            recovered_route_count)
        candidate_generation_report["candidate_filter_report"] = list(
            candidate_filter_report)
        candidate_generation_report["candidate_recovery_ranking"] = list(
            recovery_ranking_records)
        candidate_generation_report.update(dict(recovery_report or {}))
        if self.topology_profile == "arm":
            ik_success_count = int(candidate_generation_report.get(
                "topology_ik_valid_count", 0))
            if ik_success_count > 0:
                candidate_generation_report[
                    "arm_pose_ik_execution_recovery_used"] = True
                candidate_generation_report[
                    "arm_candidate_recovery_success_count"] = max(
                        int(candidate_generation_report.get(
                            "arm_candidate_recovery_success_count", 0)),
                        ik_success_count)
                candidate_generation_report[
                    "candidate_recovery_success_count"] = max(
                        int(candidate_generation_report.get(
                            "candidate_recovery_success_count", 0)),
                        ik_success_count)
        attempted = int(candidate_generation_report.get(
            "candidate_recovery_attempted", 0) or 0)
        success_count = int(candidate_generation_report.get(
            "candidate_recovery_success_count", 0) or 0)
        if attempted <= 0:
            candidate_generation_report["candidate_recovery_success_count"] = 0
            success_count = 0
        candidate_generation_report["candidate_recovery_used"] = bool(
            attempted > 0 and success_count > 0)
        candidate_generation_report["recovery_used"] = bool(
            candidate_generation_report["candidate_recovery_used"])
        candidate_generation_report["topology_corridor_recovery_used"] = bool(
            candidate_generation_report["candidate_recovery_used"])
        if generated_route_count and not topology_routes:
            best = max(
                route_filter_records,
                key=lambda item: float(item.get(
                    "trajectory_min_clearance",
                    item.get("min_clearance", 0.0))))
            candidate_generation_report["candidate_selection_status"] = (
                "no_safety_feasible_morse_candidate")
            candidate_generation_report["candidate_selection_mode"] = (
                "morse_route_local_refinement_failed")
            candidate_generation_report["best_rejected_candidate"] = str(
                best.get("corridor_id", best.get("label", "")))
            candidate_generation_report["best_rejected_min_clearance"] = float(
                best.get("trajectory_min_clearance",
                         best.get("min_clearance", 0.0)))
        for route in topology_routes:
            route["route_source"] = "morse_topology"
            route["candidate_source"] = str(route.get(
                "candidate_source", "morse_topology"))
            node_ids = list(route.get("node_sequence", []))
            corr = append_corridor(
                list(route.get("cells", [])),
                node_ids,
                float(route.get("base_cost", 0.0)),
                route=route)
            if corr is None:
                mark_critical_nodes(
                    node_ids, "candidate", "rejected",
                    last_append_reject.get("reason", "candidate_rejected"))
                continue
            corr.candidate_id = str(route.get("candidate_id", ""))
            corr.candidate_generation_method = "morse_topology_induced"
            corr.candidate_generation_role = str(
                route.get("candidate_generation_role",
                          "morse_graph_topology_path"))
            corr.topology_source = "morse_graph"
            corr.route_source = str(route.get(
                "route_source", "morse_topology"))
            corr.candidate_source = str(route.get(
                "candidate_source", "morse_topology"))
            corr.route_generation_level = str(route.get(
                "route_generation_level", ""))
            corr.critical_point_sequence = list(
                route.get("critical_point_sequence", []))
            corr.centerline = np.asarray(route.get("centerline", []), float)
            corr.boundary = dict(route.get("boundary", {}) or {})
            corr.corridor_width_profile = list(route.get(
                "corridor_width_profile",
                corr.boundary.get("corridor_width_profile",
                                  corr.boundary.get("width", []))))
            corr.speed_profile = list(route.get("speed_profile", []))
            corr.manifold_adaptive = bool(route.get(
                "manifold_adaptive", True))
            corr.risk_adaptive_width = bool(route.get(
                "risk_adaptive_width", True))
            corr.manifold_valid = bool(route.get("manifold_valid", True))
            corr.manifold_feasible = bool(route.get("manifold_feasible", True))
            corr.candidate_tube_valid = bool(route.get(
                "candidate_tube_valid", True))
            corr.tube_valid = bool(corr.candidate_tube_valid)
            corr.ik_valid = bool(route.get("ik_valid", False))
            corr.link_collision_valid = bool(route.get(
                "link_collision_valid", False))
            corr.arm_pose_optimization_used = bool(route.get(
                "arm_pose_optimization_used", False))
            corr.arm_ik_candidate_count = int(route.get(
                "arm_ik_candidate_count", 0))
            corr.arm_ik_candidate_attempts = list(route.get(
                "arm_ik_candidate_attempts", []))
            corr.recovery_used = bool(route.get("recovery_used", False))
            corr.topology_recovery_used = bool(route.get(
                "topology_recovery_used", corr.recovery_used))
            corr.candidate_recovery_mode = str(route.get(
                "candidate_recovery_mode", ""))
            corr.candidate_recovered = bool(route.get(
                "candidate_recovered", False))
            corr.before_clearance = float(route.get("before_clearance", 0.0))
            corr.after_clearance = float(route.get(
                "after_clearance", route.get("trajectory_min_clearance", 0.0)))
            corr.recovery_success = bool(route.get("recovery_success", False))
            corr.candidate_recovery_iterations = int(route.get(
                "candidate_recovery_iterations", 0))
            corr.adaptive_corridor_width = bool(route.get(
                "adaptive_corridor_width", False))
            corr.clearance_optimization_used = bool(route.get(
                "clearance_optimization_used", False))
            corr.candidate_status = str(route.get("candidate_status", "feasible"))
            corr.candidate_filter_class = str(route.get(
                "candidate_filter_class",
                "safe" if corr.candidate_status == "feasible" else
                corr.candidate_status))
            corr.failure_reason = str(route.get("failure_reason", ""))
            corr.raw_recovery_cost = float(route.get(
                "raw_recovery_cost", route.get("recovery_cost", 0.0)))
            corr.recovery_cost = float(route.get("recovery_cost", 0.0))
            corr.recoverable_level = str(route.get(
                "recoverable_level",
                _arm_recoverable_level(
                    route.get("failure_reason", ""),
                    validation=route.get("arm_route_validation", {}),
                    feasibility=route.get("manifold_feasibility", {}),
                    cost=corr.raw_recovery_cost)))
            corr.manifold_feasibility = dict(route.get(
                "manifold_feasibility", {}) or {})
            corr.min_corridor_clearance = float(route.get(
                "min_clearance", getattr(corr, "min_clearance", 0.0)))
            corr.min_clearance = float(corr.min_corridor_clearance)
            corr.trajectory_min_clearance = float(route.get(
                "trajectory_min_clearance", corr.min_corridor_clearance))
            corr.trajectory_max_risk = float(route.get(
                "trajectory_max_risk", getattr(corr, "max_phi_on_path", 0.0)))
            corr.planning_clearance_margin = float(
                self.planning_clearance_margin)
            corr.average_corridor_width = float(route.get(
                "average_corridor_width", 0.0))
            corr.manifold_validation = dict(route.get(
                "manifold_validation", {}) or {})
            corr.topology_valid = bool(route.get("topology_valid", True))
            corr.manifold_constraint = {
                "boundary": corr.boundary,
                "minimum_clearance": float(corr.manifold_feasibility.get(
                    "minimum_clearance", getattr(self, "hard_clearance", 0.0))),
                "min_clearance": float(corr.manifold_feasibility.get(
                    "minimum_clearance", getattr(self, "hard_clearance", 0.0))),
                "risk_threshold": float(getattr(self, "rho", 1.0)),
                "safe_threshold": float(getattr(self, "rho", 1.0)),
                "manifold_constraint_mode": self.manifold_constraint_mode,
                "type": "safe_manifold",
                "used": True,
            }
            planning_context = build_safety_context(
                social_field=self.field,
                manifold_constraint=corr.manifold_constraint,
                source="TopologyPlanner.field", strict=True)
            corr.planning_safety_context_fingerprint = str(
                planning_context.get("fingerprint", ""))
            refine_topology_path(
                corr,
                samples_per_segment=12,
                max_curvature=self.max_corridor_curvature,
                max_turn=self.max_corridor_turn,
                corridor_constraint={
                    "centerline": np.asarray(corr.centerline, float).tolist(),
                    "radius": float(getattr(corr, "radius", 0.0)),
                },
                manifold_constraint=corr.manifold_constraint,
                safety_context=planning_context,
                require_social_context=True)
            corr.morse_induced = True
            corr.morse_forced = 0
            forced_count += 1
            if any(node_by_id.get(n) is not None and
                   node_by_id[n].kind == "minimum"
                   for n in node_ids[1:-1]):
                task_minima_count += 1
            if sum(1 for n in node_ids[1:-1]
                   if node_by_id.get(n) is not None and
                   node_by_id[n].kind == "saddle") > 1:
                pair_count += 1
        morse_corridor_count = int(len(corridors))
        has_morse_saddle = bool(saddles)
        if not topology_routes:
            candidate_generation_report.setdefault(
                "failure_reason", "no_topology_candidate")

        base_order = sorted(corridors, key=lambda c: c.base_cost)
        for rank, corridor in enumerate(base_order):
            corridor.rank_base = int(rank)
        if corridors and critic is not None and feature_builder is not None and lambda_adp > 0.0:
            raw = np.asarray([c._adp_raw_score for c in corridors], float)
            lo = float(np.percentile(raw, 10))
            hi = float(np.percentile(raw, 90))
            if hi - lo < 1e-6:
                norm = np.zeros_like(raw)
            else:
                norm = np.clip((raw - lo) / (hi - lo), 0.0, 1.0)
            base_vals = np.asarray([c.base_cost for c in corridors], float)
            adp_scale = max(float(np.max(base_vals) - np.min(base_vals)),
                            0.10 * float(np.mean(np.abs(base_vals))), 1.0)
            for corridor, adp_norm in zip(corridors, norm):
                corridor.adp_norm = float(adp_norm)
                corridor.adp_cost = float(adp_norm) * adp_scale
                corridor.cost = corridor.base_cost + float(lambda_adp) * corridor.adp_cost
                corridor.total_cost = float(corridor.cost)
        else:
            for corridor in corridors:
                corridor.cost = corridor.base_cost
                corridor.total_cost = float(corridor.cost)
        balanced_non_morse_context_count = 0
        ranking_rejects = [
            corridor for corridor in corridors
            if str(getattr(corridor, "candidate_status", "feasible"))
            not in ("feasible", "recoverable")
        ]
        if ranking_rejects:
            for corridor in ranking_rejects:
                corridor.reject_reason = (
                    str(getattr(corridor, "failure_reason", "")) or
                    "manifold_infeasible")
                candidate_before_filter.append(self._candidate_record(
                    corridor, reject_reason=corridor.reject_reason))
            corridors[:] = [
                corridor for corridor in corridors
                if str(getattr(corridor, "candidate_status", "feasible"))
                in ("feasible", "recoverable")
            ]
        self._assign_topology_diversity(corridors, baseline_risk_ref)
        for corridor in corridors:
            score = self.evaluate_corridor(
                corridor,
                getattr(corridor, "topology_nodes", []),
                getattr(corridor, "graph_search_cost", 0.0),
                start=start, goal=goal)
            corridor.risk_cost = float(score["risk_cost"])
            corridor.distance_cost = float(score["distance_cost"])
            corridor.length_cost = float(score["length_cost"])
            corridor.task_cost = float(score["task_cost"])
            corridor.task_state = str(score.get("task_state", ""))
            corridor.task_cost_breakdown = dict(score.get(
                "task_cost_breakdown", {}) or {})
            corridor.task_candidate_cost = float(score["task_candidate_cost"])
            corridor.task_mode = str(score["task_mode"])
            corridor.task_weight = dict(score["task_weight"])
            corridor.task_weight_used = bool(score["task_weight_used"])
            corridor.smooth_cost = float(score["smooth_cost"])
            corridor.motion_cost = float(score["motion_cost"])
            corridor.execution_cost = float(score["execution_cost"])
            corridor.curvature_cost = float(score["curvature_cost"])
            corridor.topology_value = float(score["topology_value"])
            corridor.topology_diversity = float(score["topology_diversity"])
            corridor.clearance_cost = float(8.0 * getattr(
                corridor, "clearance_penalty", score.get("clearance_cost", 0.0)))
            corridor.clearance_cost = float(score.get(
                "clearance_cost", corridor.clearance_cost))
            corridor.feasibility_cost = float(score.get("feasibility_cost", 0.0))
            corridor.task_specific_cost = float(score.get(
                "task_specific_cost", 0.0))
            corridor.candidate_cost = float(score.get(
                "candidate_cost", score["score"]))
            corridor.candidate_cost_breakdown = dict(score.get(
                "candidate_cost_breakdown", {}))
        self._apply_normalized_scores(corridors)
        for corridor in corridors:
            corridor.decision_robot_type = str(self.topology_profile)
        total_order, decision_records = self.components.candidate_ranker(corridors)
        decision_by_id = {
            str(row.get("candidate_id", "")): dict(row)
            for row in decision_records
        }
        for row in candidate_filter_report:
            decision = decision_by_id.get(str(row.get("candidate_id", "")), {})
            if decision:
                row.update({
                    "topology_valid": bool(decision.get("topology_valid", False)),
                    "safety_feasible": bool(decision.get("safety_feasible", False)),
                    "execution_feasible": bool(decision.get(
                        "execution_feasible", False)),
                    "hard_feasible": bool(decision.get("hard_feasible", False)),
                    "decision_stage": str(decision.get("decision_stage", "")),
                    "decision_reason": str(decision.get("decision_reason", "")),
                    "ranking_eligible": bool(decision.get(
                        "ranking_eligible", False)),
                    "original_topology_identity": str(decision.get(
                        "original_topology_identity", "")),
                    "ranking_decomposition": dict(decision.get(
                        "ranking_decomposition", {})),
                    "rank": int(decision.get("rank", -1)),
                    "selected": bool(decision.get("selected", False)),
                    "ranking_score": float(decision.get(
                        "ranking_score", decision.get("total_score", 0.0))),
                })
        candidate_generation_report["candidate_decision_pipeline"] = list(
            decision_records)
        candidate_generation_report["hard_infeasible_candidate_count"] = int(
            sum(1 for row in decision_records if not row.get("hard_feasible")))
        candidate_generation_report["ranked_candidate_count"] = int(
            len(total_order))
        candidate_generation_report["topology_diversity_identity_count"] = int(
            len(set(candidate_topology_identity(c) for c in total_order)))
        pre_dedupe_count = len(total_order)
        total_order = self._dedupe_corridors_by_geometry(
            total_order, start2, goal2, min_keep=self.candidate_pool_min)
        dedupe_removed_count = pre_dedupe_count - len(total_order)
        for rank, corridor in enumerate(total_order):
            corridor.rank_total = int(rank)
            corridor.pre_rank = int(getattr(corridor, "rank_base", rank))
            corridor.risk_rank = int(rank)
            corridor.adp_rank = int(rank)
            corridor.corridor_id = "%s_c%04d" % (self.topology_profile, rank + 1)
            corridor.source = "morse_topology_graph"
            corridor.node_sequence = list(getattr(corridor, "topology_nodes", []))
            corridor.node_type_sequence = self._corridor_node_type_sequence(
                corridor)
        rejected_candidate_records = [
            rec for rec in candidate_before_filter
            if str(rec.get("reject_reason", ""))
        ]
        candidate_after_filter = [
            self._candidate_record(corridor)
            for corridor in total_order
        ]
        candidate_before_filter = (
            rejected_candidate_records + list(candidate_after_filter))
        top_k = max(1, int(k))
        corridors[:] = list(total_order)
        for idx, corridor in enumerate(corridors):
            corridor.selected = int(idx == 0)
            corridor.node_type_sequence = self._corridor_node_type_sequence(
                corridor)
        discovered_candidate_count = int(
            candidate_generation_report.get("num_candidates_generated", 0))
        candidate_generation_report["num_topology_paths_discovered"] = (
            discovered_candidate_count)
        candidate_generation_report["num_candidates_generated"] = int(len(corridors))
        candidate_generation_report["morse_induced_candidate"] = bool(corridors)
        candidate_generation_report["generation_method"] = (
            "morse_topology_induced" if corridors else "no_topology_candidate")
        candidate_after_top_k = [
            self._candidate_record(corridor, selected=bool(idx == 0))
            for idx, corridor in enumerate(corridors)
        ]
        task_state_info = infer_task_state(
            self.topology_profile, self.task_mode,
            phase="approach" if self.topology_profile == "arm" else "navigation",
            progress=0.0)
        candidate_task_cost_breakdown = []
        for idx, corridor in enumerate(corridors):
            cid = str(getattr(corridor, "corridor_id", ""))
            breakdown = dict(getattr(corridor, "task_cost_breakdown", {}) or {})
            terms = dict(breakdown.get("terms", {}) or {})
            candidate_task_cost_breakdown.append({
                "rank": int(idx + 1),
                "candidate_id": cid,
                "selected": bool(idx == 0),
                "task_mode": str(getattr(corridor, "task_mode", self.task_mode)),
                "task_state": str(getattr(corridor, "task_state", "")),
                "distance_cost": float(terms.get("distance_cost", 0.0)),
                "orientation_cost": float(terms.get(
                    "orientation_cost", terms.get("goal_alignment_cost", 0.0))),
                "feasibility_cost": float(terms.get(
                    "feasibility_cost", getattr(corridor, "feasibility_cost", 0.0))),
                "interaction_cost": float(terms.get(
                    "interaction_cost", terms.get("passage_width_cost", 0.0))),
                "task_cost": float(getattr(corridor, "task_cost", 0.0)),
                "total_task_cost": float(getattr(corridor, "task_cost", 0.0)),
                "task_cost_breakdown": breakdown,
                "cost_contributions": terms,
                "ranking_score": float(getattr(
                    corridor, "total_score",
                    getattr(corridor, "cost", 0.0))),
                "selection_reason": (
                    "lowest_candidate_cost_after_task_safety_feasibility_ranking"
                    if idx == 0 else "higher_ranking_score"),
            })
        removed_candidates = int(sum(
            1 for rec in candidate_before_filter
            if str(rec.get("candidate_status", "feasible"))
            not in ("feasible", "recoverable") or
            str(rec.get("reject_reason", ""))))
        recovery_ranking_rows = [
            self._candidate_record(corridor, selected=bool(idx == 0))
            for idx, corridor in enumerate(corridors)
            if str(getattr(corridor, "candidate_filter_class",
                           getattr(corridor, "candidate_status", "")))
            in ("safe", "recoverable") or
            str(getattr(corridor, "candidate_status", ""))
            in ("feasible", "recoverable")
        ]
        ranking_summary = {
            "total_candidates": int(generated_route_count),
            "feasible_candidates": int(len(total_order)),
            "safe_candidates": int(sum(
                1 for corridor in total_order
                if str(getattr(corridor, "candidate_filter_class", "")) == "safe" or
                str(getattr(corridor, "candidate_status", "")) == "feasible")),
            "recoverable_candidates": int(sum(
                1 for corridor in total_order
                if str(getattr(corridor, "candidate_filter_class", "")) == "recoverable" or
                str(getattr(corridor, "candidate_status", "")) == "recoverable")),
            "invalid_candidates": int(removed_candidates),
            "removed_candidates": int(removed_candidates),
            "filtered_candidates": int(removed_candidates),
            "filtered_infeasible_candidates": int(removed_candidates),
            "selected_candidate": str(getattr(
                corridors[0], "corridor_id", "") if corridors else ""),
            "selected_min_clearance": float(getattr(
                corridors[0], "trajectory_min_clearance",
                getattr(corridors[0], "min_corridor_clearance",
                        getattr(corridors[0], "min_clearance", 0.0)))
                if corridors else 0.0),
            "selected_max_risk": float(getattr(
                corridors[0], "trajectory_max_risk",
                getattr(corridors[0], "max_phi_on_path", 0.0))
                if corridors else 0.0),
        }
        if candidate_generation_report.get("candidate_selection_status"):
            ranking_summary["candidate_selection_status"] = str(
                candidate_generation_report.get("candidate_selection_status"))
        candidate_generation_report["candidate_min_clearance"] = float(
            ranking_summary["selected_min_clearance"])
        candidate_generation_report["candidate_max_risk"] = float(
            ranking_summary["selected_max_risk"])
        candidate_generation_report["candidate_manifold_feasible"] = bool(
            corridors and getattr(corridors[0], "manifold_feasible", False))
        candidate_generation_report["candidate_manifold_valid"] = bool(
            candidate_generation_report["candidate_manifold_feasible"])
        candidate_generation_report["candidate_tube_valid"] = bool(
            corridors and getattr(corridors[0], "candidate_tube_valid", False))
        reported_recoverable_count = int(sum(
            1 for row in candidate_filter_report
            if str(row.get("candidate_filter_class", "")) == "recoverable" or
            str(row.get("candidate_status", "")) == "recoverable"))
        candidate_generation_report["candidate_recoverable_count"] = int(
            reported_recoverable_count)
        candidate_generation_report["candidate_recovery_success_count"] = int(
            recovered_route_count)
        candidate_generation_report["candidate_after_recovery"] = int(
            len(total_order))
        candidate_generation_report["selected_min_clearance"] = float(
            ranking_summary["selected_min_clearance"])
        candidate_generation_report["filtered_infeasible_candidates"] = int(
            ranking_summary["filtered_infeasible_candidates"])
        candidate_generation_report["candidate_recovery_ranking"] = list(
            recovery_ranking_rows)
        candidate_generation_report["planning_clearance_margin"] = float(
            self.planning_clearance_margin)
        self.last_debug["num_ranked_corridors_before_top_k"] = int(len(total_order))
        selected_critical_ids = set()
        candidate_critical_ids = {}
        for corridor in corridors:
            rank = int(getattr(corridor, "rank_total", -1))
            for node_id in getattr(corridor, "topology_nodes", [])[1:-1]:
                node = node_by_id.get(node_id)
                crit_id = getattr(node, "critical_id", "") if node is not None else ""
                if not crit_id:
                    continue
                candidate_critical_ids.setdefault(crit_id, rank)
                if rank == 0:
                    selected_critical_ids.add(crit_id)
        for crit_id, rank in candidate_critical_ids.items():
            if crit_id in selected_critical_ids:
                mark_critical_id(
                    crit_id, "used", "selected", "",
                    selected_rank=int(rank))
            else:
                mark_critical_id(
                    crit_id, "used", "not_selected", "higher_cost",
                    best_candidate_rank=int(rank))
        for kind in ("minima", "saddles"):
            for rec in critical_chain.get(kind, []):
                if rec.get("status") == "node_added":
                    rec["stage"] = "used"
                    rec["status"] = "not_used"
                    rec["reason"] = "no_candidate_path"
        debug_counts = dict(self.last_debug)
        debug_counts["safety_audit_corridor_counts"] = corridor_audit_counts
        debug_counts["safety_audit_corridors"] = corridor_audit
        disconnect_reason = ""
        if not corridors:
            if not has_morse_saddle:
                disconnect_reason = "no_morse_saddle"
            elif morse_corridor_count <= 0:
                disconnect_reason = "no_valid_morse_corridor"
            elif debug_counts.get("edge_forbidden_reject_count", 0):
                disconnect_reason = "forbidden"
            elif debug_counts.get("edge_clearance_reject_count", 0):
                disconnect_reason = "clearance"
            elif debug_counts.get("edge_astar_fail_count", 0):
                disconnect_reason = "astar_fail"
            elif not topology_routes:
                disconnect_reason = str(
                    candidate_generation_report.get(
                        "failure_reason", "no_topology_candidate"))
            else:
                disconnect_reason = "candidate_rejected"
        used_minima_ids = set(
            n for c in corridors
            for n in getattr(c, "topology_nodes", [])
            if "minimum" in n)
        used_saddle_ids = set(
            n for c in corridors
            for n in getattr(c, "topology_nodes", [])
            if "saddle" in n)
        used_minima = len(used_minima_ids)
        used_saddles = len(used_saddle_ids)
        task_minima = ""
        minima_goal_distance = ""
        if critical.get("minima"):
            best_min = min(
                critical.get("minima", []),
                key=lambda item: float(np.linalg.norm(
                    self._plane(item.get("point", item.get("p2", goal2))) - goal2)))
            task_minima = str(best_min.get("id", ""))
            minima_goal_distance = float(np.linalg.norm(
                self._plane(best_min.get("point", best_min.get("p2", goal2))) - goal2))
        selected_minima = 0
        selected_saddles = 0
        critical_reason_counts = {}
        critical_stage_counts = {}
        for kind in ("minima", "saddles", "maxima"):
            for rec in critical_chain.get(kind, []):
                status = str(rec.get("status", ""))
                stage = str(rec.get("stage", ""))
                reason = str(rec.get("reason", ""))
                key = "%s:%s" % (stage, status)
                critical_stage_counts[key] = int(
                    critical_stage_counts.get(key, 0)) + 1
                if reason:
                    critical_reason_counts[reason] = int(
                        critical_reason_counts.get(reason, 0)) + 1
                if status == "selected":
                    if kind == "minima":
                        selected_minima += 1
                    elif kind == "saddles":
                        selected_saddles += 1
        num_minima = int(len(critical.get("minima", [])))
        num_saddle = int(len(critical.get("saddles", [])))
        num_critical_points = int(
            num_minima + num_saddle + len(critical.get("maxima", [])))
        graph_nodes = int(len(nodes))
        graph_edges = int(sum(len(v) for v in edges.values()) // 2)
        start_node_id = str(debug_counts.get("start_node", ""))
        goal_node_id = str(debug_counts.get("goal_node", ""))
        route_count = int(generated_route_count)
        if num_critical_points == 0:
            route_generation_status = "critical_point_failure"
        elif num_saddle == 0:
            route_generation_status = "saddle_missing"
        elif not start_node_id or not goal_node_id:
            route_generation_status = "pose_association_failure"
        elif not bool(debug_counts.get("topology_start_goal_connected", False)):
            route_generation_status = "graph_disconnected"
        elif route_count <= 0:
            route_generation_status = "route_search_failed"
        elif not corridors:
            route_generation_status = "candidate_filter_failed"
        else:
            route_generation_status = "ok"
        route_source = (
            "morse_topology"
            if route_count > 0 or corridors else
            "saddle_recovery")
        candidate_generated_count = int(candidate_generation_report.get(
            "candidate_generated",
            candidate_generation_report.get(
                "num_candidates_generated", generated_route_count)))
        candidate_feasible_count = int(len(total_order))
        selected_candidate_id = str(getattr(
            corridors[0], "corridor_id", "") if corridors else "")
        selected_candidate_source = str(getattr(
            corridors[0], "candidate_source",
            getattr(corridors[0], "route_source", "")) if corridors else "")
        planning_stage_status = {
            "morse_route_generation": (
                "success" if route_count > 0 else route_generation_status),
            "candidate_generation": (
                "success" if candidate_generated_count > 0 else
                "failed"),
            "candidate_filter": (
                "success" if candidate_generated_count == candidate_feasible_count and
                candidate_feasible_count > 0 else
                "partial_failure" if candidate_feasible_count > 0 else
                "failed"),
            "candidate_selection": (
                "success" if selected_candidate_id else "failed"),
        }
        planning_history = [
            {
                "stage": "morse",
                "status": planning_stage_status["morse_route_generation"],
                "count": int(route_count),
            },
            {
                "stage": "candidate_generation",
                "status": planning_stage_status["candidate_generation"],
                "count": int(candidate_generated_count),
            },
            {
                "stage": "filter",
                "status": planning_stage_status["candidate_filter"],
                "count": int(candidate_feasible_count),
                "rejected": int(max(0, candidate_generated_count -
                                    candidate_feasible_count)),
            },
            {
                "stage": "local_refinement",
                "status": (
                    "not_used" if not bool(candidate_generation_report.get(
                        "topology_corridor_recovery_used", False)) else
                    "success" if int(candidate_generation_report.get(
                        "candidate_after_recovery",
                        candidate_generation_report.get(
                            "recovered_feasible_candidates", 0))) > 0 else
                    "failed"),
                "count": int(candidate_generation_report.get(
                    "candidate_after_recovery",
                    candidate_generation_report.get(
                        "recovered_feasible_candidates", 0))),
            },
            {
                "stage": "ranking",
                "status": "success" if candidate_feasible_count > 0 else "failed",
                "count": int(candidate_feasible_count),
            },
            {
                "stage": "mpc",
                "status": "pending",
                "count": int(1 if selected_candidate_id else 0),
            },
        ]
        candidate_after_recovery_count = int(candidate_generation_report.get(
            "candidate_after_recovery",
            candidate_generation_report.get(
                "recovered_feasible_candidates", 0)))
        candidate_statistics = {
            "morse_routes": int(route_count),
            "candidate_generated": int(candidate_generated_count),
            "candidate_feasible": int(candidate_feasible_count),
            "candidate_after_recovery": int(candidate_after_recovery_count),
            "candidate_selected": selected_candidate_id,
            "selected_candidate_source": selected_candidate_source,
            "recovery_used": bool(
                int(candidate_generation_report.get(
                    "candidate_recovery_attempted", 0) or 0) > 0 and
                candidate_after_recovery_count > 0),
            "candidate_recovery_used": bool(
                int(candidate_generation_report.get(
                    "candidate_recovery_attempted", 0) or 0) > 0 and
                candidate_after_recovery_count > 0),
            "local_refinement_used": bool(candidate_generation_report.get(
                "topology_corridor_recovery_used", False)),
        }
        morse_route_records = []
        morse_route_evaluation_records = []
        for idx, route in enumerate(all_generated_topology_routes):
            route_eval = dict(route.get("morse_route_evaluation", {}) or {})
            pts = np.asarray(route.get("centerline", []), float)
            length = float(route.get("path_length", 0.0) or 0.0)
            if length <= 0.0 and len(pts) >= 2:
                length = float(np.sum(np.linalg.norm(
                    np.diff(pts[:, :2], axis=0), axis=1)))
            morse_route_records.append({
                "route_id": str(route.get(
                    "candidate_id", "morse_route_%04d" % (idx + 1))),
                "critical_sequence": list(route.get(
                    "critical_point_sequence", [])),
                "length": float(length),
                "clearance": float(route.get(
                    "trajectory_min_clearance",
                    route.get("min_clearance", 0.0))),
                "risk": float(route.get(
                    "trajectory_max_risk",
                    route.get("max_risk", 0.0))),
                "route_source": str(route.get(
                    "route_source", "morse_topology")),
                "route_generation_level": str(route.get(
                    "route_generation_level", "")),
            })
            if route_eval:
                route_eval.setdefault("route_id", str(route.get(
                    "candidate_id", "morse_route_%04d" % (idx + 1))))
                morse_route_evaluation_records.append(route_eval)
        morse_diagnostics = {
            "num_minima": int(num_minima),
            "num_saddle": int(num_saddle),
            "num_critical_points": int(num_critical_points),
            "graph_nodes": int(graph_nodes),
            "graph_edges": int(graph_edges),
            "candidate_count": int(candidate_generated_count),
            "start_node": start_node_id or None,
            "goal_node": goal_node_id or None,
            "route_count": int(route_count),
            "route_generation_status": route_generation_status,
            "route_source": route_source,
            "semantic_topology_recovery_used": bool(
                debug_counts.get("semantic_topology_recovery_used", False)),
            "planning_stage_status": planning_stage_status,
            "planning_history": planning_history,
        }
        self.last_debug = {
            "num_minima": num_minima,
            "num_saddle": num_saddle,
            "num_critical_points": num_critical_points,
            "num_critical_minima": len(critical.get("minima", [])),
            "num_critical_saddles": len(critical.get("saddles", [])),
            "num_critical_maxima": len(critical.get("maxima", [])),
            "num_raw_minima": int(critical.get("_raw_counts", {}).get("minima", 0)),
            "num_raw_saddles": int(critical.get("_raw_counts", {}).get("saddles", 0)),
            "num_raw_maxima": int(critical.get("_raw_counts", {}).get("maxima", 0)),
            "raw_minima_count": int(critical.get("_raw_counts", {}).get("minima", 0)),
            "raw_saddle_count": int(critical.get("_raw_counts", {}).get("saddles", 0)),
            "num_safe_minima": int(critical.get("_safe_counts", {}).get("minima", 0)),
            "num_safe_saddles": int(critical.get("_safe_counts", {}).get("saddles", 0)),
            "num_safe_maxima": int(critical.get("_safe_counts", {}).get("maxima", 0)),
            "safe_minima_count": int(critical.get("_safe_counts", {}).get("minima", 0)),
            "safe_saddle_count": int(critical.get("_safe_counts", {}).get("saddles", 0)),
            "num_filtered_minima": len(critical.get("minima", [])),
            "num_filtered_saddles": len(critical.get("saddles", [])),
            "num_filtered_maxima": len(critical.get("maxima", [])),
            "filtered_minima_count": len(critical.get("minima", [])),
            "filtered_saddle_count": len(critical.get("saddles", [])),
            "num_usable_minima": len(critical.get("minima", [])),
            "num_usable_saddles": len(critical.get("saddles", [])),
            "num_used_minima": used_minima,
            "num_used_saddles": used_saddles,
            "num_selected_minima": selected_minima,
            "num_selected_saddles": selected_saddles,
            "used_minima": used_minima,
            "used_saddles": used_saddles,
            "used_minima_count": used_minima,
            "used_saddle_count": used_saddles,
            "num_forced_critical_corridors": int(forced_count),
            "num_saddle_pair_corridors": int(pair_count),
            "num_task_minima_corridors": int(task_minima_count),
            "route_count": int(route_count),
            "route_generation_status": route_generation_status,
            "route_source": route_source,
            "semantic_topology_recovery_used": bool(
                debug_counts.get("semantic_topology_recovery_used", False)),
            "semantic_topology_recovery_added_minima": int(
                debug_counts.get("semantic_topology_recovery_added_minima", 0)),
            "semantic_topology_recovery_added_saddles": int(
                debug_counts.get("semantic_topology_recovery_added_saddles", 0)),
            "planning_stage_status": planning_stage_status,
            "planning_history": planning_history,
            "candidate_statistics": candidate_statistics,
            "morse_route_evaluation": morse_route_evaluation_records,
            "morse_diagnostics": morse_diagnostics,
            "morse_routes": morse_route_records,
            "num_ranked_corridors_before_top_k": int(len(total_order)),
            "candidate_before_filter_count": int(len(candidate_before_filter)),
            "candidate_after_filter_count": int(len(candidate_after_filter)),
            "candidate_after_top_k_count": int(len(candidate_after_top_k)),
            "candidate_ranking": dict(ranking_summary),
            "candidate_ranking_diagnostics": dict(ranking_summary),
            "candidate_recovery_ranking": list(
                candidate_generation_report.get(
                    "candidate_recovery_ranking", [])),
            "candidate_task_cost_breakdown": list(
                candidate_task_cost_breakdown),
            "task_mode": str(self.task_mode),
            "task_weight": dict(self.task_weight),
            "task_weight_used": True,
            "task_state": str(task_state_info.get("task_state", "")),
            "task_state_diagnostics": dict(task_state_info),
            "candidate_selection_status": str(
                candidate_generation_report.get(
                    "candidate_selection_status",
                    "ranked_feasible" if corridors else "no_feasible_candidate")),
            "candidate_min_clearance": float(getattr(
                corridors[0], "trajectory_min_clearance",
                getattr(corridors[0], "min_corridor_clearance",
                        getattr(corridors[0], "min_clearance", 0.0)))
                if corridors else 0.0),
            "candidate_max_risk": float(getattr(
                corridors[0], "trajectory_max_risk",
                getattr(corridors[0], "max_phi_on_path", 0.0))
                if corridors else 0.0),
            "candidate_manifold_feasible": bool(getattr(
                corridors[0], "manifold_feasible", False)
                if corridors else False),
            "candidate_manifold_valid": bool(getattr(
                corridors[0], "manifold_feasible", False)
                if corridors else False),
            "candidate_tube_valid": bool(getattr(
                corridors[0], "candidate_tube_valid", False)
                if corridors else False),
            "num_manifold_filtered_candidates": int(
                candidate_generation_report.get(
                    "num_manifold_filtered_candidates",
                    debug_counts.get("num_manifold_filtered_candidates", 0))),
            "filtered_infeasible_candidates": int(
                candidate_generation_report.get(
                    "filtered_infeasible_candidates",
                    candidate_generation_report.get(
                        "num_manifold_filtered_candidates", 0))),
            "topology_corridor_recovery_used": bool(
                candidate_generation_report.get(
                    "topology_corridor_recovery_used", False)),
            "recovered_feasible_candidates": int(
                candidate_generation_report.get(
                    "recovered_feasible_candidates", 0)),
            "candidate_after_recovery": int(
                candidate_generation_report.get(
                    "candidate_after_recovery",
                    candidate_generation_report.get(
                        "recovered_feasible_candidates", 0))),
            "candidate_filter_report": list(
                candidate_generation_report.get("candidate_filter_report", [])),
            "planning_clearance_margin": float(
                self.planning_clearance_margin),
            "num_corridors_before_dedupe": int(pre_dedupe_count),
            "num_corridors_deduped": int(dedupe_removed_count),
            "saddle_validity_reject_count": int(
                debug_counts.get("saddle_validity_reject_count", 0)),
            "candidate_kinematic_reject_count": int(
                debug_counts.get("candidate_kinematic_reject_count", 0)),
            "max_corridor_turn": float(self.max_corridor_turn),
            "max_corridor_curvature": float(self.max_corridor_curvature),
            "min_segment_length": float(self.min_segment_length),
            "require_risk_improvement": bool(self.require_risk_improvement),
            "num_morse_core_corridors": int(morse_corridor_count),
            "candidate_generation_method": str(
                candidate_generation_report.get(
                    "generation_method", "no_topology_candidate")),
            "candidate_number": int(len(corridors)),
            "morse_induced_candidate": bool(
                candidate_generation_report.get(
                    "morse_induced_candidate", bool(corridors))),
            "heuristic_fallback_used": bool(
                candidate_generation_report.get(
                    "heuristic_fallback_used", False)),
            "heuristic_sampling_used": bool(
                candidate_generation_report.get(
                    "heuristic_sampling_used", False)),
            "candidate_generation_report": dict(candidate_generation_report),
            "route_validation_report": list(
                candidate_generation_report.get("route_validation_report", [])),
            "arm_route_ranking": list(
                candidate_generation_report.get("arm_route_ranking", [])),
            "arm_route_ranking_report": list(
                candidate_generation_report.get("arm_route_ranking", [])),
            "arm_morse_search_diagnostics": list(
                candidate_generation_report.get(
                    "arm_morse_search_diagnostics", [])),
            "arm_morse_search_diagnostics_report": list(
                candidate_generation_report.get(
                    "arm_morse_search_diagnostics", [])),
            "morse_saddle_route_debug": list(
                candidate_generation_report.get(
                    "morse_saddle_route_debug", [])),
            "morse_saddle_route_debug_report": list(
                candidate_generation_report.get(
                    "morse_saddle_route_debug", [])),
            "candidate_width_profile": [
                {
                    "candidate_id": str(getattr(
                        c, "candidate_id", getattr(c, "corridor_id", ""))),
                    "corridor_id": str(getattr(c, "corridor_id", "")),
                    "candidate_source": str(getattr(
                        c, "candidate_source",
                        getattr(c, "route_source", ""))),
                    "corridor_width_profile": list(getattr(
                        c, "corridor_width_profile",
                        getattr(c, "boundary", {}).get(
                            "corridor_width_profile",
                            getattr(c, "boundary", {}).get("width", [])))),
                }
                for c in corridors
            ],
            "manifold_adaptive_corridor": bool(
                candidate_generation_report.get(
                    "manifold_adaptive_corridor", bool(corridors))),
            "risk_adaptive_width": bool(
                candidate_generation_report.get(
                    "risk_adaptive_width", bool(corridors))),
            "average_corridor_width": float(
                getattr(corridors[0], "average_corridor_width", 0.0)
                if corridors else
                candidate_generation_report.get(
                    "average_corridor_width", 0.0)),
            "min_corridor_clearance": float(
                getattr(corridors[0], "min_corridor_clearance", 0.0)
                if corridors else
                candidate_generation_report.get(
                    "min_corridor_clearance", 0.0)),
            "num_graph_fallback_corridors": int(used_graph_fallback_count),
            "num_balanced_non_morse_context_candidates": int(
                balanced_non_morse_context_count),
            "num_graph_non_morse_suppressed": int(
                suppressed_graph_non_morse_count),
            "allow_semantic_with_morse": self.allow_semantic_with_morse,
            "allow_ring_with_morse": self.allow_ring_with_morse,
            "allow_graph_fallback_with_morse": self.allow_graph_fallback_with_morse,
            "morse_core_required": self.morse_core_required,
            "morse_decision_mode": self.morse_decision_mode,
            "candidate_decision_rule": (
                "weighted_cost_over_morse_generated_candidates"
                if self.morse_decision_mode == "balanced"
                else "morse_priority_tie_break"),
            "candidate_source_rule": (
                "morse_graph_balanced_candidate_set"
                if self.morse_decision_mode == "balanced"
                else "morse_graph_saddle_only"),
            "candidate_source_required": "morse_topology_graph",
            "candidate_generation_flow": [
                "risk_field",
                "safety_manifold",
                "morse_critical_points",
                "topology_graph",
                "candidate_corridor",
                "manifold_feasibility_filter",
                "ranking",
            ],
            "minima_semantic_type": (
                "handover_stable_approach" if self.topology_profile == "arm"
                else "waiting_or_docking"),
            "selected_task_minima": task_minima,
            "minima_goal_distance": minima_goal_distance,
            "top_k": int(top_k),
            "candidate_pool_min": int(self.candidate_pool_min),
            "lambda_execution": self.lambda_execution,
            "semantic_nodes_added": int(debug_counts.get("semantic_nodes_added", 0)),
            "ring_nodes_added": int(debug_counts.get("ring_nodes_added", 0)),
            "semantic_nodes_suppressed_by_morse": int(
                debug_counts.get("semantic_nodes_suppressed_by_morse", 0)),
            "ring_nodes_suppressed_by_morse": int(
                debug_counts.get("ring_nodes_suppressed_by_morse", 0)),
            "reject_by_gradient_count": int(
                critical.get("_reject_counts", {}).get("reject_by_gradient_count", 0)),
            "reject_by_gradient": int(
                critical.get("_reject_counts", {}).get("reject_by_gradient_count", 0)),
            "reject_by_degenerate_count": int(
                critical.get("_reject_counts", {}).get("reject_by_degenerate_count", 0)),
            "reject_by_forbidden_count": int(
                critical.get("_reject_counts", {}).get("reject_by_forbidden_count", 0)),
            "reject_by_forbidden": int(
                critical.get("_reject_counts", {}).get("reject_by_forbidden_count", 0)),
            "reject_by_clearance_count": int(
                critical.get("_reject_counts", {}).get("reject_by_clearance_count", 0)),
            "reject_by_clearance": int(
                critical.get("_reject_counts", {}).get("reject_by_clearance_count", 0)),
            "reject_by_unsafe_count": int(
                critical.get("_reject_counts", {}).get("reject_by_unsafe_count", 0)),
            "reject_by_unsafe": int(
                critical.get("_reject_counts", {}).get("reject_by_unsafe_count", 0)),
            "merge_count": int(sum(
                critical.get("_merge_counts", {}).get(kind, 0)
                for kind in ("minima", "saddles", "maxima"))),
            "merge_minima_count": int(
                critical.get("_merge_counts", {}).get("minima", 0)),
            "merge_saddle_count": int(
                critical.get("_merge_counts", {}).get("saddles", 0)),
            "num_topology_nodes": len(nodes),
            "num_topology_edges": sum(len(v) for v in edges.values()) // 2,
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges,
            "start_node": start_node_id,
            "goal_node": goal_node_id,
            "topology_pose_association": dict(debug_counts.get(
                "topology_pose_association", {})),
            "topology_start_goal_connected_before_repair": bool(
                debug_counts.get(
                    "topology_start_goal_connected_before_repair", False)),
            "topology_start_goal_connected": bool(debug_counts.get(
                "topology_start_goal_connected", False)),
            "local_graph_repair": dict(debug_counts.get(
                "local_graph_repair", {})),
            "num_candidate_corridors": len(corridors),
            "num_morse_minima_corridors": int(sum(
                1 for c in corridors
                if str(getattr(c, "label", "")).startswith("morse_minima_"))),
            "num_morse_saddle_corridors": int(sum(
                1 for c in corridors
                if str(getattr(c, "label", "")).startswith("morse_saddle_"))),
            "num_morse_mix_corridors": int(sum(
                1 for c in corridors
                if str(getattr(c, "label", "")).startswith("morse_mix_"))),
            "num_morse_saddle_pair_corridors": int(sum(
                1 for c in corridors
                if str(getattr(c, "label", "")).startswith(
                    "morse_saddle_pair_"))),
            "num_graph_direct_corridors": int(sum(
                1 for c in corridors
                if str(getattr(c, "label", "")).startswith("graph_direct_"))),
            "num_graph_semantic_corridors": int(sum(
                1 for c in corridors
                if str(getattr(c, "label", "")).startswith("graph_semantic_"))),
            "critical": critical,
            "critical_chain": critical_chain,
            "critical_reason_counts": critical_reason_counts,
            "critical_stage_counts": critical_stage_counts,
            "safety_audit": {
                "grid_counts": dict(debug_counts.get(
                    "safety_audit_grid_counts", {})),
                "grid_samples": list(debug_counts.get(
                    "safety_audit_grid_samples", [])),
                "edge_counts": dict(debug_counts.get(
                    "safety_audit_edge_counts", {})),
                "edges": list(debug_counts.get("safety_audit_edges", [])),
                "corridor_counts": dict(debug_counts.get(
                    "safety_audit_corridor_counts", {})),
                "corridors": list(debug_counts.get(
                    "safety_audit_corridors", [])),
            },
            "safety_audit_grid_counts": dict(debug_counts.get(
                "safety_audit_grid_counts", {})),
            "safety_audit_edge_counts": dict(debug_counts.get(
                "safety_audit_edge_counts", {})),
            "safety_audit_corridor_counts": dict(debug_counts.get(
                "safety_audit_corridor_counts", {})),
            "nodes": nodes,
            "edges": edges,
            "candidate_before_filter": candidate_before_filter,
            "candidate_after_filter": candidate_after_filter,
            "candidate_after_top_k": candidate_after_top_k,
            "candidate_corridors": [
                {
                    "corridor_id": str(getattr(c, "corridor_id", "")),
                    "label": str(getattr(c, "label", "")),
                    "source": str(getattr(c, "source", "")),
                    "topology_source": str(getattr(
                        c, "topology_source",
                        getattr(c, "source", ""))),
                    "route_source": str(getattr(c, "route_source", "")),
                    "candidate_source": str(getattr(
                        c, "candidate_source",
                        getattr(c, "route_source", ""))),
                    "route_generation_level": str(getattr(
                        c, "route_generation_level", "")),
                    "generation_method": str(getattr(
                        c, "candidate_generation_method", "")),
                    "candidate_generation_role": str(getattr(
                        c, "candidate_generation_role", "")),
                    "critical_point_sequence": list(getattr(
                        c, "critical_point_sequence", [])),
                    "centerline": np.asarray(getattr(
                        c, "centerline",
                        getattr(c, "waypoints", [])), float).tolist(),
                    "boundary": getattr(c, "boundary", {}),
                    "corridor_width_profile": list(getattr(
                        c, "corridor_width_profile",
                        getattr(c, "boundary", {}).get(
                            "corridor_width_profile",
                            getattr(c, "boundary", {}).get("width", [])))),
                    "manifold_adaptive": bool(getattr(
                        c, "manifold_adaptive", False)),
                    "risk_adaptive_width": bool(getattr(
                        c, "risk_adaptive_width", False)),
                    "manifold_valid": bool(getattr(c, "manifold_valid", True)),
                    "manifold_feasible": bool(getattr(
                        c, "manifold_feasible", False)),
                    "candidate_status": str(getattr(
                        c, "candidate_status", "feasible")),
                    "failure_reason": str(getattr(
                        c, "failure_reason", "")),
                    "risk_valid": bool(getattr(c, "risk_valid", True)),
                    "trajectory_min_clearance": float(getattr(
                        c, "trajectory_min_clearance",
                        getattr(c, "min_corridor_clearance",
                                getattr(c, "min_clearance", 0.0)))),
                    "trajectory_max_risk": float(getattr(
                        c, "trajectory_max_risk",
                        getattr(c, "max_phi_on_path", 0.0))),
                    "planning_clearance_margin": float(getattr(
                        c, "planning_clearance_margin",
                        self.planning_clearance_margin)),
                    "min_corridor_clearance": float(getattr(
                        c, "min_corridor_clearance",
                        getattr(c, "min_clearance", 0.0))),
                    "average_corridor_width": float(getattr(
                        c, "average_corridor_width", 0.0)),
                    "manifold_validation": dict(getattr(
                        c, "manifold_validation", {}) or {}),
                    "manifold_feasibility": dict(getattr(
                        c, "manifold_feasibility", {}) or {}),
                    "topology_valid": bool(getattr(c, "topology_valid", True)),
                    "node_sequence": list(getattr(c, "node_sequence", [])),
                    "node_type_sequence": list(getattr(
                        c, "node_type_sequence",
                        self._corridor_node_type_sequence(c))),
                    "semantic_sequence": list(getattr(
                        c, "semantic_sequence", semantic_sequence(c))),
                    "topology_class": str(getattr(c, "topology_class", "")),
                    "topology_route_class": str(getattr(
                        c, "topology_route_class",
                        getattr(c, "topology_class", ""))),
                    "task_semantic_class": str(getattr(
                        c, "task_semantic_class", "")),
                    "source_graph_id": str(getattr(c, "source_graph_id", "")),
                    "source_saddle_ids": list(getattr(c, "source_saddle_ids", [])),
                    "source_minima_ids": list(getattr(c, "source_minima_ids", [])),
                    "waypoints": np.asarray(
                        getattr(c, "waypoints", []), float).tolist(),
                    "topology_ordered_waypoints": np.asarray(getattr(
                        c, "topology_ordered_waypoints", []), float).tolist(),
                    "pre_rank": int(getattr(c, "pre_rank", -1)),
                    "risk_rank": int(getattr(c, "risk_rank", -1)),
                    "adp_rank": int(getattr(c, "adp_rank", -1)),
                    "selected": bool(getattr(c, "selected", 0)),
                    "execution_corridor_id": (
                        str(getattr(c, "corridor_id", ""))
                        if bool(getattr(c, "selected", 0)) else ""),
                    "reject_reason": str(getattr(c, "reject_reason", "")),
                    "topology_role": str(getattr(c, "topology_role", "")),
                    "morse_priority_class": int(getattr(
                        c, "morse_priority_class", 3)),
                    "morse_priority_applied": bool(getattr(
                        c, "morse_priority_applied", False)),
                    "topology_selection_reason": str(getattr(
                        c, "topology_selection_reason", "")),
                    "topology_selection_window": float(getattr(
                        c, "topology_selection_window", 0.0)),
                    "morse_node_ids": list(getattr(c, "morse_node_ids", [])),
                    "morse_node_types": [
                        self._public_node_type(t)
                        for t in list(getattr(c, "morse_node_types", []))
                    ],
                    "morse_induced": bool(getattr(c, "morse_induced", False)),
                    "auxiliary_node_ids": list(getattr(c, "auxiliary_node_ids", [])),
                    "auxiliary_node_count": int(getattr(c, "auxiliary_node_count", 0)),
                    "morse_bonus": float(getattr(c, "morse_bonus", 0.0)),
                    "saddle_value_bonus": float(getattr(
                        c, "saddle_value_bonus", 0.0)),
                    "topological_value_bonus": float(getattr(
                        c, "topological_value_bonus", 0.0)),
                    "topology_cost": float(getattr(c, "topology_cost", 0.0)),
                    "risk_cost": float(getattr(c, "risk_cost", 0.0)),
                    "length_cost": float(getattr(c, "distance_cost", 0.0)),
                    "task_cost": float(getattr(c, "task_cost", 0.0)),
                    "risk_norm": float(getattr(c, "risk_norm", 0.0)),
                    "length_norm": float(getattr(c, "length_norm", 0.0)),
                    "smooth_norm": float(getattr(c, "smooth_norm", 0.0)),
                    "task_norm": float(getattr(c, "task_norm", 0.0)),
                    "execution_norm": float(getattr(c, "execution_norm", 0.0)),
                    "execution_cost": float(getattr(c, "execution_cost", 0.0)),
                    "topology_diversity": float(getattr(
                        c, "topology_diversity", 0.0)),
                    "total_score": float(getattr(
                        c, "total_score",
                        getattr(c, "total_cost", getattr(c, "cost", 0.0)))),
                    "total_cost": float(getattr(
                        c, "total_cost", getattr(c, "cost", 0.0))),
                    "distance_cost": float(getattr(c, "distance_cost", 0.0)),
                    "smooth_cost": float(getattr(c, "smooth_cost", 0.0)),
                    "motion_cost": float(getattr(c, "motion_cost", 0.0)),
                    "curvature_cost": float(getattr(c, "curvature_cost", 0.0)),
                    "topology_value": float(getattr(c, "topology_value", 0.0)),
                    "graph_search_cost": float(getattr(
                        c, "graph_search_cost", 0.0)),
                    "path_length": float(getattr(c, "path_length", 0.0)),
                    "risk_per_meter": float(getattr(c, "risk_per_meter", 0.0)),
                    "baseline_risk_per_meter": float(getattr(
                        c, "baseline_risk_per_meter", 0.0)),
                    "baseline_similarity": float(getattr(
                        c, "baseline_similarity", 0.0)),
                    "mean_distance_to_baseline": float(getattr(
                        c, "mean_distance_to_baseline", 0.0)),
                    "max_lateral_offset": float(getattr(
                        c, "max_lateral_offset", 0.0)),
                    "risk_gain": float(getattr(c, "risk_gain", 0.0)),
                    "max_turn_angle": float(getattr(c, "max_turn_angle", 0.0)),
                    "mean_turn_angle": float(getattr(c, "mean_turn_angle", 0.0)),
                    "execution_cost": float(getattr(c, "execution_cost", 0.0)),
                    "raw_topology_waypoints": np.asarray(getattr(
                        c, "raw_topology_waypoints", []), float).tolist(),
                    "refined_waypoints": np.asarray(getattr(
                        c, "refined_waypoints", []), float).tolist(),
                    "refinement_used": int(getattr(c, "refinement_used", 0)),
                    "refinement_reject_reason": str(getattr(
                        c, "refinement_reject_reason", "")),
                    "refinement_manifold_checked": bool(getattr(
                        c, "refinement_manifold_checked", False)),
                    "refinement_manifold_valid": bool(getattr(
                        c, "refinement_manifold_valid", False)),
                    "pre_refinement_clearance": float(getattr(
                        c, "pre_refinement_clearance", 0.0)),
                    "post_refinement_clearance": float(getattr(
                        c, "post_refinement_clearance", 0.0)),
                    "refinement_fallback": bool(getattr(
                        c, "refinement_fallback", False)),
                    "trajectory_manifold_violation_count": int(getattr(
                        c, "trajectory_manifold_violation_count", 0)),
                    "refinement_trace": list(getattr(
                        c, "refinement_trace", [])),
                    "refined_path_length": float(getattr(
                        c, "refined_path_length", 0.0)),
                    "refined_max_turn_angle": float(getattr(
                        c, "refined_max_turn_angle", 0.0)),
                    "refined_mean_turn_angle": float(getattr(
                        c, "refined_mean_turn_angle", 0.0)),
                    "refined_max_curvature": float(getattr(
                        c, "refined_max_curvature", 0.0)),
                    "tracking_cost": float(getattr(c, "tracking_cost", 0.0)),
                    "max_curvature": float(getattr(c, "max_curvature", 0.0)),
                    "corridor_max_turn": float(getattr(
                        c, "corridor_max_turn", 0.0)),
                    "corridor_max_curvature": float(getattr(
                        c, "corridor_max_curvature", 0.0)),
                    "corridor_self_intersection": bool(getattr(
                        c, "corridor_self_intersection", False)),
                    "corridor_kinematic_valid": int(getattr(
                        c, "corridor_kinematic_valid", 1)),
                    "smoothing_preserved_saddle": int(getattr(
                        c, "smoothing_preserved_saddle", 1)),
                    "curvature_violation": float(getattr(
                        c, "curvature_violation", 0.0)),
                    "turn_violation": float(getattr(c, "turn_violation", 0.0)),
                    "expected_progress": float(getattr(
                        c, "expected_progress", 0.0)),
                    "base_cost": float(getattr(c, "base_cost", 0.0)),
                    "cost": float(getattr(c, "cost", 0.0)),
                    "clearance_margin": float(getattr(
                        c, "clearance_margin", 0.0)),
                }
                for c in corridors
            ],
            "topology_disconnect_reason": disconnect_reason,
            "topology_profile": self.topology_profile,
            "topology_workspace_dimension": self.workspace_dimension,
            "topology_motion_space": self.motion_space,
            "topology_neighbor_strategy": self.neighbor_strategy,
            "topology_grid_resolution": self.grid_resolution,
            "hard_clearance": self.hard_clearance,
            "clearance_target": self.min_clearance,
            "neighbor_k": self.neighbor_k,
            "lambda_tracking": self.lambda_tracking,
            "lambda_saddle_value": self.lambda_saddle_value,
            "selected_saddle_value_bonus": float(getattr(
                corridors[0], "saddle_value_bonus", 0.0) if corridors else 0.0),
            "selected_candidate_total_score": float(getattr(
                corridors[0], "total_score", getattr(
                    corridors[0], "cost", 0.0)) if corridors else 0.0),
            "selected_tracking_cost": float(getattr(
                corridors[0], "tracking_cost", 0.0) if corridors else 0.0),
            "selected_max_curvature": float(getattr(
                corridors[0], "max_curvature", 0.0) if corridors else 0.0),
            "selected_curvature_violation": float(getattr(
                corridors[0], "curvature_violation", 0.0) if corridors else 0.0),
            "selected_turn_violation": float(getattr(
                corridors[0], "turn_violation", 0.0) if corridors else 0.0),
            "selected_expected_progress": float(getattr(
                corridors[0], "expected_progress", 0.0) if corridors else 0.0),
            "selected_morse_induced": bool(getattr(
                corridors[0], "morse_induced", False) if corridors else False),
            "selected_corridor_id": str(getattr(
                corridors[0], "corridor_id", "") if corridors else ""),
            "execution_corridor_id": str(getattr(
                corridors[0], "corridor_id", "") if corridors else ""),
            "selected_without_adp": str(getattr(
                min(corridors, key=lambda c: int(getattr(c, "rank_base", 0))),
                "corridor_id", "") if corridors else ""),
            "selected_with_adp": str(getattr(
                corridors[0], "corridor_id", "") if corridors else ""),
            "saddle_tie_ratio": self.saddle_tie_ratio,
            "morse_priority_ratio": self.morse_priority_ratio,
            "morse_saddle_priority_ratio": self.morse_saddle_priority_ratio,
            "morse_mix_priority_ratio": self.morse_mix_priority_ratio,
            "morse_minima_priority_ratio": self.morse_minima_priority_ratio,
            "morse_primary": self.morse_primary,
            "morse_core_required": self.morse_core_required,
        }
        self.last_debug["num_forbidden_cells"] = int(
            debug_counts.get("num_forbidden_cells", 0))
        self.last_debug["candidate_forbidden_reject_count"] = int(
            debug_counts.get("candidate_forbidden_reject_count", 0))
        self.last_debug["selected_corridor_forbidden_hits"] = int(
            getattr(corridors[0], "forbidden_hits", 0) if corridors else 0)
        self.last_debug["clearance_reject_count"] = int(
            debug_counts.get("clearance_reject_count", 0))
        self.last_debug["edge_clearance_reject_count"] = int(
            debug_counts.get("edge_clearance_reject_count", 0))
        self.last_debug["edge_forbidden_reject_count"] = int(
            debug_counts.get("edge_forbidden_reject_count", 0))
        self.last_debug["edge_astar_fail_count"] = int(
            debug_counts.get("edge_astar_fail_count", 0))
        self.last_debug["neighbor_pair_attempt_count"] = int(
            debug_counts.get("neighbor_pair_attempt_count", 0))
        self.last_debug["safety_region_cell_counts"] = dict(
            debug_counts.get("safety_region_cell_counts", {}))
        self.last_debug["safety_regions"] = list(
            debug_counts.get("safety_regions", []))
        return corridors

    def _corridor_score(self, corridor, node_ids, graph_search_cost,
                        start=None, goal=None):
        weights = self._unified_score_weights()
        pts = self._corridor_points(corridor)
        kinds = list(getattr(corridor, "topology_kinds", []))
        label = str(getattr(corridor, "label", ""))
        saddle_count = sum(1 for kind in kinds if kind == "saddle")
        minimum_count = sum(1 for kind in kinds if kind == "minimum")
        auxiliary_count = int(getattr(corridor, "auxiliary_node_count", 0))
        task_minima = label.startswith("morse_task_minima_")
        topology_complexity = 0.10 * max(0, len(node_ids) - 3)
        topology_value = (1.0 if saddle_count else 0.0) + (
            0.5 if task_minima else 0.0)
        topology_value += 0.25 * max(0, saddle_count - 1)
        topology_value -= 0.10 * auxiliary_count
        topology_cost = topology_complexity
        risk_cost = (
            float(getattr(corridor, "path_length", 0.0)) *
            float(getattr(corridor, "mean_phi_on_path", 0.0)) +
            weights["max_risk_internal"] *
            float(getattr(corridor, "max_phi_on_path", 0.0)))
        distance_cost = float(getattr(corridor, "path_length", 0.0))
        task_breakdown = evaluate_task_cost_breakdown(
            corridor, self.topology_profile, start=start, goal=goal,
            task_mode=self.task_mode)
        task_cost = float(task_breakdown.get("task_cost", 0.0))
        task_state_info = infer_task_state(
            self.topology_profile, self.task_mode, progress=0.0)
        task_candidate_cost = weighted_task_candidate_cost(
            risk_cost, distance_cost, task_cost, self.task_weight)
        curvature_cost = float(getattr(corridor, "max_curvature", 0.0))
        smooth_cost = float(getattr(corridor, "tracking_cost", 0.0))
        motion_cost = float(getattr(
            corridor, "motion_cost", getattr(corridor, "execution_cost", 0.0)))
        execution_cost = float(getattr(corridor, "execution_cost", motion_cost))
        clearance_cost = self._clearance_cost(corridor)
        feasibility_cost = self._feasibility_cost(corridor)
        recovery_cost = float(getattr(corridor, "recovery_cost", 0.0))
        recovery_weight = self._unified_score_weights().get("recovery", 0.3)
        task_specific_terms = self._task_specific_candidate_terms(
            corridor, pts, start=start, goal=goal)
        task_specific_cost = float(sum(task_specific_terms.values()))
        topology_diversity = float(getattr(corridor, "topology_diversity", 0.0))
        risk_improvement_penalty = 0.0
        if int(getattr(corridor, "risk_not_improved", 0)):
            base_risk = max(
                float(getattr(corridor, "baseline_risk_per_meter", 0.0)),
                1e-6)
            risk_improvement_penalty = 2.0 * (
                float(getattr(corridor, "mean_phi_on_path", 0.0)) / base_risk)
        length_penalty = 0.0
        if int(getattr(corridor, "length_over_baseline", 0)):
            base_len = max(float(getattr(corridor, "baseline_length", 0.0)), 1e-6)
            length_penalty = 3.0 * (
                float(getattr(corridor, "path_length", 0.0)) / base_len)
        score = float(
            weights["risk"] * risk_cost +
            weights["length"] * distance_cost +
            weights["smooth"] * smooth_cost +
            weights["task"] * task_cost +
            weights["execution"] * execution_cost -
            weights["topology"] * topology_value -
            weights["diversity"] * topology_diversity +
            recovery_weight * recovery_cost)
        cost_breakdown = {
            "risk_cost": float(risk_cost),
            "length_cost": float(distance_cost),
            "smoothness_cost": float(smooth_cost),
            "task_cost": float(task_cost),
            "task_state": str(task_state_info.get("task_state", "")),
            "task_cost_breakdown": dict(task_breakdown),
            "clearance_cost": float(clearance_cost),
            "feasibility_cost": float(feasibility_cost),
            "hard_feasibility_in_score": False,
            "raw_recovery_cost": float(getattr(
                corridor, "raw_recovery_cost", recovery_cost)),
            "normalized_recovery_cost": float(getattr(
                corridor, "normalized_recovery_cost", recovery_cost)),
            "recovery_cost": float(recovery_cost),
            "recovery_weight": float(recovery_weight),
            "weighted_recovery_cost": float(recovery_weight * recovery_cost),
            "task_specific_cost": float(task_specific_cost),
            "topology_value": float(topology_value),
            "topology_value_term": float(-weights["topology"] * topology_value),
            "geometry_tie_breaker": float(distance_cost),
            "execution_cost": float(execution_cost),
            "execution_cost_term": float(weights["execution"] * execution_cost),
            "ranking_score": float(score),
            "candidate_cost": float(score),
        }
        cost_breakdown.update(task_specific_terms)
        return {
            "score": float(score),
            "topology_cost": float(topology_cost),
            "risk_cost": float(risk_cost),
            "distance_cost": float(distance_cost),
            "length_cost": float(distance_cost),
            "task_cost": float(task_cost),
            "task_state": str(task_state_info.get("task_state", "")),
            "task_cost_breakdown": dict(task_breakdown),
            "task_candidate_cost": float(task_candidate_cost),
            "task_mode": str(self.task_mode),
            "task_weight": dict(self.task_weight),
            "task_weight_used": True,
            "smooth_cost": float(smooth_cost),
            "smoothness_cost": float(smooth_cost),
            "motion_cost": float(motion_cost),
            "execution_cost": float(execution_cost),
            "clearance_cost": float(clearance_cost),
            "feasibility_cost": float(feasibility_cost),
            "raw_recovery_cost": float(getattr(
                corridor, "raw_recovery_cost", recovery_cost)),
            "normalized_recovery_cost": float(getattr(
                corridor, "normalized_recovery_cost", recovery_cost)),
            "recovery_cost": float(recovery_cost),
            "recovery_weight": float(recovery_weight),
            "task_specific_cost": float(task_specific_cost),
            "candidate_cost": float(score),
            "candidate_cost_breakdown": cost_breakdown,
            "curvature_cost": float(curvature_cost),
            "topology_value": float(topology_value),
            "topology_diversity": float(topology_diversity),
            "risk_improvement_penalty": float(risk_improvement_penalty),
            "length_penalty": float(length_penalty),
            "graph_search_cost": float(graph_search_cost),
        }

    def _unified_score_weights(self):
        raw = dict(self.corridor_score_weights or {})
        def pick(name, default, *aliases):
            keys = (name,) + aliases
            for key in keys:
                if key in raw:
                    return float(raw[key])
                prefixed = "w_" + key
                if prefixed in raw:
                    return float(raw[prefixed])
            return float(default)
        return {
            "risk": pick("risk", 4.0),
            "max_risk_internal": pick("max_risk", 2.0),
            "length": pick("length", 1.0),
            "smooth": pick("smooth", 1.5),
            "task": pick("task", 2.0),
            "execution": pick("execution", 1.0, "exec"),
            "topology": pick("topology", 0.5),
            "diversity": pick("diversity", 0.5),
            "recovery": pick("recovery", 0.3),
        }

    def _apply_normalized_scores(self, corridors):
        corridors = list(corridors or [])
        if not corridors:
            return
        weights = self._unified_score_weights()

        def values(name):
            return np.asarray([
                float(getattr(c, name, 0.0)) for c in corridors
            ], float)

        def normalize(name, attr):
            vals = values(name)
            lo = float(np.min(vals))
            hi = float(np.max(vals))
            if hi - lo <= 1e-9:
                norm = np.zeros_like(vals)
            else:
                norm = (vals - lo) / (hi - lo)
            for c, val in zip(corridors, norm):
                setattr(c, attr, float(val))

        normalize("risk_cost", "risk_norm")
        normalize("length_cost", "length_norm")
        normalize("smooth_cost", "smooth_norm")
        normalize("task_cost", "task_norm")
        normalize("clearance_cost", "clearance_norm")
        normalize("feasibility_cost", "feasibility_norm")
        normalize("execution_cost", "execution_norm")
        recovery_raw = np.asarray([
            float(getattr(c, "raw_recovery_cost",
                          getattr(c, "recovery_cost", 0.0)))
            for c in corridors
        ], float)
        rec_hi = float(np.max(recovery_raw))
        if rec_hi <= 1e-9:
            recovery_norm = np.zeros_like(recovery_raw)
        else:
            recovery_norm = recovery_raw / rec_hi
        for c, raw, norm in zip(corridors, recovery_raw, recovery_norm):
            c.raw_recovery_cost = float(raw)
            c.normalized_recovery_cost = float(np.clip(norm, 0.0, 1.0))
            c.recovery_cost = float(c.normalized_recovery_cost)
            c.recovery_norm = float(c.normalized_recovery_cost)
            c.recoverable_level = _arm_recoverable_level(
                getattr(c, "failure_reason", ""),
                validation=getattr(c, "arm_route_validation", {}),
                feasibility=getattr(c, "manifold_feasibility", {}),
                cost=c.normalized_recovery_cost)
        for corridor in corridors:
            score = (
                weights["risk"] * float(getattr(corridor, "risk_cost", 0.0)) +
                weights["length"] * float(getattr(corridor, "length_cost", 0.0)) +
                weights["smooth"] * float(getattr(corridor, "smooth_cost", 0.0)) +
                weights["task"] * float(getattr(corridor, "task_cost", 0.0)) +
                weights["execution"] * float(getattr(
                    corridor, "execution_cost", 0.0)) -
                weights["topology"] * float(getattr(
                    corridor, "topology_value", 0.0)) -
                weights["diversity"] * float(getattr(
                    corridor, "topology_diversity", 0.0)) +
                weights["recovery"] * float(getattr(
                    corridor, "normalized_recovery_cost",
                    getattr(corridor, "recovery_cost", 0.0))))
            corridor.total_score = float(score)
            corridor.total_cost = float(score)
            corridor.base_cost = float(score)
            corridor.cost = float(score)
            breakdown = dict(getattr(corridor, "candidate_cost_breakdown", {}) or {})
            breakdown.update({
                "risk_cost": float(getattr(corridor, "risk_cost", 0.0)),
                "length_cost": float(getattr(corridor, "length_cost", 0.0)),
                "smoothness_cost": float(getattr(corridor, "smooth_cost", 0.0)),
                "task_cost": float(getattr(corridor, "task_cost", 0.0)),
                "clearance_cost": float(getattr(corridor, "clearance_cost", 0.0)),
                "feasibility_cost": float(getattr(corridor, "feasibility_cost", 0.0)),
                "hard_feasibility_in_score": False,
                "raw_recovery_cost": float(getattr(corridor, "raw_recovery_cost", 0.0)),
                "normalized_recovery_cost": float(getattr(
                    corridor, "normalized_recovery_cost", 0.0)),
                "recovery_cost": float(getattr(corridor, "recovery_cost", 0.0)),
                "weighted_recovery_cost": float(
                    weights["recovery"] * float(getattr(
                        corridor, "normalized_recovery_cost",
                        getattr(corridor, "recovery_cost", 0.0)))),
                "recovery_weight": float(weights["recovery"]),
                "topology_value": float(getattr(corridor, "topology_value", 0.0)),
                "topology_value_term": float(-weights["topology"] * float(
                    getattr(corridor, "topology_value", 0.0))),
                "geometry_tie_breaker": float(getattr(
                    corridor, "path_length",
                    getattr(corridor, "length_cost", 0.0))),
                "execution_cost": float(getattr(corridor, "execution_cost", 0.0)),
                "execution_cost_term": float(weights["execution"] * float(
                    getattr(corridor, "execution_cost", 0.0))),
                "ranking_score": float(score),
                "task_specific_cost": float(getattr(corridor, "task_specific_cost", 0.0)),
                "candidate_cost": float(score),
            })
            corridor.candidate_cost = float(score)
            corridor.candidate_cost_breakdown = breakdown

    def _assign_topology_diversity(self, corridors, baseline_ref=None):
        corridors = list(corridors or [])
        if not corridors:
            return
        baseline_ref = baseline_ref or {}
        baseline_class = "center_passage"
        if len(corridors) == 1:
            stats = self._candidate_baseline_stats(corridors[0], baseline_ref)
            different_side = 1 if int(stats.get("side", 0)) != 0 else 0
            different_class = 1 if str(getattr(
                corridors[0], "topology_class", "")) != baseline_class else 0
            corridors[0].mean_distance_to_baseline = float(
                stats.get("mean_distance", 0.0))
            corridors[0].max_lateral_offset = float(
                stats.get("max_lateral_offset", 0.0))
            corridors[0].topology_diversity = float(
                stats.get("mean_distance", 0.0) +
                0.5 * different_side + 0.5 * different_class)
            return
        for corridor in corridors:
            stats = self._candidate_baseline_stats(corridor, baseline_ref)
            different_side = 1 if int(stats.get("side", 0)) != 0 else 0
            different_class = 1 if str(getattr(
                corridor, "topology_class", "")) != baseline_class else 0
            corridor.mean_distance_to_baseline = float(
                stats.get("mean_distance", 0.0))
            corridor.max_lateral_offset = float(
                stats.get("max_lateral_offset", 0.0))
            corridor.baseline_side = int(stats.get("side", 0))
            corridor.topology_diversity = float(
                stats.get("mean_distance", 0.0) +
                0.5 * different_side + 0.5 * different_class)

    def _corridor_side(self, corridor, start2, goal2):
        pts = np.asarray(getattr(corridor, "waypoints", []), float)
        if len(pts) == 0:
            return 0
        a = np.asarray(start2, float)[:2]
        b = np.asarray(goal2, float)[:2]
        ab = b - a
        if float(np.linalg.norm(ab)) <= 1e-9:
            return 0
        vals = []
        for p in pts[:, :2]:
            rel = p - a
            vals.append(float(ab[0] * rel[1] - ab[1] * rel[0]))
        mean = float(np.mean(vals)) if vals else 0.0
        if abs(mean) < 1e-9:
            return 0
        return 1 if mean > 0.0 else -1

    def _hausdorff_corridor_distance(self, left, right, samples=24):
        a = self._resample(np.asarray(left.waypoints, float), max_points=samples)
        b = self._resample(np.asarray(right.waypoints, float), max_points=samples)
        if len(a) == 0 or len(b) == 0:
            return float("inf")
        d_ab = [
            float(np.min(np.linalg.norm(b[:, :2] - p[:2], axis=1)))
            for p in a]
        d_ba = [
            float(np.min(np.linalg.norm(a[:, :2] - p[:2], axis=1)))
            for p in b]
        return float(max(max(d_ab), max(d_ba)))

    def _dedupe_corridors_by_geometry(self, corridors, start2, goal2,
                                      min_keep=1):
        threshold = float(self.corridor_dedupe_distance)
        if threshold <= 0.0 or len(corridors) <= 1:
            return list(corridors)
        min_keep = max(1, int(min_keep or 1))
        kept = []
        skipped = []
        for candidate in corridors:
            side = self._corridor_side(candidate, start2, goal2)
            duplicate = False
            for old in kept:
                if side != self._corridor_side(old, start2, goal2):
                    continue
                if self._hausdorff_corridor_distance(candidate, old) < threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
            else:
                skipped.append(candidate)
        if len(kept) < min_keep:
            for candidate in skipped:
                candidate.geometry_duplicate_retained = True
                candidate.diversity_retention_reason = (
                    "candidate_pool_min_preserves_ranking_competition")
                kept.append(candidate)
                if len(kept) >= min_keep:
                    break
        return kept

    def _sort_with_saddle_tie(self, corridors):
        if not corridors:
            return []
        if getattr(self, "morse_decision_mode", "balanced") == "balanced":
            for corridor in corridors:
                corridor.topology_selection_window = 0.0
                corridor.morse_priority_applied = False
                corridor.topology_selection_reason = "balanced_weighted_cost"
            return sorted(
                corridors, key=lambda c: float(getattr(c, "cost", 0.0)))
        best_cost = min(float(getattr(c, "cost", 0.0)) for c in corridors)
        min_eps = self.grid_resolution * 0.25

        def key(corridor):
            cost = float(getattr(corridor, "cost", 0.0))
            role = getattr(corridor, "topology_role", "")
            if not role:
                role = self._corridor_role(getattr(corridor, "label", ""))
                corridor.topology_role = role
            priority = self._morse_priority_class(role)
            ratio = self._role_priority_ratio(role)
            window = max(abs(best_cost) * ratio, min_eps)
            near_best = cost <= best_cost + window
            corridor.topology_selection_window = float(window)
            corridor.morse_priority_applied = bool(
                self.morse_primary and near_best and priority < 3)
            if corridor.morse_priority_applied:
                corridor.topology_selection_reason = (
                    "%s_priority_within_%.2f" % (role, ratio))
                return (priority, cost)
            corridor.topology_selection_reason = "cost_order"
            return (3, cost)

        return sorted(corridors, key=key)

    def _corridor_role(self, label, kinds=None):
        label = str(label or "")
        kinds = kinds or []
        if label.startswith("morse_mix_"):
            return "morse_mix"
        if label.startswith("morse_saddle_") or "saddle" in kinds:
            return "morse_saddle"
        if label.startswith("morse_minima_") or "minimum" in kinds:
            return "morse_minima"
        if label.startswith("graph_semantic_"):
            return "graph_semantic"
        if label.startswith("graph_direct_"):
            return "graph_direct"
        return "graph"

    def _morse_priority_class(self, role):
        if role == "morse_saddle":
            return 0
        if role == "morse_mix":
            return 1
        if role == "morse_minima":
            return 2
        return 3

    def _role_priority_ratio(self, role):
        if role == "morse_saddle":
            return self.morse_saddle_priority_ratio
        if role == "morse_mix":
            return self.morse_mix_priority_ratio
        if role == "morse_minima":
            return self.morse_minima_priority_ratio
        return self.saddle_tie_ratio

    def _default_semantic_nodes(self, start, goal):
        start2 = self._plane(start)
        goal2 = self._plane(goal)
        direction = goal2 - start2
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            return []
        normal = np.array([-direction[1], direction[0]], float) / length
        mid = 0.5 * (start2 + goal2)
        offsets = [self.merge_radius * 2.0, -self.merge_radius * 2.0,
                   self.merge_radius * 3.5, -self.merge_radius * 3.5,
                   self.merge_radius * 5.0, -self.merge_radius * 5.0]
        nodes = []
        (xmin, xmax), (ymin, ymax) = self.bounds
        for off in offsets:
            p2 = mid + off * normal
            p2[0] = min(max(p2[0], xmin), xmax)
            p2[1] = min(max(p2[1], ymin), ymax)
            ij = self._nearest_safe_ij(p2)
            if ij is not None:
                nodes.append(self._world(self._ij_to_p2(ij)))
        return nodes

    def _path_phi(self, pts, mode):
        vals = [max(float(self.field.phi_s(p)), self._interest_risk(p))
                for p in pts]
        if not vals:
            return 0.0
        if mode == "max":
            return float(np.max(vals))
        return float(np.mean(vals))

    def _path_clearance(self, cells):
        vals = [self._clearance_cells(ij) for ij in cells]
        if not vals:
            return 0.0
        return float(np.min(vals))

    def _apply_wheelchair_adp(self, corridor, goal, critic, feature_builder,
                              feature_context):
        samples = int(feature_context.get("adp_samples", 9))
        pts = self._resample(corridor.waypoints, max_points=samples)
        raw_values = []
        cost_values = []
        for p in pts:
            pose2d = np.array([p[0], p[1], feature_context.get("yaw", 0.0)], float)
            features = feature_builder.build_wheelchair(
                pose2d,
                np.asarray(goal, float)[:2],
                self.field,
                gate_info=feature_context.get("gate_info", {}),
                interest_risk=feature_context.get("interest_risk", {}),
                corridor=corridor,
                u=feature_context.get("u"))
            detail = critic.predict_detail(features)
            clipped = float(detail.get("clipped", detail.get("raw", 0.0)))
            raw_values.append(clipped)
            cost_values.append(max(0.0, clipped))
        arr = np.asarray(raw_values, float)
        cost_arr = np.asarray(cost_values, float)
        if len(arr):
            corridor.adp_raw_mean = float(np.mean(arr))
            corridor.adp_raw_max = float(np.max(arr))
            corridor.adp_raw_end = float(arr[-1])
        weights = feature_context.get("adp_corridor_weights", {})
        w_mean = float(weights.get("mean", 0.4))
        w_max = float(weights.get("max", 0.4))
        w_end = float(weights.get("end", 0.2))
        cost_mean = float(np.mean(cost_arr)) if len(cost_arr) else 0.0
        cost_max = float(np.max(cost_arr)) if len(cost_arr) else 0.0
        cost_end = float(cost_arr[-1]) if len(cost_arr) else 0.0
        corridor._adp_raw_score = (
            w_mean * cost_mean + w_max * cost_max + w_end * cost_end)
        corridor.adp_mean = cost_mean
        corridor.adp_max = cost_max
        corridor.adp_end = cost_end
