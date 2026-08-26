#!/usr/bin/env python
import csv
import copy
import heapq
import json
import os
import sys
sys.dont_write_bytecode = True
import time
import numpy as np
import rospy
from std_msgs.msg import Bool, Float64, Float64MultiArray, String
from geometry_msgs.msg import Twist, PointStamped
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SetModelState
from tf.transformations import euler_from_quaternion, quaternion_from_euler

PACKAGE_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if os.path.isdir(os.path.join(PACKAGE_SRC, "stsm_madp")) and PACKAGE_SRC not in sys.path:
    sys.path.insert(0, PACKAGE_SRC)

from stsm_madp.social_field import HumanState, SemanticAnchor, SocialField, SocialFieldParams
from stsm_madp.manifold import SafetyManifold, Corridor
from stsm_madp.corridor import require_corridor_contract
from stsm_madp.mpc import (
    WheelchairMPC, build_mpc_constraint_inputs, generate_topology_tube,
    run_mpc_tracking, wheelchair_nonholonomic_execution_profile,
    write_mpc_outputs)
from stsm_madp.topology_constraint import write_topology_constraint
from stsm_madp.safety_gate import SafetyGate, SafetyGateResult
from stsm_madp.adp import ADPCritic, ADPFeatureBuilder
from stsm_madp.topology import topology_param_or_auto, topology_profile_defaults
from stsm_madp.topology_candidate_generator import (
    recover_candidate_corridor_feasibility)
from stsm_madp.topology_refinement import (
    refine_topology_path, smooth_wheelchair_corners)
from stsm_madp.decision_trace import trace_from_debug, write_trace
from stsm_madp.topology_diagnostics_writer import (
    _jsonable, write_failed_topology_diagnostics,
    write_stsm_candidate_ranking_alias)
from stsm_madp.critical_point_association import (
    associate_corridor_critical_points, write_critical_point_association)
from stsm_madp.interest_points import (
    DEFAULT_WC_LOCAL_POINTS, WC_LABELS, transform_points_2d,
    aggregate_point_risks, forbidden_anchor_hit, pose_interest_risk)
from stsm_madp.task_config import resolve_task_mode, resolve_task_weight
from stsm_madp.task_semantics import (
    evaluate_task_cost_breakdown, infer_task_state)

def _pt(d):
    return np.array(d, float)

def _topology_param(value, cast=float):
    value = topology_param_or_auto(value)
    if value is None:
        return None
    return cast(value)

def _effective(value, default):
    return default if value is None else value


def _resolve_manifold_constraint_mode(mode_a, mode_b=None, node_name="stsm"):
    mode_a = str(mode_a if mode_a not in (None, "") else "soft").strip().lower()
    mode_b = str(mode_b if mode_b not in (None, "") else mode_a).strip().lower()
    if mode_a not in ("soft", "hard"):
        raise ValueError("%s invalid manifold_constraint_mode=%s" %
                         (node_name, mode_a))
    if mode_b not in ("soft", "hard"):
        raise ValueError("%s invalid mpc_manifold_constraint_mode=%s" %
                         (node_name, mode_b))
    if mode_a != mode_b:
        raise ValueError(
            "%s inconsistent manifold constraint mode: %s vs %s" %
            (node_name, mode_a, mode_b))
    return mode_a

class WheelchairNode:
    def __init__(self):
        rospy.init_node("stsm_wheelchair")
        self.baseline = rospy.get_param("~baseline", False)
        self.experiment_mode = str(rospy.get_param(
            "~experiment_mode", "debug")).strip().lower()
        if self.experiment_mode not in ("debug", "paper"):
            self.experiment_mode = "debug"
        self.baseline_type = str(rospy.get_param(
            "~baseline_type", "direct")).strip().lower()
        try:
            self.manifold_constraint_mode = _resolve_manifold_constraint_mode(
                rospy.get_param("~mpc/manifold_constraint_mode", "soft"),
                rospy.get_param("~mpc/mpc_manifold_constraint_mode", None),
                "stsm_wheelchair")
        except ValueError as exc:
            rospy.logfatal(str(exc))
            raise
        self.manifold_soft_tolerance = float(rospy.get_param(
            "~mpc/manifold_soft_tolerance", 0.08))
        self.manifold_hard_tolerance = float(rospy.get_param(
            "~mpc/manifold_hard_tolerance", 0.25))
        aliases = {
            "classic": "traditional",
            "original": "traditional",
            "planner": "traditional",
            "traditional_planner": "traditional",
            "baseline0": "direct",
            "baseline-0": "direct",
            "b0": "direct",
            "baseline1": "mpc_safe",
            "baseline-1": "mpc_safe",
            "b1": "mpc_safe",
        }
        self.baseline_type = aliases.get(self.baseline_type, self.baseline_type)
        if self.baseline_type not in ("traditional", "direct", "mpc_safe"):
            raise ValueError("unknown wheelchair baseline_type={}".format(
                self.baseline_type))
        self.goal = _pt(rospy.get_param("~goal", [-0.55, 0.55]))
        self.goal_tolerance = float(rospy.get_param("~goal_tolerance", 0.08))
        self.completion_tolerance = float(rospy.get_param(
            "~completion_tolerance", 0.25))
        self.completion_hold_s = float(rospy.get_param(
            "~completion_hold_s", 1.5))
        self.strict_goal_completion = bool(rospy.get_param(
            "~strict_goal_completion", False))
        self.max_runtime_s = float(rospy.get_param("~max_runtime_s", 180.0))
        self.no_progress_timeout_s = float(rospy.get_param(
            "~no_progress_timeout_s", 45.0))
        self.no_progress_epsilon = float(rospy.get_param(
            "~no_progress_epsilon", 0.02))
        self.replan_period = float(rospy.get_param("~replan_period", 5.0))
        self.no_progress_replan_time = float(rospy.get_param(
            "~no_progress_replan_time", 5.0))
        self.progress_eps = float(rospy.get_param("~progress_eps", 0.01))
        self.replan_tube_margin = float(rospy.get_param(
            "~replan_tube_margin", 0.08))
        self.replan_min_budget_s = float(rospy.get_param(
            "~replan_min_budget_s", 15.0))
        self.replan_budget_factor = float(rospy.get_param(
            "~replan_budget_factor", 1.25))
        self.near_goal_radius = float(rospy.get_param("~near_goal_radius", 0.50))
        self.near_goal_adp_scale = float(rospy.get_param(
            "~near_goal_adp_scale", 0.20))
        self.min_progress_per_solve = float(rospy.get_param(
            "~min_progress_per_solve", 0.005))
        self.near_goal_goal_weight = float(rospy.get_param(
            "~near_goal_goal_weight", 18.0))
        self.near_goal_social_scale = float(rospy.get_param(
            "~near_goal_social_scale", 0.5))
        self.lam_stall = float(rospy.get_param("~lam_stall", 10.0))
        self.progress_reward_weight = float(rospy.get_param(
            "~progress_reward_weight", 2.8))
        self.speed_reward_weight = float(rospy.get_param(
            "~speed_reward_weight", 0.25))
        self.ref_progress_reward_weight = float(rospy.get_param(
            "~ref_progress_reward_weight", 1.0))
        self.corridor_speed_slowdown_gain = float(rospy.get_param(
            "~corridor_speed_slowdown_gain", 0.18))
        self.corridor_tube_gain = float(rospy.get_param(
            "~corridor_tube_gain", 0.80))
        self.baseline_corridor_follow_gain = float(rospy.get_param(
            "~baseline_corridor_follow_gain", 0.8))
        self.baseline_corridor_follow_blend = float(rospy.get_param(
            "~baseline_corridor_follow_blend", 0.15))
        self.baseline_corridor_follow_lookahead = int(rospy.get_param(
            "~baseline_corridor_follow_lookahead", 4))
        self.baseline_grid_resolution = float(rospy.get_param(
            "~baseline_grid_resolution", 0.10))
        self.baseline_risk_weight = float(rospy.get_param(
            "~baseline_risk_weight", 2.5))
        self.baseline_turn_weight = float(rospy.get_param(
            "~baseline_turn_weight", 0.08))
        self.baseline_min_lateral_deviation = float(rospy.get_param(
            "~baseline_min_lateral_deviation", 0.25))
        self.baseline_direct_retry_weight = float(rospy.get_param(
            "~baseline_direct_retry_weight", 8.0))
        self.final_approach_radius = float(rospy.get_param(
            "~final_approach_radius", 0.90))
        self.final_approach_entry_radius = float(rospy.get_param(
            "~final_approach_entry_radius", 0.50))
        self.final_approach_goal_weight_scale = float(rospy.get_param(
            "~final_approach_goal_weight_scale", 2.0))
        self.final_approach_social_scale = float(rospy.get_param(
            "~final_approach_social_scale", 0.35))
        self.final_heading_threshold = float(rospy.get_param(
            "~final_heading_threshold", 0.75))
        self.final_heading_gain = float(rospy.get_param(
            "~final_heading_gain", 1.6))
        self.final_creep_v = float(rospy.get_param("~final_creep_v", 0.10))
        self.final_min_v = float(rospy.get_param("~final_min_v", 0.16))
        self.final_max_v = float(rospy.get_param("~final_max_v", 0.30))
        self.final_forward_gain = float(rospy.get_param(
            "~final_forward_gain", 0.75))
        self.stsm_w_max = float(rospy.get_param("~stsm_w_max", 0.95))
        self.stsm_progress_floor_v = float(rospy.get_param(
            "~stsm_progress_floor_v", 0.12))
        self.stsm_progress_floor_min_gate_scale = float(rospy.get_param(
            "~stsm_progress_floor_min_gate_scale", 0.85))
        self.stsm_progress_floor_min_adp_scale = float(rospy.get_param(
            "~stsm_progress_floor_min_adp_scale", 0.85))
        self.stsm_progress_floor_w_max = float(rospy.get_param(
            "~stsm_progress_floor_w_max", 0.65))
        self.stsm_liveness_progress_stale_s = float(rospy.get_param(
            "~stsm_liveness_progress_stale_s", self.no_progress_replan_time))
        self.stsm_liveness_floor_v = float(rospy.get_param(
            "~stsm_liveness_floor_v", 0.14))
        self.stsm_liveness_w_max = float(rospy.get_param(
            "~stsm_liveness_w_max", min(0.65, self.stsm_w_max)))
        self.final_w_max = float(rospy.get_param("~final_w_max", 0.85))
        self.w_slew_limit = float(rospy.get_param("~w_slew_limit", 1.00))
        self.lam_heading = float(rospy.get_param("~lam_heading", 2.5))
        self.final_direct_override_enabled = bool(rospy.get_param(
            "~final_direct_override_enabled", True))
        self.final_direct_override_radius = float(rospy.get_param(
            "~final_direct_override_radius", self.final_approach_radius))
        if self.completion_tolerance < self.goal_tolerance:
            self.completion_tolerance = self.goal_tolerance
        self.execution_stop_tolerance = (
            self.goal_tolerance if self.strict_goal_completion
            else self.completion_tolerance)
        self.use_world = rospy.get_param("~use_world_pose", True)
        self.model_name = rospy.get_param("~model_name", "wheelchair")
        self.state_source = "/gazebo/model_states" if self.use_world else "/wheelchair/diff_drive_controller/odom"
        self.reset_on_start = rospy.get_param("~reset_on_start", True)
        self.start_pose = _pt(rospy.get_param("~start_pose", [2.0, 1.5, -2.4]))
        self.state = None
        self.world_vel = np.zeros(3)
        self.velocity_valid = False
        self.u_prev = np.zeros(2)
        self.last_cmd_twist = Twist()
        self.last_cmd_time = rospy.Time(0)
        self.command_hold_s = float(rospy.get_param(
            "~command_hold_s", 4.0))
        self.command_keepalive_hz = float(rospy.get_param(
            "~command_keepalive_hz", 8.0))
        self.command_keepalive_enabled = bool(rospy.get_param(
            "~command_keepalive_enabled", True))
        self.command_keepalive_publish_count = 0
        self.stop_triggered = False
        self.stop_reason = ""
        self.task_completed = False
        self.adp_requested = bool(rospy.get_param("~adp_enabled", True))
        self.adp_enabled = self.adp_requested and not self.baseline
        self.adp_model = rospy.get_param(
            "~adp_model",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                         "config", "adp_critic.yaml")))
        self.lambda_adp = float(rospy.get_param("~lambda_adp", 0.005))
        self.lambda_adp_corridor = float(rospy.get_param(
            "~lambda_adp_corridor", self.lambda_adp))
        self.lambda_adp_terminal = float(rospy.get_param(
            "~lambda_adp_terminal", self.lambda_adp))
        self.mpc_use_adp_terminal = bool(rospy.get_param(
            "~mpc_use_adp_terminal", True))
        self.adp_post_scale_enabled = bool(rospy.get_param(
            "~adp_post_scale_enabled", False))
        self.adp_min_scale = float(rospy.get_param("~adp_min_scale", 0.35))
        self.adp_debug = bool(rospy.get_param("~adp_debug", False))
        self.adp_critic = None
        self.adp_features = ADPFeatureBuilder()
        self.last_adp_value = 0.0
        self.selected_corridor = None
        self.execution_corridor = None
        self.last_valid_topology_debug = {}
        self.runtime_topology_candidate_pool = []
        self.runtime_rejected_topology_corridor_ids = set()
        self.runtime_topology_candidate_switch_trials = []
        self.runtime_topology_candidate_switch_count = 0
        self.runtime_replan_fallback_count = 0
        self.last_corridor_plan_duration_s = 0.0
        self.replan_deadline_skip_count = 0
        self.decision_trace_out = rospy.get_param("~decision_trace_out", "")
        self.mpc_reference_out = rospy.get_param("~mpc_reference_out", "")
        self.mpc_diagnostics_out = rospy.get_param("~mpc_diagnostics_out", "")
        self.mpc_cost_breakdown_out = rospy.get_param("~mpc_cost_breakdown_out", "")
        self.mpc_reference_records = []
        self.mpc_executed_records = []
        self.mpc_runtime_records = []
        self.baseline_reference_records = []
        self.baseline_mpc_output_records = []
        self._baseline_reference_solve_index = 0
        self._mpc_reference_solve_index = 0
        self.topology_profile = str(rospy.get_param(
            "~topology/profile", "wheelchair")).strip().lower()
        profile_defaults = topology_profile_defaults(self.topology_profile)
        self.topology_enabled = bool(rospy.get_param("~topology/enabled", True))
        self.topology_grid_resolution = _topology_param(rospy.get_param(
            "~topology/grid_resolution", None))
        self.topology_merge_radius = _topology_param(rospy.get_param(
            "~topology/merge_radius", None))
        self.topology_min_clearance = _topology_param(rospy.get_param(
            "~topology/min_clearance", None))
        self.topology_hard_clearance = _topology_param(rospy.get_param(
            "~topology/hard_clearance", None))
        self.topology_neighbor_k = _topology_param(rospy.get_param(
            "~topology/neighbor_k", None), int)
        self.topology_profile_defaults = profile_defaults
        self.topology_saddle_tie_ratio = float(rospy.get_param(
            "~topology/saddle_tie_ratio", 0.05))
        self.topology_morse_priority_ratio = float(rospy.get_param(
            "~topology/morse_priority_ratio", 0.25))
        self.topology_morse_saddle_priority_ratio = float(rospy.get_param(
            "~topology/morse_saddle_priority_ratio", 0.50))
        self.topology_morse_mix_priority_ratio = float(rospy.get_param(
            "~topology/morse_mix_priority_ratio", 0.50))
        self.topology_morse_minima_priority_ratio = float(rospy.get_param(
            "~topology/morse_minima_priority_ratio", 0.25))
        self.topology_morse_core_required = bool(rospy.get_param(
            "~topology/morse_core_required",
            profile_defaults["morse_core_required"]))
        self.topology_morse_decision_mode = str(rospy.get_param(
            "~topology/morse_decision_mode", "balanced")).strip().lower()
        self.topology_morse_w_goal = float(rospy.get_param(
            "~topology/morse_w_goal", 0.35))
        self.topology_morse_w_social = float(rospy.get_param(
            "~topology/morse_w_social", 1.0))
        self.topology_morse_w_barrier = float(rospy.get_param(
            "~topology/morse_w_barrier", 0.6))
        self.topology_morse_grad_eps = float(rospy.get_param(
            "~topology/morse_grad_eps", 0.6))
        self.topology_k_paths = int(rospy.get_param("~topology/k_paths", 3))
        self.topology_max_graph_nodes = int(rospy.get_param(
            "~topology/max_graph_nodes", 32))
        self.topology_safety_regions = rospy.get_param(
            "~topology/safety_regions", [
                {
                    "name": "bedside_docking",
                    "shape": "circle",
                    "center": [-0.55, 0.55],
                    "radius": 0.70,
                    "rho": 2.4,
                    "interest_rho": 7.5,
                    "min_clearance": 0.10,
                    "hard_clearance": 0.04,
                },
                {
                    "name": "door_yield",
                    "shape": "rect",
                    "center": [0.6, 1.05],
                    "half_extent": [0.9, 0.55],
                    "rho": 1.6,
                    "interest_rho": 5.5,
                    "min_clearance": 0.22,
                    "hard_clearance": 0.10,
                },
            ])
        self.topology_allow_semantic_with_morse = bool(rospy.get_param(
            "~topology/allow_semantic_with_morse", False))
        self.topology_allow_semantic_topology_recovery = bool(rospy.get_param(
            "~topology/allow_semantic_topology_recovery", True))
        self.topology_allow_ring_with_morse = bool(rospy.get_param(
            "~topology/allow_ring_with_morse", False))
        self.topology_allow_graph_fallback_with_morse = bool(rospy.get_param(
            "~topology/allow_graph_fallback_with_morse", False))
        self.topology_lambda_execution = float(rospy.get_param(
            "~topology/lambda_execution", 0.20))
        self.topology_lambda_tracking = float(rospy.get_param(
            "~topology/lambda_tracking", profile_defaults["lambda_tracking"]))
        self.topology_lambda_saddle_value = float(rospy.get_param(
            "~topology/lambda_saddle_value",
            profile_defaults["lambda_saddle_value"]))
        self.topology_max_corridor_turn = float(rospy.get_param(
            "~topology/max_corridor_turn", 3.40))
        self.topology_max_corridor_curvature = float(rospy.get_param(
            "~topology/max_corridor_curvature", 40.0))
        self.topology_min_segment_length = float(rospy.get_param(
            "~topology/min_segment_length", 0.01))
        self.topology_corridor_dedupe_distance = float(rospy.get_param(
            "~topology/corridor_dedupe_distance", 0.05))
        self.topology_candidate_pool_min = int(rospy.get_param(
            "~topology/candidate_pool_min", 3))
        self.topology_route_max_paths = max(1, int(rospy.get_param(
            "~topology/route_max_paths", 512)))
        self.topology_route_max_routes = max(1, int(rospy.get_param(
            "~topology/route_max_routes", 256)))
        if not self.baseline:
            self.topology_route_max_paths = min(
                int(self.topology_route_max_paths), 64)
            self.topology_route_max_routes = min(
                int(self.topology_route_max_routes), 32)
        self.topology_require_risk_improvement = bool(rospy.get_param(
            "~topology/require_risk_improvement", True))
        self.topology_candidate_max_risk = _topology_param(rospy.get_param(
            "~topology/candidate_max_risk",
            float(rospy.get_param("~safety_gate/rho_stop", 2.5))))
        self.topology_corridor_score_weights = rospy.get_param(
            "~topology/corridor_score_weights",
            {"risk": 4.0, "max_risk": 2.0, "length": 1.0,
             "task": 2.0, "smooth": 2.0, "motion": 2.0,
             "curvature": 1.5, "exec": 1.0, "topology": 0.35})
        self.task_config = dict(rospy.get_param("~task_config", {}) or {})
        self.task_mode = resolve_task_mode(rospy.get_param(
            "~task_mode", "navigation"), robot_type="wheelchair")
        self.task_weight = resolve_task_weight(
            self.task_mode, task_config=self.task_config,
            task_weight=rospy.get_param("~task_weight", {}),
            robot_type="wheelchair")
        self.mpc_cost_weights = dict(rospy.get_param(
            "~mpc/weights", {}) or {})
        self.mpc_phase_cost_weights = dict(rospy.get_param(
            "~mpc/phase_cost_weights", {}) or {})
        self.topology_preferred_side = str(rospy.get_param(
            "~topology/preferred_side", "auto")).strip().lower()
        self.recovery_max_footprint_phi = float(rospy.get_param(
            "~topology/recovery_max_footprint_phi",
            float(rospy.get_param("~interest_points/rho_stop", 7.0))))
        self.recovery_max_mean_footprint_phi = float(rospy.get_param(
            "~topology/recovery_max_mean_footprint_phi", 1.25))
        self.recovery_max_center_phi = float(rospy.get_param(
            "~topology/recovery_max_center_phi",
            float(rospy.get_param("~safety_gate/rho_stop", 2.5))))
        self.recovery_mpc_tracking_margin = float(rospy.get_param(
            "~topology/recovery_mpc_tracking_margin", 0.30))
        self.safe_fallback_max_footprint_phi = float(rospy.get_param(
            "~topology/safe_fallback_max_footprint_phi",
            float(rospy.get_param("~interest_points/rho_stop", 7.0))))
        self.safe_fallback_max_mean_footprint_phi = float(rospy.get_param(
            "~topology/safe_fallback_max_mean_footprint_phi", 1.25))
        self.safe_fallback_max_center_phi = float(rospy.get_param(
            "~topology/safe_fallback_max_center_phi",
            float(rospy.get_param("~safety_gate/rho_stop", 2.5))))
        self.topology_fallback_enabled = bool(rospy.get_param(
            "~topology/fallback_enabled", True))
        if self.experiment_mode == "paper" and not self.baseline:
            self.topology_fallback_enabled = False
        self.topology_periodic_replan = bool(rospy.get_param(
            "~topology/periodic_replan", False))
        self.topology_replan_min_interval = float(rospy.get_param(
            "~topology/replan_min_interval", 30.0))
        self.topology_replan_on_tube_exit = bool(rospy.get_param(
            "~topology/replan_on_tube_exit", True))
        self.topology_replan_on_no_progress = bool(rospy.get_param(
            "~topology/replan_on_no_progress", True))
        self.runtime_blocking_replan_enabled = bool(rospy.get_param(
            "~topology/runtime_blocking_replan_enabled", False))
        self.topology_refinement_enabled = bool(rospy.get_param(
            "~topology/refinement_enabled", True))
        self.topology_refinement_samples = int(rospy.get_param(
            "~topology/refinement_samples_per_segment", 14))
        if not self.baseline:
            self.topology_refinement_samples = min(
                int(self.topology_refinement_samples), 6)
        self.topology_refinement_max_candidates = int(rospy.get_param(
            "~topology/refinement_max_candidates",
            self.topology_candidate_pool_min))
        self.last_refined_footprint_max = 0.0
        self.last_refined_footprint_mean = 0.0
        self.last_refined_footprint_checked_points = 0
        self.max_refined_footprint_check_points = max(8, int(rospy.get_param(
            "~topology/max_refined_footprint_check_points", 48)))
        self.max_refinement_path_points = max(8, int(rospy.get_param(
            "~topology/max_refinement_path_points", 48)))
        self.interest_enabled = rospy.get_param("~interest_points/enabled", True)
        self.interest_gate_enabled = rospy.get_param(
            "~interest_points/gate_enabled", True)
        self.footprint_forbidden_stop_enabled = rospy.get_param(
            "~interest_points/forbidden_stop_enabled", True)
        self.wc_ip_labels = list(WC_LABELS)
        self.wc_local_points = dict(DEFAULT_WC_LOCAL_POINTS)
        ns = "/wheelchair/diff_drive_controller"
        self.cmd_pub = rospy.Publisher(ns + "/cmd_vel", Twist, queue_size=1)
        self.phi_pub = rospy.Publisher("/stsm/wc_phi_s", Float64, queue_size=10)
        self.risk_components_pub = rospy.Publisher(
            "/stsm/wc_risk_components", Float64MultiArray, queue_size=10)
        self.velocity_monitor_pub = rospy.Publisher(
            "/stsm/wc_velocity_monitor", Float64MultiArray, queue_size=10)
        self.gate_pub = rospy.Publisher(
            "/stsm/wc_gate_state", String, queue_size=10, latch=True)
        self.gate_info_pub = rospy.Publisher(
            "/stsm/wc_gate_info", Float64MultiArray, queue_size=10)
        self.gate_reason_pub = rospy.Publisher(
            "/stsm/wc_gate_reason", String, queue_size=10, latch=True)
        self.gate_source_pub = rospy.Publisher(
            "/stsm/wc_gate_source", String, queue_size=10, latch=True)
        self.interest_gate_info_pub = rospy.Publisher(
            "/stsm/wc_interest_gate_info", Float64MultiArray, queue_size=10)
        self.pos_pub = rospy.Publisher("/stsm/wc_pos", PointStamped, queue_size=10)
        self.wc_interest_pub = rospy.Publisher(
            "/stsm/wc_interest_risk", Float64MultiArray, queue_size=10)
        self.wc_pose2d_pub = rospy.Publisher(
            "/stsm/wc_pose2d", Float64MultiArray, queue_size=10)
        self.mode_pub = rospy.Publisher("/stsm/wc_mode", String, queue_size=1, latch=True)
        self.task_complete_pub = rospy.Publisher(
            "/stsm/wc_task_complete", Bool, queue_size=1, latch=True)
        self.adp_value_pub = rospy.Publisher(
            "/stsm/wc_adp_value", Float64, queue_size=10)
        self.adp_feature_pub = rospy.Publisher(
            "/stsm/adp_features", Float64MultiArray, queue_size=10)
        self.adp_status_pub = rospy.Publisher(
            "/stsm/adp_status", String, queue_size=10, latch=True)
        self.selected_corridor_pub = rospy.Publisher(
            "/stsm/wc_selected_corridor", String, queue_size=10, latch=True)
        self.adp_mpc_info_pub = rospy.Publisher(
            "/stsm/wc_adp_mpc_info", Float64MultiArray, queue_size=10)
        self.topology_info_pub = rospy.Publisher(
            "/stsm/wc_topology_info", Float64MultiArray, queue_size=10,
            latch=True)
        if self.use_world:
            rospy.Subscriber("/gazebo/model_states", ModelStates,
                             self._model_cb, queue_size=1)
        else:
            rospy.Subscriber(ns + "/odom", Odometry, self._odom_cb)
        self._build_scene()
        self._load_adp()

    def _corridor_id(self, corridor, default=""):
        if corridor is None:
            return str(default or "")
        cid = str(getattr(corridor, "corridor_id", "") or "")
        if cid:
            return cid
        label = str(getattr(corridor, "label", "") or "")
        return label or str(default or "")

    def _corridor_label(self, corridor, default=""):
        if corridor is None:
            return str(default or "")
        return str(getattr(corridor, "label", "") or "") or self._corridor_id(
            corridor, default=default)

    def _valid_reference_source(self, source):
        return str(source or "") in (
            "refined", "refined_waypoints", "turn_recovered_refined",
            "candidate_fallback", "refinement", "candidate", "fallback",
            "selected_candidate_waypoints", "raw_waypoints",
            "diff_drive_launch_prefix", "raw_diff_drive_launch_prefix",
            "diff_drive_launch_prefix_turn_recovered",
            "runtime_replan_fallback")

    def _ensure_corridor_runtime_contract(self, corridor,
                                          fallback_id="wheelchair_runtime_c0001",
                                          fallback_source="runtime_replan_fallback"):
        if corridor is None:
            return None
        strict_stsm = bool(not self.baseline)
        if strict_stsm:
            try:
                require_corridor_contract(
                    corridor, require_morse=True, require_tube=True)
            except ValueError as exc:
                rospy.logerr("[wc][corridor_contract] reject: %s", exc)
                return None
        cid = self._corridor_id(corridor, fallback_id)
        if not cid:
            cid = fallback_id
        if not str(getattr(corridor, "corridor_id", "") or ""):
            corridor.corridor_id = cid
        if not str(getattr(corridor, "label", "") or ""):
            corridor.label = cid
        points, source = self._sync_selected_corridor_geometry(corridor)
        if len(points) == 0:
            return None
        reference_source = str(getattr(corridor, "final_reference_source", "") or "")
        if not self._valid_reference_source(reference_source):
            corridor.final_reference_source = fallback_source
        if not strict_stsm and not list(getattr(corridor, "node_sequence", []) or []):
            corridor.node_sequence = ["start", "goal"]
        if not strict_stsm and not list(getattr(corridor, "topology_nodes", []) or []):
            corridor.topology_nodes = list(corridor.node_sequence)
        if not strict_stsm and not list(getattr(corridor, "node_type_sequence", []) or []):
            corridor.node_type_sequence = ["start", "goal"]
        try:
            association = associate_corridor_critical_points(corridor, points)
        except Exception:
            association = {"critical_points": []}
        corridor.critical_point_association = association
        corridor.critical_point_projection_index = {
            str(item.get("id", "")): int(item.get("trajectory_index", -1))
            for item in association.get("critical_points", [])
            if isinstance(item, dict)
        }
        corridor.reference_path_count = int(len(points))
        corridor.reference_source = str(getattr(
            corridor, "final_reference_source", fallback_source))
        return corridor

    def _load_adp(self):
        if not self.adp_enabled:
            reason = "baseline" if self.baseline else "parameter"
            self.adp_status_pub.publish(String(
                "wheelchair ADP disabled: %s" % reason))
            return
        try:
            self.adp_critic = ADPCritic.load_yaml(self.adp_model)
            self.adp_features = ADPFeatureBuilder(self.adp_critic.feature_names)
            self.adp_status_pub.publish(String(
                "wheelchair ADP loaded: %s" % self.adp_critic.critic_version))
            rospy.loginfo("[wc][adp] loaded %s (%s)",
                          self.adp_model, self.adp_critic.critic_version)
        except Exception as exc:
            self.adp_enabled = False
            self.adp_critic = None
            self.adp_status_pub.publish(String(
                "wheelchair ADP disabled: %s" % exc))
            rospy.logwarn("[wc][adp] cannot load %s: %s",
                          self.adp_model, exc)

    def _reset_model_pose(self):
        if not self.reset_on_start:
            return
        if len(self.start_pose) != 3:
            rospy.logwarn("[wc] invalid start_pose=%s; expected [x, y, yaw]",
                          self.start_pose)
            return
        try:
            rospy.wait_for_service("/gazebo/set_model_state", timeout=10.0)
            set_state = rospy.ServiceProxy("/gazebo/set_model_state",
                                           SetModelState)
            msg = ModelState()
            msg.model_name = self.model_name
            msg.reference_frame = "world"
            msg.pose.position.x = float(self.start_pose[0])
            msg.pose.position.y = float(self.start_pose[1])
            msg.pose.position.z = 0.05
            q = quaternion_from_euler(0.0, 0.0, float(self.start_pose[2]))
            msg.pose.orientation.x = q[0]
            msg.pose.orientation.y = q[1]
            msg.pose.orientation.z = q[2]
            msg.pose.orientation.w = q[3]
            resp = set_state(msg)
            if resp.success:
                rospy.loginfo("[wc] reset %s to start pose %s",
                              self.model_name, np.round(self.start_pose, 3))
            else:
                rospy.logwarn("[wc] failed to reset %s: %s",
                              self.model_name, resp.status_message)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logwarn("[wc] cannot reset %s through Gazebo: %s",
                          self.model_name, exc)

    def _build_scene(self):

        self.human = HumanState(pos=[-1.6, 0.2, 0.0], heading=np.pi / 2,
                                posture="transferring", vulnerability=1.4)
        bed = SemanticAnchor("bed", [-1.6, -1.0, 0.0], [0.5, 1.0, 0.5],
                             weight=2.0, forbidden=True)
        transfer = SemanticAnchor("transfer-zone", [-0.7, -1.0, 0.0],
                                  [0.4, 1.0, 0.5], weight=2.5, forbidden=True)
        table = SemanticAnchor("table", [0.55, 0.0, 0.0], [0.3, 0.5, 0.4],
                               weight=1.0, forbidden=True)
        self.field = SocialField(SocialFieldParams(
            lam_prox=1.2, lam_close=1.0, lam_dir=0.5, lam_body=0.0,
            lam_env=1.5, sigma_env=0.4))
        self.field.set_scene([self.human], [bed, transfer, table])
        self.manifold = SafetyManifold(self.field, rho=2.0, lam_s=1.0)
        self.mpc = WheelchairMPC(
            horizon=int(rospy.get_param("~mpc/horizon", 12)),
            dt=float(rospy.get_param("~mpc/dt", 0.2)),
            v_max=0.75,
            w_max=(1.2 if self.baseline else self.stsm_w_max),
            a_max=float(rospy.get_param("~mpc/a_max", 0.5)),
            alpha_max=float(rospy.get_param("~mpc/alpha_max", 1.5)),
            beam_width=int(rospy.get_param("~mpc/beam_width", 12)),
            lam_social=0.4)
        self.mpc_base_v_max = float(self.mpc.v_max)
        self.mpc_base_lam_tube = float(self.mpc.lam_tube)
        self.mpc_base_lam_track = float(self.mpc.lam_track)
        self.mpc_base_lam_social = float(self.mpc.lam_social)
        self.mpc_base_lam_goal_terminal = float(self.mpc.lam_goal_terminal)
        self.mpc.min_progress_per_solve = self.min_progress_per_solve
        self.mpc.near_goal_radius = self.near_goal_radius
        self.mpc.near_goal_goal_weight = self.near_goal_goal_weight
        self.mpc.near_goal_adp_scale = self.near_goal_adp_scale
        self.mpc.near_goal_social_scale = self.near_goal_social_scale
        self.mpc.lam_stall = self.lam_stall
        self.mpc.lam_progress = self.progress_reward_weight
        self.mpc.lam_speed = self.speed_reward_weight
        self.mpc.lam_ref_progress = self.ref_progress_reward_weight
        self.mpc.final_approach_radius = (
            0.0 if self.baseline else self.final_approach_entry_radius)
        self.mpc.final_heading_threshold = self.final_heading_threshold
        self.mpc.final_heading_gain = self.final_heading_gain
        self.mpc.final_creep_v = self.final_creep_v
        self.mpc.final_min_v = self.final_min_v
        self.mpc.final_max_v = self.final_max_v
        self.mpc.final_forward_gain = self.final_forward_gain
        self.mpc.lam_heading = self.lam_heading
        self.mpc.first_step_progress_ratio = float(rospy.get_param(
            "~mpc/first_step_progress_ratio", 0.50))
        self.mpc.heading_recovery_w_max = float(rospy.get_param(
            "~mpc/heading_recovery_w_max", self.stsm_w_max))
        self.bounds = [(-2.0, 2.5), (-2.0, 2.0)]
        self.gate = SafetyGate(
            rho_warn=rospy.get_param("~safety_gate/rho_warn", 1.6),
            rho_stop=rospy.get_param("~safety_gate/rho_stop", 2.5),
            min_scale=rospy.get_param("~safety_gate/min_scale", 0.20),
            enabled=rospy.get_param("~safety_gate/enabled", True))
        self.footprint_gate = SafetyGate(
            rho_warn=rospy.get_param("~interest_points/rho_warn", 5.0),
            rho_stop=rospy.get_param("~interest_points/rho_stop", 7.0),
            min_scale=rospy.get_param("~interest_points/min_scale", 0.20),
            enabled=self.interest_gate_enabled)
        self.abort_on_stop = bool(
            rospy.get_param("~safety_gate/abort_on_stop", True))

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.state = np.array([p.x, p.y, yaw])
        v = float(msg.twist.twist.linear.x)
        self.world_vel = np.array([v * np.cos(yaw), v * np.sin(yaw), 0.0])
        self.velocity_valid = True

    def _model_cb(self, msg):
        if self.model_name not in msg.name:
            return
        i = msg.name.index(self.model_name)
        p = msg.pose[i].position
        q = msg.pose[i].orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.state = np.array([p.x, p.y, yaw])
        tw = msg.twist[i].linear
        self.world_vel = np.array([tw.x, tw.y, tw.z], float)
        self.velocity_valid = True

    def _direct_baseline_corridor(self, start):
        label = "baseline_%s" % self.baseline_type
        start3 = np.array([start[0], start[1], 0.0], float)
        goal3 = np.array([self.goal[0], self.goal[1], 0.0], float)
        direct = np.array([start3, goal3], float)
        corr = Corridor(direct, radius=0.18, label=label)
        direct_stats = self._corridor_footprint_stats(corr)
        stats = direct_stats
        direct_safe = (
            not bool(direct_stats.get("forbidden", False)) and
            float(direct_stats.get("max_footprint_phi", 0.0)) <=
            float(self.footprint_gate.rho_warn))
        if not direct_safe:
            wps, info = self._baseline_grid_astar(
                start3, goal3, radius=0.45, avoid_direct=False)
            if wps is not None:
                raw_count = len(wps)
                wps = self._compress_baseline_safe_path(wps, radius=0.45)
                corr = Corridor(wps, radius=0.45, label=label)
                stats = self._corridor_footprint_stats(corr)
                corr.baseline_planner = "risk_grid_astar_safe_direct"
                corr.planner_source = "risk_grid_astar_safe_direct"
                corr.baseline_grid_resolution = float(info.get(
                    "grid_resolution", self.baseline_grid_resolution))
                corr.baseline_astar_expanded = int(info.get("expanded", 0))
                rospy.logwarn(
                    "[wc][baseline] direct line unsafe (%s max_fp=%.3f); "
                    "using risk_grid_astar_safe_direct points=%d->%d max_fp=%.3f",
                    str(direct_stats.get("forbidden_reason", "risk")),
                    float(direct_stats.get("max_footprint_phi", 0.0)),
                    int(raw_count), len(wps),
                    float(stats.get("max_footprint_phi", 0.0)))
            else:
                rospy.logwarn(
                    "[wc][baseline] direct line unsafe and A* repair failed: %s",
                    str(info.get("reason", "unknown")))
                corr.baseline_planner = "direct_connection"
                corr.planner_source = "direct_connection"
        else:
            corr.baseline_planner = "direct_connection"
            corr.planner_source = "direct_connection"
        corr.base_cost = 0.0
        corr.cost = 0.0
        corr.corridor_id = corr.label
        corr.source = "baseline_direct"
        corr.refined_waypoints = np.asarray(corr.waypoints, float)
        corr.path_length = self._path_length(corr.waypoints)
        corr.mean_phi_on_path = float(stats.get("mean_center_phi", 0.0))
        corr.max_phi_on_path = float(stats.get("max_center_phi", 0.0))
        corr.mean_footprint_phi_on_path = float(
            stats.get("mean_footprint_phi", 0.0))
        corr.max_footprint_phi_on_path = float(
            stats.get("max_footprint_phi", 0.0))
        corr.forbidden_hits = int(bool(stats.get("forbidden", False)))
        corr.footprint_checked = True
        corr.baseline_inputs_complete = True
        corr.baseline_uses_stsm = 0
        corr.morse_induced = False
        return corr

    def _compress_baseline_safe_path(self, points, radius=0.45):
        pts = np.asarray(points, float)
        if pts.ndim != 2 or len(pts) <= 2:
            return pts
        risk_limit = min(
            float(self.footprint_gate.rho_stop) - 0.05,
            float(self.safe_fallback_max_footprint_phi))

        def segment_safe(a, b):
            probe = Corridor(np.asarray([a, b], float), radius=radius,
                             label="baseline_segment_probe")
            stats = self._corridor_footprint_stats(
                probe, samples_per_segment=16)
            return (
                not bool(stats.get("forbidden", False)) and
                float(stats.get("max_footprint_phi", 0.0)) <= risk_limit)

        out = [pts[0]]
        idx = 0
        while idx < len(pts) - 1:
            nxt = idx + 1
            for cand in range(len(pts) - 1, idx, -1):
                if segment_safe(pts[idx], pts[cand]):
                    nxt = cand
                    break
            out.append(pts[nxt])
            idx = nxt
        return np.asarray(out, float)

    def _corridor_lateral_deviation(self, corridor):
        pts = np.asarray(getattr(corridor, "waypoints", []), float)
        if pts.ndim != 2 or len(pts) < 3:
            return 0.0
        start = pts[0, :2]
        goal = pts[-1, :2]
        axis = goal - start
        length = float(np.linalg.norm(axis))
        if length <= 1e-9:
            return 0.0
        normal = np.array([-axis[1], axis[0]], float) / length
        offsets = [abs(float(np.dot(p[:2] - start, normal)))
                   for p in pts[1:-1]]
        return float(max(offsets) if offsets else 0.0)

    def _is_direct_connection_corridor(self, corridor, min_lateral=None):
        if min_lateral is None:
            min_lateral = self.baseline_min_lateral_deviation
        return self._corridor_lateral_deviation(corridor) < float(min_lateral)

    def _baseline_grid_axes(self):
        res = max(float(self.baseline_grid_resolution), 0.03)
        (xmin, xmax), (ymin, ymax) = self.bounds
        xs = np.arange(float(xmin), float(xmax) + 0.5 * res, res)
        ys = np.arange(float(ymin), float(ymax) + 0.5 * res, res)
        return xs, ys, res

    def _baseline_world_to_cell(self, point, xs, ys):
        point = np.asarray(point, float)[:2]
        ix = int(np.argmin(np.abs(xs - point[0])))
        iy = int(np.argmin(np.abs(ys - point[1])))
        return ix, iy

    def _baseline_cell_to_pose(self, cell, xs, ys):
        ix, iy = cell
        return np.array([float(xs[ix]), float(ys[iy]), 0.0], float)

    def _baseline_pose_safety(self, pose, radius):
        pose = np.asarray(pose, float)
        (xmin, xmax), (ymin, ymax) = self.bounds
        if not (xmin <= pose[0] <= xmax and ymin <= pose[1] <= ymax):
            return False, 1e6, 1e6, "out_of_bounds"
        center_phi = float(self.field.phi_s(
            np.array([pose[0], pose[1], 0.0], float)))
        if center_phi > float(self.safe_fallback_max_center_phi):
            return False, center_phi, center_phi, "center_risk"
        yaw = float(self.state[2]) if self.state is not None else 0.0
        summary = pose_interest_risk(
            self.field, np.array([pose[0], pose[1], yaw], float),
            local_points=self.wc_local_points,
            labels=self.wc_ip_labels)
        hit, _label, _anchor, reason = forbidden_anchor_hit(
            self.field, summary.get("labels", []),
            summary.get("points", []))
        footprint_phi = float(summary.get("phi_max", center_phi))
        if hit:
            return False, center_phi, footprint_phi, reason or "forbidden_anchor"
        if footprint_phi > float(self.safe_fallback_max_footprint_phi):
            return False, center_phi, footprint_phi, "footprint_risk"
        for theta in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
            probe = np.array([
                pose[0] + float(radius) * np.cos(theta),
                pose[1] + float(radius) * np.sin(theta),
                0.0,
            ], float)
            probe_phi = float(self.field.phi_s(probe))
            if probe_phi > float(self.safe_fallback_max_center_phi):
                return False, center_phi, max(footprint_phi, probe_phi), "radius_risk"
        return True, center_phi, footprint_phi, ""

    def _baseline_nearest_safe_cell(self, seed, xs, ys, radius, safe_cache):
        sx, sy = seed
        max_ring = max(len(xs), len(ys))
        for ring in range(max_ring):
            cells = []
            for dx in range(-ring, ring + 1):
                for dy in range(-ring, ring + 1):
                    if max(abs(dx), abs(dy)) != ring:
                        continue
                    ix, iy = sx + dx, sy + dy
                    if 0 <= ix < len(xs) and 0 <= iy < len(ys):
                        cells.append((ix, iy))
            cells.sort(key=lambda c: (c[0] - sx) ** 2 + (c[1] - sy) ** 2)
            for cell in cells:
                safe, _center, _footprint, _reason = safe_cache(cell)
                if safe:
                    return cell
        return None

    def _baseline_grid_astar(self, start3, goal3, radius=0.45,
                             avoid_direct=False):
        xs, ys, res = self._baseline_grid_axes()
        start_cell = self._baseline_world_to_cell(start3, xs, ys)
        goal_cell = self._baseline_world_to_cell(goal3, xs, ys)
        safety_cache = {}

        def safe_cache(cell):
            if cell not in safety_cache:
                safety_cache[cell] = self._baseline_pose_safety(
                    self._baseline_cell_to_pose(cell, xs, ys), radius)
            return safety_cache[cell]

        start_cell = self._baseline_nearest_safe_cell(
            start_cell, xs, ys, radius, safe_cache)
        goal_cell = self._baseline_nearest_safe_cell(
            goal_cell, xs, ys, radius, safe_cache)
        if start_cell is None or goal_cell is None:
            return None, {"reason": "safe_endpoint_not_found"}

        start2 = np.asarray(start3, float)[:2]
        goal2 = np.asarray(goal3, float)[:2]
        axis = goal2 - start2
        axis_len = float(np.linalg.norm(axis))
        normal = np.array([0.0, 0.0], float)
        if axis_len > 1e-9:
            normal = np.array([-axis[1], axis[0]], float) / axis_len

        def heuristic(cell):
            p = self._baseline_cell_to_pose(cell, xs, ys)[:2]
            q = self._baseline_cell_to_pose(goal_cell, xs, ys)[:2]
            return float(np.linalg.norm(p - q))

        def step_cost(cell, nxt, prev):
            p = self._baseline_cell_to_pose(cell, xs, ys)[:2]
            q = self._baseline_cell_to_pose(nxt, xs, ys)[:2]
            dist = float(np.linalg.norm(q - p))
            _safe, center_phi, footprint_phi, _reason = safe_cache(nxt)
            risk = max(0.0, center_phi) + 0.35 * max(0.0, footprint_phi)
            cost = dist * (1.0 + self.baseline_risk_weight * risk)
            if prev is not None:
                r = self._baseline_cell_to_pose(prev, xs, ys)[:2]
                a = p - r
                b = q - p
                na = float(np.linalg.norm(a))
                nb = float(np.linalg.norm(b))
                if na > 1e-9 and nb > 1e-9:
                    cosang = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
                    cost += self.baseline_turn_weight * (1.0 - cosang)
            if avoid_direct and axis_len > 1e-9:
                rel = q - start2
                progress = float(np.dot(rel, axis) / (axis_len * axis_len))
                lateral = abs(float(np.dot(rel, normal)))
                if 0.12 < progress < 0.88 and lateral < self.baseline_min_lateral_deviation:
                    gap = self.baseline_min_lateral_deviation - lateral
                    cost += self.baseline_direct_retry_weight * gap
            return float(cost)

        moves = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),           (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]
        open_heap = []
        counter = 0
        heapq.heappush(open_heap, (heuristic(start_cell), counter, start_cell, None))
        best = {start_cell: 0.0}
        parent = {}
        prev_cell = {start_cell: None}
        closed = set()

        while open_heap:
            _f, _count, cell, prev = heapq.heappop(open_heap)
            if cell in closed:
                continue
            closed.add(cell)
            if cell == goal_cell:
                break
            for dx, dy in moves:
                nxt = (cell[0] + dx, cell[1] + dy)
                if not (0 <= nxt[0] < len(xs) and 0 <= nxt[1] < len(ys)):
                    continue
                safe, _center, _footprint, _reason = safe_cache(nxt)
                if not safe:
                    continue
                cand = best[cell] + step_cost(cell, nxt, prev)
                if cand + 1e-9 < best.get(nxt, 1e18):
                    best[nxt] = cand
                    parent[nxt] = cell
                    prev_cell[nxt] = cell
                    counter += 1
                    heapq.heappush(
                        open_heap,
                        (cand + heuristic(nxt), counter, nxt, cell))

        if goal_cell not in best:
            return None, {"reason": "astar_no_path"}
        cells = [goal_cell]
        while cells[-1] != start_cell:
            cells.append(parent[cells[-1]])
        cells.reverse()
        points = [np.array([start2[0], start2[1], 0.0], float)]
        for cell in cells[1:-1]:
            points.append(self._baseline_cell_to_pose(cell, xs, ys))
        points.append(np.array([goal2[0], goal2[1], 0.0], float))
        return np.asarray(points, float), {
            "reason": "",
            "expanded": len(closed),
            "grid_resolution": float(res),
            "cost": float(best[goal_cell]),
        }

    def _path_length(self, points):
        pts = np.asarray(points, float)
        if pts.ndim != 2 or len(pts) < 2:
            return 0.0
        return float(sum(np.linalg.norm(b[:2] - a[:2])
                         for a, b in zip(pts[:-1], pts[1:])))

    def _traditional_baseline_corridor(self, start):
        start3 = np.array([start[0], start[1], 0.0], float)
        goal3 = np.array([self.goal[0], self.goal[1], 0.0], float)
        wps, info = self._baseline_grid_astar(
            start3, goal3, radius=0.45, avoid_direct=False)
        if wps is not None:
            probe = Corridor(wps, radius=0.45, label="baseline_traditional")
            if self._is_direct_connection_corridor(probe):
                wps, info = self._baseline_grid_astar(
                    start3, goal3, radius=0.45, avoid_direct=True)
        if wps is None:
            raise RuntimeError(
                "traditional baseline planner failed: %s" %
                str(info.get("reason", "unknown")))
        corr = Corridor(wps, radius=0.45, label="baseline_traditional")
        if self._is_direct_connection_corridor(corr):
            raise RuntimeError(
                "traditional baseline planner returned direct connection")
        stats = self._corridor_footprint_stats(corr)
        path_length = self._path_length(wps)
        corr.base_cost = float(info.get("cost", path_length))
        corr.cost = float(corr.base_cost)
        corr.corridor_id = "baseline_traditional"
        corr.source = "baseline_traditional_planner"
        corr.original_label = "risk_grid_astar"
        corr.refined_waypoints = np.asarray(wps, float)
        corr.path_length = float(path_length)
        corr.mean_phi_on_path = float(stats["mean_center_phi"])
        corr.max_phi_on_path = float(stats["max_center_phi"])
        corr.mean_footprint_phi_on_path = float(stats["mean_footprint_phi"])
        corr.max_footprint_phi_on_path = float(stats["max_footprint_phi"])
        corr.forbidden_hits = int(bool(stats["forbidden"]))
        corr.collision_checked = True
        corr.footprint_checked = True
        corr.robot_radius = 0.45
        corr.baseline_inputs_complete = True
        corr.baseline_planner = "risk_grid_astar"
        corr.baseline_uses_stsm = 0
        corr.baseline_grid_resolution = float(info.get(
            "grid_resolution", self.baseline_grid_resolution))
        corr.baseline_astar_expanded = int(info.get("expanded", 0))
        corr.recovery_rejected = 0
        corr.morse_induced = False
        corr.topology_nodes = []
        corr.node_sequence = []
        corr.node_type_sequence = []
        corr.risk_cost = float(path_length * stats["mean_footprint_phi"] +
                               2.0 * stats["max_footprint_phi"])
        corr.topology_cost = 0.0
        corr.distance_cost = float(path_length)
        corr.length_cost = float(path_length)
        corr.smooth_cost = 0.0
        corr.curvature_cost = 0.0
        rospy.logwarn(
            "[wc][baseline] selected traditional risk_grid_astar points=%d "
            "length=%.3f max_lat=%.3f max_fp=%.3f max_center=%.3f",
            len(wps), corr.path_length, self._corridor_lateral_deviation(corr),
            corr.max_footprint_phi_on_path, corr.max_phi_on_path)
        return corr

    def _corridor_footprint_stats(self, corridor, samples_per_segment=24):
        footprint_values = []
        center_values = []
        forbidden = False
        forbidden_reason = ""
        for p in self._polyline_samples(
                getattr(corridor, "waypoints", []), samples_per_segment):
            pose = np.array([p[0], p[1], self.state[2]], float)
            center_values.append(float(self.field.phi_s(
                np.array([p[0], p[1], 0.0], float))))
            summary = pose_interest_risk(
                self.field, pose,
                local_points=self.wc_local_points,
                labels=self.wc_ip_labels)
            hit, _label, _anchor, reason = forbidden_anchor_hit(
                self.field, summary.get("labels", []),
                summary.get("points", []))
            forbidden = forbidden or bool(hit)
            if hit and not forbidden_reason:
                forbidden_reason = reason
            footprint_values.append(float(summary.get("phi_max", 0.0)))
        if not footprint_values:
            footprint_values = [0.0]
        if not center_values:
            center_values = [0.0]
        return {
            "mean_footprint_phi": float(np.mean(footprint_values)),
            "max_footprint_phi": float(np.max(footprint_values)),
            "mean_center_phi": float(np.mean(center_values)),
            "max_center_phi": float(np.max(center_values)),
            "forbidden": bool(forbidden),
            "forbidden_reason": forbidden_reason,
        }

    def _wheelchair_safe_fallback_corridor(self, start3, goal3, label,
                                           corridor_id, radius=0.45,
                                           use_adp=False):
        feature_context = {
            "yaw": self.state[2],
            "u": self.u_prev,
            "radius": radius,
            "adp_samples": 9,
            "gate_info": {
                "state": "NORMAL",
                "rho_warn": self.gate.rho_warn,
                "stop": False,
            },
        }
        candidates = self.manifold.enumerate_corridors(
            start3, goal3, self.bounds, radius=radius,
            critic=self.adp_critic if use_adp else None,
            feature_builder=self.adp_features,
            lambda_adp=self.lambda_adp_corridor if use_adp else 0.0,
            feature_context=feature_context)
        start2 = np.asarray(start3, float)[:2]
        goal2 = np.asarray(goal3, float)[:2]
        mid = 0.5 * (start2 + goal2)
        direction = goal2 - start2
        length = float(np.linalg.norm(direction))
        if length > 1e-9:
            normal = np.array([-direction[1], direction[0]], float) / length
            (xmin, xmax), (ymin, ymax) = self.bounds
            for offset in (0.9, -0.9, 1.2, -1.2, 1.5, -1.5):
                via = mid + float(offset) * normal
                via[0] = float(np.clip(via[0], xmin + 0.05, xmax - 0.05))
                via[1] = float(np.clip(via[1], ymin + 0.05, ymax - 0.05))
                wps = np.array([
                    [start2[0], start2[1], 0.0],
                    [via[0], via[1], 0.0],
                    [goal2[0], goal2[1], 0.0],
                ])
                name = "extra_left_{:.1f}".format(abs(offset))
                if offset < 0.0:
                    name = "extra_right_{:.1f}".format(abs(offset))
                extra = Corridor(
                    wps, radius,
                    label=name,
                    cost=float(self.manifold._corridor_cost(wps)))
                extra.base_cost = float(extra.cost)
                candidates.append(extra)
        if not candidates:
            return None
        scored = []
        for idx, corr in enumerate(candidates):
            stats = self._corridor_footprint_stats(corr)
            pts = np.asarray(corr.waypoints, float)
            path_length = 0.0
            for a, b in zip(pts[:-1], pts[1:]):
                path_length += float(np.linalg.norm(b[:2] - a[:2]))
            allowed = (
                not stats["forbidden"] and
                stats["max_footprint_phi"] <= self.safe_fallback_max_footprint_phi and
                stats["mean_footprint_phi"] <= self.safe_fallback_max_mean_footprint_phi and
                stats["max_center_phi"] <= self.safe_fallback_max_center_phi)
            scored.append((allowed, stats, path_length, idx, corr))
        allowed = [item for item in scored if item[0]]
        non_forbidden = [item for item in scored if not item[1]["forbidden"]]
        pool = allowed if allowed else (non_forbidden if non_forbidden else scored)
        if label == "baseline_traditional":
            detour_pool = [
                item for item in pool
                if (str(getattr(item[4], "label", "")) != "direct" and
                    not self._is_direct_connection_corridor(item[4]))
            ]
            if detour_pool:
                pool = detour_pool
            else:
                raise RuntimeError(
                    "traditional baseline planner found no collision-checked detour")
        selected_allowed, stats, path_length, _idx, selected = sorted(
            pool,
            key=lambda item: (
                float(item[1]["mean_footprint_phi"]),
                float(item[1]["max_center_phi"]),
                float(item[1]["max_footprint_phi"]),
                0.10 * float(item[2]),
                float(item[2]),
                float(getattr(item[4], "cost", 0.0))))[0]
        selected.original_label = str(getattr(selected, "label", ""))
        selected.label = label
        selected.corridor_id = corridor_id
        selected.source = (
            "baseline_mpc_safe" if label == "baseline_mpc_safe"
            else "baseline_traditional" if label == "baseline_traditional"
            else "wheelchair_safe_fallback")
        selected.radius = float(radius)
        selected.path_length = float(path_length)
        selected.mean_phi_on_path = float(stats["mean_center_phi"])
        selected.max_phi_on_path = float(stats["max_center_phi"])
        selected.mean_footprint_phi_on_path = float(stats["mean_footprint_phi"])
        selected.max_footprint_phi_on_path = float(stats["max_footprint_phi"])
        selected.forbidden_hits = int(bool(stats["forbidden"]))
        selected.collision_checked = True
        selected.footprint_checked = True
        selected.robot_radius = float(radius)
        selected.baseline_inputs_complete = True
        selected.baseline_planner = "traditional_safe_corridor"
        selected.recovery_rejected = 0
        selected.morse_induced = False
        selected.topology_nodes = []
        selected.node_sequence = []
        selected.node_type_sequence = []
        selected.risk_cost = float(
            selected.path_length * selected.mean_footprint_phi_on_path +
            2.0 * selected.max_footprint_phi_on_path)
        selected.topology_cost = 0.0
        selected.distance_cost = float(path_length)
        selected.smooth_cost = 0.0
        selected.curvature_cost = 0.0
        rospy.logwarn(
            "[wc][fallback] selected %s from %s max_fp=%.3f mean_fp=%.3f max_center=%.3f allowed=%d",
            selected.corridor_id, selected.original_label,
            selected.max_footprint_phi_on_path,
            selected.mean_footprint_phi_on_path,
            selected.max_phi_on_path,
            1 if selected_allowed else 0)
        return selected

    def _log_topology_empty_debug(self, stage):
        dbg = getattr(self.manifold, "last_topology_debug", {}) or {}
        if not dbg:
            rospy.logwarn("[wc][topology] no corridors at %s: no debug payload", stage)
            return
        rospy.logwarn(
            "[wc][topology] no corridors at %s: candidates=%s nodes=%s edges=%s "
            "saddles(raw/safe/usable/used)=%s/%s/%s/%s minima(raw/safe/usable/used)=%s/%s/%s/%s "
            "rejects grad=%s degenerate=%s forbidden=%s clearance=%s unsafe=%s "
            "candidate_forbidden=%s clearance=%s edge_clearance=%s edge_forbidden=%s astar_fail=%s "
            "disconnect=%s recovery=%s",
            stage,
            dbg.get("num_candidate_corridors", 0),
            dbg.get("num_topology_nodes", 0),
            dbg.get("num_topology_edges", 0),
            dbg.get("num_raw_saddles", 0),
            dbg.get("num_safe_saddles", 0),
            dbg.get("num_usable_saddles", 0),
            dbg.get("num_used_saddles", 0),
            dbg.get("num_raw_minima", 0),
            dbg.get("num_safe_minima", 0),
            dbg.get("num_usable_minima", 0),
            dbg.get("num_used_minima", 0),
            dbg.get("reject_by_gradient_count", 0),
            dbg.get("reject_by_degenerate_count", 0),
            dbg.get("reject_by_forbidden_count", 0),
            dbg.get("reject_by_clearance_count", 0),
            dbg.get("reject_by_unsafe_count", 0),
            dbg.get("candidate_forbidden_reject_count", 0),
            dbg.get("clearance_reject_count", 0),
            dbg.get("edge_clearance_reject_count", 0),
            dbg.get("edge_forbidden_reject_count", 0),
            dbg.get("edge_astar_fail_count", 0),
            dbg.get("topology_disconnect_reason", ""),
            dbg.get("topology_recovery_reject_reason", ""))

    def _plan_corridor(self):
        start = self.state[:2]
        if self.baseline:
            if self.baseline_type == "mpc_safe":
                corr = self._wheelchair_safe_fallback_corridor(
                    np.array([start[0], start[1], 0.0]),
                    np.array([self.goal[0], self.goal[1], 0.0]),
                    label="baseline_mpc_safe",
                    corridor_id="baseline_mpc_safe",
                    radius=0.45,
                    use_adp=False)
            elif self.baseline_type == "direct":
                corr = self._direct_baseline_corridor(start)
            elif self.baseline_type == "traditional":
                corr = self._traditional_baseline_corridor(start)
            else:
                raise ValueError("unknown wheelchair baseline_type={}".format(
                    self.baseline_type))
            self.selected_corridor = corr
            self.selected_corridor_pub.publish(String(corr.corridor_id))
            self._publish_topology_info(False, False, topology_enabled=False)
            return corr
        start3 = np.array([start[0], start[1], 0.0])
        goal3 = np.array([self.goal[0], self.goal[1], 0.0])
        feature_context = {
            "yaw": self.state[2],
            "u": self.u_prev,
            "radius": 0.4,
            "adp_samples": 9,
            "gate_info": {
                "state": "NORMAL",
                "rho_warn": self.gate.rho_warn,
                "stop": False,
            },
        }
        interest_config = {
            "enabled": bool(self.interest_enabled),
            "local_points": self.wc_local_points,
            "labels": self.wc_ip_labels,
            "yaw": self.state[2],
            "rho": self.footprint_gate.rho_stop,
        }
        corrs = []
        used_topology = False
        fallback_used = False
        plan_t0 = time.time()
        if self.topology_enabled:
            try:
                wc_executable_curvature = min(
                    float(self.topology_max_corridor_curvature), 8.0)
                rospy.loginfo(
                    "[wc][topology] enumerate start k=%d route_max_paths=%d route_max_routes=%d max_curv=%.3f",
                    int(self.topology_k_paths),
                    int(self.topology_route_max_paths),
                    int(self.topology_route_max_routes),
                    float(wc_executable_curvature))
                corrs = self.manifold.enumerate_topological_corridors(
                    start3, goal3, self.bounds, radius=0.4,
                    grid_resolution=self.topology_grid_resolution,
                    merge_radius=self.topology_merge_radius,
                    min_clearance=self.topology_min_clearance,
                    hard_clearance=self.topology_hard_clearance,
                    neighbor_k=self.topology_neighbor_k,
                    k=self.topology_k_paths,
                    max_graph_nodes=self.topology_max_graph_nodes,
                    critic=self.adp_critic if self.adp_enabled else None,
                    feature_builder=self.adp_features,
                    lambda_adp=(
                        self.lambda_adp_corridor
                        if self.adp_enabled else 0.0),
                    feature_context=feature_context,
                    interest_config=interest_config,
                    topology_profile=self.topology_profile,
                    topology_params={
                        "saddle_tie_ratio": self.topology_saddle_tie_ratio,
                        "morse_priority_ratio": self.topology_morse_priority_ratio,
                        "morse_saddle_priority_ratio": self.topology_morse_saddle_priority_ratio,
                        "morse_mix_priority_ratio": self.topology_morse_mix_priority_ratio,
                        "morse_minima_priority_ratio": self.topology_morse_minima_priority_ratio,
                        "morse_core_required": self.topology_morse_core_required,
                        "morse_decision_mode": self.topology_morse_decision_mode,
                        "morse_w_goal": self.topology_morse_w_goal,
                        "morse_w_social": self.topology_morse_w_social,
                        "morse_w_barrier": self.topology_morse_w_barrier,
                        "morse_grad_eps": self.topology_morse_grad_eps,
                        "safety_regions": self.topology_safety_regions,
                        "allow_semantic_with_morse": self.topology_allow_semantic_with_morse,
                        "allow_semantic_topology_recovery": self.topology_allow_semantic_topology_recovery,
                        "allow_ring_with_morse": self.topology_allow_ring_with_morse,
                        "allow_graph_fallback_with_morse": self.topology_allow_graph_fallback_with_morse,
                        "lambda_execution": self.topology_lambda_execution,
                        "lambda_tracking": self.topology_lambda_tracking,
                        "lambda_saddle_value": self.topology_lambda_saddle_value,
                        "max_corridor_turn": self.topology_max_corridor_turn,
                        "max_corridor_curvature": wc_executable_curvature,
                        "min_segment_length": self.topology_min_segment_length,
                        "corridor_dedupe_distance": self.topology_corridor_dedupe_distance,
                        "candidate_pool_min": self.topology_candidate_pool_min,
                        "route_max_paths": self.topology_route_max_paths,
                        "route_max_routes": self.topology_route_max_routes,
                        "require_risk_improvement": self.topology_require_risk_improvement,
                        "candidate_max_risk": self.topology_candidate_max_risk,
                        "corridor_score_weights": self.topology_corridor_score_weights,
                        "task_mode": self.task_mode,
                        "task_config": self.task_config,
                        "task_weight": self.task_weight,
                        "manifold_constraint_mode": self.manifold_constraint_mode,
                        "task_minima_points": [
                            {
                                "type": "waiting",
                                "position": [0.55, 1.20, 0.0],
                            },
                            {
                                "type": "parking",
                                "position": [-0.78, 0.72, 0.0],
                            },
                        ],
                        "dynamics_profile": {
                            "type": "wheelchair",
                            "v_max": self.mpc_base_v_max,
                            "w_max": self.mpc.w_max,
                            "a_max": self.mpc.a_max,
                            "nominal_speed": min(
                                0.35, max(0.10, 0.6 * self.mpc_base_v_max)),
                            "max_tracking_turn": self.topology_max_corridor_turn,
                            "max_curvature": wc_executable_curvature,
                            "min_progress": self.min_progress_per_solve,
                        },
                    })
                rospy.loginfo(
                    "[wc][topology] enumerate done candidates=%d elapsed=%.3fs",
                    int(len(corrs)), time.time() - plan_t0)
                used_topology = len(corrs) > 0
                if not corrs:
                    self._log_topology_empty_debug("enumerate")
            except Exception as exc:
                rospy.logwarn("[wc][topology] failed, fallback=%s: %s",
                              self.topology_fallback_enabled, exc)
        if not corrs:
            rospy.logwarn(
                "[wc][topology] no Morse topology candidate survived; "
                "independent morse_recovery fallback is disabled")
        if not corrs:
            self.selected_corridor_pub.publish(String("planning_failed"))
            self._publish_topology_info(False, False)
            rospy.logerr(
                "[wc][topology] planning_failed: no hard-filtered Morse candidate corridors in %s mode",
                self.experiment_mode)
            self._write_failed_topology_diagnostics(
                "topology planner returned no STSM corridors")
            raise RuntimeError("topology planner returned no STSM corridors")
        if corrs:
            corrs = self._prepare_executable_corridors(corrs)
        if not corrs:
            self.selected_corridor_pub.publish(String("planning_failed"))
            self._publish_topology_info(False, False)
            rospy.logerr(
                "[wc][refine] planning_failed: no executable refined STSM corridors in %s mode",
                self.experiment_mode)
            self._write_failed_topology_diagnostics(
                "topology refinement returned no executable STSM corridors")
            raise RuntimeError("topology refinement returned no executable STSM corridors")
        corrs = self._rescore_executable_corridors(corrs)
        self.runtime_topology_candidate_pool = list(corrs)
        self.runtime_rejected_topology_corridor_ids = set()
        self.runtime_topology_candidate_switch_count = 0
        self._publish_topology_info(used_topology, fallback_used)
        for c in corrs:
            rospy.loginfo(
                "[wc][corridor] %s base=%.3f adp=%.3f total=%.3f mean_phi=%.3f max_phi=%.3f clearance=%.3f rank_base=%d rank_total=%d nodes=%s",
                c.label, c.base_cost, c.adp_cost, c.cost,
                getattr(c, "mean_phi_on_path", 0.0),
                getattr(c, "max_phi_on_path", 0.0),
                getattr(c, "min_clearance", 0.0),
                c.rank_base, c.rank_total,
                ",".join(getattr(c, "topology_nodes", [])))
        selected = self._select_wheelchair_corridor(corrs)
        if selected is None:
            selected = self._runtime_replan_fallback_corridor(
                None, "empty_selected_corridor",
                RuntimeError("topology candidate selection returned None"))
        if selected is None:
            raise RuntimeError("topology candidate selection returned no corridor")
        self.selected_corridor = selected
        self.execution_corridor = selected
        self._sync_selected_corridor_geometry(self.selected_corridor)
        self._sync_runtime_topology_debug(corrs, selected)
        self.last_valid_topology_debug = copy.deepcopy(
            getattr(self.manifold, "last_topology_debug", {}) or {})
        self._publish_topology_info(used_topology, fallback_used)
        execution_id = self._corridor_id(selected, "wheelchair_selected")
        self.selected_corridor_pub.publish(String(execution_id))
        rospy.loginfo("[wc] selected corridor: %s label=%s (cost %.3f, source=%s)",
                      execution_id, self._corridor_label(selected, execution_id),
                      float(getattr(selected, "cost", 0.0)),
                      "topology" if used_topology else "fallback")
        rospy.loginfo("[wc] corridor planning time: %.3fs",
                      time.time() - plan_t0)
        self.last_corridor_plan_duration_s = float(time.time() - plan_t0)
        return selected

    def _publish_topology_info(self, used_topology, fallback_used,
                               topology_enabled=None):
        dbg = getattr(self.manifold, "last_topology_debug", {}) or {}
        enabled = self.topology_enabled if topology_enabled is None else bool(topology_enabled)
        self.topology_info_pub.publish(Float64MultiArray(data=[
            1.0 if enabled else 0.0,
            1.0 if used_topology else 0.0,
            1.0 if fallback_used else 0.0,
            float(dbg.get("num_critical_minima", 0)),
            float(dbg.get("num_critical_saddles", 0)),
            float(dbg.get("num_critical_maxima", 0)),
            float(dbg.get("num_raw_minima", 0)),
            float(dbg.get("num_raw_saddles", 0)),
            float(dbg.get("num_raw_maxima", 0)),
            float(dbg.get("num_safe_minima", 0)),
            float(dbg.get("num_safe_saddles", 0)),
            float(dbg.get("num_safe_maxima", 0)),
            float(dbg.get("num_filtered_minima", 0)),
            float(dbg.get("num_filtered_saddles", 0)),
            float(dbg.get("num_filtered_maxima", 0)),
            float(dbg.get("num_usable_minima", 0)),
            float(dbg.get("num_usable_saddles", 0)),
            float(dbg.get("num_used_minima", 0)),
            float(dbg.get("num_used_saddles", 0)),
            float(dbg.get("num_forced_critical_corridors", 0)),
            float(dbg.get("num_morse_minima_corridors", 0)),
            float(dbg.get("num_morse_saddle_corridors", 0)),
            float(dbg.get("num_morse_mix_corridors", 0)),
            float(dbg.get("num_graph_direct_corridors", 0)),
            float(dbg.get("num_graph_semantic_corridors", 0)),
            float(dbg.get("reject_by_gradient_count", 0)),
            float(dbg.get("reject_by_degenerate_count", 0)),
            float(dbg.get("reject_by_forbidden_count", 0)),
            float(dbg.get("reject_by_clearance_count", 0)),
            float(dbg.get("reject_by_unsafe_count", 0)),
            float(dbg.get("num_topology_nodes", 0)),
            float(dbg.get("num_topology_edges", 0)),
            float(dbg.get("num_candidate_corridors", 0)),
            float(dbg.get("topology_grid_resolution",
                          _effective(self.topology_grid_resolution,
                                     self.topology_profile_defaults["grid_resolution"]))),
            float(self.manifold.rho),
            float(dbg.get("num_forbidden_cells", 0)),
            float(dbg.get("selected_corridor_forbidden_hits", 0)),
            float(dbg.get("candidate_forbidden_reject_count", 0)),
            float(dbg.get("clearance_reject_count", 0)),
            float(dbg.get("edge_clearance_reject_count", 0)),
            float(dbg.get("edge_forbidden_reject_count", 0)),
            float(dbg.get("edge_astar_fail_count", 0)),
            float(dbg.get("neighbor_pair_attempt_count", 0)),
            float(dbg.get("hard_clearance", _effective(
                self.topology_hard_clearance,
                self.topology_profile_defaults["hard_clearance"]))),
            float(dbg.get("clearance_target", _effective(
                self.topology_min_clearance,
                self.topology_profile_defaults["min_clearance"]))),
            float(dbg.get("neighbor_k", _effective(
                self.topology_neighbor_k,
                self.topology_profile_defaults["neighbor_k"]))),
            float(dbg.get("selected_saddle_value_bonus", 0.0)),
            float(dbg.get("selected_candidate_total_score", 0.0)),
            float(dbg.get("selected_tracking_cost", 0.0)),
            float(dbg.get("selected_max_curvature", 0.0)),
            float(dbg.get("selected_curvature_violation", 0.0)),
            float(dbg.get("selected_turn_violation", 0.0)),
            float(dbg.get("selected_expected_progress", 0.0)),
            float(dbg.get("selected_refinement_used", 0.0)),
            float(dbg.get("selected_refined_path_length", 0.0)),
            float(dbg.get("selected_topology_diversity", 0.0)),
            float(dbg.get("selected_raw_waypoints_count", 0.0)),
            float(dbg.get("selected_refined_waypoints_count", 0.0)),
            float(dbg.get("mpc_used", 0.0)),
            1.0 if str(dbg.get("mpc_reference_source", "")) == "refined_waypoints" else 0.0,
            float(dbg.get("candidate_min_clearance", 0.0)),
            float(dbg.get("candidate_max_risk", 0.0)),
            1.0 if bool(dbg.get("candidate_manifold_feasible", False)) else 0.0,
            1.0 if bool(dbg.get(
                "candidate_manifold_valid",
                dbg.get("candidate_manifold_feasible", False))) else 0.0,
            1.0 if bool(dbg.get("candidate_tube_valid", False)) else 0.0,
            float(dbg.get("num_manifold_filtered_candidates", 0.0)),
            float(dbg.get("filtered_infeasible_candidates", 0.0)),
            float(dbg.get("planning_clearance_margin", 0.0)),
        ]))

    def _direct_goal_control(self, dist):
        desired = np.arctan2(self.goal[1] - self.state[1],
                             self.goal[0] - self.state[0])
        herr = np.arctan2(np.sin(desired - self.state[2]),
                          np.cos(desired - self.state[2]))
        w = float(np.clip(1.8 * herr, -0.9, 0.9))
        align = max(0.0, np.cos(herr))
        v = min(0.45, 0.7 * float(dist)) * align
        if abs(herr) > 0.9:
            v = min(v, 0.06)
        return float(v), float(w)

    def _goal_heading_error(self):
        desired = np.arctan2(self.goal[1] - self.state[1],
                             self.goal[0] - self.state[0])
        return float(np.arctan2(np.sin(desired - self.state[2]),
                                np.cos(desired - self.state[2])))

    def _in_final_approach(self, dist):
        return (
            (not self.baseline) and
            float(dist) < float(self.final_approach_entry_radius))

    def _apply_final_approach_profile(self, dist):
        active = self._in_final_approach(dist)
        if active:
            self.mpc.near_goal_radius = max(
                float(self.near_goal_radius),
                float(self.final_approach_entry_radius))
            self.mpc.final_approach_radius = float(
                self.final_approach_entry_radius)
            self.mpc.lam_track = self.mpc_base_lam_track * max(
                1.0, float(self.final_approach_goal_weight_scale))
            self.mpc.lam_goal_terminal = (
                self.mpc_base_lam_goal_terminal *
                max(1.0, float(self.final_approach_goal_weight_scale)))
            self.mpc.near_goal_goal_weight = (
                self.near_goal_goal_weight *
                max(1.0, float(self.final_approach_goal_weight_scale)))
            self.mpc.lam_social = self.mpc_base_lam_social * max(
                0.0, float(self.final_approach_social_scale))
            self.mpc.near_goal_social_scale = min(
                float(self.near_goal_social_scale),
                max(0.0, float(self.final_approach_social_scale)))
            return True
        self.mpc.near_goal_radius = self.near_goal_radius
        self.mpc.final_approach_radius = (
            0.0 if self.baseline else self.final_approach_entry_radius)
        self.mpc.lam_track = self.mpc_base_lam_track
        self.mpc.lam_goal_terminal = self.mpc_base_lam_goal_terminal
        self.mpc.near_goal_goal_weight = self.near_goal_goal_weight
        self.mpc.lam_social = self.mpc_base_lam_social
        self.mpc.near_goal_social_scale = self.near_goal_social_scale
        return False

    def _corridor_is_topological(self, corridor):
        return (
            corridor is not None and
            str(getattr(corridor, "label", "")).startswith(
                ("morse_", "graph_")))

    def _path_yaw(self, path, idx):
        pts = np.asarray(path, float)
        if len(pts) < 2:
            return float(self.state[2])
        i = min(max(int(idx), 0), len(pts) - 2)
        delta = pts[i + 1, :2] - pts[i, :2]
        if float(np.linalg.norm(delta)) <= 1e-9 and i > 0:
            delta = pts[i, :2] - pts[i - 1, :2]
        if float(np.linalg.norm(delta)) <= 1e-9:
            return float(self.state[2])
        return float(np.arctan2(delta[1], delta[0]))

    def _footprint_path_checker(self, path):
        pts = np.asarray(path, float)
        raw_count = int(len(pts))
        self.last_refined_footprint_checked_points = raw_count
        if raw_count > int(self.max_refined_footprint_check_points):
            keep = np.linspace(
                0, raw_count - 1,
                int(self.max_refined_footprint_check_points))
            indices = sorted(set(int(round(v)) for v in keep))
            if indices[0] != 0:
                indices.insert(0, 0)
            if indices[-1] != raw_count - 1:
                indices.append(raw_count - 1)
            pts = np.asarray([pts[i] for i in indices], float)
        phi_values = []
        max_phi = 0.0
        for idx, point in enumerate(pts):
            pose = np.array([point[0], point[1], self._path_yaw(pts, idx)], float)
            points_map = transform_points_2d(pose, self.wc_local_points)
            labels = list(self.wc_ip_labels)
            points = [points_map[label] for label in labels]
            summary = aggregate_point_risks(
                self.field, labels, points, [np.zeros(3) for _ in labels])
            hit, _label, _anchor, reason = forbidden_anchor_hit(
                self.field, labels, points)
            if hit:
                self.last_refined_footprint_checked_points = int(len(pts))
                return False, "refined_footprint_forbidden:" + str(reason)
            phi = float(summary.get("phi_max", 0.0))
            phi_values.append(phi)
            max_phi = max(max_phi, phi)
            if phi > float(self.footprint_gate.rho_stop) + 1e-9:
                self.last_refined_footprint_checked_points = int(len(pts))
                return False, "refined_footprint_risk"
        self.last_refined_footprint_max = float(max_phi)
        self.last_refined_footprint_mean = (
            float(np.mean(phi_values)) if phi_values else 0.0)
        self.last_refined_footprint_checked_points = int(len(pts))
        return True, ""

    def _project_points_to_corridor(self, path, corridor, margin=0.85):
        pts = np.asarray(path, float)
        if pts.size == 0:
            return pts.reshape((0, 3))
        pts = pts.copy()
        radius = float(getattr(corridor, "radius", 0.0)) * float(margin)
        if radius <= 0.0 or not hasattr(corridor, "project"):
            return pts
        for idx, point in enumerate(pts):
            try:
                projected, dist = corridor.project(point)
            except Exception:
                continue
            if float(dist) <= radius + 1e-9:
                continue
            dim = min(len(projected), pts.shape[1])
            pull = np.asarray(projected, float)[:dim] - pts[idx, :dim]
            pts[idx, :dim] += pull * (
                1.0 - radius / max(float(dist), 1e-9))
        return pts

    def _make_heading_progress_prefix(self, reference, corridor):
        """Build a short launch segment aligned with the current diff-drive pose."""
        if self.state is None or self.goal is None:
            return None
        ref = np.asarray(reference, float)
        if ref.size == 0 or ref.ndim != 2 or ref.shape[1] < 2:
            return None
        if len(ref) > 64:
            keep = np.linspace(0, len(ref) - 1, 64)
            indices = sorted(set(int(round(v)) for v in keep))
            if indices[0] != 0:
                indices.insert(0, 0)
            if indices[-1] != len(ref) - 1:
                indices.append(len(ref) - 1)
            ref = np.asarray([ref[i] for i in indices], float)
        start = np.asarray(self.state[:2], float)
        goal = np.asarray(self.goal[:2], float)
        to_goal = goal - start
        goal_dist = float(np.linalg.norm(to_goal))
        if goal_dist <= 1e-6:
            return None
        heading = np.array([
            np.cos(float(self.state[2])),
            np.sin(float(self.state[2]))], float)
        goal_dir = to_goal / goal_dist
        # Use the current heading when it already points generally goalward;
        # otherwise bias the prefix toward the goal rather than asking the
        # diff-drive base to track a reverse-facing first waypoint.
        blend = heading + goal_dir
        if float(np.linalg.norm(blend)) <= 1e-6 or np.dot(heading, goal_dir) < 0.0:
            blend = goal_dir
        launch_dir = blend / max(float(np.linalg.norm(blend)), 1e-9)
        step = max(0.06, min(0.12, 4.0 * float(self.topology_min_segment_length)))
        launch_len = min(0.55, max(0.24, 0.22 * goal_dist))
        prefix_count = max(3, int(np.ceil(launch_len / step)))
        prefix = []
        for i in range(prefix_count + 1):
            p2 = start + launch_dir * min(launch_len, step * i)
            if ref.shape[1] >= 3:
                prefix.append([p2[0], p2[1], 0.0])
            else:
                prefix.append([p2[0], p2[1]])
        prefix = np.asarray(prefix, float)
        join = prefix[-1, :2]
        candidates = []
        if len(ref) > 1:
            dists = np.linalg.norm(ref[:, :2] - join.reshape(1, 2), axis=1)
            nearest_idx = int(np.argmin(dists))
            # Keep this repair strictly bounded.  R002 showed that scanning a
            # large join/scale grid inside planning can stall before any
            # diagnostics are written.  These few joins cover the nearest
            # reconnection plus the empirically stable early-Morse handoff
            # region without turning refinement into another planner.
            join_indices = []
            for idx in (
                    nearest_idx,
                    nearest_idx + 2,
                    min(len(ref) - 1, 8),
                    min(len(ref) - 1, 10)):
                idx = int(min(max(idx, 1), len(ref) - 1))
                if idx not in join_indices:
                    join_indices.append(idx)
            for join_idx in sorted(join_indices):
                bridge_end = ref[join_idx]
                if join_idx + 1 < len(ref):
                    tail_vec = ref[join_idx + 1, :2] - ref[join_idx, :2]
                else:
                    tail_vec = goal - ref[join_idx, :2]
                tail_norm = float(np.linalg.norm(tail_vec))
                tail_dir = tail_vec / tail_norm if tail_norm > 1e-9 else goal_dir
                bridge_len = float(np.linalg.norm(bridge_end[:2] - start))
                if bridge_len <= 1e-6:
                    continue
                for scale in (0.45, 0.55):
                    c1 = start + heading * bridge_len * float(scale)
                    c2 = bridge_end[:2] - tail_dir * bridge_len * float(scale)
                    sample_count = min(
                        14, max(8, int(np.ceil(bridge_len / 0.10))))
                    bridge = []
                    for j in range(sample_count + 1):
                        u = float(j) / float(sample_count)
                        p2 = (
                            (1.0 - u) ** 3 * start +
                            3.0 * (1.0 - u) ** 2 * u * c1 +
                            3.0 * (1.0 - u) * u ** 2 * c2 +
                            u ** 3 * bridge_end[:2])
                        if ref.shape[1] >= 3:
                            bridge.append([p2[0], p2[1], 0.0])
                        else:
                            bridge.append([p2[0], p2[1]])
                    bridge = np.asarray(bridge, float)
                    tail = (
                        ref[join_idx + 1:]
                        if join_idx + 1 < len(ref) else ref[-1:])
                    repaired = np.vstack([bridge, tail])
                    candidates.append(np.asarray(repaired, float))
        if not candidates:
            repaired = np.vstack([prefix, ref])
            candidates.append(np.asarray(repaired, float))
        from stsm_madp.deform import path_curvature_metrics
        scored = []
        for candidate in candidates:
            profile = wheelchair_nonholonomic_execution_profile(
                candidate, self.state, self.goal,
                min_step=max(0.03, self.topology_min_segment_length),
                initial_lookahead=0.12,
                horizon_points=min(10, max(4, len(candidate))))
            turns = path_curvature_metrics(candidate)
            score = (
                max(0.0, float(turns.get("max_turn", 0.0)) - 0.40) * 20.0 +
                max(0.0, float(turns.get("max_curvature", 0.0)) - 8.0) * 4.0 +
                float(profile.get("execution_profile_cost", 0.0)))
            scored.append((score, candidate, profile, turns))
        if not scored:
            return None
        _score, best, _profile, _turns = min(
            scored, key=lambda item: item[0])
        return np.asarray(best, float)

    def _select_wheelchair_execution_reference(self, corr, refined, metrics):
        """Prefer a reference with executable heading and monotonic launch."""
        if self.baseline or self.state is None:
            return np.asarray(refined, float), dict(metrics or {}), False
        base = np.asarray(refined, float)
        if base.size == 0:
            return base, dict(metrics or {}), False
        current = wheelchair_nonholonomic_execution_profile(
            base, self.state, self.goal,
            min_step=max(0.03, self.topology_min_segment_length),
            initial_lookahead=0.12,
            horizon_points=min(10, max(4, len(base))))
        needs_repair = (
            float(current.get("initial_heading_error", 0.0)) > 1.85 or
            float(current.get("monotonic_regression_ratio", 0.0)) > 0.18 or
            float(current.get("nonmonotonic_fraction", 0.0)) > 0.30 or
            float(current.get("heading_oscillation", 0.0)) > 0.50)
        if not needs_repair:
            out = dict(metrics or {})
            out["nonholonomic_execution_profile"] = dict(current)
            out["diff_drive_reference_repaired"] = False
            return base, out, False
        from stsm_madp.deform import path_curvature_metrics
        candidates = [(
            "refined", base, current, path_curvature_metrics(base))]
        repaired = self._make_heading_progress_prefix(base, corr)
        if repaired is not None and len(repaired) >= 2:
            repaired_profile = wheelchair_nonholonomic_execution_profile(
                repaired, self.state, self.goal,
                min_step=max(0.03, self.topology_min_segment_length),
                initial_lookahead=0.12,
                horizon_points=min(10, max(4, len(repaired))))
            candidates.append((
                "diff_drive_launch_prefix",
                repaired,
                repaired_profile,
                path_curvature_metrics(repaired)))
        raw = np.asarray(getattr(corr, "raw_topology_waypoints", []), float)
        if raw.size == 0:
            raw = np.asarray(getattr(corr, "topology_ordered_waypoints", []), float)
        if raw.size > 0 and raw.ndim == 2 and len(raw) >= 2:
            raw_repaired = self._make_heading_progress_prefix(raw, corr)
            if raw_repaired is not None and len(raw_repaired) >= 2:
                raw_profile = wheelchair_nonholonomic_execution_profile(
                    raw_repaired, self.state, self.goal,
                    min_step=max(0.03, self.topology_min_segment_length),
                    initial_lookahead=0.12,
                    horizon_points=min(10, max(4, len(raw_repaired))))
                candidates.append((
                    "raw_diff_drive_launch_prefix",
                    raw_repaired,
                    raw_profile,
                    path_curvature_metrics(raw_repaired)))
        best_source, best_path, best_profile, best_turn_metrics = min(
            candidates,
            key=lambda item: (
                max(0.0, float(item[3].get("max_turn", 0.0)) - 0.40) * 20.0 +
                max(0.0, float(item[3].get("max_curvature", 0.0)) - 8.0) * 4.0 +
                float(item[2].get("execution_profile_cost", 0.0))))
        repaired_used = best_source != "refined" and (
            float(best_profile.get("execution_profile_cost", 0.0)) + 1e-6 <
            float(current.get("execution_profile_cost", 0.0)))
        out = dict(metrics or {})
        out["nonholonomic_execution_profile"] = dict(best_profile)
        out["pre_repair_nonholonomic_execution_profile"] = dict(current)
        out["diff_drive_reference_repaired"] = bool(repaired_used)
        out["diff_drive_reference_source"] = str(best_source)
        if repaired_used:
            from stsm_madp.deform import path_curvature_metrics, path_length
            repaired_metrics = dict(best_turn_metrics)
            out.update(repaired_metrics)
            out["refined_path_length"] = float(path_length(best_path))
            out["reference_path_count"] = int(len(best_path))
            out["reference_source"] = str(best_source)
            trace = list(getattr(corr, "refinement_trace", []) or [])
            trace.append({
                "iteration": "diff_drive_launch_prefix",
                "accepted": True,
                "failure_reason": "",
                "trajectory_valid": True,
                "execution_profile_cost_before": float(
                    current.get("execution_profile_cost", 0.0)),
                "execution_profile_cost_after": float(
                    best_profile.get("execution_profile_cost", 0.0)),
                "initial_heading_error_before": float(
                    current.get("initial_heading_error", 0.0)),
                "initial_heading_error_after": float(
                    best_profile.get("initial_heading_error", 0.0)),
                "monotonic_regression_before": float(
                    current.get("monotonic_regression", 0.0)),
                "monotonic_regression_after": float(
                    best_profile.get("monotonic_regression", 0.0)),
            })
            corr.refinement_trace = trace
            out["refinement_trace"] = trace
        return np.asarray(best_path, float), out, bool(repaired_used)

    def _prepare_executable_corridors(self, corrs):
        if not self.topology_refinement_enabled:
            if not self.baseline:
                rospy.logerr("[wc][refine] disabled in STSM mode")
                return []
            return list(corrs or [])
        prepared = []
        refinement_attempts = []
        max_refine = max(1, int(self.topology_refinement_max_candidates))
        for index, corr in enumerate(list(corrs or [])):
            cid = str(getattr(corr, "corridor_id", getattr(corr, "label", "")))
            if index >= max_refine:
                corr.refinement_skipped = True
                corr.refinement_skip_reason = "refinement_max_candidates"
                refinement_attempts.append({
                    "robot_type": "wheelchair",
                    "stage": "refine_topology_path",
                    "corridor_id": cid,
                    "candidate_id": cid,
                    "label": str(getattr(corr, "label", cid)),
                    "accepted": False,
                    "reject_reason": "refinement_max_candidates",
                    "failure_reason": "refinement_max_candidates",
                    "reference_source": "unrefined_candidate_for_ranking_only",
                    "reference_path_count": int(len(np.asarray(
                        getattr(corr, "waypoints",
                                getattr(corr, "centerline", [])), float))),
                    "refinement_candidate_index": int(index),
                    "refinement_max_candidates": int(max_refine),
                })
                rospy.logwarn(
                    "[wc][refine] skip %s reason=refinement_max_candidates index=%d max=%d",
                    cid, index, max_refine)
                continue
            if not self._corridor_is_topological(corr):
                prepared.append(corr)
                continue
            rospy.loginfo(
                "[wc][refine] start %s index=%d samples=%d max_points=%d footprint_points=%d",
                cid, int(index), int(self.topology_refinement_samples),
                int(self.max_refinement_path_points),
                int(self.max_refined_footprint_check_points))
            refine_t0 = time.time()
            ok, refined, metrics, reason = refine_topology_path(
                corr,
                samples_per_segment=self.topology_refinement_samples,
                max_curvature=min(
                    float(self.topology_max_corridor_curvature), 8.0),
                max_turn=self.topology_max_corridor_turn,
                footprint_checker=self._footprint_path_checker,
                max_refinement_points=self.max_refinement_path_points)
            rospy.loginfo(
                "[wc][refine] done %s ok=%s reason=%s elapsed=%.3fs",
                cid, bool(ok), str(reason), time.time() - refine_t0)
            attempt = self._refinement_attempt_payload(
                corr, metrics, reason, stage="refine_topology_path")
            attempt["max_refinement_path_points"] = int(
                self.max_refinement_path_points)
            attempt["max_refined_footprint_check_points"] = int(
                self.max_refined_footprint_check_points)
            attempt["refined_footprint_checked_points"] = int(
                self.last_refined_footprint_checked_points)
            if not ok:
                corr.reject_reason = str(reason)
                attempt["accepted"] = False
                attempt["reject_reason"] = str(corr.reject_reason)
                refinement_attempts.append(attempt)
                rospy.logwarn("[wc][refine] reject %s reason=%s",
                              getattr(corr, "corridor_id",
                                      getattr(corr, "label", "")),
                              reason)
                continue
            reference_source = str(metrics.get(
                "reference_source",
                getattr(corr, "final_reference_source", "refined")))
            fallback_used = bool(metrics.get(
                "refinement_fallback",
                getattr(corr, "refinement_fallback", False)))
            refined_max_curvature = float(metrics.get("max_curvature", 0.0))
            refined_max_turn = float(metrics.get("max_turn", 0.0))
            executable_curvature_limit = min(
                float(self.topology_max_corridor_curvature), 8.0)
            executable_turn_limit = min(
                float(self.topology_max_corridor_turn), 0.40)
            executable_turn_tolerance = 0.03
            if (not fallback_used and refined_max_turn > (
                    executable_turn_limit + executable_turn_tolerance + 1e-9)):
                recovery = self._recover_refined_turn_limit(
                    corr, refined, executable_turn_limit,
                    executable_turn_tolerance)
                if recovery is not None:
                    refined, recovery_metrics = recovery
                    metrics.update(recovery_metrics)
                    refined_max_curvature = float(metrics.get(
                        "max_curvature", refined_max_curvature))
                    refined_max_turn = float(metrics.get(
                        "max_turn", refined_max_turn))
                    attempt["turn_recovery_used"] = True
                    attempt["turn_recovery_metrics"] = dict(recovery_metrics)
            refined, execution_metrics, execution_repaired = (
                self._select_wheelchair_execution_reference(
                    corr, refined, metrics))
            if execution_repaired:
                metrics.update(execution_metrics)
                reference_source = str(metrics.get(
                    "reference_source", reference_source))
                refined_max_curvature = float(metrics.get(
                    "max_curvature", refined_max_curvature))
                refined_max_turn = float(metrics.get(
                    "max_turn", refined_max_turn))
                if refined_max_turn > (
                        executable_turn_limit +
                        executable_turn_tolerance + 1e-9):
                    recovery = self._recover_refined_turn_limit(
                        corr, refined, executable_turn_limit,
                        executable_turn_tolerance)
                    if recovery is not None:
                        recovered, recovery_metrics = recovery
                        recovered_profile = (
                            wheelchair_nonholonomic_execution_profile(
                                recovered, self.state, self.goal,
                                min_step=max(
                                    0.03, self.topology_min_segment_length),
                                initial_lookahead=0.12,
                                horizon_points=min(
                                    10, max(4, len(np.asarray(
                                        recovered, float))))))
                        pre_profile = dict(metrics.get(
                            "pre_repair_nonholonomic_execution_profile", {}))
                        pre_cost = float(pre_profile.get(
                            "execution_profile_cost",
                            metrics.get("nonholonomic_execution_profile", {})
                            .get("execution_profile_cost", float("inf"))))
                        if (
                                float(recovered_profile.get(
                                    "execution_profile_cost", float("inf"))) <
                                pre_cost and
                                float(recovered_profile.get(
                                    "initial_heading_error", 0.0)) < 1.50):
                            refined = np.asarray(recovered, float)
                            metrics.update(recovery_metrics)
                            metrics["nonholonomic_execution_profile"] = dict(
                                recovered_profile)
                            metrics["diff_drive_reference_repaired"] = True
                            metrics["diff_drive_reference_source"] = (
                                "diff_drive_launch_prefix_turn_recovered")
                            metrics["reference_source"] = (
                                "diff_drive_launch_prefix_turn_recovered")
                            reference_source = str(metrics["reference_source"])
                            refined_max_curvature = float(metrics.get(
                                "max_curvature", refined_max_curvature))
                            refined_max_turn = float(metrics.get(
                                "max_turn", refined_max_turn))
                            attempt["diff_drive_turn_recovery_used"] = True
                            attempt["diff_drive_turn_recovery_metrics"] = dict(
                                recovery_metrics)
                attempt["diff_drive_reference_repaired"] = True
                attempt["diff_drive_reference_source"] = str(
                    metrics.get("diff_drive_reference_source",
                                reference_source))
                attempt["nonholonomic_execution_profile"] = dict(
                    metrics.get("nonholonomic_execution_profile", {}))
                attempt["pre_repair_nonholonomic_execution_profile"] = dict(
                    metrics.get(
                        "pre_repair_nonholonomic_execution_profile", {}))
            else:
                metrics.update(execution_metrics)
            if execution_repaired and not fallback_used:
                ok, footprint_reason = self._footprint_path_checker(refined)
                if not ok:
                    corr.reject_reason = (
                        "diff_drive_launch_prefix_footprint:" +
                        str(footprint_reason))
                    attempt["accepted"] = False
                    attempt["reject_reason"] = str(corr.reject_reason)
                    attempt["diff_drive_reference_repaired"] = True
                    attempt["diff_drive_reference_source"] = str(metrics.get(
                        "diff_drive_reference_source", reference_source))
                    attempt["nonholonomic_execution_profile"] = dict(
                        metrics.get("nonholonomic_execution_profile", {}))
                    refinement_attempts.append(attempt)
                    rospy.logwarn(
                        "[wc][refine] reject %s reason=%s",
                        getattr(corr, "corridor_id",
                                getattr(corr, "label", "")),
                        corr.reject_reason)
                    continue
            if (not fallback_used and
                    refined_max_curvature > executable_curvature_limit + 1e-9):
                corr.reject_reason = "refined_execution_curvature_limit"
                attempt["accepted"] = False
                attempt["reject_reason"] = str(corr.reject_reason)
                attempt["execution_curvature_limit"] = float(
                    executable_curvature_limit)
                attempt["execution_turn_limit"] = float(executable_turn_limit)
                refinement_attempts.append(attempt)
                rospy.logwarn(
                    "[wc][refine] reject %s reason=%s max_curv=%.3f limit=%.3f",
                    getattr(corr, "corridor_id", getattr(corr, "label", "")),
                    corr.reject_reason, refined_max_curvature,
                    executable_curvature_limit)
                continue
            if (not fallback_used and refined_max_turn > (
                    executable_turn_limit + executable_turn_tolerance + 1e-9)):
                corr.reject_reason = "refined_execution_turn_limit"
                attempt["accepted"] = False
                attempt["reject_reason"] = str(corr.reject_reason)
                attempt["execution_curvature_limit"] = float(
                    executable_curvature_limit)
                attempt["execution_turn_limit"] = float(executable_turn_limit)
                attempt["execution_turn_tolerance"] = float(
                    executable_turn_tolerance)
                refinement_attempts.append(attempt)
                rospy.logwarn(
                    "[wc][refine] reject %s reason=%s max_turn=%.3f limit=%.3f",
                    getattr(corr, "corridor_id", getattr(corr, "label", "")),
                    corr.reject_reason, refined_max_turn,
                    executable_turn_limit)
                continue
            if refined_max_turn > executable_turn_limit + 1e-9:
                rospy.logwarn(
                    "[wc][refine] accept %s with turn tolerance max_turn=%.3f limit=%.3f tolerance=%.3f",
                    getattr(corr, "corridor_id", getattr(corr, "label", "")),
                    refined_max_turn, executable_turn_limit,
                    executable_turn_tolerance)
            attempt["accepted"] = True
            attempt["reject_reason"] = ""
            attempt["execution_curvature_limit"] = float(
                executable_curvature_limit)
            attempt["execution_turn_limit"] = float(executable_turn_limit)
            attempt["execution_turn_tolerance"] = float(
                executable_turn_tolerance)
            attempt["diff_drive_reference_repaired"] = bool(
                metrics.get("diff_drive_reference_repaired", False))
            attempt["diff_drive_reference_source"] = str(metrics.get(
                "diff_drive_reference_source", reference_source))
            attempt["nonholonomic_execution_profile"] = dict(metrics.get(
                "nonholonomic_execution_profile", {}))
            refinement_attempts.append(attempt)
            corr.waypoints = np.asarray(refined, float)
            corr.refined_waypoints = np.asarray(refined, float)
            corr.centerline = np.asarray(refined, float)
            corr.refined_path_length = float(metrics.get(
                "refined_path_length",
                self._path_length(np.asarray(refined, float))))
            corr.refined_max_turn_angle = float(metrics.get("max_turn", 0.0))
            corr.refined_mean_turn_angle = float(metrics.get("mean_turn", 0.0))
            corr.refined_max_curvature = float(metrics.get("max_curvature", 0.0))
            refinement_output = dict(getattr(corr, "refinement_output", {}) or {})
            refinement_output["trajectory"] = np.asarray(refined, float).tolist()
            refinement_output["final_trajectory"] = np.asarray(refined, float).tolist()
            refinement_output["reference_source"] = str(reference_source)
            refinement_output["reference_path_count"] = int(len(np.asarray(refined, float)))
            refinement_output["diff_drive_reference_repaired"] = bool(
                metrics.get("diff_drive_reference_repaired", False))
            refinement_output["diff_drive_reference_source"] = str(metrics.get(
                "diff_drive_reference_source", reference_source))
            refinement_output["nonholonomic_execution_profile"] = dict(
                metrics.get("nonholonomic_execution_profile", {}))
            refinement_output["pre_repair_nonholonomic_execution_profile"] = dict(
                metrics.get("pre_repair_nonholonomic_execution_profile", {}))
            corr.refinement_output = refinement_output
            association = associate_corridor_critical_points(corr, refined)
            corr.critical_point_association = association
            corr.critical_point_projection_index = {
                str(item.get("id", "")): int(item.get("trajectory_index", -1))
                for item in association.get("critical_points", [])
            }
            refinement_output = dict(getattr(corr, "refinement_output", {}) or {})
            refinement_output["critical_point_association"] = association
            refinement_output["topology_stage_sequence"] = [
                {
                    "id": str(item.get("id", "")),
                    "type": str(item.get("type", "")),
                    "trajectory_index": int(item.get("trajectory_index", -1)),
                    "stage_order": int(item.get(
                        "stage_order", item.get("order", idx + 1))),
                    "critical_point_status": str(item.get(
                        "critical_point_status", "")),
                }
                for idx, item in enumerate(
                    association.get("critical_points", []))
            ]
            corr.refinement_output = refinement_output
            corr.refinement_used = int(not fallback_used)
            corr.refinement_success = bool(not fallback_used)
            corr.refinement_fallback = bool(fallback_used)
            corr.final_reference_source = reference_source
            corr.refinement_reject_reason = str(
                metrics.get("fallback_reason", "")) if fallback_used else ""
            corr.path_length = float(corr.refined_path_length)
            corr.max_turn_angle = float(metrics.get("max_turn", 0.0))
            corr.mean_turn_angle = float(metrics.get("mean_turn", 0.0))
            corr.max_curvature = float(metrics.get("max_curvature", 0.0))
            nonholonomic_profile = wheelchair_nonholonomic_execution_profile(
                refined, self.state, self.goal,
                min_step=max(0.03, self.topology_min_segment_length),
                initial_lookahead=0.12,
                horizon_points=min(10, max(4, len(np.asarray(refined, float)))))
            corr.nonholonomic_execution_profile = dict(nonholonomic_profile)
            corr.diff_drive_reference_repaired = bool(metrics.get(
                "diff_drive_reference_repaired", False))
            corr.diff_drive_reference_source = str(metrics.get(
                "diff_drive_reference_source", reference_source))
            corr.diff_drive_execution_cost = float(
                nonholonomic_profile.get("execution_profile_cost", 0.0))
            corr.initial_heading_error = float(
                nonholonomic_profile.get("initial_heading_error", 0.0))
            corr.monotonic_regression = float(
                nonholonomic_profile.get("monotonic_regression", 0.0))
            corr.nonmonotonic_fraction = float(
                nonholonomic_profile.get("nonmonotonic_fraction", 0.0))
            corr.heading_oscillation = float(
                nonholonomic_profile.get("heading_oscillation", 0.0))
            corr.execution_cost = float(
                corr.max_turn_angle + 0.2 * corr.max_curvature +
                corr.diff_drive_execution_cost)
            corr.motion_cost = float(corr.execution_cost)
            center_vals = [
                float(self.field.phi_s(np.array([p[0], p[1], 0.0], float)))
                for p in np.asarray(refined, float)
            ]
            corr.mean_phi_on_path = (
                float(np.mean(center_vals)) if center_vals else 0.0)
            corr.max_phi_on_path = (
                float(np.max(center_vals)) if center_vals else 0.0)
            corr.risk_per_meter = float(corr.mean_phi_on_path)
            corr.tracking_cost = float(
                np.sum(np.asarray(metrics.get("turns", []), float) ** 2)
                if "turns" in metrics else
                corr.max_turn_angle * corr.max_turn_angle +
                0.2 * corr.max_curvature)
            corr.total_cost = float(getattr(corr, "cost", 0.0))
            breakdown = dict(getattr(corr, "candidate_cost_breakdown", {}) or {})
            breakdown["nonholonomic_execution_profile"] = dict(
                nonholonomic_profile)
            breakdown["diff_drive_execution_cost"] = float(
                corr.diff_drive_execution_cost)
            breakdown["initial_heading_error"] = float(
                corr.initial_heading_error)
            breakdown["monotonic_regression"] = float(
                corr.monotonic_regression)
            breakdown["nonmonotonic_fraction"] = float(
                corr.nonmonotonic_fraction)
            breakdown["heading_oscillation"] = float(
                corr.heading_oscillation)
            breakdown["diff_drive_reference_repaired"] = bool(
                corr.diff_drive_reference_repaired)
            breakdown["diff_drive_reference_source"] = str(
                corr.diff_drive_reference_source)
            corr.candidate_cost_breakdown = breakdown
            prepared.append(corr)
            rospy.loginfo(
                "[wc][refine] accepted %s length=%.3f max_turn=%.3f max_curv=%.3f diff_drive=%.3f init_head=%.3f mono_reg=%.3f source=%s",
                getattr(corr, "corridor_id", getattr(corr, "label", "")),
                corr.path_length, corr.max_turn_angle, corr.max_curvature,
                corr.diff_drive_execution_cost, corr.initial_heading_error,
                corr.monotonic_regression,
                reference_source)
        self._sync_refinement_attempt_debug(refinement_attempts)
        return prepared

    def _recover_refined_turn_limit(self, corr, refined,
                                    executable_turn_limit,
                                    executable_turn_tolerance):
        from stsm_madp.deform import path_curvature_metrics, path_length

        limit = float(executable_turn_limit) + float(executable_turn_tolerance)
        best = np.asarray(refined, float)
        if len(best) > 64:
            keep = np.linspace(0, len(best) - 1, 64)
            indices = sorted(set(int(round(v)) for v in keep))
            if indices[0] != 0:
                indices.insert(0, 0)
            if indices[-1] != len(best) - 1:
                indices.append(len(best) - 1)
            best = np.asarray([best[i] for i in indices], float)
        best_metrics = path_curvature_metrics(best)
        rospy.loginfo(
            "[wc][refine] turn recovery start %s points=%d max_turn=%.3f limit=%.3f",
            getattr(corr, "corridor_id", getattr(corr, "label", "")),
            int(len(best)), float(best_metrics.get("max_turn", 0.0)), limit)
        for samples, passes in ((4, 1), (6, 1), (8, 1)):
            candidate = smooth_wheelchair_corners(
                best, samples_per_segment=samples, passes=passes)
            if len(candidate) > 64:
                keep = np.linspace(0, len(candidate) - 1, 64)
                indices = sorted(set(int(round(v)) for v in keep))
                if indices[0] != 0:
                    indices.insert(0, 0)
                if indices[-1] != len(candidate) - 1:
                    indices.append(len(candidate) - 1)
                candidate = np.asarray([candidate[i] for i in indices], float)
            ok, reason = self._footprint_path_checker(candidate)
            metrics = path_curvature_metrics(candidate)
            if not ok:
                rospy.logwarn(
                    "[wc][refine] turn recovery reject footprint %s samples=%d reason=%s points=%d",
                    getattr(corr, "corridor_id", getattr(corr, "label", "")),
                    int(samples), str(reason), int(len(candidate)))
                continue
            if (float(metrics.get("max_turn", 0.0)) <= limit + 1e-9 and
                    float(metrics.get("max_curvature", 0.0)) <=
                    min(float(self.topology_max_corridor_curvature), 8.0) + 1e-9):
                metrics["refined_path_length"] = float(path_length(candidate))
                metrics["reference_path_count"] = int(len(candidate))
                metrics["reference_source"] = "turn_recovered_refined"
                metrics["turn_recovery_used"] = True
                metrics["turn_recovery_samples_per_segment"] = int(samples)
                metrics["turn_recovery_passes"] = int(passes)
                metrics["turn_recovery_reason"] = "execution_turn_limit"
                metrics["turn_recovery_bounded"] = True
                metrics["failure_reason"] = ""
                metrics["success"] = True
                trace = list(getattr(corr, "refinement_trace", []) or [])
                trace.append({
                    "iteration": "execution_turn_recovery",
                    "accepted": True,
                    "failure_reason": "",
                    "trajectory_valid": True,
                    "max_turn": float(metrics.get("max_turn", 0.0)),
                    "max_curvature": float(metrics.get("max_curvature", 0.0)),
                    "reference_path_count": int(len(candidate)),
                    "turn_recovery_bounded": True,
                })
                corr.refinement_trace = trace
                metrics["refinement_trace"] = trace
                return np.asarray(candidate, float), metrics
            if float(metrics.get("max_turn", 0.0)) < float(
                    best_metrics.get("max_turn", 0.0)):
                best = candidate
                best_metrics = dict(metrics)
        rospy.logwarn(
            "[wc][refine] turn recovery failed %s max_turn=%.3f limit=%.3f",
            getattr(corr, "corridor_id", getattr(corr, "label", "")),
            float(best_metrics.get("max_turn", 0.0)), limit)
        return None

    def _refinement_attempt_payload(self, corr, metrics, reason,
                                    stage="refine_topology_path"):
        metrics = dict(metrics or {})
        cid = str(getattr(corr, "corridor_id", getattr(corr, "label", "")))
        trace = list(metrics.get(
            "refinement_trace",
            getattr(corr, "refinement_trace", [])) or [])
        refined_points = getattr(corr, "refined_waypoints", [])
        try:
            refined_count = len(refined_points)
        except Exception:
            refined_count = 0
        return {
            "robot_type": "wheelchair",
            "stage": str(stage),
            "corridor_id": cid,
            "candidate_id": cid,
            "label": str(getattr(corr, "label", cid)),
            "accepted": bool(metrics.get("success", False)),
            "reject_reason": str(reason or metrics.get("failure_reason", "")),
            "failure_reason": str(metrics.get("failure_reason", reason or "")),
            "fallback_used": bool(metrics.get("fallback_used", False)),
            "fallback_reason": str(metrics.get("fallback_reason", "")),
            "reference_source": str(metrics.get(
                "reference_source",
                getattr(corr, "final_reference_source", ""))),
            "reference_path_count": int(metrics.get(
                "reference_path_count", refined_count) or 0),
            "max_turn": float(metrics.get(
                "max_turn", getattr(corr, "refined_max_turn_angle", 0.0)) or 0.0),
            "mean_turn": float(metrics.get(
                "mean_turn", getattr(corr, "refined_mean_turn_angle", 0.0)) or 0.0),
            "max_curvature": float(metrics.get(
                "max_curvature", getattr(corr, "refined_max_curvature", 0.0)) or 0.0),
            "refined_path_length": float(metrics.get(
                "refined_path_length",
                getattr(corr, "refined_path_length", 0.0)) or 0.0),
            "pre_refinement_clearance": float(metrics.get(
                "pre_refinement_clearance",
                getattr(corr, "pre_refinement_clearance", 0.0)) or 0.0),
            "post_refinement_clearance": float(metrics.get(
                "post_refinement_clearance",
                getattr(corr, "post_refinement_clearance", 0.0)) or 0.0),
            "refinement_manifold_valid": bool(metrics.get(
                "refinement_manifold_valid",
                getattr(corr, "refinement_manifold_valid", False))),
            "refinement_tube_valid": bool(metrics.get(
                "refinement_tube_valid",
                getattr(corr, "refinement_tube_valid", False))),
            "topology_tracking_error": float(metrics.get(
                "topology_tracking_error",
                getattr(corr, "topology_tracking_error", 0.0)) or 0.0),
            "critical_point_status": str(metrics.get(
                "critical_point_status", "")),
            "topology_sequence_valid": bool(metrics.get(
                "topology_sequence_valid", False)),
            "topology_stage_sequence": list(metrics.get(
                "topology_stage_sequence", [])),
            "refinement_trace": trace,
        }

    def _sync_refinement_attempt_debug(self, refinement_attempts):
        attempts = list(refinement_attempts or [])
        dbg = dict(getattr(self.manifold, "last_topology_debug", {}) or {})
        dbg["refinement_attempts"] = attempts
        dbg["refinement_trace"] = attempts
        dbg["topology_refinement"] = {
            "robot_type": "wheelchair",
            "attempted": bool(attempts),
            "attempt_count": int(len(attempts)),
            "accepted_count": int(sum(
                1 for item in attempts
                if bool(item.get("accepted", False)))),
            "rejected_count": int(sum(
                1 for item in attempts
                if not bool(item.get("accepted", False)))),
            "attempts": attempts,
        }
        self.manifold.last_topology_debug = dbg

    def _as_corridor_points(self, value):
        if value is None or isinstance(value, str):
            return np.zeros((0, 3), float)
        try:
            arr = np.asarray(value, float)
        except Exception:
            return np.zeros((0, 3), float)
        if arr.size == 0:
            return np.zeros((0, 3), float)
        if arr.ndim == 1:
            arr = arr.reshape((1, arr.shape[0]))
        if arr.shape[1] == 2:
            arr = np.hstack([arr, np.zeros((arr.shape[0], 1), float)])
        return np.asarray(arr[:, :3], float)

    def _final_corridor_trajectory(self, corridor):
        if corridor is None:
            return np.zeros((0, 3), float), "fallback"
        refinement_output = dict(getattr(corridor, "refinement_output", {}) or {})
        for value, source in (
                (refinement_output.get("final_trajectory"), "refinement"),
                (refinement_output.get("trajectory"), "refinement"),
                (getattr(corridor, "refined_waypoints", None), "refinement"),
                (getattr(corridor, "centerline", None), "candidate"),
                (getattr(corridor, "waypoints", None), "candidate")):
            pts = self._as_corridor_points(value)
            if len(pts) > 0:
                return pts, source
        return np.zeros((0, 3), float), "fallback"

    def _sync_selected_corridor_geometry(self, corridor):
        pts, source = self._final_corridor_trajectory(corridor)
        if corridor is None or len(pts) == 0:
            return pts, source
        corridor.centerline = np.asarray(pts, float)
        corridor.refined_waypoints = np.asarray(pts, float)
        corridor.waypoints = np.asarray(pts, float)
        if not str(getattr(corridor, "final_reference_source", "")):
            corridor.final_reference_source = (
                "refined" if source == "refinement" else
                "candidate_fallback" if source == "candidate" else "fallback")
        output = dict(getattr(corridor, "refinement_output", {}) or {})
        output["final_trajectory"] = np.asarray(pts, float).tolist()
        output["reference_source"] = str(getattr(
            corridor, "final_reference_source", source))
        output["reference_path_count"] = int(len(pts))
        corridor.refinement_output = output
        return pts, source

    def _offset_boundaries_from_centerline(self, centerline, radius):
        centerline = self._as_corridor_points(centerline)
        if len(centerline) == 0:
            return [], []
        radius = float(radius if radius not in (None, "") else 0.0)
        left = []
        right = []
        for idx, p in enumerate(centerline):
            if len(centerline) == 1:
                tangent = np.array([1.0, 0.0], float)
            elif idx == 0:
                tangent = centerline[1, :2] - centerline[0, :2]
            elif idx == len(centerline) - 1:
                tangent = centerline[-1, :2] - centerline[-2, :2]
            else:
                tangent = centerline[idx + 1, :2] - centerline[idx - 1, :2]
            norm = float(np.linalg.norm(tangent))
            if norm < 1e-9:
                normal = np.array([0.0, 1.0], float)
            else:
                t = tangent / norm
                normal = np.array([-t[1], t[0]], float)
            lp = np.array([p[0], p[1], p[2]], float)
            rp = np.array([p[0], p[1], p[2]], float)
            lp[:2] += radius * normal
            rp[:2] -= radius * normal
            left.append(lp.tolist())
            right.append(rp.tolist())
        return left, right

    def _corridor_boundary_points(self, corridor, centerline):
        boundary = dict(getattr(corridor, "boundary", {}) or {}) if corridor is not None else {}
        left = self._as_corridor_points(
            boundary.get("left", boundary.get("left_boundary", []))).tolist()
        right = self._as_corridor_points(
            boundary.get("right", boundary.get("right_boundary", []))).tolist()
        if left and right:
            return left, right
        radius = float(getattr(corridor, "radius", 0.35)) if corridor is not None else 0.35
        return self._offset_boundaries_from_centerline(centerline, radius)

    def _selected_corridor_payload(self, corridor):
        centerline, reference_source = self._sync_selected_corridor_geometry(corridor)
        cid = str(getattr(corridor, "corridor_id", getattr(corridor, "label", ""))) if corridor is not None else ""
        radius = float(getattr(corridor, "radius", 0.35)) if corridor is not None else 0.35
        tube = generate_topology_tube(centerline, radius)
        left, right = self._corridor_boundary_points(corridor, centerline)
        tube["left_boundary"] = left
        tube["right_boundary"] = right
        source_raw = str(getattr(
            corridor, "final_reference_source", reference_source)) if corridor is not None else reference_source
        source = (
            "refinement" if source_raw in ("refined", "refined_waypoints") else
            "candidate" if source_raw == "candidate_fallback" else
            source_raw if source_raw else "fallback")
        return {
            "corridor_id": cid,
            "selected_corridor_id": cid,
            "label": str(getattr(corridor, "label", cid)) if corridor is not None else "",
            "candidate_source": str(getattr(
                corridor, "candidate_source",
                getattr(corridor, "route_source", ""))) if corridor is not None else "",
            "topology_class": str(getattr(
                corridor, "topology_route_class",
                getattr(corridor, "topology_class", ""))) if corridor is not None else "",
            "node_sequence": list(getattr(
                corridor, "node_sequence", []) or []) if corridor is not None else [],
            "critical_point_ids": list(getattr(
                corridor, "critical_point_ids",
                getattr(corridor, "morse_node_ids", [])) or []) if corridor is not None else [],
            "recovery_level": str(getattr(
                corridor, "recovery_level",
                getattr(corridor, "candidate_recovery_mode", "none"))) if corridor is not None else "none",
            "corridor_contract_version": str(getattr(
                corridor, "corridor_contract_version", "")) if corridor is not None else "",
            "reference_source": source,
            "centerline": centerline.tolist(),
            "centerline_count": int(len(centerline)),
            "left_boundary": left,
            "right_boundary": right,
            "tube_points": list(tube.get("tube_points", [])),
            "tube_point_count": int(len(tube.get("tube_points", []))),
            "radius": radius,
            "refinement_success": bool(getattr(
                corridor, "refinement_success", False)) if corridor is not None else False,
            "refinement_fallback": bool(getattr(
                corridor, "refinement_fallback", False)) if corridor is not None else False,
            "nonholonomic_execution_profile": dict(getattr(
                corridor, "nonholonomic_execution_profile", {}) or {}) if corridor is not None else {},
            "diff_drive_execution_cost": float(getattr(
                corridor, "diff_drive_execution_cost", 0.0)) if corridor is not None else 0.0,
            "initial_heading_error": float(getattr(
                corridor, "initial_heading_error", 0.0)) if corridor is not None else 0.0,
            "monotonic_regression": float(getattr(
                corridor, "monotonic_regression", 0.0)) if corridor is not None else 0.0,
            "nonmonotonic_fraction": float(getattr(
                corridor, "nonmonotonic_fraction", 0.0)) if corridor is not None else 0.0,
            "heading_oscillation": float(getattr(
                corridor, "heading_oscillation", 0.0)) if corridor is not None else 0.0,
            "diff_drive_reference_repaired": bool(getattr(
                corridor, "diff_drive_reference_repaired", False)) if corridor is not None else False,
            "diff_drive_reference_source": str(getattr(
                corridor, "diff_drive_reference_source", "")) if corridor is not None else "",
        }

    def _write_selected_corridor_debug(self, out_dir):
        if self.baseline or self.selected_corridor is None or not out_dir:
            return {}
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        payload = self._selected_corridor_payload(self.selected_corridor)
        with open(os.path.join(out_dir, "selected_corridor.json"), "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        return payload

    def _sync_runtime_topology_debug(self, corrs, selected=None):
        dbg = dict(getattr(self.manifold, "last_topology_debug", {}) or {})
        rows = []
        selected_id = str(getattr(selected, "corridor_id", "")) if selected is not None else ""
        for rank, corr in enumerate(list(corrs or []), start=1):
            cid = str(getattr(corr, "corridor_id", ""))
            task_state_info = infer_task_state(
                "wheelchair", self.task_mode, phase="moving", progress=0.0)
            task_state = str(getattr(
                corr, "task_state", task_state_info.get("task_state", "moving")))
            task_breakdown = dict(getattr(
                corr, "task_cost_breakdown", {}) or {})
            if not task_breakdown:
                task_breakdown = evaluate_task_cost_breakdown(
                    corr, "wheelchair", start=self.start_pose,
                    goal=self.goal, task_mode=self.task_mode,
                    task_state=task_state)
            task_cost_value = float(task_breakdown.get(
                "task_cost", getattr(corr, "task_cost", 0.0)) or 0.0)
            feasibility_cost_value = float(getattr(
                corr, "feasibility_cost",
                task_breakdown.get("terms", {}).get("feasibility_cost", 0.0))
                or 0.0)
            row = {
                "rank": rank,
                "corridor_id": cid,
                "label": str(getattr(corr, "label", "")),
                "source": str(getattr(corr, "source", "")),
                "route_source": str(getattr(corr, "route_source", "")),
                "candidate_source": str(getattr(
                    corr, "candidate_source",
                    getattr(corr, "route_source", ""))),
                "route_generation_level": str(getattr(
                    corr, "route_generation_level", "")),
                "candidate_generation_role": str(getattr(corr, "candidate_generation_role", "")),
                "node_sequence": list(getattr(corr, "node_sequence", [])),
                "node_type_sequence": list(getattr(corr, "node_type_sequence", [])),
                "semantic_sequence": list(getattr(corr, "semantic_sequence", [])),
                "topology_class": str(getattr(corr, "topology_class", "")),
                "topology_route_class": str(getattr(
                    corr, "topology_route_class",
                    getattr(corr, "topology_class", ""))),
                "task_semantic_class": str(getattr(
                    corr, "task_semantic_class", "")),
                "source_graph_id": str(getattr(corr, "source_graph_id", "")),
                "source_saddle_ids": list(getattr(corr, "source_saddle_ids", [])),
                "source_minima_ids": list(getattr(corr, "source_minima_ids", [])),
                "topology_ordered_waypoints": np.asarray(getattr(corr, "topology_ordered_waypoints", []), float).tolist(),
                "waypoints": np.asarray(getattr(corr, "waypoints", []), float).tolist(),
                "raw_topology_waypoints": np.asarray(getattr(corr, "raw_topology_waypoints", []), float).tolist(),
                "refined_waypoints": np.asarray(getattr(corr, "refined_waypoints", []), float).tolist(),
                "critical_point_association": dict(getattr(
                    corr, "critical_point_association", {}) or {}),
                "critical_point_projection_index": dict(getattr(
                    corr, "critical_point_projection_index", {}) or {}),
                "refinement_used": int(getattr(corr, "refinement_used", 0)),
                "refinement_success": bool(getattr(
                    corr, "refinement_success",
                    int(getattr(corr, "refinement_used", 0)) == 1)),
                "refinement_fallback": bool(getattr(
                    corr, "refinement_fallback", False)),
                "reference_source": str(getattr(
                    corr, "final_reference_source",
                    "refined" if int(getattr(corr, "refinement_used", 0)) == 1
                    else "candidate_fallback")),
                "reference_path_count": int(len(np.asarray(
                    getattr(corr, "refined_waypoints",
                            getattr(corr, "centerline", [])), float))),
                "refinement_reject_reason": str(getattr(corr, "refinement_reject_reason", "")),
                "refined_path_length": float(getattr(corr, "refined_path_length", 0.0)),
                "refined_max_turn_angle": float(getattr(corr, "refined_max_turn_angle", 0.0)),
                "refined_max_curvature": float(getattr(corr, "refined_max_curvature", 0.0)),
                "risk_cost": float(getattr(corr, "risk_cost", 0.0)),
                "length_cost": float(getattr(corr, "length_cost", getattr(corr, "distance_cost", 0.0))),
                "smooth_cost": float(getattr(corr, "smooth_cost", 0.0)),
                "task_cost": task_cost_value,
                "feasibility_cost": feasibility_cost_value,
                "task_state": task_state,
                "task_cost_breakdown": task_breakdown,
                "execution_cost": float(getattr(corr, "execution_cost", 0.0)),
                "nonholonomic_execution_profile": dict(getattr(
                    corr, "nonholonomic_execution_profile", {}) or {}),
                "diff_drive_execution_cost": float(getattr(
                    corr, "diff_drive_execution_cost", 0.0)),
                "initial_heading_error": float(getattr(
                    corr, "initial_heading_error", 0.0)),
                "monotonic_regression": float(getattr(
                    corr, "monotonic_regression", 0.0)),
                "nonmonotonic_fraction": float(getattr(
                    corr, "nonmonotonic_fraction", 0.0)),
                "heading_oscillation": float(getattr(
                    corr, "heading_oscillation", 0.0)),
                "diff_drive_reference_repaired": bool(getattr(
                    corr, "diff_drive_reference_repaired", False)),
                "diff_drive_reference_source": str(getattr(
                    corr, "diff_drive_reference_source", "")),
                "risk_norm": float(getattr(corr, "risk_norm", 0.0)),
                "length_norm": float(getattr(corr, "length_norm", 0.0)),
                "smooth_norm": float(getattr(corr, "smooth_norm", 0.0)),
                "task_norm": float(getattr(corr, "task_norm", 0.0)),
                "execution_norm": float(getattr(corr, "execution_norm", 0.0)),
                "topology_value": float(getattr(corr, "topology_value", 0.0)),
                "topology_diversity": float(getattr(corr, "topology_diversity", 0.0)),
                "total_score": float(getattr(corr, "total_score", getattr(corr, "cost", 0.0))),
                "total_cost": float(getattr(corr, "total_cost", getattr(corr, "cost", 0.0))),
                "path_length": float(getattr(corr, "path_length", 0.0)),
                "risk_per_meter": float(getattr(corr, "risk_per_meter", getattr(corr, "mean_phi_on_path", 0.0))),
                "mean_phi_on_path": float(getattr(corr, "mean_phi_on_path", 0.0)),
                "max_phi_on_path": float(getattr(corr, "max_phi_on_path", 0.0)),
                "candidate_status": str(getattr(corr, "candidate_status", "feasible")),
                "manifold_feasible": bool(getattr(corr, "manifold_feasible", True)),
                "candidate_manifold_valid": bool(getattr(
                    corr, "candidate_manifold_valid",
                    getattr(corr, "manifold_feasible", True))),
                "candidate_tube_valid": bool(getattr(corr, "candidate_tube_valid", True)),
                "tube_valid": bool(getattr(
                    corr, "tube_valid", getattr(corr, "candidate_tube_valid", True))),
                "trajectory_min_clearance": float(getattr(
                    corr, "trajectory_min_clearance",
                    getattr(corr, "min_clearance", 0.0))),
                "trajectory_max_risk": float(getattr(
                    corr, "trajectory_max_risk",
                    getattr(corr, "max_phi_on_path", 0.0))),
                "min_clearance": float(getattr(
                    corr, "trajectory_min_clearance",
                    getattr(corr, "min_clearance", 0.0))),
                "max_risk": float(getattr(
                    corr, "trajectory_max_risk",
                    getattr(corr, "max_phi_on_path", 0.0))),
                "required_clearance": float(getattr(corr, "required_clearance", 0.0)),
                "planning_clearance_margin": float(getattr(
                    corr, "planning_clearance_margin",
                    dbg.get("planning_clearance_margin", 0.0))),
                "recovery_mpc_tracking_margin": float(getattr(
                    corr, "recovery_mpc_tracking_margin",
                    dbg.get("recovery_mpc_tracking_margin", 0.0))),
                "topology_corridor_recovery_used": bool(getattr(
                    corr, "topology_recovery_used",
                    getattr(corr, "recovery_used", False))),
                "candidate_recovery_mode": str(getattr(corr, "candidate_recovery_mode", "")),
                "candidate_recovered": bool(getattr(
                    corr, "candidate_recovered",
                    getattr(corr, "topology_recovery_used",
                            getattr(corr, "recovery_used", False)))),
                "pre_refinement_clearance": float(getattr(
                    corr, "pre_refinement_clearance", 0.0)),
                "post_refinement_clearance": float(getattr(
                    corr, "post_refinement_clearance", 0.0)),
                "refinement_tube_valid": bool(getattr(
                    corr, "refinement_tube_valid", False)),
                "refinement_result": dict(getattr(
                    corr, "refinement_output", {}) or {}),
                "refinement_trace": list(getattr(
                    corr, "refinement_trace", [])),
                "before_clearance": float(getattr(corr, "before_clearance", 0.0)),
                "after_clearance": float(getattr(
                    corr, "after_clearance",
                    getattr(corr, "trajectory_min_clearance",
                            getattr(corr, "min_clearance", 0.0)))),
                "recovery_success": bool(getattr(
                    corr, "recovery_success",
                    getattr(corr, "manifold_feasible", False))),
                "candidate_recovery_iterations": int(getattr(
                    corr, "candidate_recovery_iterations",
                    1 if bool(getattr(corr, "topology_recovery_used",
                                      getattr(corr, "recovery_used", False))) else 0)),
                "adaptive_corridor_width": bool(getattr(
                    corr, "adaptive_corridor_width",
                    bool(getattr(corr, "corridor_width_profile", [])))),
                "risk_gain": float(getattr(corr, "risk_gain", 0.0)),
                "mean_distance_to_baseline": float(getattr(corr, "mean_distance_to_baseline", 0.0)),
                "max_lateral_offset": float(getattr(corr, "max_lateral_offset", 0.0)),
                "selected": bool(cid and cid == selected_id),
                "execution_corridor_id": selected_id if cid and cid == selected_id else "",
                "reject_reason": str(getattr(corr, "reject_reason", "")),
                "morse_node_ids": list(getattr(corr, "morse_node_ids", [])),
                "morse_node_types": list(getattr(corr, "morse_node_types", [])),
                "morse_induced": bool(getattr(corr, "morse_induced", False)),
            }
            rows.append(row)
        if rows:
            dbg["candidate_corridors"] = rows
            dbg["candidate_after_filter"] = rows
            dbg["candidate_after_top_k"] = rows
            dbg["candidate_after_filter_count"] = len(rows)
            dbg["candidate_after_top_k_count"] = len(rows)
            dbg["num_candidate_corridors"] = len(rows)
        else:
            existing_candidates = list(
                dbg.get("candidate_corridors") or
                dbg.get("candidate_before_filter") or [])
            dbg["candidate_corridors"] = existing_candidates
            dbg["candidate_after_filter"] = list(
                dbg.get("candidate_after_filter") or [])
            dbg["candidate_after_top_k"] = list(
                dbg.get("candidate_after_top_k") or [])
            dbg["candidate_after_filter_count"] = int(
                dbg.get("candidate_after_filter_count", 0) or 0)
            dbg["candidate_after_top_k_count"] = int(
                dbg.get("candidate_after_top_k_count", 0) or 0)
            dbg["num_candidate_corridors"] = int(
                dbg.get("num_candidate_corridors", 0) or 0)
        dbg["risk_field_used"] = 1
        dbg["manifold_used"] = 1
        dbg["morse_used"] = 1
        dbg["topology_graph_used"] = 1
        dbg["candidate_corridor_used"] = 1 if rows else 0
        dbg["candidate_ranking_used"] = 1 if rows and selected_id else 0
        recovery_used = any(bool(row.get("topology_corridor_recovery_used", False))
                            for row in rows)
        dbg["fallback_used"] = 0
        dbg["topology_corridor_recovery_used"] = 1 if recovery_used else int(
            dbg.get("topology_corridor_recovery_used", 0) or 0)
        dbg["mpc_used"] = 1 if selected is not None else 0
        dbg["mpc_reference_source"] = str(getattr(
            selected, "final_reference_source", "")) if selected is not None else ""
        if not dbg["mpc_reference_source"]:
            dbg["mpc_reference_source"] = "refined"
        dbg["final_path_source"] = "Morse->Candidate->Ranking->Refinement->MPC"
        dbg["adp_role"] = (
            "ranking_modifier"
            if any(float(getattr(c, "rank_base", 0)) != float(getattr(c, "rank_total", 0))
                   for c in list(corrs or []))
            else ("evaluation_only" if self.adp_enabled else "disabled"))
        if selected is not None:
            dbg["selected_corridor_label"] = selected_id
            dbg["selected_corridor_id"] = selected_id
            dbg["execution_corridor_id"] = selected_id
            dbg["selected_refinement_used"] = int(getattr(selected, "refinement_used", 0))
            dbg["selected_refined_path_length"] = float(getattr(
                selected, "refined_path_length", 0.0))
            dbg["selected_raw_waypoints_count"] = int(len(np.asarray(
                getattr(selected, "raw_topology_waypoints",
                        getattr(selected, "topology_ordered_waypoints", [])),
                float)))
            dbg["selected_refined_waypoints_count"] = int(len(np.asarray(
                getattr(selected, "refined_waypoints", []), float)))
            dbg["pre_refinement_clearance"] = float(getattr(
                selected, "pre_refinement_clearance", 0.0))
            dbg["post_refinement_clearance"] = float(getattr(
                selected, "post_refinement_clearance", 0.0))
            dbg["refinement_success"] = bool(getattr(
                selected, "refinement_success",
                int(getattr(selected, "refinement_used", 0)) == 1))
            dbg["refinement_fallback"] = bool(getattr(
                selected, "refinement_fallback", False))
            dbg["refinement_tube_valid"] = bool(getattr(
                selected, "refinement_tube_valid", False))
            dbg["reference_source"] = str(getattr(
                selected, "final_reference_source", dbg["mpc_reference_source"]))
            dbg["mpc_reference_source"] = dbg["reference_source"]
            dbg["reference_path_count"] = int(len(np.asarray(
                getattr(selected, "refined_waypoints",
                        getattr(selected, "centerline", [])), float)))
            dbg["selected_critical_point_association_used"] = int(bool(
                getattr(selected, "critical_point_association", {}) or {}))
            dbg["selected_topology_class"] = str(getattr(selected, "topology_class", ""))
            dbg["selected_topology_route_class"] = str(getattr(
                selected, "topology_route_class",
                getattr(selected, "topology_class", "")))
            dbg["selected_task_semantic_class"] = str(getattr(
                selected, "task_semantic_class", ""))
            dbg["selected_topology_diversity"] = float(getattr(selected, "topology_diversity", 0.0))
            dbg["selected_diff_drive_execution_cost"] = float(getattr(
                selected, "diff_drive_execution_cost", 0.0))
            dbg["selected_initial_heading_error"] = float(getattr(
                selected, "initial_heading_error", 0.0))
            dbg["selected_monotonic_regression"] = float(getattr(
                selected, "monotonic_regression", 0.0))
            dbg["selected_nonmonotonic_fraction"] = float(getattr(
                selected, "nonmonotonic_fraction", 0.0))
            dbg["selected_heading_oscillation"] = float(getattr(
                selected, "heading_oscillation", 0.0))
            dbg["selected_diff_drive_reference_repaired"] = int(bool(getattr(
                selected, "diff_drive_reference_repaired", False)))
            dbg["selected_diff_drive_reference_source"] = str(getattr(
                selected, "diff_drive_reference_source", ""))
            dbg["candidate_selection_status"] = str(getattr(
                selected, "candidate_status", dbg.get(
                    "candidate_selection_status", "feasible")))
            dbg["route_source"] = str(getattr(
                selected, "route_source", dbg.get(
                    "route_source", "morse_topology")))
            dbg["candidate_source"] = str(getattr(
                selected, "candidate_source", dbg["route_source"]))
            dbg["selected_candidate_source"] = dbg["candidate_source"]
            dbg["selected_route_source"] = dbg["route_source"]
            dbg["route_generation_level"] = str(getattr(
                selected, "route_generation_level",
                dbg.get("route_generation_level", "")))
            if bool(getattr(selected, "topology_recovery_used",
                            getattr(selected, "recovery_used", False))):
                dbg["candidate_selection_mode"] = (
                    "topology_corridor_feasibility_recovery")
            dbg["candidate_min_clearance"] = float(getattr(
                selected, "trajectory_min_clearance",
                dbg.get("candidate_min_clearance", 0.0)))
            dbg["candidate_max_risk"] = float(getattr(
                selected, "trajectory_max_risk",
                dbg.get("candidate_max_risk", 0.0)))
            dbg["selected_min_clearance"] = float(
                dbg["candidate_min_clearance"])
            dbg["candidate_manifold_feasible"] = bool(getattr(
                selected, "manifold_feasible",
                dbg.get("candidate_manifold_feasible", True)))
            dbg["candidate_manifold_valid"] = bool(getattr(
                selected, "candidate_manifold_valid",
                dbg.get("candidate_manifold_valid",
                        dbg["candidate_manifold_feasible"])))
            dbg["candidate_tube_valid"] = bool(getattr(
                selected, "candidate_tube_valid",
                dbg.get("candidate_tube_valid", True)))
            dbg["planning_clearance_margin"] = float(getattr(
                selected, "planning_clearance_margin",
                dbg.get("planning_clearance_margin", 0.0)))
            for row in rows:
                if str(row.get("corridor_id", "")) == selected_id:
                    dbg["selected_rank"] = int(row.get("rank", 0))
                    break
        self.manifold.last_topology_debug = dbg

    def _corridor_side(self, corridor):
        pts = np.asarray(getattr(corridor, "waypoints", []), float)
        if len(pts) == 0:
            return 0
        start = np.asarray(self.state[:2], float)
        goal = np.asarray(self.goal[:2], float)
        axis = goal - start
        if float(np.linalg.norm(axis)) <= 1e-9:
            return 0
        vals = [
            float(axis[0] * (p[1] - start[1]) -
                  axis[1] * (p[0] - start[0]))
            for p in pts[:, :2]]
        mean = float(np.mean(vals)) if vals else 0.0
        if abs(mean) < 1e-9:
            return 0
        return 1 if mean > 0.0 else -1

    def _select_wheelchair_corridor(self, corrs):
        corrs = list(corrs or [])
        if not corrs:
            return None
        side_name = self.topology_preferred_side
        wanted = 1 if side_name == "left" else -1 if side_name == "right" else 0
        if wanted:
            sided = [c for c in corrs if self._corridor_side(c) == wanted]
            if sided:
                corrs = sided
        def key(c):
            return (
                float(getattr(c, "total_score", getattr(c, "cost", 0.0))),
                float(getattr(c, "mean_phi_on_path", 0.0)),
                float(getattr(c, "max_phi_on_path", 0.0)),
                float(getattr(c, "path_length", 0.0)),
                float(getattr(c, "max_curvature", 0.0)),
            )
        return sorted(corrs, key=key)[0]

    def _rescore_executable_corridors(self, corrs):
        raw = dict(self.topology_corridor_score_weights or {})
        def pick(name, default, *aliases):
            for key in (name,) + aliases:
                if key in raw:
                    return float(raw[key])
                prefixed = "w_" + key
                if prefixed in raw:
                    return float(raw[prefixed])
            return float(default)
        weights = {
            "risk": pick("risk", 4.0),
            "max_risk_internal": pick("max_risk", 2.0),
            "length": pick("length", 1.0),
            "task": pick("task", 2.0),
            "smooth": pick("smooth", 2.0),
            "execution": pick("execution", 1.0, "exec"),
            "topology": pick("topology", 0.5),
            "diversity": pick("diversity", 0.5),
        }
        rescored = []
        for corr in list(corrs or []):
            if int(getattr(corr, "refinement_used", 0)) != 1:
                corr.reject_reason = "refinement_required_for_stsm"
                continue
            path_length = float(getattr(corr, "path_length", 0.0))
            risk_cost = (
                path_length * float(getattr(corr, "mean_phi_on_path", 0.0)) +
                weights["max_risk_internal"] *
                float(getattr(corr, "max_phi_on_path", 0.0)))
            distance_cost = path_length
            smooth_cost = float(getattr(corr, "tracking_cost", 0.0))
            motion_cost = float(getattr(
                corr, "motion_cost", getattr(corr, "execution_cost", 0.0)))
            execution_cost = float(getattr(corr, "execution_cost", motion_cost))
            curvature_cost = float(getattr(corr, "max_curvature", 0.0))
            topology_cost = float(getattr(corr, "topology_cost", 0.0))
            task_cost = float(getattr(corr, "task_cost", 0.0))
            topology_value = float(getattr(corr, "topology_value", 0.0))
            topology_diversity = float(getattr(corr, "topology_diversity", 0.0))
            corr.risk_cost = float(risk_cost)
            corr.distance_cost = float(distance_cost)
            corr.length_cost = float(distance_cost)
            corr.smooth_cost = float(smooth_cost)
            corr.motion_cost = float(motion_cost)
            corr.execution_cost = float(execution_cost)
            corr.curvature_cost = float(curvature_cost)
            corr.topology_cost = float(topology_cost)
            rescored.append(corr)
        def normalize(name, attr):
            vals = np.asarray([float(getattr(c, name, 0.0)) for c in rescored], float)
            if vals.size == 0:
                return
            lo = float(np.min(vals))
            hi = float(np.max(vals))
            norm = np.zeros_like(vals) if hi - lo <= 1e-9 else (vals - lo) / (hi - lo)
            for corr, val in zip(rescored, norm):
                setattr(corr, attr, float(val))
        normalize("risk_cost", "risk_norm")
        normalize("length_cost", "length_norm")
        normalize("smooth_cost", "smooth_norm")
        normalize("task_cost", "task_norm")
        normalize("execution_cost", "execution_norm")
        for corr in rescored:
            score = (
                weights["risk"] * float(getattr(corr, "risk_cost", 0.0)) +
                weights["length"] * float(getattr(corr, "length_cost", 0.0)) +
                weights["smooth"] * float(getattr(corr, "smooth_cost", 0.0)) +
                weights["task"] * float(getattr(corr, "task_cost", 0.0)) +
                weights["execution"] * float(getattr(
                    corr, "execution_cost", 0.0)) -
                weights["topology"] * float(getattr(
                    corr, "topology_value", 0.0)) -
                weights["diversity"] * float(getattr(
                    corr, "topology_diversity", 0.0)))
            corr.total_cost = float(score)
            corr.total_score = float(score)
            corr.base_cost = float(score)
            corr.cost = float(score)
            breakdown = dict(getattr(corr, "candidate_cost_breakdown", {}) or {})
            breakdown.update({
                "risk_cost": float(getattr(corr, "risk_cost", 0.0)),
                "risk_cost_term": float(weights["risk"] * float(getattr(
                    corr, "risk_cost", 0.0))),
                "length_cost": float(getattr(corr, "length_cost", 0.0)),
                "length_cost_term": float(weights["length"] * float(getattr(
                    corr, "length_cost", 0.0))),
                "smoothness_cost": float(getattr(corr, "smooth_cost", 0.0)),
                "smoothness_cost_term": float(weights["smooth"] * float(getattr(
                    corr, "smooth_cost", 0.0))),
                "task_cost": float(getattr(corr, "task_cost", 0.0)),
                "task_cost_term": float(weights["task"] * float(getattr(
                    corr, "task_cost", 0.0))),
                "execution_cost": float(getattr(corr, "execution_cost", 0.0)),
                "execution_cost_term": float(weights["execution"] * float(
                    getattr(corr, "execution_cost", 0.0))),
                "mpc_execution_cost_in_score": True,
                "topology_value": float(getattr(corr, "topology_value", 0.0)),
                "topology_value_term": float(-weights["topology"] * float(
                    getattr(corr, "topology_value", 0.0))),
                "topology_diversity": float(getattr(
                    corr, "topology_diversity", 0.0)),
                "topology_diversity_term": float(-weights["diversity"] * float(
                    getattr(corr, "topology_diversity", 0.0))),
                "ranking_score": float(score),
                "candidate_cost": float(score),
            })
            corr.candidate_cost = float(score)
            corr.candidate_cost_breakdown = breakdown
        rescored.sort(key=lambda c: float(getattr(c, "cost", 0.0)))
        for rank, corr in enumerate(rescored):
            corr.rank_total = int(rank)
        return rescored

    def _polyline_samples(self, points, samples_per_segment=16):
        points = [np.asarray(p, float)[:3] for p in points]
        out = []
        for a, b in zip(points[:-1], points[1:]):
            for idx in range(int(samples_per_segment) + 1):
                if out and idx == 0:
                    continue
                alpha = float(idx) / float(max(int(samples_per_segment), 1))
                out.append(a + alpha * (b - a))
        return out

    def _wheelchair_recovery_safety(self, start3, saddle3, goal3):
        footprint_values = []
        max_footprint_phi = 0.0
        max_center_phi = 0.0
        for p in self._polyline_samples([start3, saddle3, goal3]):
            yaw = self.state[2]
            center_phi = float(self.field.phi_s(
                np.array([p[0], p[1], 0.0], float)))
            max_center_phi = max(max_center_phi, center_phi)
            if center_phi > float(self.recovery_max_center_phi):
                return False, "recovery_center_risk_over_warn", center_phi
            summary = pose_interest_risk(
                self.field, np.array([p[0], p[1], yaw], float),
                local_points=self.wc_local_points,
                labels=self.wc_ip_labels)
            hit, _label, _anchor, reason = forbidden_anchor_hit(
                self.field, summary.get("labels", []),
                summary.get("points", []))
            if hit:
                return False, reason, float("inf")
            phi = float(summary.get("phi_max", 0.0))
            footprint_values.append(phi)
            max_footprint_phi = max(max_footprint_phi, phi)
            if phi > float(self.recovery_max_footprint_phi):
                return False, "recovery_footprint_risk_over_stop", phi
            if phi >= float(self.footprint_gate.rho_stop):
                return False, "risk_stop", phi
        mean_footprint_phi = float(np.mean(footprint_values)) if footprint_values else 0.0
        if mean_footprint_phi > float(self.recovery_max_mean_footprint_phi):
            return False, "recovery_mean_footprint_risk_high", mean_footprint_phi
        return True, "", float(max_footprint_phi)

    def _wheelchair_morse_recovery_corridor(self, start3, goal3):
        dbg = getattr(self.manifold, "last_topology_debug", {}) or {}
        nodes = list(dbg.get("nodes", []))
        saddles = [n for n in nodes if getattr(n, "kind", "") == "saddle"]
        if not saddles:
            critical = dbg.get("critical", {}) or {}
            saddles = list(critical.get("saddles", []) or [])
        if not saddles:
            chain = dbg.get("critical_chain", {}) or {}
            saddles = [
                item for item in list(chain.get("saddles", []) or [])
                if str(item.get("status", "")) in ("filtered", "selected")
            ]
        if not saddles:
            dbg = dict(dbg)
            dbg["topology_recovery_used"] = 0
            dbg["topology_corridor_recovery_used"] = 0
            dbg["topology_recovery_reject_reason"] = "no_recoverable_saddle_record"
            self.manifold.last_topology_debug = dbg
            rospy.logwarn(
                "[wc][topology] recovery unavailable: no recoverable saddle record "
                "(critical_keys=%s chain_keys=%s)",
                ",".join(sorted((dbg.get("critical", {}) or {}).keys())),
                ",".join(sorted((dbg.get("critical_chain", {}) or {}).keys())))
            return None
        start2 = np.asarray(start3, float)[:2]
        goal2 = np.asarray(goal3, float)[:2]
        axis = goal2 - start2
        axis_len2 = float(np.dot(axis, axis))

        def get_saddle_point(node):
            if isinstance(node, dict):
                return np.asarray(node.get("point", node.get("p2", start3)), float)[:3]
            return np.asarray(getattr(node, "point", start3), float)[:3]

        def saddle_id(node):
            if isinstance(node, dict):
                return str(node.get("id", "saddle"))
            return str(getattr(node, "id", "saddle"))

        def repaired_path_via(point):
            first, first_info = self._baseline_grid_astar(
                start3, point, radius=0.45, avoid_direct=False)
            second, second_info = self._baseline_grid_astar(
                point, goal3, radius=0.45, avoid_direct=False)
            if first is None or second is None:
                return None, "astar_repair_failed:%s:%s" % (
                    str(first_info.get("reason", "")),
                    str(second_info.get("reason", "")))
            first = np.asarray(first, float)
            second = np.asarray(second, float)
            if len(first) == 0 or len(second) == 0:
                return None, "astar_repair_empty"
            return np.vstack([first, second[1:]]), ""

        def segment_footprint_safe(a, b):
            for p in self._polyline_samples([a, b], samples_per_segment=12):
                pose = np.array([p[0], p[1], self.state[2]], float)
                center_phi = float(self.field.phi_s(
                    np.array([p[0], p[1], 0.0], float)))
                if center_phi > float(self.recovery_max_center_phi):
                    return False
                summary = pose_interest_risk(
                    self.field, pose,
                    local_points=self.wc_local_points,
                    labels=self.wc_ip_labels)
                hit, _label, _anchor, _reason = forbidden_anchor_hit(
                    self.field, summary.get("labels", []),
                    summary.get("points", []))
                if hit:
                    return False
                phi = float(summary.get("phi_max", 0.0))
                if phi > float(self.recovery_max_footprint_phi):
                    return False
            return True

        def compress_piece(points):
            pts = np.asarray(points, float)
            if len(pts) <= 2:
                return pts
            out = [pts[0]]
            idx = 0
            while idx < len(pts) - 1:
                nxt = idx + 1
                for cand in range(len(pts) - 1, idx, -1):
                    if segment_footprint_safe(pts[idx], pts[cand]):
                        nxt = cand
                        break
                out.append(pts[nxt])
                idx = nxt
            return np.asarray(out, float)

        def compress_via_saddle(path, point):
            pts = np.asarray(path, float)
            if len(pts) <= 2:
                return pts
            d = np.linalg.norm(pts[:, :2] - point[:2][None, :], axis=1)
            pivot = int(np.argmin(d))
            left = compress_piece(pts[:pivot + 1])
            right = compress_piece(pts[pivot:])
            if len(left) == 0:
                return right
            if len(right) == 0:
                return left
            return np.vstack([left, right[1:]])

        def recovery_path_safety(path):
            probe = Corridor(np.asarray(path, float), radius=0.45,
                             label="morse_recovery_probe")
            stats = self._corridor_footprint_stats(probe)
            if stats["forbidden"]:
                return False, stats.get("forbidden_reason", "forbidden"), stats
            risk_limit = float(_effective(
                self.topology_candidate_max_risk,
                self.footprint_gate.rho_stop))
            if stats["max_footprint_phi"] > risk_limit:
                return False, "recovery_footprint_risk_over_threshold", stats
            if stats["max_center_phi"] > risk_limit:
                return False, "recovery_center_risk_over_threshold", stats
            return True, "", stats

        def offset_score(node):
            point = get_saddle_point(node)
            rel = point[:2] - start2
            if axis_len2 <= 1e-12:
                progress = 0.0
                offset = float(np.linalg.norm(rel))
            else:
                progress = np.clip(float(np.dot(rel, axis)) / axis_len2, 0.0, 1.0)
                nearest = start2 + progress * axis
                offset = float(np.linalg.norm(point[:2] - nearest))
            return (offset, -abs(progress - 0.5))

        safe = []
        rejected_reason = "no_safe_saddle"
        for saddle in saddles:
            point = get_saddle_point(saddle)
            ok, reason, max_phi = self._wheelchair_recovery_safety(
                start3, point, goal3)
            if ok:
                safe.append((saddle, np.asarray([start3, point, goal3], float),
                             {"max_center_phi": float(max_phi),
                              "max_footprint_phi": float(max_phi),
                              "mean_footprint_phi": float(max_phi)}))
                continue
            repaired, repair_reason = repaired_path_via(point)
            if repaired is None:
                rejected_reason = repair_reason or reason
                continue
            repaired = compress_via_saddle(repaired, point)
            repair_ok, repair_reject, stats = recovery_path_safety(repaired)
            if repair_ok:
                safe.append((saddle, repaired, stats))
            else:
                rejected_reason = repair_reject or reason
        if not safe:
            dbg = dict(dbg)
            dbg["topology_recovery_used"] = 0
            dbg["topology_recovery_reject_reason"] = rejected_reason
            dbg["topology_disconnect_reason"] = "no_safe_wheelchair_morse_recovery"
            self.manifold.last_topology_debug = dbg
            rospy.logwarn(
                "[wc][topology] recovery rejected before hard feasibility: %s",
                rejected_reason)
            return None
        saddle, recovered_path, recovery_stats = max(
            safe, key=lambda item: offset_score(item[0]))
        saddle_point = get_saddle_point(saddle)
        corr = Corridor(
            np.asarray(recovered_path, float),
            radius=0.4,
            label="morse_recovery_saddle",
            cost=0.0)
        corr.corridor_id = "wheelchair_morse_recovery_c0001"
        corr.source = "morse_recovery"
        recovered_saddle_id = saddle_id(saddle)
        corr.topology_nodes = ["start", recovered_saddle_id, "goal"]
        corr.node_sequence = list(corr.topology_nodes)
        corr.topology_kinds = ["saddle"]
        corr.node_type_sequence = ["start", "saddle", "goal"]
        corr.morse_node_ids = [recovered_saddle_id]
        corr.morse_node_types = ["saddle"]
        corr.morse_induced = True
        corr.morse_forced = 1
        corr.rank_base = 0
        corr.rank_total = 0
        corr.path_length = self._path_length(corr.waypoints)
        corr.mean_phi_on_path = float(recovery_stats.get("mean_center_phi", 0.0))
        corr.max_phi_on_path = float(recovery_stats.get("max_center_phi", 0.0))
        max_phi = float(recovery_stats.get(
            "max_footprint_phi", corr.max_phi_on_path))
        corr.max_curvature = 0.0
        corr.execution_cost = 0.0
        corr.adp_cost = 0.0
        corr.risk_cost = float(corr.path_length * corr.mean_phi_on_path +
                               2.0 * corr.max_phi_on_path)
        corr.topology_cost = -1.0
        corr.distance_cost = float(corr.path_length)
        corr.smooth_cost = 0.0
        corr.curvature_cost = 0.0
        corr.topology_value = 1.0
        corr.cost = 0.0
        corr.base_cost = 0.0
        minimum_clearance = float(_effective(
            self.topology_hard_clearance,
            self.topology_profile_defaults.get("hard_clearance", 0.10)))
        minimum_clearance = max(minimum_clearance, 0.10)
        planning_margin = float(dbg.get(
            "planning_clearance_margin",
            self.topology_profile_defaults.get(
                "planning_clearance_margin", 0.03)))
        recovery_route = {
            "candidate_id": corr.corridor_id,
            "corridor_id": corr.corridor_id,
            "centerline": np.asarray(corr.waypoints, float).tolist(),
            "waypoints": np.asarray(corr.waypoints, float).tolist(),
            "radius": float(corr.radius),
            "candidate_generation_role": "topology_corridor_recovery",
            "node_sequence": list(corr.node_sequence),
            "node_type_sequence": list(corr.node_type_sequence),
        }
        recovery_constraint = {
            "minimum_clearance": minimum_clearance,
            "min_clearance": minimum_clearance,
            "risk_threshold": float(_effective(
                self.topology_candidate_max_risk,
                self.footprint_gate.rho_stop)),
            "planning_clearance_margin": planning_margin,
            "risk_field": self.field,
        }
        recovery_padding = max(0.03, float(self.recovery_mpc_tracking_margin))
        recovered_route, recovery_status = recover_candidate_corridor_feasibility(
            recovery_route, recovery_constraint, risk_field=self.field,
            clearance_padding=recovery_padding)
        if recovered_route is None or not bool(recovery_status.get("feasible", False)):
            dbg = dict(dbg)
            dbg["topology_recovery_used"] = 0
            dbg["topology_corridor_recovery_used"] = 0
            dbg["topology_recovery_reject_reason"] = str(
                recovery_status.get("failure_reason", "recovery_infeasible"))
            dbg["topology_disconnect_reason"] = "no_hard_feasible_wheelchair_morse_recovery"
            dbg["candidate_min_clearance"] = float(
                recovery_status.get("trajectory_min_clearance", 0.0))
            dbg["candidate_max_risk"] = float(
                recovery_status.get("trajectory_max_risk", 0.0))
            dbg["candidate_manifold_feasible"] = False
            dbg["candidate_manifold_valid"] = False
            dbg["candidate_tube_valid"] = False
            dbg["planning_clearance_margin"] = float(planning_margin)
            self.manifold.last_topology_debug = dbg
            rospy.logwarn(
                "[wc][topology] recovery rejected by hard feasibility reason=%s min_clearance=%.3f required=%.3f",
                dbg["topology_recovery_reject_reason"],
                float(recovery_status.get("trajectory_min_clearance", 0.0)),
                float(recovery_status.get(
                    "required_clearance", minimum_clearance + planning_margin)))
            return None
        for key, value in recovered_route.items():
            if key in ("centerline", "waypoints"):
                continue
            setattr(corr, key, value)
        corr.boundary = dict(recovered_route.get("boundary", {}) or {})
        corr.corridor_width_profile = list(
            recovered_route.get("corridor_width_profile", []) or [])
        corr.minimum_clearance = float(recovery_status.get(
            "minimum_clearance", minimum_clearance))
        corr.required_clearance = float(recovery_status.get(
            "required_clearance", minimum_clearance + planning_margin))
        corr.planning_clearance_margin = float(planning_margin)
        corr.recovery_mpc_tracking_margin = float(recovery_padding)
        corr.trajectory_min_clearance = float(recovery_status.get(
            "trajectory_min_clearance", 0.0))
        corr.trajectory_max_risk = float(recovery_status.get(
            "trajectory_max_risk", 0.0))
        corr.manifold_feasible = True
        corr.candidate_manifold_valid = True
        corr.candidate_tube_valid = True
        corr.tube_valid = True
        corr.candidate_recovered = True
        corr.before_clearance = 0.0
        corr.after_clearance = float(corr.trajectory_min_clearance)
        corr.recovery_success = True
        corr.candidate_recovery_iterations = 1
        corr.adaptive_corridor_width = True
        corr.candidate_status = "feasible"
        corr.failure_reason = ""
        dbg = dict(dbg)
        dbg["topology_recovery_used"] = 1
        dbg["topology_corridor_recovery_used"] = 1
        dbg["recovered_feasible_candidates"] = 1
        dbg["num_candidate_corridors"] = max(1, int(dbg.get("num_candidate_corridors", 0)))
        dbg["candidate_after_filter_count"] = 1
        dbg["candidate_after_top_k_count"] = 1
        dbg["total_candidates"] = max(1, int(dbg.get("total_candidates", 0)))
        dbg["feasible_candidates"] = 1
        dbg["filtered_candidates"] = int(dbg.get("filtered_candidates", 0))
        dbg["num_used_saddles"] = max(1, int(dbg.get("num_used_saddles", 0)))
        dbg["used_saddles"] = max(1, int(dbg.get("used_saddles", 0)))
        dbg["used_saddle_count"] = max(1, int(dbg.get("used_saddle_count", 0)))
        dbg["num_forced_critical_corridors"] = max(
            1, int(dbg.get("num_forced_critical_corridors", 0)))
        dbg["selected_corridor_label"] = corr.label
        dbg["selected_corridor_id"] = corr.corridor_id
        dbg["execution_corridor_id"] = corr.corridor_id
        dbg["selected_morse_induced"] = True
        dbg["selected_corridor_max_phi"] = float(max_phi)
        dbg["candidate_selection_status"] = "feasible"
        dbg["candidate_selection_mode"] = "topology_corridor_feasibility_recovery"
        dbg["candidate_min_clearance"] = float(corr.trajectory_min_clearance)
        dbg["candidate_max_risk"] = float(corr.trajectory_max_risk)
        dbg["selected_min_clearance"] = float(corr.trajectory_min_clearance)
        dbg["candidate_manifold_feasible"] = True
        dbg["candidate_manifold_valid"] = True
        dbg["candidate_tube_valid"] = True
        dbg["planning_clearance_margin"] = float(planning_margin)
        dbg["recovery_mpc_tracking_margin"] = float(recovery_padding)
        dbg["candidate_corridors"] = [{
            "corridor_id": corr.corridor_id,
            "label": corr.label,
            "source": corr.source,
            "candidate_generation_role": "topology_corridor_recovery",
            "node_sequence": list(corr.node_sequence),
            "node_type_sequence": list(corr.node_type_sequence),
            "path_length": float(corr.path_length),
            "risk_cost": float(corr.risk_cost),
            "topology_cost": float(corr.topology_cost),
            "selected": True,
            "execution_corridor_id": corr.corridor_id,
            "candidate_status": "feasible",
            "manifold_feasible": True,
            "candidate_manifold_valid": True,
            "candidate_tube_valid": True,
            "tube_valid": True,
            "trajectory_min_clearance": float(corr.trajectory_min_clearance),
            "trajectory_max_risk": float(corr.trajectory_max_risk),
            "min_clearance": float(corr.trajectory_min_clearance),
            "max_risk": float(corr.trajectory_max_risk),
            "required_clearance": float(corr.required_clearance),
            "planning_clearance_margin": float(planning_margin),
            "recovery_mpc_tracking_margin": float(recovery_padding),
            "topology_corridor_recovery_used": True,
            "candidate_recovery_mode": "topology_tube_expansion",
            "candidate_recovered": True,
            "before_clearance": 0.0,
            "after_clearance": float(corr.trajectory_min_clearance),
            "recovery_success": True,
            "candidate_recovery_iterations": 1,
            "adaptive_corridor_width": True,
            "morse_induced": True,
            "morse_node_ids": list(corr.morse_node_ids),
            "morse_node_types": list(corr.morse_node_types),
            "max_sampled_phi": float(max_phi),
        }]
        dbg["candidate_before_filter"] = list(dbg["candidate_corridors"])
        dbg["candidate_after_filter"] = list(dbg["candidate_corridors"])
        dbg["candidate_before_filter_count"] = 1
        dbg["candidate_after_filter_count"] = 1
        self.manifold.last_topology_debug = dbg
        rospy.logwarn(
            "[wc][topology] recovered Morse saddle corridor id=%s saddle=%s max_phi=%.3f",
            corr.corridor_id, corr.morse_node_ids[0], float(max_phi))
        return corr

    def _maybe_replan_corridor(self, corridor, now, reason, force=False,
                               deadline=None):
        is_topology = self._corridor_is_topological(corridor)
        if is_topology and not force:
            elapsed = (now - self.last_topology_replan_time).to_sec()
            if elapsed < self.topology_replan_min_interval:
                rospy.loginfo_throttle(
                    5.0,
                    "[wc][topology] skip replan reason=%s elapsed=%.1fs min=%.1fs; keep %s",
                    reason, elapsed, self.topology_replan_min_interval,
                    getattr(corridor, "label", ""))
                return corridor, False
        if deadline is not None:
            remaining = float((deadline - now).to_sec())
            required = max(
                float(self.replan_min_budget_s),
                float(self.last_corridor_plan_duration_s) *
                max(1.0, float(self.replan_budget_factor)))
            if remaining <= required:
                self.replan_deadline_skip_count += 1
                rospy.logwarn(
                    "[wc] skip blocking replan reason=%s remaining=%.1fs required=%.1fs; keep %s",
                    reason, remaining, required,
                    self._corridor_label(corridor, ""))
                return corridor, False
        try:
            rospy.loginfo("[wc] replanning corridor reason=%s current=%s",
                          reason, self._corridor_label(corridor, ""))
            new_corridor = self._plan_corridor()
            new_corridor = self._ensure_corridor_runtime_contract(
                new_corridor,
                fallback_id="wheelchair_replan_c%04d" % (
                    self.runtime_replan_fallback_count + 1),
                fallback_source="refinement")
            if new_corridor is None:
                raise RuntimeError("replan produced empty runtime corridor")
            if self._corridor_is_topological(new_corridor):
                self.last_topology_replan_time = now
            return new_corridor, True
        except Exception as exc:
            # A failed trial replan is diagnostic evidence, but it must not
            # replace the selected corridor/debug state used by execution.
            self.selected_corridor = corridor
            self.execution_corridor = corridor
            self._sync_selected_corridor_geometry(corridor)
            if self.last_valid_topology_debug:
                self.manifold.last_topology_debug = copy.deepcopy(
                    self.last_valid_topology_debug)
                self._publish_topology_info(
                    self._corridor_is_topological(corridor), False)
            self.selected_corridor_pub.publish(String(
                self._corridor_id(corridor, "wheelchair_selected")))
            if (self.experiment_mode == "paper" and not self.baseline and
                    not self.topology_fallback_enabled):
                rospy.logerr(
                    "[wc] replan failed closed in paper mode reason=%s current=%s: %s",
                    reason, self._corridor_label(corridor, ""), exc)
                return corridor, False
            fallback = self._runtime_replan_fallback_corridor(
                corridor, reason, exc)
            if fallback is not None:
                rospy.logwarn(
                    "[wc] replan failed reason=%s; using fallback %s after %s",
                    reason, self._corridor_id(fallback), exc)
                return fallback, True
            kept = self._ensure_corridor_runtime_contract(
                corridor,
                fallback_id="wheelchair_replan_keep_c%04d" % (
                    self.runtime_replan_fallback_count + 1),
                fallback_source="runtime_replan_fallback")
            rospy.logwarn("[wc] replan failed reason=%s; keep %s: %s",
                          reason, self._corridor_label(kept, ""), exc)
            return kept if kept is not None else corridor, False

    def _runtime_candidate_first_step_status(self, candidate):
        """Check whether a ranked topology candidate is executable now.

        This preflight uses the same predictive MPC contract as execution but
        does not publish a command.  Runtime switching is allowed only when the
        candidate can produce a live first step from the current measured
        wheelchair pose; otherwise the previous corridor remains active.
        """
        cid = self._corridor_id(candidate, "")
        status = {
            "corridor_id": cid,
            "runtime_switch_precheck": True,
            "accepted": False,
            "failure_reason": "",
            "solver_status": "not_run",
            "first_control": [0.0, 0.0],
            "objective_terms": {},
            "constraint_violation": {},
        }
        if self.state is None:
            status["failure_reason"] = "state_unavailable"
            return status
        try:
            ref = self._horizon_ref(candidate)
            _ti, _ci, _mi, topology_constraint_for_mpc = (
                build_mpc_constraint_inputs(
                    candidate, self.manifold, ref,
                    safe_threshold=float(self.manifold.rho),
                    minimum_clearance=0.10,
                    phase="navigation", robot_type="wheelchair",
                    manifold_constraint_mode=self.manifold_constraint_mode,
                    strict_stsm=bool(not self.baseline),
                    expected_corridor_id=cid))
            v, w = self.mpc.solve(
                self.state, ref, self.field,
                corridor=candidate, u_prev=self.u_prev,
                critic=self.adp_critic if self.adp_enabled else None,
                feature_builder=self.adp_features,
                lambda_adp_terminal=(
                    self.lambda_adp_terminal if self.adp_enabled else 0.0),
                goal=self.goal,
                gate_info={
                    "state": "runtime_switch_precheck",
                    "stop": False,
                    "rho_warn": self.gate.rho_warn,
                },
                interest_risk={},
                use_adp_terminal=(
                    self.adp_enabled and self.mpc_use_adp_terminal),
                interest_constraints={
                    "enabled": bool(self.interest_enabled),
                    "local_points": self.wc_local_points,
                    "labels": self.wc_ip_labels,
                    "rho": self.footprint_gate.rho_stop,
                },
                topology_constraint=topology_constraint_for_mpc,
                predictive=True)
            objective = dict(self.mpc.last_objective_terms or {})
            solver_status = str(self.mpc.last_solver_status)
            first_step_live = bool(objective.get("first_step_live", False))
            heading_recovery_live = bool(objective.get(
                "heading_recovery_live", False))
            accepted = bool(
                not solver_status.startswith("safe_stop:") and
                (first_step_live or heading_recovery_live))
            status.update({
                "accepted": accepted,
                "solver_status": solver_status,
                "first_control": [float(v), float(w)],
                "objective_terms": objective,
                "constraint_violation": dict(
                    self.mpc.last_constraint_violation or {}),
                "first_step_live": first_step_live,
                "heading_recovery_live": heading_recovery_live,
                "sequence_progress": float(self.mpc.last_sequence_progress),
            })
            if not accepted:
                status["failure_reason"] = (
                    "first_step_not_executable:%s" % solver_status)
        except Exception as exc:
            status["failure_reason"] = "%s:%s" % (
                type(exc).__name__, str(exc)[:160])
        return status

    def _switch_to_ranked_topology_candidate(self, current_corridor, reason):
        """Switch within the already ranked Morse/topology candidate pool.

        This is intentionally not a direct/runtime fallback.  It is used when
        execution diagnostics prove that the currently selected topological
        corridor is non-progressive, while other hard-filtered and refined
        topology candidates from the same planning decision remain available.
        """
        if self.baseline or not self.topology_replan_on_no_progress:
            return current_corridor, False
        current_id = self._corridor_id(current_corridor, "")
        if current_id:
            self.runtime_rejected_topology_corridor_ids.add(current_id)
        switch_trials = []
        pool = [
            corr for corr in list(self.runtime_topology_candidate_pool or [])
            if corr is not None and self._corridor_is_topological(corr)
        ]
        if not pool:
            return current_corridor, False
        pool = sorted(pool, key=lambda corr: (
            int(getattr(corr, "rank_total",
                        getattr(corr, "rank_base", 999999))),
            float(getattr(corr, "total_score",
                          getattr(corr, "cost", 0.0))),
            self._corridor_id(corr, "")))
        for candidate in pool:
            cid = self._corridor_id(candidate, "")
            if not cid or cid == current_id:
                continue
            if cid in self.runtime_rejected_topology_corridor_ids:
                continue
            if not bool(getattr(candidate, "candidate_manifold_valid",
                                getattr(candidate, "manifold_feasible", True))):
                continue
            if not bool(getattr(candidate, "candidate_tube_valid",
                                getattr(candidate, "tube_valid", True))):
                continue
            ready = self._ensure_corridor_runtime_contract(
                candidate,
                fallback_id=cid,
                fallback_source=str(getattr(
                    candidate, "final_reference_source", "refinement") or
                    "refinement"))
            if ready is None:
                self.runtime_rejected_topology_corridor_ids.add(cid)
                continue
            precheck = self._runtime_candidate_first_step_status(ready)
            switch_trials.append(precheck)
            ready.runtime_switch_first_step_status = dict(precheck)
            if not bool(precheck.get("accepted", False)):
                breakdown = dict(getattr(
                    ready, "candidate_cost_breakdown", {}) or {})
                breakdown["runtime_switch_precheck"] = dict(precheck)
                breakdown["runtime_switch_reject_reason"] = str(
                    precheck.get("failure_reason", ""))
                ready.candidate_cost_breakdown = breakdown
                rospy.logwarn(
                    "[wc][topology] runtime switch reject %s reason=%s status=%s first_step=%s",
                    cid, precheck.get("failure_reason", ""),
                    precheck.get("solver_status", ""),
                    precheck.get("first_control", []))
                continue
            self.runtime_topology_candidate_switch_count += 1
            self.runtime_topology_candidate_switch_trials.extend(switch_trials)
            ready.runtime_topology_candidate_switch = True
            ready.runtime_topology_switch_reason = str(reason or "")
            ready.runtime_topology_previous_corridor_id = current_id
            ready.dynamic_replan_fallback = False
            self.selected_corridor = ready
            self.execution_corridor = ready
            self._sync_selected_corridor_geometry(ready)
            self._sync_runtime_topology_debug(
                self.runtime_topology_candidate_pool, ready)
            dbg = dict(getattr(self.manifold, "last_topology_debug", {}) or {})
            dbg["topology_runtime_candidate_switch_used"] = 1
            dbg["topology_runtime_candidate_switch_count"] = int(
                self.runtime_topology_candidate_switch_count)
            dbg["runtime_switch_reason"] = str(reason or "")
            dbg["runtime_switch_previous_corridor_id"] = current_id
            dbg["runtime_switch_selected_corridor_id"] = self._corridor_id(
                ready, "")
            dbg["runtime_switch_precheck_used"] = 1
            dbg["runtime_switch_precheck_trials"] = list(switch_trials)
            dbg["runtime_switch_selected_precheck"] = dict(precheck)
            dbg["fallback_used"] = 0
            dbg["topology_fallback_used"] = 0
            self.manifold.last_topology_debug = dbg
            self._publish_topology_info(True, False)
            self.selected_corridor_pub.publish(String(self._corridor_id(ready)))
            rospy.logwarn(
                "[wc][topology] runtime candidate switch reason=%s %s -> %s label=%s",
                reason, current_id, self._corridor_id(ready),
                self._corridor_label(ready, ""))
            return ready, True
        if switch_trials:
            self.runtime_topology_candidate_switch_trials.extend(switch_trials)
            dbg = dict(getattr(self.manifold, "last_topology_debug", {}) or {})
            dbg["runtime_switch_precheck_used"] = 1
            dbg["runtime_switch_precheck_trials"] = list(
                self.runtime_topology_candidate_switch_trials[-20:])
            dbg["runtime_switch_reject_all"] = 1
            dbg["runtime_switch_reason"] = str(reason or "")
            self.manifold.last_topology_debug = dbg
        return current_corridor, False

    def _runtime_replan_fallback_corridor(self, current_corridor,
                                          reason, original_error=None):
        if (self.experiment_mode == "paper" and not self.baseline and
                not self.topology_fallback_enabled):
            rospy.logerr(
                "[wc][replan_fallback] disabled in paper mode reason=%s error=%s",
                reason, original_error)
            return None
        if self.state is None:
            return None
        self.runtime_replan_fallback_count += 1
        start3 = np.array([self.state[0], self.state[1], 0.0], float)
        goal3 = np.array([self.goal[0], self.goal[1], 0.0], float)
        cid = "wheelchair_c_runtime_replan_%04d" % (
            self.runtime_replan_fallback_count)
        candidates = []
        try:
            safe = self._wheelchair_safe_fallback_corridor(
                start3, goal3,
                label="runtime_replan_fallback",
                corridor_id=cid,
                radius=0.45,
                use_adp=self.adp_enabled)
            if safe is not None:
                candidates.append(safe)
        except Exception as exc:
            rospy.logwarn("[wc][replan_fallback] safe fallback failed: %s", exc)
        try:
            direct = Corridor(np.asarray([start3, goal3], float), radius=0.45,
                              label="runtime_replan_direct")
            direct.corridor_id = cid + "_direct"
            direct.source = "runtime_replan_direct"
            direct.path_length = self._path_length(direct.waypoints)
            direct.mean_phi_on_path = 0.0
            direct.max_phi_on_path = 0.0
            direct.cost = float(direct.path_length)
            candidates.append(direct)
        except Exception:
            pass
        for candidate in candidates:
            candidate.dynamic_replan_fallback = True
            candidate.replan_failure_reason = str(reason or "")
            candidate.replan_original_error = str(original_error or "")
            candidate.refinement_used = int(getattr(candidate, "refinement_used", 0))
            candidate.refinement_success = bool(getattr(
                candidate, "refinement_success", False))
            candidate.refinement_fallback = True
            candidate.final_reference_source = "runtime_replan_fallback"
            candidate.node_sequence = list(getattr(
                candidate, "node_sequence", []) or ["start", "goal"])
            candidate.topology_nodes = list(getattr(
                candidate, "topology_nodes", []) or candidate.node_sequence)
            candidate.node_type_sequence = list(getattr(
                candidate, "node_type_sequence", []) or ["start", "goal"])
            ready = self._ensure_corridor_runtime_contract(
                candidate, fallback_id=cid,
                fallback_source="runtime_replan_fallback")
            if ready is not None:
                self.selected_corridor = ready
                self._sync_runtime_topology_debug([ready], ready)
                self._publish_topology_info(False, True)
                self.selected_corridor_pub.publish(String(
                    self._corridor_id(ready)))
                return ready
        return None

    def _plan_heuristic_corridor(self):
        start = self.state[:2]
        corrs = self.manifold.enumerate_corridors(
            np.array([start[0], start[1], 0.0]),
            np.array([self.goal[0], self.goal[1], 0.0]),
            self.bounds, radius=0.4,
            critic=self.adp_critic if self.adp_enabled else None,
            feature_builder=self.adp_features,
            lambda_adp=self.lambda_adp_corridor if self.adp_enabled else 0.0,
            feature_context={
                "yaw": self.state[2],
                "u": self.u_prev,
                "radius": 0.4,
                "adp_samples": 9,
                "gate_info": {
                    "state": "NORMAL",
                    "rho_warn": self.gate.rho_warn,
                    "stop": False,
                },
            })
        return corrs[0]

    def _horizon_ref(self, corridor):
        N = self.mpc.N
        self._sync_selected_corridor_geometry(corridor)
        _, d0 = corridor.project(self.state[:2])

        pts = []
        waypoints, _ = self._final_corridor_trajectory(corridor)
        if waypoints.ndim == 1 and waypoints.size:
            waypoints = waypoints.reshape((1, waypoints.shape[0]))
        if len(waypoints) == 1:
            pts = np.asarray([waypoints[0, :2]], float)
        else:
            pts = []
            for a, b in zip(waypoints[:-1], waypoints[1:]):
                for t in np.linspace(0, 1, 20):
                    pts.append((a + t * (b - a))[:2])
            pts = np.array(pts)
        if pts.size == 0:
            self._sync_selected_corridor_geometry(self.selected_corridor)
            waypoints, _ = self._final_corridor_trajectory(self.selected_corridor)
            if len(waypoints) > 0:
                pts = np.asarray(waypoints[:, :2], float)
        if pts.size == 0:
            dbg = dict(getattr(self.manifold, "last_topology_debug", {}) or {})
            dbg["planning_chain_failure"] = "reference_path_empty"
            dbg["reference_path_count"] = 0
            self.manifold.last_topology_debug = dbg
            raise RuntimeError("planning_failure: reference_path_empty")
        dists = np.linalg.norm(pts - self.state[:2], axis=1)
        i0 = int(np.argmin(dists))
        next_idx = min(i0 + 1, len(pts) - 1)
        corridor.reference_index = int(i0)
        corridor.reference_next_index = int(next_idx)
        corridor.reference_distance = float(dists[i0])
        corridor.reference_next_distance = float(
            np.linalg.norm(pts[next_idx] - self.state[:2]))
        ref = []
        for k in range(N):
            ref.append(pts[min(i0 + 1 + k * 2, len(pts) - 1)])
        ref = np.array(ref)
        if self.state is not None:
            dist_goal = float(np.linalg.norm(self.state[:2] - self.goal))
            if self._in_final_approach(dist_goal):
                goal2 = np.asarray(self.goal[:2], float)
                if len(ref) == 0:
                    ref = np.asarray([goal2], float)
                else:
                    ref[-1] = goal2
        self._record_mpc_reference(corridor, ref)
        self._record_baseline_reference_before_mpc(corridor, ref)
        return ref

    def _record_mpc_reference(self, corridor, ref):
        if self.baseline or corridor is None or ref is None:
            return
        cid = str(getattr(corridor, "corridor_id", getattr(corridor, "label", "")))
        source_raw = (
            str(getattr(corridor, "final_reference_source", "")) or
            ("refined" if int(getattr(corridor, "refinement_used", 0)) == 1
             else "candidate_fallback"))
        source = (
            "refinement" if source_raw in ("refined", "refined_waypoints") else
            "candidate" if source_raw == "candidate_fallback" else
            source_raw if source_raw else "fallback")
        solve_idx = int(self._mpc_reference_solve_index)
        self._mpc_reference_solve_index += 1
        for idx, p in enumerate(np.asarray(ref, float)):
            global_idx = len(self.mpc_reference_records)
            self.mpc_reference_records.append({
                "robot": "wheelchair",
                "corridor_id": cid,
                "reference_source": source,
                "phase": "navigation",
                "solve_index": solve_idx,
                "horizon_point_index": idx,
                "trajectory_point_index": global_idx,
                "timestamp_or_s_index": global_idx,
                "x": float(p[0]),
                "y": float(p[1]),
                "z": 0.0,
            })

    def _record_baseline_reference_before_mpc(self, corridor, ref):
        if not self.baseline or corridor is None or ref is None:
            return
        cid = str(getattr(corridor, "corridor_id", getattr(corridor, "label", "")))
        source = str(getattr(
            corridor, "planner_source",
            "direct_connection" if self.baseline_type == "direct"
            else getattr(corridor, "baseline_planner", "")))
        raw = np.asarray(getattr(corridor, "waypoints", []), float)
        raw_count = int(len(raw)) if raw.ndim >= 2 else 0
        solve_idx = int(self._baseline_reference_solve_index)
        self._baseline_reference_solve_index += 1
        for idx, p in enumerate(np.asarray(ref, float)):
            self.baseline_reference_records.append({
                "robot": "wheelchair",
                "variant": "baseline",
                "baseline_type": str(self.baseline_type),
                "planner_source": source,
                "raw_reference_source": source,
                "corridor_id": cid,
                "raw_waypoint_count": raw_count,
                "solve_index": solve_idx,
                "horizon_point_index": idx,
                "x": float(p[0]),
                "y": float(p[1]),
                "z": 0.0,
            })

    def _record_baseline_mpc_output(self, corridor, ref, v_mpc, w_mpc,
                                    v_final, w_final, gate, adp_scale,
                                    topology_constraint):
        if not self.baseline or corridor is None:
            return
        cid = str(getattr(corridor, "corridor_id", getattr(corridor, "label", "")))
        ref_arr = np.asarray(ref if ref is not None else [], float)
        if ref_arr.size == 0:
            ref_first = [0.0, 0.0]
            ref_last = [0.0, 0.0]
        else:
            if ref_arr.ndim == 1:
                ref_arr = ref_arr.reshape((1, ref_arr.shape[0]))
            ref_first = ref_arr[0, :2].tolist()
            ref_last = ref_arr[-1, :2].tolist()
        theta = float(self.state[2])
        pred = np.array([
            float(self.state[0]) + float(v_final) * np.cos(theta) * self.mpc.dt,
            float(self.state[1]) + float(v_final) * np.sin(theta) * self.mpc.dt,
        ])
        self.baseline_mpc_output_records.append({
            "robot": "wheelchair",
            "variant": "baseline",
            "baseline_type": str(self.baseline_type),
            "planner_source": str(getattr(
                corridor, "planner_source",
                "direct_connection" if self.baseline_type == "direct"
                else getattr(corridor, "baseline_planner", ""))),
            "corridor_id": cid,
            "step_index": len(self.baseline_mpc_output_records),
            "state_x": float(self.state[0]),
            "state_y": float(self.state[1]),
            "state_yaw": theta,
            "ref_first_x": float(ref_first[0]),
            "ref_first_y": float(ref_first[1]),
            "ref_last_x": float(ref_last[0]),
            "ref_last_y": float(ref_last[1]),
            "v_mpc": float(v_mpc),
            "omega_mpc": float(w_mpc),
            "gate_scale": float(getattr(gate, "scale", 1.0)),
            "adp_scale": float(adp_scale),
            "v_final": float(v_final),
            "omega_final": float(w_final),
            "pred_next_x": float(pred[0]),
            "pred_next_y": float(pred[1]),
            "mpc_solver_status": str(getattr(self.mpc, "last_solver_status", "")),
            "mpc_used": 1,
            "risk_cost_used": int(float(getattr(self.mpc, "lam_social", 0.0)) > 0.0),
            "constraint_used": int(bool(topology_constraint)),
            "topology_constraint_used": int(bool(
                (topology_constraint or {}).get("topology_constraint_used", False))),
            "corridor_constraint_used": int(bool(
                (topology_constraint or {}).get("corridor_constraint_used", False))),
            "manifold_constraint_used": int(bool(
                (topology_constraint or {}).get("manifold_constraint_used", False))),
        })

    def _write_mpc_reference_path(self):
        if not self.mpc_reference_out or self.baseline:
            return
        out_dir = os.path.dirname(self.mpc_reference_out)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        fields = [
            "robot", "corridor_id", "reference_source", "phase", "solve_index",
            "horizon_point_index", "trajectory_point_index",
            "timestamp_or_s_index", "x", "y", "z",
        ]
        with open(self.mpc_reference_out, "w") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in self.mpc_reference_records:
                writer.writerow({key: row.get(key, "") for key in fields})
        rospy.loginfo("[wc][trace] wrote MPC reference path %s rows=%d",
                      self.mpc_reference_out, len(self.mpc_reference_records))

    def _write_baseline_mpc_trace_files(self, out_dir):
        if not self.baseline:
            return
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        ref_fields = [
            "robot", "variant", "baseline_type", "planner_source",
            "raw_reference_source", "corridor_id", "raw_waypoint_count",
            "solve_index", "horizon_point_index", "x", "y", "z",
        ]
        ref_path = os.path.join(out_dir, "baseline_reference_before_mpc.csv")
        with open(ref_path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=ref_fields)
            writer.writeheader()
            for row in self.baseline_reference_records:
                writer.writerow({key: row.get(key, "") for key in ref_fields})
        out_fields = [
            "robot", "variant", "baseline_type", "planner_source",
            "corridor_id", "step_index", "state_x", "state_y", "state_yaw",
            "ref_first_x", "ref_first_y", "ref_last_x", "ref_last_y",
            "v_mpc", "omega_mpc", "gate_scale", "adp_scale",
            "v_final", "omega_final", "pred_next_x", "pred_next_y",
            "mpc_solver_status", "mpc_used", "risk_cost_used",
            "constraint_used", "topology_constraint_used",
            "corridor_constraint_used", "manifold_constraint_used",
        ]
        mpc_path = os.path.join(out_dir, "baseline_mpc_output.csv")
        with open(mpc_path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=out_fields)
            writer.writeheader()
            for row in self.baseline_mpc_output_records:
                writer.writerow({key: row.get(key, "") for key in out_fields})

    def _mpc_output_paths(self):
        base = os.path.dirname(self.mpc_reference_out or self.decision_trace_out)
        diag = self.mpc_diagnostics_out or os.path.join(
            base, "mpc_diagnostics.json")
        breakdown = self.mpc_cost_breakdown_out or os.path.join(
            base, "mpc_cost_breakdown.csv")
        return diag, breakdown

    def _write_failed_topology_diagnostics(self, failure_reason):
        if self.baseline:
            return
        base = os.path.dirname(
            self.decision_trace_out or self.mpc_reference_out or
            self.mpc_diagnostics_out or "")
        if not base:
            return
        debug = getattr(self.manifold, "last_topology_debug", {}) or {}
        write_failed_topology_diagnostics(
            base, "wheelchair", debug, failure_reason=failure_reason)

    def _write_mpc_diagnostics(self):
        if self.baseline:
            return
        corridor = self.execution_corridor or self.selected_corridor
        final_trajectory, final_source = self._sync_selected_corridor_geometry(corridor)
        cid = str(getattr(corridor, "corridor_id", getattr(corridor, "label", "")))
        valid_stsm_cid = bool(cid)
        if (not valid_stsm_cid or
                any(not self._valid_reference_source(
                        row.get("reference_source", ""))
                    for row in self.mpc_reference_records)):
            rospy.logerr(
                "[wc][mpc] refusing formal rolling MPC diagnostics: "
                "invalid corridor_id=%s or invalid reference source", cid)
            return
        ref = [dict(row) for row in self.mpc_reference_records]
        ref_points = [
            [row.get("x", 0.0), row.get("y", 0.0), row.get("z", 0.0)]
            for row in ref
        ]
        if not ref and len(final_trajectory) > 0:
            ref_points = final_trajectory.tolist()
            ref = [
                {
                    "x": float(p[0]),
                    "y": float(p[1]),
                    "z": float(p[2]) if len(p) > 2 else 0.0,
                    "phase": "navigation",
                }
                for p in ref_points
            ]
        if not ref:
            diag, _ = self._mpc_output_paths()
            out_dir = os.path.dirname(diag)
            if out_dir and not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            with open(diag, "w") as f:
                json.dump({
                    "final_status": "infeasible_reference_empty",
                    "final_mpc_status": "infeasible_reference_empty",
                    "mpc_feasibility_status": "infeasible_reference_empty",
                    "mpc_used": False,
                    "success": False,
                    "planning_chain_failure": "reference_path_empty",
                    "temporary_failure_reason": "reference_path_empty",
                    "final_failure_reason": "reference_path_empty",
                    "failure_reason": "reference_path_empty",
                    "reference_source": "fallback",
                    "reference_path_count": 0,
                    "corridor_centerline_count": int(len(final_trajectory)),
                    "tube_point_count": 0,
                    "task_mode": str(self.task_mode),
                    "task_weight": dict(self.task_weight),
                    "task_weight_used": True,
                }, f, indent=2, sort_keys=True)
            return
        max_diag_points = int(rospy.get_param(
            "~mpc_diagnostics_max_points", 160))
        if max_diag_points > 0 and len(ref) > max_diag_points:
            idx = np.linspace(0, len(ref) - 1, max_diag_points).astype(int)
            ref = [ref[int(i)] for i in idx]
            ref_points = [ref_points[int(i)] for i in idx]
        start_state = np.asarray(ref_points[0], float)
        if len(ref_points) > 1:
            p0 = np.asarray(ref_points[0], float)
            p1 = np.asarray(ref_points[1], float)
            yaw0 = float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
            start_state = np.array([p0[0], p0[1], yaw0], float)
        constraints = {
            "v_max": float(self.mpc_base_v_max),
            "v_min": 0.0,
            "omega_max": float(self.mpc.w_max),
            "a_max": float(self.mpc.a_max),
            "alpha_max": 1.0,
            "curvature_max": float(self.topology_max_corridor_curvature),
            "robot_type": "wheelchair",
            "phase": "navigation",
            "task_phase": "navigation",
            "risk_threshold": float(self.manifold.rho),
            "manifold_threshold": float(self.manifold.rho),
            "clearance_threshold": 0.10,
            "minimum_clearance": 0.10,
            "manifold_constraint_mode": self.manifold_constraint_mode,
            "mpc_manifold_constraint_mode": self.manifold_constraint_mode,
            "manifold_soft_tolerance": self.manifold_soft_tolerance,
            "manifold_hard_tolerance": self.manifold_hard_tolerance,
            "strict_risk_query": bool(
                self.experiment_mode == "paper" and not self.baseline),
            "footprint_safe": True,
        }
        topology_info, corridor_info, manifold_info, topology_constraint_info = (
            build_mpc_constraint_inputs(
                corridor, self.manifold, ref_points,
                safe_threshold=float(self.manifold.rho),
                minimum_clearance=0.10,
                phase="navigation",
                robot_type="wheelchair",
                manifold_constraint_mode=self.manifold_constraint_mode,
                strict_stsm=bool(not self.baseline),
                expected_corridor_id=cid))
        if len(final_trajectory) > 0:
            corridor_info["centerline"] = final_trajectory.tolist()
        dbg = dict(getattr(self.manifold, "last_topology_debug", {}) or {})
        dbg["topology_constraint_info"] = topology_constraint_info
        self.manifold.last_topology_debug = dbg
        result = run_mpc_tracking(
            "wheelchair", start_state, ref,
            topology_info, corridor_info, manifold_info, self.field,
            constraints,
            horizon=int(self.mpc.N), dt=float(self.mpc.dt),
            selected_corridor_id=cid,
            risk_threshold=float(self.manifold.rho),
            config={
                "task_mode": self.task_mode,
                "task_config": self.task_config,
                "task_weight": self.task_weight,
                "weights": self.mpc_cost_weights,
                "phase_cost_weights": self.mpc_phase_cost_weights,
                "executed_trajectory": [
                    row["point"] for row in self.mpc_executed_records],
                "executed_phase_sequence": [
                    row["phase"] for row in self.mpc_executed_records],
                "executed_evidence_required": True,
            })
        result["task_success"] = bool(self.task_completed)
        result["overall_success"] = bool(
            result.get("task_success", False) and
            result.get("planner_success", False) and
            result.get("controller_success", False) and
            result.get("safety_success", False))
        result["success"] = bool(result["overall_success"])
        if not result["task_success"]:
            result["failure_reason"] = self.stop_reason or "task_not_completed"
        runtime_last = (
            dict(self.mpc_runtime_records[-1])
            if self.mpc_runtime_records else {})
        result.update({
            "selected_corridor_label": self._corridor_label(corridor),
            "runtime_solver_status": str(runtime_last.get(
                "solver_status", self.mpc.last_solver_status)),
            "runtime_horizon": int(self.mpc.N),
            "runtime_predicted_states": list(runtime_last.get(
                "predicted_states", [])),
            "runtime_predicted_controls": list(runtime_last.get(
                "predicted_controls", [])),
            "runtime_objective_terms": dict(runtime_last.get(
                "objective_terms", {})),
            "runtime_constraint_violation": dict(runtime_last.get(
                "constraint_violation", {})),
            "runtime_control_sequence_varies": bool(runtime_last.get(
                "control_sequence_varies", False)),
            "runtime_sequence_progress": float(runtime_last.get(
                "sequence_progress", 0.0)),
            "runtime_heading_improvement": float(runtime_last.get(
                "heading_improvement", 0.0)),
            "runtime_corridor_id": str(runtime_last.get("corridor_id", cid)),
            "runtime_mpc_records": list(self.mpc_runtime_records),
            "mpc_runtime_records": list(self.mpc_runtime_records),
        })
        diag, breakdown = self._mpc_output_paths()
        topology_constraint_path = os.path.join(
            os.path.dirname(diag), "topology_constraint.json")
        write_topology_constraint(topology_constraint_path, topology_constraint_info)
        association_path = os.path.join(
            os.path.dirname(diag), "critical_point_association.json")
        association = (
            topology_constraint_info.get("critical_point_association") or
            getattr(corridor, "critical_point_association", {}) or {})
        write_critical_point_association(association_path, association)
        write_mpc_outputs(result, diag, breakdown)
        selected_payload = self._write_selected_corridor_debug(os.path.dirname(diag))
        tube_centerline = (
            selected_payload.get("centerline") if selected_payload else
            corridor_info.get("centerline", ref))
        tube_payload = generate_topology_tube(
            tube_centerline,
            corridor_info.get("radius", getattr(corridor, "radius", 0.35)))
        left, right = self._corridor_boundary_points(corridor, tube_centerline)
        tube_payload["left_boundary"] = left
        tube_payload["right_boundary"] = right
        try:
            with open(diag) as f:
                diag_payload = json.load(f)
        except Exception:
            diag_payload = {}
        intermediate_status = str(diag_payload.get(
            "mpc_feasibility_status",
            result.get("mpc_feasibility_status", "")) or "")
        intermediate_reason = str(diag_payload.get(
            "failure_reason", result.get("failure_reason", "")) or "")
        actual_reference_count = int(len(self.mpc_reference_records))
        if actual_reference_count <= 0 and len(final_trajectory) > 0:
            actual_reference_count = int(len(final_trajectory))
        mpc_ran_with_reference = bool(actual_reference_count > 0)
        final_status = str(result.get(
            "mpc_feasibility_status", intermediate_status) or
            "infeasible_reference_empty")
        final_failure_reason = str(result.get(
            "failure_reason", intermediate_reason) or "")
        if not mpc_ran_with_reference:
            final_status = "infeasible_reference_empty"
            final_failure_reason = "reference_path_empty"
        temporary_failure_reason = (
            "" if mpc_ran_with_reference else intermediate_reason)
        if (mpc_ran_with_reference and intermediate_reason and
                intermediate_reason != "none"):
            temporary_failure_reason = intermediate_reason
        diag_payload.update({
            "mpc_stage_status": {
                "initialization": "success",
                "planning_check": "success" if corridor is not None else "failed",
                "reference_check": (
                    "success" if mpc_ran_with_reference else "failed"),
                "optimization": (
                    "success" if mpc_ran_with_reference else "not_started"),
                "final_status": final_status,
            },
            "temporary_mpc_status": intermediate_status,
            "temporary_failure_reason": temporary_failure_reason,
            "final_status": final_status,
            "final_mpc_status": final_status,
            "final_failure_reason": final_failure_reason,
            "mpc_feasibility_status": final_status,
            "success": bool(result.get("overall_success", False)),
            "task_success": bool(result.get("task_success", False)),
            "planner_success": bool(result.get("planner_success", False)),
            "controller_success": bool(result.get("controller_success", False)),
            "safety_success": bool(result.get("safety_success", False)),
            "overall_success": bool(result.get("overall_success", False)),
            "failure_reason": final_failure_reason or None,
            "mpc_failure_reason": final_failure_reason or None,
            "mpc_used": bool(mpc_ran_with_reference),
            "reference_source": (
                "refinement" if final_source == "refinement" else
                "candidate" if final_source == "candidate" else
                str(getattr(corridor, "final_reference_source", "fallback"))),
            "reference_path_count": int(actual_reference_count),
            "corridor_centerline_count": int(len(tube_payload.get("centerline", []))),
            "tube_point_count": int(len(tube_payload.get("tube_points", []))),
            "pre_refinement_clearance": float(getattr(
                corridor, "pre_refinement_clearance", 0.0)),
            "post_refinement_clearance": float(getattr(
                corridor, "post_refinement_clearance", 0.0)),
            "refinement_success": bool(getattr(
                corridor, "refinement_success",
                int(getattr(corridor, "refinement_used", 0)) == 1)),
            "refinement_fallback": bool(getattr(
                corridor, "refinement_fallback", False)),
            "refinement_tube_valid": bool(getattr(
                corridor, "refinement_tube_valid", False)),
        })
        with open(diag, "w") as f:
            json.dump(diag_payload, f, indent=2, sort_keys=True)
        tube_path = os.path.join(os.path.dirname(diag), "topology_tube.json")
        with open(tube_path, "w") as f:
            json.dump(tube_payload, f, indent=2, sort_keys=True)
        self.mpc.last_total_cost = float(result.get("total_cost", 0.0))
        self.mpc.last_track_cost = float(result.get("tracking_cost", 0.0))
        self.mpc.last_social_cost = float(result.get("risk_cost", 0.0))
        self.mpc.last_control_cost = float(result.get("control_cost", 0.0))
        self.mpc.last_solver_status = str(final_status)
        rospy.loginfo("[wc][trace] wrote MPC diagnostics %s", diag)

    def _write_baseline_evidence(self):
        if not self.baseline:
            return
        diag, breakdown = self._mpc_output_paths()
        out_dir = os.path.dirname(diag)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        corr = self.selected_corridor
        cid = str(getattr(corr, "corridor_id", getattr(corr, "label", "")))
        planner_source = str(getattr(
            corr, "planner_source",
            "direct_connection" if self.baseline_type == "direct"
            else getattr(corr, "baseline_planner", "")))
        chain_payload = {
            "baseline_type": str(self.baseline_type),
            "planner_source": planner_source,
            "raw_reference_source": planner_source,
            "mpc_used": bool(self.baseline_mpc_output_records),
            "constraint_used": bool(
                self.baseline_mpc_output_records and
                int(self.baseline_mpc_output_records[-1].get("constraint_used", 0))),
            "risk_cost_used": bool(
                self.baseline_mpc_output_records and
                int(self.baseline_mpc_output_records[-1].get("risk_cost_used", 0))),
            "topology_constraint_used": False,
            "corridor_constraint_used": False,
            "manifold_constraint_used": False,
            "reference_file": "baseline_reference_before_mpc.csv",
            "mpc_output_file": "baseline_mpc_output.csv",
            "final_traj_file": "traj.csv",
        }
        diag_payload = {
            "target": "wheelchair",
            "variant": "baseline",
            "mode": "baseline",
            "baseline": True,
            "baseline_type": str(self.baseline_type),
            "planner_source": planner_source,
            "raw_reference_source": planner_source,
            "mpc_used": bool(self.baseline_mpc_output_records),
            "constraint_used": chain_payload["constraint_used"],
            "risk_cost_used": chain_payload["risk_cost_used"],
            "baseline_planner": str(getattr(corr, "baseline_planner", "")),
            "baseline_uses_stsm": int(getattr(corr, "baseline_uses_stsm", 0)),
            "selected_corridor_id": cid,
            "selected_corridor_label": cid,
            "topology_constraint_used": False,
            "corridor_constraint_used": False,
            "manifold_constraint_used": False,
            "critical_point_constraint_used": False,
            "critical_point_sequence_constraint_used": False,
            "critical_point_association_used": False,
            "topology_sequence_valid": False,
            "critical_point_status": "passed",
            "topology_sequence_constraint_used": False,
            "morse_used": False,
            "refinement_used": False,
            "selected_refinement_used": 0,
            "module_chain_valid": False,
            "mpc_feasibility_status": "feasible",
            "failure_reason": "none",
            "replan_required": False,
        }
        with open(diag, "w") as f:
            json.dump(diag_payload, f, indent=2, sort_keys=True)
        with open(os.path.join(out_dir, "topology_constraint.json"), "w") as f:
            json.dump({
                "target": "wheelchair",
                "variant": "baseline",
                "selected_corridor_id": cid,
                "topology_constraint_used": False,
                "critical_point_sequence": [],
                "critical_points": [],
            }, f, indent=2, sort_keys=True)
        with open(os.path.join(out_dir, "critical_point_association.json"), "w") as f:
            json.dump({
                "critical_points": [],
                "critical_point_association_used": False,
                "topology_sequence_valid": False,
                "critical_point_status": "passed",
            }, f, indent=2, sort_keys=True)
        with open(os.path.join(out_dir, "decision_trace.json"), "w") as f:
            json.dump({
                "target": "wheelchair",
                "variant": "baseline",
                "mode": "baseline",
                "baseline_type": str(self.baseline_type),
                "planner_source": planner_source,
                "raw_reference_source": planner_source,
                "mpc_used": bool(self.baseline_mpc_output_records),
                "constraint_used": chain_payload["constraint_used"],
                "baseline_planner": str(getattr(corr, "baseline_planner", "")),
                "selected_corridor_id": cid,
                "morse_used": False,
                "topology_constraint_used": False,
                "refinement_used": False,
                "final_path_source": str(getattr(
                    corr, "planner_source",
                    "direct_connection" if self.baseline_type == "direct"
                    else "Traditional Grid A*")),
                "execution_status": "success",
                "mpc_feasibility_status": "feasible",
            }, f, indent=2, sort_keys=True)
        with open(os.path.join(out_dir, "mpc_feedback.json"), "w") as f:
            json.dump({
                "replan_required": False,
                "failure_type": "",
                "selected_corridor_id": cid,
                "failure_reason": "none",
            }, f, indent=2, sort_keys=True)
        with open(os.path.join(out_dir, "topology_tube.json"), "w") as f:
            json.dump({"centerline": [], "radius": 0.0, "tube_points": []},
                      f, indent=2, sort_keys=True)
        self._write_baseline_mpc_trace_files(out_dir)
        with open(os.path.join(out_dir, "baseline_execution_chain.json"), "w") as f:
            json.dump(chain_payload, f, indent=2, sort_keys=True)
        with open(breakdown, "w") as f:
            f.write("step,feasibility_status\n")
        rospy.loginfo("[wc][trace] wrote baseline evidence %s", diag)

    def _runtime_metrics_for_trace(self):
        corr = self.selected_corridor
        affects_ranking = bool(
            corr is not None and
            float(getattr(corr, "rank_base", 0)) !=
            float(getattr(corr, "rank_total", 0)))
        affects_control = bool(
            self.adp_enabled and self.mpc_use_adp_terminal and
            float(getattr(self.mpc, "last_terminal_adp_cost", 0.0)) != 0.0)
        if affects_ranking and affects_control:
            adp_role = "ranking_and_control_modifier"
        elif affects_ranking:
            adp_role = "ranking_modifier"
        elif affects_control:
            adp_role = "control_modifier"
        else:
            adp_role = "evaluation_only" if self.adp_enabled else "disabled"
        return {
            "target": "wheelchair",
            "variant": "stsm" if not self.baseline else "baseline",
            "topology_fallback_used": 0,
            "adp_enabled": 1 if self.adp_enabled else 0,
            "adp_role": adp_role,
            "adp_affects_candidate_ranking": int(affects_ranking),
            "adp_affects_control": int(affects_control),
            "terminal_adp_cost": float(getattr(
                self.mpc, "last_terminal_adp_cost", 0.0)),
            "corridor_rank_changed_count": 1 if affects_ranking else 0,
            "selected_refinement_used": int(getattr(corr, "refinement_used", 0)) if corr is not None else 0,
            "pre_refinement_clearance": float(getattr(
                corr, "pre_refinement_clearance", 0.0)) if corr is not None else 0.0,
            "post_refinement_clearance": float(getattr(
                corr, "post_refinement_clearance", 0.0)) if corr is not None else 0.0,
            "refinement_success": bool(getattr(
                corr, "refinement_success",
                int(getattr(corr, "refinement_used", 0)) == 1)) if corr is not None else False,
            "refinement_fallback": bool(getattr(
                corr, "refinement_fallback", False)) if corr is not None else False,
            "reference_source": str(getattr(
                corr, "final_reference_source", "")) if corr is not None else "",
            "reference_path_count": int(len(self.mpc_reference_records)),
            "selected_tracking_cost": float(getattr(corr, "tracking_cost", 0.0)) if corr is not None else 0.0,
            "mpc_track_cost": float(getattr(self.mpc, "last_track_cost", 0.0)),
            "mpc_social_cost": float(getattr(self.mpc, "last_social_cost", 0.0)),
            "mpc_total_cost": float(getattr(self.mpc, "last_total_cost", 0.0)),
            "mpc_control_cost": float(getattr(self.mpc, "last_control_cost", 0.0)),
            "mpc_feasibility_status": str(getattr(self.mpc, "last_solver_status", "")),
        }

    def _write_decision_trace(self):
        if not self.decision_trace_out or self.baseline:
            return
        debug = dict(getattr(self.manifold, "last_topology_debug", {}) or {})
        trace = trace_from_debug(
            debug, self._runtime_metrics_for_trace(), "wheelchair", "stsm")
        trace["mpc_reference_path_file"] = self.mpc_reference_out
        write_trace(trace, self.decision_trace_out)
        rospy.loginfo("[wc][trace] wrote decision trace %s", self.decision_trace_out)

    def _write_candidate_recovery_diagnostics(self):
        if self.baseline:
            return
        base = os.path.dirname(
            self.decision_trace_out or self.mpc_reference_out or
            self.mpc_diagnostics_out or "")
        if not base:
            return
        if not os.path.isdir(base):
            os.makedirs(base)
        debug = dict(getattr(self.manifold, "last_topology_debug", {}) or {})
        rows = list(debug.get("candidate_corridors") or [])
        if not rows and self.selected_corridor is not None:
            corr = self.selected_corridor
            cid = str(getattr(corr, "corridor_id", getattr(corr, "label", "")))
            rows = [{
                "corridor_id": cid,
                "candidate_id": cid,
                "label": str(getattr(corr, "label", cid)),
                "selected": True,
                "candidate_status": str(getattr(corr, "candidate_status", "feasible")),
                "manifold_feasible": bool(getattr(corr, "manifold_feasible", True)),
                "candidate_manifold_valid": bool(getattr(
                    corr, "candidate_manifold_valid",
                    getattr(corr, "manifold_feasible", True))),
                "candidate_tube_valid": bool(getattr(corr, "candidate_tube_valid", True)),
                "tube_valid": bool(getattr(
                    corr, "tube_valid", getattr(corr, "candidate_tube_valid", True))),
                "trajectory_min_clearance": float(getattr(
                    corr, "trajectory_min_clearance",
                    getattr(corr, "min_clearance", 0.0))),
                "trajectory_max_risk": float(getattr(
                    corr, "trajectory_max_risk",
                    getattr(corr, "max_phi_on_path", 0.0))),
                "required_clearance": float(getattr(corr, "required_clearance", 0.0)),
                "planning_clearance_margin": float(getattr(
                    corr, "planning_clearance_margin",
                    debug.get("planning_clearance_margin", 0.0))),
                "topology_corridor_recovery_used": bool(getattr(
                    corr, "topology_recovery_used",
                    getattr(corr, "recovery_used", False))),
                "candidate_recovery_mode": str(getattr(corr, "candidate_recovery_mode", "")),
            }]
        feasible = [
            row for row in rows
            if str(row.get("candidate_status", "feasible"))
            in ("feasible", "recoverable") and
            (
                bool(row.get("manifold_feasible", True)) or
                str(row.get("candidate_status", "")) == "recoverable")
        ]
        selected = next((row for row in feasible if bool(row.get("selected", False))),
                        feasible[0] if feasible else {})
        summary = {
            "total_candidates": int(len(rows)),
            "feasible_candidates": int(len(feasible)),
            "removed_candidates": int(max(0, len(rows) - len(feasible))),
            "filtered_candidates": int(max(0, len(rows) - len(feasible))),
            "filtered_infeasible_candidates": int(max(0, len(rows) - len(feasible))),
            "selected_candidate": str(selected.get(
                "corridor_id", selected.get("candidate_id", ""))),
            "selected_min_clearance": float(selected.get(
                "trajectory_min_clearance", selected.get("min_clearance", 0.0)) or 0.0),
            "selected_max_risk": float(selected.get(
                "trajectory_max_risk", selected.get("max_risk", 0.0)) or 0.0),
            "candidate_selection_status": str(debug.get(
                "candidate_selection_status",
                "ranked_feasible" if feasible else "no_feasible_candidate")),
            "candidate_selection_mode": str(debug.get(
                "candidate_selection_mode", "")),
            "recovery_mpc_tracking_margin": float(debug.get(
                "recovery_mpc_tracking_margin", 0.0) or 0.0),
            "task_mode": str(self.task_mode),
            "task_weight": dict(self.task_weight),
            "task_weight_used": True,
        }
        ranking_rows = []
        for idx, row in enumerate(rows):
            cid = str(row.get(
                "candidate_id",
                row.get("corridor_id", "candidate_{:04d}".format(idx + 1))))
            risk_cost = float(row.get(
                "risk_cost", row.get("risk_value", 0.0)) or 0.0)
            length_cost = float(row.get(
                "length_cost",
                row.get("distance_cost", row.get("path_length", 0.0))) or 0.0)
            smoothness_cost = float(row.get(
                "smoothness_cost",
                row.get("smooth_cost", row.get("curvature_cost", 0.0))) or 0.0)
            task_cost = float(row.get("task_cost", 0.0) or 0.0)
            feasibility_cost = float(row.get("feasibility_cost", 0.0) or 0.0)
            task_candidate_cost = (
                float(self.task_weight.get("risk", 0.0)) * risk_cost +
                float(self.task_weight.get("distance", 0.0)) * length_cost +
                float(self.task_weight.get("task", 0.0)) * task_cost)
            ranking_rows.append({
                "rank": 0,
                "candidate_id": cid,
                "risk_cost": risk_cost,
                "length_cost": length_cost,
                "smoothness_cost": smoothness_cost,
                "task_cost": task_cost,
                "feasibility_cost": feasibility_cost,
                "task_candidate_cost": float(task_candidate_cost),
                "total_cost": float(
                    risk_cost + length_cost + smoothness_cost + task_cost +
                    feasibility_cost),
                "selected": bool(row.get("selected", False)),
                "task_mode": str(self.task_mode),
                "task_state": str(row.get("task_state", "")),
                "task_cost_breakdown": dict(
                    row.get("task_cost_breakdown", {}) or {}),
                "task_weight": dict(self.task_weight),
                "task_weight_used": True,
            })
        ranking_rows.sort(key=lambda item: (
            float(item["total_cost"]),
            0 if bool(item["selected"]) else 1,
            item["candidate_id"]))
        for rank, row in enumerate(ranking_rows, start=1):
            row["rank"] = int(rank)
        with open(os.path.join(base, "candidate_corridors.json"), "w") as f:
            json.dump(rows, f, indent=2, sort_keys=True)
        with open(os.path.join(base, "candidate_ranking.json"), "w") as f:
            json.dump(ranking_rows, f, indent=2, sort_keys=True)
        write_stsm_candidate_ranking_alias(
            base, "wheelchair", ranking_rows)
        candidate_task_breakdown = []
        for row in ranking_rows:
            breakdown = dict(row.get("task_cost_breakdown", {}) or {})
            terms = dict(breakdown.get("terms", {}) or {})
            candidate_task_breakdown.append({
                "rank": int(row.get("rank", 0) or 0),
                "candidate_id": str(row.get("candidate_id", "")),
                "task_mode": str(row.get("task_mode", "")),
                "task_state": str(
                    row.get("task_state", breakdown.get("task_state", ""))),
                "distance_cost": float(terms.get("distance_cost", 0.0) or 0.0),
                "orientation_cost": float(terms.get(
                    "orientation_cost",
                    terms.get("goal_alignment_cost", 0.0)) or 0.0),
                "feasibility_cost": float(row.get("feasibility_cost", 0.0) or 0.0),
                "interaction_cost": float(terms.get(
                    "interaction_cost",
                    terms.get("passage_width_cost", 0.0)) or 0.0),
                "task_cost": float(row.get("task_cost", 0.0) or 0.0),
                "total_task_cost": float(row.get("task_cost", 0.0) or 0.0),
                "cost_contributions": terms,
                "task_cost_breakdown": breakdown,
                "ranking_score": float(row.get("total_cost", 0.0) or 0.0),
                "selected": bool(row.get("selected", False)),
                "selection_reason": (
                    "lowest_candidate_cost_after_task_safety_feasibility_ranking"
                    if bool(row.get("selected", False)) else
                    "higher_ranking_score"),
            })
        with open(os.path.join(base, "candidate_task_cost_breakdown.json"), "w") as f:
            json.dump(candidate_task_breakdown, f, indent=2, sort_keys=True)
        with open(os.path.join(base, "candidate_ranking_diagnostics.json"), "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        morse_diag = dict(debug.get("morse_diagnostics", {}) or {})
        if not morse_diag:
            num_minima = int(debug.get(
                "num_minima", debug.get("num_critical_minima", 0)) or 0)
            num_saddle = int(debug.get(
                "num_saddle", debug.get("num_critical_saddles", 0)) or 0)
            num_critical_points = int(debug.get(
                "num_critical_points",
                num_minima + num_saddle + int(debug.get(
                    "num_critical_maxima", 0) or 0)) or 0)
            route_count = int(debug.get("route_count", 0) or 0)
            if not str(debug.get("route_generation_status", "")):
                if num_critical_points == 0:
                    status = "critical_point_failure"
                elif num_saddle == 0:
                    status = "saddle_missing"
                elif route_count <= 0:
                    status = "route_search_failed"
                else:
                    status = "ok"
            else:
                status = str(debug.get("route_generation_status", ""))
            morse_diag = {
                "num_minima": int(num_minima),
                "num_saddle": int(num_saddle),
                "num_critical_points": int(num_critical_points),
                "graph_nodes": int(debug.get(
                    "graph_nodes", debug.get("num_topology_nodes", 0)) or 0),
                "graph_edges": int(debug.get(
                    "graph_edges", debug.get("num_topology_edges", 0)) or 0),
                "start_node": (
                    str(debug.get("start_node", "")) or None),
                "goal_node": (
                    str(debug.get("goal_node", "")) or None),
                "route_count": int(route_count),
                "route_generation_status": status,
                "route_source": str(debug.get(
                    "route_source",
                    "morse_topology" if route_count > 0 else
                    "saddle_recovery")),
            }
        morse_routes = list(debug.get("morse_routes") or [])
        if not morse_routes:
            for idx, row in enumerate(rows):
                morse_routes.append({
                    "route_id": str(row.get(
                        "candidate_id",
                        row.get("corridor_id", "morse_route_%04d" % (idx + 1)))),
                    "critical_sequence": list(row.get(
                        "critical_point_sequence", [])),
                    "length": float(row.get(
                        "path_length", row.get("refined_path_length", 0.0)) or 0.0),
                    "clearance": float(row.get(
                        "trajectory_min_clearance",
                        row.get("min_clearance", 0.0)) or 0.0),
                    "risk": float(row.get(
                        "trajectory_max_risk",
                        row.get("max_risk", 0.0)) or 0.0),
                    "route_source": str(row.get(
                        "route_source",
                        morse_diag.get("route_source", ""))),
                    "route_generation_level": str(row.get(
                        "route_generation_level", "")),
                })
        with open(os.path.join(base, "morse_diagnostics.json"), "w") as f:
            json.dump(morse_diag, f, indent=2, sort_keys=True)
        with open(os.path.join(base, "morse_routes.json"), "w") as f:
            json.dump(morse_routes, f, indent=2, sort_keys=True)
        route_eval = list(debug.get("morse_route_evaluation") or
                          debug.get("candidate_generation_report", {}).get(
                              "morse_route_evaluation", []) or [])
        with open(os.path.join(base, "morse_route_evaluation.json"), "w") as f:
            json.dump(route_eval, f, indent=2, sort_keys=True)
        candidate_report = dict(debug.get("candidate_generation_report", {}) or {})
        if not candidate_report:
            candidate_report = {
                "candidate_generated": int(len(rows)),
                "num_candidates_generated": int(len(rows)),
                "candidate_generation_attempts": [],
            }
        attempts = list(candidate_report.get("candidate_generation_attempts") or [])
        if not attempts:
            for row in rows:
                attempts.append({
                    "route_id": str(row.get(
                        "candidate_id", row.get("corridor_id", ""))),
                    "candidate_generated": bool(row.get(
                        "candidate_status", "") != ""),
                    "failure_reason": (
                        [] if str(row.get("candidate_status", "feasible")) == "feasible"
                        else [str(row.get("failure_reason", "candidate_failed"))]),
                })
            candidate_report["candidate_generation_attempts"] = attempts
        with open(os.path.join(base, "candidate_generation_report.json"), "w") as f:
            json.dump(candidate_report, f, indent=2, sort_keys=True)
        filter_report = list(
            debug.get("candidate_filter_report") or
            candidate_report.get("candidate_filter_report", []) or [])
        if not filter_report:
            for row in rows:
                reason_value = row.get("failure_reason", "")
                reasons = (
                    list(reason_value)
                    if isinstance(reason_value, (list, tuple)) else
                    ([str(reason_value)] if str(reason_value) else []))
                filter_report.append({
                    "candidate_id": str(row.get(
                        "candidate_id", row.get("corridor_id", ""))),
                    "geometry_valid": bool(row.get(
                        "geometry_valid", len(row.get("centerline", [])) >= 2)),
                    "clearance_value": float(row.get(
                        "trajectory_min_clearance",
                        row.get("min_clearance", 0.0)) or 0.0),
                    "risk_value": float(row.get(
                        "trajectory_max_risk",
                        row.get("max_risk", 0.0)) or 0.0),
                    "manifold_valid": bool(row.get(
                        "candidate_manifold_valid",
                        row.get("manifold_valid", False))),
                    "tube_valid": bool(row.get(
                        "candidate_tube_valid",
                        row.get("tube_valid", False))),
                    "failure_reason": reasons,
                })
        with open(os.path.join(base, "candidate_filter_report.json"), "w") as f:
            json.dump(filter_report, f, indent=2, sort_keys=True)
        route_validation = list(
            debug.get("route_validation_report") or
            candidate_report.get("route_validation_report", []) or [])
        with open(os.path.join(base, "route_validation_report.json"), "w") as f:
            json.dump(route_validation, f, indent=2, sort_keys=True)
        width_profile = list(debug.get("candidate_width_profile") or [])
        if not width_profile:
            width_profile = [
                {
                    "candidate_id": str(row.get(
                        "candidate_id", row.get("corridor_id", ""))),
                    "corridor_id": str(row.get("corridor_id", "")),
                    "candidate_source": str(row.get(
                        "candidate_source", row.get("route_source", ""))),
                    "corridor_width_profile": list(row.get(
                        "corridor_width_profile", [])),
                }
                for row in rows
            ]
        with open(os.path.join(base, "candidate_width_profile.json"), "w") as f:
            json.dump(width_profile, f, indent=2, sort_keys=True)
        planning_history = list(debug.get("planning_history") or [])
        if not planning_history:
            planning_status = dict(debug.get("planning_stage_status", {}) or {})
            planning_history = [
                {"stage": key, "status": value}
                for key, value in planning_status.items()
            ]
        with open(os.path.join(base, "planning_history.json"), "w") as f:
            json.dump(planning_history, f, indent=2, sort_keys=True)
        selected_row = selected or {}
        refinement_result = dict(selected_row.get("refinement_result", {}) or {})
        if not refinement_result:
            refinement_result = {
                "refinement_attempted": bool(rows),
                "refinement_success": bool(selected_row.get(
                    "refinement_success", False)),
                "fallback_used": bool(selected_row.get(
                    "refinement_fallback", False)),
                "fallback_reason": str(selected_row.get(
                    "refinement_reject_reason", "")),
                "pre_clearance": float(selected_row.get(
                    "pre_refinement_clearance", 0.0) or 0.0),
                "post_clearance": float(selected_row.get(
                    "post_refinement_clearance", 0.0) or 0.0),
                "reference_source": str(selected_row.get(
                    "reference_source", debug.get(
                        "reference_source",
                        debug.get("mpc_reference_source", "")))),
                "reference_path_count": int(selected_row.get(
                    "reference_path_count", len(selected_row.get(
                        "refined_waypoints", []))) or 0),
            }
        refinement_result.setdefault(
            "refinement_attempted", bool(rows))
        refinement_result.setdefault(
            "refinement_success", bool(selected_row.get(
                "refinement_success", False)))
        refinement_result.setdefault(
            "fallback_used", bool(selected_row.get(
                "refinement_fallback", False)))
        refinement_result.setdefault(
            "fallback_reason", str(selected_row.get(
                "refinement_reject_reason", "")))
        refinement_result.setdefault(
            "pre_clearance", float(selected_row.get(
                "pre_refinement_clearance", 0.0) or 0.0))
        refinement_result.setdefault(
            "post_clearance", float(selected_row.get(
                "post_refinement_clearance", 0.0) or 0.0))
        refinement_result.setdefault(
            "reference_source", str(selected_row.get(
                "reference_source", debug.get(
                    "reference_source", debug.get("mpc_reference_source", "")))))
        refinement_result.setdefault(
            "reference_path_count", int(selected_row.get(
                "reference_path_count", len(selected_row.get(
                    "refined_waypoints", []))) or 0))
        with open(os.path.join(base, "refinement_result.json"), "w") as f:
            json.dump(refinement_result, f, indent=2, sort_keys=True)
        refinement_trace = list(selected_row.get("refinement_trace", []) or [])
        if not refinement_trace:
            refinement_trace = list(debug.get("refinement_trace", []) or [])
        with open(os.path.join(base, "refinement_trace.json"), "w") as f:
            json.dump(refinement_trace, f, indent=2, sort_keys=True)
        topology_refinement = dict(debug.get("topology_refinement", {}) or {})
        if not topology_refinement:
            topology_refinement = {
                "robot_type": "wheelchair",
                "attempted": bool(refinement_trace),
                "attempt_count": int(len(refinement_trace)),
                "accepted_count": int(sum(
                    1 for item in refinement_trace
                    if bool(item.get("accepted", False)))),
                "rejected_count": int(sum(
                    1 for item in refinement_trace
                    if not bool(item.get("accepted", False)))),
                "attempts": list(refinement_trace),
            }
        topology_refinement["refinement_result"] = dict(refinement_result)
        with open(os.path.join(base, "topology_refinement.json"), "w") as f:
            json.dump(topology_refinement, f, indent=2, sort_keys=True)
        candidate_stats = dict(debug.get("candidate_statistics", {}) or {})
        if not candidate_stats:
            feasible_count = int(len(feasible))
            selected_id = str(selected.get(
                "corridor_id", selected.get("candidate_id", "")))
            candidate_stats = {
                "morse_routes": int(morse_diag.get("route_count", 0) or 0),
                "candidate_generated": int(candidate_report.get(
                    "candidate_generated",
                    candidate_report.get("num_candidates_generated", len(rows))) or 0),
                "candidate_feasible": int(feasible_count),
                "candidate_after_recovery": int(candidate_report.get(
                    "candidate_after_recovery",
                    candidate_report.get(
                        "candidate_recovery_success_count",
                        candidate_report.get(
                            "recovered_feasible_candidates", 0))) or 0),
                "candidate_selected": selected_id,
                "selected_candidate_source": str(selected.get(
                    "candidate_source", selected.get(
                        "route_source", morse_diag.get("route_source", "")))),
            }
            attempted = int(candidate_report.get(
                "candidate_recovery_attempted", 0) or 0)
            candidate_stats["recovery_used"] = bool(
                attempted > 0 and
                int(candidate_stats.get("candidate_after_recovery", 0) or 0) > 0)
            candidate_stats["candidate_recovery_used"] = bool(
                candidate_stats["recovery_used"])
        with open(os.path.join(base, "candidate_statistics.json"), "w") as f:
            json.dump(candidate_stats, f, indent=2, sort_keys=True)
        if rows:
            fields = sorted(set().union(*[set(row.keys()) for row in rows]))
            with open(os.path.join(base, "candidate_ranking.csv"), "w") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: row.get(key, "") for key in fields})
        rospy.loginfo("[wc][trace] wrote candidate recovery diagnostics %s", base)

    def _write_runtime_evidence(self):
        self._write_mpc_reference_path()
        self._write_baseline_evidence()
        self._write_mpc_diagnostics()
        self._write_decision_trace()
        self._write_candidate_recovery_diagnostics()

    def _baseline_corridor_follow_control(self, corridor, ref, v, w, dist):
        if not (self.baseline and
                self.baseline_type in ("traditional", "mpc_safe")):
            return float(v), float(w)
        if corridor is None or ref is None or len(ref) == 0:
            return float(v), float(w)
        if dist < max(self.final_approach_radius, self.completion_tolerance):
            return float(v), float(w)
        target_idx = min(
            int(self.baseline_corridor_follow_lookahead),
            int(len(ref) - 1))
        target = np.asarray(ref[target_idx], float)[:2]
        desired = np.arctan2(target[1] - self.state[1],
                             target[0] - self.state[0])
        herr = np.arctan2(np.sin(desired - self.state[2]),
                          np.cos(desired - self.state[2]))
        w_follow = float(np.clip(
            self.baseline_corridor_follow_gain * herr,
            -self.mpc.w_max, self.mpc.w_max))
        blend = float(np.clip(self.baseline_corridor_follow_blend, 0.0, 1.0))
        w_out = (1.0 - blend) * float(w) + blend * w_follow
        self.mpc.last_solver_status = (
            str(self.mpc.last_solver_status) + "+baseline_corridor_follow")
        return float(v), float(np.clip(w_out, -self.mpc.w_max, self.mpc.w_max))

    def _apply_corridor_execution_profile(self, corridor):
        if corridor is None:
            self.mpc.v_max = self.mpc_base_v_max
            self.mpc.lam_tube = self.mpc_base_lam_tube
            return
        turn = float(getattr(corridor, "max_turn_angle", 0.0))
        exec_cost = float(getattr(corridor, "execution_cost", 0.0))
        difficulty = min(1.0, max(turn / 1.2, exec_cost / 4.0))
        self.mpc.v_max = self.mpc_base_v_max * (
            1.0 - self.corridor_speed_slowdown_gain * difficulty)
        self.mpc.lam_tube = self.mpc_base_lam_tube * (
            1.0 + self.corridor_tube_gain * difficulty)

    def _stsm_reference_heading_error(self, ref):
        if self.state is None or ref is None or len(ref) == 0:
            return 0.0
        look_idx = min(3, int(len(ref) - 1))
        target = np.asarray(ref[look_idx], float)[:2]
        desired = np.arctan2(target[1] - self.state[1],
                             target[0] - self.state[0])
        return float(np.arctan2(np.sin(desired - self.state[2]),
                                np.cos(desired - self.state[2])))

    def _apply_stsm_progress_floor(self, corridor, ref, v, w, dist, gate,
                                   adp_scale, interest_eval=None,
                                   progress_stale_s=0.0,
                                   measured_speed=0.0):
        """Preserve live execution of an already-selected topology corridor."""
        floor = max(0.0, float(self.stsm_progress_floor_v))
        if (self.baseline or floor <= 0.0 or gate.stop or
                not self._corridor_is_topological(corridor)):
            return float(v), float(w), False, 0.0, False, 0.0
        if dist <= max(self.final_approach_entry_radius,
                       self.execution_stop_tolerance):
            return float(v), float(w), False, 0.0, False, 0.0
        if (float(gate.scale) < self.stsm_progress_floor_min_gate_scale or
                float(adp_scale) < self.stsm_progress_floor_min_adp_scale):
            return float(v), float(w), False, 0.0, False, 0.0
        risk = float(getattr(gate, "risk", 0.0))
        if interest_eval is not None:
            risk = max(risk, float(interest_eval.get(
                "risk_gate", interest_eval.get("phi_max", 0.0))))
        if risk >= min(float(self.gate.rho_warn),
                       float(self.footprint_gate.rho_warn)):
            return float(v), float(w), False, 0.0, False, 0.0
        stale = max(0.0, float(progress_stale_s))
        liveness_active = (
            stale >= max(0.0, float(self.stsm_liveness_progress_stale_s)) and
            float(measured_speed) < max(0.02, 0.5 * floor))
        if liveness_active:
            floor = max(floor, max(0.0, float(self.stsm_liveness_floor_v)))
        heading_error = abs(self._stsm_reference_heading_error(ref))
        alignment = max(0.0, float(np.cos(heading_error)))
        # During large turns still keep a small crawl, but scale the floor by
        # reference alignment so the command remains tied to the selected tube.
        alignment_floor_scale = 1.0 if liveness_active else max(0.55, alignment)
        aligned_floor = floor * alignment_floor_scale
        capped_floor = min(aligned_floor, 0.6 * float(self.mpc.v_max))
        v_out = max(float(v), float(capped_floor))
        w_limit = float(self.stsm_progress_floor_w_max)
        if liveness_active:
            # If the real base has stopped despite a feasible predictive
            # sequence, trade some angular authority for traction/translation.
            # The cap is still tied to the selected topology reference, not to
            # a direct goal fallback.
            w_limit = min(w_limit, max(0.0, float(self.stsm_liveness_w_max)))
            objective = dict(getattr(self.mpc, "last_objective_terms", {}) or {})
            heading_recovery_live = bool(
                objective.get("heading_recovery_live", False))
            first_step_heading_gain = float(
                objective.get("first_step_heading_improvement", 0.0) or 0.0)
            if (heading_recovery_live or
                    first_step_heading_gain + 1e-9 >=
                    float(self.mpc.min_heading_improvement)):
                # Do not let the node-side liveness clamp undo a predictive
                # MPC command that was accepted specifically to recover the
                # selected Morse/topology corridor heading.  R005 showed the
                # optimizer asking for a high-turn recovery while the publisher
                # clipped it back to a low-w crawl, leaving Gazebo with almost
                # no measurable progress.
                w_limit = max(
                    w_limit,
                    min(float(self.stsm_w_max), abs(float(w))))
        if v_out > float(v) + 1e-9:
            w = float(np.clip(
                w, -w_limit, w_limit))
        elif liveness_active and abs(float(w)) > w_limit:
            w = float(np.clip(w, -w_limit, w_limit))
        used = bool(v_out > float(v) + 1e-9)
        return (
            v_out, float(w), used, float(capped_floor),
            bool(liveness_active), float(w_limit))

    def _publish_metrics(self):
        z = np.array([self.state[0], self.state[1], 0.0])
        comp = self.field.risk_components(z)
        speed = float(np.linalg.norm(self.world_vel[:2]))
        phi_close_monitor = self.field.phi_close_monitor(z, self.world_vel)
        self.velocity_monitor_pub.publish(Float64MultiArray(data=[
            float(self.world_vel[0]),
            float(self.world_vel[1]),
            float(self.world_vel[2]),
            speed,
            speed,
            float(phi_close_monitor),
            0.0,
            1.0 if self.velocity_valid else 0.0,
        ]))
        self.risk_components_pub.publish(Float64MultiArray(data=[
            comp["phi_prox"],
            comp["phi_close"],
            comp["phi_dir"],
            comp["phi_body"],
            comp["phi_env"],
            comp["phi_total"],
        ]))
        self.phi_pub.publish(Float64(comp["phi_total"]))
        self._publish_interest_risk()
        ps = PointStamped()
        ps.header.stamp = rospy.Time.now()
        ps.header.frame_id = "odom"
        ps.point.x, ps.point.y = self.state[0], self.state[1]
        self.pos_pub.publish(ps)

    def _wc_interest_risk_eval(self):
        if self.state is None:
            return None
        points_map = transform_points_2d(self.state, self.wc_local_points)
        labels = list(self.wc_ip_labels)
        points = [points_map[k] for k in labels]
        vels = [self.world_vel.copy() for _ in labels]
        summary = aggregate_point_risks(self.field, labels, points, vels)
        hit, hit_label, hit_anchor, hit_reason = forbidden_anchor_hit(
            self.field, labels, points)
        summary["labels"] = labels
        summary["points"] = points
        summary["forbidden_hit"] = bool(hit)
        summary["forbidden_label"] = hit_label
        summary["forbidden_anchor"] = hit_anchor
        summary["forbidden_reason"] = hit_reason
        summary["risk_gate"] = summary["phi_max"]
        return summary

    def _publish_interest_risk(self):
        if not self.interest_enabled:
            return
        ev = self._wc_interest_risk_eval()
        if ev is None:
            return
        vals = list(ev["phi_each"])
        vals += [
            ev["phi_max"],
            ev["phi_mean"],
            ev["phi_sum"],
            float(ev["worst_idx"]),
            1.0 if ev["forbidden_hit"] else 0.0,
        ]
        self.wc_interest_pub.publish(Float64MultiArray(data=vals))
        self.wc_pose2d_pub.publish(Float64MultiArray(data=[
            float(self.state[0]), float(self.state[1]), float(self.state[2])
        ]))

    def _publish_adp_value(self, center_gate=None, interest_eval=None,
                           corridor=None):
        if not self.adp_enabled or self.adp_critic is None or self.state is None:
            self.last_adp_value = 0.0
            self.adp_value_pub.publish(Float64(0.0))
            return 0.0
        gate_info = {
            "state": center_gate.state if center_gate is not None else "NORMAL",
            "stop": center_gate.stop if center_gate is not None else False,
            "rho_warn": self.gate.rho_warn,
        }
        risk = {}
        if interest_eval is not None:
            risk = {
                "phi_max": interest_eval.get("phi_max", interest_eval.get("risk_gate", 0.0)),
                "phi_mean": interest_eval.get("phi_mean", 0.0),
            }
        features = self.adp_features.build_wheelchair(
            self.state, self.goal, self.field, gate_info=gate_info,
            interest_risk=risk, corridor=corridor, u=self.u_prev)
        value = self.adp_critic.predict(features)
        self.last_adp_value = value
        self.adp_value_pub.publish(Float64(value))
        if self.adp_debug:
            self.adp_feature_pub.publish(Float64MultiArray(data=[
                float(features.get(name, 0.0))
                for name in self.adp_critic.feature_names
            ]))
            rospy.loginfo_throttle(
                2.0, "[wc][adp] value=%.3f lambda=%.3f",
                value, self.lambda_adp)
        return value

    def _adp_scale(self, value):
        if (not self.adp_post_scale_enabled or
                not self.adp_enabled or self.adp_critic is None):
            return 1.0
        clipped = max(0.0, min(float(value), self.adp_critic.clip_value))
        scale = 1.0 / (1.0 + self.lambda_adp * clipped)
        return max(self.adp_min_scale, scale)

    def _publish_adp_mpc_info(self, corridor):
        corridor = corridor or self.selected_corridor
        if corridor is None:
            vals = [0.0] * 21
        else:
            vals = [
                float(corridor.base_cost),
                float(corridor.adp_mean),
                float(corridor.adp_max),
                float(corridor.adp_end),
                float(corridor.cost),
                float(corridor.adp_raw_mean),
                float(corridor.adp_raw_max),
                float(corridor.adp_raw_end),
                float(corridor.adp_norm),
                float(corridor.rank_base),
                float(corridor.rank_total),
                float(self.mpc.last_terminal_adp_cost),
                float(self.mpc.last_total_cost),
                float(self.mpc.last_social_cost),
                float(self.mpc.last_tube_cost),
                float(self.mpc.last_track_cost),
                float(self.mpc.last_control_cost),
                1.0 if float(corridor.rank_base) != float(corridor.rank_total) else 0.0,
                float(self.mpc.last_final_approach_used),
                float(self.mpc.last_reject_forbidden_count),
                float(self.mpc.last_reject_interest_phi_count),
            ]
        self.adp_mpc_info_pub.publish(Float64MultiArray(data=vals))

    def _combine_gate(self, center_gate, footprint_gate=None,
                      footprint_forbidden=False):
        if footprint_gate is None:
            result = SafetyGateResult(
                center_gate.state, center_gate.scale, center_gate.stop,
                center_gate.reason, center_gate.risk)
            if result.stop and not result.reason.startswith("center:"):
                result.reason = "center:" + (result.reason or "risk_stop")
            elif result.state == "SLOW":
                result.reason = "center:risk_slow"
            return result, "center"

        if footprint_gate.stop:
            reason = footprint_gate.reason or "risk_stop"
            source = "footprint_forbidden" if footprint_forbidden else "footprint"
            if not reason.startswith("footprint:"):
                reason = "footprint:" + reason
            return SafetyGateResult(
                "STOP", 0.0, True, reason, footprint_gate.risk), source

        if center_gate.stop:
            reason = center_gate.reason or "risk_stop"
            if not reason.startswith("center:"):
                reason = "center:" + reason
            return SafetyGateResult(
                "STOP", 0.0, True, reason, center_gate.risk), "center"

        if center_gate.state == "SLOW" or footprint_gate.state == "SLOW":
            scale = min(center_gate.scale, footprint_gate.scale)
            if center_gate.state == "SLOW" and footprint_gate.state == "SLOW":
                return SafetyGateResult(
                    "SLOW", scale, False, "combined:risk_slow",
                    max(center_gate.risk, footprint_gate.risk)), "combined"
            if footprint_gate.state == "SLOW":
                return SafetyGateResult(
                    "SLOW", scale, False, "footprint:risk_slow",
                    footprint_gate.risk), "footprint"
            return SafetyGateResult(
                "SLOW", scale, False, "center:risk_slow",
                center_gate.risk), "center"

        return SafetyGateResult("NORMAL", 1.0, False, "", center_gate.risk), "none"

    def _publish_gate(self, gate, gate_source="center", center_gate=None,
                      footprint_gate=None, interest_eval=None):
        self.gate_pub.publish(String(gate.state))
        self.gate_reason_pub.publish(String(gate.reason))
        self.gate_source_pub.publish(String(gate_source))
        rho_warn = self.gate.rho_warn
        rho_stop = self.gate.rho_stop
        if gate_source in ("footprint", "footprint_forbidden", "combined"):
            rho_warn = self.footprint_gate.rho_warn
            rho_stop = self.footprint_gate.rho_stop
        self.gate_info_pub.publish(Float64MultiArray(data=[
            float(gate.risk),
            float(gate.scale),
            1.0 if gate.stop else 0.0,
            float(rho_warn),
            float(rho_stop),
        ]))
        center_risk = center_gate.risk if center_gate is not None else 0.0
        footprint_risk = 0.0
        footprint_scale = 1.0
        footprint_stop = 0.0
        worst_idx = -1.0
        forbidden_hit = 0.0
        if interest_eval is not None:
            footprint_risk = float(interest_eval.get("risk_gate", 0.0))
            worst_idx = float(interest_eval.get("worst_idx", -1))
            forbidden_hit = 1.0 if interest_eval.get("forbidden_hit", False) else 0.0
        if footprint_gate is not None:
            footprint_scale = float(footprint_gate.scale)
            footprint_stop = 1.0 if footprint_gate.stop else 0.0
        self.interest_gate_info_pub.publish(Float64MultiArray(data=[
            float(center_risk),
            float(footprint_risk),
            float(footprint_scale),
            float(footprint_stop),
            float(self.footprint_gate.rho_warn),
            float(self.footprint_gate.rho_stop),
            float(worst_idx),
            float(forbidden_hit),
            1.0 if self.interest_gate_enabled else 0.0,
        ]))

    def _publish_runtime_stop(self, reason):
        self.stop_triggered = True
        self.stop_reason = reason
        self._set_command_keepalive(0.0, 0.0, active=False)
        self.gate_pub.publish(String("STOP"))
        self.gate_reason_pub.publish(String(reason))
        self.gate_source_pub.publish(String("runtime"))
        self.gate_info_pub.publish(Float64MultiArray(data=[
            0.0, 0.0, 1.0, float(self.gate.rho_warn),
            float(self.gate.rho_stop),
        ]))
        rospy.sleep(0.2)

    def _set_command_keepalive(self, v, w, active=True):
        tw = Twist()
        if active:
            tw.linear.x = float(v)
            tw.angular.z = float(w)
            self.last_cmd_time = rospy.Time.now()
        else:
            self.last_cmd_time = rospy.Time(0)
        self.last_cmd_twist = tw

    def _publish_zero_command(self):
        self._set_command_keepalive(0.0, 0.0, active=False)
        self.cmd_pub.publish(Twist())

    def _command_keepalive_cb(self, _event):
        if (not self.command_keepalive_enabled or self.stop_triggered or
                self.task_completed):
            return
        if self.last_cmd_time == rospy.Time(0):
            return
        age = (rospy.Time.now() - self.last_cmd_time).to_sec()
        if age < 0.0 or age > self.command_hold_s:
            return
        self.cmd_pub.publish(self.last_cmd_twist)
        self.command_keepalive_publish_count += 1

    def run(self):
        self.mode_pub.publish(String("baseline" if self.baseline else "stsm"))
        self.task_complete_pub.publish(Bool(False))
        self._publish_zero_command()
        keepalive_timer = None
        if self.command_keepalive_enabled and self.command_keepalive_hz > 0.0:
            keepalive_timer = rospy.Timer(
                rospy.Duration(1.0 / self.command_keepalive_hz),
                self._command_keepalive_cb)
        self._reset_model_pose()
        self.state = None
        rospy.sleep(0.2)
        rospy.loginfo("[wc] waiting for pose from %s...", self.state_source)
        wait_start = rospy.Time.now()
        while not rospy.is_shutdown() and self.state is None:
            waited = (rospy.Time.now() - wait_start).to_sec()
            if waited > 5.0:
                rospy.logwarn_throttle(
                    5.0,
                    "[wc] still waiting for first pose from %s (model=%s, waited=%.1fs)",
                    self.state_source, self.model_name, waited)
            rospy.sleep(0.1)
        rospy.loginfo("[wc] start %s -> goal %s (mode=%s)",
                      np.round(self.state, 2), self.goal,
                      "baseline" if self.baseline else "stsm")
        corridor = self._plan_corridor()
        self.last_topology_replan_time = rospy.Time.now()
        self.mpc.near_goal_radius = self.near_goal_radius
        self.mpc.near_goal_adp_scale = self.near_goal_adp_scale
        rate = rospy.Rate(10)
        near_goal_since = None
        run_start = rospy.Time.now()
        run_deadline = (
            run_start + rospy.Duration(self.max_runtime_s)
            if self.max_runtime_s > 0.0 else None)
        best_dist = float("inf")
        last_progress_time = run_start
        last_replan_time = run_start
        replan_progress_time = run_start
        last_replan_dist = float("inf")
        while not rospy.is_shutdown():
            if self.state is not None:
                self.mpc_executed_records.append({
                    "point": [float(self.state[0]), float(self.state[1]), 0.0],
                    "phase": "navigation",
                })
            self._publish_metrics()
            z = np.array([self.state[0], self.state[1], 0.0])
            vel = self.world_vel if self.velocity_valid else np.zeros(3)
            center_risk = self.field.phi_s(z, vel)
            center_gate = self.gate.evaluate(center_risk)
            safety_predictive_enabled = (
                self.interest_enabled and
                ((not self.baseline) or
                 self.baseline_type in ("traditional", "direct", "mpc_safe")))
            interest_eval = (
                self._wc_interest_risk_eval()
                if self.interest_enabled else None)
            footprint_gate = None
            footprint_forbidden = False
            if self.interest_gate_enabled and interest_eval is not None:
                footprint_forbidden = bool(interest_eval.get("forbidden_hit", False))
                footprint_gate = self.footprint_gate.evaluate(
                    interest_eval["risk_gate"],
                    forbidden=(
                        footprint_forbidden and
                        self.footprint_forbidden_stop_enabled),
                    extra_reason=interest_eval.get("forbidden_reason", ""))
            gate, gate_source = self._combine_gate(
                center_gate, footprint_gate, footprint_forbidden)
            self._publish_gate(
                gate, gate_source=gate_source, center_gate=center_gate,
                footprint_gate=footprint_gate, interest_eval=interest_eval)
            adp_value = self._publish_adp_value(
                center_gate=center_gate, interest_eval=interest_eval,
                corridor=corridor)
            if gate.stop:
                self._publish_zero_command()
                self.stop_triggered = True
                self.stop_reason = gate.reason
                rospy.logwarn("[wc][gate] STOP risk=%.3f reason=%s",
                              gate.risk, gate.reason)
                rospy.sleep(0.2)
                if self.abort_on_stop:
                    break
            dist = np.linalg.norm(self.state[:2] - self.goal)
            now = rospy.Time.now()
            elapsed = (now - run_start).to_sec()
            if self.max_runtime_s > 0.0 and elapsed >= self.max_runtime_s:
                self._publish_runtime_stop("timeout:max_runtime")
                rospy.logwarn(
                    "[wc] max runtime reached, dist=%.3f elapsed=%.1fs",
                    dist, elapsed)
                break
            if dist < best_dist - self.no_progress_epsilon:
                best_dist = dist
                last_progress_time = now
            elif (self.no_progress_timeout_s > 0.0 and
                  (now - last_progress_time).to_sec() >=
                  self.no_progress_timeout_s and
                  dist >= self.execution_stop_tolerance):
                speed = float(np.linalg.norm(self.world_vel[:2]))
                if not self.baseline and self.runtime_blocking_replan_enabled:
                    fallback, did_replan = self._maybe_replan_corridor(
                        corridor, now, "no_progress_timeout", force=True,
                        deadline=run_deadline)
                    if fallback is not None and did_replan:
                        corridor = fallback
                        last_progress_time = now
                        last_replan_time = now
                        last_replan_dist = dist
                        replan_progress_time = now
                        rospy.logwarn(
                            "[wc] no_progress timeout avoided with fallback corridor=%s dist=%.3f best=%.3f",
                            self._corridor_id(corridor), dist, best_dist)
                        continue
                elif not self.baseline:
                    switched, did_switch = self._switch_to_ranked_topology_candidate(
                        corridor, "no_progress_timeout")
                    if switched is not None and did_switch:
                        corridor = switched
                        last_progress_time = now
                        last_replan_time = now
                        last_replan_dist = dist
                        replan_progress_time = now
                        rospy.logwarn(
                            "[wc] no_progress timeout avoided by topology candidate switch corridor=%s dist=%.3f best=%.3f",
                            self._corridor_id(corridor), dist, best_dist)
                        continue
                self._publish_runtime_stop("timeout:no_progress")
                rospy.logwarn(
                    "[wc] no progress timeout, dist=%.3f best=%.3f "
                    "corridor=%s ref_idx=%s next_idx=%s ref_dist=%.3f "
                    "next_dist=%.3f speed=%.3f gate_scale=%.3f phi=%.3f",
                    dist, best_dist,
                    str(getattr(corridor, "corridor_id", "")),
                    str(getattr(corridor, "reference_index", "")),
                    str(getattr(corridor, "reference_next_index", "")),
                    float(getattr(corridor, "reference_distance", 0.0)),
                    float(getattr(corridor, "reference_next_distance", 0.0)),
                    speed,
                    float(getattr(gate, "scale", 0.0)),
                    float(getattr(gate, "risk", 0.0)))
                break
            completion_radius = (
                self.goal_tolerance if self.strict_goal_completion else
                self.completion_tolerance)
            if dist < completion_radius:
                self._publish_zero_command()
                self.u_prev = np.zeros(2)
                if near_goal_since is None:
                    near_goal_since = now
                elif (now - near_goal_since).to_sec() >= self.completion_hold_s:
                    self.task_completed = True
                    self.task_complete_pub.publish(Bool(True))
                    rospy.loginfo(
                        "[wc] completion tolerance reached, dist=%.3f "
                        "(goal_tolerance=%.3f, completion_tolerance=%.3f)",
                        dist, self.goal_tolerance, self.completion_tolerance)
                    break
                rate.sleep()
                continue
            else:
                near_goal_since = None
            if (not self.baseline):
                need_replan = False
                replan_reason = ""
                is_topology_corridor = self._corridor_is_topological(corridor)
                if ((not is_topology_corridor or self.topology_periodic_replan) and
                        (now - last_replan_time).to_sec() >= self.replan_period):
                    need_replan = True
                    replan_reason = "periodic"
                if dist < last_replan_dist - self.progress_eps:
                    last_replan_dist = dist
                    replan_progress_time = now
                elif ((now - replan_progress_time).to_sec() >=
                      self.no_progress_replan_time):
                    if (not is_topology_corridor or
                            self.topology_replan_on_no_progress):
                        if self.runtime_blocking_replan_enabled:
                            need_replan = True
                            replan_reason = "no_progress"
                        else:
                            switched, did_switch = (
                                self._switch_to_ranked_topology_candidate(
                                    corridor, "no_progress"))
                            if switched is not None and did_switch:
                                corridor = switched
                                last_replan_time = now
                                last_replan_dist = dist
                                replan_progress_time = now
                                last_progress_time = now
                                rospy.logwarn(
                                    "[wc] topology candidate switch on no_progress current=%s",
                                    self._corridor_label(corridor, ""))
                            else:
                                self.replan_deadline_skip_count += 1
                                replan_progress_time = now
                                rospy.logwarn_throttle(
                                    5.0,
                                    "[wc] skip runtime blocking replan reason=no_progress current=%s",
                                    self._corridor_label(corridor, ""))
                if corridor is not None:
                    _, d_tube = corridor.project(
                        np.array([self.state[0], self.state[1], 0.0]))
                    if d_tube > corridor.radius + self.replan_tube_margin:
                        if (not is_topology_corridor or
                                self.topology_replan_on_tube_exit):
                            need_replan = True
                            replan_reason = "tube_exit"
                if need_replan:
                    emergency_replan = False
                    if replan_reason == "tube_exit" and interest_eval is not None:
                        ip_risk = float(interest_eval.get(
                            "risk_gate", interest_eval.get("phi_max", 0.0)))
                        emergency_replan = (
                            bool(interest_eval.get("forbidden_hit", False)) or
                            ip_risk >= float(self.footprint_gate.rho_warn))
                    corridor, did_replan = self._maybe_replan_corridor(
                        corridor, now, replan_reason, force=emergency_replan,
                        deadline=run_deadline)
                    last_replan_time = now
                    last_replan_dist = dist
                    replan_progress_time = now
                    if did_replan:
                        last_progress_time = now
            corridor = self._ensure_corridor_runtime_contract(
                corridor,
                fallback_id="wheelchair_runtime_c%04d" % (
                    self.runtime_replan_fallback_count + 1),
                fallback_source="runtime_replan_fallback")
            if corridor is None:
                corridor = self._runtime_replan_fallback_corridor(
                    corridor, "pre_mpc_invalid_corridor",
                    RuntimeError("invalid runtime corridor before MPC"))
            if corridor is None:
                self._publish_runtime_stop("timeout:no_valid_corridor")
                rospy.logerr("[wc] stopping: no valid corridor before MPC")
                break
            self.selected_corridor = corridor
            self.execution_corridor = corridor
            self._apply_corridor_execution_profile(corridor)
            final_approach_active = self._apply_final_approach_profile(dist)
            ref = self._horizon_ref(corridor)
            topology_constraint_for_mpc = {}
            try:
                _ti, _ci, _mi, topology_constraint_for_mpc = (
                    build_mpc_constraint_inputs(
                        corridor, self.manifold, ref,
                        safe_threshold=float(self.manifold.rho),
                        minimum_clearance=0.10,
                        phase="navigation", robot_type="wheelchair",
                        manifold_constraint_mode=self.manifold_constraint_mode,
                        strict_stsm=bool(not self.baseline),
                        expected_corridor_id=self._corridor_id(corridor)))
            except Exception as exc:
                if not self.baseline:
                    self._publish_runtime_stop(
                        "planning:invalid_corridor_contract:%s" % exc)
                    rospy.logerr("[wc] invalid STSM corridor before MPC: %s", exc)
                    break
                topology_constraint_for_mpc = {}
            v, w = self.mpc.solve(
                self.state, ref, self.field,
                corridor=corridor, u_prev=self.u_prev,
                critic=self.adp_critic if self.adp_enabled else None,
                feature_builder=self.adp_features,
                lambda_adp_terminal=(
                    self.lambda_adp_terminal if self.adp_enabled else 0.0),
                goal=self.goal,
                gate_info={
                    "state": gate.state,
                    "stop": gate.stop,
                    "rho_warn": self.gate.rho_warn,
                },
                interest_risk=interest_eval or {},
                use_adp_terminal=(
                    self.adp_enabled and self.mpc_use_adp_terminal),
                interest_constraints={
                    "enabled": bool(safety_predictive_enabled),
                    "local_points": self.wc_local_points,
                    "labels": self.wc_ip_labels,
                    "rho": self.footprint_gate.rho_stop,
                },
                topology_constraint=topology_constraint_for_mpc,
                predictive=bool(not self.baseline))
            runtime_record = {
                "solve_index": int(len(self.mpc_runtime_records)),
                "corridor_id": self._corridor_id(corridor),
                "solver_status": str(self.mpc.last_solver_status),
                "horizon": int(self.mpc.N),
                "predicted_states": list(self.mpc.last_predicted_states),
                "predicted_controls": list(self.mpc.last_predicted_controls),
                "objective_terms": dict(self.mpc.last_objective_terms),
                "constraint_violation": dict(
                    self.mpc.last_constraint_violation),
                "control_sequence_varies": bool(
                    self.mpc.last_control_sequence_varies),
                "sequence_progress": float(self.mpc.last_sequence_progress),
                "heading_improvement": float(
                    self.mpc.last_heading_improvement),
                "alignment_translation": float(
                    self.mpc.last_alignment_translation),
                "replan_deadline_skip_count": int(
                    self.replan_deadline_skip_count),
                "topology_runtime_candidate_switch_count": int(
                    self.runtime_topology_candidate_switch_count),
                "topology_runtime_candidate_switch": bool(getattr(
                    corridor, "runtime_topology_candidate_switch", False)),
                "runtime_topology_previous_corridor_id": str(getattr(
                    corridor, "runtime_topology_previous_corridor_id", "")),
                "last_corridor_plan_duration_s": float(
                    self.last_corridor_plan_duration_s),
                "first_control": [float(v), float(w)],
            }
            self.mpc_runtime_records.append(runtime_record)
            if len(self.mpc_runtime_records) > 200:
                self.mpc_runtime_records = self.mpc_runtime_records[-200:]
            if (not self.baseline and
                    str(self.mpc.last_solver_status).startswith("safe_stop:")):
                self._publish_runtime_stop(
                    "mpc:%s" % self.mpc.last_solver_status)
                rospy.logerr(
                    "[wc][mpc] no feasible predictive sequence: %s",
                    self.mpc.last_solver_status)
                break
            v, w = self._baseline_corridor_follow_control(
                corridor, ref, v, w, dist)
            v_mpc_raw = float(v)
            w_mpc_raw = float(w)
            final_override_active = False
            stsm_final_radius = max(
                float(self.goal_tolerance),
                float(self.final_direct_override_radius))
            if (self.baseline and self.final_direct_override_enabled and
                    final_approach_active and dist < stsm_final_radius and
                    not gate.stop):
                if self._corridor_is_topological(corridor):
                    v_goal, w_goal = self._direct_goal_control(dist)
                    heading_error = abs(self._goal_heading_error())
                    if heading_error >= self.final_heading_threshold:
                        v_goal = 0.0
                    else:
                        v_goal = min(float(v_goal), 0.45 * float(dist))
                else:
                    v_goal, w_goal = self.mpc._goal_seek_u(self.state, self.goal)
                v, w = float(v_goal), float(w_goal)
                final_override_active = True
                self.mpc.last_final_approach_used = 1
                self.mpc.last_solver_status = "final_direct_override"
            self._publish_adp_mpc_info(corridor)
            gate_scale = gate.scale
            adp_scale = self._adp_scale(adp_value)
            if final_override_active:
                adp_scale = 1.0
            v *= gate_scale
            v *= adp_scale
            w *= max(gate_scale, 0.5)
            w *= max(adp_scale, 0.5)
            if not self.baseline:
                w_limit = self.stsm_w_max
                if final_approach_active:
                    blend = np.clip(
                        (self.final_approach_entry_radius - float(dist)) /
                        max(self.final_approach_entry_radius - self.goal_tolerance, 1e-6),
                        0.0, 1.0)
                    w_limit = (
                        (1.0 - float(blend)) * self.stsm_w_max +
                        float(blend) * self.final_w_max)
                w = float(np.clip(w, -w_limit, w_limit))
                prev_w = float(self.u_prev[1]) if self.u_prev is not None else 0.0
                dw = float(np.clip(
                    w - prev_w, -self.w_slew_limit, self.w_slew_limit))
                w = prev_w + dw
            if (final_override_active and not gate.stop and
                    dist > self.goal_tolerance and v > 0.0):
                v = max(v, 0.8 * self.final_creep_v)
            progress_stale_s = float((now - last_progress_time).to_sec())
            measured_speed = float(np.linalg.norm(self.world_vel[:2]))
            (v, w, progress_floor_used, progress_floor_value,
             liveness_active, liveness_w_limit) = (
                self._apply_stsm_progress_floor(
                    corridor, ref, v, w, dist, gate, adp_scale,
                    interest_eval=interest_eval,
                    progress_stale_s=progress_stale_s,
                    measured_speed=measured_speed))
            runtime_record["published_control"] = [float(v), float(w)]
            runtime_record["gate_scale"] = float(gate_scale)
            runtime_record["adp_scale"] = float(adp_scale)
            runtime_record["progress_stale_s"] = float(progress_stale_s)
            runtime_record["measured_speed"] = float(measured_speed)
            runtime_record["stsm_progress_floor_used"] = bool(
                progress_floor_used)
            runtime_record["stsm_progress_floor_value"] = float(
                progress_floor_value)
            runtime_record["stsm_liveness_active"] = bool(liveness_active)
            runtime_record["stsm_liveness_w_limit"] = float(liveness_w_limit)
            self._record_baseline_mpc_output(
                corridor, ref, v_mpc_raw, w_mpc_raw, v, w, gate, adp_scale,
                topology_constraint_for_mpc)
            self.u_prev = np.array([v, w])
            tw = Twist()
            tw.linear.x = v
            tw.angular.z = w
            self._set_command_keepalive(v, w, active=(not gate.stop))
            self.cmd_pub.publish(tw)
            rospy.loginfo_throttle(
                1.0,
                "[wc] pos=(%.2f, %.2f, %.2f) dist=%.3f cmd=(v=%.3f, w=%.3f)",
                self.state[0], self.state[1], self.state[2], dist, v, w)
            rate.sleep()
        self.task_complete_pub.publish(Bool(bool(self.task_completed)))
        self._publish_zero_command()
        if keepalive_timer is not None:
            keepalive_timer.shutdown()
        self._write_runtime_evidence()
        rospy.loginfo("[wc] done (stop=%s, reason=%s)",
                      self.stop_triggered, self.stop_reason)

if __name__ == "__main__":
    try:
        WheelchairNode().run()
    except rospy.ROSInterruptException:
        pass
