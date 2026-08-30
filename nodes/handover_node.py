#!/usr/bin/env python
import csv
import json
import sys
sys.dont_write_bytecode = True
import os
import numpy as np
import rospy
from std_msgs.msg import Float64, Float64MultiArray, Int32, String
from geometry_msgs.msg import PointStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import moveit_commander

PACKAGE_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.isdir(os.path.join(PACKAGE_SRC, "stsm_madp")) and PACKAGE_SRC not in sys.path:
    sys.path.insert(0, PACKAGE_SRC)

from stsm_madp.social_field import HumanState, SemanticAnchor, SocialField, SocialFieldParams
from stsm_madp.manifold import SafetyManifold, Corridor
from stsm_madp.corridor import require_corridor_contract
from stsm_madp.deform import (
    deform_trajectory, interpolate_by_segments, path_length,
    protected_waypoint_distances, topology_preserving_shortcut)
from stsm_madp.mpc import (
    ArmMPC, build_mpc_constraint_inputs, generate_topology_tube,
    run_mpc_tracking, write_mpc_outputs)
from stsm_madp.topology_constraint import write_topology_constraint
from stsm_madp.safety_gate import SafetyGate, SafetyGateResult
from stsm_madp.adp import (
    ADPCritic, ADPFeatureBuilder, ADPTransitionLearner,
    adp_ranking_adjustments, adp_role_from_runtime, clone_critic,
    candidate_feature_values, require_feature_schema, save_and_verify_critic,
    validate_critic_runtime_identity, evaluate_promotion_gate, apply_critic_lineage)
from stsm_madp.topology import topology_param_or_auto, topology_profile_defaults
from stsm_madp.topology_refinement import (
    check_refinement_manifold_validity, refine_topology_path)
from stsm_madp.decision_trace import trace_from_debug, write_trace
from stsm_madp.topology_diagnostics_writer import write_failed_topology_diagnostics
from stsm_madp.task_config import resolve_task_mode, resolve_task_weight
from stsm_madp.task_semantics import infer_task_context

JOINTS = ["elfin_joint%d" % i for i in range(1, 7)]

def _pt(d):
    return np.array(d, float)

def _topology_param(value, cast=float):
    value = topology_param_or_auto(value)
    if value is None:
        return None
    return cast(value)

def _effective(value, default):
    return default if value is None else value

def _resolve_manifold_constraint_mode(config, node_name="stsm"):
    config = dict(config or {})
    mode_a = str(config.get("manifold_constraint_mode", "soft")).strip().lower()
    mode_b = str(config.get("mpc_manifold_constraint_mode", mode_a)).strip().lower()
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

class HandoverNode:
    def __init__(self):
        rospy.init_node("stsm_handover")
        self.baseline = rospy.get_param("~baseline", False)
        self.experiment_mode = str(rospy.get_param(
            "~experiment_mode", "paper")).strip().lower()
        if self.experiment_mode not in ("debug", "paper"):
            self.experiment_mode = "paper"
        self.scene = rospy.get_param("~scene", {})
        self.mpc_config = dict(self.scene.get("mpc", {}) or {})
        try:
            self.manifold_constraint_mode = _resolve_manifold_constraint_mode(
                self.mpc_config, "stsm_handover")
        except ValueError as exc:
            rospy.logfatal(str(exc))
            raise
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.group = moveit_commander.MoveGroupCommander("elfin_arm")
        self.group.set_max_velocity_scaling_factor(0.2)
        self.group.set_max_acceleration_scaling_factor(0.2)
        self.cmd_pub = rospy.Publisher(
            "/elfin_arm_controller/command", JointTrajectory, queue_size=1)
        self.ee_pub = rospy.Publisher("/stsm/ee_pose", PointStamped, queue_size=10)
        self.phi_pub = rospy.Publisher("/stsm/phi_s", Float64, queue_size=10)
        self.risk_components_pub = rospy.Publisher(
            "/stsm/risk_components", Float64MultiArray, queue_size=10)
        self.velocity_monitor_pub = rospy.Publisher(
            "/stsm/velocity_monitor", Float64MultiArray, queue_size=10)
        self.gate_pub = rospy.Publisher(
            "/stsm/arm_gate_state", String, queue_size=10, latch=True)
        self.gate_info_pub = rospy.Publisher(
            "/stsm/arm_gate_info", Float64MultiArray, queue_size=10)
        self.gate_reason_pub = rospy.Publisher(
            "/stsm/arm_gate_reason", String, queue_size=10, latch=True)
        self.gate_source_pub = rospy.Publisher(
            "/stsm/arm_gate_source", String, queue_size=10, latch=True)
        self.arm_interest_pub = rospy.Publisher(
            "/stsm/arm_interest_risk", Float64MultiArray, queue_size=10)
        self.arm_interest_points_pub = rospy.Publisher(
            "/stsm/arm_interest_points", Float64MultiArray, queue_size=10)
        self.arm_interest_gate_info_pub = rospy.Publisher(
            "/stsm/arm_interest_gate_info", Float64MultiArray, queue_size=10)
        self.handover_status_pub = rospy.Publisher(
            "/stsm/arm_handover_status", Float64MultiArray, queue_size=10,
            latch=True)
        self.handover_event_pub = rospy.Publisher(
            "/stsm/arm_handover_event", String, queue_size=10, latch=True)
        self.phase_pub = rospy.Publisher(
            "/stsm/arm_phase", Int32, queue_size=1, latch=True)
        self.mode_pub = rospy.Publisher("/stsm/mode", String, queue_size=1, latch=True)
        self.phase = -1
        self.last_ee = None
        self.last_ee_time = None
        self.home_ee_ref = None
        self.ee_vel_filtered = np.zeros(3)
        self.arm_ip_last = {}
        self.arm_ip_last_time = None
        self.arm_ip_vel_filtered = {}
        self.arm_ip_warned_links = set()
        self.stop_triggered = False
        self.stop_reason = ""
        self.gripper_handover_event = False
        self.adp_requested = bool(rospy.get_param("~adp_enabled", True))
        self.adp_enabled = self.adp_requested and not self.baseline
        self.adp_model = rospy.get_param(
            "~adp_model",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                         "config", "adp_critic_arm_candidate_conditioned.yaml")))
        self.adp_expected_critic_path = rospy.get_param("~adp/expected_critic_path", "")
        self.adp_expected_critic_version = rospy.get_param("~adp/expected_critic_version", "")
        self.adp_expected_theta_hash = rospy.get_param("~adp/expected_theta_hash", "")
        self.adp_runtime_identity = {}
        self.lambda_adp = float(rospy.get_param("~lambda_adp", 0.005))
        self.lambda_adp_path = float(rospy.get_param(
            "~lambda_adp_path", self.lambda_adp))
        self.lambda_adp_arm = float(rospy.get_param(
            "~lambda_adp_arm", 0.008))
        self.adp_grad_eps = float(rospy.get_param("~adp_grad_eps", 0.01))
        self.adp_descent_gain = float(rospy.get_param(
            "~adp_descent_gain", 0.04))
        self.adp_grad_clip = float(rospy.get_param("~adp_grad_clip", 8.0))
        self.adp_solver_mode = rospy.get_param("~adp_solver_mode", "dls_adp")
        self.use_cvxpy = bool(rospy.get_param("~use_cvxpy", False))
        self.adp_blend_alpha = float(rospy.get_param(
            "~adp_blend_alpha", 0.08))
        self.adp_post_scale_enabled = bool(rospy.get_param(
            "~adp_post_scale_enabled", False))
        self.adp_min_scale = float(rospy.get_param("~adp_min_scale", 0.35))
        self.adp_debug = bool(rospy.get_param("~adp_debug", False))
        self.adp_critic = None
        self.adp_ranking_critic = None
        self.adp_features = ADPFeatureBuilder()
        self.adp_influence_enabled = False
        self.adp_decision_influence_enabled = False
        self.adp_ranking_influence_enabled = False
        self.adp_mpc_influence_enabled = False
        self.adp_learning = None
        self._adp_prev_ee = None
        self._adp_active_candidate_features = {}
        self._adp_active_candidate_missing = {}
        self.last_adp_value = 0.0
        self.path_adp_info = {
            "path_adp_mean": 0.0,
            "path_adp_max": 0.0,
            "path_adp_delta": 0.0,
            "adp_path_enabled": 0,
            "protected_saddle_count": 0,
            "protected_saddle_max_dist": 0.0,
            "protected_saddle_ok": 0,
            "mandatory_topology_node_count": 0,
            "mandatory_saddle_reached": 0,
            "mandatory_saddle_max_dist": 0.0,
            "corridor_violation_count": 0,
            "topology_tracking_error": 0.0,
            "mpc_segment_count": 0,
        }
        self.mandatory_topology_indices = []
        self.execution_corridor = None
        self.decision_trace_out = rospy.get_param("~decision_trace_out", "")
        self.mpc_reference_out = rospy.get_param("~mpc_reference_out", "")
        self.mpc_diagnostics_out = rospy.get_param("~mpc_diagnostics_out", "")
        self.mpc_cost_breakdown_out = rospy.get_param("~mpc_cost_breakdown_out", "")
        self.arm_handover_debug_out = rospy.get_param(
            "~arm_handover_debug_out", "")
        self.mpc_handover_diagnostics_out = rospy.get_param(
            "~mpc_handover_diagnostics_out", "")
        self.mpc_reference_records = []
        self.mpc_executed_records = []
        self.task_context = {}
        self.task_context_records = []
        self.corridor_evaluation_active = False
        self.task_completed = False
        self.arm_ee_debug_samples = []
        self.arm_handover_debug = {
            "state_machine": [],
            "handover_checks": [],
            "target_pose": {},
            "latest_ee_pose": {},
            "thresholds": {},
            "mpc_trajectory_difference": {},
        }
        self.mpc_handover_diagnostics = {
            "phase3_enter_count": 0,
            "records": [],
        }
        self.handover_protection_active = False
        self._mpc_reference_solve_index = 0
        self.adp_value_pub = rospy.Publisher(
            "/stsm/arm_adp_value", Float64, queue_size=10)
        self.adp_feature_pub = rospy.Publisher(
            "/stsm/adp_features", Float64MultiArray, queue_size=10)
        self.adp_status_pub = rospy.Publisher(
            "/stsm/adp_status", String, queue_size=10, latch=True)
        self.adp_path_info_pub = rospy.Publisher(
            "/stsm/arm_adp_path_info", Float64MultiArray, queue_size=10,
            latch=True)
        self.selected_corridor_pub = rospy.Publisher(
            "/stsm/arm_selected_corridor", String, queue_size=10, latch=True)
        self.topology_info_pub = rospy.Publisher(
            "/stsm/arm_topology_info", Float64MultiArray, queue_size=10,
            latch=True)
        self.adp_control_info_pub = rospy.Publisher(
            "/stsm/arm_adp_control_info", Float64MultiArray, queue_size=10)
        self._build_scene()
        self._load_adp()
        ip = self.scene.get("interest_points", {})
        self.arm_interest_enabled = bool(rospy.get_param(
            "~interest_points/enabled", ip.get("enabled", True)))
        self.arm_interest_gate_enabled = bool(rospy.get_param(
            "~interest_points/gate_enabled", ip.get("gate_enabled", True)))
        self.arm_interest_rho_warn = float(rospy.get_param(
            "~interest_points/rho_warn", ip.get("rho_warn", 3.5)))
        self.arm_interest_rho_stop = float(rospy.get_param(
            "~interest_points/rho_stop", ip.get("rho_stop", 6.0)))
        self.arm_interest_min_scale = float(rospy.get_param(
            "~interest_points/min_scale", ip.get("min_scale", 0.20)))
        self.arm_interest_gate = SafetyGate(
            rho_warn=self.arm_interest_rho_warn,
            rho_stop=self.arm_interest_rho_stop,
            min_scale=self.arm_interest_min_scale,
            enabled=self.arm_interest_gate_enabled)
        self.arm_interest_publish_points = bool(ip.get("publish_points", True))
        self.require_gripper_handover_event = bool(rospy.get_param(
            "~require_gripper_handover_event",
            self.scene.get("require_gripper_handover_event", False)))
        rospy.Subscriber("/stsm/arm_gripper_event", String,
                         self._gripper_event_cb)

    def _publish_path_adp_info(self):
        keys = [
            "path_adp_mean", "path_adp_max", "path_adp_delta",
            "adp_path_enabled", "protected_saddle_count",
            "protected_saddle_max_dist", "protected_saddle_ok",
            "mandatory_topology_node_count", "mandatory_saddle_reached",
            "mandatory_saddle_max_dist", "corridor_violation_count",
            "topology_tracking_error", "mpc_segment_count",
        ]
        self.adp_path_info_pub.publish(Float64MultiArray(data=[
            float(self.path_adp_info.get(key, 0.0)) for key in keys
        ]))

    def _publish_handover_status(self, goal_reached=False,
                                 pose_reached=False,
                                 complete=False,
                                 pos_err=float("inf"),
                                 orientation_err=float("inf"),
                                 stable_time=0.0):
        self.handover_status_pub.publish(Float64MultiArray(data=[
            1.0 if goal_reached else 0.0,
            1.0 if pose_reached else 0.0,
            1.0 if complete else 0.0,
            float(pos_err),
            float(orientation_err),
            float(stable_time),
            1.0 if self.gripper_handover_event else 0.0,
        ]))
        if complete:
            self.handover_event_pub.publish(String("handover_complete"))

    def _record_debug_state(self, event):
        try:
            self.arm_handover_debug.setdefault("state_machine", []).append({
                "t": float(rospy.Time.now().to_sec()),
                "event": str(event),
                "phase": int(self.phase),
                "stop_triggered": bool(self.stop_triggered),
                "stop_reason": str(self.stop_reason),
            })
        except Exception:
            pass

    def _record_debug_ee(self, ee):
        try:
            quat = self._ee_quat()
            sample = {
                "t": float(rospy.Time.now().to_sec()),
                "phase": int(self.phase),
                "position": [float(v) for v in np.asarray(ee, float)[:3]],
                "orientation_xyzw": [float(v) for v in quat],
                "speed_filtered": float(np.linalg.norm(self.ee_vel_filtered)),
            }
            self.arm_handover_debug["latest_ee_pose"] = sample
            self.arm_ee_debug_samples.append(sample)
            if len(self.arm_ee_debug_samples) > 2000:
                self.arm_ee_debug_samples = self.arm_ee_debug_samples[-2000:]
        except Exception:
            pass

    def _record_handover_check_debug(self, goal_reached, pose_reached,
                                     complete, pos_err, orientation_err,
                                     stable_time):
        try:
            self.arm_handover_debug.setdefault("handover_checks", []).append({
                "t": float(rospy.Time.now().to_sec()),
                "phase": int(self.phase),
                "goal_reached": bool(goal_reached),
                "handover_pose_reached": bool(pose_reached),
                "handover_complete": bool(complete),
                "position_error": float(pos_err),
                "orientation_error": float(orientation_err),
                "stable_time_s": float(stable_time),
                "speed_filtered": float(np.linalg.norm(self.ee_vel_filtered)),
                "gripper_handover_event": bool(self.gripper_handover_event),
            })
            checks = self.arm_handover_debug["handover_checks"]
            if len(checks) > 500:
                self.arm_handover_debug["handover_checks"] = checks[-500:]
        except Exception:
            pass

    def _verify_handover_complete(self, orientation_ref):
        try:
            self.arm_handover_debug["target_pose"] = {
                "position": [
                    float(v) for v in np.asarray(self.handover, float)[:3]],
                "orientation_xyzw": [
                    float(v) for v in np.asarray(orientation_ref, float)[:4]],
            }
        except Exception:
            pass
        stable_time = 0.0
        prev_t = rospy.Time.now().to_sec()
        deadline = prev_t + max(self.handover_verify_timeout_s, self.handover_hold_s)
        while not rospy.is_shutdown():
            ee = self._ee_pos()
            now = rospy.Time.now().to_sec()
            dt = max(0.0, now - prev_t)
            pos_err = float(np.linalg.norm(ee - self.handover))
            orientation_err = self._quat_error(self._ee_quat(), orientation_ref)
            speed = float(np.linalg.norm(self.ee_vel_filtered))
            goal_reached = bool(pos_err <= max(self.handover_pos_tol, 0.08))
            pose_reached = bool(
                pos_err <= self.handover_pos_tol and
                orientation_err <= self.handover_orientation_tol)
            event_ok = (
                self.gripper_handover_event or
                not self.require_gripper_handover_event)
            stable = bool(
                pose_reached and speed <= self.handover_speed_tol and event_ok)
            stable_time = stable_time + dt if stable else 0.0
            complete = bool(stable_time >= self.handover_hold_s)
            self._publish_metrics(ee)
            self._record_handover_check_debug(
                goal_reached, pose_reached, complete,
                pos_err, orientation_err, stable_time)
            self._publish_handover_status(
                goal_reached=goal_reached,
                pose_reached=pose_reached,
                complete=complete,
                pos_err=pos_err,
                orientation_err=orientation_err,
                stable_time=stable_time)
            if complete:
                rospy.loginfo(
                    "[handover] handover complete pos_err=%.4f orient_err=%.4f stable=%.2fs gripper=%d",
                    pos_err, orientation_err, stable_time,
                    int(self.gripper_handover_event))
                return True
            if now >= deadline:
                rospy.logwarn(
                    "[handover] handover verification failed pos_err=%.4f orient_err=%.4f speed=%.4f stable=%.2fs gripper=%d",
                    pos_err, orientation_err, speed, stable_time,
                    int(self.gripper_handover_event))
                return False
            prev_t = now
            rospy.sleep(max(0.02, min(0.1, self.handover_hold_s / 10.0)))

    def _load_adp(self):
        if not self.adp_enabled:
            reason = "baseline" if self.baseline else "parameter"
            self.adp_status_pub.publish(String("arm ADP disabled: %s" % reason))
            return
        try:
            self.adp_critic = ADPCritic.load_yaml(self.adp_model)
            require_feature_schema(self.adp_critic)
            self.adp_runtime_identity = validate_critic_runtime_identity(
                self.adp_critic, self.adp_model, self.adp_expected_critic_path,
                self.adp_expected_critic_version, self.adp_expected_theta_hash,
                robot_type="arm")
            if not self.adp_runtime_identity["validated"]:
                raise RuntimeError("adp_critic_identity_invalid:%s" % ",".join(
                    self.adp_runtime_identity["validation_reasons"]))
            self.adp_features = ADPFeatureBuilder(self.adp_critic.feature_names)
            self._configure_adp_learning()
            self.adp_status_pub.publish(String(
                "arm ADP loaded: %s" % self.adp_critic.critic_version))
            rospy.loginfo("[handover][adp] loaded %s (%s)",
                          self.adp_model, self.adp_critic.critic_version)
        except Exception as exc:
            if (self.adp_expected_critic_path or self.adp_expected_critic_version or
                    self.adp_expected_theta_hash):
                raise RuntimeError("formal_adp_seed_validation_failed:%s" % exc)
            self.adp_enabled = False
            self.adp_critic = None
            self.adp_status_pub.publish(String("arm ADP disabled: %s" % exc))
            rospy.logwarn("[handover][adp] cannot load %s: %s",
                          self.adp_model, exc)

    def _configure_adp_learning(self):
        config = dict(self.adp_critic.learning_config or {})
        for key in ("enabled", "decision_influence_enabled",
                    "ranking_influence_enabled", "mpc_influence_enabled",
                    "adp_value_normalization", "adp_norm_clip",
                    "adp_contribution_clip", "lambda_adp", "alpha",
                    "td_error_clip", "theta_delta_norm_max",
                    "min_transition_dt", "save_updated_critic",
                    "save_every_n_transitions", "risk_scale",
                    "failure_terminal_penalty", "value_outlier_z",
                    "promotion_td_clip_ratio_max", "promotion_td_error_abs_mean_max",
                    "promotion_theta_delta_norm_total_max",
                    "promotion_value_outlier_ratio_max", "promotion_auto_promote"):
            param = "~adp/" + key
            if rospy.has_param(param):
                config[key] = rospy.get_param(param)
        self.adp_decision_influence_enabled = bool(
            self.adp_enabled and config.get("decision_influence_enabled", False))
        self.adp_ranking_influence_enabled = bool(
            self.adp_decision_influence_enabled and
            config.get("ranking_influence_enabled", False))
        self.adp_mpc_influence_enabled = bool(
            self.adp_decision_influence_enabled and
            config.get("mpc_influence_enabled", False))
        # Keep pre-existing path/MPC ADP hooks disabled in ranking-only Phase 2.
        self.adp_influence_enabled = self.adp_mpc_influence_enabled
        config["enabled"] = bool(self.adp_enabled and config.get("enabled", True))
        self.adp_ranking_critic = clone_critic(self.adp_critic)
        self.adp_learning = ADPTransitionLearner(
            self.adp_critic, config=config, robot="arm")
        rospy.loginfo(
            "[handover][adp] learning=%s ranking=%s mpc=%s snapshot=%s",
            bool(config["enabled"]), self.adp_ranking_influence_enabled,
            self.adp_mpc_influence_enabled,
            bool(self.adp_ranking_critic is not None))

    def _adp_learning_features(self, ee, gate=None, interest_eval=None,
                               control=None):
        gate_info = {
            "state": gate.state if gate is not None else "NORMAL",
            "stop": gate.stop if gate is not None else False,
            "rho_warn": self.gate.rho_warn,
        }
        interest_eval = interest_eval or {}
        risk = {
            "phi_max": interest_eval.get("phi_max", 0.0),
            "phi_mean": interest_eval.get("phi_mean", 0.0),
        }
        target = (self.home_ee_ref if self._task_context_phase_name() == "return" and
                  self.home_ee_ref is not None else self.handover)
        return self.adp_features.build_arm(
            ee, target, self.field, gate_info=gate_info,
            interest_risk=risk, phase=self.phase,
            u=control, prev_ee_pos=self._adp_prev_ee,
            candidate_features=self._adp_active_candidate_features)

    @staticmethod
    def _adp_candidate_features(corridor):
        """Read ranking values already computed for a corridor; never resample risk."""
        raw = {}
        for name in ("path_length", "length_cost", "risk_mean", "risk_cost",
                     "mean_phi_on_path",
                     "risk_max", "max_risk", "max_phi_on_path", "min_clearance",
                     "clearance_value", "task_cost", "execution_cost",
                     "motion_cost", "smoothness_cost", "smooth_cost"):
            value = getattr(corridor, name, None)
            if value is not None:
                raw[name] = value
        if "max_phi_on_path" in raw and "risk_max" not in raw:
            raw["risk_max"] = raw["max_phi_on_path"]
        if "mean_phi_on_path" in raw and "risk_mean" not in raw:
            raw["risk_mean"] = raw["mean_phi_on_path"]
        if "smooth_cost" in raw and "smoothness_cost" not in raw:
            raw["smoothness_cost"] = raw["smooth_cost"]
        return candidate_feature_values(raw)

    def _record_adp_transition(self, ee, gate=None, interest_eval=None,
                               control=None, terminal=False):
        if self.adp_learning is None:
            return
        features = self._adp_learning_features(
            np.asarray(ee, float), gate=gate, interest_eval=interest_eval,
            control=control)
        _, candidate_missing = candidate_feature_values(
            self._adp_active_candidate_features)
        if self._adp_active_candidate_missing:
            candidate_missing = self._adp_active_candidate_missing
        self.adp_learning.observe(
            features, rospy.Time.now().to_sec(),
            task_state=self.task_context.get("task_state", ""),
            corridor_id=str(getattr(self.execution_corridor, "corridor_id", "")),
            control_effort=float(np.linalg.norm(
                np.asarray(control if control is not None else [], float))),
            terminal=terminal, success=bool(self.task_completed),
            failure_reason=self.stop_reason,
            feature_missing=candidate_missing)
        self._adp_prev_ee = np.asarray(ee, float).copy()

    def _write_adp_learning_diagnostics(self):
        if self.adp_learning is None or self.baseline:
            return
        base = os.path.dirname(
            self.mpc_reference_out or self.decision_trace_out or
            self.mpc_diagnostics_out or "")
        if not base:
            return
        if not os.path.isdir(base):
            os.makedirs(base)
        payload = self.adp_learning.diagnostics()
        payload["critic_source"] = self.adp_model
        payload["adp_runtime_identity"] = dict(self.adp_runtime_identity)
        payload["critic_saved"] = False
        payload["critic_reload_verified"] = False
        if (payload["learning_enabled"] and
                bool(self.adp_learning.config.get("save_updated_critic", True))):
            updated = os.path.join(base, "adp_critic_updated.yaml")
            try:
                payload["critic_reload_verified"] = save_and_verify_critic(
                    self.adp_critic, updated)
                payload["critic_saved"] = True
                payload["updated_critic_path"] = updated
            except Exception as exc:
                payload["critic_save_error"] = str(exc)
        promotion = evaluate_promotion_gate(
            payload, self.adp_critic, self.adp_model, robot_type="arm",
            execution_regression=not bool(self.task_completed),
            hard_safety_regression=bool(self.stop_triggered and any(
                token in str(self.stop_reason).lower()
                for token in ("safety", "forbidden", "gate"))),
            reload_verified=payload["critic_reload_verified"])
        promotion["updated_critic_path"] = payload.get("updated_critic_path", "")
        promotion["adp_cross_run_seed_source"] = self.adp_model
        if payload.get("critic_saved"):
            promotion["updated_identity"] = apply_critic_lineage(
                self.adp_critic, self.adp_runtime_identity, "arm", base,
                payload, promotion)
            payload["critic_reload_verified"] = save_and_verify_critic(
                self.adp_critic, payload["updated_critic_path"])
            promotion["reload_verified"] = payload["critic_reload_verified"]
        promotion["promoted_critic_path"] = ""
        if (promotion["promotion_passed"] and bool(
                self.adp_learning.config.get("promotion_auto_promote", False))):
            promoted = os.path.join(base, "adp_critic_promoted.yaml")
            promotion["promoted_reload_verified"] = save_and_verify_critic(
                self.adp_critic, promoted)
            if promotion["promoted_reload_verified"]:
                promotion["promoted_critic_path"] = promoted
        payload.update({
            "adp_learning_stable": bool(promotion["promotion_passed"]),
            "adp_promotion_candidate": bool(promotion["promotion_candidate"]),
            "adp_promotion_passed": bool(promotion["promotion_passed"]),
            "adp_cross_run_seed_source": self.adp_model,
            "adp_parent_critic_id": promotion["identity"]["parent_critic_id"],
            "adp_critic_generation": promotion["identity"]["critic_generation"],
        })
        with open(os.path.join(base, "adp_promotion_diagnostics.json"), "w") as handle:
            json.dump(promotion, handle, indent=2, sort_keys=True)
        path = os.path.join(base, "adp_learning_diagnostics.json")
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        rospy.loginfo("[handover][adp] wrote learning diagnostics %s", path)

    def _build_scene(self):
        p = self.scene.get("person", {})
        bp = {}
        for name, d in p.get("body_parts", {}).items():
            bp[name] = (_pt(d["pos"]), float(d["weight"]), float(d["sigma"]))
        self.human = HumanState(
            pos=_pt(p.get("ref_pos", [0.78, 0, 0.31])),
            heading=float(p.get("heading", np.pi)),
            posture=p.get("posture", "sitting"),
            vulnerability=float(p.get("vulnerability", 1.3)),
            body_parts=bp)
        anchors = []
        for _, a in self.scene.get("anchors", {}).items():
            anchors.append(SemanticAnchor(
                a["type"], _pt(a["center"]), _pt(a["half_extent"]),
                weight=float(a.get("weight", 1.0)),
                forbidden=bool(a.get("forbidden", False))))
        fc = self.scene.get("field", {})
        self.field = SocialField(SocialFieldParams(
            lam_prox=fc.get("lam_prox", 1.0), lam_close=fc.get("lam_close", 1.2),
            lam_dir=fc.get("lam_dir", 0.6), lam_body=fc.get("lam_body", 2.5),
            lam_env=fc.get("lam_env", 0.8), sigma_env=fc.get("sigma_env", 0.25)))
        self.field.set_scene([self.human], anchors)
        mc = self.scene.get("manifold", {})
        self.manifold = SafetyManifold(self.field, rho=mc.get("rho_ee", 2.0),
                                       lam_s=mc.get("lam_s", 1.0))
        tc = self.scene.get("topology", {})
        self.topology_profile = str(tc.get("profile", "arm")).strip().lower()
        profile_defaults = topology_profile_defaults(self.topology_profile)
        self.topology_enabled = bool(tc.get("enabled", True)) and not self.baseline
        self.topology_grid_resolution = _topology_param(
            tc.get("grid_resolution", None))
        self.topology_merge_radius = _topology_param(
            tc.get("merge_radius", None))
        self.topology_min_clearance = _topology_param(
            tc.get("min_clearance", None))
        self.topology_hard_clearance = _topology_param(
            tc.get("hard_clearance", None))
        self.topology_neighbor_k = _topology_param(
            tc.get("neighbor_k", None), int)
        self.topology_profile_defaults = profile_defaults
        self.topology_saddle_tie_ratio = float(tc.get("saddle_tie_ratio", 0.10))
        self.topology_morse_priority_ratio = float(
            tc.get("morse_priority_ratio", 0.25))
        self.topology_morse_saddle_priority_ratio = float(
            tc.get("morse_saddle_priority_ratio", 0.50))
        self.topology_morse_mix_priority_ratio = float(
            tc.get("morse_mix_priority_ratio", 0.50))
        self.topology_morse_minima_priority_ratio = float(
            tc.get("morse_minima_priority_ratio", 0.25))
        self.topology_morse_core_required = bool(
            tc.get("morse_core_required",
                   profile_defaults["morse_core_required"]))
        self.topology_morse_decision_mode = str(
            tc.get("morse_decision_mode", "balanced")).strip().lower()
        self.topology_morse_w_goal = float(tc.get("morse_w_goal", 0.25))
        self.topology_morse_w_social = float(tc.get("morse_w_social", 1.20))
        self.topology_morse_w_barrier = float(tc.get("morse_w_barrier", 0.80))
        self.topology_morse_grad_eps = float(tc.get("morse_grad_eps", 0.60))
        self.topology_k_paths = int(tc.get("k_paths", 3))
        self.topology_max_graph_nodes = int(tc.get("max_graph_nodes", 40))
        self.topology_safety_regions = list(tc.get("safety_regions", []))
        self.topology_allow_semantic_with_morse = bool(
            tc.get("allow_semantic_with_morse", False))
        self.topology_allow_semantic_topology_recovery = bool(
            tc.get("allow_semantic_topology_recovery", True))
        self.topology_allow_ring_with_morse = bool(
            tc.get("allow_ring_with_morse", False))
        self.topology_allow_graph_fallback_with_morse = bool(
            tc.get("allow_graph_fallback_with_morse", False))
        self.topology_lambda_execution = float(tc.get("lambda_execution", 0.20))
        self.topology_lambda_tracking = float(tc.get(
            "lambda_tracking", profile_defaults["lambda_tracking"]))
        self.topology_lambda_saddle_value = float(tc.get(
            "lambda_saddle_value", profile_defaults["lambda_saddle_value"]))
        self.topology_fallback_enabled = bool(tc.get("fallback_enabled", True))
        self.topology_lateral_margin = float(tc.get("lateral_margin", 0.35))
        self.topology_longitudinal_margin = float(
            tc.get("longitudinal_margin", 0.08))
        self.topology_corridor_radius = float(tc.get("corridor_radius", 0.08))
        self.topology_min_saddle_offset = float(
            tc.get("min_saddle_offset", 0.09))
        self.topology_goal_saddle_exclusion = float(
            tc.get("goal_saddle_exclusion", 0.08))
        self.topology_corridor_dedupe_distance = float(
            tc.get("corridor_dedupe_distance", 0.04))
        self.topology_candidate_pool_min = int(
            tc.get("candidate_pool_min", 3))
        self.topology_protected_tolerance = float(
            tc.get("protected_saddle_tolerance", 0.035))
        self.mandatory_topology_tolerance = float(
            tc.get("mandatory_topology_tolerance", 0.025))
        self.corridor_violation_gain = float(
            tc.get("corridor_violation_gain", 1.5))
        self.topology_max_corridor_turn = float(
            tc.get("max_corridor_turn", 2.09))
        self.topology_max_corridor_curvature = float(
            tc.get("max_corridor_curvature", 8.0))
        self.topology_min_segment_length = float(
            tc.get("min_segment_length", 0.01))
        self.topology_require_risk_improvement = bool(
            tc.get("require_risk_improvement", True))
        self.topology_candidate_max_risk = _topology_param(
            tc.get("candidate_max_risk", 6.0))
        self.topology_corridor_score_weights = dict(
            tc.get("corridor_score_weights", {}))
        self.task_config = dict(rospy.get_param(
            "~task_config", self.scene.get("task_config", {})) or {})
        self.task_mode = resolve_task_mode(rospy.get_param(
            "~task_mode", self.scene.get("task_mode", "handover")),
            robot_type="arm")
        self.task_weight = resolve_task_weight(
            self.task_mode, task_config=self.task_config,
            task_weight=rospy.get_param("~task_weight", {}),
            robot_type="arm")
        self.topology_refinement_enabled = bool(
            tc.get("refinement_enabled", True))
        self.topology_refinement_samples = int(
            tc.get("refinement_samples_per_segment", 10))
        self.grasp = _pt(self.scene.get("grasp_pose", [0.34, 0.16, 0.05]))
        self.handover = _pt(self.scene.get("handover_pose", [0.42, 0.0, 0.21]))
        self.wait = _pt(self.scene.get("wait_pose", [0.30, 0.0, 0.30]))
        self.handover_minima = _pt(self.scene.get(
            "handover_minima_pose", [0.36, 0.05, 0.18]))
        mp = self.scene.get("mpc", {})
        manifold_cfg = dict(self.scene.get("manifold", {}) or {})
        self.manifold_phase_config = dict(
            self.scene.get(
                "manifold_phase_config",
                manifold_cfg.get("manifold_phase_config", {})) or {})
        self.mpc_cost_weights = dict(mp.get("weights", {}) or {})
        self.mpc_phase_cost_weights = dict(mp.get("phase_cost_weights", {}) or {})
        self.stsm_speed_scale = float(mp.get("stsm_speed_scale", 0.60))
        speed_scale = 1.0 if self.baseline else self.stsm_speed_scale
        joint_lower, joint_upper, joint_limit_source = (
            self._resolve_mpc_joint_limits(mp))
        self.arm_joint_limit_source = joint_limit_source
        self.mpc = ArmMPC(
            dq_max=mp.get("dq_max", 0.6),
            v_cap=mp.get("v_cap_far", 0.18) * speed_scale,
            adp_grad_clip=self.adp_grad_clip,
            horizon=mp.get("horizon", 6),
            beam_width=mp.get("beam_width", 10),
            ddq_max=mp.get("ddq_max", 1.2),
            joint_lower=joint_lower,
            joint_upper=joint_upper,
            phase_cost_weights=self.mpc_phase_cost_weights,
            min_terminal_progress_ratio=mp.get(
                "min_terminal_progress_ratio", 0.01),
            task_progress_tolerance=mp.get(
                "task_progress_tolerance", 1e-3))
        self.v_cap_far = mp.get("v_cap_far", 0.18) * speed_scale
        self.v_cap_near = mp.get("v_cap_near", 0.04) * speed_scale
        self.near_radius = mp.get("near_radius", 0.12)
        self.v_cap_body = mp.get("v_cap_body", 0.05) * speed_scale
        self.body_slow_radius = mp.get("body_slow_radius", 0.45)
        self.v_cap_risk = mp.get("v_cap_risk", 0.06) * speed_scale
        self.risk_slow_threshold = mp.get("risk_slow_threshold", 2.6)
        self.risk_stop_threshold = mp.get("risk_stop_threshold", 6.0)
        self.arm_servo_gain = float(mp.get("servo_gain", 1.35))
        self.arm_mpc_no_progress_cycles = max(
            1, int(mp.get("no_progress_cycles", 25)))
        self.arm_mpc_progress_epsilon = max(
            0.0, float(mp.get("progress_epsilon", 1e-4)))
        self.handover_tracking_weight = float(rospy.get_param(
            "~handover_tracking_weight",
            mp.get("handover_tracking_weight", 8.0)))
        self.handover_protect_radius = float(rospy.get_param(
            "~handover_protect_radius",
            mp.get("handover_protect_radius", 0.08)))
        self.path_shortcut_enabled = bool(mp.get("path_shortcut_enabled", True))
        self.path_shortcut_samples = int(mp.get("path_shortcut_samples", 8))
        self.path_shortcut_passes = int(mp.get("path_shortcut_passes", 2))
        self.path_shortcut_rho = float(mp.get("path_shortcut_rho", 6.0))
        sg = self.scene.get("safety_gate", {})
        self.gate = SafetyGate(
            rho_warn=sg.get("rho_warn", 3.5),
            rho_stop=sg.get("rho_stop", 6.0),
            min_scale=sg.get("min_scale", 0.15),
            enabled=sg.get("enabled", True))
        self.abort_on_stop = bool(sg.get("abort_on_stop", True))
        self.hold_dt = float(sg.get("hold_dt", 0.2))
        self.task_complete_pos_tol = float(rospy.get_param(
            "~task_complete_pos_tol", sg.get("task_complete_pos_tol", 0.02)))
        self.task_complete_speed_tol = float(rospy.get_param(
            "~task_complete_speed_tol", sg.get("task_complete_speed_tol", 0.006)))
        self.task_complete_stable_cycles = int(rospy.get_param(
            "~task_complete_stable_cycles",
            sg.get("task_complete_stable_cycles", 6)))
        self.task_complete_check_dt = float(rospy.get_param(
            "~task_complete_check_dt", sg.get("task_complete_check_dt", 0.1)))
        self.task_complete_max_cycles = int(rospy.get_param(
            "~task_complete_max_cycles", sg.get("task_complete_max_cycles", 60)))
        self.handover_pos_tol = float(rospy.get_param(
            "~handover_pos_tol", self.scene.get("handover_pos_tol", 0.035)))
        self.handover_orientation_tol = float(rospy.get_param(
            "~handover_orientation_tol",
            self.scene.get("handover_orientation_tol", 0.35)))
        self.handover_speed_tol = float(rospy.get_param(
            "~handover_speed_tol", self.scene.get("handover_speed_tol", 0.01)))
        self.handover_hold_s = float(rospy.get_param(
            "~handover_hold_s", self.scene.get("handover_hold_s", 1.0)))
        self.handover_verify_timeout_s = float(rospy.get_param(
            "~handover_verify_timeout_s",
            self.scene.get("handover_verify_timeout_s", 3.0)))

    def _cur_joints(self):
        return np.array(self.group.get_current_joint_values(), float)

    def _ee_pos(self):
        p = self.group.get_current_pose().pose.position
        return np.array([p.x, p.y, p.z], float)

    def _ee_quat(self):
        q = self.group.get_current_pose().pose.orientation
        quat = np.array([q.x, q.y, q.z, q.w], float)
        n = float(np.linalg.norm(quat))
        return quat / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])

    def _quat_error(self, q, q_ref):
        q = np.asarray(q, float)
        q_ref = np.asarray(q_ref, float)
        nq = float(np.linalg.norm(q))
        nr = float(np.linalg.norm(q_ref))
        if nq <= 1e-12 or nr <= 1e-12:
            return 0.0
        q = q / nq
        q_ref = q_ref / nr
        dot = abs(float(np.dot(q, q_ref)))
        dot = min(1.0, max(-1.0, dot))
        return float(2.0 * np.arccos(dot))

    def _gripper_event_cb(self, msg):
        event = str(msg.data or "").strip().lower()
        if event in ("handover", "handover_complete", "released",
                     "object_released", "opened", "open"):
            self.gripper_handover_event = True

    def _link_pos_safe(self, link_name):
        try:
            p = self.group.get_current_pose(link_name).pose.position
            return np.array([p.x, p.y, p.z], float), True
        except Exception as exc:
            if link_name not in self.arm_ip_warned_links:
                rospy.logwarn("[handover][arm_ip] cannot read link %s: %s",
                              link_name, exc)
                self.arm_ip_warned_links.add(link_name)
            return None, False

    def _arm_interest_points(self, ee):
        cfg = self.scene.get("interest_points", {})
        labels = ["ee"]
        points = [np.array(ee, float)]
        valid = [1.0]

        links = cfg.get("links", {})
        for label in ("wrist", "elbow"):
            item = links.get(label, {})
            if not bool(item.get("enabled", True)):
                continue
            link_name = item.get("link_name", "")
            if not link_name:
                continue
            p, ok = self._link_pos_safe(link_name)
            if ok:
                labels.append(label)
                points.append(p)
                valid.append(1.0)

        if bool(cfg.get("object", {}).get("enabled", True)):
            off = np.array(cfg.get("object_offset", [0.0, 0.0, -0.05]), float)
            labels.append("object")
            points.append(np.array(ee, float) + off)
            valid.append(1.0)

        return labels, points, valid

    def _arm_interest_offsets(self, ee):
        cfg = self.scene.get("interest_points", {})
        ee = np.array(ee, float)
        offsets = {"ee": np.zeros(3)}
        links = cfg.get("links", {})
        for label in ("wrist", "elbow"):
            item = links.get(label, {})
            if not bool(item.get("enabled", True)):
                continue
            link_name = item.get("link_name", "")
            if not link_name:
                continue
            p, ok = self._link_pos_safe(link_name)
            if ok:
                offsets[label] = np.array(p, float) - ee
        if bool(cfg.get("object", {}).get("enabled", True)):
            offsets["object"] = np.array(
                cfg.get("object_offset", [0.0, 0.0, -0.05]), float)
        return offsets

    def _arm_interest_velocities(self, labels, points):
        now = rospy.Time.now().to_sec()
        if self.arm_ip_last_time is None:
            self.arm_ip_last_time = now
            self.arm_ip_last = {
                label: np.array(point, float)
                for label, point in zip(labels, points)
            }
            return [np.zeros(3) for _ in labels]

        dt = now - self.arm_ip_last_time
        vels = []
        if dt <= 1e-6 or dt > 0.5:
            vels = [np.zeros(3) for _ in labels]
        else:
            for label, point in zip(labels, points):
                point = np.array(point, float)
                prev = self.arm_ip_last.get(label, point)
                raw = (point - prev) / dt
                old = self.arm_ip_vel_filtered.get(label, raw)
                filt = 0.35 * raw + 0.65 * old
                self.arm_ip_vel_filtered[label] = filt
                vels.append(filt)

        self.arm_ip_last_time = now
        self.arm_ip_last = {
            label: np.array(point, float)
            for label, point in zip(labels, points)
        }
        return vels

    def _arm_interest_risk_eval(self, ee):
        if not self.arm_interest_enabled:
            return None

        labels, points, _ = self._arm_interest_points(ee)
        if not labels:
            return None
        vels = self._arm_interest_velocities(labels, points)

        label_order = ["ee", "wrist", "elbow", "object"]
        phi_map = {label: np.nan for label in label_order}
        phi_values = []
        valid_labels = []
        for label, point, vel in zip(labels, points, vels):
            phi = float(self.field.phi_s(point, vel))
            phi_map[label] = phi
            phi_values.append(phi)
            valid_labels.append(label)

        arr = np.array(phi_values, float)
        worst_local = int(np.nanargmax(arr)) if len(arr) else -1
        worst_label = valid_labels[worst_local] if worst_local >= 0 else ""
        worst_idx = {"ee": 0, "wrist": 1, "elbow": 2, "object": 3}.get(
            worst_label, -1)
        finite = [v for v in phi_map.values() if not np.isnan(float(v))]
        if finite:
            phi_max = float(np.max(finite))
            phi_mean = float(np.mean(finite))
            phi_sum = float(np.sum(finite))
        else:
            phi_max = phi_mean = phi_sum = float("nan")

        return {
            "labels": labels,
            "points": points,
            "phi_map": phi_map,
            "phi_max": phi_max,
            "phi_mean": phi_mean,
            "phi_sum": phi_sum,
            "worst_idx": worst_idx,
            "worst_label": worst_label,
            "valid_count": len(labels),
        }

    def _publish_arm_interest_risk(self, interest_eval):
        if interest_eval is None:
            return
        label_order = ["ee", "wrist", "elbow", "object"]
        labels = interest_eval["labels"]
        points = interest_eval["points"]
        phi_map = interest_eval["phi_map"]

        self.arm_interest_pub.publish(Float64MultiArray(data=[
            float(phi_map["ee"]),
            float(phi_map["wrist"]),
            float(phi_map["elbow"]),
            float(phi_map["object"]),
            float(interest_eval["phi_max"]),
            float(interest_eval["phi_mean"]),
            float(interest_eval["phi_sum"]),
            float(interest_eval["worst_idx"]),
            float(interest_eval["valid_count"]),
        ]))

        if not self.arm_interest_publish_points:
            return
        flat = []
        for label in label_order:
            if label in labels:
                p = points[labels.index(label)]
                flat.extend([float(p[0]), float(p[1]), float(p[2])])
            else:
                flat.extend([float("nan"), float("nan"), float("nan")])
        self.arm_interest_points_pub.publish(Float64MultiArray(data=flat))

    def _jacobian(self, q):
        J = np.array(self.group.get_jacobian_matrix(list(q)))
        return J[:3, :]

    def _resolve_mpc_joint_limits(self, config):
        lower = config.get("joint_lower")
        upper = config.get("joint_upper")
        if (lower is None) != (upper is None):
            raise ValueError(
                "mpc joint_lower and joint_upper must be configured together")
        if lower is not None:
            return list(lower), list(upper), "explicit_config"
        try:
            names = list(self.group.get_active_joints())
            bounds = [self.robot.get_joint(name).bounds() for name in names]
            if len(bounds) != 6 or any(len(item) != 2 for item in bounds):
                raise RuntimeError("expected six bounded active arm joints")
            return ([float(item[0]) for item in bounds],
                    [float(item[1]) for item in bounds],
                    "moveit_robot_model")
        except Exception as exc:
            if self.experiment_mode == "paper" and not self.baseline:
                raise RuntimeError(
                    "cannot load real MoveIt joint limits for STSM MPC: {}".format(
                        exc))
            rospy.logwarn(
                "[handover][mpc] joint limits unavailable outside paper STSM: %s",
                exc)
            return None, None, "unbounded_debug_fallback"

    def _send_joint(self, q, dt):
        msg = JointTrajectory()
        msg.joint_names = JOINTS
        pt = JointTrajectoryPoint()
        pt.positions = list(q)
        pt.time_from_start = rospy.Duration(dt)
        msg.points = [pt]
        self.cmd_pub.publish(msg)

    def _hold_current(self, q=None):
        if q is None:
            q = self._cur_joints()
        self._send_joint(q, self.hold_dt)

    def _combine_arm_gates(self, ee_gate, arm_gate=None, interest_eval=None):
        if arm_gate is None:
            reason = ee_gate.reason or ""
            if ee_gate.stop:
                reason = "ee:" + (reason or "risk_stop")
            elif ee_gate.state == "SLOW":
                reason = "ee:risk_slow"
            return SafetyGateResult(
                ee_gate.state, ee_gate.scale, ee_gate.stop,
                reason, ee_gate.risk), "ee" if ee_gate.state != "NORMAL" else "none"

        worst = "unknown"
        if interest_eval is not None:
            worst = interest_eval.get("worst_label", "unknown") or "unknown"

        if arm_gate.stop:
            return SafetyGateResult(
                "STOP", 0.0, True,
                "arm_interest:risk_stop:%s" % worst,
                arm_gate.risk), "arm_interest"

        if ee_gate.stop:
            return SafetyGateResult(
                "STOP", 0.0, True,
                "ee:%s" % (ee_gate.reason or "risk_stop"),
                ee_gate.risk), "ee"

        if ee_gate.state == "SLOW" or arm_gate.state == "SLOW":
            scale = min(ee_gate.scale, arm_gate.scale)
            if ee_gate.state == "SLOW" and arm_gate.state == "SLOW":
                return SafetyGateResult(
                    "SLOW", scale, False,
                    "combined:risk_slow:%s" % worst,
                    max(ee_gate.risk, arm_gate.risk)), "combined"
            if arm_gate.state == "SLOW":
                return SafetyGateResult(
                    "SLOW", scale, False,
                    "arm_interest:risk_slow:%s" % worst,
                    arm_gate.risk), "arm_interest"
            return SafetyGateResult(
                "SLOW", scale, False, "ee:risk_slow",
                ee_gate.risk), "ee"

        return SafetyGateResult("NORMAL", 1.0, False, "", ee_gate.risk), "none"

    def _publish_gate(self, gate, gate_source="none", arm_gate=None,
                      interest_eval=None):
        self.gate_pub.publish(String(gate.state))
        self.gate_reason_pub.publish(String(gate.reason))
        self.gate_source_pub.publish(String(gate_source))
        self.gate_info_pub.publish(Float64MultiArray(data=[
            float(gate.risk),
            float(gate.scale),
            1.0 if gate.stop else 0.0,
            float(self.gate.rho_warn),
            float(self.gate.rho_stop),
        ]))
        arm_risk = 0.0
        arm_scale = 1.0
        arm_stop = 0.0
        arm_slow = 0.0
        worst_idx = -1.0
        if interest_eval is not None:
            arm_risk = float(interest_eval.get("phi_max", 0.0))
            worst_idx = float(interest_eval.get("worst_idx", -1))
        if arm_gate is not None:
            arm_scale = float(arm_gate.scale)
            arm_stop = 1.0 if arm_gate.stop else 0.0
            arm_slow = 1.0 if arm_gate.state == "SLOW" else 0.0
        self.arm_interest_gate_info_pub.publish(Float64MultiArray(data=[
            1.0 if self.arm_interest_gate_enabled else 0.0,
            float(arm_risk),
            float(arm_scale),
            float(arm_stop),
            float(self.arm_interest_gate.rho_warn),
            float(self.arm_interest_gate.rho_stop),
            float(arm_slow),
            float(worst_idx),
        ]))

    def _publish_adp_value(self, ee, gate=None, interest_eval=None, dq=None):
        if not self.adp_enabled or self.adp_critic is None:
            self.last_adp_value = 0.0
            self.adp_value_pub.publish(Float64(0.0))
            return 0.0
        risk = {}
        if interest_eval is not None:
            risk = {
                "phi_max": interest_eval.get("phi_max", 0.0),
                "phi_mean": interest_eval.get("phi_mean", 0.0),
            }
        gate_info = {
            "state": gate.state if gate is not None else "NORMAL",
            "stop": gate.stop if gate is not None else False,
            "rho_warn": self.gate.rho_warn,
        }
        features = self.adp_features.build_arm(
            ee, self.handover, self.field, gate_info=gate_info,
            interest_risk=risk, phase=self.phase, u=dq)
        value = self.adp_critic.predict(features)
        self.last_adp_value = value
        self.adp_value_pub.publish(Float64(value))
        if self.adp_debug:
            self.adp_feature_pub.publish(Float64MultiArray(data=[
                float(features.get(name, 0.0))
                for name in self.adp_critic.feature_names
            ]))
            rospy.loginfo_throttle(
                2.0, "[handover][adp] value=%.3f lambda=%.3f",
                value, self.lambda_adp)
        return value

    def _adp_scale(self, value):
        if (not self.adp_post_scale_enabled or
                not self.adp_influence_enabled or self.adp_critic is None):
            return 1.0
        clipped = max(0.0, min(float(value), self.adp_critic.clip_value))
        scale = 1.0 / (1.0 + self.lambda_adp * clipped)
        return max(self.adp_min_scale, scale)

    def _publish_metrics(self, ee):
        now = rospy.Time.now().to_sec()
        vel_raw = np.zeros(3)
        dt_used = 0.0
        velocity_valid = 0.0
        if self.last_ee is not None and self.last_ee_time is not None:
            dt = now - self.last_ee_time
            if 0.0 < dt <= 0.5:
                vel_raw = (ee - self.last_ee) / dt
                dt_used = dt
                velocity_valid = 1.0
                alpha = 0.35
                self.ee_vel_filtered = (
                    alpha * vel_raw + (1.0 - alpha) * self.ee_vel_filtered)
        self.last_ee = np.array(ee, float)
        self.last_ee_time = now
        phase_name = (
            "handover" if int(self.phase) == 3 else
            "return" if int(self.phase) == 4 else "approach")
        self.mpc_executed_records.append({
            "point": [float(ee[0]), float(ee[1]), float(ee[2])],
            "phase": phase_name,
            "corridor_active": bool(self.corridor_evaluation_active),
        })

        comp = self.field.risk_components(ee)
        phi_close_monitor = self.field.phi_close_monitor(
            ee, self.ee_vel_filtered if velocity_valid else np.zeros(3))
        self.velocity_monitor_pub.publish(Float64MultiArray(data=[
            float(vel_raw[0]),
            float(vel_raw[1]),
            float(vel_raw[2]),
            float(np.linalg.norm(vel_raw)),
            float(np.linalg.norm(self.ee_vel_filtered)),
            float(phi_close_monitor),
            float(dt_used),
            float(velocity_valid),
        ]))
        self.risk_components_pub.publish(Float64MultiArray(data=[
            comp["phi_prox"],
            comp["phi_close"],
            comp["phi_dir"],
            comp["phi_body"],
            comp["phi_env"],
            comp["phi_total"],
        ]))
        interest_eval = self._arm_interest_risk_eval(ee)
        self._publish_arm_interest_risk(interest_eval)
        self.phi_pub.publish(Float64(comp["phi_total"]))
        ps = PointStamped()
        ps.header.stamp = rospy.Time.now()
        ps.header.frame_id = "elfin_base_link"
        ps.point.x, ps.point.y, ps.point.z = ee
        self.ee_pub.publish(ps)
        self._record_debug_ee(ee)
        return interest_eval

    def _set_phase(self, phase):
        prev_phase = int(self.phase)
        self.phase = int(phase)
        if self.phase == 3 and prev_phase != 3:
            self.mpc_handover_diagnostics["phase3_enter_count"] = int(
                self.mpc_handover_diagnostics.get(
                    "phase3_enter_count", 0)) + 1
        self.phase_pub.publish(Int32(self.phase))
        self._update_task_context()
        self._record_debug_state("phase_set")

    def _task_context_phase_name(self):
        """Map the real handover state-machine phase for diagnostics only."""
        return {
            0: "approach",
            1: "align",
            2: "handover",
            3: "hold",
            4: "return",
        }.get(int(self.phase), "approach")

    def _update_task_context(self):
        phase = self._task_context_phase_name()
        phase_progress = float(np.clip(float(self.phase) / 4.0, 0.0, 1.0))
        ee = self._ee_pos()
        target = (
            self.home_ee_ref if phase == "return" and
            self.home_ee_ref is not None else self.handover)
        dist_to_goal = float(np.linalg.norm(
            np.asarray(ee, float) - np.asarray(target, float)))
        risk_ahead = float(self.field.phi_s(ee))
        previous = dict(self.task_context or {})
        context = infer_task_context(
            "arm", self.task_mode, phase=phase, progress=phase_progress,
            context={
                "dist_to_goal": dist_to_goal,
                "risk_ahead": risk_ahead,
                "obstacle_ahead": False,
                "near_narrow_passage": False,
                "near_critical_point": False,
            }, config=self.task_config)
        self.task_context = context
        self.field.set_task_context(context)
        self.task_context_records.append({
            "task_state": str(context.get("task_state", "")),
            "state_trigger": str(context.get("state_trigger", "")),
            "source": "runtime_task_context",
            "progress": context.get("progress", None),
            "dist_to_goal": context.get("dist_to_goal", None),
            "risk_ahead": context.get("risk_ahead", None),
            "obstacle_ahead": bool(context.get("obstacle_ahead", False)),
            "near_narrow_passage": bool(context.get(
                "near_narrow_passage", False)),
            "near_critical_point": bool(context.get(
                "near_critical_point", False)),
            "effective_social_weights": self.field.get_effective_weights(),
            "timestamp": float(context.get("timestamp", rospy.Time.now().to_sec())),
            "phase": phase,
            "current_phase": phase,
            "state_transition": "{}->{}".format(
                previous.get("task_state", "start"),
                context.get("task_state", "")),
        })
        return dict(context)

    def _phase3_handover_protect_active(self, target):
        if int(self.phase) != 3 or not bool(self.handover_protection_active):
            return False
        try:
            target = np.asarray(target, float)[:3]
            handover = np.asarray(self.handover, float)[:3]
            radius = max(float(self.handover_protect_radius),
                         float(self.handover_pos_tol) * 2.0)
            return bool(np.linalg.norm(target - handover) <= radius)
        except Exception:
            return False

    def _path_contains_handover_target(self, path):
        try:
            pts = np.asarray(path, float)
            if pts.size == 0:
                return False
            pts = pts.reshape((-1, pts.shape[-1]))[:, :3]
            handover = np.asarray(self.handover, float)[:3]
            radius = max(float(self.handover_protect_radius),
                         float(self.handover_pos_tol) * 2.0)
            return bool(np.min(np.linalg.norm(pts - handover, axis=1)) <= radius)
        except Exception:
            return False

    def _pose_list(self, pose):
        if pose is None:
            return []
        try:
            return [float(v) for v in np.asarray(pose, float)[:3]]
        except Exception:
            return []

    def _record_phase3_chain_event(self, event, stsm_reference_pose=None,
                                   mpc_output_pose=None, ee_actual_pose=None,
                                   extra=None):
        try:
            if int(self.phase) != 3:
                return
            item = {
                "t": float(rospy.Time.now().to_sec()),
                "event": str(event),
                "phase": int(self.phase),
                "stsm_reference_pose": self._pose_list(stsm_reference_pose),
                "mpc_output_pose": self._pose_list(mpc_output_pose),
                "ee_actual_pose": self._pose_list(ee_actual_pose),
            }
            if extra:
                item.update(dict(extra))
            chain = self.mpc_handover_diagnostics.setdefault(
                "phase3_execution_chain", [])
            chain.append(item)
            if len(chain) > 1000:
                self.mpc_handover_diagnostics[
                    "phase3_execution_chain"] = chain[-1000:]
        except Exception:
            pass

    def _record_mpc_handover_diagnostic(self, target, ee_before, J, dq, dt):
        info = dict(getattr(self.mpc, "last_handover_protection", {}) or {})
        if not bool(info.get("active", False)):
            return
        target = np.asarray(target, float)[:3]
        ee_before = np.asarray(ee_before, float)[:3]
        mpc_point = ee_before + np.dot(np.asarray(J, float), np.asarray(dq, float))[:3] * float(dt)
        record = {
            "t": float(rospy.Time.now().to_sec()),
            "phase": int(self.phase),
            "handover_reference_point": [float(v) for v in target],
            "mpc_output_point": [float(v) for v in mpc_point],
            "current_ee_point": [float(v) for v in ee_before],
            "guard_applied": bool(info.get("guard_applied", False)),
            "selected_output": str(info.get("selected_output", "")),
            "before_error": float(info.get(
                "before_error", np.linalg.norm(mpc_point - target))),
            "after_error": float(info.get(
                "after_error", np.linalg.norm(mpc_point - target))),
            "raw_mpc_output_point": info.get("raw_mpc_output_point", []),
            "protected_mpc_output_point": info.get(
                "protected_mpc_output_point", [float(v) for v in mpc_point]),
        }
        records = self.mpc_handover_diagnostics.setdefault("records", [])
        records.append(record)
        if len(records) > 1000:
            self.mpc_handover_diagnostics["records"] = records[-1000:]
        self._record_phase3_chain_event(
            "mpc_step",
            stsm_reference_pose=target,
            mpc_output_pose=mpc_point,
            ee_actual_pose=ee_before,
            extra={
                "guard_applied": record["guard_applied"],
                "selected_output": record["selected_output"],
                "before_error": record["before_error"],
                "after_error": record["after_error"],
                "raw_mpc_output_point": record["raw_mpc_output_point"],
                "protected_mpc_output_point": record[
                    "protected_mpc_output_point"],
            })

    def _ee_v_cap(self, ee):
        cap = self.v_cap_far

        d_goal = np.linalg.norm(ee - self.handover)
        if d_goal < self.near_radius:
            cap = min(cap, self.v_cap_near)

        body_distances = []
        for name in ("head", "chest"):
            if name in self.human.body_parts:
                p, _, _ = self.human.body_parts[name]
                body_distances.append(np.linalg.norm(ee - p))
        if body_distances and min(body_distances) < self.body_slow_radius:
            cap = min(cap, self.v_cap_body)

        risk = self.field.phi_s(ee)
        if risk > self.risk_slow_threshold:
            alpha = min(1.0, (risk - self.risk_slow_threshold) /
                        max(self.risk_stop_threshold - self.risk_slow_threshold, 1e-6))
            risk_cap = (1.0 - alpha) * self.v_cap_far + alpha * self.v_cap_risk
            cap = min(cap, risk_cap)

        return cap

    def _nominal_path(self, start, goal, n=30):
        start = np.asarray(start, float)
        goal = np.asarray(goal, float)
        alpha = np.linspace(0.0, 1.0, n)[:, None]
        return start[None, :] + alpha * (goal - start)[None, :]

    def _corridor_reference_path(self, corr, start, goal, n=30):
        cached = np.asarray(getattr(corr, "execution_reference_path", []), float)
        cached_endpoints = np.asarray(
            getattr(corr, "execution_reference_endpoints", []), float)
        requested_endpoints = np.asarray([start, goal], float)
        if (len(cached) >= 2 and cached_endpoints.shape == requested_endpoints.shape and
                np.allclose(cached_endpoints, requested_endpoints, atol=1e-9)):
            return cached.copy(), list(getattr(
                corr, "execution_reference_protected_indices", [0, len(cached) - 1]))
        candidate_raw_points = np.asarray(getattr(
            corr, "raw_topology_waypoints",
            getattr(corr, "centerline", getattr(corr, "waypoints", []))), float).copy()
        ordered = np.asarray(
            getattr(corr, "topology_ordered_waypoints", []), float)
        if len(ordered) >= 3:
            reference_points = [p for p in ordered[1:-1]]
        else:
            task_points = list(getattr(corr, "task_minima_waypoints", []))
            channel_points = list(getattr(corr, "channel_waypoints", []))
            reference_points = channel_points + task_points
        skeleton = [np.asarray(start, float)]
        for p in reference_points:
            arr = np.asarray(p, float)
            if arr.shape[0] >= 3:
                skeleton.append(arr[:3])
        skeleton.append(np.asarray(goal, float))
        path, protected_indices = interpolate_by_segments(skeleton, n=n)
        # The selected topology route includes the real task endpoints.  This is
        # the immutable execution tube; a later MPC reference must not redefine it.
        execution_tube_centerline = np.asarray(path, float).copy()
        corr.execution_tube_centerline = execution_tube_centerline
        corr.reference_skeleton = np.asarray(skeleton, float)
        mandatory = np.asarray(skeleton[1:-1], float)
        corr.protected_waypoints = mandatory
        if self.topology_refinement_enabled and not self.baseline:
            original = np.asarray(getattr(corr, "waypoints", []), float)
            corr.waypoints = np.asarray(skeleton, float)
            self._refinement_interest_offsets = self._arm_interest_offsets(start)
            ok, refined, metrics, reason = refine_topology_path(
                corr,
                samples_per_segment=self.topology_refinement_samples,
                max_curvature=self.topology_max_corridor_curvature,
                max_turn=self.topology_max_corridor_turn,
                footprint_checker=self._arm_refined_path_checker,
                corridor_constraint={
                    "centerline": execution_tube_centerline,
                    "radius": float(getattr(corr, "radius", 0.0)),
                })
            self._refinement_interest_offsets = None
            execution_feasibility = {
                "ik_valid": bool(getattr(corr, "ik_valid", False)),
                "collision_valid": bool(getattr(
                    corr, "link_collision_valid", True)) and not bool(getattr(
                        corr, "link_collision", False)),
                "manifold_valid": bool(metrics.get(
                    "refinement_manifold_valid", False)),
                "tube_valid": bool(metrics.get("refinement_tube_valid", False)),
            }
            corr.execution_reference_feasibility = dict(execution_feasibility)
            if ok and not all(execution_feasibility.values()):
                failed = [
                    key for key, valid in execution_feasibility.items()
                    if not valid]
                ok = False
                reason = "execution_reference_infeasible:" + ",".join(failed)
            if ok:
                path = np.asarray(refined, float)
                corr.waypoints = path.copy()
                corr.max_turn_angle = float(metrics.get("max_turn", 0.0))
                corr.mean_turn_angle = float(metrics.get("mean_turn", 0.0))
                corr.max_curvature = float(metrics.get("max_curvature", 0.0))
                corr.path_length = float(getattr(
                    corr, "refined_path_length", path_length(path)))
                corr.execution_cost = float(
                    corr.max_turn_angle + 0.2 * corr.max_curvature)
                corr.motion_cost = float(corr.execution_cost)
                fallback_used = bool(metrics.get("refinement_fallback", False))
                corr.refinement_used = int(not fallback_used)
                corr.final_reference_source = (
                    "topology_skeleton_validated" if fallback_used
                    else str(metrics.get("reference_source", "refined_waypoints")))
                dbg = dict(getattr(self.manifold, "last_topology_debug", {}) or {})
                selected_id = str(getattr(corr, "corridor_id", ""))
                dbg["selected_refinement_used"] = int(not fallback_used)
                dbg["selected_refined_path_length"] = float(
                    getattr(corr, "refined_path_length", corr.path_length))
                dbg["selected_topology_diversity"] = float(
                    getattr(corr, "topology_diversity", 0.0))
                dbg["selected_topology_class"] = str(
                    getattr(corr, "topology_class", ""))
                dbg["selected_topology_route_class"] = str(getattr(
                    corr, "topology_route_class",
                    getattr(corr, "topology_class", "")))
                dbg["selected_task_semantic_class"] = str(getattr(
                    corr, "task_semantic_class", ""))
                dbg["mpc_used"] = 1
                dbg["mpc_reference_source"] = str(
                    getattr(corr, "final_reference_source", "refined_waypoints"))
                dbg["final_path_source"] = "Morse->Candidate->Ranking->Refinement->MPC"
                dbg["adp_role"] = adp_role_from_runtime(
                    self.adp_enabled, bool(self.adp_learning and self.adp_learning.config.get("enabled", False)),
                    self.adp_decision_influence_enabled,
                    effective_lambda=(self.adp_learning.config.get("lambda_adp", 0.0)
                                      if self.adp_learning else 0.0),
                    ranking_contribution=self.adp_ranking_influence_enabled,
                    control_contribution=False)
                for item in dbg.get("candidate_corridors", []):
                    if str(item.get("corridor_id", "")) == selected_id:
                        item["selected"] = True
                        item["execution_corridor_id"] = selected_id
                        item["refinement_used"] = int(not fallback_used)
                        item["refined_waypoints"] = path.tolist()
                        item["refined_path_length"] = float(
                            getattr(corr, "refined_path_length", corr.path_length))
                        item["refined_max_turn_angle"] = float(corr.max_turn_angle)
                        item["refined_max_curvature"] = float(corr.max_curvature)
                        item["waypoints"] = path.tolist()
                    else:
                        item["selected"] = False
                        item["execution_corridor_id"] = ""
                self.manifold.last_topology_debug = dbg
                self._write_failed_topology_diagnostics("success")
            else:
                corr.refinement_used = 0
                corr.refinement_reject_reason = str(reason)
                rospy.logwarn("[handover][refine] reject %s reason=%s "
                              "max_curvature=%.4f limit=%.4f "
                              "max_turn=%.4f limit=%.4f length=%.4f",
                              getattr(corr, "corridor_id",
                                      getattr(corr, "label", "")),
                              reason,
                              float(metrics.get("max_curvature", 0.0)),
                              1.15 * float(self.topology_max_corridor_curvature),
                              float(metrics.get("max_turn", 0.0)),
                              1.15 * float(self.topology_max_corridor_turn),
                              float(path_length(np.asarray(refined, float))))
                raise RuntimeError(
                    "handover topology refinement rejected selected corridor: %s" %
                    str(reason))
        else:
            corr.waypoints = np.asarray(skeleton, float)
        protected_indices = [0, len(path) - 1]
        mandatory_indices = []
        for waypoint in mandatory:
            if len(path) == 0:
                continue
            idx = int(np.argmin(np.linalg.norm(
                path - waypoint[None, :], axis=1)))
            protected_indices.append(idx)
            mandatory_indices.append(idx)
        protected_indices = sorted(set(protected_indices))
        corr.protected_indices = protected_indices
        corr.mandatory_topology_indices = list(mandatory_indices)
        original_centerline = np.asarray(getattr(corr, "centerline", []), float)
        raw_status = check_refinement_manifold_validity(
            candidate_raw_points,
            corridor_constraint={"centerline": original_centerline,
                                 "radius": float(getattr(corr, "radius", 0.0))})
        fallback_status = check_refinement_manifold_validity(
            np.asarray(skeleton, float),
            corridor_constraint={"centerline": execution_tube_centerline,
                                 "radius": float(getattr(corr, "radius", 0.0))})
        reference_status = check_refinement_manifold_validity(
            path,
            corridor_constraint={"centerline": execution_tube_centerline,
                                 "radius": float(getattr(corr, "radius", 0.0))})
        corr.raw_candidate_corridor_violation_count = int(
            raw_status.get("corridor_violation_count", 0))
        corr.fallback_corridor_violation_count = int(
            fallback_status.get("corridor_violation_count", 0))
        corr.mpc_reference_corridor_violation_count = int(
            reference_status.get("corridor_violation_count", 0))
        corr.execution_reference_path = np.asarray(path, float).copy()
        corr.execution_reference_endpoints = requested_endpoints.copy()
        corr.execution_reference_protected_indices = list(protected_indices)
        return path, protected_indices

    def _sync_selected_corridor_debug(self, corr):
        execution_id = str(getattr(corr, "corridor_id", getattr(corr, "label", "")))
        dbg = dict(getattr(self.manifold, "last_topology_debug", {}) or {})
        dbg["selected_corridor_label"] = execution_id
        dbg["selected_corridor_id"] = execution_id
        dbg["execution_corridor_id"] = execution_id
        dbg["selected_refinement_used"] = int(getattr(corr, "refinement_used", 0))
        dbg["selected_refined_path_length"] = float(getattr(
            corr, "refined_path_length", getattr(corr, "path_length", 0.0)))
        dbg["selected_raw_waypoints_count"] = int(len(np.asarray(
            getattr(corr, "raw_topology_waypoints",
                    getattr(corr, "topology_ordered_waypoints", [])), float)))
        dbg["selected_refined_waypoints_count"] = int(len(np.asarray(
            getattr(corr, "refined_waypoints", []), float)))
        dbg["selected_topology_class"] = str(getattr(corr, "topology_class", ""))
        dbg["selected_topology_route_class"] = str(getattr(
            corr, "topology_route_class", getattr(corr, "topology_class", "")))
        dbg["selected_task_semantic_class"] = str(getattr(
            corr, "task_semantic_class", ""))
        dbg["selected_topology_diversity"] = float(getattr(
            corr, "topology_diversity", 0.0))
        dbg["risk_field_used"] = 1
        dbg["manifold_used"] = 1
        dbg["morse_used"] = 1
        dbg["topology_graph_used"] = 1
        dbg["candidate_corridor_used"] = 1
        dbg["candidate_ranking_used"] = 1
        dbg["fallback_used"] = 0
        dbg["mpc_used"] = 1
        reference_source = str(getattr(corr, "final_reference_source", "") or (
            "refined_waypoints"
            if int(getattr(corr, "refinement_used", 0)) == 1
            else "selected_candidate_waypoints"))
        dbg["mpc_reference_source"] = reference_source
        dbg["final_path_source"] = "Morse->Candidate->Ranking->Refinement->MPC"
        dbg["adp_role"] = adp_role_from_runtime(
            self.adp_enabled, bool(self.adp_learning and self.adp_learning.config.get("enabled", False)),
            self.adp_decision_influence_enabled,
            effective_lambda=(self.adp_learning.config.get("lambda_adp", 0.0)
                              if self.adp_learning else 0.0),
            ranking_contribution=self.adp_ranking_influence_enabled,
            control_contribution=False)
        for item in dbg.get("candidate_corridors", []):
            selected = str(item.get("corridor_id", "")) == execution_id
            item["selected"] = bool(selected)
            item["execution_corridor_id"] = execution_id if selected else ""
            if selected:
                dbg["selected_rank"] = int(item.get("rank", 0))
                item["refinement_used"] = int(getattr(corr, "refinement_used", 0))
                item["refined_path_length"] = float(getattr(
                    corr, "refined_path_length", getattr(corr, "path_length", 0.0)))
                item["refined_waypoints"] = np.asarray(
                    getattr(corr, "refined_waypoints", corr.waypoints),
                    float).tolist()
                item["waypoints"] = np.asarray(corr.waypoints, float).tolist()
        self.manifold.last_topology_debug = dbg
        self._publish_topology_info(True, False)

    def _arm_refined_path_checker(self, path):
        offsets = getattr(self, "_refinement_interest_offsets", None)
        if offsets is None:
            points = np.asarray(path, float)
            offsets = self._arm_interest_offsets(points[0]) if len(points) else {
                "ee": np.zeros(3)
            }
        if not self.arm_interest_enabled:
            offsets = {"ee": np.zeros(3)}
        for ee in np.asarray(path, float):
            for label, offset in offsets.items():
                point = np.asarray(ee, float) + np.asarray(offset, float)
                if self._point_forbidden(point):
                    return False, "refined_arm_forbidden"
                phi = float(self.field.phi_s(point))
                limit = (
                    float(self.gate.rho_stop)
                    if str(label) == "ee"
                    else float(self.arm_interest_rho_stop))
                if phi > limit + 1e-9:
                    if str(label) == "ee":
                        return False, "refined_arm_risk"
                    return False, "refined_arm_interest_risk"
        return True, ""

    def _deformed_path(self, start, goal):
        nominal = self._nominal_path(start, goal)
        if self.baseline:
            return nominal

        corr, source = self._handover_corridor(start, goal)
        self.execution_corridor = corr
        nominal, protected_indices = self._corridor_reference_path(
            corr, start, goal, n=30)
        require_corridor_contract(
            corr, reference_path=nominal,
            expected_corridor_id=str(getattr(corr, "corridor_id", "")),
            require_morse=True, require_tube=True)
        if int(getattr(corr, "refinement_used", 0)) == 1:
            mandatory_count = int(len(getattr(corr, "protected_waypoints", [])))
            mandatory_indices = [
                int(i) for i in protected_indices[1:1 + mandatory_count]
                if 0 <= int(i) < len(nominal)]
            corr.mandatory_topology_indices = list(mandatory_indices)
            self.path_adp_info = {
                "path_adp_mean": 0.0,
                "path_adp_max": 0.0,
                "path_adp_delta": 0.0,
                "adp_path_enabled": 0,
                "protected_saddle_count": int(len(getattr(corr, "protected_waypoints", []))),
                "protected_saddle_max_dist": 0.0,
                "protected_saddle_ok": 1,
                "mandatory_topology_node_count": int(len(mandatory_indices)),
                "mandatory_saddle_reached": 0,
                "mandatory_saddle_max_dist": 0.0,
                "corridor_violation_count": 0,
                "topology_tracking_error": 0.0,
                "mpc_segment_count": max(1, len(mandatory_indices) + 1),
            }
            self._publish_path_adp_info()
            execution_id = str(getattr(corr, "corridor_id", corr.label))
            self._sync_selected_corridor_debug(corr)
            self.selected_corridor_pub.publish(String(execution_id))
            rospy.loginfo(
                "[handover] selected refined corridor: %s label=%s (source=%s cost=%.3f)",
                execution_id, corr.label, source, corr.cost)
            self.mandatory_topology_indices = mandatory_indices
            return np.asarray(nominal, float)
        feature_context = {
            "target_pos": goal,
            "phase": self.phase,
            "gate_info": {
                "state": "NORMAL",
                "stop": False,
                "rho_warn": self.gate.rho_warn,
            },
            "interest_risk": {},
        }
        turn = float(getattr(corr, "max_turn_angle", 0.0))
        exec_cost = float(getattr(corr, "execution_cost", 0.0))
        difficulty = min(1.0, max(turn / 1.2, exec_cost / 2.0))
        path = deform_trajectory(
            nominal, self.field, corridor=corr,
            lam_social=0.10, lam_smooth=0.18 + 0.08 * difficulty,
            iters=80,
            critic=self.adp_critic if self.adp_influence_enabled else None,
            feature_builder=self.adp_features,
            lambda_adp_path=(
                self.lambda_adp_path if self.adp_influence_enabled else 0.0),
            feature_context=feature_context,
            protected_indices=protected_indices)
        deformed_length = path_length(path)
        if self.path_shortcut_enabled:
            path, protected_indices = topology_preserving_shortcut(
                path, protected_indices, self.field, corridor=corr,
                rho=self.path_shortcut_rho,
                samples=self.path_shortcut_samples,
                max_passes=self.path_shortcut_passes)
        shortcut_length = path_length(path)
        final_reference_status = check_refinement_manifold_validity(
            path,
            manifold_constraint={
                "risk_threshold": float(self.arm_interest_rho_stop),
                "minimum_clearance": 0.0,
            },
            corridor_constraint={
                "centerline": np.asarray(getattr(
                    corr, "execution_tube_centerline", nominal), float),
                "radius": float(getattr(corr, "radius", 0.0)),
            },
            risk_field=self.field)
        corr.mpc_reference_corridor_violation_count = int(
            final_reference_status.get("corridor_violation_count", 0))
        corr.mpc_reference_manifold_violation_count = int(
            final_reference_status.get("manifold_violation_count", 0))
        if not bool(final_reference_status.get("valid", False)):
            raise RuntimeError(
                "arm MPC reference rejected after deformation: "
                "corridor_violation=%d manifold_violation=%d" % (
                    corr.mpc_reference_corridor_violation_count,
                    corr.mpc_reference_manifold_violation_count))
        protected_dist = protected_waypoint_distances(
            path, getattr(corr, "protected_waypoints", []))
        protected_ok = bool(
            len(protected_dist) == len(getattr(corr, "protected_waypoints", [])) and
            (len(protected_dist) == 0 or
             float(np.max(protected_dist)) <= self.topology_protected_tolerance))
        feature_context["path_deformed_length"] = float(deformed_length)
        feature_context["path_shortcut_length"] = float(shortcut_length)
        mandatory_count = int(len(getattr(corr, "protected_waypoints", [])))
        mandatory_indices = [
            int(i) for i in protected_indices[1:1 + mandatory_count]
            if 0 <= int(i) < len(path)]
        corr.mandatory_topology_indices = list(mandatory_indices)
        self.path_adp_info = {
            "path_adp_mean": float(feature_context.get("path_adp_mean", 0.0)),
            "path_adp_max": float(feature_context.get("path_adp_max", 0.0)),
            "path_adp_delta": float(feature_context.get("path_adp_delta", 0.0)),
            "adp_path_enabled": int(feature_context.get("adp_path_enabled", 0)),
            "protected_saddle_count": int(len(getattr(corr, "protected_waypoints", []))),
            "protected_saddle_max_dist": (
                float(np.max(protected_dist)) if len(protected_dist) else 0.0),
            "protected_saddle_ok": int(protected_ok),
            "mandatory_topology_node_count": int(len(mandatory_indices)),
            "mandatory_saddle_reached": 0,
            "mandatory_saddle_max_dist": 0.0,
            "corridor_violation_count": 0,
            "topology_tracking_error": 0.0,
            "mpc_segment_count": max(1, len(mandatory_indices) + 1),
        }
        self._publish_path_adp_info()
        rospy.loginfo(
            "[handover][adp_path] enabled=%d mean=%.3f max=%.3f delta=%.3f length=%.3f->%.3f protected=%d max_dist=%.3f ok=%d mandatory=%d",
            self.path_adp_info["adp_path_enabled"],
            self.path_adp_info["path_adp_mean"],
            self.path_adp_info["path_adp_max"],
            self.path_adp_info["path_adp_delta"],
            deformed_length, shortcut_length,
            self.path_adp_info["protected_saddle_count"],
            self.path_adp_info["protected_saddle_max_dist"],
            self.path_adp_info["protected_saddle_ok"],
            self.path_adp_info["mandatory_topology_node_count"])
        execution_id = str(getattr(corr, "corridor_id", corr.label))
        self._sync_selected_corridor_debug(corr)
        self.selected_corridor_pub.publish(String(execution_id))
        rospy.loginfo("[handover] selected corridor: %s label=%s (source=%s cost=%.3f)",
                      execution_id, corr.label, source, corr.cost)
        self.mandatory_topology_indices = mandatory_indices
        return path

    def _line_offset(self, point, start, goal):
        point = np.asarray(point, float)[:3]
        start = np.asarray(start, float)[:3]
        goal = np.asarray(goal, float)[:3]
        axis = goal - start
        denom = float(np.dot(axis, axis))
        if denom <= 1e-12:
            return float(np.linalg.norm(point - start))
        alpha = np.clip(float(np.dot(point - start, axis)) / denom, 0.0, 1.0)
        nearest = start + alpha * axis
        return float(np.linalg.norm(point - nearest))

    def _point_forbidden(self, point):
        point = np.asarray(point, float)[:3]
        for anchor in getattr(self.field, "anchors", []):
            if not getattr(anchor, "forbidden", False):
                continue
            if float(anchor.signed_distance(point)) <= 0.0:
                return True
        return False

    def _polyline_samples(self, points, samples_per_segment=12):
        points = [np.asarray(p, float)[:3] for p in points]
        out = []
        for a, b in zip(points[:-1], points[1:]):
            for idx in range(int(samples_per_segment) + 1):
                if out and idx == 0:
                    continue
                alpha = float(idx) / float(max(int(samples_per_segment), 1))
                out.append(a + alpha * (b - a))
        return out

    def _recovery_path_safety(self, start, saddle, goal):
        offsets = self._arm_interest_offsets(start)
        if not self.arm_interest_enabled:
            offsets = {"ee": np.zeros(3)}
        ee_stop = min(float(self.gate.rho_stop), float(self.risk_stop_threshold))
        arm_stop = float(self.arm_interest_gate.rho_stop)
        max_phi = 0.0
        worst_label = ""
        for ee in self._polyline_samples([start, saddle, goal], samples_per_segment=14):
            for label, offset in offsets.items():
                point = np.asarray(ee, float) + np.asarray(offset, float)
                if self._point_forbidden(point):
                    return False, "forbidden:%s" % label, float("inf"), label
                phi = float(self.field.phi_s(point))
                if phi > max_phi:
                    max_phi = phi
                    worst_label = str(label)
                limit = ee_stop if label == "ee" else arm_stop
                if phi >= limit:
                    return False, "risk_stop:%s" % label, phi, label
        return True, "", float(max_phi), worst_label

    def _morse_recovery_corridor(self, start, goal):
        dbg = getattr(self.manifold, "last_topology_debug", {}) or {}
        nodes = list(dbg.get("nodes", []))
        saddles = [
            n for n in nodes
            if (str(n.get("kind", "")) if isinstance(n, dict)
                else str(getattr(n, "kind", ""))) == "saddle"]
        if not saddles:
            critical = dbg.get("critical", {}) or {}
            saddles = list(critical.get("saddles", []) or [])
        if not saddles:
            chain = dbg.get("critical_chain", {}) or {}
            saddles = [
                item for item in list(chain.get("saddles", []) or [])
                if str(item.get("status", "")) in (
                    "kept", "selected", "candidate", "filtered")]
        if not saddles:
            return None
        start = np.asarray(start, float)[:3]
        goal = np.asarray(goal, float)[:3]
        axis = goal - start
        axis_len2 = float(np.dot(axis, axis))

        def saddle_point(node):
            if isinstance(node, dict):
                return np.asarray(node.get("point", node.get("world", start)), float)[:3]
            return np.asarray(getattr(node, "point", start), float)[:3]

        def saddle_id(node):
            if isinstance(node, dict):
                return str(node.get("id", node.get("critical_id", "saddle")))
            return str(getattr(node, "id", getattr(node, "critical_id", "saddle")))

        def saddle_score(node):
            point = saddle_point(node)
            offset = self._line_offset(point, start, goal)
            if axis_len2 <= 1e-12:
                progress = 0.0
            else:
                progress = np.clip(
                    float(np.dot(point - start, axis)) / axis_len2, 0.0, 1.0)
            phi = float(self.field.phi_s(point))
            return (offset, -phi, -abs(progress - 0.5))

        safe_saddles = []
        rejected = []
        for saddle in saddles:
            point = saddle_point(saddle)
            ok, reason, max_phi, label = self._recovery_path_safety(
                start, point, goal)
            if ok:
                safe_saddles.append((saddle, max_phi))
            else:
                rejected.append((saddle, reason, max_phi, label))
        if not safe_saddles:
            dbg = dict(dbg)
            dbg["topology_recovery_used"] = 0
            dbg["topology_recovery_reject_reason"] = (
                rejected[0][1] if rejected else "no_safe_saddle")
            dbg["topology_disconnect_reason"] = "no_safe_morse_recovery"
            self.manifold.last_topology_debug = dbg
            rospy.logwarn(
                "[handover][topology] Morse recovery rejected: no safe saddle corridor reason=%s",
                dbg["topology_recovery_reject_reason"])
            return None

        saddle, recovery_max_phi = max(
            safe_saddles, key=lambda item: saddle_score(item[0]))
        selected_saddle_point = saddle_point(saddle)
        selected_saddle_id = saddle_id(saddle)
        corr = Corridor(np.asarray([start, selected_saddle_point, goal], float),
                        radius=self.topology_corridor_radius,
                        label="morse_recovery_saddle",
                        cost=0.0)
        corr.corridor_id = "arm_morse_recovery_c0001"
        corr.source = "morse_recovery"
        corr.topology_nodes = ["start", selected_saddle_id, "goal"]
        corr.node_sequence = list(corr.topology_nodes)
        corr.topology_kinds = ["saddle"]
        corr.node_type_sequence = ["start", "saddle", "goal"]
        corr.morse_nodes = [{
            "id": selected_saddle_id,
            "type": "saddle",
        }]
        corr.morse_node_ids = [corr.morse_nodes[0]["id"]]
        corr.morse_node_types = ["saddle"]
        corr.morse_induced = True
        corr.morse_forced = 1
        corr.protected_waypoints = np.asarray([selected_saddle_point], float)
        corr.auxiliary_node_ids = []
        corr.auxiliary_node_count = 0
        corr.topology_role = "morse_saddle"
        corr.morse_priority_class = 0
        corr.path_length = float(
            np.linalg.norm(saddle_point - start) +
            np.linalg.norm(goal - saddle_point))
        corr.min_clearance = 0.0
        corr.mean_phi_on_path = float(np.mean([
            self.field.phi_s(start),
            self.field.phi_s(selected_saddle_point),
            self.field.phi_s(goal),
        ]))
        corr.max_phi_on_path = float(max(
            self.field.phi_s(start),
            self.field.phi_s(selected_saddle_point),
            self.field.phi_s(goal)))
        corr.recovery_max_phi = float(recovery_max_phi)
        corr.risk_cost = float(corr.path_length * corr.mean_phi_on_path +
                               2.0 * corr.max_phi_on_path)
        corr.topology_cost = -1.0
        corr.distance_cost = float(corr.path_length)
        corr.smooth_cost = 0.0
        corr.curvature_cost = 0.0
        corr.topology_value = 1.0

        dbg = dict(dbg)
        dbg["topology_recovery_used"] = 1
        dbg["topology_disconnect_reason"] = "morse_recovery_after_empty_candidates"
        dbg["num_candidate_corridors"] = max(1, int(dbg.get("num_candidate_corridors", 0)))
        dbg["num_used_saddles"] = max(1, int(dbg.get("num_used_saddles", 0)))
        dbg["used_saddles"] = max(1, int(dbg.get("used_saddles", 0)))
        dbg["used_saddle_count"] = max(1, int(dbg.get("used_saddle_count", 0)))
        dbg["num_forced_critical_corridors"] = max(
            1, int(dbg.get("num_forced_critical_corridors", 0)))
        dbg["selected_corridor_label"] = corr.label
        dbg["selected_corridor_id"] = corr.corridor_id
        dbg["execution_corridor_id"] = corr.corridor_id
        dbg["selected_morse_induced"] = True
        dbg["selected_protected_saddle_count"] = 1
        dbg["selected_corridor_max_phi"] = float(corr.max_phi_on_path)
        dbg["selected_recovery_max_sampled_phi"] = float(recovery_max_phi)
        dbg["candidate_corridors"] = [{
            "corridor_id": corr.corridor_id,
            "label": corr.label,
            "source": corr.source,
            "node_sequence": list(corr.node_sequence),
            "node_type_sequence": list(corr.node_type_sequence),
            "path_length": float(corr.path_length),
            "risk_cost": float(corr.risk_cost),
            "topology_cost": float(corr.topology_cost),
            "selected": True,
            "execution_corridor_id": corr.corridor_id,
            "morse_induced": True,
            "morse_node_ids": list(corr.morse_node_ids),
            "morse_node_types": list(corr.morse_node_types),
            "protected_saddle_count": 1,
            "max_sampled_phi": float(recovery_max_phi),
        }]
        dbg["candidate_before_filter"] = list(dbg["candidate_corridors"])
        dbg["candidate_after_filter"] = list(dbg["candidate_corridors"])
        dbg["candidate_before_filter_count"] = 1
        dbg["candidate_after_filter_count"] = 1
        self.manifold.last_topology_debug = dbg
        rospy.logwarn(
            "[handover][topology] recovered Morse saddle corridor id=%s saddle=%s offset=%.3f",
            corr.corridor_id, corr.morse_node_ids[0],
            self._line_offset(selected_saddle_point, start, goal))
        return corr

    def _morse_sampled_recovery_corridor(self, start, goal, bounds, to_world):
        """Recover an arm STSM corridor from sampled Morse saddle evidence.

        This is used only when graph assembly returns no candidate.  It still
        starts from the configured Morse potential over the local handover
        plane and exports the selected sampled saddle as the protected
        topology point; it is not a semantic/direct fallback.
        """
        start = np.asarray(start, float)[:3]
        goal = np.asarray(goal, float)[:3]
        (xmin, xmax), (ymin, ymax) = bounds
        nx = 21
        ny = 21
        xs = np.linspace(float(xmin), float(xmax), nx)
        ys = np.linspace(float(ymin), float(ymax), ny)
        P = np.full((nx, ny), float("inf"))
        safe = np.zeros((nx, ny), bool)
        goal2 = np.array([float(xmax), 0.0])
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                p = np.asarray(to_world(np.array([x, y], float)), float)[:3]
                if self._point_forbidden(p):
                    continue
                phi = float(self.field.phi_s(p))
                if phi >= min(float(self.gate.rho_stop),
                              float(self.arm_interest_gate.rho_stop)):
                    continue
                safe[i, j] = True
                g = float(np.sum((np.array([x, y], float) - goal2) ** 2))
                m = self.manifold.eps_m * float(
                    np.sin(3.0 * x) * np.cos(3.0 * y))
                P[i, j] = g + self.manifold.lam_s * phi + m
        candidates = []
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                if not (safe[i, j] and safe[i - 1, j] and safe[i + 1, j] and
                        safe[i, j - 1] and safe[i, j + 1]):
                    continue
                c = P[i, j]
                if not np.isfinite(c):
                    continue
                fxx = P[i + 1, j] - 2.0 * c + P[i - 1, j]
                fyy = P[i, j + 1] - 2.0 * c + P[i, j - 1]
                if fxx * fyy >= 0.0:
                    continue
                point = np.asarray(to_world(np.array([xs[i], ys[j]], float)), float)[:3]
                ok, reason, max_phi, _label = self._recovery_path_safety(
                    start, point, goal)
                if not ok:
                    continue
                progress = (float(xs[i] - xmin) /
                            max(float(xmax - xmin), 1e-9))
                offset = abs(float(ys[j]))
                candidates.append({
                    "id": "sampled_saddle_%02d_%02d" % (i, j),
                    "point": point,
                    "phi": float(self.field.phi_s(point)),
                    "max_phi": float(max_phi),
                    "score": (
                        -abs(progress - 0.5),
                        offset,
                        -float(self.field.phi_s(point))),
                })
        if not candidates:
            dbg = dict(getattr(self.manifold, "last_topology_debug", {}) or {})
            dbg["topology_recovery_used"] = 0
            dbg["topology_recovery_reject_reason"] = "no_sampled_morse_saddle"
            dbg["sampled_morse_safe_cells"] = int(np.sum(safe))
            self.manifold.last_topology_debug = dbg
            return None
        selected = max(candidates, key=lambda item: item["score"])
        dbg = dict(getattr(self.manifold, "last_topology_debug", {}) or {})
        dbg.setdefault("critical", {})["saddles"] = [{
            "id": selected["id"],
            "point": selected["point"].tolist(),
            "kind": "saddle",
            "source": "sampled_morse_potential",
        }]
        dbg.setdefault("critical_chain", {}).setdefault("saddles", []).append({
            "id": selected["id"],
            "point": selected["point"].tolist(),
            "kind": "saddle",
            "status": "selected",
            "stage": "sampled_morse_recovery",
            "source": "sampled_morse_potential",
        })
        dbg["num_critical_saddles"] = max(
            1, int(dbg.get("num_critical_saddles", 0) or 0))
        dbg["num_safe_saddles"] = max(
            1, int(dbg.get("num_safe_saddles", 0) or 0))
        dbg["sampled_morse_saddle_count"] = int(len(candidates))
        self.manifold.last_topology_debug = dbg
        return self._morse_recovery_corridor(start, goal)

    def _handover_corridor(self, start, goal):
        if self.topology_enabled:
            try:
                corr = self._topology_handover_corridors(start, goal)
                if corr is not None:
                    self._publish_topology_info(True, False)
                    return corr, "topology"
            except Exception as exc:
                rospy.logwarn("[handover][topology] failed, fallback=%s: %s",
                              self.topology_fallback_enabled, exc)
                self._write_failed_topology_diagnostics(str(exc))
        self._write_failed_topology_diagnostics(
            "handover topology planner returned no STSM corridor")
        raise RuntimeError("handover topology planner returned no STSM corridor")

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
            base, "arm", debug, failure_reason=failure_reason)

    def _topology_handover_corridors(self, start, goal):
        start = np.asarray(start, float)
        goal = np.asarray(goal, float)
        axis = goal - start
        length = float(np.linalg.norm(axis))
        if length < 1e-6:
            return None
        e1 = axis / length
        risk_points = []
        for name in ("head", "chest"):
            if name in self.human.body_parts:
                risk_points.append(self.human.body_parts[name][0])
        risk_center = np.mean(np.asarray(risk_points, float), axis=0) if risk_points else self.human.pos
        lateral = risk_center - start
        lateral = lateral - float(np.dot(lateral, e1)) * e1
        if np.linalg.norm(lateral) < 1e-6:
            lateral = np.array([0.0, 1.0, 0.0])
            lateral = lateral - float(np.dot(lateral, e1)) * e1
        e2 = lateral / max(float(np.linalg.norm(lateral)), 1e-9)
        origin = start

        def to_world(p2):
            return origin + e1 * float(p2[0]) + e2 * float(p2[1])

        def to_plane(p3):
            rel = np.asarray(p3, float) - origin
            return np.array([float(np.dot(rel, e1)), float(np.dot(rel, e2))])

        bounds = [
            (-self.topology_longitudinal_margin,
             length + self.topology_longitudinal_margin),
            (-self.topology_lateral_margin, self.topology_lateral_margin),
        ]
        semantic = []
        rc2 = to_plane(risk_center)
        for side in (-1.0, 1.0):
            semantic.append(to_world(np.array([
                np.clip(rc2[0], bounds[0][0], bounds[0][1]),
                np.clip(rc2[1] + side * 0.20, bounds[1][0], bounds[1][1]),
            ])))
        for p in (self.wait, self.handover):
            p2 = to_plane(p)
            for side in (-1.0, 1.0):
                semantic.append(to_world(np.array([
                    np.clip(p2[0], bounds[0][0], bounds[0][1]),
                    np.clip(p2[1] + side * 0.12, bounds[1][0], bounds[1][1]),
                ])))
        interest_config = {
            "enabled": bool(self.arm_interest_enabled),
            "offsets": self._arm_interest_offsets(start),
            "labels": ["ee", "wrist", "elbow", "object"],
            "rho": self.arm_interest_gate.rho_stop,
        }
        base_hard_clearance = _effective(
            self.topology_hard_clearance,
            self.topology_profile_defaults["hard_clearance"])
        hard_attempts = [
            self.topology_hard_clearance,
            min(base_hard_clearance, 0.03),
            min(base_hard_clearance, 0.02),
        ]
        corrs = []
        seen_hard = set()
        for hard_clearance in hard_attempts:
            hard_key = (
                "auto" if hard_clearance is None
                else round(float(hard_clearance), 4))
            if hard_key in seen_hard:
                continue
            seen_hard.add(hard_key)
            corrs = self.manifold.enumerate_topological_corridors(
                start, goal, bounds, radius=self.topology_corridor_radius,
                grid_resolution=self.topology_grid_resolution,
                merge_radius=self.topology_merge_radius,
                min_clearance=self.topology_min_clearance,
                hard_clearance=hard_clearance,
                neighbor_k=self.topology_neighbor_k,
                k=self.topology_k_paths,
                max_graph_nodes=self.topology_max_graph_nodes,
                semantic_nodes=semantic,
                to_world=to_world,
                to_plane=to_plane,
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
                    "min_saddle_offset": self.topology_min_saddle_offset,
                    "goal_saddle_exclusion": self.topology_goal_saddle_exclusion,
                    "corridor_dedupe_distance": self.topology_corridor_dedupe_distance,
                    "candidate_pool_min": self.topology_candidate_pool_min,
                    "max_corridor_turn": self.topology_max_corridor_turn,
                    "max_corridor_curvature": self.topology_max_corridor_curvature,
                    "min_segment_length": self.topology_min_segment_length,
                    "require_risk_improvement": self.topology_require_risk_improvement,
                    "candidate_max_risk": self.topology_candidate_max_risk,
                    "corridor_score_weights": self.topology_corridor_score_weights,
                        "task_mode": self.task_mode,
                        "task_config": self.task_config,
                        "task_weight": self.task_weight,
                        "manifold_constraint_mode": self.manifold_constraint_mode,
                        "task_minima_points": [
                            {
                                "type": "handover",
                                "position": self.handover_minima.tolist(),
                        },
                    ],
                    "dynamics_profile": {
                        "type": "arm",
                        "nominal_speed": self.v_cap_far,
                        "max_tracking_turn": self.topology_max_corridor_turn,
                        "max_curvature": self.topology_max_corridor_curvature,
                        "min_progress": 0.01,
                    },
                })
            if corrs:
                if hard_clearance != self.topology_hard_clearance:
                    rospy.logwarn(
                        "[handover][topology] recovered with relaxed hard_clearance %.3f -> %.3f",
                        base_hard_clearance,
                        _effective(hard_clearance, base_hard_clearance))
                break
        for c in corrs:
            rospy.loginfo(
                "[handover][corridor] id=%s label=%s base=%.3f total=%.3f mean_phi=%.3f max_phi=%.3f clearance=%.3f nodes=%s",
                getattr(c, "corridor_id", ""), c.label, c.base_cost, c.cost,
                getattr(c, "mean_phi_on_path", 0.0),
                getattr(c, "max_phi_on_path", 0.0),
                getattr(c, "min_clearance", 0.0),
                ",".join(getattr(c, "topology_nodes", [])))
        if not corrs:
            recovery = self._morse_recovery_corridor(start, goal)
            if recovery is None:
                recovery = self._morse_sampled_recovery_corridor(
                    start, goal, bounds, to_world)
            if recovery is not None:
                corrs = [recovery]
            else:
                return None
        if self.topology_refinement_enabled and not self.baseline:
            executable = []
            for c in corrs:
                try:
                    refined, _protected = self._corridor_reference_path(
                        c, start, goal, n=30)
                    refined = np.asarray(refined, float)
                    if len(refined):
                        risks = [float(self.field.phi_s(p)) for p in refined]
                        c.risk_cost = float(np.mean(risks)) if risks else 0.0
                        c.length_cost = float(path_length(refined))
                        c.smooth_cost = float(
                            getattr(c, "max_turn_angle", 0.0) +
                            getattr(c, "mean_turn_angle", 0.0))
                        c.execution_cost = float(getattr(
                            c, "execution_cost", c.smooth_cost))
                    executable.append(c)
                except Exception as exc:
                    c.reject_reason = str(exc)
                    rospy.logwarn(
                        "[handover][refine] skip %s reason=%s",
                        getattr(c, "corridor_id", getattr(c, "label", "")),
                        exc)
            if not executable:
                return None
            self._rescore_handover_executable_corridors(executable)
            executable.sort(key=lambda item: float(
                getattr(item, "total_score", getattr(item, "cost", 0.0))))
            for rank, item in enumerate(executable, start=1):
                item.rank = int(rank)
            selected = executable[0]
            self._adp_active_candidate_features = dict(getattr(
                selected, "adp_candidate_features", {}) or {})
            _, self._adp_active_candidate_missing = self._adp_candidate_features(
                selected)
            selected_id = str(getattr(selected, "corridor_id",
                                      getattr(selected, "label", "")))
            for c in corrs:
                setattr(c, "selected", str(getattr(
                    c, "corridor_id", getattr(c, "label", ""))) == selected_id)
            dbg = dict(getattr(self.manifold, "last_topology_debug", {}) or {})
            dbg["selected_corridor_id"] = selected_id
            dbg["selected_corridor_label"] = selected_id
            dbg["execution_corridor_id"] = selected_id
            dbg["selected_rank"] = 1
            dbg["selection_override_reason"] = ""
            dbg["selected_candidate_total_score"] = float(getattr(
                selected, "total_score", getattr(selected, "cost", 0.0)))
            dbg["selected_raw_waypoints_count"] = int(len(np.asarray(
                getattr(selected, "raw_topology_waypoints",
                        getattr(selected, "topology_ordered_waypoints", [])),
                float)))
            dbg["selected_refined_waypoints_count"] = int(len(np.asarray(
                getattr(selected, "refined_waypoints", []), float)))
            dbg["risk_field_used"] = 1
            dbg["manifold_used"] = 1
            dbg["morse_used"] = 1
            dbg["topology_graph_used"] = 1
            dbg["candidate_corridor_used"] = 1
            dbg["candidate_ranking_used"] = 1
            dbg["fallback_used"] = 0
            dbg["mpc_used"] = 1
            dbg["mpc_reference_source"] = str(getattr(
                selected, "final_reference_source", "") or (
                "refined_waypoints"
                if int(getattr(selected, "refinement_used", 0)) == 1
                else "selected_candidate_waypoints"))
            dbg["final_path_source"] = "Morse->Candidate->Ranking->Refinement->MPC"
            dbg["adp_role"] = adp_role_from_runtime(
                self.adp_enabled, bool(self.adp_learning and self.adp_learning.config.get("enabled", False)),
                self.adp_decision_influence_enabled,
                effective_lambda=(self.adp_learning.config.get("lambda_adp", 0.0)
                                  if self.adp_learning else 0.0),
                ranking_contribution=self.adp_ranking_influence_enabled,
                control_contribution=False)
            final_rows = []
            for c in executable:
                cid = str(getattr(c, "corridor_id", getattr(c, "label", "")))
                final_rows.append({
                    "candidate_id": cid, "corridor_id": cid,
                    "label": str(getattr(c, "label", cid)),
                    "candidate_status": "safe", "candidate_filter_class": "safe",
                    "selected": cid == selected_id,
                    "execution_corridor_id": selected_id if cid == selected_id else "",
                    "rank": int(getattr(c, "rank_total", getattr(c, "rank", 0))),
                    "base_total_cost": float(getattr(c, "base_cost", 0.0)),
                    "adp_value_raw": float(getattr(c, "adp_value_raw", 0.0)),
                    "adp_value_normalized": float(getattr(c, "adp_value_normalized", 0.0)),
                    "adp_value_normalized_preclip": float(
                        (getattr(c, "adp_ranking_audit", {}) or {}).get(
                            "ranking_normalization", {}).get(
                            "normalized_before_clip", 0.0)),
                    "effective_lambda_adp": float(getattr(c, "effective_lambda_adp", 0.0)),
                    "adp_cost": float(getattr(c, "adp_cost", 0.0)),
                    "total_cost_with_adp": float(getattr(c, "total_cost", 0.0)),
                    "total_cost": float(getattr(c, "total_cost", 0.0)),
                    "total_score": float(getattr(c, "total_score", 0.0)),
                    "rank_before_adp": int(getattr(c, "rank_before_adp", 0)),
                    "rank_after_adp": int(getattr(c, "rank_after_adp", 0)),
                    "adp_changed_rank": bool(getattr(c, "adp_changed_rank", False)),
                    "ranking_theta_source": str(getattr(c, "adp_ranking_theta_source", "")),
                    "adp_ranking_audit": dict(getattr(
                        c, "adp_ranking_audit", {}) or {}),
                    "candidate_adp_features_raw": dict(getattr(
                        c, "adp_candidate_features", {}) or {}),
                    "candidate_adp_features_normalized": dict((getattr(
                        c, "adp_ranking_audit", {}) or {}).get(
                        "candidate_adp_features_normalized", {}) or {}),
                    "candidate_feature_missing": dict(getattr(
                        c, "adp_candidate_feature_missing", {}) or {}),
                    "adp_role": dbg["adp_role"],
                    "adp_affects_candidate_ranking": int(
                        self.adp_ranking_influence_enabled and self.adp_learning is not None and
                        abs(float(self.adp_learning.config.get("lambda_adp", 0.0))) > 1e-12),
                    "adp_affects_control": 0,
                    "mpc_adp_enabled": int(self.adp_mpc_influence_enabled),
                    "refinement_used": int(getattr(c, "refinement_used", 0)),
                    "refined_path_length": float(getattr(
                        c, "refined_path_length", getattr(c, "path_length", 0.0))),
                    "raw_topology_waypoints": np.asarray(getattr(
                        c, "raw_topology_waypoints",
                        getattr(c, "topology_ordered_waypoints", [])), float).tolist(),
                    "refined_waypoints": np.asarray(getattr(
                        c, "refined_waypoints", getattr(c, "waypoints", [])), float).tolist(),
                })
            dbg["candidate_corridors"] = final_rows
            dbg["final_candidate_ranking"] = list(final_rows)
            dbg["candidate_feature_delta_summary"] = self._candidate_feature_delta_summary(
                final_rows)
            dbg["candidate_after_filter"] = list(final_rows)
            dbg["candidate_after_top_k"] = list(final_rows)
            self.manifold.last_topology_debug = dbg
            self._write_failed_topology_diagnostics("success")
            rospy.loginfo(
                "[handover][refine] selected executable %s total=%.3f risk=%.3f length=%.3f",
                selected_id,
                float(getattr(selected, "total_score", getattr(selected, "cost", 0.0))),
                float(getattr(selected, "risk_cost", 0.0)),
                float(getattr(selected, "length_cost", 0.0)))
            return selected
        return corrs[0]

    @staticmethod
    def _candidate_feature_delta_summary(rows):
        if len(rows) < 2:
            return {"candidate_count": len(rows), "pair_deltas": []}
        first, second = rows[0], rows[1]
        left = first.get("candidate_adp_features_raw", {}) or {}
        right = second.get("candidate_adp_features_raw", {}) or {}
        deltas = {name: float(left.get(name, 0.0)) - float(right.get(name, 0.0))
                  for name in sorted(set(left) | set(right))}
        raw_gap = float(first.get("adp_value_raw", 0.0)) - float(
            second.get("adp_value_raw", 0.0))
        adp_gap = float(first.get("adp_cost", 0.0)) - float(
            second.get("adp_cost", 0.0))
        return {"candidate_count": len(rows), "num_candidate_pairs": 1,
                "num_pairs_with_feature_difference": int(any(abs(v) > 1e-12 for v in deltas.values())),
                "num_pairs_with_raw_value_difference": int(abs(raw_gap) > 1e-12),
                "num_pairs_with_adp_cost_difference": int(abs(adp_gap) > 1e-12),
                "max_raw_value_gap": abs(raw_gap), "max_adp_cost_gap": abs(adp_gap),
                "pair_deltas": [{
            "candidate_a": first.get("candidate_id", ""),
            "candidate_b": second.get("candidate_id", ""),
            "feature_deltas": deltas, "adp_value_raw_delta": raw_gap,
        }]}

    def _rescore_handover_executable_corridors(self, corridors):
        corridors = list(corridors or [])
        if not corridors:
            return

        def weight(name, default, *aliases):
            raw = dict(self.topology_corridor_score_weights or {})
            for key in (name,) + aliases:
                if key in raw:
                    return float(raw[key])
                prefixed = "w_" + key
                if prefixed in raw:
                    return float(raw[prefixed])
            return float(default)

        weights = {
            "risk": weight("risk", 4.0),
            "length": weight("length", 1.0),
            "smooth": weight("smooth", 1.5),
            "task": weight("task", 2.0),
            "execution": weight("execution", 1.0, "exec"),
            "topology": weight("topology", 0.5),
            "diversity": weight("diversity", 0.5),
        }

        def normalize(attr, norm_attr):
            vals = np.asarray([float(getattr(c, attr, 0.0)) for c in corridors], float)
            lo = float(np.min(vals))
            hi = float(np.max(vals))
            if hi - lo <= 1e-9:
                norms = np.zeros_like(vals)
            else:
                norms = (vals - lo) / (hi - lo)
            for c, val in zip(corridors, norms):
                setattr(c, norm_attr, float(val))

        normalize("risk_cost", "risk_norm")
        normalize("length_cost", "length_norm")
        normalize("smooth_cost", "smooth_norm")
        normalize("task_cost", "task_norm")
        normalize("execution_cost", "execution_norm")
        for c in corridors:
            score = (
                float(getattr(c, "risk_cost", 0.0)) +
                float(getattr(c, "length_cost", 0.0)) +
                float(getattr(c, "smooth_cost", 0.0)) +
                float(getattr(c, "task_cost", 0.0)))
            c.total_score = float(score)
            c.total_cost = float(score)
            c.cost = float(score)
            c.base_cost = float(score)
        def candidate_id(corr):
            return str(getattr(corr, "corridor_id", getattr(corr, "label", "")))
        corridors.sort(key=lambda c: (float(getattr(c, "base_cost", 0.0)), candidate_id(c)))
        for rank, corr in enumerate(corridors, start=1):
            corr.rank_base = int(rank)
        snapshot = self.adp_ranking_critic
        raw_values = []
        ranking_audits = []
        if snapshot is not None:
            ee = self._ee_pos()
            target = self.handover
            for corr in corridors:
                candidate_features, candidate_missing = self._adp_candidate_features(corr)
                features = self.adp_features.build_arm(
                    ee, target, self.field,
                    gate_info={"state": "RANKING", "stop": False,
                               "rho_warn": self.gate.rho_warn},
                    phase=self.phase, candidate_features=candidate_features)
                try:
                    _unused, d_corridor = corr.project(ee)
                    features["d_corridor"] = float(d_corridor)
                except Exception:
                    pass
                normalized = snapshot.featurize(features)
                prediction = snapshot.predict_detail(features)
                feature_vector = []
                for name, normalized_value, theta in zip(
                        snapshot.feature_names, normalized, snapshot.theta):
                    raw_value = float(features.get(
                        name, 1.0 if name == "bias" else 0.0))
                    feature_vector.append({
                        "feature_name": str(name),
                        "raw_value": raw_value,
                        "normalized_value": float(normalized_value),
                        "theta": float(theta),
                        "weighted_value": float(theta * normalized_value),
                    })
                candidate_context = {}
                for name in (
                        "path_length", "risk_cost", "min_clearance",
                        "clearance", "task_cost", "execution_cost",
                        "motion_cost", "max_turn_angle", "max_curvature",
                        "topology_value", "topology_diversity"):
                    value = getattr(corr, name, None)
                    try:
                        candidate_context[name] = float(value)
                    except (TypeError, ValueError):
                        candidate_context[name] = None
                ranking_audits.append({
                    "candidate_id": candidate_id(corr),
                    "feature_vector": feature_vector,
                    "candidate_context": candidate_context,
                    "candidate_adp_features_raw": dict(candidate_features),
                    "candidate_feature_missing": dict(candidate_missing),
                    "candidate_specific_critic_features": [
                        "d_corridor", "candidate_path_length",
                        "candidate_risk_mean", "candidate_risk_max",
                        "candidate_min_clearance", "candidate_task_cost",
                        "candidate_execution_cost"],
                    "prediction_raw": float(prediction["raw"]),
                    "prediction_clipped": float(prediction["clipped"]),
                    "prediction_clip_hit": bool(prediction["clip_hit"]),
                })
                corr.adp_candidate_features = dict(candidate_features)
                corr.adp_candidate_feature_missing = dict(candidate_missing)
                raw_values.append(float(prediction["raw"]))
        adjustments, norm_meta = adp_ranking_adjustments(
            raw_values, metadata=(snapshot.metadata if snapshot else {}),
            lambda_adp=(self.adp_learning.config.get("lambda_adp", 0.0)
                        if self.adp_ranking_influence_enabled and self.adp_learning
                        else 0.0),
            normalization=(self.adp_learning.config.get("adp_value_normalization", "robust")
                           if self.adp_learning else "robust"),
            norm_clip=(self.adp_learning.config.get("adp_norm_clip", 3.0)
                       if self.adp_learning else 3.0),
            contribution_clip=(self.adp_learning.config.get("adp_contribution_clip", 0.10)
                               if self.adp_learning else 0.10))
        norm_clip = (self.adp_learning.config.get("adp_norm_clip", 3.0)
                     if self.adp_learning else 3.0)
        for corr, item, audit in zip(corridors, adjustments, ranking_audits):
            normalized_before_clip = (
                (float(item["adp_value_raw"]) - float(norm_meta["center"])) /
                max(float(norm_meta["scale"]), 1e-6))
            audit["ranking_normalization"] = {
                "source": str(norm_meta.get("source", "")),
                "center": float(norm_meta["center"]),
                "scale": float(norm_meta["scale"]),
                "normalized_before_clip": float(normalized_before_clip),
                "normalized_after_clip": float(item["adp_value_normalized"]),
                "norm_clip": float(norm_clip),
            }
            audit["candidate_adp_features_normalized"] = {
                item["feature_name"]: float(item["normalized_value"])
                for item in audit["feature_vector"]
                if item["feature_name"].startswith("candidate_")}
            corr.adp_value_raw = float(item["adp_value_raw"])
            corr.adp_value_normalized = float(item["adp_value_normalized"])
            corr.effective_lambda_adp = float(item["effective_lambda_adp"])
            corr.adp_cost = float(item["adp_cost"])
            corr.total_cost = float(corr.base_cost + corr.adp_cost)
            corr.total_score = float(corr.total_cost)
            corr.cost = float(corr.total_cost)
            corr.adp_ranking_theta_source = "run_start_snapshot"
            corr.adp_normalization = dict(norm_meta)
            corr.adp_ranking_audit = dict(audit)
        corridors.sort(key=lambda c: (float(getattr(c, "total_score", 0.0)), candidate_id(c)))
        for rank, corr in enumerate(corridors, start=1):
            corr.rank_total = int(rank)
            corr.rank_before_adp = int(getattr(corr, "rank_base", rank))
            corr.rank_after_adp = int(rank)
            corr.adp_changed_rank = bool(corr.rank_before_adp != corr.rank_after_adp)

    def _publish_topology_info(self, used_topology, fallback_used):
        dbg = getattr(self.manifold, "last_topology_debug", {}) or {}
        self.topology_info_pub.publish(Float64MultiArray(data=[
            1.0 if self.topology_enabled else 0.0,
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
        ]))

    def _servo_to_path(self, path, dt=0.1, tol=0.04, max_steps=400,
                       mandatory_indices=None, corridor=None,
                       reference_corridor=None):
        rate = rospy.Rate(1.0 / dt)
        idx = 0
        steps = 0
        mandatory_indices = set(int(i) for i in (mandatory_indices or []))
        mandatory_min_dist = {int(i): float("inf") for i in mandatory_indices}
        corridor_violation_count = 0
        topology_error_sum = 0.0
        topology_error_count = 0
        previous_dq = np.zeros(6, float)
        best_target_error = float("inf")
        no_progress_cycles = 0
        handover_servo_requested = bool(
            int(self.phase) == 3 and self._path_contains_handover_target(path))
        if handover_servo_requested:
            self._record_phase3_chain_event(
                "servo_to_path_start",
                stsm_reference_pose=np.asarray(path, float)[-1],
                ee_actual_pose=self._ee_pos(),
                extra={
                    "path_len": int(len(path)),
                    "tol": float(tol),
                    "max_steps": int(max_steps),
                    "handover_target": self._pose_list(self.handover),
                })
        while not rospy.is_shutdown() and idx < len(path) and steps < max_steps:
            ee = self._ee_pos()
            for midx in mandatory_indices:
                if 0 <= midx < len(path):
                    mandatory_min_dist[midx] = min(
                        mandatory_min_dist[midx],
                        float(np.linalg.norm(ee - path[midx])))
            interest_eval = self._publish_metrics(ee)
            risk = self.field.phi_s(
                ee, self.ee_vel_filtered if np.linalg.norm(self.ee_vel_filtered) > 1e-9
                else np.zeros(3))
            ee_gate = self.gate.evaluate(risk)
            arm_gate = None
            if self.arm_interest_gate_enabled and interest_eval is not None:
                arm_gate = self.arm_interest_gate.evaluate(interest_eval["phi_max"])
            gate, gate_source = self._combine_arm_gates(
                ee_gate, arm_gate, interest_eval)
            self._publish_gate(
                gate, gate_source=gate_source, arm_gate=arm_gate,
                interest_eval=interest_eval)
            adp_value = self._publish_adp_value(
                ee, gate=gate, interest_eval=interest_eval)
            if gate.stop:
                q = self._cur_joints()
                self._hold_current(q)
                self.stop_triggered = True
                self.stop_reason = gate.reason
                rospy.logwarn("[handover][gate] STOP risk=%.3f reason=%s",
                              gate.risk, gate.reason)
                if handover_servo_requested:
                    self._record_phase3_chain_event(
                        "servo_to_path_gate_stop",
                        stsm_reference_pose=path[idx],
                        ee_actual_pose=ee,
                        extra={
                            "idx": int(idx),
                            "steps": int(steps),
                            "gate_reason": str(gate.reason),
                        })
                rospy.sleep(0.2)
                return False
            target = path[idx]
            self._record_mpc_reference(
                corridor if corridor is not None else reference_corridor,
                idx, target)
            err = target - ee
            local_tol = (
                min(tol, self.mandatory_topology_tolerance)
                if idx in mandatory_indices else tol)
            if np.linalg.norm(err) < local_tol:
                if handover_servo_requested:
                    self._record_phase3_chain_event(
                        "servo_to_path_target_reached",
                        stsm_reference_pose=target,
                        ee_actual_pose=ee,
                        extra={
                            "idx": int(idx),
                            "position_error": float(np.linalg.norm(err)),
                            "tol": float(local_tol),
                        })
                if idx in mandatory_indices:
                    rospy.loginfo("[handover] mandatory topology node reached idx=%d dist=%.3f",
                                  idx, np.linalg.norm(err))
                idx += 1
                best_target_error = float("inf")
                no_progress_cycles = 0
                continue
            target_error = float(np.linalg.norm(err))
            if target_error < (
                    best_target_error - self.arm_mpc_progress_epsilon):
                best_target_error = target_error
                no_progress_cycles = 0
            else:
                no_progress_cycles += 1
            if (not self.baseline and
                    no_progress_cycles >= self.arm_mpc_no_progress_cycles):
                q = self._cur_joints()
                self._hold_current(q)
                self.stop_triggered = True
                self.stop_reason = "mpc:no_task_progress"
                rospy.logerr(
                    "[handover][mpc] no measured task progress "
                    "idx=%d error=%.6f best=%.6f cycles=%d",
                    idx, target_error, best_target_error,
                    no_progress_cycles)
                return False
            q = self._cur_joints()
            J = self._jacobian(q)
            v_des = self.arm_servo_gain * err
            handover_protect = self._phase3_handover_protect_active(target)
            if handover_protect:
                v_des = self.arm_servo_gain * (self.handover - ee)
            if corridor is not None and not handover_protect:
                q_corr, d_corr = corridor.project(ee)
                topology_error_sum += float(d_corr)
                topology_error_count += 1
                if d_corr > float(corridor.radius):
                    corridor_violation_count += 1
                    pull = np.zeros_like(v_des)
                    pull[:len(q_corr)] = q_corr - ee[:len(q_corr)]
                    v_des += self.corridor_violation_gain * pull
            v_cap = self._ee_v_cap(ee) * gate.scale * self._adp_scale(adp_value)
            phase_name = (
                "handover" if int(self.phase) == 3 else
                "return" if int(self.phase) == 4 else "approach")
            dq = self.mpc.solve(J, v_des, dq_nom=previous_dq,
                                v_cap=v_cap, ee_pos=ee, dt=dt,
                                critic=(
                                    self.adp_critic
                                    if self.adp_influence_enabled else None),
                                feature_builder=self.adp_features,
                                field=self.field,
                                gate_info={
                                    "state": gate.state,
                                    "stop": gate.stop,
                                    "rho_warn": self.gate.rho_warn,
                                },
                                interest_risk=interest_eval or {},
                                target_pos=target,
                                phase=phase_name,
                                lambda_adp_arm=(
                                    self.lambda_adp_arm
                                    if self.adp_influence_enabled else 0.0),
                                adp_grad_eps=self.adp_grad_eps,
                                adp_descent_gain=self.adp_descent_gain,
                                solver_mode=self.adp_solver_mode,
                                adp_blend_alpha=self.adp_blend_alpha,
                                use_cvxpy=self.use_cvxpy,
                                interest_constraints={
                                    "enabled": bool(self.arm_interest_enabled),
                                    "offsets": self._arm_interest_offsets(ee),
                                    "labels": ["ee", "wrist", "elbow", "object"],
                                    "rho": self.arm_interest_gate.rho_stop,
                                    # Waypoint reach tolerance is deliberately
                                    # looser than the predictive MPC progress
                                    # tolerance.  Passing local_tol here lets
                                    # the beam objective treat a still-unreached
                                    # waypoint as "close enough" and repeatedly
                                    # choose hold_current, which appeared
                                    # remotely as dq_adp_norm == 0 with a
                                    # nonzero v_des_raw_norm.
                                    "task_progress_tolerance": float(
                                        min(local_tol,
                                            self.mpc.task_progress_tolerance)),
                                },
                                handover_protect=handover_protect,
                                handover_target=(
                                    self.handover if handover_protect else None),
                                handover_tracking_weight=(
                                    self.handover_tracking_weight),
                                q=q,
                                corridor=(
                                    corridor if not handover_protect else None),
                                predictive=bool(
                                    not self.baseline and
                                    corridor is not None and
                                    not handover_protect),
                                kinematics_source="real",
                                phase_cost_weights=self.mpc_phase_cost_weights)
            if (not self.baseline and str(
                    self.mpc.last_solver_status).startswith("safe_stop:")):
                self._hold_current(q)
                self.stop_triggered = True
                self.stop_reason = "mpc:%s" % self.mpc.last_solver_status
                rospy.logerr(
                    "[handover][mpc] predictive solve failed closed: %s "
                    "violations=%s",
                    self.mpc.last_solver_status,
                    self.mpc.last_constraint_violation)
                return False
            self._record_mpc_handover_diagnostic(target, ee, J, dq, dt)
            self.adp_control_info_pub.publish(Float64MultiArray(data=[
                float(self.mpc.last_adp_grad_norm),
                float(self.mpc.last_adp_soft_cost),
                float(self.mpc.last_v_adp_alignment),
                1.0 if self.adp_influence_enabled else 0.0,
                float(self.mpc.last_dls_adp_used),
                float(self.mpc.last_qp_used),
                float(self.mpc.solve_success_count),
                float(self.mpc.fallback_count),
                float(self.mpc.last_v_des_raw_norm),
                float(self.mpc.last_v_des_adp_norm),
                float(self.mpc.last_v_des_delta_norm),
                float(self.mpc.last_dq_nominal_norm),
                float(self.mpc.last_dq_adp_norm),
                float(self.mpc.last_dq_delta_norm),
                float(self.mpc.last_reject_forbidden_count),
                float(self.mpc.last_reject_interest_phi_count),
                1.0 if self.adp_learning is not None and
                self.adp_learning.config.get("enabled", False) else 0.0,
                1.0 if self.adp_decision_influence_enabled else 0.0,
                float(self.adp_learning.config.get("lambda_adp", 0.0)
                      if self.adp_learning and self.adp_ranking_influence_enabled else 0.0),
                1.0 if self.adp_ranking_influence_enabled else 0.0,
                1.0 if self.adp_mpc_influence_enabled else 0.0,
            ]))
            rospy.loginfo_throttle(
                5.0,
                "Arm solver: calls=%d, success=%d, fallback=%d, status=%s adp_grad=%.3f adp_soft=%.3f align=%.3f",
                self.mpc.solve_count,
                self.mpc.solve_success_count,
                self.mpc.fallback_count,
                self.mpc.last_solver_status,
                self.mpc.last_adp_grad_norm,
                self.mpc.last_adp_soft_cost,
                self.mpc.last_v_adp_alignment)
            self._send_joint(q + dq * dt, dt * 1.5)
            self._record_adp_transition(
                ee, gate=gate, interest_eval=interest_eval,
                control=previous_dq)
            previous_dq = np.asarray(dq, float).copy()
            steps += 1
            rate.sleep()
        ok = idx >= len(path)
        if not ok and not self.stop_reason:
            self.stop_triggered = True
            self.stop_reason = "mpc:path_not_completed"
        if handover_servo_requested:
            self._record_phase3_chain_event(
                "servo_to_path_end",
                stsm_reference_pose=np.asarray(path, float)[-1],
                ee_actual_pose=self._ee_pos(),
                extra={
                    "ok": bool(ok),
                    "idx": int(idx),
                    "steps": int(steps),
                    "path_len": int(len(path)),
                })
        if mandatory_min_dist:
            finite = [
                d for d in mandatory_min_dist.values()
                if np.isfinite(float(d))]
            max_dist = float(max(finite)) if finite else float("inf")
            reached = all(
                float(d) <= float(self.mandatory_topology_tolerance)
                for d in finite)
        else:
            max_dist = 0.0
            reached = True
        self.path_adp_info["mandatory_saddle_reached"] = int(reached)
        self.path_adp_info["mandatory_saddle_max_dist"] = float(max_dist)
        self.path_adp_info["corridor_violation_count"] = int(
            self.path_adp_info.get("corridor_violation_count", 0) +
            corridor_violation_count)
        if topology_error_count:
            prev_n = int(self.path_adp_info.get("_topology_error_count", 0))
            prev_mean = float(self.path_adp_info.get(
                "topology_tracking_error", 0.0))
            new_n = prev_n + topology_error_count
            self.path_adp_info["topology_tracking_error"] = float(
                (prev_mean * prev_n + topology_error_sum) / max(new_n, 1))
            self.path_adp_info["_topology_error_count"] = int(new_n)
        rospy.loginfo(
            "[handover][topology_exec] ok=%d mandatory_reached=%d max_saddle_dist=%.3f corridor_violations=%d topology_error=%.3f",
            int(ok), int(reached), float(max_dist), int(corridor_violation_count),
            float(self.path_adp_info.get("topology_tracking_error", 0.0)))
        self._publish_path_adp_info()
        return ok

    def _servo_to_path_segmented(self, path, mandatory_indices=None, corridor=None):
        mandatory_indices = sorted(set(
            int(i) for i in (mandatory_indices or [])
            if 0 < int(i) < len(path) - 1))
        cuts = [0] + mandatory_indices + [len(path) - 1]
        self.path_adp_info["mpc_segment_count"] = max(1, len(cuts) - 1)
        self._publish_path_adp_info()
        for seg_idx, (a, b) in enumerate(zip(cuts[:-1], cuts[1:]), 1):
            segment = np.asarray(path[a:b + 1], float)
            local_mandatory = [len(segment) - 1] if b in mandatory_indices else []
            rospy.loginfo(
                "[handover][mpc_segment] segment=%d/%d path_idx=%d->%d mandatory_end=%d",
                seg_idx, len(cuts) - 1, a, b, int(bool(local_mandatory)))
            if not self._servo_to_path(
                    segment, mandatory_indices=local_mandatory,
                    corridor=corridor):
                return False
        return True

    def run(self):
        self.mode_pub.publish(String("baseline" if self.baseline else "stsm"))
        if self.baseline:
            self._publish_topology_info(False, False)
        rospy.sleep(2.0)
        rospy.loginfo("[handover] going home")
        self.group.set_named_target("home")
        self.group.go(wait=True)
        self.group.stop()
        rospy.sleep(0.5)

        ee0 = self._ee_pos()
        self.home_ee_ref = np.array(ee0, float)
        rospy.loginfo("[handover] EE start %s", np.round(ee0, 3))
        self._publish_metrics(ee0)

        rospy.loginfo("[handover] phase 0: reach object")
        self._set_phase(0)
        if not self._servo_to_path(self._nominal_path(ee0, self.grasp)):
            if self.abort_on_stop:
                self._return_home()
                self._log_done()
                return

        rospy.loginfo("[handover] phase 1: %s handover approach",
                      "baseline" if self.baseline else "STSM-deformed")
        self._set_phase(1)
        path = self._deformed_path(self._ee_pos(), self.handover)
        self.corridor_evaluation_active = bool(
            not self.baseline and self.execution_corridor is not None)
        if not self._servo_to_path_segmented(
                path, mandatory_indices=self.mandatory_topology_indices,
                corridor=self.execution_corridor):
            if self.abort_on_stop:
                self.corridor_evaluation_active = False
                self._return_home()
                self._log_done()
                return
        handover_orientation_ref = self._ee_quat()
        rospy.loginfo("[handover] phase 2: handover pose reached")
        self._set_phase(2)
        ee_handover = self._ee_pos()
        handover_pos_err = float(np.linalg.norm(ee_handover - self.handover))
        handover_orientation_err = self._quat_error(
            handover_orientation_ref, handover_orientation_ref)
        self._publish_metrics(ee_handover)
        self._publish_handover_status(
            goal_reached=bool(handover_pos_err <= max(
                self.handover_pos_tol, 0.08)),
            pose_reached=bool(
                handover_pos_err <= self.handover_pos_tol and
                handover_orientation_err <= self.handover_orientation_tol),
            complete=False,
            pos_err=handover_pos_err,
            orientation_err=handover_orientation_err,
            stable_time=0.0)
        rospy.loginfo("[handover] phase 3: handover hold")
        self._set_phase(3)
        self._record_mpc_reference(self.execution_corridor, 0, self.handover)
        self._record_phase3_chain_event(
            "phase3_enter_before_protected_approach",
            stsm_reference_pose=self.handover,
            ee_actual_pose=ee_handover,
            extra={
                "position_error": float(handover_pos_err),
                "handover_pos_tol": float(self.handover_pos_tol),
                "handover_pose_reached": bool(
                    handover_pos_err <= self.handover_pos_tol),
            })
        if handover_pos_err > self.handover_pos_tol:
            rospy.logwarn(
                "[handover] phase 3 protected approach pos_err=%.4f tol=%.4f",
                handover_pos_err, self.handover_pos_tol)
            self.handover_protection_active = True
            self._servo_to_path(
                np.asarray([self.handover], float),
                tol=min(self.handover_pos_tol, 0.025),
                max_steps=80,
                corridor=self.execution_corridor)
            self.handover_protection_active = False
        else:
            self.handover_protection_active = False
        ee_after_phase3_approach = self._ee_pos()
        pos_err_after_phase3 = float(np.linalg.norm(
            ee_after_phase3_approach - self.handover))
        orientation_err_after_phase3 = self._quat_error(
            self._ee_quat(), handover_orientation_ref)
        self._publish_metrics(ee_after_phase3_approach)
        self._publish_handover_status(
            goal_reached=bool(pos_err_after_phase3 <= max(
                self.handover_pos_tol, 0.08)),
            pose_reached=bool(
                pos_err_after_phase3 <= self.handover_pos_tol and
                orientation_err_after_phase3 <= self.handover_orientation_tol),
            complete=False,
            pos_err=pos_err_after_phase3,
            orientation_err=orientation_err_after_phase3,
            stable_time=0.0)
        latest_mpc = {}
        records = self.mpc_handover_diagnostics.get("records", [])
        if records:
            latest_mpc = dict(records[-1])
        self._record_phase3_chain_event(
            "phase3_after_protected_approach",
            stsm_reference_pose=self.handover,
            mpc_output_pose=latest_mpc.get("mpc_output_point"),
            ee_actual_pose=ee_after_phase3_approach,
            extra={
                "position_error": pos_err_after_phase3,
                "orientation_error": orientation_err_after_phase3,
                "handover_pos_tol": float(self.handover_pos_tol),
                "last_mpc_before_error": latest_mpc.get("before_error", ""),
                "last_mpc_after_error": latest_mpc.get("after_error", ""),
                "last_mpc_guard_applied": latest_mpc.get("guard_applied", ""),
            })
        if not self._verify_handover_complete(handover_orientation_ref):
            self.stop_triggered = True
            self.stop_reason = "handover_not_complete"
            if self.abort_on_stop:
                self.corridor_evaluation_active = False
                self._return_home()
                self._log_done()
                return

        rospy.loginfo("[handover] phase 3: retreat after handover")
        self.handover_protection_active = False
        # The selected Morse corridor ends at the handover target.  Retreat
        # and home-return are separate task phases, not hidden extensions of
        # the selected approach tube.
        self.corridor_evaluation_active = False
        self._set_phase(4)
        if not self._servo_to_path(
                self._nominal_path(self._ee_pos(), self.wait)):
            self.corridor_evaluation_active = False
            self._return_home()
            if self.abort_on_stop:
                self._log_done()
                return
        self.corridor_evaluation_active = False
        self.task_completed = bool(self._return_home())
        self._log_done()

    def _record_mpc_reference(self, corridor, idx, target):
        if self.baseline or corridor is None:
            return
        cid = str(getattr(corridor, "corridor_id", getattr(corridor, "label", "")))
        source = (
            "refined_waypoints"
            if int(getattr(corridor, "refinement_used", 0)) == 1
            else "raw_waypoints")
        p = np.asarray(target, float)
        global_idx = len(self.mpc_reference_records)
        phase_id = int(self.phase)
        phase_name = (
            "handover" if phase_id == 3 else
            "return" if phase_id == 4 else
            "approach")
        self.mpc_reference_records.append({
            "robot": "arm",
            "corridor_id": cid,
            "reference_source": source,
            "phase": phase_name,
            "solve_index": int(self._mpc_reference_solve_index),
            "path_point_index": int(idx),
            "trajectory_point_index": global_idx,
            "timestamp_or_s_index": global_idx,
            "x": float(p[0]),
            "y": float(p[1]),
            "z": float(p[2]) if len(p) > 2 else 0.0,
            "corridor_active": bool(self.corridor_evaluation_active),
        })
        self._mpc_reference_solve_index += 1

    def _write_mpc_reference_path(self):
        if not self.mpc_reference_out or self.baseline:
            return
        out_dir = os.path.dirname(self.mpc_reference_out)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        fields = [
            "robot", "corridor_id", "reference_source", "phase", "solve_index",
            "path_point_index", "trajectory_point_index",
            "timestamp_or_s_index", "x", "y", "z", "corridor_active",
        ]
        with open(self.mpc_reference_out, "w") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in self.mpc_reference_records:
                writer.writerow({key: row.get(key, "") for key in fields})
        rospy.loginfo("[handover][trace] wrote MPC reference path %s rows=%d",
                      self.mpc_reference_out, len(self.mpc_reference_records))

    def _arm_handover_debug_path(self):
        if self.arm_handover_debug_out:
            return self.arm_handover_debug_out
        base = os.path.dirname(self.decision_trace_out or self.mpc_reference_out)
        if base:
            return os.path.join(base, "arm_handover_debug.json")
        return os.path.join(
            ROOT, "results", "run", "arm", "stsm",
            "arm_handover_debug.json")

    def _mpc_handover_diagnostics_path(self):
        if self.mpc_handover_diagnostics_out:
            return self.mpc_handover_diagnostics_out
        base = os.path.dirname(
            self.mpc_reference_out or self.decision_trace_out or
            self.mpc_diagnostics_out)
        if base:
            return os.path.join(base, "mpc_handover_diagnostics.json")
        return os.path.join(
            ROOT, "results", "run", "arm", "stsm",
            "mpc_handover_diagnostics.json")

    def _write_mpc_handover_diagnostics(self):
        if self.baseline:
            return
        payload = dict(self.mpc_handover_diagnostics)
        records = list(payload.get("records", []) or [])
        chain = list(payload.get("phase3_execution_chain", []) or [])
        before_errors = [
            float(r.get("before_error", 0.0)) for r in records
            if r.get("before_error", "") != ""]
        after_errors = [
            float(r.get("after_error", 0.0)) for r in records
            if r.get("after_error", "") != ""]
        payload.update({
            "record_count": int(len(records)),
            "phase3_execution_chain_count": int(len(chain)),
            "handover_reference_point": [
                float(v) for v in np.asarray(self.handover, float)[:3]],
            "phase3_before_after": {
                "before": [
                    item for item in chain
                    if item.get("event") ==
                    "phase3_enter_before_protected_approach"
                ][-1:] or [],
                "after": [
                    item for item in chain
                    if item.get("event") ==
                    "phase3_after_protected_approach"
                ][-1:] or [],
            },
            "max_before_error": (
                float(max(before_errors)) if before_errors else None),
            "max_after_error": (
                float(max(after_errors)) if after_errors else None),
            "mean_before_error": (
                float(np.mean(before_errors)) if before_errors else None),
            "mean_after_error": (
                float(np.mean(after_errors)) if after_errors else None),
        })
        path = self._mpc_handover_diagnostics_path()
        norm_path = path.replace("\\", "/")
        if "/arm/STSM/" in norm_path:
            path = path.replace("/arm/STSM/", "/arm/stsm/")
        out_dir = os.path.dirname(path)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        rospy.loginfo("[handover][mpc] wrote handover diagnostics %s", path)

    def _mpc_trajectory_debug_summary(self):
        refs = list(self.mpc_reference_records or [])
        samples = list(self.arm_ee_debug_samples or [])
        pairs = []
        errors = []
        count = min(len(refs), len(samples))
        for idx in range(count):
            ref = refs[idx]
            sample = samples[idx]
            ref_p = np.array([
                float(ref.get("x", 0.0)),
                float(ref.get("y", 0.0)),
                float(ref.get("z", 0.0)),
            ], float)
            ee_p = np.asarray(sample.get("position", [0.0, 0.0, 0.0]), float)
            err = float(np.linalg.norm(ee_p[:3] - ref_p[:3]))
            errors.append(err)
            if idx < 200:
                pairs.append({
                    "index": int(idx),
                    "phase": sample.get("phase", ""),
                    "reference": [float(v) for v in ref_p],
                    "actual_ee": [float(v) for v in ee_p[:3]],
                    "position_error": err,
                })
        return {
            "reference_count": int(len(refs)),
            "actual_sample_count": int(len(samples)),
            "paired_count": int(count),
            "mean_position_error": (
                float(np.mean(errors)) if errors else None),
            "max_position_error": (
                float(np.max(errors)) if errors else None),
            "pairs_preview": pairs,
        }

    def _write_arm_handover_debug(self):
        if self.baseline:
            return
        out_path = self._arm_handover_debug_path()
        norm_path = out_path.replace("\\", "/")
        if "/arm/STSM/" in norm_path:
            out_path = out_path.replace("/arm/STSM/", "/arm/stsm/")
        out_paths = [out_path]
        ee = self._ee_pos()
        quat = self._ee_quat()
        pos_err = float(np.linalg.norm(ee - self.handover))
        orientation_ref = quat
        checks = self.arm_handover_debug.get("handover_checks", [])
        if checks:
            handover_pos_err = float(checks[-1].get("position_error", pos_err))
            orientation_err = float(checks[-1].get("orientation_error", 0.0))
        else:
            handover_pos_err = pos_err
            orientation_err = self._quat_error(quat, orientation_ref)
        payload = dict(self.arm_handover_debug)
        payload.update({
            "state_machine_current": {
                "phase": int(self.phase),
                "stop_triggered": bool(self.stop_triggered),
                "stop_reason": str(self.stop_reason),
            },
            "target_pose": {
                "position": [float(v) for v in np.asarray(self.handover, float)[:3]],
                "orientation_reference": payload.get(
                    "target_pose", {}).get(
                        "orientation_xyzw", "captured_at_handover_check"),
            },
            "ee_actual_pose": {
                "position": [float(v) for v in ee],
                "orientation_xyzw": [float(v) for v in quat],
            },
            "errors": {
                "position_error": handover_pos_err,
                "orientation_error": orientation_err,
                "final_ee_to_handover_position_error": pos_err,
            },
            "thresholds": {
                "handover_pos_tol": float(self.handover_pos_tol),
                "handover_orientation_tol": float(self.handover_orientation_tol),
                "handover_speed_tol": float(self.handover_speed_tol),
                "handover_hold_s": float(self.handover_hold_s),
                "handover_verify_timeout_s": float(self.handover_verify_timeout_s),
                "require_gripper_handover_event": bool(
                    self.require_gripper_handover_event),
            },
            "mpc_trajectory_difference": self._mpc_trajectory_debug_summary(),
        })
        written = []
        for path in sorted(set(out_paths)):
            out_dir = os.path.dirname(path)
            if out_dir and not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            with open(path, "w") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            written.append(path)
        rospy.loginfo("[handover][debug] wrote %s", ",".join(written))

    def _mpc_output_paths(self):
        base = os.path.dirname(self.mpc_reference_out or self.decision_trace_out)
        diag = self.mpc_diagnostics_out or os.path.join(
            base, "mpc_diagnostics.json")
        breakdown = self.mpc_cost_breakdown_out or os.path.join(
            base, "mpc_cost_breakdown.csv")
        return diag, breakdown

    def _runtime_mpc_diagnostic_fields(self):
        return {
            "runtime_solver_status": str(self.mpc.last_solver_status),
            "runtime_horizon": int(self.mpc.N),
            "runtime_predicted_joint_states": list(
                self.mpc.last_predicted_joint_states),
            "runtime_predicted_controls": list(
                self.mpc.last_predicted_controls),
            "runtime_predicted_ee_states": list(
                self.mpc.last_predicted_ee_states),
            "runtime_objective_terms": dict(self.mpc.last_objective_terms),
            "runtime_constraint_violation": dict(
                self.mpc.last_constraint_violation),
            "runtime_control_sequence_varies": bool(
                self.mpc.last_control_sequence_varies),
            "kinematics_source": str(self.mpc.last_kinematics_source),
            "interest_point_kinematics_source": str(
                self.mpc.last_interest_kinematics_source),
            "joint_limit_source": str(self.arm_joint_limit_source),
            "prediction_model": str(self.mpc.last_prediction_model),
            "dls_role": "warm_start_only",
        }

    def _write_mpc_diagnostics(self):
        if self.baseline:
            return
        corr = self.execution_corridor
        cid = str(getattr(corr, "corridor_id", getattr(corr, "label", "")))
        ref = [
            dict(row) for row in self.mpc_reference_records
            if bool(row.get("corridor_active", True))]
        if not ref:
            rospy.logerr(
                "[handover][mpc] no active selected-corridor reference rows")
            diag, breakdown = self._mpc_output_paths()
            result = {
                "robot_type": "arm",
                "selected_corridor_id": cid or "planning_failed",
                "selected_corridor_label": str(getattr(
                    corr, "label", cid or "planning_failed")),
                "horizon": int(self.mpc.N),
                "dt": 0.1,
                "mpc_used": bool(self.mpc.solve_count > 0),
                "mpc_feasibility_status": "infeasible_reference_empty",
                "final_status": "infeasible_reference_empty",
                "final_mpc_status": "infeasible_reference_empty",
                "task_success": bool(self.task_completed),
                "planner_success": False,
                "controller_success": bool(self.mpc.solve_success_count > 0),
                "safety_success": bool(self.mpc.fallback_count == 0),
                "overall_success": False,
                "success": False,
                "failure_reason": self.stop_reason or "reference_empty",
                "reference_path_count": 0,
            }
            result.update(self._runtime_mpc_diagnostic_fields())
            result["task_state_diagnostics"] = list(
                self.task_context_records)
            write_mpc_outputs(result, diag, breakdown)
            return
        ref_points = [
            [row.get("x", 0.0), row.get("y", 0.0), row.get("z", 0.0)]
            for row in ref
        ]
        diagnostic_phase = str(ref[0].get("phase", "approach") or "approach")
        constraints = {
            "joint_velocity_max": float(self.mpc.dq_max),
            "ee_speed_max": float(self.mpc.v_cap),
            "control_delta_max": 0.1,
            "robot_type": "arm",
            "phase": diagnostic_phase,
            "risk_threshold": float(self.arm_interest_rho_stop),
            "manifold_threshold": float(self.arm_interest_rho_stop),
            "clearance_threshold": 0.08,
            "minimum_clearance": 0.08,
            "manifold_constraint_mode": self.manifold_constraint_mode,
            "mpc_manifold_constraint_mode": self.manifold_constraint_mode,
            "manifold_soft_tolerance": float(self.mpc_config.get(
                "manifold_soft_tolerance", 0.08)),
            "manifold_hard_tolerance": float(self.mpc_config.get(
                "manifold_hard_tolerance", 0.25)),
            "strict_risk_query": bool(
                self.experiment_mode == "paper" and not self.baseline),
        }
        # Replay diagnostics from the selected corridor entry, not from the
        # post-task home pose that exists when evidence is written.
        state = np.asarray(ref_points[0], float)
        topology_info, corridor_info, manifold_info, topology_constraint_info = (
            build_mpc_constraint_inputs(
                corr, self.manifold, ref_points,
                safe_threshold=float(self.arm_interest_rho_stop),
                minimum_clearance=0.08,
                phase=diagnostic_phase,
                robot_type="arm",
                manifold_constraint_mode=self.manifold_constraint_mode,
                phase_params=self.manifold_phase_config,
                strict_stsm=bool(not self.baseline),
                expected_corridor_id=str(getattr(corr, "corridor_id", ""))))
        dbg = dict(getattr(self.manifold, "last_topology_debug", {}) or {})
        dbg["topology_constraint_info"] = topology_constraint_info
        self.manifold.last_topology_debug = dbg
        result = run_mpc_tracking(
            "arm", state, ref,
            topology_info, corridor_info, manifold_info, self.field,
            constraints,
            horizon=10, dt=0.1, selected_corridor_id=cid,
            risk_threshold=float(self.arm_interest_rho_stop),
            config={
                "task_mode": self.task_mode,
                "task_config": self.task_config,
                "task_weight": self.task_weight,
                "task_context": dict(self.task_context),
                "task_state_diagnostics": list(self.task_context_records),
                "effective_social_weights": self.field.get_effective_weights(),
                "weights": self.mpc_cost_weights,
                "phase_cost_weights": self.mpc_phase_cost_weights,
                "phase_clearance_schedule": self.manifold_phase_config,
                "executed_trajectory": [
                    row["point"] for row in self.mpc_executed_records],
                "executed_phase_sequence": [
                    row["phase"] for row in self.mpc_executed_records],
                "executed_corridor_active_sequence": [
                    bool(row.get("corridor_active", False))
                    for row in self.mpc_executed_records],
                "executed_evidence_required": True,
            })
        result["selected_corridor_label"] = str(getattr(corr, "label", cid))
        result.update(self._runtime_mpc_diagnostic_fields())
        if corr is not None:
            result.update({
                "raw_candidate_corridor_violation_count": int(getattr(
                    corr, "raw_candidate_corridor_violation_count", 0)),
                "refined_corridor_violation_count": int(getattr(
                    corr, "trajectory_corridor_violation_count", 0)),
                "fallback_corridor_violation_count": int(getattr(
                    corr, "fallback_corridor_violation_count", 0)),
                "mpc_reference_corridor_violation_count": int(getattr(
                    corr, "mpc_reference_corridor_violation_count", 0)),
                "execution_tube_source": "selected_topology_route_with_task_endpoints",
                "execution_reference_feasibility": dict(getattr(
                    corr, "execution_reference_feasibility", {})),
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
        diag, breakdown = self._mpc_output_paths()
        topology_constraint_path = os.path.join(
            os.path.dirname(diag), "topology_constraint.json")
        write_topology_constraint(topology_constraint_path, topology_constraint_info)
        write_mpc_outputs(result, diag, breakdown)
        tube_path = os.path.join(os.path.dirname(diag), "topology_tube.json")
        with open(tube_path, "w") as f:
            json.dump(generate_topology_tube(
                corridor_info.get("centerline", ref),
                corridor_info.get("radius", 0.35)), f, indent=2, sort_keys=True)
        self.path_adp_info["mpc_total_cost"] = float(result.get("total_cost", 0.0))
        self.path_adp_info["mpc_tracking_cost"] = float(result.get("tracking_cost", 0.0))
        self.path_adp_info["mpc_control_cost"] = float(result.get("control_cost", 0.0))
        self.path_adp_info["mpc_risk_cost"] = float(result.get("risk_cost", 0.0))
        rospy.loginfo("[handover][trace] wrote MPC diagnostics %s", diag)

    def _runtime_metrics_for_trace(self):
        corr = self.execution_corridor
        affects_control = bool(
            self.adp_mpc_influence_enabled and (
                float(getattr(self.mpc, "last_dls_adp_used", 0.0)) > 0.5 or
                float(getattr(self.mpc, "last_v_des_delta_norm", 0.0)) > 1e-9))
        affects_ranking = bool(self.adp_ranking_influence_enabled and corr is not None)
        return {
            "target": "arm",
            "variant": "stsm" if not self.baseline else "baseline",
            "topology_fallback_used": 0,
            "adp_enabled": 1 if self.adp_enabled else 0,
            "adp_decision_influence_enabled": int(self.adp_decision_influence_enabled),
            "adp_ranking_influence_enabled": int(self.adp_ranking_influence_enabled),
            "adp_mpc_influence_enabled": int(self.adp_mpc_influence_enabled),
            "mpc_adp_enabled": int(self.adp_mpc_influence_enabled),
            "adp_effective_lambda": float(self.adp_learning.config.get("lambda_adp", 0.0)
                                              if self.adp_learning and self.adp_ranking_influence_enabled else 0.0),
            "adp_role": adp_role_from_runtime(
                self.adp_enabled, bool(self.adp_learning and self.adp_learning.config.get("enabled", False)),
                self.adp_decision_influence_enabled,
                effective_lambda=(self.adp_learning.config.get("lambda_adp", 0.0)
                                  if self.adp_learning else 0.0),
                ranking_contribution=affects_ranking,
                control_contribution=affects_control),
            "adp_affects_candidate_ranking": int(affects_ranking),
            "adp_affects_control": int(affects_control),
            "corridor_rank_changed_count": int(bool(
                corr is not None and getattr(corr, "adp_changed_rank", False))),
            "arm_dls_adp_used": 1 if affects_control else 0,
            "v_des_delta_norm": float(getattr(self.mpc, "last_v_des_delta_norm", 0.0)),
            "selected_refinement_used": int(getattr(corr, "refinement_used", 0)) if corr is not None else 0,
            "selected_tracking_cost": float(getattr(corr, "tracking_cost", 0.0)) if corr is not None else 0.0,
            "mpc_track_cost": float(self.path_adp_info.get("topology_tracking_error", 0.0)),
            "mpc_social_cost": float(getattr(self.mpc, "last_adp_soft_cost", 0.0)),
            "mpc_total_cost": float(self.path_adp_info.get("mpc_total_cost", 0.0)),
            "mpc_control_cost": float(self.path_adp_info.get("mpc_control_cost", 0.0)),
            "mpc_feasibility_status": "feasible",
        }

    def _write_decision_trace(self):
        if not self.decision_trace_out or self.baseline:
            return
        debug = dict(getattr(self.manifold, "last_topology_debug", {}) or {})
        trace = trace_from_debug(
            debug, self._runtime_metrics_for_trace(), "arm", "stsm")
        trace["mpc_reference_path_file"] = self.mpc_reference_out
        write_trace(trace, self.decision_trace_out)
        rospy.loginfo("[handover][trace] wrote decision trace %s",
                      self.decision_trace_out)

    def _write_runtime_evidence(self):
        self._write_mpc_reference_path()
        self._write_mpc_diagnostics()
        self._write_decision_trace()
        try:
            self._write_mpc_handover_diagnostics()
        except Exception as exc:
            rospy.logwarn(
                "[handover][mpc] failed to write handover diagnostics: %s",
                exc)
        try:
            self._write_arm_handover_debug()
        except Exception as exc:
            rospy.logwarn("[handover][debug] failed to write handover debug: %s",
                          exc)

    def _wait_for_task_completion_stable(self, target):
        if target is None:
            return False
        target = np.asarray(target, float)
        stable = 0
        prev_ee = None
        prev_t = None
        rate = rospy.Rate(1.0 / max(self.task_complete_check_dt, 1e-3))
        max_cycles = max(1, int(self.task_complete_max_cycles))
        need_cycles = max(1, int(self.task_complete_stable_cycles))
        for _ in range(max_cycles):
            if rospy.is_shutdown():
                return False
            ee = self._ee_pos()
            now = rospy.Time.now().to_sec()
            pos_err = float(np.linalg.norm(ee - target))
            speed = 0.0
            if prev_ee is not None and prev_t is not None:
                dt = now - prev_t
                if dt > 1e-6:
                    speed = float(np.linalg.norm(ee - prev_ee) / dt)
            self._publish_metrics(ee)
            if (pos_err <= self.task_complete_pos_tol and
                    speed <= self.task_complete_speed_tol):
                stable += 1
            else:
                stable = 0
            if stable >= need_cycles:
                self._hold_current()
                rospy.loginfo(
                    "[handover] task completion stable pos_err=%.4f speed=%.4f cycles=%d",
                    pos_err, speed, stable)
                return True
            prev_ee = np.array(ee, float)
            prev_t = now
            rate.sleep()
        rospy.logwarn(
            "[handover] task completion stability timeout target_err=%.4f speed=%.4f stable=%d/%d",
            pos_err, speed, stable, need_cycles)
        return False

    def _return_home(self):
        try:
            rospy.loginfo("[handover] returning arm to home")
            self._set_phase(4)
            self.group.set_named_target("home")
            self.group.go(wait=True)
            self.group.stop()
            return bool(self._wait_for_task_completion_stable(self.home_ee_ref))
        except Exception as exc:
            rospy.logwarn("[handover] return home failed: %s", exc)
            return False

    def _log_done(self):
        self._record_adp_transition(
            self._ee_pos(), control=np.zeros(6), terminal=True)
        self._write_runtime_evidence()
        self._write_adp_learning_diagnostics()
        rospy.loginfo(
            "Arm solver summary: calls=%d, success=%d, fallback=%d, status=%s",
            self.mpc.solve_count,
            self.mpc.solve_success_count,
            self.mpc.fallback_count,
            self.mpc.last_solver_status)
        rospy.loginfo("[handover] done (mode=%s, stop=%s, reason=%s)",
                      "baseline" if self.baseline else "stsm",
                      self.stop_triggered, self.stop_reason)

if __name__ == "__main__":
    try:
        HandoverNode().run()
    except rospy.ROSInterruptException:
        pass
