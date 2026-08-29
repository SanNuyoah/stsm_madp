#!/usr/bin/env python
import csv
import json
import os
import sys
sys.dont_write_bytecode = True
import numpy as np
import rospy
from std_msgs.msg import Bool, Float64, Float64MultiArray, Int32, String
from geometry_msgs.msg import PointStamped

PACKAGE_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if PACKAGE_SRC not in sys.path:
    sys.path.insert(0, PACKAGE_SRC)
from stsm_madp.adp import adp_role_from_runtime

class MetricsNode:
    def __init__(self):
        rospy.init_node("stsm_metrics")
        self.target = rospy.get_param("~target", "arm")
        self.out = rospy.get_param(
            "~out", os.path.expanduser("~/stsm_metrics.csv"))
        self.traj_out = rospy.get_param("~traj_out", "")
        self.mpc_diagnostics_path = rospy.get_param("~mpc_diagnostics", "")
        self.trajectory_debug_level = int(rospy.get_param(
            "~trajectory_debug_level", 1))
        self.run_id = rospy.get_param("/stsm/run_id",
                                      rospy.get_param("~run_id", ""))
        self.scenario = rospy.get_param("~scenario", "eldercare")
        self.variant = rospy.get_param("~variant", "")
        self.controller_version = rospy.get_param("~controller_version",
                                                 "gate_v1")
        self.interest_required = bool(rospy.get_param("~interest_required", True))
        self.dt = 0.05
        self.mode = "unknown"
        self.reset()
        if self.target == "arm":
            self.risk_threshold = 1.6
            self.head = np.array(rospy.get_param("~head", [0.78, 0.0, 0.61]))
            self.chest = np.array(rospy.get_param("~chest", [0.78, 0.0, 0.31]))
            self.hand = np.array(rospy.get_param("~hand", [0.42, 0.0, 0.21]))
            self.wait_pose = np.array(rospy.get_param(
                "~wait_pose", [0.30, 0.0, 0.30]))
            self.hand_tolerance = float(rospy.get_param(
                "~hand_tolerance", 0.08))
            self.wait_tolerance = float(rospy.get_param(
                "~wait_tolerance", 0.10))
            self.home_tolerance = float(rospy.get_param(
                "~home_tolerance", 0.12))
            self.hold_min_s = float(rospy.get_param("~hold_min_s", 1.0))
            rospy.Subscriber("/stsm/ee_pose", PointStamped, self._arm_pose_cb)
            rospy.Subscriber("/stsm/phi_s", Float64, self._phi_cb)
            rospy.Subscriber("/stsm/risk_components", Float64MultiArray,
                             self._risk_components_cb)
            rospy.Subscriber("/stsm/velocity_monitor", Float64MultiArray,
                             self._velocity_monitor_cb)
            rospy.Subscriber("/stsm/arm_phase", Int32, self._phase_cb)
            rospy.Subscriber("/stsm/mode", String, self._mode_cb)
            rospy.Subscriber("/stsm/arm_gate_state", String,
                             self._gate_state_cb)
            rospy.Subscriber("/stsm/arm_gate_info", Float64MultiArray,
                             self._gate_info_cb)
            rospy.Subscriber("/stsm/arm_gate_reason", String,
                             self._gate_reason_cb)
            rospy.Subscriber("/stsm/arm_gate_source", String,
                             self._arm_gate_source_cb)
            rospy.Subscriber("/stsm/arm_interest_gate_info", Float64MultiArray,
                             self._arm_interest_gate_cb)
            rospy.Subscriber("/stsm/arm_interest_risk", Float64MultiArray,
                             self._arm_interest_cb)
            rospy.Subscriber("/stsm/arm_interest_points", Float64MultiArray,
                             self._arm_interest_points_cb)
            rospy.Subscriber("/stsm/arm_adp_value", Float64,
                             self._adp_value_cb)
            rospy.Subscriber("/stsm/arm_adp_control_info", Float64MultiArray,
                             self._arm_adp_control_info_cb)
            rospy.Subscriber("/stsm/arm_selected_corridor", String,
                             self._selected_corridor_cb)
            rospy.Subscriber("/stsm/arm_topology_info", Float64MultiArray,
                             self._topology_info_cb)
            rospy.Subscriber("/stsm/arm_handover_status", Float64MultiArray,
                             self._arm_handover_status_cb)
        else:
            self.risk_threshold = 0.8
            self.person = np.array(rospy.get_param("~person", [-1.6, 0.2]))
            self.transfer_c = np.array(rospy.get_param("~transfer_c", [-0.7, -1.0]))
            self.transfer_h = np.array(rospy.get_param("~transfer_h", [0.4, 1.0]))
            self.goal = np.array(rospy.get_param("~goal", [-0.55, 0.55]))
            self.success_goal_tolerance = float(rospy.get_param(
                "~success_goal_tolerance", 0.08))
            rospy.Subscriber("/stsm/wc_pos", PointStamped, self._wc_pose_cb)
            rospy.Subscriber("/stsm/wc_phi_s", Float64, self._phi_cb)
            rospy.Subscriber("/stsm/wc_risk_components", Float64MultiArray,
                             self._risk_components_cb)
            rospy.Subscriber("/stsm/wc_velocity_monitor", Float64MultiArray,
                             self._velocity_monitor_cb)
            rospy.Subscriber("/stsm/wc_mode", String, self._mode_cb)
            rospy.Subscriber("/stsm/wc_gate_state", String,
                             self._gate_state_cb)
            rospy.Subscriber("/stsm/wc_gate_info", Float64MultiArray,
                             self._gate_info_cb)
            rospy.Subscriber("/stsm/wc_gate_reason", String,
                             self._gate_reason_cb)
            rospy.Subscriber("/stsm/wc_gate_source", String,
                             self._gate_source_cb)
            rospy.Subscriber("/stsm/wc_interest_gate_info", Float64MultiArray,
                             self._wc_interest_gate_cb)
            rospy.Subscriber("/stsm/wc_interest_risk", Float64MultiArray,
                             self._wc_interest_cb)
            rospy.Subscriber("/stsm/wc_pose2d", Float64MultiArray,
                             self._wc_pose2d_cb)
            rospy.Subscriber("/stsm/wc_adp_value", Float64,
                             self._adp_value_cb)
            rospy.Subscriber("/stsm/wc_adp_mpc_info", Float64MultiArray,
                             self._wc_adp_mpc_info_cb)
            rospy.Subscriber("/stsm/wc_selected_corridor", String,
                             self._selected_corridor_cb)
            rospy.Subscriber("/stsm/wc_topology_info", Float64MultiArray,
                             self._topology_info_cb)
            rospy.Subscriber("/stsm/wc_task_complete", Bool,
                             self._wc_task_complete_cb)
        rospy.Subscriber("/stsm/adp_status", String, self._adp_status_cb)
        rospy.Subscriber("/stsm/arm_adp_path_info", Float64MultiArray,
                         self._arm_adp_path_info_cb)
        rospy.on_shutdown(self._finish)

    def _load_mpc_diagnostics(self):
        if not self.mpc_diagnostics_path:
            return {}
        try:
            if not os.path.exists(self.mpc_diagnostics_path):
                return {}
            with open(self.mpc_diagnostics_path) as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            rospy.logwarn("[metrics] failed to read MPC diagnostics %s: %s",
                          self.mpc_diagnostics_path, exc)
            return {}

    def reset(self):
        self.prev_pos = None
        self.prev_t = None
        self.min_head = np.inf
        self.min_chest = np.inf
        self.min_person = np.inf
        self.max_speed = 0.0
        self.speed_near = []
        self.j_social = 0.0
        self.last_risk_time = None
        self.phi_sum = 0.0
        self.phi_n = 0
        self.phi_values = []
        self.max_phi = 0.0
        self.risk_exceed_s = 0.0
        self.valid_risk_duration_s = 0.0
        self.transfer_time = 0.0
        self.reached = False
        self.wc_task_completed = False
        self.last_pos = None
        self.path_length_m = 0.0
        self.phase = ""
        self.phase_stop = ""
        self.arm_home_ref = None
        self.arm_min_hand = np.inf
        self.arm_min_wait = np.inf
        self.arm_min_home_after_return = np.inf
        self.arm_reached_hand = False
        self.arm_retreat_started = False
        self.arm_wait_reached = False
        self.arm_home_return_started = False
        self.arm_home_returned = False
        self.arm_goal_reached = False
        self.arm_handover_pose_reached = False
        self.arm_handover_complete = False
        self.arm_handover_pos_err = ""
        self.arm_handover_orientation_err = ""
        self.arm_handover_stable_time_s = 0.0
        self.arm_gripper_handover_event = False
        self.arm_hold_first_t = None
        self.arm_hold_last_t = None
        self.arm_hold_duration_s = 0.0
        self.gate_state = "NORMAL"
        self.gate_reason = ""
        self.gate_scale = 1.0
        self.gate_stop = False
        self.rho_warn = ""
        self.rho_stop = ""
        self.stop_triggered = False
        self.stop_reason = ""
        self.first_stop_time_s = ""
        self.stop_row_written = False
        self.gate_slow_count = 0
        self.gate_stop_count = 0
        self.max_gate_risk = 0.0
        self.gate_scale_sum = 0.0
        self.gate_scale_n = 0
        self.gate_source = ""
        self.arm_gate_source = ""
        self.wc_interest_received = False
        self.wc_pose2d_received = False
        self.arm_interest_received = False
        self.arm_interest_points_received = False
        self.arm_interest_gate_received = False
        self.arm_gate_source_received = False
        self.footprint_gate_enabled = 0
        self.footprint_gate_risk = ""
        self.footprint_gate_scale = ""
        self.footprint_gate_stop = 0
        self.footprint_rho_warn = ""
        self.footprint_rho_stop = ""
        self.footprint_slow_count = 0
        self.footprint_stop_count = 0
        self.first_footprint_stop_time_s = ""
        self.footprint_stop_reason = ""
        self.wc_yaw = ""
        self.latest_wc_interest = {
            "phi_center": "",
            "phi_front_center": "",
            "phi_front_left": "",
            "phi_front_right": "",
            "phi_footrest_left": "",
            "phi_footrest_right": "",
            "phi_rear_left": "",
            "phi_rear_right": "",
            "phi_max_point": "",
            "phi_mean_point": "",
            "phi_sum_point": "",
            "worst_point_idx": "",
            "forbidden_hit": "",
        }
        self.ip_phi_max_sum = 0.0
        self.ip_phi_max_n = 0
        self.ip_max_phi_point = 0.0
        self.forbidden_hit_count = 0
        self.latest_arm_interest = {
            "phi_ee_point": "",
            "phi_wrist": "",
            "phi_elbow": "",
            "phi_object": "",
            "phi_arm_max_point": "",
            "phi_arm_mean_point": "",
            "phi_arm_sum_point": "",
            "arm_worst_point_idx": "",
            "arm_interest_valid_count": "",
        }
        self.latest_arm_interest_points = {
            "arm_ip_ee_x": "",
            "arm_ip_ee_y": "",
            "arm_ip_ee_z": "",
            "arm_ip_wrist_x": "",
            "arm_ip_wrist_y": "",
            "arm_ip_wrist_z": "",
            "arm_ip_elbow_x": "",
            "arm_ip_elbow_y": "",
            "arm_ip_elbow_z": "",
            "arm_ip_object_x": "",
            "arm_ip_object_y": "",
            "arm_ip_object_z": "",
        }
        self.arm_ip_phi_max_sum = 0.0
        self.arm_ip_phi_max_n = 0
        self.arm_ip_max_phi_point = 0.0
        self.arm_interest_gate_enabled = 0
        self.arm_interest_gate_risk = ""
        self.arm_interest_gate_scale = ""
        self.arm_interest_gate_stop = 0
        self.arm_interest_rho_warn = ""
        self.arm_interest_rho_stop = ""
        self.arm_interest_gate_worst_idx = ""
        self.arm_interest_slow_count = 0
        self.arm_interest_stop_count = 0
        self.arm_interest_gate_risk_sum = 0.0
        self.arm_interest_gate_risk_n = 0
        self.max_arm_interest_gate_risk = 0.0
        self.latest_risk_components = {
            "phi_prox": "",
            "phi_close": "",
            "phi_dir": "",
            "phi_body": "",
            "phi_env": "",
            "phi_total": "",
        }
        self.latest_velocity_monitor = {
            "vx": "",
            "vy": "",
            "vz": "",
            "speed_raw": "",
            "speed_filtered": "",
            "phi_close_monitor": "",
            "dt_used": "",
            "velocity_valid": "",
        }
        self.adp_value = ""
        self.prev_adp_value = ""
        self.adp_delta = ""
        self.adp_enabled = 0
        self.critic_version = ""
        self.adp_values = []
        self.adp_clip_value = float(rospy.get_param("~adp_clip_value", 100.0))
        self.adp_clip_hits = 0
        self.selected_corridor_label = ""
        self.latest_topology_info = {
            "topology_enabled": "",
            "topology_used": "",
            "topology_fallback_used": "",
            "num_critical_minima": "",
            "num_critical_saddles": "",
            "num_critical_maxima": "",
            "num_raw_minima": "",
            "num_raw_saddles": "",
            "num_raw_maxima": "",
            "num_safe_minima": "",
            "num_safe_saddles": "",
            "num_safe_maxima": "",
            "num_filtered_minima": "",
            "num_filtered_saddles": "",
            "num_filtered_maxima": "",
            "num_usable_minima": "",
            "num_usable_saddles": "",
            "num_used_minima": "",
            "num_used_saddles": "",
            "num_forced_critical_corridors": "",
            "num_morse_minima_corridors": "",
            "num_morse_saddle_corridors": "",
            "num_morse_mix_corridors": "",
            "num_graph_direct_corridors": "",
            "num_graph_semantic_corridors": "",
            "reject_by_gradient_count": "",
            "reject_by_degenerate_count": "",
            "reject_by_forbidden_count": "",
            "reject_by_clearance_count": "",
            "reject_by_unsafe_count": "",
            "num_topology_nodes": "",
            "num_topology_edges": "",
            "num_candidate_corridors": "",
            "topology_grid_resolution": "",
            "topology_rho": "",
            "num_forbidden_cells": "",
            "selected_corridor_forbidden_hits": "",
            "candidate_forbidden_reject_count": "",
            "clearance_reject_count": "",
            "edge_clearance_reject_count": "",
            "edge_forbidden_reject_count": "",
            "edge_astar_fail_count": "",
            "neighbor_pair_attempt_count": "",
            "topology_hard_clearance": "",
            "topology_clearance_target": "",
            "topology_neighbor_k": "",
            "selected_saddle_value_bonus": "",
            "selected_candidate_total_score": "",
            "selected_tracking_cost": "",
            "selected_max_curvature": "",
            "selected_curvature_violation": "",
            "selected_turn_violation": "",
            "selected_expected_progress": "",
            "candidate_min_clearance": "",
            "candidate_max_risk": "",
            "candidate_manifold_feasible": "",
            "candidate_manifold_valid": "",
            "candidate_tube_valid": "",
            "num_manifold_filtered_candidates": "",
            "filtered_infeasible_candidates": "",
            "planning_clearance_margin": "",
        }
        self.latest_adp_mpc_info = {
            "corridor_base_cost": "",
            "corridor_adp_mean": "",
            "corridor_adp_max": "",
            "corridor_adp_end": "",
            "corridor_total_cost": "",
            "corridor_adp_raw_mean": "",
            "corridor_adp_raw_max": "",
            "corridor_adp_raw_end": "",
            "corridor_adp_norm": "",
            "corridor_rank_base": "",
            "corridor_rank_total": "",
            "terminal_adp_cost": "",
            "mpc_total_cost": "",
            "mpc_social_cost": "",
            "mpc_tube_cost": "",
            "mpc_track_cost": "",
            "mpc_control_cost": "",
            "corridor_rank_changed_count": "",
            "final_approach_used": "",
            "mpc_reject_forbidden_count": "",
            "mpc_reject_interest_phi_count": "",
            "adp_learning_enabled": "",
            "adp_decision_influence_enabled": "",
            "adp_effective_lambda": "",
            "adp_ranking_influence_enabled": "",
            "adp_mpc_influence_enabled": "",
        }
        self.path_adp_info = {
            "path_adp_mean": "",
            "path_adp_max": "",
            "path_adp_delta": "",
            "adp_path_enabled": "",
            "protected_saddle_count": "",
            "protected_saddle_max_dist": "",
            "protected_saddle_ok": "",
            "mandatory_topology_node_count": "",
            "mandatory_saddle_reached": "",
            "mandatory_saddle_max_dist": "",
            "corridor_violation_count": "",
            "topology_tracking_error": "",
            "mpc_segment_count": "",
            "arm_adp_grad_norm": "",
            "arm_adp_soft_cost": "",
            "arm_v_adp_alignment": "",
            "arm_adp_control_enabled": "",
            "arm_dls_adp_used": "",
            "arm_qp_used": "",
            "arm_solver_success_count": "",
            "arm_solver_fallback_count": "",
            "arm_solver_success_rate": "",
            "v_des_raw_norm": "",
            "v_des_adp_norm": "",
            "v_des_delta_norm": "",
            "dq_nominal_norm": "",
            "dq_adp_norm": "",
            "dq_delta_norm": "",
            "mpc_reject_forbidden_count": "",
            "mpc_reject_interest_phi_count": "",
        }
        self.corridor_rank_changed_count = 0
        self.traj = []

    def _mode_cb(self, msg):
        self.mode = msg.data

    def _phase_cb(self, msg):
        self.phase = int(msg.data)
        if self.target == "arm":
            if self.phase >= 3:
                self.arm_retreat_started = True
            if self.phase >= 4:
                self.arm_home_return_started = True

    def _arm_handover_status_cb(self, msg):
        data = list(msg.data)
        if len(data) >= 1:
            self.arm_goal_reached = bool(data[0] >= 0.5)
        if len(data) >= 2:
            self.arm_handover_pose_reached = bool(data[1] >= 0.5)
        if len(data) >= 3:
            self.arm_handover_complete = bool(data[2] >= 0.5)
        if len(data) >= 4:
            self.arm_handover_pos_err = float(data[3])
        if len(data) >= 5:
            self.arm_handover_orientation_err = float(data[4])
        if len(data) >= 6:
            self.arm_handover_stable_time_s = float(data[5])
        if len(data) >= 7:
            self.arm_gripper_handover_event = bool(data[6] >= 0.5)

    def _gate_state_cb(self, msg):
        self.gate_state = msg.data or "NORMAL"
        if self.gate_state == "SLOW":
            self.gate_slow_count += 1
        elif self.gate_state == "STOP":
            self.gate_stop = True
            self.gate_scale = 0.0
            if not self.gate_reason:
                self.gate_reason = "risk_stop"

    def _gate_reason_cb(self, msg):
        self.gate_reason = msg.data
        if self.stop_triggered and self.gate_reason:
            if not self.stop_reason or self.stop_reason == "risk_stop":
                self.stop_reason = self.gate_reason
            if self.stop_row_written and self.traj:
                for last in self.traj[-2:]:
                    if last.get("gate_state") == "STOP":
                        if (not last.get("gate_reason") or
                                last.get("gate_reason") == "risk_stop"):
                            last["gate_reason"] = self.gate_reason

    def _gate_source_cb(self, msg):
        self.gate_source = msg.data
        if self.stop_row_written and self.traj and self.gate_source:
            for last in self.traj[-2:]:
                if last.get("gate_state") == "STOP":
                    last["gate_source"] = self.gate_source
                    self._fill_footprint_gate_row(last)

    def _fill_footprint_gate_row(self, row):
        if self.target != "wheelchair":
            return
        row["gate_source"] = self.gate_source
        row["footprint_gate_enabled"] = self.footprint_gate_enabled
        row["footprint_gate_risk"] = self.footprint_gate_risk
        row["footprint_gate_scale"] = self.footprint_gate_scale
        row["footprint_gate_stop"] = self.footprint_gate_stop
        row["footprint_rho_warn"] = self.footprint_rho_warn
        row["footprint_rho_stop"] = self.footprint_rho_stop

    def _backfill_stop_footprint_gate_rows(self):
        if self.target != "wheelchair" or not self.stop_row_written or not self.traj:
            return
        for last in self.traj[-2:]:
            if last.get("gate_state") == "STOP":
                self._fill_footprint_gate_row(last)

    def _arm_gate_source_cb(self, msg):
        self.arm_gate_source_received = True
        self.arm_gate_source = msg.data
        if self.stop_row_written and self.traj and self.arm_gate_source:
            for last in self.traj[-2:]:
                if last.get("gate_state") == "STOP":
                    last["arm_gate_source"] = self.arm_gate_source
                    self._fill_arm_interest_gate_row(last)

    def _fill_arm_interest_gate_row(self, row):
        if self.target != "arm":
            return
        row["arm_gate_source"] = self.arm_gate_source
        row["arm_interest_gate_enabled"] = self.arm_interest_gate_enabled
        row["arm_interest_gate_risk"] = self.arm_interest_gate_risk
        row["arm_interest_gate_scale"] = self.arm_interest_gate_scale
        row["arm_interest_gate_stop"] = self.arm_interest_gate_stop
        row["arm_interest_rho_warn"] = self.arm_interest_rho_warn
        row["arm_interest_rho_stop"] = self.arm_interest_rho_stop
        row["arm_interest_gate_worst_idx"] = self.arm_interest_gate_worst_idx

    def _backfill_stop_arm_interest_gate_rows(self):
        if self.target != "arm" or not self.stop_row_written or not self.traj:
            return
        for last in self.traj[-2:]:
            if last.get("gate_state") == "STOP":
                self._fill_arm_interest_gate_row(last)

    def _gate_info_cb(self, msg):
        data = list(msg.data)
        if len(data) >= 1:
            self.max_gate_risk = max(self.max_gate_risk, float(data[0]))
        if len(data) >= 2:
            self.gate_scale = float(data[1])
            self.gate_scale_sum += self.gate_scale
            self.gate_scale_n += 1
        if len(data) >= 3:
            self.gate_stop = bool(data[2] >= 0.5)
        if len(data) >= 4:
            self.rho_warn = float(data[3])
        if len(data) >= 5:
            self.rho_stop = float(data[4])
        if self.gate_stop:
            self.gate_stop_count += 1
            if not self.stop_triggered:
                self.stop_triggered = True
                self.stop_reason = self.gate_reason or "risk_stop"
                self.phase_stop = self.phase
                if self.traj:
                    self.first_stop_time_s = round(
                        rospy.Time.now().to_sec() - float(self.traj[0]["t"]), 4)
                else:
                    self.first_stop_time_s = 0.0
                self._append_stop_traj_row()

    def _append_stop_traj_row(self):
        if self.stop_row_written or self.last_pos is None:
            return
        now = rospy.Time.now().to_sec()
        reason = self.stop_reason or "risk_stop"
        if self.traj:
            last = self.traj[-1]
            last["gate_state"] = "STOP"
            last["gate_scale"] = 0.0
            last["gate_stop"] = 1
            last["gate_reason"] = reason
            last["rho_warn"] = self.rho_warn
            last["rho_stop"] = self.rho_stop
        if self.traj:
            last_t = float(self.traj[-1]["t"])
            if now <= last_t:
                now = last_t + 1e-4
        if self.target == "arm":
            p = np.asarray(self.last_pos, float)
            row = self._traj_row(now, "arm", p[0], p[1], p[2])
        else:
            p = np.asarray(self.last_pos, float)
            row = self._traj_row(now, "wheelchair", p[0], p[1], 0.0)
        row["gate_state"] = "STOP"
        row["gate_scale"] = 0.0
        row["gate_stop"] = 1
        row["gate_reason"] = reason
        row["rho_warn"] = self.rho_warn
        row["rho_stop"] = self.rho_stop
        self.traj.append(row)
        self.stop_row_written = True

    def _risk_components_cb(self, msg):
        keys = ["phi_prox", "phi_close", "phi_dir", "phi_body", "phi_env", "phi_total"]
        for key, value in zip(keys, list(msg.data)):
            self.latest_risk_components[key] = float(value)

    def _velocity_monitor_cb(self, msg):
        keys = [
            "vx", "vy", "vz", "speed_raw", "speed_filtered",
            "phi_close_monitor", "dt_used", "velocity_valid",
        ]
        for key, value in zip(keys, list(msg.data)):
            self.latest_velocity_monitor[key] = float(value)

    def _adp_value_cb(self, msg):
        value = float(msg.data)
        if self.adp_value != "":
            self.adp_delta = value - float(self.adp_value)
        else:
            self.adp_delta = 0.0
        self.prev_adp_value = self.adp_value
        self.adp_value = value
        self.adp_values.append(value)
        if abs(value) >= self.adp_clip_value - 1e-6:
            self.adp_clip_hits += 1
        if abs(value) > 1e-12:
            self.adp_enabled = 1

    def _adp_status_cb(self, msg):
        data = msg.data or ""
        if "loaded:" in data:
            self.adp_enabled = 1
            self.critic_version = data.split("loaded:", 1)[1].strip()
        elif "disabled" in data and not self.critic_version:
            self.critic_version = "disabled"

    def _selected_corridor_cb(self, msg):
        self.selected_corridor_label = msg.data

    def _selected_corridor_type(self):
        label = str(self.selected_corridor_label or "")
        if "_c" in label and label.split("_c", 1)[0] in ("arm", "wheelchair", "generic"):
            return "morse_topology_graph"
        for prefix in (
                "morse_handover_saddle", "morse_handover_minima",
                "morse_saddle", "morse_minima", "graph_direct",
                "graph_semantic", "baseline_direct", "visible_front",
                "arm_fallback_visible_front"):
            if label.startswith(prefix):
                return prefix
        return label.split("_", 1)[0] if label else ""

    def _topology_aliases(self):
        return {
            "raw_minima": self.latest_topology_info.get("num_raw_minima", ""),
            "raw_saddle": self.latest_topology_info.get("num_raw_saddles", ""),
            "safe_minima": self.latest_topology_info.get("num_safe_minima", ""),
            "safe_saddle": self.latest_topology_info.get("num_safe_saddles", ""),
            "filtered_minima": self.latest_topology_info.get("num_filtered_minima", ""),
            "filtered_saddle": self.latest_topology_info.get("num_filtered_saddles", ""),
            "used_minima": self.latest_topology_info.get("num_used_minima", ""),
            "used_saddle": self.latest_topology_info.get("num_used_saddles", ""),
            "used_minima_count": self.latest_topology_info.get("num_used_minima", ""),
            "used_saddle_count": self.latest_topology_info.get("num_used_saddles", ""),
            "candidate_corridor_count": self.latest_topology_info.get(
                "num_candidate_corridors", ""),
            "selected_corridor_type": self._selected_corridor_type(),
        }

    def _topology_info_cb(self, msg):
        keys = [
            "topology_enabled", "topology_used", "topology_fallback_used",
            "num_critical_minima", "num_critical_saddles",
            "num_critical_maxima", "num_raw_minima", "num_raw_saddles",
            "num_raw_maxima", "num_safe_minima", "num_safe_saddles",
            "num_safe_maxima", "num_filtered_minima", "num_filtered_saddles",
            "num_filtered_maxima", "num_usable_minima", "num_usable_saddles",
            "num_used_minima", "num_used_saddles",
            "num_forced_critical_corridors", "num_morse_minima_corridors",
            "num_morse_saddle_corridors", "num_morse_mix_corridors",
            "num_graph_direct_corridors",
            "num_graph_semantic_corridors", "reject_by_gradient_count",
            "reject_by_degenerate_count", "reject_by_forbidden_count",
            "reject_by_clearance_count", "reject_by_unsafe_count",
            "num_topology_nodes",
            "num_topology_edges", "num_candidate_corridors",
            "topology_grid_resolution", "topology_rho",
            "num_forbidden_cells", "selected_corridor_forbidden_hits",
            "candidate_forbidden_reject_count", "clearance_reject_count",
            "edge_clearance_reject_count", "edge_forbidden_reject_count",
            "edge_astar_fail_count", "neighbor_pair_attempt_count",
            "topology_hard_clearance", "topology_clearance_target",
            "topology_neighbor_k",
            "selected_saddle_value_bonus",
            "selected_candidate_total_score",
            "selected_tracking_cost", "selected_max_curvature",
            "selected_curvature_violation", "selected_turn_violation",
            "selected_expected_progress", "selected_refinement_used",
            "selected_refined_path_length", "selected_topology_diversity",
            "selected_raw_waypoints_count",
            "selected_refined_waypoints_count",
            "mpc_used", "mpc_reference_is_refined",
            "candidate_min_clearance", "candidate_max_risk",
            "candidate_manifold_feasible",
            "candidate_manifold_valid", "candidate_tube_valid",
            "num_manifold_filtered_candidates",
            "filtered_infeasible_candidates",
            "planning_clearance_margin",
        ]
        for key, value in zip(keys, list(msg.data)):
            self.latest_topology_info[key] = float(value)

    def _wc_adp_mpc_info_cb(self, msg):
        keys = [
            "corridor_base_cost", "corridor_adp_mean",
            "corridor_adp_max", "corridor_adp_end",
            "corridor_total_cost",
            "corridor_adp_raw_mean", "corridor_adp_raw_max",
            "corridor_adp_raw_end", "corridor_adp_norm",
            "corridor_rank_base", "corridor_rank_total",
            "terminal_adp_cost",
            "mpc_total_cost", "mpc_social_cost", "mpc_tube_cost",
            "mpc_track_cost", "mpc_control_cost",
            "corridor_rank_changed_count",
            "final_approach_used",
            "mpc_reject_forbidden_count",
            "mpc_reject_interest_phi_count",
            "adp_learning_enabled", "adp_decision_influence_enabled",
            "adp_effective_lambda", "adp_ranking_influence_enabled",
            "adp_mpc_influence_enabled",
        ]
        for key, value in zip(keys, list(msg.data)):
            self.latest_adp_mpc_info[key] = float(value)
        try:
            if (float(self.latest_adp_mpc_info.get("corridor_rank_base", 0.0)) !=
                    float(self.latest_adp_mpc_info.get("corridor_rank_total", 0.0))):
                self.corridor_rank_changed_count += 1
            self.latest_adp_mpc_info["corridor_rank_changed_count"] = (
                self.corridor_rank_changed_count)
        except (TypeError, ValueError):
            pass

    def _arm_adp_path_info_cb(self, msg):
        keys = [
            "path_adp_mean", "path_adp_max", "path_adp_delta",
            "adp_path_enabled", "protected_saddle_count",
            "protected_saddle_max_dist", "protected_saddle_ok",
            "mandatory_topology_node_count", "mandatory_saddle_reached",
            "mandatory_saddle_max_dist", "corridor_violation_count",
            "topology_tracking_error", "mpc_segment_count",
        ]
        for key, value in zip(keys, list(msg.data)):
            self.path_adp_info[key] = float(value)

    def _arm_adp_control_info_cb(self, msg):
        keys = [
            "arm_adp_grad_norm", "arm_adp_soft_cost",
            "arm_v_adp_alignment", "arm_adp_control_enabled",
            "arm_dls_adp_used", "arm_qp_used",
            "arm_solver_success_count", "arm_solver_fallback_count",
            "v_des_raw_norm", "v_des_adp_norm", "v_des_delta_norm",
            "dq_nominal_norm", "dq_adp_norm", "dq_delta_norm",
            "mpc_reject_forbidden_count", "mpc_reject_interest_phi_count",
            "adp_learning_enabled", "adp_decision_influence_enabled",
            "adp_effective_lambda", "adp_ranking_influence_enabled",
            "adp_mpc_influence_enabled",
        ]
        for key, value in zip(keys, list(msg.data)):
            self.path_adp_info[key] = float(value)
        success = self.path_adp_info.get("arm_solver_success_count", "")
        fallback = self.path_adp_info.get("arm_solver_fallback_count", "")
        try:
            total = float(success) + float(fallback)
            self.path_adp_info["arm_solver_success_rate"] = (
                float(success) / total if total > 1e-9 else 0.0)
        except (TypeError, ValueError):
            self.path_adp_info["arm_solver_success_rate"] = ""

    def _wc_interest_gate_cb(self, msg):
        data = list(msg.data)
        if len(data) >= 2:
            self.footprint_gate_risk = float(data[1])
        footprint_stop = False
        if len(data) >= 4:
            footprint_stop = bool(data[3] >= 0.5)
            self.footprint_gate_stop = int(footprint_stop)
        if len(data) >= 3:
            self.footprint_gate_scale = float(data[2])
            if self.footprint_gate_scale < 0.999 and not footprint_stop:
                self.footprint_slow_count += 1
        if len(data) >= 5:
            self.footprint_rho_warn = float(data[4])
        if len(data) >= 6:
            self.footprint_rho_stop = float(data[5])
        if len(data) >= 9:
            self.footprint_gate_enabled = int(data[8] >= 0.5)
        self._backfill_stop_footprint_gate_rows()
        if self.footprint_gate_stop:
            self.footprint_stop_count += 1
            if self.first_footprint_stop_time_s == "":
                if self.traj:
                    self.first_footprint_stop_time_s = round(
                        rospy.Time.now().to_sec() - float(self.traj[0]["t"]), 4)
                else:
                    self.first_footprint_stop_time_s = 0.0
                self.footprint_stop_reason = self.gate_reason or "footprint:risk_stop"

    def _wc_interest_cb(self, msg):
        self.wc_interest_received = True
        keys = [
            "phi_center", "phi_front_center", "phi_front_left",
            "phi_front_right", "phi_footrest_left", "phi_footrest_right",
            "phi_rear_left", "phi_rear_right", "phi_max_point",
            "phi_mean_point", "phi_sum_point", "worst_point_idx",
            "forbidden_hit",
        ]
        for key, value in zip(keys, list(msg.data)):
            self.latest_wc_interest[key] = float(value)
        if self.latest_wc_interest["phi_max_point"] != "":
            m = float(self.latest_wc_interest["phi_max_point"])
            self.ip_phi_max_sum += m
            self.ip_phi_max_n += 1
            self.ip_max_phi_point = max(self.ip_max_phi_point, m)
        if float(self.latest_wc_interest.get("forbidden_hit", 0.0) or 0.0) >= 0.5:
            self.forbidden_hit_count += 1

    def _wc_pose2d_cb(self, msg):
        data = list(msg.data)
        if len(data) >= 3:
            self.wc_pose2d_received = True
            self.wc_yaw = float(data[2])

    def _arm_interest_gate_cb(self, msg):
        self.arm_interest_gate_received = True
        data = list(msg.data)
        if len(data) >= 1:
            self.arm_interest_gate_enabled = int(data[0] >= 0.5)
        if len(data) >= 2:
            self.arm_interest_gate_risk = float(data[1])
            if not np.isnan(self.arm_interest_gate_risk):
                self.arm_interest_gate_risk_sum += self.arm_interest_gate_risk
                self.arm_interest_gate_risk_n += 1
                self.max_arm_interest_gate_risk = max(
                    self.max_arm_interest_gate_risk,
                    self.arm_interest_gate_risk)
        arm_stop = False
        if len(data) >= 4:
            arm_stop = bool(data[3] >= 0.5)
            self.arm_interest_gate_stop = int(arm_stop)
        if len(data) >= 3:
            self.arm_interest_gate_scale = float(data[2])
            if self.arm_interest_gate_scale < 0.999 and not arm_stop:
                self.arm_interest_slow_count += 1
        if len(data) >= 5:
            self.arm_interest_rho_warn = float(data[4])
        if len(data) >= 6:
            self.arm_interest_rho_stop = float(data[5])
        if len(data) >= 8:
            self.arm_interest_gate_worst_idx = float(data[7])
        self._backfill_stop_arm_interest_gate_rows()
        if self.arm_interest_gate_stop:
            self.arm_interest_stop_count += 1

    def _arm_interest_cb(self, msg):
        self.arm_interest_received = True
        keys = [
            "phi_ee_point", "phi_wrist", "phi_elbow", "phi_object",
            "phi_arm_max_point", "phi_arm_mean_point", "phi_arm_sum_point",
            "arm_worst_point_idx", "arm_interest_valid_count",
        ]
        for key, value in zip(keys, list(msg.data)):
            self.latest_arm_interest[key] = float(value)
        m = self.latest_arm_interest.get("phi_arm_max_point", "")
        if m != "" and not np.isnan(float(m)):
            m = float(m)
            self.arm_ip_phi_max_sum += m
            self.arm_ip_phi_max_n += 1
            self.arm_ip_max_phi_point = max(self.arm_ip_max_phi_point, m)

    def _arm_interest_points_cb(self, msg):
        self.arm_interest_points_received = True
        keys = [
            "arm_ip_ee_x", "arm_ip_ee_y", "arm_ip_ee_z",
            "arm_ip_wrist_x", "arm_ip_wrist_y", "arm_ip_wrist_z",
            "arm_ip_elbow_x", "arm_ip_elbow_y", "arm_ip_elbow_z",
            "arm_ip_object_x", "arm_ip_object_y", "arm_ip_object_z",
        ]
        for key, value in zip(keys, list(msg.data)):
            self.latest_arm_interest_points[key] = float(value)

    def _should_write_traj_row(self):
        if not self.interest_required:
            return True
        if self.target == "wheelchair":
            return self.wc_interest_received and self.wc_pose2d_received
        if self.target == "arm":
            return (self.arm_interest_received and
                    self.arm_interest_gate_received and
                    self.arm_gate_source_received)
        return True

    def _traj_row(self, t, target, x, y, z):
        row = {
            "t": round(t, 4),
            "run_id": self.run_id,
            "mode": self.mode,
            "variant": self.variant or self.mode,
            "target": target,
            "phase": self.phase if target == "arm" else "",
            "x": x,
            "y": y,
            "z": z,
            "gate_state": self.gate_state,
            "gate_scale": self.gate_scale,
            "gate_stop": int(self.gate_stop),
            "gate_reason": self.gate_reason,
            "rho_warn": self.rho_warn,
            "rho_stop": self.rho_stop,
            "adp_value": self.adp_value,
            "adp_delta": self.adp_delta,
            "adp_enabled": self.adp_enabled,
            "critic_version": self.critic_version,
            "selected_corridor_label": self.selected_corridor_label,
            "selected_corridor_type": self._selected_corridor_type(),
            "corridor_id": self.selected_corridor_label,
            "execution_corridor_id": self.selected_corridor_label,
        }
        row.update(self.latest_adp_mpc_info)
        row.update(self.path_adp_info)
        row.update(self.latest_risk_components)
        row.update(self.latest_velocity_monitor)
        if self.trajectory_debug_level >= 2:
            row.update(self.latest_topology_info)
            row.update(self._topology_aliases())
        if target == "arm":
            row.update(self.latest_arm_interest)
            row.update(self.latest_arm_interest_points)
            row["arm_gate_source"] = self.arm_gate_source
            row["arm_interest_gate_enabled"] = self.arm_interest_gate_enabled
            row["arm_interest_gate_risk"] = self.arm_interest_gate_risk
            row["arm_interest_gate_scale"] = self.arm_interest_gate_scale
            row["arm_interest_gate_stop"] = self.arm_interest_gate_stop
            row["arm_interest_rho_warn"] = self.arm_interest_rho_warn
            row["arm_interest_rho_stop"] = self.arm_interest_rho_stop
            row["arm_interest_gate_worst_idx"] = self.arm_interest_gate_worst_idx
        if target == "wheelchair":
            row["yaw"] = self.wc_yaw
            row.update(self.latest_wc_interest)
            row["gate_source"] = self.gate_source
            row["footprint_gate_enabled"] = self.footprint_gate_enabled
            row["footprint_gate_risk"] = self.footprint_gate_risk
            row["footprint_gate_scale"] = self.footprint_gate_scale
            row["footprint_gate_stop"] = self.footprint_gate_stop
            row["footprint_rho_warn"] = self.footprint_rho_warn
            row["footprint_rho_stop"] = self.footprint_rho_stop
        return row

    def _phi_cb(self, msg):
        now = rospy.Time.now().to_sec()
        phi = float(msg.data)
        if self.last_risk_time is not None:
            dt = now - self.last_risk_time
            if 0.0 < dt < 0.5:
                self.valid_risk_duration_s += dt
                self.j_social += phi * dt
                if phi > self.risk_threshold:
                    self.risk_exceed_s += dt
        self.last_risk_time = now

        self.phi_sum += phi
        self.phi_n += 1
        self.phi_values.append(phi)
        self.max_phi = max(self.max_phi, phi)

    def _speed(self, pos, t):
        if self.prev_pos is not None and self.prev_t is not None:
            dt = (t - self.prev_t)
            if dt > 1e-3:
                sp = np.linalg.norm(pos - self.prev_pos) / dt
                self.prev_pos, self.prev_t = pos, t
                return sp
        self.prev_pos, self.prev_t = pos, t
        return 0.0

    def _windowed_speeds(self, horizon=0.6):
        if len(self.traj) < 2:
            return np.zeros(len(self.traj), float)

        t = np.array([row["t"] for row in self.traj], float)
        t -= t[0]
        xyz = np.array([
            [row["x"], row["y"], row["z"]]
            for row in self.traj
        ], float)
        speeds = np.zeros(len(self.traj), float)
        half = float(horizon) / 2.0
        for i, ti in enumerate(t):
            left = int(np.searchsorted(t, ti - half, side="left"))
            right = int(np.searchsorted(t, ti + half, side="right")) - 1
            if right <= left:
                left = max(0, i - 1)
                right = min(len(t) - 1, i + 1)
            dt = t[right] - t[left]
            if dt > 1e-6:
                speeds[i] = np.linalg.norm(xyz[right] - xyz[left]) / dt
        return speeds

    def _speed_summary(self):
        speeds = self._windowed_speeds()
        max_speed = float(np.max(speeds)) if len(speeds) else 0.0
        if self.target != "arm" or not len(speeds):
            return max_speed, 0.0

        xyz = np.array([
            [row["x"], row["y"], row["z"]]
            for row in self.traj
        ], float)
        near = np.linalg.norm(xyz - self.hand[None, :], axis=1) < 0.12
        mean_near = float(np.mean(speeds[near])) if np.any(near) else 0.0
        return max_speed, mean_near

    def _duration(self):
        if len(self.traj) < 2:
            return 0.0
        return float(self.traj[-1]["t"] - self.traj[0]["t"])

    def _arm_pose_cb(self, msg):
        if not self._should_write_traj_row():
            return
        p = np.array([msg.point.x, msg.point.y, msg.point.z])
        t = msg.header.stamp.to_sec()
        if self.arm_home_ref is None:
            self.arm_home_ref = np.array(p, float)
        self.traj.append(self._traj_row(t, "arm", p[0], p[1], p[2]))
        if self.last_pos is not None:
            self.path_length_m += float(np.linalg.norm(p - self.last_pos))
        self.last_pos = p
        self.min_head = min(self.min_head, np.linalg.norm(p - self.head))
        self.min_chest = min(self.min_chest, np.linalg.norm(p - self.chest))
        d_hand = np.linalg.norm(p - self.hand)
        d_wait = np.linalg.norm(p - self.wait_pose)
        d_home = (
            np.linalg.norm(p - self.arm_home_ref)
            if self.arm_home_ref is not None else np.inf)
        self.arm_min_hand = min(self.arm_min_hand, d_hand)
        self.arm_min_wait = min(self.arm_min_wait, d_wait)
        if self.arm_home_return_started:
            self.arm_min_home_after_return = min(
                self.arm_min_home_after_return, d_home)
        if d_hand < self.hand_tolerance:
            self.arm_reached_hand = True
            self.reached = True
        if self.phase == 3:
            if self.arm_hold_first_t is None:
                self.arm_hold_first_t = t
            self.arm_hold_last_t = t
            if self.arm_reached_hand:
                self.arm_hold_duration_s = max(
                    self.arm_hold_duration_s,
                    self.arm_hold_last_t - self.arm_hold_first_t)
        if self.phase >= 3 and d_wait < self.wait_tolerance:
            self.arm_wait_reached = True
        if self.phase >= 4 and d_home < self.home_tolerance:
            self.arm_home_returned = True

    def _wc_pose_cb(self, msg):
        if not self._should_write_traj_row():
            return
        p = np.array([msg.point.x, msg.point.y])
        t = msg.header.stamp.to_sec()
        self.traj.append(self._traj_row(t, "wheelchair", p[0], p[1], 0.0))
        if self.last_pos is not None:
            self.path_length_m += float(np.linalg.norm(p - self.last_pos))
        self.last_pos = p
        self.min_person = min(self.min_person, np.linalg.norm(p - self.person))
        if np.all(np.abs(p - self.transfer_c) <= self.transfer_h):
            self.transfer_time += self.dt
        if np.linalg.norm(p - self.goal) < self.success_goal_tolerance:
            self.reached = True

    def _wc_task_complete_cb(self, msg):
        self.wc_task_completed = bool(msg.data)

    def _finish(self):
        if self.stop_triggered and not self.stop_row_written:
            self._append_stop_traj_row()
        max_speed, mean_speed_near = self._speed_summary()
        duration = self._duration()
        arm_hold_completed = (
            self.arm_hold_duration_s >= self.hold_min_s
            if self.target == "arm" else False)
        arm_handover_home_success = (
            self.arm_handover_complete and self.arm_home_returned)
        arm_task_complete = (
            self.arm_goal_reached and
            self.arm_handover_pose_reached and
            self.arm_handover_complete and
            arm_hold_completed and
            self.arm_retreat_started and self.arm_wait_reached and
            self.arm_home_return_started and self.arm_home_returned)
        final_dist = ""
        reached_0p35 = reached_0p25 = reached_0p08 = ""
        goal_reached_observed = bool(self.reached)
        final_distance_ok = False
        if self.target != "arm" and self.last_pos is not None:
            final_dist = float(np.linalg.norm(self.last_pos - self.goal))
            reached_0p35 = int(final_dist < 0.35)
            reached_0p25 = int(final_dist < 0.25)
            reached_0p08 = int(final_dist < 0.08)
            final_distance_ok = bool(final_dist < self.success_goal_tolerance)
            self.reached = bool(
                self.wc_task_completed or
                (goal_reached_observed and final_distance_ok))
        success_goal = int(arm_task_complete if self.target == "arm" else self.reached)
        success_safe = int(self.reached and not self.stop_triggered)
        if self.target == "arm":
            success_safe = int(
                arm_task_complete and arm_handover_home_success and
                not self.stop_triggered)
        mpc_diag = self._load_mpc_diagnostics()
        variant_name = str(self.variant or self.mode or "").strip().lower()
        mpc_status = str(
            mpc_diag.get("final_status") or
            mpc_diag.get("final_mpc_status") or
            mpc_diag.get("mpc_feasibility_status", "") or "")
        if not mpc_status and variant_name == "baseline":
            mpc_status = "feasible"
        if self.target == "arm" and not mpc_diag:
            mpc_success_allowed = True
        else:
            diag_safety_success = str(
                mpc_diag.get("safety_success", "")).strip().lower()
            safety_contract_ok = diag_safety_success in (
                "1", "1.0", "true", "yes")
            if variant_name == "baseline":
                mpc_success_allowed = mpc_status in ("", "feasible")
            elif mpc_status == "feasible_with_soft_violation":
                mpc_success_allowed = bool(safety_contract_ok)
            else:
                mpc_success_allowed = mpc_status == "feasible"
        success_safe = int(bool(success_safe) and bool(mpc_success_allowed))
        if self.target == "arm":
            success_safe = int(
                bool(success_safe) and bool(arm_handover_home_success))
        mean_gate_scale = (
            self.gate_scale_sum / self.gate_scale_n
            if self.gate_scale_n else 1.0)
        mean_phi_max_point = (
            self.ip_phi_max_sum / self.ip_phi_max_n
            if self.ip_phi_max_n else 0.0)
        mean_phi_arm_max_point = (
            self.arm_ip_phi_max_sum / self.arm_ip_phi_max_n
            if self.arm_ip_phi_max_n else 0.0)
        mean_arm_interest_gate_risk = (
            self.arm_interest_gate_risk_sum / self.arm_interest_gate_risk_n
            if self.arm_interest_gate_risk_n else 0.0)
        mean_adp_value = (
            float(np.mean(self.adp_values)) if self.adp_values else 0.0)
        max_adp_value = (
            float(np.max(self.adp_values)) if self.adp_values else 0.0)
        adp_value_before_stop = ""
        if self.stop_triggered and self.traj:
            stop_t = float(self.traj[-1]["t"])
            vals = []
            for item in self.traj:
                try:
                    t = float(item.get("t", 0.0))
                    v = item.get("adp_value", "")
                    if v != "" and stop_t - 1.0 <= t <= stop_t:
                        vals.append(float(v))
                except (TypeError, ValueError):
                    pass
            if vals:
                adp_value_before_stop = round(float(np.mean(vals)), 4)
        clip_ratio = (
            float(self.adp_clip_hits) / float(len(self.adp_values))
            if self.adp_values else 0.0)
        risk_exceed_pct = (
            100.0 * self.risk_exceed_s / self.valid_risk_duration_s
            if self.valid_risk_duration_s > 1e-6 else 0.0)
        risk_mean_time = (
            self.j_social / self.valid_risk_duration_s
            if self.valid_risk_duration_s > 1e-6 else 0.0)
        risk_per_meter = (
            self.j_social / self.path_length_m
            if self.path_length_m > 1e-6 else 0.0)
        risk_p95 = (
            float(np.percentile(np.asarray(self.phi_values, float), 95))
            if self.phi_values else 0.0)
        base = {
            "run_id": self.run_id,
            "mode": self.mode,
            "variant": self.variant or self.mode,
            "target": self.target,
            "scenario": self.scenario,
            "controller_version": self.controller_version,
            "duration_s": round(duration, 3),
            "risk_exceed_pct": round(risk_exceed_pct, 2),
            "risk_exceed_s": round(self.risk_exceed_s, 3),
            "valid_risk_duration_s": round(self.valid_risk_duration_s, 3),
            "mean_phi_s": round(self.phi_sum / self.phi_n, 4) if self.phi_n else 0.0,
            "max_phi_s": round(self.max_phi, 4),
            "J_social": round(self.j_social, 3),
            "risk_mean_time": round(risk_mean_time, 4),
            "risk_per_meter": round(risk_per_meter, 4),
            "risk_p95": round(risk_p95, 4),
            "path_length_m": round(self.path_length_m, 4),
            "high_risk_ratio": round(risk_exceed_pct / 100.0, 4),
            "high_risk_duration": round(self.risk_exceed_s, 3),
            "success_goal": success_goal,
            "wc_task_completed": (
                int(self.wc_task_completed) if self.target != "arm" else ""),
            "success_safe": success_safe,
            "success": success_safe,
            "goal_reached": (
                int(self.arm_goal_reached) if self.target == "arm"
                else int(self.reached)),
            "handover_pose_reached": (
                int(self.arm_handover_pose_reached)
                if self.target == "arm" else ""),
            "handover_complete": (
                int(self.arm_handover_complete)
                if self.target == "arm" else ""),
            "arm_reached_hand": int(self.arm_reached_hand) if self.target == "arm" else "",
            "arm_goal_reached": int(self.arm_goal_reached) if self.target == "arm" else "",
            "arm_handover_pose_reached": int(self.arm_handover_pose_reached) if self.target == "arm" else "",
            "arm_handover_complete": int(self.arm_handover_complete) if self.target == "arm" else "",
            "arm_handover_pos_err": (
                round(float(self.arm_handover_pos_err), 4)
                if self.target == "arm" and self.arm_handover_pos_err != "" else ""),
            "arm_handover_orientation_err": (
                round(float(self.arm_handover_orientation_err), 4)
                if self.target == "arm" and self.arm_handover_orientation_err != "" else ""),
            "arm_handover_stable_time_s": (
                round(float(self.arm_handover_stable_time_s), 3)
                if self.target == "arm" else ""),
            "arm_gripper_handover_event": (
                int(self.arm_gripper_handover_event)
                if self.target == "arm" else ""),
            "arm_hold_completed": int(arm_hold_completed) if self.target == "arm" else "",
            "arm_hold_duration_s": round(self.arm_hold_duration_s, 3) if self.target == "arm" else "",
            "arm_retreat_started": int(self.arm_retreat_started) if self.target == "arm" else "",
            "arm_wait_reached": int(self.arm_wait_reached) if self.target == "arm" else "",
            "arm_home_return_started": int(self.arm_home_return_started) if self.target == "arm" else "",
            "arm_home_returned": int(self.arm_home_returned) if self.target == "arm" else "",
            "arm_handover_home_success": int(arm_handover_home_success) if self.target == "arm" else "",
            "arm_task_complete": int(arm_task_complete) if self.target == "arm" else "",
            "arm_min_hand_dist": round(self.arm_min_hand, 4) if self.target == "arm" and np.isfinite(self.arm_min_hand) else "",
            "arm_min_wait_dist": round(self.arm_min_wait, 4) if self.target == "arm" and np.isfinite(self.arm_min_wait) else "",
            "arm_min_home_return_dist": (
                round(self.arm_min_home_after_return, 4)
                if self.target == "arm" and np.isfinite(self.arm_min_home_after_return)
                else ""),
            "stop_triggered": int(self.stop_triggered),
            "stop_reason": self.stop_reason,
            "first_stop_time_s": self.first_stop_time_s,
            "gate_slow_count": self.gate_slow_count,
            "gate_stop_count": self.gate_stop_count,
            "max_gate_risk": round(self.max_gate_risk, 4),
            "mean_gate_scale": round(mean_gate_scale, 4),
            "phase_stop": self.phase_stop if self.target == "arm" else "",
            "mean_phi_max_point": round(mean_phi_max_point, 4) if self.target != "arm" else "",
            "max_phi_max_point": round(self.ip_max_phi_point, 4) if self.target != "arm" else "",
            "forbidden_hit_count": self.forbidden_hit_count if self.target != "arm" else "",
            "interest_enabled": int(self.ip_phi_max_n > 0) if self.target != "arm" else "",
            "footprint_gate_enabled": self.footprint_gate_enabled if self.target != "arm" else "",
            "footprint_slow_count": self.footprint_slow_count if self.target != "arm" else "",
            "footprint_stop_count": self.footprint_stop_count if self.target != "arm" else "",
            "first_footprint_stop_time_s": self.first_footprint_stop_time_s if self.target != "arm" else "",
            "footprint_stop_reason": self.footprint_stop_reason if self.target != "arm" else "",
            "mean_phi_arm_max_point": round(mean_phi_arm_max_point, 4) if self.target == "arm" else "",
            "max_phi_arm_max_point": round(self.arm_ip_max_phi_point, 4) if self.target == "arm" else "",
            "arm_interest_enabled": int(self.arm_ip_phi_max_n > 0) if self.target == "arm" else "",
            "arm_interest_gate_enabled": self.arm_interest_gate_enabled if self.target == "arm" else "",
            "arm_interest_slow_count": self.arm_interest_slow_count if self.target == "arm" else "",
            "arm_interest_stop_count": self.arm_interest_stop_count if self.target == "arm" else "",
            "mean_arm_interest_gate_risk": round(mean_arm_interest_gate_risk, 4) if self.target == "arm" else "",
            "max_arm_interest_gate_risk": round(self.max_arm_interest_gate_risk, 4) if self.target == "arm" else "",
            "mean_adp_value": round(mean_adp_value, 4),
            "max_adp_value": round(max_adp_value, 4),
            "adp_value_before_stop": adp_value_before_stop,
            "adp_enabled": self.adp_enabled,
            "critic_version": self.critic_version,
            "clip_ratio": round(clip_ratio, 4),
            "selected_corridor_label": self.selected_corridor_label,
            "selected_corridor_type": self._selected_corridor_type(),
            "corridor_id": self.selected_corridor_label,
            "execution_corridor_id": self.selected_corridor_label,
            "reached_goal_0p35": reached_0p35,
            "reached_completion_0p25": reached_0p25,
            "reached_goal_0p08": reached_0p08,
            "success_goal_tolerance": (
                self.success_goal_tolerance if self.target != "arm" else ""),
        }
        base.update(self.latest_topology_info)
        base.update(self._topology_aliases())
        if mpc_diag:
            diag_fields = [
                "mpc_feasibility_status", "failure_reason",
                "mpc_failure_reason", "failed_constraint_type",
                "final_status", "final_mpc_status", "temporary_mpc_status",
                "temporary_failure_reason", "final_failure_reason",
                "mpc_stage_status", "success",
                "replan_required", "mpc_used", "tube_constraint_used",
                "topology_constraint_used",
                "corridor_constraint_used", "manifold_constraint_used",
                "critical_point_constraint_used",
                "critical_point_sequence_constraint_used",
                "critical_point_sequence_valid",
                "critical_point_association_used",
                "topology_sequence_valid",
                "critical_point_status",
                "current_topology_stage",
                "passed_critical_points",
                "critical_point_soft_violation_count",
                "critical_point_hard_violation_count",
                "critical_point_soft_radius", "critical_point_hard_radius",
                "critical_point_constraint_mode",
                "topology_sequence_constraint_used",
                "topology_infeasible_count",
                "corridor_override_count", "manifold_override_count",
                "pre_refinement_clearance",
                "post_refinement_clearance",
                "planning_clearance",
                "predicted_clearance",
                "execution_clearance",
                "soft_constraint_used",
                "minor_violation",
                "major_violation",
                "minor_violation_count",
                "major_violation_count",
                "mpc_warning",
                "refinement_success",
                "refinement_fallback",
                "refinement_tube_valid",
                "reference_source",
                "reference_path_count",
                "selected_corridor_id", "selected_corridor_label",
                "module_chain_valid", "risk_field_used", "manifold_used",
                "morse_used", "topology_graph_used",
                "candidate_corridor_used", "candidate_ranking_used",
            ]
            for key in diag_fields:
                if key in mpc_diag:
                    base[key] = mpc_diag.get(key)
        else:
            base.setdefault("mpc_feasibility_status", mpc_status)
        if "final_status" in base or "final_mpc_status" in base:
            final_status = str(
                base.get("final_status") or base.get("final_mpc_status") or "")
            base["final_status"] = final_status
            base["final_mpc_status"] = final_status
            base["mpc_feasibility_status"] = final_status
        if "final_failure_reason" in base:
            base["failure_reason"] = str(base.get("final_failure_reason") or "")
            base["mpc_failure_reason"] = str(base.get("final_failure_reason") or "")
        if variant_name == "baseline":
            for key in (
                    "morse_used", "topology_constraint_used",
                    "corridor_constraint_used", "manifold_constraint_used",
                    "critical_point_sequence_constraint_used",
                    "critical_point_constraint_used",
                    "critical_point_association_used",
                    "topology_sequence_valid"):
                base[key] = 0
            base["critical_point_status"] = "passed"
            base.setdefault("baseline_type", "direct")
            base.setdefault("baseline_planner", "direct_connection")
            base.setdefault("planner_source", "direct_connection")
        reference_source = str(base.get("reference_source", "") or "")
        if not reference_source:
            reference_source = (
                "refined_waypoints"
                if float(base.get("mpc_reference_is_refined") or 0.0) > 0.5
                else "raw_waypoints")
        reference_path_count = float(
            base.get("reference_path_count") or
            base.get("mpc_reference_count") or
            base.get("selected_refined_waypoints_count") or
            base.get("selected_raw_waypoints_count") or 0.0)
        refinement_success_flag = float(
            base.get("refinement_success") or
            base.get("selected_refinement_success") or 0.0) > 0.5
        refinement_fallback_flag = float(
            base.get("refinement_fallback") or
            base.get("selected_refinement_fallback") or 0.0) > 0.5
        refinement_tube_valid_flag = float(
            base.get("refinement_tube_valid") or
            base.get("selected_refinement_tube_valid") or 0.0) > 0.5
        reference_is_refined = reference_source in (
            "refined", "refined_waypoints", "refinement",
            "ik_validated_trajectory")
        reference_is_candidate_fallback = (
            reference_source in ("candidate_fallback", "candidate", "raw_waypoints") and
            refinement_fallback_flag)
        refinement_trace_valid = int(
            reference_path_count > 0.0 and (
                (refinement_success_flag and refinement_tube_valid_flag and reference_is_refined) or
                reference_is_candidate_fallback))
        selected_label = str(base.get("selected_corridor_label", "") or "")
        selected_id = str(
            base.get("selected_corridor_id", base.get("corridor_id", "")) or "")
        formal_selected = bool(
            (selected_id or selected_label) and
            selected_id != "planning_failed" and selected_label != "planning_failed" and
            str(base.get("selected_corridor_type", "")) != "fallback")
        if (self.target == "arm" and reference_is_refined and
                reference_path_count > 0.0 and formal_selected):
            refinement_trace_valid = 1
        explicit_chain = mpc_diag.get("module_chain_valid") if mpc_diag else None
        constraint_chain = bool(
            float(base.get("topology_constraint_used") or 0.0) > 0.5 and
            float(base.get("corridor_constraint_used") or 0.0) > 0.5 and
            float(base.get("manifold_constraint_used") or 0.0) > 0.5)
        inferred_chain = bool(
            str(base.get("variant", base.get("mode", ""))).lower() == "stsm" and
            formal_selected and
            float(base.get("topology_used") or 0.0) > 0.5 and
            float(base.get("topology_fallback_used") or 0.0) < 0.5 and
            float(base.get("num_topology_nodes") or 0.0) > 0.0 and
            float(base.get("num_candidate_corridors") or 0.0) > 0.0 and
            refinement_trace_valid and constraint_chain)
        module_chain_valid = int(
            bool(explicit_chain) if explicit_chain is not None else inferred_chain)
        selected_rank = 1 if (
            str(base.get("variant", base.get("mode", ""))).lower() == "stsm"
            and formal_selected) else 0
        mpc_status = str(base.get("mpc_feasibility_status", mpc_status) or "")
        major_safety_violation = bool(
            float(base.get("major_violation_count") or 0.0) > 0.5 or
            str(base.get("major_violation", "")).strip().lower() in
            ("1", "true", "yes"))
        mpc_used_flag = float(base.get("mpc_used") or 0.0) > 0.5
        reference_count_for_status = float(
            base.get("reference_path_count") or
            base.get("mpc_reference_count") or 0.0)
        if (mpc_used_flag and reference_count_for_status > 0.0 and
                mpc_status == "infeasible_reference_empty"):
            mpc_status = "feasible"
            base["mpc_feasibility_status"] = "feasible"
            base["final_status"] = "feasible"
            base["final_mpc_status"] = "feasible"
            base["failure_reason"] = ""
            base["mpc_failure_reason"] = ""
            base["final_failure_reason"] = ""
        if self.target == "arm" and not mpc_diag:
            mpc_success_allowed = True
        else:
            diag_safety_success = str(
                base.get("safety_success", "")).strip().lower()
            safety_contract_ok = diag_safety_success in (
                "1", "1.0", "true", "yes")
            if mpc_status == "feasible_with_soft_violation":
                mpc_success_allowed = bool(safety_contract_ok)
            else:
                mpc_success_allowed = mpc_status == "feasible"
        if major_safety_violation:
            mpc_success_allowed = False
        if variant_name == "baseline" and not mpc_status:
            mpc_success_allowed = True
        execution_status = "success" if (
            (module_chain_valid or variant_name == "baseline") and
            int(success_goal) == 1 and bool(mpc_success_allowed) and
            not self.stop_triggered) else "failed"
        failure_stage = ""
        if execution_status != "success":
            if variant_name == "baseline":
                failure_stage = "execution"
            elif not formal_selected:
                failure_stage = "planning"
            elif not refinement_trace_valid:
                failure_stage = "refinement"
            else:
                failure_stage = "execution"
        else:
            failure_stage = "none"
            if not self.stop_reason:
                self.stop_reason = "none"
        adp_info = (self.path_adp_info if self.target == "arm"
                    else self.latest_adp_mpc_info)
        adp_enabled = float(base.get("adp_enabled") or 0.0) > 0.5
        learning_enabled = float(adp_info.get(
            "adp_learning_enabled", adp_enabled) or 0.0) > 0.5
        influence_enabled = float(adp_info.get(
            "adp_decision_influence_enabled", 0.0) or 0.0) > 0.5
        effective_lambda = float(adp_info.get("adp_effective_lambda", 0.0) or 0.0)
        ranking_enabled = float(adp_info.get(
            "adp_ranking_influence_enabled", 0.0) or 0.0) > 0.5
        mpc_enabled = float(adp_info.get(
            "adp_mpc_influence_enabled", 0.0) or 0.0) > 0.5
        affects_ranking = bool(
            influence_enabled and ranking_enabled and effective_lambda != 0.0)
        affects_control = bool(
            influence_enabled and mpc_enabled and effective_lambda != 0.0 and (
                float(self.latest_adp_mpc_info.get("terminal_adp_cost") or 0.0) != 0.0 or
                float(self.path_adp_info.get("arm_adp_grad_norm") or 0.0) > 1e-9 or
                float(self.path_adp_info.get("arm_adp_soft_cost") or 0.0) != 0.0))
        adp_role = adp_role_from_runtime(
            adp_enabled, learning_enabled, influence_enabled,
            effective_lambda=effective_lambda,
            ranking_contribution=affects_ranking,
            control_contribution=affects_control)
        adp_affects_candidate_ranking = int(
            adp_role in ("ranking_modifier", "ranking_and_control_modifier"))
        adp_affects_control = int(
            adp_role in ("control_modifier", "ranking_and_control_modifier"))
        base.update({
            "risk_field_used": int(float(base.get(
                "risk_field_used", module_chain_valid) or 0.0) > 0.5),
            "manifold_used": int(float(base.get(
                "manifold_used", module_chain_valid) or 0.0) > 0.5),
            "morse_used": int(float(base.get(
                "morse_used", module_chain_valid) or 0.0) > 0.5),
            "topology_graph_used": int(float(base.get(
                "topology_graph_used", module_chain_valid) or 0.0) > 0.5),
            "candidate_corridor_used": int(float(base.get(
                "candidate_corridor_used", module_chain_valid) or 0.0) > 0.5),
            "candidate_ranking_used": int(float(base.get(
                "candidate_ranking_used", module_chain_valid) or 0.0) > 0.5),
            "fallback_used": int(float(base.get("topology_fallback_used") or 0.0)),
            "selected_corridor_id": selected_id,
            "selected_rank": selected_rank,
            "selected_total_score": base.get("corridor_total_cost", ""),
            "total_score": base.get("corridor_total_cost", ""),
            "selection_override_reason": "",
            "raw_waypoints_count": base.get("selected_raw_waypoints_count", ""),
            "refined_waypoints_count": base.get("selected_refined_waypoints_count", ""),
            "mpc_reference_source": reference_source,
            "reference_source": reference_source,
            "reference_path_count": reference_path_count,
            "mpc_reference_is_refined": int(reference_is_refined),
            "adp_used": int(float(base.get("adp_enabled") or 0.0) > 0.5),
            "adp_role": adp_role,
            "adp_affects_candidate_ranking": adp_affects_candidate_ranking,
            "adp_affects_control": adp_affects_control,
            "final_path_source": (
                "Morse->Candidate->Ranking->Refinement->MPC"
                if refinement_trace_valid else ""),
            "module_chain_valid": module_chain_valid,
            "selection_consistent": int(selected_rank == 1),
            "refinement_trace_valid": refinement_trace_valid,
            "execution_status": execution_status,
            "failure_stage": failure_stage,
            "success_goal": success_goal,
            "success_safe": success_safe,
            "success": success_safe,
            "topology_route_class": base.get("selected_topology_route_class", ""),
            "task_semantic_class": base.get("selected_task_semantic_class", ""),
        })
        if self.target != "arm":
            base.update({
                key: self.latest_adp_mpc_info.get(key, "")
                for key in self.latest_adp_mpc_info.keys()
            })
        else:
            base.update({
                key: self.path_adp_info.get(key, "")
                for key in self.path_adp_info.keys()
            })
        selected_total_score = base.get("selected_total_score", "")
        if selected_total_score in ("", None):
            selected_total_score = base.get("selected_candidate_total_score", "")
        if selected_total_score in ("", None):
            selected_total_score = base.get("corridor_total_cost", "")
        if selected_total_score in ("", None):
            selected_total_score = base.get("selected_tracking_cost", "")
        base["selected_total_score"] = selected_total_score
        base["total_score"] = selected_total_score
        base["adp_affects_candidate_ranking"] = int(
            str(base.get("adp_role", "")) in (
                "ranking_modifier", "ranking_and_control_modifier"))
        base["adp_affects_control"] = int(
            str(base.get("adp_role", "")) in (
                "control_modifier", "ranking_and_control_modifier"))
        if self.target == "arm":
            row = {
                "min_head_dist": round(self.min_head, 4),
                "min_chest_dist": round(self.min_chest, 4),
                "max_ee_speed": round(max_speed, 4),
                "mean_speed_near_hand": round(mean_speed_near, 4),
                "min_person_dist": "",
                "transfer_intrusion_s": "",
                "max_speed": "",
                "final_dist_to_goal": "",
            }
        else:
            row = {
                "min_head_dist": "",
                "min_chest_dist": "",
                "max_ee_speed": "",
                "mean_speed_near_hand": "",
                "min_person_dist": round(self.min_person, 4),
                "transfer_intrusion_s": round(self.transfer_time, 3),
                "max_speed": round(max_speed, 4),
                "final_dist_to_goal": round(final_dist, 4) if final_dist != "" else "",
            }
        base.update(row)
        row = base
        self._write(row)
        self._write_consistency_check(row, mpc_diag)
        self._write_traj()
        rospy.loginfo("[metrics] === %s / %s ===", row["target"], row["mode"])
        for k, v in row.items():
            rospy.loginfo("[metrics]   %-22s %s", k, v)

    def _write(self, row):
        rows = []
        if os.path.exists(self.out):
            with open(self.out, "r") as f:
                for old in csv.DictReader(f):
                    old_mode = old.get("mode", "").strip().lower()
                    old_target = old.get("target", "").strip().lower()
                    old_variant = old.get("variant", "").strip().lower()
                    if not old_variant:
                        old_variant = old_mode
                    new_variant = row.get("variant", row["mode"]).strip().lower()
                    if old_mode not in ("baseline", "stsm"):
                        continue
                    if old_target != row["target"]:
                        continue
                    if old_variant == new_variant:
                        continue
                    rows.append(old)
        rows.append(row)
        order = {
            "baseline": 0,
            "stsm": 1,
        }
        rows.sort(key=lambda r: order.get(
            r.get("variant", r.get("mode", "")).strip().lower(), 99))

        open_kwargs = {}
        mode = "w"
        if sys.version_info[0] >= 3:
            open_kwargs["newline"] = ""
        else:
            mode = "wb"
        with open(self.out, mode, **open_kwargs) as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            for item in rows:
                cleaned = {}
                for key in row.keys():
                    cleaned[key] = item.get(key, "")
                w.writerow(cleaned)
        rospy.loginfo("[metrics] wrote %s", self.out)

    def _write_consistency_check(self, row, mpc_diag):
        if not mpc_diag:
            return
        out_dir = os.path.dirname(self.out)
        if not out_dir:
            return
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        def _float_value(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        def _bool_value(value):
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            return text in ("1", "1.0", "true", "yes", "success")

        metrics_ref = int(round(_float_value(row.get("reference_path_count"))))
        diag_ref = int(round(_float_value(mpc_diag.get("reference_path_count"))))
        metrics_success = _bool_value(row.get("success"))
        diag_success = _bool_value(mpc_diag.get("success"))
        checks = {
            "reference_path_count_match": metrics_ref == diag_ref,
            "success_match": metrics_success == diag_success,
        }
        warnings = []
        if not checks["reference_path_count_match"]:
            warnings.append("reference_path_count_mismatch")
        if not checks["success_match"]:
            warnings.append("success_mismatch")
        payload = {
            "metrics_reference_path_count": metrics_ref,
            "mpc_diagnostics_reference_path_count": diag_ref,
            "metrics_success": metrics_success,
            "mpc_diagnostics_success": diag_success,
            "consistent": not warnings,
            "warning": warnings,
        }
        payload.update(checks)
        path = os.path.join(out_dir, "consistency_check.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        if warnings:
            rospy.logwarn("[metrics] consistency warnings: %s", ",".join(warnings))

    def _write_traj(self):
        if not self.traj_out:
            return
        out_dir = os.path.dirname(self.traj_out)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        open_kwargs = {}
        mode = "w"
        if sys.version_info[0] >= 3:
            open_kwargs["newline"] = ""
        else:
            mode = "wb"
        with open(self.traj_out, mode, **open_kwargs) as f:
            compact_fields = [
                "t", "mode", "variant", "target", "phase", "x", "y", "z",
                "phi_prox", "phi_close", "phi_dir", "phi_body", "phi_env",
                "phi_total", "vx", "vy", "vz", "speed_raw",
                "speed_filtered", "dt_used", "velocity_valid",
                "gate_state", "gate_scale", "gate_stop", "gate_reason",
                "adp_value", "adp_delta", "adp_enabled",
                "selected_corridor_label", "selected_corridor_type",
                "corridor_id", "execution_corridor_id",
                "path_adp_mean", "path_adp_max", "path_adp_delta",
                "adp_path_enabled", "protected_saddle_count",
                "protected_saddle_max_dist", "protected_saddle_ok",
                "mandatory_topology_node_count", "mandatory_saddle_reached",
                "mandatory_saddle_max_dist", "corridor_violation_count",
                "topology_tracking_error", "mpc_segment_count",
                "arm_adp_grad_norm", "arm_adp_soft_cost",
                "arm_v_adp_alignment",
                "phi_ee_point", "phi_wrist", "phi_elbow", "phi_object",
                "phi_arm_max_point", "phi_arm_mean_point",
                "arm_interest_gate_risk", "arm_interest_gate_scale",
                "yaw", "phi_center", "phi_max_point", "phi_mean_point",
                "forbidden_hit", "footprint_gate_risk",
                "footprint_gate_scale", "footprint_gate_stop",
            ]
            full_fields = [
                "t", "run_id", "mode", "variant", "target", "phase", "x", "y", "z",
                "phi_prox", "phi_close", "phi_dir", "phi_body", "phi_env",
                "phi_total", "vx", "vy", "vz", "speed_raw",
                "speed_filtered", "phi_close_monitor", "dt_used",
                "velocity_valid", "gate_state", "gate_scale", "gate_stop",
                "gate_reason", "rho_warn", "rho_stop",
                "adp_value", "adp_delta", "adp_enabled", "critic_version",
                "selected_corridor_label", "selected_corridor_type",
                "corridor_id", "execution_corridor_id",
                "topology_enabled", "topology_used",
                "topology_fallback_used", "num_critical_minima",
                "num_critical_saddles", "num_critical_maxima",
                "num_raw_minima", "num_raw_saddles", "num_raw_maxima",
                "num_safe_minima", "num_safe_saddles", "num_safe_maxima",
                "num_filtered_minima", "num_filtered_saddles",
                "num_filtered_maxima", "num_usable_minima",
                "num_usable_saddles", "num_used_minima", "num_used_saddles",
                "num_forced_critical_corridors", "num_morse_minima_corridors",
                "num_morse_saddle_corridors", "num_morse_mix_corridors",
                "num_graph_direct_corridors",
                "num_graph_semantic_corridors", "reject_by_gradient_count",
                "reject_by_degenerate_count", "reject_by_forbidden_count",
                "reject_by_clearance_count", "reject_by_unsafe_count",
                "num_topology_nodes", "num_topology_edges",
                "num_candidate_corridors", "topology_grid_resolution",
                "topology_rho", "num_forbidden_cells",
                "selected_corridor_forbidden_hits",
                "candidate_forbidden_reject_count",
                "clearance_reject_count", "edge_clearance_reject_count",
                "edge_forbidden_reject_count", "edge_astar_fail_count",
                "neighbor_pair_attempt_count", "topology_hard_clearance",
                "topology_clearance_target", "topology_neighbor_k",
                "selected_saddle_value_bonus",
                "selected_tracking_cost", "selected_max_curvature",
                "selected_curvature_violation", "selected_turn_violation",
                "selected_expected_progress", "selected_refinement_used",
                "selected_refined_path_length", "selected_topology_diversity",
                "raw_minima", "raw_saddle", "safe_minima", "safe_saddle",
                "filtered_minima", "filtered_saddle", "used_minima",
                "used_saddle", "used_minima_count", "used_saddle_count",
                "candidate_corridor_count",
                "corridor_base_cost",
                "corridor_adp_mean", "corridor_adp_max",
                "corridor_adp_end", "corridor_total_cost",
                "corridor_adp_raw_mean", "corridor_adp_raw_max",
                "corridor_adp_raw_end", "corridor_adp_norm",
                "corridor_rank_base", "corridor_rank_total",
                "terminal_adp_cost", "mpc_total_cost", "mpc_social_cost",
                "mpc_tube_cost", "mpc_track_cost", "mpc_control_cost",
                "corridor_rank_changed_count", "final_approach_used",
                "mpc_reject_forbidden_count",
                "mpc_reject_interest_phi_count",
                "path_adp_mean", "path_adp_max", "path_adp_delta",
                "adp_path_enabled", "arm_adp_grad_norm",
                "protected_saddle_count", "protected_saddle_max_dist",
                "protected_saddle_ok", "mandatory_topology_node_count",
                "mandatory_saddle_reached", "mandatory_saddle_max_dist",
                "corridor_violation_count", "topology_tracking_error",
                "mpc_segment_count",
                "arm_adp_soft_cost", "arm_v_adp_alignment",
                "arm_adp_control_enabled", "arm_dls_adp_used",
                "arm_qp_used", "arm_solver_success_count",
                "arm_solver_fallback_count", "arm_solver_success_rate",
                "v_des_raw_norm", "v_des_adp_norm", "v_des_delta_norm",
                "dq_nominal_norm", "dq_adp_norm", "dq_delta_norm",
                "phi_ee_point", "phi_wrist", "phi_elbow", "phi_object",
                "phi_arm_max_point", "phi_arm_mean_point",
                "phi_arm_sum_point", "arm_worst_point_idx",
                "arm_interest_valid_count",
                "arm_ip_ee_x", "arm_ip_ee_y", "arm_ip_ee_z",
                "arm_ip_wrist_x", "arm_ip_wrist_y", "arm_ip_wrist_z",
                "arm_ip_elbow_x", "arm_ip_elbow_y", "arm_ip_elbow_z",
                "arm_ip_object_x", "arm_ip_object_y", "arm_ip_object_z",
                "arm_gate_source", "arm_interest_gate_enabled",
                "arm_interest_gate_risk", "arm_interest_gate_scale",
                "arm_interest_gate_stop", "arm_interest_rho_warn",
                "arm_interest_rho_stop", "arm_interest_gate_worst_idx",
                "yaw", "phi_center", "phi_front_center", "phi_front_left",
                "phi_front_right", "phi_footrest_left", "phi_footrest_right",
                "phi_rear_left", "phi_rear_right", "phi_max_point",
                "phi_mean_point", "phi_sum_point", "worst_point_idx",
                "forbidden_hit", "gate_source", "footprint_gate_enabled",
                "footprint_gate_risk", "footprint_gate_scale",
                "footprint_gate_stop", "footprint_rho_warn",
                "footprint_rho_stop",
            ]
            fields = full_fields if self.trajectory_debug_level >= 2 else compact_fields
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in self.traj:
                w.writerow({key: row.get(key, "") for key in fields})
        rospy.loginfo("[metrics] wrote trajectory %s", self.traj_out)

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    MetricsNode().run()
