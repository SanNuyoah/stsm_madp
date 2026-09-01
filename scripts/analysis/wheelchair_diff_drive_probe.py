#!/usr/bin/env python
"""Standalone Wheelchair Gazebo diff-drive actuation probe.

This intentionally contains no STSM, MPC, watchdog, or recovery logic.  It
reports controller-kinematic wheel targets alongside joint-state velocity and
effort, plus Gazebo and controller odometry displacement.  In suspended mode
it disables gravity for all dynamic wheelchair links and raises the model,
isolating the velocity actuators from wheel-ground contact.
"""
from __future__ import print_function

import json
import math
import os
import time

import rospy
from gazebo_msgs.msg import ContactsState, ModelState, ModelStates
from gazebo_msgs.srv import (GetLinkProperties, GetLinkState, GetPhysicsProperties,
                             SetLinkProperties, SetModelState,
                             SetPhysicsProperties)
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState


WHEEL_JOINTS = ("left_wheel_joint", "right_wheel_joint")
# Gazebo lumps fixed children (base_link, caster and seat_back) into the
# model's base_footprint body.  These are the scoped dynamic bodies reported
# by /gazebo/get_model_properties, not merely the URDF link names.
WHEEL_LINKS = ("wheelchair::base_footprint", "wheelchair::left_wheel",
               "wheelchair::right_wheel")
CMD_TOPIC = "/wheelchair/diff_drive_controller/cmd_vel"
ODOM_TOPIC = "/wheelchair/diff_drive_controller/odom"
JOINT_TOPIC = "/wheelchair/joint_states"
CONTACT_TOPICS = {
    "left_wheel_joint": "/wheelchair/left_wheel/contact",
    "right_wheel_joint": "/wheelchair/right_wheel/contact",
}


def wheel_contact_direction_audit():
    """Return the fixed geometry derivation used by the fdir1-only probe.

    Gazebo Classic / ODE interprets fdir1 in the collision-fixed frame.  A
    URDF cylinder is along local +z; the collision rpy rotates that axis onto
    link -y, collinear with the joint +y axle.  Collision +x is unchanged by
    that rotation and is tangential to ground in the link rolling direction.
    The sign of a friction direction is immaterial, so +x is used for both
    mirrored drive wheels and gives them the same physical meaning.
    """
    row = {
        "joint_axis_link": [0.0, 1.0, 0.0],
        "cylinder_axis_collision": [0.0, 0.0, 1.0],
        "collision_rpy": [1.5708, 0.0, 0.0],
        "cylinder_axis_link": [0.0, -1.0, 0.0],
        "rolling_direction_link": [1.0, 0.0, 0.0],
        "lateral_direction_link": [0.0, 1.0, 0.0],
        "rolling_direction_collision": [1.0, 0.0, 0.0],
        "lateral_direction_collision": [0.0, 0.0, -1.0],
        "candidate_fdir1": [1.0, 0.0, 0.0],
        "candidate_fdir1_semantic": "rolling_direction_collision",
    }
    return {
        "contract": "wheelchair_ode_contact_direction_audit_v1",
        "fdir1_frame": "collision_fixed",
        "mu1_direction": "fdir1",
        "mu2_direction": "contact_normal_cross_fdir1",
        "slip1_direction": "fdir1",
        "slip2_direction": "contact_normal_cross_fdir1",
        "left": dict(row),
        "right": dict(row),
    }


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _angle_delta(before, after):
    return math.atan2(math.sin(after - before), math.cos(after - before))


class DiffDriveProbe(object):
    def __init__(self):
        self.joint = {}
        self.model = None
        self.odom = None
        self.contacts = dict((name, {"contact_count": 0, "contacts": []})
                             for name in WHEEL_JOINTS)
        self.radius = float(rospy.get_param(
            "/wheelchair/diff_drive_controller/wheel_radius", 0.15))
        self.separation = float(rospy.get_param(
            "/wheelchair/diff_drive_controller/wheel_separation", 0.62))
        self.pub = rospy.Publisher(CMD_TOPIC, Twist, queue_size=10)
        rospy.Subscriber(JOINT_TOPIC, JointState, self._joint_cb, queue_size=20)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self._model_cb,
                         queue_size=20)
        rospy.Subscriber(ODOM_TOPIC, Odometry, self._odom_cb, queue_size=20)
        for joint, topic in CONTACT_TOPICS.items():
            rospy.Subscriber(topic, ContactsState, self._contact_cb,
                             callback_args=joint, queue_size=50)

    def _joint_cb(self, msg):
        velocity = dict(zip(msg.name, msg.velocity))
        effort = dict(zip(msg.name, msg.effort))
        self.joint = dict((name, {
            "velocity": float(velocity.get(name, 0.0)),
            "effort": float(effort.get(name, 0.0)),
        }) for name in WHEEL_JOINTS)

    def _model_cb(self, msg):
        try:
            index = msg.name.index("wheelchair")
        except ValueError:
            return
        pose = msg.pose[index]
        self.model = {"x": float(pose.position.x),
                      "y": float(pose.position.y),
                      "yaw": float(_yaw(pose.orientation))}

    def _odom_cb(self, msg):
        pose = msg.pose.pose
        self.odom = {"x": float(pose.position.x),
                     "y": float(pose.position.y),
                     "yaw": float(_yaw(pose.orientation))}

    @staticmethod
    def _other_collision(joint, first, second):
        own = "left_wheel_collision" if joint == "left_wheel_joint" else "right_wheel_collision"
        if own in first:
            return second
        if own in second:
            return first
        return second

    @staticmethod
    def _contact_kind(name):
        lowered = str(name).lower()
        if "ground_plane" in lowered:
            return "ground"
        if "wheelchair" in lowered and "wheel" not in lowered:
            return "chassis_or_caster"
        if "wheelchair" in lowered:
            return "other_wheel"
        return "unknown"

    def _contact_cb(self, msg, joint):
        rows = []
        for state in msg.states:
            force = state.total_wrench.force
            normal = state.contact_normals[0] if state.contact_normals else None
            fx, fy, fz = float(force.x), float(force.y), float(force.z)
            normal_force = None
            tangential_force = None
            if normal is not None:
                dot = fx * normal.x + fy * normal.y + fz * normal.z
                normal_force = abs(float(dot))
                tangential_force = math.sqrt(max(0.0, fx * fx + fy * fy + fz * fz - dot * dot))
            other = self._other_collision(joint, state.collision1_name,
                                          state.collision2_name)
            rows.append({
                "collision1": str(state.collision1_name),
                "collision2": str(state.collision2_name),
                "other_collision_name": str(other),
                "other_kind": self._contact_kind(other),
                "contact_position": ([float(state.contact_positions[0].x),
                                      float(state.contact_positions[0].y),
                                      float(state.contact_positions[0].z)]
                                     if state.contact_positions else None),
                "contact_normal": ([float(normal.x), float(normal.y), float(normal.z)]
                                   if normal is not None else None),
                "penetration_depth": (max([float(depth) for depth in state.depths])
                                      if state.depths else 0.0),
                "force": [fx, fy, fz],
                "normal_force": normal_force,
                "tangential_force": tangential_force,
            })
        self.contacts[joint] = {"contact_count": len(rows), "contacts": rows}

    def ready(self):
        return self.model is not None and self.odom is not None and all(
            name in self.joint for name in WHEEL_JOINTS)

    def snapshot(self, target=None):
        item = {
            "wall_time_s": float(time.time()),
            "gazebo_pose": dict(self.model or {}),
            "odom": dict(self.odom or {}),
            "wheel_actual": dict(self.joint),
            "wheel_contact": dict((name, dict(self.contacts.get(name, {})))
                                 for name in WHEEL_JOINTS),
        }
        if target is not None:
            item["wheel_target"] = dict(target)
            item["wheel_velocity_error"] = dict((name, float(
                target[name] - item["wheel_actual"].get(name, {}).get(
                    "velocity", 0.0))) for name in WHEEL_JOINTS)
        return item

    def targets(self, v, w):
        half = 0.5 * self.separation
        return {
            "left_wheel_joint": float((v - w * half) / self.radius),
            "right_wheel_joint": float((v + w * half) / self.radius),
        }

    def _set_suspended(self):
        get_props = rospy.ServiceProxy("/gazebo/get_link_properties",
                                       GetLinkProperties)
        set_props = rospy.ServiceProxy("/gazebo/set_link_properties",
                                       SetLinkProperties)
        set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        rospy.wait_for_service("/gazebo/get_link_properties", timeout=15.0)
        rospy.wait_for_service("/gazebo/set_link_properties", timeout=15.0)
        rospy.wait_for_service("/gazebo/set_model_state", timeout=15.0)
        applied = []
        for link in WHEEL_LINKS:
            props = get_props(link)
            if not props.success:
                raise RuntimeError("get_link_properties failed for %s: %s" %
                                   (link, props.status_message))
            result = set_props(link, props.com, False, props.mass, props.ixx,
                               props.ixy, props.ixz, props.iyy, props.iyz,
                               props.izz)
            if not result.success:
                raise RuntimeError("set_link_properties failed for %s: %s" %
                                   (link, result.status_message))
            applied.append(link)
        state = ModelState()
        state.model_name = "wheelchair"
        state.reference_frame = "world"
        state.pose.position.x = self.model["x"]
        state.pose.position.y = self.model["y"]
        state.pose.position.z = 0.50
        state.pose.orientation.w = 1.0
        result = set_state(state)
        if not result.success:
            raise RuntimeError("set_model_state failed: %s" % result.status_message)
        return applied

    def wheel_height_audit(self):
        get_state = rospy.ServiceProxy("/gazebo/get_link_state", GetLinkState)
        rospy.wait_for_service("/gazebo/get_link_state", timeout=15.0)
        rows = {}
        for joint, link in (("left_wheel_joint", "wheelchair::left_wheel"),
                            ("right_wheel_joint", "wheelchair::right_wheel")):
            response = get_state(link, "world")
            if not response.success:
                raise RuntimeError("get_link_state failed for %s: %s" %
                                   (link, response.status_message))
            center_z = float(response.link_state.pose.position.z)
            rows[joint] = {"link": link, "center_z": center_z,
                           "radius": self.radius,
                           "ground_z": 0.0,
                           "wheel_bottom_z": center_z - self.radius}
        return rows

    def physics_audit(self):
        """Capture the live Gazebo ODE configuration before solver trials."""
        get_physics = rospy.ServiceProxy("/gazebo/get_physics_properties",
                                         GetPhysicsProperties)
        rospy.wait_for_service("/gazebo/get_physics_properties", timeout=15.0)
        response = get_physics()
        if not response.success:
            raise RuntimeError("get_physics_properties failed: %s" %
                               response.status_message)
        ode = response.ode_config
        return {
            "time_step": float(response.time_step),
            "max_update_rate": float(response.max_update_rate),
            "paused": bool(response.pause),
            "gravity": [float(response.gravity.x), float(response.gravity.y),
                        float(response.gravity.z)],
            "ode": {
                "auto_disable_bodies": bool(ode.auto_disable_bodies),
                "sor_pgs_precon_iters": int(ode.sor_pgs_precon_iters),
                "sor_pgs_iters": int(ode.sor_pgs_iters),
                "sor_pgs_w": float(ode.sor_pgs_w),
                "sor_pgs_rms_error_tol": float(ode.sor_pgs_rms_error_tol),
                "contact_surface_layer": float(ode.contact_surface_layer),
                "contact_max_correcting_vel": float(
                    ode.contact_max_correcting_vel),
                "cfm": float(ode.cfm), "erp": float(ode.erp),
                "max_contacts": int(ode.max_contacts),
            },
        }

    def set_ode_sor_pgs_iterations(self, iterations):
        """Override only ODE SOR-PGS iterations for this probe process."""
        get_physics = rospy.ServiceProxy("/gazebo/get_physics_properties",
                                         GetPhysicsProperties)
        set_physics = rospy.ServiceProxy("/gazebo/set_physics_properties",
                                         SetPhysicsProperties)
        rospy.wait_for_service("/gazebo/get_physics_properties", timeout=15.0)
        rospy.wait_for_service("/gazebo/set_physics_properties", timeout=15.0)
        before = get_physics()
        if not before.success:
            raise RuntimeError("get_physics_properties failed: %s" %
                               before.status_message)
        ode = before.ode_config
        original = int(ode.sor_pgs_iters)
        ode.sor_pgs_iters = int(iterations)
        result = set_physics(before.time_step, before.max_update_rate,
                             before.gravity, ode)
        if not result.success:
            raise RuntimeError("set_physics_properties failed: %s" %
                               result.status_message)
        return {"parameter": "sor_pgs_iters", "before": original,
                "requested": int(iterations), "after": self.physics_audit()}

    def idle(self, duration_s, rate_hz):
        samples = []
        rate = rospy.Rate(rate_hz)
        deadline = time.time() + duration_s
        while not rospy.is_shutdown() and time.time() < deadline:
            samples.append(self.snapshot())
            rate.sleep()
        return {"duration_wall_s": float(duration_s), "samples": samples}

    def run(self, name, v, w, duration_s, rate_hz):
        target = self.targets(v, w)
        before = self.snapshot(target)
        samples = []
        message = Twist()
        message.linear.x = float(v)
        message.angular.z = float(w)
        rate = rospy.Rate(rate_hz)
        deadline = time.time() + duration_s
        while not rospy.is_shutdown() and time.time() < deadline:
            self.pub.publish(message)
            samples.append(self.snapshot(target))
            rate.sleep()
        self.pub.publish(Twist())
        time.sleep(0.25)
        after = self.snapshot(target)
        gazebo_translation = math.hypot(
            after["gazebo_pose"]["x"] - before["gazebo_pose"]["x"],
            after["gazebo_pose"]["y"] - before["gazebo_pose"]["y"])
        odom_translation = math.hypot(
            after["odom"]["x"] - before["odom"]["x"],
            after["odom"]["y"] - before["odom"]["y"])
        return {"name": name, "cmd_v": float(v), "cmd_w": float(w),
                "duration_wall_s": float(duration_s), "rate_hz": float(rate_hz),
                "target": target, "before": before, "after": after,
                "samples": samples,
                "gazebo_translation": float(gazebo_translation),
                "gazebo_delta_yaw": float(_angle_delta(
                    before["gazebo_pose"]["yaw"], after["gazebo_pose"]["yaw"])),
                "odom_translation": float(odom_translation),
                "odom_delta_yaw": float(_angle_delta(
                    before["odom"]["yaw"], after["odom"]["yaw"]))}


def main():
    rospy.init_node("wheelchair_diff_drive_probe")
    probe = DiffDriveProbe()
    suspended = bool(rospy.get_param("~suspended", False))
    duration_s = float(rospy.get_param("~duration_s", 4.0))
    rate_hz = float(rospy.get_param("~rate_hz", 20.0))
    output_path = rospy.get_param("~output_path",
                                  "/tmp/wheelchair_diff_drive_probe.json")
    spawn_z = float(rospy.get_param("~spawn_z", 0.05))
    wheel_mu = float(rospy.get_param("~wheel_mu", 1.0))
    wheel_fdir1_enabled = bool(rospy.get_param("~wheel_fdir1_enabled", False))
    wheel_lateral_slip_enabled = bool(rospy.get_param(
        "~wheel_lateral_slip_enabled", False))
    wheel_lateral_slip = float(rospy.get_param("~wheel_lateral_slip", 0.0))
    wheel_contact_stiffness_enabled = bool(rospy.get_param(
        "~wheel_contact_stiffness_enabled", False))
    wheel_contact_kp = float(rospy.get_param("~wheel_contact_kp", 10000000.0))
    wheel_contact_kd = float(rospy.get_param("~wheel_contact_kd", 1.0))
    wheel_contact_min_depth_enabled = bool(rospy.get_param(
        "~wheel_contact_min_depth_enabled", False))
    wheel_contact_min_depth = float(rospy.get_param(
        "~wheel_contact_min_depth", 0.001))
    wheel_contact_max_vel_enabled = bool(rospy.get_param(
        "~wheel_contact_max_vel_enabled", False))
    wheel_contact_max_vel = float(rospy.get_param("~wheel_contact_max_vel", 0.1))
    ode_sor_pgs_iterations = int(rospy.get_param("~ode_sor_pgs_iterations", 0))
    deadline = time.time() + 15.0
    while not rospy.is_shutdown() and not probe.ready() and time.time() < deadline:
        time.sleep(0.05)
    if not probe.ready():
        raise RuntimeError("probe topics were not ready")
    suspended_links = probe._set_suspended() if suspended else []
    physics_before = probe.physics_audit()
    physics_override = (probe.set_ode_sor_pgs_iterations(ode_sor_pgs_iterations)
                        if ode_sor_pgs_iterations > 0 else None)
    time.sleep(0.5)
    payload = {
        "contract": "wheelchair_diff_drive_physical_actuation_probe_v1",
        "mode": "suspended" if suspended else "grounded",
        "cmd_topic": CMD_TOPIC, "joint_state_topic": JOINT_TOPIC,
        "odom_topic": ODOM_TOPIC, "wheel_radius": probe.radius,
        "wheel_separation": probe.separation,
        "probe_spawn_z": spawn_z, "probe_wheel_mu": wheel_mu,
        "probe_wheel_fdir1_enabled": wheel_fdir1_enabled,
        "probe_wheel_lateral_slip_enabled": wheel_lateral_slip_enabled,
        "probe_wheel_lateral_slip": wheel_lateral_slip,
        "probe_wheel_contact_stiffness_enabled": wheel_contact_stiffness_enabled,
        "probe_wheel_contact_kp": wheel_contact_kp,
        "probe_wheel_contact_kd": wheel_contact_kd,
        "probe_wheel_contact_min_depth_enabled": wheel_contact_min_depth_enabled,
        "probe_wheel_contact_min_depth": wheel_contact_min_depth,
        "probe_wheel_contact_max_vel_enabled": wheel_contact_max_vel_enabled,
        "probe_wheel_contact_max_vel": wheel_contact_max_vel,
        "wheel_contact_direction_audit": wheel_contact_direction_audit(),
        "physics_audit": physics_before,
        "physics_override": physics_override,
        "suspended_links": suspended_links,
        "wheel_height_audit": probe.wheel_height_audit(),
        "idle": probe.idle(2.5, rate_hz),
        "straight": probe.run("straight", 0.2, 0.0, duration_s, rate_hz),
        "rotate": probe.run("rotate", 0.0, 0.5, duration_s, rate_hz),
    }
    parent = os.path.dirname(output_path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    direction_path = os.path.join(
        os.path.dirname(output_path) or ".", "wheel_contact_direction_audit.json")
    with open(direction_path, "w") as handle:
        json.dump(payload["wheel_contact_direction_audit"], handle,
                  indent=2, sort_keys=True)
    rospy.loginfo("[wheelchair_diff_drive_probe] wrote %s", output_path)


if __name__ == "__main__":
    main()
