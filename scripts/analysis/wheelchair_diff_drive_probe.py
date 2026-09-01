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
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import GetLinkProperties, SetLinkProperties, SetModelState
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
        self.radius = float(rospy.get_param(
            "/wheelchair/diff_drive_controller/wheel_radius", 0.15))
        self.separation = float(rospy.get_param(
            "/wheelchair/diff_drive_controller/wheel_separation", 0.62))
        self.pub = rospy.Publisher(CMD_TOPIC, Twist, queue_size=10)
        rospy.Subscriber(JOINT_TOPIC, JointState, self._joint_cb, queue_size=20)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self._model_cb,
                         queue_size=20)
        rospy.Subscriber(ODOM_TOPIC, Odometry, self._odom_cb, queue_size=20)

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

    def ready(self):
        return self.model is not None and self.odom is not None and all(
            name in self.joint for name in WHEEL_JOINTS)

    def snapshot(self, target=None):
        item = {
            "wall_time_s": float(time.time()),
            "gazebo_pose": dict(self.model or {}),
            "odom": dict(self.odom or {}),
            "wheel_actual": dict(self.joint),
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
    deadline = time.time() + 15.0
    while not rospy.is_shutdown() and not probe.ready() and time.time() < deadline:
        time.sleep(0.05)
    if not probe.ready():
        raise RuntimeError("probe topics were not ready")
    suspended_links = probe._set_suspended() if suspended else []
    time.sleep(0.5)
    payload = {
        "contract": "wheelchair_diff_drive_physical_actuation_probe_v1",
        "mode": "suspended" if suspended else "grounded",
        "cmd_topic": CMD_TOPIC, "joint_state_topic": JOINT_TOPIC,
        "odom_topic": ODOM_TOPIC, "wheel_radius": probe.radius,
        "wheel_separation": probe.separation,
        "suspended_links": suspended_links,
        "straight": probe.run("straight", 0.2, 0.0, duration_s, rate_hz),
        "rotate": probe.run("rotate", 0.0, 0.5, duration_s, rate_hz),
    }
    parent = os.path.dirname(output_path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    rospy.loginfo("[wheelchair_diff_drive_probe] wrote %s", output_path)


if __name__ == "__main__":
    main()
