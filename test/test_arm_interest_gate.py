import os
import sys
import types


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
NODES = os.path.join(ROOT, "nodes")
for path in (SRC, NODES):
    if path not in sys.path:
        sys.path.insert(0, path)


def _install_module(name, module):
    if name not in sys.modules:
        sys.modules[name] = module


def _stub_ros_modules():
    rospy = types.ModuleType("rospy")
    rospy.init_node = lambda *args, **kwargs: None
    rospy.get_param = lambda name, default=None: default
    rospy.Publisher = lambda *args, **kwargs: None
    rospy.Time = type("Time", (), {"now": staticmethod(lambda: None)})
    rospy.Duration = lambda value: value
    rospy.Rate = lambda hz: None
    rospy.is_shutdown = lambda: False
    rospy.loginfo = lambda *args, **kwargs: None
    rospy.logwarn = lambda *args, **kwargs: None
    rospy.sleep = lambda *args, **kwargs: None
    _install_module("rospy", rospy)

    moveit = types.ModuleType("moveit_commander")
    moveit.roscpp_initialize = lambda *args, **kwargs: None
    moveit.RobotCommander = lambda *args, **kwargs: None
    moveit.MoveGroupCommander = lambda *args, **kwargs: None
    _install_module("moveit_commander", moveit)

    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    for name in ("Float64", "Float64MultiArray", "Int32", "String"):
        setattr(std_msgs_msg, name, type(name, (), {}))
    _install_module("std_msgs", std_msgs)
    _install_module("std_msgs.msg", std_msgs_msg)

    geometry_msgs = types.ModuleType("geometry_msgs")
    geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")
    geometry_msgs_msg.PointStamped = type("PointStamped", (), {})
    _install_module("geometry_msgs", geometry_msgs)
    _install_module("geometry_msgs.msg", geometry_msgs_msg)

    trajectory_msgs = types.ModuleType("trajectory_msgs")
    trajectory_msgs_msg = types.ModuleType("trajectory_msgs.msg")
    trajectory_msgs_msg.JointTrajectory = type("JointTrajectory", (), {})
    trajectory_msgs_msg.JointTrajectoryPoint = type(
        "JointTrajectoryPoint", (), {})
    _install_module("trajectory_msgs", trajectory_msgs)
    _install_module("trajectory_msgs.msg", trajectory_msgs_msg)


_stub_ros_modules()

from handover_node import HandoverNode
from stsm_madp.safety_gate import SafetyGate


def _combine(ee_gate, arm_gate=None, interest_eval=None):
    method = HandoverNode._combine_arm_gates
    func = getattr(method, "__func__", None)
    if func is None:
        func = getattr(method, "im" + "_func", method)
    return func(None, ee_gate, arm_gate, interest_eval)


def test_arm_interest_stop_overrides_normal_ee_gate():
    ee_gate = SafetyGate(rho_warn=3.5, rho_stop=6.0).evaluate(0.2)
    arm_gate = SafetyGate(rho_warn=3.5, rho_stop=6.0).evaluate(6.1)
    interest_eval = {"worst_label": "wrist", "phi_max": 6.1}

    gate, source = _combine(ee_gate, arm_gate, interest_eval)

    assert gate.state == "STOP"
    assert gate.stop is True
    assert gate.scale == 0.0
    assert gate.reason == "arm_interest:risk_stop:wrist"
    assert gate.risk == 6.1
    assert source == "arm_interest"


def test_arm_interest_warn_triggers_slow_from_normal_ee_gate():
    ee_gate = SafetyGate(rho_warn=3.5, rho_stop=6.0).evaluate(0.2)
    arm_gate = SafetyGate(
        rho_warn=3.5, rho_stop=6.0, min_scale=0.2).evaluate(4.75)
    interest_eval = {"worst_label": "elbow", "phi_max": 4.75}

    gate, source = _combine(ee_gate, arm_gate, interest_eval)

    assert gate.state == "SLOW"
    assert gate.stop is False
    assert abs(gate.scale - 0.6) < 1e-9
    assert gate.reason == "arm_interest:risk_slow:elbow"
    assert gate.risk == 4.75
    assert source == "arm_interest"


def test_combined_slow_uses_stricter_scale_and_combined_source():
    ee_gate = SafetyGate(
        rho_warn=3.5, rho_stop=6.0, min_scale=0.2).evaluate(4.0)
    arm_gate = SafetyGate(
        rho_warn=3.5, rho_stop=6.0, min_scale=0.2).evaluate(5.0)
    interest_eval = {"worst_label": "gripper", "phi_max": 5.0}

    gate, source = _combine(ee_gate, arm_gate, interest_eval)

    assert gate.state == "SLOW"
    assert gate.stop is False
    assert abs(gate.scale - arm_gate.scale) < 1e-9
    assert gate.reason == "combined:risk_slow:gripper"
    assert gate.risk == 5.0
    assert source == "combined"
