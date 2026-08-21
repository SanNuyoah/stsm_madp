#!/usr/bin/env python
import os
import sys
sys.dont_write_bytecode = True
import numpy as np
import rospy

from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

PACKAGE_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if os.path.isdir(os.path.join(PACKAGE_SRC, "stsm_madp")) and PACKAGE_SRC not in sys.path:
    sys.path.insert(0, PACKAGE_SRC)

from stsm_madp.social_field import HumanState, SemanticAnchor, SocialField, SocialFieldParams


def _pt(data):
    return np.array(data, float)


def _rgba(r, g, b, a):
    from std_msgs.msg import ColorRGBA
    return ColorRGBA(float(r), float(g), float(b), float(a))


class SocialFieldVizNode:
    def __init__(self):
        rospy.init_node("stsm_social_field_viz")
        self.target = rospy.get_param("~target", "arm")
        self.resolution = float(rospy.get_param("~resolution", 0.08))
        self.max_risk = float(rospy.get_param("~max_risk", 4.0))
        self.paths = {"baseline": [], "stsm": []}
        self.mode = "stsm"
        self.current = None

        if self.target == "wheelchair":
            self.frame_id = rospy.get_param("~frame_id", "odom")
            self.slice_z = float(rospy.get_param("~slice_z", 0.03))
            self.bounds = rospy.get_param("~bounds", [-2.1, 2.5, -2.0, 2.0])
            self.field = self._build_wheelchair_field()
            rospy.Subscriber("/stsm/wc_pos", PointStamped, self._point_cb, queue_size=50)
            rospy.Subscriber("/stsm/wc_mode", String, self._mode_cb, queue_size=5)
        else:
            self.frame_id = rospy.get_param("~frame_id", "elfin_base_link")
            self.slice_z = float(rospy.get_param("~slice_z", 0.21))
            self.bounds = rospy.get_param("~bounds", [0.05, 1.0, -0.55, 0.55])
            scene = rospy.get_param("/stsm_handover/scene", self._default_arm_scene())
            self.field = self._build_arm_field(scene)
            rospy.Subscriber("/stsm/ee_pose", PointStamped, self._point_cb, queue_size=50)
            rospy.Subscriber("/stsm/mode", String, self._mode_cb, queue_size=5)

        self.pub = rospy.Publisher("/stsm/social_field_markers", MarkerArray,
                                   queue_size=1, latch=True)
        self.timer = rospy.Timer(rospy.Duration(0.2), self._publish)
        rospy.sleep(0.5)
        self._publish(None)

    def _default_arm_scene(self):
        return {
            "person": {
                "heading": 3.14159,
                "posture": "sitting",
                "vulnerability": 1.3,
                "ref_pos": [0.78, 0.0, 0.31],
                "body_parts": {
                    "head": {"pos": [0.78, 0.0, 0.61], "weight": 3.0, "sigma": 0.13},
                    "chest": {"pos": [0.78, 0.0, 0.31], "weight": 1.6, "sigma": 0.18},
                    "hand": {"pos": [0.42, 0.0, 0.21], "weight": 0.3, "sigma": 0.10},
                },
            },
            "anchors": {
                "table": {
                    "type": "table", "center": [0.55, 0.0, -0.37],
                    "half_extent": [0.30, 0.50, 0.37],
                    "weight": 1.0, "forbidden": False,
                }
            },
            "field": {
                "lam_prox": 1.0, "lam_close": 1.2, "lam_dir": 0.6,
                "lam_body": 2.5, "lam_env": 0.8, "sigma_env": 0.25,
            },
        }

    def _build_arm_field(self, scene):
        p = scene.get("person", {})
        bp = {}
        for name, data in p.get("body_parts", {}).items():
            bp[name] = (_pt(data["pos"]), float(data["weight"]), float(data["sigma"]))
        human = HumanState(
            pos=_pt(p.get("ref_pos", [0.78, 0.0, 0.31])),
            heading=float(p.get("heading", np.pi)),
            posture=p.get("posture", "sitting"),
            vulnerability=float(p.get("vulnerability", 1.3)),
            body_parts=bp)
        anchors = []
        for _, data in scene.get("anchors", {}).items():
            anchors.append(SemanticAnchor(
                data["type"], _pt(data["center"]), _pt(data["half_extent"]),
                weight=float(data.get("weight", 1.0)),
                forbidden=bool(data.get("forbidden", False))))
        fc = scene.get("field", {})
        field = SocialField(SocialFieldParams(
            lam_prox=fc.get("lam_prox", 1.0),
            lam_close=fc.get("lam_close", 1.2),
            lam_dir=fc.get("lam_dir", 0.6),
            lam_body=fc.get("lam_body", 2.5),
            lam_env=fc.get("lam_env", 0.8),
            sigma_env=fc.get("sigma_env", 0.25)))
        field.set_scene([human], anchors)
        return field

    def _build_wheelchair_field(self):
        human = HumanState(pos=[-1.6, 0.2, 0.0], heading=np.pi / 2,
                           posture="transferring", vulnerability=1.4)
        bed = SemanticAnchor("bed", [-1.6, -1.0, 0.0], [0.5, 1.0, 0.5],
                             weight=2.0, forbidden=True)
        transfer = SemanticAnchor("transfer-zone", [-0.7, -1.0, 0.0],
                                  [0.4, 1.0, 0.5], weight=2.5, forbidden=True)
        table = SemanticAnchor("table", [0.55, 0.0, 0.0], [0.3, 0.5, 0.4],
                               weight=1.0, forbidden=True)
        field = SocialField(SocialFieldParams(
            lam_prox=1.2, lam_close=1.0, lam_dir=0.5, lam_body=0.0,
            lam_env=1.5, sigma_env=0.4))
        field.set_scene([human], [bed, transfer, table])
        return field

    def _mode_cb(self, msg):
        if msg.data in self.paths:
            self.mode = msg.data
            if self.target == "wheelchair":
                self.paths[msg.data] = []
                self.current = None

    def _point_cb(self, msg):
        p = np.array([msg.point.x, msg.point.y, msg.point.z], float)
        self.current = p
        path = self.paths.setdefault(self.mode, [])
        if not path or np.linalg.norm(path[-1] - p) > 0.015:
            path.append(p)

    def _risk_color(self, risk):
        t = max(0.0, min(1.0, risk / self.max_risk))
        if t < 0.5:
            k = t / 0.5
            return _rgba(0.18 + 0.72 * k, 0.49 + 0.25 * k, 0.74 - 0.54 * k, 0.28)
        k = (t - 0.5) / 0.5
        return _rgba(0.90 + 0.02 * k, 0.74 - 0.43 * k, 0.20 - 0.05 * k, 0.34)

    def _make_risk_marker(self):
        x_min, x_max, y_min, y_max = [float(v) for v in self.bounds]
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "social_risk_field"
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.resolution * 0.92
        marker.scale.y = self.resolution * 0.92
        marker.scale.z = 0.012 if self.target == "wheelchair" else 0.018
        xs = np.arange(x_min, x_max + self.resolution * 0.5, self.resolution)
        ys = np.arange(y_min, y_max + self.resolution * 0.5, self.resolution)
        for x in xs:
            for y in ys:
                risk = self.field.phi_s(np.array([x, y, self.slice_z]))
                marker.points.append(Point(float(x), float(y), self.slice_z))
                marker.colors.append(self._risk_color(risk))
        return marker

    def _make_path_marker(self, mode, marker_id, color):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "social_field_paths"
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.025 if self.target == "wheelchair" else 0.018
        marker.color = color
        for p in self.paths.get(mode, []):
            marker.points.append(Point(float(p[0]), float(p[1]), float(p[2])))
        return marker

    def _make_current_marker(self):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "social_field_current"
        marker.id = 3
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD if self.current is not None else Marker.DELETE
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = (
            0.09 if self.target == "wheelchair" else 0.045)
        marker.color = _rgba(0.12, 0.12, 0.12, 1.0)
        if self.current is not None:
            marker.pose.position = Point(float(self.current[0]),
                                         float(self.current[1]),
                                         float(self.current[2]))
        return marker

    def _publish(self, _event):
        markers = MarkerArray()
        markers.markers.append(self._make_risk_marker())
        markers.markers.append(self._make_path_marker(
            "baseline", 1, _rgba(0.0, 0.45, 0.70, 1.0)))
        markers.markers.append(self._make_path_marker(
            "stsm", 2, _rgba(0.84, 0.37, 0.0, 1.0)))
        markers.markers.append(self._make_current_marker())
        self.pub.publish(markers)


if __name__ == "__main__":
    try:
        SocialFieldVizNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
