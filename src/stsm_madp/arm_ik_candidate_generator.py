import sys
sys.dont_write_bytecode = True

import copy

import numpy as np


def _as_points(points):
    if points is None or isinstance(points, str):
        return np.zeros((0, 3), float)
    try:
        arr = np.asarray(points, float)
    except Exception:
        return np.zeros((0, 3), float)
    if arr.size == 0:
        return np.zeros((0, 3), float)
    if arr.ndim == 1:
        arr = arr.reshape((1, arr.shape[0]))
    if arr.shape[1] == 2:
        arr = np.hstack([arr, np.zeros((arr.shape[0], 1), float)])
    return arr[:, :3]


class ArmIKCandidateGenerator(object):
    """Generate arm execution variants for a fixed Morse topology route."""

    def __init__(self, lateral_offsets=None, vertical_offsets=None,
                 wrist_seed_offsets=None, elbow_postures=None,
                 wrist_angles=None, approach_directions=None,
                 max_variants=36):
        self.lateral_offsets = list(lateral_offsets or [0.0, 0.04, -0.04])
        self.vertical_offsets = list(vertical_offsets or [0.0, 0.04])
        self.wrist_seed_offsets = list(wrist_seed_offsets or [0.0])
        self.elbow_postures = list(elbow_postures or ["middle", "up", "down"])
        self.wrist_angles = list(wrist_angles or [-1.57079632679, 0.0, 1.57079632679])
        self.approach_directions = list(approach_directions or [
            "front", "left", "right"])
        self.max_variants = int(max(1, max_variants))

    def generate(self, candidate, seed=None):
        route = copy.deepcopy(candidate if isinstance(candidate, dict) else {})
        points = np.zeros((0, 3), float)
        for key in ("centerline", "waypoints", "refined_waypoints", "path"):
            points = _as_points(route.get(key, None))
            if len(points):
                break
        if len(points) == 0:
            return []
        variants = []
        tangent = points[-1, :2] - points[0, :2] if len(points) > 1 else np.array([1.0, 0.0])
        norm = float(np.linalg.norm(tangent))
        if norm <= 1e-9:
            tangent = np.array([1.0, 0.0], float)
        else:
            tangent = tangent / norm
        normal = np.array([-tangent[1], tangent[0], 0.0], float)
        vertical_axis = np.array([0.0, 0.0, 1.0], float)
        for approach in self.approach_directions:
            approach_sign = (
                -1.0 if approach == "left" else
                1.0 if approach == "right" else 0.0)
            for elbow in self.elbow_postures:
                elbow_sign = (
                    1.0 if elbow == "up" else
                    -1.0 if elbow == "down" else 0.0)
                for wrist_angle in self.wrist_angles:
                    for lateral in self.lateral_offsets:
                        for vertical in self.vertical_offsets:
                            pts = points.copy()
                            offset = (
                                (float(lateral) + 0.05 * approach_sign) * normal +
                                float(vertical) * vertical_axis)
                            if len(pts) > 2:
                                pts[1:-1, :3] = pts[1:-1, :3] + offset
                            if len(pts) > 3 and approach_sign != 0.0:
                                start_idx = max(1, len(pts) - 4)
                                pts[start_idx:-1, :3] = (
                                    pts[start_idx:-1, :3] +
                                    0.04 * approach_sign * normal)
                            wrist = float(wrist_angle)
                            variant = copy.deepcopy(route)
                            variant["centerline"] = pts.tolist()
                            variant["waypoints"] = pts.tolist()
                            variant["link_sample_records"] = self._link_records(
                                pts, normal, elbow_sign, wrist)
                            variant["link_sample_points"] = [
                                rec["point"] for rec in variant["link_sample_records"]
                            ]
                            variant["arm_ik_variant"] = {
                                "approach_direction": str(approach),
                                "elbow_posture": str(elbow),
                                "wrist_angle": float(wrist_angle),
                                "lateral_offset": float(lateral),
                                "vertical_offset": float(vertical),
                                "wrist_seed_offset": float(wrist),
                            }
                            q_seed = None
                            if seed is not None:
                                try:
                                    q_seed = np.asarray(seed, float).reshape((-1,)).copy()
                                    if len(q_seed) >= 5:
                                        q_seed[4] += float(wrist)
                                    if len(q_seed) >= 3:
                                        q_seed[2] += 0.35 * float(elbow_sign)
                                except Exception:
                                    q_seed = None
                            variants.append((variant, q_seed))
                            if len(variants) >= self.max_variants:
                                return variants
        return variants

    def _link_records(self, points, normal, elbow_sign, wrist_angle):
        records = []
        base = np.zeros(3, float)
        bend_axis = np.asarray(normal, float)[:3]
        wrist_axis = np.array([0.0, 0.0, 1.0], float)
        for tidx, ee in enumerate(np.asarray(points, float)):
            ee = np.asarray(ee, float)[:3]
            vec = ee - base
            elbow = base + 0.45 * vec + 0.10 * float(elbow_sign) * bend_axis
            wrist = base + 0.82 * vec + 0.04 * np.sin(float(wrist_angle)) * wrist_axis
            chain = [
                ("upper_arm", base, elbow),
                ("forearm", elbow, wrist),
                ("wrist", wrist, ee),
            ]
            for link, start, goal in chain:
                for sample in self._segment_samples(start, goal, 3):
                    records.append({
                        "trajectory_index": int(tidx),
                        "name": str(link),
                        "link": str(link),
                        "radius": 0.012 if link == "wrist" else 0.018,
                        "point": np.asarray(sample, float)[:3].tolist(),
                    })
            records.append({
                "trajectory_index": int(tidx),
                "name": "end_effector",
                "link": "end_effector",
                "radius": 0.0,
                "point": ee.tolist(),
            })
        return records

    def _segment_samples(self, start, goal, count):
        start = np.asarray(start, float)[:3]
        goal = np.asarray(goal, float)[:3]
        out = []
        for idx in range(1, int(count) + 1):
            alpha = float(idx) / float(int(count) + 1)
            out.append(start + alpha * (goal - start))
        return out
