import sys
sys.dont_write_bytecode = True

import numpy as np


def _as_point(point):
    try:
        arr = np.asarray(point, float).reshape((-1,))
    except Exception:
        return np.zeros(3, float)
    if arr.size < 3:
        arr = np.hstack([arr, np.zeros(3 - arr.size, float)])
    return arr[:3]


def _segment_points(start, goal, count):
    start = _as_point(start)
    goal = _as_point(goal)
    n = max(1, int(count))
    return [
        (start + (float(idx + 1) / float(n + 1)) * (goal - start)).tolist()
        for idx in range(n)
    ]


class ArmInterestSampler(object):
    """Generate link-level arm interest points for safety evaluation."""

    DEFAULT_LINK_NAMES = [
        "base",
        "upper_arm",
        "forearm",
        "wrist",
        "end_effector",
    ]

    def __init__(self, upper_arm_samples=4, forearm_samples=4,
                 wrist_samples=2, include_object=True):
        self.upper_arm_samples = max(1, int(upper_arm_samples))
        self.forearm_samples = max(1, int(forearm_samples))
        self.wrist_samples = max(1, int(wrist_samples))
        self.include_object = bool(include_object)

    def sample_from_link_positions(self, link_positions, grasp_object=None):
        positions = dict(link_positions or {})
        base = _as_point(positions.get("base", positions.get("shoulder", [0, 0, 0])))
        upper = _as_point(positions.get("upper_arm", positions.get("elbow", base)))
        forearm = _as_point(positions.get("forearm", positions.get("wrist", upper)))
        wrist = _as_point(positions.get("wrist", forearm))
        ee = _as_point(positions.get("end_effector", positions.get("ee", wrist)))

        samples = []
        samples.append({"name": "base", "point": base.tolist(), "link": "base"})
        for point in _segment_points(base, upper, self.upper_arm_samples):
            samples.append({"name": "upper_arm", "point": point, "link": "upper_arm"})
        for point in _segment_points(upper, forearm, self.forearm_samples):
            samples.append({"name": "forearm", "point": point, "link": "forearm"})
        for point in _segment_points(forearm, wrist, self.wrist_samples):
            samples.append({"name": "wrist", "point": point, "link": "wrist"})
        samples.append({"name": "end_effector", "point": ee.tolist(), "link": "end_effector"})
        if self.include_object and grasp_object is not None:
            samples.append({
                "name": "grasp_object",
                "point": _as_point(grasp_object).tolist(),
                "link": "grasp_object",
            })
        return samples

    def sample_proxy_chain(self, ee_point, base_point=None, grasp_object=None):
        base = _as_point(base_point if base_point is not None else [0, 0, 0])
        ee = _as_point(ee_point)
        vec = ee - base
        links = {
            "base": base,
            "upper_arm": base + 0.45 * vec,
            "forearm": base + 0.75 * vec,
            "wrist": base + 0.92 * vec,
            "end_effector": ee,
        }
        return self.sample_from_link_positions(links, grasp_object=grasp_object)

    def sample_trajectory(self, ee_trajectory, base_point=None,
                          grasp_object_trajectory=None):
        points = []
        records = []
        ee = np.asarray(ee_trajectory if ee_trajectory is not None else [], float)
        if ee.size == 0:
            return {"points": [], "records": [], "points_by_link": {}}
        if ee.ndim == 1:
            ee = ee.reshape((1, ee.shape[0]))
        by_link = {}
        for idx, point in enumerate(ee):
            obj = None
            if grasp_object_trajectory is not None:
                try:
                    obj_arr = np.asarray(grasp_object_trajectory, float)
                    obj = obj_arr[min(idx, len(obj_arr) - 1)]
                except Exception:
                    obj = None
            for sample in self.sample_proxy_chain(point, base_point, obj):
                rec = dict(sample)
                rec["trajectory_index"] = int(idx)
                records.append(rec)
                points.append(list(rec["point"]))
                by_link.setdefault(str(rec["link"]), []).append(list(rec["point"]))
        return {"points": points, "records": records, "points_by_link": by_link}
