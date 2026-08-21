import sys
sys.dont_write_bytecode = True

import numpy as np


class CriticalPointTracker(object):
    INIT = "INIT"
    REACH_SADDLE = "REACH_SADDLE"
    REACH_MINIMUM = "REACH_MINIMUM"
    GOAL = "GOAL"

    def __init__(self, critical_points=None, threshold=0.25):
        self.threshold = float(threshold)
        self.sequence = self._normalize_sequence(critical_points or [])
        self.index = 0
        self.passed_points = []
        self.sequence_valid = True
        self.invalid_reason = ""
        self.state = self.INIT
        self._update_state()

    def _normalize_sequence(self, critical_points):
        items = []
        for idx, item in enumerate(critical_points or []):
            if not isinstance(item, dict):
                continue
            point = item.get("point", item.get("position", None))
            if point is None:
                continue
            try:
                p = np.asarray(point, float)
            except Exception:
                continue
            if p.size < 2:
                continue
            if p.size == 2:
                p = np.asarray([p[0], p[1], 0.0], float)
            order = item.get("order", idx + 1)
            try:
                order = int(order)
            except Exception:
                order = idx + 1
            kind = str(item.get("type", item.get("kind", "critical")))
            if kind == "minima":
                kind = "minimum"
            items.append({
                "id": str(item.get("id", "critical_{}".format(idx))),
                "type": kind,
                "point": p[:3],
                "order": order,
            })
        items.sort(key=lambda x: int(x.get("order", 0)))
        return items

    def _update_state(self):
        if self.index >= len(self.sequence):
            self.state = self.GOAL
            return
        target_type = str(self.sequence[self.index].get("type", ""))
        if target_type == "saddle":
            self.state = self.REACH_SADDLE
        elif target_type in ("minimum", "minima"):
            self.state = self.REACH_MINIMUM
        else:
            self.state = self.INIT

    def current_target(self):
        if self.index >= len(self.sequence):
            return None
        return self.sequence[self.index]

    def update(self, point):
        p = np.asarray(point, float)
        if p.size == 2:
            p = np.asarray([p[0], p[1], 0.0], float)
        p = p[:3]
        target = self.current_target()
        target_id = ""
        target_distance = 0.0
        if target is not None:
            target_id = str(target.get("id", ""))
            target_distance = float(np.linalg.norm(
                p - np.asarray(target.get("point"), float)[:3]))
            for future in self.sequence[self.index + 1:]:
                future_d = float(np.linalg.norm(
                    p - np.asarray(future.get("point"), float)[:3]))
                if future_d <= self.threshold and target_distance > self.threshold:
                    self.sequence_valid = False
                    self.invalid_reason = "out_of_order:{}_before_{}".format(
                        future.get("id", ""), target_id)
                    break
            if target_distance <= self.threshold:
                self.passed_points.append(target_id)
                self.index += 1
                self._update_state()
        status = self.status()
        status.update({
            "critical_point_target": target_id,
            "critical_point_distance": float(target_distance),
            "critical_sequence_state": self.state,
            "topology_sequence_valid": bool(self.sequence_valid),
        })
        return status

    def status(self):
        target = self.current_target()
        return {
            "current_target": str(target.get("id", "")) if target else "",
            "passed_points": list(self.passed_points),
            "sequence_valid": bool(self.sequence_valid),
            "invalid_reason": self.invalid_reason,
            "state": self.state,
            "complete": bool(self.index >= len(self.sequence)),
        }
