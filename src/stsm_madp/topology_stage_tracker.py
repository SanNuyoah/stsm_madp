import sys
sys.dont_write_bytecode = True


class TopologyStageTracker(object):
    def __init__(self, association=None):
        association = association or {}
        self.points = list(association.get("critical_points", []) or [])
        self.points.sort(key=lambda item: int(item.get("stage_order", 0)))
        self.current_stage = 0
        self.passed_stages = [0]
        self.passed_critical_points = []
        self.sequence_valid = True
        self.invalid_reason = ""

    def update_index(self, trajectory_index):
        idx = int(trajectory_index)
        for point in self.points:
            order = int(point.get("stage_order", 0))
            if order <= self.current_stage:
                continue
            target_idx = int(point.get("trajectory_index", -1))
            if target_idx < 0:
                self.sequence_valid = False
                self.invalid_reason = "missing_stage_projection"
                break
            if idx >= target_idx:
                if order != self.current_stage + 1:
                    self.sequence_valid = False
                    self.invalid_reason = "stage_order_invalid"
                    break
                self.current_stage = order
                self.passed_stages.append(order)
                self.passed_critical_points.append(str(point.get("id", "")))
        return self.status()

    def finish(self):
        if self.sequence_valid:
            goal_stage = len(self.points) + 1
            if self.current_stage == len(self.points):
                self.current_stage = goal_stage
                self.passed_stages.append(goal_stage)
        return self.status()

    def status(self):
        return {
            "current_stage": int(self.current_stage),
            "passed_stages": list(self.passed_stages),
            "passed_critical_points": list(self.passed_critical_points),
            "sequence_valid": bool(self.sequence_valid),
            "invalid_reason": str(self.invalid_reason),
        }
