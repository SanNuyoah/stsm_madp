import sys
sys.dont_write_bytecode = True

from stsm_madp.safety_evaluator import SafetyEvaluator


def _finite_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    if value != value:
        return float(default)
    return float(value)


class ManifoldConstraintEvaluator(SafetyEvaluator):
    """Unified safety evaluator for planning, refinement, MPC and metrics."""

    def __init__(self, manifold_constraint=None, corridor_constraint=None,
                 risk_field=None, planning_clearance_margin=0.0,
                 soft_tolerance=0.08, hard_tolerance=0.25):
        super(ManifoldConstraintEvaluator, self).__init__(
            manifold_constraint=manifold_constraint,
            corridor_constraint=corridor_constraint,
            risk_field=risk_field,
            planning_clearance_margin=planning_clearance_margin)
        self.soft_tolerance = _finite_float(soft_tolerance, 0.08)
        self.hard_tolerance = _finite_float(hard_tolerance, 0.25)

    def classify_violation(self, violation, consecutive_count=1):
        violation = max(0.0, _finite_float(violation, 0.0))
        consecutive_count = int(consecutive_count or 0)
        if violation <= 1e-9:
            return {
                "level": "none",
                "minor_violation": False,
                "major_violation": False,
            }
        major = bool(
            violation > self.hard_tolerance or
            (consecutive_count >= 5 and violation > self.soft_tolerance))
        return {
            "level": "major" if major else "minor",
            "minor_violation": not major,
            "major_violation": major,
        }

    def evaluate_clearances(self, planning=None, predicted=None,
                            execution=None):
        out = {}
        for prefix, traj in (("planning", planning),
                             ("predicted", predicted),
                             ("execution", execution)):
            if traj is None:
                continue
            status = self.evaluate_trajectory(traj)
            out["{}_clearance".format(prefix)] = float(
                status.get("min_clearance", 0.0))
            out["{}_max_risk".format(prefix)] = float(
                status.get("max_risk", 0.0))
            out["{}_manifold_valid".format(prefix)] = bool(
                status.get("manifold_violation_count", 0) == 0)
            out["{}_trajectory_valid".format(prefix)] = bool(
                status.get("valid", False))
        return out
