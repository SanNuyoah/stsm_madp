"""Hard candidate validation shared by planning and execution audits.

The validator is deliberately side-effect free.  Callers may attach the
returned ``execution_validation`` payload to a candidate before ranking; the
existing planner remains backward compatible when no dynamic context is
available.
"""
import numpy as np


def _points(value):
    try:
        arr = np.asarray(value, float)
    except (TypeError, ValueError):
        return np.zeros((0, 3), float)
    if arr.size == 0:
        return np.zeros((0, 3), float)
    if arr.ndim == 1:
        arr = arr.reshape((1, -1))
    if arr.shape[1] == 2:
        arr = np.hstack([arr, np.zeros((len(arr), 1), float)])
    return arr[:, :3]


def validate_candidate_execution(candidate, state=None, goal=None,
                                 robot_type="wheelchair", limits=None):
    """Validate candidate geometry and optional dynamic execution profile.

    Safety/manifold/topology validity is supplied by the candidate generator;
    this function adds the execution-side hard contract without inventing a
    fallback path.  Missing dynamic context is reported as ``not_evaluated``
    rather than treated as safe.
    """
    limits = dict(limits or {})
    points = _points(candidate.get("waypoints", candidate) if isinstance(
        candidate, dict) else getattr(candidate, "waypoints", candidate))
    result = {
        "geometry_valid": bool(len(points) >= 2),
        "dynamic_evaluated": False,
        "dynamic_valid": True,
        "hard_valid": bool(len(points) >= 2),
        "reject_reason": "" if len(points) >= 2 else "geometry_invalid",
    }
    if len(points) < 2 or state is None or goal is None:
        result["dynamic_status"] = "not_evaluated"
        return result

    state = np.asarray(state, float).reshape(-1)
    goal = np.asarray(goal, float).reshape(-1)
    if state.size < 3 or goal.size < 2:
        result["dynamic_status"] = "not_evaluated"
        return result

    result["dynamic_evaluated"] = True
    robot = str(robot_type or "").lower()
    if robot == "wheelchair":
        from stsm_madp.mpc import wheelchair_nonholonomic_execution_profile
        profile = wheelchair_nonholonomic_execution_profile(
            points, state, goal,
            min_step=float(limits.get("min_step", 0.03)),
            initial_lookahead=float(limits.get("initial_lookahead", 0.12)),
            horizon_points=int(limits.get("horizon_points", 10)),
            executable_curvature=float(limits.get("max_curvature", 8.0)))
        result["execution_profile"] = dict(profile)
        # Optional kinodynamic checks use the same sampled path and never
        # alter it.  They are enabled when a positive sample period is given.
        # A geometric corridor has no timing semantics.  Only use dynamic
        # limits when the caller supplies a real sample period or timestamps.
        sample_times = candidate.get("sample_times") if isinstance(
            candidate, dict) else getattr(candidate, "sample_times", None)
        dt = float(limits.get("dt", 0.0) or 0.0)
        if sample_times is not None:
            try:
                times = np.asarray(sample_times, float).reshape(-1)
                if len(times) == len(points) and np.all(np.diff(times) > 1e-9):
                    dt = times
            except (TypeError, ValueError):
                pass
        has_timing = ((np.isscalar(dt) and float(dt) > 0.0) or
                      (not np.isscalar(dt) and len(dt) == len(points)))
        if has_timing and len(points) >= 3:
            lengths = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
            if np.isscalar(dt):
                intervals = np.full(len(lengths), float(dt), float)
            else:
                intervals = np.diff(dt)
            speeds = lengths / np.maximum(intervals, 1e-9)
            accelerations = np.diff(speeds) / np.maximum(
                intervals[1:], 1e-9)
            headings = np.unwrap(np.arctan2(
                np.diff(points[:, 1]), np.diff(points[:, 0])))
            omegas = (np.diff(headings) / np.maximum(intervals[1:], 1e-9)
                      if len(headings) >= 2 else np.zeros(0))
            alphas = (np.diff(omegas) / np.maximum(intervals[2:], 1e-9)
                      if len(omegas) >= 2 else np.zeros(0))
            result["kinodynamic"] = {
                "max_speed": float(np.max(speeds)) if len(speeds) else 0.0,
                "max_acceleration": float(np.max(np.abs(accelerations)))
                if len(accelerations) else 0.0,
                "max_omega": float(np.max(np.abs(omegas)))
                if len(omegas) else 0.0,
                "max_alpha": float(np.max(np.abs(alphas)))
                if len(alphas) else 0.0,
                "max_speed_limit": float(limits.get("max_speed", float("inf"))),
                "max_acceleration_limit": float(
                    limits.get("max_acceleration", float("inf"))),
                "max_omega_limit": float(limits.get("max_omega", float("inf"))),
                "max_alpha_limit": float(limits.get("max_alpha", float("inf"))),
            }
        else:
            result["kinodynamic"] = {"status": "not_evaluated"}
        checks = {
            "diff_drive_execution_cost": (
                float(profile.get("execution_profile_cost", 0.0)),
                float(limits.get("max_execution_cost", 50.0))),
            "initial_heading_error": (
                float(profile.get("initial_heading_error", 0.0)),
                float(limits.get("max_heading_error", 1.50))),
            "max_local_curvature": (
                float(profile.get("max_local_curvature", 0.0)),
                float(limits.get("max_curvature", 8.0))),
        }
        result["dynamic_checks"] = {
            key: {"value": value, "limit": limit, "valid": bool(
                np.isfinite(value) and value <= limit + 1e-9)}
            for key, (value, limit) in checks.items()}
        if result["kinodynamic"].get("status") != "not_evaluated":
            kin = result["kinodynamic"]
            result["dynamic_checks"].update({
                "max_speed": {"value": kin["max_speed"],
                              "limit": kin["max_speed_limit"],
                              "valid": kin["max_speed"] <=
                              kin["max_speed_limit"] + 1e-9},
                "max_acceleration": {"value": kin["max_acceleration"],
                                     "limit": kin["max_acceleration_limit"],
                                     "valid": kin["max_acceleration"] <=
                                     kin["max_acceleration_limit"] + 1e-9},
                "max_omega": {"value": kin["max_omega"],
                               "limit": kin["max_omega_limit"],
                               "valid": kin["max_omega"] <=
                               kin["max_omega_limit"] + 1e-9},
                "max_alpha": {"value": kin["max_alpha"],
                               "limit": kin["max_alpha_limit"],
                               "valid": kin["max_alpha"] <=
                               kin["max_alpha_limit"] + 1e-9},
            })
        bad = [key for key, item in result["dynamic_checks"].items()
               if not item["valid"]]
        result["dynamic_valid"] = not bad
        result["dynamic_status"] = "valid" if not bad else "invalid"
        result["hard_valid"] = bool(result["geometry_valid"] and
                                     result["dynamic_valid"])
        result["reject_reason"] = "|".join(bad)
        return result

    # Arm candidates provide IK/link validation upstream.  Preserve those
    # facts here so ranking can consume one common contract.
    ik_valid = candidate.get("ik_valid", True) if isinstance(candidate, dict) \
        else getattr(candidate, "ik_valid", True)
    link_valid = candidate.get("link_collision_valid", True) \
        if isinstance(candidate, dict) else getattr(
            candidate, "link_collision_valid", True)
    result["dynamic_status"] = "valid"
    result["dynamic_valid"] = bool(ik_valid and link_valid)
    result["hard_valid"] = bool(result["geometry_valid"] and
                                 result["dynamic_valid"])
    if not result["dynamic_valid"]:
        result["reject_reason"] = "ik_or_link_collision_invalid"
    return result


__all__ = ["validate_candidate_execution"]
