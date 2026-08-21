import csv
import math
import os


def _float(value, default=0.0):
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_rows(path):
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def evaluate_arm_trajectory_rows(rows, risk_threshold=6.0):
    """Evaluate arm-specific safety from recorded EE and link-risk samples."""
    rows = list(rows or [])
    link_fields = [
        "phi_wrist",
        "phi_elbow",
        "phi_object",
        "phi_arm_max_point",
        "arm_interest_gate_risk",
    ]
    max_by_field = {}
    for field in link_fields:
        values = [_float(row.get(field), 0.0) for row in rows]
        max_by_field[field] = max(values) if values else 0.0
    max_link_risk = max(max_by_field.values()) if max_by_field else 0.0
    ee_values = [_float(row.get("phi_ee_point", row.get("phi_total")), 0.0)
                 for row in rows]
    max_ee_risk = max(ee_values) if ee_values else 0.0
    position_error = 0.0
    if rows:
        last = rows[-1]
        position_error = math.sqrt(
            _float(last.get("vx"), 0.0) ** 2 +
            _float(last.get("vy"), 0.0) ** 2 +
            _float(last.get("vz"), 0.0) ** 2)
    yaw_values = [_float(row.get("yaw"), 0.0) for row in rows
                  if row.get("yaw", "") not in ("", None)]
    orientation_error = 0.0
    if len(yaw_values) >= 2:
        orientation_error = abs(yaw_values[-1] - yaw_values[0])
        while orientation_error > math.pi:
            orientation_error = abs(orientation_error - 2.0 * math.pi)
    gripper_risk = max_by_field.get("phi_object", 0.0)
    violation_count = sum(
        1 for row in rows
        if max(_float(row.get(field), 0.0) for field in link_fields) >
        float(risk_threshold))
    return {
        "link_risk_evaluated": bool(rows),
        "link_sample_count": int(len(rows)),
        "max_link_risk": float(max_link_risk),
        "max_end_effector_risk": float(max_ee_risk),
        "end_effector_error": float(position_error),
        "orientation_error": float(orientation_error),
        "gripper_valid": bool(gripper_risk <= float(risk_threshold)),
        "risk_threshold": float(risk_threshold),
        "link_violation_count": int(violation_count),
        "link_risk_by_name": {
            "link1": float(max_by_field.get("phi_elbow", 0.0)),
            "link2": float(max_by_field.get("phi_wrist", 0.0)),
            "link3": float(max_by_field.get("phi_arm_max_point", 0.0)),
            "gripper": float(max_by_field.get("phi_object", 0.0)),
        },
    }


def evaluate_arm_trajectory_csv(path, risk_threshold=6.0):
    return evaluate_arm_trajectory_rows(
        _read_rows(path), risk_threshold=risk_threshold)
