import sys
sys.dont_write_bytecode = True


DEFAULT_TASK_CONFIG = {
    "handover": {
        "task_weight": {
            "risk": 0.5,
            "distance": 0.3,
            "task": 0.2,
        },
    },
    "navigation": {
        "task_weight": {
            "risk": 0.7,
            "distance": 0.2,
            "task": 0.1,
        },
    },
}


def resolve_task_mode(mode, robot_type=""):
    name = str(mode or "").strip().lower()
    if not name:
        robot = str(robot_type or "").strip().lower()
        name = "handover" if robot == "arm" else "navigation"
    if name not in DEFAULT_TASK_CONFIG:
        name = "handover" if name == "arm" else name
        name = "navigation" if name == "wheelchair" else name
    return name if name in DEFAULT_TASK_CONFIG else "navigation"


def _float_or_default(value, default):
    try:
        return float(value)
    except Exception:
        return float(default)


def resolve_task_weight(task_mode, task_config=None, task_weight=None,
                        robot_type=""):
    mode = resolve_task_mode(task_mode, robot_type=robot_type)
    merged_config = dict(DEFAULT_TASK_CONFIG)
    for key, value in (task_config or {}).items():
        if isinstance(value, dict):
            base = dict(merged_config.get(str(key), {}) or {})
            nested = dict(base.get("task_weight", {}) or {})
            nested.update(dict(value.get("task_weight", {}) or {}))
            base.update(value)
            base["task_weight"] = nested
            merged_config[str(key)] = base
    weights = dict(merged_config.get(mode, {}).get("task_weight", {}) or {})
    weights.update(dict(task_weight or {}))
    return {
        "risk": _float_or_default(weights.get("risk", 0.0), 0.0),
        "distance": _float_or_default(weights.get("distance", 0.0), 0.0),
        "task": _float_or_default(weights.get("task", 0.0), 0.0),
    }


def weighted_task_candidate_cost(risk_cost, distance_cost, task_cost, weights):
    weights = dict(weights or {})
    return float(
        float(weights.get("risk", 0.0)) * float(risk_cost) +
        float(weights.get("distance", 0.0)) * float(distance_cost) +
        float(weights.get("task", 0.0)) * float(task_cost))
