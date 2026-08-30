import os
import sys
import hashlib
sys.dont_write_bytecode = True

import numpy as np
import yaml


STATE_FEATURE_NAMES = [
    "bias",
    "phi_total",
    "phi_prox",
    "phi_close",
    "phi_body",
    "phi_env",
    "risk_exceed",
    "gate_slow",
    "gate_stop",
    "d_goal",
    "progress",
    "speed",
    "ip_phi_max",
    "ip_phi_mean",
    "d_corridor",
    "phase_norm",
]

CANDIDATE_FEATURE_NAMES = [
    "candidate_path_length",
    "candidate_risk_mean",
    "candidate_risk_max",
    "candidate_min_clearance",
    "candidate_task_cost",
    "candidate_execution_cost",
]

FEATURE_SCHEMA_VERSION = "candidate_conditioned_v2"
DEFAULT_FEATURE_NAMES = STATE_FEATURE_NAMES + CANDIDATE_FEATURE_NAMES

DEFAULT_COST_WEIGHTS = {
    "w_phi": 1.0,
    "w_ip": 0.6,
    "w_exceed": 3.0,
    "w_stop": 20.0,
    "w_corr": 0.5,
    "w_u": 0.05,
    "w_prog": 0.4,
}

DEFAULT_LEARNING_CONFIG = {
    "enabled": True,
    "decision_influence_enabled": False,
    "ranking_influence_enabled": False,
    "mpc_influence_enabled": False,
    "alpha": 0.001,
    "td_error_clip": 5.0,
    "theta_delta_norm_max": 0.05,
    "min_transition_dt": 0.1,
    "save_updated_critic": True,
    "save_every_n_transitions": 50,
    "lambda_adp": 0.0,
    "risk_scale": 2.0,
    "failure_terminal_penalty": 3.0,
    "adp_value_normalization": "robust",
    "adp_norm_clip": 3.0,
    "adp_contribution_clip": 0.10,
}

DEFAULT_TRANSITION_COST_WEIGHTS = {
    "risk": 1.0,
    "progress": 1.0,
    "control": 0.1,
    "task": 0.2,
    "tube": 1.0,
    "failure": 1.0,
}


def _as_float(value, default=0.0):
    try:
        if value == "" or value is None:
            return float(default)
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return float(default)
        return value
    except (TypeError, ValueError):
        return float(default)


def _safe_std(std):
    std = np.asarray(std, float)
    std[np.abs(std) < 1e-6] = 1.0
    return std


def candidate_feature_values(candidate=None):
    """Normalize the stable candidate summary into the shared v2 schema."""
    candidate = dict(candidate or {})
    aliases = {
        "candidate_path_length": ("path_length", "length_cost"),
        "candidate_risk_mean": ("risk_mean", "risk_cost"),
        "candidate_risk_max": ("risk_max", "max_risk", "risk_value"),
        "candidate_min_clearance": ("min_clearance", "clearance_value"),
        "candidate_task_cost": ("task_cost",),
        "candidate_execution_cost": (
            "execution_cost", "motion_cost", "smoothness_cost"),
    }
    values = {}
    missing = {}
    for name in CANDIDATE_FEATURE_NAMES:
        value = None
        for key in aliases[name]:
            if candidate.get(key) not in (None, ""):
                value = candidate.get(key)
                break
        missing[name] = value is None
        values[name] = _as_float(value, 0.0)
    return values, missing


def require_feature_schema(critic, expected_version=FEATURE_SCHEMA_VERSION,
                           expected_feature_names=DEFAULT_FEATURE_NAMES):
    """Reject an incompatible critic instead of silently resizing its weights."""
    metadata = dict(getattr(critic, "metadata", {}) or {})
    version = str(metadata.get("feature_schema_version", ""))
    names = list(getattr(critic, "feature_names", []) or [])
    if version != str(expected_version) or names != list(expected_feature_names):
        raise ValueError(
            "critic_feature_schema_mismatch: expected {} / {} features, got {} / {}".format(
                expected_version, len(expected_feature_names), version, len(names)))
    return True


class ADPCritic(object):
    def __init__(self, feature_names=None, theta=None, mean=None, std=None,
                 gamma=0.95, clip_value=100.0, cost_weights=None,
                 critic_version="linear_adp_v1", metadata=None,
                 learning_config=None):
        self.feature_names = list(feature_names or DEFAULT_FEATURE_NAMES)
        n = len(self.feature_names)
        self.theta = np.asarray(theta if theta is not None else np.zeros(n), float)
        self.mean = np.asarray(mean if mean is not None else np.zeros(n), float)
        self.std = _safe_std(std if std is not None else np.ones(n))
        self.gamma = float(gamma)
        self.clip_value = float(clip_value)
        self.cost_weights = dict(DEFAULT_COST_WEIGHTS)
        if cost_weights:
            self.cost_weights.update(cost_weights)
        self.critic_version = critic_version
        self.metadata = dict(metadata or {})
        self.learning_config = dict(DEFAULT_LEARNING_CONFIG)
        if learning_config:
            self.learning_config.update(learning_config)
        for name, values in (("theta", self.theta), ("mean", self.mean),
                             ("std", self.std)):
            if values.shape[0] != n:
                raise ValueError(
                    "critic_feature_schema_mismatch: {} has {} values for {} features".format(
                        name, values.shape[0], n))

    def featurize(self, raw_feature_dict):
        raw = raw_feature_dict or {}
        values = []
        for name in self.feature_names:
            default = 1.0 if name == "bias" else 0.0
            values.append(_as_float(raw.get(name, default), default))
        x = np.asarray(values, float)
        return (x - self.mean) / self.std

    def predict(self, raw_feature_dict):
        return self.predict_detail(raw_feature_dict)["clipped"]

    def predict_detail(self, raw_feature_dict):
        raw = float(np.dot(self.theta, self.featurize(raw_feature_dict)))
        clipped = float(np.clip(raw, -self.clip_value, self.clip_value))
        return {
            "raw": raw,
            "clipped": clipped,
            "clip_hit": bool(abs(raw - clipped) > 1e-9),
        }

    def update_td_detail(self, f_t, cost_t, f_next, alpha=1e-3,
                         terminal=False, td_error_clip=None,
                         theta_delta_norm_max=None):
        x_t = self.featurize(f_t)
        x_next = self.featurize(f_next)
        if (not np.all(np.isfinite(x_t)) or not np.all(np.isfinite(x_next)) or
                not np.all(np.isfinite(self.theta))):
            return {"updated": False, "reason": "nonfinite_input"}
        theta_before = self.theta.copy()
        pred = float(np.dot(self.theta, x_t))
        target = float(cost_t)
        if not terminal:
            target += self.gamma * float(np.dot(self.theta, x_next))
        raw_delta = float(target - pred)
        delta = raw_delta
        if td_error_clip is not None:
            clip = abs(float(td_error_clip))
            if clip > 0.0:
                delta = float(np.clip(delta, -clip, clip))
        theta_delta = float(alpha) * delta * x_t
        theta_delta_norm = float(np.linalg.norm(theta_delta))
        if theta_delta_norm_max is not None:
            max_norm = abs(float(theta_delta_norm_max))
            if max_norm > 0.0 and theta_delta_norm > max_norm:
                theta_delta *= max_norm / theta_delta_norm
                theta_delta_norm = max_norm
        candidate = theta_before + theta_delta
        if not np.all(np.isfinite(candidate)):
            self.theta = theta_before
            return {"updated": False, "reason": "nonfinite_theta"}
        self.theta = candidate
        return {
            "updated": True,
            "td_error": float(delta),
            "raw_td_error": float(raw_delta),
            "target": float(target),
            "prediction": float(pred),
            "theta_delta_norm": float(theta_delta_norm),
            "terminal": bool(terminal),
        }

    def update_td(self, f_t, cost_t, f_next, alpha=1e-3,
                  terminal=False, td_error_clip=None,
                  theta_delta_norm_max=None):
        detail = self.update_td_detail(
            f_t, cost_t, f_next, alpha=alpha, terminal=terminal,
            td_error_clip=td_error_clip,
            theta_delta_norm_max=theta_delta_norm_max)
        return detail.get("td_error", 0.0)

    def fit_lstsq(self, feature_matrix, returns, ridge=1e-4):
        x = np.asarray(feature_matrix, float)
        y = np.asarray(returns, float)
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("empty feature matrix")
        self.mean = np.mean(x, axis=0)
        self.std = _safe_std(np.std(x, axis=0))
        for i, name in enumerate(self.feature_names):
            if name == "bias":
                self.mean[i] = 0.0
                self.std[i] = 1.0
        xn = (x - self.mean[None, :]) / self.std[None, :]
        reg = float(ridge) * np.eye(xn.shape[1])
        self.theta = np.linalg.solve(np.dot(xn.T, xn) + reg, np.dot(xn.T, y))
        pred = np.dot(xn, self.theta)
        loss = float(np.mean((pred - y) ** 2))
        return loss

    def to_dict(self):
        return {
            "critic_version": self.critic_version,
            "gamma": self.gamma,
            "clip_value": self.clip_value,
            "feature_names": list(self.feature_names),
            "mean": [float(x) for x in self.mean],
            "std": [float(x) for x in self.std],
            "theta": [float(x) for x in self.theta],
            "cost_weights": dict(self.cost_weights),
            "metadata": dict(self.metadata),
            "learning": dict(self.learning_config),
        }

    def fingerprint(self):
        payload = yaml.safe_dump(self.to_dict(), default_flow_style=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def save_yaml(self, path):
        out_dir = os.path.dirname(path)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def load_yaml(cls, path):
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls(
            feature_names=data.get("feature_names", DEFAULT_FEATURE_NAMES),
            theta=data.get("theta"),
            mean=data.get("mean"),
            std=data.get("std"),
            gamma=data.get("gamma", 0.95),
            clip_value=data.get("clip_value", 100.0),
            cost_weights=data.get("cost_weights"),
            critic_version=data.get("critic_version", "linear_adp_v1"),
            metadata=data.get("metadata", {}),
            learning_config=data.get("learning", {}))


def clone_critic(critic):
    """Freeze a critic for decision-time ranking while learning continues live."""
    if critic is None:
        return None
    return ADPCritic(
        feature_names=list(critic.feature_names), theta=np.asarray(critic.theta, float).copy(),
        mean=np.asarray(critic.mean, float).copy(),
        std=np.asarray(critic.std, float).copy(), gamma=critic.gamma,
        clip_value=critic.clip_value, cost_weights=dict(critic.cost_weights),
        critic_version=critic.critic_version, metadata=dict(critic.metadata),
        learning_config=dict(critic.learning_config))


def adp_ranking_adjustments(raw_values, metadata=None, lambda_adp=0.0,
                            normalization="robust", norm_clip=3.0,
                            contribution_clip=0.10):
    """Return bounded ranking terms for calibrated cost-to-go values.

    The calibrated target mean/p95 establishes the default scale.  A robust
    candidate-pool scale is used only when that calibration metadata is absent.
    """
    values = np.asarray(list(raw_values or []), float)
    if values.size == 0:
        return [], {"source": "empty", "center": 0.0, "scale": 1.0}
    values = np.where(np.isfinite(values), values, 0.0)
    metadata = dict(metadata or {})
    center = _as_float(metadata.get(
        "ranking_value_center", metadata.get("target_mean")), np.nan)
    scale = _as_float(metadata.get("ranking_value_scale"), 0.0)
    if abs(scale) <= 1e-12:
        p95 = _as_float(metadata.get("target_p95"), np.nan)
        scale = (abs(p95 - center)
                 if np.isfinite(center) and np.isfinite(p95) else 0.0)
    source = ("metadata_ranking_value_distribution"
              if "ranking_value_center" in metadata else
              "metadata_target_mean_p95")
    if not np.isfinite(center) or scale < 1e-6:
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        spread = float(np.percentile(values, 95) - center)
        scale = max(1.4826 * mad, abs(spread), 1e-6)
        source = "candidate_pool_robust_fallback"
    if str(normalization).strip().lower() not in ("robust", "calibrated"):
        center, scale, source = 0.0, 1.0, "identity"
    limit = max(abs(_as_float(norm_clip, 3.0)), 0.0)
    contribution_limit = max(abs(_as_float(contribution_clip, 0.10)), 0.0)
    lam = _as_float(lambda_adp)
    normalized = (values - center) / max(scale, 1e-6)
    if limit > 0.0:
        normalized = np.clip(normalized, -limit, limit)
    contributions = lam * normalized
    if contribution_limit > 0.0:
        contributions = np.clip(contributions, -contribution_limit,
                                contribution_limit)
    return [
        {"adp_value_raw": float(raw), "adp_value_normalized": float(norm),
         "effective_lambda_adp": float(lam), "adp_cost": float(cost)}
        for raw, norm, cost in zip(values, normalized, contributions)
    ], {"source": source, "center": float(center), "scale": float(scale)}


class ADPFeatureBuilder(object):
    def __init__(self, feature_names=None):
        self.feature_names = list(feature_names or DEFAULT_FEATURE_NAMES)

    def build_wheelchair(self, pose2d, goal, field, gate_info=None,
                         interest_risk=None, corridor=None, u=None,
                         prev_pose2d=None, candidate_features=None):
        pose = np.asarray(pose2d, float)
        goal = np.asarray(goal, float)
        z = np.array([pose[0], pose[1], 0.0], float)
        comp = field.risk_components(z)
        speed = 0.0
        if u is not None and len(u):
            speed = abs(float(u[0]))
        d_goal = float(np.linalg.norm(pose[:2] - goal[:2]))
        progress = 0.0
        if prev_pose2d is not None:
            prev = np.asarray(prev_pose2d, float)
            progress = float(np.linalg.norm(prev[:2] - goal[:2]) - d_goal)
        d_corridor = 0.0
        if corridor is not None:
            _, d_corridor = corridor.project(z)
        raw = self._common(comp, gate_info, interest_risk)
        raw.update({
            "d_goal": d_goal,
            "progress": progress,
            "speed": speed,
            "d_corridor": float(d_corridor),
            "phase_norm": 0.0,
        })
        raw.update(candidate_feature_values(candidate_features)[0])
        return raw

    def build_arm(self, ee_pos, target_pos, field, gate_info=None,
                  interest_risk=None, phase=None, u=None, prev_ee_pos=None,
                  candidate_features=None):
        ee = np.asarray(ee_pos, float)
        target = np.asarray(target_pos, float)
        comp = field.risk_components(ee)
        speed = 0.0
        if u is not None and len(u):
            speed = float(np.linalg.norm(np.asarray(u, float)))
        d_goal = float(np.linalg.norm(ee - target))
        progress = 0.0
        if prev_ee_pos is not None:
            prev = np.asarray(prev_ee_pos, float)
            progress = float(np.linalg.norm(prev - target) - d_goal)
        phase_norm = 0.0
        if phase is not None and phase != "":
            phase_norm = max(0.0, min(1.0, _as_float(phase) / 3.0))
        raw = self._common(comp, gate_info, interest_risk)
        raw.update({
            "d_goal": d_goal,
            "progress": progress,
            "speed": speed,
            "d_corridor": 0.0,
            "phase_norm": phase_norm,
        })
        raw.update(candidate_feature_values(candidate_features)[0])
        return raw

    def from_row(self, row, prev_row=None):
        target = (row.get("target") or "").strip().lower()
        x = _as_float(row.get("x"))
        y = _as_float(row.get("y"))
        z = _as_float(row.get("z"))
        if target == "arm":
            goal = np.array([0.42, 0.0, 0.21], float)
            d_goal = float(np.linalg.norm(np.array([x, y, z], float) - goal))
            phase_norm = max(0.0, min(1.0, _as_float(row.get("phase")) / 3.0))
            ip_phi_max = _as_float(row.get("phi_arm_max_point"))
            ip_phi_mean = _as_float(row.get("phi_arm_mean_point"))
        else:
            goal = np.array([-0.55, 0.55], float)
            d_goal = float(np.linalg.norm(np.array([x, y], float) - goal))
            phase_norm = 0.0
            ip_phi_max = _as_float(row.get("phi_max_point"))
            ip_phi_mean = _as_float(row.get("phi_mean_point"))
        progress = 0.0
        if prev_row is not None:
            px = _as_float(prev_row.get("x"))
            py = _as_float(prev_row.get("y"))
            pz = _as_float(prev_row.get("z"))
            if target == "arm":
                prev_d = float(np.linalg.norm(np.array([px, py, pz], float) - goal))
            else:
                prev_d = float(np.linalg.norm(np.array([px, py], float) - goal))
            progress = prev_d - d_goal
        gate_state = (row.get("gate_state") or "").strip().upper()
        return {
            "bias": 1.0,
            "phi_total": _as_float(row.get("phi_total")),
            "phi_prox": _as_float(row.get("phi_prox")),
            "phi_close": _as_float(row.get("phi_close")),
            "phi_body": _as_float(row.get("phi_body")),
            "phi_env": _as_float(row.get("phi_env")),
            "risk_exceed": 1.0 if _as_float(row.get("phi_total")) > _as_float(row.get("rho_warn"), 1.6) else 0.0,
            "gate_slow": 1.0 if gate_state == "SLOW" else 0.0,
            "gate_stop": 1.0 if _as_float(row.get("gate_stop")) >= 0.5 or gate_state == "STOP" else 0.0,
            "d_goal": d_goal,
            "progress": progress,
            "speed": _as_float(row.get("speed_filtered"), _as_float(row.get("speed_raw"))),
            "ip_phi_max": ip_phi_max,
            "ip_phi_mean": ip_phi_mean,
            "d_corridor": _as_float(row.get("d_corridor")),
            "phase_norm": phase_norm,
        }

    def _common(self, comp, gate_info=None, interest_risk=None):
        gate_info = gate_info or {}
        interest_risk = interest_risk or {}
        phi_total = float(comp.get("phi_total", 0.0))
        rho_warn = _as_float(gate_info.get("rho_warn"), 1.6)
        gate_state = (gate_info.get("state") or "").upper()
        return {
            "bias": 1.0,
            "phi_total": phi_total,
            "phi_prox": float(comp.get("phi_prox", 0.0)),
            "phi_close": float(comp.get("phi_close", 0.0)),
            "phi_body": float(comp.get("phi_body", 0.0)),
            "phi_env": float(comp.get("phi_env", 0.0)),
            "risk_exceed": 1.0 if phi_total > rho_warn else 0.0,
            "gate_slow": 1.0 if gate_state == "SLOW" else 0.0,
            "gate_stop": 1.0 if bool(gate_info.get("stop", False)) else 0.0,
            "ip_phi_max": _as_float(interest_risk.get("phi_max")),
            "ip_phi_mean": _as_float(interest_risk.get("phi_mean")),
        }


def stage_cost(features, weights=None):
    w = dict(DEFAULT_COST_WEIGHTS)
    if weights:
        w.update(weights)
    f = features or {}
    return (
        w["w_phi"] * _as_float(f.get("phi_total")) +
        w["w_ip"] * _as_float(f.get("ip_phi_max")) +
        w["w_exceed"] * _as_float(f.get("risk_exceed")) +
        w["w_stop"] * _as_float(f.get("gate_stop")) +
        w["w_corr"] * (_as_float(f.get("d_corridor")) ** 2) +
        w["w_u"] * (_as_float(f.get("speed")) ** 2) -
        w["w_prog"] * max(_as_float(f.get("progress")), 0.0)
    )


def transition_stage_cost(previous_features, current_features,
                          control_effort=0.0, task_penalty=0.0,
                          tube_violation=0.0, failure_penalty=0.0,
                          risk_scale=2.0, weights=None):
    """Stable, executed-transition cost for online shadow learning."""
    w = dict(DEFAULT_TRANSITION_COST_WEIGHTS)
    if weights:
        w.update(weights)
    prev = previous_features or {}
    curr = current_features or {}
    scale = max(abs(_as_float(risk_scale, 2.0)), 1e-6)
    risk = max(0.0, _as_float(curr.get("phi_total"))) / scale
    progress_penalty = max(
        0.0, _as_float(curr.get("d_goal")) - _as_float(prev.get("d_goal")))
    components = {
        "risk": float(min(risk, 10.0)),
        "progress": float(min(progress_penalty, 10.0)),
        "control": float(min(max(_as_float(control_effort), 0.0), 10.0)),
        "task": float(min(max(_as_float(task_penalty), 0.0), 10.0)),
        "tube": float(min(max(_as_float(tube_violation), 0.0), 10.0)),
        "failure": float(min(max(_as_float(failure_penalty), 0.0), 10.0)),
    }
    total = sum(float(w[key]) * components[key] for key in components)
    return float(total), components


class ADPTransitionLearner(object):
    """Records only measured state transitions and keeps TD updates bounded."""
    def __init__(self, critic, config=None, robot=""):
        self.critic = critic
        self.config = dict(DEFAULT_LEARNING_CONFIG)
        if config:
            self.config.update(config)
        self.robot = str(robot)
        self.previous_features = None
        self.previous_time = None
        self.records = []
        self.transition_count = 0
        self.update_count = 0
        self.skipped_transition_count = 0
        self.td_errors = []
        self.theta_delta_norm_total = 0.0
        self.theta_initial = np.asarray(critic.theta, float).copy()

    def observe(self, features, timestamp, task_state="", corridor_id="",
                control_effort=0.0, task_penalty=0.0, tube_violation=0.0,
                terminal=False, success=False, failure_reason=""):
        current = dict(features or {})
        now = _as_float(timestamp)
        record = {
            "robot": self.robot,
            "task_state": str(task_state),
            "corridor_id": str(corridor_id),
            "timestamp": float(now),
            "terminal": bool(terminal),
            "success": bool(success),
            "failure_reason": str(failure_reason or ""),
        }
        if not bool(self.config.get("enabled", True)):
            record["status"] = "learning_disabled"
            self.records.append(record)
            return record
        x = self.critic.featurize(current)
        if not np.all(np.isfinite(x)):
            self.skipped_transition_count += 1
            record.update({"status": "skipped", "reason": "nonfinite_features"})
            self.records.append(record)
            return record
        if self.previous_features is None:
            self.previous_features = current
            self.previous_time = now
            record["status"] = "seed"
            self.records.append(record)
            return record
        dt = float(now - _as_float(self.previous_time))
        min_dt = max(0.0, _as_float(self.config.get("min_transition_dt"), 0.1))
        if not terminal and dt < min_dt:
            self.skipped_transition_count += 1
            record.update({"status": "skipped", "reason": "min_transition_dt", "dt": dt})
            self.records.append(record)
            return record
        failure_penalty = 0.0
        if terminal and not success:
            failure_penalty = _as_float(
                self.config.get("failure_terminal_penalty"), 3.0)
        cost, components = transition_stage_cost(
            self.previous_features, current, control_effort=control_effort,
            task_penalty=task_penalty, tube_violation=tube_violation,
            failure_penalty=failure_penalty,
            risk_scale=self.config.get("risk_scale", 2.0),
            weights=self.config.get("stage_cost_weights"))
        value_t_detail = self.critic.predict_detail(self.previous_features)
        value_next_detail = self.critic.predict_detail(current)
        value_t = float(value_t_detail["raw"])
        value_next = float(value_next_detail["raw"])
        features_t = dict(self.previous_features)
        detail = self.critic.update_td_detail(
            self.previous_features, cost, current,
            alpha=_as_float(self.config.get("alpha"), 0.001),
            terminal=terminal,
            td_error_clip=self.config.get("td_error_clip", 5.0),
            theta_delta_norm_max=self.config.get("theta_delta_norm_max", 0.05))
        self.transition_count += 1
        record.update({
            "dt": dt,
            "stage_cost": float(cost),
            "stage_cost_components": components,
            "features_t": features_t,
            "features_next": current,
            "value_t": value_t,
            "value_next": value_next,
            "value_t_clipped": float(value_t_detail["clipped"]),
            "value_next_clipped": float(value_next_detail["clipped"]),
            "bootstrap_value": (
                0.0 if terminal else float(self.critic.gamma * value_next)),
        })
        record.update(detail)
        if detail.get("updated", False):
            self.update_count += 1
            self.theta_delta_norm_total += float(detail.get("theta_delta_norm", 0.0))
            self.td_errors.append(float(detail.get("td_error", 0.0)))
            record["status"] = "updated"
        else:
            self.skipped_transition_count += 1
            record["status"] = "skipped"
        self.records.append(record)
        self.previous_features = None if terminal else current
        self.previous_time = None if terminal else now
        return record

    def diagnostics(self, max_records=200):
        errors = np.asarray(self.td_errors, float)
        theta_changed = bool(not np.allclose(self.theta_initial, self.critic.theta))
        updated = [record for record in self.records
                   if record.get("updated", False)]
        def stats(name):
            values = np.asarray([_as_float(record.get(name))
                                 for record in updated], float)
            if not len(values):
                return {"mean": 0.0, "std": 0.0, "p50": 0.0,
                        "p95": 0.0, "max": 0.0, "min": 0.0}
            return {
                "mean": float(np.mean(values)), "std": float(np.std(values)),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "max": float(np.max(values)), "min": float(np.min(values)),
            }
        raw_errors = np.asarray([
            _as_float(record.get("raw_td_error")) for record in updated], float)
        terminal_stats = {}
        for name, predicate in (("success", lambda row: bool(row.get("success"))),
                                ("failure", lambda row: not bool(row.get("success")))):
            values = np.asarray([
                _as_float(row.get("raw_td_error")) for row in updated
                if row.get("terminal", False) and predicate(row)], float)
            terminal_stats[name] = {
                "count": int(len(values)),
                "raw_td_error_mean": float(np.mean(values)) if len(values) else 0.0,
                "raw_td_error_max_abs": float(np.max(np.abs(values))) if len(values) else 0.0,
            }
        return {
            "robot": self.robot,
            "learning_enabled": bool(self.config.get("enabled", True)),
            "decision_influence_enabled": bool(
                self.config.get("decision_influence_enabled", False)),
            "transition_count": int(self.transition_count),
            "update_count": int(self.update_count),
            "skipped_transition_count": int(self.skipped_transition_count),
            "td_error_mean": float(np.mean(errors)) if len(errors) else 0.0,
            "td_error_abs_mean": float(np.mean(np.abs(errors))) if len(errors) else 0.0,
            "td_error_max_abs": float(np.max(np.abs(errors))) if len(errors) else 0.0,
            "td_clip_count": int(np.sum(np.abs(raw_errors) >= abs(_as_float(
                self.config.get("td_error_clip"), 5.0)))) if len(raw_errors) else 0,
            "td_clip_ratio": float(np.mean(np.abs(raw_errors) >= abs(_as_float(
                self.config.get("td_error_clip"), 5.0)))) if len(raw_errors) else 0.0,
            "td_scale_audit": {
                "value_t": stats("value_t"),
                "stage_cost": stats("stage_cost"),
                "bootstrap_value": stats("bootstrap_value"),
                "td_target": stats("target"),
                "raw_td_error": stats("raw_td_error"),
                "clipped_td_error": stats("td_error"),
                "terminal": terminal_stats,
            },
            "theta_delta_norm_total": float(self.theta_delta_norm_total),
            "theta_changed": theta_changed,
            "critic_version": str(self.critic.critic_version),
            "critic_fingerprint": self.critic.fingerprint(),
            "records": list(self.records[-int(max_records):]),
        }


def save_and_verify_critic(critic, path):
    critic.save_yaml(path)
    reloaded = ADPCritic.load_yaml(path)
    return bool(np.allclose(critic.theta, reloaded.theta))


def adp_role_from_runtime(adp_enabled, learning_enabled,
                          decision_influence_enabled,
                          effective_lambda=0.0,
                          ranking_contribution=False,
                          control_contribution=False):
    """Classify ADP from explicit runtime state, never generic DLS deltas."""
    if not bool(adp_enabled) or not bool(learning_enabled):
        return "inactive"
    if (not bool(decision_influence_enabled) or
            abs(_as_float(effective_lambda)) <= 1e-12):
        return "shadow_learning"
    if bool(ranking_contribution) and bool(control_contribution):
        return "ranking_and_control_modifier"
    if bool(ranking_contribution):
        return "ranking_modifier"
    if bool(control_contribution):
        return "control_modifier"
    return "evaluation_only"


def fit_critic_from_transition_records(records, template_critic,
                                       gamma=None, ridge=1e-4):
    """Fit a fresh critic to measured online costs-to-go, split at terminals."""
    episodes = []
    current = []
    for record in records or []:
        if not record.get("updated", False):
            continue
        features = record.get("features_t")
        if not isinstance(features, dict):
            continue
        cost = _as_float(record.get("stage_cost"))
        current.append((dict(features), cost))
        if bool(record.get("terminal", False)):
            episodes.append(current)
            current = []
    if current:
        episodes.append(current)
    if not episodes:
        raise ValueError("no valid transition records")
    feature_names = list(template_critic.feature_names)
    x_all = []
    target_all = []
    gamma = float(template_critic.gamma if gamma is None else gamma)
    for episode in episodes:
        costs = [item[1] for item in episode]
        returns = discounted_returns(costs, gamma=gamma)
        for (features, _cost), value in zip(episode, returns):
            x_all.append([features.get(name, 0.0) for name in feature_names])
            target_all.append(float(value))
    critic = ADPCritic(
        feature_names=feature_names, gamma=gamma,
        clip_value=template_critic.clip_value,
        cost_weights=template_critic.cost_weights,
        critic_version=str(template_critic.critic_version) + "_calibrated",
        learning_config=template_critic.learning_config)
    targets = np.asarray(target_all, float)
    loss = critic.fit_lstsq(np.asarray(x_all, float), targets, ridge=ridge)
    return critic, {
        "sample_count": int(len(targets)),
        "episode_count": int(len(episodes)),
        "fit_loss": float(loss),
        "target_min": float(np.min(targets)),
        "target_mean": float(np.mean(targets)),
        "target_p95": float(np.percentile(targets, 95)),
        "target_max": float(np.max(targets)),
    }


def discounted_returns(costs, gamma=0.97, clip_value=None):
    out = np.zeros(len(costs), float)
    acc = 0.0
    for i in range(len(costs) - 1, -1, -1):
        acc = float(costs[i]) + float(gamma) * acc
        if clip_value is not None:
            acc = float(np.clip(acc, -float(clip_value), float(clip_value)))
        out[i] = acc
    return out
