import os
import sys
sys.dont_write_bytecode = True

import numpy as np
import yaml


DEFAULT_FEATURE_NAMES = [
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

DEFAULT_COST_WEIGHTS = {
    "w_phi": 1.0,
    "w_ip": 0.6,
    "w_exceed": 3.0,
    "w_stop": 20.0,
    "w_corr": 0.5,
    "w_u": 0.05,
    "w_prog": 0.4,
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


class ADPCritic(object):
    def __init__(self, feature_names=None, theta=None, mean=None, std=None,
                 gamma=0.97, clip_value=100.0, cost_weights=None,
                 critic_version="linear_adp_v1", metadata=None):
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
        if self.theta.shape[0] != n:
            self.theta = np.resize(self.theta, n)
        if self.mean.shape[0] != n:
            self.mean = np.resize(self.mean, n)
        if self.std.shape[0] != n:
            self.std = _safe_std(np.resize(self.std, n))

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

    def update_td(self, f_t, cost_t, f_next, alpha=1e-3):
        x_t = self.featurize(f_t)
        x_next = self.featurize(f_next)
        target = float(cost_t) + self.gamma * float(np.dot(self.theta, x_next))
        pred = float(np.dot(self.theta, x_t))
        delta = target - pred
        self.theta = self.theta + float(alpha) * delta * x_t
        return delta

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
        }

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
            gamma=data.get("gamma", 0.97),
            clip_value=data.get("clip_value", 100.0),
            cost_weights=data.get("cost_weights"),
            critic_version=data.get("critic_version", "linear_adp_v1"),
            metadata=data.get("metadata", {}))


class ADPFeatureBuilder(object):
    def __init__(self, feature_names=None):
        self.feature_names = list(feature_names or DEFAULT_FEATURE_NAMES)

    def build_wheelchair(self, pose2d, goal, field, gate_info=None,
                         interest_risk=None, corridor=None, u=None,
                         prev_pose2d=None):
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
        return raw

    def build_arm(self, ee_pos, target_pos, field, gate_info=None,
                  interest_risk=None, phase=None, u=None, prev_ee_pos=None):
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


def discounted_returns(costs, gamma=0.97, clip_value=None):
    out = np.zeros(len(costs), float)
    acc = 0.0
    for i in range(len(costs) - 1, -1, -1):
        acc = float(costs[i]) + float(gamma) * acc
        if clip_value is not None:
            acc = float(np.clip(acc, -float(clip_value), float(clip_value)))
        out[i] = acc
    return out
