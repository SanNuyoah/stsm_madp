import sys
sys.dont_write_bytecode = True

import numpy as np

EPS = 1e-6

class HumanState:

    def __init__(self, pos, vel=None, heading=0.0, posture="sitting",
                 vulnerability=1.0, body_parts=None):
        self.pos = np.asarray(pos, dtype=float)
        self.vel = np.zeros_like(self.pos) if vel is None else np.asarray(vel, float)
        try:
            parsed_heading = float(heading)
        except (TypeError, ValueError):
            parsed_heading = float("nan")
        self.heading = parsed_heading if np.isfinite(parsed_heading) else None
        self.posture = posture
        self.vulnerability = float(vulnerability)

        self.body_parts = body_parts if body_parts is not None else {}

    def social_radii(self):
        base_a, base_b = 0.45, 0.35
        scale = self.vulnerability
        if self.posture in ("standing_unstable", "transferring"):
            scale *= 1.3
        return base_a * scale, base_b * scale

class SemanticAnchor:

    def __init__(self, anchor_type, center, half_extent, weight=1.0,
                 forbidden=False):
        self.type = anchor_type
        self.center = np.asarray(center, float)
        self.half_extent = np.asarray(half_extent, float)
        self.weight = float(weight)
        self.forbidden = forbidden

    def signed_distance(self, z):
        d = np.abs(z[:self.center.shape[0]] - self.center) - self.half_extent
        outside = np.linalg.norm(np.maximum(d, 0.0))
        inside = min(np.max(d), 0.0)
        return outside + inside

DEFAULT_TASK_MULTIPLIERS = {
    "moving": {"prox": 1.00, "close": 1.00, "dir": 0.80,
               "body": 1.00, "env": 1.00},
    "avoiding": {"prox": 1.30, "close": 1.60, "dir": 1.20,
                 "body": 1.00, "env": 1.20},
    "passing": {"prox": 1.40, "close": 1.40, "dir": 1.50,
                "body": 1.10, "env": 1.30},
    "arriving": {"prox": 1.00, "close": 0.80, "dir": 0.60,
                 "body": 1.00, "env": 1.10},
}


class SocialFieldParams:

    def __init__(self, lam_prox=1.0, lam_close=1.2, lam_dir=0.6,
                 lam_body=2.0, lam_env=1.5, sigma_env=0.25,
                 direction_model="legacy_piecewise",
                 task_aware_enabled=False, task_multipliers=None,
                 direction_base=0.4, direction_front=1.1,
                 direction_rear=0.1):
        self.lam_prox = lam_prox
        self.lam_close = lam_close
        self.lam_dir = lam_dir
        self.lam_body = lam_body
        self.lam_env = lam_env
        self.sigma_env = sigma_env
        self.direction_model = str(direction_model or "legacy_piecewise")
        self.task_aware_enabled = bool(task_aware_enabled)
        self.task_multipliers = dict(task_multipliers or {})
        self.direction_base = float(direction_base)
        self.direction_front = float(direction_front)
        self.direction_rear = float(direction_rear)

class SocialField:
    def __init__(self, params=None):
        self.params = params or SocialFieldParams()
        self.humans = []
        self.anchors = []
        self.task_context = {}
        self._effective_weights = self._compute_effective_weights()

    def set_scene(self, humans, anchors):
        self.humans = list(humans)
        self.anchors = list(anchors)

    def set_task_context(self, task_context):
        self.task_context = dict(task_context or {})
        self._effective_weights = self._compute_effective_weights()

    def get_effective_weights(self):
        return dict(self._effective_weights)

    def _compute_effective_weights(self):
        p = self.params
        base = {
            "lam_prox": float(p.lam_prox), "lam_close": float(p.lam_close),
            "lam_dir": float(p.lam_dir), "lam_body": float(p.lam_body),
            "lam_env": float(p.lam_env),
        }
        if not bool(p.task_aware_enabled):
            return base
        state = str(self.task_context.get("task_state", "") or "").lower()
        multipliers = dict(DEFAULT_TASK_MULTIPLIERS)
        multipliers.update(dict(p.task_multipliers or {}))
        state_multipliers = dict(multipliers.get(state, {}) or {})
        return {
            "lam_prox": base["lam_prox"] * float(state_multipliers.get("prox", 1.0)),
            "lam_close": base["lam_close"] * float(state_multipliers.get("close", 1.0)),
            "lam_dir": base["lam_dir"] * float(state_multipliers.get("dir", 1.0)),
            "lam_body": base["lam_body"] * float(state_multipliers.get("body", 1.0)),
            "lam_env": base["lam_env"] * float(state_multipliers.get("env", 1.0)),
        }

    @staticmethod
    def _heading_or_none(human):
        try:
            heading = float(human.heading)
        except (TypeError, ValueError):
            return None
        return heading if np.isfinite(heading) else None

    def phi_prox(self, z, h):
        r2 = z[:2] - h.pos[:2]
        a, b = h.social_radii()
        heading = self._heading_or_none(h)
        c, s = np.cos(heading or 0.0), np.sin(heading or 0.0)
        R = np.array([[c, -s], [s, c]])
        local = np.dot(R.T, r2)
        m = (local[0] / a) ** 2 + (local[1] / b) ** 2
        return np.exp(-0.5 * m)

    def phi_close(self, z, zdot, h):
        r = z[:2] - h.pos[:2]
        vrel = zdot[:2] - h.vel[:2]
        rn = np.linalg.norm(r) + EPS
        sc = max(0.0, -float(np.dot(r, vrel)) / rn)
        return sc * sc / rn

    def phi_dir(self, z, h):
        r2 = z[:2] - h.pos[:2]
        heading = self._heading_or_none(h)
        if heading is None:
            return 1.0
        c, s = np.cos(heading), np.sin(heading)
        R = np.array([[c, -s], [s, c]])
        local = np.dot(R.T, r2)
        alpha = float(np.arctan2(local[1], local[0]))
        if str(self.params.direction_model).lower() == "continuous":
            front_factor = 0.5 * (1.0 + np.cos(alpha))
            rear_factor = 1.0 - front_factor
            value = (self.params.direction_base +
                     self.params.direction_front * front_factor +
                     self.params.direction_rear * rear_factor)
            return float(max(0.0, value)) if np.isfinite(value) else 1.0
        a = abs(alpha)
        if a < np.pi / 3:
            return 0.8
        if a > 2 * np.pi / 3:
            return 1.5
        return 0.4

    def phi_body(self, z, h):
        total = 0.0
        for _, (p, w, sigma) in h.body_parts.items():
            d2 = float(np.sum((z[:p.shape[0]] - p) ** 2))
            total += w * np.exp(-d2 / (2.0 * sigma * sigma))
        return total

    def phi_env(self, z):
        total = 0.0
        for a in self.anchors:
            d = a.signed_distance(z)
            if a.forbidden and d <= 0.0:
                total += a.weight
            else:
                total += a.weight * np.exp(-(max(d, 0.0) ** 2) /
                                           (2.0 * self.params.sigma_env ** 2))
        return total

    def phi_s(self, z, zdot=None):
        z = np.asarray(z, float)
        zdot = np.zeros_like(z) if zdot is None else np.asarray(zdot, float)
        p = self.get_effective_weights()
        val = 0.0
        for h in self.humans:
            val += p["lam_prox"] * self.phi_prox(z, h)
            val += p["lam_close"] * self.phi_close(z, zdot, h)
            val += p["lam_dir"] * self.phi_dir(z, h) * self.phi_prox(z, h)
            val += p["lam_body"] * self.phi_body(z, h)
        val += p["lam_env"] * self.phi_env(z)
        return val

    def phi_s_batch(self, points, velocities=None):
        """Vectorized equivalent of ``phi_s`` for a collection of points."""
        z = np.asarray(points, float)
        if z.ndim == 1:
            z = z.reshape((1, z.shape[0]))
        if z.size == 0:
            return np.zeros(0, float)
        if velocities is None:
            zdot = np.zeros_like(z)
        else:
            zdot = np.asarray(velocities, float)
            if zdot.ndim == 1:
                zdot = np.tile(zdot, (len(z), 1))
        p = self.get_effective_weights()
        values = np.zeros(len(z), float)
        for human in self.humans:
            relative = z[:, :2] - human.pos[:2]
            a, b = human.social_radii()
            heading = self._heading_or_none(human)
            c, s = np.cos(heading or 0.0), np.sin(heading or 0.0)
            rotation = np.array([[c, -s], [s, c]])
            local = np.dot(relative, rotation)
            prox = np.exp(-0.5 * (
                (local[:, 0] / a) ** 2 + (local[:, 1] / b) ** 2))
            relative_velocity = zdot[:, :2] - human.vel[:2]
            relative_norm = np.linalg.norm(relative, axis=1) + EPS
            closing = np.maximum(
                0.0,
                -np.einsum(
                    "ij,ij->i", relative, relative_velocity) / relative_norm)
            if heading is None:
                directional = np.ones(len(z), float)
            elif str(self.params.direction_model).lower() == "continuous":
                alpha = np.arctan2(local[:, 1], local[:, 0])
                front_factor = 0.5 * (1.0 + np.cos(alpha))
                directional = np.maximum(
                    0.0, self.params.direction_base +
                    self.params.direction_front * front_factor +
                    self.params.direction_rear * (1.0 - front_factor))
            else:
                alpha = np.abs(np.arctan2(local[:, 1], local[:, 0]))
                directional = np.where(
                    alpha < np.pi / 3.0, 0.8,
                    np.where(alpha > 2.0 * np.pi / 3.0, 1.5, 0.4))
            body = np.zeros(len(z), float)
            for _name, (part, weight, sigma) in human.body_parts.items():
                dim = part.shape[0]
                squared_distance = np.sum(
                    (z[:, :dim] - part) ** 2, axis=1)
                body += weight * np.exp(
                    -squared_distance / (2.0 * sigma * sigma))
            values += p["lam_prox"] * prox
            values += p["lam_close"] * closing ** 2 / relative_norm
            values += p["lam_dir"] * directional * prox
            values += p["lam_body"] * body
        for anchor in self.anchors:
            dim = anchor.center.shape[0]
            delta = np.abs(z[:, :dim] - anchor.center) - anchor.half_extent
            outside = np.linalg.norm(np.maximum(delta, 0.0), axis=1)
            inside = np.minimum(np.max(delta, axis=1), 0.0)
            distance = outside + inside
            contribution = anchor.weight * np.exp(
                -(np.maximum(distance, 0.0) ** 2) /
                (2.0 * self.params.sigma_env ** 2))
            if anchor.forbidden:
                contribution = np.where(
                    distance <= 0.0, anchor.weight, contribution)
            values += p["lam_env"] * contribution
        return values

    def risk_components(self, z, zdot=None):
        z = np.asarray(z, float)
        zdot = np.zeros_like(z) if zdot is None else np.asarray(zdot, float)
        p = self.get_effective_weights()
        comp = {
            "phi_prox": 0.0,
            "phi_close": 0.0,
            "phi_dir": 0.0,
            "phi_body": 0.0,
            "phi_env": 0.0,
        }
        for h in self.humans:
            prox = self.phi_prox(z, h)
            comp["phi_prox"] += p["lam_prox"] * prox
            comp["phi_close"] += p["lam_close"] * self.phi_close(z, zdot, h)
            comp["phi_dir"] += p["lam_dir"] * self.phi_dir(z, h) * prox
            comp["phi_body"] += p["lam_body"] * self.phi_body(z, h)
        comp["phi_env"] += p["lam_env"] * self.phi_env(z)
        comp["phi_total"] = sum(comp.values())
        return comp

    def phi_close_monitor(self, z, zdot):
        z = np.asarray(z, float)
        zdot = np.asarray(zdot, float)
        return float(sum(self.phi_close(z, zdot, h) for h in self.humans))

    def phi_robot(self, points, vels=None, weights=None):
        n = len(points)
        vels = [None] * n if vels is None else vels
        weights = [1.0] * n if weights is None else weights
        return float(sum(w * self.phi_s(z, v)
                         for z, v, w in zip(points, vels, weights)))

    def grad_phi_s(self, z, zdot=None, eps=1e-4):
        z = np.asarray(z, float)
        g = np.zeros_like(z)
        for i in range(z.shape[0]):
            dz = np.zeros_like(z)
            dz[i] = eps
            g[i] = (self.phi_s(z + dz, zdot) - self.phi_s(z - dz, zdot)) / (2 * eps)
        return g
