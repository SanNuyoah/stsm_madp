import sys
sys.dont_write_bytecode = True

import numpy as np

class Corridor:

    def __init__(self, waypoints, radius, label="", cost=0.0):
        self.waypoints = np.asarray(waypoints, float)
        self.radius = float(radius)
        self.label = label
        self.cost = float(cost)
        self.base_cost = float(cost)
        self.adp_cost = 0.0
        self.adp_mean = 0.0
        self.adp_max = 0.0
        self.adp_end = 0.0
        self.adp_raw_mean = 0.0
        self.adp_raw_max = 0.0
        self.adp_raw_end = 0.0
        self.adp_norm = 0.0
        self.rank_base = -1
        self.rank_total = -1

    def project(self, p):
        p = np.asarray(p, float)
        dim = min(p.shape[0], self.waypoints.shape[1])
        p = p[:dim]
        wps = self.waypoints[:, :dim]
        best_pt, best_d = wps[0], np.inf
        for a, b in zip(wps[:-1], wps[1:]):
            ab = b - a
            denom = float(np.dot(ab, ab)) + 1e-9
            t = np.clip(float(np.dot(p - a, ab)) / denom, 0.0, 1.0)
            q = a + t * ab
            d = np.linalg.norm(p - q)
            if d < best_d:
                best_d, best_pt = d, q
        return best_pt, best_d

class SafetyManifold:
    def __init__(self, field, rho=1.0, lam_s=1.0, eps_m=0.02):
        self.field = field
        self.rho = float(rho)
        self.lam_s = float(lam_s)
        self.eps_m = float(eps_m)
        self._rng = np.random.RandomState(0)

    def psi(self, z, z_goal, Qg=1.0):
        z = np.asarray(z, float)
        gamma = Qg * float(np.sum((z[:2] - np.asarray(z_goal, float)[:2]) ** 2))

        m = self.eps_m * float(np.sin(3.0 * z[0]) * np.cos(3.0 * z[1]))
        return gamma + self.lam_s * self.field.phi_s(z) + m

    def in_safe_set(self, z, zdot=None):
        return self.field.phi_s(z, zdot) <= self.rho

    def critical_points(self, bounds, z_goal, n=21, z_height=0.0):
        (xmin, xmax), (ymin, ymax) = bounds
        xs = np.linspace(xmin, xmax, n)
        ys = np.linspace(ymin, ymax, n)
        P = np.zeros((n, n))
        safe = np.zeros((n, n), bool)
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                z = np.array([x, y, z_height], float)
                safe[i, j] = bool(self.in_safe_set(z))
                P[i, j] = self.psi(z, z_goal) if safe[i, j] else float("inf")
        minima, saddles, maxima = [], [], []
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                if not safe[i, j]:
                    continue
                if not (safe[i - 1, j] and safe[i + 1, j] and
                        safe[i, j - 1] and safe[i, j + 1]):
                    continue
                c = P[i, j]
                neigh = [P[i - 1, j], P[i + 1, j], P[i, j - 1], P[i, j + 1]]
                if not np.isfinite(c) or not all(np.isfinite(v) for v in neigh):
                    continue
                fxx = P[i + 1, j] - 2 * c + P[i - 1, j]
                fyy = P[i, j + 1] - 2 * c + P[i, j - 1]
                pt = np.array([xs[i], ys[j], z_height])
                if all(c <= v for v in neigh):
                    minima.append(pt)
                elif all(c >= v for v in neigh):
                    maxima.append(pt)
                elif fxx * fyy < 0:
                    saddles.append(pt)
        return {"minima": minima, "saddles": saddles, "maxima": maxima}

    def enumerate_corridors(self, start, z_goal, bounds, radius=0.25,
                            z_height=0.0, critic=None, feature_builder=None,
                            lambda_adp=0.0, feature_context=None):
        start = np.asarray(start, float)
        goal = np.asarray(z_goal, float)
        start2, goal2 = start[:2], goal[:2]
        mid = 0.5 * (start2 + goal2)
        direction = goal2 - start2
        L = np.linalg.norm(direction) + 1e-9
        normal = np.array([-direction[1], direction[0]]) / L

        candidates = []
        for offset, label in [(0.0, "direct"),
                              (0.35, "left_detour"),
                              (-0.35, "right_detour"),
                              (0.6, "wide_left"),
                              (-0.6, "wide_right")]:
            via = mid + offset * normal
            wps = np.array([
                [start2[0], start2[1], z_height],
                [via[0], via[1], z_height],
                [goal2[0], goal2[1], z_height],
            ])
            base_cost = self._corridor_cost(wps)
            adp_stats = {"mean": 0.0, "max": 0.0, "end": 0.0, "score": 0.0}
            if critic is not None and feature_builder is not None and lambda_adp > 0.0:
                adp_stats = self._corridor_adp_stats(
                    wps, critic, feature_builder, z_goal, feature_context)
            corridor = Corridor(wps, radius, label, base_cost)
            corridor.base_cost = float(base_cost)
            corridor.adp_raw_mean = float(adp_stats.get("mean", 0.0))
            corridor.adp_raw_max = float(adp_stats.get("max", 0.0))
            corridor.adp_raw_end = float(adp_stats.get("end", 0.0))
            corridor.adp_mean = corridor.adp_raw_mean
            corridor.adp_max = corridor.adp_raw_max
            corridor.adp_end = corridor.adp_raw_end
            corridor._adp_raw_score = float(adp_stats.get("score", 0.0))
            candidates.append(corridor)

        base_order = sorted(candidates, key=lambda c: c.base_cost)
        for rank, corridor in enumerate(base_order):
            corridor.rank_base = int(rank)

        if critic is not None and feature_builder is not None and lambda_adp > 0.0:
            norm = self._relative_normalize([c._adp_raw_score for c in candidates])
            base_vals = np.asarray([c.base_cost for c in candidates], float)
            base_span = float(np.max(base_vals) - np.min(base_vals))
            base_mean = float(np.mean(np.abs(base_vals)))
            adp_scale = max(base_span, 0.10 * base_mean, 1.0)
            for corridor, adp_norm in zip(candidates, norm):
                corridor.adp_norm = float(adp_norm)
                corridor.adp_cost = float(adp_norm) * adp_scale
                corridor.cost = corridor.base_cost + float(lambda_adp) * corridor.adp_cost
        else:
            for corridor in candidates:
                corridor.cost = corridor.base_cost

        total_order = sorted(candidates, key=lambda c: c.cost)
        for rank, corridor in enumerate(total_order):
            corridor.rank_total = int(rank)
        candidates.sort(key=lambda c: c.cost)
        return candidates

    def enumerate_topological_corridors(
            self, start, z_goal, bounds, radius=0.35, z_height=0.0,
            grid_resolution=None, merge_radius=None, min_clearance=None,
            hard_clearance=None, neighbor_k=None, k=3, max_graph_nodes=None,
            critic=None, feature_builder=None, lambda_adp=0.0,
            feature_context=None, semantic_nodes=None, to_world=None,
            to_plane=None, interest_config=None, topology_params=None,
            topology_profile="generic"):
        from stsm_madp.topology import TopologicalCorridorPlanner
        topology_params = dict(topology_params or {})
        planner = TopologicalCorridorPlanner(
            self.field, rho=self.rho, bounds=bounds,
            grid_resolution=grid_resolution, merge_radius=merge_radius,
            min_clearance=min_clearance, hard_clearance=hard_clearance,
            neighbor_k=neighbor_k, max_graph_nodes=max_graph_nodes,
            lam_s=self.lam_s, eps_m=self.eps_m, z_height=z_height,
            to_world=to_world, to_plane=to_plane,
            interest_config=interest_config,
            topology_profile=topology_profile, **topology_params)
        corridors = planner.corridor_from_morse_graph(
            start, z_goal, k=k, radius=radius, critic=critic,
            feature_builder=feature_builder, lambda_adp=lambda_adp,
            feature_context=feature_context, semantic_nodes=semantic_nodes)
        self.last_topology_debug = planner.last_debug
        self.last_safe_grid = planner._grid
        if planner._grid is not None:
            safe = planner._grid.get("safe")
            base_safe = planner._grid.get("base_safe")
            clearance = planner._grid.get("clearance")
            self.last_safe_manifold_summary = {
                "num_cells": int(safe.size) if safe is not None else 0,
                "num_safe_cells": int(np.sum(safe)) if safe is not None else 0,
                "num_base_safe_cells": (
                    int(np.sum(base_safe)) if base_safe is not None else 0),
                "min_clearance": (
                    float(np.nanmin(clearance)) if clearance is not None else 0.0),
                "hard_clearance": float(planner.hard_clearance),
                "target_clearance": float(planner.min_clearance),
                "topology_profile": planner.topology_profile,
                "workspace_dimension": int(planner.workspace_dimension),
            }
        return corridors

    def _relative_normalize(self, values, eps=1e-6):
        arr = np.asarray(values, float)
        if arr.size == 0:
            return arr
        lo = float(np.percentile(arr, 10))
        hi = float(np.percentile(arr, 90))
        if hi - lo < float(eps):
            return np.zeros_like(arr)
        return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

    def _corridor_cost(self, waypoints, samples=12):
        total = 0.0
        for a, b in zip(waypoints[:-1], waypoints[1:]):
            for t in np.linspace(0, 1, samples):
                total += self.field.phi_s(a + t * (b - a))
        return total / (samples * (len(waypoints) - 1))

    def _sample_polyline(self, waypoints, samples=9):
        waypoints = np.asarray(waypoints, float)
        samples = max(2, int(samples))
        seg_lens = []
        total_len = 0.0
        for a, b in zip(waypoints[:-1], waypoints[1:]):
            length = float(np.linalg.norm(b - a))
            seg_lens.append(length)
            total_len += length
        if total_len <= 1e-9:
            return np.repeat(waypoints[:1], samples, axis=0)
        distances = np.linspace(0.0, total_len, samples)
        pts = []
        for d in distances:
            acc = 0.0
            for i, length in enumerate(seg_lens):
                if d <= acc + length or i == len(seg_lens) - 1:
                    alpha = (d - acc) / max(length, 1e-9)
                    pts.append(waypoints[i] + alpha * (waypoints[i + 1] - waypoints[i]))
                    break
                acc += length
        return np.asarray(pts, float)

    def _corridor_adp_stats(self, waypoints, critic, feature_builder, z_goal,
                            feature_context=None):
        feature_context = feature_context or {}
        corridor = Corridor(waypoints, feature_context.get("radius", 0.25),
                            feature_context.get("label", ""), 0.0)
        raw_values = []
        cost_values = []
        samples = int(feature_context.get("adp_samples", 9))
        for p in self._sample_polyline(waypoints, samples=samples):
            pose2d = np.array([
                p[0], p[1], feature_context.get("yaw", 0.0)
            ], float)
            features = feature_builder.build_wheelchair(
                pose2d,
                np.asarray(z_goal, float)[:2],
                self.field,
                gate_info=feature_context.get("gate_info", {}),
                interest_risk=feature_context.get("interest_risk", {}),
                corridor=corridor,
                u=feature_context.get("u"))
            detail = critic.predict_detail(features)
            clipped = float(detail.get("clipped", detail.get("raw", 0.0)))
            raw_values.append(clipped)
            cost_values.append(max(0.0, clipped))
        arr = np.asarray(raw_values, float)
        cost_arr = np.asarray(cost_values, float)
        stats = {
            "mean": float(np.mean(arr)) if len(arr) else 0.0,
            "max": float(np.max(arr)) if len(arr) else 0.0,
            "end": float(arr[-1]) if len(arr) else 0.0,
            "cost_mean": float(np.mean(cost_arr)) if len(cost_arr) else 0.0,
            "cost_max": float(np.max(cost_arr)) if len(cost_arr) else 0.0,
            "cost_end": float(cost_arr[-1]) if len(cost_arr) else 0.0,
        }
        weights = feature_context.get("adp_corridor_weights", {})
        w_mean = float(weights.get("mean", 0.4))
        w_max = float(weights.get("max", 0.4))
        w_end = float(weights.get("end", 0.2))
        stats["score"] = (
            w_mean * stats["cost_mean"] +
            w_max * stats["cost_max"] +
            w_end * stats["cost_end"])
        return stats
