import sys
sys.dont_write_bytecode = True

import numpy as np

def path_length(path):
    pts = np.asarray(path, float)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def interpolate_by_segments(waypoints, n=30):
    waypoints = np.asarray(waypoints, float)
    if len(waypoints) <= 1:
        return waypoints.copy(), list(range(len(waypoints)))
    n = max(len(waypoints), int(n))
    seg_lens = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    total = float(np.sum(seg_lens))
    if total <= 1e-9:
        return waypoints.copy(), list(range(len(waypoints)))

    remaining = max(0, n - 1 - len(seg_lens))
    raw = seg_lens / total * remaining
    extra = np.floor(raw).astype(int)
    for idx in np.argsort(-(raw - extra))[:remaining - int(np.sum(extra))]:
        extra[idx] += 1

    pts = [waypoints[0]]
    protected_indices = [0]
    for seg_idx, (a, b) in enumerate(zip(waypoints[:-1], waypoints[1:])):
        count = int(extra[seg_idx]) + 1
        for step in range(1, count + 1):
            alpha = float(step) / float(count)
            pts.append(a + alpha * (b - a))
        protected_indices.append(len(pts) - 1)
    return np.asarray(pts, float), protected_indices


def protected_waypoint_distances(path, protected_waypoints):
    pts = np.asarray(path, float)
    protected = np.asarray(protected_waypoints, float)
    if len(pts) == 0 or len(protected) == 0:
        return np.zeros(0, float)
    out = []
    for waypoint in protected:
        out.append(float(np.min(np.linalg.norm(pts - waypoint[None, :], axis=1))))
    return np.asarray(out, float)


def _segment_is_safe(a, b, field, corridor=None, rho=float("inf"),
                     samples=8):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    samples = max(2, int(samples))
    for alpha in np.linspace(0.0, 1.0, samples):
        p = a + alpha * (b - a)
        if float(field.phi_s(p)) > float(rho):
            return False
        if corridor is not None:
            _q, d = corridor.project(p)
            if d > float(corridor.radius) + 1e-9:
                return False
    return True


def shortcut_trajectory(path, field, corridor=None, rho=float("inf"),
                        samples=8, max_passes=2):
    pts = np.asarray(path, float)
    if len(pts) <= 2:
        return pts
    out = pts.copy()
    for _ in range(max(1, int(max_passes))):
        if len(out) <= 2:
            break
        changed = False
        kept = [out[0]]
        i = 0
        while i < len(out) - 1:
            best = i + 1
            for j in range(len(out) - 1, i + 1, -1):
                if _segment_is_safe(
                        out[i], out[j], field, corridor=corridor,
                        rho=rho, samples=samples):
                    best = j
                    break
            kept.append(out[best])
            if best > i + 1:
                changed = True
            i = best
        out = np.asarray(kept, float)
        if not changed:
            break
    return out


def topology_preserving_shortcut(path, protected_indices, field, corridor=None,
                                 rho=float("inf"), samples=8, max_passes=2):
    pts = np.asarray(path, float)
    if len(pts) <= 2:
        return pts, list(range(len(pts)))
    protected = sorted(set(
        int(i) for i in protected_indices
        if 0 <= int(i) < len(pts)))
    if not protected or protected[0] != 0:
        protected = [0] + protected
    if protected[-1] != len(pts) - 1:
        protected.append(len(pts) - 1)

    merged = []
    new_protected = []
    for left, right in zip(protected[:-1], protected[1:]):
        segment = pts[left:right + 1]
        shortened = shortcut_trajectory(
            segment, field, corridor=corridor, rho=rho,
            samples=samples, max_passes=max_passes)
        if merged:
            new_protected.append(len(merged) - 1)
            shortened = shortened[1:]
        else:
            new_protected.append(0)
        merged.extend(shortened)
    new_protected.append(len(merged) - 1)
    return np.asarray(merged, float), new_protected


def bezier_smooth_polyline(waypoints, samples_per_segment=10):
    pts = np.asarray(waypoints, float)
    if len(pts) <= 2:
        return pts.copy()
    samples_per_segment = max(2, int(samples_per_segment))
    out = [pts[0]]
    for i in range(len(pts) - 1):
        p0 = pts[i - 1] if i > 0 else pts[i]
        p1 = pts[i]
        p2 = pts[i + 1]
        p3 = pts[i + 2] if i + 2 < len(pts) else pts[i + 1]
        for t in np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)[1:]:
            t2 = t * t
            t3 = t2 * t
            out.append(0.5 * (
                (2.0 * p1) +
                (-p0 + p2) * t +
                (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
                (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3))
        out.append(p2)
    return np.asarray(out, float)


def path_curvature_metrics(path):
    pts = np.asarray(path, float)
    if len(pts) < 3:
        return {"max_turn": 0.0, "mean_turn": 0.0, "max_curvature": 0.0}
    turns = []
    curvatures = []
    for a, b, c in zip(pts[:-2], pts[1:-1], pts[2:]):
        u = b[:2] - a[:2]
        v = c[:2] - b[:2]
        nu = float(np.linalg.norm(u))
        nv = float(np.linalg.norm(v))
        if nu <= 1e-9 or nv <= 1e-9:
            continue
        dot = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
        turn = float(np.arccos(dot))
        turns.append(turn)
        curvatures.append(turn / max(0.5 * (nu + nv), 1e-9))
    return {
        "max_turn": float(np.max(turns)) if turns else 0.0,
        "mean_turn": float(np.mean(turns)) if turns else 0.0,
        "max_curvature": float(np.max(curvatures)) if curvatures else 0.0,
        "turns": list(turns),
    }


def deform_trajectory(tau0, field, corridor=None, lam_social=0.08,
                      lam_smooth=0.15, lam_nominal=0.0, iters=60, step=0.05,
                      fix_endpoints=True, critic=None, feature_builder=None,
                      lambda_adp_path=0.0, feature_context=None,
                      adp_eps=1e-3, protected_indices=None):
    tau = np.array(tau0, dtype=float)
    n, dim = tau.shape
    tau_ref = np.array(tau0, dtype=float)
    feature_context = feature_context if feature_context is not None else {}
    protected = set(int(i) for i in (protected_indices or [])
                    if 0 <= int(i) < n)

    def adp_value(point):
        if critic is None or feature_builder is None or lambda_adp_path <= 0.0:
            return 0.0
        target = feature_context.get("target_pos", tau_ref[-1])
        features = feature_builder.build_arm(
            point,
            target,
            field,
            gate_info=feature_context.get("gate_info", {}),
            interest_risk=feature_context.get("interest_risk", {}),
            phase=feature_context.get("phase", 1),
            u=None)
        return max(0.0, critic.predict(features))

    def adp_grad(point):
        g = np.zeros(dim, float)
        if critic is None or feature_builder is None or lambda_adp_path <= 0.0:
            return g
        for j in range(dim):
            d = np.zeros(dim, float)
            d[j] = float(adp_eps)
            g[j] = (adp_value(point + d) - adp_value(point - d)) / (2.0 * adp_eps)
        return g

    for _ in range(iters):
        grad = np.zeros_like(tau)
        for i in range(n):
            if (fix_endpoints and (i == 0 or i == n - 1)) or i in protected:
                continue

            grad[i] += lam_social * field.grad_phi_s(tau[i])
            grad[i] += float(lambda_adp_path) * adp_grad(tau[i])

            if lam_nominal > 0:
                grad[i] += lam_nominal * (tau[i] - tau_ref[i])

            if 0 < i < n - 1:
                lap = tau[i - 1] - 2 * tau[i] + tau[i + 1]
                grad[i] += -lam_smooth * lap
        tau = tau - step * grad

        if corridor is not None:
            for i in range(n):
                q, d = corridor.project(tau[i])
                if d > corridor.radius:
                    pull = (q - tau[i][: q.shape[0]])
                    tau[i][: q.shape[0]] += pull * (1.0 - corridor.radius /
                                                    (d + 1e-9))
        if fix_endpoints:
            tau[0] = tau_ref[0]
            tau[-1] = tau_ref[-1]
        for idx in protected:
            tau[idx] = tau_ref[idx]
    values = [adp_value(p) for p in tau]
    ref_values = [adp_value(p) for p in tau_ref]
    if values:
        feature_context["path_adp_mean"] = float(np.mean(values))
        feature_context["path_adp_max"] = float(np.max(values))
        feature_context["path_adp_delta"] = float(np.mean(values) - np.mean(ref_values))
        feature_context["adp_path_enabled"] = int(
            critic is not None and feature_builder is not None and lambda_adp_path > 0.0)
    return tau
