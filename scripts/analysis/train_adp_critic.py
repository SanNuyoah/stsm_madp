#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import time
sys.dont_write_bytecode = True

import numpy as np

PACKAGE_SRC = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "src"))
if os.path.isdir(os.path.join(PACKAGE_SRC, "stsm_madp")) and PACKAGE_SRC not in sys.path:
    sys.path.insert(0, PACKAGE_SRC)

from stsm_madp.adp import (  # noqa: E402
    ADPCritic, ADPFeatureBuilder, DEFAULT_FEATURE_NAMES,
    discounted_returns, stage_cost,
)


def load_rows(paths):
    rows = []
    for path in paths:
        with open(path, "r") as f:
            for row in csv.DictReader(f):
                row["_source_file"] = os.path.basename(path)
                rows.append(row)
    return rows


def row_float(row, key, default=0.0):
    try:
        value = row.get(key, default)
        if value == "" or value is None:
            return float(default)
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return float(default)
        return value
    except (TypeError, ValueError):
        return float(default)


def valid_row(row):
    if row.get("phi_total", "") == "":
        return False
    if row.get("x", "") == "" or row.get("y", "") == "":
        return False
    return True


def split_episodes(rows):
    episodes = {}
    for row in rows:
        if not valid_row(row):
            continue
        key = (
            row.get("_source_file", ""),
            row.get("run_id", ""),
            row.get("target", ""),
            row.get("mode", ""),
        )
        episodes.setdefault(key, []).append(row)
    ordered = []
    for key, ep in episodes.items():
        ep.sort(key=lambda r: row_float(r, "t"))
        if len(ep) >= 3:
            ordered.append((key, ep))
    return ordered


def build_dataset(episodes, builder, gamma, weights, clip_return):
    x_all = []
    g_all = []
    meta = []
    for key, ep in episodes:
        feats = []
        costs = []
        prev = None
        for row in ep:
            feat = builder.from_row(row, prev)
            feats.append(feat)
            costs.append(stage_cost(feat, weights))
            prev = row
        returns = discounted_returns(costs, gamma=gamma, clip_value=clip_return)
        for feat, ret in zip(feats, returns):
            x_all.append([feat.get(name, 0.0) for name in builder.feature_names])
            g_all.append(float(ret))
        meta.append((key, len(ep), float(np.mean(returns)), float(np.max(returns))))
    return np.asarray(x_all, float), np.asarray(g_all, float), meta


def main():
    parser = argparse.ArgumentParser(description="Train linear ADP critic from STSM trajectory CSV files.")
    parser.add_argument("--traj", nargs="+", required=True, help="Trajectory CSV files.")
    parser.add_argument("--out", required=True, help="Output YAML model path.")
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--clip-value", type=float, default=200.0)
    parser.add_argument("--clip-return", type=float, default=200.0)
    parser.add_argument("--feature", action="append", dest="features",
                        help="Feature name. Repeat to override default feature list.")
    args = parser.parse_args()

    missing = [p for p in args.traj if not os.path.exists(p)]
    if missing:
        raise SystemExit("missing trajectory files: {}".format(", ".join(missing)))

    feature_names = args.features or DEFAULT_FEATURE_NAMES
    builder = ADPFeatureBuilder(feature_names)
    rows = load_rows(args.traj)
    episodes = split_episodes(rows)
    if not episodes:
        raise SystemExit("no valid episodes found")

    critic = ADPCritic(feature_names=feature_names, gamma=args.gamma,
                       clip_value=args.clip_value)
    x, g, meta = build_dataset(
        episodes, builder, args.gamma, critic.cost_weights, args.clip_return)
    loss = critic.fit_lstsq(x, g, ridge=args.ridge)

    pred = np.dot((x - critic.mean[None, :]) / critic.std[None, :], critic.theta)
    clipped = np.clip(pred, -critic.clip_value, critic.clip_value)
    clip_ratio = float(np.mean(np.abs(pred - clipped) > 1e-9))
    abs_clipped = np.abs(clipped)
    critic.metadata.update({
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "train_sources": [os.path.abspath(p) for p in args.traj],
        "train_samples": int(x.shape[0]),
        "train_episodes": int(len(episodes)),
        "train_loss": float(loss),
        "value_min": float(np.min(clipped)),
        "value_mean": float(np.mean(clipped)),
        "value_p95": float(np.percentile(abs_clipped, 95)),
        "clip_ratio": clip_ratio,
    })
    critic.save_yaml(args.out)
    print("trained {}".format(args.out))
    print("episodes: {}  samples: {}  features: {}".format(
        len(episodes), x.shape[0], x.shape[1]))
    print("loss: {:.6f}  target_mean: {:.4f}  pred_mean: {:.4f}".format(
        loss, float(np.mean(g)), float(np.mean(pred))))
    print("value_min: {:.4f}  value_mean: {:.4f}  value_p95_abs: {:.4f}  clip_ratio: {:.2%}".format(
        float(np.min(clipped)), float(np.mean(clipped)),
        float(np.percentile(abs_clipped, 95)), clip_ratio))
    for key, n, mean_ret, max_ret in meta:
        print("episode {} n={} mean_return={:.3f} max_return={:.3f}".format(
            "/".join([str(x) for x in key if x]), n, mean_ret, max_ret))


if __name__ == "__main__":
    main()
