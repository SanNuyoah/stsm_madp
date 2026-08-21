#!/usr/bin/env python3
from __future__ import print_function

import argparse
import csv
import os
import sys
sys.dont_write_bytecode = True

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.font_manager import FontProperties

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
SCRIPTS = os.path.join(ROOT, "scripts")
if os.path.isdir(os.path.join(SRC, "stsm_madp")) and SRC not in sys.path:
    sys.path.insert(0, SRC)
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from plot_risk_field_paths import grid_phi, grid_phi_xz, make_arm_field, make_wc_field


FONT_REGULAR = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

BLUE = "#0072B2"
ORANGE = "#D55E00"
BLACK = "#202124"
MUTED = "#667085"
GRID = "#E6E8EB"
SAFE = "#D9F0D3"
RISK = "#F7B267"
RED = "#C44E52"
GREEN = "#4DBD5B"
PURPLE = "#7B61A8"

STATIC_DPI = 300


def setup_rc():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 9.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "axes.linewidth": 0.8,
        "savefig.dpi": STATIC_DPI,
    })


def load_traj(path):
    rows = []
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            rows.append([
                float(row["t"]),
                float(row["x"]),
                float(row["y"]),
                float(row.get("z", 0.0)),
            ])
    if len(rows) < 2:
        raise RuntimeError("trajectory needs at least two rows: {0}".format(path))
    data = np.asarray(rows, float)
    data[:, 0] -= data[0, 0]
    return data


def existing_result_file(results_dir, names):
    for name in names:
        path = os.path.join(results_dir, name)
        if os.path.exists(path):
            return path
    return os.path.join(results_dir, names[0])


def load_metrics(path):
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            mode = row.get("mode", "").strip().lower()
            if not mode:
                continue
            data = {}
            for key, value in row.items():
                try:
                    data[key] = float(value)
                except (TypeError, ValueError):
                    data[key] = value
            out[mode] = data
    return out


def velocities(t, xyz):
    vel = np.zeros_like(xyz)
    dt = np.diff(t)
    dt = np.maximum(dt, 1e-3)
    vel[1:] = np.diff(xyz, axis=0) / dt[:, None]
    return vel


def windowed_speed(t, xyz, horizon=0.6):
    """Estimate speed over a fixed time window to reject timestamp jitter."""
    t = np.asarray(t, float)
    xyz = np.asarray(xyz, float)
    speed = np.zeros(len(t), float)
    if len(t) < 2:
        return speed

    half = float(horizon) / 2.0
    for i, ti in enumerate(t):
        left = int(np.searchsorted(t, ti - half, side="left"))
        right = int(np.searchsorted(t, ti + half, side="right")) - 1
        if right <= left:
            left = max(0, i - 1)
            right = min(len(t) - 1, i + 1)
        dt = t[right] - t[left]
        if dt > 1e-6:
            speed[i] = np.linalg.norm(xyz[right] - xyz[left]) / dt
    return speed


def moving_average(values, window=7):
    values = np.asarray(values, float)
    if len(values) < window:
        return values
    kernel = np.ones(window, float) / float(window)
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def style_axis(ax, grid_axis="both"):
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.75)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#B8C0CC")
    ax.spines["bottom"].set_color("#B8C0CC")
    ax.tick_params(colors=MUTED, length=3)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(FONT_REGULAR)


def apply_text_props(ax):
    ax.title.set_fontproperties(FONT_BOLD)
    ax.xaxis.label.set_fontproperties(FONT_REGULAR)
    ax.yaxis.label.set_fontproperties(FONT_REGULAR)
    leg = ax.get_legend()
    if leg:
        for text in leg.get_texts():
            text.set_fontproperties(FONT_REGULAR)


def save_fig(fig, out_path):
    fig.tight_layout()
    fig.savefig(out_path, dpi=STATIC_DPI, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def risk_series(field, traj):
    t = traj[:, 0]
    xyz = traj[:, 1:4]
    vel = velocities(t, xyz)
    return np.asarray([field.phi_s(p, v) for p, v in zip(xyz, vel)], float)


def speed_series(traj):
    t = traj[:, 0]
    xyz = traj[:, 1:4]
    return windowed_speed(t, xyz)


def signed_rect_distance_xy(points, center, half_extent):
    center = np.asarray(center[:2], float)
    half_extent = np.asarray(half_extent[:2], float)
    d = np.abs(points[:, :2] - center[None, :]) - half_extent[None, :]
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
    inside = np.minimum(np.max(d, axis=1), 0.0)
    return outside + inside


def draw_arm_safety(out_path, baseline, stsm):
    field = make_arm_field()
    xlim = (-0.02, 0.92)
    zlim = (0.05, 1.00)
    xx, zz, phi = grid_phi_xz(field, xlim, zlim, y=0.0, nx=240, nz=220)
    threshold = 1.60

    fig, ax = plt.subplots(figsize=(6.7, 5.4))
    ax.contourf(xx, zz, phi, levels=[0.0, threshold, max(4.0, float(np.max(phi)))],
                colors=[SAFE, RISK], alpha=0.72)
    cs = ax.contour(xx, zz, phi, levels=[threshold], colors=BLACK, linewidths=1.25)
    ax.clabel(cs, fmt={threshold: r"$\Phi_s=1.60$"}, inline=True, fontsize=8)

    ax.plot(baseline[:, 1], baseline[:, 3], color=BLUE, lw=2.0, label="Baseline")
    ax.plot(stsm[:, 1], stsm[:, 3], color=ORANGE, lw=2.0, label="STSM")
    ax.scatter([baseline[0, 1]], [baseline[0, 3]], marker="s", s=70, color=BLUE,
               edgecolor="white", linewidth=0.8, zorder=5)
    ax.scatter([0.42], [0.21], marker="*", s=160, color="#EFC000",
               edgecolor=BLACK, linewidth=0.7, zorder=5)
    ax.add_patch(patches.Circle((0.78, 0.61), 0.10, color=RED, alpha=0.50))
    ax.add_patch(patches.Circle((0.78, 0.31), 0.075, color="#F39C35", alpha=0.55))
    ax.add_patch(patches.Circle((0.42, 0.21), 0.05, color=GREEN, alpha=0.60))
    ax.text(0.80, 0.55, "头部/胸部高风险", fontproperties=FONT_REGULAR)
    ax.text(0.44, 0.16, "递物目标", fontproperties=FONT_REGULAR)

    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("末端 X (m)")
    ax.set_ylabel("末端高度 Z (m)")
    ax.set_title("机械臂递物社会风险阈值子水平集与末端轨迹")
    ax.legend(loc="lower right", frameon=True)
    style_axis(ax)
    apply_text_props(ax)
    save_fig(fig, out_path)


def draw_wc_safety(out_path, baseline, stsm):
    field = make_wc_field()
    xlim = (-2.1, 2.5)
    ylim = (-2.0, 2.0)
    xx, yy, phi = grid_phi(field, xlim, ylim, z=0.03, nx=240, ny=210)
    threshold = 0.80
    goal = np.array([-0.55, 0.55])

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    ax.contourf(xx, yy, phi, levels=[0.0, threshold, max(4.0, float(np.max(phi)))],
                colors=[SAFE, RISK], alpha=0.72)
    cs = ax.contour(xx, yy, phi, levels=[threshold], colors=BLACK, linewidths=1.25)
    ax.clabel(cs, fmt={threshold: r"$\Phi_s=0.80$"}, inline=True, fontsize=8)
    ax.add_patch(patches.Rectangle((-2.1, -2.0), 1.0, 2.0, color="#6B5CA5", alpha=0.28))
    ax.add_patch(patches.Rectangle((-1.1, -2.0), 0.8, 2.0, fill=False,
                                   edgecolor=RED, linewidth=1.4, linestyle="--"))
    ax.add_patch(patches.Rectangle((0.32, -0.55), 0.6, 1.1, color="#A77743", alpha=0.45))
    ax.add_patch(patches.Circle((-1.6, 0.2), 0.12, color=PURPLE, alpha=0.85))

    ax.plot(baseline[:, 1], baseline[:, 2], color=BLUE, lw=2.0, label="Baseline")
    ax.plot(stsm[:, 1], stsm[:, 2], color=ORANGE, lw=2.0, label="STSM")
    ax.scatter([baseline[0, 1]], [baseline[0, 2]], marker="s", s=70, color=BLUE,
               edgecolor="white", linewidth=0.8, zorder=5)
    ax.scatter([goal[0]], [goal[1]], marker="*", s=180, color="#EFC000",
               edgecolor=BLACK, linewidth=0.7, zorder=5)
    ax.text(goal[0] + 0.06, goal[1] + 0.06, "泊靠目标", fontproperties=FONT_REGULAR)
    ax.text(-1.75, -1.15, "床", color="white", fontproperties=FONT_REGULAR)
    ax.text(-0.98, -1.55, "转移区", color=RED, fontproperties=FONT_REGULAR)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("轮椅中心 X (m)")
    ax.set_ylabel("轮椅中心 Y (m)")
    ax.set_title("轮椅泊靠社会风险阈值子水平集与中心轨迹")
    ax.legend(loc="upper right", frameon=True)
    style_axis(ax)
    apply_text_props(ax)
    save_fig(fig, out_path)


def draw_two_line_plot(out_path, title, ylabel, baseline_t, baseline_y, stsm_t, stsm_y,
                       threshold=None, threshold_label=None, smooth=True):
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    baseline_plot = moving_average(baseline_y) if smooth else baseline_y
    stsm_plot = moving_average(stsm_y) if smooth else stsm_y
    ax.plot(baseline_t, baseline_plot, color=BLUE, lw=1.9, label="Baseline")
    ax.plot(stsm_t, stsm_plot, color=ORANGE, lw=1.9, label="STSM")
    if threshold is not None:
        ax.axhline(threshold, color=BLACK, lw=1.0, ls="--", alpha=0.75)
        if threshold_label:
            ax.text(0.99, threshold, threshold_label, ha="right", va="bottom",
                    transform=ax.get_yaxis_transform(), fontsize=8.0,
                    color=BLACK, fontproperties=FONT_REGULAR)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=True)
    style_axis(ax)
    apply_text_props(ax)
    save_fig(fig, out_path)


def draw_arm_distances(out_path, baseline, stsm):
    points = {
        "末端-头部距离": np.array([0.78, 0.0, 0.61]),
        "末端-胸部距离": np.array([0.78, 0.0, 0.31]),
        "末端-手部目标距离": np.array([0.42, 0.0, 0.21]),
    }
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.1), sharex=False)
    for ax, (label, point) in zip(axes, points.items()):
        b = np.linalg.norm(baseline[:, 1:4] - point[None, :], axis=1)
        s = np.linalg.norm(stsm[:, 1:4] - point[None, :], axis=1)
        ax.plot(baseline[:, 0], moving_average(b), color=BLUE, lw=1.8, label="Baseline")
        ax.plot(stsm[:, 0], moving_average(s), color=ORANGE, lw=1.8, label="STSM")
        ax.set_title(label)
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("距离 (m)")
        style_axis(ax)
        apply_text_props(ax)
    axes[-1].legend(loc="upper right", frameon=True)
    apply_text_props(axes[-1])
    save_fig(fig, out_path)


def draw_arm_body_risk(out_path, baseline, stsm):
    field = make_arm_field()
    human = field.humans[0]
    params = field.params

    def components(traj):
        total = risk_series(field, traj)
        body = []
        env = []
        for point in traj[:, 1:4]:
            body.append(params.lam_body * field.phi_body(point, human))
            env.append(params.lam_env * field.phi_env(point))
        body = np.asarray(body, float)
        env = np.asarray(env, float)
        social = np.maximum(total - body - env, 0.0)
        return total, body, env, social

    b_total, b_body, b_env, b_social = components(baseline)
    s_total, s_body, s_env, s_social = components(stsm)
    parts = [
        ("总社会风险", b_total, s_total),
        ("身体部位风险", b_body, s_body),
        ("社交/环境风险", b_social + b_env, s_social + s_env),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.1), sharex=False)
    for ax, (label, b, s) in zip(axes, parts):
        ax.plot(baseline[:, 0], moving_average(b), color=BLUE, lw=1.8, label="Baseline")
        ax.plot(stsm[:, 0], moving_average(s), color=ORANGE, lw=1.8, label="STSM")
        ax.set_title(label)
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("风险")
        style_axis(ax)
        apply_text_props(ax)
    axes[-1].legend(loc="upper right", frameon=True)
    apply_text_props(axes[-1])
    save_fig(fig, out_path)


def draw_wc_distances(out_path, baseline, stsm):
    person = np.array([-1.6, 0.2])
    transfer_center = np.array([-0.7, -1.0, 0.0])
    transfer_half = np.array([0.4, 1.0, 0.5])
    b_person = np.linalg.norm(baseline[:, 1:3] - person[None, :], axis=1)
    s_person = np.linalg.norm(stsm[:, 1:3] - person[None, :], axis=1)
    b_transfer = signed_rect_distance_xy(baseline[:, 1:4], transfer_center, transfer_half)
    s_transfer = signed_rect_distance_xy(stsm[:, 1:4], transfer_center, transfer_half)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3))
    axes[0].plot(baseline[:, 0], moving_average(b_person), color=BLUE, lw=1.9, label="Baseline")
    axes[0].plot(stsm[:, 0], moving_average(s_person), color=ORANGE, lw=1.9, label="STSM")
    axes[0].set_title("轮椅中心-老人距离")
    axes[0].set_xlabel("时间 (s)")
    axes[0].set_ylabel("距离 (m)")
    axes[1].plot(baseline[:, 0], moving_average(b_transfer), color=BLUE, lw=1.9, label="Baseline")
    axes[1].plot(stsm[:, 0], moving_average(s_transfer), color=ORANGE, lw=1.9, label="STSM")
    axes[1].axhline(0.0, color=BLACK, lw=1.0, ls="--", alpha=0.75)
    axes[1].set_title("床边转移区净距")
    axes[1].set_xlabel("时间 (s)")
    axes[1].set_ylabel("有符号距离 (m)")
    axes[1].legend(loc="upper right", frameon=True)
    for ax in axes:
        style_axis(ax)
        apply_text_props(ax)
    save_fig(fig, out_path)


def build_plots(results_dir, out_dir):
    setup_rc()
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    arm_baseline = load_traj(existing_result_file(
        results_dir, ["arm/baseline/traj.csv", "arm_baseline_traj.csv"]))
    arm_stsm = load_traj(existing_result_file(
        results_dir, ["arm/stsm/traj.csv", "arm_stsm_traj.csv"]))
    wc_baseline = load_traj(existing_result_file(
        results_dir, ["wheelchair/baseline/traj.csv", "wc_baseline_traj.csv"]))
    wc_stsm = load_traj(existing_result_file(
        results_dir, ["wheelchair/stsm/traj.csv", "wc_stsm_traj.csv"]))

    draw_arm_safety(os.path.join(out_dir, "arm_social_risk_sublevel_set.png"),
                    arm_baseline, arm_stsm)
    draw_wc_safety(os.path.join(out_dir, "wheelchair_social_risk_sublevel_set.png"),
                   wc_baseline, wc_stsm)

    arm_field = make_arm_field()
    wc_field = make_wc_field()
    draw_two_line_plot(
        os.path.join(out_dir, "arm_risk_time.png"),
        "机械臂递物社会风险演化",
        r"社会风险 $\Phi_s$",
        arm_baseline[:, 0],
        risk_series(arm_field, arm_baseline),
        arm_stsm[:, 0],
        risk_series(arm_field, arm_stsm),
        threshold=1.60,
        threshold_label=r"$\Phi_s=1.60$",
    )
    draw_two_line_plot(
        os.path.join(out_dir, "wheelchair_risk_time.png"),
        "轮椅泊靠社会风险演化",
        r"社会风险 $\Phi_s$",
        wc_baseline[:, 0],
        risk_series(wc_field, wc_baseline),
        wc_stsm[:, 0],
        risk_series(wc_field, wc_stsm),
        threshold=0.80,
        threshold_label=r"$\Phi_s=0.80$",
    )
    draw_arm_distances(os.path.join(out_dir, "arm_distance_time.png"),
                       arm_baseline, arm_stsm)
    draw_arm_body_risk(os.path.join(out_dir, "arm_body_risk_time.png"),
                       arm_baseline, arm_stsm)
    draw_wc_distances(os.path.join(out_dir, "wheelchair_distance_time.png"),
                      wc_baseline, wc_stsm)

def main():
    parser = argparse.ArgumentParser(
        description="Compatibility entry point; plot_results.py now creates all result figures.")
    parser.add_argument("--results", default=os.path.join(ROOT, "results"))
    parser.add_argument("--out", default=os.path.join(ROOT, "pictures"))
    args = parser.parse_args()
    from plot_results import build_plots as build_all_plots
    build_all_plots(args.results, args.out)
    print("wrote {0}".format(args.out))


if __name__ == "__main__":
    main()
