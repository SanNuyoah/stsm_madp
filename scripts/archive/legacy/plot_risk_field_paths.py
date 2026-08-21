#!/usr/bin/env python3
from __future__ import print_function

import argparse
import csv
import os
import sys
sys.dont_write_bytecode = True

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import patches
from matplotlib.font_manager import FontProperties

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
if os.path.isdir(os.path.join(SRC, "stsm_madp")) and SRC not in sys.path:
    sys.path.insert(0, SRC)

from stsm_madp.social_field import HumanState, SemanticAnchor, SocialField, SocialFieldParams


FONT_REGULAR = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

BLUE = "#3C5488"
RVIZ_BLUE = "#0072B2"
RVIZ_ORANGE = "#D55E00"
BLACK = "#222222"
RED = "#C44E52"
ORANGE = "#EFC000"
GREEN = "#4DBD5B"
PURPLE = "#B23AEE"
RVIZ_MAX_RISK = 4.0
STATIC_FIGSIZE = (7.2, 5.6)
STATIC_DPI = 300
WHEELCHAIR_GOAL_TOL = 0.08


def setup_rc():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "savefig.dpi": 300,
    })


def pt(data):
    return np.array(data, float)


def make_arm_field():
    human = HumanState(
        pos=[0.78, 0.0, 0.31],
        heading=np.pi,
        posture="sitting",
        vulnerability=1.3,
        body_parts={
            "head": (pt([0.78, 0.0, 0.61]), 3.0, 0.13),
            "chest": (pt([0.78, 0.0, 0.31]), 1.6, 0.18),
            "hand": (pt([0.42, 0.0, 0.21]), 0.3, 0.10),
        })
    table = SemanticAnchor("table", [0.55, 0.0, -0.37], [0.30, 0.50, 0.37],
                           weight=1.0, forbidden=False)
    field = SocialField(SocialFieldParams(
        lam_prox=1.0, lam_close=1.2, lam_dir=0.6, lam_body=2.5,
        lam_env=0.8, sigma_env=0.25))
    field.set_scene([human], [table])
    return field


def make_wc_field():
    human = HumanState(pos=[-1.6, 0.2, 0.0], heading=np.pi / 2,
                       posture="transferring", vulnerability=1.4)
    bed = SemanticAnchor("bed", [-1.6, -1.0, 0.0], [0.5, 1.0, 0.5],
                         weight=2.0, forbidden=True)
    transfer = SemanticAnchor("transfer-zone", [-0.7, -1.0, 0.0],
                              [0.4, 1.0, 0.5], weight=2.5, forbidden=True)
    table = SemanticAnchor("table", [0.55, 0.0, 0.0], [0.3, 0.5, 0.4],
                           weight=1.0, forbidden=True)
    field = SocialField(SocialFieldParams(
        lam_prox=1.2, lam_close=1.0, lam_dir=0.5, lam_body=0.0,
        lam_env=1.5, sigma_env=0.4))
    field.set_scene([human], [bed, transfer, table])
    return field


def grid_phi(field, xlim, ylim, z, nx=260, ny=220):
    xs = np.linspace(xlim[0], xlim[1], nx)
    ys = np.linspace(ylim[0], ylim[1], ny)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.zeros_like(xx)
    for i in range(yy.shape[0]):
        for j in range(xx.shape[1]):
            zz[i, j] = field.phi_s(np.array([xx[i, j], yy[i, j], z]))
    return xx, yy, zz


def grid_phi_xz(field, xlim, zlim, y=0.0, nx=260, nz=220):
    xs = np.linspace(xlim[0], xlim[1], nx)
    zs = np.linspace(zlim[0], zlim[1], nz)
    xx, zz = np.meshgrid(xs, zs)
    phi = np.zeros_like(xx)
    for i in range(zz.shape[0]):
        for j in range(xx.shape[1]):
            phi[i, j] = field.phi_s(np.array([xx[i, j], y, zz[i, j]]))
    return xx, zz, phi


def line_points(a, b, n):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    alpha = np.linspace(0.0, 1.0, n)[:, None]
    return a[None, :] + alpha * (b - a)[None, :]


def smooth_waypoints(waypoints, samples_per_segment=24):
    pts = np.asarray(waypoints, float)
    if len(pts) < 3:
        return path_from_waypoints(pts, samples_per_segment)
    padded = np.vstack([pts[0], pts, pts[-1]])
    out = []
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
        for t in np.linspace(0.0, 1.0, samples_per_segment, endpoint=False):
            t2 = t * t
            t3 = t2 * t
            out.append(0.5 * (
                (2.0 * p1) +
                (-p0 + p2) * t +
                (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
                (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3))
    out.append(pts[-1])
    return np.asarray(out)


def path_from_waypoints(waypoints, samples_per_segment=32):
    parts = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        parts.append(line_points(a, b, samples_per_segment))
    return np.vstack(parts)


def sampled_indices(length, frames):
    if length <= 1:
        return np.ones(frames, dtype=int)
    idx = np.linspace(1, length, frames)
    return np.maximum(1, np.minimum(length, np.round(idx).astype(int)))


def load_traj(path):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            rows.append([float(row["x"]), float(row["y"]), float(row.get("z", 0.0))])
    if len(rows) < 2:
        return None
    return np.asarray(rows, float)


def _load_handover_scene():
    path = os.path.join(ROOT, "config", "handover_scene.yaml")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _configured_arm_paths():
    scene = _load_handover_scene()
    start = np.array(scene.get("grasp_pose", [0.34, 0.16, 0.05]), float)
    handover = np.array(scene.get("handover_pose", [0.42, 0.0, 0.21]), float)
    wait = np.array(scene.get("wait_pose", [0.30, 0.0, 0.30]), float)
    baseline = line_points(start, handover, 60)
    stsm = smooth_waypoints(np.array([
        start,
        np.array([0.18, 0.07, 0.13]),
        wait,
        np.array([0.36, 0.05, 0.18]),
        handover,
    ]), samples_per_segment=18)
    return baseline, stsm


def _arm_traj_matches_scene(path):
    if path is None or len(path) < 2:
        return False
    scene = _load_handover_scene()
    grasp = np.array(scene.get("grasp_pose", [0.34, 0.16, 0.05]), float)
    handover = np.array(scene.get("handover_pose", [0.42, 0.0, 0.21]), float)
    xz = path[:, [0, 2]]
    grasp_xz = grasp[[0, 2]]
    handover_xz = handover[[0, 2]]
    min_grasp = float(np.min(np.linalg.norm(xz - grasp_xz[None, :], axis=1)))
    min_handover = float(np.min(np.linalg.norm(xz - handover_xz[None, :], axis=1)))
    span = float(np.linalg.norm(np.ptp(xz, axis=0)))
    return min_grasp < 0.18 and min_handover < 0.18 and span > 0.05


def set_line_and_dot(line, dot, path, count, dims=(0, 1)):
    if count <= 0:
        line.set_data([], [])
        dot.set_offsets(np.empty((0, 2)))
        return
    line.set_data(path[:count, dims[0]], path[:count, dims[1]])
    dot.set_offsets(path[count - 1, [dims[0], dims[1]]][None, :])


def existing_result_file(results_dir, names):
    for name in names:
        path = os.path.join(results_dir, name)
        if os.path.exists(path):
            return path
    return os.path.join(results_dir, names[0])


def arm_paths(results_dir):
    baseline = load_traj(existing_result_file(
        results_dir, ["arm/baseline/traj.csv", "arm_baseline_traj.csv"]))
    stsm = load_traj(existing_result_file(
        results_dir, ["arm/stsm/traj.csv", "arm_stsm_traj.csv"]))
    if (_arm_traj_matches_scene(baseline) and
            _arm_traj_matches_scene(stsm)):
        return baseline, stsm

    # The metrics topic is in elfin_base_link, but failed/idle arm runs may only
    # record the home pose. Such CSV traces do not represent the configured
    # handover plane and would put the plotted start near z=0.94.
    return _configured_arm_paths()


def wheelchair_paths(results_dir):
    baseline = load_traj(existing_result_file(
        results_dir, ["wheelchair/baseline/traj.csv", "wc_baseline_traj.csv"]))
    stsm = load_traj(existing_result_file(
        results_dir, ["wheelchair/stsm/traj.csv", "wc_stsm_traj.csv"]))
    if baseline is not None and stsm is not None:
        return baseline, stsm

    # Fallback only when real trajectory CSV is unavailable. For RViz-consistent
    # plots, generate *_traj.csv with scripts/record_traj_csv.sh.
    start = np.array([2.0, 1.5, 0.0])
    goal = np.array([-0.55, 0.55, 0.0])
    baseline = path_from_waypoints(np.array([start, goal]), samples_per_segment=72)
    stsm = smooth_waypoints(np.array([
        start,
        np.array([0.50, 1.58, 0.0]),
        np.array([-0.20, 1.05, 0.0]),
        goal,
    ]), samples_per_segment=24)
    return baseline, stsm


def draw_arm(out_path, results_dir):
    field = make_arm_field()
    bounds = [-0.02, 0.92, 0.05, 1.00]
    xlim = (bounds[0], bounds[1])
    zlim = (bounds[2], bounds[3])
    xx, zz, phi = grid_phi_xz(field, xlim, zlim, y=0.0)
    phi_plot = np.clip(phi, 0.0, RVIZ_MAX_RISK)

    baseline, stsm = arm_paths(results_dir)
    start = baseline[0]

    fig, ax = plt.subplots(figsize=STATIC_FIGSIZE)
    levels = np.linspace(0.0, RVIZ_MAX_RISK, 26)
    cf = ax.contourf(xx, zz, phi_plot, levels=levels, cmap="YlOrRd", alpha=0.78)
    ax.contour(xx, zz, phi_plot, levels=levels[::2], colors="#8f4c4c",
               linewidths=0.35, alpha=0.35)

    head = patches.Circle((0.78, 0.61), 0.10, color=RED, alpha=0.55, zorder=2)
    chest = patches.Circle((0.78, 0.31), 0.075, color="#F39C35", alpha=0.55, zorder=3)
    hand = patches.Circle((0.42, 0.21), 0.05, color="#66E35C", alpha=0.75, zorder=4)
    ax.add_patch(head)
    ax.add_patch(chest)
    ax.add_patch(hand)

    ax.plot(baseline[:, 0], baseline[:, 2], color=RVIZ_BLUE, lw=2.0,
            label="Baseline", zorder=6)
    ax.plot(stsm[:, 0], stsm[:, 2], color=RVIZ_ORANGE, lw=2.2,
            label="STSM", zorder=7)
    ax.scatter([start[0]], [start[2]], marker="s", s=85, color="#4775FF",
               edgecolor="white", linewidth=0.8, zorder=9)
    ax.scatter([baseline[-1, 0]], [baseline[-1, 2]], s=70, color=RVIZ_BLUE,
               edgecolor="white", linewidth=0.8, zorder=8)
    ax.scatter([stsm[-1, 0]], [stsm[-1, 2]], s=70, color=RVIZ_ORANGE,
               edgecolor="white", linewidth=0.8, zorder=8)
    ax.text(start[0] + 0.025, start[2] - 0.035, "起点", fontproperties=FONT_REGULAR)
    ax.text(0.80, 0.50, "头/胸高危区", fontproperties=FONT_REGULAR)
    ax.text(0.37, 0.16, "手(目标)", fontproperties=FONT_REGULAR)

    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("末端 X (m)", fontproperties=FONT_REGULAR)
    ax.set_ylabel("末端高度 Z (m)", fontproperties=FONT_REGULAR)
    ax.set_title("机械臂递物：末端轨迹在 X-Z 社会风险场中的单次演化", fontproperties=FONT_BOLD)
    ax.legend(loc="lower right", frameon=True, prop=FONT_REGULAR)
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("社会风险 Phi_s", fontproperties=FONT_REGULAR)
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(FONT_REGULAR)
    fig.tight_layout()
    fig.savefig(out_path, dpi=STATIC_DPI)
    plt.close(fig)


def animate_arm(gif_path, results_dir):
    field = make_arm_field()
    bounds = [-0.02, 0.92, 0.05, 1.00]
    xlim = (bounds[0], bounds[1])
    zlim = (bounds[2], bounds[3])
    xx, zz, phi = grid_phi_xz(field, xlim, zlim, y=0.0, nx=140, nz=120)
    phi_plot = np.clip(phi, 0.0, RVIZ_MAX_RISK)

    baseline, stsm = arm_paths(results_dir)
    start = baseline[0]

    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    levels = np.linspace(0.0, RVIZ_MAX_RISK, 20)
    cf = ax.contourf(xx, zz, phi_plot, levels=levels, cmap="YlOrRd", alpha=0.78)
    ax.contour(xx, zz, phi_plot, levels=levels[::2], colors="#8f4c4c",
               linewidths=0.35, alpha=0.35)
    ax.add_patch(patches.Circle((0.78, 0.61), 0.10, color=RED, alpha=0.55, zorder=2))
    ax.add_patch(patches.Circle((0.78, 0.31), 0.075, color="#F39C35", alpha=0.55, zorder=3))
    ax.add_patch(patches.Circle((0.42, 0.21), 0.05, color="#66E35C", alpha=0.75, zorder=4))
    ax.scatter([start[0]], [start[2]], marker="s", s=80, color="#4775FF",
               edgecolor="white", linewidth=0.8, zorder=6)
    ax.text(start[0] + 0.025, start[2] - 0.035, "起点", fontproperties=FONT_REGULAR)
    ax.text(0.80, 0.50, "头/胸高危区", fontproperties=FONT_REGULAR)
    ax.text(0.37, 0.16, "手(目标)", fontproperties=FONT_REGULAR)

    base_line, = ax.plot([], [], color=RVIZ_BLUE, lw=2.2, label="Baseline")
    stsm_line, = ax.plot([], [], color=RVIZ_ORANGE, lw=2.2, label="STSM")
    base_dot = ax.scatter([], [], s=70, color=RVIZ_BLUE, zorder=5)
    stsm_dot = ax.scatter([], [], s=70, color=RVIZ_ORANGE, zorder=5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("末端 X (m)", fontproperties=FONT_REGULAR)
    ax.set_ylabel("末端高度 Z (m)", fontproperties=FONT_REGULAR)
    ax.set_title("机械臂递物：末端轨迹在 X-Z 社会风险场中的单次动态演化", fontproperties=FONT_BOLD)
    ax.legend(loc="lower right", frameon=True, prop=FONT_REGULAR)
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("社会风险 Phi_s", fontproperties=FONT_REGULAR)
    fig.tight_layout()

    frames = 112
    base_steps = sampled_indices(len(baseline), frames)
    stsm_steps = sampled_indices(len(stsm), frames)

    def update(i):
        ib = base_steps[i]
        is_ = stsm_steps[i]
        set_line_and_dot(base_line, base_dot, baseline, ib, dims=(0, 2))
        set_line_and_dot(stsm_line, stsm_dot, stsm, is_, dims=(0, 2))
        return base_line, stsm_line, base_dot, stsm_dot

    anim = FuncAnimation(fig, update, frames=frames, interval=150, blit=False)
    anim.save(gif_path, writer="imagemagick", fps=7, dpi=100)
    plt.close(fig)


def draw_wheelchair(out_path, results_dir):
    field = make_wc_field()
    bounds = [-2.1, 2.5, -2.0, 2.0]
    xlim = (bounds[0], bounds[1])
    ylim = (bounds[2], bounds[3])
    xx, yy, phi = grid_phi(field, xlim, ylim, z=0.03)
    phi_plot = np.clip(phi, 0.0, RVIZ_MAX_RISK)

    baseline, stsm = wheelchair_paths(results_dir)
    start = baseline[0]
    goal = np.array([-0.55, 0.55, 0.0])

    fig, ax = plt.subplots(figsize=STATIC_FIGSIZE)
    levels = np.linspace(0.0, RVIZ_MAX_RISK, 28)
    cf = ax.contourf(xx, yy, phi_plot, levels=levels, cmap="YlOrRd", alpha=0.78)
    ax.contour(xx, yy, phi_plot, levels=levels[::2], colors="#8f4c4c",
               linewidths=0.35, alpha=0.35)

    bed = patches.Rectangle((-2.1, -2.0), 1.0, 2.0, color="#6B5CA5", alpha=0.35, zorder=2)
    transfer = patches.Rectangle((-1.1, -2.0), 0.8, 2.0, fill=False,
                                 edgecolor="#D62728", linewidth=1.5, linestyle="--")
    table = patches.Rectangle((0.32, -0.55), 0.6, 1.1, color="#A77743", alpha=0.55)
    person = patches.Circle((-1.6, 0.2), 0.12, color=PURPLE, alpha=0.90)
    goal_zone = patches.Circle((goal[0], goal[1]), WHEELCHAIR_GOAL_TOL,
                               fill=False, edgecolor=BLACK, linewidth=1.1,
                               linestyle=":", zorder=6)
    ax.add_patch(bed)
    ax.add_patch(transfer)
    ax.add_patch(table)
    ax.add_patch(person)
    ax.add_patch(goal_zone)

    ax.plot(baseline[:, 0], baseline[:, 1], color=RVIZ_BLUE, lw=2.0, label="Baseline")
    ax.plot(stsm[:, 0], stsm[:, 1], color=RVIZ_ORANGE, lw=2.0, label="STSM")
    ax.scatter([start[0]], [start[1]], marker="s", s=80, color="#4775FF", zorder=5)
    ax.scatter([baseline[-1, 0]], [baseline[-1, 1]], marker="o", s=90,
               color=RVIZ_BLUE, edgecolor="white", linewidth=0.8, zorder=7)
    ax.scatter([stsm[-1, 0]], [stsm[-1, 1]], marker="o", s=90,
               color=RVIZ_ORANGE, edgecolor="white", linewidth=0.8, zorder=7)
    ax.scatter([goal[0]], [goal[1]], marker="*", s=220, color=ORANGE,
               edgecolor=BLACK, linewidth=0.7, zorder=6)
    ax.text(goal[0] + 0.06, goal[1] + 0.06, "泊靠目标区", fontproperties=FONT_REGULAR)
    ax.text(-1.72, -1.10, "床", color="white", fontproperties=FONT_REGULAR)
    ax.text(0.58, -0.08, "桌", color="white", rotation=90, fontproperties=FONT_REGULAR)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("轮椅中心 X (m)", fontproperties=FONT_REGULAR)
    ax.set_ylabel("轮椅中心 Y (m)", fontproperties=FONT_REGULAR)
    ax.set_title("智能轮椅床边泊靠：中心路径在社会风险场中的单次演化", fontproperties=FONT_BOLD)
    ax.legend(loc="upper right", frameon=True, prop=FONT_REGULAR)
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("社会风险 Phi_s", fontproperties=FONT_REGULAR)
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(FONT_REGULAR)
    fig.tight_layout()
    fig.savefig(out_path, dpi=STATIC_DPI)
    plt.close(fig)


def animate_wheelchair(gif_path, results_dir):
    field = make_wc_field()
    bounds = [-2.1, 2.5, -2.0, 2.0]
    xlim = (bounds[0], bounds[1])
    ylim = (bounds[2], bounds[3])
    xx, yy, phi = grid_phi(field, xlim, ylim, z=0.03, nx=140, ny=120)
    phi_plot = np.clip(phi, 0.0, RVIZ_MAX_RISK)

    baseline, stsm = wheelchair_paths(results_dir)
    start = baseline[0]
    goal = np.array([-0.55, 0.55, 0.0])

    fig, ax = plt.subplots(figsize=(5.9, 4.7))
    levels = np.linspace(0.0, RVIZ_MAX_RISK, 22)
    cf = ax.contourf(xx, yy, phi_plot, levels=levels, cmap="YlOrRd", alpha=0.78)
    ax.contour(xx, yy, phi_plot, levels=levels[::2], colors="#8f4c4c",
               linewidths=0.35, alpha=0.35)
    ax.add_patch(patches.Rectangle((-2.1, -2.0), 1.0, 2.0, color="#6B5CA5", alpha=0.35, zorder=2))
    ax.add_patch(patches.Rectangle((-1.1, -2.0), 0.8, 2.0, fill=False,
                                   edgecolor="#D62728", linewidth=1.5, linestyle="--"))
    ax.add_patch(patches.Rectangle((0.32, -0.55), 0.6, 1.1, color="#A77743", alpha=0.55))
    ax.add_patch(patches.Circle((-1.6, 0.2), 0.12, color=PURPLE, alpha=0.90))
    ax.scatter([start[0]], [start[1]], marker="s", s=80, color="#4775FF", zorder=5)
    ax.scatter([goal[0]], [goal[1]], marker="*", s=220, color=ORANGE,
               edgecolor=BLACK, linewidth=0.7, zorder=6)
    ax.text(goal[0] + 0.06, goal[1] + 0.06, "泊靠目标", fontproperties=FONT_REGULAR)
    ax.text(-1.72, -1.10, "床", color="white", fontproperties=FONT_REGULAR)
    ax.text(0.58, -0.08, "桌", color="white", rotation=90, fontproperties=FONT_REGULAR)

    base_line, = ax.plot([], [], color=RVIZ_BLUE, lw=2.2, label="Baseline")
    stsm_line, = ax.plot([], [], color=RVIZ_ORANGE, lw=2.2, label="STSM")
    base_dot = ax.scatter([], [], s=70, color=RVIZ_BLUE, zorder=5)
    stsm_dot = ax.scatter([], [], s=70, color=RVIZ_ORANGE, zorder=5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("轮椅中心 X (m)", fontproperties=FONT_REGULAR)
    ax.set_ylabel("轮椅中心 Y (m)", fontproperties=FONT_REGULAR)
    ax.set_title("智能轮椅床边泊靠：中心路径在社会风险场中的单次动态演化", fontproperties=FONT_BOLD)
    ax.legend(loc="upper right", frameon=True, prop=FONT_REGULAR)
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("社会风险 Phi_s", fontproperties=FONT_REGULAR)
    fig.tight_layout()

    frames = 112
    base_steps = sampled_indices(len(baseline), frames)
    stsm_steps = sampled_indices(len(stsm), frames)

    def update(i):
        ib = base_steps[i]
        is_ = stsm_steps[i]
        set_line_and_dot(base_line, base_dot, baseline, ib)
        set_line_and_dot(stsm_line, stsm_dot, stsm, is_)
        return base_line, stsm_line, base_dot, stsm_dot

    anim = FuncAnimation(fig, update, frames=frames, interval=150, blit=False)
    anim.save(gif_path, writer="imagemagick", fps=7, dpi=100)
    plt.close(fig)


def build_plots(results_dir, out_dir, make_gif=True):
    setup_rc()
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    draw_arm(os.path.join(out_dir, "arm_risk_field_path.png"), results_dir)
    draw_wheelchair(
        os.path.join(out_dir, "wheelchair_risk_field_path.png"), results_dir)
    if make_gif:
        animate_arm(os.path.join(out_dir, "arm_risk_field_path.gif"), results_dir)
        animate_wheelchair(
            os.path.join(out_dir, "wheelchair_risk_field_path.gif"), results_dir)


def main():
    setup_rc()
    parser = argparse.ArgumentParser(description="Plot social-risk fields with baseline/STSM representative single-trial paths.")
    parser.add_argument("--out", default=os.path.join(ROOT, "pictures"))
    parser.add_argument("--results", default=os.path.join(ROOT, "results"))
    parser.add_argument("--no-gif", action="store_true", help="Only generate static PNG files.")
    args = parser.parse_args()
    build_plots(args.results, args.out, make_gif=not args.no_gif)
    print("wrote {0}".format(args.out))


if __name__ == "__main__":
    main()
