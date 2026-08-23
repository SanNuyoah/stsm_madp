#!/usr/bin/env python3
from __future__ import print_function

import ast
import argparse
import csv
import json
import math
import os
import sys

sys.dont_write_bytecode = True

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(ROOT, "results")
DEFAULT_RUN_DIR = os.path.join(RESULTS_DIR, "run")
DEFAULT_FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
AUDIT_NAME = "visualization_audit.json"
ROBOTS = ("arm", "wheelchair")
CJK_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
ARM_RISK_MAX = 4.0


def configure_plot_fonts():
    import matplotlib
    from matplotlib import font_manager

    # Older Matplotlib releases call ``copy()`` on ``fontproperties`` and
    # crash when it is explicitly None.  Always return a FontProperties
    # instance; use the bundled CJK font when the container provides it.
    cjk_font = font_manager.FontProperties()
    if os.path.isfile(CJK_FONT_PATH):
        try:
            font_manager.fontManager.addfont(CJK_FONT_PATH)
        except Exception:
            pass
        cjk_font = font_manager.FontProperties(fname=CJK_FONT_PATH)
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Noto Sans CJK SC", "Noto Sans CJK JP",
            "Noto Sans CJK TC", "Droid Sans Fallback",
            "DejaVu Sans"],
        "axes.unicode_minus": False,
    })
    return cjk_font


def relpath(path):
    try:
        return os.path.relpath(path, ROOT)
    except Exception:
        return path


def stsm_dir(run_dir, robot):
    return os.path.join(run_dir, robot, "stsm")


def parse_value(value, default=None):
    if default is None:
        default = []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except Exception:
            try:
                return ast.literal_eval(text)
            except Exception:
                return default
    return value if value is not None else default


def load_json(path, default=None):
    if default is None:
        default = {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def as_float(value):
    try:
        if value in (None, ""):
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    except Exception:
        return None


def load_xy_csv(path):
    points = []
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return points
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                x = as_float(row.get("x"))
                y = as_float(row.get("y"))
                if x is None or y is None:
                    continue
                points.append([x, y])
    except Exception:
        return []
    return points


def load_xy_csv_with_rows(path):
    points = []
    rows = []
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return points, rows
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                x = as_float(row.get("x"))
                y = as_float(row.get("y"))
                if x is None or y is None:
                    continue
                points.append([x, y])
                rows.append(row)
    except Exception:
        return [], []
    return points, rows


def load_xz_csv(path):
    points = []
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return points
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                x = as_float(row.get("x"))
                z = as_float(row.get("z"))
                if x is None or z is None:
                    continue
                points.append([x, z])
    except Exception:
        return []
    return points


def load_csv_rows(path):
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return []
    try:
        with open(path) as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def first_existing_trajectory(base_dir, names):
    for name in names:
        path = os.path.join(base_dir, name)
        points = load_xy_csv(path)
        if len(points) >= 2:
            return path, points
    return "", []


def metrics_failed(base_dir):
    metrics = load_json(os.path.join(base_dir, "metrics.json"), {})
    return str(metrics.get("execution_status", "")).strip().lower() == "failed"


def point2(value):
    value = parse_value(value, [])
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return [float(value[0]), float(value[1])]
    except Exception:
        return None


def points_from_record(record):
    for key in ("waypoints", "centerline", "refined_waypoints",
                "raw_topology_waypoints"):
        values = parse_value(record.get(key, []), [])
        points = []
        for value in values:
            point = point2(value)
            if point is not None:
                points.append(point)
        if len(points) >= 2:
            return points
    return []


def graph_nodes(graph):
    nodes = []
    for item in parse_value(graph.get("nodes", []), []):
        if not isinstance(item, dict):
            continue
        point = point2(item.get("point", item.get("p2", [])))
        if point is None:
            continue
        nodes.append({
            "id": str(item.get("id", "")),
            "kind": str(item.get("kind", item.get("node_type", ""))).lower(),
            "point": point,
        })
    return nodes


def graph_edges(graph, node_by_id):
    out = []
    edges = parse_value(graph.get("edges", {}), {})
    if isinstance(edges, dict):
        for source_id, items in edges.items():
            for item in parse_value(items, []):
                if not isinstance(item, dict):
                    continue
                target_id = str(item.get("to", item.get("target", "")))
                cells = parse_value(item.get("points", item.get("path", [])), [])
                path = [point2(p) for p in cells]
                path = [p for p in path if p is not None]
                if len(path) >= 2:
                    out.append(path)
                elif source_id in node_by_id and target_id in node_by_id:
                    out.append([node_by_id[source_id], node_by_id[target_id]])
    elif isinstance(edges, list):
        for item in edges:
            if not isinstance(item, dict):
                continue
            source_id = str(item.get("from", item.get("source", "")))
            target_id = str(item.get("to", item.get("target", "")))
            path = [point2(p) for p in parse_value(item.get("points", []), [])]
            path = [p for p in path if p is not None]
            if len(path) >= 2:
                out.append(path)
            elif source_id in node_by_id and target_id in node_by_id:
                out.append([node_by_id[source_id], node_by_id[target_id]])
    return out


def selected_candidate(candidates):
    for item in candidates:
        if isinstance(item, dict) and bool(item.get("selected", False)):
            return item
    feasible = [
        item for item in candidates
        if isinstance(item, dict) and str(item.get("candidate_status", "")).lower() == "feasible"
    ]
    return feasible[0] if feasible else (candidates[0] if candidates else {})


def fallback_paths(base_dir):
    candidates = load_json(os.path.join(base_dir, "candidate_corridors.json"), [])
    candidates = parse_value(candidates.get("candidates", []) if isinstance(candidates, dict) else candidates, [])
    paths = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        points = points_from_record(item)
        if len(points) >= 2:
            paths.append({
                "label": str(item.get("corridor_id", item.get("label", "candidate"))),
                "points": points,
                "selected": bool(item.get("selected", False)),
            })
    if paths:
        return paths

    routes = load_json(os.path.join(base_dir, "morse_routes.json"), [])
    routes = parse_value(routes.get("routes", []) if isinstance(routes, dict) else routes, [])
    graph = load_json(os.path.join(base_dir, "topology_graph.json"), {})
    node_by_id = {node["id"]: node["point"] for node in graph_nodes(graph)}
    for item in routes:
        if not isinstance(item, dict):
            continue
        sequence = parse_value(item.get("node_sequence", item.get("critical_sequence", [])), [])
        points = [node_by_id[str(node_id)] for node_id in sequence if str(node_id) in node_by_id]
        if len(points) >= 2:
            paths.append({
                "label": str(item.get("route_id", "route")),
                "points": points,
                "selected": False,
            })
    return paths


def set_bounds(ax, points):
    if not points:
        return
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    padx = max((max(xs) - min(xs)) * 0.12, 0.05)
    pady = max((max(ys) - min(ys)) * 0.12, 0.05)
    ax.set_xlim(min(xs) - padx, max(xs) + padx)
    ax.set_ylim(min(ys) - pady, max(ys) + pady)


def plot_morse_topology(robot, base_dir, figures_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = os.path.join(figures_dir, "{}_morse_topology.png".format(robot))
    graph_path = os.path.join(base_dir, "topology_graph.json")
    graph = load_json(graph_path, {})
    nodes = graph_nodes(graph if isinstance(graph, dict) else {})
    node_by_id = {node["id"]: node["point"] for node in nodes}
    edges = graph_edges(graph if isinstance(graph, dict) else {}, node_by_id)
    candidates = load_json(os.path.join(base_dir, "candidate_corridors.json"), [])
    candidates = parse_value(candidates.get("candidates", []) if isinstance(candidates, dict) else candidates, [])
    selected = selected_candidate(candidates)
    selected_path = points_from_record(selected) if isinstance(selected, dict) else []
    paths = []
    source = "topology_graph.json"
    if not nodes and not edges:
        paths = fallback_paths(base_dir)
        source = "candidate_corridors.json or morse_routes.json"
        if not selected_path:
            selected_items = [p for p in paths if p.get("selected")]
            selected_path = (selected_items[0]["points"] if selected_items
                             else (paths[0]["points"] if paths else []))

    if not nodes and not edges and not paths and not selected_path:
        return {
            "status": "missing_data",
            "path": relpath(out_path),
            "reason": "missing topology graph nodes/edges and fallback paths",
        }

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    ax.set_title("{} Morse topology".format(robot))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.8)

    all_points = []
    for edge in edges:
        xs = [p[0] for p in edge]
        ys = [p[1] for p in edge]
        all_points.extend(edge)
        ax.plot(xs, ys, color="#8a8a8a", linewidth=1.1, alpha=0.55,
                label="topology edge")
    if paths:
        for item in paths:
            points = item["points"]
            all_points.extend(points)
            ax.plot([p[0] for p in points], [p[1] for p in points],
                    color="#0072B2", linewidth=1.5, alpha=0.45,
                    label="fallback route")
    if selected_path:
        all_points.extend(selected_path)
        ax.plot([p[0] for p in selected_path], [p[1] for p in selected_path],
                color="#D55E00", linewidth=3.0, alpha=0.95,
                label="selected route")

    label_seen = set()
    for node in nodes:
        x, y = node["point"]
        all_points.append(node["point"])
        kind = node["kind"]
        if kind == "start":
            marker, color, size, label = "o", "#ffffff", 90, "start"
        elif kind == "goal":
            marker, color, size, label = "*", "#009E73", 150, "goal"
        elif kind in ("minimum", "minima"):
            marker, color, size, label = "v", "#F0E442", 85, "minimum"
        elif kind == "saddle":
            marker, color, size, label = "x", "#0072B2", 90, "saddle"
        else:
            marker, color, size, label = "s", "#999999", 55, kind or "node"
        shown_label = label if label not in label_seen else "_nolegend_"
        label_seen.add(label)
        ax.scatter([x], [y], c=color, marker=marker, s=size,
                   edgecolors="#000000" if marker != "x" else None,
                   linewidths=1.0, label=shown_label, zorder=4)
    if not nodes and selected_path:
        start, goal = selected_path[0], selected_path[-1]
        all_points.extend([start, goal])
        ax.scatter([start[0]], [start[1]], c="#ffffff", marker="o", s=90,
                   edgecolors="#000000", label="start", zorder=4)
        ax.scatter([goal[0]], [goal[1]], c="#009E73", marker="*", s=150,
                   edgecolors="#000000", label="goal", zorder=4)

    handles, labels = ax.get_legend_handles_labels()
    unique_handles, unique_labels, seen = [], [], set()
    for handle, label in zip(handles, labels):
        if label and label != "_nolegend_" and label not in seen:
            unique_handles.append(handle)
            unique_labels.append(label)
            seen.add(label)
    if unique_handles:
        ax.legend(unique_handles, unique_labels, loc="best", fontsize=8)
    ax.text(0.01, 0.01, "source: {}".format(source), transform=ax.transAxes,
            fontsize=8, color="#555555", va="bottom")
    set_bounds(ax, all_points)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return {"status": "generated", "path": relpath(out_path), "reason": ""}


def filter_status(item, selected_ids):
    candidate_id = str(item.get("candidate_id", item.get("corridor_id", "")))
    if candidate_id in selected_ids or bool(item.get("selected", False)):
        return "selected"
    status = str(item.get("candidate_status", item.get("filter_status", ""))).lower()
    if status in ("feasible", "recoverable", "invalid"):
        return status
    if bool(item.get("geometry_valid", True)) and bool(item.get("manifold_valid", True)):
        return "feasible"
    return "invalid"


def plot_candidate_filter(robot, base_dir, figures_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = os.path.join(figures_dir, "{}_candidate_filter.png".format(robot))
    report = load_json(os.path.join(base_dir, "candidate_filter_report.json"), [])
    report = parse_value(report.get("candidates", []) if isinstance(report, dict) else report, [])
    candidates = load_json(os.path.join(base_dir, "candidate_corridors.json"), [])
    candidates = parse_value(candidates.get("candidates", []) if isinstance(candidates, dict) else candidates, [])
    selected_ids = set()
    for item in candidates:
        if isinstance(item, dict) and bool(item.get("selected", False)):
            selected_ids.add(str(item.get("corridor_id", item.get("candidate_id", ""))))
    if not report and candidates:
        report = candidates
    if not report:
        return {
            "status": "missing_data",
            "path": relpath(out_path),
            "reason": "missing candidate_filter_report.json and candidate_corridors.json",
        }

    colors = {
        "feasible": "#2ca02c",
        "selected": "#d62728",
        "recoverable": "#f0c419",
        "invalid": "#7f7f7f",
    }
    labels = []
    risks = []
    clearances = []
    statuses = []
    for index, item in enumerate(report):
        if not isinstance(item, dict):
            continue
        labels.append(str(item.get("candidate_id", item.get("corridor_id", index + 1))))
        risks.append(float(item.get("risk_value", item.get("risk", 0.0)) or 0.0))
        clearances.append(float(item.get("clearance_value", item.get("clearance", 0.0)) or 0.0))
        statuses.append(filter_status(item, selected_ids))
    if not labels:
        return {
            "status": "missing_data",
            "path": relpath(out_path),
            "reason": "candidate records have no plottable status",
        }

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    xs = list(range(len(labels)))
    bar_colors = [colors.get(status, colors["invalid"]) for status in statuses]
    ax.bar(xs, risks, color=bar_colors, alpha=0.85, label="risk")
    ax.plot(xs, clearances, color="#0072B2", linewidth=1.8,
            marker="o", markersize=3.0, label="clearance")
    step = max(1, len(labels) // 12)
    ax.set_xticks(xs[::step])
    ax.set_xticklabels([labels[i] for i in xs[::step]], rotation=35,
                       ha="right", fontsize=7)
    ax.set_title("{} candidate filter status".format(robot))
    ax.set_xlabel("candidate")
    ax.set_ylabel("value")
    from matplotlib.patches import Patch
    legend = [Patch(facecolor=colors[name], label=name)
              for name in ("feasible", "selected", "recoverable", "invalid")]
    ax.legend(handles=legend, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return {"status": "generated", "path": relpath(out_path), "reason": ""}


def plot_execution(robot, base_dir, figures_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = os.path.join(figures_dir, "{}_execution.png".format(robot))
    failed = metrics_failed(base_dir)
    trajectory_path, trajectory_points = first_existing_trajectory(
        base_dir, ("trajectory.csv", "mpc_executed_trajectory.csv"))

    if not trajectory_points:
        return {
            "status": "missing_data",
            "path": relpath(out_path),
            "reason": "missing trajectory.csv or mpc_executed_trajectory.csv",
        }

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    ax.set_title("{} execution result".format(robot))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.8)

    all_points = []
    all_points.extend(trajectory_points)
    ax.plot([p[0] for p in trajectory_points],
            [p[1] for p in trajectory_points],
            color="#0072B2", linewidth=2.6, alpha=0.95,
            label="execution trajectory")

    start = trajectory_points[0]
    goal = trajectory_points[-1]
    ax.scatter([start[0]], [start[1]], c="#ffffff", marker="o", s=90,
               edgecolors="#000000", linewidths=1.0, label="start", zorder=4)
    ax.scatter([goal[0]], [goal[1]], c="#009E73", marker="*", s=155,
               edgecolors="#000000", linewidths=1.0, label="target", zorder=4)
    if failed:
        ax.text(0.03, 0.94, "failed execution", transform=ax.transAxes,
                color="#b00020", fontsize=13, weight="bold")
    set_bounds(ax, all_points)
    ax.legend(loc="best", fontsize=8)
    ax.text(0.01, 0.01, "source: {}".format(relpath(trajectory_path)),
            transform=ax.transAxes, fontsize=8, color="#555555", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return {
        "status": "experiment_failed" if failed else "generated",
        "generated": True,
        "path": relpath(out_path),
        "source": relpath(trajectory_path),
        "reason": "metrics.json execution_status=failed" if failed else "",
    }


def _arm_risk_value_xz(x, z):
    import math
    head = 3.0 * math.exp(-(((x - 0.78) ** 2 + (z - 0.61) ** 2) /
                            (2.0 * 0.13 ** 2)))
    chest = 1.6 * math.exp(-(((x - 0.78) ** 2 + (z - 0.31) ** 2) /
                             (2.0 * 0.18 ** 2)))
    hand = 0.3 * math.exp(-(((x - 0.42) ** 2 + (z - 0.21) ** 2) /
                            (2.0 * 0.10 ** 2)))
    prox = 0.35 * max(0.0, x)
    table = 0.35 * math.exp(-(((x - 0.55) ** 2 + (z - 0.0) ** 2) /
                              (2.0 * 0.25 ** 2)))
    return head + chest + hand + prox + table


def _configured_arm_reference_path():
    return [
        [0.34, 0.05],
        [0.18, 0.13],
        [0.30, 0.30],
        [0.36, 0.18],
        [0.42, 0.21],
    ]


def plot_arm_trajectory_on_risk_field(run_dir, figures_dir):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patches

    cjk_font = configure_plot_fonts()
    out_path = os.path.join(figures_dir, "arm_trajectory_on_risk_field.png")
    baseline_dir = os.path.join(run_dir, "arm", "baseline")
    stsm_base = stsm_dir(run_dir, "arm")

    baseline = load_xz_csv(os.path.join(baseline_dir, "trajectory.csv"))
    stsm = load_xz_csv(os.path.join(stsm_base, "trajectory.csv"))

    if len(stsm) < 2 and len(baseline) < 2:
        return {
            "status": "missing_data",
            "path": relpath(out_path),
            "reason": "missing arm trajectory data",
        }

    xlim = (-0.02, 0.92)
    zlim = (0.05, 1.00)
    xs = np.linspace(xlim[0], xlim[1], 260)
    zs = np.linspace(zlim[0], zlim[1], 220)
    xx, zz = np.meshgrid(xs, zs)
    phi = np.zeros_like(xx)
    for i in range(zz.shape[0]):
        for j in range(xx.shape[1]):
            phi[i, j] = _arm_risk_value_xz(float(xx[i, j]), float(zz[i, j]))
    phi = np.clip(phi, 0.0, ARM_RISK_MAX)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    levels = np.linspace(0.0, ARM_RISK_MAX, 26)
    cf = ax.contourf(xx, zz, phi, levels=levels, cmap="YlOrRd", alpha=0.78)
    ax.contour(xx, zz, phi, levels=levels[::2], colors="#8f4c4c",
               linewidths=0.35, alpha=0.35)

    ax.add_patch(patches.Circle((0.78, 0.61), 0.10, color="#C44E52",
                                alpha=0.55, zorder=2))
    ax.add_patch(patches.Circle((0.78, 0.31), 0.075, color="#F39C35",
                                alpha=0.55, zorder=3))
    ax.add_patch(patches.Circle((0.42, 0.21), 0.05, color="#66E35C",
                                alpha=0.75, zorder=4))

    if len(baseline) >= 2:
        b = np.asarray(baseline, float)
        ax.plot(b[:, 0], b[:, 1], color="#0072B2", lw=2.0,
                label="Baseline", zorder=6)
    if len(stsm) >= 2:
        s = np.asarray(stsm, float)
        ax.plot(s[:, 0], s[:, 1], color="#D55E00", lw=2.2,
                label="STSM", zorder=7)

    marker_path = np.asarray(baseline or stsm, float)
    if len(marker_path) > 0:
        start = marker_path[0]
        ax.scatter([start[0]], [start[1]], marker="s", s=85,
                   color="#4775FF", edgecolor="white", linewidth=0.8,
                   zorder=9)
        ax.text(start[0] + 0.025, start[1] - 0.035, "起点",
                fontproperties=cjk_font)
    ax.scatter([0.42], [0.21], s=70, color="#D55E00",
               edgecolor="white", linewidth=0.8, zorder=9)
    ax.text(0.80, 0.50, "头/胸高危区", fontproperties=cjk_font)
    ax.text(0.37, 0.16, "手(目标)", fontproperties=cjk_font)

    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("末端 X (m)", fontproperties=cjk_font)
    ax.set_ylabel("末端高度 Z (m)", fontproperties=cjk_font)
    ax.set_title("机械臂递物：末端轨迹在 X-Z 社会风险场中的单次演化",
                 fontproperties=cjk_font)
    ax.legend(loc="lower right", frameon=True,
              prop=cjk_font if cjk_font else None)
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("社会风险 Phi_s", fontproperties=cjk_font)
    if cjk_font:
        for label in cbar.ax.get_yticklabels():
            label.set_fontproperties(cjk_font)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(cjk_font)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return {"status": "generated", "path": relpath(out_path), "reason": ""}


def _row_risk_value(row):
    return first_number(row, (
        "phi_total", "footprint_gate_risk", "phi_max_point",
        "risk", "risk_value"))


def _selected_corridor_points(base_dir):
    selected = load_json(os.path.join(base_dir, "selected_corridor.json"), {})
    points = points_from_record(selected) if isinstance(selected, dict) else []
    if len(points) >= 2:
        return points, str(selected.get("corridor_id", selected.get(
            "label", "selected corridor")))
    candidates = load_json(os.path.join(base_dir, "candidate_corridors.json"), [])
    candidates = parse_value(
        candidates.get("candidates", []) if isinstance(candidates, dict)
        else candidates, [])
    item = selected_candidate(candidates)
    points = points_from_record(item) if isinstance(item, dict) else []
    if len(points) >= 2:
        return points, str(item.get("corridor_id", item.get(
            "label", "selected corridor")))
    return [], ""


def plot_wheelchair_trajectory_on_risk_field(run_dir, figures_dir):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patches
    import matplotlib.tri as mtri

    cjk_font = configure_plot_fonts()
    out_path = os.path.join(
        figures_dir, "wheelchair_trajectory_on_risk_field.png")
    baseline_dir = os.path.join(run_dir, "wheelchair", "baseline")
    stsm_base = stsm_dir(run_dir, "wheelchair")

    baseline, baseline_rows = load_xy_csv_with_rows(
        os.path.join(baseline_dir, "trajectory.csv"))
    stsm, stsm_rows = load_xy_csv_with_rows(
        os.path.join(stsm_base, "trajectory.csv"))

    if len(stsm) < 2 and len(baseline) < 2:
        return {
            "status": "missing_data",
            "path": relpath(out_path),
            "reason": "missing wheelchair trajectory.csv data",
        }

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.8)

    risk_points = []
    risk_values = []
    for points, rows in ((baseline, baseline_rows), (stsm, stsm_rows)):
        for point, row in zip(points, rows):
            value = _row_risk_value(row)
            if value is None:
                continue
            risk_points.append(point)
            risk_values.append(float(value))
    if len(risk_points) >= 8:
        rp = np.asarray(risk_points, float)
        rv = np.asarray(risk_values, float)
        try:
            tri = mtri.Triangulation(rp[:, 0], rp[:, 1])
            levels = np.linspace(float(np.min(rv)), float(np.max(rv)), 18)
            if levels[-1] > levels[0]:
                cf = ax.tricontourf(tri, rv, levels=levels, cmap="YlOrRd",
                                    alpha=0.45, zorder=1)
                cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("社会风险", fontproperties=cjk_font)
                if cjk_font:
                    for label in cbar.ax.get_yticklabels():
                        label.set_fontproperties(cjk_font)
        except Exception:
            pass

    corridor_points, corridor_label = _selected_corridor_points(stsm_base)
    if len(corridor_points) >= 2:
        cp = np.asarray(corridor_points, float)
        ax.plot(cp[:, 0], cp[:, 1], color="#000000", linewidth=1.8,
                linestyle="--", alpha=0.72, zorder=5,
                label="selected STSM corridor")
        mid = cp[len(cp) // 2]
        ax.text(mid[0], mid[1], corridor_label, fontsize=8,
                color="#000000", fontproperties=cjk_font,
                bbox=dict(facecolor="white", edgecolor="#888888",
                          alpha=0.70, boxstyle="round,pad=0.18"))

    if len(baseline) >= 2:
        b = np.asarray(baseline, float)
        ax.plot(b[:, 0], b[:, 1], color="#0072B2", lw=2.0,
                label="Baseline", zorder=6)
    if len(stsm) >= 2:
        s = np.asarray(stsm, float)
        ax.plot(s[:, 0], s[:, 1], color="#D55E00", lw=2.2,
                label="STSM", zorder=7)

    goal = (-0.55, 0.55)
    ax.add_patch(patches.Circle(goal, 0.08, facecolor="#009E73",
                                edgecolor="#005A32", linewidth=1.8,
                                alpha=0.24, zorder=4,
                                label="goal tolerance r=0.08 m"))
    ax.scatter([goal[0]], [goal[1]], marker="*", s=160,
               color="#009E73", edgecolor="white", linewidth=0.9, zorder=9)
    ax.text(goal[0] + 0.05, goal[1] + 0.05, "goal区域",
            fontproperties=cjk_font)

    marker_path = np.asarray(baseline or stsm, float)
    if len(marker_path) > 0:
        start = marker_path[0]
        ax.scatter([start[0]], [start[1]], marker="s", s=85,
                   color="#4775FF", edgecolor="white", linewidth=0.8,
                   zorder=9)
        ax.text(start[0] + 0.05, start[1] + 0.05, "起点",
                fontproperties=cjk_font)

    all_points = []
    all_points.extend(baseline)
    all_points.extend(stsm)
    all_points.extend(corridor_points)
    all_points.append([goal[0], goal[1]])
    set_bounds(ax, all_points)
    ax.set_xlabel("X (m)", fontproperties=cjk_font)
    ax.set_ylabel("Y (m)", fontproperties=cjk_font)
    ax.set_title("轮椅导航：实际轨迹在社会风险场中的演化",
                 fontproperties=cjk_font)
    ax.legend(loc="best", frameon=True,
              prop=cjk_font if cjk_font else None)
    ax.text(0.01, 0.01, "source: wheelchair/baseline|stsm/trajectory.csv",
            transform=ax.transAxes, fontsize=8, color="#555555", va="bottom")
    if cjk_font:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(cjk_font)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return {"status": "generated", "path": relpath(out_path), "reason": ""}


def plot_mpc_clearance(robot, base_dir, figures_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = os.path.join(figures_dir, "{}_mpc_clearance_time.png".format(robot))
    rollout_path = os.path.join(base_dir, "mpc_rollout_log.csv")
    diagnostics = load_json(os.path.join(base_dir, "mpc_diagnostics.json"), {})
    rows = load_csv_rows(rollout_path)
    if not rows:
        return {
            "status": "missing_data",
            "path": relpath(out_path),
            "reason": "missing mpc_rollout_log.csv time series",
        }

    xs = []
    ys = []
    for index, row in enumerate(rows):
        clearance = as_float(row.get("minimum_clearance"))
        if clearance is None:
            clearance = as_float(row.get("manifold_clearance"))
        if clearance is None:
            clearance = as_float(row.get("boundary_distance"))
        if clearance is None:
            continue
        time_value = as_float(row.get("global_step"))
        if time_value is None:
            time_value = as_float(row.get("solve_index"))
        if time_value is None:
            time_value = float(index)
        xs.append(time_value)
        ys.append(clearance)
    if not ys:
        return {
            "status": "missing_data",
            "path": relpath(out_path),
            "reason": "mpc rollout has no clearance columns",
        }

    threshold = as_float(diagnostics.get("minimum_clearance"))
    if threshold is None:
        threshold = as_float(diagnostics.get("topology_clearance_target"))
    if threshold is None:
        threshold = as_float(diagnostics.get("planning_clearance"))

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.plot(xs, ys, color="#0072B2", linewidth=1.8, label="minimum clearance")
    if threshold is not None:
        ax.axhline(threshold, color="#D55E00", linestyle="--",
                   linewidth=1.6, label="safety threshold")
    ax.set_title("{} MPC clearance over time".format(robot))
    ax.set_xlabel("time")
    ax.set_ylabel("minimum clearance")
    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return {"status": "generated", "path": relpath(out_path), "reason": ""}


def plot_mpc_cost(robot, base_dir, figures_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = os.path.join(figures_dir, "{}_mpc_cost.png".format(robot))
    rows = load_csv_rows(os.path.join(base_dir, "mpc_cost_breakdown.csv"))
    if not rows:
        return {
            "status": "missing_data",
            "path": relpath(out_path),
            "reason": "missing mpc_cost_breakdown.csv",
        }
    fields = (
        "tracking_cost",
        "risk_cost",
        "control_cost",
        "smoothness_cost",
        "topology_cost",
        "corridor_cost",
        "manifold_cost",
    )
    series = {}
    xs = []
    for index, row in enumerate(rows):
        x = as_float(row.get("global_step"))
        if x is None:
            x = as_float(row.get("solve_index"))
        if x is None:
            x = float(index)
        xs.append(x)
    for field in fields:
        values = []
        for row in rows:
            value = as_float(row.get(field))
            values.append(value)
        if any(value is not None for value in values):
            series[field] = [0.0 if value is None else value for value in values]
    if not series:
        return {
            "status": "missing_data",
            "path": relpath(out_path),
            "reason": "mpc_cost_breakdown.csv has no requested cost fields",
        }

    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    for field, values in series.items():
        ax.plot(xs, values, linewidth=1.6, label=field)
    ax.set_title("{} MPC cost breakdown".format(robot))
    ax.set_xlabel("time")
    ax.set_ylabel("cost")
    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return {"status": "generated", "path": relpath(out_path), "reason": ""}


def canonical_method(row):
    mode = str(row.get("mode", row.get("variant", ""))).strip().lower()
    variant = str(row.get("variant", "")).strip().lower()
    adp_used = str(row.get("adp_used", row.get("adp_enabled", ""))).strip().lower()
    mpc_used = str(row.get("mpc_used", "")).strip().lower()
    if mode == "baseline" or variant == "baseline":
        return "baseline"
    if adp_used in ("1", "1.0", "true", "yes"):
        return "STSM-MPC-ADP"
    if mpc_used in ("1", "1.0", "true", "yes"):
        return "STSM-MPC"
    if mode == "stsm" or variant == "stsm":
        return "STSM"
    return mode or variant or "unknown"


def first_number(row, keys):
    for key in keys:
        value = as_float(row.get(key))
        if value is not None:
            return value
    return None


def moving_average(values, window=7):
    import numpy as np
    arr = np.asarray(values, float)
    if arr.size < int(window):
        return arr
    kernel = np.ones(int(window), float) / float(window)
    return np.convolve(arr, kernel, mode="same")


def _trajectory_series(rows, value_fn):
    times = []
    values = []
    for idx, row in enumerate(rows or []):
        t = first_number(row, ("t", "time", "stamp"))
        if t is None:
            t = float(idx)
        value = value_fn(row)
        if value is None:
            continue
        times.append(float(t))
        values.append(float(value))
    if not times:
        return [], []
    t0 = float(times[0])
    return [float(t - t0) for t in times], values


def plot_wheelchair_risk_time_series(run_dir, figures_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = os.path.join(figures_dir, "wheelchair_risk_time_series.png")
    baseline_rows = load_csv_rows(os.path.join(
        run_dir, "wheelchair", "baseline", "traj.csv"))
    stsm_rows = load_csv_rows(os.path.join(
        stsm_dir(run_dir, "wheelchair"), "traj.csv"))
    if not baseline_rows or not stsm_rows:
        return {
            "status": "missing_data",
            "path": relpath(out_path),
            "reason": "missing wheelchair baseline/stsm trajectory csv",
        }

    cjk_font = configure_plot_fonts()
    plt.rcParams.update({
        "font.size": 9.0,
    })

    def total_risk(row):
        return first_number(row, (
            "phi_total", "footprint_gate_risk", "risk", "risk_value"))

    def body_risk(row):
        return first_number(row, (
            "phi_max_point", "footprint_gate_risk",
            "phi_mean_point", "phi_center"))

    def social_env_risk(row):
        center = first_number(row, ("phi_center",))
        env = first_number(row, ("phi_env",))
        if center is None and env is None:
            return first_number(row, ("phi_total", "risk", "risk_value"))
        return float(center or 0.0) + float(env or 0.0)

    panels = [
        ("总社会风险", total_risk),
        ("身体部位风险", body_risk),
        ("社交/环境风险", social_env_risk),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.1), sharex=False)
    colors = {"Baseline": "#0072B2", "STSM": "#D55E00"}
    for ax, (title, value_fn) in zip(axes, panels):
        bx, by = _trajectory_series(baseline_rows, value_fn)
        sx, sy = _trajectory_series(stsm_rows, value_fn)
        if len(by) >= 2:
            ax.plot(bx, moving_average(by), color=colors["Baseline"],
                    linewidth=1.8, label="Baseline")
        if len(sy) >= 2:
            ax.plot(sx, moving_average(sy), color=colors["STSM"],
                    linewidth=1.8, label="STSM")
        ax.set_title(title, fontproperties=cjk_font)
        ax.set_xlabel("时间 (s)", fontproperties=cjk_font)
        ax.set_ylabel("风险", fontproperties=cjk_font)
        ax.grid(True, color="#e3e7eb", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#b9c2cf")
        ax.spines["bottom"].set_color("#b9c2cf")
        ax.tick_params(colors="#6b778c")
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            if cjk_font:
                tick.set_fontproperties(cjk_font)
    axes[-1].legend(loc="upper right", frameon=True)
    if cjk_font:
        for text in axes[-1].get_legend().get_texts():
            text.set_fontproperties(cjk_font)
    fig.tight_layout(w_pad=1.8)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {"status": "generated", "path": relpath(out_path), "reason": ""}


def read_metrics_pair(path):
    rows = load_csv_rows(path)
    by_mode = {}
    for row in rows:
        key = str(row.get("variant") or row.get("mode") or "").strip().lower()
        if key.startswith("baseline"):
            by_mode["baseline"] = row
        elif key == "stsm":
            by_mode["stsm"] = row
    return by_mode.get("baseline"), by_mode.get("stsm")


def value_text(value):
    value = float(value)
    if abs(value) >= 100:
        return "{:.1f}".format(value)
    if abs(value) >= 10:
        return "{:.2f}".format(value)
    if abs(value) >= 1:
        return "{:.3f}".format(value)
    return "{:.4f}".format(value)


def metric_value_text(value, unit):
    text = value_text(value)
    if unit == "%":
        return "{}%".format(text)
    return text


def plot_robot_metrics_compare(robot, run_dir, figures_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    compare_path = os.path.join(run_dir, "compare", "{}_compare_metrics.csv".format(robot))
    out_path = os.path.join(figures_dir, "{}_metrics_compare.png".format(robot))
    baseline, stsm = read_metrics_pair(compare_path)
    if not baseline or not stsm:
        return {
            "status": "missing_data",
            "path": relpath(out_path),
            "reason": "missing baseline/stsm rows in {}".format(relpath(compare_path)),
        }
    if robot == "arm":
        metrics = [
            ("mean_phi_s", "平均社会风险", ""),
            ("risk_exceed_pct", "风险超阈占比", "%"),
            ("duration_s", "任务时长", "s"),
            ("min_head_dist", "末端-头部最小距离", "m"),
            ("min_chest_dist", "末端-胸部最小距离", "m"),
            ("mean_speed_near_hand", "近手平均速度", "m/s"),
        ]
        title = "Baseline 与 STSM 方法指标对比 - 机械臂"
    else:
        metrics = [
            ("mean_phi_s", "平均社会风险", ""),
            ("max_phi_s", "最大社会风险", ""),
            ("risk_exceed_pct", "风险超阈占比", "%"),
            ("duration_s", "任务时长", "s"),
            ("min_person_dist", "轮椅中心-人体最小距离", "m"),
            ("max_speed", "轮椅最大速度", "m/s"),
        ]
        title = "Baseline 与 STSM 方法指标对比 - 轮椅"

    text = "#1f2933"
    muted = "#5b6770"
    axis = "#9aa4ad"
    grid = "#e8edf1"
    blue = "#0072B2"
    orange = "#D55E00"
    cjk_font = configure_plot_fonts()
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.0,
        "axes.linewidth": 0.8,
        "savefig.dpi": 300,
    })
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 5.5))
    for ax, (key, label, unit) in zip(axes.ravel(), metrics):
        b = first_number(baseline, (key,))
        s = first_number(stsm, (key,))
        if b is None or s is None:
            ax.axis("off")
            continue
        values = [b, s]
        bars = ax.bar([0, 1], values, width=0.58, color=[blue, orange],
                      edgecolor="none")
        ymax = max(values) * 1.42 if max(values) > 0 else 1.0
        ax.set_ylim(0.0, ymax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Baseline", "STSM"], fontproperties=cjk_font,
                           fontsize=8.8)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_title(label, loc="center", color=text, fontweight="bold",
                     fontproperties=cjk_font)
        if unit:
            ax.set_ylabel(unit, color=muted, fontproperties=cjk_font)
        ax.grid(axis="y", color=grid, linewidth=0.7, alpha=0.85)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(axis)
        ax.tick_params(axis="x", length=0, colors=text)
        ax.tick_params(axis="y", length=0, colors=muted)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontproperties(cjk_font)
            tick.set_fontsize(8.3)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + ymax * 0.035,
                metric_value_text(value, unit),
                ha="center",
                va="bottom",
                fontsize=8.4,
                color=text,
                fontproperties=cjk_font,
                zorder=5,
            )
    fig.text(0.5, 0.965, title, ha="center", va="top",
             fontsize=14.0, color=text, fontweight="bold",
             fontproperties=cjk_font)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.84, bottom=0.09,
                        wspace=0.34, hspace=0.52)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return {"status": "generated", "path": relpath(out_path), "reason": ""}


def plot_adp_value(robot, base_dir, figures_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = os.path.join(figures_dir, "{}_adp_value.png".format(robot))
    source_path = ""
    rows = []
    for filename in ("trajectory.csv", "traj.csv"):
        source_path = os.path.join(base_dir, filename)
        rows = load_csv_rows(source_path)
        if rows:
            break
    if not rows:
        return {
            "status": "missing_data",
            "path": relpath(out_path),
            "reason": "missing trajectory csv for ADP value history",
        }

    xs = []
    values = []
    td_errors = []
    critic_losses = []
    for index, row in enumerate(rows):
        value = first_number(row, ("adp_value", "value_history", "value"))
        td = first_number(row, ("td_error", "TD_error"))
        loss = first_number(row, ("critic_loss", "loss"))
        x = first_number(row, ("t", "global_step", "step"))
        if x is None:
            x = float(index)
        if value is not None:
            xs.append(x)
            values.append(value)
        if td is not None:
            td_errors.append((x, td))
        if loss is not None:
            critic_losses.append((x, loss))
    if len(values) < 2 and not td_errors and not critic_losses:
        return {
            "status": "missing_data",
            "path": relpath(out_path),
            "reason": "missing value history, TD error, and critic loss",
        }

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    if values:
        ax.plot(xs, values, color="#0072B2", linewidth=1.8, label="value history")
    if td_errors:
        ax.plot([x for x, _v in td_errors], [v for _x, v in td_errors],
                color="#D55E00", linewidth=1.4, label="TD error")
    if critic_losses:
        ax.plot([x for x, _v in critic_losses], [v for _x, v in critic_losses],
                color="#009E73", linewidth=1.4, label="critic loss")
    ax.set_title("{} ADP value".format(robot))
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.legend(loc="best", fontsize=8)
    ax.text(0.01, 0.01, "source: {}".format(relpath(source_path)),
            transform=ax.transAxes, fontsize=8, color="#555555", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return {"status": "generated", "path": relpath(out_path), "reason": ""}


def write_audit(figures_dir, audit):
    if not os.path.isdir(figures_dir):
        os.makedirs(figures_dir)
    path = os.path.join(figures_dir, AUDIT_NAME)
    with open(path, "w") as f:
        json.dump(audit, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def move_if_exists(src, dst):
    if not os.path.isfile(src):
        return False
    directory = os.path.dirname(dst)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    if os.path.abspath(src) != os.path.abspath(dst):
        if os.path.exists(dst):
            os.remove(dst)
        os.rename(src, dst)
    return True


def finalize_figure_layout(figures_dir, audit=None):
    audit = audit or {}
    figures = audit.get("figures", {})
    supplement_dir = os.path.join(figures_dir, "supplement")
    if not os.path.isdir(supplement_dir):
        os.makedirs(supplement_dir)
    for name in (
            "arm_morse_points_diagnostics.png",
            "wheelchair_morse_points_diagnostics.png",
            "arm_execution.png",
            "wheelchair_execution.png",
            "arm_adp_value.png",
            "wheelchair_adp_value.png"):
        if move_if_exists(os.path.join(figures_dir, name),
                          os.path.join(supplement_dir, name)):
            figures.setdefault(name, {})
            figures[name]["path"] = relpath(os.path.join(supplement_dir, name))
    aliases = {
        "arm_body_risk_time.png": "arm_risk_time_series.png",
        "wheelchair_body_risk_time.png": "wheelchair_risk_time_series.png",
    }
    for old_name, new_name in aliases.items():
        old_path = os.path.join(figures_dir, old_name)
        new_path = os.path.join(figures_dir, new_name)
        if os.path.isfile(old_path) and not os.path.isfile(new_path):
            os.rename(old_path, new_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=RESULTS_DIR + "/run")
    parser.add_argument("--figures-dir", default=DEFAULT_FIGURES_DIR)
    args = parser.parse_args()

    if not os.path.isdir(args.figures_dir):
        os.makedirs(args.figures_dir)
    audit = {"figures": {}, "missing_data": []}
    risk_result = plot_arm_trajectory_on_risk_field(
        args.run_dir, args.figures_dir)
    audit["figures"]["arm_trajectory_on_risk_field.png"] = risk_result
    if risk_result.get("status") not in ("generated", "experiment_failed"):
        audit["missing_data"].append({
            "figure": "arm_trajectory_on_risk_field.png",
            "reason": risk_result.get("reason", ""),
        })
    wheelchair_field_result = plot_wheelchair_trajectory_on_risk_field(
        args.run_dir, args.figures_dir)
    audit["figures"]["wheelchair_trajectory_on_risk_field.png"] = (
        wheelchair_field_result)
    if wheelchair_field_result.get("status") not in (
            "generated", "experiment_failed"):
        audit["missing_data"].append({
            "figure": "wheelchair_trajectory_on_risk_field.png",
            "reason": wheelchair_field_result.get("reason", ""),
        })
    for robot in ROBOTS:
        base_dir = stsm_dir(args.run_dir, robot)
        outputs = [
            ("{}_morse_topology.png".format(robot),
             plot_morse_topology(robot, base_dir, args.figures_dir)),
            ("{}_candidate_filter.png".format(robot),
             plot_candidate_filter(robot, base_dir, args.figures_dir)),
            ("{}_execution.png".format(robot),
             plot_execution(robot, base_dir, args.figures_dir)),
            ("{}_mpc_clearance_time.png".format(robot),
             plot_mpc_clearance(robot, base_dir, args.figures_dir)),
            ("{}_mpc_cost.png".format(robot),
             plot_mpc_cost(robot, base_dir, args.figures_dir)),
            ("{}_adp_value.png".format(robot),
             plot_adp_value(robot, base_dir, args.figures_dir)),
            ("{}_metrics_compare.png".format(robot),
             plot_robot_metrics_compare(robot, args.run_dir, args.figures_dir)),
        ]
        if robot == "wheelchair":
            outputs.append((
                "wheelchair_risk_time_series.png",
                plot_wheelchair_risk_time_series(
                    args.run_dir, args.figures_dir)))
        for name, result in outputs:
            audit["figures"][name] = result
            if result.get("status") not in ("generated", "experiment_failed"):
                audit["missing_data"].append({
                    "figure": name,
                    "reason": result.get("reason", ""),
                })
    finalize_figure_layout(args.figures_dir, audit)
    write_audit(args.figures_dir, audit)
    print("execution trajectory source:")
    for robot in ROBOTS:
        result = audit["figures"].get("{}_execution.png".format(robot), {})
        print("{}: {}".format(robot, result.get("source", "")))
    for name, result in sorted(audit["figures"].items()):
        if result.get("status") == "generated":
            print(os.path.join(args.figures_dir, name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
