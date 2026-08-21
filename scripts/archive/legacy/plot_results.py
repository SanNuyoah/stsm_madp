#!/usr/bin/env python3
from __future__ import print_function

import argparse
import csv
import os
import sys
sys.dont_write_bytecode = True

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
SCRIPTS = os.path.join(ROOT, "scripts")
if os.path.isdir(os.path.join(SRC, "stsm_madp")) and SRC not in sys.path:
    sys.path.insert(0, SRC)
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
from plot_extended_results import build_plots as build_extended_plots


BLUE = "#0072B2"
ORANGE = "#D55E00"

TEXT = "#1f2933"
MUTED = "#5b6770"
AXIS = "#9aa4ad"
GRID = "#e8edf1"

FONT_REGULAR = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")


def read_one(path):
    with open(path, "r") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("empty csv: {0}".format(path))

    data = {}
    for key, value in rows[0].items():
        try:
            data[key] = float(value)
        except (TypeError, ValueError):
            data[key] = value
    return data


def read_metrics_table(path):
    with open(path, "r") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("empty csv: {0}".format(path))

    by_mode = {}
    for row in rows:
        mode = row.get("mode", "").strip().lower()
        variant = row.get("variant", "").strip().lower()
        key = variant or mode
        if key in ("baseline", "baseline0", "baseline1", "stsm"):
            data = {}
            for field, value in row.items():
                try:
                    data[field] = float(value)
                except (TypeError, ValueError):
                    data[field] = value
            by_mode[key] = data
            if key.startswith("baseline"):
                by_mode["baseline"] = data
    stsm_key = "stsm"
    if "baseline" not in by_mode or stsm_key not in by_mode:
        raise RuntimeError("metrics table must contain baseline and stsm rows: {0}".format(path))
    return by_mode["baseline"], by_mode[stsm_key]


def read_pair(results_dir, combined_name):
    combined_path = os.path.join(results_dir, combined_name)
    if not os.path.exists(combined_path):
        combined_path = os.path.join(results_dir, "compare", combined_name)
    return read_metrics_table(combined_path)


def existing_metrics_path(results_dir, names):
    for name in names:
        path = os.path.join(results_dir, name)
        if os.path.exists(path):
            return path, name
    return os.path.join(results_dir, names[0]), names[0]


def existing_result_file(results_dir, names):
    for name in names:
        path = os.path.join(results_dir, name)
        if os.path.exists(path):
            return path
    return os.path.join(results_dir, names[0])


def read_traj(path):
    with open(path, "r") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("empty trajectory csv: {0}".format(path))
    out = {}
    for key in rows[0].keys():
        values = []
        numeric = True
        for row in rows:
            value = row.get(key, "")
            if value == "":
                values.append(float("nan"))
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                numeric = False
                break
        if numeric:
            out[key] = values
    if "t" in out and out["t"]:
        t0 = out["t"][0]
        out["t"] = [t - t0 for t in out["t"]]
    return out


def value_text(value):
    value = float(value)
    if abs(value) >= 100:
        return "{0:.1f}".format(value)
    if abs(value) >= 10:
        return "{0:.2f}".format(value)
    if abs(value) >= 1:
        return "{0:.3f}".format(value)
    return "{0:.4f}".format(value)


def setup_rc():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 8.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.0,
        "axes.linewidth": 0.8,
        "savefig.dpi": 300,
    })


def style_axis(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="x", length=0, colors=TEXT)
    ax.tick_params(axis="y", length=0, colors=MUTED)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(FONT_REGULAR)


def draw_metric(ax, metric, baseline, stsm):
    key = metric["key"]
    values = [float(baseline[key]), float(stsm[key])]
    colors = [BLUE, ORANGE]
    bars = ax.bar([0, 1], values, width=0.58, color=colors, edgecolor="none")

    ymax = max(values) * 1.28 if max(values) > 0 else 1.0
    ax.set_ylim(0.0, ymax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline", "STSM"], fontproperties=FONT_REGULAR)
    ax.set_title(metric["label"], loc="center", color=TEXT, fontproperties=FONT_BOLD)
    if metric.get("unit"):
        ax.set_ylabel(metric["unit"], color=MUTED, fontproperties=FONT_REGULAR)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + ymax * 0.035,
            value_text(value),
            ha="center",
            va="bottom",
            fontsize=7.8,
            color=TEXT,
            fontproperties=FONT_REGULAR,
        )

    style_axis(ax)


def draw_dashboard(title, subtitle, metrics, baseline, stsm, out_path, cols=3):
    setup_rc()

    rows = int((len(metrics) + cols - 1) / cols)
    fig_w = 8.6 if cols == 3 else 7.0
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, 2.75 * rows))
    if rows == 1:
        axes = [axes]

    flat_axes = []
    for row in axes:
        try:
            flat_axes.extend(list(row))
        except TypeError:
            flat_axes.append(row)

    for ax, metric in zip(flat_axes, metrics):
        draw_metric(ax, metric, baseline, stsm)

    for ax in flat_axes[len(metrics):]:
        ax.axis("off")

    fig.text(0.5, 0.965, title, ha="center", va="top",
             fontsize=14.0, color=TEXT, fontproperties=FONT_BOLD)
    legend_y = 0.045
    fig.patches.append(plt.Rectangle((0.405, legend_y - 0.008), 0.018, 0.016,
                                     transform=fig.transFigure, color=BLUE, clip_on=False))
    fig.text(0.428, legend_y, "Baseline", ha="left", va="center",
             fontsize=8.2, color=TEXT, fontproperties=FONT_REGULAR)
    fig.patches.append(plt.Rectangle((0.515, legend_y - 0.008), 0.018, 0.016,
                                     transform=fig.transFigure, color=ORANGE, clip_on=False))
    fig.text(0.538, legend_y, "STSM", ha="left", va="center",
             fontsize=8.2, color=TEXT, fontproperties=FONT_REGULAR)

    fig.subplots_adjust(left=0.08, right=0.985, top=0.84, bottom=0.14,
                        wspace=0.34, hspace=0.52)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def draw_close_monitor(out_path, title, baseline, stsm, show_phase=False):
    setup_rc()
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    for data, color, label in ((baseline, BLUE, "Baseline"),
                               (stsm, ORANGE, "STSM")):
        t = data.get("t", [])
        axes[0].plot(t, data.get("speed_filtered", data.get("speed_raw", [])),
                     color=color, lw=1.8, label=label)
        axes[1].plot(t, data.get("phi_close_monitor", []),
                     color=color, lw=1.8, label=label)
        if show_phase and "phase" in data:
            phase = data["phase"]
            last = None
            for ti, ph in zip(t, phase):
                if ph != ph:
                    continue
                if last is None:
                    last = ph
                elif ph != last:
                    axes[1].axvline(ti, color=AXIS, lw=0.8, ls=":", alpha=0.8)
                    last = ph
    axes[0].set_title(title, color=TEXT, fontproperties=FONT_BOLD)
    axes[0].set_ylabel("速度 (m/s)", color=MUTED, fontproperties=FONT_REGULAR)
    axes[1].set_ylabel("phi_close_monitor", color=MUTED, fontproperties=FONT_REGULAR)
    axes[1].set_xlabel("时间 (s)", color=MUTED, fontproperties=FONT_REGULAR)
    axes[0].legend(loc="upper right", frameon=True)
    for ax in axes:
        style_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def draw_monitor_plots(results_dir, pictures_dir):
    arm_base_path = existing_result_file(
        results_dir, ["arm/baseline/traj.csv", "arm_baseline_traj.csv"])
    arm_stsm_path = existing_result_file(
        results_dir, ["arm/stsm/traj.csv", "arm_stsm_traj.csv"])
    wc_base_path = existing_result_file(
        results_dir, ["wheelchair/baseline/traj.csv", "wc_baseline_traj.csv"])
    wc_stsm_path = existing_result_file(
        results_dir, ["wheelchair/stsm/traj.csv", "wc_stsm_traj.csv"])
    required = [arm_base_path, arm_stsm_path, wc_base_path, wc_stsm_path]
    if not all(os.path.exists(path) for path in required):
        return
    arm_baseline = read_traj(arm_base_path)
    arm_stsm = read_traj(arm_stsm_path)
    wc_baseline = read_traj(wc_base_path)
    wc_stsm = read_traj(wc_stsm_path)
    if "phi_close_monitor" in arm_baseline and "phi_close_monitor" in arm_stsm:
        draw_close_monitor(
            os.path.join(pictures_dir, "arm_close_risk_monitor.png"),
            "机械臂末端速度与闭合风险影子监测",
            arm_baseline,
            arm_stsm,
            show_phase=True,
        )
    if "phi_close_monitor" in wc_baseline and "phi_close_monitor" in wc_stsm:
        draw_close_monitor(
            os.path.join(pictures_dir, "wheelchair_close_risk_monitor.png"),
            "轮椅中心速度与闭合风险影子监测",
            wc_baseline,
            wc_stsm,
            show_phase=False,
        )


def build_plots(results_dir, pictures_dir):
    if not os.path.isdir(pictures_dir):
        os.makedirs(pictures_dir)

    arm_metrics = [
        {"key": "mean_phi_s", "label": "平均社会风险"},
        {"key": "risk_exceed_pct", "label": "风险超阈占比", "unit": "%"},
        {"key": "duration_s", "label": "任务时长", "unit": "s"},
        {"key": "min_head_dist", "label": "末端-头部最小距离", "unit": "m"},
        {"key": "min_chest_dist", "label": "末端-胸部最小距离", "unit": "m"},
        {"key": "mean_speed_near_hand", "label": "近手平均速度", "unit": "m/s"},
    ]
    wheelchair_metrics = [
        {"key": "mean_phi_s", "label": "平均社会风险"},
        {"key": "max_phi_s", "label": "最大社会风险"},
        {"key": "risk_exceed_pct", "label": "风险超阈占比", "unit": "%"},
        {"key": "duration_s", "label": "任务时长", "unit": "s"},
        {"key": "min_person_dist", "label": "轮椅中心-人体最小距离", "unit": "m"},
        {"key": "max_speed", "label": "轮椅最大速度", "unit": "m/s"},
    ]

    arm_metrics_path = existing_result_file(
        results_dir, ["compare/arm_compare_metrics.csv", "arm_metrics.csv"])
    wc_metrics_path, wc_metrics_name = existing_metrics_path(
        results_dir, ["compare/wheelchair_compare_metrics.csv", "wc_metrics.csv", "wheelchair_metrics.csv"])
    if os.path.exists(arm_metrics_path):
        arm_baseline, arm_stsm = read_metrics_table(arm_metrics_path)
        draw_dashboard(
            "Baseline vs STSM Method - Representative Single Trial (Arm)",
            "",
            arm_metrics,
            arm_baseline,
            arm_stsm,
            os.path.join(pictures_dir, "arm_compare.png"),
            cols=3,
        )
    else:
        print("skip arm dashboard: missing {0}".format(arm_metrics_path))

    if os.path.exists(wc_metrics_path):
        wc_baseline, wc_stsm = read_metrics_table(wc_metrics_path)
        draw_dashboard(
            "Baseline vs STSM Method - Representative Single Trial (Wheelchair)",
            "",
            wheelchair_metrics,
            wc_baseline,
            wc_stsm,
            os.path.join(pictures_dir, "wheelchair_compare.png"),
            cols=3,
        )
    else:
        print("skip wheelchair dashboard: missing {0}".format(wc_metrics_path))

    extended_required = [
        existing_result_file(results_dir, ["arm/baseline/traj.csv", "arm_baseline_traj.csv"]),
        existing_result_file(results_dir, ["arm/stsm/traj.csv", "arm_stsm_traj.csv"]),
        existing_result_file(results_dir, ["wheelchair/baseline/traj.csv", "wc_baseline_traj.csv"]),
        existing_result_file(results_dir, ["wheelchair/stsm/traj.csv", "wc_stsm_traj.csv"]),
    ]
    if all(os.path.exists(path) for path in extended_required):
        build_extended_plots(results_dir, pictures_dir)
    else:
        print("skip extended plots: missing one or more trajectory inputs")
    draw_monitor_plots(results_dir, pictures_dir)


def main():
    parser = argparse.ArgumentParser(description="Create baseline-vs-STSM representative single-trial bar charts.")
    parser.add_argument("--results", default=os.path.join(ROOT, "results"))
    parser.add_argument("--out", default=os.path.join(ROOT, "pictures"))
    args = parser.parse_args()
    build_plots(args.results, args.out)
    print("wrote {0}".format(args.out))


if __name__ == "__main__":
    main()
