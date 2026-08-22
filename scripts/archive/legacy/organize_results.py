#!/usr/bin/env python3
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from datetime import date
sys.dont_write_bytecode = True


ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
RESULTS = os.path.join(ROOT, "results")
ARCHIVE_ROOT = os.path.expanduser("~/stsm_madp_results_archive")
VARIANTS = ("baseline", "stsm")
ROBOTS = ("wheelchair", "arm")
RUN_ID_RE = re.compile(r"^\d{8}_R\d{3}$")
SUMMARY_PLACEHOLDERS = {
    "ablation_table.csv": [
        "run_id", "robot", "variant", "success_goal", "success_safe",
        "duration_s", "J_social", "risk_exceed_pct", "mean_adp_value",
        "max_adp_value",
    ],
    "best_runs.csv": [
        "robot", "variant", "run_id", "success_goal", "success_safe",
        "duration_s", "J_social", "risk_exceed_pct",
    ],
}


def ensure_dirs(run_id):
    for path in [
        RESULTS,
        os.path.join(RESULTS, "runs"),
        os.path.join(RESULTS, "summary"),
        os.path.join(RESULTS, "figures"),
    ]:
        os.makedirs(path, exist_ok=True)
    run_dir = os.path.join(RESULTS, "runs", run_id)
    os.makedirs(os.path.join(run_dir, "config"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "compare"), exist_ok=True)
    for robot in ROBOTS:
        for variant in VARIANTS:
            os.makedirs(os.path.join(run_dir, robot, variant), exist_ok=True)
    return run_dir


def copy_if_exists(src, dst):
    if os.path.exists(src):
        shutil.copy2(src, dst)


def command_output(cmd):
    try:
        safe_cmd = list(cmd)
        if safe_cmd and safe_cmd[0] == "git":
            safe_cmd[1:1] = ["-c", "safe.directory={}".format(ROOT)]
        return subprocess.check_output(
            safe_cmd, cwd=ROOT, stderr=subprocess.STDOUT).decode()
    except Exception as exc:
        return "%s failed: %s\n" % (" ".join(cmd), exc)


def write_config(run_dir):
    cfg_dir = os.path.join(run_dir, "config")
    interesting = [
        "RUN_ID", "TARGET", "GUI", "RVIZ", "PLOT", "CLEAN_ENV",
        "ADP_ENABLED", "ADP_SOLVER_MODE", "USE_CVXPY", "ADP_BLEND_ALPHA",
        "ADP_DESCENT_GAIN",
        "LAMBDA_ADP", "LAMBDA_ADP_CORRIDOR", "LAMBDA_ADP_TERMINAL",
        "LAMBDA_ADP_PATH", "LAMBDA_ADP_ARM", "WC_COMPLETION_TOLERANCE",
        "WC_REPLAN_PERIOD", "WC_NEAR_GOAL_RADIUS",
        "WC_NEAR_GOAL_ADP_SCALE", "WC_NEAR_GOAL_GOAL_WEIGHT",
        "WC_NO_PROGRESS_REPLAN_TIME", "WC_PROGRESS_REWARD_WEIGHT",
        "WC_FINAL_APPROACH_RADIUS", "WC_FINAL_HEADING_THRESHOLD",
        "WC_FINAL_HEADING_GAIN", "WC_FINAL_CREEP_V", "WC_FINAL_MIN_V",
        "WC_FINAL_MAX_V", "WC_FINAL_FORWARD_GAIN", "WC_LAM_HEADING",
        "WC_FINAL_DIRECT_OVERRIDE_ENABLED", "WC_FINAL_DIRECT_OVERRIDE_RADIUS",
        "WC_MPC_HORIZON", "WC_MPC_DT", "WC_MPC_A_MAX",
        "WC_MPC_ALPHA_MAX", "WC_MPC_BEAM_WIDTH",
    ]
    with open(os.path.join(cfg_dir, "launch_args.txt"), "w") as f:
        for key in interesting:
            f.write("%s=%s\n" % (key, os.environ.get(key, "")))
    copy_if_exists(os.path.join(ROOT, "config", "adp_critic.yaml"),
                   os.path.join(cfg_dir, "adp_critic.yaml"))
    copy_if_exists(os.path.join(ROOT, "config", "handover_scene.yaml"),
                   os.path.join(cfg_dir, "arm_params.yaml"))
    copy_if_exists(os.path.join(ROOT, "config", "wheelchair_control.yaml"),
                   os.path.join(cfg_dir, "wheelchair_params.yaml"))
    with open(os.path.join(cfg_dir, "git_info.txt"), "w") as f:
        f.write(command_output(["git", "rev-parse", "--show-toplevel"]))
        f.write(command_output(["git", "rev-parse", "HEAD"]))
        f.write(command_output(["git", "status", "--short"]))


def write_results_readme():
    path = os.path.join(RESULTS, "README.md")
    if os.path.exists(path):
        return
    with open(path, "w") as f:
        f.write("# STSM-MADP Results\n\n")
        f.write("Use `summary/all_metrics.csv` for aggregate analysis. ")
        f.write("Current compact run data lives under `run/`; aggregate data lives under `summary/`.\n")


def ensure_summary_files():
    summary_dir = os.path.join(RESULTS, "summary")
    for name, fields in SUMMARY_PLACEHOLDERS.items():
        path = os.path.join(summary_dir, name)
        if os.path.exists(path):
            continue
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()


def ensure_paper_figure_source(run_id):
    path = os.path.join(RESULTS, "figures", "figure_source.md")
    if os.path.exists(path):
        return
    with open(path, "w") as f:
        f.write("# Figure Sources\n\n")
        f.write("Figures in this directory are generated from standardized run data.\n\n")
        f.write("- Compact run: `results/run`\n")
        f.write("- Current organized run: `results/runs/%s`\n" % run_id)
        f.write("- Aggregate metrics: `results/summary/all_metrics.csv`\n")


def write_run_readme(run_dir, run_id, purpose, stage, status):
    path = os.path.join(run_dir, "README.md")
    with open(path, "w") as f:
        f.write("# %s\n\n" % run_id)
        f.write("## Purpose\n%s\n\n" % (purpose or ""))
        f.write("## Stage\n%s\n\n" % (stage or ""))
        f.write("## Status\n%s\n\n" % (status or ""))
        f.write("## Variants\n")
        for variant in VARIANTS:
            f.write("- %s\n" % variant)
        f.write("\n## Key settings\n")
        for key in ("USE_CVXPY", "ADP_SOLVER_MODE", "ADP_ENABLED",
                    "LAMBDA_ADP_CORRIDOR", "LAMBDA_ADP_ARM",
                    "ADP_BLEND_ALPHA"):
            f.write("- %s = %s\n" % (key, os.environ.get(key, "")))


def update_index(run_id, purpose, stage, status, notes):
    path = os.path.join(RESULTS, "summary", "experiment_index.csv")
    fields = ["run_id", "date", "stage", "purpose", "status", "main_result", "notes"]
    rows = []
    if os.path.exists(path):
        with open(path, "r") as f:
            rows = [row for row in csv.DictReader(f) if row.get("run_id") != run_id]
    rows.append({
        "run_id": run_id,
        "date": str(date.today()),
        "stage": stage or "",
        "purpose": purpose or "",
        "status": status or "",
        "main_result": "",
        "notes": notes or "",
    })
    rows.sort(key=lambda row: row.get("run_id", ""))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def update_latest(run_dir):
    latest = os.path.join(RESULTS, "latest")
    rel = os.path.relpath(run_dir, RESULTS)
    try:
        if os.path.lexists(latest):
            os.unlink(latest)
        os.symlink(rel, latest)
    except OSError:
        with open(latest + ".txt", "w") as f:
            f.write(rel + "\n")


def clean_flat():
    names = []
    for name in os.listdir(RESULTS):
        path = os.path.join(RESULTS, name)
        if os.path.isfile(path) and (
                name.endswith(".csv") or name.endswith(".png") or
                name.endswith(".gif") or name.endswith(".log")):
            names.append(name)
    if not names:
        return ""
    archive = os.path.join(
        ARCHIVE_ROOT, "old_flat_results_%s" % date.today().strftime("%Y%m%d"))
    os.makedirs(archive, exist_ok=True)
    for name in names:
        shutil.move(os.path.join(RESULTS, name), os.path.join(archive, name))
    return archive


def archive_legacy_runs():
    runs_dir = os.path.join(RESULTS, "runs")
    if not os.path.isdir(runs_dir):
        return ""
    names = [
        name for name in os.listdir(runs_dir)
        if os.path.isdir(os.path.join(runs_dir, name)) and not RUN_ID_RE.match(name)
    ]
    if not names:
        return ""
    archive = os.path.join(
        ARCHIVE_ROOT, "old_runs_%s" % date.today().strftime("%Y%m%d"))
    os.makedirs(archive, exist_ok=True)
    for name in sorted(names):
        src = os.path.join(runs_dir, name)
        dst = os.path.join(archive, name)
        if os.path.exists(dst):
            base = dst
            index = 1
            while os.path.exists(dst):
                dst = "%s_%02d" % (base, index)
                index += 1
        shutil.move(src, dst)
    return archive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--purpose", default="")
    parser.add_argument("--stage", default="")
    parser.add_argument("--status", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--clean-flat", action="store_true")
    parser.add_argument("--archive-legacy-runs", action="store_true")
    parser.add_argument("--update-latest", action="store_true")
    args = parser.parse_args()
    write_results_readme()
    run_dir = ensure_dirs(args.run_id)
    ensure_summary_files()
    ensure_paper_figure_source(args.run_id)
    write_config(run_dir)
    write_run_readme(run_dir, args.run_id, args.purpose, args.stage, args.status)
    update_index(args.run_id, args.purpose, args.stage, args.status, args.notes)
    if args.clean_flat:
        archive = clean_flat()
        if archive:
            print("archived flat result files to %s" % archive)
    if args.archive_legacy_runs:
        archive = archive_legacy_runs()
        if archive:
            print("archived legacy run directories to %s" % archive)
    if args.update_latest:
        update_latest(run_dir)
    print("organized %s" % run_dir)


if __name__ == "__main__":
    main()
