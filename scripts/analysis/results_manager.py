#!/usr/bin/env python3
from __future__ import print_function

import argparse
import csv
import os
import re
import shutil
import sys
from datetime import date

sys.dont_write_bytecode = True

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEGACY = os.path.join(ROOT, "scripts", "archive", "legacy")
RESULTS = os.path.join(ROOT, "results")
RUN_RE = re.compile(r"^\d{8}_R\d{3}$")
ROBOTS = ("wheelchair", "arm")
VARIANTS = ("baseline", "stsm")

if LEGACY not in sys.path:
    sys.path.insert(0, LEGACY)

import collect_metrics
import organize_results


def run_dir(run_id):
    return os.path.join(RESULTS, "runs", run_id)


def ensure_tree(run_id):
    organize_results.ensure_dirs(run_id)
    organize_results.write_results_readme()
    organize_results.ensure_summary_files()
    organize_results.ensure_paper_figure_source(run_id)
    os.makedirs(os.path.join(RESULTS, "figures"), exist_ok=True)


def command_organize(args):
    ensure_tree(args.run_id)
    rd = run_dir(args.run_id)
    organize_results.write_config(rd)
    organize_results.write_run_readme(
        rd, args.run_id, args.purpose, args.stage, args.status)
    organize_results.update_index(
        args.run_id, args.purpose, args.stage, args.status, args.notes)
    if args.update_latest:
        organize_results.update_latest(rd)
    print("organized {}".format(rd))


def command_collect(args):
    out = args.out or os.path.join(args.results_root, "summary", "all_metrics.csv")
    rows = collect_metrics.collect(args.results_root)
    collect_metrics.write_rows(rows, out)
    collect_metrics.write_robot_summary(rows, args.results_root, "wheelchair")
    collect_metrics.write_robot_summary(rows, args.results_root, "arm")
    collect_metrics.write_ablation_table(rows, args.results_root)
    collect_metrics.write_best_runs(rows, args.results_root)
    print("wrote {} ({} rows)".format(out, len(rows)))


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def write_csv(path, fields, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def command_fix_index(args):
    path = os.path.join(args.results_root, "summary", "experiment_index.csv")
    fields = ["run_id", "date", "stage", "purpose", "status", "main_result", "notes"]
    rows = {row.get("run_id", ""): row for row in read_csv(path)}
    defaults = {
        "20260701_R001": {
            "date": "2026-07-01",
            "stage": "wc_adp_verify",
            "purpose": "verify wheelchair ADP completion",
            "status": "valid",
            "main_result": "wheelchair_adp_valid",
            "notes": "Arm soft cost still zero; use as wheelchair ADP stage evidence",
        },
        "20260701_R002": {
            "date": "2026-07-01",
            "stage": "arm_dls_verify",
            "purpose": "verify ADP-aware DLS without cvxpy",
            "status": "valid",
            "main_result": "arm_dls_adp_active",
            "notes": "Wheelchair STSM failed no_progress; mechanical risk not improved",
        },
    }
    for run_id, data in defaults.items():
        row = rows.get(run_id, {"run_id": run_id})
        for key, value in data.items():
            if args.force or not row.get(key):
                row[key] = value
        rows[run_id] = row
    ordered = [rows[key] for key in sorted(rows) if key]
    write_csv(path, fields, ordered)
    print("updated {}".format(path))


def nonempty_config(path):
    rows = []
    for line in open(path, "r"):
        if "=" not in line:
            continue
        key, value = line.rstrip("\n").split("=", 1)
        rows.append((key, value))
    required = ("TARGET", "GUI", "RVIZ", "PLOT", "CLEAN_ENV")
    return all(value != "" for key, value in rows if key in required)


def command_validate(args):
    rd = run_dir(args.run_id)
    errors = []
    for rel in ("README.md", "manifest.csv", "config/launch_args.txt"):
        path = os.path.join(rd, rel)
        if not os.path.exists(path):
            errors.append("missing {}".format(path))
    launch_args = os.path.join(rd, "config", "launch_args.txt")
    if os.path.exists(launch_args) and not nonempty_config(launch_args):
        errors.append("empty required launch args in {}".format(launch_args))
    for robot in ROBOTS:
        for variant in VARIANTS:
            base = os.path.join(rd, robot, variant)
            for name in ("metrics.csv", "traj.csv"):
                path = os.path.join(base, name)
                if not os.path.exists(path):
                    errors.append("missing {}".format(path))
    if errors:
        for error in errors:
            print("ERROR: {}".format(error), file=sys.stderr)
        return 1
    print("validated {}".format(rd))
    return 0


def latest_row(path):
    rows = read_csv(path)
    return rows[-1] if rows else {}


def command_repair_run(args):
    rd = run_dir(args.run_id)
    if not os.path.isdir(rd):
        print("missing run directory: {}".format(rd), file=sys.stderr)
        return 1
    defaults = {
        "RUN_ID": args.run_id,
        "TARGET": args.target,
        "GUI": args.gui,
        "RVIZ": args.rviz,
        "PLOT": args.plot,
        "CLEAN_ENV": args.clean_env,
        "ADP_ENABLED": args.adp_enabled,
        "ADP_SOLVER_MODE": args.adp_solver_mode,
        "USE_CVXPY": args.use_cvxpy,
        "ADP_BLEND_ALPHA": args.adp_blend_alpha,
        "ADP_DESCENT_GAIN": args.adp_descent_gain,
        "LAMBDA_ADP": args.lambda_adp,
        "LAMBDA_ADP_CORRIDOR": args.lambda_adp_corridor,
        "LAMBDA_ADP_TERMINAL": args.lambda_adp_terminal,
        "LAMBDA_ADP_PATH": args.lambda_adp_path,
        "LAMBDA_ADP_ARM": args.lambda_adp_arm,
        "WC_COMPLETION_TOLERANCE": args.wc_completion_tolerance,
        "WC_REPLAN_PERIOD": args.wc_replan_period,
        "WC_NEAR_GOAL_RADIUS": args.wc_near_goal_radius,
        "WC_NEAR_GOAL_ADP_SCALE": args.wc_near_goal_adp_scale,
        "WC_NEAR_GOAL_GOAL_WEIGHT": args.wc_near_goal_goal_weight,
        "WC_NO_PROGRESS_REPLAN_TIME": args.wc_no_progress_replan_time,
        "WC_PROGRESS_REWARD_WEIGHT": args.wc_progress_reward_weight,
        "WC_FINAL_APPROACH_RADIUS": args.wc_final_approach_radius,
        "WC_FINAL_HEADING_THRESHOLD": args.wc_final_heading_threshold,
        "WC_FINAL_HEADING_GAIN": args.wc_final_heading_gain,
        "WC_FINAL_CREEP_V": args.wc_final_creep_v,
        "WC_FINAL_MIN_V": args.wc_final_min_v,
        "WC_FINAL_MAX_V": args.wc_final_max_v,
        "WC_FINAL_FORWARD_GAIN": args.wc_final_forward_gain,
        "WC_LAM_HEADING": args.wc_lam_heading,
        "WC_FINAL_DIRECT_OVERRIDE_ENABLED": args.wc_final_direct_override_enabled,
        "WC_FINAL_DIRECT_OVERRIDE_RADIUS": args.wc_final_direct_override_radius,
        "WC_MPC_HORIZON": args.wc_mpc_horizon,
        "WC_MPC_DT": args.wc_mpc_dt,
        "WC_MPC_A_MAX": args.wc_mpc_a_max,
        "WC_MPC_ALPHA_MAX": args.wc_mpc_alpha_max,
        "WC_MPC_BEAM_WIDTH": args.wc_mpc_beam_width,
    }
    cfg = os.path.join(rd, "config")
    os.makedirs(cfg, exist_ok=True)
    launch_args = os.path.join(cfg, "launch_args.txt")
    with open(launch_args, "w") as f:
        for key in sorted(defaults):
            f.write("{}={}\n".format(key, defaults[key]))

    fields = [
        "run_id", "robot", "variant", "mode", "start_time", "end_time",
        "status", "exit_code", "metrics_path", "traj_path", "log_path",
    ]
    manifest_rows = []
    for robot in ROBOTS:
        compare_rows = []
        for variant in VARIANTS:
            base = os.path.join(rd, robot, variant)
            metrics = os.path.join(base, "metrics.csv")
            traj = os.path.join(base, "traj.csv")
            log = os.path.join(base, "ros.log")
            if not os.path.exists(metrics) or not os.path.exists(traj):
                continue
            os.makedirs(base, exist_ok=True)
            if not os.path.exists(log):
                with open(log, "w") as f:
                    f.write("Migrated run: original ROS log was not available.\n")
            row = latest_row(metrics)
            mode = row.get("mode") or ("baseline" if variant == "baseline" else "stsm")
            status = "ok" if row else "unknown"
            manifest_rows.append({
                "run_id": args.run_id,
                "robot": robot,
                "variant": variant,
                "mode": mode,
                "start_time": "",
                "end_time": "",
                "status": status,
                "exit_code": "0" if status == "ok" else "",
                "metrics_path": os.path.relpath(metrics, rd),
                "traj_path": os.path.relpath(traj, rd),
                "log_path": os.path.relpath(log, rd),
            })
            if row:
                compare_rows.append(row)
        if compare_rows:
            compare_path = os.path.join(rd, "compare", "{}_compare_metrics.csv".format(robot))
            os.makedirs(os.path.dirname(compare_path), exist_ok=True)
            compare_fields = []
            for row in compare_rows:
                for key in row:
                    if key not in compare_fields:
                        compare_fields.append(key)
            write_csv(compare_path, compare_fields, compare_rows)
    write_csv(os.path.join(rd, "manifest.csv"), fields, manifest_rows)
    index_rows = {
        row.get("run_id"): row
        for row in read_csv(os.path.join(RESULTS, "summary", "experiment_index.csv"))
    }
    info = index_rows.get(args.run_id, {})
    with open(os.path.join(rd, "README.md"), "w") as f:
        f.write("# {}\n\n".format(args.run_id))
        f.write("## Purpose\n{}\n\n".format(info.get("purpose", "")))
        f.write("## Stage\n{}\n\n".format(info.get("stage", "")))
        f.write("## Status\n{}\n\n".format(info.get("status", "")))
        f.write("## Main result\n{}\n\n".format(info.get("main_result", "")))
        f.write("## Notes\n{}\n\n".format(info.get("notes", "")))
        f.write("## Variants\n")
        for robot in ROBOTS:
            for variant in VARIANTS:
                f.write("- {}/{}\n".format(robot, variant))
    print("repaired {}".format(rd))
    return 0


def command_archive_pictures(args):
    src = os.path.join(ROOT, "pictures")
    if not os.path.isdir(src):
        print("no pictures directory to archive")
        return 0
    archive = os.path.join(
        os.path.expanduser("~/stsm_madp_results_archive"),
        "old_pictures_{}".format(date.today().strftime("%Y%m%d")))
    os.makedirs(os.path.dirname(archive), exist_ok=True)
    if os.path.exists(archive):
        index = 1
        base = archive
        while os.path.exists(archive):
            archive = "{}_{:02d}".format(base, index)
            index += 1
    shutil.move(src, archive)
    print("archived pictures to {}".format(archive))
    return 0


def command_archive_legacy(args):
    archive = organize_results.archive_legacy_runs()
    if archive:
        print("archived legacy run directories to {}".format(archive))
    else:
        print("no legacy run directories to archive")
    return 0


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("organize")
    p.add_argument("--run-id", required=True)
    p.add_argument("--purpose", default="")
    p.add_argument("--stage", default="")
    p.add_argument("--status", default="")
    p.add_argument("--notes", default="")
    p.add_argument("--update-latest", action="store_true")
    p.set_defaults(func=command_organize)

    p = sub.add_parser("collect")
    p.add_argument("--results-root", default=RESULTS)
    p.add_argument("--out", default="")
    p.set_defaults(func=command_collect)

    p = sub.add_parser("fix-index")
    p.add_argument("--results-root", default=RESULTS)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=command_fix_index)

    p = sub.add_parser("validate")
    p.add_argument("--run-id", required=True)
    p.set_defaults(func=command_validate)

    p = sub.add_parser("repair-run")
    p.add_argument("--run-id", required=True)
    p.add_argument("--target", default="all")
    p.add_argument("--gui", default="false")
    p.add_argument("--rviz", default="false")
    p.add_argument("--plot", default="true")
    p.add_argument("--clean-env", default="true")
    p.add_argument("--adp-enabled", default="true")
    p.add_argument("--adp-solver-mode", default="dls_adp")
    p.add_argument("--use-cvxpy", default="false")
    p.add_argument("--adp-blend-alpha", default="0.08")
    p.add_argument("--adp-descent-gain", default="0.04")
    p.add_argument("--lambda-adp", default="0.005")
    p.add_argument("--lambda-adp-corridor", default="0.05")
    p.add_argument("--lambda-adp-terminal", default="0.0015")
    p.add_argument("--lambda-adp-path", default="0.005")
    p.add_argument("--lambda-adp-arm", default="0.008")
    p.add_argument("--wc-completion-tolerance", default="0.25")
    p.add_argument("--wc-replan-period", default="3.0")
    p.add_argument("--wc-near-goal-radius", default="0.40")
    p.add_argument("--wc-near-goal-adp-scale", default="0.20")
    p.add_argument("--wc-near-goal-goal-weight", default="18.0")
    p.add_argument("--wc-no-progress-replan-time", default="2.5")
    p.add_argument("--wc-progress-reward-weight", default="2.8")
    p.add_argument("--wc-final-approach-radius", default="0.90")
    p.add_argument("--wc-final-heading-threshold", default="0.45")
    p.add_argument("--wc-final-heading-gain", default="2.2")
    p.add_argument("--wc-final-creep-v", default="0.04")
    p.add_argument("--wc-final-min-v", default="0.12")
    p.add_argument("--wc-final-max-v", default="0.30")
    p.add_argument("--wc-final-forward-gain", default="0.75")
    p.add_argument("--wc-lam-heading", default="2.5")
    p.add_argument("--wc-final-direct-override-enabled", default="true")
    p.add_argument("--wc-final-direct-override-radius", default="0.90")
    p.add_argument("--wc-mpc-horizon", default="12")
    p.add_argument("--wc-mpc-dt", default="0.2")
    p.add_argument("--wc-mpc-a-max", default="0.5")
    p.add_argument("--wc-mpc-alpha-max", default="1.5")
    p.add_argument("--wc-mpc-beam-width", default="12")
    p.set_defaults(func=command_repair_run)

    p = sub.add_parser("archive-pictures")
    p.set_defaults(func=command_archive_pictures)

    p = sub.add_parser("archive-legacy-runs")
    p.set_defaults(func=command_archive_legacy)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    result = args.func(args)
    return int(result or 0)


if __name__ == "__main__":
    sys.exit(main())
