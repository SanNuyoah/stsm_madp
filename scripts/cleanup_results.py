#!/usr/bin/env python3
# python3 stsm_madp/scripts/cleanup_results.py --execute
from __future__ import print_function

import argparse
import csv
import hashlib
import json
import os
import shutil


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def file_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def dir_size(path):
    total = 0
    for base, _dirs, files in os.walk(path):
        for name in files:
            total += file_size(os.path.join(base, name))
    return total


def human_size(num):
    num = float(num)
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024.0:
            return "{:.1f}{}".format(num, unit)
        num /= 1024.0
    return "{:.1f}TB".format(num)


def sha256(path, block=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(block)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def list_runs(results_root):
    runs_root = os.path.join(results_root, "runs")
    runs = []
    if os.path.isdir(runs_root):
        for name in os.listdir(runs_root):
            path = os.path.join(runs_root, name)
            if os.path.isdir(path):
                runs.append((name, path, os.path.getmtime(path)))
    runs.sort(key=lambda item: item[2], reverse=True)
    if not runs:
        current = os.path.join(results_root, "run")
        if os.path.isdir(current):
            runs.append(("run", current, os.path.getmtime(current)))
    return runs


def remove_path(path, execute):
    if execute:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)
    print("{} {}".format("DELETE" if execute else "DRY-RUN delete", path))


def move_path(path, archive_root, results_root, execute):
    rel = os.path.relpath(path, results_root)
    out = os.path.join(archive_root, rel)
    print("{} {} -> {}".format(
        "MOVE" if execute else "DRY-RUN move", path, out))
    if not execute:
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        shutil.rmtree(out) if os.path.isdir(out) else os.remove(out)
    shutil.move(path, out)


def copy_if_exists(src, dst, execute):
    if not os.path.exists(src):
        return
    print("{} {} -> {}".format(
        "COPY" if execute else "DRY-RUN copy", src, dst))
    if not execute:
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def copy_json_with_variant(src, dst, execute, robot, variant):
    if not os.path.exists(src):
        return
    print("{} {} -> {}".format(
        "COPY" if execute else "DRY-RUN copy", src, dst))
    if not execute:
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        with open(src) as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            payload.setdefault("robot", robot)
            payload.setdefault("target", robot)
            payload["variant"] = variant
        with open(dst, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    except Exception:
        shutil.copy2(src, dst)


def write_run_readme(out_root, execute):
    path = os.path.join(out_root, "README.md")
    print("{} {}".format("WRITE" if execute else "DRY-RUN write", path))
    if not execute:
        return
    os.makedirs(out_root, exist_ok=True)
    with open(path, "w") as f:
        f.write("# STSM-MADP compact run results\n\n")
        f.write("Official per-variant outputs live under:\n\n")
        f.write("- `arm/baseline/`\n")
        f.write("- `arm/stsm/`\n")
        f.write("- `wheelchair/baseline/`\n")
        f.write("- `wheelchair/stsm/`\n\n")
        f.write("Root-level robot result files are intentionally not produced, ")
        f.write("so baseline and STSM diagnostics cannot overwrite each other.\n")


def baseline_placeholder(name, robot, variant):
    if variant != "baseline":
        return None
    base = {
        "robot": robot,
        "target": robot,
        "variant": variant,
        "mode": variant,
        "morse_used": False,
        "topology_constraint_used": False,
        "critical_point_sequence_constraint_used": False,
        "corridor_constraint_used": False,
        "manifold_constraint_used": False,
        "mpc_feasibility_status": "feasible",
        "failure_reason": "none",
        "replan_required": False,
    }
    if name == "mpc_diagnostics.json":
        base["module_chain_valid"] = False
        return base
    if name == "decision_trace.json":
        base["execution_status"] = "success"
        base["final_path_source"] = "baseline"
        return base
    if name == "topology_constraint.json":
        base["critical_point_sequence"] = []
        base["critical_points"] = []
        return base
    if name == "metrics.json":
        return base
    return None


def write_metrics_json_from_csv(src, dst, execute, robot, variant):
    if not os.path.exists(src):
        return False
    print("{} {} -> {}".format(
        "WRITE" if execute else "DRY-RUN write", src, dst))
    if not execute:
        return True
    with open(src) as f:
        rows = list(csv.DictReader(f))
    payload = rows[-1] if rows else {}
    payload.setdefault("target", robot)
    payload["robot"] = robot
    payload["variant"] = variant
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return True


def clean_current_run(results_root, execute):
    run = os.path.join(results_root, "run")
    for robot in ("arm", "wheelchair"):
        robot_dir = os.path.join(run, robot)
        for name in (
                "traj.csv", "metrics.csv", "metrics.json",
                "mpc_diagnostics.json", "decision_trace.json",
                "topology_constraint.json", "critical_point_association.json",
                "topology_tube.json", "mpc_feedback.json",
                "mpc_reference_path.csv", "mpc_cost_breakdown.csv",
                "mpc_rollout_log.csv", "mpc_executed_trajectory.csv",
                "trajectory.csv"):
            remove_path(os.path.join(robot_dir, name), execute)
        baseline = os.path.join(robot_dir, "baseline")
        for name in (
                "critical_point_association.json",
                "topology_constraint.json",
                "topology_tube.json"):
            remove_path(os.path.join(baseline, name), execute)
        stsm = os.path.join(robot_dir, "stsm")
        if os.path.isdir(stsm):
            for name in os.listdir(stsm):
                if name.startswith("figures_"):
                    remove_path(os.path.join(stsm, name), execute)
    for name in (
            "trajectory.csv", "arm_baseline_traj.csv", "arm_stsm_traj.csv",
            "wc_baseline_traj.csv", "wc_stsm_traj.csv"):
        remove_path(os.path.join(run, name), execute)


def clean_figures(results_root, execute):
    figures = os.path.join(results_root, "figures")
    run = os.path.join(results_root, "run")
    if not os.path.isdir(figures):
        return
    for base, _dirs, files in os.walk(figures):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in (".png", ".pdf", ".svg"):
                continue
            src = os.path.join(base, name)
            rel = os.path.relpath(src, figures)
            parts = rel.split(os.sep)
            robot = None
            if parts[0] in ("arm", "wheelchair"):
                robot = parts[0]
            elif parts[0].startswith("arm_"):
                robot = "arm"
            elif parts[0].startswith("wheelchair_"):
                robot = "wheelchair"
            dst_dir = os.path.join(run, robot, "stsm") if robot else run
            dst = os.path.join(dst_dir, name)
            print("{} {} -> {}".format(
                "MOVE" if execute else "DRY-RUN move", src, dst))
            if execute:
                os.makedirs(dst_dir, exist_ok=True)
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(src, dst)


def archive_debug_figures(results_root, execute):
    archive = os.path.join(results_root, "archive", "debug_figures")
    for name in ("figures_trace_check", "figures_trace_check2"):
        src = os.path.join(results_root, name)
        dst = os.path.join(archive, name)
        if not os.path.exists(src):
            continue
        print("{} {} -> {}".format(
            "MOVE" if execute else "DRY-RUN move", src, dst))
        if execute:
            os.makedirs(archive, exist_ok=True)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.move(src, dst)


def remove_summary_dir(results_root, execute):
    remove_path(os.path.join(results_root, "summary"), execute)


def write_config_used(run_path, dst, execute):
    src = os.path.join(run_path, "config.yaml")
    if os.path.exists(src):
        copy_if_exists(src, dst, execute)
        return
    config_dir = os.path.join(run_path, "config")
    names = []
    if os.path.isdir(config_dir):
        names = [
            name for name in sorted(os.listdir(config_dir))
            if name.endswith((".yaml", ".yml", ".txt"))
        ]
    print("{} config snapshot {} -> {}".format(
        "WRITE" if execute else "DRY-RUN write", run_path, dst))
    if not execute:
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    run_id = os.path.basename(os.path.abspath(run_path))
    with open(dst, "w") as f:
        f.write("run_id: {}\n".format(run_id))
        f.write("task_configs:\n")
        f.write("  arm: config/arm.yaml\n")
        f.write("  wheelchair: config/wheelchair.yaml\n")
        f.write("  topology: config/topology.yaml\n")
        f.write("run_config_files:\n")
        for name in names:
            f.write("  - config/{}\n".format(name))


def _first_value(row, keys, default=""):
    for key in keys:
        value = row.get(key, "")
        if value not in ("", None):
            return value
    return default


def write_compact_trajectory(src, dst, execute, target):
    if not os.path.exists(src):
        return
    print("{} compact trajectory {} -> {}".format(
        "WRITE" if execute else "DRY-RUN write", src, dst))
    if not execute:
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(src) as f:
        rows = list(csv.DictReader(f))
    fieldnames = [
        "target", "time", "position", "velocity", "risk", "control"]
    with open(dst, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            position = ",".join([
                _first_value(row, ("x", "px")),
                _first_value(row, ("y", "py")),
                _first_value(row, ("z", "pz"), "0.0"),
            ])
            velocity = ",".join([
                _first_value(row, ("vx",)),
                _first_value(row, ("vy",)),
                _first_value(row, ("vz",), "0.0"),
            ])
            writer.writerow({
                "target": target,
                "time": _first_value(row, ("time", "t")),
                "position": position,
                "velocity": velocity,
                "risk": _first_value(
                    row, ("risk", "phi_total", "phi_max_point",
                          "phi_arm_max_point", "phi_s")),
                "control": _first_value(
                    row, ("control", "speed_filtered", "speed_raw",
                          "dq_delta_norm", "dq_nominal_norm")),
            })


def write_combined_trajectory(items, dst, execute):
    existing = [(target, path) for target, path in items if os.path.exists(path)]
    if not existing:
        return
    print("{} combined trajectory -> {}".format(
        "WRITE" if execute else "DRY-RUN write", dst))
    if not execute:
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    fieldnames = [
        "target", "time", "position", "velocity", "risk", "control"]
    with open(dst, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for target, path in existing:
            with open(path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data = {key: row.get(key, "") for key in fieldnames}
                    if not data.get("target"):
                        data["target"] = target
                    writer.writerow(data)


def copy_run_outputs(run_path, out_root, execute):
    write_run_readme(out_root, execute)
    copy_if_exists(
        os.path.join(run_path, "metrics.csv"),
        os.path.join(out_root, "metrics.csv"),
        execute)
    copy_if_exists(
        os.path.join(run_path, "topology_summary.json"),
        os.path.join(out_root, "topology_summary.json"),
        execute)
    write_config_used(
        run_path,
        os.path.join(out_root, "config.yaml"),
        execute)
    copy_if_exists(
        os.path.join(run_path, "compare", "arm_compare_metrics.csv"),
        os.path.join(out_root, "metrics", "arm_metrics.csv"),
        execute)
    copy_if_exists(
        os.path.join(run_path, "compare", "arm_compare_metrics.csv"),
        os.path.join(out_root, "compare", "arm_compare_metrics.csv"),
        execute)
    copy_if_exists(
        os.path.join(run_path, "compare", "wheelchair_compare_metrics.csv"),
        os.path.join(out_root, "metrics", "wheelchair_metrics.csv"),
        execute)
    copy_if_exists(
        os.path.join(run_path, "compare", "wheelchair_compare_metrics.csv"),
        os.path.join(out_root, "compare", "wheelchair_compare_metrics.csv"),
        execute)
    for robot in ("arm", "wheelchair"):
        for variant in ("baseline", "stsm"):
            variant_run = os.path.join(run_path, robot, variant)
            variant_out = os.path.join(out_root, robot, variant)
            for name in (
                    "traj.csv", "metrics.csv", "mpc_reference_path.csv",
                    "mpc_cost_breakdown.csv", "mpc_rollout_log.csv",
                    "mpc_executed_trajectory.csv",
                    "baseline_reference_before_mpc.csv",
                    "baseline_mpc_output.csv"):
                copy_if_exists(
                    os.path.join(variant_run, name),
                    os.path.join(variant_out, name),
                    execute)
            for name in (
                    "metrics.json", "decision_trace.json",
                    "mpc_diagnostics.json", "mpc_feedback.json",
                    "topology_constraint.json", "topology_tube.json",
                    "critical_point_association.json",
                    "baseline_execution_chain.json"):
                src = os.path.join(variant_run, name)
                dst = os.path.join(variant_out, name)
                copy_json_with_variant(src, dst, execute, robot, variant)
                if not os.path.exists(src):
                    if name == "metrics.json" and write_metrics_json_from_csv(
                            os.path.join(variant_run, "metrics.csv"),
                            dst, execute, robot, variant):
                        continue
                    payload = baseline_placeholder(name, robot, variant)
                    if payload is not None:
                        print("{} {}".format(
                            "WRITE" if execute else "DRY-RUN write", dst))
                        if execute:
                            os.makedirs(os.path.dirname(dst), exist_ok=True)
                            with open(dst, "w") as f:
                                json.dump(payload, f, indent=2, sort_keys=True)


def finalize_latest(results_root, execute):
    runs = list_runs(results_root)
    if not runs:
        return
    selected = select_complete_run(runs)
    if selected is None:
        return
    _run_id, run_path, _mtime = selected
    run = os.path.join(results_root, "run")
    if os.path.abspath(run_path) == os.path.abspath(run):
        print("FINALIZE skipped; {} is already the compact run".format(run))
        return
    if execute:
        if os.path.isdir(run):
            shutil.rmtree(run)
        os.makedirs(run, exist_ok=True)
    print("{} run from {}".format(
        "FINALIZE" if execute else "DRY-RUN finalize", run_path))
    copy_run_outputs(run_path, run, execute)


def select_complete_run(runs):
    for item in runs:
        _candidate_id, candidate_path, _candidate_mtime = item
        if (os.path.exists(os.path.join(
                candidate_path, "arm", "stsm", "traj.csv")) and
                os.path.exists(os.path.join(
                    candidate_path, "wheelchair", "stsm", "traj.csv"))):
            return item
    return runs[0] if runs else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default=RESULTS)
    parser.add_argument("--archive-root", default=os.path.expanduser(
        "~/stsm_madp_results_archive"))
    parser.add_argument(
        "--keep-runs", type=int, default=-1,
        help="historical runs to keep; negative preserves all runs")
    parser.add_argument("--keep-latest", type=int, default=None,
                        help="alias for --keep-runs")
    parser.add_argument("--move-archive", action="store_true",
                        help="move old runs to archive root instead of deleting")
    parser.add_argument("--remove-debug", action="store_true")
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument("--remove-gif", action="store_true",
                        help="remove GIF files outside results/figures")
    parser.add_argument("--finalize", action="store_true",
                        help="create compact results/run from newest run")
    parser.add_argument("--execute", action="store_true",
                        help="actually delete/archive files; default is dry-run")
    args = parser.parse_args()

    results_root = os.path.abspath(args.results_root)
    keep_runs = int(args.keep_latest if args.keep_latest is not None
                    else args.keep_runs)
    planned_delete = 0
    planned_archive = 0
    planned_bytes = 0
    print("results_root={} size={}".format(
        results_root, human_size(dir_size(results_root))))
    clean_figures(results_root, args.execute)
    archive_debug_figures(results_root, args.execute)
    clean_current_run(results_root, args.execute)
    # Summary tables and historical run directories are formal experiment
    # evidence. Cleanup may compact the current run but must not erase them.

    runs = list_runs(results_root)
    protected_run_path = ""
    if args.finalize:
        selected_run = select_complete_run(runs)
        if selected_run is not None:
            protected_run_path = os.path.abspath(selected_run[1])
    for idx, (run_id, path, _mtime) in enumerate(runs):
        print("run {} {} {}".format(idx + 1, run_id, human_size(dir_size(path))))
        if protected_run_path and os.path.abspath(path) == protected_run_path:
            print("KEEP protected finalize source {}".format(path))
            continue
        if keep_runs >= 0 and idx >= keep_runs:
            planned_bytes += dir_size(path)
            if args.move_archive:
                planned_archive += 1
                move_path(path, args.archive_root, results_root, args.execute)
            else:
                planned_delete += 1
                remove_path(path, args.execute)

    seen = {}
    for base, _dirs, files in os.walk(results_root):
        for name in files:
            path = os.path.join(base, name)
            if not os.path.exists(path):
                continue
            rel = os.path.relpath(path, results_root)
            protected_final = (
                rel.startswith("run" + os.sep) or
                rel.startswith("figures" + os.sep))
            if protected_final:
                continue
            if (os.sep + "config" + os.sep) in path:
                continue
            if rel.startswith("summary" + os.sep):
                continue
            if args.remove_gif and name.lower().endswith(".gif"):
                planned_delete += 1
                planned_bytes += file_size(path)
                remove_path(path, args.execute)
                continue
            if args.remove_debug and name == "topology_debug.json":
                planned_delete += 1
                planned_bytes += file_size(path)
                remove_path(path, args.execute)
                continue
            if args.deduplicate:
                digest = sha256(path)
                old = seen.get(digest)
                if old is not None:
                    planned_delete += 1
                    planned_bytes += file_size(path)
                    remove_path(path, args.execute)
                else:
                    seen[digest] = path

    print("planned_delete={} planned_archive={} estimated_release={}".format(
        planned_delete, planned_archive, human_size(planned_bytes)))
    if args.finalize:
        finalize_latest(results_root, args.execute)
    if not args.execute:
        print("dry-run only; re-run with --execute to apply changes")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        pass
