#!/usr/bin/env python3
import argparse
import csv
import os
import sys
sys.dont_write_bytecode = True


VARIANTS = ("baseline", "stsm")
ROBOTS = ("wheelchair", "arm")

FIELDS = [
    "run_id", "robot", "variant", "mode", "success_safe",
    "success_goal", "stop_triggered", "stop_reason", "duration_s",
    "arm_reached_hand", "arm_hold_completed", "arm_hold_duration_s",
    "arm_retreat_started", "arm_wait_reached",
    "arm_home_return_started", "arm_home_returned",
    "arm_task_complete", "arm_min_hand_dist", "arm_min_wait_dist",
    "arm_min_home_return_dist",
    "J_social", "path_length_m", "risk_per_meter", "risk_mean_time",
    "risk_p95", "risk_exceed_pct", "mean_phi_s", "final_dist_to_goal",
    "min_head_dist", "min_chest_dist", "min_person_dist",
    "topology_used", "topology_fallback_used", "candidate_corridor_count",
    "num_candidate_corridors", "selected_corridor_label",
    "execution_corridor_id", "corridor_id", "selected_corridor_type",
    "selected_refinement_used", "selected_refined_path_length",
    "selected_topology_diversity", "selected_tracking_cost",
    "selected_max_curvature", "selected_curvature_violation",
    "mean_adp_value", "max_adp_value", "terminal_adp_cost",
    "corridor_adp_norm", "corridor_adp_raw_mean", "corridor_rank_base",
    "corridor_rank_total", "corridor_rank_changed_count",
    "final_approach_used",
    "arm_adp_grad_norm", "arm_adp_soft_cost",
    "arm_v_adp_alignment", "arm_dls_adp_used", "arm_qp_used",
    "arm_solver_success_rate", "v_des_raw_norm", "v_des_adp_norm",
    "v_des_delta_norm", "dq_nominal_norm", "dq_adp_norm",
    "dq_delta_norm", "critic_version",
    "risk_field_used", "manifold_used", "morse_used",
    "topology_graph_used", "candidate_corridor_used",
    "candidate_ranking_used", "fallback_used", "selected_corridor_id",
    "selected_rank", "selected_total_score", "total_score",
    "selection_override_reason", "raw_waypoints_count",
    "refined_waypoints_count", "mpc_used", "mpc_reference_source",
    "adp_used", "adp_role", "adp_affects_candidate_ranking",
    "adp_affects_control", "final_path_source",
    "module_chain_valid", "selection_consistent",
    "refinement_trace_valid",
]
ABLATION_FIELDS = [
    "run_id", "robot", "variant", "success_goal", "success_safe",
    "arm_task_complete", "arm_reached_hand", "arm_hold_completed",
    "arm_wait_reached", "arm_home_returned",
    "duration_s", "path_length_m", "J_social", "risk_per_meter",
    "risk_exceed_pct", "topology_fallback_used",
    "selected_refinement_used", "selected_refined_path_length",
    "mean_adp_value",
    "max_adp_value", "terminal_adp_cost", "corridor_adp_norm",
    "corridor_rank_changed_count", "final_approach_used", "arm_adp_grad_norm",
    "arm_adp_soft_cost", "v_des_delta_norm", "dq_delta_norm",
    "critic_version", "module_chain_valid", "selection_consistent",
    "refinement_trace_valid", "adp_role",
    "adp_affects_candidate_ranking", "adp_affects_control",
]
BEST_FIELDS = [
    "robot", "variant", "run_id", "success_goal", "success_safe",
    "arm_task_complete", "duration_s", "J_social", "risk_per_meter",
    "risk_exceed_pct",
]


def read_last_row(path):
    with open(path, "r") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def collect(results_root):
    runs_dir = os.path.join(results_root, "runs")
    rows = []
    if not os.path.isdir(runs_dir):
        return rows
    for run_id in sorted(os.listdir(runs_dir)):
        run_dir = os.path.join(runs_dir, run_id)
        if not os.path.isdir(run_dir):
            continue
        for robot in ROBOTS:
            for variant in VARIANTS:
                path = os.path.join(run_dir, robot, variant, "metrics.csv")
                if not os.path.exists(path):
                    continue
                row = read_last_row(path)
                if not row:
                    continue
                out = {key: row.get(key, "") for key in FIELDS}
                out["run_id"] = row.get("run_id") or run_id
                out["robot"] = robot
                out["variant"] = row.get("variant") or variant
                rows.append(out)
    return rows


def write_rows(rows, out_path):
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDS})


def write_robot_summary(rows, results_root, robot):
    out_path = os.path.join(results_root, "summary", "%s_metrics_summary.csv" % robot)
    write_rows([row for row in rows if row.get("robot") == robot], out_path)


def num(row, key, default=0.0):
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def write_table(rows, path, fields):
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_ablation_table(rows, results_root):
    path = os.path.join(results_root, "summary", "ablation_table.csv")
    ordered = sorted(rows, key=lambda row: (
        row.get("robot", ""), row.get("variant", ""), row.get("run_id", "")))
    write_table(ordered, path, ABLATION_FIELDS)


def write_best_runs(rows, results_root):
    best = {}
    for row in rows:
        key = (row.get("robot", ""), row.get("variant", ""))
        score = (
            -num(row, "success_goal"),
            -num(row, "success_safe"),
            num(row, "risk_exceed_pct", 1e9),
            num(row, "J_social", 1e9),
            num(row, "duration_s", 1e9),
        )
        if key not in best or score < best[key][0]:
            best[key] = (score, row)
    selected = [item[1] for item in sorted(best.values(), key=lambda item: (
        item[1].get("robot", ""), item[1].get("variant", "")))]
    path = os.path.join(results_root, "summary", "best_runs.csv")
    write_table(selected, path, BEST_FIELDS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results"))
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    out = args.out or os.path.join(args.results_root, "summary", "all_metrics.csv")
    rows = collect(args.results_root)
    write_rows(rows, out)
    write_robot_summary(rows, args.results_root, "wheelchair")
    write_robot_summary(rows, args.results_root, "arm")
    write_ablation_table(rows, args.results_root)
    write_best_runs(rows, args.results_root)
    print("wrote %s (%d rows)" % (out, len(rows)))


if __name__ == "__main__":
    main()
