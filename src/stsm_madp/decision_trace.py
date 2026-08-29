import sys
sys.dont_write_bytecode = True

import json
import os

from .adp import adp_role_from_runtime


MODULE_FLAGS = (
    "risk_field_used",
    "manifold_used",
    "morse_used",
    "topology_graph_used",
    "candidate_corridor_used",
    "candidate_ranking_used",
    "refinement_used",
    "mpc_used",
    "adp_used",
    "fallback_used",
)

STEP_NAMES = (
    "RiskField",
    "SafetyManifold",
    "MorseCriticalPoints",
    "TopologyGraph",
    "CandidateCorridors",
    "CandidateRanking",
    "SelectedCorridor",
    "Refinement",
    "MPC",
    "ADP",
)

STEP_TO_FLAG = {
    "RiskField": "risk_field_used",
    "SafetyManifold": "manifold_used",
    "MorseCriticalPoints": "morse_used",
    "TopologyGraph": "topology_graph_used",
    "CandidateCorridors": "candidate_corridor_used",
    "CandidateRanking": "candidate_ranking_used",
    "Refinement": "refinement_used",
    "MPC": "mpc_used",
    "ADP": "adp_used",
}


def _num(value, default=0.0):
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def init_trace(robot="", variant=""):
    trace = {
        "robot": str(robot or ""),
        "variant": str(variant or ""),
        "selected_corridor_id": "",
        "corridor_id": "",
        "execution_corridor_id": "",
        "selected_corridor_type": "",
        "topology_route_class": "",
        "task_semantic_class": "",
        "selected_rank": 0,
        "selected_total_score": 0.0,
        "total_score": 0.0,
        "selection_override_reason": "",
        "selection_consistent": 0,
        "raw_waypoints_count": 0,
        "refined_waypoints_count": 0,
        "refinement_failed_reason": "",
        "mpc_reference_source": "",
        "mpc_tracking_cost": 0.0,
        "mpc_social_cost": 0.0,
        "mpc_input_waypoint_count": 0,
        "mpc_feasibility_status": "",
        "mpc_total_cost": 0.0,
        "mpc_affects_candidate_ranking": 0,
        "adp_role": "disabled",
        "adp_affects_candidate_ranking": 0,
        "adp_affects_control": 0,
        "adp_decision_influence_enabled": 0,
        "adp_ranking_influence_enabled": 0,
        "adp_mpc_influence_enabled": 0,
        "mpc_adp_enabled": 0,
        "adp_effective_lambda": 0.0,
        "final_path_source": "",
        "execution_status": "",
        "failure_stage": "",
        "stop_reason": "",
        "success_goal": "",
        "topology_fallback_used": 0,
        "planning_failed": 0,
        "steps": {},
        "module_chain": [],
    }
    for flag in MODULE_FLAGS:
        trace[flag] = 0
    for name in STEP_NAMES:
        trace["steps"][name] = {
            "module_name": name,
            "used": 0,
            "input_source": "",
            "output_target": "",
            "status": "not_used",
            "evidence_file": "",
            "failure_reason": "",
        }
    return trace


def set_step(trace, name, used=False, input_source="", output_target="",
             status="", evidence_file=""):
    if "steps" not in trace:
        trace["steps"] = {}
    if name not in trace["steps"]:
        trace["steps"][name] = {}
    failure_reason = "" if used else str(status or "not_used")
    trace["steps"][name].update({
        "module_name": str(name),
        "used": 1 if used else 0,
        "input_source": str(input_source or ""),
        "output_target": str(output_target or ""),
        "status": str(status or ("used" if used else "not_used")),
        "evidence_file": str(evidence_file or ""),
        "failure_reason": failure_reason,
    })
    flag = STEP_TO_FLAG.get(name)
    if flag:
        trace[flag] = 1 if used else 0
    return trace


def mark_used(trace, module, used=True):
    key = "%s_used" % str(module).strip().lower()
    if key in trace:
        trace[key] = 1 if used else 0
    return trace


def set_selected_corridor(trace, corridor_id, selected_rank=0,
                          override_reason=""):
    trace["selected_corridor_id"] = str(corridor_id or "")
    trace["corridor_id"] = str(corridor_id or "")
    trace["execution_corridor_id"] = str(corridor_id or "")
    trace["selected_rank"] = int(_num(selected_rank, 0))
    trace["selection_override_reason"] = str(override_reason or "")
    return trace


def set_refinement(trace, used=False, raw_count=0, refined_count=0,
                   failure_reason=""):
    trace["refinement_used"] = 1 if used else 0
    trace["raw_waypoints_count"] = int(_num(raw_count, 0))
    trace["refined_waypoints_count"] = int(_num(refined_count, 0))
    trace["refinement_failed_reason"] = str(failure_reason or "")
    return trace


def set_mpc(trace, used=False, reference_source="", tracking_cost=0.0,
            social_cost=0.0, input_waypoint_count=0):
    trace["mpc_used"] = 1 if used else 0
    trace["mpc_reference_source"] = str(reference_source or "")
    trace["mpc_tracking_cost"] = float(_num(tracking_cost, 0.0))
    trace["mpc_social_cost"] = float(_num(social_cost, 0.0))
    trace["mpc_input_waypoint_count"] = int(_num(input_waypoint_count, 0))
    return trace


def set_adp(trace, used=False, role="disabled",
            affects_candidate_ranking=False, affects_control=False):
    trace["adp_used"] = 1 if used else 0
    trace["adp_role"] = str(role or ("evaluation_only" if used else "disabled"))
    trace["adp_affects_candidate_ranking"] = int(bool(affects_candidate_ranking))
    trace["adp_affects_control"] = int(bool(affects_control))
    return trace


def _rows(debug):
    return list((debug or {}).get("candidate_corridors", []) or [])


def trace_from_debug(debug, metrics=None, robot="", variant="stsm"):
    debug = dict(debug or {})
    metrics = dict(metrics or {})
    trace = init_trace(robot or metrics.get("target") or metrics.get("robot"), variant)
    rows = _rows(debug)
    metric_execution_id = str(
        metrics.get("execution_corridor_id") or
        metrics.get("corridor_id") or
        metrics.get("selected_corridor_id") or "")
    selected_id = str(
        metric_execution_id or
        debug.get("execution_corridor_id") or
        debug.get("selected_corridor_id") or "")
    selected = None
    for row in rows:
        cid = str(row.get("corridor_id") or row.get("label") or "")
        if cid == selected_id or bool(row.get("selected")):
            selected = row
            selected_id = cid
            break
    rank = int(_num((selected or {}).get("rank"), 0))
    if not rank and rows and selected_id:
        for idx, row in enumerate(sorted(
                rows, key=lambda r: _num(r.get("total_score", r.get("cost")), 0.0)),
                start=1):
            if str(row.get("corridor_id") or row.get("label") or "") == selected_id:
                rank = idx
                break

    trace["risk_field_used"] = 1 if rows or _num(metrics.get("J_social"), 0.0) > 0 else 0
    trace["manifold_used"] = 1 if _num(debug.get("num_safe_minima"), 0) or _num(debug.get("num_safe_saddles"), 0) or rows else 0
    trace["morse_used"] = 1 if _num(debug.get("num_used_saddles"), 0) or _num(debug.get("num_used_minima"), 0) or (selected and selected.get("morse_node_ids")) else 0
    trace["topology_graph_used"] = 1 if _num(debug.get("num_topology_nodes"), 0) and rows else 0
    trace["candidate_corridor_used"] = 1 if len(rows) > 0 else 0
    trace["candidate_ranking_used"] = 1 if selected_id and len(rows) > 0 else 0
    trace["topology_fallback_used"] = int(_num(
        metrics.get("topology_fallback_used",
                    debug.get("topology_fallback_used", 0)), 0))
    trace["fallback_used"] = int(_num(
        metrics.get("fallback_used",
                    debug.get("fallback_used",
                              trace.get("topology_fallback_used", 0))), 0))
    set_selected_corridor(
        trace, selected_id, rank,
        debug.get("selection_override_reason") or
        (selected or {}).get("selection_override_reason") or "")
    trace["selected_corridor_type"] = str(
        metrics.get("selected_corridor_type") or
        debug.get("selected_corridor_type") or
        ("morse_topology_graph"
         if selected_id and selected_id != "planning_failed" else ""))
    trace["topology_route_class"] = str(
        metrics.get("topology_route_class") or
        debug.get("selected_topology_route_class") or
        (selected or {}).get("topology_route_class") or "")
    trace["task_semantic_class"] = str(
        metrics.get("task_semantic_class") or
        debug.get("selected_task_semantic_class") or
        (selected or {}).get("task_semantic_class") or "")
    trace["execution_status"] = str(metrics.get("execution_status") or "")
    trace["failure_stage"] = str(metrics.get("failure_stage") or "")
    trace["stop_reason"] = str(metrics.get("stop_reason") or "")
    trace["success_goal"] = metrics.get("success_goal", "")
    if selected_id == "planning_failed":
        trace["planning_failed"] = 1
        if not trace["execution_status"]:
            trace["execution_status"] = "failed"
        if not trace["failure_stage"]:
            trace["failure_stage"] = "planning"

    raw = (selected or {}).get("raw_topology_waypoints") or (selected or {}).get("topology_ordered_waypoints") or []
    refined = (selected or {}).get("refined_waypoints") or []
    raw_count = int(_num((selected or {}).get("raw_waypoints_count"), 0))
    refined_count = int(_num((selected or {}).get("refined_waypoints_count"), 0))
    if raw_count <= 0:
        raw_count = len(raw)
    if refined_count <= 0:
        refined_count = len(refined)
    if raw_count <= 0:
        raw_count = int(_num(
            debug.get("selected_raw_waypoints_count",
                      metrics.get("raw_waypoints_count", 0)), 0))
    if refined_count <= 0:
        refined_count = int(_num(
            debug.get("selected_refined_waypoints_count",
                      metrics.get("refined_waypoints_count", 0)), 0))
    set_refinement(
        trace,
        bool(_num((selected or {}).get("refinement_used", metrics.get("selected_refinement_used")), 0)),
        raw_count, refined_count,
        (selected or {}).get("refinement_reject_reason", ""))
    set_mpc(
        trace,
        bool(debug.get("mpc_used", 1 if selected_id else 0)),
        debug.get("mpc_reference_source") or ("refined_waypoints" if trace["refinement_used"] else "raw_waypoints"),
        debug.get("mpc_tracking_cost", metrics.get("mpc_tracking_cost", metrics.get("mpc_track_cost", metrics.get("selected_tracking_cost", 0)))),
        debug.get("mpc_social_cost", metrics.get("mpc_risk_cost", metrics.get("mpc_social_cost", 0))),
        debug.get("mpc_input_waypoint_count", trace["refined_waypoints_count"] or trace["raw_waypoints_count"]))
    trace["mpc_feasibility_status"] = str(
        metrics.get("mpc_feasibility_status") or
        debug.get("mpc_feasibility_status") or "feasible")
    trace["mpc_total_cost"] = _num(
        metrics.get("mpc_total_cost", debug.get("mpc_total_cost", 0.0)), 0.0)
    trace["mpc_affects_candidate_ranking"] = int(_num(
        metrics.get("mpc_affects_candidate_ranking",
                    debug.get("mpc_affects_candidate_ranking", 0)), 0))
    trace["mpc_candidate_feasibility_used"] = int(_num(
        metrics.get("mpc_candidate_feasibility_used",
                    debug.get("mpc_candidate_feasibility_used", 0)), 0))
    trace["mpc_execution_cost_in_score"] = int(_num(
        metrics.get("mpc_execution_cost_in_score",
                    debug.get("mpc_execution_cost_in_score", 0)), 0))
    for key in (
            "risk_query_source", "risk_sanity_status",
            "mpc_rollout_mode", "mpc_rollout_log_file",
            "mpc_executed_trajectory_file"):
        trace[key] = str(metrics.get(key) or debug.get(key) or "")
    trace["mpc_rollout_solve_count"] = int(_num(
        metrics.get("mpc_rollout_solve_count",
                    debug.get("mpc_rollout_solve_count", 0)), 0))
    trace["mpc_executed_trajectory_count"] = int(_num(
        metrics.get("mpc_executed_trajectory_count",
                    debug.get("mpc_executed_trajectory_count", 0)), 0))
    trace["mpc_rollout_horizon_rows"] = int(_num(
        metrics.get("mpc_rollout_horizon_rows",
                    debug.get("mpc_rollout_horizon_rows", 0)), 0))
    trace["risk_query_called"] = int(_num(
        metrics.get("risk_query_called", debug.get("risk_query_called", 0)), 0))
    trace["risk_query_valid_count"] = int(_num(
        metrics.get("risk_query_valid_count",
                    debug.get("risk_query_valid_count", 0)), 0))
    trace["mpc_risk_cost"] = _num(
        metrics.get("mpc_risk_cost", debug.get("mpc_risk_cost", 0.0)), 0.0)
    trace["mpc_max_risk"] = _num(
        metrics.get("mpc_max_risk", debug.get("mpc_max_risk", 0.0)), 0.0)
    adp_enabled = bool(_num(
        metrics.get("adp_enabled", debug.get("adp_used", 0)), 0))
    learning_enabled = bool(_num(
        metrics.get("adp_learning_enabled",
                    debug.get("adp_learning_enabled", adp_enabled)), 0))
    influence_enabled = bool(_num(
        metrics.get("adp_decision_influence_enabled",
                    debug.get("adp_decision_influence_enabled", 0)), 0))
    effective_lambda = _num(
        metrics.get("adp_effective_lambda",
                    debug.get("adp_effective_lambda", 0.0)), 0.0)
    ranking_signal = bool(_num(
        metrics.get("adp_affects_candidate_ranking",
                    debug.get("adp_affects_candidate_ranking",
                              metrics.get("corridor_rank_changed_count", 0))), 0))
    control_signal = bool(_num(
        metrics.get("adp_affects_control",
                    debug.get("adp_affects_control",
                              metrics.get("arm_dls_adp_used", 0))), 0))
    role = adp_role_from_runtime(
        adp_enabled, learning_enabled, influence_enabled,
        effective_lambda=effective_lambda,
        ranking_contribution=ranking_signal,
        control_contribution=control_signal)
    affects_ranking = role in (
        "ranking_modifier", "ranking_and_control_modifier")
    affects_control = role in (
        "control_modifier", "ranking_and_control_modifier")
    set_adp(
        trace,
        adp_enabled,
        role,
        affects_candidate_ranking=affects_ranking,
        affects_control=affects_control)
    trace["adp_decision_influence_enabled"] = int(influence_enabled)
    trace["adp_ranking_influence_enabled"] = int(_num(
        metrics.get("adp_ranking_influence_enabled",
                    debug.get("adp_ranking_influence_enabled", 0)), 0))
    trace["adp_mpc_influence_enabled"] = int(_num(
        metrics.get("adp_mpc_influence_enabled",
                    debug.get("adp_mpc_influence_enabled", 0)), 0))
    trace["mpc_adp_enabled"] = int(_num(
        metrics.get("mpc_adp_enabled", trace["adp_mpc_influence_enabled"]), 0))
    trace["adp_effective_lambda"] = float(effective_lambda)
    trace["final_path_source"] = debug.get("final_path_source") or (
        "Morse->Candidate->Ranking->Refinement->MPC"
        if trace["mpc_reference_source"] == "refined_waypoints"
        else "Morse->Candidate->Ranking->RawWaypoints->MPC")
    selected_total_score = _num(
        metrics.get("selected_total_score",
                    debug.get("selected_candidate_total_score",
                              (selected or {}).get(
                                  "total_score",
                                  metrics.get("total_score", "")))),
        0.0)
    trace["selected_total_score"] = selected_total_score
    trace["total_score"] = selected_total_score
    _populate_steps(trace, debug, metrics, selected)
    finalize_trace(trace)
    return trace


def _populate_steps(trace, debug, metrics, selected):
    selected = selected or {}
    robot = str(trace.get("robot") or metrics.get("target") or metrics.get("robot") or "")
    run_prefix = "results/run/{}".format(robot) if robot else "results/run"
    fig_prefix = "results/figures/{}".format(robot) if robot else "results/figures"
    topology_png = "results/figures/{}_topology_graph.png".format(robot) if robot else "results/figures/topology_graph.png"
    trajectory_file = "{}/trajectory.csv".format(run_prefix)
    metrics_file = "{}/metrics.csv".format(run_prefix)
    mpc_reference_file = "{}/mpc_reference_path.csv".format(run_prefix)
    candidate_corridors_file = "{}/candidate_corridors.json".format(fig_prefix)
    candidate_ranking_file = "{}/candidate_ranking.csv".format(fig_prefix)
    selected_corridor_file = "{}/selected_corridor.json".format(fig_prefix)
    topology_summary_file = "results/figures/{}_topology_summary.json".format(robot) if robot else "results/figures/topology_summary.json"
    morse_file = "results/figures/{}_morse_points_lifecycle.csv".format(robot) if robot else "results/figures/morse_points_lifecycle.csv"
    set_step(
        trace, "RiskField", bool(trace.get("risk_field_used")),
        "social_field.py:phi_s", "candidate risk_cost and trajectory risk",
        "used" if trace.get("risk_field_used") else "not_used",
        metrics.get("risk_time_series_file", trajectory_file))
    set_step(
        trace, "SafetyManifold", bool(trace.get("manifold_used")),
        "SafetyManifold safe sublevel set", "topology graph feasible nodes",
        "used" if trace.get("manifold_used") else "not_used",
        topology_summary_file)
    set_step(
        trace, "MorseCriticalPoints", bool(trace.get("morse_used")),
        "Morse critical point detector", "valid minima/saddles in node_sequence",
        "used" if trace.get("morse_used") else "not_used",
        morse_file)
    set_step(
        trace, "TopologyGraph", bool(trace.get("topology_graph_used")),
        "safe Morse critical points", "candidate graph paths",
        "used" if trace.get("topology_graph_used") else "not_used",
        topology_png)
    set_step(
        trace, "CandidateCorridors", bool(trace.get("candidate_corridor_used")),
        "Morse topology graph", "candidate_corridors.json",
        "used" if trace.get("candidate_corridor_used") else "not_used",
        candidate_corridors_file)
    set_step(
        trace, "CandidateRanking", bool(trace.get("candidate_ranking_used")),
        "eligible candidates final_total_score", "selected_corridor_id",
        "rank_{}".format(trace.get("selected_rank", 0))
        if trace.get("candidate_ranking_used") else "not_used",
        candidate_ranking_file)
    set_step(
        trace, "SelectedCorridor", bool(trace.get("selected_corridor_id")),
        "CandidateRanking", "Refinement",
        str(trace.get("selected_corridor_id", "")) or "not_selected",
        selected_corridor_file)
    set_step(
        trace, "Refinement", bool(trace.get("refinement_used")),
        "selected_corridor.raw_waypoints", "mpc_reference_path.csv",
        "success" if trace.get("refinement_used") else
        str(trace.get("refinement_failed_reason") or "not_used"),
        selected_corridor_file)
    set_step(
        trace, "MPC", bool(trace.get("mpc_used")),
        "{}/mpc_reference_path.csv".format(run_prefix),
        "{}/mpc_executed_trajectory.csv".format(run_prefix),
        str(trace.get("mpc_feasibility_status") or "success")
        if trace.get("mpc_used") else "not_used",
        "{};{}/mpc_diagnostics.json;{}/mpc_cost_breakdown.csv;{}/mpc_rollout_log.csv;{}/mpc_executed_trajectory.csv".format(
            mpc_reference_file, run_prefix, run_prefix, run_prefix, run_prefix))
    trace["steps"]["MPC"]["mpc_feasibility_status"] = trace.get(
        "mpc_feasibility_status", "")
    trace["steps"]["MPC"]["mpc_total_cost"] = trace.get("mpc_total_cost", 0.0)
    trace["steps"]["MPC"]["mpc_affects_candidate_ranking"] = trace.get(
        "mpc_affects_candidate_ranking", 0)
    for key in (
            "risk_query_source", "risk_query_called",
            "risk_query_valid_count", "risk_sanity_status",
            "mpc_rollout_mode", "mpc_rollout_log_file",
            "mpc_rollout_solve_count", "mpc_rollout_horizon_rows",
            "mpc_executed_trajectory_count",
            "mpc_executed_trajectory_file",
            "mpc_risk_cost", "mpc_max_risk",
            "mpc_candidate_feasibility_used",
            "mpc_execution_cost_in_score"):
        trace["steps"]["MPC"][key] = trace.get(key, "")
    set_step(
        trace, "ADP", bool(trace.get("adp_used")),
        "critic features", trace.get("adp_role", ""),
        "ranking={} control={}".format(
            int(trace.get("adp_affects_candidate_ranking", 0)),
            int(trace.get("adp_affects_control", 0))),
        metrics_file)


def finalize_trace(trace):
    required = (
        "risk_field_used", "manifold_used", "morse_used",
        "topology_graph_used", "candidate_corridor_used",
        "candidate_ranking_used", "refinement_used", "mpc_used")
    selected_id = str(trace.get("selected_corridor_id", ""))
    fallback = int(_num(trace.get("fallback_used", 0), 0))
    topology_fallback = int(_num(trace.get("topology_fallback_used", 0), 0))
    refined_ok = (
        int(_num(trace.get("refined_waypoints_count", 0), 0)) > 0 and
        trace.get("mpc_reference_source") == "refined_waypoints")
    trace["module_chain_valid"] = int(
        all(int(trace.get(key, 0)) == 1 for key in required) and
        fallback == 0 and topology_fallback == 0 and refined_ok and
        selected_id not in ("", "planning_failed"))
    trace["selection_consistent"] = int(
        int(trace.get("selected_rank", 0)) == 1 or
        bool(str(trace.get("selection_override_reason", "")).strip()))
    trace["refinement_trace_valid"] = int(
        int(trace.get("refinement_used", 0)) == 1 and
        int(trace.get("refined_waypoints_count", 0)) >
        int(trace.get("raw_waypoints_count", 0)) and
        trace.get("mpc_reference_source") == "refined_waypoints")
    if fallback or topology_fallback:
        trace["selected_corridor_type"] = "fallback"
        trace["execution_status"] = (
            trace.get("execution_status") or "fallback_or_failed")
        trace["module_chain_valid"] = 0
    if selected_id == "planning_failed":
        trace["planning_failed"] = 1
        trace["execution_status"] = trace.get("execution_status") or "failed"
        trace["failure_stage"] = trace.get("failure_stage") or "planning"
        trace["module_chain_valid"] = 0
    if trace["module_chain_valid"]:
        trace["execution_status"] = trace.get("execution_status") or "success"
    elif not trace.get("execution_status"):
        trace["execution_status"] = "failed"
    if not trace.get("failure_stage"):
        trace["failure_stage"] = "none" if trace.get("execution_status") == "success" else "unknown"
    if not trace.get("stop_reason"):
        trace["stop_reason"] = "none" if trace.get("execution_status") == "success" else "unknown"
    trace["module_chain"] = [
        dict(trace.get("steps", {}).get(name, {"module_name": name}))
        for name in STEP_NAMES
    ]
    return trace


def flatten_trace(trace):
    finalize_trace(trace)
    keys = list(MODULE_FLAGS) + [
        "selected_corridor_id", "corridor_id", "execution_corridor_id",
        "selected_corridor_type", "topology_route_class",
        "task_semantic_class", "selected_rank",
        "selection_override_reason", "selected_total_score",
        "raw_waypoints_count", "refined_waypoints_count",
        "refinement_failed_reason", "mpc_reference_source",
        "mpc_tracking_cost", "mpc_social_cost", "mpc_input_waypoint_count",
        "mpc_feasibility_status", "mpc_total_cost",
        "mpc_affects_candidate_ranking", "mpc_candidate_feasibility_used",
        "mpc_execution_cost_in_score", "risk_query_source",
        "risk_query_called", "risk_query_valid_count",
        "risk_sanity_status", "mpc_rollout_mode",
        "mpc_rollout_log_file", "mpc_rollout_solve_count",
        "mpc_rollout_horizon_rows", "mpc_executed_trajectory_count",
        "mpc_executed_trajectory_file", "mpc_risk_cost", "mpc_max_risk",
        "adp_role", "adp_affects_candidate_ranking", "adp_affects_control",
        "final_path_source", "module_chain_valid",
        "selection_consistent", "refinement_trace_valid", "planning_failed",
        "execution_status", "failure_stage", "stop_reason", "success_goal",
        "topology_fallback_used",
    ]
    return {key: trace.get(key, "") for key in keys}


def write_trace(trace, path):
    finalize_trace(trace)
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    with open(path, "w") as f:
        json.dump(trace, f, indent=2, sort_keys=True)
    return path
