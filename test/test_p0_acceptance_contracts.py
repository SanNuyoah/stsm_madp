import os
import sys

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
SCRIPTS = os.path.join(ROOT, "scripts")
for path in (SRC, SCRIPTS):
    if path not in sys.path:
        sys.path.insert(0, path)


from ensure_result_diagnostics import runtime_validation_fields
from stsm_madp.manifold_constraint import (
    ManifoldConstraint,
    assert_manifold_mode_consistency,
)
from stsm_madp.manifold_constraint_evaluator import ManifoldConstraintEvaluator
from stsm_madp.mpc import run_mpc_tracking
from stsm_madp.mpc import build_mpc_constraint_inputs
from stsm_madp.mpc import audit_reference_safety
from stsm_madp.mpc import _phase_constraint_diagnostics_payload
from stsm_madp.safety_evaluator import SafetyEvaluator
from stsm_madp.topology import TopologicalCorridorPlanner


class ZeroField(object):
    anchors = []

    def phi_s(self, point, velocity=None):
        return 0.0

    def grad_phi_s(self, point):
        return np.zeros_like(np.asarray(point, float))


def test_topology_uses_runtime_manifold_mode_instead_of_hard_default():
    planner = TopologicalCorridorPlanner(
        ZeroField(), rho=1.0, bounds=[(0.0, 1.0), (0.0, 1.0)],
        topology_profile="wheelchair", manifold_constraint_mode="soft")

    assert planner.manifold_constraint_mode == "soft"


def test_wheelchair_manifold_constraint_does_not_inherit_arm_phase_clearance():
    constraint = ManifoldConstraint(
        minimum_clearance=0.10,
        risk_threshold=1.0,
        mode="soft",
        robot_type="wheelchair",
        phase="navigation")

    assert constraint["robot_type"] == "wheelchair"
    assert constraint["phase"] == "navigation"
    assert constraint["effective_minimum_clearance"] == 0.10
    assert "handover" not in constraint["phase_clearance_schedule"]
    assert "approach" not in constraint["phase_clearance_schedule"]
    assert "return" not in constraint["phase_clearance_schedule"]


def test_arm_handover_clearance_remains_phase_aware():
    constraint = ManifoldConstraint(
        minimum_clearance=0.08,
        risk_threshold=1.0,
        mode="soft",
        robot_type="arm",
        phase="handover")

    assert constraint["robot_type"] == "arm"
    assert constraint["phase"] == "handover"
    assert constraint["effective_minimum_clearance"] == 0.15
    assert "handover" in constraint["phase_clearance_schedule"]


def test_manifold_mode_consistency_rejects_conflicting_sources():
    assert assert_manifold_mode_consistency("soft", "soft", None) == "soft"

    try:
        assert_manifold_mode_consistency("soft", "hard")
    except ValueError as exc:
        assert "inconsistent manifold modes" in str(exc)
    else:
        raise AssertionError("conflicting manifold modes must fail closed")


def test_candidate_and_mpc_evaluator_use_effective_clearance():
    boundary = {"left": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]}
    constraint = ManifoldConstraint(
        boundary=boundary,
        minimum_clearance=0.08,
        risk_threshold=1.0,
        robot_type="arm",
        phase="handover",
        mode="hard")
    evaluator = ManifoldConstraintEvaluator(
        manifold_constraint=constraint,
        corridor_constraint={"centerline": [[0.0, 0.16, 0.0], [1.0, 0.16, 0.0]],
                             "radius": 1.0})

    status = evaluator.evaluate_trajectory(
        [[0.0, 0.14, 0.0], [1.0, 0.14, 0.0]])

    assert status["minimum_clearance"] == 0.15
    assert status["effective_minimum_clearance"] == 0.15
    assert status["nominal_minimum_clearance"] == 0.08
    assert status["manifold_violation_count"] == 2


def test_wheelchair_mpc_constraint_inputs_keep_navigation_clearance():
    topology_info, _corridor_info, manifold_info, _constraint_info = (
        build_mpc_constraint_inputs(
            corridor=None,
            manifold={"minimum_clearance": 0.15, "risk_threshold": 1.0},
            reference_path=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            safe_threshold=1.0,
            minimum_clearance=0.10,
            phase="navigation",
            robot_type="wheelchair",
            manifold_constraint_mode="soft"))
    constraint = manifold_info

    assert constraint["robot_type"] == "wheelchair"
    assert constraint["phase"] == "navigation"
    assert constraint["minimum_clearance"] == 0.10
    assert constraint["effective_minimum_clearance"] == 0.10
    assert constraint["phase_clearance_schedule"] == {"navigation": {}}
    assert topology_info["manifold_constraint"]["minimum_clearance"] == 0.10


def test_hard_manifold_violation_rejects_rollout_before_execution():
    boundary = {"left": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]}
    result = run_mpc_tracking(
        "wheelchair",
        [0.0, 0.05, 0.0],
        [[0.0, 0.05, 0.0], [1.0, 0.05, 0.0]],
        {"critical_point_sequence": [
            {"id": "cp0", "type": "saddle", "point": [0.0, 0.05, 0.0], "order": 1}
        ]},
        {"centerline": [[0.0, 0.05, 0.0], [1.0, 0.05, 0.0]], "radius": 1.0},
        {
            "boundary": boundary,
            "minimum_clearance": 0.10,
            "effective_minimum_clearance": 0.10,
            "risk_threshold": 1.0,
            "effective_risk_threshold": 1.0,
            "robot_type": "wheelchair",
            "phase": "navigation",
            "manifold_constraint_mode": "hard",
        },
        ZeroField(),
        {
            "manifold_constraint_mode": "hard",
            "mpc_manifold_constraint_mode": "hard",
            "strict_risk_query": False,
            "risk_threshold": 1.0,
            "minimum_clearance": 0.10,
        },
        horizon=2,
        dt=0.1,
        selected_corridor_id="hard_violation",
        risk_threshold=1.0,
        config={"rollout_mode": "single_window"})

    assert result["mpc_feasibility_status"] == "reference_manifold_infeasible"
    assert result["success"] is False
    assert result["reference_safety_audit"]["feasible"] is False


def test_soft_major_violation_is_not_safety_success():
    boundary = {"left": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]}
    fields = runtime_validation_fields(
        {
            "constraints": {
                "manifold_constraint_mode": "soft",
                "manifold_soft_tolerance": 0.005,
            },
            "manifold_violation_count": 1,
            "corridor_violation_count": 0,
            "major_violation_count": 1,
            "max_manifold_violation": 0.05,
            "executed_trajectory_count": 1,
            "mpc_used": True,
            "mpc_feasibility_status": "feasible_with_soft_violation",
        },
        metrics={"success_goal": True},
        selected={"candidate_source": "morse_topology"},
        ref_count=1)

    assert fields["safety_success"] is False
    assert fields["overall_success"] is False
    assert fields["warning_reason"] == "soft_manifold_violation_not_accepted"


def test_runtime_validation_uses_runtime_violation_counts():
    fields = runtime_validation_fields(
        {
            "constraints": {"manifold_constraint_mode": "hard"},
            "manifold_violation_count": 5,
            "corridor_violation_count": 2,
            "clearance_violation_count": 9,
            "mpc_used": True,
            "mpc_feasibility_status": "feasible",
        },
        metrics={"success_goal": True},
        selected={"candidate_source": "morse_topology"},
        ref_count=3)

    assert fields["manifold_violation_count"] == 5
    assert fields["corridor_violation_count"] == 2
    assert fields["violation_count"] == 7
    assert fields["hard_violation_count"] == 7
    assert fields["safety_success"] is False
    assert fields["overall_success"] is False


def test_same_evaluator_reference_and_execution():
    boundary = {"left": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]}
    evaluator = SafetyEvaluator(
        manifold_constraint={
            "boundary": boundary,
            "minimum_clearance": 0.10,
            "effective_minimum_clearance": 0.10,
            "risk_threshold": 1.0,
        },
        corridor_constraint={"centerline": [[0.0, 0.2, 0.0], [1.0, 0.2, 0.0]],
                             "radius": 1.0})
    reference = evaluator.evaluate_state_or_points(
        state=[0.5, 0.12, 0.0], interest_points=[[0.5, 0.12, 0.0]],
        robot_type="wheelchair", task_phase="navigation",
        effective_minimum_clearance=0.10)
    execution = evaluator.evaluate_state_or_points(
        state=[0.5, 0.12, 0.0], interest_points=[[0.5, 0.12, 0.0]],
        robot_type="wheelchair", task_phase="navigation",
        effective_minimum_clearance=0.10)

    assert reference["min_clearance"] == execution["min_clearance"]
    assert reference["clearance_margin"] == execution["clearance_margin"]


def test_paper_mode_missing_interest_points_fails_closed():
    evaluator = SafetyEvaluator(
        manifold_constraint={"minimum_clearance": 0.10, "risk_threshold": 1.0},
        corridor_constraint={})

    status = evaluator.evaluate_state_or_points(
        state=[0.0, 0.0, 0.0], interest_points=[],
        robot_type="arm", task_phase="handover",
        effective_minimum_clearance=0.10, paper_mode=True)

    assert status["valid"] is False
    assert status["clearance_source"] == "invalid"
    assert status["interest_point_count"] == 0


def test_arm_phase_reference_audit_blocks_infeasible_reference():
    context = {
        "robot_type": "arm",
        "centerline": [[0.0, 0.05, 0.0], [1.0, 0.05, 0.0]],
        "radius": 1.0,
        "safe_threshold": 1.0,
        "manifold_boundary": {"left": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]},
        "manifold_constraint": {
            "robot_type": "arm",
            "phase_clearance_schedule": {
                "return": {"start_clearance": 0.20, "end_clearance": 0.20}
            },
        },
        "manifold_soft_tolerance": 0.005,
    }
    audit = audit_reference_safety(
        [[0.0, 0.05, 0.0], [1.0, 0.05, 0.0]], context,
        robot_type="arm", phase_sequence=["return", "return"])

    assert audit["feasible"] is False
    assert audit["violation_count"] == 2
    assert audit["worst_phase"] == "return"


def test_mpc_result_exports_success_contract_fields():
    boundary = {"left": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]}
    result = run_mpc_tracking(
        "wheelchair",
        [0.0, 0.4, 0.0],
        [[0.0, 0.4, 0.0], [0.2, 0.4, 0.0]],
        {"critical_point_sequence": [
            {"id": "cp0", "type": "saddle", "point": [0.0, 0.4, 0.0], "order": 1}
        ]},
        {"centerline": [[0.0, 0.4, 0.0], [0.2, 0.4, 0.0]], "radius": 1.0},
        {
            "boundary": boundary,
            "minimum_clearance": 0.10,
            "effective_minimum_clearance": 0.10,
            "risk_threshold": 1.0,
            "robot_type": "wheelchair",
            "phase": "navigation",
            "manifold_constraint_mode": "hard",
        },
        ZeroField(),
        {
            "manifold_constraint_mode": "hard",
            "strict_risk_query": False,
            "risk_threshold": 1.0,
            "minimum_clearance": 0.10,
        },
        horizon=2,
        dt=0.1,
        selected_corridor_id="success_contract",
        risk_threshold=1.0,
        config={"rollout_mode": "single_window"})

    assert result["task_success"] is True
    assert result["planner_success"] is True
    assert result["controller_success"] is True
    assert result["safety_success"] is True
    assert result["overall_success"] is True


def test_persistent_override_triggers_replan():
    fields = runtime_validation_fields(
        {
            "constraints": {"manifold_constraint_mode": "soft"},
            "manifold_violation_count": 0,
            "corridor_violation_count": 0,
            "manifold_override_count": 4,
            "consecutive_manifold_override_max": 4,
            "override_replan_limit": 4,
            "executed_trajectory_count": 4,
            "mpc_used": True,
            "mpc_feasibility_status": "feasible",
        },
        metrics={"success_goal": True},
        selected={"candidate_source": "morse_topology"},
        ref_count=4)

    assert fields["controller_success"] is False
    assert fields["overall_success"] is False


def test_reference_source_defaults_to_selected_candidate_when_not_refined():
    boundary = {"left": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]}
    result = run_mpc_tracking(
        "wheelchair",
        [0.0, 0.4, 0.0],
        [[0.0, 0.4, 0.0], [0.2, 0.4, 0.0]],
        {"critical_point_sequence": [
            {"id": "cp0", "type": "saddle", "point": [0.0, 0.4, 0.0], "order": 1}
        ]},
        {"centerline": [[0.0, 0.4, 0.0], [0.2, 0.4, 0.0]], "radius": 1.0},
        {
            "boundary": boundary,
            "minimum_clearance": 0.10,
            "effective_minimum_clearance": 0.10,
            "risk_threshold": 1.0,
            "robot_type": "wheelchair",
            "phase": "navigation",
            "manifold_constraint_mode": "hard",
        },
        ZeroField(),
        {
            "manifold_constraint_mode": "hard",
            "strict_risk_query": False,
            "risk_threshold": 1.0,
            "minimum_clearance": 0.10,
        },
        horizon=2,
        dt=0.1,
        selected_corridor_id="source_semantics",
        risk_threshold=1.0,
        config={"rollout_mode": "single_window"})

    assert result["reference_source"] == "selected_candidate_waypoints"


def test_phase_violation_sum_matches_validation():
    rows = [
        {"phase": "approach", "progress": 0.0, "manifold_violation": 0.01,
         "clearance_constraint_violation": 0.01,
         "manifold_constraint_status": "soft_violation"},
        {"phase": "return", "progress": 1.0, "manifold_violation": 0.0,
         "clearance_constraint_violation": 0.0,
         "manifold_constraint_status": "feasible"},
    ]
    payload = _phase_constraint_diagnostics_payload(
        {"robot_type": "arm", "manifold_violation_count": 1}, rows, rows)

    assert payload["executed_violation_total"] == 1
    assert payload["validation_manifold_violation_count"] == 1
    assert payload["consistency_check"] is True


def test_refinement_source_semantics():
    refinement_success = True
    refinement_changed_path = False
    reference_source = (
        "refined_waypoints"
        if refinement_success and refinement_changed_path
        else "selected_candidate_waypoints")

    assert reference_source == "selected_candidate_waypoints"


def test_candidate_recovery_stats_semantics():
    attempted = 0
    success_count = 1
    selected_candidate_source = "morse_topology"
    candidate_recovery_used = attempted > 0 and success_count > 0
    selected_candidate_recovery_used = (
        False if selected_candidate_source == "morse_topology"
        else candidate_recovery_used)

    assert candidate_recovery_used is False
    assert selected_candidate_recovery_used is False


def test_success_contract_requires_safety():
    fields = runtime_validation_fields(
        {
            "constraints": {"manifold_constraint_mode": "hard"},
            "manifold_violation_count": 1,
            "corridor_violation_count": 0,
            "mpc_used": True,
            "mpc_feasibility_status": "feasible",
        },
        metrics={"success_goal": True},
        selected={"candidate_source": "morse_topology"},
        ref_count=1)

    assert fields["task_success"] is True
    assert fields["planner_success"] is True
    assert fields["controller_success"] is True
    assert fields["safety_success"] is False
    assert fields["overall_success"] is False


def test_strict_risk_query_fails_closed_without_social_field():
    result = run_mpc_tracking(
        "wheelchair",
        [0.0, 0.0, 0.0],
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
        {},
        {"centerline": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], "radius": 1.0},
        {
            "boundary": [],
            "minimum_clearance": 0.0,
            "risk_threshold": 1.0,
            "manifold_constraint_mode": "hard",
        },
        None,
        {
            "manifold_constraint_mode": "hard",
            "mpc_manifold_constraint_mode": "hard",
            "strict_risk_query": True,
            "risk_threshold": 1.0,
        },
        horizon=2,
        dt=0.1,
        selected_corridor_id="test_corridor",
        risk_threshold=1.0,
        config={"rollout_mode": "single_window"})

    assert result["mpc_feasibility_status"] == "risk_query_invalid"
    assert result["failure_reason"] == "risk_query_invalid"
    assert result["success"] is False
