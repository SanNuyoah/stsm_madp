import os
import sys
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
SCRIPTS = os.path.join(ROOT, "scripts")
for path in (SRC, SCRIPTS):
    if path not in sys.path:
        sys.path.insert(0, path)


from ensure_result_diagnostics import runtime_validation_fields
from visualization.generate_results_figures import as_float
from stsm_madp.manifold_constraint import (
    ManifoldConstraint,
    assert_manifold_mode_consistency,
    distance_to_manifold_boundary,
)
from stsm_madp.manifold_constraint_evaluator import ManifoldConstraintEvaluator
from stsm_madp.mpc import ArmMPC, WheelchairMPC, run_mpc_tracking
from stsm_madp.mpc import build_mpc_constraint_inputs
from stsm_madp.mpc import audit_reference_safety
from stsm_madp.mpc import _phase_constraint_diagnostics_payload
from stsm_madp.mpc import evaluate_executed_trajectory
from stsm_madp.safety_evaluator import SafetyEvaluator
from stsm_madp.social_field import (
    HumanState, SemanticAnchor, SocialField, SocialFieldParams)
from stsm_madp.topology import TopologicalCorridorPlanner
from stsm_madp.topology_candidate_generator import (
    TopologyDrivenCandidateGenerator, candidate_topology_identity,
    rank_feasible_candidates)
from stsm_madp.topology_constraint import build_topology_constraint
from stsm_madp.corridor import (
    CorridorContractError, require_corridor_contract,
    validate_corridor_contract)


class ZeroField(object):
    anchors = []

    def phi_s(self, point, velocity=None):
        return 0.0

    def grad_phi_s(self, point):
        return np.zeros_like(np.asarray(point, float))


class CountingZeroField(ZeroField):
    def __init__(self):
        self.phi_calls = 0

    def phi_s(self, point, velocity=None):
        self.phi_calls += 1
        return 0.0


def test_manifold_boundary_distance_handles_segments_and_degenerate_points():
    boundary = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
    ]

    assert np.isclose(
        distance_to_manifold_boundary([1.0, 0.5, 0.0], boundary), 0.5)
    assert np.isclose(
        distance_to_manifold_boundary([2.5, 1.0, 0.0], boundary), 0.5)


def test_manifold_boundary_distance_preserves_named_boundary_minimum():
    boundary = {
        "left": [[0.0, 0.0], [0.0, 2.0]],
        "right": [[3.0, 0.0], [3.0, 2.0]],
    }

    assert np.isclose(
        distance_to_manifold_boundary([2.75, 1.0], boundary), 0.25)


def test_headless_experiments_do_not_run_social_field_visualizer():
    for launch_name in ("arm_view.launch", "wheelchair_view.launch"):
        root = ElementTree.parse(
            os.path.join(ROOT, "launch", launch_name)).getroot()
        visualizers = [
            node for node in root.findall("node")
            if node.get("type") == "social_field_viz_node.py"
        ]
        assert len(visualizers) == 1
        assert visualizers[0].get("if") == "$(arg rviz)"


def test_social_field_batch_matches_scalar_risk_exactly():
    human = HumanState(
        pos=[-1.6, 0.2, 0.0], vel=[0.03, -0.02, 0.0],
        heading=np.pi / 2.0, posture="transferring", vulnerability=1.4)
    anchors = [
        SemanticAnchor(
            "bed", [-1.6, -1.0, 0.0], [0.5, 1.0, 0.5],
            weight=2.0, forbidden=True),
        SemanticAnchor(
            "table", [0.55, 0.0, 0.0], [0.3, 0.5, 0.4],
            weight=1.0, forbidden=False),
    ]
    field = SocialField(SocialFieldParams(
        lam_prox=1.2, lam_close=1.0, lam_dir=0.5,
        lam_body=0.0, lam_env=1.5, sigma_env=0.4))
    field.set_scene([human], anchors)
    points = np.array([
        [1.8, 1.3, 0.0],
        [-1.6, -1.0, 0.0],
        [0.3, -0.2, 0.0],
    ])
    velocities = np.array([
        [0.0, 0.0, 0.0],
        [0.1, -0.1, 0.0],
        [-0.2, 0.05, 0.0],
    ])

    scalar = np.array([
        field.phi_s(point, velocity)
        for point, velocity in zip(points, velocities)
    ])
    batched = field.phi_s_batch(points, velocities)

    assert np.allclose(batched, scalar, rtol=1e-12, atol=1e-12)


def test_topology_uses_runtime_manifold_mode_instead_of_hard_default():
    planner = TopologicalCorridorPlanner(
        ZeroField(), rho=1.0, bounds=[(0.0, 1.0), (0.0, 1.0)],
        topology_profile="wheelchair", manifold_constraint_mode="soft")

    assert planner.manifold_constraint_mode == "soft"


def test_topology_route_search_budget_is_configurable_and_bounded():
    planner = TopologicalCorridorPlanner(
        ZeroField(), rho=1.0, bounds=[(0.0, 1.0), (0.0, 1.0)],
        topology_profile="wheelchair", route_max_paths=17,
        route_max_routes=9)

    assert planner.route_max_paths == 17
    assert planner.route_max_routes == 9

    default_planner = TopologicalCorridorPlanner(
        ZeroField(), rho=1.0, bounds=[(0.0, 1.0), (0.0, 1.0)],
        topology_profile="wheelchair")
    assert default_planner.route_max_paths == 512
    assert default_planner.route_max_routes == 256


def test_morse_route_budget_is_filled_by_global_lowest_cost_paths():
    generator = TopologyDrivenCandidateGenerator(max_paths=2, max_routes=2)
    edges = {
        "start": [
            {"to": "deep", "cost": 1.0},
            {"to": "goal", "cost": 2.0},
        ],
        "deep": [{"to": "goal", "cost": 100.0}],
    }

    routes = list(generator._enumerate_morse_paths(edges))

    assert [row[0] for row in routes] == [2.0, 101.0]
    assert routes[0][1] == ["start", "goal"]


def test_plot_metric_conversion_rejects_non_finite_values():
    assert as_float("inf") is None
    assert as_float("-inf") is None
    assert as_float("nan") is None
    assert as_float("1.25") == 1.25


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
    assert constraint["boundary"] == []


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

    missing_execution = runtime_validation_fields(
        {
            "constraints": {"manifold_constraint_mode": "soft"},
            "executed_evidence_required": True,
            "actual_executed_trajectory_count": 0,
            "manifold_violation_count": 0,
            "corridor_violation_count": 0,
            "mpc_used": True,
            "mpc_feasibility_status": "feasible",
        },
        metrics={"success_goal": True},
        selected={"candidate_source": "morse_topology"},
        ref_count=1)
    assert missing_execution["safety_success"] is False
    assert missing_execution["executed_evidence_complete"] is False


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

    derived = _phase_constraint_diagnostics_payload(
        {"robot_type": "arm", "executed_manifold_violation_count": 1},
        [{
            "trajectory_source": "executed",
            "phase": "return",
            "actual_clearance": 0.10,
            "clearance_threshold": 0.30,
            "manifold_constraint_status": "soft_violation",
        }], [])
    assert derived["executed_violation_total"] == 1
    assert derived["records"][0]["trajectory_source"] == "executed"


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


def test_metrics_does_not_read_output_row_before_it_is_built():
    """Shutdown aggregation may only use loaded diagnostics before ``base`` exists."""
    with open(os.path.join(ROOT, "nodes", "metrics_node.py"), "r") as handle:
        source = handle.read()
    base_assignment = source.index("        base = {")
    assert 'base.get("safety_success"' not in source[:base_assignment]


def test_execution_tube_is_not_redefined_by_mpc_reference():
    corridor = SimpleNamespace(
        corridor_id="arm_test",
        radius=0.08,
        execution_tube_centerline=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        refined_waypoints=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        centerline=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        node_sequence=[])
    reference = [[0.0, 0.5, 0.0], [1.0, 0.5, 0.0]]

    constraint = build_topology_constraint(
        selected_corridor=corridor, refined_reference=reference,
        safe_threshold=1.0, minimum_clearance=0.0)

    assert np.allclose(
        constraint["corridor_centerline"],
        corridor.execution_tube_centerline)
    evaluator = SafetyEvaluator(
        manifold_constraint={"minimum_clearance": 0.0, "risk_threshold": 1.0},
        corridor_constraint={"centerline": constraint["corridor_centerline"],
                             "radius": constraint["corridor_radius"]})
    assert evaluator.evaluate_trajectory(reference)["corridor_violation_count"] == 2


def test_executed_corridor_counts_only_active_scope():
    context = {
        "centerline": np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "radius": 0.08,
        "safe_threshold": 10.0,
        "minimum_clearance": 0.0,
        "nominal_minimum_clearance": 0.0,
        "manifold_boundary": [],
        "manifold_constraint": {"minimum_clearance": 0.0,
                                "risk_threshold": 10.0},
        "topology_tube_constraint": {},
        "robot_type": "arm",
    }
    points = [[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [1.0, 0.5, 0.0]]
    rows, summary = evaluate_executed_trajectory(
        points, context, social_field=ZeroField(), robot_type="arm",
        phase_sequence=["approach", "approach", "return"],
        corridor_active_sequence=[False, True, False])

    assert summary["corridor_violation_count"] == 0
    assert rows[0]["corridor_constraint_status"] == "not_applicable"
    assert rows[0]["corridor_out_of_scope_violation"] > 0.0
    assert rows[1]["corridor_active"] is True


def _valid_morse_corridor(corridor_id="morse_c001", radius=0.2):
    return SimpleNamespace(
        corridor_id=corridor_id,
        label=corridor_id,
        topology_class="saddle_channel",
        topology_route_class="saddle_channel",
        node_sequence=["start", "saddle_0", "goal"],
        topology_nodes=["start", "saddle_0", "goal"],
        morse_node_ids=["saddle_0"],
        morse_induced=True,
        candidate_source="morse_topology",
        waypoints=np.asarray([[0.0, 0.0, 0.0],
                              [0.5, 0.2, 0.0],
                              [1.0, 0.0, 0.0]]),
        radius=radius,
        boundary={},
    )


def test_stsm_corridor_contract_rejects_missing_corridor():
    status = validate_corridor_contract(
        None, require_morse=True, require_tube=True)
    assert status["valid"] is False
    assert "corridor_id_missing" in status["failure_reason"]


def test_stsm_corridor_contract_rejects_wrong_corridor_id():
    corridor = _valid_morse_corridor()
    try:
        require_corridor_contract(
            corridor, expected_corridor_id="morse_c999",
            require_morse=True, require_tube=True)
    except CorridorContractError as exc:
        assert "corridor_id_mismatch" in str(exc)
    else:
        raise AssertionError("mismatched corridor id must fail closed")


def test_stsm_corridor_contract_rejects_missing_tube():
    corridor = _valid_morse_corridor(radius=0.0)
    status = validate_corridor_contract(
        corridor, require_morse=True, require_tube=True)
    assert status["valid"] is False
    assert "trajectory_tube_missing" in status["failure_reason"]


def test_strict_mpc_inputs_preserve_corridor_identity():
    corridor = _valid_morse_corridor()
    topology, corridor_info, _manifold, constraint = build_mpc_constraint_inputs(
        corridor=corridor,
        reference_path=corridor.waypoints,
        safe_threshold=1.0,
        minimum_clearance=0.0,
        manifold_constraint_mode="hard",
        strict_stsm=True,
        expected_corridor_id=corridor.corridor_id)
    assert corridor_info["corridor_id"] == corridor.corridor_id
    assert constraint["topology_tube_constraint"]["corridor_id"] == corridor.corridor_id
    assert topology["topology_tube_constraint"]["corridor_id"] == corridor.corridor_id


def test_mpc_inputs_preserve_candidate_execution_ranking_evidence():
    corridor = _valid_morse_corridor()
    corridor.hard_feasible = True
    corridor.execution_feasible = True
    corridor.execution_cost = 0.42
    corridor.candidate_cost_breakdown = {
        "execution_cost": 0.42,
        "execution_cost_term": 0.42,
        "mpc_execution_cost_in_score": True,
    }

    topology, corridor_info, _manifold, _constraint = build_mpc_constraint_inputs(
        corridor=corridor,
        reference_path=corridor.waypoints,
        safe_threshold=1.0,
        minimum_clearance=0.0,
        manifold_constraint_mode="soft",
        strict_stsm=True,
        expected_corridor_id=corridor.corridor_id)

    assert corridor_info["execution_feasible"] is True
    assert corridor_info["candidate_cost_breakdown"]["execution_cost_term"] == 0.42
    assert topology["candidate_cost_breakdown"]["mpc_execution_cost_in_score"] is True


def test_mpc_diagnostics_report_candidate_execution_in_ranking():
    result = run_mpc_tracking(
        "wheelchair",
        [0.0, 0.0, 0.0],
        [[0.0, 0.0, 0.0], [0.15, 0.0, 0.0], [0.30, 0.0, 0.0]],
        {"topology_class": "direct_safe_channel"},
        {
            "corridor_id": "wheelchair_c0001",
            "centerline": [[0.0, 0.0, 0.0], [0.30, 0.0, 0.0]],
            "radius": 0.4,
            "hard_feasible": True,
            "execution_feasible": True,
            "execution_cost": 0.2,
            "candidate_cost_breakdown": {
                "execution_cost": 0.2,
                "execution_cost_term": 0.2,
                "mpc_execution_cost_in_score": True,
            },
        },
        {"safe_threshold": 1.0, "minimum_clearance": 0.0},
        ZeroField(),
        {"risk_threshold": 1.0, "manifold_constraint_mode": "soft"},
        horizon=2,
        dt=0.1,
        selected_corridor_id="wheelchair_c0001",
        risk_threshold=1.0,
        rollout_mode="single")

    assert result["mpc_candidate_feasibility_used"] == 1
    assert result["mpc_execution_cost_in_score"] == 1
    assert result["mpc_affects_candidate_ranking"] == 1


def test_strict_mpc_inputs_reject_reference_without_corridor():
    try:
        build_mpc_constraint_inputs(
            corridor=None,
            reference_path=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            strict_stsm=True)
    except CorridorContractError as exc:
        assert "corridor_id_missing" in str(exc)
    else:
        raise AssertionError("STSM must not synthesize a tube from reference only")


def _decision_candidate(candidate_id, score, feasible=True, recovered=False):
    return SimpleNamespace(
        candidate_id=candidate_id,
        topology_class="saddle_channel",
        critical_point_sequence=["start", "saddle_0", "goal"],
        centerline=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        topology_valid=True,
        manifold_feasible=bool(feasible),
        candidate_tube_valid=bool(feasible),
        risk_valid=bool(feasible),
        execution_feasible=True,
        candidate_recovered=bool(recovered),
        recovery_cost=0.2 if recovered else 0.0,
        total_score=float(score),
        topology_value=1.0,
        path_length=1.0)


def test_hard_infeasible_candidate_is_never_ranked_even_with_low_score():
    unsafe = _decision_candidate("unsafe", -100.0, feasible=False)
    safe = _decision_candidate("safe", 10.0, feasible=True)

    ranked, records = rank_feasible_candidates([unsafe, safe])

    assert [item.candidate_id for item in ranked] == ["safe"]
    unsafe_record = next(row for row in records if row["candidate_id"] == "unsafe")
    assert unsafe_record["hard_feasible"] is False
    assert unsafe_record["ranking_eligible"] is False
    assert "manifold_infeasible" in unsafe_record["decision_reason"]


def test_recovery_preserves_topology_identity_and_not_topology_diversity():
    original = _decision_candidate("original", 2.0)
    recovered = _decision_candidate("recovered", 2.2, recovered=True)
    recovered.original_topology_identity = candidate_topology_identity(original)

    ranked, records = rank_feasible_candidates([recovered, original])

    assert candidate_topology_identity(ranked[0]) == candidate_topology_identity(ranked[1])
    assert len(set(row["original_topology_identity"] for row in records)) == 1


def test_candidate_ranking_is_deterministic_under_input_permutation():
    first = _decision_candidate("b", 1.0)
    second = _decision_candidate("a", 1.0)

    left, _ = rank_feasible_candidates([first, second])
    right, _ = rank_feasible_candidates([second, first])

    assert [item.candidate_id for item in left] == ["a", "b"]
    assert [item.candidate_id for item in right] == ["a", "b"]


def test_arm_predictive_sequence_must_reduce_terminal_task_error():
    mpc = ArmMPC(
        n_joints=3, dq_max=1.0, v_cap=1.0, horizon=3,
        beam_width=12, ddq_max=10.0,
        joint_lower=[-2.0] * 3, joint_upper=[2.0] * 3)

    dq = mpc.solve(
        np.eye(3), [1.0, 0.0, 0.0], dq_nom=np.zeros(3),
        q=np.zeros(3), ee_pos=np.zeros(3), target_pos=[1.0, 0.0, 0.0],
        dt=0.1, predictive=True,
        interest_constraints={"enabled": False})

    assert dq[0] > 0.0
    initial = mpc.last_objective_terms["initial_target_error"]
    terminal = mpc.last_objective_terms["terminal_target_error"]
    required = mpc.last_objective_terms["required_terminal_progress"]
    first_error = mpc.last_objective_terms["first_step_target_error"]
    first_required = mpc.last_objective_terms["required_first_step_progress"]
    assert required > 0.0
    assert first_required > 0.0
    assert first_error <= initial + mpc.task_progress_tolerance + 1e-9
    assert terminal <= initial - required + 1e-9
    assert mpc.last_constraint_violation["task_progress"] >= 0
    assert mpc.last_constraint_violation["first_step_task_progress"] >= 0


def test_arm_predictive_sequence_fails_closed_without_task_progress():
    mpc = ArmMPC(
        n_joints=3, dq_max=1.0, v_cap=1.0, horizon=3,
        beam_width=12, ddq_max=10.0,
        joint_lower=[-2.0] * 3, joint_upper=[2.0] * 3)

    dq = mpc.solve(
        np.zeros((3, 3)), [1.0, 0.0, 0.0], dq_nom=np.zeros(3),
        q=np.zeros(3), ee_pos=np.zeros(3), target_pos=[1.0, 0.0, 0.0],
        dt=0.1, predictive=True,
        interest_constraints={"enabled": False})

    assert np.allclose(dq, 0.0)
    assert mpc.last_solver_status == "safe_stop: no_feasible_joint_sequence"
    assert mpc.solve_success_count == 0
    assert mpc.last_constraint_violation["task_progress"] > 0


def test_arm_progress_gate_uses_active_waypoint_tolerance():
    mpc = ArmMPC(
        n_joints=3, dq_max=1.0, v_cap=1.0, horizon=3,
        beam_width=12, ddq_max=10.0,
        joint_lower=[-2.0] * 3, joint_upper=[2.0] * 3)

    dq = mpc.solve(
        np.eye(3), [0.041, 0.0, 0.0], dq_nom=np.zeros(3),
        q=np.zeros(3), ee_pos=np.zeros(3), target_pos=[0.041, 0.0, 0.0],
        dt=0.1, predictive=True,
        interest_constraints={
            "enabled": False,
            "task_progress_tolerance": 0.04,
        })

    assert dq[0] > 0.0
    assert mpc.last_objective_terms["task_progress_tolerance"] == 0.04
    assert mpc.last_objective_terms["required_terminal_progress"] < 1e-4


def test_wheelchair_heading_recovery_applies_forward_creep_now():
    mpc = WheelchairMPC(horizon=12, dt=0.1, beam_width=16)

    v, w = mpc.solve(
        [0.0, 0.0, np.pi],
        [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
        ZeroField(), u_prev=[0.0, 0.0], goal=[1.0, 0.0], predictive=True)

    assert abs(w) > 0.0
    assert v >= 0.02
    assert mpc.last_predicted_controls[0][0] >= 0.02
    assert mpc.last_alignment_translation > 0.0


def test_wheelchair_progress_gate_does_not_require_max_acceleration_step():
    mpc = WheelchairMPC(horizon=3, dt=0.2, a_max=0.5, beam_width=4)
    mpc._sequence_step_controls = lambda previous, warm_u=None, goal_u=None: [
        np.array([0.05, 0.0], float)]

    v, w = mpc.solve(
        [0.0, 0.0, 0.0],
        [[0.1, 0.0], [0.2, 0.0], [0.3, 0.0]],
        ZeroField(), goal=[0.3, 0.0], predictive=True)

    assert v == 0.05
    assert w == 0.0
    assert mpc.last_solver_status.startswith("predictive_beam")
    assert mpc.last_objective_terms["required_first_speed"] == 0.02


def test_wheelchair_beam_keeps_executable_first_step_before_pruning():
    mpc = WheelchairMPC(horizon=12, dt=0.2, a_max=0.5, beam_width=12)
    ref = np.column_stack([
        np.linspace(1.813, 1.647, 12),
        np.full(12, 1.555),
    ])

    v, _w = mpc.solve(
        [1.824, 1.368, -2.49], ref, ZeroField(),
        u_prev=[0.0, 0.0], goal=[-0.55, 0.55], predictive=True)

    assert v >= 0.02
    assert mpc.last_predicted_controls[0][0] >= 0.02
    assert mpc.last_sequence_progress >= mpc.min_progress_per_solve


def test_wheelchair_beam_reuses_identical_state_safety_evaluations():
    field = CountingZeroField()
    mpc = WheelchairMPC(horizon=12, dt=0.2, a_max=0.5, beam_width=12)
    ref = np.column_stack([
        np.linspace(1.813, 1.647, 12),
        np.full(12, 1.555),
    ])

    v, _w = mpc.solve(
        [1.824, 1.368, -2.49], ref, field,
        u_prev=[0.0, 0.0], goal=[-0.55, 0.55], predictive=True)

    assert v >= 0.02
    assert field.phi_calls < 500


def test_runtime_sources_preserve_p0_execution_contracts():
    with open(os.path.join(ROOT, "nodes", "wheelchair_node.py"), "r") as handle:
        wheelchair_source = handle.read()
    with open(os.path.join(ROOT, "nodes", "metrics_node.py"), "r") as handle:
        metrics_source = handle.read()
    with open(os.path.join(ROOT, "scripts", "run_experiments.sh"), "r") as handle:
        experiment_source = handle.read()

    assert "executable_candidate_available" not in wheelchair_source
    assert "copy.deepcopy(\n            getattr(self.manifold" in wheelchair_source
    assert "copy.deepcopy(\n                    self.last_valid_topology_debug)" in wheelchair_source
    assert "rospy.Timer" in wheelchair_source
    assert "_command_keepalive_cb" in wheelchair_source
    assert "runtime_blocking_replan_enabled" in wheelchair_source
    assert "skip runtime blocking replan reason=no_progress" in wheelchair_source
    assert '"/stsm/wc_task_complete"' in wheelchair_source
    assert '"/stsm/wc_task_complete"' in metrics_source
    assert "metrics_goal_tolerance=\"${wc_completion_tolerance}\"" in experiment_source
    assert "def first_value(*values):" in experiment_source


def test_baseline_failure_stage_is_execution_not_refinement():
    with open(os.path.join(ROOT, "nodes", "metrics_node.py"), "r") as handle:
        source = handle.read()
    assert 'if variant_name == "baseline":\n                failure_stage = "execution"' in source
