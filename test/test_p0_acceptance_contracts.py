import json
import os
import sys
import tempfile
import importlib.util
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
SCRIPTS = os.path.join(ROOT, "scripts")
for path in (SRC, SCRIPTS):
    if path not in sys.path:
        sys.path.insert(0, path)


from ensure_result_diagnostics import runtime_validation_fields, selected_candidate
from visualization.generate_results_figures import as_float
from stsm_madp.manifold_constraint import (
    ManifoldConstraint,
    assert_manifold_mode_consistency,
    distance_to_manifold_boundary,
)
from stsm_madp.manifold_constraint_evaluator import ManifoldConstraintEvaluator
from stsm_madp.manifold import SafetyManifold
from stsm_madp.mpc import ArmMPC, WheelchairMPC, run_mpc_tracking
from stsm_madp.mpc import build_mpc_constraint_inputs
from stsm_madp.mpc import audit_reference_safety
from stsm_madp.mpc import _phase_constraint_diagnostics_payload
from stsm_madp.mpc import _task_state_diagnostics_payload
from stsm_madp.mpc import evaluate_executed_trajectory
from stsm_madp.mpc import (
    wheelchair_nonholonomic_execution_profile, wheelchair_sharp_turn_audit)
from stsm_madp.safety_evaluator import (
    SafetyEvaluator, build_safety_context, terminal_acceptance_preflight)
from stsm_madp.interest_points import pose_interest_risk, pose_interest_risk_batch
from stsm_madp.social_field import (
    HumanState, SemanticAnchor, SocialField, SocialFieldParams)
from stsm_madp.adp import (
    ADPCritic, ADPTransitionLearner, adp_role_from_runtime,
    adp_ranking_adjustments, clone_critic,
    candidate_feature_values, fit_critic_from_transition_records,
    recenter_critic_feature_normalization, require_feature_schema,
    save_and_verify_critic, critic_theta_hash, evaluate_promotion_gate,
    validate_critic_runtime_identity)
from stsm_madp.decision_trace import trace_from_debug
from stsm_madp.task_semantics import infer_task_context
from stsm_madp.topology import TopologicalCorridorPlanner
from stsm_madp.topology_refinement import _limit_refinement_points
from stsm_madp.topology_refinement import refine_topology_path
from stsm_madp.topology_candidate_generator import (
    TopologyDrivenCandidateGenerator, candidate_topology_identity,
    rank_feasible_candidates)
from stsm_madp.topology_diagnostics_writer import (
    _candidate_ranking_rows, write_failed_topology_diagnostics)
from stsm_madp.topology_constraint import build_topology_constraint
from stsm_madp.corridor import (
    Corridor, CorridorContractError, require_corridor_contract,
    validate_corridor_contract)


class ZeroField(object):
    anchors = []

    def phi_s(self, point, velocity=None):
        return 0.0

    def grad_phi_s(self, point):
        return np.zeros_like(np.asarray(point, float))


class BatchRiskField(object):
    anchors = []

    def phi_s(self, point, velocity=None):
        point = np.asarray(point, float)
        return float(point[0] ** 2 + 0.5 * point[1] ** 2)

    def phi_s_batch(self, points, velocities=None):
        points = np.asarray(points, float)
        return points[:, 0] ** 2 + 0.5 * points[:, 1] ** 2


def test_strict_safety_evaluator_fails_closed_without_social_context():
    evaluator = SafetyEvaluator(
        manifold_constraint={"minimum_clearance": 0.10,
                             "risk_threshold": 2.0})
    status = evaluator.evaluate_trajectory(
        [[0.0, 0.0, 0.0]], require_social_context=True)

    assert not status["valid"]
    assert status["failure_reason"] == "missing_safety_context"


def test_strict_refinement_uses_explicit_context_and_rejects_unsafe_point():
    field = SocialField(SocialFieldParams(lam_env=1.5, sigma_env=0.4))
    field.set_scene([], [SemanticAnchor(
        "forbidden", [0.0, 0.0, 0.0], [0.2, 0.2, 0.2],
        weight=2.0, forbidden=True)])
    constraint = {"boundary": [], "minimum_clearance": 0.10,
                  "risk_threshold": 2.0, "safe_threshold": 2.0}
    context = build_safety_context(field, constraint, strict=True)
    corridor = Corridor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]], radius=0.5)
    corridor.planning_safety_context_fingerprint = context["fingerprint"]

    ok, _path, metrics, reason = refine_topology_path(
        corridor, corridor_constraint={"centerline": corridor.waypoints,
                                       "radius": corridor.radius},
        manifold_constraint=constraint, max_refinement_points=48,
        safety_context=context, require_social_context=True)

    assert not ok
    assert reason in ("clearance_violation", "risk_violation",
                      "refined_manifold_violation")
    assert (metrics["planning_safety_context_fingerprint"] ==
            metrics["refinement_safety_context_fingerprint"])


def test_batched_wheelchair_footprint_risk_matches_per_pose_evaluation():
    field = BatchRiskField()
    poses = np.asarray([
        [0.1, -0.2, 0.0],
        [0.3, 0.4, np.pi / 3.0],
    ])
    batched = pose_interest_risk_batch(field, poses)
    scalar = [pose_interest_risk(field, pose) for pose in poses]

    assert len(batched) == len(scalar)
    for batch_summary, scalar_summary in zip(batched, scalar):
        assert batch_summary["labels"] == scalar_summary["labels"]
        assert np.allclose(batch_summary["phi_each"], scalar_summary["phi_each"])
        assert batch_summary["phi_max"] == scalar_summary["phi_max"]
        assert batch_summary["phi_mean"] == scalar_summary["phi_mean"]


class CountingZeroField(ZeroField):
    def __init__(self):
        self.phi_calls = 0

    def phi_s(self, point, velocity=None):
        self.phi_calls += 1
        return 0.0


def test_adp_ranking_adjustment_is_bounded_and_uses_calibration_scale():
    terms, meta = adp_ranking_adjustments(
        [20.0, 22.0, 50.0],
        metadata={"target_mean": 22.0, "target_p95": 26.0},
        lambda_adp=0.02, normalization="robust", norm_clip=10.0,
        contribution_clip=0.10)
    assert meta["source"] == "metadata_target_mean_p95"
    assert terms[0]["adp_cost"] < 0.0
    assert terms[-1]["adp_cost"] == 0.10
    assert all(abs(item["adp_cost"]) <= 0.10 for item in terms)


def test_adp_ranking_lambda_zero_preserves_zero_contribution():
    terms, _meta = adp_ranking_adjustments(
        [1.0, 4.0], metadata={}, lambda_adp=0.0)
    assert [item["adp_cost"] for item in terms] == [0.0, 0.0]


def test_adp_ranking_can_softly_reorder_two_legal_candidates():
    terms, _meta = adp_ranking_adjustments(
        [26.0, 18.0],
        metadata={"target_mean": 22.0, "target_p95": 26.0},
        lambda_adp=0.02)
    candidates = []
    for candidate_id, base_total_cost, term in (
            ("candidate_a", 1.00, terms[0]),
            ("candidate_b", 1.03, terms[1])):
        candidate = {
            "candidate_id": candidate_id,
            "base_total_cost": base_total_cost,
            "candidate_status": "safe",
            "hard_violation": False,
        }
        candidate.update(term)
        candidates.append(candidate)
    for candidate in candidates:
        candidate["total_cost_with_adp"] = (
            candidate["base_total_cost"] + candidate["adp_cost"])

    ranked = sorted(candidates, key=lambda item: item["total_cost_with_adp"])
    assert ranked[0]["candidate_id"] == "candidate_b"
    assert ranked[0]["adp_cost"] < ranked[1]["adp_cost"]


def test_final_candidate_ranking_keeps_adp_fields_when_writer_enriches_rows():
    final_rows = [
        {"candidate_id": "arm_c0001", "candidate_status": "safe",
         "candidate_filter_class": "safe", "base_total_cost": 1.00,
         "adp_value_raw": 26.0, "adp_value_normalized": 1.0,
         "effective_lambda_adp": 0.02, "adp_cost": 0.02,
         "total_cost_with_adp": 1.02, "total_cost": 1.02,
         "rank_before_adp": 1, "rank_after_adp": 1,
         "adp_changed_rank": False,
         "ranking_theta_source": "run_start_snapshot", "selected": True},
        {"candidate_id": "arm_c0002", "candidate_status": "safe",
         "candidate_filter_class": "safe", "base_total_cost": 1.03,
         "adp_value_raw": 18.0, "adp_value_normalized": -1.0,
         "effective_lambda_adp": 0.02, "adp_cost": -0.02,
         "total_cost_with_adp": 1.01, "total_cost": 1.01,
         "rank_before_adp": 2, "rank_after_adp": 2,
         "adp_changed_rank": False,
         "ranking_theta_source": "run_start_snapshot", "selected": False},
    ]
    rows = _candidate_ranking_rows(
        {"final_candidate_ranking": final_rows,
         "selected_corridor_id": "arm_c0001"}, {}, [])

    assert [row["candidate_id"] for row in rows] == ["arm_c0001", "arm_c0002"]
    for row in rows:
        assert row["ranking_record_stage"] == "final_legal"
        assert row["ranking_theta_source"] == "run_start_snapshot"
        assert "adp_value_raw" in row
        assert "adp_cost" in row
        assert "total_cost_with_adp" in row


def test_adp_ranking_snapshot_does_not_change_when_live_critic_learns():
    live = ADPCritic(theta=np.ones(22), metadata={"target_mean": 2.0,
                                                    "target_p95": 3.0})
    snapshot = clone_critic(live)
    live.theta[0] = 99.0
    assert snapshot.theta[0] == 1.0
    assert snapshot.fingerprint() != live.fingerprint()


def test_adp_ranking_role_has_no_control_contribution():
    assert adp_role_from_runtime(
        True, True, True, effective_lambda=0.02,
        ranking_contribution=True, control_contribution=False) == "ranking_modifier"


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


def test_wheelchair_task_context_prefers_events_over_progress_fallback():
    cfg = {"wheelchair": {"arriving_radius": 0.8,
                           "avoiding_risk_threshold": 1.6,
                           "progress_fallback_enabled": True}}
    assert infer_task_context(
        "wheelchair", progress=0.95,
        context={"dist_to_goal": 0.3}, config=cfg)["task_state"] == "arriving"
    avoiding = infer_task_context(
        "wheelchair", progress=0.9,
        context={"risk_ahead": 2.0, "dist_to_goal": 0.3}, config=cfg)
    assert avoiding["task_state"] == "avoiding"
    assert avoiding["state_trigger"] == "social_risk_ahead"
    obstacle = infer_task_context(
        "wheelchair", progress=0.1,
        context={"obstacle_ahead": True}, config=cfg)
    assert obstacle["task_state"] == "avoiding"
    assert obstacle["state_trigger"] == "obstacle_ahead"
    passing = infer_task_context(
        "wheelchair", progress=0.2,
        context={"near_narrow_passage": True}, config=cfg)
    assert passing["task_state"] == "passing"
    assert passing["state_trigger"] == "narrow_passage"
    fallback = infer_task_context("wheelchair", progress=0.55, config=cfg)
    assert fallback["task_state"] == "passing"
    assert fallback["state_trigger"] == "progress_fallback"


def _task_aware_test_field():
    field = SocialField(SocialFieldParams(
        lam_prox=1.2, lam_close=1.0, lam_dir=0.5, lam_body=0.0,
        lam_env=0.0, direction_model="continuous", task_aware_enabled=True))
    field.set_scene([HumanState(
        pos=[0.0, 0.0, 0.0], heading=0.0, posture="sitting")], [])
    return field


def test_task_context_changes_social_field_and_can_roll_back_to_base_weights():
    field = _task_aware_test_field()
    point = np.array([0.45, 0.10, 0.0])
    field.set_task_context({"task_state": "moving"})
    moving_risk = field.phi_s(point)
    moving_weights = field.get_effective_weights()
    field.set_task_context({"task_state": "passing"})
    passing_risk = field.phi_s(point)
    assert passing_risk != moving_risk
    assert field.get_effective_weights()["lam_prox"] > moving_weights["lam_prox"]

    field.params.task_aware_enabled = False
    field.set_task_context({"task_state": "avoiding"})
    assert field.get_effective_weights() == {
        "lam_prox": 1.2, "lam_close": 1.0, "lam_dir": 0.5,
        "lam_body": 0.0, "lam_env": 0.0}


def test_continuous_direction_is_finite_ordered_and_matches_batch():
    field = _task_aware_test_field()
    angles = np.deg2rad(np.array([0., 30., 60., 90., 120., 150., 180.]))
    points = np.column_stack((np.cos(angles), np.sin(angles),
                              np.zeros(len(angles))))
    values = np.array([field.phi_dir(point, field.humans[0]) for point in points])
    assert np.all(np.isfinite(values))
    assert values[0] > values[3] > values[-1]
    assert np.all(np.diff(values) < 0.0)
    assert np.allclose(field.phi_s_batch(points),
                       [field.phi_s(point) for point in points])
    field.params.direction_model = "legacy_piecewise"
    legacy = np.array([
        field.phi_dir(point, field.humans[0]) for point in points])
    assert np.allclose(legacy, [0.8, 0.8, 0.4, 0.4, 0.4, 1.5, 1.5])


def test_task_context_changes_safety_manifold_value_and_margin():
    field = _task_aware_test_field()
    manifold = SafetyManifold(field, rho=1.0, lam_s=1.0)
    point = np.array([0.45, 0.10, 0.0])
    goal = np.array([1.0, 0.0, 0.0])
    field.set_task_context({"task_state": "moving"})
    moving_phi = field.phi_s(point)
    moving_psi = manifold.psi(point, goal)
    moving_margin = manifold.rho - moving_phi
    field.set_task_context({"task_state": "avoiding"})
    avoiding_phi = field.phi_s(point)
    avoiding_psi = manifold.psi(point, goal)
    avoiding_margin = manifold.rho - avoiding_phi
    assert avoiding_phi > moving_phi
    assert avoiding_psi > moving_psi
    assert avoiding_margin < moving_margin


def test_runtime_task_state_diagnostics_keep_event_context_over_progress():
    runtime_records = [{
        "task_state": "moving",
        "state_trigger": "default_motion",
        "progress": 0.10,
        "dist_to_goal": 2.5,
        "risk_ahead": 0.2,
        "near_narrow_passage": False,
        "near_critical_point": False,
        "effective_social_weights": {"lam_prox": 1.2},
        "timestamp": 1.0,
    }, {
        "task_state": "avoiding",
        "state_trigger": "social_risk_ahead",
        "progress": 0.30,
        "dist_to_goal": 1.8,
        "risk_ahead": 1.9,
        "near_narrow_passage": False,
        "near_critical_point": False,
        "effective_social_weights": {"lam_prox": 1.5},
        "timestamp": 2.0,
    }]

    payload = _task_state_diagnostics_payload(
        {"robot_type": "wheelchair", "task_mode": "navigation"},
        [{"progress": 0.75}], [], runtime_records=runtime_records)

    assert payload["source"] == "runtime_task_context"
    assert payload["transitions"] == ["start->moving", "moving->avoiding"]
    assert payload["records"][-1]["state_trigger"] == "social_risk_ahead"
    assert payload["records"][-1]["risk_ahead"] == 1.9
    assert payload["records"][-1]["effective_social_weights"] == {
        "lam_prox": 1.5}


def test_arm_runtime_task_state_diagnostics_preserve_real_phases():
    runtime_records = []
    for index, state in enumerate(
            ("approach", "align", "handover", "hold", "return")):
        runtime_records.append({
            "task_state": state,
            "state_trigger": "explicit_phase",
            "progress": 0.0,
            "dist_to_goal": 1.0 - 0.1 * index,
            "risk_ahead": 0.2 + 0.1 * index,
            "obstacle_ahead": False,
            "near_narrow_passage": False,
            "near_critical_point": False,
            "effective_social_weights": {"lam_prox": 1.0},
            "phase": state,
            "timestamp": float(index),
        })

    payload = _task_state_diagnostics_payload(
        {"robot_type": "arm", "task_mode": "handover"}, [], [],
        runtime_records=runtime_records)

    assert payload["source"] == "runtime_task_context"
    assert [record["task_state"] for record in payload["records"]] == [
        "approach", "align", "handover", "hold", "return"]
    assert set(record["state_trigger"] for record in payload["records"]) == {
        "explicit_phase"}


def test_task_semantic_sensitivity_covers_all_wheelchair_states():
    field = _task_aware_test_field()
    manifold = SafetyManifold(field, rho=1.0, lam_s=1.0)
    point = np.array([0.45, 0.10, 0.0])
    goal = np.array([1.0, 0.0, 0.0])
    values = {}

    for state in ("moving", "avoiding", "passing", "arriving"):
        field.set_task_context({"task_state": state})
        phi_s = field.phi_s(point)
        values[state] = {
            "effective_social_weights": field.get_effective_weights(),
            "phi_s": phi_s,
            "psi": manifold.psi(point, goal),
            "safe_margin": manifold.rho - phi_s,
        }

    assert values["moving"]["phi_s"] != values["avoiding"]["phi_s"]
    assert values["moving"]["phi_s"] != values["passing"]["phi_s"]
    assert values["moving"]["psi"] != values["avoiding"]["psi"]
    assert values["moving"]["safe_margin"] != values["passing"]["safe_margin"]
    assert field.params.lam_prox == 1.2


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


def test_wheelchair_predictive_mpc_rejects_nonprogressive_rollout():
    mpc = WheelchairMPC(horizon=4, dt=0.2, a_max=0.5, beam_width=4)
    mpc._sequence_step_controls = lambda previous, warm_u=None, goal_u=None: [
        np.array([0.05, 0.0], float)]

    v, w = mpc.solve(
        [0.0, 0.0, np.pi],
        [[0.2, 0.0], [0.4, 0.0], [0.6, 0.0], [0.8, 0.0]],
        ZeroField(), goal=[1.0, 0.0], predictive=True)

    assert v == 0.0
    assert w == 0.0
    assert mpc.last_solver_status == "safe_stop: insufficient_progress"
    assert mpc.last_constraint_violation["nonprogressive_rollout"] > 0
    assert mpc.last_objective_terms["best_first_step_goal_progress"] < 0.0
    assert mpc.last_objective_terms["best_reference_progress"] < 0.0


def test_wheelchair_heading_recovery_first_step_limits_spin():
    mpc = WheelchairMPC(
        horizon=4, dt=0.2, a_max=0.5, alpha_max=5.0,
        w_max=1.0, beam_width=4)
    mpc.heading_recovery_w_max = 0.35
    mpc._sequence_step_controls = lambda previous, warm_u=None, goal_u=None: [
        np.array([0.05, -0.8], float)]

    v, w = mpc.solve(
        [0.0, 0.0, 2.0],
        [[0.2, 0.0], [0.4, 0.0], [0.6, 0.0], [0.8, 0.0]],
        ZeroField(), goal=[1.0, 0.0], predictive=True)

    assert v == 0.0
    assert w == 0.0
    assert mpc.last_solver_status == "safe_stop: insufficient_progress"
    assert mpc.last_objective_terms["max_heading_recovery_w"] == 0.35
    assert mpc.last_objective_terms["best_first_step_angular_speed"] > 0.35


def test_wheelchair_heading_recovery_can_use_stsm_turn_budget():
    mpc = WheelchairMPC(
        horizon=4, dt=0.2, a_max=0.5, alpha_max=5.0,
        w_max=1.0, beam_width=4)
    mpc.heading_recovery_w_max = 0.95
    mpc._sequence_step_controls = lambda previous, warm_u=None, goal_u=None: [
        np.array([0.05, -0.8], float)]

    v, w = mpc.solve(
        [0.0, 0.0, 2.0],
        [[0.2, 0.0], [0.4, 0.0], [0.6, 0.0], [0.8, 0.0]],
        ZeroField(), goal=[1.0, 0.0], predictive=True)

    assert v == 0.05
    assert w == -0.8
    assert mpc.last_solver_status.startswith("predictive_beam")
    assert mpc.last_objective_terms["heading_recovery_live"] is True
    assert mpc.last_objective_terms["first_step_live"] is False


def test_wheelchair_nonholonomic_profile_penalizes_regressive_initial_path():
    state = np.array([0.0, 0.0, 0.0])
    goal = np.array([1.0, 0.0])
    forward = np.array([
        [0.0, 0.0],
        [0.25, 0.0],
        [0.50, 0.0],
        [0.75, 0.0],
        [1.00, 0.0],
    ])
    regressive = np.array([
        [0.0, 0.0],
        [-0.12, 0.10],
        [-0.05, -0.12],
        [0.20, 0.12],
        [0.45, -0.05],
        [1.00, 0.0],
    ])

    good = wheelchair_nonholonomic_execution_profile(forward, state, goal)
    bad = wheelchair_nonholonomic_execution_profile(regressive, state, goal)

    assert good["execution_profile_cost"] < bad["execution_profile_cost"]
    assert good["monotonic_regression"] == 0.0
    assert bad["monotonic_regression"] > 0.0
    assert bad["initial_heading_error"] > good["initial_heading_error"]
    assert bad["heading_oscillation"] > good["heading_oscillation"]


def test_terminal_preflight_audits_existing_goal_acceptance_region_only():
    context = build_safety_context(
        ZeroField(), {"minimum_clearance": 0.10, "risk_threshold": 2.0},
        strict=True)
    audit = terminal_acceptance_preflight(
        [-0.55, 0.55], 0.25, context, radial_samples=2, angular_samples=8)

    assert audit["goal"] == [-0.55, 0.55, 0.0]
    assert audit["goal_acceptance_radius"] == 0.25
    assert audit["terminal_acceptance_candidate_count"] == 17
    assert audit["safe_terminal_candidate_count"] == 17
    assert audit["selected_terminal_point"] is None
    assert audit["selection_performed"] is False


def test_wheelchair_sharp_turn_audit_reports_exact_violation_indices():
    audit = wheelchair_sharp_turn_audit(np.asarray([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0],
    ]), turn_limit=0.40)

    assert audit["sharp_turn_indices"] == [1, 2]
    assert audit["max_turn"] == np.pi / 2.0
    assert all(item["local_turn"] > 0.40 for item in audit["sharp_turns"])


def test_wheelchair_refinement_point_limit_preserves_topology_waypoints():
    pts = np.column_stack([
        np.linspace(0.0, 4.0, 120),
        np.sin(np.linspace(0.0, 2.0, 120)) * 0.05,
        np.zeros(120),
    ])
    protected = np.asarray([
        pts[0],
        pts[57],
        pts[-1],
    ])

    bounded, limited = _limit_refinement_points(
        pts, max_points=16, protected_points=protected)

    assert limited is True
    assert len(bounded) <= 16
    assert np.min(np.linalg.norm(bounded - pts[57], axis=1)) < 1e-9
    assert np.allclose(bounded[0], pts[0])
    assert np.allclose(bounded[-1], pts[-1])


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
    assert mpc.last_timing["safety_eval_call_count"] > 0
    assert mpc.last_timing["cache_miss_count"] > 0
    assert (mpc.last_timing["cache_hit_count"] > 0 and
            mpc.last_timing["safety_eval_call_count"] <
            mpc.last_timing["unique_rollout_state_count"])


def test_wheelchair_predictive_mpc_reports_phase_timing():
    mpc = WheelchairMPC(horizon=4, dt=0.2, a_max=0.5, beam_width=4)

    mpc.solve(
        [0.0, 0.0, 0.0],
        [[0.2, 0.0], [0.4, 0.0], [0.6, 0.0], [0.8, 0.0]],
        ZeroField(), goal=[1.0, 0.0], predictive=True)

    timing = mpc.last_timing
    for field in (
            "t_reference_s", "t_rollout_s", "t_safety_eval_s",
            "t_search_s", "t_post_s", "solve_wall_s",
            "safety_eval_call_count", "unique_rollout_state_count",
            "cache_hit_count", "cache_miss_count",
            "hard_safety_prune_count"):
        assert field in timing
        assert timing[field] >= 0.0
    assert timing["solve_wall_s"] >= timing["t_rollout_s"]


def test_recovered_diagnostics_preserve_roslog_selected_corridor():
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "metrics.csv"), "w") as handle:
            handle.write(
                "selected_corridor_id,selected_corridor_label,morse_used\n"
                "planning_failed,planning_failed,1\n")
        with open(os.path.join(tmp, "ros.log"), "w") as handle:
            handle.write(
                "[INFO] [0.0]: [wc][corridor] morse_saddle_2 "
                "base=22.086 adp=0.000 total=22.086 nodes=start,saddle_0,minimum_4,goal\n"
                "[INFO] [0.0]: [wc] selected corridor: wheelchair_c0001 "
                "label=morse_saddle_2 (cost 22.086, source=topology)\n")

        selected_id, selected = selected_candidate(tmp, "wheelchair")

    assert selected_id == "wheelchair_c0001"
    assert selected["corridor_id"] == "wheelchair_c0001"
    assert selected["label"] == "morse_saddle_2"
    assert selected["candidate_source"] == "topology"
    assert selected["topology_nodes"] == [
        "start", "saddle_0", "minimum_4", "goal"]


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
    assert "watchdog_command_age_s" in wheelchair_source
    assert "zero_command_duty_ratio" in wheelchair_source
    assert "mpc_phase_timing" in wheelchair_source
    assert "def _audit_runtime_replan_connectability(" in wheelchair_source
    assert "def _runtime_replan_path_status(" in wheelchair_source
    assert "runtime_replan_connectability_attempts" in wheelchair_source
    assert "runtime_replan_connectability.json" in wheelchair_source
    assert "runtime_recovery_diagnostics.json" in wheelchair_source
    assert "runtime_replan_connector_turn_limit" in wheelchair_source
    assert "runtime_replan_goal_progress_insufficient" in wheelchair_source
    assert "def _make_runtime_safe_connector(" in wheelchair_source
    assert "runtime_replan_no_safe_join_point" in wheelchair_source
    assert "connector_search_expansions" in wheelchair_source
    assert "join_idx_selected" in wheelchair_source
    assert "def _runtime_replan_join_point_audit(" in wheelchair_source
    assert "join_point_source_count" in wheelchair_source
    assert "join_point_audit" in wheelchair_source
    assert "_runtime_replan_connector_safety(" in wheelchair_source
    assert "critical_point_sequence_invalid" in wheelchair_source
    assert "def _runtime_recovery(" in wheelchair_source
    assert "def _try_runtime_corridor_suffix_repair(" in wheelchair_source
    assert "CORRIDOR_SUFFIX_REPAIR" in wheelchair_source
    assert "runtime_suffix_start_index" in wheelchair_source
    assert "suffix_outside_existing_tube" in wheelchair_source
    assert "suffix_current_pose_unsafe" in wheelchair_source
    assert "deferred_to_existing_runtime_chain" in wheelchair_source
    assert wheelchair_source.count("_runtime_recovery(") == 6
    assert wheelchair_source.count("new_corridor = self._plan_corridor()") == 1
    assert "runtime_full_replan_count" in wheelchair_source
    assert "candidate_rank >= current_rank" not in wheelchair_source
    assert "candidate_rank > current_rank" not in wheelchair_source
    assert "runtime_replan_fallback_count" not in wheelchair_source
    assert "_runtime_replan_fallback_corridor" not in wheelchair_source
    assert "_maybe_replan_corridor" not in wheelchair_source
    assert "final_direct_override_enabled" not in wheelchair_source
    assert "final_direct_override_enabled" not in open(
        os.path.join(ROOT, "launch", "wheelchair_action.launch"), "r").read()
    assert '"final_direct_override_active"] = False' in wheelchair_source
    assert "replan_period" not in wheelchair_source
    assert "topology_periodic_replan" not in wheelchair_source
    assert "topology_replan_on_tube_exit" not in wheelchair_source
    assert "topology_replan_on_no_progress" not in wheelchair_source
    assert "runtime_topology_candidate_pool" in wheelchair_source
    assert "_switch_to_ranked_topology_candidate" in wheelchair_source
    assert "topology_runtime_candidate_switch_used" in wheelchair_source
    assert "dynamic_replan_fallback = False" in wheelchair_source
    assert "stsm_liveness_floor_v" in wheelchair_source
    assert "stsm_liveness_w_max" in wheelchair_source
    assert "stsm_liveness_active" in wheelchair_source
    assert "_runtime_candidate_first_step_status" in wheelchair_source
    assert "runtime_switch_precheck_trials" in wheelchair_source
    assert "first_step_not_executable" in wheelchair_source
    assert "ready.runtime_switch_first_step_status" in wheelchair_source
    assert "first_step_progress_ratio" in wheelchair_source
    assert "heading_recovery_w_max" in wheelchair_source
    assert "alignment_floor_scale = 1.0 if liveness_active" in wheelchair_source
    assert "heading_recovery_live" in wheelchair_source
    assert "mpc_heading_recovery_w_max" in open(
        os.path.join(ROOT, "launch", "wheelchair_action.launch"), "r").read()
    assert "first_step_positive_progress" in open(
        os.path.join(ROOT, "src", "stsm_madp", "mpc.py"), "r").read()
    assert "reference_progress_corridor_id" in wheelchair_source
    assert "reference_nearest_index" in wheelchair_source
    assert "mpc_reference_min_lookahead_m" in wheelchair_source
    assert "mpc_reference_min_goal_progress_m" in wheelchair_source
    assert "reference_horizon_goal_progress" in wheelchair_source
    assert "runtime_switch_insufficient_goal_progress" in wheelchair_source
    assert "runtime_switch_not_rank_improving" not in wheelchair_source
    assert "_mark_corridor_runtime_failed" in wheelchair_source
    assert "runtime_failed_corridors" in wheelchair_source
    assert "mpc_local:%s" in wheelchair_source
    assert "global_safe_stop" in wheelchair_source
    assert "post_scale_first_step_live" in wheelchair_source
    assert wheelchair_source.count("_runtime_post_scale_cmd(") == 3
    assert "published_cmd_preview" in wheelchair_source
    assert "post_scale_progress_floor_used" in wheelchair_source
    assert "raw_mpc_cmd" in wheelchair_source
    assert "published_cmd" in wheelchair_source
    assert "final_approach_corridor_weight" in wheelchair_source
    assert "execution_speed_floor" in wheelchair_source
    assert "self.mpc.lam_heading = self.mpc_base_lam_heading *" in wheelchair_source
    assert "final_approach_active = self._in_final_approach(dist)" in wheelchair_source
    assert "not final_approach_active and" in wheelchair_source
    assert "first_speed_shortfall_cost" in open(
        os.path.join(ROOT, "src", "stsm_madp", "mpc.py"), "r").read()
    assert "sequence_speed_shortfall_cost" in open(
        os.path.join(ROOT, "src", "stsm_madp", "mpc.py"), "r").read()
    assert "wheelchair_nonholonomic_execution_profile" in wheelchair_source
    assert "diff_drive_execution_cost" in wheelchair_source
    assert "selected_monotonic_regression" in wheelchair_source
    assert "_select_wheelchair_execution_reference" in wheelchair_source
    assert "diff_drive_launch_prefix" in wheelchair_source
    assert "diff_drive_launch_prefix_turn_recovered" in wheelchair_source
    assert "diff_drive_turn_recovery_used" in wheelchair_source
    assert "selected_diff_drive_reference_repaired" in wheelchair_source
    assert "pre_repair_nonholonomic_execution_profile" in wheelchair_source
    assert "max_refinement_path_points" in wheelchair_source
    assert "max_refined_footprint_check_points" in wheelchair_source
    assert "bounded_reference_path_count" in open(
        os.path.join(ROOT, "src", "stsm_madp",
                     "topology_refinement.py"), "r").read()
    assert '"mpc_runtime_records": list(self.mpc_runtime_records)' in wheelchair_source
    assert "[wc][recovery] no_progress exhausted" in wheelchair_source
    assert '"/stsm/wc_task_complete"' in wheelchair_source
    assert '"/stsm/wc_task_complete"' in metrics_source
    assert "metrics_goal_tolerance=\"${wc_completion_tolerance}\"" in experiment_source
    assert "def first_value(*values):" in experiment_source


def test_baseline_failure_stage_is_execution_not_refinement():
    with open(os.path.join(ROOT, "nodes", "metrics_node.py"), "r") as handle:
        source = handle.read()
    assert 'if variant_name == "baseline":\n                failure_stage = "execution"' in source


def test_adp_terminal_td_target_does_not_bootstrap_next_value():
    critic = ADPCritic(
        feature_names=["bias"], theta=[2.0], mean=[0.0], std=[1.0],
        gamma=0.5)
    current = {"bias": 1.0}

    nonterminal = critic.update_td_detail(
        current, 1.0, current, alpha=0.1, terminal=False)
    terminal = critic.update_td_detail(
        current, 1.0, current, alpha=0.1, terminal=True)

    assert np.isclose(nonterminal["target"], 2.0)
    assert np.isclose(nonterminal["td_error"], 0.0)
    assert np.isclose(terminal["target"], 1.0)
    assert terminal["td_error"] < 0.0


def test_adp_shadow_learning_bounds_updates_and_preserves_seed_file():
    critic = ADPCritic(feature_names=["bias", "phi_total"], theta=[0.0, 0.0],
                        mean=[0.0, 0.0], std=[1.0, 1.0], gamma=0.95)
    learner = ADPTransitionLearner(critic, config={
        "alpha": 1.0,
        "td_error_clip": 0.5,
        "theta_delta_norm_max": 0.05,
        "min_transition_dt": 0.1,
    }, robot="wheelchair")
    learner.observe({"bias": 1.0, "phi_total": 0.0}, 0.0)
    updated = learner.observe(
        {"bias": 1.0, "phi_total": 100.0}, 0.2,
        control_effort=1.0)

    assert updated["status"] == "updated"
    assert abs(updated["td_error"]) <= 0.5
    assert updated["theta_delta_norm"] <= 0.05
    assert learner.diagnostics()["theta_changed"]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "adp_critic_updated.yaml")
        assert save_and_verify_critic(critic, path)
        assert np.allclose(critic.theta, ADPCritic.load_yaml(path).theta)


def test_adp_shadow_learning_skips_nonfinite_and_honors_disable_flag():
    invalid = ADPCritic(feature_names=["bias"], theta=[0.0], mean=[np.nan],
                        std=[1.0])
    learner = ADPTransitionLearner(invalid)
    record = learner.observe({"bias": 1.0}, 0.0)
    assert record["status"] == "skipped"
    assert record["reason"] == "nonfinite_features"

    critic = ADPCritic(feature_names=["bias"], theta=[1.0], mean=[0.0],
                        std=[1.0])
    learner = ADPTransitionLearner(critic, config={"enabled": False})
    learner.observe({"bias": 1.0}, 0.0)
    learner.observe({"bias": 1.0}, 1.0, terminal=True)
    assert np.allclose(critic.theta, [1.0])
    assert not learner.diagnostics()["decision_influence_enabled"]


def test_adp_stability_audit_keeps_td_rule_and_attributes_task_state():
    critic = ADPCritic(
        feature_names=["bias", "candidate_path_length"], theta=[0.0, 0.0],
        mean=[0.0, 0.0], std=[1.0, 1.0],
        metadata={"target_mean": 0.0, "target_p95": 1.0})
    learner = ADPTransitionLearner(critic, config={
        "alpha": 1.0, "td_error_clip": 0.5, "theta_delta_norm_max": 0.05,
        "min_transition_dt": 0.0, "value_outlier_z": 2.0})
    learner.observe({"bias": 1.0, "candidate_path_length": 0.1}, 0.0,
                    task_state="approach", feature_missing={
                        "candidate_path_length": False})
    learner.observe({"bias": 1.0, "candidate_path_length": 10.0}, 1.0,
                    task_state="handover", control_effort=10.0, task_penalty=10.0,
                    feature_missing={"candidate_path_length": False})
    diag = learner.diagnostics()
    assert diag["task_state_breakdown"]["handover"]["count"] == 1
    assert diag["td_clip_count"] == 1
    assert diag["feature_stats"]["candidate_path_length"]["missing_count"] == 0
    assert diag["feature_stats"]["candidate_path_length"]["normalized"]["max"] == 0.1


def test_candidate_conditioned_learner_skips_missing_candidate_context():
    critic = ADPCritic(
        feature_names=["bias", "candidate_path_length"], theta=[0.0, 0.0],
        mean=[0.0, 0.0], std=[1.0, 1.0])
    learner = ADPTransitionLearner(critic, config={"min_transition_dt": 0.0})
    missing = learner.observe(
        {"bias": 1.0, "candidate_path_length": 0.0}, 0.0,
        task_state="approach", feature_missing={"candidate_path_length": True})
    seeded = learner.observe(
        {"bias": 1.0, "candidate_path_length": 1.0}, 1.0,
        task_state="align", feature_missing={"candidate_path_length": False})
    assert missing["reason"] == "candidate_context_unavailable"
    assert not missing["candidate_context_available"]
    assert seeded["status"] == "seed"
    cross = learner.diagnostics()["phase_candidate_context_cross_stats"]
    assert cross["approach|candidate_context_missing"]["skipped_count"] == 1


def test_adp_promotion_gate_blocks_unstable_critic_without_replacing_seed():
    critic = ADPCritic(feature_names=["bias"], theta=[1.0], mean=[0.0],
                        std=[1.0])
    result = evaluate_promotion_gate({
        "update_count": 3, "td_clip_ratio": 0.20, "td_error_abs_mean": 3.0,
        "theta_delta_norm_total": 0.1, "value_outlier_ratio": 0.0},
        critic, "seed.yaml", robot_type="arm", reload_verified=True)
    assert result["promotion_candidate"]
    assert not result["promotion_passed"]
    assert "td_clip_ratio_high" in result["promotion_reasons"]
    assert "td_error_abs_mean_high" in result["promotion_reasons"]
    assert result["identity"]["theta_hash"] == critic_theta_hash(critic)


def test_adp_value_guard_uses_td_return_scale_not_ranking_scale():
    critic = ADPCritic(
        feature_names=["bias"], theta=[100.0], mean=[0.0], std=[1.0],
        metadata={"target_mean": 0.0, "target_p95": 1.0,
                  "ranking_value_center": 1000.0,
                  "ranking_value_scale": 10000.0})
    learner = ADPTransitionLearner(
        critic, config={"min_transition_dt": 0.0, "value_outlier_z": 8.0})
    learner.observe({"bias": 1.0}, 0.0)
    learner.observe({"bias": 1.0}, 1.0)
    assert learner.diagnostics()["value_outlier_ratio"] == 1.0


def test_adp_runtime_identity_rejects_wrong_expected_seed_hash():
    critic = ADPCritic(feature_names=["bias"], theta=[1.0], mean=[0.0],
                        std=[1.0], critic_version="test_v1")
    identity = validate_critic_runtime_identity(
        critic, "seed.yaml", "seed.yaml", "test_v1", "not-the-hash", "arm")
    assert not identity["validated"]
    assert identity["validation_reasons"] == ["expected_theta_hash_mismatch"]


def test_adp_role_uses_explicit_learning_and_influence_signals():
    assert adp_role_from_runtime(True, True, False, effective_lambda=0.0) == (
        "shadow_learning")
    assert adp_role_from_runtime(True, True, True, effective_lambda=0.0,
                                 ranking_contribution=True) == "shadow_learning"
    assert adp_role_from_runtime(True, True, True, effective_lambda=0.02,
                                 control_contribution=False) == "evaluation_only"
    assert adp_role_from_runtime(True, True, True, effective_lambda=0.02,
                                 control_contribution=True) == "control_modifier"
    assert adp_role_from_runtime(True, False, True, effective_lambda=0.02) == (
        "inactive")


def test_decision_trace_ignores_generic_dls_delta_without_adp_influence():
    trace = trace_from_debug(
        {},
        metrics={
            "adp_enabled": 1,
            "adp_learning_enabled": 1,
            "adp_decision_influence_enabled": 0,
            "adp_effective_lambda": 0.0,
            "v_des_delta_norm": 0.5,
            "arm_dls_adp_used": 0,
        }, robot="arm")
    assert trace["adp_role"] == "shadow_learning"
    assert trace["adp_affects_control"] == 0


def test_formal_experiments_default_to_target_calibrated_critics():
    with open(os.path.join(ROOT, "scripts", "run_experiments.sh"), "r") as handle:
        source = handle.read()
    assert "adp_model_override=\"${ADP_MODEL:-}\"" in source
    assert "adp_critic_arm_candidate_conditioned.yaml" in source
    assert "adp_critic_wheelchair_candidate_conditioned.yaml" in source
    assert 'adp_model:=\"${run_adp_model}\"' in source


def test_candidate_conditioned_schema_rejects_legacy_critic():
    legacy = ADPCritic(feature_names=["bias"], theta=[0.0], mean=[0.0],
                       std=[1.0], metadata={"feature_schema_version": "v1"})
    try:
        require_feature_schema(legacy)
        assert False, "legacy critic must not be silently expanded"
    except ValueError as exc:
        assert "critic_feature_schema_mismatch" in str(exc)


def test_arm_and_wheelchair_candidate_conditioned_critics_match_schema():
    for filename in ("adp_critic_arm_candidate_conditioned.yaml",
                     "adp_critic_wheelchair_candidate_conditioned.yaml"):
        critic = ADPCritic.load_yaml(os.path.join(ROOT, "config", filename))
        assert require_feature_schema(critic)


def test_candidate_conditioned_values_change_with_same_robot_state():
    candidate_a, missing_a = candidate_feature_values({
        "path_length": 0.3, "risk_cost": 2.2, "max_risk": 1.4,
        "min_clearance": 0.04, "task_cost": 3.5, "execution_cost": 4.1})
    candidate_b, missing_b = candidate_feature_values({
        "path_length": 1.1, "risk_cost": 1.8, "max_risk": 1.0,
        "min_clearance": 0.01, "task_cost": 3.3, "execution_cost": 3.0})
    assert not any(missing_a.values())
    assert not any(missing_b.values())
    names = ["bias", "candidate_path_length", "candidate_risk_mean"]
    critic = ADPCritic(feature_names=names, theta=[0.0, 2.0, 1.0],
                       mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    value_a = critic.predict_detail(dict(bias=1.0, **candidate_a))["raw"]
    value_b = critic.predict_detail(dict(bias=1.0, **candidate_b))["raw"]
    assert value_a != value_b
    terms, _meta = adp_ranking_adjustments(
        [value_a, value_b], lambda_adp=0.02, norm_clip=10.0)
    assert terms[0]["adp_value_raw"] != terms[1]["adp_value_raw"]
    zero_terms, _meta = adp_ranking_adjustments(
        [value_a, value_b], lambda_adp=0.0, norm_clip=10.0)
    assert all(term["adp_cost"] == 0.0 for term in zero_terms)


def test_candidate_feature_builder_keeps_canonical_candidate_schema_values():
    canonical = {
        "candidate_path_length": 1.25,
        "candidate_risk_mean": 0.75,
        "candidate_risk_max": 1.5,
        "candidate_min_clearance": 0.08,
        "candidate_task_cost": 2.5,
        "candidate_execution_cost": 3.5,
    }
    values, missing = candidate_feature_values(canonical)
    assert values == canonical
    assert not any(missing.values())


def test_recentered_candidate_normalization_preserves_critic_value():
    critic = ADPCritic(
        feature_names=["bias", "candidate_path_length"],
        theta=[2.0, 3.0], mean=[0.0, 1.0], std=[1.0, 0.5])
    features = {"bias": 1.0, "candidate_path_length": 2.0}
    before = critic.predict_detail(features)["raw"]
    recenter_critic_feature_normalization(
        critic, {"candidate_path_length": {"mean": 2.0, "std": 1.0}})
    assert np.isclose(before, critic.predict_detail(features)["raw"])
    assert np.isclose(critic.featurize(features)[1], 0.0)


def test_adp_calibration_fits_real_transition_cost_to_go_targets():
    template = ADPCritic(feature_names=["bias", "phi_total"], theta=[7.0, 2.0],
                         mean=[0.0, 0.0], std=[1.0, 1.0], gamma=0.5)
    records = [
        {"updated": True, "features_t": {"bias": 1.0, "phi_total": 0.0},
         "stage_cost": 1.0, "terminal": False},
        {"updated": True, "features_t": {"bias": 1.0, "phi_total": 1.0},
         "stage_cost": 2.0, "terminal": True},
    ]
    critic, summary = fit_critic_from_transition_records(records, template)

    assert summary["sample_count"] == 2
    assert summary["episode_count"] == 1
    assert np.isclose(summary["target_max"], 2.0)
    assert np.isfinite(critic.theta).all()
    assert critic.critic_version.endswith("_calibrated")


def test_failed_wheelchair_diagnostics_persist_candidate_path_trace():
    trace = {
        "candidate_id": "wheelchair_c0001",
        "label": "morse_saddle_2",
        "raw_candidate": {"point_count": 2, "points": [
            {"index": 0, "x": 0.0, "y": 0.0,
             "source_stage": "raw_candidate", "source_index": 0},
            {"index": 1, "x": 1.0, "y": 0.0,
             "source_stage": "raw_candidate", "source_index": 1},
        ]},
        "refinement": {"point_count": 2, "points": [
            {"index": 0, "x": 0.0, "y": 0.0,
             "source_stage": "raw_candidate", "source_index": 0},
            {"index": 1, "x": 1.0, "y": 0.0,
             "source_stage": "raw_candidate", "source_index": 1},
        ]},
        "safe_terminal_rebuild": {"applied": False, "points": None},
        "turn_repair": {"applied": False, "points": None},
        "final_reference": None,
        "path_trace_complete": False,
    }
    with tempfile.TemporaryDirectory() as tmp:
        write_failed_topology_diagnostics(
            tmp, "wheelchair", {"refinement_attempts": [
                {"candidate_path_trace": trace}]})
        with open(os.path.join(tmp, "candidate_path_trace.json"), "r") as handle:
            payload = json.load(handle)

    assert payload["robot_type"] == "wheelchair"
    assert payload["candidate_count"] == 1
    saved = payload["candidates"][0]
    assert saved["candidate_id"] == "wheelchair_c0001"
    assert saved["refinement"]["points"][1]["source_index"] == 1
    assert saved["final_reference"] is None
    assert not saved["path_trace_complete"]


def test_r009_safe_terminal_replay_recreates_authoritative_terminal_audit():
    script = os.path.join(
        ROOT, "scripts", "analysis", "replay_c0001_safe_terminal_trials.py")
    spec = importlib.util.spec_from_file_location("r009_terminal_replay", script)
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)

    audit = terminal_acceptance_preflight(
        replay.GOAL, 0.25, replay._scene_context())

    assert np.isclose(audit["goal_clearance"], -0.07193582655542459)
    assert np.isclose(audit["goal_risk"], 2.5016565419952275)
    assert audit["safe_terminal_candidate_count"] == 6

    trace = os.path.join(ROOT, "results", "runs", "20260831_R009",
                         "wheelchair", "stsm", "candidate_path_trace.json")
    with tempfile.TemporaryDirectory() as tmp:
        replayed = replay.run(trace, os.path.join(tmp, "turn_origin.json"))
    trial = next(item for item in replayed["trials"]
                 if item["rebuild_start_idx"] == 17 and
                 np.allclose(item["selected_terminal"][:2],
                             [-0.4782468564315457, 0.7232274123458663]))
    turn = trial["turn_origin_audit"]
    prefix = trial["launch_prefix_audit"]
    assert prefix["launch_prefix_point_count"] > 5
    assert prefix["launch_prefix_max_turn"] <= replay.TURN + 1e-9
    assert prefix["prefix_join_max_turn"] <= replay.TURN + 1e-9
    assert prefix["launch_prefix_hard_valid"]
    assert prefix["launch_prefix_min_clearance"] >= replay.CLEARANCE
    assert turn["max_turn_index"] != 1
    assert np.isclose(turn["max_turn"], 0.6590043705190771)
    assert turn["turn_origin"] == "refined_main_path"
