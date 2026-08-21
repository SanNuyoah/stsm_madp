import os
import sys

import numpy as np


SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)


from stsm_madp.mpc import ArmMPC, WheelchairMPC, _phase_clearance_threshold
from stsm_madp.manifold import Corridor
from stsm_madp.manifold_constraint import build_manifold_constraint
from stsm_madp.deform import (
    deform_trajectory, interpolate_by_segments, path_length,
    protected_waypoint_distances, shortcut_trajectory,
    topology_preserving_shortcut)
from stsm_madp.topology import (
    TopologicalCorridorPlanner, TopologyNode,
    _arm_candidate_filter_classification, _arm_recovery_cost,
    topology_param_or_auto, topology_profile_defaults)


class StepField(object):
    def __init__(self, x_limit):
        self.x_limit = float(x_limit)
        self.anchors = []

    def phi_s(self, p, v=None):
        p = np.asarray(p, float)
        return 10.0 if float(p[0]) > self.x_limit else 0.0


class ZeroField(object):
    def __init__(self, anchors=None):
        self.anchors = anchors or []

    def phi_s(self, p, v=None):
        return 0.0

    def grad_phi_s(self, p):
        return np.zeros_like(np.asarray(p, float))


class ForbiddenXAnchor(object):
    forbidden = True
    type = "table"

    def __init__(self, x_min):
        self.x_min = float(x_min)

    def signed_distance(self, p):
        p = np.asarray(p, float)
        return -1.0 if float(p[0]) >= self.x_min else 1.0


def test_topology_profiles_encode_robot_specific_defaults():
    wc = topology_profile_defaults("wheelchair")
    arm = topology_profile_defaults("arm")

    assert wc["workspace_dimension"] == 2
    assert arm["workspace_dimension"] == 3
    assert wc["grid_resolution"] > arm["grid_resolution"]
    assert wc["min_clearance"] > arm["min_clearance"]
    assert wc["motion_space"] == "planar_body"
    assert arm["motion_space"] == "projected_3d_handover"
    assert wc["neighbor_strategy"] == "sparse_body"
    assert arm["neighbor_strategy"] == "dense_projected_3d"
    assert arm["neighbor_k"] > wc["neighbor_k"]


def test_topology_preserving_shortcut_keeps_protected_saddle():
    field = ZeroField()
    skeleton = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.2, 0.0],
        [1.0, 0.0, 0.0],
    ])
    path, protected = interpolate_by_segments(skeleton, n=15)
    direct = shortcut_trajectory(path, field)
    kept, kept_protected = topology_preserving_shortcut(
        path, protected, field)

    assert len(direct) == 2
    assert len(kept_protected) == 3
    assert protected_waypoint_distances(
        kept, np.array([[0.5, 0.2, 0.0]]))[0] < 1e-9


def test_deform_trajectory_keeps_protected_waypoint_fixed():
    field = ZeroField()
    path, protected = interpolate_by_segments(np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.2, 0.0],
        [1.0, 0.0, 0.0],
    ]), n=15)

    out = deform_trajectory(
        path, field, lam_social=0.0, lam_smooth=1.0, iters=10,
        protected_indices=protected)

    assert np.linalg.norm(out[protected[1]] - path[protected[1]]) < 1e-9


def test_topology_profile_defaults_apply_and_explicit_values_override():
    arm_planner = TopologicalCorridorPlanner(
        ZeroField(), rho=1.0, bounds=[(0.0, 1.0), (0.0, 1.0)],
        topology_profile="arm")
    wc_planner = TopologicalCorridorPlanner(
        ZeroField(), rho=1.0, bounds=[(0.0, 1.0), (0.0, 1.0)],
        topology_profile="wheelchair", grid_resolution=0.11,
        min_clearance=0.09)

    assert arm_planner.topology_profile == "arm"
    assert arm_planner.workspace_dimension == 3
    assert arm_planner.grid_resolution == 0.04
    assert arm_planner.hard_clearance == 0.03
    assert wc_planner.topology_profile == "wheelchair"
    assert wc_planner.workspace_dimension == 2
    assert wc_planner.grid_resolution == 0.11
    assert wc_planner.min_clearance == 0.09


def test_topology_auto_values_defer_to_robot_profile():
    assert topology_param_or_auto("auto") is None
    assert topology_param_or_auto("profile") is None
    arm_defaults = topology_profile_defaults("arm")
    planner = TopologicalCorridorPlanner(
        ZeroField(), rho=1.0, bounds=[(0.0, 1.0), (0.0, 1.0)],
        topology_profile="arm", grid_resolution="auto",
        merge_radius="auto", min_clearance="auto",
        hard_clearance="auto", neighbor_k="auto")

    assert planner.grid_resolution == arm_defaults["grid_resolution"]
    assert planner.merge_radius == arm_defaults["merge_radius"]
    assert planner.min_clearance == arm_defaults["min_clearance"]
    assert planner.hard_clearance == arm_defaults["hard_clearance"]
    assert planner.neighbor_k == arm_defaults["neighbor_k"]


def test_handover_phase_manifold_relaxes_clearance_not_collision_risk():
    constraint = build_manifold_constraint(
        minimum_clearance=0.10,
        risk_threshold=6.0,
        phase="handover",
        robot_type="arm")

    assert constraint["phase"] == "handover"
    assert constraint["interaction_region_allowed"] is True
    assert constraint["effective_minimum_clearance"] == 0.15
    assert constraint["effective_risk_threshold"] == constraint["risk_threshold"]


def test_arm_candidate_filter_classifies_minor_recoverable_failures():
    clearance_validation = {
        "failure_reason": "end_effector_clearance_violation",
        "min_end_effector_clearance": 0.08,
        "required_clearance": 0.10,
        "max_risk": 2.0,
        "risk_threshold": 6.0,
    }
    orientation_validation = {
        "failure_reason": "orientation_error",
        "orientation_error": 0.15,
        "max_risk": 2.0,
        "risk_threshold": 6.0,
    }

    assert _arm_candidate_filter_classification(
        "end_effector_clearance_violation",
        validation=clearance_validation) == "recoverable"
    assert _arm_candidate_filter_classification(
        "orientation_error",
        validation=orientation_validation) == "recoverable"
    assert _arm_candidate_filter_classification(
        "link_collision",
        validation={"failure_reason": "link_collision"}) == "invalid"
    assert _arm_recovery_cost(
        "end_effector_clearance_violation",
        validation=clearance_validation) > 0.0


def test_arm_approach_phase_clearance_schedule_interpolates_by_progress():
    context = {
        "minimum_clearance": 0.08,
        "manifold_constraint": build_manifold_constraint(
            minimum_clearance=0.08,
            risk_threshold=6.0,
            phase="approach",
            robot_type="arm"),
    }

    assert abs(_phase_clearance_threshold(
        context, "approach", 0.0) - 0.35) < 1e-9
    assert abs(_phase_clearance_threshold(
        context, "approach", 1.0) - 0.20) < 1e-9
    assert abs(_phase_clearance_threshold(
        context, "handover", 0.5) - 0.15) < 1e-9
    assert abs(_phase_clearance_threshold(
        context, "return", 0.5) - 0.275) < 1e-9


def test_candidate_pool_min_preserves_ranking_competition_after_dedupe():
    planner = TopologicalCorridorPlanner(
        ZeroField(), rho=1.0, bounds=[(0.0, 1.0), (0.0, 1.0)],
        topology_profile="wheelchair", corridor_dedupe_distance=0.20,
        candidate_pool_min=3)
    corridors = []
    for idx, offset in enumerate([0.0, 0.01, 0.02]):
        corr = Corridor(np.array([
            [0.0, 0.0, 0.0],
            [0.5, offset, 0.0],
            [1.0, 0.0, 0.0],
        ]), radius=0.35, label="c{}".format(idx), cost=float(idx))
        corr.cost = float(idx)
        corridors.append(corr)

    kept = planner._dedupe_corridors_by_geometry(
        corridors, np.array([0.0, 0.0]), np.array([1.0, 0.0]),
        min_keep=3)

    assert len(kept) == 3
    assert any(getattr(c, "geometry_duplicate_retained", False) for c in kept)


def test_robot_profiles_apply_distinct_neighbor_strategies():
    nodes = [
        TopologyNode("start", "start", [0.0, 0.0, 0.0], ij=(0, 0)),
        TopologyNode("a", "minimum", [0.2, 0.0, 0.0], ij=(1, 0)),
        TopologyNode("b", "saddle", [0.4, 0.0, 0.0], ij=(2, 0)),
        TopologyNode("c", "minimum", [0.6, 0.0, 0.0], ij=(3, 0)),
        TopologyNode("d", "saddle", [0.8, 0.0, 0.0], ij=(4, 0)),
        TopologyNode("e", "minimum", [1.0, 0.0, 0.0], ij=(5, 0)),
        TopologyNode("goal", "goal", [1.2, 0.0, 0.0], ij=(6, 0)),
    ]
    wc = TopologicalCorridorPlanner(
        ZeroField(), rho=1.0, bounds=[(0.0, 1.2), (0.0, 1.0)],
        topology_profile="wheelchair", grid_resolution=0.05,
        merge_radius=0.05, neighbor_k=12)
    arm = TopologicalCorridorPlanner(
        ZeroField(), rho=1.0, bounds=[(0.0, 1.2), (0.0, 1.0)],
        topology_profile="arm", grid_resolution=0.05,
        merge_radius=0.05, neighbor_k=12)

    assert len(arm._neighbor_pairs(nodes)) > len(wc._neighbor_pairs(nodes))


def test_topology_tracking_cost_penalizes_hard_to_follow_turns():
    planner = TopologicalCorridorPlanner(
        ZeroField(), rho=1.0, bounds=[(0.0, 1.0), (0.0, 1.0)],
        topology_profile="wheelchair",
        dynamics_profile={"v_max": 0.75, "w_max": 1.2, "nominal_speed": 0.35})
    straight = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    sharp = np.array([
        [0.0, 0.0, 0.0],
        [0.25, 0.0, 0.0],
        [0.25, 0.25, 0.0],
        [1.0, 0.25, 0.0],
    ])

    straight_metrics = planner._path_tracking_metrics(straight)
    sharp_metrics = planner._path_tracking_metrics(sharp)

    assert straight_metrics["tracking_cost"] == 0.0
    assert sharp_metrics["tracking_cost"] > straight_metrics["tracking_cost"]
    assert sharp_metrics["max_curvature"] > 0.0
    assert sharp_metrics["turn_violation"] > 0.0


def test_topology_grid_rejects_cell_when_interest_point_is_unsafe():
    planner = TopologicalCorridorPlanner(
        StepField(0.6), rho=1.0, bounds=[(0.0, 0.0), (0.0, 0.0)],
        grid_resolution=0.1,
        interest_config={
            "enabled": True,
            "offsets": {
                "ee": [0.0, 0.0, 0.0],
                "object": [1.0, 0.0, 0.0],
            },
            "labels": ["ee", "object"],
            "rho": 1.0,
        })

    grid = planner.build_grid(np.array([0.0, 0.0, 0.0]))

    assert grid["phi"][0, 0] == 10.0
    assert grid["interest_phi"][0, 0] == 10.0
    assert bool(grid["safe"][0, 0]) is False


def test_topology_build_graph_adds_astar_edges_between_safe_nodes():
    planner = TopologicalCorridorPlanner(
        ZeroField(), rho=1.0, bounds=[(0.0, 1.0), (0.0, 1.0)],
        grid_resolution=0.25, min_clearance=0.0)
    start = np.array([0.0, 0.0, 0.0])
    goal = np.array([1.0, 1.0, 0.0])
    planner.build_grid(goal)

    _nodes, edges, _critical = planner.build_graph(
        start, goal,
        critical={"minima": [], "saddles": [], "maxima": []},
        semantic_nodes=[])

    assert "start" in edges
    assert any(edge["to"] == "goal" for edge in edges["start"])


def test_topology_neighbor_pairs_are_sparse_not_complete_graph():
    planner = TopologicalCorridorPlanner(
        ZeroField(), rho=1.0, bounds=[(0.0, 3.0), (0.0, 3.0)],
        grid_resolution=0.5, min_clearance=0.0)
    goal = np.array([3.0, 3.0, 0.0])
    planner.build_grid(goal)
    semantic = [
        np.array([0.0, 3.0, 0.0]), np.array([3.0, 0.0, 0.0]),
        np.array([1.5, 0.0, 0.0]), np.array([0.0, 1.5, 0.0]),
        np.array([1.5, 3.0, 0.0]), np.array([3.0, 1.5, 0.0]),
        np.array([1.5, 1.5, 0.0]), np.array([0.75, 2.25, 0.0]),
    ]
    nodes, edges, _critical = planner.build_graph(
        np.array([0.0, 0.0, 0.0]), goal,
        critical={"minima": [], "saddles": [], "maxima": []},
        semantic_nodes=semantic)

    undirected_edges = sum(len(v) for v in edges.values()) // 2
    complete_edges = len(nodes) * (len(nodes) - 1) // 2
    assert undirected_edges > 0
    assert undirected_edges < complete_edges


def test_topology_clearance_below_target_is_soft_penalty_not_reject():
    planner = TopologicalCorridorPlanner(
        StepField(0.9), rho=1.0, bounds=[(0.0, 1.0), (0.0, 0.5)],
        grid_resolution=0.25, min_clearance=0.40, hard_clearance=0.05,
        neighbor_k=8)
    start = np.array([0.0, 0.25, 0.0])
    goal = np.array([0.75, 0.25, 0.0])
    critical = {
        "minima": [],
        "saddles": [{
            "point": np.array([0.5, 0.25, 0.0], float),
            "p2": np.array([0.5, 0.25], float),
            "ij": (2, 1),
            "psi": 0.0,
            "phi": 0.0,
            "clearance": 0.25,
        }],
        "maxima": [],
        "_raw_counts": {"minima": 0, "saddles": 1, "maxima": 0},
        "_safe_counts": {"minima": 0, "saddles": 1, "maxima": 0},
        "_reject_counts": {},
    }
    planner.detect_morse_points = lambda: critical

    corridors = planner.enumerate_corridors(start, goal, k=2, radius=0.2)

    assert corridors
    assert corridors[0].label.startswith("morse_saddle_")
    assert corridors[0].min_clearance < planner.min_clearance
    assert corridors[0].min_clearance >= planner.hard_clearance
    assert corridors[0].clearance_penalty > 0.0
    assert planner.last_debug["num_candidate_corridors"] > 0
    assert planner.last_debug["edge_clearance_reject_count"] == 0


def test_topology_generates_forced_morse_saddle_corridor():
    planner = TopologicalCorridorPlanner(
        ZeroField(), rho=1.0, bounds=[(0.0, 1.0), (0.0, 1.0)],
        grid_resolution=0.25, min_clearance=0.0, hard_clearance=0.0,
        neighbor_k=8)
    saddle_ij = (2, 2)
    saddle_p2 = np.array([0.5, 0.5], float)
    critical = {
        "minima": [],
        "saddles": [{
            "point": np.array([0.5, 0.5, 0.0], float),
            "p2": saddle_p2,
            "ij": saddle_ij,
            "psi": 0.0,
            "phi": 0.0,
            "clearance": float("inf"),
        }],
        "maxima": [],
        "_raw_counts": {"minima": 0, "saddles": 1, "maxima": 0},
        "_reject_counts": {},
    }
    planner.detect_morse_points = lambda: critical

    corridors = planner.enumerate_corridors(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
        k=2, radius=0.2, semantic_nodes=[])

    assert any(c.label.startswith("morse_saddle_") for c in corridors)
    morse = [c for c in corridors if c.label.startswith("morse_saddle_")][0]
    assert morse.morse_node_ids == ["saddle_0"]
    assert morse.morse_node_types == ["saddle"]
    assert planner.last_debug["num_used_saddles"] > 0
    assert planner.last_debug["used_saddles"] > 0
    assert planner.last_debug["raw_saddle_count"] == 1
    assert planner.last_debug["filtered_saddle_count"] == 1
    assert planner.last_debug["num_forced_critical_corridors"] > 0


def test_topology_morse_primary_limits_default_graph_and_semantic_candidates():
    planner = TopologicalCorridorPlanner(
        ZeroField(), rho=1.0, bounds=[(0.0, 1.0), (0.0, 1.0)],
        grid_resolution=0.25, min_clearance=0.0, hard_clearance=0.0,
        neighbor_k=8, morse_decision_mode="priority")
    saddle_p2 = np.array([0.5, 0.5], float)
    critical = {
        "minima": [],
        "saddles": [{
            "point": np.array([0.5, 0.5, 0.0], float),
            "p2": saddle_p2,
            "ij": (2, 2),
            "psi": 0.0,
            "phi": 0.0,
            "clearance": float("inf"),
        }],
        "maxima": [],
        "_raw_counts": {"minima": 0, "saddles": 1, "maxima": 0},
        "_safe_counts": {"minima": 0, "saddles": 1, "maxima": 0},
        "_reject_counts": {},
    }
    planner.detect_morse_points = lambda: critical

    corridors = planner.enumerate_corridors(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
        k=1, radius=0.2)

    assert [c.label for c in corridors] == ["morse_saddle_0"]
    assert planner.last_debug["num_graph_fallback_corridors"] == 0
    assert not any(n.kind == "semantic" for n in planner.last_debug["nodes"])


def test_topology_morse_core_suppresses_plain_graph_fallback_even_when_allowed():
    planner = TopologicalCorridorPlanner(
        ZeroField(), rho=1.0, bounds=[(0.0, 1.0), (0.0, 1.0)],
        grid_resolution=0.25, min_clearance=0.0, hard_clearance=0.0,
        neighbor_k=8, allow_graph_fallback_with_morse=True,
        morse_core_required=True, morse_decision_mode="priority")
    critical = {
        "minima": [],
        "saddles": [{
            "id": "saddle_raw_0",
            "point": np.array([0.5, 0.5, 0.0], float),
            "p2": np.array([0.5, 0.5], float),
            "ij": (2, 2),
            "psi": 0.0,
            "phi": 0.0,
            "clearance": float("inf"),
        }],
        "maxima": [],
        "_raw_counts": {"minima": 0, "saddles": 1, "maxima": 0},
        "_safe_counts": {"minima": 0, "saddles": 1, "maxima": 0},
        "_merge_counts": {"minima": 0, "saddles": 0, "maxima": 0},
        "_reject_counts": {},
        "_critical_chain": {
            "minima": [],
            "saddles": [{
                "id": "saddle_raw_0", "stage": "filtered",
                "status": "filtered", "reason": "",
            }],
            "maxima": [],
        },
    }
    planner.detect_morse_points = lambda: critical

    corridors = planner.enumerate_corridors(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
        k=3, radius=0.2)

    assert corridors
    assert all(getattr(c, "morse_induced", False) for c in corridors)
    assert not any(c.label.startswith("graph_direct_") for c in corridors)
    assert planner.last_debug["morse_core_required"] is True
    assert planner.last_debug["selected_morse_induced"] is True


def test_topology_morse_primary_prefers_near_cost_critical_corridor():
    planner = TopologicalCorridorPlanner(
        ZeroField(), rho=1.0, bounds=[(0.0, 1.0), (0.0, 1.0)],
        grid_resolution=0.25, saddle_tie_ratio=0.10,
        morse_decision_mode="priority")
    graph = Corridor(np.zeros((2, 3)), 0.2, "graph_direct_0", 1.0)
    morse = Corridor(np.zeros((2, 3)), 0.2, "morse_minima_0", 1.05)

    ordered = planner._sort_with_saddle_tie([graph, morse])

    assert ordered[0].label == "morse_minima_0"


def test_topology_balanced_mode_orders_by_weighted_cost_not_morse_priority():
    planner = TopologicalCorridorPlanner(
        ZeroField(), rho=1.0, bounds=[(0.0, 1.0), (0.0, 1.0)],
        grid_resolution=0.25)
    graph = Corridor(np.zeros((2, 3)), 0.2, "graph_direct_0", 1.0)
    saddle = Corridor(np.zeros((2, 3)), 0.2, "morse_saddle_0", 1.20)

    ordered = planner._sort_with_saddle_tie([graph, saddle])

    assert ordered[0].label == "graph_direct_0"
    assert not saddle.morse_priority_applied
    assert saddle.topology_selection_reason == "balanced_weighted_cost"


def test_topology_morse_priority_can_prefer_saddle_beyond_tie_ratio():
    planner = TopologicalCorridorPlanner(
        ZeroField(), rho=1.0, bounds=[(0.0, 1.0), (0.0, 1.0)],
        grid_resolution=0.25, saddle_tie_ratio=0.05,
        morse_priority_ratio=0.25, morse_decision_mode="priority")
    graph = Corridor(np.zeros((2, 3)), 0.2, "graph_direct_0", 1.0)
    saddle = Corridor(np.zeros((2, 3)), 0.2, "morse_saddle_0", 1.20)

    ordered = planner._sort_with_saddle_tie([graph, saddle])

    assert ordered[0].label == "morse_saddle_0"
    assert saddle.morse_priority_applied


def test_topology_type_selection_prefers_saddle_over_lower_cost_minimum():
    planner = TopologicalCorridorPlanner(
        ZeroField(), rho=1.0, bounds=[(0.0, 1.0), (0.0, 1.0)],
        grid_resolution=0.25, saddle_tie_ratio=0.05,
        morse_priority_ratio=0.25,
        morse_saddle_priority_ratio=0.50,
        morse_mix_priority_ratio=0.50,
        morse_minima_priority_ratio=0.25, morse_decision_mode="priority")
    graph = Corridor(np.zeros((2, 3)), 0.2, "graph_direct_0", 1.0)
    minima = Corridor(np.zeros((2, 3)), 0.2, "morse_minima_0", 1.05)
    saddle = Corridor(np.zeros((2, 3)), 0.2, "morse_saddle_0", 1.40)

    ordered = planner._sort_with_saddle_tie([graph, minima, saddle])

    assert ordered[0].label == "morse_saddle_0"
    assert saddle.morse_priority_applied
    assert saddle.topology_selection_reason.startswith("morse_saddle_priority")


def test_topology_saddle_value_bonus_is_not_given_to_minima():
    planner = TopologicalCorridorPlanner(
        ZeroField(), rho=1.0, bounds=[(0.0, 1.0), (0.0, 1.0)],
        grid_resolution=0.25, lambda_saddle_value=0.4,
        morse_decision_mode="priority")

    saddle_bonus = planner._saddle_value_bonus(["saddle"], 2.0)
    mix_bonus = planner._saddle_value_bonus(["saddle", "minimum"], 2.0)
    minima_bonus = planner._saddle_value_bonus(["minimum"], 2.0)

    assert saddle_bonus > 0.0
    assert mix_bonus > 0.0
    assert mix_bonus < saddle_bonus
    assert minima_bonus == 0.0


def test_topology_saddle_corridor_records_value_bonus():
    planner = TopologicalCorridorPlanner(
        ZeroField(), rho=1.0, bounds=[(0.0, 1.0), (0.0, 1.0)],
        grid_resolution=0.25, min_clearance=0.0, hard_clearance=0.0,
        neighbor_k=8, lambda_saddle_value=0.4,
        morse_decision_mode="priority")
    saddle_p2 = np.array([0.5, 0.5], float)
    critical = {
        "minima": [],
        "saddles": [{
            "id": "saddle_raw_0",
            "point": np.array([0.5, 0.5, 0.0], float),
            "p2": saddle_p2,
            "ij": (2, 2),
            "psi": 0.0,
            "phi": 0.0,
            "clearance": float("inf"),
        }],
        "maxima": [],
        "_raw_counts": {"minima": 0, "saddles": 1, "maxima": 0},
        "_safe_counts": {"minima": 0, "saddles": 1, "maxima": 0},
        "_merge_counts": {"minima": 0, "saddles": 0, "maxima": 0},
        "_reject_counts": {},
        "_critical_chain": {
            "minima": [],
            "saddles": [{
                "id": "saddle_raw_0", "stage": "filtered",
                "status": "filtered", "reason": "",
            }],
            "maxima": [],
        },
    }
    planner.detect_morse_points = lambda: critical

    corridors = planner.enumerate_corridors(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
        k=1, radius=0.2, semantic_nodes=[])

    assert corridors
    assert corridors[0].label.startswith("morse_saddle_")
    assert corridors[0].saddle_value_bonus > 0.0
    assert planner.last_debug["selected_saddle_value_bonus"] > 0.0


def test_topology_generates_morse_saddle_pair_corridor():
    planner = TopologicalCorridorPlanner(
        ZeroField(), rho=1.0, bounds=[(0.0, 1.0), (0.0, 1.0)],
        grid_resolution=0.25, min_clearance=0.0, hard_clearance=0.0,
        neighbor_k=8, morse_decision_mode="priority")
    critical = {
        "minima": [],
        "saddles": [
            {
                "point": np.array([0.25, 0.25, 0.0], float),
                "p2": np.array([0.25, 0.25], float),
                "ij": (1, 1),
                "psi": 0.0,
                "phi": 0.0,
                "clearance": float("inf"),
            },
            {
                "point": np.array([0.75, 0.75, 0.0], float),
                "p2": np.array([0.75, 0.75], float),
                "ij": (3, 3),
                "psi": 0.0,
                "phi": 0.0,
                "clearance": float("inf"),
            },
        ],
        "maxima": [],
        "_raw_counts": {"minima": 0, "saddles": 2, "maxima": 0},
        "_safe_counts": {"minima": 0, "saddles": 2, "maxima": 0},
        "_reject_counts": {},
    }
    planner.detect_morse_points = lambda: critical

    corridors = planner.enumerate_corridors(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
        k=3, radius=0.2, semantic_nodes=[])

    pairs = [c for c in corridors if c.label.startswith("morse_saddle_pair_")]
    assert pairs
    assert pairs[0].morse_node_types == ["saddle", "saddle"]
    assert planner.last_debug["num_saddle_pair_corridors"] > 0


def test_topology_records_critical_chain_selected_and_not_selected_reasons():
    planner = TopologicalCorridorPlanner(
        ZeroField(), rho=1.0, bounds=[(0.0, 1.0), (0.0, 1.0)],
        grid_resolution=0.25, min_clearance=0.0, hard_clearance=0.0,
        neighbor_k=8, morse_decision_mode="priority")
    critical = {
        "minima": [],
        "saddles": [
            {
                "id": "saddle_raw_0",
                "point": np.array([0.25, 0.25, 0.0], float),
                "p2": np.array([0.25, 0.25], float),
                "ij": (1, 1),
                "psi": 0.0,
                "phi": 0.0,
                "clearance": float("inf"),
            },
            {
                "id": "saddle_raw_1",
                "point": np.array([0.75, 0.25, 0.0], float),
                "p2": np.array([0.75, 0.25], float),
                "ij": (3, 1),
                "psi": 0.0,
                "phi": 0.0,
                "clearance": float("inf"),
            },
        ],
        "maxima": [],
        "_raw_counts": {"minima": 0, "saddles": 2, "maxima": 0},
        "_safe_counts": {"minima": 0, "saddles": 2, "maxima": 0},
        "_merge_counts": {"minima": 0, "saddles": 0, "maxima": 0},
        "_reject_counts": {},
        "_critical_chain": {
            "minima": [],
            "saddles": [
                {
                    "id": "saddle_raw_0", "stage": "filtered",
                    "status": "filtered", "reason": "",
                },
                {
                    "id": "saddle_raw_1", "stage": "filtered",
                    "status": "filtered", "reason": "",
                },
            ],
            "maxima": [],
        },
    }
    planner.detect_morse_points = lambda: critical

    corridors = planner.enumerate_corridors(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
        k=2, radius=0.2, semantic_nodes=[])

    assert corridors
    chain = planner.last_debug["critical_chain"]["saddles"]
    statuses = {rec["id"]: rec["status"] for rec in chain}
    reasons = {rec["id"]: rec.get("reason", "") for rec in chain}
    assert set(statuses.values()) == {"selected", "not_selected"}
    assert "higher_cost" in reasons.values()
    assert planner.last_debug["num_selected_saddles"] == 1
    assert planner.last_debug["critical_reason_counts"]["higher_cost"] == 1


def test_topology_safety_audit_records_grid_edge_and_corridor_conditions():
    planner = TopologicalCorridorPlanner(
        ZeroField(), rho=1.0, bounds=[(0.0, 1.0), (0.0, 1.0)],
        grid_resolution=0.25, min_clearance=0.0, hard_clearance=0.0,
        neighbor_k=8)
    critical = {
        "minima": [],
        "saddles": [{
            "point": np.array([0.5, 0.5, 0.0], float),
            "p2": np.array([0.5, 0.5], float),
            "ij": (2, 2),
            "psi": 0.0,
            "phi": 0.0,
            "clearance": float("inf"),
        }],
        "maxima": [],
        "_raw_counts": {"minima": 0, "saddles": 1, "maxima": 0},
        "_safe_counts": {"minima": 0, "saddles": 1, "maxima": 0},
        "_reject_counts": {},
    }
    planner.detect_morse_points = lambda: critical

    corridors = planner.enumerate_corridors(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
        k=1, radius=0.2, semantic_nodes=[])

    assert corridors
    audit = planner.last_debug["safety_audit"]
    assert audit["grid_counts"]["kept"] > 0
    assert audit["edge_counts"]["kept"] > 0
    assert audit["corridor_counts"]["kept"] > 0
    sample = audit["corridors"][0]
    for key in (
            "center_phi", "center_rho", "interest_phi", "interest_rho",
            "forbidden_hit", "clearance", "hard_clearance",
            "clearance_safe", "safe"):
        assert key in sample


def test_topology_grid_rejects_forbidden_interest_hit_even_with_low_risk():
    planner = TopologicalCorridorPlanner(
        ZeroField([ForbiddenXAnchor(0.8)]), rho=1.0,
        bounds=[(0.0, 0.0), (0.0, 0.0)],
        grid_resolution=0.1,
        interest_config={
            "enabled": True,
            "offsets": {
                "ee": [0.0, 0.0, 0.0],
                "object": [1.0, 0.0, 0.0],
            },
            "labels": ["ee", "object"],
            "rho": 1.0,
        })

    grid = planner.build_grid(np.array([0.0, 0.0, 0.0]))

    assert grid["phi"][0, 0] == 0.0
    assert bool(grid["forbidden"][0, 0]) is True
    assert bool(grid["safe"][0, 0]) is False
    assert planner.last_debug["num_forbidden_cells"] > 0


def test_arm_mpc_limits_step_when_predicted_interest_point_crosses_rho():
    mpc = ArmMPC(n_joints=3, dq_max=2.0, v_cap=2.0)
    J = np.eye(3)

    dq = mpc.solve(
        J, np.array([1.0, 0.0, 0.0]), ee_pos=np.zeros(3), dt=1.0,
        field=StepField(0.6),
        interest_constraints={
            "enabled": True,
            "offsets": {"ee": [0.0, 0.0, 0.0]},
            "labels": ["ee"],
            "rho": 1.0,
        })

    assert dq[0] <= 0.6
    assert dq[0] > 0.0
    assert np.allclose(dq[1:], np.zeros(2))
    assert mpc.last_solver_status.endswith("_ip_limited")


def test_shortcut_trajectory_reduces_safe_redundant_handover_path():
    path = np.array([
        [0.0, 0.0, 0.0],
        [0.2, 0.2, 0.0],
        [0.4, -0.2, 0.0],
        [0.6, 0.0, 0.0],
    ], float)
    corridor = Corridor(np.array([
        [0.0, 0.0, 0.0],
        [0.6, 0.0, 0.0],
    ], float), radius=0.5, label="wide")

    shortened = shortcut_trajectory(
        path, ZeroField(), corridor=corridor, rho=1.0,
        samples=5, max_passes=2)

    assert np.allclose(shortened[0], path[0])
    assert np.allclose(shortened[-1], path[-1])
    assert len(shortened) < len(path)
    assert path_length(shortened) < path_length(path)


def test_wheelchair_predictive_solver_rejects_unsafe_interest_candidates():
    mpc = WheelchairMPC(horizon=3, dt=0.2, v_max=0.75)
    x0 = np.array([0.0, 0.0, 0.0])
    ref = np.array([[0.2, 0.0], [0.4, 0.0], [0.6, 0.0]])

    v, _w = mpc._sampled_predictive_solve(
        x0, ref, StepField(0.7), None, np.zeros(2),
        None, None, 0.0, ref[-1], {}, {},
        interest_constraints={
            "enabled": True,
            "local_points": {"front": [0.5, 0.0]},
            "labels": ["front"],
            "rho": 1.0,
        })

    x = x0.copy()
    for _ in range(mpc.N):
        x = mpc._step(x, np.array([v, _w]))
        assert x[0] + 0.5 <= 0.7


def test_wheelchair_predictive_solver_rejects_forbidden_interest_hit():
    mpc = WheelchairMPC(horizon=3, dt=0.2, v_max=0.75)
    x0 = np.array([0.0, 0.0, 0.0])
    ref = np.array([[0.2, 0.0], [0.4, 0.0], [0.6, 0.0]])
    field = ZeroField([ForbiddenXAnchor(0.62)])

    v, _w = mpc._sampled_predictive_solve(
        x0, ref, field, None, np.zeros(2),
        None, None, 0.0, ref[-1], {}, {},
        interest_constraints={
            "enabled": True,
            "local_points": {"front": [0.5, 0.0]},
            "labels": ["front"],
            "rho": 100.0,
        })

    x = x0.copy()
    for _ in range(mpc.N):
        x = mpc._step(x, np.array([v, _w]))
        assert x[0] + 0.5 < 0.62
    assert mpc.last_reject_forbidden_count > 0
    assert mpc.first_predicted_forbidden_reason == (
        "footprint:forbidden_zone:front:table")


def test_wheelchair_predictive_solver_speed_reward_improves_forward_motion():
    x0 = np.array([0.0, 0.0, 0.0])
    ref = np.array([[0.2, 0.0], [0.4, 0.0], [0.6, 0.0]])

    slow = WheelchairMPC(horizon=3, dt=0.2, v_max=0.75, lam_u=5.0)
    slow.lam_track = 0.0
    slow.lam_social = 0.0
    slow.lam_progress = 0.0
    slow.lam_ref_progress = 0.0
    slow.lam_goal_terminal = 0.0
    slow.lam_speed = 0.0
    slow.min_progress_per_solve = 0.0
    v_slow, _ = slow._sampled_predictive_solve(
        x0, ref, ZeroField(), None, np.zeros(2),
        None, None, 0.0, ref[-1], {}, {})

    fast = WheelchairMPC(horizon=3, dt=0.2, v_max=0.75, lam_u=5.0)
    fast.lam_track = 0.0
    fast.lam_social = 0.0
    fast.lam_progress = 0.0
    fast.lam_ref_progress = 0.0
    fast.lam_goal_terminal = 0.0
    fast.lam_speed = 5.0
    fast.min_progress_per_solve = 0.0
    v_fast, _ = fast._sampled_predictive_solve(
        x0, ref, ZeroField(), None, np.zeros(2),
        None, None, 0.0, ref[-1], {}, {})

    assert v_slow == 0.0
    assert v_fast > v_slow

    fast.final_heading_threshold = 0.5
    rotate = fast._goal_seek_u(
        np.array([0.0, 0.0, np.pi]), np.array([0.2, 0.0]))
    approach = fast._goal_seek_u(
        np.array([0.0, 0.0, 0.0]), np.array([0.2, 0.0]))
    closer = fast._goal_seek_u(
        np.array([0.0, 0.0, 0.0]), np.array([0.1, 0.0]))
    assert rotate[0] == 0.0
    assert 0.0 < closer[0] < approach[0]


def test_wheelchair_stsm_mpc_optimizes_time_varying_control_sequence():
    mpc = WheelchairMPC(
        horizon=4, dt=0.2, v_max=0.75, w_max=1.0,
        a_max=0.5, alpha_max=1.5, beam_width=20)
    mpc.lam_progress = 8.0
    mpc.lam_speed = 2.0
    mpc.min_progress_per_solve = 0.0
    x0 = np.array([0.0, 0.0, 0.0])
    ref = np.array([
        [0.1, 0.0], [0.25, 0.02], [0.4, 0.08], [0.55, 0.18]])
    corridor = Corridor(np.array([
        [0.0, 0.0, 0.0], [0.55, 0.18, 0.0]]), radius=0.5,
        label="predictive")

    v, w = mpc.solve(
        x0, ref, ZeroField(), corridor=corridor, u_prev=np.zeros(2),
        goal=ref[-1], predictive=True,
        topology_constraint={
            "tube_constraint_mode": "hard",
            "corridor_constraint": {
                "used": True,
                "tube_constraint": {"mode": "hard"},
            },
        })

    controls = np.asarray(mpc.last_predicted_controls, float)
    states = np.asarray(mpc.last_predicted_states, float)
    assert mpc.last_solver_status.startswith("predictive_beam")
    assert controls.shape == (4, 2)
    assert states.shape == (4, 3)
    assert np.allclose([v, w], controls[0])
    assert mpc.last_control_sequence_varies is True
    assert np.all(controls[:, 0] >= 0.0)
    assert np.all(controls[:, 0] <= mpc.v_max + 1e-9)
    assert np.all(np.abs(controls[:, 1]) <= mpc.w_max + 1e-9)
    prior = np.vstack([np.zeros((1, 2)), controls[:-1]])
    delta = np.abs(controls - prior)
    assert np.all(delta[:, 0] <= mpc.a_max * mpc.dt + 1e-9)
    assert np.all(delta[:, 1] <= mpc.alpha_max * mpc.dt + 1e-9)
    assert "tracking" in mpc.last_objective_terms
    assert "terminal_goal" in mpc.last_objective_terms


def test_wheelchair_stsm_mpc_hard_tube_infeasible_is_safe_stop():
    mpc = WheelchairMPC(horizon=3, dt=0.2, v_max=0.75)
    mpc.min_progress_per_solve = 0.0
    x0 = np.array([0.0, 0.0, 0.0])
    ref = np.array([[0.1, 0.0], [0.2, 0.0], [0.3, 0.0]])
    corridor = Corridor(np.array([
        [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]), radius=0.01,
        label="disconnected")

    control = mpc.solve(
        x0, ref, ZeroField(), corridor=corridor, u_prev=np.zeros(2),
        goal=ref[-1], predictive=True,
        topology_constraint={
            "tube_constraint_mode": "hard",
            "corridor_constraint": {
                "used": True,
                "tube_constraint": {"mode": "hard"},
            },
        })

    assert control == (0.0, 0.0)
    assert mpc.last_solver_status == "safe_stop: no_feasible_sequence"
    assert mpc.last_constraint_violation["trajectory_tube"] > 0


def test_wheelchair_baseline_keeps_pure_pursuit_outside_stsm_optimizer():
    mpc = WheelchairMPC(horizon=3, dt=0.2, v_max=0.75)
    ref = np.array([[0.1, 0.0], [0.2, 0.0], [0.3, 0.0]])

    control = mpc.solve(
        np.array([0.0, 0.0, 0.0]), ref, ZeroField(),
        u_prev=np.zeros(2), goal=ref[-1], predictive=False)

    assert control[0] >= 0.0
    assert mpc.last_solver_status == "baseline_pure_pursuit"
    assert len(mpc.last_predicted_controls) == 1


def test_arm_mpc_evidence_excludes_out_of_scope_return_from_corridor_reference():
    path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "nodes", "handover_node.py"))
    with open(path, "r") as handle:
        source = handle.read()

    assert '"corridor_active": bool(self.corridor_evaluation_active)' in source
    assert 'if bool(row.get("corridor_active", True))' in source
    retreat = source[source.index(
        'rospy.loginfo("[handover] phase 3: retreat after handover")'):]
    retreat = retreat[:retreat.index("self.task_completed")]
    assert "reference_corridor=self.execution_corridor" not in retreat
