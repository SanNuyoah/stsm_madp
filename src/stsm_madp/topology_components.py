"""Composable role bindings for the topology planning pipeline.

This keeps the large legacy planner compatible while making its major
responsibilities explicit and replaceable one at a time.
"""
from dataclasses import dataclass

import stsm_madp.candidate_ranker as candidate_ranker


@dataclass(frozen=True)
class TopologyPlanningComponents:
    candidate_generator: object
    candidate_validator: object
    candidate_ranker: object
    corridor_manager: object
    recovery_manager: object


def default_topology_components():
    """Return role bindings backed by the existing implementations."""
    from stsm_madp.topology_candidate_generator import (
        TopologyDrivenCandidateGenerator,
    )
    from stsm_madp.candidate_recovery import recover_candidates
    from stsm_madp.topology_refinement import refine_topology_path
    from stsm_madp.candidate_validator import validate_candidate_execution
    return TopologyPlanningComponents(
        candidate_generator=TopologyDrivenCandidateGenerator,
        candidate_validator=validate_candidate_execution,
        candidate_ranker=candidate_ranker.rank_feasible_candidates,
        corridor_manager=refine_topology_path,
        recovery_manager=recover_candidates,
    )


__all__ = ["TopologyPlanningComponents", "default_topology_components"]
