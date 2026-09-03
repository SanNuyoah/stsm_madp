"""Composable role bindings for the topology planning pipeline.

This keeps the large legacy planner compatible while making its major
responsibilities explicit and replaceable one at a time.
"""
import stsm_madp.candidate_ranker as candidate_ranker


class TopologyPlanningComponents:
    """Python 2/3-compatible immutable-by-convention role bindings."""

    def __init__(self, candidate_generator, candidate_validator,
                 candidate_ranker, corridor_manager, recovery_manager):
        self.candidate_generator = candidate_generator
        self.candidate_validator = candidate_validator
        self.candidate_ranker = candidate_ranker
        self.corridor_manager = corridor_manager
        self.recovery_manager = recovery_manager


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
