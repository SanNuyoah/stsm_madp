"""Stable ranking boundary for topology candidates.

The implementation remains in ``topology_candidate_generator`` for
backwards compatibility. Planners import ranking through this role-specific
module so graph generation, validation, and ranking can be split safely.
"""

from stsm_madp.topology_candidate_generator import (
    candidate_decision_record,
    candidate_topology_identity,
    rank_feasible_candidates,
)

__all__ = ["candidate_decision_record", "candidate_topology_identity",
           "rank_feasible_candidates"]
