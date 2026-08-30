"""MRR — Mean Reciprocal Rank, measuring how early the first correct evidence appears.

Built on the first matching hit: the rank of the first hit is a shared
concept, so it is exported publicly as ``first_hit_rank`` for any metric
needing the first-hit position (MRR today; nDCG can read from the same
concept tomorrow).
"""
from __future__ import annotations

from typing import Sequence


def first_hit_rank(ranked_hits: Sequence[bool]) -> int | None:
    """Rank (one-based) of the first hit; None when there is no hit."""
    return next((rank for rank, hit in enumerate(ranked_hits, start=1) if hit), None)


def reciprocal_rank(ranked_hits: Sequence[bool]) -> float:
    """RR for one query: 1/rank of the first hit; 0.0 when there is no hit.

    MRR = ``aggregate.mean`` applied to RR across queries — this module
    only owns the per-query part.
    """
    rank = first_hit_rank(ranked_hits)
    return 1.0 / rank if rank else 0.0
