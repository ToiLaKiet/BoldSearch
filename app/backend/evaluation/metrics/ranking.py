from __future__ import annotations

from math import log2
from typing import Sequence


def normalize_k_values(k_values: Sequence[int]) -> list[int]:
    """Validate and sort unique positive integer k values."""
    if not k_values or any(type(k) is not int or k < 1 for k in k_values):
        raise ValueError("k values must contain positive integers")
    return sorted(set(k_values))


def first_relevant_rank(ranked_relevances: Sequence[int]) -> int | None:
    """Return the one-based rank of the first relevant candidate."""
    return next(
        (rank for rank, relevance in enumerate(ranked_relevances, start=1) if relevance > 0),
        None,
    )


def recall_at_k(ranked_relevances: Sequence[int], k: int) -> float:
    """Return task-success Recall@K for a ranked candidate list."""
    return float(any(relevance > 0 for relevance in ranked_relevances[:k]))


def reciprocal_rank(ranked_relevances: Sequence[int]) -> float:
    """Return reciprocal rank for the first relevant candidate, or zero."""
    rank = first_relevant_rank(ranked_relevances)
    return 1 / rank if rank else 0.0


def ndcg_at_k(
    ranked_relevances: Sequence[int],
    ideal_relevances: Sequence[int],
    k: int,
) -> float:
    """Return normalized discounted cumulative gain at K."""
    actual_dcg = _dcg(ranked_relevances[:k])
    ideal_dcg = _dcg(sorted(ideal_relevances, reverse=True)[:k])
    return actual_dcg / ideal_dcg if ideal_dcg else 0.0


def _dcg(relevance_values: Sequence[int]) -> float:
    return sum(
        (2**relevance - 1) / log2(rank + 1)
        for rank, relevance in enumerate(relevance_values, start=1)
    )
