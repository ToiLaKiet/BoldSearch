"""Hit@K — top-K success counting (baseline v1).

One of the two baseline v1 metrics, beside ``mrr.py``. This family will
keep growing (set-based ``recall_at_k`` once a query carries multiple GT
and coverage matters), so the module is cut per metric from the start.
"""
from __future__ import annotations

from typing import Sequence


def hit_at_k(ranked_hits: Sequence[bool], k: int) -> float:
    """Hit@K: 1.0 if any hit falls within the top-k positions.

    The name is Hit@K, NOT Recall — set-wise Recall means
    ``|topK ∩ GT| / |GT|`` and will live here once coverage matters.
    """
    return float(any(ranked_hits[:k]))
