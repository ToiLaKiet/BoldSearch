"""Spec for Hit@K — see the contract in evaluation/metrics/recall.py."""
from __future__ import annotations

import pytest

from evaluation.metrics.recall import hit_at_k


@pytest.mark.parametrize(
    ("ranked_hits", "k", "expected"),
    [
        ([False, False, True, True], 2, 0.0),  # hit sits past rank k
        ([False, False, True, True], 3, 1.0),  # hit lands exactly at rank k
        ([True, False, True], 1, 1.0),
        ([False, False], 10, 0.0),  # k larger than the ranking length
        ([], 5, 0.0),  # empty ranking
    ],
)
def test_hit_at_k(ranked_hits: list[bool], k: int, expected: float) -> None:
    assert hit_at_k(ranked_hits, k) == expected
