"""Spec for MRR — see the contract in evaluation/metrics/mrr.py."""
from __future__ import annotations

import pytest

from evaluation.metrics.mrr import first_hit_rank, reciprocal_rank


@pytest.mark.parametrize(
    ("ranked_hits", "expected"),
    [
        ([False, False, True], 3),
        ([True, True], 1),
        ([False, False], None),
        ([], None),
    ],
)
def test_first_hit_rank(ranked_hits: list[bool], expected: int | None) -> None:
    assert first_hit_rank(ranked_hits) == expected


@pytest.mark.parametrize(
    ("ranked_hits", "expected"),
    [
        ([False, False, True], 1 / 3),
        ([True, True], 1.0),
        ([False, False], 0.0),
        ([], 0.0),
    ],
)
def test_reciprocal_rank(ranked_hits: list[bool], expected: float) -> None:
    assert reciprocal_rank(ranked_hits) == pytest.approx(expected)
