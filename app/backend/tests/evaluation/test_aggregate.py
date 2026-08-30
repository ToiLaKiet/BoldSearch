"""Spec for the aggregate stage — see the contract in evaluation/metrics/aggregate.py."""
from __future__ import annotations

import pytest

from evaluation.metrics.aggregate import mean


def test_mean_scores_queries() -> None:
    assert mean([0.5, 0.25]) == pytest.approx(0.375)
    assert mean([]) == 0.0
