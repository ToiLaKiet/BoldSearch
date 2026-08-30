"""Collapse per-query scores into group-level scores — the pipeline's aggregate stage.

Not a metric: everything here consumes results from ``recall``/``mrr``.
This module will later also hold the macro-average per task type and the
``n_missing`` counter (queries lacking GT are excluded from scoring but
must surface in the report).
"""
from __future__ import annotations

from typing import Sequence


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean; empty input scores 0.0."""
    return sum(values) / len(values) if values else 0.0
