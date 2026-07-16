"""Provider-neutral vector-store data shapes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from uuid import UUID

@dataclass(frozen=True)
class VectorPoint:
    """One precomputed vector to ingest."""

    id: UUID
    source_id: str
    vector: Sequence[float]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class SearchHit:
    """One search result, with ``score`` normalized so higher is more similar."""

    id: UUID
    source_id: str
    score: float
    metadata: Mapping[str, Any]
