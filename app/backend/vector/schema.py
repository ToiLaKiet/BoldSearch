"""
Pydantic schemas for the Vector module.

Both routes are vector-in: queries and ingested points carry precomputed
vectors. Encoding is not part of this surface, so no route names a model.
"""

from __future__ import annotations

from typing import Any, Dict, List
from uuid import UUID

from pydantic import BaseModel, Field


# ── Requests ─────────────────────────────────────────────────────────


class VectorPointPayload(BaseModel):
    """One precomputed point to ingest."""

    id: UUID = Field(description="Stable point identity; re-sending it upserts.")
    source_id: str = Field(
        min_length=1,
        description="What this vector was computed from, e.g. an image path.",
    )
    vector: List[float] = Field(min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    """Body for POST /api/vector/ingest"""

    points: List[VectorPointPayload] = Field(min_length=1)


class SearchSimilarityRequest(BaseModel):
    """Body for POST /api/vector/search-similarity"""

    query_vector: List[float] = Field(min_length=1)
    top_k: int = Field(default=20, ge=1, le=200)


# ── Responses ────────────────────────────────────────────────────────


class IngestResponse(BaseModel):
    """How many points the store accepted."""

    ingested: int


class SimilarityMatch(BaseModel):
    """One neighbour, with ``score`` normalized so higher is more similar."""

    id: UUID
    source_id: str
    score: float
    metadata: Dict[str, Any]


class SearchSimilarityResponse(BaseModel):
    """Neighbours ordered by relevance."""

    top_k: int
    matches: List[SimilarityMatch]
