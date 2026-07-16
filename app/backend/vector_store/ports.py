"""Provider-neutral vector-store capabilities."""

from __future__ import annotations

from typing import Protocol, Sequence

from .schemas import SearchHit, VectorPoint


class VectorStore(Protocol):
    """Search and ingest vectors without exposing provider details."""

    def search(
        self,
        vector: Sequence[float],
        limit: int,
    ) -> list[SearchHit]:
        """Return nearest vectors ordered by relevance."""
        ...

    def ingest(self, points: Sequence[VectorPoint]) -> None:
        """Insert or replace points and make them searchable."""
        ...
