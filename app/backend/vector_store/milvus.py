"""Milvus vector-search and ingestion adapters."""

from __future__ import annotations

from typing import Sequence
from uuid import UUID

from pymilvus import MilvusClient

from .schemas import SearchHit, VectorPoint


class MilvusStore:
    """Search and ingest vectors through one Milvus client."""

    def __init__(self, client: MilvusClient, collection: str) -> None:
        self._client = client
        self._collection = collection
        self._loaded = False

    def search(
        self,
        vector: Sequence[float],
        limit: int,
    ) -> list[SearchHit]:
        """Return nearest vectors ordered by relevance."""
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError(f"limit must be a positive int, got {limit!r}")

        self._ensure_loaded()
        rows = self._client.search(
            collection_name=self._collection,
            data=[list(vector)],
            limit=limit,
            output_fields=["id", "source_id", "metadata"],
        )[0]

        return [
            SearchHit(
                id=UUID(row["entity"]["id"]),
                source_id=row["entity"]["source_id"],
                score=float(row["distance"]),
                metadata=row["entity"]["metadata"],
            )
            for row in rows
        ]

    def _ensure_loaded(self) -> None:
        """Load the collection once before its first search."""
        if self._loaded:
            return
        self._client.load_collection(self._collection)
        self._loaded = True

    def ingest(self, points: Sequence[VectorPoint]) -> None:
        """Upsert points and flush for visibility under default consistency."""
        if not points:
            return

        self._client.upsert(
            collection_name=self._collection,
            data=[
                {
                    "id": str(point.id),
                    "vector": list(point.vector),
                    "source_id": point.source_id,
                    "metadata": dict(point.metadata),
                }
                for point in points
            ],
        )
        self._client.flush(self._collection)
