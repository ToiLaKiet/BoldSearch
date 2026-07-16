"""Qdrant vector-search and ingestion adapters."""

from __future__ import annotations

from typing import Sequence
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from .schemas import SearchHit, VectorPoint


class QdrantStore:
    """Search and ingest vectors through one Qdrant client."""

    def __init__(self, client: QdrantClient, collection: str) -> None:
        self._client = client
        self._collection = collection

    def search(
        self,
        vector: Sequence[float],
        limit: int,
    ) -> list[SearchHit]:
        """Return nearest vectors ordered by relevance."""
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError(f"limit must be a positive int, got {limit!r}")

        points = self._client.query_points(
            collection_name=self._collection,
            query=list(vector),
            limit=limit,
        ).points

        return [
            SearchHit(
                id=UUID(str(point.id)),
                source_id=point.payload["source_id"],
                score=float(point.score),
                metadata=point.payload["metadata"],
            )
            for point in points
        ]

    def ingest(self, points: Sequence[VectorPoint]) -> None:
        """Insert or replace points and make them searchable."""
        if not points:
            return

        self._client.upsert(
            collection_name=self._collection,
            points=[
                qmodels.PointStruct(
                    id=str(point.id),
                    vector=list(point.vector),
                    payload={
                        "source_id": point.source_id,
                        "metadata": dict(point.metadata),
                    },
                )
                for point in points
            ],
        )
