"""
Vector router — ingest and similarity search over the configured store.

Prefix: /api/vector

Both routes are vector-in, so this module needs the store and nothing else:
no encoder, and therefore no orchestration between two capabilities. Whoever
produces the vectors (an offline keyframe pipeline today) owns the model.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from vector import schema
from vector_store.ports import VectorStore
from vector_store.schemas import VectorPoint

router = APIRouter(prefix="/vector", tags=["vector"])


def get_vector_store(request: Request) -> VectorStore:
    """Return the vector store opened once by the application lifespan."""
    return request.app.state.vector_store


@router.post("/ingest", response_model=schema.IngestResponse)
def ingest(
    body: schema.IngestRequest,
    store: VectorStore = Depends(get_vector_store),
):
    """Insert or replace precomputed points, keyed by their stable id."""
    points = [
        VectorPoint(
            id=point.id,
            source_id=point.source_id,
            vector=point.vector,
            metadata=point.metadata,
        )
        for point in body.points
    ]
    store.ingest(points)
    return schema.IngestResponse(ingested=len(points))


@router.post("/search-similarity", response_model=schema.SearchSimilarityResponse)
def search_similarity(
    body: schema.SearchSimilarityRequest,
    store: VectorStore = Depends(get_vector_store),
):
    """Rank stored vectors against one precomputed query vector."""
    hits = store.search(body.query_vector, limit=body.top_k)
    return schema.SearchSimilarityResponse(
        top_k=body.top_k,
        matches=[
            schema.SimilarityMatch(
                id=hit.id,
                source_id=hit.source_id,
                score=hit.score,
                metadata=dict(hit.metadata),
            )
            for hit in hits
        ],
    )
