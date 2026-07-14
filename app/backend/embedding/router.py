"""
Embedding router — CLIP/FG-CLIP encoding, Milvus indexing, and similarity search.

Prefix: /api/embedding
"""

from __future__ import annotations

from fastapi import APIRouter

from embedding import schema

router = APIRouter(prefix="/embedding", tags=["embedding"])


@router.post("/encode-image", response_model=schema.EmbeddingResult)
async def encode_image(body: schema.EncodeImageRequest):
    """
    Encode an image or video keyframe into an embedding vector.

    TODO: Integrate FG-CLIP / CLIP model here (Task #9 — Phú).
    """
    return schema.EmbeddingResult(
        vector=[],
        model=body.model,
        dimension=0,
    )


@router.post("/encode-text", response_model=schema.EmbeddingResult)
async def encode_text(body: schema.EncodeTextRequest):
    """
    Encode a text query into an embedding vector.

    TODO: Integrate FG-CLIP / CLIP text encoder here.
    """
    return schema.EmbeddingResult(
        vector=[],
        model=body.model,
        dimension=0,
    )


@router.post("/search", response_model=schema.SimilaritySearchResponse)
async def similarity_search(body: schema.SimilaritySearchRequest):
    """
    Search the Milvus vector index for the most similar keyframes.

    TODO: Connect to Milvus and perform ANN search.
    """
    return schema.SimilaritySearchResponse(
        query_model=body.model,
        top_k=body.top_k,
        matches=[],
    )


@router.post("/index", response_model=schema.IndexResponse)
async def index_video(body: schema.IndexRequest):
    """
    Index all keyframes of a video into Milvus.

    Includes optional deduplication of near-duplicate keyframes (Task #10 — Long).

    TODO: Implement keyframe extraction → dedup → encoding → Milvus upsert.
    """
    return schema.IndexResponse(
        video_id=body.video_id,
        frames_indexed=0,
        frames_deduplicated=0,
        model=body.model,
        status="not_implemented",
    )
