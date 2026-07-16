"""
Pydantic schemas for the Embedding module.

Models: FG-CLIP, BEiT-3
Vector store: configured provider
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ── Encoding requests ────────────────────────────────────────────────


class EncodeImageRequest(BaseModel):
    """Body for POST /api/embedding/encode-image"""

    image_url: Optional[str] = Field(
        default=None,
        description="URL of the image to encode.",
    )
    video_id: Optional[str] = Field(
        default=None,
        description="Video ID — encode all keyframes.",
    )
    frame_index: Optional[int] = None
    model: str = Field(
        default="fg-clip",
        description="Embedding model to use: 'fg-clip', 'clip', 'bert'.",
    )


class EncodeTextRequest(BaseModel):
    """Body for POST /api/embedding/encode-text"""

    text: str = Field(description="Text query to encode.")
    model: str = Field(
        default="fg-clip",
        description="Embedding model to use: 'fg-clip', 'clip', 'bert'.",
    )


# ── Similarity search ───────────────────────────────────────────────


class SimilaritySearchRequest(BaseModel):
    """Body for POST /api/embedding/search"""

    query_text: Optional[str] = Field(
        default=None,
        description="Text query — will be encoded then searched.",
    )
    query_vector: Optional[List[float]] = Field(
        default=None,
        description="Pre-computed query vector.",
    )
    top_k: int = Field(default=20, ge=1, le=200)
    model: str = "fg-clip"


# ── Index management ────────────────────────────────────────────────


class IndexRequest(BaseModel):
    """Body for POST /api/embedding/index"""

    video_id: str = Field(description="Video ID to index keyframes for.")
    model: str = "fg-clip"
    deduplicate: bool = Field(
        default=True,
        description="Remove near-duplicate keyframes before indexing (Task #10).",
    )


# ── Responses ────────────────────────────────────────────────────────


class EmbeddingResult(BaseModel):
    vector: List[float] = Field(default_factory=list)
    model: str = ""
    dimension: int = 0


class SimilarityMatch(BaseModel):
    video_id: str
    frame_index: int
    score: float
    metadata: Dict = Field(default_factory=dict)


class SimilaritySearchResponse(BaseModel):
    query_model: str
    top_k: int
    matches: List[SimilarityMatch] = Field(default_factory=list)


class IndexResponse(BaseModel):
    video_id: str
    frames_indexed: int = 0
    frames_deduplicated: int = 0
    model: str = ""
    status: str = "pending"
