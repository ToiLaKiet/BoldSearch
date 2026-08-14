"""
Schemas for frame retrieval backed by Zilliz hybrid search.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ObjectQuery(BaseModel):
    """A free-form object query and expected quantity."""

    query: str = ""
    count: int = Field(default=1, ge=1)


class FrameContext(BaseModel):
    """A keyframe from the current UI result set used to scope staged search."""

    path: str = ""
    video_id: Optional[str] = None
    frame_id: Optional[Union[str, int]] = None
    shot_id: Optional[Union[str, int]] = None
    score: Optional[float] = None


class QueryRequest(BaseModel):
    """Body for POST /api/search/query."""

    query: str = ""
    queries: List[str] = Field(default_factory=list)
    task: str = "KIS"
    modalities: List[str] = Field(default_factory=list)
    objects: List[str] = Field(default_factory=list)
    objectQueries: List[ObjectQuery] = Field(default_factory=list)
    minConfidence: float = Field(default=0, ge=0, le=1)
    topK: Optional[int] = Field(default=None, ge=1)
    frames_path: Optional[List[str]] = None
    used_queries: List[str] = Field(default_factory=list)
    frames_context: List[FrameContext] = Field(default_factory=list)


class VisualQueryRequest(BaseModel):
    """Body for POST /api/search/visual_query."""

    task: str = "VKIS"
    imageCue: Dict[str, Any] = Field(default_factory=dict)
    imageEmbedding: Optional[List[float]] = None
    minConfidence: float = Field(default=0, ge=0, le=1)
    topK: Optional[int] = Field(default=None, ge=1)


class SubmitRequest(BaseModel):
    """Body for POST /api/search/submit/kis — Known Item Search."""

    id: Optional[Union[str, int]] = None
    frameId: Optional[Union[str, int]] = None
    frame_id: Optional[Union[str, int]] = None
    shotId: Optional[Union[str, int]] = None
    shot_id: Optional[Union[str, int]] = None
    videoId: Optional[str] = None
    video_id: Optional[str] = None
    task: str = "KIS"


class VQASubmitRequest(BaseModel):
    """Body for POST /api/search/submit/vqa — Visual Question Answering."""

    video_id: str
    frame_id: Union[str, int]
    answer: str
    task: str = "VQA"


class TrakeSubmitRequest(BaseModel):
    """Body for POST /api/search/submit/trake — Temporal Retrieval and Key-event."""

    video_id: str
    frame_ids: List[Union[str, int]]
    task: str = "TRAKE"


class SubmitResponse(BaseModel):
    status: str
    system: str
    submission: Dict[str, Any]
    message: Optional[str] = None


class VQASubmitResponse(BaseModel):
    status: str
    system: str
    submission: Dict[str, Any]
    message: Optional[str] = None


class TrakeSubmitResponse(BaseModel):
    status: str
    system: str
    submission: Dict[str, Any]
    message: Optional[str] = None


class ObjectMatch(BaseModel):
    query: str
    requested: int
    matched: int


class FrameResult(BaseModel):
    id: Union[str, int]
    frame_id: Optional[Union[str, int]] = None
    shot_id: Optional[Union[str, int]] = None
    video_id: Optional[str] = None
    distance: Optional[float] = None
    score: float
    asr_text: str = ""
    ocr_text: str = ""
    transcript: str = ""
    objects: List[str] = Field(default_factory=list)
    objectMatches: List[ObjectMatch] = Field(default_factory=list)
    objectMetadata: Dict[str, Any] = Field(default_factory=dict)
    object_count_delta: Optional[float] = None
    reasons: List[str] = Field(default_factory=list)

    # Compatibility fields for the current frontend.
    videoId: str = ""
    shotId: str = ""
    title: str = ""
    description: str = ""
    thumbnail: str = ""
    videoUrl: str = ""
    start: str = ""
    end: str = ""
    duration: Union[int, float] = 0

    embedding: Optional[List[float]] = None
    raw: Dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    system: str
    task: str
    query: str
    count: int
    cacheHitRatio: Optional[float] = None
    cost: Optional[Union[int, float]] = None
    topks: List[int] = Field(default_factory=list)
    results: List[FrameResult]
