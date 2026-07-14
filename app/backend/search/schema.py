"""
Pydantic schemas for the search module.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


# ── Request Schemas ──────────────────────────────────────────────────


class SearchRequest(BaseModel):
    """Body for POST /api/search/query"""

    query: str = ""
    task: str = Field(default="KIS", description="KIS or VKIS")
    modalities: List[str] = Field(default_factory=list)
    objects: List[str] = Field(default_factory=list)
    colors: List[str] = Field(default_factory=list)
    temporal: str = ""
    minConfidence: float = Field(default=0, ge=0, le=1)
    imageCue: Optional[Dict] = None


class SubmitRequest(BaseModel):
    """Body for POST /api/search/submit"""

    shotId: str
    task: str = "KIS"


# ── Response Schemas ─────────────────────────────────────────────────


class ShotBase(BaseModel):
    """Core shot fields returned from the data store."""

    id: str
    videoId: str
    shotId: str
    title: str
    description: str
    thumbnail: str
    videoUrl: str
    start: str
    end: str
    duration: Union[int, float]
    confidence: float
    location: str
    objects: List[str]
    colors: List[str]
    tags: List[str]
    transcript: str


class ShotResult(ShotBase):
    """A shot enriched with a relevance score and match reasons."""

    score: float = 0.0
    reasons: List[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    system: str
    task: str
    query: str
    count: int
    results: List[ShotResult]


class ShotsResponse(BaseModel):
    shots: List[ShotBase]


class TaskInfo(BaseModel):
    id: str
    name: str
    description: str
    recommendedSignals: List[str]


class TasksResponse(BaseModel):
    system: str
    tasks: List[TaskInfo]


class SubmissionDetail(BaseModel):
    shotId: str
    videoId: str
    timestamp: str


class SubmitResponse(BaseModel):
    status: str
    system: str
    submission: Optional[SubmissionDetail] = None
    message: Optional[str] = None
