"""
Pydantic schemas for the ASR module.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class AsrRequest(BaseModel):
    """Body for POST /api/asr/transcribe"""

    audio_url: Optional[str] = Field(
        default=None,
        description="URL of the audio file to transcribe.",
    )
    video_id: Optional[str] = Field(
        default=None,
        description="Video ID — extract and transcribe the audio track.",
    )
    language: str = Field(
        default="vi",
        description="Language hint (ISO 639-1). Default: Vietnamese.",
    )


class TranscriptSegment(BaseModel):
    """A single timed transcript segment."""

    start: float = Field(description="Start time in seconds.")
    end: float = Field(description="End time in seconds.")
    text: str
    confidence: float = Field(default=0.0, ge=0, le=1)


class AsrResponse(BaseModel):
    """Response from the ASR transcription endpoint."""

    video_id: Optional[str] = None
    language: str = ""
    segments: List[TranscriptSegment] = Field(default_factory=list)
    full_transcript: str = Field(
        default="",
        description="All segments concatenated.",
    )
