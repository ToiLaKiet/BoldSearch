"""Data models and API schemas for the ASR module."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AsrRequest(BaseModel):
    """Body for POST /api/asr/transcribe"""

    model_config = ConfigDict(extra="forbid")

    video_id: str = Field(
        min_length=1,
        description="Video ID — extract and transcribe the audio track.",
    )
    language: str = Field(
        default="vi",
        description="Language hint (ISO 639-1). Default: Vietnamese.",
    )
    keyframes: list[KeyframeRecord] = Field(default_factory=list)


class TranscriptSegment(BaseModel):
    """One timed span of recognised speech, in seconds."""

    model_config = ConfigDict(frozen=True)

    video_id: str
    start: float = Field(description="Start time in seconds.")
    end: float = Field(description="End time in seconds.")
    text: str
    confidence: float | None = Field(default=None, ge=0, le=1)


class TranscriptResult(BaseModel):
    """Normalised output of one transcription call."""

    model_config = ConfigDict(frozen=True)

    segments: tuple[TranscriptSegment, ...]
    language: str = "vi"
    text: str = ""


class KeyframeRecord(BaseModel):
    """A keyframe awaiting ASR text."""

    model_config = ConfigDict(frozen=True, extra="allow")

    video_id: str
    frame_id: str
    timestamp: float = Field(description="Time from the start of the video, in seconds.")
    img_path: str = Field(min_length=1, description="Path to the extracted keyframe image.")


class KeyframeWithText(BaseModel):
    """A keyframe and the text of its containing ASR segment."""

    model_config = ConfigDict(frozen=True, extra="allow")

    video_id: str
    frame_id: str
    timestamp: float = Field(description="Time from the start of the video, in seconds.")
    img_path: str = Field(min_length=1, description="Path to the extracted keyframe image.")
    text: str | None = None


class AsrResponse(BaseModel):
    """Response from the ASR transcription endpoint."""

    video_id: str | None = None
    language: str = ""
    segments: list[TranscriptSegment] = Field(default_factory=list)
    keyframes: list[KeyframeWithText] = Field(default_factory=list)
    full_transcript: str = Field(
        default="",
        description="All segments concatenated.",
    )
