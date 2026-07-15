"""
Pydantic schemas for the OCR module.

Model dự kiến: PaddleOCR
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class OcrRequest(BaseModel):
    """Body for POST /api/ocr/extract"""

    image_url: Optional[str] = Field(
        default=None,
        description="URL of the frame/image to run OCR on.",
    )
    video_id: Optional[str] = Field(
        default=None,
        description="Video ID — if provided, OCR all keyframes of this video.",
    )
    frame_index: Optional[int] = Field(
        default=None,
        description="Specific frame index within the video.",
    )


class OcrBox(BaseModel):
    """A single detected text region."""

    text: str
    confidence: float = Field(ge=0, le=1)
    bbox: List[List[float]] = Field(
        description="Bounding box as list of [x, y] corner points.",
    )


class OcrResponse(BaseModel):
    """Response from the OCR extraction endpoint."""

    video_id: Optional[str] = None
    frame_index: Optional[int] = None
    detections: List[OcrBox] = Field(default_factory=list)
    full_text: str = Field(
        default="",
        description="All detected text concatenated.",
    )
