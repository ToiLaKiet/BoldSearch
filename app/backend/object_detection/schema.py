"""
Pydantic schemas for the Object Detection module.

Output format theo gợi ý Task #6 (Nguyên):
  { object: 'car', color: 'red', quantity: 2 }
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class DetectionRequest(BaseModel):
    """Body for POST /api/object-detection/detect"""

    image_url: Optional[str] = Field(
        default=None,
        description="URL of the frame/image to run detection on.",
    )
    video_id: Optional[str] = Field(
        default=None,
        description="Video ID — detect objects in all keyframes.",
    )
    frame_index: Optional[int] = Field(
        default=None,
        description="Specific frame index within the video.",
    )


class DetectedObject(BaseModel):
    """A single detected object with its dominant color."""

    object: str = Field(description="Object class label (e.g. 'car', 'person').")
    color: str = Field(
        default="",
        description="Dominant color of the detected object.",
    )
    quantity: int = Field(
        default=1,
        description="Number of instances of this object-color combination.",
    )
    confidence: float = Field(default=0.0, ge=0, le=1)
    bbox: List[float] = Field(
        default_factory=list,
        description="Bounding box [x1, y1, x2, y2] (normalized 0-1).",
    )


class DetectionResponse(BaseModel):
    """Response from the object detection endpoint."""

    video_id: Optional[str] = None
    frame_index: Optional[int] = None
    objects: List[DetectedObject] = Field(default_factory=list)
    summary: List[Dict] = Field(
        default_factory=list,
        description=(
            "Aggregated summary: [{object, color, quantity}, ...] "
            "as requested in Task #6."
        ),
    )
