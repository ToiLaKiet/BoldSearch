"""
Object Detection router — detect objects and dominant colors from frames.

Prefix: /api/object-detection
"""

from __future__ import annotations

from fastapi import APIRouter

from object_detection import schema

router = APIRouter(prefix="/object-detection", tags=["object-detection"])


@router.post("/detect", response_model=schema.DetectionResponse)
async def detect_objects(body: schema.DetectionRequest):
    """
    Detect objects and their dominant colors in a frame.

    TODO: Integrate object detection model + color extraction here.
    """
    # ── Placeholder response ─────────────────────────────────────
    return schema.DetectionResponse(
        video_id=body.video_id,
        frame_index=body.frame_index,
        objects=[],
        summary=[],
    )
