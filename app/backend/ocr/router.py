"""
OCR router — text extraction from video frames.

Prefix: /api/ocr
Model: PaddleOCR (to be integrated)
"""

from __future__ import annotations

from fastapi import APIRouter

from ocr import schema

router = APIRouter(prefix="/ocr", tags=["ocr"])


@router.post("/extract", response_model=schema.OcrResponse)
async def extract_text(body: schema.OcrRequest):
    """
    Extract text from a frame or image using OCR.

    TODO: Integrate PaddleOCR model here.
    """
    # ── Placeholder response ─────────────────────────────────────
    return schema.OcrResponse(
        video_id=body.video_id,
        frame_index=body.frame_index,
        detections=[],
        full_text="",
    )
