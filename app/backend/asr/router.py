"""
ASR router — speech-to-text transcription.

Prefix: /api/asr
"""

from __future__ import annotations

from fastapi import APIRouter

from asr import schema

router = APIRouter(prefix="/asr", tags=["asr"])


@router.post("/transcribe", response_model=schema.AsrResponse)
async def transcribe(body: schema.AsrRequest):
    """
    Transcribe audio from a video or audio file.

    TODO: Integrate ASR model (e.g. Whisper, wav2vec2) here.
    """
    # ── Placeholder response ─────────────────────────────────────
    return schema.AsrResponse(
        video_id=body.video_id,
        language=body.language,
        segments=[],
        full_transcript="",
    )
