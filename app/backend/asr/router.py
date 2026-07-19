"""HTTP endpoint for ASR transcription."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

from asr import schema
from asr.audio import create_temp_audio_path, normalize_audio
from asr.transcript import attach_transcripts_to_keyframes

router = APIRouter(prefix="/asr", tags=["asr"])
logger = logging.getLogger(__name__)


def _get_transcriber(request: Request):
    transcriber = getattr(request.app.state, "asr_transcriber", None)
    if transcriber is None:
        raise HTTPException(status_code=503, detail="ASR_INFERENCE_FAILED")
    return transcriber


def _get_media_resolver(request: Request):
    return getattr(request.app.state, "asr_media_resolver", None)


@router.post("/transcribe", response_model=schema.AsrResponse)
def transcribe(
    body: schema.AsrRequest,
    transcriber=Depends(_get_transcriber),
    media_resolver=Depends(_get_media_resolver),
):
    """Transcribe the Vietnamese audio track of one video."""
    if body.language and body.language != "vi":
        raise HTTPException(status_code=422, detail="ASR_LANGUAGE_UNSUPPORTED")

    if media_resolver is None:
        raise HTTPException(status_code=503, detail="MEDIA_RESOLVER_UNAVAILABLE")

    temp_audio_path: Path | None = None
    try:
        try:
            media_path = Path(media_resolver.resolve(body.video_id))
        except FileNotFoundError as exc:
            logger.exception("MEDIA_NOT_FOUND for video_id=%s", body.video_id)
            raise HTTPException(status_code=404, detail="MEDIA_NOT_FOUND") from exc

        temp_audio_path = create_temp_audio_path()
        try:
            normalized_path = normalize_audio(media_path, temp_audio_path)
        except RuntimeError as exc:
            logger.exception("AUDIO_NORMALIZATION_FAILED for video_id=%s", body.video_id)
            raise HTTPException(
                status_code=422, detail="AUDIO_NORMALIZATION_FAILED"
            ) from exc

        try:
            result = transcriber.transcribe(normalized_path, video_id=body.video_id)
        except ValueError as exc:
            logger.exception("ASR_RESULT_INVALID for video_id=%s", body.video_id)
            raise HTTPException(status_code=502, detail="ASR_RESULT_INVALID") from exc
        except RuntimeError as exc:
            logger.exception("ASR_INFERENCE_FAILED for video_id=%s", body.video_id)
            raise HTTPException(status_code=503, detail="ASR_INFERENCE_FAILED") from exc

        try:
            keyframes = attach_transcripts_to_keyframes(
                body.keyframes,
                result.segments,
                video_id=body.video_id,
            )
        except ValueError as exc:
            logger.exception("KEYFRAME_SCHEMA_INVALID for video_id=%s", body.video_id)
            raise HTTPException(status_code=422, detail="KEYFRAME_SCHEMA_INVALID") from exc
    finally:
        if temp_audio_path is not None:
            temp_audio_path.unlink(missing_ok=True)

    return schema.AsrResponse(
        video_id=body.video_id,
        language=result.language,
        segments=list(result.segments),
        keyframes=keyframes,
        full_transcript=result.text,
    )
