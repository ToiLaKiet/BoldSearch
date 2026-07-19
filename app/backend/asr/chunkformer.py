"""ChunkFormer RNNT adapter for the pinned Vietnamese checkpoint."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from chunkformer import ChunkFormerModel

from asr.schema import TranscriptResult, TranscriptSegment

logger = logging.getLogger(__name__)

CHECKPOINT = "khanhld/chunkformer-rnnt-large-vie"
REVISION = "9863592e00d6c02634c4ce2db8a4f3414f1ba7d6"

CHUNK_SIZE = 64
LEFT_CONTEXT_SIZE = 128
RIGHT_CONTEXT_SIZE = 128
TOTAL_BATCH_DURATION = 1800
MAX_SILENCE_DURATION = 0.5

class ChunkFormerTranscriber:
    """Load and reuse one ChunkFormer model for Vietnamese transcription."""

    _TIMESTAMP_RE = re.compile(r"^(\d+):(\d+):(\d+)[.:](\d{3})$")

    def __init__(self, device: str = "cpu") -> None:
        logger.info("Loading ChunkFormer %s (revision=%s) on %s", CHECKPOINT, REVISION, device)
        try:
            self._model = ChunkFormerModel.from_pretrained(CHECKPOINT, revision=REVISION).to(device)
            self._model.eval()
        except Exception as exc:
            raise RuntimeError(f"Failed to load ChunkFormer model: {exc}") from exc
        logger.info("ChunkFormer ready on %s", device)

    def transcribe(self, audio_path: str | Path, *, video_id: str) -> TranscriptResult:
        """Transcribe a PCM mono 16 kHz audio file."""
        try:
            raw = self._model.endless_decode(
                audio_path=str(audio_path),
                chunk_size=CHUNK_SIZE,
                left_context_size=LEFT_CONTEXT_SIZE,
                right_context_size=RIGHT_CONTEXT_SIZE,
                total_batch_duration=TOTAL_BATCH_DURATION,
                return_timestamps=True,
                max_silence_duration=MAX_SILENCE_DURATION,
            )
        except Exception as exc:
            raise RuntimeError(f"ChunkFormer inference failed on {audio_path}: {exc}") from exc

        return self._normalize_result(raw, video_id=video_id)

    @classmethod
    def _normalize_result(cls, raw: list, *, video_id: str) -> TranscriptResult:
        if not isinstance(raw, list):
            raise ValueError(f"Expected list from endless_decode, got {type(raw).__name__}")

        segments = []
        full_text_parts = []
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError(f"Expected dict in segment list, got {type(item).__name__}")

            missing = {key for key in ("decode", "start", "end") if key not in item}
            if missing:
                raise ValueError(f"ChunkFormer segment missing fields: {sorted(missing)}")

            text = str(item["decode"]).strip()
            start = cls._parse_timestamp(str(item["start"]))
            end = cls._parse_timestamp(str(item["end"]))
            if start > end:
                raise ValueError(f"ChunkFormer segment has start={start} after end={end}")

            segments.append(
                TranscriptSegment(video_id=video_id, start=start, end=end, text=text)
            )
            if text:
                full_text_parts.append(text)

        return TranscriptResult(
            language="vi",
            text=" ".join(full_text_parts),
            segments=tuple(segments),
        )

    @classmethod
    def _parse_timestamp(cls, timestamp: str) -> float:
        match = cls._TIMESTAMP_RE.match(timestamp)
        if not match:
            raise ValueError(f"Cannot parse timestamp: {timestamp!r}")
        hours, minutes, seconds, milliseconds = map(int, match.groups())
        return float(hours * 3600 + minutes * 60 + seconds) + milliseconds / 1000.0
