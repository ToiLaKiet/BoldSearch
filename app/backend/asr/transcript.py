"""Attach timed transcript segments to keyframes."""

from __future__ import annotations

import math
from typing import Sequence

from asr.schema import KeyframeRecord, TranscriptSegment


def attach_transcripts_to_keyframes(
    keyframes: Sequence[KeyframeRecord],
    segments: Sequence[TranscriptSegment],
    *,
    video_id: str,
) -> list[dict]:
    """Attach the containing segment's text to each keyframe."""
    _validate_keyframes(keyframes, video_id)
    _validate_segments(segments, video_id)

    ordered = sorted(segments, key=lambda segment: (segment.start, segment.end))
    records: list[dict] = []

    for keyframe in keyframes:
        matched = next(
            (s for s in ordered if s.start <= keyframe.timestamp < s.end),
            None,
        )
        records.append(
            {
                **keyframe.model_dump(),
                "text": matched.text if matched else None,
            }
        )

    return records


def _validate_keyframes(keyframes: Sequence[KeyframeRecord], video_id: str) -> None:
    seen: set[str] = set()
    for position, keyframe in enumerate(keyframes):
        if keyframe.video_id != video_id or not keyframe.frame_id:
            raise ValueError(
                f"Keyframe {position}: expected video_id={video_id!r}, got "
                f"video_id={keyframe.video_id!r}, frame_id={keyframe.frame_id!r}",
            )
        if not math.isfinite(keyframe.timestamp) or keyframe.timestamp < 0:
            raise ValueError(
                f"Keyframe {position}: timestamp must be finite and >= 0",
            )
        if keyframe.frame_id in seen:
            raise ValueError(
                f"Duplicate frame_id: {(video_id, keyframe.frame_id)}",
            )
        seen.add(keyframe.frame_id)


def _validate_segments(
    segments: Sequence[TranscriptSegment], video_id: str
) -> None:
    for position, segment in enumerate(segments):
        if segment.video_id != video_id:
            raise ValueError(
                f"Segment {position}: expected video_id={video_id!r}, "
                f"got {segment.video_id!r}",
            )
        if (
            not math.isfinite(segment.start)
            or not math.isfinite(segment.end)
            or segment.start < 0
            or segment.start > segment.end
            or not isinstance(segment.text, str)
        ):
            raise ValueError(f"Segment {position}: invalid start/end/text")
