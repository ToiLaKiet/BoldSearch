"""Load extracted keyframes and attach timed transcript segments to them."""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import Sequence

from asr.schema import KeyframeRecord, KeyframeWithText, TranscriptSegment


def load_keyframes(metadata_dir: str | Path, video_id: str) -> list[KeyframeRecord]:
    """Load the KEPT keyframes for one video from its ``Frame.json`` manifest.

    The extraction pipeline records every sampled frame with a millisecond
    timestamp and a KEPT/DUPLICATE status; only KEPT frames have an image on
    disk.  Timestamps are stored as-is in milliseconds and ``img_path``
    points to the extracted frame image.
    """
    manifest = Path(metadata_dir) / video_id / "Frame.json"
    frames = json.loads(manifest.read_text())["frames"]
    return [
        KeyframeRecord(
            video_id=video_id,
            frame_id=frame["frame_id"],
            timestamp_ms=frame["timestamp_ms"],
            img_path=f"data/frames/{video_id}/{frame['frame_id']}.png",
        )
        for frame in frames
        if frame["final_status"] == "KEPT"
    ]


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
            (s for s in ordered if s.start <= keyframe.timestamp_ms / 1000 < s.end),
            None,
        )
        records.append(
            {
                **keyframe.model_dump(),
                "video_id": video_id,
                "text": matched.text if matched else None,
            }
        )

    return records


def _validate_keyframes(keyframes: Sequence[KeyframeRecord], video_id: str) -> None:
    seen: set[str] = set()
    for position, keyframe in enumerate(keyframes):
        if not keyframe.frame_id:
            raise ValueError(
                f"Keyframe {position}: frame_id must not be empty",
            )
        if not isinstance(keyframe.timestamp_ms, int) or keyframe.timestamp_ms < 0:
            raise ValueError(
                f"Keyframe {position}: timestamp_ms must be a non-negative integer (ms)",
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


# ── CSV export ──────────────────────────────────────────────────────────


def segments_to_csv(
    segments: Sequence[TranscriptSegment],
) -> str:
    """Export transcript segments as a CSV string.

    Columns: ``video_id``, ``start``, ``end``, ``text``, ``confidence``.

    Each row is one timed segment of recognised speech.
    """
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["video_id", "start", "end", "text", "confidence"])
    for segment in segments:
        writer.writerow([
            segment.video_id,
            segment.start,
            segment.end,
            segment.text,
            segment.confidence if segment.confidence is not None else "",
        ])
    return output.getvalue()


def keyframes_to_csv(
    keyframes: Sequence[KeyframeWithText | dict],
) -> str:
    """Export keyframes with their attached transcript text as a CSV string.

    Columns: ``video_id``, ``frame_id``, ``timestamp_ms``, ``img_path``, ``text``.

    Accepts schema :class:`KeyframeWithText` objects *or* the raw dictionaries
    returned by :func:`attach_transcripts_to_keyframes`.
    """
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["video_id", "frame_id", "timestamp_ms", "img_path", "text"])

    for keyframe in keyframes:
        if isinstance(keyframe, dict):
            writer.writerow([
                keyframe.get("video_id", ""),
                keyframe.get("frame_id", ""),
                keyframe.get("timestamp_ms", ""),
                keyframe.get("img_path", ""),
                keyframe.get("text", ""),
            ])
        else:
            writer.writerow([
                keyframe.video_id,
                keyframe.frame_id,
                keyframe.timestamp_ms,
                keyframe.img_path,
                keyframe.text if keyframe.text is not None else "",
            ])
    return output.getvalue()
