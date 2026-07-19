"""Tests for ASR contracts, domain errors, and the timestamp-containment join."""

import pytest
from pydantic import ValidationError

from asr.schema import KeyframeRecord, TranscriptResult, TranscriptSegment
from asr.transcript import (
    attach_transcripts_to_keyframes,
)


def _frame(frame_id: str, timestamp: float, video_id: str = "v1", **extra: object) -> KeyframeRecord:
    return KeyframeRecord(
        video_id=video_id,
        frame_id=frame_id,
        timestamp=timestamp,
        img_path=f"{frame_id}.png",
        **extra,
    )


# ── Contracts ──────────────────────────────────────────────────────────


def test_keyframe_contract_uses_direct_timestamp_and_frame_id():
    record = KeyframeRecord(
        video_id="v1",
        frame_id="002",
        timestamp=83.0,
        img_path="002.png",
    )
    assert record.frame_id == "002"
    assert record.timestamp == 83.0
    assert record.model_dump()["img_path"] == "002.png"


def test_segment_contract_includes_video_id_and_text():
    segment = TranscriptSegment(video_id="v1", start=60.0, end=103.0, text="xin chào")
    assert segment.model_dump() == {
        "video_id": "v1",
        "start": 60.0,
        "end": 103.0,
        "text": "xin chào",
        "confidence": None,
    }


def test_keyframe_record_is_immutable():
    record = KeyframeRecord(video_id="v1", frame_id="002", timestamp=83.0)
    with pytest.raises(ValidationError):
        record.frame_id = "003"  # type: ignore[misc]


# ── Timestamp containment ──────────────────────────────────────────────


def test_boundary_silence_and_after_last_segment_are_deterministic():
    records = attach_transcripts_to_keyframes(
        [_frame("001", 1.0), _frame("002", 2.0), _frame("003", 4.5), _frame("004", 7.0)],
        [
            TranscriptSegment(video_id="v1", start=0.0, end=2.0, text="  first\tsegment "),
            TranscriptSegment(video_id="v1", start=5.0, end=7.0, text="second"),
        ],
        video_id="v1",
    )

    assert [record["text"] for record in records] == ["  first\tsegment ", None, None, None]


def test_segment_start_boundary_belongs_to_that_segment():
    records = attach_transcripts_to_keyframes(
        [_frame("002", 2.0)],
        [
            TranscriptSegment(video_id="v1", start=0.0, end=2.0, text="first"),
            TranscriptSegment(video_id="v1", start=2.0, end=4.0, text="second"),
        ],
        video_id="v1",
    )
    assert records[0]["text"] == "second"


def test_overlapping_segments_use_first_sorted_match():
    records = attach_transcripts_to_keyframes(
        [_frame("002", 3.0)],
        [
            TranscriptSegment(video_id="v1", start=2.0, end=4.0, text="later"),
            TranscriptSegment(video_id="v1", start=0.0, end=5.0, text="first"),
        ],
        video_id="v1",
    )
    assert records[0]["text"] == "first"


def test_preserves_original_fields_and_only_adds_text():
    records = attach_transcripts_to_keyframes(
        [_frame("002", 3.0, score=0.9, label="person")],
        [TranscriptSegment(video_id="v1", start=0.0, end=4.0, text="hello")],
        video_id="v1",
    )
    assert records == [{
        "img_path": "002.png", "score": 0.9, "label": "person",
        "video_id": "v1", "frame_id": "002", "timestamp": 3.0, "text": "hello",
    }]


def test_keyframes_from_another_video_are_rejected():
    with pytest.raises(ValueError, match="expected video_id"):
        attach_transcripts_to_keyframes(
            [_frame("001", 1.0, "v1"), _frame("001", 1.0, "v2")],
            [TranscriptSegment(video_id="v1", start=0.0, end=2.0, text="only v1")],
            video_id="v1",
        )


def test_duplicate_frame_id_is_rejected():
    with pytest.raises(ValueError, match="Duplicate frame_id"):
        attach_transcripts_to_keyframes(
            [_frame("001", 1.0), _frame("001", 2.0)], [], video_id="v1"
        )


@pytest.mark.parametrize("timestamp", [-1.0, float("nan"), float("inf")])
def test_invalid_keyframe_timestamp_is_rejected(timestamp: float):
    with pytest.raises(ValueError, match="timestamp must be finite"):
        attach_transcripts_to_keyframes([_frame("001", timestamp)], [], video_id="v1")


def test_segments_from_another_video_are_rejected():
    with pytest.raises(ValueError, match="Segment 0: expected video_id"):
        attach_transcripts_to_keyframes(
            [_frame("001", 1.0)],
            [TranscriptSegment(video_id="v2", start=0.0, end=2.0, text="wrong video")],
            video_id="v1",
        )
