"""Tests for ASR contracts, domain errors, and the timestamp-containment join."""

import json

import pytest
from pydantic import ValidationError

from asr.schema import KeyframeRecord, KeyframeWithText, TranscriptResult, TranscriptSegment
from asr.transcript import (
    attach_transcripts_to_keyframes,
    keyframes_to_csv,
    load_keyframes,
    segments_to_csv,
)


def _write_frame_manifest(metadata_dir, video_id, frames):
    video_dir = metadata_dir / video_id
    video_dir.mkdir(parents=True)
    (video_dir / "Frame.json").write_text(
        json.dumps({"video_id": video_id, "frames": frames})
    )


def test_load_keyframes_keeps_only_kept_and_stores_ms_directly(tmp_path):
    _write_frame_manifest(
        tmp_path,
        "L21_V01",
        [
            {"frame_id": "00000000", "timestamp_ms": 0, "final_status": "KEPT"},
            {"frame_id": "00000025", "timestamp_ms": 1000, "final_status": "DUPLICATE"},
            {"frame_id": "00000078", "timestamp_ms": 3120, "final_status": "KEPT"},
        ],
    )

    keyframes = load_keyframes(tmp_path, "L21_V01")

    assert [k.frame_id for k in keyframes] == ["00000000", "00000078"]
    assert keyframes[1].timestamp_ms == 3120
    assert keyframes[1].img_path == "data/frames/L21_V01/00000078.png"
    assert all(k.video_id == "L21_V01" for k in keyframes)


def _frame(frame_id: str, timestamp_ms: int, *, video_id: str = "v1") -> KeyframeRecord:
    return KeyframeRecord(
        video_id=video_id,
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        img_path=f"{frame_id}.png",
    )


# ── Contracts ──────────────────────────────────────────────────────────


def test_keyframe_contract_uses_direct_timestamp_and_frame_id():
    record = KeyframeRecord(
        video_id="v1",
        frame_id="002",
        timestamp_ms=83,
        img_path="002.png",
    )
    assert record.frame_id == "002"
    assert record.timestamp_ms == 83
    assert record.img_path == "002.png"


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
    record = KeyframeRecord(
        video_id="v1", frame_id="002", timestamp_ms=83, img_path="002.png"
    )
    with pytest.raises(ValidationError):
        record.frame_id = "003"  # type: ignore[misc]


# ── Timestamp containment ──────────────────────────────────────────────


def test_boundary_silence_and_after_last_segment_are_deterministic():
    records = attach_transcripts_to_keyframes(
        [_frame("001", 1000), _frame("002", 2000), _frame("003", 4500), _frame("004", 7000)],
        [
            TranscriptSegment(video_id="v1", start=0.0, end=2.0, text="  first\tsegment "),
            TranscriptSegment(video_id="v1", start=5.0, end=7.0, text="second"),
        ],
        video_id="v1",
    )

    assert [record["text"] for record in records] == ["  first\tsegment ", None, None, None]


def test_segment_start_boundary_belongs_to_that_segment():
    records = attach_transcripts_to_keyframes(
        [_frame("002", 2000)],
        [
            TranscriptSegment(video_id="v1", start=0.0, end=2.0, text="first"),
            TranscriptSegment(video_id="v1", start=2.0, end=4.0, text="second"),
        ],
        video_id="v1",
    )
    assert records[0]["text"] == "second"


def test_overlapping_segments_use_first_sorted_match():
    records = attach_transcripts_to_keyframes(
        [_frame("002", 3000)],
        [
            TranscriptSegment(video_id="v1", start=2.0, end=4.0, text="later"),
            TranscriptSegment(video_id="v1", start=0.0, end=5.0, text="first"),
        ],
        video_id="v1",
    )
    assert records[0]["text"] == "first"


def test_adds_video_id_and_text_to_autoshot_keyframe():
    records = attach_transcripts_to_keyframes(
        [_frame("002", 3000)],
        [TranscriptSegment(video_id="v1", start=0.0, end=4.0, text="hello")],
        video_id="v1",
    )
    assert records == [{
        "img_path": "002.png",
        "video_id": "v1", "frame_id": "002", "timestamp_ms": 3000, "text": "hello",
    }]


def test_duplicate_frame_id_is_rejected():
    with pytest.raises(ValueError, match="Duplicate frame_id"):
        attach_transcripts_to_keyframes(
            [_frame("001", 1000), _frame("001", 2000)], [], video_id="v1"
        )


def test_negative_keyframe_timestamp_is_rejected():
    with pytest.raises(ValueError, match="timestamp_ms must be a non-negative integer"):
        attach_transcripts_to_keyframes([_frame("001", -1)], [], video_id="v1")


def test_segments_from_another_video_are_rejected():
    with pytest.raises(ValueError, match="Segment 0: expected video_id"):
        attach_transcripts_to_keyframes(
            [_frame("001", 1.0)],
            [TranscriptSegment(video_id="v2", start=0.0, end=2.0, text="wrong video")],
            video_id="v1",
        )


# ── CSV export ──────────────────────────────────────────────────────────


def test_segments_to_csv_returns_header_and_data():
    csv = segments_to_csv([
        TranscriptSegment(video_id="v1", start=0.0, end=2.0, text="xin chào"),
        TranscriptSegment(video_id="v1", start=2.0, end=4.0, text="bạn khoẻ không", confidence=0.95),
    ])
    lines = csv.strip().split("\n")
    assert len(lines) == 3
    assert lines[0] == "video_id,start,end,text,confidence"
    assert lines[1] == "v1,0.0,2.0,xin chào,"
    assert lines[2] == "v1,2.0,4.0,bạn khoẻ không,0.95"


def test_keyframes_to_csv_accepts_dicts_from_attach():
    records = attach_transcripts_to_keyframes(
        [_frame("001", 1000), _frame("002", 3000)],
        [
            TranscriptSegment(video_id="v1", start=0.0, end=2.0, text="đoạn một"),
            TranscriptSegment(video_id="v1", start=2.0, end=5.0, text="đoạn hai"),
        ],
        video_id="v1",
    )
    csv = keyframes_to_csv(records)
    lines = csv.strip().split("\n")
    assert len(lines) == 3
    assert lines[0] == "video_id,frame_id,timestamp_ms,img_path,text"
    assert lines[1].endswith(",đoạn một")
    assert lines[2].endswith(",đoạn hai")


def test_keyframes_to_csv_handles_empty():
    assert keyframes_to_csv([]) == "video_id,frame_id,timestamp_ms,img_path,text\n"


def test_keyframes_to_csv_accepts_keyframewithtext_objects():
    from asr.schema import KeyframeWithText

    csv = keyframes_to_csv([
        KeyframeWithText(video_id="v1", frame_id="001", timestamp_ms=1000, img_path="001.png", text="chào"),
        KeyframeWithText(video_id="v1", frame_id="002", timestamp_ms=3000, img_path="002.png", text=None),
    ])
    lines = csv.strip().split("\n")
    assert lines[1] == "v1,001,1000,001.png,chào"
    assert lines[2] == "v1,002,3000,002.png,"
