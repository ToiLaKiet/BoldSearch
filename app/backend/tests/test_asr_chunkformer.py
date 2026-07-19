"""Tests for the ChunkFormer adapter."""

from __future__ import annotations

import pytest

from asr.chunkformer import ChunkFormerTranscriber


class TestParseTimestamp:
    def test_parses_real_chunkformer_1_2_2_format(self):
        """Package 1.2.2 emits ``HH:MM:SS:mmm`` from model_utils."""
        assert ChunkFormerTranscriber._parse_timestamp("00:01:30:500") == 90.5

    def test_parses_full_format(self):
        assert ChunkFormerTranscriber._parse_timestamp("00:01:30.500") == 90.5

    def test_parses_hour_boundary(self):
        assert ChunkFormerTranscriber._parse_timestamp("01:00:00.000") == 3600.0

    def test_parses_minute_second(self):
        assert ChunkFormerTranscriber._parse_timestamp("00:05:10.050") == 310.05

    def test_rejects_malformed(self):
        with pytest.raises(ValueError, match="Cannot parse timestamp"):
            ChunkFormerTranscriber._parse_timestamp("not-a-timestamp")


class TestNormalizeResult:
    def test_empty_list(self):
        result = ChunkFormerTranscriber._normalize_result([], video_id="v1")
        assert result.language == "vi"
        assert result.text == ""
        assert result.segments == ()

    def test_single_segment(self):
        raw = [
            {"decode": "xin chào", "start": "00:00:00.080", "end": "00:00:01.200"}
        ]
        result = ChunkFormerTranscriber._normalize_result(raw, video_id="v1")
        assert result.text == "xin chào"
        assert len(result.segments) == 1
        assert result.segments[0].start == 0.08
        assert result.segments[0].end == 1.2
        assert result.segments[0].video_id == "v1"

    def test_multiple_segments_joined_with_space(self):
        raw = [
            {"decode": "chào", "start": "00:00:00.000", "end": "00:00:01.000"},
            {"decode": "bạn", "start": "00:00:01.000", "end": "00:00:02.000"},
        ]
        result = ChunkFormerTranscriber._normalize_result(raw, video_id="v1")
        assert result.text == "chào bạn"
        assert len(result.segments) == 2

    def test_rejects_non_list(self):
        with pytest.raises(ValueError, match="Expected list"):
            ChunkFormerTranscriber._normalize_result("invalid", video_id="v1")

    @pytest.mark.parametrize(
        "raw",
        [
            [{"decode": "missing timestamps"}],
            [{"decode": "reversed", "start": "00:00:02:000", "end": "00:00:01:000"}],
        ],
    )
    def test_rejects_invalid_segment_shape(self, raw):

        with pytest.raises(ValueError):
            ChunkFormerTranscriber._normalize_result(raw, video_id="v1")
