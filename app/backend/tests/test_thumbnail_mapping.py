from __future__ import annotations

from types import SimpleNamespace

from search.service import _thumbnail


def test_thumbnail_uses_nearest_official_keyframe_mapping(tmp_path) -> None:
    (tmp_path / "L22_V024.csv").write_text(
        "n,pts_time,fps,frame_idx\n"
        "174,701.0,25.0,17525\n"
        "175,705.08,25.0,17627\n",
        encoding="utf-8",
    )
    config = SimpleNamespace(
        FRAME_IMAGE_URL_TEMPLATE="/keyframes/{video_id}/{keyframe_number}.jpg",
        KEYFRAME_MAP_DIR=tmp_path,
    )

    thumbnail = _thumbnail(
        config,
        {"video_id": "L22_V024", "frame_id": 17630, "shot_id": ""},
    )

    assert thumbnail == "/keyframes/L22_V024/175.jpg"


def test_thumbnail_pads_official_keyframe_numbers_to_three_digits(tmp_path) -> None:
    (tmp_path / "L21_V006.csv").write_text(
        "n,pts_time,fps,frame_idx\n"
        "1,4.0,25.0,100\n",
        encoding="utf-8",
    )
    config = SimpleNamespace(
        FRAME_IMAGE_URL_TEMPLATE="/keyframes/{video_id}/{keyframe_number}.jpg",
        KEYFRAME_MAP_DIR=tmp_path,
    )

    thumbnail = _thumbnail(
        config,
        {"video_id": "L21_V006", "frame_id": 100, "shot_id": ""},
    )

    assert thumbnail == "/keyframes/L21_V006/001.jpg"
