from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import (decode_small_frames, make_temporal_batches, normalize_scenes,
                  predictions_to_scenes, read_video_metadata)


def make_video(path: Path, count: int = 5) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             10, (64, 64))
    assert writer.isOpened()
    for index in range(count):
        image = np.full((64, 64, 3), index * 20, dtype=np.uint8)
        writer.write(image)
    writer.release()


def test_read_and_decode_video_metadata(tmp_path: Path) -> None:
    video = tmp_path / "input.mp4"
    make_video(video, 5)
    metadata = read_video_metadata(video)
    frames = decode_small_frames(video)
    assert metadata["fps"] == pytest.approx(10.0)
    assert len(frames) == 5
    assert frames.shape[1:] == (27, 48, 3)


def test_temporal_batches_match_autoshot_window_shape() -> None:
    frames = np.zeros((51, 27, 48, 3), dtype=np.uint8)
    batches = list(make_temporal_batches(frames))
    assert len(batches) == 2
    assert [batch.shape for batch in batches] == [(100, 27, 48, 3)] * 2


def test_predictions_and_scene_normalization_cover_every_frame() -> None:
    predictions = np.array([0, 0, 1, 1, 0, 0], dtype=np.uint8)
    raw = predictions_to_scenes(predictions)
    normalized = normalize_scenes(raw, len(predictions))
    assert normalized == [(0, 3), (4, 5)]
    assert normalized[0][0] == 0
    assert normalized[-1][1] == len(predictions) - 1
    assert all(left[1] + 1 == right[0]
               for left, right in zip(normalized, normalized[1:]))


def test_empty_scenes_fallback_to_full_video() -> None:
    assert normalize_scenes(np.empty((0, 2), dtype=np.int32), 7) == [(0, 6)]
