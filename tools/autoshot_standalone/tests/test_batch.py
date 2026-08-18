from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_batch import build_tasks, read_video_list


def test_read_video_list_ignores_comments_and_resolves_relative_paths(tmp_path: Path) -> None:
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"video")
    second.write_bytes(b"video")
    listing = tmp_path / "videos.txt"
    listing.write_text("# comment\nfirst.mp4\n\n" + str(second) + "\n", encoding="utf-8")
    assert read_video_list(listing) == [first.resolve(), second.resolve()]


def test_build_tasks_rejects_duplicate_output_stems(tmp_path: Path) -> None:
    left = tmp_path / "left" / "same.mp4"
    right = tmp_path / "right" / "same.mp4"
    left.parent.mkdir()
    right.parent.mkdir()
    left.write_bytes(b"video")
    right.write_bytes(b"video")
    with pytest.raises(ValueError, match="stems must be unique"):
        build_tasks([left, right], tmp_path / "out")
