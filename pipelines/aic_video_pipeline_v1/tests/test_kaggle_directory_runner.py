import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "kaggle_directory_runner.py"
SPEC = importlib.util.spec_from_file_location("kaggle_directory_runner", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_discover_videos_accepts_only_direct_matching_l28_mp4s(
        tmp_path: Path) -> None:
    root = tmp_path / "Videos_L28_a" / "video"
    root.mkdir(parents=True)
    for name in ("L28_V010.mp4", "L28_V002.mp4", "L28_V001.MP4",
                 "L27_V001.mp4", "notes.mp4"):
        (root / name).write_bytes(b"video")
    nested = root / "nested"
    nested.mkdir()
    (nested / "L28_V003.mp4").write_bytes(b"video")

    assert [path.name for path in runner.discover_videos(root, "L28")] == [
        "L28_V001.MP4", "L28_V002.mp4", "L28_V010.mp4",
    ]


def test_discover_videos_requires_a_matching_direct_mp4(tmp_path: Path) -> None:
    root = tmp_path / "video"
    root.mkdir()
    (root / "L27_V001.mp4").write_bytes(b"video")

    with pytest.raises(ValueError, match="no L28"):
        runner.discover_videos(root, "L28")


def test_discover_videos_from_l26_parts_sorts_and_rejects_duplicates(
        tmp_path: Path) -> None:
    part_a = tmp_path / "Videos_L26_a" / "video"
    part_b = tmp_path / "Videos_L26_b" / "video"
    part_a.mkdir(parents=True)
    part_b.mkdir(parents=True)
    (part_a / "L26_V002.mp4").write_bytes(b"video")
    (part_b / "L26_V010.mp4").write_bytes(b"video")
    (part_b / "L26_V003.mp4").write_bytes(b"video")

    assert [path.name for path in runner.discover_videos_from_roots(
            [part_b, part_a], "L26")] == [
        "L26_V002.mp4", "L26_V003.mp4", "L26_V010.mp4",
    ]

    (part_b / "L26_V002.mp4").write_bytes(b"duplicate")
    with pytest.raises(ValueError, match="duplicate video_id L26_V002"):
        runner.discover_videos_from_roots([part_a, part_b], "L26")


def test_state_records_all_direct_input_roots(tmp_path: Path) -> None:
    roots = [tmp_path / "a", tmp_path / "b"]
    state = runner.new_state(roots, "L26")
    assert state["input_roots"] == [str(root) for root in roots]
    assert "input_root" not in state
