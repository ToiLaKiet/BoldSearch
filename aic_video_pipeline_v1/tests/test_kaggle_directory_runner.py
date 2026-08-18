import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "kaggle_directory_runner.py"
SPEC = importlib.util.spec_from_file_location("kaggle_directory_runner", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_discover_videos_uses_only_matching_direct_l26_mp4s(tmp_path: Path) -> None:
    root = tmp_path / "Videos_L26_d" / "video"
    root.mkdir(parents=True)
    for name in (
        "L26_V010.mp4",
        "L26_V002.mp4",
        "L26_V001.MP4",
        "L25_V001.mp4",
        "notes.mp4",
    ):
        (root / name).write_bytes(b"video")
    nested = root / "nested"
    nested.mkdir()
    (nested / "L26_V003.mp4").write_bytes(b"video")

    assert [path.name for path in runner.discover_videos(root, "L26")] == [
        "L26_V001.MP4",
        "L26_V002.mp4",
        "L26_V010.mp4",
    ]


def test_merge_parts_sorts_and_rejects_duplicate_video_ids(tmp_path: Path) -> None:
    part_d = tmp_path / "Videos_L26_d" / "video"
    part_e = tmp_path / "Videos_L26_e" / "video"
    part_d.mkdir(parents=True)
    part_e.mkdir(parents=True)
    (part_d / "L26_V102.mp4").write_bytes(b"video")
    (part_e / "L26_V110.mp4").write_bytes(b"video")
    (part_e / "L26_V103.mp4").write_bytes(b"video")

    assert [path.name for path in runner.discover_videos_from_roots(
        [part_e, part_d], "L26"
    )] == ["L26_V102.mp4", "L26_V103.mp4", "L26_V110.mp4"]

    (part_e / "L26_V102.mp4").write_bytes(b"duplicate")
    with pytest.raises(ValueError, match="duplicate video_id L26_V102"):
        runner.discover_videos_from_roots([part_d, part_e], "L26")


def test_state_records_direct_input_roots(tmp_path: Path) -> None:
    roots = [tmp_path / "d", tmp_path / "e"]
    state = runner.new_state(roots, "L26")
    assert state["input_roots"] == [str(root) for root in roots]
    assert state["completed_video_ids"] == []


def test_select_video_range_is_inclusive_and_requires_exact_endpoints(
        tmp_path: Path) -> None:
    videos = [
        tmp_path / f"L26_V{number:03d}.mp4"
        for number in range(173, 202)
    ]

    selected = runner.select_video_range(videos, "L26_V175", "L26_V199")

    assert selected[0].stem == "L26_V175"
    assert selected[-1].stem == "L26_V199"
    assert len(selected) == 25
    with pytest.raises(ValueError, match="start video not found"):
        runner.select_video_range(videos, "L26_V1740", "L26_V199")
