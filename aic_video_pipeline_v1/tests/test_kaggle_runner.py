import importlib.util
import tarfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "kaggle_archive_runner.py"
SPEC = importlib.util.spec_from_file_location("kaggle_archive_runner", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_archive_list_matches_requested_datasets() -> None:
    assert runner.DEFAULT_ARCHIVES == [
        "Videos_L26_a.zip", "Videos_L26_b.zip",
        "Videos_L26_c.zip", "Videos_L26_d.zip",
        "Videos_L26_e.zip",
    ]


def test_package_is_valid_before_materialized_output_is_removed(tmp_path: Path) -> None:
    data = tmp_path / "data"
    result = tmp_path / "result"
    video_id = "L21_V001"
    files = {
        data / "metadata" / video_id / "Shot.json": b"{}",
        data / "metadata" / video_id / "Frame.json": b"{}",
        data / "frames" / video_id / "0.png": b"png",
        data / "vectors" / video_id / "0.npy": b"npy",
        data / "checkpoints" / f"{video_id}.json": b"{}",
    }
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    archive = runner.package_video(data, result, video_id)
    assert runner.valid_result_tar(archive, video_id)
    runner.remove_materialized_video(data, video_id)
    assert archive.is_file()
    assert not (data / "metadata" / video_id).exists()
    assert not (data / "frames" / video_id).exists()
    assert not (data / "vectors" / video_id).exists()
    assert not (data / "checkpoints" / f"{video_id}.json").exists()


def test_previous_kaggle_state_is_merged_without_duplicate_members() -> None:
    current = {"schema_version": "1.0", "archives": {
        "Videos_L21_a.zip": {"completed_members": ["video/L21_V001.mp4"],
                             "done": False},
    }}
    previous = {"schema_version": "1.0", "archives": {
        "Videos_L21_a.zip": {"completed_members": ["video/L21_V001.mp4",
                                                      "video/L21_V002.mp4"],
                             "done": True},
    }}
    runner.merge_state(current, previous)
    archive = current["archives"]["Videos_L21_a.zip"]
    assert archive["completed_members"] == ["video/L21_V001.mp4",
                                             "video/L21_V002.mp4"]
    assert archive["done"] is True


def test_reconcile_previous_state_uses_only_valid_final_tar(tmp_path: Path) -> None:
    root = tmp_path / "previous"
    root.mkdir()
    video_id = "L21_V001"
    with tarfile.open(root / f"{video_id}.tar", "w") as archive:
        for name in (
            f"metadata/{video_id}/Shot.json",
            f"metadata/{video_id}/Frame.json",
            f"checkpoints/{video_id}.json",
        ):
            payload = tmp_path / name.replace("/", "_")
            payload.write_text("{}", encoding="utf-8")
            archive.add(payload, arcname=name)
    (root / "L21_V002.tar.tmp").write_bytes(b"partial")
    state = {"schema_version": "1.0", "archives": {
        "Videos_L21_a.zip": {
            "completed_members": ["video/L21_V001.mp4", "video/L21_V002.mp4"],
            "done": True,
        },
    }}

    runner.reconcile_completed_members(state, [root])
    archive = state["archives"]["Videos_L21_a.zip"]
    assert archive["completed_members"] == ["video/L21_V001.mp4"]
    assert archive["done"] is False


def make_extracted_result(root: Path, video_id: str) -> Path:
    extracted = root / video_id
    files = {
        extracted / "metadata" / video_id / "Shot.json": b"{}",
        extracted / "metadata" / video_id / "Frame.json": b"{}",
        extracted / "frames" / video_id / "1.png": b"png",
        extracted / "vectors" / video_id / "1.npy": b"npy",
        extracted / "checkpoints" / f"{video_id}.json": b"{}",
    }
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return extracted


def test_kaggle_extracted_tar_directory_is_a_valid_previous_result(
        tmp_path: Path) -> None:
    root = tmp_path / "previous"
    video_id = "L22_V030"
    extracted = make_extracted_result(root, video_id)

    assert runner.valid_result_directory(extracted, video_id)
    assert runner.result_exists(root, video_id)
    assert runner.previous_result_exists([root], video_id)


def test_reconcile_previous_state_accepts_kaggle_extracted_tar(
        tmp_path: Path) -> None:
    root = tmp_path / "previous"
    make_extracted_result(root, "L22_V001")
    (root / "L22_V002.tar.tmp").write_bytes(b"partial")
    state = {"schema_version": "1.0", "archives": {
        "Videos_L22_a.zip": {
            "completed_members": ["video/L22_V001.mp4", "video/L22_V002.mp4"],
            "done": True,
        },
    }}

    runner.reconcile_completed_members(state, [root])
    archive = state["archives"]["Videos_L22_a.zip"]
    assert archive["completed_members"] == ["video/L22_V001.mp4"]
    assert archive["done"] is False
