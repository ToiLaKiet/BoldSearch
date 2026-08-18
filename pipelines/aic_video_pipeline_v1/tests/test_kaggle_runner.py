import importlib.util
import json
import tarfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "kaggle_archive_runner.py"
PROJECT = SCRIPT.parents[1]
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
        data / "metadata" / video_id / "Shot.json": json.dumps({
            "video_id": video_id, "shots": []
        }).encode(),
        data / "metadata" / video_id / "Frame.json": json.dumps({
            "video_id": video_id,
            "source_video_path": "/tmp/video/L21_V001.mp4",
            "frames": [{
                "frame_id": "0",
                "frame_path": str(data / "frames" / video_id / "0.png"),
                "vector_path": str(data / "vectors" / video_id / "0.npy"),
            }],
        }).encode(),
        data / "frames" / video_id / "0.png": b"png",
        data / "vectors" / video_id / "0.npy": b"npy",
        data / "checkpoints" / f"{video_id}.json": json.dumps({
            "video_id": video_id,
            "source": {"path": "/tmp/video/L21_V001.mp4", "size": 1},
        }).encode(),
    }
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    archive = runner.package_video(data, result, video_id)
    assert runner.valid_result_tar(archive, video_id)
    with tarfile.open(archive, "r") as packaged:
        frame_document = json.load(packaged.extractfile(
            f"metadata/{video_id}/Frame.json"
        ))
        checkpoint = json.load(packaged.extractfile(
            f"checkpoints/{video_id}.json"
        ))
    assert frame_document["source_video_path"] == "L21_V001.mp4"
    assert frame_document["frames"][0]["frame_path"] == (
        f"frames/{video_id}/0.png"
    )
    assert frame_document["frames"][0]["vector_path"] == (
        f"vectors/{video_id}/0.npy"
    )
    assert checkpoint["source"]["path"] == "L21_V001.mp4"
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
            f"frames/{video_id}/1.png",
            f"vectors/{video_id}/1.npy",
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


def test_kaggle_runtime_assets_are_outside_saved_working_output() -> None:
    setup = (PROJECT / "scripts" / "kaggle_setup_and_run.sh").read_text(
        encoding="utf-8"
    )
    assert 'mktemp -d -p /tmp aic_video_pipeline_runtime.XXXXXX' in setup
    assert 'RESULT_ROOT="${KAGGLE_WORKING_ROOT}/aic_pipeline_results"' in setup
    assert '${KAGGLE_WORKING_ROOT}/models/fgclip2' not in setup
    assert '${KAGGLE_WORKING_ROOT}/aic_pipeline_work' not in setup
    assert 'local_dir="/kaggle/working/models/fgclip2"' not in setup


def test_remote_level_launchers_select_exactly_one_video_archive() -> None:
    for level in (27,):
        launcher = (PROJECT / "scripts" / f"kaggle_run_l{level}.sh").read_text(
            encoding="utf-8"
        )
        assert f'export AIC_ARCHIVES="Videos_L{level}_a.zip"' in launcher
        for other in {27, 28, 29, 30} - {level}:
            assert f"Videos_L{other}_a.zip" not in launcher


def test_direct_level_launchers_use_attached_mp4_directories() -> None:
    for level in (25, 28, 29, 30):
        launcher = (PROJECT / "scripts" / f"kaggle_run_l{level}.sh").read_text(
            encoding="utf-8"
        )
        assert f'export AIC_VIDEO_LEVEL="L{level}"' in launcher
        assert f"Videos_L{level}_a/video" in launcher
        assert "modelfortrainning/aic_l28_offline_models" in launcher
        assert "export AIC_ARCHIVES=" not in launcher


def test_l26_launcher_uses_all_five_attached_video_parts() -> None:
    launcher = (PROJECT / "scripts" / "kaggle_run_l26.sh").read_text(
        encoding="utf-8"
    )
    assert 'export AIC_VIDEO_LEVEL="L26"' in launcher
    assert "AIC_VIDEO_ROOTS=" in launcher
    for part in "abcde":
        assert f"Videos_L26_{part}/video" in launcher
    assert "modelfortrainning/aic_l28_offline_models" in launcher
    assert "export AIC_ARCHIVES=" not in launcher

    setup = (PROJECT / "scripts" / "kaggle_setup_and_run.sh").read_text(
        encoding="utf-8"
    )
    assert 'IFS=\':\' read -r -a direct_video_roots <<< "$AIC_VIDEO_ROOTS"' in setup
    assert 'video_root_args+=(--video-root "$direct_video_root")' in setup

    setup = (PROJECT / "scripts" / "kaggle_setup_and_run.sh").read_text(
        encoding="utf-8"
    )
    assert "kaggle_directory_runner.py" in setup
    assert '"${video_root_args[@]}"' in setup
    assert '--level "$AIC_VIDEO_LEVEL"' in setup
    assert "AIC_OFFLINE_MODEL_ROOT" in setup
    assert 'export PYTHONPATH="${PIPELINE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"' in setup
    assert "Python dependencies cho AutoShot/FG-CLIP2: OK" in setup
    assert '("safetensors", "import safetensors")' in setup
    assert '"ffmpeg-python", "import ffmpeg"' not in setup
    assert '"matplotlib", "import matplotlib"' not in setup
    assert "from torchvision.ops import roi_align" in setup
    assert "from transformers.modeling_layers import GradientCheckpointingLayer" in setup
    assert "USE_OFFLINE_MODEL_ASSETS" in setup

    offline_install_block = setup.split(
        "if (( USE_OFFLINE_MODEL_ASSETS == 1 )); then\n"
        "    # Both Kaggle runners",
        maxsplit=1,
    )[1].split("\nelse\n", maxsplit=1)[0]
    assert "pip install" not in offline_install_block


def test_l28_cell_discovers_direct_or_zipped_kaggle_inputs() -> None:
    cell = (PROJECT / "scripts" / "kaggle_cell_run_l28.txt").read_text(
        encoding="utf-8"
    )
    assert "quanglongl040305/croodd" in cell
    assert 'CODE_SEARCH_ROOT="${AIC_CODE_DATASET_ROOT:-$DEFAULT_CODE_DATASET}"' in cell
    assert 'CODE_ROOT="${AIC_CODE_ROOT:-}"' in cell
    assert "aic_video_pipeline_l28.zip" in cell
    assert "python -m zipfile -e" in cell
    assert 'MODEL_ROOT="${AIC_OFFLINE_MODEL_ROOT:-$DEFAULT_MODEL_ROOT}"' in cell
    assert "*/fgclip2/model.safetensors" in cell
    assert 'export AIC_OFFLINE_MODEL_ROOT="$MODEL_ROOT"' in cell
    assert 'VIDEO_ROOT="${AIC_VIDEO_ROOT:-$DEFAULT_VIDEO_ROOT}"' in cell
    assert "L28_V*.mp4" in cell
    assert 'export AIC_VIDEO_ROOT="$VIDEO_ROOT"' in cell
    assert 'RUN_SCRIPT="${CODE_ROOT}/scripts/kaggle_run_l28.sh"' in cell
    assert "ZIP source vẫn là bản cũ còn lỗi dependency offline" in cell
    assert "def _decode_autoshot_frames" in cell
    assert '("ffmpeg-python", "import ffmpeg")' in cell
    assert 'mktemp -d -p /tmp aic_l28_code.XXXXXX' in cell
    assert 'mktemp -d -p /kaggle/working' not in cell


def test_remote_kaggle_cell_extracts_source_outside_working() -> None:
    for level in (27,):
        cell = (PROJECT / "scripts" / f"kaggle_cell_run_l{level}.txt").read_text(
            encoding="utf-8"
        )
        assert f'BUNDLE_NAME="aic_video_pipeline_l{level}.zip"' in cell
        assert f'mktemp -d -p /tmp aic_video_pipeline_l{level}_bundle.XXXXXX' in cell
        assert "/kaggle/working/aic_video_pipeline" not in cell


def test_l29_l30_cells_use_fixed_bundle_and_direct_mp4_input() -> None:
    for level in (29, 30):
        cell = (PROJECT / "scripts" / f"kaggle_cell_run_l{level}.txt").read_text(
            encoding="utf-8"
        )
        assert f'LEVEL="L{level}"' in cell
        assert f'BUNDLE_NAME="aic_video_pipeline_l{level}.zip"' in cell
        assert f"Videos_L{level}_a/video" in cell
        assert f"L{level}_V*.mp4" in cell
        assert f"kaggle_run_l{level}.sh" in cell
        assert 'export PYTHONPATH="${PIPELINE_ROOT}/src' in cell
        assert "def _decode_autoshot_frames" in cell
        assert '("ffmpeg-python", "import ffmpeg")' in cell
        assert 'export AIC_OFFLINE_MODEL_ROOT="$MODEL_ROOT"' in cell
        assert 'export AIC_VIDEO_ROOT="$VIDEO_ROOT"' in cell
        assert "AIC_ARCHIVES" not in cell
        assert f"mktemp -d -p /tmp aic_video_pipeline_l{level}_bundle.XXXXXX" in cell
        assert 'mktemp -d -p /kaggle/working' not in cell


def test_l25_cell_uses_its_unique_bundle_and_direct_mp4_input() -> None:
    cell = (PROJECT / "scripts" / "kaggle_cell_run_l25.txt").read_text(
        encoding="utf-8"
    )
    assert 'LEVEL="L25"' in cell
    assert "quanglongl040305/l25running/aic_video_pipeline_v1" in cell
    assert 'CODE_ROOT="${AIC_CODE_ROOT:-$DEFAULT_CODE_ROOT}"' in cell
    assert 'BUNDLE_NAME="aic_video_pipeline_l25.zip"' in cell
    assert "Videos_L25_a/video/L25_V*.mp4" in cell
    assert "kaggle_run_l25.sh" in cell
    assert "def _decode_autoshot_frames" in cell
    assert 'export AIC_VIDEO_ROOT="$VIDEO_ROOT"' in cell
    assert "export AIC_ARCHIVES=" not in cell


def test_l26_cell_discovers_five_parts_and_exports_multiple_roots() -> None:
    cell = (PROJECT / "scripts" / "kaggle_cell_run_l26.txt").read_text(
        encoding="utf-8"
    )
    assert 'LEVEL="L26"' in cell
    assert 'BUNDLE_NAME="aic_video_pipeline_l26.zip"' in cell
    assert 'AIC_L26_PARTS:-a b c d e' in cell
    assert 'Videos_L26_${part}/video' in cell
    assert "L26_V*.mp4" in cell
    assert 'export AIC_VIDEO_ROOTS="$VIDEO_ROOTS_VALUE"' in cell
    assert "kaggle_run_l26.sh" in cell
    assert "export AIC_ARCHIVES=" not in cell


def test_level_bundle_builder_includes_the_copyable_kaggle_cell() -> None:
    builder = (PROJECT / "scripts" / "build_kaggle_level_bundles.sh").read_text(
        encoding="utf-8"
    )
    assert 'kaggle_cell_run_l${level}.txt' in builder
    assert '^(25|26|27|28|29|30)$' in builder
    assert "archive.testzip()" in builder


def test_runner_rejects_cross_level_video_members() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'expected_level = archive_name.removeprefix("Videos_")' in source
    assert 'startswith(f"{expected_level}_V")' in source
