import os
import subprocess
import tarfile
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def test_colab_l27_launcher_uses_exact_archive_and_drive_output() -> None:
    launcher = (PROJECT / "scripts" / "colab_run_l27.sh").read_text(
        encoding="utf-8"
    )
    assert "https://aic-data.ledo.io.vn/Videos_L27_a.zip" in launcher
    assert 'EXPECTED_VIDEO_COUNT=16' in launcher
    assert "/content/drive/MyDrive/aic_pipeline_results/L27" in launcher
    assert '--archives "$ARCHIVE_NAME"' in launcher
    assert '--base-url "$ARCHIVE_BASE_URL"' in launcher
    assert 'AIC_ARCHIVES' not in launcher
    assert "/kaggle/" not in launcher


def test_colab_l27_launcher_preserves_old_processing_config() -> None:
    config = (PROJECT / "configs" / "default.yaml").read_text(encoding="utf-8")
    launcher = (PROJECT / "scripts" / "colab_run_l27.sh").read_text(
        encoding="utf-8"
    )
    assert "sample_every_frames: 10" in config
    assert "threshold: 0.296" in config
    assert "batch_size: 4" in config
    assert "threshold: 0.5" in config
    assert 'config["indexer"]["sample_every_frames"] == 10' in launcher
    assert 'config["embedding"]["batch_size"] == 4' in launcher
    assert 'config["similarity"]["threshold"] == 0.5' in launcher


def test_colab_cell_mounts_drive_and_uses_unique_bundle() -> None:
    cell = (PROJECT / "scripts" / "colab_cell_run_l27.txt").read_text(
        encoding="utf-8"
    )
    assert "drive.mount('/content/drive')" in cell
    assert 'BUNDLE_NAME="aic_video_pipeline_v1_l27_colab.zip"' in cell
    assert "python -m zipfile -e" in cell
    assert "scripts/colab_run_l27.sh" in cell
    assert "/content/drive/MyDrive/aic_pipeline_results/L27" in cell


def test_colab_bundle_builder_includes_only_required_runtime_source() -> None:
    builder = (PROJECT / "scripts" / "build_colab_l27_bundle.sh").read_text(
        encoding="utf-8"
    )
    assert "aic_video_pipeline_v1_l27_colab.zip" in builder
    assert "archive.testzip()" in builder
    assert 'PYTHON_BIN="/home/long/miniconda3/bin/python"' in builder
    assert "kaggle_archive_runner.py" in builder
    assert "colab_run_l27.sh" in builder
    assert "colab_cell_run_l27.txt" in builder
    assert "model.safetensors" not in builder


def test_colab_launcher_skips_network_when_all_l27_results_exist(
    tmp_path: Path,
) -> None:
    result_root = tmp_path / "results"
    result_root.mkdir()
    payload = tmp_path / "payload"
    payload.write_bytes(b"ok")

    for number in range(1, 17):
        video_id = f"L27_V{number:03d}"
        with tarfile.open(result_root / f"{video_id}.tar", "w") as archive:
            for name in (
                f"metadata/{video_id}/Shot.json",
                f"metadata/{video_id}/Frame.json",
                f"checkpoints/{video_id}.json",
                f"frames/{video_id}/0.png",
                f"vectors/{video_id}/0.npy",
            ):
                archive.add(payload, arcname=name)

    environment = os.environ.copy()
    environment["AIC_RESULT_ROOT"] = str(result_root)
    environment["AIC_RUNTIME_ROOT"] = str(tmp_path / "runtime")
    result = subprocess.run(
        ["bash", str(PROJECT / "scripts" / "colab_run_l27.sh")],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "16/16" in result.stdout
    assert "không tải lại model hoặc video" in result.stdout
    assert not (tmp_path / "runtime" / "work" / "downloads").exists()
