from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def test_l26_offline_launcher_uses_direct_video_inputs_and_offline_wheels() -> None:
    launcher = (PROJECT / "scripts" / "kaggle_run_l26_de_offline.sh").read_text(
        encoding="utf-8"
    )
    assert "Videos_L26_d" in launcher
    assert "Videos_L26_e" in launcher
    assert '"${PIPELINE_ROOT}/scripts/kaggle_directory_runner.py"' in launcher
    assert '--video-root "$L26_D_VIDEO_ROOT"' in launcher
    assert '--video-root "$L26_E_VIDEO_ROOT"' in launcher
    assert '--gpu-workers "$GPU_WORKERS"' in launcher
    assert "PIP_NO_INDEX=1" in launcher
    assert "HF_HUB_OFFLINE=1" in launcher
    assert "TRANSFORMERS_OFFLINE=1" in launcher
    assert "--previous-result-root" not in launcher
    assert "wget " not in launcher
    assert "apt-get " not in launcher
    assert "snapshot_download" not in launcher


def test_l26_offline_cell_is_a_single_bash_cell_with_explicit_model_root() -> None:
    cell = (PROJECT / "scripts" / "kaggle_cell_run_l26_de_offline.txt").read_text(
        encoding="utf-8"
    )
    assert cell.startswith("%%bash\n")
    assert "/kaggle/input/datasets/quanglongl040305/model2/aic_l28_offline_models" in cell
    assert "kaggle_run_l26_de_offline.sh" in cell


def test_l26_offline_builder_carries_all_required_runtime_assets() -> None:
    builder = (PROJECT / "scripts" / "build_kaggle_l26_de_offline_bundle.sh").read_text(
        encoding="utf-8"
    )
    assert "aic_video_pipeline_v1_l26_de_offline_kaggle.zip" in builder
    assert "offline_wheels" in builder
    assert "kaggle_directory_runner.py" in builder
    assert "archive.testzip()" in builder
