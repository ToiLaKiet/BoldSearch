from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def test_l26_b_launcher_is_exactly_bounded_and_offline() -> None:
    launcher = (
        PROJECT / "scripts" / "kaggle_run_l26_b_v175_v199_offline.sh"
    ).read_text(encoding="utf-8")
    assert 'START_VIDEO_ID="L26_V175"' in launcher
    assert 'END_VIDEO_ID="L26_V199"' in launcher
    assert 'Videos_L26_b/video' in launcher
    assert '--start-at-video "$START_VIDEO_ID"' in launcher
    assert '--end-at-video "$END_VIDEO_ID"' in launcher
    assert "PIP_NO_INDEX=1" in launcher
    assert "HF_HUB_OFFLINE=1" in launcher
    assert "TRANSFORMERS_OFFLINE=1" in launcher
    assert "--previous-result-root" not in launcher
    assert "wget " not in launcher
    assert "apt-get " not in launcher


def test_l26_b_cell_uses_user_provided_model_and_video_paths() -> None:
    cell = (
        PROJECT / "scripts" / "kaggle_cell_run_l26_b_v175_v199_offline.txt"
    ).read_text(encoding="utf-8")
    assert cell.startswith("%%bash\n")
    assert "/kaggle/input/datasets/quanglongl040305/model2/aic_l28_offline_models" in cell
    assert "/kaggle/input/datasets/miphu2005/aic2026-videos-l21-a/Videos_L26_b/video" in cell


def test_l26_b_bundle_builder_includes_range_runner_and_offline_wheels() -> None:
    builder = (
        PROJECT / "scripts" / "build_kaggle_l26_b_v175_v199_offline_bundle.sh"
    ).read_text(encoding="utf-8")
    assert "aic_video_pipeline_v1_l26_b_v175_v199_offline_kaggle.zip" in builder
    assert "offline_wheels" in builder
    assert "kaggle_directory_runner.py" in builder
    assert "archive.testzip()" in builder
