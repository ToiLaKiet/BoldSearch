from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def test_l22_kaggle_launcher_starts_exactly_at_v030_without_old_results() -> None:
    launcher = (PROJECT / "scripts" / "kaggle_run_l22_from_v030.sh").read_text(
        encoding="utf-8"
    )
    assert 'ARCHIVE_NAME="Videos_L22_a.zip"' in launcher
    assert 'START_VIDEO_ID="L22_V030"' in launcher
    assert '--archives "$ARCHIVE_NAME"' in launcher
    assert '--start-at-video "$START_VIDEO_ID"' in launcher
    assert "AIC_PREVIOUS_RESULT_ROOT" not in launcher
    assert "--previous-result-root" not in launcher
    assert 'KAGGLE_WORKING_ROOT="/kaggle/working"' in launcher
    assert 'aic_l22_from_v030_results' in launcher


def test_l22_kaggle_cell_uses_extracted_input_source_and_no_old_output() -> None:
    cell = (PROJECT / "scripts" / "kaggle_cell_run_l22_v030.txt").read_text(
        encoding="utf-8"
    )
    assert 'PIPELINE_ROOT="/kaggle/input/continue-for-l22/aic_video_pipeline_v1"' in cell
    assert "python -m zipfile -e" not in cell
    assert "BUNDLE_NAME=" not in cell
    assert "scripts/kaggle_run_l22_from_v030.sh" in cell
    assert "AIC_PREVIOUS_RESULT_ROOT" in cell
    assert "export AIC_PREVIOUS_RESULT_ROOT" not in cell


def test_l22_bundle_builder_includes_runner_start_option() -> None:
    builder = (PROJECT / "scripts" / "build_kaggle_l22_v030_bundle.sh").read_text(
        encoding="utf-8"
    )
    runner = (PROJECT / "scripts" / "kaggle_archive_runner.py").read_text(
        encoding="utf-8"
    )
    assert "aic_video_pipeline_v1_l22_from_v030_kaggle.zip" in builder
    assert "archive.testzip()" in builder
    assert "kaggle_run_l22_from_v030.sh" in builder
    assert '"--start-at-video"' in runner
