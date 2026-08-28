import json
from pathlib import Path

from boldsearch_integration.cli import main
from test_publisher_contract import make_pipeline_output


def test_publish_command_reports_release(tmp_path: Path, capsys) -> None:
    data_root = make_pipeline_output(tmp_path / "data", dimension=4)

    assert main([
        "publish", "--data-root", str(data_root), "--video-id", "L21_V001",
        "--corpus-version", "test-v1", "--expected-vector-dim", "4",
        "--output-root", str(tmp_path / "public"), "--thumbnail-width", "80",
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["row_count"] == 2
    assert Path(result["release_root"]).is_dir()


def test_ingest_dry_run_never_requires_zilliz(tmp_path: Path, capsys) -> None:
    data_root = make_pipeline_output(tmp_path / "data", dimension=4)

    assert main([
        "ingest", "--data-root", str(data_root), "--video-id", "L21_V001",
        "--corpus-version", "test-v1", "--expected-vector-dim", "4",
        "--collection", "BoldSearchV1", "--dry-run",
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["row_count"] == 2
    assert result["collection"] == "BoldSearchV1"
