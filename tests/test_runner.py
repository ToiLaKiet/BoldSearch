from pathlib import Path

import pytest

from boldsearch_integration.runner import _pipeline_provenance, normalize_video_inputs


def test_normalize_video_inputs_accepts_mp4_and_rejects_duplicate_ids(tmp_path: Path) -> None:
    first = tmp_path / "L21_V001.mp4"
    second = tmp_path / "L21_V002.mp4"
    first.write_bytes(b"mp4")
    second.write_bytes(b"mp4")
    assert normalize_video_inputs([first, second]) == (first.resolve(), second.resolve())

    with pytest.raises(ValueError, match="duplicate"):
        normalize_video_inputs([first, first])


def test_normalize_video_inputs_requires_existing_mp4(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"\.mp4"):
        normalize_video_inputs([tmp_path / "L21_V001.mov"])


def test_pipeline_provenance_pins_config_checksum_and_revision(tmp_path: Path) -> None:
    config = tmp_path / "legacy_compatible.yaml"
    config.write_text("similarity:\n  threshold: 0.5\n", encoding="utf-8")
    provenance = _pipeline_provenance(tmp_path, config)
    assert provenance["pipeline"] == "aic_video_pipeline_v1"
    assert provenance["config_name"] == "legacy_compatible.yaml"
    assert provenance["config_sha256"].startswith("sha256:")
    assert provenance["pipeline_revision"] == "unknown"
