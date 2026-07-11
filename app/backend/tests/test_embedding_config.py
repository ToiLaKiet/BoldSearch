from __future__ import annotations

from pathlib import Path

import pytest

from unittest.mock import patch

from encoders.config import ConfigError, build_encoder, load_embedding_config


def _write_config(tmp_path: Path, contents: str) -> Path:
    path = tmp_path / "embedding.yaml"
    path.write_text(contents, encoding="utf-8")
    return path


def test_loads_selected_supported_model_and_its_local_assets(tmp_path: Path):
    config = load_embedding_config(
        _write_config(
            tmp_path,
            """
embedding:
  selected_model: beit3_base_itc
  models:
    beit3_base_itc:
      status: supported
      adapter: beit3
      assets:
        checkpoint_path: models/beit3/base.pth
        tokenizer_path: models/beit3/tokenizer.model
      dimension: 768
""",
        )
    )

    assert config.selected.name == "beit3_base_itc"
    assert config.selected.adapter == "beit3"
    assert config.selected.assets["checkpoint_path"] == "models/beit3/base.pth"


def test_rejects_a_candidate_as_the_selected_model(tmp_path: Path):
    path = _write_config(
        tmp_path,
        """
embedding:
  selected_model: beit3_large_itc
  models:
    beit3_large_itc:
      status: candidate
      adapter: beit3
      dimension: 1024
""",
    )

    with pytest.raises(ConfigError, match="not supported"):
        load_embedding_config(path)


def test_builds_the_selected_static_adapter(tmp_path: Path):
    config = load_embedding_config(
        _write_config(
            tmp_path,
            """
embedding:
  selected_model: fg_clip_large
  device: cpu
  models:
    fg_clip_large:
      status: supported
      adapter: fg_clip
      dimension: 768
      assets: {}
""",
        )
    )

    with patch("encoders.fg_clip.FGClipEncoder") as encoder_class:
        encoder = build_encoder(config)

    assert encoder is encoder_class.return_value
    encoder_class.assert_called_once_with(device="cpu")
