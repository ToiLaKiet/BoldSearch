from __future__ import annotations

from pathlib import Path

import pytest

from unittest.mock import patch

from pydantic import ValidationError

from encoders.config import load_embedding_config
from encoders.loader import build_encoder


def _write_config(tmp_path: Path, contents: str) -> Path:
    path = tmp_path / "embedding.yaml"
    path.write_text(contents, encoding="utf-8")
    return path


def test_loads_selected_model_and_its_local_assets(tmp_path: Path):
    config = load_embedding_config(
        _write_config(
            tmp_path,
            """
embedding:
  model:
    name: beit3_base_itc
    type: beit3
    dimension: 768
    checkpoint_path: models/beit3/base.pth
    tokenizer_path: models/beit3/tokenizer.model
""",
        )
    )

    assert config.model.name == "beit3_base_itc"
    assert config.model.type == "beit3"
    assert config.model.checkpoint_path == "models/beit3/base.pth"


def test_rejects_beit3_without_a_tokenizer_path(tmp_path: Path):
    path = _write_config(
        tmp_path,
        """
embedding:
  model:
    name: beit3_base_itc
    type: beit3
    dimension: 768
    checkpoint_path: models/beit3/base.pth
""",
    )

    with pytest.raises(ValidationError, match="tokenizer_path"):
        load_embedding_config(path)


def test_builds_the_selected_static_adapter(tmp_path: Path):
    config = load_embedding_config(
        _write_config(
            tmp_path,
            """
embedding:
  device: cpu
  model:
    name: fg_clip_large
    type: fg_clip
    dimension: 768
""",
        )
    )

    with patch("encoders.fg_clip.FGClipEncoder") as encoder_class:
        encoder = build_encoder(config)

    assert encoder is encoder_class.return_value
    encoder_class.assert_called_once_with(device="cpu")
