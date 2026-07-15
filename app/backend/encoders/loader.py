"""Construct the encoder selected in the embedding config."""
from __future__ import annotations

from typing import Any

from encoders.config import EmbeddingConfig


def build_encoder(config: EmbeddingConfig) -> Any:
    """Instantiate the encoder matching ``config.model.type``."""
    builders = {
        "beit3": _build_beit3,
        "fg_clip": _build_fg_clip,
    }
    return builders[config.model.type](config)


def _build_beit3(config: EmbeddingConfig) -> Any:
    from encoders.beit3 import Beit3Encoder

    return Beit3Encoder(
        checkpoint_path=config.model.checkpoint_path,
        tokenizer_path=config.model.tokenizer_path,
        device=config.device,
    )


def _build_fg_clip(config: EmbeddingConfig) -> Any:
    from encoders.fg_clip import FGClipEncoder

    return FGClipEncoder(device=config.device)
