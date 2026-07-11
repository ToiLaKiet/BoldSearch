"""Load the explicit encoder selection from the embedding YAML file."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an embedding configuration cannot safely select a model."""


@dataclass(frozen=True)
class ModelConfig:
    """A configured model entry and its locally resolved asset locations."""

    name: str
    status: str
    adapter: str
    dimension: int
    assets: dict[str, str]


@dataclass(frozen=True)
class EmbeddingConfig:
    """The selected, runnable encoder configuration."""

    device: str | None
    selected: ModelConfig


def load_embedding_config(path: str | Path) -> EmbeddingConfig:
    """Load and validate one supported model selection from a YAML file."""
    source = Path(path)
    with source.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)

    if not isinstance(document, dict) or not isinstance(document.get("embedding"), dict):
        raise ConfigError("configuration must contain an 'embedding' mapping")

    embedding = document["embedding"]
    selected_name = embedding.get("selected_model")
    models = embedding.get("models")
    if not isinstance(selected_name, str) or not isinstance(models, dict):
        raise ConfigError("embedding.selected_model and embedding.models are required")

    raw_model = models.get(selected_name)
    if not isinstance(raw_model, dict):
        raise ConfigError(f"selected model '{selected_name}' is not declared")
    if raw_model.get("status") != "supported":
        raise ConfigError(f"selected model '{selected_name}' is not supported")

    adapter = raw_model.get("adapter")
    dimension = raw_model.get("dimension")
    assets = raw_model.get("assets", {})
    if adapter not in {"fg_clip", "beit3"}:
        raise ConfigError(f"selected model '{selected_name}' has an unknown adapter")
    if not isinstance(dimension, int) or dimension <= 0:
        raise ConfigError(f"selected model '{selected_name}' needs a positive dimension")
    if not isinstance(assets, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in assets.items()
    ):
        raise ConfigError(f"selected model '{selected_name}' has invalid assets")
    required_assets = {
        "fg_clip": set(),
        "beit3": {"checkpoint_path", "tokenizer_path"},
    }[adapter]
    if required_assets - assets.keys():
        missing = ", ".join(sorted(required_assets - assets.keys()))
        raise ConfigError(f"selected model '{selected_name}' is missing assets: {missing}")

    device = embedding.get("device")
    if device is not None and device not in {"cpu", "cuda", "mps"}:
        raise ConfigError("embedding.device must be cpu, cuda, mps, or omitted")

    return EmbeddingConfig(
        device=device,
        selected=ModelConfig(
            name=selected_name,
            status="supported",
            adapter=adapter,
            dimension=dimension,
            assets=assets,
        ),
    )


def build_encoder(config: EmbeddingConfig) -> Any:
    """Construct the selected known adapter without dynamic imports or a factory."""
    if config.selected.adapter == "fg_clip":
        from encoders.fg_clip import FGClipEncoder

        return FGClipEncoder(device=config.device)

    if config.selected.adapter == "beit3":
        from encoders.beit3 import Beit3Encoder

        return Beit3Encoder(
            checkpoint_path=config.selected.assets["checkpoint_path"],
            tokenizer_path=config.selected.assets["tokenizer_path"],
            device=config.device,
        )

    raise ConfigError(f"selected adapter '{config.selected.adapter}' is not runnable")
