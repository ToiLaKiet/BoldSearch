"""Load and validate the selected encoder from the embedding YAML file."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class FGClipConfig(BaseModel):
    """Runtime inputs needed by the current FG-CLIP adapter."""

    model_config = ConfigDict(extra="forbid")

    name: str
    type: Literal["fg_clip"]
    dimension: PositiveInt


class Beit3Config(BaseModel):
    """Runtime inputs needed by the local, DVC-managed BEiT-3 adapter."""

    model_config = ConfigDict(extra="forbid")

    name: str
    type: Literal["beit3"]
    dimension: PositiveInt
    checkpoint_path: str
    tokenizer_path: str


ModelConfig = Annotated[
    FGClipConfig | Beit3Config,
    Field(discriminator="type"),
]


class EmbeddingConfig(BaseModel):
    """Validated embedding runtime settings selected from the YAML file."""

    model_config = ConfigDict(extra="forbid")

    device: Literal["cpu", "cuda", "mps"] | None = None
    model: ModelConfig


def load_embedding_config(path: str | Path) -> EmbeddingConfig:
    """Read YAML and delegate schema and semantic validation to Pydantic."""
    with Path(path).open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    return EmbeddingConfig.model_validate(document["embedding"])


