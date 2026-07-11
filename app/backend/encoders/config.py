"""Load and validate the selected encoder from the embedding YAML file."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, PositiveInt, model_validator


class ModelConfig(BaseModel):
    """Runtime inputs needed to construct one supported encoder adapter."""

    model_config = ConfigDict(extra="forbid")

    adapter: Literal["fg_clip", "beit3"]
    dimension: PositiveInt
    checkpoint_path: str | None = None
    tokenizer_path: str | None = None

    @model_validator(mode="after")
    def require_local_beit3_assets(self) -> ModelConfig:
        """Require the two DVC-managed paths needed by the BEiT-3 adapter."""
        if self.adapter == "beit3" and not (
            self.checkpoint_path and self.tokenizer_path
        ):
            raise ValueError("BEiT-3 requires checkpoint_path and tokenizer_path")
        return self


class EmbeddingConfig(BaseModel):
    """Validated embedding runtime settings selected from the YAML file."""

    model_config = ConfigDict(extra="forbid")

    selected_model: str
    device: Literal["cpu", "cuda", "mps"] | None = None
    models: dict[str, ModelConfig]

    @model_validator(mode="after")
    def require_declared_selection(self) -> EmbeddingConfig:
        """Reject selections that do not name a declared model entry."""
        if self.selected_model not in self.models:
            raise ValueError("selected_model must name an entry in models")
        return self

    @property
    def selected(self) -> ModelConfig:
        """Return the one model selected for this embedding run."""
        return self.models[self.selected_model]


def load_embedding_config(path: str | Path) -> EmbeddingConfig:
    """Read YAML and delegate schema and semantic validation to Pydantic."""
    with Path(path).open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    return EmbeddingConfig.model_validate(document["embedding"])


def build_encoder(config: EmbeddingConfig) -> Any:
    """Construct the selected known adapter without dynamic imports or a factory."""
    if config.selected.adapter == "fg_clip":
        from encoders.fg_clip import FGClipEncoder

        return FGClipEncoder(device=config.device)

    from encoders.beit3 import Beit3Encoder

    return Beit3Encoder(
        checkpoint_path=config.selected.checkpoint_path,
        tokenizer_path=config.selected.tokenizer_path,
        device=config.device,
    )
