from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """Settings read from the environment or a local `.env`."""

    # ── API ──────────────────────────────────────────────────────
    SYSTEM_NAME: str = "BoldSearcher"
    API_PREFIX: str = "/api"

    # ── Server ───────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── Vector store ─────────────────────────────────────────────
    VECTOR_STORE_PROVIDER: Literal["qdrant", "milvus"] = "qdrant"
    VECTOR_STORE_COLLECTION: str = "keyframes"
    QDRANT_URL: str = "http://localhost:6333"
    MILVUS_URI: str = "http://localhost:19530"

    # ── Config files ─────────────────────────────────────────────
    VECTOR_STORE_CONFIG_PATH: str = "config/vector_store.yaml"
    EMBEDDING_CONFIG_PATH: str = "config/embedding.yaml"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @classmethod
    def load(cls, **overrides: Any) -> Self:
        """Merge environment settings with vector-store decisions from YAML."""
        settings = cls(**overrides)
        path = Path(settings.VECTOR_STORE_CONFIG_PATH)
        if not path.is_absolute():
            path = Path(__file__).parent / path

        with path.open(encoding="utf-8") as stream:
            vector_store = yaml.safe_load(stream)["vector_store"]

        if "url" in vector_store:
            raise ValueError(f"{path}: 'url' must come from environment settings")

        values = settings.model_dump()
        values.update(
            VECTOR_STORE_PROVIDER=vector_store["type"],
            VECTOR_STORE_COLLECTION=vector_store["collection"],
        )
        return cls(_env_file=None, **values)


app_config = AppConfig.load()
