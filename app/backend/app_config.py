from __future__ import annotations


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
    QDRANT_URL: str = "http://localhost:6333"
    MILVUS_URI: str = "http://localhost:19530"

    # ── ASR ──────────────────────────────────────────────────────
    ASR_DEVICE: str = "cpu"

    # ── Config files ─────────────────────────────────────────────
    VECTOR_STORE_CONFIG_PATH: str = "config/vector_store.yaml"
    EMBEDDING_CONFIG_PATH: str = "config/embedding.yaml"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


app_config = AppConfig()
