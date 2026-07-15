"""Environment-sourced application settings.

The single place that reads the environment. Holds what varies per machine
(connection targets, server binding) and which config file to load. Design
decisions -- which provider, which model, which metric -- live in the yaml
files under `config/`, so they stay in version control and get validated by
their own schema.

Infrastructure packages (`encoders/`, and `vector_store/` later) never import
this module: they take plain arguments. That is what keeps them testable
without an environment.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parent

# Constants, not settings: they do not vary per machine, and the frontend pins
# `/api` in vite.config.js and App.jsx. Letting an env var move the API prefix
# would break the UI silently, so these stay outside AppConfig where nothing
# can override them. They live here only to keep one source for the ~10 call
# sites in main.py and search/router.py.
SYSTEM_NAME = "BoldSearcher"
API_PREFIX = "/api"


class AppConfig(BaseSettings):
    """Settings read from the environment or a local `.env`."""

    # ── Server ───────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── Paths ────────────────────────────────────────────────────
    DATA_DIR: Path = _BACKEND_DIR / "data"

    # ── Vector store ─────────────────────────────────────────────
    QDRANT_URL: str = "http://localhost:6333"
    MILVUS_URI: str = "http://localhost:19530"

    # ── Config files ─────────────────────────────────────────────
    VECTOR_STORE_CONFIG_PATH: str = "config/vector_store.yaml"
    EMBEDDING_CONFIG_PATH: str = "config/embedding.yaml"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


app_config = AppConfig()
