"""TDD for app_config: reading settings from the environment.

Every AppConfig here passes _env_file=None so a developer's real .env can never
change the outcome of a test run.
"""

from __future__ import annotations

from app_config import API_PREFIX, SYSTEM_NAME, AppConfig


def test_constants_are_not_overridable_by_env(monkeypatch):
    """The frontend pins /api; an env var must not be able to move it."""
    monkeypatch.setenv("API_PREFIX", "/v2")
    monkeypatch.setenv("SYSTEM_NAME", "Other")

    AppConfig(_env_file=None)

    assert API_PREFIX == "/api"
    assert SYSTEM_NAME == "BoldSearcher"


def test_defaults_apply_when_nothing_is_set():
    config = AppConfig(_env_file=None)

    assert config.HOST == "0.0.0.0"
    assert config.PORT == 8000
    assert config.QDRANT_URL == "http://localhost:6333"
    assert config.MILVUS_URI == "http://localhost:19530"
    assert config.VECTOR_STORE_CONFIG_PATH == "config/vector_store.yaml"
    assert config.EMBEDDING_CONFIG_PATH == "config/embedding.yaml"


def test_data_dir_is_absolute_and_exists():
    """search/service.py reads it on every request, from any working directory."""
    config = AppConfig(_env_file=None)

    assert config.DATA_DIR.is_absolute()
    assert (config.DATA_DIR / "shots.json").exists()


def test_port_from_env_is_coerced_to_int(monkeypatch):
    """Env vars arrive as strings; uvicorn needs a real int."""
    monkeypatch.setenv("PORT", "8080")

    config = AppConfig(_env_file=None)

    assert config.PORT == 8080


def test_env_vars_override_the_defaults(monkeypatch):
    monkeypatch.setenv("MILVUS_URI", "./milvus_demo.db")
    monkeypatch.setenv("VECTOR_STORE_CONFIG_PATH", "config/vector_store.milvus.yaml")

    config = AppConfig(_env_file=None)

    assert config.MILVUS_URI == "./milvus_demo.db"
    assert config.VECTOR_STORE_CONFIG_PATH == "config/vector_store.milvus.yaml"


def test_both_provider_urls_coexist(monkeypatch):
    """Benchmarking switches provider via the yaml; both urls stay configured."""
    monkeypatch.setenv("QDRANT_URL", "https://cluster.qdrant.io:6333")
    monkeypatch.setenv("MILVUS_URI", "./milvus_demo.db")

    config = AppConfig(_env_file=None)

    assert config.QDRANT_URL == "https://cluster.qdrant.io:6333"
    assert config.MILVUS_URI == "./milvus_demo.db"


def test_unknown_env_vars_are_ignored_not_rejected(monkeypatch):
    """A shared .env may carry vars for other tools; they must not break boot."""
    monkeypatch.setenv("SOME_UNRELATED_TOOL_KEY", "x")

    AppConfig(_env_file=None)
