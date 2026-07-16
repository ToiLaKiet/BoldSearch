"""TDD for app_config. `_env_file=None` keeps a real .env out of the results."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app_config import AppConfig


def test_defaults_apply_when_nothing_is_set():
    config = AppConfig(_env_file=None)

    assert config.SYSTEM_NAME == "BoldSearcher"
    assert config.API_PREFIX == "/api"
    assert config.HOST == "0.0.0.0"
    assert config.PORT == 8000
    assert config.VECTOR_STORE_PROVIDER == "qdrant"
    assert config.VECTOR_STORE_COLLECTION == "keyframes"
    assert config.QDRANT_URL == "http://localhost:6333"
    assert config.MILVUS_URI == "http://localhost:19530"
    assert config.VECTOR_STORE_CONFIG_PATH == "config/vector_store.yaml"
    assert config.EMBEDDING_CONFIG_PATH == "config/embedding.yaml"


def test_port_from_env_is_coerced_to_int(monkeypatch):
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
    """Both provider URLs remain available when switching the selected provider."""
    monkeypatch.setenv("QDRANT_URL", "https://cluster.qdrant.io:6333")
    monkeypatch.setenv("MILVUS_URI", "./milvus_demo.db")

    config = AppConfig(_env_file=None)

    assert config.QDRANT_URL == "https://cluster.qdrant.io:6333"
    assert config.MILVUS_URI == "./milvus_demo.db"


def test_unknown_vector_store_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE_PROVIDER", "unknown")

    with pytest.raises(ValidationError):
        AppConfig(_env_file=None)


def test_load_merges_yaml_decisions_into_app_config(tmp_path):
    path = tmp_path / "vector_store.yaml"
    path.write_text(
        "vector_store:\n"
        "  type: milvus\n"
        "  collection: video-keyframes\n"
        "  metric: cosine\n",
        encoding="utf-8",
    )

    config = AppConfig.load(
        _env_file=None,
        VECTOR_STORE_CONFIG_PATH=str(path),
        MILVUS_URI="./milvus.db",
    )

    assert config.VECTOR_STORE_PROVIDER == "milvus"
    assert config.VECTOR_STORE_COLLECTION == "video-keyframes"
    assert config.MILVUS_URI == "./milvus.db"


def test_yaml_provider_decision_overrides_environment(monkeypatch, tmp_path):
    path = tmp_path / "vector_store.yaml"
    path.write_text(
        "vector_store:\n"
        "  type: milvus\n"
        "  collection: video-keyframes\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("VECTOR_STORE_PROVIDER", "qdrant")
    monkeypatch.setenv("VECTOR_STORE_COLLECTION", "wrong-collection")

    config = AppConfig.load(
        _env_file=None,
        VECTOR_STORE_CONFIG_PATH=str(path),
    )

    assert config.VECTOR_STORE_PROVIDER == "milvus"
    assert config.VECTOR_STORE_COLLECTION == "video-keyframes"


def test_load_rejects_provider_url_in_yaml(tmp_path):
    path = tmp_path / "vector_store.yaml"
    path.write_text(
        "vector_store:\n"
        "  type: qdrant\n"
        "  collection: keyframes\n"
        "  url: http://wrong-source\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="url.*environment settings"):
        AppConfig.load(
            _env_file=None,
            VECTOR_STORE_CONFIG_PATH=str(path),
        )


def test_unknown_env_vars_are_ignored_not_rejected(monkeypatch):
    """A shared .env carries vars for other tools; they must not break boot."""
    monkeypatch.setenv("SOME_UNRELATED_TOOL_KEY", "x")

    AppConfig(_env_file=None)
