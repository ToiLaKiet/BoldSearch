from __future__ import annotations

from typing import Any

from app_config import AppConfig

_milvus_client: Any = None


def init_milvus(config: AppConfig) -> Any:
    """Create the Zilliz/Milvus client used by query endpoints."""
    global _milvus_client

    if not config.ZILLIZ_URI:
        return None

    try:
        from pymilvus import MilvusClient
    except ImportError as exc:
        raise RuntimeError("pymilvus is required for Zilliz Cloud access") from exc

    client_kwargs: dict[str, Any] = {"uri": config.ZILLIZ_URI}
    if config.ZILLIZ_TOKEN:
        client_kwargs["token"] = config.ZILLIZ_TOKEN
    _milvus_client = MilvusClient(**client_kwargs)
    return _milvus_client


def close_connections() -> None:
    global _milvus_client

    if _milvus_client is not None and hasattr(_milvus_client, "close"):
        _milvus_client.close()

    _milvus_client = None
