"""Tests for FastAPI-owned application resources."""

from __future__ import annotations

import asyncio

import pymilvus
import qdrant_client
from fastapi import FastAPI

import main
from vector_store.milvus import MilvusStore
from vector_store.qdrant import QdrantStore


class FakeClient:
    """Record whether lifespan closes its provider client."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        """Record client closure."""
        self.closed = True


def test_lifespan_owns_qdrant_store(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda **_: client)
    monkeypatch.setattr(main.app_config, "VECTOR_STORE_PROVIDER", "qdrant")
    test_app = FastAPI()

    async def run_lifespan() -> None:
        async with main.lifespan(test_app):
            assert isinstance(test_app.state.vector_store, QdrantStore)
            assert not client.closed

    asyncio.run(run_lifespan())

    assert client.closed


def test_lifespan_owns_milvus_store(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(pymilvus, "MilvusClient", lambda **_: client)
    monkeypatch.setattr(main.app_config, "VECTOR_STORE_PROVIDER", "milvus")
    test_app = FastAPI()

    async def run_lifespan() -> None:
        async with main.lifespan(test_app):
            assert isinstance(test_app.state.vector_store, MilvusStore)
            assert not client.closed

    asyncio.run(run_lifespan())

    assert client.closed
