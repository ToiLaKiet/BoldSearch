"""Shared search and ingest behavior for every vector-store provider."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import pytest

from vector_store.milvus import MilvusStore
from vector_store.ports import VectorStore
from vector_store.qdrant import QdrantStore
from vector_store.schemas import VectorPoint

VECTOR_SIZE = 2
QUERY = [1.0, 0.0]


def make_point(
    logical_id: str,
    vector: Sequence[float],
    **metadata: Any,
) -> VectorPoint:
    """Build deterministic provider-neutral test data."""
    return VectorPoint(
        id=uuid5(NAMESPACE_URL, logical_id),
        source_id=logical_id,
        vector=vector,
        metadata=metadata,
    )


def build_qdrant(points: Sequence[VectorPoint]):
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels

    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="contract",
        vectors_config=qmodels.VectorParams(
            size=VECTOR_SIZE,
            distance=qmodels.Distance.COSINE,
        ),
    )
    if points:
        client.upsert(
            collection_name="contract",
            points=[
                qmodels.PointStruct(
                    id=str(point.id),
                    vector=list(point.vector),
                    payload={
                        "source_id": point.source_id,
                        "metadata": dict(point.metadata),
                    },
                )
                for point in points
            ],
        )
    return client, QdrantStore(client, "contract")


def build_milvus(points: Sequence[VectorPoint], database: Path):
    from pymilvus import DataType, MilvusClient

    client = MilvusClient(uri=str(database))
    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=36)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=VECTOR_SIZE)
    schema.add_field("source_id", DataType.VARCHAR, max_length=2_048)
    schema.add_field("metadata", DataType.JSON)
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="FLAT",
        metric_type="COSINE",
    )
    client.create_collection(
        "contract",
        schema=schema,
        index_params=index_params,
    )
    if points:
        client.upsert(
            collection_name="contract",
            data=[
                {
                    "id": str(point.id),
                    "vector": list(point.vector),
                    "source_id": point.source_id,
                    "metadata": dict(point.metadata),
                }
                for point in points
            ],
        )
        client.flush("contract")
    return client, MilvusStore(client, "contract")


@pytest.fixture
def default_points() -> list[VectorPoint]:
    """Return points with a deterministic cosine ranking."""
    return [
        make_point("same", [1.0, 0.0], name="same"),
        make_point("orthogonal", [0.0, 1.0], name="orthogonal"),
        make_point("opposite", [-1.0, 0.0], name="opposite"),
    ]


@pytest.fixture(params=["milvus", "qdrant"])
def make_store(
    request,
    tmp_path: Path,
) -> Iterator[Callable[[Sequence[VectorPoint]], VectorStore]]:
    """Build a fresh embedded provider with optional pre-seeded points."""
    clients = []

    def build(points: Sequence[VectorPoint] = ()) -> VectorStore:
        if request.param == "qdrant":
            client, store = build_qdrant(points)
        else:
            database = tmp_path / f"contract-{len(clients)}.db"
            client, store = build_milvus(points, database)
        clients.append(client)
        return store

    yield build

    for client in clients:
        client.close()


def test_search_ranks_by_cosine_similarity(make_store, default_points):
    store = make_store(default_points)

    hits = store.search(QUERY, limit=3)

    assert [hit.source_id for hit in hits] == [
        "same",
        "orthogonal",
        "opposite",
    ]
    assert [hit.score for hit in hits] == sorted(
        [hit.score for hit in hits],
        reverse=True,
    )


def test_search_respects_limit(make_store, default_points):
    store = make_store(default_points)

    assert len(store.search(QUERY, limit=2)) == 2


def test_search_returns_empty_collection(make_store):
    store = make_store()

    assert store.search(QUERY, limit=1) == []


def test_search_returns_provider_neutral_hit(make_store, default_points):
    store = make_store(default_points)

    hit = store.search(QUERY, limit=1)[0]

    assert hit.id == default_points[0].id
    assert hit.source_id == "same"
    assert hit.metadata == {"name": "same"}


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_search_rejects_invalid_limit(make_store, limit):
    store = make_store()

    with pytest.raises(ValueError, match="limit must be a positive int"):
        store.search(QUERY, limit=limit)


def test_ingest_makes_points_searchable(make_store, default_points):
    store = make_store()

    store.ingest(default_points)

    hits = store.search(QUERY, limit=3)
    assert [hit.source_id for hit in hits] == [
        "same",
        "orthogonal",
        "opposite",
    ]


def test_ingest_remains_visible_after_search_has_loaded_collection(make_store):
    store = make_store()
    point = make_point("after-load", [1.0, 0.0], stage="ingested")

    assert store.search(QUERY, limit=1) == []

    store.ingest([point])

    hit = store.search(QUERY, limit=1)[0]
    assert hit.id == point.id
    assert hit.metadata == {"stage": "ingested"}


def test_ingest_empty_batch_is_a_noop(make_store):
    store = make_store()

    store.ingest([])

    assert store.search(QUERY, limit=1) == []


def test_ingest_same_id_replaces_existing_point(make_store):
    store = make_store()
    first = make_point("same-id", [1.0, 0.0], version="first")
    replacement = make_point("same-id", [0.0, 1.0], version="second")

    store.ingest([first])
    store.ingest([replacement])

    hits = store.search(QUERY, limit=10)
    assert len(hits) == 1
    assert hits[0].metadata == {"version": "second"}


def test_ingest_keeps_source_id_separate_from_metadata(make_store):
    store = make_store()
    point = make_point(
        "canonical",
        [1.0, 0.0],
        source_id="metadata-value",
    )

    store.ingest([point])

    hit = store.search(QUERY, limit=1)[0]
    assert hit.source_id == "canonical"
    assert hit.metadata == {"source_id": "metadata-value"}
