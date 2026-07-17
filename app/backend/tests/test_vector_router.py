"""HTTP contract for the vector ingest and similarity-search surface."""

from __future__ import annotations

from typing import Sequence
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vector.router import router
from vector_store.schemas import SearchHit, VectorPoint

POINT_ID = UUID("eb41e5f5-4ae7-47d5-88d1-a4b27b39b24a")


class FakeStore:
    """Record neutral vector-store calls and return preset search hits."""

    def __init__(self) -> None:
        self.hits: list[SearchHit] = []
        self.ingested: list[VectorPoint] = []
        self.searches: list[tuple[list[float], int]] = []

    def ingest(self, points: Sequence[VectorPoint]) -> None:
        """Record points received from the HTTP adapter."""
        self.ingested.extend(points)

    def search(self, vector: Sequence[float], limit: int) -> list[SearchHit]:
        """Record the neutral query and return preset hits."""
        self.searches.append((list(vector), limit))
        return self.hits


@pytest.fixture
def store() -> FakeStore:
    """Return a fresh store test double."""
    return FakeStore()


@pytest.fixture
def client(store: FakeStore) -> TestClient:
    """Serve the vector router with a worker-owned store resource."""
    app = FastAPI()
    app.state.vector_store = store
    app.include_router(router, prefix="/api")
    return TestClient(app)


def test_ingest_maps_the_body_to_provider_neutral_points(client, store):
    response = client.post(
        "/api/vector/ingest",
        json={
            "points": [
                {
                    "id": str(POINT_ID),
                    "source_id": "L21_V01/001.png",
                    "vector": [1.0, 0.0],
                    "metadata": {"video_id": "L21_V01", "shot_id": "001"},
                }
            ]
        },
    )

    assert response.status_code == 200
    assert response.json() == {"ingested": 1}
    assert store.ingested == [
        VectorPoint(
            id=POINT_ID,
            source_id="L21_V01/001.png",
            vector=[1.0, 0.0],
            metadata={"video_id": "L21_V01", "shot_id": "001"},
        )
    ]


def test_ingest_rejects_an_empty_batch(client, store):
    response = client.post("/api/vector/ingest", json={"points": []})

    assert response.status_code == 422
    assert store.ingested == []


def test_ingest_rejects_a_point_without_a_vector(client, store):
    response = client.post(
        "/api/vector/ingest",
        json={
            "points": [
                {"id": str(POINT_ID), "source_id": "a.png", "vector": []},
            ]
        },
    )

    assert response.status_code == 422
    assert store.ingested == []


def test_search_similarity_maps_neutral_hits_to_the_http_response(client, store):
    store.hits = [
        SearchHit(
            id=POINT_ID,
            source_id="L21_V01/001.png",
            score=0.91,
            metadata={"video_id": "L21_V01", "shot_id": "001"},
        )
    ]

    response = client.post(
        "/api/vector/search-similarity",
        json={"query_vector": [1.0, 0.0], "top_k": 5},
    )

    assert response.status_code == 200
    assert store.searches == [([1.0, 0.0], 5)]
    assert response.json() == {
        "top_k": 5,
        "matches": [
            {
                "id": str(POINT_ID),
                "source_id": "L21_V01/001.png",
                "score": 0.91,
                "metadata": {"video_id": "L21_V01", "shot_id": "001"},
            }
        ],
    }


def test_search_similarity_defaults_top_k_without_reaching_the_store_twice(
    client, store
):
    response = client.post(
        "/api/vector/search-similarity",
        json={"query_vector": [1.0, 0.0]},
    )

    assert response.status_code == 200
    assert store.searches == [([1.0, 0.0], 20)]


@pytest.mark.parametrize(
    "body",
    [
        {"query_vector": [], "top_k": 5},
        {"query_vector": [1.0, 0.0], "top_k": 0},
        {"query_vector": [1.0, 0.0], "top_k": 201},
        {"top_k": 5},
    ],
)
def test_search_similarity_rejects_invalid_bodies(client, store, body):
    response = client.post("/api/vector/search-similarity", json=body)

    assert response.status_code == 422
    assert store.searches == []
