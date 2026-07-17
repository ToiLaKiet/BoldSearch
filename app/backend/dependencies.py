"""Resolve worker-owned resources for request handlers.

The only module that reads `app.state`, so routers stay unaware of how the
application lifespan builds and owns those resources.
"""

from __future__ import annotations

from fastapi import Request

from vector_store.ports import VectorStore


def get_vector_store(request: Request) -> VectorStore:
    """Return the vector store opened once by the application lifespan."""
    return request.app.state.vector_store
