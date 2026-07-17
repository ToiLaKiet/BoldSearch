"""
BoldSearcher — FastAPI entry point.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app_config import app_config
from vector_store.ports import VectorStore

# ── Module routers ───────────────────────────────────────────────────
from search.router import router as search_router
from ocr.router import router as ocr_router
from asr.router import router as asr_router
from object_detection.router import router as object_detection_router
from embedding.router import router as embedding_router
from vector.router import router as vector_router


# ── Application lifecycle ────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Own one configured vector-store resource for this worker."""
    store: VectorStore
    if app_config.VECTOR_STORE_PROVIDER == "qdrant":
        from qdrant_client import QdrantClient

        from vector_store.qdrant import QdrantStore

        client = QdrantClient(url=app_config.QDRANT_URL)
        store = QdrantStore(client, app_config.VECTOR_STORE_COLLECTION)
    else:
        from pymilvus import MilvusClient

        from vector_store.milvus import MilvusStore

        client = MilvusClient(
            uri=app_config.MILVUS_URI, token=app_config.MILVUS_TOKEN
        )
        store = MilvusStore(client, app_config.VECTOR_STORE_COLLECTION)

    app.state.vector_store = store
    try:
        yield
    finally:
        client.close()


# ── App factory ──────────────────────────────────────────────────────
app = FastAPI(
    title=app_config.SYSTEM_NAME,
    description="Interactive video shot retrieval system for AI Challenge.",
    version="0.2.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ─────────────────────────────────────────
# Each router already defines its own sub-prefix (e.g. /search, /ocr).
# We mount them all under /api so the full paths become:
#   /api/search/query, /api/ocr/extract, /api/asr/transcribe, etc.

app.include_router(search_router, prefix=app_config.API_PREFIX)
app.include_router(ocr_router, prefix=app_config.API_PREFIX)
app.include_router(asr_router, prefix=app_config.API_PREFIX)
app.include_router(object_detection_router, prefix=app_config.API_PREFIX)
app.include_router(embedding_router, prefix=app_config.API_PREFIX)
app.include_router(vector_router, prefix=app_config.API_PREFIX)

# ── Global health endpoint ───────────────────────────────────

@app.get(f"{app_config.API_PREFIX}/health", tags=["system"])
async def health():
    return {"status": "ok", "system": app_config.SYSTEM_NAME}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=app_config.HOST, port=app_config.PORT, reload=True)
