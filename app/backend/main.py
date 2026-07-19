"""
BoldSearcher — FastAPI entry point.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from logging import getLogger

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app_config import app_config
from asr.chunkformer import CHECKPOINT, REVISION, ChunkFormerTranscriber

# ── Module routers ───────────────────────────────────────────────────
from search.router import router as search_router
from ocr.router import router as ocr_router
from asr.router import router as asr_router
from object_detection.router import router as object_detection_router
from embedding.router import router as embedding_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ASR transcriber on startup and release it on shutdown."""
    try:
        app.state.asr_transcriber = ChunkFormerTranscriber(app_config.ASR_DEVICE)
        logger.info(
            "ASR transcriber configured (checkpoint=%s, revision=%s)",
            CHECKPOINT,
            REVISION,
        )
    except Exception as exc:
        logger.warning("ASR transcriber not available: %s", exc)
        app.state.asr_transcriber = None

    yield
    app.state.asr_transcriber = None


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

# ── Global health endpoint ───────────────────────────────────

@app.get(f"{app_config.API_PREFIX}/health", tags=["system"])
async def health():
    return {"status": "ok", "system": app_config.SYSTEM_NAME}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=app_config.HOST, port=app_config.PORT, reload=True)
