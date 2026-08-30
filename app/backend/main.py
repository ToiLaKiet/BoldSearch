"""
BoldSearcher — FastAPI entry point.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app_config import app_config
from connections import close_connections, init_milvus
from encoders.loader import load_fg_clip_encoder
from search.object_index import load_object_index

from search.router import router as search_router


# ── App factory ──────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Starting {app_config.SYSTEM_NAME}...")
    app.state.object_index = load_object_index(app_config)
    print(f"Loaded object index with {len(app.state.object_index)} entries.")
    app.state.milvus_client = init_milvus(app_config)
    print(f"Connected to Zilliz at {app_config.ZILLIZ_URI}.")
    app.state.embedding_encoder = (
        load_fg_clip_encoder(app_config.FG_CLIP_DEVICE or None, app_config.HF_TOKEN)
        if app_config.LOAD_FG_CLIP_ON_STARTUP
        else None
    )
    try:
        yield
    finally:
        close_connections()


app = FastAPI(
    title=app_config.SYSTEM_NAME,
    description="Frame retrieval API backed by Zilliz hybrid search and CSV object metadata.",
    version="0.3.0",
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

# Keyframe media is stored outside the frontend bundle to avoid copying the
# corpus into Vite's production output.
app.mount(
    "/keyframes",
    StaticFiles(directory=app_config.KEYFRAMES_DIR, check_dir=False),
    name="keyframes",
)
app.mount(
    "/map-keyframes",
    StaticFiles(directory=app_config.KEYFRAME_MAP_DIR, check_dir=False),
    name="map-keyframes",
)

# ── Register routers ─────────────────────────────────────────

app.include_router(search_router, prefix=app_config.API_PREFIX)

# ── Global health endpoint ───────────────────────────────────

@app.get(f"{app_config.API_PREFIX}/health", tags=["system"])
async def health():
    return {"status": "ok", "system": app_config.SYSTEM_NAME}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=app_config.HOST, port=app_config.PORT, reload=True)
