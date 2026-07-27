"""
Search router.

Public endpoints:
- POST /api/search/query
- POST /api/search/visual_query
- POST /api/search/submit
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app_config import app_config
from search import schema, service
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/query", response_model=schema.SearchResponse)
async def query(body: schema.QueryRequest, request: Request):
    """Run text/object hybrid search and enrich returned frames from objects.csv."""
    try:
        return service.run_text_query(
            body=body,
            config=app_config,
            milvus_client=getattr(request.app.state, "milvus_client", None),
            object_index=getattr(request.app.state, "object_index", None),
            embedding_encoder=getattr(request.app.state, "embedding_encoder", None),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("Failed to load FG-CLIP")
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/visual_query", response_model=schema.SearchResponse)
async def visual_query(body: schema.VisualQueryRequest, request: Request):
    """Run image-embedding hybrid search and enrich returned frames from objects.csv."""
    try:
        return service.run_visual_query(
            body=body,
            config=app_config,
            milvus_client=getattr(request.app.state, "milvus_client", None),
            object_index=getattr(request.app.state, "object_index", None),
            embedding_encoder=getattr(request.app.state, "embedding_encoder", None),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/submit", response_model=schema.SubmitResponse)
async def submit(body: schema.SubmitRequest):
    """Submit one retrieved frame as the selected answer."""
    try:
        return service.submit_frame(body, app_config)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
