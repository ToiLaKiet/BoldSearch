"""
Search router — shot retrieval, task listing, and submission endpoints.

Prefix: /api/search
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app_config import SYSTEM_NAME
from search import schema, service

router = APIRouter(prefix="/search", tags=["search"])


# ── GET /api/search/tasks ────────────────────────────────────────────


@router.get("/tasks", response_model=schema.TasksResponse)
async def get_tasks():
    """Return the list of supported retrieval tasks."""
    return schema.TasksResponse(
        system=SYSTEM_NAME,
        tasks=[
            schema.TaskInfo(
                id="KIS",
                name="Known Item Search",
                description="Find one known target shot from a textual description.",
                recommendedSignals=["caption", "transcript", "objects", "time"],
            ),
            schema.TaskInfo(
                id="VKIS",
                name="Visual Known Item Search",
                description=(
                    "Find one known target shot from visual cues such as "
                    "objects, colors, layout, or an image reference."
                ),
                recommendedSignals=["image", "objects", "colors", "temporal"],
            ),
        ],
    )


# ── GET /api/search/shots ───────────────────────────────────────────


@router.get("/shots", response_model=schema.ShotsResponse)
async def get_shots():
    """Return every shot in the catalogue."""
    return schema.ShotsResponse(shots=service.load_shots())


# ── POST /api/search/query ──────────────────────────────────────────


@router.post("/query", response_model=schema.SearchResponse)
async def search_query(body: schema.SearchRequest):
    """Score all shots against the supplied query signals and return ranked results."""
    task = body.task.upper() if body.task else "KIS"
    results: list[dict] = []

    for shot in service.load_shots():
        score, reasons = service.score_shot(
            shot=shot,
            query=body.query,
            task=task,
            modalities=body.modalities,
            objects=body.objects,
            colors=body.colors,
            temporal=body.temporal,
        )
        if score >= body.minConfidence:
            enriched = dict(shot)
            enriched["score"] = round(score, 3)
            enriched["reasons"] = reasons
            results.append(enriched)

    results.sort(key=lambda item: item["score"], reverse=True)

    return schema.SearchResponse(
        system=SYSTEM_NAME,
        task=task if task in {"KIS", "VKIS"} else "KIS",
        query=body.query,
        count=len(results),
        results=results,
    )


# ── POST /api/search/submit ─────────────────────────────────────────


@router.post("/submit", response_model=schema.SubmitResponse)
async def submit_shot(body: schema.SubmitRequest):
    """Submit a shot as the answer for the active task."""
    shot = next(
        (item for item in service.load_shots() if item["id"] == body.shotId),
        None,
    )
    if not shot:
        raise HTTPException(status_code=404, detail="Shot not found")

    return schema.SubmitResponse(
        status="accepted",
        system=SYSTEM_NAME,
        submission=schema.SubmissionDetail(
            shotId=shot["id"],
            videoId=shot["videoId"],
            timestamp=shot["start"],
        ),
    )
