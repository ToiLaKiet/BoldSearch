"""
Business logic for the search module.

Migrated from the original Flask app.py — scoring, tokenization, and
data loading live here so the router stays thin.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

SHOTS_PATH = Path("data/shots.json")


# ── Data loading ─────────────────────────────────────────────────────


def load_shots() -> List[Dict[str, Any]]:
    """Read the shots catalogue from the JSON file."""
    with SHOTS_PATH.open(encoding="utf-8") as file:
        return json.load(file)["shots"]


# ── Scoring ──────────────────────────────────────────────────────────


def score_shot(
    shot: Dict[str, Any],
    query: str,
    task: str,
    modalities: List[str],
    objects: List[str],
    colors: List[str],
    temporal: str,
) -> Tuple[float, List[str]]:
    """
    Compute a relevance score for *shot* given the user's query signals.

    Returns ``(score, reasons)`` where *score* is clamped to [0, 1].
    """
    reasons: List[str] = []
    searchable = " ".join(
        [
            shot.get("title", ""),
            shot.get("description", ""),
            shot.get("transcript", ""),
            " ".join(shot.get("tags", [])),
            " ".join(shot.get("objects", [])),
            " ".join(shot.get("colors", [])),
            shot.get("location", ""),
        ]
    ).lower()

    # ── Text matching ────────────────────────────────────────────
    query_tokens = _tokenize(query)
    matches = [token for token in query_tokens if token in searchable]
    text_score = len(matches) / max(len(query_tokens), 1)
    if matches:
        reasons.append(f"Text match: {', '.join(matches[:4])}")

    # ── Object matching ──────────────────────────────────────────
    requested_objects = _normalize_list(objects)
    object_matches = [
        value
        for value in requested_objects
        if value in _normalize_list(shot.get("objects", []))
    ]
    object_score = len(object_matches) / max(len(requested_objects), 1)
    if object_matches:
        reasons.append(f"Objects: {', '.join(object_matches)}")

    # ── Color matching ───────────────────────────────────────────
    requested_colors = _normalize_list(colors)
    color_matches = [
        value
        for value in requested_colors
        if value in _normalize_list(shot.get("colors", []))
    ]
    color_score = len(color_matches) / max(len(requested_colors), 1)
    if color_matches:
        reasons.append(f"Colors: {', '.join(color_matches)}")

    # ── Temporal matching ────────────────────────────────────────
    temporal_score = 0.0
    if temporal:
        temporal_tokens = _tokenize(temporal)
        temporal_matches = [
            token for token in temporal_tokens if token in searchable
        ]
        temporal_score = len(temporal_matches) / max(len(temporal_tokens), 1)
        if temporal_matches:
            reasons.append(f"Temporal cue: {', '.join(temporal_matches[:3])}")

    # ── Composite score ──────────────────────────────────────────
    modality_bonus = 0.04 * len(
        set(modalities).intersection({"text", "image", "temporal", "objects"})
    )
    confidence = float(shot.get("confidence", 0.6))

    if task == "VKIS":
        raw_score = (
            text_score * 0.32
            + object_score * 0.24
            + color_score * 0.20
            + temporal_score * 0.12
            + confidence * 0.12
            + modality_bonus
        )
    else:
        raw_score = (
            text_score * 0.52
            + object_score * 0.14
            + color_score * 0.08
            + temporal_score * 0.10
            + confidence * 0.16
            + modality_bonus
        )

    if not reasons:
        reasons.append("Ranked by baseline confidence")

    return min(math.ceil(raw_score * 1000) / 1000, 1.0), reasons


# ── Helpers ──────────────────────────────────────────────────────────


def _tokenize(value: str) -> List[str]:
    return [
        token
        for token in re.split(r"[^a-zA-Z0-9]+", value.lower())
        if len(token) > 2
    ]


def _normalize_list(values: List[str]) -> List[str]:
    return [str(value).strip().lower() for value in values if str(value).strip()]
