from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .publisher import load_kept_frames


class MilvusUpsertClient(Protocol):
    def describe_collection(self, collection_name: str) -> Mapping[str, Any]:
        ...

    def upsert(self, *, collection_name: str, data: list[dict[str, Any]]) -> Any:
        ...


def validate_collection_schema(
    description: Mapping[str, Any], *, expected_vector_dim: int = 1024
) -> set[str]:
    """Fail closed unless a collection can store the V1 visual projection."""
    if expected_vector_dim <= 0 or not isinstance(description, Mapping):
        raise ValueError("invalid collection description")
    fields = description.get("fields")
    if not isinstance(fields, list):
        raise ValueError("collection schema has no fields")
    names: set[str] = set()
    primary_keys: set[str] = set()
    visual_dim: int | None = None
    for field in fields:
        if not isinstance(field, Mapping):
            continue
        name = field.get("name") or field.get("field_name")
        if not isinstance(name, str) or not name:
            continue
        names.add(name)
        if field.get("is_primary") or field.get("is_primary_key"):
            primary_keys.add(name)
        if name == "visual_embedding":
            params = field.get("params")
            if isinstance(params, Mapping) and params.get("dim") is not None:
                visual_dim = int(params["dim"])
    required = {"id", "video_id", "frame_id", "shot_id", "visual_embedding"}
    missing = required - names
    if missing:
        raise ValueError(f"collection schema missing fields: {sorted(missing)}")
    if "id" not in primary_keys:
        raise ValueError("collection schema must mark id as primary key")
    if visual_dim != expected_vector_dim:
        raise ValueError(
            f"visual_embedding dim mismatch: expected {expected_vector_dim}, got {visual_dim}"
        )
    return names


def stable_primary_key(corpus_version: str, video_id: str, frame_id: int) -> int:
    """Return a deterministic positive int64 key for one published frame."""
    if not corpus_version or not video_id or frame_id < 0:
        raise ValueError("invalid primary-key inputs")
    digest = hashlib.sha256(
        f"{corpus_version}\0{video_id}\0{frame_id}".encode("utf-8")
    ).digest()
    return (int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)) or 1


def build_milvus_rows(
    *,
    data_root: Path,
    video_ids: Iterable[str],
    corpus_version: str,
    expected_vector_dim: int = 1024,
    thumbnail_base: str = "/keyframes",
) -> list[dict[str, Any]]:
    """Project validated V1 KEPT artifacts into a visual Milvus row list."""
    if (
        not corpus_version
        or re.fullmatch(r"/[A-Za-z0-9_-]*(?:/[A-Za-z0-9_-]+)*", thumbnail_base) is None
    ):
        raise ValueError("corpus_version and thumbnail_base are required")
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for video_id in sorted(set(video_ids)):
        for frame in load_kept_frames(
            data_root, video_id, expected_vector_dim=expected_vector_dim,
        ):
            vector = np.load(frame.vector_path, allow_pickle=False)
            row_id = stable_primary_key(corpus_version, video_id, frame.frame_id)
            if row_id in seen:
                raise ValueError(f"duplicate generated primary key: {row_id}")
            seen.add(row_id)
            rows.append({
                "id": row_id,
                "video_id": video_id,
                "frame_id": frame.frame_id,
                "shot_id": frame.shot_id,
                "visual_embedding": vector.astype(np.float32, copy=False).tolist(),
                "thumbnail": (
                    f"{thumbnail_base.rstrip('/')}/{video_id}/{frame.frame_id}.webp"
                ),
                "corpus_version": corpus_version,
            })
    if not rows:
        raise ValueError("no KEPT frames to index")
    return rows


def _validate_rows(rows: list[dict[str, Any]]) -> None:
    ids: set[int] = set()
    dimensions: set[int] = set()
    for row in rows:
        row_id = row.get("id")
        if not isinstance(row_id, int) or row_id <= 0:
            raise ValueError("row id must be a positive integer")
        if row_id in ids:
            raise ValueError(f"duplicate row id: {row_id}")
        ids.add(row_id)
        vector = np.asarray(row.get("visual_embedding"), dtype=np.float32)
        if vector.ndim != 1 or not vector.size:
            raise ValueError(f"row {row_id}: visual_embedding must be a vector")
        if not np.isfinite(vector).all():
            raise ValueError(f"row {row_id}: visual_embedding contains non-finite values")
        dimensions.add(int(vector.shape[0]))
        norm = float(np.linalg.norm(vector))
        if abs(norm - 1.0) > 1e-3:
            raise ValueError(f"row {row_id}: visual_embedding is not normalized")
        if not isinstance(row.get("video_id"), str) or not row["video_id"]:
            raise ValueError(f"row {row_id}: video_id is required")
        try:
            frame_id = int(row["frame_id"])
            shot_id = int(row["shot_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"row {row_id}: frame_id and shot_id must be integers") from exc
        if frame_id < 0 or shot_id < 0:
            raise ValueError(f"row {row_id}: frame_id and shot_id must be non-negative")
    if len(dimensions) > 1:
        raise ValueError("visual_embedding dimensions are inconsistent")


def _ack_count(response: Any) -> int | None:
    if isinstance(response, Mapping):
        for key in ("upsert_count", "insert_count", "count"):
            if key in response:
                return int(response[key])
    for key in ("upsert_count", "insert_count", "count"):
        value = getattr(response, key, None)
        if value is not None:
            return int(value)
    return None


def ingest_rows(
    client: MilvusUpsertClient,
    collection_name: str,
    rows: Iterable[dict[str, Any]],
    *,
    batch_size: int = 256,
) -> int:
    """Validate and idempotently upsert rows in bounded batches.

    The function deliberately knows only the visual modality. A caller must
    create a schema with the matching fields before invoking it.
    """
    if not collection_name or batch_size <= 0:
        raise ValueError("collection_name and positive batch_size are required")
    materialized = list(rows)
    if not materialized:
        return 0
    _validate_rows(materialized)
    total = 0
    for start in range(0, len(materialized), batch_size):
        batch = materialized[start:start + batch_size]
        response = client.upsert(collection_name=collection_name, data=batch)
        acknowledged = _ack_count(response)
        if acknowledged is not None and acknowledged != len(batch):
            raise RuntimeError(
                f"Milvus acknowledged {acknowledged}/{len(batch)} rows"
            )
        total += len(batch)
    return total


def ingest_collection(
    client: MilvusUpsertClient,
    collection_name: str,
    rows: Iterable[dict[str, Any]],
    *,
    expected_vector_dim: int = 1024,
    batch_size: int = 256,
) -> int:
    """Validate the remote schema before sending any mutation."""
    description = client.describe_collection(collection_name)
    validate_collection_schema(description, expected_vector_dim=expected_vector_dim)
    return ingest_rows(client, collection_name, rows, batch_size=batch_size)
