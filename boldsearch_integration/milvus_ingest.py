from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
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
    retries: int = 2,
    progress_path: Path | None = None,
) -> int:
    """Validate and idempotently upsert rows in bounded batches.

    The function deliberately knows only the visual modality. A caller must
    create a schema with the matching fields before invoking it.
    """
    if not collection_name or batch_size <= 0 or retries < 0:
        raise ValueError("collection_name, positive batch_size, and non-negative retries are required")
    materialized = list(rows)
    if not materialized:
        return 0
    _validate_rows(materialized)
    acknowledged_ids = _load_progress(progress_path, collection_name)
    pending = [row for row in materialized if row["id"] not in acknowledged_ids]
    total = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                response = client.upsert(collection_name=collection_name, data=batch)
                acknowledged = _ack_count(response)
                if acknowledged is not None and acknowledged != len(batch):
                    raise RuntimeError(
                        f"Milvus acknowledged {acknowledged}/{len(batch)} rows"
                    )
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt == retries:
                    raise RuntimeError(
                        f"Milvus upsert failed after {retries + 1} attempts"
                    ) from exc
        if last_error is not None:
            raise RuntimeError("Milvus upsert failed") from last_error
        acknowledged_ids.update(row["id"] for row in batch)
        total += len(batch)
        _save_progress(progress_path, collection_name, acknowledged_ids)
    return total


def _load_progress(path: Path | None, collection_name: str) -> set[int]:
    if path is None or not path.is_file():
        return set()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if value.get("schema_version") != "1.0" or value.get("collection") != collection_name:
        return set()
    row_ids = value.get("row_ids")
    if not isinstance(row_ids, list) or not all(isinstance(item, int) for item in row_ids):
        return set()
    return set(row_ids)


def _save_progress(path: Path | None, collection_name: str, row_ids: set[int]) -> None:
    if path is None:
        return
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump({
            "schema_version": "1.0",
            "collection": collection_name,
            "row_ids": sorted(row_ids),
        }, handle)
        handle.write("\n")
        handle.flush()
    os.replace(temporary, path)


def ingest_collection(
    client: MilvusUpsertClient,
    collection_name: str,
    rows: Iterable[dict[str, Any]],
    *,
    expected_vector_dim: int = 1024,
    batch_size: int = 256,
    retries: int = 2,
    progress_path: Path | None = None,
) -> int:
    """Validate the remote schema before sending any mutation."""
    description = client.describe_collection(collection_name)
    validate_collection_schema(description, expected_vector_dim=expected_vector_dim)
    return ingest_rows(
        client, collection_name, rows, batch_size=batch_size,
        retries=retries, progress_path=progress_path,
    )
