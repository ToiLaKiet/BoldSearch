from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class VectorStore(Protocol):
    """Stable storage contract for embeddings, independent of the backend."""

    model_version: str

    def path_for(self, video_id: str, frame_id: str) -> Path: ...
    def put(self, video_id: str, frame_id: str, vector: np.ndarray) -> Path: ...
    def get(self, path: str | Path, expected_dimension: int | None = None) -> np.ndarray: ...
    def exists(self, video_id: str, frame_id: str) -> bool: ...


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    json.loads(temporary.read_text(encoding="utf-8"))
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


class NpyFileVectorStore:
    def __init__(self, root: Path, model_version: str) -> None:
        self.root, self.model_version = root, model_version

    def path_for(self, video_id: str, frame_id: str) -> Path:
        return self.root / video_id / f"{frame_id}.npy"

    def exists(self, video_id: str, frame_id: str) -> bool:
        return self.path_for(video_id, frame_id).is_file()

    def put(self, video_id: str, frame_id: str, vector: np.ndarray) -> Path:
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim != 1 or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
            raise ValueError("vector must be a finite non-zero one-dimensional array")
        vector = vector / np.linalg.norm(vector)
        path = self.path_for(video_id, frame_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".npy.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, vector)
        loaded = np.load(temporary)
        if loaded.ndim != 1 or loaded.dtype != np.float32 or not np.isfinite(loaded).all():
            raise ValueError("vector validation failed")
        os.replace(temporary, path)
        return path

    def get(self, path: str | Path, expected_dimension: int | None = None) -> np.ndarray:
        raw = np.load(path)
        if raw.dtype != np.float32:
            raise ValueError(f"vector dtype must be float32: {path}")
        vector = raw.astype(np.float32, copy=False)
        if vector.ndim != 1 or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
            raise ValueError(f"invalid vector: {path}")
        if expected_dimension is not None and vector.shape[0] != expected_dimension:
            raise ValueError(f"vector dimension mismatch: {path}")
        return vector / np.linalg.norm(vector)
