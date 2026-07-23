from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    json.loads(temporary.read_text(encoding="utf-8"))
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


class NpyVectorStore:
    def __init__(self, root: Path, model_version: str) -> None:
        self.root, self.model_version = root, model_version

    def path_for(self, video_id: str, frame_id: str) -> Path:
        return self.root / video_id / f"{frame_id}.npy"

    def exists(self, video_id: str, frame_id: str) -> bool:
        return self.path_for(video_id, frame_id).is_file()

    def put(self, video_id: str, frame_id: str, vector: np.ndarray) -> Path:
        value = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(value))
        if value.ndim != 1 or not np.isfinite(value).all() or norm == 0:
            raise ValueError("embedding must be a finite non-zero vector")
        value = value / norm
        path = self.path_for(video_id, frame_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".npy.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, value)
        loaded = np.load(temporary)
        if loaded.dtype != np.float32 or loaded.ndim != 1:
            raise ValueError("stored embedding validation failed")
        os.replace(temporary, path)
        return path

    def get(self, path: str | Path, dimension: int | None = None) -> np.ndarray:
        value = np.load(path)
        if value.dtype != np.float32 or value.ndim != 1 or not np.isfinite(value).all():
            raise ValueError(f"invalid embedding: {path}")
        if dimension is not None and len(value) != dimension:
            raise ValueError(f"embedding dimension mismatch: {path}")
        norm = float(np.linalg.norm(value))
        if norm == 0:
            raise ValueError(f"zero embedding: {path}")
        return value / norm
