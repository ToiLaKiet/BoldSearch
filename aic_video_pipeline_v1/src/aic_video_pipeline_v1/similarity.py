from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .models import FrameRecord
from .storage import NpyVectorStore


@dataclass(frozen=True)
class SimilaritySummary:
    compared: int
    kept: int
    duplicate: int
    threshold: float


class OnlineRepresentativeSimilarity:
    """Classify normalized vectors without materializing DUPLICATE artifacts.

    A representative is kept independently for every shot.  The input video is
    processed in frame order, so the stored representative is always the latest
    KEPT frame in that shot.
    """

    def __init__(self, threshold: float) -> None:
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("similarity threshold must be between -1 and 1")
        self.threshold = threshold
        self.representatives: dict[str, tuple[str, np.ndarray]] = {}

    def restore(self, shot_id: str, frame_id: str, vector: np.ndarray) -> None:
        value = self._normalized(vector)
        self.representatives[shot_id] = (frame_id, value)

    def classify(self, shot_id: str, frame_id: str,
                 vector: np.ndarray) -> tuple[str, str | None, float | None]:
        value = self._normalized(vector)
        representative = self.representatives.get(shot_id)
        if representative is None:
            self.representatives[shot_id] = (frame_id, value)
            return "KEPT", None, None
        score = float(np.dot(representative[1], value))
        if score >= self.threshold:
            return "DUPLICATE", representative[0], score
        self.representatives[shot_id] = (frame_id, value)
        return "KEPT", None, score

    @staticmethod
    def _normalized(vector: np.ndarray) -> np.ndarray:
        value = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(value))
        if value.ndim != 1 or not np.isfinite(value).all() or norm == 0:
            raise ValueError("embedding must be a finite non-zero vector")
        return value / norm


def apply_representative_similarity(frames: list[FrameRecord], store: NpyVectorStore,
                                    threshold: float) -> SimilaritySummary:
    """Keep the first frame and compare later frames to the latest KEPT frame.

    The representative is reset at every shot boundary. A DUPLICATE never
    becomes the representative, so long runs of near-identical frames collapse
    instead of producing one kept frame per pair.
    """
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("similarity threshold must be between -1 and 1")
    ordered = sorted((frame for frame in frames if frame.vector_path),
                     key=lambda frame: (frame.shot_id, frame.frame_index))
    representatives: dict[str, tuple[FrameRecord, np.ndarray]] = {}
    dimensions: dict[str, int] = {}
    for frame in ordered:
        representative = representatives.get(frame.shot_id)
        if representative is None:
            frame.final_status, frame.representative_frame_id, frame.similarity_score = "KEPT", None, None
            vector = store.get(frame.vector_path)
            representatives[frame.shot_id] = (frame, vector)
            dimensions[frame.shot_id] = len(vector)
            continue
        current = store.get(frame.vector_path, dimensions[frame.shot_id])
        score = float(np.dot(representative[1], current))
        frame.similarity_score = score
        if score >= threshold:
            frame.final_status = "DUPLICATE"
            frame.representative_frame_id = representative[0].frame_id
        else:
            frame.final_status, frame.representative_frame_id = "KEPT", None
            representatives[frame.shot_id] = (frame, current)
    kept = sum(frame.final_status == "KEPT" for frame in frames)
    duplicate = sum(frame.final_status == "DUPLICATE" for frame in frames)
    return SimilaritySummary(len(ordered), kept, duplicate, threshold)


def remove_duplicate_artifacts(frames: list[FrameRecord]) -> None:
    """Delete DUPLICATE artifacts and remove their metadata records in-place."""
    for frame in frames:
        if frame.final_status != "DUPLICATE":
            continue
        for text in (frame.frame_path, frame.vector_path):
            if text:
                Path(text).unlink(missing_ok=True)
        frame.frame_path = None
        frame.vector_path = None
    frames[:] = [frame for frame in frames if frame.final_status == "KEPT"]
