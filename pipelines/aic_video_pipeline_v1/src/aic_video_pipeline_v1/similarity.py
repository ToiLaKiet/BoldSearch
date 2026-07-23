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
    for frame in frames:
        if frame.final_status != "DUPLICATE":
            continue
        for text in (frame.frame_path, frame.vector_path):
            if text:
                Path(text).unlink(missing_ok=True)
        frame.frame_path = None
        frame.vector_path = None
