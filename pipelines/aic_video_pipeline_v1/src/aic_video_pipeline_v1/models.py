from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class FrameRecord:
    video_id: str
    frame_id: str
    frame_index: int
    timestamp_ms: int
    shot_id: str
    batch_id: str = ""
    batch_position: int = 0
    preliminary_status: str = "KEPT"
    mapping_status: str = "PENDING"
    embedding_status: str = "PENDING"
    final_status: str = "PENDING"
    frame_path: str | None = None
    vector_path: str | None = None
    representative_frame_id: str | None = None
    similarity_score: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"frame_id": self.frame_id, "frame_index": self.frame_index,
                "timestamp_ms": self.timestamp_ms, "shot_id": self.shot_id,
                "preliminary_status": self.preliminary_status,
                "frame_path": self.frame_path, "vector_path": self.vector_path,
                "final_status": self.final_status,
                "representative_frame_id": self.representative_frame_id,
                "similarity_score": self.similarity_score}

    @classmethod
    def from_dict(cls, value: dict[str, Any], video_id: str) -> "FrameRecord":
        allowed = {"frame_id", "frame_index", "timestamp_ms", "shot_id",
                   "preliminary_status", "frame_path", "vector_path",
                   "final_status", "representative_frame_id", "similarity_score"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Frame.json has unknown fields: {sorted(unknown)}")
        return cls(video_id=video_id, **dict(value))


def validate_frame(frame: FrameRecord) -> None:
    if frame.frame_index < 0 or not frame.frame_id:
        raise ValueError("invalid frame identity")
    for text, suffix in ((frame.frame_path, ".png"), (frame.vector_path, ".npy")):
        if text is None:
            continue
        path = Path(text)
        if path.suffix != suffix or path.stem != frame.frame_id or path.parent.name != frame.video_id:
            raise ValueError(f"invalid artifact path: {path}")
    if frame.preliminary_status not in {"KEPT", "DUPLICATE"}:
        raise ValueError("invalid preliminary_status")
    if frame.final_status not in {"PENDING", "KEPT", "DUPLICATE"}:
        raise ValueError("invalid final_status")
