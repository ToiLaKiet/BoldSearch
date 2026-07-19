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
    mapping_status: str = "PENDING"  # Internal execution state; not written to Frame.json.
    embedding_status: str = "PENDING"  # Internal execution state; not written to Frame.json.
    final_status: str = "PENDING"
    is_active: bool = False
    frame_path: str | None = None
    vector_path: str | None = None
    embedding_model_version: str | None = None
    image_checksum: str | None = None
    representative_frame_id: str | None = None
    similarity_score: float | None = None
    error: str | None = None

    @property
    def frame_key(self) -> str:
        return f"{self.video_id}::{self.frame_id}"

    def to_dict(self) -> dict[str, Any]:
        """The persisted Frame.json schema from AIC_PROCESSING_ARCHITECTURE(2)."""
        return {
            "frame_id": self.frame_id,
            "frame_index": self.frame_index,
            "timestamp_ms": self.timestamp_ms,
            "shot_id": self.shot_id,
            "preliminary_status": self.preliminary_status,
            "frame_path": self.frame_path,
            "vector_path": self.vector_path,
            "final_status": self.final_status,
            "representative_frame_id": self.representative_frame_id,
            "similarity_score": self.similarity_score,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any], video_id: str) -> "FrameRecord":
        value = dict(value)
        allowed = {"frame_id", "frame_index", "timestamp_ms", "shot_id", "preliminary_status", "frame_path",
                   "vector_path", "final_status", "representative_frame_id", "similarity_score"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Frame.json has fields outside the plan: {sorted(unknown)}")
        return cls(video_id=video_id, **value)


def validate_frame(record: FrameRecord) -> None:
    if not record.frame_id or record.frame_index < 0:
        raise ValueError("frame_id and frame_index must be valid")
    for path_text, suffix in ((record.frame_path, ".png"), (record.vector_path, ".npy")):
        if path_text is not None:
            path = Path(path_text)
            if path.suffix != suffix or path.stem != record.frame_id or path.parent.name != record.video_id:
                raise ValueError(f"{suffix} path must mirror video_id/frame_id: {path}")
    allowed = {
        "preliminary_status": {"KEPT", "DUPLICATE"},
        "final_status": {"PENDING", "KEPT", "DUPLICATE"},
    }
    for field, values in allowed.items():
        if getattr(record, field) not in values:
            raise ValueError(f"invalid {field}")
