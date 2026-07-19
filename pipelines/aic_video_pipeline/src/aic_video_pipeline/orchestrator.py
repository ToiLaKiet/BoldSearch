from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .models import FrameRecord, validate_frame
from .services import (
    FGClipEmbedder,
    HistogramEmbedder,
    detect_shots,
    embed_frames,
    extract_frames,
    index_frames,
    preliminary_dedup_from_config,
    probe_video,
    validate_shots,
)
from .storage import NpyFileVectorStore, atomic_json


class VideoPipelineOrchestrator:
    """Only the ten stages and two metadata files defined by architecture (2)."""

    VIDEO_ID_PATTERN = re.compile(r"^L\d{2}_V\d{2}$")

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @classmethod
    def from_yaml(cls, path: Path) -> "VideoPipelineOrchestrator":
        path = path.expanduser().resolve()
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        data_root = Path(config["paths"]["data_root"])
        if not data_root.is_absolute():
            config["paths"]["data_root"] = str((path.parent / data_root).resolve())
        return cls(config)

    def _data_root(self) -> Path:
        return Path(self.config["paths"]["data_root"])

    def _embedder(self, provider: str):
        cfg = self.config["embedding"]
        if provider == "fgclip":
            return FGClipEmbedder(cfg["model_id"], cfg["model_version"]), cfg["model_version"]
        if provider == "histogram":
            return HistogramEmbedder(), HistogramEmbedder.model_version
        raise ValueError(f"unsupported embedding provider: {provider}")

    def _paths(self, video_id: str) -> tuple[Path, Path, Path, Path]:
        data = self._data_root()
        return data / "videos" / f"{video_id}.mp4", data / "metadata" / video_id / "Shot.json", \
            data / "metadata" / video_id / "Frame.json", data / "frames"

    def run(self, video_path: Path, embedding_provider: str | None = None) -> dict[str, Any]:
        # Keep the logical input name: resolving a symlink would replace
        # Lxx_Vxx.mp4 with the target's unrelated filename.
        video_path = video_path.absolute()
        if video_path.suffix.lower() != ".mp4":
            raise ValueError("only .mp4 input is supported")
        video_id = video_path.stem
        if not self.VIDEO_ID_PATTERN.fullmatch(video_id):
            raise ValueError("video filename must follow Lxx_Vxx.mp4, for example L21_V01.mp4")
        expected_video, shot_path, frame_path, frames_root = self._paths(video_id)
        if video_path != expected_video.absolute():
            raise ValueError(f"video must be stored at {expected_video}")

        metadata = probe_video(video_path)
        shots = detect_shots(video_id, video_path, metadata, self.config["autoshot"])
        atomic_json(shot_path, {"video_id": video_id, "shots": shots["shots"]})

        frames = index_frames(video_id, shots, int(self.config["indexer"]["sample_every_frames"]),
                              int(self.config["pipeline"]["batch_size"]), int(self.config["frame_id"]["zero_pad_width"]))
        frames = preliminary_dedup_from_config(self.config.get("preliminary_dedup", {})).select(frames).frames
        self._save_frame_json(frame_path, video_id, video_path, frames)

        extract_frames(video_path, frames, frames_root, force_overwrite=bool(self.config.get("mapping", {}).get("overwrite_existing", False)))
        self._save_frame_json(frame_path, video_id, video_path, frames)

        provider = embedding_provider or self.config["embedding"]["provider"]
        embedder, version = self._embedder(provider)
        store = NpyFileVectorStore(self._data_root() / "vectors", version)
        embed_frames(frames, embedder, store, int(self.config["embedding"]["batch_size"]),
                     force_overwrite=bool(self.config.get("embedding", {}).get("overwrite_existing", False)))
        self._save_frame_json(frame_path, video_id, video_path, frames)

        self._apply_similarity(frames, store, float(self.config["similarity"]["threshold"]))
        self._save_frame_json(frame_path, video_id, video_path, frames)
        self.validate(video_id)
        return {"video_id": video_id, "frame_count": len(frames),
                "kept_count": sum(frame.final_status == "KEPT" for frame in frames)}

    @staticmethod
    def _save_frame_json(path: Path, video_id: str, video_path: Path, frames: list[FrameRecord]) -> None:
        atomic_json(path, {"video_id": video_id, "source_video_path": str(video_path), "batch_size": 10,
                           "frames": [frame.to_dict() for frame in frames]})

    @staticmethod
    def _apply_similarity(frames: list[FrameRecord], store: NpyFileVectorStore, threshold: float) -> None:
        ordered = sorted((frame for frame in frames if frame.preliminary_status == "KEPT" and frame.vector_path),
                         key=lambda frame: frame.frame_index)
        expected_dimension: int | None = None
        for index in range(0, len(ordered), 2):
            left = ordered[index]
            if index + 1 == len(ordered):
                left.final_status, left.representative_frame_id, left.similarity_score = "KEPT", None, None
                continue
            right = ordered[index + 1]
            left_vector = store.get(left.vector_path, expected_dimension)
            expected_dimension = len(left_vector)
            right_vector = store.get(right.vector_path, expected_dimension)
            score = float(np.dot(left_vector, right_vector))
            if score > threshold:
                left.final_status, left.representative_frame_id, left.similarity_score = "DUPLICATE", right.frame_id, score
                right.final_status, right.representative_frame_id, right.similarity_score = "KEPT", None, score
            else:
                left.final_status, left.representative_frame_id, left.similarity_score = "KEPT", None, score
                right.final_status, right.representative_frame_id, right.similarity_score = "KEPT", None, score

    def validate(self, video_id: str) -> None:
        if not self.VIDEO_ID_PATTERN.fullmatch(video_id):
            raise ValueError("invalid video_id")
        _, shot_path, frame_path, _ = self._paths(video_id)
        if not shot_path.is_file() or not frame_path.is_file():
            raise ValueError("Shot.json and Frame.json must exist")
        shot_doc = json.loads(shot_path.read_text(encoding="utf-8"))
        frame_doc = json.loads(frame_path.read_text(encoding="utf-8"))
        if set(shot_doc) != {"video_id", "shots"} or shot_doc["video_id"] != video_id:
            raise ValueError("Shot.json does not follow the required schema")
        # Reconstruct lightweight probe fields only for the boundary validator.
        probe = probe_video(Path(frame_doc["source_video_path"]))
        validate_shots({**probe, "shots": shot_doc["shots"]})
        if frame_doc.get("video_id") != video_id or not isinstance(frame_doc.get("frames"), list):
            raise ValueError("Frame.json does not follow the required schema")
        frames = [FrameRecord.from_dict(item, video_id) for item in frame_doc["frames"]]
        by_id = {frame.frame_id: frame for frame in frames}
        if len(by_id) != len(frames):
            raise ValueError("duplicate frame_id")
        for frame in frames:
            validate_frame(frame)
            if frame.frame_path:
                if not Path(frame.frame_path).is_file():
                    raise ValueError("PNG file is missing")
            if frame.vector_path:
                NpyFileVectorStore(self._data_root() / "vectors", "unused").get(frame.vector_path)
            if frame.final_status == "DUPLICATE" and frame.representative_frame_id not in by_id:
                raise ValueError("duplicate representative is missing")
            if frame.final_status == "KEPT" and frame.representative_frame_id is not None:
                raise ValueError("kept frame cannot have a representative")
