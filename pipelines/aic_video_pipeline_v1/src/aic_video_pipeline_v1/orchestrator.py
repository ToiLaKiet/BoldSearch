from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import yaml

from .models import FrameRecord, validate_frame
from .services import (FGClipEmbedder, HistogramEmbedder, detect_shots,
                       embed_frames, extract_frames, index_frames, probe_video)
from .similarity import apply_representative_similarity, remove_duplicate_artifacts
from .storage import NpyVectorStore, atomic_json


class VideoPipelineV1:
    VIDEO_ID_PATTERN = re.compile(r"^L\d{2}_V\d{2}$")

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @classmethod
    def from_yaml(cls, path: Path) -> "VideoPipelineV1":
        path = path.expanduser().resolve()
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        for section, key in (("paths", "data_root"), ("embedding", "model_path")):
            value = config.get(section, {}).get(key)
            if value and not Path(value).expanduser().is_absolute():
                config[section][key] = str((path.parent / value).resolve())
        return cls(config)

    def _data_root(self) -> Path:
        return Path(self.config["paths"]["data_root"])

    def _metadata_paths(self, video_id: str) -> tuple[Path, Path]:
        root = self._data_root() / "metadata" / video_id
        return root / "Shot.json", root / "Frame.json"

    def _embedder(self, provider: str):
        cfg = self.config["embedding"]
        if provider == "histogram":
            return HistogramEmbedder(), "histogram_v1"
        if provider == "fgclip":
            return FGClipEmbedder(cfg["model_id"], cfg["model_version"], cfg.get("model_path")), cfg["model_version"]
        raise ValueError(f"unsupported embedding provider: {provider}")

    def _clean_video_outputs(self, video_id: str) -> None:
        # Only this V1 component's target is removed; the old pipeline is untouched.
        for target in (self._data_root() / "metadata" / video_id,
                       self._data_root() / "frames" / video_id,
                       self._data_root() / "vectors" / video_id):
            if target.exists():
                shutil.rmtree(target)

    @staticmethod
    def _save_frame_json(path: Path, video_id: str, source: Path, batch_size: int,
                         frames: list[FrameRecord]) -> None:
        atomic_json(path, {"video_id": video_id, "source_video_path": str(source),
                           "batch_size": batch_size,
                           "frames": [frame.to_dict() for frame in frames]})

    def run(self, video: Path, embedding_provider: str | None = None,
            video_id: str | None = None) -> dict[str, Any]:
        video = video.expanduser().resolve()
        if video.suffix.lower() != ".mp4" or not video.is_file():
            raise ValueError(f"input must be an existing .mp4 file: {video}")
        video_id = video_id or video.stem
        if not self.VIDEO_ID_PATTERN.fullmatch(video_id):
            raise ValueError("video_id must follow Lxx_Vxx; pass --video-id for arbitrary filenames")
        self._clean_video_outputs(video_id)
        shot_path, frame_path = self._metadata_paths(video_id)
        metadata = probe_video(video)
        shots = detect_shots(video_id, video, metadata, self.config["autoshot"])
        atomic_json(shot_path, {"video_id": video_id, "shots": shots["shots"]})

        frames = index_frames(video_id, shots, int(self.config["indexer"]["sample_every_frames"]),
                              int(self.config["pipeline"]["batch_size"]),
                              int(self.config["frame_id"]["zero_pad_width"]))
        extract_frames(video, frames, self._data_root() / "frames")
        self._save_frame_json(frame_path, video_id, video, int(self.config["pipeline"]["batch_size"]), frames)

        provider = embedding_provider or self.config["embedding"]["provider"]
        embedder, version = self._embedder(provider)
        store = NpyVectorStore(self._data_root() / "vectors", version)
        embed_frames(frames, embedder, store, int(self.config["embedding"]["batch_size"]))
        summary = apply_representative_similarity(frames, store, float(self.config["similarity"]["threshold"]))
        remove_duplicate_artifacts(frames)
        self._save_frame_json(frame_path, video_id, video, int(self.config["pipeline"]["batch_size"]), frames)
        self.validate(video_id)
        return {"video_id": video_id, "embedding_provider": provider,
                "frame_count": len(frames), "compared_count": summary.compared,
                "kept_count": summary.kept, "duplicate_count": summary.duplicate,
                "similarity_threshold": summary.threshold,
                "frame_json": str(frame_path)}

    def validate(self, video_id: str) -> None:
        if not self.VIDEO_ID_PATTERN.fullmatch(video_id):
            raise ValueError("invalid video_id")
        shot_path, frame_path = self._metadata_paths(video_id)
        if not shot_path.is_file() or not frame_path.is_file():
            raise ValueError("Shot.json and Frame.json must exist")
        shots = json.loads(shot_path.read_text(encoding="utf-8"))
        document = json.loads(frame_path.read_text(encoding="utf-8"))
        if shots.get("video_id") != video_id or set(shots) != {"video_id", "shots"}:
            raise ValueError("invalid Shot.json")
        if document.get("video_id") != video_id or not isinstance(document.get("frames"), list):
            raise ValueError("invalid Frame.json")
        source = Path(document["source_video_path"])
        metadata = probe_video(source)
        store = NpyVectorStore(self._data_root() / "vectors", "validation")
        for item in shots["shots"]:
            if not 0 <= int(item["frame_start"]) <= int(item["frame_end"]) < metadata["total_frames"]:
                raise ValueError("invalid shot bounds")
        frames = [FrameRecord.from_dict(item, video_id) for item in document["frames"]]
        by_id = {frame.frame_id: frame for frame in frames}
        if len(by_id) != len(frames):
            raise ValueError("duplicate frame_id")
        for frame in frames:
            validate_frame(frame)
            if frame.final_status == "KEPT":
                if not frame.frame_path or not frame.vector_path:
                    raise ValueError("KEPT frame must retain PNG and NPY")
                if not Path(frame.frame_path).is_file() or not Path(frame.vector_path).is_file():
                    raise ValueError("KEPT artifact is missing")
                store.get(frame.vector_path)
            if frame.final_status == "DUPLICATE":
                if frame.frame_path or frame.vector_path:
                    raise ValueError("DUPLICATE frame retains an artifact")
                if frame.representative_frame_id not in by_id:
                    raise ValueError("DUPLICATE representative is missing")
                representative = by_id[frame.representative_frame_id]
                if representative.final_status != "KEPT" or representative.shot_id != frame.shot_id:
                    raise ValueError("DUPLICATE representative must be KEPT in the same shot")
