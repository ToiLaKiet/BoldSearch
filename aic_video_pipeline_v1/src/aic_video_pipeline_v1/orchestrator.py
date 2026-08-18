from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .models import FrameRecord, validate_frame
from .services import (FGClip2Embedder, detect_shots, embed_frames,
                       extract_frames, index_frames, iter_sampled_images,
                       map_frames_to_shots, probe_video, write_png)
from .similarity import (OnlineRepresentativeSimilarity,
                         apply_representative_similarity,
                         remove_duplicate_artifacts)
from .storage import NpyVectorStore, atomic_json


class VideoPipelineV1:
    """Two-branch video pipeline with resumable materialized stages."""

    VIDEO_ID_PATTERN = re.compile(r"^L\d{2}_V\d{2,3}$")
    CHECKPOINT_SCHEMA = "2.0"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._checkpoint_lock = threading.Lock()
        self._embedder_instance: FGClip2Embedder | None = None

    @classmethod
    def from_yaml(cls, path: Path, *, data_root: Path | None = None,
                  model_path: Path | None = None,
                  autoshot_root: Path | None = None,
                  autoshot_checkpoint: Path | None = None) -> "VideoPipelineV1":
        path = path.expanduser().resolve()
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        for section, key in (("paths", "data_root"), ("embedding", "model_path")):
            value = config.get(section, {}).get(key)
            if value and not Path(value).expanduser().is_absolute():
                config[section][key] = str((path.parent / value).resolve())
        if data_root is not None:
            config["paths"]["data_root"] = str(data_root.expanduser().resolve())
        if model_path is not None:
            config["embedding"]["model_path"] = str(model_path.expanduser().resolve())
        if autoshot_root is not None:
            config["autoshot"]["root"] = str(autoshot_root.expanduser().resolve())
        if autoshot_checkpoint is not None:
            config["autoshot"]["checkpoint"] = str(
                autoshot_checkpoint.expanduser().resolve()
            )
        return cls(config)

    def _data_root(self) -> Path:
        return Path(self.config["paths"]["data_root"])

    def _metadata_paths(self, video_id: str) -> tuple[Path, Path]:
        root = self._data_root() / "metadata" / video_id
        return root / "Shot.json", root / "Frame.json"

    def _checkpoint_path(self, video_id: str) -> Path:
        return self._data_root() / "checkpoints" / f"{video_id}.json"

    def _embedder(self) -> tuple[FGClip2Embedder, str]:
        cfg = self.config["embedding"]
        if cfg.get("provider") != "fgclip2":
            raise ValueError("embedding.provider must be fgclip2")
        if self._embedder_instance is None:
            self._embedder_instance = FGClip2Embedder(
                cfg["model_id"], cfg["model_version"],
                cfg.get("model_path"), cfg.get("revision")
            )
        return self._embedder_instance, cfg["model_version"]

    def _clean_video_outputs(self, video_id: str) -> None:
        for target in (self._data_root() / "metadata" / video_id,
                       self._data_root() / "frames" / video_id,
                       self._data_root() / "vectors" / video_id):
            if target.exists():
                shutil.rmtree(target)
        self._checkpoint_path(video_id).unlink(missing_ok=True)

    def _config_signature(self) -> str:
        relevant = {
            key: self.config[key]
            for key in ("indexer", "autoshot", "embedding", "similarity")
        }
        raw = json.dumps(relevant, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _source_identity(video: Path) -> dict[str, Any]:
        stat = video.stat()
        return {"path": str(video), "size": stat.st_size}

    def _new_checkpoint(self, video_id: str, video: Path) -> dict[str, Any]:
        return {
            "schema_version": self.CHECKPOINT_SCHEMA,
            "video_id": video_id,
            "source": self._source_identity(video),
            "config_signature": self._config_signature(),
            "stages": {
                "shots_ready": False,
                "frames_extracted": False,
                "frames_mapped": False,
                "embeddings_ready": False,
                "similarity_classified": False,
                "deduplicated": False,
                "validated": False,
            },
            "metrics": {},
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _load_checkpoint(self, video_id: str) -> dict[str, Any] | None:
        path = self._checkpoint_path(video_id)
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def _checkpoint_matches(self, value: dict[str, Any], video_id: str,
                            video: Path) -> bool:
        return (
            value.get("schema_version") == self.CHECKPOINT_SCHEMA
            and value.get("video_id") == video_id
            and value.get("source") == self._source_identity(video)
            and value.get("config_signature") == self._config_signature()
            and isinstance(value.get("stages"), dict)
        )

    def _write_checkpoint(self, video_id: str, checkpoint: dict[str, Any]) -> None:
        with self._checkpoint_lock:
            checkpoint["updated_at"] = datetime.now(timezone.utc).isoformat()
            atomic_json(self._checkpoint_path(video_id), checkpoint)

    def _mark_stage(self, video_id: str, checkpoint: dict[str, Any], stage: str,
                    metrics: dict[str, Any] | None = None) -> None:
        with self._checkpoint_lock:
            checkpoint["stages"][stage] = True
            if metrics:
                checkpoint["metrics"].update(metrics)
            checkpoint["updated_at"] = datetime.now(timezone.utc).isoformat()
            atomic_json(self._checkpoint_path(video_id), checkpoint)

    def _save_frame_json(self, path: Path, video_id: str, source: Path,
                         frames: list[FrameRecord], stage: str) -> None:
        cfg = self.config["embedding"]
        atomic_json(path, {
            "schema_version": "2.0",
            "video_id": video_id,
            "source_video_path": str(source),
            "stage": stage,
            "sample_every_frames": int(self.config["indexer"]["sample_every_frames"]),
            "embedding": {
                "provider": cfg["provider"],
                "model_id": cfg["model_id"],
                "revision": cfg.get("revision"),
                "model_version": cfg["model_version"],
            },
            "frames": [frame.to_dict() for frame in frames],
        })

    def _save_streaming_frame_json(
        self, path: Path, video_id: str, source: Path,
        frames: list[FrameRecord], stage: str, progress: dict[str, Any]
    ) -> None:
        cfg = self.config["embedding"]
        atomic_json(path, {
            "schema_version": "2.0",
            "video_id": video_id,
            "source_video_path": str(source),
            "stage": stage,
            "sample_every_frames": int(self.config["indexer"]["sample_every_frames"]),
            "execution_mode": "streaming_kept_only_v1",
            "embedding": {
                "provider": cfg["provider"],
                "model_id": cfg["model_id"],
                "revision": cfg.get("revision"),
                "model_version": cfg["model_version"],
                "batch_size": int(cfg.get("batch_size", 2)),
            },
            "progress": progress,
            "frames": [frame.to_dict() for frame in frames],
        })

    @staticmethod
    def _load_frames(path: Path, video_id: str) -> list[FrameRecord]:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("video_id") != video_id or not isinstance(document.get("frames"), list):
            raise ValueError("invalid resumable Frame.json")
        return [FrameRecord.from_dict(item, video_id) for item in document["frames"]]

    @staticmethod
    def _load_shots(path: Path, video_id: str) -> dict[str, Any]:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("video_id") != video_id or not isinstance(document.get("shots"), list):
            raise ValueError("invalid resumable Shot.json")
        return document

    def _result(self, video_id: str, frame_path: Path,
                checkpoint: dict[str, Any]) -> dict[str, Any]:
        metrics = checkpoint.get("metrics", {})
        return {
            "video_id": video_id,
            "embedding_provider": "fgclip2",
            "sample_every_frames": int(self.config["indexer"]["sample_every_frames"]),
            "sampled_count": int(metrics.get("sampled_count", 0)),
            "extracted_count": int(metrics.get("extracted_count", 0)),
            "compared_count": int(metrics.get("compared_count", 0)),
            "kept_count": int(metrics.get("kept_count", 0)),
            "duplicate_count": int(metrics.get("duplicate_count", 0)),
            "similarity_threshold": float(self.config["similarity"]["threshold"]),
            "frame_json": str(frame_path),
            "checkpoint": str(self._checkpoint_path(video_id)),
        }

    def run(self, video: Path, video_id: str | None = None,
            fresh: bool = False) -> dict[str, Any]:
        video = video.expanduser().resolve()
        if video.suffix.lower() != ".mp4" or not video.is_file():
            raise ValueError(f"input must be an existing .mp4 file: {video}")
        video_id = video_id or video.stem
        if not self.VIDEO_ID_PATTERN.fullmatch(video_id):
            raise ValueError(
                "video_id must follow Lxx_Vxx or Lxx_Vxxx; pass --video-id for arbitrary filenames"
            )

        if fresh:
            self._clean_video_outputs(video_id)
        checkpoint = self._load_checkpoint(video_id)
        if checkpoint is not None and not self._checkpoint_matches(checkpoint, video_id, video):
            self._clean_video_outputs(video_id)
            checkpoint = None
        if checkpoint is None:
            self._clean_video_outputs(video_id)
            checkpoint = self._new_checkpoint(video_id, video)
            self._write_checkpoint(video_id, checkpoint)

        shot_path, frame_path = self._metadata_paths(video_id)
        if checkpoint["stages"].get("validated"):
            self.validate(video_id)
            return self._result(video_id, frame_path, checkpoint)

        metadata = probe_video(video)
        interval = int(self.config["indexer"]["sample_every_frames"])

        shots: dict[str, Any] | None = None
        frames: list[FrameRecord] | None = None

        if checkpoint["stages"].get("shots_ready") and shot_path.is_file():
            shots = self._load_shots(shot_path, video_id)
        if checkpoint["stages"].get("frames_extracted") and frame_path.is_file():
            frames = self._load_frames(frame_path, video_id)

        def shot_branch() -> dict[str, Any]:
            detected = detect_shots(video_id, video, metadata, self.config["autoshot"])
            document = {
                "schema_version": "2.0",
                "video_id": video_id,
                "source_video_path": str(video),
                **metadata,
                "detector": detected.get("detector", "autoshot_v1"),
                "shots": detected["shots"],
            }
            atomic_json(shot_path, document)
            self._mark_stage(video_id, checkpoint, "shots_ready",
                             {"shot_count": len(document["shots"])})
            return document

        def frame_branch() -> list[FrameRecord]:
            selected = index_frames(video_id, metadata, interval)
            extract_frames(video, selected, self._data_root() / "frames")
            self._save_frame_json(frame_path, video_id, video, selected, "EXTRACTED")
            self._mark_stage(video_id, checkpoint, "frames_extracted", {
                "sampled_count": (metadata["total_frames"] + interval - 1) // interval,
                "extracted_count": len(selected),
            })
            return selected

        pending: dict[str, Any] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            if shots is None:
                pending["shots"] = pool.submit(shot_branch)
            if frames is None:
                pending["frames"] = pool.submit(frame_branch)
            if "shots" in pending:
                shots = pending["shots"].result()
            if "frames" in pending:
                frames = pending["frames"].result()

        if shots is None or frames is None:
            raise RuntimeError("parallel preparation did not produce shots and frames")

        if not checkpoint["stages"].get("frames_mapped"):
            map_frames_to_shots(frames, shots)
            self._save_frame_json(frame_path, video_id, video, frames, "MAPPED")
            self._mark_stage(video_id, checkpoint, "frames_mapped")

        store = NpyVectorStore(self._data_root() / "vectors",
                               self.config["embedding"]["model_version"])

        if not checkpoint["stages"].get("similarity_classified"):
            if not checkpoint["stages"].get("embeddings_ready"):
                embedder, _version = self._embedder()
                checkpoint_every = max(
                    1, int(self.config["embedding"].get("checkpoint_every_frames", 10))
                )

                def embedding_checkpoint(current: int, total: int) -> None:
                    if current % checkpoint_every == 0 or current == total:
                        self._save_frame_json(frame_path, video_id, video, frames,
                                              "EMBEDDING")
                        checkpoint["metrics"]["embedded_count"] = current
                        self._write_checkpoint(video_id, checkpoint)

                embed_frames(frames, embedder, store, embedding_checkpoint)
                self._save_frame_json(frame_path, video_id, video, frames, "EMBEDDED")
                self._mark_stage(video_id, checkpoint, "embeddings_ready",
                                 {"embedded_count": len(frames)})

            summary = apply_representative_similarity(
                frames, store, float(self.config["similarity"]["threshold"])
            )
            self._save_frame_json(frame_path, video_id, video, frames,
                                  "SIMILARITY_CLASSIFIED")
            self._mark_stage(video_id, checkpoint, "similarity_classified", {
                "compared_count": summary.compared,
                "kept_count": summary.kept,
                "duplicate_count": summary.duplicate,
            })

        remove_duplicate_artifacts(frames)
        self._save_frame_json(frame_path, video_id, video, frames, "FINAL")
        self._mark_stage(video_id, checkpoint, "deduplicated")
        self.validate(video_id)
        self._mark_stage(video_id, checkpoint, "validated")
        return self._result(video_id, frame_path, checkpoint)

    def run_streaming(self, video: Path, video_id: str | None = None,
                      fresh: bool = False) -> dict[str, Any]:
        """Kaggle path: batch in RAM and never materialize DUPLICATE frames."""
        video = video.expanduser().resolve()
        if video.suffix.lower() != ".mp4" or not video.is_file():
            raise ValueError(f"input must be an existing .mp4 file: {video}")
        video_id = video_id or video.stem
        if not self.VIDEO_ID_PATTERN.fullmatch(video_id):
            raise ValueError(
                "video_id must follow Lxx_Vxx or Lxx_Vxxx; pass --video-id for arbitrary filenames"
            )

        mode = "streaming_kept_only_v1"
        if fresh:
            self._clean_video_outputs(video_id)
        checkpoint = self._load_checkpoint(video_id)
        if (checkpoint is not None
                and (not self._checkpoint_matches(checkpoint, video_id, video)
                     or checkpoint.get("execution_mode") != mode)):
            self._clean_video_outputs(video_id)
            checkpoint = None
        if checkpoint is None:
            self._clean_video_outputs(video_id)
            checkpoint = self._new_checkpoint(video_id, video)
            checkpoint["execution_mode"] = mode
            self._write_checkpoint(video_id, checkpoint)

        shot_path, frame_path = self._metadata_paths(video_id)
        if checkpoint["stages"].get("validated"):
            self.validate(video_id, verify_source=False)
            return self._result(video_id, frame_path, checkpoint)

        if checkpoint["stages"].get("shots_ready") and shot_path.is_file():
            shots = self._load_shots(shot_path, video_id)
            metadata = {
                key: shots[key] for key in (
                    "fps", "total_frames", "duration_ms", "width", "height",
                    "codec", "source_video_checksum"
                )
            }
        else:
            metadata = probe_video(video)
            detected = detect_shots(video_id, video, metadata, self.config["autoshot"])
            shots = {
                "schema_version": "2.0",
                "video_id": video_id,
                "source_video_path": str(video),
                **metadata,
                "detector": detected.get("detector", "autoshot_v1"),
                "shots": detected["shots"],
            }
            atomic_json(shot_path, shots)
            self._mark_stage(video_id, checkpoint, "shots_ready",
                             {"shot_count": len(shots["shots"])})

        kept_frames: list[FrameRecord] = []
        progress: dict[str, Any] = {
            "last_processed_frame_index": -1,
            "sampled_count": 0,
            "extracted_count": 0,
            "compared_count": 0,
            "kept_count": 0,
            "duplicate_count": 0,
        }
        if frame_path.is_file():
            document = json.loads(frame_path.read_text(encoding="utf-8"))
            if (document.get("execution_mode") == mode
                    and isinstance(document.get("frames"), list)
                    and isinstance(document.get("progress"), dict)):
                kept_frames = [FrameRecord.from_dict(item, video_id)
                               for item in document["frames"]]
                progress.update(document["progress"])

        store = NpyVectorStore(self._data_root() / "vectors",
                               self.config["embedding"]["model_version"])
        similarity = OnlineRepresentativeSimilarity(
            float(self.config["similarity"]["threshold"])
        )
        for frame in sorted(kept_frames, key=lambda item: item.frame_index):
            if not frame.shot_id or not frame.vector_path:
                raise ValueError("invalid streaming resume frame")
            similarity.restore(frame.shot_id, frame.frame_id,
                               store.get(frame.vector_path))

        expected_png = {Path(frame.frame_path).resolve() for frame in kept_frames
                        if frame.frame_path}
        expected_npy = {Path(frame.vector_path).resolve() for frame in kept_frames
                        if frame.vector_path}
        for path in (self._data_root() / "frames" / video_id).glob("*.png"):
            if path.resolve() not in expected_png:
                path.unlink(missing_ok=True)
        for path in (self._data_root() / "vectors" / video_id).glob("*.npy"):
            if path.resolve() not in expected_npy:
                path.unlink(missing_ok=True)

        ordered_shots = sorted(shots["shots"],
                               key=lambda item: int(item["frame_start"]))
        if not ordered_shots:
            raise ValueError("AutoShot returned no shots")
        shot_position = 0
        interval = int(self.config["indexer"]["sample_every_frames"])
        batch_size = max(1, int(self.config["embedding"].get("batch_size", 2)))
        checkpoint_every = max(
            batch_size,
            int(self.config["embedding"].get("checkpoint_every_frames", 100)),
        )
        last_saved_count = int(progress["sampled_count"])
        last_processed = int(progress["last_processed_frame_index"])
        embedder, _version = self._embedder()
        pending: list[tuple[FrameRecord, Any]] = []

        def shot_for(frame_index: int) -> str:
            nonlocal shot_position
            while (shot_position + 1 < len(ordered_shots)
                   and frame_index > int(ordered_shots[shot_position]["frame_end"])):
                shot_position += 1
            shot = ordered_shots[shot_position]
            if not int(shot["frame_start"]) <= frame_index <= int(shot["frame_end"]):
                raise ValueError(f"frame {frame_index} is not covered by any shot")
            return str(shot["shot_id"])

        def save_progress(stage: str = "STREAMING") -> None:
            checkpoint["metrics"].update({
                key: int(progress[key]) for key in (
                    "sampled_count", "extracted_count", "compared_count",
                    "kept_count", "duplicate_count"
                )
            })
            self._save_streaming_frame_json(
                frame_path, video_id, video, kept_frames, stage, dict(progress)
            )
            self._write_checkpoint(video_id, checkpoint)

        def flush_batch() -> None:
            nonlocal last_saved_count
            if not pending:
                return
            vectors = embedder.embed_images([image for _frame, image in pending])
            if len(vectors) != len(pending):
                raise ValueError("FG-CLIP2 returned an invalid batch size")
            for (frame, image), vector in zip(pending, vectors):
                status, representative_id, score = similarity.classify(
                    str(frame.shot_id), frame.frame_id, vector
                )
                progress["sampled_count"] += 1
                progress["extracted_count"] += 1
                progress["compared_count"] += 1
                progress["last_processed_frame_index"] = frame.frame_index
                if status == "DUPLICATE":
                    progress["duplicate_count"] += 1
                    continue
                frame.mapping_status = "MAPPED"
                frame.embedding_status = "EMBEDDED"
                frame.final_status = "KEPT"
                frame.representative_frame_id = None
                frame.similarity_score = score
                png = self._data_root() / "frames" / video_id / f"{frame.frame_id}.png"
                frame.frame_path = str(write_png(image, png))
                frame.vector_path = str(store.put(video_id, frame.frame_id, vector))
                kept_frames.append(frame)
                progress["kept_count"] += 1
            pending.clear()
            if int(progress["sampled_count"]) - last_saved_count >= checkpoint_every:
                save_progress()
                last_saved_count = int(progress["sampled_count"])
                print(
                    f"[STREAM {video_id}] sampled={progress['sampled_count']} "
                    f"kept={progress['kept_count']} "
                    f"duplicate={progress['duplicate_count']}",
                    flush=True,
                )

        for frame_index, image in iter_sampled_images(
                video, int(metadata["total_frames"]), interval):
            if frame_index <= last_processed:
                continue
            frame = FrameRecord(
                video_id=video_id,
                frame_id=str(int(frame_index)),
                frame_index=int(frame_index),
                timestamp_ms=round(frame_index / float(metadata["fps"]) * 1000),
                shot_id=shot_for(frame_index),
                mapping_status="MAPPED",
            )
            pending.append((frame, image))
            if len(pending) >= batch_size:
                flush_batch()
        flush_batch()
        if int(progress["sampled_count"]) == 0:
            raise ValueError("video decoding produced no sampled frames")
        save_progress("FINAL")

        checkpoint["stages"].update({
            "frames_extracted": True,
            "frames_mapped": True,
            "embeddings_ready": True,
            "similarity_classified": True,
            "deduplicated": True,
        })
        self._write_checkpoint(video_id, checkpoint)
        self.validate(video_id, verify_source=False)
        checkpoint["stages"]["validated"] = True
        self._write_checkpoint(video_id, checkpoint)
        return self._result(video_id, frame_path, checkpoint)

    def validate(self, video_id: str, *, verify_source: bool = True) -> None:
        if not self.VIDEO_ID_PATTERN.fullmatch(video_id):
            raise ValueError("invalid video_id")
        shot_path, frame_path = self._metadata_paths(video_id)
        if not shot_path.is_file() or not frame_path.is_file():
            raise ValueError("Shot.json and Frame.json must exist")
        shots = json.loads(shot_path.read_text(encoding="utf-8"))
        document = json.loads(frame_path.read_text(encoding="utf-8"))
        required_shot_fields = {
            "schema_version", "video_id", "source_video_path", "fps",
            "total_frames", "duration_ms", "width", "height", "codec",
            "source_video_checksum", "detector", "shots",
        }
        if (shots.get("schema_version") != "2.0"
                or shots.get("video_id") != video_id
                or not required_shot_fields.issubset(shots)
                or not isinstance(shots.get("shots"), list)):
            raise ValueError("invalid Shot.json")
        if (document.get("schema_version") != "2.0"
                or document.get("video_id") != video_id
                or document.get("stage") != "FINAL"
                or not isinstance(document.get("frames"), list)):
            raise ValueError("invalid final Frame.json")
        source = Path(document["source_video_path"])
        total_frames = int(shots["total_frames"])
        if verify_source and source.is_file():
            source_metadata = probe_video(source)
            if (source_metadata["total_frames"] != total_frames
                    or source_metadata["source_video_checksum"]
                    != shots["source_video_checksum"]):
                raise ValueError("source video no longer matches Shot.json")
        ordered_shots = sorted(shots["shots"], key=lambda item: int(item["frame_start"]))
        if not ordered_shots or int(ordered_shots[0]["frame_start"]) != 0:
            raise ValueError("shots must start at frame zero")
        previous_end = -1
        for item in ordered_shots:
            start, end = int(item["frame_start"]), int(item["frame_end"])
            if start != previous_end + 1 or not start <= end < total_frames:
                raise ValueError("shots must be ordered and gap-free")
            previous_end = end
        if previous_end != total_frames - 1:
            raise ValueError("shots must cover the complete source video")

        frames = [FrameRecord.from_dict(item, video_id) for item in document["frames"]]
        by_id = {frame.frame_id: frame for frame in frames}
        if len(by_id) != len(frames):
            raise ValueError("duplicate frame_id")
        store = NpyVectorStore(self._data_root() / "vectors", "validation")
        expected_png: set[Path] = set()
        expected_npy: set[Path] = set()
        for frame in frames:
            validate_frame(frame, require_final=True)
            if frame.representative_frame_id is not None:
                raise ValueError("KEPT frame must not reference another representative")
            if not frame.frame_path or not frame.vector_path:
                raise ValueError("KEPT frame must retain PNG and NPY")
            png, npy = Path(frame.frame_path), Path(frame.vector_path)
            if png.stem != npy.stem or png.stem != str(frame.frame_index):
                raise ValueError("PNG and NPY names must equal the integer frame index")
            if not png.is_file() or not npy.is_file():
                raise ValueError("KEPT artifact is missing")
            store.get(npy)
            expected_png.add(png.resolve())
            expected_npy.add(npy.resolve())

        actual_png = {path.resolve() for path in
                      (self._data_root() / "frames" / video_id).glob("*.png")}
        actual_npy = {path.resolve() for path in
                      (self._data_root() / "vectors" / video_id).glob("*.npy")}
        if actual_png != expected_png or actual_npy != expected_npy:
            raise ValueError("artifact directories must contain exactly the KEPT mappings")
