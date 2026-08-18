import json
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from aic_video_pipeline_v1.models import FrameRecord
from aic_video_pipeline_v1.orchestrator import VideoPipelineV1
from aic_video_pipeline_v1.services import (embed_frames, index_frames,
                                            map_frames_to_shots)
from aic_video_pipeline_v1.similarity import (OnlineRepresentativeSimilarity,
                                              apply_representative_similarity,
                                              remove_duplicate_artifacts)
from aic_video_pipeline_v1.storage import NpyVectorStore


def make_frame(index: int, shot: str = "shot_000001") -> FrameRecord:
    return FrameRecord("L21_V01", str(index), index, index * 40, shot,
                       mapping_status="EXTRACTED", embedding_status="EMBEDDED")


def test_indexer_samples_without_batch_metadata() -> None:
    metadata = {"fps": 25.0, "total_frames": 52}
    frames = index_frames("L21_V01", metadata, interval=10)
    assert [frame.frame_index for frame in frames] == [0, 10, 20, 30, 40, 50]
    assert [frame.frame_id for frame in frames] == ["0", "10", "20", "30", "40", "50"]
    assert all(not hasattr(frame, "batch_id") for frame in frames)


def test_independent_samples_are_mapped_after_shots_finish() -> None:
    metadata = {"fps": 25.0, "total_frames": 41}
    frames = index_frames("L21_V01", metadata, interval=10)
    shots = {"shots": [
        {"shot_id": "shot_000001", "frame_start": 0, "frame_end": 14},
        {"shot_id": "shot_000002", "frame_start": 15, "frame_end": 40},
    ]}
    assert all(frame.shot_id is None for frame in frames)
    map_frames_to_shots(frames, shots)
    assert [frame.shot_id for frame in frames] == [
        "shot_000001", "shot_000001", "shot_000002",
        "shot_000002", "shot_000002",
    ]
    assert all(frame.mapping_status == "MAPPED" for frame in frames)


def test_embedding_reads_each_png_directly_without_batches(tmp_path: Path) -> None:
    class FakeFGClip:
        def __init__(self) -> None:
            self.paths: list[Path] = []

        def embed_frame(self, path: Path) -> np.ndarray:
            self.paths.append(path)
            return np.array([int(path.stem) + 1, 1], dtype=np.float32)

    frames = [make_frame(index) for index in range(3)]
    for frame in frames:
        image = tmp_path / "frames" / frame.video_id / f"{frame.frame_id}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"png")
        frame.frame_path = str(image)
        frame.embedding_status = "PENDING"
    embedder = FakeFGClip()
    store = NpyVectorStore(tmp_path / "vectors", "test")
    embed_frames(frames, embedder, store)
    assert embedder.paths == [Path(frame.frame_path) for frame in frames]
    assert all(frame.embedding_status == "EMBEDDED" for frame in frames)
    assert all(Path(frame.vector_path).is_file() for frame in frames)


def test_embedding_resume_reuses_valid_npy(tmp_path: Path) -> None:
    class FakeFGClip2:
        def __init__(self) -> None:
            self.paths: list[Path] = []

        def embed_frame(self, path: Path) -> np.ndarray:
            self.paths.append(path)
            return np.array([0, 1], dtype=np.float32)

    frames = [make_frame(0), make_frame(10)]
    for frame in frames:
        image = tmp_path / "frames" / frame.video_id / f"{frame.frame_id}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"png")
        frame.frame_path = str(image)
        frame.embedding_status = "PENDING"
    store = NpyVectorStore(tmp_path / "vectors", "test")
    store.put(frames[0].video_id, frames[0].frame_id,
              np.array([1, 0], dtype=np.float32))
    embedder = FakeFGClip2()
    embed_frames(frames, embedder, store)
    assert embedder.paths == [Path(frames[1].frame_path)]
    assert all(frame.embedding_status == "EMBEDDED" for frame in frames)


def test_long_chain_uses_latest_kept_representative(tmp_path: Path) -> None:
    store = NpyVectorStore(tmp_path / "vectors", "test")
    frames = [make_frame(index) for index in range(4)]
    vectors = [np.array([1, 0], np.float32), np.array([1, 0], np.float32),
               np.array([1, 0], np.float32), np.array([0, 1], np.float32)]
    for frame, vector in zip(frames, vectors):
        frame.vector_path = str(store.put(frame.video_id, frame.frame_id, vector))
    summary = apply_representative_similarity(frames, store, 0.9)
    assert (summary.kept, summary.duplicate) == (2, 2)
    assert [frame.final_status for frame in frames] == ["KEPT", "DUPLICATE", "DUPLICATE", "KEPT"]
    assert frames[1].representative_frame_id == frames[0].frame_id
    assert frames[2].representative_frame_id == frames[0].frame_id


def test_shot_boundary_resets_representative(tmp_path: Path) -> None:
    store = NpyVectorStore(tmp_path / "vectors", "test")
    frames = [make_frame(0), make_frame(1, "shot_000002")]
    for frame in frames:
        frame.vector_path = str(store.put(frame.video_id, frame.frame_id, np.array([1, 0], np.float32)))
    summary = apply_representative_similarity(frames, store, 0.9)
    assert (summary.kept, summary.duplicate) == (2, 0)


def test_online_similarity_does_not_require_duplicate_artifacts() -> None:
    similarity = OnlineRepresentativeSimilarity(0.9)
    first = similarity.classify("shot_000001", "0", np.array([1, 0], np.float32))
    duplicate = similarity.classify("shot_000001", "10", np.array([1, 0], np.float32))
    changed = similarity.classify("shot_000001", "20", np.array([0, 1], np.float32))
    new_shot = similarity.classify("shot_000002", "30", np.array([0, 1], np.float32))
    assert first == ("KEPT", None, None)
    assert duplicate[0:2] == ("DUPLICATE", "0")
    assert changed[0:2] == ("KEPT", None)
    assert new_shot == ("KEPT", None, None)


def test_duplicate_artifacts_are_removed(tmp_path: Path) -> None:
    store = NpyVectorStore(tmp_path / "vectors", "test")
    frames = [make_frame(0), make_frame(1)]
    for frame in frames:
        image = tmp_path / "frames" / frame.video_id / f"{frame.frame_id}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"png")
        frame.frame_path = str(image)
        frame.vector_path = str(store.put(frame.video_id, frame.frame_id, np.array([1, 0], np.float32)))
    apply_representative_similarity(frames, store, 0.9)
    duplicate = frames[1]
    remove_duplicate_artifacts(frames)
    assert duplicate.frame_path is None and duplicate.vector_path is None
    assert [frame.frame_id for frame in frames] == ["0"]
    assert all(frame.final_status == "KEPT" for frame in frames)
    assert not (tmp_path / "frames" / "L21_V01" / "1.png").exists()
    assert frames[0].frame_path is not None and Path(frames[0].frame_path).exists()


def test_pipeline_runs_parallel_branches_and_resumes(tmp_path: Path, monkeypatch) -> None:
    import aic_video_pipeline_v1.orchestrator as module

    video = tmp_path / "L21_V001.mp4"
    video.write_bytes(b"fake-video")
    data_root = tmp_path / "data"
    config = {
        "indexer": {"sample_every_frames": 10},
        "autoshot": {"root": str(tmp_path), "checkpoint": str(tmp_path / "a.pth"),
                     "threshold": 0.296, "allow_fallback": False},
        "embedding": {"provider": "fgclip2", "model_id": "fake/fgclip2",
                      "model_version": "test", "model_path": None,
                      "checkpoint_every_frames": 2},
        "similarity": {"threshold": 0.9},
        "paths": {"data_root": str(data_root)},
    }
    barrier = threading.Barrier(2, timeout=2)
    calls = {"shots": 0, "frames": 0, "embeddings": 0}

    def fake_probe(_path: Path) -> dict:
        return {"fps": 25.0, "total_frames": 31, "duration_ms": 1240,
                "width": 1920, "height": 1080, "codec": "fake",
                "source_video_checksum": "sha256:fake"}

    def fake_detect(video_id: str, path: Path, metadata: dict, cfg: dict) -> dict:
        calls["shots"] += 1
        barrier.wait()
        return {"video_id": video_id, "shots": [
            {"shot_id": "shot_000001", "frame_start": 0, "frame_end": 30,
             "start_ms": 0, "end_ms": 1200},
        ]}

    def fake_extract(_path: Path, frames: list[FrameRecord], root: Path) -> None:
        calls["frames"] += 1
        barrier.wait()
        for frame in frames:
            output = root / frame.video_id / f"{frame.frame_id}.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"png")
            frame.frame_path = str(output)
            frame.mapping_status = "EXTRACTED"

    def fake_embed(frames: list[FrameRecord], _embedder, store: NpyVectorStore,
                   checkpoint_callback=None) -> None:
        calls["embeddings"] += 1
        vectors = ([1, 0], [1, 0], [0, 1], [0, 1])
        for position, (frame, vector) in enumerate(zip(frames, vectors), 1):
            frame.vector_path = str(store.put(frame.video_id, frame.frame_id,
                                              np.asarray(vector, np.float32)))
            frame.embedding_status = "EMBEDDED"
            if checkpoint_callback:
                checkpoint_callback(position, len(frames))

    monkeypatch.setattr(module, "probe_video", fake_probe)
    monkeypatch.setattr(module, "detect_shots", fake_detect)
    monkeypatch.setattr(module, "extract_frames", fake_extract)
    monkeypatch.setattr(module, "embed_frames", fake_embed)
    monkeypatch.setattr(VideoPipelineV1, "_embedder", lambda self: (object(), "test"))

    pipeline = VideoPipelineV1(config)
    result = pipeline.run(video, "L21_V001")
    assert result["sampled_count"] == 4
    assert result["kept_count"] == 2
    assert result["duplicate_count"] == 2
    assert calls == {"shots": 1, "frames": 1, "embeddings": 1}

    frame_doc = json.loads((data_root / "metadata" / "L21_V001" / "Frame.json")
                           .read_text(encoding="utf-8"))
    assert frame_doc["stage"] == "FINAL"
    assert [item["frame_id"] for item in frame_doc["frames"]] == ["0", "20"]
    assert sorted(path.name for path in (data_root / "frames" / "L21_V001").glob("*.png")) == [
        "0.png", "20.png"
    ]
    assert sorted(path.name for path in (data_root / "vectors" / "L21_V001").glob("*.npy")) == [
        "0.npy", "20.npy"
    ]

    resumed = pipeline.run(video, "L21_V001")
    assert resumed["kept_count"] == 2
    assert calls == {"shots": 1, "frames": 1, "embeddings": 1}
    checkpoint = json.loads((data_root / "checkpoints" / "L21_V001.json")
                            .read_text(encoding="utf-8"))
    assert all(checkpoint["stages"].values())
    video.unlink()
    pipeline.validate("L21_V001")


def test_streaming_pipeline_batches_in_ram_and_writes_kept_only(
        tmp_path: Path, monkeypatch) -> None:
    import aic_video_pipeline_v1.orchestrator as module

    video = tmp_path / "L21_V001.mp4"
    video.write_bytes(b"fake-video")
    data_root = tmp_path / "data"
    config = {
        "indexer": {"sample_every_frames": 10},
        "autoshot": {"root": str(tmp_path), "checkpoint": str(tmp_path / "a.pth"),
                     "threshold": 0.296, "allow_fallback": False},
        "embedding": {"provider": "fgclip2", "model_id": "fake/fgclip2",
                      "model_version": "test", "model_path": None,
                      "batch_size": 4, "checkpoint_every_frames": 100},
        "similarity": {"threshold": 0.9},
        "paths": {"data_root": str(data_root)},
    }
    calls: list[int] = []

    def fake_probe(_path: Path) -> dict:
        return {"fps": 25.0, "total_frames": 31, "duration_ms": 1240,
                "width": 32, "height": 18, "codec": "fake",
                "source_video_checksum": "sha256:fake"}

    def fake_detect(video_id: str, path: Path, metadata: dict, cfg: dict) -> dict:
        return {"video_id": video_id, "detector": "test", "shots": [
            {"shot_id": "shot_000001", "frame_start": 0, "frame_end": 30,
             "start_ms": 0, "end_ms": 1200},
        ]}

    def fake_images(_path: Path, _total: int, _interval: int):
        for index in (0, 10, 20, 30):
            yield index, Image.new("RGB", (32, 18), color=(index, 0, 0))

    class FakeBatchEmbedder:
        def embed_images(self, images: list[Image.Image]) -> np.ndarray:
            calls.append(len(images))
            return np.asarray(([1, 0], [1, 0], [0, 1], [0, 1]), np.float32)

    monkeypatch.setattr(module, "probe_video", fake_probe)
    monkeypatch.setattr(module, "detect_shots", fake_detect)
    monkeypatch.setattr(module, "iter_sampled_images", fake_images)
    pipeline = VideoPipelineV1(config)
    monkeypatch.setattr(pipeline, "_embedder", lambda: (FakeBatchEmbedder(), "test"))

    result = pipeline.run_streaming(video, "L21_V001")
    assert calls == [4]
    assert result["sampled_count"] == 4
    assert result["kept_count"] == 2
    assert result["duplicate_count"] == 2
    assert sorted(path.name for path in (data_root / "frames" / "L21_V001").glob("*.png")) == [
        "0.png", "20.png"
    ]
    assert sorted(path.name for path in (data_root / "vectors" / "L21_V001").glob("*.npy")) == [
        "0.npy", "20.npy"
    ]
    document = json.loads((data_root / "metadata" / "L21_V001" / "Frame.json")
                          .read_text(encoding="utf-8"))
    assert document["execution_mode"] == "streaming_kept_only_v1"
    assert document["stage"] == "FINAL"
    assert [item["frame_id"] for item in document["frames"]] == ["0", "20"]

    resumed = pipeline.run_streaming(video, "L21_V001")
    assert resumed["kept_count"] == 2
    assert calls == [4]
