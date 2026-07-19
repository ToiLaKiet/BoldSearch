import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from aic_video_pipeline.models import FrameRecord
from aic_video_pipeline.orchestrator import VideoPipelineOrchestrator
from aic_video_pipeline.storage import NpyFileVectorStore


def frame(index: int) -> FrameRecord:
    return FrameRecord("L21_V01", f"{index:08d}", index, index * 40, "shot_000001", "batch_000001", index + 1,
                       preliminary_status="KEPT", mapping_status="EXTRACTED", embedding_status="EMBEDDED")


def test_step_two_preserves_frame_names_and_statuses(tmp_path: Path) -> None:
    store = NpyFileVectorStore(tmp_path / "data" / "vectors", "unused")
    frames = [frame(index) for index in range(5)]
    vectors = [np.array([1, 0], dtype=np.float32), np.array([1, 0], dtype=np.float32),
               np.array([1, 0], dtype=np.float32), np.array([0, 1], dtype=np.float32), np.array([1, 0], dtype=np.float32)]
    for record, vector in zip(frames, vectors):
        record.vector_path = str(store.put(record.video_id, record.frame_id, vector))
        assert Path(record.vector_path) == tmp_path / "data" / "vectors" / "L21_V01" / f"{record.frame_id}.npy"

    VideoPipelineOrchestrator._apply_similarity(frames, store, 0.4)
    assert [item.final_status for item in frames] == ["DUPLICATE", "KEPT", "KEPT", "KEPT", "KEPT"]
    assert frames[0].representative_frame_id == "00000001"
    assert frames[4].similarity_score is None


def test_dimension_mismatch_raises(tmp_path: Path) -> None:
    store = NpyFileVectorStore(tmp_path / "data" / "vectors", "unused")
    first, second = frame(1), frame(2)
    first.vector_path = str(store.put(first.video_id, first.frame_id, np.array([1, 0], dtype=np.float32)))
    second.vector_path = str(store.put(second.video_id, second.frame_id, np.array([1, 0, 0], dtype=np.float32)))
    with pytest.raises(ValueError, match="dimension mismatch"):
        VideoPipelineOrchestrator._apply_similarity([first, second], store, 0.4)


def _config(data_root: Path) -> dict:
    return {
        "paths": {"data_root": str(data_root)}, "pipeline": {"batch_size": 10},
        "frame_id": {"zero_pad_width": 8}, "indexer": {"sample_every_frames": 2},
        "preliminary_dedup": {"enabled": False, "strategy": "none"},
        "autoshot": {"root": "/not-installed", "checkpoint": "/not-installed/model.pth", "allow_fallback": True},
        "mapping": {"overwrite_existing": False},
        "embedding": {"provider": "histogram", "model_id": "not-used", "model_version": "fgclip_v1", "batch_size": 10},
        "similarity": {"threshold": 0.4, "step": 2},
    }


def _video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
    assert writer.isOpened()
    for index in range(8):
        writer.write(np.full((24, 32, 3), (index * 30, 20, 40), dtype=np.uint8))
    writer.release()


def test_plan_two_outputs_only(tmp_path: Path) -> None:
    data = tmp_path / "data"
    source = data / "videos" / "L21_V01.mp4"
    _video(source)
    pipeline = VideoPipelineOrchestrator(_config(data))
    result = pipeline.run(source, "histogram")
    assert result["video_id"] == "L21_V01"

    metadata = data / "metadata" / "L21_V01"
    assert sorted(path.name for path in metadata.iterdir()) == ["Frame.json", "Shot.json"]
    frame_doc = json.loads((metadata / "Frame.json").read_text(encoding="utf-8"))
    assert set(frame_doc) == {"video_id", "source_video_path", "batch_size", "frames"}
    assert set(frame_doc["frames"][0]) == {"frame_id", "frame_index", "timestamp_ms", "shot_id", "preliminary_status", "frame_path", "vector_path", "final_status", "representative_frame_id", "similarity_score"}
    assert (data / "frames" / "L21_V01" / "00000000.png").is_file()
    assert (data / "vectors" / "L21_V01" / "00000000.npy").is_file()
    assert not (data / "runs").exists()
    assert not (data / "logs").exists()
    pipeline.validate("L21_V01")
