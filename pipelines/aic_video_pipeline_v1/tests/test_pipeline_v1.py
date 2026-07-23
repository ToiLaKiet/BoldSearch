from pathlib import Path

import numpy as np

from aic_video_pipeline_v1.models import FrameRecord
from aic_video_pipeline_v1.similarity import apply_representative_similarity, remove_duplicate_artifacts
from aic_video_pipeline_v1.storage import NpyVectorStore


def make_frame(index: int, shot: str = "shot_000001") -> FrameRecord:
    return FrameRecord("L21_V01", f"{index:08d}", index, index * 40, shot,
                       mapping_status="EXTRACTED", embedding_status="EMBEDDED")


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
    remove_duplicate_artifacts(frames)
    assert frames[1].frame_path is None and frames[1].vector_path is None
    assert not (tmp_path / "frames" / "L21_V01" / "00000001.png").exists()
    assert frames[0].frame_path is not None and Path(frames[0].frame_path).exists()
