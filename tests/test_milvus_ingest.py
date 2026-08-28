from pathlib import Path

import numpy as np
import pytest

from boldsearch_integration.milvus_ingest import (
    build_milvus_rows,
    ingest_rows,
    stable_primary_key,
)
from test_publisher_contract import make_pipeline_output


def test_rows_are_deterministic_and_keep_visual_modality_only(tmp_path: Path) -> None:
    data_root = make_pipeline_output(tmp_path / "data", dimension=4)

    first = build_milvus_rows(
        data_root=data_root,
        video_ids=["L21_V001"],
        corpus_version="v1",
        expected_vector_dim=4,
        thumbnail_base="/keyframes",
    )
    second = build_milvus_rows(
        data_root=data_root,
        video_ids=["L21_V001"],
        corpus_version="v1",
        expected_vector_dim=4,
        thumbnail_base="/keyframes",
    )

    assert first == second
    assert [row["frame_id"] for row in first] == [0, 20]
    assert all("caption_embedding" not in row for row in first)
    assert first[0]["thumbnail"] == "/keyframes/L21_V001/0.webp"
    assert first[0]["id"] == stable_primary_key("v1", "L21_V001", 0)


def test_ingest_batches_and_checks_acknowledged_rows() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls = []

        def upsert(self, *, collection_name, data):
            self.calls.append((collection_name, data))
            return {"upsert_count": len(data)}

    rows = [
         {"id": index + 1, "video_id": "L21_V001", "frame_id": index,
         "shot_id": 1, "visual_embedding": [1.0, 0.0]}
        for index in range(5)
    ]
    client = FakeClient()

    assert ingest_rows(client, "BoldSearch", rows, batch_size=2) == 5
    assert [len(data) for _, data in client.calls] == [2, 2, 1]
    assert all(name == "BoldSearch" for name, _ in client.calls)


def test_ingest_rejects_duplicate_or_non_normalized_rows() -> None:
    row = {"id": 1, "video_id": "L21_V001", "frame_id": 0,
           "shot_id": 1, "visual_embedding": [1.0, 0.0]}
    with pytest.raises(ValueError, match="duplicate"):
        ingest_rows(object(), "BoldSearch", [row, dict(row)], batch_size=2)

    bad = dict(row, visual_embedding=[2.0, 0.0])
    with pytest.raises(ValueError, match="normalized"):
        ingest_rows(object(), "BoldSearch", [bad], batch_size=2)


def test_row_projection_rejects_unsafe_thumbnail_base(tmp_path: Path) -> None:
    data_root = make_pipeline_output(tmp_path / "data", dimension=4)
    with pytest.raises(ValueError, match="thumbnail_base"):
        build_milvus_rows(
            data_root=data_root,
            video_ids=["L21_V001"],
            corpus_version="v1",
            expected_vector_dim=4,
            thumbnail_base="/keyframes/../private",
        )
