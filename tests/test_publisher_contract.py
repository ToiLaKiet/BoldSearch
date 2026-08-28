import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from boldsearch_integration.publisher import (
    load_kept_frames,
    publish_manifest,
    resolve_active_release,
)


def make_pipeline_output(root: Path, *, dimension: int = 4) -> Path:
    video_id = "L21_V001"
    metadata = root / "metadata" / video_id
    frames = root / "frames" / video_id
    vectors = root / "vectors" / video_id
    metadata.mkdir(parents=True)
    frames.mkdir(parents=True)
    vectors.mkdir(parents=True)

    records = []
    for frame_id, shot_id, color in ((20, "shot_000002", (0, 100, 200)),
                                     (0, "shot_000001", (200, 100, 0))):
        png = frames / f"{frame_id}.png"
        Image.new("RGB", (120, 60), color).save(png)
        vector = np.zeros(dimension, dtype=np.float32)
        vector[0] = 1.0
        np.save(vectors / f"{frame_id}.npy", vector)
        records.append({
            "frame_id": str(frame_id),
            "frame_index": frame_id,
            "timestamp_ms": frame_id * 40,
            "shot_id": shot_id,
            "preliminary_status": "KEPT",
            "mapping_status": "MAPPED",
            "embedding_status": "EMBEDDED",
            "frame_path": str(png),
            "vector_path": str(vectors / f"{frame_id}.npy"),
            "final_status": "KEPT",
            "representative_frame_id": None,
            "similarity_score": None,
        })
    records.append({
        "frame_id": "10",
        "frame_index": 10,
        "timestamp_ms": 400,
        "shot_id": "shot_000001",
        "preliminary_status": "DUPLICATE",
        "mapping_status": "MAPPED",
        "embedding_status": "EMBEDDED",
        "frame_path": None,
        "vector_path": None,
        "final_status": "DUPLICATE",
        "representative_frame_id": "0",
        "similarity_score": 1.0,
    })
    (metadata / "Frame.json").write_text(json.dumps({
        "schema_version": "2.0",
        "video_id": video_id,
        "source_video_path": "L21_V001.mp4",
        "stage": "FINAL",
        "frames": records,
    }), encoding="utf-8")
    (metadata / "Shot.json").write_text(json.dumps({
        "schema_version": "2.0",
        "video_id": video_id,
        "source_video_path": "L21_V001.mp4",
        "source_video_checksum": "sha256:test-fixture",
        "fps": 25.0,
        "total_frames": 30,
        "width": 120,
        "height": 60,
        "codec": "fixture",
        "duration_ms": 1200,
        "detector": "fixture",
        "shots": [],
    }), encoding="utf-8")
    return root


def test_publisher_emits_stable_manifest_and_webp_derivatives(tmp_path: Path) -> None:
    data_root = make_pipeline_output(tmp_path / "data")

    report = publish_manifest(
        data_root=data_root,
        video_ids=["L21_V001"],
        output_root=tmp_path / "public",
        expected_vector_dim=4,
        thumbnail_width=80,
    )

    assert report.row_count == 2
    release = resolve_active_release(tmp_path / "public")
    assert release == report.release_root
    corpus = json.loads((release / "corpus-manifest.json").read_text(encoding="utf-8"))
    assert corpus["videos"]["L21_V001"]["source_video_checksum"] == "sha256:test-fixture"
    assert corpus["videos"]["L21_V001"]["frame_ids"] == [0, 20]
    with (release / "Frames.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {"video_id": "L21_V001", "frame_id": "0", "shot_id": "1"},
        {"video_id": "L21_V001", "frame_id": "20", "shot_id": "2"},
    ]
    image = release / "keyframes" / "L21_V001" / "0.webp"
    assert image.is_file()
    with Image.open(image) as decoded:
        assert decoded.format == "WEBP"
        assert decoded.width == 80


def test_invalid_vector_does_not_publish_or_replace_active_release(tmp_path: Path) -> None:
    data_root = make_pipeline_output(tmp_path / "data", dimension=4)
    public = tmp_path / "public"
    first = publish_manifest(
        data_root=data_root,
        video_ids=["L21_V001"],
        output_root=public,
        expected_vector_dim=4,
    )
    np.save(data_root / "vectors/L21_V001/20.npy", np.zeros(3, dtype=np.float32))

    with pytest.raises(ValueError, match="dimension"):
        publish_manifest(
            data_root=data_root,
            video_ids=["L21_V001"],
            output_root=public,
            expected_vector_dim=4,
        )

    assert resolve_active_release(public) == first.release_root


def test_load_kept_frames_rejects_non_numeric_shot_id(tmp_path: Path) -> None:
    data_root = make_pipeline_output(tmp_path / "data")
    frame_path = data_root / "metadata/L21_V001/Frame.json"
    document = json.loads(frame_path.read_text(encoding="utf-8"))
    document["frames"][0]["shot_id"] = "intro"
    frame_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="shot_id"):
        load_kept_frames(data_root, "L21_V001", expected_vector_dim=4)
