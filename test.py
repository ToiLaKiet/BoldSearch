from __future__ import annotations

import numpy as np

from processframe.pipeline import build_deduplicated_index


def _normalize(values: np.ndarray) -> np.ndarray:
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def _cluster_of(frame_id: int, clusters: list[dict]) -> int:
    for cluster_id, cluster in enumerate(clusters):
        if frame_id in cluster["member_ids"]:
            return cluster_id
    raise AssertionError(f"frame {frame_id} is not covered")


def test_pipeline() -> None:
    print("=== Testing corrected deduplication pipeline ===")
    rng = np.random.default_rng(42)
    dim = 128

    embeddings = rng.normal(size=(10, dim))
    embeddings[1] = embeddings[0] + 0.01 * rng.normal(size=dim)
    embeddings[2] = embeddings[0] + 0.02 * rng.normal(size=dim)
    embeddings[7] = embeddings[6] + 0.01 * rng.normal(size=dim)
    embeddings[8] = embeddings[6] + 0.02 * rng.normal(size=dim)
    embeddings[9] = embeddings[6] + 0.01 * rng.normal(size=dim)
    embeddings = _normalize(embeddings)

    frames = [f"frame_{i:04d}.jpg" for i in range(10)]
    shot_ids = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    timestamps = np.arange(10, dtype=float) * 0.5
    sharpness = rng.uniform(0, 100, size=10)

    ocr_texts = [""] * 10
    ocr_texts[0] = "Apple"
    ocr_texts[1] = "Banana"
    objects_list = [[] for _ in range(10)]

    result = build_deduplicated_index(
        frames=frames,
        embeddings=embeddings,
        shot_ids=shot_ids,
        timestamps=timestamps,
        ocr_texts=ocr_texts,
        objects_list=objects_list,
        sharpness=sharpness,
        beta=1.5,
        window=3,
        tau_cluster=0.15,
        shot_budget=3,
    )

    covered = {
        member_id
        for cluster in result.clusters
        for member_id in cluster["member_ids"]
    }
    assert covered == set(range(10)), "Every frame must appear in exactly one cluster"
    assert sum(len(c["member_ids"]) for c in result.clusters) == 10

    # Visual embedding gần nhau nhưng OCR mâu thuẫn phải tách cluster.
    assert _cluster_of(0, result.clusters) != _cluster_of(1, result.clusters)

    # Bốn frame tĩnh của shot 1 phải được gộp.
    shot1_clusters = [c for c in result.clusters if c["shot_id"] == 1]
    assert len(shot1_clusters) == 1
    assert set(shot1_clusters[0]["member_ids"]) == {6, 7, 8, 9}

    # Mỗi representative của cluster nằm trong dedup representatives.
    assert all(
        cluster["representative_id"] in result.dedup_representatives
        for cluster in result.clusters
    )

    # Indexed representatives là tập sau diversity budget, có thể nhỏ hơn dedup set.
    assert set(result.indexed_representatives).issubset(result.dedup_representatives)
    for shot_id in np.unique(shot_ids):
        indexed_in_shot = [
            idx for idx in result.indexed_representatives if shot_ids[idx] == shot_id
        ]
        assert len(indexed_in_shot) <= 3

    print("Indexed representatives:", result.indexed_representatives)
    print("Dedup representatives:", result.dedup_representatives)
    print("Shot thresholds:", result.shot_thresholds)
    print("Clusters:")
    for cluster in result.clusters:
        print(cluster)
    print("Status: SUCCESS")


def test_static_plateau() -> None:
    base = np.arange(1, 17, dtype=float)
    embeddings = np.tile(base, (5, 1))
    embeddings = _normalize(embeddings)

    result = build_deduplicated_index(
        frames=list(range(5)),
        embeddings=embeddings,
        shot_ids=[0] * 5,
        shot_budget=None,
    )
    assert len(result.clusters) == 1
    assert result.clusters[0]["member_ids"] == [0, 1, 2, 3, 4]


def test_object_count_guard() -> None:
    base = np.arange(1, 17, dtype=float)
    embeddings = _normalize(np.vstack([base, base + 1e-5]))

    result = build_deduplicated_index(
        frames=["a", "b"],
        embeddings=embeddings,
        shot_ids=[0, 0],
        objects_list=[["person"], ["person", "person", "person"]],
        jaccard_object_thresh=0.8,
        shot_budget=None,
    )
    assert len(result.clusters) == 2


if __name__ == "__main__":
    test_pipeline()
    test_static_plateau()
    test_object_count_guard()
