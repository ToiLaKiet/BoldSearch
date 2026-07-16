from __future__ import annotations

import numpy as np


def select_representative(
    cluster_embeddings: np.ndarray,
    sharpness: np.ndarray | None = None,
    lambda_sharpness: float = 0.15,
) -> int:
    """Chọn medoid thật trong cluster, cộng thêm điểm sharpness."""
    embeddings = np.asarray(cluster_embeddings, dtype=float)
    if embeddings.ndim != 2:
        raise ValueError("cluster_embeddings must have shape [N, D]")
    if lambda_sharpness < 0:
        raise ValueError("lambda_sharpness must be non-negative")

    n = len(embeddings)
    if n == 0:
        raise ValueError("cluster_embeddings cannot be empty")
    if n == 1:
        return 0

    similarity = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    distances = 1.0 - similarity
    np.fill_diagonal(distances, 0.0)
    mean_distance = distances.sum(axis=1) / (n - 1)
    score = -mean_distance

    if sharpness is not None:
        quality = np.asarray(sharpness, dtype=float)
        if len(quality) != n:
            raise ValueError("sharpness length must match cluster size")
        if not np.all(np.isfinite(quality)):
            raise ValueError("sharpness contains NaN or Inf")

        spread = float(np.ptp(quality))
        if spread > 1e-8:
            normalized = (quality - quality.min()) / spread
            score = score + lambda_sharpness * normalized

    return int(np.argmax(score))


def farthest_point_selection(embeddings: np.ndarray, budget: int) -> list[int]:
    """Chọn tập đại diện đa dạng bằng greedy farthest-point selection.

    Đây là baseline diversity, không phải triển khai MaxInfo/SVD-MaxVol.
    """
    embeddings = np.asarray(embeddings, dtype=float)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [N, D]")
    if budget <= 0:
        raise ValueError("budget must be positive")

    n = len(embeddings)
    if n <= budget:
        return list(range(n))

    similarity = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    selected = [int(np.argmax(similarity.mean(axis=1)))]

    while len(selected) < budget:
        max_similarity = similarity[:, selected].max(axis=1)
        max_similarity[selected] = np.inf
        selected.append(int(np.argmin(max_similarity)))

    return selected
