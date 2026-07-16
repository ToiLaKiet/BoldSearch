from __future__ import annotations

import numpy as np

try:
    from sklearn.cluster import AgglomerativeClustering

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def pairwise_cosine_distance(embeddings: np.ndarray) -> np.ndarray:
    """Tính ma trận khoảng cách cosine, giả định vector đã L2-normalize."""
    embeddings = np.asarray(embeddings, dtype=float)
    similarity = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    return distance


def estimate_adaptive_threshold(
    embeddings: np.ndarray,
    max_threshold: float = 0.12,
    min_threshold: float | None = None,
    mad_scale: float = 1.4826,
) -> float:
    """Ước lượng ngưỡng clustering bảo thủ cho từng shot.

    Ngưỡng được suy ra từ khoảng cách nearest-neighbor của từng frame và luôn
    bị chặn không vượt quá ``max_threshold``. Vì vậy cơ chế adaptive chỉ làm
    ngưỡng chặt hơn, không làm tăng nguy cơ false merge so với cấu hình gốc.
    """
    embeddings = np.asarray(embeddings, dtype=float)
    n = len(embeddings)
    if n <= 2:
        return float(max_threshold)
    if max_threshold <= 0:
        raise ValueError("max_threshold must be positive")

    if min_threshold is None:
        min_threshold = max(1e-3, max_threshold * 0.35)
    if not 0 < min_threshold <= max_threshold:
        raise ValueError("min_threshold must be in (0, max_threshold]")

    distances = pairwise_cosine_distance(embeddings)
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    nearest = nearest[np.isfinite(nearest)]
    if nearest.size == 0:
        return float(max_threshold)

    median = float(np.median(nearest))
    mad = float(np.median(np.abs(nearest - median)))
    estimate = median + mad_scale * mad
    return float(np.clip(estimate, min_threshold, max_threshold))


def cluster_candidates(
    embeddings: np.ndarray,
    tau_cluster: float = 0.12,
) -> np.ndarray:
    """Gom cụm bằng Agglomerative Complete Linkage và cosine distance.

    Complete linkage bảo đảm khoảng cách lớn nhất giữa hai điểm thuộc cùng cụm
    không vượt quá ngưỡng tại thời điểm gộp, phù hợp hơn connected components
    cho duplicate detection.
    """
    embeddings = np.asarray(embeddings, dtype=float)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [N, D]")
    if tau_cluster <= 0:
        raise ValueError("tau_cluster must be positive")

    n = len(embeddings)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)

    if HAS_SKLEARN:
        kwargs = dict(
            n_clusters=None,
            linkage="complete",
            distance_threshold=float(tau_cluster),
        )
        try:
            model = AgglomerativeClustering(metric="cosine", **kwargs)
        except TypeError:  # sklearn cũ
            model = AgglomerativeClustering(affinity="cosine", **kwargs)
        return model.fit_predict(embeddings).astype(int)

    distances = pairwise_cosine_distance(embeddings)
    clusters: dict[int, list[int]] = {i: [i] for i in range(n)}

    while True:
        min_distance = np.inf
        merge_pair: tuple[int, int] | None = None
        keys = list(clusters)

        for i, key_a in enumerate(keys):
            for key_b in keys[i + 1 :]:
                complete_distance = float(
                    np.max(distances[np.ix_(clusters[key_a], clusters[key_b])])
                )
                if complete_distance < min_distance:
                    min_distance = complete_distance
                    merge_pair = (key_a, key_b)

        if merge_pair is None or min_distance > tau_cluster:
            break

        keep, drop = merge_pair
        clusters[keep].extend(clusters.pop(drop))

    labels = np.empty(n, dtype=int)
    for label, members in enumerate(clusters.values()):
        labels[members] = label
    return labels
