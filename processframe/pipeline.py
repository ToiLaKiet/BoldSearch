from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import numpy as np

from .clustering import cluster_candidates, estimate_adaptive_threshold
from .guards import verify_duplicate_guard
from .local_dedup import select_dp_peaks
from .representative import farthest_point_selection, select_representative


@dataclass(slots=True)
class DeduplicationResult:
    """Kết quả tách riêng deduplication và diversity/index budget."""

    indexed_representatives: list[int]
    dedup_representatives: list[int]
    clusters: list[dict[str, Any]]
    shot_thresholds: dict[int, float]

    def __iter__(self) -> Iterator[object]:
        """Giữ tương thích với code cũ: ``representatives, clusters = result``."""
        yield self.indexed_representatives
        yield self.clusters


def _validate_inputs(
    frames: Sequence[str | int],
    embeddings: np.ndarray,
    shot_ids: Sequence[int],
    timestamps: Sequence[float] | None,
    ocr_texts: Sequence[str] | None,
    objects_list: Sequence[Sequence[object]] | None,
    sharpness: Sequence[float] | None,
    shot_budget: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[list[object]], np.ndarray]:
    embeddings = np.asarray(embeddings, dtype=float)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [N, D]")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("embeddings contain NaN or Inf")

    n = len(embeddings)
    if len(frames) != n:
        raise ValueError("frames length must match embeddings")

    shot_ids_array = np.asarray(shot_ids)
    if shot_ids_array.ndim != 1 or len(shot_ids_array) != n:
        raise ValueError("shot_ids must be a one-dimensional array of length N")

    if timestamps is None:
        timestamp_array = np.zeros(n, dtype=float)
    else:
        timestamp_array = np.asarray(timestamps, dtype=float)
        if timestamp_array.ndim != 1 or len(timestamp_array) != n:
            raise ValueError("timestamps must be a one-dimensional array of length N")
        if not np.all(np.isfinite(timestamp_array)):
            raise ValueError("timestamps contain NaN or Inf")

    if ocr_texts is None:
        ocr_values = [""] * n
    else:
        if len(ocr_texts) != n:
            raise ValueError("ocr_texts length must match embeddings")
        ocr_values = [str(value or "") for value in ocr_texts]

    if objects_list is None:
        object_values: list[list[object]] = [[] for _ in range(n)]
    else:
        if len(objects_list) != n:
            raise ValueError("objects_list length must match embeddings")
        object_values = [list(value or []) for value in objects_list]

    if sharpness is None:
        sharpness_array = np.zeros(n, dtype=float)
    else:
        sharpness_array = np.asarray(sharpness, dtype=float)
        if sharpness_array.ndim != 1 or len(sharpness_array) != n:
            raise ValueError("sharpness must be a one-dimensional array of length N")
        if not np.all(np.isfinite(sharpness_array)):
            raise ValueError("sharpness contains NaN or Inf")

    if shot_budget is not None and shot_budget <= 0:
        raise ValueError("shot_budget must be positive or None")

    norms = np.linalg.norm(embeddings, axis=1)
    zero_ids = np.flatnonzero(norms <= 1e-12)
    if zero_ids.size:
        raise ValueError(f"zero-norm embeddings at indices {zero_ids.tolist()}")

    normalized = embeddings / norms[:, None]
    return (
        normalized,
        shot_ids_array,
        timestamp_array,
        ocr_values,
        object_values,
        sharpness_array,
    )


def _guard_compatible_with_group(
    frame_idx: int,
    group: list[int],
    ocr_texts: list[str],
    objects_list: list[list[object]],
    ocr_similarity_thresh: float,
    jaccard_object_thresh: float,
) -> bool:
    return all(
        verify_duplicate_guard(
            ocr_texts[frame_idx],
            ocr_texts[member_idx],
            objects_list[frame_idx],
            objects_list[member_idx],
            ocr_similarity_thresh=ocr_similarity_thresh,
            jaccard_object_thresh=jaccard_object_thresh,
        )
        for member_idx in group
    )


def _visual_compatible_with_group(
    frame_idx: int,
    group: list[int],
    embeddings: np.ndarray,
    tau_cluster: float,
) -> tuple[bool, float]:
    similarities = np.clip(embeddings[group] @ embeddings[frame_idx], -1.0, 1.0)
    distances = 1.0 - similarities
    max_distance = float(np.max(distances))
    mean_distance = float(np.mean(distances))
    return max_distance <= tau_cluster, mean_distance


def _split_anchor_cluster_by_guards(
    members: list[int],
    embeddings: np.ndarray,
    ocr_texts: list[str],
    objects_list: list[list[object]],
    tau_cluster: float,
    ocr_similarity_thresh: float,
    jaccard_object_thresh: float,
) -> list[list[int]]:
    groups: list[list[int]] = []
    for frame_idx in sorted(members):
        best_group_id: int | None = None
        best_distance = np.inf

        for group_id, group in enumerate(groups):
            visual_ok, mean_distance = _visual_compatible_with_group(
                frame_idx, group, embeddings, tau_cluster
            )
            if not visual_ok:
                continue
            if not _guard_compatible_with_group(
                frame_idx,
                group,
                ocr_texts,
                objects_list,
                ocr_similarity_thresh,
                jaccard_object_thresh,
            ):
                continue
            if mean_distance < best_distance:
                best_distance = mean_distance
                best_group_id = group_id

        if best_group_id is None:
            groups.append([frame_idx])
        else:
            groups[best_group_id].append(frame_idx)

    return groups


def build_deduplicated_index(
    frames: Sequence[str | int],
    embeddings: np.ndarray,
    shot_ids: Sequence[int],
    timestamps: Sequence[float] | None = None,
    ocr_texts: Sequence[str] | None = None,
    objects_list: Sequence[Sequence[object]] | None = None,
    sharpness: Sequence[float] | None = None,
    beta: float = 2.0,
    window: int = 3,
    tau_cluster: float = 0.12,
    shot_budget: int | None = None,
    ocr_similarity_thresh: float = 0.85,
    jaccard_object_thresh: float = 0.80,
    adaptive_clustering: bool = True,
    min_change: float = 1e-4,
    lambda_sharpness: float = 0.15,
) -> DeduplicationResult:
    """Xây dựng tập keyframe deduplicated theo từng shot.

    Pipeline:
        1. DPSelect tạo anchor cục bộ nhưng không xóa frame.
        2. Complete-linkage clustering trên anchor.
        3. Chia lại theo OCR/Object guard.
        4. Gán toàn bộ frame còn lại vào cluster phù hợp; frame không phù hợp
           được promote thành cluster mới.
        5. Chọn medoid + sharpness cho từng duplicate cluster.
        6. Tùy chọn dùng FPS để tạo tập indexed representatives theo budget.

    Returns:
        ``DeduplicationResult``. Có thể unpack như code cũ:
        ``indexed_representatives, clusters = build_deduplicated_index(...)``.
    """
    if tau_cluster <= 0:
        raise ValueError("tau_cluster must be positive")

    (
        normalized_embeddings,
        shot_ids_array,
        timestamp_array,
        ocr_values,
        object_values,
        sharpness_array,
    ) = _validate_inputs(
        frames,
        embeddings,
        shot_ids,
        timestamps,
        ocr_texts,
        objects_list,
        sharpness,
        shot_budget,
    )

    n = len(normalized_embeddings)
    if n == 0:
        return DeduplicationResult([], [], [], {})

    clusters: list[dict[str, Any]] = []
    dedup_representatives: list[int] = []
    indexed_representatives: list[int] = []
    shot_thresholds: dict[int, float] = {}

    # Giữ thứ tự xuất hiện của shot thay vì sort cứng theo ID.
    unique_shots = list(dict.fromkeys(int(value) for value in shot_ids_array.tolist()))

    for shot_id in unique_shots:
        shot_indices = np.flatnonzero(shot_ids_array == shot_id).astype(int)
        shot_embeddings = normalized_embeddings[shot_indices]

        shot_tau = (
            estimate_adaptive_threshold(shot_embeddings, max_threshold=tau_cluster)
            if adaptive_clustering
            else float(tau_cluster)
        )
        shot_thresholds[int(shot_id)] = float(shot_tau)

        local_anchor_ids = select_dp_peaks(
            shot_embeddings,
            beta=beta,
            window=window,
            min_change=min_change,
        )
        anchor_indices = shot_indices[np.asarray(local_anchor_ids, dtype=int)]
        anchor_set = set(int(value) for value in anchor_indices)

        anchor_labels = cluster_candidates(
            normalized_embeddings[anchor_indices], tau_cluster=shot_tau
        )

        groups: list[list[int]] = []
        for label in np.unique(anchor_labels):
            visual_members = [
                int(value)
                for value in anchor_indices[np.flatnonzero(anchor_labels == label)]
            ]
            groups.extend(
                _split_anchor_cluster_by_guards(
                    visual_members,
                    normalized_embeddings,
                    ocr_values,
                    object_values,
                    shot_tau,
                    ocr_similarity_thresh,
                    jaccard_object_thresh,
                )
            )

        # Không xóa các frame DPSelect không chọn. Mỗi frame được gán vào cluster
        # tương thích nhất hoặc trở thành một cluster mới.
        for frame_idx in shot_indices:
            frame_idx = int(frame_idx)
            if frame_idx in anchor_set:
                continue

            best_group_id: int | None = None
            best_distance = np.inf
            for group_id, group in enumerate(groups):
                visual_ok, mean_distance = _visual_compatible_with_group(
                    frame_idx, group, normalized_embeddings, shot_tau
                )
                if not visual_ok:
                    continue
                if not _guard_compatible_with_group(
                    frame_idx,
                    group,
                    ocr_values,
                    object_values,
                    ocr_similarity_thresh,
                    jaccard_object_thresh,
                ):
                    continue
                if mean_distance < best_distance:
                    best_distance = mean_distance
                    best_group_id = group_id

            if best_group_id is None:
                groups.append([frame_idx])
            else:
                groups[best_group_id].append(frame_idx)

        groups = [sorted(group) for group in groups]
        groups.sort(key=lambda group: group[0])

        shot_cluster_records: list[dict[str, Any]] = []
        shot_rep_ids: list[int] = []
        for group in groups:
            member_array = np.asarray(group, dtype=int)
            rep_local = select_representative(
                normalized_embeddings[member_array],
                sharpness_array[member_array],
                lambda_sharpness=lambda_sharpness,
            )
            rep_idx = int(member_array[rep_local])
            shot_rep_ids.append(rep_idx)
            dedup_representatives.append(rep_idx)

            shot_cluster_records.append(
                {
                    "shot_id": int(shot_id),
                    "representative_id": rep_idx,
                    "representative_frame": frames[rep_idx],
                    "member_ids": [int(value) for value in member_array],
                    "member_frames": [frames[int(value)] for value in member_array],
                    "timestamps": [float(timestamp_array[int(value)]) for value in member_array],
                    "member_count": int(len(member_array)),
                    "cluster_threshold": float(shot_tau),
                    "indexed": False,
                }
            )

        if shot_budget is not None and len(shot_rep_ids) > shot_budget:
            selected_local = farthest_point_selection(
                normalized_embeddings[np.asarray(shot_rep_ids, dtype=int)],
                budget=shot_budget,
            )
            selected_shot_reps = [shot_rep_ids[i] for i in selected_local]
        else:
            selected_shot_reps = shot_rep_ids.copy()

        selected_set = set(selected_shot_reps)
        for record in shot_cluster_records:
            record["indexed"] = record["representative_id"] in selected_set

        indexed_representatives.extend(selected_shot_reps)
        clusters.extend(shot_cluster_records)

    indexed_representatives = sorted(set(int(value) for value in indexed_representatives))
    dedup_representatives = sorted(set(int(value) for value in dedup_representatives))

    covered = {
        int(member_id)
        for cluster in clusters
        for member_id in cluster["member_ids"]
    }
    expected = set(range(n))
    total_memberships = sum(len(c["member_ids"]) for c in clusters)
    duplicated_count = total_memberships - len(covered)
    if covered != expected or total_memberships != n:
        missing = sorted(expected - covered)
        raise RuntimeError(
            f"cluster coverage invariant failed: missing={missing}, "
            f"duplicate_memberships={duplicated_count}"
        )

    return DeduplicationResult(
        indexed_representatives=indexed_representatives,
        dedup_representatives=dedup_representatives,
        clusters=clusters,
        shot_thresholds=shot_thresholds,
    )
