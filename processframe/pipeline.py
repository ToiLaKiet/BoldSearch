import numpy as np
from .local_dedup import select_dp_peaks
from .clustering import cluster_candidates
from .representative import select_representative, farthest_point_selection
from .guards import verify_duplicate_guard

def build_deduplicated_index(
    frames: list[str] | list[int],
    embeddings: np.ndarray,
    shot_ids: np.ndarray,
    timestamps: np.ndarray | list[float] | None = None,
    ocr_texts: list[str] | None = None,
    objects_list: list[list[str]] | None = None,
    sharpness: np.ndarray | None = None,
    beta: float = 2.0,
    window: int = 3,
    tau_cluster: float = 0.12,
    shot_budget: int = 5,
    ocr_similarity_thresh: float = 0.85,
    jaccard_object_thresh: float = 0.80,
) -> tuple[list[int], list[dict]]:
    """
    Pipeline hoàn chỉnh xây dựng tập keyframe không trùng lặp (Offline Indexing).
    Quy trình: Chia shot -> DPSelect local peaks -> Adaptive clustering -> Guards -> Medoid -> Farthest Point.
    
    Parameters:
        frames: Danh sách định danh frame (tên file hoặc index).
        embeddings: [num_frames, embedding_dim] - Ma trận vector embeddings.
        shot_ids: Mảng định danh shot tương ứng với mỗi frame.
        timestamps: Mảng timestamp (giây) tương ứng với mỗi frame (optional).
        ocr_texts: Danh sách text OCR tương ứng với mỗi frame (optional).
        objects_list: Danh sách các đối tượng thị giác tương ứng với mỗi frame (optional).
        sharpness: Mảng độ sắc nét (sharpness) tương ứng với mỗi frame (optional).
        beta: Trọng số lọc đỉnh của DPSelect.
        window: Cửa sổ tìm đỉnh cục bộ của DPSelect.
        tau_cluster: Ngưỡng khoảng cách phân cụm.
        shot_budget: Số lượng keyframe tối đa cho phép trong mỗi shot.
        ocr_similarity_thresh: Ngưỡng tương đồng OCR để gộp nhóm.
        jaccard_object_thresh: Ngưỡng tương đồng Object để gộp nhóm.
        
    Returns:
        tuple: (final_representative_indices, clusters_metadata)
            - final_representative_indices: list các index của các frame đại diện được giữ lại.
            - clusters_metadata: list các dict chứa thông tin chi tiết về từng cụm trùng lặp.
    """
    n = len(embeddings)
    if n == 0:
        return [], []

    # Đảm bảo embeddings đã được L2-normalized
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Tránh chia cho 0
    norms[norms == 0] = 1e-8
    normalized_embeddings = embeddings / norms

    # Khởi tạo giá trị mặc định nếu rỗng
    if timestamps is None:
        timestamps = np.zeros(n)
    else:
        timestamps = np.array(timestamps)

    if ocr_texts is None:
        ocr_texts = [""] * n

    if objects_list is None:
        objects_list = [[] for _ in range(n)]

    if sharpness is None:
        sharpness = np.zeros(n)

    all_clusters = []
    final_representatives = []

    unique_shots = np.unique(shot_ids)
    
    for shot_id in unique_shots:
        # Lấy tất cả các frame thuộc shot hiện tại
        indices = np.where(shot_ids == shot_id)[0]
        if len(indices) == 0:
            continue

        shot_embeddings = normalized_embeddings[indices]

        # 1. RETAKE DPSelect: Lọc thô cục bộ các thay đổi động chính (Local Novelty)
        local_ids = select_dp_peaks(shot_embeddings, beta=beta, window=window)
        candidate_indices = indices[local_ids]
        candidate_embeddings = normalized_embeddings[candidate_indices]

        # 2. LMSKE Adaptive Clustering: Phân cụm các ứng viên có nội dung tĩnh giống nhau
        labels = cluster_candidates(candidate_embeddings, tau_cluster=tau_cluster)
        
        shot_representatives = []
        unique_labels = np.unique(labels)

        # 3. Chọn Medoid & Kiểm tra Guards
        for label in unique_labels:
            member_mask = labels == label
            member_indices = candidate_indices[member_mask]

            # Tìm frame đại diện tạm thời (Medoid) trong cụm ban đầu
            sub_embeddings = normalized_embeddings[member_indices]
            sub_sharpness = sharpness[member_indices]
            
            rep_local_idx = select_representative(sub_embeddings, sub_sharpness)
            rep_idx = member_indices[rep_local_idx]

            # Kiểm tra guards bảo vệ thông tin OCR/Object so với frame đại diện
            verified_members = [rep_idx]
            for m_idx in member_indices:
                if m_idx == rep_idx:
                    continue
                    
                # So sánh frame thành viên với frame đại diện qua lớp Guard
                if verify_duplicate_guard(
                    ocr_texts[rep_idx], ocr_texts[m_idx],
                    objects_list[rep_idx], objects_list[m_idx],
                    ocr_similarity_thresh=ocr_similarity_thresh,
                    jaccard_object_thresh=jaccard_object_thresh
                ):
                    verified_members.append(m_idx)
                else:
                    # Vi phạm Guard -> Không cho gộp, tách frame này làm cụm riêng biệt
                    final_representatives.append(m_idx)
                    all_clusters.append({
                        "shot_id": int(shot_id),
                        "representative_id": int(m_idx),
                        "member_ids": [int(m_idx)],
                        "member_frames": [frames[m_idx]],
                        "timestamps": [float(timestamps[m_idx])],
                    })

            # Lưu cụm đã qua kiểm duyệt
            all_clusters.append({
                "shot_id": int(shot_id),
                "representative_id": int(rep_idx),
                "member_ids": [int(x) for x in verified_members],
                "member_frames": [frames[x] for x in verified_members],
                "timestamps": [float(timestamps[x]) for x in verified_members],
            })
            shot_representatives.append(rep_idx)

        # 4. MaxInfo / Farthest Point Selection: Khống chế budget tối đa cho mỗi shot
        if len(shot_representatives) > shot_budget:
            # Chọn ra tập keyframes đại diện có độ bao phủ đa dạng nhất
            diverse_local_ids = farthest_point_selection(
                normalized_embeddings[shot_representatives], 
                budget=shot_budget
            )
            shot_representatives = [shot_representatives[x] for x in diverse_local_ids]

        final_representatives.extend(shot_representatives)

    return sorted(final_representatives), all_clusters
