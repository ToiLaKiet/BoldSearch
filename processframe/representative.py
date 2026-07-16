import numpy as np

def select_representative(
    cluster_embeddings: np.ndarray,
    sharpness: np.ndarray | None = None,
    lambda_sharpness: float = 0.15,
) -> int:
    """
    Chọn frame đại diện (Medoid) trong cụm, kết hợp thêm chỉ số độ sắc nét (sharpness) 
    để tránh chọn phải frame mờ nhòe.
    
    Parameters:
        cluster_embeddings: [num_cluster_members, embedding_dim] - Các vector embedding đã normalize.
        sharpness: [num_cluster_members] - Chỉ số sắc nét của các frame tương ứng trong cụm (optional).
        lambda_sharpness: Hệ số trọng số của độ sắc nét (sharpness).
        
    Returns:
        int: Chỉ mục tương đối (0 đến len-1) của frame đại diện được chọn trong cụm.
    """
    n = len(cluster_embeddings)
    if n == 0:
        raise ValueError("cluster_embeddings cannot be empty.")
    if n == 1:
        return 0

    # Tính ma trận tương đồng cosine
    similarity = cluster_embeddings @ cluster_embeddings.T
    
    # Khoảng cách cosine trung bình của mỗi frame tới các frame khác trong cụm
    mean_distance = np.mean(1.0 - similarity, axis=1)

    # Điểm đại diện (gần tâm nhất -> khoảng cách trung bình nhỏ nhất -> -mean_distance lớn nhất)
    score = -mean_distance

    # Cộng thêm điểm chất lượng ảnh (sharpness) nếu có
    if sharpness is not None and len(sharpness) == n:
        sharpness_arr = np.array(sharpness, dtype=float)
        ptp = np.ptp(sharpness_arr)
        if ptp > 1e-8:
            # Chuẩn hóa về [0, 1]
            normalized_sharpness = (sharpness_arr - sharpness_arr.min()) / ptp
        else:
            normalized_sharpness = np.zeros(n)
            
        score += lambda_sharpness * normalized_sharpness

    return int(np.argmax(score))


def farthest_point_selection(
    embeddings: np.ndarray,
    budget: int,
) -> list[int]:
    """
    Thuật toán Farthest Point Selection (FPS) để lọc ra tập các frame đa dạng nhất
    (diversity budget) từ một tập các frame đại diện.
    
    Parameters:
        embeddings: [num_frames, embedding_dim] - Các vector embedding đã normalize.
        budget: Số lượng frame tối đa được phép giữ lại.
        
    Returns:
        list[int]: Danh sách các chỉ mục của các frame được chọn.
    """
    n = len(embeddings)
    if n <= budget:
        return list(range(n))

    similarity = embeddings @ embeddings.T

    # Khởi đầu bằng cách chọn frame có độ đại diện bao quát cao nhất (gần tâm nhất của toàn tập)
    selected = [int(np.argmax(similarity.mean(axis=1)))]

    while len(selected) < budget:
        # Tìm khoảng cách tương đồng lớn nhất của từng ứng viên tới các điểm đã được chọn
        max_sim_to_selected = similarity[:, selected].max(axis=1)
        
        # Bỏ qua các điểm đã chọn bằng cách gán độ tương đồng vô hạn
        max_sim_to_selected[selected] = np.inf

        # Chọn điểm có độ tương đồng lớn nhất tới tập đã chọn nhỏ nhất (tức là xa tập đã chọn nhất)
        next_index = int(np.argmin(max_sim_to_selected))
        selected.append(next_index)

    return selected
