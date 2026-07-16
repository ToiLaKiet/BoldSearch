import numpy as np

def select_dp_peaks(
    embeddings: np.ndarray,
    beta: float = 2.0,
    window: int = 3,
) -> list[int]:
    """
    DPSelect (Dist-Peak Select) với ngưỡng thích ứng (adaptive threshold).
    
    Parameters:
        embeddings: np.ndarray của các frame embedding dạng [num_frames, embedding_dim]. 
                    Yêu cầu các vector đã được L2-normalize.
        beta: Tham số điều chỉnh độ nhạy của ngưỡng phát hiện đỉnh (prominence factor).
              Giá trị thường dùng: 1.5 - 3.0.
        window: Kích thước cửa sổ (window size) tìm kiếm local maximum (thường dùng = 3).
        
    Returns:
        list[int]: Danh sách các chỉ mục (indices) của các frame được giữ lại.
    """
    n = len(embeddings)
    if n == 0:
        return []
    if n <= 2:
        return list(range(n))

    # Tính cosine similarity giữa các frame liền kề: cos_sim(e_t, e_{t+1})
    similarities = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
    
    # Cosine distance/dissimilarity d_t = 1 - cos_sim
    distances = 1.0 - similarities

    # Tính median absolute deviation (MAD) để làm ngưỡng thích ứng
    median = np.median(distances)
    mad = np.median(np.abs(distances - median))
    
    # 1.4826 là hệ số nhân để MAD tương đương Standard Deviation của phân phối chuẩn
    threshold = median + beta * 1.4826 * mad

    radius = window // 2
    selected = {0, n - 1}  # Luôn luôn giữ frame đầu và cuối shot

    for i in range(len(distances)):
        left = max(0, i - radius)
        # i + radius + 1 để bao gồm cả phần tử bên phải khi slice
        right = min(len(distances), i + radius + 1)

        # Kiểm tra xem khoảng cách d_i có phải là cực đại địa phương trong window hay không
        is_local_peak = distances[i] >= np.max(distances[left:right])

        # Đỉnh phải vượt qua ngưỡng thích ứng để tránh giữ các biến đổi nhiễu trong cảnh tĩnh
        if is_local_peak and distances[i] >= threshold:
            # Khoảng cách distances[i] thể hiện sự thay đổi giữa frame i và i+1
            # Do đó chúng ta chọn giữ frame tiếp theo (i+1)
            selected.add(i + 1)

    return sorted(selected)
