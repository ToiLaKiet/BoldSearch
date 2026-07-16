from __future__ import annotations

import numpy as np


def select_dp_peaks(
    embeddings: np.ndarray,
    beta: float = 2.0,
    window: int = 3,
    min_change: float = 1e-4,
    ensure_change_anchor: bool = True,
) -> list[int]:
    """Chọn các anchor theo biến đổi cục bộ giữa những frame liên tiếp.

    Đây là phiên bản lấy cảm hứng từ DPSelect của ReTaKe. Hàm chỉ tạo các
    anchor ban đầu; các frame không được chọn vẫn phải được gán lại vào cluster
    trong pipeline, tuyệt đối không bị xóa khỏi metadata.

    Args:
        embeddings: Ma trận [N, D], kỳ vọng đã L2-normalize.
        beta: Hệ số của ngưỡng thích ứng median + beta * 1.4826 * MAD.
        window: Kích thước cửa sổ local maximum. Giá trị chẵn sẽ được tăng 1.
        min_change: Khoảng cách cosine tối thiểu để xem là thay đổi có ý nghĩa.
        ensure_change_anchor: Nếu không tìm được peak nhưng có biến đổi rõ ràng,
            giữ vị trí có dissimilarity lớn nhất làm anchor dự phòng.

    Returns:
        Danh sách chỉ mục tương đối trong shot.
    """
    embeddings = np.asarray(embeddings, dtype=float)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [N, D]")
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if min_change < 0:
        raise ValueError("min_change must be non-negative")

    n = len(embeddings)
    if n == 0:
        return []
    if n <= 2:
        return list(range(n))

    if window < 3:
        window = 3
    if window % 2 == 0:
        window += 1

    similarities = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
    similarities = np.clip(similarities, -1.0, 1.0)
    distances = 1.0 - similarities

    median = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median)))
    raw_threshold = median + beta * 1.4826 * mad

    # Không để MAD threshold vượt quá toàn bộ tín hiệu và làm mất mọi biến đổi.
    # Quantile 90% vẫn giữ bộ lọc bảo thủ nhưng đảm bảo peak nổi bật có cơ hội được chọn.
    robust_cap = float(np.quantile(distances, 0.90))
    threshold = max(min_change, min(raw_threshold, robust_cap))

    radius = window // 2
    selected: set[int] = {0, n - 1}
    added_change_anchor = False

    for i, current in enumerate(distances):
        left = max(0, i - radius)
        right = min(len(distances), i + radius + 1)
        local_values = distances[left:right]
        local_max = float(np.max(local_values))

        # Với plateau, chỉ lấy vị trí đầu tiên đạt cực đại để tránh giữ mọi điểm.
        first_local_max = left + int(np.argmax(local_values))
        is_local_peak = bool(np.isclose(current, local_max)) and i == first_local_max

        if is_local_peak and current >= threshold and current >= min_change:
            selected.add(i + 1)
            added_change_anchor = True

    if ensure_change_anchor and not added_change_anchor:
        max_i = int(np.argmax(distances))
        if float(distances[max_i]) >= min_change:
            selected.add(max_i + 1)

    return sorted(selected)
