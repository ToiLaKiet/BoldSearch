import numpy as np
try:
    from sklearn.cluster import AgglomerativeClustering
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

def cluster_candidates(
    embeddings: np.ndarray,
    tau_cluster: float = 0.12,
) -> np.ndarray:
    """
    Gom cụm các frame ứng viên sử dụng Agglomerative Clustering với linkage='complete'.
    Complete linkage đảm bảo mọi cặp trong một cluster đều có khoảng cách cosine < tau_cluster.
    
    Parameters:
        embeddings: [num_candidates, embedding_dim] - Các vector embedding đã L2-normalize.
        tau_cluster: Ngưỡng khoảng cách cosine (max allowed distance). 
                     Nếu khoảng cách lớn hơn ngưỡng này, các cụm sẽ không gộp vào nhau.
                     
    Returns:
        np.ndarray: Mảng nhãn cụm (cluster labels) cho từng frame.
    """
    n = len(embeddings)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)

    if HAS_SKLEARN:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="complete",
            distance_threshold=tau_cluster
        )
        return clustering.fit_predict(embeddings)
    else:
        # Fallback thủ công nếu không cài sklearn:
        # Triển khai thuật toán gộp cụm Complete Linkage đơn giản
        labels = np.arange(n)
        
        # Tính ma trận khoảng cách cosine (1 - cos_sim)
        sim_matrix = embeddings @ embeddings.T
        dist_matrix = 1.0 - sim_matrix
        np.fill_diagonal(dist_matrix, 0.0)
        
        # Tạo cấu trúc cụm ban đầu
        clusters = {i: [i] for i in range(n)}
        
        while True:
            # Tìm cặp cụm có khoảng cách complete-linkage nhỏ nhất
            min_dist = np.inf
            merge_pair = None
            
            cluster_keys = list(clusters.keys())
            for idx_a in range(len(cluster_keys)):
                for idx_b in range(idx_a + 1, len(cluster_keys)):
                    c_a = cluster_keys[idx_a]
                    c_b = cluster_keys[idx_b]
                    
                    # Complete linkage: Khoảng cách tối đa giữa các điểm của 2 cụm
                    sub_dists = dist_matrix[np.ix_(clusters[c_a], clusters[c_b])]
                    max_d = np.max(sub_dists)
                    
                    if max_d < min_dist:
                        min_dist = max_d
                        merge_pair = (c_a, c_b)
            
            # Chỉ gộp nếu khoảng cách nhỏ hơn ngưỡng tau_cluster
            if merge_pair is not None and min_dist <= tau_cluster:
                c_keep, c_drop = merge_pair
                clusters[c_keep].extend(clusters[c_drop])
                del clusters[c_drop]
            else:
                break
                
        # Cập nhật nhãn cụm
        out_labels = np.zeros(n, dtype=int)
        for label_id, (key, val) in enumerate(clusters.items()):
            for idx in val:
                out_labels[idx] = label_id
        return out_labels
