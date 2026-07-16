import numpy as np
from processframe.pipeline import build_deduplicated_index

def test_pipeline():
    print("=== Testing Deduplication Pipeline ===")
    
    # 1. Tạo dữ liệu giả lập (10 frames trong 2 shots)
    # Shot 0: 6 frames (3 frame đầu giống nhau, 3 frame sau thay đổi động nhẹ)
    # Shot 1: 4 frames tĩnh
    num_frames = 10
    dim = 128
    
    # Sinh embeddings ngẫu nhiên
    np.random.seed(42)
    embs = np.random.randn(num_frames, dim)
    
    # Shot 0: Gán 3 frame đầu tiên gần như trùng lặp (cosine similarity rất cao)
    embs[1] = embs[0] + 0.01 * np.random.randn(dim)
    embs[2] = embs[0] + 0.02 * np.random.randn(dim)
    
    # Shot 1: Gán 4 frame trùng lặp nhau
    embs[7] = embs[6] + 0.01 * np.random.randn(dim)
    embs[8] = embs[6] + 0.02 * np.random.randn(dim)
    embs[9] = embs[6] + 0.01 * np.random.randn(dim)

    # Normalize các vector giả lập
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    frames = [f"frame_{i:04d}.jpg" for i in range(num_frames)]
    shot_ids = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    timestamps = [float(i * 0.5) for i in range(num_frames)]
    
    # Thử nghiệm với OCR văn bản
    # Cố tình đặt ocr khác nhau ở 2 frame trùng của Shot 0 để kích hoạt OCR guard
    ocr_texts = [""] * num_frames
    ocr_texts[0] = "Apple"
    ocr_texts[1] = "Banana"  # Khác biệt ocr với frame 0 mặc dù embedding giống nhau
    
    objects_list = [[] for _ in range(num_frames)]
    sharpness = np.random.rand(num_frames) * 100

    print(f"Input: {num_frames} frames across {len(np.unique(shot_ids))} shots.")
    
    # Gọi hàm pipeline
    representatives, clusters = build_deduplicated_index(
        frames=frames,
        embeddings=embs,
        shot_ids=shot_ids,
        timestamps=timestamps,
        ocr_texts=ocr_texts,
        objects_list=objects_list,
        sharpness=sharpness,
        beta=1.5,
        window=3,
        tau_cluster=0.15,
        shot_budget=3
    )

    print("\n--- RESULTS ---")
    print(f"Selected representative frames (indices): {representatives}")
    print(f"Selected representative filenames: {[frames[i] for i in representatives]}")
    print(f"Reduction Ratio: {(1 - len(representatives)/num_frames)*100:.1f}%")
    
    print("\nClusters Metadata:")
    for cluster in clusters:
        print(f"Cluster (Shot {cluster['shot_id']}):")
        print(f"  Representative: {cluster['member_frames'][0]} (Index {cluster['representative_id']})")
        print(f"  Members: {cluster['member_frames']}")
        print(f"  Timestamps: {cluster['timestamps']}")
        
    # Một số kiểm tra cơ bản
    assert len(representatives) > 0, "No keyframes selected!"
    assert all(r in representatives for r in [c['representative_id'] for c in clusters]), "Medoid mismatch in representatives!"
    print("\nStatus: SUCCESS - Pipeline is running correctly.")

if __name__ == "__main__":
    test_pipeline()
