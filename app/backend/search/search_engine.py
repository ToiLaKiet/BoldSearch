import numpy as np
from collections import defaultdict
import os
from dotenv import load_dotenv


# Đọc biến từ file .env
load_dotenv()

# Lấy API key
milvus_key = os.getenv("MILVUS_TOKEN")
ocr_key = os.getenv("OCR_API_KEY")
asr_key = os.getenv("ASR_API_KEY")


def normalize_top_k(results, score_key="scores", higher_is_better=True):
    """
    Chuẩn hóa score trong một danh sách kết quả về khoảng [0, 1].

    Args:
        results: Danh sách kết quả dạng:
            [
                {
                    "video_id": ...,
                    "frame_id": ...,
                    "scores": ...
                },
                ...
            ]
        score_key: Tên trường chứa score.
        higher_is_better:
            True  -> score càng lớn càng tốt, ví dụ cosine similarity.
            False -> score càng nhỏ càng tốt, ví dụ L2 distance.

    Returns:
        Danh sách kết quả có thêm:
            raw_score: score ban đầu
            normalized_score: score đã chuẩn hóa
    """
    if not results:
        return []

    raw_scores = [float(item[score_key]) for item in results]

    min_score = min(raw_scores)
    max_score = max(raw_scores)

    normalized_results = []

    for item, raw_score in zip(results, raw_scores):
        new_item = item.copy()
        new_item["raw_score"] = raw_score

        if max_score == min_score:
            # Tất cả kết quả có cùng score
            normalized_score = 1.0
        else:
            normalized_score = (
                (raw_score - min_score) /
                (max_score - min_score)
            )

        # Với distance: giá trị nhỏ hơn phải có normalized score lớn hơn
        if not higher_is_better:
            normalized_score = 1.0 - normalized_score

        new_item["normalized_score"] = normalized_score
        normalized_results.append(new_item)

    return normalized_results

def search_milvus(query, video_id=None, start_end=None):
    pass

def search_ocr(query, video_id=None, start_end=None):
    pass

def search_asr(query, video_id=None, start_end=None):
    pass

from collections import defaultdict


def merge_and_mean_scores(
    milvus_results,
    ocr_results,
    asr_results,
    missing_score=0.0
):
    """
    Gộp kết quả theo video_id và frame_id, sau đó tính mean score
    của ba luồng Milvus, OCR và ASR.

    Args:
        milvus_results: Kết quả Milvus đã normalize.
        ocr_results: Kết quả OCR đã normalize.
        asr_results: Kết quả ASR đã normalize.
        missing_score:
            Điểm dùng khi frame không xuất hiện trong một luồng.
            Mặc định là 0.0.

    Returns:
        Danh sách đã gộp và sắp xếp theo mean_score giảm dần.
    """
    grouped_results = defaultdict(
        lambda: {
            "milvus_score": None,
            "ocr_score": None,
            "asr_score": None
        }
    )

    # Gộp kết quả Milvus
    for item in milvus_results:
        key = (
            item["video_id"],
            item["frame_id"]
        )

        grouped_results[key]["milvus_score"] = item[
            "normalized_score"
        ]

    # Gộp kết quả OCR
    if ocr_results is not None:
        for item in ocr_results:
            key = (
                item["video_id"],
                item["frame_id"]
            )

            grouped_results[key]["ocr_score"] = item[
                "normalized_score"
            ]

    # Gộp kết quả ASR
    if asr_results is not None:
        for item in asr_results:
            key = (
            item["video_id"],
            item["frame_id"]
        )

        grouped_results[key]["asr_score"] = item[
            "normalized_score"
        ]

    final_results = []

    for (video_id, frame_id), scores in grouped_results.items():
        milvus_score = (
            scores["milvus_score"]
            if scores["milvus_score"] is not None
            else missing_score
        )

        ocr_score = (
            scores["ocr_score"]
            if scores["ocr_score"] is not None
            else missing_score
        )

        asr_score = (
            scores["asr_score"]
            if scores["asr_score"] is not None
            else missing_score
        )

        mean_score = (
            milvus_score
            + ocr_score
            + asr_score
        ) / 3

        final_results.append({
            "video_id": video_id,
            "frame_id": frame_id,
            # "milvus_score": milvus_score,
            # "ocr_score": ocr_score,
            # "asr_score": asr_score,
            "score": mean_score
        })

    final_results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return final_results

def search_query(query, video_id=None, start_end=None):
    '''
    Nhận input đầu vào là câu query, qua pipeline xử lý sẽ 
    trả về danh sách các frame kết quả phù hợp. 
    Có cấu trúc :
    [
        {
            'video_id': video_id,
            'frame_id': frame_id,
            'score': score
        },
        ...
    ]    
    Sắp xếp theo score giảm dần.       
    '''
    top_k_milvus_search = normalize_top_k(search_milvus(query, video_id, start_end), higher_is_better=True)
    top_k_ocr_search = normalize_top_k(search_ocr(query, video_id, start_end), higher_is_better=True)
    top_k_asr_search = normalize_top_k(search_asr(query, video_id, start_end), higher_is_better=True)

    merged_results = merge_and_mean_scores(
        top_k_milvus_search,
        top_k_ocr_search,
        top_k_asr_search,
        missing_score=0.0
    )
    
    return merged_results

def temporal_search(query_1, query_2, frames):
    """
    Hàm này đầu tiên sẽ thực hiện truy vấn q1 trước, rồi sẽ tiếp tục rerank dựa
    trên kết quả truy vấn q2 (truy vấn này được thực hiện dựa trên tập con của 
    các frame được trả về từ truy vấn q1).
    
    Args:
        query_1 (list): The first query string to search for.
        query_2 (list): The second query string to search for.
        frames (list): A list of frames paths to consider for the search query 1.
    Returns:
        list: A list of paths results matching the query.
    """
    threshold = 2000
    
    results = []
    # If no frames are provided, perform a search using the first query if available, otherwise use the second query.
    if frames is None:
        query = query_1 if query_1 is not None else query_2
        return search_query(query)

    frames_q1_scores = defaultdict(list)
    
    # Trích xuất các trường thông tin từ các frames trả về sau truy vấn q1
    # Theo dạng: {video_id: [{'frame_id': ..., 'score': ...}, ...]}
    for video_id, frame_id, score in zip(
        frames['video_id'],
        frames['frame_id'],
        frames['scores']
    ):
        frames_q1_scores[video_id].append({
            'frame_id': frame_id,
            'score': score
        })
    
    
    frames_q2_scores = defaultdict(list)
    
    # Tra soát qua các frames trả về từ truy vấn q1, và thực hiện truy vấn q2 dựa trên các frame_id của q1
    for matches_q1 in frames:  
        video_id = matches_q1['video_id']
        
        frame_id = matches_q1['frame_id']
        
        start_end = zip(frame_id, frame_id + threshold)
        
        # Thực hiện truy vấn q2 dựa trên video_id và shots được xác định từ các frame_id của q1
        match_frames_q2 = search_query(query_2, video_id, start_end)
        
        # Trả về frame có score cao nhất từ kết quả truy vấn q2 cho mỗi video_id, frame_id
        frame_q2 = match_frames_q2[0] 
        
        # Lưu trữ kết quả truy vấn q2 theo dạng: {(video_id, frame_id): [{'frame_id': ..., 'score': ...}, ...]}
        frames_q2_scores[video_id, frame_id].append({
            'frame_id': frame_q2['frame_id'],
            'score': frame_q2['scores']
        })
    
    # Rà soát qua từng video_id, frame_id trong kết quả truy vấn q1
    for video_id in frames_q1_scores:
        # Rà soát qua từng frame_id trong kho video_id trả về, và lấy kết quả truy vấn q2 tương ứng dựa trên video_id và frame_id
        for match_q1 in frames_q1_scores[video_id]:
            match_q2 = frames_q2_scores[video_id, match_q1['frame_id']]
            results.append({
                'video_id': video_id,
                'frame_id_q1': match_q1['frame_id'],
                # 'score_q1': match_q1['score'],
                # 'frame_id_q2': match_q2['frame_id'],
                # 'score_q2': match_q2['score'],
                'score' : np.sqrt(match_q1['score'] * match_q2['score'])
            })
    return results
        
        
        
        

    
