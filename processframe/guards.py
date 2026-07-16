def verify_duplicate_guard(
    ocr_a: str | None,
    ocr_b: str | None,
    objects_a: list[str] | None,
    objects_b: list[str] | None,
    ocr_similarity_thresh: float = 0.85,
    jaccard_object_thresh: float = 0.80,
) -> bool:
    """
    Lớp bảo vệ đa mô-đun (OCR & Object Guards).
    Ngăn chặn việc gộp nhóm 2 frame mặc dù có embedding tương đồng cao nhưng nội dung chi tiết
    như văn bản (OCR) hoặc các đối tượng thị giác (Objects) lại có sự khác biệt quan trọng.
    
    Parameters:
        ocr_a, ocr_b: Văn bản OCR của hai frame.
        objects_a, objects_b: Danh sách các nhãn vật thể được phát hiện trong hai frame.
        ocr_similarity_thresh: Ngưỡng tỷ lệ tương đồng tối thiểu cho văn bản OCR.
        jaccard_object_thresh: Ngưỡng chỉ số Jaccard tối thiểu cho các đối tượng.
        
    Returns:
        bool: True nếu hai frame đủ điều kiện để gộp (không vi phạm guard), False nếu cần tách.
    """
    # 1. OCR Guard
    txt_a = (ocr_a or "").strip().lower()
    txt_b = (ocr_b or "").strip().lower()
    
    # Nếu một bên có text rõ ràng còn bên kia rỗng, hoặc text khác hẳn nhau
    if txt_a != txt_b:
        # Nếu cả hai đều có nội dung văn bản, tính khoảng cách tương đồng
        if txt_a and txt_b:
            # Sử dụng tỷ lệ Levenshtein khoảng cách đơn giản
            from difflib import SequenceMatcher
            ratio = SequenceMatcher(None, txt_a, txt_b).ratio()
            if ratio < ocr_similarity_thresh:
                return False
        else:
            # Một frame có text, frame kia hoàn toàn không có -> Trạng thái màn hình đã thay đổi
            # Chỉ chặn nếu độ dài chuỗi văn bản đủ lớn (ví dụ > 3 ký tự) để loại bỏ nhiễu OCR
            if len(txt_a) > 3 or len(txt_b) > 3:
                return False

    # 2. Object Guard
    objs_a = set([str(x).strip().lower() for x in (objects_a or []) if str(x).strip()])
    objs_b = set([str(x).strip().lower() for x in (objects_b or []) if str(x).strip()])
    
    if objs_a or objs_b:
        intersection = objs_a.intersection(objs_b)
        union = objs_a.union(objs_b)
        jaccard = len(intersection) / len(union) if union else 1.0
        if jaccard < jaccard_object_thresh:
            return False

    return True
