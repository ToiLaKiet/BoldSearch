from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
import re
from typing import Iterable


def _normalize_text(text: str | None) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def text_similarity(text_a: str | None, text_b: str | None) -> float:
    """Tính độ giống OCR trong [0, 1]. Dùng RapidFuzz nếu có."""
    a = _normalize_text(text_a)
    b = _normalize_text(text_b)
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0

    try:
        from rapidfuzz.fuzz import ratio

        return float(ratio(a, b)) / 100.0
    except ImportError:
        return float(SequenceMatcher(None, a, b).ratio())


def _normalize_objects(objects: Iterable[object] | None) -> Counter[str]:
    normalized: Counter[str] = Counter()
    for item in objects or []:
        # Cho phép đầu vào là string hoặc dict từ detector.
        if isinstance(item, dict):
            value = item.get("label", item.get("class", item.get("name", "")))
        else:
            value = item
        label = str(value).strip().lower()
        if label:
            normalized[label] += 1
    return normalized


def multiset_jaccard(
    objects_a: Iterable[object] | None,
    objects_b: Iterable[object] | None,
) -> float:
    """Jaccard có xét số lượng object, không làm mất count như set Jaccard."""
    a = _normalize_objects(objects_a)
    b = _normalize_objects(objects_b)
    labels = set(a) | set(b)
    if not labels:
        return 1.0

    intersection = sum(min(a[label], b[label]) for label in labels)
    union = sum(max(a[label], b[label]) for label in labels)
    return float(intersection / union) if union else 1.0


def verify_duplicate_guard(
    ocr_a: str | None,
    ocr_b: str | None,
    objects_a: list[object] | None,
    objects_b: list[object] | None,
    ocr_similarity_thresh: float = 0.85,
    jaccard_object_thresh: float = 0.80,
    min_ocr_length: int = 4,
) -> bool:
    """Bảo vệ thông tin OCR và object trước khi gộp hai frame.

    ``True`` nghĩa là hai frame không có mâu thuẫn rõ ràng từ OCR/object và có
    thể được gộp nếu visual embedding cũng đủ gần.
    """
    if not 0 <= ocr_similarity_thresh <= 1:
        raise ValueError("ocr_similarity_thresh must be in [0, 1]")
    if not 0 <= jaccard_object_thresh <= 1:
        raise ValueError("jaccard_object_thresh must be in [0, 1]")

    text_a = _normalize_text(ocr_a)
    text_b = _normalize_text(ocr_b)

    if text_a != text_b:
        if text_a and text_b:
            if text_similarity(text_a, text_b) < ocr_similarity_thresh:
                return False
        elif max(len(text_a), len(text_b)) >= min_ocr_length:
            # Một frame có text đủ dài, frame kia không có: giữ riêng để tránh
            # làm mất subtitle, scoreboard, giá tiền hoặc bảng hiệu.
            return False

    if multiset_jaccard(objects_a, objects_b) < jaccard_object_thresh:
        return False

    return True
