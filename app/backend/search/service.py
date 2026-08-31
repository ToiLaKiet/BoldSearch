from __future__ import annotations

import base64
import binascii
import csv
from io import BytesIO
import json
import math
from pathlib import Path
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from app_config import AppConfig
from search import schema
from search.object_index import FrameObjectIndex, object_doc_for_row


_argos_installed: set[tuple[str, str]] = set()  # cache of (from_code, to_code) pairs


def _ensure_argos_package(from_code: str, to_code: str) -> None:
    """Download & install the argostranslate language package if not already available."""
    pair = (from_code, to_code)
    if pair in _argos_installed:
        return

    import argostranslate.package

    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()
    pkg = next(
        (p for p in available if p.from_code == from_code and p.to_code == to_code),
        None,
    )
    if pkg is None:
        raise RuntimeError(f"No argostranslate package found for {from_code} → {to_code}")

    argostranslate.package.install_from_path(pkg.download())
    _argos_installed.add(pair)


import os
from functools import lru_cache

import requests

AZURE_TRANSLATOR_URL = (
    "https://api.cognitive.microsofttranslator.com/translate"
)


@lru_cache(maxsize=10_000)
def translate_to_english(text: str) -> str:
    text = text.strip()

    if not text:
        return text

    try:
        response = requests.post(
            AZURE_TRANSLATOR_URL,
            params={
                "api-version": "3.0",
                "from": "vi",
                "to": "en",
            },
            headers={
                "Ocp-Apim-Subscription-Key":
                    os.environ["AZURE_TRANSLATOR_KEY"],
                "Ocp-Apim-Subscription-Region":
                    os.environ["AZURE_TRANSLATOR_REGION"],
                "Content-Type": "application/json",
            },
            json=[{"text": text}],
            timeout=15,
        )

        response.raise_for_status()

        translated = response.json()[0]["translations"][0]["text"]
        translated = translated.strip()

        print(f"[translate] '{text}' -> '{translated}'")
        return translated

    except Exception as exc:
        print(f"[translate] failed: {exc}")
        return text


def temporal_search(
    body: schema.QueryRequest,
    client: Any,
    config: AppConfig,
    embedding_encoder: Any,
    queries: List[str],
) -> Dict[str, Any]:
    top_k = body.topK or config.SEARCH_TOP_K
    frame_paths = _frame_paths(body.frames_path)
    current_context: List[Dict[str, Any]] = []
    raw_response: Optional[Dict[str, Any]] = None

    if frame_paths is None:
        raw_response = _search_text_stage(
            query_text=queries[0],
            embedding_encoder=embedding_encoder,
            client=client,
            config=config,
            top_k=top_k,
            expr=None,
        )
        used_query_count = 1
        current_context = _context_from_rows(_frame_rows(raw_response))
    else:
        used_query_count = _used_query_count(queries, body.used_queries)
        current_context = _normalize_frame_contexts(
            frame_paths=frame_paths,
            frames_context=body.frames_context,
        )

    if used_query_count >= len(queries):
        return raw_response or _raw_response_from_context(current_context, top_k)

    for index in range(used_query_count, len(queries)):
        expr = _build_context_filter(
            current_context,
            total_queries=len(queries),
            used_query_count=index,
        )
        if not expr:
            return {"code": 0, "data": [], "topks": [top_k]}

        raw_response = _search_text_stage(
            query_text=queries[index],
            embedding_encoder=embedding_encoder,
            client=client,
            config=config,
            top_k=top_k,
            expr=expr,
        )
        current_context = _context_from_rows(_frame_rows(raw_response))

    return raw_response or {"code": 0, "data": [], "topks": [top_k]}


def _search_text_stage(
    query_text: str,
    embedding_encoder: Any,
    client: Any,
    config: AppConfig,
    top_k: int,
    expr: Optional[str],
) -> Dict[str, Any]:
    query_embedding = _encode_text_query(embedding_encoder, query_text)
    results = _hybrid_search(
        config=config,
        client=client,
        query_text=query_text,
        query_embedding=query_embedding,
        top_k=top_k,
        expr=expr,
    )
    return results


def run_text_query(
    body: schema.QueryRequest,
    config: AppConfig,
    milvus_client: Any,
    object_index: Optional[FrameObjectIndex],
    embedding_encoder: Any,
) -> schema.SearchResponse:

    object_queries = _object_queries(body.objectQueries, body.objects)
    print('objects:', object_queries)
    text_queries = _text_queries(body)
    response_query = _query_text(body, object_queries)

    if len(text_queries) > 1:
        raw_response = temporal_search(
            body=body,
            config=config,
            client=milvus_client,
            embedding_encoder=embedding_encoder,
            queries=text_queries,
        )
    else:
        search_text = text_queries[0] if text_queries else response_query
        raw_response = _search_text_stage(
            query_text=search_text,
            embedding_encoder=embedding_encoder,
            client=milvus_client,
            config=config,
            top_k=body.topK or config.SEARCH_TOP_K,
            expr=None,
        )

    return _response_from_milvus(
        raw_response=raw_response,
        task=body.task,
        query=response_query,
        object_queries=object_queries,
        config=config,
        object_index=object_index,
    )


def run_visual_query(
    body: schema.VisualQueryRequest,
    config: AppConfig,
    milvus_client: Any,
    object_index: Optional[FrameObjectIndex],
    embedding_encoder: Any,
) -> schema.SearchResponse:
    
    embedding = _visual_embedding(body, embedding_encoder)


    raw_response = _hybrid_search(
        config=config,
        client=milvus_client,
        query_text=None,
        query_embedding=embedding,
        top_k=body.topK or config.SEARCH_TOP_K,
    )
    query_label = str(body.imageCue.get("name", "")).strip() or "visual_query"

    return _response_from_milvus(
        raw_response=raw_response,
        task=body.task,
        query=query_label,
        object_queries=[],
        config=config,
        object_index=object_index,
    )


def _encode_text_query(embedding_encoder: Any, query_text: str) -> List[float]:
    if not query_text:
        raise ValueError("query text or object query is required")
    if embedding_encoder is None:
        raise RuntimeError("embedding encoder is not loaded")

    english_text = translate_to_english(query_text)
    embeddings = embedding_encoder.encode_texts([english_text])
    return _embedding_to_list(embeddings[0])


def _visual_embedding(
    body: schema.VisualQueryRequest,
    embedding_encoder: Any,
) -> List[float]:
    existing = body.imageEmbedding or body.imageCue.get("embedding") or body.imageCue.get("vector")
    if existing:
        return _embedding_to_list(existing)

    if embedding_encoder is None:
        raise RuntimeError("embedding encoder is not loaded")

    image = _image_from_cue(body.imageCue)
    
    embeddings = embedding_encoder.encode_images([image])

    return _embedding_to_list(embeddings[0])


def _image_from_cue(image_cue: Dict[str, Any]) -> Any:
    encoded = (
        image_cue.get("dataUrl")
        or image_cue.get("data_url")
        or image_cue.get("base64")
        or image_cue.get("content")
    )
    if not encoded:
        raise ValueError("visual_query requires imageCue.dataUrl/base64 or imageEmbedding")

    if isinstance(encoded, str) and "," in encoded and encoded.lower().startswith("data:"):
        encoded = encoded.split(",", 1)[1]

    try:
        raw = base64.b64decode(encoded, validate=True)
    except (TypeError, ValueError, binascii.Error) as exc:
        raise ValueError("imageCue contains invalid base64 image data") from exc

    try:
        from PIL import Image, ImageOps
        img = Image.open(BytesIO(raw))
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")
    except Exception as exc:
        raise ValueError("imageCue image data could not be decoded") from exc


# Dùng để chuyển đổi embedding sang dạng list float, đảm bảo rằng embedding là một sequence số học hợp lệ.
def _embedding_to_list(value: Any) -> List[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("embedding must be a numeric sequence")

    return [float(item) for item in value]


def submit_frame(
    body: schema.SubmitRequest,
    config: AppConfig,
) -> schema.SubmitResponse:
    frame_id = body.frame_id if body.frame_id is not None else body.frameId
    shot_id = body.shot_id if body.shot_id is not None else body.shotId
    video_id = body.video_id or body.videoId
    result_id = body.id or _fallback_id(str(video_id or ""), shot_id, frame_id)

    if result_id == "frame":
        raise ValueError("submit requires id, frameId/frame_id, shotId/shot_id, or videoId/video_id")

    return schema.SubmitResponse(
        status="accepted",
        system=config.SYSTEM_NAME,
        submission={
            "id": result_id,
            "frame_id": frame_id,
            "shot_id": shot_id,
            "video_id": video_id,
            "task": (body.task or "KIS").upper(),
        },
    )


def submit_vqa(
    body: schema.VQASubmitRequest,
    config: AppConfig,
) -> schema.VQASubmitResponse:
    """Submit a VQA answer: video_id + frame_id + free-text answer."""
    if not body.video_id:
        raise ValueError("submit_vqa requires video_id")
    if body.frame_id is None or body.frame_id == "":
        raise ValueError("submit_vqa requires frame_id")
    if not body.answer and body.answer != "0":
        raise ValueError("submit_vqa requires a non-empty answer")

    return schema.VQASubmitResponse(
        status="accepted",
        system=config.SYSTEM_NAME,
        submission={
            "video_id": body.video_id,
            "frame_id": body.frame_id,
            "answer": body.answer,
            "task": "VQA",
        },
    )


def submit_trake(
    body: schema.TrakeSubmitRequest,
    config: AppConfig,
) -> schema.TrakeSubmitResponse:
    """Submit a TRAKE answer: video_id + list of frame_ids."""
    if not body.video_id:
        raise ValueError("submit_trake requires video_id")
    if not body.frame_ids:
        raise ValueError("submit_trake requires at least one frame_id")

    return schema.TrakeSubmitResponse(
        status="accepted",
        system=config.SYSTEM_NAME,
        submission={
            "video_id": body.video_id,
            "frame_ids": list(body.frame_ids),
            "task": "TRAKE",
        },
    )


def _hybrid_search(
    config: AppConfig,
    client: Any,
    query_text: Optional[str],
    query_embedding: Optional[Sequence[float]],
    top_k: int,
    expr: Optional[str] = None,
) -> Dict[str, Any]:
    if client is None:
        raise RuntimeError("ZILLIZ_URI is not configured")

    try:
        from pymilvus import AnnSearchRequest, WeightedRanker
    except ImportError as exc:
        raise RuntimeError("pymilvus is required for Zilliz hybrid search") from exc

    if query_embedding is None:
        raise ValueError("query_embedding is required")

    # Tất cả query (text và visual) đều search trong visual_embedding.
    # BEiT-3 retrieval maps text and image embeddings into the same space.
    # caption_embedding hiện tại chứa vector 0 (chưa index) → tạm thời không dùng.
    visual_req = AnnSearchRequest(
        data=[query_embedding],
        anns_field="visual_embedding",
        param={"metric_type": "COSINE", "params": {}},
        limit=top_k,
        expr=expr,
    )

    print(f"Search visual_embedding: text={query_text is not None}, visual={query_embedding is not None}")

    raw = client.hybrid_search(
        collection_name=config.MILVUS_COLLECTION,
        reqs=[visual_req],
        ranker=WeightedRanker(1.0),
        limit=top_k,
        output_fields=_csv_list(config.MILVUS_OUTPUT_FIELDS),
    )

    if isinstance(raw, dict):
        return raw

    return {"code": 0, "data": _flatten_milvus_hits(raw), "topks": [top_k]}


def _response_from_milvus(
    raw_response: Dict[str, Any],
    task: str,
    query: str,
    object_queries: List[Dict[str, Any]],
    config: AppConfig,
    object_index: Optional[FrameObjectIndex],
) -> schema.SearchResponse:
    
    frames = _frame_rows(raw_response)
    
    # If there are no object queries, we can directly return the results without further processing.
    if not object_queries:
        results = [
            _frame_result(
                row=row,
                score=_score(row),
                object_summary=Counter(),
                object_matches=[],
                object_count_delta=None,
                object_doc={},
                config=config,
            )
            for row in frames
        ]
        return schema.SearchResponse(
            system=config.SYSTEM_NAME,
            task=(task or "KIS").upper(),
            query=query,
            count=len(results),
            cacheHitRatio=raw_response.get("cache_hit_ratio"),
            cost=raw_response.get("cost"),
            topks=list(raw_response.get("topks") or []),
            results=results,
        )
        
    if object_queries and object_index is None:
        raise RuntimeError("object_index is required for object queries")

    ranked_results: List[Tuple[float, float, int, schema.FrameResult]] = []

    for i, row in enumerate(frames):
        object_doc = object_doc_for_row(row, object_index)
        object_summary = _object_summary(object_doc)
        object_matches = _match_objects(object_queries, object_summary)
        if object_queries and not _matches_all_object_queries(object_queries, object_matches):
            continue

        score = _score(row)

        object_count_delta = _object_count_delta(object_queries, object_summary)

        ranked_results.append(
            (
                object_count_delta,
                score,
                _frame_result(
                    row=row,
                    score=score,
                    object_summary=object_summary,
                    object_matches=object_matches,
                    object_count_delta=object_count_delta if object_queries else None,
                    object_doc=object_doc,
                    config=config,
                ),
            )
        )

    if object_queries:
        ranked_results.sort(key=lambda item: (-item[1],item[0])) # item represents a tuple of (object_count_delta, -score, frame_result). The sort key is a tuple of (item[0], item[1]), which means the results will be sorted first by object_count_delta in ascending order, and then by -score in ascending order (which effectively sorts by score in descending order). This ensures that frames with the least object count delta and highest score are prioritized.

    results = [item[2] for item in ranked_results]

    return schema.SearchResponse(
        system=config.SYSTEM_NAME,
        task=(task or "KIS").upper(),
        query=query,
        count=len(results),
        cacheHitRatio=raw_response.get("cache_hit_ratio"),
        cost=raw_response.get("cost"),
        topks=list(raw_response.get("topks") or []),
        results=results,
    )


def _query_text(
    body: schema.QueryRequest,
    object_queries: List[Dict[str, Any]],
) -> str:
    queries = [item.strip() for item in [body.query, *body.queries] if item.strip()]
    query_text = " ".join(dict.fromkeys(queries)).strip()
    if query_text:
        return query_text

    object_text = " ".join(str(item["count"]) + " " + item["query"] for item in object_queries).strip()
    if object_text:
        return object_text

    raise ValueError("query text or object query is required")


def _text_queries(body: schema.QueryRequest) -> List[str]:
    candidates = body.queries if body.queries else [body.query]
    return [item.strip() for item in candidates if item.strip()]


def _frame_paths(frames_path: Optional[List[str]]) -> Optional[List[str]]:
    if not frames_path:
        return None
    paths = [str(path).strip() for path in frames_path if str(path).strip()]
    return paths or None


def _used_query_count(queries: List[str], used_queries: List[str]) -> int:
    count = 0
    for used_query in [item.strip() for item in used_queries if item.strip()]:
        if count >= len(queries) or queries[count] != used_query:
            break
        count += 1
    return count


def _normalize_frame_contexts(
    frame_paths: List[str],
    frames_context: List[schema.FrameContext],
) -> List[Dict[str, Any]]:
    context_candidates = [
        _context_from_frame_context(item)
        for item in frames_context
    ]
    by_path = {
        item["path"]: item
        for item in context_candidates
        if item.get("path")
    }
    by_video_frame = {
        (str(item.get("video_id")), str(item.get("frame_id"))): item
        for item in context_candidates
        if item.get("video_id") and item.get("frame_id") is not None
    }

    normalized: List[Dict[str, Any]] = []
    seen = set()
    for path in frame_paths:
        parsed = _parse_frame_path(path)
        metadata = by_path.get(path) or by_video_frame.get(
            (str(parsed.get("video_id")), str(parsed.get("frame_id")))
        ) or {}
        item = {
            "path": path,
            "video_id": metadata.get("video_id") or parsed.get("video_id"),
            "frame_id": metadata.get("frame_id") if metadata.get("frame_id") is not None else parsed.get("frame_id"),
            "shot_id": metadata.get("shot_id") if metadata.get("shot_id") is not None else parsed.get("shot_id"),
            "score": metadata.get("score"),
        }
        key = (item.get("path"), item.get("video_id"), item.get("frame_id"), item.get("shot_id"))
        if key in seen:
            continue
        seen.add(key)
        normalized.append(item)

    return normalized


def _context_from_frame_context(item: schema.FrameContext) -> Dict[str, Any]:
    parsed = _parse_frame_path(item.path)
    return {
        "path": item.path or parsed.get("path") or "",
        "video_id": item.video_id or parsed.get("video_id"),
        "frame_id": item.frame_id if item.frame_id is not None else parsed.get("frame_id"),
        "shot_id": item.shot_id if item.shot_id is not None else parsed.get("shot_id"),
        "score": item.score,
    }


def _context_from_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "path": row.get("frame_path") or "",
            "video_id": row.get("video_id"),
            "frame_id": row.get("frame_id"),
            "shot_id": row.get("shot_id"),
            "score": _score(row),
        }
        for row in rows
    ]


def _raw_response_from_context(
    context: List[Dict[str, Any]],
    top_k: int,
) -> Dict[str, Any]:
    data = []
    for item in context[:top_k]:
        row = {
            "video_id": item.get("video_id"),
            "frame_id": item.get("frame_id"),
            "shot_id": item.get("shot_id"),
            "thumbnail": item.get("path") or "",
        }
        if item.get("score") is not None:
            row["distance"] = item["score"]
        data.append(row)
    return {"code": 0, "data": data, "topks": [top_k]}


def _build_context_filter(
    context: List[Dict[str, Any]],
    total_queries: int,
    used_query_count: int,
) -> str:
    remaining_query_count = total_queries - used_query_count
    if remaining_query_count <= 0:
        return ""

    shots_by_video: Dict[str, set[int]] = {}
    for item in context:
        video_id = str(item.get("video_id") or "").strip()
        shot_id = _int_or_none(item.get("shot_id"))
        if not video_id or shot_id is None:
            continue

        shot_ids = shots_by_video.setdefault(video_id, set()) # setdefault() is a method in Python that returns the value of a specified key in a dictionary. If the key does not exist, it inserts the key with a specified default value and returns that value. In this case, it retrieves the set of shot IDs for the given video_id from the shots_by_video dictionary. If the video_id does not exist in the dictionary, it creates a new entry with an empty set as the default value and returns that empty set. This allows for easy accumulation of shot IDs associated with each video ID.
        for offset in range(remaining_query_count + 3):
            shot_ids.add(shot_id + offset)

    clauses = []
    for video_id in sorted(shots_by_video):
        shot_ids = sorted(shots_by_video[video_id])
        if not shot_ids:
            continue
        clauses.append(
            f'(video_id == "{_escape_milvus_string(video_id)}" and shot_id in [{", ".join(str(item) for item in shot_ids)}])'
        )

    return " or ".join(clauses)


def _parse_frame_path(path: str) -> Dict[str, Any]:
    value = str(path or "").strip()
    if not value:
        return {"path": ""}

    parsed = urlparse(value)
    path_value = parsed.path or value.split("?", 1)[0]
    parts = [part for part in path_value.split("/") if part]
    frame_name = parts[-1] if parts else ""
    frame_id = frame_name.rsplit(".", 1)[0] if frame_name else None
    video_id = None
    shot_id = None

    if len(parts) >= 4 and parts[-4] == "keyframes":
        video_id = parts[-3]
        shot_id = parts[-2]
    elif len(parts) >= 2:
        video_id = parts[-2]

    return {
        "path": value,
        "video_id": video_id,
        "frame_id": frame_id,
        "shot_id": shot_id,
    }


def _frame_rows(raw_response: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = raw_response.get("data") or []
    return [row for row in data if isinstance(row, dict)]


def _flatten_milvus_hits(raw: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    hit_groups = raw if isinstance(raw, list) else [raw]
    for group in hit_groups:
        hits = group if isinstance(group, list) else [group]
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            entity = hit.get("entity") or {}
            row = {**entity, **{key: value for key, value in hit.items() if key != "entity"}}
            rows.append(row)
    return rows


def _object_summary(object_doc: Dict[str, Any]) -> Counter:
    counter: Counter = Counter()
    candidates = (
        object_doc.get("objects")
        or object_doc.get("detections")
        or object_doc.get("labels")
        or []
    )

    if isinstance(candidates, dict):
        for label, count in candidates.items():
            counter[_normalize_label(label)] += _int_count(count)
        return counter

    if not isinstance(candidates, list):
        return counter

    for item in candidates:
        if isinstance(item, str):
            counter[_normalize_label(item)] += 1
            continue

        if not isinstance(item, dict):
            continue

        label = (
            item.get("label")
            or item.get("name")
            or item.get("object")
            or item.get("class")
            or item.get("query")
        )

        counter[_normalize_label(label)] += _int_count(item.get("count", item.get("quantity", 1)))

    return counter


def _match_objects(
    object_queries: List[Dict[str, Any]],
    object_summary: Counter,
) -> List[schema.ObjectMatch]:
    matches: List[schema.ObjectMatch] = []
    for item in object_queries:
        query = item["query"]
        requested = item["count"]
        matched = _matched_object_count(query, object_summary)
        if matched:
            matches.append(
                schema.ObjectMatch(
                    query=query,
                    requested=requested,
                    matched=matched,
                )
            )
    return matches


def _matches_all_object_queries(
    object_queries: List[Dict[str, Any]],
    matches: List[schema.ObjectMatch],
) -> bool:
    matched_by_query = {item.query: item.matched for item in matches}
    return all(
        matched_by_query.get(item["query"], 0) > 0
        for item in object_queries
    )


def _object_count_delta(
    object_queries: List[Dict[str, Any]],
    object_summary: Counter,
) -> float:
    if not object_queries:
        return 0.0

    return sum(
        abs(_matched_object_count(item["query"], object_summary) - item["count"])
        / max(item["count"], 1)
        for item in object_queries
    ) # for e.g : if the requested count is 3 and the matched count is 1, the delta would be abs(1 - 3) / max(3, 1) = 2 / 3 = 0.6667. This means that the matched count is 66.67% less than the requested count. The sum of these deltas across all object queries gives an overall measure of how well the matched counts align with the requested counts.


def _matched_object_count(query: str, object_summary: Counter) -> int:
    query_tokens = set(_tokens(query))
    total = 0
    for label, count in object_summary.items():
        label_tokens = set(_tokens(label))
        if query == label or query_tokens.issubset(label_tokens): #issubset() is a method in Python that checks if all elements of one set are present in another set. In this case, it checks if all tokens from the query are present in the label tokens. If they are, it means the label matches the query, and the count for that label is added to the total matched count.
            total += count
    return total


def _frame_result(
    row: Dict[str, Any],
    score: float,
    object_summary: Counter,
    object_matches: List[schema.ObjectMatch],
    object_count_delta: Optional[float],
    object_doc: Dict[str, Any],
    config: AppConfig,
) -> schema.FrameResult:
    frame_id = row.get("frame_id")
    shot_id = row.get("shot_id")
    video_id = str(row.get("video_id") or "")
    asr_text = str(row.get("asr_text") or "")
    ocr_text = str(row.get("ocr_text") or "")
    title = row.get("title") or _title(video_id, frame_id)
    description = row.get("description") or " ".join(text for text in [asr_text, ocr_text] if text)
    thumbnail = (
        row.get("thumbnail")
        or row.get("frames_path")
        or row.get("frame_path")
        or row.get("image_url")
        or row.get("frame_url")
        or _thumbnail(config, row)
    )
    result_id = row.get("id") or _fallback_id(video_id, shot_id, frame_id)

    result = schema.FrameResult(
        id=result_id,
        frame_id=frame_id,
        shot_id=shot_id,
        video_id=video_id or None,
        distance=_float_or_none(row.get("distance")),
        score=score,

        reasons=_reasons(row, object_matches, object_count_delta),
        videoId=video_id,
        shotId=str(shot_id or ""),
        title=str(title),
        description=description,
        thumbnail=str(thumbnail or ""),
        videoUrl=str(row.get("video_url") or ""),
        start=str(row.get("start") or ""),
        end=str(row.get("end") or ""),
        duration=_float_or_none(row.get("duration")) or 0,
        raw=_response_raw(row, config),
    )

    if config.INCLUDE_EMBEDDING_IN_RESPONSE and isinstance(row.get("embedding"), list):
        result.embedding = row["embedding"]

    return result


def _reasons(
    row: Dict[str, Any],
    object_matches: List[schema.ObjectMatch],
    object_count_delta: Optional[float],
) -> List[str]:
    reasons = []
    distance = _float_or_none(row.get("distance"))
    if distance is not None:
        reasons.append(f"Milvus distance: {distance:.4f}")
    for match in object_matches:
        reasons.append(f"Object: {match.query} x{match.matched}/{match.requested}")
    if object_count_delta is not None:
        reasons.append(f"Object count delta: {object_count_delta:.2f}")
    if not reasons:
        reasons.append("Milvus hybrid search")
    return reasons



def _object_queries(
    object_queries: List[schema.ObjectQuery],
    objects: List[str],
) -> List[Dict[str, Any]]:
    normalized = [
        {"query": item.query.strip().lower(), "count": max(item.count, 1)}
        for item in object_queries
        if item.query.strip()
    ]
    if normalized:
        return normalized
    return [{"query": item.strip().lower(), "count": 1} for item in objects if item.strip()]


def _score(row: Dict[str, Any]) -> float:
    distance = _float_or_none(row.get("distance"))
    if distance is None:
        return 1.0
    return distance


def _response_raw(row: Dict[str, Any], config: AppConfig) -> Dict[str, Any]:
    if config.INCLUDE_EMBEDDING_IN_RESPONSE:
        return dict(row)
    return {key: value for key, value in row.items() if key != "embedding"}


def _thumbnail(config: AppConfig, row: Dict[str, Any]) -> str:
    template = config.FRAME_IMAGE_URL_TEMPLATE
    if not template:
        return ""
    video_id = str(row.get("video_id") or "")
    keyframe_number = _nearest_keyframe_number(
        config.KEYFRAME_MAP_DIR,
        video_id,
        row.get("frame_id"),
    )
    if "{keyframe_number}" in template and keyframe_number is None:
        return ""
    return template.format(
        frame_id=row.get("frame_id") or "",
        shot_id=row.get("shot_id") or "",
        video_id=video_id,
        keyframe_number=f"{keyframe_number:03d}" if keyframe_number is not None else "",
    )


def _nearest_keyframe_number(
    keyframe_map_dir: Path | str | None,
    video_id: str,
    frame_id: Any,
) -> int | None:
    try:
        target_frame = int(frame_id)
    except (TypeError, ValueError):
        return None
    if not keyframe_map_dir or not video_id:
        return None

    map_path = Path(keyframe_map_dir) / f"{video_id}.csv"
    try:
        with map_path.open(encoding="utf-8-sig", newline="") as handle:
            candidates = [
                (int(row["n"]), int(row["frame_idx"]))
                for row in csv.DictReader(handle)
                if row.get("n") and row.get("frame_idx")
            ]
    except (OSError, ValueError):
        return None
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: abs(candidate[1] - target_frame))[0]


def _title(video_id: str, frame_id: Any) -> str:
    if video_id and frame_id is not None:
        return f"{video_id} frame {frame_id}"
    if frame_id is not None:
        return f"Frame {frame_id}"
    return "Retrieved frame"


def _fallback_id(video_id: str, shot_id: Any, frame_id: Any) -> str:
    parts = [str(item) for item in [video_id, shot_id, frame_id] if item is not None and item != ""]
    return ":".join(parts) or "frame"


def _json_dict(value: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _ranker_weights(config: AppConfig, count: int) -> List[float]:
    weights = [
        _float_or_none(item) or 0.0
        for item in config.MILVUS_RANKER_WEIGHTS.split(",")
        if item.strip()
    ]
    if not weights:
        weights = [1.0]
    while len(weights) < count:
        weights.append(weights[-1])
    return [1,0]


def _csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _normalize_label(value: Any) -> str:
    return str(value).strip().lower()


def _tokens(value: str) -> List[str]:
    return [token for token in re.split(r"[^\w]+", value.lower()) if token] # split() is a technique of splitting a string into a list of substrings based on a specified delimiter. In this case, the delimiter is any non-word character (i.e., anything that is not a letter, digit, or underscore). The resulting tokens are then converted to lowercase and filtered to remove any empty strings.


def _int_count(value: Any) -> int:
    try:
        return max(int(value), 1)
    except (TypeError, ValueError):
        return 1


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _escape_milvus_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
