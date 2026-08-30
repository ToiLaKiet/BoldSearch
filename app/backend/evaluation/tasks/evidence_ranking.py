from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from evaluation.metrics.ranking import (
    first_relevant_rank,
    ndcg_at_k,
    normalize_k_values,
    recall_at_k,
    reciprocal_rank,
)


FrameIdentity = tuple[str, str]


def evaluate_evidence_rankings(
    task_cards: Sequence[Mapping[str, Any]],
    rankings: Sequence[Mapping[str, Any]],
    k_values: Sequence[int],
) -> dict[str, Any]:
    """Evaluate task evidence frames from a retrieval ranking export.

    This adapter owns JSONL task-card conventions. Metric formulas remain in
    ``evaluation.metrics`` so VQA answer and TRAKE temporal evaluators can use
    their own task contracts without inheriting frame-ranking behavior.
    """
    if not task_cards:
        raise ValueError("task cards must not be empty")
    if not rankings:
        raise ValueError("rankings must not be empty")
    normalized_k_values = normalize_k_values(k_values)
    task_cards_by_task_id = _task_cards_by_task_id(task_cards)
    rankings_by_task_id = _rankings_by_task_id(rankings)
    if task_cards_by_task_id.keys() != rankings_by_task_id.keys():
        raise ValueError("task card and ranking task_ids must match")
    per_task = [
        _evaluate_task(task_card, rankings_by_task_id[task_id], normalized_k_values)
        for task_id, task_card in task_cards_by_task_id.items()
    ]
    return {
        "overall": _aggregate(per_task, normalized_k_values),
        "by_task_type": _aggregate_by_task_type(per_task, normalized_k_values),
        "per_task": per_task,
    }


def _evaluate_task(
    task_card: Mapping[str, Any],
    ranking: Mapping[str, Any],
    k_values: Sequence[int],
) -> dict[str, Any]:
    relevant_frames = _relevant_frames(task_card)
    candidates = _unique_candidates(ranking["results"])
    relevance_by_frame = {
        _frame_identity(frame): frame.get("relevance", 1) for frame in relevant_frames
    }
    ranked_relevances = [relevance_by_frame.get(candidate, 0) for candidate in candidates]
    largest_k = k_values[-1]

    result = {
        "task_id": _task_id(task_card),
        "task_type": _task_type(task_card),
        "first_relevant_rank": first_relevant_rank(ranked_relevances),
        "reciprocal_rank": reciprocal_rank(ranked_relevances),
        f"ndcg_at_{largest_k}": ndcg_at_k(
            ranked_relevances,
            list(relevance_by_frame.values()),
            largest_k,
        ),
    }
    for k in k_values:
        result[f"recall_at_{k}"] = recall_at_k(ranked_relevances, k)
    return result


def _aggregate(per_task: Sequence[Mapping[str, Any]], k_values: Sequence[int]) -> dict[str, float | int]:
    if not per_task:
        return _empty_summary(k_values)

    largest_k = k_values[-1]
    summary: dict[str, float | int] = {"task_count": len(per_task)}
    for k in k_values:
        summary[f"recall_at_{k}"] = _mean(item[f"recall_at_{k}"] for item in per_task)
    summary["mrr"] = _mean(item["reciprocal_rank"] for item in per_task)
    summary[f"ndcg_at_{largest_k}"] = _mean(item[f"ndcg_at_{largest_k}"] for item in per_task)
    return summary


def _aggregate_by_task_type(
    per_task: Sequence[Mapping[str, Any]],
    k_values: Sequence[int],
) -> dict[str, dict[str, float | int]]:
    tasks_by_type: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for task in per_task:
        tasks_by_type[str(task["task_type"])].append(task)
    return {task_type: _aggregate(tasks, k_values) for task_type, tasks in tasks_by_type.items()}


def _rankings_by_task_id(rankings: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    rankings_by_task_id = {}
    for ranking in rankings:
        task_id = _task_id(ranking)
        if task_id in rankings_by_task_id:
            raise ValueError(f"duplicate ranking for task_id: {task_id}")
        if "results" not in ranking:
            raise ValueError(f"ranking for task_id {task_id} requires results")
        rankings_by_task_id[task_id] = ranking
    return rankings_by_task_id


def _task_cards_by_task_id(
    task_cards: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    task_cards_by_task_id = {}
    for task_card in task_cards:
        task_id = _task_id(task_card)
        if task_id in task_cards_by_task_id:
            raise ValueError(f"duplicate task card for task_id: {task_id}")
        task_cards_by_task_id[task_id] = task_card
    return task_cards_by_task_id


def _relevant_frames(task_card: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    relevant_frames = task_card.get("relevant_frames", [])
    if not isinstance(relevant_frames, list) or not relevant_frames:
        raise ValueError(f"task_id {_task_id(task_card)} requires at least one relevant frame")
    seen_frames = set()
    for frame in relevant_frames:
        frame_identity = _frame_identity(frame)
        relevance = frame.get("relevance", 1)
        if isinstance(relevance, bool) or not isinstance(relevance, int) or relevance < 1:
            raise ValueError("relevance must be a positive integer")
        if frame_identity in seen_frames:
            raise ValueError(f"duplicate relevant frame for task_id {_task_id(task_card)}")
        seen_frames.add(frame_identity)
    return relevant_frames


def _unique_candidates(candidates: Any) -> list[FrameIdentity]:
    if not isinstance(candidates, list):
        raise ValueError("ranking results must be a list")

    unique_candidates = []
    seen_candidates = set()
    for candidate in candidates:
        frame_identity = _frame_identity(candidate)
        if frame_identity not in seen_candidates:
            seen_candidates.add(frame_identity)
            unique_candidates.append(frame_identity)
    return unique_candidates


def _frame_identity(frame: Mapping[str, Any]) -> FrameIdentity:
    video_id = frame.get("video_id", frame.get("videoId"))
    frame_id = frame.get("frame_id", frame.get("frameId"))
    if video_id in (None, "") or frame_id in (None, ""):
        raise ValueError("each frame requires video_id and frame_id")
    return str(video_id), str(frame_id)


def _task_id(record: Mapping[str, Any]) -> str:
    task_id = str(record.get("task_id", "")).strip()
    if not task_id:
        raise ValueError("each task card and ranking requires task_id")
    return task_id


def _task_type(task_card: Mapping[str, Any]) -> str:
    task_type = task_card.get("task_type")
    if not isinstance(task_type, str) or not task_type.strip():
        raise ValueError("task card requires task_type")
    return task_type.upper()


def _mean(values: Iterable[float]) -> float:
    values_list = list(values)
    return sum(values_list) / len(values_list) if values_list else 0.0


def _empty_summary(k_values: Sequence[int]) -> dict[str, float | int]:
    largest_k = k_values[-1]
    summary: dict[str, float | int] = {
        "task_count": 0,
        "mrr": 0.0,
        f"ndcg_at_{largest_k}": 0.0,
    }
    for k in k_values:
        summary[f"recall_at_{k}"] = 0.0
    return summary
