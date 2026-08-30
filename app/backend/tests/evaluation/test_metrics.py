import pytest

from evaluation.tasks.evidence_ranking import evaluate_evidence_rankings


def test_evaluate_rankings_reports_recall_mrr_and_ndcg_for_evidence_frames() -> None:
    task_cards = [
        {
            "task_id": "kis-001",
            "task_type": "KIS",
            "relevant_frames": [
                {"video_id": "L21_V001", "frame_id": "100", "relevance": 2},
                {"video_id": "L21_V002", "frame_id": "200", "relevance": 1},
            ],
        }
    ]
    rankings = [
        {
            "task_id": "kis-001",
            "results": [
                {"video_id": "L21_V999", "frame_id": "1"},
                {"video_id": "L21_V002", "frame_id": "200"},
                {"video_id": "L21_V001", "frame_id": "100"},
            ],
        }
    ]

    report = evaluate_evidence_rankings(task_cards, rankings, k_values=[1, 2, 3])

    assert report["overall"] == {
        "task_count": 1,
        "recall_at_1": 0.0,
        "recall_at_2": 1.0,
        "recall_at_3": 1.0,
        "mrr": 0.5,
        "ndcg_at_3": 0.58688267143572,
    }
    assert report["by_task_type"]["KIS"] == report["overall"]
    assert report["per_task"] == [
        {
            "task_id": "kis-001",
            "task_type": "KIS",
            "first_relevant_rank": 2,
            "reciprocal_rank": 0.5,
            "recall_at_1": 0.0,
            "recall_at_2": 1.0,
            "recall_at_3": 1.0,
            "ndcg_at_3": 0.58688267143572,
        }
    ]


def test_evaluate_rankings_requires_one_ranking_for_every_task_card() -> None:
    task_cards = [
        {
            "task_id": "kis-001",
            "task_type": "KIS",
            "relevant_frames": [{"video_id": "L21_V001", "frame_id": "100"}],
        }
    ]
    rankings = [{"task_id": "kis-002", "results": []}]

    with pytest.raises(ValueError, match="task card and ranking task_ids must match"):
        evaluate_evidence_rankings(task_cards, rankings, k_values=[1])


def test_evaluate_rankings_rejects_non_positive_relevance() -> None:
    task_cards = [
        {
            "task_id": "kis-001",
            "task_type": "KIS",
            "relevant_frames": [
                {"video_id": "L21_V001", "frame_id": "100", "relevance": -1}
            ],
        }
    ]
    rankings = [{"task_id": "kis-001", "results": []}]

    with pytest.raises(ValueError, match="relevance must be a positive integer"):
        evaluate_evidence_rankings(task_cards, rankings, k_values=[1])


def test_evaluate_rankings_rejects_duplicate_evidence_frames() -> None:
    task_cards = [
        {
            "task_id": "kis-001",
            "task_type": "KIS",
            "relevant_frames": [
                {"video_id": "L21_V001", "frame_id": "100", "relevance": 2},
                {"video_id": "L21_V001", "frame_id": "100", "relevance": 1},
            ],
        }
    ]
    rankings = [{"task_id": "kis-001", "results": []}]

    with pytest.raises(ValueError, match="duplicate relevant frame"):
        evaluate_evidence_rankings(task_cards, rankings, k_values=[1])


@pytest.mark.parametrize("k_values", [[True], [1.5], [0]])
def test_evaluate_rankings_rejects_invalid_k_values(k_values) -> None:
    task_cards = [
        {
            "task_id": "kis-001",
            "task_type": "KIS",
            "relevant_frames": [{"video_id": "L21_V001", "frame_id": "100"}],
        }
    ]
    rankings = [{"task_id": "kis-001", "results": []}]

    with pytest.raises(ValueError, match="k values must contain positive integers"):
        evaluate_evidence_rankings(task_cards, rankings, k_values=k_values)


def test_evaluate_rankings_rejects_empty_benchmark_inputs() -> None:
    with pytest.raises(ValueError, match="task cards must not be empty"):
        evaluate_evidence_rankings([], [], k_values=[1])


def test_evaluate_rankings_keeps_task_type_scores_separate() -> None:
    task_cards = [
        {
            "task_id": "kis-001",
            "task_type": "KIS",
            "relevant_frames": [{"video_id": "L21_V001", "frame_id": "100"}],
        },
        {
            "task_id": "vqa-001",
            "task_type": "VQA",
            "relevant_frames": [{"video_id": "L21_V002", "frame_id": "200"}],
        },
    ]
    rankings = [
        {"task_id": "kis-001", "results": [{"video_id": "L21_V001", "frame_id": "100"}]},
        {"task_id": "vqa-001", "results": []},
    ]

    report = evaluate_evidence_rankings(task_cards, rankings, k_values=[1])

    assert report["overall"]["recall_at_1"] == 0.5
    assert report["by_task_type"]["KIS"]["recall_at_1"] == 1.0
    assert report["by_task_type"]["VQA"]["recall_at_1"] == 0.0


def test_evaluate_rankings_requires_task_type() -> None:
    task_cards = [
        {
            "task_id": "kis-001",
            "relevant_frames": [{"video_id": "L21_V001", "frame_id": "100"}],
        }
    ]
    rankings = [{"task_id": "kis-001", "results": []}]

    with pytest.raises(ValueError, match="task card requires task_type"):
        evaluate_evidence_rankings(task_cards, rankings, k_values=[1])
