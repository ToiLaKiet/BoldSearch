from evaluation.metrics.ranking import first_relevant_rank, ndcg_at_k, recall_at_k, reciprocal_rank


def test_ranking_metrics_score_relevance_without_task_card_dependencies() -> None:
    ranked_relevances = [0, 1, 2]

    assert first_relevant_rank(ranked_relevances) == 2
    assert recall_at_k(ranked_relevances, 1) == 0.0
    assert recall_at_k(ranked_relevances, 2) == 1.0
    assert reciprocal_rank(ranked_relevances) == 0.5
    assert ndcg_at_k(ranked_relevances, ideal_relevances=[2, 1], k=3) == 0.58688267143572


def test_ranking_metrics_handle_missing_relevant_candidates_and_large_k_values() -> None:
    ranked_relevances = [0, 0]

    assert first_relevant_rank(ranked_relevances) is None
    assert recall_at_k(ranked_relevances, 10) == 0.0
    assert reciprocal_rank(ranked_relevances) == 0.0
    assert ndcg_at_k(ranked_relevances, ideal_relevances=[], k=10) == 0.0
