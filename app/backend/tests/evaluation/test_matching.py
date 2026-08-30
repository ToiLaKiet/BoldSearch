"""Spec for the matching boundary — see the contract in evaluation/matching.py.

Locked decisions live in the module docstring: exact (video_id, frame_id)
equality, per-evidence claim anti-inflation, one bool per candidate.
"""
from __future__ import annotations

from evaluation.matching import build_ranked_hits, is_match


# --- is_match -------------------------------------------------------------


def test_is_match_identical_evidence_is_hit() -> None:
    candidate = {"video_id": "L21_V001", "frame_id": "100"}
    gt_evidence = {"video_id": "L21_V001", "frame_id": "100"}

    assert is_match(candidate, gt_evidence) is True


def test_is_match_different_video_is_never_hit() -> None:
    """A different video_id must be False regardless of everything else — short-circuit property."""
    candidate = {"video_id": "L21_V002", "frame_id": "100"}
    gt_evidence = {"video_id": "L21_V001", "frame_id": "100"}

    assert is_match(candidate, gt_evidence) is False


def test_is_match_normalizes_frame_id_types() -> None:
    """Same identity convention as evidence_ranking._frame_identity: str-coerced."""
    assert is_match(
        {"video_id": "L21_V001", "frame_id": 100},
        {"video_id": "L21_V001", "frame_id": "100"},
    ) is True


# --- build_ranked_hits ----------------------------------------------------


def test_build_ranked_hits_empty_inputs_yield_no_hits() -> None:
    assert build_ranked_hits([], []) == []
    assert build_ranked_hits(
        [{"video_id": "L21_V001", "frame_id": "100"}],
        [],
    ) == [False]


def test_build_ranked_hits_duplicate_candidate_does_not_double_count() -> None:
    candidates = [
        {"video_id": "L21_V001", "frame_id": "100"},
        {"video_id": "L21_V001", "frame_id": "100"},  # exact duplicate
    ]
    gt_evidences = [{"video_id": "L21_V001", "frame_id": "100"}]

    hits = build_ranked_hits(candidates, gt_evidences)

    assert hits.count(True) == 1  # count each (video_id, frame_id) exactly once


def test_build_ranked_hits_keeps_one_bool_per_candidate() -> None:
    """Dedupe must never drop positions — ranks feed MRR unchanged."""
    candidates = [
        {"video_id": "L21_V001", "frame_id": "98"},
        {"video_id": "L21_V001", "frame_id": "100"},
        {"video_id": "L21_V001", "frame_id": "102"},
    ]
    gt_evidences = [{"video_id": "L21_V001", "frame_id": "100"}]

    assert build_ranked_hits(candidates, gt_evidences) == [False, True, False]


def test_build_ranked_hits_claims_each_evidence_once_in_rank_order() -> None:
    """Two GT frames hit by two candidates → each claims one, in rank order."""
    candidates = [
        {"video_id": "L21_V001", "frame_id": "100"},
        {"video_id": "L21_V002", "frame_id": "200"},
        {"video_id": "L21_V002", "frame_id": "200"},  # duplicate claims nothing
    ]
    gt_evidences = [
        {"video_id": "L21_V002", "frame_id": "200"},
        {"video_id": "L21_V001", "frame_id": "100"},
    ]

    assert build_ranked_hits(candidates, gt_evidences) == [True, True, False]
