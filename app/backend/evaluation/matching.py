"""Candidate ↔ ground-truth matching boundary — the input to every metric.

Not a metric: this is where "what counts as a hit" is decided. Every
score discrepancy originates from exactly one place. When task contracts
fork (KIS by tolerance range, TRAKE by anchors), each task's match
function moves into its own task module — for now one shared contract,
because GT is currently loose frame_id points.

Locked decisions:
- Matching key: exact (video_id, frame_id) equality after ``str()``
  coercion — same identity as ``tasks.evidence_ranking._frame_identity``.
  Timestamp tolerance is deferred until real annotations carry ranges;
  when they do, per-task match functions fork out of this module.
- Anti-inflation: each GT evidence is claimed at most once, by its first
  matching candidate. N consecutive frames of one video therefore
  contribute at most one hit per GT evidence.
- Output alignment: one bool per input candidate, in ranking order —
  dedupe must never drop positions, or ranks shift and MRR lies.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

FrameLike = Mapping[str, Any]


def is_match(candidate: FrameLike, gt_evidence: FrameLike) -> bool:
    """Decide whether one candidate counts as hitting one GT evidence (1-vs-1)."""
    candidate_video, candidate_frame = _frame_identity(candidate)
    gt_video, gt_frame = _frame_identity(gt_evidence)
    return candidate_video == gt_video and candidate_frame == gt_frame


def build_ranked_hits(
    candidates: Sequence[FrameLike],
    gt_evidences: Sequence[FrameLike],
) -> list[bool]:
    """Collapse ranked candidates into a hit sequence, claiming each evidence once."""
    unclaimed = list(range(len(gt_evidences)))
    hits: list[bool] = []
    for candidate in candidates:
        claimed = next(
            (index for index in unclaimed if is_match(candidate, gt_evidences[index])),
            None,
        )
        if claimed is not None:
            unclaimed.remove(claimed)
        hits.append(claimed is not None)
    return hits


def _frame_identity(frame: FrameLike) -> tuple[str, str]:
    return str(frame["video_id"]), str(frame["frame_id"])
