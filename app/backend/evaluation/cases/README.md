# Evaluation test cases

These files are **synthetic templates** for developing the evaluation pipeline.
They are not human-verified relevance judgments and must not be used to claim
that one embedding model is better than another.

## Layout

```text
cases/
  kis/v1.jsonl
  vqa/v1.jsonl
  trake/v1.jsonl
```

Each JSONL line is one `test_case`. The shared fields are:

```json
{
  "case_id": "kis-001",
  "task_type": "KIS",
  "query": {"texts": ["..."]},
  "relevant_frames": [
    {"video_id": "SYNTH_V001", "frame_id": "100", "relevance": 2}
  ],
  "source": "synthetic-template"
}
```

- `case_id` is the stable identifier used to join a test case with a ranking.
- `query.texts` preserves ordered text inputs; TRAKE uses more than one text.
- `relevant_frames` is the evidence ground truth used by the current ranking
  metrics.
- `source` makes it explicit that these labels are placeholders.

Task-specific fields:

- **KIS**: one target frame is expected.
- **VQA**: `expected_answer` is the answer contract; the current evidence
  evaluator only scores the supporting frame, not answer correctness.
- **TRAKE**: `expected_track` preserves the required frame order; the current
  evidence evaluator only scores whether relevant evidence appears, not temporal
  order or track continuity.

Ranking exports should use the same `case_id`:

```json
{"case_id":"kis-001","results":[{"video_id":"SYNTH_V001","frame_id":"100"}]}
```

Replace these synthetic cases with human-verified cases before using them as a
model-selection gate.

The current legacy runner still expects `task_id` because the core rename has
not landed yet. These fixtures intentionally use the new `case_id` vocabulary;
the next refactor step is to update the runner and its tests together.
