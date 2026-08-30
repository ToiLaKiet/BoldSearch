# Query Evaluation

`evaluation` measures exported rankings without starting FastAPI, FG-CLIP, or Milvus. It is the first gate for query, fusion, and embedding changes: compare the same versioned test cases against different ranking exports before re-embedding the full corpus.

## Layout

```text
evaluation/
├── runner.py                  # CLI entry: orchestration, validation, report writing
├── report.py                  # summary rendering: fixed-width terminal table + Markdown artifact
├── matching.py                # candidate ↔ GT boundary: is_match, build_ranked_hits (WIP)
├── metrics/
│   ├── recall.py              # Hit@K (set-based recall lands here when coverage matters)
│   ├── mrr.py                 # first_hit_rank + reciprocal_rank
│   ├── aggregate.py           # cross-query mean; future macro-average + n_missing
│   └── ranking.py             # legacy formulas consumed by the current evidence adapter
├── tasks/
│   └── evidence_ranking.py    # current frame-evidence evaluator (KIS/VQA evidence, TRAKE evidence)
├── cases/                     # versioned test-case templates, split by task
└── runs/                      # gitignored local experiment artifacts
```

Metric formulas stay pure functions over `Sequence[bool]`. Future answer-scoring and temporal-track evaluators belong in separate modules under `tasks/`; they should reuse metric formulas rather than extend the evidence-ranking adapter.

## Metrics

- **Hit@K** — at least one relevant evidence frame appears in the first K unique candidates. The legacy module names it `recall_at_k`; the rebuilt module calls it `hit_at_k` on purpose.
- **MRR** — `1 / rank` of the first relevant frame, averaged across tasks.
- **nDCG@largest-k** — `2^relevance − 1` gains (legacy adapter only).

Report columns: `Tasks`, one `Recall@K` per k value, `MRR`, `nDCG@largest`. K values are normalized to sorted unique positive integers. Relevance must be a positive integer; each evidence frame appears once per task. Empty task or ranking files are rejected rather than reported as zero-score experiments.

## Inputs

Task cards are versioned JSONL. Each line needs a stable `task_id`, a task type, and at least one evidence frame:

```json
{"task_id":"kis-001","task_type":"KIS","relevant_frames":[{"video_id":"L21_V001","frame_id":"100","relevance":2}]}
```

Ranking exports are JSONL. Each line has the matching `task_id` and ranked candidates; the two ID sets must match exactly. `video_id`/`frame_id` and the frontend-compatible `videoId`/`frameId` are accepted:

```json
{"task_id":"kis-001","results":[{"video_id":"L21_V009","frame_id":"5"},{"video_id":"L21_V001","frame_id":"100"}]}
```

Run metadata is a required JSON object with `run_id`, `corpus_version`, `corpus_sha256`, `collection`, `index_config_sha256`, `encoder`, `encoder_revision`, `preprocessing_version`, and `query_strategy`.

## Run

The usual flow is one command against the serving backend: it queries the API per case, writes an immutable run folder, and prints the score table.

```bash
cd app/backend
uv run python -m evaluation.export --run-id baseline-v1                # all cases
uv run python -m evaluation.export --run-id baseline-v1 --cases kis-001,kis-003
```

Requirements: the backend must be serving (`uv run python main.py`), and three human labels must be filled in `app/backend/.env` (gitignored): `EVALUATION_CORPUS_VERSION`, `EVALUATION_PREPROCESSING_VERSION`, and `EVALUATION_QUERY_STRATEGY`. The hashes and encoder identity are computed at export time — `corpus_sha256` fingerprints the keyframes tree, `index_config_sha256` fingerprints the Milvus settings, and `encoder`/`encoder_revision` come from the pinned FG-CLIP constants. Missing labels are rejected with their names. Each run writes `evaluation/runs/<run-id>/` containing `tasks.jsonl`, `rankings.jsonl`, `metadata.json`, `report.json`, and `report.md`; existing run folders are never overwritten. Cases use `case_id`; the exporter bridges it to the runner's `task_id` vocabulary.

The runner CLI stays available for scoring JSONL exports directly, without the API:

```bash
uv run python -m evaluation.runner \
  --tasks evaluation/runs/demo-cli/tasks.jsonl \
  --rankings evaluation/runs/demo-cli/rankings.jsonl \
  --metadata evaluation/runs/demo-cli/metadata.json \
  --output evaluation/runs/demo-cli/report.json \
  --markdown-output evaluation/runs/demo-cli/report.md \
  --k 1,5,10
```

The CLI prints a fixed-width summary table; `--markdown-output` writes the same summary plus provenance as a portable CI artifact. The JSON report contains overall scores, scores by `task_type`, per-task evidence ranks, and SHA-256 hashes for every input file. `evaluation/runs/` holds ranking exports and generated reports; they are local artifacts and stay out of git. Commit approved test cases and qrels.

## Test cases

`cases/` holds versioned JSONL templates split by task — KIS, VQA, TRAKE. They are **not** approved relevance judgments and must not be used to claim that one embedding model beats another.

```json
{
  "case_id": "kis-001",
  "task_type": "KIS",
  "query": {"texts": ["..."]},
  "relevant_frames": [{"video_id": "SYNTH_V001", "frame_id": "100", "relevance": 2}],
  "source": "synthetic-template"
}
```

- `case_id` is the stable identifier used to join a test case with a ranking.
- `query.texts` preserves ordered text inputs; TRAKE uses more than one text.
- `relevant_frames` is the evidence ground truth used by the current ranking metrics.
- **KIS**: one target frame is expected.
- **VQA**: `expected_answer` is the answer contract; the current evidence evaluator scores the supporting frame only, not answer correctness.
- **TRAKE**: `expected_track` preserves the required frame order; the current evidence evaluator scores whether relevant evidence appears, not temporal order or track continuity.

### Vision-draft cases from the local corpus

`vision-draft-v1.jsonl` files use real images from the ignored local corpus under `data/keyframes/`. Their `frame_id` values are source-frame IDs resolved through `data/map-keyframes/<video_id>.csv`; `keyframe_number` is an audit pointer to the inspected JPG. Labels were selected by visual inspection and carry `annotation_status: "vision-draft"` plus `review_required: true` — a human must confirm query wording, VQA answers, and TRAKE event boundaries before they become benchmark qrels.

The legacy runner expects `task_id` while these fixtures use the new `case_id` vocabulary; the runner migration renames that field and updates the runner and its tests together.

## Manual CI

Run the `Evaluate rankings` workflow from GitHub Actions with paths relative to `app/backend` for the task-record file, ranking export, and metadata file. The workflow uploads `report.json` and `report.md` as the `evaluation-report` artifact. Inputs must exist in the checked-out revision; upload a ranking export to the repository, or make it available through an approved preceding step, before dispatching the run.
