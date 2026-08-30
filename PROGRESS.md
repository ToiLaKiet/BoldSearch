### [2026-08-30 12:03 UTC+07:00] — [Docs] Move workflow documents to repository root

**Done:**

- Moved `GIT_CONVENTION.md`, `PROGRESS.md`, and `BLOCKERS.md` to the repository root with `git mv` (history preserved); `docs/` keeps only long-lived guides.
- Added repository-wide Copilot instructions (`.github/copilot-instructions.md`) carrying the Conventional Commit rules for Copilot-generated commit messages.
- Updated active references in `README.md` and `AGENTS.md`; historical log entries below keep their original paths.

**Changed files:**

- `GIT_CONVENTION.md`, `PROGRESS.md`, `BLOCKERS.md` (moved from `docs/`)
- `.github/copilot-instructions.md` (new)
- `CLAUDE.md` (new; imports `AGENTS.md` via `@AGENTS.md`)
- `README.md`, `AGENTS.md` (references)

**Verified:** `git diff --check` clean; renames detected by `git status`; markdown links to the new root paths resolve.

### [2026-08-27 18:29 UTC+07:00] — [Feature] Land keyframe media serving; evaluation core back to WIP

**Done:**

- Committed the keyframe media chain as one feature commit: static mounts for `/keyframes` and `/map-keyframes`, data-dir config with path validators, thumbnail mapping via nearest keyframe lookup, and its tests.
- Committed the evaluation module, then reset the commit — review is not finished; the module is untracked WIP again.
- Split ignore rules for the local data dir and evaluation runs into a chore commit.
- Verified the full backend suite before committing.

**Changed files:**

- `app/backend/main.py`, `app/backend/app_config.py` — static media mounts and local data paths
- `app/backend/search/service.py`, `app/backend/search/object_index.py` — thumbnail map lookup
- `app/backend/tests/test_thumbnail_mapping.py`, `app/backend/tests/test_data_paths.py` — tests
- `app/backend/evaluation/**`, `app/backend/tests/evaluation/**` — evaluation core (untracked WIP, review pending)
- `.gitignore`, `app/.gitignore` — ignore `/data/` and evaluation runs

**Verified:** `uv run pytest tests/ -q` → 41 passed, 4 skipped; `uv run pytest tests/evaluation tests/test_data_paths.py -q` → 39 passed, 4 skipped.

### [2026-08-27 13:15 UTC+07:00] — [Docs] Replace root README

**Done:**

- Rewrote the repository entry point around the current React + FastAPI retrieval workflow, data prerequisites, API surface, runtime configuration, verification commands, and operator submission boundary.
- Checked all local Markdown links, fenced code blocks, and secret-shaped values in the new document.

**Changed files:**

- `README.md` — replaced with an evidence-backed product and contributor guide.

**Verified:** `git diff --check -- README.md`; local Markdown links resolve; fenced blocks are balanced; README contains no secret-shaped value.

### [2026-08-27 10:26 UTC+07:00] - [Docs] Add repository agent guidance

**Done:**

- Added concise repository instructions covering application boundaries, setup commands, focused verification, configuration and data paths, evaluation-contract limitations, submission constraints, and commit conventions.

**Changed files:**

- `AGENTS.md` - created verified OpenCode guidance for future sessions.

**Verified:** `uv run pytest tests/evaluation -q` -> 35 passed, 4 skipped; `npm test` -> 5 passed; `npm run build` -> passed.

### [2026-08-27 13:21 UTC+07:00] — [Evaluation] Merge cases README into global eval README; refresh stale docs

**Done:**

- `evaluation/README.md` is now the single global doc for the module: layout tree covering runner/report/matching/metrics/tasks/cases/runs, metric semantics (Hit@K naming honesty, MRR, nDCG), input schemas, run command, merged test-case schema + vision-draft lifecycle (from `cases/README.md`), and Manual CI.
- Removed stale claims: "Rich terminal table" (renderer is now plain text) and the metrics/`ranking.py`-only description (recall/mrr/aggregate/matching split undocumented before).
- Deleted `cases/README.md` (content merged); no dangling references outside historical PROGRESS entries.

**Changed files:**

- `app/backend/evaluation/README.md` — rewritten as global module doc.
- `app/backend/evaluation/cases/README.md` — deleted (merged).

**Verified:** re-read + grep: no "Rich" claims, no dangling `cases/README` references outside historical logs.

### [2026-08-27 13:07 UTC+07:00] — [Config] Flatten app_config to HEAD layout + eval storage fields (grouped refactor removed)

**Done:**

- `app_config.py`: removed the 7 grouped settings classes + 7 mapping properties (a previous-session uncommitted refactor); now a single flat `AppConfig` = HEAD layout plus the four storage fields the eval component and keyframe serving need (`DATA_DIR`, `KEYFRAMES_DIR`, `KEYFRAME_MAP_DIR`, `EVALUATION_ARTIFACT_DIR`) with relative-path resolution and DATA_DIR re-rooting kept from the working-tree logic.
- Migrated ~35 consumer sites from grouped access to flat fields across `main.py`, `connections.py`, `search/service.py`, `search/object_index.py`, `evaluation/runner.py`, `tests/test_data_paths.py`, `tests/test_thumbnail_mapping.py` (SimpleNamespace fixtures included).
- Environment variable names unchanged (flat, e.g. `ZILLIZ_TOKEN`, `DATA_DIR`) — existing `.env` keeps working.
- Deferred follow-up (user-owned): nested settings refactor (`env_nested_delimiter="__"`, verified feasible on pydantic-settings 2.7) is planned as a separate future feature.

**Changed files:**

- `app/backend/app_config.py`, `app/backend/main.py`, `app/backend/connections.py`, `app/backend/search/service.py`, `app/backend/search/object_index.py`, `app/backend/evaluation/runner.py`, `app/backend/tests/test_data_paths.py`, `app/backend/tests/test_thumbnail_mapping.py`.

**Verified:** `uv run pytest -q` → 41 passed, 4 skipped; config import smoke (`KEYFRAMES_DIR`, `EVALUATION_ARTIFACT_DIR` resolve correctly); demo eval CLI prints the summary table.

### [2026-08-27 12:57 UTC+07:00] — [Evaluation] Rename presentation module to report.py

**Done:**

- `evaluation/presentation.py` → `evaluation/report.py`; test file renamed to match. Name now describes content (it renders reports) instead of a layer role.
- Runner + test imports updated; no other references existed (grep-verified; PROGRESS log entries are historical records and stay as written).
- Re-confirmed the runner entry contract: `python evaluation/runner.py` (script-style) raises `ModuleNotFoundError` by design of package-internal modules; supported invocation is `uv run python -m evaluation.runner` from `app/backend`.

**Changed files:**

- `app/backend/evaluation/report.py` (renamed from `presentation.py`), `app/backend/evaluation/runner.py`, `app/backend/tests/evaluation/test_report.py` (renamed from `test_presentation.py`).

**Verified:** `uv run pytest tests/evaluation -q` → 35 passed, 4 skipped; demo CLI prints the summary table via `python -m evaluation.runner`.

### [2026-08-27 12:53 UTC+07:00] — [Evaluation] Replace rich table with dependency-free text renderer

**Done:**

- `presentation.py`: `build_summary_table` (rich Table) → `render_summary_text` (fixed-width f-string table, str→str, deterministic) reusing `_cutoffs`/`_metric_headers`/`_scopes`/`_summary_cells`.
- `runner.py`: prints plain text; dropped `rich.console` import — `rich` now has zero usages in the backend (removal from `pyproject.toml` left as a one-line follow-up, not yet done).
- Updated specs: `test_presentation.py` asserts returned string directly (no Console ceremony); `test_runner.py` CLI test renamed to text-summary and asserts header row (fixed a double `capsys.readouterr()` drain bug while updating).

**Changed files:**

- `app/backend/evaluation/presentation.py`, `app/backend/evaluation/runner.py`, `app/backend/tests/evaluation/test_presentation.py`, `app/backend/tests/evaluation/test_runner.py`.

**Verified:** `uv run pytest -q` → 41 passed, 4 skipped; demo CLI (`evaluation/runs/demo-cli/`) prints the fixed-width table end-to-end.

### [2026-08-27 12:29 UTC+07:00] — [Evaluation] Simplify presentation module (refactor, behavior-identical)

**Done:**

- Deduplicated `presentation.py` (90 → 82 lines): shared `_metric_headers`, `_scopes`, `_markdown_definitions` now feed both the Rich table and the Markdown report; removed the redundant `_cutoffs` Sequence/str/bytes validation (runner already validates cutoffs at its entry — validation-layers policy).
- Kept CommonMark `_code_span` backtick-run handling and cell escaping — both are tested behavior.
- Also converted all metric-core module and test comments to English (recall/mrr/aggregate/matching + their tests) as a prior chore step.
- Legacy contract untouched: report shape (`cutoffs`/`overall`/`by_task_type`, recall/mrr/ndcg keys) unchanged for the legacy runner.

**Changed files:**

- `app/backend/evaluation/presentation.py` — refactor.
- `app/backend/evaluation/metrics/{recall,mrr,aggregate}.py`, `app/backend/evaluation/matching.py`, `app/backend/tests/evaluation/test_{recall,mrr,aggregate,matching}.py` — English comments only, no logic change.

**Verified:** `uv run pytest tests/evaluation -q` → 35 passed, 4 skipped (identical before/after refactor).

### [2026-08-27 00:39 UTC+07:00] — [Evaluation] Split metric core into per-metric modules

**Done:**

- Replaced single `metrics/baseline.py` with `metrics/recall.py` (Hit@K), `metrics/mrr.py` (`first_hit_rank` + `reciprocal_rank`), `metrics/aggregate.py` (cross-query `mean`; future home of macro-average + `n_missing`), and package-level `evaluation/matching.py` (`is_match` + `build_ranked_hits` TODO(human) stubs with contracts).
- Module cut follows pipeline stage: per-query formulas vs cross-query aggregation vs GT-matching boundary. Matching moves into per-task modules when task contracts fork (KIS tolerance vs TRAKE anchors).
- Split spec tests to mirror modules: `test_recall.py`, `test_mrr.py`, `test_aggregate.py`, `test_matching.py`.
- Fixed one wrong spec assertion during verification (`first_hit_rank([F,F,T])` is 3, not 2).

**Changed files:**

- `app/backend/evaluation/metrics/{recall,mrr,aggregate}.py` — new per-stage metric modules.
- `app/backend/evaluation/matching.py` — GT-matching boundary with TODO(human) #1/#2 contracts.
- `app/backend/tests/evaluation/test_{recall,mrr,aggregate,matching}.py` — split specs.
- Removed `evaluation/metrics/baseline.py` + `tests/evaluation/test_baseline_metrics.py`.

**Verified:** `uv run pytest tests/evaluation -q` → 35 passed, 4 skipped (same case count as pre-split; legacy suite untouched and green).

### [2026-08-27 00:11 UTC+07:00] — [Evaluation] Rebuild metric core as baseline module (step 1)

**Done:**

- Reviewed legacy `metrics/ranking.py` + `tasks/evidence_ranking.py`: misnamed Hit@K (`recall_at_k`), unused graded-relevance machinery (nDCG, int-relevance validation), dual-casing normalization (`video_id`/`videoId`), validation spread over three layers.
- Added clean baseline v1 core on `Sequence[bool]`: `mean`, `hit_at_k`, `reciprocal_rank` implemented; `is_match`, `build_ranked_hits` left as TODO(human) stubs with written contracts.
- Legacy module untouched — swap happens after VQA-answer/TRAKE evaluators land.

**Changed files:**

- `app/backend/evaluation/metrics/baseline.py` — new pure metric core with TODO(human) contracts.
- `app/backend/tests/evaluation/test_baseline_metrics.py` — spec tests; 4 skipped pending TODO(human) implementation.

**Verified:** `uv run pytest tests/evaluation -q` → 31 passed, 4 skipped (baseline before change: 21 passed).

### [2026-08-26 13:51 UTC+07:00] — [Evaluation] Add vision-draft cases from local keyframes

**Done:**

- Inspected real local keyframes and resolved source `frame_id` values through `data/map-keyframes/`.
- Added 10 KIS, 10 VQA, and 10 TRAKE cases using real corpus images.
- Marked every case `vision-draft` and `review_required` so these labels cannot be mistaken for approved qrels.

**Changed files:**

- `app/backend/evaluation/cases/kis/vision-draft-v1.jsonl` — 10 real-frame KIS cases.
- `app/backend/evaluation/cases/vqa/vision-draft-v1.jsonl` — 10 real-frame VQA cases.
- `app/backend/evaluation/cases/trake/vision-draft-v1.jsonl` — 10 real-frame temporal draft cases.
- `app/backend/evaluation/cases/README.md` — local-corpus provenance and review guidance.

**Flow explained:**

`local keyframe JPG -> map-keyframes n -> source frame_id -> vision-draft query/qrel case`.

### [2026-08-26 12:53 UTC+07:00] — [Evaluation] Add versioned synthetic test-case templates

**Done:**

- Added 30 clearly synthetic JSONL test cases: 10 KIS, 10 VQA, and 10 TRAKE.
- Documented the shared `case_id`/`query`/`relevant_frames` envelope and task-specific fields.
- Updated evaluation documentation to distinguish the new `case_id` templates from the legacy `task_id` runner.

**Changed files:**

- `app/backend/evaluation/cases/` — test-case templates and schema documentation.
- `app/backend/evaluation/README.md`, `app/backend/README.md` — evaluation terminology and layout.

**Flow explained:**

`versioned test case -> model ranking export -> future case_id-aware evaluator -> quality report`.

### [2026-08-25 21:05 UTC+07:00] — [Config] Group application settings by responsibility

**Done:**

- Split `AppConfig` into API, server, Zilliz, encoder, object-index, storage, and response setting models.
- Updated backend consumers to use the responsible setting group.
- Restored nearest-keyframe thumbnail resolution through the configured keyframe-map directory.
- Verified grouped configuration views and the full backend test suite.

**Changed files:**

- `app/backend/app_config.py` — grouped settings views over the existing environment contract.
- `app/backend/connections.py`, `app/backend/main.py`, `app/backend/search/`, `app/backend/evaluation/runner.py` — grouped configuration consumers.
- `app/backend/tests/test_data_paths.py`, `app/backend/tests/test_thumbnail_mapping.py` — grouped configuration and thumbnail mapping coverage.

**Flow explained:**

`.env` value -> `AppConfig` field -> typed settings group -> backend consumer -> configured runtime behavior.

### [2026-08-25 07:30 UTC+07:00] — [Data] Move static media outside the frontend bundle

**Done:**

- Moved the 29 GB keyframe corpus and map-keyframe CSVs from Vite `public/` into ignored root `data/` storage.
- Added centralized FastAPI configuration for data, static media, and evaluation artifact paths.
- Mounted static-media routes in FastAPI and proxied them through Vite, preserving `/keyframes` and `/map-keyframes` URLs.
- Normalized Vite proxy targets and restored root ignore rules so API paths and local secrets remain safe after the migration.
- Verified that the frontend production build no longer copies the media corpus.

**Changed files:**

- `.gitignore`, `data/keyframes`, `data/map-keyframes` — ignored local corpus storage.
- `app/backend/app_config.py`, `app/backend/main.py`, `app/backend/.env` — data path configuration and static routes.
- `app/frontend/vite.config.js` — development proxy for static media.
- `README.md`, `app/README.md`, `docs/ARCHITECTURE.md`, `docs/BLOCKERS.md` — current runtime and resolved build blocker.

**Flow explained:**

Browser `/keyframes` or `/map-keyframes` URL -> Vite proxy in development, or `VITE_STATIC_MEDIA_URL`/API origin in production -> FastAPI static mount -> ignored root `data/` asset.

### [2026-08-25 07:03 UTC+07:00] — [Evaluation] Add reproducible offline evidence-ranking gate

**Done:**

- Added an offline JSONL runner that scores exported retrieval rankings without loading FastAPI, FG-CLIP, or Milvus.
- Separated pure ranking metrics from the evidence-ranking task adapter so future VQA answer and TRAKE temporal evaluators can use their own contracts.
- Added strict task/ranking validation, immutable run provenance, input SHA-256 hashes, and output-overwrite protection.
- Added Rich terminal summaries, Markdown report artifacts, and a manual GitHub Actions workflow for artifact upload.

**Changed files:**

- `app/backend/evaluation/` — ranking metric formulas, evidence-ranking adapter, CLI runner, and contract documentation.
- `app/backend/tests/evaluation/` — metric, task-contract, provenance, and runner regression coverage.
- `app/backend/pyproject.toml`, `app/backend/uv.lock` — Rich terminal rendering dependency.
- `.github/workflows/evaluate-rankings.yml` — manual evaluation-report artifact workflow.
- `app/.gitignore`, `app/backend/README.md`, `docs/technical/00-embedding-vector-store-evaluation.md` — evaluation artifact handling and usage references.

**Flow explained:**

Versioned task cards + exported query rankings + immutable run metadata -> offline task adapter -> pure ranking metrics -> SHA-attested JSON report.

### [2026-08-25 05:20 UTC+07:00] — [Architecture] Align the documented runtime with the retrieval system

**Done:**

- Replaced the obsolete Flask/sample-data architecture with the implemented FastAPI, FG-CLIP, Zilliz/Milvus, CSV detection-metadata, and Vite static-media flow.
- Defined the search capability boundary and an incremental refactoring path: isolate Milvus translation, extract pure result/temporal logic, then add an evaluated provider contract only when needed.
- Identified external static-media hosting as the required deployment path for the keyframe corpus, avoiding Vite build duplication.
- Regenerated the system-overview SVG from the updated Mermaid source.

**Changed files:**

- `docs/ARCHITECTURE.md`, `architecture/system-overview.mmd`, `architecture/system-overview.svg` — current runtime architecture and diagram.
- `README.md`, `app/README.md`, `app/backend/README.md` — synchronized application structure and operations guidance.
- `docs/technical/00-embedding-vector-store-evaluation.md` — separated current Milvus runtime constraints from future benchmark work.

**Flow explained:**

React UI -> FastAPI search routes -> FG-CLIP -> Zilliz/Milvus hybrid retrieval -> detection metadata enrichment -> normalized frame results and local submission payloads.

### [2026-08-25 04:59 UTC+07:00] — [Search] Remove LFM reranking implementation

**Done:**

- Removed the LFM encoder, reranking implementation, evaluation command, fixture, and LFM-specific tests.
- Restored the backend files that integrated LFM to their `HEAD` versions.
- Retained `data/` (737 files) and all 873 map-keyframe CSV files for revising the existing system.
- Verified that the backend imports successfully and contains no remaining LFM references.

**Changed files:**

- `app/backend/encoders/lfm.py` — deleted.
- `app/backend/search/lfm_evaluation.py` — deleted.
- `app/backend/search/lfm_reranking.py` — deleted.
- `app/backend/tests/fixtures/lfm_rerank_eval.jsonl` — deleted.
- `app/backend/tests/test_lfm_*.py` — deleted.
- `app/backend/app_config.py`, `app/backend/encoders/loader.py`, `app/backend/main.py`, `app/backend/search/router.py`, `app/backend/search/service.py`, `app/backend/README.md` — restored to `HEAD`.

**Flow explained:**

Existing search backend -> FG-CLIP/Milvus retrieval -> existing response path; data and keyframe mappings remain available for the next revision.
