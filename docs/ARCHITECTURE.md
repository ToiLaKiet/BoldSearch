# Architecture Guide

Status: **project convention + planning baseline**  
Project: HCM AI Challenge Pipeline 2026 / BoldSearch

## Source and claim status

| Claim | Status | Evidence |
|---|---|---|
| Current app is a Flask API plus Vite React UI prototype. | Verified | `app/README.md`, `app/backend/app.py`, `app/frontend/package.json` |
| Current search is lexical/object/color/temporal over sample `shots.json`. | Verified | `app/backend/app.py`, `app/backend/data/shots.json` |
| Embedding/vector-store pipeline is planned but not production-implemented. | Verified | dependency inventory and `docs/technical/00-embedding-vector-store-evaluation.md` |
| Final vector-store provider should be chosen by benchmark, not preference. | Inferred project decision | `docs/technical/00-embedding-vector-store-evaluation.md` |
| BEiT-3 checkpoint, dataset scale, SLO, and benchmark weights remain open. | Unresolved | no approved config or ADR in repo |

## Architecture style

Use a **modular monolith** for the 2026 challenge pipeline until benchmark or deployment constraints prove a split service is needed.

Why:

- The current system is small and benefits from fast local iteration.
- Retrieval quality and benchmark correctness are more important than service topology.
- Provider adapters already isolate the likely volatile parts: model runtimes and vector databases.

## System view

Source diagram: `architecture/system-overview.mmd`; exported preview: `architecture/system-overview.svg`.

```text
Challenge operator / participant
  -> React UI
  -> Flask API
  -> retrieval/submission use cases
  -> pure scoring and validation policies
  -> shot catalog repository, embedding encoders, vector-store adapters
  -> sample JSON now; embedding artifacts and selected vector DB later
```

## Main capabilities

| Capability | Purpose | Current state | Target direction |
|---|---|---|---|
| `shot_catalog` | Load and validate shot/keyframe metadata. | JSON file loaded directly by Flask. | Repository module with typed records and fixture-backed tests. |
| `retrieval` | Validate query, score/search candidates, group keyframes to shots, shape results. | Inline functions in `app.py`. | Pure scoring plus application use case. |
| `embedding` | Encode text/images with FG-CLIP or BEiT-3 and write immutable artifacts. | Exploratory notebook only. | Encoder adapters plus artifact manifest/checksum validation. |
| `vector_store` | Store/search vectors through a provider-neutral contract. | Not implemented. | Milvus and Qdrant adapters behind shared contract tests. |
| `benchmark` | Compare providers, index settings, and models reproducibly. | Planning doc only. | Harness with fixed manifests, query labels, raw run metadata, and reports. |
| `submission` | Prepare challenge answer payloads and audit accepted submissions. | `/api/submit` returns a local payload. | Submission use case with stable payload contract and audit record. |
| `frontend` | Operator UI for KIS/VKIS query construction and result review. | React prototype. | Keep API client thin; move reusable UI behaviors into hooks/components only when needed. |

## Request flow

### Current prototype search

1. UI sends `POST /api/search` with query, task, modalities, objects, colors, temporal cue, and minimum confidence.
2. Flask route parses the payload.
3. `score_shot` ranks each shot from `shots.json`.
4. Results are sorted by score and returned to the UI.

### Target vector-backed search

1. Route validates task, query/reference image, filters, and top-k.
2. Retrieval use case selects the configured model namespace.
3. Encoder adapter creates a query vector using the same model/checkpoint as the indexed artifact.
4. VectorStore adapter searches the selected provider and normalizes score semantics to **higher is better**.
5. Retrieval use case groups keyframes by `shot_id`, applies tie-breaking, and returns metadata.
6. Route maps the result to the public API response.

## Offline pipeline

1. Extract videos into deterministic shot/keyframe metadata.
2. Encode keyframes with a pinned model/checkpoint.
3. Validate vector dimension, finite values, dtype, and L2 norm.
4. Write an immutable artifact with manifest and checksum.
5. Ingest the same artifact into Milvus and Qdrant through the same contract.
6. Run exact-search correctness, ANN benchmark, resource measurement, and restart/backup drills.
7. Record the benchmark report and write an ADR before choosing a production provider.

Detailed vector-store design lives in `docs/technical/00-embedding-vector-store-evaluation.md`.

## Runtime boundaries

| Boundary | Pattern |
|---|---|
| HTTP API | Flask app factory; thin routes; typed app errors mapped to JSON. |
| UI | Vite React; one API base; local component state until reuse demands extraction. |
| Model inference | Adapter class with explicit `describe`, `encode_images`, and `encode_texts`. |
| Vector database | Provider-neutral port with Milvus/Qdrant implementations and shared contract tests. |
| Artifacts | Immutable manifest + checksum; no silent mutation after benchmark starts. |
| Secrets | Environment variables or local ignored `.env`; never notebook literals or committed text files. |

## Guardrails

- Do not mix embeddings from different models/checkpoints in one namespace.
- Do not compare raw similarity scores across models.
- Do not let provider SDK response shapes leak into retrieval/domain code.
- Do not benchmark database latency while inference time is included unless the report explicitly names it as end-to-end latency.
- Do not choose Milvus or Qdrant without approved dataset scale, SLO, hardware profile, relevance labels, and decision weights.
