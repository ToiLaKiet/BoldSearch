# BoldSearch Architecture

Status: **current runtime and approved evolution**
Project: HCM AI Challenge Pipeline 2026 / BoldSearch

## Scope and evidence

This document describes the implementation in `app/` as of 2026-08-27 and the smallest approved direction for evolving it. It does not claim that planned adapters, benchmark tooling, or services already exist.

| Claim | Status | Evidence |
|---|---|---|
| The UI is a Vite React single-page application. | Verified | `app/frontend/src/App.jsx`, `app/frontend/package.json` |
| The API is a FastAPI process with an application lifespan. | Verified | `app/backend/main.py` |
| Search uses FG-CLIP query embeddings and Zilliz/Milvus hybrid search over visual and caption embedding fields. | Verified | `app/backend/encoders/fg_clip.py`, `app/backend/search/service.py` |
| Object metadata is loaded from `detections.csv` into memory at startup and enriches results after retrieval. | Verified | `app/backend/main.py`, `app/backend/search/object_index.py` |
| Submissions are local accepted payloads; exporting the official BTC package remains a manual workflow. | Verified | `app/backend/search/router.py`, `docs/SUBMISSION_GUIDE_R1.md` |
| A benchmark-driven second vector-store provider is not currently justified. | Decision | Current runtime has one provider and no shared provider contract tests. |

## Architectural decision

BoldSearch remains a **modular monolith**: one React UI, one FastAPI process, and managed retrieval infrastructure. Retrieval quality, repeatability, and operational simplicity are the current priorities; splitting the application into services would add deployment and debugging cost without solving a demonstrated constraint.

The `search` capability is the first-class backend boundary. Its public HTTP contract lives in `search/router.py` and `search/schema.py`; its workflow currently lives in `search/service.py`. The service is allowed to orchestrate the current single Milvus provider, but provider SDK types and response shapes must not spread beyond this capability.

## Current system view

Source diagram: `architecture/system-overview.mmd`; exported preview: [`architecture/system-overview.svg`](architecture/system-overview.svg).

```text
Challenge operator
  -> Vite React UI
  -> FastAPI /api/search routes
  -> search workflow
       -> FG-CLIP encoder
       -> Zilliz/Milvus hybrid search
       -> in-memory detection metadata
  -> normalized frame results and local submission payloads

FastAPI serves official keyframe images and frame-map CSV files from ignored root `data/`; Vite proxies them in development.
```

## Runtime responsibilities

| Boundary | Owns | Current implementation |
|---|---|---|
| Frontend | Query composition, task selection, result review, local submission state. | `app/frontend/src/App.jsx`, `taskMode.js` |
| HTTP edge | Pydantic validation, endpoint selection, and HTTP error mapping. | `search/router.py`, `search/schema.py` |
| Search workflow | Text/image query orchestration, staged temporal narrowing, object enrichment, and response shaping. | `search/service.py` |
| Model adapter | Long-lived FG-CLIP model state and normalized text/image embeddings. | `encoders/fg_clip.py` |
| Retrieval adapter | Zilliz/Milvus client lifecycle and hybrid search invocation. | `connections.py`, Milvus-specific code in `search/service.py` |
| Metadata store | CSV validation and frame-to-object lookup. | `search/object_index.py`, `detections.csv` |
| Static media | Official keyframes and per-video frame maps. | ignored `data/keyframes`, `data/map-keyframes`, served by FastAPI |

`main.py` is the composition root. Its FastAPI lifespan loads the object index, opens the Milvus client, and loads FG-CLIP once; these resources are stored in `app.state`. Shutdown explicitly closes the Milvus client; the in-memory object index and encoder are released with process teardown.

## Request flows

### Text and staged temporal retrieval

1. The UI sends `POST /api/search/query` with one or more text queries, task mode, object hints, and an optional previous result context.
2. The router validates the Pydantic request and delegates to `run_text_query`.
3. The search workflow translates text opportunistically, encodes it with FG-CLIP, and queries the Milvus visual and caption embedding fields with a weighted ranker.
4. For multiple queries, each subsequent query is scoped to the prior frame context through a Milvus expression.
5. Returned rows are enriched from the in-memory detection index, mapped to the public `SearchResponse`, and rendered with static keyframe URLs.

### Visual retrieval

1. The UI sends `POST /api/search/visual_query` with a base64 image or an embedding.
2. The workflow decodes and normalizes the image when an embedding is not already supplied.
3. FG-CLIP creates the image embedding; the same Milvus hybrid path returns and maps frames.

### Submission

1. The UI posts KIS, VQA, or TRAKE selections to `/api/search/submit/*`.
2. The API validates and returns a local accepted payload.
3. The operator transfers the result to the official BTC CSV/ZIP submission workflow described in `docs/SUBMISSION_GUIDE_R1.md`.

## Evolution plan

The next changes should keep external behavior stable while making the existing `search/service.py` easier to test and change.

1. Extract the Milvus request/response translation into one `search`-local adapter. This is justified now because provider SDK details and ranker configuration are mixed into the workflow.
2. Extract pure result mapping and temporal-context expression construction into focused modules with fixture-backed tests.
3. Keep the router and schemas unchanged as the public contract while the workflow is split internally.
4. Completed: keyframes and maps now live in ignored root `data/`, FastAPI serves them, and Vite proxies them during development. Production builds use `VITE_STATIC_MEDIA_URL` or the configured API origin rather than copying the corpus.
5. Add an embedding artifact manifest, corpus version, and query/relevance fixtures before evaluating a second vector database. Only then introduce a narrow provider contract and benchmark Milvus against another implementation.

No generic `VectorStore` port, microservice split, or frontend state library is warranted before those triggers occur.

## Operational guardrails

- A Milvus collection must contain embeddings from one model/checkpoint/preprocessing version and match the query encoder dimension.
- Preserve the response model while refactoring internals; the frontend currently relies on both snake_case and compatibility camelCase frame fields.
- Treat external translation as best-effort: an unavailable translator must not make retrieval unavailable.
- Store future credentials only in environment variables or ignored local `.env` files. Remove the current committed credential defaults before sharing or deploying the repository.
- Do not copy the keyframe corpus into application build artifacts; serve it through a dedicated static-media boundary.
- Add a narrow regression test whenever query mapping, temporal narrowing, object enrichment, or submission validation changes.
