# Architecture

Status: current implementation plus agreed near-term boundaries. Claims marked
**verified** are backed by the current tree; target items are not implemented.

## Current system

BoldSearch is a modular monolith: one FastAPI backend and one Vite React UI.

| Capability | Current state | Evidence |
|---|---|---|
| Search | Lexical, object, color, and temporal scoring over sample `shots.json`. | `app/backend/search/` |
| Embedding | Encoder adapters exist; HTTP routes are placeholders. | `app/backend/encoders/`, `app/backend/embedding/` |
| OCR, ASR, detection | HTTP contracts are placeholders. | corresponding backend feature packages |
| Vector store | One neutral `VectorStore` Protocol with Qdrant and Milvus adapters and shared contract tests. | `app/backend/vector_store/`, `app/backend/tests/contract/` |
| Vector-store lifecycle | One configured client/adapter is opened per FastAPI worker and stored on `app.state`; no endpoint consumes it yet. | `app/backend/main.py`, `app/backend/tests/test_main.py` |
| Frontend | React prototype using the Vite `/api` proxy. | `app/frontend/` |
| Benchmark/offline ingest | Not implemented. | technical plan only |

Source diagram: `architecture/system-overview.mmd`.

## Boundaries

```text
browser
  -> FastAPI router             external HTTP schema and translation
  -> application/pure logic     retrieval and result shaping
  -> provider-neutral contract  VectorStore / encoder behavior
  -> adapter                    Qdrant, Milvus, or model SDK
```

- Pydantic feature schemas are external HTTP contracts.
- Vector-store dataclasses are internal method inputs and outputs.
- `VectorStore` is a structural Protocol because consumers need behavior, not
  shared implementation or lifecycle hooks.
- Qdrant and Milvus adapters own reusable client and collection state.
- Pure SDK-to-neutral transformations stay as functions or private methods; a
  class is warranted only when it uses adapter state.
- Provider field names and SDK response objects must not cross the adapter
  boundary.

## Runtime ownership

FastAPI lifespan is the composition root for the current process:

1. Read the provider and collection from `AppConfig`.
2. Create the selected SDK client after application startup begins.
3. Wrap it in one `VectorStore` adapter and expose it through `app.state`.
4. Close the client during shutdown.

The provider branch belongs only in this composition root. Search and ingest
consumers should receive the neutral store and must not repeat the
Qdrant/Milvus selection.

The store is intentionally not split into search and ingest ports yet because
there is only one lifecycle and no real consumer requiring a narrower surface.
Revisit that decision if search and ingest move to separate processes or acquire
different permissions/scaling needs.

## Configuration ownership

Environment settings hold deployment-specific values such as service URLs.
Versioned YAML holds reviewed experiment/design decisions such as selected
provider, collection, encoder, and metric. `AppConfig` is the single merged
runtime settings object; provider-specific config subclasses are unnecessary
until their validation or consumers materially diverge.

## Current and target flows

Current search:

1. `POST /api/search/query` validates `SearchRequest`.
2. The router loads sample shots and calls pure scoring logic.
3. Ranked results are returned through the HTTP response schema.

Target vector-backed search:

1. A use case receives the shared `VectorStore` and selected encoder.
2. The encoder creates a vector from the locked model/checkpoint.
3. `VectorStore.search()` returns provider-neutral `SearchHit` values.
4. The use case groups keyframes, shapes results, and the router maps them to
   the public response schema.

Target ingestion remains deliberately small in the current baseline:

1. A consumer creates provider-neutral `VectorPoint` values.
2. `VectorStore.ingest()` writes one batch and makes it searchable.
3. Provisioning, GPU batching, artifact orchestration, and collection rebuilds
   remain outside the adapter until their workflows are designed.

## Guardrails

- Do not open provider connections at module import time.
- Do not let HTTP code import provider SDK types.
- Do not let the API create, recreate, or drop collections.
- Do not mix vectors from different model/checkpoint identities.
- Do not compare provider or model scores without normalized semantics and a
  reproducible benchmark.
- Do not add base classes, factories, loaders, or narrower ports until a current
  consumer needs the extra behavior.

Evaluation gates and unresolved model/provider decisions live in
`docs/technical/00-embedding-vector-store-evaluation.md`.
