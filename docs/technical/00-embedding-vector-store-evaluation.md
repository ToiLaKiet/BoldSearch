# Embedding and vector-store evaluation plan

Status: **active plan; adapter baseline implemented, evaluation incomplete**

## Goal

Choose one image-text retrieval model using a fixed corpus, query set, and
qrels. Lock its artifact before comparing Qdrant and Milvus so provider results
are not confounded by different embeddings.

## Pipeline

```text
config/embedding.yaml
  -> select a supported encoder
  -> pull pinned local assets
  -> create normalized float32 embeddings
  -> write immutable vectors + manifest + checksum
  -> evaluate exact cosine ranking
  -> lock one model artifact
  -> ingest the same artifact into Qdrant and Milvus
  -> compare correctness, relevance, latency, and operations
  -> record the provider decision
```

`config/embedding.yaml` describes the selected model and its runtime inputs.
Asset transfer and versioning belong to DVC/R2; encoders receive local paths and
must not download mutable assets implicitly during a benchmark.

## Models in scope

| Key | Adapter | Dimension | Status |
|---|---|---:|---|
| `fg_clip_large` | `FGClipEncoder` | 768 | Supported from a pinned Hugging Face revision; migrate to one immutable local asset when its DVC layout is defined. |
| `beit3_base_itc` | `Beit3Encoder` | 768 | Supported; local checkpoint checksum is required before artifact generation. |
| `beit3_large_itc` | none | 1024 | Planned; requires its own adapter and verification. |

Adding another candidate requires an explicit configuration entry, adapter
contract coverage, and a real-model smoke test.

## Implemented baseline

Verified in the current tree:

- `FGClipEncoder` and `Beit3Encoder` expose normalized image/text encoding.
- YAML selection validates known encoder configuration without dynamic imports.
- `VectorStore` defines provider-neutral `search()` and single-batch `ingest()`.
- `QdrantStore` and `MilvusStore` implement the same contract.
- Shared contract tests validate ordering, neutral result shapes, replacement,
  empty ingest, and provider-specific fixtures.
- FastAPI lifespan creates and closes the configured store resource per worker.

Not implemented:

- immutable embedding artifact writer and checksum verification;
- approved corpus, qrels, quality metrics, SLO, and benchmark weights;
- production collection provisioning and index tuning;
- endpoint services that consume the store;
- offline GPU/multi-batch ingestion orchestration;
- final provider selection.

The runtime adapter intentionally does not provision collections. Milvus keeps
an explicit flush after each current single-batch ingest because the tested
default-consistency setup did not guarantee immediate visibility without it.
Revisit that trade-off only with a throughput/consistency benchmark.

## Gates

### Before model selection

1. Record selected checkpoint revisions and SHA-256 values.
2. Write, read, and verify immutable artifacts for each candidate.
3. Approve corpus, query set, and qrels.
4. Evaluate exact ranking with the same metrics.
5. Lock one model/checkpoint/artifact identity.

### Before provider selection

1. Provision equivalent Qdrant and Milvus collections outside the API process.
2. Ingest the same locked artifact into both providers.
3. Verify exact/small-fixture correctness before ANN performance tests.
4. Benchmark with declared hardware, concurrency, index settings, warm-up, and
   latency percentiles.
5. Score relevance, latency, resource cost, and operational complexity using
   approved weights.
6. Record the decision and rollback conditions in an ADR.

Until these gates pass, Qdrant and Milvus remain supported candidates rather
than a production winner.
