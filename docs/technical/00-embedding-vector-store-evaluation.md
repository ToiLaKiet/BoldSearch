# Embedding and Vector Retrieval Evolution

Status: **current runtime constraints and future evaluation plan**

## Current goal

Keep the deployed FG-CLIP-to-Milvus retrieval path compatible while creating the evidence needed to evaluate a model or provider change. The current worktree implements online retrieval through Zilliz/Milvus and includes a JSONL evidence-ranking runner in `app/backend/evaluation/`; it does not yet implement a reproducible embedding-artifact pipeline or a full benchmark harness.

## Exact flow

```text
current runtime:
FG-CLIP query embedding -> Zilliz/Milvus collection -> hybrid result mapping

required evaluation path:
versioned corpus + pinned encoder -> normalized embeddings -> artifact manifest/checksum
  -> exact retrieval evaluation with approved qrels -> benchmark report
  -> model or provider decision
```

The current runtime loads FG-CLIP from its pinned Hugging Face revision. A future artifact pipeline must own corpus version, preprocessing, model revision, dimension, and checksum together. Candidate models remain planning work until they have a dedicated adapter, contract tests, and a real-model smoke test.

## Models in scope

| Key | Status | Adapter | Dimension | Notes |
| --- | --- | --- | --- | --- |
| `fg_clip2_large` | `FGClipEncoder` | 1024 | Implemented online query encoder; its model revision and collection embedding fields must stay compatible. |
| `beit3_base_itc` | none in the current tree | 768 | Candidate only; requires an adapter, local checkpoint provenance, and artifacts before evaluation. |
| `beit3_large_itc` | none | 1024 | Candidate only; requires a separate adapter and verification. |

Do not treat this as a list of every model in the ecosystem. It is the bounded
set we are willing to evaluate in this project. Adding a model starts with an
explicit YAML candidate, followed by its adapter and tests.

## Current implementation boundary

`FGClipEncoder` is the implemented model-runtime adapter. Its public behavior is `encode_images()` and `encode_texts()`, each returning L2-normalized `float32` vectors. `search/service.py` currently combines query orchestration and the Milvus request construction; extract a search-local Milvus adapter before adding another provider.

FG-CLIP2 returns 1024-dimensional embeddings. A collection must use the same model revision, dimension, normalization, and preprocessing as the query encoder; a mismatch requires re-indexing rather than a runtime fallback.

When an artifact store takes ownership of model assets, FG-CLIP should follow the same local-asset rule: retrieve one immutable model directory, pass it directly to the encoder, and record its identity in the artifact manifest. Do not add storage paths until the artifact layout and retrieval command are defined.

No base encoder class, factory, or generic vector-store port is needed yet. If an artifact writer/evaluator needs one shared type, add a small `Protocol` for the two encode methods and run the same behavior tests for all implementations.

## Gates before changing the vector database

1. Record the corpus, preprocessing, checkpoint revision, and SHA-256 for every evaluated artifact.
2. Write, read, and verify immutable embedding artifacts.
3. Evaluate exact cosine ranking against approved qrels.
4. Select a model baseline using quality metrics before latency or provider preference.
5. Capture dataset scale, SLO, hardware profile, relevance labels, and benchmark weights.
6. Define a narrow provider contract only after the second provider is being implemented against the locked artifact.

Adding a second provider remains deferred: without a stable artifact, dataset scale, qrels, and latency target, provider comparison would be premature. The existing Milvus integration remains the production path until that evidence exists.
