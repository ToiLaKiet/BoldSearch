# Embedding pipeline plan

Status: **active implementation plan**

## Current goal

Choose one image-text retrieval model using the same corpus, query set and qrels.
The current branch does not implement online retrieval or a vector database.

## Exact flow

```text
config/embedding.yaml
  -> select one supported encoder
  -> DVC pulls its pinned assets from R2 to local paths
  -> encoder creates normalized float32 embeddings
  -> artifact writer stores vectors + manifest + checksum
  -> exact cosine evaluation produces a JSON report
  -> choose the model baseline
  -> only then benchmark Milvus and Qdrant with that locked artifact
```

`config/embedding.yaml` contains only runnable choices and their runtime inputs.
DVC/R2 own file transfer and versioning; encoders only receive already-local
asset paths. Candidate models stay in this plan until they receive a dedicated
adapter, contract tests and a real-model smoke test.

## Models in scope

| Key | Status | Adapter | Dimension | Notes |
| --- | --- | --- | --- | --- |
| `fg_clip_large` | `FGClipEncoder` | 768 | Supported; Hugging Face revision is pinned in the adapter. |
| `beit3_base_itc` | `Beit3Encoder` | 768 | Supported; local DVC/R2 checkpoint checksum is required before artifact generation. |
| `beit3_large_itc` | none yet | 1024 | Planned; requires a separate large-model adapter and verification. |

Do not treat this as a list of every model in the ecosystem. It is the bounded
set we are willing to evaluate in this project. Adding a model starts with an
explicit YAML candidate, followed by its adapter and tests.

## Current implementation boundary

`FGClipEncoder` and `Beit3Encoder` are the model-runtime adapters. Their public
behavior is only `encode_images()` and `encode_texts()`, each returning
L2-normalized `float32` vectors. The YAML loader uses Pydantic to validate the
selected runtime inputs, then a small static bootstrap selects its known adapter;
it does not dynamically import Python classes.

No base encoder class, factory, Flask integration, or vector-store adapter is
needed yet. If the artifact writer/evaluator needs one shared type, add a small
`Protocol` for those two encode methods and run the same behavior tests for all
implementations.

## Gates before vector DB

1. Record each selected checkpoint revision and SHA-256.
2. Write/read/verify separate immutable artifacts for FG-CLIP and BEiT-3.
3. Evaluate exact cosine ranking against approved qrels.
4. Select the model baseline using quality metrics first.
5. Define a `VectorStore` contract, then implement Milvus and Qdrant in
   separate commits against the locked artifact.

Vector DB work is intentionally deferred: without a stable model, artifact,
dataset scale, qrels and latency target, index configuration and provider
comparison would be premature.
