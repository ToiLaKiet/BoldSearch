# Handoff — Embedding pipeline

## Current state

- Branch: `feat/vectordb-integration`
- Latest commit: `43d216c feat(embedding): add BEiT-3 retrieval encoder`
- Implemented `FGClipEncoder.encode_images()` and `encode_texts()`.
- Model repository is pinned to revision `5a8f0f23b5a06dc92310e907599b2a0c2d58fe6f` because it uses `trust_remote_code=True`.
- Encoder output is converted to `float32` and L2-normalized.
- Runtime dependencies include `torchvision` and `einops`, which are imported by the pinned remote model code.
- `transformers` is constrained to `<5.0.0` because the pinned remote model code imports `transformers.onnx` from the 4.x API.
- Implemented `Beit3Encoder` with the official `BEiT3-base-itc` retrieval checkpoint, 224px preprocessing, and 768-dimensional image/text projections.
- `Beit3Encoder` receives local checkpoint and SentencePiece paths; S3/DVC owns asset fetch, versioning, and storage.

## Verification

- Python syntax compilation: passed with the local `.venv`.
- Backend tests: `20 passed` in `app/backend/tests/test_fg_clip_encoder.py`.
- Real CPU smoke test: passed on `053.jpg` and one text query; both outputs are `(1, 768)`, `float32`, and unit-norm within floating-point tolerance.
- Backend test suite after BEiT-3: `23 passed`.
- Real BEiT-3 CPU smoke test with local model assets: passed on `053.jpg` and one text query; both outputs are `(1, 768)`, `float32`, and unit-norm within floating-point tolerance.
- Staged diff/secret scan: passed before the previous commit; no new secret was added in this continuation.

## Next steps

1. Record the provisional BEiT-3 model choice and checkpoint checksum in the technical spec before artifact generation.
2. After both encoders exist, extract shared L2 normalization to `encoders/normalization.py` and add YAML selection using one value such as `embedding.encoder: fg_clip`; do not use multiple booleans.
3. Add artifact writing, then attach offline evaluation/MLflow at the end of the model phase.

## Continuation status

- FG-CLIP runtime gap is fixed in `app/backend/requirements.txt`.
- BEiT-3 is implemented provisionally as `beit3_base_itc_patch16_224` with dimension `768`; review the model asset revision/checksum before committing.

## Precision policy

- Persisted/output embeddings: always normalized `float32`.
- CPU and MPS: start with model/default `float32` until real smoke tests pass.
- CUDA: benchmark FP16/BF16 later; represent compute dtype as runtime config, not separate adapter classes.
- Do not add mixed precision to the current FG-CLIP commit without real-model tests.

## Scope guard

- Do not refactor Flask/FastAPI yet.
- Do not add Milvus/Qdrant until the model baseline and artifact contract are stable.
- Keep classes only for reusable state; use free functions for artifact transforms and evaluation.
