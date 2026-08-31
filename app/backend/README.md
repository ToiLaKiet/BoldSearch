# BoldSearcher Backend

FastAPI backend for frame retrieval over a large image/frame store.

The API is intentionally thin:

- `POST /api/search/query`: text and object-hint query. Encodes text with cached FG-CLIP, runs Zilliz/Milvus hybrid search, and enriches returned frames with detection metadata.
- `POST /api/search/visual_query`: image query. Encodes the provided image with the cached FG-CLIP model, then runs Zilliz/Milvus hybrid search.
- `POST /api/search/submit/kis`, `/vqa`, and `/trake`: validate local KIS, VQA, and TRAKE submission payloads.
- `GET /api/health`: lightweight service health check.

## Runtime Data Flow

1. Frontend sends the current query shape:
   - `query`
   - `queries`
   - `objects`
   - `objectQueries`
   - `minConfidence`
   - `topK`
2. Cached BEiT-3 creates dense 768-dimensional embeddings for text or image queries.
3. `search.service` calls `MilvusClient.hybrid_search`.
4. Milvus/Zilliz returns frame rows in `data`.
5. Backend looks up detection metadata for each returned frame in `detections.csv`.
6. The search workflow maps rows and metadata into frontend-compatible frame results.

## Configuration

Provide these values through environment variables or an ignored local `.env` file:

- `ZILLIZ_URI`
- `ZILLIZ_TOKEN`
- `MILVUS_COLLECTION`
- `OBJECTS_CSV_PATH`
- `FG_CLIP_DEVICE` optionally forces `cpu`, `cuda`, or `mps`

`main.py` initializes Zilliz/Milvus, the object CSV index, and BEiT-3 once during FastAPI lifespan startup, then stores them in `app.state` for request handlers.

`OBJECTS_CSV_PATH` can be absolute or relative to `app/backend`. The CSV must use this schema:

```csv
video_id,frame_id,object,quantity,bbox_x,bbox_y,bbox_w,bbox_h
L21_V01,001,car,1,110,123,23,100
L21_V01,001,dog,2,110,123,23,100
L21_V01,001,giraffe,3,110,123,23,100
```

## Local Data

The ignored repository-root `data/` directory owns local media and generated evaluation artifacts. `AppConfig` exposes `DATA_DIR`, `KEYFRAMES_DIR`, `KEYFRAME_MAP_DIR`, and `EVALUATION_ARTIFACT_DIR`; each can be overridden through the backend environment. FastAPI serves `/keyframes` and `/map-keyframes` from the configured directories, while the Vite development server proxies those paths to FastAPI.

## Query Evaluation

`evaluation/` provides an offline JSONL runner for comparing exported rankings before changing embedding strategy or re-indexing the corpus. It reports evidence-frame `Recall@K`, `MRR`, and `nDCG`; see `evaluation/README.md` for the test-case contract and command. Synthetic KIS/VQA/TRAKE templates live under `evaluation/cases/`. The manual GitHub workflow only reads task, ranking, and metadata files already present below `app/backend` in the checked-out branch.

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
