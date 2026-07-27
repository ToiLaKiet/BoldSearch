# BoldSearcher Backend

FastAPI backend for frame retrieval over a large image/frame store.

The API is intentionally thin:

- `POST /api/search/query`: text + object-count query. Encodes the text with the cached FG-CLIP model, runs Zilliz/Milvus hybrid search, then enriches, filters, and reranks returned frames with `objects.csv`.
- `POST /api/search/visual_query`: image query. Encodes the provided image with the cached FG-CLIP model, then runs Zilliz/Milvus hybrid search.
- `POST /api/search/submit`: accepts the selected frame/shot/video identifiers.
- `GET /api/health`: lightweight service health check.

## Runtime Data Flow

1. Frontend sends the current query shape:
   - `query`
   - `queries`
   - `objects`
   - `objectQueries`
   - `minConfidence`
   - `topK`
2. Cached FG-CLIP creates dense embeddings for text or image queries.
3. `search.service` calls `MilvusClient.hybrid_search`.
4. Milvus/Zilliz returns frame rows in `data`.
5. Backend looks up object metadata for each returned frame in `objects.csv`.
6. If object queries exist, backend removes frames that do not contain every queried object label.
7. Matching frames are reranked by how close their detected object counts are to the requested counts, with Milvus score used as the tie-breaker.
8. Backend returns frontend-compatible frame results.

## Configuration

Copy `.env.example` to `.env` and fill:

- `ZILLIZ_URI`
- `ZILLIZ_TOKEN`
- `MILVUS_COLLECTION`
- `OBJECTS_CSV_PATH`
- `FG_CLIP_DEVICE` optionally forces `cpu`, `cuda`, or `mps`

`main.py` initializes Zilliz/Milvus, the object CSV index, and FG-CLIP once during FastAPI lifespan startup, then stores them in `app.state` for request handlers.

`OBJECTS_CSV_PATH` can be absolute or relative to `app/backend`. The CSV must use this schema:

```csv
video_id,frame_id,object,quantity,bbox_x,bbox_y,bbox_w,bbox_h
L21_V01,001,car,1,110,123,23,100
L21_V01,001,dog,2,110,123,23,100
L21_V01,001,giraffe,3,110,123,23,100
```

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
