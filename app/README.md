# BoldSearcher

BoldSearcher is a React + FastAPI video-retrieval application. It uses FG-CLIP query embeddings with Zilliz/Milvus hybrid retrieval, enriches frames from `detections.csv`, and displays official keyframes through FastAPI static-media routes.

Reference: https://github.com/aimh-lab/visione

## Tasks

- `KIS`: Known Item Search. Use when the user knows the target shot and describes it mostly with text.
- `VKIS`: Visual Known Item Search. The UI uses this task for uploaded image-reference searches.
- `VQA`: Visual Question Answering. Submit one frame and its free-text answer.
- `TRAKE`: Temporal Retrieval and Key-event. Select and submit one or more ordered frames per video.

## Structure

```text
app/
  backend/
    main.py
    app_config.py
    connections.py
    encoders/                 # FG-CLIP runtime adapter
    search/                   # schemas, routes, retrieval, object metadata
    detections.csv
  frontend/
    index.html
    package.json
    vite.config.js
    src/
      App.jsx
      main.jsx
      styles.css
```

## Runtime flow

```text
React UI -> /api/search/query or /api/search/visual_query -> FastAPI router
  -> FG-CLIP embedding -> Zilliz/Milvus hybrid search -> object metadata enrichment
  -> normalized frame results -> keyframe image resolved from FastAPI-served frame-map CSV
```

`main.py` loads the detection index, Milvus client, and FG-CLIP encoder once during FastAPI startup. The backend returns local accepted submission payloads only; create the BTC CSV/ZIP package separately as described in `../docs/knowledge/SUBMISSION_GUIDE.md`.

## Static media

`../data/keyframes` and `../data/map-keyframes` contain the official assets for browser image resolution. FastAPI serves them at `/keyframes` and `/map-keyframes`; Vite proxies those paths during development. Production uses `VITE_STATIC_MEDIA_URL` when configured, otherwise the origin from `VITE_API_URL`. The corpus is outside Vite `public`, so production builds no longer copy it into `dist/`.

## Run

Backend:

```bash
cd app/backend
uv sync                       # creates .venv from uv.lock
uv run python main.py
```

Frontend:

```bash
cd app/frontend
npm install
npm run dev
```

Open http://localhost:5173. The Vite dev server proxies `/api` to the FastAPI backend on http://127.0.0.1:8000.
