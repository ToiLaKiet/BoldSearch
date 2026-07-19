# BoldSearcher

BoldSearcher is a React + FastAPI prototype for querying video shots. The UI is inspired by VISIONE's interactive video retrieval ideas: text search, visual cues, temporal hints, object/color filters, keyframe browsing, and shot submission.

Reference: https://github.com/aimh-lab/visione

## Tasks

- `KIS`: Known Item Search. Use when the user knows the target shot and describes it mostly with text.
- `VKIS`: Visual Known Item Search. Use when the user knows the target shot and relies on visual cues such as objects, colors, layout, or an uploaded reference image.

## Structure

```text
app/
  backend/
    main.py
    app_config.py
    asr/
    data/shots.json
  frontend/
    index.html
    package.json
    vite.config.js
    src/
      App.jsx
      main.jsx
      styles.css
```

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

## ASR

`POST /api/asr/transcribe` nhận `video_id` cùng danh sách keyframe và trả ASR
segments cùng các keyframe đã gắn `text` theo timestamp.

`main.py` load ChunkFormer khi startup nhưng chưa cấu hình production media
resolver. Nếu `app.state.asr_media_resolver` chưa được inject, endpoint trả
`503 MEDIA_RESOLVER_UNAVAILABLE`.

Contract request/response, validation, errors và bằng chứng HTTP nằm tại
[`docs/technical/01-asr-keyframe-transcript-plan.md`](../docs/technical/01-asr-keyframe-transcript-plan.md).
