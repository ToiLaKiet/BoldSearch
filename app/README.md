# BoldSearcher

BoldSearcher is a React + Flask prototype for querying video shots. The UI is inspired by VISIONE's interactive video retrieval ideas: text search, visual cues, temporal hints, object/color filters, keyframe browsing, and shot submission.

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
