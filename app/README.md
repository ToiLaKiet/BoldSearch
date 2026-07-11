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
    app.py
    requirements.txt
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
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Frontend:

```bash
cd app/frontend
npm install
npm run dev
```

Open http://localhost:5173. The Vite dev server proxies `/api` to Flask on http://127.0.0.1:5001.
