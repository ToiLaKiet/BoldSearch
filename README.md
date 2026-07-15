# BoldSearch — HCM AI Challenge Pipeline 2026

BoldSearch is the working repository for an HCM AI Challenge 2026 video retrieval pipeline. The current app is a prototype named **BoldSearcher**: a Flask API and Vite React UI for Known Item Search (KIS) and Visual Known Item Search (VKIS) over sample shot metadata.

## Current baseline

Verified on 2026-07-11:

- `app/backend/app.py` serves `/api/health`, `/api/tasks`, `/api/shots`, `/api/search`, and `/api/submit` from `app/backend/data/shots.json`.
- `app/frontend` is a Vite + React single-page UI that calls the Flask API through a Vite `/api` proxy.
- Project-wide architecture is documented in `docs/ARCHITECTURE.md`; planned embedding/vector-store work is documented in `docs/technical/00-embedding-vector-store-evaluation.md` and `architecture/embedding-vector-pipeline.mmd`.
- No production vector database, benchmark harness, or provider adapter is implemented yet.

## Repository layout

```text
.
├── AGENTS.md                         # local coding/agent conventions
├── README.md                         # project overview and quickstart
├── app/
│   ├── README.md                     # app-specific run instructions
│   ├── backend/                      # Flask API prototype
│   └── frontend/                     # Vite React prototype
├── architecture/                     # diagrams and system views
├── docs/
│   ├── CODE_PATTERN.md               # module boundaries and coding style
│   ├── GIT_CONVENTION.md             # branch, commit, PR, and release rules
│   └── technical/                    # technical design records
└── fg-clip.ipynb                     # exploratory notebook; move secrets to env before reuse
```

## Quickstart

Backend:

```bash
cd app/backend
uv sync                       # creates .venv from uv.lock
cp .env.example .env          # optional: defaults work for local dev
uv run python main.py
```

Frontend:

```bash
cd app/frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The Vite dev server proxies `/api` to the FastAPI backend on `http://127.0.0.1:8000`.

## Architecture direction

Use a modular monolith until benchmark data proves another deployment shape is needed.

```text
browser UI
  -> API routes
  -> application use cases
  -> domain scoring/retrieval policies
  -> repositories/adapters
  -> local data, embedding artifacts, or selected vector store
```

Near-term modules:

- `shot_catalog`: shot/keyframe metadata loading and validation.
- `retrieval`: query validation, scoring, grouping, and result shaping.
- `embedding`: FG-CLIP/BEiT-3 encoder adapters and immutable embedding artifacts.
- `vector_store`: Milvus/Qdrant provider adapters behind one contract.
- `benchmark`: reproducible experiment runner and report generation.
- `submission`: competition answer payload preparation and audit trail.

See `docs/ARCHITECTURE.md` and `architecture/system-overview.mmd` for the intended module flow.

## Development guardrails

- Keep the prototype simple: add a boundary only when the current change needs it.
- Keep HTTP/UI code at the edges; keep ranking and validation rules in pure functions where possible.
- Do not commit credentials, dataset dumps, model weights, generated caches, or benchmark artifacts.
- Treat `docs/technical/00-embedding-vector-store-evaluation.md` as planning guidance until BEiT-3 checkpoint, dataset scale, SLO, and benchmark decision weights are approved.

## Verification commands

Use the narrowest check that proves your change:

```bash
python3 -m py_compile app/backend/app.py
cd app/frontend && npm run build
```

When tests are added, keep the default local gate in a `make verify` or equivalent script and update this README.
