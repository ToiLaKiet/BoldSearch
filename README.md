# BoldSearch

BoldSearch is the HCM AI Challenge 2026 video-retrieval workspace. The current
prototype combines a FastAPI backend and a Vite React UI for Known Item Search
(KIS) and Visual Known Item Search (VKIS).

## Current state

- Lexical, object, color, and temporal search runs against sample `shots.json`.
- OCR, ASR, object detection, and embedding HTTP routes are placeholders.
- Qdrant and Milvus implement one provider-neutral `VectorStore` contract for
  search and single-batch ingest.
- FastAPI lifespan opens the configured vector-store client once per worker and
  closes it at shutdown. No endpoint consumes that store yet.
- Encoder evaluation, immutable embedding artifacts, vector-store benchmarking,
  and offline multi-batch ingest are still planned work.

The UI direction was inspired by
[VISIONE](https://github.com/aimh-lab/visione), but this repository owns its own
runtime and architecture decisions.

## Repository layout

```text
.
├── app/
│   ├── backend/       # FastAPI service, tests, encoders, vector-store adapters
│   └── frontend/      # Vite React prototype
├── architecture/      # Mermaid sources and exported diagrams
├── docs/
│   ├── ARCHITECTURE.md
│   ├── GIT_CONVENTION.md
│   └── technical/     # implementation plans and evaluation gates
├── AGENTS.md          # local engineering instructions
└── fg-clip.ipynb      # exploratory notebook
```

Documentation ownership:

- This README: repository overview and first run.
- `app/backend/README.md`: backend setup, configuration, API, and runtime notes.
- `docs/ARCHITECTURE.md`: verified current boundaries and agreed target flow.
- `docs/technical/*`: detailed plans, evidence, and unresolved evaluation gates.

Do not add another README unless a component has an independent setup or
release lifecycle that cannot be explained by one of these documents.

## Run locally

Backend:

```bash
cd app/backend
uv sync
cp .env.example .env  # optional; defaults target local services
uv run python main.py
```

Frontend, in another terminal:

```bash
cd app/frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Vite proxies `/api` to
`http://127.0.0.1:8000`; FastAPI exposes Swagger UI at
`http://localhost:8000/docs`.

Starting the backend also connects to the provider selected in
`app/backend/config/vector_store.yaml`. Provision the configured collection or
use the contract-test fixtures before exercising vector-store behavior.

## Development

Backend dependencies and checks:

```bash
cd app/backend
uv sync
uv run pytest
```

Frontend check:

```bash
cd app/frontend
npm run build
```

Architecture details live in `docs/ARCHITECTURE.md`; the active embedding and
vector-store evaluation gates live in
`docs/technical/00-embedding-vector-store-evaluation.md`.
