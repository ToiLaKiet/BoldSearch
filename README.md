# BoldSearch

**A local video-frame retrieval workspace for HCM AI Challenge tasks, helping an operator find, inspect, and record KIS, VQA, TRAKE, and image-reference answers.**

BoldSearch pairs a React operator interface with a FastAPI retrieval API. It creates FG-CLIP text or image embeddings, searches a Zilliz/Milvus collection, enriches results with object-detection metadata, and serves the local keyframe corpus for visual verification.

> The in-app **Submit** action validates and accepts a payload locally. Official BTC delivery is a separate CSV-and-ZIP workflow; see the [submission guide](docs/knowledge/SUBMISSION_GUIDE.md).

---

## Capabilities

| Retrieval and review | Task workflow |
| --- | --- |
| **Text, object, and image-reference search**<br>Search frame embeddings from text queries, optional object hints, or an uploaded visual cue. | **KIS and VKIS**<br>Find a known target frame from text or an image reference. |
| **Staged temporal narrowing**<br>Use multiple text queries to scope later retrieval stages to the preceding result context. | **VQA**<br>Select a frame and record a free-text answer. |
| **Frame inspection**<br>Open returned keyframes, load per-video frame maps, and inspect nearby frames before choosing an answer. | **TRAKE**<br>Select multiple ordered frames from one video for temporal key events. |
| **Offline evaluation**<br>Evaluate exported rankings against task cards without starting FastAPI, FG-CLIP, or Milvus. | **Local media serving**<br>Keep the keyframe corpus outside the frontend bundle while serving it through FastAPI. |

## Competition knowledge

- [General submission guide](docs/knowledge/SUBMISSION_GUIDE.md)

---

## Quick start

### Prerequisites

| Requirement | Purpose |
| --- | --- |
| Python 3.12+ and [uv](https://docs.astral.sh/uv/) | Backend dependencies and commands |
| Node.js and npm | React development server |
| A Zilliz/Milvus collection compatible with the configured FG-CLIP encoder | Hybrid frame retrieval |
| Local keyframes, frame maps, and detection metadata | Result images, frame navigation, and object enrichment |

### 1. Prepare local runtime data and configuration

Keep the media corpus outside the Vite application bundle:

```text
data/
├── keyframes/        # local keyframe images
└── map-keyframes/    # per-video frame-map CSV files

app/backend/
└── detections.csv    # frame object-detection metadata
```

Create an ignored `app/backend/.env` file and provide the connection settings for your environment. At minimum, configure `ZILLIZ_URI`, `ZILLIZ_TOKEN`, and `MILVUS_COLLECTION`; use `OBJECTS_CSV_PATH`, `DATA_DIR`, or `FG_CLIP_DEVICE` only when their defaults do not fit your machine or file layout.

Do not commit credentials, local data, model artifacts, or generated evaluation output.

### 2. Start the API

```sh
cd app/backend
uv sync --locked --group dev
uv run python main.py
```

The API starts on `http://localhost:8000` by default. Confirm the process is responding before starting the UI:

```sh
curl http://localhost:8000/api/health
```

### 3. Start the operator UI

```sh
cd app/frontend
npm ci
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). During development, Vite proxies `/api`, `/keyframes`, and `/map-keyframes` to `http://localhost:8000` unless `VITE_API_URL` overrides the backend origin.

---

## How it works

1. The operator composes text queries, optional object hints, or an image cue in the React UI and chooses a task mode.
2. The UI calls `POST /api/search/query` for text/object retrieval or `POST /api/search/visual_query` for image-reference retrieval.
3. FastAPI validates the request. FG-CLIP encodes the query, then the search workflow executes hybrid retrieval against Zilliz/Milvus.
4. Multiple text queries narrow subsequent stages to the current frame context. Retrieved rows are enriched from the in-memory `detections.csv` index.
5. The API returns normalized frame results. FastAPI serves keyframes and frame-map CSVs from `data/` so the UI can display and inspect the evidence.
6. The UI sends selected KIS, VQA, or TRAKE answers to the local submission endpoints. Package official results separately as described in the [submission guide](docs/knowledge/SUBMISSION_GUIDE.md).

## API surface

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Confirm that the FastAPI process is responding. |
| `POST` | `/api/search/query` | Search from text queries and optional object hints. |
| `POST` | `/api/search/visual_query` | Search from an image cue or image embedding. |
| `POST` | `/api/search/submit/kis` | Validate a local KIS frame selection. |
| `POST` | `/api/search/submit/vqa` | Validate a local VQA frame-and-answer selection. |
| `POST` | `/api/search/submit/trake` | Validate local ordered TRAKE frame selections. |

The backend also serves `/keyframes` and `/map-keyframes` from the configured local data directories.

---

## Project structure

```text
.
├── app/
│   ├── backend/              # FastAPI API, FG-CLIP adapter, Milvus access, and evaluation runner
│   └── frontend/             # Vite React operator interface
├── docs/                     # architecture, diagrams, operational guides, and competition knowledge
├── data/                     # ignored local media and evaluation artifacts
├── GIT_CONVENTION.md         # branch and commit rules
├── PROGRESS.md               # implementation history and team handoff
├── BLOCKERS.md               # current blockers
└── README.md                 # project entry point
```

`app/backend/main.py` is the composition root. At startup it loads the detection index, opens the Milvus client, and optionally loads FG-CLIP once into application state. The `search` module owns the public HTTP contract, request orchestration, result shaping, and local submission validation.

For the current runtime boundaries and planned evolution, read [Architecture](docs/architecture/ARCHITECTURE.md). For backend data contracts and offline evaluation, see the [backend README](app/backend/README.md) and [evaluation README](app/backend/evaluation/README.md).

## Configuration

Backend settings read environment variables and `app/backend/.env`. Relative backend paths resolve from `app/backend`.

| Group | Settings | Purpose |
| --- | --- | --- |
| Retrieval service | `ZILLIZ_URI`, `ZILLIZ_TOKEN`, `MILVUS_COLLECTION` | Connect to the hybrid-search collection. |
| Model runtime | `LOAD_FG_CLIP_ON_STARTUP`, `FG_CLIP_DEVICE`, `HF_TOKEN` | Control the encoder lifecycle and device. |
| Local data | `OBJECTS_CSV_PATH`, `DATA_DIR`, `KEYFRAMES_DIR`, `KEYFRAME_MAP_DIR` | Locate detection metadata and served media. |
| API presentation | `HOST`, `PORT`, `API_PREFIX`, `FRAME_IMAGE_URL_TEMPLATE` | Configure the FastAPI process and result image URLs. |
| Frontend origin | `VITE_API_URL`, `VITE_STATIC_MEDIA_URL` | Configure the API proxy and production static-media origin. |

`ZILLIZ_TOKEN` and `HF_TOKEN` are secrets. Keep them server-side in the backend environment; values prefixed with `VITE_` are browser-visible and must not contain secrets.

---

## Development commands

| Command | Purpose |
| --- | --- |
| `cd app/backend && uv sync --locked --group dev` | Install locked backend and test dependencies. |
| `cd app/backend && uv run python main.py` | Run FastAPI with the configured host and port. |
| `cd app/backend && uv run pytest` | Run the backend test suite. |
| `cd app/backend && uv run pytest tests/evaluation -q` | Run focused offline evaluation tests. |
| `cd app/frontend && npm ci` | Install the locked frontend dependencies. |
| `cd app/frontend && npm test` | Run frontend unit tests. |
| `cd app/frontend && npm run build` | Build the frontend production bundle. |

## Verification

Run the narrowest check that proves your change. Before sharing a cross-stack change, run:

```sh
cd app/backend && uv run pytest
cd app/frontend && npm test && npm run build
```

For retrieval-quality changes, export rankings and use the offline evaluation runner rather than treating a successful API start as relevance evidence. The runner contract and command are documented in [`app/backend/evaluation/README.md`](app/backend/evaluation/README.md).

## Operational notes

- A Milvus collection must match the query encoder's model, checkpoint, preprocessing version, and embedding dimension.
- Translation is best-effort: a translation failure preserves the original text query rather than making search unavailable.
- Preserve both snake_case and compatibility camelCase frame fields in API responses; the frontend relies on both forms.
- The official corpus remains under ignored `data/`; do not copy it into Vite `public` or `dist`.
- Follow [GIT_CONVENTION.md](GIT_CONVENTION.md) for branches and commits. See [PROGRESS.md](PROGRESS.md) and [BLOCKERS.md](BLOCKERS.md) for the current team handoff state.
