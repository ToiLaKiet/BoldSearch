# BoldSearch backend

FastAPI backend for shot retrieval and the in-progress multimodal pipeline.

## Setup

```bash
uv sync
cp .env.example .env  # optional
uv run python main.py
```

- API: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- Tests: `uv run pytest`

Use `uv add <package>` to change dependencies. `pyproject.toml` is the declared
dependency set and `uv.lock` is the generated lock file.

## Module ownership

```text
backend/
├── main.py               # FastAPI assembly and process lifecycle
├── app_config.py         # merge environment settings with YAML decisions
├── search/               # working sample-data search and submission routes
├── embedding/            # placeholder HTTP contracts
├── ocr/                  # placeholder HTTP contracts
├── asr/                  # placeholder HTTP contracts
├── object_detection/     # placeholder HTTP contracts
├── encoders/             # model-runtime adapters and selection
├── vector_store/         # neutral schemas, contract, Qdrant/Milvus adapters
├── config/               # reviewed model and vector-store decisions
├── data/                 # sample shot catalogue
└── tests/                # unit, lifecycle, and provider contract tests
```

For HTTP features, `schema.py` owns request/response contracts, `router.py` owns
HTTP translation, and `service.py` owns reusable application or pure business
logic when that logic exists. Do not create empty layers just to match this
shape.

Infrastructure adapters such as `encoders/` and `vector_store/` must not import
FastAPI. Their models describe internal method inputs/outputs; feature schemas
describe the external HTTP contract.

## Configuration

`AppConfig` is the runtime settings object. It merges two sources:

| Source | Owns |
|---|---|
| Environment or `.env` | machine/deployment values such as host, port, Qdrant URL, and Milvus URI |
| `config/*.yaml` | reviewed project decisions such as provider, collection, model, and metric |

Provider URLs must not be added to YAML. Vector metric remains a provisioning
and benchmark decision until runtime code has a real consumer for it.

## Vector-store lifecycle

`main.lifespan` selects the configured provider, creates one client and one
`VectorStore` adapter per FastAPI worker, assigns it to
`app.state.vector_store`, and closes the client at shutdown.

The contract intentionally contains only current behavior:

```python
class VectorStore(Protocol):
    def search(self, vector: Sequence[float], limit: int) -> list[SearchHit]: ...
    def ingest(self, points: Sequence[VectorPoint]) -> None: ...
```

Both Qdrant and Milvus implement this contract. Provider SDK response shapes
are normalized to the schemas in `vector_store/schemas.py`; consumers should
only see `VectorPoint`, `SearchHit`, and their neutral fields.

Current boundaries:

- The shared store is initialized but no endpoint calls `search()` or
  `ingest()` yet.
- Collections must already exist. Runtime code does not create, recreate, or
  drop them.
- Ingest is one application batch, not a GPU/multi-batch orchestration layer.
- Milvus calls `flush()` after ingest so the next request observes the write
  under the tested default-consistency baseline.
- Search/ingest-specific ports should only be split when separate consumers
  actually need narrower dependencies.

Provider behavior is checked through the same contract suite in
`tests/contract/test_vector_store.py`; lifespan ownership and cleanup are checked
in `tests/test_main.py`.

## Routes

| Method | Path | State |
|---|---|---|
| `GET` | `/api/health` | working |
| `GET` | `/api/search/tasks` | working |
| `GET` | `/api/search/shots` | working |
| `POST` | `/api/search/query` | working against sample JSON |
| `POST` | `/api/search/submit` | working locally |
| `POST` | `/api/ocr/extract` | placeholder |
| `POST` | `/api/asr/transcribe` | placeholder |
| `POST` | `/api/object-detection/detect` | placeholder |
| `POST` | `/api/embedding/encode-image` | placeholder |
| `POST` | `/api/embedding/encode-text` | placeholder |
| `POST` | `/api/embedding/search` | placeholder |
| `POST` | `/api/embedding/index` | placeholder |

When a placeholder becomes real, construct long-lived model/database resources
in lifespan and pass them to the consumer. Keep request validation and response
mapping in the router; keep provider-specific translation inside its adapter.
