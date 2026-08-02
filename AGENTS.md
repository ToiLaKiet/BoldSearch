# Repository Guidelines

## Project Structure & Module Organization

BoldSearch is a modular monolith with a FastAPI backend and a Vite/React UI. Backend code lives in `app/backend/`; feature packages such as `search/`, `embedding/`, `asr/`, `ocr/`, and `object_detection/` keep schemas, routers, and services together. Encoder implementations are in `app/backend/encoders/`, configuration is in `app/backend/config/`, and backend tests are in `app/backend/tests/`. Frontend code and styles live in `app/frontend/src/`. Use `docs/` for architecture and technical records, `architecture/` for diagrams, and keep exploratory work in notebooks such as `fg-clip.ipynb`.

## Build, Test, and Development Commands

- `cd app/backend && uv sync` installs Python 3.13 dependencies from `uv.lock`, including the development group.
- `cd app/backend && uv run python main.py` starts the API on port 8000.
- `cd app/backend && uv run pytest` runs the backend test suite.
- `cd app/frontend && npm install` installs the locked frontend dependencies.
- `cd app/frontend && npm run dev` starts Vite on port 5173 and proxies `/api` to the backend.
- `cd app/frontend && npm run build` creates the production bundle; use it as the minimum frontend verification.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions/modules, `PascalCase` classes, type hints, and Pydantic models in Python. Keep routers thin and put ranking, validation, and transformations in testable pure functions. In JSX, follow the existing two-space indentation, `PascalCase` components, `camelCase` state/handlers, single quotes, and semicolons. No formatter or linter is currently configured, so match adjacent code and avoid unrelated reformatting.

## Testing Guidelines

Pytest discovers `app/backend/tests/test_*.py`; name test functions `test_<behavior>`. Add regression tests for bug fixes and focused tests for validation, scoring, configuration, and API contracts. Run the narrowest relevant test first, then the full suite. No coverage threshold or frontend test runner is configured; document any verification gap in the pull request.

## Commit & Pull Request Guidelines

Follow Conventional Commits used in recent history: `feat(retrieval): add shot grouping`, `fix(backend): reject empty queries`, or `docs(architecture): update request flow`. Branches use lowercase `<type>/<short-scope>`, such as `feature/retrieval-service` or `fix/search-score-ties`. Pull requests should explain the change and rationale, link relevant issues, list verification commands, note risks or configuration changes, and include screenshots for visible UI work.

## Security & Configuration

Copy the relevant `.env.example` locally. Never commit credentials, dataset dumps, model weights, generated caches, or benchmark artifacts. Configure the frontend backend target with `VITE_API_URL` when the default `http://localhost:8000` is unsuitable.
