# BoldSearch Agent Notes

## Structure

- `okf/` is the project knowledge bundle (OKF v0.2); read `okf/index.md` for curated knowledge with provenance before answering "why is it like this" questions.
- `app/backend` is a Python 3.12 FastAPI application; run its commands from that directory. `main.py` is the composition root: its lifespan loads the FG-CLIP encoder, Milvus client, and `detections.csv` index once into `app.state`.
- Keep the public search contract at `app/backend/search/router.py` and `schema.py`; keep Milvus SDK details inside `search`. Preserve both snake_case and compatibility camelCase frame fields in responses.
- `app/frontend` is a separate Vite React application. Text requests retain `KIS`, `VQA`, or `TRAKE`; image-reference requests use `VKIS`.

## Setup And Verification

```bash
cd app/backend && uv sync --locked --group dev
cd app/frontend && npm ci
```

- Backend: `cd app/backend && uv run python main.py` starts the API on the configured port (8000 by default). Full tests: `uv run pytest`; focused evaluation tests: `uv run pytest tests/evaluation -q`.
- Frontend: `cd app/frontend && npm test`; a single test file is `node --test src/taskMode.test.js`. Build with `npm run build`.
- No lint or typecheck command is configured. Do not invent one as a required gate; use the narrowest relevant test and build check.

## Runtime Data And Configuration

- Backend settings load environment variables and `app/backend/.env`, not a root `.env`. Relative backend settings resolve from `app/backend`.
- The ignored root `data/` directory holds `keyframes/`, `map-keyframes/`, and evaluation artifacts. Keep the keyframe corpus outside Vite `public` and `dist`; FastAPI serves it at `/keyframes` and `/map-keyframes`, and Vite proxies both paths in development.
- `VITE_API_URL` is the frontend proxy target; it may include `/api`, which the Vite config strips. The default target is `http://localhost:8000`.
- Use environment variables or ignored local `.env` files for credentials. Do not repeat the committed connection defaults in new code, and remove them before sharing or deploying the repository.

## Evaluation And Submission

- `evaluation.runner` is an offline gate: it evaluates exported rankings without starting FastAPI, FG-CLIP, or Milvus. It requires metadata and writes experiment artifacts under ignored `app/backend/evaluation/runs/` when paths there are chosen.
- The current runner and its inputs use `task_id`. The versioned `evaluation/cases/` templates use `case_id` and are intentionally incompatible until the runner migration lands; update the runner and its tests together.
- Synthetic and `vision-draft` cases are not approved relevance judgments and must not be used for model-selection claims.
- UI submit endpoints only accept local KIS/VQA/TRAKE payloads. Official BTC delivery is a separate manual ZIP containing a top-level `submission/` directory; see `docs/knowledge/SUBMISSION_GUIDE.md`.

## Git

- Follow `GIT_CONVENTION.md` at the repository root when committing: lowercase `<type>/<scope>` branches and Conventional Commit messages. If HEAD is `main` or `master`, create the branch before the first commit; never force-push a default branch.
- `.github/copilot-instructions.md` applies the same Conventional Commit rules to Copilot-generated commit messages.
- When resuming unfinished work, read `PROGRESS.md` and `BLOCKERS.md` at the repository root first.
