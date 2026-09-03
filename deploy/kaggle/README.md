# Kaggle deploy pipeline

Deploys the BoldSearch dev environment to Kaggle from GitHub Actions and
verifies it with a batch run. Batch sessions are ephemeral; a long-lived dev
session still needs a human to open the notebook and Run All (Kaggle has no
API for interactive sessions).

## Components

| File | Runs where | Purpose |
|---|---|---|
| `kaggle_wrapper.ipynb` | Kaggle | The only runtime piece, one concern per cell: hardware → input datasets → git pull → `.env` from dataset (+3 dynamic keys) → `AppConfig` (pydantic) validates → BE + FE + smoke → public tunnel, session stays alive |
| `kernel-metadata.json` | CI | Kernel descriptor; placeholders filled by CI (`sed`) |
| `dataset-metadata.json` | CI | Private env-dataset descriptor; CI pushes `.env` verbatim via plain `kaggle datasets create/version` CLI |
| `deploy-kaggle.yml` | CI | Whole pipeline: publish env dataset → `kernels push` → wait for live session → `deploy-report.md` |

No Python modules here on purpose. The runtime env is the repo's own
`app/backend/.env` stored verbatim as the `KAGGLE_RUNTIME_ENV` Actions secret —
CI never re-validates it; `AppConfig` (pydantic) validates on Kaggle at import.
The wrapper just copies it to `backend/.env`, so the secret must already carry
the Kaggle paths, e.g.:

```
KEYFRAMES_DIR=/kaggle/input/<keyframes-dataset-slug>/keyframes
FRAME_IMAGE_URL_TEMPLATE=/keyframes/{video_id}{frame_id}.png
```

`KEYFRAME_MAP_DIR` is optional (the backend mounts it with `check_dir=False`).
Keep the dataset slug in `KEYFRAMES_DIR` in sync with what is attached to the
kernel Input panel.

Config lives in `app_config.AppConfig.DEPLOY_KAGGLE` (typed, pydantic-validated;
override via `DEPLOY_KAGGLE__*` env keys). CI overrides the repo default
`REPO_URL` with the GitHub context. Deploys always track `main`.

## Semantics

Deploy keeps the Kaggle session **alive**: after smoke passes, the notebook
publishes a cloudflared URL (UI + `/api` + keyframes through one origin),
prints `TUNNEL_URL=...`, and blocks. CI waits for that marker, records the URL
in the report, and exits — the session keeps running until stopped from the
Kaggle UI or replaced by the next deploy. This consumes GPU quota for the
whole session (~30 GPU h/week budget).

## Prerequisites (one-time)

1. Phone-verified Kaggle account (internet + GPU).
2. GitHub secrets:
   - `KAGGLE_USERNAME`, `KAGGLE_KEY` — Kaggle API auth for CI.
   - `KAGGLE_RUNTIME_ENV` — the verbatim content of `app/backend/.env`
     plus a `GH_PAT=...` line (fine-grained, `contents: read`) for the
     private clone. Nothing else is curated; `AppConfig` validates on Kaggle.
3. Rotated credentials (security pass) — no real secret may remain committed.

## Run

Actions → **Deploy Kaggle** → Run workflow. The job uploads the
`kaggle-deploy-report` artifact (markdown table: timestamp, kernel, pinned
SHA, push/kernel status, smoke, **public URL**, duration) and appends it to
the job summary.

## Teardown

Stop the kernel from the Kaggle UI (kernel page → running version → stop), or
simply run the next deploy. There is no Kaggle API to stop a run.

## Triage

- `timeout` after 60 min: check the kernel log artifact, likely GPU queue or model load.
- `error` early: quota exhausted (~30 GPU h/week), internet toggle, or phone verification.
- Push 429: Kaggle rate limit; rerun the dispatch (concurrency group serializes runs).
