---
type: Convention
title: Data layout contract
description: Where runtime data lives, what is tracked vs ignored, and why the corpus stays outside git.
tags: [data, storage, convention]
generated: { by: opencode_agent/glm-5.3-flash, at: 2026-08-30T05:20:00Z }
verified:
  - { by: human:miphu, at: 2026-08-30T05:20:00Z }
status: stable
sources:
  - id: agents-md
    resource: /AGENTS.md
    title: BoldSearch Agent Notes
    last_modified: 2026-08-30
---

# Data layout contract

The boundary is: **per-machine corpus and generated artifacts → ignored `data/`; small load-bearing metadata → tracked, next to the code that consumes it.**

| Location | Tracked | Content | Consumer |
| --- | --- | --- | --- |
| Served corpus: `data/keyframes/`, `data/map-keyframes/` | No (ignored) | Per-machine corpus, ~29 GB | FastAPI static mounts at `/keyframes`, `/map-keyframes` |
| Raw downloads: `data/aic2026-downloads/` | No | Unextracted BTC packages, as received | Manual extraction source |
| Round working dirs: `data/aic2026-<round>/` | No | Submission ZIP, `results/`, submission CSVs, `picks.json`; pipelines retire after each round (sources recoverable in git history) | Manual submission workflow |
| Derived artifacts: `data/{frames,metadata,vectors}/<video_id>/`, `data/evaluation-artifacts/`, `data/csv/` | No | Per-video extraction/embedding outputs, evaluation runs; scratch | Local pipelines |
| `app/backend/detections.csv` | Yes | Object-detection metadata, 168 KB | Backend via `OBJECTS_CSV_PATH` |
| `app/frontend/public/Frames.csv` | Yes | Frame lookup table, 1.7 MB | Browser `fetch('/Frames.csv')` |

## Rules

1. Never move the corpus into Vite `public/` or `dist`; it must not enter the frontend bundle or git.[^agents-md]
2. Never track corpus binaries or generated artifacts in git; `data/` is the landing layout on every machine.
3. Small metadata CSVs stay tracked so a fresh clone runs immediately; moving them into ignored `data/` would break clone-and-run for every teammate.
4. Scripts must not live in `data/` long-term: git ignores it, so submission/pipeline scripts there are invisible to teammates and history. Move them into the tracked tree (e.g. `scripts/`) once they are load-bearing.

## Why not "unify" everything into `data/`

Moving tracked CSVs into ignored `data/` forces per-machine manual placement plus config churn (`OBJECTS_CSV_PATH` default, gitignore exceptions) for zero benefit. `Frames.csv` must stay browser-reachable, and Vite `public/` is the native mechanism that works in both dev and production builds.

## Scaling

For team scale and deployment, see [Data scaling path](/scale-path.md): `data/` remains the local layout while the source of truth moves to object storage behind a bootstrap script.

[^agents-md]: BoldSearch Agent Notes — runtime data and configuration section.
