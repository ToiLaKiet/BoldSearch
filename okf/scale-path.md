---
type: Playbook
title: Data scaling path
description: Layered plan from the local data/ layout to object storage and CDN serving without changing the on-disk contract.
tags: [data, scaling, deployment]
generated: { by: opencode_agent/glm-5.3-flash, at: 2026-08-30T05:20:00Z }
status: draft
sources:
  - id: app-config
    resource: /app/backend/app_config.py
    title: AppConfig — FRAME_IMAGE_URL_TEMPLATE, VITE_STATIC_MEDIA_URL
    last_modified: 2026-08-30
  - id: architecture-doc
    resource: /docs/ARCHITECTURE.md
    title: BoldSearch Architecture
---

# Data scaling path

`data/` is the standard **local layout on each machine**, not the scaling destination. Scaling moves the *source of truth* outward while the on-disk contract stays fixed.

## Step 1 — Team scale: corpus sharing across machines

The problem is "where does a new machine get the data from", not "where do files live".

```text
Object storage (S3 / HF dataset / NAS)   ← source of truth
        ↓  bootstrap script (e.g. make data-download)
data/keyframes/, data/map-keyframes/     ← layout unchanged, code untouched
```

Git never holds binaries. One command gives every machine the same `data/` layout.

## Step 2 — Deployment scale: serving many users

FastAPI `StaticFiles` serving ~29 GB from one box is a development setup. At scale, keyframes move to object storage + CDN.

The migration seam already exists in configuration: `FRAME_IMAGE_URL_TEMPLATE` and `VITE_STATIC_MEDIA_URL` re-point frame image URLs without code changes.[^app-config]

## Step 3 — Metadata scale

- `detections.csv` (currently 168 KB, tracked): if it grows or becomes pipeline-regenerated, it is an *artifact* → move to `data/` (ignored) with the generator script.
- Vector search already lives outside the app (Zilliz/Milvus); not a scaling concern here.[^architecture-doc]

[^app-config]: AppConfig exposes `FRAME_IMAGE_URL_TEMPLATE` and `VITE_STATIC_MEDIA_URL` for exactly this re-pointing.
[^architecture-doc]: Architecture doc records Zilliz/Milvus as the external retrieval service.
