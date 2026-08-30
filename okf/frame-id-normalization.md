---
type: Risk
title: Frame ID format mismatch between detections.csv and Frames.csv
description: detections.csv uses L21_V01-style video ids while Frames.csv uses L21_V001; impact on the object-enrichment join is unverified.
tags: [data, risk, ids]
generated: { by: opencode_agent/glm-5.3-flash, at: 2026-08-30T05:20:00Z }
status: draft
sources:
  - id: detections-csv
    resource: /app/backend/detections.csv
    title: Object-detection metadata (backend)
    last_modified: 2026-08-20
  - id: frames-csv
    resource: /app/frontend/public/Frames.csv
    title: Frame lookup table (frontend public)
    last_modified: 2026-08-20
---

# Frame ID format mismatch

## Observation (verified)

Sampled rows show different `video_id` conventions:

| File | Sample `video_id` |
| --- | --- |
| `detections.csv` | `L21_V01` |
| `Frames.csv` | `L21_V001` |

[^detections-csv]: Header `video_id,frame_id,object,...`; first data row uses `L21_V01`.
[^frames-csv]: Header `video_id,frame_id,shot_id`; first data row uses `L21_V001`.

## What is unverified

Whether `search/service.py` normalizes or pads video ids when joining object metadata onto retrieval results. If it joins raw strings, object enrichment can silently mismatch for affected videos.

## Next action

Inspect the join/lookup path in `app/backend/search/service.py` and `app/backend/search/object_index.py`; if raw equality is used, decide between normalizing at index-load time (preferred, single place) or per-request.
