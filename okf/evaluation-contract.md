---
type: Reference
title: Offline evaluation contract
description: How the offline ranking gate runs, what it requires, and the current task_id/case_id vocabulary split.
tags: [evaluation, testing, quality]
generated: { by: opencode_agent/glm-5.3-flash, at: 2026-08-30T05:20:00Z }
verified:
  - { by: opencode_agent/glm-5.3-flash, at: 2026-08-30T05:20:00Z }
status: stable
sources:
  - id: eval-readme
    resource: /app/backend/evaluation/README.md
    title: Evaluation module README
    last_modified: 2026-08-30
  - id: agents-md
    resource: /AGENTS.md
    title: BoldSearch Agent Notes
    last_modified: 2026-08-30
---

# Offline evaluation contract

## What it is

`evaluation.runner` is an offline gate: it scores exported retrieval rankings **without** starting FastAPI, FG-CLIP, or Milvus. It requires metadata and writes experiment artifacts under ignored `app/backend/evaluation/runs/` when paths there are chosen.[^agents-md]

## Vocabulary split (current, intentional)

- The runner and its inputs use `task_id`.[^agents-md]
- The versioned `evaluation/cases/` templates use `case_id` and are **incompatible** until the runner migration lands; update the runner and its tests together.[^agents-md]

## Relevance-judgment policy

Synthetic and `vision-draft` cases are **not** approved relevance judgments and must not be used for model-selection claims.[^agents-md]

## Commands

```sh
cd app/backend
uv run pytest tests/evaluation -q        # focused evaluation tests
uv run python -m evaluation.runner ...   # offline runner (module invocation)
```

[^agents-md]: BoldSearch Agent Notes — Evaluation And Submission section.
[^eval-readme]: Evaluation module README — runner semantics and artifact layout.
