---
okf_version: "0.2"
---

# BoldSearch Knowledge Bundle

Curated, provenance-tracked knowledge for the BoldSearch competition project. Start here; each concept links to its canonical source in the repository.

## System

* [Data layout contract](/data-layout.md) - Where runtime data lives and why the corpus stays outside git.
* [Data scaling path](/scale-path.md) - Layered plan from the local `data/` layout to object storage and CDN serving.

## Evaluation and data quality

* [Offline evaluation contract](/evaluation-contract.md) - How the offline ranking gate works and its current `task_id`/`case_id` split.
* [Frame ID format mismatch](/frame-id-normalization.md) - Unverified join risk between `detections.csv` and `Frames.csv` video ids.
