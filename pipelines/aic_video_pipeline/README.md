# AIC video pipeline

This component implements the video-processing architecture independently of
the web application. It owns shot detection, keyframe indexing, PNG extraction,
embedding creation, pairwise duplicate selection, and validation.

It deliberately has no imports from `app/`. This keeps a pipeline change out
of FastAPI/React merge conflicts and gives the backend one small integration
surface: `VideoPipelineOrchestrator`.

## Layout

```text
pipelines/aic_video_pipeline/
├── configs/default.yaml
├── docs/AIC_PROCESSING_ARCHITECTURE.md
├── notebooks/AIC_PIPELINE_WALKTHROUGH.ipynb
├── src/aic_video_pipeline/
├── tests/
└── data/                         # local only, ignored by Git
    ├── videos/<video_id>.mp4
    ├── metadata/<video_id>/{Shot,Frame}.json
    ├── frames/<video_id>/<frame_id>.png
    └── vectors/<video_id>/<frame_id>.npy
```

`video_id` must follow `Lxx_Vxx`, for example `L21_V01`. The pipeline creates
only the two JSON metadata files plus PNG/NPY artifacts specified above; it
does not create run folders, logs, manifests, or checkpoints.

## Local use

```bash
cd pipelines/aic_video_pipeline
PYTHONPATH=src python -m aic_video_pipeline.cli run \
  --video data/videos/L21_V01.mp4 \
  --embedding-provider histogram

PYTHONPATH=src python -m aic_video_pipeline.cli validate --video-id L21_V01
PYTHONPATH=src pytest -q
```

The default production embedding is FG-CLIP. `histogram` is an offline smoke
test provider. Shot detection uses the locally configured AutoShot checkpoint.

## Backend connection

Install this component as a package, or add
`pipelines/aic_video_pipeline/src` to the backend environment's import path.
The FastAPI layer should call the component from a dedicated use case or router;
it must not copy its processing logic into `app/backend/main.py`.

```python
from pathlib import Path

from aic_video_pipeline import VideoPipelineOrchestrator

pipeline = VideoPipelineOrchestrator.from_yaml(
    Path("pipelines/aic_video_pipeline/configs/default.yaml")
)
result = pipeline.run(
    Path("pipelines/aic_video_pipeline/data/videos/L21_V01.mp4"),
    embedding_provider="histogram",
)
```

The returned dictionary contains `video_id`, `frame_count`, and `kept_count`.
Consumers read `data/metadata/<video_id>/Frame.json` for frame paths, vector
paths, final statuses, and representatives.
