# BoldSearch pipeline integration

This package is intentionally outside the BoldSearch application source. It
turns validated `aic_video_pipeline_v1` output into a versioned static release
and visual Milvus rows.

Run the existing pipeline directly on an MP4 first:

```bash
PYTHONPATH=pipelines/aic_video_pipeline_v1/src \
python -m aic_video_pipeline_v1.cli run --streaming \
  --config pipelines/aic_video_pipeline_v1/configs/default.yaml \
  --video /kaggle/input/.../Videos_L21_a/video/L21_V001.mp4 \
  --video-id L21_V001
```

For a single resident pipeline process followed by an atomic publish, use the
integration runner. It accepts direct MP4 paths and keeps the original V1
artifact tree under `--data-root`:

```bash
PYTHONPATH=. python -m boldsearch_integration.cli run \
  --pipeline-root pipelines/aic_video_pipeline_v1 \
  --config pipelines/aic_video_pipeline_v1/configs/legacy_compatible.yaml \
  --video /kaggle/input/.../Videos_L21_a/video/L21_V001.mp4 \
  --data-root /kaggle/working/aic_pipeline_data \
  --output-root /kaggle/working/boldsearch-public \
  --corpus-version l21-legacy-v1
```

Then publish its output without changing the V1 artifact files:

```bash
PYTHONPATH=. python -m boldsearch_integration.cli publish \
  --data-root pipelines/aic_video_pipeline_v1/data \
  --video-id L21_V001 \
  --corpus-version l21-v1 \
  --output-root /kaggle/working/boldsearch-public
```

Validate the rows without connecting to Zilliz:

```bash
PYTHONPATH=. python -m boldsearch_integration.cli ingest \
  --data-root pipelines/aic_video_pipeline_v1/data \
  --video-id L21_V001 \
  --corpus-version l21-v1 \
  --collection BoldSearchV1 \
  --dry-run
```

Non-dry-run ingestion requires `pymilvus`, `ZILLIZ_URI`, and `ZILLIZ_TOKEN`.
The adapter writes only `visual_embedding`; it never fabricates a caption
embedding. The search service must therefore select the visual modality for a
V1 visual-only collection.
