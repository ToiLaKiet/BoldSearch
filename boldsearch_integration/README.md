# BoldSearch pipeline integration

This package is intentionally outside the BoldSearch application source. It
turns validated `aic_video_pipeline_v1` output into a versioned static release
and visual Milvus rows.

For Kaggle, use [the Run All notebook](../notebooks/kaggle_mp4_run_all.ipynb).
It clones the app and this runtime branch separately, bootstraps a visual-only
collection, and starts the backend, gateway, and Quick Tunnel after ingestion.

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
artifact tree under `--data-root`. The generated release manifest also pins
the config checksum and pipeline Git revision:

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

Before the first non-dry-run ingest, create a separate visual-only collection.
The command is idempotent: an existing collection is validated, never dropped
or changed.

```bash
PYTHONPATH=. python -m boldsearch_integration.cli bootstrap \
  --collection BoldSearchV1 --expected-vector-dim 1024
```

Non-dry-run ingestion requires `pymilvus`, `ZILLIZ_URI`, and `ZILLIZ_TOKEN`.
The adapter writes only `visual_embedding`; it never fabricates a caption
embedding. It validates the collection schema before the first upsert, retries
transient batches, and can resume from an atomic progress ledger:

```bash
PYTHONPATH=. python -m boldsearch_integration.cli ingest \
  --data-root /kaggle/working/aic_pipeline_data \
  --video-id L21_V001 --corpus-version l21-v1 \
  --collection BoldSearchV1 --batch-size 128 --retries 3 \
  --progress-path /kaggle/working/boldsearch-public/milvus-progress.json
```

The search service must select the visual modality for a V1 visual-only
collection. `select_search_modalities()` defaults to `visual_embedding` and
requires an explicit schema field before allowing `caption_embedding`; a
visual vector is never copied into a caption field.

To canary or roll back without deleting any release:

```bash
PYTHONPATH=. python -m boldsearch_integration.cli rollback \
  --output-root /kaggle/working/boldsearch-public \
  --release-id 20260828T120000Z-ab12cd34
```

When the archived BoldSearch clone is kept unchanged, launch it through the
source-safe overlay. It introspects the collection once, defaults to visual
only, and supports an explicit `BOLDSEARCH_SEARCH_MODALITIES=visual,caption`
only when both fields exist:

```bash
cd /kaggle/working/BoldSearch/app/backend
PYTHONPATH=/kaggle/working/boldsearch-integration \
uv run python -m boldsearch_integration.fastapi_launcher \
  --app-root "$PWD" \
  --host 127.0.0.1 --port 8000
```

This overlay is needed for a newly indexed V1 collection because the archived
service otherwise submits a non-existent `caption_embedding` request. It does
not change files in the clone.

Build the frontend without editing the cloned BoldSearch source. The runtime
Vite config rewrites the archived absolute API URL in memory and emits the
production assets into an external directory:

```bash
export BOLDSEARCH_FRONTEND_ROOT=/kaggle/working/BoldSearch/app/frontend
export BOLDSEARCH_FRONTEND_DIST=/kaggle/working/boldsearch-public/frontend-dist
export VITE_RUNTIME_CONFIG=/kaggle/working/boldsearch-integration/boldsearch_integration/vite.runtime.mjs
npm --prefix "$BOLDSEARCH_FRONTEND_ROOT" ci --ignore-scripts
npm --prefix "$BOLDSEARCH_FRONTEND_ROOT" exec -- vite build \
  --config "$VITE_RUNTIME_CONFIG"
```

Serve that directory and the active publisher release through
`boldsearch_integration.gateway`; expose only the gateway port to cloudflared.
For example:

```bash
export BOLDSEARCH_PUBLIC_ROOT=/kaggle/working/boldsearch-public
export BOLDSEARCH_FRONTEND_DIST=/kaggle/working/boldsearch-public/frontend-dist
export BOLDSEARCH_BACKEND=http://127.0.0.1:8000
PYTHONPATH=/kaggle/working/boldsearch-integration \
python -m boldsearch_integration.gateway
```

The tunnel helper downloads the architecture-specific official binary on first
use and records its own PID, so rerunning it does not kill unrelated processes:

```bash
PYTHONPATH=. python - <<'PY'
from pathlib import Path
from boldsearch_integration.tunnel import ensure_cloudflared, start_quick_tunnel

binary = ensure_cloudflared(Path('/kaggle/working/cloudflared'))
process, public_url = start_quick_tunnel(
    binary, 'http://127.0.0.1:7860',
    log_path=Path('/kaggle/working/boldsearch-public/tunnel.log'),
    pid_path=Path('/kaggle/working/boldsearch-public/tunnel.pid'),
)
print(process.pid, public_url)
PY
```
