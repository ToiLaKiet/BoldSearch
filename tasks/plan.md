# Kế hoạch: BoldSearch chạy trực tiếp từ MP4

## Mục tiêu

Clone BoldSearch, dùng `aic_video_pipeline_v1` để đọc MP4 trực tiếp từ Kaggle,
giữ nguyên contract artifact của pipeline, index vào Milvus/Zilliz, và phục vụ
tìm kiếm/keyframe qua một origin Cloudflare.

Input phải hỗ trợ nguyên trạng:

```text
/kaggle/input/datasets/miphu2005/aic2026-videos-l21-a/
  Videos_L21_a/video/L21_V001.mp4
  Videos_L23_a/video/L23_V001.mp4
  Videos_L23_a/video/L23_V002.mp4
```

## Sự thật đã xác nhận

- `VideoPipelineV1.run_streaming()` đã nhận file `.mp4` trực tiếp: AutoShot,
  sample source frame, FG-CLIP2, dedup, rồi sinh `Shot.json`, `Frame.json`,
  PNG và NPY. Không cần keyframe đầu vào.
- Directory runner đã validate group `Videos_Lxx_<part>/video` và video ID
  `Lxx_Vxx[x].mp4`.
- Contract artifact hiện tại phải được giữ:

  ```text
  metadata/<video_id>/Shot.json
  metadata/<video_id>/Frame.json
  frames/<video_id>/<integer-frame-id>.png
  vectors/<video_id>/<integer-frame-id>.npy
  checkpoints/<video_id>.json
  ```

- BoldSearch không tự ingest pipeline output. Backend legacy tìm
  `visual_embedding` và `caption_embedding`; V1 chỉ sinh visual vector 1024
  chiều. Không copy visual vector thành caption vector cho có.
- Frontend cần `Frames.csv` với header `video_id,frame_id,shot_id` và fallback
  `/keyframes/<video_id>/<integer-frame-id>.png`.

## Kiến trúc đề xuất

Giữ pipeline là package độc lập, pin commit/version trong repo clone. Không
viết lại AutoShot/FG-CLIP trong FastAPI; thêm lớp **publisher/indexer** sau
pipeline.

```text
MP4 mounted input
  -> V1 discovery + run_streaming (output bất biến)
  -> durable TAR: PNG + NPY + JSON + checkpoint
  -> publisher/indexer
       -> validate Frame.json/PNG/NPY
       -> projection shot_000001 -> 1
       -> batch upsert Milvus theo corpus version
       -> Frames.csv + WebP dẫn xuất
  -> active corpus manifest (atomic switch)
  -> FastAPI search + static keyframes + Vite build
  -> same-origin gateway
  -> một Cloudflare Tunnel
```

### Invariant bắt buộc

1. Không đổi schema/tên file/ý nghĩa field của pipeline. Frame ID là source
   integer, không zero-pad; NPY `float32` 1024 chiều L2-normalized; PNG gốc
   vẫn trong archive.
2. Publisher chỉ projection sang Milvus, `Frames.csv`, thumbnail WebP và URL;
   không ghi ngược `Frame.json`.
3. Video chỉ searchable sau khi vector, metadata, CSV và keyframe cùng validate.
   Không publish trạng thái nửa chừng.
4. API response frontend giữ nguyên shape; chỉ query plan và URL ảnh đổi nội bộ.

### Hai chế độ corpus

| Chế độ | Khi dùng | Kết quả |
|---|---|---|
| `legacy-compatible` | Tái dùng collection hiện có | Pin model/revision/threshold cũ; frame IDs phải qua golden comparison |
| `new-corpus-v1` | Video mới hoặc đổi threshold | Rebuild/upsert index, `Frames.csv` và ảnh cùng version |

Không ghép output threshold `0.8` với collection/`Frames.csv` sinh bởi threshold
khác. Nếu “output vẫn vậy” bao gồm frame ID cũ, chọn `legacy-compatible` và
fail sớm khi golden comparison khác.

## Phases triển khai

### 0. Khóa contract và baseline

- Pin commit BoldSearch/pipeline, model revision, AutoShot checkpoint, sample
  interval, threshold và archive format trong `corpus-manifest.json`.
- Chọn `L21_V001` làm golden fixture; đo frame IDs, schema JSON, SHA PNG,
  vector shape/dtype/norm, latency search và byte ảnh.

**Nghiệm thu:** cùng input + config pin tạo manifest/frame IDs khớp golden;
khác biệt fail và không publish.

### 1. Đưa pipeline vào repo clone tái lập

- Dùng pinned package/submodule hoặc package archive có checksum; không phụ
  thuộc bản local không version hóa.
- Thêm profile L21/L23, `video_root` tuyệt đối khi mount trùng group.
- Bắt buộc `--dry-run`; reject duplicate ID/mismatch level.
- Reuse model qua nhiều video, hỗ trợ checkpoint/resume; pipeline xong mới nạp
  model search để không tranh VRAM.

**Nghiệm thu:** discovery chỉ nhận MP4 hợp lệ trực tiếp dưới
`Videos_Lxx_*/video`; rerun skip artifact hợp lệ.

### 2. Publisher và validation artifact

- Đọc `Frame.json` chỉ lấy `final_status=KEPT`; validate mọi PNG/NPY/vector.
- Map `shot_000001` thành `1` chỉ trong Milvus/CSV; metadata V1 không đổi.
- Tạo `Frames.csv` sort ổn định; WebP 480–640px cho grid và 960px preview.
- Ghi staging release, validate đầy đủ rồi atomic switch active manifest.

**Nghiệm thu:** row CSV = frame KEPT = vector/ảnh validate; crash không đổi
active corpus.

### 3. Milvus/Zilliz ingestion và search

- Schema versioned: primary key ổn định, `video_id`, `frame_id`, `shot_id`,
  `visual_embedding` (1024/COSINE), thumbnail URL, corpus version, metadata.
- Batch upsert idempotent theo `(corpus_version, video_id, frame_id)` và lưu
  progress/checksum để retry.
- Query builder config-driven: corpus V1 visual-only search visual field;
  legacy hybrid chỉ dùng caption khi caption vector thực sự tồn tại.
- Caption/ASR/OCR là stage sau, không phải fake vector.

**Nghiệm thu:** query text/visual trả video MP4 mới; 100% hit có URL ảnh 200;
dim/metric/field sai fail trước upsert.

### 4. Frontend và keyframe nhanh

- API cùng origin `/api`; grid dùng WebP thumbnail, detail tải ảnh lớn khi mở;
  fallback PNG giữ tương thích artifact cũ.
- Lazy/async decoding, image dimensions, virtualize/paginate grid và filmstrip.
- Static trả ETag, Content-Type đúng và immutable cache cho asset versioned.
- Đặt default `SEARCH_TOP_K=50`; only rerank candidates lớn hơn khi benchmark
  chứng minh cần thiết.

**Nghiệm thu:** 50 card không tải PNG full-resolution; ảnh ngoài viewport không
request; response không chứa embedding.

### 5. One-origin gateway và Cloudflare

- Gateway port 7860: `/api/*` proxy FastAPI; `/keyframes/*` và `/Frames.csv`
  từ active corpus; phần còn lại Vite/SPA.
- Một tunnel tới gateway. Quick Tunnel cho development; named tunnel + Access
  và rate limit cho chia sẻ ổn định.
- Health gate: publish -> Milvus -> FastAPI -> gateway -> local smoke ->
  public tunnel smoke. CORS same-origin/restricted, không wildcard credentials.

**Nghiệm thu:** một public URL phục vụ UI/API/ảnh; health/search/CSV/sample
WebP thành công; restart không còn process stale.

### 6. Đo lường, rollout và rollback

- Ghi timing AutoShot/decode/embed/publish, kept rate, GPU peak, retry và byte
  WebP theo video. Trace từng search: encode, Milvus ANN, rerank, response.
- Benchmark local và Cloudflare tách riêng; chỉ giữ tối ưu vượt nhiễu và không
  phá golden contract.
- Canary một corpus version; active manifest hỗ trợ rollback tức thời.

**Nghiệm thu:** report before/after, ingestion lỗi không lộ partial corpus,
rollback phục hồi được search/ảnh version trước.

## Ngân sách hiệu năng khởi đầu

| Hạng mục | Mục tiêu |
|---|---:|
| Milvus ANN warm p95 | ≤ 500 ms |
| API search local warm p95, không tính cold model load | ≤ 1.5 s |
| Kết quả trả về mặc định | 50 |
| Payload search không embedding | ≤ 1 MiB |
| Grid 50 thumbnail | ≤ 5 MiB trước cache |
| Thumbnail card | WebP median ≤ 100 KiB, width 480–640px |
| Detail image | WebP 960px, chỉ tải khi mở |
| Hit có ảnh 200 | 100% |
| Publish partial corpus | 0 |

Các số này được xác nhận bằng baseline Kaggle/Zilliz thật; Quick Tunnel đo
riêng vì không phải SLA.

## Bảo mật và vận hành

- Zilliz/Hugging Face credential đã xuất hiện trong archive: rotate ngay, bỏ
  hardcode, dùng Kaggle Secrets/.env không commit, không log .env.
- Validate path, video ID, TAR member/static path; giới hạn API body/timeout/rate.
- Quick Tunnel chỉ development. Chia sẻ lâu dài dùng named tunnel, Access và
  hostname kiểm soát. [Cloudflare Quick Tunnel docs](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/).
- Dùng lockfile/frozen install và audit dependency trước release.

## File dự kiến thay đổi khi implementation bắt đầu

- `pipelines/aic_video_pipeline_v1/`: profile L21/L23, publisher adapter/CLI.
- `app/backend/`: config/query schema, ingestion worker, readiness.
- `app/frontend/`: API runtime config, thumbnail/detail URL, lazy/virtual UI.
- `infra/` hoặc `scripts/`: Kaggle runbook, manifest, gateway/tunnel/rollback.
- `tests/`: golden artifact, ingestion schema, search-to-image E2E, browser
  performance, failure/resume.

## Câu hỏi cần chốt trước khi code

1. “Output vẫn vậy” bắt buộc frame IDs khớp 100% collection cũ hay chỉ cần
   schema/tên file/API response không đổi? Plan mặc định chọn mức nghiêm ngặt.
2. Video mới cần visual retrieval trước hay caption/ASR/OCR ngay đợt đầu?
3. Cloudflare chỉ demo Kaggle hay cần domain ổn định + Access?
