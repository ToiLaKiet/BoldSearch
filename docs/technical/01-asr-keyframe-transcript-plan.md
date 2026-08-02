# ASR: gắn text vào keyframe theo timestamp

## 1. Trạng thái

Feature đã được triển khai cho tiếng Việt tại `POST /api/asr/transcribe`.

| Bằng chứng | Trạng thái |
|---|---|
| Schema, temporal join, ffmpeg adapter, ChunkFormer adapter và router | Verified từ code |
| Backend test suite | Verified: 70 tests pass |
| HTTP qua Uvicorn với model và audio thật | Verified: HTTP 200 |
| Media resolver trong `main.py` | Chưa có; deployment phải inject |
| Benchmark video 15 phút | Chưa hoàn tất |

## 2. Mục tiêu và phạm vi

Với một video hoặc audio đã được nhận diện bằng `video_id`:

1. Resolve media ở server.
2. Dùng ffmpeg chuyển media thành WAV PCM mono 16 kHz.
3. Chạy ChunkFormer đúng một lần để lấy các đoạn lời nói.
4. Với mỗi keyframe, tìm đoạn ASR chứa `timestamp` của frame.
5. Gắn một giá trị `text` vào keyframe; nếu không có đoạn phù hợp thì dùng
   `null`.

Feature join trực tiếp theo timestamp, không đi qua shot và không tìm đoạn gần
nhất.

## 3. Data contract

Tất cả `start` và `end` là số giây; `timestamp_ms` là số millisecond tính từ
đầu media, dùng cùng time base.

### 3.1. ASR segment

Mỗi phần tử trong `segments` là một object, không phải chuỗi:

```json
{
  "video_id": "L21_V91",
  "start": 60.0,
  "end": 103.0,
  "text": "Tụi t là BoldWarriors",
  "confidence": null
}
```

`confidence` là optional. ChunkFormer adapter hiện chưa cung cấp confidence nên
giá trị trả về là `null`.

### 3.2. Keyframe input

```json
{
  "frame_id": "002",
  "timestamp_ms": 83000,
  "img_path": "L21_V91/002.png"
}
```

`frame_id`, `timestamp_ms` và `img_path` là các field bắt buộc. `video_id` nằm ở
cấp video trong request AutoShot và được endpoint điền vào từng keyframe output.

### 3.3. Keyframe output

Mỗi keyframe chỉ có một `text`, không chứa `list[text]`:

```json
{
  "video_id": "L21_V91",
  "frame_id": "002",
  "timestamp_ms": 83000,
  "img_path": "L21_V91/002.png",
  "text": "Tụi t là BoldWarriors"
}
```

Nhiều keyframe có thể nhận cùng một text khi timestamp của chúng nằm trong cùng
một segment. Frame không nằm trong segment nào có `"text": null`.

Không có field `unmatched_count`; `text: null` đã thể hiện đầy đủ kết quả.

## 4. API contract

### `POST /api/asr/transcribe`

Media không được gửi trực tiếp trong request. Server dùng `video_id` để gọi
`asr_media_resolver.resolve(video_id)`.

Request:

```json
{
  "video_id": "L21_V91",
  "language": "vi",
  "keyframes": [
    {
      "frame_id": "001",
      "timestamp_ms": 60000,
      "img_path": "L21_V91/001.png"
    },
    {
      "frame_id": "002",
      "timestamp_ms": 83000,
      "img_path": "L21_V91/002.png"
    }
  ]
}
```

Response `200`:

```json
{
  "video_id": "L21_V91",
  "language": "vi",
  "segments": [
    {
      "video_id": "L21_V91",
      "start": 60.0,
      "end": 103.0,
      "text": "Tụi t là BoldWarriors",
      "confidence": null
    }
  ],
  "keyframes": [
    {
      "video_id": "L21_V91",
      "frame_id": "001",
      "timestamp_ms": 60000,
      "img_path": "L21_V91/001.png",
      "text": "Tụi t là BoldWarriors"
    },
    {
      "video_id": "L21_V91",
      "frame_id": "002",
      "timestamp_ms": 83000,
      "img_path": "L21_V91/002.png",
      "text": "Tụi t là BoldWarriors"
    }
  ],
  "full_transcript": "Tụi t là BoldWarriors"
}
```

`full_transcript` là text của tất cả segment nối bằng một dấu cách.

## 5. Quy tắc mapping

Keyframe có `timestamp_ms` thuộc segment khi:

```text
segment.start * 1000 <= timestamp_ms < segment.end * 1000
```

Đây là khoảng nửa mở `[start, end)`:

- `timestamp_ms == start * 1000`: thuộc segment.
- `timestamp_ms == end * 1000`: không thuộc segment đó.
- Không có segment phù hợp: `text = null`.
- Segment overlap: chọn segment đầu tiên sau khi sort theo `(start, end)`.
- Segment phải cùng `video_id` với video ở cấp request.
- Thứ tự keyframe output giống request.

Ví dụ: `83000 ∈ [60000, 103000)`, nên frame `002` nhận text của segment đó.

## 6. Module và trách nhiệm

| Module | Trách nhiệm | Không chịu trách nhiệm |
|---|---|---|
| `asr/schema.py` | Request, response và model dữ liệu | Chạy model hoặc ffmpeg |
| `asr/audio.py` | Gọi ffmpeg, tạo WAV PCM mono 16 kHz | ASR và keyframe mapping |
| `asr/chunkformer.py` | Giữ model trong memory, decode và chuẩn hóa segment | Resolve media và HTTP |
| `asr/transcript.py` | Validate và gắn `segment.text` vào keyframe | I/O, ffmpeg và model |
| `asr/router.py` | Điều phối endpoint và map lỗi sang HTTP | Logic containment |
| `main.py` | Load một transcriber cho mỗi process | Hiện chưa cung cấp media resolver |

`ChunkFormerTranscriber` là class vì giữ model dùng lại giữa các request. Temporal
join là pure function vì không giữ state.

## 7. Validation và lỗi HTTP

Validation chạy theo thứ tự: request schema → language → resolver → audio
normalization → ASR → keyframe mapping.

| Điều kiện | HTTP | `detail` |
|---|---:|---|
| Language khác `vi` | 422 | `ASR_LANGUAGE_UNSUPPORTED` |
| Thiếu hoặc rỗng `video_id` | 422 | FastAPI/Pydantic validation detail |
| Keyframe thiếu hoặc rỗng `img_path` | 422 | FastAPI/Pydantic validation detail |
| Chưa inject media resolver | 503 | `MEDIA_RESOLVER_UNAVAILABLE` |
| Resolver không tìm thấy media | 404 | `MEDIA_NOT_FOUND` |
| ffmpeg lỗi, timeout hoặc không có trên PATH | 422 | `AUDIO_NORMALIZATION_FAILED` |
| Output ChunkFormer sai format | 502 | `ASR_RESULT_INVALID` |
| Model load/decode lỗi | 503 | `ASR_INFERENCE_FAILED` |
| Keyframe có timestamp sai hoặc trùng `frame_id` | 422 | `KEYFRAME_SCHEMA_INVALID` |

FastAPI/Pydantic trả lỗi validation `422` chuẩn cho request không đúng kiểu dữ
liệu trước khi vào router.

## 8. Requirements và acceptance criteria

- **FR-ASR-01:** Chạy model một lần cho mỗi request/video.
- **FR-ASR-02:** Trả segment có `video_id`, `start`, `end`, `text`.
- **FR-ASR-03:** Nhận AutoShot JSON có `video_id` cấp trên; mỗi keyframe có
  `frame_id`, `timestamp`, `img_path`.
- **FR-ASR-04:** Gắn đúng một `text | null` vào mỗi keyframe bằng `[start, end)`.
- **FR-ASR-05:** Giữ thứ tự keyframe và thêm `video_id`, `text` vào output.
- **FR-ASR-06:** Dọn temporary WAV sau cả success lẫn failure.

- **BR-ASR-01:** Chỉ hỗ trợ `language = "vi"` trong MVP.
- **BR-ASR-02:** Media chỉ được tham chiếu bằng `video_id`; field ngoài schema
  bị reject.
- **BR-ASR-03:** Segment phải cùng `video_id` với request.
- **BR-ASR-04:** Frame không có lời nói nhận `text: null`; không dùng fallback.

- **AC-ASR-01:** Frame tại `83000` nhận text của segment `[60, 103)`.
- **AC-ASR-02:** Frame tại `103000` không nhận text của segment `[60, 103)`.
- **AC-ASR-03:** Hai frame trong cùng segment có thể nhận cùng text.
- **AC-ASR-04:** Frame ngoài tất cả segment có `text: null`.
- **AC-ASR-05:** Segment cross-video, timestamp âm/không hữu hạn và duplicate
  frame bị reject.
- **AC-ASR-06:** Response không có `unmatched_count`.
- **AC-ASR-07:** Audio tiếng Việt thật qua HTTP trả `200`, segment có text và
  keyframe mapping đúng khoảng thời gian.
- **AC-ASR-08:** Temporary WAV được xóa sau cả response thành công và lỗi.
- **AC-ASR-09:** Language khác `vi` bị reject với
  `ASR_LANGUAGE_UNSUPPORTED`.
- **AC-ASR-10:** OpenAPI không có `audio_url`; request chứa field này bị reject
  với HTTP `422`.

### Traceability

| Requirement/rule | Acceptance | Coverage |
|---|---|---|
| `FR-ASR-01` | `AC-ASR-07` | Verified từ call path; cần thêm spy test cho call count |
| `FR-ASR-02` | `AC-ASR-07` | `test_asr_chunkformer.py`, HTTP runtime test |
| `FR-ASR-03` | `AC-ASR-05` | `test_asr_transcript.py`, `test_asr_router.py` |
| `FR-ASR-04` | `AC-ASR-01..04` | `test_asr_transcript.py` |
| `FR-ASR-05` | `AC-ASR-03` | `test_asr_transcript.py` |
| `FR-ASR-06` | `AC-ASR-08` | Verified từ `finally`; cần thêm endpoint cleanup test |
| `BR-ASR-01` | `AC-ASR-09` | `test_asr_router.py` |
| `BR-ASR-02` | `AC-ASR-10` | `test_asr_router.py` |
| `BR-ASR-03` | `AC-ASR-05` | `test_asr_transcript.py` |
| `BR-ASR-04` | `AC-ASR-04`, `AC-ASR-06` | `test_asr_transcript.py`, `test_asr_router.py` |

## 9. Kiểm chứng

### Automated tests

```bash
cd app/backend
uv run pytest -q
```

Kết quả gần nhất: `70 passed`, cùng 17 warning deprecation từ `torch.jit.script`.

### HTTP runtime test đã thực hiện

Test chạy Uvicorn thật trên localhost, đăng ký production ASR router, inject
`ChunkFormerTranscriber("cpu")` và một resolver tạm trỏ tới audio WAV tiếng Việt
4.59 giây. Request được gửi qua HTTP socket, không dùng `TestClient`.

Kết quả:

- HTTP `200`.
- Một segment: `[0.0, 4.4)`.
- Keyframe tại `0.0` và `2.0` nhận segment text.
- Keyframe tại `4.5` và `4.7` nhận `text: null`.
- Response không có `unmatched_count`.

Model nhận nhầm một từ trong câu thử. Đây là chất lượng ASR, không phải lỗi API
hoặc temporal join.

## 10. Cấu hình runtime

- Checkpoint: `khanhld/chunkformer-rnnt-large-vie`.
- Revision được pin trong `asr/chunkformer.py`.
- Device lấy từ `ASR_DEVICE`, mặc định `cpu`.
- ffmpeg phải có trên `PATH`.
- Mỗi Uvicorn worker load một model riêng.

## 11. Việc chưa hoàn tất

1. Implement production `video_id -> media path` resolver và inject vào
   `app.state.asr_media_resolver`.
2. Chạy benchmark video tiếng Việt 15 phút để đo thời gian và memory.
3. Xác định consumer/nơi lưu keyframe JSON sau khi đã gắn text.
4. Thêm spy test xác nhận một lần gọi model/request và test cleanup temporary WAV
   khi endpoint success/failure.

## 12. Ngoài phạm vi

Diarization, word-level alignment, streaming, tự dò shot boundary, mapping theo
shot, nearest-segment fallback và ghi vector store.
