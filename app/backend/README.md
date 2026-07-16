# BoldSearcher Backend

Backend API cho hệ thống — xây dựng bằng **FastAPI** với kiến trúc modular.

## Cấu trúc thư mục

```text
backend/
├── main.py                          # Entry point — khởi tạo app, đăng ký routers
├── app_config.py                    # Đọc .env + hằng số chung (SYSTEM_NAME, API_PREFIX)
├── search/                          # Module tìm kiếm shot
│   ├── __init__.py
│   ├── router.py                    # Endpoints: tasks, shots, query, submit
│   ├── schema.py                    # Pydantic models cho request/response
│   └── service.py                   # Logic chấm điểm, tokenize, load data
├── ocr/                             # Module OCR (PaddleOCR)
│   ├── __init__.py
│   ├── router.py                    # POST /api/ocr/extract
│   └── schema.py                    # OcrRequest, OcrBox, OcrResponse
├── asr/                             # Module ASR (Speech-to-Text)
│   ├── __init__.py
│   ├── router.py                    # POST /api/asr/transcribe
│   └── schema.py                    # AsrRequest, TranscriptSegment, AsrResponse
├── object_detection/                # Module Object & Color Detection
│   ├── __init__.py
│   ├── router.py                    # POST /api/object-detection/detect
│   └── schema.py                    # DetectionRequest, DetectedObject, DetectionResponse
├── embedding/                       # Module Embedding (FG-CLIP + Milvus)
│   ├── __init__.py
│   ├── router.py                    # encode-image, encode-text, search, index
│   └── schema.py                    # Vector encoding + similarity search schemas
├── data/
│   └── shots.json                   # Dữ liệu mẫu (8 shots)
├── config/                          # Quyết định thiết kế: model, vector store
│   ├── embedding.yaml               # Encoder nào, dimension, checkpoint
│   └── vector_store.yaml            # Provider nào, collection, metric
├── .env.example                     # Thứ đổi theo máy: HOST/PORT, url provider
├── pyproject.toml                   # Dependencies + cấu hình pytest
└── uv.lock                          # Khoá version — đừng sửa tay
```

Ranh giới config: `.env` giữ thứ đổi theo máy, `config/*.yaml` giữ quyết định
thiết kế (review được trong PR), `app_config.py` là nơi duy nhất đọc env.

Dependencies khai ở `pyproject.toml`, khoá bởi `uv.lock` — cả hai nằm cùng thư mục này.

## Cài đặt & Chạy

```bash
cd app/backend
uv sync                              # dựng .venv từ uv.lock
uv run python main.py
```

- API server: `http://localhost:8000`
- Swagger UI (tài liệu API tự động): `http://localhost:8000/docs`

## API Endpoints

### System

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/api/health` | Health check |

### Search (`/api/search/...`)

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/api/search/tasks` | Danh sách task types (KIS, VKIS) |
| GET | `/api/search/shots` | Toàn bộ shots trong catalogue |
| POST | `/api/search/query` | Tìm kiếm shots theo query đa phương thức |
| POST | `/api/search/submit` | Nộp shot đã chọn làm kết quả |

### OCR (`/api/ocr/...`)

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| POST | `/api/ocr/extract` | Trích xuất text từ frame/ảnh |

### ASR (`/api/asr/...`)

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| POST | `/api/asr/transcribe` | Chuyển audio thành text |

### Object Detection (`/api/object-detection/...`)

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| POST | `/api/object-detection/detect` | Nhận diện vật thể + màu sắc chủ đạo |

### Embedding (`/api/embedding/...`)

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| POST | `/api/embedding/encode-image` | Encode ảnh/keyframe thành vector |
| POST | `/api/embedding/encode-text` | Encode text query thành vector |
| POST | `/api/embedding/search` | Tìm kiếm tương tự trên Milvus |
| POST | `/api/embedding/index` | Index keyframes của video vào Milvus |

## Cách thêm logic vào module

Mỗi module placeholder đã có sẵn `router.py` (endpoint) và `schema.py` (input/output format). Để tích hợp model AI:

### Bước 1 — Tạo `service.py`

```python
# Ví dụ: ocr/service.py
from paddleocr import PaddleOCR

ocr_model = PaddleOCR(use_angle_cls=True, lang='en')

def extract_text(image_path: str):
    results = ocr_model.ocr(image_path)
    # Xử lý kết quả, trả về theo format OcrResponse
    ...
```

### Bước 2 — Import vào `router.py`

```python
# ocr/router.py — chỉ cần sửa hàm endpoint
from ocr import service

@router.post("/extract", response_model=schema.OcrResponse)
async def extract_text(body: schema.OcrRequest):
    result = service.extract_text(body.image_url)
    return result
```

**Không cần sửa `main.py`** hay bất kỳ file nào bên ngoài module.

### Bước 3 — Thêm dependencies

Dùng `uv add` ở thư mục này — nó tự cập nhật `pyproject.toml` và `uv.lock`.
Đừng sửa tay `uv.lock`:

```bash
uv add 'paddleocr==2.9.1' 'paddlepaddle==3.1.0'
```

## Cách thêm module mới

1. Tạo thư mục mới (ví dụ `scene_classification/`)
2. Tạo `__init__.py`, `schema.py`, `router.py`
3. Đăng ký router trong `main.py`:

```python
from scene_classification.router import router as scene_router
app.include_router(scene_router, prefix=API_PREFIX)
```

## Cách tắt module

Comment 1 dòng trong `main.py`:

```python
# app.include_router(ocr_router, prefix=API_PREFIX)  # Tạm tắt OCR
```

## Tech Stack

Nguồn chính xác là `pyproject.toml` + `uv.lock`; bảng này chỉ để tham khảo nhanh.

| Component | Version |
|-----------|---------|
| Python | 3.13+ |
| FastAPI | 0.115.12 |
| Uvicorn | 0.34.3 |
| Pydantic | 2.13.4 |
| pydantic-settings | 2.7.0 |

## Ghi chú

- Frontend (Vite) proxy `/api` đến `http://127.0.0.1:8000` — xem `frontend/vite.config.js`
- Swagger UI tại `/docs` tự động sinh tài liệu từ schema + docstring
- Tất cả response đều có validation qua Pydantic — sai format sẽ trả 422 kèm chi tiết lỗi
