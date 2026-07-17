# BoldSearch

BoldSearch là workspace video-retrieval cho HCM AI Challenge 2026. Prototype
hiện tại kết hợp một backend FastAPI và một UI Vite React cho Known Item
Search (KIS) và Visual Known Item Search (VKIS).

## Trạng thái hiện tại

- Search lexical, object, color, và temporal chạy trên `shots.json` mẫu.
- Route HTTP cho OCR, ASR, object detection, và embedding vẫn là placeholder.
- Qdrant và Milvus implement chung một contract `VectorStore` trung lập cho
  search và ingest theo batch đơn.
- `/api/vector/ingest` và `/api/vector/search-similarity` đã hoạt động, đọc/
  ghi thẳng store dùng chung với vector do caller cung cấp — chưa có encoder
  nào ở giữa.
- FastAPI lifespan mở client vector-store đã cấu hình một lần cho mỗi worker
  và đóng lại khi shutdown.
- Đánh giá encoder, artifact embedding bất biến, benchmark vector-store, và
  ingest multi-batch offline vẫn là công việc đang được lên kế hoạch.

Hướng thiết kế UI lấy cảm hứng từ
[VISIONE](https://github.com/aimh-lab/visione), nhưng repo này tự sở hữu
runtime và quyết định kiến trúc riêng.

## Cấu trúc repository

```text
.
├── app/
│   ├── backend/       # dịch vụ FastAPI, test, encoder, adapter vector-store
│   └── frontend/      # prototype Vite React
├── architecture/      # nguồn Mermaid và diagram đã export
├── docs/
│   ├── ARCHITECTURE.md
│   ├── GIT_CONVENTION.md
│   └── technical/     # kế hoạch triển khai và evaluation gate
├── AGENTS.md          # hướng dẫn kỹ thuật nội bộ
└── fg-clip.ipynb      # notebook thử nghiệm
```

Sở hữu tài liệu:

- README này: tổng quan repository và chạy lần đầu.
- `app/backend/README.md`: setup backend, cấu hình, API, và ghi chú runtime.
- `docs/ARCHITECTURE.md`: ranh giới hiện tại đã xác minh và luồng mục tiêu đã
  thống nhất.
- `docs/technical/*`: kế hoạch chi tiết, bằng chứng, và evaluation gate chưa
  chốt.

Không thêm README khác trừ khi một component có setup/release lifecycle độc
lập mà không tài liệu nào trong số này giải thích được.

## Chạy cục bộ

Backend:

```bash
cd app/backend
uv sync
cp .env.example .env  # tùy chọn; mặc định trỏ tới service local
uv run python main.py
```

Frontend, ở terminal khác:

```bash
cd app/frontend
npm install
npm run dev
```

Mở `http://localhost:5173`. Vite proxy `/api` sang
`http://127.0.0.1:8000`; FastAPI expose Swagger UI tại
`http://localhost:8000/docs`.

Khởi động backend cũng đồng thời kết nối tới provider được chọn bởi
`VECTOR_STORE_PROVIDER`. Hãy provision collection đã cấu hình hoặc dùng
fixture của contract test trước khi thao tác với hành vi vector-store.

## Phát triển

Dependency và check cho backend:

```bash
cd app/backend
uv sync
uv run pytest
```

Check cho frontend:

```bash
cd app/frontend
npm run build
```

Chi tiết kiến trúc nằm ở `docs/ARCHITECTURE.md`; evaluation gate đang hoạt
động cho embedding và vector-store nằm ở
`docs/technical/00-embedding-vector-store-evaluation.md`.
