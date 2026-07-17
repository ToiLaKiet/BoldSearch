# BoldSearch backend

Backend FastAPI cho việc truy xuất shot và pipeline multimodal đang xây dựng.

## Cài đặt

```bash
uv sync
cp .env.example .env  # tùy chọn
uv run python main.py
```

- API: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- Test: `uv run pytest`

Dùng `uv add <package>` để thay đổi dependency. `pyproject.toml` là tập
dependency khai báo và `uv.lock` là lock file được sinh ra.

## Sở hữu module

```text
backend/
├── main.py               # lắp ráp FastAPI và vòng đời tiến trình
├── app_config.py         # settings runtime đọc từ môi trường/.env
├── search/               # route search và submit hoạt động trên dữ liệu mẫu
├── embedding/             # contract HTTP dạng placeholder
├── ocr/                  # contract HTTP dạng placeholder
├── asr/                  # contract HTTP dạng placeholder
├── object_detection/     # contract HTTP dạng placeholder
├── encoders/             # adapter model-runtime và lựa chọn encoder
├── vector/               # contract HTTP cho ingest/search vector đã tính sẵn
├── vector_store/         # schema trung lập, contract, adapter Qdrant/Milvus
├── config/               # quyết định encoder/model đã review (config/embedding.yaml)
├── data/                 # danh mục shot mẫu
└── tests/                # test unit, lifecycle, và contract theo provider
```

Với các HTTP feature, `schema.py` sở hữu contract request/response,
`router.py` sở hữu việc dịch HTTP, và `service.py` sở hữu logic
application/pure business khi logic đó tồn tại. Không tạo layer rỗng chỉ để
khớp hình dạng này.

Các adapter hạ tầng như `encoders/` và `vector_store/` không được import
FastAPI. Model của chúng mô tả input/output nội bộ của method; feature schema
mô tả contract HTTP bên ngoài.

## Cấu hình

`AppConfig` là đối tượng settings runtime, đọc hoàn toàn từ môi trường hoặc
`.env`: host, port, `VECTOR_STORE_PROVIDER` (mặc định `milvus`),
`VECTOR_STORE_COLLECTION`, `QDRANT_URL`, và `MILVUS_URI`. Không có merge YAML
cho các giá trị này — lựa chọn vector store là một giá trị deployment, không
phải một quyết định thiết kế đã review.

`config/embedding.yaml` (đường dẫn đặt bởi `EMBEDDING_CONFIG_PATH`) là nguồn
YAML duy nhất còn lại, và nó chỉ scope cho encoder: model đã chọn, dimension,
device, và đường dẫn checkpoint/tokenizer, được `encoders/config.py` load.
Metric của vector vẫn là một quyết định provisioning/benchmark cho đến khi mã
runtime có consumer thật sự cho nó.

## Vòng đời vector store

`main.lifespan` chọn provider đã cấu hình, tạo một client và một adapter
`VectorStore` cho mỗi worker FastAPI, gán vào `app.state.vector_store`, và
đóng client khi shutdown.

Contract cố tình chỉ chứa hành vi hiện tại:

```python
class VectorStore(Protocol):
    def search(self, vector: Sequence[float], limit: int) -> list[SearchHit]: ...
    def ingest(self, points: Sequence[VectorPoint]) -> None: ...
```

Cả Qdrant và Milvus đều implement contract này. Response shape của provider
SDK được chuẩn hóa về schema trong `vector_store/schemas.py`; consumer chỉ nên
thấy `VectorPoint`, `SearchHit`, và các field trung lập của chúng.

Ranh giới hiện tại:

- `vector/router.py` expose `POST /api/vector/ingest` và
  `POST /api/vector/search-similarity`, gọi thẳng `search()`/`ingest()` với
  vector do caller cung cấp. Cả hai route đều không encode gì; vẫn chưa có
  endpoint nào đi từ ảnh/text thô sang vector (xem các placeholder ở
  `embedding/router.py`).
- Collection phải đã tồn tại sẵn. Mã runtime không tạo, tạo lại, hay xóa
  chúng.
- Ingest là một batch application, không phải một lớp điều phối
  GPU/multi-batch.
- Milvus gọi `flush()` sau ingest để request tiếp theo thấy được write dưới
  baseline default-consistency đã kiểm chứng.
- Port riêng cho search/ingest chỉ nên tách khi có consumer riêng thực sự cần
  dependency hẹp hơn.

Hành vi của provider được kiểm tra qua cùng bộ contract test ở
`tests/contract/test_vector_store.py`; việc sở hữu và dọn dẹp lifespan được
kiểm tra ở `tests/test_main.py`.

## Route

| Method | Path | Trạng thái |
|---|---|---|
| `GET` | `/api/health` | hoạt động |
| `GET` | `/api/search/tasks` | hoạt động |
| `GET` | `/api/search/shots` | hoạt động |
| `POST` | `/api/search/query` | hoạt động trên JSON mẫu |
| `POST` | `/api/search/submit` | hoạt động cục bộ |
| `POST` | `/api/vector/ingest` | hoạt động, nhận vector đã tính sẵn |
| `POST` | `/api/vector/search-similarity` | hoạt động, nhận vector đã tính sẵn |
| `POST` | `/api/ocr/extract` | placeholder |
| `POST` | `/api/asr/transcribe` | placeholder |
| `POST` | `/api/object-detection/detect` | placeholder |
| `POST` | `/api/embedding/encode-image` | placeholder |
| `POST` | `/api/embedding/encode-text` | placeholder |
| `POST` | `/api/embedding/search` | placeholder |
| `POST` | `/api/embedding/index` | placeholder |

Khi một placeholder trở thành thật, hãy tạo resource model/database sống lâu
trong lifespan rồi truyền cho consumer. Giữ validation request và mapping
response ở router; giữ việc dịch riêng theo provider bên trong adapter của nó.
