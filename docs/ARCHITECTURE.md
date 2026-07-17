# Kiến trúc

Trạng thái: hiện trạng triển khai cộng với các ranh giới ngắn hạn đã thống nhất.
Các mục đánh dấu **verified** được xác nhận bởi cây mã nguồn hiện tại; các mục
target chưa được triển khai.

## Hệ thống hiện tại

BoldSearch là một modular monolith: một backend FastAPI và một UI Vite React.

| Năng lực | Trạng thái hiện tại | Bằng chứng |
|---|---|---|
| Search | Chấm điểm lexical, object, color, temporal trên `shots.json` mẫu. | `app/backend/search/` |
| Embedding | Encoder adapter đã có; các route HTTP (encode, search, index) vẫn là placeholder. | `app/backend/encoders/`, `app/backend/embedding/` |
| OCR, ASR, detection | Contract HTTP là placeholder. | các feature package backend tương ứng |
| Vector store | Một Protocol `VectorStore` trung lập với adapter Qdrant và Milvus, cùng bộ contract test dùng chung. | `app/backend/vector_store/`, `app/backend/tests/contract/` |
| Vòng đời vector store | Một client/adapter đã cấu hình được mở mỗi worker FastAPI, lưu trên `app.state`, đóng khi shutdown. | `app/backend/main.py`, `app/backend/tests/test_main.py` |
| Ingest/search vector đã tính sẵn | `/api/vector/ingest` và `/api/vector/search-similarity` đọc/ghi thẳng vào store dùng chung; caller tự cung cấp vector, bề mặt này không encode gì cả. | `app/backend/vector/`, `app/backend/tests/test_vector_router.py` |
| Frontend | Prototype React dùng proxy `/api` của Vite. | `app/frontend/` |
| Benchmark/offline ingest | Chưa triển khai. | chỉ có kế hoạch kỹ thuật |

Sơ đồ nguồn: `architecture/system-overview.mmd`.

## Ranh giới

```text
browser
  -> FastAPI router             schema HTTP bên ngoài và dịch dữ liệu
  -> application/pure logic     truy vấn và định hình kết quả
  -> provider-neutral contract  hành vi VectorStore / encoder
  -> adapter                    Qdrant, Milvus, hoặc model SDK
```

- Pydantic feature schema là contract HTTP bên ngoài.
- Dataclass của vector-store là input/output nội bộ của method.
- `VectorStore` là một structural Protocol vì consumer cần hành vi, không cần
  implementation dùng chung hay lifecycle hook.
- Adapter Qdrant và Milvus sở hữu state client và collection tái sử dụng được.
- Các phép biến đổi thuần từ SDK sang dạng trung lập giữ nguyên là function
  hoặc private method; chỉ dùng class khi nó thực sự cần state của adapter.
- Tên field của provider và response object của SDK không được vượt qua ranh
  giới adapter.

## Sở hữu runtime

FastAPI lifespan là composition root cho tiến trình hiện tại:

1. Đọc provider và collection từ `AppConfig`.
2. Tạo SDK client đã chọn sau khi application bắt đầu khởi động.
3. Bọc nó trong một adapter `VectorStore` và expose qua `app.state`.
4. Đóng client khi shutdown.

Nhánh rẽ theo provider chỉ thuộc về composition root này. Consumer của
search/ingest chỉ nên nhận store trung lập, không được lặp lại việc chọn
Qdrant/Milvus.

Store cố tình chưa được tách thành port search và ingest riêng vì hiện chỉ có
một vòng đời và chưa có consumer thực sự nào cần bề mặt hẹp hơn. Xem lại quyết
định này nếu search và ingest chuyển sang tiến trình riêng hoặc có nhu cầu
quyền/scale khác nhau.

## Sở hữu cấu hình

`AppConfig` (`app_config.py`) đọc thẳng các cấu hình deployment/runtime từ môi
trường hoặc `.env`: provider vector store, collection, và các URL dịch vụ đều
mặc định và override tại đây — không có merge YAML cho các giá trị này.
YAML được version hóa chỉ dành cho các quyết định encoder/model đã review
(`config/embedding.yaml`, được `encoders/config.py` load); nó không chứa cấu
hình vector store. Config subclass riêng theo provider là không cần thiết cho
đến khi validation hoặc consumer của chúng thực sự phân hóa.

## Luồng hiện tại và luồng mục tiêu

Search hiện tại:

1. `POST /api/search/query` validate `SearchRequest`.
2. Router load shot mẫu và gọi logic chấm điểm thuần.
3. Kết quả xếp hạng được trả về qua response schema HTTP.

Ingest/search vector đã tính sẵn hiện tại (`vector/router.py`):

1. `POST /api/vector/ingest` validate một batch `VectorPointPayload` (vector
   do caller cung cấp cộng id/source/metadata) và gọi thẳng
   `VectorStore.ingest()`.
2. `POST /api/vector/search-similarity` validate một `query_vector` do caller
   cung cấp, gọi `VectorStore.search()`, và map giá trị `SearchHit` sang
   response schema công khai.
3. Cả hai route đều không encode gì cả — ai tạo ra vector (hiện là một pipeline
   keyframe offline) thì người đó sở hữu model.

Search dựa trên vector mục tiêu (vẫn chưa được nối):

1. Một use case nhận `VectorStore` dùng chung và encoder đã chọn.
2. Encoder tạo vector từ model/checkpoint đã khóa.
3. `VectorStore.search()` trả về giá trị `SearchHit` trung lập.
4. Use case gom nhóm keyframe, định hình kết quả, và router `embedding` map
   sang response schema công khai.

Ingest từ media thô vẫn cố tình nằm ngoài phạm vi baseline hiện tại:

1. Một consumer tạo giá trị `VectorPoint` trung lập từ một encoder, không phải
   từ vector do caller cung cấp.
2. `VectorStore.ingest()` ghi một batch và làm nó searchable.
3. Provisioning, GPU batching, điều phối artifact, và rebuild collection vẫn
   nằm ngoài adapter cho đến khi workflow của chúng được thiết kế.

## Guardrail

- Không mở kết nối provider tại thời điểm import module.
- Không để mã HTTP import type của provider SDK.
- Không để API tạo, tạo lại, hoặc xóa collection.
- Không trộn vector từ các model/checkpoint khác identity.
- Không so sánh điểm số giữa provider hoặc model nếu chưa chuẩn hóa semantics
  và chưa có benchmark tái lập được.
- Không thêm base class, factory, loader, hay port hẹp hơn cho đến khi một
  consumer hiện tại thực sự cần hành vi đó.

Các gate đánh giá và quyết định model/provider chưa chốt nằm ở
`docs/technical/00-embedding-vector-store-evaluation.md`.
