# Kế hoạch đánh giá embedding và vector-store

Trạng thái: **kế hoạch đang hoạt động; baseline adapter đã triển khai, đánh giá chưa hoàn tất**

## Mục tiêu

Chọn một model image-text retrieval bằng một corpus, query set, và qrels cố
định. Khóa artifact của nó trước khi so sánh Qdrant và Milvus để kết quả giữa
các provider không bị nhiễu bởi embedding khác nhau.

## Pipeline

```text
config/embedding.yaml
  -> chọn một encoder được hỗ trợ
  -> kéo asset local đã pin
  -> tạo embedding float32 đã chuẩn hóa
  -> ghi vector bất biến + manifest + checksum
  -> đánh giá xếp hạng cosine chính xác
  -> khóa một artifact model
  -> ingest cùng artifact đó vào Qdrant và Milvus
  -> so sánh độ chính xác, relevance, latency, và vận hành
  -> ghi lại quyết định provider
```

`config/embedding.yaml` mô tả model đã chọn và input runtime của nó. Việc
truyền tải và versioning asset thuộc về DVC/R2; encoder chỉ nhận đường dẫn
local và không được tự ý tải asset có thể thay đổi trong lúc benchmark.

## Model trong phạm vi

| Key | Adapter | Dimension | Trạng thái |
|---|---|---:|---|
| `fg_clip_large` | `FGClipEncoder` | 768 | Được hỗ trợ từ một revision Hugging Face đã pin; sẽ chuyển sang một asset local bất biến khi layout DVC của nó được định nghĩa. |
| `beit3_base_itc` | `Beit3Encoder` | 768 | Được hỗ trợ; cần checksum checkpoint local trước khi sinh artifact. |
| `beit3_large_itc` | chưa có | 1024 | Đang lên kế hoạch; cần adapter riêng và xác minh. |

Thêm một ứng viên mới đòi hỏi một entry cấu hình tường minh, contract test cho
adapter, và một smoke test với model thật.

## Baseline đã triển khai

Đã xác minh trong cây mã nguồn hiện tại:

- `FGClipEncoder` và `Beit3Encoder` expose encode ảnh/text đã chuẩn hóa.
- Lựa chọn qua YAML validate cấu hình encoder đã biết mà không dynamic import.
- `VectorStore` định nghĩa `search()` trung lập theo provider và `ingest()`
  theo batch đơn.
- `QdrantStore` và `MilvusStore` implement cùng một contract.
- Bộ contract test dùng chung kiểm tra thứ tự, hình dạng kết quả trung lập,
  replace, ingest rỗng, và fixture riêng theo provider.
- FastAPI lifespan tạo và đóng resource store đã cấu hình cho mỗi worker.
- `POST /api/vector/ingest` và `POST /api/vector/search-similarity` (trong
  `vector/router.py`) đã expose `ingest()`/`search()` qua HTTP cho vector do
  caller cung cấp sẵn — đây là ingest/search "vector-in", không đi qua encoder.

Chưa triển khai:

- writer artifact embedding bất biến và xác minh checksum;
- corpus, qrels, quality metric, SLO, và trọng số benchmark đã duyệt;
- provisioning collection production và tuning index;
- endpoint đi từ ảnh/text thô qua encoder tới `VectorStore.ingest()`/
  `search()` (khác với ingest/search "vector-in" đã có ở `vector/router.py`);
- điều phối ingest offline GPU/multi-batch;
- quyết định provider cuối cùng.

Adapter runtime cố tình không provision collection. Milvus giữ một flush
tường minh sau mỗi lần ingest batch đơn hiện tại vì setup default-consistency
đã kiểm chứng không đảm bảo visibility ngay lập tức nếu thiếu nó. Chỉ xem lại
trade-off đó cùng với một benchmark throughput/consistency.

## Gate

### Trước khi chọn model

1. Ghi lại revision checkpoint đã chọn và giá trị SHA-256.
2. Ghi, đọc, và xác minh artifact bất biến cho mỗi ứng viên.
3. Duyệt corpus, query set, và qrels.
4. Đánh giá xếp hạng chính xác với cùng metric.
5. Khóa một identity model/checkpoint/artifact.

### Trước khi chọn provider

1. Provision collection Qdrant và Milvus tương đương bên ngoài tiến trình API.
2. Ingest cùng artifact đã khóa vào cả hai provider.
3. Xác minh độ chính xác exact/small-fixture trước các test hiệu năng ANN.
4. Benchmark với phần cứng, concurrency, cấu hình index, warm-up, và latency
   percentile đã khai báo.
5. Chấm điểm relevance, latency, chi phí resource, và độ phức tạp vận hành
   theo trọng số đã duyệt.
6. Ghi lại quyết định và điều kiện rollback trong một ADR.

Cho đến khi các gate này pass, Qdrant và Milvus vẫn là ứng viên được hỗ trợ
chứ chưa phải người thắng cuộc cho production.
