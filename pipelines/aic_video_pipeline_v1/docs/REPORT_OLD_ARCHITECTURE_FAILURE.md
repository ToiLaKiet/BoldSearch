# Báo cáo: vì sao kiến trúc pipeline cũ xử lý trùng frame không đúng

## 1. Kết luận ngắn

Lỗi chính không nằm ở việc đọc `threshold` từ YAML hay ở model FGCLIP. Lỗi
nằm ở thuật toán similarity của pipeline cũ: pipeline chia toàn bộ frame thành
các cặp cố định `(0,1), (2,3), (4,5), ...` và chỉ so sánh một lần trong từng
cặp. Nó không so sánh frame mới với đại diện `KEPT` gần nhất và cũng không reset
đại diện theo shot.

Vì vậy, khi hai frame trong một cặp giống nhau, pipeline chỉ loại được frame
đầu tiên của cặp; các cặp tiếp theo vẫn được xử lý độc lập. Một chuỗi hàng trăm
frame gần như giống nhau vẫn còn khoảng một nửa số frame.

## 2. Code cũ thực sự đang làm gì

File thực hiện similarity là:

```text
pipelines/aic_video_pipeline/src/aic_video_pipeline/orchestrator.py
```

Đoạn logic cũ có dạng:

```python
for index in range(0, len(ordered), 2):
    left = ordered[index]
    right = ordered[index + 1]
    score = dot(left_vector, right_vector)

    if score > threshold:
        left = DUPLICATE
        right = KEPT
    else:
        left = KEPT
        right = KEPT
```

Config cũ có:

```yaml
similarity:
  threshold: 0.3
  step: 2
```

Tuy nhiên, giá trị `similarity.step` không được đọc để điều khiển vòng lặp;
code hard-code số `2` trong `range(..., 2)`. Do đó thay `step` trong YAML không
thay đổi thuật toán.

## 3. Vì sao hạ threshold không loại thêm frame

Giả sử có 8 frame giống nhau:

```text
(F0, F1), (F2, F3), (F4, F5), (F6, F7)
```

Nếu similarity của mọi cặp đều lớn hơn threshold, kết quả là:

```text
F0 DUPLICATE, F1 KEPT
F2 DUPLICATE, F3 KEPT
F4 DUPLICATE, F5 KEPT
F6 DUPLICATE, F7 KEPT
```

Kết quả vẫn giữ 4 frame. Pipeline không bao giờ đem `F3` so sánh với `F4`,
không đem đại diện `F1` so sánh với `F2`, và không biết rằng cả 8 frame thuộc
cùng một chuỗi hình ảnh.

Về mặt toán học, với `N` frame:

```text
số frame KEPT tối thiểu của thuật toán cũ = ceil(N / 2)
```

Do đó, khi các cặp đã có score cao hơn threshold hiện tại, hạ threshold tiếp
không thể làm số frame giảm thêm. Threshold chỉ quyết định kết quả bên trong
từng cặp, không thay đổi phạm vi so sánh.

## 4. Bằng chứng từ kết quả chạy thật

Output hiện tại của video `L21_V01` trong pipeline cũ:

```text
Frame.json records : 1783
KEPT               : 892
DUPLICATE          : 891
PNG còn lại        : 892
NPY còn lại        : 892
```

Trong đoạn frame index `79–2529`:

```text
records            : 99
KEPT               : 50
DUPLICATE          : 49
```

Mẫu kết quả là các frame cách đều nhau, ví dụ frame đầu của mỗi cặp được đánh
`DUPLICATE` và frame sau được đánh `KEPT`. Đây là đúng hành vi của vòng lặp
`range(0, len(ordered), 2)`, không phải hành vi của một phép deduplication theo
chuỗi.

`Frame.json` vẫn giữ record `DUPLICATE` để truy vết, nhưng PNG/NPY của chúng đã
bị xóa ở bước cleanup. Vì vậy cần phân biệt hai con số:

```text
Frame.json records = số frame đã được xử lý
PNG/NPY            = số frame còn được vật chất hóa sau cleanup
```

## 5. Các vấn đề kiến trúc khác của pipeline cũ

### 5.1. Không reset đại diện theo shot

Similarity cũ sort frame theo `frame_index` rồi ghép cặp toàn cục. Nó không
reset so sánh khi `shot_id` thay đổi. Frame cuối shot trước có thể bị ghép với
frame đầu shot sau, dù hai shot là các cảnh độc lập.

### 5.2. Đại diện không được chọn theo chất lượng hoặc tính liên tục

Trong cặp tương đồng, pipeline luôn giữ frame bên phải. Quyết định này phụ
thuộc vào vị trí cặp, không phải chất lượng hình ảnh, độ rõ, hay khoảng cách tới
đại diện hiện tại. Khi đổi cách lấy mẫu, frame được giữ có thể đổi chỉ vì lệch
vị trí trong cặp.

### 5.3. `step` tồn tại trong config nhưng không có hiệu lực

Config tạo cảm giác có thể điều chỉnh chiến lược bằng `similarity.step`, nhưng
code không truyền giá trị đó vào thuật toán. Đây là lỗi giao diện cấu hình:
tham số được công bố nhưng không được áp dụng.

### 5.4. Rerun trên artifact cũ dễ gây trạng thái không nhất quán

Pipeline cũ có thể đọc lại `Frame.json`, PNG hoặc NPY còn sót từ lần chạy trước.
Nếu metadata có trạng thái chưa hoàn tất hoặc artifact bị xóa không đồng bộ,
bước validate có thể gặp lỗi như `invalid final_status`. Vì vậy mỗi lần đổi
threshold phải xóa đúng ba thư mục output của video trước khi chạy lại.

## 6. Kiến trúc V1 đã sửa lỗi như thế nào

V1 nằm độc lập tại:

```text
pipelines/aic_video_pipeline_v1/
```

Thuật toán mới trong
`src/aic_video_pipeline_v1/similarity.py` thực hiện:

```text
với mỗi shot:
    frame đầu tiên → KEPT và làm representative
    frame tiếp theo:
        so sánh với representative KEPT gần nhất
        score >= threshold → DUPLICATE
        score < threshold  → KEPT và trở thành representative mới
```

Đặc điểm quan trọng:

- Một `DUPLICATE` không bao giờ trở thành đại diện mới.
- Đại diện được reset khi chuyển sang shot khác.
- Chuỗi dài frame giống nhau được gom về một đại diện thay vì giữ một frame
  trong mỗi cặp.
- Sau similarity, chỉ PNG/NPY của `KEPT` được giữ lại.
- Mặc định V1 dùng checkpoint để tiếp tục đúng `video_id`; chỉ khi dùng
  `--fresh` hoặc source/config thay đổi thì mới xóa output của `video_id`, và
  không chạm dữ liệu component khác.

## 7. So sánh hai phiên bản

| Nội dung | Pipeline cũ | Pipeline V1 |
|---|---|---|
| Phạm vi so sánh | Cặp cố định toàn video | Tuần tự trong từng shot |
| Đại diện | Luôn giữ frame bên phải của cặp | Frame `KEPT` gần nhất |
| Chuỗi frame giống nhau | Còn khoảng một nửa | Gộp theo đại diện |
| `step` config | Bị hard-code là `2` | Không dùng cơ chế cặp cố định |
| Shot boundary | Không reset | Reset theo `shot_id` |
| Artifact cuối | Cleanup sau thuật toán cặp | Chỉ giữ PNG/NPY của `KEPT` |
| Dữ liệu pipeline cũ | Có thể bị ảnh hưởng khi rerun | V1 tách riêng hoàn toàn |

## 8. Kết luận

Việc thay threshold trong pipeline cũ không giải quyết được hiện tượng frame
79–2529 còn nhiều frame vì threshold không phải điểm nghẽn duy nhất. Thuật toán
pairwise đã giới hạn số frame có thể loại ngay từ cấu trúc vòng lặp.

Để deduplicate đúng theo chuỗi thời gian, cần thay chiến lược pairwise bằng
chiến lược representative-based theo shot. Đó là lý do pipeline V1 được tách
thành component riêng thay vì tiếp tục sửa trực tiếp pipeline cũ.
