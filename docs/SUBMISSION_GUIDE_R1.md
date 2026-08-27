# BoldSearch — Submission Guide (Round 1 · SOTUYEN1)

Guide này dùng cho bộ đề chính thức SOTUYEN1 gồm 25 câu: 20 Textual Known Item
Search (KIS), 4 Question Answering (Q&A), và 1 Temporal Retrieval and Alignment
of Key Events (TRAKE).

> **Quan trọng:** nút `Submit` của BoldSearch chỉ lưu/xác nhận câu trả lời ở
> backend local. Cổng BTC không nhận từng `VIDEO / Frame` trên giao diện; nó
> chỉ nhận **một file `.zip`** có đúng cấu trúc CSV ở phần [Đóng gói và nộp lên
> BTC](#đóng-gói-và-nộp-lên-btc).

## Chuẩn bị trước khi làm bài

1. Mở BoldSearch tại `http://localhost:5173` và chắc backend đang chạy ở cổng
   `8000`.
2. Chọn đúng mode ở góc trên: `KIS`, `VQA`, hoặc `TRAKE`.
3. Đảm bảo thẻ kết quả có ảnh hiển thị. Ảnh trắng, đen, hoặc xám đôi khi là
   keyframe gốc của video, không nhất thiết là ảnh lỗi.
4. Ghi đáp án vào bảng riêng trước khi nộp: mã câu, `video_id`, `frame_id`, và
   ghi chú ngắn vì sao đó là đáp án.

## KIS — tìm một frame đúng

Dùng cho các câu có hậu tố `-kis`.

1. Tách đề thành hai lớp dấu hiệu:
   - **Dấu hiệu hiếm, chính:** tên riêng, loài vật, hoạt động, đồ vật khác
     thường, địa danh hoặc dòng chữ.
   - **Dấu hiệu xác nhận:** màu áo, số người, bố cục, bối cảnh, diễn biến.
2. Nhập 1–2 dấu hiệu chính ở ô **Query**, rồi bấm **Search keyframes**.
   Không cần dán nguyên đoạn đề dài ở lần đầu.
3. Xem các ảnh trả về. Nếu chưa đúng, thêm một query phụ hoặc thay bằng một
   chi tiết hiếm hơn; ví dụ tên món ăn, loài chim, tên nhân vật, hoặc hành động.
4. Bấm vào card phù hợp để mở trình xem frame của cả video. Dùng mũi tên hoặc
   filmstrip để dịch đến đúng khoảnh khắc.
5. Khi chắc chắn, chọn frame rồi bấm **Submit**. App sẽ hiện:
   `Saved locally — submit this answer in the BTC portal.`
6. Bấm **Copy answer** hoặc ghi `VIDEO / Frame` vào file CSV của câu này.

### Mẹo truy vấn KIS

- Ưu tiên tên riêng và đặc điểm hiếm: `Nguyễn Trung Trực`, `Lausanne`,
  `panna cotta`, `bạch tuộc đỏ`, `măng tây`.
- Với cảnh đông người, truy vấn hoạt động trước (`đua xe đạp`, `múa lân`), sau
  đó lọc bằng màu trang phục, số người, hay góc máy.
- Với cảnh nấu ăn, tìm nguyên liệu/món ăn trước, rồi kiểm tra thứ tự thao tác
  trong trình xem frame.
- Đừng chọn chỉ vì score cao: score là gợi ý, ảnh và frame trong modal mới là
  bằng chứng cuối cùng.

## Q&A — tìm video rồi trả lời chính xác

Dùng cho các câu có hậu tố `-qa`.

1. Dùng truy vấn để tìm đúng video/frame chứa thông tin cần đọc hoặc nghe.
2. Mở modal và kiểm tra lân cận frame tìm được; đọc kỹ chữ trên màn hình,
   subtitle, OCR, hoặc ngữ cảnh video.
3. Chuyển sang mode **VQA**, nhập đáp án văn bản ngắn, chính xác, không thêm
   giải thích nếu BTC không yêu cầu.
4. Bấm **Submit** để lưu local, rồi ghi `video_id`, `frame_id`, và câu trả lời
   vào file CSV của câu này.

Không suy đoán đáp án Q&A từ kết quả retrieval nếu chưa nhìn/nghe thấy bằng
chứng trong video.

## TRAKE — căn 4 mốc sự kiện

Dùng cho các câu có hậu tố `-trake`.

1. Tìm đúng video bằng hoạt động/nội dung chính trong đề.
2. Di chuyển trong modal đến từng mốc `E1` → `E4` theo thứ tự; mỗi mốc là
   **frame đầu tiên** thỏa điều kiện được nêu trong đề, trừ khi đề nói rõ là
   thời điểm cuối cùng.
3. Ghi riêng `frame_id` cho từng event. Đừng dùng một frame cho nhiều event
   nếu hình chưa thỏa chính xác điều kiện của event đó.
4. Ở mode **TRAKE**, chọn các frame trong cùng video và Submit local. Ghi chúng
   thành một dòng CSV đúng thứ tự `E1, E2, E3, E4` trước khi đóng gói nộp BTC.

## Đóng gói và nộp lên BTC

Portal BTC chỉ chấp nhận **một file `.zip`**. Bên trong ZIP phải có đúng thư
mục cấp đầu tên `submission/`; không nén trực tiếp các CSV rời.

```text
team_round1.zip
└── submission/
    ├── query-p1-1-kis.csv
    ├── query-p1-2-kis.csv
    ├── ...
    └── query-p1-25-kis.csv
```

Mỗi file CSV tương ứng một câu query và có tối đa 100 dòng dự đoán. Tên file
phải khớp chính xác tên query BTC phát hành, chỉ thay `.txt` bằng `.csv`.

| Loại | Mỗi dòng CSV (không có header) |
| --- | --- |
| KIS | `<video_id>,<frame_id>` |
| Q&A | `<video_id>,<frame_id>,<answer>` |
| TRAKE | `<video_id>,<frame_E1>,<frame_E2>,...,<frame_EN>` |

Ví dụ:

```csv
L21_V015,31940
L24_V012,2420,"Tên món ăn"
L22_V005,1200,1850,2100,2450
```

Quy chuẩn bắt buộc:

- CSV văn bản UTF-8, phân cách bằng dấu phẩy `,`, không có header.
- `video_id` không có đuôi `.mp4`; `frame_id` là số nguyên.
- Q&A tối đa 100 ký tự. Nếu answer có dấu phẩy, dấu ngoặc kép, hoặc xuống dòng,
  bao answer bằng `"..."`; dấu ngoặc kép bên trong ghi thành `""`.
- TRAKE phải có đúng số frame bằng số event và theo thứ tự thời gian.
- Nén **thư mục** `submission/`, không nén trực tiếp các file CSV.

Một 400 Bad Request từ portal thường là ZIP sai cấu trúc, sai tên CSV, file
không phải CSV UTF-8, có header, hoặc số cột/frame không đúng loại câu.

## Danh sách câu theo mode

| Mode | Câu |
| --- | --- |
| KIS (20) | p1-1, p1-2, p1-4, p1-5, p1-6, p1-7, p1-8, p1-10, p1-11, p1-12, p1-13, p1-14, p1-18, p1-19, p1-20, p1-21, p1-22, p1-23, p1-24, p1-25 |
| Q&A (4) | p1-3, p1-9, p1-15, p1-17 |
| TRAKE (1) | p1-16 |

## Checklist trước khi nộp BTC

- [ ] Đúng mã câu và đúng mode.
- [ ] Đã xem ảnh/frame thật trong modal, không chỉ nhìn card đầu tiên.
- [ ] `video_id` và `frame_id` được copy/ghi lại.
- [ ] Q&A có đáp án văn bản chính xác, không phải mô tả suy đoán.
- [ ] TRAKE có đủ bốn event, đúng thứ tự và cùng video.
- [ ] Từng CSV đúng tên query, UTF-8, không header và đúng số cột.
- [ ] ZIP chứa thư mục cấp đầu `submission/`, không phải CSV rời.
- [ ] Đã nộp ZIP lên BTC; trạng thái `Submitted` trên BoldSearch chỉ xác nhận
      local, **không** xác nhận BTC đã nhận bài.

## Xử lý lỗi thường gặp

| Hiện tượng | Cách xử lý |
| --- | --- |
| Card/modal có icon ảnh vỡ | Refresh trang; nếu vẫn lỗi, chạy backend và frontend rồi thử lại. |
| Ảnh trắng/đen/xám nhưng không có icon lỗi | Đây có thể là keyframe gốc trống; bỏ qua nếu không liên quan đề. |
| Bấm Submit rồi không thấy gửi BTC | Đúng với thiết kế hiện tại: ghi đáp án vào CSV, nén thư mục `submission/`, rồi upload ZIP lên BTC. |
| BTC trả `400 Bad Request` | Kiểm tra ZIP có thư mục `submission/`, tên từng CSV, UTF-8, không header, delimiter `,`, và số cột đúng KIS/Q&A/TRAKE. |
| Kết quả chưa đúng | Rút query ngắn hơn, dùng dấu hiệu hiếm, rồi kiểm tra các frame lân cận trong modal. |
