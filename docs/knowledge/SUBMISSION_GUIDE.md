# AIC26 — Cuộc thi AI

## Hướng dẫn nộp bài sơ tuyển

Tài liệu này mô tả các quy định chung áp dụng cho việc chuẩn bị và nộp kết quả
ở vòng sơ tuyển.

## 1. Các loại truy vấn

Vòng sơ tuyển gồm ba dạng truy vấn chính:

- **Textual Known Item Search (Textual KIS):** tìm kiếm chính xác theo văn bản.
- **Visual Question Answering (Q&A):** truy vấn dạng hỏi và trả lời.
- **Temporal Retrieval and Alignment of Key Events (TRAKE):** truy xuất và căn
  chỉnh các sự kiện video theo thời gian.

## 2. Các gói truy vấn

BTC sẽ cung cấp các gói câu truy vấn theo nhiều đợt. Với mỗi gói, đội thi phải
tạo kết quả tương ứng cho từng câu truy vấn và nộp trên hệ thống thi bằng tài
khoản BTC đã cấp.

Mỗi câu truy vấn được cung cấp trong một file văn bản riêng. Ví dụ, một gói có
thể gồm:

```text
query-1-kis.txt
query-2-kis.txt
query-3-qa.txt
query-4-trake.txt
```

Quy ước hậu tố file:

| Hậu tố | Loại truy vấn |
| --- | --- |
| `kis` | Textual Known Item Search |
| `qa` | Question Answering |
| `trake` | Temporal Retrieval and Alignment of Key Events |

## 3. Yêu cầu kết quả

Mỗi câu truy vấn tương ứng với một file `.csv`. Mỗi dòng là một dự đoán kết
quả. Mỗi file được phép có tối đa 100 dòng và không được có dòng tiêu đề.

### 3.1. Textual KIS

Định dạng mỗi dòng:

```text
<video_name>,<frame_id>
```

Ví dụ:

```csv
L00_V000,1234
L00_V055,5555
L01_V028,25300
```

### 3.2. Q&A

Định dạng mỗi dòng:

```text
<video_name>,<frame_id>,<answer>
```

Quy định đối với `answer`:

- Tối đa 100 ký tự.
- Có thể viết bằng tiếng Việt hoặc tiếng Anh.
- Được đánh giá theo ngữ nghĩa so với đáp án.

Ví dụ:

```csv
L01_V028,3450,5
L02_V011,1200,Năm người
L03_V005,2800,Màu đỏ
```

### 3.3. TRAKE

Định dạng mỗi dòng:

```text
<video_name>,<frame_id_1>,<frame_id_2>,...,<frame_id_N>
```

Trong đó, các frame tương ứng với `N` event được yêu cầu trong câu truy vấn.
Số lượng frame phải khớp chính xác với số event và thứ tự frame phải tuân theo
thứ tự thời gian của các event.

Ví dụ cho chuỗi bốn event:

```csv
L10_V001,1200,1850,2100,2450
L10_V001,1180,1820,2080,2420
L11_V003,5100,5700,6200,6800
```

## 4. Quy chuẩn CSV

CSV là file văn bản thuần túy, không phải file Excel. Phải nộp file có đuôi
`.csv`, không nộp file `.xlsx` hoặc `.xls`.

Quy tắc chung:

- Encoding: UTF-8.
- Delimiter: dấu phẩy (`,`).
- Line ending: CRLF (`\r\n`) hoặc LF (`\n`).
- Không có header row; file bắt đầu trực tiếp bằng dữ liệu.
- Không thêm khoảng trắng không cần thiết quanh các giá trị. Khoảng trắng đầu
  hoặc cuối giá trị được xem là một phần dữ liệu và không tự động bị loại bỏ.
- Tên video không có phần đuôi `.mp4`.
- Frame ID phải được ghi dưới dạng số nguyên.

### 4.1. Dấu ngoặc kép trong answer

Answer đơn giản không bắt buộc phải có dấu ngoặc kép:

```csv
L01_V028,3450,5
L02_V011,1200,Năm người
L03_V005,2800,Màu đỏ
```

Answer bắt buộc phải được bao quanh bằng dấu ngoặc kép khi chứa dấu phẩy,
dấu ngoặc kép hoặc ký tự xuống dòng.

```csv
L01_V028,3450,"Có 3 người, bao gồm nam và nữ"
L02_V011,1200,"Anh ấy nói ""Xin chào"""
L03_V005,2800,"Dòng 1
Dòng 2"
```

Dấu ngoặc kép bên trong answer được escape bằng cách viết thành hai dấu ngoặc
kép liên tiếp. Để giảm lỗi định dạng, có thể luôn bao quanh answer bằng dấu
ngoặc kép; cả hai cách đều được hệ thống chấp nhận.

Ví dụ Q&A hợp lệ:

```csv
L01_V028,3450,"5"
L02_V011,1200,"Năm người"
L03_V005,2800,"Màu đỏ, rất đẹp"
L04_V012,4100,"Anh ấy nói ""Tuyệt vời"""
```

## 5. Cách tạo file CSV

### Microsoft Excel

1. Nhập dữ liệu theo đúng định dạng.
2. Chọn **File**, sau đó chọn **Save As**.
3. Chọn loại file **CSV (Comma delimited) (*.csv)**.
4. Đặt tên file theo đúng tên câu truy vấn, chẳng hạn `query-1-kis.csv`.
5. Nếu Excel hỏi xác nhận compatibility, chọn tiếp tục lưu ở định dạng CSV.

### Google Sheets

1. Nhập dữ liệu theo đúng định dạng.
2. Chọn **File**, sau đó chọn **Download** và **Comma Separated Values (.csv)**.
3. Kiểm tra lại tên và nội dung file sau khi tải xuống.

### Notepad hoặc text editor

1. Gõ dữ liệu trực tiếp theo đúng định dạng CSV.
2. Lưu file với đuôi `.csv`.
3. Chọn encoding UTF-8 nếu trình soạn thảo có tùy chọn này.
4. Mở lại file bằng text editor để kiểm tra dữ liệu dạng văn bản thuần túy.

Các lỗi thường gặp khi tạo CSV:

- Lưu nhầm file Excel với đuôi `.xlsx` hoặc `.xls`.
- Dùng encoding khác UTF-8 làm ký tự tiếng Việt bị lỗi.
- Dùng dấu chấm phẩy (`;`) thay cho dấu phẩy (`,`).
- Thêm header row.
- Không bao quanh answer có dấu phẩy bằng dấu ngoặc kép.

## 6. Đóng gói và nộp kết quả

Mỗi đội đăng nhập bằng tài khoản BTC đã cấp, vào đúng vòng thi và nộp một file
`.zip` trên hệ thống thi.

Chuẩn bị file nộp theo các bước:

1. Tạo một thư mục có tên `submission`.
2. Đặt tất cả file CSV của gói truy vấn vào thư mục này.
3. Nén thư mục `submission` thành một file `.zip`.
4. Có thể đổi tên file ZIP, chẳng hạn `team_ABC_submission.zip`.

Cấu trúc bắt buộc:

```text
team_ABC_submission.zip
└── submission/
    ├── query-1-kis.csv
    ├── query-2-qa.csv
    ├── query-3-trake.csv
    └── ...
```

Phải có thư mục cấp đầu `submission/` bên trong ZIP. Không nén trực tiếp các
file CSV mà không có thư mục này. Chỉ định dạng ZIP được chấp nhận.

## 7. Đánh giá và xếp hạng

- Public Leaderboard được tính trên 50% đáp án của BTC.
- Kết quả cuối cùng được tính trên 100% đáp án và dùng cho Private Leaderboard
  của vòng sơ tuyển.
- Mỗi gói truy vấn được nộp tối đa ba lần.
- Kết quả của lần nộp cuối cùng được dùng để xếp hạng.
- Một lần nộp sai định dạng vẫn được tính là một lần nộp.
- Mỗi đội chỉ được sử dụng một tài khoản để nộp bài.

## 8. Checklist trước khi nộp

- [ ] File kết quả có đuôi `.csv`, không phải `.xlsx` hoặc `.xls`.
- [ ] File có encoding UTF-8 và delimiter là dấu phẩy.
- [ ] File không có header row.
- [ ] Tên file khớp với tên câu truy vấn, chỉ thay `.txt` bằng `.csv`.
- [ ] Định dạng dòng đúng với loại truy vấn KIS, Q&A hoặc TRAKE.
- [ ] Mỗi file có tối đa 100 dòng.
- [ ] Answer Q&A không quá 100 ký tự.
- [ ] Answer có ký tự đặc biệt được escape đúng chuẩn CSV.
- [ ] Tên video không có đuôi `.mp4`.
- [ ] Frame ID là số nguyên.
- [ ] TRAKE có đúng số frame theo số event và đúng thứ tự thời gian.
- [ ] Tất cả CSV nằm trong thư mục cấp đầu `submission/`.
- [ ] File nộp có định dạng `.zip`.
- [ ] Đã kiểm tra số lần nộp còn lại và nội dung của lần nộp cuối.

## 9. Kiểm tra khi gặp lỗi

Nếu hệ thống từ chối file, kiểm tra lần lượt:

1. ZIP có thư mục cấp đầu `submission/` hay không.
2. Tên file CSV có khớp chính xác với tên query hay không.
3. File có thực sự là CSV UTF-8 hay chỉ là file Excel được đổi tên hay không.
4. File có header hoặc dùng sai delimiter hay không.
5. Số cột có đúng với loại query hay không.
6. TRAKE có đủ số frame theo số event hay không.

Nếu vẫn gặp lỗi kỹ thuật sau khi đã kiểm tra các mục trên, liên hệ BTC.
