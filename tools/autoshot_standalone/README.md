# AutoShot standalone

Component này chỉ làm một việc: nhận video và xuất danh sách shot. Nó không
phụ thuộc `aic_video_pipeline_v1`, không trích xuất PNG và không embedding.

```text
MP4
 → đọc metadata
 → FFmpeg resize 48×27, vsync=0
 → AutoShot/TransNetV2Supernet
 → threshold
 → normalize scene ranges
 → Shot.json
```

`vsync=0` rất quan trọng: FFmpeg không tự chèn frame để bù duration. Vì vậy
`total_frames` trong output là số frame decode thực tế; `container_frame_count`
là số do metadata MP4 khai báo để đối chiếu.

## 1. Chuẩn bị trên máy local

Thư mục `--autoshot-root` phải chứa ít nhất:

```text
supernet_flattransf_3_8_8_8_13_12_0_16_60.py
linear.py
ckpt_0_200_0.pth
```

Cài dependency:

```bash
sudo apt-get install ffmpeg
cd /home/long/Documents/AIC/BoldSearch
/home/long/miniconda3/bin/pip install -r tools/autoshot_standalone/requirements.txt
```

Chạy một video:

```bash
PYTHONPATH=tools/autoshot_standalone \
/home/long/miniconda3/bin/python tools/autoshot_standalone/run_autoshot.py \
  --video "/path/to/video.mp4" \
  --autoshot-root "/home/long/Documents/AIC/AutoShot" \
  --checkpoint "/home/long/Documents/AIC/AutoShot/ckpt_0_200_0.pth" \
  --output "/path/to/output/video_Shot.json" \
  --device auto
```

Output gồm `Shot.json` với `frame_start`, `frame_end`, timestamp và metadata
video. `--allow-fallback` chỉ nên dùng khi chấp nhận một shot toàn video nếu
model lỗi.

## 2. Chạy trên Google Colab

Ví dụ sau giả sử AutoShot và video nằm trên Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Nếu source chưa có trong Colab:

```bash
!git clone https://github.com/ToiLaKiet/BoldSearch.git /content/BoldSearch
```

Sau đó cài FFmpeg và dependency:

```bash
%cd /content/BoldSearch
!apt-get -qq update && apt-get -qq install -y ffmpeg
!pip install -q -r tools/autoshot_standalone/requirements.txt
```

Nếu repository đã có sẵn, chỉ cần chạy từ lệnh `%cd` trở xuống.

Chạy GPU:

```bash
!PYTHONPATH=tools/autoshot_standalone python tools/autoshot_standalone/run_autoshot.py \
  --video "/content/drive/MyDrive/videos/input.mp4" \
  --autoshot-root "/content/drive/MyDrive/AutoShot" \
  --checkpoint "/content/drive/MyDrive/AutoShot/ckpt_0_200_0.pth" \
  --output "/content/drive/MyDrive/autoshot_results/input_Shot.json" \
  --device cuda
```

Trong Colab, chọn `Runtime → Change runtime type → T4 GPU` trước khi chạy.

## 3. Chạy nhiều video từ file TXT

File `videos.txt` có một đường dẫn mỗi dòng; dòng trống và dòng bắt đầu bằng
`#` được bỏ qua:

```text
/content/drive/MyDrive/videos/a.mp4
/content/drive/MyDrive/videos/b.mp4
# /content/drive/MyDrive/videos/disabled.mp4
```

Chạy tuần tự, phù hợp với một GPU:

```bash
PYTHONPATH=tools/autoshot_standalone \
python tools/autoshot_standalone/run_batch.py \
  --video-list /content/drive/MyDrive/videos.txt \
  --output-dir /content/drive/MyDrive/autoshot_results \
  --autoshot-root /content/drive/MyDrive/AutoShot \
  --checkpoint /content/drive/MyDrive/AutoShot/ckpt_0_200_0.pth \
  --device cuda \
  --workers 1
```

Chạy song song nhiều process:

```bash
... --device cpu --workers 4
```

`--workers 4` tạo bốn process, mỗi process tải một model. Với một GPU, không
nên tăng workers vì mỗi process chiếm VRAM riêng; dùng `--workers 1`. Với CPU,
có thể tăng workers theo số lõi và RAM. Mỗi video tạo một file:

```text
autoshot_results/a_Shot.json
autoshot_results/b_Shot.json
```

Tên stem video phải duy nhất trong cùng một thư mục output.

## 4. Kiểm tra từng thành phần

```bash
cd /home/long/Documents/AIC/BoldSearch/tools/autoshot_standalone
PYTHONPATH=. /home/long/miniconda3/bin/python -m pytest -q
```

Test kiểm tra metadata, batch cửa sổ temporal, normalize scene range và
đường dẫn danh sách video. Test không cần tải checkpoint nên có thể chạy trên
máy không có GPU.
