# AIC video pipeline V1

Đây là component độc lập với `pipelines/aic_video_pipeline`. V1 không import
source của pipeline cũ và chỉ dùng chung model cache `models/fgclip`.

Luồng xử lý:

```text
MP4 → AutoShot → Shot.json → index N frame → PNG → FGCLIP/Histogram
    → so sánh với đại diện KEPT mới nhất trong từng shot
    → xoá PNG/NPY của DUPLICATE → Frame.json → validate
```

Với mỗi shot, frame đầu tiên được giữ. Frame sau là `DUPLICATE` khi
`cosine_similarity(frame, đại_diện_KEPT) >= similarity.threshold`; frame không
trùng trở thành đại diện KEPT mới. Vì vậy một chuỗi dài frame giống nhau không
còn bị giữ lại theo từng cặp.

Chạy với MP4 bất kỳ (cần `--video-id` theo quy tắc `Lxx_Vxx`):

```bash
cd /home/long/Documents/AIC/BoldSearch/pipelines/aic_video_pipeline_v1
PYTHONPATH=src python -m aic_video_pipeline_v1.cli run \
  --config configs/default.yaml \
  --video "/path/to/video.mp4" \
  --video-id L21_V01 \
  --embedding-provider fgclip
```

Kết quả chỉ nằm trong component này:

```text
data/metadata/L21_V01/{Shot.json,Frame.json}
data/frames/L21_V01/<frame_id>.png       # chỉ KEPT
data/vectors/L21_V01/<frame_id>.npy      # chỉ KEPT
```

Mỗi lần `run` sẽ xoá đúng ba thư mục output của `video_id` trong V1 để tránh
dùng artifact cũ; không chạm vào `pipelines/aic_video_pipeline/data`.

```bash
PYTHONPATH=src python -m aic_video_pipeline_v1.cli validate \
  --config configs/default.yaml --video-id L21_V01
PYTHONPATH=src pytest -q
```
