# ProcessFrame deduplication

## Chạy test

```bash
cd processframe_fixed
python test.py
```

## API

```python
from processframe import build_deduplicated_index

result = build_deduplicated_index(
    frames=frames,
    embeddings=embeddings,
    shot_ids=shot_ids,
    shot_budget=5,
)

print(result.dedup_representatives)    # Một representative cho mỗi duplicate cluster
print(result.indexed_representatives)  # Tập sau diversity budget
print(result.clusters)                 # Metadata bao phủ toàn bộ frame
```

Code cũ vẫn có thể unpack:

```python
representatives, clusters = build_deduplicated_index(...)
```
