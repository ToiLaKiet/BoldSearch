"""Definitive alignment test using frames that exist in BOTH Milvus and disk (map)."""
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/miphu/Projects/BoldSearch/app/backend")
sys.path.insert(0, "/Users/miphu/Projects/BoldSearch/data/aic2026-p2")
from app_config import app_config  # noqa: E402
from resolve import _load_map  # noqa: E402
from pymilvus import MilvusClient  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = "/Users/miphu/Projects/BoldSearch"
VID = "L26_V418"
client = MilvusClient(uri=app_config.ZILLIZ_URI, token=app_config.ZILLIZ_TOKEN)

rows = client.query(collection_name="BoldSearch", filter=f'video_id == "{VID}"',
                    output_fields=["frame_id"], limit=300)
milvus_ids = {int(r["frame_id"]) for r in rows}
fid_to_n = dict(_load_map(VID))  # frame_idx -> n
both = sorted(milvus_ids & set(fid_to_n))
print("overlap ids:", both[:14])

n_to_fid = {n: f for f, n in fid_to_n.items()}

from encoders.fg_clip import FGClipEncoder
enc = FGClipEncoder(None)

for fid in both[:4]:
    n = fid_to_n[fid]
    img_path = os.path.join(ROOT, "data", "keyframes", VID, f"{n:03d}.jpg")
    img = Image.open(img_path)
    print(f"\nframe {fid} -> n={n}, size={img.size}")
    vrows = client.query(collection_name="BoldSearch",
                         filter=f'video_id == "{VID}" and frame_id == {fid}',
                         output_fields=["visual_embedding"], limit=1)
    stored = np.array(vrows[0]["visual_embedding"], dtype=np.float32)
    fresh = np.asarray(enc.encode_images([img.convert("RGB")]))[0].astype(np.float32)
    cos = float(stored @ fresh / (np.linalg.norm(stored) * np.linalg.norm(fresh)))
    print(f"cos(stored, fresh SAME frame) = {cos:.4f}  (stored norm={np.linalg.norm(stored):.3f})")
