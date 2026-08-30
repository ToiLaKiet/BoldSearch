"""Test if fg-clip2-large LATEST revision matches the stored Zilliz vectors."""
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoImageProcessor, AutoTokenizer
from PIL import Image

sys.path.insert(0, "/Users/miphu/Projects/BoldSearch/app/backend")
sys.path.insert(0, "/Users/miphu/Projects/BoldSearch/data/aic2026-p2")
from app_config import app_config  # noqa: E402
from resolve import _load_map  # noqa: E402
from pymilvus import MilvusClient  # noqa: E402

ROOT = "/Users/miphu/Projects/BoldSearch"
MODEL_ID = "qihoo360/fg-clip2-large"
REV = None  # latest

print("loading latest revision (no pin)...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True).eval()
proc = AutoImageProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

def embed_image(img):
    inputs = proc(images=[img.convert("RGB")], max_num_patches=1024, return_tensors="pt")
    with torch.no_grad():
        f = model.get_image_features(**inputs)
        f = f / f.norm(p=2, dim=-1, keepdim=True)
    return f[0].float().numpy()

client = MilvusClient(uri=app_config.ZILLIZ_URI, token=app_config.ZILLIZ_TOKEN)
VID = "L26_V418"
fid_to_n = dict(_load_map(VID))

for fid, n in [(0, 1), (300, 13), (2510, 60)]:
    img_path = os.path.join(ROOT, "data", "keyframes", VID, f"{n:03d}.jpg")
    fresh = embed_image(Image.open(img_path))
    vrows = client.query(collection_name="BoldSearch",
                         filter=f'video_id == "{VID}" and frame_id == {fid}',
                         output_fields=["visual_embedding"], limit=1)
    if not vrows:
        print(f"{fid}: not in DB")
        continue
    stored = np.array(vrows[0]["visual_embedding"], dtype=np.float32)
    cos = float(stored @ fresh / (np.linalg.norm(stored) * np.linalg.norm(fresh)))
    print(f"LATEST rev cos(stored, fresh) frame {fid} (n={n}): {cos:.4f}")
