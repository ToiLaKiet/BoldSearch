"""Generate kaggle_p2_batch_query.ipynb with inlined translated queries."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
pack = json.load(open(os.path.join(HERE, "queries.json"), encoding="utf-8"))
trans_path = os.path.join(HERE, "translations.json")
trans = json.load(open(trans_path, encoding="utf-8")) if os.path.exists(trans_path) else {}

queries = []
for q in pack["queries"]:
    if q["type"] == "trake":
        for i, ev in enumerate(q["events"], 1):
            queries.append({"id": f"{q['id']}--e{i}", "text": trans.get(ev, ev)})
    else:
        queries.append({"id": q["id"], "text": trans.get(q["text"], q["text"])})

queries_repr = json.dumps(queries, ensure_ascii=False, indent=1)

cell1 = '''# Cell 1 - locate the FG-CLIP2 offline model dataset (the one used for Milvus ingestion)
from pathlib import Path

def find_model_root():
    known = Path('/kaggle/input/datasets/quanglongl040305/model2/aic_l28_offline_models/fgclip2')
    if known.is_dir():
        return known
    for cfg in Path('/kaggle/input').rglob('config.json'):
        p = cfg.parent
        if 'fgclip' in str(p).lower() and (any(p.glob('*.safetensors')) or any(p.glob('*.bin'))):
            return p
    raise RuntimeError('Attach the FG-CLIP2 model dataset (quanglongl040305/model2) and re-run.')

MODEL_ROOT = find_model_root()
print('model root:', MODEL_ROOT)
'''

cell2 = '''# Cell 2 - load ZILLIZ_URI / ZILLIZ_TOKEN from a.env (private Kaggle Input)
from pathlib import Path

ENV_INPUT_FILENAME = 'a.env'

def load_a_env(path):
    settings = {}
    for raw in path.read_text(encoding='utf-8-sig').splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        settings[key.strip().removeprefix('export ').strip()] = value.strip().strip('"').strip("'")
    missing = [k for k in ('ZILLIZ_URI', 'ZILLIZ_TOKEN') if not settings.get(k)]
    if missing:
        raise RuntimeError(f'{path} missing: {missing}')
    return settings

candidates = sorted({p.resolve() for p in Path('/kaggle/input').rglob(ENV_INPUT_FILENAME)})
if not candidates:
    raise RuntimeError(f'Attach the private Kaggle Input containing {ENV_INPUT_FILENAME}')
ENV = load_a_env(candidates[0])
print('a.env loaded from', candidates[0])
'''

cell3 = '''# Cell 3 - load model + tokenizer (same settings as app/backend/encoders/fg_clip.py)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForCausalLM.from_pretrained(str(MODEL_ROOT), trust_remote_code=True, local_files_only=True).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_ROOT), trust_remote_code=True, local_files_only=True)
print('model loaded on', device)

MAX_TEXT_LENGTH = 64
TEXT_WALK_TYPE = 'short'

@torch.no_grad()
def encode_text(text):
    inputs = tokenizer([text], max_length=MAX_TEXT_LENGTH, padding='max_length',
                       truncation=True, return_tensors='pt').to(device)
    feats = model.get_text_features(**inputs, walk_type=TEXT_WALK_TYPE)
    feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    return feats[0].float().cpu().numpy().tolist()
'''

cell4 = f'''# Cell 4 - AIC2026 P2 queries (English translations pre-computed)
QUERIES = {queries_repr}

print(len(QUERIES), 'queries ready')
'''

cell5 = '''# Cell 5 - search Milvus (visual_embedding only; caption_embedding is all-zero in this collection)
from pymilvus import MilvusClient, AnnSearchRequest
import json, time

client = MilvusClient(uri=ENV['ZILLIZ_URI'], token=ENV['ZILLIZ_TOKEN'])
COLLECTION = 'BoldSearch'
TOP_K = 100

results = {}
for q in QUERIES:
    t0 = time.time()
    vec = encode_text(q['text'])
    req = AnnSearchRequest(data=[vec], anns_field='visual_embedding',
                           param={'metric_type': 'COSINE', 'params': {}}, limit=TOP_K)
    hits = client.hybrid_search(collection_name=COLLECTION, reqs=[req],
                                ranker=None, limit=TOP_K,
                                output_fields=['video_id', 'frame_id', 'shot_id'])
    rows = []
    dists = []
    for group in (hits if isinstance(hits, list) else [hits]):
        for h in (group if isinstance(group, list) else [group]):
            ent = h.get('entity') or {}
            rows.append({'video_id': ent.get('video_id'), 'frame_id': ent.get('frame_id'),
                         'shot_id': ent.get('shot_id'), 'distance': float(h.get('distance', 0))})
            dists.append(float(h.get('distance', 0)))
    results[q['id']] = rows
    if dists:
        spread = max(dists) - min(dists)
        health = 'OK (discriminative)' if spread > 0.05 else 'FLAT - model mismatch suspected!'
        print(f"{q['id']}: top1={rows[0]['video_id']}#{rows[0]['frame_id']} d={dists[0]:.3f} spread={spread:.3f} {health} [{time.time()-t0:.1f}s]")

with open('/kaggle/working/p2_results.json', 'w', encoding='utf-8') as fh:
    json.dump(results, fh, ensure_ascii=False)
print('saved /kaggle/working/p2_results.json')
'''

cells = [
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}
    for src in (cell1, cell2, cell3, cell4, cell5)
]
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}
out = os.path.join(HERE, "kaggle_p2_batch_query.ipynb")
with open(out, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, ensure_ascii=False, indent=1)
print("notebook written:", out, "| cells:", len(cells), "| queries:", len(queries))
