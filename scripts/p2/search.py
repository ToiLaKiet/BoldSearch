"""Batch search for AIC2026 P2 queries against the local BoldSearcher API.

Pre-translates Vietnamese queries to English (with retry + cache) to avoid
Google rate-limit poisoning the backend's inline translation.
"""
import json
import os
import time
import urllib.request

BASE = "http://127.0.0.1:8000/api/search/query"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "data", "aic2026-p2")  # untracked working dir: results
RESULTS = os.path.join(DATA, "results")
QUERIES = os.path.join(HERE, "queries.json")
TRANS = os.path.join(HERE, "translations.json")
PER_REQUEST_TIMEOUT = 240


def _looks_like_translate_error(text):
    head = text.strip().lower()
    return (
        not head
        or head.startswith("error ")
        or "there was an error" in head
        or "please try again later" in head
    )


def translate_all(texts):
    cache = {}
    if os.path.exists(TRANS):
        cache = json.load(open(TRANS, encoding="utf-8"))
    from deep_translator import GoogleTranslator

    changed = False
    for text in texts:
        if text in cache:
            continue
        for attempt in range(6):
            try:
                en = GoogleTranslator(source="auto", target="en").translate(text)
            except Exception as exc:  # noqa: BLE001
                print(f"[trans-retry {attempt}] {exc}")
                en = None
            if en and not _looks_like_translate_error(en):
                cache[text] = en.strip()
                changed = True
                break
            time.sleep(5)
        else:
            print(f"[trans-FAIL] using original: {text[:40]}")
            cache[text] = text
            changed = True
        time.sleep(1.5)
    if changed:
        json.dump(cache, open(TRANS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return cache


def search(text, task, top_k):
    body = json.dumps({"query": text, "task": task, "topK": top_k}).encode("utf-8")
    req = urllib.request.Request(
        BASE, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=PER_REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_one(qid, text, task, top_k, translations):
    out = os.path.join(RESULTS, qid + ".json")
    if os.path.exists(out) and os.path.getsize(out) > 100:
        print(f"[skip] {qid} (exists)")
        return
    english = translations.get(text, text)
    for attempt in (1, 2):
        try:
            t0 = time.time()
            data = search(english, task, top_k)
            results = data.get("results") or []
            slim = [
                {
                    "video_id": r.get("video_id"),
                    "frame_id": r.get("frame_id"),
                    "shot_id": r.get("shot_id"),
                    "distance": r.get("distance"),
                    "score": r.get("score"),
                }
                for r in results
            ]
            with open(out, "w", encoding="utf-8") as fh:
                json.dump({"query": text, "count": len(slim), "results": slim}, fh, ensure_ascii=False, indent=1)
            top = slim[0] if slim else {}
            print(f"[ok] {qid}: {len(slim)} rows in {time.time()-t0:.1f}s top1={top.get('video_id')}#{top.get('frame_id')}")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"[retry {attempt}] {qid}: {exc}")
            time.sleep(3)
    print(f"[FAIL] {qid}")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    pack = json.load(open(QUERIES, encoding="utf-8"))
    all_texts = []
    for q in pack["queries"]:
        if q["type"] == "trake":
            all_texts.extend(q["events"])
        else:
            all_texts.append(q["text"])
    translations = translate_all(all_texts)
    print(f"[trans] {len(translations)} translations ready")
    for q in pack["queries"]:
        qid, qtype = q["id"], q["type"]
        if qtype == "trake":
            for i, ev in enumerate(q["events"], 1):
                run_one(f"{qid}--e{i}", ev, "TRAKE", 40, translations)
        else:
            run_one(qid, q["text"], qtype.upper(), 60, translations)
    print("ALL DONE")


if __name__ == "__main__":
    main()
