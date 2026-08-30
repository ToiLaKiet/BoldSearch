"""Resolve Milvus (video_id, frame_id) -> keyframe image path on disk.

Mapping: data/map-keyframes/{video_id}.csv has columns n,pts_time,fps,frame_idx
where n (1-based) maps to image data/keyframes/{video_id}/{n:03d}.jpg and
frame_idx equals the Milvus frame_id.
"""
import bisect
import csv
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
MAP_DIR = os.path.join(ROOT, "data", "map-keyframes")
KF_DIR = os.path.join(ROOT, "data", "keyframes")
DATA = os.path.join(ROOT, "data", "aic2026-p2")  # untracked working dir: results, picks

_cache = {}


def _load_map(video_id):
    if video_id in _cache:
        return _cache[video_id]
    path = os.path.join(MAP_DIR, f"{video_id}.csv")
    entries = []  # (frame_idx:int, n:int)
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    entries.append((int(row["frame_idx"]), int(row["n"])))
                except (TypeError, ValueError, KeyError):
                    continue
        entries.sort()
    _cache[video_id] = entries
    return entries


def resolve_image(video_id, frame_id):
    """Return absolute image path for a Milvus row, or None."""
    entries = _load_map(video_id)
    if not entries:
        return None
    try:
        fid = int(str(frame_id))
    except (TypeError, ValueError):
        return None
    idxs = [e[0] for e in entries]
    pos = bisect.bisect_left(idxs, fid)
    for cand in (pos, pos - 1):  # exact or nearest
        if 0 <= cand < len(entries):
            frame_idx, n = entries[cand]
            if abs(frame_idx - fid) <= max(30, fid // 100):  # tolerant nearest
                img = os.path.join(KF_DIR, video_id, f"{n:03d}.jpg")
                if os.path.exists(img):
                    return img
    return None


def candidates_from_results(results_path, limit=8):
    data = json.load(open(results_path, encoding="utf-8"))
    out = []
    seen = set()
    for row in data.get("results", []):
        vid, fid = row.get("video_id"), row.get("frame_id")
        key = (vid, str(fid))
        if key in seen:
            continue
        seen.add(key)
        img = resolve_image(vid, fid)
        out.append(
            {
                "video_id": vid,
                "frame_id": fid,
                "shot_id": row.get("shot_id"),
                "distance": row.get("distance"),
                "image": img,
                "image_exists": bool(img),
            }
        )
        if len(out) >= limit:
            break
    return out


def sheet(qid, limit=8):
    """Print a JSON candidate sheet for one query id."""
    res = os.path.join(DATA, "results", f"{qid}.json")
    cands = candidates_from_results(res, limit)
    print(json.dumps({"qid": qid, "candidates": cands}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    for qid in sys.argv[1:]:
        sheet(qid)
