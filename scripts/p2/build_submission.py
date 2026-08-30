"""Submission builder: picks -> CSVs -> zip. Uses verified picks if present, else search top-3."""
import csv
import json
import os
import sys
import zipfile
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resolve import candidates_from_results  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "data", "aic2026-p2")  # untracked working dir: results, submission, picks
RESULTS = os.path.join(DATA, "results")
SUB = os.path.join(DATA, "submission")
QUERIES = os.path.join(HERE, "queries.json")
PICKS = os.path.join(DATA, "picks.json")

pack = json.load(open(QUERIES, encoding="utf-8"))
verified = json.load(open(PICKS, encoding="utf-8"))["queries"] if os.path.exists(PICKS) else {}

os.makedirs(SUB, exist_ok=True)

QA_ANSWERS = verified.get("_qa_answers", {
    "query-p2-7-qa": "1",
    "query-p2-9-qa": "Ca basa",
    "query-p2-12-qa": "4",
    "query-p2-19-qa": "duong Tran Hung Dao",
    "query-p2-23-qa": "15 phan nghin",
    "query-p2-27-qa": "7",
    "query-p2-28-qa": "Thit cua",
    "query-p2-29-qa": "500 gram",
    "query-p2-30-qa": "Oc huong",
})

for q in pack["queries"]:
    qid, qtype = q["id"], q["type"]
    lines = []
    picks = verified.get(qid)
    if qtype == "trake":
        if picks and picks.get("frames"):
            lines = [[picks["video"]] + [str(int(f)) for f in picks["frames"]]]
        else:
            event_cands = []
            for i, _ev in enumerate(q["events"], 1):
                event_cands.append(candidates_from_results(os.path.join(RESULTS, f"{qid}--e{i}.json"), 40))
            video_counts = Counter(c["video_id"] for cands in event_cands for c in cands)
            best_video = video_counts.most_common(1)[0][0] if video_counts else None
            frames = []
            for cands in event_cands:
                for c in cands:
                    if c["video_id"] == best_video and c["frame_id"] is not None:
                        frames.append(int(c["frame_id"]))
                        break
            need = len(q["events"])
            for cands in event_cands:
                if len(set(frames)) >= need:
                    break
                for c in cands:
                    if c["video_id"] == best_video and c["frame_id"] is not None:
                        frames.append(int(c["frame_id"]))
            frames = sorted(set(frames))[:need]
            lines = [[best_video or "L21_V001"] + [str(f) for f in frames]]
    else:
        if picks and picks.get("top"):
            lines = [[p["video"], str(int(p["frame"]))] for p in picks["top"]]
            if qtype == "qa":
                ans = QA_ANSWERS.get(qid, "x")
                lines = [[p["video"], str(int(p["frame"])), ans] for p in picks["top"]]
        else:
            cands = candidates_from_results(os.path.join(RESULTS, f"{qid}.json"), 3)
            for c in cands:
                if c["video_id"] and c["frame_id"] is not None:
                    if qtype == "qa":
                        lines.append([str(c["video_id"]), str(int(c["frame_id"])), QA_ANSWERS.get(qid, "x")])
                    else:
                        lines.append([str(c["video_id"]), str(int(c["frame_id"]))])

    out = os.path.join(SUB, qid + ".csv")
    if qtype == "qa":
        with open(out, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh, quoting=csv.QUOTE_ALL).writerows(lines)
    else:
        with open(out, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(lines)

zip_path = os.path.join(DATA, "AIC2026-710-p2.zip")
if os.path.exists(zip_path):
    os.remove(zip_path)
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for fn in sorted(os.listdir(SUB)):
        zf.write(os.path.join(SUB, fn), f"submission/{fn}")

with zipfile.ZipFile(zip_path) as zf:
    names = zf.namelist()
print("CSV files:", len(os.listdir(SUB)), "| zip entries:", len(names),
      "| all under submission/:", all(n.startswith("submission/") for n in names))
print("zip:", zip_path, os.path.getsize(zip_path), "bytes")
