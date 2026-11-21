"""Slim the computed data and inline it into page.html -> out/session2_visual_lab.html (self-contained; load fonts only from Google Fonts)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = json.loads((HERE / "out" / "data.json").read_text())
qs = {q["id"]: q for q in D["chunking"]["questions"]}
for key, r in D["chunking"]["recipes"].items():
    for c in r["chunks"]:
        if c["doc"] != "HomePolicy_HP-2026-0417":
            c["text"] = ""  # only the 2026 home policy is displayed chunk by chunk
    for qid, pq in r["per_question"].items():
        markers = qs[qid]["markers"]
        pq["markers_at_k"] = {}
        for k in range(1, 9):
            joined = " ".join(h["context"] for h in pq["hits"][:k]).lower()
            pq["markers_at_k"][str(k)] = {m: (m.lower() in joined) for m in markers}
        for h in pq["hits"]:
            h["text"] = h["text"][:200]
            h.pop("context", None)
D["chunking"].pop("docs", None)
slim = json.dumps(D, separators=(",", ":")).replace("</", "<\\/")
html = (HERE / "page.html").read_text().replace("/*DATA*/", slim)
(HERE / "out" / "session2_visual_lab.html").write_text(html)
print("html", len(html) // 1024, "KB")
