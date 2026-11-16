"""Download open arXiv papers (NOT redistributed: each paper keeps its own licence) and verify/record sha256.
First run records hashes into sources.json; later runs verify them, so everyone evaluates on identical bytes."""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "data" / "raw" / "open"
PAPERS = {"arxiv-2005.11401-RAG-Lewis": "2005.11401", "arxiv-2310.11511-Self-RAG": "2310.11511", "arxiv-2401.15884-CRAG": "2401.15884"}
src_path = HERE / "sources.json"
src = json.loads(src_path.read_text()) if src_path.exists() else {}
OUT.mkdir(parents=True, exist_ok=True)
bad = 0
for stem, aid in PAPERS.items():
    p = OUT / f"{stem}.pdf"
    if not p.exists():
        req = urllib.request.Request(f"https://arxiv.org/pdf/{aid}", headers={"User-Agent": "orca-rag-dataset-fetch/1.0"})
        p.write_bytes(urllib.request.urlopen(req, timeout=60).read())
        print("downloaded", stem)
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if stem in src and src[stem]["sha256"] != h:
        print("SHA MISMATCH (arXiv may have published a new version):", stem)
        bad += 1
    src.setdefault(stem, {"arxiv": aid, "sha256": h})
src_path.write_text(json.dumps(src, indent=1))
sys.exit(bad)
