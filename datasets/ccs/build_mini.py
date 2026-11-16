"""datasets/ccs-mini: the CCS documents of <= 30 pages and every gold question whose gold docs are all inside that subset.
~5x cheaper to (re)index than the full corpus, so many-variant sweeps (embedding models, chunk sizes) are affordable on CPU.
Conclusions that rest on the long documents (falloff-test pair, scanned report) must be confirmed on the full set."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from orca_rag.eval.schema import GoldSet  # noqa: E402
from orca_rag.ingestion.parsers import parse_cached  # noqa: E402

RAW, CACHE = HERE.parents[1] / "data" / "raw" / "ccs", HERE.parents[1] / "data" / "parsed"
src = json.loads((HERE / "sources.json").read_text())
keep = [m["file"] for m in src["files"] if len(parse_cached(str(RAW / m["file"]), CACHE).pages) <= 30]
stems = {Path(f).stem for f in keep}
full = GoldSet.load(HERE / "gold.json")
qs = [q for q in full.questions if all(r.doc in stems for r in q.gold)]
out = HERE.parent / "ccs-mini"
out.mkdir(exist_ok=True)
GoldSet(name="ccs-mini", description="CCS documents <= 30 pages (see datasets/ccs/build_mini.py).", questions=qs).save(out / "gold.json")
(out / "manifest.json").write_text(json.dumps({"name": "ccs-mini", "files": [{"file": f"../../data/raw/ccs/{f}", "tenant": "default", "acl": ["public"]} for f in keep]}, indent=1))
print(f"{len(keep)} docs, {len(qs)}/{len(full.questions)} questions: {GoldSet(name='x', questions=qs).by_category()}")
