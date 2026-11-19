"""Why meaning search alone struggles with exact identifiers: a small, real experiment with the project's own models.

    PYTHONPATH=src python scripts/demo_exact_terms.py

Builds 40 near-identical 'well summary' chunks (ids MB-101 .. MB-140 style, same wording, different numbers), then asks for each well by its id
in dense-only, keyword-only (BM25) and hybrid mode and reports how often the right well is ranked first. Uses bge-small + Qdrant/bm25 (cached models)."""
import random

from qdrant_client import QdrantClient

from orca_rag.config.settings import ChunkingCfg, EmbeddingCfg, Env, Profile
from orca_rag.store.base import Principal, point_id
from orca_rag.store.qdrant_store import QdrantStore

random.seed(7)
ids = [f"MB-{n}" for n in (101, 104, 110, 114, 140, 141, 144, 401, 410, 414, 104 + 1000)] + [f"MB-{random.randint(100, 499)}" for _ in range(40)]
ids = sorted(set(ids))[:40]
prof = Profile(embedding=EmbeddingCfg(dense_model="BAAI/bge-small-en-v1.5", sparse_model="Qdrant/bm25"), chunking=ChunkingCfg())
store = QdrantStore(Env(), prof, client=QdrantClient(":memory:"), collection="demo")
depth = {i: random.randint(3000, 9000) for i in ids}
rows = []
for k, wid in enumerate(ids):
    txt = f"Well Summary: {wid}\n\nOperator: Northgate Petroleum. Target formation: Maquoketa Group. Total depth {depth[wid]:,} ft. Status: injecting."
    rows.append({"id": point_id("d", "child", k), "text": txt, "embed_text": txt, "payload": {"kind": "child", "doc_id": f"d{k}", "doc_name": wid, "tenant": "t", "acl": ["public"], "source_key": wid, "page_start": 1, "page_end": 1}})
store.upsert_children(rows)
me = Principal("u", "t")


def run(label):
    """top_k=20 candidates per branch, as in the project's default recipe; report 'ranked first' and 'in the first 5'."""
    res = {}
    for mode in ("dense", "sparse", "hybrid"):
        first = top5 = 0
        gaps = []
        for wid in ids:
            hits = store.search(f"Which well has ID {wid}?", me, mode=mode, top_k=20)
            names = [h.payload["doc_name"] for h in hits]
            first += bool(names) and names[0] == wid
            top5 += wid in names[:5]
            if len(hits) > 1:
                gaps.append(hits[0].score - hits[1].score)
        res[mode] = (first, top5, sum(gaps) / max(1, len(gaps)))
    print(f"[{label}] {len(ids)} chunks; query = 'Which well has ID <id>?'")
    for m, (f, t5, g) in res.items():
        print(f"  {m:7s} right well FIRST {f:2d}/{len(ids)} = {f / len(ids):4.0%}   in first 5: {t5:2d}/{len(ids)} = {t5 / len(ids):4.0%}   (mean score gap 1st vs 2nd: {g:.4f})")


run("A: short, near-identical chunks")
vocab = ("pressure injection reservoir seal permeability monitoring pump valve sensor casing flow rate depth pipeline inspection calibration shutdown "
         "maintenance compliance sampling anomaly baseline threshold telemetry cable pad integrity cement perforation tracer plume saline aquifer fault caprock").split()
rows2 = []
for k, wid in enumerate(ids):
    rnd = random.Random(k)
    body = " ".join(" ".join(rnd.choice(vocab) for _ in range(30)).capitalize() + "." for _ in range(4))
    txt = f"Report on well {wid}. {body} Total depth {depth[wid]:,} ft."
    rows2.append({"id": point_id("d2", "child", k), "text": txt, "embed_text": txt, "payload": {"kind": "child", "doc_id": f"e{k}", "doc_name": wid, "tenant": "t2", "acl": ["public"], "source_key": wid, "page_start": 1, "page_end": 1}})
store.upsert_children(rows2)
me = Principal("u", "t2")
run("B: longer chunks where the id is one small part")
me = Principal("u", "t")
wid = "MB-104"
print(f"\nExample query 'Which well has ID {wid}?'")
for mode in ("dense", "sparse"):
    print(f"  {mode}:")
    for h in store.search(f"Which well has ID {wid}?", me, mode=mode, top_k=4):
        print(f"     {h.score:.4f}  {h.payload['doc_name']}")
