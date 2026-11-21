"""Compute the real data behind the Session 2 visual lab with the project's own models (bge-small + BM25), no network after the models are cached.

    PYTHONPATH=src python scripts/session2/build_data.py  ->  scripts/session2/out/data.json"""
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from corpus import DOCS, GOLD, PROBES, QUESTIONS, SENTENCES  # noqa: E402

from orca_rag.config.settings import ChunkingCfg, EmbeddingCfg, Env, Profile  # noqa: E402
from orca_rag.ingestion.chunkers import chunk_document, count_tokens  # noqa: E402
from orca_rag.ingestion.models import Page, ParsedDocument  # noqa: E402
from orca_rag.ingestion.pipeline import Ingestor  # noqa: E402
from orca_rag.store.base import Principal  # noqa: E402
from orca_rag.store.embedders import get_dense, get_sparse  # noqa: E402
from orca_rag.store.qdrant_store import QdrantStore  # noqa: E402

DENSE, SPARSE = "BAAI/bge-small-en-v1.5", "Qdrant/bm25"
ME = Principal("u", "t")
OUT = {}
dense = get_dense(DENSE)
sparse = get_sparse(SPARSE)


def parsed(name):
    txt = DOCS[name]
    return ParsedDocument(name, name + ".md", hashlib.sha256(txt.encode()).hexdigest(), [Page(1, txt)])


# ------------------------------------------------------------------ A. embeddings
sents = [s for _, s in SENTENCES]
X = np.array(dense.embed_docs(sents + PROBES), dtype=np.float32)
C = X @ X.T
n = len(sents)
Xc = X - X.mean(axis=0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
P2 = Xc @ Vt[:2].T
var = (S ** 2) / (S ** 2).sum()
OUT["embeddings"] = {
    "model": DENSE, "dim": int(X.shape[1]),
    "items": [{"group": g, "text": s} for g, s in SENTENCES] + [{"group": "query", "text": q} for q in PROBES],
    "cos": [[round(float(v), 4) for v in row] for row in C],
    "pca": [[round(float(a), 4), round(float(b), 4)] for a, b in P2],
    "pca_explained": [round(float(var[0]), 4), round(float(var[1]), 4)],
}
tok_file = glob.glob(str(Path.home() / "Documents/AltGAN/content_material/rag-for-unstructured-data-v2/data/models/models--Qdrant--bge-small-en-v1.5-onnx-Q/snapshots/*/tokenizer.json"))[0]
tk = Tokenizer.from_file(tok_file)
OUT["tokens"] = {s: tk.encode(s).tokens for s in ["MB-104", "MB-140", "HP-2026-0417", "CL-4471", "Water damage from a burst pipe is not covered.", "The escape of water limit is $10,000 per claim."]}

# ------------------------------------------------------------------ D. anatomy (the real lab point: Q3 board update, slide 3)
from orca_rag.ingestion.parsers import parse_file  # noqa: E402

d = parse_file(str(HERE.parents[1] / "datasets/formats/corpus/Q3_Board_Update.pptx"))
kids, _ = chunk_document(d, ChunkingCfg(size=350, overlap=50, parent_size=1500), embedder=None)
c = next(k for k in kids if k.page_start == 3)
et = c.embed_text(d.name, True)
dv = dense.embed_docs([et])[0]
si, sv = sparse.embed_docs([et])[0]
words = {}
import re  # noqa: E402

for w in set(re.findall(r"\w+", et.lower())):
    idx, val = sparse.embed_docs([w])[0]
    for i in idx:
        words.setdefault(int(i), w)
OUT["anatomy"] = {
    "embed_text": et, "text": c.text, "dense_first6": [round(float(x), 3) for x in dv[:6]], "dense": [round(float(x), 4) for x in dv],
    "sparse": [{"index": int(i), "value": round(float(v), 2), "word": words.get(int(i), "?")} for i, v in zip(si, sv)],
    "payload": {"doc_id": "305c46c5-4e94-...", "doc_name": "Q3_Board_Update", "source_key": "Q3_Board_Update.pptx", "tenant": "lab", "acl": ["public"], "version": 1,
                "doc_year": 2026, "doc_type": "other", "kind": "child", "chunk_ord": 2, "page_start": 3, "page_end": 3, "section": c.section, "is_table": c.is_table,
                "tokens": c.tokens, "content_hash": "9b1a288413c215ec", "context": "", "parent_id": "...:p:2"},
}

# ------------------------------------------------------------------ B/C. chunk recipes and retrieval
RECIPES = {
    "fixed": ("Fixed windows (500 characters)", ChunkingCfg(strategy="fixed_words", size=500, overlap=50, parent_child=False)),
    "structure": ("Structure-aware (up to 150 tokens)", ChunkingCfg(strategy="structure", size=150, overlap=20, parent_child=False)),
    "small": ("Structure-aware, tiny (up to 40 tokens)", ChunkingCfg(strategy="structure", size=40, overlap=8, parent_child=False)),
    "parent": ("Structure-aware, small children + parent sections", ChunkingCfg(strategy="structure", size=100, overlap=20, parent_child=True, parent_size=1500)),
}


def is_gold(q, doc, text):
    return any(doc == gd and gs.lower() in text.lower() for gd, gs in GOLD[q])


def run_recipe(key, label, cfg, modes=("hybrid",), depths=(20,)):
    prof = Profile(embedding=EmbeddingCfg(dense_model=DENSE, sparse_model=SPARSE), chunking=cfg)
    store = QdrantStore(Env(), prof, client=QdrantClient(":memory:"), collection=f"c_{key}")
    ing = Ingestor(store, prof, parse_cache="/private/tmp/s2parse")
    for name in DOCS:
        r = ing.ingest_file(name + ".md", parsed=parsed(name), tenant="t")
        assert r.status == "ingested", r
    chunks = []
    for name in DOCS:
        ch, par = chunk_document(parsed(name), cfg, embedder=None)
        chunks += [{"doc": name, "ord": x.ordinal, "section": x.section.split(" > ")[-1] if x.section else "", "tokens": x.tokens, "chars": len(x.text), "text": x.text,
                    "parent": x.parent_ordinal, "is_table": x.is_table, "pages": [x.page_start, x.page_end]} for x in ch]
    return store, prof, chunks


chunk_out = {}
for key, (label, cfg) in RECIPES.items():
    store, prof, chunks = run_recipe(key, label, cfg)
    per_q = {}
    for qid, q, kind, markers, note in QUESTIONS:
        hits = store.search(q, ME, mode="hybrid", top_k=20)
        res = []
        for h in hits[:8]:
            ctx = h.text
            if cfg.parent_child and h.payload.get("parent_id"):
                ctx = store.fetch_parents([h.payload["parent_id"]], ME).get(h.payload["parent_id"], h).text
            res.append({"doc": h.payload["doc_name"], "section": (h.payload.get("section") or "").split(" > ")[-1], "score": round(h.score, 4), "tokens": h.payload.get("tokens"),
                        "text": h.text, "context": ctx, "gold": is_gold(qid, h.payload["doc_name"], h.text)})
        # answer-in-context at k: all markers present in the first k contexts
        aic = {}
        for k in range(1, 9):
            joined = " ".join(x["context"] for x in res[:k]).lower()
            aic[k] = all(m.lower() in joined for m in markers)
        per_q[qid] = {"hits": res, "answer_in_context_at_k": aic}
    chunk_out[key] = {"label": label, "n_chunks": len(chunks), "avg_tokens": round(sum(c["tokens"] for c in chunks) / len(chunks), 1), "chunks": chunks, "per_question": per_q}
OUT["chunking"] = {"docs": {k: v for k, v in DOCS.items()}, "recipes": chunk_out,
                   "questions": [{"id": a, "question": b, "kind": c_, "markers": d_, "note": e} for a, b, c_, d_, e in QUESTIONS]}

# ------------------------------------------------------------------ E. search arena (structure-aware recipe)
store, prof, chunks = run_recipe("arena", "arena", RECIPES["structure"][1])
arena = {}
for qid, q, kind, markers, note in QUESTIONS:
    entry = {"question": q, "kind": kind, "note": note, "modes": {}}
    ranks = {}
    for mode in ("dense", "sparse"):
        hs = store.search(q, ME, mode=mode, top_k=50)
        ranks[mode] = {h.id: i + 1 for i, h in enumerate(hs)}
    for mode in ("dense", "sparse", "hybrid"):
        for depth in (5, 10, 20):
            hs = store.search(q, ME, mode=mode, top_k=depth)
            rows = []
            for i, h in enumerate(hs[:8]):
                rows.append({"rank": i + 1, "doc": h.payload["doc_name"], "section": (h.payload.get("section") or "").split(" > ")[-1], "score": round(h.score, 4),
                             "dense_rank": ranks["dense"].get(h.id), "sparse_rank": ranks["sparse"].get(h.id), "gold": is_gold(qid, h.payload["doc_name"], h.text),
                             "text": h.text[:260]})
            gold_rank = next((i + 1 for i, h in enumerate(hs) if is_gold(qid, h.payload["doc_name"], h.text)), None)
            entry["modes"][f"{mode}|{depth}"] = {"rows": rows, "gold_rank": gold_rank}
    arena[qid] = entry
# suite summary
summary = {}
for mode in ("dense", "sparse", "hybrid"):
    for depth in (5, 10, 20):
        rs = [arena[q]["modes"][f"{mode}|{depth}"]["gold_rank"] for q in arena]
        summary[f"{mode}|{depth}"] = {"n": len(rs), "found_top5": sum(1 for r in rs if r and r <= 5), "first": sum(1 for r in rs if r == 1), "mrr": round(sum(1 / r for r in rs if r) / len(rs), 3), "ranks": rs}
OUT["arena"] = {"questions": arena, "summary": summary, "n_chunks": len(chunks)}

Path(HERE / "out").mkdir(exist_ok=True)
(HERE / "out" / "data.json").write_text(json.dumps(OUT, indent=1))
print("ok", (HERE / "out" / "data.json").stat().st_size, "bytes")
