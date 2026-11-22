"""Session 2 lab helper (runs inside the lab image).

  s2.py ingest [--profile s2_arena]            ingest data/session2 (six invented policies) with a chunking recipe
  s2.py ask q3|q11|q1|"your question" [--profile P] [-k 3] [--full]
                                               rank the same chunks three ways: meaning (dense), keywords (BM25), hybrid
  s2.py collections                            one line per collection: which recipe, how many chunks
"""
import argparse
import sys

from orca_rag.cli import _store
from orca_rag.config.settings import Profile, get_env
from orca_rag.ingestion.pipeline import Ingestor
from orca_rag.llm.gateway import LLMGateway
from orca_rag.store.base import Principal

ME = Principal("cli", "lab", ())
# prepared questions: (question, [(document, text the right chunk contains)]); every text must be present in the top chunks
PREP = {
    "q1": ("Is water damage from a burst pipe covered?", [("HomePolicy_HP-2026-0417", "10,000 per claim"), ("HomePolicy_HP-2026-0417", "48 consecutive hours")]),
    "q3": ("What is the deductible for escape of water under policy HP-2026-0417?", [("HomePolicy_HP-2026-0417", "Escape of water | $10,000 | $750")]),
    "q11": ("Will you reimburse me if my plumbing ruptures and soaks the carpet in my house?", [("HomePolicy_HP-2026-0417", "48 consecutive hours")]),
}


def ingest(a):
    env, prof, store = _store(a.profile)
    t = __import__("time").time()
    reps = Ingestor(store, prof, LLMGateway(mock=env.llm.mock)).ingest_dir("/data/session2", tenant="lab")
    for r in reps:
        print(f"{r.status:9s} {r.source_key[:44]:44s} chunks={r.chunks:3d} {r.error}")
    print(f"recipe={prof.name} collection={store.collection} {store.health().get('points')} points, {round(__import__('time').time() - t, 1)}s")


def context(store, prof, h):
    pid = h.payload.get("parent_id")
    if prof.chunking.parent_child and pid:
        return store.fetch_parents([pid], ME).get(pid, h).text
    return h.text


def ask(a):
    env, prof, store = _store(a.profile)
    question, needs = PREP.get(a.question, (a.question, []))
    print(f"\nQuestion: {question}\nRecipe: {prof.name}   showing top {a.k}\n")
    for mode, label in (("dense", "MEANING (dense)"), ("sparse", "KEYWORDS (BM25)"), ("hybrid", "HYBRID (both, fused)")):
        hits = store.search(question, ME, mode=mode, top_k=20)
        ctxs = [context(store, prof, h) for h in hits]
        rank = next((i + 1 for i, h in enumerate(hits) if any(d == h.doc and t.lower() in h.text.lower() for d, t in needs)), None)
        print(f"== {label}" + (f"   right chunk at rank {rank}" if needs and rank else "   right chunk NOT in the top 20" if needs else ""))
        for i, h in enumerate(hits[:a.k]):
            hit = any(d == h.doc and t.lower() in h.text.lower() for d, t in needs)
            text = " ".join(ctxs[i].split()) if a.full else " ".join(h.text.split())[:90]
            print(f"  #{i + 1} {h.score:6.2f} {h.doc[:26]:26s} {h.payload.get('section', '').split(' > ')[-1][:22]:22s} {'<== right chunk' if hit else ''}\n        {text}")
        if needs:
            top = " ".join(ctxs[:a.k]).lower()
            ok = all(t.lower() in top for _, t in needs)
            print(f"  all the facts needed are in the top {a.k} chunks sent to the model: {'YES' if ok else 'NO'}")
        print()


def collections(a):
    from qdrant_client import QdrantClient
    from qdrant_client import models as qm
    env = get_env()
    cl = QdrantClient(url=env.qdrant.url)
    known = {}
    for p in ("advanced", "s2_arena", "s2_fixed", "s2_tiny", "s2_parent"):
        try:
            known[Profile.load(p).collection_name()] = p
        except Exception:  # noqa: BLE001
            pass
    for c in cl.get_collections().collections:
        pts, _ = cl.scroll(c.name, limit=2000, with_payload=["kind", "tokens"], scroll_filter=qm.Filter(must=[qm.FieldCondition(key="kind", match=qm.MatchValue(value="child"))]))
        toks = [p.payload.get("tokens") or 0 for p in pts]
        print(f"{c.name}  recipe={known.get(c.name, '?'):9s} searchable chunks={len(pts):3d}  avg tokens={sum(toks) / max(1, len(toks)):.0f}")


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
s = sub.add_parser("ingest"); s.add_argument("--profile", default="s2_arena"); s.set_defaults(f=ingest)
s = sub.add_parser("ask"); s.add_argument("question"); s.add_argument("--profile", default="s2_arena"); s.add_argument("-k", type=int, default=3); s.add_argument("--full", action="store_true"); s.set_defaults(f=ask)
s = sub.add_parser("collections"); s.set_defaults(f=collections)
a = ap.parse_args()
a.f(a)
