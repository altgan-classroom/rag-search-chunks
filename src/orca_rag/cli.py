"""orca <command>:  ingest | search | stats"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from .config.settings import get_env, load_profile
from .ingestion.pipeline import Ingestor
from .llm.gateway import LLMGateway
from .store.qdrant_store import Principal, QdrantStore


def _store(profile_name: str):
    env, prof = get_env(), load_profile(profile_name)
    return env, prof, QdrantStore(env, prof)


def cmd_ingest(a):
    env, prof, store = _store(a.profile)
    ing = Ingestor(store, prof, LLMGateway(mock=env.llm.mock))
    t0 = time.time()
    reps = ing.ingest_dir(a.path, tenant=a.tenant, force=a.force)
    for r in reps:
        print(f"{r.status:9s} {r.source_key[:60]:60s} pages={r.pages:3d} skip={r.pages_skipped:3d} chunks={r.chunks:4d} {r.seconds:6.1f}s {r.error}")
    print(f"profile={prof.name} collection={store.collection} total={time.time() - t0:.0f}s stats={store.stats()}")
    return 1 if any(r.status == "failed" for r in reps) else 0


def cmd_search(a):
    env, prof, store = _store(a.profile)
    hits = store.search(a.query, Principal("cli", a.tenant), mode=prof.retrieval.mode, top_k=a.k)
    for h in hits:
        print(f"{h.score:.3f} {h.doc[:48]:48s} p.{h.pages[0]}-{h.pages[1]} {h.payload.get('section','')[:30]:30s} | {h.text[:110]!r}")


def cmd_stats(a):
    _, _, store = _store(a.profile)
    print(json.dumps({**store.health(), **store.stats()}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="orca")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--tenant", default="default")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest"); p.add_argument("path"); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_ingest)
    p = sub.add_parser("search"); p.add_argument("query"); p.add_argument("-k", type=int, default=5); p.set_defaults(fn=cmd_search)
    p = sub.add_parser("stats"); p.set_defaults(fn=cmd_stats)
    a = ap.parse_args(argv)
    logging.basicConfig(level=get_env().log_level, format="%(levelname)s %(name)s: %(message)s")
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
