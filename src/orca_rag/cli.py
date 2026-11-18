"""orca <command>:  ingest | search | stats"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

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
    if (Path(a.path) / "manifest.json").is_file() or Path(a.path).name == "manifest.json":
        reps = ing.ingest_manifest(a.path, force=a.force)  # per-file tenant + ACL, e.g. datasets/synthetic
    else:
        reps = ing.ingest_dir(a.path, tenant=a.tenant, force=a.force)
    for r in reps:
        print(f"{r.status:9s} {r.source_key[:60]:60s} pages={r.pages:3d} skip={r.pages_skipped:3d} chunks={r.chunks:4d} {r.seconds:6.1f}s {r.error}")
    print(f"profile={prof.name} collection={store.collection} total={time.time() - t0:.0f}s stats={store.stats()}")
    return 1 if any(r.status == "failed" for r in reps) else 0


def cmd_search(a):
    env, prof, store = _store(a.profile)
    groups = tuple(g for g in a.groups.split(",") if g)
    hits = store.search(a.query, Principal("cli", a.tenant, groups), mode=prof.retrieval.mode, top_k=a.k)
    for h in hits:
        print(f"{h.score:.3f} {h.doc[:48]:48s} p.{h.pages[0]}-{h.pages[1]} {h.payload.get('section','')[:30]:30s} | {h.text[:110]!r}")


def _warm(path: str):
    from .ingestion.parsers import parse_cached
    t = time.time()
    d = parse_cached(path, get_env().data_dir / "parsed")
    return path, len(d.pages), sum(p.skipped for p in d.pages), round(time.time() - t, 1)


def cmd_parse(a):
    """Parse (layout + OCR) every file once, in parallel, into the parsed-document cache."""
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path
    from .ingestion.parsers import SUPPORTED
    files = sorted(str(p) for p in Path(a.path).rglob("*") if p.suffix.lower() in SUPPORTED)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for path, n, sk, sec in ex.map(_warm, files):
            print(f"parsed {Path(path).name[:62]:62s} pages={n:4d} skipped={sk:3d} {sec:6.1f}s", flush=True)


def cmd_eval(a):
    import asyncio
    from pathlib import Path
    from qdrant_client import QdrantClient
    from .eval.ablate import run_variants
    from .eval.harness import category_table, comparison_table
    from .eval.schema import CorpusManifest, GoldSet
    env = get_env()
    gold = GoldSet.load(Path(a.dataset) / "gold.json")
    manifest = CorpusManifest.load(Path(a.dataset) / "manifest.json")
    llm = LLMGateway(mock=env.llm.mock or not a.real_llm)
    client = QdrantClient(path=str(env.data_dir / "eval_db" / gold.name))
    variants = [(n, load_profile(n)) for n in a.profiles.split(",")]
    out = Path(a.out or f"reports/{gold.name}")
    runs = asyncio.run(run_variants(env, client, variants, manifest, gold, llm, generate=a.generate, out_root=out,
                                    categories=set(a.categories.split(",")) if a.categories else None))
    sums = [r.summary() for r in runs]
    m = ["recall@1", "recall@5", "mrr", "ndcg@10", "answer_in_context", "context_tokens", "retrieval_ms"]
    print(comparison_table(sums, m))
    print(category_table(sums, "recall@5"))
    return 0


def cmd_ablate(a):
    """Cumulative ladder (baseline -> advanced) or leave-one-out, on one dataset. Writes reports + a markdown table."""
    import asyncio
    from pathlib import Path
    from qdrant_client import QdrantClient
    from .eval.ablate import run_variants
    from .eval.harness import category_table, comparison_table
    from .eval.ladders import ladder, leave_one_out
    from .eval.schema import CorpusManifest, GoldSet
    env = get_env()
    gold = GoldSet.load(Path(a.dataset) / "gold.json")
    manifest = CorpusManifest.load(Path(a.dataset) / "manifest.json")
    llm = LLMGateway(mock=True)
    client = QdrantClient(path=str(env.data_dir / "eval_db" / gold.name))
    base = load_profile("baseline")
    variants = ladder(base) if a.mode == "ladder" else leave_one_out(base)
    out = Path(a.out or f"reports/{gold.name}-{a.mode}")
    runs = asyncio.run(run_variants(env, client, variants, manifest, gold, llm, generate=a.generate, out_root=out))
    sums = [r.summary() for r in runs]
    metrics = ["recall@1", "recall@5", "mrr", "ndcg@10", "answer_in_context", "context_tokens", "retrieval_ms"]
    md = f"## {gold.name} / {a.mode}\n\n" + comparison_table(sums, metrics, ci=a.ci) + "\n\n### recall@5 by category\n\n" + category_table(sums, "recall@5") + "\n"
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "report.md").write_text(md)
    print(md)
    return 0


def cmd_sweep(a):
    import asyncio
    from pathlib import Path
    from qdrant_client import QdrantClient
    from .eval.sweep import run_sweep, to_markdown
    env = get_env()
    ds = Path(a.dataset)
    client = QdrantClient(path=str(env.data_dir / "eval_db" / f"sweep-{ds.name}"))
    rows = asyncio.run(run_sweep(env, client, ds, a.experiment, a.models.split(",") if a.models else None,
                                 set(a.only.split(";")) if a.only else None))
    md = f"## sweep: {a.experiment} on {ds.name}\n\n" + to_markdown(rows)
    out = Path(a.out or f"reports/sweep-{ds.name}-{a.experiment}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(md)
    return 0


def cmd_compare(a):
    from .eval.compare import ladder_markdown
    print(ladder_markdown(a.dir, skip=tuple(a.skip.split(",")) if a.skip else ()))
    return 0


def cmd_stats(a):
    _, _, store = _store(a.profile)
    print(json.dumps({**store.health(), **store.stats()}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="orca")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--tenant", default="default")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest"); p.add_argument("path"); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_ingest)
    p = sub.add_parser("search"); p.add_argument("query"); p.add_argument("-k", type=int, default=5)
    p.add_argument("--groups", default="", help="comma-separated groups the caller belongs to, e.g. group:hr");  p.set_defaults(fn=cmd_search)
    p = sub.add_parser("parse"); p.add_argument("path"); p.add_argument("--workers", type=int, default=4); p.set_defaults(fn=cmd_parse)
    p = sub.add_parser("eval"); p.add_argument("dataset"); p.add_argument("--profiles", default="baseline,advanced")
    p.add_argument("--generate", action="store_true"); p.add_argument("--real-llm", action="store_true"); p.add_argument("--out")
    p.add_argument("--categories"); p.set_defaults(fn=cmd_eval)
    p = sub.add_parser("ablate"); p.add_argument("dataset"); p.add_argument("--mode", choices=["ladder", "loo"], default="ladder")
    p.add_argument("--generate", action="store_true"); p.add_argument("--ci", action="store_true"); p.add_argument("--out"); p.set_defaults(fn=cmd_ablate)
    p = sub.add_parser("sweep"); p.add_argument("dataset"); p.add_argument("--experiment", default="all", choices=["models", "size", "overlap", "strategy", "retrieval", "all"])
    p.add_argument("--models"); p.add_argument("--only"); p.add_argument("--out"); p.set_defaults(fn=cmd_sweep)
    p = sub.add_parser("compare"); p.add_argument("dir"); p.add_argument("--skip", default=""); p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("stats"); p.set_defaults(fn=cmd_stats)
    a = ap.parse_args(argv)
    logging.basicConfig(level=get_env().log_level, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "qdrant_client", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
