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
from .store.base import make_store
from .store.base import Principal


def _store(profile_name: str):
    env, prof = get_env(), load_profile(profile_name)
    return env, prof, make_store(env, prof)


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


def cmd_variants(a):
    """Run named recipe variants (overrides on a base profile) from a YAML spec; per-question results are saved so
    `compare --ref` can compute paired confidence intervals. Used for the model refresh (configs/experiments/)."""
    import asyncio
    from pathlib import Path
    import yaml
    from qdrant_client import QdrantClient
    from .eval.ablate import run_variants, with_overrides
    from .eval.harness import comparison_table
    from .eval.schema import CorpusManifest, GoldSet
    env = get_env()
    spec = yaml.safe_load(Path(a.spec).read_text())
    base = load_profile(spec.get("base", "advanced"))
    gold = GoldSet.load(Path(a.dataset) / "gold.json")
    manifest = CorpusManifest.load(Path(a.dataset) / "manifest.json")
    keep = set(a.only.split(",")) if a.only else None
    variants = [(v["name"], with_overrides(base, v["name"], v.get("overrides") or {})) for v in spec["variants"]
                if not keep or v["name"] in keep]
    client = QdrantClient(path=str(env.data_dir / "eval_db" / f"variants-{gold.name}"))
    out = Path(a.out or f"reports/models-{Path(a.dataset).name}")
    runs = []
    for name, prof in variants:  # one at a time so a model that cannot load does not lose the others
        try:
            runs += asyncio.run(run_variants(env, client, [(name, prof)], manifest, gold, LLMGateway(mock=True), out_root=out))
        except Exception as e:
            print(f"[{name}] SKIPPED: {type(e).__name__}: {e}", flush=True)
    sums = [r.summary() for r in runs]
    print(comparison_table(sums, ["recall@1", "recall@5", "mrr", "ndcg@10", "answer_in_context", "context_tokens", "retrieval_ms"], ci=True))
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
    from .eval.compare import ladder_markdown, reference_markdown
    if a.ref:
        print(reference_markdown(a.dir, a.ref))
        return 0
    print(ladder_markdown(a.dir, skip=tuple(a.skip.split(",")) if a.skip else ()))
    return 0


def cmd_verify(a):
    """What is actually in the vector database? Collection, points by kind, vector shapes, one real payload, one search.
    Parents and manifests are payload-only points (no vectors), so raw `points_count` alone is misleading."""
    import math
    from qdrant_client import models as qm
    env, prof, store = _store(a.profile)
    c, name = store.client, store.collection
    print(f"qdrant      {env.qdrant.url or env.qdrant.path}")
    print(f"collections {[x.name for x in c.get_collections().collections]}")
    if not c.collection_exists(name):
        print(f"collection  {name}: NOT FOUND (run the ingestion first)")
        return 1
    info = c.get_collection(name)
    vecs = info.config.params.vectors
    print(f"collection  {name}  status={info.status}  points_count={info.points_count}")
    print("vectors     " + ", ".join(f"{k}: size={v.size} distance={v.distance.name}" for k, v in vecs.items())
          + (f", sparse: {list((info.config.params.sparse_vectors or {}).keys())}"))
    by = {k: c.count(name, count_filter=qm.Filter(must=[qm.FieldCondition(key="kind", match=qm.MatchValue(value=k))]), exact=True).count
          for k in ("child", "parent", "doc")}
    print(f"by kind     child={by['child']} (searchable, have vectors)  parent={by['parent']} (context text, no vectors)  doc={by['doc']} (one manifest per file)")
    docs, _ = c.scroll(name, limit=100, with_payload=True, scroll_filter=qm.Filter(must=[qm.FieldCondition(key="kind", match=qm.MatchValue(value="doc"))]))
    for d in sorted(docs, key=lambda p: p.payload.get("source_key", "")):
        pl = d.payload
        print(f"  file      {pl.get('source_key', '?'):32s} tenant={pl.get('tenant')} pages={pl.get('page_count')} chunks={pl.get('chunk_count')} v{pl.get('version')}")
    pts, _ = c.scroll(name, limit=1, with_payload=True, with_vectors=True, scroll_filter=qm.Filter(must=[qm.FieldCondition(key="kind", match=qm.MatchValue(value="child"))]))
    if pts:
        p0, v = pts[0], pts[0].vector
        dense = v.get("dense")
        sp = v.get("sparse")
        pl = p0.payload
        print(f"sample point {p0.id}")
        if dense:
            print(f"  dense     dim={len(dense)} first6={[round(x, 3) for x in dense[:6]]} L2norm={math.sqrt(sum(x * x for x in dense)):.3f}")
        if sp is not None:
            print(f"  sparse    non-zero terms={len(sp.indices)} e.g. {list(zip(sp.indices[:3], [round(x, 2) for x in sp.values[:3]]))}")
        print(f"  payload   tenant={pl.get('tenant')} acl={pl.get('acl')} doc={pl.get('doc_name')} page={pl.get('page_start')} section={pl.get('section')!r}")
        print(f"  text      {pl.get('text', '')[:110]!r}")
    if a.query:
        from .store.base import Principal
        print(f"search      {a.query!r} as tenant={a.tenant}")
        for h in store.search(a.query, Principal("verify", a.tenant), mode=prof.retrieval.mode, top_k=3):
            lo, hi = h.pages
            print(f"  {h.score:.3f}  {h.doc}  p.{lo}-{hi}  {h.text[:90]!r}")
    return 0


def _write_synthetic(root: Path, n: int) -> None:
    """N DISTINCT small documents (different words, one unique fact each) so a load test spreads over many documents instead of
    hammering the same ones. Deterministic: the same N always produces the same files, so re-running is idempotent."""
    import random
    vocab = ("pressure injection reservoir seal permeability monitoring pump valve sensor casing flow rate depth pipeline "
             "inspection calibration shutdown maintenance compliance sampling anomaly baseline threshold telemetry cable "
             "well pad integrity cement perforation tracer plume saline aquifer fault caprock porosity logging").split()
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        rnd = random.Random(i)
        paras = [" ".join(rnd.choice(vocab) for _ in range(45)).capitalize() + "." for _ in range(8)]
        (root / f"load_{i:04d}.md").write_text(f"# Site report {i:04d}\n\n" + "\n\n".join(paras[:4]) + f"\n\n## Equipment\n\nThe pump serial for unit {i:04d} is PX-{i * 7 + 1000}.\n\n"
                                              + "\n\n".join(paras[4:]) + "\n")


def cmd_enqueue(a):
    """Put ingestion jobs on the Celery queue (Redis) and, with --wait, show how they were processed. Workers see the path
    as THEY mount it, so give the container path (e.g. /data/raw)."""
    import collections
    import os
    from .ingestion.parsers import SUPPORTED
    from .ingestion.tasks import ingest_file
    if a.synthetic:
        _write_synthetic(Path(a.path), a.synthetic)
    files = sorted(str(p) for p in Path(a.path).rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED) if Path(a.path).is_dir() else [a.path]
    jobs = [f for _ in range(a.copies) for f in files]
    t0 = time.time()
    res = [ingest_file.delay(f, tenant=a.tenant, force=a.force or a.copies > 1) for f in jobs]
    print(f"enqueued {len(res)} jobs ({len(files)} files x {a.copies}); broker={os.environ.get('ORCA_REDIS_URL', 'redis://localhost:6379/0').split('@')[-1]}")
    if not a.wait:
        return 0
    status, workers = collections.Counter(), collections.Counter()
    for r in res:
        try:
            d = r.get(timeout=a.timeout)
            status[d["status"]] += 1
            workers[d.get("worker", "?")] += 1
            if d["status"] == "failed":
                print(f"  FAILED {d['source_key']}: {d['error'][:100]}")
        except Exception as e:
            status["no-result"] += 1
            print(f"  no result for {r.id[:8]}: {type(e).__name__}")
    print(f"done in {time.time() - t0:.1f}s  status={dict(status)}  by worker={dict(workers)}")
    return 1 if status.get("failed") or status.get("no-result") else 0


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
    p = sub.add_parser("variants"); p.add_argument("dataset"); p.add_argument("spec"); p.add_argument("--only", default=""); p.add_argument("--out", default=""); p.set_defaults(fn=cmd_variants)
    p = sub.add_parser("sweep"); p.add_argument("dataset"); p.add_argument("--experiment", default="all", choices=["models", "size", "overlap", "strategy", "retrieval", "all"])
    p.add_argument("--models"); p.add_argument("--only"); p.add_argument("--out"); p.set_defaults(fn=cmd_sweep)
    p = sub.add_parser("compare"); p.add_argument("dir"); p.add_argument("--skip", default=""); p.add_argument("--ref", default="", help="compare every variant to this one (paired bootstrap) instead of adjacent rungs"); p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("verify"); p.add_argument("--query", default=""); p.set_defaults(fn=cmd_verify)
    p = sub.add_parser("enqueue"); p.add_argument("path"); p.add_argument("--copies", type=int, default=1); p.add_argument("--force", action="store_true")
    p.add_argument("--synthetic", type=int, default=0, help="first write N distinct small documents into PATH"); p.add_argument("--wait", action="store_true"); p.add_argument("--timeout", type=int, default=600); p.set_defaults(fn=cmd_enqueue)
    p = sub.add_parser("stats"); p.set_defaults(fn=cmd_stats)
    a = ap.parse_args(argv)
    logging.basicConfig(level=get_env().log_level, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "qdrant_client", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
