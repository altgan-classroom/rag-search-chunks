"""One-factor-at-a-time sweeps that answer 'which embedding model / chunk size / overlap / strategy should I use FOR
MY DOMAIN'. Retrieval is deliberately simple (hybrid, no rerank, no parent expansion) so the factor under test is the
only thing that changes. Reports quality AND cost: index time, chunk count, vector bytes, query latency."""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from qdrant_client import QdrantClient

from ..config.settings import Env, Profile
from ..llm.gateway import LLMGateway
from ..store.qdrant_store import QdrantStore
from .ablate import ensure_indexed, with_overrides
from .harness import evaluate
from .schema import CorpusManifest, GoldSet

# Embedding candidates that run on CPU with ONNX (no torch/keys). Larger models are listed in docs/CHOOSING.md.
MODELS = ["BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "snowflake/snowflake-arctic-embed-s",
          "sentence-transformers/all-MiniLM-L6-v2", "nomic-ai/nomic-embed-text-v1.5", "minishlab/potion-retrieval-32M"]

BASE = {"chunking": {"strategy": "structure", "size": 350, "overlap": 50, "parent_child": False, "keep_tables_whole": True},
        "embedding": {"dense_model": "BAAI/bge-small-en-v1.5", "sparse_model": "Qdrant/bm25"},
        "retrieval": {"mode": "hybrid", "top_k": 40, "rerank": {"enabled": False}, "parent_expand": False, "final_k": 10,
                      "mmr_lambda": None, "rewrite_with_history": False, "router": False, "decompose": False},
        "generation": {"verify_grounding": False, "corrective": False, "abstain_below": None}}


@dataclass
class Row:
    label: str
    factor: str
    model: str
    strategy: str
    size: int
    overlap: int
    chunks: int
    index_s: float
    docs_per_s: float
    dim: int
    vec_mb: float
    p50_ms: float
    p95_ms: float
    metrics: dict[str, float] = field(default_factory=dict)


def experiments(which: str, models: list[str] | None = None) -> list[tuple[str, str, dict]]:
    exps: list[tuple[str, str, dict]] = []
    if which in ("models", "all"):
        for m in models or MODELS:
            exps.append((f"model={m.split('/')[-1]}", "embedding", {"embedding": {"dense_model": m}}))
        exps.append(("model=bge-small+instruction", "embedding", {"embedding": {"query_prefix": "Represent this sentence for searching relevant passages: "}}))
    if which in ("size", "all"):
        for sz in (128, 256, 350, 512, 1024):
            exps.append((f"size={sz}", "chunk_size", {"chunking": {"size": sz, "overlap": max(0, int(sz * 0.15))}}))
    if which in ("overlap", "all"):
        for pct in (0, 15, 30):
            exps.append((f"overlap={pct}%", "overlap", {"chunking": {"overlap": int(350 * pct / 100)}}))
    if which in ("strategy", "all"):
        exps.append(("strategy=fixed_words(2048ch)", "strategy", {"chunking": {"strategy": "fixed_words", "size": 2048, "overlap": 64}}))
        exps.append(("strategy=structure", "strategy", {}))
        exps.append(("strategy=semantic", "strategy", {"chunking": {"strategy": "semantic", "size": 350, "overlap": 0}}))
    if which in ("retrieval", "all"):
        for mode in ("dense", "sparse", "hybrid"):
            exps.append((f"mode={mode}", "retrieval_mode", {"retrieval": {"mode": mode}}))
    seen, out = set(), []
    for e in exps:  # 'strategy=structure' == BASE == 'size=350' == 'overlap=15%': evaluate each distinct recipe once
        if e[0] not in seen:
            seen.add(e[0])
            out.append(e)
    return out


async def run_sweep(env: Env, client: QdrantClient, dataset: Path, which: str, models: list[str] | None = None,
                    only: set[str] | None = None) -> list[Row]:
    gold = GoldSet.load(dataset / "gold.json")
    manifest = CorpusManifest.load(dataset / "manifest.json")
    llm = LLMGateway(mock=True)
    base = Profile.model_validate({**BASE, "name": "base"})
    rows: list[Row] = []
    for label, factor, ov in experiments(which, models):
        if only and label not in only:
            continue
        prof = with_overrides(base, label, ov)
        store = QdrantStore(env, prof, client=client)
        existed = store.client.collection_exists(store.collection)
        t0 = time.perf_counter()
        try:
            idx = ensure_indexed(store, prof, manifest, llm)
        except Exception as e:  # a model that cannot load must not abort the whole sweep
            print(f"[{label}] SKIPPED: {type(e).__name__}: {e}", flush=True)
            continue
        index_s = idx["index_s"] if not existed else float("nan")
        run = await evaluate(prof, gold, store, llm)
        lat = sorted(r.metrics["retrieval_ms"] for r in run.results if "retrieval_ms" in r.metrics)
        s = run.summary()["overall"]
        n = store.stats()["points"]
        c = prof.chunking
        rows.append(Row(label, factor, prof.embedding.dense_model, c.strategy, c.size, c.overlap, idx["chunks"], index_s,
                        (len(manifest.files) / index_s) if index_s == index_s and index_s else float("nan"),
                        store.dense.dim, n * store.dense.dim * 4 / 1e6, statistics.median(lat),
                        lat[int(0.95 * (len(lat) - 1))],
                        {k: s[k]["mean"] for k in ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@10", "answer_in_context") if k in s}))
        r = rows[-1]
        print(f"[{label}] chunks={r.chunks} idx={r.index_s}s recall@5={r.metrics.get('recall@5')} mrr={r.metrics.get('mrr')} "
              f"aic={r.metrics.get('answer_in_context')} p50={r.p50_ms:.0f}ms ({time.perf_counter() - t0:.0f}s)", flush=True)
    return rows


def to_markdown(rows: list[Row]) -> str:
    h = "| experiment | dim | chunks | index s | vec MB | p50 ms | recall@1 | recall@5 | recall@10 | MRR | nDCG@10 | answer-in-ctx |"
    out = [h, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        m = r.metrics
        f = lambda k: f"{m.get(k, float('nan')):.3f}"
        out.append(f"| {r.label} | {r.dim} | {r.chunks} | {r.index_s:.0f} | {r.vec_mb:.1f} | {r.p50_ms:.0f} | {f('recall@1')} | {f('recall@5')} | "
                   f"{f('recall@10')} | {f('mrr')} | {f('ndcg@10')} | {f('answer_in_context')} |")
    return "\n".join(out)
