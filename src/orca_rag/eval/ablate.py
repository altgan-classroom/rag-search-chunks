"""Ablations: apply recipe overrides on top of a base profile, (re)index when the recipe changes what is stored, evaluate.

Query-time-only changes (rerank, MMR, decompose, ...) reuse the same collection. Ingestion-time changes (chunking,
embeddings) get their own collection, derived from the recipe signature, so variants coexist and re-runs are free."""
from __future__ import annotations

import copy
import time
from pathlib import Path

from qdrant_client import QdrantClient

from ..config.settings import Env, Profile
from ..ingestion.pipeline import Ingestor
from ..llm.gateway import LLMGateway
from ..store.base import VectorStore
from ..store.qdrant_store import QdrantStore
from .harness import EvalRun, evaluate
from .schema import CorpusManifest, GoldSet


def deep_update(d: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


def with_overrides(base: Profile, name: str, overrides: dict) -> Profile:
    data = deep_update(copy.deepcopy(base.model_dump()), overrides)
    data["name"] = name
    return Profile.model_validate(data)


def ensure_indexed(store: VectorStore, profile: Profile, manifest: CorpusManifest, llm: LLMGateway | None = None) -> dict:
    ing = Ingestor(store, profile, llm)
    t0 = time.perf_counter()
    stats = {"ingested": 0, "unchanged": 0, "failed": 0, "chunks": 0}
    for cf in manifest.files:
        r = ing.ingest_file(str(Path(manifest.root) / cf.file), tenant=cf.tenant, acl=cf.acl)
        key = r.status if r.status in stats else "ingested"
        stats[key] += 1
        stats["chunks"] += r.chunks
        if r.status == "failed":
            print(f"  ! ingest failed {cf.file}: {r.error}")
    stats["index_s"] = round(time.perf_counter() - t0, 1)
    return stats


async def run_variants(env: Env, client: QdrantClient, variants: list[tuple[str, Profile]], manifest: CorpusManifest,
                       gold: GoldSet, llm: LLMGateway, *, generate: bool = False, out_root: str | Path | None = None,
                       categories: set[str] | None = None) -> list[EvalRun]:
    runs = []
    for label, prof in variants:
        store = QdrantStore(env, prof, client=client)
        idx = ensure_indexed(store, prof, manifest, llm)
        run = await evaluate(prof, gold, store, llm, generate=generate, categories=categories)
        run.profile = label
        run.meta = {"collection": store.collection, "index": idx, "points": store.stats()["points"]}
        if out_root:
            run.save(Path(out_root) / label)
        s = run.summary()["overall"]
        print(f"[{label}] n={len(run.results)} recall@5={s.get('recall@5', {}).get('mean')} mrr={s.get('mrr', {}).get('mean')} "
              f"ndcg@10={s.get('ndcg@10', {}).get('mean')} ctx_tokens={s.get('context_tokens', {}).get('mean')} "
              f"index={idx}", flush=True)
        runs.append(run)
    return runs
