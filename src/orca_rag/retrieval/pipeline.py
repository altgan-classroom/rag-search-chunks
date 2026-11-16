"""The retrieval pipeline. One `Retriever.retrieve()` call runs the recipe selected by the Profile and returns the
final context plus a step-by-step trace (what each technique did, how long it took). The trace is what the eval
harness and the UI "how was this answered" panel use."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from qdrant_client import models as qm

from ..config.settings import Profile
from ..llm.gateway import LLMGateway
from ..store.qdrant_store import Hit, Principal, QdrantStore
from . import query as q
from .fusion import mmr, rrf
from .rerank import get_reranker


@dataclass
class Step:
    name: str
    ms: float
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Retrieval:
    query: str
    effective_queries: list[str]
    route: str
    hits: list[Hit]  # final context chunks (children, or parents when expanded)
    children: list[Hit]  # the matched child chunks behind them (for page/citation precision)
    best_score: float | None  # top reranker score (None when no reranker)
    trace: list[Step] = field(default_factory=list)


class Retriever:
    def __init__(self, store: QdrantStore, profile: Profile, llm: LLMGateway | None = None):
        self.store, self.p = store, profile
        self.llm = llm or LLMGateway(mock=False)
        self.reranker = get_reranker(profile.retrieval.rerank.model if profile.retrieval.rerank.enabled else None)

    async def retrieve(self, question: str, principal: Principal, history: list[tuple[str, str]] | None = None,
                       filters: list[qm.Condition] | None = None) -> Retrieval:
        cfg, trace = self.p.retrieval, []
        fast = self.p.generation.fast_model

        def step(name, t0, **info):
            trace.append(Step(name, round((time.perf_counter() - t0) * 1000, 1), info))

        # 1. history-aware rewrite -------------------------------------------------------------
        query = question
        if cfg.rewrite_with_history and history:
            t0 = time.perf_counter()
            query = await q.rewrite_with_history(self.llm, fast, question, history)
            step("rewrite", t0, before=question, after=query)

        # 2. route ---------------------------------------------------------------------------
        route = "retrieve"
        if cfg.router:
            t0 = time.perf_counter()
            route = await q.route(self.llm, fast, query)
            step("route", t0, route=route)
            if route == "direct":
                return Retrieval(question, [query], route, [], [], None, trace)

        # 3. expand into queries (decompose / HyDE) ------------------------------------------
        queries = [query]
        if cfg.decompose and route == "decompose":
            t0 = time.perf_counter()
            subs = await q.decompose(self.llm, fast, query)
            queries += subs
            step("decompose", t0, subqueries=subs)
        hyde_vec = None
        if cfg.hyde:
            t0 = time.perf_counter()
            doc = await q.hyde(self.llm, fast, query)
            if doc:
                hyde_vec = self.store.dense.embed_query(doc)
            step("hyde", t0, used=bool(doc))

        # 4. first-stage retrieval (hybrid dense+sparse inside Qdrant), per query ------------
        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()

        def run(qs: str, vec=None):
            return self.store.search(qs, principal, mode=cfg.mode, top_k=cfg.top_k, fusion=cfg.fusion, extra=filters,
                                     query_dense=vec)

        lists = await asyncio.gather(*[loop.run_in_executor(None, run, qs, hyde_vec if i == 0 else None)
                                       for i, qs in enumerate(queries)])
        lists = list(lists)
        if hyde_vec is not None:  # keep the plain query too: HyDE can drift
            lists.append(await loop.run_in_executor(None, run, query, None))
        fused = rrf(lists) if len(lists) > 1 else lists[0]
        step("search", t0, mode=cfg.mode, queries=len(queries), candidates=len(fused))

        # 5. rerank --------------------------------------------------------------------------
        best = None
        cand = fused[: max(cfg.rerank.candidates, cfg.final_k)]
        if cfg.rerank.enabled and cand:
            t0 = time.perf_counter()
            first_stage = list(cand)
            cand = await loop.run_in_executor(None, self.reranker.rerank, query, cand, max(cfg.rerank.top_n, cfg.final_k))
            best = cand[0].score if cand else None  # raw reranker score (kept for abstention) before any blending
            if cfg.rerank.blend > 0 and cand:
                # the reranker must not be able to veto first-stage evidence outright: fuse both orders
                b = cfg.rerank.blend
                scores = {h.id: h.score for h in cand}
                fused_b = rrf([cand, [h for h in first_stage if h.id in scores]], weights=[1 - b, b])
                cand = [Hit(h.id, scores[h.id], h.text, h.payload) for h in fused_b]
            if cfg.rerank.min_score is not None:
                cand = [h for h in cand if h.score >= cfg.rerank.min_score]
            step("rerank", t0, model=self.reranker.name, kept=len(cand), best=best)

        # 6. diversity (MMR) -----------------------------------------------------------------
        if cfg.mmr_lambda is not None and len(cand) > cfg.final_k:
            t0 = time.perf_counter()
            vecs = np.array(self.store.dense.embed_docs([h.text for h in cand]), dtype=np.float32)
            qv = np.array(self.store.dense.embed_query(query), dtype=np.float32)
            rel = np.array([h.score for h in cand], dtype=np.float32) if cfg.rerank.enabled else None
            cand = mmr(cand, vecs, qv, cfg.mmr_lambda, cfg.final_k, relevance=rel)
            step("mmr", t0, lam=cfg.mmr_lambda, kept=len(cand))
        children = cand[: cfg.final_k]

        # 7. parent expansion: small chunks matched, larger parents returned (deduped, order kept) --
        context = children
        if cfg.parent_expand:
            t0 = time.perf_counter()
            pids = [h.payload.get("parent_id") for h in children if h.payload.get("parent_id")]
            parents = await loop.run_in_executor(None, self.store.fetch_parents, list(dict.fromkeys(pids)), principal)
            seen, context = set(), []
            for h in children:
                pid = h.payload.get("parent_id")
                if pid and pid in parents:
                    if pid in seen:
                        continue
                    seen.add(pid)
                    ph = parents[pid]
                    context.append(Hit(ph.id, h.score, ph.text, {**ph.payload, "matched_pages": h.pages}))
                else:
                    context.append(h)
            step("parent_expand", t0, children=len(children), contexts=len(context))
        return Retrieval(question, queries, route, context, children, best, trace)
