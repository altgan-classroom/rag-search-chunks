"""Corrective retrieval as a LlamaIndex Workflow (CRAG-style).

    start -> retrieve -> grade --sufficient--> done
                          \\--insufficient & rounds left--> (better query) -> retrieve ...

The grader is an LLM judge on a *cheap* model. Without an LLM the heuristic is: sufficient iff we retrieved anything
and (when a reranker exists) the best score clears `abstain_below`. Retries merge result lists with RRF so the first
round's good chunks are never lost."""
from __future__ import annotations

from dataclasses import dataclass

from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, Workflow, step

from ..config.settings import Profile
from ..generation.context import build_sources, render_sources
from ..generation.prompts import GRADE
from ..llm.gateway import LLMGateway
from ..retrieval.fusion import rrf
from ..retrieval.pipeline import Retrieval, Retriever, Step
from ..store.base import Principal

import time


class RetrieveEvent(Event):
    query: str
    round: int


class GradeEvent(Event):
    round: int


@dataclass
class _State:
    result: Retrieval | None = None
    rounds: int = 0


class CorrectiveRetrieval(Workflow):
    def __init__(self, retriever: Retriever, profile: Profile, llm: LLMGateway, **kw):
        super().__init__(timeout=120, **kw)
        self.r, self.p, self.llm = retriever, profile, llm

    @step
    async def start(self, ctx: Context, ev: StartEvent) -> RetrieveEvent:
        await ctx.store.set("state", _State())
        await ctx.store.set("args", {"question": ev.question, "principal": ev.principal, "history": ev.history})
        return RetrieveEvent(query=ev.question, round=0)

    @step
    async def retrieve(self, ctx: Context, ev: RetrieveEvent) -> GradeEvent:
        a = await ctx.store.get("args")
        st: _State = await ctx.store.get("state")
        res = await self.r.retrieve(ev.query, a["principal"], a["history"] if ev.round == 0 else None)
        if st.result is None:
            st.result = res
        else:  # merge with earlier rounds
            merged = rrf([st.result.hits, res.hits])[: self.p.retrieval.final_k]
            st.result = Retrieval(a["question"], st.result.effective_queries + res.effective_queries, res.route, merged,
                                  st.result.children + res.children, max(filter(None, [st.result.best_score, res.best_score]), default=None),
                                  st.result.trace + res.trace)
        st.rounds = ev.round
        return GradeEvent(round=ev.round)

    @step
    async def grade(self, ctx: Context, ev: GradeEvent) -> RetrieveEvent | StopEvent:
        a = await ctx.store.get("args")
        st: _State = await ctx.store.get("state")
        res, g = st.result, self.p.generation
        t0 = time.perf_counter()
        sufficient, better = bool(res.hits), ""
        if res.route == "direct":
            sufficient = True
        elif res.hits and self.llm.available:
            src = render_sources(build_sources(res.hits, 1800))
            out = await self.llm.json(GRADE.format(question=a["question"], sources=src), model=g.fast_model, task="grade",
                                      default=None, max_tokens=160)
            if isinstance(out, dict) and "sufficient" in out:
                sufficient, better = bool(out["sufficient"]), str(out.get("better_query") or "").strip()
        elif res.hits and res.best_score is not None and g.abstain_below is not None:
            sufficient = res.best_score >= g.abstain_below
        res.trace.append(Step("grade", round((time.perf_counter() - t0) * 1000, 1),
                              {"round": ev.round, "sufficient": sufficient, "better_query": better}))
        if sufficient or ev.round >= g.max_corrective_rounds or (not better and self.llm.available and res.hits):
            return StopEvent(result=res)
        return RetrieveEvent(query=better or a["question"], round=ev.round + 1)


async def corrective_retrieve(retriever: Retriever, profile: Profile, llm: LLMGateway, question: str,
                              principal: Principal, history=None) -> Retrieval:
    wf = CorrectiveRetrieval(retriever, profile, llm)
    return await wf.run(question=question, principal=principal, history=history)
