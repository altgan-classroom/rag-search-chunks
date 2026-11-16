"""Answer orchestration. Emits the event stream the UI consumes:

  {type: context, text: <json sources>}   sources, sent BEFORE tokens so citation chips can resolve while streaming
  {type: tokens,  text: <delta>}          answer deltas
  {type: answer,  text: <full>}           authoritative final text (UI replaces streamed tokens)
  {type: related, text: <question>}       follow-ups (one per event)
  {type: trace|grounding, text: <json>}   observability, ignored by older UIs
  {type: error,   text: <message>}        never raised mid-stream
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

from ..agentic.corrective import corrective_retrieve
from ..config.settings import Profile
from ..llm.gateway import LLMGateway
from ..retrieval.pipeline import Retrieval, Retriever
from ..store.qdrant_store import Principal
from .context import build_sources, render_sources, sources_event_payload
from .prompts import ABSTAIN, ANSWER, RELATED, SYSTEM
from .verify import check_grounding, verify_llm

logger = logging.getLogger(__name__)
GREETING_REPLY = "Hello! Ask me anything about the ingested documents and I'll answer with citations."


class Answerer:
    def __init__(self, retriever: Retriever, profile: Profile, llm: LLMGateway):
        self.r, self.p, self.llm = retriever, profile, llm

    async def run(self, question: str, principal: Principal, history: list[tuple[str, str]] | None = None) -> AsyncIterator[dict]:
        rid = uuid.uuid4().hex[:12]
        ev = lambda t, text: {"response_id": rid, "type": t, "text": text}
        g, t_start = self.p.generation, time.perf_counter()
        try:
            if g.corrective:
                ret: Retrieval = await corrective_retrieve(self.r, self.p, self.llm, question, principal, history)
            else:
                ret = await self.r.retrieve(question, principal, history)

            if ret.route == "direct":
                yield ev("greeting", GREETING_REPLY)
                yield ev("answer", GREETING_REPLY)
                return

            sources = build_sources(ret.hits, g.max_context_tokens)
            yield ev("context", sources_event_payload(sources) if sources else "{}")

            low = (g.abstain_below is not None and ret.best_score is not None and ret.best_score < g.abstain_below)
            if not sources or low:
                yield ev("tokens", ABSTAIN)
                yield ev("answer", ABSTAIN)
                yield ev("trace", json.dumps(self._trace(ret, t_start, abstained=True, reason="low_relevance" if low else "no_results")))
                return

            prompt = ANSWER.format(abstain=ABSTAIN, sources=render_sources(sources), question=question)
            parts: list[str] = []
            async for d in self.llm.stream(prompt, model=g.model, task="generate", system=SYSTEM, temperature=g.temperature,
                                           max_tokens=g.max_tokens, timeout_s=g.timeout_s):
                parts.append(d)
                yield ev("tokens", d)
            answer = "".join(parts).strip()
            yield ev("answer", answer)

            grounding = None
            if g.verify_grounding:
                gr = check_grounding(answer, sources)
                extra = await verify_llm(self.llm, g.fast_model, answer, sources)
                grounding = {"score": round(gr.score, 3), "ok": gr.ok, "unsupported": gr.unsupported[:5],
                             "invalid_citations": gr.invalid_citations, "uncited": len(gr.uncited),
                             "llm_unsupported": extra}
                yield ev("grounding", json.dumps(grounding))

            if self.llm.available and answer and ABSTAIN not in answer:
                rel = await self.llm.json(RELATED.format(question=question, answer=answer[:1500]), model=g.fast_model,
                                          task="related", default=[], max_tokens=200)
                for q in (rel if isinstance(rel, list) else [])[:3]:
                    if isinstance(q, str) and q.strip():
                        yield ev("related", q.strip())
            yield ev("trace", json.dumps(self._trace(ret, t_start, grounding=grounding)))
        except Exception as e:  # noqa: BLE001 - after the first byte HTTP can't change status: surface in-band, never raise
            logger.exception("answer failed rid=%s", rid)  # full detail in logs; client gets a generic message
            yield ev("error", f"Sorry, something went wrong ({type(e).__name__}).")

    def _trace(self, ret: Retrieval, t0: float, **extra) -> dict:
        return {"profile": self.p.name, "route": ret.route, "queries": ret.effective_queries,
                "best_score": ret.best_score, "total_ms": round((time.perf_counter() - t0) * 1000),
                "steps": [{"name": s.name, "ms": s.ms, **s.info} for s in ret.trace], **extra}
