"""LLM-assisted query understanding. Every function is *safe*: with no LLM or on bad output it returns the
original query / a retrieve route, so retrieval never gets worse than the baseline because a helper failed."""
from __future__ import annotations

import re

from ..llm.gateway import LLMGateway

GREETING = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|good (morning|afternoon|evening)|bye)\b[\s!.?]*$", re.I)
_COMPOUND = re.compile(r"\b(compare|versus|vs\.?|difference between|both|as well as|and also)\b|\?.*\?", re.I)

REWRITE = """Rewrite the user's latest question as a single standalone search query, resolving pronouns and
references using the conversation. Keep names, numbers and identifiers exactly. If it is already standalone,
return it unchanged. Output only the query.

Conversation:
{history}

Question: {question}"""

ROUTE = """Classify the question for a document-QA system. Reply with JSON only: {{"route": "<r>"}} where <r> is
"direct" (greeting/chit-chat/thanks, needs no documents), "decompose" (needs several separate lookups or a
comparison/aggregation), or "retrieve" (a single lookup).

Question: {question}"""

DECOMPOSE = """Break the question into at most {n} independent, self-contained search queries that together
answer it. Keep identifiers (well names, dates, units) verbatim. Reply with a JSON array of strings only.

Question: {question}"""

HYDE = """Write a short, factual passage (3-4 sentences) that would plausibly appear in a technical report and
answer this question. Invent no specific numbers; use placeholders like "X psi" if unsure. Output only the passage.

Question: {question}"""


def heuristic_route(q: str) -> str:
    if GREETING.match(q):
        return "direct"
    return "decompose" if _COMPOUND.search(q) else "retrieve"


async def rewrite_with_history(llm: LLMGateway, model: str, question: str, history: list[tuple[str, str]]) -> str:
    if not history or not llm.available:
        return question
    h = "\n".join(f"{r}: {t[:400]}" for r, t in history[-6:])
    try:
        out = (await llm.complete(REWRITE.format(history=h, question=question), model=model, task="rewrite",
                                  max_tokens=120)).strip().strip('"')
    except Exception:
        return question
    return out if 3 <= len(out) <= 400 else question


async def route(llm: LLMGateway, model: str, question: str, use_llm: bool = True) -> str:
    base = heuristic_route(question)
    if base == "direct" or not (use_llm and llm.available):
        return base
    out = await llm.json(ROUTE.format(question=question), model=model, task="route", default=None, max_tokens=30)
    r = out.get("route") if isinstance(out, dict) else None
    return r if r in ("direct", "decompose", "retrieve") else base


async def decompose(llm: LLMGateway, model: str, question: str, n: int = 3) -> list[str]:
    if not llm.available:
        return []
    out = await llm.json(DECOMPOSE.format(question=question, n=n), model=model, task="decompose", default=[], max_tokens=250)
    if not isinstance(out, list):
        return []
    qs = [q.strip() for q in out if isinstance(q, str) and 3 <= len(q.strip()) <= 300]
    return qs[:n] if len(qs) > 1 else []  # a single sub-query adds nothing over the original


async def hyde(llm: LLMGateway, model: str, question: str) -> str:
    if not llm.available:
        return ""
    try:
        return (await llm.complete(HYDE.format(question=question), model=model, task="hyde", max_tokens=200,
                                   temperature=0.3)).strip()
    except Exception:
        return ""
