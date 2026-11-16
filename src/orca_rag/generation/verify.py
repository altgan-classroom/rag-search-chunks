"""Grounding verification.

Deterministic (always on, keyless): are cited numbers valid, and is each sentence lexically supported by the sources
it cites? This is a *faithfulness proxy*: it catches fabricated citations and clearly unsupported sentences, not
subtle paraphrase errors. The optional LLM check (`verify_llm`) adds claim-level judgement when a key exists."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..llm.gateway import LLMGateway
from .context import Source, render_sources
from .prompts import ABSTAIN, VERIFY

_CITE = re.compile(r"\[(\d+)\]")
_SENT = re.compile(r"(?<=[.!?])\s+(?!\[\d+\])|\n+")  # never split before a trailing citation: "... depth. [1]"
_WORD = re.compile(r"[a-z0-9][a-z0-9\-\.]*[a-z0-9]|[a-z0-9]")
_STOP = set("the a an of to in on for and or is are was were be been by with as at from that this it its which than then "
            "into over per not no yes can could may might also has have had will would should".split())


def _content(s: str) -> set[str]:
    return {w for w in _WORD.findall(s.lower()) if w not in _STOP and len(w) > 1}


@dataclass
class Grounding:
    score: float  # share of factual sentences that are supported (1.0 = all)
    unsupported: list[str] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    uncited: list[str] = field(default_factory=list)
    checked: int = 0

    @property
    def ok(self) -> bool:
        return not self.invalid_citations and self.score >= 0.7


def check_grounding(answer: str, sources: list[Source], min_overlap: float = 0.5) -> Grounding:
    if ABSTAIN.lower() in answer.lower():
        return Grounding(1.0)
    by_n = {s.n: s for s in sources}
    invalid = sorted({int(n) for n in _CITE.findall(answer) if int(n) not in by_n})
    sents = [s.strip() for s in _SENT.split(answer) if len(s.strip()) > 25 and not s.strip().startswith(("|", "#"))]
    unsupported, uncited, ok = [], [], 0
    for s in sents:
        cited = [int(n) for n in _CITE.findall(s) if int(n) in by_n]
        words = _content(_CITE.sub("", s))
        if not words:
            continue
        if not cited:
            uncited.append(s)
            pool = set().union(*(_content(x.text) for x in sources)) if sources else set()
        else:
            pool = set().union(*(_content(by_n[n].text) for n in cited))
        cov = len(words & pool) / len(words)
        if cov >= min_overlap and cited:
            ok += 1
        elif cov >= min_overlap and not cited:
            ok += 0.5  # supported by *some* source but uncited: half credit, still reported
        else:
            unsupported.append(s)
    n = len(sents)
    return Grounding(score=(ok / n) if n else 1.0, unsupported=unsupported, invalid_citations=invalid,
                     uncited=uncited, checked=n)


async def verify_llm(llm: LLMGateway, model: str, answer: str, sources: list[Source]) -> list[str] | None:
    """Claim-level check. Returns unsupported claims, or None if the judge was unavailable/unparseable."""
    if not llm.available or ABSTAIN.lower() in answer.lower():
        return None
    out = await llm.json(VERIFY.format(sources=render_sources(sources)[:12000], answer=answer), model=model,
                         task="verify", default=None, max_tokens=300)
    if isinstance(out, dict) and isinstance(out.get("unsupported_claims"), list):
        return [str(c) for c in out["unsupported_claims"]]
    return None
