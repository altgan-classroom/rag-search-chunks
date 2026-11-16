"""Chunk enrichment: cheap regex metadata (always) and LLM contextual retrieval (opt-in)."""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

from ..llm.gateway import LLMGateway
from .models import Chunk, ParsedDocument

_YEAR = re.compile(r"(?<!\d)(19[89]\d|20[0-3]\d)(?!\d)")
_DATE8 = re.compile(r"(?<!\d)((?:19|20)\d{2})(0[1-9]|1[0-2])([0-3]\d)(?!\d)")
_TYPES = [("well_summary", r"well[_ -]?summary"), ("completion_report", r"completion[_ -]?report"),
          ("falloff_test", r"falloff"), ("risk_report", r"risk"), ("fact_sheet", r"fact[_ -]?sheet|flyer|faq"),
          ("geomechanics", r"geomech"), ("integrity_report", r"integrity"), ("topical_report", r"topical"),
          ("terms", r"terms|licen[cs]e")]


def extract_doc_meta(doc: ParsedDocument) -> dict:
    """doc_year / doc_type from filename (+ first page fallback). Used for metadata filters & temporal routing."""
    name = Path(doc.source_key).stem
    year = None
    m = _DATE8.search(name)
    if m:
        year = int(m.group(1))
    else:
        ys = _YEAR.findall(name)
        year = int(ys[0]) if ys else None
    if year is None and doc.pages:
        ys = _YEAR.findall(doc.pages[0].markdown[:1500])
        year = int(ys[0]) if ys else None
    dtype = next((t for t, rx in _TYPES if re.search(rx, name, re.I)), "other")
    return {"doc_year": year, "doc_type": dtype}


CONTEXT_PROMPT = """<document_excerpt>
{window}
</document_excerpt>

Here is a chunk from that document:
<chunk>
{chunk}
</chunk>

Write ONE short sentence (max 40 words) that situates this chunk within the document so it can be found by search
(what document/section/subject it belongs to, key entities, dates, well names). Output only that sentence."""


async def contextualize(chunks: list[Chunk], parents: dict[int, str], doc: ParsedDocument, llm: LLMGateway, model: str,
                        concurrency: int = 8, window_chars: int = 6000) -> int:
    """Anthropic-style contextual retrieval. Returns number of chunks enriched; failures leave a chunk unchanged."""
    sem = asyncio.Semaphore(concurrency)
    head = f"Document: {doc.name}\n"

    async def one(c: Chunk) -> bool:
        window = head + parents.get(c.parent_ordinal if c.parent_ordinal is not None else -1, "")[:window_chars]
        if len(window) <= len(head):
            window = head + "\n".join(p.markdown for p in doc.pages[: max(1, (c.page_start or 1))][-2:])[:window_chars]
        async with sem:
            try:
                out = await llm.complete(CONTEXT_PROMPT.format(window=window, chunk=c.text[:2000]), model=model,
                                         task="contextualize", max_tokens=100, temperature=0.0)
            except Exception:
                return False
        out = out.strip().replace("\n", " ")
        if out:
            c.context = out[:300]
            return True
        return False

    return sum(await asyncio.gather(*(one(c) for c in chunks)))
