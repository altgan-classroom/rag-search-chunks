from __future__ import annotations

import json
from dataclasses import dataclass

from ..ingestion.chunkers import count_tokens
from ..store.qdrant_store import Hit


@dataclass
class Source:
    n: int  # 1-based citation number
    doc_name: str
    page: int | None
    section: str
    text: str
    hit: Hit


def build_sources(hits: list[Hit], max_tokens: int) -> list[Source]:
    """Number sources in rank order within a token budget. Drops exact-duplicate texts."""
    out, used, seen = [], 0, set()
    for h in hits:
        key = h.text.strip()[:200]
        if key in seen:
            continue
        t = count_tokens(h.text)
        if out and used + t > max_tokens:
            break
        seen.add(key)
        pages = h.payload.get("matched_pages") or h.pages
        out.append(Source(len(out) + 1, h.payload.get("doc_name", "?"), pages[0] if pages else None,
                          h.payload.get("section", ""), h.text, h))
        used += t
    return out


def render_sources(sources: list[Source]) -> str:
    return "\n".join(
        f"[{s.n}] ({s.doc_name}{f', p.{s.page}' if s.page else ''}{f' | {s.section}' if s.section else ''})\n{s.text}"
        for s in sources)


def sources_event_payload(sources: list[Source]) -> str:
    """The `context` chunk body the existing UI parses: {"0": {doc_name, page_num, chunk}, ...} (key = n-1)."""
    return json.dumps({str(s.n - 1): {"doc_name": s.doc_name, "page_num": s.page, "chunk": s.text[:400],
                                      "section": s.section} for s in sources})
