from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any

DOC_NS = uuid.UUID("0b7d6a1e-9c1d-4c0e-8d57-2f4a8d1d6e01")
PARSER_VERSION = "2.2"  # 2.2: multi-format parsers; slides/sheets are hard page boundaries


def doc_id_for(sha256: str) -> str:
    """Content-addressed: same bytes => same id => idempotent ingest."""
    return str(uuid.uuid5(DOC_NS, sha256))


def sha256_file(path: str, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(bufsize):
            h.update(b)
    return h.hexdigest()


@dataclass
class Page:
    number: int  # 1-based
    markdown: str
    quality: float = 1.0  # share of tokens that look like real words (0..1)
    ocr: bool = False
    skipped: bool = False  # excluded from indexing (blank / garbage)


@dataclass
class ParsedDocument:
    name: str
    source_key: str
    sha256: str
    pages: list[Page]
    meta: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def doc_id(self) -> str:
        return doc_id_for(self.sha256)


@dataclass
class Chunk:
    ordinal: int
    text: str  # what the user/LLM sees
    page_start: int | None
    page_end: int | None
    section: str = ""  # heading path "A > B"
    is_table: bool = False
    parent_ordinal: int | None = None
    context: str = ""  # LLM-written situating context (contextual retrieval), embedded but not shown
    tokens: int = 0

    def embed_text(self, doc_name: str, add_title: bool = True) -> str:
        head = []
        if add_title:
            head.append(f"Document: {doc_name}")
        if self.section:
            head.append(f"Section: {self.section}")
        if self.context:
            head.append(self.context)
        return ("\n".join(head) + "\n\n" if head else "") + self.text

    @property
    def content_hash(self) -> str:
        return hashlib.sha1(self.text.encode()).hexdigest()[:16]
