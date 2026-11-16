"""Idempotent, versioned ingestion.

  sha256(file) -> doc_id (content addressed). Same bytes + same recipe => skipped.
  Changed bytes for the same source_key => old chunks deleted, new ones written, version += 1.
  Chunk ids are uuid5(doc_id, kind:ordinal) so a retried task rewrites the same points (no duplicates).
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..config.settings import Profile
from ..llm.gateway import LLMGateway
from ..store.qdrant_store import PUBLIC, QdrantStore, point_id
from .chunkers import chunk_document
from .enrich import contextualize, extract_doc_meta
from .models import PARSER_VERSION, ParsedDocument
from .parsers import parse_file

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    source_key: str
    status: str  # ingested | unchanged | replaced | failed
    doc_id: str = ""
    pages: int = 0
    pages_skipped: int = 0
    chunks: int = 0
    parents: int = 0
    contextualized: int = 0
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)
    error: str = ""

    def dict(self):
        return asdict(self)


def recipe_signature(profile: Profile) -> str:
    return hashlib.sha1(profile.collection_name().encode() + PARSER_VERSION.encode()).hexdigest()[:12]


class Ingestor:
    def __init__(self, store: QdrantStore, profile: Profile, llm: LLMGateway | None = None):
        self.store, self.profile, self.llm = store, profile, llm

    def ingest_file(self, path: str, *, tenant: str = "default", acl: list[str] | None = None, force: bool = False,
                    parsed: ParsedDocument | None = None, **parse_kw) -> IngestReport:
        t0 = time.perf_counter()
        source_key = Path(path).name
        rep = IngestReport(source_key=source_key, status="failed")
        try:
            doc = parsed or parse_file(path, **parse_kw)
            source_key = rep.source_key = doc.source_key  # the parsed document's key is authoritative
            rep.doc_id, rep.pages = doc.doc_id, len(doc.pages)
            rep.pages_skipped = sum(p.skipped for p in doc.pages)
            rep.warnings = list(doc.warnings)
            sig = recipe_signature(self.profile)
            old = self.store.get_manifest(source_key, tenant)
            if old and old.get("sha256") == doc.sha256 and old.get("recipe") == sig and not force:
                rep.status = "unchanged"
                rep.chunks = old.get("chunk_count", 0)
                return rep
            version = 1
            if old:
                version = int(old.get("version", 1)) + 1
                self.store.delete_source(source_key, tenant)  # drop old version's points (children, parents, manifest)
                rep.status = "replaced"
            else:
                rep.status = "ingested"
            self._write(doc, tenant, acl or [PUBLIC], version, sig, rep)
        except Exception as e:  # report, don't crash a batch; callers decide on retry policy
            logger.exception("ingest failed for %s", path)
            rep.status, rep.error = "failed", f"{type(e).__name__}: {e}"
        finally:
            rep.seconds = round(time.perf_counter() - t0, 2)
        return rep

    # ------------------------------------------------------------------
    def _write(self, doc: ParsedDocument, tenant: str, acl: list[str], version: int, sig: str, rep: IngestReport) -> None:
        cfg = self.profile.chunking
        children, parents = chunk_document(doc, cfg, embedder=self.store.dense)
        if not children:
            raise ValueError("no indexable text (all pages empty or low quality)")
        if cfg.contextual:
            if self.llm and self.llm.available:
                ptxt = {p.ordinal: p.text for p in parents}
                rep.contextualized = asyncio.run(contextualize(children, ptxt, doc, self.llm, self.profile.generation.fast_model))
            else:
                rep.warnings.append("contextual retrieval requested but no LLM available: skipped")
        meta = extract_doc_meta(doc)
        did = doc.doc_id
        now = datetime.now(timezone.utc).isoformat()
        base = {"doc_id": did, "doc_name": doc.name, "source_key": doc.source_key, "tenant": tenant, "acl": acl,
                "version": version, **meta}
        pid = lambda o: f"{did}:p:{o}"
        rows = []
        for c in children:
            rows.append({"id": point_id(did, "child", c.ordinal), "text": c.text,
                         "embed_text": c.embed_text(doc.name, cfg.add_doc_title),
                         "payload": {**base, "kind": "child", "chunk_ord": c.ordinal, "page_start": c.page_start,
                                     "page_end": c.page_end, "section": c.section, "is_table": c.is_table,
                                     "tokens": c.tokens, "content_hash": c.content_hash, "context": c.context,
                                     "parent_id": pid(c.parent_ordinal) if c.parent_ordinal is not None else None}})
        self.store.upsert_children(rows)
        if parents:
            self.store.upsert_payload_only([{"id": point_id(did, "parent", p.ordinal), "text": p.text,
                                             "payload": {**base, "kind": "parent", "parent_id": pid(p.ordinal),
                                                         "page_start": p.page_start, "page_end": p.page_end,
                                                         "section": p.section}} for p in parents])
        self.store.upsert_payload_only([{"id": point_id(did, "doc", 0), "text": "", "payload": {
            **base, "kind": "doc", "sha256": doc.sha256, "recipe": sig, "parser_version": PARSER_VERSION,
            "chunk_count": len(children), "page_count": len(doc.pages), "ingested_at": now}}])
        rep.chunks, rep.parents = len(children), len(parents)

    def ingest_dir(self, root: str, *, pattern: str = "**/*", **kw) -> list[IngestReport]:
        from .parsers import SUPPORTED
        files = sorted(p for p in Path(root).glob(pattern) if p.is_file() and p.suffix.lower() in SUPPORTED)
        return [self.ingest_file(str(f), **kw) for f in files]
