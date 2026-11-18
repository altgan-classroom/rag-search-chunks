"""LlamaIndex adapters: use Orca's retrieval and ingestion *inside* LlamaIndex pipelines and query engines.

    OrcaRetriever        BaseRetriever  -> works with RetrieverQueryEngine, CitationQueryEngine, agents, Workflows
    PageReader           BaseReader     -> one Document per page (layout-aware, OCR, quality-gated)
    StructureChunker     TransformComponent -> structure-aware / parent-child chunks inside an IngestionPipeline
    to_llamaindex_llm    LLMGateway -> LiteLLM (or MockLLM offline) for synthesizers

They are intentionally thin: behaviour (ACL filtering, hybrid search, rerank, MMR, parent expansion, tracing) lives in the
core modules and is the same whichever interface you call it through.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
from pathlib import Path
from typing import Any

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.llms import LLM, MockLLM
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document, NodeWithScore, QueryBundle, TextNode, TransformComponent

from ..config.settings import ChunkingCfg
from ..ingestion.chunkers import chunk_document
from ..ingestion.models import Page, ParsedDocument
from ..ingestion.parsers import parse_file
from ..llm.gateway import LLMGateway
from ..retrieval.pipeline import Retriever
from ..store.base import Hit, Principal


def hit_to_node(h: Hit) -> NodeWithScore:
    meta = {k: v for k, v in h.payload.items() if k not in ("text", "acl", "tenant") and v not in (None, "")}
    meta["page_label"] = (h.payload.get("matched_pages") or h.pages or [None])[0]
    return NodeWithScore(node=TextNode(id_=h.id, text=h.text, metadata=meta), score=h.score)


class OrcaRetriever(BaseRetriever):
    """Principal-scoped retriever. The principal is fixed at construction time (built server-side from auth),
    so no query string can widen what the retriever is allowed to see."""

    def __init__(self, retriever: Retriever, principal: Principal, history: list[tuple[str, str]] | None = None):
        super().__init__()
        self._r, self._p, self._h = retriever, principal, history

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        ret = await self._r.retrieve(query_bundle.query_str, self._p, self._h)
        return [hit_to_node(h) for h in ret.hits]

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        coro = self._aretrieve(query_bundle)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        with concurrent.futures.ThreadPoolExecutor(1) as ex:  # called from inside a running loop
            return ex.submit(asyncio.run, coro).result()


class PageReader(BaseReader):
    """Reads a PDF/markdown/text file into one Document per *page* with quality metadata."""

    def load_data(self, file: str | Path, **kw: Any) -> list[Document]:
        doc = parse_file(str(file), **kw)
        return [Document(text=p.markdown, metadata={"file_name": doc.source_key, "doc_name": doc.name, "page_number": p.number,
                                                    "skipped": p.skipped, "sha256": doc.sha256}, id_=f"{doc.sha256[:16]}-p{p.number}")
                for p in doc.pages]


class StructureChunker(TransformComponent):
    """IngestionPipeline step: page Documents (from PageReader) -> structure-aware chunks with page ranges and heading path."""

    cfg: ChunkingCfg = ChunkingCfg()

    def __call__(self, nodes: list, **kwargs: Any) -> list:
        by_file: dict[str, list] = {}
        for n in nodes:
            by_file.setdefault(n.metadata.get("file_name", "doc"), []).append(n)
        out: list[TextNode] = []
        for fname, ns in by_file.items():
            pages = [Page(number=int(n.metadata.get("page_number", i + 1)), markdown=n.get_content(),
                          skipped=bool(n.metadata.get("skipped", False))) for i, n in enumerate(ns)]
            doc = ParsedDocument(name=Path(fname).stem, source_key=fname, sha256=ns[0].metadata.get("sha256", fname), pages=pages)
            children, parents = chunk_document(doc, self.cfg)
            ptext = {p.ordinal: p.text for p in parents}
            for c in children:
                out.append(TextNode(text=c.text, metadata={
                    "file_name": fname, "doc_name": doc.name, "page_start": c.page_start, "page_end": c.page_end,
                    "section": c.section, "is_table": c.is_table, "parent_text": ptext.get(c.parent_ordinal, "")},
                    excluded_embed_metadata_keys=["parent_text", "is_table"], excluded_llm_metadata_keys=["parent_text"]))
        return out


def to_llamaindex_llm(gw: LLMGateway, model: str) -> LLM:
    """LiteLLM-backed LlamaIndex LLM for a configured model id; MockLLM when the gateway is in offline mock mode."""
    if gw.mock:
        return MockLLM(max_tokens=256)
    from llama_index.llms.litellm import LiteLLM
    return LiteLLM(model=model, num_retries=gw.num_retries)
