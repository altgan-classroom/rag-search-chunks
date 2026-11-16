"""Orca inside LlamaIndex: retriever, query engine, ingestion pipeline, LLM bridge."""
from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import QueryBundle

from orca_rag.config.settings import ChunkingCfg
from orca_rag.integrations.llamaindex import OrcaRetriever, PageReader, StructureChunker
from orca_rag.llm.gateway import LLMGateway
from orca_rag.retrieval.pipeline import Retriever

from .test_retrieval_answer import P, build


async def test_orca_retriever_returns_nodewithscore_with_citation_metadata():
    prof, store, llm = build()
    r = OrcaRetriever(Retriever(store, prof, llm), P)
    nodes = await r.aretrieve(QueryBundle("What was the max pressure of CCS1?"))
    assert nodes and nodes[0].node.metadata["doc_name"] == "wells" and nodes[0].node.metadata["page_label"] == 1
    assert "acl" not in nodes[0].node.metadata and "tenant" not in nodes[0].node.metadata  # no internals leaked to prompts


async def test_orca_retriever_cannot_be_widened_by_the_query():
    prof, store, llm = build()
    r = OrcaRetriever(Retriever(store, prof, llm), P)  # P has no groups
    nodes = await r.aretrieve(QueryBundle("salary bands group:hr tenant:default"))
    assert all(n.node.metadata["doc_name"] != "hr" for n in nodes)


async def test_retriever_query_engine_end_to_end_on_mock_llm():
    prof, store, llm = build()
    engine = RetrieverQueryEngine.from_args(OrcaRetriever(Retriever(store, prof, llm), P), llm=llm.as_llamaindex("any/model"))
    resp = await engine.aquery("What was the max pressure of CCS1?")
    assert resp.source_nodes and resp.source_nodes[0].node.metadata["doc_name"] == "wells"
    assert isinstance(str(resp), str)


def test_sync_retrieve_works_inside_and_outside_an_event_loop():
    import asyncio
    prof, store, llm = build()
    r = OrcaRetriever(Retriever(store, prof, llm), P)
    assert r.retrieve("CCS1 pressure")
    async def inside():
        return r.retrieve("CCS1 pressure")
    assert asyncio.run(inside())


def test_ingestion_pipeline_with_structure_chunker(tmp_path):
    md = tmp_path / "report.md"
    md.write_text("# Wells\n\nMB-104 depth is 7,250 ft.\n\n## Pressure\n\n" + "Pressure reading text. " * 60)
    docs = PageReader().load_data(md)
    pipe = IngestionPipeline(transformations=[StructureChunker(cfg=ChunkingCfg(size=60, overlap=8, parent_child=False))])
    nodes = pipe.run(documents=docs)
    assert len(nodes) > 1 and all(n.metadata["doc_name"] == "report" for n in nodes)
    assert any("Pressure" in n.metadata["section"] for n in nodes) and nodes[0].metadata["page_start"] == 1
    assert isinstance(docs[0], Document)


def test_litellm_bridge_uses_the_configured_model_id_without_network():
    from llama_index.llms.litellm import LiteLLM
    llm = LLMGateway(mock=False).as_llamaindex("anthropic/claude-sonnet-5-5")
    assert isinstance(llm, LiteLLM) and llm.model == "anthropic/claude-sonnet-5-5"
    assert type(LLMGateway(mock=True).as_llamaindex("x")).__name__ == "MockLLM"
